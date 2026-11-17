"""Supplier and item de-duplication (union-find over normalised keys, strict fuzzy merges,
and - when the client supplies golden keys - a learned alias table)."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from .textnorm import SupplierNormaliser, clean_text, compact, is_generic, series_map


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller id is the root
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def _h(s: str, n=8) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:n]


def _fuzzy_pairs(keys: list[str], min_len: int, max_edits: int, min_ratio: float, block: int = 2):
    """Yield pairs of near-identical keys. Blocking on the first `block` chars keeps it fast."""
    by = defaultdict(list)
    for k in keys:
        if len(k) >= min_len:
            by[k[:block]].append(k)
    # also block on the last chars so a typo in the first letters can still be caught
    by2 = defaultdict(list)
    for k in keys:
        if len(k) >= min_len:
            by2[k[-block:]].append(k)
    seen = set()
    for groups in (by, by2):
        for ks in groups.values():
            if len(ks) < 2:
                continue
            sc = process.cdist(ks, ks, scorer=fuzz.ratio, score_cutoff=min_ratio)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    if sc[i, j] and (ks[i], ks[j]) not in seen:
                        if Levenshtein.distance(ks[i], ks[j]) <= max_edits:
                            seen.add((ks[i], ks[j]))
                            yield ks[i], ks[j]


def public_key(keys: pd.Series, prefix: str) -> pd.Series:
    """Opaque, deterministic output key: the same internal key always gives the same public key, so
    grouping is unchanged, but no client code, supplier name or ID appears in the output.
    'I-UNSPEC-...' keys keep their tag."""
    def one(k: str) -> str:
        if not k:
            return ""
        tag = prefix + "UNSPEC-" if k.startswith(prefix + "UNSPEC-") else prefix
        return tag + _h("key:" + k, 12)
    return series_map(keys.fillna("").astype(str), one)


# --------------------------------------------------------------------------- suppliers
def supplier_keys(df: pd.DataFrame, rules: dict, known: dict | None = None, cfg: dict | None = None) -> pd.Series:
    """known: {'by_name': {name_key: golden}, 'by_id': {supplier_id: golden}} learned from labels."""
    cfg = cfg or {}
    norm = SupplierNormaliser(rules["legal_suffixes"])
    name_key = series_map(df["supplier_name"], norm.key)
    sid = df["supplier_id"].fillna("")
    uf = UnionFind()
    nodes = pd.Series("", index=df.index, dtype=object)
    has_name = name_key != ""
    nodes[has_name] = "N:" + name_key[has_name]
    nodes[~has_name & (sid != "")] = "I:" + sid[~has_name & (sid != "")]
    nodes[nodes == ""] = "UNKNOWN"
    for n in pd.unique(nodes):
        uf.find(n)
    if cfg.get("link_by_supplier_id", True):
        both = has_name & (sid != "")
        for a, b in set(zip(nodes[both], "I:" + sid[both])):
            uf.union(a, b)
    if cfg.get("fuzzy", True):
        uniq = sorted(set(name_key[has_name]))
        for a, b in _fuzzy_pairs(uniq, min_len=cfg.get("fuzzy_min_len", 8), max_edits=cfg.get("fuzzy_max_edits", 1),
                                 min_ratio=cfg.get("fuzzy_min_ratio", 90)):
            uf.union("N:" + a, "N:" + b)
    root = nodes.map(uf.find)
    # golden aliases (client's own supplier master) take precedence where known
    if known:
        gold = pd.Series("", index=df.index, dtype=object)
        if known.get("by_name"):
            gold = name_key.map(known["by_name"]).fillna("")
        if known.get("by_id"):
            gid = sid.map(known["by_id"]).fillna("")
            gold = gold.where(gold != "", gid)
        # unknown members inherit the majority golden key of their cluster
        cl = pd.DataFrame({"root": root, "gold": gold})
        maj = cl[cl.gold != ""].groupby("root")["gold"].agg(lambda s: s.value_counts().index[0])
        gold = gold.where(gold != "", root.map(maj).fillna(""))
        known_mask = gold != ""
    else:
        gold = pd.Series("", index=df.index, dtype=object)
        known_mask = gold != ""
    # opaque key per cluster (hash of the cluster root): no supplier names or IDs leak into outputs
    hashed = root.map(lambda r: "S-" + _h(r, 10))
    return gold.where(known_mask, hashed)


# --------------------------------------------------------------------------- items
def item_keys(df: pd.DataFrame, rules: dict, known: dict | None = None, cfg: dict | None = None) -> pd.Series:
    """One key per real item.

    * An item code identifies an item; codes whose dominant description is identical are merged
      (different codes, same thing).
    * Lines without a code are keyed on their cleaned description; near-identical descriptions
      (OCR noise, spacing) are merged, and a description equal to a code's dominant text joins it.
    * Generic/blank descriptions without a code cannot be grouped: they get a per-supplier
      'unspecified' key.
    known: {'by_code': {code: golden}, 'by_text': {compact_text: golden}} learned from labels.
    """
    cfg = cfg or {}
    generic = set(rules["generic_descriptions"])
    code = df["item_code"].fillna("").str.strip().str.upper()
    desc = df["description"].fillna("")
    gen = series_map(desc, lambda d: is_generic(d, generic))
    ctext = series_map(desc, compact)
    ctext = ctext.where(~gen, "")
    uf = UnionFind()
    node = pd.Series("", index=df.index, dtype=object)
    node[code != ""] = "C:" + code[code != ""]
    m = (code == "") & (ctext != "")
    node[m] = "T:" + ctext[m]
    # dominant text per code
    has = (code != "") & (ctext != "")
    dom = pd.DataFrame({"c": code[has], "t": ctext[has]}).groupby("c")["t"].agg(lambda s: s.value_counts().index[0])
    for c, t in dom.items():
        uf.union("C:" + c, "T:" + t)
    for n in pd.unique(node[node != ""]):
        uf.find(n)
    if cfg.get("fuzzy", True):
        texts = sorted(set(ctext[ctext != ""]))
        for a, b in _fuzzy_pairs(texts, min_len=cfg.get("fuzzy_min_len", 10), max_edits=cfg.get("fuzzy_max_edits", 2),
                                 min_ratio=cfg.get("fuzzy_min_ratio", 92), block=3):
            if _numbers(a) == _numbers(b):  # never merge 20MM with 22MM
                uf.union("T:" + a, "T:" + b)
    root = node.map(lambda n: uf.find(n) if n else "")
    out = pd.Series("", index=df.index, dtype=object)
    if known:
        g = pd.Series("", index=df.index, dtype=object)
        if known.get("by_code"):
            g = code.map(known["by_code"]).fillna("")
        if known.get("by_text"):
            g = g.where(g != "", ctext.map(known["by_text"]).fillna(""))
        cl = pd.DataFrame({"root": root, "g": g})
        maj = cl[(cl.g != "") & (cl.root != "")].groupby("root")["g"].agg(lambda s: s.value_counts().index[0])
        g = g.where(g != "", root.map(maj).fillna(""))
        out = g
    readable = root.map(lambda r: ("I-" + _h(r, 10)) if r else "")
    out = out.where(out != "", readable)
    unspecified = "I-UNSPEC-" + df["supplier_key"].astype(str) if "supplier_key" in df else "I-UNSPEC"
    out = out.where(out != "", unspecified)
    return out


def _numbers(s: str) -> tuple:
    import re
    return tuple(re.findall(r"\d+", s))
