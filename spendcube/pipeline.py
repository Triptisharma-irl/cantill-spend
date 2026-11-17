"""End-to-end: ingest -> clean -> entities -> exclusions -> count once -> classify -> output."""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from .classify.core import ClassifierConfig, SpendClassifier
from .clean import clean, load_fx_table
from .entities import item_keys, public_key, supplier_keys
from .exclusions import apply_learned, learn_exclusion_rules, rule_non_procurement, status_code
from .ingest import Config, ingest, read_csv
from .matching import copy_groups, cross_match
from .taxonomy import load_taxonomy, resolve
from .textnorm import SupplierNormaliser, compact, series_map

log = logging.getLogger("spendcube")
OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"]


def _holdout_mask(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.Series:
    """Evaluation only: hide labels for a random `frac` of PO groups (records without PO are their own group)."""
    grp = df["po"].where(df["po"] != "", df["rid"])
    u = pd.Series(pd.unique(grp))
    rng = np.random.default_rng(seed)
    held = set(u[rng.random(len(u)) < frac])
    return grp.isin(held)


def _load_labels(cfg: Config, df: pd.DataFrame, train_dir: str, tax, holdout: float = 0.0) -> pd.DataFrame:
    """Attach label columns (label_leaf, label_include, label_exclusion, gold_supplier, gold_item) to df."""
    spec = cfg.get("labels")
    df = df.copy()
    for c in ["label_leaf", "label_include", "label_exclusion", "gold_supplier", "gold_item"]:
        df[c] = np.nan if c in ("label_leaf", "label_include") else ""
    if not spec:
        return df
    path = resolve(spec["file"], train_dir)
    if not path:
        log.warning("labels file %s not found -> running without labels", spec["file"])
        return df
    lab = read_csv(path)
    lab["rid"] = spec.get("id_prefix", "") + lab[spec["id"]].astype(str)
    lab = lab.drop_duplicates("rid").set_index("rid")
    m = df["rid"].isin(lab.index)
    L = lab.reindex(df["rid"][m])
    leaf_idx = {p: i for i, p in enumerate(tax.leaves)}
    if spec.get("levels"):
        def path_of(row):
            p = tuple(str(row[c]).strip() for c in spec["levels"])
            while p and p[-1] in ("", "nan"):
                p = p[:-1]
            return leaf_idx.get(p, np.nan)
        df.loc[m, "label_leaf"] = L.apply(path_of, axis=1).values
    if spec.get("include"):
        inc = pd.to_numeric(L[spec["include"]], errors="coerce").values
        df.loc[m, "label_include"] = inc
        df.loc[m, "label_exclusion"] = L[spec["exclusion_reason"]].fillna("").values if spec.get("exclusion_reason") else ""
    if spec.get("supplier_key"):
        df.loc[m, "gold_supplier"] = L[spec["supplier_key"]].fillna("").values
    if spec.get("item_key"):
        df.loc[m, "gold_item"] = L[spec["item_key"]].fillna("").values
    if holdout:
        h = _holdout_mask(df, holdout)
        df["holdout"] = h & m
        df.loc[h, ["label_leaf", "label_include"]] = np.nan
        df.loc[h, ["label_exclusion", "gold_supplier", "gold_item"]] = ""
        log.info("evaluation holdout: labels hidden for %d records", (h & m).sum())
    log.info("labels: %d records matched (%d with a category)", m.sum(), df["label_leaf"].notna().sum())
    return df


def _knowledge(train: pd.DataFrame, rules: dict) -> tuple[dict, dict]:
    norm = SupplierNormaliser(rules["legal_suffixes"])
    t = train[train["gold_supplier"] != ""]
    sup = {}
    if len(t):
        nk = series_map(t["supplier_name"], norm.key)
        sup["by_name"] = pd.DataFrame({"k": nk, "g": t["gold_supplier"]})[nk != ""].groupby("k")["g"].agg(lambda s: s.value_counts().index[0]).to_dict()
        sid = t["supplier_id"]
        sup["by_id"] = pd.DataFrame({"k": sid, "g": t["gold_supplier"]})[sid != ""].groupby("k")["g"].agg(lambda s: s.value_counts().index[0]).to_dict()
    t = train[train["gold_item"] != ""]
    item = {}
    if len(t):
        code = t["item_code"].str.upper().str.strip()
        item["by_code"] = pd.DataFrame({"k": code, "g": t["gold_item"]})[code != ""].groupby("k")["g"].agg(lambda s: s.value_counts().index[0]).to_dict()
        ct = series_map(t["description"], compact)
        item["by_text"] = pd.DataFrame({"k": ct, "g": t["gold_item"]})[ct != ""].groupby("k")["g"].agg(lambda s: s.value_counts().index[0]).to_dict()
    return sup, item


def prepare(cfg: Config, data_dir: str, fx) -> pd.DataFrame:
    df = ingest(cfg, data_dir)
    return clean(df, cfg, fx)


def run(cfg: Config, data_dir: str, out_path: str, train_dir: str | None = None, use_labels: bool = True,
        use_embeddings: bool = True, audit: bool = True, holdout: float = 0.0) -> pd.DataFrame:
    t0 = time.time()
    rules = cfg.rules
    train_dir = train_dir or data_dir
    tax_path = resolve(cfg["taxonomy"]["file"], data_dir, train_dir)
    tax = load_taxonomy(tax_path, cfg["taxonomy"])
    log.info("tree: %d leaves, depth %d (%s)", tax.n, tax.depth, tax_path)
    fx = load_fx_table(cfg, data_dir, train_dir)
    df = prepare(cfg, data_dir, fx)
    log.info("ingested %d records from %s in %.1fs", len(df), data_dir, time.time() - t0)

    # ---------------------------------------------------------------- training corpus (labels)
    has_labels = use_labels and cfg.get("labels") and resolve(cfg["labels"]["file"], train_dir)
    if has_labels:
        same = os.path.abspath(train_dir) == os.path.abspath(data_dir)
        train = df if same else prepare(cfg, train_dir, fx)
        train = _load_labels(cfg, train, train_dir, tax, holdout)
    else:
        train = None
    sup_known, item_known = _knowledge(train, rules) if train is not None else ({}, {})

    # ---------------------------------------------------------------- entities
    ecfg = cfg.get("entities", {})
    df["supplier_cluster"] = supplier_keys(df, rules, None, ecfg.get("supplier", {}))
    df["supplier_key"] = supplier_keys(df, rules, sup_known, ecfg.get("supplier", {})) if sup_known else df["supplier_cluster"]
    df["item_key"] = item_keys(df, rules, item_known or None, ecfg.get("item", {}))

    # ---------------------------------------------------------------- what counts as spend
    df["excl"] = ""
    df["excl_note"] = ""
    df["excl_conf"] = 0.0
    st, st_note = status_code(df["status"], rules)
    m = st != ""
    df.loc[m, "excl"] = st[m]
    df.loc[m, "excl_note"] = st_note[m]
    df.loc[m, "excl_conf"] = 0.97
    np_hit, np_why = rule_non_procurement(df, rules)
    learned = learn_exclusion_rules(train) if train is not None else []
    if learned:
        log.info("learned %d non-procurement rules from labels", len(learned))
    l_hit, l_why = apply_learned(df, learned)
    hit = (np_hit | l_hit) & (df["excl"] == "")
    df.loc[hit, "excl"] = "NON_PROCUREMENT"
    df.loc[hit, "excl_note"] = np.where(np_hit[hit], np_why[hit], l_why[hit])
    df.loc[hit, "excl_conf"] = 0.9

    # ---------------------------------------------------------------- count once
    df["dup_of"] = ""
    df["match_how"] = ""
    comb = cfg.get("combine")
    if comb:
        mt = cross_match(df, comb)
        log.info("cross-system matches: %d (%s)", len(mt), mt["match_how"].value_counts().to_dict() if len(mt) else {})
        mi = df.set_index("rid")
        prim = mi.loc[mt["matched_rid"]]
        pos = df.index[df["rid"].isin(mt["rid"])]
        mt = mt.set_index("rid").loc[df.loc[pos, "rid"]]
        df.loc[pos, "dup_of"] = mt["matched_rid"].values
        df.loc[pos, "match_how"] = mt["match_how"].values + ":" + mt["match_score"].astype(str).values
        # the matched primary record is the golden version of this purchase
        pk = mi.loc[mt["matched_rid"].values]
        df.loc[pos, "supplier_key"] = pk["supplier_key"].values
        df.loc[pos, "item_key"] = pk["item_key"].values
    for src, keys in (cfg.get("dedupe_within_source") or {}).items():
        gid = copy_groups(df, src, keys)
        g = df[gid != ""].assign(gid=gid[gid != ""])
        for _, grp in g.groupby("gid"):
            if len(grp) < 2:
                continue
            matched = grp[grp["dup_of"] != ""]
            if len(matched):
                anchor = matched["dup_of"].iloc[0]
                rest = grp[grp["dup_of"] == ""]
                df.loc[rest.index, "dup_of"] = anchor
                df.loc[rest.index, "match_how"] = "copy-of-matched-record"
                keep_keys = df.loc[matched.index[0], ["supplier_key", "item_key"]]
                df.loc[rest.index, "supplier_key"] = keep_keys["supplier_key"]
                df.loc[rest.index, "item_key"] = keep_keys["item_key"]
                continue
            ok = grp[grp["excl"] == ""]
            keeper = ok.index[0] if len(ok) else grp.index[0]
            rest = grp.index.drop(keeper)
            df.loc[rest, "dup_of"] = df.loc[keeper, "rid"]
            df.loc[rest, "match_how"] = "exported-twice"
            df.loc[rest, "supplier_key"] = df.loc[keeper, "supplier_key"]
            df.loc[rest, "item_key"] = df.loc[keeper, "item_key"]
    d = (df["dup_of"] != "") & (df["excl"] == "")
    df.loc[d, "excl"] = "DUPLICATE"
    df.loc[d, "excl_note"] = "duplicate of " + df.loc[d, "dup_of"] + " (" + df.loc[d, "match_how"] + ")"
    df.loc[d, "excl_conf"] = 0.9

    # ---------------------------------------------------------------- classify included lines
    inc = df["excl"] == ""
    df["spend"] = np.where(inc, df["spend_all"], 0.0)
    df["spend_abs"] = df["spend_all"].abs()
    clf = SpendClassifier(tax, rules["abbreviations"], ClassifierConfig(
        use_embeddings=use_embeddings, **cfg.get("classifier", {})),
        generic_descriptions=rules.get("generic_descriptions", ()))
    tr = None
    if train is not None and train["label_leaf"].notna().any():
        tr = train[train["label_leaf"].notna()].copy()
        df["holdout"] = df["rid"].map(train.set_index("rid")["holdout"]).fillna(False) if "holdout" in train else False
        if "label_include" in tr and tr["label_include"].notna().any():
            tr = tr[tr["label_include"].fillna(1) == 1]
        tr["leaf"] = tr["label_leaf"].astype(int)
        tr["spend_abs"] = tr["spend_all"].abs()
        keys = df.set_index("rid")[["supplier_key", "item_key"]]
        if tr["rid"].isin(keys.index).all():   # training records are the records being processed
            tr["supplier_key"] = keys.loc[tr["rid"], "supplier_key"].values
            tr["item_key"] = keys.loc[tr["rid"], "item_key"].values
        else:
            tr["supplier_key"] = supplier_keys(tr, rules, sup_known, ecfg.get("supplier", {}))
            tr["item_key"] = item_keys(tr, rules, item_known or None, ecfg.get("item", {}))
    lines = df.loc[inc]
    res = clf.classify(lines, tr)
    for c in ["l1", "l2", "l3", "l4"]:
        df[c] = ""
        df.loc[inc, c] = res[c].values
    df["confidence"] = df["excl_conf"]
    df.loc[inc, "confidence"] = res["confidence"].values

    # ---------------------------------------------------------------- output
    out = pd.DataFrame({
        "record_id": df["rid"], "include": inc.astype(int), "exclusion_reason": df["excl"],
        "supplier_key": public_key(df["supplier_key"], "S-"), "item_key": public_key(df["item_key"], "I-"),
        "spend": df["spend"].round(2),
        "l1": df["l1"], "l2": df["l2"], "l3": df["l3"], "l4": df["l4"], "confidence": df["confidence"].round(4)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    if audit:
        aud = df[["rid", "src", "excl", "excl_note", "dup_of", "match_how", "amount_raw", "basis", "tax_rate", "currency",
                  "fx_rate", "fx_source", "fx_date", "fx_date_flag", "date", "date_flag", "amount_flag", "spend_all",
                  "supplier_cluster", "supplier_key", "item_key"] + (["holdout"] if "holdout" in df else [])]
        aud.to_csv(out_path.replace(".csv", "_audit.csv"), index=False)
    log.info("wrote %s: %d records, %d included, spend %.2f %s, %.1fs", out_path, len(out), inc.sum(),
             out["spend"].sum(), cfg["reporting_currency"], time.time() - t0)
    log.info("exclusions: %s", df["excl"].replace("", "INCLUDED").value_counts().to_dict())
    return df
