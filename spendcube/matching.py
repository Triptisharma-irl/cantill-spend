"""Count once: match a secondary system (e.g. AP invoice tool) to a primary one (e.g. ERP receipts),
and collapse records exported more than once inside a source."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .textnorm import compact, series_map


def _rel(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1.0)


def _text_sim(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.array([fuzz.ratio(x, y) / 100.0 if x and y else 0.0 for x, y in zip(a, b)])


def cross_match(df: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    """Returns DataFrame[rid (secondary), matched_rid (primary), match_score, match_how]."""
    P = df[df.src == mcfg["primary"]].copy()
    S = df[df.src == mcfg["secondary"]].copy()
    tol_txt = mcfg.get("amount_tolerance", 0.05)          # when text agrees
    tol_strict = mcfg.get("amount_tolerance_strict", 0.005)  # when text does not agree
    abs_tol = mcfg.get("amount_abs_tolerance", 0.05)
    for X in (P, S):
        X["ctext"] = series_map(X["description"], compact)
    # a line without PO inherits the PO of its invoice when the invoice's other lines agree on one PO
    if mcfg.get("infer_po_from_invoice", True) and "invoice_key" in S:
        inv_po = S[(S.po != "") & (S.invoice_key != "")].groupby("invoice_key")["po"].agg(lambda x: x.iloc[0] if x.nunique() == 1 else "")
        blank = (S.po == "") & S.invoice_key.isin(inv_po.index)
        S.loc[blank, "po"] = S.loc[blank, "invoice_key"].map(inv_po).fillna("")
    pcols = ["rid", "po", "po_line", "amount_net_doc", "quantity", "ctext", "supplier_cluster"]
    scols = ["rid", "po", "po_line", "amount_net_doc", "quantity", "ctext", "supplier_cluster"]

    def score(c: pd.DataFrame) -> pd.DataFrame:
        ad = np.abs(c.amount_net_doc_s - c.amount_net_doc_p)
        rd = _rel(c.amount_net_doc_s, c.amount_net_doc_p)
        ts = _text_sim(c.ctext_s, c.ctext_p)
        qd = _rel(c.quantity_s.fillna(0), c.quantity_p.fillna(0))
        same_sup = (c.supplier_cluster_s == c.supplier_cluster_p).values
        line_ok = ((c.po_line_s == c.po_line_p) | (c.po_line_s == "")).values
        amount_ok = (ad <= abs_tol) | (rd <= tol_strict) | ((rd <= tol_txt) & (ts >= 0.75))
        c = c.assign(rd=rd, ts=ts, qd=qd, ok=amount_ok & line_ok,
                     score=1.0 - np.minimum(rd, 1) * 5 + 0.5 * ts + 0.2 * same_sup + 0.1 * (qd < 0.01) + 0.2 * line_ok)
        return c

    used_p, used_s, rows = set(), set(), []

    def assign(c: pd.DataFrame, how: str):
        c = c[c.ok].sort_values("score", ascending=False)
        for r in c.itertuples(index=False):
            if r.rid_s in used_s or r.rid_p in used_p:
                continue
            used_s.add(r.rid_s)
            used_p.add(r.rid_p)
            rows.append((r.rid_s, r.rid_p, round(float(r.score), 3), how))

    # pass 1: same PO reference
    sp = S[S.po != ""][scols]
    pp = P[P.po != ""][pcols]
    c = sp.merge(pp, on="po", suffixes=("_s", "_p"))
    if len(c):
        assign(score(c), "po+amount")
    # pass 2: no PO match -> same supplier, amount within tolerance, near-identical text
    # only records without any PO reference: a PO that exists but is not in the primary system
    # means the purchase genuinely lives only in the secondary system
    rest = S[~S.rid.isin(used_s) & (S.po == "")]
    Pfree = P[~P.rid.isin(used_p)]
    tol2 = mcfg.get("nopo_amount_tolerance", 0.02)
    min_txt = mcfg.get("nopo_min_text_similarity", 0.85)
    cands = []
    for sup, g in rest.groupby("supplier_cluster"):
        pg = Pfree[Pfree.supplier_cluster == sup]
        if pg.empty:
            continue
        pg = pg.sort_values("amount_net_doc")
        pa = pg.amount_net_doc.values
        for r in g.itertuples(index=False):
            a = r.amount_net_doc
            if not np.isfinite(a):
                continue
            lo, hi = sorted([a * (1 - tol2), a * (1 + tol2)])
            i0, i1 = np.searchsorted(pa, lo - abs_tol), np.searchsorted(pa, hi + abs_tol, side="right")
            if i1 > i0:
                sub = pg.iloc[i0:i1]
                cands.append(pd.DataFrame({
                    "rid_s": r.rid, "rid_p": sub.rid.values, "amount_net_doc_s": a, "amount_net_doc_p": sub.amount_net_doc.values,
                    "quantity_s": r.quantity, "quantity_p": sub.quantity.values, "ctext_s": r.ctext, "ctext_p": sub.ctext.values,
                    "supplier_cluster_s": sup, "supplier_cluster_p": sup, "po_line_s": "", "po_line_p": sub.po_line.values}))
    if cands:
        c2 = score(pd.concat(cands, ignore_index=True))
        c2["ok"] = (c2.rd <= tol2) & (c2.ts >= min_txt)
        assign(c2, "supplier+amount+text")
    return pd.DataFrame(rows, columns=["rid", "matched_rid", "match_score", "match_how"])


def copy_groups(df: pd.DataFrame, src: str, keys: list[str]) -> pd.Series:
    """Group id for records of `src` that are the same document exported more than once."""
    S = df[df.src == src]
    k = pd.DataFrame(index=S.index)
    for col in keys:
        v = S[col]
        k[col] = v.round(2).astype(str) if v.dtype.kind == "f" else v.astype(str)
    valid = (k != "").all(axis=1) & (k != "nan").all(axis=1)
    gid = pd.Series("", index=df.index, dtype=object)
    gid[S.index[valid]] = src + "|" + k[valid].agg("|".join, axis=1)
    return gid
