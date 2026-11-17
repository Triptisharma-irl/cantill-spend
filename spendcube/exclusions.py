"""What counts as spend: status rules, non-procurement rules and rules learned from labels."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .textnorm import strip_accents

FIELD_ALIASES = {"supplier_type_desc": "supplier_type", "document_type_desc": "document_type"}


def status_code(status: pd.Series, rules: dict) -> tuple[pd.Series, pd.Series]:
    """Map raw status text -> '' (ok) / CANCELLED / NOT_APPROVED. Returns (code, note)."""
    st = rules["status"]
    vocab = []
    for code in ["CANCELLED", "NOT_APPROVED", "ok"]:
        for w in st.get(code, []):
            vocab.append((w.lower(), "" if code == "ok" else code))
    vocab.sort(key=lambda x: -len(x[0]))  # longest phrase wins ("partially paid" before "paid")

    def one(s: str):
        t = re.sub(r"[\s_\-]+", " ", strip_accents(s).lower()).strip()
        if not t:
            return ("" if st.get("blank_policy", "ok") == "ok" else st["blank_policy"]), "status blank"
        for w, code in vocab:
            if t == w:
                return code, f"status '{s}'"
        for w, code in vocab:
            if re.search(r"\b" + re.escape(w) + r"\b", t):
                return code, f"status '{s}'"
        pol = st.get("unknown_policy", "NOT_APPROVED")
        return ("" if pol == "ok" else pol), f"status '{s}' unrecognised -> {pol}"

    u = pd.unique(status.fillna(""))
    lut = {v: one(v) for v in u}
    m = status.fillna("").map(lut)
    return m.map(lambda x: x[0]), m.map(lambda x: x[1])


def rule_non_procurement(df: pd.DataFrame, rules: dict) -> tuple[pd.Series, pd.Series]:
    hit = pd.Series(False, index=df.index)
    why = pd.Series("", index=df.index, dtype=object)
    for r in rules["non_procurement"]:
        rx = re.compile(r["regex"], re.I)
        un = re.compile(r["unless"], re.I) if r.get("unless") else None
        for fld in r["fields"]:
            col = FIELD_ALIASES.get(fld, fld)
            if col not in df.columns:
                continue
            vals = df[col].fillna("").astype(str)
            u = pd.unique(vals)
            lut = {v: bool(v) and bool(rx.search(strip_accents(v))) and not (un and un.search(v)) for v in u}
            m = vals.map(lut) & ~hit
            if m.any():
                hit |= m
                why[m] = f"rule:{r['name']} on {col}"
    return hit, why


def learn_exclusion_rules(train: pd.DataFrame, min_support: int = 5, min_rate: float = 0.98,
                          fields=("gl_desc", "gl_code", "supplier_type")) -> list[dict]:
    """From labelled lines with an include flag, find field values that are (almost) always NON_PROCUREMENT."""
    if "label_exclusion" not in train.columns:
        return []
    lab = train[train["label_include"].notna()].copy()
    if lab.empty:
        return []
    lab["np"] = lab["label_exclusion"].eq("NON_PROCUREMENT")
    learned = []
    for f in fields:
        if f not in lab.columns:
            continue
        g = lab[lab[f] != ""].groupby(f)["np"].agg(["sum", "size"])
        g = g[(g["size"] >= min_support) & (g["sum"] / g["size"] >= min_rate)]
        # a value is only trusted if it is also never seen on an included labelled line
        g = g[g["sum"] == g["size"]] if min_rate >= 0.98 else g
        for v, row in g.iterrows():
            learned.append({"field": f, "value": v, "support": int(row["size"])})
    return learned


def apply_learned(df: pd.DataFrame, learned: list[dict]) -> tuple[pd.Series, pd.Series]:
    hit = pd.Series(False, index=df.index)
    why = pd.Series("", index=df.index, dtype=object)
    for r in learned:
        if r["field"] not in df.columns:
            continue
        m = df[r["field"]].eq(r["value"]) & ~hit
        hit |= m
        why[m] = f"learned:{r['field']}={r['value']} (n={r['support']})"
    return hit, why
