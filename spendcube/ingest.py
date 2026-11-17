"""Config-driven ingestion: every source file is mapped onto one canonical line schema.

Canonical columns (all strings unless noted):
  rid, src, amount_raw(float), basis, tax_rate(float), currency, line_fx_rate(float),
  date_raw, fx_date_raw, supplier_name, supplier_id, description, item_code, quantity(float),
  po, po_line, invoice, status, gl_code, gl_desc, supplier_type, document_type, context
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .parse import parse_amount, parse_rate

ROOT = Path(__file__).resolve().parent.parent

TEXT_FIELDS = ["supplier_name", "supplier_id", "description", "item_code", "po", "po_line", "invoice", "status",
               "gl_code", "gl_desc", "supplier_type", "document_type", "context"]


@dataclass
class Config:
    raw: dict
    path: str
    rules: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, d=None):
        return self.raw.get(k, d)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = json.load(f)
    rules_path = raw.get("rules_file") or str(ROOT / "rules" / "default_rules.json")
    with open(rules_path) as f:
        rules = json.load(f)
    # company-level extensions of the generic rules
    rules["abbreviations"].update(raw.get("abbreviations", {}))
    for code, words in raw.get("status_map", {}).items():
        rules["status"].setdefault(code, [])
        rules["status"][code] = list(words) + rules["status"][code]
    rules["non_procurement"] = rules["non_procurement"] + raw.get("non_procurement_rules", [])
    rules["generic_descriptions"] += raw.get("generic_descriptions", [])
    return Config(raw=raw, path=path, rules=rules)


def read_csv(path: str, spec: dict | None = None) -> pd.DataFrame:
    spec = spec or {}
    kw = dict(dtype=str, keep_default_na=False, encoding=spec.get("encoding", "utf-8-sig"))
    if spec.get("sep"):
        kw["sep"] = spec["sep"]
    else:
        with open(path, "rb") as fh:
            head = fh.readline().decode("utf-8", "ignore")
        kw["sep"] = max([",", ";", "\t", "|"], key=head.count)
    try:
        df = pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        df = pd.read_csv(path, **kw)
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, spec, joiner=" | ") -> pd.Series:
    """spec: None | 'col' | ['c1','c2'] (joined non-empty) | {'value': const} | {'first': [cols]}."""
    n = len(df)
    if spec is None:
        return pd.Series([""] * n, index=df.index, dtype=object)
    if isinstance(spec, dict):
        if "value" in spec:
            return pd.Series([str(spec["value"])] * n, index=df.index, dtype=object)
        if "first" in spec:
            out = pd.Series([""] * n, index=df.index, dtype=object)
            for c in spec["first"]:
                v = df[c].astype(str).str.strip() if c in df else ""
                out = out.where(out != "", v)
            return out
        if "column" in spec:
            return _col(df, spec["column"], joiner)
    if isinstance(spec, str):
        if spec not in df.columns:
            raise KeyError(f"column '{spec}' not found; available: {list(df.columns)}")
        return df[spec].astype(str).str.strip()
    parts = [df[c].astype(str).str.strip() if c in df.columns else pd.Series("", index=df.index) for c in spec]
    out = parts[0].copy()
    for p in parts[1:]:
        both = (out != "") & (p != "") & (out.str.upper() != p.str.upper())
        out = np.where(both, out + joiner + p, np.where(out == "", p, out))
        out = pd.Series(out, index=df.index, dtype=object)
    return out


def ingest_source(src: dict, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, src["file"])
    df = read_csv(path, src.get("csv"))
    f = src.get("fields", {})
    out = pd.DataFrame(index=df.index)
    out["rid"] = src.get("record_prefix", "") + _col(df, src["id"])
    out["src"] = src["name"]
    amt = src["amount"]
    out["amount_raw"] = parse_amount(_col(df, amt["column"]), amt.get("decimal"))
    out["basis"] = amt.get("basis", "net")
    if amt.get("basis") == "gross":
        rate = parse_rate(_col(df, amt.get("tax_rate_column")), amt.get("default_tax_rate", 0.0)) \
            if amt.get("tax_rate_column") else pd.Series(amt.get("default_tax_rate", 0.0), index=df.index)
        out["tax_rate"] = rate.fillna(amt.get("default_tax_rate", 0.0))
    else:
        out["tax_rate"] = 0.0
    cur = src.get("currency", {"value": ""})
    out["currency"] = _col(df, cur).str.upper().str.strip()
    fx = src.get("fx", {})
    out["line_fx_rate"] = parse_amount(_col(df, fx.get("line_rate_column"))) if fx.get("line_rate_column") else np.nan
    out["date_raw"] = _col(df, {"first": f.get("date", [])} if isinstance(f.get("date"), list) else f.get("date"))
    fxd = fx.get("date_field")
    out["fx_date_raw"] = _col(df, fxd) if fxd else out["date_raw"]
    out["fx_date_fallback_raw"] = _col(df, fx.get("fallback_date_field")) if fx.get("fallback_date_field") else out["date_raw"]
    for k in TEXT_FIELDS:
        if k == "context":
            spec = f.get("context")
            if spec:
                cols = [c for c in spec if c in df.columns]
                out[k] = df[cols].astype(str).apply(lambda r: " | ".join(f"{c}={v.strip()}" for c, v in zip(cols, r) if v.strip()), axis=1) if cols else ""
            else:
                out[k] = ""
            continue
        out[k] = _col(df, f.get(k)) if f.get(k) is not None else ""
    out["quantity"] = parse_amount(_col(df, f.get("quantity"))) if f.get("quantity") else np.nan
    # optional reference normalisation (PO numbers typed as 'PO#123', '123/10.0', ...)
    if src.get("po_pattern"):
        out["po"] = out["po"].str.extract(src["po_pattern"], expand=False).fillna("")
    if src.get("po_line_pattern"):
        pl = out["po"].copy()
        raw_po = _col(df, f.get("po"))
        out["po_line"] = raw_po.str.extract(src["po_line_pattern"], expand=False).fillna(out["po_line"])
    out["po_line"] = out["po_line"].map(_norm_line_no)
    out["invoice_key"] = out["invoice"].map(_invoice_key)
    for c in TEXT_FIELDS:
        out[c] = out[c].fillna("").astype(str).str.strip()
    out["raw_index"] = np.arange(len(out))
    return out.reset_index(drop=True)


def _norm_line_no(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    try:
        return str(int(float(x)))
    except ValueError:
        return x.strip().lstrip("0")


def _invoice_key(x: str) -> str:
    """'INV/2020/123456', 'TAX INV 123456', 'IN0123456' and '123456' all -> '123456'.

    Uses the trailing digit run (year prefixes like 2020/ are dropped) with leading zeros stripped."""
    if not isinstance(x, str) or not x.strip():
        return ""
    runs = re.findall(r"\d+", x)
    if not runs:
        return re.sub(r"[^A-Z0-9]", "", x.upper())
    return runs[-1].lstrip("0") or "0"


def ingest(cfg: Config, data_dir: str) -> pd.DataFrame:
    frames = [ingest_source(s, data_dir) for s in cfg["sources"]]
    df = pd.concat(frames, ignore_index=True)
    if df["rid"].duplicated().any():
        dup = df.loc[df["rid"].duplicated(), "rid"].head(3).tolist()
        raise ValueError(f"record ids are not unique, e.g. {dup}")
    return df
