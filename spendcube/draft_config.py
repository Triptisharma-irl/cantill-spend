"""Draft a company config from an unseen file: python -m spendcube.draft_config --data DIR --lines FILE --tree FILE

Guesses which column plays which role from names and contents, prints a profile of every column
(so a human can check the guess) and writes a JSON config to review. No company data is hard-coded."""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from .ingest import read_csv
from .parse import parse_amount, parse_dates

# Order matters: specific roles claim their column before generic ones (gross before amount,
# posting date before any other date).
ROLE_PATTERNS = {
    "id": r"^(line_?(id|ref|key)|record_?(id|ref)|id|row_?id|txn_?id|transaction_?id|recordid)$",
    "gross_amount": r"(gross|incl.*(vat|tax)|amount_?incl|total_?incl)",
    "amount": r"(net_?amount|line_?amount|amount|net_?value|value|spend|extended|line_?total|total)",
    "tax_rate": r"(vat_?rate|tax_?rate|vat_?pct|tax_?pct|vat%|tax%|vatrate|taxrate)",
    "currency": r"^(currency|curr|ccy|currency_?code|doc_?currency|document_?currency|cur)$",
    "fx_rate": r"(currency_?rate|fx_?rate|exchange_?rate|exch_?rate|exchrate|rate_?to)",
    "invoice_date": r"(invoice_?date|inv_?date|document_?date|doc_?date|bill_?date)",
    "posting_date": r"(posting_?date|post_?date|gl_?date|accounting_?date|booking_?date|entry_?date)",
    "supplier_name": r"(supplier_?name|vendor_?name|supplier$|vendor$|payee|creditor_?name|supplier_name_as)",
    "supplier_id": r"(vendor_?id|supplier_?id|vendor_?(no|num|code)|supplier_?(no|num|code)|creditor_?(id|no))",
    "description": r"(description|desc$|line_?text|item_?text|short_?text|memo|narrative|details|item_?name|service_?text)",
    "item_code": r"(material|item_?(code|no|num|id)|sku|product_?(code|id)|part_?(no|num))",
    "quantity": r"^(qty|quantity|units)$",
    "status": r"(status|approval|state)",
    "gl_code": r"^(gl|gl_?account|account|gl_?code|account_?code|nominal|gl_?acct)$",
    "gl_desc": r"(gl_?account_?desc|account_?desc|gl_?desc|account_?name|expense_?type|nominal_?desc|cost_?type)",
    "po_line": r"(po_?line|po_?item|order_?line)",
    "po": r"(po_?(number|no|num|ref|reference)$|^po$|purchase_?order|^po_?ref)",
    "invoice": r"(invoice_?(number|no|num|id)|inv_?(no|num)|document_?(number|no)|doc_?(no|num)$|^invoice$)",
    "document_type": r"(document_?type|doc_?type|transaction_?type|txn_?type)",
    "supplier_type": r"(supplier_?type|vendor_?type|vendor_?class|supplier_?class)",
}
# columns that are never the date to convert FX at
_NOT_A_TXN_DATE = r"(due|paid|pay|clear|terms|deliver|created|updated)"
_FX_FILE = r"(fx|exchange|currenc|rates?)"


def guess(df: pd.DataFrame) -> dict:
    roles = {}
    used = set()
    for role, pat in ROLE_PATTERNS.items():
        for c in df.columns:
            if c in used:
                continue
            if re.search(pat, c.lower()):
                roles[role] = c
                used.add(c)
                break
    # no explicit invoice/posting date: fall back to any other transaction-like date column
    if "posting_date" not in roles:
        for c in df.columns:
            if c not in used and "date" in c.lower() and not re.search(_NOT_A_TXN_DATE, c.lower()):
                roles["posting_date"] = c
                used.add(c)
                break
    return roles


def profile(df: pd.DataFrame) -> None:
    for c in df.columns:
        v = df[c].astype(str)
        top = v.value_counts().head(5)
        print(f"- {c}: unique={v.nunique()} blank={(v.str.strip() == '').sum()} top={dict(top)}")


def context_columns(df: pd.DataFrame, used: set, max_cols: int = 6) -> list[str]:
    """Low-cardinality categorical columns (system, site, order type, ...): they help classify lines
    whose description is blank. Dates, amounts and free text are skipped."""
    out = []
    for c in df.columns:
        if c in used or re.search(r"date|amount|value|qty|quantity|rate|price|terms", c.lower()):
            continue
        v = df[c].astype(str).str.strip()
        filled = v[v != ""]
        if len(filled) < 0.1 * len(df):
            continue
        nu = filled.nunique()
        numeric = filled.head(2000).str.fullmatch(r"-?[\d.,]+").mean() > 0.9
        if 2 <= nu <= 500 and not (numeric and nu > 20):  # keep codes and 0/1 flags, skip measures
            out.append((nu, c))
    return [c for _, c in sorted(out)[:max_cols]]


def po_patterns(po: pd.Series) -> dict:
    """If PO references carry prefixes/suffixes ('PO#', '/10'), propose extraction patterns."""
    v = po.astype(str).str.strip()
    v = v[v != ""]
    if v.empty or v.str.fullmatch(r"\d+").mean() > 0.95:
        return {}
    runs = v.str.extract(r"(\d{6,})")[0].dropna()
    if runs.empty:
        return {}
    n = int(runs.str.len().mode()[0])
    pre = runs[runs.str.len() == n].str[:2]
    top, share = pre.mode()[0], (pre == pre.mode()[0]).mean()
    out = {"po_pattern": f"({top}\\d{{{n - 2}}})" if share > 0.8 else f"(\\d{{{n}}})"}
    if v.str.contains(r"\d{6,}\s*/\s*\d+").mean() > 0.2:
        out["po_line_pattern"] = r"/(\d+)"
    return out


def find_fx_table(data_dir: str, skip: set) -> dict | None:
    for f in sorted(os.listdir(data_dir)):
        if f in skip or not f.lower().endswith(".csv") or not re.search(_FX_FILE, f.lower()):
            continue
        t = read_csv(os.path.join(data_dir, f))
        cols = {c.lower(): c for c in t.columns}
        cur = next((cols[c] for c in cols if re.search(r"curr|ccy", c)), None)
        mon = next((cols[c] for c in cols if re.search(r"month|period|date", c)), None)
        rate = next((cols[c] for c in cols if re.search(r"rate|per|units|value", c) and cols[c] not in (cur, mon)), None)
        if not (cur and mon and rate):
            continue
        name = rate.lower()
        if re.search(r"(usd|eur|gbp|zar|reporting)_?per|to_?(usd|eur|gbp|zar|reporting)", name):
            sem = "reporting_per_unit"
        elif re.search(r"per_?(usd|eur|gbp|zar|reporting)|units?_?per", name):
            sem = "units_per_reporting"
        else:  # most currencies are worth less than USD/EUR, so 'units per reporting' rates are mostly > 1
            sem = "units_per_reporting" if parse_amount(t[rate]).median() > 1 else "reporting_per_unit"
        return {"file": f, "currency": cur, "month": mon, "rate": rate, "semantics": sem}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lines", required=True, help="invoice-line file name inside --data")
    ap.add_argument("--tree", required=True, help="category tree file name inside --data")
    ap.add_argument("--company", default="C")
    ap.add_argument("--currency", default="USD", help="reporting currency")
    ap.add_argument("--default-tax-rate", type=float, default=0.15, help="VAT rate assumed when a gross line has none")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = read_csv(os.path.join(a.data, a.lines))
    tree = read_csv(os.path.join(a.data, a.tree))
    print(f"## {a.lines}: {len(df)} rows, {len(df.columns)} columns")
    profile(df)
    print(f"## {a.tree}: columns {list(tree.columns)}")
    r = guess(df)
    print("## guessed roles:", json.dumps(r, indent=1))
    warnings = []
    gross = "gross_amount" in r and "amount" not in r
    amount_col = r.get("amount") or r.get("gross_amount")
    if amount_col:
        bad = parse_amount(df[amount_col]).isna().sum()
        print(f"amount '{amount_col}' ({'gross: VAT will be removed' if gross else 'net'}): {bad} unparseable")
    else:
        warnings.append("no amount column recognised - set sources[0].amount.column by hand")
    for dc in [r.get("invoice_date"), r.get("posting_date")]:
        if dc:
            raw = df[dc].astype(str).str.strip()
            d = parse_dates(df[dc])
            print(f"date '{dc}': {(raw == '').sum()} blank, {(d.isna() & (raw != '')).sum()} unparseable, "
                  f"range {d.min()} .. {d.max()}")
    if r.get("status"):
        print("status values:", df[r["status"]].value_counts().to_dict())
    # ---- currency and FX
    fx_table = find_fx_table(a.data, {a.lines, a.tree})
    cur_counts = df[r["currency"]].astype(str).str.strip().str.upper().replace("", a.currency.upper()).value_counts() \
        if r.get("currency") else None
    if cur_counts is not None:
        foreign = cur_counts.drop(a.currency.upper(), errors="ignore")
        print(f"currencies: {cur_counts.head(10).to_dict()}")
        if len(foreign) and not fx_table and not r.get("fx_rate"):
            warnings.append(f"{int(foreign.sum())} lines are not in {a.currency} and there is no FX table or rate "
                            f"column: the run will stop. Add an fx_table, or check --currency.")
        elif len(foreign) and not fx_table:
            warnings.append("non-reporting currencies rely on the line rate column only (no FX table found)")
    if fx_table:
        print(f"FX table: {json.dumps(fx_table)}  <- check 'semantics' (units_per_reporting = local units per 1 {a.currency})")
    used = set(r.values())
    ctx = context_columns(df, used)
    pos = po_patterns(df[r["po"]]) if r.get("po") else {}
    date_cols = [c for c in [r.get("invoice_date"), r.get("posting_date")] if c]
    src = {
        "name": a.company, "file": a.lines, "id": r.get("id", df.columns[0]),
        "amount": {"column": amount_col, "basis": "gross" if gross else "net"},
        "currency": {"column": r["currency"]} if r.get("currency") else {"value": a.currency},
        "fields": {k: v for k, v in {
            "date": date_cols,
            "supplier_name": r.get("supplier_name"), "supplier_id": r.get("supplier_id"),
            "description": [r["description"]] if r.get("description") else None,
            "item_code": r.get("item_code"), "quantity": r.get("quantity"), "status": r.get("status"),
            "gl_code": r.get("gl_code"), "gl_desc": r.get("gl_desc"), "po": r.get("po"), "po_line": r.get("po_line"),
            "invoice": r.get("invoice"), "document_type": r.get("document_type"),
            "supplier_type": r.get("supplier_type"), "context": ctx}.items() if v is not None},
        **pos,
    }
    if gross:
        src["amount"].update({"tax_rate_column": r.get("tax_rate"), "default_tax_rate": a.default_tax_rate})
    if r.get("fx_rate") or fx_table:
        fx = {"date_field": date_cols[0] if date_cols else None}
        if r.get("fx_rate"):
            fx["line_rate_column"] = r["fx_rate"]
        if len(date_cols) > 1:
            fx["fallback_date_field"] = date_cols[1]
        src["fx"] = fx
    if not r.get("description"):
        warnings.append("no description column recognised - classification will rely on supplier/GL/context only")
    cfg = {"company": a.company, "reporting_currency": a.currency,
           "taxonomy": {"file": a.tree}, "dates": {"valid_from": "2015-01-01", "valid_to": "2026-12-31"},
           "sources": [src]}
    if fx_table:
        cfg["fx_table"] = fx_table
    out = a.out or f"configs/{a.company}.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"context columns: {ctx}" + (f"; PO extraction: {pos}" if pos else ""))
    for w in warnings:
        print("WARNING:", w)
    print(f"## draft written to {out} - review every field before running")


if __name__ == "__main__":
    main()
