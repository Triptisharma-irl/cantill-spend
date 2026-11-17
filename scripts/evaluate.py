"""Score an output file against the build labels.

python scripts/evaluate.py --config <path-to-config.json> --out outputs/tripti__A.csv --kit <path to the task kit>
Reports classification accuracy per level (by count and by spend), on all labelled records and on
held-out ones (if the run used --holdout), plus supplier/item grouping quality (pairwise P/R/F1)
and, for B, exclusion agreement with the analyst's seed labels."""
import argparse
import os

import numpy as np
import pandas as pd


def pair_prf(pred: pd.Series, gold: pd.Series):
    """Pairwise precision/recall of a clustering, computed from the contingency table."""
    ok = gold.notna() & (gold != "")
    pred, gold = pred[ok].astype(str), gold[ok].astype(str)
    c2 = lambda n: n * (n - 1) / 2
    tp = c2(pd.crosstab(pred, gold).values.astype(float)).sum() if len(pred) else 0
    pp = c2(pred.value_counts().values.astype(float)).sum()
    gp = c2(gold.value_counts().values.astype(float)).sum()
    p = tp / pp if pp else 1.0
    r = tp / gp if gp else 1.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def cls_report(df, levels, mask, name):
    d = df[mask]
    if not len(d):
        return
    w = d["spend_abs"].values
    parts = []
    for i, _ in enumerate(levels):
        hit = np.ones(len(d), bool)
        for j in range(i + 1):
            hit &= d[f"l{j+1}"].fillna("").values == d[f"g{j+1}"].fillna("").values
        parts.append(f"L{i+1} {hit.mean():.3f}/{(w*hit).sum()/max(w.sum(),1e-9):.3f}")
    print(f"  {name:28s} n={len(d):6d}  " + "  ".join(parts) + "   (count/spend)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", required=True, choices=["A", "B"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kit", required=True, help="folder holding company_A/ and company_B/ (the task kit)")
    a = ap.parse_args()
    out = pd.read_csv(a.out, dtype={"record_id": str}, keep_default_na=False)
    aud_path = a.out.replace(".csv", "_audit.csv")
    aud = pd.read_csv(aud_path, keep_default_na=False, low_memory=False) if os.path.exists(aud_path) else None
    if a.company == "A":
        lab = pd.read_csv(os.path.join(a.kit, "company_A/labels_A.csv"), dtype=str, keep_default_na=False)
        levels = ["category_l1", "category_l2", "category_l3"]
        df = out.merge(lab, left_on="record_id", right_on="record_id", how="inner")
    else:
        lab = pd.read_csv(os.path.join(a.kit, "company_B/seed_labels.csv"), dtype=str, keep_default_na=False)
        levels = ["category_l1", "category_l2", "category_l3"]
        df = out.merge(lab, left_on="record_id", right_on="line_id", how="inner")
    for i, l in enumerate(levels):
        df[f"g{i+1}"] = df[l]
    df["spend_abs"] = pd.to_numeric(df["spend"], errors="coerce").abs().fillna(0)
    if aud is not None:
        df = df.merge(aud[["rid", "spend_all"] + (["holdout"] if "holdout" in aud else [])], left_on="record_id", right_on="rid", how="left")
        df["spend_abs"] = df["spend_all"].abs()
    print(f"== {a.out}")
    if a.company == "B":
        inc_pred = df["include_x"].astype(int)
        inc_gold = df["include_y"].astype(int)
        print("  exclusion agreement with seed: %.4f" % (inc_pred == inc_gold).mean())
        print(pd.crosstab(df["exclusion_reason_y"].replace("", "INCLUDED"), df["exclusion_reason_x"].replace("", "INCLUDED"),
                          rownames=["seed"], colnames=["ours"]).to_string())
        df = df[(inc_gold == 1) & (inc_pred == 1)]
    else:
        df = df[df["include"].astype(int) == 1]
    cls_report(df, levels, np.ones(len(df), bool), "all labelled")
    if "holdout" in df and df["holdout"].astype(str).isin(["True", "true", "1"]).any():
        h = df["holdout"].astype(str).isin(["True", "true", "1"])
        cls_report(df, levels, h.values, "held-out (labels hidden)")
        cls_report(df, levels, (~h).values, "seen")
    if a.company == "A":
        for k in ["supplier_key", "item_key"]:
            col_pred = k + "_x" if k + "_x" in df else k
            col_gold = k + "_y" if k + "_y" in df else k
            p, r, f = pair_prf(df[col_pred], df[col_gold])
            print(f"  {k:12s} pairwise P={p:.3f} R={r:.3f} F1={f:.3f}  (pred groups {df[col_pred].nunique()}, gold {df[col_gold].replace('', np.nan).nunique()})")
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    hit = (df["l3"] == df["g3"]) if "g3" in df else None
    if hit is not None:
        bins = pd.cut(conf, [0, .3, .5, .7, .9, 1.0])
        print("  calibration (conf bin -> leaf acc):", hit.groupby(bins, observed=True).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
