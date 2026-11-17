"""Sanity check of the no-PO matching pass: hide the PO on records the PO pass matched, re-match, compare."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spendcube.ingest import load_config
from spendcube.pipeline import prepare
from spendcube.entities import supplier_keys
from spendcube.matching import cross_match

cfg = load_config("configs/A.json")
df = prepare(cfg, os.path.expanduser(sys.argv[1]), None)
df["supplier_cluster"] = supplier_keys(df, cfg.rules)
m = cross_match(df, cfg["combine"])
p1 = m[m.match_how == "po+amount"]
rng = np.random.default_rng(1)
hide = set(p1.rid[rng.random(len(p1)) < 0.3])
d2 = df.copy()
d2.loc[d2.rid.isin(hide), "po"] = ""
m2 = cross_match(d2, cfg["combine"]).set_index("rid")
truth = p1.set_index("rid")["matched_rid"]
h = sorted(hide)
got = m2["matched_rid"].reindex(h)
print(f"hidden {len(h)}; re-matched {got.notna().mean():.3f}; same ERP line {(got == truth.reindex(h)).mean():.3f}")
# same ERP line OR an identical twin ERP line (same PO, same amount) counts as correct
E = df.set_index("rid")
same_po_amt = [(isinstance(g, str) and E.at[g, "po"] == E.at[t, "po"] and abs(E.at[g, "amount_net_doc"] - E.at[t, "amount_net_doc"]) < 0.05)
               for g, t in zip(got, truth.reindex(h))]
print(f"equivalent ERP line (same PO & amount): {np.mean(same_po_amt):.3f}")
print("PO-not-in-ERP tool lines matched by pass 2 (should be ~0):", m[(m.match_how != 'po+amount')].rid.isin(df[(df.src=='TOOL') & (df.po != '')].rid).sum())

# false-positive check: tool-only records (PO not in ERP) with their PO hidden should stay unmatched
erp_po = set(df[df.src == "ERP"].po)
tool_only = df[(df.src == "TOOL") & (df.po != "") & ~df.po.isin(erp_po)].rid
hide2 = set(tool_only[rng.random(len(tool_only)) < 0.5])
d3 = df.copy()
d3.loc[d3.rid.isin(hide2), "po"] = ""
m3 = cross_match(d3, cfg["combine"])
fp = m3.rid.isin(hide2).sum()
print(f"tool-only records with PO hidden: {len(hide2)}; wrongly matched to ERP: {fp} ({fp/len(hide2):.3f})")
