"""Fast loop for the label-free classifier: prepare included lines once (cached), then score variants.

SPENDCUBE_KIT=<path to the task kit> python scripts/tune_zeroshot.py A '{"self_training_rounds":0}'"""
import json, os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spendcube.ingest import load_config
from spendcube.taxonomy import load_taxonomy, resolve
from spendcube.classify.core import SpendClassifier, ClassifierConfig
import logging; logging.basicConfig(level=logging.WARNING)

KIT = os.environ.get("SPENDCUBE_KIT") or sys.exit("set SPENDCUBE_KIT to the task kit folder (holds company_A/, company_B/)")
c = sys.argv[1]
over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
d = f"{KIT}/company_{c}"
cache = f".cache/tune_{c}.pkl"
cfg = load_config(f"configs/{c}.json")
tax = load_taxonomy(resolve(cfg["taxonomy"]["file"], d), cfg["taxonomy"])
if not os.path.exists(cache):
    from spendcube.pipeline import run
    df = run(cfg, d, f".cache/tune_{c}.csv", use_labels=False, use_embeddings=False, audit=False)
    lines = df[df.excl == ""][["rid", "description", "supplier_key", "supplier_name", "gl_desc", "context", "item_key", "spend_all"]].copy()
    lines["spend_abs"] = lines.spend_all.abs()
    pickle.dump(lines, open(cache, "wb"))
lines = pickle.load(open(cache, "rb"))
if c == "A":
    L = pd.read_csv(f"{d}/labels_A.csv", dtype=str, keep_default_na=False).rename(columns={"record_id": "rid"})
else:
    L = pd.read_csv(f"{d}/seed_labels.csv", dtype=str, keep_default_na=False).rename(columns={"line_id": "rid"})
    L = L[L.include == "1"]
gold = lines[["rid", "spend_abs"]].merge(L, on="rid")
t = time.time()
emb = over.pop("_emb", True)
clf = SpendClassifier(tax, cfg.rules["abbreviations"], ClassifierConfig(use_embeddings=emb, **{**cfg.get("classifier", {}), **over}))
res = clf.classify(lines.reset_index(drop=True))
res["rid"] = lines.rid.values
m = gold.merge(res, on="rid")
w = m.spend_abs.values
out = []
for i in range(3):
    hit = np.ones(len(m), bool)
    for j in range(i + 1):
        hit &= m[f"l{j+1}"].values == m[f"category_l{j+1}"].values
    out.append(f"L{i+1} {hit.mean():.3f}/{(w*hit).sum()/w.sum():.3f}")
print(c, json.dumps(over), " ".join(out), f"({time.time()-t:.0f}s)")
