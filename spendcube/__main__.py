"""python -m spendcube --config configs/A.json --data <dir with the company's files> --out outputs/x__A.csv"""
from __future__ import annotations

import argparse
import logging
import sys

from .clean import DataError
from .ingest import load_config
from .pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spendcube", description="Raw purchase data + category tree -> spend cube")
    ap.add_argument("--config", required=True, help="company config (JSON)")
    ap.add_argument("--data", required=True, help="directory holding the files named in the config")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--train", default=None,
                    help="directory with the labelled build data (labels, earlier files). Defaults to --data. "
                         "For A2/B2 point this at the original kit folder so the tool reuses what it learned.")
    ap.add_argument("--no-labels", action="store_true", help="ignore labels: classify from the tree alone")
    ap.add_argument("--no-embeddings", action="store_true", help="lexical features only (no model download)")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--holdout", type=float, default=0.0, help=argparse.SUPPRESS)  # evaluation only
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    try:
        run(cfg, a.data, a.out, train_dir=a.train, use_labels=not a.no_labels,
            use_embeddings=not a.no_embeddings, audit=not a.no_audit, holdout=a.holdout)
    except DataError as e:
        logging.getLogger("spendcube").error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
