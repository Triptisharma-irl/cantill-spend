"""Category tree read at run time. No category names live in code."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd

from .ingest import read_csv


@dataclass
class Taxonomy:
    leaves: list[tuple[str, ...]]          # full paths, variable depth
    depth: int                             # max depth in this tree
    descriptions: dict[tuple, str]         # optional free-text per leaf (if the tree file has one)

    @property
    def n(self):
        return len(self.leaves)

    def leaf_index(self) -> dict[tuple, int]:
        return {p: i for i, p in enumerate(self.leaves)}

    def ancestors(self, level: int) -> list[tuple[str, ...]]:
        return [p[:level] for p in self.leaves]

    def path_text(self, i: int, sep=" > ") -> str:
        return sep.join(self.leaves[i])


def _level_columns(cols: list[str]) -> list[str]:
    def lvl(c):
        m = re.search(r"(?:^|[^a-z])(?:l|lvl|level|cat(?:egory)?[_ ]?l?)[_ ]?(\d)(?![0-9])", c.lower())
        return int(m.group(1)) if m else None
    found = [(lvl(c), c) for c in cols if lvl(c) is not None and not re.search(r"code|id$|_id|desc", c.lower())]
    if found:
        return [c for _, c in sorted(found)]
    return list(cols)


def load_taxonomy(path: str, spec: dict | None = None) -> Taxonomy:
    spec = spec or {}
    df = read_csv(path)
    cols = spec.get("levels") or _level_columns(list(df.columns))
    desc_col = spec.get("description_column")
    leaves, descs, seen = [], {}, set()
    for _, r in df.iterrows():
        path_ = tuple(str(r[c]).strip() for c in cols)
        # trim blank trailing levels -> variable depth leaves
        while path_ and path_[-1] in ("", "nan", "None", "-"):
            path_ = path_[:-1]
        if not path_ or path_ in seen:
            continue
        seen.add(path_)
        leaves.append(path_)
        if desc_col and desc_col in df.columns:
            descs[path_] = str(r[desc_col])
    # a node that is a strict prefix of another path is an internal node, not a leaf
    prefixes = {p[:k] for p in leaves for k in range(1, len(p))}
    leaves = [p for p in leaves if p not in prefixes]
    depth = max(len(p) for p in leaves)
    return Taxonomy(leaves=leaves, depth=depth, descriptions=descs)


def resolve(path: str, *dirs: str) -> str | None:
    for d in dirs:
        if d and os.path.exists(os.path.join(d, path)):
            return os.path.join(d, path)
    return path if os.path.exists(path) else None
