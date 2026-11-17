"""Optional LLM classifier (Anthropic Claude) for trees without labelled examples.

Used only when credentials are available (ANTHROPIC_API_KEY or an `ant auth login` profile) and
the run has no labels. Unique (description, supplier, GL) units are sent in batches with the
numbered leaf list; Claude returns a leaf number + confidence per unit (JSON schema enforced).
Results are cached on disk, so a re-run costs nothing. Any failure -> local classifier only.
Cost: see APPROACH.md (about $0.5-1 per 1,000 unique units with claude-opus-5-5 at low effort).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger("spendcube")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = """You classify company purchase (spend) lines into the company's own category tree.
Each line shows: the line description (may be abbreviated, OCR-damaged, in any language, or empty),
the supplier name, the general-ledger account text and other context. Use procurement knowledge:
what the supplier sells, what the abbreviation means, what the GL account implies.
Pick exactly one leaf from the numbered list below - the most specific category that fits.
When the description says nothing, decide from supplier and GL account and lower the confidence.
confidence is your probability (0-1) that the leaf is correct.

CATEGORY TREE (leaf number: path):
{tree}"""


def available() -> bool:
    if os.environ.get("SPENDCUBE_NO_LLM"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


class LLMClassifier:
    def __init__(self, leaves: list[tuple], model: str | None = None, batch: int = 40, workers: int = 4,
                 max_units: int = 8000):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.anthropic = anthropic
        self.model = model or os.environ.get("SPENDCUBE_LLM_MODEL", DEFAULT_MODEL)
        self.leaves = leaves
        self.tree = "\n".join(f"{i}: {' > '.join(p)}" for i, p in enumerate(leaves))
        self.system = SYSTEM.format(tree=self.tree)
        self.batch, self.workers, self.max_units = batch, workers, max_units
        th = hashlib.md5((self.model + self.tree).encode()).hexdigest()[:10]
        self.cache_path = ROOT / ".cache" / f"llm_{th}.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "calls": 0}

    def _schema(self):
        return {"type": "object", "additionalProperties": False, "required": ["results"],
                "properties": {"results": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["id", "leaf", "confidence"],
                    "properties": {"id": {"type": "integer"}, "leaf": {"type": "integer"},
                                   "confidence": {"type": "number"}}}}}}

    def _call(self, items: list[tuple[str, str]]) -> dict[str, tuple[int, float]]:
        lines = "\n".join(f"[{i}] {txt}" for i, (_, txt) in enumerate(items))
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=8000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": self._schema()}},
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Classify each line. Return one result per id.\n\n{lines}"}],
            )
        except self.anthropic.APIStatusError as e:
            log.warning("LLM call failed (%s): %s", e.status_code, e.message)
            return {}
        except self.anthropic.APIConnectionError as e:
            log.warning("LLM connection error: %s", e)
            return {}
        if resp.stop_reason in ("refusal", "max_tokens"):
            log.warning("LLM batch stopped: %s", resp.stop_reason)
            return {}
        u = resp.usage
        self.usage["input"] += u.input_tokens or 0
        self.usage["output"] += u.output_tokens or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["calls"] += 1
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)["results"]
        except (ValueError, KeyError):
            return {}
        out = {}
        for r in data:
            i, leaf = r.get("id"), r.get("leaf")
            if isinstance(i, int) and 0 <= i < len(items) and isinstance(leaf, int) and 0 <= leaf < len(self.leaves):
                out[items[i][0]] = (leaf, float(min(max(r.get("confidence", 0.5), 0.0), 1.0)))
        return out

    def classify(self, texts: list[str], priority: np.ndarray) -> np.ndarray:
        """texts: one prompt line per unit; priority: spend (highest first is sent first).
        Returns matrix (n_units x n_leaves) with LLM probability mass, rows of zeros where unknown."""
        keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        order = np.argsort(-priority)
        todo, seen = [], set()
        for i in order:
            k = keys[i]
            if k not in self.cache and k not in seen:
                seen.add(k)
                todo.append((k, texts[i][:400]))
        todo = todo[: self.max_units]
        if todo:
            log.info("LLM (%s): classifying %d new units in batches of %d", self.model, len(todo), self.batch)
            chunks = [todo[j:j + self.batch] for j in range(0, len(todo), self.batch)]
            with ThreadPoolExecutor(self.workers) as ex:
                for res in ex.map(self._call, chunks):
                    for k, v in res.items():
                        self.cache[k] = v
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
            log.info("LLM usage: %s", self.usage)
        P = np.zeros((len(texts), len(self.leaves)))
        for i, k in enumerate(keys):
            if k in self.cache:
                leaf, conf = self.cache[k]
                P[i] = (1 - conf) / max(len(self.leaves) - 1, 1)
                P[i, leaf] = conf
        return P
