"""Optional local multilingual sentence embeddings (fastembed / ONNX, CPU only).

If fastembed or the model is unavailable the classifier silently falls back to lexical features.
Vectors are cached on disk keyed by text hash so re-runs are fast."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("spendcube")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self.model_name = model
        self.cache_dir = Path(cache_dir or ROOT / ".cache" / "emb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self.available = True
        try:
            import fastembed  # noqa: F401
        except Exception:  # pragma: no cover
            self.available = False
            log.warning("fastembed not installed -> classifier runs without embeddings")

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.model_name, cache_dir=str(ROOT / ".models"),
                                        threads=os.cpu_count())
        return self._model

    def _cache_file(self) -> Path:
        return self.cache_dir / (hashlib.md5(self.model_name.encode()).hexdigest()[:10] + ".npz")

    def encode(self, texts: list[str]) -> np.ndarray | None:
        if not self.available:
            return None
        texts = [t if t else "-" for t in texts]
        keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        cf = self._cache_file()
        cache: dict[str, np.ndarray] = {}
        if cf.exists():
            try:
                z = np.load(cf, allow_pickle=False)
                cache = dict(zip(z["keys"].tolist(), z["vecs"]))
            except Exception:
                cache = {}
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
        if todo:
            try:
                model = self._load()
            except Exception as e:  # network / model download failure
                log.warning("embedding model unavailable (%s) -> lexical only", e)
                self.available = False
                return None
            log.info("embedding %d new texts with %s", len(todo), self.model_name)
            vecs = np.array(list(model.embed([t for _, t in todo], batch_size=64)), dtype=np.float32)
            for (k, _), v in zip(todo, vecs):
                cache[k] = v
            np.savez(cf, keys=np.array(list(cache.keys())), vecs=np.stack(list(cache.values())))
        M = np.stack([cache[k] for k in keys]).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        return M
