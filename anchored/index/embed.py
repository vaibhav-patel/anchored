"""Stage 3 — embedding: text -> dense vectors via fastembed (ONNX, CPU-friendly).

A thin singleton wrapper around fastembed so the same model serves both indexing and
query-time embedding (they must match).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from fastembed import TextEmbedding

from anchored.config import settings

# bge-small-en-v1.5 is 384-dim. Kept as a constant so the ES mapping and embedder agree.
EMBED_DIM = 384

_model: TextEmbedding | None = None
_model_name: str | None = None


def get_model(model_name: str | None = None) -> TextEmbedding:
    """Return a cached TextEmbedding for ``model_name`` (defaults to settings.embed_model).

    ONNX thread count is bounded (not all cores) — oversubscription on many-core hosts
    causes severe throughput collapse under concurrent index writes.
    """
    global _model, _model_name
    name = model_name or settings.embed_model
    if _model is None or _model_name != name:
        threads = min(4, os.cpu_count() or 1)
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
        _model = TextEmbedding(model_name=name, threads=threads, cache_dir=cache_dir)
        _model_name = name
    return _model


def embed_texts(texts: Iterable[str], *, batch_size: int = 64) -> list[list[float]]:
    """Embed a batch of documents/passages."""
    model = get_model()
    return [vec.tolist() for vec in model.embed(list(texts), batch_size=batch_size)]


def embed_query(text: str) -> list[float]:
    """Embed a single query.

    fastembed's bge models prepend the recommended query instruction internally via
    ``query_embed``, which improves retrieval over using ``embed`` for queries.
    """
    model = get_model()
    return next(iter(model.query_embed(text))).tolist()
