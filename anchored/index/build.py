"""Stage 4 — indexing: embed chunks and bulk-load them into Elasticsearch.

The mapping carries both a ``dense_vector`` (used by the Week 1 kNN baseline) and a BM25
``text`` field, so Phase 1 hybrid (BM25 + dense + RRF) needs no reindex.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from anchored.config import settings
from anchored.index.embed import EMBED_DIM, embed_texts
from anchored.schema import Chunk

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "contract_id": {"type": "keyword"},
            "title": {"type": "text"},
            "text": {"type": "text"},  # BM25 — unused in Week 1, ready for hybrid
            "char_start": {"type": "integer"},
            "char_end": {"type": "integer"},
            "chunk_index": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBED_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


def get_client() -> Elasticsearch:
    return Elasticsearch(settings.es_url, request_timeout=60)


def recreate_index(es: Elasticsearch, index: str) -> None:
    """Drop and recreate the index so a rebuild is deterministic."""
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, **INDEX_MAPPING)


def _batched(iterable: Iterator[Chunk], size: int) -> Iterator[list[Chunk]]:
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def _batch_actions(index: str, batch: list[Chunk]) -> list[dict]:
    vectors = embed_texts([c.text for c in batch])
    return [
        {
            "_index": index,
            "_id": chunk.chunk_id,
            "_source": {**chunk.model_dump(), "embedding": vec},
        }
        for chunk, vec in zip(batch, vectors, strict=True)
    ]


def build_index(
    chunks: Iterator[Chunk],
    *,
    index: str | None = None,
    batch_size: int = 64,
    total: int | None = None,
) -> int:
    """Recreate the index and load embedded chunks batch-by-batch.

    Embedding and indexing are decoupled per batch (embed fully, then bulk-write) so
    throughput is stable and progress is observable. A single refresh at the end.
    """
    import time

    es = get_client()
    name = index or settings.es_index
    recreate_index(es, name)

    indexed = 0
    t0 = time.perf_counter()
    for batch in _batched(chunks, batch_size):
        actions = _batch_actions(name, batch)
        bulk(es, actions, refresh=False)
        indexed += len(actions)
        rate = indexed / max(time.perf_counter() - t0, 1e-6)
        suffix = f"/{total}" if total else ""
        print(f"  indexed {indexed}{suffix} chunks ({rate:.1f}/s)", flush=True)

    es.indices.refresh(index=name)
    return indexed
