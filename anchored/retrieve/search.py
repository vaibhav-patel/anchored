"""Stage 5 — dense query search: query -> top-k chunks via Elasticsearch kNN.

Naive single-vector dense retrieval (the Week 1 baseline). Every query is traced.
"""

from __future__ import annotations

import time

from elasticsearch import Elasticsearch

from anchored.config import settings
from anchored.index.embed import embed_query
from anchored.schema import Chunk, RetrievedChunk
from anchored.trace.logger import log_retrieval


def get_client() -> Elasticsearch:
    return Elasticsearch(settings.es_url, request_timeout=60)


def dense_search(
    query: str,
    *,
    k: int | None = None,
    index: str | None = None,
    es: Elasticsearch | None = None,
    trace: bool = True,
) -> list[RetrievedChunk]:
    """Embed the query and return the top-k most similar chunks (cosine kNN)."""
    top_k = k or settings.top_k
    name = index or settings.es_index
    client = es or get_client()

    t0 = time.perf_counter()
    vector = embed_query(query)
    resp = client.search(
        index=name,
        knn={
            "field": "embedding",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": max(top_k * 10, 100),
        },
        source_excludes=["embedding"],
        size=top_k,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    results: list[RetrievedChunk] = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append(RetrievedChunk(chunk=Chunk(**src), score=hit["_score"]))

    if trace:
        log_retrieval(
            {
                "query": query,
                "k": top_k,
                "index": name,
                "latency_ms": round(latency_ms, 2),
                "retrieved_chunk_ids": [r.chunk.chunk_id for r in results],
                "scores": [round(r.score, 4) for r in results],
            }
        )
    return results
