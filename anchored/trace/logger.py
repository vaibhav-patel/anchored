"""Retrieval tracing — observability built in from day one.

Every retrieval appends one append-only JSONL record so failures are *provable* rather
than guessable. This is the foundation of the break-measure-fix-write loop: you can
replay what was retrieved, with what scores, at what latency and cost.

Records are written to ``traces/retrieval.jsonl`` (gitignored). The schema is versioned
so later analysis can evolve without breaking old traces.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from anchored.config import settings

# Bump when the record shape changes in a backward-incompatible way.
TRACE_SCHEMA_VERSION = 1

# Local embedding (fastembed/ONNX) has no per-call API charge. Kept as a field so a
# future hosted embedder / LLM step can populate a real number without a schema change.
EMBED_COST_PER_QUERY_USD = 0.0


class RetrievalTrace(BaseModel):
    """One retrieval event. Append-only, machine-parseable."""

    schema_version: int = TRACE_SCHEMA_VERSION
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    event: str = "retrieval"

    query: str
    k: int
    index: str
    retriever: str = "dense"  # the retrieval strategy (hybrid/rerank come later)
    embed_model: str

    latency_ms: float
    num_results: int
    retrieved_chunk_ids: list[str]
    scores: list[float]
    top_score: float | None = None

    est_cost_usd: float = EMBED_COST_PER_QUERY_USD
    source: str = "cli"  # cli | api | eval — where the query originated


def _trace_path(traces_dir: str | Path | None = None) -> Path:
    base = Path(traces_dir or settings.traces_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / "retrieval.jsonl"


def log_retrieval_trace(trace: RetrievalTrace, *, traces_dir: str | Path | None = None) -> None:
    """Append a fully-formed RetrievalTrace as one JSONL line."""
    path = _trace_path(traces_dir)
    with path.open("a", encoding="utf-8") as f:
        f.write(trace.model_dump_json() + "\n")


def log_retrieval(record: dict[str, Any], *, traces_dir: str | Path | None = None) -> None:
    """Backward-compatible helper: build a RetrievalTrace from a loose dict and append it.

    Fills required fields with sensible defaults so existing call sites keep working while
    new fields (embed_model, est_cost, top_score, source) get populated.
    """
    scores = record.get("scores", [])
    trace = RetrievalTrace(
        query=record["query"],
        k=record.get("k", len(scores)),
        index=record.get("index", settings.es_index),
        embed_model=record.get("embed_model", settings.embed_model),
        latency_ms=record.get("latency_ms", 0.0),
        num_results=record.get("num_results", len(scores)),
        retrieved_chunk_ids=record.get("retrieved_chunk_ids", []),
        scores=scores,
        top_score=scores[0] if scores else None,
        est_cost_usd=record.get("est_cost_usd", EMBED_COST_PER_QUERY_USD),
        source=record.get("source", os.environ.get("ANCHORED_TRACE_SOURCE", "cli")),
    )
    log_retrieval_trace(trace, traces_dir=traces_dir)
