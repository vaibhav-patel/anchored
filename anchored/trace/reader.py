"""Read and summarize retrieval traces.

Turns the append-only JSONL log into something you can actually analyze — the point of
tracing. Used by the ``anchored trace`` CLI command and available for ad-hoc analysis.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from anchored.config import settings


def trace_path(traces_dir: str | Path | None = None) -> Path:
    return Path(traces_dir or settings.traces_dir) / "retrieval.jsonl"


def read_traces(traces_dir: str | Path | None = None) -> Iterator[dict]:
    """Stream trace records (tolerant of older/looser lines)."""
    path = trace_path(traces_dir)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(traces_dir: str | Path | None = None) -> dict:
    """Aggregate stats over all traces: count, latency, score, cost, sources."""
    latencies: list[float] = []
    top_scores: list[float] = []
    total_cost = 0.0
    sources: dict[str, int] = {}
    n = 0

    for rec in read_traces(traces_dir):
        n += 1
        if (lat := rec.get("latency_ms")) is not None:
            latencies.append(lat)
        ts = rec.get("top_score")
        if ts is None and rec.get("scores"):
            ts = rec["scores"][0]
        if ts is not None:
            top_scores.append(ts)
        total_cost += rec.get("est_cost_usd", 0.0) or 0.0
        src = rec.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    def _pctl(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        i = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return round(s[i], 3)

    return {
        "traces": n,
        "latency_ms": {
            "avg": _avg(latencies),
            "p50": _pctl(latencies, 0.5),
            "p95": _pctl(latencies, 0.95),
        },
        "top_score": {
            "avg": _avg(top_scores),
            "min": round(min(top_scores), 3) if top_scores else None,
        },
        "est_cost_usd_total": round(total_cost, 6),
        "sources": sources,
    }
