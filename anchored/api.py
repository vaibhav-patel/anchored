"""FastAPI demo server — see retrieval in action.

Serves a minimal single-page UI and a JSON search endpoint over the dense retriever.
Week 1 is retrieval-first: the response is the set of top-k cited contract spans, not an
LLM-synthesized answer.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from anchored.config import settings
from anchored.index.build import get_client
from anchored.retrieve.search import dense_search

app = FastAPI(title="anchored — retrieval demo")

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    contract_id: str
    title: str
    char_start: int
    char_end: int
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    k: int
    latency_ms: float
    results: list[SearchHit]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    es = get_client()
    try:
        ok = es.ping()
        count = es.count(index=settings.es_index)["count"] if ok else 0
        return {"elasticsearch": ok, "index": settings.es_index, "documents": count}
    except Exception as exc:  # noqa: BLE001
        return {"elasticsearch": False, "error": str(exc)}


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    import time

    t0 = time.perf_counter()
    results = dense_search(req.query, k=req.k)
    latency_ms = (time.perf_counter() - t0) * 1000

    hits = [
        SearchHit(
            rank=i,
            chunk_id=r.chunk.chunk_id,
            contract_id=r.chunk.contract_id,
            title=r.chunk.title,
            char_start=r.chunk.char_start,
            char_end=r.chunk.char_end,
            score=round(r.score, 4),
            snippet=r.chunk.text.strip()[:600],
        )
        for i, r in enumerate(results, start=1)
    ]
    return SearchResponse(
        query=req.query, k=req.k, latency_ms=round(latency_ms, 1), results=hits
    )
