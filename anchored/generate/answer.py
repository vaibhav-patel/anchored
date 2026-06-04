"""Stage 5 (presentation) — format retrieved spans into a cited answer.

Week 1 is retrieval-first: the "answer" is the set of top-k contract spans, each labeled
with its citation (contract + char range). An LLM synthesis step is optional and only used
if an LLM endpoint is configured; it must cite the same chunk ids.
"""

from __future__ import annotations

from anchored.schema import RetrievedChunk


def _citation(rc: RetrievedChunk) -> str:
    c = rc.chunk
    return f"[{c.chunk_id} | {c.title} chars {c.char_start}-{c.char_end} | score {rc.score:.3f}]"


def format_cited_spans(results: list[RetrievedChunk], *, max_chars: int = 500) -> str:
    """Render top-k retrieved spans with citations (the baseline 'answer')."""
    if not results:
        return "No results."
    lines: list[str] = []
    for i, rc in enumerate(results, start=1):
        snippet = rc.chunk.text.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + " …"
        lines.append(f"{i}. {_citation(rc)}\n   {snippet}")
    return "\n".join(lines)
