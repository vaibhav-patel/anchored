"""Shared data contracts for the pipeline.

Every chunk carries provenance (contract id + character offsets) so a retrieved chunk
can be checked for overlap against CUAD's gold answer spans — the basis for recall@k /
precision@k in the eval harness.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of text with provenance back to its source contract."""

    chunk_id: str = Field(..., description="Stable id: '<contract_id>::<index>'")
    contract_id: str = Field(..., description="Source contract identifier (CUAD title)")
    title: str = Field(..., description="Human-readable contract title")
    text: str = Field(..., description="Chunk text")
    char_start: int = Field(..., description="Inclusive start offset into the contract text")
    char_end: int = Field(..., description="Exclusive end offset into the contract text")
    chunk_index: int = Field(..., description="0-based position of the chunk within the contract")

    def overlaps(self, span_start: int, span_end: int) -> bool:
        """True if this chunk's char range overlaps the half-open span [start, end)."""
        return self.char_start < span_end and span_start < self.char_end


class RetrievedChunk(BaseModel):
    """A chunk returned by search, with its relevance score."""

    chunk: Chunk
    score: float
