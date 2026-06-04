"""Stage 2 — chunking: normalized contract text -> overlapping Chunk records.

Naive, token-based sliding window (the deliberate Week 1 baseline we A/B against later).
Char offsets are preserved exactly so each chunk can be checked against CUAD gold spans.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import tiktoken

from anchored.ingest.load import Document
from anchored.schema import Chunk

# cl100k_base is a reasonable, widely-used tokenizer for sizing; the exact tokenizer is
# a baseline choice, not a tuned one.
_ENCODING = "cl100k_base"


def _token_char_starts(enc: tiktoken.Encoding, text: str, token_ids: list[int]) -> list[int]:
    """Char offset at which each token begins, plus a final sentinel == len(text).

    For ASCII-dominant text (CUAD contracts) token boundaries align to char boundaries and
    this is exact. A correctness guard falls back to a proportional mapping in the rare case
    a multibyte character is split across tokens.
    """
    starts: list[int] = []
    pos = 0
    for tok in token_ids:
        starts.append(pos)
        pos += len(enc.decode([tok]))
    starts.append(pos)

    if pos != len(text):  # pragma: no cover - guard for non-ASCII edge cases
        n = len(token_ids)
        starts = [round(i * len(text) / n) for i in range(n)] + [len(text)]
    return starts


def chunk_document(
    doc: Document,
    *,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = _ENCODING,
) -> list[Chunk]:
    """Split one document into overlapping token windows with exact char provenance."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(doc.text)
    starts = _token_char_starts(enc, doc.text, token_ids)

    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    index = 0
    for i in range(0, len(token_ids), stride):
        j = min(i + chunk_size, len(token_ids))
        char_start = starts[i]
        char_end = starts[j]
        text = doc.text[char_start:char_end]
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.contract_id}::{index}",
                contract_id=doc.contract_id,
                title=doc.title,
                text=text,
                char_start=char_start,
                char_end=char_end,
                chunk_index=index,
            )
        )
        index += 1
        if j == len(token_ids):
            break
    return chunks


def chunk_documents(
    docs: Iterable[Document], *, chunk_size: int, chunk_overlap: int
) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def write_chunks(chunks: Iterable[Chunk], out_path: str | Path) -> int:
    """Persist chunks to a JSONL file; return the count written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_chunks(path: str | Path) -> Iterator[Chunk]:
    """Stream chunks back from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Chunk.model_validate_json(line)
