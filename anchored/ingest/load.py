"""Stage 1 — document processing: raw CUAD contracts -> normalized text + metadata.

Normalization is intentionally light and deterministic (whitespace/encoding only) so
character offsets remain meaningful for span citations downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """A normalized contract ready for chunking."""

    contract_id: str
    title: str
    text: str
    source_path: str


def normalize(text: str) -> str:
    """Light, deterministic normalization.

    - Normalize CRLF/CR to LF so offsets are stable across platforms.
    - Strip a UTF-8 BOM if present.
    - Drop trailing whitespace at EOF.

    Intentionally does *not* collapse internal whitespace — that would shift offsets and
    destroy the alignment with CUAD's ``answer_start`` gold spans.
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip() + "\n"


def contract_id_from_path(path: Path) -> str:
    """Derive a stable contract id from a txt filename (the CUAD title stem)."""
    return path.stem


def load_contracts(data_dir: str | Path) -> list[Document]:
    """Load all CUAD plain-text contracts from ``data/raw/CUAD_v1/full_contract_txt``."""
    txt_dir = Path(data_dir) / "raw" / "CUAD_v1" / "full_contract_txt"
    if not txt_dir.is_dir():
        raise FileNotFoundError(
            f"{txt_dir} not found — run `make data` first to acquire the corpus."
        )

    docs: list[Document] = []
    for path in sorted(txt_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = normalize(raw)
        cid = contract_id_from_path(path)
        docs.append(
            Document(contract_id=cid, title=cid, text=text, source_path=str(path))
        )
    return docs
