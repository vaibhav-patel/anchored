"""Acquire and verify the CUAD corpus.

CUAD (Contract Understanding Atticus Dataset) v1 is distributed as a single zip on
Zenodo under CC BY 4.0. This module downloads it, verifies the MD5 checksum, extracts
it into ``data/raw/``, and reports sanity counts.

Canonical source: https://zenodo.org/records/4595826
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

# Pinned release artifact (Zenodo record 4595826).
CUAD_URL = "https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content"
CUAD_ZIP_NAME = "CUAD_v1.zip"
CUAD_MD5 = "c38f490a984420b8a62600db401fafd5"
CUAD_LICENSE = "CC BY 4.0"

# Expected contents after extraction (used for sanity checks).
EXPECTED_CONTRACTS = 510  # entries in CUAD_v1.json
EXPECTED_TXT_FILES = 510  # full_contract_txt/*.txt
EXPECTED_CATEGORIES = 41  # clause categories


@dataclass
class CuadStats:
    """Sanity counts produced after a successful acquisition."""

    contracts: int
    txt_files: int
    categories: int
    total_questions: int
    answerable_questions: int

    def as_dict(self) -> dict[str, int]:
        return {
            "contracts": self.contracts,
            "txt_files": self.txt_files,
            "categories": self.categories,
            "total_questions": self.total_questions,
            "answerable_questions": self.answerable_questions,
        }


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check, not security
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, *, force: bool = False) -> Path:
    """Download CUAD_v1.zip into ``raw_dir`` and verify its MD5 checksum."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / CUAD_ZIP_NAME

    if zip_path.exists() and not force:
        if _md5(zip_path) == CUAD_MD5:
            return zip_path
        # Corrupt/partial download — re-fetch.
        zip_path.unlink()

    tmp_path = zip_path.with_suffix(".zip.part")
    with urlopen(CUAD_URL) as resp, tmp_path.open("wb") as out:  # noqa: S310 - pinned https URL
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)

    actual = _md5(tmp_path)
    if actual != CUAD_MD5:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"CUAD checksum mismatch: expected {CUAD_MD5}, got {actual}")

    tmp_path.rename(zip_path)
    return zip_path


def extract(zip_path: Path, raw_dir: Path) -> Path:
    """Extract the CUAD zip into ``raw_dir``; return the CUAD_v1 directory."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    cuad_dir = raw_dir / "CUAD_v1"
    if not cuad_dir.is_dir():
        raise FileNotFoundError(f"Expected {cuad_dir} after extraction")
    return cuad_dir


def compute_stats(cuad_dir: Path) -> CuadStats:
    """Compute sanity counts from the extracted corpus."""
    json_path = cuad_dir / "CUAD_v1.json"
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)["data"]

    qas = [qa for entry in data for para in entry["paragraphs"] for qa in para["qas"]]
    categories = {qa["id"].split("__", 1)[1] for qa in qas if "__" in qa["id"]}
    answerable = sum(1 for qa in qas if qa.get("answers"))
    txt_files = len(list((cuad_dir / "full_contract_txt").glob("*.txt")))

    return CuadStats(
        contracts=len(data),
        txt_files=txt_files,
        categories=len(categories),
        total_questions=len(qas),
        answerable_questions=answerable,
    )


def verify_stats(stats: CuadStats) -> list[str]:
    """Return a list of human-readable warnings if counts deviate from expectations."""
    warnings: list[str] = []
    if stats.contracts != EXPECTED_CONTRACTS:
        warnings.append(f"contracts={stats.contracts} (expected {EXPECTED_CONTRACTS})")
    if stats.txt_files != EXPECTED_TXT_FILES:
        warnings.append(f"txt_files={stats.txt_files} (expected {EXPECTED_TXT_FILES})")
    if stats.categories != EXPECTED_CATEGORIES:
        warnings.append(f"categories={stats.categories} (expected {EXPECTED_CATEGORIES})")
    return warnings


def acquire(data_dir: str | Path, *, force: bool = False) -> CuadStats:
    """End-to-end: download → verify → extract → stats. Idempotent."""
    raw_dir = Path(data_dir) / "raw"
    zip_path = download(raw_dir, force=force)
    extract(zip_path, raw_dir)
    return compute_stats(raw_dir / "CUAD_v1")
