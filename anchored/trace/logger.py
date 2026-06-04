"""Minimal retrieval tracing.

Appends one JSONL record per query so retrieval is observable from day one. Issue #4
expands this (latency, cost, generation traces); this is the seed the pipeline writes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from anchored.config import settings


def log_retrieval(record: dict[str, Any], *, traces_dir: str | Path | None = None) -> None:
    """Append a retrieval trace record to traces/retrieval.jsonl."""
    base = Path(traces_dir or settings.traces_dir)
    base.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), **record}
    with (base / "retrieval.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
