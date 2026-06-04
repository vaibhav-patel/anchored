"""Build a labeled retrieval eval set from CUAD annotations.

Each case is a (question, gold span) pair scoped to a single contract — the realistic
contract-review task: *find clause X within this document*. We deliberately do NOT pool
all 510 contracts into one corpus for this eval, because CUAD's question text is a generic
template (identical across contracts), so a corpus-wide query carries no signal about which
contract to retrieve from. Scoping to the target contract isolates the thing we actually
want to measure: can dense retrieval surface the right clause inside a contract?

Coordinate safety: CUAD's ``answer_start`` indexes the JSON ``context`` string, but our
chunks store offsets into the *normalized* contract text (load.normalize). Rather than
trust offsets across two coordinate systems, we re-locate each gold answer's text inside
the normalized text by exact string search and recompute the span there. Cases whose gold
text can't be found in the normalized text are dropped and reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from anchored.ingest.load import load_contracts, normalize


@dataclass
class EvalCase:
    """One labeled retrieval case: a question with a gold span inside one contract."""

    case_id: str
    contract_id: str
    category: str
    question: str
    # (start, end) offsets in normalized contract text
    gold_spans: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "contract_id": self.contract_id,
            "category": self.category,
            "question": self.question,
            "gold_spans": [list(s) for s in self.gold_spans],
        }

    @staticmethod
    def from_dict(d: dict) -> EvalCase:
        return EvalCase(
            case_id=d["case_id"],
            contract_id=d["contract_id"],
            category=d["category"],
            question=d["question"],
            gold_spans=[tuple(s) for s in d["gold_spans"]],
        )


def _category_from_qid(qid: str) -> str:
    return qid.split("__", 1)[1] if "__" in qid else qid


def _relocate_spans(normalized_text: str, answers: list[dict]) -> list[tuple[int, int]]:
    """Find each gold answer's text in the normalized contract text (coordinate-free)."""
    spans: list[tuple[int, int]] = []
    for ans in answers:
        text = ans.get("text", "").strip()
        if not text:
            continue
        idx = normalized_text.find(text)
        if idx >= 0:
            spans.append((idx, idx + len(text)))
    return spans


def build_cases(
    data_dir: str | Path,
    *,
    per_category: int = 1,
    max_cases: int | None = None,
    seed: int = 13,
) -> tuple[list[EvalCase], dict]:
    """Construct eval cases spanning many clause categories.

    Strategy: walk contracts in stable order; for each answerable question whose gold text
    relocates cleanly, emit a case — capping at ``per_category`` cases per clause category
    so the set spreads across categories rather than clustering on a few contracts.

    Returns (cases, stats).
    """
    json_path = Path(data_dir) / "raw" / "CUAD_v1" / "CUAD_v1.json"
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)["data"]

    # Map contract title -> normalized text (the same text we chunked/indexed).
    docs = {d.contract_id: d.text for d in load_contracts(data_dir)}

    per_cat_count: dict[str, int] = {}
    cases: list[EvalCase] = []
    stats = {"considered": 0, "no_answer": 0, "unaligned": 0, "no_doc": 0, "emitted": 0}

    for entry in data:
        # The JSON title IS the contract_id (== txt filename stem). Do NOT run it through
        # Path().stem — titles contain dots (e.g. "EX-10.26") that get mis-parsed as a
        # file extension, silently mangling the id.
        contract_id = entry["title"]
        norm = docs.get(contract_id)
        if norm is None:
            # Title/filename mismatch; skip (rare).
            stats["no_doc"] += 1
            norm = normalize(entry["paragraphs"][0]["context"])  # best-effort fallback

        for qa in entry["paragraphs"][0]["qas"]:
            answers = qa.get("answers", [])
            if not answers:
                continue
            stats["considered"] += 1
            category = _category_from_qid(qa["id"])
            if per_cat_count.get(category, 0) >= per_category:
                continue

            spans = _relocate_spans(norm, answers)
            if not spans:
                stats["unaligned"] += 1
                continue

            cases.append(
                EvalCase(
                    case_id=qa["id"],
                    contract_id=contract_id,
                    category=category,
                    question=qa["question"],
                    gold_spans=spans,
                )
            )
            per_cat_count[category] = per_cat_count.get(category, 0) + 1
            stats["emitted"] += 1
            if max_cases and len(cases) >= max_cases:
                break
        if max_cases and len(cases) >= max_cases:
            break

    stats["categories"] = len(per_cat_count)
    return cases, stats


def write_dataset(cases: list[EvalCase], out_path: str | Path) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return len(cases)


def read_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(EvalCase.from_dict(json.loads(line)))
    return cases
