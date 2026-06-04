"""Evaluate dense retrieval against the labeled CUAD eval set.

Metrics (retrieval-only; no LLM-as-judge — that's a later phase):

- **recall@k**  — fraction of cases where at least one retrieved chunk in the top-k
  overlaps a gold span. (Did we surface the right clause at all?)
- **precision@5** — average fraction of the top-5 retrieved chunks that overlap a gold
  span. NOTE: gold spans are sparse (often 1-2 relevant chunks per contract), so the
  ceiling on precision@5 is low by construction; read it as a relative signal, not an
  absolute target.

A retrieved chunk counts as relevant if its [char_start, char_end) overlaps any gold span,
using ``Chunk.overlaps`` — both live in the normalized-text coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass

from anchored.retrieve.search import dense_search
from anchored.schema import Chunk
from evals.dataset import EvalCase


def _chunk_is_relevant(chunk: Chunk, gold_spans: list[tuple[int, int]]) -> bool:
    return any(chunk.overlaps(s, e) for s, e in gold_spans)


@dataclass
class CaseResult:
    case_id: str
    category: str
    hit_at_5: bool
    hit_at_10: bool
    precision_at_5: float
    top_score: float | None
    num_retrieved: int


@dataclass
class EvalReport:
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    n_cases: int
    case_results: list[CaseResult]
    unscoped_recall_at_5: float | None = None  # same set without the contract filter

    def worst_cases(self, n: int = 10) -> list[CaseResult]:
        """Cases that missed at @10, then @5, ordered for error analysis."""
        missed = [c for c in self.case_results if not c.hit_at_10]
        near = [c for c in self.case_results if c.hit_at_10 and not c.hit_at_5]
        return (missed + near)[:n]

    def by_category(self) -> dict[str, dict]:
        agg: dict[str, dict] = {}
        for c in self.case_results:
            a = agg.setdefault(c.category, {"n": 0, "hit5": 0, "hit10": 0})
            a["n"] += 1
            a["hit5"] += int(c.hit_at_5)
            a["hit10"] += int(c.hit_at_10)
        for a in agg.values():
            a["recall@5"] = round(a["hit5"] / a["n"], 3)
            a["recall@10"] = round(a["hit10"] / a["n"], 3)
        return agg


def evaluate(
    cases: list[EvalCase],
    *,
    k_eval: int = 10,
    scoped_to_contract: bool = True,
) -> EvalReport:
    """Run retrieval for each case and compute recall@5/@10 and precision@5."""
    results: list[CaseResult] = []

    for case in cases:
        retrieved = dense_search(
            case.question,
            k=k_eval,
            contract_id=case.contract_id if scoped_to_contract else None,
            trace=False,
        )
        flags = [_chunk_is_relevant(r.chunk, case.gold_spans) for r in retrieved]

        hit_at_5 = any(flags[:5])
        hit_at_10 = any(flags[:10])
        top5 = flags[:5]
        precision_at_5 = (sum(top5) / 5.0) if top5 else 0.0

        results.append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                hit_at_5=hit_at_5,
                hit_at_10=hit_at_10,
                precision_at_5=precision_at_5,
                top_score=retrieved[0].score if retrieved else None,
                num_retrieved=len(retrieved),
            )
        )

    n = len(results) or 1
    return EvalReport(
        recall_at_5=round(sum(r.hit_at_5 for r in results) / n, 4),
        recall_at_10=round(sum(r.hit_at_10 for r in results) / n, 4),
        precision_at_5=round(sum(r.precision_at_5 for r in results) / n, 4),
        n_cases=len(results),
        case_results=results,
    )
