"""Render an EvalReport into BASELINE.md — the first numbers, with the exact config.

BASELINE.md is the most important artifact of Week 1: the measured starting line that the
break-measure-fix-write loop spends the next phases beating.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evals.retrieval_eval import EvalReport


def write_baseline_md(path: str | Path, report: EvalReport, dataset_stats: dict, settings) -> None:
    by_cat = report.by_category()
    worst = report.worst_cases(10)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# BASELINE.md")
    lines.append("")
    lines.append(
        "> The measured starting line for **anchored**. Naive dense retrieval over CUAD. "
        "These numbers exist to be beaten — every later fix is justified by moving them."
    )
    lines.append("")
    lines.append(f"_Generated {now}_")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| **recall@5** | **{report.recall_at_5}** |")
    lines.append(f"| **recall@10** | **{report.recall_at_10}** |")
    lines.append(f"| **precision@5** | **{report.precision_at_5}** |")
    lines.append(f"| cases evaluated | {report.n_cases} |")
    lines.append("")

    lines.append("## Configuration (what produced these numbers)")
    lines.append("")
    lines.append("| Knob | Value |")
    lines.append("|---|---|")
    lines.append("| retriever | dense kNN (cosine), scoped to the target contract |")
    lines.append(f"| embed model | `{settings.embed_model}` |")
    lines.append("| vector store | Elasticsearch (`dense_vector`) |")
    lines.append(
        f"| chunk size / overlap | {settings.chunk_size} / {settings.chunk_overlap} tokens |"
    )
    lines.append("| chunker | tiktoken `cl100k_base`, fixed-size sliding window |")
    lines.append(f"| index | `{settings.es_index}` |")
    lines.append("")

    lines.append("## Eval set")
    lines.append("")
    lines.append(
        f"- **{dataset_stats.get('emitted', report.n_cases)} labeled cases** across "
        f"**{dataset_stats.get('categories', '?')} clause categories**, built from CUAD's "
        "expert annotations (`evals/cuad_retrieval.jsonl`)."
    )
    lines.append(
        f"- Gold spans relocated into normalized contract text by exact match; "
        f"**{dataset_stats.get('unaligned', 0)}** answerable questions dropped as unalignable."
    )
    lines.append("")

    lines.append("## How to read these")
    lines.append("")
    lines.append(
        "- **Task = within-contract clause retrieval.** CUAD's question text is a generic "
        "template (identical across all 510 contracts), so a corpus-wide query carries no "
        "signal about *which* contract to search. We therefore scope retrieval to the target "
        "contract — the realistic contract-review task (find clause X in *this* document). "
        "Pooling all contracts would measure title disambiguation, not clause retrieval."
    )
    lines.append(
        "- **recall@k** = a relevant chunk (overlapping a gold span) appears in the top-k. "
        "This is the metric that matters most for a retrieval baseline."
    )
    lines.append(
        "- **precision@5** has a low ceiling here: gold spans are sparse (often 1-2 relevant "
        "chunks per contract), so even perfect retrieval caps around 0.2-0.4. Treat it as a "
        "relative signal across experiments, not an absolute target."
    )
    lines.append("")
    if report.unscoped_recall_at_5 is not None:
        lines.append(
            f"> **Why scope to the contract (empirical):** the same eval run *without* the "
            f"contract filter scores recall@5 = **{report.unscoped_recall_at_5}** "
            f"(vs **{report.recall_at_5}** scoped). Pooling all 510 contracts mostly measures "
            f"whether the generic question text lands in the right *document* — not whether we "
            f"retrieve the right *clause*. Scoping isolates the latter."
        )
        lines.append("")

    lines.append("## Recall by clause category")
    lines.append("")
    lines.append("| Category | n | recall@5 | recall@10 |")
    lines.append("|---|---|---|---|")
    for cat in sorted(by_cat, key=lambda c: (by_cat[c]["recall@10"], by_cat[c]["recall@5"])):
        a = by_cat[cat]
        lines.append(f"| {cat} | {a['n']} | {a['recall@5']} | {a['recall@10']} |")
    lines.append("")

    lines.append("## Worst cases (error-analysis seeds for Phase 1)")
    lines.append("")
    if worst:
        lines.append("| Category | top score | hit@5 | hit@10 |")
        lines.append("|---|---|---|---|")
        for c in worst:
            ts = f"{c.top_score:.3f}" if c.top_score is not None else "—"
            lines.append(
                f"| {c.category} | {ts} | {'✓' if c.hit_at_5 else '✗'} | "
                f"{'✓' if c.hit_at_10 else '✗'} |"
            )
    else:
        lines.append("_No misses at @10 — every case surfaced a relevant chunk in the top-10._")
    lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("make data            # download + verify CUAD")
    lines.append("make ingest && make index   # process → embed → index (~30 min, one-time)")
    lines.append("make baseline        # regenerate this file")
    lines.append("```")
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
