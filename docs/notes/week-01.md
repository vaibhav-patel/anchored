# Week 1 notes — naive RAG baseline

The point of Week 1 was never a good RAG system. It was a *measurable* one. We now have a
naive dense-retrieval baseline over CUAD with numbers we can spend the next phases beating.
See [`BASELINE.md`](../../BASELINE.md) for the live metrics; this note records what the
numbers *mean* and where retrieval already breaks.

## The baseline, in one line

Naive single-vector dense retrieval (bge-small-en-v1.5, 512/64 token chunks, cosine kNN in
Elasticsearch), scoped to the target contract:

| recall@5 | recall@10 | precision@5 | cases |
|---|---|---|---|
| ~0.68 | ~0.81 | ~0.19 | 205 (5 / category, 41 categories) |

## Three decisions that shaped the measurement

1. **Within-contract retrieval, not corpus-wide.** CUAD's question text is a *generic
   template* — the "Governing Law" question is byte-identical across all 510 contracts. A
   corpus-wide query therefore carries no signal about *which* contract to search, so it
   mostly measures document disambiguation, not clause retrieval. We scope retrieval to the
   target contract. The empirical justification is stark: the same eval **unscoped** scores
   recall@5 ≈ 0.18 vs **0.68 scoped**. Scoping isolates the thing we actually care about.

2. **Coordinate-free gold spans.** CUAD's `answer_start` indexes the JSON `context` string,
   but our chunks store offsets into the *normalized* contract text (CRLF→LF, BOM strip,
   rstrip — all of which shift offsets). Trusting offsets across two coordinate systems
   would silently corrupt the ground truth. Instead we re-locate each gold answer's text
   inside the normalized text by exact string match and recompute the span there. Result:
   **0 unaligned cases** — every gold span maps cleanly.

3. **precision@5 has a low ceiling.** Gold spans are sparse (often 1–2 relevant chunks per
   contract), so even perfect retrieval caps precision@5 around 0.2–0.4. We track it as a
   *relative* signal across experiments, not an absolute target. recall@k is the headline.

## A bug worth recording

The eval set builder initially ran the JSON `title` through `Path(...).stem` to derive a
contract id. Contract titles contain dots (`...EX-10.26-PROMOTION...`), which `.stem`
mis-parses as a file extension and truncates — so **457/510 contracts silently failed to
match** their indexed text and fell back to a wrong coordinate system. Caught it by asserting
that every emitted case maps to an indexed contract. Fix: the JSON title *is* the contract
id; don't path-parse it. Lesson: validate the join key, don't assume it.

## Where retrieval already breaks (Phase 1 seeds)

From the per-category recall table (worst first):

- **License-type clauses are the weak spot.** "Unlimited/All-You-Can-Eat-License" (0.0 @5),
  "Non-Transferable License", "Affiliate License-*", "Irrevocable Or Perpetual License" all
  lag. These are *defined-term* clauses whose language is diffuse and cross-referential —
  exactly where a single dense vector over a 512-token window loses the thread.
- **Rare / negotiated clauses** ("Most Favored Nation", "Volume Restriction", "Covenant Not
  To Sue") under-retrieve — likely long-tail vocabulary the small embedder handles poorly.
- **Strong categories** ("Governing Law", "Parties", "Audit Rights", "Effective Date") sit
  at ~0.8 — short, formulaic, lexically distinctive clauses are easy for dense retrieval.

### The pattern, and the candidate fixes (for Phase 1)

The misses cluster on clauses that need **lexical / exact-term** signal (defined terms,
section labels) that a single dense vector smooths away. That points squarely at the
Phase-1 plan:

- **Hybrid (BM25 + dense)** — recover exact-term matches the embedder misses.
- **Reranking** over a larger candidate set — fix "right chunk, wrong rank" (note several
  misses have a top score in the 0.79–0.86 band: a relevant chunk was likely retrieved but
  out-of-top-k).
- **Better chunking** for diffuse/cross-referential clauses (contextual or late chunking).
- **A relevance threshold** — the "Connecticut" out-of-distribution query (top score ~0.81,
  all from one irrelevant contract) shows the system has no notion of "no confident match".

Every one of these is a break-measure-fix-write loop with this baseline as the starting line.

## Reproduce

```bash
make data
make ingest && make index      # ~30 min, one-time (CPU embedding)
make baseline                  # regenerates BASELINE.md + evals/cuad_retrieval.jsonl
```
