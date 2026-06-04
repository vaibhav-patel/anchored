# anchored

**Grounded, eval-instrumented agentic RAG — every answer anchored to verifiable evidence.**

`anchored` is a retrieval-and-reasoning assistant over a messy legal corpus
([CUAD](https://www.atticusprojectai.org/cuad) — commercial contracts). It plans and executes
retrieval, reasons across agents, and returns **cited, verifiable** answers — with an eval
harness, tracing, and cost governance as first-class citizens, not afterthoughts.

## Why this exists

Most RAG demos answer a question and stop. `anchored` is built around the opposite discipline:

- **Retrieval and generation are evaluated separately** — so a wrong answer is provably a
  *retrieval miss* or a *generation miss*, never a guess.
- **Every claim carries a citation** to the exact contract span it came from.
- **We induce failure on purpose**, measure it with numbers, fix it with a named pattern,
  and publish the before/after.

## Status

✅ **Week 1 — Foundations complete.** A measurable, traced, demo-able naive RAG baseline
over CUAD, with a from-scratch eval harness. See the [results](#baseline-results-2026-06-04)
below and [`BASELINE.md`](BASELINE.md) for the live numbers. Next: Phase 1 — breaking
retrieval on purpose (hybrid + reranking).

## Stack

Everything runs in Docker — `docker compose up` and a stranger can reproduce the baseline.

| Layer | Choice (Week 1) |
|---|---|
| Corpus | CUAD v1 (510 commercial contracts, 41 clause categories) |
| Doc processing | Python pipeline → normalized text → token chunks (tiktoken `cl100k_base`, 512/64) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (local, CPU, 384-dim cosine) |
| Vector store | **Elasticsearch 8.13** (`dense_vector` kNN + BM25 `text` field, hybrid-ready) |
| Generation | Ollama (local, optional) / OpenAI-compatible fallback — *not yet wired (Phase 2)* |
| Eval | Recall@k / Precision@k on CUAD's labeled spans — **own harness, no framework** |

## Quickstart

```bash
git clone https://github.com/vaibhav-patel/anchored.git
cd anchored
cp .env.example .env        # configure (no secrets needed for the baseline)
make data                   # download + verify CUAD
docker compose up --build   # start Elasticsearch + the app
make ingest && make index   # process → embed → index (one-time, ~30 min on CPU)
make baseline               # produce BASELINE.md numbers
```

## Demo

With the stack up and the corpus indexed, open **http://localhost:8000** (or `make ui`) to
see retrieval in action — ask a question and watch the top-k contract spans come back, each
anchored to its exact clause (contract, char range, similarity score). Retrieval-only for
now; LLM-synthesized answers come later.

## Baseline results (2026-06-04)

Naive single-vector dense retrieval, scoped to the target contract (the realistic
contract-review task: *find clause X in this document*). Full breakdown in
[`BASELINE.md`](BASELINE.md); methodology in [`docs/notes/week-01.md`](docs/notes/week-01.md).

**Dataset:** [CUAD v1](https://www.atticusprojectai.org/cuad) (CC BY 4.0) — 510 commercial
contracts, 41 expert-annotated clause categories, 20,910 questions (6,702 answerable).
Indexed as **12,572 chunks**. The eval set is **205 labeled cases** (5 per category × 41),
gold spans relocated into normalized contract text by exact match (0 unaligned).

| Metric | Value |
|---|---|
| **recall@5** | **0.68** |
| **recall@10** | **0.81** |
| **precision@5** | **0.19** ¹ |
| cases | 205 |

¹ precision@5 has a low ceiling here — gold spans are sparse (~1–2 relevant chunks per
contract), so even perfect retrieval caps around 0.2–0.4. Read it as a relative signal.

> **Why scope to the contract?** CUAD's question text is a generic template (identical
> across all 510 contracts), so a corpus-wide query carries no signal about *which* contract
> to search. The same eval **unscoped** collapses to recall@5 ≈ **0.18** — proof that pooling
> contracts would measure document disambiguation, not clause retrieval.

### ✅ Working well (recall@10 = 1.0 — 14 of 41 categories)

Short, formulaic, lexically distinctive clauses are easy for dense retrieval:

`Cap On Liability` · `Uncapped Liability` · `Change Of Control` · `Anti-Assignment` ·
`Expiration Date` · `Effective Date` · `Minimum Commitment` · `Post-Termination Services` ·
`Warranty Duration` · `Third Party Beneficiary` · `Price Restrictions` ·
`Joint IP Ownership` · `Non-Compete` · `Insurance`

(`Governing Law`, `Parties`, and `Audit Rights` are also strong at recall@10 = 0.8.)

### ❌ Failure modes (where naive retrieval breaks — Phase 1 targets)

| Category | recall@5 | recall@10 | Why it's hard |
|---|---|---|---|
| Unlimited/All-You-Can-Eat-License | 0.0 | 0.2 | defined-term, diffuse, cross-referential |
| Volume Restriction | 0.4 | 0.4 | rare/negotiated, long-tail vocabulary |
| Most Favored Nation | 0.2 | 0.6 | concept spread across multiple sentences |
| Exclusivity | 0.4 | 0.6 | implied by context, not a labeled heading |
| Non-Transferable License | 0.4 | 0.6 | defined-term license language |
| Covenant Not To Sue | 0.2 | 0.8 | relevant chunk retrieved but out-of-top-5 |

**The pattern:** misses cluster on clauses needing **lexical / exact-term** signal (defined
terms, section labels) that a single dense vector smooths away — and on "right chunk, wrong
rank" cases (top score 0.79–0.86, relevant chunk just outside top-k). That points squarely
at Phase 1: **hybrid (BM25 + dense)**, **reranking**, and **better chunking** for diffuse
clauses. A relevance threshold would also catch out-of-distribution queries (e.g. asking for
"Connecticut discussions" returns a confident-but-wrong span at ~0.81).

## Contributing

This is a learning-in-public build. Issues, ideas, and PRs are welcome once the Week 1
skeleton lands. The metric here is **demonstrated judgment**, not PR count — a real fix
with a measured before/after beats ten trivial ones.

## License

MIT — see [LICENSE](LICENSE).
