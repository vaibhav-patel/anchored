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

🚧 **Week 1 — Foundations.** Document processing + naive RAG baseline.

## Stack

Everything runs in Docker — `docker compose up` and a stranger can reproduce the baseline.

| Layer | Choice (Week 1) |
|---|---|
| Corpus | CUAD (510 commercial contracts, 41 clause categories) |
| Doc processing | Python pipeline → normalized text → chunks |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU-friendly) |
| Vector store | Qdrant |
| Generation | Ollama (local, optional) / OpenAI-compatible fallback |
| Eval | Recall@k / Precision@k on CUAD's labeled spans (own harness first) |

## Quickstart

```bash
git clone https://github.com/vaibhav-patel/anchored.git
cd anchored
cp .env.example .env        # configure (no secrets needed for the baseline)
make data                   # download + verify CUAD
docker compose up --build   # ingest → index → serve
make baseline               # produce BASELINE.md numbers
```

## Contributing

This is a learning-in-public build. Issues, ideas, and PRs are welcome once the Week 1
skeleton lands. The metric here is **demonstrated judgment**, not PR count — a real fix
with a measured before/after beats ten trivial ones.

## License

MIT — see [LICENSE](LICENSE).
