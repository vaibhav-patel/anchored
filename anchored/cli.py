"""Anchored command-line interface.

Thin Typer app that the Makefile wraps. Pipeline commands are stubs in this issue
(#1, skeleton) and get implemented in #2–#5.
"""

from __future__ import annotations

import typer

from anchored.config import settings

app = typer.Typer(help="anchored — grounded, eval-instrumented agentic RAG", no_args_is_help=True)


@app.command()
def config() -> None:
    """Print the resolved configuration (useful for verifying .env wiring)."""
    for key, value in settings.model_dump().items():
        if key == "llm_api_key" and value:
            value = "***"
        typer.echo(f"{key} = {value}")


@app.command()
def health() -> None:
    """Check connectivity to Elasticsearch."""
    from elasticsearch import Elasticsearch

    es = Elasticsearch(settings.es_url, request_timeout=5)
    try:
        ok = es.ping()
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Elasticsearch unreachable at {settings.es_url}: {exc}", fg="red")
        raise typer.Exit(code=1) from exc
    if ok:
        info = es.info()
        typer.secho(
            f"Elasticsearch OK at {settings.es_url} "
            f"(cluster={info['cluster_name']}, version={info['version']['number']})",
            fg="green",
        )
    else:
        typer.secho(f"Elasticsearch ping failed at {settings.es_url}", fg="red")
        raise typer.Exit(code=1)


@app.command()
def data(
    force: bool = typer.Option(False, "--force", help="Re-download even if the zip exists"),
) -> None:
    """Download + verify the CUAD corpus, then report sanity counts."""
    from anchored.ingest import cuad

    typer.echo(f"Acquiring CUAD into {settings.data_dir}/raw (license: {cuad.CUAD_LICENSE}) ...")
    stats = cuad.acquire(settings.data_dir, force=force)

    for key, value in stats.as_dict().items():
        typer.echo(f"  {key} = {value}")

    warnings = cuad.verify_stats(stats)
    if warnings:
        typer.secho("Sanity warnings: " + "; ".join(warnings), fg="yellow")
        raise typer.Exit(code=1)
    typer.secho("CUAD acquired and verified.", fg="green")


@app.command()
def ingest() -> None:
    """Process raw contracts → chunks.jsonl (load → normalize → chunk)."""
    from pathlib import Path

    from anchored.ingest.chunk import chunk_documents, write_chunks
    from anchored.ingest.load import load_contracts

    docs = load_contracts(settings.data_dir)
    typer.echo(f"Loaded {len(docs)} contracts.")

    out_path = Path(settings.data_dir) / "processed" / "chunks.jsonl"
    chunks = chunk_documents(
        docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    n = write_chunks(chunks, out_path)
    typer.secho(
        f"Wrote {n} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap}) "
        f"→ {out_path}",
        fg="green",
    )


@app.command()
def index() -> None:
    """Embed chunks and (re)build the Elasticsearch index."""
    from pathlib import Path

    from anchored.index.build import build_index
    from anchored.ingest.chunk import read_chunks

    chunks_path = Path(settings.data_dir) / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        typer.secho(f"{chunks_path} not found — run `anchored ingest` first.", fg="red")
        raise typer.Exit(code=1)

    total = sum(1 for _ in read_chunks(chunks_path))
    typer.echo(
        f"Embedding ({settings.embed_model}) and indexing {total} chunks "
        f"into '{settings.es_index}' ..."
    )
    n = build_index(read_chunks(chunks_path), total=total)
    typer.secho(f"Indexed {n} chunks into '{settings.es_index}'.", fg="green")


@app.command()
def ask(
    q: str = typer.Argument(..., help="Question to answer over the corpus"),
    k: int = typer.Option(None, "--k", help="Top-k (defaults to settings.top_k)"),
) -> None:
    """Run an end-to-end dense query and print cited spans."""
    from anchored.generate.answer import format_cited_spans
    from anchored.retrieve.search import dense_search

    results = dense_search(q, k=k)
    typer.echo(format_cited_spans(results))


@app.command()
def baseline() -> None:
    """Run the retrieval eval and write BASELINE.md. (Implemented in #5.)"""
    typer.secho("Not implemented yet — tracked in issue #5.", fg="yellow")


if __name__ == "__main__":
    app()
