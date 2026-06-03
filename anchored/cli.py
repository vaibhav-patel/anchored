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
def data() -> None:
    """Download + verify the CUAD corpus. (Implemented in #2.)"""
    typer.secho("Not implemented yet — tracked in issue #2.", fg="yellow")


@app.command()
def ingest() -> None:
    """Process raw contracts → chunks.jsonl. (Implemented in #3.)"""
    typer.secho("Not implemented yet — tracked in issue #3.", fg="yellow")


@app.command()
def index() -> None:
    """Embed chunks and build the Elasticsearch index. (Implemented in #3.)"""
    typer.secho("Not implemented yet — tracked in issue #3.", fg="yellow")


@app.command()
def ask(q: str = typer.Argument(..., help="Question to answer over the corpus")) -> None:
    """Run an end-to-end query and print cited spans. (Implemented in #3.)"""
    typer.secho("Not implemented yet — tracked in issue #3.", fg="yellow")


@app.command()
def baseline() -> None:
    """Run the retrieval eval and write BASELINE.md. (Implemented in #5.)"""
    typer.secho("Not implemented yet — tracked in issue #5.", fg="yellow")


if __name__ == "__main__":
    app()
