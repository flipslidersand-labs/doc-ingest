"""doc-ingest CLI entrypoint."""

import logging

import click
from dotenv import load_dotenv

load_dotenv()

from ingest.arxiv import ingest_arxiv
from ingest.design_docs import ingest_design_docs
from ingest.external import sync_external
from ingest.notes import ingest_notes


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def main(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@main.command()
@click.argument("url")
@click.option("--tags", default="", help="Comma-separated project tags (e.g. forge,fluxion)")
def arxiv(url: str, tags: str):
    """Ingest an arxiv paper or tech blog post."""
    if not url.startswith(("http://", "https://")):
        raise click.BadParameter("http(s):// で始まる URL を指定してください", param_hint="'URL'")
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    ingest_arxiv(url, tags=tag_list)


@main.command()
@click.argument("file", required=False)
def design(file: str | None):
    """Ingest design docs (CLAUDE.md / ADR / docs/). Runs changed files when called from git hook."""
    ingest_design_docs(file)


@main.command()
@click.argument("file", required=False)
def notes(file: str | None):
    """Ingest curated Obsidian vault notes (10-projects/20-areas/30-resources). Runs changed files when called from git hook."""
    ingest_notes(file)


@main.command()
@click.option("--force", is_flag=True, help="Skip ETag check and re-ingest all sources")
@click.option("--dry-run", is_flag=True, help="Show what would be ingested without writing")
def sync(force: bool, dry_run: bool):
    """Sync external API docs (ETag-based freshness check)."""
    sync_external(force=force, dry_run=dry_run)


@main.command()
@click.argument("query")
@click.option(
    "--collection",
    default="external-docs",
    type=click.Choice(["external-docs", "research", "design", "notes"]),
    help="Qdrant collection to search",
)
@click.option(
    "--limit",
    default=5,
    show_default=True,
    help="Number of results (1-100)",
    type=click.IntRange(min=1, max=100),
)
@click.option("--nugget", is_flag=True, help="Extract BM25 nugget sentences instead of full chunks")
@click.option("--nugget-k", default=3, show_default=True, help="Sentences per chunk when --nugget")
def search(query: str, collection: str, limit: int, nugget: bool, nugget_k: int):
    """Search ingested documents by semantic similarity."""
    from core.qdrant import search as qdrant_search

    results = qdrant_search(collection, query, limit=limit)
    if not results:
        click.echo("No results found.")
        return

    if nugget:
        from core.nugget import apply_nuggets

        results = apply_nuggets(query, results, top_k=nugget_k)

    for i, r in enumerate(results, 1):
        score = r.pop("score", None)
        src = r.get("source_url") or r.get("file_path") or r.get("arxiv_id", "")
        section = r.get("section", "")
        score_str = f"score={score:.4f}  " if score is not None else ""
        click.echo(f"\n[{i}] {score_str}{src}")
        if section:
            click.echo(f"    section: {section}")
        text = r.get("text", "")
        orig_len = r.get("nugget_source_len")
        suffix = f" (nugget from {orig_len}c)" if orig_len else ("…" if len(text) > 200 else "")
        preview = text[:200].replace(chr(10), " ")
        click.echo(f"    {preview}{suffix}")


@main.command("list")
def list_cmd():
    """List Qdrant collections with point counts and last ingest time."""
    from core.qdrant import list_collections

    rows = list_collections()
    if not rows:
        click.echo("No collections found.")
        return

    col_w = max(len(r["name"]) for r in rows)
    click.echo(f"{'collection':<{col_w}}  {'points':>6}  last_ingested")
    click.echo(f"{'-' * col_w}  {'------':>6}  {'-------------------'}")
    for r in rows:
        click.echo(f"{r['name']:<{col_w}}  {r['points']:>6}  {r['last_ingested']}")


@main.command()
@click.argument("path")
@click.option("--tags", default="", help="Comma-separated tags")
def pdf(path: str, tags: str):
    """Ingest a local PDF file into the research collection."""
    from ingest.pdf import ingest_pdf

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        ingest_pdf(path, tags=tag_list)
    except FileNotFoundError:
        raise click.ClickException(f"ファイルが見つかりません: {path}")


if __name__ == "__main__":
    main()
