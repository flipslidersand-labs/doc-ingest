"""doc-ingest CLI entrypoint."""
import click

from ingest.arxiv import ingest_arxiv
from ingest.design_docs import ingest_design_docs
from ingest.external import sync_external


@click.group()
def main():
    pass


@main.command()
@click.argument("url")
@click.option("--tags", default="", help="Comma-separated project tags (e.g. forge,fluxion)")
def arxiv(url: str, tags: str):
    """Ingest an arxiv paper or tech blog post."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    ingest_arxiv(url, tags=tag_list)


@main.command()
@click.argument("file", required=False)
def design(file: str | None):
    """Ingest design docs (CLAUDE.md / ADR / docs/). Runs changed files when called from git hook."""
    ingest_design_docs(file)


@main.command()
@click.option("--force", is_flag=True, help="Skip ETag check and re-ingest all sources")
@click.option("--dry-run", is_flag=True, help="Show what would be ingested without writing")
def sync(force: bool, dry_run: bool):
    """Sync external API docs (ETag-based freshness check)."""
    sync_external(force=force, dry_run=dry_run)


@main.command()
@click.argument("query")
@click.option("--collection", default="external-docs",
              type=click.Choice(["external-docs", "research", "design"]),
              help="Qdrant collection to search")
@click.option("--limit", default=5, show_default=True, help="Number of results")
def search(query: str, collection: str, limit: int):
    """Search ingested documents by semantic similarity."""
    from core.qdrant import search as qdrant_search

    results = qdrant_search(collection, query, limit=limit)
    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r.pop("score")
        src = r.get("source_url") or r.get("file_path") or r.get("arxiv_id", "")
        section = r.get("section", "")
        click.echo(f"\n[{i}] score={score}  {src}")
        if section:
            click.echo(f"    section: {section}")
        text = r.get("text", "")
        click.echo(f"    {text[:200].replace(chr(10), ' ')}{'…' if len(text) > 200 else ''}")


if __name__ == "__main__":
    main()
