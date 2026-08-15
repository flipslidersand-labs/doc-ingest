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
def sync(force: bool):
    """Sync external API docs (ETag-based freshness check)."""
    sync_external(force=force)


if __name__ == "__main__":
    main()
