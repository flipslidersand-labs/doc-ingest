"""doc-ingest CLI entrypoint."""

import logging
import re

import click
import httpx

from core.config import get_settings
from core.ingest_pipeline import IngestResult
from core.schema import Collection, collection_names, payload_origin
from core.url_guard import UnsafeURLError, assert_safe_url, describe_http_error
from ingest.arxiv import ingest_arxiv
from ingest.design_docs import ingest_design_docs
from ingest.external import sync_external
from ingest.notes import ingest_notes

# このリポのパッケージ。レベルはここにだけ設定し、root は WARNING のままにして httpx / urllib3 /
# qdrant_client など第三者のログ (httpx は INFO でリクエスト URL を出す) を静かに保つ。
_HELP_KEY = "doc_ingest.help_requested"
_OWN_LOGGERS = ("core", "ingest", "cli", "__main__")


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _clean(value: object) -> str:
    """外部文書由来の表示値から C0/C1 制御文字 (ESC/OSC 等) を除く。改行・タブは残す。"""
    return _CONTROL_CHARS.sub("", str(value))


def setup_logging(verbose: bool) -> None:
    """`-v` で自パッケージだけ DEBUG、通常は INFO。出力は stderr (root のハンドラ)。"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    level = logging.DEBUG if verbose else logging.INFO
    for name in _OWN_LOGGERS:
        logging.getLogger(name).setLevel(level)


class _Group(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # callback はサブコマンドの --help より先に走り、その時点で args は消費済み。先に控えておく。
        ctx.meta[_HELP_KEY] = "--help" in args
        return super().parse_args(ctx, args)


@click.group(cls=_Group)
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def main(verbose: bool):
    setup_logging(verbose)
    ctx = click.get_current_context(silent=True)
    if ctx is not None and ctx.meta.get(_HELP_KEY):
        return  # group callback はサブコマンドの --help より先に走る。help は設定なしで出せるようにする
    # .env を読み、設定エラー (EMBED_BATCH<1 等) を初回利用ではなく起動時に出す。
    try:
        get_settings()
    except ValueError as e:
        raise click.ClickException(f"invalid configuration: {e}") from e


@main.command()
@click.argument("url")
@click.option("--tags", default="", help="Comma-separated project tags (e.g. forge,fluxion)")
def arxiv(url: str, tags: str):
    """Ingest an arxiv paper or tech blog post."""
    try:
        assert_safe_url(url)
    except UnsafeURLError as exc:
        raise click.BadParameter(str(exc), param_hint="'URL'") from exc
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        ingest_arxiv(url, tags=tag_list)
    except ValueError as exc:  # 不正な arXiv ID・論文が見つからない・本文を抽出できない
        raise click.ClickException(str(exc)) from exc
    except httpx.HTTPStatusError as exc:  # str(exc) は query 付きの URL を含むので伏せて出す
        raise click.ClickException(describe_http_error(exc)) from exc


def _exit_on_failures(result: IngestResult) -> None:
    """Exit 1 when a file failed to ingest (#332). Distill-skipped chunks keep their old point,
    so they only warn (#350). The git hooks keep `|| true`."""
    summary = (
        f"{result.files_failed} file(s) failed, "
        f"{result.chunks_skipped} chunk(s) skipped (distill failed; old points kept)"
    )
    if result.files_failed:
        raise click.ClickException(f"{summary}; see warnings above")
    if result.chunks_skipped:
        click.echo(f"warning: {summary}", err=True)


@main.command()
@click.argument("file", required=False)
def design(file: str | None):
    """Ingest design docs (CLAUDE.md / ADR / docs/). Runs changed files when called from git hook."""
    _exit_on_failures(ingest_design_docs(file))


@main.command()
@click.argument("file", required=False)
def notes(file: str | None):
    """Ingest curated Obsidian vault notes (10-projects/20-areas/30-resources). Runs changed files when called from git hook."""
    _exit_on_failures(ingest_notes(file))


@main.command()
@click.option("--force", is_flag=True, help="Skip ETag check and re-ingest all sources")
@click.option("--dry-run", is_flag=True, help="Show what would be ingested without writing")
def sync(force: bool, dry_run: bool):
    """Sync external API docs (ETag-based freshness check)."""
    failed = sync_external(force=force, dry_run=dry_run)
    if failed:
        raise click.ClickException(f"{failed} source(s) failed (see warnings above)")


@main.command()
@click.argument("query")
@click.option(
    "--collection",
    default=Collection.EXTERNAL_DOCS.value,
    type=click.Choice(collection_names()),
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
@click.option(
    "--nugget-k",
    default=3,
    show_default=True,
    help="Sentences per chunk when --nugget",
    type=click.IntRange(min=1),
)
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
        src = _clean(payload_origin(r))
        section = _clean(r.get("section", ""))
        score_str = f"score={score:.4f}  " if score is not None else ""
        click.echo(f"\n[{i}] {score_str}{src}")
        click.echo(f"    trust: {_clean(r.get('trust') or 'unknown')}")
        if section:
            click.echo(f"    section: {section}")
        text = _clean(r.get("text", ""))
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

    col_w = max(len("collection"), *(len(r["name"]) for r in rows))
    click.echo(f"{'collection':<{col_w}}  {'points':>6}  last_ingested")
    click.echo(f"{'-' * col_w}  {'------':>6}  {'-------------------'}")
    for r in rows:
        click.echo(f"{r['name']:<{col_w}}  {r['points']:>6}  {r['last_ingested']}")


@main.command()
@click.argument("path")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option(
    "--allow-cloud",
    is_flag=True,
    help="Allow summarising a local PDF with Anthropic when the local LLM fails (default: local only)",
)
def pdf(path: str, tags: str, allow_cloud: bool):
    """Ingest a local PDF file into the research collection."""
    from ingest.pdf import ingest_pdf

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        ingest_pdf(path, tags=tag_list, allow_cloud=allow_cloud)
    except FileNotFoundError:
        raise click.ClickException(f"ファイルが見つかりません: {path}")
    except UnsafeURLError as e:
        raise click.ClickException(f"取り込めない URL です: {e}")
    except httpx.HTTPStatusError as e:  # str(e) は query 付きの URL を含むので伏せて出す
        raise click.ClickException(f"取得に失敗しました: {describe_http_error(e)}") from e


if __name__ == "__main__":
    main()
