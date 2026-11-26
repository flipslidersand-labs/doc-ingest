# doc-ingest CLI specification

Source of truth: `cli.py` (click definitions), `core/config.py` (environment variables) and
`tests/test_cli.py` (exit codes). The console script is `doc-ingest = "cli:main"`
(`pyproject.toml`).

## Global

```
doc-ingest [-v|--verbose] COMMAND [ARGS]...
```

| Option | Default | Description |
|--------|---------|-------------|
| `-v`, `--verbose` | off | Set the `core`, `ingest`, `cli` and `__main__` loggers to DEBUG (INFO otherwise). The root logger stays at WARNING so third-party libraries (httpx, qdrant_client) stay quiet. Logs go to stderr. |

On start-up (except when `--help` is requested) the group callback loads settings
(`core.config.get_settings()`), which reads `.env` via `python-dotenv` and validates values.
An invalid value (for example `EMBED_BATCH=0`) aborts with `Error: invalid configuration: ...`
and a non-zero exit code. `COMMAND --help` works even with an invalid configuration.

## Commands

### `arxiv URL`

Ingest an arxiv paper or a tech blog post into the `research` collection.

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `URL` (required) | - | HTTPS URL. Checked with `core.url_guard.assert_safe_url` (https only, public addresses only) before anything is fetched. An arxiv `abs`/`pdf` URL is ingested as a paper; any other URL as a blog post. |
| `--tags TEXT` | `""` | Comma-separated tags (`forge,fluxion`); blank items are dropped. |

### `design [FILE]`

Ingest design docs (`CLAUDE.md`, `AGENTS.md`, `docs/*.md`, `adr/*.md`, `ADR/*.md`) into the
`design` collection. Distillation is local-LLM only.

| Argument | Default | Description |
|----------|---------|-------------|
| `FILE` (optional) | none | With a file: ingest just that file (validated by `core.path_guard.assert_safe_path`, then by the same target test as the hook mode: a file that is not a design doc is skipped with a warning and counts as one failure). Without: ingest the matching files changed in the last commit (git post-commit hook mode). |

### `notes [FILE]`

Ingest curated Obsidian vault notes (`10-projects/`, `20-areas/`, `30-resources/`, `.md` only)
into the `notes` collection. Distillation is local-LLM only. `FILE` behaves as for `design`: it must also be a `.md` under one of the three directories (judged relative to its git repo root, or to the base directory outside a repo), so `00-inbox/` and `90-daily/` stay out even when named explicitly.

### `sync`

Sync the external API docs listed in `sources/external.yaml` into `external-docs`
(ETag-based freshness check).

| Option | Default | Description |
|--------|---------|-------------|
| `--force` | off | Skip the ETag check and re-ingest all sources. |
| `--dry-run` | off | Show what would be ingested without writing (no Qdrant writes, no source-state write-back, no Discord notification). |

### `search QUERY`

Semantic search. Needs the embedding service and Qdrant.

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `QUERY` (required) | - | Query text (embedded with mode `search`). |
| `--collection` | `external-docs` | One of `external-docs`, `research`, `design`, `notes`. |
| `--limit` | `5` | Number of results, integer 1-100. |
| `--nugget` | off | Replace each chunk with its BM25-ranked nugget sentences. |
| `--nugget-k` | `3` | Sentences per chunk when `--nugget` is set; integer >= 1. |

Prints `No results found.` when nothing matches. Otherwise each hit shows score, origin
(URL or path), `trust`, optional `section` and a 200-character preview. C0/C1 control characters
in payload values are stripped before printing.

### `list`

List Qdrant collections with point count and last ingest time (`No collections found.` if none).

### `pdf PATH`

Ingest a PDF into the `research` collection. `PATH` is a local path or, in the implementation
(`ingest/pdf.py`), an `http`/`https` URL (downloaded through `core.url_guard.safe_download`); the
help text only mentions local files. A `PATH` is a URL when its scheme (case-insensitive) is
`http` or `https`; anything else, including a file named `http_notes.pdf`, is a local path.
A local path must resolve inside `DOC_INGEST_BASE_DIR` (default: cwd) after following symlinks,
as for `design` / `notes`; otherwise `PermissionError` is raised before the file is opened. A URL
is not subject to this check (it goes through `core.url_guard`).

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `PATH` (required) | - | Local PDF path or URL. |
| `--tags TEXT` | `""` | Comma-separated tags. |
| `--allow-cloud` | off | Allow summarising a *local* PDF with Anthropic when the local LLM fails. Default is local only. A URL source is treated as public and may use the Anthropic fallback. |

## Help output

Captured with `uv run python cli.py ...` (the installed `doc-ingest` console script could not
be spawned by `uv run doc-ingest` in the checkout used to write this, because the package was
not installed there; the Usage line therefore shows `cli.py`). Nothing connecting to Qdrant or
the embedding service was run.

```
$ uv run python cli.py --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose  Enable DEBUG logging
  --help         Show this message and exit.

Commands:
  arxiv   Ingest an arxiv paper or tech blog post.
  design  Ingest design docs (CLAUDE.md / ADR / docs/).
  list    List Qdrant collections with point counts and last ingest time.
  notes   Ingest curated Obsidian vault notes...
  pdf     Ingest a local PDF file into the research collection.
  search  Search ingested documents by semantic similarity.
  sync    Sync external API docs (ETag-based freshness check).
```

```
$ uv run python cli.py search --help
Usage: cli.py search [OPTIONS] QUERY

  Search ingested documents by semantic similarity.

Options:
  --collection [external-docs|research|design|notes]
                                  Qdrant collection to search
  --limit INTEGER RANGE           Number of results (1-100)  [default: 5;
                                  1<=x<=100]
  --nugget                        Extract BM25 nugget sentences instead of
                                  full chunks
  --nugget-k INTEGER RANGE        Sentences per chunk when --nugget  [default:
                                  3; x>=1]
  --help                          Show this message and exit.
```

```
$ uv run python cli.py pdf --help
Usage: cli.py pdf [OPTIONS] PATH

  Ingest a local PDF file into the research collection.

Options:
  --tags TEXT    Comma-separated tags
  --allow-cloud  Allow summarising a local PDF with Anthropic when the local
                 LLM fails (default: local only)
  --help         Show this message and exit.
```

```
$ uv run python cli.py notes --help
Usage: cli.py notes [OPTIONS] [FILE]

  Ingest curated Obsidian vault notes (10-projects/20-areas/30-resources).
  Runs changed files when called from git hook.

Options:
  --help  Show this message and exit.
```

`arxiv`, `design`, `sync` and `list` `--help` outputs follow the same shape and list exactly
the options in the tables above.

## Environment variables

Loaded once by `core.config.load_settings()` (`.env` through `python-dotenv`, then
`os.environ`) into a frozen `Settings` object; modules call `get_settings()`. `.env.example`
is the template for local setup. Integer/float values that do not parse, or violate the
minimums below, abort start-up.

| Variable | Default | Purpose | Read by |
|----------|---------|---------|---------|
| `EMBED_URL` | `http://localhost:9092/embed/batch` | Primary HTTP embedding endpoint. | `core/embedding.py` |
| `EMBED_FALLBACK_URL` | empty (none) | Secondary endpoint used when the primary is unreachable or answers 501. | `core/embedding.py` |
| `EMBED_API_KEY` | unset | Sent as `X-API-Key` when set. | `core/embedding.py` |
| `EMBED_COLLECTION` | `sessions` | `collection` field of the embedding request (model routing on the service side). | `core/embedding.py` |
| `EMBED_TIMEOUT` | `180` (seconds) | HTTP timeout per embedding request. | `core/embedding.py` |
| `EMBED_BATCH` | `16` (>= 1) | Max texts per embedding POST. | `core/embedding.py` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint. | `core/qdrant.py` |
| `QDRANT_API_KEY` | unset | Qdrant API key (unauthenticated when unset). | `core/qdrant.py` |
| `LOCAL_LLM_URL` | `http://localhost:8080` | llama-server base URL (`/v1/chat/completions`). | `core/distiller.py` |
| `LOCAL_LLM_MODEL` | empty | `model` field of the request; omitted when empty. | `core/distiller.py` |
| `LOCAL_LLM_TIMEOUT` | `120` (seconds) | Local LLM request timeout. | `core/distiller.py` |
| `LOCAL_LLM_MAX_TOKENS` | `1500` | `max_tokens` for local distillation. | `core/distiller.py` |
| `ANTHROPIC_API_KEY` | empty | Enables the Anthropic fallback (public sources only). | `core/distiller.py` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Model for the fallback. | `core/distiller.py` |
| `DISTILL_WORKERS` | `1` (>= 1) | Threads distilling chunks of one file. | `core/ingest_pipeline.py` |
| `SYNC_WORKERS` | `4` (>= 1) | Threads fetching sources in `sync`. | `ingest/external.py` |
| `DISCORD_WEBHOOK_URL` | empty | Sync summary notification target (the URL is a credential). | `ingest/external.py` |
| `MAX_CHUNKS_PER_DOC` | `1000` (>= 1) | Chunk cap per document. | `core/chunker.py` (`cap_chunks`) |
| `MAX_LOCAL_BYTES` | `10485760` (>= 1) | Largest local file read (design/notes markdown, local PDF). | `core/ingest_pipeline.py`, `ingest/pdf.py` |
| `MAX_DOWNLOAD_BYTES` | `52428800` (>= 1) | Default cap for `safe_get` / `safe_download` bodies. | `core/url_guard.py` |
| `DOC_INGEST_BASE_DIR` | empty (current directory) | Root that `design`/`notes` FILE arguments and local `pdf` paths must stay inside. | `core/path_guard.py` |
| `EMBED_BACKEND` | - | Ignored (a warning is logged if set); the ONNX backend was removed. | `core/config.py` |

`core/doc_key.py` also reads `os.environ`, but only to strip `GIT_*` variables before calling git.

## Exit codes

| Command | Behaviour (from code and `tests/test_cli.py`) |
|---------|-----------------------------------------------|
| all, on success | `0`. |
| all, invalid configuration | non-zero (`Error: invalid configuration: ...`), except `--help`. |
| `design`, `notes` | `1` when `ingest_*` reports one or more failures (a chunk whose distillation failed, a file that raised, or a FILE that is not a target of the collection); message `N failure(s) while ingesting (see warnings above)`. `0` when nothing failed or nothing changed. The git hooks keep `\|\| true`. |
| `design`, `notes` with a FILE outside the allowed base | `PermissionError` is not caught by the CLI: traceback, exit code `1`. |
| `arxiv` | Unsafe URL (not https, private address): usage error (click `BadParameter`, exit code 2). `ValueError` from ingestion (bad arxiv id, not found, no extractable body): `Error: ...`, exit code 1. Other exceptions propagate as a traceback. |
| `pdf` | `FileNotFoundError` and `UnsafeURLError` become `Error: ...` (exit code 1). Other exceptions propagate. |
| `pdf` with a local PATH outside the allowed base | `PermissionError` is not caught by the CLI: traceback, exit code `1` (same as `design` / `notes`). |
| `sync` | Return value is not checked: failed sources are logged, recorded as `failed_at` in the state file and listed in the Discord summary, but the exit code stays `0`. Unhandled exceptions give a traceback. |
| `search`, `list` | No special handling; connection errors propagate as a traceback. |

## 日本語要約

- コマンド: `arxiv URL [--tags]`、`design [FILE]`、`notes [FILE]`、`sync [--force] [--dry-run]`、`search QUERY [--collection] [--limit 1-100 既定5] [--nugget] [--nugget-k 既定3]`、`list`、`pdf PATH [--tags] [--allow-cloud]`。グローバルは `-v/--verbose`(自パッケージのみ DEBUG)。
- `design` / `notes` は FILE 省略時に直近コミットの変更ファイルを取り込む(git フック用)。蒸留はローカル LLM のみ。FILE は `core/path_guard.py` で `DOC_INGEST_BASE_DIR`(既定は cwd)配下に制限され、さらにフック経由と同じ対象判定(design は `CLAUDE.md` 等のパターン、notes は 3 ディレクトリ配下の `.md`)を通る。外れたファイルは警告を出して取り込まず、失敗 1 件として数える(終了コード 1)。
- `pdf` は実装上 URL も受け付けるが help 文はローカル PDF のみ。URL 判定は scheme(`http`/`https`、大文字小文字不問)で行い、ローカル PATH は `DOC_INGEST_BASE_DIR` 配下に制限される(外は `PermissionError`)。`--allow-cloud` はローカル PDF の要約失敗時に Anthropic へ送ることを許可する(既定は送らない)。
- 終了コード: `design` / `notes` は取り込み失敗が 1 件以上なら 1。`arxiv` の不正 URL は usage エラー(2)、取り込み時の `ValueError` は 1。`pdf` の `FileNotFoundError` / `UnsafeURLError` は 1。`sync` は失敗ソースがあっても 0(`failed_at` と Discord 通知で把握)。
- 環境変数は上の表の通り(`core/config.py` が `.env` と環境から一括で読み、不正値は起動時に失敗)。
- `uv run doc-ingest --help` は検証環境では起動できなかったため、出力は `uv run python cli.py --help` のもの(Usage 行が `cli.py`)。
