# doc-ingest

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Release type: `library`

## English

Document ingestion pipeline — feeds external API docs, design docs, and arxiv papers into [Qdrant](https://qdrant.tech/) vector DB for use in RAG / AI-assistant workflows.

### Modules

| Module | Source | Trigger |
|--------|--------|---------|
| `ingest/external.py` | External API docs (GitHub, Anthropic, Qdrant…) | Weekly cron + ETag diff |
| `ingest/design_docs.py` | CLAUDE.md / ADR / docs/ | post-commit git hook |
| `ingest/notes.py` | Obsidian vault (10-projects/20-areas/30-resources) | post-commit git hook |
| `ingest/arxiv.py` | arxiv papers / tech blogs | CLI manual |
| `ingest/pdf.py` | Local PDF files or HTTPS PDF URLs | CLI manual |

### Setup

```bash
pip install -e ".[dev]"
bash scripts/install-precommit.sh  # required once per clone (use this instead of `pre-commit install`)
cp .env.example .env  # fill in QDRANT_URL, EMBED_URL, EMBED_API_KEY
```

Hook results are appended to `~/.local/state/precommit-metrics/events.jsonl` (`pass` / `fixed` / `fail`).
The script sets `core.hooksPath` to `.git/managed-hooks` (same for every worktree and branch). `pre-commit install` is refused once it is set, so it cannot overwrite the logging wrapper; hooks other than `pre-commit` (e.g. a `pre-push` secret check) are forwarded to `.git/hooks`. Re-run the script to update; it is idempotent and removes the old `.git/hooks/pre-commit` wrapper.

### Usage

```bash
# Ingest an arxiv paper
doc-ingest arxiv https://arxiv.org/abs/2401.00123 --tags forge,fluxion

# Ingest a tech blog post
doc-ingest arxiv https://example.com/blog/post

# Sync external API docs (ETag-based)
doc-ingest sync
doc-ingest sync --force    # skip ETag check
doc-ingest sync --dry-run  # preview without writing

# Ingest a specific design doc manually
doc-ingest design CLAUDE.md

# Ingest a curated Obsidian note manually (no argument = files changed in the last commit)
doc-ingest notes 10-projects/my-note.md

# Search ingested content
doc-ingest search "Claude API streaming"
doc-ingest search "qdrant vector" --collection external-docs --limit 3
doc-ingest search "qdrant vector" --nugget             # BM25 nugget sentences instead of full chunks
doc-ingest search "qdrant vector" --nugget --nugget-k 5  # sentences per chunk (default 3)

# List Qdrant collections with point counts and last ingest time
doc-ingest list

# Ingest a PDF (local file or HTTPS URL)
doc-ingest pdf ./paper.pdf --tags forge
doc-ingest pdf ./paper.pdf --allow-cloud  # allow Anthropic if the local LLM fails (default: local only)
doc-ingest pdf https://example.com/paper.pdf  # URL sources are treated as public (Anthropic fallback allowed)
```

`design` and `notes` exit with code 1 when one or more files or chunks failed to ingest. Details: [docs/spec/cli.md](docs/spec/cli.md).

### Git Hook

Install the post-commit hook into one or more repos:

```bash
# Default repos (mesh-drop / forge / fluxion)
bash scripts/install-hooks.sh

# Specific repo
bash scripts/install-hooks.sh /path/to/repo

# Preview without writing
bash scripts/install-hooks.sh --dry-run

# Remove hook snippet
bash scripts/install-hooks.sh --uninstall
```

Design docs (CLAUDE.md, ADR, docs/) are auto-ingested on every commit.

The hook points at the checkout that ran the installer (override with `DOC_INGEST_DIR`). It never
fails a commit, but if `doc-ingest` cannot start or exits non-zero it appends the time, repo, exit
code and last 5 stderr lines to `${XDG_STATE_HOME:-~/.local/state}/doc-ingest/post-commit.log`.
Hooks installed before this change keep the old fixed path: run `--uninstall`, then install again.

For an Obsidian vault (default `~/notes`), `--notes` installs a post-commit hook that runs the
`notes` command instead, so curated notes flow into the `notes` collection. It uses the same
safe `.env` loading as the design hook (KEY=VALUE lines only, never `source`d) and can live in the
same `post-commit` file as the design hook:

```bash
bash scripts/install-hooks.sh --notes               # ~/notes
bash scripts/install-hooks.sh --notes /path/to/vault
bash scripts/install-hooks.sh --notes --uninstall   # removes only the notes block
```

If the vault already has a hand-written hook from an older version of this README (one that
`source`s `.env`), the script refuses to append and tells you to remove it first — that recipe
executes arbitrary commands found in `.env` (#80, #221).

### Upgrading from pre-#216 points (`design` / `notes`)

Documents are now keyed `<repo>/<path-in-repo>` (stored as `file_path`, plus a `repo` payload), so two repos' `CLAUDE.md` no longer collide and the post-commit hook and manual runs agree. Points ingested before that have no `repo` payload and become orphans. Delete them, then re-ingest (the hook only picks up files changed in the last commit):

```bash
.venv/bin/python scripts/migrate_doc_keys.py            # dry-run: counts + sample paths
.venv/bin/python scripts/migrate_doc_keys.py --apply    # delete points without `repo`
# re-ingest every design doc of each hooked repo (run inside that repo; paths must be under cwd)
cd ~/projects/forge && for f in CLAUDE.md AGENTS.md docs/*.md adr/*.md ADR/*.md; do [ -f "$f" ] && ~/projects/doc-ingest/.venv/bin/doc-ingest design "$f"; done
```

### Collections

| Qdrant collection | Content |
|-------------------|---------|
| `research` | arxiv papers + tech blogs |
| `design` | CLAUDE.md / ADR / docs/ |
| `notes` | curated Obsidian vault notes (private thinking) |
| `external-docs` | external API references |

### Distillation

LLM distillation pipeline: local **llama-server** (OpenAI-compatible `/v1/chat/completions`; set `LOCAL_LLM_URL`, optional `LOCAL_LLM_MODEL` / `LOCAL_LLM_TIMEOUT` / `LOCAL_LLM_MAX_TOKENS`) → **Anthropic Haiku** (if `ANTHROPIC_API_KEY` is set). If every backend fails, the chunk is skipped (never embedded as raw text). Private sources (notes, design docs, and local PDFs unless `pdf --allow-cloud` is given) never go to Anthropic. Point `LOCAL_LLM_URL` at a server with enough context (inputs are up to ~5000 chars); distillation runs serially by default (`DISTILL_WORKERS=1`).


---

## 日本語

外部 API ドキュメント・設計ドキュメント・arxiv 論文を [Qdrant](https://qdrant.tech/) ベクトル DB に投入するドキュメント取り込みパイプラインです。RAG / AI アシスタントワークフローに活用できます。

### モジュール

| モジュール | ソース | トリガー |
| --------- | ------ | ------- |
| `ingest/external.py` | 外部 API ドキュメント（GitHub, Anthropic, Qdrant…） | 週次 cron + ETag 差分 |
| `ingest/design_docs.py` | CLAUDE.md / ADR / docs/ | post-commit git フック |
| `ingest/notes.py` | Obsidian ノート（10-projects/20-areas/30-resources） | post-commit git フック |
| `ingest/arxiv.py` | arxiv 論文 / 技術ブログ | CLI 手動実行 |
| `ingest/pdf.py` | ローカル PDF ファイル / HTTPS の PDF URL | CLI 手動実行 |

### セットアップ

```bash
pip install -e ".[dev]"
bash scripts/install-precommit.sh  # クローンごとに 1 回必須（`pre-commit install` の代わり）。未実行だとコミット時に lint されない
cp .env.example .env  # QDRANT_URL, EMBED_URL, EMBED_API_KEY を設定
```

フックの結果は `~/.local/state/precommit-metrics/events.jsonl` に追記されます（`pass` / `fixed` / `fail`）。
スクリプトは `core.hooksPath` を `.git/managed-hooks`（全 worktree・全ブランチで共通）に設定します。設定後は `pre-commit install` が拒否されるため、記録用ラッパーを上書きされません。`pre-commit` 以外のフック（`pre-push` の秘密検査など）は `.git/hooks` の同名ファイルへ転送されます。更新するときは再実行します（何度実行しても同じ状態になり、旧方式の `.git/hooks/pre-commit` ラッパーは削除されます）。

### 使い方

```bash
# arxiv 論文を取り込む
doc-ingest arxiv https://arxiv.org/abs/2401.00123 --tags forge,fluxion

# 技術ブログ記事を取り込む
doc-ingest arxiv https://example.com/blog/post

# 外部 API ドキュメントを同期（ETag ベース）
doc-ingest sync
doc-ingest sync --force    # ETag チェックをスキップ
doc-ingest sync --dry-run  # 書き込まずプレビュー

# 設計ドキュメントを手動取り込み
doc-ingest design CLAUDE.md

# Obsidian ノートを手動取り込み（引数なし = 直近コミットで変わったファイル）
doc-ingest notes 10-projects/my-note.md

# 取り込み済みコンテンツを検索
doc-ingest search "Claude API streaming"
doc-ingest search "qdrant vector" --collection external-docs --limit 3
doc-ingest search "qdrant vector" --nugget             # チャンク全文ではなく BM25 で選んだ文を表示
doc-ingest search "qdrant vector" --nugget --nugget-k 5  # チャンクあたりの文数（既定 3）

# Qdrant のコレクション一覧（point 数と最終取り込み時刻）
doc-ingest list

# PDF を取り込む（ローカルファイルまたは HTTPS URL）
doc-ingest pdf ./paper.pdf --tags forge
doc-ingest pdf ./paper.pdf --allow-cloud  # ローカル LLM が失敗したとき Anthropic を許可（既定はローカルのみ）
doc-ingest pdf https://example.com/paper.pdf  # URL は公開ソース扱い（Anthropic へのフォールバックあり）
```

`design` と `notes` は、取り込みに失敗したファイルやチャンクが 1 件以上あると終了コード 1 になります。詳細は [docs/spec/cli.md](docs/spec/cli.md)。

### Git フック

post-commit フックを 1 つ以上のリポジトリにインストール:

```bash
# デフォルトリポ（mesh-drop / forge / fluxion）
bash scripts/install-hooks.sh

# 特定リポを指定
bash scripts/install-hooks.sh /path/to/repo

# ドライラン（書き込まず確認）
bash scripts/install-hooks.sh --dry-run

# フックスニペットを削除
bash scripts/install-hooks.sh --uninstall
```

コミットのたびに設計ドキュメント（CLAUDE.md / ADR / docs/）が自動取り込みされます。

フックはインストーラを実行したチェックアウトを指します（`DOC_INGEST_DIR` で上書き可）。コミットは失敗させませんが、`doc-ingest` が起動できない／非 0 で終了した場合は、時刻・リポ・終了コード・stderr 末尾 5 行を `${XDG_STATE_HOME:-~/.local/state}/doc-ingest/post-commit.log` に追記します。この変更より前に入れたフックは旧い固定パスのままなので、`--uninstall` のあと再インストールしてください。

Obsidian ノート（既定 `~/notes`）には `--notes` で、`notes` コマンドを実行する post-commit フックを入れます。キュレーションされたノートが `notes` コレクションに自動登録されます。`.env` の読み込みは design 用と同じ安全な方式（`KEY=VALUE` の行だけを文字列として読み、`source` しない）で、design 用と同じ `post-commit` ファイルに共存できます。

```bash
bash scripts/install-hooks.sh --notes               # ~/notes
bash scripts/install-hooks.sh --notes /path/to/vault
bash scripts/install-hooks.sh --notes --uninstall   # notes 用のブロックだけ削除
```

旧版の README の手順で作った手書きフック（`.env` を `source` するもの）が既にある場合、スクリプトは追記せず、先に削除するよう案内します。その手順は `.env` 内の任意のコマンドを実行してしまいます（#80、#221）。

### #216 以前の point からの移行（`design` / `notes`）

文書の識別子は `<リポ名>/<リポ内の相対パス>`（`file_path` に保存。`repo` も payload に追加）になりました。別リポの `CLAUDE.md` が衝突せず、post-commit フックと手動実行でも同じ ID になります。それ以前に取り込んだ point は `repo` が無く、孤立します。削除してから再取り込みしてください（フックは直近のコミットで変わったファイルしか拾いません）。

```bash
.venv/bin/python scripts/migrate_doc_keys.py            # dry-run: 件数とパスの例
.venv/bin/python scripts/migrate_doc_keys.py --apply    # `repo` の無い point を削除
# フック導入済みの各リポで設計 doc を全件再取り込み（そのリポの中で実行。パスは cwd 配下）
cd ~/projects/forge && for f in CLAUDE.md AGENTS.md docs/*.md adr/*.md ADR/*.md; do [ -f "$f" ] && ~/projects/doc-ingest/.venv/bin/doc-ingest design "$f"; done
```

### コレクション

| Qdrant コレクション | コンテンツ |
| ------------------- | --------- |
| `research` | arxiv 論文 + 技術ブログ |
| `design` | CLAUDE.md / ADR / docs/ |
| `notes` | Obsidian ノート（プライベートメモ） |
| `external-docs` | 外部 API リファレンス |

### 蒸留（Distillation）

LLM 蒸留パイプライン: ローカル **llama-server**（OpenAI 互換 `/v1/chat/completions`。`LOCAL_LLM_URL`、任意で `LOCAL_LLM_MODEL` / `LOCAL_LLM_TIMEOUT` / `LOCAL_LLM_MAX_TOKENS`）→ **Anthropic Haiku**（`ANTHROPIC_API_KEY` 設定時）。全て失敗したチャンクはスキップし、生テキストは埋め込まない。非公開ソース（notes・設計 doc、`pdf --allow-cloud` なしのローカル PDF）は Anthropic に送らない。入力は最大約 5000 字なので、`LOCAL_LLM_URL` は n_ctx に余裕のある台を指す。既定は直列実行（`DISTILL_WORKERS=1`）。


## Version / Documentation

Current version: 0.1.0 (not yet released; no tags). Changes are tracked in [CHANGELOG.md](CHANGELOG.md).
バージョン: 0.1.0（未リリース、タグなし）。変更履歴は [CHANGELOG.md](CHANGELOG.md)。

| Document | Content / 内容 |
|----------|----------------|
| [docs/spec/cli.md](docs/spec/cli.md) | CLI commands, environment variables, exit codes / コマンド・環境変数・終了コード |
| [docs/design.md](docs/design.md) | Architecture, data flow, constraints / 構成・データの流れ・制約 |
| [CHANGELOG.md](CHANGELOG.md) | Change history / 変更履歴 |
| [docs/adr/](docs/adr/README.md) | Architecture decision records / 設計判断 |

## License

MIT — see [NOTICE.md](NOTICE.md) for third-party copyleft dependency notes (trafilatura: GPL-3.0, PyMuPDF: AGPL-3.0).


## 設計判断 / Architecture decisions

[ADR 一覧](docs/adr/README.md) — 設計判断、根拠資料、未決定事項。
