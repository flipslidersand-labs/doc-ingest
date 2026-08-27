# doc-ingest

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## English

Document ingestion pipeline — feeds external API docs, design docs, and arxiv papers into [Qdrant](https://qdrant.tech/) vector DB for use in RAG / AI-assistant workflows.

### Modules

| Module | Source | Trigger |
|--------|--------|---------|
| `ingest/external.py` | External API docs (GitHub, Anthropic, Qdrant…) | Weekly cron + ETag diff |
| `ingest/design_docs.py` | CLAUDE.md / ADR / docs/ | post-commit git hook |
| `ingest/notes.py` | Obsidian vault (10-projects/20-areas/30-resources) | post-commit git hook |
| `ingest/arxiv.py` | arxiv papers / tech blogs | CLI manual |

### Setup

```bash
pip install -e .
cp .env.example .env  # fill in QDRANT_URL, EMBED_URL, EMBED_API_KEY
```

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

# Search ingested content
doc-ingest search "Claude API streaming"
doc-ingest search "qdrant vector" --collection external-docs --limit 3
```

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

For an Obsidian vault (e.g. `~/notes`), install a post-commit hook that runs the
`notes` command instead, so curated notes flow into the `notes` collection:

```bash
printf '#!/usr/bin/env bash\n_D="$HOME/projects/doc-ingest"\n[ -f "$_D/.env" ] && { set -a; . "$_D/.env"; set +a; }\nPYTHONPATH="$_D" "$_D/.venv/bin/doc-ingest" notes 2>/dev/null || true\n' \
  > ~/notes/.git/hooks/post-commit && chmod +x ~/notes/.git/hooks/post-commit
```

### Collections

| Qdrant collection | Content |
|-------------------|---------|
| `research` | arxiv papers + tech blogs |
| `design` | CLAUDE.md / ADR / docs/ |
| `notes` | curated Obsidian vault notes (private thinking) |
| `external-docs` | external API references |

### Distillation

LLM distillation pipeline: **Ollama** (`qwen2.5:7b`) → **Anthropic Haiku** (if `ANTHROPIC_API_KEY` is set) → raw text fallback.

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

### セットアップ

```bash
pip install -e .
cp .env.example .env  # QDRANT_URL, EMBED_URL, EMBED_API_KEY を設定
```

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

# 取り込み済みコンテンツを検索
doc-ingest search "Claude API streaming"
doc-ingest search "qdrant vector" --collection external-docs --limit 3
```

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

Obsidian ノート（例: `~/notes`）の場合は、post-commit フックで `notes` コマンドを実行するよう設定すると、キュレーションされたノートが `notes` コレクションに自動登録されます。

### コレクション

| Qdrant コレクション | コンテンツ |
| ------------------- | --------- |
| `research` | arxiv 論文 + 技術ブログ |
| `design` | CLAUDE.md / ADR / docs/ |
| `notes` | Obsidian ノート（プライベートメモ） |
| `external-docs` | 外部 API リファレンス |

### 蒸留（Distillation）

LLM 蒸留パイプライン: **Ollama**（`qwen2.5:7b`）→ **Anthropic Haiku**（`ANTHROPIC_API_KEY` 設定時）→ 生テキストフォールバック。

## License

MIT
