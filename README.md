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
| `ingest/pdf.py` | Local PDF files | CLI manual |

### Setup

```bash
pip install -e ".[dev]"
bash scripts/install-precommit.sh  # required once per clone (use this instead of `pre-commit install`)
cp .env.example .env  # fill in QDRANT_URL, EMBED_URL, EMBED_API_KEY
```

Hook results are appended to `~/.local/state/precommit-metrics/events.jsonl` (`pass` / `fixed` / `fail`).
Do not run `pre-commit install` afterwards; re-run `scripts/install-precommit.sh` instead (it is idempotent).

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

# Ingest a local PDF file
doc-ingest pdf ./paper.pdf --tags forge
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

LLM distillation pipeline: local **llama-server** (OpenAI-compatible `/v1/chat/completions`; set `LOCAL_LLM_URL`, optional `LOCAL_LLM_MODEL` / `LOCAL_LLM_TIMEOUT` / `LOCAL_LLM_MAX_TOKENS`) → **Anthropic Haiku** (if `ANTHROPIC_API_KEY` is set). If every backend fails, the chunk is skipped (never embedded as raw text). Private sources (notes, design docs) never go to Anthropic. Point `LOCAL_LLM_URL` at a server with enough context (inputs are up to ~5000 chars); distillation runs serially by default (`DISTILL_WORKERS=1`).


### Local ONNX embedding (experimental)

`EMBED_BACKEND=onnx_gpu` embeds locally with onnxruntime instead of the HTTP embedding service (`ONNX_MODEL_PATH` must point to a multilingual-e5-**base** ONNX export, 768-dim; the `onnx-gpu` extra installs the runtime). **Experimental:** it applies the same `passage: ` prefix, mean-pooling and L2 normalization as the HTTP service (covered by unit tests with a fake session), but it has **not been verified on a GPU**. On dev-nodee (GTX 1080, sm_61) onnxruntime 1.19.2 could not load the CUDA provider (`libcublasLt.so.12` missing) and silently fell back to CPU — a warning is now logged when `CUDAExecutionProvider` is not active. Default is `EMBED_BACKEND=http`; do not mix vectors from both backends in one collection unless they were produced with the same model and prefix.

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
| `ingest/pdf.py` | ローカル PDF ファイル | CLI 手動実行 |

### セットアップ

```bash
pip install -e ".[dev]"
bash scripts/install-precommit.sh  # クローンごとに 1 回必須（`pre-commit install` の代わり）。未実行だとコミット時に lint されない
cp .env.example .env  # QDRANT_URL, EMBED_URL, EMBED_API_KEY を設定
```

フックの結果は `~/.local/state/precommit-metrics/events.jsonl` に追記されます（`pass` / `fixed` / `fail`）。
以降に `pre-commit install` を実行しないでください（ラッパーが外れる）。やり直すときは `scripts/install-precommit.sh` を再実行します（何度実行しても同じ状態になります）。

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

# ローカル PDF ファイルを取り込む
doc-ingest pdf ./paper.pdf --tags forge
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

LLM 蒸留パイプライン: ローカル **llama-server**（OpenAI 互換 `/v1/chat/completions`。`LOCAL_LLM_URL`、任意で `LOCAL_LLM_MODEL` / `LOCAL_LLM_TIMEOUT` / `LOCAL_LLM_MAX_TOKENS`）→ **Anthropic Haiku**（`ANTHROPIC_API_KEY` 設定時）。全て失敗したチャンクはスキップし、生テキストは埋め込まない。非公開ソース（notes・設計 doc）は Anthropic に送らない。入力は最大約 5000 字なので、`LOCAL_LLM_URL` は n_ctx に余裕のある台を指す。既定は直列実行（`DISTILL_WORKERS=1`）。

### ローカル ONNX 埋め込み（実験的）

`EMBED_BACKEND=onnx_gpu` は、HTTP の埋め込みサービスの代わりに onnxruntime でローカル推論します（`ONNX_MODEL_PATH` に multilingual-e5-**base** の ONNX（768 次元）を指定。ランタイムは extra `onnx-gpu`）。**実験的**: HTTP サービスと同じ `passage: ` プレフィックス・mean pooling・L2 正規化を行い（fake セッションの単体テストで確認）、**GPU 実機では未検証**です。dev-nodee（GTX 1080, sm_61）では onnxruntime 1.19.2 が CUDA プロバイダを読み込めず（`libcublasLt.so.12` 不足）、黙って CPU に落ちました。`CUDAExecutionProvider` が有効でないときは警告を出します。既定は `EMBED_BACKEND=http`。同一 collection に、モデルとプレフィックスが異なるベクトルを混在させないでください。

## License

MIT — see [NOTICE.md](NOTICE.md) for a third-party (GPL-3.0) dependency note.
