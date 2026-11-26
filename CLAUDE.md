# CLAUDE.md

Qdrant にドキュメントを投入する CLI(`cli.py`、`core/`、`ingest/`)。仕様は [docs/spec/cli.md](docs/spec/cli.md)、構成と制約は [docs/design.md](docs/design.md)、設計判断は [docs/adr/README.md](docs/adr/README.md)。

## コマンド

- テスト: `uv run --extra dev pytest -q`(CI は `--timeout=60`、カバレッジ下限 70%)
- lint: `uv run --extra dev ruff check .`
- format 確認: `uv run --extra dev ruff format --check .`(行長 100)
- ヘルプ: `uv run python cli.py --help`(パッケージ未インストールだと `uv run doc-ingest` は起動しない)

## このリポの罠

- pre-commit は `pre-commit install` ではなく `bash scripts/install-precommit.sh` で入れる(クローンごとに 1 回。`core.hooksPath` を使う)。
- `.env` を `source` しない。設定は `core/config.py` が `python-dotenv` で読む。フックスクリプトも `KEY=VALUE` を文字列として解析する。
- `design` / `notes` は取り込みに失敗したファイルが 1 件でもあると終了コード 1(蒸留スキップのチャンクは警告のみ)。`sync` は失敗ソースが 1 件でもあると 1。git フック側は `|| true`。
- 非公開ソース(design・notes・ローカル PDF)は Anthropic に送らない。この制約を壊す変更をしない。
- ruff のバージョンは `.pre-commit-config.yaml` と `.github/workflows/ci.yml` で揃える。
- 変更は PR で出し、merge-queue 経由で staging から main に入る(git log の `chore(queue)` と staging からのマージ)。

## 触らない

- `.env` と `.env.example`(秘密情報・内部設定。読まない、編集しない)。
- `sources/external.state.yaml`(実行状態。追跡対象外)。

リポ全体の規約ファイル(AGENTS.md 等)はこのリポに存在しない。NEEDS_CONFIRMATION: 規約の置き場所。
