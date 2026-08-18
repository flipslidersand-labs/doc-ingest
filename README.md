# doc-ingest

Document ingestion pipeline — feeds external API docs, design docs, and arxiv papers into Qdrant.

## Modules

| module | source | trigger |
|--------|--------|---------|
| `ingest/external.py` | External API docs (GitHub, Anthropic, Qdrant…) | Weekly cron + ETag diff |
| `ingest/design_docs.py` | CLAUDE.md / ADR / docs/ | post-commit git hook |
| `ingest/notes.py` | Obsidian vault (10-projects/20-areas/30-resources) | post-commit git hook |
| `ingest/arxiv.py` | arxiv papers / tech blogs | CLI manual |

## Setup

```bash
pip install -e .
cp .env.example .env  # fill in QDRANT_URL, EMBED_URL, EMBED_API_KEY
```

## Usage

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

## Git Hook

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

## Collections

| Qdrant collection | content |
|-------------------|---------|
| `research` | arxiv papers + tech blogs |
| `design` | CLAUDE.md / ADR / docs/ |
| `notes` | curated Obsidian vault notes (private thinking) |
| `external-docs` | external API references |

## Distillation

LLM distillation runs as: **Ollama** (`qwen2.5:7b`) → **Anthropic Haiku** (if `ANTHROPIC_API_KEY` set) → raw text fallback.
