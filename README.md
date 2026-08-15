# doc-ingest

Document ingestion pipeline — feeds external API docs, design docs, and arxiv papers into Qdrant.

## Modules

| module | source | trigger |
|--------|--------|---------|
| `ingest/external.py` | External API docs (GitHub, Anthropic, GCP…) | Weekly cron + ETag diff |
| `ingest/design_docs.py` | CLAUDE.md / ADR / docs/ | post-commit git hook |
| `ingest/arxiv.py` | arxiv papers / tech blogs | CLI manual |

## Setup

```bash
pip install -e .
cp .env.example .env  # set QDRANT_URL, EMBED_URL, EMBED_API_KEY, ANTHROPIC_API_KEY
```

## Usage

```bash
# Ingest an arxiv paper
doc-ingest arxiv https://arxiv.org/abs/2401.00123 --tags forge,fluxion

# Ingest a tech blog post
doc-ingest arxiv https://example.com/blog/post

# Sync external API docs (ETag-based)
doc-ingest sync
doc-ingest sync --force  # skip ETag check

# Ingest a specific design doc manually
doc-ingest design CLAUDE.md
```

## Git Hook

```bash
cp hooks/post-commit .git/hooks/post-commit && chmod +x .git/hooks/post-commit
```

Design docs are auto-ingested on every commit.

## Collections

| Qdrant collection | content |
|-------------------|---------|
| `research` | arxiv papers + tech blogs |
| `design` | CLAUDE.md / ADR / docs/ |
| `external-docs` | external API references |
