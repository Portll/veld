# Ingest

The ingest pipeline accepts content from outside agents and turns it into
memories.

## Entry points

| Endpoint | Source |
|---|---|
| `POST /api/remember` | Single agent-driven `remember` call |
| `POST /api/remember/batch` | Bulk agent-driven `remember` |
| `POST /api/upsert` | Idempotent create-or-update by content hash |
| `POST /api/ingest` | Multi-format text extraction → bulk `remember` |
| `POST /api/seed` | Cold-start project seeding |
| `POST /webhook/linear` | Linear issue updates |
| `POST /webhook/github` | GitHub events |

## Extractors

[`src/ingest/extractors.rs`](https://github.com/Portll/veld/blob/main/src/ingest/extractors.rs)
implements format-aware extraction:

| Format | Status |
|---|---|
| Plain text | Built-in |
| Markdown | Built-in (with frontmatter parsing) |
| PDF | Behind `pdf` feature flag (`pdf-extract`) |
| HTML | Built-in (with boilerplate removal) |
| Source code | Built-in (chunked by syntax) |

## Remote ingestors

- [`src/ingest/github.rs`](https://github.com/Portll/veld/blob/main/src/ingest/github.rs)
  — fetch repository contents, optionally chunked per file.
- [`src/ingest/gdrive.rs`](https://github.com/Portll/veld/blob/main/src/ingest/gdrive.rs)
  — Google Drive document ingestion.

## Project seeding

`POST /api/seed` ([src/handlers/seed.rs](https://github.com/Portll/veld/blob/main/src/handlers/seed.rs))
ingests a project's contents in bulk. Useful for cold-starting an agent on
an existing codebase — it walks the project tree, extracts source/docs/configs,
and stores them with project-aware facets.

## Webhooks

Linear and GitHub webhooks are authenticated and rate-limited. Inbound
events are translated into memories with appropriate facets (`Who` =
issue reporter, `Where` = repo/project, `When` = event timestamp).

## Atomicity (planned)

Currently, `POST /api/ingest` writes memories one at a time. The
[LLM-Wiki plan](https://github.com/Portll/veld/blob/main/CLAUDE.md) Phase 2
introduces an atomicity envelope: one ingest call = one transactional
batch, with per-user lock and Redb transactional-batch semantics. Partial
failure leaves no half-state.

## See also

- [Consolidation](consolidation.md) — what happens after ingest
- [Memory tiers](memory-tiers.md) — new memories start in Working
