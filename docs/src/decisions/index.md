---
id: index
status: living
date: 2026-05-28
---

# Architecture Decision Records

Each decision below is a single, dated record explaining *why* a particular
architectural choice was made. The format follows the spirit of Michael Nygard's
ADRs: context, decision, consequences. We add status (`accepted`, `superseded`,
`deferred`) and a chronological ID.

If you supersede a decision, do not delete the old record — write a new ADR
pointing back at it with `supersedes:` in the frontmatter and add
`superseded-by:` to the original. The historical record matters.

| ID | Title | Status | Date |
|---|---|---|---|
| [0001](0001-redb-migration.md) | Redb migration target | accepted | 2026-05-28 |
| [0002](0002-llm-wiki-dual-pathway.md) | LLM wiki as rendered projection over veld | accepted | 2026-05-28 |
| [0003](0003-epistemic-hygiene-location.md) | Epistemic hygiene lives in CLAUDE.md as a named section | accepted | 2026-05-28 |
| [0004](0004-docs-sidecar-stack.md) | Docs sidecar built on mdBook + Rust generators | accepted | 2026-05-28 |
| [0005](0005-facts-purge-and-narratives.md) | Facts purge, narratives, and the `purged_at` field | accepted | 2026-05-29 |
| [0006](0006-backend-parity-and-on-demand-migration.md) | Backend parity and on-demand migration | proposed | 2026-09-11 |
| [0007](0007-rocksdb-default-engine.md) | RocksDB is the default engine and the port target | accepted | 2026-10-03 |
| [0008](0008-local-ipc-transport.md) | Whether veld takes upstream's authenticated local IPC transport | proposed | 2026-10-04 |
| [0009](0009-index-text-derived-not-stored.md) | Index text is derived from content, never stored as content | accepted | 2026-10-19 |
