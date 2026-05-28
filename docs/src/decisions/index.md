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
