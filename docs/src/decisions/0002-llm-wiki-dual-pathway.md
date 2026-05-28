---
id: 0002
title: LLM wiki as rendered projection over veld
status: accepted
date: 2026-05-28
---

# 0002 — LLM wiki as rendered projection over veld

## Context

The [LLM Wiki pattern](https://kasari.io/llm-wiki/) (published 2026) proposes
that, instead of pure RAG retrieval from immutable sources, the LLM
incrementally builds and maintains a persistent, interlinked markdown wiki
between the user and their raw sources. The wiki is a fixed-point that drifts
forward — every ingest folds a new source into the existing structure. The user
browses the wiki in Obsidian; the LLM owns the writing.

Veld is veld's most coherent intellectual rival to this idea. Veld bets on
**invisible substrate + retrieval + decay**. LLM Wiki bets on **visible
artifact + compilation + lint**. The user wants both — to keep veld's HNSW,
Hebbian, calibrated-confidence retrieval, *and* to gain the audit/trust benefit
of a browsable markdown layer.

Four architectures were evaluated:

- **Option A — Wiki canonical, veld derived:** throws away veld's HNSW /
  Hebbian / decay.
- **Option B — Veld canonical, wiki as rendered projection:** memory records
  are canonical; the wiki is a derived markdown view. Edit-back through a
  watched filesystem.
- **Option C — Dual independent writes:** two-phase commit, version vectors.
- **Option D — Single canonical RocksDB/Redb store, two views:** functionally
  identical to B.

## Decision

**Adopt Option B/D (collapsed):** veld remains the canonical store. A new
`src/wiki/` module renders the current per-user state into a configurable
markdown directory (`~/veld-wiki/<user-id>/`). The render is event-driven and
incremental — when a memory changes, the affected pages re-render through a
per-user queue.

The wiki is **never canonical**. Direct edits in Obsidian feed back into veld
through a file watcher + hash check + REVERSE-conflict detection (page hash
stored alongside the memory record).

## Consequences

- **Pro:** preserves veld's retrieval strengths entirely.
- **Pro:** time-coherence is a single-source-of-truth problem, not a
  distributed-systems one. The `intent_log/` module already provides an
  append-only journal that projects bit-for-bit into the wiki's `log.md`.
- **Pro:** existing veld primitives map cleanly:
  - `RecordKind` + `ExperienceType` → page-type taxonomy
  - `RelationshipEdge` → wikilinks
  - `ContextBlock` (Letta-style mutable state) → synthesis pages
  - `MemoryTier::Archive` → archive-tier pages
  - Per-user `CONSOLIDATION_LOCKS` → render serialization
- **Con:** introduces a new write path (Obsidian → veld) that did not exist
  before. The REVERSE-conflict surface is novel and needs explicit testing.
- **Con:** materialization cost grows with wiki size. Incremental render plus
  the archive tier keep this bounded.

## Foundations the wiki layer requires

The full implementation plan is in
`~/.claude/plans/veld-llm-wiki-dual-pathway-plan.md`. Critical foundations
(landed in Phase 1):

- `cited_sources` field on `Memory` — slug-form provenance citations.
- `is_derived: bool` on `Memory` — distinguishes raw observations from
  LLM-synthesized summaries. Anti-autophagy guard.
- `page_type: PageType` derived from `RecordKind` + tags.
- `schema_version: u32` on `Memory` — records which schema version
  authored the record.

Without provenance + derived flag, the wiki's epistemic hygiene story (see
[0003](0003-epistemic-hygiene-location.md)) cannot land.

## Time-coherence guarantee

At any time *T*, the wiki reflects veld's state at some *T'* ≤ *T*. The wiki
never displays a state veld never held. Bound: render lag p99 < 2 seconds under
no load; tunable via `VELD_WIKI_RENDER_LAG_MS`.

## Out of scope (deferred)

- **Cross-wiki citations** (J6 in the source evaluation chain): deferred until
  the user maintains multiple wikis.
- **Multi-user wiki collaboration:** veld's per-user lock model doesn't cover
  concurrent writers from different users; deferred.

## Related

- [0003](0003-epistemic-hygiene-location.md) — epistemic hygiene placement.
- `~/.claude/plans/veld-llm-wiki-dual-pathway-plan.md` — full phased plan.
