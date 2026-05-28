---
id: 0003
title: Epistemic hygiene lives in CLAUDE.md as a named section
status: accepted
date: 2026-05-28
---

# 0003 — Epistemic hygiene lives in CLAUDE.md as a named section

## Context

The LLM Wiki pattern (see [0002](0002-llm-wiki-dual-pathway.md)) introduces a
discipline we call **epistemic hygiene** — the rules that prevent confidence
inflation, citation cycles, and model autophagy:

- Source-tier labels (`primary | secondary | opinion | derived-elsewhere`)
- Derived-memory flag (filed-back answers don't re-enter as sources)
- Staleness propagation (derived pages mark stale when their consulted sources
  change)
- Provenance citations (`[src:slug#section]` form, hash-linked)
- Citation-aggregation lint (single-primary amplification detection)

The question: should these rules live **inside** CLAUDE.md, or in a **separate**
`trust-framework.md` that CLAUDE.md imports?

## Options evaluated

| Option | Single-source-of-truth | Reusable | Sync risk | LLM context cost |
|---|---|---|---|---|
| Embedded (one CLAUDE.md) | ✓ | ✗ | None | + ~120 lines |
| Separate-imported (`trust-framework.md vN`) | ✗ | ✓ | High | + ~120 lines split |
| Hybrid (named section with stable anchors) | ✓ | ~ (via anchor-quote / fork) | None | + ~120 lines named |

A Bifocal+ evaluation across 12 dimensions /120 produced:

- Embedded: **84.0**
- Separate-imported: **75.5**
- Hybrid: **86.1** ← winner

The Hybrid wins on Coherence, Robustness, and Ease-of-use without sacrificing
Elasticity. Separate-imported's only meaningful advantage — reusability across
multiple wikis — relies on J6 (cross-wiki citations), which was deferred in
the source evaluation chain. With J6 deferred, the reusability case is
speculative.

## Decision

**Add a named `## Epistemic Hygiene` section to CLAUDE.md with stable subsection
anchors.** Do not create a separate `trust-framework.md` doc.

Stable anchors that lint output and external references can quote:

- `### Source Tiers`
- `### Derived Flag`
- `### Provenance Citations`
- `### Citation Aggregation`
- `### Staleness Propagation`
- `### Anti-Autophagy`

Anchor renames are breaking changes requiring a schema-version bump. The
anchor names are an API.

## Consequences

- **Pro:** single doc that an LLM session reads as one block — no two-doc
  coordination dance.
- **Pro:** anchor stability provides the reusability benefit (external tools
  can reference `CLAUDE.md#epistemic-hygiene-source-tiers`) without two-doc
  sync risk.
- **Pro:** the docs sidecar mirrors the section as
  `docs/src/architecture/epistemic-hygiene.md` via a deterministic generator
  ([0004](0004-docs-sidecar-stack.md)). External readers get a clean docs page;
  CLAUDE.md remains the authoritative source.
- **Con:** if epistemic hygiene grows beyond ~150 lines, CLAUDE.md becomes
  unwieldy. Mitigation: the K1 schema-of-schema split is on standby once
  CLAUDE.md exceeds ~450 lines total.
- **Con:** anchor stability requires discipline. A casual section rename will
  break lint output and external references silently. The schema-version field
  catches this on the *page* side but not on the *anchor* side; we mitigate by
  treating CLAUDE.md anchor renames as breaking changes in code review.

## Implementation

The section is added to CLAUDE.md as part of LLM-Wiki Phase 5
(see [0002](0002-llm-wiki-dual-pathway.md) and the full plan at
`~/.claude/plans/veld-llm-wiki-dual-pathway-plan.md`).

The docs sidecar's `gen-claude-sections` generator (see
[0004](0004-docs-sidecar-stack.md)) copies the named section into the docs
site automatically.

## Related

- [0002](0002-llm-wiki-dual-pathway.md) — wiki architecture this hygiene
  discipline serves.
- [0004](0004-docs-sidecar-stack.md) — how the section gets mirrored to the
  public docs.
