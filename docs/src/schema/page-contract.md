<!-- This page will be auto-mirrored from CLAUDE.md's `## Page Contract`
     section once LLM-Wiki Phase 1 lands. See decision 0002 and decision 0004. -->

# Page contract

The page contract specifies what fields every rendered LLM-Wiki page must
carry. It is the schema for the *output* of the wiki render module.

> **Status:** this contract is being defined as part of LLM-Wiki Phase 1.
> Until that work lands, this page documents the intended fields. After
> Phase 1 ships, this page is auto-generated from the `## Page Contract`
> section in `CLAUDE.md` by the `gen-claude-sections` generator.

## Required frontmatter

```yaml
---
id: <slug>-<uuid>            # Stable across renames
page_type: entity | concept | source | comparison | synthesis | log
schema_version: <integer>    # Which schema version this page was rendered under
cited_sources:               # Provenance — every page claims its sources
  - slug: <source-slug>
    section: <optional-section-anchor>
last_rendered_at: <ISO 8601 timestamp>
last_rendered_hash: <sha256 of page body at render time>
is_derived: <bool>           # True for synthesis pages, false for raw observations
sources_tier:                # When is_derived is true
  - primary | secondary | opinion | derived-elsewhere
---
```

## Body conventions

- All claims carry inline `[src:slug#section]` citations.
- Wikilinks use `[[other-page]]` syntax (Obsidian-compatible).
- Truncation: if the LLM truncates body to fit a budget, it emits a
  `[TRUNCATED]` marker — lint surfaces truncated pages for follow-up.
- Page type drives section layout (entity pages have `## Identity`,
  `## Properties`, `## Related`; concept pages have `## Definition`,
  `## Examples`, `## Counterexamples`; etc.).

## See also

- [Schema changelog](changelog.md)
- [Migrations](migrations.md)
- [Decision 0002 — LLM wiki dual-pathway](../decisions/0002-llm-wiki-dual-pathway.md)
