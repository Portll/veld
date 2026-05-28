---
id: 0004
title: Docs sidecar built on mdBook + Rust generators
status: accepted
date: 2026-05-28
---

# 0004 — Docs sidecar built on mdBook + Rust generators

## Context

Veld needs a developer-facing documentation site that:

1. Documents the HTTP API, MCP tools, configuration, errors, metrics, and
   architecture for integrators, contributors, and curious users.
2. **Auto-updates from source** as the codebase changes — new endpoints, new
   tools, new config keys should appear in the docs without anyone hand-writing
   them.
3. Does not duplicate or compete with rustdoc (the Rust API reference).
4. Does not conflate with the LLM Wiki (see [0002](0002-llm-wiki-dual-pathway.md)) —
   the wiki renders per-user memory contents; this docs site documents the
   project itself.

## Stack evaluated

| Stack | Verdict |
|---|---|
| **mdBook** | **Selected.** Rust-native, single binary, fast, built-in lunr search, mature (powers The Rust Book, The Cargo Book). Preprocessors are Rust binaries on stdin/stdout — trivial integration with generators. |
| Docusaurus | Rejected — Node + React + 100s of deps; overkill; doesn't match veld's "single binary, runs offline" identity. |
| VitePress | Rejected — adds Vue/Vite dependency to an otherwise pure-Rust project. |
| Astro | Rejected — JS-heavy; content-collections add complexity we don't need. |
| Zola | Rejected — solid SSG but mdBook has stronger docs-specific conventions and ecosystem fit. |
| Hugo | Rejected — Go runtime introduces a third toolchain. |
| Rustdoc alone | Rejected as standalone — doesn't cover HTTP/MCP API, config, guides. We still need mdBook on top. Rustdoc is built separately and stitched into `<docs-site>/api/`. |

## Decision

**Use mdBook for the docs shell, with ~5 small Rust generator binaries under
`docs/generators/` that read veld source files and emit markdown into
`docs/src/reference/`.**

Generators implemented in this commit:

| Generator | Input | Output |
|---|---|---|
| `gen-http-api` | `src/handlers/router.rs` (parsed via `syn`) | `docs/src/reference/http-api.md` |
| `gen-mcp-tools` | `mcp-server/index.ts` (regex over `name:`/`description:`) | `docs/src/reference/mcp-tools.md` |
| `gen-config-ref` | `src/**/*.rs` (`env::var("VELD_*")` walk) | `docs/src/reference/config.md` |
| `gen-errors` | `src/errors.rs` (parsed via `syn`) | `docs/src/reference/errors.md` |
| `gen-metrics` | `src/**/*.rs` (LazyLock metric declarations) | `docs/src/reference/metrics.md` |

Future generators on the roadmap (not implemented yet):

- `gen-cli-ref` — runtime `veld --help` invocation
- `gen-module-index` — `src/lib.rs` `pub mod` walk
- `gen-schema-changelog` — parse `## Schema Version & CHANGELOG` from CLAUDE.md
  (lands when LLM-Wiki Phase 1 ships)
- `gen-claude-sections` — mirror named CLAUDE.md sections (`## Epistemic
  Hygiene`, `## Page Contract`, etc.)
- `gen-changelog` — git-tag-based release notes
- `lint-decisions` — ADR frontmatter / ID continuity check

## Consequences

- **Pro:** the docs site stays in sync with source by construction. A
  developer adding a route to `router.rs` runs `regenerate.sh` (or the
  pre-commit hook does it for them); the docs update in the same commit.
- **Pro:** every generator is fast (<1s); the full sweep finishes in seconds.
- **Pro:** consistent toolchain — only Rust, no Node or Python in the docs
  pipeline.
- **Pro:** generator output is deterministic — CI verifies `git diff
  --exit-code docs/src/` after running all generators, catching drift.
- **Con:** when a generator misses an edge case (a non-literal route path,
  for example), it emits a `<!-- WARNING -->` comment and CI fails loudly.
  This is by design — silent miss is worse than loud failure.
- **Con:** generators must be maintained alongside their source files. A
  major restructure of `router.rs` requires updating `gen-http-api`. The
  generators are small (each <200 lines), so this is acceptable.

## Deployment

- **Build:** `cd docs && mdbook build` produces `docs/book/` as static HTML.
- **Deploy:** GitHub Actions builds on push to `main` and deploys to
  `gh-pages` branch. Site lives at `portll.github.io/veld/` initially.
- **Versioning:** mdBook has no native versioning. CI on tag push copies the
  build output to `docs/book/v{tag}/` subdir; main `gh-pages` deploy serves
  `latest` at the root. Custom domain `docs.veld.dev` optional later.

## Operational details

- Pre-commit hook (`docs/regenerate.sh`) runs the cheap generators on every
  commit and stages generated files.
- CI runs the full generator set (including `gen-cli-ref` which needs a
  build) and verifies no drift.
- The `docs/generators/` workspace is **standalone** — it declares its own
  `[workspace]` so it does not interfere with veld's root workspace.

## Related

- Full plan: `~/.claude/plans/veld-docs-sidecar-plan.md`.
- LLM Wiki: [0002](0002-llm-wiki-dual-pathway.md) (independent system; docs
  sidecar mirrors selected CLAUDE.md sections that the wiki plan introduces).
