# Contributing

Veld welcomes contributions. The canonical document is
[CONTRIBUTING.md](https://github.com/Portll/veld/blob/main/CONTRIBUTING.md)
in the repo; this page summarises the workflow and adds notes specific to
the docs sidecar.

## Quick start

```sh
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/veld
cd veld
git remote add upstream https://github.com/Portll/veld

# Stable Rust 1.70+ required
rustup default stable
cargo check --workspace
```

## Development discipline

- **No `cargo build` from inside an AI assistant session** — the developer
  keeps a build running in the background. Use `cargo check` and
  `cargo clippy` instead.
- **macOS:** prefix Cargo calls with `./scripts/cargo-dev.sh` so the
  libclang/RocksDB env shim is picked up.
- **Production-grade code only.** No TODOs, placeholders, mocks, or stubs
  in committed code. See [CLAUDE.md](https://github.com/Portll/veld/blob/main/CLAUDE.md)
  for the in-tree style notes.
- **Per-user locking:** any new write handler that mutates shared per-user
  state should follow the `CONSOLIDATION_LOCKS` pattern in
  [src/handlers/consolidation.rs](https://github.com/Portll/veld/blob/main/src/handlers/consolidation.rs).
- **Storage migrations:** when adding a `Memory` field, use
  `#[serde(default)]` so existing records round-trip.

## Documentation contributions

The docs sidecar lives in `docs/`. Two kinds of pages:

### Hand-authored pages

Architecture pages, guides, decision records — written by hand. Edit the
markdown directly under `docs/src/`.

### Generated pages

Reference pages (HTTP API, MCP tools, config, errors, metrics) are
auto-generated from veld source by the binaries in `docs/generators/`. If
you add a route, a tool, an env var, an error variant, or a metric, **run
`docs/regenerate.sh`** before committing. CI verifies generator output
matches what's committed.

A pre-commit hook is the simplest way to never forget:

```sh
ln -s ../../docs/regenerate.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

See [decision 0004](decisions/0004-docs-sidecar-stack.md) for the generator
architecture.

## Decision records

Architectural decisions land as ADR files in `docs/src/decisions/`. Format:

```markdown
---
id: NNNN
title: <short title>
status: accepted | superseded | deferred
date: YYYY-MM-DD
---

# NNNN — <title>

## Context
...

## Decision
...

## Consequences
...
```

`lint-decisions` (planned) will enforce frontmatter completeness and ID
continuity. Superseding a decision: write a new ADR with `supersedes: NNNN`
in the frontmatter and add `superseded-by: MMMM` to the original.

## Testing

```sh
cargo test --workspace          # unit + integration tests
cargo test --workspace --no-run # compile-only check
```

The `mcp-server/` TypeScript tests run with Bun:

```sh
cd mcp-server
bun test
```

## Submitting

- Open a pull request against `main`.
- Follow Conventional Commits for the title (`feat:`, `fix:`, `docs:`,
  `ci:`, `refactor:`, etc.).
- Pass CI (build + clippy + test + docs-drift check).
- Sign off if the project ever requires DCO (currently not).

## See also

- [CONTRIBUTING.md](https://github.com/Portll/veld/blob/main/CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](https://github.com/Portll/veld/blob/main/CODE_OF_CONDUCT.md)
- [Decisions index](decisions/index.md)
