# Veld docs sidecar

Developer-facing documentation site for Veld — Agentic Memory. Built on
[mdBook](https://rust-lang.github.io/mdBook/). Sources live in `docs/src/`.

The static site is published to GitHub Pages on every push to `main`. The
canonical URL once the first deploy lands: `https://portll.github.io/veld/`.

## Architecture decision

See [docs/src/decisions/0004-docs-sidecar-stack.md](src/decisions/0004-docs-sidecar-stack.md)
for the why-mdBook-not-Docusaurus reasoning and the generator design.

## Layout

```
docs/
├── book.toml                # mdBook config (theme, preprocessors)
├── theme/custom.css         # theme overrides
├── src/                     # markdown sources
│   ├── SUMMARY.md           # site navigation
│   ├── intro.md, quickstart.md, faq.md
│   ├── architecture/        # hand-authored deep-dives + module-index (generated)
│   ├── reference/           # ALL GENERATED — http-api, mcp-tools, config, errors, metrics
│   ├── guides/              # client integration, deploy, multi-tenant, etc.
│   ├── schema/              # changelog, migrations, page-contract (some generated)
│   ├── decisions/           # ADRs (hand-authored; lint-decisions enforces frontmatter)
│   ├── security.md, contributing.md, benchmarks.md
│   ├── glossary.md, changelog.md
├── generators/              # Rust workspace with 8 generator binaries
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # shared helpers
│       └── bin/
│           ├── gen-http-api.rs        # src/handlers/router.rs → reference/http-api.md
│           ├── gen-mcp-tools.rs       # mcp-server/index.ts → reference/mcp-tools.md
│           ├── gen-config-ref.rs      # src/**/env::var → reference/config.md
│           ├── gen-errors.rs          # src/errors.rs → reference/errors.md
│           ├── gen-metrics.rs         # src/**/LazyLock metrics → reference/metrics.md
│           ├── gen-module-index.rs    # src/lib.rs pub mod → architecture/module-index.md
│           ├── gen-claude-sections.rs # CLAUDE.md sections → mirror pages
│           └── lint-decisions.rs      # ADR frontmatter + ID continuity
├── regenerate.sh            # runs all generators
├── install-pre-commit.sh    # installs the docs pre-commit hook
└── README.md                # this file
```

## Working on the docs

### Build + preview

```sh
# Install mdBook and preprocessors (once per machine)
cargo install --locked mdbook mdbook-mermaid mdbook-toc mdbook-linkcheck mdbook-admonish

# Preview locally
cd docs && mdbook serve --open
```

### Regenerate auto-generated pages

```sh
cd docs && bash regenerate.sh
```

You don't need to do this manually if you've installed the pre-commit hook
(see below) — it runs automatically on every commit.

### Install the pre-commit hook (recommended)

```sh
bash docs/install-pre-commit.sh
```

After this, every `git commit` re-runs generators and stages updated docs.

### What's generated vs hand-authored

| Path | Source of truth |
|---|---|
| `src/reference/http-api.md` | Generated from `src/handlers/router.rs` |
| `src/reference/mcp-tools.md` | Generated from `mcp-server/index.ts` |
| `src/reference/config.md` | Generated from `src/**/*.rs` (env::var calls) |
| `src/reference/errors.md` | Generated from `src/errors.rs` |
| `src/reference/metrics.md` | Generated from `src/**/*.rs` (LazyLock metric decls) |
| `src/architecture/module-index.md` | Generated from `src/lib.rs` |
| `src/architecture/epistemic-hygiene.md` | Mirrored from CLAUDE.md `## Epistemic Hygiene` (when present) |
| `src/schema/page-contract.md` | Mirrored from CLAUDE.md `## Page Contract` (when present) |
| `src/guides/{encoding-conventions,scale-and-migration,tool-coupling}.md` | Mirrored from CLAUDE.md sections (when present) |
| everything else | Hand-authored |

If you change one of the "source of truth" files above, run `regenerate.sh`.
CI verifies generators produce committed output (`git diff --exit-code
docs/src/`).

## CI

`.github/workflows/docs.yml`:

1. Run all generators (including `gen-cli-ref` which needs a build — not
   implemented yet).
2. Verify `git diff --exit-code docs/src/` (drift detection).
3. Build mdBook.
4. Build rustdoc (`cargo doc --no-deps`).
5. Stitch rustdoc into `docs/book/api/`.
6. Deploy `docs/book/` to `gh-pages` on push to main; upload as artifact on PRs.

## Adding a new decision (ADR)

```sh
# Get the next number
ls docs/src/decisions/ | grep '^[0-9]' | sort | tail -1

# Create the file
cat > docs/src/decisions/NNNN-your-title.md <<'EOF'
---
id: NNNN
title: Your title
status: accepted
date: 2026-MM-DD
---

# NNNN — Your title

## Context
...

## Decision
...

## Consequences
...
EOF

# Add to the index
$EDITOR docs/src/decisions/index.md
```

`lint-decisions` enforces ID continuity and frontmatter completeness.

## Adding a new architecture page

Hand-author it in `docs/src/architecture/`, add to `docs/src/SUMMARY.md`,
cross-link from related pages, regenerate-and-rebuild.

## Common gotchas

- **mdbook-linkcheck** fails CI on broken internal links. Use relative paths
  (`../decisions/0001-redb-migration.md`) not absolute (`/decisions/...`).
- **Mermaid diagrams** must be inside ` ```mermaid ` code blocks. The
  `mdbook-mermaid` preprocessor handles them.
- **Admonitions** use `mdbook-admonish` syntax: ` ```admonish warning `,
  ` ```admonish tip `, ` ```admonish note `.
- **Generated headers** must stay at the top of generated files. Don't
  hand-edit generated files — your changes will be lost on the next
  regenerate. Edit the source instead.

## See also

- [Decision 0004 — Docs sidecar stack](src/decisions/0004-docs-sidecar-stack.md)
- [Contributing](src/contributing.md)
- Repo-level [CONTRIBUTING.md](../CONTRIBUTING.md)
