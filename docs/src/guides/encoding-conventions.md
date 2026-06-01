<!-- GENERATED FILE — do not edit by hand.
     Source: CLAUDE.md (## Encoding Conventions)
     Generator: docs/generators/src/bin/gen-claude-sections.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-claude-sections`) -->

# Encoding Conventions

> Mirrored from CLAUDE.md. Edit the source there, not this file.

How to use the memory tools well — what to `remember`, what to leave to the automatic hook layer, and how to tag.

### When to call `remember` explicitly

The hook layer encodes most context automatically (Edit/Write/Bash/Read tool patterns, prompt submissions, session-start surfaces). Reach for explicit `remember(content, importance)` only when:

- A **decision** was made that future sessions need to honour (e.g., "we chose Redb over RocksDB because …" — see [decision 0001](docs/src/decisions/0001-redb-migration.md)).
- A **user preference** was stated (e.g., "always run cargo check, never cargo build").
- A **load-bearing invariant** was discovered (e.g., "the `public_router_has_no_per_user_handlers` test is the security contract").
- A **non-obvious cause-of-failure** was diagnosed and would otherwise be re-discovered.

Do not `remember` routine tool patterns, file paths, or anything visible from `git log`/`git blame`. The hook layer already covers those.

### Importance band

| Importance | Use for |
|---|---|
| `0.9–1.0` | Anchorable invariants (won't decay; consume sparingly — these compete for the top of recall) |
| `0.7–0.8` | Decisions, user preferences, security contracts |
| `0.4–0.6` | Diagnostic notes, useful-to-have context |
| ≤ `0.3` | Don't bother — leave it to the hook layer |

### Tag taxonomy

Tags are the cheap "what kind of thing" signal (scoring signal 15). Keep them short and reusable. Veld doesn't enforce a taxonomy, but converging on a small set makes tag-match scoring effective:

- **Domain**: `auth`, `storage`, `retrieval`, `consolidation`, `ingest`, `docs`, `tui`, `ci`, `release`
- **Kind**: `decision`, `bug`, `preference`, `invariant`, `gotcha`, `todo`
- **Status**: `wip`, `blocked`, `done` (mostly for todos)

A memory typically carries one domain + one kind tag (`["storage", "decision"]`); add more only if you'll search by them.

### Citation format

Reference code locations as `[path/to/file.rs:42](src/path/to/file.rs#L42)`. Reference decisions as `[0001-redb-migration.md](docs/src/decisions/0001-redb-migration.md)`. Reference HTTP routes as `` `POST /api/remember` `` (backticks, full path). The wiki-render layer (LLM-Wiki Phase 3) consumes this format directly; sticking to it now means no migration cost later.

---
