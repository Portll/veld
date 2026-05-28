# Veld — Agentic Memory

Veld is an **edge-native cognitive memory system** for AI agents. It runs as a single Rust binary, stores everything locally, and requires no external databases, no API keys, and no cloud services.

## What it does

AI agents like Claude Code and VS Code Copilot live in isolated sessions. Without persistent memory, every session starts from scratch — the agent cannot recall prior decisions, cannot build on past context, and cannot learn your patterns over time.

Veld closes this gap by:

- **Storing** memories across sessions with semantic embeddings, knowledge-graph edges, and structured facets.
- **Retrieving** relevant past context automatically at the start of each session (and on demand during a session).
- **Decaying** memories over time in a psychologically realistic way — frequently accessed and important memories persist; peripheral details fade.
- **Consolidating** memories in the background — extracting facts, strengthening graph edges, merging near-duplicates.

## Who it's for

- **Developers** who work daily with AI coding assistants and want the assistant to remember architectural decisions, preferred patterns, and ongoing work across sessions.
- **AI engineers** building agents that need persistent episodic memory without a cloud dependency.
- **Robotics / embedded** deployments (via the Zenoh transport) where an edge node needs local memory that synchronises with a collective store.

## How it integrates

Veld exposes an HTTP API and two MCP transports:

| Client | Integration point |
|---|---|
| Claude Code | `.claude/settings.json` + automatic memory hooks |
| VS Code Copilot | `.vscode/mcp.json` — registers 46 MCP tools |
| Cursor | MCP stdio server (`veld serve --mcp`) |
| Any HTTP client | REST API at `http://127.0.0.1:3030/api/` |
| Python | `pip install veld` (PyO3 bindings) |

## Licence and distribution

Veld is licensed under the **Business Source License 1.1 (BUSL-1.1)**. The production-use restrictions convert to Apache-2.0 on the change date specified in the licence file.

| Surface | Identifier |
|---|---|
| Rust crate | `veld` on crates.io |
| Node / MCP | `@veld/memory-mcp` on npm |
| Python | `veld` on PyPI |
| Docker | `varunveld/veld` |
| MCP Registry | `veld` |

## Project status

```admonish warning
The current branch is `v0.7.6-unstable`, being stabilized toward a clean
`v0.8` release. Treat the branch tip as internal/unstable unless a tagged
release says otherwise. The next public-release-quality target is `v0.9`
([decision 0001](decisions/0001-redb-migration.md)).
```

## What this site documents

- **[Quickstart](quickstart.md)** — install + run + store the first memory.
- **[Architecture](architecture/overview.md)** — how the retrieval pipeline,
  storage, consolidation, and intent log fit together.
- **[Reference](reference/http-api.md)** — every HTTP route, every MCP tool,
  every config variable, every error, every metric — auto-generated from
  source.
- **[Guides](guides/claude-code-integration.md)** — client integration,
  deployment, retrieval tuning, multi-tenant operation.
- **[Decisions](decisions/index.md)** — architectural decision records.
- **[Schema](schema/changelog.md)** — record-shape evolution, migrations,
  page contract.

## Next step

→ [Quickstart](quickstart.md) — get veld running and store your first memory in 60 seconds.
