# FAQ

Common questions about Veld — Agentic Memory.

## Why agentic memory?

Most LLM agents are amnesiac. Every session starts from scratch. They
re-derive the same context, repeat the same mistakes, and never accumulate
domain knowledge. Veld is a substrate that lets agents *remember* in a
psychologically realistic way — important things persist, irrelevant things
fade, related memories surface together.

## Why not just RAG over notes?

RAG retrieves from immutable chunks at query time. Veld actively
*consolidates* what it stores — extracts facts, strengthens edges between
co-retrieved memories, decays unused memories, calibrates confidence. The
substrate gets smarter; RAG corpora don't.

See [decision 0002](decisions/0002-llm-wiki-dual-pathway.md) for how veld
relates to the [LLM Wiki pattern](https://kasari.io/llm-wiki/) — they're
complementary.

## Does veld send my data anywhere?

No. Veld is a single binary that runs entirely offline. It stores data
locally and uses local embedders (ONNX MiniLM by default, or any
OpenAI-compatible HTTP endpoint — LM Studio, Ollama, vLLM). No telemetry,
no API keys, no cloud round-trips.

The `telemetry` feature flag (off by default) enables OpenTelemetry
exporters for ops teams who want distributed tracing.

## What's the storage backend?

Today: **RocksDB**. Tomorrow (v0.9): **Redb**. The trait abstraction
(`PrimaryMemoryStore`, `GraphStore`, `KeyValueStore`) is already in place;
the runtime cutover happens once Redb passes its acceptance benchmarks.

See [decision 0001](decisions/0001-redb-migration.md) and
[Storage](architecture/storage.md) for details.

## Why two MCP servers (Rust + TypeScript)?

Both exist; choose by deployment preference:

- **TypeScript** (`@veld/memory-mcp`) is published on npm. It wraps the
  HTTP API and is the path Claude Code / VS Code Copilot / Cursor use by
  default. Easy to install (`npm install -g`), works on any machine with
  Node.
- **Rust** (`veld serve`) speaks the same MCP protocol via `rmcp` directly.
  Single binary, no Node required. Use when you don't want a Node toolchain
  on the host.

Both expose the same 46 tools. Choose whichever fits your environment.

## Can I run multiple agents against the same memory?

Yes. Veld's `user_id` parameter scopes most operations. Default IDs:

| Client | Default user ID |
|---|---|
| Claude Code | `claude-code` |
| Claude CLI | `claude-code` (shared) |
| VS Code Copilot | `vscode-copilot` |
| Cursor | `cursor` |

To share memory across clients, set the same `user_id`. To isolate, use
different ones. For true multi-tenant operation, build with the
`multi-tenant` feature flag (see [Multi-tenant](guides/multi-tenant.md)).

## How do I bring my own embedder?

Veld talks to any OpenAI-compatible HTTP embeddings endpoint. Configure
via `VELD_EMBED_URL`:

```sh
export VELD_EMBED_URL=http://127.0.0.1:11434/v1   # Ollama
veld server
```

LM Studio, Ollama, vLLM, and OpenAI-API-compatible servers all work out of
the box. Veld auto-detects the backend.

For embedded ONNX (no HTTP backend), `veld init` downloads MiniLM-L6-v2 to
`~/.veld/models/` on first run.

## How does decay work?

Memories accumulate an importance signal at creation, then *decay* over time
according to multi-time-scale Fourier-learned decay scales per memory type
(see [src/decay.rs](https://github.com/Portll/veld/blob/main/src/decay.rs)
and [src/decay_scales.rs](https://github.com/Portll/veld/blob/main/src/decay_scales.rs)).
Frequently retrieved memories regain importance (access count is a scoring
signal). Anchored memories (`POST /api/anchor`) are exempt.

See [Memory tiers](architecture/memory-tiers.md).

## What if two memories contradict?

The retrieval pipeline detects pairwise semantic opposition at **Layer
4.92** and demotes the older of two contradictory memories. This is a
heuristic — for genuine epistemic discipline (citation provenance,
source-tier weighting, autophagy guards), see the LLM-Wiki plan in
[decision 0002](decisions/0002-llm-wiki-dual-pathway.md) and the planned
`## Epistemic Hygiene` section in CLAUDE.md.

## Is this production-ready?

The current branch is `v0.7.6-unstable`, being stabilized toward a clean
`v0.8` release. Treat the branch tip as internal/unstable unless a tagged
release says otherwise. The next public-release-quality target is `v0.9`
(see [PROGRESS.md](https://github.com/Portll/veld/blob/main/PROGRESS.md)).

For local-developer use today, it's fine — many people run it on their
own machines daily. For production multi-tenant deployment, wait for v0.9.

## How do I uninstall?

```sh
# Stop services first (see Deploying guide)
cargo uninstall veld
npm uninstall -g @veld/memory-mcp
rm -rf ~/.local/share/veld   # Linux; adjust path per platform
```

## See also

- [Quickstart](quickstart.md)
- [Architecture overview](architecture/overview.md)
- [Glossary](glossary.md)
