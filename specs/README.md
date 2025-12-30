# SHODH Memory API Specification

Shared API specification for interoperable memory implementations.

## Goal

Define a common interface that allows:
- Seamless switching between backends (local, edge, cloud)
- Migration between implementations (with re-embedding)
- Hybrid setups (local for sensitive data, edge for sync)

## Format

- **OpenAPI 3.1** - Machine-readable API definition
- **Markdown** - Design rationale, usage examples

## Core Operations

| Operation | Description |
|-----------|-------------|
| `remember` | Store a new memory |
| `recall` | Semantic search across memories |
| `recall_by_tags` | Tag-based filtering |
| `forget` | Delete memory by ID |
| `consolidate` | Trigger decay/consolidation (optional) |

## Implementations

| Project | Backend | Embeddings |
|---------|---------|------------|
| [shodh-memory](https://github.com/varun29ankuS/shodh-memory) | RocksDB (local) | MiniLM-L6-v2 (ONNX) |
| [shodh-cloudflare](https://github.com/doobidoo/shodh-cloudflare) | D1 + Vectorize (edge) | Workers AI (bge-small) |
| [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) | Local | ONNX |

## Contributing

PRs welcome! Start with the OpenAPI spec in `openapi.yaml`.
