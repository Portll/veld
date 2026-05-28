# Knowledge graph

Veld's knowledge graph is the Hebbian-learning substrate. Nodes are entities
or episodes; edges are weighted relationships that strengthen when their
endpoints are retrieved together.

## Node types

```rust
// src/graph_memory.rs
pub struct EntityNode { ... }      // Named entities (people, projects, files)
pub struct EpisodicNode { ... }    // Specific memory events
pub struct RelationshipEdge { ... } // Weighted edge between two nodes
```

Implementation: [`src/graph_memory.rs`](https://github.com/Portll/veld/blob/main/src/graph_memory.rs)
plus the `GraphStore` trait in
[`src/storage/mod.rs`](https://github.com/Portll/veld/blob/main/src/storage/mod.rs).

## Hebbian learning

> "Neurons that fire together, wire together."

When two memories surface together during recall, the edge between their
associated graph nodes strengthens. This happens during
[consolidation](consolidation.md) via `strengthen_memory_edges`. Over time,
frequently-co-retrieved memories become densely connected, and spreading
activation surfaces them as a cluster.

## Spreading activation

[`src/memory/graph_retrieval.rs`](https://github.com/Portll/veld/blob/main/src/memory/graph_retrieval.rs)
implements spreading activation. From a seed memory (or set of memories),
activation propagates along edges proportional to edge weight; the activation
becomes scoring signal 8 in the [retrieval pipeline](retrieval.md).

## Entity resolution

`POST /api/entity/*` endpoints provide CRUD over entity nodes:

- `/api/entity/resolve` — find or create an entity by name
- `/api/entity/attribute` — set a structured attribute (e.g., `role: "engineer"`)
- `/api/entity/alias` — register an alias for an existing entity
- `/api/entity/merge` — merge two entity nodes (preserves edge history)

Entity resolution is also used during [ingest](ingest.md) — named entities
discovered in incoming content are resolved against the graph and either
linked to existing nodes or created fresh.

## Gap topology

[`src/memory/gap_topology/`](https://github.com/Portll/veld/tree/main/src/memory/gap_topology)
runs Voronoi decomposition over the graph to find entity neighbourhoods
with sparse coverage. These "gaps" become hints for what the agent should
ask the user about, or where new ingest should be prioritised.

## External dimension scores

Sleight (an external evaluator) can push topological-health scores via
`POST /api/sleight/dimensions`:

- `density` — entity density in the region
- `coherence` — semantic coherence of neighbours
- `closure` — fraction of potential triangles closed
- `confidence` — average edge confidence
- `isotropy` — directional balance of knowledge

These scores modulate retrieval rank when fresh (signal 20 in the pipeline)
and are tracked in `ExternalDimensionScores`. Staleness detection is built
in: `is_stale()` returns true if no fresh score push has happened in over
an hour.

## See also

- [Retrieval pipeline](retrieval.md) — how graph edges feed retrieval scoring
- [Consolidation](consolidation.md) — when edges actually strengthen
