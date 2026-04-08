# Veld Architecture Guide

## What Each Component Does

### Memory (src/memory/)

The memory system stores experiences — text content with metadata — and retrieves
them through multi-signal ranking. Think of it as a persistent semantic search
engine that learns which memories matter through Hebbian reinforcement.

| Module | Purpose |
|--------|---------|
| `mod.rs` | `MemorySystem` — the main entry point. Stores, retrieves, decays, consolidates. |
| `types.rs` | Core types: `Memory`, `Experience`, `Query`, `MemoryTier` (Working → Episodic → Semantic). |
| `storage.rs` | RocksDB persistence layer. Content-hash deduplication. |
| `recall.rs` | Multi-layer retrieval: semantic vectors (L1), graph spreading (L2), BM25 keywords (L3), ontological re-ranking (L4.9), RRF fusion. |
| `hybrid_search.rs` | Fuses vector and keyword scores using reciprocal rank fusion. |
| `graph_retrieval.rs` | Entity-aware recall — spreading activation through the knowledge graph to find memories connected by shared entities. |
| `compression.rs` | `MemoryCompressor` — extracts semantic facts from memories. `FactConsolidator` — merges overlapping facts. |
| `facts.rs` | `SemanticFactStore` — stores extracted facts like "Alice works at Acme". Indexed by entity and type. |
| `sessions.rs` | Session boundaries — marks when conversations start/end. |
| `context.rs` | (Unused) ContextBuilder — was superseded by proactive_context in handlers. |
| `context_blocks.rs` | Key-value context store for persistent blocks (e.g., "user_preferences"). |
| `prospective.rs` | Reminders — time-triggered and context-triggered memory surfacing. |
| `todos.rs` | GTD-style todo system with projects, dependencies, subtasks, comments. |
| `replay.rs` | Dream replay — re-activates and strengthens memory pathways during consolidation. |
| `voronoi.rs` | Voronoi analysis — finds gaps and voids in the memory landscape topology. |
| `wavelet_sessions.rs` | Detects natural session boundaries from temporal patterns. |
| `files.rs` | File memory — indexes codebase files for project-aware recall. |
| `lineage.rs` | Memory provenance — tracks which memories derived from which. |
| `segmentation.rs` | Breaks long content into coherent segments for embedding. |
| `feedback.rs` | Reinforcement signals — user feedback strengthens/weakens memories. |
| `injection.rs` | (Unused) Planned injection selector for proactive surfacing. |

### Knowledge Graph (src/graph_memory.rs)

The graph stores entities (people, places, things, concepts) and relationships
between them. Edges strengthen through Hebbian co-activation — entities mentioned
together become more strongly linked.

**Entity lifecycle:**
```
Text input
  → NER extraction (TinyBERT: Person, Organization, Location, Misc)
  → Regex extraction (ALL_CAPS → Technology, JIRA-123 → Issue)
  → Tag extraction (memory tags → Technology)
  → EntityNode creation (uuid, name, labels, salience, attributes)
  → Graph insertion (RocksDB CF_ENTITIES)
  → Co-occurrence edges between all entities in same memory
  → Hebbian strengthening on each co-retrieval
```

**Edge tiers (memory consolidation):**
- **L1 Working**: New edges, high decay. Initial strength ~0.3.
- **L2 Episodic**: Proven edges (survived decay). Moderate decay.
- **L3 Semantic**: Consolidated, near-permanent. Very slow decay.

**Long-Term Potentiation (LTP):** Edges that activate frequently gain protection:
- Burst (5+ in 24h): 2× slower decay
- Weekly (3+/week for 2+ weeks): 3× slower decay
- Full (10+ lifetime): 10× slower decay

### How People and Places Are Stored Today

An entity like "Alice Smith" arrives through NER detection:

```
EntityNode {
  uuid: <generated>,
  name: "Alice Smith",
  labels: [Person],
  mention_count: 1,
  salience: 0.85,        // NER confidence
  is_proper_noun: true,
  attributes: {},         // empty — no structured data
  name_embedding: <384-dim vector>,
  pii_classification: PersonalIdentity,
}
```

A place like "Melbourne" arrives the same way:

```
EntityNode {
  name: "Melbourne",
  labels: [Location],
  attributes: {},         // no coordinates, no country, no timezone
}
```

**The problem:** These are bags of text. There's no:
- Canonical identity (is "Alice Smith" the same as "A. Smith" or "alice smith"?)
- Structured attributes (DOB, email, coordinates, timezone)
- External anchoring (Wikidata ID, OpenStreetMap ID)
- Temporal validity (Alice was CEO from 2020-2023)
- Alias resolution ("Melbourne" = "Melbourne, Australia" ≠ "Melbourne, Florida")

### Embeddings (src/embeddings/)

| Module | Purpose |
|--------|---------|
| `minilm.rs` | MiniLM-L6 (384-dim) — fast local embeddings for memory content |
| `nomic.rs` | Nomic Embed v2 (768-dim) — higher quality, optional upgrade |
| `ner.rs` | TinyBERT NER — extracts Person/Org/Location/Misc entities |
| `cross_encoder.rs` | Cross-encoder reranking — pairwise relevance scoring |
| `http_embedder.rs` | Proxy to external embedding APIs |
| `zenoh_embedder.rs` | Distributed embeddings over Zenoh pub/sub |
| `circuit_breaker.rs` | (Unused) Resilient embedder wrapper with circuit breaker pattern |
| `keywords.rs` | BM25 keyword extraction for hybrid search |
| `chunking.rs` | Splits long text into embedding-sized chunks |

### Handlers (src/handlers/)

These are the HTTP API surface. Each handler validates input, calls the memory
system or graph, and returns JSON.

| Handler | Key Endpoints |
|---------|--------------|
| `remember.rs` | `/api/remember`, `/api/remember/batch`, `/api/upsert` |
| `recall.rs` | `/api/recall`, `/api/proactive_context`, `/api/context_summary`, `/api/relevant` |
| `crud.rs` | `/api/memory/{id}` (GET/PUT/DELETE), `/api/forget/{id}`, `/api/anchor` |
| `graph.rs` | `/api/graph/entity/find`, `/api/graph/traverse`, `/api/graph/{user}/stats` |
| `todos.rs` | Full GTD: `/api/todos/*`, `/api/projects/*`, `/api/reminders/*` |
| `facts.rs` | `/api/facts/list`, `/api/facts/search`, `/api/facts/by-entity` |
| `consolidation.rs` | `/api/consolidate`, `/api/index/verify`, `/api/backup/*` |
| `gap_analysis.rs` | `/api/gap/analyze`, `/api/gap/voronoi`, `/api/gap/persistence` |
| `lineage.rs` | `/api/lineage/trace`, `/api/lineage/link` |
| `sessions.rs` | `/api/sessions`, `/api/sessions/end` |
| `mif.rs` | `/api/export/mif`, `/api/import/mif` — memory interchange format |
| `external_dimensions.rs` | `/api/sleight/dimensions` — overlook vector push from sleight |

### MCP Server (mcp-server/)

TypeScript bridge that exposes veld as MCP tools for Claude Code / Claude Desktop.
38 tools covering remember, recall, forget, todos, projects, reminders, facts,
graph operations, gap analysis, and system health.

### Hooks (hooks/)

Claude Code integration hooks:
- `session-start.sh` — seeds context from prior sessions on launch
- `memory-hook.ts` — auto-records tool usage, surfaces proactive context
- `stop.sh` — persists session summary on exit

---

## What's Missing

### 1. Prompt Generation (veld_prompt_gen)

There is no endpoint that takes a goal/question and assembles a complete prompt
from memories, entities, facts, todos, and context blocks. Today, the caller
must make 3-5 separate API calls and stitch the results together. We need a
single endpoint that does the assembly.

### 2. Entity Exactness

Entities are fuzzy bags of text. We need:
- **Canonical resolution**: "Alice Smith" = "A. Smith" = "alice smith"
- **Structured attributes**: typed fields per entity type (Person → DOB, email; Location → lat/lng, timezone)
- **Entity merging**: when duplicates detected, merge edges and update references
- **Alias management**: explicit alias→canonical mapping

### 3. Learning Loop Closure

The system records memories and retrieves them, but doesn't close the loop:
- No way to say "this retrieval was helpful/unhelpful" that feeds back into ranking
- `reinforce` endpoint exists but isn't wired into the scoring layers
- No prompt quality feedback → memory importance adjustment

### 4. Sleight ↔ Veld Evaluation Feedback

Sleight evaluates code quality, but evaluation results don't flow back to
influence which memories are surfaced. The `/api/sleight/dimensions` endpoint
exists but the pushed dimensions don't affect recall scoring.
