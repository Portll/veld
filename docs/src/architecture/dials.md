# Cognitive Dials

`CognitiveDials` is the unified, runtime-tunable surface for veld's cognitive
system. One place to toggle or tune the parameters a human (via the monitor's
"Dials" tab) or Spine (via the automated-tuning path) can drive at runtime —
rather than a second tuning system beside
[`FeedbackStore`](../reference/http-api.md)'s learned weights and the
Spine-pushed [`ExternalDimensionScores`](knowledge-graph.md#external-dimension-scores).

Source: [src/memory/dials.rs](https://github.com/Portll/veld/blob/main/src/memory/dials.rs).

## Two groups of dials

| Group | Touches | Where it runs |
|---|---|---|
| **Reconsolidation (plasticity)** | The FIX-R1 labile-window close and the composable re-encode layers | `run_maintenance` — background, off the synchronous recall hot path |
| **Retrieval (fast ↔ accurate)** | Optional overrides for the hybrid-search knobs (cross-encoder budget, RRF k, refiner mode, etc.) | The recall hot path itself, when set |

`Default` is everything OFF / `None`. A freshly-seeded `CognitiveDials`
changes nothing — every retrieval dial is `None` so the engine uses its
learned-weight default; every plasticity dial is `false`/zero so
reconsolidation behaves exactly as it did pre-dials.

### Candidate-generation switches

| Dial | Env | Default | What it switches |
|---|---|---|---|
| `date_window_leg` | `VELD_DIALS_DATE_WINDOW_LEG` | `None` (on) | The date window candidate leg (recall Layer 3.7). When a query names a calendar day ("1 May, 2022") or month-year ("April 2022"), the memories whose `created_at` falls in that window — a day widened by one day either side, a month as written — enter RRF fusion as their own list, ranked by similarity to the query and capped at `candidate_count` (or `max_results × 8`). Year-only mentions never fire it. `Some(false)` leaves dated queries to the Layer 4.55 temporal boost alone. |
| `relative_priors` | `VELD_DIALS_RELATIVE_PRIORS` | `None` (off) | How Layer 5 combines the fused base with its priors (recency, arousal, credibility, temporal match, access, graph strength and the rest). Off, they are added to the base, and since the RRF base at k = 20 is about 0.033 at rank 1 and 0.010 at rank 50, priors worth up to ~0.5 decide the order. On, the base is divided by the best base among the candidates and multiplied by `1 + Σ priors`, so relevance leads and the priors adjust it; the Hebbian bonus and the 2-hour session bonus are not added a second time. This also changes the cross-encoder blend, whose share then applies to a base on the same 0–1 scale as its own score. Off until a locomo-holdout A/B decides it. |
| `graph_episode_entities` | `VELD_DIALS_GRAPH_EPISODE_ENTITIES` | `None` (every entity) | How many of the entities recall's graph traversal reaches (Layer 2) have their recent episodes read as candidates. The entities kept are those with the highest decay factor (the nearest to a query entity; an entity reached twice counts at its best), ties broken by salience, then name. Each expanded entity costs an index scan and up to `RECALL_GRAPH_EPISODES_PER_ENTITY` episode reads, so a dense graph can spend most of a recall here. Clamped to 1–1000. |
| `confidence_gate_all_paths` | `VELD_DIALS_CONFIDENCE_GATE_ALL_PATHS` | `None` (semantic recall only) | Where the calibrated-confidence gate (`0.85 + 0.15 × confidence` once a memory has five feedback observations) and abstention (importance × confidence below the abstention threshold drops a memory with more than two feedback events, unless the graph reached it with activation above 0.3) apply. Unset, only semantic recall applies them. `Some(true)` applies them to spreading-activation recall and filter-only recall too (the gate scales their order, then abstention filters), and abstention to tag, date and advanced-search recall, which order by time or by the store, so the multiplier would move nothing there. |
| `bm25_read_staleness_ms` | `VELD_DIALS_BM25_READ_STALENESS_MS` | `None` (commit every buffered write) | How long a journaled write may sit buffered in the BM25 index before a recall commits it itself. `None` commits everything buffered before the search, so a memory is keyword-findable on the first recall after its write (a `/api/remember` memory has no embedding until its background embed finishes, so BM25 is then its only route into recall). `Some(ms)` leaves younger writes to the BM25 projection's background committer, which commits within `COMMIT_INTERVAL` (200 ms) of the previous commit; a recall then pays no commit, and can miss the keyword match of a memory written less than that long ago. Clamped to at most 60 000. |

The recall gate records the overrides a run was measured under
(`retrieval_dials` in `tests/recall/locomo-gate-baseline.json`: `date_window_leg`,
`relative_priors`, `graph_episode_entities` and `confidence_gate_all_paths`) and
refuses to diff two runs that overrode them differently. `bm25_read_staleness_ms` is not recorded: the gate ingests through
`MemorySystem::remember`, which commits BM25 as it writes, so nothing is ever
buffered when it recalls.

### Signal switches

`disabled_signals` (env `VELD_DIALS_DISABLE_SIGNALS`, comma-separated) switches named
recall signals off; empty, the default, runs all of them. The ids are `SIGNAL_IDS` in
`src/memory/dials.rs`, the set recall passes to `signal_enabled`. An id outside it
disables nothing, so it is not accepted silently: `PUT /api/dials` refuses the body with
a 400 naming the unknown id, and the env var and a persisted blob drop it with a warning
naming it.

## Lifecycle

```mermaid
flowchart LR
    ENV[VELD_DIALS_*<br/>env vars] -->|from_env| INIT[Seed at startup]
    KV[(KV store<br/>per-user blob)] -->|deserialize + overlay| INIT
    INIT --> ARC[Arc&lt;RwLock&lt;CognitiveDials&gt;&gt;]
    ARC --> HS[hybrid_search_engine.attach_dials]
    ARC --> MS[MemorySystem]
    PUT[PUT /api/dials] -->|sanitize + stamp updated_at| ARC
    ARC -->|serde_json::to_vec| KV
```

1. **Seed from env** — `CognitiveDials::from_env()` reads `VELD_DIALS_*`
   variables once at startup.
2. **Overlay from KV** — if the user has a persisted blob at
   `COGNITIVE_DIALS_KEY`, deserialize and `sanitized()`-overlay it. A corrupt
   blob is logged and skipped (not fatal).
3. **Attach** — the `Arc<RwLock<CognitiveDials>>` is handed to the hybrid
   search engine and stored on `MemorySystem`. The engine's recall path
   reads through this handle on every call, so a runtime change takes
   effect immediately.
4. **Persist** — `PUT /api/dials` replaces the in-memory state, calls
   `sanitized()` to clamp values into legal ranges, stamps `updated_at`,
   and writes the JSON blob back to the KV store.

## REST surface

Two routes on the **protected** router (per-user, X-API-Key required):

```
GET /api/dials?user_id=…   → CognitiveDials JSON
PUT /api/dials             → replace with the body's CognitiveDials JSON
```

Full-replace REST semantics — the client GETs, edits, and PUTs the whole
struct. There is no PATCH; partial updates would race against
Spine-driven updates on the same key.

Source: [src/handlers/dials.rs](https://github.com/Portll/veld/blob/main/src/handlers/dials.rs).

## TUI surface

The Veld Monitor exposes a **"Dials" tab** (see
[tui/src/tabs/dials.rs](https://github.com/Portll/veld/blob/main/tui/src/tabs/dials.rs))
that GETs the current dials, renders each value as a tunable widget, and
PUTs on commit. Real-time visibility on what's currently set + an easy way
to undo experiments.

## Reranker dials

Five retrieval dials drive the Layer 5.3 cross-encoder
([retrieval.md](retrieval.md#cross-encoder-reranking)):

| Dial | Type | `None` means | Env seed |
|---|---|---|---|
| `use_reranking` | `Option<bool>` | the layer runs | `VELD_DIALS_USE_RERANKING` |
| `rerank_count` | `Option<usize>`, clamped to 1..=200 | at most 20 candidates (30 for a decomposed query), and never more than twice `max_results`; a set value is the budget itself | `VELD_DIALS_RERANK_COUNT` |
| `rerank_pool` | `Option<usize>`, clamped to 10..=500 | Layer 4 keeps 30 fused candidates (50 for a decomposed query) for the layer; never fewer than `rerank_count` | `VELD_DIALS_RERANK_POOL` |
| `ce_blend` | `Option<f32>`, clamped to 0..=1 | each query shape takes its own cross-encoder share (`ce_blend_<shape>`, else `constants::CE_BLEND_*`); a set value is the share for every shape, and a decomposed query never takes less than its own | `VELD_TUNE_CE_BLEND` |
| `rerank_model` | `Option<String>`, a registry id | the process-wide model: `VELD_CROSS_ENCODER`, else `mxbai-rerank-xsmall-v1` | `VELD_DIALS_RERANK_MODEL` |

Depth takes all three of `rerank_pool`, `rerank_count` and `ce_blend`. On the LoCoMo
gate a quarter of all gold sits between rank 11 and rank 100. A larger `rerank_count`
cannot reach gold that Layer 4 already cut, and at a share near 18% the cross-encoder moves
a candidate a few places, not from rank 40 into the top ten. Each candidate in the
budget is one cross-encoder forward pass, so latency grows with `rerank_count`.

`rerank_model` is applied when the dials are loaded at startup and on every
`PUT /api/dials`: the user's `CrossEncoderReranker` switches to the named
model, keeps every model it has selected before loaded, and pre-fetches a new
one in the background. `sanitized()` drops an id the registry does not know, and
a model that cannot be built clears the dial before it is stored, so
`GET /api/dials` never names a model that is not reranking. Both dials default to
`None`: the shipped pipeline is unchanged until someone turns them.

The recall gate records the model, cap, pool and blend it measured under
(`RerankSetting` in `src/recall_harness/report.rs`) and refuses to gate across
them. `recall.yml` takes `rerank_model`, `rerank_count`, `rerank_pool` and
`ce_blend` as dispatch inputs for a comparison run (`record=true`). The baseline
is `mxbai-rerank-xsmall-v1` at cap 20 with the shipped pool and blend, and the
parity arm against upstream shodh-memory is `ms-marco-MiniLM-L-6-v2` at cap 30.

Selectable models (`src/embeddings/cross_encoder.rs`). Two are pinned to a
revision and verified by SHA-256 before they load; the rest follow their
repository's `main` branch and should be pinned before they are relied on for a
measurement.

| Registry id | Params | Licence | Pinned |
|---|---|---|---|
| `mxbai-rerank-xsmall-v1` (default) | 70 M | Apache-2.0 | `b5c6e9da` |
| `ms-marco-MiniLM-L-6-v2` (parity arm) | 22 M | Apache-2.0 | `233902d2` |
| `mxbai-rerank-base-v1` | 184 M | Apache-2.0 | no |
| `mxbai-rerank-large-v1` | 435 M | Apache-2.0 | no |
| `jina-reranker-v1-tiny-en` | 33 M | Apache-2.0 | no |
| `jina-reranker-v1-turbo-en` | 37 M | Apache-2.0 | no |
| `bge-reranker-v2-m3` (hooman650 ONNX export) | 568 M | Apache-2.0 | no |
| `jina-reranker-v2-base-multilingual` | 278 M | CC-BY-NC-4.0 (non-commercial; never a default) | no |

Each entry's licence is the `license` field of its `CrossEncoderConfig`, as the
model card declared it at the revision recorded here.

## Forward-compatible serialisation

Every field in `CognitiveDials` carries `#[serde(default)]` so older
persisted blobs decode forward-compatibly when new dials land. The
sanitisation pass clamps each retrieval dial to its valid range
(`None` is always valid) before the value reaches the engine — a corrupt
or out-of-range write cannot destabilise retrieval.

## Why dials, not just config

`ServerConfig` and the `VELD_*` env vars are *startup-time* knobs. Dials are
the *runtime* surface for things that benefit from being tweaked while the
server is live:

- Quick A/B sweeps without a restart.
- Spine-driven automated tuning that wants to push a new RRF k or
  cross-encoder budget every few minutes based on quality signals.
- Per-user overrides — every user's dials are independent.

The boundary is: structural choices (storage backend, embedder URL, bind
address) stay in config; behavioural-tuning choices (rerank budget, decay
factor, refiner mode) live in dials.

## See also

- [Retrieval pipeline](retrieval.md) — the 20-signal pipeline whose
  cross-encoder / RRF / refiner knobs the retrieval dials override.
- [Consolidation](consolidation.md) — the plasticity dials gate the
  reconsolidation paths that run inside `run_maintenance`.
- [Instance registry](instance-registry.md) — sibling subsystem that the
  monitor's other tabs read from.
