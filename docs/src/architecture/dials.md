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

### Ingest dial: `turn_context`

One retrieval dial acts at write time rather than on the recall path.
`turn_context: Option<bool>` controls whether a linked memory (a `Conversation`
turn, or any memory its writer threaded into an episode with an `episode_id` or
a `preceding_memory_id`) is indexed with its date and the memory it follows (`src/memory/index_text.rs`,
[ADR 0009](../decisions/0009-index-text-derived-not-stored.md)). `None`
keeps the engine default, which is **on**; `Some(false)` indexes the bare
content. Env seed: `VELD_DIALS_TURN_CONTEXT=0|1`. Flipping it changes what
the embedder is fed for turns written afterwards; records already written
keep their embedding until re-indexed. Prior-turn resolution is counted in
`veld_turn_context_total{source="preceding_id"|"conversation_key"|"none"}`.

### Recall dials: neighbour expansion

An anaphoric reply ("Nate: I've had them for 3 years now!") shares no word
with the question about it ("How long has Nate had his turtles?"), but the
turn it answers usually does. Neighbour expansion (recall Layer 4.97, after
fusion and before the pre-rerank cut) brings the turn that followed a
well-ranked linked memory into the pool just below it, and the
cross-encoder judges it on its own text. A linked memory is a conversation
turn or an episode-threaded memory, so an agent session's fix follows the
error it fixed the way a reply follows its question.

| Dial | Env | Default | What it switches |
|---|---|---|---|
| `neighbour_expansion` | `VELD_DIALS_NEIGHBOUR_EXPANSION=0\|1` | `None` (on) | For each of the top `neighbour_expansion_top` fused candidates that is a live linked memory, the memories recorded as following it (at most two) join the pool at the parent's fused score × `NEIGHBOUR_EXPANSION_SCORE_DECAY` (0.95, `src/constants.rs`). A neighbour already in the pool below that score is lifted to it. A neighbour always scores below its parent, so the top fused candidate does not change. `Some(false)` leaves the pool as fusion built it. |
| `neighbour_expansion_top` | `VELD_DIALS_NEIGHBOUR_EXPANSION_TOP=<n>` | `None` (10) | How many top fused candidates are read as parents. Sanitised to `1..=50` (50 is the widest pre-rerank pool, a decomposed query's). |
| `neighbour_expansion_predecessor` | `VELD_DIALS_NEIGHBOUR_EXPANSION_PREDECESSOR=0\|1` | `None` (off) | Also brings in the turn each ranked turn follows (its `preceding_memory_id`), at the same score. Has no effect while `neighbour_expansion` is off. |

The successors come from a per-user index written by `remember`
(`src/memory/turn_successors.rs`, one KV row per prior → successor pair,
the same on RocksDB and redb), so a turn ranked after a restart still finds
its reply. A row is written only when `remember` resolves the prior turn,
which `turn_context` governs: turns written while `turn_context` is off have
no successor rows. A successor forgotten since its row was written is
skipped. The leg runs in the `Full` layer mode only, so the layer-attribution
rungs below it are unchanged. Turns placed are counted in
`veld_recall_neighbour_candidates_total{direction="successor"|"predecessor"}`,
and the gold funnel records the leg's own list as the `neighbour` stage.

### Candidate-generation switches

| Dial | Env | Default | What it switches |
|---|---|---|---|
| `date_window_leg` | `VELD_DIALS_DATE_WINDOW_LEG` | `None` (on) | The date window candidate leg (recall Layer 3.7). When a query names a calendar day ("1 May, 2022") or month-year ("April 2022"), the memories whose `created_at` falls in that window — a day widened by one day either side, a month as written — enter RRF fusion as their own list, ranked by similarity to the query and capped at `candidate_count` (or `max_results × 8`). Year-only mentions never fire it. `Some(false)` leaves dated queries to the Layer 4.55 temporal boost alone. |

The recall gate records the overrides a run was measured under
(`retrieval_dials` in `tests/recall/locomo-gate-baseline.json`) and refuses to
diff two runs that overrode them differently.

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
