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
`turn_context: Option<bool>` controls whether a `Conversation` memory is
indexed with its date and the turn it follows (`src/memory/index_text.rs`,
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
well-ranked conversation turn into the pool just below it, and the
cross-encoder judges it on its own text.

| Dial | Env | Default | What it switches |
|---|---|---|---|
| `neighbour_expansion` | `VELD_DIALS_NEIGHBOUR_EXPANSION=0\|1` | `None` (on) | For each of the top `neighbour_expansion_top` fused candidates that is a live conversation turn, the turns recorded as following it (at most two) join the pool at the parent's fused score × `NEIGHBOUR_EXPANSION_SCORE_DECAY` (0.95, `src/constants.rs`). A neighbour already in the pool below that score is lifted to it. A neighbour always scores below its parent, so the top fused candidate does not change. `Some(false)` leaves the pool as fusion built it. |
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
