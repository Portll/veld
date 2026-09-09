//! Recall Harness
//!
//! Quality measurement for the `MemorySystem::recall_*` pipeline: NDCG@k,
//! recall@k, precision@k, MRR, P@1, MAP, plus fixture loading and the JSON
//! report/baseline-comparison format.
//!
//! Imported from shodh-memory upstream, which runs it as a held-out LoCoMo
//! gate. Three modules are live. The remaining eight are on disk unmodified.
//!
//! The capture layer they need is now HERE: `memory::gold_funnel` and
//! `memory::fusion_features` landed in `803033d`, with four
//! `gold_funnel::record` stages wired into `memory/recall.rs` (graph, vector,
//! fusion, final). What still blocks the eight, measured against this tree:
//!
//! - Five APIs upstream has and veld does not. Three are ported as of this
//!   commit (`relation_type_distribution`, `synthetic_for_sim`,
//!   `annotate_query_ner`); two go deeper than their line counts suggested —
//!   `simulate_edge_aging` needs `GraphMemory::apply_decay_at` and
//!   `force_vector_quality_rebuild` needs `RetrievalEngine::force_quality_rebuild`,
//!   neither of which exists here.
//! - Three counters that instrument a directed-edge fix veld does not have:
//!   no `edge_neighbor`, no `dir_fix` branch. Adding the accessors would make
//!   them return a permanent zero, and `EDGE_DIR_FLIPS=0` reads as "the fix
//!   ran and flipped nothing" rather than "there is no such fix here".
//!   See `docs/HANDOFF-2026-09-07.md` §A1 — a decision, not a task.
//! - `bridge_harness` also needs five `UserMemory` -> `MemorySystem`
//!   unwrappings, veld's wrapper being absent upstream.
//!
//! They stay undeclared rather than half-adapted, because an installed harness
//! nobody has run is the failure this import exists to end.

pub mod fixtures;
pub mod metrics;
pub mod report;
