//! Recall Harness
//!
//! Quality measurement for the `MemorySystem::recall_*` pipeline: NDCG@k,
//! recall@k, precision@k, MRR, P@1, MAP, plus fixture loading and the JSON
//! report/baseline-comparison format.
//!
//! Imported from shodh-memory upstream, which runs it as a held-out LoCoMo
//! gate. Four modules are live. The remaining eight are on disk unmodified and
//! need, measured against this tree:
//!
//! - `runner` — five instrumentation call sites backed by two capture modules
//!   veld lacks: `memory::gold_funnel` (begin/take) and
//!   `memory::fusion_features` (begin/take + a jsonl dump). Diagnostic capture,
//!   not ranking, so porting them does not move retrieval behaviour.
//! - `bridge_harness`, `forgetting_harness`, `lineage_harness`, `multihop`,
//!   `ontology_harness`, `temporal_harness` — only `runner`'s helpers
//!   (`build_manager`, `ingest_corpus`, `run_smoke_suite_with_ranks`,
//!   `RunInputs`, `EVAL_USER`, `HARNESS_CLOCK_ANCHOR`), so they unblock with it.
//! - `bridge_harness` also needs five `UserMemory` -> `MemorySystem`
//!   unwrappings, veld's wrapper being absent upstream.
//! - `decay_sim` — `RelationshipEdge::synthetic_for_sim`, an upstream test
//!   constructor.
//!
//! They stay undeclared rather than half-adapted because `runner` cannot be
//! executed here without embedder models and a real corpus ingest, and an
//! installed harness nobody has run is the failure this import exists to end.

pub mod fixtures;
pub mod metrics;
pub mod report;
