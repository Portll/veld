//! Recall Harness
//!
//! Quality measurement for the `MemorySystem::recall_*` pipeline: NDCG@k,
//! recall@k, precision@k, MRR, P@1, MAP, plus fixture loading and the JSON
//! report/baseline-comparison format.
//!
//! Imported from shodh-memory upstream, which runs it as a held-out LoCoMo
//! gate. Four modules are live here; the eight ablation harnesses below need
//! `LayerMode` — an ablation switch through the retrieval pipeline that veld
//! does not have — and `runner` additionally needs `gold_funnel` and
//! `fusion_features`. Their sources are kept in this directory unmodified so
//! the follow-up is a wiring job rather than a re-import:
//!
//!   bridge_harness  decay_sim  forgetting_harness  lineage_harness
//!   multihop  ontology_harness  runner  temporal_harness

pub mod fixtures;
pub mod metrics;
pub mod report;
