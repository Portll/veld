//! Recall Harness
//!
//! Quality measurement for the `MemorySystem::recall_*` pipeline: NDCG@k,
//! recall@k, precision@k, MRR, P@1, MAP, plus fixture loading and the JSON
//! report/baseline-comparison format.
//!
//! Imported from shodh-memory upstream, which runs it as a held-out LoCoMo
//! gate. All eleven modules are declared and compiled.
//!
//! Where veld has no counterpart to an upstream mechanism the harness says so
//! rather than reporting zero: `EDGE_DIR_FLIPS` and `DIRECTED_REVERSE_COLLAPSE`
//! print `n/a`, because veld has no directed-edge traversal fix to instrument.
//!
//! The ablation arms in `runner::analyze_ablation` do not yet say so. They set
//! upstream `SHODH_*` pipeline flags that nothing in veld reads (the graph
//! expand/traverse/edge-dir/boost/fusion flags, `SHODH_DISABLE_BOOSTS`,
//! `SHODH_DISABLE_FACT_LAYERS`), so every such arm measures the baseline under
//! another name. Likewise `SHODH_EVAL_NOW` pins no clock here. Read an arm that
//! matches baseline as "not wired in veld", never as "no effect".

pub mod bridge_harness;
pub mod decay_sim;
pub mod fixtures;
pub mod forgetting_harness;
pub mod lineage_harness;
pub mod metrics;
pub mod multihop;
pub mod ontology_harness;
pub mod report;
pub mod runner;
pub mod temporal_harness;
