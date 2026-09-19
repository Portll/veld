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
//! The ablation arms in `runner::analyze_ablation` do the same. Flags veld reads
//! (`SHODH_DISABLE_BOOSTS` for the families in `memory::ablation::FAMILIES`,
//! `SHODH_DISABLE_FACT_LAYERS`, `SHODH_GRAPH_FUSION_WEIGHT`) reach their code
//! paths; `SHODH_EVAL_NOW` freezes `memory::scoring_now`. An arm naming a flag
//! listed in `runner::UNAVAILABLE_IN_VELD`, or an upstream-only family, is not
//! run and reports `unavailable_in_veld` with the reason. A guard test fails if
//! an arm sets a flag that is neither read in `src/` nor declared unavailable.

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
