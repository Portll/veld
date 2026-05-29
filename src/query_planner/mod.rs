//! W6 query planner — foundation tier.
//!
//! This module is the *data model* and *planning logic* for veld's multi-source
//! query planner. It does NOT integrate with `RelationalStore`, the live Vamana
//! index, or the knowledge graph — a follow-up agent wires the trait surfaces
//! defined in [`executor`] to real backends.
//!
//! ## Layers
//!
//! - [`predicate`] — `Query`, `RelationalPredicate`, `VectorPredicate`,
//!   `GraphPredicate`, `StageCaps`. Pure data, serde-friendly.
//! - [`plan`] — `PhysicalPlan` and `PhysicalOp`. The planner's output and the
//!   executor's input.
//! - [`planner`] — selectivity heuristic + plan generator. Sorts predicates
//!   ascending by estimated cardinality and emits `Scan*` for the cheapest plus
//!   `Filter*` for the rest. Stage caps clamp every estimate.
//! - [`executor`] — `Executor` skeleton that walks a `PhysicalPlan` over three
//!   backend traits (`RelationalQuerier`, `VectorQuerier`, `GraphQuerier`).
//!   Mock impls in `executor::tests` prove the join.
//!
//! ## Why a separate planner
//!
//! Today's `recall` path bakes predicate ordering into the retrieval pipeline
//! (see `src/memory/recall.rs` and `src/memory/hybrid_search.rs`). The planner
//! decouples *what to ask* from *what to ask first*, so we can layer in
//! cost-based join ordering, stage budgets, and explainability without
//! rewriting recall.

pub mod adapters;
pub mod executor;
pub mod plan;
pub mod planner;
pub mod predicate;

pub use adapters::{
    GraphProvider, RealGraphQuerier, RealRelationalQuerier, RealVectorQuerier,
    StubGraphQuerier, VamanaProvider,
};
pub use executor::{
    Executor, GraphQuerier, RelationalQuerier, ScoredMemoryId, VectorQuerier,
};
pub use plan::{PhysicalOp, PhysicalPlan};
pub use planner::plan as build_plan;
pub use predicate::{
    GraphPredicate, Query, RelationalPredicate, StageCaps, VectorPredicate,
};
