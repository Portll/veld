//! Real backend adapters for the W6 query planner trait surfaces.
//!
//! The planner's three backend traits (`RelationalQuerier`, `VectorQuerier`,
//! `GraphQuerier`) are defined in `super::executor`. This module supplies
//! implementations that consume Veld's existing stores:
//!
//! - [`RealRelationalQuerier`] queries the slow-store `memories` table via
//!   the W4 `RelationalStore` trait.
//! - [`RealVectorQuerier`] queries per-kind Vamana indices through an
//!   injected [`vector::VamanaProvider`] (state-wiring is deferred to a
//!   follow-up so this module stays additive — no manager / handler edits).
//! - [`StubGraphQuerier`] returns empty results pending the real graph
//!   adapter; intentional placeholder so the executor compiles end-to-end.
//!
//! Each adapter is constructible directly in tests so the planner can be
//! exercised against in-memory fixtures without touching production wiring.

pub mod graph_stub;
pub mod relational;
pub mod vector;

pub use graph_stub::StubGraphQuerier;
pub use relational::{RealRelationalQuerier, ALLOWED_RELATIONAL_COLUMNS};
pub use vector::{RealVectorQuerier, VamanaProvider};
