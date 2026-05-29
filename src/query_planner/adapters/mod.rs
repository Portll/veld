//! Real backend adapters for the W6 query planner trait surfaces.
//!
//! The planner's three backend traits (`RelationalQuerier`, `VectorQuerier`,
//! `GraphQuerier`) are defined in `super::executor`. This module supplies
//! implementations that consume Veld's existing stores:
//!
//! - [`RealRelationalQuerier`] queries the slow-store `memories` table via
//!   the W4 `RelationalStore` trait.
//! - [`RealVectorQuerier`] queries per-kind Vamana indices through an
//!   injected [`vector::VamanaProvider`].
//! - [`RealGraphQuerier`] resolves entity/episode predicates through an
//!   injected [`graph::GraphProvider`].
//! - [`StubGraphQuerier`] returns empty results — a sealed null object for
//!   relational/vector-only deployments and the executor's own tests.
//!
//! The `*Provider` traits keep each adapter free of any dependency on the
//! HTTP/state layer; the live bindings (`impl … for
//! MultiUserMemoryManager`) live next to the manager in `handlers::state`.
//! Each adapter is constructible directly in tests so the planner can be
//! exercised against in-memory fixtures without touching production wiring.

pub mod graph;
pub mod graph_stub;
pub mod relational;
pub mod vector;

pub use graph::{GraphProvider, RealGraphQuerier};
pub use graph_stub::StubGraphQuerier;
pub use relational::{RealRelationalQuerier, ALLOWED_RELATIONAL_COLUMNS};
pub use vector::{RealVectorQuerier, VamanaProvider};
