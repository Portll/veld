//! Backend-agnostic relational storage — the W4 foundation.
//!
//! This module defines the [`RelationalStore`] trait and supporting value
//! types ([`Param`], [`Row`], [`ColumnValue`], [`ColumnError`], [`ColumnMeta`],
//! [`OwnedColumn`], and the [`FromColumn`] decoder trait), along with the
//! SQLite-backed adapter [`SqliteRelationalStore`].
//!
//! Postgres / Supabase / MSSQL implementations land in follow-up commits.
//! Call-site ports of the existing rusqlite-backed slow stores also land
//! separately — this module is intentionally additive so other agents can
//! build against a stable surface.
//!
//! # Implementing your own backend
//!
//! Downstream crates can implement [`RelationalStore`] for their own
//! driver. Construct rows via [`Row::from_owned`] with the public
//! [`OwnedColumn`] value type.
//!
//! ```ignore
//! use std::borrow::Cow;
//! use async_trait::async_trait;
//! use veld::storage::relational::{
//!     ColumnMeta, OwnedColumn, Param, RelationalBackend, RelationalStore, Row,
//! };
//!
//! pub struct DuckDbRelationalStore { /* pool */ }
//!
//! #[async_trait]
//! impl RelationalStore for DuckDbRelationalStore {
//!     type Error = duckdb::Error;
//!
//!     async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, Self::Error> {
//!         todo!("translate Param → duckdb bindings, run, return rows affected")
//!     }
//!
//!     async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, Self::Error> {
//!         let columns = vec![ColumnMeta { name: "id".into(), sql_type: "BIGINT".into() }];
//!         let values  = vec![OwnedColumn::I64(42)];
//!         Ok(vec![Row::from_owned(columns, values)])
//!     }
//!
//!     fn backend(&self) -> RelationalBackend {
//!         RelationalBackend::Custom(Cow::Borrowed("duckdb"))
//!     }
//! }
//! ```
//!
//! ## Type-erasing the error for `Arc<dyn ...>` consumers
//!
//! Veld's read/write paths accept `Arc<dyn RelationalStore<Error = BoxError>>`,
//! where [`BoxError`] is `Box<dyn std::error::Error + Send + Sync + 'static>`.
//! `anyhow::Error` cannot be used as `type Error` because it does not implement
//! `std::error::Error` (deliberate choice in the anyhow crate). Wrap your
//! native error at the boundary:
//!
//! ```ignore
//! use std::sync::Arc;
//! use veld::storage::relational::{BoxError, RelationalStore};
//!
//! struct Adapter<S>(Arc<S>);
//!
//! #[async_trait]
//! impl<S: RelationalStore> RelationalStore for Adapter<S> {
//!     type Error = BoxError;
//!     async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, BoxError> {
//!         self.0.execute(sql, params).await.map_err(|e| Box::new(e) as BoxError)
//!     }
//!     async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
//!         self.0.query(sql, params).await.map_err(|e| Box::new(e) as BoxError)
//!     }
//!     fn backend(&self) -> RelationalBackend { self.0.backend() }
//! }
//! ```

pub mod sqlite;
pub mod store;
pub mod types;

pub use sqlite::SqliteRelationalStore;
pub use store::RelationalStore;
pub use types::{
    ColumnError, ColumnMeta, ColumnValue, FromColumn, OwnedColumn, Param, RelationalBackend, Row,
};

/// Type-erased error type for `Arc<dyn RelationalStore<Error = ...>>` bindings.
///
/// The trait's associated `Error` bound is `std::error::Error + Send + Sync +
/// 'static`. `anyhow::Error` does NOT implement `std::error::Error` (deliberate
/// choice in the anyhow crate); a binding `Error = anyhow::Error` fails at any
/// impl site even if the dyn-trait-object declaration compiles.
///
/// `BoxError` is `Box<dyn StdError + Send + Sync>`, which DOES implement
/// `StdError` via the blanket impl on boxed errors, and is therefore the
/// canonical type-erased error for dyn bindings. Backends adapt their native
/// error by `.map_err(|e| Box::new(e) as BoxError)`.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
