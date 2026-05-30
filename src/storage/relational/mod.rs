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
//! where [`BoxError`] is a newtype around
//! `Box<dyn std::error::Error + Send + Sync + 'static>`. `anyhow::Error` cannot
//! be used as `type Error` because it does not implement `std::error::Error`
//! (deliberate choice in the anyhow crate). A bare
//! `Box<dyn std::error::Error + Send + Sync>` cannot be used either: std only
//! provides `impl<T: Error + Sized> Error for Box<T>`, and the unsized trait
//! object does not satisfy it — so the boxed error is *not* itself an
//! `std::error::Error`, which the trait's associated-type bound requires.
//! [`BoxError`] is therefore a concrete wrapper that does implement the trait.
//! Wrap your native error at the boundary:
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
//!         self.0.execute(sql, params).await.map_err(BoxError::from)
//!     }
//!     async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
//!         self.0.query(sql, params).await.map_err(BoxError::from)
//!     }
//!     fn backend(&self) -> RelationalBackend { self.0.backend() }
//! }
//! ```

pub mod blocking;
#[cfg(feature = "postgres")]
pub mod postgres;
pub mod sqlite;
pub mod store;
#[cfg(feature = "postgres")]
pub mod supabase;
pub mod types;

pub use blocking::BlockingRelationalStore;
#[cfg(feature = "postgres")]
pub use postgres::PostgresRelationalStore;
pub use sqlite::SqliteRelationalStore;
pub use store::RelationalStore;
#[cfg(feature = "postgres")]
pub use supabase::SupabaseRelationalStore;
pub use types::{
    ColumnError, ColumnMeta, ColumnValue, FromColumn, OwnedColumn, Param, RelationalBackend, Row,
};

/// Type-erased error type for `Arc<dyn RelationalStore<Error = ...>>` bindings.
///
/// The trait's associated `Error` bound is `std::error::Error + Send + Sync +
/// 'static`. Two tempting candidates both fail that bound:
///
/// - `anyhow::Error` does NOT implement `std::error::Error` (a deliberate
///   choice in the anyhow crate).
/// - A bare `Box<dyn std::error::Error + Send + Sync + 'static>` does NOT
///   implement `std::error::Error` either: std only provides
///   `impl<T: Error + Sized> Error for Box<T>`, and the unsized trait object
///   `dyn Error + Send + Sync` is not `Sized`, so the blanket does not apply.
///
/// `BoxError` is therefore a concrete newtype wrapping the boxed error. It
/// implements `std::error::Error` (delegating `Display` and `source` to the
/// inner box), so it satisfies the associated-type bound and works with
/// `anyhow`'s `Context`. Backends adapt their native error at the boundary
/// with `.map_err(BoxError::new)`.
#[derive(Debug)]
pub struct BoxError(pub Box<dyn std::error::Error + Send + Sync + 'static>);

impl BoxError {
    /// Wrap any native error in a `BoxError`.
    ///
    /// This is the boundary adapter backends use:
    /// `native_call().await.map_err(BoxError::new)`.
    pub fn new<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        BoxError(Box::new(error))
    }
}

impl std::fmt::Display for BoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for BoxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl From<Box<dyn std::error::Error + Send + Sync + 'static>> for BoxError {
    fn from(error: Box<dyn std::error::Error + Send + Sync + 'static>) -> Self {
        BoxError(error)
    }
}
