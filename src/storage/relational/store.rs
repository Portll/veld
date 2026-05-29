//! `RelationalStore` — the backend-agnostic relational executor trait.
//!
//! This is the W4 foundation. The trait surface is intentionally minimal:
//! `execute`, `query`, and a `backend()` discriminator. A `transaction`
//! method will be added in a follow-up commit once the executor patterns
//! are settled and the call-site port is in flight.
//!
//! Backends today: [`crate::storage::relational::sqlite::SqliteRelationalStore`].
//! Backends queued: Postgres, Supabase, MSSQL.

use async_trait::async_trait;

use super::types::{Param, RelationalBackend, Row};

/// Backend-agnostic relational executor.
///
/// Implementations wrap a native pool (e.g. `sqlx::SqlitePool`,
/// `sqlx::PgPool`) and translate the borrowed [`Param`] variants into native
/// bind calls. Returning [`Row`] values keeps query results portable across
/// engines.
///
/// ## Error type
///
/// Each backend uses its native error type (e.g. `sqlx::Error`). Callers
/// that need to be backend-generic should box the error or wrap with
/// `anyhow::Error`.
#[async_trait]
pub trait RelationalStore: Send + Sync {
    /// Backend-native error type.
    ///
    /// Bound is intentionally weak: `anyhow::Error` is the documented type
    /// for the application-facing `Arc<dyn RelationalStore<Error =
    /// anyhow::Error>>` form, and `anyhow::Error` does not implement
    /// `std::error::Error` (to keep `?` propagation unambiguous in
    /// application code that already uses `anyhow::Result`). Concrete
    /// backends can still use their native error type; consumers that
    /// need `std::error::Error` should construct their own newtype.
    type Error: Send + Sync + 'static;

    /// Execute a non-result-returning statement. Returns the number of rows
    /// affected (where the backend reports it).
    async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, Self::Error>;

    /// Execute a result-returning statement and collect every row.
    ///
    /// Backends should not buffer beyond what the underlying driver requires;
    /// for streaming, callers should use the backend directly until a streaming
    /// API is added to this trait.
    async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, Self::Error>;

    /// Discriminator for the concrete backend behind this store.
    fn backend(&self) -> RelationalBackend;
}
