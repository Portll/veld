//! Backend-agnostic relational storage — the W4 foundation.
//!
//! This module defines the [`RelationalStore`] trait and supporting value
//! types ([`Param`], [`Row`], [`ColumnValue`], [`ColumnError`], [`ColumnMeta`],
//! and the [`FromColumn`] decoder trait), along with the SQLite-backed
//! adapter [`SqliteRelationalStore`].
//!
//! Postgres / Supabase / MSSQL implementations land in follow-up commits.
//! Call-site ports of the existing rusqlite-backed slow stores also land
//! separately — this module is intentionally additive so other agents can
//! build against a stable surface.

pub mod sqlite;
pub mod store;
pub mod types;

pub use sqlite::SqliteRelationalStore;
pub use store::RelationalStore;
pub use types::{ColumnError, ColumnMeta, ColumnValue, FromColumn, Param, RelationalBackend, Row};
