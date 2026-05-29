//! Datasets — structured tabular storage with row-level graph links.
//!
//! W7 foundation: this module defines the *data model* and *trait surface*
//! for datasets. The current backing implementation
//! ([`store::InMemoryDatasetStore`]) is intended for tests and for callers
//! in dependent modules that don't yet have a relational store wired in.
//! A follow-up agent wires the trait to a real `RelationalStore`, and a
//! third lands HTTP / MCP / ingest integration.
//!
//! Top-level types:
//!
//! - [`schema::DatasetSchema`] / [`schema::ColumnDef`] / [`schema::ColumnType`]
//!   describe the structural shape of a dataset and render dialect-specific
//!   DDL strings (SQLite + Postgres).
//! - [`store::DatasetStore`] is the async storage interface; every method
//!   is bounded by tenant (`user_id`) and refuses cross-tenant access.
//! - [`link::RowLink`] connects a specific row (identified by primary-key
//!   values) to either a knowledge-graph entity or a memory record.

pub mod link;
pub mod link_store;
pub mod relational_store;
pub mod schema;
pub mod store;

pub use link::{LinkKind, RowLink, RowPk};
pub use link_store::{LinkStore, RelationalLinkStore, LINK_TABLE};
pub use relational_store::{RelationalDatasetStore, CATALOG_TABLE};
pub use schema::{ColumnDef, ColumnType, DatasetSchema};
pub use store::{
    DatasetError, DatasetMeta, DatasetRef, DatasetRow, DatasetStore, InMemoryDatasetStore,
    MAX_TABLE_NAME_LEN, sanitise_sql_identifier, sanitise_table_name,
};
