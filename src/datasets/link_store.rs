//! Row-link store — persists [`RowLink`] records and supports reverse
//! lookups by graph target.
//!
//! The wire types ([`RowLink`], [`LinkKind`], [`RowPk`]) live in
//! [`crate::datasets::link`]; this module owns the *storage* side.
//!
//! Links are stored in a single dedicated table
//! (`__veld_dataset_row_links`) keyed by `(dataset_table, row_pk_json,
//! kind, target_id)`. `row_pk_json` is the canonical JSON encoding of the
//! row's primary-key tuple so composite keys round-trip across the storage
//! boundary without a custom encoding.
//!
//! ## Tenant isolation
//!
//! `(dataset_table)` already encodes the tenant via the sanitised
//! `{user}__dataset__{name}` shape, so a caller cannot insert a link
//! targeting another tenant's dataset table without controlling the table
//! name. Higher layers (the HTTP handler) additionally verify that the
//! caller owns `dataset_table` via [`crate::datasets::store::DatasetStore`]
//! before reaching this surface.

use std::sync::Arc;

use async_trait::async_trait;

use crate::datasets::link::{LinkKind, RowLink, RowPk};
use crate::datasets::store::{DatasetError, DatasetRef};
use crate::storage::relational::{Param, RelationalBackend, RelationalStore};

/// Table name for the link store.
pub const LINK_TABLE: &str = "__veld_dataset_row_links";

/// DDL applied idempotently when the store is constructed (SQLite + Postgres).
///
/// The composite primary key gives idempotency for free: a re-link of the
/// same `(table, row, kind, target)` tuple is a no-op insert. `TEXT` columns
/// carry no length limit on either engine, so `row_pk_json` (an arbitrary
/// JSON-encoded key tuple) can participate in the key directly.
const LINK_DDL: &str = "CREATE TABLE IF NOT EXISTS __veld_dataset_row_links (\n    \
     dataset_table TEXT NOT NULL,\n    \
     row_pk_json TEXT NOT NULL,\n    \
     kind TEXT NOT NULL,\n    \
     target_id TEXT NOT NULL,\n    \
     PRIMARY KEY (dataset_table, row_pk_json, kind, target_id)\n\
     );";

/// Link DDL for SQL Server.
///
/// `row_pk_json` is an *unbounded* JSON-encoded key tuple, so it cannot join
/// a SQL Server primary key or index (which cap at 900 bytes and forbid
/// `NVARCHAR(MAX)` key columns). The MSSQL link table is therefore a plain
/// heap with no key constraint; idempotency is enforced instead by the
/// `IF NOT EXISTS` guard in [`RelationalLinkStore::insert`]. Link volumes per
/// dataset are small, so the reverse-lookup `SELECT` scanning the heap is
/// acceptable — this is the secondary backend, not the hot retrieval path.
const LINK_DDL_MSSQL: &str = "IF OBJECT_ID(N'__veld_dataset_row_links', N'U') IS NULL\n\
     CREATE TABLE __veld_dataset_row_links (\n    \
     dataset_table NVARCHAR(450) NOT NULL,\n    \
     row_pk_json NVARCHAR(MAX) NOT NULL,\n    \
     kind NVARCHAR(32) NOT NULL,\n    \
     target_id NVARCHAR(450) NOT NULL\n\
     );";

/// Select the link DDL for the backend's SQL dialect.
fn link_ddl(backend: RelationalBackend) -> &'static str {
    match backend {
        RelationalBackend::Mssql => LINK_DDL_MSSQL,
        _ => LINK_DDL,
    }
}

fn kind_to_str(kind: LinkKind) -> &'static str {
    match kind {
        LinkKind::Entity => "entity",
        LinkKind::Memory => "memory",
    }
}

#[allow(dead_code)] // W7 datasets: link-kind parser, wired as link ingestion lands
fn kind_from_str(s: &str) -> Result<LinkKind, DatasetError> {
    match s {
        "entity" => Ok(LinkKind::Entity),
        "memory" => Ok(LinkKind::Memory),
        other => Err(DatasetError::Internal(format!(
            "unknown link kind '{other}'"
        ))),
    }
}

/// Encode a [`RowPk`] into the canonical JSON-array form used as the
/// storage key. Round-trip with [`decode_row_pk`].
fn encode_row_pk(pk: &RowPk) -> Result<String, DatasetError> {
    serde_json::to_string(&pk.values)
        .map_err(|e| DatasetError::Internal(format!("encode row_pk: {e}")))
}

fn decode_row_pk(s: &str) -> Result<RowPk, DatasetError> {
    let values: Vec<serde_json::Value> = serde_json::from_str(s)
        .map_err(|e| DatasetError::Internal(format!("decode row_pk: {e}")))?;
    Ok(RowPk { values })
}

/// Storage interface for [`RowLink`] records.
///
/// All operations are scoped to a single dataset table (and therefore a
/// single tenant — see the module-level docstring). The trait does not
/// itself enforce that the caller owns the dataset; that check happens
/// at the HTTP handler boundary via the [`DatasetStore`] catalog.
#[async_trait]
pub trait LinkStore: Send + Sync {
    /// Create a link from a dataset row to a knowledge-graph entity.
    async fn link_row_to_entity(
        &self,
        dataset: &DatasetRef,
        row_pk: &RowPk,
        entity_id: &str,
    ) -> Result<(), DatasetError>;

    /// Create a link from a dataset row to a memory record.
    async fn link_row_to_memory(
        &self,
        dataset: &DatasetRef,
        row_pk: &RowPk,
        memory_id: &str,
    ) -> Result<(), DatasetError>;

    /// List every row currently linked to `entity_id`. Returned [`RowLink`]
    /// records carry the [`DatasetRef`] supplied by the caller — we do not
    /// have enough information at the link layer to reconstruct the owning
    /// `(user_id, name)` tuple, so the caller must scope the query to a
    /// dataset they own. (`dataset.user_id` is informational on the
    /// returned link.)
    async fn rows_for_entity(
        &self,
        dataset: &DatasetRef,
        entity_id: &str,
    ) -> Result<Vec<RowLink>, DatasetError>;

    /// List every row currently linked to `memory_id`. See
    /// [`Self::rows_for_entity`] for the scoping contract.
    async fn rows_for_memory(
        &self,
        dataset: &DatasetRef,
        memory_id: &str,
    ) -> Result<Vec<RowLink>, DatasetError>;
}

/// Relational implementation of [`LinkStore`].
///
/// Backed by any [`RelationalStore`] with an `anyhow::Error` error type so
/// it can be composed with [`crate::datasets::relational_store::RelationalDatasetStore`]
/// over the same backing database.
pub struct RelationalLinkStore {
    store: Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>,
    link_table: &'static str,
}

impl RelationalLinkStore {
    /// Build a link store and ensure the link table exists. Idempotent —
    /// safe to construct repeatedly against the same database.
    pub async fn new(
        store: Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>,
    ) -> Result<Self, DatasetError> {
        store
            .execute(link_ddl(store.backend()), &[])
            .await
            .map_err(|e| DatasetError::Internal(format!("link DDL failed: {e}")))?;
        Ok(Self {
            store,
            link_table: LINK_TABLE,
        })
    }

    /// Name of the link table.
    pub fn link_table(&self) -> &'static str {
        self.link_table
    }

    async fn insert(
        &self,
        dataset: &DatasetRef,
        row_pk: &RowPk,
        kind: LinkKind,
        target_id: &str,
    ) -> Result<(), DatasetError> {
        let pk_json = encode_row_pk(row_pk)?;
        let kind_str = kind_to_str(kind);
        let t = self.link_table;
        // Idempotent insert — re-linking the same row to the same target is a
        // no-op — but the "ignore the duplicate" syntax is dialect-specific.
        // SQLite has `INSERT OR IGNORE`; Postgres uses `ON CONFLICT DO
        // NOTHING` against the composite primary key; SQL Server has neither,
        // so it guards the insert with `IF NOT EXISTS` (its link table is a
        // keyless heap — see [`LINK_DDL_MSSQL`]).
        let result = match self.store.backend() {
            RelationalBackend::Postgres | RelationalBackend::Supabase => {
                let sql = format!(
                    "INSERT INTO {t} (dataset_table, row_pk_json, kind, target_id) \
                     VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING"
                );
                self.store
                    .execute(
                        &sql,
                        &[
                            Param::Text(&dataset.table),
                            Param::Text(&pk_json),
                            Param::Text(kind_str),
                            Param::Text(target_id),
                        ],
                    )
                    .await
            }
            RelationalBackend::Mssql => {
                let sql = format!(
                    "IF NOT EXISTS (SELECT 1 FROM {t} \
                     WHERE dataset_table = ? AND row_pk_json = ? AND kind = ? AND target_id = ?) \
                     INSERT INTO {t} (dataset_table, row_pk_json, kind, target_id) \
                     VALUES (?, ?, ?, ?)"
                );
                self.store
                    .execute(
                        &sql,
                        &[
                            // EXISTS probe …
                            Param::Text(&dataset.table),
                            Param::Text(&pk_json),
                            Param::Text(kind_str),
                            Param::Text(target_id),
                            // … then the INSERT values (same order).
                            Param::Text(&dataset.table),
                            Param::Text(&pk_json),
                            Param::Text(kind_str),
                            Param::Text(target_id),
                        ],
                    )
                    .await
            }
            _ => {
                let sql = format!(
                    "INSERT OR IGNORE INTO {t} (dataset_table, row_pk_json, kind, target_id) \
                     VALUES (?, ?, ?, ?)"
                );
                self.store
                    .execute(
                        &sql,
                        &[
                            Param::Text(&dataset.table),
                            Param::Text(&pk_json),
                            Param::Text(kind_str),
                            Param::Text(target_id),
                        ],
                    )
                    .await
            }
        };
        result.map_err(|e| DatasetError::Internal(format!("link insert failed: {e}")))?;
        Ok(())
    }

    async fn select_links(
        &self,
        dataset: &DatasetRef,
        kind: LinkKind,
        target_id: &str,
    ) -> Result<Vec<RowLink>, DatasetError> {
        let sql = format!(
            "SELECT row_pk_json FROM {} \
             WHERE dataset_table = ? AND kind = ? AND target_id = ? \
             ORDER BY row_pk_json",
            self.link_table
        );
        let rows = self
            .store
            .query(
                &sql,
                &[
                    Param::Text(&dataset.table),
                    Param::Text(kind_to_str(kind)),
                    Param::Text(target_id),
                ],
            )
            .await
            .map_err(|e| DatasetError::Internal(format!("link query failed: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let pk_json: String = row
                .get(0)
                .map_err(|e| DatasetError::Internal(format!("link decode pk: {e}")))?;
            let pk = decode_row_pk(&pk_json)?;
            out.push(RowLink {
                dataset: dataset.clone(),
                row_pk: pk,
                kind,
                target_id: target_id.to_string(),
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl LinkStore for RelationalLinkStore {
    async fn link_row_to_entity(
        &self,
        dataset: &DatasetRef,
        row_pk: &RowPk,
        entity_id: &str,
    ) -> Result<(), DatasetError> {
        self.insert(dataset, row_pk, LinkKind::Entity, entity_id).await
    }

    async fn link_row_to_memory(
        &self,
        dataset: &DatasetRef,
        row_pk: &RowPk,
        memory_id: &str,
    ) -> Result<(), DatasetError> {
        self.insert(dataset, row_pk, LinkKind::Memory, memory_id).await
    }

    async fn rows_for_entity(
        &self,
        dataset: &DatasetRef,
        entity_id: &str,
    ) -> Result<Vec<RowLink>, DatasetError> {
        self.select_links(dataset, LinkKind::Entity, entity_id).await
    }

    async fn rows_for_memory(
        &self,
        dataset: &DatasetRef,
        memory_id: &str,
    ) -> Result<Vec<RowLink>, DatasetError> {
        self.select_links(dataset, LinkKind::Memory, memory_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::relational::{BoxError, RelationalBackend, Row, SqliteRelationalStore};
    use async_trait::async_trait;

    struct BoxErrorSqlite(SqliteRelationalStore);

    #[async_trait]
    impl RelationalStore for BoxErrorSqlite {
        type Error = BoxError;
        async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, BoxError> {
            self.0
                .execute(sql, params)
                .await
                .map_err(BoxError::new)
        }
        async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
            self.0
                .query(sql, params)
                .await
                .map_err(BoxError::new)
        }
        fn backend(&self) -> RelationalBackend {
            self.0.backend()
        }
    }

    async fn fresh_link_store() -> RelationalLinkStore {
        let sqlite = SqliteRelationalStore::in_memory()
            .await
            .expect("open sqlite");
        let store: Arc<dyn RelationalStore<Error = BoxError>> =
            Arc::new(BoxErrorSqlite(sqlite));
        RelationalLinkStore::new(store).await.expect("init link store")
    }

    fn dref(table: &str) -> DatasetRef {
        DatasetRef {
            user_id: "alice".to_string(),
            name: "events".to_string(),
            table: table.to_string(),
        }
    }

    fn pk(id: i64) -> RowPk {
        RowPk {
            values: vec![serde_json::json!(id)],
        }
    }

    #[tokio::test]
    async fn link_three_rows_to_entity_and_query_back() {
        let ls = fresh_link_store().await;
        let d = dref("alice__dataset__events");
        let entity = "550e8400-e29b-41d4-a716-446655440000";

        ls.link_row_to_entity(&d, &pk(1), entity).await.expect("link 1");
        ls.link_row_to_entity(&d, &pk(2), entity).await.expect("link 2");
        ls.link_row_to_entity(&d, &pk(3), entity).await.expect("link 3");

        let rows = ls.rows_for_entity(&d, entity).await.expect("rows_for_entity");
        assert_eq!(rows.len(), 3);
        let ids: Vec<i64> = rows
            .iter()
            .map(|l| {
                l.row_pk
                    .values
                    .first()
                    .and_then(|v| v.as_i64())
                    .expect("i64 pk component")
            })
            .collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[tokio::test]
    async fn link_to_memory_disjoint_from_entity_results() {
        let ls = fresh_link_store().await;
        let d = dref("alice__dataset__events");
        let memory = "mem-1";
        let entity = "ent-1";

        ls.link_row_to_memory(&d, &pk(1), memory).await.expect("mem link");
        ls.link_row_to_entity(&d, &pk(2), entity).await.expect("ent link");

        let mem_rows = ls.rows_for_memory(&d, memory).await.expect("mem query");
        assert_eq!(mem_rows.len(), 1);
        let ent_rows = ls.rows_for_entity(&d, entity).await.expect("ent query");
        assert_eq!(ent_rows.len(), 1);

        // The memory query should not return entity links and vice versa.
        let cross_mem = ls.rows_for_memory(&d, entity).await.expect("cross mem");
        assert!(cross_mem.is_empty());
        let cross_ent = ls.rows_for_entity(&d, memory).await.expect("cross ent");
        assert!(cross_ent.is_empty());
    }

    #[tokio::test]
    async fn link_is_idempotent() {
        let ls = fresh_link_store().await;
        let d = dref("alice__dataset__events");
        let entity = "ent-1";

        ls.link_row_to_entity(&d, &pk(1), entity).await.expect("first");
        ls.link_row_to_entity(&d, &pk(1), entity).await.expect("second");

        let rows = ls.rows_for_entity(&d, entity).await.expect("query");
        assert_eq!(rows.len(), 1, "duplicate link should not produce duplicates");
    }
}
