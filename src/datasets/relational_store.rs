//! Relational-backed [`DatasetStore`].
//!
//! [`RelationalDatasetStore`] persists dataset rows directly in a
//! [`RelationalStore`] (SQLite today; Postgres / Supabase / MSSQL when those
//! backends land). Each user-defined dataset becomes a dedicated table
//! created via [`DatasetSchema::to_create_table_sql_sqlite`] /
//! [`DatasetSchema::to_create_table_sql_postgres`]; a single catalog table
//! (`__veld_dataset_catalog`) records `(user_id, name, table_name,
//! schema_json, created_at)` so [`DatasetStore::list_datasets`] can be
//! served without scanning `sqlite_master` / `information_schema`.
//!
//! ## Tenant isolation
//!
//! Every mutating or reading op rejects cross-tenant access by checking the
//! `(user_id, name)` tuple against the catalog before touching the data
//! table. A mismatch returns [`DatasetError::NotFound`] — never
//! [`DatasetError::TenantIsolation`] — so the existence of a sibling
//! tenant's dataset is not leaked.
//!
//! ## SQL injection
//!
//! Table and column names are interpolated as identifiers, but only after
//! [`sanitise_sql_identifier`] / [`sanitise_table_name`] have stripped
//! everything outside `[a-z0-9_]`. All *values* travel through
//! [`Param`] binding — never string-formatted into the SQL. The
//! handler-layer query endpoint that accepts a user-supplied `WHERE`
//! fragment honours the same rule: the fragment is embedded verbatim, but
//! every bound value flows through [`Param`].

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;

use crate::datasets::schema::DatasetSchema;
use crate::datasets::store::{
    sanitise_sql_identifier, sanitise_table_name, DatasetError, DatasetMeta, DatasetRef,
    DatasetRow, DatasetStore,
};
use crate::storage::relational::{Param, RelationalBackend, RelationalStore};

/// Name of the per-database catalog table that records dataset metadata.
pub const CATALOG_TABLE: &str = "__veld_dataset_catalog";

/// DDL for the catalog table. Applied idempotently on every construction so
/// a fresh database is always ready for dataset operations.
const CATALOG_DDL: &str = "CREATE TABLE IF NOT EXISTS __veld_dataset_catalog (\n    \
     user_id TEXT NOT NULL,\n    \
     name TEXT NOT NULL,\n    \
     table_name TEXT NOT NULL,\n    \
     schema_json TEXT NOT NULL,\n    \
     created_at TEXT NOT NULL,\n    \
     PRIMARY KEY (user_id, name)\n\
     );";

/// [`DatasetStore`] backed by any [`RelationalStore`] whose error type is
/// `anyhow::Error`. Use
/// [`crate::storage::relational::SqliteRelationalStore`] as the SQLite
/// implementation; downstream backends can be wrapped via an adapter that
/// maps the native error to `anyhow::Error` (see the docstring example on
/// [`crate::storage::relational`]).
pub struct RelationalDatasetStore {
    store: Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>,
    catalog_table: &'static str,
}

impl RelationalDatasetStore {
    /// Build a store over `store` and ensure the catalog table exists.
    ///
    /// The catalog DDL is `CREATE TABLE IF NOT EXISTS`, so calling this
    /// twice on the same database is a no-op.
    pub async fn new(
        store: Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>,
    ) -> Result<Self, DatasetError> {
        store
            .execute(CATALOG_DDL, &[])
            .await
            .map_err(|e| DatasetError::Internal(format!("catalog DDL failed: {e}")))?;
        Ok(Self {
            store,
            catalog_table: CATALOG_TABLE,
        })
    }

    /// Name of the catalog table this store reads / writes.
    pub fn catalog_table(&self) -> &'static str {
        self.catalog_table
    }

    /// Look up `(user_id, name)` in the catalog. Returns
    /// `(table_name, schema, created_at_iso)` when present, `None` otherwise.
    ///
    /// This is the single tenant-isolation enforcement point used by every
    /// per-dataset operation: drop / insert / count / link / query all start
    /// here, and they all map a `None` to [`DatasetError::NotFound`].
    async fn catalog_lookup(
        &self,
        user_id: &str,
        name: &str,
    ) -> Result<Option<(String, DatasetSchema, String)>, DatasetError> {
        let sql = format!(
            "SELECT table_name, schema_json, created_at FROM {} WHERE user_id = ? AND name = ?",
            self.catalog_table
        );
        let rows = self
            .store
            .query(&sql, &[Param::Text(user_id), Param::Text(name)])
            .await
            .map_err(|e| DatasetError::Internal(format!("catalog lookup failed: {e}")))?;

        let row = match rows.first() {
            Some(r) => r,
            None => return Ok(None),
        };
        let table_name: String = row
            .get(0)
            .map_err(|e| DatasetError::Internal(format!("catalog decode table_name: {e}")))?;
        let schema_json: String = row
            .get(1)
            .map_err(|e| DatasetError::Internal(format!("catalog decode schema_json: {e}")))?;
        let created_at: String = row
            .get(2)
            .map_err(|e| DatasetError::Internal(format!("catalog decode created_at: {e}")))?;
        let schema: DatasetSchema = serde_json::from_str(&schema_json)
            .map_err(|e| DatasetError::Internal(format!("catalog schema deserialise: {e}")))?;
        Ok(Some((table_name, schema, created_at)))
    }

    /// Borrow the underlying executor — useful for adjacent stores that need
    /// to issue their own statements against the same database (e.g.
    /// [`crate::datasets::link_store::RelationalLinkStore`]).
    pub fn store(&self) -> Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>> {
        self.store.clone()
    }
}

/// Render a JSON value into the borrowed [`Param`] that will bind it.
///
/// Bytes are not supported through JSON ingest at the row level — callers
/// who need to round-trip blobs should base64-encode and store as `Text`.
fn json_to_param<'a>(value: &'a serde_json::Value) -> Result<Param<'a>, DatasetError> {
    match value {
        serde_json::Value::Null => Ok(Param::Null),
        serde_json::Value::Bool(b) => Ok(Param::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Param::I64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Param::F64(f))
            } else {
                Err(DatasetError::SchemaViolation(format!(
                    "unsupported numeric value {n}"
                )))
            }
        }
        serde_json::Value::String(s) => Ok(Param::Text(s.as_str())),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => Ok(Param::Json(value)),
    }
}

#[async_trait]
impl DatasetStore for RelationalDatasetStore {
    async fn create_dataset(
        &self,
        user_id: &str,
        schema: &DatasetSchema,
    ) -> Result<DatasetRef, DatasetError> {
        if user_id.is_empty() {
            return Err(DatasetError::Internal("user_id must not be empty".into()));
        }
        if schema.name.is_empty() {
            return Err(DatasetError::SchemaViolation(
                "dataset name must not be empty".into(),
            ));
        }

        let declared: std::collections::HashSet<&str> =
            schema.columns.iter().map(|c| c.name.as_str()).collect();
        if declared.len() != schema.columns.len() {
            return Err(DatasetError::SchemaViolation(
                "duplicate column names in schema".into(),
            ));
        }
        for pk in &schema.primary_key {
            if !declared.contains(pk.as_str()) {
                return Err(DatasetError::SchemaViolation(format!(
                    "primary key column '{pk}' not in schema"
                )));
            }
        }
        for col in &schema.columns {
            let sanitised = sanitise_sql_identifier(&col.name);
            if sanitised != col.name {
                return Err(DatasetError::SchemaViolation(format!(
                    "column name '{}' contains characters outside [a-z0-9_]",
                    col.name
                )));
            }
            if col.name.is_empty() {
                return Err(DatasetError::SchemaViolation(
                    "column name must not be empty".into(),
                ));
            }
        }

        // Reject a duplicate up-front. The catalog has a composite PK on
        // (user_id, name), so an INSERT would fail anyway — but the catalog
        // error path is opaque (driver-specific); a structured check first
        // gives callers a clean `AlreadyExists`.
        if self
            .catalog_lookup(user_id, &schema.name)
            .await?
            .is_some()
        {
            return Err(DatasetError::AlreadyExists(schema.name.clone()));
        }

        let table = sanitise_table_name(user_id, &schema.name);

        // Render the data-table DDL per backend dialect. Both renderers
        // produce a `CREATE TABLE IF NOT EXISTS` so a previously-created
        // table (from a partially-failed earlier insertion) is preserved.
        let ddl = match self.store.backend() {
            RelationalBackend::Postgres | RelationalBackend::Supabase => {
                schema.to_create_table_sql_postgres(&table)
            }
            _ => schema.to_create_table_sql_sqlite(&table),
        };
        self.store
            .execute(&ddl, &[])
            .await
            .map_err(|e| DatasetError::Internal(format!("create table failed: {e}")))?;

        let schema_json = serde_json::to_string(schema)
            .map_err(|e| DatasetError::Internal(format!("encode schema: {e}")))?;
        let created_at = Utc::now().to_rfc3339();
        let insert_sql = format!(
            "INSERT INTO {} (user_id, name, table_name, schema_json, created_at) \
             VALUES (?, ?, ?, ?, ?)",
            self.catalog_table
        );
        self.store
            .execute(
                &insert_sql,
                &[
                    Param::Text(user_id),
                    Param::Text(&schema.name),
                    Param::Text(&table),
                    Param::Text(&schema_json),
                    Param::Text(&created_at),
                ],
            )
            .await
            .map_err(|e| DatasetError::Internal(format!("catalog insert failed: {e}")))?;

        Ok(DatasetRef {
            user_id: user_id.to_string(),
            name: schema.name.clone(),
            table,
        })
    }

    async fn drop_dataset(&self, dataset: &DatasetRef) -> Result<(), DatasetError> {
        let (table, _schema, _ts) = match self
            .catalog_lookup(&dataset.user_id, &dataset.name)
            .await?
        {
            Some(found) => found,
            None => return Err(DatasetError::NotFound(dataset.name.clone())),
        };

        // Drop the data table first; if the catalog row remains after a
        // failed DROP, a retry will eventually clean both up. The reverse
        // would orphan the table.
        let drop_sql = format!("DROP TABLE IF EXISTS \"{table}\"");
        self.store
            .execute(&drop_sql, &[])
            .await
            .map_err(|e| DatasetError::Internal(format!("drop table failed: {e}")))?;

        let delete_sql = format!(
            "DELETE FROM {} WHERE user_id = ? AND name = ?",
            self.catalog_table
        );
        self.store
            .execute(
                &delete_sql,
                &[Param::Text(&dataset.user_id), Param::Text(&dataset.name)],
            )
            .await
            .map_err(|e| DatasetError::Internal(format!("catalog delete failed: {e}")))?;

        Ok(())
    }

    async fn insert_rows(
        &self,
        dataset: &DatasetRef,
        rows: &[DatasetRow],
    ) -> Result<u64, DatasetError> {
        if rows.is_empty() {
            return Ok(0);
        }

        let (table, schema, _ts) = match self
            .catalog_lookup(&dataset.user_id, &dataset.name)
            .await?
        {
            Some(found) => found,
            None => return Err(DatasetError::NotFound(dataset.name.clone())),
        };

        // Validate every row's columns against the schema. We use the
        // schema's declared column order to drive the INSERT, so missing
        // non-nullable columns are caught here rather than at the driver.
        let declared: std::collections::HashSet<&str> =
            schema.columns.iter().map(|c| c.name.as_str()).collect();
        for row in rows {
            for key in row.values.keys() {
                if !declared.contains(key.as_str()) {
                    return Err(DatasetError::SchemaViolation(format!(
                        "unknown column '{key}'"
                    )));
                }
            }
            for col in &schema.columns {
                match row.values.get(&col.name) {
                    None => {
                        if !col.nullable {
                            return Err(DatasetError::SchemaViolation(format!(
                                "missing required column '{}'",
                                col.name
                            )));
                        }
                    }
                    Some(v) => {
                        if v.is_null() && !col.nullable {
                            return Err(DatasetError::SchemaViolation(format!(
                                "column '{}' is not nullable",
                                col.name
                            )));
                        }
                    }
                }
            }
        }

        // Build the column list + placeholders once. `?` placeholders are
        // SQLite-native; sqlx translates them for Postgres into `$N`.
        let mut col_list = String::new();
        let mut placeholders = String::new();
        for (idx, col) in schema.columns.iter().enumerate() {
            if idx > 0 {
                col_list.push_str(", ");
                placeholders.push_str(", ");
            }
            col_list.push('"');
            col_list.push_str(&col.name);
            col_list.push('"');
            placeholders.push('?');
        }
        let insert_sql = format!(
            "INSERT INTO \"{table}\" ({col_list}) VALUES ({placeholders})"
        );

        // Execute one prepared INSERT per row. A single multi-row INSERT
        // would be more efficient on SQLite, but per-row keeps the bind
        // vector small and lets us surface a row-specific error message if
        // any of them fail. The hot path for ingest is a follow-up; this
        // implementation prioritises clarity and correctness.
        let mut inserted = 0u64;
        for row in rows {
            let mut params: Vec<Param<'_>> = Vec::with_capacity(schema.columns.len());
            for col in &schema.columns {
                let p = match row.values.get(&col.name) {
                    Some(v) => json_to_param(v)?,
                    None => Param::Null,
                };
                params.push(p);
            }
            let affected = self
                .store
                .execute(&insert_sql, &params)
                .await
                .map_err(|e| DatasetError::Internal(format!("insert failed: {e}")))?;
            inserted += affected;
        }

        Ok(inserted)
    }

    async fn count_rows(&self, dataset: &DatasetRef) -> Result<u64, DatasetError> {
        let (table, _schema, _ts) = match self
            .catalog_lookup(&dataset.user_id, &dataset.name)
            .await?
        {
            Some(found) => found,
            None => return Err(DatasetError::NotFound(dataset.name.clone())),
        };
        let sql = format!("SELECT COUNT(*) FROM \"{table}\"");
        let rows = self
            .store
            .query(&sql, &[])
            .await
            .map_err(|e| DatasetError::Internal(format!("count query failed: {e}")))?;
        let row = rows.first().ok_or_else(|| {
            DatasetError::Internal("count query returned no rows".into())
        })?;
        let count: i64 = row
            .get(0)
            .map_err(|e| DatasetError::Internal(format!("count decode: {e}")))?;
        Ok(count.max(0) as u64)
    }

    async fn list_datasets(&self, user_id: &str) -> Result<Vec<DatasetMeta>, DatasetError> {
        let sql = format!(
            "SELECT name, schema_json, created_at, table_name FROM {} \
             WHERE user_id = ? ORDER BY name",
            self.catalog_table
        );
        let rows = self
            .store
            .query(&sql, &[Param::Text(user_id)])
            .await
            .map_err(|e| DatasetError::Internal(format!("list query failed: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let name: String = row
                .get(0)
                .map_err(|e| DatasetError::Internal(format!("list decode name: {e}")))?;
            let schema_json: String = row
                .get(1)
                .map_err(|e| DatasetError::Internal(format!("list decode schema: {e}")))?;
            let created_at_str: String = row
                .get(2)
                .map_err(|e| DatasetError::Internal(format!("list decode created_at: {e}")))?;
            let table_name: String = row
                .get(3)
                .map_err(|e| DatasetError::Internal(format!("list decode table_name: {e}")))?;

            let schema: DatasetSchema = serde_json::from_str(&schema_json).map_err(|e| {
                DatasetError::Internal(format!("list decode schema_json: {e}"))
            })?;
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| {
                    DatasetError::Internal(format!("list decode created_at parse: {e}"))
                })?
                .with_timezone(&chrono::Utc);

            let count_sql = format!("SELECT COUNT(*) FROM \"{table_name}\"");
            let count_rows = self
                .store
                .query(&count_sql, &[])
                .await
                .map_err(|e| DatasetError::Internal(format!("list count query: {e}")))?;
            let row_count = count_rows
                .first()
                .and_then(|r| r.get::<i64>(0).ok())
                .unwrap_or(0)
                .max(0) as u64;

            out.push(DatasetMeta {
                name,
                schema,
                row_count,
                created_at,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::schema::{ColumnDef, ColumnType};
    use crate::storage::relational::{BoxError, SqliteRelationalStore};
    use async_trait::async_trait;
    use std::collections::HashMap;

    // ---------------------------------------------------------------------
    // Test plumbing: wrap the native sqlx error of `SqliteRelationalStore`
    // into `BoxError` so the store satisfies
    // `RelationalStore<Error = BoxError>`, the bound that
    // `RelationalDatasetStore` requires (anyhow::Error can't be used as
    // the type-erased error because it doesn't implement std::error::Error).
    // ---------------------------------------------------------------------

    struct BoxErrorSqlite(SqliteRelationalStore);

    #[async_trait]
    impl RelationalStore for BoxErrorSqlite {
        type Error = BoxError;

        async fn execute(
            &self,
            sql: &str,
            params: &[Param<'_>],
        ) -> Result<u64, BoxError> {
            self.0
                .execute(sql, params)
                .await
                .map_err(BoxError::new)
        }

        async fn query(
            &self,
            sql: &str,
            params: &[Param<'_>],
        ) -> Result<Vec<crate::storage::relational::Row>, BoxError> {
            self.0
                .query(sql, params)
                .await
                .map_err(BoxError::new)
        }

        fn backend(&self) -> RelationalBackend {
            self.0.backend()
        }
    }

    async fn fresh_store() -> RelationalDatasetStore {
        let sqlite = SqliteRelationalStore::in_memory()
            .await
            .expect("open in-memory sqlite");
        let store: Arc<dyn RelationalStore<Error = BoxError>> =
            Arc::new(BoxErrorSqlite(sqlite));
        RelationalDatasetStore::new(store)
            .await
            .expect("init relational dataset store")
    }

    fn sample_schema(name: &str) -> DatasetSchema {
        DatasetSchema {
            name: name.to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: ColumnType::I64,
                    nullable: false,
                },
                ColumnDef {
                    name: "label".to_string(),
                    ty: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "note".to_string(),
                    ty: ColumnType::Text,
                    nullable: true,
                },
            ],
            primary_key: vec!["id".to_string()],
        }
    }

    fn row(id: i64, label: &str) -> DatasetRow {
        let mut values = HashMap::new();
        values.insert("id".to_string(), serde_json::json!(id));
        values.insert("label".to_string(), serde_json::json!(label));
        DatasetRow { values }
    }

    #[tokio::test]
    async fn end_to_end_create_insert_count_list_drop() {
        let ds = fresh_store().await;

        let dref = ds
            .create_dataset("alice", &sample_schema("events"))
            .await
            .expect("create");
        assert_eq!(dref.user_id, "alice");
        assert_eq!(dref.name, "events");
        assert_eq!(dref.table, "alice__dataset__events");

        let inserted = ds
            .insert_rows(
                &dref,
                &[
                    row(1, "a"),
                    row(2, "b"),
                    row(3, "c"),
                    row(4, "d"),
                    row(5, "e"),
                ],
            )
            .await
            .expect("insert");
        assert_eq!(inserted, 5);

        let count = ds.count_rows(&dref).await.expect("count");
        assert_eq!(count, 5);

        let listed = ds.list_datasets("alice").await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "events");
        assert_eq!(listed[0].row_count, 5);

        ds.drop_dataset(&dref).await.expect("drop");
        let after = ds.list_datasets("alice").await.expect("list after drop");
        assert!(after.is_empty(), "drop should remove from catalog");
    }

    #[tokio::test]
    async fn tenant_isolation_rejects_cross_tenant_access() {
        let ds = fresh_store().await;
        let alice_ref = ds
            .create_dataset("alice", &sample_schema("foo"))
            .await
            .expect("alice create");
        ds.insert_rows(&alice_ref, &[row(1, "secret")])
            .await
            .expect("alice insert");

        // Bob tries to read alice's dataset by forging the ref.
        let spoof = DatasetRef {
            user_id: "bob".to_string(),
            name: "foo".to_string(),
            table: alice_ref.table.clone(),
        };
        match ds.count_rows(&spoof).await {
            Err(DatasetError::NotFound(_)) => {}
            other => panic!("expected NotFound for cross-tenant read, got {other:?}"),
        }
        match ds.drop_dataset(&spoof).await {
            Err(DatasetError::NotFound(_)) => {}
            other => panic!("expected NotFound for cross-tenant drop, got {other:?}"),
        }
        match ds
            .insert_rows(&spoof, &[row(99, "evil")])
            .await
        {
            Err(DatasetError::NotFound(_)) => {}
            other => panic!("expected NotFound for cross-tenant insert, got {other:?}"),
        }

        // Bob sees no datasets of his own.
        let bob_list = ds.list_datasets("bob").await.expect("bob list");
        assert!(bob_list.is_empty());

        // Alice's data survives.
        let alice_count = ds.count_rows(&alice_ref).await.expect("alice count");
        assert_eq!(alice_count, 1);
    }

    #[tokio::test]
    async fn create_dataset_rejects_duplicate() {
        let ds = fresh_store().await;
        ds.create_dataset("alice", &sample_schema("dup"))
            .await
            .expect("first create");
        let again = ds
            .create_dataset("alice", &sample_schema("dup"))
            .await
            .expect_err("second create should fail");
        match again {
            DatasetError::AlreadyExists(name) => assert_eq!(name, "dup"),
            other => panic!("expected AlreadyExists, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn insert_rejects_missing_required_column() {
        let ds = fresh_store().await;
        let dref = ds
            .create_dataset("alice", &sample_schema("events"))
            .await
            .expect("create");

        // Row missing the required 'label' column.
        let mut bad = HashMap::new();
        bad.insert("id".to_string(), serde_json::json!(1));
        let err = ds
            .insert_rows(&dref, &[DatasetRow { values: bad }])
            .await
            .expect_err("should reject");
        match err {
            DatasetError::SchemaViolation(msg) => {
                assert!(msg.contains("label"), "should name column: {msg}");
            }
            other => panic!("expected SchemaViolation, got {other:?}"),
        }

        // Table should remain empty after the rejection.
        assert_eq!(ds.count_rows(&dref).await.expect("count"), 0);
    }
}
