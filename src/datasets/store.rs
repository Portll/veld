//! Dataset storage interface and an in-memory reference implementation.
//!
//! [`DatasetStore`] is the seam between dataset metadata management and
//! whatever relational engine (SQLite, Postgres, etc.) ultimately persists
//! rows. [`InMemoryDatasetStore`] implements the trait against a
//! [`dashmap::DashMap`] for tests and for callers in dependent modules
//! that don't yet have a real `RelationalStore` wired in.
//!
//! The in-memory store is deliberately strict about tenant isolation:
//! every operation is keyed by `(user_id, dataset_name)`, and a
//! cross-tenant access silently returns [`DatasetError::NotFound`] rather
//! than [`DatasetError::TenantIsolation`] to avoid leaking the existence
//! of a sibling tenant's dataset.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::datasets::schema::DatasetSchema;

/// Stable handle to a dataset across the API surface.
///
/// `table` is the sanitised SQL identifier produced by
/// [`sanitise_table_name`]; downstream callers should never derive the
/// table name themselves from `user_id` + `name`, since the sanitisation
/// rules are authoritative.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DatasetRef {
    pub user_id: String,
    pub name: String,
    pub table: String,
}

/// Catalog-level metadata about a dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetMeta {
    pub name: String,
    pub schema: DatasetSchema,
    pub row_count: u64,
    pub created_at: DateTime<Utc>,
}

/// A single row to be inserted.
///
/// Keys must exactly match the column names of the target schema. Missing
/// non-nullable columns are rejected with [`DatasetError::SchemaViolation`];
/// extra columns not declared in the schema are likewise rejected so that
/// schema drift surfaces as a hard error at the API boundary rather than
/// silent data loss.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetRow {
    pub values: HashMap<String, serde_json::Value>,
}

/// Errors returned by [`DatasetStore`] operations.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("dataset already exists: {0}")]
    AlreadyExists(String),
    #[error("dataset not found: {0}")]
    NotFound(String),
    #[error("schema violation: {0}")]
    SchemaViolation(String),
    #[error("tenant isolation violation")]
    TenantIsolation,
    #[error("internal error: {0}")]
    Internal(String),
}

/// Async storage interface for datasets.
///
/// All methods are per-tenant: every read or write is bounded by
/// `user_id`, either explicitly or via [`DatasetRef::user_id`].
/// Implementations must never let one tenant observe or mutate another
/// tenant's datasets.
#[async_trait]
pub trait DatasetStore: Send + Sync {
    /// Create a new dataset for `user_id`. Returns the canonical
    /// [`DatasetRef`] (including the sanitised table name).
    async fn create_dataset(
        &self,
        user_id: &str,
        schema: &DatasetSchema,
    ) -> Result<DatasetRef, DatasetError>;

    /// Drop a dataset and all of its rows.
    async fn drop_dataset(&self, dataset: &DatasetRef) -> Result<(), DatasetError>;

    /// Insert rows. Returns the number of rows inserted on success.
    async fn insert_rows(
        &self,
        dataset: &DatasetRef,
        rows: &[DatasetRow],
    ) -> Result<u64, DatasetError>;

    /// Count rows currently stored in the dataset.
    async fn count_rows(&self, dataset: &DatasetRef) -> Result<u64, DatasetError>;

    /// List all datasets owned by `user_id`.
    async fn list_datasets(&self, user_id: &str) -> Result<Vec<DatasetMeta>, DatasetError>;
}

/// Maximum identifier length, matching Postgres' `NAMEDATALEN - 1 = 63`.
///
/// SQLite has no practical limit, but applying the Postgres bound here
/// keeps generated table names portable across the two dialects.
pub const MAX_TABLE_NAME_LEN: usize = 63;

/// Sanitise an arbitrary user-supplied name into a valid SQL identifier.
///
/// Rule: lowercase ASCII, every byte that is not `[a-z0-9_]` (after
/// lowercasing) becomes `_`, then the result is truncated to
/// [`MAX_TABLE_NAME_LEN`] bytes. Non-ASCII input collapses to underscores
/// rather than being lossily transliterated.
pub fn sanitise_sql_identifier(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.extend(ch.to_lowercase());
        } else {
            // Underscore stays; every other non-alphanumeric collapses to one.
            out.push('_');
        }
    }
    if out.len() > MAX_TABLE_NAME_LEN {
        out.truncate(MAX_TABLE_NAME_LEN);
    }
    out
}

/// Build the canonical table name for `(user_id, dataset_name)`.
///
/// Both inputs are sanitised, joined with the literal separator
/// `__dataset__`, and the result truncated to [`MAX_TABLE_NAME_LEN`].
pub fn sanitise_table_name(user_id: &str, dataset_name: &str) -> String {
    let user = sanitise_sql_identifier(user_id);
    let name = sanitise_sql_identifier(dataset_name);
    let mut combined = String::with_capacity(user.len() + name.len() + 11);
    combined.push_str(&user);
    combined.push_str("__dataset__");
    combined.push_str(&name);
    if combined.len() > MAX_TABLE_NAME_LEN {
        combined.truncate(MAX_TABLE_NAME_LEN);
    }
    combined
}

fn validate_row(schema: &DatasetSchema, row: &DatasetRow) -> Result<(), DatasetError> {
    let declared: HashSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();

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
            Some(value) => {
                if value.is_null() && !col.nullable {
                    return Err(DatasetError::SchemaViolation(format!(
                        "column '{}' is not nullable",
                        col.name
                    )));
                }
            }
        }
    }

    Ok(())
}

#[allow(dead_code)] // W7 datasets: in-memory row buffer; not all fields wired yet
struct InMemoryState {
    schema: DatasetSchema,
    table: String,
    created_at: DateTime<Utc>,
    rows: Vec<DatasetRow>,
}

/// In-memory [`DatasetStore`] used by unit tests and by callers that don't
/// yet have a relational backend.
///
/// Implementation is `(user_id, dataset_name) → state` over a `DashMap`,
/// with each per-dataset state behind a `RwLock` so that row inserts and
/// counts contend per-dataset rather than globally.
pub struct InMemoryDatasetStore {
    inner: DashMap<(String, String), Arc<RwLock<InMemoryState>>>,
}

impl InMemoryDatasetStore {
    pub fn new() -> Self {
        Self {
            inner: DashMap::new(),
        }
    }

    fn lookup(
        &self,
        dataset: &DatasetRef,
    ) -> Result<Arc<RwLock<InMemoryState>>, DatasetError> {
        // Tenant isolation: keyed strictly by (user_id, name). The `table`
        // field on `DatasetRef` is informational and is not used for
        // routing — even if a caller forges a matching `table`, the
        // (user_id, name) tuple has to line up with what `create_dataset`
        // stored.
        let key = (dataset.user_id.clone(), dataset.name.clone());
        match self.inner.get(&key) {
            Some(state) => Ok(state.clone()),
            None => Err(DatasetError::NotFound(dataset.name.clone())),
        }
    }
}

impl Default for InMemoryDatasetStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DatasetStore for InMemoryDatasetStore {
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

        let declared: HashSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
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

        let table = sanitise_table_name(user_id, &schema.name);
        let key = (user_id.to_string(), schema.name.clone());

        // DashMap entry API is sync; this is fine because we hold the
        // entry only long enough to insert.
        use dashmap::mapref::entry::Entry;
        match self.inner.entry(key) {
            Entry::Occupied(_) => Err(DatasetError::AlreadyExists(schema.name.clone())),
            Entry::Vacant(slot) => {
                let state = InMemoryState {
                    schema: schema.clone(),
                    table: table.clone(),
                    created_at: Utc::now(),
                    rows: Vec::new(),
                };
                slot.insert(Arc::new(RwLock::new(state)));
                Ok(DatasetRef {
                    user_id: user_id.to_string(),
                    name: schema.name.clone(),
                    table,
                })
            }
        }
    }

    async fn drop_dataset(&self, dataset: &DatasetRef) -> Result<(), DatasetError> {
        let key = (dataset.user_id.clone(), dataset.name.clone());
        match self.inner.remove(&key) {
            Some(_) => Ok(()),
            None => Err(DatasetError::NotFound(dataset.name.clone())),
        }
    }

    async fn insert_rows(
        &self,
        dataset: &DatasetRef,
        rows: &[DatasetRow],
    ) -> Result<u64, DatasetError> {
        let state = self.lookup(dataset)?;
        let mut guard = state.write();
        for row in rows {
            validate_row(&guard.schema, row)?;
        }
        guard.rows.extend(rows.iter().cloned());
        Ok(rows.len() as u64)
    }

    async fn count_rows(&self, dataset: &DatasetRef) -> Result<u64, DatasetError> {
        let state = self.lookup(dataset)?;
        let guard = state.read();
        Ok(guard.rows.len() as u64)
    }

    async fn list_datasets(&self, user_id: &str) -> Result<Vec<DatasetMeta>, DatasetError> {
        let mut out = Vec::new();
        for entry in self.inner.iter() {
            let (owner, _name) = entry.key();
            if owner != user_id {
                continue;
            }
            let guard = entry.value().read();
            out.push(DatasetMeta {
                name: guard.schema.name.clone(),
                schema: guard.schema.clone(),
                row_count: guard.rows.len() as u64,
                created_at: guard.created_at,
            });
        }
        // Stable order so callers (and tests) don't have to rely on hash
        // iteration order.
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::schema::{ColumnDef, ColumnType};

    fn schema(name: &str) -> DatasetSchema {
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
    async fn in_memory_store_round_trip() {
        let store = InMemoryDatasetStore::new();
        let dref = store
            .create_dataset("alice", &schema("events"))
            .await
            .expect("create");
        assert_eq!(dref.user_id, "alice");
        assert_eq!(dref.name, "events");
        assert_eq!(dref.table, "alice__dataset__events");

        let inserted = store
            .insert_rows(&dref, &[row(1, "a"), row(2, "b"), row(3, "c")])
            .await
            .expect("insert");
        assert_eq!(inserted, 3);

        let count = store.count_rows(&dref).await.expect("count");
        assert_eq!(count, 3);

        let listed = store.list_datasets("alice").await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "events");
        assert_eq!(listed[0].row_count, 3);
    }

    #[tokio::test]
    async fn in_memory_tenant_isolation() {
        let store = InMemoryDatasetStore::new();
        let alice_ref = store
            .create_dataset("alice", &schema("foo"))
            .await
            .expect("alice create");

        // Bob tries to drop alice's dataset by spoofing the ref.
        let spoof = DatasetRef {
            user_id: "bob".to_string(),
            name: "foo".to_string(),
            table: alice_ref.table.clone(),
        };
        let err = store.drop_dataset(&spoof).await.unwrap_err();
        match err {
            DatasetError::NotFound(_) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }

        // Bob can create his own dataset named "foo" without colliding.
        let bob_ref = store
            .create_dataset("bob", &schema("foo"))
            .await
            .expect("bob create");
        assert_ne!(bob_ref.table, alice_ref.table);

        // Alice still sees only her own dataset.
        let alice_list = store.list_datasets("alice").await.expect("alice list");
        assert_eq!(alice_list.len(), 1);
        let bob_list = store.list_datasets("bob").await.expect("bob list");
        assert_eq!(bob_list.len(), 1);
    }

    #[tokio::test]
    async fn schema_violation_on_missing_column() {
        let store = InMemoryDatasetStore::new();
        let dref = store
            .create_dataset("alice", &schema("events"))
            .await
            .expect("create");

        // Missing the required 'label' column.
        let mut bad = HashMap::new();
        bad.insert("id".to_string(), serde_json::json!(1));
        let result = store
            .insert_rows(&dref, &[DatasetRow { values: bad }])
            .await;
        match result {
            Err(DatasetError::SchemaViolation(msg)) => {
                assert!(msg.contains("label"), "message should name column: {msg}");
            }
            other => panic!("expected SchemaViolation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn schema_violation_on_extra_column() {
        let store = InMemoryDatasetStore::new();
        let dref = store
            .create_dataset("alice", &schema("events"))
            .await
            .expect("create");

        let mut bad = HashMap::new();
        bad.insert("id".to_string(), serde_json::json!(1));
        bad.insert("label".to_string(), serde_json::json!("x"));
        bad.insert("not_a_column".to_string(), serde_json::json!(true));
        let result = store
            .insert_rows(&dref, &[DatasetRow { values: bad }])
            .await;
        match result {
            Err(DatasetError::SchemaViolation(msg)) => {
                assert!(msg.contains("not_a_column"));
            }
            other => panic!("expected SchemaViolation, got {other:?}"),
        }
    }

    #[test]
    fn table_name_sanitisation() {
        let table = sanitise_table_name("alice", "My Dataset!");
        assert_eq!(table, "alice__dataset__my_dataset_");
    }

    #[test]
    fn table_name_sanitisation_lowercases() {
        let table = sanitise_table_name("Alice-1", "Events");
        assert_eq!(table, "alice_1__dataset__events");
    }

    #[test]
    fn table_name_truncation_to_63_chars() {
        let very_long = "x".repeat(120);
        let table = sanitise_table_name("alice", &very_long);
        assert!(
            table.len() <= MAX_TABLE_NAME_LEN,
            "len was {}: {table}",
            table.len()
        );
        assert!(table.starts_with("alice__dataset__"));
    }
}
