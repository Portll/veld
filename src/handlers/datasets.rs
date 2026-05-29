//! Dataset HTTP handlers (W7).
//!
//! Endpoints in this module:
//!
//! | Method | Path                              | Purpose                          |
//! | ------ | --------------------------------- | -------------------------------- |
//! | POST   | `/api/datasets`                   | Create a dataset                 |
//! | GET    | `/api/datasets`                   | List the caller's datasets       |
//! | GET    | `/api/datasets/{name}`            | Metadata + row count             |
//! | DELETE | `/api/datasets/{name}`            | Drop a dataset                   |
//! | POST   | `/api/datasets/{name}/rows`       | Insert rows                      |
//! | POST   | `/api/datasets/{name}/query`      | Parametrised SELECT              |
//! | POST   | `/api/datasets/{name}/link`       | Link a row to an entity / memory |
//!
//! All endpoints are mounted behind API-key auth on the protected router.
//! `user_id` is resolved via [`resolve_request_user_id`] so the
//! authenticated tenant always wins over the request body / query string.
//!
//! ## SQL injection contract
//!
//! The `query` endpoint accepts a user-supplied `WHERE` fragment. This
//! fragment is embedded verbatim into the SELECT — callers are
//! authenticated and are signing their own queries. **All** bound values
//! flow through [`Param`] binding; user input is never concatenated into
//! SQL outside the explicit WHERE clause.

use std::sync::Arc;

use axum::{
    extract::{Extension, Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::datasets::link::{LinkKind, RowPk};
use crate::datasets::{DatasetError, DatasetMeta, DatasetRef, DatasetRow, DatasetSchema};
use crate::errors::{AppError, ValidationErrorExt};
use crate::storage::relational::Param;
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

// =============================================================================
// REQUEST / RESPONSE TYPES
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct DatasetUserQuery {
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateDatasetRequest {
    pub user_id: Option<String>,
    pub schema: DatasetSchema,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDatasetResponse {
    pub dataset: DatasetRef,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DropDatasetResponse {
    pub dropped: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListDatasetsResponse {
    pub datasets: Vec<DatasetMeta>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct DatasetMetadataResponse {
    pub dataset: DatasetRef,
    pub meta: DatasetMeta,
}

#[derive(Debug, Deserialize)]
pub struct InsertRowsRequest {
    pub user_id: Option<String>,
    pub rows: Vec<DatasetRow>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InsertRowsResponse {
    pub inserted: u64,
}

/// Parametrised SELECT.
///
/// `where_clause` is embedded verbatim after `WHERE` (callers are
/// authenticated). `params` is the bind vector — each `serde_json::Value`
/// maps onto a [`Param`] variant. `limit` is enforced by appending `LIMIT
/// N` to the SQL; we cap the absolute maximum at [`MAX_QUERY_LIMIT`] to
/// protect the server.
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub user_id: Option<String>,
    #[serde(rename = "where")]
    pub where_clause: Option<String>,
    #[serde(default)]
    pub params: Vec<serde_json::Value>,
    pub limit: Option<u32>,
}

/// Absolute ceiling on `LIMIT` enforced for the SELECT endpoint.
pub const MAX_QUERY_LIMIT: u32 = 1_000;

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct LinkRequest {
    pub user_id: Option<String>,
    pub row_pk: RowPk,
    pub kind: LinkKindWire,
    pub target_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKindWire {
    Entity,
    Memory,
}

#[derive(Debug, Serialize)]
pub struct LinkResponse {
    pub linked: bool,
}

impl From<LinkKindWire> for LinkKind {
    fn from(k: LinkKindWire) -> Self {
        match k {
            LinkKindWire::Entity => LinkKind::Entity,
            LinkKindWire::Memory => LinkKind::Memory,
        }
    }
}

// =============================================================================
// ERROR MAPPING
// =============================================================================

/// Map [`DatasetError`] onto the application's [`AppError`].
fn map_dataset_error(err: DatasetError) -> AppError {
    match err {
        DatasetError::AlreadyExists(name) => AppError::InvalidInput {
            field: "name".to_string(),
            reason: format!("dataset '{name}' already exists"),
        },
        DatasetError::NotFound(name) => AppError::InvalidInput {
            field: "name".to_string(),
            reason: format!("dataset '{name}' not found"),
        },
        DatasetError::SchemaViolation(msg) => AppError::InvalidInput {
            field: "schema".to_string(),
            reason: msg,
        },
        DatasetError::TenantIsolation => AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "tenant isolation violation".to_string(),
        },
        DatasetError::Internal(msg) => {
            AppError::Internal(anyhow::anyhow!("dataset store internal: {msg}"))
        }
    }
}

/// Return the dataset store or a 503 if not configured.
fn require_dataset_store(
    state: &MultiUserMemoryManager,
) -> Result<Arc<dyn crate::datasets::DatasetStore>, AppError> {
    state
        .dataset_store
        .clone()
        .ok_or_else(|| AppError::ServiceUnavailable("dataset store not configured".to_string()))
}

/// Return the link store or a 503 if not configured.
fn require_link_store(
    state: &MultiUserMemoryManager,
) -> Result<Arc<dyn crate::datasets::LinkStore>, AppError> {
    state
        .link_store
        .clone()
        .ok_or_else(|| AppError::ServiceUnavailable("link store not configured".to_string()))
}

/// Helper: locate the dataset by `(user_id, name)` — returns the canonical
/// [`DatasetRef`], or 404 if the caller does not own a dataset by that name.
///
/// Implementation lists datasets owned by the authenticated tenant and
/// filters; this is the only public surface for resolving a name → table
/// mapping. We deliberately do not expose `catalog_lookup` on the trait —
/// the list+filter is O(n) per call but n is bounded by per-tenant dataset
/// count which stays tiny in practice, and keeps the trait minimal.
async fn locate_dataset(
    store: &Arc<dyn crate::datasets::DatasetStore>,
    user_id: &str,
    name: &str,
) -> Result<DatasetRef, AppError> {
    let listed = store.list_datasets(user_id).await.map_err(map_dataset_error)?;
    let found = listed.iter().any(|m| m.name == name);
    if !found {
        return Err(AppError::InvalidInput {
            field: "name".to_string(),
            reason: format!("dataset '{name}' not found"),
        });
    }
    Ok(DatasetRef {
        user_id: user_id.to_string(),
        name: name.to_string(),
        table: crate::datasets::sanitise_table_name(user_id, name),
    })
}

// =============================================================================
// HANDLERS
// =============================================================================

/// POST /api/datasets — create a dataset for the authenticated tenant.
#[tracing::instrument(skip(state, req))]
pub async fn create_dataset(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<CreateDatasetRequest>,
) -> Result<Json<CreateDatasetResponse>, AppError> {
    let user_id = resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let dref = store
        .create_dataset(&user_id, &req.schema)
        .await
        .map_err(map_dataset_error)?;
    Ok(Json(CreateDatasetResponse { dataset: dref }))
}

/// GET /api/datasets — list the authenticated tenant's datasets.
#[tracing::instrument(skip(state))]
pub async fn list_datasets(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    axum::extract::Query(q): axum::extract::Query<DatasetUserQuery>,
) -> Result<Json<ListDatasetsResponse>, AppError> {
    let user_id = resolve_request_user_id(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let datasets = store
        .list_datasets(&user_id)
        .await
        .map_err(map_dataset_error)?;
    let total = datasets.len();
    Ok(Json(ListDatasetsResponse { datasets, total }))
}

/// GET /api/datasets/{name} — metadata + row count for one dataset.
#[tracing::instrument(skip(state))]
pub async fn get_dataset_metadata(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(name): Path<String>,
    axum::extract::Query(q): axum::extract::Query<DatasetUserQuery>,
) -> Result<Json<DatasetMetadataResponse>, AppError> {
    let user_id = resolve_request_user_id(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let listed = store
        .list_datasets(&user_id)
        .await
        .map_err(map_dataset_error)?;
    let meta = listed.into_iter().find(|m| m.name == name).ok_or_else(|| {
        AppError::InvalidInput {
            field: "name".to_string(),
            reason: format!("dataset '{name}' not found"),
        }
    })?;
    let dref = DatasetRef {
        user_id: user_id.clone(),
        name: name.clone(),
        table: crate::datasets::sanitise_table_name(&user_id, &name),
    };
    Ok(Json(DatasetMetadataResponse { dataset: dref, meta }))
}

/// DELETE /api/datasets/{name} — drop a dataset.
#[tracing::instrument(skip(state))]
pub async fn drop_dataset(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(name): Path<String>,
    axum::extract::Query(q): axum::extract::Query<DatasetUserQuery>,
) -> Result<Json<DropDatasetResponse>, AppError> {
    let user_id = resolve_request_user_id(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let dref = locate_dataset(&store, &user_id, &name).await?;
    store.drop_dataset(&dref).await.map_err(map_dataset_error)?;
    Ok(Json(DropDatasetResponse { dropped: true }))
}

/// POST /api/datasets/{name}/rows — insert rows.
#[tracing::instrument(skip(state, req))]
pub async fn insert_rows(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(name): Path<String>,
    Json(req): Json<InsertRowsRequest>,
) -> Result<Json<InsertRowsResponse>, AppError> {
    let user_id = resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let dref = locate_dataset(&store, &user_id, &name).await?;
    let inserted = store
        .insert_rows(&dref, &req.rows)
        .await
        .map_err(map_dataset_error)?;
    Ok(Json(InsertRowsResponse { inserted }))
}

/// POST /api/datasets/{name}/query — parametrised SELECT.
#[tracing::instrument(skip(state, req))]
pub async fn query_dataset(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(name): Path<String>,
    Json(req): Json<QueryRequest>,
) -> Result<Response, AppError> {
    let user_id = resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let dref = locate_dataset(&store, &user_id, &name).await?;

    // The DatasetStore trait does not expose raw SELECT — that path goes
    // through the relational executor handle the manager keeps alongside
    // the trait object. Both point at the same database.
    let executor = state
        .dataset_executor
        .clone()
        .ok_or_else(|| {
            AppError::ServiceUnavailable(
                "query endpoint requires a relational executor".to_string(),
            )
        })?;

    // Build the SQL. Every value is bound via `Param` — the only
    // user-supplied string interpolated into the SQL is the WHERE clause
    // fragment (callers are authenticated and signing their own queries).
    let limit = req.limit.unwrap_or(MAX_QUERY_LIMIT).min(MAX_QUERY_LIMIT);
    let mut sql = format!("SELECT * FROM \"{}\"", dref.table);
    if let Some(w) = req.where_clause.as_ref().filter(|s| !s.trim().is_empty()) {
        sql.push_str(" WHERE ");
        sql.push_str(w);
    }
    sql.push_str(&format!(" LIMIT {limit}"));

    // Convert each JSON param to a borrowed `Param`. JSON-array/object
    // params bind as `Param::Json`; everything else maps onto the closest
    // primitive variant.
    let params: Vec<Param<'_>> = req
        .params
        .iter()
        .map(json_to_param)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| AppError::InvalidInput {
            field: "params".to_string(),
            reason: e,
        })?;

    let rows = executor
        .query(&sql, &params)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("query failed: {e}")))?;

    let columns: Vec<String> = rows
        .first()
        .map(|r| r.columns().iter().map(|c| c.name.clone()).collect())
        .unwrap_or_default();
    let mut out_rows: Vec<Vec<serde_json::Value>> = Vec::with_capacity(rows.len());
    for r in &rows {
        let mut cells = Vec::with_capacity(r.columns().len());
        for idx in 0..r.columns().len() {
            // `Option<Value>` round-trips NULL → JSON null; the inner
            // `Value` decoder may still error on truly corrupt rows, in
            // which case we substitute null rather than fail the whole
            // query.
            let cell = match r.get::<Option<serde_json::Value>>(idx) {
                Ok(Some(v)) => v,
                Ok(None) => serde_json::Value::Null,
                Err(_) => serde_json::Value::Null,
            };
            cells.push(cell);
        }
        out_rows.push(cells);
    }

    let body = QueryResponse {
        columns,
        rows: out_rows,
    };
    Ok((StatusCode::OK, Json(body)).into_response())
}

/// POST /api/datasets/{name}/link — link a row to an entity or memory.
#[tracing::instrument(skip(state, req))]
pub async fn link_row(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(name): Path<String>,
    Json(req): Json<LinkRequest>,
) -> Result<Json<LinkResponse>, AppError> {
    let user_id = resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let store = require_dataset_store(&state)?;
    let links = require_link_store(&state)?;
    let dref = locate_dataset(&store, &user_id, &name).await?;

    let kind: LinkKind = req.kind.into();
    match kind {
        LinkKind::Entity => links
            .link_row_to_entity(&dref, &req.row_pk, &req.target_id)
            .await
            .map_err(map_dataset_error)?,
        LinkKind::Memory => links
            .link_row_to_memory(&dref, &req.row_pk, &req.target_id)
            .await
            .map_err(map_dataset_error)?,
    }
    Ok(Json(LinkResponse { linked: true }))
}

// =============================================================================
// PRIVATE HELPERS
// =============================================================================

fn json_to_param(value: &serde_json::Value) -> Result<Param<'_>, String> {
    match value {
        serde_json::Value::Null => Ok(Param::Null),
        serde_json::Value::Bool(b) => Ok(Param::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Param::I64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Param::F64(f))
            } else {
                Err(format!("unsupported numeric value {n}"))
            }
        }
        serde_json::Value::String(s) => Ok(Param::Text(s.as_str())),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => Ok(Param::Json(value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::schema::{ColumnDef, ColumnType};
    use crate::datasets::{
        DatasetStore, LinkStore, RelationalDatasetStore, RelationalLinkStore,
    };
    use crate::handlers::test_helpers::{post_json, send_typed, TestHarness};
    use crate::storage::relational::{BoxError, RelationalStore, Row, SqliteRelationalStore};
    use async_trait::async_trait;
    use axum::http::StatusCode;

    struct BoxErrorSqlite(SqliteRelationalStore);

    #[async_trait]
    impl RelationalStore for BoxErrorSqlite {
        type Error = BoxError;
        async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, BoxError> {
            self.0.execute(sql, params).await.map_err(BoxError::new)
        }
        async fn query(
            &self,
            sql: &str,
            params: &[Param<'_>],
        ) -> Result<Vec<Row>, BoxError> {
            self.0.query(sql, params).await.map_err(BoxError::new)
        }
        fn backend(&self) -> crate::storage::relational::RelationalBackend {
            self.0.backend()
        }
    }

    async fn harness_with_datasets() -> (TestHarness, axum::Router) {
        let sqlite = SqliteRelationalStore::in_memory()
            .await
            .expect("sqlite open");
        let backing: Arc<dyn RelationalStore<Error = BoxError>> =
            Arc::new(BoxErrorSqlite(sqlite));
        let ds: Arc<dyn DatasetStore> = Arc::new(
            RelationalDatasetStore::new(backing.clone())
                .await
                .expect("init ds"),
        );
        let ls: Arc<dyn LinkStore> = Arc::new(
            RelationalLinkStore::new(backing.clone())
                .await
                .expect("init ls"),
        );

        let (manager, temp_dir) = TestHarness::fresh_manager();
        let manager = manager.with_dataset_stores(ds, ls, backing);
        let h = TestHarness::with_manager(manager, temp_dir);
        let router = h.router();
        (h, router)
    }

    fn sample_schema() -> DatasetSchema {
        DatasetSchema {
            name: "events".to_string(),
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
            ],
            primary_key: vec!["id".to_string()],
        }
    }

    #[tokio::test]
    async fn round_trip_create_insert_list_drop() {
        let (_h, app) = harness_with_datasets().await;

        // Create.
        let req = post_json(
            "/api/datasets",
            &serde_json::json!({
                "user_id": "alice",
                "schema": sample_schema(),
            }),
        );
        let (status, body): (StatusCode, CreateDatasetResponse) =
            send_typed(app.clone(), req).await;
        assert_eq!(status, StatusCode::OK, "create");
        assert_eq!(body.dataset.name, "events");
        assert_eq!(body.dataset.user_id, "alice");

        // Insert 5 rows.
        let rows = (1..=5)
            .map(|i| {
                serde_json::json!({
                    "values": { "id": i, "label": format!("row-{i}") }
                })
            })
            .collect::<Vec<_>>();
        let req = post_json(
            "/api/datasets/events/rows",
            &serde_json::json!({
                "user_id": "alice",
                "rows": rows,
            }),
        );
        let (status, body): (StatusCode, InsertRowsResponse) =
            send_typed(app.clone(), req).await;
        assert_eq!(status, StatusCode::OK, "insert");
        assert_eq!(body.inserted, 5);

        // List — single dataset with row_count = 5.
        let req = crate::handlers::test_helpers::get("/api/datasets?user_id=alice");
        let (status, body): (StatusCode, ListDatasetsResponse) =
            send_typed(app.clone(), req).await;
        assert_eq!(status, StatusCode::OK, "list");
        assert_eq!(body.total, 1);
        assert_eq!(body.datasets[0].row_count, 5);

        // Drop.
        let req = crate::handlers::test_helpers::delete(
            "/api/datasets/events?user_id=alice",
        );
        let (status, body): (StatusCode, DropDatasetResponse) =
            send_typed(app.clone(), req).await;
        assert_eq!(status, StatusCode::OK, "drop");
        assert!(body.dropped);
    }
}
