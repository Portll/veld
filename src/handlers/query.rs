//! W6 query-planner HTTP surface.
//!
//! | Method | Path                 | Purpose                              |
//! | ------ | -------------------- | ------------------------------------ |
//! | POST   | `/api/query/plan`    | Build + explain the physical plan    |
//! | POST   | `/api/query/execute` | Build the plan and run the executor  |
//!
//! Both are mounted behind API-key auth on the protected router. The body
//! is a planner [`Query`]; the resolved tenant id always wins: we run the
//! body's `user_id` through [`resolve_request_user_id`] (so a tenant-bound
//! key cannot query another tenant) and write the result back onto the
//! query before planning, so the executor's tenant scoping cannot be
//! spoofed by the body.
//!
//! `/api/recall` is intentionally untouched — it stays RRF-based for
//! back-compat. This surface is the explicit, explainable planner path.
//!
//! ## Backend wiring
//!
//! - vector: [`RealVectorQuerier`] over the tenant's live Vamana
//!   projections (via `VamanaProvider for MultiUserMemoryManager`).
//! - graph: [`RealGraphQuerier`] over the tenant's entity/episode topology
//!   (via `GraphProvider for MultiUserMemoryManager`).
//! - relational: [`RealRelationalQuerier`] over the configured relational
//!   executor (`MultiUserMemoryManager::dataset_executor`). That executor
//!   is optional; when it is absent, a query carrying relational
//!   predicates is rejected with 503, and predicate-free queries use a
//!   null relational source (never invoked) so vector/graph-only planning
//!   still works.

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    extract::{Extension, State},
    response::Json,
};
use serde::Serialize;

use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::query_planner::predicate::RelationalPredicate;
use crate::query_planner::{
    build_plan, Executor, Query, RealGraphQuerier, RealRelationalQuerier, RealVectorQuerier,
    RelationalQuerier,
};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

/// Response for `POST /api/query/plan`: the human-readable plan plus the
/// op count. The full `PhysicalPlan` is intentionally not serialised — the
/// `explain` string is its canonical, stable rendering.
#[derive(Debug, Serialize)]
pub struct PlanResponse {
    pub explain: String,
    pub op_count: usize,
}

/// One scored result row. Mirrors the executor's `ScoredMemoryId` as a
/// serialisable wire type so the planner crate stays free of HTTP concerns.
#[derive(Debug, Serialize)]
pub struct ScoredMemoryIdWire {
    pub memory_id: String,
    pub score: f32,
}

/// Response for `POST /api/query/execute`.
#[derive(Debug, Serialize)]
pub struct ExecuteResponse {
    pub results: Vec<ScoredMemoryIdWire>,
    pub total: usize,
}

/// Resolve the tenant binding and stamp it onto the query. Shared by both
/// endpoints so plan and execute scope identically.
fn bind_tenant(
    mut query: Query,
    authenticated_user: Option<&AuthenticatedUser>,
) -> Result<Query, AppError> {
    let user_id = resolve_request_user_id(Some(query.user_id.as_str()), authenticated_user)?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    query.user_id = user_id;
    Ok(query)
}

/// POST /api/query/plan — build the physical plan and return its explain.
#[tracing::instrument(skip_all)]
pub async fn plan_query(
    State(_state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Json(query): Json<Query>,
) -> Result<Json<PlanResponse>, AppError> {
    let query = bind_tenant(query, authenticated_user.as_ref().map(|e| &e.0))?;
    let plan = build_plan(&query);
    Ok(Json(PlanResponse {
        op_count: plan.ordered_ops.len(),
        explain: plan.explain,
    }))
}

/// POST /api/query/execute — build the plan and run the executor.
#[tracing::instrument(skip_all)]
pub async fn execute_query(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Json(query): Json<Query>,
) -> Result<Json<ExecuteResponse>, AppError> {
    let query = bind_tenant(query, authenticated_user.as_ref().map(|e| &e.0))?;

    // Relational predicates need a configured relational backend. Reject
    // up front rather than silently filtering to nothing.
    if !query.relational.is_empty() && state.dataset_executor.is_none() {
        return Err(AppError::ServiceUnavailable(
            "relational predicates require a configured relational backend".to_string(),
        ));
    }

    let relational: Arc<dyn RelationalQuerier> = match state.dataset_executor.clone() {
        Some(store) => Arc::new(RealRelationalQuerier::new(store)),
        // Unreachable for relational-bearing queries (rejected above); a
        // null source keeps the executor's required slot filled so
        // vector/graph-only queries run without a relational backend.
        None => Arc::new(NullRelationalQuerier),
    };

    let executor = Executor {
        relational,
        vector: Arc::new(RealVectorQuerier::new(state.clone())),
        graph: Arc::new(RealGraphQuerier::new(state.clone())),
    };

    let plan = build_plan(&query);
    let results = executor
        .run(&query, &plan)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("query execution failed: {e}")))?;

    let results: Vec<ScoredMemoryIdWire> = results
        .into_iter()
        .map(|r| ScoredMemoryIdWire {
            memory_id: r.memory_id,
            score: r.score,
        })
        .collect();
    let total = results.len();
    Ok(Json(ExecuteResponse { results, total }))
}

/// Relational source used when no relational backend is configured. Only
/// reachable for queries with no relational predicates (those are rejected
/// up front), so its methods are never invoked at runtime — it exists
/// solely to satisfy the [`Executor`]'s required relational slot.
struct NullRelationalQuerier;

#[async_trait]
impl RelationalQuerier for NullRelationalQuerier {
    async fn scan(&self, _user_id: &str, _p: &RelationalPredicate) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }
    async fn matches(
        &self,
        _user_id: &str,
        _memory_id: &str,
        _p: &RelationalPredicate,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }
}
