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
use super::temporal_derivation::{derive_answer_for_user_query, DerivedAnswer};
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::query_planner::predicate::RelationalPredicate;
use crate::query_planner::{
    build_plan, Executor, Query, RealGraphQuerier, RealRelationalQuerier, RealVectorQuerier,
    RelationalQuerier,
};
use crate::validation;

// Re-export the shared wire types so existing downstream code that references
// `handlers::query::DerivedAnswer` / `TemporalFactWire` continues to compile.
pub use super::temporal_derivation::TemporalFactWire;

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
    /// B4 temporal answer-derivation. `None` (and omitted from JSON) when no
    /// temporal answer is derivable — response is then byte-identical to the
    /// pre-B4 shape. Never affects `results`/`total`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_answer: Option<DerivedAnswer>,
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

    // B4: temporal answer-derivation. Purely additive — `results`/`total` above
    // are already final and untouched. Only fires when the request carries raw
    // `query_text` AND it asks for a temporal answer AND a date is derivable;
    // otherwise `derived_answer` is `None` and the response is byte-identical to
    // the pre-B4 shape.
    let derived_answer = derive_answer_for_query(&state, &query).await;

    Ok(Json(ExecuteResponse {
        results,
        total,
        derived_answer,
    }))
}

/// Run B4 temporal answer-derivation for an execute request.
///
/// Returns `None` (the common case) when:
/// - the request carried no raw `query_text` (structured-only payload), or
/// - the query is not a "when did/is X?" question, or
/// - no date can be computed from anchors or stored temporal facts, or
/// - the tenant's memory substrate is unavailable.
///
/// Delegates to [`derive_answer_for_user_query`] in `handlers::temporal_derivation`
/// so the logic is shared with `/api/recall`. That lookup can open the tenant and reads its
/// fact and event stores, so it runs on the blocking pool, as `/api/recall` runs it.
async fn derive_answer_for_query(state: &AppState, query: &Query) -> Option<DerivedAnswer> {
    let query_text = query.query_text.clone()?;
    let state = Arc::clone(state);
    let user_id = query.user_id.clone();
    tokio::task::spawn_blocking(move || derive_answer_for_user_query(&state, &user_id, &query_text))
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("temporal answer derivation task failed: {e}");
            None
        })
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

#[cfg(test)]
mod planner_sources_while_a_tenant_opens {
    //! The graph and vector sources reach the manager through sync providers that open a
    //! tenant, or its journaled writer, on first use. Each test runtime has one thread, so a
    //! source that ran its provider on it would stall the timeout until the init lock is let go.
    use super::*;
    use crate::handlers::test_helpers::{hold_init_lock, TestHarness};
    use crate::query_planner::{GraphPredicate, GraphQuerier, VectorPredicate, VectorQuerier};
    use std::time::Duration;

    #[tokio::test(flavor = "current_thread")]
    async fn a_graph_scan_opens_its_tenant_off_the_runtime_thread() {
        const TENANT: &str = "planner-graph-mid-open";
        let harness = TestHarness::new();

        let (release, opener) = hold_init_lock(harness.manager.user_memory_init_lock(TENANT));
        let querier = RealGraphQuerier::new(harness.manager.clone());
        let predicate = GraphPredicate::LinkedToEntity {
            entity_id: uuid::Uuid::new_v4(),
        };
        let mut scan = tokio::spawn(async move { querier.scan(TENANT, &predicate).await });
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut scan).await;
        assert!(
            waited.is_err(),
            "the scan finished while the tenant's init lock was held, so it held the runtime \
             thread until the lock was let go"
        );

        release.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        let ids = scan.await.expect("scan task").expect("scan");
        assert!(ids.is_empty(), "a new tenant holds no memories: {ids:?}");
        assert!(
            harness.manager.user_memories.get(TENANT).is_some(),
            "the scan opened the tenant"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_vector_scan_opens_its_writer_off_the_runtime_thread() {
        const TENANT: &str = "planner-vector-mid-open";
        let harness = TestHarness::new();

        let (release, opener) = hold_init_lock(harness.manager.journaled_writer_init_lock(TENANT));
        let querier = RealVectorQuerier::new(harness.manager.clone());
        let predicate = VectorPredicate {
            kind: "vamana-text-primary".to_string(),
            query_vector: vec![1.0, 0.0, 0.0],
            top_k: 5,
        };
        let mut scan = tokio::spawn(async move { querier.scan(TENANT, &predicate).await });
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut scan).await;
        assert!(
            waited.is_err(),
            "the scan finished while the writer's init lock was held, so it held the runtime \
             thread until the lock was let go"
        );

        release.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        let hits = scan.await.expect("scan task").expect("scan");
        assert!(hits.is_empty(), "a new tenant has no vectors: {hits:?}");
        assert!(
            harness.manager.journaled_writers.get(TENANT).is_some(),
            "the scan opened the tenant's writer"
        );
    }
}
