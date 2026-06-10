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

/// A temporal fact projected to the wire (B4 supporting evidence).
///
/// Mirrors the load-bearing fields of [`crate::memory::temporal_facts::TemporalFact`]
/// so the answer-derivation surface can justify a derived date without leaking
/// internal storage shapes.
#[derive(Debug, Serialize)]
pub struct TemporalFactWire {
    pub entity: String,
    pub event: String,
    pub resolved_time: String,
    pub source_memory_id: String,
    pub confidence: f32,
    pub source_text: String,
}

/// A derived answer to a "when did/is X?" temporal question (B4).
///
/// This is the answer-side complement to B1/B2 (query-side date parsing). It is
/// purely additive: `results` is unaffected, and the field is `None` whenever no
/// date is derivable (non-temporal queries, or temporal queries with no
/// computable date).
#[derive(Debug, Serialize)]
pub struct DerivedAnswer {
    /// ISO `YYYY-MM-DD` (anchor paths) or the fact's resolved-time repr
    /// (temporal-fact path, may be coarser e.g. `2023-06`).
    pub derived_date: String,
    /// `"anchor_relative"` | `"event_anchor"` | `"temporal_fact"`.
    pub method: String,
    /// Stored facts that justified the derived date (empty for pure
    /// query-computed anchor-relative answers).
    pub supporting_facts: Vec<TemporalFactWire>,
    /// Source memory ids backing the supporting facts.
    pub source_memories: Vec<String>,
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
    let derived_answer = derive_answer_for_query(&state, &query);

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
/// Never touches retrieval/ranking; only reads temporal facts.
fn derive_answer_for_query(state: &AppState, query: &Query) -> Option<DerivedAnswer> {
    use crate::memory::query_parser::{asks_for_temporal_answer, derive_temporal_answer};

    let query_text = query.query_text.as_deref()?;
    // Cheap gate first — avoids touching the substrate for non-temporal queries.
    if !asks_for_temporal_answer(query_text) {
        return None;
    }

    let user_id = query.user_id.clone();
    let user_memory = state.get_user_memory(&user_id).ok()?;
    let mem = user_memory.read();

    // Event-anchor resolver (B2): "the Monday after the wedding".
    let event_resolver = |event_phrase: &str| -> Option<chrono::NaiveDate> {
        mem.resolve_event_to_date(&user_id, event_phrase)
    };

    // Temporal-fact lookup: collect candidate facts whose event keywords appear
    // in the query. Reuses the same event index `resolve_event_to_date` walks.
    let fact_lookup = |q: &str| -> Vec<crate::memory::temporal_facts::TemporalFact> {
        let stopwords = [
            "when", "did", "does", "is", "was", "were", "the", "a", "an", "of", "to", "do",
            "what", "date", "day", "time", "on", "in", "at", "go", "have", "has", "and", "or",
            "her", "his", "their", "they", "she", "he",
        ];
        let mut seen = std::collections::HashSet::new();
        let mut facts = Vec::new();
        for word in q
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .filter(|w| !stopwords.contains(&w.to_lowercase().as_str()))
        {
            if let Ok(found) = mem.find_temporal_facts_by_event(&user_id, word, 10) {
                for f in found {
                    if seen.insert(f.id.clone()) {
                        facts.push(f);
                    }
                }
            }
        }
        facts
    };

    let derived = derive_temporal_answer(query_text, Some(&event_resolver), Some(&fact_lookup))?;

    // Project the pure derivation result onto the wire type.
    let supporting_facts: Vec<TemporalFactWire> = derived
        .supporting_facts
        .iter()
        .map(|f| TemporalFactWire {
            entity: f.entity.clone(),
            event: f.event.clone(),
            resolved_time: f.resolved_time.to_sortable_string(),
            source_memory_id: f.source_memory_id.0.to_string(),
            confidence: f.confidence,
            source_text: f.source_text.clone(),
        })
        .collect();
    let source_memories: Vec<String> = derived
        .source_memories
        .iter()
        .map(|m| m.0.to_string())
        .collect();

    Some(DerivedAnswer {
        derived_date: derived.derived_date,
        method: derived.method.as_str().to_string(),
        supporting_facts,
        source_memories,
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
