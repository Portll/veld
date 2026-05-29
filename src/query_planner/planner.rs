//! Selectivity heuristic + plan generator for the W6 query planner.
//!
//! Algorithm (foundation-tier — no histograms, no learned stats):
//!
//! 1. Estimate cardinality for every predicate (relational / vector / graph) using
//!    fixed heuristic constants (documented per-arm below).
//! 2. Sort all predicates ascending by estimate.
//! 3. The cheapest predicate becomes the `Scan*` op (materializes candidate ids).
//! 4. All remaining predicates become `Filter*` ops (probed per id).
//! 5. Each estimate is clamped against the matching `StageCaps` bucket.
//! 6. Append `Limit(query.limit)` as the terminal op.
//!
//! These constants are intentionally crude. The follow-up agent that wires real
//! `RelationalStore` / `VectorQuerier` / `GraphQuerier` backends will replace them
//! with backend-reported statistics (rowcount, recall@k, edge fan-out).

use super::plan::{PhysicalOp, PhysicalPlan};
use super::predicate::{
    GraphPredicate, Query, RelationalPredicate, StageCaps, VectorPredicate,
};
use std::fmt::Write as _;

// ─── Selectivity constants ────────────────────────────────────────────────────
//
// These are deliberate magic numbers. Each one encodes a prior belief about how
// many candidate memory ids a backend would return for a predicate of that
// shape, *before* clamping against `StageCaps`. They are NOT statistically
// derived — they are ordering hints. The planner only needs them to break ties
// in roughly the right direction.

/// `column = value` — a tight equality match. Roughly the rowcount of a unique
/// index probe assuming light cardinality on the column.
const EST_RELATIONAL_EQUALS: usize = 100;

/// `column IN (v0, v1, ..., vn)` — one Equals worth per element in the IN set.
const EST_RELATIONAL_IN_PER_VALUE: usize = 100;

/// `column BETWEEN lo AND hi` — assume ~10% of a notional 100_000-row table.
const EST_RELATIONAL_RANGE: usize = 10_000;

/// `user_id = X` — a tenancy filter, not a real filter. Assume ~1000 rows per
/// tenant; this is what most veld test accounts carry.
const EST_RELATIONAL_USER_ID: usize = 1_000;

/// `GraphPredicate::LinkedToEntity` — average entity has ~50 memories pointing
/// at it (Hebbian graph fan-in).
const EST_GRAPH_LINKED_TO_ENTITY: usize = 50;

/// `GraphPredicate::SharesEntity` — pairs of memories that share *any* entity
/// with a target memory; tighter than LinkedToEntity.
const EST_GRAPH_SHARES_ENTITY: usize = 20;

/// `GraphPredicate::EpisodeContains` — episodes are mid-sized clusters.
const EST_GRAPH_EPISODE_CONTAINS: usize = 30;

/// Public estimator for a relational predicate.
pub fn estimate_relational(p: &RelationalPredicate) -> usize {
    match p {
        RelationalPredicate::Equals { .. } => EST_RELATIONAL_EQUALS,
        RelationalPredicate::In { values, .. } => {
            EST_RELATIONAL_IN_PER_VALUE.saturating_mul(values.len())
        }
        RelationalPredicate::Range { .. } => EST_RELATIONAL_RANGE,
        RelationalPredicate::UserIdEquals(_) => EST_RELATIONAL_USER_ID,
    }
}

/// Public estimator for a vector predicate — by construction the ANN backend
/// returns exactly `top_k` ids, so the estimate is `top_k` itself.
pub fn estimate_vector(p: &VectorPredicate) -> usize {
    p.top_k
}

/// Public estimator for a graph predicate.
pub fn estimate_graph(p: &GraphPredicate) -> usize {
    match p {
        GraphPredicate::LinkedToEntity { .. } => EST_GRAPH_LINKED_TO_ENTITY,
        GraphPredicate::SharesEntity { .. } => EST_GRAPH_SHARES_ENTITY,
        GraphPredicate::EpisodeContains { .. } => EST_GRAPH_EPISODE_CONTAINS,
    }
}

/// Internal tagged-predicate enum so we can hold all three predicate flavours
/// in one sortable list while we run the selectivity pass.
enum TaggedPred<'a> {
    Relational(&'a RelationalPredicate),
    Vector(&'a VectorPredicate),
    Graph(&'a GraphPredicate),
}

impl<'a> TaggedPred<'a> {
    fn estimate(&self) -> usize {
        match self {
            TaggedPred::Relational(p) => estimate_relational(p),
            TaggedPred::Vector(p) => estimate_vector(p),
            TaggedPred::Graph(p) => estimate_graph(p),
        }
    }

    fn clamp(&self, caps: &StageCaps, raw: usize) -> usize {
        let cap = match self {
            TaggedPred::Relational(_) => caps.relational,
            TaggedPred::Vector(_) => caps.vector,
            TaggedPred::Graph(_) => caps.graph,
        };
        raw.min(cap)
    }
}

/// Plan a query. Returns a `PhysicalPlan` whose first op is a `Scan*` of the
/// cheapest predicate, followed by `Filter*` ops in ascending-estimate order,
/// followed by `Limit(query.limit)`.
///
/// If the query has zero predicates, the plan is just `Limit(query.limit)` — the
/// executor handles that case by returning an empty result.
pub fn plan(query: &Query) -> PhysicalPlan {
    // 1. Flatten every predicate into a single tagged list.
    let mut tagged: Vec<TaggedPred<'_>> = Vec::with_capacity(
        query.relational.len() + usize::from(query.vector.is_some()) + query.graph.len(),
    );
    for p in &query.relational {
        tagged.push(TaggedPred::Relational(p));
    }
    if let Some(p) = &query.vector {
        tagged.push(TaggedPred::Vector(p));
    }
    for p in &query.graph {
        tagged.push(TaggedPred::Graph(p));
    }

    // 2. Sort ascending by raw (pre-clamp) estimate. We sort on raw rather than
    //    clamped values so that two predicates with identical clamped estimates
    //    (because both got pinned to the same cap) still order by their natural
    //    selectivity.
    tagged.sort_by_key(|tp| tp.estimate());

    // 3. Build ops. First → Scan*, rest → Filter*. Append Limit at the end.
    let mut ordered_ops: Vec<PhysicalOp> = Vec::with_capacity(tagged.len() + 1);
    let mut explain = String::from("plan: ");
    let mut first = true;

    for tp in &tagged {
        let raw = tp.estimate();
        let est = tp.clamp(&query.stage_caps, raw);

        if first {
            first = false;
            let op = scan_for(tp, est);
            append_explain(&mut explain, &op);
            ordered_ops.push(op);
        } else {
            let op = filter_for(tp);
            // Filters don't expose est_card in the op itself, but the explain
            // string still benefits from showing the raw estimate so a reader
            // can see why this predicate ended up downstream of the scan.
            append_explain(&mut explain, &op);
            ordered_ops.push(op);
        }
    }

    let limit_op = PhysicalOp::Limit(query.limit);
    append_explain(&mut explain, &limit_op);
    ordered_ops.push(limit_op);

    PhysicalPlan {
        ordered_ops,
        explain,
    }
}

fn scan_for(tp: &TaggedPred<'_>, est_card: usize) -> PhysicalOp {
    match tp {
        TaggedPred::Relational(p) => PhysicalOp::ScanRelational {
            predicate: (*p).clone(),
            est_card,
        },
        TaggedPred::Vector(p) => PhysicalOp::ScanVector {
            predicate: (*p).clone(),
            est_card,
        },
        TaggedPred::Graph(p) => PhysicalOp::ScanGraph {
            predicate: (*p).clone(),
            est_card,
        },
    }
}

fn filter_for(tp: &TaggedPred<'_>) -> PhysicalOp {
    match tp {
        TaggedPred::Relational(p) => PhysicalOp::FilterRelational {
            predicate: (*p).clone(),
        },
        TaggedPred::Vector(p) => PhysicalOp::FilterVector {
            predicate: (*p).clone(),
        },
        TaggedPred::Graph(p) => PhysicalOp::FilterGraph {
            predicate: (*p).clone(),
        },
    }
}

fn append_explain(buf: &mut String, op: &PhysicalOp) {
    // Separator: append " → " between ops, but not before the first one.
    // The plan string starts with "plan: " so any subsequent op gets a separator
    // unless `buf` is still exactly "plan: ".
    if buf != "plan: " {
        buf.push_str(" → ");
    }
    match op {
        PhysicalOp::ScanRelational { predicate, est_card } => {
            let _ = write!(buf, "ScanRelational({}) [est={est_card}]", describe_relational(predicate));
        }
        PhysicalOp::ScanVector { predicate, est_card } => {
            let _ = write!(
                buf,
                "ScanVector(kind={}, top_k={}) [est={est_card}]",
                predicate.kind, predicate.top_k
            );
        }
        PhysicalOp::ScanGraph { predicate, est_card } => {
            let _ = write!(buf, "ScanGraph({}) [est={est_card}]", describe_graph(predicate));
        }
        PhysicalOp::FilterRelational { predicate } => {
            let _ = write!(buf, "FilterRelational({})", describe_relational(predicate));
        }
        PhysicalOp::FilterVector { predicate } => {
            let _ = write!(
                buf,
                "FilterVector(kind={}, top_k={})",
                predicate.kind, predicate.top_k
            );
        }
        PhysicalOp::FilterGraph { predicate } => {
            let _ = write!(buf, "FilterGraph({})", describe_graph(predicate));
        }
        PhysicalOp::Limit(n) => {
            let _ = write!(buf, "Limit({n})");
        }
    }
}

fn describe_relational(p: &RelationalPredicate) -> String {
    match p {
        RelationalPredicate::Equals { column, value } => format!("{column}={value}"),
        RelationalPredicate::In { column, values } => format!("{column} IN [{}]", values.len()),
        RelationalPredicate::Range { column, lo, hi } => {
            let lo_s = lo.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "-∞".into());
            let hi_s = hi.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "+∞".into());
            format!("{column} ∈ [{lo_s},{hi_s}]")
        }
        RelationalPredicate::UserIdEquals(u) => format!("user_id={u}"),
    }
}

fn describe_graph(p: &GraphPredicate) -> String {
    match p {
        GraphPredicate::LinkedToEntity { entity_id } => format!("LinkedToEntity({entity_id})"),
        GraphPredicate::SharesEntity { other_memory_id } => {
            format!("SharesEntity({other_memory_id})")
        }
        GraphPredicate::EpisodeContains { episode_id } => format!("EpisodeContains({episode_id})"),
    }
}
