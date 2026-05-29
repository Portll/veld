//! Executor skeleton for the W6 query planner.
//!
//! `Executor` consumes a `PhysicalPlan` plus the three backend trait surfaces
//! (`RelationalQuerier`, `VectorQuerier`, `GraphQuerier`) and produces a
//! `Vec<ScoredMemoryId>`.
//!
//! The first op in the plan is a `Scan*` — it materializes the initial candidate
//! set. Every subsequent op is a `Filter*` that probes per-id. The terminal
//! `Limit(n)` truncates. If a vector scan ran at any point, its score is carried
//! through; otherwise the score is `0.0` (relational/graph predicates do not
//! produce a similarity score on their own).
//!
//! The trait surfaces are defined here so the follow-up agent that wires real
//! backends (Vamana, RelationalStore, graph) can implement them without
//! reaching across module boundaries. Tests in this file use mock impls.

use super::plan::{PhysicalOp, PhysicalPlan};
use super::predicate::{GraphPredicate, Query, RelationalPredicate, VectorPredicate};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Relational candidate source.
///
/// `scan` materializes ids that match the predicate; `matches` is the per-id
/// filter probe used when this predicate is not the leading scan.
#[async_trait]
pub trait RelationalQuerier: Send + Sync {
    async fn scan(
        &self,
        user_id: &str,
        p: &RelationalPredicate,
    ) -> anyhow::Result<Vec<String>>;
    async fn matches(
        &self,
        user_id: &str,
        memory_id: &str,
        p: &RelationalPredicate,
    ) -> anyhow::Result<bool>;
}

/// Vector ANN candidate source.
///
/// `scan` returns `(memory_id, score)` pairs in the backend's native order;
/// `score` is per-backend (cosine, dot, etc.) and is carried through to the
/// final result. `score` (the per-id probe) is `None` if the id isn't in the
/// ANN result set — the executor treats `None` as "filter out".
#[async_trait]
pub trait VectorQuerier: Send + Sync {
    async fn scan(
        &self,
        user_id: &str,
        p: &VectorPredicate,
    ) -> anyhow::Result<Vec<(String, f32)>>;
    async fn score(
        &self,
        user_id: &str,
        memory_id: &str,
        p: &VectorPredicate,
    ) -> anyhow::Result<Option<f32>>;
}

/// Graph reachability candidate source.
#[async_trait]
pub trait GraphQuerier: Send + Sync {
    async fn scan(
        &self,
        user_id: &str,
        p: &GraphPredicate,
    ) -> anyhow::Result<Vec<String>>;
    async fn matches(
        &self,
        user_id: &str,
        memory_id: &str,
        p: &GraphPredicate,
    ) -> anyhow::Result<bool>;
}

/// Output row from `Executor::run`.
#[derive(Debug, Clone)]
pub struct ScoredMemoryId {
    pub memory_id: String,
    pub score: f32,
}

/// The plan executor. Holds one `Arc<dyn ...>` per backend trait.
pub struct Executor {
    pub relational: Arc<dyn RelationalQuerier>,
    pub vector: Arc<dyn VectorQuerier>,
    pub graph: Arc<dyn GraphQuerier>,
}

impl Executor {
    /// Run `plan` against `query`'s tenant. Walks the plan left-to-right:
    /// first op is a `Scan*` that produces the candidate set; subsequent ops
    /// filter; `Limit(n)` truncates.
    ///
    /// Returns `Vec<ScoredMemoryId>`. Order matches the leading scan's order
    /// (i.e. if `ScanVector` ran first, results are ordered by ANN rank).
    pub async fn run(
        &self,
        query: &Query,
        plan: &PhysicalPlan,
    ) -> anyhow::Result<Vec<ScoredMemoryId>> {
        // Preserve insertion order of the leading scan so vector-rank ordering
        // survives intersection. `scores` is a side map keyed by memory_id.
        let mut candidates: Vec<String> = Vec::new();
        let mut scores: HashMap<String, f32> = HashMap::new();

        let mut have_scan = false;
        let mut limit: Option<usize> = None;

        for op in &plan.ordered_ops {
            match op {
                PhysicalOp::ScanRelational { predicate, est_card } => {
                    let ids = self.relational.scan(&query.user_id, predicate).await?;
                    candidates = clamp_take(ids, *est_card);
                    have_scan = true;
                }
                PhysicalOp::ScanVector { predicate, est_card } => {
                    let pairs = self.vector.scan(&query.user_id, predicate).await?;
                    candidates.clear();
                    for (id, score) in pairs.into_iter().take(*est_card) {
                        scores.insert(id.clone(), score);
                        candidates.push(id);
                    }
                    have_scan = true;
                }
                PhysicalOp::ScanGraph { predicate, est_card } => {
                    let ids = self.graph.scan(&query.user_id, predicate).await?;
                    candidates = clamp_take(ids, *est_card);
                    have_scan = true;
                }
                PhysicalOp::FilterRelational { predicate } => {
                    debug_assert!(have_scan, "filter before scan in plan");
                    let mut kept: Vec<String> = Vec::with_capacity(candidates.len());
                    for id in candidates.drain(..) {
                        if self
                            .relational
                            .matches(&query.user_id, &id, predicate)
                            .await?
                        {
                            kept.push(id);
                        }
                    }
                    candidates = kept;
                }
                PhysicalOp::FilterVector { predicate } => {
                    debug_assert!(have_scan, "filter before scan in plan");
                    let mut kept: Vec<String> = Vec::with_capacity(candidates.len());
                    for id in candidates.drain(..) {
                        if let Some(score) =
                            self.vector.score(&query.user_id, &id, predicate).await?
                        {
                            scores.insert(id.clone(), score);
                            kept.push(id);
                        }
                    }
                    candidates = kept;
                }
                PhysicalOp::FilterGraph { predicate } => {
                    debug_assert!(have_scan, "filter before scan in plan");
                    let mut kept: Vec<String> = Vec::with_capacity(candidates.len());
                    for id in candidates.drain(..) {
                        if self.graph.matches(&query.user_id, &id, predicate).await? {
                            kept.push(id);
                        }
                    }
                    candidates = kept;
                }
                PhysicalOp::Limit(n) => {
                    limit = Some(*n);
                }
            }
        }

        if let Some(n) = limit {
            candidates.truncate(n);
        }

        Ok(candidates
            .into_iter()
            .map(|id| {
                let score = scores.get(&id).copied().unwrap_or(0.0);
                ScoredMemoryId {
                    memory_id: id,
                    score,
                }
            })
            .collect())
    }
}

fn clamp_take(mut ids: Vec<String>, cap: usize) -> Vec<String> {
    if ids.len() > cap {
        ids.truncate(cap);
    }
    ids
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_planner::planner;
    use crate::query_planner::predicate::{StageCaps, VectorPredicate};
    use std::collections::HashSet;

    // ─── Mock querier impls ───────────────────────────────────────────────

    /// Mock relational backend. `scan_rows[col]` returns candidate ids for a
    /// scan on that column; `row_columns[id]` is the set of columns that
    /// memory belongs to (used by `matches`).
    struct MockRelational {
        /// column → ids returned by `scan`.
        scan_rows: HashMap<String, Vec<String>>,
        /// memory_id → set of "column" keys it satisfies.
        row_columns: HashMap<String, HashSet<String>>,
    }

    #[async_trait]
    impl RelationalQuerier for MockRelational {
        async fn scan(
            &self,
            _user_id: &str,
            p: &RelationalPredicate,
        ) -> anyhow::Result<Vec<String>> {
            let key = relational_key(p);
            Ok(self.scan_rows.get(&key).cloned().unwrap_or_default())
        }
        async fn matches(
            &self,
            _user_id: &str,
            memory_id: &str,
            p: &RelationalPredicate,
        ) -> anyhow::Result<bool> {
            let key = relational_key(p);
            Ok(self
                .row_columns
                .get(memory_id)
                .map(|cols| cols.contains(&key))
                .unwrap_or(false))
        }
    }

    fn relational_key(p: &RelationalPredicate) -> String {
        match p {
            RelationalPredicate::Equals { column, value } => format!("eq:{column}={value}"),
            RelationalPredicate::In { column, values } => {
                format!("in:{column}=[{}]", values.len())
            }
            RelationalPredicate::Range { column, .. } => format!("range:{column}"),
            RelationalPredicate::UserIdEquals(u) => format!("uid:{u}"),
        }
    }

    /// Mock vector backend. `scan_results` is the canned ANN result list;
    /// `score_map[id]` is the per-id probe score (None → not in index).
    struct MockVector {
        scan_results: Vec<(String, f32)>,
        score_map: HashMap<String, f32>,
    }

    #[async_trait]
    impl VectorQuerier for MockVector {
        async fn scan(
            &self,
            _user_id: &str,
            p: &VectorPredicate,
        ) -> anyhow::Result<Vec<(String, f32)>> {
            Ok(self.scan_results.iter().take(p.top_k).cloned().collect())
        }
        async fn score(
            &self,
            _user_id: &str,
            memory_id: &str,
            _p: &VectorPredicate,
        ) -> anyhow::Result<Option<f32>> {
            Ok(self.score_map.get(memory_id).copied())
        }
    }

    /// Mock graph backend. `scan_results[key]` returns candidates for a graph
    /// scan; `row_edges[id]` lists the graph-keys an id satisfies.
    struct MockGraph {
        scan_results: HashMap<String, Vec<String>>,
        row_edges: HashMap<String, HashSet<String>>,
    }

    #[async_trait]
    impl GraphQuerier for MockGraph {
        async fn scan(
            &self,
            _user_id: &str,
            p: &GraphPredicate,
        ) -> anyhow::Result<Vec<String>> {
            let key = graph_key(p);
            Ok(self.scan_results.get(&key).cloned().unwrap_or_default())
        }
        async fn matches(
            &self,
            _user_id: &str,
            memory_id: &str,
            p: &GraphPredicate,
        ) -> anyhow::Result<bool> {
            let key = graph_key(p);
            Ok(self
                .row_edges
                .get(memory_id)
                .map(|edges| edges.contains(&key))
                .unwrap_or(false))
        }
    }

    fn graph_key(p: &GraphPredicate) -> String {
        match p {
            GraphPredicate::LinkedToEntity { entity_id } => format!("link:{entity_id}"),
            GraphPredicate::SharesEntity { other_memory_id } => format!("share:{other_memory_id}"),
            GraphPredicate::EpisodeContains { episode_id } => format!("ep:{episode_id}"),
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    fn empty_relational() -> Arc<dyn RelationalQuerier> {
        Arc::new(MockRelational {
            scan_rows: HashMap::new(),
            row_columns: HashMap::new(),
        })
    }
    fn empty_vector() -> Arc<dyn VectorQuerier> {
        Arc::new(MockVector {
            scan_results: Vec::new(),
            score_map: HashMap::new(),
        })
    }
    fn empty_graph() -> Arc<dyn GraphQuerier> {
        Arc::new(MockGraph {
            scan_results: HashMap::new(),
            row_edges: HashMap::new(),
        })
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    // ─── 1. selectivity_orders_most_selective_first ───────────────────────

    #[test]
    fn selectivity_orders_most_selective_first() {
        // VectorPredicate(top_k=5) → est=5. UserIdEquals → est=1000.
        // Plan must scan vector first.
        let query = Query {
            user_id: "u1".into(),
            relational: vec![RelationalPredicate::UserIdEquals("u1".into())],
            vector: Some(VectorPredicate {
                kind: "vamana-text-primary".into(),
                query_vector: vec![0.1, 0.2, 0.3],
                top_k: 5,
            }),
            graph: vec![],
            limit: 10,
            stage_caps: StageCaps::default(),
        };
        let plan = planner::plan(&query);
        match &plan.ordered_ops[0] {
            PhysicalOp::ScanVector { est_card, .. } => assert_eq!(*est_card, 5),
            other => panic!("expected ScanVector first, got {other:?}"),
        }
        match &plan.ordered_ops[1] {
            PhysicalOp::FilterRelational { .. } => {}
            other => panic!("expected FilterRelational next, got {other:?}"),
        }
        assert!(matches!(
            plan.ordered_ops.last().unwrap(),
            PhysicalOp::Limit(10)
        ));
    }

    // ─── 2. relational_only_plan_intersects_correctly ─────────────────────

    #[test]
    fn relational_only_plan_intersects_correctly() {
        // Three Equals predicates with different IN-set sizes. Smallest scans,
        // the other two filter; intersection-only ids survive.
        let p_small = RelationalPredicate::In {
            column: "tag".into(),
            values: vec![serde_json::json!("a")],
        };
        let p_mid = RelationalPredicate::In {
            column: "tag".into(),
            values: vec![serde_json::json!("a"), serde_json::json!("b")],
        };
        let p_big = RelationalPredicate::In {
            column: "tag".into(),
            values: (0..5).map(|i| serde_json::json!(i)).collect(),
        };

        // Build mock so that scan(small) → [a, b, c, d], and matches() says
        // a satisfies mid+big, b satisfies mid only, c satisfies big only,
        // d satisfies neither. Intersection should be {a} only.
        let mut scan_rows = HashMap::new();
        scan_rows.insert(
            relational_key(&p_small),
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
        );

        let mid_key = relational_key(&p_mid);
        let big_key = relational_key(&p_big);
        let small_key = relational_key(&p_small);

        let mut row_columns: HashMap<String, HashSet<String>> = HashMap::new();
        for (id, keys) in [
            ("a", vec![&small_key, &mid_key, &big_key]),
            ("b", vec![&small_key, &mid_key]),
            ("c", vec![&small_key, &big_key]),
            ("d", vec![&small_key]),
        ] {
            row_columns.insert(
                id.into(),
                keys.into_iter().map(|s| s.clone()).collect(),
            );
        }

        let exec = Executor {
            relational: Arc::new(MockRelational { scan_rows, row_columns }),
            vector: empty_vector(),
            graph: empty_graph(),
        };

        let query = Query {
            user_id: "u1".into(),
            relational: vec![p_small.clone(), p_mid.clone(), p_big.clone()],
            vector: None,
            graph: vec![],
            limit: 100,
            stage_caps: StageCaps::default(),
        };

        let plan = planner::plan(&query);
        // Smallest IN(1) → 100 est, IN(2) → 200, IN(5) → 500. So scan = p_small.
        assert!(matches!(
            plan.ordered_ops[0],
            PhysicalOp::ScanRelational { .. }
        ));
        // Two filters between scan and Limit.
        assert_eq!(
            plan.ordered_ops
                .iter()
                .filter(|o| matches!(o, PhysicalOp::FilterRelational { .. }))
                .count(),
            2
        );

        let result = rt().block_on(exec.run(&query, &plan)).unwrap();
        let ids: HashSet<String> =
            result.into_iter().map(|s| s.memory_id).collect();
        assert_eq!(ids, ["a".to_string()].into_iter().collect::<HashSet<_>>());
    }

    // ─── 3. vector_plus_relational_join ───────────────────────────────────

    #[test]
    fn vector_plus_relational_join() {
        let vec_pred = VectorPredicate {
            kind: "vamana-text-primary".into(),
            query_vector: vec![0.0; 4],
            top_k: 10,
        };
        let rel_pred = RelationalPredicate::Equals {
            column: "tag".into(),
            value: serde_json::json!("important"),
        };

        // Vector scan returns {m1, m2, m3} with descending scores.
        let scan_results = vec![
            ("m1".to_string(), 0.95_f32),
            ("m2".to_string(), 0.88_f32),
            ("m3".to_string(), 0.71_f32),
        ];

        // Relational matches() says only m1 + m3 have the tag.
        let rel_key = relational_key(&rel_pred);
        let mut row_columns: HashMap<String, HashSet<String>> = HashMap::new();
        row_columns.insert(
            "m1".into(),
            [rel_key.clone()].into_iter().collect(),
        );
        row_columns.insert(
            "m3".into(),
            [rel_key.clone()].into_iter().collect(),
        );

        let exec = Executor {
            relational: Arc::new(MockRelational {
                scan_rows: HashMap::new(),
                row_columns,
            }),
            vector: Arc::new(MockVector {
                scan_results,
                score_map: HashMap::new(),
            }),
            graph: empty_graph(),
        };

        let query = Query {
            user_id: "u1".into(),
            relational: vec![rel_pred],
            vector: Some(vec_pred),
            graph: vec![],
            limit: 100,
            stage_caps: StageCaps::default(),
        };

        let plan = planner::plan(&query);
        // Vector est=10, Equals est=100 → ScanVector first, FilterRelational second.
        assert!(matches!(
            plan.ordered_ops[0],
            PhysicalOp::ScanVector { .. }
        ));
        assert!(matches!(
            plan.ordered_ops[1],
            PhysicalOp::FilterRelational { .. }
        ));

        let result = rt().block_on(exec.run(&query, &plan)).unwrap();
        let ids: Vec<String> = result.iter().map(|s| s.memory_id.clone()).collect();
        // Intersection is {m1, m3}, in vector-scan order.
        assert_eq!(ids, vec!["m1".to_string(), "m3".to_string()]);
        // Scores survive from the vector scan.
        assert!((result[0].score - 0.95).abs() < 1e-6);
        assert!((result[1].score - 0.71).abs() < 1e-6);
    }

    // ─── 4. stage_caps_clamp_estimates ────────────────────────────────────

    #[test]
    fn stage_caps_clamp_estimates() {
        // top_k=1000 but stage_caps.vector=10 → est_card on the scan must clamp to 10.
        let query = Query {
            user_id: "u1".into(),
            relational: vec![],
            vector: Some(VectorPredicate {
                kind: "vamana-text-primary".into(),
                query_vector: vec![0.0; 4],
                top_k: 1000,
            }),
            graph: vec![],
            limit: 100,
            stage_caps: StageCaps {
                vector: 10,
                graph: 1000,
                relational: 10_000,
            },
        };
        let plan = planner::plan(&query);
        match &plan.ordered_ops[0] {
            PhysicalOp::ScanVector { est_card, .. } => assert_eq!(*est_card, 10),
            other => panic!("expected ScanVector first, got {other:?}"),
        }
        assert!(plan.explain.contains("[est=10]"));
    }

    // ─── 5. explain_string_round_trips ────────────────────────────────────

    #[test]
    fn explain_string_round_trips() {
        let query = Query {
            user_id: "u1".into(),
            relational: vec![RelationalPredicate::Equals {
                column: "tag".into(),
                value: serde_json::json!("X"),
            }],
            vector: Some(VectorPredicate {
                kind: "vamana-text-primary".into(),
                query_vector: vec![0.0; 4],
                top_k: 10,
            }),
            graph: vec![],
            limit: 5,
            stage_caps: StageCaps::default(),
        };
        let plan = planner::plan(&query);
        let s = &plan.explain;
        // Must begin with "plan: ".
        assert!(s.starts_with("plan: "), "explain: {s}");
        // Must show the vector scan with est and the relational filter.
        assert!(s.contains("ScanVector"), "explain: {s}");
        assert!(s.contains("top_k=10"), "explain: {s}");
        assert!(s.contains("[est=10]"), "explain: {s}");
        assert!(s.contains("FilterRelational"), "explain: {s}");
        assert!(s.contains("tag="), "explain: {s}");
        // Must end with the Limit op.
        assert!(s.contains("Limit(5)"), "explain: {s}");
        // Separator must be the arrow.
        assert!(s.contains(" → "), "explain: {s}");
    }

    // ─── Smoke: graph scan + filter ──────────────────────────────────────

    #[test]
    fn graph_scan_with_relational_filter_intersects() {
        let entity = uuid::Uuid::new_v4();
        let gp = GraphPredicate::LinkedToEntity { entity_id: entity };
        let rel = RelationalPredicate::Equals {
            column: "tag".into(),
            value: serde_json::json!("relevant"),
        };

        // Graph scan returns {g1, g2, g3}.
        let g_key = graph_key(&gp);
        let mut scan_results = HashMap::new();
        scan_results.insert(
            g_key.clone(),
            vec!["g1".into(), "g2".into(), "g3".into()],
        );

        // Relational matches() says g2 only.
        let rel_key = relational_key(&rel);
        let mut row_columns: HashMap<String, HashSet<String>> = HashMap::new();
        row_columns.insert(
            "g2".into(),
            [rel_key.clone()].into_iter().collect(),
        );

        let exec = Executor {
            relational: Arc::new(MockRelational {
                scan_rows: HashMap::new(),
                row_columns,
            }),
            vector: empty_vector(),
            graph: Arc::new(MockGraph {
                scan_results,
                row_edges: HashMap::new(),
            }),
        };

        let query = Query {
            user_id: "u1".into(),
            relational: vec![rel],
            vector: None,
            graph: vec![gp],
            limit: 10,
            stage_caps: StageCaps::default(),
        };

        let plan = planner::plan(&query);
        // LinkedToEntity est=50, Equals est=100 → graph scans first.
        assert!(matches!(
            plan.ordered_ops[0],
            PhysicalOp::ScanGraph { .. }
        ));
        assert!(matches!(
            plan.ordered_ops[1],
            PhysicalOp::FilterRelational { .. }
        ));

        let result = rt().block_on(exec.run(&query, &plan)).unwrap();
        let ids: Vec<String> = result.iter().map(|s| s.memory_id.clone()).collect();
        assert_eq!(ids, vec!["g2".to_string()]);
    }

    // ─── Smoke: empty query → empty Limit-only plan ──────────────────────

    #[test]
    fn empty_query_yields_limit_only_plan() {
        let query = Query {
            user_id: "u1".into(),
            relational: vec![],
            vector: None,
            graph: vec![],
            limit: 7,
            stage_caps: StageCaps::default(),
        };
        let plan = planner::plan(&query);
        assert_eq!(plan.ordered_ops.len(), 1);
        assert!(matches!(plan.ordered_ops[0], PhysicalOp::Limit(7)));

        let exec = Executor {
            relational: empty_relational(),
            vector: empty_vector(),
            graph: empty_graph(),
        };
        let result = rt().block_on(exec.run(&query, &plan)).unwrap();
        assert!(result.is_empty());
    }
}
