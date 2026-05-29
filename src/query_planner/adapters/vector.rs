//! Vector candidate source backed by per-kind Vamana indices.
//!
//! `RealVectorQuerier` doesn't own the Vamana state directly — Vamana
//! projections live inside the [`crate::intent_log::JournaledWriter`] for
//! the writing path. Instead, the querier delegates to a
//! [`VamanaProvider`] that knows how to look up the right index and the
//! right reverse `(vector_id → memory_id)` map for a given user + kind.
//!
//! The wiring that backs a real `VamanaProvider` against
//! `MultiUserMemoryManager` is intentionally deferred — it requires
//! reshaping how Vamana indices are stored on the manager (today they're
//! moved into the writer and never re-exposed). This module stays purely
//! additive so the trait surface can be reviewed and tested ahead of that
//! larger refactor. Tests in this file use an in-memory provider impl.

use crate::query_planner::executor::VectorQuerier;
use crate::query_planner::predicate::VectorPredicate;
use anyhow::Result;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::vector_db::VamanaIndex;

/// Indirection used by [`RealVectorQuerier`] to reach per-user, per-kind
/// Vamana state. Implementations provide:
///
/// - `index(...)` — the shared `Arc<RwLock<VamanaIndex>>` to query against
/// - `memory_id_for(...)` — the reverse `(vector_id → memory_id)` lookup,
///   which lives on the Vamana projection (`VamanaProjection::memory_id_for`)
///
/// The provider is deliberately backend-agnostic: tests use an in-memory
/// map; the live binding will route through `MultiUserMemoryManager` once
/// the manager surfaces its Vamana state. The `kind` parameter is the
/// projection-name string used in the `VectorPredicate` payload (e.g.
/// `"vamana-text-primary"`, `"vamana-text-secondary"`, etc.).
pub trait VamanaProvider: Send + Sync {
    fn index(&self, user_id: &str, kind: &str) -> Option<Arc<RwLock<VamanaIndex>>>;
    fn memory_id_for(&self, user_id: &str, kind: &str, vector_id: u32) -> Option<String>;
}

/// Adapter that runs `VectorPredicate`s against a Vamana index supplied
/// by an injected [`VamanaProvider`].
#[derive(Clone)]
pub struct RealVectorQuerier {
    provider: Arc<dyn VamanaProvider>,
}

impl RealVectorQuerier {
    pub fn new(provider: Arc<dyn VamanaProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl VectorQuerier for RealVectorQuerier {
    async fn scan(
        &self,
        user_id: &str,
        p: &VectorPredicate,
    ) -> Result<Vec<(String, f32)>> {
        let index = match self.provider.index(user_id, &p.kind) {
            Some(idx) => idx,
            None => return Ok(Vec::new()),
        };
        // Vamana's index search is synchronous (and cheap); hold the read
        // lock only for the duration of the call.
        let hits = {
            let guard = index.read();
            guard.search(&p.query_vector, p.top_k)?
        };
        let mut out = Vec::with_capacity(hits.len());
        for (vector_id, score) in hits {
            if let Some(memory_id) =
                self.provider.memory_id_for(user_id, &p.kind, vector_id)
            {
                out.push((memory_id, score));
            } else {
                tracing::debug!(
                    user_id = %user_id,
                    kind = %p.kind,
                    vector_id = vector_id,
                    "Vamana hit has no memory_id mapping; skipping"
                );
            }
        }
        Ok(out)
    }

    async fn score(
        &self,
        user_id: &str,
        memory_id: &str,
        p: &VectorPredicate,
    ) -> Result<Option<f32>> {
        // Probe: re-run scan at top_k and look for the requested id. This
        // is correct under Vamana's monotone scoring assumption (a hit
        // present in top_k is the highest-scoring occurrence for that id);
        // if the id is not in the top_k slice we report `None`, which the
        // executor treats as "filter out".
        let hits = self.scan(user_id, p).await?;
        for (mid, score) in hits {
            if mid == memory_id {
                return Ok(Some(score));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_db::{DistanceMetric, VamanaConfig, VamanaIndex};
    use std::collections::HashMap;

    /// Simple in-memory provider: a single user / single kind / known
    /// index and reverse map. Used by the unit tests below.
    struct MockProvider {
        index: Arc<RwLock<VamanaIndex>>,
        reverse: HashMap<u32, String>,
        accepted_user: String,
        accepted_kind: String,
    }

    impl VamanaProvider for MockProvider {
        fn index(&self, user_id: &str, kind: &str) -> Option<Arc<RwLock<VamanaIndex>>> {
            if user_id == self.accepted_user && kind == self.accepted_kind {
                Some(self.index.clone())
            } else {
                None
            }
        }
        fn memory_id_for(
            &self,
            user_id: &str,
            kind: &str,
            vector_id: u32,
        ) -> Option<String> {
            if user_id == self.accepted_user && kind == self.accepted_kind {
                self.reverse.get(&vector_id).cloned()
            } else {
                None
            }
        }
    }

    fn build_index_with_three_vectors() -> (Arc<RwLock<VamanaIndex>>, HashMap<u32, String>) {
        let cfg = VamanaConfig {
            dimension: 3,
            max_degree: 4,
            search_list_size: 8,
            alpha: 1.2,
            use_mmap: false,
            distance_metric: DistanceMetric::Cosine,
        };
        let mut index = VamanaIndex::new(cfg).expect("new index");
        // Three nearly-orthogonal vectors so cosine scores are distinct.
        // `build` assigns vector ids 0..N-1 in input order.
        let vectors = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        index.build(vectors).expect("build graph");
        let mut reverse = HashMap::new();
        reverse.insert(0, "mem-x".to_string());
        reverse.insert(1, "mem-y".to_string());
        reverse.insert(2, "mem-z".to_string());
        (Arc::new(RwLock::new(index)), reverse)
    }

    #[tokio::test]
    async fn scan_returns_top_hit_for_aligned_query() {
        let (index, reverse) = build_index_with_three_vectors();
        let provider = Arc::new(MockProvider {
            index,
            reverse,
            accepted_user: "alice".into(),
            accepted_kind: "vamana-text-primary".into(),
        });
        let q = RealVectorQuerier::new(provider);
        let p = VectorPredicate {
            kind: "vamana-text-primary".into(),
            query_vector: vec![1.0, 0.0, 0.0],
            top_k: 1,
        };
        let hits = q.scan("alice", &p).await.expect("scan");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "mem-x");
    }

    #[tokio::test]
    async fn scan_for_unknown_user_or_kind_returns_empty() {
        let (index, reverse) = build_index_with_three_vectors();
        let provider = Arc::new(MockProvider {
            index,
            reverse,
            accepted_user: "alice".into(),
            accepted_kind: "vamana-text-primary".into(),
        });
        let q = RealVectorQuerier::new(provider);
        let p = VectorPredicate {
            kind: "vamana-text-primary".into(),
            query_vector: vec![1.0, 0.0, 0.0],
            top_k: 3,
        };
        // Wrong user — provider returns None for index lookup.
        let hits = q.scan("bob", &p).await.expect("scan");
        assert!(hits.is_empty());
        // Wrong kind — same outcome.
        let other_kind = VectorPredicate {
            kind: "vamana-image".into(),
            ..p.clone()
        };
        let hits = q.scan("alice", &other_kind).await.expect("scan");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn score_returns_some_when_id_is_in_topk_else_none() {
        let (index, reverse) = build_index_with_three_vectors();
        let provider = Arc::new(MockProvider {
            index,
            reverse,
            accepted_user: "alice".into(),
            accepted_kind: "vamana-text-primary".into(),
        });
        let q = RealVectorQuerier::new(provider);
        let p = VectorPredicate {
            kind: "vamana-text-primary".into(),
            query_vector: vec![1.0, 0.0, 0.0],
            top_k: 1,
        };
        // mem-x is the top hit at top_k=1.
        assert!(q.score("alice", "mem-x", &p).await.expect("score").is_some());
        // mem-y is the closest after mem-x but not in a top_k=1 slice.
        assert!(q.score("alice", "mem-y", &p).await.expect("score").is_none());
    }

    #[tokio::test]
    async fn skip_hits_without_reverse_mapping() {
        let cfg = VamanaConfig {
            dimension: 2,
            max_degree: 4,
            search_list_size: 8,
            alpha: 1.2,
            use_mmap: false,
            distance_metric: DistanceMetric::Cosine,
        };
        let mut index = VamanaIndex::new(cfg).expect("new index");
        index
            .build(vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]])
            .expect("build graph");
        // Reverse map deliberately missing vector_id 1 — only 0 has a mapping.
        let mut reverse = HashMap::new();
        reverse.insert(0, "mem-x".to_string());
        let provider = Arc::new(MockProvider {
            index: Arc::new(RwLock::new(index)),
            reverse,
            accepted_user: "alice".into(),
            accepted_kind: "k".into(),
        });
        let q = RealVectorQuerier::new(provider);
        let p = VectorPredicate {
            kind: "k".into(),
            query_vector: vec![1.0, 0.0],
            top_k: 5,
        };
        let hits = q.scan("alice", &p).await.expect("scan");
        // Only the hit with a known reverse mapping survives.
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "mem-x");
    }
}
