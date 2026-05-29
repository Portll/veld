//! Graph candidate source backed by Veld's per-tenant memory topology.
//!
//! Like the vector adapter, `RealGraphQuerier` does not reach into
//! `MultiUserMemoryManager` directly — it delegates to an injected
//! [`GraphProvider`]. That keeps the `GraphPredicate` dispatch logic
//! unit-testable against an in-memory fixture and leaves the
//! `query_planner` module free of any dependency on the HTTP/state layer
//! (the live binding is `impl GraphProvider for MultiUserMemoryManager`,
//! which lives next to the manager in `handlers::state`).
//!
//! ## Predicate dispatch
//!
//! - [`GraphPredicate::LinkedToEntity`] → memories whose `EntityRef` set
//!   contains the entity id.
//! - [`GraphPredicate::SharesEntity`] → memories that share at least one
//!   entity with the anchor memory, excluding the anchor itself.
//! - [`GraphPredicate::EpisodeContains`] → memories whose episode context
//!   names the episode id.
//!
//! Replaces the parking-place `StubGraphQuerier`, which stays in the tree
//! as a sealed null object for callers that want an explicitly empty
//! graph (e.g. relational/vector-only deployments and the executor's own
//! mock-driven tests).

use crate::query_planner::executor::GraphQuerier;
use crate::query_planner::predicate::GraphPredicate;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

/// Indirection used by [`RealGraphQuerier`] to reach a tenant's
/// entity/episode topology. Implementations resolve three primitives over
/// Veld's memory store; the querier composes them into the three
/// `GraphPredicate` variants.
///
/// Every primitive is tenant-scoped via `user_id` and returns owned data,
/// so the querier never holds a store lock across an `await`.
pub trait GraphProvider: Send + Sync {
    /// Entity ids referenced by a memory (its `EntityRef` set). Empty when
    /// the memory is unknown to this tenant or carries no entities.
    fn entity_ids_of_memory(&self, user_id: &str, memory_id: &str) -> Vec<Uuid>;

    /// Memory ids whose `EntityRef` set contains `entity_id`.
    fn memory_ids_with_entity(&self, user_id: &str, entity_id: &Uuid) -> Vec<String>;

    /// Memory ids whose episode context names `episode_id`.
    fn memory_ids_in_episode(&self, user_id: &str, episode_id: &Uuid) -> Vec<String>;
}

/// Adapter that resolves `GraphPredicate`s against an injected
/// [`GraphProvider`].
#[derive(Clone)]
pub struct RealGraphQuerier {
    provider: Arc<dyn GraphProvider>,
}

impl RealGraphQuerier {
    pub fn new(provider: Arc<dyn GraphProvider>) -> Self {
        Self { provider }
    }

    /// Shared resolution for `SharesEntity`: every memory that shares at
    /// least one entity with `anchor`, excluding the anchor itself.
    /// De-duplicated across entities (a memory sharing two entities with
    /// the anchor appears once).
    fn shares_entity(&self, user_id: &str, anchor: &str) -> Vec<String> {
        let entities = self.provider.entity_ids_of_memory(user_id, anchor);
        if entities.is_empty() {
            return Vec::new();
        }
        let mut seen: HashSet<String> = HashSet::new();
        for entity_id in &entities {
            for mid in self.provider.memory_ids_with_entity(user_id, entity_id) {
                if mid != anchor {
                    seen.insert(mid);
                }
            }
        }
        seen.into_iter().collect()
    }
}

#[async_trait]
impl GraphQuerier for RealGraphQuerier {
    async fn scan(&self, user_id: &str, p: &GraphPredicate) -> Result<Vec<String>> {
        let ids = match p {
            GraphPredicate::LinkedToEntity { entity_id } => {
                self.provider.memory_ids_with_entity(user_id, entity_id)
            }
            GraphPredicate::SharesEntity { other_memory_id } => {
                self.shares_entity(user_id, other_memory_id)
            }
            GraphPredicate::EpisodeContains { episode_id } => {
                self.provider.memory_ids_in_episode(user_id, episode_id)
            }
        };
        Ok(ids)
    }

    async fn matches(
        &self,
        user_id: &str,
        memory_id: &str,
        p: &GraphPredicate,
    ) -> Result<bool> {
        let hit = match p {
            GraphPredicate::LinkedToEntity { entity_id } => self
                .provider
                .entity_ids_of_memory(user_id, memory_id)
                .contains(entity_id),
            GraphPredicate::SharesEntity { other_memory_id } => {
                if memory_id == other_memory_id {
                    false
                } else {
                    let anchor: HashSet<Uuid> = self
                        .provider
                        .entity_ids_of_memory(user_id, other_memory_id)
                        .into_iter()
                        .collect();
                    !anchor.is_empty()
                        && self
                            .provider
                            .entity_ids_of_memory(user_id, memory_id)
                            .iter()
                            .any(|e| anchor.contains(e))
                }
            }
            GraphPredicate::EpisodeContains { episode_id } => self
                .provider
                .memory_ids_in_episode(user_id, episode_id)
                .iter()
                .any(|m| m == memory_id),
        };
        Ok(hit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory provider fixture. `entities[memory_id]` is that memory's
    /// entity set; `episodes[memory_id]` is its episode. The reverse
    /// lookups are derived on demand so the fixture stays declarative.
    struct MockProvider {
        accepted_user: String,
        entities: HashMap<String, Vec<Uuid>>,
        episodes: HashMap<String, Uuid>,
    }

    impl GraphProvider for MockProvider {
        fn entity_ids_of_memory(&self, user_id: &str, memory_id: &str) -> Vec<Uuid> {
            if user_id != self.accepted_user {
                return Vec::new();
            }
            self.entities.get(memory_id).cloned().unwrap_or_default()
        }
        fn memory_ids_with_entity(&self, user_id: &str, entity_id: &Uuid) -> Vec<String> {
            if user_id != self.accepted_user {
                return Vec::new();
            }
            let mut out: Vec<String> = self
                .entities
                .iter()
                .filter(|(_, ents)| ents.contains(entity_id))
                .map(|(mid, _)| mid.clone())
                .collect();
            out.sort();
            out
        }
        fn memory_ids_in_episode(&self, user_id: &str, episode_id: &Uuid) -> Vec<String> {
            if user_id != self.accepted_user {
                return Vec::new();
            }
            let mut out: Vec<String> = self
                .episodes
                .iter()
                .filter(|(_, ep)| *ep == episode_id)
                .map(|(mid, _)| mid.clone())
                .collect();
            out.sort();
            out
        }
    }

    fn fixture() -> (RealGraphQuerier, Uuid, Uuid, Uuid) {
        let e1 = Uuid::new_v4();
        let e2 = Uuid::new_v4();
        let ep = Uuid::new_v4();
        let mut entities = HashMap::new();
        // m1 ↔ {e1, e2}; m2 ↔ {e1}; m3 ↔ {e2}; m4 ↔ {} (no entities).
        entities.insert("m1".to_string(), vec![e1, e2]);
        entities.insert("m2".to_string(), vec![e1]);
        entities.insert("m3".to_string(), vec![e2]);
        entities.insert("m4".to_string(), Vec::new());
        let mut episodes = HashMap::new();
        episodes.insert("m1".to_string(), ep);
        episodes.insert("m3".to_string(), ep);
        let provider = Arc::new(MockProvider {
            accepted_user: "alice".into(),
            entities,
            episodes,
        });
        (RealGraphQuerier::new(provider), e1, e2, ep)
    }

    #[tokio::test]
    async fn linked_to_entity_scans_all_holders() {
        let (q, e1, _e2, _ep) = fixture();
        let p = GraphPredicate::LinkedToEntity { entity_id: e1 };
        let mut hits = q.scan("alice", &p).await.expect("scan");
        hits.sort();
        assert_eq!(hits, vec!["m1".to_string(), "m2".to_string()]);
        assert!(q.matches("alice", "m2", &p).await.expect("matches"));
        assert!(!q.matches("alice", "m3", &p).await.expect("matches"));
        // Wrong tenant sees nothing.
        assert!(q.scan("bob", &p).await.expect("scan").is_empty());
    }

    #[tokio::test]
    async fn shares_entity_excludes_anchor_and_dedupes() {
        let (q, _e1, _e2, _ep) = fixture();
        // m1 holds {e1, e2}; m2 shares e1, m3 shares e2. Anchor m1 excluded.
        let p = GraphPredicate::SharesEntity {
            other_memory_id: "m1".into(),
        };
        let mut hits = q.scan("alice", &p).await.expect("scan");
        hits.sort();
        assert_eq!(hits, vec!["m2".to_string(), "m3".to_string()]);
        assert!(q.matches("alice", "m2", &p).await.expect("matches"));
        // m1 vs m1 is never a self-match.
        assert!(!q.matches("alice", "m1", &p).await.expect("matches"));
        // m4 has no entities → shares nothing.
        assert!(!q.matches("alice", "m4", &p).await.expect("matches"));
    }

    #[tokio::test]
    async fn episode_contains_lists_members() {
        let (q, _e1, _e2, ep) = fixture();
        let p = GraphPredicate::EpisodeContains { episode_id: ep };
        let mut hits = q.scan("alice", &p).await.expect("scan");
        hits.sort();
        assert_eq!(hits, vec!["m1".to_string(), "m3".to_string()]);
        assert!(q.matches("alice", "m3", &p).await.expect("matches"));
        assert!(!q.matches("alice", "m2", &p).await.expect("matches"));
    }
}
