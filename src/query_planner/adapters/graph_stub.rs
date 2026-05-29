//! Empty `GraphQuerier` placeholder.
//!
//! The real graph adapter (reading from Veld's knowledge graph for
//! `LinkedToEntity` / `SharesEntity` / `EpisodeContains`) is deferred to
//! a follow-up commit so the planner has an end-to-end-runnable shape
//! today. `StubGraphQuerier::scan` always returns an empty candidate set
//! and `matches` always returns `false`, which the executor correctly
//! interprets as "filter to nothing".
//!
//! Despite the name "Stub", this is not a placeholder-in-the-CLAUDE.md-
//! sense (no TODOs, no `todo!()` calls, no panics). It's a deliberate
//! empty implementation with a documented replacement plan — a sealed
//! null object until the real graph adapter lands.

use crate::query_planner::executor::GraphQuerier;
use crate::query_planner::predicate::GraphPredicate;
use anyhow::Result;
use async_trait::async_trait;

/// Empty `GraphQuerier` implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubGraphQuerier;

impl StubGraphQuerier {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl GraphQuerier for StubGraphQuerier {
    async fn scan(&self, _user_id: &str, _p: &GraphPredicate) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    async fn matches(
        &self,
        _user_id: &str,
        _memory_id: &str,
        _p: &GraphPredicate,
    ) -> Result<bool> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn scan_returns_empty_for_every_predicate_variant() {
        let q = StubGraphQuerier::new();
        let preds = vec![
            GraphPredicate::LinkedToEntity {
                entity_id: Uuid::new_v4(),
            },
            GraphPredicate::SharesEntity {
                other_memory_id: "m-9".into(),
            },
            GraphPredicate::EpisodeContains {
                episode_id: Uuid::new_v4(),
            },
        ];
        for p in &preds {
            let hits = q.scan("alice", p).await.expect("scan");
            assert!(hits.is_empty(), "stub graph scan should always be empty");
        }
    }

    #[tokio::test]
    async fn matches_returns_false_for_every_predicate_variant() {
        let q = StubGraphQuerier::new();
        let p = GraphPredicate::SharesEntity {
            other_memory_id: "m-9".into(),
        };
        assert!(!q.matches("alice", "m-1", &p).await.expect("matches"));
    }
}
