//! Bounded reads of the graph for the SQLite mirror: a family a key prefix at a
//! time, and the rows its change log names.

use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

use super::change_log::GraphChangeLog;
use super::{EntityNode, GraphMemory, RelationshipEdge, CF_ENTITIES, CF_RELATIONSHIPS};

impl GraphMemory {
    /// The log of this graph's entity and relationship writes.
    pub fn change_log(&self) -> &Arc<GraphChangeLog> {
        &self.change_log
    }

    /// Entities whose key starts with `prefix`; rows that do not decode are skipped.
    pub fn entities_with_key_prefix(&self, prefix: &[u8]) -> Result<Vec<EntityNode>> {
        Ok(self
            .store
            .scan_cf(CF_ENTITIES, Some(prefix), None)?
            .into_iter()
            .filter_map(|(_, value)| {
                bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                )
                .ok()
                .map(|(entity, _)| entity)
            })
            .collect())
    }

    /// Live relationships whose key starts with `prefix`. Lookup rows and rows that
    /// do not decode are skipped.
    pub fn relationships_with_key_prefix(&self, prefix: &[u8]) -> Result<Vec<RelationshipEdge>> {
        Ok(self
            .store
            .scan_cf(CF_RELATIONSHIPS, Some(prefix), None)?
            .into_iter()
            .filter(|(key, _)| key.len() == 16)
            .filter_map(|(_, value)| RelationshipEdge::decode(&value).ok())
            .filter(|edge| edge.invalidated_at.is_none())
            .collect())
    }

    /// The relationship stored under `uuid` unless it is gone or invalidated.
    pub fn live_relationship(&self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        Ok(self
            .get_relationship(uuid)?
            .filter(|edge| edge.invalidated_at.is_none()))
    }

    /// Every live relationship with `entity` at either end.
    pub fn live_relationships_of(&self, entity: &Uuid) -> Result<Vec<RelationshipEdge>> {
        let mut edges = self.stored_entity_relationships(entity)?;
        edges.retain(|edge| edge.invalidated_at.is_none());
        Ok(edges)
    }
}
