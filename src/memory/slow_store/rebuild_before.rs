//! The mirror rebuild as it ran before chunked syncs, kept to measure against:
//! 2048-row transactions with no write turn, the projection's automatic
//! checkpoint left on, the retired tables dropped in one statement, and a
//! pair keeping its first edge's uuid with its last edge's strength.

use anyhow::Result;
use rusqlite::{params, TransactionBehavior};
use std::collections::HashSet;
use uuid::Uuid;

use super::{
    count_left_out, edges_table_ddl, entities_table_ddl, index_exists, seal_text_value, SlowStore,
    SyncStats, ALT_INDEX_SUFFIX, EDGE_INDEXES, EDGE_PAIR_INDEX,
};
use crate::graph_memory::{EntityNode, RelationshipEdge};

const CHUNK_ROWS: usize = 2048;

impl SlowStore {
    pub(crate) fn rebuild_mirror_as_before(
        &self,
        entities: &[EntityNode],
        edges: &[RelationshipEdge],
    ) -> Result<()> {
        let mut conn = Self::open_writer(&self.path)?;
        conn.execute_batch(
            "DROP TABLE IF EXISTS edges_shadow;
             DROP TABLE IF EXISTS entities_shadow;
             DROP TABLE IF EXISTS edges_old;
             DROP TABLE IF EXISTS entities_old;",
        )?;
        let suffix = if index_exists(&conn, EDGE_PAIR_INDEX)? {
            ALT_INDEX_SUFFIX
        } else {
            ""
        };
        conn.execute_batch(&entities_table_ddl("entities_shadow"))?;
        conn.execute_batch(&edges_table_ddl("edges_shadow", "entities_shadow"))?;
        for (base, columns, unique) in EDGE_INDEXES {
            let unique = if unique { "UNIQUE " } else { "" };
            conn.execute_batch(&format!(
                "CREATE {unique}INDEX {base}{suffix} ON edges_shadow({columns});"
            ))?;
        }
        let mut stats = SyncStats::default();
        for chunk in entities.chunks(CHUNK_ROWS) {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            Self::sync_entities(
                &tx,
                "entities_shadow",
                chunk,
                &mut stats,
                self.encryptor.as_ref(),
                &self.enc,
            )?;
            tx.commit()?;
        }
        let synced: HashSet<Uuid> = entities.iter().map(|e| e.uuid).collect();
        for chunk in edges.chunks(CHUNK_ROWS) {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut stmt = tx.prepare_cached(
                    "INSERT INTO edges_shadow (uuid, from_entity, to_entity, relation_type,
                                               strength, tier, ltp_status, activation_count,
                                               created_at, last_activated, context)
                     VALUES (?1, ?2, ?3, ?4, ?5, 'L1Working', 'None', ?6, ?7, ?8, ?9)
                     ON CONFLICT(from_entity, to_entity) DO UPDATE SET
                        strength = excluded.strength,
                        activation_count = excluded.activation_count,
                        last_activated = excluded.last_activated,
                        context = excluded.context",
                )?;
                for edge in chunk {
                    if !synced.contains(&edge.from_entity) || !synced.contains(&edge.to_entity) {
                        count_left_out(edge, &mut stats);
                        continue;
                    }
                    let uuid = edge.uuid.to_string();
                    let context = seal_text_value(
                        self.encryptor.as_ref(),
                        &edge.context,
                        self.enc.graph,
                        uuid.as_bytes(),
                    )?;
                    stmt.execute(params![
                        uuid,
                        edge.from_entity.to_string(),
                        edge.to_entity.to_string(),
                        edge.relation_type.as_str(),
                        edge.strength,
                        edge.activation_count,
                        edge.created_at.to_rfc3339(),
                        edge.last_activated.to_rfc3339(),
                        context,
                    ])?;
                }
            }
            tx.commit()?;
        }
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "ALTER TABLE edges RENAME TO edges_old;
             ALTER TABLE entities RENAME TO entities_old;
             ALTER TABLE entities_shadow RENAME TO entities;
             ALTER TABLE edges_shadow RENAME TO edges;",
        )?;
        tx.commit()?;
        conn.execute_batch("DROP TABLE edges_old; DROP TABLE entities_old;")?;
        conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()))?;
        Ok(())
    }
}
