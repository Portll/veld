//! Embedding loading from the SQLite slow store.
//!
//! Provides bulk and targeted embedding retrieval for gap topology scoring,
//! Voronoi analysis, and Planet X detection.

use anyhow::Result;
use rusqlite::params;
use std::collections::HashMap;

use super::SlowStore;

impl SlowStore {
    /// Load embeddings for a set of entity UUIDs.
    ///
    /// Returns a map from UUID string to embedding vector.
    /// Used by gap topology scoring to compute cosine similarity between gap endpoints.
    pub fn load_embeddings(&self, uuids: &[&str]) -> Result<HashMap<String, Vec<f32>>> {
        let conn = self.conn.lock();
        let mut result = HashMap::new();

        let mut stmt = conn.prepare_cached(
            "SELECT uuid, embedding FROM entities WHERE uuid = ?1 AND embedding IS NOT NULL",
        )?;

        for uuid in uuids {
            if let Ok(row) = stmt.query_row(params![uuid], |row| {
                let uuid_str: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((uuid_str, blob))
            }) {
                // Decode f32 embedding from little-endian bytes
                let floats: Vec<f32> = row
                    .1
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                if !floats.is_empty() {
                    result.insert(row.0, floats);
                }
            }
        }

        Ok(result)
    }

    /// Load ALL entity embeddings from the slow store.
    ///
    /// Returns (uuid, name, embedding) triples for all entities that have embeddings.
    /// Used by Voronoi analysis to reason about the shape of knowledge space.
    pub fn load_all_embeddings(&self) -> Result<Vec<(String, String, Vec<f32>)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT uuid, name, embedding FROM entities WHERE embedding IS NOT NULL",
        )?;

        let results = stmt
            .query_map([], |row| {
                let uuid: String = row.get(0)?;
                let name: String = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                Ok((uuid, name, blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(uuid, name, blob)| {
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                if floats.is_empty() {
                    None
                } else {
                    Some((uuid, name, floats))
                }
            })
            .collect();

        Ok(results)
    }

    /// Load all directed edges (from_uuid, to_uuid) from the slow store.
    ///
    /// Used by Planet X detection to find where edges converge.
    pub fn load_all_edge_pairs(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare_cached("SELECT from_entity, to_entity FROM edges")?;

        let results = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(results)
    }
}
