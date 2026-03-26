//! Gap and thought persistence.
//!
//! CRUD operations for detected gap topologies and generated thoughts
//! in the SQLite slow store.

use anyhow::Result;
use chrono::Utc;
use rusqlite::params;
use serde::{Deserialize, Serialize};

use super::SlowStore;

/// A stored gap topology record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredGap {
    pub id: String,
    pub gap_type: String,
    pub shape_signature: String,
    pub entities_json: String,
    pub missing_links_json: String,
    pub confidence: f32,
    pub embedding_distance: Option<f32>,
    pub impact_score: f32,
    pub detected_at: String,
    pub scope: String,
}

/// A stored thought record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredThought {
    pub id: String,
    pub kind: String,
    pub scope: String,
    pub confidence: f32,
    pub description: String,
    pub hypothesis: Option<String>,
    pub evidence_json: String,
    pub impact_score: f32,
    pub entities_json: String,
    pub created_at: String,
    pub surfaced_count: usize,
}

impl SlowStore {
    /// Store a detected gap topology
    #[allow(clippy::too_many_arguments)]
    pub fn store_gap(
        &self,
        id: &str,
        gap_type: &str,
        shape_signature: &str,
        entities_json: &str,
        missing_links_json: &str,
        confidence: f32,
        embedding_distance: Option<f32>,
        impact_score: f32,
        scope: &str,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO gap_topologies
                (id, gap_type, shape_signature, entities_json, missing_links_json,
                 confidence, embedding_distance, impact_score, detected_at, last_verified, scope)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                embedding_distance = excluded.embedding_distance,
                impact_score = excluded.impact_score,
                last_verified = excluded.last_verified",
            params![
                id,
                gap_type,
                shape_signature,
                entities_json,
                missing_links_json,
                confidence,
                embedding_distance,
                impact_score,
                now,
                scope,
            ],
        )?;
        Ok(())
    }

    /// Store a generated thought
    #[allow(clippy::too_many_arguments)]
    pub fn store_thought(
        &self,
        id: &str,
        kind: &str,
        scope: &str,
        confidence: f32,
        description: &str,
        hypothesis: Option<&str>,
        evidence_json: &str,
        impact_score: f32,
        entities_json: &str,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO thoughts
                (id, kind, scope, confidence, description, hypothesis,
                 evidence_json, impact_score, entities_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                description = excluded.description,
                hypothesis = excluded.hypothesis,
                impact_score = excluded.impact_score",
            params![
                id,
                kind,
                scope,
                confidence,
                description,
                hypothesis,
                evidence_json,
                impact_score,
                entities_json,
                now,
            ],
        )?;
        Ok(())
    }

    /// Get active (non-dismissed) thoughts, ordered by confidence
    pub fn get_active_thoughts(&self, limit: usize) -> Result<Vec<StoredThought>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT id, kind, scope, confidence, description, hypothesis,
                    evidence_json, impact_score, entities_json, created_at,
                    surfaced_count
             FROM thoughts
             WHERE dismissed = 0
             ORDER BY impact_score DESC, confidence DESC
             LIMIT ?1",
        )?;

        let thoughts = stmt
            .query_map(params![limit], |row| {
                Ok(StoredThought {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    scope: row.get(2)?,
                    confidence: row.get(3)?,
                    description: row.get(4)?,
                    hypothesis: row.get(5)?,
                    evidence_json: row.get(6)?,
                    impact_score: row.get(7)?,
                    entities_json: row.get(8)?,
                    created_at: row.get(9)?,
                    surfaced_count: row.get(10)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(thoughts)
    }

    /// Mark a thought as surfaced (increment count)
    pub fn mark_thought_surfaced(&self, thought_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE thoughts SET surfaced_count = surfaced_count + 1 WHERE id = ?1",
            params![thought_id],
        )?;
        Ok(())
    }

    /// Dismiss a thought (user decided it's not useful)
    pub fn dismiss_thought(&self, thought_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE thoughts SET dismissed = 1 WHERE id = ?1",
            params![thought_id],
        )?;
        Ok(())
    }

    /// Mark a gap as resolved (an edge was created that closes it)
    pub fn resolve_gap(&self, gap_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE gap_topologies SET resolved_at = ?1 WHERE id = ?2",
            params![now, gap_id],
        )?;
        Ok(())
    }

    /// Get unresolved gaps by type
    pub fn get_unresolved_gaps(&self, gap_type: &str, limit: usize) -> Result<Vec<StoredGap>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT id, gap_type, shape_signature, entities_json, missing_links_json,
                    confidence, embedding_distance, impact_score, detected_at, scope
             FROM gap_topologies
             WHERE resolved_at IS NULL AND gap_type = ?1
             ORDER BY impact_score DESC, confidence DESC
             LIMIT ?2",
        )?;

        let gaps = stmt
            .query_map(params![gap_type, limit], |row| {
                Ok(StoredGap {
                    id: row.get(0)?,
                    gap_type: row.get(1)?,
                    shape_signature: row.get(2)?,
                    entities_json: row.get(3)?,
                    missing_links_json: row.get(4)?,
                    confidence: row.get(5)?,
                    embedding_distance: row.get(6)?,
                    impact_score: row.get(7)?,
                    detected_at: row.get(8)?,
                    scope: row.get(9)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(gaps)
    }
}
