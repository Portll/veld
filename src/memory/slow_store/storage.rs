//! Gap and thought persistence.
//!
//! CRUD operations for detected gap topologies and generated thoughts
//! in the SQLite slow store.

use anyhow::Result;
use chrono::Utc;
use rusqlite::{params, OptionalExtension};
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

/// A row from the `memories` projection table.
///
/// One row per `(user_id, memory_id)`. `lsn` is the intent-log position of
/// the write that produced this row — used by the projection layer as a
/// write-skew tie-breaker when a live UPSERT races against a replay
/// re-apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMemoryRow {
    pub user_id: String,
    pub memory_id: String,
    pub lsn: u64,
    pub memory_bincode: Vec<u8>,
    pub importance: f32,
    pub updated_at: String,
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
                last_verified = excluded.last_verified,
                resolved_at = NULL",
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

    /// Get all unresolved gaps for a scope (across gap types), most impactful
    /// first. Scope-filtered to prevent cross-user gap leakage (M5).
    pub fn get_unresolved_gaps_all(&self, scope: &str, limit: usize) -> Result<Vec<StoredGap>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT id, gap_type, shape_signature, entities_json, missing_links_json,
                    confidence, embedding_distance, impact_score, detected_at, scope
             FROM gap_topologies
             WHERE resolved_at IS NULL AND scope = ?1
             ORDER BY impact_score DESC, confidence DESC
             LIMIT ?2",
        )?;
        let gaps = stmt
            .query_map(params![scope, limit], |row| {
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

    /// Unresolved gaps whose entity set intersects `entity_names`
    /// (case-insensitive), scope-filtered. The query-in-gap primitive: tells
    /// whether a query sits in/near a known knowledge gap (M5). Surface only —
    /// never triggers acquisition (avoids a curiosity feedback loop).
    pub fn gaps_touching_entities(
        &self,
        scope: &str,
        entity_names: &[String],
        limit: usize,
    ) -> Result<Vec<StoredGap>> {
        if entity_names.is_empty() {
            return Ok(Vec::new());
        }
        let wanted: Vec<String> = entity_names.iter().map(|n| n.to_lowercase()).collect();
        // Scan the scoped, impact-ordered, bounded unresolved set and match in
        // Rust against the stored entity names.
        let candidates = self.get_unresolved_gaps_all(scope, 500)?;
        let mut hits = Vec::new();
        for gap in candidates {
            let matched = serde_json::from_str::<serde_json::Value>(&gap.entities_json)
                .ok()
                .and_then(|v| {
                    v.as_array().map(|arr| {
                        arr.iter().any(|e| {
                            e.get("name")
                                .and_then(|n| n.as_str())
                                .map(|n| wanted.iter().any(|w| w == &n.to_lowercase()))
                                .unwrap_or(false)
                        })
                    })
                })
                .unwrap_or(false);
            if matched {
                hits.push(gap);
                if hits.len() >= limit {
                    break;
                }
            }
        }
        Ok(hits)
    }

    /// Resolve gaps not re-verified since `cutoff_rfc3339` (hysteresis: a gap
    /// must be absent for the stale window before being marked resolved, so
    /// transient graph churn does not flap it). Scope-filtered. Returns the
    /// number resolved. A re-detected gap is re-opened by `store_gap` clearing
    /// `resolved_at`.
    pub fn resolve_stale_gaps(&self, scope: &str, cutoff_rfc3339: &str) -> Result<usize> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        let n = conn.execute(
            "UPDATE gap_topologies SET resolved_at = ?1
             WHERE scope = ?2 AND resolved_at IS NULL AND last_verified < ?3",
            params![now, scope, cutoff_rfc3339],
        )?;
        Ok(n)
    }

    // ========================================================================
    // Memory projection table — populated by `SqliteProjection`.
    //
    // These methods are the SQL substrate the projection layer calls. They
    // are intentionally simple (no business logic, no caching) — every
    // write is keyed by `(user_id, memory_id)` and stamped with the
    // intent-log LSN so a replay can resolve write-skew with a higher-wins
    // rule.
    // ========================================================================

    /// UPSERT a memory row at a specific LSN. The `ON CONFLICT` clause
    /// keeps the row with the *higher* LSN — so a live write at LSN 7 is
    /// never overwritten by a stale replay at LSN 3. Equal LSN replays
    /// (the normal idempotent re-apply) overwrite without semantic change
    /// because the bytes are identical.
    pub fn upsert_memory(
        &self,
        user_id: &str,
        memory_id: &str,
        lsn: u64,
        memory_bincode: &[u8],
        importance: f32,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO memories
                (user_id, memory_id, lsn, memory_bincode, importance, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(user_id, memory_id) DO UPDATE SET
                lsn = excluded.lsn,
                memory_bincode = excluded.memory_bincode,
                importance = excluded.importance,
                updated_at = excluded.updated_at
             WHERE excluded.lsn >= memories.lsn",
            params![user_id, memory_id, lsn as i64, memory_bincode, importance, now],
        )?;
        Ok(())
    }

    /// UPDATE only the `importance` column for `(user_id, memory_id)`,
    /// stamping the supplied LSN. Equivalent to `upsert_memory` but
    /// doesn't require a fresh memory bincode — used for `IntentPayload::Anchor`.
    /// Like `upsert_memory`, the update is gated on `lsn >= current` to
    /// preserve the higher-wins invariant.
    ///
    /// If the row does not exist (anchoring a memory that was never
    /// recorded in the projection), this is a silent no-op — replay will
    /// see the preceding `Remember`/`Update` first and then re-apply the
    /// anchor with the correct LSN ordering.
    pub fn anchor_memory_importance(
        &self,
        user_id: &str,
        memory_id: &str,
        lsn: u64,
        importance: f32,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE memories
                SET importance = ?3, lsn = ?4, updated_at = ?5
              WHERE user_id = ?1 AND memory_id = ?2 AND ?4 >= lsn",
            params![user_id, memory_id, importance, lsn as i64, now],
        )?;
        Ok(())
    }

    /// DELETE the row for `(user_id, memory_id)`. Idempotent — a delete
    /// against a non-existent row returns successfully with zero rows
    /// affected.
    pub fn delete_memory(&self, user_id: &str, memory_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM memories WHERE user_id = ?1 AND memory_id = ?2",
            params![user_id, memory_id],
        )?;
        Ok(())
    }

    /// Fetch the bincoded memory blob for `(user_id, memory_id)`, or
    /// `None` if the row does not exist. Read-only — used by tests and by
    /// the future "read-your-writes" check from the SQLite projection.
    pub fn get_memory_blob(
        &self,
        user_id: &str,
        memory_id: &str,
    ) -> Result<Option<StoredMemoryRow>> {
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT user_id, memory_id, lsn, memory_bincode, importance, updated_at
                 FROM memories
                 WHERE user_id = ?1 AND memory_id = ?2",
                params![user_id, memory_id],
                |row| {
                    let lsn_i64: i64 = row.get(2)?;
                    Ok(StoredMemoryRow {
                        user_id: row.get(0)?,
                        memory_id: row.get(1)?,
                        lsn: lsn_i64 as u64,
                        memory_bincode: row.get(3)?,
                        importance: row.get(4)?,
                        updated_at: row.get(5)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Total number of memory rows for a tenant. Used by tests and the
    /// admin dashboard to confirm replay caught up.
    pub fn count_memories(&self, user_id: &str) -> Result<u64> {
        let conn = self.conn.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?1",
            params![user_id],
            |row| row.get(0),
        )?;
        Ok(count as u64)
    }
}
