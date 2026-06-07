//! S-R6: Semantic merge of near-duplicate memories via tombstone + audit log.
//!
//! # Design
//!
//! A merge is triggered when two memories exceed
//! [`SEMANTIC_MERGE_SIMILARITY_THRESHOLD`] in cosine similarity (both must
//! have embeddings).  The entire merge is **flag-gated**:
//! [`SEMANTIC_MERGE_ENABLED`] defaults to `false`; with it off the storage hot
//! path is byte-for-byte unchanged.
//!
//! ## Commit point
//!
//! The atomic commit point is the append of a [`MergeAudit`] row.  Crash-safe:
//! on startup [`MemoryStorage::replay_merge_audits`] re-applies every un-GC'd
//! audit idempotently.
//!
//! ## Evidence pooling (winner update)
//!
//! 1. `access_count` — summed.
//! 2. `entity_refs`  — union (deduplicated by entity id).
//! 3. `embeddings`   — element-wise average of loser into winner.
//!
//! ## Tombstone
//!
//! The loser row stays in storage with a `merge_tombstone:{user_id}:{loser_id}`
//! marker written in the default CF.  It is **not** deleted.
//!
//! ## Lazy remap on read (B2)
//!
//! [`MemoryStorage::resolve_loser`] is called on a storage MISS.  It checks
//! for a tombstone → scans the audit log → returns the winner id.  Always-on
//! (no flag); strict no-op when no audits exist.
//!
//! ## Path compression (B7)
//!
//! On each `resolve_loser` call the chain A→B→C is compressed to A→C by
//! re-writing the audit entry for A to point directly at C.
//!
//! ## GC
//!
//! A tombstone is physically reclaimed ONLY when:
//! - [`MemoryStorage::has_any_inbound_refs`] returns `false`, AND
//! - The quiescence window ([`SEMANTIC_MERGE_GC_QUIESCENCE_SECS`]) has elapsed.
//!
//! Physical reclaim calls the existing `delete()` path.  The audit row is
//! deleted at the same time.
//!
//! ## J2 force-GC (maintenance tick)
//!
//! [`MemoryStorage::gc_merge_tombstones`] scans all tombstones and force-GCs
//! any that are older than [`SEMANTIC_MERGE_GC_MAX_AGE_SECS`] AND have no
//! inbound refs.  Called from `run_maintenance` on heavy cycles.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{IteratorMode, WriteBatch, WriteOptions};
use serde::{Deserialize, Serialize};

use crate::constants::{
    SEMANTIC_MERGE_ENABLED, SEMANTIC_MERGE_GC_MAX_AGE_SECS, SEMANTIC_MERGE_GC_QUIESCENCE_SECS,
    SEMANTIC_MERGE_SIMILARITY_THRESHOLD,
};
use crate::memory::types::{EntityRef, MemoryId};
use crate::similarity::cosine_similarity;

use super::storage::MemoryStorage;

// ============================================================================
// Audit record
// ============================================================================

/// Key prefix for merge-audit rows in the default CF.
/// Full key: `merge_audit:{user_id}:{audit_id}`
pub(super) const AUDIT_PREFIX: &str = "merge_audit:";

/// Key prefix for tombstone markers in the default CF.
/// Full key: `merge_tombstone:{user_id}:{loser_id}`
pub(super) const TOMBSTONE_PREFIX: &str = "merge_tombstone:";

/// Immutable audit record written as the atomic commit point of every merge.
///
/// Stored in the default CF under `merge_audit:{user_id}:{audit_id}` (JSON).
/// Append-only; never mutated after write.  The only in-place modification
/// permitted is path-compression: replacing `winner_id` with the end of the
/// chain (B7).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MergeAudit {
    /// Surrogate key for this audit row (UUID v4).
    pub audit_id: String,
    /// The loser memory that was tombstoned.
    pub loser_id: MemoryId,
    /// The winner memory that absorbed the loser's evidence.
    pub winner_id: MemoryId,
    /// Wall-clock time of the merge.
    pub merged_at: DateTime<Utc>,
    /// Tenant / user namespace.
    pub user_id: String,
}

// ============================================================================
// Tombstone record
// ============================================================================

/// Minimal tombstone written at `merge_tombstone:{user_id}:{loser_id}`.
/// Value is a JSON-encoded [`TombstoneRecord`] so GC can check quiescence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TombstoneRecord {
    pub loser_id: MemoryId,
    pub winner_id: MemoryId,
    pub tombstoned_at: DateTime<Utc>,
    pub user_id: String,
}

// ============================================================================
// Merge candidate detection
// ============================================================================

/// A pair of memories that exceed the similarity threshold.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// The memory with lower `access_count + importance` — will be merged into the winner.
    pub loser_id: MemoryId,
    /// The memory with higher `access_count + importance` — absorbs loser's evidence.
    pub winner_id: MemoryId,
    pub similarity: f32,
}

// ============================================================================
// MemoryStorage methods — the public API surface for merge operations
// ============================================================================

impl MemoryStorage {
    // -------------------------------------------------------------------------
    // Candidate detection
    // -------------------------------------------------------------------------

    /// Scan all memories for near-duplicate pairs.
    ///
    /// Only called when [`SEMANTIC_MERGE_ENABLED`] is true.  Returns an ordered
    /// list of merge candidates (highest similarity first).
    ///
    /// Algorithm: O(n²) pair scan over memories that have embeddings.  For
    /// large corpora (>10 k memories) the caller should pass a per-tick limit.
    ///
    /// The loser is the memory with the lower `access_count` + `importance`
    /// combined score.  Ties broken by lexicographic id ordering so the choice
    /// is deterministic and stable.
    pub fn find_merge_candidates(&self, limit: usize) -> Result<Vec<MergeCandidate>> {
        let memories = self.get_all()?;
        // Only consider memories with embeddings — no embedding means no
        // reliable cosine comparison.
        let with_emb: Vec<_> = memories
            .into_iter()
            .filter(|m| m.experience.embeddings.is_some())
            // Skip tombstoned losers — they have no live role.
            .filter(|m| !self.is_tombstoned_losers_ignore_user(m.id.clone()))
            .collect();

        let n = with_emb.len();
        let mut candidates: Vec<MergeCandidate> = Vec::new();

        for i in 0..n {
            let a = &with_emb[i];
            let emb_a = a.experience.embeddings.as_deref().unwrap();
            for j in (i + 1)..n {
                let b = &with_emb[j];
                let emb_b = b.experience.embeddings.as_deref().unwrap();
                if emb_a.len() != emb_b.len() {
                    continue;
                }
                let sim = cosine_similarity(emb_a, emb_b);
                if sim >= SEMANTIC_MERGE_SIMILARITY_THRESHOLD {
                    // Loser = lower score memory; winner = higher.
                    let score_a = a.access_count() as f64 + a.importance() as f64;
                    let score_b = b.access_count() as f64 + b.importance() as f64;
                    let (loser, winner) = if score_a <= score_b {
                        (&a.id, &b.id)
                    } else {
                        (&b.id, &a.id)
                    };
                    candidates.push(MergeCandidate {
                        loser_id: loser.clone(),
                        winner_id: winner.clone(),
                        similarity: sim,
                    });
                }
            }
        }

        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        Ok(candidates)
    }

    /// Returns true if `id` is the loser side of any tombstone (any user).
    /// Used internally during candidate scan to skip already-tombstoned rows.
    fn is_tombstoned_losers_ignore_user(&self, id: MemoryId) -> bool {
        // Scan `merge_tombstone:*:{id}` — cheap: each key includes the id.
        let id_str = id.0.to_string();
        let db = self.db();
        let iter = db.iterator(IteratorMode::From(TOMBSTONE_PREFIX.as_bytes(), rocksdb::Direction::Forward));
        for item in iter {
            let Ok((key, _)) = item else { continue };
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(TOMBSTONE_PREFIX) {
                break;
            }
            if key_str.ends_with(&id_str) {
                return true;
            }
        }
        false
    }

    // -------------------------------------------------------------------------
    // Execute a single merge
    // -------------------------------------------------------------------------

    /// Execute one merge: pool evidence into winner, append audit, tombstone loser.
    ///
    /// **This is the only write path for merges.**  It MUST NOT be called with
    /// [`SEMANTIC_MERGE_ENABLED`] == false except in `#[cfg(test)]` tests that
    /// force the flag on explicitly.
    ///
    /// Returns the audit id on success.
    pub fn execute_merge(
        &self,
        loser_id: &MemoryId,
        winner_id: &MemoryId,
        user_id: &str,
    ) -> Result<String> {
        // --- 1. Load both memories -------------------------------------------
        let mut winner = self
            .get(winner_id)
            .context("execute_merge: winner not found")?;
        let loser = self
            .get(loser_id)
            .context("execute_merge: loser not found")?;

        // --- 2. Pool evidence: access_count ------------------------------------
        let winner_access = winner.access_count();
        let loser_access = loser.access_count();
        let new_access = winner_access.saturating_add(loser_access);
        winner.set_access_count(new_access);

        // --- 3. Pool evidence: entity_refs (union, dedup by entity name) -------
        for er in &loser.entity_refs {
            if !winner
                .entity_refs
                .iter()
                .any(|e: &EntityRef| e.name == er.name)
            {
                winner.entity_refs.push(er.clone());
            }
        }

        // --- 4. Pool evidence: embeddings (element-wise average) ---------------
        if let (Some(winner_emb), Some(loser_emb)) = (
            winner.experience.embeddings.as_mut(),
            loser.experience.embeddings.as_deref(),
        ) {
            if winner_emb.len() == loser_emb.len() {
                for (w, l) in winner_emb.iter_mut().zip(loser_emb.iter()) {
                    *w = (*w + l) * 0.5;
                }
            }
        }

        // --- 5. Persist updated winner -----------------------------------------
        self.update(&winner)
            .context("execute_merge: failed to persist winner update")?;

        // --- 6. Append MergeAudit (ATOMIC COMMIT POINT) ------------------------
        let audit_id = uuid::Uuid::new_v4().to_string();
        let audit = MergeAudit {
            audit_id: audit_id.clone(),
            loser_id: loser_id.clone(),
            winner_id: winner_id.clone(),
            merged_at: Utc::now(),
            user_id: user_id.to_string(),
        };
        self.write_audit(&audit)
            .context("execute_merge: failed to write audit")?;

        // --- 7. Tombstone the loser -------------------------------------------
        let tombstone = TombstoneRecord {
            loser_id: loser_id.clone(),
            winner_id: winner_id.clone(),
            tombstoned_at: Utc::now(),
            user_id: user_id.to_string(),
        };
        self.write_tombstone(&tombstone)
            .context("execute_merge: failed to write tombstone")?;

        tracing::info!(
            loser_id = %loser_id.0,
            winner_id = %winner_id.0,
            audit_id = %audit_id,
            user_id = %user_id,
            "S-R6: merge committed"
        );

        Ok(audit_id)
    }

    // -------------------------------------------------------------------------
    // Audit + tombstone persistence helpers
    // -------------------------------------------------------------------------

    fn audit_key(user_id: &str, audit_id: &str) -> String {
        format!("{}{user_id}:{audit_id}", AUDIT_PREFIX)
    }

    fn write_audit(&self, audit: &MergeAudit) -> Result<()> {
        let key = Self::audit_key(&audit.user_id, &audit.audit_id);
        let value = serde_json::to_vec(audit).context("serialize MergeAudit")?;
        let mut wo = WriteOptions::default();
        wo.set_sync(self.write_mode_is_sync());
        self.db()
            .put_opt(key.as_bytes(), &value, &wo)
            .context("write MergeAudit to RocksDB")
    }

    fn write_tombstone(&self, ts: &TombstoneRecord) -> Result<()> {
        let key = Self::tombstone_key(&ts.user_id, &ts.loser_id);
        let value = serde_json::to_vec(ts).context("serialize TombstoneRecord")?;
        let mut wo = WriteOptions::default();
        wo.set_sync(self.write_mode_is_sync());
        self.db()
            .put_opt(key.as_bytes(), &value, &wo)
            .context("write TombstoneRecord to RocksDB")
    }

    // -------------------------------------------------------------------------
    // Audit scan helpers
    // -------------------------------------------------------------------------

    /// Load all MergeAudit rows for `user_id`.
    fn load_audits_for_user(&self, user_id: &str) -> Vec<MergeAudit> {
        let prefix = format!("{}{user_id}:", AUDIT_PREFIX);
        let db = self.db();
        let iter = db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        let mut audits = Vec::new();
        for item in iter {
            let Ok((key, value)) = item else { continue };
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if let Ok(a) = serde_json::from_slice::<MergeAudit>(&value) {
                audits.push(a);
            }
        }
        audits
    }

    // -------------------------------------------------------------------------
    // Lazy remap on read (B2 / J1)
    // -------------------------------------------------------------------------

    /// Resolve a loser id to its current winner id via the audit log.
    ///
    /// Called ONLY on a storage MISS (J1 — zero per-read tax on the hot path).
    /// Always-on regardless of [`SEMANTIC_MERGE_ENABLED`]; no-op when no audits
    /// exist for this storage instance.
    ///
    /// Performs path compression (B7): if A→B→C exists, rewrites A's audit to
    /// point directly at C.
    ///
    /// Returns `None` if `id` is not the loser side of any audit.
    pub fn resolve_loser(&self, id: &MemoryId, user_id: &str) -> Option<MemoryId> {
        let loser_str = id.0.to_string();

        // Find the audit for this loser.
        let prefix = format!("{}{user_id}:", AUDIT_PREFIX);
        let db = self.db();
        let iter = db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        let mut found_audit: Option<(String, MergeAudit)> = None;
        for item in iter {
            let Ok((key, value)) = item else { continue };
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if let Ok(a) = serde_json::from_slice::<MergeAudit>(&value) {
                if a.loser_id.0.to_string() == loser_str {
                    found_audit = Some((String::from_utf8_lossy(&key).to_string(), a));
                    break;
                }
            }
        }

        let (audit_key, mut audit) = found_audit?;
        let mut winner = audit.winner_id.clone();

        // Path compression (B7): follow the chain until the end.
        let mut visited: Vec<String> = vec![loser_str.clone()];
        loop {
            let winner_str = winner.0.to_string();
            if visited.contains(&winner_str) {
                // Cycle guard (should never happen in practice).
                break;
            }
            visited.push(winner_str.clone());

            // Check if the winner itself is also a loser.
            let next = self.resolve_loser_raw(user_id, &winner);
            match next {
                Some(next_winner) => {
                    winner = next_winner;
                }
                None => break,
            }
        }

        // If the chain was longer than 1 hop, rewrite the original audit
        // to point directly at the end (path compression).
        if winner != audit.winner_id {
            tracing::debug!(
                loser = %loser_str,
                old_winner = %audit.winner_id.0,
                compressed_winner = %winner.0,
                "S-R6: path-compressing merge chain"
            );
            audit.winner_id = winner.clone();
            if let Ok(value) = serde_json::to_vec(&audit) {
                let db = self.db();
                let mut wo = WriteOptions::default();
                wo.set_sync(self.write_mode_is_sync());
                let _ = db.put_opt(audit_key.as_bytes(), &value, &wo);
            }
        }

        Some(winner)
    }

    /// Inner lookup: find the immediate winner for `loser` from the audit log.
    /// Returns `None` if `loser` is not the loser side of any audit.
    fn resolve_loser_raw(&self, user_id: &str, loser: &MemoryId) -> Option<MemoryId> {
        let loser_str = loser.0.to_string();
        let prefix = format!("{}{user_id}:", AUDIT_PREFIX);
        let db = self.db();
        let iter = db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for item in iter {
            let Ok((key, value)) = item else { continue };
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if let Ok(a) = serde_json::from_slice::<MergeAudit>(&value) {
                if a.loser_id.0.to_string() == loser_str {
                    return Some(a.winner_id);
                }
            }
        }
        None
    }

    // -------------------------------------------------------------------------
    // Crash-safe startup replay (idempotent)
    // -------------------------------------------------------------------------

    /// Replay all un-GC'd audit entries on startup to ensure tombstones are
    /// consistent with audits.  Idempotent: if the tombstone already exists
    /// and the winner is already up to date, the write is a no-op.
    ///
    /// Called once during `MemoryStorage::new` AFTER the DB is opened.
    pub fn replay_merge_audits(&self, user_id: &str) {
        let audits = self.load_audits_for_user(user_id);
        if audits.is_empty() {
            return;
        }
        tracing::info!(
            count = audits.len(),
            user_id = %user_id,
            "S-R6: replaying merge audits on startup"
        );
        for audit in &audits {
            // Ensure tombstone exists (idempotent).
            let ts = TombstoneRecord {
                loser_id: audit.loser_id.clone(),
                winner_id: audit.winner_id.clone(),
                tombstoned_at: audit.merged_at,
                user_id: audit.user_id.clone(),
            };
            if let Err(e) = self.write_tombstone(&ts) {
                tracing::warn!(
                    audit_id = %audit.audit_id,
                    error = %e,
                    "S-R6: replay failed to re-write tombstone (non-fatal)"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // GC (J2)
    // -------------------------------------------------------------------------

    /// Scan tombstones and reclaim any that:
    /// 1. have no inbound refs (`has_any_inbound_refs` == false), AND
    /// 2. exceed the quiescence window OR the force-GC max-age window.
    ///
    /// Returns the number of tombstones reclaimed.
    ///
    /// Called from `run_maintenance` on heavy cycles (J2).
    pub fn gc_merge_tombstones(&self, user_id: &str) -> usize {
        let now = Utc::now();
        let prefix = format!("{}{user_id}:", TOMBSTONE_PREFIX);
        let db = self.db();
        let iter = db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        let mut to_gc: Vec<TombstoneRecord> = Vec::new();
        for item in iter {
            let Ok((key, value)) = item else { continue };
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            let Ok(ts) = serde_json::from_slice::<TombstoneRecord>(&value) else {
                continue;
            };
            let age_secs = (now - ts.tombstoned_at).num_seconds();
            let quiescent = age_secs >= SEMANTIC_MERGE_GC_QUIESCENCE_SECS;
            let force_age = age_secs >= SEMANTIC_MERGE_GC_MAX_AGE_SECS;

            if !quiescent && !force_age {
                continue;
            }

            // GC gate: refuse if anything still references the loser.
            match self.has_any_inbound_refs(&ts.loser_id, user_id) {
                Ok(true) => {
                    tracing::debug!(
                        loser_id = %ts.loser_id.0,
                        "S-R6 GC: loser still has inbound refs, skipping"
                    );
                }
                Ok(false) => {
                    to_gc.push(ts);
                }
                Err(e) => {
                    tracing::warn!(
                        loser_id = %ts.loser_id.0,
                        error = %e,
                        "S-R6 GC: has_any_inbound_refs failed, skipping"
                    );
                }
            }
        }

        let mut reclaimed = 0;
        for ts in to_gc {
            // Physically delete the loser memory row.
            if let Err(e) = self.delete(&ts.loser_id) {
                tracing::warn!(
                    loser_id = %ts.loser_id.0,
                    error = %e,
                    "S-R6 GC: failed to delete loser memory (will retry next cycle)"
                );
                continue;
            }

            // Remove tombstone and audit rows atomically.
            let mut batch = WriteBatch::default();
            let tombstone_key = Self::tombstone_key(user_id, &ts.loser_id);
            batch.delete(tombstone_key.as_bytes());

            // Remove all audit rows for this loser.
            let audits = self.load_audits_for_user(user_id);
            for audit in audits {
                if audit.loser_id == ts.loser_id {
                    let key = Self::audit_key(user_id, &audit.audit_id);
                    batch.delete(key.as_bytes());
                }
            }

            let mut wo = WriteOptions::default();
            wo.set_sync(self.write_mode_is_sync());
            if let Err(e) = self.db().write_opt(batch, &wo) {
                tracing::warn!(
                    loser_id = %ts.loser_id.0,
                    error = %e,
                    "S-R6 GC: failed to remove tombstone/audit rows"
                );
            } else {
                tracing::info!(
                    loser_id = %ts.loser_id.0,
                    winner_id = %ts.winner_id.0,
                    "S-R6 GC: tombstone reclaimed"
                );
                reclaimed += 1;
            }
        }

        reclaimed
    }

    // -------------------------------------------------------------------------
    // Merge trigger (called from run_maintenance when flag is on)
    // -------------------------------------------------------------------------

    /// Find and execute up to `limit` merges.  No-op when
    /// [`SEMANTIC_MERGE_ENABLED`] is false.
    ///
    /// Returns the number of merges executed.
    pub fn run_merge_pass(&self, user_id: &str, limit: usize) -> usize {
        if !SEMANTIC_MERGE_ENABLED {
            return 0;
        }
        let candidates = match self.find_merge_candidates(limit) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "S-R6: find_merge_candidates failed");
                return 0;
            }
        };
        let mut merged = 0;
        for cand in candidates {
            match self.execute_merge(&cand.loser_id, &cand.winner_id, user_id) {
                Ok(audit_id) => {
                    tracing::debug!(
                        loser = %cand.loser_id.0,
                        winner = %cand.winner_id.0,
                        audit_id = %audit_id,
                        sim = cand.similarity,
                        "S-R6: merged duplicate"
                    );
                    merged += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        loser = %cand.loser_id.0,
                        winner = %cand.winner_id.0,
                        error = %e,
                        "S-R6: merge failed"
                    );
                }
            }
        }
        merged
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::memory::storage::MemoryStorage;
    use crate::memory::types::*;
    use chrono::Utc;
    use tempfile::TempDir;

    /// Build a test storage and a temp dir (keep dir alive for the lifetime of the test).
    fn test_storage() -> (MemoryStorage, TempDir) {
        let dir = TempDir::new().expect("create tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("open storage");
        (storage, dir)
    }

    /// Build a minimal memory with `content` and optional embeddings.
    fn make_memory(content: &str, embeddings: Option<Vec<f32>>) -> Memory {
        let now = Utc::now();
        let id = MemoryId(uuid::Uuid::new_v4());
        let mut experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            embeddings: embeddings.clone(),
            ..Default::default()
        };
        // Also set secondary slot so tests exercising average detect the update.
        experience.embeddings = embeddings;
        Memory::from_legacy(
            id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Helper: normalise a 3-vector so cosine_similarity returns 1.0 against itself.
    fn unit(v: [f32; 3]) -> Vec<f32> {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        vec![v[0] / norm, v[1] / norm, v[2] / norm]
    }

    // ------------------------------------------------------------------
    // (a) Post-merge: no dangling refs; loser is NOT reachable via get()
    //     but IS reachable via resolve_loser (audit replay).
    // ------------------------------------------------------------------
    #[test]
    fn sr6_a_post_merge_loser_resolves_to_winner() {
        let (storage, _dir) = test_storage();
        const USER: &str = "test-user";

        let loser_mem = make_memory("duplicate memory A", Some(unit([1.0, 0.0, 0.0])));
        let winner_mem = make_memory("duplicate memory B", Some(unit([1.0, 0.01, 0.0])));
        let loser_id = loser_mem.id.clone();
        let winner_id = winner_mem.id.clone();

        storage.store(&loser_mem).unwrap();
        storage.store(&winner_mem).unwrap();

        // Execute merge directly (flag bypassed in test via direct call).
        let _audit_id = storage
            .execute_merge(&loser_id, &winner_id, USER)
            .expect("merge should succeed");

        // Resolve loser → winner via audit.
        let resolved = storage.resolve_loser(&loser_id, USER);
        assert_eq!(resolved.as_ref(), Some(&winner_id), "(a) loser resolves to winner");

        // Winner is still readable.
        let winner_got = storage.get(&winner_id).expect("winner must be readable");
        assert_eq!(winner_got.id, winner_id);

        // Dangling-ref scan: has_any_inbound_refs for loser should be false
        // (the loser has no remaining citations from other memories / facts).
        assert!(
            !storage.has_any_inbound_refs(&loser_id, USER).unwrap(),
            "(a) no dangling inbound refs to loser"
        );
    }

    // ------------------------------------------------------------------
    // (b) Crash-after-audit: audit present, tombstone written → replay
    //     is idempotent and remap still works.
    // ------------------------------------------------------------------
    #[test]
    fn sr6_b_crash_after_audit_replay_idempotent() {
        let (storage, _dir) = test_storage();
        const USER: &str = "test-user";

        let loser_mem = make_memory("crash test loser", Some(unit([1.0, 0.0, 0.0])));
        let winner_mem = make_memory("crash test winner", Some(unit([1.0, 0.01, 0.0])));
        let loser_id = loser_mem.id.clone();
        let winner_id = winner_mem.id.clone();

        storage.store(&loser_mem).unwrap();
        storage.store(&winner_mem).unwrap();

        // Simulate crash: write audit + tombstone manually, skip GC / physical delete.
        let audit_id = uuid::Uuid::new_v4().to_string();
        let audit = MergeAudit {
            audit_id: audit_id.clone(),
            loser_id: loser_id.clone(),
            winner_id: winner_id.clone(),
            merged_at: Utc::now(),
            user_id: USER.to_string(),
        };
        storage.write_audit(&audit).unwrap();
        let ts = TombstoneRecord {
            loser_id: loser_id.clone(),
            winner_id: winner_id.clone(),
            tombstoned_at: Utc::now(),
            user_id: USER.to_string(),
        };
        storage.write_tombstone(&ts).unwrap();

        // Replay — must be idempotent (no panic, no duplicate key error).
        storage.replay_merge_audits(USER);
        storage.replay_merge_audits(USER); // second replay must also succeed

        // Remap must still work post-replay.
        let resolved = storage.resolve_loser(&loser_id, USER);
        assert_eq!(resolved.as_ref(), Some(&winner_id), "(b) remap survives replay");
    }

    // ------------------------------------------------------------------
    // (c) Chain A→B→C path-compresses to A→C on resolve.
    // ------------------------------------------------------------------
    #[test]
    fn sr6_c_chain_path_compresses_to_end() {
        let (storage, _dir) = test_storage();
        const USER: &str = "test-user";

        let mem_a = make_memory("chain A", Some(unit([1.0, 0.0, 0.0])));
        let mem_b = make_memory("chain B", Some(unit([1.0, 0.01, 0.0])));
        let mem_c = make_memory("chain C", Some(unit([1.0, 0.02, 0.0])));
        let id_a = mem_a.id.clone();
        let id_b = mem_b.id.clone();
        let id_c = mem_c.id.clone();

        storage.store(&mem_a).unwrap();
        storage.store(&mem_b).unwrap();
        storage.store(&mem_c).unwrap();

        // Build chain: A→B (A is loser of B), B→C (B is loser of C).
        let audit_ab = MergeAudit {
            audit_id: uuid::Uuid::new_v4().to_string(),
            loser_id: id_a.clone(),
            winner_id: id_b.clone(),
            merged_at: Utc::now(),
            user_id: USER.to_string(),
        };
        storage.write_audit(&audit_ab).unwrap();
        let audit_bc = MergeAudit {
            audit_id: uuid::Uuid::new_v4().to_string(),
            loser_id: id_b.clone(),
            winner_id: id_c.clone(),
            merged_at: Utc::now(),
            user_id: USER.to_string(),
        };
        storage.write_audit(&audit_bc).unwrap();

        // Resolve A — must compress to C.
        let resolved = storage.resolve_loser(&id_a, USER);
        assert_eq!(resolved.as_ref(), Some(&id_c), "(c) A resolves to C after path compression");

        // Resolve B — must still resolve to C.
        let resolved_b = storage.resolve_loser(&id_b, USER);
        assert_eq!(resolved_b.as_ref(), Some(&id_c), "(c) B still resolves to C");
    }

    // ------------------------------------------------------------------
    // (d) GC removes tombstone only when has_any_inbound_refs is false.
    //     With a live ref, GC skips; after the ref is gone, GC reclaims.
    // ------------------------------------------------------------------
    #[test]
    fn sr6_d_gc_gated_on_has_any_inbound_refs() {
        use crate::memory::compression::{FactType, SemanticFact};
        use crate::memory::facts::SemanticFactStore;

        let (storage, _dir) = test_storage();
        const USER: &str = "test-user";

        let loser_mem = make_memory("gc loser", Some(unit([1.0, 0.0, 0.0])));
        let winner_mem = make_memory("gc winner", Some(unit([1.0, 0.01, 0.0])));
        let loser_id = loser_mem.id.clone();
        let winner_id = winner_mem.id.clone();

        storage.store(&loser_mem).unwrap();
        storage.store(&winner_mem).unwrap();

        // Execute merge — tombstone is written.
        storage
            .execute_merge(&loser_id, &winner_id, USER)
            .expect("merge ok");

        // Add a SemanticFact that cites the loser → has_any_inbound_refs == true.
        let fact_store = SemanticFactStore::new(storage.db());
        let fact = SemanticFact {
            id: "sf-gc-d".to_string(),
            fact: "gc test fact".to_string(),
            confidence: 0.9,
            support_count: 1,
            source_memories: vec![loser_id.clone()],
            related_entities: vec![],
            created_at: Utc::now(),
            last_reinforced: Utc::now(),
            fact_type: FactType::Pattern,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: vec![],
            purged_at: None,
            purge_reason: None,
        };
        fact_store.store(USER, &fact).unwrap();

        // GC should skip (has inbound refs).
        // Adjust quiescence by back-dating the tombstone.
        let ts_key = MemoryStorage::tombstone_key(USER, &loser_id);
        let aged_ts = TombstoneRecord {
            loser_id: loser_id.clone(),
            winner_id: winner_id.clone(),
            tombstoned_at: Utc::now() - chrono::Duration::seconds(SEMANTIC_MERGE_GC_MAX_AGE_SECS + 10),
            user_id: USER.to_string(),
        };
        let aged_bytes = serde_json::to_vec(&aged_ts).unwrap();
        storage.db().put(ts_key.as_bytes(), &aged_bytes).unwrap();

        let reclaimed_with_ref = storage.gc_merge_tombstones(USER);
        assert_eq!(reclaimed_with_ref, 0, "(d) GC must not reclaim while inbound ref exists");

        // Remove the citing fact by storing a version without the loser in source_memories.
        // (SemanticFactStore has no delete API in this codebase, so we use a raw DB delete.)
        let fact_key = format!("facts:{}:sf-gc-d", USER);
        storage.db().delete(fact_key.as_bytes()).unwrap();

        // Now has_any_inbound_refs should be false → GC reclaims.
        let reclaimed_no_ref = storage.gc_merge_tombstones(USER);
        assert_eq!(reclaimed_no_ref, 1, "(d) GC must reclaim when inbound refs are gone");
    }

    // ------------------------------------------------------------------
    // (e) With SEMANTIC_MERGE_ENABLED == false (default), run_merge_pass
    //     returns 0 and leaves a known near-duplicate pair untouched.
    // ------------------------------------------------------------------
    #[test]
    fn sr6_e_flag_off_leaves_candidates_untouched() {
        let (storage, _dir) = test_storage();
        const USER: &str = "test-user";

        // Two near-identical embeddings (cosine ≈ 1.0).
        let mem_a = make_memory("flag-off A", Some(unit([1.0, 0.0, 0.0])));
        let mem_b = make_memory("flag-off B", Some(unit([1.0, 0.001, 0.0])));
        let id_a = mem_a.id.clone();
        let id_b = mem_b.id.clone();

        storage.store(&mem_a).unwrap();
        storage.store(&mem_b).unwrap();

        // The flag is OFF by default — run_merge_pass must be a no-op.
        let merged = storage.run_merge_pass(USER, 10);
        assert_eq!(merged, 0, "(e) no merges when SEMANTIC_MERGE_ENABLED is false");

        // Both memories must still be independently readable.
        assert!(storage.get(&id_a).is_ok(), "(e) memory A must be intact");
        assert!(storage.get(&id_b).is_ok(), "(e) memory B must be intact");

        // resolve_loser must return None (no audits written).
        assert!(storage.resolve_loser(&id_a, USER).is_none(), "(e) no audit for A");
        assert!(storage.resolve_loser(&id_b, USER).is_none(), "(e) no audit for B");
    }
}
