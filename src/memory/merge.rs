//! S-R6: Semantic merge of near-duplicate memories via tombstone + audit log.
//!
//! One implementation for every engine: each function takes a
//! [`PrimaryMemoryStore`] and keeps its rows in the store's default keyspace,
//! so RocksDB and redb run the same merge and GC, and `veld migrate` carries
//! the rows as `primary/default`.
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
//! The atomic commit point is the append of a [`MergeAudit`] row;
//! [`replay_merge_audits`] re-writes the tombstone of every un-GC'd audit
//! idempotently.
//!
//! ## Evidence pooling (winner update)
//!
//! 1. `access_count` — summed.
//! 2. `entity_refs`  — union (deduplicated by entity name).
//! 3. `embeddings`   — element-wise average of loser into winner.
//!
//! ## Tombstone
//!
//! The loser row stays in storage with a `merge_tombstone:{user_id}:{loser_id}`
//! marker in the default keyspace.  It is **not** deleted.
//!
//! ## Lazy remap (B2) and path compression (B7)
//!
//! [`resolve_loser`] follows the audit log from a loser to its current winner
//! and re-writes the first audit to point at the end of the chain.
//!
//! ## GC (J2)
//!
//! A tombstone is physically reclaimed ONLY when
//! [`PrimaryMemoryStore::has_any_inbound_refs`] is `false` and the quiescence
//! window ([`SEMANTIC_MERGE_GC_QUIESCENCE_SECS`]) or the force window
//! ([`SEMANTIC_MERGE_GC_MAX_AGE_SECS`]) has elapsed.  Reclaim deletes the loser
//! through the store, then its tombstone and audit rows.

use std::collections::HashSet;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::constants::{
    SEMANTIC_MERGE_ENABLED, SEMANTIC_MERGE_GC_MAX_AGE_SECS, SEMANTIC_MERGE_GC_QUIESCENCE_SECS,
    SEMANTIC_MERGE_SIMILARITY_THRESHOLD,
};
use crate::memory::types::{EntityRef, MemoryId};
use crate::similarity::bars::SimilarityBar;
use crate::similarity::cosine_similarity;
use crate::storage::{KeyValueStore, PrimaryMemoryStore};

/// Key prefix for merge-audit rows: `merge_audit:{user_id}:{audit_id}`.
pub(crate) const AUDIT_PREFIX: &str = "merge_audit:";

/// Key prefix for tombstone markers: `merge_tombstone:{user_id}:{loser_id}`.
pub(crate) const TOMBSTONE_PREFIX: &str = "merge_tombstone:";

/// Immutable audit record written as the atomic commit point of every merge.
///
/// Stored in the default keyspace under `merge_audit:{user_id}:{audit_id}` (JSON).
/// Append-only; the only in-place modification permitted is path compression:
/// replacing `winner_id` with the end of the chain (B7).
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

/// Minimal tombstone written at `merge_tombstone:{user_id}:{loser_id}`.
/// Value is a JSON-encoded [`TombstoneRecord`] so GC can check quiescence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TombstoneRecord {
    pub loser_id: MemoryId,
    pub winner_id: MemoryId,
    pub tombstoned_at: DateTime<Utc>,
    pub user_id: String,
}

/// A pair of memories that exceed the similarity threshold.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// The memory with lower `access_count + importance` — will be merged into the winner.
    pub loser_id: MemoryId,
    /// The memory with higher `access_count + importance` — absorbs loser's evidence.
    pub winner_id: MemoryId,
    pub similarity: f32,
}

pub(crate) fn audit_key(user_id: &str, audit_id: &str) -> String {
    format!("{AUDIT_PREFIX}{user_id}:{audit_id}")
}

pub(crate) fn tombstone_key(user_id: &str, loser_id: &MemoryId) -> String {
    format!("{TOMBSTONE_PREFIX}{user_id}:{}", loser_id.0)
}

fn write_json<T: Serialize>(kv: &dyn KeyValueStore, key: &str, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value).context("serialize merge row")?;
    kv.put_bytes(key.as_bytes(), &bytes)
}

/// Every audit row for `user_id`, with its key. A row that will not decode is
/// skipped: an audit is only ever written whole, by [`execute_merge`].
fn load_audits(kv: &dyn KeyValueStore, user_id: &str) -> Result<Vec<(Vec<u8>, MergeAudit)>> {
    let prefix = format!("{AUDIT_PREFIX}{user_id}:");
    Ok(kv
        .scan_prefix(prefix.as_bytes(), None)?
        .into_iter()
        .filter_map(|(k, v)| {
            serde_json::from_slice::<MergeAudit>(&v)
                .ok()
                .map(|a| (k, a))
        })
        .collect())
}

/// Losers of every tombstone in the store, for any user.
fn tombstoned_losers(kv: &dyn KeyValueStore) -> Result<HashSet<MemoryId>> {
    Ok(kv
        .scan_prefix(TOMBSTONE_PREFIX.as_bytes(), None)?
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<TombstoneRecord>(&v).ok())
        .map(|ts| ts.loser_id)
        .collect())
}

/// Scan all memories for near-duplicate pairs, highest similarity first.
///
/// O(n²) over memories that have embeddings; the caller passes a per-tick
/// `limit`.  The loser is the memory with the lower `access_count` +
/// `importance`; a tie makes the earlier memory in store order the loser.
pub fn find_merge_candidates(
    store: &dyn PrimaryMemoryStore,
    limit: usize,
) -> Result<Vec<MergeCandidate>> {
    let tombstoned = tombstoned_losers(store.default_key_value_store().as_ref())?;
    let with_emb: Vec<_> = store
        .get_all()?
        .into_iter()
        .filter(|m| m.experience.embeddings.is_some())
        .filter(|m| !tombstoned.contains(&m.id))
        .collect();

    let mut candidates: Vec<MergeCandidate> = Vec::new();
    let (mut compared, mut crossed) = (0u64, 0u64);
    for (i, a) in with_emb.iter().enumerate() {
        let emb_a = a.experience.embeddings.as_deref().unwrap_or_default();
        for b in &with_emb[i + 1..] {
            let emb_b = b.experience.embeddings.as_deref().unwrap_or_default();
            if emb_a.len() != emb_b.len() {
                continue;
            }
            let sim = crate::similarity::scale::to_reference(cosine_similarity(emb_a, emb_b));
            compared += 1;
            if sim >= SEMANTIC_MERGE_SIMILARITY_THRESHOLD {
                crossed += 1;
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
    SimilarityBar::SemanticMerge.record_many(compared, crossed);

    candidates.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(limit);
    Ok(candidates)
}

/// Execute one merge: pool evidence into winner, append audit, tombstone loser.
///
/// **This is the only write path for merges.**  Production reaches it only
/// through [`run_merge_pass`], which refuses while [`SEMANTIC_MERGE_ENABLED`]
/// is false.  Returns the audit id.
pub fn execute_merge(
    store: &dyn PrimaryMemoryStore,
    loser_id: &MemoryId,
    winner_id: &MemoryId,
    user_id: &str,
) -> Result<String> {
    let mut winner = store
        .get(winner_id)
        .context("execute_merge: winner not found")?;
    let loser = store
        .get(loser_id)
        .context("execute_merge: loser not found")?;

    winner.set_access_count(winner.access_count().saturating_add(loser.access_count()));

    for er in &loser.entity_refs {
        if !winner
            .entity_refs
            .iter()
            .any(|e: &EntityRef| e.name == er.name)
        {
            winner.entity_refs.push(er.clone());
        }
    }

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

    store
        .update(&winner)
        .context("execute_merge: failed to persist winner update")?;

    let kv = store.default_key_value_store();
    let audit_id = uuid::Uuid::new_v4().to_string();
    let audit = MergeAudit {
        audit_id: audit_id.clone(),
        loser_id: loser_id.clone(),
        winner_id: winner_id.clone(),
        merged_at: Utc::now(),
        user_id: user_id.to_string(),
    };
    write_json(kv.as_ref(), &audit_key(user_id, &audit_id), &audit)
        .context("execute_merge: failed to write audit")?;

    let tombstone = TombstoneRecord {
        loser_id: loser_id.clone(),
        winner_id: winner_id.clone(),
        tombstoned_at: Utc::now(),
        user_id: user_id.to_string(),
    };
    write_json(kv.as_ref(), &tombstone_key(user_id, loser_id), &tombstone)
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

/// Resolve a loser id to its current winner id via the audit log, compressing
/// a chain A→B→C to A→C (B7).  `None` when `id` is no audit's loser.
pub fn resolve_loser(
    store: &dyn PrimaryMemoryStore,
    id: &MemoryId,
    user_id: &str,
) -> Result<Option<MemoryId>> {
    let kv = store.default_key_value_store();
    let audits = load_audits(kv.as_ref(), user_id)?;
    let next_winner = |loser: &MemoryId| {
        audits
            .iter()
            .find(|(_, a)| a.loser_id == *loser)
            .map(|(_, a)| a.winner_id.clone())
    };

    let Some((audit_key, first)) = audits.iter().find(|(_, a)| a.loser_id == *id) else {
        return Ok(None);
    };
    let mut winner = first.winner_id.clone();
    let mut visited = vec![id.clone()];
    while !visited.contains(&winner) {
        visited.push(winner.clone());
        match next_winner(&winner) {
            Some(next) => winner = next,
            None => break,
        }
    }

    if winner != first.winner_id {
        tracing::debug!(
            loser = %id.0,
            old_winner = %first.winner_id.0,
            compressed_winner = %winner.0,
            "S-R6: path-compressing merge chain"
        );
        let compressed = MergeAudit {
            winner_id: winner.clone(),
            ..first.clone()
        };
        let bytes = serde_json::to_vec(&compressed).context("serialize MergeAudit")?;
        kv.put_bytes(audit_key, &bytes)?;
    }
    Ok(Some(winner))
}

/// Re-write the tombstone of every audit for `user_id`, so a crash between the
/// audit (the commit point) and the tombstone leaves no loser live.
/// Idempotent.  Returns the number of audits replayed.
pub fn replay_merge_audits(store: &dyn PrimaryMemoryStore, user_id: &str) -> Result<usize> {
    let kv = store.default_key_value_store();
    let audits = load_audits(kv.as_ref(), user_id)?;
    for (_, audit) in &audits {
        let ts = TombstoneRecord {
            loser_id: audit.loser_id.clone(),
            winner_id: audit.winner_id.clone(),
            tombstoned_at: audit.merged_at,
            user_id: audit.user_id.clone(),
        };
        write_json(kv.as_ref(), &tombstone_key(user_id, &ts.loser_id), &ts)
            .with_context(|| format!("replaying merge audit {}", audit.audit_id))?;
    }
    Ok(audits.len())
}

/// Reclaim every tombstone for `user_id` that has no inbound refs and is past
/// the quiescence or force window.  Returns the number reclaimed; a tombstone
/// whose reclaim fails stays for the next cycle.
pub fn gc_merge_tombstones(store: &dyn PrimaryMemoryStore, user_id: &str) -> Result<usize> {
    let now = Utc::now();
    let kv = store.default_key_value_store();
    let prefix = format!("{TOMBSTONE_PREFIX}{user_id}:");

    let mut to_gc: Vec<TombstoneRecord> = Vec::new();
    for (_, value) in kv.scan_prefix(prefix.as_bytes(), None)? {
        let Ok(ts) = serde_json::from_slice::<TombstoneRecord>(&value) else {
            continue;
        };
        let age_secs = (now - ts.tombstoned_at).num_seconds();
        // Past either window is enough; the shorter one decides.
        if age_secs < SEMANTIC_MERGE_GC_QUIESCENCE_SECS.min(SEMANTIC_MERGE_GC_MAX_AGE_SECS) {
            continue;
        }
        match store.has_any_inbound_refs(&ts.loser_id, user_id) {
            Ok(false) => to_gc.push(ts),
            Ok(true) => tracing::debug!(
                loser_id = %ts.loser_id.0,
                "S-R6 GC: loser still has inbound refs, skipping"
            ),
            Err(e) => tracing::warn!(
                loser_id = %ts.loser_id.0,
                error = %e,
                "S-R6 GC: has_any_inbound_refs failed, skipping"
            ),
        }
    }
    if to_gc.is_empty() {
        return Ok(0);
    }

    let audits = load_audits(kv.as_ref(), user_id)?;
    let mut reclaimed = 0;
    for ts in to_gc {
        if let Err(e) = store.delete(&ts.loser_id) {
            tracing::warn!(
                loser_id = %ts.loser_id.0,
                error = %e,
                "S-R6 GC: failed to delete loser memory (will retry next cycle)"
            );
            continue;
        }
        let mut keys = vec![tombstone_key(user_id, &ts.loser_id).into_bytes()];
        keys.extend(
            audits
                .iter()
                .filter(|(_, a)| a.loser_id == ts.loser_id)
                .map(|(k, _)| k.clone()),
        );
        match kv.delete_many(&keys) {
            Ok(()) => {
                tracing::info!(
                    loser_id = %ts.loser_id.0,
                    winner_id = %ts.winner_id.0,
                    "S-R6 GC: tombstone reclaimed"
                );
                reclaimed += 1;
            }
            Err(e) => tracing::warn!(
                loser_id = %ts.loser_id.0,
                error = %e,
                "S-R6 GC: failed to remove tombstone/audit rows"
            ),
        }
    }
    Ok(reclaimed)
}

/// Find and execute up to `limit` merges.  `Ok(0)` while
/// [`SEMANTIC_MERGE_ENABLED`] is false.  A merge that fails is logged and
/// skipped; the count is of merges committed.
pub fn run_merge_pass(
    store: &dyn PrimaryMemoryStore,
    user_id: &str,
    limit: usize,
) -> Result<usize> {
    if !SEMANTIC_MERGE_ENABLED {
        return Ok(0);
    }
    let mut merged = 0;
    for cand in find_merge_candidates(store, limit)? {
        match execute_merge(store, &cand.loser_id, &cand.winner_id, user_id) {
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
            Err(e) => tracing::warn!(
                loser = %cand.loser_id.0,
                winner = %cand.winner_id.0,
                error = %e,
                "S-R6: merge failed"
            ),
        }
    }
    Ok(merged)
}

/// Whether a `SemanticFact`'s `source_memories` or a `TemporalFact`'s
/// `source_memory_id` still cites `target`, read from the facts in the store's
/// default keyspace.  redb's fact half of
/// [`PrimaryMemoryStore::has_any_inbound_refs`], whose memory half both engines
/// answer from an inbound-reference index; neither engine indexes fact citations,
/// and RocksDB reads the same rows with its own prefix scans.
pub fn fact_cites_memory(
    store: &dyn PrimaryMemoryStore,
    target: &MemoryId,
    user_id: &str,
) -> Result<bool> {
    let kv = store.default_key_value_store();
    // Primary keys are `{prefix}{user_id}:{fact_id}`; secondary index rows carry more colons.
    let primary_rows = |prefix: &str| -> Result<Vec<Vec<u8>>> {
        let full = format!("{prefix}{user_id}:");
        Ok(kv
            .scan_prefix(full.as_bytes(), None)?
            .into_iter()
            .filter(|(k, _)| String::from_utf8_lossy(k).matches(':').count() == 2)
            .map(|(_, v)| v)
            .collect())
    };
    if primary_rows("facts:")?.iter().any(|v| {
        super::facts::decode_for_gc(v).is_ok_and(|fact| fact.source_memories.contains(target))
    }) {
        return Ok(true);
    }
    Ok(primary_rows("temporal_facts:")?.iter().any(|v| {
        super::temporal_facts::decode_for_gc(v).is_ok_and(|fact| fact.source_memory_id == *target)
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::memory::types::*;
    use crate::storage::test_support::{assert_both_backends, primary_stores};

    const USER: &str = "test-user";

    fn make_memory(content: &str, embeddings: Option<Vec<f32>>) -> Memory {
        let now = Utc::now();
        Memory::from_legacy(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                experience_type: ExperienceType::Observation,
                content: content.to_string(),
                embeddings,
                ..Default::default()
            },
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

    fn unit(v: [f32; 3]) -> Vec<f32> {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        vec![v[0] / norm, v[1] / norm, v[2] / norm]
    }

    fn stored_pair(store: &dyn PrimaryMemoryStore, tag: &str) -> (MemoryId, MemoryId) {
        let loser = make_memory(&format!("{tag} loser"), Some(unit([1.0, 0.0, 0.0])));
        let winner = make_memory(&format!("{tag} winner"), Some(unit([1.0, 0.01, 0.0])));
        // Unequal scores decide the loser; equal ones fall to store order, which is key order.
        winner.set_access_count(1);
        store.store(&loser).unwrap();
        store.store(&winner).unwrap();
        (loser.id, winner.id)
    }

    fn back_date_tombstone(store: &dyn PrimaryMemoryStore, loser: &MemoryId, winner: &MemoryId) {
        let aged = TombstoneRecord {
            loser_id: loser.clone(),
            winner_id: winner.clone(),
            tombstoned_at: Utc::now()
                - chrono::Duration::seconds(SEMANTIC_MERGE_GC_MAX_AGE_SECS + 10),
            user_id: USER.to_string(),
        };
        write_json(
            store.default_key_value_store().as_ref(),
            &tombstone_key(USER, loser),
            &aged,
        )
        .unwrap();
    }

    #[test]
    fn a_merge_pools_evidence_the_same_way_on_both_engines() {
        let stores = primary_stores();
        assert_both_backends(&stores);
        let mut pooled = Vec::new();
        for (backend, store, _tmp) in &stores {
            let mut loser = make_memory("pool loser", Some(vec![1.0, 0.0, 0.0]));
            let winner = make_memory("pool winner", Some(vec![0.0, 1.0, 0.0]));
            loser.set_access_count(3);
            winner.set_access_count(4);
            loser.entity_refs.push(EntityRef {
                entity_id: uuid::Uuid::new_v4(),
                name: "alpha".into(),
                relation: "mentions".into(),
            });
            store.store(&loser).unwrap();
            store.store(&winner).unwrap();

            execute_merge(store.as_ref(), &loser.id, &winner.id, USER).unwrap();

            let merged = store.get(&winner.id).unwrap();
            assert_eq!(
                merged.access_count(),
                7,
                "{backend:?}: access counts summed"
            );
            assert!(
                merged.entity_refs.iter().any(|e| e.name == "alpha"),
                "{backend:?}: entity refs unioned"
            );
            assert_eq!(
                merged.experience.embeddings.as_deref(),
                Some(&[0.5, 0.5, 0.0][..]),
                "{backend:?}: embeddings averaged"
            );
            let kv = store.default_key_value_store();
            assert!(kv
                .get_bytes(tombstone_key(USER, &loser.id).as_bytes())
                .unwrap()
                .is_some());
            assert_eq!(load_audits(kv.as_ref(), USER).unwrap().len(), 1);
            pooled.push((merged.access_count(), merged.experience.embeddings.clone()));
        }
        assert!(
            pooled.windows(2).all(|w| w[0] == w[1]),
            "engines disagree: {pooled:?}"
        );
    }

    #[test]
    fn a_loser_resolves_to_its_winner_and_replay_is_idempotent() {
        let stores = primary_stores();
        assert_both_backends(&stores);
        for (backend, store, _tmp) in &stores {
            let (loser, winner) = stored_pair(store.as_ref(), "resolve");
            execute_merge(store.as_ref(), &loser, &winner, USER).unwrap();

            assert_eq!(replay_merge_audits(store.as_ref(), USER).unwrap(), 1);
            assert_eq!(replay_merge_audits(store.as_ref(), USER).unwrap(), 1);
            assert_eq!(
                resolve_loser(store.as_ref(), &loser, USER).unwrap(),
                Some(winner.clone()),
                "{backend:?}"
            );
            assert_eq!(
                resolve_loser(store.as_ref(), &winner, USER).unwrap(),
                None,
                "{backend:?}"
            );
            assert!(
                !store.has_any_inbound_refs(&loser, USER).unwrap(),
                "{backend:?}: nothing cites the loser"
            );
        }
    }

    #[test]
    fn a_chain_path_compresses_to_its_end() {
        let stores = primary_stores();
        assert_both_backends(&stores);
        for (backend, store, _tmp) in &stores {
            let ids: Vec<MemoryId> = (0..3)
                .map(|i| {
                    let m = make_memory(
                        &format!("chain {i}"),
                        Some(unit([1.0, 0.01 * i as f32, 0.0])),
                    );
                    store.store(&m).unwrap();
                    m.id
                })
                .collect();
            let kv = store.default_key_value_store();
            for (loser, winner) in [(&ids[0], &ids[1]), (&ids[1], &ids[2])] {
                let audit = MergeAudit {
                    audit_id: uuid::Uuid::new_v4().to_string(),
                    loser_id: loser.clone(),
                    winner_id: winner.clone(),
                    merged_at: Utc::now(),
                    user_id: USER.to_string(),
                };
                write_json(kv.as_ref(), &audit_key(USER, &audit.audit_id), &audit).unwrap();
            }

            assert_eq!(
                resolve_loser(store.as_ref(), &ids[0], USER).unwrap(),
                Some(ids[2].clone()),
                "{backend:?}"
            );
            let compressed = load_audits(kv.as_ref(), USER)
                .unwrap()
                .into_iter()
                .find(|(_, a)| a.loser_id == ids[0])
                .unwrap()
                .1;
            assert_eq!(
                compressed.winner_id, ids[2],
                "{backend:?}: A's audit rewritten to C"
            );
            assert_eq!(
                resolve_loser(store.as_ref(), &ids[1], USER).unwrap(),
                Some(ids[2].clone()),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn gc_waits_for_every_citation_to_go_on_both_engines() {
        use crate::memory::compression::{FactType, SemanticFact};
        use crate::memory::facts::SemanticFactStore;

        let stores = primary_stores();
        assert_both_backends(&stores);
        for (backend, store, _tmp) in &stores {
            let (loser, winner) = stored_pair(store.as_ref(), "gc");
            let holder = {
                let mut m = make_memory("gc holder", None);
                m.experience.related_memories.push(loser.clone());
                store.store(&m).unwrap();
                m.id
            };
            execute_merge(store.as_ref(), &loser, &winner, USER).unwrap();
            back_date_tombstone(store.as_ref(), &loser, &winner);

            let facts = SemanticFactStore::with_store(store.default_key_value_store());
            facts
                .store(
                    USER,
                    &SemanticFact {
                        id: "sf-gc".to_string(),
                        fact: "gc test fact".to_string(),
                        confidence: 0.9,
                        support_count: 1,
                        source_memories: vec![loser.clone()],
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
                    },
                )
                .unwrap();

            assert!(
                store.has_any_inbound_refs(&loser, USER).unwrap(),
                "{backend:?}"
            );
            assert_eq!(
                gc_merge_tombstones(store.as_ref(), USER).unwrap(),
                0,
                "{backend:?}: memory and fact cite it"
            );

            store.delete(&holder).unwrap();
            assert_eq!(
                gc_merge_tombstones(store.as_ref(), USER).unwrap(),
                0,
                "{backend:?}: the fact still cites it"
            );

            let kv = store.default_key_value_store();
            kv.delete_bytes(format!("facts:{USER}:sf-gc").as_bytes())
                .unwrap();
            assert!(
                !store.has_any_inbound_refs(&loser, USER).unwrap(),
                "{backend:?}"
            );
            assert_eq!(
                gc_merge_tombstones(store.as_ref(), USER).unwrap(),
                1,
                "{backend:?}"
            );

            assert!(
                store.get_memory(&loser).unwrap().is_none(),
                "{backend:?}: loser deleted"
            );
            assert!(kv
                .get_bytes(tombstone_key(USER, &loser).as_bytes())
                .unwrap()
                .is_none());
            assert!(
                load_audits(kv.as_ref(), USER).unwrap().is_empty(),
                "{backend:?}: audit removed"
            );
            assert!(
                store.get_memory(&winner).unwrap().is_some(),
                "{backend:?}: winner kept"
            );
        }
    }

    #[test]
    fn a_young_tombstone_is_not_collected() {
        let stores = primary_stores();
        assert_both_backends(&stores);
        for (backend, store, _tmp) in &stores {
            let (loser, winner) = stored_pair(store.as_ref(), "young");
            execute_merge(store.as_ref(), &loser, &winner, USER).unwrap();
            assert_eq!(
                gc_merge_tombstones(store.as_ref(), USER).unwrap(),
                0,
                "{backend:?}"
            );
            assert!(store.get_memory(&loser).unwrap().is_some(), "{backend:?}");
        }
    }

    #[test]
    fn candidates_skip_tombstoned_losers() {
        let stores = primary_stores();
        assert_both_backends(&stores);
        for (backend, store, _tmp) in &stores {
            let (loser, winner) = stored_pair(store.as_ref(), "candidates");
            let found = find_merge_candidates(store.as_ref(), 10).unwrap();
            assert_eq!(found.len(), 1, "{backend:?}");
            assert_eq!(
                (&found[0].loser_id, &found[0].winner_id),
                (&loser, &winner),
                "{backend:?}"
            );

            execute_merge(store.as_ref(), &loser, &winner, USER).unwrap();
            assert!(
                find_merge_candidates(store.as_ref(), 10)
                    .unwrap()
                    .is_empty(),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn with_the_flag_off_the_trait_pass_merges_nothing_on_both_engines() {
        let stores = primary_stores();
        assert_both_backends(&stores);
        for (backend, store, _tmp) in &stores {
            let a = make_memory("flag-off A", Some(unit([1.0, 0.0, 0.0])));
            let b = make_memory("flag-off B", Some(unit([1.0, 0.001, 0.0])));
            store.store(&a).unwrap();
            store.store(&b).unwrap();

            let store: Arc<dyn PrimaryMemoryStore> = Arc::clone(store);
            assert_eq!(store.run_merge_pass(USER, 10).unwrap(), 0, "{backend:?}");
            assert_eq!(store.gc_merge_tombstones(USER).unwrap(), 0, "{backend:?}");
            assert!(store.get_memory(&a.id).unwrap().is_some());
            assert!(store.get_memory(&b.id).unwrap().is_some());
            assert!(resolve_loser(store.as_ref(), &a.id, USER)
                .unwrap()
                .is_none());
        }
    }
}
