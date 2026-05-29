//! R27 — observation graduation pathway.
//!
//! Sleep-time observations are persisted with origin
//! [`MemoryOrigin::BackgroundSleepTimeObservation`]. They are excluded from
//! both NREM and REM evidence packs (L1 confabulation prevention). A
//! foreground-accessed observation that has earned trust transitions to
//! [`MemoryOrigin::BackgroundSleepTimeGraduated`] — at which point it becomes
//! visible to REM-mode evidence assembly and (in a future pass) eligible for
//! fact extraction.
//!
//! Why a separate store, not a metadata mutation: `Memory.experience` is not
//! interior-mutable. Working- and session-tier memories are held as
//! `Arc<Memory>` and mutating their `experience.metadata` would require
//! replacing the entire memory in the tier map — a much wider blast radius
//! than necessary. A small per-memory registry in its own RocksDB column
//! family ("sleep_time_graduations") gives O(1) lookup and zero coupling
//! to the memory-mutation API.
//!
//! The store is consulted by [`super::observer::assemble_evidence_pack`] when
//! it classifies each candidate memory's origin. If an observation memory's
//! id is present in the registry, its effective origin is upgraded to
//! Graduated; if absent, it stays Observation and is filtered out (NREM) or
//! invisible (REM) as before.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::memory::types::MemoryId;

/// Column family for graduation records.
pub const CF_SLEEP_TIME_GRADUATIONS: &str = "sleep_time_graduations";

/// Default threshold for foreground accesses required to graduate an
/// observation. Configurable via [`SleepTimeConfig`] in V2.1.
pub const DEFAULT_GRADUATION_ACCESS_THRESHOLD: u32 = 3;

/// Per-observation graduation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraduationRecord {
    /// When the observation graduated.
    pub graduated_at: DateTime<Utc>,
    /// Access count at the moment of graduation. Useful diagnostic — large
    /// values indicate the graduation pass was delayed.
    pub access_count_at_graduation: u32,
}

pub struct GraduationStore {
    db: Arc<DB>,
}

impl GraduationStore {
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        vec![ColumnFamilyDescriptor::new(
            CF_SLEEP_TIME_GRADUATIONS,
            opts,
        )]
    }

    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    fn cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_SLEEP_TIME_GRADUATIONS)
            .expect("sleep_time_graduations CF must exist in shared DB")
    }

    fn key_for(memory_id: &MemoryId) -> [u8; 16] {
        *memory_id.0.as_bytes()
    }

    /// O(1) lookup: is this memory id present in the graduation registry?
    pub fn is_graduated(&self, memory_id: &MemoryId) -> bool {
        self.db
            .get_cf(self.cf(), Self::key_for(memory_id))
            .ok()
            .flatten()
            .is_some()
    }

    /// Record a graduation. Idempotent — re-recording for an already-
    /// graduated memory updates `graduated_at` to now and the access count
    /// to the most recent observation. Callers should check
    /// [`Self::is_graduated`] first if they want to avoid the overwrite.
    pub fn record_graduation(
        &self,
        memory_id: &MemoryId,
        access_count: u32,
    ) -> Result<()> {
        let record = GraduationRecord {
            graduated_at: Utc::now(),
            access_count_at_graduation: access_count,
        };
        let bytes = bincode::serde::encode_to_vec(&record, bincode::config::standard())
            .context("encode GraduationRecord")?;
        self.db
            .put_cf(self.cf(), Self::key_for(memory_id), bytes)
            .context("write sleep_time_graduations")?;
        Ok(())
    }

    /// Inspect the record for a graduated memory; `None` if not graduated.
    pub fn get(&self, memory_id: &MemoryId) -> Result<Option<GraduationRecord>> {
        let raw = self
            .db
            .get_cf(self.cf(), Self::key_for(memory_id))
            .context("read sleep_time_graduations")?;
        match raw {
            None => Ok(None),
            Some(bytes) => {
                let (rec, _): (GraduationRecord, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .context("decode GraduationRecord")?;
                Ok(Some(rec))
            }
        }
    }

    /// Remove a graduation (rollback path — when V3's forget cascade
    /// degraduates an observation after the source rewrite is forgotten).
    pub fn forget(&self, memory_id: &MemoryId) -> Result<bool> {
        let existed = self.is_graduated(memory_id);
        if existed {
            self.db
                .delete_cf(self.cf(), Self::key_for(memory_id))
                .context("delete sleep_time_graduations")?;
        }
        Ok(existed)
    }
}

// =============================================================================
// Scheduled graduation pass
// =============================================================================

use crate::memory::sleep_time::observation::origin_of;
use crate::memory::sleep_time::types::MemoryOrigin;
use crate::memory::MemorySystem;

/// Result of one graduation pass.
#[derive(Debug, Default, Clone)]
pub struct GraduationPassResult {
    /// Observations scanned (origin == BackgroundSleepTimeObservation).
    pub observations_scanned: usize,
    /// New graduations recorded.
    pub graduated: usize,
    /// Observations already in the registry (no-op).
    pub already_graduated: usize,
}

/// Scan every memory in `mem_sys` and graduate any sleep-time observation
/// whose foreground access count meets `threshold`. Idempotent — running
/// twice with the same corpus produces the same registry state.
///
/// Intended to be called from the heavy maintenance cycle; the existing
/// `MemorySystem` access-count tracker is the authoritative input. The
/// call is cheap: a single linear scan over `get_all_memories()` plus one
/// RocksDB read per observation and one RocksDB write per *new* graduation.
pub fn graduate_eligible_observations(
    mem_sys: &MemorySystem,
    store: &GraduationStore,
    threshold: u32,
) -> Result<GraduationPassResult> {
    let mut out = GraduationPassResult::default();

    let memories = mem_sys
        .get_all_memories()
        .context("get_all_memories for graduation pass")?;

    for mem in memories {
        // Skip non-observations cheaply.
        if origin_of(&mem.experience) != MemoryOrigin::BackgroundSleepTimeObservation {
            continue;
        }
        out.observations_scanned += 1;

        // Skip if already in the registry.
        if store.is_graduated(&mem.id) {
            out.already_graduated += 1;
            continue;
        }

        // Foreground access threshold check.
        let access = mem.access_count();
        if access >= threshold {
            store.record_graduation(&mem.id, access)?;
            out.graduated += 1;
        }
    }

    if out.graduated > 0 {
        tracing::info!(
            scanned = out.observations_scanned,
            graduated = out.graduated,
            already = out.already_graduated,
            threshold,
            "sleep-time graduation pass complete"
        );
    } else {
        tracing::debug!(
            scanned = out.observations_scanned,
            already = out.already_graduated,
            threshold,
            "sleep-time graduation pass — no new graduations"
        );
    }

    Ok(out)
}

/// Effective origin of a memory's experience, with graduation overlay.
///
/// This is the consumer-facing form of `origin_of` — checks the graduation
/// registry first, falling back to the metadata-stored origin. Observer
/// evidence-assembly and any future feedback / fact-extraction integration
/// should use this rather than [`origin_of`] directly when handed both the
/// memory id and a [`GraduationStore`].
pub fn effective_origin(
    memory_id: &MemoryId,
    experience: &crate::memory::types::Experience,
    store: &GraduationStore,
) -> MemoryOrigin {
    let raw = origin_of(experience);
    if raw == MemoryOrigin::BackgroundSleepTimeObservation && store.is_graduated(memory_id) {
        MemoryOrigin::BackgroundSleepTimeGraduated
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::Experience;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_test_db() -> (Arc<DB>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        let cfs = GraduationStore::cf_descriptors();
        let db = DB::open_cf_descriptors(&db_opts, tmp.path(), cfs).unwrap();
        (Arc::new(db), tmp)
    }

    fn fresh_id() -> MemoryId {
        MemoryId(Uuid::new_v4())
    }

    fn observation_experience() -> Experience {
        let mut exp = Experience::default();
        exp.metadata.insert(
            "origin".to_string(),
            "background_sleep_time_observation".to_string(),
        );
        exp
    }

    #[test]
    fn not_graduated_when_registry_empty() {
        let (db, _tmp) = open_test_db();
        let store = GraduationStore::new(db);
        assert!(!store.is_graduated(&fresh_id()));
    }

    #[test]
    fn record_and_lookup_round_trip() {
        let (db, _tmp) = open_test_db();
        let store = GraduationStore::new(db);
        let id = fresh_id();
        store.record_graduation(&id, 5).unwrap();
        assert!(store.is_graduated(&id));
        let rec = store.get(&id).unwrap().unwrap();
        assert_eq!(rec.access_count_at_graduation, 5);
    }

    #[test]
    fn record_is_idempotent_and_updates_access_count() {
        let (db, _tmp) = open_test_db();
        let store = GraduationStore::new(db);
        let id = fresh_id();
        store.record_graduation(&id, 3).unwrap();
        store.record_graduation(&id, 9).unwrap(); // re-graduation overwrites
        let rec = store.get(&id).unwrap().unwrap();
        assert_eq!(rec.access_count_at_graduation, 9);
    }

    #[test]
    fn forget_removes_from_registry() {
        let (db, _tmp) = open_test_db();
        let store = GraduationStore::new(db);
        let id = fresh_id();
        store.record_graduation(&id, 3).unwrap();
        assert!(store.forget(&id).unwrap());
        assert!(!store.is_graduated(&id));
        assert!(!store.forget(&id).unwrap()); // second forget is a no-op
    }

    #[test]
    fn effective_origin_overrides_observation_when_graduated() {
        let (db, _tmp) = open_test_db();
        let store = GraduationStore::new(db);
        let id = fresh_id();
        let exp = observation_experience();
        // Pre-graduation: stays Observation.
        assert_eq!(
            effective_origin(&id, &exp, &store),
            MemoryOrigin::BackgroundSleepTimeObservation
        );
        store.record_graduation(&id, 3).unwrap();
        // Post-graduation: upgrades to Graduated.
        assert_eq!(
            effective_origin(&id, &exp, &store),
            MemoryOrigin::BackgroundSleepTimeGraduated
        );
    }

    #[test]
    fn effective_origin_leaves_foreground_alone() {
        let (db, _tmp) = open_test_db();
        let store = GraduationStore::new(db);
        let id = fresh_id();
        let exp = Experience::default(); // no origin metadata → ForegroundUser
        // Even if we record a graduation against a non-observation memory
        // (shouldn't happen in practice), the effective origin only
        // overrides Observation. Foreground stays Foreground.
        store.record_graduation(&id, 5).unwrap();
        assert_eq!(
            effective_origin(&id, &exp, &store),
            MemoryOrigin::ForegroundUser
        );
    }
}
