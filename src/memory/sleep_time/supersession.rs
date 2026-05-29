//! R42 + R48 + R59 — observation supersession.
//!
//! V1 captured supersession via `ObservationDraft.supersedes: Option<MemoryId>`
//! and stored the claim in the persisted observation's
//! `experience.metadata["supersedes"]`. That works as a one-way pointer
//! (newer → older) but has three limitations the V2 store closes:
//!
//!   1. **Bidirectional** (R59) — retrieval can walk both `newer → older`
//!      and `older → newer` so it can find either end of a chain in one
//!      lookup, without scanning every memory.
//!   2. **Decayable confidence** (R48) — the supersession *claim* has its
//!      own confidence that decays independently of the memory's confidence.
//!      A user's "actually, never mind" reverts the claim before it reverts
//!      the memory.
//!   3. **Forget-cascade ready** (V3) — `forget_supersession` peels the
//!      claim cleanly without touching either memory.
//!
//! Like [`super::graduation::GraduationStore`], we use a dedicated RocksDB
//! column family. Two keys per relationship (forward + reverse) so either
//! end can be queried in O(1).

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::memory::types::MemoryId;

/// Column family for supersession edges.
pub const CF_SLEEP_TIME_SUPERSESSIONS: &str = "sleep_time_supersessions";

/// Initial confidence for a fresh supersession claim. Reduced by decay each
/// maintenance cycle (R48); a foreground accept boosts it back toward 1.0.
pub const DEFAULT_SUPERSESSION_CONFIDENCE: f32 = 0.85;

/// Per-cycle multiplicative decay applied to supersession confidence by
/// [`SupersessionStore::decay_all`]. Half-life ~= ln(2)/decay ≈ 30 cycles at
/// the default.
pub const DEFAULT_SUPERSESSION_DECAY: f32 = 0.0225;

/// Confidence floor: below this the claim is treated as expired and the
/// `effective_supersession` helper returns `None`.
pub const SUPERSESSION_EXPIRY_FLOOR: f32 = 0.10;

/// Direction marker baked into the RocksDB key so the same struct shape
/// stores both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Forward: superseder → superseded.
    Forward,
    /// Reverse: superseded → superseder.
    Reverse,
}

impl Direction {
    fn marker(self) -> u8 {
        match self {
            Self::Forward => b'F',
            Self::Reverse => b'R',
        }
    }
}

/// Per-edge record (same shape both directions; the *direction* lives in the
/// key, the *other end* lives in the body).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupersessionRecord {
    /// The memory id on the other end of the edge.
    pub other: MemoryId,
    /// Current claim confidence in `[0, 1]`. Recorded at write time, decayed
    /// by [`SupersessionStore::decay_all`]. Reads expose the *stored* value
    /// directly — apply [`SUPERSESSION_EXPIRY_FLOOR`] at the caller.
    pub confidence: f32,
    /// When the supersession was first recorded.
    pub recorded_at: DateTime<Utc>,
    /// When the supersession confidence was last refreshed by a positive
    /// foreground signal (V3 hook).
    pub last_reinforced_at: Option<DateTime<Utc>>,
}

pub struct SupersessionStore {
    db: Arc<DB>,
}

impl SupersessionStore {
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        vec![ColumnFamilyDescriptor::new(
            CF_SLEEP_TIME_SUPERSESSIONS,
            opts,
        )]
    }

    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    fn cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_SLEEP_TIME_SUPERSESSIONS)
            .expect("sleep_time_supersessions CF must exist in shared DB")
    }

    /// Key layout: `[direction_byte | 16 uuid bytes]`. 17 bytes total.
    fn key_for(direction: Direction, memory_id: &MemoryId) -> [u8; 17] {
        let mut k = [0u8; 17];
        k[0] = direction.marker();
        k[1..].copy_from_slice(memory_id.0.as_bytes());
        k
    }

    /// Record a supersession claim: `superseder` replaces `superseded`.
    ///
    /// Writes both directions atomically via `WriteBatch` so retrieval can
    /// walk the edge from either side. If a claim already exists in either
    /// direction it is overwritten — caller should check
    /// [`Self::supersession_for`] first if they want to detect re-claims.
    pub fn record(
        &self,
        superseder: &MemoryId,
        superseded: &MemoryId,
        confidence: f32,
    ) -> Result<()> {
        let now = Utc::now();
        let forward = SupersessionRecord {
            other: superseded.clone(),
            confidence: confidence.clamp(0.0, 1.0),
            recorded_at: now,
            last_reinforced_at: None,
        };
        let reverse = SupersessionRecord {
            other: superseder.clone(),
            confidence: confidence.clamp(0.0, 1.0),
            recorded_at: now,
            last_reinforced_at: None,
        };
        let f_bytes = bincode::serde::encode_to_vec(&forward, bincode::config::standard())
            .context("encode forward SupersessionRecord")?;
        let r_bytes = bincode::serde::encode_to_vec(&reverse, bincode::config::standard())
            .context("encode reverse SupersessionRecord")?;
        let mut batch = rocksdb::WriteBatch::default();
        batch.put_cf(
            self.cf(),
            Self::key_for(Direction::Forward, superseder),
            f_bytes,
        );
        batch.put_cf(
            self.cf(),
            Self::key_for(Direction::Reverse, superseded),
            r_bytes,
        );
        self.db
            .write(batch)
            .context("write SupersessionStore batch")?;
        Ok(())
    }

    /// Read the *forward* edge: what does `superseder` claim to supersede?
    /// Returns `Ok(None)` if no claim or if the claim has fallen below
    /// [`SUPERSESSION_EXPIRY_FLOOR`].
    pub fn supersession_for(&self, superseder: &MemoryId) -> Result<Option<SupersessionRecord>> {
        self.read(Direction::Forward, superseder)
    }

    /// Read the *reverse* edge: who claims to supersede `superseded`?
    pub fn superseded_by(&self, superseded: &MemoryId) -> Result<Option<SupersessionRecord>> {
        self.read(Direction::Reverse, superseded)
    }

    fn read(
        &self,
        direction: Direction,
        memory_id: &MemoryId,
    ) -> Result<Option<SupersessionRecord>> {
        let raw = self
            .db
            .get_cf(self.cf(), Self::key_for(direction, memory_id))
            .context("read sleep_time_supersessions")?;
        let Some(bytes) = raw else { return Ok(None) };
        let (rec, _): (SupersessionRecord, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .context("decode SupersessionRecord")?;
        if rec.confidence < SUPERSESSION_EXPIRY_FLOOR {
            Ok(None)
        } else {
            Ok(Some(rec))
        }
    }

    /// Delete both edges for a supersession claim. Used by the V3 forget
    /// cascade when the source rewrite is forgotten.
    pub fn forget(&self, superseder: &MemoryId, superseded: &MemoryId) -> Result<bool> {
        let f = Self::key_for(Direction::Forward, superseder);
        let r = Self::key_for(Direction::Reverse, superseded);
        let existed = self
            .db
            .get_cf(self.cf(), f)
            .ok()
            .flatten()
            .is_some()
            || self.db.get_cf(self.cf(), r).ok().flatten().is_some();
        let mut batch = rocksdb::WriteBatch::default();
        batch.delete_cf(self.cf(), f);
        batch.delete_cf(self.cf(), r);
        self.db.write(batch).context("forget supersession batch")?;
        Ok(existed)
    }

    /// Reinforce a supersession claim — caller in the V3 feedback path
    /// invokes this on confirming foreground evidence. Boosts confidence
    /// toward 1.0 by a fixed amount and stamps `last_reinforced_at`.
    pub fn reinforce(
        &self,
        superseder: &MemoryId,
        boost: f32,
    ) -> Result<Option<SupersessionRecord>> {
        let Some(mut rec) = self.read_raw(Direction::Forward, superseder)? else {
            return Ok(None);
        };
        rec.confidence = (rec.confidence + boost.max(0.0)).min(1.0);
        rec.last_reinforced_at = Some(Utc::now());
        // Mirror on reverse.
        let superseded = rec.other.clone();
        let mut rev = self
            .read_raw(Direction::Reverse, &superseded)?
            .unwrap_or_else(|| SupersessionRecord {
                other: superseder.clone(),
                confidence: rec.confidence,
                recorded_at: rec.recorded_at,
                last_reinforced_at: rec.last_reinforced_at,
            });
        rev.confidence = rec.confidence;
        rev.last_reinforced_at = rec.last_reinforced_at;

        let f_bytes = bincode::serde::encode_to_vec(&rec, bincode::config::standard())
            .context("encode reinforced forward record")?;
        let r_bytes = bincode::serde::encode_to_vec(&rev, bincode::config::standard())
            .context("encode reinforced reverse record")?;
        let mut batch = rocksdb::WriteBatch::default();
        batch.put_cf(
            self.cf(),
            Self::key_for(Direction::Forward, superseder),
            f_bytes,
        );
        batch.put_cf(
            self.cf(),
            Self::key_for(Direction::Reverse, &superseded),
            r_bytes,
        );
        self.db
            .write(batch)
            .context("write reinforce batch")?;
        Ok(Some(rec))
    }

    /// Read *without* applying the [`SUPERSESSION_EXPIRY_FLOOR`] gate.
    /// Used internally where we need the actual stored value.
    fn read_raw(
        &self,
        direction: Direction,
        memory_id: &MemoryId,
    ) -> Result<Option<SupersessionRecord>> {
        let raw = self
            .db
            .get_cf(self.cf(), Self::key_for(direction, memory_id))
            .context("read raw sleep_time_supersessions")?;
        let Some(bytes) = raw else { return Ok(None) };
        let (rec, _): (SupersessionRecord, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .context("decode raw SupersessionRecord")?;
        Ok(Some(rec))
    }

    /// Apply R48 decay to every supersession edge. Forward and reverse
    /// records are updated together so confidences stay in sync. Records
    /// that drop to zero or below are deleted to bound the CF size.
    /// Returns `(decayed, removed)`.
    pub fn decay_all(&self, decay: f32) -> Result<(usize, usize)> {
        let decay = decay.clamp(0.0, 1.0);
        if decay == 0.0 {
            return Ok((0, 0));
        }
        let mut decayed = 0usize;
        let mut removed = 0usize;

        // Pass 1: collect mutations. We can't mutate while iterating because
        // RocksDB's iterator handle borrows the CF.
        let mut updates: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut deletes: Vec<Vec<u8>> = Vec::new();

        let iter = self.db.iterator_cf(self.cf(), rocksdb::IteratorMode::Start);
        for entry in iter {
            let (k, v) = entry.context("iterate sleep_time_supersessions")?;
            let Ok((mut rec, _)): std::result::Result<
                (SupersessionRecord, _),
                bincode::error::DecodeError,
            > = bincode::serde::decode_from_slice(&v, bincode::config::standard()) else {
                continue;
            };
            rec.confidence = (rec.confidence - decay).max(0.0);
            if rec.confidence <= 0.0 {
                deletes.push(k.into_vec());
                removed += 1;
            } else {
                let encoded = bincode::serde::encode_to_vec(&rec, bincode::config::standard())
                    .context("encode decayed record")?;
                updates.push((k.into_vec(), encoded));
                decayed += 1;
            }
        }

        // Pass 2: apply.
        if !updates.is_empty() || !deletes.is_empty() {
            let mut batch = rocksdb::WriteBatch::default();
            for (k, v) in &updates {
                batch.put_cf(self.cf(), k, v);
            }
            for k in &deletes {
                batch.delete_cf(self.cf(), k);
            }
            self.db.write(batch).context("write decay batch")?;
        }

        Ok((decayed, removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_test_db() -> (Arc<DB>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        let cfs = SupersessionStore::cf_descriptors();
        let db = DB::open_cf_descriptors(&db_opts, tmp.path(), cfs).unwrap();
        (Arc::new(db), tmp)
    }

    fn id() -> MemoryId {
        MemoryId(Uuid::new_v4())
    }

    #[test]
    fn record_writes_both_directions() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        let new = id();
        let old = id();
        store
            .record(&new, &old, DEFAULT_SUPERSESSION_CONFIDENCE)
            .unwrap();

        // Forward: newer claims to supersede older.
        let f = store.supersession_for(&new).unwrap().unwrap();
        assert_eq!(f.other, old);

        // Reverse: older is superseded by newer.
        let r = store.superseded_by(&old).unwrap().unwrap();
        assert_eq!(r.other, new);
    }

    #[test]
    fn read_returns_none_when_below_floor() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        let new = id();
        let old = id();
        store
            .record(&new, &old, SUPERSESSION_EXPIRY_FLOOR / 2.0)
            .unwrap();
        assert!(store.supersession_for(&new).unwrap().is_none());
        assert!(store.superseded_by(&old).unwrap().is_none());
    }

    #[test]
    fn forget_removes_both_directions() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        let new = id();
        let old = id();
        store.record(&new, &old, 0.8).unwrap();
        assert!(store.forget(&new, &old).unwrap());
        assert!(store.supersession_for(&new).unwrap().is_none());
        assert!(store.superseded_by(&old).unwrap().is_none());
        // Second forget returns false.
        assert!(!store.forget(&new, &old).unwrap());
    }

    #[test]
    fn reinforce_boosts_confidence_clamped_to_one() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        let new = id();
        let old = id();
        store.record(&new, &old, 0.5).unwrap();
        let rec = store.reinforce(&new, 0.8).unwrap().unwrap();
        assert!((rec.confidence - 1.0).abs() < f32::EPSILON);
        assert!(rec.last_reinforced_at.is_some());
        // Reverse also reinforced.
        let rev = store.superseded_by(&old).unwrap().unwrap();
        assert!((rev.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reinforce_on_missing_returns_none() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        assert!(store.reinforce(&id(), 0.1).unwrap().is_none());
    }

    #[test]
    fn decay_reduces_confidence_and_removes_zero() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        let a_new = id();
        let a_old = id();
        let b_new = id();
        let b_old = id();
        store.record(&a_new, &a_old, 0.6).unwrap();
        store.record(&b_new, &b_old, 0.04).unwrap(); // below floor on next decay

        let (decayed, removed) = store.decay_all(0.05).unwrap();
        // a decays from 0.60 → 0.55 (still above 0), b drops to 0.0 and is removed.
        // Each edge is stored twice (forward + reverse), so a = 2 decayed, b = 2 removed.
        assert_eq!(decayed, 2);
        assert_eq!(removed, 2);

        // a still present at the reduced confidence.
        let a = store.supersession_for(&a_new).unwrap().unwrap();
        assert!((a.confidence - 0.55).abs() < 1e-5);

        // b gone.
        assert!(store.supersession_for(&b_new).unwrap().is_none());
        assert!(store.superseded_by(&b_old).unwrap().is_none());
    }

    #[test]
    fn record_overwrites_existing() {
        let (db, _tmp) = open_test_db();
        let store = SupersessionStore::new(db);
        let new = id();
        let old = id();
        store.record(&new, &old, 0.4).unwrap();
        store.record(&new, &old, 0.9).unwrap();
        let rec = store.supersession_for(&new).unwrap().unwrap();
        assert!((rec.confidence - 0.9).abs() < f32::EPSILON);
    }
}
