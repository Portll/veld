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
//! Like [`super::graduation::GraduationStore`], we use a dedicated column
//! family. Two keys per relationship (forward + reverse) so either
//! end can be queried in O(1).

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
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

/// Direction marker baked into the key so the same struct shape
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
    /// One family, so a CF-bound handle.
    store: Arc<dyn crate::storage::KeyValueStore>,
}

impl SupersessionStore {
    /// Bind a handle directly — the constructor a backend-agnostic caller uses.
    pub fn with_store(store: Arc<dyn crate::storage::KeyValueStore>) -> Self {
        Self { store }
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
    /// Writes both directions atomically via `put_many` so retrieval can
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
        // one family, so put_many is the atomic write — see KeyValueStore
        self.store
            .put_many(&[
                (
                    Self::key_for(Direction::Forward, superseder).to_vec(),
                    f_bytes,
                ),
                (
                    Self::key_for(Direction::Reverse, superseded).to_vec(),
                    r_bytes,
                ),
            ])
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
            .store
            .get_bytes(&Self::key_for(direction, memory_id))
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
        let existed = self.store.get_bytes(&f).ok().flatten().is_some()
            || self.store.get_bytes(&r).ok().flatten().is_some();
        self.store
            .delete_many(&[f.to_vec(), r.to_vec()])
            .context("forget supersession batch")?;
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
        self.store
            .put_many(&[
                (
                    Self::key_for(Direction::Forward, superseder).to_vec(),
                    f_bytes,
                ),
                (
                    Self::key_for(Direction::Reverse, &superseded).to_vec(),
                    r_bytes,
                ),
            ])
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
            .store
            .get_bytes(&Self::key_for(direction, memory_id))
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

        // Pass 1: collect mutations; pass 2 applies them.
        let mut updates: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut deletes: Vec<Vec<u8>> = Vec::new();

        for (k, v) in self.store.scan_prefix(b"", None)? {
            let Ok((mut rec, _)): std::result::Result<
                (SupersessionRecord, _),
                bincode::error::DecodeError,
            > = bincode::serde::decode_from_slice(&v, bincode::config::standard()) else {
                continue;
            };
            rec.confidence = (rec.confidence - decay).max(0.0);
            if rec.confidence <= 0.0 {
                deletes.push(k.clone());
                removed += 1;
            } else {
                let encoded = bincode::serde::encode_to_vec(&rec, bincode::config::standard())
                    .context("encode decayed record")?;
                updates.push((k.clone(), encoded));
                decayed += 1;
            }
        }

        // Pass 2: apply.
        if !updates.is_empty() {
            self.store
                .put_many(&updates)
                .context("write decay updates")?;
        }
        if !deletes.is_empty() {
            self.store
                .delete_many(&deletes)
                .context("write decay deletes")?;
        }

        Ok((decayed, removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageBackend;
    use crate::storage::test_support::{assert_both_backends, column_stores};
    use crate::storage::FamilyKeyValueStore;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn stores() -> Vec<(StorageBackend, SupersessionStore, TempDir)> {
        let out: Vec<_> = column_stores(&[CF_SLEEP_TIME_SUPERSESSIONS])
            .into_iter()
            .map(|(backend, column, tmp)| {
                let kv = FamilyKeyValueStore::new(column, CF_SLEEP_TIME_SUPERSESSIONS).unwrap();
                (backend, SupersessionStore::with_store(Arc::new(kv)), tmp)
            })
            .collect();
        assert_both_backends(&out);
        out
    }

    fn id() -> MemoryId {
        MemoryId(Uuid::new_v4())
    }

    #[test]
    fn record_writes_both_directions() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
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
    }

    #[test]
    fn read_returns_none_when_below_floor() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
            let new = id();
            let old = id();
            store
                .record(&new, &old, SUPERSESSION_EXPIRY_FLOOR / 2.0)
                .unwrap();
            assert!(store.supersession_for(&new).unwrap().is_none());
            assert!(store.superseded_by(&old).unwrap().is_none());
        }
    }

    #[test]
    fn forget_removes_both_directions() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
            let new = id();
            let old = id();
            store.record(&new, &old, 0.8).unwrap();
            assert!(store.forget(&new, &old).unwrap());
            assert!(store.supersession_for(&new).unwrap().is_none());
            assert!(store.superseded_by(&old).unwrap().is_none());
            // Second forget returns false.
            assert!(!store.forget(&new, &old).unwrap());
        }
    }

    #[test]
    fn reinforce_boosts_confidence_clamped_to_one() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
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
    }

    #[test]
    fn reinforce_on_missing_returns_none() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
            assert!(store.reinforce(&id(), 0.1).unwrap().is_none());
        }
    }

    #[test]
    fn decay_reduces_confidence_and_removes_zero() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
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
    }

    #[test]
    fn record_overwrites_existing() {
        for (backend, store, _tmp) in stores() {
            eprintln!("backend: {backend:?}");
            let new = id();
            let old = id();
            store.record(&new, &old, 0.4).unwrap();
            store.record(&new, &old, 0.9).unwrap();
            let rec = store.supersession_for(&new).unwrap().unwrap();
            assert!((rec.confidence - 0.9).abs() < f32::EPSILON);
        }
    }
}
