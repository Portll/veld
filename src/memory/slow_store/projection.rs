//! First real intent-log projection: the SQLite slow store.
//!
//! ## Why this projection exists
//!
//! Up to this point Veld's intent log was a write-side primitive with no
//! reader on the other end — every CRUD landed in RocksDB and the log was
//! either silent or wrote frames that nothing replayed. This module closes
//! that gap by making the SQLite slow store a true projection of the log:
//!
//! - Live writes flow through [`JournaledWriter::record_and_apply`] which
//!   calls [`SqliteProjection::apply`] (the [`TypedProjection`] arm).
//! - On startup the same projection is fed to
//!   [`intent_log::projection::replay`] (the [`Projection`] arm) so any
//!   LSNs the SQLite store missed catch up before traffic resumes.
//!
//! Both arms are wired here, in one place, so the live-write and replay
//! code paths never drift.
//!
//! ## Idempotency
//!
//! Every operation is keyed by `(user_id, memory_id)`:
//!
//! - `Remember` / `Update` — UPSERT, gated on `lsn >= current` so a late
//!   replay can never overwrite a newer live write.
//! - `Forget` — DELETE; already-deleted rows are a no-op.
//! - `Anchor` — UPDATE of the `importance` column with the same LSN gate.
//!
//! Re-applying any LSN twice produces the same state — the trait contract
//! the replay driver depends on.
//!
//! ## Checkpoint persistence
//!
//! The projection holds an in-memory checkpoint `Lsn` and a reference to a
//! shared [`CheckpointStore`] file. The static `Projection` arm flushes
//! the checkpoint to the store via `persist_checkpoint` — the replay
//! driver calls this after every batch. The dyn `TypedProjection` arm
//! used at live-write time also persists the checkpoint after each apply
//! so a crash between live writes resumes from the most recent durable
//! position, not from the start of the log.

use std::error::Error;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::intent_log::{
    CheckpointStore, CheckpointStoreError, IntentPayload, IntentRecord, Lsn, PayloadError,
    Projection, TypedProjection,
};

use super::SlowStore;

/// Errors raised by [`SqliteProjection`]. Distinct from `anyhow::Error`
/// because [`Projection::Error`] requires `std::error::Error` and
/// `anyhow::Error` does not implement it. Each variant wraps the
/// underlying error so callers can match on the precise failure mode.
#[derive(Debug, thiserror::Error)]
pub enum SqliteProjectionError {
    /// A SQL operation against the SQLite slow store failed.
    #[error("sqlite slow store error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Decoding the bincoded payload off the intent log frame failed.
    /// Surfaces only from the static `Projection::apply` arm where the
    /// driver hands us raw record bytes.
    #[error("intent log payload decode error: {0}")]
    PayloadDecode(#[from] PayloadError),
    /// Persisting the checkpoint to disk failed.
    #[error("checkpoint store error: {0}")]
    Checkpoint(#[from] CheckpointStoreError),
    /// Catch-all for higher-level slow-store ops (which today bottom
    /// out at `anyhow::Error` from helpers like `upsert_memory`). We
    /// flatten the chain into a `String` via its `Display` impl because
    /// `anyhow::Error` does not itself implement `std::error::Error`,
    /// which the [`Projection::Error`] trait bound requires.
    #[error("slow store op failed: {0}")]
    Storage(String),
}

/// Name reported to the [`CheckpointStore`] and to the
/// `projection_apply_*` Prometheus metrics. Stable across versions — the
/// checkpoint store is keyed on this string, so renaming it is a manual
/// migration.
pub const PROJECTION_NAME: &str = "slow_store.sqlite";

/// Bridge between the intent log and the SQLite slow store.
///
/// One per tenant. Holds an `Arc<SlowStore>` (the SQLite handle), a
/// shared [`CheckpointStore`] (the per-projection LSN bookkeeping), and
/// an in-memory copy of the last-applied LSN.
pub struct SqliteProjection {
    store: Arc<SlowStore>,
    checkpoint_store: Arc<Mutex<CheckpointStore>>,
    /// LSN of the last record this projection successfully applied. Lives
    /// in memory; `persist_checkpoint` synchronises it to disk.
    checkpoint: Option<Lsn>,
}

impl SqliteProjection {
    /// Construct a projection around an open `SlowStore`. Reads the
    /// last-persisted checkpoint from `checkpoint_store` so subsequent
    /// `replay` / live `apply` calls resume from the right LSN.
    pub fn new(
        store: Arc<SlowStore>,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
    ) -> Self {
        let checkpoint = checkpoint_store.lock().get(PROJECTION_NAME);
        Self {
            store,
            checkpoint_store,
            checkpoint,
        }
    }

    /// Borrow the underlying store. Used by tests and admin tooling that
    /// needs to query the projection directly without going through the
    /// public `SlowStore` API.
    pub fn store(&self) -> &Arc<SlowStore> {
        &self.store
    }

    /// Apply a typed payload at a specific LSN. Pulled into a helper so
    /// both the dyn `TypedProjection::apply` arm and the static
    /// `Projection::apply` arm dispatch through one code path.
    fn apply_typed(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), SqliteProjectionError> {
        match payload {
            IntentPayload::Remember {
                user_id,
                memory_id,
                memory_bincode,
                ..
            }
            | IntentPayload::Update {
                user_id,
                memory_id,
                memory_bincode,
                ..
            } => {
                // Importance isn't carried on the wire for Remember/Update
                // (the bincoded Memory holds it). We persist a neutral
                // 0.5 so the importance column has a defined value;
                // callers that need the precise number decode the blob.
                self.store
                    .upsert_memory(user_id, memory_id, lsn.0, memory_bincode, 0.5)
                    .map_err(|e| SqliteProjectionError::Storage(e.to_string()))?;
            }
            IntentPayload::Forget {
                user_id,
                memory_id,
                ..
            } => {
                self.store
                    .delete_memory(user_id, memory_id)
                    .map_err(|e| SqliteProjectionError::Storage(e.to_string()))?;
            }
            IntentPayload::Anchor {
                user_id,
                memory_id,
                importance,
                ..
            } => {
                self.store
                    .anchor_memory_importance(user_id, memory_id, lsn.0, *importance)
                    .map_err(|e| SqliteProjectionError::Storage(e.to_string()))?;
            }
        }
        self.checkpoint = Some(lsn);
        Ok(())
    }

    fn persist(&mut self) -> Result<(), SqliteProjectionError> {
        if let Some(lsn) = self.checkpoint {
            let mut store = self.checkpoint_store.lock();
            store.set(PROJECTION_NAME, lsn)?;
            store.sync()?;
        }
        Ok(())
    }
}

/// Dyn-friendly arm: the [`JournaledWriter`] hands us typed payloads at
/// live-write time. We immediately persist the checkpoint so a crash
/// after one successful apply doesn't silently roll back to the start of
/// the log.
impl TypedProjection for SqliteProjection {
    fn name(&self) -> &str {
        PROJECTION_NAME
    }

    fn apply(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.apply_typed(lsn, payload)
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        // Live-write path persists every step. Replay batches use the
        // static `Projection` arm below which can amortise via
        // `persist_every`.
        self.persist()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Static arm: the replay driver hands us [`IntentRecord`]s straight off
/// the log. We decode the payload here so the trait stays generic over
/// payload shape — the replay driver doesn't know about `IntentPayload`.
impl Projection for SqliteProjection {
    type Error = SqliteProjectionError;

    fn name(&self) -> &str {
        PROJECTION_NAME
    }

    fn apply(&mut self, record: &IntentRecord) -> Result<(), Self::Error> {
        let (lsn, payload) = crate::intent_log::payload::decode_record(record)?;
        self.apply_typed(lsn, &payload)
    }

    fn checkpoint(&self) -> Option<Lsn> {
        self.checkpoint
    }

    fn persist_checkpoint(&mut self) -> Result<(), Self::Error> {
        self.persist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent_log::{
        payload::CURRENT_PAYLOAD_SCHEMA_VERSION, replay, IntentLog,
    };
    use std::path::PathBuf;

    fn tmp_paths(stem: &str) -> (PathBuf, PathBuf, PathBuf) {
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!("veld-sqlite-proj-{stem}-{pid}-{stamp}"));
        std::fs::create_dir_all(&base).unwrap();
        (
            base.join("slow_store.db"),
            base.join("intent.log"),
            base.join("checkpoints.bin"),
        )
    }

    fn open_projection(
        sqlite_path: &std::path::Path,
        checkpoint_path: &std::path::Path,
    ) -> (Arc<SlowStore>, Arc<Mutex<CheckpointStore>>, SqliteProjection) {
        let store = Arc::new(SlowStore::open(sqlite_path).unwrap());
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(checkpoint_path).unwrap()));
        let proj = SqliteProjection::new(store.clone(), ckpt.clone());
        (store, ckpt, proj)
    }

    fn mk_remember(user: &str, mid: &str, body: &[u8]) -> IntentPayload {
        IntentPayload::Remember {
            user_id: user.into(),
            memory_id: mid.into(),
            memory_bincode: body.to_vec(),
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        }
    }

    #[test]
    fn round_trip_through_typed_apply_writes_three_memories() {
        let (sqlite_path, _, checkpoint_path) = tmp_paths("round_trip");
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);

        // Three live writes through the TypedProjection arm.
        TypedProjection::apply(
            &mut proj,
            Lsn(0),
            &mk_remember("alice", "m-1", b"first"),
        )
        .unwrap();
        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &mk_remember("alice", "m-2", b"second"),
        )
        .unwrap();
        TypedProjection::apply(
            &mut proj,
            Lsn(2),
            &mk_remember("alice", "m-3", b"third"),
        )
        .unwrap();

        // All three present in the SQLite memories table.
        assert_eq!(store.count_memories("alice").unwrap(), 3);
        let r = store.get_memory_blob("alice", "m-2").unwrap().unwrap();
        assert_eq!(r.memory_bincode, b"second");
        assert_eq!(r.lsn, 1);
    }

    #[test]
    fn re_applying_the_same_lsn_twice_is_idempotent() {
        let (sqlite_path, _, checkpoint_path) = tmp_paths("idempotent");
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);

        let payload = mk_remember("u", "m", b"once");
        TypedProjection::apply(&mut proj, Lsn(5), &payload).unwrap();
        // Second apply at the same LSN succeeds and produces the same state.
        TypedProjection::apply(&mut proj, Lsn(5), &payload).unwrap();

        assert_eq!(store.count_memories("u").unwrap(), 1);
        let r = store.get_memory_blob("u", "m").unwrap().unwrap();
        assert_eq!(r.memory_bincode, b"once");
        assert_eq!(r.lsn, 5);
    }

    #[test]
    fn forget_is_idempotent_on_missing_rows() {
        let (sqlite_path, _, checkpoint_path) = tmp_paths("forget_idem");
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);

        let forget = IntentPayload::Forget {
            user_id: "u".into(),
            memory_id: "ghost".into(),
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        // No row exists — apply must still succeed.
        TypedProjection::apply(&mut proj, Lsn(0), &forget).unwrap();
        TypedProjection::apply(&mut proj, Lsn(1), &forget).unwrap();
        assert_eq!(store.count_memories("u").unwrap(), 0);
    }

    #[test]
    fn replay_catches_up_a_fresh_projection_from_pre_existing_log() {
        let (sqlite_path, log_path, checkpoint_path) = tmp_paths("replay");

        // Write 5 records to the intent log *directly*, bypassing the
        // JournaledWriter. This is what a server restart looks like: the
        // log has data, the projection has not yet seen any of it.
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..5 {
                crate::intent_log::payload::append(
                    &mut log,
                    &mk_remember("alice", &format!("m-{i}"), format!("body-{i}").as_bytes()),
                )
                .unwrap();
            }
            log.sync().unwrap();
        }

        // Spin up a brand-new SQLite store + projection. Run replay.
        let log = IntentLog::open(&log_path).unwrap();
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);
        let applied = replay(&log, &mut proj, Some(2)).unwrap();
        assert_eq!(applied, 5);
        assert_eq!(store.count_memories("alice").unwrap(), 5);

        // Inner ordering preserved — m-3 carries body-3, lsn 3.
        let r = store.get_memory_blob("alice", "m-3").unwrap().unwrap();
        assert_eq!(r.memory_bincode, b"body-3");
        assert_eq!(r.lsn, 3);
    }

    #[test]
    fn restart_resumes_from_persisted_checkpoint() {
        let (sqlite_path, log_path, checkpoint_path) = tmp_paths("restart");

        // Write 6 records to the log; projection applies the first 3,
        // persists checkpoint, then "crashes" (gets dropped).
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..6 {
                crate::intent_log::payload::append(
                    &mut log,
                    &mk_remember("u", &format!("m-{i}"), format!("v-{i}").as_bytes()),
                )
                .unwrap();
            }
            log.sync().unwrap();
        }

        {
            let log = IntentLog::open(&log_path).unwrap();
            let (_store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);
            // Manually apply first three records to simulate a partial
            // batch — we use the static Projection trait so we can call
            // replay with a checkpoint-style early stop.
            let records: Vec<_> = log.iter().unwrap().take(3).collect::<Result<_, _>>().unwrap();
            for r in &records {
                Projection::apply(&mut proj, r).unwrap();
            }
            // Persist the checkpoint to disk.
            Projection::persist_checkpoint(&mut proj).unwrap();
            assert_eq!(proj.checkpoint(), Some(Lsn(2)));
        }

        // Reopen projection from scratch — checkpoint must be recovered.
        // The replay driver should resume at Lsn(3), not from the start.
        let log = IntentLog::open(&log_path).unwrap();
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);
        assert_eq!(proj.checkpoint(), Some(Lsn(2)));
        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 3); // only lsn 3, 4, 5 re-applied
        assert_eq!(store.count_memories("u").unwrap(), 6);
        // All six bodies present.
        for i in 0..6 {
            let r = store.get_memory_blob("u", &format!("m-{i}")).unwrap().unwrap();
            assert_eq!(r.memory_bincode, format!("v-{i}").into_bytes());
        }
    }

    #[test]
    fn higher_lsn_wins_when_races_collide() {
        // Simulate a write-skew: replay re-applies an OLD remember at
        // LSN 1 *after* a newer remember at LSN 5 has already landed.
        // The newer payload must survive.
        let (sqlite_path, _, checkpoint_path) = tmp_paths("higher_wins");
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);

        TypedProjection::apply(
            &mut proj,
            Lsn(5),
            &mk_remember("u", "m", b"newer"),
        )
        .unwrap();
        // Stale re-apply at LSN 1 — must not overwrite the newer body.
        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &mk_remember("u", "m", b"older"),
        )
        .unwrap();

        let r = store.get_memory_blob("u", "m").unwrap().unwrap();
        assert_eq!(r.memory_bincode, b"newer");
        assert_eq!(r.lsn, 5);
    }

    #[test]
    fn anchor_updates_importance_without_touching_body() {
        let (sqlite_path, _, checkpoint_path) = tmp_paths("anchor");
        let (store, _ckpt, mut proj) = open_projection(&sqlite_path, &checkpoint_path);

        TypedProjection::apply(
            &mut proj,
            Lsn(0),
            &mk_remember("u", "m", b"body-bytes"),
        )
        .unwrap();
        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &IntentPayload::Anchor {
                user_id: "u".into(),
                memory_id: "m".into(),
                importance: 0.93,
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();

        let r = store.get_memory_blob("u", "m").unwrap().unwrap();
        assert!((r.importance - 0.93).abs() < 1e-6);
        assert_eq!(r.memory_bincode, b"body-bytes");
        assert_eq!(r.lsn, 1);
    }
}
