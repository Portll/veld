//! Second real intent-log projection: the BM25 inverted index (tantivy).
//!
//! ## Why this projection exists
//!
//! The SQLite slow store landed as the first real projection of the W5
//! intent log — every CRUD operation now journals through
//! [`crate::intent_log::JournaledWriter::record_and_apply`] which calls
//! `SqliteProjection::apply`. This module does the same job for BM25:
//! the tantivy-backed inverted index used by hybrid search now derives
//! its state from the same intent log instead of being a side-effect of
//! the in-process write path.
//!
//! Why bother? Because the goal of W5 is "RocksDB is source of truth;
//! everything else is rebuildable from the log". As soon as BM25 ceases
//! to be a true projection it drifts — a write may land in RocksDB,
//! sync to SQLite, but never make it into the BM25 index, and search
//! quietly returns stale results until the next backfill. Closing that
//! gap is the whole point of the projection layer.
//!
//! - Live writes flow through
//!   [`crate::intent_log::JournaledWriter::record_and_apply`] which calls
//!   the [`TypedProjection`] arm on [`Bm25Projection`].
//! - On startup the same projection is fed to
//!   [`crate::intent_log::projection::replay`] via the [`Projection`]
//!   arm so any LSNs the BM25 index missed catch up before traffic
//!   resumes.
//!
//! Both arms are wired here, in one place, so the live-write and replay
//! code paths never drift apart.
//!
//! ## Indexed fields
//!
//! Every `Remember` / `Update` decodes the bincoded `Memory` snapshot and
//! pushes the following text into tantivy:
//!
//! - `content` — `Memory::experience::content` (the primary indexable
//!   surface form),
//! - `tags`    — `Memory::experience::tags`, space-joined,
//! - `entities` — `Memory::experience::entities`, space-joined.
//!
//! These mirror what the existing
//! [`crate::memory::HybridSearchEngine::index_memory`] backfill writes, so
//! a projection-driven rebuild and a backfill-driven rebuild produce the
//! same searchable corpus.
//!
//! ## Commit cadence
//!
//! Tantivy `IndexWriter` writes are buffered: an `add_document` /
//! `delete_term` call doesn't make the change visible to the index reader
//! until `commit()` runs. Each commit fsyncs the segment files and the meta,
//! so the projection commits in batches, and on the live-write path not on the
//! writer's thread: `apply` buffers the change under the tenant's journal
//! mutex and leaves the commit to a background committer, one per projection,
//! started on the first live apply.
//!
//! The committer commits once something is buffered and the index's last
//! commit is at least [`COMMIT_INTERVAL`] old, or at once when
//! [`COMMIT_EVERY`] records are waiting, and then persists the projection
//! checkpoint. A search commits what is still buffered before it reads (see
//! [`crate::memory::HybridSearchEngine::commit_for_read`]), which is how a
//! recall sees a memory written just before it.
//!
//! The checkpoint never names a record the index has not committed: the
//! committer reads the last applied LSN, then commits, then persists that LSN,
//! and every record up to it was buffered before the commit began. Persists
//! never move the checkpoint backwards, whichever of the committer, a drain or
//! replay makes them.
//!
//! The committer stops, after a final commit, when the projection is dropped
//! or [`TypedProjection::stop_background`] is called (the journaled writer's
//! shutdown and a restore); applies after that commit inline on the same
//! cadence. The periodic flush and the shutdown flush reach it through
//! [`crate::intent_log::journal::DeferredDrainHandle::drain`], without the
//! writer's mutex.
//!
//! Replay commits on the replaying thread. The batch size is owned in one
//! place — [`COMMIT_EVERY`] — and the replay driver is told to persist the
//! projection checkpoint with the same period via
//! `replay(..., Some(COMMIT_EVERY))`. This means the on-disk BM25 segments and
//! the persisted intent-log checkpoint advance together: a crash that loses an
//! unflushed batch also loses the checkpoint for that batch, so on restart the
//! replay driver re-applies exactly the records the BM25 index never durably
//! saw.
//!
//! ## Idempotency
//!
//! Every operation is keyed by `memory_id` (the UUID inside `MemoryId`):
//!
//! - `Remember` / `Update` — `BM25Index::upsert` does a `delete_term` on
//!   the `id` field followed by `add_document`. Replaying the same LSN
//!   twice produces exactly one document in the index. An `Update` whose
//!   content, tags and entities are what the index last wrote for that
//!   memory writes nothing (`BM25Index::upsert_if_changed`): most updates
//!   change importance, anchoring or embeddings, not the indexed text.
//! - `Forget` — `BM25Index::delete` is a `delete_term` on `id`. Deleting a
//!   missing document is a no-op in tantivy.
//! - `Anchor` — no-op. Anchoring affects importance/decay, not text
//!   content; the BM25 corpus is unchanged.
//!
//! Re-applying any LSN twice produces the same state — the trait contract
//! the replay driver depends on.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::intent_log::{
    BackgroundFlush, CheckpointStore, CheckpointStoreError, IntentPayload, IntentRecord, Lsn,
    PayloadError, Projection, TypedProjection,
};

use super::hybrid_search::BM25Index;
use super::types::{Memory, MemoryId};

/// Errors raised by [`Bm25Projection`]. Distinct from `anyhow::Error`
/// because [`Projection::Error`] requires `std::error::Error` and
/// `anyhow::Error` does not implement it.
#[derive(Debug, thiserror::Error)]
pub enum Bm25ProjectionError {
    /// Tantivy / BM25 index operation failed.
    #[error("bm25 index error: {0}")]
    Bm25(String),
    /// Decoding the bincoded payload off the intent log frame failed.
    /// Surfaces only from the static `Projection::apply` arm where the
    /// driver hands us raw record bytes.
    #[error("intent log payload decode error: {0}")]
    PayloadDecode(#[from] PayloadError),
    /// Persisting the checkpoint to disk failed.
    #[error("checkpoint store error: {0}")]
    Checkpoint(#[from] CheckpointStoreError),
    /// The payload's `memory_id` could not be parsed as a UUID. BM25's
    /// `MemoryId` is a UUID wrapper, so a malformed id means the payload
    /// is from a foreign schema and cannot be indexed.
    #[error("memory_id is not a valid UUID: {0}")]
    InvalidMemoryId(String),
    /// Decoding the `memory_bincode` blob into a `Memory` value failed.
    /// The intent log frame is still durable; the bad blob is logged and
    /// skipped at the live-write call site (the journaled writer collects
    /// the error in `WriteOutcome.apply_errors` without failing the write).
    #[error("memory bincode decode error: {0}")]
    MemoryDecode(String),
}

/// Name reported to the [`CheckpointStore`] and to the
/// `projection_apply_*` Prometheus metrics. Stable across versions — the
/// checkpoint store is keyed on this string, so renaming it is a manual
/// migration.
pub const PROJECTION_NAME: &str = "bm25";

/// Applies waiting for a commit at which the live path commits without
/// waiting for [`COMMIT_INTERVAL`], AND the number of applies between
/// `persist_checkpoint` calls in the replay path.
///
/// Tantivy commits are expensive (fsync of segment files + meta), so we
/// amortise. The replay driver is given the same number as
/// `persist_every` so the BM25 segments and the intent-log checkpoint
/// advance in lock-step: any batch we lose to a crash is the same batch
/// the next replay will re-apply.
pub const COMMIT_EVERY: u64 = 100;

/// Shortest gap between two live-path commits. A write arriving after a quiet
/// spell is committed at once; one arriving sooner waits until the last commit
/// is this old, so a burst shares one commit. Bounds how long a journaled write
/// stays invisible to a reader that does not itself commit.
pub const COMMIT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

/// How long dropping a projection waits for its committer's final commit.
const COMMITTER_STOP_TIMEOUT: Duration = Duration::from_secs(10);

/// Bridge between the intent log and the BM25 inverted index.
///
/// One per tenant. Owns an `Arc<BM25Index>` (the tantivy handle shared
/// between the projection and any read-side hybrid-search reader), the
/// committer that commits it in the background (which holds the shared
/// [`CheckpointStore`]), and an in-memory copy of the last-applied LSN.
pub struct Bm25Projection {
    index: Arc<BM25Index>,
    /// LSN of the last record this projection successfully applied. Lives
    /// in memory; the committer, or replay, persists it.
    checkpoint: Option<Lsn>,
    /// Number of successful applies since the last commit on the inline
    /// path, which runs when there is no background committer.
    applies_since_commit: u64,
    /// Live-path time trigger; [`COMMIT_INTERVAL`] unless overridden.
    commit_interval: Duration,
    committer: Arc<Committer>,
    worker: CommitterWorker,
}

/// The background committer's thread, as the projection sees it.
enum CommitterWorker {
    /// No live apply yet; the thread starts on the first.
    NotStarted,
    Running(std::thread::JoinHandle<()>),
    /// Stopped, or it could not start: live applies commit inline.
    Stopped,
}

/// State the projection shares with its committer thread and with drain handles.
struct Committer {
    index: Arc<BM25Index>,
    checkpoint_store: Arc<Mutex<CheckpointStore>>,
    state: Mutex<CommitterState>,
    /// Signalled when a record is applied, the committer is told to stop, and when it exits.
    wake: Condvar,
    /// The checkpoint last persisted, held across the store write so that persists from
    /// the committer, a drain and replay never move it backwards.
    persisted: Mutex<Option<Lsn>>,
}

struct CommitterState {
    /// LSN of the last record applied to the index.
    applied: Option<Lsn>,
    /// Live applies since the committer last committed.
    waiting: u64,
    stop: bool,
    exited: bool,
}

impl Committer {
    fn new(
        index: Arc<BM25Index>,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
        persisted: Option<Lsn>,
    ) -> Self {
        Self {
            index,
            checkpoint_store,
            state: Mutex::new(CommitterState {
                applied: None,
                waiting: 0,
                stop: false,
                exited: false,
            }),
            wake: Condvar::new(),
            persisted: Mutex::new(persisted),
        }
    }

    /// Record `lsn` as applied to the index. A live apply also counts toward the
    /// committer's batch and wakes it.
    fn record(&self, lsn: Lsn, live: bool) {
        let mut st = self.state.lock();
        st.applied = Some(lsn);
        if live {
            st.waiting += 1;
            if st.waiting == 1 || st.waiting >= COMMIT_EVERY {
                self.wake.notify_all();
            }
        }
    }

    fn run(self: Arc<Self>, interval: Duration) {
        loop {
            let target = {
                let mut st = self.state.lock();
                while st.waiting == 0 && !st.stop {
                    self.wake.wait(&mut st);
                }
                if st.stop {
                    break;
                }
                let due = Instant::now() + interval.saturating_sub(self.index.since_last_commit());
                while !st.stop && st.waiting < COMMIT_EVERY && Instant::now() < due {
                    self.wake.wait_until(&mut st, due);
                }
                if st.stop {
                    break;
                }
                st.waiting = 0;
                st.applied
            };
            if let Err(error) = self.commit_through(target) {
                tracing::warn!(
                    projection = PROJECTION_NAME,
                    error = %error,
                    "background BM25 commit failed; the next write or search commits the batch, \
                     and replay re-applies it after a crash",
                );
            }
        }
        if let Err(error) = self.flush_now() {
            tracing::warn!(
                projection = PROJECTION_NAME,
                error = %error,
                "final BM25 commit failed as the committer stopped (replay recovers the tail)",
            );
        }
        self.state.lock().exited = true;
        self.wake.notify_all();
    }

    /// Tell the thread to stop and wait, until `deadline`, for its final commit.
    fn stop(&self, deadline: Instant) -> bool {
        let mut st = self.state.lock();
        st.stop = true;
        self.wake.notify_all();
        while !st.exited {
            if self.wake.wait_until(&mut st, deadline).timed_out() {
                return st.exited;
            }
        }
        true
    }

    /// Commit and persist everything applied so far.
    fn flush_now(&self) -> Result<(), Bm25ProjectionError> {
        let target = self.state.lock().applied;
        self.commit_through(target)
    }

    /// Commit what the index has buffered, then persist `target`. Every record at or
    /// below `target` was applied before this call, so the commit covers it; one a
    /// search already committed leaves nothing to commit.
    fn commit_through(&self, target: Option<Lsn>) -> Result<(), Bm25ProjectionError> {
        self.index
            .commit_pending()
            .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
        self.persist(target)
    }

    /// Persist `target` as the checkpoint unless one at or past it already is.
    fn persist(&self, target: Option<Lsn>) -> Result<(), Bm25ProjectionError> {
        let Some(lsn) = target else {
            return Ok(());
        };
        let mut persisted = self.persisted.lock();
        if persisted.is_some_and(|done| done >= lsn) {
            return Ok(());
        }
        let mut store = self.checkpoint_store.lock();
        store.set(PROJECTION_NAME, lsn)?;
        store.sync()?;
        *persisted = Some(lsn);
        Ok(())
    }
}

impl BackgroundFlush for Committer {
    fn flush(&self) -> bool {
        match self.flush_now() {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    projection = PROJECTION_NAME,
                    error = %error,
                    "BM25 flush failed (replay recovers the tail)",
                );
                false
            }
        }
    }
}

impl Bm25Projection {
    /// Construct a projection around an open BM25 index. Reads the
    /// last-persisted checkpoint from `checkpoint_store` so subsequent
    /// `replay` / live `apply` calls resume from the right LSN.
    pub fn new(index: Arc<BM25Index>, checkpoint_store: Arc<Mutex<CheckpointStore>>) -> Self {
        let checkpoint = checkpoint_store.lock().get(PROJECTION_NAME);
        Self::with_checkpoint(index, checkpoint_store, checkpoint)
    }

    /// A projection over an index that starts empty every time (held in memory for
    /// an encrypted tenant): it ignores the persisted checkpoint and replays the
    /// whole log, because the checkpoint describes an index that no longer exists.
    pub fn new_from_start(
        index: Arc<BM25Index>,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
    ) -> Self {
        Self::with_checkpoint(index, checkpoint_store, None)
    }

    fn with_checkpoint(
        index: Arc<BM25Index>,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
        checkpoint: Option<Lsn>,
    ) -> Self {
        let committer = Arc::new(Committer::new(
            Arc::clone(&index),
            checkpoint_store,
            checkpoint,
        ));
        Self {
            index,
            checkpoint,
            applies_since_commit: 0,
            commit_interval: COMMIT_INTERVAL,
            committer,
            worker: CommitterWorker::NotStarted,
        }
    }

    /// Replace the live-path time trigger (default [`COMMIT_INTERVAL`]). Takes effect
    /// for a committer not yet started, which is any before the first live apply.
    pub fn with_commit_interval(mut self, interval: Duration) -> Self {
        self.commit_interval = interval;
        self
    }

    /// Borrow the underlying index. Used by tests and admin tooling that
    /// needs to query the projection directly without going through the
    /// hybrid-search engine.
    pub fn index(&self) -> &Arc<BM25Index> {
        &self.index
    }

    /// Apply a typed payload at a specific LSN. Pulled into a helper so
    /// both the dyn `TypedProjection::apply` arm and the static
    /// `Projection::apply` arm dispatch through one code path.
    fn apply_typed(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Bm25ProjectionError> {
        match payload {
            IntentPayload::Remember {
                memory_id,
                memory_bincode,
                ..
            }
            | IntentPayload::Update {
                memory_id,
                memory_bincode,
                ..
            } => {
                let mid = parse_memory_id(memory_id)?;
                let memory = decode_memory(memory_bincode)?;
                // A soft-forgotten record stays in the store (the write gate keeps an
                // absorbed memory that way) but leaves the keyword index.
                if memory.is_forgotten() {
                    self.index
                        .delete(&mid)
                        .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
                } else {
                    let content = memory.experience.content.as_str();
                    let tags = memory.experience.tags.as_slice();
                    let entities = memory.experience.entities.as_slice();
                    let written = if matches!(payload, IntentPayload::Update { .. }) {
                        self.index
                            .upsert_if_changed(&mid, content, tags, entities)
                            .map(|_| ())
                    } else {
                        self.index.upsert(&mid, content, tags, entities)
                    };
                    written.map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
                }
            }
            IntentPayload::Forget { memory_id, .. } => {
                let mid = parse_memory_id(memory_id)?;
                self.index
                    .delete(&mid)
                    .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
            }
            IntentPayload::Anchor { .. } | IntentPayload::SetAnchored { .. } => {
                // Anchoring changes importance / decay resistance only.
                // The BM25 corpus has no concept of importance, so the
                // text index is unaffected. We still advance the
                // checkpoint below so the projection's view of the log
                // stays in sync — without this, a long run of Anchors
                // would make the BM25 projection look perpetually
                // behind on the replay-lag metric.
            }
        }
        self.checkpoint = Some(lsn);
        Ok(())
    }

    /// Commit the underlying tantivy writer if it holds buffered documents,
    /// and reload the reader so a hybrid search engine sharing this
    /// `Arc<BM25Index>` sees them. The index, not `applies_since_commit`,
    /// decides: a search may already have committed this batch through
    /// `commit_pending`, and committing again would be an empty fsync.
    fn commit_index(&mut self) -> Result<(), Bm25ProjectionError> {
        self.index
            .commit_pending()
            .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
        self.applies_since_commit = 0;
        Ok(())
    }

    /// Whether the inline live path should close the current batch now.
    fn commit_due(&self) -> bool {
        self.applies_since_commit >= COMMIT_EVERY
            || self.index.since_last_commit() >= self.commit_interval
    }

    /// Persist the in-memory checkpoint to the shared checkpoint store.
    /// Called by the replay driver on its `persist_every` cadence and at
    /// every inline commit boundary.
    fn persist(&mut self) -> Result<(), Bm25ProjectionError> {
        self.committer.persist(self.checkpoint)
    }

    /// Whether the background committer takes this live apply's commit, starting it on
    /// first use. `false` when there is none and the caller commits inline.
    fn commit_in_background(&mut self) -> bool {
        if matches!(self.worker, CommitterWorker::NotStarted) {
            let committer = Arc::clone(&self.committer);
            let interval = self.commit_interval;
            self.worker = match std::thread::Builder::new()
                .name("veld-bm25-commit".to_string())
                .spawn(move || committer.run(interval))
            {
                Ok(handle) => CommitterWorker::Running(handle),
                Err(error) => {
                    tracing::warn!(
                        projection = PROJECTION_NAME,
                        error = %error,
                        "could not start the BM25 committer; committing inline",
                    );
                    CommitterWorker::Stopped
                }
            };
        }
        matches!(self.worker, CommitterWorker::Running(_))
    }
}

impl Drop for Bm25Projection {
    fn drop(&mut self) {
        if matches!(self.worker, CommitterWorker::Running(_)) {
            self.stop_background(Instant::now() + COMMITTER_STOP_TIMEOUT);
        }
    }
}

/// Decode a memory_id string from an `IntentPayload` into a [`MemoryId`].
/// Surfaces `InvalidMemoryId` as a structured error so the journaled
/// writer can log and skip cleanly rather than panic.
fn parse_memory_id(s: &str) -> Result<MemoryId, Bm25ProjectionError> {
    uuid::Uuid::parse_str(s)
        .map(MemoryId)
        .map_err(|e| Bm25ProjectionError::InvalidMemoryId(format!("{s}: {e}")))
}

/// Decode a bincoded `Memory` snapshot off an `IntentPayload`. Goes through the store's
/// decoder, legacy arms included: a snapshot journaled before a positional
/// layout change is the old wire, exactly as a stored record would be.
fn decode_memory(bytes: &[u8]) -> Result<Memory, Bm25ProjectionError> {
    crate::memory::storage::decode_memory_snapshot(bytes)
        .map_err(|e| Bm25ProjectionError::MemoryDecode(e.to_string()))
}

fn boxed(e: Bm25ProjectionError) -> Box<dyn Error + Send + Sync> {
    Box::new(e)
}

/// Dyn-friendly arm: the [`crate::intent_log::JournaledWriter`] hands us
/// typed payloads at live-write time, under the tenant's journal mutex. The
/// change is buffered here and committed by the background committer, which
/// persists the checkpoint only after the commit that covers it, so the
/// checkpoint can trail the segments (replay re-applies idempotently) but
/// never lead them.
impl TypedProjection for Bm25Projection {
    fn name(&self) -> &str {
        PROJECTION_NAME
    }

    fn apply(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.apply_typed(lsn, payload).map_err(boxed)?;
        if self.commit_in_background() {
            self.committer.record(lsn, true);
            return Ok(());
        }
        self.committer.record(lsn, false);
        self.applies_since_commit += 1;
        if self.commit_due() {
            self.commit_index().map_err(boxed)?;
            self.persist().map_err(boxed)?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn background_flush(&self) -> Option<Arc<dyn BackgroundFlush>> {
        Some(Arc::clone(&self.committer) as Arc<dyn BackgroundFlush>)
    }

    fn stop_background(&mut self, deadline: Instant) -> bool {
        match std::mem::replace(&mut self.worker, CommitterWorker::Stopped) {
            CommitterWorker::Running(handle) => {
                if !self.committer.stop(deadline) {
                    tracing::warn!(
                        projection = PROJECTION_NAME,
                        "BM25 committer still committing at the deadline; detached, it persists \
                         the checkpoint itself when the commit ends",
                    );
                    return false;
                }
                if handle.join().is_err() {
                    tracing::warn!(projection = PROJECTION_NAME, "BM25 committer panicked");
                }
                true
            }
            CommitterWorker::NotStarted | CommitterWorker::Stopped => {
                BackgroundFlush::flush(self.committer.as_ref())
            }
        }
    }
}

/// Static arm: the replay driver hands us [`IntentRecord`]s straight off
/// the log. We decode the payload here so the trait stays generic over
/// payload shape — the replay driver doesn't know about `IntentPayload`.
impl Projection for Bm25Projection {
    type Error = Bm25ProjectionError;

    fn name(&self) -> &str {
        PROJECTION_NAME
    }

    fn apply(&mut self, record: &IntentRecord) -> Result<(), Self::Error> {
        let (lsn, payload) = crate::intent_log::payload::decode_record(record)?;
        self.apply_typed(lsn, &payload)?;
        self.committer.record(lsn, false);
        self.applies_since_commit += 1;
        Ok(())
    }

    fn checkpoint(&self) -> Option<Lsn> {
        self.checkpoint
    }

    /// Persist the checkpoint AND commit the tantivy writer.
    ///
    /// The replay driver guarantees this is called at least once at the
    /// end of replay and (if `persist_every` was passed) on the batch
    /// cadence. We commit the BM25 index first so the segments are
    /// durable before the checkpoint advances — a crash between commit
    /// and checkpoint just means the next replay re-applies the same
    /// records, which the idempotency contract handles correctly.
    fn persist_checkpoint(&mut self) -> Result<(), Self::Error> {
        self.commit_index()?;
        self.persist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent_log::{payload::CURRENT_PAYLOAD_SCHEMA_VERSION, replay, IntentLog};
    use std::time::{Duration, Instant};

    /// A payload journaled before 4a54053 carries the old session layout; the projection must
    /// read it the way the store does, or replay stalls at that lsn forever (lsn 12753,
    /// 2026-09-03 to 2026-09-16).
    #[test]
    fn a_pre_writer_snapshot_decodes_through_the_store_arms() {
        use crate::memory::storage::wire_fixtures::{
            encode, memory_with_session, strip_writer_structs,
        };
        let original = memory_with_session("journaled before the writer structs existed");
        let old_wire = strip_writer_structs(&encode(&original));
        assert!(bincode::serde::decode_from_slice::<Memory, _>(
            &old_wire,
            bincode::config::standard()
        )
        .is_err());
        let back = decode_memory(&old_wire).expect("legacy arm through the store decoder");
        assert_eq!(back.id, original.id);
        assert_eq!(back.experience.content, original.experience.content);
    }
    use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId};
    use std::path::{Path, PathBuf};
    use uuid::Uuid;

    fn tmp_paths(stem: &str) -> (PathBuf, PathBuf, PathBuf) {
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!("veld-bm25-proj-{stem}-{pid}-{stamp}"));
        std::fs::create_dir_all(&base).unwrap();
        (
            base.join("bm25_index"),
            base.join("intent.log"),
            base.join("checkpoints.bin"),
        )
    }

    fn open_projection(
        bm25_path: &Path,
        checkpoint_path: &Path,
    ) -> (Arc<BM25Index>, Arc<Mutex<CheckpointStore>>, Bm25Projection) {
        std::fs::create_dir_all(bm25_path).unwrap();
        let index = Arc::new(BM25Index::new(bm25_path).unwrap());
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(checkpoint_path).unwrap()));
        let proj = Bm25Projection::new(index.clone(), ckpt.clone());
        (index, ckpt, proj)
    }

    fn mk_memory(id: Uuid, content: &str, tags: &[&str], entities: &[&str]) -> Memory {
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            ..Experience::default()
        };
        Memory::new(MemoryId(id), experience, 0.5, None, None, None, None)
    }

    fn remember_payload(memory: &Memory) -> IntentPayload {
        let bytes = bincode::serde::encode_to_vec(memory, bincode::config::standard()).unwrap();
        IntentPayload::Remember {
            user_id: "alice".into(),
            memory_id: memory.id.0.to_string(),
            memory_bincode: bytes,
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        }
    }

    /// Round-trip: three memories remembered through `TypedProjection::apply`
    /// are all searchable by their distinctive content.
    #[test]
    fn round_trip_three_memories_each_searchable_by_content() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("round_trip");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        let id_rust = Uuid::new_v4();
        let id_python = Uuid::new_v4();
        let id_jwt = Uuid::new_v4();
        let m_rust = mk_memory(
            id_rust,
            "The user prefers Rust programming language for systems development",
            &["rust", "programming"],
            &["Rust"],
        );
        let m_python = mk_memory(
            id_python,
            "Python is great for machine learning and data science projects",
            &["python", "ml"],
            &["Python"],
        );
        let m_jwt = mk_memory(
            id_jwt,
            "The authentication system uses JWT tokens for security",
            &["auth", "security"],
            &["JWT"],
        );

        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&m_rust)).unwrap();
        TypedProjection::apply(&mut proj, Lsn(1), &remember_payload(&m_python)).unwrap();
        TypedProjection::apply(&mut proj, Lsn(2), &remember_payload(&m_jwt)).unwrap();

        // The dyn arm batches commits via COMMIT_EVERY — force a flush so
        // the reader can see the new documents.
        Projection::persist_checkpoint(&mut proj).unwrap();

        let rust_hits = index.search("Rust programming", 10).unwrap();
        assert!(!rust_hits.is_empty(), "expected at least one Rust hit");
        assert_eq!(rust_hits[0].0, MemoryId(id_rust));

        let python_hits = index.search("Python machine learning", 10).unwrap();
        assert!(!python_hits.is_empty(), "expected at least one Python hit");
        assert_eq!(python_hits[0].0, MemoryId(id_python));

        let jwt_hits = index.search("JWT authentication", 10).unwrap();
        assert!(!jwt_hits.is_empty(), "expected at least one JWT hit");
        assert_eq!(jwt_hits[0].0, MemoryId(id_jwt));
    }

    /// Idempotency: re-applying the same Remember after the first batch
    /// has been committed produces exactly one indexed document. The
    /// projection's `upsert` does a `delete_term` + `add_document` on
    /// every apply, which tantivy collapses to one row provided the
    /// prior apply is already in a sealed segment (the `delete_term`
    /// only sees committed docs — same condition as a real crash-replay,
    /// where the commit boundary is forced by the process restart).
    #[test]
    fn re_applying_the_same_remember_is_idempotent() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("idempotent");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        let id = Uuid::new_v4();
        let memory = mk_memory(id, "exactly one body", &[], &[]);
        let payload = remember_payload(&memory);

        TypedProjection::apply(&mut proj, Lsn(7), &payload).unwrap();
        // Force a commit boundary — mirrors what `replay()` does between
        // the failed-apply checkpoint and the retry on restart.
        Projection::persist_checkpoint(&mut proj).unwrap();
        TypedProjection::apply(&mut proj, Lsn(7), &payload).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();

        assert_eq!(index.len(), 1);
        let hits = index.search("exactly one body", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, MemoryId(id));
    }

    /// Forget removes the document. A second Forget on the same id is a
    /// no-op — already-deleted documents are not an error in tantivy.
    #[test]
    fn forget_removes_document_and_is_idempotent_on_missing() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("forget");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        let id = Uuid::new_v4();
        let memory = mk_memory(id, "indexable surface form", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&memory)).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();
        assert_eq!(index.len(), 1);

        let forget = IntentPayload::Forget {
            user_id: "alice".into(),
            memory_id: id.to_string(),
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        TypedProjection::apply(&mut proj, Lsn(1), &forget).unwrap();
        // Second Forget on the now-missing id must still succeed.
        TypedProjection::apply(&mut proj, Lsn(2), &forget).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();

        assert_eq!(index.len(), 0);
        let hits = index.search("indexable surface form", 10).unwrap();
        assert!(hits.is_empty(), "no documents should remain after forget");
    }

    /// Replay from scratch: write 5 records straight to a fresh intent
    /// log, then point a brand-new BM25 projection at it. After
    /// `replay()` returns, all 5 documents are searchable.
    #[test]
    fn replay_catches_up_a_fresh_projection_from_pre_existing_log() {
        let (bm25_path, log_path, checkpoint_path) = tmp_paths("replay");

        // Write 5 records to the intent log *directly*, bypassing the
        // JournaledWriter. This is what a server restart looks like.
        let mut ids = Vec::with_capacity(5);
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..5 {
                let id = Uuid::new_v4();
                ids.push(id);
                let memory =
                    mk_memory(id, &format!("replay body {i}"), &[&format!("tag-{i}")], &[]);
                crate::intent_log::payload::append(&mut log, &remember_payload(&memory)).unwrap();
            }
            log.sync().unwrap();
        }

        let log = IntentLog::open(&log_path).unwrap();
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);
        // `persist_every` is `COMMIT_EVERY` in production wiring; the
        // test passes `Some(2)` so the batch commit is exercised even
        // for a 5-record run.
        let applied = replay(&log, &mut proj, Some(2)).unwrap();
        assert_eq!(applied, 5);
        assert_eq!(index.len(), 5);

        for (i, id) in ids.iter().enumerate() {
            let hits = index.search(&format!("replay body {i}"), 10).unwrap();
            assert!(!hits.is_empty(), "expected hit for memory {i}");
            assert_eq!(hits[0].0, MemoryId(*id), "memory {i} ranked first");
        }
    }

    /// An encrypted tenant's index lives in memory and starts empty on every open,
    /// so its projection must replay the whole log even though the checkpoint says
    /// every frame was applied. Resuming from the checkpoint would leave it empty.
    #[test]
    fn an_in_memory_index_replays_the_whole_log_whatever_the_checkpoint_says() {
        let (bm25_path, log_path, checkpoint_path) = tmp_paths("in_memory_replay");
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..3 {
                let memory = mk_memory(Uuid::new_v4(), &format!("sealed body {i}"), &[], &[]);
                crate::intent_log::payload::append(&mut log, &remember_payload(&memory)).unwrap();
            }
            log.sync().unwrap();
        }
        let log = IntentLog::open(&log_path).unwrap();
        {
            let (_index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);
            assert_eq!(replay(&log, &mut proj, Some(1)).unwrap(), 3);
        }
        assert!(
            bm25_path.exists(),
            "precondition: an on-disk index and a checkpoint at the head"
        );

        let index = Arc::new(BM25Index::in_memory(&bm25_path).unwrap());
        assert!(
            !bm25_path.exists(),
            "the plaintext on-disk index is removed"
        );
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&checkpoint_path).unwrap()));
        let mut resumed = Bm25Projection::new(index.clone(), ckpt.clone());
        assert_eq!(
            replay(&log, &mut resumed, Some(1)).unwrap(),
            0,
            "the checkpoint is at the head"
        );

        let mut from_start = Bm25Projection::new_from_start(index.clone(), ckpt);
        assert_eq!(replay(&log, &mut from_start, Some(1)).unwrap(), 3);
        assert_eq!(index.len(), 3);
        assert!(
            !bm25_path.exists(),
            "replaying into it wrote nothing to disk"
        );
    }

    /// Restart resumes from the persisted checkpoint: a projection that
    /// applied the first half of a log, persisted, then "crashed" picks
    /// up at the right LSN on reopen and only re-applies the tail.
    #[test]
    fn restart_resumes_from_persisted_checkpoint() {
        let (bm25_path, log_path, checkpoint_path) = tmp_paths("restart");

        let mut ids = Vec::with_capacity(6);
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..6 {
                let id = Uuid::new_v4();
                ids.push(id);
                let memory = mk_memory(id, &format!("body-{i}"), &[], &[]);
                crate::intent_log::payload::append(&mut log, &remember_payload(&memory)).unwrap();
            }
            log.sync().unwrap();
        }

        {
            let log = IntentLog::open(&log_path).unwrap();
            let (_index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);
            let records: Vec<_> = log
                .iter()
                .unwrap()
                .take(3)
                .collect::<Result<_, _>>()
                .unwrap();
            for r in &records {
                Projection::apply(&mut proj, r).unwrap();
            }
            Projection::persist_checkpoint(&mut proj).unwrap();
            assert_eq!(proj.checkpoint(), Some(Lsn(2)));
        }

        // Reopen — checkpoint must be recovered, replay applies only the
        // tail (lsn 3..=5).
        let log = IntentLog::open(&log_path).unwrap();
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);
        assert_eq!(proj.checkpoint(), Some(Lsn(2)));
        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 3);
        assert_eq!(index.len(), 6);
        for (i, id) in ids.iter().enumerate() {
            let hits = index.search(&format!("body-{i}"), 10).unwrap();
            assert!(!hits.is_empty(), "expected hit for memory {i}");
            assert_eq!(hits[0].0, MemoryId(*id));
        }
    }

    /// Anchor payloads are no-ops for BM25 (text content unchanged) but
    /// still advance the checkpoint so the replay-lag metric tracks
    /// reality.
    #[test]
    fn anchor_is_a_noop_but_advances_checkpoint() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("anchor");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        let id = Uuid::new_v4();
        let memory = mk_memory(id, "body before anchor", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&memory)).unwrap();
        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &IntentPayload::Anchor {
                user_id: "alice".into(),
                memory_id: id.to_string(),
                importance: 0.93,
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();

        assert_eq!(index.len(), 1);
        let hits = index.search("body before anchor", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, MemoryId(id));
        assert_eq!(proj.checkpoint(), Some(Lsn(1)));
    }

    /// Poll `done` for up to ten seconds; the committer runs on its own thread.
    fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A live apply only buffers. The committer commits once the last commit is an
    /// interval old, and the checkpoint it persists names only committed records.
    #[test]
    fn the_committer_commits_after_the_interval_and_persists_after_the_commit() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("commit_interval");
        let (index, ckpt, proj) = open_projection(&bm25_path, &checkpoint_path);
        let mut proj = proj.with_commit_interval(Duration::from_millis(500));
        index.commit().unwrap();

        let first = mk_memory(Uuid::new_v4(), "buffered tamarind entry", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&first)).unwrap();
        assert!(
            index.has_uncommitted(),
            "the apply leaves the commit to the committer"
        );
        assert!(index.search("tamarind", 10).unwrap().is_empty());
        assert_eq!(ckpt.lock().get(PROJECTION_NAME), None);

        let second = mk_memory(Uuid::new_v4(), "late tamarind entry", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(1), &remember_payload(&second)).unwrap();
        wait_for("the checkpoint at lsn 1", || {
            ckpt.lock().get(PROJECTION_NAME) == Some(Lsn(1))
        });
        assert!(
            !index.has_uncommitted(),
            "the checkpoint is persisted after the commit that covers it"
        );
        assert_eq!(index.search("tamarind", 10).unwrap().len(), 2);
    }

    /// A drain flushes what the committer is still waiting on, without the writer, and
    /// the writer's shutdown stops the committer.
    #[test]
    fn a_drain_flushes_the_committer_through_the_journaled_writer() {
        let (bm25_path, log_path, checkpoint_path) = tmp_paths("drain");
        let (index, ckpt, proj) = open_projection(&bm25_path, &checkpoint_path);
        let proj = proj.with_commit_interval(Duration::from_secs(3600));
        index.commit().unwrap();
        let mut writer =
            crate::intent_log::JournaledWriter::new(IntentLog::open(&log_path).unwrap());
        writer.add_projection(Box::new(proj));

        let memory = mk_memory(Uuid::new_v4(), "drained quince entry", &[], &[]);
        let lsn = writer
            .record_and_apply(&remember_payload(&memory))
            .unwrap()
            .lsn;
        assert!(index.has_uncommitted());

        assert!(writer
            .deferred_drain_handle()
            .drain(Duration::from_secs(10)));
        assert!(!index.has_uncommitted());
        assert_eq!(index.search("quince", 10).unwrap().len(), 1);
        assert_eq!(ckpt.lock().get(PROJECTION_NAME), Some(lsn));

        assert!(writer.shutdown_deferred(Duration::from_secs(10)));
    }

    /// Dropping the projection stops its committer after a final commit.
    #[test]
    fn dropping_the_projection_commits_what_it_buffered() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("drop");
        let (index, ckpt, proj) = open_projection(&bm25_path, &checkpoint_path);
        let mut proj = proj.with_commit_interval(Duration::from_secs(3600));
        index.commit().unwrap();

        let memory = mk_memory(Uuid::new_v4(), "persimmon entry", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(4), &remember_payload(&memory)).unwrap();
        assert!(index.has_uncommitted());
        drop(proj);

        assert!(!index.has_uncommitted());
        assert_eq!(index.search("persimmon", 10).unwrap().len(), 1);
        assert_eq!(ckpt.lock().get(PROJECTION_NAME), Some(Lsn(4)));
    }

    /// After `stop_background` the projection commits on its own thread again, on the
    /// same cadence: with a zero interval, at every apply.
    #[test]
    fn a_stopped_committer_hands_commits_back_to_the_apply() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("stopped");
        let (index, ckpt, proj) = open_projection(&bm25_path, &checkpoint_path);
        let mut proj = proj.with_commit_interval(Duration::ZERO);

        let first = mk_memory(Uuid::new_v4(), "first medlar entry", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&first)).unwrap();
        assert!(proj.stop_background(Instant::now() + Duration::from_secs(10)));
        assert_eq!(ckpt.lock().get(PROJECTION_NAME), Some(Lsn(0)));

        let second = mk_memory(Uuid::new_v4(), "second medlar entry", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(1), &remember_payload(&second)).unwrap();
        assert!(
            !index.has_uncommitted(),
            "committed inline, before apply returned"
        );
        assert_eq!(ckpt.lock().get(PROJECTION_NAME), Some(Lsn(1)));
        assert_eq!(index.search("medlar", 10).unwrap().len(), 2);
    }

    fn update_payload(memory: &Memory) -> IntentPayload {
        let bytes = bincode::serde::encode_to_vec(memory, bincode::config::standard()).unwrap();
        IntentPayload::Update {
            user_id: "alice".into(),
            memory_id: memory.id.0.to_string(),
            memory_bincode: bytes,
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        }
    }

    /// An `Update` that leaves content, tags and entities as indexed writes nothing to
    /// the index yet still advances the checkpoint; one that changes them, or forgets
    /// the memory, is applied.
    #[test]
    fn an_update_with_unchanged_text_writes_nothing() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("unchanged_update");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        let mut memory = mk_memory(
            Uuid::new_v4(),
            "loquat orchard notes",
            &["fruit"],
            &["Loquat"],
        );
        Projection::apply(&mut proj, &record_of(Lsn(0), &remember_payload(&memory))).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();
        assert!(!index.has_uncommitted());

        memory.experience.embeddings = Some(vec![0.25; 8]);
        Projection::apply(&mut proj, &record_of(Lsn(1), &update_payload(&memory))).unwrap();
        assert!(
            !index.has_uncommitted(),
            "an embedding-only update leaves the index alone"
        );
        assert_eq!(proj.checkpoint(), Some(Lsn(1)));

        memory.experience.tags.push("orchard".to_string());
        Projection::apply(&mut proj, &record_of(Lsn(2), &update_payload(&memory))).unwrap();
        assert!(index.has_uncommitted(), "a tag change is re-indexed");
        Projection::persist_checkpoint(&mut proj).unwrap();

        memory
            .experience
            .metadata
            .insert("forgotten".to_string(), "true".to_string());
        Projection::apply(&mut proj, &record_of(Lsn(3), &update_payload(&memory))).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();
        assert_eq!(index.len(), 0, "a forgotten snapshot leaves the index");
        assert!(index.search("loquat", 10).unwrap().is_empty());
    }

    /// The record the replay driver would hand the projection for `payload` at `lsn`.
    fn record_of(lsn: Lsn, payload: &IntentPayload) -> IntentRecord {
        IntentRecord {
            lsn,
            payload: crate::intent_log::payload::encode(payload).unwrap(),
        }
    }

    /// A search-side `commit_pending` commits what the projection buffered;
    /// the projection's next batch close finds nothing to commit and still
    /// persists its checkpoint.
    #[test]
    fn a_reader_flush_and_the_projection_batch_share_one_commit() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("reader_flush");
        let (index, ckpt, proj) = open_projection(&bm25_path, &checkpoint_path);
        let mut proj = proj.with_commit_interval(std::time::Duration::from_secs(3600));
        index.commit().unwrap();

        let memory = mk_memory(Uuid::new_v4(), "flushed by the reader", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&memory)).unwrap();
        assert!(
            index.commit_pending().unwrap(),
            "buffered work is committed"
        );
        assert!(!index.commit_pending().unwrap(), "and only once");
        assert_eq!(index.search("flushed reader", 10).unwrap().len(), 1);

        Projection::persist_checkpoint(&mut proj).unwrap();
        assert_eq!(ckpt.lock().get(PROJECTION_NAME), Some(Lsn(0)));
    }

    /// Malformed memory_id in the payload returns a structured error
    /// instead of panicking — the journaled writer surfaces it via
    /// `WriteOutcome.apply_errors` without failing the write.
    #[test]
    fn malformed_memory_id_produces_structured_error() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("bad_uuid");
        let (_index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        let memory = mk_memory(Uuid::new_v4(), "doesn't matter", &[], &[]);
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
        let payload = IntentPayload::Remember {
            user_id: "alice".into(),
            memory_id: "not-a-uuid".into(),
            memory_bincode: bytes,
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        };

        let err = TypedProjection::apply(&mut proj, Lsn(0), &payload).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not-a-uuid"),
            "expected error message to mention the bad id, got: {msg}",
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Unified-index tests: projection writes ↔ HybridSearchEngine reads
    // ─────────────────────────────────────────────────────────────────────
    //
    // These guard the W5 unification invariant: a journaled write that
    // lands in the `Bm25Projection` MUST become visible to a
    // `HybridSearchEngine` sharing the same `Arc<BM25Index>` as soon as
    // the projection's commit boundary returns — no separate reader
    // reload on the engine side, no parallel on-disk directory.
    //
    // We instantiate `HybridSearchEngine` with reranking disabled so we
    // do not need a real embedder; the BM25 read path is exercised via
    // `bm25_index().search(...)` which is what `recall::semantic_retrieve`
    // ultimately calls.

    struct NoopEmbedder;

    impl crate::embeddings::Embedder for NoopEmbedder {
        fn id(&self) -> &str {
            "stub-noop"
        }

        fn encode(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            // Reranking is disabled in these tests, so this is never invoked.
            // Returning a fixed zero vector keeps the implementation total
            // without pulling in an ONNX model just to satisfy the type.
            Ok(vec![0.0; 8])
        }
        fn dimension(&self) -> usize {
            8
        }
    }

    fn open_engine_sharing(
        index: Arc<crate::memory::hybrid_search::BM25Index>,
    ) -> crate::memory::hybrid_search::HybridSearchEngine {
        let config = crate::memory::hybrid_search::HybridSearchConfig {
            use_reranking: false,
            ..crate::memory::hybrid_search::HybridSearchConfig::default()
        };
        crate::memory::hybrid_search::HybridSearchEngine::with_bm25_index(
            index,
            std::sync::Arc::new(NoopEmbedder),
            config,
        )
    }

    /// Round-trip across the unified boundary: journal a `Remember`
    /// through the projection, then read it back via the
    /// retrieval-side `HybridSearchEngine` that holds the same
    /// `Arc<BM25Index>`. Before the unification this would fail because
    /// the engine opened a parallel index at `bm25_index/` while the
    /// projection wrote to `bm25_projection_index/`.
    #[test]
    fn projection_write_is_immediately_visible_to_hybrid_search_engine() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("unified_round_trip");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);

        // Build the retrieval engine around the SAME Arc the projection holds.
        let engine = open_engine_sharing(index.clone());

        let id = Uuid::new_v4();
        let memory = mk_memory(
            id,
            "SIGHUP is the canonical reload signal on POSIX systems",
            &["linux", "signals"],
            &["SIGHUP"],
        );

        // Write via the journaled-writer projection arm…
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&memory)).unwrap();
        // …and force the commit + reader reload (what `COMMIT_EVERY`
        // would have done automatically once we crossed the batch size).
        Projection::persist_checkpoint(&mut proj).unwrap();

        // The retrieval engine — pointed at the same Arc — sees it.
        let hits = engine.bm25_index().search("SIGHUP", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "engine sharing the projection's Arc<BM25Index> must see committed writes"
        );
        assert_eq!(hits[0].0, MemoryId(id));
    }

    /// A search keeps read-your-writes by default: it commits the write the committer is
    /// still holding. Under a staleness bound it leaves a younger write to the committer.
    #[test]
    fn the_read_commit_honours_the_staleness_bound() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("read_staleness");
        let (index, _ckpt, proj) = open_projection(&bm25_path, &checkpoint_path);
        let mut proj = proj.with_commit_interval(Duration::from_secs(3600));
        index.commit().unwrap();
        let engine = open_engine_sharing(index.clone());

        let memory = mk_memory(Uuid::new_v4(), "jujube harvest", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&memory)).unwrap();

        assert!(!engine
            .commit_for_read(Some(Duration::from_secs(3600)))
            .unwrap());
        assert!(engine.bm25_index().search("jujube", 10).unwrap().is_empty());
        assert!(engine.commit_for_read(None).unwrap());
        assert_eq!(engine.bm25_index().search("jujube", 10).unwrap().len(), 1);
    }

    /// Idempotency at the unified boundary: applying the same
    /// `Remember` twice (e.g. a replay that re-applies past the last
    /// checkpoint) still leaves exactly one document, and the engine
    /// agrees on the count.
    #[test]
    fn duplicate_remember_through_projection_yields_single_doc_via_engine() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("unified_idempotent");
        let (index, _ckpt, mut proj) = open_projection(&bm25_path, &checkpoint_path);
        let engine = open_engine_sharing(index.clone());

        let id = Uuid::new_v4();
        let memory = mk_memory(id, "exactly one body via engine", &[], &[]);
        let payload = remember_payload(&memory);

        TypedProjection::apply(&mut proj, Lsn(5), &payload).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();
        TypedProjection::apply(&mut proj, Lsn(5), &payload).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();

        assert_eq!(engine.bm25_doc_count(), 1);
        let hits = engine
            .bm25_index()
            .search("exactly one body via engine", 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, MemoryId(id));
    }

    /// Opening the projection-owned index at a tenant's path must NOT
    /// create the legacy `bm25_index/` sibling directory. The unified
    /// design uses a single tantivy directory at `bm25_projection_index/`;
    /// any artefact at the old path would be evidence of a code path
    /// that still goes through `HybridSearchEngine::new` with the old
    /// directory name.
    #[test]
    fn legacy_bm25_index_directory_is_not_created_on_fresh_tenant() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("no_legacy_dir");
        // `bm25_path` is `{tenant}/bm25_index` in the test helper; for
        // the unification assertion we need the *tenant* root and to
        // open the projection at the canonical
        // `bm25_projection_index` subdirectory.
        let tenant = bm25_path.parent().unwrap().to_path_buf();
        let unified_index_dir = tenant.join(crate::memory::BM25_INDEX_DIR);
        let legacy_index_dir = tenant.join("bm25_index");

        let (_index, _ckpt, mut proj) = open_projection(&unified_index_dir, &checkpoint_path);

        let id = Uuid::new_v4();
        let memory = mk_memory(id, "tenant body", &[], &[]);
        TypedProjection::apply(&mut proj, Lsn(0), &remember_payload(&memory)).unwrap();
        Projection::persist_checkpoint(&mut proj).unwrap();

        assert!(
            unified_index_dir.exists(),
            "projection-owned BM25 directory must exist after a write"
        );
        assert!(
            !legacy_index_dir.exists(),
            "legacy '{}' directory must not be created — it is a sentinel for the pre-unification dual-write bug",
            legacy_index_dir.display()
        );
    }
}
