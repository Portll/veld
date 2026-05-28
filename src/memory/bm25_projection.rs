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
//! until `commit()` runs. Committing on every record is correct but slow
//! (each commit fsyncs the segment + bumps the meta), so the projection
//! commits in batches.
//!
//! The cadence is owned in one place — [`COMMIT_EVERY`] — and the replay
//! driver is told to persist the projection checkpoint with the same
//! period via `replay(..., Some(COMMIT_EVERY))`. This means the
//! on-disk BM25 segments and the persisted intent-log checkpoint advance
//! together: a crash that loses an unflushed batch also loses the
//! checkpoint for that batch, so on restart the replay driver re-applies
//! exactly the records the BM25 index never durably saw.
//!
//! ## Idempotency
//!
//! Every operation is keyed by `memory_id` (the UUID inside `MemoryId`):
//!
//! - `Remember` / `Update` — `BM25Index::upsert` does a `delete_term` on
//!   the `id` field followed by `add_document`. Replaying the same LSN
//!   twice produces exactly one document in the index.
//! - `Forget` — `BM25Index::delete` is a `delete_term` on `id`. Deleting a
//!   missing document is a no-op in tantivy.
//! - `Anchor` — no-op. Anchoring affects importance/decay, not text
//!   content; the BM25 corpus is unchanged.
//!
//! Re-applying any LSN twice produces the same state — the trait contract
//! the replay driver depends on.

use std::error::Error;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::intent_log::{
    CheckpointStore, CheckpointStoreError, IntentPayload, IntentRecord, Lsn, PayloadError,
    Projection, TypedProjection,
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

/// Number of successful applies between `IndexWriter::commit` calls in
/// the live-write path AND between `persist_checkpoint` calls in the
/// replay path.
///
/// Tantivy commits are expensive (fsync of segment files + meta), so we
/// amortise. The replay driver is given the same number as
/// `persist_every` so the BM25 segments and the intent-log checkpoint
/// advance in lock-step: any batch we lose to a crash is the same batch
/// the next replay will re-apply.
pub const COMMIT_EVERY: u64 = 100;

/// Bridge between the intent log and the BM25 inverted index.
///
/// One per tenant. Owns an `Arc<BM25Index>` (the tantivy handle shared
/// between the projection and any read-side hybrid-search reader), a
/// shared [`CheckpointStore`] (the per-projection LSN bookkeeping), and
/// an in-memory copy of the last-applied LSN.
pub struct Bm25Projection {
    index: Arc<BM25Index>,
    checkpoint_store: Arc<Mutex<CheckpointStore>>,
    /// LSN of the last record this projection successfully applied. Lives
    /// in memory; `persist_checkpoint` synchronises it to disk.
    checkpoint: Option<Lsn>,
    /// Number of successful applies since the last `commit`. Reset on
    /// commit. Lets the dyn-`TypedProjection` arm batch commits without
    /// the replay driver's `persist_every` ceremony.
    applies_since_commit: u64,
}

impl Bm25Projection {
    /// Construct a projection around an open BM25 index. Reads the
    /// last-persisted checkpoint from `checkpoint_store` so subsequent
    /// `replay` / live `apply` calls resume from the right LSN.
    pub fn new(
        index: Arc<BM25Index>,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
    ) -> Self {
        let checkpoint = checkpoint_store.lock().get(PROJECTION_NAME);
        Self {
            index,
            checkpoint_store,
            checkpoint,
            applies_since_commit: 0,
        }
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
                let content = memory.experience.content.as_str();
                let tags = memory.experience.tags.as_slice();
                let entities = memory.experience.entities.as_slice();
                self.index
                    .upsert(&mid, content, tags, entities)
                    .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
            }
            IntentPayload::Forget { memory_id, .. } => {
                let mid = parse_memory_id(memory_id)?;
                self.index
                    .delete(&mid)
                    .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
            }
            IntentPayload::Anchor { .. } => {
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

    /// Commit the underlying tantivy writer if at least one record has
    /// been applied since the last commit. Reloads the reader so reads
    /// see the freshly-committed segments without an external nudge.
    fn commit_index(&mut self) -> Result<(), Bm25ProjectionError> {
        if self.applies_since_commit == 0 {
            return Ok(());
        }
        self.index
            .commit()
            .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
        // Reload the reader so any read-side hybrid search engine sharing
        // this `Arc<BM25Index>` sees the new documents immediately. The
        // reload is cheap — it just re-mmaps the segment files.
        self.index
            .reload()
            .map_err(|e| Bm25ProjectionError::Bm25(e.to_string()))?;
        self.applies_since_commit = 0;
        Ok(())
    }

    /// Persist the in-memory checkpoint to the shared checkpoint store.
    /// Called by the replay driver on its `persist_every` cadence and at
    /// the end of every live-write commit boundary.
    fn persist(&mut self) -> Result<(), Bm25ProjectionError> {
        if let Some(lsn) = self.checkpoint {
            let mut store = self.checkpoint_store.lock();
            store.set(PROJECTION_NAME, lsn)?;
            store.sync()?;
        }
        Ok(())
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

/// Decode a bincoded `Memory` snapshot off an `IntentPayload`. Bincode
/// configuration mirrors what `handlers/remember.rs` uses when it encodes
/// the snapshot — `bincode::config::standard()` — so a snapshot written
/// by the live-write path round-trips here without a config mismatch.
fn decode_memory(bytes: &[u8]) -> Result<Memory, Bm25ProjectionError> {
    let (memory, _consumed): (Memory, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| Bm25ProjectionError::MemoryDecode(e.to_string()))?;
    Ok(memory)
}

/// Dyn-friendly arm: the [`crate::intent_log::JournaledWriter`] hands us
/// typed payloads at live-write time. We batch commits via
/// [`COMMIT_EVERY`] and persist the checkpoint every time we commit so a
/// crash never leaves the BM25 segments and the checkpoint out of sync.
impl TypedProjection for Bm25Projection {
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
        self.applies_since_commit += 1;
        if self.applies_since_commit >= COMMIT_EVERY {
            self.commit_index()
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
            self.persist()
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        }
        Ok(())
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
    use crate::intent_log::{
        payload::CURRENT_PAYLOAD_SCHEMA_VERSION, replay, IntentLog,
    };
    use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId};
    use std::path::{Path, PathBuf};
    use uuid::Uuid;

    fn tmp_paths(stem: &str) -> (PathBuf, PathBuf, PathBuf) {
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir()
            .join(format!("veld-bm25-proj-{stem}-{pid}-{stamp}"));
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
        let ckpt = Arc::new(Mutex::new(
            CheckpointStore::open(checkpoint_path).unwrap(),
        ));
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
        let bytes =
            bincode::serde::encode_to_vec(memory, bincode::config::standard()).unwrap();
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
                let memory = mk_memory(
                    id,
                    &format!("replay body {i}"),
                    &[&format!("tag-{i}")],
                    &[],
                );
                crate::intent_log::payload::append(&mut log, &remember_payload(&memory))
                    .unwrap();
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
            let hits = index
                .search(&format!("replay body {i}"), 10)
                .unwrap();
            assert!(!hits.is_empty(), "expected hit for memory {i}");
            assert_eq!(hits[0].0, MemoryId(*id), "memory {i} ranked first");
        }
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
                crate::intent_log::payload::append(&mut log, &remember_payload(&memory))
                    .unwrap();
            }
            log.sync().unwrap();
        }

        {
            let log = IntentLog::open(&log_path).unwrap();
            let (_index, _ckpt, mut proj) =
                open_projection(&bm25_path, &checkpoint_path);
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
        let (index, _ckpt, mut proj) =
            open_projection(&bm25_path, &checkpoint_path);
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
        let (index, _ckpt, mut proj) =
            open_projection(&bm25_path, &checkpoint_path);

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

    /// Malformed memory_id in the payload returns a structured error
    /// instead of panicking — the journaled writer surfaces it via
    /// `WriteOutcome.apply_errors` without failing the write.
    #[test]
    fn malformed_memory_id_produces_structured_error() {
        let (bm25_path, _, checkpoint_path) = tmp_paths("bad_uuid");
        let (_index, _ckpt, mut proj) =
            open_projection(&bm25_path, &checkpoint_path);

        let memory = mk_memory(Uuid::new_v4(), "doesn't matter", &[], &[]);
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
            .unwrap();
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
}
