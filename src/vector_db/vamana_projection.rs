//! Second real intent-log projection: the Vamana vector index.
//!
//! ## Why this projection exists
//!
//! `c673647` made the SQLite slow store the first real projection of the
//! W5 intent log. Vamana — the on-disk vector index that backs similarity
//! search — was still updated *out of band* by the existing
//! `RetrievalEngine::index_memory` path, with no replayable record. If a
//! crash interrupted the gap between RocksDB and Vamana, the vector
//! index silently drifted from the source of truth.
//!
//! This module closes that gap. The journaled writer now dispatches every
//! state-changing memory operation to a Vamana projection too, and on
//! restart the projection replays the log to catch up. Both live writes
//! and replay flow through one helper so the two code paths cannot drift.
//!
//! ## Embedding kinds
//!
//! `Memory` carries up to five embedding kinds (primary text, secondary
//! text, image, audio, video). One projection handles ONE kind — each
//! tenant therefore runs up to five parallel Vamana projections, with
//! their own checkpoint, their own on-disk graph, and their own
//! `memory_id → vector_id` map. Memories that don't carry the relevant
//! field are skipped at warn-log level (the checkpoint still advances so
//! replay doesn't loop on that frame forever).
//!
//! Per-kind separation is cleaner than a multi-vector node type:
//!
//! - dimensions usually differ (primary 384, secondary 768, CLIP 512+)
//! - update rates differ (text dominates; audio/video are rare)
//! - retrieval scoring fuses results in a separate stage, not the index
//!
//! Checkpoint names + on-disk paths are derived from the kind, so a
//! wipe-and-replay of one kind doesn't touch the others.
//!
//! ## Idempotency
//!
//! Every operation is keyed by `(user_id, memory_id)`:
//!
//! - `Remember` / `Update` — UPSERT. If the same `memory_id` is seen
//!   again, the old vector is soft-deleted in Vamana and a fresh one is
//!   inserted, gated on `lsn >= last_applied_for(memory_id)` so a stale
//!   replay can never overwrite a newer live write.
//! - `Forget` — soft-delete the existing vector(s) for this memory and
//!   drop the mapping. Already-deleted memories are a no-op.
//! - `Anchor` — no-op. Anchors change importance, not vector state.
//!
//! Re-applying any LSN twice produces the same state — the trait
//! contract the replay driver depends on.
//!
//! ## Checkpoint persistence
//!
//! The projection holds an in-memory checkpoint `Lsn` and a reference to
//! a shared [`CheckpointStore`] file. The static `Projection` arm
//! flushes the checkpoint via `persist_checkpoint`. The dyn
//! `TypedProjection` arm — used by the journaled writer at live-write
//! time — also persists after each apply so a crash between live writes
//! resumes from the most recent durable position, not from the start of
//! the log.
//!
//! ## `memory_id → vector_id` persistence
//!
//! The in-memory map of memory_id → vector_id is also persisted to the
//! checkpoint store as side data after each apply. On open, the
//! projection loads its side data and uses it instead of rebuilding from
//! the log — but only if the side-data's stamped LSN matches the
//! projection's checkpoint LSN exactly. Any mismatch (corrupt side data,
//! stale snapshot, replayed log under it) discards the side data and
//! falls back to a fresh replay. This is defence in depth: the log is
//! the canonical source of truth; side data is just opportunistic
//! acceleration that we never let outvote the log.

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::intent_log::{
    CheckpointStore, CheckpointStoreError, IntentPayload, IntentRecord, Lsn, PayloadError,
    Projection, TypedProjection,
};
use crate::memory::types::Memory;

use super::vamana::{VamanaConfig, VamanaIndex};

/// Errors raised by [`VamanaProjection`]. Distinct from `anyhow::Error`
/// because [`Projection::Error`] requires `std::error::Error` and
/// `anyhow::Error` does not implement it. Each variant wraps the
/// underlying error so callers can match on the precise failure mode.
#[derive(Debug, thiserror::Error)]
pub enum VamanaProjectionError {
    /// A Vamana-side operation (insert/delete) failed.
    #[error("vamana index op failed: {0}")]
    Vamana(String),
    /// Decoding the bincoded payload off the intent log frame failed.
    /// Surfaces only from the static `Projection::apply` arm where the
    /// driver hands us raw record bytes.
    #[error("intent log payload decode error: {0}")]
    PayloadDecode(#[from] PayloadError),
    /// Decoding the inner bincoded `Memory` snapshot off the payload
    /// failed. The intent log frame is well-formed but its inner blob is
    /// not a valid `Memory` for this binary — the operator likely needs
    /// to run a schema migration.
    #[error("memory bincode decode error: {0}")]
    MemoryDecode(String),
    /// Persisting the checkpoint to disk failed.
    #[error("checkpoint store error: {0}")]
    Checkpoint(#[from] CheckpointStoreError),
    /// A memory embedding's dimension does not match the projection's
    /// Vamana index dimension. Surfaces when a fresh projection has been
    /// configured for a different dimension than the corpus was embedded
    /// at — recoverable only by rebuilding the projection with the
    /// correct dimension.
    #[error(
        "embedding dimension mismatch for memory {memory_id}: \
         memory={memory_dim}, index={index_dim}"
    )]
    DimensionMismatch {
        memory_id: String,
        memory_dim: usize,
        index_dim: usize,
    },
}

/// Embedding kind that a [`VamanaProjection`] indexes. One projection
/// per kind per tenant; each carries its own checkpoint name, on-disk
/// path component, and field selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VamanaEmbeddingKind {
    /// `experience.embeddings` — primary text embedding. The default
    /// model (typically MiniLM 384d). This is the kind that was indexed
    /// before multi-kind support landed.
    TextPrimary,
    /// `experience.embeddings_secondary` — secondary text embedding
    /// (typically Nomic-embed-text-v1.5 768d) used for dual-index
    /// competitive retrieval.
    TextSecondary,
    /// `experience.image_embeddings` — CLIP/SigLIP image embedding.
    Image,
    /// `experience.audio_embeddings` — Whisper/wav2vec audio embedding.
    Audio,
    /// `experience.video_embeddings` — frame-averaged or keyframe video
    /// embedding.
    Video,
}

impl VamanaEmbeddingKind {
    /// Stable projection name persisted to the [`CheckpointStore`].
    /// CRITICAL: do not rename these in place — operators' checkpoint
    /// stores are keyed on these strings, and a rename would be a manual
    /// migration. Add a new variant first, migrate, then remove.
    pub fn projection_name(self) -> &'static str {
        match self {
            VamanaEmbeddingKind::TextPrimary => "vamana-text-primary",
            VamanaEmbeddingKind::TextSecondary => "vamana-text-secondary",
            VamanaEmbeddingKind::Image => "vamana-image",
            VamanaEmbeddingKind::Audio => "vamana-audio",
            VamanaEmbeddingKind::Video => "vamana-video",
        }
    }

    /// Directory component for this kind under
    /// `{user_path}/vamana_projection/{component}/`. Matches the
    /// projection name minus the `vamana-` prefix to keep the layout
    /// short and human-greppable.
    pub fn dir_component(self) -> &'static str {
        match self {
            VamanaEmbeddingKind::TextPrimary => "text-primary",
            VamanaEmbeddingKind::TextSecondary => "text-secondary",
            VamanaEmbeddingKind::Image => "image",
            VamanaEmbeddingKind::Audio => "audio",
            VamanaEmbeddingKind::Video => "video",
        }
    }

    /// Pull the embedding field this kind cares about off a decoded
    /// `Memory`. Cloned out of the decoded snapshot — the snapshot is
    /// discarded after extraction, so a clone here is unavoidable.
    /// Returns `None` if the memory does not carry that kind.
    pub fn extract_embedding(self, memory: &Memory) -> Option<Vec<f32>> {
        match self {
            VamanaEmbeddingKind::TextPrimary => memory.experience.embeddings.clone(),
            VamanaEmbeddingKind::TextSecondary => {
                memory.experience.embeddings_secondary.clone()
            }
            VamanaEmbeddingKind::Image => memory.experience.image_embeddings.clone(),
            VamanaEmbeddingKind::Audio => memory.experience.audio_embeddings.clone(),
            VamanaEmbeddingKind::Video => memory.experience.video_embeddings.clone(),
        }
    }

    /// Enumerate every supported kind in stable order. Used by the
    /// projection registrar to construct one projection per kind per
    /// tenant without hand-listing variants.
    pub fn all() -> &'static [VamanaEmbeddingKind] {
        &[
            VamanaEmbeddingKind::TextPrimary,
            VamanaEmbeddingKind::TextSecondary,
            VamanaEmbeddingKind::Image,
            VamanaEmbeddingKind::Audio,
            VamanaEmbeddingKind::Video,
        ]
    }
}

/// Side-data key under which the projection persists its
/// `memory_id → (vector_id, last_applied_lsn)` snapshot. One slot per
/// projection — different kinds use different projection names so
/// there's no collision.
const SIDE_DATA_KEY_ID_MAP: &str = "id_map";

/// Per-`memory_id` bookkeeping kept inside the projection. We need the
/// `u32` vector-id assigned by Vamana so a later `Forget`/`Update` can
/// soft-delete the right node, and we need the `last_applied_lsn` so
/// stale replays can be skipped (higher LSN wins).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct MemoryEntry {
    vector_id: u32,
    last_applied_lsn: Lsn,
}

/// Template the projection uses to bootstrap a fresh Vamana index when
/// it sees its first embedding. Lets the projection be opened *before*
/// the embedder dimension is known — the dim comes off the first
/// `embeddings: Some(_)` payload off the log.
#[derive(Debug, Clone)]
pub struct VamanaProjectionBootstrap {
    /// Optional on-disk storage directory for the index (passed straight
    /// through to `VamanaIndex::with_storage_path`). `None` keeps the
    /// index fully in-memory (used by tests).
    pub storage_path: Option<PathBuf>,
    /// All Vamana knobs except `dimension`, which is filled in lazily
    /// from the first observed embedding.
    pub config_template: VamanaConfig,
}

impl VamanaProjectionBootstrap {
    /// Convenience: bootstrap that creates a default Vamana with the
    /// given storage path. The dim is left at 0 (template-only) — the
    /// projection fills it in on first apply.
    pub fn at(storage_path: PathBuf) -> Self {
        Self {
            storage_path: Some(storage_path),
            config_template: VamanaConfig {
                // dimension is intentionally not the default here — the
                // projection rewrites it as soon as the first embedding
                // arrives. Setting 0 here makes a bug obvious if someone
                // ever instantiates the index before that point.
                dimension: 0,
                ..VamanaConfig::default()
            },
        }
    }
}

/// Bridge between the intent log and a [`VamanaIndex`].
///
/// One per tenant per [`VamanaEmbeddingKind`]. Either holds an
/// already-constructed `Arc<RwLock<VamanaIndex>>` (the eager variant —
/// caller knows the dim) OR a `VamanaProjectionBootstrap` that builds
/// the index on the first embedded memory it sees (the lazy variant —
/// caller doesn't know the dim yet because no embedder has been opened
/// for this tenant). Holds a shared [`CheckpointStore`] for
/// per-projection LSN bookkeeping, an in-memory `memory_id → u32` map,
/// and an in-memory copy of the last-applied LSN.
pub struct VamanaProjection {
    kind: VamanaEmbeddingKind,
    state: IndexState,
    checkpoint_store: Arc<Mutex<CheckpointStore>>,
    /// `memory_id → (vector_id, last_applied_lsn)`. Lives in memory; on
    /// startup the projection tries to restore it from the checkpoint
    /// store's side-data slot, and falls back to log replay if the
    /// side-data is missing or stale.
    memories: HashMap<String, MemoryEntry>,
    /// LSN of the last record this projection successfully applied. Lives
    /// in memory; `persist_checkpoint` synchronises it to disk.
    checkpoint: Option<Lsn>,
}

/// Either an already-open Vamana index or a template waiting to learn
/// its dimension from the first embedding.
enum IndexState {
    Ready(Arc<RwLock<VamanaIndex>>),
    Pending(VamanaProjectionBootstrap),
}

impl IndexState {
    /// Borrow the index if it's been materialised. Returns `None` for
    /// the pending state — callers that need the index must go through
    /// `ensure_ready` first.
    fn as_ready(&self) -> Option<&Arc<RwLock<VamanaIndex>>> {
        match self {
            IndexState::Ready(idx) => Some(idx),
            IndexState::Pending(_) => None,
        }
    }
}

impl VamanaProjection {
    /// Construct a projection around an already-open Vamana index for
    /// the given embedding kind. Use this when the caller knows the
    /// embedding dimension up-front (e.g. they read it off the embedder
    /// cache before opening the projection). Reads the last-persisted
    /// checkpoint from `checkpoint_store` so subsequent `replay` / live
    /// `apply` calls resume from the right LSN, and attempts to restore
    /// the `memory_id → vector_id` map from the checkpoint store's
    /// side-data slot (falling back to an empty map if absent or stale).
    pub fn new(
        kind: VamanaEmbeddingKind,
        index: Arc<RwLock<VamanaIndex>>,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
    ) -> Self {
        let (checkpoint, memories) = load_checkpoint_and_map(kind, &checkpoint_store);
        Self {
            kind,
            state: IndexState::Ready(index),
            checkpoint_store,
            memories,
            checkpoint,
        }
    }

    /// Construct a projection for the given embedding kind that will
    /// lazily build its Vamana index from the first embedded memory it
    /// sees. Useful when the projection is opened before any embedder
    /// is — typically at server startup, where the per-tenant
    /// `RetrievalEngine` is constructed on first request, not at boot.
    pub fn lazy(
        kind: VamanaEmbeddingKind,
        bootstrap: VamanaProjectionBootstrap,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
    ) -> Self {
        let (checkpoint, memories) = load_checkpoint_and_map(kind, &checkpoint_store);
        Self {
            kind,
            state: IndexState::Pending(bootstrap),
            checkpoint_store,
            memories,
            checkpoint,
        }
    }

    /// Embedding kind this projection handles.
    pub fn kind(&self) -> VamanaEmbeddingKind {
        self.kind
    }

    /// Stable projection name — same string used in
    /// [`CheckpointStore`] keys.
    pub fn projection_name(&self) -> &'static str {
        self.kind.projection_name()
    }

    /// Materialise the Vamana index using the bootstrap template +
    /// observed dimension. No-op if the projection was created eagerly.
    fn ensure_ready(
        &mut self,
        observed_dim: usize,
    ) -> Result<&Arc<RwLock<VamanaIndex>>, VamanaProjectionError> {
        if let IndexState::Pending(b) = &self.state {
            let mut cfg = b.config_template.clone();
            cfg.dimension = observed_dim;
            let index = match &b.storage_path {
                Some(p) => VamanaIndex::with_storage_path(cfg, Some(p.clone())),
                None => VamanaIndex::new(cfg),
            }
            .map_err(|e| VamanaProjectionError::Vamana(e.to_string()))?;
            self.state = IndexState::Ready(Arc::new(RwLock::new(index)));
        }
        // Safe because we just set Ready above (or it was already Ready).
        Ok(self.state.as_ready().expect("projection index ready"))
    }

    /// Borrow the underlying index if it has been materialised. Returns
    /// `None` for a `lazy` projection that has not yet seen its first
    /// embedded memory.
    pub fn index(&self) -> Option<&Arc<RwLock<VamanaIndex>>> {
        self.state.as_ready()
    }

    /// Look up the Vamana vector-id associated with a `memory_id`, if
    /// any. Returns `None` for memories the projection has not seen (or
    /// has soft-deleted via `Forget`).
    pub fn vector_id_for(&self, memory_id: &str) -> Option<u32> {
        self.memories.get(memory_id).map(|e| e.vector_id)
    }

    /// Number of live (non-forgotten) memory-id mappings the projection
    /// is currently tracking. Soft-deleted vectors stay in the Vamana
    /// graph until the next rebuild but are not counted here.
    pub fn tracked_memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Apply a typed payload at a specific LSN. Pulled into a helper so
    /// both the dyn `TypedProjection::apply` arm and the static
    /// `Projection::apply` arm dispatch through one code path.
    fn apply_typed(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), VamanaProjectionError> {
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
                // Stale-replay guard. If we already applied a newer LSN
                // for this memory, drop the older one on the floor — the
                // newer state already won. Advance the projection
                // checkpoint so the driver doesn't loop on this frame.
                if let Some(existing) = self.memories.get(memory_id) {
                    if lsn <= existing.last_applied_lsn {
                        self.checkpoint = Some(self.bumped_checkpoint(lsn));
                        return Ok(());
                    }
                }

                let memory = decode_memory(memory_bincode)?;
                let embedding = match self.kind.extract_embedding(&memory) {
                    Some(emb) => emb,
                    None => {
                        // Memory has no embedding for this kind —
                        // nothing to feed Vamana. This is not a hard
                        // error: a memory might carry primary text but
                        // no image, and an image projection should
                        // happily skip it. Advance the checkpoint so
                        // replay doesn't loop on this frame forever.
                        tracing::warn!(
                            projection = self.kind.projection_name(),
                            memory_id = %memory_id,
                            lsn = lsn.0,
                            "vamana projection: skipping memory without embedding for this kind (no-op apply; checkpoint advances)",
                        );
                        self.checkpoint = Some(self.bumped_checkpoint(lsn));
                        return Ok(());
                    }
                };

                // Materialise the index (no-op if eager / already
                // ready). Lazy projections learn their dim here from
                // the first embedded memory off the log.
                let index_arc = self.ensure_ready(embedding.len())?.clone();

                // Dimension check is a hard error — silently indexing a
                // wrong-dim vector would corrupt the graph (see
                // `VamanaIndex::add_vector`'s own guard). After
                // `ensure_ready`, the index dim is fixed for the
                // lifetime of the projection; subsequent applies must
                // match it or fail loudly.
                let index_dim = index_arc.read().config_dimension();
                if embedding.len() != index_dim {
                    return Err(VamanaProjectionError::DimensionMismatch {
                        memory_id: memory_id.clone(),
                        memory_dim: embedding.len(),
                        index_dim,
                    });
                }

                // Re-applying the same memory_id at a NEWER LSN means
                // the memory was edited. Soft-delete the old vector so
                // search results stop including it; the next rebuild
                // physically removes it.
                if let Some(existing) = self.memories.get(memory_id).copied() {
                    let idx = index_arc.read();
                    idx.mark_deleted(existing.vector_id);
                    drop(idx);
                }

                let new_vid = {
                    let mut idx = index_arc.write();
                    idx.add_vector(embedding)
                        .map_err(|e| VamanaProjectionError::Vamana(e.to_string()))?
                };

                self.memories.insert(
                    memory_id.clone(),
                    MemoryEntry {
                        vector_id: new_vid,
                        last_applied_lsn: lsn,
                    },
                );
            }
            IntentPayload::Forget { memory_id, .. } => {
                // Already-forgotten memories are a no-op (idempotent).
                // We still advance the checkpoint so the driver
                // progresses past this frame.
                if let Some(existing) = self.memories.remove(memory_id) {
                    // Only touch the index if it's been materialised —
                    // a `Forget` on a pending lazy projection means we
                    // saw the create-then-forget pair without ever
                    // seeing an embedding, so there's no Vamana node to
                    // delete.
                    if let Some(idx_arc) = self.state.as_ready() {
                        let idx = idx_arc.read();
                        idx.mark_deleted(existing.vector_id);
                        drop(idx);
                    }
                }
                // Else: nothing to delete — checkpoint advance below.
            }
            IntentPayload::Anchor { .. } => {
                // Anchors are an importance bump on the memory, not a
                // vector-state change. Vamana doesn't care.
            }
        }
        self.checkpoint = Some(self.bumped_checkpoint(lsn));
        Ok(())
    }

    /// Pick the new in-memory checkpoint position given that we just
    /// applied (or skipped, for idempotency) `lsn`. The checkpoint is
    /// the *highest* LSN we've ever moved past, so a stale-replay
    /// skip doesn't roll the checkpoint backwards.
    fn bumped_checkpoint(&self, lsn: Lsn) -> Lsn {
        match self.checkpoint {
            Some(existing) if existing >= lsn => existing,
            _ => lsn,
        }
    }

    fn persist(&mut self) -> Result<(), VamanaProjectionError> {
        if let Some(lsn) = self.checkpoint {
            let mut store = self.checkpoint_store.lock();
            store.set(self.kind.projection_name(), lsn)?;
            // Snapshot the id map under the same projection name so a
            // later open can restore it without replaying the log.
            // Encoded with `bincode::config::standard()` to match every
            // other on-disk encoding in this codebase.
            let encoded = bincode::serde::encode_to_vec(
                &self.memories,
                bincode::config::standard(),
            )
            .map_err(|e| {
                VamanaProjectionError::Vamana(format!("encode id_map: {e}"))
            })?;
            store.set_side_data(
                self.kind.projection_name(),
                SIDE_DATA_KEY_ID_MAP,
                &encoded,
            )?;
            store.sync()?;
        }
        Ok(())
    }
}

/// Load the projection's checkpoint LSN from the store, plus its
/// persisted `memory_id → vector_id` map *if* the side-data's stamped
/// LSN matches the checkpoint LSN. Any mismatch (stale snapshot, decode
/// failure, side data missing) returns an empty map and lets the caller
/// rebuild it via replay — that's the defence-in-depth the spec calls
/// for.
fn load_checkpoint_and_map(
    kind: VamanaEmbeddingKind,
    checkpoint_store: &Arc<Mutex<CheckpointStore>>,
) -> (Option<Lsn>, HashMap<String, MemoryEntry>) {
    let store = checkpoint_store.lock();
    let checkpoint = store.get(kind.projection_name());
    let map = match (
        checkpoint,
        store.get_side_data(kind.projection_name(), SIDE_DATA_KEY_ID_MAP),
        store.side_data_checkpoint(kind.projection_name(), SIDE_DATA_KEY_ID_MAP),
    ) {
        // Side data present AND its stamp matches the projection's
        // current checkpoint — trust it.
        (Some(ckpt), Some(bytes), Some(stamp)) if stamp == ckpt => {
            match bincode::serde::decode_from_slice::<HashMap<String, MemoryEntry>, _>(
                bytes,
                bincode::config::standard(),
            ) {
                Ok((map, _)) => map,
                Err(e) => {
                    tracing::warn!(
                        projection = kind.projection_name(),
                        error = %e,
                        "vamana projection: id_map side data failed to decode; falling back to replay-rebuild",
                    );
                    HashMap::new()
                }
            }
        }
        // Side data exists but its stamp is stale (or projection
        // checkpoint is missing). Trust the log instead — replay will
        // rebuild a fresh map.
        (ckpt, Some(_bytes), Some(stamp)) => {
            tracing::info!(
                projection = kind.projection_name(),
                side_stamp = stamp.0,
                checkpoint_lsn = ckpt.map(|l| l.0 as i128).unwrap_or(-1),
                "vamana projection: id_map side data stamp != checkpoint; discarding and replaying",
            );
            HashMap::new()
        }
        // No side data at all — first boot of this projection, or it
        // has only ever persisted checkpoints (older binary). Replay
        // will populate.
        _ => HashMap::new(),
    };
    (checkpoint, map)
}

/// Decode a bincoded `Memory` snapshot off the wire. Mirrors the
/// `bincode::config::standard()` config used in `handlers::remember` and
/// `handlers::crud` when the snapshot was encoded — keep these in sync.
fn decode_memory(bytes: &[u8]) -> Result<Memory, VamanaProjectionError> {
    let (memory, _consumed): (Memory, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| VamanaProjectionError::MemoryDecode(e.to_string()))?;
    Ok(memory)
}

/// Dyn-friendly arm: the [`JournaledWriter`] hands us typed payloads at
/// live-write time. We immediately persist the checkpoint so a crash
/// after one successful apply doesn't silently roll back to the start of
/// the log.
impl TypedProjection for VamanaProjection {
    fn name(&self) -> &str {
        self.kind.projection_name()
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
}

/// Static arm: the replay driver hands us [`IntentRecord`]s straight off
/// the log. We decode the payload here so the trait stays generic over
/// payload shape — the replay driver doesn't know about `IntentPayload`.
impl Projection for VamanaProjection {
    type Error = VamanaProjectionError;

    fn name(&self) -> &str {
        self.kind.projection_name()
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

/// Small accessor on `VamanaIndex` for the dim guard above. Lives here
/// (not in `vamana.rs`) so this projection module is the only place that
/// reaches into the config — keeps the public surface of `VamanaIndex`
/// stable.
trait VamanaIndexConfigExt {
    fn config_dimension(&self) -> usize;
}

impl VamanaIndexConfigExt for VamanaIndex {
    fn config_dimension(&self) -> usize {
        self.config.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent_log::{payload::CURRENT_PAYLOAD_SCHEMA_VERSION, replay, IntentLog};
    use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId};
    use crate::vector_db::vamana::{DistanceMetric, VamanaConfig};
    use std::path::PathBuf;
    use uuid::Uuid;

    const TEST_DIM: usize = 8;

    fn tmp_paths(stem: &str) -> (PathBuf, PathBuf) {
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!("veld-vamana-proj-{stem}-{pid}-{stamp}"));
        std::fs::create_dir_all(&base).unwrap();
        (base.join("intent.log"), base.join("checkpoints.bin"))
    }

    fn test_config() -> VamanaConfig {
        VamanaConfig {
            dimension: TEST_DIM,
            max_degree: 8,
            search_list_size: 12,
            alpha: 1.2,
            // use_mmap=false keeps the test fully in-memory; the
            // projection contract is identical either way and disk mmap
            // requires a storage path which makes test cleanup noisy.
            use_mmap: false,
            distance_metric: DistanceMetric::Cosine,
        }
    }

    fn open_projection() -> (
        Arc<RwLock<VamanaIndex>>,
        Arc<Mutex<CheckpointStore>>,
        VamanaProjection,
        PathBuf,
        PathBuf,
    ) {
        let (log_path, ckpt_path) = tmp_paths("base");
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj = VamanaProjection::new(
            VamanaEmbeddingKind::TextPrimary,
            index.clone(),
            ckpt.clone(),
        );
        (index, ckpt, proj, log_path, ckpt_path)
    }

    fn mk_memory_with_embedding(emb: Vec<f32>) -> (MemoryId, Vec<u8>) {
        assert_eq!(emb.len(), TEST_DIM);
        let id = MemoryId(Uuid::new_v4());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: format!("test memory {}", id.0),
            embeddings: Some(emb),
            ..Experience::default()
        };
        let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        let bytes =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
        (id, bytes)
    }

    /// Build a memory carrying any combination of embedding kinds. The
    /// `None` slots stay `None`. Returns the id + bincoded payload.
    fn mk_memory_multi(
        text_primary: Option<Vec<f32>>,
        text_secondary: Option<Vec<f32>>,
        image: Option<Vec<f32>>,
        audio: Option<Vec<f32>>,
        video: Option<Vec<f32>>,
    ) -> (MemoryId, Vec<u8>) {
        let id = MemoryId(Uuid::new_v4());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: format!("multi memory {}", id.0),
            embeddings: text_primary,
            embeddings_secondary: text_secondary,
            image_embeddings: image,
            audio_embeddings: audio,
            video_embeddings: video,
            ..Experience::default()
        };
        let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        let bytes =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
        (id, bytes)
    }

    fn mk_remember(memory_id: &str, bincoded_memory: Vec<u8>) -> IntentPayload {
        IntentPayload::Remember {
            user_id: "alice".into(),
            memory_id: memory_id.to_string(),
            memory_bincode: bincoded_memory,
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        }
    }

    /// Generate a unit-norm `TEST_DIM`-d vector that points mostly along
    /// axis `axis`. Used to seed distinct, separable memories without
    /// having to call an embedder.
    fn axis_vec(axis: usize) -> Vec<f32> {
        let mut v = vec![0.01f32; TEST_DIM];
        v[axis % TEST_DIM] = 1.0;
        // Crude normalise so cosine-distance comparisons are meaningful.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn round_trip_three_memories_then_query_returns_the_right_id() {
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let (mid_a, b_a) = mk_memory_with_embedding(axis_vec(0));
        let (mid_b, b_b) = mk_memory_with_embedding(axis_vec(2));
        let (mid_c, b_c) = mk_memory_with_embedding(axis_vec(5));

        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid_a.0.to_string(), b_a))
            .unwrap();
        TypedProjection::apply(&mut proj, Lsn(1), &mk_remember(&mid_b.0.to_string(), b_b))
            .unwrap();
        TypedProjection::apply(&mut proj, Lsn(2), &mk_remember(&mid_c.0.to_string(), b_c))
            .unwrap();

        // Vamana now holds three vectors, projection tracks three ids.
        assert_eq!(index.read().len(), 3);
        assert_eq!(proj.tracked_memory_count(), 3);

        // Query along axis 2 → should match mid_b best. We resolve the
        // vector id back to the memory id via the projection's own map.
        let query = axis_vec(2);
        let hits = index.read().search(&query, 1).unwrap();
        assert_eq!(hits.len(), 1);
        let hit_vid = hits[0].0;
        let mid_b_vid = proj.vector_id_for(&mid_b.0.to_string()).unwrap();
        assert_eq!(hit_vid, mid_b_vid, "axis-2 query must return mid_b");
    }

    #[test]
    fn idempotent_re_apply_does_not_duplicate_node() {
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let (mid, bytes) = mk_memory_with_embedding(axis_vec(1));
        let payload = mk_remember(&mid.0.to_string(), bytes);

        TypedProjection::apply(&mut proj, Lsn(7), &payload).unwrap();
        // Second apply at the SAME lsn: stale-replay guard kicks in, no
        // new vector is added and no error is raised. The checkpoint
        // stays at 7 — re-applying the head is allowed and idempotent.
        TypedProjection::apply(&mut proj, Lsn(7), &payload).unwrap();

        assert_eq!(
            index.read().len(),
            1,
            "duplicate lsn must not add a second vector"
        );
        assert_eq!(proj.tracked_memory_count(), 1);
        assert_eq!(proj.checkpoint(), Some(Lsn(7)));
    }

    #[test]
    fn replay_from_scratch_rebuilds_full_state_from_log() {
        let (log_path, ckpt_path) = tmp_paths("replay");

        // Write 5 records directly to the intent log — simulates a
        // restart where the log has data and the projection has not
        // yet seen any of it.
        let mut memory_ids: Vec<String> = Vec::new();
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..5 {
                let (mid, bytes) = mk_memory_with_embedding(axis_vec(i));
                memory_ids.push(mid.0.to_string());
                crate::intent_log::payload::append(
                    &mut log,
                    &mk_remember(&mid.0.to_string(), bytes),
                )
                .unwrap();
            }
            log.sync().unwrap();
        }

        // Spin up a brand-new Vamana + projection. Run replay.
        let log = IntentLog::open(&log_path).unwrap();
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let mut proj = VamanaProjection::new(
            VamanaEmbeddingKind::TextPrimary,
            index.clone(),
            ckpt,
        );

        let applied = replay(&log, &mut proj, Some(2)).unwrap();
        assert_eq!(applied, 5);
        assert_eq!(index.read().len(), 5);
        assert_eq!(proj.tracked_memory_count(), 5);
        // Every memory_id has a mapping after replay.
        for mid in &memory_ids {
            assert!(
                proj.vector_id_for(mid).is_some(),
                "missing mapping for {mid} after replay"
            );
        }
    }

    #[test]
    fn forget_soft_deletes_and_is_idempotent_on_missing_rows() {
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let (mid, bytes) = mk_memory_with_embedding(axis_vec(3));
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid.0.to_string(), bytes))
            .unwrap();
        assert_eq!(index.read().len(), 1);

        let forget = IntentPayload::Forget {
            user_id: "alice".into(),
            memory_id: mid.0.to_string(),
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        TypedProjection::apply(&mut proj, Lsn(1), &forget).unwrap();
        // The Vamana node is soft-deleted (graph still holds it; search
        // filters it out). Projection mapping is gone.
        assert_eq!(index.read().deleted_count(), 1);
        assert!(proj.vector_id_for(&mid.0.to_string()).is_none());

        // Apply the same Forget again — no row in the mapping anymore,
        // no error.
        TypedProjection::apply(&mut proj, Lsn(2), &forget).unwrap();
        assert_eq!(index.read().deleted_count(), 1, "second forget must not add a delete");

        // And a Forget for a memory the projection has never seen is
        // also a no-op.
        let ghost = IntentPayload::Forget {
            user_id: "alice".into(),
            memory_id: Uuid::new_v4().to_string(),
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        TypedProjection::apply(&mut proj, Lsn(3), &ghost).unwrap();
    }

    #[test]
    fn update_replaces_old_vector_under_same_memory_id() {
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let id = MemoryId(Uuid::new_v4());
        let id_str = id.0.to_string();

        let mk = |emb: Vec<f32>| -> Vec<u8> {
            let experience = Experience {
                experience_type: ExperienceType::Observation,
                content: "test".to_string(),
                embeddings: Some(emb),
                ..Experience::default()
            };
            let m = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
            bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap()
        };

        TypedProjection::apply(
            &mut proj,
            Lsn(0),
            &IntentPayload::Remember {
                user_id: "alice".into(),
                memory_id: id_str.clone(),
                memory_bincode: mk(axis_vec(0)),
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();
        let first_vid = proj.vector_id_for(&id_str).unwrap();

        // Update at a higher LSN — old vector must be soft-deleted, a
        // fresh one inserted, and the mapping updated to the new vid.
        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &IntentPayload::Update {
                user_id: "alice".into(),
                memory_id: id_str.clone(),
                memory_bincode: mk(axis_vec(3)),
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();
        let second_vid = proj.vector_id_for(&id_str).unwrap();
        assert_ne!(first_vid, second_vid, "update must assign a new vector id");
        assert!(index.read().is_deleted(first_vid));
        assert!(!index.read().is_deleted(second_vid));
        // Vamana now has 2 raw vectors (one soft-deleted) and the
        // projection tracks exactly one live mapping for this id.
        assert_eq!(index.read().len(), 2);
        assert_eq!(proj.tracked_memory_count(), 1);
    }

    #[test]
    fn higher_lsn_wins_when_stale_replay_collides() {
        // Simulate a race: a newer Remember at LSN 5 lands first
        // (live), then replay re-feeds the older LSN-1 frame. The newer
        // state must survive.
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let id = MemoryId(Uuid::new_v4());
        let id_str = id.0.to_string();

        let mk = |emb: Vec<f32>| -> Vec<u8> {
            let experience = Experience {
                experience_type: ExperienceType::Observation,
                content: "test".to_string(),
                embeddings: Some(emb),
                ..Experience::default()
            };
            let m = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
            bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap()
        };

        TypedProjection::apply(
            &mut proj,
            Lsn(5),
            &IntentPayload::Remember {
                user_id: "alice".into(),
                memory_id: id_str.clone(),
                memory_bincode: mk(axis_vec(4)),
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();
        let new_vid = proj.vector_id_for(&id_str).unwrap();

        // Stale replay at LSN 1 — must NOT replace the newer mapping.
        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &IntentPayload::Remember {
                user_id: "alice".into(),
                memory_id: id_str.clone(),
                memory_bincode: mk(axis_vec(0)),
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();
        assert_eq!(proj.vector_id_for(&id_str), Some(new_vid));
        // Only one vector ever made it into Vamana — the stale frame
        // was dropped on the floor.
        assert_eq!(index.read().len(), 1);
        // Checkpoint stays at the highest LSN we've ever moved past.
        assert_eq!(proj.checkpoint(), Some(Lsn(5)));
    }

    #[test]
    fn anchor_is_a_noop_on_vector_state() {
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let (mid, bytes) = mk_memory_with_embedding(axis_vec(6));
        TypedProjection::apply(
            &mut proj,
            Lsn(0),
            &mk_remember(&mid.0.to_string(), bytes),
        )
        .unwrap();

        let before = (
            index.read().len(),
            index.read().deleted_count(),
            proj.vector_id_for(&mid.0.to_string()),
        );

        TypedProjection::apply(
            &mut proj,
            Lsn(1),
            &IntentPayload::Anchor {
                user_id: "alice".into(),
                memory_id: mid.0.to_string(),
                importance: 0.99,
                schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
            },
        )
        .unwrap();

        let after = (
            index.read().len(),
            index.read().deleted_count(),
            proj.vector_id_for(&mid.0.to_string()),
        );
        assert_eq!(before, after, "Anchor must not touch the vector index");
        assert_eq!(proj.checkpoint(), Some(Lsn(1)));
    }

    #[test]
    fn memory_without_primary_embedding_is_skipped_but_checkpoint_advances() {
        let (index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let id = MemoryId(Uuid::new_v4());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "no embedding".to_string(),
            embeddings: None,
            ..Experience::default()
        };
        let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        let bytes =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();

        TypedProjection::apply(
            &mut proj,
            Lsn(0),
            &mk_remember(&id.0.to_string(), bytes),
        )
        .unwrap();

        // No vector landed in Vamana, no mapping, but the checkpoint
        // moved forward so replay doesn't loop on this frame.
        assert_eq!(index.read().len(), 0);
        assert!(proj.vector_id_for(&id.0.to_string()).is_none());
        assert_eq!(proj.checkpoint(), Some(Lsn(0)));
    }

    #[test]
    fn restart_resumes_from_persisted_checkpoint() {
        let (log_path, ckpt_path) = tmp_paths("restart");

        // Write 6 records to the log; projection applies the first 3,
        // persists checkpoint, then "crashes" (gets dropped).
        let mut memory_ids: Vec<String> = Vec::new();
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for i in 0..6 {
                let (mid, bytes) = mk_memory_with_embedding(axis_vec(i));
                memory_ids.push(mid.0.to_string());
                crate::intent_log::payload::append(
                    &mut log,
                    &mk_remember(&mid.0.to_string(), bytes),
                )
                .unwrap();
            }
            log.sync().unwrap();
        }

        // First boot: apply only LSN 0..=2 via the static Projection arm.
        // Note: this run does NOT call persist_checkpoint via the dyn
        // arm, so no id_map side data is persisted — only the LSN.
        {
            let log = IntentLog::open(&log_path).unwrap();
            let config = VamanaConfig {
                dimension: TEST_DIM,
                max_degree: 8,
                search_list_size: 12,
                alpha: 1.2,
                use_mmap: false,
                distance_metric: DistanceMetric::Cosine,
            };
            let index = Arc::new(RwLock::new(VamanaIndex::new(config).unwrap()));
            let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                index,
                ckpt,
            );
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

        // Second boot: brand-new index + projection. The persisted
        // checkpoint is recovered from disk; replay re-applies the
        // first 3 LSNs (idempotent) and then 3, 4, 5 fresh — but
        // *because the in-memory state was lost*, every LSN must be
        // re-applied. The checkpoint behaviour matches the SqliteProjection
        // template: replay resumes at checkpoint+1, so only 3 new
        // applies happen. The fact that the Vamana side starts empty
        // means those 3 are the ONLY vectors in the index after replay.
        let log = IntentLog::open(&log_path).unwrap();
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let mut proj = VamanaProjection::new(
            VamanaEmbeddingKind::TextPrimary,
            index.clone(),
            ckpt,
        );
        assert_eq!(proj.checkpoint(), Some(Lsn(2)));
        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 3, "replay should only touch lsn 3,4,5");
        assert_eq!(index.read().len(), 3);
        // The three mapped ids are the last three the seed loop wrote.
        for mid in memory_ids.iter().skip(3) {
            assert!(
                proj.vector_id_for(mid).is_some(),
                "missing mapping for {mid} after restart-replay"
            );
        }
        // And the first three (already applied on the previous boot)
        // are NOT in the new projection's map — they live only in the
        // SQLite projection / RocksDB. This matches the intent: the
        // Vamana projection is rebuildable from the log starting at any
        // checkpoint, and the operator is expected to wipe-and-replay
        // from `Lsn::ZERO` if they want full reconstruction.
        for mid in memory_ids.iter().take(3) {
            assert!(
                proj.vector_id_for(mid).is_none(),
                "checkpointed lsn should not be re-applied: {mid}"
            );
        }
    }

    #[test]
    fn lazy_projection_materialises_index_on_first_embedded_memory() {
        let (_log_path, ckpt_path) = tmp_paths("lazy");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let bootstrap = VamanaProjectionBootstrap {
            storage_path: None,
            config_template: test_config(),
        };
        let mut proj = VamanaProjection::lazy(
            VamanaEmbeddingKind::TextPrimary,
            bootstrap,
            ckpt,
        );

        // Before any embedded memory arrives, the index is still
        // pending — `.index()` reports `None`.
        assert!(proj.index().is_none());

        let (mid, bytes) = mk_memory_with_embedding(axis_vec(0));
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid.0.to_string(), bytes))
            .unwrap();

        // After the first apply with an embedding, the index exists and
        // holds exactly one vector.
        let idx = proj.index().expect("index materialised");
        assert_eq!(idx.read().len(), 1);
        assert_eq!(idx.read().config_dimension(), TEST_DIM);
    }

    #[test]
    fn lazy_projection_skips_embeddingless_memory_without_materialising() {
        let (_log_path, ckpt_path) = tmp_paths("lazy_skip");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let bootstrap = VamanaProjectionBootstrap {
            storage_path: None,
            config_template: test_config(),
        };
        let mut proj = VamanaProjection::lazy(
            VamanaEmbeddingKind::TextPrimary,
            bootstrap,
            ckpt,
        );

        // A memory with no embedding must not force the index to
        // materialise (we don't know the dim yet). Checkpoint still
        // advances.
        let id = MemoryId(Uuid::new_v4());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "no embedding".into(),
            embeddings: None,
            ..Experience::default()
        };
        let m = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        let bytes = bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap();

        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&id.0.to_string(), bytes))
            .unwrap();
        assert!(proj.index().is_none());
        assert_eq!(proj.checkpoint(), Some(Lsn(0)));
    }

    #[test]
    fn dimension_mismatch_is_a_hard_error() {
        let (_index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let id = MemoryId(Uuid::new_v4());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "wrong dim".to_string(),
            embeddings: Some(vec![1.0f32; TEST_DIM + 4]),
            ..Experience::default()
        };
        let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        let bytes =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();

        let err = proj
            .apply_typed(Lsn(0), &mk_remember(&id.0.to_string(), bytes))
            .unwrap_err();
        match err {
            VamanaProjectionError::DimensionMismatch {
                memory_dim,
                index_dim,
                ..
            } => {
                assert_eq!(memory_dim, TEST_DIM + 4);
                assert_eq!(index_dim, TEST_DIM);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // ----------------------------------------------------------------
    // Multi-embedding-kind tests (Deliverable A)
    // ----------------------------------------------------------------

    #[test]
    fn each_kind_indexes_only_its_own_embedding_field() {
        let (_log_path, ckpt_path) = tmp_paths("multi_live");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));

        // Build one projection per kind, all sharing the same
        // checkpoint store — exactly how `get_user_journaled_writer`
        // wires them up.
        let mk_proj = |kind: VamanaEmbeddingKind| {
            let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
            let proj = VamanaProjection::new(kind, index.clone(), ckpt.clone());
            (index, proj)
        };
        let (idx_primary, mut p_primary) = mk_proj(VamanaEmbeddingKind::TextPrimary);
        let (idx_secondary, mut p_secondary) =
            mk_proj(VamanaEmbeddingKind::TextSecondary);
        let (idx_image, mut p_image) = mk_proj(VamanaEmbeddingKind::Image);
        let (idx_audio, mut p_audio) = mk_proj(VamanaEmbeddingKind::Audio);
        let (idx_video, mut p_video) = mk_proj(VamanaEmbeddingKind::Video);

        // Memory A: primary + secondary + image only.
        let (_, m_a) = mk_memory_multi(
            Some(axis_vec(0)),
            Some(axis_vec(1)),
            Some(axis_vec(2)),
            None,
            None,
        );
        // Memory B: primary + audio only.
        let (_, m_b) = mk_memory_multi(
            Some(axis_vec(3)),
            None,
            None,
            Some(axis_vec(4)),
            None,
        );
        // Memory C: primary + secondary + video.
        let (_, m_c) = mk_memory_multi(
            Some(axis_vec(5)),
            Some(axis_vec(6)),
            None,
            None,
            Some(axis_vec(7)),
        );

        // Drive each projection through all three memories.
        let id_a = uuid::Uuid::new_v4().to_string();
        let id_b = uuid::Uuid::new_v4().to_string();
        let id_c = uuid::Uuid::new_v4().to_string();
        for (lsn, (mid, bytes)) in [
            (0u64, (&id_a, m_a.clone())),
            (1u64, (&id_b, m_b.clone())),
            (2u64, (&id_c, m_c.clone())),
        ] {
            let pay = mk_remember(mid, bytes);
            for p in [
                &mut p_primary,
                &mut p_secondary,
                &mut p_image,
                &mut p_audio,
                &mut p_video,
            ] {
                TypedProjection::apply(p, Lsn(lsn), &pay).unwrap();
            }
        }

        // text-primary saw 3, secondary 2 (a + c), image 1 (a), audio 1 (b),
        // video 1 (c). The "skipped because no embedding for this kind"
        // branch is exercised end-to-end here — none of the projections
        // erred, and their counts match the per-kind presence pattern.
        assert_eq!(idx_primary.read().len(), 3, "primary indexes all three");
        assert_eq!(idx_secondary.read().len(), 2, "secondary indexes A + C");
        assert_eq!(idx_image.read().len(), 1, "image indexes A only");
        assert_eq!(idx_audio.read().len(), 1, "audio indexes B only");
        assert_eq!(idx_video.read().len(), 1, "video indexes C only");

        assert_eq!(p_primary.tracked_memory_count(), 3);
        assert_eq!(p_secondary.tracked_memory_count(), 2);
        assert_eq!(p_image.tracked_memory_count(), 1);
        assert_eq!(p_audio.tracked_memory_count(), 1);
        assert_eq!(p_video.tracked_memory_count(), 1);
    }

    #[test]
    fn replay_from_scratch_per_kind_matches_live_counts() {
        // Same three memories as the live test, but written to the log
        // first then replayed into fresh projections. Each projection
        // independently catches up and ends with the per-kind count it
        // would have on the live path.
        let (log_path, ckpt_path) = tmp_paths("multi_replay");

        let (id_a, m_a) = mk_memory_multi(
            Some(axis_vec(0)),
            Some(axis_vec(1)),
            Some(axis_vec(2)),
            None,
            None,
        );
        let (id_b, m_b) = mk_memory_multi(
            Some(axis_vec(3)),
            None,
            None,
            Some(axis_vec(4)),
            None,
        );
        let (id_c, m_c) = mk_memory_multi(
            Some(axis_vec(5)),
            Some(axis_vec(6)),
            None,
            None,
            Some(axis_vec(7)),
        );
        {
            let mut log = IntentLog::open(&log_path).unwrap();
            for (mid, bytes) in [(&id_a, m_a), (&id_b, m_b), (&id_c, m_c)] {
                crate::intent_log::payload::append(
                    &mut log,
                    &mk_remember(&mid.0.to_string(), bytes),
                )
                .unwrap();
            }
            log.sync().unwrap();
        }

        let expected_counts = [
            (VamanaEmbeddingKind::TextPrimary, 3usize),
            (VamanaEmbeddingKind::TextSecondary, 2),
            (VamanaEmbeddingKind::Image, 1),
            (VamanaEmbeddingKind::Audio, 1),
            (VamanaEmbeddingKind::Video, 1),
        ];

        for (kind, expected) in expected_counts {
            let log = IntentLog::open(&log_path).unwrap();
            let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
            let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
            // Per-kind fresh checkpoint store entry — we use one shared
            // file across kinds so each kind's name keys its own row.
            let mut proj = VamanaProjection::new(kind, index.clone(), ckpt);
            let applied = replay(&log, &mut proj, Some(2)).unwrap();
            assert_eq!(
                applied, 3,
                "{} replay must touch every frame (skipped or applied)",
                kind.projection_name()
            );
            assert_eq!(
                index.read().len(),
                expected,
                "{}: expected {} vectors after replay",
                kind.projection_name(),
                expected,
            );
            assert_eq!(
                proj.tracked_memory_count(),
                expected,
                "{}: expected {} tracked ids after replay",
                kind.projection_name(),
                expected,
            );
        }
    }

    #[test]
    fn projection_names_are_stable_and_unique() {
        // The CheckpointStore is keyed on these strings. Any rename
        // would be a silent on-disk migration — pin them with an
        // explicit assertion so accidental renames fail loud.
        let names: Vec<_> = VamanaEmbeddingKind::all()
            .iter()
            .map(|k| k.projection_name())
            .collect();
        assert_eq!(
            names,
            vec![
                "vamana-text-primary",
                "vamana-text-secondary",
                "vamana-image",
                "vamana-audio",
                "vamana-video",
            ]
        );
        // All five must be distinct strings so the checkpoint store
        // doesn't accidentally share an LSN slot.
        let uniq: std::collections::HashSet<_> = names.iter().copied().collect();
        assert_eq!(uniq.len(), names.len());
    }

    // ----------------------------------------------------------------
    // Side-data round-trip tests (Deliverable B)
    // ----------------------------------------------------------------

    #[test]
    fn id_map_survives_reopen_via_side_data() {
        // Live-write three memories, drop the projection (without
        // touching the log on reopen — the side-data path is the only
        // thing letting the map come back). The fresh projection MUST
        // see the same memory_id → vector_id mappings.
        let (_log_path, ckpt_path) = tmp_paths("idmap_reopen");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));

        let (id_a, b_a) = mk_memory_with_embedding(axis_vec(0));
        let (id_b, b_b) = mk_memory_with_embedding(axis_vec(2));
        let (id_c, b_c) = mk_memory_with_embedding(axis_vec(5));

        let saved_ids = {
            let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                index,
                ckpt.clone(),
            );
            TypedProjection::apply(
                &mut proj,
                Lsn(0),
                &mk_remember(&id_a.0.to_string(), b_a),
            )
            .unwrap();
            TypedProjection::apply(
                &mut proj,
                Lsn(1),
                &mk_remember(&id_b.0.to_string(), b_b),
            )
            .unwrap();
            TypedProjection::apply(
                &mut proj,
                Lsn(2),
                &mk_remember(&id_c.0.to_string(), b_c),
            )
            .unwrap();
            // Capture the assigned vector ids before the projection drops.
            vec![
                (
                    id_a.0.to_string(),
                    proj.vector_id_for(&id_a.0.to_string()).unwrap(),
                ),
                (
                    id_b.0.to_string(),
                    proj.vector_id_for(&id_b.0.to_string()).unwrap(),
                ),
                (
                    id_c.0.to_string(),
                    proj.vector_id_for(&id_c.0.to_string()).unwrap(),
                ),
            ]
        };

        // Re-open. Brand-new index + projection, same checkpoint store.
        // No replay runs here — the side data must carry the map back.
        let index2 = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt2 = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj2 = VamanaProjection::new(
            VamanaEmbeddingKind::TextPrimary,
            index2,
            ckpt2,
        );
        assert_eq!(proj2.checkpoint(), Some(Lsn(2)));
        assert_eq!(proj2.tracked_memory_count(), 3);
        for (mid, expected_vid) in saved_ids {
            assert_eq!(
                proj2.vector_id_for(&mid),
                Some(expected_vid),
                "id_map side data must round-trip vector ids exactly"
            );
        }
    }

    #[test]
    fn stale_side_data_is_discarded_on_open() {
        // Live-write three memories at LSN 0..=2 (which stamps side
        // data at LSN 2). Then hand-advance the projection's checkpoint
        // to LSN 9 without writing matching side data. On the next
        // open, the side-data stamp (2) no longer matches the
        // checkpoint (9), so the projection MUST discard the map and
        // start empty — the brief's defence-in-depth contract.
        let (_log_path, ckpt_path) = tmp_paths("stale_side_data");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));

        {
            let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                index,
                ckpt.clone(),
            );
            let (id_a, b_a) = mk_memory_with_embedding(axis_vec(0));
            let (id_b, b_b) = mk_memory_with_embedding(axis_vec(2));
            let (id_c, b_c) = mk_memory_with_embedding(axis_vec(5));
            TypedProjection::apply(
                &mut proj,
                Lsn(0),
                &mk_remember(&id_a.0.to_string(), b_a),
            )
            .unwrap();
            TypedProjection::apply(
                &mut proj,
                Lsn(1),
                &mk_remember(&id_b.0.to_string(), b_b),
            )
            .unwrap();
            TypedProjection::apply(
                &mut proj,
                Lsn(2),
                &mk_remember(&id_c.0.to_string(), b_c),
            )
            .unwrap();
            // proj drops; checkpoint store now has lsn=2 + side data
            // stamped at lsn=2.
        }

        // Drop the original ckpt handle so the next reopen sees the
        // on-disk state cleanly.
        drop(ckpt);

        // Hand-advance the checkpoint to LSN 9 *without* updating side
        // data. Simulates a torn-write / aborted-shutdown where the
        // checkpoint ran ahead of the snapshot.
        {
            let mut store = CheckpointStore::open(&ckpt_path).unwrap();
            store
                .set(VamanaEmbeddingKind::TextPrimary.projection_name(), Lsn(9))
                .unwrap();
            store.sync().unwrap();
        }

        // Boot 2: open the projection. Side-data stamp (2) ≠ checkpoint
        // (9), so the discard branch fires. Map starts empty;
        // checkpoint comes from disk.
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj = VamanaProjection::new(
            VamanaEmbeddingKind::TextPrimary,
            index,
            ckpt,
        );
        assert_eq!(proj.checkpoint(), Some(Lsn(9)));
        assert_eq!(
            proj.tracked_memory_count(),
            0,
            "stale side data must be discarded — defence in depth"
        );
    }
}
