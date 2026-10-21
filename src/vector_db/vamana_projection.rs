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
//! time — persists on a cadence (see below), so a crash between live
//! writes resumes at most [`ID_MAP_PERSIST_EVERY`] records behind the
//! head, never from the start of the log; replay re-applies that tail
//! idempotently.
//!
//! ## `memory_id → vector_id` persistence
//!
//! The in-memory map of memory_id → vector_id is persisted to the
//! checkpoint store as side data. On open, the projection loads its side
//! data and uses it instead of rebuilding from the log — but only if the
//! checkpoint provably describes that snapshot: either the snapshot's
//! stamped LSN equals the checkpoint, or the snapshot is the one this
//! projection last vouched for (`{name}#id-map-stamp`) and the checkpoint
//! is at or past it. Any other combination (corrupt side data, stale
//! snapshot, a checkpoint that ran ahead of its snapshot) discards the
//! side data and falls back to replay. The log is the canonical source of
//! truth; side data is opportunistic acceleration that never outvotes it.
//!
//! A snapshot bincodes the WHOLE map, so it is written only when the map
//! changed: every [`ID_MAP_PERSIST_EVERY`] live applies while it is dirty,
//! whenever the checkpoint is persisted, and on drop. The snapshot is made
//! durable BEFORE the checkpoint record that refers to it, so the
//! checkpoint never names a map that is not on disk. While the map is
//! clean (no embedding for this kind, an anchor, a stale replay, a forget
//! of an unknown id) a live apply advances the checkpoint with one small
//! buffered record and no side-data write or fsync; the vouch record is
//! what lets the load check accept a snapshot older than that checkpoint.

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::embeddings::space::SpaceId;
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
            VamanaEmbeddingKind::TextSecondary => memory.experience.embeddings_secondary.clone(),
            VamanaEmbeddingKind::Image => memory.experience.image_embeddings.clone(),
            VamanaEmbeddingKind::Audio => memory.experience.audio_embeddings.clone(),
            VamanaEmbeddingKind::Video => memory.experience.video_embeddings.clone(),
        }
    }

    /// The space the memory records for this kind's vector. Only the text kinds carry one; an
    /// absent space is a vector written before spaces were recorded.
    pub fn recorded_space(self, memory: &Memory) -> Option<SpaceId> {
        match self {
            VamanaEmbeddingKind::TextPrimary => memory.experience.embedding_space.clone(),
            VamanaEmbeddingKind::TextSecondary => {
                memory.experience.embedding_space_secondary.clone()
            }
            VamanaEmbeddingKind::Image
            | VamanaEmbeddingKind::Audio
            | VamanaEmbeddingKind::Video => None,
        }
    }

    /// Checkpoint-store name under which this kind records the stamp of the
    /// id-map snapshot it vouches for: the map has not changed since that
    /// snapshot up to the projection's persisted checkpoint. Stable on disk,
    /// like [`projection_name`](Self::projection_name).
    fn id_map_vouch_name(self) -> &'static str {
        match self {
            VamanaEmbeddingKind::TextPrimary => "vamana-text-primary#id-map-stamp",
            VamanaEmbeddingKind::TextSecondary => "vamana-text-secondary#id-map-stamp",
            VamanaEmbeddingKind::Image => "vamana-image#id-map-stamp",
            VamanaEmbeddingKind::Audio => "vamana-audio#id-map-stamp",
            VamanaEmbeddingKind::Video => "vamana-video#id-map-stamp",
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

/// Live applies between id-map snapshots while the map is dirty: each
/// snapshot is O(map) bytes plus two fsyncs, and 64 bounds the replay tail
/// after a crash to 64 idempotent records.
pub const ID_MAP_PERSIST_EVERY: u32 = 64;

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
    /// The embedding space this projection's index holds, bound from the first vector that
    /// records one. A vector in another space, or of another width, is skipped: one index holds
    /// one space, and a log spanning an embedder change holds several.
    space: Option<SpaceId>,
    /// Vectors skipped for being in another space or width than the index's.
    skipped_other_space: u64,

    /// `memories` differs from the latest durable side-data snapshot, so the
    /// checkpoint cannot advance on disk until a new snapshot is written.
    map_dirty: bool,
    /// Live applies since the map became dirty without a snapshot.
    dirty_applies: u32,
    /// Checkpoint last written to the store, so a persist with nothing new
    /// writes nothing.
    persisted_checkpoint: Option<Lsn>,
}

/// What [`load_checkpoint_and_map`] did with the persisted id map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapLoad {
    /// The snapshot was accepted: the in-memory map equals it.
    Restored,
    /// No snapshot exists: the empty map is exactly what a reload yields.
    Absent,
    /// A snapshot exists but was refused: the in-memory map differs from it.
    Discarded,
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
        let (checkpoint, memories, load) = load_checkpoint_and_map(kind, &checkpoint_store);
        let restored = memories.len();
        let memories = reconcile_map_with_index(kind, memories, &index);
        let map_dirty = load == MapLoad::Discarded || memories.len() != restored;
        Self {
            kind,
            state: IndexState::Ready(index),
            checkpoint_store,
            memories,
            checkpoint,
            space: None,
            skipped_other_space: 0,
            map_dirty,
            dirty_applies: 0,
            persisted_checkpoint: checkpoint,
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
        let (checkpoint, memories, load) = load_checkpoint_and_map(kind, &checkpoint_store);
        Self {
            kind,
            state: IndexState::Pending(bootstrap),
            checkpoint_store,
            memories,
            checkpoint,
            space: None,
            skipped_other_space: 0,
            map_dirty: load == MapLoad::Discarded,
            dirty_applies: 0,
            persisted_checkpoint: checkpoint,
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

    /// Bind the index to `space` before any vector arrives: the tenant's active space, known
    /// when the projection opens. A log replayed from the start then builds the index in that
    /// space and skips every vector of another, instead of taking the width of whatever the log
    /// holds first — on a tenant whose log spans an embedder change, the old one.
    pub fn bind_space(mut self, space: Option<SpaceId>) -> Self {
        if let Some(space) = space.filter(SpaceId::is_recorded) {
            self.space = Some(space);
        }
        self
    }

    /// The space this projection's index is bound to, once a vector recorded one.
    pub fn space(&self) -> Option<&SpaceId> {
        self.space.as_ref()
    }

    /// Vectors skipped for being in another space or width than the index's.
    pub fn skipped_other_space(&self) -> u64 {
        self.skipped_other_space
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

    /// Inverse of [`Self::vector_id_for`]. Returns the `memory_id` that
    /// maps to `vector_id`, or `None` if no such mapping exists.
    ///
    /// The forward map (`memory_id → vector_id`) is populated by
    /// `apply()` / `replay()` and persisted via the `CheckpointStore`
    /// side data; the inverse is materialised on demand by walking the
    /// forward map. The walk is O(n) in the number of live (non-
    /// forgotten) memory-id mappings the projection currently tracks —
    /// search-result resolution is already top-k bounded, so the
    /// per-call cost stays in proportion to the result set, not the
    /// corpus. Soft-deleted entries are dropped from the forward map at
    /// `Forget` time, so this accessor will never return a `memory_id`
    /// for a vector that has been logically deleted.
    pub fn memory_id_for(&self, vector_id: u32) -> Option<String> {
        self.memories
            .iter()
            .find(|(_, entry)| entry.vector_id == vector_id)
            .map(|(memory_id, _)| memory_id.clone())
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
                        // DEBUG, not WARN. Image/Audio/Video embeddings are never
                        // written anywhere in the tree — those three fields only
                        // ever get `None` — so their projections emit this for
                        // EVERY memory, forever. Measured 2026-08-25: 35,015 of
                        // 131,912 lines, 26.5% of everything veld had ever logged,
                        // were this one message. A warning that fires on every
                        // normal operation is not a warning; it buries the ones
                        // that mean something. The skip itself is correct and the
                        // checkpoint must still advance, so the behaviour is
                        // unchanged — only the volume.
                        tracing::debug!(
                            projection = self.kind.projection_name(),
                            memory_id = %memory_id,
                            lsn = lsn.0,
                            "vamana projection: skipping memory without embedding for this kind (no-op apply; checkpoint advances)",
                        );
                        self.checkpoint = Some(self.bumped_checkpoint(lsn));
                        return Ok(());
                    }
                };

                // A bound space decides the width before the index exists, so the first vector
                // off the log cannot set it to another space's.
                if self
                    .space
                    .as_ref()
                    .is_some_and(|bound| bound.dimension() != embedding.len())
                {
                    if let (Some(existing), Some(index)) =
                        (self.memories.remove(memory_id), self.state.as_ready())
                    {
                        index.read().mark_deleted(existing.vector_id);
                    }
                    self.skipped_other_space += 1;
                    if self.skipped_other_space.is_power_of_two() {
                        tracing::warn!(
                            projection = self.kind.projection_name(),
                            memory_id = %memory_id,
                            memory_dim = embedding.len(),
                            index_space = ?self.space,
                            skipped = self.skipped_other_space,
                            "vamana projection: skipping a vector of another width than the bound space",
                        );
                    }
                    self.checkpoint = Some(self.bumped_checkpoint(lsn));
                    return Ok(());
                }

                // Materialise the index (no-op if eager / already
                // ready). Lazy projections learn their dim here from
                // the first embedded memory off the log.
                let index_arc = self.ensure_ready(embedding.len())?.clone();

                // One index holds one space. A vector of another width would corrupt the graph
                // (see `VamanaIndex::add_vector`'s own guard), and one recording another space
                // would be compared by a cosine that means nothing. Either is skipped, not an
                // error: a log spanning an embedder change holds several spaces, and replay
                // must pass them. A skipped memory loses any vector it had here, since its
                // current one is no longer in this space.
                let index_dim = index_arc.read().config_dimension();
                let recorded = self.kind.recorded_space(&memory);
                let other_space =
                    matches!((&self.space, &recorded), (Some(bound), Some(s)) if bound != s);
                if embedding.len() != index_dim || other_space {
                    if let Some(existing) = self.memories.remove(memory_id) {
                        index_arc.read().mark_deleted(existing.vector_id);
                    }
                    self.skipped_other_space += 1;
                    if self.skipped_other_space.is_power_of_two() {
                        tracing::warn!(
                            projection = self.kind.projection_name(),
                            memory_id = %memory_id,
                            memory_dim = embedding.len(),
                            index_dim,
                            memory_space = ?recorded,
                            index_space = ?self.space,
                            skipped = self.skipped_other_space,
                            "vamana projection: skipping a vector in another space than the index's",
                        );
                    }
                    self.checkpoint = Some(self.bumped_checkpoint(lsn));
                    return Ok(());
                }
                if self.space.is_none() {
                    self.space = recorded.filter(SpaceId::is_recorded);
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
                self.map_dirty = true;
            }
            IntentPayload::Forget { memory_id, .. } => {
                // Already-forgotten memories are a no-op (idempotent).
                // We still advance the checkpoint so the driver
                // progresses past this frame.
                if let Some(existing) = self.memories.remove(memory_id) {
                    self.map_dirty = true;
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
            IntentPayload::Anchor { .. } | IntentPayload::SetAnchored { .. } => {
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

    /// Durably persist the checkpoint, snapshotting the id map first when it
    /// changed since the last snapshot.
    fn persist(&mut self) -> Result<(), VamanaProjectionError> {
        let Some(lsn) = self.checkpoint else {
            return Ok(());
        };
        let name = self.kind.projection_name();
        if self.map_dirty {
            // Encoded with `bincode::config::standard()` to match every
            // other on-disk encoding in this codebase.
            let encoded =
                bincode::serde::encode_to_vec(&self.memories, bincode::config::standard())
                    .map_err(|e| VamanaProjectionError::Vamana(format!("encode id_map: {e}")))?;
            let mut store = self.checkpoint_store.lock();
            store.set_side_data_at(name, SIDE_DATA_KEY_ID_MAP, lsn, &encoded)?;
            store.sync_side_data()?;
            store.set(name, lsn)?;
            store.sync_checkpoints()?;
            self.map_dirty = false;
        } else if self.persisted_checkpoint != Some(lsn) {
            let mut store = self.checkpoint_store.lock();
            advance_clean_checkpoint(&mut store, self.kind, lsn)?;
            store.sync_checkpoints()?;
        }
        self.persisted_checkpoint = Some(lsn);
        self.dirty_applies = 0;
        Ok(())
    }

    /// Live-write persistence. A dirty map is snapshotted every
    /// [`ID_MAP_PERSIST_EVERY`] applies and the checkpoint waits for it; a
    /// clean map advances the checkpoint with one buffered record and no
    /// fsync, since replay recovers an unsynced tail idempotently.
    fn persist_live(&mut self) -> Result<(), VamanaProjectionError> {
        let Some(lsn) = self.checkpoint else {
            return Ok(());
        };
        if self.map_dirty {
            self.dirty_applies += 1;
            if self.dirty_applies >= ID_MAP_PERSIST_EVERY {
                self.persist()?;
            }
            return Ok(());
        }
        if self.persisted_checkpoint != Some(lsn) {
            let mut store = self.checkpoint_store.lock();
            advance_clean_checkpoint(&mut store, self.kind, lsn)?;
            self.persisted_checkpoint = Some(lsn);
        }
        Ok(())
    }
}

/// Write `lsn` as `kind`'s checkpoint while its id map equals the latest
/// snapshot on disk. The first such advance after a snapshot records that
/// snapshot's stamp as vouched for, which is what lets the load check accept
/// a snapshot older than the checkpoint. The vouch record precedes the
/// checkpoint record in the same append-only file, so a torn tail can drop
/// the checkpoint but never keep it without its vouch.
fn advance_clean_checkpoint(
    store: &mut CheckpointStore,
    kind: VamanaEmbeddingKind,
    lsn: Lsn,
) -> Result<(), CheckpointStoreError> {
    if let Some(stamp) = store.side_data_checkpoint(kind.projection_name(), SIDE_DATA_KEY_ID_MAP) {
        if store.get(kind.id_map_vouch_name()) != Some(stamp) {
            store.set(kind.id_map_vouch_name(), stamp)?;
        }
    }
    store.set(kind.projection_name(), lsn)
}

/// Flush what the cadence has not persisted yet, so a graceful drop does not
/// leave the next boot a replay tail.
impl Drop for VamanaProjection {
    fn drop(&mut self) {
        if let Err(e) = self.persist() {
            tracing::warn!(
                projection = self.kind.projection_name(),
                error = %e,
                "vamana projection: final checkpoint persist failed; replay will re-apply the tail",
            );
        }
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
) -> (Option<Lsn>, HashMap<String, MemoryEntry>, MapLoad) {
    let store = checkpoint_store.lock();
    let checkpoint = store.get(kind.projection_name());
    let vouched = store.get(kind.id_map_vouch_name());
    let (map, load) = match (
        checkpoint,
        store.get_side_data(kind.projection_name(), SIDE_DATA_KEY_ID_MAP),
        store.side_data_checkpoint(kind.projection_name(), SIDE_DATA_KEY_ID_MAP),
    ) {
        // Side data present AND the checkpoint describes it: its stamp is
        // the checkpoint, or it is the snapshot the projection vouched for
        // when it last advanced the checkpoint over an unchanged map.
        (Some(ckpt), Some(bytes), Some(stamp))
            if stamp == ckpt || (vouched == Some(stamp) && stamp <= ckpt) =>
        {
            match bincode::serde::decode_from_slice::<HashMap<String, MemoryEntry>, _>(
                bytes,
                bincode::config::standard(),
            ) {
                Ok((map, _)) => (map, MapLoad::Restored),
                Err(e) => {
                    tracing::warn!(
                        projection = kind.projection_name(),
                        error = %e,
                        "vamana projection: id_map side data failed to decode; falling back to replay-rebuild",
                    );
                    (HashMap::new(), MapLoad::Discarded)
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
            (HashMap::new(), MapLoad::Discarded)
        }
        // No side data at all — first boot of this projection, or it
        // has only ever persisted checkpoints (older binary). Replay
        // will populate.
        _ => (HashMap::new(), MapLoad::Absent),
    };
    (checkpoint, map, load)
}

/// A RESTORED ID MAP IS ONLY VALID FOR THE INDEX IT WAS BUILT AGAINST.
///
/// `load_checkpoint_and_map` validates the side data's stamp against the projection's
/// checkpoint LSN, which proves the map is not STALE. It cannot prove the map still
/// DESCRIBES anything, because the index is a separate artifact with its own lifetime.
///
/// The gap that leaves is not a missing lookup, it is a WRONG one.
/// `VamanaIndex::add_vector` allocates ids from that index's own counter starting at 0.
/// Restore a map naming vector ids 0,1,2 alongside a freshly-built empty index and replay
/// hands 0,1,2 straight back out — to different memories. Measured 2026-08-20: six
/// memory_ids sharing three vector_ids, so `vector_id_for()` on a pre-checkpoint memory
/// returned a vector belonging to someone else, and `memory_id_for()` — which walks this
/// same forward map — was ambiguous in the other direction. For a memory store that is
/// returning the wrong memory's content.
///
/// So: every id the map names must exist in the index. Ids are dense from 0, so the test
/// is `max(vector_id) < index.len()`. On failure the whole map is DISCARDED and replay
/// rebuilds it, which is the treatment the stale-stamp arm already applies one function
/// up — this extends an existing defence rather than inventing one.
///
/// Discarding is not a loss of the side-data feature. That feature buys "restore without
/// replaying the log", and it still does — whenever the index was persisted too, which is
/// the only case where the map meant anything. What it stops doing is answering confidently
/// from a map whose index is gone.
fn reconcile_map_with_index(
    kind: VamanaEmbeddingKind,
    memories: HashMap<String, MemoryEntry>,
    index: &Arc<RwLock<VamanaIndex>>,
) -> HashMap<String, MemoryEntry> {
    if memories.is_empty() {
        return memories;
    }
    let index_len = index.read().len();
    let highest = memories.values().map(|e| e.vector_id as usize).max();
    match highest {
        Some(max_vid) if max_vid >= index_len => {
            tracing::warn!(
                projection = kind.projection_name(),
                mapped_entries = memories.len(),
                highest_vector_id = max_vid,
                index_len,
                "vamana projection: restored id_map names vector ids the index does not have \
                 (index rebuilt without its vectors); DISCARDING the map and replaying. Keeping \
                 it would alias pre-checkpoint memories onto vectors replay is about to write.",
            );
            HashMap::new()
        }
        _ => memories,
    }
}

/// Decode a bincoded `Memory` snapshot off the wire. Goes through the store's
/// decoder, legacy arms included: a snapshot journaled before a positional
/// layout change is the old wire, exactly as a stored record would be.
fn decode_memory(bytes: &[u8]) -> Result<Memory, VamanaProjectionError> {
    crate::memory::storage::decode_memory_snapshot(bytes)
        .map_err(|e| VamanaProjectionError::MemoryDecode(e.to_string()))
}

/// Dyn-friendly arm: the [`JournaledWriter`] hands us typed payloads at
/// live-write time and the checkpoint is persisted on the live cadence
/// described in the module docs.
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
        self.persist_live()
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

    // Persisted with bincode, which cannot decode a type needing deserialize_any (an
    // internally tagged enum, say); tests/persisted_layout.rs cannot reach this private root.
    #[test]
    fn persisted_snapshot_is_bincode_decodable() {
        use serde_reflection::{Samples, Tracer, TracerConfig};
        let mut tracer = Tracer::new(TracerConfig::default().is_human_readable(false));
        tracer
            .trace_type::<std::collections::HashMap<String, MemoryEntry>>(&Samples::new())
            .unwrap_or_else(|e| panic!("tracing id-map snapshot: {e}"));
        tracer
            .registry()
            .unwrap_or_else(|e| panic!("incomplete trace of id-map snapshot: {e}"));
    }
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
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
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
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
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

        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid_a.0.to_string(), b_a)).unwrap();
        TypedProjection::apply(&mut proj, Lsn(1), &mk_remember(&mid_b.0.to_string(), b_b)).unwrap();
        TypedProjection::apply(&mut proj, Lsn(2), &mk_remember(&mid_c.0.to_string(), b_c)).unwrap();

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
        let mut proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index.clone(), ckpt);

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
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid.0.to_string(), bytes)).unwrap();
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
        assert_eq!(
            index.read().deleted_count(),
            1,
            "second forget must not add a delete"
        );

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
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid.0.to_string(), bytes)).unwrap();

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
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();

        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&id.0.to_string(), bytes)).unwrap();

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

        // First boot: apply only LSN 0..=2 via the static Projection arm, then
        // persist. THIS WRITES THE id_map SIDE DATA AS WELL AS THE LSN —
        // `Projection::persist_checkpoint` calls `persist()`, which snapshots
        // `self.memories` under SIDE_DATA_KEY_ID_MAP precisely "so a later open
        // can restore it without replaying the log". A previous comment here
        // claimed the opposite ("no id_map side data is persisted — only the
        // LSN"); that was true only before a7bb7c4 added the side-data path on
        // 2026-05-28, the same day this test landed in b398301, and it is what
        // made the final assertion below wrong for three months.
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
            let mut proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index, ckpt);
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
        let mut proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index.clone(), ckpt);
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
        // THE MAP COMES BACK; THE INDEX DOES NOT. This is the distinction the
        // test exists to pin, and it is the one an operator has to understand.
        //
        // The first three were applied on the previous boot and are NOT replayed
        // (see `applied == 3` above — the checkpoint did its job). Their
        // memory_id → vector_id mappings still resolve, because `persist()`
        // snapshotted them into CheckpointStore side data and the second boot
        // restored them; `id_map_survives_reopen_via_side_data` pins that path
        // directly. What is NOT restored is their VECTORS: the Vamana index
        // starts empty and only replay writes to it, which is why
        // `index.len() == 3` above rather than 6.
        //
        // So the projection can NAME a pre-checkpoint memory but cannot SEARCH
        // it. That asymmetry is the real operational fact: the operator must
        // wipe-and-replay from `Lsn::ZERO` for full reconstruction, and a
        // restored id map is not evidence that the index is whole.
        // Pre-checkpoint ids do NOT resolve — and the REASON matters, because this
        // assertion has now been right, wrong, and right again for three different
        // reasons in three months.
        //
        //   originally: "no id_map side data is persisted" — simply FALSE;
        //               persist_checkpoint() has snapshotted the map since a7bb7c4.
        //   2026-08-20: changed to is_some(), which was true and was the BUG —
        //               the map came back naming vectors this fresh index lacks.
        //   now:        the map is DISCARDED at open because the index cannot back
        //               it, so replay rebuilds from the checkpoint and only the
        //               post-checkpoint memories are mapped.
        //
        // Same assertion as the original, opposite justification. If it ever fails
        // again, read reconcile_map_with_index before touching this line.
        for mid in memory_ids.iter().take(3) {
            assert!(
                proj.vector_id_for(mid).is_none(),
                "pre-checkpoint {mid} must not resolve: its map entry named a vector the \
                 rebuilt index never had, so the map was discarded"
            );
        }
    }

    /// PINS A REAL DEFECT. `#[ignore]`d so it does not gate the suite, NOT
    /// because it is uncertain — it fails deterministically and the mechanism
    /// is understood. Run it with `cargo test -- --ignored`.
    ///
    /// THE COLLISION: `VamanaIndex::add_vector` allocates ids from that index's
    /// own `num_vectors` counter, starting at 0. On a restart-from-checkpoint
    /// the id map is restored from CheckpointStore side data while the index is
    /// rebuilt EMPTY, so replay hands out 0, 1, 2 again — to different memories.
    /// Six memory_ids then share three vector_ids, and every pre-checkpoint
    /// lookup resolves to a vector belonging to some other memory.
    ///
    /// This is worse than the "can name but cannot search" asymmetry the sibling
    /// test documents: the projection does not fail to answer, it answers
    /// WRONGLY, and `memory_id_for()` walks the forward map so its inverse is
    /// ambiguous too. For a memory store that is returning the wrong memory.
    ///
    /// Not repaired here because the fix is a design choice in someone else's
    /// subsystem — persist the index alongside the map, or make vector_id
    /// allocation survive a restart (a monotonic counter in side data), or
    /// discard the restored map when the index is empty. Discarding is the
    /// smallest and matches the module's stated "wipe-and-replay from Lsn::ZERO"
    /// posture, but it silently drops the side-data feature's whole benefit, so
    /// it is a call for the owner rather than a cleanup.
    #[tokio::test]
    async fn restored_id_map_is_discarded_when_the_index_cannot_back_it() {
        let (log_path, ckpt_path) = tmp_paths("idmap_collision");
        let mut memory_ids = Vec::new();
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
        {
            let log = IntentLog::open(&log_path).unwrap();
            let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
            let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
            let mut proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index, ckpt);
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
        }
        let log = IntentLog::open(&log_path).unwrap();
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let mut proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index.clone(), ckpt);
        replay(&log, &mut proj, None).unwrap();

        // Every live memory_id must own a DISTINCT vector_id. It does not.
        let mut seen: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        for mid in &memory_ids {
            if let Some(vid) = proj.vector_id_for(mid) {
                if let Some(prev) = seen.insert(vid, mid.clone()) {
                    panic!(
                        "vector_id {vid} is claimed by BOTH {prev} and {mid} — a restored \
                         pre-checkpoint mapping now points at a vector replay wrote for a \
                         different memory"
                    );
                }
            }
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
        let mut proj = VamanaProjection::lazy(VamanaEmbeddingKind::TextPrimary, bootstrap, ckpt);

        // Before any embedded memory arrives, the index is still
        // pending — `.index()` reports `None`.
        assert!(proj.index().is_none());

        let (mid, bytes) = mk_memory_with_embedding(axis_vec(0));
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid.0.to_string(), bytes)).unwrap();

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
        let mut proj = VamanaProjection::lazy(VamanaEmbeddingKind::TextPrimary, bootstrap, ckpt);

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

        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&id.0.to_string(), bytes)).unwrap();
        assert!(proj.index().is_none());
        assert_eq!(proj.checkpoint(), Some(Lsn(0)));
    }

    #[test]
    fn a_vector_of_another_width_is_skipped_and_replay_moves_on() {
        let (_index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let id = MemoryId(Uuid::new_v4());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "wrong dim".to_string(),
            embeddings: Some(vec![1.0f32; TEST_DIM + 4]),
            ..Experience::default()
        };
        let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();

        proj.apply_typed(Lsn(0), &mk_remember(&id.0.to_string(), bytes))
            .expect("another width is skipped, not an error");
        assert_eq!(proj.skipped_other_space(), 1);
        assert!(proj.vector_id_for(&id.0.to_string()).is_none());
        assert_eq!(proj.checkpoint(), Some(Lsn(0)));
    }

    /// A tenant whose log spans an embedder change, replayed from the start: the old width
    /// comes first and must not decide the index.
    #[test]
    fn a_bound_space_skips_an_older_width_that_comes_first() {
        let (_log_path, ckpt_path) = tmp_paths("bound_space");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let mut proj = VamanaProjection::lazy(
            VamanaEmbeddingKind::TextPrimary,
            VamanaProjectionBootstrap {
                storage_path: None,
                config_template: VamanaConfig {
                    dimension: 0,
                    ..VamanaConfig::default()
                },
            },
            ckpt,
        )
        .bind_space(Some(SpaceId::new("model-new", TEST_DIM)));

        let old = MemoryId(Uuid::new_v4());
        let experience = Experience {
            content: "written by the old embedder".to_string(),
            embeddings: Some(vec![1.0f32; TEST_DIM + 4]),
            ..Experience::default()
        };
        let m = Memory::new(old.clone(), experience, 0.5, None, None, None, None);
        let bytes = bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap();
        proj.apply_typed(Lsn(0), &mk_remember(&old.0.to_string(), bytes))
            .unwrap();
        assert!(
            proj.index().is_none(),
            "the old width did not build the index"
        );

        let new = MemoryId(Uuid::new_v4());
        let experience = Experience {
            content: "written by the new embedder".to_string(),
            embeddings: Some(vec![1.0f32; TEST_DIM]),
            ..Experience::default()
        };
        let m = Memory::new(new.clone(), experience, 0.5, None, None, None, None);
        let bytes = bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap();
        proj.apply_typed(Lsn(1), &mk_remember(&new.0.to_string(), bytes))
            .unwrap();
        assert!(proj.vector_id_for(&new.0.to_string()).is_some());
        assert_eq!(proj.skipped_other_space(), 1);
    }

    #[test]
    fn the_first_recorded_space_binds_the_index_and_another_is_skipped() {
        let (_index, _ckpt, mut proj, _log, _ckptp) = open_projection();
        let bound = SpaceId::new("model-a", TEST_DIM);
        let other = SpaceId::new("model-b", TEST_DIM);

        let first = MemoryId(Uuid::new_v4());
        let mut experience = Experience {
            content: "in model a".to_string(),
            ..Experience::default()
        };
        experience.set_primary_vector(vec![1.0f32; TEST_DIM], bound.clone());
        let m = Memory::new(first.clone(), experience, 0.5, None, None, None, None);
        let bytes = bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap();
        proj.apply_typed(Lsn(0), &mk_remember(&first.0.to_string(), bytes))
            .unwrap();
        assert_eq!(proj.space(), Some(&bound));

        // The same memory re-embedded into another space leaves this index.
        let mut moved = m.experience.clone();
        moved.set_primary_vector(vec![0.5f32; TEST_DIM], other);
        let m2 = Memory::new(first.clone(), moved, 0.5, None, None, None, None);
        let bytes = bincode::serde::encode_to_vec(&m2, bincode::config::standard()).unwrap();
        proj.apply_typed(Lsn(1), &mk_remember(&first.0.to_string(), bytes))
            .unwrap();
        assert_eq!(proj.skipped_other_space(), 1);
        assert!(proj.vector_id_for(&first.0.to_string()).is_none());
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
        let (idx_secondary, mut p_secondary) = mk_proj(VamanaEmbeddingKind::TextSecondary);
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
        let (_, m_b) = mk_memory_multi(Some(axis_vec(3)), None, None, Some(axis_vec(4)), None);
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
        let (id_b, m_b) = mk_memory_multi(Some(axis_vec(3)), None, None, Some(axis_vec(4)), None);
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
                applied,
                3,
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

        // The index is hoisted OUT of the block below so it outlives the first
        // projection. That is the whole point of the rewrite: the map is only
        // restorable when the index it describes survived too.
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let saved_ids = {
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                Arc::clone(&index),
                ckpt.clone(),
            );
            TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&id_a.0.to_string(), b_a))
                .unwrap();
            TypedProjection::apply(&mut proj, Lsn(1), &mk_remember(&id_b.0.to_string(), b_b))
                .unwrap();
            TypedProjection::apply(&mut proj, Lsn(2), &mk_remember(&id_c.0.to_string(), b_c))
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

        // Re-open with a NEW projection but the SURVIVING index, same checkpoint
        // store. No replay runs — the side data alone must carry the map back.
        let ckpt2 = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj2 =
            VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, Arc::clone(&index), ckpt2);
        assert_eq!(proj2.checkpoint(), Some(Lsn(2)));
        assert_eq!(proj2.tracked_memory_count(), 3);
        for (mid, expected_vid) in &saved_ids {
            assert_eq!(
                proj2.vector_id_for(mid),
                Some(*expected_vid),
                "id_map side data must round-trip vector ids exactly"
            );
        }

        // AND THE OTHER HALF, WHICH IS NEW AND IS THE POINT.
        //
        // This test used to re-open against a BRAND-NEW EMPTY index and assert the
        // ids came back anyway. They did — and that was the bug, not the feature.
        // Every VamanaProjection construction site in this crate builds
        // `VamanaIndex::new(...)`; nothing on the projection path calls
        // `VamanaIndex::load`. So "restored map + empty index" is not a test
        // convenience, it is the PRODUCTION shape, and it aliases pre-checkpoint
        // memories onto vectors replay is about to write (see
        // `restored_id_map_is_discarded_when_the_index_cannot_back_it`).
        //
        // The map is now discarded when the index cannot back it. That NARROWS the
        // side-data feature to the case where the index survived, which is the only
        // case where a vector_id meant anything.
        let empty_index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let ckpt3 = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj3 = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, empty_index, ckpt3);
        assert_eq!(
            proj3.checkpoint(),
            Some(Lsn(2)),
            "the CHECKPOINT is still restored — only the id map is refused"
        );
        assert_eq!(
            proj3.tracked_memory_count(),
            0,
            "an id map naming vectors an empty index does not have must be discarded, \
             not handed to a caller that will resolve it to the wrong memory"
        );
        for (mid, _) in &saved_ids {
            assert!(proj3.vector_id_for(mid).is_none());
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
            let mut proj =
                VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index, ckpt.clone());
            let (id_a, b_a) = mk_memory_with_embedding(axis_vec(0));
            let (id_b, b_b) = mk_memory_with_embedding(axis_vec(2));
            let (id_c, b_c) = mk_memory_with_embedding(axis_vec(5));
            TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&id_a.0.to_string(), b_a))
                .unwrap();
            TypedProjection::apply(&mut proj, Lsn(1), &mk_remember(&id_b.0.to_string(), b_b))
                .unwrap();
            TypedProjection::apply(&mut proj, Lsn(2), &mk_remember(&id_c.0.to_string(), b_c))
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
        let proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index, ckpt);
        assert_eq!(proj.checkpoint(), Some(Lsn(9)));
        assert_eq!(
            proj.tracked_memory_count(),
            0,
            "stale side data must be discarded — defence in depth"
        );
    }

    // ----------------------------------------------------------------
    // Reverse-lookup accessor (memory_id_for) tests
    // ----------------------------------------------------------------

    #[test]
    fn memory_id_for_returns_known_id() {
        let (_index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        let (mid, bytes) = mk_memory_with_embedding(axis_vec(0));
        let mid_str = mid.0.to_string();
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid_str, bytes)).unwrap();

        let vid = proj
            .vector_id_for(&mid_str)
            .expect("forward map must hold the freshly applied memory");
        assert_eq!(
            proj.memory_id_for(vid),
            Some(mid_str),
            "reverse lookup must round-trip through the live forward map",
        );
    }

    #[test]
    fn memory_id_for_unknown_returns_none() {
        let (_index, _ckpt, mut proj, _log, _ckptp) = open_projection();

        // Empty projection → reverse lookup is None.
        assert!(proj.memory_id_for(999_999).is_none());

        // Even after applying one memory, an unrelated vector_id stays
        // unmapped (the assigned id is a small u32; 999_999 will not
        // collide with the Vamana allocator's choice for the first
        // insert).
        let (mid, bytes) = mk_memory_with_embedding(axis_vec(1));
        TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&mid.0.to_string(), bytes)).unwrap();
        let assigned = proj.vector_id_for(&mid.0.to_string()).unwrap();
        let bogus = assigned.wrapping_add(1).max(999_999);
        assert!(
            proj.memory_id_for(bogus).is_none(),
            "vector_id outside the forward map must resolve to None",
        );
    }

    #[test]
    fn memory_id_for_round_trips_through_persist() {
        // Live-write three memories and capture their vector ids. Then
        // drop the projection, re-open against the same checkpoint
        // store (which restores the forward map from side data) and
        // assert that the reverse accessor still resolves each
        // vector_id back to the same memory_id.
        let (_log_path, ckpt_path) = tmp_paths("memory_id_for_persist");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));

        let (id_a, b_a) = mk_memory_with_embedding(axis_vec(0));
        let (id_b, b_b) = mk_memory_with_embedding(axis_vec(2));
        let (id_c, b_c) = mk_memory_with_embedding(axis_vec(5));

        // Hoisted so the index OUTLIVES the first projection — the restored map is
        // only honoured when the index it names still holds those vectors.
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let saved = {
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                Arc::clone(&index),
                ckpt.clone(),
            );
            TypedProjection::apply(&mut proj, Lsn(0), &mk_remember(&id_a.0.to_string(), b_a))
                .unwrap();
            TypedProjection::apply(&mut proj, Lsn(1), &mk_remember(&id_b.0.to_string(), b_b))
                .unwrap();
            TypedProjection::apply(&mut proj, Lsn(2), &mk_remember(&id_c.0.to_string(), b_c))
                .unwrap();
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

        // Re-open: new projection over the SURVIVING index, same checkpoint store.
        // The forward map is restored from side data (covered by
        // `id_map_survives_reopen_via_side_data`). The reverse accessor must work
        // over that restored map.
        //
        // This previously re-opened against a brand-new EMPTY index. That passed only
        // because the map was restored regardless of whether the index could back it —
        // so the reverse lookup was resolving vector ids the index did not have. See
        // `restored_id_map_is_discarded_when_the_index_cannot_back_it`.
        let ckpt2 = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj2 =
            VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, Arc::clone(&index), ckpt2);
        assert_eq!(proj2.tracked_memory_count(), 3);
        for (expected_mid, vid) in saved {
            assert_eq!(
                proj2.memory_id_for(vid),
                Some(expected_mid),
                "reverse lookup must survive a persist + reopen via side data",
            );
        }
    }

    // ----------------------------------------------------------------
    // Side-data cadence and compaction
    // ----------------------------------------------------------------

    fn forget_unknown(lsn_tag: usize) -> IntentPayload {
        IntentPayload::Forget {
            user_id: "alice".into(),
            memory_id: format!("never-seen-{lsn_tag}"),
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        }
    }

    #[test]
    fn three_live_applies_write_one_id_map_snapshot() {
        let (_index, ckpt, mut proj, _log, _ckptp) = open_projection();
        for lsn in 0..3u64 {
            let (mid, bytes) = mk_memory_with_embedding(axis_vec(lsn as usize));
            TypedProjection::apply(&mut proj, Lsn(lsn), &mk_remember(&mid.0.to_string(), bytes))
                .unwrap();
        }
        {
            let store = ckpt.lock();
            assert_eq!(
                store.side_data_durable_end_offset(),
                0,
                "no snapshot is written per live apply"
            );
            assert_eq!(
                store.get(VamanaEmbeddingKind::TextPrimary.projection_name()),
                None,
                "the checkpoint waits for the snapshot that matches it"
            );
        }
        drop(proj);
        let store = ckpt.lock();
        let name = VamanaEmbeddingKind::TextPrimary.projection_name();
        assert_eq!(
            store.side_data_durable_end_offset(),
            store.side_data_live_bytes(),
            "exactly one snapshot record"
        );
        assert_eq!(
            store.side_data_checkpoint(name, SIDE_DATA_KEY_ID_MAP),
            Some(Lsn(2))
        );
        assert_eq!(store.get(name), Some(Lsn(2)));
    }

    #[test]
    fn a_dirty_map_is_snapshotted_every_id_map_persist_every_applies() {
        let (_index, ckpt, mut proj, _log, _ckptp) = open_projection();
        let name = VamanaEmbeddingKind::TextPrimary.projection_name();
        for lsn in 0..u64::from(ID_MAP_PERSIST_EVERY) {
            let (mid, bytes) = mk_memory_with_embedding(axis_vec(lsn as usize));
            TypedProjection::apply(&mut proj, Lsn(lsn), &mk_remember(&mid.0.to_string(), bytes))
                .unwrap();
        }
        let store = ckpt.lock();
        let last = Lsn(u64::from(ID_MAP_PERSIST_EVERY) - 1);
        assert_eq!(
            store.side_data_durable_end_offset(),
            store.side_data_live_bytes(),
            "one snapshot for the whole cadence window"
        );
        assert_eq!(
            store.side_data_checkpoint(name, SIDE_DATA_KEY_ID_MAP),
            Some(last)
        );
        assert_eq!(store.get(name), Some(last));
    }

    #[test]
    fn an_unchanged_map_advances_the_checkpoint_without_side_data() {
        let (_log_path, ckpt_path) = tmp_paths("clean_advance");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let name = VamanaEmbeddingKind::TextPrimary.projection_name();
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let mut ids = Vec::new();
        {
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                Arc::clone(&index),
                ckpt.clone(),
            );
            for lsn in 0..2u64 {
                let (mid, bytes) = mk_memory_with_embedding(axis_vec(lsn as usize));
                ids.push(mid.0.to_string());
                TypedProjection::apply(
                    &mut proj,
                    Lsn(lsn),
                    &mk_remember(&mid.0.to_string(), bytes),
                )
                .unwrap();
            }
            Projection::persist_checkpoint(&mut proj).unwrap();
            let snapshot_end = ckpt.lock().side_data_durable_end_offset();

            // Forgets of ids the projection never saw leave the map as it is.
            for lsn in 2..5u64 {
                TypedProjection::apply(&mut proj, Lsn(lsn), &forget_unknown(lsn as usize)).unwrap();
            }
            let mut store = ckpt.lock();
            assert_eq!(store.side_data_durable_end_offset(), snapshot_end);
            assert_eq!(store.get(name), Some(Lsn(4)));
            assert_eq!(
                store.side_data_checkpoint(name, SIDE_DATA_KEY_ID_MAP),
                Some(Lsn(1))
            );
            store.sync().unwrap();
        }

        // The checkpoint (4) is past the snapshot's stamp (1), and the vouch
        // record is what lets the reopen trust the older snapshot.
        let ckpt2 = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj2 = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index, ckpt2);
        assert_eq!(proj2.checkpoint(), Some(Lsn(4)));
        assert_eq!(proj2.tracked_memory_count(), 2);
        for id in &ids {
            assert!(proj2.vector_id_for(id).is_some());
        }
    }

    #[test]
    fn kinds_without_their_embedding_never_write_side_data() {
        let (_log_path, ckpt_path) = tmp_paths("clean_kind");
        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let bootstrap = VamanaProjectionBootstrap {
            storage_path: None,
            config_template: test_config(),
        };
        let mut proj = VamanaProjection::lazy(VamanaEmbeddingKind::Image, bootstrap, ckpt.clone());
        for lsn in 0..3u64 {
            let (mid, bytes) = mk_memory_with_embedding(axis_vec(lsn as usize));
            TypedProjection::apply(&mut proj, Lsn(lsn), &mk_remember(&mid.0.to_string(), bytes))
                .unwrap();
        }
        drop(proj);
        let store = ckpt.lock();
        assert_eq!(
            store.get(VamanaEmbeddingKind::Image.projection_name()),
            Some(Lsn(2))
        );
        assert!(
            !store.side_data_path().exists(),
            "no side data for an always-empty map"
        );
    }

    #[test]
    fn a_compacted_side_data_file_restores_the_same_map() {
        let (_log_path, ckpt_path) = tmp_paths("compacted_reopen");
        let name = VamanaEmbeddingKind::TextPrimary.projection_name();
        let index = Arc::new(RwLock::new(VamanaIndex::new(test_config()).unwrap()));
        let saved: Vec<(String, u32)> = {
            let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
            let mut proj = VamanaProjection::new(
                VamanaEmbeddingKind::TextPrimary,
                Arc::clone(&index),
                ckpt.clone(),
            );
            let mut ids = Vec::new();
            for lsn in 0..5u64 {
                let (mid, bytes) = mk_memory_with_embedding(axis_vec(lsn as usize));
                ids.push(mid.0.to_string());
                TypedProjection::apply(
                    &mut proj,
                    Lsn(lsn),
                    &mk_remember(&mid.0.to_string(), bytes),
                )
                .unwrap();
                // A snapshot per apply, so the file holds superseded records.
                Projection::persist_checkpoint(&mut proj).unwrap();
            }
            let mut store = ckpt.lock();
            let before = store.side_data_durable_end_offset();
            store.compact_side_data().unwrap();
            assert!(store.side_data_durable_end_offset() < before);
            assert_eq!(
                store.side_data_checkpoint(name, SIDE_DATA_KEY_ID_MAP),
                Some(Lsn(4))
            );
            drop(store);
            ids.into_iter()
                .map(|id| {
                    let vid = proj.vector_id_for(&id).unwrap();
                    (id, vid)
                })
                .collect()
        };

        let ckpt = Arc::new(Mutex::new(CheckpointStore::open(&ckpt_path).unwrap()));
        let proj = VamanaProjection::new(VamanaEmbeddingKind::TextPrimary, index, ckpt);
        assert_eq!(proj.checkpoint(), Some(Lsn(4)));
        assert_eq!(proj.tracked_memory_count(), saved.len());
        for (id, vid) in &saved {
            assert_eq!(proj.vector_id_for(id), Some(*vid));
        }
    }
}
