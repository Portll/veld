//! UserMemory↔Veld seam: takes UserMemory capabilities (embedder, vector index) and applies Veld search strategy.
//! Production-grade retrieval engine for memory search
//! Integrated with Vamana graph-based ANN and MiniLM embeddings
//!
//! Features Hebbian-inspired adaptive learning:
//! - Outcome feedback: Memories that help complete tasks get reinforced
//! - Co-activation strengthening: Memories retrieved together form associations
//! - Time-based decay: Unused associations naturally weaken

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

use super::introspection::ConsolidationEventBuffer;
use super::storage::SearchCriteria;
use super::types::*;
use crate::constants::{
    PREFETCH_RECENCY_FULL_BOOST, PREFETCH_RECENCY_FULL_HOURS, PREFETCH_RECENCY_PARTIAL_BOOST,
    PREFETCH_RECENCY_PARTIAL_HOURS, PREFETCH_TEMPORAL_WINDOW_HOURS,
    VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
};
use crate::embeddings::competitive::CompetitiveEmbedder;
use crate::embeddings::Embedder;
use crate::vector_db::vamana::{VamanaConfig, VamanaIndex, VamanaSearchParams};

/// Filename for persisted Vamana index (instant startup)
const VAMANA_INDEX_FILE: &str = "vamana.idx";
/// Present beside a saved index when every stored vector mapping refers to it. Each path that
/// assigns vector ids writes its mappings, and rebuilds remove the marker until the next save.
pub(crate) const MAPPINGS_MARKER_FILE: &str = "mappings.authoritative";
/// Filename for persisted secondary Vamana index
const SECONDARY_VAMANA_INDEX_FILE: &str = "secondary_vamana.idx";
/// The secondary index's memory ↔ vector id map, written beside the index file by the same
/// save and naming that file by its checksum. The primary's map is the stored vector mappings;
/// the secondary has none in the store, and rebuilding it at open decoded every memory.
const SECONDARY_ID_MAP_FILE: &str = "secondary_id_map.bin";
const SECONDARY_ID_MAP_MAGIC: [u8; 4] = *b"VSID";
/// Vector mappings a rebuild or compaction writes per commit: one commit per memory made a
/// 100k-memory rebuild on redb run for minutes, and one for the whole index would hold every
/// encoded mapping at once.
const MAPPING_WRITE_BATCH: usize = 10_000;
const SECONDARY_ID_MAP_VERSION: u32 = 1;

/// When [`RetrievalEngine::save_if_due`] writes the index files: a save rewrites every vector
/// (413 MB for the largest live tenant), so it waits for enough changes to be worth one, or
/// for changes to have gone unsaved long enough. Shutdown and eviction save unconditionally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexSavePolicy {
    /// Changes that make a save due, at least: `VELD_VECTOR_INDEX_SAVE_CHANGES`.
    pub min_changes: u64,
    /// The share of the index's vectors that makes a save due when that is more, so a large
    /// index is not rewritten for a change count that is small beside it.
    pub changes_per_vector: f64,
    /// How long a change may go unsaved: `VELD_VECTOR_INDEX_SAVE_INTERVAL_SECS`.
    pub max_unsaved_age: std::time::Duration,
}

impl Default for IndexSavePolicy {
    fn default() -> Self {
        Self {
            min_changes: 1_000,
            changes_per_vector: 0.05,
            max_unsaved_age: std::time::Duration::from_secs(15 * 60),
        }
    }
}

impl IndexSavePolicy {
    /// The policy this process saves with: the environment, read once.
    pub fn process() -> Self {
        static POLICY: std::sync::OnceLock<IndexSavePolicy> = std::sync::OnceLock::new();
        *POLICY.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let defaults = Self::default();
        let positive = |name: &str| {
            lookup(name).and_then(|raw| match raw.trim().parse::<u64>() {
                Ok(n) if n > 0 => Some(n),
                _ => {
                    warn!("{name}='{raw}' ignored: not a positive whole number");
                    None
                }
            })
        };
        Self {
            min_changes: positive("VELD_VECTOR_INDEX_SAVE_CHANGES").unwrap_or(defaults.min_changes),
            changes_per_vector: defaults.changes_per_vector,
            max_unsaved_age: positive("VELD_VECTOR_INDEX_SAVE_INTERVAL_SECS")
                .map(std::time::Duration::from_secs)
                .unwrap_or(defaults.max_unsaved_age),
        }
    }

    /// Whether `unsaved` changes, the oldest `unsaved_for` old, over an index of `vectors`
    /// vectors call for a save now.
    pub fn is_due(&self, unsaved: u64, unsaved_for: std::time::Duration, vectors: usize) -> bool {
        if unsaved == 0 {
            return false;
        }
        let threshold = self
            .min_changes
            .max((vectors as f64 * self.changes_per_vector) as u64);
        unsaved >= threshold || unsaved_for >= self.max_unsaved_age
    }
}

/// Whether this process walks its Vamana graphs on int8 codes (`VELD_VECTOR_INT8_TRAVERSAL`),
/// rescoring each walk's beam in f32. Off unless set to 1/true/on/yes. Read when an index is
/// created or loaded: the codes are built then.
fn int8_traversal_enabled() -> bool {
    std::env::var("VELD_VECTOR_INT8_TRAVERSAL")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

/// Multi-modal retrieval engine with production vector search
///
/// # Lock Ordering (SHO-72)
///
/// To prevent deadlocks, locks MUST be acquired in ascending order:
///
/// 1. `vector_index` - Primary vector similarity search index (384d)
/// 2. `id_mapping` - Primary memory ID ↔ vector ID mapping
/// 3. `secondary_vector_index` - Secondary vector index (768d, optional)
/// 4. `secondary_id_mapping` - Secondary memory ID ↔ vector ID mapping (optional)
/// 5. `consolidation_events` - Introspection event buffer
///
/// **Rules:**
/// - Never acquire a lower-numbered lock while holding a higher-numbered one
/// - For read operations, prefer `read()` over `write()` when possible
/// - Release locks as soon as possible (don't hold during I/O)
///
/// **Note:** Memory graph (Hebbian learning) has been consolidated into GraphMemory
/// which is managed at the API layer (MultiUserMemoryManager.graph_memories)
pub struct RetrievalEngine {
    /// Vector index inserts lost after their error was handled and ingest
    /// continued — the memory is stored and semantically unreachable.
    vector_index_failures: std::sync::atomic::AtomicU64,
    storage: Arc<dyn crate::storage::PrimaryMemoryStore>,
    embedder: Arc<CompetitiveEmbedder>,
    /// Lock order: 1 - Acquire first
    vector_index: Arc<RwLock<VamanaIndex>>,
    /// Lock order: 2 - Primary memory ID ↔ vector ID mapping
    id_mapping: Arc<RwLock<IdMapping>>,
    /// Lock order: 3 - Secondary Vamana (768d), present only when secondary model is configured
    secondary_vector_index: Option<Arc<RwLock<VamanaIndex>>>,
    /// Lock order: 4 - Secondary ID mapping, mirrors secondary_vector_index
    secondary_id_mapping: Option<Arc<RwLock<IdMapping>>>,
    /// Storage path for persisting vector index and ID mapping
    storage_path: PathBuf,
    /// Lock order: 5 - Acquire last
    /// Shared consolidation event buffer for introspection
    /// Records edge formation, strengthening, and pruning events
    consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
    /// The owning `MemorySystem`'s runtime dials, read once per vector search for the
    /// `vector_exact_search_max` and `vector_search_beam` overrides. `None` keeps the index's
    /// own sizing.
    dials: RwLock<Option<Arc<RwLock<crate::memory::dials::CognitiveDials>>>>,
    /// Index changes since the files were last saved; see [`Self::save_if_due`].
    unsaved_changes: std::sync::atomic::AtomicU64,
    /// The index was rebuilt or compacted since the files were saved, so they describe
    /// none of it however few changes were counted.
    rewritten_since_save: std::sync::atomic::AtomicBool,
    /// When the oldest unsaved change was made; `None` while everything is saved. Not a
    /// lock in the order above: held only to read or swap the instant.
    unsaved_since: parking_lot::Mutex<Option<std::time::Instant>>,
    /// Held for a whole [`Self::save`], before any lock in the order above. Saves come from
    /// the drainer's tick, eviction and shutdown, and two at once would write the same
    /// temporary files.
    save_lock: parking_lot::Mutex<()>,
}

/// Bidirectional mapping between memory IDs and vector IDs
///
/// Supports multiple vectors per memory for chunked embeddings.
/// When long content is split into chunks, each chunk gets its own vector ID,
/// but all map back to the same MemoryId.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct IdMapping {
    /// Maps each memory to ALL its vector IDs (supports chunked embeddings)
    memory_to_vectors: HashMap<MemoryId, Vec<u32>>,
    /// Maps each vector ID back to its parent memory
    vector_to_memory: HashMap<u32, MemoryId>,
}

impl IdMapping {
    fn new() -> Self {
        Self {
            memory_to_vectors: HashMap::new(),
            vector_to_memory: HashMap::new(),
        }
    }

    /// Insert a single vector for a memory (legacy/simple case)
    ///
    /// Idempotent: removes any existing mappings for this memory first
    /// to prevent orphaned vector IDs from accumulating.
    fn insert(&mut self, memory_id: MemoryId, vector_id: u32) {
        // Remove stale mappings to prevent orphans on re-index
        if let Some(old_ids) = self.memory_to_vectors.remove(&memory_id) {
            for old_id in old_ids {
                self.vector_to_memory.remove(&old_id);
            }
        }
        self.vector_to_memory.insert(vector_id, memory_id.clone());
        self.memory_to_vectors.insert(memory_id, vec![vector_id]);
    }

    /// Insert multiple vectors for a memory (chunked embedding case)
    ///
    /// Idempotent: removes any existing mappings for this memory first
    /// to prevent orphaned vector IDs from accumulating.
    fn insert_chunks(&mut self, memory_id: MemoryId, vector_ids: Vec<u32>) {
        // Remove stale mappings to prevent orphans on re-index
        if let Some(old_ids) = self.memory_to_vectors.remove(&memory_id) {
            for old_id in old_ids {
                self.vector_to_memory.remove(&old_id);
            }
        }
        for &vid in &vector_ids {
            self.vector_to_memory.insert(vid, memory_id.clone());
        }
        self.memory_to_vectors.insert(memory_id, vector_ids);
    }

    fn get_memory_id(&self, vector_id: u32) -> Option<&MemoryId> {
        self.vector_to_memory.get(&vector_id)
    }

    /// Remove a memory and return ALL its vector IDs
    fn remove_all(&mut self, memory_id: &MemoryId) -> Vec<u32> {
        if let Some(vector_ids) = self.memory_to_vectors.remove(memory_id) {
            for vid in &vector_ids {
                self.vector_to_memory.remove(vid);
            }
            vector_ids
        } else {
            Vec::new()
        }
    }

    /// Number of unique memories in the mapping
    fn len(&self) -> usize {
        self.memory_to_vectors.len()
    }

    fn clear(&mut self) {
        self.memory_to_vectors.clear();
        self.vector_to_memory.clear();
    }

    /// Check if a memory ID has any vector mappings (for deletion verification)
    fn contains(&self, memory_id: &MemoryId) -> bool {
        self.memory_to_vectors.contains_key(memory_id)
    }
}

/// Write the secondary index's id map beside the index file at `index_path`, naming that file
/// by the checksum its save recorded and by its vector count.
///
/// Layout, little-endian: magic `VSID`, version u32, index checksum u64, index vector count
/// u64, entry count u64, then per entry the 16-byte memory id, a u32 vector count and the
/// vector ids as u32.
fn write_secondary_id_map(
    index_path: &Path,
    index_vectors: usize,
    entries: &[(MemoryId, Vec<u32>)],
) -> Result<()> {
    let checksum = VamanaIndex::saved_checksum(index_path)
        .context("reading the checksum of the secondary index just saved")?;
    let vector_ids: usize = entries.iter().map(|(_, ids)| ids.len()).sum();
    let mut bytes = Vec::with_capacity(32 + entries.len() * 20 + vector_ids * 4);
    bytes.extend_from_slice(&SECONDARY_ID_MAP_MAGIC);
    bytes.extend_from_slice(&SECONDARY_ID_MAP_VERSION.to_le_bytes());
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes.extend_from_slice(&(index_vectors as u64).to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (memory_id, ids) in entries {
        bytes.extend_from_slice(memory_id.0.as_bytes());
        bytes.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        for id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
    }
    crate::vector_db::vamana_persist::write_file_atomically(
        &index_path.with_file_name(SECONDARY_ID_MAP_FILE),
        &bytes,
    )
}

/// The id map saved with the secondary index file at `index_path`, checked against `index`,
/// which was loaded from that file. `None` when there is no map, or it was written by
/// another save: then nothing ties its ids to these vectors. Entries for vectors the index
/// does not hold or has deleted are dropped, and vectors no entry names are marked deleted,
/// as the primary load retires unreferenced vectors.
fn read_secondary_id_map(index_path: &Path, index: &VamanaIndex) -> Result<Option<IdMapping>> {
    let map_path = index_path.with_file_name(SECONDARY_ID_MAP_FILE);
    let bytes = match fs::read(&map_path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", map_path.display())),
    };
    let mut cursor = SliceReader::new(&bytes);
    if cursor.take(4)? != SECONDARY_ID_MAP_MAGIC {
        anyhow::bail!("{} is not a secondary id map", map_path.display());
    }
    let version = cursor.u32()?;
    if version != SECONDARY_ID_MAP_VERSION {
        anyhow::bail!("secondary id map version {version} is not one this build reads");
    }
    let checksum = cursor.u64()?;
    let index_vectors = cursor.u64()?;
    if checksum != VamanaIndex::saved_checksum(index_path)? || index_vectors != index.len() as u64
    {
        return Ok(None);
    }
    let entries = cursor.u64()?;
    let vectors = index.len();
    let mut map = IdMapping::new();
    let mut referenced = HashSet::new();
    for _ in 0..entries {
        let memory_id = MemoryId(uuid::Uuid::from_bytes(cursor.take(16)?.try_into()?));
        let count = cursor.u32()? as usize;
        let mut ids = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let id = cursor.u32()?;
            if (id as usize) < vectors && !index.is_deleted(id) {
                ids.push(id);
            }
        }
        if !ids.is_empty() {
            referenced.extend(ids.iter().copied());
            map.insert_chunks(memory_id, ids);
        }
    }
    if !cursor.is_empty() {
        anyhow::bail!("{} has bytes past its last entry", map_path.display());
    }
    for id in 0..vectors as u32 {
        if !referenced.contains(&id) && !index.is_deleted(id) {
            index.mark_deleted(id);
        }
    }
    Ok(Some(map))
}

/// Little-endian reads over a byte slice that fail, rather than panic, past its end.
struct SliceReader<'a> {
    rest: &'a [u8],
}

impl<'a> SliceReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.rest.len() < n {
            anyhow::bail!("truncated: wanted {n} bytes, {} left", self.rest.len());
        }
        let (head, tail) = self.rest.split_at(n);
        self.rest = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }
}

/// What a corpus needs, decided from the two markers alone.
///
/// Pure, and separated from the migration for one reason: the defect this replaced
/// was a decision, not a loop. `recorded_dim == target_dim` returned early over a
/// corpus embedded by a DIFFERENT model at the same dimension, and the per-memory
/// length check downstream was blind for the same reason — every vector the right
/// shape, the wrong content. Two guards, one blind spot. A decision that can be
/// enumerated in a test is one that cannot quietly regain it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CorpusAction {
    /// Dimension and embedder both match what is configured.
    UpToDate,
    /// Dimension matches and nothing recorded WHICH model wrote the corpus. Not
    /// knowable after the fact, so it is neither claimed current nor re-embedded.
    RecordUnverified,
    /// Dimension changed: the per-memory length check identifies what to redo.
    MigrateChangedLength,
    /// Embedder changed at the SAME dimension: length identifies nothing, so
    /// everything must be redone.
    MigrateAll,
}

pub(crate) fn corpus_action(
    recorded_dim: Option<usize>,
    recorded_id: Option<&str>,
    target_dim: usize,
    target_id: &str,
) -> CorpusAction {
    if recorded_dim != Some(target_dim) {
        return CorpusAction::MigrateChangedLength;
    }
    match recorded_id {
        Some(id) if id == target_id => CorpusAction::UpToDate,
        Some(_) => CorpusAction::MigrateAll,
        None => CorpusAction::RecordUnverified,
    }
}

impl RetrievalEngine {
    /// Create new retrieval engine with shared embedder (CRITICAL: embedder loaded only once)
    ///
    /// ATOMIC ARCHITECTURE: RocksDB is the ONLY source of truth.
    /// - Vector mappings are stored atomically with memories in RocksDB
    /// - Vamana index is rebuilt from RocksDB on startup (pure in-memory cache)
    /// - No more file-based IdMapping = no more orphaned memories
    pub fn new(
        storage: Arc<dyn crate::storage::PrimaryMemoryStore>,
        storage_path: &std::path::Path,
        embedder: Arc<CompetitiveEmbedder>,
    ) -> Result<Self> {
        Self::with_event_buffer(storage, storage_path, embedder, None)
    }

    /// Create retrieval engine with event buffer for consolidation introspection
    ///
    /// The event buffer is used to record Hebbian learning events:
    /// - Edge formation (new associations)
    /// - Edge strengthening (co-activation)
    /// - Edge potentiation (LTP)
    /// - Edge pruning (decay below threshold)
    ///
    /// ATOMIC STARTUP: Rebuilds Vamana from RocksDB mappings for crash safety.
    pub fn with_event_buffer(
        storage: Arc<dyn crate::storage::PrimaryMemoryStore>,
        storage_path: &std::path::Path,
        embedder: Arc<CompetitiveEmbedder>,
        consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
    ) -> Result<Self> {
        // The path is a PARAMETER, not a store method: where files live is an
        // engine detail the trait deliberately does not carry.
        let storage_path = storage_path.to_path_buf();

        // Initialize primary Vamana index optimized for 10M+ memories per user.
        // Dimension follows the active primary embedder (Nomic 768d / Matryoshka,
        // or MiniLM 384d fallback) — never hardcoded, or a model swap silently
        // corrupts the index.
        let primary_dim = embedder.primary_dimension();
        let vamana_config = VamanaConfig {
            dimension: primary_dim,
            max_degree: 32,        // Increased for better recall at scale
            search_list_size: 100, // 2x for better accuracy with 10M vectors
            alpha: 1.2,
            use_mmap: true, // Memory-mapped: OS manages paging, RSS stays low at scale
            quantized_traversal: int8_traversal_enabled(),
            ..Default::default()
        };

        let vamana_storage = storage_path.join("vector_index");
        std::fs::create_dir_all(&vamana_storage)?;
        let vector_index = VamanaIndex::with_storage_path(vamana_config, Some(vamana_storage))
            .context("Failed to initialize Vamana vector index")?;
        let id_mapping = IdMapping::new();

        // Initialize secondary Vamana index if secondary embedder is available
        let (secondary_vector_index, secondary_id_mapping) =
            if let Some(sec_dim) = embedder.secondary_dimension() {
                let sec_config = VamanaConfig {
                    dimension: sec_dim,
                    max_degree: 32,
                    search_list_size: 100,
                    alpha: 1.2,
                    use_mmap: true, // mmap by default for secondary to control memory footprint
                    quantized_traversal: int8_traversal_enabled(),
                    ..Default::default()
                };
                let sec_storage = storage_path.join("secondary_vector_index");
                std::fs::create_dir_all(&sec_storage)?;
                let sec_index = VamanaIndex::with_storage_path(sec_config, Some(sec_storage))
                    .context("Failed to initialize secondary Vamana vector index")?;
                info!("Secondary Vamana index initialized ({}d, mmap)", sec_dim);
                (
                    Some(Arc::new(RwLock::new(sec_index))),
                    Some(Arc::new(RwLock::new(IdMapping::new()))),
                )
            } else {
                (None, None)
            };

        let engine = Self {
            vector_index_failures: std::sync::atomic::AtomicU64::new(0),
            storage,
            embedder,
            vector_index: Arc::new(RwLock::new(vector_index)),
            id_mapping: Arc::new(RwLock::new(id_mapping)),
            secondary_vector_index,
            secondary_id_mapping,
            storage_path,
            consolidation_events,
            dials: RwLock::new(None),
            unsaved_changes: std::sync::atomic::AtomicU64::new(0),
            rewritten_since_save: std::sync::atomic::AtomicBool::new(false),
            unsaved_since: parking_lot::Mutex::new(None),
            save_lock: parking_lot::Mutex::new(()),
        };

        // ATOMIC STARTUP: Rebuild Vamana from RocksDB (single source of truth)
        engine.rebuild_from_rocksdb()?;

        // Alignment onboarding: load (or autofit, if VELD_ALIGNMENT_AUTOFIT=1)
        // a learned projection from secondary→primary space. No-op for
        // primary-only installs. Errors are downgraded to a warning so an
        // unreadable alignment file doesn't take down the engine.
        if let (Some(secondary), _) = (
            engine.embedder.secondary_embedder(),
            engine.embedder.has_alignment(),
        ) {
            let pair_id = crate::embeddings::AlignmentPairId::new(
                engine.embedder.primary_embedder().model_id(),
                secondary.model_id(),
            );
            // Hand the LIVE secondary instance to onboarding so auto-fit aligns
            // the SAME vectors the secondary produces at ingest, and never tries
            // to re-load an HTTP/runtime-only secondary by its (non-loadable)
            // `model_id()` — which is the "unknown embedder id" autofit failure.
            let live_secondary = secondary.clone();
            match super::alignment_onboarding::resolve_or_fit_with_secondary(
                &pair_id,
                live_secondary,
            ) {
                Ok(Some(alignment)) => {
                    if let Err(e) = engine.embedder.install_alignment(alignment) {
                        tracing::warn!("could not install alignment {:?}: {}", pair_id, e);
                    }
                }
                Ok(None) => {}
                Err(e) => tracing::warn!("alignment onboarding failed for {:?}: {}", pair_id, e),
            }
        }

        Ok(engine)
    }

    /// Re-embed stored vectors when the primary embedding dimension changes.
    ///
    /// Triggered by promoting Nomic to primary, or by changing `VELD_NOMIC_DIM`.
    /// A marker file (`.veld_embedding_dim`) records the dimension the corpus was
    /// last embedded at; when it differs from the active embedder, every memory
    /// whose primary embedding has the wrong length is re-embedded from content
    /// and persisted.
    ///
    /// Properties:
    /// - **Idempotent**: a matching marker short-circuits in O(1).
    /// - **Crash-resumable**: the marker advances only after a clean pass; the
    ///   per-memory length check skips already-migrated memories on retry.
    /// - **Reuses vectors**: an old Nomic `embeddings_secondary` already at the
    ///   target length is promoted in place instead of re-running ONNX.
    fn migrate_embeddings_if_dimension_changed(&self) -> Result<()> {
        let target = self.embedder.primary_dimension();
        let target_id = self.embedder.primary_id().to_string();
        let marker = self.storage_path.join(".veld_embedding_dim");
        let id_marker = self.storage_path.join(".veld_embedder");

        let recorded: Option<usize> = std::fs::read_to_string(&marker)
            .ok()
            .and_then(|s| s.trim().parse().ok());
        let recorded_id: Option<String> = std::fs::read_to_string(&id_marker)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // DIMENSION IS NOT IDENTITY. Two different models at the same dimension —
        // the hash-bigram fallback and MiniLM are both 384 — leave `recorded ==
        // Some(target)` true, so this returned early over a corpus embedded by
        // something else entirely. The length check below cannot see it either:
        // every vector is the right SHAPE and the wrong CONTENT.
        let action = corpus_action(recorded, recorded_id.as_deref(), target, &target_id);
        if action == CorpusAction::UpToDate {
            return Ok(()); // corpus is at the active dimension AND the active embedder
        }

        let ids = self.storage.get_all_ids()?;
        if ids.is_empty() {
            // Fresh / empty store — record both; nothing to migrate.
            let _ = std::fs::write(&marker, target.to_string());
            let _ = std::fs::write(&id_marker, &target_id);
            return Ok(());
        }

        // A corpus written before this marker existed. We CANNOT tell what embedded
        // it, and neither answer is safe to assume: re-embedding thousands of vectors
        // unasked is expensive, and writing the current id claims a provenance nobody
        // established. So the unknown gets its own value — `unverified:<id>` — which
        // is not equal to `<id>`, so it never satisfies the early return above, and
        // says in the file itself that this was believed rather than checked.
        if action == CorpusAction::RecordUnverified {
            warn!(
                embedder = %target_id,
                memories = ids.len(),
                "embedding provenance UNKNOWN for this corpus: the dimension matches the active \
                 embedder but nothing recorded which model wrote it, and two models at one \
                 dimension are indistinguishable in the store. Not re-embedding unasked. POST \
                 /api/index/reembed to make the corpus current, after which this resolves."
            );
            let _ = std::fs::write(&id_marker, format!("unverified:{target_id}"));
            return Ok(());
        }

        // An embedder swap at the SAME dimension invalidates every vector while
        // leaving all their lengths correct, so the per-memory length check below is
        // blind to it and everything must be re-embedded.
        let force_all = action == CorpusAction::MigrateAll;

        info!(
            "Embedding dimension change detected (was {:?}, now {}d) — checking {} memories",
            recorded,
            target,
            ids.len()
        );

        // Stale persisted indices must not be loaded against the new dimension.
        let _ = std::fs::remove_file(
            self.storage_path
                .join("vector_index")
                .join(VAMANA_INDEX_FILE),
        );
        let _ = std::fs::remove_file(
            self.storage_path
                .join("secondary_vector_index")
                .join(SECONDARY_VAMANA_INDEX_FILE),
        );

        let mut reembedded = 0usize;
        let mut reused = 0usize;
        let mut unchanged = 0usize;
        let mut failed = 0usize;

        for id in &ids {
            let mut memory = match self.storage.get(id) {
                Ok(m) => m,
                Err(e) => {
                    // Named, because a count of failures cannot be acted on and an id can:
                    // 24 of these blocked the marker on the 2026-09-13 deploy.
                    warn!("Embedding migration cannot load {}: {}", id.0, e);
                    failed += 1;
                    continue;
                }
            };

            // Already the correct length — nothing to do (keeps retries cheap).
            // Unless the EMBEDDER changed at the same dimension, in which case length
            // is exactly the property that stayed correct while the content did not.
            let needs_migration =
                force_all || matches!(&memory.experience.embeddings, Some(e) if e.len() != target);
            if !needs_migration {
                unchanged += 1;
                continue;
            }

            // Prefer an existing secondary embedding already at the target length
            // (installs that ran the old Nomic secondary have one for free).
            let new_emb = match &memory.experience.embeddings_secondary {
                Some(sec) if sec.len() == target => {
                    reused += 1;
                    sec.clone()
                }
                _ => match self.embedder.encode(&memory.experience.content) {
                    Ok(e) if e.len() == target => {
                        reembedded += 1;
                        e
                    }
                    Ok(e) => {
                        warn!(
                            "Re-embed for {} produced {}d, expected {}d — skipping",
                            id.0,
                            e.len(),
                            target
                        );
                        failed += 1;
                        continue;
                    }
                    Err(e) => {
                        warn!("Re-embed failed for {}: {}", id.0, e);
                        failed += 1;
                        continue;
                    }
                },
            };

            memory.experience.embeddings = Some(new_emb);
            if let Err(e) = self.storage.update(&memory) {
                warn!("Failed to persist re-embedded memory {}: {}", id.0, e);
                failed += 1;
            }
        }

        // Advance the marker only after a clean pass — a crash or failure leaves
        // it stale so the next startup resumes the remaining memories.
        if failed == 0 {
            let _ = std::fs::write(&marker, target.to_string());
            // Both markers advance together, or neither does. Advancing the
            // dimension alone would leave the corpus recorded as unverified
            // forever and re-run the whole pass on every startup.
            let _ = std::fs::write(&id_marker, &target_id);
        } else {
            warn!(
                "Embedding migration finished with {} failures — marker not advanced, will retry next startup",
                failed
            );
        }

        info!(
            "Embedding migration: {} re-embedded, {} reused from secondary, {} already correct, {} failed",
            reembedded, reused, unchanged, failed
        );

        Ok(())
    }

    /// Initialize Vamana index from persisted file or rebuild from RocksDB
    ///
    /// INSTANT STARTUP ARCHITECTURE:
    /// 1. Try loading .vamana file (instant, ~10ms for 500k vectors)
    /// 2. Fall back to RocksDB rebuild (slow, ~seconds for 500k vectors)
    ///
    /// RocksDB remains the source of truth for ID mappings.
    /// The .vamana file is a cache that can be regenerated.
    fn rebuild_from_rocksdb(&self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // MIGRATION: if the primary embedding dimension changed (Nomic promoted
        // to primary, or VELD_NOMIC_DIM changed), re-embed stored vectors before
        // any index load/rebuild so the corpus and index dimensions agree.
        if let Err(e) = self.migrate_embeddings_if_dimension_changed() {
            warn!("Embedding dimension migration failed (rebuilding best-effort): {e}");
        }

        let mut primary_loaded_from_file = false;

        // Try instant startup from persisted Vamana file
        let vamana_path = self
            .storage_path
            .join("vector_index")
            .join(VAMANA_INDEX_FILE);
        if vamana_path.exists() {
            if let Ok(loaded) = self.try_load_persisted_vamana(&vamana_path) {
                if loaded {
                    info!(
                        "Instant startup: loaded Vamana in {:.2}ms",
                        start_time.elapsed().as_secs_f64() * 1000.0
                    );
                    primary_loaded_from_file = true;
                }
            }
        }

        if !primary_loaded_from_file {
            self.rebuild_primary_index()?;
        }

        // Rebuild secondary index if configured
        self.rebuild_secondary_from_rocksdb()?;

        Ok(())
    }

    /// Rebuild secondary Vamana index from RocksDB memory data
    ///
    /// Pre-fetches all memory data (I/O) outside lock scope to minimize
    /// lock hold time, then acquires secondary locks (3, 4) to populate.
    /// Secondary has no RocksDB vector mappings — it rebuilds from
    /// `experience.embeddings_secondary` stored in each memory.
    fn rebuild_secondary_from_rocksdb(&self) -> Result<()> {
        let (sec_index, sec_mapping) =
            match (&self.secondary_vector_index, &self.secondary_id_mapping) {
                (Some(idx), Some(map)) => (idx, map),
                _ => return Ok(()), // No secondary configured
            };

        let start_time = std::time::Instant::now();

        // Try instant startup from persisted secondary Vamana file
        let sec_vamana_path = self
            .storage_path
            .join("secondary_vector_index")
            .join(SECONDARY_VAMANA_INDEX_FILE);

        if sec_vamana_path.exists() {
            match VamanaIndex::load_from_file(&sec_vamana_path) {
                // Dimension guard: only adopt a persisted secondary index whose
                // dimension matches the current secondary embedder.
                Ok(loaded_index)
                    if self.embedder.secondary_dimension()
                        != Some(loaded_index.config.dimension) =>
                {
                    warn!(
                        "Persisted secondary Vamana dimension {} != expected {:?}, rebuilding",
                        loaded_index.config.dimension,
                        self.embedder.secondary_dimension()
                    );
                }
                Ok(mut loaded_index) => {
                    // Restore search_list_size to runtime config
                    loaded_index.config.search_list_size = 100;
                    let loaded_count = loaded_index.len();

                    // The map saved with this file. Deriving it from the store instead
                    // decoded every memory, and paired ids by store order, which is not the
                    // order vectors were inserted in once any were added after a rebuild.
                    match read_secondary_id_map(&sec_vamana_path, &loaded_index) {
                        Ok(Some(map)) => {
                            let memories = map.len();
                            loaded_index.set_quantized_traversal(int8_traversal_enabled());
                            *sec_index.write() = loaded_index;
                            *sec_mapping.write() = map;
                            info!(
                                "Secondary Vamana loaded from file: {} vectors, {} memories in {:.2}ms",
                                loaded_count,
                                memories,
                                start_time.elapsed().as_secs_f64() * 1000.0
                            );
                            return Ok(());
                        }
                        Ok(None) => warn!(
                            "Secondary Vamana file has no id map saved with it, rebuilding"
                        ),
                        Err(e) => warn!("Secondary Vamana id map unreadable ({e:#}), rebuilding"),
                    }
                }
                Err(e) => {
                    warn!("Failed to load secondary Vamana file: {}, rebuilding", e);
                }
            }
        }

        // Full rebuild: pre-fetch all memories with secondary embeddings (I/O outside locks)
        let all_ids = self.storage.get_all_ids()?;
        let mut to_index: Vec<(MemoryId, Vec<f32>)> = Vec::new();

        for memory_id in &all_ids {
            if let Ok(memory) = self.storage.get(memory_id) {
                if memory.is_forgotten() {
                    continue;
                }
                if let Some(ref sec_emb) = memory.experience.embeddings_secondary {
                    to_index.push((memory_id.clone(), sec_emb.clone()));
                }
            }
        }

        if to_index.is_empty() {
            info!("No secondary embeddings found, secondary index empty");
            return Ok(());
        }

        // LOCK ORDERING: secondary_vector_index (3) before secondary_id_mapping (4)
        let mut sec_idx = sec_index.write();
        let mut sec_map = sec_mapping.write();
        sec_map.clear();

        let mut indexed = 0usize;
        let mut failed = 0usize;

        for (memory_id, embedding) in &to_index {
            match sec_idx.add_vector(embedding.clone()) {
                Ok(vector_id) => {
                    sec_map.insert(memory_id.clone(), vector_id);
                    indexed += 1;
                }
                Err(e) => {
                    warn!("Failed to index secondary for {}: {}", memory_id.0, e);
                    failed += 1;
                }
            }
        }

        drop(sec_map);
        drop(sec_idx);
        self.mark_rewritten();
        info!(
            "Rebuilt secondary Vamana from RocksDB: {} indexed, {} failed in {:.2}s",
            indexed,
            failed,
            start_time.elapsed().as_secs_f64()
        );

        Ok(())
    }

    /// Try loading Vamana from persisted file for instant startup
    ///
    /// Returns Ok(true) if successfully loaded, Ok(false) if should fall back to rebuild.
    /// Verifies checksum and cross-checks with RocksDB mappings.
    fn try_load_persisted_vamana(&self, vamana_path: &Path) -> Result<bool> {
        if !vamana_path.with_file_name(MAPPINGS_MARKER_FILE).exists() {
            info!("saved vector index predates authoritative mappings; rebuilding it");
            return Ok(false);
        }
        // Verify file integrity first
        if !VamanaIndex::verify_index_file(vamana_path)? {
            warn!("Vamana file checksum mismatch, will rebuild");
            return Ok(false);
        }

        // Load the persisted index
        let mut loaded_index = match VamanaIndex::load_from_file(vamana_path) {
            Ok(idx) => idx,
            Err(e) => {
                warn!("Failed to load Vamana file: {}, will rebuild", e);
                return Ok(false);
            }
        };

        // Dimension guard: a persisted index from a different embedding model
        // (e.g. a pre-Nomic-promotion 384d file) must never be adopted against
        // the current embedder. Migration deletes such files, but this is the
        // defensive backstop if one survives.
        let expected_dim = self.embedder.primary_dimension();
        if loaded_index.config.dimension != expected_dim {
            warn!(
                "Persisted Vamana dimension {} != active embedder dimension {}, will rebuild",
                loaded_index.config.dimension, expected_dim
            );
            return Ok(false);
        }

        let loaded_count = loaded_index.len();
        let deleted = loaded_index.deleted_ids.read().clone();
        let live = |vid: u32| (vid as usize) < loaded_count && !deleted.contains(vid);

        // A mapping whose ids are missing or deleted in this file was written after the last save;
        // its memory is queued at open for the embedding backlog, whose drain re-indexes and
        // re-records it.
        let mut usable: Vec<(MemoryId, Vec<u32>)> = Vec::new();
        let mut stale: Vec<MemoryId> = Vec::new();
        for (memory_id, entry) in self.storage.get_all_vector_mappings()? {
            match entry.text_vectors() {
                Some(ids) if !ids.is_empty() && ids.iter().all(|&vid| live(vid)) => {
                    usable.push((memory_id, ids.clone()))
                }
                Some(ids) if !ids.is_empty() => stale.push(memory_id),
                _ => {}
            }
        }

        let (sampled, mismatched) = self.own_vector_mismatches(&loaded_index, &usable);
        if mismatched * 10 > sampled {
            warn!(
                sampled,
                mismatched, "saved vector index disagrees with memories' own vectors; rebuilding"
            );
            return Ok(false);
        }

        let referenced: HashSet<u32> = usable
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        let unreferenced: Vec<u32> = (0..loaded_count as u32)
            .filter(|vid| !referenced.contains(vid) && !deleted.contains(*vid))
            .collect();
        for &vid in &unreferenced {
            loaded_index.mark_deleted(vid);
        }
        for memory_id in &stale {
            if let Err(e) = self.storage.delete_vector_mapping(memory_id) {
                warn!(
                    "Failed to clear stale vector mapping for {}: {}",
                    memory_id.0, e
                );
            }
        }

        // Persistence stores a construction default; recall searches with the runtime list size.
        loaded_index.config.search_list_size = 100;
        loaded_index.set_quantized_traversal(int8_traversal_enabled());
        let needs_compaction = loaded_index.needs_compaction();
        {
            let mut index = self.vector_index.write();
            let mut id_mapping = self.id_mapping.write();
            *index = loaded_index;
            id_mapping.clear();
            for (memory_id, ids) in usable {
                id_mapping.insert_chunks(memory_id, ids);
            }
        }

        info!(
            vectors = loaded_count,
            memories = self.id_mapping.read().len(),
            retired_vectors = unreferenced.len(),
            stale_mappings = stale.len(),
            "Loaded saved vector index"
        );
        if needs_compaction {
            self.compact()?;
        }
        Ok(true)
    }

    /// Over a sample of single-vector memories that keep their own vector: how many were checked,
    /// and how many map to an index vector that is not that vector.
    fn own_vector_mismatches(
        &self,
        index: &VamanaIndex,
        usable: &[(MemoryId, Vec<u32>)],
    ) -> (usize, usize) {
        const SAMPLE: usize = 256;
        let singles: Vec<&(MemoryId, Vec<u32>)> =
            usable.iter().filter(|(_, ids)| ids.len() == 1).collect();
        let step = (singles.len() / SAMPLE).max(1);
        let (mut sampled, mut mismatched) = (0, 0);
        for (memory_id, ids) in singles.into_iter().step_by(step).take(SAMPLE) {
            let Ok(memory) = self.storage.get(memory_id) else {
                continue;
            };
            let Some(own) = memory
                .experience
                .embeddings
                .as_ref()
                .filter(|e| e.len() == index.config.dimension)
            else {
                continue;
            };
            sampled += 1;
            let agrees = index.vector(ids[0]).is_ok_and(|indexed| {
                crate::memory::hybrid_search::cosine_similarity_pub(own, &indexed) >= 0.999
            });
            if !agrees {
                mismatched += 1;
            }
        }
        (sampled, mismatched)
    }

    /// The saved index stops being authoritative the moment stored mappings start to change.
    fn invalidate_saved_index(&self) {
        self.mark_rewritten();
        let marker = self
            .storage_path
            .join("vector_index")
            .join(MAPPINGS_MARKER_FILE);
        if let Err(e) = fs::remove_file(&marker) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("Failed to remove {}: {}", marker.display(), e);
            }
        }
    }

    /// A fresh index for this engine, in memory: rebuilds replace the mapped vector file wholesale.
    fn fresh_index(&self) -> Result<VamanaIndex> {
        let current = self.vector_index.read();
        let config = VamanaConfig {
            use_mmap: false,
            ..current.config.clone()
        };
        VamanaIndex::with_storage_path(config, current.storage_path.clone())
    }

    /// Build the primary index from storage. A memory that embeds as one vector, its own, is built
    /// in with Vamana's batch construction and its mapping rewritten; a chunked memory, or one with
    /// no own vector, loses its mapping and is queued at open for the embedding backlog, whose
    /// drain re-chunks it.
    fn rebuild_primary_index(&self) -> Result<()> {
        use crate::embeddings::chunking::{chunk_text, ChunkConfig};

        let start = std::time::Instant::now();
        self.invalidate_saved_index();
        let dimension = self.embedder.primary_dimension();
        let chunk_config = ChunkConfig::default();
        let mapped: HashSet<MemoryId> = self
            .storage
            .get_all_vector_mappings()?
            .into_iter()
            .map(|(memory_id, _)| memory_id)
            .collect();

        let mut members = Vec::new();
        let mut vectors = Vec::new();
        let mut left_to_repair = 0usize;
        for memory_id in self.storage.get_all_ids()? {
            let Ok(memory) = self.storage.get(&memory_id) else {
                continue;
            };
            let own = memory
                .experience
                .embeddings
                .as_ref()
                .filter(|e| e.len() == dimension);
            let single = own.is_some()
                && !chunk_text(&Self::extract_searchable_text(&memory), &chunk_config).was_chunked;
            match own {
                Some(own) if single => {
                    vectors.push(own.clone());
                    members.push(memory_id);
                }
                _ => {
                    left_to_repair += 1;
                    if mapped.contains(&memory_id) {
                        if let Err(e) = self.storage.delete_vector_mapping(&memory_id) {
                            warn!("Failed to clear vector mapping for {}: {}", memory_id.0, e);
                        }
                    }
                }
            }
        }

        let mut index = self.fresh_index()?;
        index.build(vectors)?;
        let mappings: Vec<(MemoryId, Vec<u32>)> = members
            .into_iter()
            .enumerate()
            .map(|(vid, memory_id)| (memory_id, vec![vid as u32]))
            .collect();
        for batch in mappings.chunks(MAPPING_WRITE_BATCH) {
            self.storage.update_vector_mappings(batch)?;
        }
        let mut id_mapping = IdMapping::new();
        for (memory_id, ids) in mappings {
            id_mapping.insert_chunks(memory_id, ids);
        }

        let built = id_mapping.len();
        *self.vector_index.write() = index;
        *self.id_mapping.write() = id_mapping;
        info!(
            built,
            left_to_repair,
            secs = format!("{:.2}", start.elapsed().as_secs_f64()),
            "Rebuilt primary vector index from storage"
        );
        Ok(())
    }

    /// Rebuild the index from its own live vectors, dropping soft-deleted ones. Chunk vectors
    /// survive, so nothing is re-embedded; every memory's mapping is rewritten to the new ids.
    pub fn compact(&self) -> Result<()> {
        let start = std::time::Instant::now();
        self.invalidate_saved_index();
        let mut fresh = self.fresh_index()?;

        // LOCK ORDERING: vector_index (1) before id_mapping (2)
        let mut index = self.vector_index.write();
        let mut id_mapping = self.id_mapping.write();
        let mut members: Vec<(MemoryId, Vec<u32>)> = id_mapping
            .memory_to_vectors
            .iter()
            .map(|(memory_id, ids)| (memory_id.clone(), ids.clone()))
            .collect();
        members.sort_by(|a, b| a.0.cmp(&b.0));

        let before = index.len();
        let mut vectors = Vec::new();
        let mut remapped = IdMapping::new();
        for (memory_id, old_ids) in members {
            let mut new_ids = Vec::with_capacity(old_ids.len());
            for old in old_ids {
                vectors.push(index.vector(old)?);
                new_ids.push((vectors.len() - 1) as u32);
            }
            remapped.insert_chunks(memory_id, new_ids);
        }
        fresh.build(vectors)?;
        let mappings: Vec<(MemoryId, Vec<u32>)> = remapped
            .memory_to_vectors
            .iter()
            .map(|(memory_id, ids)| (memory_id.clone(), ids.clone()))
            .collect();
        for batch in mappings.chunks(MAPPING_WRITE_BATCH) {
            self.storage.update_vector_mappings(batch)?;
        }

        *index = fresh;
        *id_mapping = remapped;
        info!(
            before,
            after = index.len(),
            memories = id_mapping.len(),
            secs = format!("{:.2}", start.elapsed().as_secs_f64()),
            "Compacted primary vector index"
        );
        Ok(())
    }

    /// Attach the shared runtime dials so their vector-search overrides take effect. Pass the
    /// same `Arc` the owning `MemorySystem` holds.
    pub fn attach_dials(&self, dials: Arc<RwLock<crate::memory::dials::CognitiveDials>>) {
        *self.dials.write() = Some(dials);
    }

    /// The vector-search overrides the attached dials set now.
    fn vector_search_params(&self) -> VamanaSearchParams {
        self.dials
            .read()
            .as_ref()
            .map(|dials| {
                let dials = dials.read();
                VamanaSearchParams {
                    exact_search_max: dials.vector_exact_search_max,
                    beam_width: dials.vector_search_beam,
                }
            })
            .unwrap_or_default()
    }

    /// Set the consolidation event buffer (for late binding after construction)
    pub fn set_consolidation_events(&mut self, events: Arc<RwLock<ConsolidationEventBuffer>>) {
        self.consolidation_events = Some(events);
    }

    /// Save Vamana index to disk for instant startup
    ///
    /// HYBRID ARCHITECTURE:
    /// - RocksDB: Source of truth for memories and ID mappings
    /// - .vamana file: Persisted graph for instant startup (skip rebuild)
    ///
    /// On next startup, if .vamana exists and is valid, we load it directly.
    /// Otherwise, we fall back to rebuilding from RocksDB.
    pub fn save(&self) -> Result<()> {
        let _saving = self.save_lock.lock();
        let index_path = self.storage_path.join("vector_index");
        fs::create_dir_all(&index_path)?;

        let vamana_path = index_path.join(VAMANA_INDEX_FILE);
        // Changes counted before the save began are in what it writes; any counted while it
        // runs may not be, so only these are cleared when it succeeds.
        let pending = self
            .unsaved_changes
            .load(std::sync::atomic::Ordering::Acquire);
        let rewritten = self
            .rewritten_since_save
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        let save_began = std::time::Instant::now();
        let mut saved_all = true;

        // LOCK ORDERING: Always acquire vector_index (1) before id_mapping (2). The mapping
        // is only counted: holding it through the save would stall every insert.
        let vector_index = self.vector_index.read();
        let vector_count = self.id_mapping.read().len();
        if vector_count > 0 {
            // save_to_file replaces the file atomically: on a crash the old one survives.
            let started = std::time::Instant::now();
            match vector_index.save_to_file(&vamana_path) {
                Ok(()) => {
                    crate::metrics::VECTOR_INDEX_SAVE_DURATION
                        .with_label_values(&["primary"])
                        .observe(started.elapsed().as_secs_f64());
                    if let Err(e) = fs::write(index_path.join(MAPPINGS_MARKER_FILE), b"") {
                        warn!("Failed to write the vector mappings marker: {}", e);
                    }
                    info!(
                        "Saved Vamana index: {} memories to {} (instant startup enabled)",
                        vector_count,
                        vamana_path.display()
                    );
                }
                Err(e) => {
                    saved_all = false;
                    warn!(
                        "Failed to save Vamana index (will rebuild on restart): {}",
                        e
                    );
                }
            }
        } else {
            info!("Vamana index empty, skipping persistence");
        }
        // Release primary locks before acquiring secondary
        drop(vector_index);

        // Save secondary Vamana index if present
        // LOCK ORDERING: secondary_vector_index (3) before secondary_id_mapping (4)
        if let (Some(ref sec_index), Some(ref sec_mapping)) =
            (&self.secondary_vector_index, &self.secondary_id_mapping)
        {
            let sec_path = self.storage_path.join("secondary_vector_index");
            fs::create_dir_all(&sec_path)?;
            let sec_vamana_path = sec_path.join(SECONDARY_VAMANA_INDEX_FILE);

            let sec_idx = sec_index.read();
            if !sec_idx.is_empty() {
                // Read under the index lock, which holds inserts off for the save: a vector
                // is added before its mapping, so every entry read here names a vector this
                // snapshot contains.
                let entries: Vec<(MemoryId, Vec<u32>)> = sec_mapping
                    .read()
                    .memory_to_vectors
                    .iter()
                    .map(|(memory_id, vector_ids)| (memory_id.clone(), vector_ids.clone()))
                    .collect();
                let started = std::time::Instant::now();
                match sec_idx.save_to_file(&sec_vamana_path) {
                    Ok(()) => {
                        crate::metrics::VECTOR_INDEX_SAVE_DURATION
                            .with_label_values(&["secondary"])
                            .observe(started.elapsed().as_secs_f64());
                        match write_secondary_id_map(&sec_vamana_path, sec_idx.len(), &entries) {
                            Ok(()) => info!(
                                "Saved secondary Vamana index: {} vectors, {} memories",
                                sec_idx.len(),
                                entries.len()
                            ),
                            Err(e) => {
                                saved_all = false;
                                warn!(
                                    "Saved the secondary Vamana index but not its id map; the next \
                                     open rebuilds it: {}",
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        saved_all = false;
                        warn!(
                            "Failed to save secondary Vamana (will rebuild on restart): {}",
                            e
                        );
                    }
                }
            }
        }

        if saved_all {
            let left = match self.unsaved_changes.fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |count| Some(count.saturating_sub(pending)),
            ) {
                Ok(before) | Err(before) => before.saturating_sub(pending),
            };
            let mut since = self.unsaved_since.lock();
            *since = (left > 0 || self.rewritten_since_save.load(std::sync::atomic::Ordering::Acquire))
                .then_some(save_began);
        } else if rewritten {
            self.rewritten_since_save
                .store(true, std::sync::atomic::Ordering::Release);
        }

        Ok(())
    }

    /// Record that an index was rebuilt in memory, so the saved files describe none of it and
    /// the next [`Self::save_if_due`] saves whatever was counted.
    fn mark_rewritten(&self) {
        self.rewritten_since_save
            .store(true, std::sync::atomic::Ordering::Release);
        self.note_unsaved(0);
    }

    /// Count `changes` index changes toward the next save, starting the unsaved clock if
    /// everything was saved.
    fn note_unsaved(&self, changes: u64) {
        self.unsaved_changes
            .fetch_add(changes, std::sync::atomic::Ordering::AcqRel);
        self.unsaved_since
            .lock()
            .get_or_insert_with(std::time::Instant::now);
    }

    /// Save the index files when `policy` says enough has changed since the last save, or
    /// changes have waited long enough; returns whether it saved. The backlog drainer calls
    /// this on its tick, which makes the tick the timer.
    pub fn save_if_due(&self, policy: &IndexSavePolicy) -> Result<bool> {
        let unsaved_for = match *self.unsaved_since.lock() {
            Some(since) => since.elapsed(),
            None => return Ok(false),
        };
        let unsaved = self
            .unsaved_changes
            .load(std::sync::atomic::Ordering::Acquire);
        let rewritten = self
            .rewritten_since_save
            .load(std::sync::atomic::Ordering::Acquire);
        let vectors = self.vector_index.read().len();
        if !rewritten && !policy.is_due(unsaved, unsaved_for, vectors) {
            return Ok(false);
        }
        self.save()?;
        Ok(true)
    }

    /// Index changes not yet in the saved files.
    pub fn unsaved_changes(&self) -> u64 {
        self.unsaved_changes
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.id_mapping.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get set of all indexed memory IDs (for integrity checking)
    /// Whether the vector index holds this memory, without cloning the whole id set.
    /// The vectors the indexes hold for `memory_id`, read without its record: the primary
    /// when the memory is indexed as a single vector (a chunked memory's vectors are its
    /// chunks', not its record's embedding, and give `None`), and the secondary when one
    /// is indexed.
    pub fn indexed_vectors(&self, memory_id: &MemoryId) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
        fn single(mapping: &IdMapping, memory_id: &MemoryId) -> Option<u32> {
            match mapping.memory_to_vectors.get(memory_id).map(Vec::as_slice) {
                Some([only]) => Some(*only),
                _ => None,
            }
        }
        // Each mapping is read and released before its index is read, so the lock order
        // (index before mapping) is never inverted.
        let primary_id = single(&self.id_mapping.read(), memory_id);
        let primary = primary_id.and_then(|id| self.vector_index.read().vector(id).ok());
        let secondary = match (&self.secondary_vector_index, &self.secondary_id_mapping) {
            (Some(index), Some(mapping)) => {
                let secondary_id = single(&mapping.read(), memory_id);
                secondary_id.and_then(|id| index.read().vector(id).ok())
            }
            _ => None,
        };
        (primary, secondary)
    }

    pub fn is_indexed(&self, memory_id: &MemoryId) -> bool {
        self.id_mapping
            .read()
            .memory_to_vectors
            .contains_key(memory_id)
    }

    #[cfg(test)]
    pub(crate) fn vector_ids_of(&self, memory_id: &MemoryId) -> Vec<u32> {
        self.id_mapping
            .read()
            .memory_to_vectors
            .get(memory_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_indexed_memory_ids(&self) -> HashSet<MemoryId> {
        self.id_mapping
            .read()
            .memory_to_vectors
            .keys()
            .cloned()
            .collect()
    }

    /// Add memory to vector index with atomic RocksDB storage
    ///
    /// ATOMIC ARCHITECTURE: This method stores the vector mapping atomically
    /// in RocksDB alongside the memory data, ensuring no orphaned memories.
    ///
    /// For long content, this chunks the text and creates multiple embeddings
    /// to ensure ALL content is searchable, not just the first 256 tokens.
    /// Vector index inserts lost since process start. Non-zero means memories
    /// are stored and semantically unreachable — the orphan condition the
    /// open-time scan queues for the backlog, counted while it is accumulating rather than
    /// only after a restart notices.
    pub fn vector_index_failure_count(&self) -> u64 {
        self.vector_index_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn index_memory(&self, memory: &Memory) -> Result<()> {
        self.counting_index_failures(|| {
            let vector_ids = self.insert_vectors(memory)?;
            // ATOMIC: Store vector mapping in RocksDB
            // This ensures the mapping survives restarts and can't become orphaned
            self.storage
                .update_vector_mapping(&memory.id, vector_ids)
                .context("Failed to persist vector mapping to RocksDB")
        })
    }

    /// [`Self::index_memory`] without persisting the mapping: returns the text vector
    /// ids, for a caller that writes them in the same storage transaction as the
    /// memory itself (`PrimaryMemoryStore::store_with_vectors`).
    pub fn index_vectors(&self, memory: &Memory) -> Result<Vec<u32>> {
        self.counting_index_failures(|| self.insert_vectors(memory))
    }

    /// Counted at the boundary, not at the seven call sites.
    fn counting_index_failures<T>(&self, index: impl FnOnce() -> Result<T>) -> Result<T> {
        let outcome = index();
        if outcome.is_err() {
            self.vector_index_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        outcome
    }

    /// Embed `memory` into the in-memory primary and secondary indexes and return its
    /// primary vector ids; nothing is persisted.
    fn insert_vectors(&self, memory: &Memory) -> Result<Vec<u32>> {
        use crate::embeddings::chunking::{chunk_text, ChunkConfig};

        let text = Self::extract_searchable_text(memory);
        let chunk_config = ChunkConfig::default();
        let chunk_result = chunk_text(&text, &chunk_config);

        let vector_ids = if chunk_result.was_chunked {
            // Long content: embed each chunk separately
            // Pre-compute all embeddings OUTSIDE the write lock to avoid blocking searches
            let embeddings: Vec<Vec<f32>> = chunk_result
                .chunks
                .iter()
                .map(|chunk| {
                    self.embedder
                        .encode(chunk)
                        .context("Failed to generate chunk embedding")
                })
                .collect::<Result<Vec<_>>>()?;

            // Insert pre-computed vectors under a short write lock
            let mut ids = Vec::with_capacity(embeddings.len());
            let mut index = self.vector_index.write();
            for embedding in embeddings {
                let vector_id = index
                    .add_vector(embedding)
                    .context("Failed to add chunk vector to index")?;
                ids.push(vector_id);
            }
            drop(index);

            // Update in-memory mapping
            self.id_mapping
                .write()
                .insert_chunks(memory.id.clone(), ids.clone());

            tracing::debug!(
                "Indexed memory {} with {} chunks (original: {} chars)",
                memory.id.0,
                chunk_result.chunks.len(),
                chunk_result.original_length
            );

            ids
        } else {
            // Short content: single embedding (use pre-computed if available)
            let embedding = if let Some(emb) = &memory.experience.embeddings {
                emb.clone()
            } else {
                self.embedder
                    .encode(&text)
                    .context("Failed to generate embedding")?
            };

            let mut index = self.vector_index.write();
            let vector_id = index
                .add_vector(embedding)
                .context("Failed to add vector to index")?;

            // Update in-memory mapping
            self.id_mapping.write().insert(memory.id.clone(), vector_id);

            vec![vector_id]
        };

        // Index into secondary Vamana if available and memory has secondary embeddings
        if let (Some(ref sec_index), Some(ref sec_mapping)) =
            (&self.secondary_vector_index, &self.secondary_id_mapping)
        {
            if let Some(ref sec_emb) = memory.experience.embeddings_secondary {
                // A memory the secondary already holds keeps one vector there. The saved
                // secondary can hold a memory the saved primary lost (the two files are
                // written one after the other), and the repair that re-indexes the primary
                // comes through here.
                let previous: Vec<u32> = sec_mapping
                    .read()
                    .memory_to_vectors
                    .get(&memory.id)
                    .cloned()
                    .unwrap_or_default();
                // LOCK ORDERING: secondary_vector_index (3) before secondary_id_mapping (4)
                let added = {
                    let mut sec_idx = sec_index.write();
                    for &vid in &previous {
                        sec_idx.mark_deleted(vid);
                    }
                    sec_idx.add_vector(sec_emb.clone())
                };
                match added {
                    Ok(sec_vector_id) => {
                        sec_mapping.write().insert(memory.id.clone(), sec_vector_id);
                    }
                    Err(e) => {
                        if !previous.is_empty() {
                            sec_mapping.write().remove_all(&memory.id);
                        }
                        // Secondary failure is non-fatal — primary is authoritative
                        warn!("Failed to index secondary for {}: {}", memory.id.0, e);
                    }
                }
            }
        }

        self.note_unsaved(1);
        Ok(vector_ids)
    }

    /// Re-index an existing memory with updated embeddings
    ///
    /// Used when memory content is updated via upsert() to ensure the vector
    /// index reflects the new content.
    ///
    /// Strategy: Remove old vector and add new one (Vamana doesn't support update-in-place)
    pub fn reindex_memory(&self, memory: &Memory) -> Result<()> {
        // Check if memory is already indexed (may have multiple vectors from chunking)
        let existing_vector_ids = {
            let mapping = self.id_mapping.read();
            mapping
                .memory_to_vectors
                .get(&memory.id)
                .cloned()
                .unwrap_or_default()
        };

        if !existing_vector_ids.is_empty() {
            // Soft-delete old vectors in Vamana so they're excluded from search results
            // and counted toward the compaction threshold (30% deletion ratio triggers rebuild).
            // Without this, reindexed vectors become invisible ghost entries that waste
            // search candidate slots and never trigger compaction.
            {
                let index = self.vector_index.read();
                for &vid in &existing_vector_ids {
                    index.mark_deleted(vid);
                }
            }

            // Remove old ID mappings
            let mut mapping = self.id_mapping.write();
            mapping.memory_to_vectors.remove(&memory.id);
            for vector_id in existing_vector_ids {
                mapping.vector_to_memory.remove(&vector_id);
            }
        }

        // Clean up secondary index for this memory
        // LOCK ORDERING: secondary_vector_index (3) before secondary_id_mapping (4)
        if let (Some(ref sec_index), Some(ref sec_mapping)) =
            (&self.secondary_vector_index, &self.secondary_id_mapping)
        {
            // Peek at mapping to get vector IDs without removing yet
            let old_sec_ids: Vec<u32> = {
                let map = sec_mapping.read();
                map.memory_to_vectors
                    .get(&memory.id)
                    .cloned()
                    .unwrap_or_default()
            };
            if !old_sec_ids.is_empty() {
                // Lock 3 first: mark deleted in secondary index
                let sec_idx = sec_index.read();
                for vid in &old_sec_ids {
                    sec_idx.mark_deleted(*vid);
                }
                drop(sec_idx);
                // Lock 4 second: remove from secondary mapping
                sec_mapping.write().remove_all(&memory.id);
            }
        }

        // Add with new embedding (may create multiple chunks)
        // index_memory() also handles secondary indexing
        self.index_memory(memory)
    }

    /// Remove a memory from the vector index
    ///
    /// ATOMIC ARCHITECTURE: Removes the vector mapping from RocksDB atomically.
    /// The in-memory Vamana index is updated immediately, and the RocksDB mapping
    /// is deleted to ensure consistency on restart.
    ///
    /// Returns true if the memory was found and removed, false if not indexed.
    pub fn remove_memory(&self, memory_id: &MemoryId) -> bool {
        // Remove from in-memory ID mapping and get the vector IDs
        let vector_ids = self.id_mapping.write().remove_all(memory_id);

        if !vector_ids.is_empty() {
            // Mark vectors as deleted in Vamana (soft delete)
            let index = self.vector_index.read();
            for vid in &vector_ids {
                index.mark_deleted(*vid);
            }
            drop(index);

            // Remove from secondary index if present
            // LOCK ORDERING: secondary_vector_index (3) before secondary_id_mapping (4)
            if let (Some(ref sec_index), Some(ref sec_mapping)) =
                (&self.secondary_vector_index, &self.secondary_id_mapping)
            {
                // Peek at mapping to get vector IDs without removing yet
                let sec_ids: Vec<u32> = {
                    let map = sec_mapping.read();
                    map.memory_to_vectors
                        .get(memory_id)
                        .cloned()
                        .unwrap_or_default()
                };
                if !sec_ids.is_empty() {
                    // Lock 3 first: mark deleted in secondary index
                    let sec_idx = sec_index.read();
                    for vid in &sec_ids {
                        sec_idx.mark_deleted(*vid);
                    }
                    drop(sec_idx);
                    // Lock 4 second: remove from secondary mapping
                    sec_mapping.write().remove_all(memory_id);
                }
            }

            // NOTE: Memory graph edges are managed in GraphMemory at the API layer
            // GraphMemory handles cleanup via its own mechanisms

            self.note_unsaved(1);

            // ATOMIC: Remove vector mapping from RocksDB
            if let Err(e) = self.storage.delete_vector_mapping(memory_id) {
                tracing::warn!(
                    "Failed to delete vector mapping from RocksDB for {}: {}",
                    memory_id.0,
                    e
                );
            }

            tracing::debug!(
                "Removed memory {:?} from vector index ({} vectors)",
                memory_id,
                vector_ids.len()
            );
            true
        } else {
            tracing::debug!("Memory {:?} not found in vector index", memory_id);
            false
        }
    }

    /// Check if a memory ID still has a vector mapping (for post-deletion verification).
    pub fn has_mapping(&self, memory_id: &MemoryId) -> bool {
        self.id_mapping.read().contains(memory_id)
    }

    /// Extract searchable text from memory
    /// Build searchable text for embedding. Context is PREPENDED so that every
    /// chunk produced by the chunker carries origin information — "token expiry"
    /// in an auth project embeds differently from "token expiry" in a cache project.
    ///
    /// This is the single source of truth for embedding input. Both the synchronous
    /// `index_memory()` path and the deferred `embed_and_index()` path must use this.
    pub fn extract_searchable_text(memory: &Memory) -> String {
        // Build context prefix: "[project | topic | type] "
        let mut prefix_parts: Vec<&str> = Vec::new();

        // Collect borrowed references to context fields
        let mut project_name_storage = None;
        let mut topic_storage = None;

        if let Some(context) = &memory.experience.context {
            if let Some(name) = &context.project.name {
                project_name_storage = Some(name.as_str());
            }
            if let Some(topic) = &context.conversation.topic {
                topic_storage = Some(topic.as_str());
            }
        }

        if let Some(name) = project_name_storage {
            prefix_parts.push(name);
        }
        if let Some(topic) = topic_storage {
            prefix_parts.push(topic);
        }

        // Experience type as context signal
        let type_str = format!("{:?}", memory.experience.experience_type);

        if !type_str.is_empty() {
            prefix_parts.push(&type_str);
        }

        // Build: "[prefix] content entities outcomes emotion episode"
        let mut text = if !prefix_parts.is_empty() {
            let mut t = String::with_capacity(
                prefix_parts.iter().map(|p| p.len()).sum::<usize>()
                    + 6 // brackets + separators
                    + memory.experience.content.len()
                    + 128, // headroom
            );
            t.push('[');
            t.push_str(&prefix_parts.join(" | "));
            t.push_str("] ");
            t.push_str(&memory.experience.content);
            t
        } else {
            memory.experience.content.clone()
        };

        // Add entities
        if !memory.experience.entities.is_empty() {
            text.push(' ');
            text.push_str(&memory.experience.entities.join(" "));
        }

        // Add rich context fields that improve semantic matching
        if let Some(context) = &memory.experience.context {
            // Add recent conversation messages
            if !context.conversation.recent_messages.is_empty() {
                text.push(' ');
                text.push_str(&context.conversation.recent_messages.join(" "));
            }

            // SHO-104: Add emotional context - emotion labels improve semantic matching
            if let Some(emotion) = &context.emotional.dominant_emotion {
                text.push(' ');
                text.push_str(emotion);
            }

            // SHO-104: Add episode type for episodic grouping
            if let Some(episode_type) = &context.episode.episode_type {
                text.push(' ');
                text.push_str(episode_type);
            }
        }

        // Add outcomes
        if !memory.experience.outcomes.is_empty() {
            text.push(' ');
            text.push_str(&memory.experience.outcomes.join(" "));
        }

        text
    }

    /// Search for memory IDs only (for cache-aware retrieval)
    ///
    /// With chunked embeddings, multiple vectors can map to the same memory.
    /// This function deduplicates by MemoryId, keeping the highest-scoring chunk.
    ///
    /// Returns (MemoryId, similarity_score) pairs
    pub fn search_ids(&self, query: &Query, limit: usize) -> Result<Vec<(MemoryId, f32)>> {
        // BUG-006 FIX: Log warning for empty queries
        let query_embedding = if let Some(embedding) = &query.query_embedding {
            embedding.clone()
        } else if let Some(query_text) = &query.query_text {
            self.embedder
                .encode_for_query(query_text)
                .context("Failed to generate query embedding")?
        } else {
            tracing::warn!("Empty query in search_ids: no query_text or query_embedding provided");
            return Ok(Vec::new());
        };

        // TEMPORAL PRE-FILTER: If episode_id is provided, narrow search to that episode
        // This implements the architecture: Temporal → Graph → Semantic
        // Episode filtering happens FIRST to "point in the right direction"
        let episode_candidates: Option<HashSet<MemoryId>> =
            if let Some(episode_id) = &query.episode_id {
                let episode_memories = self
                    .storage
                    .search(SearchCriteria::ByEpisode(episode_id.clone()))?;
                if episode_memories.is_empty() {
                    tracing::debug!(
                        "No memories found in episode {}, falling back to global search",
                        episode_id
                    );
                    None
                } else {
                    tracing::debug!(
                        "Episode {} has {} memories, using as temporal filter",
                        episode_id,
                        episode_memories.len()
                    );
                    Some(episode_memories.into_iter().map(|m| m.id).collect())
                }
            } else {
                None
            };

        // Search vector index - fetch more candidates for chunk deduplication
        let params = self.vector_search_params();
        let index = self.vector_index.read();
        let results = crate::metrics::timed_vector_search(|| {
            index.search_with(
                &query_embedding,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
                params,
            )
        })
        .context("Vector search failed")?;

        // Map vector IDs to memory IDs, deduplicating by MemoryId (keep highest similarity)
        //
        // CRITICAL FIX: Vamana returns DISTANCE, not similarity.
        // For NormalizedDotProduct: distance = -dot(a,b)
        // - Similar vectors have dot ≈ 1.0, so distance ≈ -1.0
        // - Orthogonal vectors have dot ≈ 0.0, so distance ≈ 0.0
        // Convert: similarity = -distance (so similarity = dot product = cosine similarity)
        let id_mapping = self.id_mapping.read();
        let mut all_chunk_scores: std::collections::HashMap<MemoryId, Vec<f32>> =
            std::collections::HashMap::new();

        for (vector_id, distance) in results {
            // Convert distance to similarity: similarity = -distance
            // For NormalizedDotProduct, this gives us the actual dot product/cosine similarity
            let similarity = -distance;

            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                // TEMPORAL FILTER: If episode pre-filter is active, skip memories outside episode
                if let Some(ref candidates) = episode_candidates {
                    if !candidates.contains(memory_id) {
                        continue; // Skip - not in target episode
                    }
                }

                // Collect all chunk scores per memory for multi-chunk aggregation
                all_chunk_scores
                    .entry(memory_id.clone())
                    .or_default()
                    .push(similarity);
            }
        }

        // Multi-chunk score aggregation: a memory whose answer spans chunks 2-4
        // but whose best single chunk only scores 0.3 should outrank a memory with
        // one chunk at 0.35 and nothing else. When a memory has multiple high-scoring
        // chunks, this is evidence of distributed relevance across the content.
        let mut memory_ids: Vec<(MemoryId, f32)> = all_chunk_scores
            .into_iter()
            .map(|(id, mut scores)| {
                scores.sort_by(|a, b| b.total_cmp(a));
                let agg = if scores.len() >= 2 {
                    // Top-2 weighted average: 60% best + 40% second-best
                    // Rewards distributed relevance without over-rewarding noise
                    scores[0] * 0.6 + scores[1] * 0.4
                } else {
                    scores[0]
                };
                (id, agg)
            })
            .collect();
        // Tie-break by MemoryId for deterministic rank order across runs (RH-10).
        memory_ids.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        memory_ids.truncate(limit);

        Ok(memory_ids)
    }

    /// Search secondary Vamana index only (768d Nomic embeddings)
    ///
    /// Returns results from the secondary index, or empty vec if secondary
    /// is not configured or the query has no secondary embedding.
    /// Used as a 4th RRF signal in Layer 4 fusion.
    pub fn search_ids_secondary(
        &self,
        query_embedding_secondary: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let (sec_index, sec_mapping) =
            match (&self.secondary_vector_index, &self.secondary_id_mapping) {
                (Some(idx), Some(map)) => (idx, map),
                _ => return Ok(Vec::new()),
            };

        // LOCK ORDERING: secondary_vector_index (3) before secondary_id_mapping (4)
        let params = self.vector_search_params();
        let index = sec_index.read();
        let results = crate::metrics::timed_vector_search(|| {
            index.search_with(
                query_embedding_secondary,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
                params,
            )
        })
        .context("Secondary vector search failed")?;

        let id_mapping = sec_mapping.read();
        let mut best_scores: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        for (vector_id, distance) in results {
            let similarity = -distance; // NormalizedDotProduct: similarity = -distance
            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                best_scores
                    .entry(memory_id.clone())
                    .and_modify(|score| {
                        if similarity > *score {
                            *score = similarity;
                        }
                    })
                    .or_insert(similarity);
            }
        }

        // Tie-break by MemoryId for deterministic rank order across runs (RH-10).
        let mut memory_ids: Vec<(MemoryId, f32)> = best_scores.into_iter().collect();
        memory_ids.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        memory_ids.truncate(limit);

        Ok(memory_ids)
    }

    /// Search the *primary* Vamana index using a query text encoded with the
    /// secondary embedder and projected through the installed alignment.
    ///
    /// Returns an empty vec if no alignment is installed or no secondary
    /// embedder is configured — the existing max-score union path is the
    /// caller's fallback.
    ///
    /// Mirrors `search_ids_secondary`: distance is inverted to similarity,
    /// duplicate `MemoryId`s are kept at max similarity, ties are broken by id
    /// for deterministic ordering.
    pub fn search_ids_aligned(
        &self,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        use crate::embeddings::competitive::AlignedSide;

        let projected = match self
            .embedder
            .encode_aligned(query_text, AlignedSide::Query)?
        {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };
        self.search_ids_projected(&projected, limit)
    }

    /// [`Self::search_ids_aligned`] for a query already projected into primary space, so a
    /// recall that holds the secondary query embedding does not encode it again.
    pub fn search_ids_projected(
        &self,
        projected: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // LOCK ORDERING: primary vector_index (1) before id_mapping (2)
        let params = self.vector_search_params();
        let index = self.vector_index.read();
        let results = crate::metrics::timed_vector_search(|| {
            index.search_with(
                projected,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
                params,
            )
        })
        .context("Aligned vector search failed")?;

        let id_mapping = self.id_mapping.read();
        let mut best_scores: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        for (vector_id, distance) in results {
            // NormalizedDotProduct: similarity = -distance
            let similarity = -distance;
            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                best_scores
                    .entry(memory_id.clone())
                    .and_modify(|score| {
                        if similarity > *score {
                            *score = similarity;
                        }
                    })
                    .or_insert(similarity);
            }
        }

        // Tie-break by MemoryId for deterministic rank order (RH-10).
        let mut memory_ids: Vec<(MemoryId, f32)> = best_scores.into_iter().collect();
        memory_ids.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        memory_ids.truncate(limit);

        Ok(memory_ids)
    }

    /// Get memory from storage by ID
    pub fn get_from_storage(&self, id: &MemoryId) -> Result<Memory> {
        self.storage.get(id)
    }

    /// Read `ids` from storage, results in the same order. The one seam recall's candidate
    /// projection reads through: one batched read per call.
    pub fn get_many_from_storage(&self, ids: &[MemoryId]) -> Vec<Result<Memory>> {
        self.storage.get_many(ids)
    }

    /// Search for similar memories by embedding directly (SHO-106)
    ///
    /// Used for interference detection to find memories similar to a new memory.
    /// Optionally excludes a specific memory ID from results.
    ///
    /// With chunked embeddings, multiple vectors can map to the same memory.
    /// This function deduplicates by MemoryId, keeping the highest-scoring chunk.
    ///
    /// Returns (MemoryId, similarity_score) pairs
    pub fn search_by_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        exclude_id: Option<&MemoryId>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // Search vector index - fetch more candidates to account for chunk deduplication
        let params = self.vector_search_params();
        let index = self.vector_index.read();
        let results = crate::metrics::timed_vector_search(|| {
            index.search_with(
                embedding,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
                params,
            )
        })
        .context("Vector search by embedding failed")?;

        // Map vector IDs to memory IDs, deduplicating by MemoryId (keep highest similarity)
        //
        // CRITICAL FIX: Vamana returns DISTANCE, not similarity.
        // Convert: similarity = -distance (for NormalizedDotProduct)
        let id_mapping = self.id_mapping.read();
        let mut best_scores: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        for (vector_id, distance) in results {
            // Convert distance to similarity
            let similarity = -distance;

            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                // Skip excluded ID
                if let Some(exclude) = exclude_id {
                    if memory_id == exclude {
                        continue;
                    }
                }

                // Keep the highest similarity for each memory (best matching chunk)
                best_scores
                    .entry(memory_id.clone())
                    .and_modify(|score| {
                        if similarity > *score {
                            *score = similarity;
                        }
                    })
                    .or_insert(similarity);
            }
        }

        // Convert to vec and sort by similarity descending (highest first).
        // Tie-break by MemoryId for deterministic rank order across runs/CPUs (RH-10).
        let mut memory_ids: Vec<(MemoryId, f32)> = best_scores.into_iter().collect();
        memory_ids.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        memory_ids.truncate(limit);

        Ok(memory_ids)
    }

    /// Search for memories using multiple retrieval modes (zero-copy with Arc)
    pub fn search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let results = match query.retrieval_mode {
            // Standard modes
            RetrievalMode::Similarity => self.similarity_search(query, limit)?,
            RetrievalMode::Temporal => self.temporal_search(query, limit)?,
            RetrievalMode::Causal => self.causal_search(query, limit)?,
            RetrievalMode::Associative => self.associative_search(query, limit)?,
            RetrievalMode::Hybrid => self.hybrid_search(query, limit)?,
            // SpreadingActivation is handled at MemorySystem level (graph_retrieval module).
            // If it reaches RetrievalEngine, fall back to hybrid vector search.
            RetrievalMode::SpreadingActivation => self.hybrid_search(query, limit)?,
            // Robotics-specific modes
            RetrievalMode::Spatial => self.spatial_search(query, limit)?,
            RetrievalMode::Mission => self.mission_search(query, limit)?,
            RetrievalMode::ActionOutcome => self.action_outcome_search(query, limit)?,
        };

        Ok(results)
    }

    /// PRODUCTION: Similarity search using Vamana graph-based ANN (sub-millisecond, zero-copy)
    fn similarity_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        // BUG-006 FIX: Log warning for empty queries
        let query_embedding = if let Some(embedding) = &query.query_embedding {
            embedding.clone()
        } else if let Some(query_text) = &query.query_text {
            self.embedder
                .encode_for_query(query_text)
                .context("Failed to generate query embedding")?
        } else {
            tracing::warn!(
                "Empty query in similarity_search: no query_text or query_embedding provided"
            );
            return Ok(Vec::new());
        };

        // Search Vamana index with candidate multiplier for filtering headroom
        let params = self.vector_search_params();
        let index = self.vector_index.read();
        let results = crate::metrics::timed_vector_search(|| {
            index.search_with(
                &query_embedding,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
                params,
            )
        })
        .context("Vector search failed")?;

        // Map vector IDs to memory IDs and fetch memories.
        // Deduplicate by MemoryId: chunked memories produce multiple vectors that
        // all map to the same MemoryId; without dedup they consume multiple result slots.
        let id_mapping = self.id_mapping.read();
        let mut memories = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for (vector_id, _distance) in results {
            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                if !seen_ids.insert(memory_id.clone()) {
                    continue; // Already included this memory from a closer chunk
                }
                if let Ok(memory) = self.storage.get(memory_id) {
                    let shared_memory = Arc::new(memory);
                    if self.matches_filters(&shared_memory, query) {
                        memories.push(shared_memory);
                        if memories.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        Ok(memories)
    }

    /// Check if memory matches query filters
    ///
    /// Delegates to Query::matches() which is the SINGLE source of truth for all filter logic.
    /// This ensures consistent filtering across all memory tiers and retrieval modes.
    #[inline]
    pub fn matches_filters(&self, memory: &Memory, query: &Query) -> bool {
        query.matches(memory)
    }

    fn temporal_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        // TEMPORAL HIERARCHY:
        // 1. Episode (most specific) - same conversation/session
        // 2. Date range (fallback) - within time window

        let criteria = if let Some(episode_id) = &query.episode_id {
            // Episode-based temporal search: memories in same episode, ordered by sequence
            SearchCriteria::ByEpisodeSequence {
                episode_id: episode_id.clone(),
                min_sequence: None, // Get all in episode
                max_sequence: None,
            }
        } else if let Some((start, end)) = &query.time_range {
            // Date-based temporal search
            SearchCriteria::ByDate {
                start: *start,
                end: *end,
            }
        } else {
            // Default: last 7 days
            let end = chrono::Utc::now();
            let start = end - chrono::Duration::days(7);
            SearchCriteria::ByDate { start, end }
        };

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        memories.retain(|m| self.matches_filters(m, query));

        // Sort by sequence if episode-based, otherwise by created_at
        if query.episode_id.is_some() {
            // Episode search already returns in sequence order from storage
            // But verify ordering by sequence_number if available
            memories.sort_by(|a, b| {
                let seq_a = a
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.sequence_number)
                    .unwrap_or(0);
                let seq_b = b
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.sequence_number)
                    .unwrap_or(0);
                // Tie-break by MemoryId so memories without sequence_number sort stably (RH-10).
                seq_a.cmp(&seq_b).then_with(|| a.id.cmp(&b.id))
            });
        } else {
            // Newest first; tie-break by MemoryId on identical timestamps (RH-10).
            memories.sort_by(|a, b| {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
        }

        memories.truncate(limit);
        Ok(memories)
    }

    fn causal_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let seeds = self.similarity_search(query, 3)?;

        let mut results = HashSet::new();
        let mut to_explore = Vec::new();

        for seed in &seeds {
            to_explore.push(seed.id.clone());
            results.insert(seed.id.clone());
        }

        while !to_explore.is_empty() && results.len() < limit {
            if let Some(current_id) = to_explore.pop() {
                if let Ok(memory) = self.storage.get(&current_id) {
                    for related_id in &memory.experience.related_memories {
                        if !results.contains(related_id) {
                            results.insert(related_id.clone());
                            to_explore.push(related_id.clone());
                        }
                    }
                }
            }
        }

        let mut memories = Vec::new();
        for id in results.into_iter().take(limit) {
            if let Ok(memory) = self.storage.get(&id) {
                memories.push(Arc::new(memory));
            }
        }

        Ok(memories)
    }

    fn associative_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        // NOTE: Associative search now uses GraphMemory at the API layer
        // GraphMemory.find_memory_associations() provides Hebbian-weighted associations
        // This method falls back to similarity search as a baseline
        self.similarity_search(query, limit)
    }

    fn hybrid_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let mut all_results: HashMap<MemoryId, SharedMemory> = HashMap::new();
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        // Weight for each retrieval mode (tuned for robotics)
        let weights = [
            (RetrievalMode::Similarity, 0.5),  // Higher weight for semantic
            (RetrievalMode::Temporal, 0.2),    // Recent memories important
            (RetrievalMode::Causal, 0.2),      // Context chains
            (RetrievalMode::Associative, 0.1), // Associations
        ];

        for (mode, weight) in weights.iter() {
            let mut mode_query = query.clone();
            mode_query.retrieval_mode = mode.clone();

            let results = match mode {
                RetrievalMode::Similarity => self.similarity_search(&mode_query, limit),
                RetrievalMode::Temporal => self.temporal_search(&mode_query, limit),
                RetrievalMode::Causal => self.causal_search(&mode_query, limit),
                RetrievalMode::Associative => self.associative_search(&mode_query, limit),
                _ => continue,
            };

            if let Ok(memories) = results {
                for (rank, memory) in memories.into_iter().enumerate() {
                    // Rank score: higher rank = higher score
                    let score = weight * (1.0 / (rank as f32 + 1.0));

                    // Clone ID before moving memory into HashMap to avoid double clone
                    let memory_id = memory.id.clone();
                    *scores.entry(memory_id.clone()).or_insert(0.0) += score;
                    all_results.insert(memory_id, memory);
                }
            }
        }

        // Apply Ebbinghaus salience scoring: combines retrieval score with time-based relevance
        // This ensures older, less-accessed memories naturally fade in ranking
        let mut sorted: Vec<(f32, SharedMemory)> = all_results
            .into_iter()
            .map(|(id, memory)| {
                let retrieval_score = scores.get(&id).copied().unwrap_or(0.0);
                // Salience score factors in recency (Ebbinghaus curve) and access frequency
                let salience = memory.salience_score_with_access();
                // Final score: 70% retrieval relevance, 30% salience (time-based decay)
                let final_score = retrieval_score * 0.7 + salience * 0.3;
                (final_score, memory)
            })
            .collect();

        // Score desc → recency desc → MemoryId asc for deterministic ranking (RH-10).
        sorted.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.created_at.cmp(&a.1.created_at))
                .then_with(|| a.1.id.cmp(&b.1.id))
        });

        Ok(sorted.into_iter().take(limit).map(|(_, m)| m).collect())
    }

    // ========================================================================
    // ROBOTICS-SPECIFIC RETRIEVAL MODES
    // ========================================================================

    /// Spatial search: Find memories within geographic radius
    /// Uses haversine distance for accurate earth-surface calculations
    fn spatial_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let geo_filter = query
            .geo_filter
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Spatial search requires geo_filter"))?;

        let criteria = SearchCriteria::ByLocation {
            lat: geo_filter.lat,
            lon: geo_filter.lon,
            radius_meters: geo_filter.radius_meters,
        };

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        // Apply additional filters
        memories.retain(|m| self.matches_filters(m, query));

        // Sort by distance (closest first). Read via resolved_geo so both
        // Place::Geo (W3 WhereFacet) and the legacy flat geo_location field
        // count toward the sort distance.
        memories.sort_by(|a, b| {
            let dist_a = match a.experience.resolved_geo() {
                Some((lat, lon, _alt)) => geo_filter.haversine_distance(lat, lon),
                None => f64::MAX,
            };
            let dist_b = match b.experience.resolved_geo() {
                Some((lat, lon, _alt)) => geo_filter.haversine_distance(lat, lon),
                None => f64::MAX,
            };
            // Tie-break: recency desc, then MemoryId asc — deterministic across runs (RH-10).
            dist_a
                .total_cmp(&dist_b)
                .then_with(|| b.created_at.cmp(&a.created_at))
                .then_with(|| a.id.cmp(&b.id))
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Mission search: Retrieve all memories from a specific mission
    /// Useful for mission replay, analysis, and learning
    fn mission_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let mission_id = query
            .mission_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Mission search requires mission_id"))?;

        let criteria = SearchCriteria::ByMission(mission_id.clone());

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        // Apply additional filters
        memories.retain(|m| self.matches_filters(m, query));

        // Sort by timestamp (chronological order for mission replay), tie-break by id (RH-10).
        memories.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Action-outcome search: Find memories with specific reward outcomes
    /// For reinforcement learning: "What actions led to positive rewards?"
    fn action_outcome_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        // Get reward range or default to positive rewards
        let (min_reward, max_reward) = query.reward_range.unwrap_or((0.0, 1.0));

        let criteria = SearchCriteria::ByReward {
            min: min_reward,
            max: max_reward,
        };

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        // Apply additional filters (action_type, robot_id, etc.)
        memories.retain(|m| self.matches_filters(m, query));

        // Sort by reward (highest first for learning from best outcomes)
        memories.sort_by(|a, b| {
            let reward_a = a.experience.reward.unwrap_or(0.0);
            let reward_b = b.experience.reward.unwrap_or(0.0);
            // Tie-break: recency desc, then MemoryId asc — deterministic across runs (RH-10).
            reward_b
                .total_cmp(&reward_a)
                .then_with(|| b.created_at.cmp(&a.created_at))
                .then_with(|| a.id.cmp(&b.id))
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Build vector index from existing memories (resumable on failure)
    ///
    /// Uses incremental indexing so partial progress is preserved:
    /// - Skips memories already in the index
    /// - On failure, next rebuild/repair continues from where it left off
    /// - Logs progress every 1000 memories for monitoring
    pub fn rebuild_index(&self) -> Result<()> {
        // Phase 1: Collect only memory IDs (16 bytes each — bounded even at 10M)
        let all_ids = self.storage.get_all_ids()?;
        let total = all_ids.len();

        if total == 0 {
            tracing::info!("No memories to index");
            return Ok(());
        }

        tracing::info!("Starting resumable index rebuild: {} memories", total);

        // Get already-indexed memory IDs to skip
        let indexed_ids = self.get_indexed_memory_ids();
        let already_indexed = indexed_ids.len();

        let mut indexed = 0;
        let mut skipped = 0;
        let mut failed = 0;
        let start_time = std::time::Instant::now();

        // Phase 2: Process one memory at a time — O(1) peak memory per iteration
        for (i, memory_id) in all_ids.iter().enumerate() {
            // Skip already indexed memories (makes rebuild resumable)
            if indexed_ids.contains(memory_id) {
                skipped += 1;
            } else {
                // Load single memory from RocksDB, index it, then drop
                match self.storage.get(memory_id) {
                    Ok(memory) => {
                        if memory.is_forgotten() {
                            skipped += 1;
                        } else {
                            match self.index_memory(&memory) {
                                Ok(_) => indexed += 1,
                                Err(e) => {
                                    failed += 1;
                                    tracing::warn!(
                                        "Failed to index memory {} during rebuild: {}",
                                        memory_id.0,
                                        e
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        tracing::warn!(
                            "Failed to load memory {} during rebuild: {}",
                            memory_id.0,
                            e
                        );
                    }
                }
            }

            // Log progress every 1000 memories
            if (i + 1) % 1000 == 0 || i + 1 == total {
                let elapsed = start_time.elapsed().as_secs();
                let rate = if elapsed > 0 {
                    (indexed + skipped) as f64 / elapsed as f64
                } else {
                    0.0
                };
                tracing::info!(
                    "Rebuild progress: {}/{} ({:.1}%), {} indexed, {} skipped, {} failed, {:.0}/sec",
                    i + 1,
                    total,
                    (i + 1) as f64 / total as f64 * 100.0,
                    indexed,
                    skipped,
                    failed,
                    rate
                );
            }
        }

        tracing::info!(
            "Index rebuild complete: {} indexed, {} already present, {} failed (total: {})",
            indexed,
            already_indexed + skipped,
            failed,
            self.len()
        );

        Ok(())
    }

    /// Re-embed all memories with fresh context-prefixed embeddings.
    ///
    /// Migration path for APP-1: existing memories have context-free embeddings
    /// from before v0.7.2. This clears each memory's pre-computed embedding,
    /// forcing `index_memory()` to re-generate it through `extract_searchable_text()`
    /// which now prepends `[project | topic | type]` context prefixes.
    ///
    /// Uses `reindex_memory()` which soft-deletes old vectors and creates new ones.
    /// Safe to call while the system is serving queries (reads from stale vectors
    /// until the new ones are indexed).
    pub fn reembed_all(&self) -> Result<(usize, usize)> {
        let all_ids = self.storage.get_all_ids()?;
        let total = all_ids.len();

        if total == 0 {
            tracing::info!("No memories to re-embed");
            return Ok((0, 0));
        }

        tracing::info!(
            "Starting context-prefix re-embedding migration: {} memories",
            total
        );
        let start_time = std::time::Instant::now();
        let mut reembedded = 0;
        let mut failed = 0;

        for (i, memory_id) in all_ids.iter().enumerate() {
            match self.storage.get(memory_id) {
                Ok(mut memory) => {
                    if memory.is_forgotten() {
                        continue;
                    }

                    // Clear pre-computed embedding to force re-generation
                    // through extract_searchable_text() → encode()
                    memory.experience.embeddings = None;

                    // Persist the cleared embedding so deferred path also re-generates
                    if let Err(e) = self.storage.update(&memory) {
                        tracing::warn!("Failed to clear embedding for {}: {e}", memory_id.0);
                        failed += 1;
                        continue;
                    }

                    // Re-index: removes old vectors, creates new ones with context prefix
                    match self.reindex_memory(&memory) {
                        Ok(_) => reembedded += 1,
                        Err(e) => {
                            tracing::warn!("Failed to re-embed {}: {e}", memory_id.0);
                            failed += 1;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load {}: {e}", memory_id.0);
                    failed += 1;
                }
            }

            if (i + 1) % 500 == 0 || i + 1 == total {
                let elapsed = start_time.elapsed().as_secs();
                let rate = if elapsed > 0 {
                    reembedded as f64 / elapsed as f64
                } else {
                    0.0
                };
                tracing::info!(
                    "Re-embed progress: {}/{} ({:.1}%), {} done, {} failed, {:.0}/sec",
                    i + 1,
                    total,
                    (i + 1) as f64 / total as f64 * 100.0,
                    reembedded,
                    failed,
                    rate
                );
            }
        }

        tracing::info!(
            "Re-embed migration complete: {} re-embedded, {} failed (total: {})",
            reembedded,
            failed,
            total
        );

        Ok((reembedded, failed))
    }

    // NOTE: Memory graph functionality has been consolidated into GraphMemory
    // which is managed at the API layer (MultiUserMemoryManager.graph_memories)
    // The following methods are preserved for API compatibility but are no-ops:
    // - add_to_graph() - GraphMemory handles entity relationships from NER
    // - record_coactivation() - Now called directly on GraphMemory in API handlers
    // - graph_maintenance() - GraphMemory.apply_decay() handles this
    // - graph_stats() - Returns empty stats (use GraphMemory.get_stats() instead)

    /// Add memory to knowledge graph - DEPRECATED
    /// Use GraphMemory at the API layer instead
    #[deprecated(note = "Use GraphMemory at API layer instead")]
    pub fn add_to_graph(&self, _memory: &Memory) {
        // No-op: GraphMemory handles entity relationships from NER
    }

    /// Record co-activation of memories - DEPRECATED
    /// Use GraphMemory.record_memory_coactivation() at API layer instead
    #[deprecated(note = "Use GraphMemory.record_memory_coactivation() at API layer instead")]
    pub fn record_coactivation(&self, _memory_ids: &[MemoryId]) {
        // No-op: Coactivation is now recorded in GraphMemory at the API layer
    }

    /// Perform graph maintenance - DEPRECATED
    /// Use GraphMemory.apply_decay() at API layer instead
    #[deprecated(note = "Use GraphMemory.apply_decay() at API layer instead")]
    pub fn graph_maintenance(&self) {
        // No-op: GraphMemory handles decay in its own maintenance cycle
    }

    /// Get memory graph statistics - DEPRECATED
    /// Use GraphMemory.get_stats() at API layer instead
    #[deprecated(note = "Use GraphMemory.get_stats() at API layer instead")]
    pub fn graph_stats(&self) -> MemoryGraphStats {
        // Return empty stats - real stats are in GraphMemory
        MemoryGraphStats {
            node_count: 0,
            edge_count: 0,
            avg_strength: 0.0,
            potentiated_count: 0,
        }
    }

    /// Compact the vector index when deletions or incremental inserts call for it; returns
    /// whether it did. Compaction rewrites the id mapping with the index, never one without the
    /// other, so it does not use Vamana's own `auto_rebuild_if_needed()`.
    pub fn auto_rebuild_index_if_needed(&self) -> Result<bool> {
        // Check if rebuild is needed (read lock only)
        {
            let index = self.vector_index.read();
            if !index.needs_rebuild() || index.is_rebuilding() {
                return Ok(false);
            }
        }

        self.compact()?;
        Ok(true)
    }

    /// Get vector index degradation info
    pub fn index_health(&self) -> IndexHealth {
        let index = self.vector_index.read();
        let secondary = self.secondary_vector_index.as_ref().map(|sec| {
            let sec_idx = sec.read();
            SecondaryIndexHealth {
                total_vectors: sec_idx.len(),
                incremental_inserts: sec_idx.incremental_insert_count(),
                deleted_count: sec_idx.deleted_count(),
                deletion_ratio: sec_idx.deletion_ratio(),
                needs_rebuild: sec_idx.needs_rebuild(),
                needs_compaction: sec_idx.needs_compaction(),
            }
        });
        IndexHealth {
            total_vectors: index.len(),
            incremental_inserts: index.incremental_insert_count(),
            deleted_count: index.deleted_count(),
            deletion_ratio: index.deletion_ratio(),
            needs_rebuild: index.needs_rebuild(),
            needs_compaction: index.needs_compaction(),
            rebuild_threshold: crate::vector_db::vamana::REBUILD_THRESHOLD,
            deletion_ratio_threshold: crate::vector_db::vamana::DELETION_RATIO_THRESHOLD,
            secondary,
        }
    }
}

/// Health information about the secondary vector index
#[derive(Debug, Clone)]
pub struct SecondaryIndexHealth {
    pub total_vectors: usize,
    pub incremental_inserts: usize,
    pub deleted_count: usize,
    pub deletion_ratio: f32,
    pub needs_rebuild: bool,
    pub needs_compaction: bool,
}

/// Health information about the vector index
#[derive(Debug, Clone)]
pub struct IndexHealth {
    pub total_vectors: usize,
    pub incremental_inserts: usize,
    pub deleted_count: usize,
    pub deletion_ratio: f32,
    pub needs_rebuild: bool,
    pub needs_compaction: bool,
    pub rebuild_threshold: usize,
    pub deletion_ratio_threshold: f32,
    pub secondary: Option<SecondaryIndexHealth>,
}

// ============================================================================
// OUTCOME FEEDBACK SYSTEM - Hebbian "Fire Together, Wire Together"
// ============================================================================

/// Outcome of a retrieval operation - used to reinforce or weaken memories
///
/// When memories are retrieved and used to complete a task, this feedback
/// tells the system whether they were helpful, enabling adaptive learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RetrievalOutcome {
    /// Memory helped complete the task successfully
    /// Triggers: +importance boost, +association strength, +access count
    Helpful,
    /// Memory was misleading or caused errors
    /// Triggers: -importance penalty, relationship weakening
    Misleading,
    /// Memory was retrieved but not actionably useful
    /// Triggers: +access count only (neutral)
    #[default]
    Neutral,
    /// I8 (R2/LTR): the retrieval abstained — surfaced but the system declined to
    /// act on it (e.g. confidence-gated below threshold). A negative training label
    /// for the future learned ranker; behaviourally treated as Neutral by live
    /// reinforcement (no importance change), captured only for the trace store.
    Abstained,
    /// I8 (R2/LTR): the memory was returned but the user/agent ignored it entirely.
    /// Like `Abstained`, this is a negative label for the learned ranker captured to
    /// the trace store; live reinforcement treats it as Neutral (no score change).
    Ignored,
}

/// Result of a retrieval with tracking for feedback
#[derive(Debug, Clone)]
pub struct TrackedRetrieval {
    /// The memories that were retrieved
    pub memories: Vec<SharedMemory>,
    /// Unique ID for this retrieval (for later feedback)
    pub retrieval_id: String,
    /// Query that produced these results
    pub query_fingerprint: u64,
    /// Timestamp of retrieval
    pub retrieved_at: chrono::DateTime<chrono::Utc>,
}

impl TrackedRetrieval {
    fn new(memories: Vec<SharedMemory>, query: &Query) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if let Some(text) = &query.query_text {
            text.hash(&mut hasher);
        }

        Self {
            memories,
            retrieval_id: uuid::Uuid::new_v4().to_string(),
            query_fingerprint: hasher.finish(),
            retrieved_at: chrono::Utc::now(),
        }
    }

    /// Get memory IDs for feedback
    pub fn memory_ids(&self) -> Vec<MemoryId> {
        self.memories.iter().map(|m| m.id.clone()).collect()
    }
}

/// Feedback record for a retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalFeedback {
    /// Which retrieval this feedback is for
    pub retrieval_id: String,
    /// The outcome
    pub outcome: RetrievalOutcome,
    /// Optional task context (what was the user trying to do)
    pub task_context: Option<String>,
    /// When feedback was provided
    pub feedback_at: chrono::DateTime<chrono::Utc>,
}

impl RetrievalEngine {
    // ========================================================================
    // OUTCOME FEEDBACK METHODS
    // ========================================================================

    /// Search with tracking for later feedback
    ///
    /// Use this when you want to provide feedback on retrieval quality.
    /// Returns a TrackedRetrieval that can be used with `reinforce_recall`.
    pub fn search_tracked(&self, query: &Query, limit: usize) -> Result<TrackedRetrieval> {
        let memories = self.search(query, limit)?;
        Ok(TrackedRetrieval::new(memories, query))
    }

    /// Reinforce memories based on task outcome (core feedback loop)
    ///
    /// This is THE key method that closes the Hebbian loop:
    /// - If outcome is Helpful: strengthen associations, boost importance
    /// - If outcome is Misleading: weaken associations, reduce importance
    /// - If outcome is Neutral: just record access (mild reinforcement)
    ///
    /// Call this after a task completes to indicate which memories helped.
    pub fn reinforce_recall(
        &self,
        memory_ids: &[MemoryId],
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        if memory_ids.is_empty() {
            return Ok(ReinforcementStats::default());
        }

        let mut stats = ReinforcementStats {
            memories_processed: memory_ids.len(),
            ..Default::default()
        };

        // Hebbian coactivation: count pair associations for non-misleading outcomes
        if !matches!(outcome, RetrievalOutcome::Misleading) && memory_ids.len() >= 2 {
            let n = memory_ids.len();
            stats.associations_strengthened = n * (n - 1) / 2;
        }

        match outcome {
            RetrievalOutcome::Helpful => {
                // Boost importance of helpful memories and PERSIST to storage
                for id in memory_ids {
                    if let Ok(memory) = self.storage.get(id) {
                        // Increment access and apply importance boost
                        memory.record_access();
                        memory.boost_importance(0.05); // +5% importance
                                                       // Bayesian confidence calibration
                        memory.update_confidence(true);

                        // PERSIST: Write updated memory back to durable storage
                        if self.storage.update(&memory).is_ok() {
                            stats.importance_boosts += 1;
                        }
                    }
                }
            }
            RetrievalOutcome::Misleading => {
                // Reduce importance of misleading memories and PERSIST to storage
                for id in memory_ids {
                    if let Ok(memory) = self.storage.get(id) {
                        memory.record_access();
                        memory.decay_importance(0.10); // -10% importance
                                                       // Bayesian confidence calibration
                        memory.update_confidence(false);

                        // PERSIST: Write updated memory back to durable storage
                        if self.storage.update(&memory).is_ok() {
                            stats.importance_decays += 1;
                        }
                    }
                }
                // Don't strengthen associations for misleading memories
            }
            // I8 (R2/LTR): Abstained/Ignored are negative TRAINING labels for the
            // future learned ranker, captured to the trace store elsewhere. For LIVE
            // reinforcement they are SHADOW-neutral — same access-only behaviour as
            // Neutral, so adding the labels changes no importance/confidence.
            RetrievalOutcome::Neutral | RetrievalOutcome::Abstained | RetrievalOutcome::Ignored => {
                // Just record access, mild reinforcement - PERSIST to storage
                for id in memory_ids {
                    if let Ok(memory) = self.storage.get(id) {
                        memory.record_access();

                        // PERSIST: Write access update to storage
                        if let Err(e) = self.storage.update(&memory) {
                            tracing::warn!(
                                "Failed to persist access update for memory {}: {}",
                                id.0,
                                e
                            );
                        }
                    }
                }
                // Association strengthening for neutral outcomes is counted above (pair counting)
            }
        }

        stats.outcome = outcome;
        Ok(stats)
    }

    /// Reinforce using a tracked retrieval (convenience wrapper)
    pub fn reinforce_tracked(
        &self,
        tracked: &TrackedRetrieval,
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        let ids = tracked.memory_ids();
        self.reinforce_recall(&ids, outcome)
    }

    /// Batch reinforce multiple retrievals (for async feedback processing)
    pub fn reinforce_batch(
        &self,
        feedbacks: &[RetrievalFeedback],
        retrieval_memories: &HashMap<String, Vec<MemoryId>>,
    ) -> Result<Vec<ReinforcementStats>> {
        let mut results = Vec::with_capacity(feedbacks.len());

        for feedback in feedbacks {
            if let Some(memory_ids) = retrieval_memories.get(&feedback.retrieval_id) {
                let stats = self.reinforce_recall(memory_ids, feedback.outcome)?;
                results.push(stats);
            }
        }

        Ok(results)
    }
}

/// Statistics from a reinforcement operation
#[derive(Debug, Clone, Default)]
pub struct ReinforcementStats {
    /// How many memories were processed
    pub memories_processed: usize,
    /// How many association edges were strengthened
    pub associations_strengthened: usize,
    /// How many importance boosts were applied
    pub importance_boosts: usize,
    /// How many importance decays were applied
    pub importance_decays: usize,
    /// The outcome that triggered this reinforcement
    pub outcome: RetrievalOutcome,
    /// How many persistence operations failed (non-zero indicates data loss risk)
    pub persist_failures: usize,
}

// NOTE: MemoryGraph has been consolidated into GraphMemory (src/graph_memory.rs)
// which provides persistent storage in RocksDB and proper Hebbian learning.
// All graph-based memory associations now go through GraphMemory at the API layer.

/// Statistics about the memory graph (for backwards compatibility)
///
/// Real statistics are available from GraphMemory.get_stats()
#[derive(Debug, Clone, Default)]
pub struct MemoryGraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_strength: f32,
    pub potentiated_count: usize,
}

// ============================================================================
// ANTICIPATORY PREFETCH - Context-aware cache warming
// ============================================================================

/// Context signals used to anticipate which memories will be needed
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefetchContext {
    /// Current project/workspace being worked on
    pub project_id: Option<String>,
    /// Current file path being edited
    pub current_file: Option<String>,
    /// Recent entities mentioned
    pub recent_entities: Vec<String>,
    /// Current time of day (for temporal patterns)
    pub hour_of_day: Option<u32>,
    /// Day of week (0=Sunday)
    pub day_of_week: Option<u32>,
    /// Recent query patterns (for predictive prefetch)
    pub recent_queries: Vec<String>,
    /// Current task type (coding, debugging, reviewing)
    pub task_type: Option<String>,

    // SHO-104: Episode context for episodic prefetching
    /// Current episode ID - memories in same episode are highly relevant
    pub episode_id: Option<String>,
    /// Current emotional valence - for mood-congruent retrieval
    pub emotional_valence: Option<f32>,
}

impl PrefetchContext {
    /// Create context from RichContext
    pub fn from_rich_context(ctx: &super::types::RichContext) -> Self {
        Self {
            project_id: ctx.project.project_id.clone(),
            current_file: ctx.code.current_file.clone(),
            recent_entities: ctx.conversation.mentioned_entities.clone(),
            hour_of_day: ctx
                .temporal
                .time_of_day
                .as_ref()
                .and_then(|t| t.parse().ok()),
            day_of_week: ctx
                .temporal
                .day_of_week
                .as_ref()
                .and_then(|d| match d.as_str() {
                    "Sunday" => Some(0),
                    "Monday" => Some(1),
                    "Tuesday" => Some(2),
                    "Wednesday" => Some(3),
                    "Thursday" => Some(4),
                    "Friday" => Some(5),
                    "Saturday" => Some(6),
                    _ => None,
                }),
            recent_queries: Vec::new(),
            task_type: ctx.project.current_task.clone(),
            // SHO-104: Episode and emotional context
            episode_id: ctx.episode.episode_id.clone(),
            emotional_valence: if ctx.emotional.valence != 0.0 {
                Some(ctx.emotional.valence)
            } else {
                None
            },
        }
    }

    /// Create context from current system state
    pub fn from_current_time() -> Self {
        let now = chrono::Utc::now();
        Self {
            hour_of_day: Some(now.hour()),
            day_of_week: Some(now.weekday().num_days_from_sunday()),
            ..Default::default()
        }
    }
}

/// Result of a prefetch operation
#[derive(Debug, Clone, Default)]
pub struct PrefetchResult {
    /// Memory IDs that were prefetched
    pub prefetched_ids: Vec<MemoryId>,
    /// Why these memories were selected
    pub reason: PrefetchReason,
    /// How many were already cached
    pub cache_hits: usize,
    /// How many were fetched from storage
    pub fetches: usize,
}

/// Reason for prefetching specific memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum PrefetchReason {
    /// Project-based: memories from same project
    Project(String),
    /// File-based: memories about related files
    RelatedFiles,
    /// Entity-based: memories mentioning same entities
    SharedEntities,
    /// Temporal: memories from similar time patterns
    TemporalPattern,
    /// Association: strongly associated with recent memories
    AssociatedMemories,
    /// Predicted: predicted from query patterns
    QueryPrediction,
    #[default]
    /// Unknown or multiple reasons
    Mixed,
}

/// Anticipatory prefetch engine
///
/// Pre-warms the memory cache based on contextual signals:
/// - Project: "I'm working on auth module" → prefetch auth-related memories
/// - File: "I opened user.rs" → prefetch memories about user.rs and imports
/// - Temporal: "It's Monday morning" → prefetch Monday morning patterns
/// - Association: "I just accessed memory A" → prefetch A's strong associations
pub struct AnticipatoryPrefetch {
    /// Maximum memories to prefetch at once
    max_prefetch: usize,
}

impl Default for AnticipatoryPrefetch {
    fn default() -> Self {
        Self::new()
    }
}

impl AnticipatoryPrefetch {
    pub fn new() -> Self {
        Self { max_prefetch: 20 }
    }

    /// Create with custom prefetch limit
    pub fn with_limit(max_prefetch: usize) -> Self {
        Self { max_prefetch }
    }

    /// Generate prefetch query based on context
    ///
    /// This is the main entry point - given a context, determine what memories
    /// are likely to be needed soon and return a query to fetch them.
    pub fn generate_prefetch_query(&self, context: &PrefetchContext) -> Option<Query> {
        // Priority 1: Project-based prefetch (strongest signal)
        if let Some(project_id) = &context.project_id {
            return Some(self.project_query(project_id));
        }

        // Priority 2: Entity-based prefetch
        if !context.recent_entities.is_empty() {
            return Some(self.entity_query(&context.recent_entities));
        }

        // Priority 3: File-based prefetch
        if let Some(file_path) = &context.current_file {
            return Some(self.file_query(file_path));
        }

        // Priority 4: Temporal pattern prefetch
        if let (Some(hour), Some(day)) = (context.hour_of_day, context.day_of_week) {
            return Some(self.temporal_query(hour, day));
        }

        None
    }

    /// Generate query for project-related memories
    fn project_query(&self, project_id: &str) -> Query {
        Query {
            query_text: Some(format!("project:{}", project_id)),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Similarity,
            ..Default::default()
        }
    }

    /// Generate query for entity-related memories
    fn entity_query(&self, entities: &[String]) -> Query {
        let query_text = entities.join(" ");
        Query {
            query_text: Some(query_text),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Similarity,
            ..Default::default()
        }
    }

    /// Generate query for file-related memories
    fn file_query(&self, file_path: &str) -> Query {
        // Extract filename for broader search
        let filename = std::path::Path::new(file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file_path);

        Query {
            query_text: Some(format!("file {} code", filename)),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Similarity,
            ..Default::default()
        }
    }

    /// Generate query for temporal pattern matching
    fn temporal_query(&self, hour: u32, _day: u32) -> Query {
        let now = chrono::Utc::now();

        // Find similar time window
        let start_hour = hour.saturating_sub(PREFETCH_TEMPORAL_WINDOW_HOURS as u32);
        let end_hour = (hour + PREFETCH_TEMPORAL_WINDOW_HOURS as u32).min(23);

        // Calculate time range for today at similar hours
        let start = now
            .with_hour(start_hour)
            .unwrap_or(now)
            .with_minute(0)
            .unwrap_or(now);
        let end = now
            .with_hour(end_hour)
            .unwrap_or(now)
            .with_minute(59)
            .unwrap_or(now);

        Query {
            time_range: Some((start, end)),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Temporal,
            ..Default::default()
        }
    }

    // NOTE: association_prefetch_ids has been removed.
    // Association-based prefetching should use GraphMemory.find_memory_associations()
    // at the API layer where GraphMemory is available.

    /// Score how relevant a memory is to the current context
    ///
    /// Higher scores mean more likely to be needed soon.
    pub fn relevance_score(&self, memory: &Memory, context: &PrefetchContext) -> f32 {
        let mut score = 0.0;

        // Project match (strong signal)
        if let Some(project_id) = &context.project_id {
            if let Some(ctx) = &memory.experience.context {
                if ctx.project.project_id.as_ref() == Some(project_id) {
                    score += 0.4;
                }
            }
        }

        // Entity overlap
        let memory_entities: HashSet<_> = memory.experience.entities.iter().collect();
        let context_entities: HashSet<_> = context.recent_entities.iter().collect();
        let overlap = memory_entities.intersection(&context_entities).count();
        if overlap > 0 {
            score += 0.2 * (overlap as f32 / context_entities.len().max(1) as f32);
        }

        // File relevance
        if let Some(current_file) = &context.current_file {
            if memory.experience.content.contains(current_file) {
                score += 0.2;
            }
            // Also check related files in context
            if let Some(ctx) = &memory.experience.context {
                if ctx.code.related_files.iter().any(|f| f == current_file) {
                    score += 0.1;
                }
            }
        }

        // Temporal relevance (same hour of day)
        if let Some(hour) = context.hour_of_day {
            let memory_hour = memory.created_at.hour();
            if (memory_hour as i32 - hour as i32).abs() <= PREFETCH_TEMPORAL_WINDOW_HOURS {
                score += 0.1;
            }
        }

        // Recency boost (using centralized constants)
        let age_hours = (chrono::Utc::now() - memory.created_at).num_hours();
        if age_hours < PREFETCH_RECENCY_FULL_HOURS {
            score += PREFETCH_RECENCY_FULL_BOOST;
        } else if age_hours < PREFETCH_RECENCY_PARTIAL_HOURS {
            score += PREFETCH_RECENCY_PARTIAL_BOOST;
        }

        // SHO-104: Emotional arousal boost - high-arousal memories are more salient
        // Research: Emotionally arousing events are better remembered (LaBar & Cabeza, 2006)
        if let Some(ctx) = &memory.experience.context {
            // High arousal memories get a relevance boost
            if ctx.emotional.arousal > 0.6 {
                score += 0.1 * ctx.emotional.arousal;
            }

            // Source credibility affects relevance - more credible = more relevant
            if ctx.source.credibility > 0.8 {
                score += 0.05;
            }

            // Episode context: same episode = highly relevant
            if let Some(current_episode) = &context.episode_id {
                if ctx.episode.episode_id.as_ref() == Some(current_episode) {
                    score += 0.3; // Strong boost for same-episode memories
                }
            }

            // Mood-congruent retrieval: similar emotional valence boosts relevance
            // Research: We recall happy memories when happy, sad when sad
            if let Some(current_valence) = context.emotional_valence {
                let valence_diff = (ctx.emotional.valence - current_valence).abs();
                if valence_diff < 0.3 {
                    // Same emotional valence quadrant
                    score += 0.1 * (1.0 - valence_diff / 0.3);
                }
            }
        }

        score.min(1.0)
    }
}

use chrono::{Datelike, Timelike};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_mapping_basic() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        mapping.insert(memory_id.clone(), 42);

        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get_memory_id(42), Some(&memory_id));
    }

    #[test]
    fn test_id_mapping_chunks() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        mapping.insert_chunks(memory_id.clone(), vec![1, 2, 3]);

        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get_memory_id(1), Some(&memory_id));
        assert_eq!(mapping.get_memory_id(2), Some(&memory_id));
        assert_eq!(mapping.get_memory_id(3), Some(&memory_id));
    }

    #[test]
    fn test_id_mapping_remove_all() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        mapping.insert_chunks(memory_id.clone(), vec![1, 2, 3]);
        let removed = mapping.remove_all(&memory_id);

        assert_eq!(removed.len(), 3);
        assert_eq!(mapping.len(), 0);
        assert!(mapping.get_memory_id(1).is_none());
    }

    #[test]
    fn test_id_mapping_clear() {
        let mut mapping = IdMapping::new();
        mapping.insert(MemoryId(uuid::Uuid::new_v4()), 1);
        mapping.insert(MemoryId(uuid::Uuid::new_v4()), 2);

        mapping.clear();

        assert_eq!(mapping.len(), 0);
    }

    #[test]
    fn test_retrieval_outcome_default() {
        let outcome = RetrievalOutcome::default();
        assert_eq!(outcome, RetrievalOutcome::Neutral);
    }

    #[test]
    fn test_reinforcement_stats_default() {
        let stats = ReinforcementStats::default();

        assert_eq!(stats.memories_processed, 0);
        assert_eq!(stats.associations_strengthened, 0);
        assert_eq!(stats.importance_boosts, 0);
        assert_eq!(stats.importance_decays, 0);
    }

    #[test]
    fn test_memory_graph_stats_default() {
        let stats = MemoryGraphStats::default();

        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.avg_strength, 0.0);
        assert_eq!(stats.potentiated_count, 0);
    }

    #[test]
    fn test_prefetch_context_default() {
        let ctx = PrefetchContext::default();

        assert!(ctx.project_id.is_none());
        assert!(ctx.current_file.is_none());
        assert!(ctx.recent_entities.is_empty());
    }

    #[test]
    fn test_prefetch_context_from_current_time() {
        let ctx = PrefetchContext::from_current_time();

        assert!(ctx.hour_of_day.is_some());
        assert!(ctx.day_of_week.is_some());
    }

    #[test]
    fn test_anticipatory_prefetch_new() {
        let prefetch = AnticipatoryPrefetch::new();
        assert_eq!(prefetch.max_prefetch, 20);
    }

    #[test]
    fn test_anticipatory_prefetch_with_limit() {
        let prefetch = AnticipatoryPrefetch::with_limit(50);
        assert_eq!(prefetch.max_prefetch, 50);
    }

    #[test]
    fn test_generate_prefetch_query_project() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            project_id: Some("my-project".to_string()),
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        assert!(query.query_text.unwrap().contains("my-project"));
    }

    #[test]
    fn test_generate_prefetch_query_entities() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            recent_entities: vec!["Rust".to_string(), "memory".to_string()],
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        let text = query.query_text.unwrap();
        assert!(text.contains("Rust"));
        assert!(text.contains("memory"));
    }

    #[test]
    fn test_generate_prefetch_query_file() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            current_file: Some("/src/memory/retrieval.rs".to_string()),
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        assert!(query.query_text.unwrap().contains("retrieval.rs"));
    }

    #[test]
    fn test_generate_prefetch_query_temporal() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            hour_of_day: Some(14),
            day_of_week: Some(1),
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        assert!(query.time_range.is_some());
    }

    #[test]
    fn test_generate_prefetch_query_empty() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext::default();

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_none());
    }

    #[test]
    fn test_prefetch_reason_default() {
        let reason = PrefetchReason::default();
        assert!(matches!(reason, PrefetchReason::Mixed));
    }

    #[test]
    fn test_prefetch_result_default() {
        let result = PrefetchResult::default();

        assert!(result.prefetched_ids.is_empty());
        assert_eq!(result.cache_hits, 0);
        assert_eq!(result.fetches, 0);
    }

    #[test]
    fn test_index_health_struct() {
        let health = IndexHealth {
            total_vectors: 1000,
            incremental_inserts: 100,
            deleted_count: 50,
            deletion_ratio: 0.05,
            needs_rebuild: false,
            needs_compaction: false,
            rebuild_threshold: 500,
            deletion_ratio_threshold: 0.2,
            secondary: None,
        };

        assert_eq!(health.total_vectors, 1000);
        assert!(!health.needs_rebuild);
    }

    #[test]
    fn test_id_mapping_insert_is_idempotent() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        // First insert
        mapping.insert(memory_id.clone(), 10);
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get_memory_id(10), Some(&memory_id));

        // Second insert for same memory with different vector_id (simulates re-index)
        mapping.insert(memory_id.clone(), 20);
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get_memory_id(20), Some(&memory_id));
        // Old vector_id should be cleaned up (no orphan)
        assert!(
            mapping.get_memory_id(10).is_none(),
            "old vector_id should be removed to prevent orphan"
        );
        assert_eq!(mapping.memory_to_vectors[&memory_id], vec![20]);
    }

    #[test]
    fn test_id_mapping_insert_chunks_is_idempotent() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        // First insert: 3 chunks
        mapping.insert_chunks(memory_id.clone(), vec![1, 2, 3]);
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.memory_to_vectors[&memory_id], vec![1, 2, 3]);

        // Second insert: 2 different chunks (simulates re-index after content change)
        mapping.insert_chunks(memory_id.clone(), vec![10, 11]);
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.memory_to_vectors[&memory_id], vec![10, 11]);
        // Old vector_ids should be cleaned up
        assert!(mapping.get_memory_id(1).is_none(), "old chunk 1 orphaned");
        assert!(mapping.get_memory_id(2).is_none(), "old chunk 2 orphaned");
        assert!(mapping.get_memory_id(3).is_none(), "old chunk 3 orphaned");
        // New ones should be present
        assert_eq!(mapping.get_memory_id(10), Some(&memory_id));
        assert_eq!(mapping.get_memory_id(11), Some(&memory_id));
    }

    #[test]
    fn test_id_mapping_vector_count_stable_after_reinsert() {
        let mut mapping = IdMapping::new();
        let m1 = MemoryId(uuid::Uuid::new_v4());
        let m2 = MemoryId(uuid::Uuid::new_v4());

        mapping.insert(m1.clone(), 1);
        mapping.insert(m2.clone(), 2);
        assert_eq!(mapping.vector_to_memory.len(), 2);

        // Re-insert m1 with new vector - total vector count should stay at 2
        mapping.insert(m1.clone(), 3);
        assert_eq!(
            mapping.vector_to_memory.len(),
            2,
            "vector_to_memory should not grow on re-insert"
        );
    }
}

#[cfg(test)]
mod corpus_action_tests {
    use super::{corpus_action, CorpusAction};

    #[test]
    fn the_case_that_was_broken_same_dimension_different_model() {
        // THE defect. Both are 384, so the old check saw `recorded == target` and
        // returned — over a corpus embedded by something else entirely. The length
        // check downstream could not see it either: right shape, wrong content.
        assert_eq!(
            corpus_action(Some(384), Some("hash-bigram"), 384, "minilm"),
            CorpusAction::MigrateAll,
            "an embedder swap at one dimension must redo everything — length identifies nothing"
        );
    }

    #[test]
    fn a_matching_pair_is_up_to_date_and_costs_nothing() {
        assert_eq!(
            corpus_action(Some(384), Some("minilm"), 384, "minilm"),
            CorpusAction::UpToDate
        );
    }

    #[test]
    fn a_dimension_change_uses_the_length_check() {
        // Still the cheap path: a 768 corpus against a 384 embedder is identifiable
        // per memory, so there is no reason to redo the ones already correct.
        assert_eq!(
            corpus_action(Some(768), Some("nomic"), 384, "minilm"),
            CorpusAction::MigrateChangedLength
        );
        // And it wins even when the id also changed — length is the sharper signal.
        assert_eq!(
            corpus_action(Some(768), Some("hash-bigram"), 384, "minilm"),
            CorpusAction::MigrateChangedLength
        );
    }

    #[test]
    fn an_unrecorded_embedder_is_neither_current_nor_wrong() {
        // The corpus that exists today: dimension matches, nothing says what wrote it.
        // Claiming UpToDate asserts a provenance nobody established; MigrateAll
        // re-embeds thousands of vectors on a suspicion. It gets its own answer.
        assert_eq!(
            corpus_action(Some(384), None, 384, "minilm"),
            CorpusAction::RecordUnverified
        );
    }

    #[test]
    fn unverified_does_not_read_as_current_on_the_next_startup() {
        // The sentinel has to be a value that never equals a real id, or the next
        // run reads it as a match and the unknown silently becomes a claim.
        assert_eq!(
            corpus_action(Some(384), Some("unverified:minilm"), 384, "minilm"),
            CorpusAction::MigrateAll,
            "an unverified marker must resolve, not persist as a permanent maybe"
        );
    }

    #[test]
    fn a_fresh_store_takes_the_migrate_path_and_the_caller_short_circuits_on_empty() {
        // No markers at all. The decision is MigrateChangedLength; the caller then
        // finds zero memories and records both markers without work. Asserted so a
        // later reader does not "fix" this into UpToDate, which would record the
        // markers for a corpus nobody looked at.
        assert_eq!(
            corpus_action(None, None, 384, "minilm"),
            CorpusAction::MigrateChangedLength
        );
    }
}

#[cfg(test)]
mod persisted_secondary_map_tests {
    use super::*;

    fn saved_index(dir: &Path, vectors: usize) -> (PathBuf, VamanaIndex) {
        saved_index_scaled(dir, vectors, 1.0)
    }

    fn saved_index_scaled(dir: &Path, vectors: usize, scale: f32) -> (PathBuf, VamanaIndex) {
        let config = VamanaConfig {
            dimension: 4,
            use_mmap: false,
            ..Default::default()
        };
        let mut index = VamanaIndex::new(config).unwrap();
        let basis: Vec<Vec<f32>> = (0..vectors)
            .map(|i| {
                let mut v = vec![0.1; 4];
                v[i % 4] = scale * (1.0 + i as f32);
                v
            })
            .collect();
        index.build(basis).unwrap();
        let path = dir.join(SECONDARY_VAMANA_INDEX_FILE);
        index.save_to_file(&path).unwrap();
        (path, index)
    }

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    /// The map comes back as saved, minus entries for vectors the file deleted, and a vector
    /// no entry names is retired rather than left to answer searches for nobody.
    #[test]
    fn a_map_saved_with_its_index_reads_back_against_that_index() {
        let dir = tempfile::tempdir().unwrap();
        let (path, index) = saved_index(dir.path(), 5);
        index.mark_deleted(3);
        index.save_to_file(&path).unwrap();
        let entries = vec![(id(1), vec![0]), (id(2), vec![4, 1]), (id(3), vec![3])];
        write_secondary_id_map(&path, index.len(), &entries).unwrap();

        let loaded = VamanaIndex::load_from_file(&path).unwrap();
        let map = read_secondary_id_map(&path, &loaded).unwrap().expect("map");
        assert_eq!(map.memory_to_vectors.get(&id(1)), Some(&vec![0]));
        assert_eq!(map.memory_to_vectors.get(&id(2)), Some(&vec![4, 1]));
        assert!(!map.contains(&id(3)), "its only vector was deleted");
        assert_eq!(map.get_memory_id(4), Some(&id(2)));
        assert!(loaded.is_deleted(2), "vector 2 belongs to no memory");
        assert!(!loaded.is_deleted(0));
    }

    /// A map is only trusted beside the file it was written with: a later save of the index
    /// that did not rewrite the map leaves a map whose ids may name other vectors.
    #[test]
    fn a_map_from_another_save_is_not_used() {
        let dir = tempfile::tempdir().unwrap();
        let (path, index) = saved_index(dir.path(), 4);
        write_secondary_id_map(&path, index.len(), &[(id(1), vec![0])]).unwrap();

        // The same vector count, so only the checksum tells the two files apart.
        let (_, rebuilt) = saved_index_scaled(dir.path(), 4, 2.0);
        let loaded = VamanaIndex::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), index.len());
        assert_eq!(rebuilt.len(), index.len());
        assert!(read_secondary_id_map(&path, &loaded).unwrap().is_none());

        std::fs::remove_file(path.with_file_name(SECONDARY_ID_MAP_FILE)).unwrap();
        assert!(read_secondary_id_map(&path, &loaded).unwrap().is_none());
    }

    #[test]
    fn a_truncated_map_is_an_error_not_a_partial_map() {
        let dir = tempfile::tempdir().unwrap();
        let (path, index) = saved_index(dir.path(), 4);
        write_secondary_id_map(&path, index.len(), &[(id(1), vec![0]), (id(2), vec![1])]).unwrap();
        let map_path = path.with_file_name(SECONDARY_ID_MAP_FILE);
        let bytes = std::fs::read(&map_path).unwrap();
        std::fs::write(&map_path, &bytes[..bytes.len() - 2]).unwrap();
        let loaded = VamanaIndex::load_from_file(&path).unwrap();
        assert!(read_secondary_id_map(&path, &loaded).is_err());
    }

    #[test]
    fn a_save_falls_due_on_changes_scaled_to_the_index_or_on_age() {
        let policy = IndexSavePolicy::default();
        let fresh = std::time::Duration::from_secs(1);
        assert!(!policy.is_due(0, std::time::Duration::from_secs(86_400), 10));
        assert!(!policy.is_due(999, fresh, 10));
        assert!(policy.is_due(1_000, fresh, 10));
        assert!(
            !policy.is_due(4_999, fresh, 100_000),
            "5% of a 100k-vector index is 5000 changes"
        );
        assert!(policy.is_due(5_000, fresh, 100_000));
        assert!(policy.is_due(1, policy.max_unsaved_age, 100_000));
    }

    #[test]
    fn the_save_policy_reads_positive_whole_numbers_from_the_environment() {
        let vars = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| value.to_string())
            }
        };
        let tuned = IndexSavePolicy::from_lookup(vars(&[
            ("VELD_VECTOR_INDEX_SAVE_CHANGES", "250"),
            ("VELD_VECTOR_INDEX_SAVE_INTERVAL_SECS", "60"),
        ]));
        assert_eq!(tuned.min_changes, 250);
        assert_eq!(tuned.max_unsaved_age, std::time::Duration::from_secs(60));
        let ignored = IndexSavePolicy::from_lookup(vars(&[
            ("VELD_VECTOR_INDEX_SAVE_CHANGES", "0"),
            ("VELD_VECTOR_INDEX_SAVE_INTERVAL_SECS", "soon"),
        ]));
        assert_eq!(ignored, IndexSavePolicy::default());
    }
}
