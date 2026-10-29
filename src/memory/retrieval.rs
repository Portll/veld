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

use super::embedding_backlog::{BacklogKey, EmbeddingBacklogStore};
use super::introspection::ConsolidationEventBuffer;
pub(crate) use super::space_index::IdMapping;
use super::space_index::{ExactVectors, SpaceCodec, SpaceIndex, SpaceIndexHealth};
use super::space_legs::{self, Leg, LegDepth, LegHits, LegIndex, LegKind};
use super::space_migration::{MigrationKind, MigrationStep, SpaceMigration, SpaceMigrationStore};
use super::storage::SearchCriteria;
use super::types::*;
use crate::constants::{
    PREFETCH_RECENCY_FULL_BOOST, PREFETCH_RECENCY_FULL_HOURS, PREFETCH_RECENCY_PARTIAL_BOOST,
    PREFETCH_RECENCY_PARTIAL_HOURS, PREFETCH_TEMPORAL_WINDOW_HOURS,
    VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
};
use crate::embeddings::competitive::CompetitiveEmbedder;
use crate::embeddings::space::{self, SpaceId};
use crate::embeddings::Embedder;
use crate::vector_db::vamana::{VamanaConfig, VamanaIndex};

/// Filename for persisted Vamana index (instant startup)
const VAMANA_INDEX_FILE: &str = "vamana.idx";
/// Present beside a saved index when every stored vector mapping refers to it. Each path that
/// assigns vector ids writes its mappings, and rebuilds remove the marker until the next save.
pub(crate) const MAPPINGS_MARKER_FILE: &str = "mappings.authoritative";
/// Where the secondary index lived before every non-primary space got its own directory under
/// `space_indexes/`. Removed at start: a cache of the old layout, never read again.
const LEGACY_SECONDARY_INDEX: (&str, &str) = ("secondary_vector_index", "secondary_vamana.idx");

/// An index refused a stored vector because it is in another space than the index's.
///
/// The refusal is already a backlog row targeting the index's space when this is returned, so
/// a caller neither warns nor records it again; see [`is_space_mismatch`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("memory {} stores a vector in {stored}; the index holds {index}", .memory.0)]
pub struct SpaceMismatch {
    pub memory: MemoryId,
    pub stored: SpaceId,
    pub index: SpaceId,
}

/// Whether an indexing error is a space refusal the engine has already recorded.
pub fn is_space_mismatch(error: &anyhow::Error) -> bool {
    error.downcast_ref::<SpaceMismatch>().is_some()
}

/// Multi-modal retrieval engine with production vector search
///
/// # Lock Ordering (SHO-72)
///
/// To prevent deadlocks, locks MUST be acquired in ascending order:
///
/// 1. `vector_index` - Primary vector similarity search index (384d)
/// 2. `id_mapping` - Primary memory ID ↔ vector ID mapping
/// 3. `space_indexes` - one index per non-primary space. The map's own lock is held only to
///    clone an index out (or, rarely, to add one); each index then locks its index before its
///    mapping, never two spaces at once
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
    /// Lock order: 3 - One index per space other than the primary's, by space. Today the
    /// secondary embedder's space; any space a query can reach has one.
    space_indexes: RwLock<std::collections::BTreeMap<SpaceId, Arc<SpaceIndex>>>,
    /// How every space index but the primary holds its vectors.
    space_codec: SpaceCodec,
    /// The space every vector in `vector_index` is in.
    index_space: SpaceId,
    /// The space the secondary embedder writes, a key of `space_indexes` when configured. The
    /// memory's secondary slot is read as a vector in this space.
    secondary_index_space: Option<SpaceId>,
    /// Where a refusal is recorded: the ledger the memory system drains, over the same store.
    backlog: EmbeddingBacklogStore,
    /// The distribution of each space's best search similarity, learned from its searches and
    /// seeded from the space registry at start.
    score_stats: RwLock<std::collections::HashMap<SpaceId, space::ScoreStats>>,
    /// How deep into each space's recall legs the returned results lay; sets the legs'
    /// candidate budgets.
    leg_depths: RwLock<std::collections::HashMap<SpaceId, LegDepth>>,
    /// Storage path for persisting vector index and ID mapping
    storage_path: PathBuf,
    /// Lock order: 5 - Acquire last
    /// Shared consolidation event buffer for introspection
    /// Records edge formation, strengthening, and pruning events
    consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
}

/// What a full re-embed did. `rewritten` holds every memory whose stored record changed,
/// which a journaling caller MUST journal as an `Update`, or followers keep the old vectors.
#[derive(Debug, Clone, Default)]
pub struct ReembedReport {
    pub reembedded: usize,
    pub failed: usize,
    pub rewritten: Vec<MemoryId>,
}

/// What retiring a space removed.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RetireReport {
    pub space: SpaceId,
    pub vectors_dropped: usize,
    pub maps_removed: usize,
}

/// Put `vector` in the memory's primary slot, keeping the vector it replaces beside it.
fn replace_primary(memory: &mut Memory, vector: Vec<f32>, space: SpaceId) {
    // The slot's space is read off the vector in it, so it is read before the vector leaves.
    let displaced_space = memory.experience.primary_space(None);
    let displaced = memory.experience.embeddings.take().zip(displaced_space);
    memory.experience.set_primary_vector(vector, space);
    if let Some((old, old_space)) = displaced {
        memory.experience.keep_vector(old_space, old);
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
            ..Default::default()
        };

        let vamana_storage = storage_path.join("vector_index");
        std::fs::create_dir_all(&vamana_storage)?;
        let vector_index = VamanaIndex::with_storage_path(vamana_config, Some(vamana_storage))
            .context("Failed to initialize Vamana vector index")?;
        let id_mapping = IdMapping::new();

        // One index per space a query can reach besides the primary's: the secondary
        // embedder's, when one is configured.
        let index_space = embedder.primary_space();
        let secondary_index_space = embedder
            .secondary_space()
            .filter(|space| *space != index_space);
        let space_codec = SpaceCodec::from_env();
        let mut space_indexes = std::collections::BTreeMap::new();
        if let Some(space) = &secondary_index_space {
            space_indexes.insert(
                space.clone(),
                Arc::new(
                    SpaceIndex::open(&storage_path, space.clone(), space_codec)
                        .with_context(|| format!("Failed to initialize the {space} index"))?,
                ),
            );
            info!("Space index initialized: {space}");
        }
        // A space a fitted map reaches from a space this engine writes is searchable too: its
        // index holds the memories' stored vectors in it, and queries arrive through the map.
        let corpus_bridges = crate::embeddings::alignment_spaces::corpus_alignments(&storage_path);
        let producible: Vec<SpaceId> = std::iter::once(index_space.clone())
            .chain(secondary_index_space.clone())
            .collect();
        for (from, to, path, _) in &corpus_bridges {
            if producible.contains(from)
                && !producible.contains(to)
                && !space_indexes.contains_key(to)
            {
                space_indexes.insert(
                    to.clone(),
                    Arc::new(
                        SpaceIndex::open(&storage_path, to.clone(), space_codec)
                            .with_context(|| format!("Failed to initialize the {to} index"))?,
                    ),
                );
                info!(
                    "Space index initialized: {to}, reached from {from} through {}",
                    path.display()
                );
            }
        }
        let backlog = EmbeddingBacklogStore::with_kv(storage.key_value_store());
        let engine = Self {
            vector_index_failures: std::sync::atomic::AtomicU64::new(0),
            storage,
            embedder,
            vector_index: Arc::new(RwLock::new(vector_index)),
            id_mapping: Arc::new(RwLock::new(id_mapping)),
            space_indexes: RwLock::new(space_indexes),
            space_codec,
            leg_depths: RwLock::new(std::collections::HashMap::new()),
            score_stats: RwLock::new(
                space::SpaceRegistry::read(&storage_path)
                    .ok()
                    .flatten()
                    .map(|registry| {
                        registry
                            .spaces
                            .into_iter()
                            .filter_map(|s| s.scores.map(|stats| (s.id, stats)))
                            .collect()
                    })
                    .unwrap_or_default(),
            ),
            index_space,
            secondary_index_space,
            backlog,
            storage_path,
            consolidation_events,
        };

        // ATOMIC STARTUP: Rebuild Vamana from RocksDB (single source of truth)
        engine.rebuild_from_rocksdb()?;

        for (from, to, path, alignment) in corpus_bridges {
            let confidence = crate::embeddings::alignment_spaces::confidence_of(alignment.as_ref());
            match engine
                .embedder
                .install_bridge(from.clone(), to.clone(), alignment, confidence)
            {
                Ok(()) => info!("Installed the {from} → {to} map from {}", path.display()),
                Err(e) => warn!("{}: {e}", path.display()),
            }
        }

        // Alignment onboarding: load (or autofit, if VELD_ALIGNMENT_AUTOFIT=1)
        // a learned projection from secondary→primary space. No-op for
        // primary-only installs. Errors are downgraded to a warning so an
        // unreadable alignment file doesn't take down the engine.
        if let (Some(secondary), _) = (
            engine.embedder.secondary_embedder(),
            engine.embedder.has_alignment(),
        ) {
            // Keyed by the spaces the map joins, not the models' names: two widths of one
            // model are two spaces, and a map fitted for one is wrong for the other.
            let pair_id = match engine.embedder.secondary_space() {
                Some(from) => {
                    crate::embeddings::alignment_spaces::pair_id_between(&from, &engine.index_space)
                }
                None => crate::embeddings::AlignmentPairId::new(
                    engine.embedder.primary_embedder().model_id(),
                    secondary.model_id(),
                ),
            };
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

    /// Decide at start what the corpus's spaces need, and leave the work to the drain.
    ///
    /// - **Current and stamped**: O(1), nothing to do.
    /// - **Never stamped**: a `Stamp` migration names the space of every vector that has none,
    ///   in the background. Until it reaches a vector, that vector reads as the index's at the
    ///   index's width, as every vector did before spaces were recorded.
    /// - **Embedder changed**: a `Reembed` migration re-embeds every primary vector not in the
    ///   active space, in the background, keeping the vector it replaces. The markers advance
    ///   now: the migration record says what is pending, so a restart resumes it rather than
    ///   starting another. A change of model at the SAME width over a corpus never stamped is
    ///   stamped here first, before the markers move, because at one width a vector with no
    ///   space would otherwise read as the new model's.
    ///
    /// Never names a model it was not told: a width the markers do not explain is recorded as
    /// `unrecorded@<width>`.
    fn reconcile_corpus_spaces(&self) -> Result<()> {
        let target = self.embedder.primary_dimension();
        let target_id = self.embedder.primary_id().to_string();
        let target_space = self.index_space.clone();
        let marker = self.storage_path.join(space::DIMENSION_MARKER);
        let id_marker = self.storage_path.join(space::EMBEDDER_MARKER);
        let stamp_marker = self.storage_path.join(space::STAMP_MARKER);

        let recorded: Option<usize> = std::fs::read_to_string(&marker)
            .ok()
            .and_then(|s| s.trim().parse().ok());
        let recorded_id: Option<String> = std::fs::read_to_string(&id_marker)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // DIMENSION IS NOT IDENTITY. Two different models at the same dimension —
        // the hash-bigram fallback and MiniLM are both 384 — leave `recorded ==
        // Some(target)` true, so a width check alone returns early over a corpus embedded by
        // something else entirely: every vector the right SHAPE and the wrong CONTENT.
        let action = corpus_action(recorded, recorded_id.as_deref(), target, &target_id);
        let stamped = stamp_marker.exists();
        if action == CorpusAction::UpToDate && stamped {
            return Ok(());
        }

        let migrations = SpaceMigrationStore::with_kv(self.storage.key_value_store());
        let pending = migrations.load()?.filter(|m| !m.is_finished());
        let ids = self.storage.get_all_ids()?;
        if ids.is_empty() {
            // Fresh / empty store — record everything; nothing to stamp or migrate.
            let _ = std::fs::write(&marker, target.to_string());
            let _ = std::fs::write(&id_marker, &target_id);
            let _ = std::fs::write(&stamp_marker, target_space.as_str());
            return Ok(());
        }

        // A corpus written before the embedder marker existed. We CANNOT tell what embedded
        // it, and neither answer is safe to assume: re-embedding thousands of vectors unasked
        // is expensive, and writing the current id claims a provenance nobody established.
        // So the unknown gets its own value — `unverified:<id>` — which is not equal to
        // `<id>`, so it never satisfies the early return above, and says in the file itself
        // that this was believed rather than checked.
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
        }

        // Read after an unverified marker is written: a corpus whose provenance was never
        // recorded is stamped with what it is believed to be, which is what it was indexed as.
        let corpus = space::corpus_space(&self.storage_path);
        let migrate = matches!(
            action,
            CorpusAction::MigrateChangedLength | CorpusAction::MigrateAll
        );

        if migrate {
            info!(
                "Embedding space change detected (corpus {:?} at {:?}d, now {}) — {} memories \
                 will be re-embedded in the background",
                recorded_id,
                recorded,
                target_space,
                ids.len()
            );
            if action == CorpusAction::MigrateAll && !stamped {
                self.stamp_now(corpus.as_ref(), &ids);
            }
            // A stale primary index must not be loaded against the new space. Other spaces'
            // indexes are bound to their own space and stay valid.
            let _ = std::fs::remove_file(
                self.storage_path
                    .join("vector_index")
                    .join(VAMANA_INDEX_FILE),
            );
            let resumes = pending
                .as_ref()
                .is_some_and(|m| m.kind == MigrationKind::Reembed && m.to == target_space);
            if !resumes {
                migrations.save(&SpaceMigration::new(
                    MigrationKind::Reembed,
                    corpus,
                    target_space.clone(),
                    ids.len(),
                ))?;
            }
            // Both markers advance together. The migration record carries what is pending.
            let _ = std::fs::write(&marker, target.to_string());
            let _ = std::fs::write(&id_marker, &target_id);
            return Ok(());
        }

        if pending.is_none() {
            info!(
                memories = ids.len(),
                space = %target_space,
                "Embedding spaces not yet recorded on this corpus — stamping in the background"
            );
            migrations.save(&SpaceMigration::new(
                MigrationKind::Stamp,
                corpus.or_else(|| Some(target_space.clone())),
                target_space,
                ids.len(),
            ))?;
        }
        Ok(())
    }

    /// Stamp every vector that has no space, now: the one case a stamp cannot wait for the
    /// drain. No embedding, a rewrite of each record that changes.
    fn stamp_now(&self, corpus: Option<&SpaceId>, ids: &[MemoryId]) {
        let mut stamped = 0usize;
        for id in ids {
            let Ok(mut memory) = self.storage.get(id) else {
                continue;
            };
            let primary = memory
                .experience
                .embeddings
                .as_ref()
                .and_then(|v| corpus.filter(|c| c.dimension() == v.len()).cloned());
            let secondary = memory
                .experience
                .embeddings_secondary
                .as_ref()
                .and_then(|v| {
                    self.secondary_index_space
                        .clone()
                        .filter(|s| s.dimension() == v.len())
                });
            if memory
                .experience
                .stamp_vector_spaces(primary.as_ref(), secondary.as_ref())
            {
                match self.storage.update(&memory) {
                    Ok(()) => stamped += 1,
                    Err(e) => warn!("Failed to stamp embedding spaces for {}: {}", id.0, e),
                }
            }
        }
        info!(stamped, "Stamped embedding spaces before the markers moved");
    }

    /// The corpus's space migration, finished or not, when one was ever recorded.
    pub fn space_migration(&self) -> Option<SpaceMigration> {
        SpaceMigrationStore::with_kv(self.storage.key_value_store())
            .load()
            .ok()
            .flatten()
    }

    /// Whether a re-embed migration is under way: its memories are the migration's to index,
    /// not the start-up repair's or the backlog's.
    pub fn is_migrating(&self) -> bool {
        self.space_migration()
            .is_some_and(|m| m.kind == MigrationKind::Reembed && !m.is_finished())
    }

    /// One batch of the corpus's space migration: at most `limit` memories after the cursor,
    /// embedded in one call. `None` when no migration is pending. The cursor is saved after the
    /// batch, so a kill anywhere costs at most this batch, which is idempotent to redo; an
    /// embedder that does not answer stops the batch without moving the cursor.
    pub fn run_space_migration(&self, limit: usize) -> Result<Option<MigrationStep>> {
        let migrations = SpaceMigrationStore::with_kv(self.storage.key_value_store());
        let Some(mut m) = migrations.load()? else {
            return Ok(None);
        };
        if m.is_finished() {
            return Ok(None);
        }
        if m.kind == MigrationKind::Reembed && m.to != self.index_space {
            info!(
                "The embedder changed again ({} → {}) before the migration finished; re-walking",
                m.to, self.index_space
            );
            m.to = self.index_space.clone();
            m.cursor = None;
        }
        let mut ids = self.storage.get_all_ids()?;
        ids.sort();
        let start = m
            .cursor
            .as_ref()
            .map_or(0, |cursor| ids.partition_point(|id| id <= cursor));
        let batch: Vec<MemoryId> = ids[start..].iter().take(limit.max(1)).cloned().collect();
        let mut step = MigrationStep::default();
        if batch.is_empty() {
            m.finished_at = Some(chrono::Utc::now());
            migrations.save(&m)?;
            let _ = std::fs::write(self.storage_path.join(space::STAMP_MARKER), m.to.as_str());
            info!(
                kind = ?m.kind,
                to = %m.to,
                walked = m.walked,
                changed = m.changed,
                failed = m.failed,
                "Embedding space migration finished"
            );
            step.finished = true;
            return Ok(Some(step));
        }

        let to = m.to.clone();
        let from = m.from.clone();
        let reembed = m.kind == MigrationKind::Reembed;
        let mut loaded: Vec<(Memory, bool)> = Vec::with_capacity(batch.len());
        for id in &batch {
            let mut memory = match self.storage.get(id) {
                Ok(memory) if memory.is_forgotten() => continue,
                Ok(memory) => memory,
                Err(e) => {
                    warn!("Space migration cannot load {}: {}", id.0, e);
                    step.failed += 1;
                    continue;
                }
            };
            let primary = memory
                .experience
                .embeddings
                .as_ref()
                .and_then(|v| from.clone().filter(|f| f.dimension() == v.len()));
            let secondary = memory
                .experience
                .embeddings_secondary
                .as_ref()
                .and_then(|v| {
                    self.secondary_index_space
                        .clone()
                        .filter(|s| s.dimension() == v.len())
                        .or_else(|| {
                            Some(to.clone()).filter(|t| reembed && t.dimension() == v.len())
                        })
                });
            let stamped = memory
                .experience
                .stamp_vector_spaces(primary.as_ref(), secondary.as_ref());
            loaded.push((memory, stamped));
        }

        // Which memories need a vector in `to`: a secondary already in it is promoted, the rest
        // are embedded in one call.
        let mut to_embed: Vec<usize> = Vec::new();
        if reembed {
            for (i, (memory, dirty)) in loaded.iter_mut().enumerate() {
                if memory.experience.primary_space(None).as_ref() == Some(&to) {
                    continue;
                }
                let promoted = memory
                    .experience
                    .embeddings_secondary
                    .clone()
                    .filter(|_| memory.experience.secondary_space(None).as_ref() == Some(&to));
                match promoted {
                    Some(vector) => {
                        replace_primary(memory, vector, to.clone());
                        *dirty = true;
                        step.changed += 1;
                    }
                    None => to_embed.push(i),
                }
            }
        }
        let mut reindex: Vec<usize> = Vec::new();
        if !to_embed.is_empty() {
            let texts: Vec<&str> = to_embed
                .iter()
                .map(|&i| loaded[i].0.experience.content.as_str())
                .collect();
            match self.embedder.encode_batch(&texts) {
                Ok(vectors) => {
                    for (&i, vector) in to_embed.iter().zip(vectors) {
                        if vector.len() != to.dimension() {
                            step.failed += 1;
                            continue;
                        }
                        let space = self.embedder.primary_space_for(&vector);
                        replace_primary(&mut loaded[i].0, vector, space);
                        loaded[i].1 = true;
                        step.changed += 1;
                        reindex.push(i);
                    }
                }
                Err(e) => {
                    m.last_error = Some(format!("{e:#}"));
                    migrations.save(&m)?;
                    step.stopped = Some(format!("{e:#}"));
                    return Ok(Some(step));
                }
            }
        }
        if !reembed {
            step.changed = loaded.iter().filter(|(_, dirty)| *dirty).count();
        }

        for (i, (memory, dirty)) in loaded.iter().enumerate() {
            if !*dirty {
                continue;
            }
            if let Err(e) = self.storage.update(memory) {
                warn!("Space migration cannot persist {}: {}", memory.id.0, e);
                step.failed += 1;
                continue;
            }
            step.rewritten.push(memory.id.clone());
            if reindex.contains(&i) || (reembed && !self.is_indexed(&memory.id)) {
                if let Err(e) = self.reindex_memory(memory) {
                    if !is_space_mismatch(&e) {
                        warn!("Space migration cannot index {}: {e}", memory.id.0);
                    }
                }
            }
        }

        step.walked = batch.len();
        m.cursor = batch.last().cloned();
        m.walked += step.walked;
        m.changed += step.changed;
        m.failed += step.failed;
        m.last_error = None;
        migrations.save(&m)?;

        if reembed {
            self.bridge_back(&mut m, &migrations);
        }
        Ok(Some(step))
    }

    /// Once enough memories hold vectors in both spaces, fit the map from the new space back
    /// into the old one, so a query reaches the memories the migration has not yet re-embedded
    /// through the old space's index. Tried once per migration.
    fn bridge_back(&self, m: &mut SpaceMigration, migrations: &SpaceMigrationStore) {
        const PAIRS_BEFORE_BRIDGE: usize = 256;
        let Some(from) = m.from.clone() else {
            return;
        };
        if m.bridge_tried || from == m.to || m.changed < PAIRS_BEFORE_BRIDGE {
            return;
        }
        m.bridge_tried = true;
        if self.embedder.alignment_between(&m.to, &from).is_none() {
            match self.fit_bridge(&m.to, &from) {
                Ok(report) => info!(
                    held_out_cosine = report.held_out_cosine,
                    "Mapped {} back into {} for the memories not yet migrated", m.to, from
                ),
                Err(e) => warn!("Could not map {} back into {}: {e:#}", m.to, from),
            }
        }
        if let Err(e) = migrations.save(m) {
            warn!("Could not record the migration's bridge attempt: {e}");
        }
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

        // Stamp unnamed vectors with the space the corpus markers name, and re-embed into the
        // active space if the embedder changed, before any index load or rebuild, so every
        // vector an index admits is in the index's space.
        if let Err(e) = self.reconcile_corpus_spaces() {
            warn!("Embedding space pass failed (rebuilding best-effort): {e}");
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

        let (legacy_dir, legacy_file) = LEGACY_SECONDARY_INDEX;
        let _ = std::fs::remove_file(self.storage_path.join(legacy_dir).join(legacy_file));
        self.load_or_rebuild_space_indexes()?;

        Ok(())
    }

    /// Every non-primary space's index, cloned out of the map so no caller holds its lock.
    fn space_index_list(&self) -> Vec<Arc<SpaceIndex>> {
        self.space_indexes.read().values().cloned().collect()
    }

    fn space_index(&self, space: &SpaceId) -> Option<Arc<SpaceIndex>> {
        self.space_indexes.read().get(space).cloned()
    }

    /// Load every space index saved cleanly with its mapping; rebuild the rest from the memories
    /// holding a vector in their space, in one pass over the store. A secondary slot holding a
    /// vector in another space than the secondary index's is refused and queued.
    fn load_or_rebuild_space_indexes(&self) -> Result<()> {
        let mut to_rebuild: Vec<Arc<SpaceIndex>> = Vec::new();
        for index in self.space_index_list() {
            match index.load() {
                Ok(true) => {}
                Ok(false) => to_rebuild.push(index),
                Err(e) => {
                    warn!("{}: saved index unreadable, rebuilding: {e}", index.space());
                    to_rebuild.push(index);
                }
            }
        }
        self.rebuild_space_indexes(to_rebuild)
    }

    /// Rebuild `indexes` from every memory holding a vector in their spaces, in one pass.
    fn rebuild_space_indexes(&self, to_rebuild: Vec<Arc<SpaceIndex>>) -> Result<()> {
        if to_rebuild.is_empty() {
            return Ok(());
        }
        let mut members: Vec<Vec<(MemoryId, Vec<f32>)>> = vec![Vec::new(); to_rebuild.len()];
        for memory_id in self.storage.get_all_ids()? {
            let Ok(memory) = self.storage.get(&memory_id) else {
                continue;
            };
            if memory.is_forgotten() {
                continue;
            }
            for (slot, index) in to_rebuild.iter().enumerate() {
                match self.stored_vector(&memory, index.space()) {
                    Some(vector) => members[slot].push((memory_id.clone(), vector.clone())),
                    None if self.secondary_index_space.as_ref() == Some(index.space())
                        && memory.experience.embeddings_secondary.is_some() =>
                    {
                        self.refuse_secondary(&memory)
                    }
                    None => {}
                }
            }
        }
        for (index, members) in to_rebuild.into_iter().zip(members) {
            index.rebuild(members)?;
        }
        Ok(())
    }

    /// The vector `memory` holds in `space`, reading absent slot spaces as the engine always
    /// has: the primary slot in the primary space, the secondary slot in the secondary's. Any
    /// cosine against a query in `space` must compare this vector and no other.
    pub fn stored_vector<'m>(&self, memory: &'m Memory, space: &SpaceId) -> Option<&'m Vec<f32>> {
        memory.experience.vector_in(
            space,
            Some(&self.index_space),
            self.secondary_index_space.as_ref(),
        )
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
        let live = |vid: u32| (vid as usize) < loaded_count && !deleted.contains(&vid);

        // A mapping whose ids are missing or deleted in this file was written after the last save;
        // its memory goes back to the startup repair, which re-indexes and re-records it.
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
            .filter(|vid| !referenced.contains(vid) && !deleted.contains(vid))
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
            let Some(own) =
                memory.experience.embeddings.as_ref().filter(|e| {
                    e.len() == index.config.dimension && self.holds_own_vector(&memory)
                })
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
    /// no own vector, loses its mapping and is left to the startup repair, which re-chunks it.
    fn rebuild_primary_index(&self) -> Result<()> {
        use crate::embeddings::chunking::{chunk_text, ChunkConfig};

        let start = std::time::Instant::now();
        self.invalidate_saved_index();
        let dimension = self.embedder.primary_dimension();
        let chunk_config = ChunkConfig::for_space(&self.index_space);
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
                .filter(|e| e.len() == dimension && self.holds_own_vector(&memory));
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
        let mut id_mapping = IdMapping::new();
        for (vid, memory_id) in members.into_iter().enumerate() {
            self.storage
                .update_vector_mapping(&memory_id, vec![vid as u32])?;
            id_mapping.insert(memory_id, vid as u32);
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
        for (memory_id, ids) in &remapped.memory_to_vectors {
            self.storage.update_vector_mapping(memory_id, ids.clone())?;
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
        let index_path = self.storage_path.join("vector_index");
        fs::create_dir_all(&index_path)?;

        let vamana_path = index_path.join(VAMANA_INDEX_FILE);

        // LOCK ORDERING: Always acquire vector_index (1) before id_mapping (2). The mapping
        // is only counted: holding it through the save would stall every insert.
        let vector_index = self.vector_index.read();
        let vector_count = self.id_mapping.read().len();
        if vector_count > 0 {
            // save_to_file replaces the file atomically: on a crash the old one survives.
            match vector_index.save_to_file(&vamana_path) {
                Ok(()) => {
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

        // LOCK ORDERING: each space index after the primary's, one at a time.
        for index in self.space_index_list() {
            if let Err(e) = index.save() {
                warn!(
                    "Failed to save the {} index (will rebuild on restart): {e}",
                    index.space()
                );
            }
        }

        Ok(())
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
    /// startup recovery repairs, counted while it is accumulating rather than
    /// only after a restart notices.
    pub fn vector_index_failure_count(&self) -> u64 {
        self.vector_index_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn index_memory(&self, memory: &Memory) -> Result<()> {
        self.recording_index_failures(memory, || {
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
        self.recording_index_failures(memory, || self.insert_vectors(memory))
    }

    /// Counted and recorded at the boundary, not at the seven call sites. A space refusal is
    /// not a loss: it becomes a row the drain fills in the index's space.
    fn recording_index_failures<T>(
        &self,
        memory: &Memory,
        index: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let outcome = index();
        if let Err(e) = &outcome {
            match e.downcast_ref::<SpaceMismatch>() {
                Some(mismatch) => self.record_refusal(&BacklogKey::primary(&memory.id), mismatch),
                None => {
                    self.vector_index_failures
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        outcome
    }

    /// The space of the primary index.
    pub fn index_space(&self) -> &SpaceId {
        &self.index_space
    }

    /// The space of the secondary index, when there is one.
    pub fn secondary_index_space(&self) -> Option<&SpaceId> {
        self.secondary_index_space.as_ref()
    }

    /// Whether the memory's primary vector is in the primary index's space. A vector stored
    /// without a space is taken to be in it when the widths agree, as before spaces existed.
    fn holds_own_vector(&self, memory: &Memory) -> bool {
        memory
            .experience
            .primary_space(Some(&self.index_space))
            .as_ref()
            == Some(&self.index_space)
    }

    fn record_refusal(&self, key: &BacklogKey, mismatch: &SpaceMismatch) {
        tracing::debug!("{mismatch}; queued for a vector in {}", mismatch.index);
        if let Err(e) = self
            .backlog
            .record_mismatch(key, &mismatch.stored, &mismatch.index)
        {
            tracing::error!(
                "embedding backlog: could not record {}: {e}",
                mismatch.memory.0
            );
        }
    }

    /// Record that the secondary index cannot hold the memory's secondary vector.
    fn refuse_secondary(&self, memory: &Memory) {
        let (Some(index), Some(vector)) = (
            &self.secondary_index_space,
            &memory.experience.embeddings_secondary,
        ) else {
            return;
        };
        let stored = space::resolve(
            memory.experience.embedding_space_secondary.as_ref(),
            vector.len(),
            Some(index),
        );
        self.record_refusal(
            &BacklogKey::in_space(&memory.id, index),
            &SpaceMismatch {
                memory: memory.id.clone(),
                stored,
                index: index.clone(),
            },
        );
    }

    /// Embed `memory` into the in-memory primary index and every other space's index and
    /// return its primary vector ids; nothing is persisted.
    fn insert_vectors(&self, memory: &Memory) -> Result<Vec<u32>> {
        use crate::embeddings::chunking::{chunk_text, ChunkConfig};

        let text = Self::extract_searchable_text(memory);
        let chunk_config = ChunkConfig::for_space(&self.index_space);
        let chunk_result = chunk_text(&text, &chunk_config);

        let vector_ids = if chunk_result.was_chunked {
            // Long content: embed each chunk separately
            // Pre-compute all embeddings OUTSIDE the write lock to avoid blocking searches
            let chunks: Vec<&str> = chunk_result.chunks.iter().map(String::as_str).collect();
            let embeddings: Vec<Vec<f32>> = self
                .embedder
                .encode_batch(&chunks)
                .context("Failed to generate chunk embedding")?;

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
            let embedding = match &memory.experience.embeddings {
                Some(emb) if self.holds_own_vector(memory) => emb.clone(),
                Some(emb) => {
                    return Err(SpaceMismatch {
                        memory: memory.id.clone(),
                        stored: space::resolve(
                            memory.experience.embedding_space.as_ref(),
                            emb.len(),
                            Some(&self.index_space),
                        ),
                        index: self.index_space.clone(),
                    }
                    .into())
                }
                None => self
                    .embedder
                    .encode(&text)
                    .context("Failed to generate embedding")?,
            };

            let mut index = self.vector_index.write();
            let vector_id = index
                .add_vector(embedding)
                .context("Failed to add vector to index")?;

            // Update in-memory mapping
            self.id_mapping.write().insert(memory.id.clone(), vector_id);

            vec![vector_id]
        };

        // Other spaces' failures are non-fatal — the primary is authoritative. Each is recorded,
        // so the drain gives that index its vector.
        self.index_other_spaces(memory);

        Ok(vector_ids)
    }

    /// Drop the memory from every index and add back only its vectors in spaces other than the
    /// primary's: for a memory whose primary vector is still to come.
    pub fn reindex_other_spaces(&self, memory: &Memory) {
        self.remove_memory(&memory.id);
        self.index_other_spaces(memory);
    }

    /// Add the memory's vector in every other space that has an index, recording any space
    /// whose index refused it.
    fn index_other_spaces(&self, memory: &Memory) {
        for index in self.space_index_list() {
            if let Err(e) = self.add_in_space(memory, &index) {
                if is_space_mismatch(&e) {
                    continue;
                }
                if let Err(record_err) = self.backlog.record_key(
                    &BacklogKey::in_space(&memory.id, index.space()),
                    super::embedding_backlog::EnqueueReason::IndexFailed,
                    Some(index.space()),
                    None,
                    Some(&e.to_string()),
                ) {
                    tracing::error!(
                        "embedding backlog: could not record {}: {record_err}",
                        memory.id.0
                    );
                }
            }
        }
    }

    /// Add the memory's vector in `index`'s space. A secondary slot holding a vector in
    /// another space than the secondary index's is refused and recorded, never added.
    fn add_in_space(&self, memory: &Memory, index: &SpaceIndex) -> Result<()> {
        match self.stored_vector(memory, index.space()) {
            Some(vector) => index.add(&memory.id, vector.clone()),
            None if self.secondary_index_space.as_ref() == Some(index.space()) => {
                let Some(sec_emb) = &memory.experience.embeddings_secondary else {
                    return Ok(());
                };
                self.refuse_secondary(memory);
                Err(SpaceMismatch {
                    memory: memory.id.clone(),
                    stored: space::resolve(
                        memory.experience.embedding_space_secondary.as_ref(),
                        sec_emb.len(),
                        Some(index.space()),
                    ),
                    index: index.space().clone(),
                }
                .into())
            }
            None => Ok(()),
        }
    }

    /// Replace the memory's vector in `space`'s index with the one it stores now. For the
    /// drain of a row targeting that space.
    pub fn reindex_space(&self, memory: &Memory, space: &SpaceId) -> Result<()> {
        let index = self
            .space_index(space)
            .with_context(|| format!("no index here holds {space}"))?;
        index.remove(&memory.id);
        self.add_in_space(memory, &index)
    }

    /// Whether `space`'s index holds the memory. The primary space reads the primary index.
    pub fn is_indexed_in(&self, space: &SpaceId, memory_id: &MemoryId) -> bool {
        if *space == self.index_space {
            return self.is_indexed(memory_id);
        }
        self.space_index(space)
            .is_some_and(|index| index.contains(memory_id))
    }

    /// The spaces this engine can search: the primary index's, then every other index's.
    pub fn spaces(&self) -> Vec<SpaceId> {
        std::iter::once(self.index_space.clone())
            .chain(self.space_indexes.read().keys().cloned())
            .collect()
    }

    /// Memories each index holds, by the index's space.
    pub fn indexed_by_space(&self) -> Vec<(SpaceId, usize)> {
        std::iter::once((self.index_space.clone(), self.len()))
            .chain(
                self.space_index_list()
                    .iter()
                    .map(|index| (index.space().clone(), index.len())),
            )
            .collect()
    }

    /// Memories keeping a vector in a space no index here holds, by that space. One pass over
    /// the store.
    pub fn census_kept(&self) -> Result<std::collections::BTreeMap<SpaceId, usize>> {
        let indexed = self.spaces();
        let mut kept = std::collections::BTreeMap::new();
        for memory_id in self.storage.get_all_ids()? {
            let Ok(memory) = self.storage.get(&memory_id) else {
                continue;
            };
            if memory.is_forgotten() {
                continue;
            }
            let mut spaces: Vec<SpaceId> = memory
                .experience
                .other_vectors
                .iter()
                .map(|v| v.space.clone())
                .collect();
            spaces.extend(memory.experience.primary_space(Some(&self.index_space)));
            spaces.extend(
                memory
                    .experience
                    .secondary_space(self.secondary_index_space.as_ref()),
            );
            spaces.sort();
            spaces.dedup();
            for space in spaces.into_iter().filter(|s| !indexed.contains(s)) {
                *kept.entry(space).or_insert(0) += 1;
            }
        }
        Ok(kept)
    }

    /// Every installed map between spaces, with where the corpus keeps it.
    pub fn bridges(&self) -> Vec<crate::embeddings::space::RegisteredAlignment> {
        self.embedder
            .bridges()
            .into_iter()
            .map(|(from, to, bridge)| {
                let path = crate::embeddings::alignment_spaces::corpus_alignment_path(
                    &self.storage_path,
                    &from,
                    &to,
                );
                crate::embeddings::space::RegisteredAlignment {
                    method: bridge.alignment.method().to_string(),
                    residual: Some(1.0 - bridge.confidence),
                    path: path.display().to_string(),
                    from,
                    to,
                }
            })
            .collect()
    }

    /// Fit the map from `from` into `to` on every memory holding a vector in both, save it
    /// beside the corpus and install it. A target space this engine does not index yet gets an
    /// index, built from the memories' stored vectors in it, when the map starts from a space
    /// this engine writes: that is the space queries arrive in.
    pub fn fit_bridge(
        &self,
        from: &SpaceId,
        to: &SpaceId,
    ) -> Result<crate::embeddings::alignment_spaces::BridgeReport> {
        use crate::embeddings::alignment_spaces as bridging;
        if from == to {
            anyhow::bail!("a map from {from} to itself bridges nothing");
        }
        let mut pairs = Vec::new();
        for memory_id in self.storage.get_all_ids()? {
            let Ok(memory) = self.storage.get(&memory_id) else {
                continue;
            };
            if memory.is_forgotten() {
                continue;
            }
            if let (Some(f), Some(t)) = (
                self.stored_vector(&memory, from),
                self.stored_vector(&memory, to),
            ) {
                pairs.push((f.clone(), t.clone()));
            }
        }
        let fit = bridging::fit_bridge(
            from,
            to,
            &pairs,
            &crate::embeddings::AlignmentConfig::from_env(),
        )?;
        let path = bridging::corpus_alignment_path(&self.storage_path, from, to);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        crate::embeddings::save_alignment(&path, &fit.alignment)?;
        self.embedder.install_bridge(
            from.clone(),
            to.clone(),
            Arc::new(fit.alignment),
            fit.report.held_out_cosine,
        )?;
        let writes_from =
            *from == self.index_space || self.secondary_index_space.as_ref() == Some(from);
        let reaches_new_space = writes_from && !self.spaces().contains(to);
        if reaches_new_space {
            let index = Arc::new(SpaceIndex::open(
                &self.storage_path,
                to.clone(),
                self.space_codec,
            )?);
            self.space_indexes.write().insert(to.clone(), index.clone());
            self.rebuild_space_indexes(vec![index])?;
        }
        info!(
            from = %from,
            to = %to,
            pairs = fit.report.pairs,
            held_out_cosine = fit.report.held_out_cosine,
            recall_at_1 = fit.report.held_out_recall_at_1,
            "Fitted a map between spaces"
        );
        Ok(fit.report)
    }

    /// Retire `space`: drop every memory's kept vector in it, its index and the maps into and
    /// out of it. Refused for a space this engine writes, for the space a running migration
    /// is leaving, and while any memory's only vector is in it: retiring would leave that
    /// memory with no vector at all.
    pub fn retire_space(&self, space: &SpaceId) -> Result<RetireReport> {
        if *space == self.index_space || self.secondary_index_space.as_ref() == Some(space) {
            anyhow::bail!("{space} is a space this server writes; change the embedder first");
        }
        if let Some(m) = self.space_migration().filter(|m| !m.is_finished()) {
            if m.from.as_ref() == Some(space) {
                anyhow::bail!(
                    "the migration out of {space} has re-embedded {} of {} memories; retire it once it finishes",
                    m.walked, m.total
                );
            }
        }
        let mut holders = Vec::new();
        let mut sole = 0usize;
        for memory_id in self.storage.get_all_ids()? {
            let Ok(memory) = self.storage.get(&memory_id) else {
                continue;
            };
            if memory.is_forgotten() || self.stored_vector(&memory, space).is_none() {
                continue;
            }
            let elsewhere = memory
                .experience
                .primary_space(Some(&self.index_space))
                .into_iter()
                .chain(
                    memory
                        .experience
                        .secondary_space(self.secondary_index_space.as_ref()),
                )
                .chain(
                    memory
                        .experience
                        .other_vectors
                        .iter()
                        .map(|v| v.space.clone()),
                )
                .any(|s| s != *space);
            if elsewhere {
                holders.push(memory);
            } else {
                sole += 1;
            }
        }
        if sole > 0 {
            anyhow::bail!(
                "{sole} memories hold a vector only in {space}; they need a vector in another space before it can be retired"
            );
        }
        let mut dropped = 0usize;
        for mut memory in holders {
            let experience = &mut memory.experience;
            if experience.primary_space(Some(&self.index_space)).as_ref() == Some(space) {
                experience.clear_primary_vector();
            }
            if experience
                .secondary_space(self.secondary_index_space.as_ref())
                .as_ref()
                == Some(space)
            {
                experience.clear_secondary_vector();
            }
            experience.forget_kept_vector(space);
            match self.storage.update(&memory) {
                Ok(()) => dropped += 1,
                Err(e) => warn!("Retiring {space}: cannot persist {}: {e}", memory.id.0),
            }
        }
        let mut maps_removed = 0usize;
        for (from, to, _) in self.embedder.bridges() {
            if from == *space || to == *space {
                self.embedder.remove_bridge(&from, &to);
                let _ =
                    fs::remove_file(crate::embeddings::alignment_spaces::corpus_alignment_path(
                        &self.storage_path,
                        &from,
                        &to,
                    ));
                maps_removed += 1;
            }
        }
        if self.space_indexes.write().remove(space).is_some() {
            let _ = fs::remove_dir_all(super::space_index::dir_for(&self.storage_path, space));
        }
        info!(space = %space, dropped, maps_removed, "Retired an embedding space");
        Ok(RetireReport {
            space: space.clone(),
            vectors_dropped: dropped,
            maps_removed,
        })
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

        // Every other space's index drops the memory too; index_memory adds back what it holds.
        for index in self.space_index_list() {
            index.remove(&memory.id);
        }

        // Add with new embedding (may create multiple chunks)
        // index_memory() also indexes the memory's other spaces
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
        // Every other space first, whether or not the primary index held the memory: a memory
        // the primary refused can still be in another space's index.
        let mut removed_elsewhere = false;
        for index in self.space_index_list() {
            removed_elsewhere |= index.remove(memory_id);
        }

        // Remove from in-memory ID mapping and get the vector IDs
        let vector_ids = self.id_mapping.write().remove_all(memory_id);

        if !vector_ids.is_empty() {
            // Mark vectors as deleted in Vamana (soft delete)
            let index = self.vector_index.read();
            for vid in &vector_ids {
                index.mark_deleted(*vid);
            }
            drop(index);

            // NOTE: Memory graph edges are managed in GraphMemory at the API layer
            // GraphMemory handles cleanup via its own mechanisms

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
            removed_elsewhere
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
        let index = self.vector_index.read();
        let results = index
            .search(
                &query_embedding,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
            )
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
        self.observe_scores(&self.index_space, &memory_ids);

        Ok(memory_ids)
    }

    /// Search the secondary embedder's space. Empty when no secondary is configured.
    /// Used as a 4th RRF signal in Layer 4 fusion.
    pub fn search_ids_secondary(
        &self,
        query_embedding_secondary: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        match &self.secondary_index_space {
            Some(space) => self.search_space(space, query_embedding_secondary, limit),
            None => Ok(Vec::new()),
        }
    }

    /// Fold a search's best similarity into its space's statistics.
    pub fn observe_scores(&self, space: &SpaceId, results: &[(MemoryId, f32)]) {
        if let Some((_, best)) = results.first() {
            let mut stats = self.score_stats.write();
            let current = stats.get(space).copied();
            stats.insert(space.clone(), space::ScoreStats::observe(current, *best));
        }
    }

    /// Each space's learned score statistics.
    pub fn score_stats(&self) -> std::collections::HashMap<SpaceId, space::ScoreStats> {
        self.score_stats.read().clone()
    }

    /// `similarity`, from `space`, on the primary space's scale, so a raw cosine from another
    /// space can be compared with the primary's. Unchanged until both spaces have seen enough
    /// searches to say what their scales are.
    pub fn calibrated_similarity(&self, space: &SpaceId, similarity: f32) -> f32 {
        if *space == self.index_space {
            return similarity;
        }
        let stats = self.score_stats.read();
        match (stats.get(space), stats.get(&self.index_space)) {
            (Some(from), Some(to)) if from.ready() && to.ready() => from.onto(to, similarity),
            _ => similarity,
        }
    }

    /// Search `space`'s index with a query vector in that space: one score per memory (its
    /// best vector), best first, ties broken by id. Empty when no index here holds the space.
    /// The primary space searches the primary index, chunk aggregation included.
    pub fn search_space(
        &self,
        space: &SpaceId,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let results = if *space == self.index_space {
            self.search_by_embedding(query, limit, None)?
        } else {
            match self.space_index(space) {
                Some(index) => index.search(
                    query,
                    limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
                    limit,
                    &*self.exact_vectors(space.clone()),
                )?,
                None => Vec::new(),
            }
        };
        self.observe_scores(space, &results);
        Ok(results)
    }

    /// The legs a query's vectors reach beyond the primary index, in the order recall fuses
    /// them: the primary through the installed secondary-to-primary map, in place of the direct
    /// secondary search, which runs only if the mapped one fails; each other space's index a
    /// query vector in `space_queries` is in; every indexed space a map reaches from a space
    /// with a query vector, unless a query vector is in it already. `dial` overrides a space's
    /// candidate factor; a space without an entry asks for what its record supports.
    pub fn space_legs(
        &self,
        primary_query: &[f32],
        space_queries: &[(SpaceId, Vec<f32>)],
        results: usize,
        dial: &std::collections::BTreeMap<String, f32>,
    ) -> Vec<Leg> {
        let limit_for = |space: &SpaceId| -> usize {
            let factor = dial
                .get(space.as_str())
                .copied()
                .unwrap_or_else(|| self.candidate_factor(space));
            space_legs::budget(factor, results)
        };
        let mut legs = Vec::new();

        let aligned = self.embedder.alignment_snapshot().and_then(|map| {
            let from = self.embedder.secondary_space()?;
            let (_, query) = space_queries.iter().find(|(s, _)| *s == from)?;
            Some(Leg {
                space: self.index_space.clone(),
                kind: LegKind::Aligned,
                index: LegIndex::Primary {
                    index: Arc::clone(&self.vector_index),
                    mapping: Arc::clone(&self.id_mapping),
                },
                map: Some(map),
                query: query.clone(),
                limit: limit_for(&self.index_space),
                fallback: false,
            })
        });
        let fallback = aligned.is_some();
        legs.extend(aligned);

        for (space, query) in space_queries {
            if *space == self.index_space {
                continue;
            }
            if let Some(index) = self.space_index(space) {
                legs.push(Leg {
                    space: space.clone(),
                    kind: LegKind::Direct,
                    index: LegIndex::Space {
                        index,
                        exact: self.exact_vectors(space.clone()),
                    },
                    map: None,
                    query: query.clone(),
                    limit: limit_for(space),
                    fallback,
                });
            }
        }

        for (from, to, bridge) in self.embedder.bridges() {
            if to == self.index_space || space_queries.iter().any(|(s, _)| *s == to) {
                continue;
            }
            let query = if from == self.index_space {
                Some(primary_query)
            } else {
                space_queries
                    .iter()
                    .find(|(s, _)| *s == from)
                    .map(|(_, q)| q.as_slice())
            };
            let (Some(query), Some(index)) = (query, self.space_index(&to)) else {
                continue;
            };
            legs.push(Leg {
                limit: limit_for(&to),
                index: LegIndex::Space {
                    index,
                    exact: self.exact_vectors(to.clone()),
                },
                space: to,
                kind: LegKind::Bridged {
                    from,
                    confidence: bridge.confidence,
                },
                map: Some(bridge.alignment),
                query: query.to_vec(),
                fallback: false,
            });
        }
        legs
    }

    /// Reads a memory's exact vector in `space` from its record, as the space's index was built
    /// from: what a quantized index reranks its candidates against.
    fn exact_vectors(&self, space: SpaceId) -> ExactVectors {
        let storage = Arc::clone(&self.storage);
        let primary = self.index_space.clone();
        let secondary = self.secondary_index_space.clone();
        Arc::new(move |memory_id: &MemoryId| {
            let memory = storage.get(memory_id).ok()?;
            if memory.is_forgotten() {
                return None;
            }
            memory
                .experience
                .vector_in(&space, Some(&primary), secondary.as_ref())
                .cloned()
        })
    }

    /// How each indexed space holds its vectors: the primary's in its graph, every other by the
    /// configured codec.
    pub fn index_codecs(&self) -> Vec<(SpaceId, &'static str)> {
        std::iter::once((self.index_space.clone(), space::CODEC_F32))
            .chain(
                self.space_index_list()
                    .iter()
                    .map(|index| (index.space().clone(), index.codec().as_str())),
            )
            .collect()
    }

    /// Compact each space index whose deletions call for it, and rebuild from the records any
    /// whose quantizer has fallen behind what it holds, or whose graph has taken too many
    /// incremental inserts. For the heavy maintenance pass.
    pub fn maintain_space_indexes(&self) -> Result<()> {
        let mut to_rebuild = Vec::new();
        for index in self.space_index_list() {
            let health = index.health();
            if health.needs_rebuild && index.codec() == SpaceCodec::Pq {
                info!(
                    space = %health.space,
                    vectors = health.total_vectors,
                    "Space index retrains its quantizer from the records"
                );
                to_rebuild.push(index);
            } else if health.needs_compaction || health.needs_rebuild {
                index.compact()?;
            }
        }
        self.rebuild_space_indexes(to_rebuild)
    }

    /// The candidate factor `space`'s record supports.
    pub fn candidate_factor(&self, space: &SpaceId) -> f32 {
        self.leg_depths
            .read()
            .get(space)
            .map_or(space_legs::DEFAULT_CANDIDATE_FACTOR, LegDepth::factor)
    }

    /// Each space's record of how deep into its legs the returned results lay.
    pub fn leg_depths(&self) -> std::collections::HashMap<SpaceId, LegDepth> {
        self.leg_depths.read().clone()
    }

    /// Record, for each leg, how deep into its list the `returned` results lay, out of the
    /// `results` the recall asked for.
    pub fn observe_leg_depths(
        &self,
        legs: &[(SpaceId, Vec<MemoryId>)],
        returned: &[MemoryId],
        results: usize,
    ) {
        if legs.is_empty() {
            return;
        }
        let mut depths = self.leg_depths.write();
        for (space, ranked) in legs {
            depths
                .entry(space.clone())
                .or_default()
                .observe(space_legs::returned_depth(ranked, returned, results));
        }
    }

    /// Fold finished legs' best similarities into their spaces' score statistics. An aligned
    /// leg's query is a mapped one, so its scores say nothing of the primary's own scale.
    pub fn observe_leg_scores(&self, legs: &[LegHits]) {
        for leg in legs {
            if leg.kind != LegKind::Aligned {
                self.observe_scores(&leg.space, &leg.hits);
            }
        }
    }

    /// Get memory from storage by ID
    pub fn get_from_storage(&self, id: &MemoryId) -> Result<Memory> {
        self.storage.get(id)
    }

    /// Read `ids` from storage, results in the same order. The one seam recall's candidate
    /// projection reads through. `PrimaryMemoryStore` has no batch read, so today this is
    /// one point read per id.
    pub fn get_many_from_storage(&self, ids: &[MemoryId]) -> Vec<Result<Memory>> {
        ids.iter().map(|id| self.storage.get(id)).collect()
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
        let index = self.vector_index.read();
        let results = index
            .search(embedding, limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2)
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
        let index = self.vector_index.read();
        let results = index
            .search(&query_embedding, limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER)
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
    pub fn reembed_all(&self) -> Result<ReembedReport> {
        let all_ids = self.storage.get_all_ids()?;
        let total = all_ids.len();

        if total == 0 {
            tracing::info!("No memories to re-embed");
            return Ok(ReembedReport::default());
        }

        tracing::info!(
            "Starting context-prefix re-embedding migration: {} memories",
            total
        );
        let start_time = std::time::Instant::now();
        let mut reembedded = 0;
        let mut failed = 0;
        let mut rewritten = Vec::new();

        for (i, memory_id) in all_ids.iter().enumerate() {
            match self.storage.get(memory_id) {
                Ok(mut memory) => {
                    if memory.is_forgotten() {
                        continue;
                    }

                    // Clear pre-computed embedding to force re-generation
                    // through extract_searchable_text() → encode()
                    memory.experience.clear_primary_vector();

                    // Persist the cleared embedding so deferred path also re-generates
                    if let Err(e) = self.storage.update(&memory) {
                        tracing::warn!("Failed to clear embedding for {}: {e}", memory_id.0);
                        failed += 1;
                        continue;
                    }
                    rewritten.push(memory_id.clone());

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

        Ok(ReembedReport {
            reembedded,
            failed,
            rewritten,
        })
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
        let secondary = self
            .secondary_index_space
            .as_ref()
            .and_then(|space| self.space_index(space))
            .map(|index| {
                let h = index.health();
                SecondaryIndexHealth {
                    total_vectors: h.total_vectors,
                    incremental_inserts: h.incremental_inserts,
                    deleted_count: h.deleted_count,
                    deletion_ratio: h.deletion_ratio,
                    needs_rebuild: h.needs_rebuild,
                    needs_compaction: h.needs_compaction,
                }
            });
        let spaces = self
            .space_index_list()
            .iter()
            .map(|index| index.health())
            .collect();
        IndexHealth {
            spaces,
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
    /// Every non-primary space's index, the secondary's included.
    pub spaces: Vec<SpaceIndexHealth>,
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
    /// The instant `hour_of_day` and `day_of_week` were read from. The temporal query takes
    /// its date from the same instant; without it the date was the wall clock's while the hour
    /// was the caller's, so a tick on a pinned scoring clock searched a day it never stored.
    /// `None` (contexts built from a request's fields): the wall clock.
    #[serde(default)]
    pub at: Option<chrono::DateTime<chrono::Utc>>,
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
            at: None,
        }
    }

    /// Create context from current system state
    pub fn from_current_time() -> Self {
        Self::at(chrono::Utc::now())
    }

    /// Context for the hour and weekday of `now`.
    pub fn at(now: chrono::DateTime<chrono::Utc>) -> Self {
        Self {
            hour_of_day: Some(now.hour()),
            day_of_week: Some(now.weekday().num_days_from_sunday()),
            at: Some(now),
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
            return Some(self.temporal_query(hour, day, context.at));
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
    fn temporal_query(
        &self,
        hour: u32,
        _day: u32,
        at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Query {
        let now = at.unwrap_or_else(chrono::Utc::now);

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
    fn a_temporal_query_searches_the_day_its_context_was_taken_at() {
        let at = chrono::DateTime::parse_from_rfc3339("2025-03-04T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let query = AnticipatoryPrefetch::new()
            .generate_prefetch_query(&PrefetchContext::at(at))
            .expect("an hour and a weekday give a temporal query");
        let (start, end) = query.time_range.expect("a temporal query has a range");
        assert!(start <= at && at <= end, "{start}..{end} must contain {at}");
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
            spaces: Vec::new(),
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

#[cfg(all(test, feature = "storage-redb"))]
mod space_tests {
    use super::*;
    use crate::memory::embedding_backlog::{BacklogKey, EmbeddingBacklogStore};
    use crate::storage::PrimaryMemoryStore;

    const DIM: usize = 8;

    struct Stub {
        id: String,
        model: &'static str,
        dim: usize,
        fail: bool,
    }

    impl Stub {
        fn new(model: &'static str, dim: usize) -> Arc<Self> {
            Arc::new(Self {
                id: format!("http://stub#{model}"),
                model,
                dim,
                fail: false,
            })
        }

        fn failing(model: &'static str, dim: usize) -> Arc<Self> {
            Arc::new(Self {
                id: format!("http://stub#{model}"),
                model,
                dim,
                fail: true,
            })
        }
    }

    impl Embedder for Stub {
        fn id(&self) -> &str {
            &self.id
        }
        fn model_id(&self) -> &str {
            self.model
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn encode(&self, text: &str) -> Result<Vec<f32>> {
            if self.fail {
                anyhow::bail!("stub embedder is down");
            }
            let seed = text
                .bytes()
                .fold(17u32, |h, b| h.wrapping_mul(31) ^ u32::from(b));
            let mut v: Vec<f32> = (0..self.dim)
                .map(|i| ((seed.wrapping_add(i as u32 * 7919) % 1000) as f32 / 1000.0) + 0.01)
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            Ok(v)
        }
    }

    fn store(dir: &Path) -> Arc<dyn PrimaryMemoryStore> {
        Arc::new(crate::storage::redb::RedbPrimaryMemoryStore::open(dir).expect("redb store"))
    }

    fn engine(
        store: &Arc<dyn PrimaryMemoryStore>,
        dir: &Path,
        primary: Arc<Stub>,
        secondary: Option<Arc<Stub>>,
    ) -> RetrievalEngine {
        let embedder = Arc::new(CompetitiveEmbedder::new(
            primary,
            secondary.map(|s| s as Arc<dyn Embedder>),
        ));
        RetrievalEngine::new(store.clone(), dir, embedder).expect("engine")
    }

    fn memory(content: &str, vector: Option<Vec<f32>>) -> Memory {
        Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                content: content.to_string(),
                embeddings: vector,
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            None,
        )
    }

    /// Run the space migration to its end, one batch of `batch` at a time.
    fn migrate(engine: &RetrievalEngine, batch: usize) -> usize {
        let mut ticks = 0;
        while let Some(step) = engine.run_space_migration(batch).unwrap() {
            ticks += 1;
            if step.finished || step.stopped.is_some() {
                break;
            }
        }
        ticks
    }

    fn markers(dir: &Path, embedder_id: &str, dim: usize) {
        std::fs::write(dir.join(space::EMBEDDER_MARKER), embedder_id).unwrap();
        std::fs::write(dir.join(space::DIMENSION_MARKER), dim.to_string()).unwrap();
    }

    #[test]
    fn the_stamp_pass_names_old_vectors_by_the_corpus_markers() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let legacy = memory("written before spaces", Some(a.encode("x").unwrap()));
        let narrow = memory("written by a narrower model", Some(vec![0.5; 4]));
        store.store(&legacy).unwrap();
        store.store(&narrow).unwrap();
        markers(tmp.path(), &a.id, DIM);

        let engine = engine(&store, tmp.path(), a, None);
        assert!(
            !tmp.path().join(space::STAMP_MARKER).exists(),
            "an ordinary start rewrites nothing; the stamp runs in the drain"
        );
        assert!(
            engine.is_indexed(&legacy.id),
            "searchable before it is stamped"
        );
        migrate(&engine, 1);

        let model_a = SpaceId::new("model-a", DIM);
        assert_eq!(engine.index_space(), &model_a);
        assert_eq!(
            store.get(&legacy.id).unwrap().experience.embedding_space,
            Some(model_a.clone())
        );
        assert_eq!(
            store.get(&narrow.id).unwrap().experience.embedding_space,
            Some(SpaceId::unrecorded(4)),
            "a width the markers do not explain names no model"
        );
        assert!(tmp.path().join(space::STAMP_MARKER).exists());
        assert!(engine.is_indexed(&legacy.id));
        assert!(!engine.is_indexed(&narrow.id));
    }

    #[test]
    fn an_embedder_change_keeps_the_old_space_on_what_it_could_not_re_embed() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let mut written = memory("written under model a", None);
        let v = a.encode(&written.experience.content).unwrap();
        written
            .experience
            .set_primary_vector(v, SpaceId::new("model-a", DIM));
        store.store(&written).unwrap();
        markers(tmp.path(), &a.id, DIM);
        drop(engine(&store, tmp.path(), a, None));

        // Same width, another model, and down: the migration cannot re-embed.
        let b = engine(&store, tmp.path(), Stub::failing("model-b", DIM), None);
        let step = b
            .run_space_migration(10)
            .unwrap()
            .expect("a migration is pending");
        assert!(step.stopped.is_some(), "a down embedder stops the batch");
        assert_eq!(
            b.space_migration().unwrap().cursor,
            None,
            "and does not move the cursor"
        );
        let read = store.get(&written.id).unwrap();
        assert_eq!(
            read.experience.embedding_space,
            Some(SpaceId::new("model-a", DIM)),
            "the vector still says where it came from"
        );
        assert_eq!(b.index_space(), &SpaceId::new("model-b", DIM));

        let err = b
            .index_memory(&read)
            .expect_err("a same-width foreign vector is refused");
        assert!(is_space_mismatch(&err));
        assert_eq!(b.vector_index_failure_count(), 0, "a refusal is not a loss");
        let row = EmbeddingBacklogStore::with_kv(store.key_value_store())
            .get(&written.id)
            .unwrap()
            .expect("the refusal is a backlog row");
        assert_eq!(row.stored_space, Some(SpaceId::new("model-a", DIM)));
        assert_eq!(row.target_space, Some(SpaceId::new("model-b", DIM)));
        assert_eq!(
            b.indexed_by_space(),
            vec![(SpaceId::new("model-b", DIM), 0)]
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join(space::EMBEDDER_MARKER)).unwrap(),
            "http://stub#model-b",
            "the markers advance at once; the migration record says what is pending"
        );
    }

    #[test]
    fn a_vector_with_no_space_at_the_index_width_is_indexed_as_before() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let engine = engine(&store, tmp.path(), a.clone(), None);
        let supplied = memory(
            "a caller supplied this vector",
            Some(a.encode("y").unwrap()),
        );
        store.store(&supplied).unwrap();
        engine.index_memory(&supplied).unwrap();
        assert!(engine.is_indexed(&supplied.id));
    }

    #[test]
    fn a_secondary_space_index_reloads_with_its_mapping_and_is_searched_by_space() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let s = Stub::new("model-s", 6);
        let model_s = SpaceId::new("model-s", 6);
        let mut ids = Vec::new();
        {
            let engine = engine(&store, tmp.path(), a.clone(), Some(s.clone()));
            for n in 0..12 {
                let content = format!("memory number {n}");
                let mut m = memory(&content, None);
                m.experience
                    .set_primary_vector(a.encode(&content).unwrap(), SpaceId::new("model-a", DIM));
                m.experience
                    .set_secondary_vector(s.encode(&content).unwrap(), model_s.clone());
                store.store(&m).unwrap();
                engine.index_memory(&m).unwrap();
                ids.push((m.id.clone(), content));
            }
            assert!(engine.remove_memory(&ids[3].0));
            engine.save().unwrap();
        }
        let engine = engine(&store, tmp.path(), a, Some(s.clone()));
        assert_eq!(
            engine.spaces(),
            vec![SpaceId::new("model-a", DIM), model_s.clone()]
        );
        assert!(engine.is_indexed_in(&model_s, &ids[5].0));
        assert!(
            !engine.is_indexed_in(&model_s, &ids[3].0),
            "a removal survives the reload"
        );
        let query = s.encode(&ids[7].1).unwrap();
        let hits = engine.search_space(&model_s, &query, 3).unwrap();
        assert_eq!(hits[0].0, ids[7].0);
        assert!(
            engine
                .search_space(&SpaceId::new("model-z", 6), &query, 3)
                .unwrap()
                .is_empty(),
            "a space no index holds has nothing to search"
        );
    }

    #[test]
    fn a_kept_vector_in_an_indexed_space_is_indexed_there() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let s = Stub::new("model-s", 6);
        let model_s = SpaceId::new("model-s", 6);
        let engine = engine(&store, tmp.path(), a.clone(), Some(s.clone()));
        let mut m = memory("kept from another instance", None);
        m.experience
            .set_primary_vector(a.encode("k").unwrap(), SpaceId::new("model-a", DIM));
        m.experience.keep_vector(
            model_s.clone(),
            s.encode("kept from another instance").unwrap(),
        );
        store.store(&m).unwrap();
        engine.index_memory(&m).unwrap();
        assert!(engine.is_indexed_in(&model_s, &m.id));
    }

    #[test]
    fn a_re_embed_keeps_the_vector_it_replaced() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let mut written = memory("written under model a", None);
        let old = a.encode(&written.experience.content).unwrap();
        written
            .experience
            .set_primary_vector(old.clone(), SpaceId::new("model-a", DIM));
        store.store(&written).unwrap();
        markers(tmp.path(), &a.id, DIM);
        drop(engine(&store, tmp.path(), a, None));

        let b = engine(&store, tmp.path(), Stub::new("model-b", DIM), None);
        assert!(
            !b.is_indexed(&written.id),
            "the start-up pass no longer re-embeds"
        );
        migrate(&b, 10);
        assert!(b.space_migration().unwrap().is_finished());
        let read = store.get(&written.id).unwrap();
        assert_eq!(
            read.experience.embedding_space,
            Some(SpaceId::new("model-b", DIM))
        );
        assert_eq!(
            read.experience
                .vector_in(&SpaceId::new("model-a", DIM), None, None),
            Some(&old),
            "the stored vector in the old space stays until that space is retired"
        );
        assert!(b.is_indexed(&written.id));
    }

    /// The space a migration left behind, or another instance wrote: no embedder here writes it,
    /// and queries reach it only through a fitted map.
    #[test]
    fn a_fitted_map_reaches_a_space_no_embedder_here_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let model_a = SpaceId::new("model-a", DIM);
        let model_b = SpaceId::new("model-b", 6);
        let truncated = |v: &[f32]| {
            let head: Vec<f32> = v[..6].to_vec();
            let n = head.iter().map(|x| x * x).sum::<f32>().sqrt();
            head.into_iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let mut ids = Vec::new();
        {
            let engine = engine(&store, tmp.path(), a.clone(), None);
            for n in 0..150 {
                let content = format!("bridged memory {n} about topic {}", n % 11);
                let va = a.encode(&content).unwrap();
                let vb = truncated(&va);
                let mut m = memory(&content, None);
                m.experience.set_primary_vector(va, model_a.clone());
                m.experience.keep_vector(model_b.clone(), vb);
                store.store(&m).unwrap();
                engine.index_memory(&m).unwrap();
                ids.push((m.id.clone(), content));
            }
            assert!(!engine.spaces().contains(&model_b));
            let report = engine.fit_bridge(&model_a, &model_b).unwrap();
            assert!(
                report.held_out_cosine > 0.9,
                "held-out cosine {}",
                report.held_out_cosine
            );
            assert!(
                engine.spaces().contains(&model_b),
                "the map's target got an index"
            );
            let (id, content) = &ids[42];
            let legs = engine.space_legs(
                &a.encode(content).unwrap(),
                &[],
                5,
                &std::collections::BTreeMap::new(),
            );
            assert_eq!(legs.len(), 1, "one leg, through the map");
            assert_eq!(legs[0].space, model_b);
            assert_eq!(legs[0].limit, 40, "the default budget, 8 per result");
            let finished = space_legs::PendingLegs::start(legs).finish();
            let leg = &finished[0];
            let LegKind::Bridged { from, confidence } = &leg.kind else {
                panic!("{:?}", leg.kind);
            };
            assert_eq!(*from, model_a);
            assert!((confidence - report.held_out_cosine).abs() < 1e-5);
            assert!(leg.hits.iter().any(|(h, _)| h == id), "{:?}", leg.hits);

            let ranked: Vec<MemoryId> = leg.hits.iter().map(|(h, _)| h.clone()).collect();
            for _ in 0..space_legs::LegDepth::MIN_SAMPLES {
                engine.observe_leg_depths(&[(model_b.clone(), ranked.clone())], &ranked[..1], 5);
            }
            assert_eq!(
                engine.candidate_factor(&model_b),
                space_legs::MIN_CANDIDATE_FACTOR,
                "a leg whose returned results lie at its top asks for less"
            );
            let dial = std::collections::BTreeMap::from([(model_b.to_string(), 3.0)]);
            let legs = engine.space_legs(&a.encode(content).unwrap(), &[], 5, &dial);
            assert_eq!(legs[0].limit, 15, "the dial overrides the record");
            assert!(crate::embeddings::alignment_spaces::corpus_alignment_path(
                tmp.path(),
                &model_a,
                &model_b
            )
            .exists());
            engine.save().unwrap();
        }
        let reopened = engine(&store, tmp.path(), a, None);
        assert!(
            reopened.spaces().contains(&model_b),
            "the map survives a restart"
        );
        assert_eq!(reopened.bridges().len(), 1);
        assert!(reopened.is_indexed_in(&model_b, &ids[7].0));
    }

    /// write-latch task 4: a kill mid-migration resumes at the cursor, not at the start.
    #[test]
    fn a_migration_killed_mid_way_resumes_at_its_cursor() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let mut ids = Vec::new();
        for n in 0..9 {
            let content = format!("memory {n} written under model a");
            let mut m = memory(&content, None);
            m.experience
                .set_primary_vector(a.encode(&content).unwrap(), SpaceId::new("model-a", DIM));
            store.store(&m).unwrap();
            ids.push(m.id.clone());
        }
        markers(tmp.path(), &a.id, DIM);
        drop(engine(&store, tmp.path(), a, None));

        let b = engine(&store, tmp.path(), Stub::new("model-b", DIM), None);
        let first = b.run_space_migration(4).unwrap().unwrap();
        assert_eq!((first.walked, first.changed), (4, 4));
        let cursor = b.space_migration().unwrap().cursor.expect("cursor saved");
        drop(b);

        let resumed = engine(&store, tmp.path(), Stub::new("model-b", DIM), None);
        let migration = resumed.space_migration().unwrap();
        assert_eq!(migration.cursor, Some(cursor), "a restart keeps the cursor");
        assert_eq!(migration.walked, 4);
        migrate(&resumed, 4);
        let done = resumed.space_migration().unwrap();
        assert!(done.is_finished());
        assert_eq!(done.changed, 9, "each memory re-embedded once");
        for id in &ids {
            assert!(resumed.is_indexed(id));
        }
    }

    #[test]
    fn a_space_is_retired_only_when_no_memory_depends_on_it() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let engine = engine(&store, tmp.path(), a.clone(), None);
        let old = SpaceId::new("model-old", 4);

        let mut only_old = memory("held only in the old space", None);
        only_old.experience.keep_vector(old.clone(), vec![0.5; 4]);
        store.store(&only_old).unwrap();
        let err = engine.retire_space(&old).unwrap_err();
        assert!(format!("{err:#}").contains("only in"), "{err:#}");

        let mut both = memory("held in both", None);
        both.experience
            .set_primary_vector(a.encode("both").unwrap(), SpaceId::new("model-a", DIM));
        both.experience.keep_vector(old.clone(), vec![0.5; 4]);
        store.store(&both).unwrap();
        store.delete(&only_old.id).unwrap();

        let report = engine.retire_space(&old).unwrap();
        assert_eq!(report.vectors_dropped, 1);
        assert!(store
            .get(&both.id)
            .unwrap()
            .experience
            .vector_in(&old, None, None)
            .is_none());
        assert!(
            engine.retire_space(&SpaceId::new("model-a", DIM)).is_err(),
            "the space this engine writes cannot be retired"
        );
    }

    /// Mid-migration, the memories not yet re-embedded stay searchable: once enough hold both
    /// vectors, the new space is mapped back into the old one.
    #[test]
    fn a_long_migration_maps_the_new_space_back_into_the_old() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        for n in 0..300 {
            let content = format!("long migration memory {n} about topic {}", n % 13);
            let mut m = memory(&content, None);
            m.experience
                .set_primary_vector(a.encode(&content).unwrap(), SpaceId::new("model-a", DIM));
            store.store(&m).unwrap();
        }
        markers(tmp.path(), &a.id, DIM);
        drop(engine(&store, tmp.path(), a, None));

        let b = engine(&store, tmp.path(), Stub::new("model-b", DIM), None);
        migrate(&b, 64);
        let back = b
            .embedder
            .alignment_between(&SpaceId::new("model-b", DIM), &SpaceId::new("model-a", DIM));
        assert!(back.is_some(), "the map back was fitted");
        assert!(b.space_migration().unwrap().bridge_tried);
        assert!(b.spaces().contains(&SpaceId::new("model-a", DIM)));
    }

    #[test]
    fn a_secondary_in_another_space_is_queued_and_the_primary_still_indexes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path());
        let a = Stub::new("model-a", DIM);
        let s = Stub::new("model-s", 6);
        let engine = engine(&store, tmp.path(), a.clone(), Some(s));
        let model_s = SpaceId::new("model-s", 6);
        assert_eq!(engine.secondary_index_space(), Some(&model_s));

        let mut m = memory("two vectors, one foreign", None);
        m.experience
            .set_primary_vector(a.encode("z").unwrap(), SpaceId::new("model-a", DIM));
        m.experience
            .set_secondary_vector(vec![0.4; 6], SpaceId::new("model-t", 6));
        store.store(&m).unwrap();
        engine.index_memory(&m).unwrap();

        assert!(engine.is_indexed(&m.id));
        assert!(!engine.is_indexed_in(&model_s, &m.id));
        let row = EmbeddingBacklogStore::with_kv(store.key_value_store())
            .get_key(&BacklogKey::in_space(&m.id, &model_s))
            .unwrap()
            .expect("a space row for the secondary");
        assert_eq!(row.stored_space, Some(SpaceId::new("model-t", 6)));
    }
}
