//! One vector index per embedding space, for every space but the primary index's.
//!
//! The primary index keeps its own layout (`vector_index/`, mappings in the store) because its
//! mappings are authoritative and its chunk vectors are not stored on the record. Every other
//! space gets a `SpaceIndex` under `space_indexes/<slug>/`: the index file, the mapping that
//! names which memory each vector belongs to, the space id it was built for, and a marker that
//! says the three were saved together. The mapping is saved with the index rather than
//! re-derived from insertion order, which was the secondary index's drift: a vector added after
//! a save, or soft-deleted, shifted every ordinal behind it.
//!
//! A space's index holds its vectors by codec. `pq`, the default, keeps product-quantized codes,
//! a 32nd of the vectors' size, and reranks a search's best candidates against the exact
//! vectors the memory records keep, so a vector is stored in full once, in its record. `f32`
//! keeps full vectors in a Vamana graph. `VELD_SPACE_CODEC` chooses; an index saved under the
//! other codec is rebuilt from the records at the next start.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::types::MemoryId;
use crate::embeddings::space::{SpaceId, CODEC_F32, CODEC_PQ};
use crate::vector_db::distance_inline::cosine_similarity_inline;
use crate::vector_db::quantized::QuantizedIndex;
use crate::vector_db::vamana::{VamanaConfig, VamanaIndex};

/// Directory under a tenant's storage holding one subdirectory per space.
pub const SPACE_INDEX_ROOT: &str = "space_indexes";
const INDEX_FILE: &str = "vamana.idx";
const CODES_FILE: &str = "pq.idx";
const MAPPING_FILE: &str = "mapping.bin";
const SPACE_FILE: &str = "space";
/// Present only while the index file and the mapping file describe the same state. Removed at
/// the first change after a save, written by the next save.
const CLEAN_MARKER: &str = "saved.clean";

/// Bidirectional mapping between memory IDs and vector IDs
///
/// Supports multiple vectors per memory for chunked embeddings.
/// When long content is split into chunks, each chunk gets its own vector ID,
/// but all map back to the same MemoryId.
#[derive(Serialize, Deserialize, Default)]
pub(crate) struct IdMapping {
    /// Maps each memory to ALL its vector IDs (supports chunked embeddings)
    pub(crate) memory_to_vectors: HashMap<MemoryId, Vec<u32>>,
    /// Maps each vector ID back to its parent memory
    pub(crate) vector_to_memory: HashMap<u32, MemoryId>,
}

impl IdMapping {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Insert a single vector for a memory. Idempotent: removes any existing mappings for this
    /// memory first so orphaned vector ids do not accumulate.
    pub(crate) fn insert(&mut self, memory_id: MemoryId, vector_id: u32) {
        self.insert_chunks(memory_id, vec![vector_id]);
    }

    /// Insert every vector of a chunked memory. Idempotent like `insert`.
    pub(crate) fn insert_chunks(&mut self, memory_id: MemoryId, vector_ids: Vec<u32>) {
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

    pub(crate) fn get_memory_id(&self, vector_id: u32) -> Option<&MemoryId> {
        self.vector_to_memory.get(&vector_id)
    }

    /// Remove a memory and return ALL its vector IDs
    pub(crate) fn remove_all(&mut self, memory_id: &MemoryId) -> Vec<u32> {
        match self.memory_to_vectors.remove(memory_id) {
            Some(vector_ids) => {
                for vid in &vector_ids {
                    self.vector_to_memory.remove(vid);
                }
                vector_ids
            }
            None => Vec::new(),
        }
    }

    /// Number of unique memories in the mapping
    pub(crate) fn len(&self) -> usize {
        self.memory_to_vectors.len()
    }

    pub(crate) fn clear(&mut self) {
        self.memory_to_vectors.clear();
        self.vector_to_memory.clear();
    }

    /// Check if a memory ID has any vector mappings (for deletion verification)
    pub(crate) fn contains(&self, memory_id: &MemoryId) -> bool {
        self.memory_to_vectors.contains_key(memory_id)
    }
}

/// What the mapping file holds: the mapping and what it was saved against.
#[derive(Serialize, Deserialize)]
struct SavedMapping {
    space: SpaceId,
    index_len: usize,
    memory_to_vectors: HashMap<MemoryId, Vec<u32>>,
}

/// Index health for one space.
#[derive(Debug, Clone)]
pub struct SpaceIndexHealth {
    pub space: SpaceId,
    pub codec: &'static str,
    pub memories: usize,
    pub total_vectors: usize,
    /// Bytes the index holds its vectors in: full vectors for `f32`, codes and codebooks (or
    /// the exact vectors held until training) for `pq`. A graph's edges are not counted.
    pub vector_bytes: usize,
    pub incremental_inserts: usize,
    pub deleted_count: usize,
    pub deletion_ratio: f32,
    pub needs_rebuild: bool,
    pub needs_compaction: bool,
}

/// How a space's index holds its vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceCodec {
    /// Full vectors in a Vamana graph.
    F32,
    /// Product-quantized codes searched flat; the best candidates reranked exactly.
    Pq,
}

impl SpaceCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            SpaceCodec::F32 => CODEC_F32,
            SpaceCodec::Pq => CODEC_PQ,
        }
    }

    /// `VELD_SPACE_CODEC`: `pq`, the default, or `f32`.
    pub fn from_env() -> Self {
        let Ok(value) = std::env::var("VELD_SPACE_CODEC") else {
            return SpaceCodec::Pq;
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => SpaceCodec::F32,
            "pq" | "" => SpaceCodec::Pq,
            other => {
                warn!("VELD_SPACE_CODEC={other:?} is neither pq nor f32; using pq");
                SpaceCodec::Pq
            }
        }
    }

    fn file(self) -> &'static str {
        match self {
            SpaceCodec::F32 => INDEX_FILE,
            SpaceCodec::Pq => CODES_FILE,
        }
    }

    fn other(self) -> Self {
        match self {
            SpaceCodec::F32 => SpaceCodec::Pq,
            SpaceCodec::Pq => SpaceCodec::F32,
        }
    }
}

/// A memory's exact vector in a space, read from where the records keep it; `None` when the
/// memory no longer holds one there.
pub(crate) type ExactVectors = std::sync::Arc<dyn Fn(&MemoryId) -> Option<Vec<f32>> + Send + Sync>;

/// Candidates a quantized search reranks against exact vectors, per result asked for.
const RERANK_FACTOR: usize = 2;

enum Store {
    Graph(VamanaIndex),
    Codes(QuantizedIndex),
}

impl Store {
    fn empty(codec: SpaceCodec, dir: &Path, dimension: usize) -> Result<Self> {
        Ok(match codec {
            SpaceCodec::F32 => Store::Graph(VamanaIndex::with_storage_path(
                config_for(dimension),
                Some(dir.to_path_buf()),
            )?),
            SpaceCodec::Pq => Store::Codes(QuantizedIndex::new(dimension)),
        })
    }

    fn len(&self) -> usize {
        match self {
            Store::Graph(g) => g.len(),
            Store::Codes(c) => c.len(),
        }
    }

    fn add(&mut self, vector: Vec<f32>) -> Result<u32> {
        match self {
            Store::Graph(g) => g.add_vector(vector),
            Store::Codes(c) => c.add(vector),
        }
    }

    fn mark_deleted(&mut self, id: u32) {
        match self {
            Store::Graph(g) => {
                g.mark_deleted(id);
            }
            Store::Codes(c) => c.mark_deleted(id),
        }
    }

    fn deleted(&self) -> HashSet<u32> {
        match self {
            Store::Graph(g) => g.deleted_ids.read().clone(),
            Store::Codes(c) => (0..c.len() as u32).filter(|&i| c.is_deleted(i)).collect(),
        }
    }

    fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        match self {
            Store::Graph(g) => g.build(vectors),
            Store::Codes(c) => c.build(vectors),
        }
    }

    /// Nearest slots with their cosine, best first; a quantized store's are estimates.
    fn search(&self, query: &[f32], candidates: usize, limit: usize) -> Result<Vec<(u32, f32)>> {
        match self {
            // NormalizedDotProduct: the index returns -dot, so similarity is -distance.
            Store::Graph(g) => Ok(g
                .search(query, candidates)?
                .into_iter()
                .map(|(id, distance)| (id, -distance))
                .collect()),
            Store::Codes(c) if c.is_quantized() => c.search(query, limit * RERANK_FACTOR),
            Store::Codes(c) => c.search(query, limit),
        }
    }

    fn estimates(&self) -> bool {
        matches!(self, Store::Codes(c) if c.is_quantized())
    }
}

/// One space's index and mapping.
///
/// Lock order inside: `index` before `mapping`. The retrieval engine never holds two
/// spaces' locks at once.
pub(crate) struct SpaceIndex {
    space: SpaceId,
    codec: SpaceCodec,
    dir: PathBuf,
    index: RwLock<Store>,
    mapping: RwLock<IdMapping>,
    dirty: AtomicBool,
    /// Set while a quantizer is trained outside the locks, so only one add trains it.
    training: AtomicBool,
}

/// `<root>/space_indexes/<readable slug>-<hash>`: the slug keeps the tree greppable, the hash
/// keeps two ids that slug alike apart.
pub(crate) fn dir_for(storage_path: &Path, space: &SpaceId) -> PathBuf {
    use sha2::{Digest, Sha256};
    let slug: String = space
        .as_str()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let digest = Sha256::digest(space.as_str().as_bytes());
    let hash: String = digest[..4].iter().map(|b| format!("{b:02x}")).collect();
    storage_path
        .join(SPACE_INDEX_ROOT)
        .join(format!("{slug}-{hash}"))
}

fn config_for(dimension: usize) -> VamanaConfig {
    VamanaConfig {
        dimension,
        max_degree: 32,
        search_list_size: 100,
        alpha: 1.2,
        use_mmap: false,
        ..Default::default()
    }
}

impl SpaceIndex {
    /// An empty index for `space` under `codec`, with its directory created. Nothing is read
    /// from disk: `load` does that.
    pub(crate) fn open(storage_path: &Path, space: SpaceId, codec: SpaceCodec) -> Result<Self> {
        let dir = dir_for(storage_path, &space);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating the {space} index directory"))?;
        let index = Store::empty(codec, &dir, space.dimension())
            .with_context(|| format!("initialising the {space} index"))?;
        Ok(Self {
            space,
            codec,
            dir,
            index: RwLock::new(index),
            mapping: RwLock::new(IdMapping::new()),
            // Clean until the first change, which clears whatever marker a past save left.
            dirty: AtomicBool::new(false),
            training: AtomicBool::new(false),
        })
    }

    pub(crate) fn space(&self) -> &SpaceId {
        &self.space
    }

    pub(crate) fn codec(&self) -> SpaceCodec {
        self.codec
    }

    pub(crate) fn len(&self) -> usize {
        self.mapping.read().len()
    }

    pub(crate) fn contains(&self, memory_id: &MemoryId) -> bool {
        self.mapping.read().contains(memory_id)
    }

    /// The first change after a save makes the saved files stale. Called while holding the
    /// exclusive lock the change needs, so it cannot interleave with a save (which holds the
    /// shared locks to the end), and before the change, so a crash before the next save
    /// rebuilds instead of loading a mapping that no longer holds.
    fn touch(&self) {
        if !self.dirty.swap(true, Ordering::AcqRel) {
            if let Err(e) = std::fs::remove_file(self.dir.join(CLEAN_MARKER)) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    warn!("{}: could not clear the saved marker: {e}", self.space);
                }
            }
        }
    }

    /// The saved index file under this index's codec, checked for this space.
    fn load_store(&self, path: &Path) -> Result<Option<Store>> {
        match self.codec {
            SpaceCodec::F32 => {
                if !VamanaIndex::verify_index_file(path)? {
                    warn!("{}: saved index fails its checksum; rebuilding", self.space);
                    return Ok(None);
                }
                let mut loaded = VamanaIndex::load_from_file(path)?;
                if loaded.config.dimension != self.space.dimension() {
                    warn!(
                        "{}: saved index is {}d; rebuilding",
                        self.space, loaded.config.dimension
                    );
                    return Ok(None);
                }
                // Persistence stores a construction default; search uses the runtime list size.
                loaded.config.search_list_size = 100;
                Ok(Some(Store::Graph(loaded)))
            }
            SpaceCodec::Pq => match QuantizedIndex::load_from_file(path) {
                Ok(loaded) if loaded.dimension() == self.space.dimension() => {
                    Ok(Some(Store::Codes(loaded)))
                }
                Ok(loaded) => {
                    warn!(
                        "{}: saved index is {}d; rebuilding",
                        self.space,
                        loaded.dimension()
                    );
                    Ok(None)
                }
                Err(e) => {
                    warn!("{}: saved index unreadable ({e:#}); rebuilding", self.space);
                    Ok(None)
                }
            },
        }
    }

    /// Load the saved index and mapping when they were saved together for this space under
    /// this codec. Returns whether the index is now the saved one; `false` leaves the index
    /// empty for a rebuild.
    pub(crate) fn load(&self) -> Result<bool> {
        let index_path = self.dir.join(self.codec.file());
        let mapping_path = self.dir.join(MAPPING_FILE);
        if !self.dir.join(CLEAN_MARKER).exists() || !index_path.exists() || !mapping_path.exists() {
            if self.dir.join(self.codec.other().file()).exists() {
                info!(
                    "{}: saved under the {} codec, now {}; rebuilding",
                    self.space,
                    self.codec.other().as_str(),
                    self.codec.as_str()
                );
            }
            return Ok(false);
        }
        let recorded = std::fs::read_to_string(self.dir.join(SPACE_FILE)).unwrap_or_default();
        if recorded.trim() != self.space.as_str() {
            warn!(
                "{}: index directory was built for {:?}; rebuilding",
                self.space,
                recorded.trim()
            );
            return Ok(false);
        }
        let Some(mut loaded) = self.load_store(&index_path)? else {
            return Ok(false);
        };
        let saved: SavedMapping = bincode::serde::decode_from_slice(
            &std::fs::read(&mapping_path)?,
            bincode::config::standard(),
        )
        .map(|(m, _)| m)
        .context("decoding the saved mapping")?;
        let count = loaded.len();
        if saved.space != self.space || saved.index_len != count {
            warn!(
                "{}: saved mapping describes another index ({} vectors, file has {count}); rebuilding",
                self.space, saved.index_len
            );
            return Ok(false);
        }
        let deleted = loaded.deleted();
        let live = |vid: u32| (vid as usize) < count && !deleted.contains(&vid);
        if !saved
            .memory_to_vectors
            .values()
            .all(|ids| !ids.is_empty() && ids.iter().all(|&vid| live(vid)))
        {
            warn!(
                "{}: saved mapping names vectors the index does not hold; rebuilding",
                self.space
            );
            return Ok(false);
        }
        let referenced: HashSet<u32> = saved
            .memory_to_vectors
            .values()
            .flatten()
            .copied()
            .collect();
        for vid in (0..count as u32).filter(|v| !referenced.contains(v) && !deleted.contains(v)) {
            loaded.mark_deleted(vid);
        }
        let mut mapping = IdMapping::new();
        for (memory_id, ids) in saved.memory_to_vectors {
            mapping.insert_chunks(memory_id, ids);
        }
        let memories = mapping.len();
        *self.index.write() = loaded;
        *self.mapping.write() = mapping;
        self.dirty.store(false, Ordering::Release);
        info!(
            space = %self.space,
            codec = self.codec.as_str(),
            vectors = count,
            memories,
            "Loaded saved space index"
        );
        if self.health().needs_compaction {
            self.compact()?;
        }
        Ok(true)
    }

    /// Replace the index with one built from `members`, one vector per memory. A `pq` index
    /// trains its quantizer on them when there are enough.
    pub(crate) fn rebuild(&self, members: Vec<(MemoryId, Vec<f32>)>) -> Result<()> {
        let start = std::time::Instant::now();
        let mut index = Store::empty(self.codec, &self.dir, self.space.dimension())?;
        let (ids, vectors): (Vec<MemoryId>, Vec<Vec<f32>>) = members
            .into_iter()
            .filter(|(_, v)| v.len() == self.space.dimension())
            .unzip();
        index.build(vectors)?;
        let mut mapping = IdMapping::new();
        for (vid, memory_id) in ids.into_iter().enumerate() {
            mapping.insert(memory_id, vid as u32);
        }
        let memories = mapping.len();
        let mut index_slot = self.index.write();
        let mut mapping_slot = self.mapping.write();
        self.touch();
        *index_slot = index;
        *mapping_slot = mapping;
        drop(mapping_slot);
        drop(index_slot);
        info!(
            space = %self.space,
            codec = self.codec.as_str(),
            memories,
            secs = format!("{:.2}", start.elapsed().as_secs_f64()),
            "Rebuilt space index from storage"
        );
        Ok(())
    }

    /// Index `vector` as the memory's only vector in this space, replacing any it had.
    pub(crate) fn add(&self, memory_id: &MemoryId, vector: Vec<f32>) -> Result<()> {
        let wants_quantizer = {
            let mut index = self.index.write();
            self.touch();
            let old = self
                .mapping
                .read()
                .memory_to_vectors
                .get(memory_id)
                .cloned()
                .unwrap_or_default();
            for vid in old {
                index.mark_deleted(vid);
            }
            let vid = index
                .add(vector)
                .with_context(|| format!("adding a vector to the {} index", self.space))?;
            self.mapping.write().insert(memory_id.clone(), vid);
            matches!(&*index, Store::Codes(c) if c.wants_quantizer())
        };
        if wants_quantizer && !self.training.swap(true, Ordering::AcqRel) {
            let trained = self.train_quantizer();
            self.training.store(false, Ordering::Release);
            if let Err(e) = trained {
                warn!(
                    "{}: quantizer training failed, staying exact: {e:#}",
                    self.space
                );
            }
        }
        Ok(())
    }

    /// Train the quantizer on a sample of the held exact vectors without holding the locks, then
    /// encode every exact vector held by then, adds made meanwhile included.
    fn train_quantizer(&self) -> Result<()> {
        let sample = match &*self.index.read() {
            Store::Codes(c) => c.training_sample(),
            Store::Graph(_) => None,
        };
        let Some(sample) = sample else {
            return Ok(());
        };
        let start = std::time::Instant::now();
        let quantizer = QuantizedIndex::train(self.space.dimension(), &sample)?;
        let mut index = self.index.write();
        if let Store::Codes(c) = &mut *index {
            if !c.is_quantized() {
                self.touch();
                c.install_quantizer(quantizer)?;
                info!(
                    space = %self.space,
                    vectors = c.live(),
                    secs = format!("{:.2}", start.elapsed().as_secs_f64()),
                    "Quantized space index"
                );
            }
        }
        Ok(())
    }

    /// Drop the memory's vectors. Returns whether it had any.
    pub(crate) fn remove(&self, memory_id: &MemoryId) -> bool {
        let mut index = self.index.write();
        let mut mapping = self.mapping.write();
        if !mapping.contains(memory_id) {
            return false;
        }
        self.touch();
        for vid in mapping.remove_all(memory_id) {
            index.mark_deleted(vid);
        }
        true
    }

    /// The memories nearest `query`, best first, one score per memory (its best vector),
    /// ties broken by id so a rank is stable across runs. `candidates` vectors are read from
    /// a graph; at most `limit` memories come back. Scores estimated from codes are replaced by
    /// the cosine to the memory's exact vector, read through `exact` once the locks are
    /// released; a candidate whose memory no longer holds one is dropped.
    pub(crate) fn search(
        &self,
        query: &[f32],
        candidates: usize,
        limit: usize,
        exact: &dyn Fn(&MemoryId) -> Option<Vec<f32>>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let (mut best, estimates) = {
            let index = self.index.read();
            let results = index
                .search(query, candidates, limit)
                .with_context(|| format!("searching the {} index", self.space))?;
            let mapping = self.mapping.read();
            let mut best: HashMap<MemoryId, f32> = HashMap::new();
            for (vector_id, similarity) in results {
                if let Some(memory_id) = mapping.get_memory_id(vector_id) {
                    best.entry(memory_id.clone())
                        .and_modify(|s| *s = s.max(similarity))
                        .or_insert(similarity);
                }
            }
            (best, index.estimates())
        };
        if estimates {
            best = best
                .into_keys()
                .filter_map(|memory_id| {
                    let vector = exact(&memory_id)?;
                    (vector.len() == query.len())
                        .then(|| (memory_id, cosine_similarity_inline(query, &vector)))
                })
                .collect();
        }
        let mut out: Vec<(MemoryId, f32)> = best.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(limit);
        Ok(out)
    }

    /// Drop soft-deleted vectors: a graph is rebuilt from its own live vectors, codes are
    /// renumbered.
    pub(crate) fn compact(&self) -> Result<()> {
        let mut index = self.index.write();
        let mut mapping = self.mapping.write();
        self.touch();
        let mut members: Vec<(MemoryId, Vec<u32>)> = mapping
            .memory_to_vectors
            .iter()
            .map(|(id, vids)| (id.clone(), vids.clone()))
            .collect();
        members.sort_by(|a, b| a.0.cmp(&b.0));
        let mut remapped = IdMapping::new();
        match &mut *index {
            Store::Graph(graph) => {
                let mut fresh = VamanaIndex::with_storage_path(
                    config_for(self.space.dimension()),
                    Some(self.dir.clone()),
                )?;
                let mut vectors = Vec::new();
                for (memory_id, old_ids) in members {
                    let mut new_ids = Vec::with_capacity(old_ids.len());
                    for old in old_ids {
                        vectors.push(graph.vector(old)?);
                        new_ids.push((vectors.len() - 1) as u32);
                    }
                    remapped.insert_chunks(memory_id, new_ids);
                }
                fresh.build(vectors)?;
                *graph = fresh;
            }
            Store::Codes(codes) => {
                let remap = codes.compact();
                for (memory_id, old_ids) in members {
                    let new_ids: Vec<u32> = old_ids
                        .into_iter()
                        .filter_map(|old| remap.get(old as usize).copied().flatten())
                        .collect();
                    if !new_ids.is_empty() {
                        remapped.insert_chunks(memory_id, new_ids);
                    }
                }
            }
        }
        *mapping = remapped;
        info!(space = %self.space, memories = mapping.len(), "Compacted space index");
        Ok(())
    }

    /// Save the index, then the mapping, then the marker that says they agree. An empty index
    /// saves nothing, and its stale files are removed, as is a file the other codec left.
    pub(crate) fn save(&self) -> Result<()> {
        let index = self.index.read();
        let mapping = self.mapping.read();
        let index_path = self.dir.join(self.codec.file());
        let mapping_path = self.dir.join(MAPPING_FILE);
        let _ = std::fs::remove_file(self.dir.join(CLEAN_MARKER));
        let _ = std::fs::remove_file(self.dir.join(self.codec.other().file()));
        if mapping.len() == 0 {
            let _ = std::fs::remove_file(&index_path);
            let _ = std::fs::remove_file(&mapping_path);
            return Ok(());
        }
        match &*index {
            Store::Graph(graph) => {
                let tmp = index_path.with_extension("idx.tmp");
                graph
                    .save_to_file(&tmp)
                    .with_context(|| format!("saving the {} index", self.space))?;
                std::fs::rename(&tmp, &index_path)?;
            }
            Store::Codes(codes) => codes
                .save_to_file(&index_path)
                .with_context(|| format!("saving the {} index", self.space))?,
        }
        let saved = SavedMapping {
            space: self.space.clone(),
            index_len: index.len(),
            memory_to_vectors: mapping.memory_to_vectors.clone(),
        };
        let tmp = mapping_path.with_extension("bin.tmp");
        std::fs::write(
            &tmp,
            bincode::serde::encode_to_vec(&saved, bincode::config::standard())?,
        )?;
        std::fs::rename(&tmp, &mapping_path)?;
        std::fs::write(self.dir.join(SPACE_FILE), self.space.as_str())?;
        std::fs::write(self.dir.join(CLEAN_MARKER), b"")?;
        self.dirty.store(false, Ordering::Release);
        info!(space = %self.space, memories = mapping.len(), "Saved space index");
        Ok(())
    }

    pub(crate) fn health(&self) -> SpaceIndexHealth {
        let index = self.index.read();
        let memories = self.mapping.read().len();
        match &*index {
            Store::Graph(graph) => SpaceIndexHealth {
                space: self.space.clone(),
                codec: self.codec.as_str(),
                memories,
                total_vectors: graph.len(),
                vector_bytes: graph.len() * self.space.dimension() * 4,
                incremental_inserts: graph.incremental_insert_count(),
                deleted_count: graph.deleted_count(),
                deletion_ratio: graph.deletion_ratio(),
                needs_rebuild: graph.needs_rebuild(),
                needs_compaction: graph.needs_compaction(),
            },
            Store::Codes(codes) => SpaceIndexHealth {
                space: self.space.clone(),
                codec: self.codec.as_str(),
                memories,
                total_vectors: codes.len(),
                vector_bytes: codes.memory_bytes(),
                incremental_inserts: codes.inserted_since_build(),
                deleted_count: codes.deleted_count(),
                deletion_ratio: codes.deletion_ratio(),
                needs_rebuild: codes.needs_retrain(),
                needs_compaction: codes.needs_compaction(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, seed: u32) -> Vec<f32> {
        let v: Vec<f32> = (0..dim)
            .map(|i| {
                ((seed
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(i as u32 * 40_503)
                    % 997) as f32)
                    + 1.0
            })
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    const CODECS: [SpaceCodec; 2] = [SpaceCodec::F32, SpaceCodec::Pq];

    fn no_exact(_: &MemoryId) -> Option<Vec<f32>> {
        None
    }

    #[test]
    fn a_saved_index_loads_with_the_mapping_it_was_saved_with() {
        for codec in CODECS {
            let dir = tempfile::tempdir().unwrap();
            let space = SpaceId::new("minilm-l6-v2", 16);
            let si = SpaceIndex::open(dir.path(), space.clone(), codec).unwrap();
            for n in 1..=20u128 {
                si.add(&id(n), unit(16, n as u32)).unwrap();
            }
            // Replace one vector and remove another: the drift the ordinal mapping could not see.
            si.add(&id(3), unit(16, 333)).unwrap();
            assert!(si.remove(&id(7)));
            si.save().unwrap();

            let reopened = SpaceIndex::open(dir.path(), space, codec).unwrap();
            assert!(reopened.load().unwrap(), "{codec:?}");
            assert_eq!(reopened.len(), 19);
            assert!(!reopened.contains(&id(7)));
            let hit = reopened.search(&unit(16, 333), 40, 1, &no_exact).unwrap();
            assert_eq!(hit[0].0, id(3), "the replaced vector maps to its memory");
            let hit = reopened.search(&unit(16, 12), 40, 1, &no_exact).unwrap();
            assert_eq!(hit[0].0, id(12));
        }
    }

    #[test]
    fn a_change_after_the_save_forces_a_rebuild_on_the_next_open() {
        for codec in CODECS {
            let dir = tempfile::tempdir().unwrap();
            let space = SpaceId::new("minilm-l6-v2", 16);
            let si = SpaceIndex::open(dir.path(), space.clone(), codec).unwrap();
            si.add(&id(1), unit(16, 1)).unwrap();
            si.save().unwrap();
            si.add(&id(2), unit(16, 2)).unwrap();
            drop(si);
            let reopened = SpaceIndex::open(dir.path(), space, codec).unwrap();
            assert!(
                !reopened.load().unwrap(),
                "unsaved changes make the files stale"
            );
        }
    }

    #[test]
    fn an_index_built_for_another_space_is_not_loaded() {
        for codec in CODECS {
            let dir = tempfile::tempdir().unwrap();
            let space = SpaceId::new("minilm-l6-v2", 16);
            let si = SpaceIndex::open(dir.path(), space.clone(), codec).unwrap();
            si.add(&id(1), unit(16, 1)).unwrap();
            si.save().unwrap();
            std::fs::write(dir_for(dir.path(), &space).join(SPACE_FILE), "other@16").unwrap();
            let reopened = SpaceIndex::open(dir.path(), space, codec).unwrap();
            assert!(!reopened.load().unwrap());
        }
    }

    #[test]
    fn an_index_saved_under_one_codec_is_rebuilt_under_the_other() {
        let dir = tempfile::tempdir().unwrap();
        let space = SpaceId::new("minilm-l6-v2", 16);
        let si = SpaceIndex::open(dir.path(), space.clone(), SpaceCodec::F32).unwrap();
        si.add(&id(1), unit(16, 1)).unwrap();
        si.save().unwrap();
        let pq = SpaceIndex::open(dir.path(), space.clone(), SpaceCodec::Pq).unwrap();
        assert!(!pq.load().unwrap());
        pq.add(&id(1), unit(16, 1)).unwrap();
        pq.save().unwrap();
        let space_dir = dir_for(dir.path(), &space);
        assert!(space_dir.join(CODES_FILE).exists());
        assert!(
            !space_dir.join(INDEX_FILE).exists(),
            "the other codec's file goes at the save"
        );
    }

    fn clustered(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let centres: Vec<Vec<f32>> = (0..24)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let v: Vec<f32> = centres[i % centres.len()]
                    .iter()
                    .map(|x| x + rng.gen_range(-0.4..0.4))
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.into_iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn a_pq_index_quantizes_at_its_threshold_and_scores_by_the_exact_vectors() {
        use crate::vector_db::quantized::TRAIN_AT;
        let dir = tempfile::tempdir().unwrap();
        let dim = 32;
        let space = SpaceId::new("minilm-l6-v2", dim);
        let vectors = clustered(TRAIN_AT + 200, dim);
        let stored: HashMap<MemoryId, Vec<f32>> = vectors
            .iter()
            .enumerate()
            .map(|(n, v)| (id(n as u128 + 1), v.clone()))
            .collect();
        let si = SpaceIndex::open(dir.path(), space.clone(), SpaceCodec::Pq).unwrap();
        for (n, v) in vectors.iter().enumerate() {
            si.add(&id(n as u128 + 1), v.clone()).unwrap();
        }
        let health = si.health();
        assert_eq!(health.codec, CODEC_PQ);
        assert!(
            health.vector_bytes < vectors.len() * dim * 4 / 2,
            "codes, not vectors: {} bytes",
            health.vector_bytes
        );

        let exact = |memory: &MemoryId| stored.get(memory).cloned();
        let probe = id(78);
        let hits = si.search(&stored[&probe], 0, 10, &exact).unwrap();
        assert_eq!(hits[0].0, probe);
        for (memory, score) in &hits {
            let truth = cosine_similarity_inline(&stored[&probe], &stored[memory]);
            assert!((score - truth).abs() < 1e-5, "reranked by the exact vector");
        }
        assert!(
            si.search(&stored[&probe], 0, 10, &no_exact)
                .unwrap()
                .is_empty(),
            "a candidate no record holds a vector for is dropped"
        );

        si.save().unwrap();
        let reopened = SpaceIndex::open(dir.path(), space, SpaceCodec::Pq).unwrap();
        assert!(reopened.load().unwrap());
        assert_eq!(
            reopened.search(&stored[&probe], 0, 10, &exact).unwrap(),
            hits
        );
    }

    #[test]
    fn compaction_keeps_every_live_memory_findable() {
        for codec in CODECS {
            let dir = tempfile::tempdir().unwrap();
            let space = SpaceId::new("minilm-l6-v2", 16);
            let vectors = clustered(50, 16);
            let si = SpaceIndex::open(dir.path(), space, codec).unwrap();
            for (n, v) in vectors.iter().enumerate() {
                si.add(&id(n as u128), v.clone()).unwrap();
            }
            for n in 0..20u128 {
                assert!(si.remove(&id(n)));
            }
            si.compact().unwrap();
            assert_eq!(si.health().deleted_count, 0, "{codec:?}");
            assert_eq!(si.len(), 30);
            for (n, v) in vectors.iter().enumerate().skip(20) {
                let hit = si.search(v, 100, 1, &no_exact).unwrap();
                assert_eq!(hit[0].0, id(n as u128), "{codec:?}");
            }
        }
    }

    #[test]
    fn two_ids_that_slug_alike_get_two_directories() {
        let root = Path::new("/tmp/x");
        assert_ne!(
            dir_for(root, &SpaceId::new("a.b", 8)),
            dir_for(root, &SpaceId::new("a-b", 8))
        );
    }
}
