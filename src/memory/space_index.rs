//! One vector index per embedding space, for every space but the primary index's.
//!
//! The primary index keeps its own layout (`vector_index/`, mappings in the store) because its
//! mappings are authoritative and its chunk vectors are not stored on the record. Every other
//! space gets a `SpaceIndex` under `space_indexes/<slug>/`: the Vamana file, the mapping that
//! names which memory each vector belongs to, the space id it was built for, and a marker that
//! says the three were saved together. The mapping is saved with the index rather than
//! re-derived from insertion order, which was the secondary index's drift: a vector added after
//! a save, or soft-deleted, shifted every ordinal behind it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::types::MemoryId;
use crate::embeddings::space::SpaceId;
use crate::vector_db::vamana::{VamanaConfig, VamanaIndex};

/// Directory under a tenant's storage holding one subdirectory per space.
pub const SPACE_INDEX_ROOT: &str = "space_indexes";
const INDEX_FILE: &str = "vamana.idx";
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
    pub memories: usize,
    pub total_vectors: usize,
    pub incremental_inserts: usize,
    pub deleted_count: usize,
    pub deletion_ratio: f32,
    pub needs_rebuild: bool,
    pub needs_compaction: bool,
}

/// One space's index and mapping.
///
/// Lock order inside: `index` before `mapping`. The retrieval engine never holds two
/// spaces' locks at once.
pub(crate) struct SpaceIndex {
    space: SpaceId,
    dir: PathBuf,
    index: RwLock<VamanaIndex>,
    mapping: RwLock<IdMapping>,
    dirty: AtomicBool,
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
    /// An empty index for `space`, with its directory created. Nothing is read from disk:
    /// `load` does that.
    pub(crate) fn open(storage_path: &Path, space: SpaceId) -> Result<Self> {
        let dir = dir_for(storage_path, &space);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating the {space} index directory"))?;
        let index =
            VamanaIndex::with_storage_path(config_for(space.dimension()), Some(dir.clone()))
                .with_context(|| format!("initialising the {space} index"))?;
        Ok(Self {
            space,
            dir,
            index: RwLock::new(index),
            mapping: RwLock::new(IdMapping::new()),
            // Clean until the first change, which clears whatever marker a past save left.
            dirty: AtomicBool::new(false),
        })
    }

    pub(crate) fn space(&self) -> &SpaceId {
        &self.space
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

    /// Load the saved index and mapping when they were saved together for this space. Returns
    /// whether the index is now the saved one; `false` leaves the index empty for a rebuild.
    pub(crate) fn load(&self) -> Result<bool> {
        let index_path = self.dir.join(INDEX_FILE);
        let mapping_path = self.dir.join(MAPPING_FILE);
        if !self.dir.join(CLEAN_MARKER).exists() || !index_path.exists() || !mapping_path.exists() {
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
        if !VamanaIndex::verify_index_file(&index_path)? {
            warn!("{}: saved index fails its checksum; rebuilding", self.space);
            return Ok(false);
        }
        let mut loaded = VamanaIndex::load_from_file(&index_path)?;
        if loaded.config.dimension != self.space.dimension() {
            warn!(
                "{}: saved index is {}d; rebuilding",
                self.space, loaded.config.dimension
            );
            return Ok(false);
        }
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
        let deleted = loaded.deleted_ids.read().clone();
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
        // Persistence stores a construction default; search uses the runtime list size.
        loaded.config.search_list_size = 100;
        let mut mapping = IdMapping::new();
        for (memory_id, ids) in saved.memory_to_vectors {
            mapping.insert_chunks(memory_id, ids);
        }
        let memories = mapping.len();
        *self.index.write() = loaded;
        *self.mapping.write() = mapping;
        self.dirty.store(false, Ordering::Release);
        info!(space = %self.space, vectors = count, memories, "Loaded saved space index");
        if self.index.read().needs_compaction() {
            self.compact()?;
        }
        Ok(true)
    }

    /// Replace the index with one built from `members`, one vector per memory.
    pub(crate) fn rebuild(&self, members: Vec<(MemoryId, Vec<f32>)>) -> Result<()> {
        let start = std::time::Instant::now();
        let mut index = VamanaIndex::with_storage_path(
            config_for(self.space.dimension()),
            Some(self.dir.clone()),
        )?;
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
            memories,
            secs = format!("{:.2}", start.elapsed().as_secs_f64()),
            "Rebuilt space index from storage"
        );
        Ok(())
    }

    /// Index `vector` as the memory's only vector in this space, replacing any it had.
    pub(crate) fn add(&self, memory_id: &MemoryId, vector: Vec<f32>) -> Result<()> {
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
            .add_vector(vector)
            .with_context(|| format!("adding a vector to the {} index", self.space))?;
        self.mapping.write().insert(memory_id.clone(), vid);
        Ok(())
    }

    /// Drop the memory's vectors. Returns whether it had any.
    pub(crate) fn remove(&self, memory_id: &MemoryId) -> bool {
        let index = self.index.read();
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
    /// the index; at most `limit` memories come back.
    pub(crate) fn search(
        &self,
        query: &[f32],
        candidates: usize,
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let index = self.index.read();
        let results = index
            .search(query, candidates)
            .with_context(|| format!("searching the {} index", self.space))?;
        let mapping = self.mapping.read();
        let mut best: HashMap<MemoryId, f32> = HashMap::new();
        for (vector_id, distance) in results {
            // NormalizedDotProduct: the index returns -dot, so similarity is -distance.
            let similarity = -distance;
            if let Some(memory_id) = mapping.get_memory_id(vector_id) {
                best.entry(memory_id.clone())
                    .and_modify(|s| *s = s.max(similarity))
                    .or_insert(similarity);
            }
        }
        let mut out: Vec<(MemoryId, f32)> = best.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(limit);
        Ok(out)
    }

    /// Rebuild from the index's own live vectors, dropping soft-deleted ones.
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
        let mut fresh = VamanaIndex::with_storage_path(
            config_for(self.space.dimension()),
            Some(self.dir.clone()),
        )?;
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
        *index = fresh;
        *mapping = remapped;
        info!(space = %self.space, memories = mapping.len(), "Compacted space index");
        Ok(())
    }

    /// Save the index, then the mapping, then the marker that says they agree. An empty index
    /// saves nothing, and its stale files are removed.
    pub(crate) fn save(&self) -> Result<()> {
        let index = self.index.read();
        let mapping = self.mapping.read();
        let index_path = self.dir.join(INDEX_FILE);
        let mapping_path = self.dir.join(MAPPING_FILE);
        let _ = std::fs::remove_file(self.dir.join(CLEAN_MARKER));
        if mapping.len() == 0 {
            let _ = std::fs::remove_file(&index_path);
            let _ = std::fs::remove_file(&mapping_path);
            return Ok(());
        }
        let tmp = index_path.with_extension("idx.tmp");
        index
            .save_to_file(&tmp)
            .with_context(|| format!("saving the {} index", self.space))?;
        std::fs::rename(&tmp, &index_path)?;
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
        SpaceIndexHealth {
            space: self.space.clone(),
            memories: self.mapping.read().len(),
            total_vectors: index.len(),
            incremental_inserts: index.incremental_insert_count(),
            deleted_count: index.deleted_count(),
            deletion_ratio: index.deletion_ratio(),
            needs_rebuild: index.needs_rebuild(),
            needs_compaction: index.needs_compaction(),
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

    #[test]
    fn a_saved_index_loads_with_the_mapping_it_was_saved_with() {
        let dir = tempfile::tempdir().unwrap();
        let space = SpaceId::new("minilm-l6-v2", 16);
        let si = SpaceIndex::open(dir.path(), space.clone()).unwrap();
        for n in 1..=20u128 {
            si.add(&id(n), unit(16, n as u32)).unwrap();
        }
        // Replace one vector and remove another: the drift the ordinal mapping could not see.
        si.add(&id(3), unit(16, 333)).unwrap();
        assert!(si.remove(&id(7)));
        si.save().unwrap();

        let reopened = SpaceIndex::open(dir.path(), space).unwrap();
        assert!(reopened.load().unwrap());
        assert_eq!(reopened.len(), 19);
        assert!(!reopened.contains(&id(7)));
        let hit = reopened.search(&unit(16, 333), 40, 1).unwrap();
        assert_eq!(hit[0].0, id(3), "the replaced vector maps to its memory");
        let hit = reopened.search(&unit(16, 12), 40, 1).unwrap();
        assert_eq!(hit[0].0, id(12));
    }

    #[test]
    fn a_change_after_the_save_forces_a_rebuild_on_the_next_open() {
        let dir = tempfile::tempdir().unwrap();
        let space = SpaceId::new("minilm-l6-v2", 16);
        let si = SpaceIndex::open(dir.path(), space.clone()).unwrap();
        si.add(&id(1), unit(16, 1)).unwrap();
        si.save().unwrap();
        si.add(&id(2), unit(16, 2)).unwrap();
        drop(si);
        let reopened = SpaceIndex::open(dir.path(), space).unwrap();
        assert!(
            !reopened.load().unwrap(),
            "unsaved changes make the files stale"
        );
    }

    #[test]
    fn an_index_built_for_another_space_is_not_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let space = SpaceId::new("minilm-l6-v2", 16);
        let si = SpaceIndex::open(dir.path(), space.clone()).unwrap();
        si.add(&id(1), unit(16, 1)).unwrap();
        si.save().unwrap();
        std::fs::write(dir_for(dir.path(), &space).join(SPACE_FILE), "other@16").unwrap();
        let reopened = SpaceIndex::open(dir.path(), space).unwrap();
        assert!(!reopened.load().unwrap());
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
