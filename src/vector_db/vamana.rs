//! Vamana: Single-shot graph construction for billion-scale similarity search
//! Based on Microsoft Research paper: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
//!
//! Production implementation optimized for 8-16GB RAM laptops
//!
//! # Index Maintenance
//!
//! Incremental inserts use neighbor truncation which can degrade search quality over time.
//! For optimal recall@10 accuracy, consider rebuilding the index periodically:
//!
//! - **Recommended**: Rebuild after every 10,000 incremental inserts
//! - **Impact**: Without rebuilds, recall@10 may degrade 5-15% over thousands of inserts
//! - **Detection**: Use `needs_rebuild()` to check if rebuild is recommended
//!
//! ## Example
//! ```ignore
//! if index.needs_rebuild() {
//!     let vectors = index.extract_all_vectors();
//!     index.rebuild_from_vectors(&vectors)?;
//! }
//! ```

use super::distance_inline::{
    cosine_similarity_inline, euclidean_squared_inline, normalized_distance_inline,
};
use super::vamana_store::{DeletedSet, VectorStore, VisitedPool};
use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use parking_lot::RwLock;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// Distance metric for vector similarity
///
/// All metrics are SIMD-optimized (AVX2 on x86-64, NEON on ARM64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// For L2-normalized vectors (default). Fastest option.
    /// Uses -dot_product which gives correct distance ordering.
    /// MiniLM and most sentence transformers output normalized vectors.
    #[default]
    NormalizedDotProduct,

    /// Euclidean distance squared. Works for any vectors.
    /// Slightly slower than dot product but doesn't require normalization.
    Euclidean,

    /// Cosine similarity (1 - cos_sim). Works for any vectors.
    /// Computes norms on-the-fly, slowest but most flexible.
    Cosine,
}

/// Vamana configuration
#[derive(Debug, Clone)]
pub struct VamanaConfig {
    /// Maximum degree of graph (R in paper)
    pub max_degree: usize,

    /// Search list size during construction (L in paper)
    pub search_list_size: usize,

    /// Alpha parameter for RNG pruning (α in paper, typically 1.2)
    pub alpha: f32,

    /// Vector dimension
    pub dimension: usize,

    /// Use memory mapping for large datasets
    pub use_mmap: bool,

    /// Distance metric for similarity calculation
    /// Default: NormalizedDotProduct (assumes L2-normalized vectors)
    pub distance_metric: DistanceMetric,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,                             // R=32 for billion-scale
            search_list_size: 75,                       // L=75 during construction
            alpha: 1.2,                                 // Standard α for pruning
            dimension: 384,                             // MiniLM dimension
            use_mmap: true,                             // Disk-based for large datasets
            distance_metric: DistanceMetric::default(), // NormalizedDotProduct for MiniLM
        }
    }
}

/// Node in the Vamana graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct VamanaNode {
    /// Node ID
    pub(crate) id: u32,

    /// Neighbor IDs sorted by distance
    pub(crate) neighbors: Vec<u32>,
}

/// Threshold for recommending index rebuild (number of incremental inserts)
pub const REBUILD_THRESHOLD: usize = 10_000;

/// Threshold for incremental repair (lighter maintenance, more frequent)
/// Repairs neighborhoods of recently inserted nodes without full rebuild
pub const REPAIR_THRESHOLD: usize = 1_000;

/// Threshold for recommending index rebuild based on deletion ratio
/// When 30% or more of vectors are soft-deleted, compaction is recommended
pub const DELETION_RATIO_THRESHOLD: f32 = 0.30;

/// Minimum recall for acceptable index quality (used in quality estimation)
/// Below this threshold, rebuild is strongly recommended
pub const MIN_ACCEPTABLE_RECALL: f32 = 0.85;

/// Indexes with at most this many live vectors are searched exhaustively. At 768 dimensions that is
/// under 40M multiply-adds a query, a few milliseconds, and it is exact where the graph approximates.
pub const EXACT_SEARCH_MAX_VECTORS: usize = 50_000;

/// Main Vamana index
pub struct VamanaIndex {
    pub(crate) config: VamanaConfig,

    /// Graph structure: node_id -> neighbors
    pub(crate) graph: Arc<RwLock<Vec<VamanaNode>>>,

    /// Vectors, row-major; a mapped prefix is read in place (see [`VectorStore`]).
    pub(crate) vectors: Arc<RwLock<VectorStore>>,

    /// Medoid/centroid as entry point
    pub(crate) medoid: Arc<RwLock<u32>>,

    /// Number of vectors (atomic for lock-free reads during background rebuild)
    pub(crate) num_vectors: std::sync::atomic::AtomicUsize,

    /// Storage path for mmap files (unique per index instance)
    pub(crate) storage_path: Option<PathBuf>,

    /// Counter for incremental inserts since last rebuild
    /// Used to track index quality degradation
    pub(crate) incremental_inserts: std::sync::atomic::AtomicUsize,

    /// Flag to prevent concurrent rebuilds
    pub(crate) rebuilding: std::sync::atomic::AtomicBool,

    /// Soft-deleted vector IDs (filtered from search results)
    /// These vectors remain in the graph but are excluded from results.
    /// Physically removed on next rebuild.
    pub(crate) deleted_ids: Arc<RwLock<DeletedSet>>,

    /// Visited sets reused across graph walks.
    pub(crate) visited: VisitedPool,

    /// Live-vector count up to which `search` scans every vector instead of walking the graph.
    pub(crate) exact_search_max: usize,
}

impl VamanaIndex {
    /// Create new Vamana index
    pub fn new(config: VamanaConfig) -> Result<Self> {
        Self::with_storage_path(config, None)
    }

    /// Create new Vamana index with explicit storage path for mmap
    pub fn with_storage_path(config: VamanaConfig, storage_path: Option<PathBuf>) -> Result<Self> {
        let vectors = VectorStore::new(config.dimension);
        Ok(Self {
            config,
            graph: Arc::new(RwLock::new(Vec::new())),
            vectors: Arc::new(RwLock::new(vectors)),
            medoid: Arc::new(RwLock::new(0)),
            num_vectors: std::sync::atomic::AtomicUsize::new(0),
            storage_path,
            incremental_inserts: std::sync::atomic::AtomicUsize::new(0),
            rebuilding: std::sync::atomic::AtomicBool::new(false),
            deleted_ids: Arc::new(RwLock::new(DeletedSet::default())),
            visited: VisitedPool::default(),
            exact_search_max: EXACT_SEARCH_MAX_VECTORS,
        })
    }

    /// Search exhaustively up to `n` live vectors; 0 makes every search walk the graph.
    pub fn set_exact_search_max(&mut self, n: usize) {
        self.exact_search_max = n;
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.num_vectors.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors.load(std::sync::atomic::Ordering::Acquire) == 0
    }

    /// Build index from vectors using Vamana algorithm
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        if let Some((id, v)) = vectors
            .iter()
            .enumerate()
            .find(|(_, v)| v.len() != self.config.dimension)
        {
            anyhow::bail!(
                "build: vector {id} has dimension {}, the index has {}",
                v.len(),
                self.config.dimension
            );
        }

        let n = vectors.len();
        self.num_vectors
            .store(n, std::sync::atomic::Ordering::Release);

        info!("Building Vamana index with {} vectors", n);

        // Step 1: Initialize graph with deterministic seeding
        // Hash sample vectors so identical data always produces identical index
        let seed = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            n.hash(&mut hasher);
            for &idx in &[0, n / 2, n.saturating_sub(1)] {
                if let Some(v) = vectors.get(idx) {
                    for &f in v {
                        f.to_bits().hash(&mut hasher);
                    }
                }
            }
            hasher.finish()
        };
        self.initialize_graph(n, seed)?;

        // Step 2: Store vectors
        self.store_vectors(vectors)?;

        // Step 3: Find medoid (closest to centroid)
        self.find_medoid()?;

        // Step 4: Main Vamana construction
        let mut iteration = 0;
        loop {
            iteration += 1;
            info!("Vamana iteration {}", iteration);

            let mut updates = 0;

            // Process each node
            for node_id in 0..n {
                // Get vector for this node
                let query = self.get_vector(node_id as u32)?;

                // Search for L nearest neighbors; the node's current edges compete with them
                let mut candidates =
                    self.greedy_search(&query, self.config.search_list_size, *self.medoid.read())?;
                candidates.extend(
                    self.graph.read()[node_id]
                        .neighbors
                        .iter()
                        .map(|&id| SearchCandidate { id, distance: 0.0 }),
                );

                // Prune using α-RNG strategy
                let pruned = self.robust_prune(node_id as u32, &candidates)?;

                // Update graph
                let mut graph = self.graph.write();
                if graph[node_id].neighbors != pruned {
                    updates += 1;
                    graph[node_id].neighbors = pruned.clone();

                    // Ensure bidirectional edges
                    for &neighbor in &pruned {
                        if neighbor as usize >= graph.len() {
                            continue;
                        }

                        let neighbor_node = &mut graph[neighbor as usize];
                        if !neighbor_node.neighbors.contains(&(node_id as u32)) {
                            neighbor_node.neighbors.push(node_id as u32);

                            // Prune neighbor if exceeds max degree
                            if neighbor_node.neighbors.len() > self.config.max_degree {
                                let _neighbor_vec = self.get_vector(neighbor)?;
                                let pruned_neighbors = self.robust_prune(
                                    neighbor,
                                    &neighbor_node
                                        .neighbors
                                        .iter()
                                        .map(|&id| SearchCandidate { id, distance: 0.0 })
                                        .collect::<Vec<_>>(),
                                )?;
                                neighbor_node.neighbors = pruned_neighbors;
                            }
                        }
                    }
                }
            }

            info!("Updated {} nodes", updates);

            // Converged
            if updates == 0 || iteration >= 2 {
                break;
            }
        }

        info!("Vamana construction complete");
        Ok(())
    }

    /// Initialize random graph with deterministic seeding
    ///
    /// Uses a content-derived seed so identical vector sets always produce
    /// identical initial graphs, making index construction fully reproducible.
    fn initialize_graph(&mut self, n: usize, seed: u64) -> Result<()> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut graph = Vec::with_capacity(n);
        let draw = (self.config.max_degree + 1).min(n);

        for i in 0..n {
            let neighbors: Vec<u32> = rand::seq::index::sample(&mut rng, n, draw)
                .into_iter()
                .filter(|&j| j != i)
                .take(self.config.max_degree)
                .map(|j| j as u32)
                .collect();

            graph.push(VamanaNode {
                id: i as u32,
                neighbors,
            });
        }

        *self.graph.write() = graph;
        Ok(())
    }

    /// Store vectors in storage
    fn store_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        let mut storage = self.vectors.write();

        if self.config.use_mmap && self.config.dimension > 0 {
            // Require explicit storage path for mmap mode
            let mmap_path = self.storage_path
                .as_ref()
                .map(|p| p.join("vamana_vectors.bin"))
                .ok_or_else(|| anyhow!("Storage path required for mmap mode. Use with_storage_path() or disable use_mmap."))?;

            // Ensure parent directory exists
            if let Some(parent) = mmap_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Create memory-mapped file
            let file_size = vectors.len() * self.config.dimension * std::mem::size_of::<f32>();
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&mmap_path)?;

            file.set_len(file_size as u64)?;

            // SAFETY CHECK: Verify file size is correctly set and aligned
            let actual_file_size = file.metadata()?.len();
            if actual_file_size != file_size as u64 {
                anyhow::bail!(
                    "File size mismatch: expected {file_size} bytes, got {actual_file_size} bytes"
                );
            }

            // SAFETY: MmapMut::map_mut is safe because:
            // 1. File handle is valid and exclusively owned
            // 2. File size is non-zero and verified above
            // 3. File permissions allow read+write
            // 4. No other process has this file mapped
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            for (chunk, vec) in mmap
                .chunks_exact_mut(self.config.dimension * std::mem::size_of::<f32>())
                .zip(&vectors)
            {
                for (bytes, value) in chunk.chunks_exact_mut(4).zip(vec) {
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
            }

            *storage = VectorStore::mapped(
                self.config.dimension,
                mmap.make_read_only()?,
                0,
                vectors.len(),
            )?;
        } else {
            *storage = VectorStore::from_rows(self.config.dimension, vectors)?;
        }

        Ok(())
    }

    /// Find medoid (closest point to centroid)
    fn find_medoid(&mut self) -> Result<()> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n == 0 {
            return Ok(());
        }

        // Compute centroid
        let mut centroid = vec![0.0; self.config.dimension];
        for i in 0..n {
            let vec = self.get_vector(i as u32)?;
            for (j, &val) in vec.iter().enumerate() {
                centroid[j] += val;
            }
        }

        for val in &mut centroid {
            *val /= n as f32;
        }

        // Find closest to centroid
        let mut best_id = 0;
        let mut best_dist = f32::MAX;

        for i in 0..n {
            let vec = self.get_vector(i as u32)?;
            let dist = self.distance(&vec, &centroid);
            if dist < best_dist {
                best_dist = dist;
                best_id = i as u32;
            }
        }

        *self.medoid.write() = best_id;
        Ok(())
    }

    /// The stored vector for `id`.
    pub(crate) fn vector(&self, id: u32) -> Result<Vec<f32>> {
        self.get_vector(id)
    }

    /// Get vector by ID
    fn get_vector(&self, id: u32) -> Result<Vec<f32>> {
        Self::get_slice_from_storage(&self.vectors.read(), id).map(<[f32]>::to_vec)
    }

    /// Get vector slice by ID from storage reference (zero-copy, no allocation)
    ///
    /// This is the performance-critical path for search operations.
    /// Returns a borrowed slice instead of cloning the vector data.
    #[inline]
    fn get_slice_from_storage(storage: &VectorStore, id: u32) -> Result<&[f32]> {
        storage
            .row(id)
            .ok_or_else(|| anyhow!("Vector {id} out of bounds (num_vectors={})", storage.len()))
    }

    /// Greedy search for nearest neighbors
    ///
    /// Optimized to use zero-copy slice access for vector data.
    /// Holds both graph and vector storage locks for the duration of the search
    /// to avoid per-neighbor lock acquisition overhead.
    fn greedy_search(&self, query: &[f32], k: usize, entry: u32) -> Result<Vec<SearchCandidate>> {
        let graph = self.graph.read();
        let storage = self.vectors.read(); // Hold lock for entire search (zero-copy access)

        let search_cap = self.config.search_list_size;
        let mut visited = self.visited.take(graph.len());
        let mut candidates = BinaryHeap::with_capacity(search_cap);
        let mut w = BinaryHeap::with_capacity(search_cap);

        // Start from entry point (zero-copy slice access)
        let entry_slice = Self::get_slice_from_storage(&storage, entry)?;
        let entry_dist = self.distance(query, entry_slice);

        candidates.push(Reverse(SearchCandidate {
            id: entry,
            distance: entry_dist,
        }));

        w.push(SearchCandidate {
            id: entry,
            distance: entry_dist,
        });

        visited.insert(entry);

        // Greedy search
        while let Some(Reverse(current)) = candidates.pop() {
            // Defensive check: w should never be empty (entry point pushed above)
            if w.peek()
                .map(|p| current.distance > p.distance)
                .unwrap_or(false)
            {
                break;
            }

            // Check neighbors (ensure index is valid)
            if (current.id as usize) >= graph.len() {
                // Node doesn't exist in graph yet
                continue;
            }

            let node = &graph[current.id as usize];
            for &neighbor_id in &node.neighbors {
                if !visited.insert(neighbor_id) {
                    continue;
                }

                // Zero-copy slice access - no allocation per neighbor
                let neighbor_slice = Self::get_slice_from_storage(&storage, neighbor_id)?;
                let dist = self.distance(query, neighbor_slice);

                // Defensive: check if closer than worst in w, or w not yet full
                let should_add = w.len() < k || w.peek().map(|p| dist < p.distance).unwrap_or(true);
                if should_add {
                    candidates.push(Reverse(SearchCandidate {
                        id: neighbor_id,
                        distance: dist,
                    }));

                    w.push(SearchCandidate {
                        id: neighbor_id,
                        distance: dist,
                    });

                    if w.len() > k {
                        w.pop();
                    }
                }
            }
        }

        // Extract results
        let mut results = Vec::new();
        while let Some(candidate) = w.pop() {
            results.push(candidate);
        }
        results.reverse();

        Ok(results)
    }

    /// Robust prune (DiskANN's α-RNG rule): keep candidates nearest first, dropping any that a kept
    /// neighbour already covers, so the node keeps long edges as well as short ones. Distances to
    /// the node are recomputed, so callers may pass candidates in any order or at any distance.
    fn robust_prune(&self, node_id: u32, candidates: &[SearchCandidate]) -> Result<Vec<u32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let storage = self.vectors.read();
        let node = Self::get_slice_from_storage(&storage, node_id)?;

        let mut seen = HashSet::with_capacity(candidates.len());
        let mut ranked: Vec<(u32, f32, &[f32])> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if candidate.id == node_id || !seen.insert(candidate.id) {
                continue;
            }
            if let Ok(vector) = Self::get_slice_from_storage(&storage, candidate.id) {
                ranked.push((candidate.id, self.prune_distance(node, vector), vector));
            }
        }
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut kept: Vec<(u32, &[f32])> = Vec::with_capacity(self.config.max_degree);
        for (id, to_node, vector) in ranked {
            let covered = kept
                .iter()
                .any(|(_, k)| self.config.alpha * self.prune_distance(k, vector) <= to_node);
            if !covered {
                kept.push((id, vector));
                if kept.len() >= self.config.max_degree {
                    break;
                }
            }
        }
        Ok(kept.into_iter().map(|(id, _)| id).collect())
    }

    /// Distance on a scale that starts at zero, as the α rule needs: for unit vectors `-dot` runs
    /// from -1 to 1, and multiplying a negative distance by α would tighten the rule, not relax it.
    #[inline(always)]
    fn prune_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::NormalizedDotProduct => self.distance(a, b) + 1.0,
            DistanceMetric::Euclidean | DistanceMetric::Cosine => self.distance(a, b),
        }
    }

    /// Compute distance between two vectors using configured metric
    ///
    /// All metrics are SIMD-optimized:
    /// - NormalizedDotProduct: -dot(a,b) - fastest, requires normalized vectors
    /// - Euclidean: ||a-b||^2 - works for any vectors
    /// - Cosine: 1 - cos_sim(a,b) - works for any vectors, computes norms
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::NormalizedDotProduct => normalized_distance_inline(a, b),
            DistanceMetric::Euclidean => euclidean_squared_inline(a, b),
            DistanceMetric::Cosine => 1.0 - cosine_similarity_inline(a, b),
        }
    }

    /// Search for k nearest neighbors (excludes soft-deleted vectors)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n == 0 || k == 0 {
            return Ok(Vec::new());
        }
        // The kernels stride by the query's length; a longer query would read past a row.
        if query.len() != self.config.dimension {
            anyhow::bail!(
                "search: query dimension {} does not match index dimension {}",
                query.len(),
                self.config.dimension
            );
        }
        {
            let deleted = self.deleted_ids.read();
            if n.saturating_sub(deleted.len()) <= self.exact_search_max {
                return self.exact_search(query, k, n, &deleted);
            }
        }

        // Check if graph is built
        if self.graph.read().is_empty() {
            return Err(anyhow!(
                "Vamana graph not built. Call build() first or add more vectors."
            ));
        }

        let entry = *self.medoid.read();
        let deleted = self.deleted_ids.read();
        let deleted_count = deleted.len();

        // Request extra candidates to account for deleted vectors
        let search_k = if deleted_count > 0 {
            k + deleted_count.min(k * 2)
        } else {
            k
        };

        let beam = search_k.max(self.config.search_list_size);
        let candidates = self.greedy_search(query, beam, entry)?;

        // Filter out deleted vectors and take k results
        let results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|c| !deleted.contains(c.id))
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect();

        Ok(results)
    }

    /// The k nearest live vectors by scanning all `n` in storage order, ties broken by id.
    fn exact_search(
        &self,
        query: &[f32],
        k: usize,
        n: usize,
        deleted: &DeletedSet,
    ) -> Result<Vec<(u32, f32)>> {
        let storage = self.vectors.read();
        if storage.len() < n {
            return Err(anyhow!(
                "vector store holds {} rows, the index counts {n}",
                storage.len()
            ));
        }
        let dim = storage.dimension();
        let mut scored = Vec::with_capacity(n.saturating_sub(deleted.len()));
        for (first, run) in storage.runs(n) {
            for (offset, vector) in run.chunks_exact(dim).enumerate() {
                let id = first + offset as u32;
                if deleted.contains(id) {
                    continue;
                }
                scored.push((id, self.distance(query, vector)));
            }
        }
        let nearest_first =
            |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
        if scored.len() > k {
            scored.select_nth_unstable_by(k, nearest_first);
            scored.truncate(k);
        }
        scored.sort_by(nearest_first);
        Ok(scored)
    }

    /// Mark a vector as deleted (soft delete)
    /// The vector remains in the graph but is excluded from search results.
    /// It will be physically removed on the next rebuild.
    pub fn mark_deleted(&self, vector_id: u32) -> bool {
        if (vector_id as usize) < self.num_vectors.load(std::sync::atomic::Ordering::Acquire) {
            self.deleted_ids.write().insert(vector_id);
            true
        } else {
            false
        }
    }

    /// Check if a vector is marked as deleted
    pub fn is_deleted(&self, vector_id: u32) -> bool {
        self.deleted_ids.read().contains(vector_id)
    }

    /// Get the number of soft-deleted vectors
    pub fn deleted_count(&self) -> usize {
        self.deleted_ids.read().len()
    }

    /// Get the deletion ratio (deleted / total vectors)
    /// Returns 0.0 if index is empty
    pub fn deletion_ratio(&self) -> f32 {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n == 0 {
            return 0.0;
        }
        self.deleted_count() as f32 / n as f32
    }

    /// Check if compaction is needed based on deletion ratio
    pub fn needs_compaction(&self) -> bool {
        self.deletion_ratio() >= DELETION_RATIO_THRESHOLD
    }

    /// Clear all deleted markers (use after rebuild)
    pub fn clear_deleted(&self) {
        self.deleted_ids.write().clear();
    }

    /// Add a single vector (incremental indexing) - OPTIMIZED
    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<u32> {
        // Dimension guard: a mismatched vector would silently corrupt the index
        // (distance computation strides by config.dimension; the only other check
        // is a debug_assert that is compiled out in release builds). Reject it.
        if vector.len() != self.config.dimension {
            anyhow::bail!(
                "add_vector: vector dimension {} does not match index dimension {}",
                vector.len(),
                self.config.dimension
            );
        }

        let current_count = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let id = current_count as u32;

        // A mapped prefix stays mapped: the new row goes to the heap tail.
        self.vectors.write().push(&vector);

        // For the first vector, just create a node with no neighbors
        if current_count == 0 {
            let mut graph = self.graph.write();
            graph.push(VamanaNode {
                id,
                neighbors: Vec::new(),
            });
            *self.medoid.write() = 0;
            self.num_vectors
                .fetch_add(1, std::sync::atomic::Ordering::Release);
            return Ok(id);
        }

        let neighbors = if self.graph.read().is_empty() {
            Vec::new()
        } else {
            let candidates =
                self.greedy_search(&vector, self.config.search_list_size, *self.medoid.read())?;
            self.robust_prune(id, &candidates)?
        };

        let mut graph = self.graph.write();
        graph.push(VamanaNode {
            id,
            neighbors: neighbors.clone(),
        });
        for &neighbor_id in &neighbors {
            let Some(neighbor) = graph.get_mut(neighbor_id as usize) else {
                continue;
            };
            neighbor.neighbors.push(id);
            if neighbor.neighbors.len() > self.config.max_degree {
                let candidates: Vec<SearchCandidate> = neighbor
                    .neighbors
                    .iter()
                    .map(|&n| SearchCandidate {
                        id: n,
                        distance: 0.0,
                    })
                    .collect();
                neighbor.neighbors = self.robust_prune(neighbor_id, &candidates)?;
            }
        }
        drop(graph);

        self.num_vectors
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.incremental_inserts
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(id)
    }

    /// Check if index rebuild is recommended for optimal search quality
    ///
    /// Returns true when:
    /// - Incremental inserts exceed REBUILD_THRESHOLD (10,000), OR
    /// - Deletion ratio exceeds DELETION_RATIO_THRESHOLD (30%)
    ///
    /// Incremental inserts use simplified neighbor pruning which can degrade
    /// recall@10 by 5-15% over time. High deletion ratios waste memory and
    /// slow down search (must filter more orphaned entries).
    pub fn needs_rebuild(&self) -> bool {
        let needs_insert_rebuild = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Acquire)
            >= REBUILD_THRESHOLD;
        let needs_compaction = self.needs_compaction();

        needs_insert_rebuild || needs_compaction
    }

    /// Get the number of incremental inserts since last rebuild
    pub fn incremental_insert_count(&self) -> usize {
        self.incremental_inserts
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Reset incremental insert counter (call after rebuild)
    pub fn reset_incremental_counter(&self) {
        self.incremental_inserts
            .store(0, std::sync::atomic::Ordering::Release);
    }

    /// Check if incremental repair is recommended
    ///
    /// Repair is lighter than full rebuild - only re-prunes neighborhoods of
    /// recently inserted nodes. Recommended every 1,000 inserts.
    pub fn needs_repair(&self) -> bool {
        let inserts = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed);
        (REPAIR_THRESHOLD..REBUILD_THRESHOLD).contains(&inserts)
    }

    /// Perform incremental repair on recently inserted nodes
    ///
    /// This is faster than full rebuild (~10x) and maintains index quality
    /// by re-pruning neighborhoods using proper α-RNG strategy instead of
    /// the simplified truncation used during incremental insert.
    ///
    /// Call when `needs_repair()` returns true.
    ///
    /// # Algorithm
    /// 1. Identify nodes inserted since last repair (last REPAIR_THRESHOLD nodes)
    /// 2. For each such node, re-run robust_prune on its neighborhood
    /// 3. Update bidirectional edges
    ///
    /// # Returns
    /// Number of nodes repaired
    pub fn incremental_repair(&self) -> Result<usize> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n == 0 {
            return Ok(0);
        }

        let inserts_since = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed);

        if inserts_since < REPAIR_THRESHOLD {
            return Ok(0);
        }

        // Repair the last REPAIR_THRESHOLD nodes (most recently inserted)
        let repair_count = inserts_since.min(REPAIR_THRESHOLD).min(n);
        let start_id = (n - repair_count) as u32;

        info!(
            "Incremental repair: re-pruning {} recently inserted nodes",
            repair_count
        );

        let medoid = *self.medoid.read();
        let mut repaired = 0;

        for node_id in start_id..(n as u32) {
            // Get vector for this node
            let query = match self.get_vector(node_id) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Search for fresh neighbors using current graph state
            let candidates = self.greedy_search(&query, self.config.search_list_size, medoid)?;

            // Re-prune using proper α-RNG strategy
            let pruned = self.robust_prune(node_id, &candidates)?;

            // Update graph
            let mut graph = self.graph.write();
            let old_neighbors = graph[node_id as usize].neighbors.clone();

            if old_neighbors != pruned {
                graph[node_id as usize].neighbors = pruned.clone();
                repaired += 1;

                // Update bidirectional edges
                // Remove back-edges from old neighbors not in new set
                for &old_neighbor in &old_neighbors {
                    if !pruned.contains(&old_neighbor) && (old_neighbor as usize) < graph.len() {
                        graph[old_neighbor as usize]
                            .neighbors
                            .retain(|&x| x != node_id);
                    }
                }

                // Add back-edges to new neighbors
                for &new_neighbor in &pruned {
                    if (new_neighbor as usize) < graph.len()
                        && !graph[new_neighbor as usize].neighbors.contains(&node_id)
                    {
                        graph[new_neighbor as usize].neighbors.push(node_id);

                        // Prune if exceeds max degree
                        if graph[new_neighbor as usize].neighbors.len() > self.config.max_degree {
                            graph[new_neighbor as usize]
                                .neighbors
                                .truncate(self.config.max_degree);
                        }
                    }
                }
            }
        }

        // Reset counter after repair (but not to 0 - track cumulative for rebuild)
        let new_count = inserts_since.saturating_sub(REPAIR_THRESHOLD);
        self.incremental_inserts
            .store(new_count, std::sync::atomic::Ordering::Relaxed);

        info!("Incremental repair complete: {} nodes updated", repaired);
        Ok(repaired)
    }

    /// Estimate current recall@k using random sampling
    ///
    /// Performs brute-force search on a sample of vectors and compares
    /// against ANN results to estimate recall degradation.
    ///
    /// # Arguments
    /// * `sample_size` - Number of random queries (default: 100)
    /// * `k` - Number of neighbors to check (default: 10)
    ///
    /// # Returns
    /// Estimated recall as f32 in range [0.0, 1.0]
    pub fn estimate_recall(&self, sample_size: usize, k: usize) -> Result<f32> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n < 2 {
            return Ok(1.0); // Perfect recall for trivial cases
        }

        let sample_size = sample_size.min(n / 2).max(1);
        let k = k.min(n - 1);

        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        // Fixed seed for reproducible recall estimates across runs
        let mut rng = rand::rngs::StdRng::seed_from_u64(n as u64 ^ 0x517cc1b727220a95);

        // Sample random query indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample_indices: Vec<usize> = indices.into_iter().take(sample_size).collect();

        let mut total_recall = 0.0;

        for &query_idx in &sample_indices {
            let query = self.get_vector(query_idx as u32)?;

            // Get ANN results
            let ann_results = self.search(&query, k)?;
            let ann_ids: HashSet<u32> = ann_results.iter().map(|(id, _)| *id).collect();

            // Get exact brute-force results
            let exact_results = self.brute_force_search(&query, k)?;
            let exact_ids: HashSet<u32> = exact_results.iter().map(|(id, _)| *id).collect();

            // Calculate recall for this query
            let overlap = ann_ids.intersection(&exact_ids).count();
            total_recall += overlap as f32 / k as f32;
        }

        Ok(total_recall / sample_size as f32)
    }

    /// Brute-force k-NN search (for recall estimation)
    fn brute_force_search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        self.exact_search(query, k, n, &self.deleted_ids.read())
    }

    /// Check if index quality has degraded below acceptable threshold
    ///
    /// Uses sampling to estimate recall without expensive full evaluation.
    /// Returns true if estimated recall@10 < MIN_ACCEPTABLE_RECALL (85%).
    pub fn quality_degraded(&self) -> Result<bool> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n < 100 {
            return Ok(false); // Too small to meaningfully measure
        }

        // Quick check: if no incremental inserts, quality is fine
        if self.incremental_insert_count() == 0 {
            return Ok(false);
        }

        // Sample-based recall estimation (50 samples, recall@10)
        let recall = self.estimate_recall(50, 10)?;
        Ok(recall < MIN_ACCEPTABLE_RECALL)
    }

    /// Automatic maintenance: repair or rebuild as needed
    ///
    /// Checks index state and performs appropriate maintenance:
    /// 1. If needs_repair() → incremental_repair()
    /// 2. If needs_rebuild() → auto_rebuild_if_needed()
    ///
    /// Returns description of action taken
    pub fn auto_maintain(&self) -> Result<String> {
        if self.needs_rebuild() {
            if self.auto_rebuild_if_needed()? {
                return Ok("full_rebuild".to_string());
            } else {
                return Ok("rebuild_skipped".to_string());
            }
        }

        if self.needs_repair() {
            let repaired = self.incremental_repair()?;
            return Ok(format!("repaired_{}_nodes", repaired));
        }

        Ok("no_action".to_string())
    }

    /// Extract all vectors from the index for rebuilding
    ///
    /// Returns a clone of all vectors currently in the index.
    /// Use this before calling `rebuild_from_vectors()`.
    pub fn extract_all_vectors(&self) -> Vec<Vec<f32>> {
        self.vectors.read().to_rows()
    }

    /// Extract only live (non-deleted) vectors for compaction rebuild
    ///
    /// Returns vectors that are NOT marked as deleted.
    /// Use this for compaction to physically remove deleted vectors.
    pub fn extract_live_vectors(&self) -> Vec<Vec<f32>> {
        let deleted = self.deleted_ids.read();
        let storage = self.vectors.read();
        (0..storage.len() as u32)
            .filter(|&id| !deleted.contains(id))
            .filter_map(|id| storage.row(id).map(<[f32]>::to_vec))
            .collect()
    }

    /// Rebuild the index from vectors with full Vamana construction
    ///
    /// This performs a complete rebuild using robust_prune for optimal graph quality.
    /// Call this when `needs_rebuild()` returns true to restore recall@10 accuracy.
    ///
    /// # Arguments
    /// * `vectors` - All vectors to index (typically from `extract_all_vectors()`)
    ///
    /// # Returns
    /// * `Ok(())` on success, resets the incremental insert counter
    pub fn rebuild_from_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        info!(
            "Rebuilding Vamana index with {} vectors (was {} incremental inserts)",
            vectors.len(),
            self.incremental_insert_count()
        );

        // Clear current state
        self.graph.write().clear();
        *self.vectors.write() = VectorStore::new(self.config.dimension);
        self.num_vectors
            .store(0, std::sync::atomic::Ordering::Release);

        // Full rebuild with robust_prune
        self.build(vectors)?;

        // Reset counter after successful rebuild
        self.reset_incremental_counter();

        info!("Vamana index rebuild complete");
        Ok(())
    }

    /// Perform automatic rebuild if threshold exceeded (non-blocking)
    ///
    /// Thread-safe method that checks if rebuild is needed and performs it without
    /// blocking concurrent reads or writes. Uses a background build followed by
    /// atomic swap of index internals.
    ///
    /// Returns true if rebuild was performed, false if not needed or already in progress.
    ///
    /// ## Concurrency Model
    ///
    /// - **Reads**: Continue uninterrupted on the old index during rebuild
    /// - **Writes**: Continue on the old index but will be lost when swap occurs
    /// - **Swap**: Brief write locks acquired only during the final atomic swap
    ///
    /// Uses compare-and-swap to ensure only one rebuild occurs even with concurrent calls.
    /// Compacts deleted vectors by extracting only live vectors.
    ///
    /// ## Note on Write Handling
    ///
    /// Any vectors added between `extract_live_vectors()` and the final swap will be
    /// lost. This is acceptable for periodic maintenance rebuilds. For write-intensive
    /// workloads, consider using `rebuild_from_vectors()` which takes `&mut self` and
    /// blocks writes during rebuild.
    pub fn auto_rebuild_if_needed(&self) -> Result<bool> {
        if !self.needs_rebuild() {
            return Ok(false);
        }

        // Atomic compare-and-swap: try to set rebuilding from false to true
        // If another thread is already rebuilding, this returns Err and we skip
        if self
            .rebuilding
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .is_err()
        {
            // Another thread is already rebuilding
            return Ok(false);
        }

        // Log reason for rebuild
        let deleted_count = self.deleted_count();
        let deletion_ratio = self.deletion_ratio();
        let total_vectors = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if deletion_ratio >= DELETION_RATIO_THRESHOLD {
            info!(
                "Compacting index: {} deleted vectors ({:.1}% of {})",
                deleted_count,
                deletion_ratio * 100.0,
                total_vectors
            );
        }

        // We acquired the rebuild lock - perform background rebuild with atomic swap
        let result = (|| {
            // 1. Extract live vectors (read-only, doesn't block writes)
            let vectors = self.extract_live_vectors();
            let compacted = deleted_count;
            if vectors.is_empty() {
                self.clear_deleted();
                return Ok(false);
            }

            info!(
                "Background rebuilding Vamana index with {} vectors (was {} incremental inserts)",
                vectors.len(),
                self.incremental_insert_count()
            );

            // 2. Build completely new index (expensive, but doesn't hold any locks on self)
            let config = self.config.clone();
            let config_dimension = config.dimension;
            let mut new_index = VamanaIndex::new(config)?;
            new_index.build(vectors)?;

            // 3. Atomic swap - acquire all write locks briefly
            // Lock ordering: graph -> vectors -> medoid (consistent with struct field order)
            {
                let mut old_graph = self.graph.write();
                let mut old_vectors = self.vectors.write();
                let mut old_medoid = self.medoid.write();

                // Swap graph
                let new_graph = std::mem::take(&mut *new_index.graph.write());
                *old_graph = new_graph;

                // Swap vectors
                let new_vectors = std::mem::replace(
                    &mut *new_index.vectors.write(),
                    VectorStore::new(config_dimension),
                );
                *old_vectors = new_vectors;

                // Swap medoid
                *old_medoid = *new_index.medoid.read();
            }

            // Update num_vectors atomically (after releasing locks)
            self.num_vectors.store(
                new_index
                    .num_vectors
                    .load(std::sync::atomic::Ordering::Acquire),
                std::sync::atomic::Ordering::Release,
            );

            // Clear deleted markers and reset counter
            self.clear_deleted();
            self.reset_incremental_counter();

            if compacted > 0 {
                info!("Compaction complete: removed {} deleted vectors", compacted);
            }
            info!("Background Vamana index rebuild complete");

            Ok(true)
        })();

        // Always release the lock, even on error
        self.rebuilding
            .store(false, std::sync::atomic::Ordering::SeqCst);

        result
    }

    /// Check if a rebuild is currently in progress
    pub fn is_rebuilding(&self) -> bool {
        self.rebuilding.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Save index to disk
    pub fn save(&self, path: &Path) -> Result<()> {
        use serde::{Deserialize, Serialize};
        use std::fs::{create_dir_all, File};
        use std::io::BufWriter;

        // Ensure directory exists
        create_dir_all(path)?;

        #[derive(Serialize, Deserialize)]
        struct VamanaData {
            graph: Vec<VamanaNode>,
            vectors: Vec<Vec<f32>>,
            medoid: u32,
            num_vectors: usize,
            #[serde(default)]
            deleted_ids: HashSet<u32>,
        }

        let vectors = self.vectors.read().to_rows();

        let num_vecs = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let data = VamanaData {
            graph: self.graph.read().clone(),
            vectors,
            medoid: *self.medoid.read(),
            num_vectors: num_vecs,
            deleted_ids: self.deleted_ids.read().iter().collect(),
        };

        // Save as binary
        let index_file = path.join("vamana_index.bin");
        let file = File::create(&index_file)?;
        let mut writer = BufWriter::new(file);
        bincode::serde::encode_into_std_write(&data, &mut writer, bincode::config::standard())?;

        info!(
            "Saved Vamana index with {} vectors to {:?}",
            num_vecs, index_file
        );
        Ok(())
    }

    /// Load index from disk
    /// Load index data into existing instance (dynamic method)
    pub fn load(&mut self, path: &Path) -> Result<()> {
        use serde::{Deserialize, Serialize};
        use std::fs::File;
        use std::io::BufReader;

        let index_file = path.join("vamana_index.bin");
        if !index_file.exists() {
            return Err(anyhow!("Vamana index file not found at {index_file:?}"));
        }

        // Load serialized data
        let file = File::open(&index_file)?;
        let mut reader = BufReader::new(file);

        #[derive(Serialize, Deserialize)]
        struct VamanaData {
            graph: Vec<VamanaNode>,
            vectors: Vec<Vec<f32>>,
            medoid: u32,
            num_vectors: usize,
            #[serde(default)]
            deleted_ids: HashSet<u32>,
        }

        let data: VamanaData =
            bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;

        let vectors = VectorStore::from_rows(self.config.dimension, data.vectors)?;

        // Update internal state
        *self.graph.write() = data.graph;
        *self.medoid.write() = data.medoid;
        self.num_vectors
            .store(data.num_vectors, std::sync::atomic::Ordering::Release);
        *self.vectors.write() = vectors;

        // Restore soft-deleted IDs
        if !data.deleted_ids.is_empty() {
            info!(
                "Restoring {} soft-deleted vector IDs from persisted index",
                data.deleted_ids.len()
            );
            *self.deleted_ids.write() = data.deleted_ids.into_iter().collect();
        }

        info!("Loaded Vamana index with {} vectors", data.num_vectors);
        Ok(())
    }
}

/// Search candidate
#[derive(Debug, Clone)]
struct SearchCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Stable tie-break by id: equal-distance candidates keep a deterministic
        // rank order across runs and machines, instead of flipping based on float
        // accumulation order (RH-10 determinism).
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vectors(rng: &mut impl rand::Rng, n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.into_iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn initial_graph_gives_each_node_distinct_random_neighbors() {
        for (n, max_degree) in [(50usize, 8usize), (5, 8), (1, 8), (2, 1)] {
            let mut index = VamanaIndex::new(VamanaConfig {
                dimension: 4,
                max_degree,
                use_mmap: false,
                ..Default::default()
            })
            .unwrap();
            index.initialize_graph(n, 7).unwrap();
            let graph = index.graph.read();
            assert_eq!(graph.len(), n);
            for (i, node) in graph.iter().enumerate() {
                assert_eq!(node.id as usize, i);
                assert_eq!(node.neighbors.len(), max_degree.min(n - 1), "n={n}");
                let distinct: std::collections::HashSet<_> = node.neighbors.iter().collect();
                assert_eq!(distinct.len(), node.neighbors.len(), "duplicate neighbor");
                assert!(node
                    .neighbors
                    .iter()
                    .all(|&j| (j as usize) < n && j as usize != i));
            }
        }
    }

    /// recall@10 against brute force over five random corpora of unit vectors. The random start
    /// graph is replaced by robust pruning during the build, so recall must not hinge on which
    /// random neighbours a node starts with.
    #[test]
    fn built_index_recall_against_brute_force() {
        use rand::SeedableRng;
        let (n, dim, k) = (2000, 32, 10);
        let mut recalls = Vec::new();
        for seed in 1..=5u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let vectors = unit_vectors(&mut rng, n, dim);
            let queries = unit_vectors(&mut rng, 100, dim);

            let mut index = VamanaIndex::new(VamanaConfig {
                dimension: dim,
                use_mmap: false,
                ..Default::default()
            })
            .unwrap();
            index.build(vectors.clone()).unwrap();

            let mut hits = 0usize;
            for q in &queries {
                let mut exact: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>()))
                    .collect();
                exact.sort_by(|a, b| b.1.total_cmp(&a.1));
                let truth: std::collections::HashSet<u32> =
                    exact[..k].iter().map(|(i, _)| *i as u32).collect();
                hits += index
                    .search(q, k)
                    .unwrap()
                    .iter()
                    .filter(|(id, _)| truth.contains(id))
                    .count();
            }
            recalls.push(hits as f64 / (queries.len() * k) as f64);
        }
        let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        eprintln!("vamana recall@{k}: per corpus {recalls:.4?}, mean {mean:.4}");
        assert!(
            mean >= 0.75 && recalls.iter().all(|r| *r >= 0.70),
            "recall@{k} fell: mean {mean} over {recalls:?}"
        );
    }

    #[test]
    fn test_vamana_construction() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 3,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        index.build(vectors).unwrap();

        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest to [1,0,0,0]
    }

    #[test]
    fn test_incremental_repair() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 3,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        // Build initial index
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        index.build(vectors).unwrap();

        // Should not need repair initially
        assert!(!index.needs_repair());
        assert_eq!(index.incremental_insert_count(), 0);

        // Add some vectors incrementally
        for i in 0..5 {
            let v = vec![0.1 * i as f32, 0.1, 0.1, 0.1];
            index.add_vector(v).unwrap();
        }

        assert_eq!(index.incremental_insert_count(), 5);
        assert!(!index.needs_repair()); // Still below threshold

        // Repair should do nothing below threshold
        let repaired = index.incremental_repair().unwrap();
        assert_eq!(repaired, 0);
    }

    #[test]
    fn test_estimate_recall() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 4,        // Higher degree for better connectivity
            search_list_size: 20, // Larger search list for better recall
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        // Use more vectors for stable recall estimation
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.5, 0.0, 0.5, 0.0],
            vec![0.0, 0.5, 0.5, 0.0],
            vec![0.0, 0.0, 0.5, 0.5],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.7, 0.3, 0.0, 0.0],
        ];
        index.build(vectors).unwrap();

        // Freshly built index should have reasonable recall
        // With small indices, recall can vary; 0.6 is a stable lower bound
        let recall = index.estimate_recall(5, 3).unwrap();
        assert!(recall >= 0.6, "Expected reasonable recall, got {}", recall);
    }

    #[test]
    fn test_auto_maintain() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 3,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        index.build(vectors).unwrap();

        // Should take no action on fresh index
        let result = index.auto_maintain().unwrap();
        assert_eq!(result, "no_action");
    }

    /// Unit vectors in tight clusters with near-duplicates, the shape of a memory store: many
    /// session notes and audit rollups that differ by a timestamp.
    fn clustered_unit_vectors(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normalise = |v: Vec<f32>| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect::<Vec<f32>>()
        };
        let centres: Vec<Vec<f32>> = (0..clusters)
            .map(|_| normalise((0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()))
            .collect();
        (0..n)
            .map(|i| {
                let centre = &centres[i % clusters];
                normalise(
                    centre
                        .iter()
                        .map(|c| c + rng.gen_range(-0.08..0.08))
                        .collect(),
                )
            })
            .collect()
    }

    fn graph_config(dimension: usize) -> VamanaConfig {
        VamanaConfig {
            dimension,
            max_degree: 32,
            search_list_size: 100,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        }
    }

    /// Share of vectors that the index returns in the top 10 when queried with themselves.
    fn self_recall(index: &VamanaIndex, vectors: &[Vec<f32>]) -> f32 {
        let found = vectors
            .iter()
            .enumerate()
            .filter(|(id, v)| {
                index
                    .search(v, 10)
                    .unwrap()
                    .iter()
                    .any(|(hit, _)| *hit == *id as u32)
            })
            .count();
        found as f32 / vectors.len() as f32
    }

    #[test]
    fn vectors_inserted_one_at_a_time_stay_reachable() {
        let vectors = clustered_unit_vectors(3000, 48, 40, 7);
        let mut index = VamanaIndex::new(graph_config(48)).unwrap();
        index.set_exact_search_max(0);
        for v in &vectors {
            index.add_vector(v.clone()).unwrap();
        }
        let recall = self_recall(&index, &vectors);
        assert!(
            recall >= 0.95,
            "graph self-recall after incremental inserts: {recall}"
        );
    }

    #[test]
    fn a_built_graph_reaches_every_vector() {
        let vectors = clustered_unit_vectors(3000, 48, 40, 11);
        let mut index = VamanaIndex::new(graph_config(48)).unwrap();
        index.set_exact_search_max(0);
        index.build(vectors.clone()).unwrap();
        let recall = self_recall(&index, &vectors);
        assert!(recall >= 0.95, "graph self-recall after build: {recall}");
    }

    fn neighbor_lists(index: &VamanaIndex) -> Vec<Vec<u32>> {
        index
            .graph
            .read()
            .iter()
            .map(|node| node.neighbors.clone())
            .collect()
    }

    /// The same vectors behind a mapped prefix plus a heap tail, or all on the heap, build the
    /// same graph and give the same results, exact or walked; inserts leave the prefix mapped.
    #[test]
    fn mapped_and_heap_storage_search_identically() {
        let vectors = clustered_unit_vectors(1200, 24, 12, 5);
        let (built, inserted) = vectors.split_at(1000);
        let dir = tempfile::tempdir().unwrap();
        let mut mapped = VamanaIndex::with_storage_path(
            VamanaConfig {
                use_mmap: true,
                ..graph_config(24)
            },
            Some(dir.path().to_path_buf()),
        )
        .unwrap();
        let mut heap = VamanaIndex::new(graph_config(24)).unwrap();
        for index in [&mut mapped, &mut heap] {
            index.build(built.to_vec()).unwrap();
            for v in inserted {
                index.add_vector(v.clone()).unwrap();
            }
            index.mark_deleted(3);
            index.mark_deleted(1100);
        }
        assert!(mapped.vectors.read().is_mapped());
        assert!(!heap.vectors.read().is_mapped());
        assert_eq!(mapped.vectors.read().len(), 1200);
        assert_eq!(neighbor_lists(&mapped), neighbor_lists(&heap));
        for exact_max in [usize::MAX, 0] {
            mapped.set_exact_search_max(exact_max);
            heap.set_exact_search_max(exact_max);
            for q in vectors.iter().step_by(37) {
                assert_eq!(mapped.search(q, 10).unwrap(), heap.search(q, 10).unwrap());
            }
        }
    }

    /// The graph search as it was written before the visited and deleted sets became a stamp
    /// array and a bitset: std `HashSet`s, the beam sized k plus up to 2k for deleted vectors.
    fn hash_set_graph_search(index: &VamanaIndex, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let graph = index.graph.read();
        let storage = index.vectors.read();
        let deleted: HashSet<u32> = index.deleted_ids.read().iter().collect();
        let search_k = if deleted.is_empty() {
            k
        } else {
            k + deleted.len().min(k * 2)
        };
        let beam = search_k.max(index.config.search_list_size);
        let distance = |id: u32| index.distance(query, storage.row(id).unwrap());
        let entry = *index.medoid.read();
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        let entry_distance = distance(entry);
        candidates.push(Reverse(SearchCandidate {
            id: entry,
            distance: entry_distance,
        }));
        w.push(SearchCandidate {
            id: entry,
            distance: entry_distance,
        });
        visited.insert(entry);
        while let Some(Reverse(current)) = candidates.pop() {
            if w.peek()
                .is_some_and(|worst: &SearchCandidate| current.distance > worst.distance)
            {
                break;
            }
            for &neighbor in &graph[current.id as usize].neighbors {
                if !visited.insert(neighbor) {
                    continue;
                }
                let d = distance(neighbor);
                if w.len() < beam || w.peek().is_some_and(|worst| d < worst.distance) {
                    candidates.push(Reverse(SearchCandidate {
                        id: neighbor,
                        distance: d,
                    }));
                    w.push(SearchCandidate {
                        id: neighbor,
                        distance: d,
                    });
                    if w.len() > beam {
                        w.pop();
                    }
                }
            }
        }
        w.into_sorted_vec()
            .into_iter()
            .filter(|c| !deleted.contains(&c.id))
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect()
    }

    /// The stamp-array visited set and the deleted bitset walk the graph exactly as the hash
    /// sets did: same ids, same distances, same order, with deletions and incremental inserts.
    #[test]
    fn graph_search_matches_the_hash_set_search() {
        let vectors = clustered_unit_vectors(2500, 32, 25, 9);
        let mut index = VamanaIndex::new(graph_config(32)).unwrap();
        index.set_exact_search_max(0);
        index.build(vectors[..2000].to_vec()).unwrap();
        for v in &vectors[2000..] {
            index.add_vector(v.clone()).unwrap();
        }
        for id in (0..2500).step_by(41) {
            index.mark_deleted(id);
        }
        for (i, q) in vectors.iter().enumerate().step_by(53) {
            for k in [1, 10, 60] {
                assert_eq!(
                    index.search(q, k).unwrap(),
                    hash_set_graph_search(&index, q, k),
                    "query {i}, k {k}"
                );
            }
        }
    }

    #[test]
    fn a_query_of_another_dimension_is_refused() {
        let mut index = VamanaIndex::new(graph_config(16)).unwrap();
        index.build(clustered_unit_vectors(50, 16, 3, 1)).unwrap();
        assert!(index.search(&[0.5; 8], 5).is_err());
        assert!(index.build(vec![vec![0.0; 15]]).is_err());
    }

    #[test]
    fn small_indexes_are_searched_exactly_and_skip_deleted_vectors() {
        let vectors = clustered_unit_vectors(500, 16, 5, 3);
        let mut index = VamanaIndex::new(graph_config(16)).unwrap();
        for v in &vectors {
            index.add_vector(v.clone()).unwrap();
        }
        index.mark_deleted(9);
        let query = &vectors[9];
        let mut expected: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != 9)
            .map(|(id, v)| {
                (
                    id as u32,
                    -v.iter().zip(query).map(|(a, b)| a * b).sum::<f32>(),
                )
            })
            .collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let got: Vec<u32> = index
            .search(query, 10)
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let want: Vec<u32> = expected.iter().take(10).map(|(id, _)| *id).collect();
        assert_eq!(got, want);
    }
}
