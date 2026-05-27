//! HNSW — Hierarchical Navigable Small World graph index.
//!
//! Third vector backend alongside Vamana (graph) and SPANN (IVF+PQ). HNSW
//! is the de-facto industry standard for in-memory ANN (default in Qdrant,
//! Weaviate, Milvus, pgvector, FAISS HNSW mode).
//!
//! This is a minimal but real layered implementation:
//! - Each node assigned a max layer by geometric distribution (mult = 1/ln(M))
//! - Insert greedy-walks down through upper layers, then ef-construction beam
//!   search at each layer ≤ node-level, connecting to the M closest
//! - Search greedy-walks down through layers, ef-search beam at layer 0
//! - Neighbor pruning at each layer: keep the M closest (or M_max_0 at L0)
//!
//! Persistence: bincode-serialized snapshot of the full graph. Not as
//! compact as Vamana's custom on-disk format, but adequate for a third
//! backend. Use Vamana if you need disk-friendly persistence.

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;
use std::sync::Arc;

use super::distance_inline::{
    cosine_similarity_inline, euclidean_squared_inline,
};
use super::vamana::DistanceMetric;

/// Default max neighbors per node per layer
const DEFAULT_M: usize = 16;
/// Default max neighbors at layer 0 (the densest layer)
const DEFAULT_M_MAX_0: usize = 32;
/// Beam width during construction
const DEFAULT_EF_CONSTRUCTION: usize = 200;
/// Beam width during search (caller-tunable)
const DEFAULT_EF_SEARCH: usize = 50;

/// HNSW configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub dimension: usize,
    /// Max neighbors per node at layers > 0
    pub m: usize,
    /// Max neighbors at layer 0
    pub m_max_0: usize,
    /// Search-list size during construction (higher = better recall, slower build)
    pub ef_construction: usize,
    /// Search-list size during queries (higher = better recall, slower query)
    pub ef_search: usize,
    pub distance_metric: DistanceMetric,
}

/// On-disk form of `HnswConfig` — `DistanceMetric` doesn't implement
/// `Serialize`/`Deserialize`, so we map it through a `u8` discriminator.
#[derive(Serialize, Deserialize)]
struct HnswConfigPersisted {
    dimension: usize,
    m: usize,
    m_max_0: usize,
    ef_construction: usize,
    ef_search: usize,
    distance_metric_tag: u8,
}

impl HnswConfigPersisted {
    fn from_runtime(c: &HnswConfig) -> Self {
        let tag = match c.distance_metric {
            DistanceMetric::NormalizedDotProduct => 0,
            DistanceMetric::Euclidean => 1,
            DistanceMetric::Cosine => 2,
        };
        Self {
            dimension: c.dimension,
            m: c.m,
            m_max_0: c.m_max_0,
            ef_construction: c.ef_construction,
            ef_search: c.ef_search,
            distance_metric_tag: tag,
        }
    }

    fn to_runtime(&self) -> Result<HnswConfig> {
        let metric = match self.distance_metric_tag {
            0 => DistanceMetric::NormalizedDotProduct,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::Cosine,
            other => return Err(anyhow!("Unknown DistanceMetric tag {other}")),
        };
        Ok(HnswConfig {
            dimension: self.dimension,
            m: self.m,
            m_max_0: self.m_max_0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            distance_metric: metric,
        })
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            m: DEFAULT_M,
            m_max_0: DEFAULT_M_MAX_0,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            distance_metric: DistanceMetric::NormalizedDotProduct,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    vector: Vec<f32>,
    /// `neighbors[layer]` = neighbor ids at that layer. `neighbors.len()` is
    /// the node's top layer + 1.
    neighbors: Vec<Vec<u32>>,
}

#[derive(Default, Serialize, Deserialize)]
struct HnswInner {
    nodes: Vec<HnswNode>,
    entry_point: Option<u32>,
    /// Highest layer index currently in the graph
    max_level: i32,
}

/// HNSW vector index.
pub struct HnswIndex {
    config: HnswConfig,
    inner: Arc<RwLock<HnswInner>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Scored {
    id: u32,
    /// Distance: smaller = more similar (consistent with Vamana/SPANN)
    dist: f32,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            inner: Arc::new(RwLock::new(HnswInner::default())),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn distance_metric(&self) -> DistanceMetric {
        self.config.distance_metric
    }

    fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
        match metric {
            DistanceMetric::NormalizedDotProduct => 1.0 - cosine_similarity_inline(a, b),
            DistanceMetric::Euclidean => euclidean_squared_inline(a, b),
            DistanceMetric::Cosine => 1.0 - cosine_similarity_inline(a, b),
        }
    }

    fn sample_level(&self) -> i32 {
        let mult = 1.0 / (self.config.m as f32).ln();
        let r: f32 = rand::thread_rng().gen::<f32>().max(f32::EPSILON);
        ((-r.ln() * mult).floor() as i32).max(0)
    }

    /// Greedy single-step descent at `layer`: from `entry`, repeatedly hop to
    /// the closest neighbor at that layer until no neighbor is closer than
    /// the current node. Returns the closest node found.
    fn greedy_search_layer(
        inner: &HnswInner,
        query: &[f32],
        entry: u32,
        layer: i32,
        metric: DistanceMetric,
    ) -> u32 {
        let mut current = entry;
        let mut current_dist = Self::distance(metric, query, &inner.nodes[current as usize].vector);
        loop {
            let neighbors = inner.nodes[current as usize]
                .neighbors
                .get(layer as usize)
                .cloned()
                .unwrap_or_default();
            let mut best = current;
            let mut best_dist = current_dist;
            for nb in neighbors {
                let d = Self::distance(metric, query, &inner.nodes[nb as usize].vector);
                if d < best_dist {
                    best = nb;
                    best_dist = d;
                }
            }
            if best == current {
                return current;
            }
            current = best;
            current_dist = best_dist;
        }
    }

    /// Beam search at a single layer. Returns up to `ef` candidates sorted
    /// by ascending distance (closest first).
    fn search_layer_ef(
        inner: &HnswInner,
        query: &[f32],
        entry: u32,
        ef: usize,
        layer: i32,
        metric: DistanceMetric,
    ) -> Vec<Scored> {
        let entry_dist = Self::distance(metric, query, &inner.nodes[entry as usize].vector);
        let entry_scored = Scored {
            id: entry,
            dist: entry_dist,
        };

        // Candidate frontier: min-heap on distance
        let mut frontier: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
        frontier.push(std::cmp::Reverse(entry_scored));

        // Result set: max-heap of the best `ef` so far (largest distance at top)
        let mut result: BinaryHeap<Scored> = BinaryHeap::new();
        result.push(entry_scored);

        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(entry);

        while let Some(std::cmp::Reverse(cand)) = frontier.pop() {
            let worst_in_result = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if cand.dist > worst_in_result && result.len() >= ef {
                break;
            }
            let neighbors = inner.nodes[cand.id as usize]
                .neighbors
                .get(layer as usize)
                .cloned()
                .unwrap_or_default();
            for nb in neighbors {
                if !visited.insert(nb) {
                    continue;
                }
                let d = Self::distance(metric, query, &inner.nodes[nb as usize].vector);
                let worst = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
                if result.len() < ef || d < worst {
                    let s = Scored { id: nb, dist: d };
                    frontier.push(std::cmp::Reverse(s));
                    result.push(s);
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out: Vec<Scored> = result.into_sorted_vec();
        // into_sorted_vec gives ascending by Ord; our Ord is by dist asc — good.
        out.truncate(ef);
        out
    }

    /// Prune a neighbor list down to at most `m_max` by keeping the closest.
    fn prune_neighbors(
        inner: &HnswInner,
        node_id: u32,
        layer: usize,
        m_max: usize,
        metric: DistanceMetric,
        out: &mut Vec<u32>,
    ) {
        let node_vec = inner.nodes[node_id as usize].vector.clone();
        let neighbors = inner.nodes[node_id as usize]
            .neighbors
            .get(layer)
            .cloned()
            .unwrap_or_default();
        if neighbors.len() <= m_max {
            *out = neighbors;
            return;
        }
        let mut scored: Vec<Scored> = neighbors
            .into_iter()
            .map(|nb| Scored {
                id: nb,
                dist: Self::distance(metric, &node_vec, &inner.nodes[nb as usize].vector),
            })
            .collect();
        scored.sort();
        scored.truncate(m_max);
        *out = scored.into_iter().map(|s| s.id).collect();
    }

    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<u32> {
        if vector.len() != self.config.dimension {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.len()
            ));
        }
        let level = self.sample_level();
        let metric = self.config.distance_metric;
        let mut inner = self.inner.write();

        // First insertion: trivial seed
        let new_id = inner.nodes.len() as u32;
        if inner.nodes.is_empty() {
            inner.nodes.push(HnswNode {
                vector,
                neighbors: vec![Vec::new(); (level + 1) as usize],
            });
            inner.entry_point = Some(new_id);
            inner.max_level = level;
            return Ok(new_id);
        }

        let entry = inner.entry_point.expect("non-empty has entry");
        let max_level_before = inner.max_level;

        // Phase 1: greedy descend from top layer down to level+1
        let mut ep = entry;
        for layer in ((level + 1)..=max_level_before).rev() {
            ep = Self::greedy_search_layer(&inner, &vector, ep, layer, metric);
        }

        // Push the new node (so it can be referenced during phase 2)
        let new_node = HnswNode {
            vector: vector.clone(),
            neighbors: vec![Vec::new(); (level + 1) as usize],
        };
        inner.nodes.push(new_node);

        // Phase 2: ef-construction search at each layer ≤ min(level, max_level_before),
        // then connect bi-directionally.
        for layer in (0..=level.min(max_level_before)).rev() {
            let candidates =
                Self::search_layer_ef(&inner, &vector, ep, self.config.ef_construction, layer, metric);
            let m_for_layer = if layer == 0 {
                self.config.m_max_0
            } else {
                self.config.m
            };
            let selected: Vec<u32> = candidates
                .iter()
                .take(m_for_layer)
                .map(|s| s.id)
                .collect();

            inner.nodes[new_id as usize].neighbors[layer as usize] = selected.clone();

            // Bi-directional link + prune each neighbor
            for nb in &selected {
                let nb_idx = *nb as usize;
                if (inner.nodes[nb_idx].neighbors.len() as i32) <= layer {
                    continue;
                }
                inner.nodes[nb_idx].neighbors[layer as usize].push(new_id);
                let mut pruned = Vec::new();
                Self::prune_neighbors(&inner, *nb, layer as usize, m_for_layer, metric, &mut pruned);
                inner.nodes[nb_idx].neighbors[layer as usize] = pruned;
            }

            // Seed the next (lower) layer's search from the closest candidate.
            if let Some(first) = selected.first() {
                ep = *first;
            }
        }

        // If new node's level exceeds previous max, promote it to entry point
        if level > max_level_before {
            inner.entry_point = Some(new_id);
            inner.max_level = level;
        }

        Ok(new_id)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        if query.len() != self.config.dimension {
            return Err(anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.config.dimension,
                query.len()
            ));
        }
        let inner = self.inner.read();
        if inner.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let metric = self.config.distance_metric;
        let entry = inner.entry_point.expect("non-empty has entry");
        let max_level = inner.max_level;

        // Greedy descend through upper layers
        let mut ep = entry;
        for layer in (1..=max_level).rev() {
            ep = Self::greedy_search_layer(&inner, query, ep, layer, metric);
        }

        // ef-search beam at layer 0
        let ef = self.config.ef_search.max(k);
        let candidates = Self::search_layer_ef(&inner, query, ep, ef, 0, metric);

        let out: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(k)
            .map(|s| (s.id, s.dist))
            .collect();
        Ok(out)
    }

    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        for v in vectors {
            self.add_vector(v)?;
        }
        Ok(())
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let inner = self.inner.read();
        let snapshot = HnswSnapshot {
            config: HnswConfigPersisted::from_runtime(&self.config),
            inner: &inner,
        };
        let bytes = bincode::serde::encode_to_vec(&snapshot, bincode::config::standard())?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let (snapshot, _): (HnswSnapshotOwned, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        Ok(Self {
            config: snapshot.config.to_runtime()?,
            inner: Arc::new(RwLock::new(snapshot.inner)),
        })
    }
}

#[derive(Serialize)]
struct HnswSnapshot<'a> {
    config: HnswConfigPersisted,
    inner: &'a HnswInner,
}

#[derive(Deserialize)]
struct HnswSnapshotOwned {
    config: HnswConfigPersisted,
    inner: HnswInner,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, d: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect()
    }

    #[test]
    fn test_hnsw_build_and_search() {
        let cfg = HnswConfig {
            dimension: 16,
            m: 8,
            m_max_0: 16,
            ef_construction: 50,
            ef_search: 32,
            distance_metric: DistanceMetric::NormalizedDotProduct,
        };
        let mut idx = HnswIndex::new(cfg);
        let vecs = random_vectors(100, 16);
        for v in &vecs {
            idx.add_vector(v.clone()).unwrap();
        }
        assert_eq!(idx.len(), 100);

        // Query with the first vector — it should be returned as the closest match
        let query = vecs[0].clone();
        let results = idx.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0, "self-query returns id=0 as nearest");
    }

    #[test]
    fn test_hnsw_save_load_roundtrip() {
        use tempfile::NamedTempFile;
        let cfg = HnswConfig {
            dimension: 8,
            m: 4,
            m_max_0: 8,
            ef_construction: 20,
            ef_search: 16,
            distance_metric: DistanceMetric::NormalizedDotProduct,
        };
        let mut idx = HnswIndex::new(cfg);
        let vecs = random_vectors(30, 8);
        for v in &vecs {
            idx.add_vector(v.clone()).unwrap();
        }

        let file = NamedTempFile::new().unwrap();
        idx.save_to_file(file.path()).unwrap();
        let loaded = HnswIndex::load_from_file(file.path()).unwrap();
        assert_eq!(loaded.len(), 30);

        // Same query should give same results
        let q = vecs[5].clone();
        let r1 = idx.search(&q, 3).unwrap();
        let r2 = loaded.search(&q, 3).unwrap();
        assert_eq!(r1.len(), r2.len());
        assert_eq!(r1[0].0, r2[0].0, "top match identical after reload");
    }

    #[test]
    fn test_hnsw_empty_search() {
        let cfg = HnswConfig {
            dimension: 4,
            ..Default::default()
        };
        let idx = HnswIndex::new(cfg);
        let q = vec![0.0, 0.0, 0.0, 0.0];
        let results = idx.search(&q, 5).unwrap();
        assert!(results.is_empty());
    }
}
