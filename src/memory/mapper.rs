//! Mapper Algorithm for Topological Data Analysis
//!
//! The Mapper algorithm produces a simplified, visualizable graph that captures
//! the topological shape of a point cloud. It works by:
//!
//! 1. Applying a filter function f: X → R to the data
//! 2. Covering the range f(X) with overlapping intervals
//! 3. For each interval, pulling back to the data and clustering
//! 4. Building the nerve of the resulting cover
//!
//! The output graph reveals branches, loops, flares, and connected components
//! in the data's topology — structure invisible to standard clustering.
//!
//! For knowledge graphs, the filter function can be any of the 10 dimensions
//! (isolation, density, novelty, etc.), and different filters reveal different
//! aspects of the knowledge topology.
//!
//! Reference: Singh, Mémoli, Carlsson (2007) "Topological Methods for the
//! Analysis of High Dimensional Data Sets and 3D Object Recognition"

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::slow_store::SlowStore;
use crate::similarity::cosine_similarity;

/// A filter function that maps entities to a scalar value.
/// Different filters reveal different topological structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MapperFilter {
    /// First principal component of the embedding (captures dominant variation)
    EmbeddingPC1,
    /// Distance to the embedding centroid (isolation/centrality)
    CentroidDistance,
    /// Local density (number of neighbors within radius)
    Density,
    /// Eccentricity (maximum distance to any other point)
    Eccentricity,
    /// Average neighbor distance (Voronoi cell size proxy)
    NeighborDistance,
}

impl MapperFilter {
    pub fn as_str(&self) -> &str {
        match self {
            Self::EmbeddingPC1 => "embedding_pc1",
            Self::CentroidDistance => "centroid_distance",
            Self::Density => "density",
            Self::Eccentricity => "eccentricity",
            Self::NeighborDistance => "neighbor_distance",
        }
    }
}

/// Configuration for the Mapper algorithm
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Which filter function to use
    pub filter: MapperFilter,
    /// Number of intervals in the cover
    pub num_intervals: usize,
    /// Overlap percentage between adjacent intervals (0.0-1.0)
    pub overlap: f32,
    /// Clustering radius (cosine distance threshold within each pullback)
    pub cluster_radius: f32,
    /// Maximum entities to process
    pub max_entities: usize,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            filter: MapperFilter::CentroidDistance,
            num_intervals: 10,
            overlap: 0.3,
            cluster_radius: 0.4,
            max_entities: 2000,
        }
    }
}

/// A node in the Mapper graph: represents a cluster of entities
/// within a specific interval of the filter function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperNode {
    /// Unique node ID
    pub id: usize,
    /// Entity indices in this cluster
    pub members: Vec<usize>,
    /// Entity names in this cluster
    pub member_names: Vec<String>,
    /// Number of entities
    pub size: usize,
    /// Average filter value of members
    pub avg_filter_value: f32,
    /// Which interval this cluster belongs to
    pub interval_index: usize,
}

/// An edge in the Mapper graph: two clusters share at least one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperEdge {
    /// Source node ID
    pub from: usize,
    /// Target node ID
    pub to: usize,
    /// Number of shared entities
    pub weight: usize,
}

/// The complete Mapper graph output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperGraph {
    /// Cluster nodes
    pub nodes: Vec<MapperNode>,
    /// Overlap edges
    pub edges: Vec<MapperEdge>,
    /// Which filter was used
    pub filter: String,
    /// Number of connected components (galaxies)
    pub num_components: usize,
    /// Number of loops (H₁ of the Mapper graph)
    pub num_loops: usize,
    /// Flare nodes: degree-1 nodes (endpoints/tendrils)
    pub flare_count: usize,
    /// Branch nodes: degree ≥ 3 (decision points in the topology)
    pub branch_count: usize,
    /// Statistics
    pub stats: MapperStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperStats {
    pub entity_count: usize,
    pub interval_count: usize,
    pub cluster_count: usize,
    pub edge_count: usize,
    pub duration_ms: u64,
}

/// Run the Mapper algorithm on entity embeddings from the slow store.
pub fn compute_mapper(store: &SlowStore, config: &MapperConfig) -> Result<MapperGraph> {
    let start = std::time::Instant::now();

    // Load embeddings
    let mut entities = store.load_all_embeddings()?;
    if entities.len() > config.max_entities {
        entities.truncate(config.max_entities);
    }

    let n = entities.len();
    if n < 2 {
        return Ok(MapperGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            filter: config.filter.as_str().to_string(),
            num_components: 0,
            num_loops: 0,
            flare_count: 0,
            branch_count: 0,
            stats: MapperStats {
                entity_count: n,
                interval_count: 0,
                cluster_count: 0,
                edge_count: 0,
                duration_ms: start.elapsed().as_millis() as u64,
            },
        });
    }

    let names: Vec<String> = entities.iter().map(|(_, name, _)| name.clone()).collect();
    let embeddings: Vec<&[f32]> = entities.iter().map(|(_, _, emb)| emb.as_slice()).collect();

    // Step 1: Compute filter values for each entity
    let filter_values = {
        let _span = tracing::info_span!("mapper.filter").entered();
        compute_filter_values(&embeddings, config.filter)
    };

    // Step 2: Build overlapping cover of the filter range
    let intervals = build_cover(&filter_values, config.num_intervals, config.overlap);

    // Step 3: For each interval, pull back and cluster
    let all_clusters: Vec<(usize, Vec<usize>)> = {
        let _span = tracing::info_span!("mapper.cluster").entered();
        let mut clusters_acc: Vec<(usize, Vec<usize>)> = Vec::new();

        for (interval_idx, (low, high)) in intervals.iter().enumerate() {
            // Pull back: find entities whose filter value falls in this interval
            let pullback: Vec<usize> = filter_values
                .iter()
                .enumerate()
                .filter(|(_, &v)| v >= *low && v <= *high)
                .map(|(i, _)| i)
                .collect();

            if pullback.is_empty() {
                continue;
            }

            // Cluster the pullback using single-linkage at cluster_radius
            let clusters = single_linkage_cluster(&pullback, &embeddings, config.cluster_radius);

            for cluster in clusters {
                if !cluster.is_empty() {
                    clusters_acc.push((interval_idx, cluster));
                }
            }
        }

        clusters_acc
    };

    // Step 4: Build nodes
    let nodes: Vec<MapperNode> = all_clusters
        .iter()
        .enumerate()
        .map(|(id, (interval_idx, members))| {
            let avg_fv = members.iter().map(|&i| filter_values[i]).sum::<f32>()
                / members.len() as f32;
            MapperNode {
                id,
                member_names: members.iter().map(|&i| names[i].clone()).collect(),
                size: members.len(),
                avg_filter_value: avg_fv,
                interval_index: *interval_idx,
                members: members.clone(),
            }
        })
        .collect();

    // Step 5: Build edges (connect clusters that share entities)
    let mut edges = Vec::new();
    #[allow(clippy::needless_range_loop)] // indexing parallel arrays by position
    for i in 0..nodes.len() {
        let set_i: HashSet<usize> = nodes[i].members.iter().copied().collect();
        for j in (i + 1)..nodes.len() {
            let overlap: usize = nodes[j]
                .members
                .iter()
                .filter(|m| set_i.contains(m))
                .count();
            if overlap > 0 {
                edges.push(MapperEdge {
                    from: i,
                    to: j,
                    weight: overlap,
                });
            }
        }
    }

    // Step 6: Compute graph topology statistics
    let num_components = count_components(nodes.len(), &edges);
    let num_loops = edges.len() + num_components - nodes.len(); // Euler characteristic: χ = V - E + F, loops = E - V + components

    // Degree analysis
    let mut degree: HashMap<usize, usize> = HashMap::new();
    for edge in &edges {
        *degree.entry(edge.from).or_default() += 1;
        *degree.entry(edge.to).or_default() += 1;
    }
    let flare_count = degree.values().filter(|&&d| d == 1).count();
    let branch_count = degree.values().filter(|&&d| d >= 3).count();

    let stats = MapperStats {
        entity_count: n,
        interval_count: intervals.len(),
        cluster_count: nodes.len(),
        edge_count: edges.len(),
        duration_ms: start.elapsed().as_millis() as u64,
    };

    tracing::info!(
        "Mapper: {} entities → {} clusters, {} edges, {} components, {} loops ({}ms)",
        n,
        nodes.len(),
        edges.len(),
        num_components,
        num_loops,
        stats.duration_ms
    );

    Ok(MapperGraph {
        nodes,
        edges,
        filter: config.filter.as_str().to_string(),
        num_components,
        num_loops,
        flare_count,
        branch_count,
        stats,
    })
}

/// Compute filter values based on the chosen filter function.
fn compute_filter_values(embeddings: &[&[f32]], filter: MapperFilter) -> Vec<f32> {
    let n = embeddings.len();

    match filter {
        MapperFilter::CentroidDistance => {
            // Distance from each point to the centroid
            let dim = embeddings[0].len();
            let mut centroid = vec![0.0f32; dim];
            for emb in embeddings {
                for (c, e) in centroid.iter_mut().zip(emb.iter()) {
                    *c += e;
                }
            }
            for c in &mut centroid {
                *c /= n as f32;
            }

            embeddings
                .iter()
                .map(|emb| 1.0 - cosine_similarity(emb, &centroid))
                .collect()
        }

        MapperFilter::Density => {
            // Local density: number of neighbors within radius 0.5
            let radius = 0.5;
            embeddings
                .iter()
                .enumerate()
                .map(|(i, emb_i)| {
                    let count = embeddings
                        .iter()
                        .enumerate()
                        .filter(|(j, emb_j)| {
                            *j != i && (1.0 - cosine_similarity(emb_i, emb_j)) <= radius
                        })
                        .count();
                    count as f32 / n as f32
                })
                .collect()
        }

        MapperFilter::Eccentricity => {
            // Maximum distance to any other point
            embeddings
                .iter()
                .enumerate()
                .map(|(i, emb_i)| {
                    embeddings
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, emb_j)| 1.0 - cosine_similarity(emb_i, emb_j))
                        .fold(0.0f32, f32::max)
                })
                .collect()
        }

        MapperFilter::NeighborDistance => {
            // Average distance to k nearest neighbors
            let k = 5.min(n - 1);
            embeddings
                .iter()
                .enumerate()
                .map(|(i, emb_i)| {
                    let mut dists: Vec<f32> = embeddings
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, emb_j)| 1.0 - cosine_similarity(emb_i, emb_j))
                        .collect();
                    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    dists.iter().take(k).sum::<f32>() / k as f32
                })
                .collect()
        }

        MapperFilter::EmbeddingPC1 => {
            // Approximate first principal component: project onto the direction
            // of maximum variance. Use power iteration for efficiency.
            let dim = embeddings[0].len();

            // Compute mean
            let mut mean = vec![0.0f32; dim];
            for emb in embeddings {
                for (m, e) in mean.iter_mut().zip(emb.iter()) {
                    *m += e;
                }
            }
            for m in &mut mean {
                *m /= n as f32;
            }

            // Power iteration to find first PC direction
            let mut pc = vec![1.0f32; dim]; // initial guess
            let norm: f32 = pc.iter().map(|x| x * x).sum::<f32>().sqrt();
            for p in &mut pc {
                *p /= norm;
            }

            for _ in 0..20 {
                // iteration
                let mut new_pc = vec![0.0f32; dim];
                for emb in embeddings {
                    // center
                    let centered: Vec<f32> =
                        emb.iter().zip(mean.iter()).map(|(e, m)| e - m).collect();
                    let projection: f32 =
                        centered.iter().zip(pc.iter()).map(|(c, p)| c * p).sum();
                    for (np, c) in new_pc.iter_mut().zip(centered.iter()) {
                        *np += projection * c;
                    }
                }
                let norm: f32 = new_pc.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for p in &mut new_pc {
                        *p /= norm;
                    }
                }
                pc = new_pc;
            }

            // Project each point onto PC1
            embeddings
                .iter()
                .map(|emb| {
                    let centered: Vec<f32> =
                        emb.iter().zip(mean.iter()).map(|(e, m)| e - m).collect();
                    centered.iter().zip(pc.iter()).map(|(c, p)| c * p).sum()
                })
                .collect()
        }
    }
}

/// Build overlapping cover of a range [min, max].
/// Returns a list of (low, high) intervals.
fn build_cover(values: &[f32], num_intervals: usize, overlap: f32) -> Vec<(f32, f32)> {
    if values.is_empty() || num_intervals == 0 {
        return Vec::new();
    }

    let min = values.iter().fold(f32::MAX, |a, &b| a.min(b));
    let max = values.iter().fold(f32::MIN, |a, &b| a.max(b));
    let range = max - min;

    if range < 1e-10 {
        return vec![(min, max)];
    }

    let step = range / num_intervals as f32;
    let overlap_size = step * overlap;

    (0..num_intervals)
        .map(|i| {
            let low = min + i as f32 * step - overlap_size;
            let high = min + (i + 1) as f32 * step + overlap_size;
            (low.max(min), high.min(max))
        })
        .collect()
}

/// Single-linkage clustering within a pullback set.
/// Merges points that are within cluster_radius of each other.
fn single_linkage_cluster(
    indices: &[usize],
    embeddings: &[&[f32]],
    radius: f32,
) -> Vec<Vec<usize>> {
    let n = indices.len();
    if n == 0 {
        return Vec::new();
    }

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            parent[px] = py;
        }
    }

    // Merge points within radius
    for i in 0..n {
        for j in (i + 1)..n {
            let dist =
                1.0 - cosine_similarity(embeddings[indices[i]], embeddings[indices[j]]);
            if dist <= radius {
                union(&mut parent, i, j);
            }
        }
    }

    // Group by root
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    #[allow(clippy::needless_range_loop)] // indexing into both parent and indices by position
    for i in 0..n {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(indices[i]);
    }

    clusters.into_values().collect()
}

/// Count connected components using BFS
fn count_components(num_nodes: usize, edges: &[MapperEdge]) -> usize {
    if num_nodes == 0 {
        return 0;
    }

    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for edge in edges {
        adj.entry(edge.from).or_default().push(edge.to);
        adj.entry(edge.to).or_default().push(edge.from);
    }

    let mut visited = vec![false; num_nodes];
    let mut components = 0;

    for start in 0..num_nodes {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    components
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_cover_empty() {
        assert!(build_cover(&[], 5, 0.2).is_empty());
    }

    #[test]
    fn test_build_cover_zero_intervals() {
        assert!(build_cover(&[1.0, 2.0], 0, 0.2).is_empty());
    }

    #[test]
    fn test_build_cover_single_value() {
        let intervals = build_cover(&[5.0, 5.0, 5.0], 3, 0.3);
        assert_eq!(intervals.len(), 1);
    }

    #[test]
    fn test_single_linkage_cluster_two_close() {
        let emb_a = vec![1.0, 0.0];
        let emb_b = vec![0.9, 0.1];
        let emb_c = vec![-1.0, 0.0];
        let embeddings: Vec<&[f32]> = vec![&emb_a, &emb_b, &emb_c];
        let clusters = single_linkage_cluster(&[0, 1, 2], &embeddings, 0.3);
        assert_eq!(clusters.len(), 2, "Should form 2 clusters");
    }

    #[test]
    fn test_count_components_disconnected() {
        assert_eq!(count_components(3, &[]), 3);
    }

    #[test]
    fn test_count_components_connected() {
        let edges = vec![
            MapperEdge { from: 0, to: 1, weight: 1 },
            MapperEdge { from: 1, to: 2, weight: 1 },
        ];
        assert_eq!(count_components(3, &edges), 1);
    }

    #[test]
    fn test_build_cover_basic() {
        let values = vec![0.0, 0.5, 1.0];
        let intervals = build_cover(&values, 5, 0.2);
        assert_eq!(intervals.len(), 5);
        // First interval should start at 0.0
        assert!((intervals[0].0 - 0.0).abs() < 0.01);
        // Last interval should end at 1.0
        assert!((intervals[4].1 - 1.0).abs() < 0.01);
        // Intervals should overlap
        assert!(intervals[0].1 > intervals[1].0);
    }
}
