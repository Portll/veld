//! Persistent Homology via Rips Filtration
//!
//! Computes the topological fingerprint of a point cloud in embedding space.
//! The pipeline has 5 phases:
//!
//! 1. **Distance matrix** — sparse pairwise cosine distances (O(n²), guarded for NaN)
//! 2. **Rips complex** — build simplicial complex from edges + clique expansion
//! 3. **Z₂ reduction** — standard persistence algorithm over the boundary matrix
//! 4. **Betti curves** — count alive features at each filtration level
//! 5. **Assembly** — sandwich bounds, statistics, final diagram
//!
//! Output:
//! - H₀ = connected components (knowledge clusters)
//! - H₁ = loops/cycles (circular reference chains)
//! - H₂ = voids (cavities surrounded by knowledge)
//!
//! The Čech-Rips Sandwich Theorem guarantees:
//!   C_ε ⊆ VR_ε ⊆ C_{2ε}
//! So any Rips feature at ε corresponds to a true topological feature
//! between radius ε and 2ε.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};

use super::slow_store::SlowStore;
use crate::similarity::cosine_similarity;

// =============================================================================
// TYPES
// =============================================================================

/// Maximum simplex dimension (3 = tetrahedra → H₀, H₁, H₂).
const MAX_SIMPLEX_DIM: usize = 3;

/// Ordered set of vertex indices. Dim 0 = vertex, 1 = edge, 2 = triangle.
type Simplex = BTreeSet<usize>;

/// A birth-death pair in the persistence diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    /// Homology dimension (0=component, 1=loop, 2=void)
    pub dimension: usize,
    /// Filtration value where this feature is born
    pub birth: f32,
    /// Filtration value where this feature dies (None = infinite persistence)
    pub death: Option<f32>,
    /// Persistence = death - birth (f32::INFINITY if never dies)
    pub persistence: f32,
    /// Vertex indices in the birth simplex
    pub birth_simplex: Vec<usize>,
}

/// Complete topological fingerprint of the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
    pub betti_curves: Vec<BettiSnapshot>,
    pub filtration_values: Vec<f32>,
    pub vertex_names: Vec<String>,
    pub sandwich_bounds: Vec<(f32, f32)>,
    pub stats: PersistenceStats,
}

/// Betti numbers at a specific filtration level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiSnapshot {
    pub epsilon: f32,
    pub beta_0: usize,
    pub beta_1: usize,
    pub beta_2: usize,
    pub simplex_count: usize,
}

/// Computation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceStats {
    pub entity_count: usize,
    pub filtration_levels: usize,
    pub total_pairs: usize,
    pub persistent_features: usize,
    pub noise_features: usize,
    pub duration_ms: u64,
}

/// Configuration for persistence computation.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    pub step_size: f32,
    pub max_radius: f32,
    pub min_persistence: f32,
    pub max_entities: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            max_radius: 1.0,
            min_persistence: 0.15,
            max_entities: 2000,
        }
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Compute persistent homology from entity embeddings.
pub fn compute_persistence(
    store: &SlowStore,
    config: &PersistenceConfig,
) -> Result<PersistenceDiagram> {
    let start = std::time::Instant::now();

    let mut entities = store.load_all_embeddings()?;
    if entities.len() > config.max_entities {
        entities.truncate(config.max_entities);
    }

    let n = entities.len();
    if n < 2 {
        return Ok(empty_diagram(
            &entities,
            start.elapsed().as_millis() as u64,
        ));
    }

    let vertex_names: Vec<String> = entities.iter().map(|(_, name, _)| name.clone()).collect();

    // Phase 1: Sparse distance matrix
    let distances = {
        let _span = tracing::info_span!("persistence.distances", entities = n).entered();
        compute_sparse_distances(&entities, config.max_radius)
    };

    // Phase 2: Build Rips complex (vertices + edges + higher simplices)
    let (all_simplices, filtration_values) = {
        let _span = tracing::info_span!("persistence.simplices").entered();
        build_rips_complex(n, &distances, config)
    };

    // Phase 3: Z₂ boundary matrix reduction → persistence pairs
    let pairs = {
        let _span = tracing::info_span!("persistence.reduce", simplices = all_simplices.len()).entered();
        reduce_boundary_matrix(&all_simplices)
    };

    // Phase 4: Betti curves at each filtration level
    let betti_curves = {
        let _span = tracing::info_span!("persistence.betti").entered();
        compute_betti_curves(&pairs, &all_simplices, &filtration_values)
    };

    // Phase 5: Assemble diagram
    let sandwich_bounds: Vec<(f32, f32)> = pairs
        .iter()
        .map(|p| (p.birth, (p.birth * 2.0).min(config.max_radius)))
        .collect();

    let persistent_features = pairs
        .iter()
        .filter(|p| p.persistence >= config.min_persistence)
        .count();
    let noise_features = pairs.len() - persistent_features;
    let duration_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        entities = n,
        pairs = pairs.len(),
        persistent = persistent_features,
        noise = noise_features,
        duration_ms,
        "Persistence computation complete"
    );

    let filtration_levels = filtration_values.len();
    Ok(PersistenceDiagram {
        pairs,
        betti_curves,
        filtration_values,
        vertex_names,
        sandwich_bounds,
        stats: PersistenceStats {
            entity_count: n,
            filtration_levels,
            total_pairs: persistent_features + noise_features,
            persistent_features,
            noise_features,
            duration_ms,
        },
    })
}

// =============================================================================
// PHASE 1: DISTANCE MATRIX
// =============================================================================

/// Sparse pairwise cosine distances. Only stores pairs within max_radius.
fn compute_sparse_distances(
    entities: &[(String, String, Vec<f32>)],
    max_radius: f32,
) -> HashMap<(usize, usize), f32> {
    let n = entities.len();
    let mut distances = HashMap::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&entities[i].2, &entities[j].2);
            let dist = 1.0 - sim;
            if dist.is_finite() && dist <= max_radius {
                distances.insert((i, j), dist);
            }
        }
    }

    distances
}

/// Symmetric distance lookup (handles (i,j) vs (j,i) ordering).
fn get_dist(distances: &HashMap<(usize, usize), f32>, i: usize, j: usize) -> Option<f32> {
    if i == j {
        return Some(0.0);
    }
    let key = if i < j { (i, j) } else { (j, i) };
    distances.get(&key).copied()
}

// =============================================================================
// PHASE 2: RIPS COMPLEX CONSTRUCTION
// =============================================================================

/// Build the Vietoris-Rips simplicial complex from the distance matrix.
///
/// Returns (simplices_in_filtration_order, filtration_level_values).
fn build_rips_complex(
    n: usize,
    distances: &HashMap<(usize, usize), f32>,
    config: &PersistenceConfig,
) -> (Vec<(f32, Simplex)>, Vec<f32>) {
    let mut simplices: Vec<(f32, Simplex)> = Vec::new();

    // Vertices at ε=0
    for i in 0..n {
        simplices.push((0.0, BTreeSet::from([i])));
    }

    // Edges sorted by distance (natural filtration order)
    let mut sorted_edges: Vec<(f32, usize, usize)> = distances
        .iter()
        .map(|(&(i, j), &d)| (d, i, j))
        .collect();
    sorted_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Add edges and expand to higher simplices (triangles, tetrahedra)
    for &(dist, i, j) in &sorted_edges {
        simplices.push((dist, BTreeSet::from([i, j])));

        if MAX_SIMPLEX_DIM >= 2 {
            expand_cliques(i, j, dist, distances, n, &mut simplices);
        }
    }

    // Sort by (filtration_value, dimension) for the reduction algorithm
    simplices.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.len().cmp(&b.1.len()))
    });

    // Build filtration level values
    let steps = (config.max_radius / config.step_size).ceil() as usize;
    let filtration_values: Vec<f32> = (0..=steps)
        .map(|i| (i as f32 * config.step_size).min(config.max_radius))
        .collect();

    (simplices, filtration_values)
}

/// Expand cliques: given edge (i,j), find common neighbors forming triangles,
/// then expand triangles to tetrahedra.
fn expand_cliques(
    i: usize,
    j: usize,
    dist: f32,
    distances: &HashMap<(usize, usize), f32>,
    n: usize,
    simplices: &mut Vec<(f32, Simplex)>,
) {
    // Find vertices connected to both i and j within distance
    let common: Vec<usize> = (0..n)
        .filter(|&k| k != i && k != j)
        .filter(|&k| {
            matches!(
                (get_dist(distances, i, k), get_dist(distances, j, k)),
                (Some(dik), Some(djk)) if dik <= dist && djk <= dist
            )
        })
        .collect();

    for &k in &common {
        // Triangle [i, j, k]
        let filt_val = dist
            .max(get_dist(distances, i, k).unwrap_or(dist))
            .max(get_dist(distances, j, k).unwrap_or(dist));
        simplices.push((filt_val, BTreeSet::from([i, j, k])));

        // Tetrahedra [i, j, k, l] for each l > k also in common
        if MAX_SIMPLEX_DIM >= 3 {
            for &l in &common {
                if l <= k {
                    continue;
                }
                if let (Some(dkl), Some(dil), Some(djl)) = (
                    get_dist(distances, k, l),
                    get_dist(distances, i, l),
                    get_dist(distances, j, l),
                ) {
                    if dkl <= dist && dil <= dist && djl <= dist {
                        let tet_filt = dist
                            .max(dkl)
                            .max(dil)
                            .max(djl)
                            .max(get_dist(distances, i, k).unwrap_or(dist))
                            .max(get_dist(distances, j, k).unwrap_or(dist));
                        simplices.push((tet_filt, BTreeSet::from([i, j, k, l])));
                    }
                }
            }
        }
    }
}

// =============================================================================
// PHASE 3: Z₂ BOUNDARY MATRIX REDUCTION
// =============================================================================

/// Standard persistence algorithm over Z₂.
///
/// Process simplices in filtration order. For each:
/// - Compute boundary (faces with coefficient 1 mod 2)
/// - Reduce column via XOR with previous columns sharing the same "low" index
/// - Zero column = new feature (birth). Non-zero = feature death.
fn reduce_boundary_matrix(simplices: &[(f32, Simplex)]) -> Vec<PersistencePair> {
    let num = simplices.len();

    // Simplex → filtration index
    let index_of: HashMap<&Simplex, usize> = simplices
        .iter()
        .enumerate()
        .map(|(idx, (_, s))| (s, idx))
        .collect();

    let mut reduced: Vec<HashSet<usize>> = Vec::with_capacity(num);
    let mut low: Vec<Option<usize>> = Vec::with_capacity(num);
    let mut low_to_col: HashMap<usize, usize> = HashMap::new();
    let mut partner: Vec<Option<usize>> = vec![None; num];

    for j in 0..num {
        let sigma = &simplices[j].1;

        // Boundary over Z₂: XOR of face indices
        let mut col = compute_boundary(sigma, &index_of);

        // Column reduction
        loop {
            match col.iter().max().copied() {
                None => break,
                Some(l) => match low_to_col.get(&l) {
                    Some(&prev) => {
                        col = col.symmetric_difference(&reduced[prev]).copied().collect();
                    }
                    None => break,
                },
            }
        }

        let low_j = col.iter().max().copied();
        reduced.push(col);
        low.push(low_j);

        if let Some(l) = low_j {
            low_to_col.insert(l, j);
            partner[j] = Some(l);
            partner[l] = Some(j);
        }
    }

    // Extract birth-death pairs
    extract_pairs(simplices, &low, &partner)
}

/// Compute boundary of a simplex over Z₂ (each face = simplex with one vertex removed).
fn compute_boundary(sigma: &Simplex, index_of: &HashMap<&Simplex, usize>) -> HashSet<usize> {
    let mut boundary = HashSet::new();
    if sigma.len() <= 1 {
        return boundary;
    }
    for vertex in sigma {
        let mut face = sigma.clone();
        face.remove(vertex);
        if let Some(&face_idx) = index_of.get(&face) {
            if !boundary.remove(&face_idx) {
                boundary.insert(face_idx);
            }
        }
    }
    boundary
}

/// Extract persistence pairs from the reduced matrix.
fn extract_pairs(
    simplices: &[(f32, Simplex)],
    low: &[Option<usize>],
    partner: &[Option<usize>],
) -> Vec<PersistencePair> {
    let mut pairs = Vec::new();

    for j in 0..simplices.len() {
        let (birth_val, ref sigma) = simplices[j];
        let dim = sigma.len() - 1;

        // Positive simplex (zero column) = creates a feature
        if low[j].is_none() {
            let death = partner[j].map(|killer| simplices[killer].0);
            let persistence = match death {
                Some(d) => d - birth_val,
                None => f32::INFINITY,
            };

            if persistence > 0.001 || death.is_none() {
                pairs.push(PersistencePair {
                    dimension: dim,
                    birth: birth_val,
                    death,
                    persistence,
                    birth_simplex: sigma.iter().copied().collect(),
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.persistence
            .partial_cmp(&a.persistence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    pairs
}

// =============================================================================
// PHASE 4: BETTI CURVES
// =============================================================================

/// Count alive features at each filtration level.
fn compute_betti_curves(
    pairs: &[PersistencePair],
    simplices: &[(f32, Simplex)],
    filtration_values: &[f32],
) -> Vec<BettiSnapshot> {
    filtration_values
        .iter()
        .map(|&eps| {
            let alive: Vec<&PersistencePair> = pairs
                .iter()
                .filter(|p| p.birth <= eps && (p.death.is_none() || p.death.unwrap() > eps))
                .collect();

            BettiSnapshot {
                epsilon: eps,
                beta_0: alive.iter().filter(|p| p.dimension == 0).count(),
                beta_1: alive.iter().filter(|p| p.dimension == 1).count(),
                beta_2: alive.iter().filter(|p| p.dimension == 2).count(),
                simplex_count: simplices.iter().filter(|(fv, _)| *fv <= eps).count(),
            }
        })
        .collect()
}

// =============================================================================
// HELPERS
// =============================================================================

/// Empty diagram for degenerate inputs (< 2 entities).
fn empty_diagram(
    entities: &[(String, String, Vec<f32>)],
    duration_ms: u64,
) -> PersistenceDiagram {
    PersistenceDiagram {
        pairs: Vec::new(),
        betti_curves: Vec::new(),
        filtration_values: Vec::new(),
        vertex_names: entities.iter().map(|(_, name, _)| name.clone()).collect(),
        sandwich_bounds: Vec::new(),
        stats: PersistenceStats {
            entity_count: entities.len(),
            filtration_levels: 0,
            total_pairs: 0,
            persistent_features: 0,
            noise_features: 0,
            duration_ms,
        },
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_simplices() {
        let pairs = reduce_boundary_matrix(&[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_single_vertex() {
        let simplices = vec![(0.0, BTreeSet::from([0]))];
        let pairs = reduce_boundary_matrix(&simplices);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].dimension, 0);
        assert!(pairs[0].death.is_none());
    }

    #[test]
    fn test_two_vertices_one_edge() {
        let simplices = vec![
            (0.0, BTreeSet::from([0])),
            (0.0, BTreeSet::from([1])),
            (0.5, BTreeSet::from([0, 1])),
        ];
        let pairs = reduce_boundary_matrix(&simplices);
        let persistent: Vec<_> = pairs
            .iter()
            .filter(|p| p.dimension == 0 && p.death.is_none())
            .collect();
        assert_eq!(persistent.len(), 1);
    }

    #[test]
    fn test_triangle_homology() {
        let simplices = vec![
            (0.0, BTreeSet::from([0])),
            (0.0, BTreeSet::from([1])),
            (0.0, BTreeSet::from([2])),
            (0.5, BTreeSet::from([0, 1])),
            (0.5, BTreeSet::from([0, 2])),
            (0.5, BTreeSet::from([1, 2])),
            (0.5, BTreeSet::from([0, 1, 2])),
        ];
        let pairs = reduce_boundary_matrix(&simplices);

        let h0: Vec<_> = pairs.iter().filter(|p| p.dimension == 0).collect();
        assert!(!h0.is_empty(), "Should have H₀ features");

        let h1_persistent: Vec<_> = pairs
            .iter()
            .filter(|p| p.dimension == 1 && p.death.is_none())
            .collect();
        assert!(h1_persistent.is_empty(), "Filled triangle = no persistent H₁");
    }

    #[test]
    fn test_sparse_distances_finite() {
        let entities = vec![
            ("a".into(), "A".into(), vec![1.0, 0.0, 0.0]),
            ("b".into(), "B".into(), vec![f32::NAN, 0.0, 0.0]),
            ("c".into(), "C".into(), vec![0.0, 1.0, 0.0]),
        ];
        let distances = compute_sparse_distances(&entities, 1.5);
        for &dist in distances.values() {
            assert!(dist.is_finite(), "All distances must be finite, got {dist}");
        }
    }

    #[test]
    fn test_get_dist_symmetry() {
        let distances = HashMap::from([((0, 1), 0.5)]);
        assert_eq!(get_dist(&distances, 0, 1), Some(0.5));
        assert_eq!(get_dist(&distances, 1, 0), Some(0.5));
        assert_eq!(get_dist(&distances, 0, 0), Some(0.0));
        assert_eq!(get_dist(&distances, 2, 3), None);
    }

    #[test]
    fn test_boundary_of_vertex_is_empty() {
        let sigma = BTreeSet::from([0]);
        let index_of = HashMap::from([(&sigma, 0)]);
        assert!(compute_boundary(&sigma, &index_of).is_empty());
    }

    #[test]
    fn test_boundary_of_edge() {
        let v0 = BTreeSet::from([0]);
        let v1 = BTreeSet::from([1]);
        let edge = BTreeSet::from([0, 1]);
        let index_of = HashMap::from([(&v0, 0), (&v1, 1), (&edge, 2)]);
        let boundary = compute_boundary(&edge, &index_of);
        assert_eq!(boundary.len(), 2);
        assert!(boundary.contains(&0));
        assert!(boundary.contains(&1));
    }

    #[test]
    fn test_empty_diagram_for_single_entity() {
        let entities = vec![("a".into(), "A".into(), vec![1.0])];
        let diagram = empty_diagram(&entities, 42);
        assert_eq!(diagram.stats.entity_count, 1);
        assert_eq!(diagram.stats.duration_ms, 42);
        assert!(diagram.pairs.is_empty());
    }

    #[test]
    fn test_build_rips_complex_three_points() {
        let distances = HashMap::from([((0, 1), 0.3), ((0, 2), 0.4), ((1, 2), 0.5)]);
        let config = PersistenceConfig::default();
        let (simplices, _) = build_rips_complex(3, &distances, &config);
        // 3 vertices + 3 edges + 1 triangle = 7 simplices
        assert!(simplices.len() >= 7, "Expected ≥7 simplices, got {}", simplices.len());
    }
}
