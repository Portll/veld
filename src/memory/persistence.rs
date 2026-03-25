//! Persistent Homology via Rips Filtration
//!
//! Builds a Vietoris-Rips filtration from entity embeddings and computes
//! persistent homology to detect topological features that persist across scales.
//!
//! - H₀ features = connected components (knowledge galaxies)
//! - H₁ features = loops/cycles (circular reasoning, U-shape chains)
//! - H₂ features = voids (empty cavities surrounded by knowledge)
//!
//! Features that persist across many filtration levels are real structure.
//! Features that appear and vanish quickly are noise.
//!
//! The Čech-Rips Sandwich Theorem guarantees:
//!   C_ε ⊆ VR_ε ⊆ C_{2ε}
//! So any Rips feature at ε corresponds to a true topological feature
//! of the underlying space between radius ε and 2ε.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};

use super::slow_store::SlowStore;
use crate::similarity::cosine_similarity;

/// Maximum simplex dimension to compute (3 = tetrahedra, giving us H₀, H₁, H₂).
/// Going higher is exponential and rarely needed.
const MAX_SIMPLEX_DIM: usize = 3;

/// A simplex: an ordered set of vertex indices.
/// Dimension 0 = vertex, 1 = edge, 2 = triangle, 3 = tetrahedron.
type Simplex = BTreeSet<usize>;

/// A birth-death pair in the persistence diagram.
/// Birth = filtration value where the feature appears.
/// Death = filtration value where the feature disappears (None = persists forever).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    /// Homology dimension (0=component, 1=loop, 2=void)
    pub dimension: usize,
    /// Filtration value where this feature is born
    pub birth: f32,
    /// Filtration value where this feature dies (None = infinite persistence)
    pub death: Option<f32>,
    /// Persistence = death - birth (or f32::INFINITY if never dies)
    pub persistence: f32,
    /// Vertex indices involved in the birth simplex
    pub birth_simplex: Vec<usize>,
}

/// The persistence diagram: a complete topological fingerprint of the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    /// All birth-death pairs, sorted by persistence (longest first)
    pub pairs: Vec<PersistencePair>,
    /// Betti numbers at each filtration level
    pub betti_curves: Vec<BettiSnapshot>,
    /// Filtration values used
    pub filtration_values: Vec<f32>,
    /// Entity names for interpreting vertex indices
    pub vertex_names: Vec<String>,
    /// Sandwich theorem bounds: for each feature at ε,
    /// the true Čech feature exists between ε and 2ε
    pub sandwich_bounds: Vec<(f32, f32)>,
    /// Computation statistics
    pub stats: PersistenceStats,
}

/// Betti numbers at a specific filtration level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiSnapshot {
    pub epsilon: f32,
    /// β₀: number of connected components
    pub beta_0: usize,
    /// β₁: number of independent loops
    pub beta_1: usize,
    /// β₂: number of voids
    pub beta_2: usize,
    /// Total simplex count at this level
    pub simplex_count: usize,
}

/// Statistics from persistent homology computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceStats {
    pub entity_count: usize,
    pub filtration_levels: usize,
    pub total_pairs: usize,
    pub persistent_features: usize,
    pub noise_features: usize,
    pub duration_ms: u64,
}

/// Configuration for persistence computation
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Step size for filtration (default 0.1)
    pub step_size: f32,
    /// Maximum filtration radius (cosine distance)
    pub max_radius: f32,
    /// Minimum persistence to count as a real feature (not noise)
    pub min_persistence: f32,
    /// Maximum number of entities to process (for performance)
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

/// Compute persistent homology from entity embeddings in the slow store.
pub fn compute_persistence(
    store: &SlowStore,
    config: &PersistenceConfig,
) -> Result<PersistenceDiagram> {
    let start = std::time::Instant::now();

    // Load embeddings
    let mut entities = store.load_all_embeddings()?;
    if entities.len() > config.max_entities {
        // Keep highest-salience entities (they're returned in DB order,
        // but for now just truncate — could sort by salience)
        entities.truncate(config.max_entities);
    }

    let n = entities.len();
    if n < 2 {
        return Ok(PersistenceDiagram {
            pairs: Vec::new(),
            betti_curves: Vec::new(),
            filtration_values: Vec::new(),
            vertex_names: entities.iter().map(|(_, name, _)| name.clone()).collect(),
            sandwich_bounds: Vec::new(),
            stats: PersistenceStats {
                entity_count: n,
                filtration_levels: 0,
                total_pairs: 0,
                persistent_features: 0,
                noise_features: 0,
                duration_ms: start.elapsed().as_millis() as u64,
            },
        });
    }

    let vertex_names: Vec<String> = entities.iter().map(|(_, name, _)| name.clone()).collect();

    // Step 1: Compute sparse pairwise distance matrix (cosine distance)
    // Only store pairs below max_radius to save memory
    let distances = compute_sparse_distances(&entities, config.max_radius);

    // Step 2: Build filtration and track persistence
    let filtration_values: Vec<f32> = {
        let steps = (config.max_radius / config.step_size).ceil() as usize;
        (0..=steps)
            .map(|i| (i as f32 * config.step_size).min(config.max_radius))
            .collect()
    };

    // Step 3: Incremental Rips complex construction + homology at each level
    let mut all_simplices: Vec<(f32, Simplex)> = Vec::new();
    let mut betti_curves = Vec::new();

    // Sort edges by distance — this gives us the natural filtration order
    let mut sorted_edges: Vec<(f32, usize, usize)> = distances
        .iter()
        .map(|(&(i, j), &d)| (d, i, j))
        .collect();
    sorted_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Add vertices at ε=0 (all vertices exist from the start)
    for i in 0..n {
        let mut s = BTreeSet::new();
        s.insert(i);
        all_simplices.push((0.0, s));
    }

    // Add edges and higher simplices at their filtration values
    for &(dist, i, j) in &sorted_edges {
        let mut edge = BTreeSet::new();
        edge.insert(i);
        edge.insert(j);
        all_simplices.push((dist, edge));

        // Check for higher-dimensional simplices (triangles, tetrahedra)
        // by finding common neighbors that are already connected
        if MAX_SIMPLEX_DIM >= 2 {
            expand_cliques(
                i,
                j,
                dist,
                &distances,
                n,
                MAX_SIMPLEX_DIM,
                &mut all_simplices,
            );
        }
    }

    // Sort all simplices by filtration value, then by dimension
    all_simplices.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.len().cmp(&b.1.len()))
    });

    // Step 4: Compute persistence via the standard algorithm
    // (incremental reduction of the boundary matrix over Z₂)
    let pairs = compute_persistence_pairs(&all_simplices, n);

    // Step 5: Compute Betti curves at each filtration level
    for &eps in &filtration_values {
        let alive_at_eps: Vec<&PersistencePair> = pairs
            .iter()
            .filter(|p| p.birth <= eps && (p.death.is_none() || p.death.unwrap() > eps))
            .collect();

        let beta_0 = alive_at_eps.iter().filter(|p| p.dimension == 0).count();
        let beta_1 = alive_at_eps.iter().filter(|p| p.dimension == 1).count();
        let beta_2 = alive_at_eps.iter().filter(|p| p.dimension == 2).count();

        let simplex_count = all_simplices
            .iter()
            .filter(|(fv, _)| *fv <= eps)
            .count();

        betti_curves.push(BettiSnapshot {
            epsilon: eps,
            beta_0,
            beta_1,
            beta_2,
            simplex_count,
        });
    }

    // Sandwich bounds
    let sandwich_bounds: Vec<(f32, f32)> = pairs
        .iter()
        .map(|p| (p.birth, (p.birth * 2.0).min(config.max_radius)))
        .collect();

    let persistent_features = pairs
        .iter()
        .filter(|p| p.persistence >= config.min_persistence)
        .count();
    let noise_features = pairs.len() - persistent_features;

    let stats = PersistenceStats {
        entity_count: n,
        filtration_levels: filtration_values.len(),
        total_pairs: pairs.len(),
        persistent_features,
        noise_features,
        duration_ms: start.elapsed().as_millis() as u64,
    };

    tracing::info!(
        "Persistence: {} entities, {} pairs ({} persistent, {} noise) in {}ms",
        n,
        pairs.len(),
        persistent_features,
        noise_features,
        stats.duration_ms
    );

    Ok(PersistenceDiagram {
        pairs,
        betti_curves,
        filtration_values,
        vertex_names,
        sandwich_bounds,
        stats,
    })
}

/// Compute sparse pairwise cosine distances, only storing pairs below max_radius.
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
            if dist <= max_radius {
                distances.insert((i, j), dist);
            }
        }
    }

    distances
}

/// Expand cliques: given a new edge (i,j) at distance dist,
/// find all vertices k that are already connected to both i and j
/// at distance ≤ dist, forming triangles [i,j,k].
/// Recursively expand to tetrahedra if max_dim ≥ 3.
fn expand_cliques(
    i: usize,
    j: usize,
    dist: f32,
    distances: &HashMap<(usize, usize), f32>,
    n: usize,
    max_dim: usize,
    simplices: &mut Vec<(f32, Simplex)>,
) {
    if max_dim < 2 {
        return;
    }

    // Find common neighbors of i and j within distance dist
    let mut common_neighbors = Vec::new();
    for k in 0..n {
        if k == i || k == j {
            continue;
        }
        let d_ik = get_dist(distances, i, k);
        let d_jk = get_dist(distances, j, k);

        if let (Some(dik), Some(djk)) = (d_ik, d_jk) {
            if dik <= dist && djk <= dist {
                common_neighbors.push(k);
            }
        }
    }

    // Add triangles [i, j, k] for each common neighbor k
    for &k in &common_neighbors {
        let mut triangle = BTreeSet::new();
        triangle.insert(i);
        triangle.insert(j);
        triangle.insert(k);

        // Filtration value = max edge distance in the simplex
        let filt_val = dist
            .max(get_dist(distances, i, k).unwrap_or(dist))
            .max(get_dist(distances, j, k).unwrap_or(dist));

        simplices.push((filt_val, triangle));

        // Expand to tetrahedra if max_dim >= 3
        if max_dim >= 3 {
            for &l in &common_neighbors {
                if l <= k {
                    continue;
                }
                let d_kl = get_dist(distances, k, l);
                let d_il = get_dist(distances, i, l);
                let d_jl = get_dist(distances, j, l);

                if let (Some(dkl), Some(dil), Some(djl)) = (d_kl, d_il, d_jl) {
                    if dkl <= dist && dil <= dist && djl <= dist {
                        let mut tet = BTreeSet::new();
                        tet.insert(i);
                        tet.insert(j);
                        tet.insert(k);
                        tet.insert(l);

                        let tet_filt = dist
                            .max(dkl)
                            .max(dil)
                            .max(djl)
                            .max(get_dist(distances, i, k).unwrap_or(dist))
                            .max(get_dist(distances, j, k).unwrap_or(dist));

                        simplices.push((tet_filt, tet));
                    }
                }
            }
        }
    }
}

/// Get distance between two vertices (handles index ordering)
fn get_dist(distances: &HashMap<(usize, usize), f32>, i: usize, j: usize) -> Option<f32> {
    if i == j {
        return Some(0.0);
    }
    let key = if i < j { (i, j) } else { (j, i) };
    distances.get(&key).copied()
}

/// Compute persistence pairs using the standard algorithm.
///
/// This implements the persistence algorithm over Z₂ (mod 2):
/// - Process simplices in filtration order
/// - For each simplex, compute its boundary
/// - Reduce the boundary matrix column by column
/// - A simplex that reduces to zero creates a new feature (birth)
/// - A simplex whose reduced column is non-zero kills a feature (death)
fn compute_persistence_pairs(
    simplices: &[(f32, Simplex)],
    _n: usize,
) -> Vec<PersistencePair> {
    let num_simplices = simplices.len();

    // Map from simplex → index in filtration order
    let simplex_index: HashMap<&Simplex, usize> = simplices
        .iter()
        .enumerate()
        .map(|(idx, (_, s))| (s, idx))
        .collect();

    // Reduced boundary matrix: for each column, store the set of row indices (Z₂)
    // A row index being present means the coefficient is 1 (mod 2)
    let mut reduced: Vec<HashSet<usize>> = Vec::with_capacity(num_simplices);
    // low[j] = the lowest row index in reduced column j (or None if zero column)
    let mut low: Vec<Option<usize>> = Vec::with_capacity(num_simplices);
    // Map from low index → column index (for the reduction algorithm)
    let mut low_to_col: HashMap<usize, usize> = HashMap::new();
    // Track which simplices are "positive" (create features) vs "negative" (kill features)
    let mut partner: Vec<Option<usize>> = vec![None; num_simplices];

    for j in 0..num_simplices {
        let (_, ref sigma) = simplices[j];

        // Compute boundary of sigma over Z₂
        // boundary(σ) = sum of faces of σ (each face is σ with one vertex removed)
        let mut boundary_col: HashSet<usize> = HashSet::new();
        if sigma.len() > 1 {
            for vertex in sigma {
                let mut face = sigma.clone();
                face.remove(vertex);
                if let Some(&face_idx) = simplex_index.get(&face) {
                    // XOR: if already present, remove (mod 2)
                    if !boundary_col.remove(&face_idx) {
                        boundary_col.insert(face_idx);
                    }
                }
            }
        }

        // Reduce column j using previous columns
        loop {
            let low_j = boundary_col.iter().max().copied();
            match low_j {
                None => break, // Zero column → positive simplex (birth)
                Some(l) => {
                    if let Some(&col_with_same_low) = low_to_col.get(&l) {
                        // XOR with that column (Z₂ elimination)
                        let other = &reduced[col_with_same_low];
                        let symmetric_diff: HashSet<usize> = boundary_col
                            .symmetric_difference(other)
                            .copied()
                            .collect();
                        boundary_col = symmetric_diff;
                    } else {
                        break; // No column has this low → we're done reducing
                    }
                }
            }
        }

        let low_j = boundary_col.iter().max().copied();
        reduced.push(boundary_col);
        low.push(low_j);

        if let Some(l) = low_j {
            // Negative simplex: kills the feature born at simplex l
            low_to_col.insert(l, j);
            partner[j] = Some(l);
            partner[l] = Some(j);
        }
    }

    // Extract persistence pairs
    let mut pairs = Vec::new();

    for j in 0..num_simplices {
        let (birth_val, ref sigma) = simplices[j];
        let dim = sigma.len() - 1; // dimension of the simplex

        if low[j].is_none() {
            // Positive simplex: creates a feature
            let death = if let Some(killer) = partner[j] {
                Some(simplices[killer].0)
            } else {
                None // Feature persists to infinity
            };

            let persistence = match death {
                Some(d) => d - birth_val,
                None => f32::INFINITY,
            };

            // Only record features with non-trivial persistence
            // (skip features that are born and die at the same ε)
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

    // Sort by persistence (longest first)
    pairs.sort_by(|a, b| {
        b.persistence
            .partial_cmp(&a.persistence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_homology() {
        // Three points forming a triangle should have:
        // H₀: 1 component (born at 0, never dies)
        // H₁: 0 loops (triangle is filled)
        let mut simplices = Vec::new();

        // Vertices at ε=0
        simplices.push((0.0, [0].into_iter().collect()));
        simplices.push((0.0, [1].into_iter().collect()));
        simplices.push((0.0, [2].into_iter().collect()));

        // Edges at ε=0.5
        simplices.push((0.5, [0, 1].into_iter().collect()));
        simplices.push((0.5, [0, 2].into_iter().collect()));
        simplices.push((0.5, [1, 2].into_iter().collect()));

        // Triangle (2-simplex) at ε=0.5
        simplices.push((0.5, [0, 1, 2].into_iter().collect()));

        let pairs = compute_persistence_pairs(&simplices, 3);

        // Should have H₀ features
        let h0: Vec<_> = pairs.iter().filter(|p| p.dimension == 0).collect();
        assert!(!h0.is_empty(), "Should have H₀ features");

        // The loop created by edges should be killed by the triangle
        // So H₁ should have no persistent features
        let h1_persistent: Vec<_> = pairs
            .iter()
            .filter(|p| p.dimension == 1 && p.death.is_none())
            .collect();
        assert!(
            h1_persistent.is_empty(),
            "Filled triangle should have no persistent H₁"
        );
    }
}
