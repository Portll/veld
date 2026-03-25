//! Voronoi Analysis of Knowledge Space
//!
//! Standard graph reasoning concentrates at structural points: entities (seeds),
//! relationships (edges), and intersections (vertices). But information also lives
//! in the *regions* — the Voronoi faces between structural elements.
//!
//! A Voronoi tessellation of the embedding space assigns every point in the space
//! to the nearest entity. The resulting cells encode:
//! - **Volume**: How isolated is this entity? (large cell = sparse knowledge)
//! - **Shape**: Is knowledge evenly distributed around it, or clustered in certain directions?
//! - **Neighbors**: Which entities share a boundary? (Delaunay dual)
//! - **Face area**: How prominent is each boundary between neighbors?
//! - **Density gradient**: How does knowledge density change across a face?
//!
//! Exact Voronoi in 384 dimensions is intractable. We approximate the key
//! properties using k-nearest-neighbor analysis in embedding space.
//!
//! This module detects:
//! - **Voids**: Large empty regions between entity clusters
//! - **Planet X**: Points where multiple entities' relationships converge but no entity exists
//! - **Anisotropy**: Directions where knowledge thins out around an entity
//! - **Density gradients**: Where knowledge drops off across a face

use anyhow::Result;
use serde::{Deserialize, Serialize};
use super::slow_store::SlowStore;
use crate::similarity::cosine_similarity;

/// Properties of one entity's approximate Voronoi cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiCell {
    /// Entity UUID
    pub uuid: String,
    /// Entity name
    pub name: String,
    /// Isolation score: average distance to k nearest neighbors.
    /// Higher = more isolated = larger Voronoi cell = sparser knowledge around this entity.
    /// Relative to the global average (1.0 = average, >1 = more isolated).
    pub isolation: f32,
    /// Anisotropy ratio: max eigenvalue / min eigenvalue of neighbor direction covariance.
    /// 1.0 = spherical cell (knowledge evenly distributed around entity).
    /// >1.0 = elongated cell (knowledge concentrated in certain directions, sparse in others).
    pub anisotropy: f32,
    /// Number of Voronoi neighbors (Delaunay adjacency)
    pub neighbor_count: usize,
    /// Voronoi neighbors: (uuid, name, face_prominence).
    /// Face prominence: how large the shared boundary is (higher = more prominent boundary).
    pub neighbors: Vec<(String, String, f32)>,
    /// Sparse directions: embedding-space directions where no neighbors exist.
    /// These are the directions where knowledge thins out around this entity.
    /// Stored as the principal component of the "empty" subspace.
    pub sparse_direction: Option<Vec<f32>>,
    /// Local density: number of entities within a radius, normalized.
    pub local_density: f32,
}

/// A void in embedding space: a region surrounded by entities but containing none.
///
/// Pure geometric data. Interpretation is left to consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiVoid {
    /// Approximate centroid of the void in embedding space
    pub centroid: Vec<f32>,
    /// Entities that border this void
    pub boundary_entities: Vec<(String, String)>, // (uuid, name)
    /// Estimated "radius" — average distance from centroid to boundary entities
    pub radius: f32,
    /// How many entities contribute to defining this void
    pub boundary_count: usize,
    /// Confidence that this is a meaningful void (not just noise)
    pub confidence: f32,
}

/// Configuration for Voronoi analysis
#[derive(Debug, Clone)]
pub struct VoronoiConfig {
    /// Number of nearest neighbors to consider per entity
    pub k: usize,
    /// Minimum isolation score to flag as a potential void boundary
    pub void_isolation_threshold: f32,
    /// Minimum number of entities that must border a void for it to be significant
    pub min_void_boundary: usize,
    /// Maximum number of voids to return
    pub max_voids: usize,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            k: 8,
            void_isolation_threshold: 1.3,
            min_void_boundary: 3,
            max_voids: 20,
        }
    }
}

/// Results from Voronoi analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiAnalysis {
    /// Cell properties for every entity with an embedding
    pub cells: Vec<VoronoiCell>,
    /// Detected voids (empty regions in embedding space)
    pub voids: Vec<VoronoiVoid>,
    /// Global statistics
    pub stats: VoronoiStats,
}

/// Global statistics from Voronoi analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiStats {
    /// Total entities analyzed
    pub entity_count: usize,
    /// Average isolation score
    pub avg_isolation: f32,
    /// Max isolation score (most isolated entity)
    pub max_isolation: f32,
    /// Average anisotropy
    pub avg_anisotropy: f32,
    /// Number of highly isolated entities (isolation > threshold)
    pub isolated_count: usize,
    /// Analysis duration
    pub duration_ms: u64,
}

/// Voronoi analyzer: computes approximate Voronoi cell properties
/// and detects voids and Planet X candidates in knowledge space.
pub struct VoronoiAnalyzer;

impl VoronoiAnalyzer {
    /// Run full Voronoi analysis on the knowledge graph.
    pub fn analyze(store: &SlowStore, config: &VoronoiConfig) -> Result<VoronoiAnalysis> {
        let start = std::time::Instant::now();

        // Load all embeddings
        let entities = store.load_all_embeddings()?;
        if entities.len() < 3 {
            return Ok(VoronoiAnalysis {
                cells: Vec::new(),
                voids: Vec::new(),
                stats: VoronoiStats {
                    entity_count: entities.len(),
                    avg_isolation: 0.0,
                    max_isolation: 0.0,
                    avg_anisotropy: 1.0,
                    isolated_count: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                },
            });
        }

        // Phase 1: Compute k-NN and cell properties
        let knn = Self::compute_knn(&entities, config.k);
        let cells = Self::compute_cells(&entities, &knn);

        // Global average isolation for normalization
        let global_avg_isolation = if cells.is_empty() {
            1.0
        } else {
            cells.iter().map(|c| c.isolation).sum::<f32>() / cells.len() as f32
        };

        // Normalize isolation scores relative to global average
        let cells: Vec<VoronoiCell> = cells
            .into_iter()
            .map(|mut c| {
                if global_avg_isolation > 0.0 {
                    c.isolation /= global_avg_isolation;
                }
                c
            })
            .collect();

        // Phase 2: Detect voids
        let voids = Self::detect_voids(&entities, &cells, config);

        // Compute stats
        let max_isolation = cells
            .iter()
            .map(|c| c.isolation)
            .fold(0.0f32, f32::max);
        let avg_anisotropy = if cells.is_empty() {
            1.0
        } else {
            cells.iter().map(|c| c.anisotropy).sum::<f32>() / cells.len() as f32
        };
        let isolated_count = cells
            .iter()
            .filter(|c| c.isolation > config.void_isolation_threshold)
            .count();

        let stats = VoronoiStats {
            entity_count: entities.len(),
            avg_isolation: 1.0, // normalized to 1.0
            max_isolation,
            avg_anisotropy,
            isolated_count,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        tracing::info!(
            "Voronoi analysis: {} entities, {} voids, \
             avg anisotropy {:.2}, {} isolated ({}ms)",
            stats.entity_count,
            voids.len(),
            stats.avg_anisotropy,
            stats.isolated_count,
            stats.duration_ms
        );

        Ok(VoronoiAnalysis {
            cells,
            voids,
            stats,
        })
    }

    /// Compute k nearest neighbors for each entity using cosine distance.
    ///
    /// Returns: entity_index → [(neighbor_index, cosine_distance)]
    fn compute_knn(
        entities: &[(String, String, Vec<f32>)],
        k: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let n = entities.len();
        let k = k.min(n - 1);

        let mut knn = Vec::with_capacity(n);
        for i in 0..n {
            let mut distances: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let sim = cosine_similarity(&entities[i].2, &entities[j].2);
                    (j, 1.0 - sim) // cosine distance
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(k);
            knn.push(distances);
        }

        knn
    }

    /// Compute Voronoi cell properties from k-NN data.
    fn compute_cells(
        entities: &[(String, String, Vec<f32>)],
        knn: &[Vec<(usize, f32)>],
    ) -> Vec<VoronoiCell> {
        let dim = entities.first().map(|(_, _, e)| e.len()).unwrap_or(384);

        entities
            .iter()
            .enumerate()
            .map(|(i, (uuid, name, embedding))| {
                let neighbors = &knn[i];

                // Isolation: average distance to k nearest neighbors
                let avg_dist = if neighbors.is_empty() {
                    1.0
                } else {
                    neighbors.iter().map(|(_, d)| d).sum::<f32>() / neighbors.len() as f32
                };

                // Anisotropy: compute from neighbor direction variance
                // Direction vectors from entity to each neighbor
                let anisotropy = Self::compute_anisotropy(embedding, entities, neighbors, dim);

                // Sparse direction: find the direction with least neighbor coverage
                let sparse_direction =
                    Self::find_sparse_direction(embedding, entities, neighbors, dim);

                // Local density: inverse of isolation (normalized later)
                let local_density = if avg_dist > 0.0 { 1.0 / avg_dist } else { 0.0 };

                // Build neighbor list with face prominence
                let neighbor_list: Vec<(String, String, f32)> = neighbors
                    .iter()
                    .map(|(j, dist)| {
                        // Face prominence: closer neighbors share a more prominent face
                        // (larger shared boundary in the Voronoi dual)
                        let prominence = if *dist > 0.0 { 1.0 / dist } else { 1.0 };
                        (
                            entities[*j].0.clone(),
                            entities[*j].1.clone(),
                            prominence,
                        )
                    })
                    .collect();

                VoronoiCell {
                    uuid: uuid.clone(),
                    name: name.clone(),
                    isolation: avg_dist,
                    anisotropy,
                    neighbor_count: neighbors.len(),
                    neighbors: neighbor_list,
                    sparse_direction,
                    local_density,
                }
            })
            .collect()
    }

    /// Compute anisotropy from neighbor direction vectors.
    ///
    /// Projects neighbor directions into a low-rank subspace and
    /// measures the variance ratio (max/min eigenvalue proxy).
    ///
    /// Anisotropy = 1.0 means neighbors are evenly distributed (spherical cell).
    /// Anisotropy > 1.0 means neighbors cluster in certain directions (elongated cell).
    fn compute_anisotropy(
        center: &[f32],
        entities: &[(String, String, Vec<f32>)],
        neighbors: &[(usize, f32)],
        _dim: usize,
    ) -> f32 {
        if neighbors.len() < 2 {
            return 1.0;
        }

        // Compute direction vectors from center to each neighbor
        let directions: Vec<Vec<f32>> = neighbors
            .iter()
            .map(|(j, _)| {
                let neighbor_emb = &entities[*j].2;
                let mut dir: Vec<f32> = center
                    .iter()
                    .zip(neighbor_emb.iter())
                    .map(|(a, b)| b - a)
                    .collect();
                // Normalize
                let norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut dir {
                        *x /= norm;
                    }
                }
                dir
            })
            .collect();

        // Compute the k×k Gram matrix (dot products between direction pairs)
        // This is much cheaper than the full d×d covariance matrix
        let k = directions.len();
        let mut gram = vec![0.0f32; k * k];
        for i in 0..k {
            for j in i..k {
                let dot: f32 = directions[i]
                    .iter()
                    .zip(directions[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                gram[i * k + j] = dot;
                gram[j * k + i] = dot;
            }
        }

        // Approximate anisotropy via off-diagonal analysis of the Gram matrix.
        // High max/avg ratio = neighbors cluster in similar directions (anisotropic).
        // Low ratio = neighbors evenly distributed (isotropic/spherical cell).
        // High ratio = neighbors cluster in similar directions (anisotropic)
        let mut off_diag_values = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                off_diag_values.push(gram[i * k + j].abs());
            }
        }

        if off_diag_values.is_empty() {
            return 1.0;
        }

        let avg_off_diag =
            off_diag_values.iter().sum::<f32>() / off_diag_values.len() as f32;
        let max_off_diag = off_diag_values
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));

        // If max off-diagonal is much larger than average, neighbors cluster in one direction
        if avg_off_diag > 0.001 {
            (max_off_diag / avg_off_diag).max(1.0)
        } else {
            1.0
        }
    }

    /// Find the direction with least neighbor coverage (the "sparse direction").
    ///
    /// This is the direction in embedding space where the entity has the
    /// fewest nearby neighbors — the direction where knowledge thins out.
    fn find_sparse_direction(
        center: &[f32],
        entities: &[(String, String, Vec<f32>)],
        neighbors: &[(usize, f32)],
        dim: usize,
    ) -> Option<Vec<f32>> {
        if neighbors.len() < 3 || dim < 2 {
            return None;
        }

        // Compute the mean direction to neighbors
        let mut mean_dir = vec![0.0f32; dim];
        for (j, _) in neighbors {
            let neighbor_emb = &entities[*j].2;
            for (d, (m, n)) in mean_dir
                .iter_mut()
                .zip(center.iter().zip(neighbor_emb.iter()))
            {
                *d += n - m;
            }
        }

        // Normalize mean direction
        let norm: f32 = mean_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-6 {
            return None;
        }
        for x in &mut mean_dir {
            *x /= norm;
        }

        // The sparse direction is approximately the OPPOSITE of the mean neighbor direction.
        // If neighbors cluster in one direction, the opposite side is sparse.
        let sparse: Vec<f32> = mean_dir.iter().map(|x| -x).collect();
        Some(sparse)
    }

    /// Detect voids: regions of embedding space surrounded by entities but empty.
    ///
    /// Strategy: find groups of isolated entities that are far from each other
    /// but all border the same empty region. The centroid of such a group
    /// is the center of the void.
    fn detect_voids(
        entities: &[(String, String, Vec<f32>)],
        cells: &[VoronoiCell],
        config: &VoronoiConfig,
    ) -> Vec<VoronoiVoid> {
        if entities.len() < config.min_void_boundary {
            return Vec::new();
        }

        let dim = entities[0].2.len();
        let mut voids = Vec::new();

        // Strategy 1: Centroid gaps between clusters of isolated entities
        // Find highly isolated entities
        let isolated: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.isolation > config.void_isolation_threshold)
            .map(|(i, _)| i)
            .collect();

        if isolated.len() >= config.min_void_boundary {
            // Check if isolated entities are spread around a central void
            // Compute their centroid
            let mut centroid = vec![0.0f32; dim];
            for &idx in &isolated {
                for (d, c) in centroid.iter_mut().zip(entities[idx].2.iter()) {
                    *d += c;
                }
            }
            for c in &mut centroid {
                *c /= isolated.len() as f32;
            }

            // Check: is the centroid far from all entities? (i.e., is it a void?)
            let min_dist_to_centroid = entities
                .iter()
                .map(|(_, _, emb)| 1.0 - cosine_similarity(&centroid, emb))
                .fold(f32::MAX, f32::min);

            if min_dist_to_centroid > 0.3 {
                let boundary: Vec<(String, String)> = isolated
                    .iter()
                    .take(10)
                    .map(|&i| (entities[i].0.clone(), entities[i].1.clone()))
                    .collect();

                let avg_radius = isolated
                    .iter()
                    .map(|&i| 1.0 - cosine_similarity(&centroid, &entities[i].2))
                    .sum::<f32>()
                    / isolated.len() as f32;

                let confidence = (isolated.len() as f32 / 10.0).clamp(0.2, 0.9)
                    * (min_dist_to_centroid / 0.5).clamp(0.3, 1.0);

                voids.push(VoronoiVoid {
                    centroid: centroid.clone(),
                    boundary_entities: boundary,
                    radius: avg_radius,
                    boundary_count: isolated.len(),
                    confidence,
                });
            }
        }

        // Strategy 2: Midpoint voids between distant Voronoi neighbors
        // For each entity, check if the midpoint between it and a distant neighbor
        // is far from all other entities
        for cell in cells {
            if cell.neighbors.len() < 2 {
                continue;
            }

            // Check the most distant neighbor (weakest Voronoi face)
            let weakest_neighbor = cell
                .neighbors
                .iter()
                .min_by(|a, b| {
                    a.2.partial_cmp(&b.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some((neighbor_uuid, neighbor_name, prominence)) = weakest_neighbor {
                if *prominence > 0.5 {
                    continue; // not weak enough to indicate a void
                }

                // Find the neighbor's embedding
                let neighbor_emb = entities
                    .iter()
                    .find(|(uuid, _, _)| uuid == neighbor_uuid);

                if let Some((_, _, n_emb)) = neighbor_emb {
                    let cell_emb = entities
                        .iter()
                        .find(|(uuid, _, _)| *uuid == cell.uuid)
                        .map(|(_, _, e)| e);

                    if let Some(c_emb) = cell_emb {
                        // Compute midpoint
                        let midpoint: Vec<f32> = c_emb
                            .iter()
                            .zip(n_emb.iter())
                            .map(|(a, b)| (a + b) / 2.0)
                            .collect();

                        // Check if midpoint is far from all entities
                        let min_dist = entities
                            .iter()
                            .map(|(_, _, emb)| 1.0 - cosine_similarity(&midpoint, emb))
                            .fold(f32::MAX, f32::min);

                        if min_dist > 0.25 {
                            let boundary = vec![
                                (cell.uuid.clone(), cell.name.clone()),
                                (neighbor_uuid.clone(), neighbor_name.clone()),
                            ];
                            voids.push(VoronoiVoid {
                                centroid: midpoint,
                                boundary_entities: boundary,
                                radius: min_dist,
                                boundary_count: 2,
                                confidence: (min_dist / 0.5).clamp(0.1, 0.7),
                            });
                        }
                    }
                }
            }
        }

        // Sort by confidence, limit
        voids.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        voids.truncate(config.max_voids);
        voids
    }

}
