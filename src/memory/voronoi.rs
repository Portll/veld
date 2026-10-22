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
//!
//! A store's analysis is kept on the store and reused until its graph mirror is
//! rebuilt, the only way the analysed rows change.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use super::slow_store::SlowStore;
use crate::vector_db::distance_inline::dot_product_inline;
use anyhow::Result;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Seeds the entity sample, fixed so that repeated analyses of one graph
/// sample the same entities.
const SAMPLE_SEED: u64 = 0x766f_726f_6e6f_6921;

/// Directions shorter than this (a neighbour at the entity's own position) have
/// no orientation and are left out of the anisotropy.
const MIN_DIRECTION_NORM: f32 = 1e-6;

/// Power iteration stops once the eigenvalue estimate moves by less than this,
/// relative to itself, or after [`POWER_ITERATION_LIMIT`] steps. The estimate
/// never passes the top eigenvalue, and each step shrinks the weight on an
/// eigenvalue `λ` against the top one `λ₁` by `(λ/λ₁)²`, so an estimate still
/// moving at the limit rests on eigenvalues close to `λ₁` and is close to it.
const POWER_ITERATION_TOLERANCE: f64 = 1e-6;
const POWER_ITERATION_LIMIT: usize = 100;

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
    /// Anisotropy: the largest eigenvalue of the second-moment matrix of the unit
    /// directions from this entity to its neighbours, over that matrix's mean
    /// eigenvalue across those directions. For `m` neighbours it lies in `[1, m]`.
    /// 1.0 = spherical cell (the directions are mutually orthogonal, knowledge
    /// evenly distributed around the entity).
    /// Higher = elongated cell (the directions share an axis, knowledge
    /// concentrated in certain directions and sparse in others); `m` when all
    /// neighbours lie along one line.
    pub anisotropy: f32,
    /// Number of Voronoi neighbors (Delaunay adjacency)
    pub neighbor_count: usize,
    /// Voronoi neighbors, nearest first: (index into [`VoronoiAnalysis::cells`],
    /// face_prominence). Face prominence: how large the shared boundary is
    /// (higher = more prominent boundary).
    pub neighbors: Vec<(usize, f32)>,
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
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiConfig {
    /// Number of nearest neighbors to consider per entity
    pub k: usize,
    /// Minimum isolation score to flag as a potential void boundary
    pub void_isolation_threshold: f32,
    /// Minimum number of entities that must border a void for it to be significant
    pub min_void_boundary: usize,
    /// Maximum number of voids to return
    pub max_voids: usize,
    /// Maximum entities to analyse (bounds the O(n²) k-NN pass). Larger stores
    /// are sampled uniformly, the same entities each time for the same graph.
    pub max_entities: usize,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            k: 8,
            void_isolation_threshold: 1.3,
            min_void_boundary: 3,
            max_voids: 20,
            max_entities: 2000,
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

/// A store's last analysis, held by the store (see [`VoronoiAnalyzer::analyze`]).
#[derive(Default)]
pub struct VoronoiCache(Mutex<Option<CachedAnalysis>>);

struct CachedAnalysis {
    /// The store's last mirror sync when the rows were read.
    synced_at: Instant,
    config: VoronoiConfig,
    analysis: Arc<VoronoiAnalysis>,
}

/// Unit-length embeddings in one row-major buffer, so cosine similarity is one
/// dot product: row `i` is `data[i * dim..(i + 1) * dim]`. A zero or non-finite
/// embedding becomes a zero row, similar to nothing, as `cosine_similarity`
/// scores it.
struct UnitRows {
    data: Vec<f32>,
    dim: usize,
}

impl UnitRows {
    fn new<'a>(embeddings: impl ExactSizeIterator<Item = &'a [f32]>, dim: usize) -> Self {
        let mut data = Vec::with_capacity(embeddings.len() * dim);
        for embedding in embeddings {
            debug_assert_eq!(embedding.len(), dim, "rows share one dimension");
            let start = data.len();
            data.extend_from_slice(embedding);
            let row = &mut data[start..];
            let norm = dot_product_inline(row, row).sqrt();
            if norm.is_finite() && norm > 0.0 {
                row.iter_mut().for_each(|x| *x /= norm);
            } else {
                row.fill(0.0);
            }
        }
        Self { data, dim }
    }

    fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Cosine distance from `point` to unit row `row`, `point_norm` being `point`'s
/// length; a zero `point` is similar to nothing, as in `cosine_similarity`.
fn distance_to(point: &[f32], point_norm: f32, row: &[f32]) -> f32 {
    if point_norm > 0.0 {
        1.0 - dot_product_inline(point, row) / point_norm
    } else {
        1.0
    }
}

/// Voronoi analyzer: computes approximate Voronoi cell properties
/// and detects voids and Planet X candidates in knowledge space.
pub struct VoronoiAnalyzer;

impl VoronoiAnalyzer {
    /// Run full Voronoi analysis on the knowledge graph.
    ///
    /// The result is kept on the store and returned again while the store's
    /// graph mirror is unchanged ([`SlowStore::last_sync`]) and the config is
    /// the same; a store that has never synced is analysed afresh every time.
    pub fn analyze(store: &SlowStore, config: &VoronoiConfig) -> Result<Arc<VoronoiAnalysis>> {
        let start = Instant::now();
        // Held through the run, so concurrent requests for one store wait for
        // one analysis rather than each repeating it.
        let mut cached = store.voronoi_cache.0.lock();
        // Read before the rows. A rebuild stamps the time again after swapping
        // the new mirror in, so the rows read below are at least as new as this
        // stamp, and an analysis is never kept under a stamp its rows predate.
        let synced_at = store.last_sync();
        if let (Some(at), Some(hit)) = (synced_at, cached.as_ref()) {
            if hit.synced_at == at && hit.config == *config {
                return Ok(Arc::clone(&hit.analysis));
            }
        }

        let entities = store.sample_embeddings(config.max_entities, SAMPLE_SEED)?;
        let analysis = Arc::new(Self::compute(entities, config, start));
        *cached = synced_at.map(|synced_at| CachedAnalysis {
            synced_at,
            config: config.clone(),
            analysis: Arc::clone(&analysis),
        });
        Ok(analysis)
    }

    /// Analyse `entities` as `(uuid, name, embedding)`. Entities whose
    /// embedding length differs from the most common one are left out: no
    /// distance relates vectors from different embedding spaces.
    fn compute(
        entities: Vec<(String, String, Vec<f32>)>,
        config: &VoronoiConfig,
        start: Instant,
    ) -> VoronoiAnalysis {
        let mut dims: HashMap<usize, usize> = HashMap::new();
        for (_, _, embedding) in &entities {
            *dims.entry(embedding.len()).or_default() += 1;
        }
        let dim = dims
            .into_iter()
            .max_by_key(|&(dim, count)| (count, dim))
            .map_or(0, |(dim, _)| dim);
        let total = entities.len();
        let (ids, embeddings): (Vec<(String, String)>, Vec<Vec<f32>>) = entities
            .into_iter()
            .filter(|(_, _, embedding)| embedding.len() == dim)
            .map(|(uuid, name, embedding)| ((uuid, name), embedding))
            .unzip();
        if ids.len() < total {
            tracing::warn!(
                kept = ids.len(),
                dropped = total - ids.len(),
                dim,
                "voronoi: entities with another embedding length left out"
            );
        }
        let rows = UnitRows::new(embeddings.iter().map(Vec::as_slice), dim);
        drop(embeddings);

        if rows.len() < 3 {
            return VoronoiAnalysis {
                cells: Vec::new(),
                voids: Vec::new(),
                stats: VoronoiStats {
                    entity_count: rows.len(),
                    avg_isolation: 0.0,
                    max_isolation: 0.0,
                    avg_anisotropy: 1.0,
                    isolated_count: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                },
            };
        }

        // Phase 1: Compute k-NN and cell properties
        let (knn, cells) = {
            let _span = tracing::info_span!("voronoi.knn", entities = rows.len()).entered();
            let knn = Self::compute_knn(&rows, config.k);
            let cells = Self::compute_cells(&rows, ids, &knn);
            (knn, cells)
        };

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
        let voids = {
            let _span = tracing::info_span!("voronoi.voids").entered();
            Self::detect_voids(&rows, &cells, &knn, config)
        };

        // Compute stats
        let max_isolation = cells.iter().map(|c| c.isolation).fold(0.0f32, f32::max);
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
            entity_count: rows.len(),
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

        VoronoiAnalysis {
            cells,
            voids,
            stats,
        }
    }

    /// The `k` nearest rows to each row by cosine distance, nearest first, ties
    /// to the lower index: what sorting each row's distances to all others and
    /// keeping `k` would give.
    ///
    /// Cosine distance is symmetric, so each pair is scored once, in the upper
    /// triangle, and offered to both rows' bounded heaps: half the dot products of
    /// scoring every ordered pair, and `O(log k)` per offer instead of a full sort
    /// per row.
    ///
    /// Returns: row index → [(neighbor_index, cosine_distance)]
    fn compute_knn(rows: &UnitRows, k: usize) -> Vec<Vec<(usize, f32)>> {
        let n = rows.len();
        let k = k.min(n.saturating_sub(1));
        if k == 0 {
            return vec![Vec::new(); n];
        }

        // Each heap holds its row's best `k` so far, the worst of them on top.
        let mut heaps: Vec<BinaryHeap<(OrderedFloat<f32>, usize)>> =
            (0..n).map(|_| BinaryHeap::with_capacity(k)).collect();
        for i in 0..n {
            let row_i = rows.row(i);
            for j in (i + 1)..n {
                let distance = OrderedFloat(1.0 - dot_product_inline(row_i, rows.row(j)));
                offer(&mut heaps[i], k, (distance, j));
                offer(&mut heaps[j], k, (distance, i));
            }
        }

        heaps
            .into_iter()
            .map(|heap| {
                heap.into_sorted_vec()
                    .into_iter()
                    .map(|(distance, j)| (j, distance.0))
                    .collect()
            })
            .collect()
    }

    /// Compute Voronoi cell properties from k-NN data.
    fn compute_cells(
        rows: &UnitRows,
        ids: Vec<(String, String)>,
        knn: &[Vec<(usize, f32)>],
    ) -> Vec<VoronoiCell> {
        let mut directions = Vec::new();
        ids.into_iter()
            .enumerate()
            .map(|(i, (uuid, name))| {
                let neighbors = &knn[i];

                // Isolation: average distance to k nearest neighbors
                let avg_dist = if neighbors.is_empty() {
                    1.0
                } else {
                    neighbors.iter().map(|(_, d)| d).sum::<f32>() / neighbors.len() as f32
                };

                let anisotropy = Self::compute_anisotropy(rows, i, neighbors, &mut directions);

                // Sparse direction: find the direction with least neighbor coverage
                let sparse_direction = Self::find_sparse_direction(rows, i, neighbors);

                // Local density: inverse of isolation (normalized later)
                let local_density = if avg_dist > 0.0 { 1.0 / avg_dist } else { 0.0 };

                // Face prominence: closer neighbors share a more prominent face
                // (larger shared boundary in the Voronoi dual)
                let neighbor_list: Vec<(usize, f32)> = neighbors
                    .iter()
                    .map(|&(j, dist)| (j, if dist > 0.0 { 1.0 / dist } else { 1.0 }))
                    .collect();

                VoronoiCell {
                    uuid,
                    name,
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

    /// Anisotropy of row `center`'s cell: see [`VoronoiCell::anisotropy`].
    ///
    /// For unit directions `u_1..u_m`, the second-moment matrix `Σ u_i u_iᵀ` has
    /// trace `m`, so its mean eigenvalue across the `m` directions is 1 and the
    /// ratio is its largest eigenvalue. That equals the largest eigenvalue of the
    /// `m × m` Gram matrix `[u_i · u_j]`, which power iteration finds without
    /// forming a `dim × dim` matrix. A max/min ratio would be undefined here:
    /// with fewer neighbours than dimensions the smallest eigenvalue is 0.
    ///
    /// `directions` is scratch space, reused across calls.
    fn compute_anisotropy(
        rows: &UnitRows,
        center: usize,
        neighbors: &[(usize, f32)],
        directions: &mut Vec<f32>,
    ) -> f32 {
        let dim = rows.dim;
        let origin = rows.row(center);
        directions.clear();
        for &(j, _) in neighbors {
            let start = directions.len();
            directions.extend(rows.row(j).iter().zip(origin).map(|(n, c)| n - c));
            let direction = &mut directions[start..];
            let norm = dot_product_inline(direction, direction).sqrt();
            if norm > MIN_DIRECTION_NORM {
                direction.iter_mut().for_each(|x| *x /= norm);
            } else {
                directions.truncate(start);
            }
        }
        let m = directions.len().checked_div(dim).unwrap_or(0);
        if m < 2 {
            return 1.0;
        }

        let direction = |a: usize| &directions[a * dim..(a + 1) * dim];
        let mut gram = vec![0.0f64; m * m];
        for a in 0..m {
            gram[a * m + a] = 1.0;
            for b in (a + 1)..m {
                let dot = f64::from(dot_product_inline(direction(a), direction(b)));
                gram[a * m + b] = dot;
                gram[b * m + a] = dot;
            }
        }
        (largest_eigenvalue(&gram, m) as f32).clamp(1.0, m as f32)
    }

    /// Find the direction with least neighbor coverage (the "sparse direction").
    ///
    /// This is the direction in embedding space where the entity has the
    /// fewest nearby neighbors — the direction where knowledge thins out.
    fn find_sparse_direction(
        rows: &UnitRows,
        center: usize,
        neighbors: &[(usize, f32)],
    ) -> Option<Vec<f32>> {
        if neighbors.len() < 3 || rows.dim < 2 {
            return None;
        }

        // Compute the mean direction to neighbors
        let origin = rows.row(center);
        let mut mean_dir = vec![0.0f32; rows.dim];
        for &(j, _) in neighbors {
            for (d, (c, n)) in mean_dir.iter_mut().zip(origin.iter().zip(rows.row(j))) {
                *d += n - c;
            }
        }

        // Normalize mean direction
        let norm = dot_product_inline(&mean_dir, &mean_dir).sqrt();
        if norm < 1e-6 {
            return None;
        }

        // The sparse direction is approximately the OPPOSITE of the mean neighbor direction.
        // If neighbors cluster in one direction, the opposite side is sparse.
        mean_dir.iter_mut().for_each(|x| *x /= -norm);
        Some(mean_dir)
    }

    /// The midpoint of row `a` and its farthest neighbour (the last of
    /// `neighbors_of_a`), and the cosine distance from it to the nearest row.
    ///
    /// The nearest row is always `a` or one of its neighbours, so only those are
    /// scored. On the unit sphere, with `θ` the angle from `a` to its farthest
    /// neighbour `b`, the midpoint lies `θ/2` from `a`; every row outside the
    /// neighbour list is at least `θ` from `a`, so by the triangle inequality at
    /// least `θ - θ/2 = θ/2` from the midpoint, never nearer than `a` itself.
    fn midpoint_clearance(
        rows: &UnitRows,
        a: usize,
        neighbors_of_a: &[(usize, f32)],
    ) -> Option<(Vec<f32>, f32)> {
        let &(b, _) = neighbors_of_a.last()?;
        let midpoint: Vec<f32> = rows
            .row(a)
            .iter()
            .zip(rows.row(b))
            .map(|(x, y)| (x + y) / 2.0)
            .collect();
        let norm = dot_product_inline(&midpoint, &midpoint).sqrt();
        let nearest = std::iter::once(a)
            .chain(neighbors_of_a.iter().map(|&(j, _)| j))
            .map(|j| distance_to(&midpoint, norm, rows.row(j)))
            .fold(f32::MAX, f32::min);
        Some((midpoint, nearest))
    }

    /// Detect voids: regions of embedding space surrounded by entities but empty.
    ///
    /// Strategy: find groups of isolated entities that are far from each other
    /// but all border the same empty region. The centroid of such a group
    /// is the center of the void.
    fn detect_voids(
        rows: &UnitRows,
        cells: &[VoronoiCell],
        knn: &[Vec<(usize, f32)>],
        config: &VoronoiConfig,
    ) -> Vec<VoronoiVoid> {
        let n = rows.len();
        if n < config.min_void_boundary {
            return Vec::new();
        }

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
            let mut centroid = vec![0.0f32; rows.dim];
            for &idx in &isolated {
                for (d, c) in centroid.iter_mut().zip(rows.row(idx)) {
                    *d += c;
                }
            }
            for c in &mut centroid {
                *c /= isolated.len() as f32;
            }
            let centroid_norm = dot_product_inline(&centroid, &centroid).sqrt();

            // Check: is the centroid far from all entities? (i.e., is it a void?)
            let min_dist_to_centroid = (0..n)
                .map(|j| distance_to(&centroid, centroid_norm, rows.row(j)))
                .fold(f32::MAX, f32::min);

            if min_dist_to_centroid > 0.3 {
                let boundary: Vec<(String, String)> = isolated
                    .iter()
                    .take(10)
                    .map(|&i| (cells[i].uuid.clone(), cells[i].name.clone()))
                    .collect();

                let avg_radius = isolated
                    .iter()
                    .map(|&i| distance_to(&centroid, centroid_norm, rows.row(i)))
                    .sum::<f32>()
                    / isolated.len() as f32;

                let confidence = (isolated.len() as f32 / 10.0).clamp(0.2, 0.9)
                    * (min_dist_to_centroid / 0.5).clamp(0.3, 1.0);

                voids.push(VoronoiVoid {
                    centroid,
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
        for (i, cell) in cells.iter().enumerate() {
            if cell.neighbors.len() < 2 {
                continue;
            }

            // Check the most distant neighbor (weakest Voronoi face): the last,
            // as neighbors are nearest first.
            let Some(&(neighbor, prominence)) = cell.neighbors.last() else {
                continue;
            };
            if prominence > 0.5 {
                continue; // not weak enough to indicate a void
            }

            let Some((midpoint, min_dist)) = Self::midpoint_clearance(rows, i, &knn[i]) else {
                continue;
            };
            if min_dist > 0.25 {
                let boundary = vec![
                    (cell.uuid.clone(), cell.name.clone()),
                    (cells[neighbor].uuid.clone(), cells[neighbor].name.clone()),
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

/// Keep `candidate` in `heap` if it is among the `k` smallest seen.
fn offer(
    heap: &mut BinaryHeap<(OrderedFloat<f32>, usize)>,
    k: usize,
    candidate: (OrderedFloat<f32>, usize),
) {
    if heap.len() < k {
        heap.push(candidate);
    } else if let Some(mut worst) = heap.peek_mut() {
        if candidate < *worst {
            *worst = candidate;
        }
    }
}

/// The largest eigenvalue of the symmetric positive semi-definite `m × m`
/// matrix `matrix` (row-major), by power iteration.
///
/// The start vector is uneven on purpose: a uniform one is orthogonal to the
/// top eigenvector of, for instance, two opposite directions (`[[1, -1], [-1, 1]]`,
/// top eigenvector `[1, -1]`), and would converge to the wrong eigenvalue.
fn largest_eigenvalue(matrix: &[f64], m: usize) -> f64 {
    let mut vector: Vec<f64> = (0..m).map(|i| 1.0 + i as f64 / m as f64).collect();
    let mut next = vec![0.0f64; m];
    let mut estimate = 0.0f64;
    for _ in 0..POWER_ITERATION_LIMIT {
        let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        vector.iter_mut().for_each(|x| *x /= norm);
        for (row, out) in matrix.chunks_exact(m).zip(next.iter_mut()) {
            *out = row.iter().zip(&vector).map(|(a, b)| a * b).sum();
        }
        // Rayleigh quotient of the unit vector: approaches the top eigenvalue from below.
        let rayleigh: f64 = next.iter().zip(&vector).map(|(a, b)| a * b).sum();
        let settled = (rayleigh - estimate).abs() <= POWER_ITERATION_TOLERANCE * rayleigh.abs();
        estimate = rayleigh;
        std::mem::swap(&mut vector, &mut next);
        if settled {
            break;
        }
    }
    estimate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::cosine_similarity;

    fn rows_of(embeddings: &[Vec<f32>]) -> UnitRows {
        let dim = embeddings.first().map_or(0, Vec::len);
        UnitRows::new(embeddings.iter().map(Vec::as_slice), dim)
    }

    /// `n` vectors of `dim` components in [-1, 1), from a fixed xorshift seed.
    fn seeded(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    /// The reference k-NN: each row's distances to every other row, sorted by
    /// (distance, index), first `k`. Each pair is scored lower index first, as
    /// the upper-triangle pass scores it, so the distances match to the bit.
    fn brute_force_knn(rows: &UnitRows, k: usize) -> Vec<Vec<(usize, f32)>> {
        (0..rows.len())
            .map(|i| {
                let mut all: Vec<(OrderedFloat<f32>, usize)> = (0..rows.len())
                    .filter(|&j| j != i)
                    .map(|j| {
                        let (lo, hi) = (i.min(j), i.max(j));
                        let similarity = dot_product_inline(rows.row(lo), rows.row(hi));
                        (OrderedFloat(1.0 - similarity), j)
                    })
                    .collect();
                all.sort();
                all.truncate(k);
                all.into_iter().map(|(d, j)| (j, d.0)).collect()
            })
            .collect()
    }

    /// Unit vector at angle `phi` from `e0` towards the axis `axis` (`axis > 0`).
    fn towards(dim: usize, axis: usize, phi: f32) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[0] = phi.cos();
        v[axis] = phi.sin();
        v
    }

    #[test]
    fn test_compute_knn_ordering() {
        let rows = rows_of(&[
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
        ]);
        let knn = VoronoiAnalyzer::compute_knn(&rows, 2);
        assert_eq!(knn.len(), 3);
        // A's nearest neighbor should be B (most similar)
        assert_eq!(knn[0][0].0, 1);
    }

    #[test]
    fn test_compute_knn_single_entity() {
        let rows = rows_of(&[vec![1.0, 0.0]]);
        let knn = VoronoiAnalyzer::compute_knn(&rows, 5);
        assert_eq!(knn.len(), 1);
        assert!(knn[0].is_empty());
    }

    #[test]
    fn test_compute_knn_empty() {
        let rows = rows_of(&[]);
        let knn = VoronoiAnalyzer::compute_knn(&rows, 5);
        assert!(knn.is_empty());
    }

    #[test]
    fn knn_matches_brute_force() {
        let mut with_duplicates = seeded(20, 8, 3);
        // Exact copies tie on distance, so the index tie-break is exercised too.
        with_duplicates.extend(with_duplicates[..10].to_vec());
        let sets = [
            seeded(60, 24, 1),
            seeded(41, 384, 7),
            seeded(9, 3, 13),
            with_duplicates,
        ];
        for (set, embeddings) in sets.iter().enumerate() {
            let rows = rows_of(embeddings);
            let n = rows.len();
            for k in [1, 3, 8, n - 1, n + 5] {
                let knn = VoronoiAnalyzer::compute_knn(&rows, k);
                assert_eq!(knn, brute_force_knn(&rows, k.min(n - 1)), "set {set} k {k}");
                for (i, neighbors) in knn.iter().enumerate() {
                    for &(j, distance) in neighbors {
                        let expected = 1.0 - cosine_similarity(&embeddings[i], &embeddings[j]);
                        assert!(
                            (distance - expected).abs() < 1e-5,
                            "set {set} k {k} pair {i},{j}: {distance} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn midpoint_clearance_matches_brute_force() {
        let rows = rows_of(&seeded(80, 16, 5));
        for k in [2, 5, 8] {
            let knn = VoronoiAnalyzer::compute_knn(&rows, k);
            for (a, neighbors) in knn.iter().enumerate() {
                let (midpoint, nearest) =
                    VoronoiAnalyzer::midpoint_clearance(&rows, a, neighbors).unwrap();
                let norm = dot_product_inline(&midpoint, &midpoint).sqrt();
                let everywhere = (0..rows.len())
                    .map(|j| distance_to(&midpoint, norm, rows.row(j)))
                    .fold(f32::MAX, f32::min);
                assert!(
                    (nearest - everywhere).abs() < 1e-5,
                    "k {k} row {a}: {nearest} vs {everywhere}"
                );
            }
        }
    }

    #[test]
    fn test_anisotropy_single_neighbor() {
        let rows = rows_of(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let a = VoronoiAnalyzer::compute_anisotropy(&rows, 0, &[(1, 1.0)], &mut Vec::new());
        assert_eq!(a, 1.0);
    }

    #[test]
    fn anisotropy_is_one_for_orthogonal_directions() {
        // Four neighbours a small angle from e0, each towards its own axis: the
        // directions to them are orthogonal up to (1 - cos 0.01) / 2.
        let mut embeddings = vec![towards(6, 1, 0.0)];
        embeddings.extend((1..=4).map(|axis| towards(6, axis, 0.01)));
        let rows = rows_of(&embeddings);
        let neighbors: Vec<(usize, f32)> = (1..=4).map(|j| (j, 0.0)).collect();
        let a = VoronoiAnalyzer::compute_anisotropy(&rows, 0, &neighbors, &mut Vec::new());
        assert!((a - 1.0).abs() < 1e-3, "{a}");
    }

    #[test]
    fn anisotropy_counts_neighbours_along_one_line() {
        // Opposite sides of the entity on one great circle: an axis, not a ball.
        // The Gram matrix is [[1, -cos 0.1], [-cos 0.1, 1]], top eigenvalue
        // 1 + cos 0.1, with top eigenvector [1, -1], orthogonal to a uniform start.
        let rows = rows_of(&[towards(3, 1, 0.0), towards(3, 1, 0.1), towards(3, 1, -0.1)]);
        let a =
            VoronoiAnalyzer::compute_anisotropy(&rows, 0, &[(1, 0.0), (2, 0.0)], &mut Vec::new());
        assert!((a - (1.0 + 0.1f32.cos())).abs() < 1e-4, "{a}");

        // Three neighbours on the same side, along one line: close to m = 3.
        let rows = rows_of(&[
            towards(3, 1, 0.0),
            towards(3, 1, 0.1),
            towards(3, 1, 0.2),
            towards(3, 1, 0.3),
        ]);
        let neighbors = [(1, 0.0), (2, 0.0), (3, 0.0)];
        let a = VoronoiAnalyzer::compute_anisotropy(&rows, 0, &neighbors, &mut Vec::new());
        assert!(a > 2.9 && a <= 3.0, "{a}");
    }

    /// Randomised invariant: kNN output length always ≤ k, distances non-negative.
    #[test]
    fn test_knn_invariants_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let embeddings: Vec<Vec<f32>> = (0..15)
            .map(|_| (0..16).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let rows = rows_of(&embeddings);

        for k in [1, 3, 5, 20] {
            let knn = VoronoiAnalyzer::compute_knn(&rows, k);
            assert_eq!(knn.len(), embeddings.len());
            for (i, neighbors) in knn.iter().enumerate() {
                assert!(neighbors.len() <= k.min(embeddings.len() - 1));
                for &(j, dist) in neighbors {
                    assert_ne!(i, j, "Entity should not be its own neighbor");
                    assert!(dist >= -1e-6, "Negative distance");
                }
            }
        }
    }

    #[test]
    fn compute_leaves_out_other_embedding_lengths() {
        let mut entities: Vec<(String, String, Vec<f32>)> = seeded(12, 8, 11)
            .into_iter()
            .enumerate()
            .map(|(i, e)| (format!("u{i:02}"), format!("E{i}"), e))
            .collect();
        entities.push(("u99".into(), "Other space".into(), vec![1.0, 0.0, 0.0]));
        let analysis =
            VoronoiAnalyzer::compute(entities, &VoronoiConfig::default(), Instant::now());
        assert_eq!(analysis.stats.entity_count, 12);
        assert!(analysis.cells.iter().all(|c| c.uuid != "u99"));
        for cell in &analysis.cells {
            assert_eq!(cell.neighbor_count, 8);
            assert!(cell
                .neighbors
                .iter()
                .all(|&(j, _)| j < analysis.cells.len()));
        }
    }
}
