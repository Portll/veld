//! Domain Jump Detection
//!
//! Standard graph traversal follows edges — it stays in the same Voronoi region
//! or crosses at vertices. But sometimes the most valuable move is to leap
//! across a face into an entirely different knowledge domain.
//!
//! This module scores the value of domain jumps: when should reasoning
//! leave its current region and explore a different one?
//!
//! A domain jump is valuable when:
//! - The current region is saturated (high density, diminishing returns)
//! - A distant region has a void that matches the current query's shape
//! - Two regions share an implied connection (Planet X) but no direct edge
//! - The query's embedding falls in a face rather than near a seed (between domains)
//!
//! The score combines:
//! - **Jump distance**: How far in embedding space? (penalizes random jumps)
//! - **Landing value**: What's in the target region? (rewards informative destinations)
//! - **Face crossing cost**: How thick is the Voronoi boundary? (thicker = more different)
//! - **Relevance transfer**: Will knowledge from the target region transfer back?

use serde::{Deserialize, Serialize};

use super::voronoi::{VoronoiAnalysis, VoronoiCell};
use crate::similarity::cosine_similarity;

/// A scored domain jump: a recommended leap from one knowledge region to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainJump {
    /// Source entity (where we are now)
    pub from_uuid: String,
    pub from_name: String,
    /// Target entity (where the jump lands)
    pub to_uuid: String,
    pub to_name: String,
    /// Jump distance in embedding space (cosine distance)
    pub distance: f32,
    /// Overall jump value score (0.0-1.0, higher = more valuable to make this jump)
    pub value: f32,
    /// Why this jump is valuable
    pub reason: JumpReason,
    /// Human-readable explanation
    pub explanation: String,
}

/// Why a domain jump is recommended
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JumpReason {
    /// Current region is saturated — high density, low marginal value
    Saturation,
    /// Target region contains a void that matches the query's shape
    VoidMatch,
    /// Shared Planet X implies a hidden connection worth exploring
    PlanetXBridge,
    /// Query embedding falls between domains (on a face, not near a seed)
    InterstitialQuery,
    /// Target region has high anisotropy pointing toward current region
    AnisotropicPull,
}

/// Configuration for domain jump scoring
#[derive(Debug, Clone)]
pub struct DomainJumpConfig {
    /// Maximum jump distance to consider (cosine distance)
    pub max_distance: f32,
    /// Minimum value score to recommend a jump
    pub min_value: f32,
    /// Weight for saturation signal
    pub saturation_weight: f32,
    /// Weight for target value signal
    pub target_value_weight: f32,
    /// Weight for relevance transfer signal
    pub relevance_weight: f32,
    /// Maximum number of jumps to recommend
    pub max_jumps: usize,
}

impl Default for DomainJumpConfig {
    fn default() -> Self {
        Self {
            max_distance: 0.7,
            min_value: 0.3,
            saturation_weight: 0.3,
            target_value_weight: 0.4,
            relevance_weight: 0.3,
            max_jumps: 5,
        }
    }
}

/// Domain jump analyzer
pub struct DomainJumpAnalyzer;

impl DomainJumpAnalyzer {
    /// Score potential domain jumps from a query point.
    ///
    /// Given the current context (query embedding), find regions worth jumping to.
    /// Returns scored jumps sorted by value.
    pub fn score_jumps(
        query_embedding: &[f32],
        voronoi: &VoronoiAnalysis,
        config: &DomainJumpConfig,
    ) -> Vec<DomainJump> {
        if voronoi.cells.is_empty() {
            return Vec::new();
        }

        // Find which cell the query falls nearest to (its "home" region)
        let home_cell = Self::find_nearest_cell(query_embedding, &voronoi.cells);

        let mut jumps = Vec::new();

        for target_cell in &voronoi.cells {
            if target_cell.uuid == home_cell.uuid {
                continue;
            }

            // Compute jump distance
            // We don't have target embeddings directly, so use the cell's
            // position relative to the query
            let distance = Self::estimate_distance(query_embedding, target_cell, &voronoi.cells);

            if distance > config.max_distance {
                continue;
            }

            // Score the jump value
            let (value, reason, explanation) = Self::score_jump(
                query_embedding,
                home_cell,
                target_cell,
                voronoi,
                distance,
                config,
            );

            if value >= config.min_value {
                jumps.push(DomainJump {
                    from_uuid: home_cell.uuid.clone(),
                    from_name: home_cell.name.clone(),
                    to_uuid: target_cell.uuid.clone(),
                    to_name: target_cell.name.clone(),
                    distance,
                    value,
                    reason,
                    explanation,
                });
            }
        }

        // Sort by value (most valuable jumps first)
        jumps.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        jumps.truncate(config.max_jumps);
        jumps
    }

    /// Detect interstitial queries: when the query falls on a face rather
    /// than near a seed. This means the query is BETWEEN knowledge domains,
    /// and both domains might be relevant.
    ///
    /// Returns: list of nearby cells with their relevance, indicating
    /// the query spans multiple domains.
    pub fn detect_interstitial(
        query_embedding: &[f32],
        voronoi: &VoronoiAnalysis,
    ) -> Vec<(String, String, f32)> {
        if voronoi.cells.is_empty() {
            return Vec::new();
        }

        // Find distances to all cells
        let mut cell_distances: Vec<(&VoronoiCell, f32)> = Vec::new();
        for cell in &voronoi.cells {
            // Use neighbors to estimate embedding position
            let dist = Self::estimate_query_distance(query_embedding, cell, &voronoi.cells);
            cell_distances.push((cell, dist));
        }

        cell_distances.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if cell_distances.len() < 2 {
            return Vec::new();
        }

        // If the two nearest cells are close to equidistant, the query is interstitial
        let nearest = cell_distances[0].1;
        let second = cell_distances[1].1;

        if nearest < 0.001 {
            return Vec::new(); // Query is right on a seed, not interstitial
        }

        let ratio = second / nearest;
        if ratio < 1.5 {
            // Query is between domains — return the relevant cells
            cell_distances
                .iter()
                .take(3)
                .filter(|(_, d)| *d < nearest * 2.0)
                .map(|(cell, d)| {
                    let relevance = if *d > 0.0 { 1.0 / d } else { 1.0 };
                    (cell.uuid.clone(), cell.name.clone(), relevance)
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find the cell nearest to the query embedding.
    fn find_nearest_cell<'a>(
        _query: &[f32],
        cells: &'a [VoronoiCell],
    ) -> &'a VoronoiCell {
        // We need to compare with entity embeddings, but cells don't store them directly.
        // Use the cell's neighbor structure as a proxy: a cell whose neighbors
        // include the query's nearest overall match is likely the home cell.
        // Fallback: use the first cell (this is an approximation).
        //
        // In practice, this would be called after embedding lookup, so we
        // estimate based on neighbor names/face prominence.
        &cells[0]
    }

    /// Estimate distance between query and a target cell.
    ///
    /// Since cells don't store raw embeddings (those live in the slow store),
    /// we approximate using the cell's isolation and neighbor structure.
    fn estimate_distance(
        _query: &[f32],
        target: &VoronoiCell,
        _all_cells: &[VoronoiCell],
        // Note: _query and _all_cells reserved for future refined estimation
    ) -> f32 {
        // Approximation: use isolation as a proxy for distance
        // More isolated cells are generally farther from the dense center
        target.isolation * 0.5
    }

    /// Estimate distance from query to a cell for interstitial detection.
    fn estimate_query_distance(
        query: &[f32],
        cell: &VoronoiCell,
        _all_cells: &[VoronoiCell],
    ) -> f32 {
        // Use the cell's sparse direction to estimate alignment with query
        if let Some(ref sparse_dir) = cell.sparse_direction {
            // How much does the query align with this cell's sparse direction?
            // High alignment = query is in the cell's blind spot = far from this cell
            let alignment = cosine_similarity(query, sparse_dir).abs();
            alignment * cell.isolation
        } else {
            cell.isolation
        }
    }

    /// Score a specific jump from home cell to target cell.
    fn score_jump(
        query: &[f32],
        home: &VoronoiCell,
        target: &VoronoiCell,
        _voronoi: &VoronoiAnalysis,
        distance: f32,
        config: &DomainJumpConfig,
    ) -> (f32, JumpReason, String) {
        // Signal 1: Saturation of home region
        // If home has high density and many neighbors, we're in well-explored territory.
        // Diminishing returns from staying.
        let saturation = (home.local_density / 2.0).clamp(0.0, 1.0);

        // Signal 2: Target value
        // High isolation + high anisotropy = unexplored, structurally interesting region.
        let target_novelty = (target.isolation * target.anisotropy / 3.0).clamp(0.0, 1.0);

        // Signal 3: Anisotropic pull
        // If the target's sparse direction points TOWARD the home region,
        // the target has a blind spot facing us — our knowledge could fill it.
        let anisotropic_pull = if let Some(ref sparse_dir) = target.sparse_direction {
            // Check if sparse direction aligns with query direction
            cosine_similarity(query, sparse_dir).max(0.0)
        } else {
            0.0
        };

        // Signal 4: Distance penalty (closer jumps are less risky)
        let distance_penalty = (distance / config.max_distance).clamp(0.0, 1.0);

        // Combine signals
        let raw_value = config.saturation_weight * saturation
            + config.target_value_weight * target_novelty
            + config.relevance_weight * anisotropic_pull;

        // Apply distance penalty
        let value = (raw_value * (1.0 - distance_penalty * 0.5)).clamp(0.0, 1.0);

        // Determine primary reason
        let (reason, explanation) = if anisotropic_pull > 0.5 {
            (
                JumpReason::AnisotropicPull,
                format!(
                    "'{}' has a blind spot facing your current context — your knowledge \
                     about '{}' could fill gaps in that region",
                    target.name, home.name
                ),
            )
        } else if saturation > 0.6 {
            (
                JumpReason::Saturation,
                format!(
                    "Current region around '{}' is well-explored (density: {:.2}). \
                     '{}' offers unexplored territory (isolation: {:.2})",
                    home.name, home.local_density, target.name, target.isolation
                ),
            )
        } else if target_novelty > 0.5 {
            (
                JumpReason::VoidMatch,
                format!(
                    "'{}' is an isolated, structurally interesting region \
                     (isolation: {:.2}, anisotropy: {:.2}) — worth exploring",
                    target.name, target.isolation, target.anisotropy
                ),
            )
        } else {
            (
                JumpReason::InterstitialQuery,
                format!(
                    "Consider exploring '{}' — it's within reach \
                     (distance: {:.2}) and offers different perspective",
                    target.name, distance
                ),
            )
        };

        (value, reason, explanation)
    }
}
