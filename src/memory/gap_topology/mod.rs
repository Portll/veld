//! Veld layer: gap analysis, TDA, golden features, persistent homology interpretation.
//! Structural Gap Detection for Knowledge Graphs
//!
//! Detects structural gaps (missing edges) in the knowledge graph by analyzing
//! graph topology. Returns raw structural data — interpretation is left to consumers.
//!
//! Gap types detected:
//!
//! - **Open Triad (U-shape)**: A→B→C, no A↔C.
//! - **Diamond Gap**: A→B, A→C, B→D, C→D, no B↔C.
//! - **Star Gap**: Hub with N spokes, spokes unconnected.
//! - **Orbit Gap**: Two clusters share attractors but no cross-links.

mod detectors;
mod scoring;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::slow_store::{RawDiamondGap, RawOpenTriad, RawStarGap};

/// Trait abstracting the storage backend for gap detection queries.
pub trait GapStore: Send + Sync {
    fn find_open_triads(&self, min_strength: f32, limit: usize) -> Result<Vec<RawOpenTriad>>;
    fn find_diamond_gaps(&self, min_strength: f32, limit: usize) -> Result<Vec<RawDiamondGap>>;
    fn find_star_gaps(&self, min_spokes: usize, max_connectivity: f32, limit: usize) -> Result<Vec<RawStarGap>>;
    fn get_adjacency_list(&self, min_strength: f32) -> Result<HashMap<String, Vec<String>>>;
    fn get_entity_names(&self, uuids: &[&str]) -> Result<HashMap<String, String>>;
    fn load_embeddings(&self, uuids: &[&str]) -> Result<HashMap<String, Vec<f32>>>;
}

/// Classification of gap topology
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GapType {
    /// A→B→C, no A↔C.
    OpenTriad,
    /// A→B, A→C, B→D, C→D, no B↔C.
    DiamondGap,
    /// Hub with isolated spokes.
    StarGap,
    /// Two clusters share attractors but have no cross-links.
    OrbitGap,
}

impl GapType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenTriad => "open_triad",
            Self::DiamondGap => "diamond_gap",
            Self::StarGap => "star_gap",
            Self::OrbitGap => "orbit_gap",
        }
    }
}

/// Topological description of a gap's shape.
///
/// The shape IS the information — it tells us what kind of thing is missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeSignature {
    /// Number of nodes participating in the gap
    pub node_count: usize,
    /// Number of existing edges around the gap
    pub existing_edges: usize,
    /// Number of missing edges that define the gap
    pub missing_edges: usize,
    /// Ratio of missing to possible edges (0.0 = fully connected, 1.0 = fully disconnected)
    pub sparsity: f32,
    /// Canonical string for deduplication and pattern matching
    pub canonical: String,
}

impl ShapeSignature {
    /// Create a signature for an open triad (always 3 nodes, 2 edges, 1 missing)
    pub fn open_triad(bridge_name: &str) -> Self {
        Self {
            node_count: 3,
            existing_edges: 2,
            missing_edges: 1,
            sparsity: 1.0 / 3.0,
            canonical: format!("triad:bridge={}", bridge_name),
        }
    }

    /// Create a signature for a diamond gap (4 nodes, 4 edges, 1 missing diagonal)
    pub fn diamond() -> Self {
        Self {
            node_count: 4,
            existing_edges: 4,
            missing_edges: 1,
            sparsity: 1.0 / 6.0,
            canonical: "diamond:missing_diagonal".to_string(),
        }
    }

    /// Create a signature for a star gap
    pub fn star(hub_name: &str, spoke_count: usize, missing: usize, possible: usize) -> Self {
        let sparsity = if possible > 0 {
            missing as f32 / possible as f32
        } else {
            0.0
        };
        Self {
            node_count: spoke_count + 1,
            existing_edges: spoke_count,
            missing_edges: missing,
            sparsity,
            canonical: format!("star:hub={},spokes={},missing={}", hub_name, spoke_count, missing),
        }
    }

    /// Create a signature for an orbit gap
    pub fn orbit(cluster_a_size: usize, cluster_b_size: usize, shared_attractors: usize) -> Self {
        let total_nodes = cluster_a_size + cluster_b_size;
        let possible_cross = cluster_a_size * cluster_b_size;
        Self {
            node_count: total_nodes,
            existing_edges: 0, // cross-edges are what's missing
            missing_edges: possible_cross,
            sparsity: 1.0,
            canonical: format!(
                "orbit:a={},b={},shared_attractors={}",
                cluster_a_size, cluster_b_size, shared_attractors
            ),
        }
    }
}

/// A scored, characterized gap in the knowledge graph.
///
/// This is the output of gap detection: not just "something is missing"
/// but "here is the shape of what is missing, here is how confident we are,
/// and here is what we think should fill it."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapTopology {
    /// Unique identifier for this gap
    pub id: String,
    /// What kind of gap this is
    pub gap_type: GapType,
    /// The topological shape of the gap
    pub shape: ShapeSignature,
    /// Entities involved in this gap
    pub entities: Vec<GapEntity>,
    /// Missing links that define this gap
    pub missing_links: Vec<MissingLink>,
    /// How confident we are this is a real gap (0.0-1.0)
    /// High confidence = strong existing edges + high embedding similarity across the gap
    pub confidence: f32,
    /// Cosine similarity between the gap endpoints in embedding space
    /// High similarity + no edge = the gap is very likely real
    pub embedding_similarity: Option<f32>,
    /// How many other gaps this gap participates in or would affect if closed
    pub impact_score: f32,
}

/// An entity participating in a gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapEntity {
    pub uuid: String,
    pub name: String,
    pub role: GapRole,
}

/// The role an entity plays in a gap topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapRole {
    /// Endpoint of a U-shape (needs connection to the other endpoint)
    Endpoint,
    /// Bridge node connecting endpoints (the middle of the U)
    Bridge,
    /// Hub of a star gap
    Hub,
    /// Spoke of a star gap (needs connections to other spokes)
    Spoke,
    /// Top/bottom of a diamond
    Apex,
    /// Left/right of a diamond (the disconnected pair)
    Lateral,
    /// Member of a cluster in an orbit gap
    ClusterMember,
}

/// A missing link that defines part of a gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingLink {
    pub from_uuid: String,
    pub from_name: String,
    pub to_uuid: String,
    pub to_name: String,
    /// Why we think this link should exist
    pub evidence: String,
}

/// Configuration for gap detection
#[derive(Debug, Clone)]
pub struct GapDetectionConfig {
    /// Minimum edge strength to consider (filters weak/noisy edges)
    pub min_edge_strength: f32,
    /// Maximum number of gaps to return per type
    pub max_gaps_per_type: usize,
    /// Minimum embedding similarity to flag a U-shape as significant
    pub min_embedding_similarity: f32,
    /// For star gaps: minimum number of spokes
    pub star_min_spokes: usize,
    /// For star gaps: maximum spoke interconnectivity (0.0-1.0)
    pub star_max_connectivity: f32,
    /// For orbit detection: minimum cluster size
    pub orbit_min_cluster_size: usize,
}

impl Default for GapDetectionConfig {
    fn default() -> Self {
        Self {
            min_edge_strength: 0.2,
            max_gaps_per_type: 50,
            min_embedding_similarity: 0.3,
            star_min_spokes: 3,
            star_max_connectivity: 0.2,
            orbit_min_cluster_size: 3,
        }
    }
}

/// Results from a full gap detection run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapDetectionResult {
    /// All detected gaps, sorted by impact score
    pub gaps: Vec<GapTopology>,
    /// How many gaps of each type were found
    pub type_counts: HashMap<String, usize>,
    /// How long detection took
    pub duration_ms: u64,
}

/// The gap topology detection engine.
///
/// Orchestrates gap detection across multiple structural patterns,
/// scores them using embedding similarity, and identifies fractal repetitions.
pub struct GapDetector;

impl GapDetector {
    /// Run full gap detection on the knowledge graph via the slow store.
    ///
    /// This is the main entry point. It:
    /// 1. Finds structural gaps via SQL (U-shapes, diamonds, stars)
    /// 2. Scores each gap using embedding similarity
    /// 3. Detects orbit gaps via label propagation clustering
    /// 4. Identifies fractal patterns (same gap shape at different scales)
    /// 5. Computes impact scores (how many other gaps would closing this one affect?)
    pub fn detect(store: &dyn GapStore, config: &GapDetectionConfig) -> Result<GapDetectionResult> {
        let start = std::time::Instant::now();
        let mut gaps = Vec::new();

        // Phase 1: Structural gap detection via SQL
        let triads = detectors::detect_open_triads(store, config)?;
        let diamonds = detectors::detect_diamond_gaps(store, config)?;
        let stars = detectors::detect_star_gaps(store, config)?;

        gaps.extend(triads);
        gaps.extend(diamonds);
        gaps.extend(stars);

        // Phase 2: Orbit gaps via clustering
        let orbits = detectors::detect_orbit_gaps(store, config)?;
        gaps.extend(orbits);

        // Phase 3: Compute impact scores (how interconnected are the gaps?)
        scoring::compute_impact_scores(&mut gaps);

        // Sort by impact score (most impactful gaps first)
        gaps.sort_by(|a, b| {
            b.impact_score
                .partial_cmp(&a.impact_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Count by type
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for gap in &gaps {
            *type_counts
                .entry(gap.gap_type.as_str().to_string())
                .or_default() += 1;
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "Gap detection complete: {} gaps found in {}ms ({:?})",
            gaps.len(),
            duration_ms,
            type_counts
        );

        Ok(GapDetectionResult {
            gaps,
            type_counts,
            duration_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_type_as_str() {
        assert_eq!(GapType::OpenTriad.as_str(), "open_triad");
        assert_eq!(GapType::DiamondGap.as_str(), "diamond_gap");
        assert_eq!(GapType::StarGap.as_str(), "star_gap");
        assert_eq!(GapType::OrbitGap.as_str(), "orbit_gap");
    }

    #[test]
    fn test_shape_signature_open_triad() {
        let sig = ShapeSignature::open_triad("bridge");
        assert_eq!(sig.node_count, 3);
        assert_eq!(sig.existing_edges, 2);
        assert_eq!(sig.missing_edges, 1);
        assert!((sig.sparsity - 1.0 / 3.0).abs() < 0.01);
        assert!(sig.canonical.contains("bridge"));
    }

    #[test]
    fn test_shape_signature_diamond() {
        let sig = ShapeSignature::diamond();
        assert_eq!(sig.node_count, 4);
        assert_eq!(sig.existing_edges, 4);
        assert_eq!(sig.missing_edges, 1);
    }

    #[test]
    fn test_shape_signature_star() {
        let sig = ShapeSignature::star("hub", 5, 8, 10);
        assert_eq!(sig.node_count, 6);
        assert_eq!(sig.missing_edges, 8);
        assert!((sig.sparsity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_shape_signature_orbit() {
        let sig = ShapeSignature::orbit(3, 4, 2);
        assert_eq!(sig.node_count, 7);
        assert_eq!(sig.missing_edges, 12);
    }

    #[test]
    fn test_default_config() {
        let config = GapDetectionConfig::default();
        assert!(config.min_edge_strength > 0.0);
        assert!(config.max_gaps_per_type > 0);
        assert!(config.star_min_spokes >= 2);
    }

    #[test]
    fn test_gap_detector_finds_open_triad() {
        use crate::memory::slow_store::SlowStore;
        use tempfile::NamedTempFile;

        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        // Generate deterministic UUIDs for our three entities
        let uuid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
        let uuid_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
        let uuid_c = "cccccccc-cccc-cccc-cccc-cccccccccccc";

        // Embeddings: 8-dimensional vectors
        // A and C share a component so cosine similarity exceeds min_embedding_similarity (0.3)
        let emb_a: Vec<f32> = vec![1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let emb_b: Vec<f32> = vec![0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let emb_c: Vec<f32> = vec![0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let encode_emb = |emb: &[f32]| -> Vec<u8> {
            emb.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()
        };

        // Insert entities directly via SQL
        {
            let conn = store.conn.lock();
            for (uuid, name, emb) in [
                (uuid_a, "Rust", &emb_a),
                (uuid_b, "Memory", &emb_b),
                (uuid_c, "Graph", &emb_c),
            ] {
                conn.execute(
                    "INSERT INTO entities (uuid, name, labels, salience, mention_count, created_at, last_seen_at, summary, embedding) VALUES (?, ?, '[]', 0.5, 1, datetime('now'), datetime('now'), '', ?)",
                    rusqlite::params![uuid, name, encode_emb(emb)],
                )
                .unwrap();
            }

            // Insert edges: A→B and B→C (strength 0.8 each)
            let edge_uuid_ab = "eeeeeee1-eeee-eeee-eeee-eeeeeeeeeeee";
            let edge_uuid_bc = "eeeeeee2-eeee-eeee-eeee-eeeeeeeeeeee";
            for (edge_uuid, from, to) in [
                (edge_uuid_ab, uuid_a, uuid_b),
                (edge_uuid_bc, uuid_b, uuid_c),
            ] {
                conn.execute(
                    "INSERT INTO edges (uuid, from_entity, to_entity, relation_type, strength, tier, ltp_status, activation_count, created_at, last_activated, context) VALUES (?, ?, ?, 'RELATES_TO', 0.8, 'L1Working', 'None', 1, datetime('now'), datetime('now'), '')",
                    rusqlite::params![edge_uuid, from, to],
                )
                .unwrap();
            }
        }

        // Run gap detection with default config
        let config = GapDetectionConfig::default();
        let result = GapDetector::detect(&store, &config).unwrap();

        // Verify at least 1 gap is detected
        assert!(
            !result.gaps.is_empty(),
            "Expected at least 1 gap, found none"
        );

        // Verify at least one gap is an OpenTriad type
        let has_open_triad = result
            .gaps
            .iter()
            .any(|g| g.gap_type == GapType::OpenTriad);
        assert!(
            has_open_triad,
            "Expected at least one OpenTriad gap, found types: {:?}",
            result
                .gaps
                .iter()
                .map(|g| g.gap_type.as_str())
                .collect::<Vec<_>>()
        );

        // Verify the entities in the gap include "Rust" and/or "Graph" (the disconnected endpoints)
        let triad_gaps: Vec<&GapTopology> = result
            .gaps
            .iter()
            .filter(|g| g.gap_type == GapType::OpenTriad)
            .collect();
        let entity_names: Vec<&str> = triad_gaps
            .iter()
            .flat_map(|g| g.entities.iter().map(|e| e.name.as_str()))
            .collect();
        let has_rust_or_graph =
            entity_names.contains(&"Rust") || entity_names.contains(&"Graph");
        assert!(
            has_rust_or_graph,
            "Expected gap entities to include 'Rust' and/or 'Graph', found: {:?}",
            entity_names
        );
    }
}
