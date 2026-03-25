//! Gap Topology Detection Engine
//!
//! The missing information always has a shape. This module detects the topology
//! of gaps in the knowledge graph and characterizes them for inference.
//!
//! Gap types, from simplest to most complex:
//!
//! - **Open Triad (U-shape)**: A→B→C, no A↔C. Missing simple inference.
//! - **Diamond Gap**: A→B, A→C, B→D, C→D, no B↔C. Parallel paths not reconciled.
//! - **Star Gap**: Hub with N spokes, spokes unconnected. Missing community structure.
//! - **Orbit Gap**: Two clusters share attractors but no cross-links. Knowledge silos.
//! - **Void**: Region of embedding space surrounded by entities but empty. Unknown unknowns.
//! - **Planet X**: Multiple entities' relationships imply an unseen entity. Gravitational inference.
//! - **Fractal Gap**: Same gap shape repeating at different scales. Systematic blindness.
//!
//! The shape of each gap tells us what kind of information should fill it.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::slow_store::SlowStore;
use super::voronoi::{VoronoiAnalyzer, VoronoiConfig};
use crate::similarity::cosine_similarity;

/// Classification of gap topology
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GapType {
    /// A→B→C, no A↔C. Simplest gap — a missing inference.
    OpenTriad,
    /// A→B, A→C, B→D, C→D, no B↔C. Parallel paths not reconciled.
    DiamondGap,
    /// Hub with isolated spokes. Missing community structure.
    StarGap,
    /// Two clusters share attractors but have no cross-links.
    OrbitGap,
    /// Region of embedding space surrounded by entities but empty.
    Void,
    /// Relationships imply an unseen entity (gravitational inference).
    PlanetX,
    /// Same gap pattern repeating at different scales.
    FractalGap,
}

impl GapType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenTriad => "open_triad",
            Self::DiamondGap => "diamond_gap",
            Self::StarGap => "star_gap",
            Self::OrbitGap => "orbit_gap",
            Self::Void => "void",
            Self::PlanetX => "planet_x",
            Self::FractalGap => "fractal_gap",
        }
    }
}

/// The scope of what a gap pertains to
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapScope {
    /// Gap in code structure (modules, functions, dependencies)
    Codebase,
    /// Gap in database schema (tables, relationships, indexes)
    Schema,
    /// Gap in information content (knowledge, facts, concepts)
    Content,
}

impl GapScope {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Codebase => "codebase",
            Self::Schema => "schema",
            Self::Content => "content",
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
    /// What domain this gap belongs to
    pub scope: GapScope,
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
    /// Scope to analyze
    pub scope: GapScope,
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
            scope: GapScope::Content,
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
    /// Fractal patterns: gap shapes that repeat at different scales
    pub fractal_patterns: Vec<FractalPattern>,
}

/// A gap pattern that repeats at different scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalPattern {
    /// The repeating shape signature
    pub shape: String,
    /// Gap IDs at each scale where this pattern appears
    pub instances: Vec<String>,
    /// Number of scales at which this pattern repeats
    pub scale_count: usize,
    /// What this pattern suggests about systemic blindness
    pub interpretation: String,
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
    pub fn detect(store: &SlowStore, config: &GapDetectionConfig) -> Result<GapDetectionResult> {
        let start = std::time::Instant::now();
        let mut gaps = Vec::new();

        // Phase 1: Structural gap detection via SQL
        let triads = Self::detect_open_triads(store, config)?;
        let diamonds = Self::detect_diamond_gaps(store, config)?;
        let stars = Self::detect_star_gaps(store, config)?;

        gaps.extend(triads);
        gaps.extend(diamonds);
        gaps.extend(stars);

        // Phase 2: Orbit gaps via clustering
        let orbits = Self::detect_orbit_gaps(store, config)?;
        gaps.extend(orbits);

        // Phase 3: Voronoi analysis — voids and Planet X
        let voronoi_config = VoronoiConfig::default();
        if let Ok(voronoi) = VoronoiAnalyzer::analyze(store, &voronoi_config) {
            // Convert voids to gap topologies
            for void in &voronoi.voids {
                let entities: Vec<GapEntity> = void
                    .boundary_entities
                    .iter()
                    .map(|(uuid, name)| GapEntity {
                        uuid: uuid.clone(),
                        name: name.clone(),
                        role: GapRole::Endpoint,
                    })
                    .collect();

                gaps.push(GapTopology {
                    id: format!("void:r={:.2}:n={}", void.radius, void.boundary_count),
                    gap_type: GapType::Void,
                    shape: ShapeSignature {
                        node_count: void.boundary_count,
                        existing_edges: 0,
                        missing_edges: void.boundary_count,
                        sparsity: 1.0,
                        canonical: format!(
                            "void:boundary={},radius={:.2}",
                            void.boundary_count, void.radius
                        ),
                    },
                    scope: config.scope.clone(),
                    entities,
                    missing_links: Vec::new(), // voids don't have specific missing links
                    confidence: void.confidence,
                    embedding_similarity: None,
                    impact_score: 0.0,
                });
            }

            // Convert Planet X candidates to gap topologies
            for px in &voronoi.planet_x_candidates {
                let entities: Vec<GapEntity> = px
                    .evidence_entities
                    .iter()
                    .map(|(uuid, name)| GapEntity {
                        uuid: uuid.clone(),
                        name: name.clone(),
                        role: GapRole::Endpoint,
                    })
                    .collect();

                gaps.push(GapTopology {
                    id: format!("planet_x:conv={}", px.convergence_count),
                    gap_type: GapType::PlanetX,
                    shape: ShapeSignature {
                        node_count: px.convergence_count,
                        existing_edges: 0,
                        missing_edges: 1, // the missing entity itself
                        sparsity: 1.0,
                        canonical: format!(
                            "planet_x:convergence={}",
                            px.convergence_count
                        ),
                    },
                    scope: config.scope.clone(),
                    entities,
                    missing_links: Vec::new(),
                    confidence: px.confidence,
                    embedding_similarity: None,
                    impact_score: 0.0,
                });
            }
        }

        // Phase 4: Compute impact scores (how interconnected are the gaps?)
        Self::compute_impact_scores(&mut gaps);

        // Phase 5: Detect fractal patterns
        let fractal_patterns = Self::detect_fractal_patterns(&gaps);

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
            fractal_patterns,
        })
    }

    /// Detect open triads (U-shapes) and score by embedding similarity
    fn detect_open_triads(
        store: &SlowStore,
        config: &GapDetectionConfig,
    ) -> Result<Vec<GapTopology>> {
        let raw_triads = store.find_open_triads(config.min_edge_strength, config.max_gaps_per_type)?;

        // Collect UUIDs for embedding lookup
        let uuids: Vec<&str> = raw_triads
            .iter()
            .flat_map(|t| [t.node_a.as_str(), t.node_c.as_str()])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let embeddings = store.load_embeddings(&uuids)?;

        let mut gaps = Vec::new();
        for triad in &raw_triads {
            // Score: how similar are the endpoints in embedding space?
            let emb_sim = match (embeddings.get(&triad.node_a), embeddings.get(&triad.node_c)) {
                (Some(a), Some(c)) => Some(cosine_similarity(a, c)),
                _ => None,
            };

            // Skip low-similarity gaps (endpoints aren't semantically related)
            if let Some(sim) = emb_sim {
                if sim < config.min_embedding_similarity {
                    continue;
                }
            }

            // Confidence: strong edges + high embedding similarity = confident gap
            let edge_confidence = (triad.ab_strength + triad.bc_strength) / 2.0;
            let emb_confidence = emb_sim.unwrap_or(0.5);
            let salience_factor = (triad.a_salience + triad.c_salience) / 2.0;
            let confidence = (edge_confidence * 0.4 + emb_confidence * 0.4 + salience_factor * 0.2)
                .clamp(0.0, 1.0);

            let gap_id = format!(
                "triad:{}:{}:{}",
                &triad.node_a[..8.min(triad.node_a.len())],
                &triad.node_b[..8.min(triad.node_b.len())],
                &triad.node_c[..8.min(triad.node_c.len())]
            );

            gaps.push(GapTopology {
                id: gap_id,
                gap_type: GapType::OpenTriad,
                shape: ShapeSignature::open_triad(&triad.node_b_name),
                scope: config.scope.clone(),
                entities: vec![
                    GapEntity {
                        uuid: triad.node_a.clone(),
                        name: triad.node_a_name.clone(),
                        role: GapRole::Endpoint,
                    },
                    GapEntity {
                        uuid: triad.node_b.clone(),
                        name: triad.node_b_name.clone(),
                        role: GapRole::Bridge,
                    },
                    GapEntity {
                        uuid: triad.node_c.clone(),
                        name: triad.node_c_name.clone(),
                        role: GapRole::Endpoint,
                    },
                ],
                missing_links: vec![MissingLink {
                    from_uuid: triad.node_a.clone(),
                    from_name: triad.node_a_name.clone(),
                    to_uuid: triad.node_c.clone(),
                    to_name: triad.node_c_name.clone(),
                    evidence: format!(
                        "Both linked to '{}' ({}: {:.2}, {}: {:.2}) but not to each other",
                        triad.node_b_name,
                        triad.ab_relation,
                        triad.ab_strength,
                        triad.bc_relation,
                        triad.bc_strength
                    ),
                }],
                confidence,
                embedding_similarity: emb_sim,
                impact_score: 0.0, // computed later
            });
        }

        Ok(gaps)
    }

    /// Detect diamond gaps and score them
    fn detect_diamond_gaps(
        store: &SlowStore,
        config: &GapDetectionConfig,
    ) -> Result<Vec<GapTopology>> {
        let raw_diamonds =
            store.find_diamond_gaps(config.min_edge_strength, config.max_gaps_per_type)?;

        let uuids: Vec<&str> = raw_diamonds
            .iter()
            .flat_map(|d| [d.left.as_str(), d.right.as_str()])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let embeddings = store.load_embeddings(&uuids)?;

        let mut gaps = Vec::new();
        for diamond in &raw_diamonds {
            let emb_sim = match (embeddings.get(&diamond.left), embeddings.get(&diamond.right)) {
                (Some(l), Some(r)) => Some(cosine_similarity(l, r)),
                _ => None,
            };

            if let Some(sim) = emb_sim {
                if sim < config.min_embedding_similarity {
                    continue;
                }
            }

            let confidence = emb_sim.unwrap_or(0.5) * 0.6 + 0.4; // diamonds are inherently strong signals

            let gap_id = format!(
                "diamond:{}:{}:{}:{}",
                &diamond.top[..8.min(diamond.top.len())],
                &diamond.left[..8.min(diamond.left.len())],
                &diamond.right[..8.min(diamond.right.len())],
                &diamond.bottom[..8.min(diamond.bottom.len())]
            );

            gaps.push(GapTopology {
                id: gap_id,
                gap_type: GapType::DiamondGap,
                shape: ShapeSignature::diamond(),
                scope: config.scope.clone(),
                entities: vec![
                    GapEntity {
                        uuid: diamond.top.clone(),
                        name: diamond.top_name.clone(),
                        role: GapRole::Apex,
                    },
                    GapEntity {
                        uuid: diamond.left.clone(),
                        name: diamond.left_name.clone(),
                        role: GapRole::Lateral,
                    },
                    GapEntity {
                        uuid: diamond.right.clone(),
                        name: diamond.right_name.clone(),
                        role: GapRole::Lateral,
                    },
                    GapEntity {
                        uuid: diamond.bottom.clone(),
                        name: diamond.bottom_name.clone(),
                        role: GapRole::Apex,
                    },
                ],
                missing_links: vec![MissingLink {
                    from_uuid: diamond.left.clone(),
                    from_name: diamond.left_name.clone(),
                    to_uuid: diamond.right.clone(),
                    to_name: diamond.right_name.clone(),
                    evidence: format!(
                        "Both reachable from '{}' and converge at '{}' via parallel paths, but not directly connected",
                        diamond.top_name, diamond.bottom_name
                    ),
                }],
                confidence,
                embedding_similarity: emb_sim,
                impact_score: 0.0,
            });
        }

        Ok(gaps)
    }

    /// Detect star gaps (hub with disconnected spokes)
    fn detect_star_gaps(
        store: &SlowStore,
        config: &GapDetectionConfig,
    ) -> Result<Vec<GapTopology>> {
        let raw_stars = store.find_star_gaps(
            config.star_min_spokes,
            config.star_max_connectivity,
            config.max_gaps_per_type,
        )?;

        let mut gaps = Vec::new();
        for star in &raw_stars {
            let confidence = (1.0 - (star.possible_edges as f32 - star.missing_edges as f32)
                / star.possible_edges.max(1) as f32)
                * star.avg_hub_strength;

            let gap_id = format!(
                "star:{}:spokes={}",
                &star.hub[..8.min(star.hub.len())],
                star.spokes.len()
            );

            let mut entities = vec![GapEntity {
                uuid: star.hub.clone(),
                name: star.hub_name.clone(),
                role: GapRole::Hub,
            }];
            entities.extend(star.spokes.iter().map(|(uuid, name)| GapEntity {
                uuid: uuid.clone(),
                name: name.clone(),
                role: GapRole::Spoke,
            }));

            // Generate missing links between spokes (sample — don't enumerate all N^2)
            let max_links = 10;
            let mut missing_links = Vec::new();
            'outer: for i in 0..star.spokes.len() {
                for j in (i + 1)..star.spokes.len() {
                    missing_links.push(MissingLink {
                        from_uuid: star.spokes[i].0.clone(),
                        from_name: star.spokes[i].1.clone(),
                        to_uuid: star.spokes[j].0.clone(),
                        to_name: star.spokes[j].1.clone(),
                        evidence: format!(
                            "Both connected to hub '{}' but not to each other",
                            star.hub_name
                        ),
                    });
                    if missing_links.len() >= max_links {
                        break 'outer;
                    }
                }
            }

            gaps.push(GapTopology {
                id: gap_id,
                gap_type: GapType::StarGap,
                shape: ShapeSignature::star(
                    &star.hub_name,
                    star.spokes.len(),
                    star.missing_edges,
                    star.possible_edges,
                ),
                scope: config.scope.clone(),
                entities,
                missing_links,
                confidence,
                embedding_similarity: None,
                impact_score: 0.0,
            });
        }

        Ok(gaps)
    }

    /// Detect orbit gaps using label propagation clustering.
    ///
    /// Two clusters that share "attractor" entities (common neighbors outside both clusters)
    /// but have no direct cross-links represent knowledge silos that should be connected.
    fn detect_orbit_gaps(
        store: &SlowStore,
        config: &GapDetectionConfig,
    ) -> Result<Vec<GapTopology>> {
        let adj = store.get_adjacency_list(config.min_edge_strength)?;
        if adj.is_empty() {
            return Ok(Vec::new());
        }

        // Label propagation clustering
        let clusters = Self::label_propagation(&adj, 20);

        // Filter to meaningful clusters
        let meaningful_clusters: Vec<(usize, Vec<String>)> = clusters
            .into_iter()
            .filter(|(_, members)| members.len() >= config.orbit_min_cluster_size)
            .collect();

        if meaningful_clusters.len() < 2 {
            return Ok(Vec::new());
        }

        // Build cluster membership lookup
        let mut entity_cluster: HashMap<&str, usize> = HashMap::new();
        for (cluster_id, members) in &meaningful_clusters {
            for member in members {
                entity_cluster.insert(member.as_str(), *cluster_id);
            }
        }

        // Find cluster pairs that share attractors but have no direct cross-links
        let mut gaps = Vec::new();
        for i in 0..meaningful_clusters.len() {
            for j in (i + 1)..meaningful_clusters.len() {
                let (id_a, members_a) = &meaningful_clusters[i];
                let (id_b, members_b) = &meaningful_clusters[j];

                // Find shared attractors: entities outside both clusters
                // that have neighbors in both
                let set_a: HashSet<&str> = members_a.iter().map(|s| s.as_str()).collect();
                let set_b: HashSet<&str> = members_b.iter().map(|s| s.as_str()).collect();

                let mut shared_attractors: HashSet<&str> = HashSet::new();
                for (entity, neighbors) in &adj {
                    if set_a.contains(entity.as_str()) || set_b.contains(entity.as_str()) {
                        continue;
                    }
                    let has_a_neighbor = neighbors.iter().any(|n| set_a.contains(n.as_str()));
                    let has_b_neighbor = neighbors.iter().any(|n| set_b.contains(n.as_str()));
                    if has_a_neighbor && has_b_neighbor {
                        shared_attractors.insert(entity.as_str());
                    }
                }

                if shared_attractors.is_empty() {
                    continue;
                }

                // Check for direct cross-links
                let mut cross_links = 0usize;
                for member_a in members_a {
                    if let Some(neighbors) = adj.get(member_a.as_str()) {
                        for neighbor in neighbors {
                            if set_b.contains(neighbor.as_str()) {
                                cross_links += 1;
                            }
                        }
                    }
                }

                // If few or no cross-links relative to shared attractors, it's an orbit gap
                let cross_ratio = cross_links as f32
                    / (members_a.len() * members_b.len()).max(1) as f32;
                if cross_ratio < 0.1 {
                    let confidence = (shared_attractors.len() as f32 / 5.0).clamp(0.0, 1.0)
                        * (1.0 - cross_ratio);

                    let gap_id = format!("orbit:{}:{}", id_a, id_b);

                    // Get names for entities
                    let all_uuids: Vec<&str> = members_a
                        .iter()
                        .chain(members_b.iter())
                        .map(|s| s.as_str())
                        .take(20) // limit for name lookup
                        .collect();
                    let names = store.get_entity_names(&all_uuids).unwrap_or_default();

                    let entities: Vec<GapEntity> = members_a
                        .iter()
                        .take(5)
                        .chain(members_b.iter().take(5))
                        .map(|uuid| GapEntity {
                            uuid: uuid.clone(),
                            name: names
                                .get(uuid.as_str())
                                .cloned()
                                .unwrap_or_else(|| uuid[..8.min(uuid.len())].to_string()),
                            role: GapRole::ClusterMember,
                        })
                        .collect();

                    // Sample missing links
                    let mut missing_links = Vec::new();
                    for a in members_a.iter().take(3) {
                        for b in members_b.iter().take(3) {
                            let a_name = names
                                .get(a.as_str())
                                .cloned()
                                .unwrap_or_else(|| a[..8.min(a.len())].to_string());
                            let b_name = names
                                .get(b.as_str())
                                .cloned()
                                .unwrap_or_else(|| b[..8.min(b.len())].to_string());
                            missing_links.push(MissingLink {
                                from_uuid: a.clone(),
                                from_name: a_name,
                                to_uuid: b.clone(),
                                to_name: b_name,
                                evidence: format!(
                                    "Clusters share {} attractor(s) but have only {} cross-link(s)",
                                    shared_attractors.len(),
                                    cross_links
                                ),
                            });
                        }
                    }

                    gaps.push(GapTopology {
                        id: gap_id,
                        gap_type: GapType::OrbitGap,
                        shape: ShapeSignature::orbit(
                            members_a.len(),
                            members_b.len(),
                            shared_attractors.len(),
                        ),
                        scope: config.scope.clone(),
                        entities,
                        missing_links,
                        confidence,
                        embedding_similarity: None,
                        impact_score: 0.0,
                    });
                }
            }
        }

        Ok(gaps)
    }

    /// Simple label propagation for community detection.
    ///
    /// Each node starts with its own label, then adopts the most common label
    /// among its neighbors. Converges to clusters of densely connected nodes.
    fn label_propagation(
        adj: &HashMap<String, Vec<String>>,
        max_iterations: usize,
    ) -> Vec<(usize, Vec<String>)> {
        let nodes: Vec<&String> = adj.keys().collect();
        let mut labels: HashMap<&str, usize> = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            labels.insert(node.as_str(), i);
        }

        for _ in 0..max_iterations {
            let mut changed = false;
            for node in &nodes {
                if let Some(neighbors) = adj.get(node.as_str()) {
                    if neighbors.is_empty() {
                        continue;
                    }
                    // Count neighbor labels
                    let mut label_counts: HashMap<usize, usize> = HashMap::new();
                    for neighbor in neighbors {
                        if let Some(&label) = labels.get(neighbor.as_str()) {
                            *label_counts.entry(label).or_default() += 1;
                        }
                    }
                    // Adopt most common neighbor label
                    if let Some((&best_label, _)) =
                        label_counts.iter().max_by_key(|(_, count)| *count)
                    {
                        let current = labels.get(node.as_str()).copied().unwrap_or(0);
                        if best_label != current {
                            labels.insert(node.as_str(), best_label);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Group by label
        let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
        for (node, label) in &labels {
            clusters
                .entry(*label)
                .or_default()
                .push(node.to_string());
        }

        clusters.into_iter().collect()
    }

    /// Compute impact scores: how many other gaps share entities with this one?
    ///
    /// A gap with high impact participates in many structural problems.
    /// Closing it would cascade through the graph and resolve multiple gaps.
    fn compute_impact_scores(gaps: &mut [GapTopology]) {
        // Build entity → gap index
        let mut entity_to_gaps: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, gap) in gaps.iter().enumerate() {
            for entity in &gap.entities {
                entity_to_gaps
                    .entry(entity.uuid.clone())
                    .or_default()
                    .push(i);
            }
            for link in &gap.missing_links {
                entity_to_gaps
                    .entry(link.from_uuid.clone())
                    .or_default()
                    .push(i);
                entity_to_gaps
                    .entry(link.to_uuid.clone())
                    .or_default()
                    .push(i);
            }
        }

        // Score each gap by how many OTHER gaps it shares entities with
        for i in 0..gaps.len() {
            let mut related_gaps: HashSet<usize> = HashSet::new();
            for entity in &gaps[i].entities {
                if let Some(gap_indices) = entity_to_gaps.get(&entity.uuid) {
                    for &idx in gap_indices {
                        if idx != i {
                            related_gaps.insert(idx);
                        }
                    }
                }
            }
            for link in &gaps[i].missing_links {
                for uuid in [&link.from_uuid, &link.to_uuid] {
                    if let Some(gap_indices) = entity_to_gaps.get(uuid) {
                        for &idx in gap_indices {
                            if idx != i {
                                related_gaps.insert(idx);
                            }
                        }
                    }
                }
            }

            // Normalize: impact = related gaps / total gaps
            let total = gaps.len().max(1) as f32;
            gaps[i].impact_score =
                (related_gaps.len() as f32 / total * gaps[i].confidence).clamp(0.0, 1.0);
        }
    }

    /// Detect fractal patterns: gap shapes that repeat at different scales.
    ///
    /// If the same topological pattern appears as individual U-shapes, as
    /// diamond-level structures, AND as inter-cluster orbit gaps, that's
    /// a fractal gap — a systematic blindness that repeats at every level.
    fn detect_fractal_patterns(gaps: &[GapTopology]) -> Vec<FractalPattern> {
        // Group gaps by bridge/hub entities
        // If entity X appears as a bridge in triads AND as part of a star gap AND in an orbit,
        // the same structural weakness repeats at multiple scales
        let mut entity_gap_types: HashMap<String, HashSet<String>> = HashMap::new();
        for gap in gaps {
            for entity in &gap.entities {
                entity_gap_types
                    .entry(entity.uuid.clone())
                    .or_default()
                    .insert(gap.gap_type.as_str().to_string());
            }
        }

        let mut patterns = Vec::new();
        for (entity_uuid, gap_types) in &entity_gap_types {
            if gap_types.len() >= 2 {
                // This entity participates in gaps at multiple structural scales
                let instances: Vec<String> = gaps
                    .iter()
                    .filter(|g| g.entities.iter().any(|e| &e.uuid == entity_uuid))
                    .map(|g| g.id.clone())
                    .collect();

                let entity_name = gaps
                    .iter()
                    .flat_map(|g| &g.entities)
                    .find(|e| &e.uuid == entity_uuid)
                    .map(|e| e.name.clone())
                    .unwrap_or_else(|| entity_uuid.clone());

                let types_str: Vec<&str> = gap_types.iter().map(|s| s.as_str()).collect();
                let interpretation = format!(
                    "'{}' appears in gaps at {} different structural levels ({}). \
                     This entity is a recurring weak point in the knowledge graph — \
                     strengthening connections around it would resolve multiple gap types simultaneously.",
                    entity_name,
                    gap_types.len(),
                    types_str.join(", ")
                );

                patterns.push(FractalPattern {
                    shape: format!("multi_scale:{}", types_str.join("+")),
                    instances,
                    scale_count: gap_types.len(),
                    interpretation,
                });
            }
        }

        // Sort by scale count (most fractal first)
        patterns.sort_by(|a, b| b.scale_count.cmp(&a.scale_count));
        patterns
    }
}
