//! Golden Feature Generation
//!
//! Given the topology of gaps in the knowledge graph, generate hypotheses about
//! what entities, relationships, or structures would close the most gaps.
//!
//! A "golden feature" is the ideal missing piece — the entity, edge, or abstraction
//! that, if added, would resolve the maximum number of structural problems.
//!
//! Think of it as Planet X inference: Neptune was discovered not by seeing it,
//! but by noticing that Uranus's orbit had a shape that implied an unseen mass.
//! The shape of the gap told astronomers what should fill it.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::gap_topology::{GapScope, GapTopology, GapType};

/// A hypothesized feature that would close one or more gaps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenFeature {
    /// Unique identifier
    pub id: String,
    /// What kind of feature this is
    pub feature_type: FeatureType,
    /// What domain this applies to
    pub scope: GapScope,
    /// Human-readable description of the hypothesis
    pub description: String,
    /// Which gaps this feature would close (gap IDs)
    pub closes_gaps: Vec<String>,
    /// How many gaps this would close
    pub gap_closure_count: usize,
    /// Confidence in this hypothesis (0.0-1.0)
    pub confidence: f32,
    /// Entities that would be involved
    pub involved_entities: Vec<(String, String)>, // (uuid, name)
    /// Suggested action to realize this feature
    pub suggested_action: String,
}

/// The type of feature being hypothesized
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// A missing direct connection between two entities
    MissingEdge,
    /// A missing intermediary entity that would bridge clusters
    MissingHub,
    /// A shared abstraction that unifies isolated spokes
    SharedAbstraction,
    /// A cross-cutting concern that connects parallel paths
    CrossCut,
    /// A bridge between knowledge silos
    SiloBridge,
}

impl FeatureType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::MissingEdge => "missing_edge",
            Self::MissingHub => "missing_hub",
            Self::SharedAbstraction => "shared_abstraction",
            Self::CrossCut => "cross_cut",
            Self::SiloBridge => "silo_bridge",
        }
    }
}

/// Generate golden features from detected gaps.
///
/// The strategy depends on the gap topology:
/// - Open triads → MissingEdge (connect the endpoints)
/// - Diamond gaps → CrossCut (bridge the parallel paths)
/// - Star gaps → SharedAbstraction (give the spokes a common structure)
/// - Orbit gaps → SiloBridge (connect the clusters)
///
/// Features are ranked by how many gaps they would close (cascade impact).
pub fn generate_golden_features(gaps: &[GapTopology]) -> Vec<GoldenFeature> {
    let mut features: Vec<GoldenFeature> = Vec::new();

    // Strategy 1: Find the most impactful missing edges
    // Group U-shapes by their endpoints — if the same pair appears in multiple triads
    // (via different bridges), connecting them would close ALL those triads at once.
    features.extend(generate_edge_features(gaps));

    // Strategy 2: Find cross-cuts for diamond gaps
    features.extend(generate_crosscut_features(gaps));

    // Strategy 3: Find shared abstractions for star gaps
    features.extend(generate_abstraction_features(gaps));

    // Strategy 4: Find silo bridges for orbit gaps
    features.extend(generate_bridge_features(gaps));

    // Sort by gap closure count (most impactful first)
    features.sort_by(|a, b| {
        b.gap_closure_count
            .cmp(&a.gap_closure_count)
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Deduplicate: if two features involve the same entities, keep the higher-ranked one
    let mut seen_entity_pairs: HashMap<String, usize> = HashMap::new();
    let mut deduped = Vec::new();
    for feature in features {
        let key = feature
            .involved_entities
            .iter()
            .map(|(uuid, _)| uuid.as_str())
            .collect::<Vec<_>>()
            .join(":");
        if let std::collections::hash_map::Entry::Vacant(e) = seen_entity_pairs.entry(key) {
            e.insert(deduped.len());
            deduped.push(feature);
        }
    }

    deduped
}

/// Generate MissingEdge features from open triads.
///
/// If the same (A, C) pair appears as endpoints in multiple U-shapes
/// (A→B1→C, A→B2→C, ...), connecting A↔C would close all of them.
fn generate_edge_features(gaps: &[GapTopology]) -> Vec<GoldenFeature> {
    // Group triads by endpoint pairs
    let mut endpoint_pairs: HashMap<(String, String), Vec<&GapTopology>> = HashMap::new();
    for gap in gaps.iter().filter(|g| g.gap_type == GapType::OpenTriad) {
        for link in &gap.missing_links {
            let key = if link.from_uuid < link.to_uuid {
                (link.from_uuid.clone(), link.to_uuid.clone())
            } else {
                (link.to_uuid.clone(), link.from_uuid.clone())
            };
            endpoint_pairs.entry(key).or_default().push(gap);
        }
    }

    let mut features = Vec::new();
    for ((uuid_a, uuid_c), triads) in &endpoint_pairs {
        let first = triads[0];
        let link = &first.missing_links[0];

        // Find the names
        let name_a = if &link.from_uuid == uuid_a {
            &link.from_name
        } else {
            &link.to_name
        };
        let name_c = if &link.to_uuid == uuid_c {
            &link.to_name
        } else {
            &link.from_name
        };

        let bridges: Vec<String> = triads
            .iter()
            .flat_map(|t| {
                t.entities
                    .iter()
                    .filter(|e| matches!(e.role, super::gap_topology::GapRole::Bridge))
                    .map(|e| e.name.clone())
            })
            .collect();

        let gap_ids: Vec<String> = triads.iter().map(|t| t.id.clone()).collect();
        let avg_confidence =
            triads.iter().map(|t| t.confidence).sum::<f32>() / triads.len() as f32;

        let description = if bridges.len() == 1 {
            format!(
                "'{}' and '{}' are both connected through '{}' but have no direct relationship",
                name_a, name_c, bridges[0]
            )
        } else {
            format!(
                "'{}' and '{}' are connected through {} different bridges ({}) but have no direct relationship",
                name_a,
                name_c,
                bridges.len(),
                bridges.join(", ")
            )
        };

        let suggested_action = format!(
            "Consider establishing a direct relationship between '{}' and '{}'",
            name_a, name_c
        );

        features.push(GoldenFeature {
            id: format!("edge:{}:{}", &uuid_a[..8.min(uuid_a.len())], &uuid_c[..8.min(uuid_c.len())]),
            feature_type: FeatureType::MissingEdge,
            scope: first.scope.clone(),
            description,
            closes_gaps: gap_ids,
            gap_closure_count: triads.len(),
            confidence: avg_confidence,
            involved_entities: vec![
                (uuid_a.clone(), name_a.clone()),
                (uuid_c.clone(), name_c.clone()),
            ],
            suggested_action,
        });
    }

    features
}

/// Generate CrossCut features from diamond gaps.
///
/// In a diamond (A→B, A→C, B→D, C→D, no B↔C), connecting B and C
/// creates a cross-cut between parallel paths.
fn generate_crosscut_features(gaps: &[GapTopology]) -> Vec<GoldenFeature> {
    let mut features = Vec::new();

    for gap in gaps.iter().filter(|g| g.gap_type == GapType::DiamondGap) {
        let laterals: Vec<&super::gap_topology::GapEntity> = gap
            .entities
            .iter()
            .filter(|e| matches!(e.role, super::gap_topology::GapRole::Lateral))
            .collect();
        let apexes: Vec<&super::gap_topology::GapEntity> = gap
            .entities
            .iter()
            .filter(|e| matches!(e.role, super::gap_topology::GapRole::Apex))
            .collect();

        if laterals.len() < 2 || apexes.len() < 2 {
            continue;
        }

        let description = format!(
            "'{}' and '{}' are parallel intermediaries between '{}' and '{}' — \
             connecting them would reconcile two independent paths to the same conclusion",
            laterals[0].name, laterals[1].name, apexes[0].name, apexes[1].name
        );

        let suggested_action = format!(
            "Investigate the relationship between '{}' and '{}' — they may represent \
             two approaches to the same problem that should be reconciled",
            laterals[0].name, laterals[1].name
        );

        features.push(GoldenFeature {
            id: format!("crosscut:{}", gap.id),
            feature_type: FeatureType::CrossCut,
            scope: gap.scope.clone(),
            description,
            closes_gaps: vec![gap.id.clone()],
            gap_closure_count: 1,
            confidence: gap.confidence,
            involved_entities: laterals
                .iter()
                .map(|e| (e.uuid.clone(), e.name.clone()))
                .collect(),
            suggested_action,
        });
    }

    features
}

/// Generate SharedAbstraction features from star gaps.
///
/// When multiple spokes connect to a hub but not to each other,
/// the spokes likely share something in common that should be made explicit.
fn generate_abstraction_features(gaps: &[GapTopology]) -> Vec<GoldenFeature> {
    let mut features = Vec::new();

    for gap in gaps.iter().filter(|g| g.gap_type == GapType::StarGap) {
        let hub = gap
            .entities
            .iter()
            .find(|e| matches!(e.role, super::gap_topology::GapRole::Hub));
        let spokes: Vec<&super::gap_topology::GapEntity> = gap
            .entities
            .iter()
            .filter(|e| matches!(e.role, super::gap_topology::GapRole::Spoke))
            .collect();

        let hub = match hub {
            Some(h) => h,
            None => continue,
        };

        let spoke_names: Vec<&str> = spokes.iter().map(|s| s.name.as_str()).collect();
        let description = format!(
            "{} entities are all connected to '{}' but isolated from each other: {}. \
             They likely share a common trait or context that should be made explicit.",
            spokes.len(),
            hub.name,
            spoke_names.join(", ")
        );

        let suggested_action = format!(
            "Look for what '{}' share in common beyond their connection to '{}' — \
             this shared trait could become a new organizing concept",
            spoke_names.join("', '"),
            hub.name
        );

        features.push(GoldenFeature {
            id: format!("abstraction:{}", gap.id),
            feature_type: FeatureType::SharedAbstraction,
            scope: gap.scope.clone(),
            description,
            closes_gaps: vec![gap.id.clone()],
            gap_closure_count: gap.missing_links.len(),
            confidence: gap.confidence,
            involved_entities: spokes
                .iter()
                .map(|s| (s.uuid.clone(), s.name.clone()))
                .collect(),
            suggested_action,
        });
    }

    features
}

/// Generate SiloBridge features from orbit gaps.
///
/// Two knowledge clusters orbiting common attractors but never interacting
/// represent information silos. A bridge entity or relationship would unify them.
fn generate_bridge_features(gaps: &[GapTopology]) -> Vec<GoldenFeature> {
    let mut features = Vec::new();

    for gap in gaps.iter().filter(|g| g.gap_type == GapType::OrbitGap) {
        let entity_names: Vec<&str> = gap.entities.iter().map(|e| e.name.as_str()).collect();

        let description = format!(
            "Two knowledge clusters ({} entities total) share common attractors \
             but have no direct cross-connections. They represent information silos \
             that likely contain complementary knowledge.",
            gap.entities.len()
        );

        let suggested_action = format!(
            "Explore connections between these clusters: {}. \
             The shared attractors suggest they address related topics from different angles.",
            entity_names.join(", ")
        );

        features.push(GoldenFeature {
            id: format!("bridge:{}", gap.id),
            feature_type: FeatureType::SiloBridge,
            scope: gap.scope.clone(),
            description,
            closes_gaps: vec![gap.id.clone()],
            gap_closure_count: gap.missing_links.len(),
            confidence: gap.confidence,
            involved_entities: gap
                .entities
                .iter()
                .map(|e| (e.uuid.clone(), e.name.clone()))
                .collect(),
            suggested_action,
        });
    }

    features
}
