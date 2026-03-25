//! Thought Surfacing Engine
//!
//! Orchestrates gap detection → golden feature generation → structured thoughts.
//!
//! A "thought" is a structured insight about the knowledge graph that is surfaced
//! to the user or agent. Thoughts are generated from gap analysis and golden features,
//! then persisted in the slow store for tracking (surfaced count, dismissal).
//!
//! Thoughts answer: "What should I know about this codebase / schema / content
//! that I can't currently see?"

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::gap_topology::{FractalPattern, GapDetectionConfig, GapDetector, GapScope, GapTopology, GapType};
use super::golden_features::{generate_golden_features, GoldenFeature};
use super::slow_store::SlowStore;

/// A structured thought surfaced from gap analysis.
///
/// Thoughts are the user-facing output of the gap topology engine.
/// They describe what's missing, why it matters, and what to do about it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// Unique identifier
    pub id: String,
    /// What kind of insight this is
    pub kind: ThoughtKind,
    /// What domain this pertains to
    pub scope: GapScope,
    /// Confidence in this thought (0.0-1.0)
    pub confidence: f32,
    /// Human-readable description of the insight
    pub description: String,
    /// Hypothesis about what should fill the gap
    pub hypothesis: Option<String>,
    /// Gap or feature IDs that support this thought
    pub evidence: Vec<String>,
    /// How impactful addressing this would be
    pub impact_score: f32,
    /// Entities involved
    pub entities: Vec<(String, String)>, // (uuid, name)
}

/// Classification of thought types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThoughtKind {
    /// A specific missing connection identified
    MissingConnection,
    /// A structural weakness (star, diamond pattern)
    StructuralWeakness,
    /// Knowledge silos that should be bridged
    KnowledgeSilo,
    /// A golden feature recommendation
    GoldenFeature,
    /// A fractal pattern (systematic blindness)
    SystematicGap,
}

impl ThoughtKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::MissingConnection => "missing_connection",
            Self::StructuralWeakness => "structural_weakness",
            Self::KnowledgeSilo => "knowledge_silo",
            Self::GoldenFeature => "golden_feature",
            Self::SystematicGap => "systematic_gap",
        }
    }
}

/// Results from a thought generation cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtReport {
    /// Generated thoughts, sorted by impact
    pub thoughts: Vec<Thought>,
    /// Summary statistics
    pub stats: ThoughtStats,
}

/// Statistics from thought generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtStats {
    pub gaps_detected: usize,
    pub golden_features_generated: usize,
    pub thoughts_generated: usize,
    pub fractal_patterns_found: usize,
    pub duration_ms: u64,
}

/// The thought engine: orchestrates gap detection, feature generation, and thought surfacing.
pub struct ThoughtEngine;

impl ThoughtEngine {
    /// Run a full thought generation cycle.
    ///
    /// 1. Detect gaps in the knowledge graph via the slow store
    /// 2. Generate golden features from the gaps
    /// 3. Package everything as structured thoughts
    /// 4. Persist thoughts in the slow store
    /// 5. Return the thought report
    pub fn generate(store: &SlowStore, config: &GapDetectionConfig) -> Result<ThoughtReport> {
        let start = std::time::Instant::now();

        // Phase 1: Detect gaps
        let detection_result = GapDetector::detect(store, config)?;

        // Phase 2: Generate golden features
        let golden_features = generate_golden_features(&detection_result.gaps);

        // Phase 3: Generate thoughts from gaps and features
        let mut thoughts = Vec::new();

        // Thoughts from high-confidence gaps
        for gap in &detection_result.gaps {
            if gap.confidence < 0.3 {
                continue; // Skip low-confidence gaps
            }

            let thought = Self::gap_to_thought(gap);
            thoughts.push(thought);
        }

        // Thoughts from golden features (these are the actionable recommendations)
        for feature in &golden_features {
            let thought = Self::feature_to_thought(feature);
            thoughts.push(thought);
        }

        // Thoughts from fractal patterns (systematic blindness — highest priority)
        for pattern in &detection_result.fractal_patterns {
            let thought = Self::fractal_to_thought(pattern);
            thoughts.push(thought);
        }

        // Sort by impact score (most impactful first)
        thoughts.sort_by(|a, b| {
            b.impact_score
                .partial_cmp(&a.impact_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Phase 4: Persist thoughts and gaps in the slow store
        for gap in &detection_result.gaps {
            let entities_json = serde_json::to_string(&gap.entities).unwrap_or_default();
            let links_json = serde_json::to_string(&gap.missing_links).unwrap_or_default();
            let _ = store.store_gap(
                &gap.id,
                gap.gap_type.as_str(),
                &gap.shape.canonical,
                &entities_json,
                &links_json,
                gap.confidence,
                gap.embedding_similarity,
                gap.impact_score,
                gap.scope.as_str(),
            );
        }

        for thought in &thoughts {
            let evidence_json = serde_json::to_string(&thought.evidence).unwrap_or_default();
            let entities_json = serde_json::to_string(&thought.entities).unwrap_or_default();
            let _ = store.store_thought(
                &thought.id,
                thought.kind.as_str(),
                thought.scope.as_str(),
                thought.confidence,
                &thought.description,
                thought.hypothesis.as_deref(),
                &evidence_json,
                thought.impact_score,
                &entities_json,
            );
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        let stats = ThoughtStats {
            gaps_detected: detection_result.gaps.len(),
            golden_features_generated: golden_features.len(),
            thoughts_generated: thoughts.len(),
            fractal_patterns_found: detection_result.fractal_patterns.len(),
            duration_ms,
        };

        tracing::info!(
            "Thought generation complete: {} gaps → {} features → {} thoughts ({}ms)",
            stats.gaps_detected,
            stats.golden_features_generated,
            stats.thoughts_generated,
            stats.duration_ms
        );

        Ok(ThoughtReport { thoughts, stats })
    }

    /// Convert a detected gap into a thought
    fn gap_to_thought(gap: &GapTopology) -> Thought {
        let (kind, description) = match gap.gap_type {
            GapType::OpenTriad => {
                let link = &gap.missing_links[0];
                (
                    ThoughtKind::MissingConnection,
                    format!(
                        "'{}' and '{}' are indirectly connected but lack a direct relationship. {}",
                        link.from_name, link.to_name, link.evidence
                    ),
                )
            }
            GapType::DiamondGap => {
                let link = &gap.missing_links[0];
                (
                    ThoughtKind::StructuralWeakness,
                    format!(
                        "Diamond pattern: '{}' and '{}' are parallel paths that converge \
                         but aren't reconciled. {}",
                        link.from_name, link.to_name, link.evidence
                    ),
                )
            }
            GapType::StarGap => {
                let hub_name = gap
                    .entities
                    .iter()
                    .find(|e| matches!(e.role, super::gap_topology::GapRole::Hub))
                    .map(|e| e.name.as_str())
                    .unwrap_or("unknown");
                let spoke_count = gap
                    .entities
                    .iter()
                    .filter(|e| matches!(e.role, super::gap_topology::GapRole::Spoke))
                    .count();
                (
                    ThoughtKind::StructuralWeakness,
                    format!(
                        "'{}' connects {} entities that are isolated from each other. \
                         The spokes of this wheel have no rim.",
                        hub_name, spoke_count
                    ),
                )
            }
            GapType::OrbitGap => (
                ThoughtKind::KnowledgeSilo,
                format!(
                    "Two knowledge clusters ({} entities) share common context but \
                     are disconnected. These silos likely contain complementary information.",
                    gap.entities.len()
                ),
            ),
            GapType::Void => {
                let boundary_names: Vec<&str> = gap
                    .entities
                    .iter()
                    .take(5)
                    .map(|e| e.name.as_str())
                    .collect();
                (
                    ThoughtKind::KnowledgeSilo,
                    format!(
                        "Empty region in knowledge space bordered by: {}. \
                         This is a blind spot — an area where related concepts exist \
                         on all sides but the interior is unexplored.",
                        boundary_names.join(", ")
                    ),
                )
            }
            GapType::PlanetX => {
                let evidence_names: Vec<&str> = gap
                    .entities
                    .iter()
                    .take(5)
                    .map(|e| e.name.as_str())
                    .collect();
                (
                    ThoughtKind::MissingConnection,
                    format!(
                        "Implied unseen concept: {} entities' relationships converge on \
                         a point in knowledge space where no entity exists ({}). \
                         Like Neptune inferred from orbital perturbations.",
                        gap.entities.len(),
                        evidence_names.join(", ")
                    ),
                )
            }
            GapType::FractalGap => (
                ThoughtKind::SystematicGap,
                format!(
                    "Fractal gap pattern: the same structural weakness repeats at {} different scales.",
                    gap.entities.len()
                ),
            ),
        };

        let hypothesis = match gap.gap_type {
            GapType::OpenTriad => {
                let link = &gap.missing_links[0];
                Some(format!(
                    "Establishing a direct connection between '{}' and '{}' would close this gap{}",
                    link.from_name,
                    link.to_name,
                    gap.embedding_similarity
                        .map(|s| format!(" (embedding similarity: {:.2})", s))
                        .unwrap_or_default()
                ))
            }
            GapType::DiamondGap => {
                let link = &gap.missing_links[0];
                Some(format!(
                    "'{}' and '{}' may represent two perspectives on the same concept — \
                     reconciling them would unify parallel reasoning paths",
                    link.from_name, link.to_name
                ))
            }
            GapType::StarGap => Some(
                "The isolated spokes likely share a common trait beyond their hub connection — \
                 identifying this shared trait would create a richer subgraph"
                    .to_string(),
            ),
            GapType::OrbitGap => Some(
                "Look for concepts that exist in both clusters — these are natural bridge points \
                 for connecting the silos"
                    .to_string(),
            ),
            GapType::Void => Some(
                "This void represents an unexplored territory. The surrounding entities \
                 suggest what kind of knowledge should fill this space — investigate the \
                 intersections between the boundary concepts."
                    .to_string(),
            ),
            GapType::PlanetX => {
                let names: Vec<&str> = gap
                    .entities
                    .iter()
                    .take(3)
                    .map(|e| e.name.as_str())
                    .collect();
                Some(format!(
                    "There may be an unrecognized concept that connects {}. \
                     Their relationships all point toward the same empty region — \
                     naming and exploring this concept could restructure understanding.",
                    names.join(", ")
                ))
            }
            GapType::FractalGap => Some(
                "This repeating pattern indicates a structural blind spot that \
                 exists at multiple levels of abstraction. Address the root pattern \
                 to cascade improvements across all scales."
                    .to_string(),
            ),
        };

        Thought {
            id: format!("thought:gap:{}", gap.id),
            kind,
            scope: gap.scope.clone(),
            confidence: gap.confidence,
            description,
            hypothesis,
            evidence: vec![gap.id.clone()],
            impact_score: gap.impact_score,
            entities: gap
                .entities
                .iter()
                .map(|e| (e.uuid.clone(), e.name.clone()))
                .collect(),
        }
    }

    /// Convert a golden feature into a thought
    fn feature_to_thought(feature: &GoldenFeature) -> Thought {
        let kind = ThoughtKind::GoldenFeature;

        let hypothesis = Some(feature.suggested_action.clone());

        Thought {
            id: format!("thought:feature:{}", feature.id),
            kind,
            scope: feature.scope.clone(),
            confidence: feature.confidence,
            description: feature.description.clone(),
            hypothesis,
            evidence: feature.closes_gaps.clone(),
            impact_score: feature.gap_closure_count as f32 / 10.0, // normalize
            entities: feature.involved_entities.clone(),
        }
    }

    /// Convert a fractal pattern into a thought (highest priority — systematic blindness)
    fn fractal_to_thought(pattern: &FractalPattern) -> Thought {
        Thought {
            id: format!("thought:fractal:{}", pattern.shape),
            kind: ThoughtKind::SystematicGap,
            scope: GapScope::Content,
            confidence: (pattern.scale_count as f32 / 4.0).clamp(0.3, 1.0),
            description: pattern.interpretation.clone(),
            hypothesis: Some(format!(
                "This pattern repeats at {} scales, suggesting a systematic blind spot. \
                 Addressing the root entity would cascade improvements across all {} gap instances.",
                pattern.scale_count,
                pattern.instances.len()
            )),
            evidence: pattern.instances.clone(),
            impact_score: (pattern.scale_count as f32 * pattern.instances.len() as f32 / 20.0)
                .clamp(0.0, 1.0),
            entities: Vec::new(), // fractal patterns span too many entities to list
        }
    }

    /// Get previously generated thoughts from the slow store (for surfacing in proactive_context)
    pub fn get_active_thoughts(store: &SlowStore, limit: usize) -> Result<Vec<Thought>> {
        let stored = store.get_active_thoughts(limit)?;

        let thoughts: Vec<Thought> = stored
            .into_iter()
            .map(|s| {
                let entities: Vec<(String, String)> =
                    serde_json::from_str(&s.entities_json).unwrap_or_default();
                let evidence: Vec<String> =
                    serde_json::from_str(&s.evidence_json).unwrap_or_default();

                Thought {
                    id: s.id,
                    kind: match s.kind.as_str() {
                        "missing_connection" => ThoughtKind::MissingConnection,
                        "structural_weakness" => ThoughtKind::StructuralWeakness,
                        "knowledge_silo" => ThoughtKind::KnowledgeSilo,
                        "golden_feature" => ThoughtKind::GoldenFeature,
                        "systematic_gap" => ThoughtKind::SystematicGap,
                        _ => ThoughtKind::MissingConnection,
                    },
                    scope: match s.scope.as_str() {
                        "codebase" => GapScope::Codebase,
                        "schema" => GapScope::Schema,
                        _ => GapScope::Content,
                    },
                    confidence: s.confidence,
                    description: s.description,
                    hypothesis: s.hypothesis,
                    evidence,
                    impact_score: s.impact_score,
                    entities,
                }
            })
            .collect();

        Ok(thoughts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_thought_engine_empty_graph() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        let config = GapDetectionConfig::default();
        let report = ThoughtEngine::generate(&store, &config).unwrap();
        assert_eq!(report.thoughts.len(), 0);
        assert_eq!(report.stats.gaps_detected, 0);
    }
}
