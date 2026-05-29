//! R43 — apply REM-mode graph edge proposals to the user's `GraphMemory`.
//!
//! The rewriter emits [`EdgeProposalDraft`]s alongside observations in REM
//! mode. The worker, after persisting observations and recording
//! supersessions, calls [`apply_edge_proposals`] which:
//!
//!   1. Resolves both `from_entity` / `to_entity` names to existing
//!      [`EntityNode::uuid`]s via `find_entity_by_name` (R54 — fabricated
//!      entities are silently dropped).
//!   2. Maps the LLM's `relation` string to a [`RelationType`] variant
//!      (unknown values default to `RelationType::CoOccurs`).
//!   3. Constructs a [`RelationshipEdge`] at [`EdgeTier::L1Working`] so the
//!      proposal must earn promotion through normal Hebbian co-activation
//!      — REM proposals never enter the graph as already-trusted L2/L3.
//!   4. Adds it to the graph via `add_relationship`, which handles
//!      idempotency (existing edges get strength bumps instead of new rows).
//!
//! Returns counts for telemetry — applied / dropped-validation / errors.

use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

use super::types::EdgeProposalDraft;
use crate::graph_memory::{
    EdgeSource, EdgeTier, GraphMemory, LtpStatus, RelationType, RelationshipEdge,
};

/// Summary of an edge-proposal application pass — one per REM-mode rewrite.
#[derive(Debug, Default, Clone)]
pub struct EdgeApplicationResult {
    pub applied: usize,
    pub dropped_unresolved_entity: usize,
    pub dropped_self_loop: usize,
    pub errors: usize,
}

/// Apply a batch of edge proposals to the user's graph.
///
/// `graph` is an exclusive write lock held for the duration of the apply —
/// we batch all proposals from one REM pass under a single lock acquisition
/// so other foreground readers see either zero or all of the new edges.
pub fn apply_edge_proposals(
    graph: &Arc<parking_lot::RwLock<GraphMemory>>,
    proposals: &[EdgeProposalDraft],
) -> Result<EdgeApplicationResult> {
    if proposals.is_empty() {
        return Ok(EdgeApplicationResult::default());
    }

    let mut out = EdgeApplicationResult::default();
    let g = graph.write();

    for p in proposals {
        // R54: name resolution.
        let from = match g.find_entity_by_name(&p.from_entity) {
            Ok(Some(node)) => node,
            Ok(None) => {
                out.dropped_unresolved_entity += 1;
                tracing::debug!(
                    from_entity = %p.from_entity,
                    "edge-proposal: from_entity not in graph; dropped"
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    from_entity = %p.from_entity,
                    error = %e,
                    "edge-proposal: from_entity lookup error"
                );
                out.errors += 1;
                continue;
            }
        };
        let to = match g.find_entity_by_name(&p.to_entity) {
            Ok(Some(node)) => node,
            Ok(None) => {
                out.dropped_unresolved_entity += 1;
                tracing::debug!(
                    to_entity = %p.to_entity,
                    "edge-proposal: to_entity not in graph; dropped"
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    to_entity = %p.to_entity,
                    error = %e,
                    "edge-proposal: to_entity lookup error"
                );
                out.errors += 1;
                continue;
            }
        };

        if from.uuid == to.uuid {
            out.dropped_self_loop += 1;
            continue;
        }

        let relation = parse_relation(&p.relation);
        let edge = make_l1_edge(from.uuid, to.uuid, relation, p.confidence);

        match g.add_relationship(edge) {
            Ok(_) => out.applied += 1,
            Err(e) => {
                tracing::warn!(
                    from = %from.uuid,
                    to = %to.uuid,
                    error = %e,
                    "edge-proposal: add_relationship failed"
                );
                out.errors += 1;
            }
        }
    }

    if out.applied > 0
        || out.dropped_unresolved_entity > 0
        || out.errors > 0
        || out.dropped_self_loop > 0
    {
        tracing::debug!(
            applied = out.applied,
            dropped_unresolved = out.dropped_unresolved_entity,
            dropped_self_loop = out.dropped_self_loop,
            errors = out.errors,
            "REM edge-proposal application complete"
        );
    }

    Ok(out)
}

/// Parse the LLM-emitted relation string into a [`RelationType`]. Unknown
/// values default to [`RelationType::CoOccurs`] — the conservative choice
/// that conveys "these appeared together" without claiming a specific
/// semantic relationship.
fn parse_relation(s: &str) -> RelationType {
    match s.trim().to_ascii_lowercase().as_str() {
        "co_occurs" | "cooccurs" | "co-occurs" => RelationType::CoOccurs,
        "related_to" | "related-to" | "relatedto" => RelationType::RelatedTo,
        "associated_with" | "associated-with" | "associatedwith" => RelationType::AssociatedWith,
        "co_retrieved" | "coretrieved" | "co-retrieved" => RelationType::CoRetrieved,
        "works_with" | "workswith" => RelationType::WorksWith,
        "part_of" | "partof" => RelationType::PartOf,
        "uses" => RelationType::Uses,
        "depends_on" | "dependson" => RelationType::DependsOn,
        "manages" => RelationType::Manages,
        "prefers" => RelationType::Prefers,
        "knows" => RelationType::Knows,
        _ => RelationType::CoOccurs,
    }
}

fn make_l1_edge(
    from_uuid: Uuid,
    to_uuid: Uuid,
    relation_type: RelationType,
    confidence: f32,
) -> RelationshipEdge {
    let now = Utc::now();
    let tier = EdgeTier::L1Working;
    let initial = tier.initial_weight();
    // Use the LLM-asserted confidence to scale the initial weight, bounded
    // by the tier's normal initial weight on the upper end. This prevents
    // a high-confidence REM proposal from skipping straight to a
    // promotion-eligible weight.
    let strength = (initial * confidence.clamp(0.5, 1.0)).clamp(0.0, initial);
    let clamped_conf = confidence.clamp(0.0, 1.0);
    RelationshipEdge {
        uuid: Uuid::new_v4(),
        from_entity: from_uuid,
        to_entity: to_uuid,
        relation_type,
        strength,
        created_at: now,
        valid_at: now,
        invalidated_at: None,
        source_episode_id: None,
        context: "sleep_time::rem_proposal".to_string(),
        last_activated: now,
        activation_count: 0,
        ltp_status: LtpStatus::default(),
        tier,
        activation_timestamps: None,
        entity_confidence: Some(clamped_conf),
        created_by: EdgeSource::default(),
        forward_strength: strength,
        backward_strength: strength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_relation_co_occurs_default() {
        assert_eq!(parse_relation("co_occurs"), RelationType::CoOccurs);
        assert_eq!(parse_relation("CoOccurs"), RelationType::CoOccurs);
        assert_eq!(parse_relation("co-occurs"), RelationType::CoOccurs);
    }

    #[test]
    fn parse_relation_unknown_falls_back_to_co_occurs() {
        assert_eq!(parse_relation("not_a_real_relation"), RelationType::CoOccurs);
        assert_eq!(parse_relation(""), RelationType::CoOccurs);
    }

    #[test]
    fn parse_relation_known_variants() {
        assert_eq!(parse_relation("manages"), RelationType::Manages);
        assert_eq!(parse_relation("depends_on"), RelationType::DependsOn);
        assert_eq!(parse_relation("uses"), RelationType::Uses);
        assert_eq!(parse_relation("PREFERS"), RelationType::Prefers);
    }

    #[test]
    fn make_l1_edge_clamps_strength_to_tier_initial() {
        let e = make_l1_edge(
            Uuid::new_v4(),
            Uuid::new_v4(),
            RelationType::CoOccurs,
            2.0, // > 1.0; clamped to [0.5, 1.0]
        );
        assert_eq!(e.tier, EdgeTier::L1Working);
        let max = EdgeTier::L1Working.initial_weight();
        assert!(e.strength <= max);
        // confidence is clamped to [0.5, 1.0] before multiplying by initial,
        // so the actual strength = initial * 1.0 = initial.
        assert!((e.strength - max).abs() < 1e-5);
    }

    #[test]
    fn make_l1_edge_scales_with_low_confidence() {
        let e = make_l1_edge(
            Uuid::new_v4(),
            Uuid::new_v4(),
            RelationType::CoOccurs,
            0.6,
        );
        let initial = EdgeTier::L1Working.initial_weight();
        // strength = initial * 0.6
        assert!((e.strength - initial * 0.6).abs() < 1e-5);
    }
}
