//! PII-aware policies for collective knowledge aggregation.

use crate::graph_memory::{EdgeTier, EntityLabel, EntityNode, PiiClassification, RelationshipEdge};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PiiSensitivity {
    None,
    QuasiIdentifier,
    Direct,
    Sensitive,
}

fn sensitivity_from_classification(classification: &PiiClassification) -> PiiSensitivity {
    match classification {
        PiiClassification::PersonalIdentity => PiiSensitivity::Direct,
        PiiClassification::OrganizationIdentity => PiiSensitivity::QuasiIdentifier,
        PiiClassification::Clean => PiiSensitivity::None,
    }
}

fn sensitivity_from_labels(labels: &[EntityLabel]) -> PiiSensitivity {
    labels.iter().fold(PiiSensitivity::None, |current, label| {
        let sensitivity = match label {
            EntityLabel::Person => PiiSensitivity::Direct,
            EntityLabel::Organization | EntityLabel::Location => PiiSensitivity::QuasiIdentifier,
            _ => PiiSensitivity::None,
        };
        current.max(sensitivity)
    })
}

pub fn classify_entity_pii(entity: &EntityNode) -> PiiSensitivity {
        sensitivity_from_classification(&entity.pii_classification)
            .max(sensitivity_from_labels(&entity.labels))
}

pub fn is_edge_safe_for_collective(
    edge: &RelationshipEdge,
    source_entity: &EntityNode,
    target_entity: &EntityNode,
) -> bool {
    if edge.invalidated_at.is_some() || edge.tier != EdgeTier::L3Semantic {
        return false;
    }

    let source_pii = classify_entity_pii(source_entity);
    let target_pii = classify_entity_pii(target_entity);

    !matches!(source_pii, PiiSensitivity::Direct | PiiSensitivity::Sensitive)
        && !matches!(target_pii, PiiSensitivity::Direct | PiiSensitivity::Sensitive)
}

pub fn pii_aware_decay_factor(source_entity: &EntityNode, target_entity: &EntityNode) -> f32 {
    let max_pii = classify_entity_pii(source_entity).max(classify_entity_pii(target_entity));
    match max_pii {
        PiiSensitivity::None => 1.0,
        PiiSensitivity::QuasiIdentifier => 0.7,
        PiiSensitivity::Direct => 0.4,
        PiiSensitivity::Sensitive => 0.2,
    }
}