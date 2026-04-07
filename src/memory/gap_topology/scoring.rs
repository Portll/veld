//! Impact scoring for detected gaps.
//!
//! Computes how many other gaps each gap shares entities with,
//! measuring the cascading effect of closing a gap.

use std::collections::{HashMap, HashSet};

use super::GapTopology;

/// Compute impact scores: how many other gaps share entities with this one?
///
/// A gap with high impact participates in many structural problems.
/// Closing it would cascade through the graph and resolve multiple gaps.
pub fn compute_impact_scores(gaps: &mut [GapTopology]) {
    // Build entity -> gap index
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
