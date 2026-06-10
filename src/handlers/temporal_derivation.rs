//! Shared B4 temporal answer-derivation logic.
//!
//! Both `/api/query/execute` and `/api/recall` surface a `derived_answer` field.
//! This module owns the wire types and the derivation function so neither handler
//! duplicates the logic.
//!
//! The entry-point is [`derive_answer_for_user_query`]; callers supply the
//! `MultiUserMemoryManager`, a tenant `user_id`, and the raw query string.
//! The result is `None` (common case) for non-temporal queries or when no date
//! is computable — the caller's response is then byte-identical to the pre-B4
//! shape.

use std::sync::Arc;

use serde::Serialize;

use super::state::MultiUserMemoryManager;

/// A temporal fact projected to the wire (B4 supporting evidence).
///
/// Mirrors the load-bearing fields of
/// [`crate::memory::temporal_facts::TemporalFact`] so the answer-derivation
/// surface can justify a derived date without leaking internal storage shapes.
#[derive(Debug, Serialize)]
pub struct TemporalFactWire {
    pub entity: String,
    pub event: String,
    pub resolved_time: String,
    pub source_memory_id: String,
    pub confidence: f32,
    pub source_text: String,
}

/// A derived answer to a "when did/is X?" temporal question (B4).
///
/// Purely additive: `results` / `memories` are unaffected, and the field is
/// `None` whenever no date is derivable (non-temporal queries, or temporal
/// queries with no computable date).
#[derive(Debug, Serialize)]
pub struct DerivedAnswer {
    /// ISO `YYYY-MM-DD` (anchor paths) or the fact's resolved-time repr
    /// (temporal-fact path, may be coarser e.g. `2023-06`).
    pub derived_date: String,
    /// `"anchor_relative"` | `"event_anchor"` | `"temporal_fact"`.
    pub method: String,
    /// Stored facts that justified the derived date (empty for pure
    /// query-computed anchor-relative answers).
    pub supporting_facts: Vec<TemporalFactWire>,
    /// Source memory ids backing the supporting facts.
    pub source_memories: Vec<String>,
}

/// Run B4 temporal answer-derivation for a raw query string.
///
/// Returns `None` (the common case) when:
/// - the query is not a "when did/is X?" question,
/// - no date can be computed from anchors or stored temporal facts, or
/// - the tenant's memory substrate is unavailable.
///
/// Never touches retrieval/ranking; only reads temporal facts and the event
/// anchor index.
pub fn derive_answer_for_user_query(
    state: &Arc<MultiUserMemoryManager>,
    user_id: &str,
    query_text: &str,
) -> Option<DerivedAnswer> {
    use crate::memory::query_parser::{asks_for_temporal_answer, derive_temporal_answer};

    // Cheap gate first — avoids touching the substrate for non-temporal queries.
    if !asks_for_temporal_answer(query_text) {
        return None;
    }

    let user_memory = state.get_user_memory(user_id).ok()?;
    let mem = user_memory.read();
    let user_id_owned = user_id.to_owned();

    // Event-anchor resolver (B2): "the Monday after the wedding".
    let event_resolver = |event_phrase: &str| -> Option<chrono::NaiveDate> {
        mem.resolve_event_to_date(&user_id_owned, event_phrase)
    };

    // Temporal-fact lookup: collect candidate facts whose event keywords appear
    // in the query. Reuses the same event index `resolve_event_to_date` walks.
    let fact_lookup = |q: &str| -> Vec<crate::memory::temporal_facts::TemporalFact> {
        let stopwords = [
            "when", "did", "does", "is", "was", "were", "the", "a", "an", "of", "to", "do",
            "what", "date", "day", "time", "on", "in", "at", "go", "have", "has", "and", "or",
            "her", "his", "their", "they", "she", "he",
        ];
        let mut seen = std::collections::HashSet::new();
        let mut facts = Vec::new();
        for word in q
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .filter(|w| !stopwords.contains(&w.to_lowercase().as_str()))
        {
            if let Ok(found) = mem.find_temporal_facts_by_event(&user_id_owned, word, 10) {
                for f in found {
                    if seen.insert(f.id.clone()) {
                        facts.push(f);
                    }
                }
            }
        }
        facts
    };

    let derived = derive_temporal_answer(query_text, Some(&event_resolver), Some(&fact_lookup))?;

    // Project the pure derivation result onto the wire type.
    let supporting_facts: Vec<TemporalFactWire> = derived
        .supporting_facts
        .iter()
        .map(|f| TemporalFactWire {
            entity: f.entity.clone(),
            event: f.event.clone(),
            resolved_time: f.resolved_time.to_sortable_string(),
            source_memory_id: f.source_memory_id.0.to_string(),
            confidence: f.confidence,
            source_text: f.source_text.clone(),
        })
        .collect();
    let source_memories: Vec<String> = derived
        .source_memories
        .iter()
        .map(|m| m.0.to_string())
        .collect();

    Some(DerivedAnswer {
        derived_date: derived.derived_date,
        method: derived.method.as_str().to_string(),
        supporting_facts,
        source_memories,
    })
}

#[cfg(test)]
mod tests {
    use crate::memory::query_parser::asks_for_temporal_answer;

    /// Verify the temporal-gate function correctly identifies questions that
    /// warrant derivation vs. queries that should produce None.
    #[test]
    fn temporal_gate_passes_when_queries() {
        assert!(asks_for_temporal_answer("when did Melanie go camping?"));
        assert!(asks_for_temporal_answer("When was the project started?"));
        assert!(asks_for_temporal_answer("what date did they get married?"));
    }

    #[test]
    fn temporal_gate_blocks_non_temporal_queries() {
        assert!(!asks_for_temporal_answer(
            "What did Melanie think about camping?"
        ));
        assert!(!asks_for_temporal_answer("Tell me about the project"));
        assert!(!asks_for_temporal_answer(""));
    }
}
