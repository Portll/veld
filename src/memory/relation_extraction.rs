//! Pattern-based relation extraction over the existing NER + RelationType
//! vocabulary. Turns free-text content into (subject, predicate, object)
//! triples that can be persisted as `RelationshipEdge`s in the knowledge
//! graph.
//!
//! # Scope
//!
//! This is an offline, local-first extractor — no LLM dependency. It uses
//! a curated set of surface patterns ("X works at Y", "X uses Y", ...) to
//! mine the high-precision/low-recall slice of the relation space. For the
//! long tail, pair it with the LLM-driven consolidator (item #10).
//!
//! Patterns are matched case-insensitively against full sentences. Subjects
//! and objects are bounded by sentence delimiters, conjunctions, or stop
//! words to keep span lengths short.
//!
//! # Output
//!
//! Each [`ExtractedRelation`] carries the surface forms of subject/object
//! plus the typed [`RelationType`] and a confidence in [0, 1]. Callers are
//! expected to resolve the surface strings to entity UUIDs (via the NER
//! results or graph membership) before persisting as `RelationshipEdge`.

use regex::Regex;
use std::sync::OnceLock;

use crate::graph_memory::RelationType;

/// A relation triple extracted from text.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedRelation {
    /// Surface form of the subject (left side of the predicate)
    pub subject: String,
    /// Mapped relation type from the graph vocabulary
    pub predicate: RelationType,
    /// Surface form of the object (right side of the predicate)
    pub object: String,
    /// Confidence in [0, 1] — derived from pattern specificity
    pub confidence: f32,
    /// The source sentence the triple was extracted from
    pub source_text: String,
}

/// One pattern in the extractor's catalog.
struct Pattern {
    /// Compiled regex with named `subject` and `object` capture groups
    regex: Regex,
    predicate: RelationType,
    /// Base confidence — higher for more specific patterns
    confidence: f32,
}

fn pattern_catalog() -> &'static [Pattern] {
    static CATALOG: OnceLock<Vec<Pattern>> = OnceLock::new();
    CATALOG.get_or_init(build_patterns)
}

fn build_patterns() -> Vec<Pattern> {
    // The subject / object groups are intentionally short (1-5 words, no
    // sentence terminators) so we don't grab the entire sentence.
    let np = r"(?P<subject>[A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+){0,4})";
    let obj = r"(?P<object>[A-Za-z0-9_\-][A-Za-z0-9_\-\.\s]{0,40}?)(?:[\.\,\;\!\?]|\s+(?:and|but|because|when|while|that|which)|$)";

    let raw_patterns: &[(&str, RelationType, f32)] = &[
        // Work patterns
        (&format!(r"{np}\s+works?\s+at\s+{obj}"), RelationType::WorksAt, 0.85),
        (&format!(r"{np}\s+(?:is|was)\s+employed\s+by\s+{obj}"), RelationType::EmployedBy, 0.9),
        (&format!(r"{np}\s+works?\s+(?:with|alongside)\s+{obj}"), RelationType::WorksWith, 0.8),

        // Structural
        (&format!(r"{np}\s+(?:is|are)\s+part\s+of\s+{obj}"), RelationType::PartOf, 0.85),
        (&format!(r"{np}\s+belongs?\s+to\s+{obj}"), RelationType::PartOf, 0.8),
        (&format!(r"{np}\s+contains?\s+{obj}"), RelationType::Contains, 0.8),
        (&format!(r"{np}\s+(?:is|are)\s+owned\s+by\s+{obj}"), RelationType::OwnedBy, 0.85),

        // Location
        (&format!(r"{np}\s+(?:is|are)\s+(?:located\s+)?in\s+{obj}"), RelationType::LocatedIn, 0.75),
        (&format!(r"{np}\s+(?:is|are)\s+(?:located\s+)?at\s+{obj}"), RelationType::LocatedAt, 0.75),

        // Usage / creation
        (&format!(r"{np}\s+(?:uses?|relies?\s+on|depends?\s+on)\s+{obj}"), RelationType::Uses, 0.8),
        (&format!(r"{np}\s+(?:was|were|is|are)\s+(?:created|built|made|developed)\s+by\s+{obj}"), RelationType::CreatedBy, 0.85),
        (&format!(r"{np}\s+(?:created|built|made|developed)\s+{obj}"), RelationType::DevelopedBy, 0.8),

        // Causal
        (&format!(r"{np}\s+causes?\s+{obj}"), RelationType::Causes, 0.85),
        (&format!(r"{np}\s+(?:leads?\s+to|results?\s+in)\s+{obj}"), RelationType::ResultsIn, 0.8),

        // Learning
        (&format!(r"{np}\s+(?:learned|learnt)\s+{obj}"), RelationType::Learned, 0.8),
        (&format!(r"{np}\s+(?:knows?|understands?)\s+{obj}"), RelationType::Knows, 0.7),
        (&format!(r"{np}\s+teaches?\s+{obj}"), RelationType::Teaches, 0.8),
    ];

    raw_patterns
        .iter()
        .filter_map(|(pat, predicate, confidence)| {
            Regex::new(&format!(r"(?i){pat}"))
                .ok()
                .map(|regex| Pattern {
                    regex,
                    predicate: predicate.clone(),
                    confidence: *confidence,
                })
        })
        .collect()
}

/// Split text into sentences using simple terminator boundaries. Strips
/// surrounding whitespace and discards empty fragments.
fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['.', '!', '?', '\n'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() > 4)
        .collect()
}

/// Extract all relation triples from `text`.
///
/// Each sentence is scanned against the pattern catalog; matches yield
/// `ExtractedRelation`s with subject/object surface forms, the typed
/// predicate, and a pattern-derived confidence.
pub fn extract_relations(text: &str) -> Vec<ExtractedRelation> {
    let mut out = Vec::new();
    let catalog = pattern_catalog();

    for sentence in split_sentences(text) {
        for pattern in catalog {
            for caps in pattern.regex.captures_iter(sentence) {
                let subject = caps.name("subject").map(|m| m.as_str().trim()).unwrap_or("");
                let object = caps.name("object").map(|m| m.as_str().trim()).unwrap_or("");
                if subject.is_empty() || object.is_empty() {
                    continue;
                }
                // Skip near-self-referential triples (subject == object)
                if subject.eq_ignore_ascii_case(object) {
                    continue;
                }
                out.push(ExtractedRelation {
                    subject: subject.to_string(),
                    predicate: pattern.predicate.clone(),
                    object: object.to_string(),
                    confidence: pattern.confidence,
                    source_text: sentence.to_string(),
                });
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracts_works_at() {
        let r = extract_relations("Alice works at Acme Corp.");
        assert!(r.iter().any(|x| {
            x.predicate == RelationType::WorksAt
                && x.subject.contains("Alice")
                && x.object.to_lowercase().contains("acme")
        }), "expected WorksAt triple, got {:?}", r);
    }

    #[test]
    fn test_extracts_uses() {
        let r = extract_relations("The auth module uses JWT tokens.");
        assert!(
            r.iter().any(|x| x.predicate == RelationType::Uses),
            "expected Uses triple, got {:?}",
            r
        );
    }

    #[test]
    fn test_extracts_created_by() {
        let r = extract_relations("Rust was created by Mozilla.");
        assert!(
            r.iter().any(|x| x.predicate == RelationType::CreatedBy
                && x.object.to_lowercase().contains("mozilla")),
            "expected CreatedBy with Mozilla object, got {:?}",
            r
        );
    }

    #[test]
    fn test_extracts_causes() {
        let r = extract_relations("Memory leaks cause crashes.");
        assert!(
            r.iter().any(|x| x.predicate == RelationType::Causes),
            "expected Causes triple, got {:?}",
            r
        );
    }

    #[test]
    fn test_skips_self_reference() {
        let r = extract_relations("Bob works with Bob.");
        // Either no triple or no self-reference triple
        assert!(
            !r.iter().any(|x| x.subject.eq_ignore_ascii_case(&x.object)),
            "unexpected self-referential triple in {:?}",
            r
        );
    }

    #[test]
    fn test_no_extraction_on_empty_or_short() {
        assert!(extract_relations("").is_empty());
        assert!(extract_relations("Hi.").is_empty());
    }

    #[test]
    fn test_multiple_sentences() {
        let r = extract_relations("Alice works at Acme. Bob uses Linux.");
        assert!(r.len() >= 2, "expected ≥2 triples across two sentences, got {:?}", r);
    }
}
