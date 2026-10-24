//! The write gate's duplicate test.
//!
//! Decided on the text alone, so it means the same thing under every embedder: a cosine bar
//! that marks a near-copy under MiniLM marks a quarter of all conversation turns under nomic.
//! Only an exact duplicate is absorbed into the memory it repeats. A near-exact one is kept,
//! tagged as a duplicate of that memory, and recall ranks it below it. Anything else is
//! distinct, however much of its wording another memory shares.

use std::collections::HashSet;
use std::sync::Arc;

use crate::constants::{
    NEAR_DUPLICATE_LENGTH_RATIO, NEAR_DUPLICATE_MIN_TOKENS, NEAR_DUPLICATE_TOKEN_JACCARD,
    WRITE_GATE_DUPLICATE_RECALL_FACTOR,
};
use crate::memory::types::SharedMemory;

/// Metadata key naming the memory a duplicate repeats.
pub const DUPLICATE_OF_KEY: &str = "duplicate_of";

/// Metadata key holding [`DuplicateKind::label`].
pub const DUPLICATE_KIND_KEY: &str = "duplicate_kind";

/// Tag added to a memory the write gate kept as a near-exact duplicate.
pub const DUPLICATE_TAG: &str = "duplicate";

/// How a new memory's text relates to an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DuplicateKind {
    /// Anything else, however close the embeddings.
    Distinct,
    /// The same words ignoring case and punctuation, or a token overlap of at least
    /// [`NEAR_DUPLICATE_TOKEN_JACCARD`] at a similar length.
    NearExact,
    /// The same text once runs of whitespace are collapsed.
    Exact,
}

impl DuplicateKind {
    pub fn label(self) -> &'static str {
        match self {
            DuplicateKind::Exact => "exact",
            DuplicateKind::NearExact => "near-exact",
            DuplicateKind::Distinct => "distinct",
        }
    }
}

/// Classify `candidate` against `existing`.
pub fn classify(candidate: &str, existing: &str) -> DuplicateKind {
    if collapse_whitespace(candidate) == collapse_whitespace(existing) {
        return DuplicateKind::Exact;
    }
    let a = word_tokens(candidate);
    let b = word_tokens(existing);
    if a.is_empty() || b.is_empty() {
        return DuplicateKind::Distinct;
    }
    if a == b {
        return DuplicateKind::NearExact;
    }
    let (shorter, longer) = if a.len() <= b.len() {
        (a.len(), b.len())
    } else {
        (b.len(), a.len())
    };
    if shorter < NEAR_DUPLICATE_MIN_TOKENS
        || (shorter as f32) < NEAR_DUPLICATE_LENGTH_RATIO * longer as f32
    {
        return DuplicateKind::Distinct;
    }
    let set_a: HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: HashSet<&str> = b.iter().map(String::as_str).collect();
    let union = set_a.union(&set_b).count();
    let shared = set_a.intersection(&set_b).count();
    if shared as f32 / union as f32 >= NEAR_DUPLICATE_TOKEN_JACCARD {
        DuplicateKind::NearExact
    } else {
        DuplicateKind::Distinct
    }
}

/// Scale the score of every memory tagged as a near-exact duplicate by
/// [`WRITE_GATE_DUPLICATE_RECALL_FACTOR`] and re-sort by score, highest first, so a copy
/// ranks below what it repeats. Returns whether any memory was demoted; the order is left
/// alone when none was.
pub fn derank(memories: &mut [SharedMemory]) -> bool {
    let mut any_demoted = false;
    for memory in memories.iter_mut() {
        if memory.experience.metadata.contains_key(DUPLICATE_OF_KEY) {
            let mut demoted = memory.as_ref().clone();
            demoted.set_score(demoted.score.unwrap_or(0.0) * WRITE_GATE_DUPLICATE_RECALL_FACTOR);
            *memory = Arc::new(demoted);
            any_demoted = true;
        }
    }
    if any_demoted {
        memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
    }
    any_demoted
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn word_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{Experience, Memory, MemoryId};

    #[test]
    fn whitespace_alone_makes_an_exact_duplicate() {
        assert_eq!(
            classify(
                "Nate:  I've had them\nfor 3 years.",
                "Nate: I've had them for 3 years."
            ),
            DuplicateKind::Exact
        );
    }

    #[test]
    fn case_and_punctuation_alone_make_a_near_exact_duplicate() {
        assert_eq!(
            classify("Nate: Thanks!", "nate: thanks."),
            DuplicateKind::NearExact
        );
        assert_eq!(
            classify("Deploy failed, rolled back.", "deploy failed - rolled back"),
            DuplicateKind::NearExact
        );
    }

    #[test]
    fn one_changed_word_in_a_long_memory_is_near_exact() {
        let a = "Refactored the storage layer from twenty separate RocksDB instances down to \
                 three each opened with column families so file descriptor pressure dropped \
                 sharply";
        let b = "Refactored the storage layer from twenty separate RocksDB instances down to \
                 three each opened with column families so file descriptor pressure dropped \
                 steeply";
        assert_eq!(classify(a, b), DuplicateKind::NearExact);
    }

    #[test]
    fn one_changed_word_in_a_short_memory_is_distinct() {
        assert_eq!(
            classify("Thanks Nate", "Thanks Jo"),
            DuplicateKind::Distinct
        );
        assert_eq!(
            classify(
                "Joanna: I watched Little Women last night",
                "Joanna: I watched Little Women last week"
            ),
            DuplicateKind::Distinct
        );
        assert_eq!(
            classify("Item 10: the deploy failed", "Item 11: the deploy failed"),
            DuplicateKind::Distinct
        );
    }

    #[test]
    fn reordered_words_are_distinct_unless_long_enough_to_overlap() {
        assert_eq!(
            classify("alice owns billing", "billing owns alice"),
            DuplicateKind::Distinct
        );
    }

    #[test]
    fn a_memory_contained_in_a_longer_one_is_distinct() {
        let short = "the deploy failed on the migration step today";
        let long = "the deploy failed on the migration step today and again tonight after the \
                    rollback because the lock table was still held";
        assert_eq!(classify(short, long), DuplicateKind::Distinct);
        assert_eq!(
            classify("the deploy failed", "the deploy failed at 3pm on Tuesday"),
            DuplicateKind::Distinct
        );
    }

    #[test]
    fn semantically_close_paraphrases_are_distinct() {
        assert_eq!(
            classify(
                "Nate: I've had them for 3 years now and they bring me tons of joy!",
                "Nate: My turtles have been with me for three years and I love them."
            ),
            DuplicateKind::Distinct
        );
    }

    #[test]
    fn empty_or_symbol_only_text_is_never_a_duplicate_of_words() {
        assert_eq!(
            classify("!!!", "hello there friend"),
            DuplicateKind::Distinct
        );
        assert_eq!(classify("", "hello"), DuplicateKind::Distinct);
    }

    #[test]
    fn kinds_order_from_distinct_to_exact() {
        assert!(DuplicateKind::Exact > DuplicateKind::NearExact);
        assert!(DuplicateKind::NearExact > DuplicateKind::Distinct);
    }

    fn scored(content: &str, score: f32, duplicate_of: Option<&Memory>) -> SharedMemory {
        let mut memory = Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                content: content.to_string(),
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            None,
        );
        if let Some(original) = duplicate_of {
            memory
                .experience
                .metadata
                .insert(DUPLICATE_OF_KEY.to_string(), original.id.0.to_string());
        }
        memory.set_score(score);
        Arc::new(memory)
    }

    #[test]
    fn a_tagged_duplicate_scoring_just_above_its_original_ranks_below_it() {
        let original = scored("the crane log shows a hydraulic fault", 0.80, None);
        let copy = scored(
            "The crane log shows a hydraulic fault.",
            0.85,
            Some(&original),
        );
        let unrelated = scored("berth four reopened", 0.60, None);
        let mut ranked = vec![
            Arc::clone(&copy),
            Arc::clone(&original),
            Arc::clone(&unrelated),
        ];

        assert!(derank(&mut ranked));

        let order: Vec<&MemoryId> = ranked.iter().map(|m| &m.id).collect();
        assert_eq!(order, vec![&original.id, &copy.id, &unrelated.id]);
        assert!(
            (ranked[1].score.unwrap() - 0.85 * WRITE_GATE_DUPLICATE_RECALL_FACTOR).abs() < 1e-6
        );
        assert_eq!(
            ranked[0].score,
            Some(0.80),
            "an untagged memory keeps its score"
        );
    }

    #[test]
    fn without_a_tagged_duplicate_the_order_is_left_alone() {
        let a = scored("first", 0.2, None);
        let b = scored("second", 0.9, None);
        let mut ranked = vec![Arc::clone(&a), Arc::clone(&b)];
        assert!(!derank(&mut ranked));
        assert_eq!(ranked[0].id, a.id);
    }
}
