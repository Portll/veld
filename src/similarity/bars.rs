//! Fire-rate counters for the similarity bars.
//!
//! Every bar compares a cosine read on MiniLM's scale (see [`super::scale`]) with a threshold
//! tuned on MiniLM. A scale map or an embedder that moves a bar's fire rate changes what merges,
//! deduplicates or interferes long before a recall metric shows it, so each comparison counts
//! as evaluated, and as fired when it crosses its bar, under one closed label set
//! (`veld_similarity_bar_evaluated_total{bar}` and `veld_similarity_bar_fired_total{bar}`).

use std::sync::LazyLock;

use prometheus::IntCounter;

use crate::metrics::{SIMILARITY_BAR_EVALUATED_TOTAL, SIMILARITY_BAR_FIRED_TOTAL};

const BAR_COUNT: usize = 15;

/// One threshold comparison on a mapped cosine. A new bar is a new variant, so the label set
/// a dashboard reads stays closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimilarityBar {
    /// Entity concept merge on name embeddings (`ENTITY_CONCEPT_MERGE_THRESHOLD`).
    EntityConceptMerge,
    /// Dream replay's discovery band, lower edge (`DREAM_REPLAY_SIMILARITY_THRESHOLD`): fires
    /// when a pair is similar enough to be a candidate.
    DreamBandFloor,
    /// Dream replay's discovery band, upper edge (`DREAM_REPLAY_SIMILARITY_CEILING`): fires
    /// when a pair is too similar to count as a discovery.
    DreamBandCeiling,
    /// Fact dedup in `find_similar` (`FACT_DEDUP_COSINE_THRESHOLD`).
    FactDedupSimilar,
    /// Fact dedup in `find_contradiction`.
    FactDedupContradiction,
    /// Fact dedup in `find_contradictions`, the invalidation pass.
    FactDedupInvalidation,
    /// Semantic merge pass (`SEMANTIC_MERGE_SIMILARITY_THRESHOLD`).
    SemanticMerge,
    /// Proactive context reads the query as a repetition (similarity above 0.8).
    QueryRepetition,
    /// Proactive context reads the query as a topic change (similarity below 0.3).
    QueryTopicChange,
    /// Interference on a synchronous write (`INTERFERENCE_SIMILARITY_THRESHOLD`).
    InterferenceSync,
    /// Interference on a deferred write.
    InterferenceDeferred,
    /// A synchronous write's neighbour is a duplicate (`INTERFERENCE_SEVERE_THRESHOLD`).
    /// Evaluated only for neighbours past [`SimilarityBar::InterferenceSync`].
    InterferenceSevereSync,
    /// A deferred write's neighbour is a duplicate.
    InterferenceSevereDeferred,
    /// The write gate's nearest neighbour is an absorption candidate
    /// (`WRITE_GATE_ABSORPTION_THRESHOLD`). Deferred writes only.
    WriteGateCandidate,
    /// A write-gate candidate whose text adds nothing, absorbed. Evaluated once per candidate,
    /// so its fire rate is the share of candidates the text check lets through.
    WriteGateAbsorbed,
}

impl SimilarityBar {
    /// Every bar, in discriminant order.
    pub const ALL: [SimilarityBar; BAR_COUNT] = [
        SimilarityBar::EntityConceptMerge,
        SimilarityBar::DreamBandFloor,
        SimilarityBar::DreamBandCeiling,
        SimilarityBar::FactDedupSimilar,
        SimilarityBar::FactDedupContradiction,
        SimilarityBar::FactDedupInvalidation,
        SimilarityBar::SemanticMerge,
        SimilarityBar::QueryRepetition,
        SimilarityBar::QueryTopicChange,
        SimilarityBar::InterferenceSync,
        SimilarityBar::InterferenceDeferred,
        SimilarityBar::InterferenceSevereSync,
        SimilarityBar::InterferenceSevereDeferred,
        SimilarityBar::WriteGateCandidate,
        SimilarityBar::WriteGateAbsorbed,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SimilarityBar::EntityConceptMerge => "entity_concept_merge",
            SimilarityBar::DreamBandFloor => "dream_band_floor",
            SimilarityBar::DreamBandCeiling => "dream_band_ceiling",
            SimilarityBar::FactDedupSimilar => "fact_dedup_similar",
            SimilarityBar::FactDedupContradiction => "fact_dedup_contradiction",
            SimilarityBar::FactDedupInvalidation => "fact_dedup_invalidation",
            SimilarityBar::SemanticMerge => "semantic_merge",
            SimilarityBar::QueryRepetition => "query_repetition",
            SimilarityBar::QueryTopicChange => "query_topic_change",
            SimilarityBar::InterferenceSync => "interference_sync",
            SimilarityBar::InterferenceDeferred => "interference_deferred",
            SimilarityBar::InterferenceSevereSync => "interference_severe_sync",
            SimilarityBar::InterferenceSevereDeferred => "interference_severe_deferred",
            SimilarityBar::WriteGateCandidate => "write_gate_candidate",
            SimilarityBar::WriteGateAbsorbed => "write_gate_absorbed",
        }
    }

    /// Count one comparison and return `fired`, so the comparison stays where it was.
    pub fn record(self, fired: bool) -> bool {
        let (evaluated, fired_total) = &COUNTERS[self as usize];
        evaluated.inc();
        if fired {
            fired_total.inc();
        }
        fired
    }

    /// Count a loop's comparisons once, for loops that run thousands of comparisons per call.
    pub fn record_many(self, evaluated: u64, fired: u64) {
        if evaluated == 0 {
            return;
        }
        let (evaluated_total, fired_total) = &COUNTERS[self as usize];
        evaluated_total.inc_by(evaluated);
        fired_total.inc_by(fired);
    }

    /// Comparisons and fires counted so far in this process.
    pub fn totals(self) -> (u64, u64) {
        let (evaluated, fired) = &COUNTERS[self as usize];
        (evaluated.get(), fired.get())
    }
}

/// The write path an interference check runs on; each has its own pair of bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePath {
    Synchronous,
    Deferred,
}

impl WritePath {
    pub fn interference_bar(self) -> SimilarityBar {
        match self {
            WritePath::Synchronous => SimilarityBar::InterferenceSync,
            WritePath::Deferred => SimilarityBar::InterferenceDeferred,
        }
    }

    pub fn severe_bar(self) -> SimilarityBar {
        match self {
            WritePath::Synchronous => SimilarityBar::InterferenceSevereSync,
            WritePath::Deferred => SimilarityBar::InterferenceSevereDeferred,
        }
    }
}

// Resolved once, so a comparison costs an atomic add rather than a label lookup.
static COUNTERS: LazyLock<[(IntCounter, IntCounter); BAR_COUNT]> = LazyLock::new(|| {
    SimilarityBar::ALL.map(|bar| {
        (
            SIMILARITY_BAR_EVALUATED_TOTAL.with_label_values(&[bar.label()]),
            SIMILARITY_BAR_FIRED_TOTAL.with_label_values(&[bar.label()]),
        )
    })
});

/// Create every bar's series at zero, so a scrape sees each bar before its first comparison.
pub(crate) fn touch_all() {
    LazyLock::force(&COUNTERS);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_bar_in_discriminant_order_with_distinct_labels() {
        let mut labels = std::collections::HashSet::new();
        for (index, bar) in SimilarityBar::ALL.iter().enumerate() {
            assert_eq!(*bar as usize, index, "{bar:?} is out of order in ALL");
            assert!(labels.insert(bar.label()), "{bar:?} repeats a label");
        }
    }
}
