//! Which entities survive `MAX_ENTITIES_PER_MEMORY`, and why it is not the first fifty.
//!
//! # The measurement this exists because of
//!
//! `merged_entities` is appended in extraction order and the cap was `truncate(50)`, so the cut was
//! by ARRIVAL. Measured 2026-08-27 over 397 memories with a tail (`docs/ENTITY-TAIL-FINDINGS`),
//! median `mention_count` by position in the list:
//!
//! | bucket | index | median mentions | on 100+ memories |
//! |---|---|---|---|
//! | head | 0-6 | **1,977** | 73.1% |
//! | mid | 7-24 | 115 | 52.0% |
//! | tail | 25+ | **106** | 50.9% |
//!
//! The discarded tail is **nineteen times more selective** than the head the cap kept. Arrival order
//! is uncorrelated with worth, so on the 67.2% of records that hit the cap it was keeping
//! near-zero-information scope markers and discarding the distinctive material.
//!
//! # What this changes, and what it does not claim
//!
//! Authored entities — the writer's own tags — are kept unconditionally and in order. They are
//! high-mention BY DESIGN (`scope:claude-code` is on thousands of records) because they exist to
//! FILTER, not to retrieve, and ranking them by selectivity would delete the one thing on the record
//! that was deliberate.
//!
//! Derived entities, which nobody chose, are ordered by ascending `mention_count` before the cut, so
//! the rarest survive. A never-seen entity sorts first: it carries the most information of anything
//! in the list, and if it turns out to be a typo it costs one slot, where a generic word that
//! appears on 2,000 records costs a slot AND cannot distinguish anything.
//!
//! **The claim is bounded.** The evidence says arrival order is uncorrelated with worth and that
//! selectivity ordering strictly dominates it on that axis. It does NOT say retrieval measurably
//! improves — that needs measuring after the fact, and `CapReport` exists so the change is
//! observable rather than assumed.

use std::collections::HashMap;

/// What the cap did, so the reordering is visible instead of silent.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CapReport {
    /// Candidates before the cut.
    pub considered: usize,
    /// Authored entries kept unconditionally.
    pub authored: usize,
    /// Derived entries dropped.
    pub dropped: usize,
    /// Derived entries that survived ONLY because of selectivity ordering — they sat past the cap
    /// in arrival order. This is the number that says whether the change did anything at all; a
    /// persistent zero means the ordering is inert and should be questioned rather than trusted.
    pub rescued_by_ordering: usize,
}

/// Cap `entities` to `cap`, keeping the leading `authored` entries and ordering the rest by
/// ascending mention count.
///
/// `mentions` maps entity name to how many memories already carry it. A name absent from the map is
/// treated as novel (0) — the most selective thing it could be.
pub fn cap_by_selectivity(
    entities: Vec<String>,
    authored: usize,
    cap: usize,
    mentions: &HashMap<String, usize>,
) -> (Vec<String>, CapReport) {
    let considered = entities.len();
    let authored = authored.min(entities.len()).min(cap);

    if considered <= cap {
        return (
            entities,
            CapReport {
                considered,
                authored,
                dropped: 0,
                rescued_by_ordering: 0,
            },
        );
    }

    let (head, tail) = entities.split_at(authored);
    let kept_head: Vec<String> = head.to_vec();

    // Arrival order is recorded before sorting so `rescued_by_ordering` can be computed. Without
    // it this function would have no way to say whether it changed anything, and a reordering
    // nobody can measure is indistinguishable from one that never fires.
    let arrival_survivors: Vec<&String> = tail.iter().take(cap - authored).collect();

    let mut derived: Vec<(usize, &String)> = tail.iter().enumerate().collect();
    // Ascending mentions; ties fall back to arrival, so the sort is total and deterministic.
    derived.sort_by(|(ia, a), (ib, b)| {
        let ma = mentions.get(a.as_str()).copied().unwrap_or(0);
        let mb = mentions.get(b.as_str()).copied().unwrap_or(0);
        ma.cmp(&mb).then_with(|| ia.cmp(ib))
    });

    let room = cap - authored;
    let kept_derived: Vec<String> = derived
        .iter()
        .take(room)
        .map(|(_, s)| (*s).clone())
        .collect();

    let rescued = kept_derived
        .iter()
        .filter(|s| !arrival_survivors.iter().any(|a| *a == *s))
        .count();

    let mut out = kept_head;
    out.extend(kept_derived);
    let report = CapReport {
        considered,
        authored,
        dropped: considered - out.len(),
        rescued_by_ordering: rescued,
    };
    (out, report)
}

/// Whether a derived tag can name something: two characters, a letter among them, and none of
/// the characters that mark a code fragment, a metric label or a number list. On the live store
/// (2026-10-24) `veld_recall_stage_duration_seconds{stage="post`, `6.8(f` and `2,041` were all
/// tags. A commit hash, a version such as `v0.8.4` and `R@10` pass; a bare date or a line range
/// does not, and dates already live in `temporal_refs`.
pub fn derived_tag_is_well_formed(tag: &str) -> bool {
    tag.chars().count() >= 2
        && tag.chars().any(char::is_alphabetic)
        && !tag
            .chars()
            .any(|c| matches!(c, '{' | '}' | '(' | ')' | '[' | ']' | '"' | ',' | ';' | '='))
}

/// Keep the authored head as it is and drop the malformed derived entries. Returns the list and
/// the number dropped. Authored tags are never judged: the writer chose them.
pub fn well_formed_derived(entities: Vec<String>, authored: usize) -> (Vec<String>, usize) {
    let authored = authored.min(entities.len());
    let before = entities.len();
    let mut kept: Vec<String> = entities[..authored].to_vec();
    kept.extend(
        entities[authored..]
            .iter()
            .filter(|t| derived_tag_is_well_formed(t))
            .cloned(),
    );
    let malformed = before - kept.len();
    (kept, malformed)
}

/// The list length at which the derived tail is capped: the authored count plus the derived
/// budget, never above the record-wide maximum.
pub fn derived_cap(authored: usize, budget: usize, max_total: usize) -> usize {
    authored.saturating_add(budget).min(max_total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    }
    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn under_the_cap_nothing_is_reordered_or_dropped() {
        let input = v(&["a", "b", "c"]);
        let (out, rep) = cap_by_selectivity(input.clone(), 1, 10, &m(&[("c", 9999)]));
        assert_eq!(
            out, input,
            "no cut means no reordering — the record is left alone"
        );
        assert_eq!(rep.dropped, 0);
        assert_eq!(rep.rescued_by_ordering, 0);
    }

    #[test]
    fn authored_entries_survive_however_generic_they_are() {
        // The whole reason authored and derived are treated differently. `scope:claude-code` is on
        // thousands of records and would sort last on selectivity — but it is the one thing on the
        // record somebody chose, and it exists to filter rather than to retrieve.
        let input = v(&["scope:claude-code", "rare-thing", "another"]);
        let mentions = m(&[
            ("scope:claude-code", 5000),
            ("rare-thing", 1),
            ("another", 2),
        ]);
        let (out, rep) = cap_by_selectivity(input, 1, 2, &mentions);
        assert_eq!(
            out[0], "scope:claude-code",
            "authored is kept in place regardless of rank"
        );
        assert_eq!(
            out[1], "rare-thing",
            "the rarest derived entry takes the remaining slot"
        );
        assert_eq!(rep.authored, 1);
    }

    #[test]
    fn the_rarest_derived_entries_win_the_remaining_slots() {
        let input = v(&["generic", "alsogeneric", "rare", "rarest"]);
        let mentions = m(&[
            ("generic", 2000),
            ("alsogeneric", 1500),
            ("rare", 12),
            ("rarest", 1),
        ]);
        let (out, rep) = cap_by_selectivity(input, 0, 2, &mentions);
        assert_eq!(out, v(&["rarest", "rare"]));
        assert_eq!(rep.dropped, 2);
        // Both survivors sat PAST the cap in arrival order — this is the case the measurement said
        // was happening on two-thirds of records.
        assert_eq!(rep.rescued_by_ordering, 2);
    }

    #[test]
    fn a_never_seen_entity_is_treated_as_maximally_selective() {
        let input = v(&["known", "brand-new"]);
        let (out, _) = cap_by_selectivity(input, 0, 1, &m(&[("known", 40)]));
        assert_eq!(
            out,
            v(&["brand-new"]),
            "absent from the graph means novel, not unknown-so-last"
        );
    }

    #[test]
    fn ties_fall_back_to_arrival_so_the_result_is_deterministic() {
        let input = v(&["first", "second", "third"]);
        let mentions = m(&[("first", 5), ("second", 5), ("third", 5)]);
        let (a, _) = cap_by_selectivity(input.clone(), 0, 2, &mentions);
        let (b, _) = cap_by_selectivity(input, 0, 2, &mentions);
        assert_eq!(a, v(&["first", "second"]));
        assert_eq!(
            a, b,
            "same inputs, byte-identical output — this repo's determinism rule"
        );
    }

    #[test]
    fn rescued_by_ordering_is_zero_when_arrival_already_had_it_right() {
        // The honest negative. If the metric could not report zero it would be decoration, and a
        // persistent zero in production is the signal that this ordering is inert.
        let input = v(&["rarest", "rare", "generic"]);
        let mentions = m(&[("rarest", 1), ("rare", 12), ("generic", 2000)]);
        let (out, rep) = cap_by_selectivity(input, 0, 2, &mentions);
        assert_eq!(out, v(&["rarest", "rare"]));
        assert_eq!(
            rep.rescued_by_ordering, 0,
            "arrival order already kept the right two"
        );
    }

    #[test]
    fn authored_never_exceeds_the_cap_itself() {
        let input = v(&["a", "b", "c", "d"]);
        let (out, rep) = cap_by_selectivity(input, 10, 2, &m(&[]));
        assert_eq!(
            out.len(),
            2,
            "an authored count larger than the cap cannot overflow it"
        );
        assert_eq!(rep.authored, 2);
    }

    #[test]
    fn a_derived_tag_needs_a_letter_and_no_code_punctuation() {
        for ok in [
            "5d61fd90",
            "fusion_features",
            "docs/src/reference/metrics",
            "veld-7d",
            "v0.8.4",
            "R@10",
            "408s",
        ] {
            assert!(derived_tag_is_well_formed(ok), "{ok} should pass");
        }
        for bad in [
            "2,041",
            "6.8(f",
            "veld_recall_stage_duration_seconds{stage=\"post",
            "15-159",
            "2026-10-19",
            "40053469645",
            "a",
            "[x]",
        ] {
            assert!(!derived_tag_is_well_formed(bad), "{bad} should fail");
        }
    }

    #[test]
    fn malformed_derived_entries_go_and_authored_ones_stay_whatever_they_look_like() {
        let input = v(&[
            "2026-10-19",
            "kind:task",
            "fusion_features",
            "2,041",
            "6.8(f",
            "veld-7d",
        ]);
        let (out, malformed) = well_formed_derived(input, 2);
        assert_eq!(
            out,
            v(&["2026-10-19", "kind:task", "fusion_features", "veld-7d"])
        );
        assert_eq!(malformed, 2);
    }

    #[test]
    fn the_derived_cap_is_authored_plus_budget_under_the_record_maximum() {
        assert_eq!(derived_cap(9, 12, 50), 21);
        assert_eq!(derived_cap(45, 12, 50), 50);
        assert_eq!(derived_cap(0, 12, 50), 12);
    }

    #[test]
    fn budget_and_selectivity_keep_the_rarest_dozen_after_the_authored_head() {
        let mut input = v(&["project:veld", "kind:task"]);
        let derived: Vec<String> = (0..30).map(|i| format!("derived{i}")).collect();
        input.extend(derived.clone());
        let mentions: HashMap<String, usize> = derived
            .iter()
            .enumerate()
            .map(|(i, d)| (d.clone(), 100 - i))
            .collect();
        let cap = derived_cap(2, 12, 50);
        let (out, rep) = cap_by_selectivity(input, 2, cap, &mentions);
        assert_eq!(out.len(), 14);
        assert_eq!(&out[..2], &v(&["project:veld", "kind:task"])[..]);
        assert_eq!(
            out[2], "derived29",
            "the least-mentioned derived entry leads the tail"
        );
        assert_eq!(rep.dropped, 18);
    }
}
