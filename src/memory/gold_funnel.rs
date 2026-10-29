//! Per-stage gold-rank funnel diagnostic.
//!
//! Tracks, for a known set of gold memory ids, the best (lowest) rank of any gold id in the
//! candidate list at each pipeline stage boundary, and which memory leads the list there. This
//! LOCATES where reachable gold is dropped or demoted — turning "fusion buries it" from a
//! hypothesis into a measured, per-stage drop-off funnel — and names the stage at which a
//! non-gold memory took rank 1 from gold.
//!
//! Implemented as a thread-local so the recall pipeline needs no signature change and
//! production callers pay nothing (the thread-local is unset → every record call is a cheap
//! no-op). The recall harness arms it per query via [`begin`], then drains it via [`take`].

use super::types::MemoryId;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    static FUNNEL: RefCell<Option<FunnelState>> = const { RefCell::new(None) };
}

struct FunnelState {
    gold: HashSet<MemoryId>,
    stages: Vec<FunnelStage>,
}

/// One stage boundary: the best gold rank there (None if no gold is in the list) and the
/// memory at rank 0 (None for an empty list).
#[derive(Debug, Clone, PartialEq)]
pub struct FunnelStage {
    pub stage: String,
    pub gold_rank: Option<usize>,
    pub leader: Option<MemoryId>,
}

/// Arm the funnel for the next recall with this query's gold ids. Clears any prior state.
pub fn begin(gold: HashSet<MemoryId>) {
    FUNNEL.with(|c| {
        *c.borrow_mut() = Some(FunnelState {
            gold,
            stages: Vec::new(),
        });
    });
}

/// Disarm and return the recorded stages (None if the funnel was never armed).
pub fn take() -> Option<Vec<FunnelStage>> {
    FUNNEL.with(|c| c.borrow_mut().take().map(|s| s.stages))
}

/// Record the best gold rank and the leader in `ids` (an ordered candidate list) for `stage`.
/// No-op unless the funnel is armed. `ids` must be in rank order (rank 0 = best).
pub fn record<'a>(stage: &str, ids: impl Iterator<Item = &'a MemoryId>) {
    FUNNEL.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(state) = borrow.as_mut() else {
            return;
        };
        let mut leader: Option<MemoryId> = None;
        let mut gold_rank: Option<usize> = None;
        for (i, id) in ids.enumerate() {
            if i == 0 {
                leader = Some(id.clone());
            }
            if state.gold.contains(id) {
                gold_rank = Some(i);
                break;
            }
        }
        state.stages.push(FunnelStage {
            stage: stage.to_string(),
            gold_rank,
            leader,
        });
    });
}

/// [`record`] for a list the pipeline has scored but not yet sorted: ranks by score, highest
/// first, ties in list order. The sort runs only when the funnel is armed.
pub fn record_by_score<'a>(stage: &str, scored: impl Iterator<Item = (&'a MemoryId, f32)>) {
    if !is_armed() {
        return;
    }
    let mut ranked: Vec<(&MemoryId, f32)> = scored.collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    record(stage, ranked.into_iter().map(|(id, _)| id));
}

/// True when the funnel is armed — lets the pipeline skip building an iterator when not needed.
pub fn is_armed() -> bool {
    FUNNEL.with(|c| c.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn id(n: u128) -> MemoryId {
        MemoryId(Uuid::from_u128(n))
    }

    #[test]
    fn records_leader_and_gold_rank_per_stage() {
        begin(HashSet::from([id(2)]));
        record("a", [id(1), id(2), id(3)].iter());
        record("b", [id(2), id(1)].iter());
        record("c", [id(1), id(3)].iter());
        record("d", std::iter::empty());
        let stages = take().expect("armed");
        let got: Vec<(&str, Option<usize>, Option<MemoryId>)> = stages
            .iter()
            .map(|s| (s.stage.as_str(), s.gold_rank, s.leader.clone()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("a", Some(1), Some(id(1))),
                ("b", Some(0), Some(id(2))),
                ("c", None, Some(id(1))),
                ("d", None, None),
            ]
        );
        assert!(take().is_none(), "take disarms");
    }

    #[test]
    fn record_by_score_ranks_highest_first_and_keeps_ties_in_order() {
        let (a, b, c) = (id(1), id(2), id(3));
        begin(HashSet::from([c.clone()]));
        record_by_score("s", [(&a, 0.5), (&b, 0.9), (&c, 0.5)].into_iter());
        let stage = &take().expect("armed")[0];
        assert_eq!(stage.leader, Some(b));
        assert_eq!(stage.gold_rank, Some(2));
    }

    #[test]
    fn unarmed_records_nothing() {
        let _ = take();
        record("a", [id(1)].iter());
        record_by_score("b", [(&id(1), 1.0)].into_iter());
        assert!(take().is_none());
    }
}
