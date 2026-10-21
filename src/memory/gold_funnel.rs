//! Per-stage gold-rank funnel diagnostic.
//!
//! Tracks, for a known set of gold memory ids, the best (lowest) rank of any gold id in the
//! candidate list at each pipeline stage boundary. This LOCATES where reachable gold is
//! dropped or demoted — turning "fusion buries it" from a hypothesis into a measured,
//! per-stage drop-off funnel.
//!
//! Implemented as a thread-local so the recall pipeline needs no signature change and
//! production callers pay nothing (the thread-local is unset → every record call is a cheap
//! no-op). The recall harness arms it per query via [`begin`], then drains it via [`take`];
//! or via [`begin_capture`] and [`take_capture`] to keep every stage's whole candidate list.

use super::types::MemoryId;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    static FUNNEL: RefCell<Option<FunnelState>> = const { RefCell::new(None) };
}

/// Every recorded stage's ordered candidate list, in pipeline order, with the stage's score
/// for each candidate where the stage has one.
pub type StageCapture = Vec<(String, Vec<(MemoryId, Option<f32>)>)>;

struct FunnelState {
    gold: HashSet<MemoryId>,
    /// (stage name, best gold rank at that stage; None if no gold present).
    stages: Vec<(String, Option<usize>)>,
    /// Set by [`begin_capture`]. Two processes' captures can then be compared stage by stage,
    /// and a difference told apart as a reordering or a change in the scores themselves.
    capture: Option<StageCapture>,
}

/// Arm the funnel for the next recall with this query's gold ids. Clears any prior state.
pub fn begin(gold: HashSet<MemoryId>) {
    FUNNEL.with(|c| {
        *c.borrow_mut() = Some(FunnelState {
            gold,
            stages: Vec::new(),
            capture: None,
        });
    });
}

/// Arm the funnel to keep every stage's full ordered candidate list for the next recall on
/// this thread. No gold is tracked. Clears any prior state.
pub fn begin_capture() {
    FUNNEL.with(|c| {
        *c.borrow_mut() = Some(FunnelState {
            gold: HashSet::new(),
            stages: Vec::new(),
            capture: Some(Vec::new()),
        });
    });
}

/// Disarm and return the captured per-stage candidate lists, in the order the pipeline
/// recorded them (None if no capture was armed).
pub fn take_capture() -> Option<StageCapture> {
    FUNNEL.with(|c| c.borrow_mut().take().and_then(|s| s.capture))
}

/// Disarm and return the recorded per-stage ranks (None if the funnel was never armed).
pub fn take() -> Option<Vec<(String, Option<usize>)>> {
    FUNNEL.with(|c| c.borrow_mut().take().map(|s| s.stages))
}

/// Record the best gold rank in `ids` (an ordered candidate list) for `stage`. No-op unless
/// the funnel is armed. `ids` must be in rank order (rank 0 = best).
pub fn record<'a>(stage: &str, ids: impl Iterator<Item = &'a MemoryId>) {
    record_inner(stage, ids.map(|id| (id, None)));
}

/// [`record`] for a stage that scores its candidates: a capture keeps the score beside each
/// id, so two runs can be compared on values and not only on order. `ids` must be in the
/// stage's own order.
pub fn record_scored<'a>(stage: &str, ids: impl Iterator<Item = (&'a MemoryId, f32)>) {
    record_inner(stage, ids.map(|(id, score)| (id, Some(score))));
}

fn record_inner<'a>(stage: &str, ids: impl Iterator<Item = (&'a MemoryId, Option<f32>)>) {
    FUNNEL.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(state) = borrow.as_mut() else {
            return;
        };
        let FunnelState {
            gold,
            stages,
            capture,
        } = state;
        let best = if let Some(capture) = capture.as_mut() {
            let list: Vec<(MemoryId, Option<f32>)> =
                ids.map(|(id, score)| (id.clone(), score)).collect();
            let best = list.iter().position(|(id, _)| gold.contains(id));
            capture.push((stage.to_string(), list));
            best
        } else {
            let mut best = None;
            for (i, (id, _)) in ids.enumerate() {
                if gold.contains(id) {
                    best = Some(i);
                    break;
                }
            }
            best
        };
        stages.push((stage.to_string(), best));
    });
}

/// True when the funnel is armed — lets the pipeline skip building an iterator when not needed.
pub fn is_armed() -> bool {
    FUNNEL.with(|c| c.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn capture_keeps_every_stage_in_order_with_its_scores() {
        let (a, b) = (id(1), id(2));
        begin_capture();
        record("graph", [&a, &b].into_iter());
        record_scored("ce", [(&b, 0.5f32), (&a, -1.25f32)].into_iter());
        let capture = take_capture().expect("capture armed");
        assert_eq!(
            capture,
            vec![
                (
                    "graph".to_string(),
                    vec![(a.clone(), None), (b.clone(), None)]
                ),
                (
                    "ce".to_string(),
                    vec![(b.clone(), Some(0.5)), (a.clone(), Some(-1.25))]
                ),
            ]
        );
        assert!(!is_armed(), "take_capture disarms");
    }

    #[test]
    fn gold_funnel_keeps_no_capture_and_scored_stages_still_rank() {
        let (a, b) = (id(1), id(2));
        begin(HashSet::from([b.clone()]));
        record_scored("vector", [(&a, 0.9f32), (&b, 0.8f32)].into_iter());
        assert_eq!(take(), Some(vec![("vector".to_string(), Some(1))]));
        begin(HashSet::new());
        assert_eq!(take_capture(), None);
    }
}
