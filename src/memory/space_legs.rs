//! Recall's vector legs beyond the primary index, searched beside the rest of recall.
//!
//! A query is embedded once in each space an embedder here writes. Every index those vectors
//! reach is a leg: another space's index searched directly, or an index reached through an
//! installed map. A leg needs nothing recall computes after the query vectors, so the legs
//! start on their own thread as soon as the vectors exist and are collected at fusion, while
//! graph traversal, the primary search and BM25 run. A space no query vector reaches, directly
//! or through a map, is not searched: the read path never embeds into it.
//!
//! Each leg asks its index for a candidate budget: the dials' `space_candidates` for the space
//! when set, else what the space's own record supports. The record is how deep into the leg's
//! list the results recall returned lay, in multiples of the results asked for.

use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::{Mutex, RwLock};

use super::space_index::{IdMapping, SpaceIndex};
use super::types::MemoryId;
use crate::constants::VECTOR_SEARCH_CANDIDATE_MULTIPLIER;
use crate::embeddings::alignment::Alignment;
use crate::embeddings::space::SpaceId;
use crate::vector_db::vamana::VamanaIndex;

/// Candidates a leg asks for, in multiples of the results recall returns, until its record
/// says otherwise: what every leg asked for before budgets existed.
pub const DEFAULT_CANDIDATE_FACTOR: f32 = 8.0;
/// The least a learned budget falls to, so a leg that has added nothing can still show that it
/// now would.
pub const MIN_CANDIDATE_FACTOR: f32 = 2.0;
/// The most the dials may ask of one leg.
pub const MAX_CANDIDATE_FACTOR: f32 = 32.0;

/// How deep into one space's legs the returned results lay, as a moving mean and variance of
/// the deepest returned rank per recall, in multiples of the results asked for.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LegDepth {
    pub mean: f32,
    pub variance: f32,
    pub samples: u64,
}

impl LegDepth {
    const ALPHA: f32 = 0.05;
    /// Recalls seen before the record sets a budget.
    pub const MIN_SAMPLES: u64 = 30;

    pub fn observe(&mut self, depth: f32) {
        if self.samples == 0 {
            self.mean = depth;
            self.variance = 0.0;
        } else {
            let delta = depth - self.mean;
            self.mean += Self::ALPHA * delta;
            self.variance = (1.0 - Self::ALPHA) * (self.variance + Self::ALPHA * delta * delta);
        }
        self.samples = self.samples.saturating_add(1);
    }

    /// The candidate factor the record supports: three deviations past the mean depth, never
    /// above the default, so a learned budget only ever saves work.
    pub fn factor(&self) -> f32 {
        if self.samples < Self::MIN_SAMPLES {
            return DEFAULT_CANDIDATE_FACTOR;
        }
        (self.mean + 3.0 * self.variance.max(0.0).sqrt())
            .clamp(MIN_CANDIDATE_FACTOR, DEFAULT_CANDIDATE_FACTOR)
    }
}

/// How deep into `ranked` the deepest of `returned` lay, in multiples of `asked`; 0 when none
/// of them came from it.
pub fn returned_depth(ranked: &[MemoryId], returned: &[MemoryId], asked: usize) -> f32 {
    let deepest = ranked
        .iter()
        .rposition(|id| returned.contains(id))
        .map_or(0, |i| i + 1);
    deepest as f32 / asked.max(1) as f32
}

/// Results to ask of a leg with `factor`, for a recall returning `results`.
pub fn budget(factor: f32, results: usize) -> usize {
    ((factor * results as f32).ceil() as usize).max(1)
}

#[derive(Debug, Clone, PartialEq)]
pub enum LegKind {
    /// A space's own index, searched with the query vector in that space.
    Direct,
    /// The primary index, searched with the secondary query vector mapped into the primary's
    /// space through the installed secondary-to-primary map.
    Aligned,
    /// Another space's index, reached from `from` through a fitted map whose held-out paired
    /// cosine is `confidence`.
    Bridged { from: SpaceId, confidence: f32 },
}

pub(crate) enum LegIndex {
    Space(Arc<SpaceIndex>),
    Primary {
        index: Arc<RwLock<VamanaIndex>>,
        mapping: Arc<RwLock<IdMapping>>,
    },
}

pub struct Leg {
    /// The space of the index searched.
    pub space: SpaceId,
    pub kind: LegKind,
    pub(crate) index: LegIndex,
    pub(crate) map: Option<Arc<dyn Alignment>>,
    pub(crate) query: Vec<f32>,
    /// Results asked of the index.
    pub limit: usize,
    /// Run only when the aligned leg before it failed: the direct search it stands in for.
    pub(crate) fallback: bool,
}

impl Leg {
    fn search(&self) -> Result<Vec<(MemoryId, f32)>> {
        let mapped;
        let query: &[f32] = match &self.map {
            Some(map) => {
                mapped = map.project(&self.query)?;
                &mapped
            }
            None => &self.query,
        };
        let candidates = self.limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2;
        match &self.index {
            LegIndex::Space(index) => index.search(query, candidates, self.limit),
            LegIndex::Primary { index, mapping } => {
                search_primary(index, mapping, query, candidates, self.limit)
            }
        }
    }
}

/// One memory per hit at its best vector's similarity, best first, ties broken by id.
fn search_primary(
    index: &RwLock<VamanaIndex>,
    mapping: &RwLock<IdMapping>,
    query: &[f32],
    candidates: usize,
    limit: usize,
) -> Result<Vec<(MemoryId, f32)>> {
    // LOCK ORDERING: primary vector_index (1) before id_mapping (2)
    let index = index.read();
    let results = index
        .search(query, candidates)
        .context("searching the primary index with a mapped query")?;
    let mapping = mapping.read();
    let mut best: std::collections::HashMap<MemoryId, f32> = std::collections::HashMap::new();
    for (vector_id, distance) in results {
        // NormalizedDotProduct: the index returns -dot, so similarity is -distance.
        let similarity = -distance;
        if let Some(memory_id) = mapping.get_memory_id(vector_id) {
            best.entry(memory_id.clone())
                .and_modify(|s| *s = s.max(similarity))
                .or_insert(similarity);
        }
    }
    let mut out: Vec<(MemoryId, f32)> = best.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(limit);
    Ok(out)
}

/// A leg's results, in the order its index ranked them.
pub struct LegHits {
    pub space: SpaceId,
    pub kind: LegKind,
    pub hits: Vec<(MemoryId, f32)>,
}

/// Run `legs` in order. A failed leg is logged and left out; a fallback leg runs only after a
/// failed aligned leg.
fn run(legs: Vec<Leg>) -> Vec<LegHits> {
    let mut out = Vec::with_capacity(legs.len());
    let mut aligned_failed = false;
    for leg in legs {
        if leg.fallback && !aligned_failed {
            continue;
        }
        match leg.search() {
            Ok(hits) => out.push(LegHits {
                space: leg.space,
                kind: leg.kind,
                hits,
            }),
            Err(e) => match &leg.kind {
                LegKind::Aligned => {
                    aligned_failed = true;
                    tracing::warn!("Aligned secondary search failed (non-fatal): {e:#}");
                }
                LegKind::Direct => {
                    tracing::warn!("{} index search failed (non-fatal): {e:#}", leg.space);
                }
                LegKind::Bridged { from, .. } => {
                    tracing::warn!("{from} → {} search failed (non-fatal): {e:#}", leg.space);
                }
            },
        }
    }
    out
}

/// Legs started, to be collected at fusion.
pub struct PendingLegs(Pending);

enum Pending {
    Running(std::thread::JoinHandle<Vec<LegHits>>),
    Done(Vec<LegHits>),
}

impl PendingLegs {
    /// Start `legs` on their own thread. With no legs nothing starts; when no thread can be
    /// had they run here.
    pub fn start(legs: Vec<Leg>) -> Self {
        if legs.is_empty() {
            return Self(Pending::Done(Vec::new()));
        }
        let slot = Arc::new(Mutex::new(Some(legs)));
        let taken = Arc::clone(&slot);
        let spawned = std::thread::Builder::new()
            .name("veld-recall-legs".to_string())
            .spawn(move || taken.lock().take().map(run).unwrap_or_default());
        match spawned {
            Ok(handle) => Self(Pending::Running(handle)),
            Err(e) => {
                tracing::warn!("no thread for recall's space legs, running them inline: {e}");
                let legs = slot.lock().take().unwrap_or_default();
                Self(Pending::Done(run(legs)))
            }
        }
    }

    pub fn finish(self) -> Vec<LegHits> {
        match self.0 {
            Pending::Done(hits) => hits,
            Pending::Running(handle) => handle.join().unwrap_or_else(|_| {
                tracing::error!("recall's space legs panicked; their results are left out");
                Vec::new()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn depth_is_the_deepest_returned_rank_over_the_results_asked() {
        let ranked: Vec<MemoryId> = (1..=40).map(id).collect();
        assert_eq!(returned_depth(&ranked, &[id(3), id(20), id(99)], 10), 2.0);
        assert_eq!(returned_depth(&ranked, &[id(99)], 10), 0.0);
        assert_eq!(returned_depth(&[], &[id(1)], 10), 0.0);
    }

    #[test]
    fn a_record_keeps_the_default_until_it_has_seen_enough_and_never_exceeds_it() {
        let mut shallow = LegDepth::default();
        for _ in 0..LegDepth::MIN_SAMPLES - 1 {
            shallow.observe(0.5);
        }
        assert_eq!(shallow.factor(), DEFAULT_CANDIDATE_FACTOR);
        shallow.observe(0.5);
        assert_eq!(shallow.factor(), MIN_CANDIDATE_FACTOR);

        let mut steady = LegDepth::default();
        for _ in 0..LegDepth::MIN_SAMPLES {
            steady.observe(5.0);
        }
        assert_eq!(steady.factor(), 5.0);

        let mut deep = LegDepth::default();
        for _ in 0..LegDepth::MIN_SAMPLES {
            deep.observe(12.0);
        }
        assert_eq!(deep.factor(), DEFAULT_CANDIDATE_FACTOR);

        let mut middling = LegDepth::default();
        for n in 0..200 {
            middling.observe(if n % 2 == 0 { 3.0 } else { 4.0 });
        }
        let f = middling.factor();
        assert!(f > 3.5 && f < DEFAULT_CANDIDATE_FACTOR, "factor {f}");
    }

    #[test]
    fn a_budget_is_at_least_one_result() {
        assert_eq!(budget(8.0, 10), 80);
        assert_eq!(budget(2.5, 3), 8);
        assert_eq!(budget(2.0, 0), 1);
    }

    #[test]
    fn no_legs_start_no_thread_and_finish_empty() {
        let pending = PendingLegs::start(Vec::new());
        assert!(matches!(pending.0, Pending::Done(ref h) if h.is_empty()));
        assert!(pending.finish().is_empty());
    }
}
