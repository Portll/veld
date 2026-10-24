//! One gap analysis: bring the mirror up to the graph, then detect over the
//! scope asked for.

use anyhow::Result;
use parking_lot::RwLock;
use serde::Serialize;
use std::time::Instant;

use super::{GapDetectionConfig, GapDetectionResult, GapDetector, GapScope, ScopeReport};
use crate::graph_memory::GraphMemory;
use crate::memory::graph_mirror::{sync_mirror, MirrorSyncOptions};
use crate::memory::slow_store::{SlowStore, SyncMode, SyncStats};

#[derive(Debug, Clone)]
pub struct GapRunParams {
    pub config: GapDetectionConfig,
    pub scope: GapScope,
    pub mirror: MirrorSyncOptions,
}

/// How a run's mirror was brought up to the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MirrorReport {
    pub mode: SyncMode,
    /// A full copy that could not account for every graph write made during it.
    pub best_effort: bool,
    /// Graph writes made during a full copy and applied after it.
    pub changed_during_copy: Option<usize>,
    /// Mirror rows written or removed.
    pub rows_written: usize,
    pub duration_ms: u64,
}

impl From<&SyncStats> for MirrorReport {
    fn from(stats: &SyncStats) -> Self {
        Self {
            mode: stats.mode,
            best_effort: stats.mode == SyncMode::Full && stats.changed_during_copy.is_none(),
            changed_during_copy: stats.changed_during_copy,
            rows_written: stats.entities_upserted
                + stats.edges_upserted
                + stats.entities_removed
                + stats.edges_removed,
            duration_ms: stats.duration_ms,
        }
    }
}

/// A finished analysis.
#[derive(Debug, Clone)]
pub struct GapRun {
    pub result: GapDetectionResult,
    pub scope: ScopeReport,
    pub mirror: MirrorReport,
    /// Detection alone.
    pub detect_ms: u64,
    /// Mirror sync and detection.
    pub duration_ms: u64,
}

/// Sync `store`'s mirror from `graph`, then detect gaps over `params.scope`.
pub fn run_gap_analysis(
    graph: &RwLock<GraphMemory>,
    store: &SlowStore,
    params: &GapRunParams,
) -> Result<GapRun> {
    let _span =
        tracing::info_span!("gap.run", whole_graph = params.scope.is_whole_graph()).entered();
    let start = Instant::now();
    let sync = sync_mirror(graph, store, &params.mirror)?;
    let detect_start = Instant::now();
    let (result, scope) = if params.scope.is_whole_graph() {
        let result = GapDetector::detect(store, &params.config)?;
        let nodes = store.entity_count()?;
        (result, params.scope.report(nodes, nodes))
    } else {
        let scoped = store.scoped_graph(&params.scope)?;
        let result = GapDetector::detect(&scoped, &params.config)?;
        (result, scoped.report().clone())
    };
    Ok(GapRun {
        result,
        scope,
        mirror: MirrorReport::from(&sync),
        detect_ms: detect_start.elapsed().as_millis() as u64,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}
