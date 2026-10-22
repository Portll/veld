//! The writes a recall makes because it happened — the retrieval counter, co-retrieval
//! reinforcement of the returned memories' graph edges, and the interference a retrieval
//! competition leaves — applied off the read path.
//!
//! Each `MemorySystem` owns one bounded queue drained by one worker thread, so writes apply
//! in the order recalls submitted them, and every write to one memory in the order it was
//! made. They persist exactly as they did inline: the counter to `stats:total_retrievals`,
//! edges through the graph store, interference records and the event count through the
//! primary store. A full queue drops the write; each drop is counted in
//! `veld_usage_writes_dropped_total{kind}` and logged.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use super::introspection::{ConsolidationEvent, ConsolidationEventBuffer};
use super::replay::{CompetitionResult, InterferenceDetector};
use super::types::MemoryStats;

/// Writes held for the worker before a recall's write is dropped instead of queued.
pub(super) const USAGE_WRITE_QUEUE_CAPACITY: usize = 1024;

/// Pairs among the leading memories of a reinforced recall that are reported as
/// strengthened edges, as the inline path reported them.
const COACTIVATION_EVENT_SPAN: usize = 5;

/// One deferred write.
pub(super) enum UsageWrite {
    /// Bump the persisted retrieval counter and mirror it into the in-memory stats.
    RetrievalCounted,
    /// Strengthen the edges between memories returned together (S-R4: one retrieval id,
    /// so a repeat is a no-op).
    Coactivation {
        retrieval_id: uuid::Uuid,
        source: String,
        memories: Vec<uuid::Uuid>,
    },
    /// Record the interference of a retrieval competition already ranked on the read path,
    /// and persist the affected memories' records and the event total.
    Competition {
        candidates: Vec<(String, f32, f32)>,
        record_state: bool,
        result: CompetitionResult,
    },
    /// Answer once every write queued before it has applied.
    Barrier(SyncSender<()>),
}

impl UsageWrite {
    fn kind(&self) -> &'static str {
        match self {
            UsageWrite::RetrievalCounted => "retrieval_count",
            UsageWrite::Coactivation { .. } => "coactivation",
            UsageWrite::Competition { .. } => "competition",
            UsageWrite::Barrier(_) => "barrier",
        }
    }
}

/// What the worker writes to: shared handles to the `MemorySystem`'s own state.
#[derive(Clone)]
pub(super) struct UsageSinks {
    pub(super) store: Arc<dyn crate::storage::PrimaryMemoryStore>,
    pub(super) stats: Arc<RwLock<MemoryStats>>,
    pub(super) graph: Option<Arc<RwLock<crate::graph_memory::GraphMemory>>>,
    pub(super) interference: Arc<RwLock<InterferenceDetector>>,
    pub(super) events: Arc<RwLock<ConsolidationEventBuffer>>,
}

impl UsageSinks {
    /// A writer applying to these sinks.
    pub(super) fn into_writer(self, capacity: usize) -> UsageWriter {
        UsageWriter::new(Arc::new(move |write| self.apply(write)), capacity)
    }

    fn apply(&self, write: UsageWrite) {
        match write {
            UsageWrite::RetrievalCounted => {
                if let Ok(count) = self.store.increment_retrieval_count() {
                    self.stats.write().total_retrievals = count;
                }
            }
            UsageWrite::Coactivation {
                retrieval_id,
                source,
                memories,
            } => self.reinforce(retrieval_id, &source, &memories),
            UsageWrite::Competition {
                candidates,
                record_state,
                result,
            } => self.record_competition(&candidates, record_state, &result),
            UsageWrite::Barrier(done) => {
                let _ = done.send(());
            }
        }
    }

    fn reinforce(&self, retrieval_id: uuid::Uuid, source: &str, memories: &[uuid::Uuid]) {
        let (Some(graph), Some((first, rest))) = (&self.graph, memories.split_first()) else {
            return;
        };
        match graph
            .read()
            .apply_reinforcement(retrieval_id, source, first, rest)
        {
            Ok(result) if result.edges_updated > 0 => {
                let leading = &memories[..memories.len().min(COACTIVATION_EVENT_SPAN)];
                let mut events = self.events.write();
                for (i, from) in leading.iter().enumerate() {
                    for to in &leading[i + 1..] {
                        events.push(ConsolidationEvent::EdgeStrengthened {
                            from_memory_id: from.to_string(),
                            to_memory_id: to.to_string(),
                            strength_before: 0.0,
                            strength_after: 0.025,
                            co_activations: 1,
                            timestamp: chrono::Utc::now(),
                        });
                    }
                }
            }
            Ok(_) => {}
            Err(e) => tracing::trace!("Coactivation recording failed (non-critical): {e}"),
        }
    }

    fn record_competition(
        &self,
        candidates: &[(String, f32, f32)],
        record_state: bool,
        result: &CompetitionResult,
    ) {
        let (records, total_events) = {
            let mut detector = self.interference.write();
            detector.record_retrieval_competition(candidates, record_state);
            let affected = detector.get_affected_ids_from_competition(result);
            if affected.is_empty() {
                return;
            }
            let records: Vec<(String, Vec<super::replay::InterferenceRecord>)> = detector
                .get_records_for_ids(&affected)
                .into_iter()
                .map(|(id, records)| (id.to_string(), records.clone()))
                .collect();
            (records, detector.stats().0)
        };
        for (id, records) in &records {
            if let Err(e) = self.store.save_interference_records(id, records) {
                tracing::debug!("Failed to persist competition interference: {e}");
            }
        }
        if let Err(e) = self.store.save_interference_event_count(total_events) {
            tracing::debug!("Failed to persist interference event count: {e}");
        }
    }
}

struct Lane {
    sender: SyncSender<UsageWrite>,
    worker: std::thread::JoinHandle<()>,
}

enum LaneState {
    NotStarted,
    Running(Lane),
    /// Shut down, or the worker could not be started: writes apply on the caller's thread.
    Inline,
}

/// Applies one write; the worker calls it in queue order.
type ApplyUsageWrite = Arc<dyn Fn(UsageWrite) + Send + Sync>;

/// The per-`MemorySystem` queue and its worker, started on the first write.
pub(super) struct UsageWriter {
    apply: ApplyUsageWrite,
    capacity: usize,
    lane: Mutex<LaneState>,
    dropped: AtomicU64,
}

impl UsageWriter {
    pub(super) fn new(apply: ApplyUsageWrite, capacity: usize) -> Self {
        Self {
            apply,
            capacity: capacity.max(1),
            lane: Mutex::new(LaneState::NotStarted),
            dropped: AtomicU64::new(0),
        }
    }

    /// Queue `write` for the worker. Never blocks: a full queue drops the write, counted
    /// and logged.
    pub(super) fn submit(&self, write: UsageWrite) {
        let Some(sender) = self.sender() else {
            (self.apply)(write);
            return;
        };
        match sender.try_send(write) {
            Ok(()) => {}
            Err(TrySendError::Full(write)) => self.note_dropped(write.kind()),
            Err(TrySendError::Disconnected(write)) => (self.apply)(write),
        }
    }

    /// Wait until every write submitted before this call has applied.
    pub(super) fn flush(&self) {
        let sender = match &*self.lane.lock() {
            LaneState::Running(running) => running.sender.clone(),
            LaneState::NotStarted | LaneState::Inline => return,
        };
        let (done, wait) = sync_channel(1);
        if sender.send(UsageWrite::Barrier(done)).is_ok() {
            let _ = wait.recv();
        }
    }

    /// Apply everything queued, stop the worker, and apply later writes inline.
    pub(super) fn shutdown(&self) {
        let state = std::mem::replace(&mut *self.lane.lock(), LaneState::Inline);
        if let LaneState::Running(lane) = state {
            drop(lane.sender);
            if lane.worker.join().is_err() {
                tracing::warn!("usage-write worker panicked; its queued writes are lost");
            }
        }
    }

    /// Writes dropped on a full queue since this writer was made.
    #[cfg(test)]
    pub(super) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn sender(&self) -> Option<SyncSender<UsageWrite>> {
        let mut lane = self.lane.lock();
        if matches!(*lane, LaneState::NotStarted) {
            *lane = self.start();
        }
        match &*lane {
            LaneState::Running(running) => Some(running.sender.clone()),
            LaneState::NotStarted | LaneState::Inline => None,
        }
    }

    fn start(&self) -> LaneState {
        let (sender, receiver) = sync_channel(self.capacity);
        let apply = Arc::clone(&self.apply);
        match std::thread::Builder::new()
            .name("veld-usage-writes".to_string())
            .spawn(move || drain(apply.as_ref(), receiver))
        {
            Ok(worker) => LaneState::Running(Lane { sender, worker }),
            Err(error) => {
                tracing::warn!(%error, "could not start the usage-write worker; writing inline");
                LaneState::Inline
            }
        }
    }

    fn note_dropped(&self, kind: &'static str) {
        crate::metrics::USAGE_WRITES_DROPPED_TOTAL
            .with_label_values(&[kind])
            .inc();
        let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped % 1000 == 0 {
            tracing::warn!(
                kind,
                dropped,
                capacity = self.capacity,
                "usage-write queue full; recall usage write dropped"
            );
        }
    }
}

fn drain(apply: &(dyn Fn(UsageWrite) + Send + Sync), receiver: Receiver<UsageWrite>) {
    while let Ok(write) = receiver.recv() {
        match write {
            UsageWrite::Barrier(done) => {
                let _ = done.send(());
            }
            write => apply(write),
        }
    }
}

impl Drop for UsageWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A writer whose worker records what it applies and can be held mid-write.
    fn recording_writer(capacity: usize) -> (UsageWriter, Arc<Mutex<Vec<String>>>, Arc<Mutex<()>>) {
        let applied: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let gate: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
        let (log, hold) = (Arc::clone(&applied), Arc::clone(&gate));
        let writer = UsageWriter::new(
            Arc::new(move |write| {
                let _held = hold.lock();
                if let UsageWrite::Coactivation { source, .. } = write {
                    log.lock().push(source);
                }
            }),
            capacity,
        );
        (writer, applied, gate)
    }

    fn reinforcement(source: &str) -> UsageWrite {
        UsageWrite::Coactivation {
            retrieval_id: uuid::Uuid::new_v4(),
            source: source.to_string(),
            memories: vec![uuid::Uuid::new_v4(), uuid::Uuid::new_v4()],
        }
    }

    /// Writes apply in the order they were submitted, and a flush returns once all of
    /// them have.
    #[test]
    fn writes_apply_in_submission_order() {
        let (writer, applied, _gate) = recording_writer(64);
        let sources: Vec<String> = (0..40).map(|i| format!("recall-{i}")).collect();
        for source in &sources {
            writer.submit(reinforcement(source));
        }
        writer.flush();
        assert_eq!(*applied.lock(), sources);
        assert_eq!(writer.dropped(), 0);
    }

    /// With the worker held and the queue full, a further write is dropped and counted
    /// instead of blocking the recall; the queued ones still apply, in order.
    #[test]
    fn a_full_queue_drops_and_counts_instead_of_blocking() {
        let (writer, applied, gate) = recording_writer(2);
        let held = gate.lock();
        writer.submit(reinforcement("first"));
        // The worker takes "first" and blocks on the gate; two more fill the queue.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            writer.submit(reinforcement("queued"));
            if writer.dropped() > 0 || std::time::Instant::now() > deadline {
                break;
            }
        }
        assert!(writer.dropped() > 0, "a full queue must drop, not block");
        drop(held);
        writer.flush();
        let applied = applied.lock();
        assert_eq!(applied.first().map(String::as_str), Some("first"));
        assert!(applied[1..].iter().all(|s| s == "queued"));
        // The worker's write and a full queue of two, or a queue of two the worker had not
        // yet taken its first from when the drop happened.
        assert!((2..=3).contains(&applied.len()), "{applied:?}");
    }

    /// Shutting down applies what was queued, and later writes apply on the caller.
    #[test]
    fn shutdown_drains_then_writes_inline() {
        let (writer, applied, _gate) = recording_writer(64);
        writer.submit(reinforcement("before"));
        writer.shutdown();
        assert_eq!(*applied.lock(), vec!["before".to_string()]);
        writer.submit(reinforcement("after"));
        assert_eq!(
            *applied.lock(),
            vec!["before".to_string(), "after".to_string()]
        );
    }
}
