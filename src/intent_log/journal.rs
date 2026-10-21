//! Writer-side primitive that ties log-append + sync + projection-apply
//! into one operation.
//!
//! ## The "log first" invariant
//!
//! Every state-changing operation flows through one path:
//!
//! ```text
//!   1. encode payload → bytes
//!   2. log.append(bytes)              ← assigns LSN
//!   3. log.sync()                     ← durable
//!   4. for each projection: apply
//!   5. record any apply errors and return
//! ```
//!
//! After step 3 the operation is *committed* — survives a crash. If any
//! projection's `apply` (step 4) fails, the failure is collected but the
//! whole `record_and_apply` call still reports success, because the log
//! holds the record and the replay driver will re-apply on restart. This
//! is what the `Projection::apply` idempotency contract is for: replay
//! after a crashed apply must not double-count.
//!
//! ## Why this lives here, not inline at the call site
//!
//! Doing the log+sync+apply dance by hand at every memory CRUD call site
//! is exactly how `health_index` ended up as the unfixed twin of
//! `health_ready` — copy-paste drift across paths that should share a
//! structural rule. One writer primitive means one place to audit, one
//! place to fix, one place to add metrics.
//!
//! ## Deferred projections
//!
//! Projections registered with [`JournaledWriter::add_deferred_projection`]
//! are derived, non-authoritative mirrors (the SQLite slow store). They are
//! not applied inside `record_and_apply`: after step 3 the record is handed
//! to a per-projection worker thread through a bounded queue, and the call
//! returns without waiting for it. The worker applies records strictly in
//! LSN order. Because the record is already durable in the log, a deferred
//! apply that never happens (crash, abandoned drain) is recovered by the
//! boot-time replay from the projection's own checkpoint.
//!
//! - **Back-pressure, never drop.** A full queue makes the writer wait up
//!   to [`DeferredLaneConfig::enqueue_wait`]; if there is still no room it
//!   applies the queued backlog and then the new record inline, under the
//!   projection's lock, so LSN order is preserved and the mirror's lag stays
//!   bounded by the queue capacity.
//! - **Failures hold the checkpoint.** A failed deferred apply calls
//!   [`TypedProjection::hold_checkpoint_below`] with the failed LSN, so the
//!   persisted checkpoint never moves past it and replay retries it.
//! - **Drain.** [`JournaledWriter::shutdown_deferred`] (also run on drop)
//!   and [`DeferredDrainHandle::drain`] wait, bounded, for the queue to
//!   empty and then persist the checkpoint.
//!
//! ## Offset hints
//!
//! With a checkpoint store attached ([`JournaledWriter::attach_checkpoint_store`])
//! the writer persists the log's [`super::FrameIndex`] there after every
//! [`FRAME_INDEX_STRIDE`]-th record and when it is dropped, so the next open
//! of the log resumes its tail scan near the end instead of at the start.
//!
//! ## Group commit
//!
//! Concurrent writers to one tenant share a [`SharedJournal`]. Each write is
//! encoded and queued before the writer's mutex is taken; whichever thread
//! next holds the mutex with its own write still queued commits the whole
//! queue as one group: every record appended, one sync, then every record
//! dispatched in LSN order. The writes that queued while that sync ran form
//! the next group. A write another thread committed finds its outcome waiting
//! when it gets the mutex and returns without a sync of its own.
//!
//! Append, sync, rotation and dispatch all still run under the writer's
//! mutex, so whoever holds it sees a log whose frames are all synced and
//! applied, projections still apply in LSN order, and no checkpoint can name
//! an LSN the log has not made durable.

use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::metrics::{
    Timer, INTENT_LOG_SYNC_SHARED_TOTAL, PROJECTION_APPLY_DURATION, PROJECTION_APPLY_TOTAL,
};

use super::checkpoint_store::{CheckpointStore, CheckpointStoreError};
use super::payload::{self, IntentPayload, PayloadError};
use super::{IntentLog, IntentLogError, Lsn, FRAME_INDEX_STRIDE};

/// Records a deferred projection may have queued before a writer starts
/// waiting for room.
pub const DEFERRED_QUEUE_CAPACITY: usize = 1024;

/// How long a writer waits for room in a full deferred queue before it
/// applies inline instead. Kept well under a second because the writer holds
/// the tenant's journal mutex while it waits.
pub const DEFERRED_ENQUEUE_WAIT: Duration = Duration::from_millis(500);

/// Upper bound on draining a deferred queue at shutdown, on drop, or before
/// a restore swaps the projection's files.
pub const DEFERRED_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Object-safe variant of `Projection` that takes a *typed* payload + the
/// LSN it was assigned, instead of a raw `IntentRecord`. The static
/// `Projection` trait still exists for replay-time consumers (which work
/// from `IntentRecord` straight off the log); this dyn-friendly variant
/// is what `JournaledWriter` calls during normal writes.
///
/// Implementors typically own a single in-memory checkpoint and a single
/// `CheckpointStore` reference. They should be cheap to keep in a list
/// inside `JournaledWriter`.
pub trait TypedProjection: Send {
    /// Stable, unique identifier for this projection.
    fn name(&self) -> &str;

    /// Apply a typed payload at a given LSN. MUST be idempotent — see
    /// `super::projection` module docs.
    fn apply(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Downcast hook. Lets a holder of `&dyn TypedProjection` recover the
    /// concrete projection type — e.g. the W6 query planner reaching a
    /// live `VamanaProjection`'s ANN index through the writer that owns
    /// it. Every impl is the one-liner `{ self }`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Never persist a checkpoint at or beyond `lsn` for the rest of this
    /// projection's lifetime. Called by the deferred worker when the apply
    /// at `lsn` failed but later records keep applying: without the hold the
    /// checkpoint would advance past the failed record and replay would never
    /// retry it. Projections that persist no checkpoint keep the default.
    fn hold_checkpoint_below(&mut self, _lsn: Lsn) {}

    /// Persist any checkpoint progress the projection is still holding in
    /// memory. Called when a deferred queue has been drained.
    fn flush_checkpoint(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

/// Tuning for one deferred projection's queue.
#[derive(Debug, Clone, Copy)]
pub struct DeferredLaneConfig {
    /// Records that may be queued before the writer waits for room.
    pub capacity: usize,
    /// How long the writer waits for room before applying inline.
    pub enqueue_wait: Duration,
}

impl Default for DeferredLaneConfig {
    fn default() -> Self {
        Self {
            capacity: DEFERRED_QUEUE_CAPACITY,
            enqueue_wait: DEFERRED_ENQUEUE_WAIT,
        }
    }
}

struct LaneState {
    queue: VecDeque<(Lsn, Arc<IntentPayload>)>,
    /// The worker has popped a record and is applying it.
    in_flight: bool,
    stop: bool,
    exited: bool,
}

/// State shared between a deferred projection's writer side and its worker.
///
/// Lock order is `projection` then `state`; nothing takes `projection` while
/// holding `state`. The worker pops a record only while holding
/// `projection`, so whoever holds `projection` knows no record is in flight,
/// which is what lets the inline fallback apply the backlog in order.
struct LaneShared {
    name: String,
    projection: Mutex<Box<dyn TypedProjection>>,
    state: Mutex<LaneState>,
    /// Signalled when a record is queued or the worker is told to stop.
    work_ready: Condvar,
    /// Signalled when a record leaves the queue, an apply finishes, or the
    /// worker exits.
    progress: Condvar,
    inline_fallbacks: AtomicU64,
    apply_failures: AtomicU64,
}

impl LaneShared {
    /// Apply one record, recording metrics. A failure or panic holds the
    /// checkpoint below `lsn` and is returned to the caller.
    fn apply_one(
        &self,
        projection: &mut Box<dyn TypedProjection>,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let timer = Timer::new(PROJECTION_APPLY_DURATION.with_label_values(&[self.name.as_str()]));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            projection.apply(lsn, payload)
        }))
        .unwrap_or_else(|_| Err("projection apply panicked".into()));
        drop(timer);
        let outcome = if result.is_ok() { "ok" } else { "error" };
        PROJECTION_APPLY_TOTAL
            .with_label_values(&[self.name.as_str(), outcome])
            .inc();
        if result.is_err() {
            self.apply_failures.fetch_add(1, Ordering::Relaxed);
            projection.hold_checkpoint_below(lsn);
        }
        result
    }

    /// Queue a record, waiting up to `wait` for room. Hands the record back
    /// when the queue stays full or the worker has been stopped.
    fn try_enqueue(
        &self,
        lsn: Lsn,
        payload: Arc<IntentPayload>,
        config: DeferredLaneConfig,
    ) -> Result<(), Arc<IntentPayload>> {
        let capacity = config.capacity.max(1);
        let deadline = Instant::now() + config.enqueue_wait;
        let mut st = self.state.lock();
        while st.queue.len() >= capacity && !st.stop {
            if self.progress.wait_until(&mut st, deadline).timed_out() {
                break;
            }
        }
        if st.stop || st.queue.len() >= capacity {
            return Err(payload);
        }
        st.queue.push_back((lsn, payload));
        drop(st);
        self.work_ready.notify_one();
        Ok(())
    }

    /// Apply everything still queued and then `(lsn, payload)` on the calling
    /// thread, in LSN order. Backlog failures are logged here; the failure of
    /// the current record is returned.
    fn apply_inline(
        &self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut projection = self.projection.lock();
        let backlog: Vec<_> = self.state.lock().queue.drain(..).collect();
        if !backlog.is_empty() {
            self.progress.notify_all();
        }
        for (queued_lsn, queued) in backlog {
            if let Err(error) = self.apply_one(&mut projection, queued_lsn, &queued) {
                tracing::warn!(
                    projection = %self.name,
                    lsn = queued_lsn.0,
                    error = %error,
                    "deferred projection apply failed during inline catch-up (log durable; replay will retry)",
                );
            }
        }
        self.apply_one(&mut projection, lsn, payload)
    }

    /// Wait until the queue is empty and nothing is in flight. Returns
    /// `false` when `deadline` passes first or the worker exited with work
    /// still queued.
    fn wait_idle(&self, deadline: Instant) -> bool {
        let mut st = self.state.lock();
        loop {
            if st.queue.is_empty() && !st.in_flight {
                return true;
            }
            if st.exited {
                return false;
            }
            if self.progress.wait_until(&mut st, deadline).timed_out() {
                return st.queue.is_empty() && !st.in_flight;
            }
        }
    }

    fn wait_exited(&self, deadline: Instant) -> bool {
        let mut st = self.state.lock();
        while !st.exited {
            if self.progress.wait_until(&mut st, deadline).timed_out() {
                return st.exited;
            }
        }
        true
    }

    /// Persist the projection's checkpoint, giving up if the projection is
    /// still busy at `deadline`.
    fn flush(&self, deadline: Instant) {
        let Some(mut projection) = self.projection.try_lock_until(deadline) else {
            tracing::warn!(
                projection = %self.name,
                "deferred projection still applying at drain deadline; checkpoint not flushed (replay will recover the tail)",
            );
            return;
        };
        if let Err(error) = projection.flush_checkpoint() {
            tracing::warn!(
                projection = %self.name,
                error = %error,
                "deferred projection checkpoint flush failed (replay will recover the tail)",
            );
        }
    }

    fn run_worker(self: Arc<Self>) {
        loop {
            {
                let mut st = self.state.lock();
                while st.queue.is_empty() && !st.stop {
                    self.work_ready.wait(&mut st);
                }
                if st.stop {
                    break;
                }
            }
            let mut projection = self.projection.lock();
            let next = {
                let mut st = self.state.lock();
                let next = st.queue.pop_front();
                st.in_flight = next.is_some();
                next
            };
            // The inline fallback may have taken the queue while this thread
            // waited for the projection lock.
            let Some((lsn, payload)) = next else {
                continue;
            };
            self.progress.notify_all();
            if let Err(error) = self.apply_one(&mut projection, lsn, &payload) {
                tracing::warn!(
                    projection = %self.name,
                    lsn = lsn.0,
                    error = %error,
                    "deferred projection apply failed (log durable; checkpoint held, replay will retry)",
                );
            }
            self.state.lock().in_flight = false;
            drop(projection);
            self.progress.notify_all();
        }
        if let Err(error) = self.projection.lock().flush_checkpoint() {
            tracing::warn!(
                projection = %self.name,
                error = %error,
                "deferred projection checkpoint flush failed at worker exit (replay will recover the tail)",
            );
        }
        self.state.lock().exited = true;
        self.progress.notify_all();
    }
}

/// One deferred projection: its shared state plus the lazily started worker.
struct DeferredLane {
    shared: Arc<LaneShared>,
    config: DeferredLaneConfig,
    worker: Option<std::thread::JoinHandle<()>>,
    shut_down: bool,
}

impl DeferredLane {
    fn ensure_worker(&mut self) -> bool {
        if self.worker.is_some() {
            return true;
        }
        let shared = Arc::clone(&self.shared);
        match std::thread::Builder::new()
            .name(format!("veld-deferred-{}", self.shared.name))
            .spawn(move || shared.run_worker())
        {
            Ok(handle) => {
                self.worker = Some(handle);
                true
            }
            Err(error) => {
                tracing::warn!(
                    projection = %self.shared.name,
                    error = %error,
                    "could not start deferred projection worker; applying inline",
                );
                false
            }
        }
    }

    /// Hand a durable record to the lane. Returns an error only when the
    /// record had to be applied inline and that apply failed.
    fn submit(
        &mut self,
        lsn: Lsn,
        payload: Arc<IntentPayload>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.shut_down || !self.ensure_worker() {
            return self.shared.apply_inline(lsn, &payload);
        }
        let payload = match self.shared.try_enqueue(lsn, payload, self.config) {
            Ok(()) => return Ok(()),
            Err(payload) => payload,
        };
        let fallbacks = self.shared.inline_fallbacks.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            projection = %self.shared.name,
            lsn = lsn.0,
            capacity = self.config.capacity,
            wait_ms = self.config.enqueue_wait.as_millis() as u64,
            inline_fallbacks = fallbacks,
            "deferred projection queue full; applying backlog and record inline",
        );
        self.shared.apply_inline(lsn, &payload)
    }

    fn shutdown(&mut self, deadline: Instant) -> bool {
        if self.shut_down {
            return true;
        }
        self.shut_down = true;
        let drained = self.shared.wait_idle(deadline);
        self.shared.state.lock().stop = true;
        self.shared.work_ready.notify_all();
        match self.worker.take() {
            Some(handle) => {
                if self.shared.wait_exited(deadline) {
                    if handle.join().is_err() {
                        tracing::warn!(
                            projection = %self.shared.name,
                            "deferred projection worker panicked during shutdown",
                        );
                    }
                } else {
                    // Detached: the worker exits after its current apply and
                    // flushes the checkpoint itself; replay covers anything
                    // left in the queue.
                    tracing::warn!(
                        projection = %self.shared.name,
                        backlog = self.shared.state.lock().queue.len(),
                        "deferred projection did not drain before the deadline; worker detached",
                    );
                }
            }
            None => self.shared.flush(deadline),
        }
        drained
    }
}

/// Cloneable handle that drains a writer's deferred projections without
/// holding the writer itself, so a slow drain never blocks journaled writes.
#[derive(Clone)]
pub struct DeferredDrainHandle {
    lanes: Vec<Arc<LaneShared>>,
}

impl DeferredDrainHandle {
    /// Wait until every deferred queue is empty, then persist each
    /// projection's checkpoint. Returns `false` if `timeout` passed first.
    pub fn drain(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut drained = true;
        for lane in &self.lanes {
            if !lane.wait_idle(deadline) {
                drained = false;
                tracing::warn!(
                    projection = %lane.name,
                    backlog = lane.state.lock().queue.len(),
                    "deferred projection did not drain before the deadline",
                );
            }
            lane.flush(deadline);
        }
        drained
    }
}

/// Errors raised by [`JournaledWriter::record_and_apply`].
#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error("intent log error: {0}")]
    Log(#[from] IntentLogError),
    #[error("payload encode error: {0}")]
    Payload(#[from] PayloadError),
    /// The thread committing this write's group stopped before it recorded
    /// the write's outcome, so whether its records are durable is unknown.
    #[error("journal write unresolved: the thread committing its group stopped before recording its outcome")]
    Unresolved,
}

/// Outcome of a successful `record_and_apply`. The LSN is the log
/// position that was assigned; `apply_errors` holds any per-projection
/// failures (these do not fail the call — the log entry survives, and
/// replay will retry on restart).
#[derive(Debug)]
pub struct WriteOutcome {
    pub lsn: Lsn,
    pub apply_errors: Vec<ApplyError>,
}

/// One projection's failed apply during a journaled write. Carried in
/// `WriteOutcome.apply_errors` so the caller can log/metrics-emit per
/// projection without taking down the whole write path.
#[derive(Debug)]
pub struct ApplyError {
    pub projection: String,
    pub source: Box<dyn Error + Send + Sync>,
}

/// Ties an [`IntentLog`] together with a list of [`TypedProjection`]s and
/// orchestrates the log-first write pattern.
pub struct JournaledWriter {
    log: IntentLog,
    projections: Vec<Box<dyn TypedProjection>>,
    deferred: Vec<DeferredLane>,
    checkpoint_store: Option<Arc<Mutex<CheckpointStore>>>,
}

impl JournaledWriter {
    /// Create a writer around an open log. Projections are added with
    /// [`add_projection`]; ordering only matters in that earlier
    /// projections see `apply` calls before later ones within the same
    /// `record_and_apply` invocation.
    pub fn new(log: IntentLog) -> Self {
        Self {
            log,
            projections: Vec::new(),
            deferred: Vec::new(),
            checkpoint_store: None,
        }
    }

    pub fn add_projection(&mut self, p: Box<dyn TypedProjection>) {
        self.projections.push(p);
    }

    /// Attach the tenant's checkpoint store: the writer persists the log's
    /// offset hints there (see the module docs), and
    /// [`Self::live_memory_ids`] keeps its resumable snapshot there.
    pub fn attach_checkpoint_store(&mut self, store: Arc<Mutex<CheckpointStore>>) {
        self.checkpoint_store = Some(store);
    }

    /// Persist the log's offset hints to the attached checkpoint store and
    /// make them durable. A no-op without an attached store.
    pub fn persist_frame_index(&self) -> Result<(), CheckpointStoreError> {
        match &self.checkpoint_store {
            Some(store) => store.lock().persist_frame_index(self.log.frame_index()),
            None => Ok(()),
        }
    }

    fn persist_frame_index_logged(&self) {
        if let Err(error) = self.persist_frame_index() {
            tracing::warn!(
                path = %self.log.path().display(),
                error = %error,
                "could not persist intent log offset hints (the next open scans further back)",
            );
        }
    }

    /// Memory ids the log considers live at its head (created, updated or
    /// anchored, and not since forgotten). With a checkpoint store attached
    /// this resumes from the snapshot the previous call left there and reads
    /// only the frames appended since; see
    /// [`super::projection::live_memory_ids_resumable`].
    pub fn live_memory_ids(&self) -> Result<HashSet<String>, PayloadError> {
        match &self.checkpoint_store {
            Some(store) => super::projection::live_memory_ids_resumable(&self.log, store),
            None => payload::live_memory_ids(&self.log),
        }
    }

    /// Attach a derived, non-authoritative projection whose live applies
    /// run on a dedicated worker thread instead of inside
    /// [`Self::record_and_apply`]. See the module docs for the ordering,
    /// back-pressure and checkpoint contract. The worker starts on the first
    /// record. Catch-up replay is the caller's job and must finish before
    /// the projection is attached.
    pub fn add_deferred_projection(&mut self, p: Box<dyn TypedProjection>) {
        self.add_deferred_projection_with(p, DeferredLaneConfig::default());
    }

    /// [`Self::add_deferred_projection`] with an explicit queue configuration.
    pub fn add_deferred_projection_with(
        &mut self,
        p: Box<dyn TypedProjection>,
        config: DeferredLaneConfig,
    ) {
        let shared = Arc::new(LaneShared {
            name: p.name().to_string(),
            projection: Mutex::new(p),
            state: Mutex::new(LaneState {
                queue: VecDeque::new(),
                in_flight: false,
                stop: false,
                exited: false,
            }),
            work_ready: Condvar::new(),
            progress: Condvar::new(),
            inline_fallbacks: AtomicU64::new(0),
            apply_failures: AtomicU64::new(0),
        });
        self.deferred.push(DeferredLane {
            shared,
            config,
            worker: None,
            shut_down: false,
        });
    }

    /// Number of projections currently attached, inline and deferred.
    pub fn projection_count(&self) -> usize {
        self.projections.len() + self.deferred.len()
    }

    /// Times a full deferred queue forced a record to be applied inline,
    /// summed over every deferred projection.
    pub fn deferred_inline_fallbacks(&self) -> u64 {
        self.deferred
            .iter()
            .map(|lane| lane.shared.inline_fallbacks.load(Ordering::Relaxed))
            .sum()
    }

    /// Failed deferred applies (worker or inline fallback), summed over
    /// every deferred projection.
    pub fn deferred_apply_failures(&self) -> u64 {
        self.deferred
            .iter()
            .map(|lane| lane.shared.apply_failures.load(Ordering::Relaxed))
            .sum()
    }

    /// Records queued for deferred projections and not yet picked up.
    pub fn deferred_backlog(&self) -> usize {
        self.deferred
            .iter()
            .map(|lane| lane.shared.state.lock().queue.len())
            .sum()
    }

    /// Handle that drains the deferred projections without holding the
    /// writer.
    pub fn deferred_drain_handle(&self) -> DeferredDrainHandle {
        DeferredDrainHandle {
            lanes: self
                .deferred
                .iter()
                .map(|lane| Arc::clone(&lane.shared))
                .collect(),
        }
    }

    /// Drain every deferred queue (bounded by `timeout`), persist the
    /// checkpoints and stop the workers. Records journaled afterwards apply
    /// inline. Returns `false` if a queue was still non-empty at the
    /// deadline; those records are recovered by replay on the next open.
    pub fn shutdown_deferred(&mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut drained = true;
        for lane in &mut self.deferred {
            drained &= lane.shutdown(deadline);
        }
        drained
    }

    /// Borrow an attached inline projection by its [`TypedProjection::name`].
    /// Returns the first match, or `None` if no attached projection
    /// reports that name. Used by read-side consumers (e.g. the query
    /// planner's vector adapter) to reach a live projection's state
    /// without owning it — the projections live here, behind the
    /// writer's mutex, for the lifetime of the tenant. Deferred projections
    /// live behind their worker's lock and are not reachable here.
    pub fn projection_by_name(&self, name: &str) -> Option<&dyn TypedProjection> {
        self.projections
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    /// Borrow the underlying log read-only. Useful for diagnostics —
    /// don't append directly to it, that bypasses the projection
    /// dispatch and breaks the log-first invariant.
    pub fn log(&self) -> &IntentLog {
        &self.log
    }

    /// The whole-system contract:
    ///
    ///   1. Encode `payload` to bytes.
    ///   2. Append to the log.
    ///   3. `sync` (at the log's durability).
    ///   4. Hand the record to every deferred projection's queue.
    ///   5. Walk every inline projection and call `apply(lsn, payload)`.
    ///
    /// If steps 1-3 succeed, returns `Ok(WriteOutcome)`. Any errors from
    /// step 5 (and from a deferred projection that had to apply inline
    /// because its queue was full) are collected into
    /// `WriteOutcome.apply_errors` — they do NOT fail the call. The log
    /// holds the record, so the replay driver will retry on restart;
    /// idempotency in `apply` makes that safe. Deferred applies that run
    /// on the worker are not awaited and do not appear here.
    ///
    /// If step 1, 2, or 3 fails, the call returns `Err`, no projection is
    /// touched, and the record is not in the log: `IntentLog` cuts a failed
    /// write back to the last synced frame and the next record takes its
    /// LSN. A frame torn by a crash is cut by `IntentLog::open`.
    ///
    /// Concurrent writers go through [`SharedJournal::record_and_apply`]
    /// instead, which shares step 3 between them.
    #[tracing::instrument(
        level = "info",
        name = "journal.record_and_apply",
        skip(self, payload),
        fields(
            user_id = %payload.user_id(),
            memory_id = %payload.memory_id(),
            projection_count = self.projection_count(),
            lsn = tracing::field::Empty,
        ),
    )]
    pub fn record_and_apply(
        &mut self,
        payload: &IntentPayload,
    ) -> Result<WriteOutcome, JournalError> {
        let ticket = Arc::new(Ticket::encode(std::slice::from_ref(payload))?);
        self.commit_group(std::slice::from_ref(&ticket));
        let outcome = single_outcome(
            ticket
                .take_outcome()
                .unwrap_or(Err(JournalError::Unresolved)),
        )?;
        tracing::Span::current().record("lsn", outcome.lsn.0);
        Ok(outcome)
    }

    /// [`Self::record_and_apply`] for several payloads with one sync. Every
    /// payload is encoded before anything is appended, so an encode error
    /// appends nothing; then all are appended, the log is synced once, and
    /// each record is dispatched in LSN order exactly as a single write
    /// would be. Returns one outcome per payload, in order.
    ///
    /// The batch is durable whole or not at all: if any of its records
    /// cannot be appended, or the sync fails, the call returns `Err`, the
    /// records it did append are cut from the log, and nothing is dispatched.
    #[tracing::instrument(
        level = "info",
        name = "journal.record_and_apply_batch",
        skip(self, payloads),
        fields(records = payloads.len(), projection_count = self.projection_count()),
    )]
    pub fn record_and_apply_batch(
        &mut self,
        payloads: &[IntentPayload],
    ) -> Result<Vec<WriteOutcome>, JournalError> {
        if payloads.is_empty() {
            return Ok(Vec::new());
        }
        let ticket = Arc::new(Ticket::encode(payloads)?);
        self.commit_group(std::slice::from_ref(&ticket));
        ticket
            .take_outcome()
            .unwrap_or(Err(JournalError::Unresolved))
    }

    /// Commit every write in `group` with one sync: append their records in
    /// order, sync once, dispatch each record in LSN order, and record each
    /// write's outcome on its ticket.
    ///
    /// A write whose record cannot be appended fails alone: the log takes
    /// back everything appended since its last sync and the rest of the group
    /// is appended again, so no write is ever left partly in the log and no
    /// other write's outcome depends on its payload. A failed sync fails every
    /// write in the group, since the log rolled all of their records back.
    #[tracing::instrument(
        level = "debug",
        name = "journal.commit_group",
        skip_all,
        fields(writes = group.len()),
    )]
    fn commit_group(&mut self, group: &[Arc<Ticket>]) {
        let appended = loop {
            match self.append_group(group) {
                Ok(appended) => break appended,
                Err((failed, error)) => {
                    tracing::error!(
                        error = %error,
                        records = group[failed].records.len(),
                        "journal write failed: intent log append error",
                    );
                    self.log.discard_unsynced();
                    group[failed].resolve(Err(JournalError::Log(error)));
                }
            }
        };
        if appended.is_empty() {
            return;
        }
        if let Err(error) = self.log.sync() {
            tracing::error!(
                error = %error,
                writes = appended.len(),
                "journal write failed: intent log sync error",
            );
            let copies = appended.len() - 1;
            for (index, _) in &appended[..copies] {
                group[*index].resolve(Err(JournalError::Log(copy_log_error(&error))));
            }
            group[appended[copies].0].resolve(Err(JournalError::Log(error)));
            return;
        }
        if appended
            .iter()
            .flat_map(|(_, lsns)| lsns)
            .any(|lsn| lsn.0.is_multiple_of(FRAME_INDEX_STRIDE))
        {
            self.persist_frame_index_logged();
        }
        for (index, lsns) in appended {
            let ticket = &group[index];
            let outcomes = lsns
                .into_iter()
                .zip(&ticket.records)
                .map(|(lsn, (_, payload))| WriteOutcome {
                    lsn,
                    apply_errors: self.dispatch(lsn, payload),
                })
                .collect();
            ticket.resolve(Ok(outcomes));
        }
    }

    /// Append the records of every unresolved write in `group`. On a failed
    /// append, the index of the write it belonged to and the error.
    fn append_group(&mut self, group: &[Arc<Ticket>]) -> Result<Appended, (usize, IntentLogError)> {
        let mut appended = Vec::with_capacity(group.len());
        for (index, ticket) in group.iter().enumerate() {
            if ticket.is_resolved() {
                continue;
            }
            let mut lsns = Vec::with_capacity(ticket.records.len());
            for (bytes, _) in &ticket.records {
                lsns.push(self.log.append(bytes).map_err(|error| (index, error))?);
            }
            appended.push((index, lsns));
        }
        Ok(appended)
    }

    /// Hand a durable record to every deferred projection and apply it to
    /// every inline projection, collecting the failures.
    fn dispatch(&mut self, lsn: Lsn, payload: &Arc<IntentPayload>) -> Vec<ApplyError> {
        let mut apply_errors = Vec::new();
        for lane in &mut self.deferred {
            if let Err(source) = lane.submit(lsn, Arc::clone(payload)) {
                tracing::error!(
                    projection = %lane.shared.name,
                    lsn = lsn.0,
                    error = %source,
                    "deferred projection inline apply failed during journaled write",
                );
                apply_errors.push(ApplyError {
                    projection: lane.shared.name.clone(),
                    source,
                });
            }
        }
        for p in &mut self.projections {
            let projection_name = p.name().to_string();
            let timer = Timer::new(
                PROJECTION_APPLY_DURATION.with_label_values(&[projection_name.as_str()]),
            );
            // A panic is contained here because the records after this one,
            // possibly other writers' in the same group, still have to reach
            // every projection; it is reported like a failed apply.
            let apply_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.apply(lsn, payload)))
                    .unwrap_or_else(|_| {
                        p.hold_checkpoint_below(lsn);
                        Err("projection apply panicked".into())
                    });
            drop(timer);
            match apply_result {
                Ok(()) => {
                    PROJECTION_APPLY_TOTAL
                        .with_label_values(&[projection_name.as_str(), "ok"])
                        .inc();
                }
                Err(source) => {
                    PROJECTION_APPLY_TOTAL
                        .with_label_values(&[projection_name.as_str(), "error"])
                        .inc();
                    tracing::error!(
                        projection = %projection_name,
                        lsn = lsn.0,
                        error = %source,
                        "projection apply failed during journaled write",
                    );
                    apply_errors.push(ApplyError {
                        projection: projection_name,
                        source,
                    });
                }
            }
        }
        apply_errors
    }
}

impl Drop for JournaledWriter {
    fn drop(&mut self) {
        self.shutdown_deferred(DEFERRED_DRAIN_TIMEOUT);
        self.persist_frame_index_logged();
    }
}

/// The outcome of one write, as [`JournaledWriter::commit_group`] leaves it.
type TicketOutcome = Result<Vec<WriteOutcome>, JournalError>;

/// Each appended write's index in its group, with the LSNs its records took.
type Appended = Vec<(usize, Vec<Lsn>)>;

/// One write waiting to be committed: its records, encoded before any lock is
/// taken, and the outcome the thread that commits it leaves for its owner.
struct Ticket {
    records: Vec<(Vec<u8>, Arc<IntentPayload>)>,
    outcome: Mutex<Option<TicketOutcome>>,
}

impl Ticket {
    fn encode(payloads: &[IntentPayload]) -> Result<Self, JournalError> {
        let records = payloads
            .iter()
            .map(|p| Ok((payload::encode(p)?, Arc::new(p.clone()))))
            .collect::<Result<Vec<_>, PayloadError>>()
            .map_err(|error| {
                tracing::error!(error = %error, "journal write failed: payload encode error");
                JournalError::Payload(error)
            })?;
        Ok(Self {
            records,
            outcome: Mutex::new(None),
        })
    }

    fn resolve(&self, outcome: TicketOutcome) {
        *self.outcome.lock() = Some(outcome);
    }

    fn is_resolved(&self) -> bool {
        self.outcome.lock().is_some()
    }

    fn take_outcome(&self) -> Option<TicketOutcome> {
        self.outcome.lock().take()
    }
}

/// The outcome of a one-record write.
fn single_outcome(outcome: TicketOutcome) -> Result<WriteOutcome, JournalError> {
    outcome?.pop().ok_or(JournalError::Unresolved)
}

/// The same failure for another write whose records one failed sync rolled
/// back: `io::Error` cannot be cloned, so its kind and message are carried.
fn copy_log_error(error: &IntentLogError) -> IntentLogError {
    let kind = match error {
        IntentLogError::Io(e) => e.kind(),
        _ => io::ErrorKind::Other,
    };
    IntentLogError::Io(io::Error::new(kind, error.to_string()))
}

/// A tenant's [`JournaledWriter`], shared by the threads that write to it so
/// that concurrent writes share one sync (see "Group commit" in the module
/// docs).
pub struct SharedJournal {
    writer: Mutex<JournaledWriter>,
    /// Writes queued for the next group, in arrival order.
    queued: Mutex<Vec<Arc<Ticket>>>,
}

impl SharedJournal {
    pub fn new(writer: JournaledWriter) -> Self {
        Self {
            writer: Mutex::new(writer),
            queued: Mutex::new(Vec::new()),
        }
    }

    /// Lock the writer: to read the log or a projection, or to keep writes
    /// out while the log is copied or swapped. Every write committed before
    /// the lock was granted is synced and applied.
    pub fn lock(&self) -> MutexGuard<'_, JournaledWriter> {
        self.writer.lock()
    }

    /// [`JournaledWriter::record_and_apply`], committed in one group with
    /// the other writes queued for this journal.
    #[tracing::instrument(
        level = "info",
        name = "journal.record_and_apply",
        skip(self, payload),
        fields(
            user_id = %payload.user_id(),
            memory_id = %payload.memory_id(),
            lsn = tracing::field::Empty,
        ),
    )]
    pub fn record_and_apply(&self, payload: &IntentPayload) -> Result<WriteOutcome, JournalError> {
        let ticket = Ticket::encode(std::slice::from_ref(payload))?;
        let outcome = single_outcome(self.commit(ticket))?;
        tracing::Span::current().record("lsn", outcome.lsn.0);
        Ok(outcome)
    }

    /// [`JournaledWriter::record_and_apply_batch`], committed in one group
    /// with the other writes queued for this journal. The batch's records
    /// stay contiguous and it remains durable whole or not at all.
    #[tracing::instrument(
        level = "info",
        name = "journal.record_and_apply_batch",
        skip(self, payloads),
        fields(records = payloads.len()),
    )]
    pub fn record_and_apply_batch(
        &self,
        payloads: &[IntentPayload],
    ) -> Result<Vec<WriteOutcome>, JournalError> {
        if payloads.is_empty() {
            return Ok(Vec::new());
        }
        self.commit(Ticket::encode(payloads)?)
    }

    fn commit(&self, ticket: Ticket) -> TicketOutcome {
        let ticket = Arc::new(ticket);
        self.queued.lock().push(Arc::clone(&ticket));
        let mut writer = self.writer.lock();
        // A group commits under the writer's mutex and records every outcome
        // before releasing it, so once the mutex is ours this write is either
        // committed already or still queued, never in another group's hands.
        if let Some(outcome) = ticket.take_outcome() {
            drop(writer);
            if outcome.is_ok() {
                INTENT_LOG_SYNC_SHARED_TOTAL.inc();
            }
            return outcome;
        }
        let group = std::mem::take(&mut *self.queued.lock());
        writer.commit_group(&group);
        drop(writer);
        ticket
            .take_outcome()
            .unwrap_or(Err(JournalError::Unresolved))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// In-memory projection that records every (lsn, payload) it sees.
    struct RecordingProjection {
        name: String,
        applied: Arc<Mutex<Vec<(Lsn, IntentPayload)>>>,
    }

    impl TypedProjection for RecordingProjection {
        fn name(&self) -> &str {
            &self.name
        }
        fn apply(
            &mut self,
            lsn: Lsn,
            payload: &IntentPayload,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            self.applied.lock().unwrap().push((lsn, payload.clone()));
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Projection that always fails on apply, so we can verify the
    /// "errors collected, write still succeeds" pathway.
    struct FailingProjection {
        name: String,
    }
    impl TypedProjection for FailingProjection {
        fn name(&self) -> &str {
            &self.name
        }
        fn apply(
            &mut self,
            _lsn: Lsn,
            _payload: &IntentPayload,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            Err("synthetic failure".into())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("veld-journal-{name}-{pid}-{stamp}.log"));
        p
    }

    #[test]
    fn record_and_apply_dispatches_to_every_projection() {
        let path = tmp_path("dispatch");
        let log = IntentLog::open(&path).unwrap();
        let mut writer = JournaledWriter::new(log);

        let a_seen = Arc::new(Mutex::new(Vec::new()));
        let b_seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(RecordingProjection {
            name: "a".into(),
            applied: a_seen.clone(),
        }));
        writer.add_projection(Box::new(RecordingProjection {
            name: "b".into(),
            applied: b_seen.clone(),
        }));

        let payload = IntentPayload::Forget {
            user_id: "u".into(),
            memory_id: "m-1".into(),
            schema_version: None,
        };
        let outcome = writer.record_and_apply(&payload).unwrap();
        assert_eq!(outcome.lsn, Lsn(0));
        assert!(outcome.apply_errors.is_empty());

        let a = a_seen.lock().unwrap();
        let b = b_seen.lock().unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0], (Lsn(0), payload.clone()));
        assert_eq!(b[0], (Lsn(0), payload));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lsns_advance_monotonically_across_calls() {
        let path = tmp_path("lsn_mono");
        let log = IntentLog::open(&path).unwrap();
        let mut writer = JournaledWriter::new(log);
        let seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(RecordingProjection {
            name: "rec".into(),
            applied: seen.clone(),
        }));

        let payloads: Vec<IntentPayload> = (0..5)
            .map(|i| IntentPayload::Anchor {
                user_id: "u".into(),
                memory_id: format!("m-{i}"),
                importance: 0.5,
                schema_version: None,
            })
            .collect();
        for p in &payloads {
            writer.record_and_apply(p).unwrap();
        }
        let seen = seen.lock().unwrap();
        let lsns: Vec<_> = seen.iter().map(|(l, _)| *l).collect();
        assert_eq!(lsns, vec![Lsn(0), Lsn(1), Lsn(2), Lsn(3), Lsn(4)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failing_projection_does_not_fail_the_write() {
        let path = tmp_path("apply_err");
        let log = IntentLog::open(&path).unwrap();
        let mut writer = JournaledWriter::new(log);
        let good_seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(FailingProjection { name: "bad".into() }));
        writer.add_projection(Box::new(RecordingProjection {
            name: "good".into(),
            applied: good_seen.clone(),
        }));

        let payload = IntentPayload::Remember {
            user_id: "u".into(),
            memory_id: "m".into(),
            memory_bincode: vec![1, 2, 3],
            schema_version: None,
        };
        let outcome = writer.record_and_apply(&payload).unwrap();
        assert_eq!(outcome.lsn, Lsn(0));
        // Bad projection collected its error.
        assert_eq!(outcome.apply_errors.len(), 1);
        assert_eq!(outcome.apply_errors[0].projection, "bad");
        // Good projection still ran.
        assert_eq!(good_seen.lock().unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_holds_the_record_even_when_every_projection_fails() {
        let path = tmp_path("log_survives");
        {
            let log = IntentLog::open(&path).unwrap();
            let mut writer = JournaledWriter::new(log);
            writer.add_projection(Box::new(FailingProjection { name: "x".into() }));
            let payload = IntentPayload::Forget {
                user_id: "u".into(),
                memory_id: "m".into(),
                schema_version: None,
            };
            let outcome = writer.record_and_apply(&payload).unwrap();
            assert_eq!(outcome.apply_errors.len(), 1);
        }
        // Reopen and confirm the record is durable.
        let log = IntentLog::open(&path).unwrap();
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    /// Checkpoint bookkeeping of a [`MirrorProjection`], shaped like the
    /// SQLite projection's: an in-memory cursor, a hold set by a failed
    /// deferred apply, and the value last persisted.
    #[derive(Default)]
    struct MirrorState {
        applied: Vec<Lsn>,
        cursor: Option<Lsn>,
        hold: Option<Lsn>,
        persisted: Option<Lsn>,
        flushes: u32,
    }

    /// Deferred-lane test double: optionally waits on a gate or sleeps
    /// before applying, and optionally fails at one LSN.
    struct MirrorProjection {
        state: Arc<Mutex<MirrorState>>,
        gate: Option<Arc<(Mutex<bool>, std::sync::Condvar)>>,
        delay: Duration,
        fail_at: Option<Lsn>,
    }

    impl MirrorProjection {
        fn new(state: Arc<Mutex<MirrorState>>) -> Self {
            Self {
                state,
                gate: None,
                delay: Duration::ZERO,
                fail_at: None,
            }
        }
    }

    impl TypedProjection for MirrorProjection {
        fn name(&self) -> &str {
            "mirror"
        }
        fn apply(
            &mut self,
            lsn: Lsn,
            _payload: &IntentPayload,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            if let Some(gate) = &self.gate {
                let (open, cv) = &**gate;
                let mut open = open.lock().unwrap();
                while !*open {
                    open = cv.wait(open).unwrap();
                }
            }
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            if Some(lsn) == self.fail_at {
                return Err("synthetic deferred failure".into());
            }
            let mut st = self.state.lock().unwrap();
            st.applied.push(lsn);
            st.cursor = Some(lsn);
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn hold_checkpoint_below(&mut self, lsn: Lsn) {
            let mut st = self.state.lock().unwrap();
            st.hold = Some(st.hold.map_or(lsn, |h| h.min(lsn)));
        }
        fn flush_checkpoint(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
            let mut st = self.state.lock().unwrap();
            st.flushes += 1;
            st.persisted = match (st.cursor, st.hold) {
                (cursor, None) => cursor,
                (_, Some(Lsn(0))) => None,
                (cursor, Some(h)) => cursor.map(|c| c.min(Lsn(h.0 - 1))),
            };
            Ok(())
        }
    }

    fn forget(i: u64) -> IntentPayload {
        IntentPayload::Forget {
            user_id: "u".into(),
            memory_id: format!("m-{i}"),
            schema_version: None,
        }
    }

    #[test]
    fn deferred_apply_runs_after_ack_in_lsn_order_and_inline_runs_before_ack() {
        let path = tmp_path("deferred_order");
        let mut writer = JournaledWriter::new(IntentLog::open(&path).unwrap());
        let inline_seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(RecordingProjection {
            name: "inline".into(),
            applied: inline_seen.clone(),
        }));
        let mirror = Arc::new(Mutex::new(MirrorState::default()));
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let mut deferred = MirrorProjection::new(mirror.clone());
        deferred.gate = Some(gate.clone());
        writer.add_deferred_projection(Box::new(deferred));
        assert_eq!(writer.projection_count(), 2);

        for i in 0..5u64 {
            let outcome = writer.record_and_apply(&forget(i)).unwrap();
            assert_eq!(outcome.lsn, Lsn(i));
            assert!(outcome.apply_errors.is_empty());
            // The inline projection has the record by the time of the ack…
            assert_eq!(inline_seen.lock().unwrap().last().unwrap().0, Lsn(i));
            // …while the gated deferred projection has not applied anything.
            assert!(mirror.lock().unwrap().applied.is_empty());
        }

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        assert!(writer
            .deferred_drain_handle()
            .drain(Duration::from_secs(10)));
        let st = mirror.lock().unwrap();
        assert_eq!(st.applied, (0..5).map(Lsn).collect::<Vec<_>>());
        assert_eq!(st.persisted, Some(Lsn(4)));
        assert_eq!(writer.deferred_inline_fallbacks(), 0);
        assert_eq!(writer.deferred_backlog(), 0);
        drop(st);
        drop(writer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn full_deferred_queue_falls_back_to_inline_apply_in_order() {
        let path = tmp_path("deferred_backpressure");
        let mut writer = JournaledWriter::new(IntentLog::open(&path).unwrap());
        let mirror = Arc::new(Mutex::new(MirrorState::default()));
        let mut deferred = MirrorProjection::new(mirror.clone());
        deferred.delay = Duration::from_millis(100);
        writer.add_deferred_projection_with(
            Box::new(deferred),
            DeferredLaneConfig {
                capacity: 1,
                enqueue_wait: Duration::from_millis(5),
            },
        );

        let mut fallbacks_seen = 0;
        for i in 0..6u64 {
            let before = writer.deferred_inline_fallbacks();
            let outcome = writer.record_and_apply(&forget(i)).unwrap();
            assert!(outcome.apply_errors.is_empty());
            if writer.deferred_inline_fallbacks() > before {
                fallbacks_seen += 1;
                // An inline fallback applies the record before the ack.
                assert!(mirror.lock().unwrap().applied.contains(&Lsn(i)));
            }
        }
        assert!(fallbacks_seen >= 1, "a capacity-1 queue must overflow");
        assert_eq!(writer.deferred_inline_fallbacks(), fallbacks_seen);

        assert!(writer.shutdown_deferred(Duration::from_secs(10)));
        let st = mirror.lock().unwrap();
        assert_eq!(st.applied, (0..6).map(Lsn).collect::<Vec<_>>());
        assert_eq!(st.persisted, Some(Lsn(5)));
        drop(st);
        drop(writer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dropping_the_writer_drains_the_deferred_queue() {
        let path = tmp_path("deferred_drop");
        let mut writer = JournaledWriter::new(IntentLog::open(&path).unwrap());
        let mirror = Arc::new(Mutex::new(MirrorState::default()));
        let mut deferred = MirrorProjection::new(mirror.clone());
        deferred.delay = Duration::from_millis(20);
        writer.add_deferred_projection(Box::new(deferred));
        for i in 0..5u64 {
            writer.record_and_apply(&forget(i)).unwrap();
        }
        drop(writer);

        let st = mirror.lock().unwrap();
        assert_eq!(st.applied, (0..5).map(Lsn).collect::<Vec<_>>());
        assert!(st.flushes >= 1);
        assert_eq!(st.persisted, Some(Lsn(4)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failed_deferred_apply_does_not_advance_the_checkpoint() {
        let path = tmp_path("deferred_fail");
        let mut writer = JournaledWriter::new(IntentLog::open(&path).unwrap());
        let mirror = Arc::new(Mutex::new(MirrorState::default()));
        let mut deferred = MirrorProjection::new(mirror.clone());
        deferred.fail_at = Some(Lsn(1));
        writer.add_deferred_projection(Box::new(deferred));

        for i in 0..3u64 {
            let outcome = writer.record_and_apply(&forget(i)).unwrap();
            // Worker-side failures are not part of the ack.
            assert!(outcome.apply_errors.is_empty());
        }
        assert!(writer
            .deferred_drain_handle()
            .drain(Duration::from_secs(10)));
        assert_eq!(writer.deferred_apply_failures(), 1);

        let st = mirror.lock().unwrap();
        // The worker kept going past the failure…
        assert_eq!(st.applied, vec![Lsn(0), Lsn(2)]);
        // …but the persisted checkpoint stays below the failed LSN.
        assert_eq!(st.hold, Some(Lsn(1)));
        assert_eq!(st.persisted, Some(Lsn(0)));
        drop(st);
        drop(writer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn projection_count_reports_attached_count() {
        let path = tmp_path("count");
        let log = IntentLog::open(&path).unwrap();
        let mut w = JournaledWriter::new(log);
        assert_eq!(w.projection_count(), 0);
        w.add_projection(Box::new(FailingProjection { name: "a".into() }));
        w.add_projection(Box::new(FailingProjection { name: "b".into() }));
        assert_eq!(w.projection_count(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn batch_appends_with_consecutive_lsns_and_dispatches_in_order() {
        let path = tmp_path("batch");
        let mut writer = JournaledWriter::new(IntentLog::open(&path).unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(RecordingProjection {
            name: "rec".into(),
            applied: seen.clone(),
        }));
        writer.record_and_apply(&forget(0)).unwrap();
        let batch: Vec<IntentPayload> = (1..5).map(forget).collect();
        let outcomes = writer.record_and_apply_batch(&batch).unwrap();
        assert_eq!(
            outcomes.iter().map(|o| o.lsn).collect::<Vec<_>>(),
            (1..5).map(Lsn).collect::<Vec<_>>()
        );
        assert!(outcomes.iter().all(|o| o.apply_errors.is_empty()));
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 5);
        for (i, (lsn, payload)) in seen.iter().enumerate() {
            assert_eq!(*lsn, Lsn(i as u64));
            assert_eq!(*payload, forget(i as u64));
        }
        assert!(writer.record_and_apply_batch(&[]).unwrap().is_empty());
        drop(seen);
        drop(writer);
        let records = IntentLog::open(&path).unwrap().iter().unwrap().count();
        assert_eq!(records, 5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn offset_hints_are_persisted_on_the_stride_and_on_drop() {
        use crate::intent_log::{CheckpointStore, FRAME_INDEX_STRIDE};
        let path = tmp_path("hints");
        let ckpt = tmp_path("hints_ckpt");
        let store = Arc::new(parking_lot::Mutex::new(
            CheckpointStore::open(&ckpt).unwrap(),
        ));
        let mut writer = JournaledWriter::new(IntentLog::open(&path).unwrap());
        writer.attach_checkpoint_store(store.clone());

        let count = FRAME_INDEX_STRIDE + 100;
        let batch: Vec<IntentPayload> = (0..count).map(forget).collect();
        writer.record_and_apply_batch(&batch).unwrap();
        let persisted = store.lock().frame_index().expect("persisted on the stride");
        assert_eq!(persisted.latest().unwrap().lsn, Lsn(count - 1));
        assert!(persisted
            .sparse()
            .iter()
            .any(|p| p.lsn == Lsn(FRAME_INDEX_STRIDE)));

        writer.record_and_apply(&forget(count)).unwrap();
        drop(writer);
        let persisted = store.lock().frame_index().unwrap();
        assert_eq!(
            persisted.latest().unwrap().lsn,
            Lsn(count),
            "persisted on drop"
        );

        let reopened = IntentLog::open_with_index(&path, None, Some(persisted.clone())).unwrap();
        assert_eq!(reopened.next_lsn(), Lsn(count + 1));
        assert_eq!(reopened.frame_index(), &persisted);
        let side = store.lock().side_data_path().to_path_buf();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&ckpt);
        let _ = std::fs::remove_file(&side);
    }

    /// Open and closed flags a test thread waits on, with the condvar both use.
    #[derive(Default)]
    struct Gate {
        /// (a held apply has started, the test let it finish)
        state: Mutex<(bool, bool)>,
        cv: std::sync::Condvar,
    }

    impl Gate {
        fn wait_entered(&self) {
            let mut st = self.state.lock().unwrap();
            while !st.0 {
                st = self.cv.wait(st).unwrap();
            }
        }

        fn open(&self) {
            self.state.lock().unwrap().1 = true;
            self.cv.notify_all();
        }
    }

    /// Inline projection that records the LSN and applying thread of every
    /// record, and holds the apply of `hold_at` until its gate opens.
    struct ThreadRecorder {
        seen: Arc<Mutex<Vec<(Lsn, std::thread::ThreadId)>>>,
        hold_at: Option<Lsn>,
        gate: Arc<Gate>,
    }

    impl TypedProjection for ThreadRecorder {
        fn name(&self) -> &str {
            "thread-recorder"
        }
        fn apply(
            &mut self,
            lsn: Lsn,
            _payload: &IntentPayload,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            if Some(lsn) == self.hold_at {
                let mut st = self.gate.state.lock().unwrap();
                st.0 = true;
                self.gate.cv.notify_all();
                while !st.1 {
                    st = self.gate.cv.wait(st).unwrap();
                }
            }
            self.seen
                .lock()
                .unwrap()
                .push((lsn, std::thread::current().id()));
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    type Seen = Arc<Mutex<Vec<(Lsn, std::thread::ThreadId)>>>;

    fn shared_journal(
        path: &std::path::Path,
        hold_at: Option<Lsn>,
    ) -> (Arc<SharedJournal>, Seen, Arc<Gate>) {
        let mut writer = JournaledWriter::new(IntentLog::open(path).unwrap());
        let seen: Seen = Arc::default();
        let gate = Arc::new(Gate::default());
        writer.add_projection(Box::new(ThreadRecorder {
            seen: seen.clone(),
            hold_at,
            gate: gate.clone(),
        }));
        (Arc::new(SharedJournal::new(writer)), seen, gate)
    }

    /// Wait until `count` writes are queued behind the writer's mutex.
    fn wait_queued(shared: &SharedJournal, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while shared.queued.lock().len() < count {
            assert!(Instant::now() < deadline, "{count} writes never queued");
            std::thread::yield_now();
        }
    }

    fn logged_lsns(path: &std::path::Path) -> Vec<u64> {
        IntentLog::open(path)
            .unwrap()
            .iter()
            .unwrap()
            .map(|r| r.unwrap().lsn.0)
            .collect()
    }

    #[test]
    fn concurrent_writers_end_durable_with_contiguous_lsns_applied_in_order() {
        let path = tmp_path("group_concurrent");
        let (shared, seen, _gate) = shared_journal(&path, None);
        const THREADS: u64 = 8;
        const WRITES: u64 = 25;
        const BATCH: u64 = 3;

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    let mut lsns = Vec::new();
                    for w in 0..WRITES {
                        let id = t * 1_000 + w * 10;
                        if t % 2 == 0 {
                            lsns.push(vec![shared.record_and_apply(&forget(id)).unwrap().lsn]);
                        } else {
                            let batch: Vec<_> = (id..id + BATCH).map(forget).collect();
                            let outcomes = shared.record_and_apply_batch(&batch).unwrap();
                            lsns.push(outcomes.iter().map(|o| o.lsn).collect());
                        }
                    }
                    lsns
                })
            })
            .collect();
        let per_write: Vec<Vec<Lsn>> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        for lsns in &per_write {
            assert!(
                lsns.windows(2).all(|w| w[1] == w[0].next()),
                "a batch's records are contiguous: {lsns:?}"
            );
        }
        let mut all: Vec<u64> = per_write.iter().flatten().map(|l| l.0).collect();
        all.sort_unstable();
        let total = (THREADS / 2) * WRITES + (THREADS / 2) * WRITES * BATCH;
        assert_eq!(
            all,
            (0..total).collect::<Vec<_>>(),
            "every LSN once, no gap"
        );

        let applied: Vec<u64> = seen.lock().unwrap().iter().map(|(l, _)| l.0).collect();
        assert_eq!(
            applied,
            (0..total).collect::<Vec<_>>(),
            "applied in LSN order"
        );

        drop(shared);
        assert_eq!(logged_lsns(&path), (0..total).collect::<Vec<_>>());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writes_queued_behind_a_commit_share_the_next_sync() {
        let path = tmp_path("group_shared_sync");
        let (shared, seen, gate) = shared_journal(&path, Some(Lsn(0)));
        let shared_before = INTENT_LOG_SYNC_SHARED_TOTAL.get();

        let first = {
            let shared = shared.clone();
            std::thread::spawn(move || shared.record_and_apply(&forget(0)).unwrap().lsn)
        };
        // The first write now holds the writer's mutex, applying LSN 0.
        gate.wait_entered();
        let followers: Vec<_> = (1..=3u64)
            .map(|i| {
                let shared = shared.clone();
                std::thread::spawn(move || shared.record_and_apply(&forget(i)).unwrap().lsn)
            })
            .collect();
        wait_queued(&shared, 3);
        gate.open();

        assert_eq!(first.join().unwrap(), Lsn(0));
        let mut lsns: Vec<Lsn> = followers.into_iter().map(|h| h.join().unwrap()).collect();
        lsns.sort();
        assert_eq!(lsns, vec![Lsn(1), Lsn(2), Lsn(3)]);

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.iter().map(|(l, _)| l.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        // One thread committed all three queued writes, after one sync.
        let committer = seen[1].1;
        assert!(seen[1..].iter().all(|(_, t)| *t == committer));
        assert_ne!(committer, seen[0].1);
        assert!(INTENT_LOG_SYNC_SHARED_TOTAL.get() >= shared_before + 2);
        drop(seen);
        drop(shared);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_write_that_cannot_be_appended_fails_alone_in_its_group() {
        let path = tmp_path("group_fail_alone");
        let (shared, seen, gate) = shared_journal(&path, Some(Lsn(0)));
        let first = {
            let shared = shared.clone();
            std::thread::spawn(move || shared.record_and_apply(&forget(0)).unwrap().lsn)
        };
        gate.wait_entered();
        let too_large = IntentPayload::Remember {
            user_id: "u".into(),
            memory_id: "too-large".into(),
            memory_bincode: vec![0; crate::intent_log::MAX_PAYLOAD_BYTES],
            schema_version: None,
        };
        // Queued in this order behind the first write: a good write, one the
        // log refuses, another good one. They commit as one group.
        let mut writes = Vec::new();
        for (queued, payload) in [forget(1), too_large, forget(2)].into_iter().enumerate() {
            let writer = shared.clone();
            writes.push(std::thread::spawn(move || {
                writer.record_and_apply(&payload)
            }));
            wait_queued(&shared, queued + 1);
        }
        gate.open();

        assert_eq!(first.join().unwrap(), Lsn(0));
        let results: Vec<_> = writes.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results[0].as_ref().unwrap().lsn, Lsn(1));
        assert!(matches!(
            results[1],
            Err(JournalError::Log(IntentLogError::PayloadTooLarge { .. }))
        ));
        assert_eq!(results[2].as_ref().unwrap().lsn, Lsn(2));
        assert_eq!(
            seen.lock()
                .unwrap()
                .iter()
                .map(|(l, _)| l.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        drop(shared);
        assert_eq!(logged_lsns(&path), vec![0, 1, 2]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_crash_loses_only_the_frames_no_sync_covered() {
        use std::io::Write as _;
        let path = tmp_path("group_crash");
        let (shared, _seen, _gate) = shared_journal(&path, None);
        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    for w in 0..10 {
                        shared.record_and_apply(&forget(t * 100 + w)).unwrap();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let SharedJournal { writer, .. } = Arc::into_inner(shared).expect("writers joined");
        let mut writer = writer.into_inner();
        assert_eq!(writer.log.next_lsn(), Lsn(40));
        let committed = writer.log.durable_end_offset();

        // Two more frames reach the file, but no sync covers them.
        for i in 0..2 {
            let bytes = payload::encode(&forget(1_000 + i)).unwrap();
            writer.log.append(&bytes).unwrap();
        }
        writer.log.writer.flush().unwrap();
        // The crash: the drive kept part of the first unsynced frame, and no
        // destructor ran.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(committed + 5)
            .unwrap();
        std::mem::forget(writer);

        assert_eq!(IntentLog::open(&path).unwrap().next_lsn(), Lsn(40));
        assert_eq!(logged_lsns(&path), (0..40).collect::<Vec<_>>());
        let _ = std::fs::remove_file(&path);
    }

    /// A `Forget` whose memory id pads its frame to about 100 bytes.
    fn padded(i: u64) -> IntentPayload {
        IntentPayload::Forget {
            user_id: "u".into(),
            memory_id: format!("{i:06}-{}", "m".repeat(64)),
            schema_version: None,
        }
    }

    #[test]
    fn rotation_happens_at_the_threshold_under_concurrent_writes() {
        const LIMIT: u64 = 4096;
        const THREADS: u64 = 6;
        const WRITES: u64 = 40;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("intent.log");
        let mut log = IntentLog::open(&path).unwrap();
        log.set_segment_bytes(Some(LIMIT));
        let shared = Arc::new(SharedJournal::new(JournaledWriter::new(log)));

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    for w in 0..WRITES {
                        shared.record_and_apply(&padded(t * WRITES + w)).unwrap();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        let max_frame = (0..THREADS * WRITES)
            .map(|i| payload::encode(&padded(i)).unwrap().len() as u64 + 16)
            .max()
            .unwrap();
        let sealed = shared.lock().log().sealed().to_vec();
        assert!(sealed.len() >= 3, "only {} rotations", sealed.len());
        for segment in &sealed {
            // A head rotates at the first sync that leaves it at the limit,
            // and one group adds at most a frame per writer.
            let len = std::fs::metadata(&segment.path).unwrap().len();
            assert!(
                len >= LIMIT,
                "{:?} sealed below the limit: {len}",
                segment.path
            );
            assert!(
                len < LIMIT + THREADS * max_frame,
                "{:?} grew past the limit by more than a group: {len}",
                segment.path
            );
        }
        drop(shared);
        assert_eq!(
            logged_lsns(&path),
            (0..THREADS * WRITES).collect::<Vec<_>>()
        );
    }
}
