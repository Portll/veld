//! Public surface for sleep-time / observational memory.
//!
//! [`SleepTimeOrchestrator`] owns:
//!   - the persistent [`Queue`] (RocksDB CF)
//!   - the [`BudgetTracker`] (per-user + global, RocksDB CF)
//!   - the in-memory [`DebounceTracker`]
//!   - the [`Rewriter`] (production: Anthropic; tests: Mock)
//!   - the [`tokio_util::sync::CancellationToken`] used to drain workers on
//!     shutdown (R3 + R36)
//!   - the spawned [`tokio::task`] worker handles
//!
//! The orchestrator is bootstrapped once from `AppState::bootstrap` with the
//! shared RocksDB instance and the `Arc<MemorySystem>` it operates against.
//! It deliberately exposes a *small* API: enqueue triggers, query status,
//! toggle locks, shutdown.

use anyhow::{Context, Result};
use chrono::Duration;
use parking_lot::RwLock;
use std::sync::{Arc, Weak};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::graduation::GraduationStore;
use super::policies::{BudgetTracker, DebounceTracker, PolicyLimits};
use super::queue::Queue;
use super::rewriter::Rewriter;
use super::supersession::SupersessionStore;
use super::types::{BudgetState, SleepMode, SleepTimeResult, SleepTimeTrigger};
use crate::config::SleepTimeConfig;
use crate::earth::SharedEarth;
use crate::handlers::state::MultiUserMemoryManager;
use crate::memory::context_blocks::ContextBlockStore;

// =============================================================================
// Status reporting
// =============================================================================

/// Operator-facing status snapshot.
#[derive(Debug, Clone)]
pub struct OrchestratorStatus {
    pub enabled: bool,
    pub num_workers: usize,
    pub queue_pending_total: usize,
    pub distinct_users_in_queue: usize,
}

/// Per-user status (light — actual events surface via the consolidation event
/// bus).
#[derive(Debug, Clone)]
pub struct UserStatus {
    pub user_id: String,
    pub pending_count: usize,
    pub budget: BudgetState,
}

// =============================================================================
// Orchestrator
// =============================================================================

pub struct SleepTimeOrchestrator {
    config: SleepTimeConfig,

    queue: Arc<Queue>,
    budget: Arc<BudgetTracker>,
    debounce: Arc<DebounceTracker>,
    rewriter: Arc<Rewriter>,
    graduation: Arc<GraduationStore>,
    supersession: Arc<SupersessionStore>,

    /// Weak reference to the manager to break the Arc cycle:
    /// `MultiUserMemoryManager` holds the orchestrator inside its
    /// `sleep_time_orchestrator` field, and the orchestrator routes back to
    /// the manager to resolve per-user `SharedEarth` (and thus
    /// `MemorySystem`). `Weak` lets the manager be dropped first; on
    /// shutdown the orchestrator's `user_earth` calls fail cleanly.
    manager: Weak<MultiUserMemoryManager>,

    block_store: Arc<ContextBlockStore>,

    /// Master cancellation signal used to drain workers on shutdown (R3 /
    /// R36). `enabled=false` does NOT cancel — it only short-circuits new
    /// enqueues. Explicit `shutdown()` is the kill path.
    ///
    /// Modelled as a `tokio::sync::watch` channel because:
    ///   - `Receiver::wait_for` plays nicely with `tokio::select!` in the
    ///     worker loop, so workers wake up immediately on cancel;
    ///   - cloning a receiver is cheap (each worker holds its own copy);
    ///   - polling `is_cancelled()` is one borrow, no atomics.
    cancel_tx: Arc<watch::Sender<bool>>,
    cancel_rx: watch::Receiver<bool>,

    /// Worker join handles; populated by [`Self::start_workers`]. Held under
    /// RwLock so `shutdown()` can drain them without an exclusive owner.
    workers: Arc<RwLock<Vec<JoinHandle<()>>>>,
}

impl SleepTimeOrchestrator {
    /// Construct an orchestrator with the given config and dependencies.
    ///
    /// Does NOT start workers — callers must invoke
    /// [`Self::start_workers`] after construction so the call site has full
    /// control over when background work begins (typically after the rest
    /// of `AppState` is ready).
    ///
    /// Calling `new()` with `config.enabled=false` is supported and useful
    /// for staging: the orchestrator is constructable but inert. Enqueues
    /// are still accepted (and persisted) but the worker loop, when
    /// started, will observe `enabled=false` and skip processing.
    pub fn new(
        config: SleepTimeConfig,
        db: Arc<rocksdb::DB>,
        manager: Weak<MultiUserMemoryManager>,
        block_store: Arc<ContextBlockStore>,
        rewriter: Rewriter,
    ) -> Result<Self> {
        let limits = PolicyLimits {
            tokens_per_hour: config.tokens_per_hour,
            calls_per_day: config.calls_per_day,
            global_tokens_per_day: config.global_tokens_per_day,
            global_calls_per_day: config.global_calls_per_day,
            idle_threshold_secs: config.idle_threshold_secs,
            debounce_secs: config.debounce_secs,
        };

        let (tx, rx) = watch::channel(false);
        Ok(Self {
            queue: Arc::new(Queue::new(db.clone())),
            budget: Arc::new(BudgetTracker::new(db.clone(), limits)),
            debounce: Arc::new(DebounceTracker::new(config.debounce_secs)),
            rewriter: Arc::new(rewriter),
            graduation: Arc::new(GraduationStore::new(db.clone())),
            supersession: Arc::new(SupersessionStore::new(db)),
            manager,
            block_store,
            cancel_tx: Arc::new(tx),
            cancel_rx: rx,
            workers: Arc::new(RwLock::new(Vec::new())),
            config,
        })
    }

    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------

    /// Run the cold-start queue purge (R31 + R67). Drops queue items older
    /// than `queue_cold_start_ttl_hours`. Returns the number purged.
    ///
    /// Should be called *once*, after [`Self::new`], BEFORE
    /// [`Self::start_workers`]. Separated so callers can log the result.
    pub fn cold_start_purge(&self) -> Result<usize> {
        let ttl = Duration::hours(self.config.queue_cold_start_ttl_hours);
        self.queue.cold_start_purge(ttl)
    }

    /// Spawn worker tasks. Each worker observes the shared
    /// `CancellationToken` for graceful drain (R3 / R36).
    ///
    /// V1: stub for the worker loop lives in [`super::worker::run_worker`].
    /// This method is the integration point.
    pub fn start_workers(self: &Arc<Self>) -> Result<()> {
        if self.config.num_workers == 0 {
            // Single-worker explicit mode: spawn one anyway. Treat 0 as 1.
            self.spawn_one_worker(0);
            return Ok(());
        }
        for idx in 0..self.config.num_workers {
            self.spawn_one_worker(idx);
        }
        Ok(())
    }

    fn spawn_one_worker(self: &Arc<Self>, idx: usize) {
        let me = Arc::clone(self);
        let cancel_rx = self.cancel_rx.clone();
        let handle = tokio::spawn(async move {
            super::worker::run_worker(me, cancel_rx, idx).await;
        });
        self.workers.write().push(handle);
    }

    /// Cancel workers and await drain. Idempotent. After return all worker
    /// tasks are stopped and no further LLM calls will be made.
    pub async fn shutdown(&self) {
        // watch::Sender::send returns Err only if all receivers have been
        // dropped, which means there is nothing to cancel — ignore.
        let _ = self.cancel_tx.send(true);
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.workers.write());
        for h in handles {
            // Workers observe `cancel` cleanly and return; await each here.
            let _ = h.await;
        }
        tracing::info!("sleep-time orchestrator shut down");
    }

    // ---------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------

    pub fn config(&self) -> &SleepTimeConfig {
        &self.config
    }

    /// Resolve the per-user [`SharedEarth`] from the manager. Returns
    /// `Err` if the manager has been dropped (orchestrator outliving its
    /// owner — should only happen during shutdown) or if the user has no
    /// earth yet.
    pub(super) fn user_earth(&self, user_id: &str) -> Result<SharedEarth> {
        let manager = self
            .manager
            .upgrade()
            .context("MultiUserMemoryManager dropped; orchestrator cannot resolve user earth")?;
        manager.get_user_earth(user_id)
    }

    /// Resolve the per-user [`GraphMemory`] handle. Used by the worker to
    /// apply REM edge proposals (R43). Same drop-resilience pattern as
    /// [`Self::user_earth`].
    pub(super) fn user_graph(
        &self,
        user_id: &str,
    ) -> Result<Arc<parking_lot::RwLock<crate::graph_memory::GraphMemory>>> {
        let manager = self
            .manager
            .upgrade()
            .context("MultiUserMemoryManager dropped; orchestrator cannot resolve user graph")?;
        manager.get_user_graph(user_id)
    }

    pub(super) fn block_store(&self) -> &Arc<ContextBlockStore> {
        &self.block_store
    }

    pub(super) fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub(super) fn budget(&self) -> &Arc<BudgetTracker> {
        &self.budget
    }

    pub(super) fn rewriter(&self) -> &Arc<Rewriter> {
        &self.rewriter
    }

    pub(super) fn graduation(&self) -> &Arc<GraduationStore> {
        &self.graduation
    }

    pub(super) fn supersession(&self) -> &Arc<SupersessionStore> {
        &self.supersession
    }

    /// Public accessor for the graduation store — used by scheduled
    /// maintenance to run the graduation pass and by the forget-rewrite
    /// cascade (V3) to degraduate.
    pub fn graduation_store(&self) -> &Arc<GraduationStore> {
        &self.graduation
    }

    /// Public accessor for the supersession store — V3 forget-cascade and
    /// retrieval-ranker integration.
    pub fn supersession_store(&self) -> &Arc<SupersessionStore> {
        &self.supersession
    }

    /// Enqueue a sleep-time trigger for a user / mode pair.
    ///
    /// Applies in-memory debounce (collapses duplicate `(user, mode)`
    /// within the debounce window) followed by persistent debounce in the
    /// queue (collapses on `(user, mode, trigger)` within the same window
    /// across restarts). Returns:
    ///   - `Ok(true)`  — accepted and persisted
    ///   - `Ok(false)` — collapsed by debounce; not persisted
    ///   - `Err(...)`  — persistence failure or shutdown-in-progress
    pub fn enqueue(
        &self,
        user_id: &str,
        mode: SleepMode,
        trigger: SleepTimeTrigger,
    ) -> Result<bool> {
        if *self.cancel_rx.borrow() {
            anyhow::bail!("sleep-time orchestrator is shutting down");
        }

        // In-memory dedup first — cheap and avoids a RocksDB read on storm.
        if self.debounce.should_debounce(user_id, mode) {
            tracing::debug!(
                user_id = user_id,
                mode = mode.as_str(),
                "sleep-time enqueue collapsed by in-memory debounce"
            );
            return Ok(false);
        }

        // Persistent dedup at the queue layer (survives restart).
        let window = Duration::seconds(self.config.debounce_secs);
        match self.queue.enqueue_debounced(user_id, mode, trigger, window)? {
            Some(_item) => Ok(true),
            None => Ok(false),
        }
    }

    /// Toggle the lock state of a context block (R14). Locks are
    /// duplicated between the budget ledger (for cheap pre-flight lookup)
    /// and the `ContextBlock.locked` field (authoritative for OCC). This
    /// method writes both so they stay in sync.
    pub fn set_block_lock(&self, user_id: &str, block_key: &str, locked: bool) -> Result<()> {
        // 1. Authoritative copy on the block itself.
        self.block_store
            .set_locked(user_id, block_key, locked)
            .with_context(|| format!("set_locked on block {block_key}"))?;
        // 2. Cached copy on the budget ledger.
        self.budget
            .set_block_lock(user_id, block_key, locked)
            .with_context(|| format!("budget set_block_lock on {block_key}"))?;
        Ok(())
    }

    /// Operator-facing status snapshot.
    pub fn status(&self) -> Result<OrchestratorStatus> {
        let users = self.queue.distinct_users()?;
        let total: usize = users
            .iter()
            .map(|u| self.queue.pending_count(u).unwrap_or(0))
            .sum();
        Ok(OrchestratorStatus {
            enabled: self.config.enabled,
            num_workers: self.config.num_workers.max(1),
            queue_pending_total: total,
            distinct_users_in_queue: users.len(),
        })
    }

    pub fn user_status(&self, user_id: &str) -> Result<UserStatus> {
        Ok(UserStatus {
            user_id: user_id.to_string(),
            pending_count: self.queue.pending_count(user_id)?,
            budget: self.budget.user_state(user_id)?,
        })
    }

    /// True when the orchestrator's cancellation signal has been triggered.
    /// Currently unused inside the worker (which holds its own `cancel_rx`),
    /// but exposed for external diagnostics / future health-check callers.
    #[allow(dead_code)]
    pub(super) fn is_cancelled(&self) -> bool {
        *self.cancel_rx.borrow()
    }

    /// Pre-flight charge: returns Ok(()) if the call slot can be reserved,
    /// or `BudgetExhausted` if any cap (per-user / global) would be
    /// breached. Caller settles actual spend via [`BudgetTracker::record_actual_spend`].
    pub(super) fn try_charge_call(
        &self,
        user_id: &str,
        projected_tokens: u32,
    ) -> SleepTimeResult<()> {
        self.budget.try_charge_call(user_id, projected_tokens)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::config::SleepTimeProfile;

    #[test]
    fn profile_disabled_disables_config() {
        let cfg = SleepTimeProfile::Disabled.to_config();
        assert!(!cfg.enabled);
    }

    #[test]
    fn profile_conservative_tightens_caps() {
        let bal = SleepTimeProfile::Balanced.to_config();
        let con = SleepTimeProfile::Conservative.to_config();
        assert!(con.enabled);
        assert!(con.tokens_per_hour < bal.tokens_per_hour);
        assert!(con.calls_per_day < bal.calls_per_day);
        assert!(con.debounce_secs > bal.debounce_secs);
    }

    #[test]
    fn profile_aggressive_loosens_caps() {
        let bal = SleepTimeProfile::Balanced.to_config();
        let agg = SleepTimeProfile::Aggressive.to_config();
        assert!(agg.enabled);
        assert!(agg.tokens_per_hour > bal.tokens_per_hour);
        assert!(agg.calls_per_day > bal.calls_per_day);
        assert!(agg.debounce_secs < bal.debounce_secs);
    }
}
