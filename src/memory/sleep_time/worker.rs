//! Sleep-time worker loop.
//!
//! Each worker runs an async loop that:
//!   1. Awaits cancellation OR a poll-tick (whichever comes first).
//!   2. Tries to claim the next item from the queue via per-user fairness.
//!   3. Assembles an evidence pack, charges the budget pre-flight, calls the
//!      rewriter.
//!   4. Settles actual token spend (R39 WAL-style).
//!   5. Applies block proposals via OCC; persists observations via the
//!      `MemorySystem::remember` path.
//!   6. Emits consolidation events for every outcome (success / OCC abort /
//!      locked / budget exhausted / validation failure).
//!
//! Per-user fairness (R3): when more than one user has pending items,
//! workers pick the user whose oldest pending item is the *most stale* —
//! round-robin by virtue of "oldest waiting served first". Implemented via
//! `Queue::distinct_users` + per-user `next_unclaimed_for_user`.

use chrono::Utc;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tokio::sync::watch;

use super::diff;
use super::observation::persist_observation;
use super::observer::assemble_evidence_pack;
use super::orchestrator::SleepTimeOrchestrator;
use super::queue::Queue;
use super::types::{
    DiffClass, QueueItem, RewriteProposal, SleepMode, SleepTimeError, SleepTimeTrigger,
};
use crate::memory::context_blocks::OccOutcome;
use crate::memory::introspection::ConsolidationEvent;
use crate::memory::sleep_time::types::ObservationDraft;

/// Idle poll interval: when the queue is empty, workers sleep this long
/// before polling again. Short enough to feel responsive; long enough to
/// avoid busy-spinning RocksDB iterators.
const IDLE_POLL_INTERVAL_SECS: u64 = 5;

/// Maximum wall-clock duration for one rewrite pass (per-mode soft budget).
/// Beyond this the worker cancels the inflight LLM call and aborts.
const NREM_DEADLINE_SECS: u64 = 90;
const REM_DEADLINE_SECS: u64 = 180;

// =============================================================================
// Worker entry point
// =============================================================================

/// Top-level worker loop. Returns when `cancel_rx` flips to `true` or any
/// non-recoverable error occurs.
pub async fn run_worker(
    orch: Arc<SleepTimeOrchestrator>,
    mut cancel_rx: watch::Receiver<bool>,
    worker_idx: usize,
) {
    let worker_id = format!("sleep-worker-{worker_idx}");
    tracing::info!(worker = %worker_id, "sleep-time worker started");

    loop {
        // Cancellation check at the top of every loop iteration. `borrow()`
        // is a cheap single-RwLock-read.
        if *cancel_rx.borrow() {
            tracing::info!(worker = %worker_id, "sleep-time worker shutdown signalled");
            return;
        }

        // Disabled config: stay idle but still observe cancellation.
        if !orch.config().enabled {
            tokio::select! {
                _ = cancel_rx.changed() => continue,
                _ = tokio::time::sleep(StdDuration::from_secs(IDLE_POLL_INTERVAL_SECS)) => continue,
            }
        }

        // Try to claim work. None → idle poll.
        let claimed = match next_claim(orch.queue(), &worker_id, orch.config().claim_lease_secs) {
            Ok(Some(item)) => item,
            Ok(None) => {
                tokio::select! {
                    _ = cancel_rx.changed() => continue,
                    _ = tokio::time::sleep(StdDuration::from_secs(IDLE_POLL_INTERVAL_SECS)) => continue,
                }
            }
            Err(e) => {
                tracing::warn!(worker = %worker_id, error = %e, "claim error; backing off");
                tokio::time::sleep(StdDuration::from_secs(IDLE_POLL_INTERVAL_SECS)).await;
                continue;
            }
        };

        // Process under a per-mode deadline, but always honour cancellation.
        let deadline_secs = match claimed.mode {
            SleepMode::Nrem => NREM_DEADLINE_SECS,
            SleepMode::Rem => REM_DEADLINE_SECS,
        };

        let deadline = tokio::time::sleep(StdDuration::from_secs(deadline_secs));
        tokio::pin!(deadline);
        let processed = process_item(
            &orch,
            &worker_id,
            claimed.clone(),
            &mut PassLimits {
                cancel_rx: &mut cancel_rx,
                deadline,
            },
        )
        .await;
        let outcome = match processed {
            Ok(outcome) => outcome,
            Err(Interrupted::Cancelled) => {
                // Cancellation mid-process: release the claim so a future
                // restart can retry. We don't try to charge for in-flight
                // tokens — the LLM call (if any) was aborted by the runtime.
                if let Err(e) = orch.queue().release(&claimed) {
                    tracing::warn!(worker = %worker_id, error = %e, "failed to release claim on cancel");
                }
                tracing::info!(worker = %worker_id, "sleep-time worker cancelled mid-process");
                return;
            }
            Err(Interrupted::DeadlineExceeded) => {
                tracing::warn!(
                    worker = %worker_id,
                    user_id = %claimed.user_id,
                    mode = claimed.mode.as_str(),
                    deadline_secs,
                    "sleep-time pass exceeded deadline; aborting"
                );
                // Best-effort event emission. If the user memory can't be
                // resolved (deletion race), skip the event but still release.
                if let Ok(user_mem) = orch.user_memory_async(&claimed.user_id).await {
                    let (user_id, mode) = (claimed.user_id.clone(), claimed.mode);
                    on_blocking_pool(move || {
                        let guard = user_mem.read();
                        emit_aborted(
                            guard.as_memory_system(),
                            &user_id,
                            None,
                            mode,
                            "deadline exceeded",
                        );
                    })
                    .await;
                }
                let _ = orch.queue().release(&claimed);
                continue;
            }
        };

        match outcome {
            ProcessOutcome::Completed => {
                if let Err(e) = orch.queue().complete(&claimed) {
                    tracing::warn!(worker = %worker_id, error = %e, "complete error");
                }
            }
            ProcessOutcome::Retryable => {
                if let Err(e) = orch.queue().release(&claimed) {
                    tracing::warn!(worker = %worker_id, error = %e, "release error");
                }
            }
            ProcessOutcome::Drop => {
                // Logically "give up" — complete the item so we don't replay.
                if let Err(e) = orch.queue().complete(&claimed) {
                    tracing::warn!(worker = %worker_id, error = %e, "complete-after-drop error");
                }
            }
        }
    }
}

// =============================================================================
// Per-user fairness on claim
// =============================================================================

/// Pick the next item to process across all queued users, preferring the
/// user with the oldest unclaimed item ("oldest waiting served first"). On
/// success the item is claimed atomically with the configured lease.
fn next_claim(
    queue: &Queue,
    worker_id: &str,
    lease_secs: i64,
) -> anyhow::Result<Option<QueueItem>> {
    // Cheapest path: empty queue.
    let users = queue.distinct_users()?;
    if users.is_empty() {
        return Ok(None);
    }

    // Find the per-user oldest unclaimed and pick the most-stale across
    // all users. This is O(users); for V1 expected scale (<= 100s) that's
    // fine. If multi-tenant scaling shifts the curve, a heap-backed
    // priority queue is the upgrade path.
    let mut best: Option<QueueItem> = None;
    for u in &users {
        if let Some(it) = queue.next_unclaimed_for_user(u)? {
            best = Some(match best {
                None => it,
                Some(prev) if prev.enqueued_at <= it.enqueued_at => prev,
                Some(_) => it,
            });
        }
    }

    let Some(candidate) = best else {
        return Ok(None);
    };
    queue.claim(&candidate, worker_id, lease_secs)
}

// =============================================================================
// Per-item processing
// =============================================================================

#[derive(Debug, PartialEq, Eq)]
enum ProcessOutcome {
    /// Successfully processed (rewrite applied or empty result).
    Completed,
    /// Transient failure — release claim, queue item stays for retry.
    Retryable,
    /// Item rejected — give up (e.g. validation failure that won't change).
    Drop,
}

/// Why a pass ended before it had an outcome.
#[derive(Debug, PartialEq, Eq)]
enum Interrupted {
    /// The worker was told to shut down.
    Cancelled,
    /// The pass ran past its per-mode deadline.
    DeadlineExceeded,
}

/// What ends a pass early: shutdown and the per-mode deadline. They are checked only where the
/// pass waits on its tenant, the rewriter or its graph. A blocking phase, once started, runs to
/// the end and its outcome stands, as it did when the phases ran inline on the worker.
struct PassLimits<'a> {
    cancel_rx: &'a mut watch::Receiver<bool>,
    deadline: Pin<&'a mut tokio::time::Sleep>,
}

impl PassLimits<'_> {
    async fn until<T>(&mut self, work: impl Future<Output = T>) -> Result<T, Interrupted> {
        tokio::select! {
            _ = self.cancel_rx.changed() => Err(Interrupted::Cancelled),
            _ = self.deadline.as_mut() => Err(Interrupted::DeadlineExceeded),
            out = work => Ok(out),
        }
    }
}

/// Runs `work` on the blocking pool. It reads or writes the tenant's store under the tenant's
/// lock, which waits behind any writer, and a tokio worker held that long is one fewer for every
/// other request. A panic in `work` resumes here, as it would have inline. `None` when the
/// runtime is shutting down and never started it.
async fn on_blocking_pool<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    match tokio::task::spawn_blocking(work).await {
        Ok(out) => Some(out),
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        Err(_) => None,
    }
}

async fn process_item(
    orch: &Arc<SleepTimeOrchestrator>,
    worker_id: &str,
    item: QueueItem,
    limits: &mut PassLimits<'_>,
) -> Result<ProcessOutcome, Interrupted> {
    let user_id = &item.user_id;
    let mode = item.mode;

    // Resolve the user's memory (multi-tenant routing). We acquire the read
    // guard *only* inside the evidence-assembly and post-LLM application
    // phases, each a blocking-pool task — never across an `.await`, because
    // parking_lot's guard is `!Send`.
    let user_mem = match limits.until(orch.user_memory_async(user_id)).await? {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                worker = %worker_id,
                user_id = %user_id,
                error = %e,
                "could not resolve user memory; releasing claim"
            );
            return Ok(ProcessOutcome::Retryable);
        }
    };

    // ---- Phase A (blocking pool): assemble evidence + pre-flight budget -----
    let phase_a = {
        let (orch, user_mem) = (Arc::clone(orch), Arc::clone(&user_mem));
        let (worker_id, user_id, trigger) = (worker_id.to_string(), user_id.clone(), item.trigger);
        move || -> Result<Option<super::types::EvidencePack>, ProcessOutcome> {
            let (worker_id, user_id) = (worker_id.as_str(), user_id.as_str());
            let user_mem_guard = user_mem.read();
            let mem_sys = user_mem_guard.as_memory_system();

            let pack = match assemble_evidence_pack(
                mem_sys,
                orch.block_store(),
                orch.budget(),
                orch.graduation(),
                user_id,
                mode,
                trigger,
            ) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(
                        worker = %worker_id,
                        user_id = %user_id,
                        error = %e,
                        "evidence-pack assembly failed; releasing claim"
                    );
                    return Err(ProcessOutcome::Retryable);
                }
            };

            if pack.memories.is_empty() && pack.blocks.is_empty() {
                tracing::debug!(worker = %worker_id, user_id = %user_id, "evidence pack empty; skipping");
                return Ok(None);
            }

            let projected = orch.rewriter().project_tokens(&pack);
            if let Err(e) = orch.try_charge_call(user_id, projected) {
                match e {
                    SleepTimeError::BudgetExhausted { what, .. } => {
                        emit_budget_exhausted(mem_sys, user_id, &what);
                        return Err(ProcessOutcome::Drop);
                    }
                    other => {
                        tracing::warn!(
                            worker = %worker_id,
                            user_id = %user_id,
                            error = %other,
                            "unexpected budget error; releasing claim"
                        );
                        return Err(ProcessOutcome::Retryable);
                    }
                }
            }

            Ok(Some(pack))
        }
    };

    let pack = match on_blocking_pool(phase_a).await {
        Some(Ok(Some(pack))) => pack,
        Some(Ok(None)) => return Ok(ProcessOutcome::Completed),
        Some(Err(outcome)) => return Ok(outcome),
        None => {
            tracing::warn!(
                worker = %worker_id,
                user_id = %user_id,
                "evidence assembly did not run (runtime shutting down); releasing claim"
            );
            return Ok(ProcessOutcome::Retryable);
        }
    };

    // ---- Phase B (async): LLM call — NO user memory guard held --------------------
    let rewriter_result = limits.until(orch.rewriter().rewrite(&pack)).await?;

    // Resolved before Phase C takes its guard: a graph not yet open is opened on the
    // blocking pool, and the guard cannot be held across that await.
    let edge_graph = match &rewriter_result {
        Ok(out) if !out.edge_proposals.is_empty() => {
            Some(limits.until(orch.user_graph_async(user_id)).await?)
        }
        _ => None,
    };

    // ---- Phase C (blocking pool): settle spend, apply, persist, emit --------
    let phase_c = {
        let (orch, user_mem) = (Arc::clone(orch), Arc::clone(&user_mem));
        let (worker_id, user_id, trigger) = (worker_id.to_string(), user_id.clone(), item.trigger);
        move || -> ProcessOutcome {
            let (worker_id, user_id) = (worker_id.as_str(), user_id.as_str());
            let user_mem_guard = user_mem.read();
            let mem_sys = user_mem_guard.as_memory_system();

            let rewriter_output = match rewriter_result {
                Ok(out) => out,
                Err(e) => {
                    // Always settle actual spend even on error — if any tokens were
                    // billed they belong to the user's ledger.
                    let _ = orch.budget().record_actual_spend(user_id, 0);
                    tracing::warn!(
                        worker = %worker_id,
                        user_id = %user_id,
                        error = %e,
                        "rewriter call failed"
                    );
                    match e {
                        SleepTimeError::OutputValidation(reason)
                        | SleepTimeError::ParseError(reason) => {
                            emit_aborted(mem_sys, user_id, None, mode, &reason);
                            return ProcessOutcome::Drop;
                        }
                        SleepTimeError::BlockLocked { block_key } => {
                            emit_aborted(mem_sys, user_id, Some(&block_key), mode, "locked");
                            return ProcessOutcome::Drop;
                        }
                        _ => return ProcessOutcome::Retryable,
                    }
                }
            };

            if let Err(e) = orch
                .budget()
                .record_actual_spend(user_id, rewriter_output.total_tokens)
            {
                tracing::warn!(
                    worker = %worker_id,
                    user_id = %user_id,
                    error = %e,
                    "failed to record actual spend"
                );
            }

            for proposal in &rewriter_output.proposals {
                match apply_proposal(&orch, mem_sys, user_id, &trigger, proposal) {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::warn!(
                            worker = %worker_id,
                            user_id = %user_id,
                            block_key = %proposal.block_key,
                            error = %e,
                            "failed to apply rewrite proposal"
                        );
                    }
                }
            }

            // R43: REM-mode edge proposals. Apply BEFORE observations so any
            // entity references the observations contain are guaranteed to have
            // their graph edges live by the time downstream retrieval runs.
            // NREM mode produces no edge proposals (filtered at rewriter parse
            // time), so this is a no-op for NREM.
            if let Some(edge_graph) = edge_graph {
                match edge_graph {
                    Ok(graph) => match super::edge_proposals::apply_edge_proposals(
                        &graph,
                        &rewriter_output.edge_proposals,
                    ) {
                        Ok(result) => {
                            if result.applied > 0 {
                                tracing::info!(
                                    worker = %worker_id,
                                    user_id = %user_id,
                                    applied = result.applied,
                                    dropped_unresolved = result.dropped_unresolved_entity,
                                    dropped_self_loop = result.dropped_self_loop,
                                    errors = result.errors,
                                    "REM edge proposals applied to graph"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                worker = %worker_id,
                                user_id = %user_id,
                                error = %e,
                                "edge-proposal apply failed (continuing)"
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            worker = %worker_id,
                            user_id = %user_id,
                            error = %e,
                            "could not resolve user graph for edge proposals (continuing)"
                        );
                    }
                }
            }

            for draft in rewriter_output.observations {
                let supersedes = draft.supersedes.clone();
                match persist_observation(&draft, mem_sys) {
                    Ok(super::observation::PersistOutcome::Stored(memory_id))
                    | Ok(super::observation::PersistOutcome::Deduped(memory_id)) => {
                        emit_observation_emitted(
                            mem_sys,
                            user_id,
                            &memory_id.0.to_string(),
                            &draft,
                            rewriter_output.total_tokens,
                        );
                        // R42 + R59: if the draft carried an explicit `supersedes`
                        // link, write the bidirectional supersession edge so
                        // retrieval can find either end in O(1). Best-effort: a
                        // store failure is logged but does not fail the rewrite —
                        // the observation itself is already persisted, and the V3
                        // maintenance pass will repair missing edges by scanning
                        // observation metadata.
                        if let Some(older) = supersedes {
                            if let Err(e) = orch.supersession().record(
                                &memory_id,
                                &older,
                                super::supersession::DEFAULT_SUPERSESSION_CONFIDENCE,
                            ) {
                                tracing::warn!(
                                    worker = %worker_id,
                                    user_id = %user_id,
                                    superseder = %memory_id.0,
                                    superseded = %older.0,
                                    error = %e,
                                    "failed to record supersession edge (observation already persisted)"
                                );
                            } else {
                                tracing::debug!(
                                    worker = %worker_id,
                                    user_id = %user_id,
                                    superseder = %memory_id.0,
                                    superseded = %older.0,
                                    "supersession edge recorded"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            worker = %worker_id,
                            user_id = %user_id,
                            error = %e,
                            "failed to persist observation"
                        );
                    }
                }
            }

            ProcessOutcome::Completed
        }
    };

    match on_blocking_pool(phase_c).await {
        Some(outcome) => Ok(outcome),
        None => {
            tracing::warn!(
                worker = %worker_id,
                user_id = %user_id,
                "rewrite application did not run (runtime shutting down); releasing claim"
            );
            Ok(ProcessOutcome::Retryable)
        }
    }
}

// =============================================================================
// Block proposal application (R1 OCC + diff gating)
// =============================================================================

fn apply_proposal(
    orch: &Arc<SleepTimeOrchestrator>,
    mem_sys: &crate::memory::MemorySystem,
    user_id: &str,
    trigger: &SleepTimeTrigger,
    proposal: &RewriteProposal,
) -> anyhow::Result<()> {
    let Some(prev) = orch.block_store().get(user_id, &proposal.block_key)? else {
        emit_aborted(
            mem_sys,
            user_id,
            Some(&proposal.block_key),
            proposal.mode,
            "live block missing at apply-time",
        );
        return Ok(());
    };

    let summary = diff::classify(&prev.content, &proposal.new_content);
    if summary.class == DiffClass::Massive {
        emit_aborted(
            mem_sys,
            user_id,
            Some(&proposal.block_key),
            proposal.mode,
            "massive shrink rejected by R29",
        );
        return Ok(());
    }

    let outcome = orch.block_store().set_with_version_check(
        user_id,
        &proposal.block_key,
        &proposal.new_content,
        proposal.expected_version,
    )?;

    match outcome {
        OccOutcome::Applied(new_block) => {
            emit_block_rewritten(
                mem_sys,
                user_id,
                proposal,
                trigger,
                proposal.expected_version,
                new_block.version,
            );
        }
        OccOutcome::VersionConflict { current } => {
            emit_aborted(
                mem_sys,
                user_id,
                Some(&proposal.block_key),
                proposal.mode,
                &format!(
                    "OCC conflict: expected v{}, live v{}",
                    proposal.expected_version, current.version
                ),
            );
        }
        OccOutcome::Locked { .. } => {
            emit_aborted(
                mem_sys,
                user_id,
                Some(&proposal.block_key),
                proposal.mode,
                "block locked",
            );
        }
        OccOutcome::Missing => {
            emit_aborted(
                mem_sys,
                user_id,
                Some(&proposal.block_key),
                proposal.mode,
                "block missing at OCC apply",
            );
        }
    }

    Ok(())
}

// =============================================================================
// Event emission helpers
// =============================================================================

fn emit_block_rewritten(
    mem_sys: &crate::memory::MemorySystem,
    user_id: &str,
    proposal: &RewriteProposal,
    trigger: &SleepTimeTrigger,
    old_version: u32,
    new_version: u32,
) {
    mem_sys.record_consolidation_event_for_user(
        user_id,
        ConsolidationEvent::SleepTimeBlockRewritten {
            user_id: user_id.to_string(),
            block_key: proposal.block_key.clone(),
            old_version,
            new_version,
            mode: proposal.mode.as_str().to_string(),
            trigger: trigger_str(trigger).to_string(),
            token_spend: proposal.token_spend,
            model: proposal.model.clone(),
            timestamp: Utc::now(),
        },
    );
}

fn emit_observation_emitted(
    mem_sys: &crate::memory::MemorySystem,
    user_id: &str,
    memory_id: &str,
    draft: &ObservationDraft,
    token_spend: u32,
) {
    mem_sys.record_consolidation_event_for_user(
        user_id,
        ConsolidationEvent::SleepTimeObservationEmitted {
            user_id: user_id.to_string(),
            memory_id: memory_id.to_string(),
            mode: draft.mode.as_str().to_string(),
            trigger: "unspecified".to_string(),
            token_spend,
            confidence: draft.confidence,
            timestamp: Utc::now(),
        },
    );
}

fn emit_aborted(
    mem_sys: &crate::memory::MemorySystem,
    user_id: &str,
    block_key: Option<&str>,
    mode: SleepMode,
    reason: &str,
) {
    mem_sys.record_consolidation_event_for_user(
        user_id,
        ConsolidationEvent::SleepTimeRewriteAborted {
            user_id: user_id.to_string(),
            block_key: block_key.map(|s| s.to_string()),
            mode: mode.as_str().to_string(),
            reason: reason.to_string(),
            timestamp: Utc::now(),
        },
    );
}

fn emit_budget_exhausted(mem_sys: &crate::memory::MemorySystem, user_id: &str, what: &str) {
    mem_sys.record_consolidation_event_for_user(
        user_id,
        ConsolidationEvent::SleepTimeBudgetExhausted {
            user_id: user_id.to_string(),
            what: what.to_string(),
            timestamp: Utc::now(),
        },
    );
}

fn trigger_str(trigger: &SleepTimeTrigger) -> &'static str {
    match trigger {
        SleepTimeTrigger::Idle => "idle",
        SleepTimeTrigger::SessionClose => "session_close",
        SleepTimeTrigger::MaintenanceHeavyCycle => "maintenance_heavy_cycle",
        SleepTimeTrigger::Manual => "manual",
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SleepTimeProfile;
    use crate::graph_memory::GraphMemory;
    use crate::handlers::test_helpers::{hold_init_lock, hold_write_lock, TestHarness};
    use crate::handlers::MultiUserMemoryManager;
    use crate::memory::sleep_time::rewriter::{MockRewriter, Rewriter};
    use crate::memory::sleep_time::types::{EdgeProposalDraft, RewriterOutput};
    use std::time::Duration;

    const WORKER: &str = "sleep-worker-test";
    const NO_DEADLINE: Duration = Duration::from_secs(3600);
    /// Tokens the scripted rewrite bills; Phase C settles them before it applies anything.
    const SETTLED_TOKENS: u32 = 7;

    #[test]
    fn process_outcome_eq() {
        assert_eq!(ProcessOutcome::Completed, ProcessOutcome::Completed);
        assert_ne!(ProcessOutcome::Completed, ProcessOutcome::Retryable);
    }

    fn orchestrator_with(
        manager: &Arc<MultiUserMemoryManager>,
        rewriter: Rewriter,
    ) -> Arc<SleepTimeOrchestrator> {
        Arc::new(
            SleepTimeOrchestrator::new(
                SleepTimeProfile::Balanced.to_config(),
                manager.shared_db.column_store(),
                Arc::downgrade(manager),
                Arc::clone(&manager.context_block_store),
                rewriter,
            )
            .expect("orchestrator over the manager's shared database"),
        )
    }

    fn orchestrator_over(manager: &Arc<MultiUserMemoryManager>) -> Arc<SleepTimeOrchestrator> {
        orchestrator_with(manager, Rewriter::Mock(MockRewriter::new("mock")))
    }

    fn claim(orch: &SleepTimeOrchestrator, tenant: &str, mode: SleepMode) -> QueueItem {
        assert!(orch
            .enqueue(tenant, mode, SleepTimeTrigger::Manual)
            .expect("enqueue"));
        next_claim(orch.queue(), WORKER, 60)
            .expect("claim")
            .expect("the queued item")
    }

    /// A pass under the limits the worker loop sets: shutdown read from `shutdown_rx`, and
    /// `deadline` from the start.
    async fn run_pass(
        orch: Arc<SleepTimeOrchestrator>,
        item: QueueItem,
        mut shutdown_rx: watch::Receiver<bool>,
        deadline: Duration,
    ) -> Result<ProcessOutcome, Interrupted> {
        let deadline = tokio::time::sleep(deadline);
        tokio::pin!(deadline);
        let mut limits = PassLimits {
            cancel_rx: &mut shutdown_rx,
            deadline,
        };
        process_item(&orch, WORKER, item, &mut limits).await
    }

    /// An open tenant with one context block, so its pass reaches the rewriter, and a rewriter
    /// that proposes one graph edge, so the pass's Phase C takes the tenant graph's write lock
    /// once it has settled the rewrite's spend.
    fn a_pass_that_applies_an_edge(
        harness: &TestHarness,
        tenant: &str,
    ) -> (
        Arc<SleepTimeOrchestrator>,
        QueueItem,
        Arc<parking_lot::RwLock<GraphMemory>>,
    ) {
        harness.manager.get_user_memory(tenant).expect("open");
        harness
            .manager
            .context_block_store
            .set(tenant, "persona", "prefers short answers", None)
            .expect("context block");
        let rewriter = MockRewriter::new("mock");
        rewriter.push(RewriterOutput {
            proposals: Vec::new(),
            observations: Vec::new(),
            edge_proposals: vec![EdgeProposalDraft {
                from_entity: "alpha".to_string(),
                to_entity: "beta".to_string(),
                relation: "related_to".to_string(),
                confidence: 0.9,
                rationale: String::new(),
            }],
            total_tokens: SETTLED_TOKENS,
            model: "mock".to_string(),
            elapsed_ms: 0,
        });
        let orch = orchestrator_with(&harness.manager, Rewriter::Mock(rewriter));
        let item = claim(&orch, tenant, SleepMode::Rem);
        let graph = harness.manager.get_user_graph(tenant).expect("graph");
        (orch, item, graph)
    }

    /// Yields to the runtime until the pass has settled its spend, the first thing Phase C does.
    async fn wait_for_settled_spend(orch: &SleepTimeOrchestrator, tenant: &str) {
        let give_up = tokio::time::Instant::now() + Duration::from_secs(5);
        while orch
            .user_status(tenant)
            .expect("budget")
            .budget
            .tokens_this_hour
            < SETTLED_TOKENS
        {
            assert!(
                tokio::time::Instant::now() < give_up,
                "the pass never reached Phase C"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// The test runtime has one thread, so a pass that opened the tenant on it would stall the
    /// timeout below until the init lock is let go.
    #[tokio::test(flavor = "current_thread")]
    async fn a_pass_for_a_tenant_still_opening_leaves_the_runtime_thread_free() {
        const TENANT: &str = "sleep-tenant-mid-open";
        let harness = TestHarness::new();
        let orch = orchestrator_over(&harness.manager);
        let item = claim(&orch, TENANT, SleepMode::Nrem);

        let (release, opener) = hold_init_lock(harness.manager.user_memory_init_lock(TENANT));
        let (_shutdown, shutdown_rx) = watch::channel(false);
        let mut pass = tokio::spawn(run_pass(Arc::clone(&orch), item, shutdown_rx, NO_DEADLINE));
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut pass).await;
        assert!(
            waited.is_err(),
            "the pass finished while the tenant's init lock was held, so it held the runtime \
             thread until the lock was let go"
        );

        release.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        assert_eq!(
            pass.await.expect("pass task"),
            Ok(ProcessOutcome::Completed)
        );
        assert!(
            harness.manager.user_memories.get(TENANT).is_some(),
            "the pass opened the tenant"
        );
    }

    /// Phase A reads the tenant under its read guard, which waits while a writer holds the
    /// tenant. Made on the one runtime thread, that wait would stall the timeout below.
    #[tokio::test(flavor = "current_thread")]
    async fn evidence_assembly_behind_a_tenant_writer_leaves_the_runtime_thread_free() {
        const TENANT: &str = "sleep-evidence-behind-writer";
        let harness = TestHarness::new();
        let orch = orchestrator_over(&harness.manager);
        let tenant = harness.manager.get_user_memory(TENANT).expect("open");
        let item = claim(&orch, TENANT, SleepMode::Nrem);

        let (release, writer) = hold_write_lock(tenant);
        let (_shutdown, shutdown_rx) = watch::channel(false);
        let mut pass = tokio::spawn(run_pass(Arc::clone(&orch), item, shutdown_rx, NO_DEADLINE));
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut pass).await;
        assert!(
            waited.is_err(),
            "the pass finished while a writer held its tenant, so evidence assembly held the \
             runtime thread until the writer let go"
        );

        release.send(()).expect("writer waits for release");
        writer.join().expect("writer thread");
        assert_eq!(
            pass.await.expect("pass task"),
            Ok(ProcessOutcome::Completed)
        );
    }

    /// Phase C applies the edge proposal under the graph's write lock, which waits while another
    /// writer holds the graph. Made on the one runtime thread, that wait would stall the timeout.
    #[tokio::test(flavor = "current_thread")]
    async fn applying_a_rewrite_behind_a_graph_writer_leaves_the_runtime_thread_free() {
        const TENANT: &str = "sleep-apply-behind-writer";
        let harness = TestHarness::new();
        let (orch, item, graph) = a_pass_that_applies_an_edge(&harness, TENANT);

        let (release, writer) = hold_write_lock(graph);
        let (_shutdown, shutdown_rx) = watch::channel(false);
        let mut pass = tokio::spawn(run_pass(Arc::clone(&orch), item, shutdown_rx, NO_DEADLINE));
        wait_for_settled_spend(&orch, TENANT).await;
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut pass).await;
        assert!(
            waited.is_err(),
            "the pass finished while a writer held its graph, so applying the rewrite held the \
             runtime thread until the writer let go"
        );

        release.send(()).expect("writer waits for release");
        writer.join().expect("writer thread");
        assert_eq!(
            pass.await.expect("pass task"),
            Ok(ProcessOutcome::Completed)
        );
    }

    /// Inline, Phase C could not be cut short: shutdown and the deadline were only seen at the
    /// pass's awaits. On the blocking pool it must still run to its outcome.
    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_during_phase_c_leaves_its_outcome_standing() {
        const TENANT: &str = "sleep-shutdown-mid-apply";
        let harness = TestHarness::new();
        let (orch, item, graph) = a_pass_that_applies_an_edge(&harness, TENANT);

        let (release, writer) = hold_write_lock(graph);
        let (shutdown, shutdown_rx) = watch::channel(false);
        let mut pass = tokio::spawn(run_pass(Arc::clone(&orch), item, shutdown_rx, NO_DEADLINE));
        wait_for_settled_spend(&orch, TENANT).await;
        shutdown.send(true).expect("the pass watches for shutdown");
        let waited = tokio::time::timeout(Duration::from_millis(100), &mut pass).await;
        assert!(waited.is_err(), "the pass ended while its graph was held");

        release.send(()).expect("writer waits for release");
        writer.join().expect("writer thread");
        assert_eq!(
            pass.await.expect("pass task"),
            Ok(ProcessOutcome::Completed),
            "shutdown cut short a pass already applying its rewrite"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn the_deadline_still_ends_a_pass_waiting_on_its_tenant() {
        const TENANT: &str = "sleep-deadline-mid-open";
        let harness = TestHarness::new();
        let orch = orchestrator_over(&harness.manager);
        let item = claim(&orch, TENANT, SleepMode::Nrem);

        let (release, opener) = hold_init_lock(harness.manager.user_memory_init_lock(TENANT));
        let (_shutdown, shutdown_rx) = watch::channel(false);
        let ended = run_pass(
            Arc::clone(&orch),
            item,
            shutdown_rx,
            Duration::from_millis(100),
        )
        .await;
        release.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        assert_eq!(ended, Err(Interrupted::DeadlineExceeded));
        harness
            .manager
            .get_user_memory_async(TENANT)
            .await
            .expect("the open the pass started finishes");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn the_edge_proposal_graph_opens_off_the_runtime_thread() {
        const TENANT: &str = "sleep-graph-mid-open";
        let harness = TestHarness::new();
        let orch = orchestrator_over(&harness.manager);

        let (release, opener) = hold_init_lock(harness.manager.user_graph_init_lock(TENANT));
        let lookup_orch = Arc::clone(&orch);
        let mut lookup = tokio::spawn(async move { lookup_orch.user_graph_async(TENANT).await });
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut lookup).await;
        assert!(
            waited.is_err(),
            "the lookup finished while the graph's init lock was held, so it held the runtime \
             thread until the lock was let go"
        );

        release.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        let opened = lookup.await.expect("lookup task").expect("open");
        let cached = harness
            .manager
            .graph_memories
            .get(TENANT)
            .expect("the open cached the graph");
        assert!(Arc::ptr_eq(&opened, &cached));
    }
}
