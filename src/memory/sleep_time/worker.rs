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

        let process_fut = process_item(&orch, &worker_id, claimed.clone());
        let outcome = tokio::select! {
            _ = cancel_rx.changed() => {
                // Cancellation mid-process: release the claim so a future
                // restart can retry. We don't try to charge for in-flight
                // tokens — the LLM call (if any) was aborted by the runtime.
                if let Err(e) = orch.queue().release(&claimed) {
                    tracing::warn!(worker = %worker_id, error = %e, "failed to release claim on cancel");
                }
                tracing::info!(worker = %worker_id, "sleep-time worker cancelled mid-process");
                return;
            }
            _ = tokio::time::sleep(StdDuration::from_secs(deadline_secs)) => {
                tracing::warn!(
                    worker = %worker_id,
                    user_id = %claimed.user_id,
                    mode = claimed.mode.as_str(),
                    deadline_secs,
                    "sleep-time pass exceeded deadline; aborting"
                );
                // Best-effort event emission. If the user's earth can't be
                // resolved (deletion race), skip the event but still release.
                if let Ok(earth) = orch.user_earth(&claimed.user_id) {
                    let guard = earth.read();
                    emit_aborted(
                        guard.as_memory_system(),
                        &claimed.user_id,
                        None,
                        claimed.mode,
                        "deadline exceeded",
                    );
                }
                let _ = orch.queue().release(&claimed);
                continue;
            }
            outcome = process_fut => outcome,
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

    let Some(candidate) = best else { return Ok(None) };
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

async fn process_item(
    orch: &Arc<SleepTimeOrchestrator>,
    worker_id: &str,
    item: QueueItem,
) -> ProcessOutcome {
    let user_id = &item.user_id;
    let mode = item.mode;

    // Resolve the user's Earth (multi-tenant routing). We acquire the read
    // guard *only* during the synchronous evidence-assembly and post-LLM
    // application phases — never across an `.await`, because parking_lot's
    // guard is `!Send`.
    let earth = match orch.user_earth(user_id) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                worker = %worker_id,
                user_id = %user_id,
                error = %e,
                "could not resolve user earth; releasing claim"
            );
            return ProcessOutcome::Retryable;
        }
    };

    // ---- Phase A (sync): assemble evidence + pre-flight budget --------------
    let phase_a = (|| -> Result<Option<super::types::EvidencePack>, ProcessOutcome> {
        let earth_guard = earth.read();
        let mem_sys = earth_guard.as_memory_system();

        let pack = match assemble_evidence_pack(
            mem_sys,
            orch.block_store(),
            orch.budget(),
            orch.graduation(),
            user_id,
            mode,
            item.trigger,
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
    })();

    let pack = match phase_a {
        Ok(Some(pack)) => pack,
        Ok(None) => return ProcessOutcome::Completed,
        Err(outcome) => return outcome,
    };

    // ---- Phase B (async): LLM call — NO earth guard held --------------------
    let rewriter_result = orch.rewriter().rewrite(&pack).await;

    // ---- Phase C (sync): settle spend, apply, persist, emit -----------------
    let earth_guard = earth.read();
    let mem_sys = earth_guard.as_memory_system();

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
        match apply_proposal(orch, mem_sys, user_id, &item.trigger, proposal) {
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
    if !rewriter_output.edge_proposals.is_empty() {
        match orch.user_graph(user_id) {
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

    #[test]
    fn process_outcome_eq() {
        assert_eq!(ProcessOutcome::Completed, ProcessOutcome::Completed);
        assert_ne!(ProcessOutcome::Completed, ProcessOutcome::Retryable);
    }
}
