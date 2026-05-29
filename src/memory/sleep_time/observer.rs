//! Evidence-pack assembly for sleep-time passes.
//!
//! Given a user, mode, and trigger, the observer produces an [`EvidencePack`]
//! the rewriter can consume. Pulls memories from all three tiers via
//! [`MemorySystem::get_all_memories`], filters by [`MemoryOrigin::visible_to_evidence`]
//! (R1 + R46), and caps the result by count and token budget (R2).
//!
//! The observer is a free function module — it does not own any state. The
//! orchestrator passes the `&MemorySystem` and the relevant stores.

use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;

use super::graduation::{effective_origin, GraduationStore};
use super::policies::BudgetTracker;
use super::types::{
    BlockSnapshot, EvidenceFact, EvidenceMemory, EvidencePack, SleepMode, SleepTimeTrigger,
};
#[cfg(test)]
use super::types::MemoryOrigin;
use crate::memory::context_blocks::ContextBlockStore;
use crate::memory::types::{MemoryTier, SharedMemory};
use crate::memory::MemorySystem;

// =============================================================================
// Configuration knobs (sensible V1 defaults; folded into SleepTimeConfig later)
// =============================================================================

/// Maximum memories carried into one evidence pack (R2 — bounded resource).
pub const MAX_EVIDENCE_MEMORIES: usize = 60;

/// Approximate token budget for the evidence pack body. Used by the
/// rewriter's `project_tokens` to reserve LLM capacity before the call (R12).
pub const MAX_EVIDENCE_TOKENS: u32 = 12_000;

/// Approximate chars-per-token ratio. Real tokenisers (BPE) vary; this is
/// the conservative average for English natural language. Used only for
/// truncation gating; final billing uses the LLM provider's `usage` block.
const APPROX_CHARS_PER_TOKEN: u32 = 4;

/// Per-mode memory pool selection — which tiers each mode draws from.
#[derive(Debug, Clone, Copy)]
struct ModePool {
    working: bool,
    session: bool,
    long_term: bool,
}

impl ModePool {
    fn for_mode(mode: SleepMode) -> Self {
        match mode {
            // NREM: recent episodic — working + session only.
            SleepMode::Nrem => Self {
                working: true,
                session: true,
                long_term: false,
            },
            // REM: long-term semantic + recent grounding context.
            SleepMode::Rem => Self {
                working: false,
                session: true,
                long_term: true,
            },
        }
    }
}

// =============================================================================
// Public entry point
// =============================================================================

/// Assemble an evidence pack for one sleep-time pass.
///
/// `user_id` is used both as the per-tenant identifier and as the
/// multi-tenancy `actor_id` filter on memories (memories with a different
/// `actor_id` are excluded).
pub fn assemble_evidence_pack(
    mem_sys: &MemorySystem,
    block_store: &Arc<ContextBlockStore>,
    budget_tracker: &Arc<BudgetTracker>,
    graduation: &Arc<GraduationStore>,
    user_id: &str,
    mode: SleepMode,
    trigger: SleepTimeTrigger,
) -> Result<EvidencePack> {
    let now = Utc::now();
    let pool = ModePool::for_mode(mode);

    // 1. Pull every visible memory across the relevant tiers for this mode.
    let all_memories = mem_sys.get_all_memories()?;
    let mut candidates: Vec<EvidenceMemory> = all_memories
        .into_iter()
        .filter(|m| tier_matches(m, pool))
        .filter(|m| user_matches(m, user_id))
        .filter_map(|m| eligible_evidence_memory(&m, mode, graduation))
        .collect();

    // 2. Sort: most recent first; ties broken by importance (descending).
    candidates.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then(b.importance.total_cmp(&a.importance))
    });

    // 3. Cap by count and approximate token budget.
    candidates.truncate(MAX_EVIDENCE_MEMORIES);
    let memories = truncate_to_token_budget(candidates, MAX_EVIDENCE_TOKENS);

    // 4. Pull the user's context blocks; populate lock state from the budget
    //    tracker (R14).
    let blocks_raw = block_store.list(user_id).unwrap_or_default();
    let blocks: Vec<BlockSnapshot> = blocks_raw
        .into_iter()
        .map(|b| {
            let locked = budget_tracker
                .is_block_locked(user_id, &b.key)
                .unwrap_or(false);
            BlockSnapshot {
                key: b.key,
                content: b.content,
                version: b.version,
                max_tokens: b.max_tokens,
                locked,
            }
        })
        .collect();

    // 5. Facts: V1 defers — empty for now (R7/V2 wires in top-K reinforced
    //    facts from `SemanticFactStore`). The pack shape is stable so V2
    //    can populate without breaking the rewriter contract.
    let facts: Vec<EvidenceFact> = Vec::new();

    // 6. Prohibitions: V1 defers (R9/R10/R23 land in V2/V3). Empty here.
    let block_prohibitions: HashMap<String, Vec<String>> = HashMap::new();

    let approx_tokens = estimate_tokens(&memories, &blocks);

    Ok(EvidencePack {
        user_id: user_id.to_string(),
        mode,
        trigger,
        memories,
        blocks,
        facts,
        block_prohibitions,
        approx_tokens,
        assembled_at: now,
    })
}

// =============================================================================
// Filters
// =============================================================================

fn tier_matches(mem: &SharedMemory, pool: ModePool) -> bool {
    match mem.tier {
        MemoryTier::Working => pool.working,
        MemoryTier::Session => pool.session,
        MemoryTier::LongTerm | MemoryTier::Archive => pool.long_term,
    }
}

fn user_matches(mem: &SharedMemory, user_id: &str) -> bool {
    match mem.actor_id.as_deref() {
        Some(a) => a == user_id,
        None => false, // memory with no actor is global; skip per-user packs
    }
}

/// Apply origin-visibility rules (R1 + R46) and project into `EvidenceMemory`.
/// Consults the graduation store to upgrade `BackgroundSleepTimeObservation`
/// → `BackgroundSleepTimeGraduated` when the registry has recorded a
/// graduation for this memory (R27).
fn eligible_evidence_memory(
    mem: &SharedMemory,
    mode: SleepMode,
    graduation: &Arc<GraduationStore>,
) -> Option<EvidenceMemory> {
    let origin = effective_origin(&mem.id, &mem.experience, graduation);
    if !origin.visible_to_evidence(mode) {
        return None;
    }
    // Pull entity refs from the metadata (sleep-time observations) and from
    // mem.entity_refs (foreground memories). De-dup string-side.
    let mut entities: Vec<String> = mem
        .entity_refs
        .iter()
        .map(|e| e.name.clone())
        .chain(mem.experience.entities.iter().cloned())
        .collect();
    entities.sort();
    entities.dedup();

    Some(EvidenceMemory {
        id: mem.id.clone(),
        content: mem.experience.content.clone(),
        created_at: mem.created_at,
        importance: mem.importance(),
        origin,
        entity_refs: entities,
    })
}

// =============================================================================
// Token budgeting
// =============================================================================

fn approx_tokens_of(s: &str) -> u32 {
    let chars = s.chars().count() as u32;
    chars.div_ceil(APPROX_CHARS_PER_TOKEN)
}

fn truncate_to_token_budget(
    memories: Vec<EvidenceMemory>,
    max_tokens: u32,
) -> Vec<EvidenceMemory> {
    let mut total = 0u32;
    let mut out = Vec::with_capacity(memories.len());
    for m in memories {
        let cost = approx_tokens_of(&m.content);
        if total.saturating_add(cost) > max_tokens {
            break;
        }
        total = total.saturating_add(cost);
        out.push(m);
    }
    out
}

fn estimate_tokens(memories: &[EvidenceMemory], blocks: &[BlockSnapshot]) -> u32 {
    let m: u32 = memories
        .iter()
        .map(|m| approx_tokens_of(&m.content))
        .sum::<u32>();
    let b: u32 = blocks
        .iter()
        .map(|b| approx_tokens_of(&b.content))
        .sum::<u32>();
    m.saturating_add(b)
}

// =============================================================================
// Tests (pure-logic only — full assembly requires a wired MemorySystem)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryId;
    use uuid::Uuid;

    fn make_evidence_memory(content: &str, origin: MemoryOrigin) -> EvidenceMemory {
        EvidenceMemory {
            id: MemoryId(Uuid::new_v4()),
            content: content.to_string(),
            created_at: Utc::now(),
            importance: 0.5,
            origin,
            entity_refs: vec![],
        }
    }

    #[test]
    fn approx_tokens_of_rounds_up() {
        assert_eq!(approx_tokens_of(""), 0);
        assert_eq!(approx_tokens_of("a"), 1);
        // 5 chars / 4 chars-per-token = 2 tokens (rounded up)
        assert_eq!(approx_tokens_of("hello"), 2);
        // 8 chars / 4 = 2 tokens
        assert_eq!(approx_tokens_of("abcdefgh"), 2);
    }

    #[test]
    fn truncate_to_token_budget_drops_overflow() {
        let memories = vec![
            make_evidence_memory(&"a".repeat(40), MemoryOrigin::ForegroundUser), // ~10 tokens
            make_evidence_memory(&"b".repeat(40), MemoryOrigin::ForegroundUser), // ~10 tokens
            make_evidence_memory(&"c".repeat(40), MemoryOrigin::ForegroundUser), // ~10 tokens
        ];
        let out = truncate_to_token_budget(memories, 25);
        // Two fit (10 + 10 = 20 ≤ 25); third would push to 30, so dropped.
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn mode_pool_nrem_excludes_long_term() {
        let pool = ModePool::for_mode(SleepMode::Nrem);
        assert!(pool.working);
        assert!(pool.session);
        assert!(!pool.long_term);
    }

    #[test]
    fn mode_pool_rem_excludes_working() {
        let pool = ModePool::for_mode(SleepMode::Rem);
        assert!(!pool.working);
        assert!(pool.session);
        assert!(pool.long_term);
    }
}
