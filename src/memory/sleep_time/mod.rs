//! Sleep-time / observational memory subsystem.
//!
//! Async LLM-driven background worker that re-authors [`crate::memory::ContextBlock`]s
//! between sessions and emits observation memories. Augments — does not replace —
//! the synchronous `maintenance.rs` consolidation loop and the deterministic
//! `compression.rs::SemanticConsolidator` fact extractor.
//!
//! ## Design references
//!
//! The complete rationale, evaluation chain (4 cycles of bifocal + overloop ×2 +
//! breakers), per-remediation WHY framing, and 4-wave shipping plan live in:
//!
//! - `C:\Users\jhancock\.claude\plans\sleep-time-final-synthesis.md`
//! - `C:\Users\jhancock\.claude\plans\sleep-time-rationale.md`
//!
//! ## Module map (V1)
//!
//! Currently landed:
//!   - [`types`] — shared types: `EvidencePack`, `RewriteProposal`,
//!     `ObservationDraft`, `MemoryOrigin`, `SleepMode`, `QueueItem`, etc.
//!   - [`diff`] — `classify(prior, new)` returning `DiffSummary` and `DiffClass`
//!     (Noop / Minor / Substantive / Massive) for R29 validation.
//!   - [`policies`] — `BudgetTracker` (RocksDB-backed per-user + global), lock
//!     state, and in-memory `DebounceTracker`.
//!   - [`rewriter`] — `Rewriter` concrete enum with `Anthropic` variant
//!     (production) and `Mock` variant (cfg(test)). Strict prompt role
//!     separation (R30), structured JSON output schema, response validation (R29).
//!
//! Deferred to the V1-completion PR:
//!   - `observer` — evidence-pack assembly from working/session/long-term tiers,
//!     facts, blocks, and feedback. Depends on `MemorySystem` internals.
//!   - `queue` — persistent debounced work queue (CF `sleep_time_queue`) with
//!     claim-on-process, schema versioning (R32), cold-start TTL (R31).
//!   - `worker` — tokio task pool with `CancellationToken` (R3), per-user
//!     fairness, overlap guard (R34).
//!   - `observation` — `ObservationDraft` → indexed `Memory` pipeline.
//!   - `orchestrator` — `SleepTimeOrchestrator` public API; mounts on
//!     `MemorySystem`.
//!
//! See `sleep-time-final-synthesis.md` § "V1 — Minimum Viable Sleep-Time" for
//! the exact remediation list each remaining module must implement.

pub mod diff;
pub mod edge_proposals;
pub mod graduation;
pub mod observation;
pub mod observer;
pub mod orchestrator;
pub mod policies;
pub mod queue;
pub mod rewriter;
pub mod supersession;
pub mod types;
pub mod worker;

pub use diff::classify;
pub use policies::{
    BudgetTracker, DebounceTracker, PolicyLimits, CF_SLEEP_TIME_BUDGET,
    CF_SLEEP_TIME_GLOBAL_BUDGET,
};
pub use observation::{
    build_experience, is_sleep_time_authored, origin_of, persist_observation, PersistOutcome,
    DEFAULT_OBSERVATION_IMPORTANCE,
};
pub use graduation::{
    effective_origin, graduate_eligible_observations, GraduationPassResult, GraduationRecord,
    GraduationStore, CF_SLEEP_TIME_GRADUATIONS, DEFAULT_GRADUATION_ACCESS_THRESHOLD,
};
pub use edge_proposals::{apply_edge_proposals, EdgeApplicationResult};
pub use supersession::{
    SupersessionRecord, SupersessionStore, CF_SLEEP_TIME_SUPERSESSIONS,
    DEFAULT_SUPERSESSION_CONFIDENCE, DEFAULT_SUPERSESSION_DECAY, SUPERSESSION_EXPIRY_FLOOR,
};
pub use observer::{assemble_evidence_pack, MAX_EVIDENCE_MEMORIES, MAX_EVIDENCE_TOKENS};
pub use orchestrator::{OrchestratorStatus, SleepTimeOrchestrator, UserStatus};
pub use queue::{Queue, CF_SLEEP_TIME_QUEUE, DEFAULT_CLAIM_LEASE_SECS};
pub use rewriter::{AnthropicRewriter, Rewriter};
pub use types::{
    BlockSnapshot, BudgetState, CostSample, DiffClass, DiffSummary, EvidenceFact, EvidenceMemory,
    EvidencePack, GlobalBudgetState, MemoryOrigin, ObservationDraft, ObservationPriority,
    QueueItem, RewriteProposal, RewriterOutput, SleepMode, SleepTimeError, SleepTimeResult,
    SleepTimeTrigger, QUEUE_ITEM_SCHEMA_VERSION,
};
