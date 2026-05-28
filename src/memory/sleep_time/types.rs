//! Shared types for the sleep-time / observational memory subsystem.
//!
//! Sleep-time is an async LLM-driven worker that re-authors [`ContextBlock`]s
//! between sessions and emits observation memories + graph-edge proposals
//! based on accumulated session experience.
//!
//! References for the design (rationale lives in
//! `C:\Users\jhancock\.claude\plans\sleep-time-rationale.md`):
//!   - Letta sleep-time compute (arXiv 2504.13171) — async-rewrite pattern.
//!   - Mastra Observational Memory — three-date temporal anchoring.
//!   - Tononi & Cirelli synaptic homeostasis — why down-selection matters.
//!   - Klinzing/Niethard/Born 2019 — NREM episodic vs REM remote-semantic split.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::memory::types::MemoryId;

// =============================================================================
// Trigger / mode
// =============================================================================

/// Why a sleep-time pass was enqueued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepTimeTrigger {
    /// User has been idle (no foreground remember/recall) for the configured
    /// threshold.
    Idle,
    /// A session-end hook fired (e.g. Claude Code SessionEnd).
    SessionClose,
    /// A heavy maintenance cycle nudged sleep-time (does not execute inline).
    MaintenanceHeavyCycle,
    /// An operator or API caller requested it explicitly.
    Manual,
}

/// Two-phase sleep mode (R40).
///
/// NREM mode integrates recent episodic material; REM mode revisits long-term
/// semantic material and emits graph-edge proposals (V2). Modes consume
/// different evidence windows, different feedback inputs (R45), and use
/// mode-aware quality gates (R44).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepMode {
    /// Non-REM equivalent: recent-experience-bias evidence; episodic fidelity.
    Nrem,
    /// REM equivalent: long-term-semantic-bias evidence; bounded abstraction.
    Rem,
}

impl SleepMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nrem => "nrem",
            Self::Rem => "rem",
        }
    }
}

// =============================================================================
// Origin tagging (R18 + R46) — replaces dual-tag scheme MX5
// =============================================================================

/// Single authoritative origin label for a memory / observation.
///
/// Replaces the previously-proposed dual-tag scheme (foreground/background
/// bool + provenance enum); see overloop cross-remediation MX5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryOrigin {
    /// User-originated memory (`POST /remember` or equivalent foreground path).
    #[default]
    ForegroundUser,
    /// Agent-originated memory (the LLM agent itself stored it during a turn).
    ForegroundAgent,
    /// A sleep-time observation that has NOT yet graduated to the fact corpus.
    /// Excluded from sleep-time evidence assembly (prevents L1 confabulation).
    BackgroundSleepTimeObservation,
    /// A sleep-time observation that has graduated (R27) via foreground access.
    /// Visible to REM-mode evidence; still excluded from NREM evidence.
    BackgroundSleepTimeGraduated,
    /// Synchronous maintenance loop (compression, decay, replay-derived).
    BackgroundMaintenance,
    /// Explicit operator action (lock/unlock/forget/rerun).
    OperatorAction,
}

impl MemoryOrigin {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ForegroundUser => "foreground_user",
            Self::ForegroundAgent => "foreground_agent",
            Self::BackgroundSleepTimeObservation => "background_sleep_time_observation",
            Self::BackgroundSleepTimeGraduated => "background_sleep_time_graduated",
            Self::BackgroundMaintenance => "background_maintenance",
            Self::OperatorAction => "operator_action",
        }
    }

    /// True if this origin should be considered "foreground" for idle detection
    /// (R13 + ML13.1).
    pub const fn is_foreground(self) -> bool {
        matches!(self, Self::ForegroundUser | Self::ForegroundAgent)
    }

    /// Visibility rule for sleep-time evidence assembly given mode (R6 + R46).
    pub const fn visible_to_evidence(self, mode: SleepMode) -> bool {
        match (self, mode) {
            // Pre-graduation observations: NEVER in own evidence (L1 fix)
            (Self::BackgroundSleepTimeObservation, _) => false,
            // Graduated observations: visible to REM only
            (Self::BackgroundSleepTimeGraduated, SleepMode::Rem) => true,
            (Self::BackgroundSleepTimeGraduated, SleepMode::Nrem) => false,
            // Everything else is visible to both modes
            _ => true,
        }
    }
}

// =============================================================================
// Observation priority (Mastra-style structured signals, typed not emoji)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPriority {
    High,
    #[default]
    Medium,
    Low,
}

// =============================================================================
// Observation draft (R41 — three-date temporal anchoring)
// =============================================================================

/// A proposed observation, as emitted by the rewriter and before persistence
/// or quality-gate evaluation.
///
/// Carries Mastra's three-date temporal model (created / referenced / relative)
/// which drives temporal-reasoning retrieval accuracy. `referenced_at` is
/// **never** LLM-fabricated (R49 + R55) — it is either explicit in source text
/// or `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationDraft {
    /// The observation content (LLM-authored narrative).
    pub content: String,

    /// Entity references extracted at draft time (R54-validated downstream).
    #[serde(default)]
    pub entity_refs: Vec<String>,

    /// Wall-clock time the rewriter authored the observation.
    pub created_at: DateTime<Utc>,

    /// The date the *content* describes, if extractable from source memories
    /// (R41/R49/R55). `None` if no date is present in source text.
    #[serde(default)]
    pub referenced_at: Option<DateTime<Utc>>,

    /// Original natural-language relative phrase (e.g. "3 days ago") preserved
    /// for audit/explanation. Resolution to `referenced_at` is deterministic
    /// via the date resolver; this stays as-written.
    #[serde(default)]
    pub relative_at_anchor: Option<String>,

    /// IDs of memories used as evidence for this observation (lineage).
    #[serde(default)]
    pub source_memory_ids: Vec<MemoryId>,

    /// The rewrite event that produced this observation (R4 lineage backlink).
    /// Used by `forget_rewrite` cascade.
    #[serde(default)]
    pub source_rewrite_id: Option<Uuid>,

    /// Tagged origin (R18 + R46). Always set by the orchestrator before
    /// persistence; serde default exists only for forward-compat read paths.
    #[serde(default)]
    pub origin: MemoryOrigin,

    /// Which sleep mode produced this observation (R40).
    pub mode: SleepMode,

    /// Confidence assigned by the rewriter, gated downstream by R16/R44.
    #[serde(default = "default_confidence")]
    pub confidence: f32,

    /// Structured priority signal (replaces Mastra's emoji approach with a typed
    /// enum suitable for routing and retrieval ranking).
    #[serde(default)]
    pub priority: ObservationPriority,

    /// Embedder version stamped at draft time (R15).
    pub embedder_version: String,

    /// If this draft supersedes a prior observation (R42), the prior
    /// observation's memory ID. Bidirectional handling lives in V2 (R59).
    #[serde(default)]
    pub supersedes: Option<MemoryId>,
}

fn default_confidence() -> f32 {
    0.5
}

impl ObservationDraft {
    pub fn new(
        content: impl Into<String>,
        mode: SleepMode,
        embedder_version: impl Into<String>,
    ) -> Self {
        Self {
            content: content.into(),
            entity_refs: Vec::new(),
            created_at: Utc::now(),
            referenced_at: None,
            relative_at_anchor: None,
            source_memory_ids: Vec::new(),
            source_rewrite_id: None,
            origin: MemoryOrigin::BackgroundSleepTimeObservation,
            mode,
            confidence: default_confidence(),
            priority: ObservationPriority::Medium,
            embedder_version: embedder_version.into(),
            supersedes: None,
        }
    }
}

// =============================================================================
// Rewrite proposal
// =============================================================================

/// A proposed rewrite of a single `ContextBlock`. Produced by the rewriter,
/// gated by the quality panel (R16/R44), reconciled against
/// `ContextBlock.version` via optimistic concurrency control (R1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteProposal {
    /// A unique ID for this proposal, used as the lineage `source_rewrite_id`
    /// on derived observations.
    pub id: Uuid,
    /// The target block key (e.g. "persona", "user_profile", "project_state").
    pub block_key: String,
    /// The version of the existing block that the rewriter saw when it
    /// generated this proposal. If the live block has advanced past this
    /// version at apply time, the apply is aborted (R1 OCC).
    pub expected_version: u32,
    /// The proposed new block content.
    pub new_content: String,
    /// A short, structured rationale produced by the rewriter (audit aid).
    pub rationale: String,
    /// Source memory IDs used as evidence for this proposal (lineage).
    #[serde(default)]
    pub source_memory_ids: Vec<MemoryId>,
    /// Tokens spent producing this proposal (sum of prompt + completion).
    pub token_spend: u32,
    /// LLM model identifier (e.g. `claude-sonnet-4-6`).
    pub model: String,
    /// Sleep mode that produced this proposal (R40).
    pub mode: SleepMode,
}

// =============================================================================
// Evidence pack
// =============================================================================

/// The bundle of inputs handed to the rewriter for one sleep-time pass.
///
/// Assembled by the observer, bounded by token + memory caps (R2). Visibility
/// rules per [`MemoryOrigin::visible_to_evidence`] applied during assembly.
#[derive(Debug, Clone)]
pub struct EvidencePack {
    /// The user whose memories the pack represents.
    pub user_id: String,
    /// Mode the pack was assembled for (R40).
    pub mode: SleepMode,
    /// Trigger that originated the pass.
    pub trigger: SleepTimeTrigger,
    /// Recent / relevant memories (count- and token-capped).
    pub memories: Vec<EvidenceMemory>,
    /// Current state of named context blocks at assembly time (each has its
    /// `version` field captured for OCC).
    pub blocks: Vec<BlockSnapshot>,
    /// Top-K reinforced facts at assembly time (read-only here).
    pub facts: Vec<EvidenceFact>,
    /// Per-block negative-feedback / suppression hints (R10 hard prohibitions).
    pub block_prohibitions: HashMap<String, Vec<String>>,
    /// Approximate total token cost of the pack.
    pub approx_tokens: u32,
    /// Wall-clock assembly time.
    pub assembled_at: DateTime<Utc>,
}

/// A single memory carried into the evidence pack (subset of `Memory`).
#[derive(Debug, Clone)]
pub struct EvidenceMemory {
    pub id: MemoryId,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub importance: f32,
    pub origin: MemoryOrigin,
    pub entity_refs: Vec<String>,
}

/// Snapshot of a context block as seen by the observer; carries the version
/// used for OCC at apply time.
#[derive(Debug, Clone)]
pub struct BlockSnapshot {
    pub key: String,
    pub content: String,
    pub version: u32,
    pub max_tokens: usize,
    pub locked: bool,
}

/// Top-K fact summary (read-only).
#[derive(Debug, Clone)]
pub struct EvidenceFact {
    pub id: String,
    pub content: String,
    pub confidence: f32,
}

// =============================================================================
// Budget state (R12 / R20 / R33 / R37)
// =============================================================================

/// Per-user budget ledger snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetState {
    /// Tokens spent in the current rolling hour.
    pub tokens_this_hour: u32,
    /// LLM calls made in the current rolling day.
    pub calls_today: u32,
    /// Timestamp of the start of the current hour window.
    pub hour_window_start: Option<DateTime<Utc>>,
    /// Timestamp of the start of the current day window.
    pub day_window_start: Option<DateTime<Utc>>,
    /// Block keys explicitly locked by the user (R14).
    #[serde(default)]
    pub locked_blocks: Vec<String>,
}

/// Global (all-users) daily budget tracker (R33).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalBudgetState {
    pub tokens_today: u64,
    pub calls_today: u64,
    pub day_window_start: Option<DateTime<Utc>>,
}

/// Cost-projection sample used by R12/R20.
#[derive(Debug, Clone)]
pub struct CostSample {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prompt_chars: u32,
    pub mode: SleepMode,
}

// =============================================================================
// Diff classification (R1 + diff.rs)
// =============================================================================

/// Classification of a proposed rewrite vs the live block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffClass {
    /// No material change (whitespace, punctuation only).
    Noop,
    /// Small edits (< 10% character change) — auto-apply allowed.
    Minor,
    /// Substantive rewrite (>= 10% character change). Subject to additional
    /// quality-panel scrutiny.
    Substantive,
    /// Very large rewrite (>= 70% replaced or < 30% retained) — may indicate
    /// hallucinated block destruction; rejected by R29 unless explicit.
    Massive,
}

/// Numerical summary of a diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub class: DiffClass,
    pub prior_chars: usize,
    pub new_chars: usize,
    pub retained_ratio: f32,
    pub shrink_ratio: f32,
}

// =============================================================================
// Rewriter inputs / outputs
// =============================================================================

/// Aggregate output of one rewriter invocation. The rewriter may propose zero
/// or more block rewrites and zero or more observations from the same evidence
/// pack and prompt.
#[derive(Debug, Clone)]
pub struct RewriterOutput {
    pub proposals: Vec<RewriteProposal>,
    pub observations: Vec<ObservationDraft>,
    /// Total tokens billed by the LLM provider for this call.
    pub total_tokens: u32,
    /// Model id used for the call.
    pub model: String,
    /// Wall-clock duration of the LLM call.
    pub elapsed_ms: u64,
}

// =============================================================================
// Queue items (R31 + R32 + R67)
// =============================================================================

/// Schema version for [`QueueItem`] persisted in RocksDB. Bumped when the
/// on-disk layout changes; per-version decoders honour this (R32).
pub const QUEUE_ITEM_SCHEMA_VERSION: u32 = 1;

/// A queued sleep-time work item. Persisted in the `sleep_time_queue` CF so
/// triggers survive restart, with cold-start TTL (R31) and version check
/// (R32) gating replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    /// Layout version (R32). Must equal [`QUEUE_ITEM_SCHEMA_VERSION`] on read.
    pub schema_version: u32,
    /// Unique item ID; doubles as the RocksDB key suffix.
    pub id: Uuid,
    /// Whose sleep-time pass to run.
    pub user_id: String,
    /// Trigger that originated the enqueue.
    pub trigger: SleepTimeTrigger,
    /// Mode requested (may be promoted to both NREM+REM in V2 if R66 active).
    pub mode: SleepMode,
    /// When the trigger fired.
    pub enqueued_at: DateTime<Utc>,
    /// If currently claimed by a worker, when the claim expires (R3 lease).
    #[serde(default)]
    pub claim_expires_at: Option<DateTime<Utc>>,
    /// Which worker holds the claim, if any.
    #[serde(default)]
    pub claimed_by: Option<String>,
}

impl QueueItem {
    pub fn new(user_id: impl Into<String>, trigger: SleepTimeTrigger, mode: SleepMode) -> Self {
        Self {
            schema_version: QUEUE_ITEM_SCHEMA_VERSION,
            id: Uuid::new_v4(),
            user_id: user_id.into(),
            trigger,
            mode,
            enqueued_at: Utc::now(),
            claim_expires_at: None,
            claimed_by: None,
        }
    }

    /// True if the claim (if any) has expired and the item is safe to re-claim.
    pub fn claim_expired(&self, now: DateTime<Utc>) -> bool {
        match self.claim_expires_at {
            Some(t) => now >= t,
            None => true,
        }
    }
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SleepTimeError {
    #[error("budget exhausted for user {user_id}: {what}")]
    BudgetExhausted { user_id: String, what: String },

    #[error("block {block_key} is locked")]
    BlockLocked { block_key: String },

    #[error("optimistic concurrency conflict on block {block_key}: expected v{expected}, found v{found}")]
    VersionConflict {
        block_key: String,
        expected: u32,
        found: u32,
    },

    #[error("output validation failed: {0}")]
    OutputValidation(String),

    #[error("rewriter call failed: {0}")]
    RewriterCall(String),

    #[error("rewriter response parse failed: {0}")]
    ParseError(String),

    #[error("cancelled")]
    Cancelled,

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type SleepTimeResult<T> = std::result::Result<T, SleepTimeError>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_visibility_excludes_pre_graduation_from_both_modes() {
        let o = MemoryOrigin::BackgroundSleepTimeObservation;
        assert!(!o.visible_to_evidence(SleepMode::Nrem));
        assert!(!o.visible_to_evidence(SleepMode::Rem));
    }

    #[test]
    fn origin_visibility_graduated_visible_to_rem_only() {
        let o = MemoryOrigin::BackgroundSleepTimeGraduated;
        assert!(!o.visible_to_evidence(SleepMode::Nrem));
        assert!(o.visible_to_evidence(SleepMode::Rem));
    }

    #[test]
    fn origin_visibility_foreground_visible_to_all() {
        for o in [MemoryOrigin::ForegroundUser, MemoryOrigin::ForegroundAgent] {
            assert!(o.visible_to_evidence(SleepMode::Nrem));
            assert!(o.visible_to_evidence(SleepMode::Rem));
        }
    }

    #[test]
    fn foreground_classification() {
        assert!(MemoryOrigin::ForegroundUser.is_foreground());
        assert!(MemoryOrigin::ForegroundAgent.is_foreground());
        assert!(!MemoryOrigin::BackgroundSleepTimeObservation.is_foreground());
        assert!(!MemoryOrigin::BackgroundMaintenance.is_foreground());
        assert!(!MemoryOrigin::OperatorAction.is_foreground());
    }

    #[test]
    fn queue_item_new_carries_schema_version() {
        let q = QueueItem::new("u1", SleepTimeTrigger::Idle, SleepMode::Nrem);
        assert_eq!(q.schema_version, QUEUE_ITEM_SCHEMA_VERSION);
        assert_eq!(q.user_id, "u1");
        assert!(q.claim_expired(Utc::now()));
    }

    #[test]
    fn observation_draft_new_defaults() {
        let d = ObservationDraft::new("hello", SleepMode::Nrem, "v1.minilm");
        assert_eq!(d.content, "hello");
        assert_eq!(d.mode, SleepMode::Nrem);
        assert_eq!(d.embedder_version, "v1.minilm");
        assert_eq!(d.origin, MemoryOrigin::BackgroundSleepTimeObservation);
        assert!(d.referenced_at.is_none());
    }
}
