//! Record kinds and typed facets — the W3 minimal-core scaffold.
//!
//! Veld stores everything as a [`crate::memory::types::Memory`] today, and its
//! domain context is split inconsistently: rich context lives in named structs
//! (`RichContext` and its members), while robotics context is flat fields on
//! `Memory`/`Experience`/`Query`. The W3 refactor consolidates this into a
//! minimal core record plus *typed facets* — small structs attached only when
//! relevant, so adding a domain never grows the core type.
//!
//! This module is the scaffold for that refactor. It defines:
//! - [`RecordKind`] — what kind of thing a record is (memory / plan / prompt /
//!   learning), so one core record serves several lifecycles without
//!   fragmenting retrieval into separate stores.
//! - [`RepositoryContext`] — version-control identity, the facet a coding agent
//!   needs. It complements `CodeContext` (the live editing cursor) rather than
//!   duplicating it.
//! - [`PlanFacet`], [`PromptFacet`], [`LearningFacet`] — the kind-specific data
//!   for the non-`Memory` record kinds.
//!
//! ## Wiring (W3 step 2 — separate, deliberate change)
//! 1. Add `#[serde(default)] pub kind: RecordKind` to `Memory`. Existing
//!    serialized memories deserialize as [`RecordKind::Memory`].
//! 2. Add `#[serde(default)] pub repository: RepositoryContext` to
//!    `RichContext` (4 construction sites: `memory/context.rs`,
//!    `memory/types.rs`, `handlers/remember.rs`, `tests/adaptive_memory_tests.rs`).
//! 3. Attach the kind facets behind a `RecordKind`-tagged optional field.
//!
//! Until then these types are defined but not yet referenced by the core
//! structs — they compile standalone and are part of the public crate API.

use serde::{Deserialize, Serialize};

use super::types::MemoryId;

/// The kind of record a core row carries.
///
/// One core record represents distinct kinds — a remembered experience, a plan,
/// a reusable prompt, a distilled learning — each with its own lifecycle and
/// kind-specific facet, while retrieval stays unified (callers filter by kind).
/// Modelling these as one discriminant rather than four top-level types keeps
/// recall from fragmenting across four stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    /// A remembered experience — the default; everything Veld stores today.
    #[default]
    Memory,
    /// A plan: a goal plus ordered steps with a status lifecycle.
    Plan,
    /// A reusable prompt with provenance.
    Prompt,
    /// A distilled insight derived from one or more memories.
    Learning,
}

impl RecordKind {
    /// Stable lowercase string form (matches the serde representation).
    pub fn as_str(&self) -> &'static str {
        match self {
            RecordKind::Memory => "memory",
            RecordKind::Plan => "plan",
            RecordKind::Prompt => "prompt",
            RecordKind::Learning => "learning",
        }
    }
}

/// Version-control identity and state for a record.
///
/// Distinct from `CodeContext`, which is the *live editing cursor* — current
/// file, current scope, call stack. `RepositoryContext` is the *repository
/// identity and VCS state*: which repo, which branch, which commit. It is the
/// structured form of what the `veld hook commit` sync currently stuffs into
/// free-text content and string tags.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepositoryContext {
    /// Repository name or slug, e.g. `"Portll/veld"`.
    pub repo: Option<String>,
    /// Active branch.
    pub branch: Option<String>,
    /// Commit SHA this record is associated with.
    pub commit: Option<String>,
    /// Remote URL.
    pub remote: Option<String>,
    /// Files touched by, or relevant to, this record.
    pub files: Vec<String>,
    /// Code symbols (functions, types) relevant to this record.
    pub symbols: Vec<String>,
    /// Associated pull/merge request identifier.
    pub pull_request: Option<String>,
    /// Whether the working tree was dirty when the record was formed.
    pub dirty: Option<bool>,
}

/// Lifecycle status of a [`PlanFacet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    /// Drafted but not yet started.
    #[default]
    Draft,
    /// Actively being executed.
    Active,
    /// Stalled on an external dependency.
    Blocked,
    /// Completed.
    Done,
    /// Dropped without completion.
    Abandoned,
}

/// One ordered step within a [`PlanFacet`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlanStep {
    /// What the step does.
    pub description: String,
    /// Whether the step is complete.
    pub done: bool,
}

/// Kind-specific data for a [`RecordKind::Plan`] record.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlanFacet {
    /// The goal this plan serves.
    pub goal: Option<String>,
    /// Ordered steps.
    pub steps: Vec<PlanStep>,
    /// Lifecycle status.
    pub status: PlanStatus,
}

/// Kind-specific data for a [`RecordKind::Prompt`] record.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptFacet {
    /// The prompt text or template.
    pub template: String,
    /// What this prompt is for / where it came from.
    pub purpose: Option<String>,
    /// Named variables the template expects.
    pub variables: Vec<String>,
}

/// Kind-specific data for a [`RecordKind::Learning`] record.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearningFacet {
    /// The insight statement.
    pub insight: String,
    /// Memories this learning was distilled from.
    pub derived_from: Vec<MemoryId>,
    /// Confidence in the learning, 0.0–1.0.
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_kind_defaults_to_memory() {
        assert_eq!(RecordKind::default(), RecordKind::Memory);
        assert_eq!(RecordKind::default().as_str(), "memory");
    }

    #[test]
    fn record_kind_serde_is_snake_case() {
        let json = serde_json::to_string(&RecordKind::Learning).unwrap();
        assert_eq!(json, "\"learning\"");
        let back: RecordKind = serde_json::from_str("\"plan\"").unwrap();
        assert_eq!(back, RecordKind::Plan);
    }

    #[test]
    fn plan_status_defaults_to_draft() {
        assert_eq!(PlanStatus::default(), PlanStatus::Draft);
    }

    #[test]
    fn facets_round_trip_through_json() {
        let plan = PlanFacet {
            goal: Some("ship W3".into()),
            steps: vec![PlanStep {
                description: "scaffold facets".into(),
                done: true,
            }],
            status: PlanStatus::Active,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let back: PlanFacet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.goal, plan.goal);
        assert_eq!(back.steps.len(), 1);
        assert!(back.steps[0].done);
        assert_eq!(back.status, PlanStatus::Active);
    }

    #[test]
    fn repository_context_defaults_are_empty() {
        let rc = RepositoryContext::default();
        assert!(rc.repo.is_none());
        assert!(rc.files.is_empty());
        assert!(rc.symbols.is_empty());
    }
}
