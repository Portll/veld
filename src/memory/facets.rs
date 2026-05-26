//! Record kinds and typed facets — the W3 minimal-core scaffold.
//!
//! Veld stores everything as a [`crate::memory::types::Memory`] today, and its
//! domain context is split inconsistently: rich context lives in named structs
//! (`RichContext` and its members), while robotics context is flat fields on
//! `Memory`/`Experience`/`Query`. The W3 refactor consolidates this into a
//! minimal core record plus *typed facets* — small structs attached only when
//! relevant, so adding a domain never grows the core type.
//!
//! Module contents:
//! - [`RecordKind`] — what kind of thing a record is (memory / plan / prompt /
//!   learning), so one core record serves several lifecycles without
//!   fragmenting retrieval into separate stores.
//! - [`WhereFacet`] + [`Place`] — the spatial / locational facet (the WHERE of
//!   an engram), with layered places: repo → module → file → scope, plus geo /
//!   host / url / named. Mirrors the cognitive-map insight that physical,
//!   organizational, and conceptual space share a code. Absorbs the first-pass
//!   `RepositoryContext` scaffold as the [`Place::Repo`] variant — see
//!   `docs/neuroscience-5w-memory-design.md` for the derivation.
//! - [`PlanFacet`], [`PromptFacet`], [`LearningFacet`] — the kind-specific data
//!   for the non-`Memory` record kinds.
//!
//! The five-W facet model (WHO/WHERE/WHEN/WHY/WHAT) treats `WhereFacet` as one
//! of three optional facets (alongside `WhoFacet` and `WhyFacet`, both pending).
//! WHAT and WHEN are slated to fold into the minimal core rather than become
//! separate facets.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::MemoryId;

// =============================================================================
// FACET SCHEMA VERSIONING
// =============================================================================
//
// Every facet struct (WhereFacet, WhoFacet, WhatFacet, WhenFacet, WhyFacet,
// EngramBinding, AgentSession, etc.) carries a compile-time
// `FACET_SCHEMA_VERSION` constant exposed via the [`FacetVersioned`] trait.
//
// The version is *not* serialised into each instance — that would inflate
// every record by several bytes for a value that is overwhelmingly going to
// be the same. Instead, the version is a *typed claim* the code makes about
// the shape of bytes it produces, used by the migration layer when a
// breaking change ships. Today every facet is at version 1.
//
// `#[serde(default)]` continues to handle additive forward-compat: when an
// old wire shape decodes with missing fields, defaults fill them in.
// [`note_facet_silent_upgrade`] emits one warning the first time such an
// up-conversion happens per process, so operators see drift in the logs
// without the warning storm a per-record log would produce.

/// Compile-time schema version claim for a facet type.
///
/// A facet bumps [`FACET_SCHEMA_VERSION`](FacetVersioned::FACET_SCHEMA_VERSION)
/// at the same time it ships a breaking wire change — the same discipline
/// the intent-log `IntentPayload` uses for its per-record `schema_version`.
/// The trait is intentionally minimal: it exists so the migration registry
/// and audit tools can iterate "every facet's claimed version" without
/// stringly-typed lookups.
pub trait FacetVersioned {
    /// The schema version this binary's code produces for this facet.
    /// Bumped only when the wire bytes for the type change in a way that
    /// older readers cannot handle via `#[serde(default)]` fill-ins.
    const FACET_SCHEMA_VERSION: u16;

    /// Stable, lowercase identifier for the facet — used in logs and in
    /// any future migration manifest.
    const FACET_NAME: &'static str;
}

/// Set of facet schema versions a reader of this codebase recognises.
/// Mirrored from `IntentPayload`'s known-versions set; encountering a
/// value outside this list is an [`FacetVersionError::Unknown`].
pub const KNOWN_FACET_SCHEMA_VERSIONS: &[u16] = &[0, 1];

/// Errors raised when validating a facet's claimed schema version.
#[derive(Debug, thiserror::Error)]
pub enum FacetVersionError {
    /// The provided version is not in [`KNOWN_FACET_SCHEMA_VERSIONS`].
    /// The reader must either be upgraded or run a migration before it
    /// attempts to interpret the bytes.
    #[error(
        "unknown facet '{facet}' schema version {found} (this binary knows {known:?})"
    )]
    Unknown {
        facet: &'static str,
        found: u16,
        known: &'static [u16],
    },
}

/// Validate that `version` is in [`KNOWN_FACET_SCHEMA_VERSIONS`] before
/// the caller deserialises. Returns [`FacetVersionError::Unknown`] when
/// it is not, so the caller surfaces a structured error instead of
/// silently dropping fields the binary doesn't understand.
///
/// `None` is treated as the legacy "version 0" shape and always accepted.
pub fn check_facet_version(
    facet_name: &'static str,
    version: Option<u16>,
) -> Result<(), FacetVersionError> {
    let v = version.unwrap_or(0);
    if KNOWN_FACET_SCHEMA_VERSIONS.contains(&v) {
        Ok(())
    } else {
        Err(FacetVersionError::Unknown {
            facet: facet_name,
            found: v,
            known: KNOWN_FACET_SCHEMA_VERSIONS,
        })
    }
}

/// One-shot guard so an operator sees "schema drift detected" warnings
/// at most once per process. The first silent up-conversion (an older
/// wire shape decoded with `#[serde(default)]`-filled fields) trips this
/// flag; subsequent up-conversions only update internal metrics rather
/// than spamming the log.
static FACET_SILENT_UPGRADE_WARNED: AtomicBool = AtomicBool::new(false);

/// Emit a single warning the first time this process notices a facet
/// wire shape that needed `#[serde(default)]` fill-ins to decode. The
/// argument is the facet name for diagnostics.
///
/// Callers don't need to know *which* fields were missing — that's the
/// job of a future structured diff tool. The point here is to make the
/// upgrade visible at all.
pub fn note_facet_silent_upgrade(facet_name: &'static str) {
    if !FACET_SILENT_UPGRADE_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            target: "veld::memory::facets",
            facet = facet_name,
            "facet wire shape required default fill-ins to decode; \
             this is the first such event this process. Schema drift?"
        );
    }
}

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

/// Spatial / locational facet — the WHERE of an engram.
///
/// Layered places, ordered coarse → fine. A code event might be
/// `[Place::Repo, Place::Module, Place::File]`; a robotics event `[Place::Geo]`.
/// Mirrors the cognitive-map insight (Constantinescu et al. 2016) that
/// physical, organizational, and conceptual space share a code: one engram may
/// simultaneously be in a repository, a module, a file — and for robotics also
/// at a geographic location.
///
/// Replaces the first-pass `RepositoryContext` scaffold by absorbing its fields
/// as the [`Place::Repo`] variant. See `docs/neuroscience-5w-memory-design.md`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhereFacet {
    /// Layered places, ordered coarse → fine.
    pub places: Vec<Place>,
    /// Conceptual neighborhood in the knowledge graph — named by dominant
    /// entities/concepts. The "cognitive-map" where: where in idea-space.
    /// Members are entity names; kept as strings for cross-module portability.
    pub conceptual_anchors: Vec<String>,
    /// Heading / direction of activity. Robotics: physical heading in radians.
    /// Reserved for "direction of work" classification on the code side.
    pub heading: Option<f32>,
}

/// One typed place layer — where in physical, organizational, or conceptual
/// space.
///
/// Serialized internally-tagged with `kind`, e.g.
/// `{"kind": "repo", "slug": "Portll/veld", "branch": "main", ...}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Place {
    /// A version-control repository. Absorbs the first-pass `RepositoryContext`.
    Repo {
        /// Repository name or slug, e.g. `"Portll/veld"`.
        slug: String,
        /// Active branch, when known.
        branch: Option<String>,
        /// Commit SHA, when known.
        commit: Option<String>,
        /// Remote URL.
        remote: Option<String>,
        /// Associated pull/merge request identifier.
        pull_request: Option<String>,
        /// Whether the working tree was dirty when the engram was formed.
        dirty: Option<bool>,
    },
    /// A module / package / namespace within a project.
    Module {
        /// Module path or qualified name.
        path: String,
    },
    /// A single source file, optionally scoped to a symbol.
    File {
        path: String,
        /// Function / type / region within the file.
        symbol: Option<String>,
    },
    /// A geographic location (robotics, geolocation-tagged memory).
    Geo {
        lat: f64,
        lon: f64,
        alt: Option<f64>,
    },
    /// A network host or deployment environment.
    Host {
        name: String,
        /// e.g. `"dev"` / `"staging"` / `"production"`.
        environment: Option<String>,
    },
    /// A web URL.
    Url { href: String },
    /// A free-form named place when nothing else fits.
    Named { label: String },
    /// A position in a named local cartesian frame (robot frame, site
    /// frame, floor plan, etc.). Absorbs the legacy
    /// `Experience.local_position` flat field — the migrator emits
    /// `frame: "robot"` for that source.
    LocalFrame {
        /// Frame identifier — e.g. `"robot"`, `"site"`, `"floor:1"`.
        frame: String,
        x: f32,
        y: f32,
        z: f32,
        /// Pose orientation in this frame, when known.
        orientation: Option<Orientation>,
    },
    /// A geographic fix with full GPS/INS metadata — for callers that
    /// have it. `Place::Geo` stays the simple-case variant; `GeoFix` is
    /// the rich one. `resolved_geo()` reads both.
    GeoFix {
        lat: f64,
        lon: f64,
        alt: Option<f64>,
        /// Horizontal accuracy in meters (95th percentile / HDOP-derived).
        accuracy_m: Option<f32>,
        /// Speed over ground (m/s).
        speed_m_s: Option<f32>,
        /// Course over ground (degrees, true north).
        course_deg: Option<f32>,
        /// When the fix was taken — distinct from `WhenFacet.encoded_at`.
        captured_at: Option<DateTime<Utc>>,
    },
}

/// Full 3-axis orientation in a [`Place::LocalFrame`]. Angles in degrees.
///
/// - **pitch** — rotation about the lateral (Y) axis (nose up / down).
/// - **roll**  — rotation about the longitudinal (X) axis (banking).
/// - **yaw**   — rotation about the vertical (Z) axis (heading; matches
///   the simple-case `WhereFacet.heading` when only yaw is meaningful).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Orientation {
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
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

// =============================================================================
// WHO — provenance + agent identity (W3.3)
// =============================================================================

/// The WHO of an engram — source-monitoring provenance plus the agents involved.
///
/// Splits the two dissociable senses identified in
/// `docs/neuroscience-5w-memory-design.md`: provenance (which channel told us
/// this, with what credibility) and agent identity (the persons in this engram
/// and their roles relative to the self).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhoFacet {
    /// Source-monitoring channel. Subsumes the first-pass `SourceContext`.
    pub provenance: Provenance,
    /// Persons / agents present in this engram with their roles.
    pub agents: Vec<AgentRef>,
}

/// Source-monitoring metadata: where the information came from, how credibly,
/// and through how many hops.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Provenance {
    /// The source channel kind (e.g. `"user"`, `"tool"`, `"github_webhook"`).
    pub source_kind: Option<String>,
    /// Identifier of the source within its kind.
    pub source_id: Option<String>,
    /// Operator-assigned credibility weight, 0.0–1.0. Feeds retrieval ranking.
    pub credibility: f32,
    /// Whether this engram has been independently verified.
    pub verified: bool,
    /// Structured relay chain — each hop the information passed through.
    pub chain: Vec<ProvenanceHop>,
}

/// One hop in a provenance chain.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProvenanceHop {
    pub source_kind: Option<String>,
    pub source_id: Option<String>,
    pub credibility: Option<f32>,
}

/// A reference to an agent (person, organization, self) in an engram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRef {
    /// Link to the graph Person/Organization entity, when known.
    pub entity_id: Option<Uuid>,
    /// Human-readable name.
    pub name: String,
    /// Role this agent plays in the engram.
    pub role: AgentRole,
}

/// How an agent participates in an engram. Self / Author / Subject / Audience /
/// Mentioned are dissociable in episodic memory (distinct social-brain
/// mPFC/TPJ functions) and must not be collapsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// The self — this agent/assistant. mPFC self-referential marker.
    SelfAgent,
    /// Produced or asserted the content.
    Author,
    /// What the content is about — the neutral default.
    #[default]
    Subject,
    /// To whom the content was addressed.
    Audience,
    /// A third party merely mentioned.
    Mentioned,
}

// =============================================================================
// WHY — goals, causal links, event-model boundaries (W3.3)
// =============================================================================

/// The WHY of an engram — goals served, typed causal links, the event model
/// active at encoding, and prediction / boundary signals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhyFacet {
    /// Goals served by this engram, immediate goal first, nesting up.
    pub goal_stack: Vec<GoalRef>,
    /// Typed causal links to other memories (replaces the bare
    /// `causal_chain: Vec<MemoryId>`).
    pub causes: Vec<CausalLink>,
    /// The active event model at encoding — Event Segmentation Theory's
    /// "what is happening and why" frame.
    pub event_model: Option<String>,
    /// Did this engram open a new event? Why did the boundary fire?
    pub boundary: Option<BoundaryCause>,
    /// Prediction at encoding and whether it held — schema-violation signal.
    pub prediction: Option<Prediction>,
    /// Identifier of the schema this engram instantiates (vmPFC schema slot).
    pub schema_id: Option<String>,
}

/// A reference to a goal a record served.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalRef {
    /// Stable identifier — a Plan record id, or a free-form name until goals
    /// get a first-class record kind.
    pub id: String,
    /// Optional human-readable label.
    pub label: Option<String>,
}

/// A typed causal link between this engram and another memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// The other memory.
    pub other: MemoryId,
    /// The kind of causal relation.
    pub relation: CausalRelation,
    /// Confidence in this link, 0.0–1.0.
    pub confidence: f32,
    /// Was this link observed or inferred? Inferred links are weaker evidence
    /// at retrieval — the source-monitoring discipline applied to Veld's own
    /// inferences.
    pub inferred: bool,
}

/// Kinds of causal relation between memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CausalRelation {
    /// Strictly caused.
    #[default]
    Caused,
    /// Triggered without strictly causing.
    Triggered,
    /// Made possible.
    Enabled,
    /// Prevented.
    Prevented,
    /// Provided motivation.
    Motivated,
}

/// Why a new event boundary fired at this engram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryCause {
    /// A new goal opened.
    GoalChange,
    /// An expectation was violated — Event Segmentation Theory's trigger.
    PredictionError,
    /// WHERE shifted significantly.
    LocationChange,
    /// WHO shifted.
    AgentChange,
    /// A long idle gap.
    TemporalGap,
}

/// A prediction made at encoding and its observed outcome.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Prediction {
    /// What was predicted.
    pub expected: Option<String>,
    /// What was observed.
    pub observed: Option<String>,
    /// Whether the prediction held — `None` if not yet evaluated.
    pub accurate: Option<bool>,
}

// =============================================================================
// CONJUNCTIVE BINDING — the hippocampal-index analogue (W3.3)
// =============================================================================

/// Cross-facet binding strength + presence — the hippocampal-index analogue.
///
/// The hippocampus does not store the five W's; it stores a sparse conjunctive
/// code that points at them. `EngramBinding` records *that* an engram's W's are
/// bound, *which* are present and reliable, and *how confidently* — so a
/// partially-reconstructed engram never presents a confabulated conjunction as
/// a real memory.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngramBinding {
    /// Stable hash over the engram's W-anchors (who_id, where_id, when_bucket,
    /// why_id). Collision = candidate duplicate / reconsolidation target.
    pub conjunctive_key: Option<String>,
    /// Recollection strength of the binding itself, 0.0 = gist only, 1.0 = full
    /// episodic detail. Decays faster than the W-facets it binds — models
    /// hippocampal-dependent detail loss while neocortical gist persists.
    pub binding_strength: f32,
    /// Which W-facets carry usable data for this engram.
    pub present: WFacetMask,
}

/// Which W-facets are populated and reliable for an engram. Serialized as an
/// object of bools so adding a sixth W later doesn't break existing data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct WFacetMask {
    pub what: bool,
    /// `where` is a Rust keyword — field name is escaped, JSON key is `where`.
    #[serde(rename = "where")]
    pub where_: bool,
    pub when: bool,
    pub who: bool,
    pub why: bool,
}

// =============================================================================
// WHAT — content with gist/verbatim separation (W3.2c, in the minimal core)
// =============================================================================

/// The WHAT of an engram — content with explicit gist/verbatim separation so
/// consolidation can semanticize without destroying the engram.
///
/// During the transition `Memory.experience.content` remains the primary
/// content; `verbatim` and `gist` are populated by encoding/consolidation as
/// they roll out. Unlike WHERE/WHO/WHY/binding, this facet is on the *core*
/// — every memory has content.
/// See `docs/neuroscience-5w-memory-design.md`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhatFacet {
    /// Original surface form. May be shed as the memory ages (hippocampal
    /// detail loss).
    pub verbatim: Option<String>,
    /// Abstracted summary; survives consolidation. Becomes the primary
    /// content once the engram semanticizes.
    pub gist: Option<String>,
    /// Type of content. Will eventually replace `ExperienceType` as the
    /// WHAT discriminant.
    pub content_kind: ContentKind,
    /// Content salience, 0.0–1.0. Distinct from emotional arousal.
    pub content_salience: f32,
    /// Abstraction level: 0.0 = raw episode, 1.0 = fully semanticized schema.
    /// Increments during consolidation as `verbatim` is shed.
    pub abstraction_level: f32,
}

/// Coarse content kind. Will eventually replace the existing `ExperienceType`
/// enum as the WHAT discriminant once migration completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContentKind {
    /// Default — a recorded observation.
    #[default]
    Observation,
    /// A decision made.
    Decision,
    /// A distilled insight (mirrors `RecordKind::Learning`).
    Learning,
    /// An action taken.
    Action,
    /// A reflection or meta-thought.
    Reflection,
    /// Conversational exchange.
    Conversation,
    /// Code or code-related artifact.
    Code,
    /// Document content.
    Document,
    /// Anything else.
    Other,
}

// =============================================================================
// WHEN — encoding/event time + TCM drift vector (W3.2c, in the minimal core)
// =============================================================================

/// The WHEN of an engram — encoding time, event time, ordinal position, and a
/// Temporal Context Model (TCM) drift vector for graded contiguity.
///
/// `encoded_at` mirrors `Memory.created_at` during the transition so the facet
/// is self-contained for indexing. `event_time` is the *separately-encoded*
/// time at which the described event actually happened. Like `WhatFacet`, this
/// is on the minimal core (every memory has a time).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhenFacet {
    /// When Veld stored this engram.
    pub encoded_at: Option<DateTime<Utc>>,
    /// When the described event actually happened — distinct from encoding.
    pub event_time: Option<TimeSpan>,
    /// Ordinal position within the episode.
    pub episode_ordinal: Option<u32>,
    /// Temporal Context Model drift vector — a low-dim slowly-drifting state
    /// captured at encoding. Cosine distance between two engrams' drift
    /// vectors = subjective-time proximity. Graded contiguity signal that
    /// replaces brittle `episode_id` equality at retrieval.
    pub context_drift: Vec<f32>,
    /// Detected recurrence pattern (e.g. "every Monday standup").
    pub recurrence: Option<RecurrencePattern>,
}

/// A bounded time interval with precision metadata.
///
/// Does not derive `Default` — there is no neutral "default time". Use
/// `Option<TimeSpan>` to express absence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSpan {
    /// Interval start.
    pub start: DateTime<Utc>,
    /// Interval end. `None` = point-in-time.
    pub end: Option<DateTime<Utc>>,
    /// Granularity of the reference: was it `"2026"`, `"May 2026"`, an instant?
    pub precision: TimePrecision,
}

/// Granularity of a time reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TimePrecision {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    #[default]
    Instant,
}

/// A detected recurrence pattern. Free-form for now; structured RRULE-style
/// fields can be added later without breaking existing data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecurrencePattern {
    /// e.g. `"weekly"`, `"daily"`, `"monthly"`.
    pub period: Option<String>,
    /// Free-form description of the recurrence rule.
    pub description: Option<String>,
}

// =============================================================================
// AGENT SESSION — worktree / branch / agent identity at encoding time
// =============================================================================

/// Records which agent was operating from which git worktree/branch at the
/// moment a memory was written.
///
/// This is the source-of-truth for "who ran what, where, and when" in the
/// worktree-per-agent topology the project is moving toward: one git worktree
/// per agent×branch, with windows and config auto-derived from the branch
/// name. Veld needs to capture that identity at encoding so memories can later
/// be filtered, attributed, and visualised by agent or by branch — and so the
/// planned branch-aware tooling (worktree viewer, branch-scoped recall, the
/// git viewer) has a reliable provenance axis to query.
///
/// `AgentSession` is intentionally separate from [`WhoFacet`]. `WhoFacet` is
/// about *who is named in the engram* (the social-brain axis). `AgentSession`
/// is about *which physical agent process* recorded the engram — a host /
/// runtime identity that exists even for a memory that has no human subjects
/// at all. Both can coexist on the same record.
///
/// All fields are `Option`-typed and `#[serde(default)]` so older records
/// (written before this facet existed) deserialize cleanly to an empty
/// session, and so partial captures (e.g. branch known, worktree path not
/// yet probed) round-trip without losing the fields that *were* captured.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AgentSession {
    /// Absolute path of the active worktree this agent was running in.
    #[serde(default)]
    pub worktree_path: Option<PathBuf>,
    /// Git branch checked out in that worktree.
    #[serde(default)]
    pub branch: Option<String>,
    /// Chat-brand identity of the agent — e.g. `"Claude"`, `"Copilot"`,
    /// `"Cursor"`. Deliberately the *agentic-chat* brand rather than the
    /// launcher binary (`claude-code` / `claude-cli` / Claude Desktop all
    /// collapse to `"Claude"`) so the same conversation identity reads the
    /// same across surfaces. Populated by auto-detecting the parent process
    /// at session start: when the parent is a known chat launcher (`code`
    /// with a Claude/Copilot extension active, the Claude binaries, `cursor`,
    /// etc.) the brand follows. The binary / launcher slug is the fallback
    /// when detection is ambiguous.
    #[serde(default)]
    pub agent_id: Option<String>,
    /// VS Code window id when the agent is running inside a VS Code window.
    /// Lets the planned viewer disambiguate concurrent windows on the same
    /// worktree.
    #[serde(default)]
    pub vscode_window_id: Option<String>,
    /// When this agent session began — distinct from the engram's
    /// `encoded_at`. Multiple memories recorded in one session share this
    /// timestamp, which is what makes per-session filtering possible.
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    /// Path of the main worktree (the original clone) when this session is
    /// running in a sibling/linked worktree. Anchors a child worktree back to
    /// its parent so a branch-aware tool can group siblings.
    #[serde(default)]
    pub parent_repo: Option<PathBuf>,
}

// =============================================================================
// FacetVersioned impls — one per first-class facet type
// =============================================================================
//
// Every impl here is `FACET_SCHEMA_VERSION: u16 = 1`. The first breaking
// change to a facet's wire shape bumps that facet's number, alongside a
// migration arm in `crate::intent_log::migrations`. Adding an impl for a new
// facet does NOT need a coordinated bump anywhere else — the trait is purely
// per-type metadata.

impl FacetVersioned for WhereFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "where";
}

impl FacetVersioned for WhoFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "who";
}

impl FacetVersioned for WhyFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "why";
}

impl FacetVersioned for WhatFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "what";
}

impl FacetVersioned for WhenFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "when";
}

impl FacetVersioned for EngramBinding {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "engram_binding";
}

impl FacetVersioned for PlanFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "plan";
}

impl FacetVersioned for PromptFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "prompt";
}

impl FacetVersioned for LearningFacet {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "learning";
}

impl FacetVersioned for AgentSession {
    const FACET_SCHEMA_VERSION: u16 = 1;
    const FACET_NAME: &'static str = "agent_session";
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
    fn where_facet_defaults_are_empty() {
        let w = WhereFacet::default();
        assert!(w.places.is_empty());
        assert!(w.conceptual_anchors.is_empty());
        assert!(w.heading.is_none());
    }

    #[test]
    fn place_repo_round_trips_with_internal_tag() {
        let p = Place::Repo {
            slug: "Portll/veld".into(),
            branch: Some("main".into()),
            commit: Some("abc123".into()),
            remote: None,
            pull_request: None,
            dirty: Some(false),
        };
        let json = serde_json::to_string(&p).unwrap();
        // Internal tagging: the `kind` field carries the variant.
        assert!(json.contains("\"kind\":\"repo\""));
        let back: Place = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn where_facet_layered_places() {
        let w = WhereFacet {
            places: vec![
                Place::Repo {
                    slug: "Portll/veld".into(),
                    branch: Some("main".into()),
                    commit: None,
                    remote: None,
                    pull_request: None,
                    dirty: None,
                },
                Place::File {
                    path: "src/memory/facets.rs".into(),
                    symbol: Some("WhereFacet".into()),
                },
            ],
            conceptual_anchors: vec!["facet-refactor".into()],
            heading: None,
        };
        assert_eq!(w.places.len(), 2);
        let json = serde_json::to_string(&w).unwrap();
        let back: WhereFacet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.places.len(), 2);
        assert_eq!(back.conceptual_anchors, w.conceptual_anchors);
    }

    #[test]
    fn agent_role_defaults_to_subject() {
        assert_eq!(AgentRole::default(), AgentRole::Subject);
    }

    #[test]
    fn who_facet_round_trip() {
        let w = WhoFacet {
            provenance: Provenance {
                source_kind: Some("user".into()),
                source_id: Some("john".into()),
                credibility: 0.9,
                verified: true,
                chain: vec![],
            },
            agents: vec![AgentRef {
                entity_id: None,
                name: "John".into(),
                role: AgentRole::Author,
            }],
        };
        let json = serde_json::to_string(&w).unwrap();
        let back: WhoFacet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agents.len(), 1);
        assert_eq!(back.agents[0].role, AgentRole::Author);
        assert!(back.provenance.verified);
    }

    #[test]
    fn why_facet_with_causal_link_round_trip() {
        let other = MemoryId(Uuid::new_v4());
        let why = WhyFacet {
            goal_stack: vec![GoalRef {
                id: "ship-w3".into(),
                label: Some("ship W3".into()),
            }],
            causes: vec![CausalLink {
                other,
                relation: CausalRelation::Triggered,
                confidence: 0.8,
                inferred: true,
            }],
            event_model: Some("planning the next step".into()),
            boundary: Some(BoundaryCause::GoalChange),
            prediction: None,
            schema_id: None,
        };
        let json = serde_json::to_string(&why).unwrap();
        let back: WhyFacet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.goal_stack.len(), 1);
        assert_eq!(back.causes.len(), 1);
        assert_eq!(back.causes[0].relation, CausalRelation::Triggered);
        assert!(back.causes[0].inferred);
        assert_eq!(back.boundary, Some(BoundaryCause::GoalChange));
    }

    #[test]
    fn engram_binding_defaults_are_inert() {
        let b = EngramBinding::default();
        assert!(b.conjunctive_key.is_none());
        assert_eq!(b.binding_strength, 0.0);
        assert!(!b.present.what);
        assert!(!b.present.where_);
        assert!(!b.present.when);
        assert!(!b.present.who);
        assert!(!b.present.why);
    }

    #[test]
    fn wfacet_mask_serializes_where_with_rust_keyword_safe_alias() {
        let m = WFacetMask {
            what: true,
            where_: true,
            when: false,
            who: false,
            why: false,
        };
        let json = serde_json::to_string(&m).unwrap();
        // The escaped `where_` Rust field serializes to JSON `"where"`.
        assert!(json.contains("\"where\":true"));
        assert!(!json.contains("where_"));
        let back: WFacetMask = serde_json::from_str(&json).unwrap();
        assert!(back.where_);
    }

    #[test]
    fn what_facet_defaults_neutral() {
        let w = WhatFacet::default();
        assert!(w.verbatim.is_none());
        assert!(w.gist.is_none());
        assert_eq!(w.content_kind, ContentKind::Observation);
        assert_eq!(w.content_salience, 0.0);
        assert_eq!(w.abstraction_level, 0.0);
    }

    #[test]
    fn content_kind_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&ContentKind::Conversation).unwrap(),
            "\"conversation\""
        );
        let back: ContentKind = serde_json::from_str("\"reflection\"").unwrap();
        assert_eq!(back, ContentKind::Reflection);
    }

    #[test]
    fn time_precision_defaults_to_instant() {
        assert_eq!(TimePrecision::default(), TimePrecision::Instant);
    }

    #[test]
    fn when_facet_round_trip_preserves_temporal_anchors() {
        let now = chrono::Utc::now();
        let w = WhenFacet {
            encoded_at: Some(now),
            event_time: Some(TimeSpan {
                start: now,
                end: None,
                precision: TimePrecision::Hour,
            }),
            episode_ordinal: Some(3),
            context_drift: vec![0.1, 0.2, 0.3],
            recurrence: None,
        };
        let json = serde_json::to_string(&w).unwrap();
        let back: WhenFacet = serde_json::from_str(&json).unwrap();
        assert!(back.encoded_at.is_some());
        assert_eq!(back.episode_ordinal, Some(3));
        assert_eq!(back.context_drift.len(), 3);
        assert!(matches!(
            back.event_time.unwrap().precision,
            TimePrecision::Hour
        ));
    }

    #[test]
    fn agent_session_default_bincode_round_trip() {
        // Every field None — the shape that lands on every existing record
        // before any agent-identity probe has run. Must round-trip cleanly
        // through the project's chosen binary format.
        let session = AgentSession::default();
        let bytes =
            bincode::serde::encode_to_vec(&session, bincode::config::standard()).unwrap();
        let (back, _): (AgentSession, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(back, session);
        assert!(back.worktree_path.is_none());
        assert!(back.branch.is_none());
        assert!(back.agent_id.is_none());
        assert!(back.vscode_window_id.is_none());
        assert!(back.started_at.is_none());
        assert!(back.parent_repo.is_none());
    }

    #[test]
    fn agent_session_populated_bincode_round_trip() {
        let started_at = chrono::Utc::now();
        let session = AgentSession {
            worktree_path: Some(PathBuf::from(
                "/c/Repositories/Portll/veld/.claude/worktrees/agent-abc",
            )),
            branch: Some("feat/agent-session-facet".into()),
            agent_id: Some("claude-code".into()),
            vscode_window_id: Some("window-42".into()),
            started_at: Some(started_at),
            parent_repo: Some(PathBuf::from("/c/Repositories/Portll/veld")),
        };
        let bytes =
            bincode::serde::encode_to_vec(&session, bincode::config::standard()).unwrap();
        let (back, _): (AgentSession, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(back, session);
    }

    #[test]
    fn agent_session_json_round_trip_for_wire_shape() {
        // JSON is the externally-visible shape (MCP / HTTP / hook payloads).
        // Lock the round-trip and confirm the field names land where callers
        // expect them.
        let started_at = chrono::Utc::now();
        let session = AgentSession {
            worktree_path: Some(PathBuf::from("/repo/wt")),
            branch: Some("main".into()),
            agent_id: Some("vscode-copilot".into()),
            vscode_window_id: Some("vscode-1".into()),
            started_at: Some(started_at),
            parent_repo: Some(PathBuf::from("/repo")),
        };
        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains("\"branch\":\"main\""));
        assert!(json.contains("\"agent_id\":\"vscode-copilot\""));
        assert!(json.contains("\"vscode_window_id\":\"vscode-1\""));
        let back: AgentSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back, session);
    }

    // ========================================================================
    // Facet schema versioning
    // ========================================================================

    #[test]
    fn every_facet_claims_schema_version_1_today() {
        // One assertion per facet — pinning the current cohort. The first
        // breaking change to any one of these bumps that facet's number
        // in isolation; the test then needs to learn the new number.
        assert_eq!(WhereFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(WhoFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(WhyFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(WhatFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(WhenFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(EngramBinding::FACET_SCHEMA_VERSION, 1);
        assert_eq!(PlanFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(PromptFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(LearningFacet::FACET_SCHEMA_VERSION, 1);
        assert_eq!(AgentSession::FACET_SCHEMA_VERSION, 1);
    }

    #[test]
    fn facet_name_strings_are_stable_lowercase_identifiers() {
        // Used in logs / future migration manifests — must not drift.
        assert_eq!(WhereFacet::FACET_NAME, "where");
        assert_eq!(WhoFacet::FACET_NAME, "who");
        assert_eq!(WhyFacet::FACET_NAME, "why");
        assert_eq!(WhatFacet::FACET_NAME, "what");
        assert_eq!(WhenFacet::FACET_NAME, "when");
        assert_eq!(EngramBinding::FACET_NAME, "engram_binding");
        assert_eq!(AgentSession::FACET_NAME, "agent_session");
    }

    #[test]
    fn known_facet_versions_accept_none_and_current() {
        check_facet_version("where", None).unwrap();
        check_facet_version("where", Some(0)).unwrap();
        check_facet_version("where", Some(1)).unwrap();
    }

    #[test]
    fn unknown_facet_version_returns_structured_error() {
        let err = check_facet_version("where", Some(999)).unwrap_err();
        match err {
            FacetVersionError::Unknown { facet, found, known } => {
                assert_eq!(facet, "where");
                assert_eq!(found, 999);
                assert_eq!(known, KNOWN_FACET_SCHEMA_VERSIONS);
            }
        }
    }
}
