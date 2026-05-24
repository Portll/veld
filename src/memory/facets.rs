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

use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
}
