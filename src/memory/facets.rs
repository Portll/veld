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
}
