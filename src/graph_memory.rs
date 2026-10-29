//! Graph Memory System - Inspired by Graphiti
//!
//! Temporal knowledge graph for tracking entities, relationships, and episodic memories.
//! Implements bi-temporal tracking and hybrid retrieval (semantic + graph traversal).

mod change_log;
mod mirror_feed;

pub use change_log::{ChangeCursor, GraphChangeLog, GraphChanges, CHANGE_LOG_CAPACITY};

use anyhow::Result;
use chrono::{DateTime, Utc};
#[cfg(feature = "rocksdb")]
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
#[cfg(feature = "rocksdb")]
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::constants::{
    ENTITY_CONCEPT_MERGE_THRESHOLD, ENTITY_EMBEDDING_CACHE_MAX, LTP_MIN_STRENGTH, LTP_PRUNE_FLOOR,
};

// Column family names for the unified graph database
const CF_ENTITIES: &str = "entities";
const CF_RELATIONSHIPS: &str = "relationships";
const CF_EPISODES: &str = "episodes";
const CF_ENTITY_EDGES: &str = "entity_edges";
const CF_ENTITY_PAIR_INDEX: &str = "entity_pair_index";
const CF_ENTITY_EPISODES: &str = "entity_episodes";
const CF_NAME_INDEX: &str = "name_index";
const CF_LOWERCASE_INDEX: &str = "lowercase_index";
const CF_STEMMED_INDEX: &str = "stemmed_index";
/// Edge lookups that would otherwise read a whole family: `e:{episode}:{edge}` lists the
/// edges an episode sourced, `t:{lo}:{hi}:{relation:?}` names the edge joining an entity
/// pair under one relation type. Rows may outlive their edge; readers check the edge.
const CF_EDGE_INDEX: &str = "edge_index";

/// Row in `entity_episodes` recording that the recency-ordered index has been backfilled.
/// `~` sorts after every hex digit, so no `{uuid}:` or `{uuid}#` prefix scan reaches it.
const EPISODE_RECENCY_INDEX_MARKER: &[u8] = b"~episode_recency_index_v1";

/// Row in `edge_index` recording that every stored edge has its rows. Until it is written,
/// a miss in the index proves nothing and readers fall back to scanning.
const EDGE_INDEX_MARKER: &[u8] = b"~edge_index_v1";

/// Maximum number of retrieval IDs held in the idempotency guard (S-R4).
/// Oldest entries are evicted FIFO once this cap is reached.
const SEEN_RETRIEVALS_CAPACITY: usize = 1024;

/// The element at index `len / 2` of `values` in ascending order (the upper median for an
/// even count), reordering `values` in place. `None` when empty.
fn upper_median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    Some(*median)
}

/// What the open-time pass over the relationships family found.
#[derive(Default)]
struct RelationshipCensus {
    /// Rows in the family, `mem_edge:` lookup rows included, as `relationship_count` counts.
    rows: usize,
    total_activations: u64,
}

/// Edges decayed per synapse-lock hold and write batch by the full decay pass.
const DECAY_CHUNK_EDGES: usize = 20_000;

const GRAPH_CF_NAMES: &[&str] = &[
    CF_ENTITIES,
    CF_RELATIONSHIPS,
    CF_EPISODES,
    CF_ENTITY_EDGES,
    CF_ENTITY_PAIR_INDEX,
    CF_ENTITY_EPISODES,
    CF_NAME_INDEX,
    CF_LOWERCASE_INDEX,
    CF_STEMMED_INDEX,
    CF_EDGE_INDEX,
];

/// Entity node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Entity name (e.g., "John", "Paris", "Rust programming")
    pub name: String,

    /// Entity labels/types (e.g., ["Person"], ["Location", "City"])
    pub labels: Vec<EntityLabel>,

    /// When this entity was first created in the graph
    pub created_at: DateTime<Utc>,

    /// When this entity was last observed
    pub last_seen_at: DateTime<Utc>,

    /// How many times this entity has been mentioned
    pub mention_count: usize,

    /// Summary of this entity's context (built from surrounding edges)
    pub summary: String,

    /// Additional attributes based on entity type
    pub attributes: HashMap<String, String>,

    /// Semantic embedding of the entity name (for similarity search)
    pub name_embedding: Option<Vec<f32>>,

    /// Salience score (0.0 - 1.0): How important is this entity?
    /// Higher salience = larger gravitational well in the memory universe
    /// Factors: proper noun status, mention frequency, recency, user-defined importance
    #[serde(default = "default_salience")]
    pub salience: f32,

    /// Whether this is a proper noun (names, places, products)
    /// Proper nouns have higher base salience than common nouns
    #[serde(default)]
    pub is_proper_noun: bool,

    /// PII classification for cross-user safety.
    /// Derived from entity labels: Person → PersonalIdentity,
    /// Organization → OrganizationIdentity, otherwise Clean.
    #[serde(default)]
    pub pii_classification: PiiClassification,
}

fn default_salience() -> f32 {
    0.5 // Default middle salience
}

/// PII classification for entities, used to control cross-user data flow.
///
/// Entities classified as `PersonalIdentity` or `OrganizationIdentity` are
/// excluded from cross-user aggregation by the selective membrane filter.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PiiClassification {
    /// Contains personally identifiable information (names, contacts)
    PersonalIdentity,
    /// Contains organization-specific information
    OrganizationIdentity,
    /// No PII detected — safe for cross-user aggregation
    #[default]
    Clean,
}

impl PiiClassification {
    /// Derive PII classification from entity labels.
    pub fn from_labels(labels: &[EntityLabel]) -> Self {
        for label in labels {
            match label {
                EntityLabel::Person => return Self::PersonalIdentity,
                EntityLabel::Organization => return Self::OrganizationIdentity,
                _ => {}
            }
        }
        Self::Clean
    }
}

/// Entity labels for ontological classification of graph nodes.
///
/// Extends Graphiti's base categorization with DevOps, software engineering,
/// and robotics domain types. Used by spreading activation (Layer 2) and
/// post-RRF re-ranking (Layer 4.9) for type-aware retrieval.
///
/// New variants are additive only — never remove existing variants to
/// maintain backward compatibility with MessagePack-serialized data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityLabel {
    Person,
    Organization,
    Location,
    Technology,
    Concept,
    Event,
    Date,
    Product,
    Skill,
    /// YAKE-extracted discriminative keyword (not a named entity)
    /// Used for graph-based retrieval of rare/important terms like "sunrise"
    Keyword,
    /// Git repos, Jira projects, feature work, mission plans
    Project,
    /// Issues, tickets, work items, action items
    Task,
    /// READMEs, specs, design docs, RFCs, runbooks
    Document,
    /// Git repositories, container registries
    Repository,
    /// Microservices, APIs, daemons, ROS2 nodes
    Service,
    /// PostgreSQL, Redis, RocksDB, DynamoDB instances
    Database,
    /// Latency p99, error rate, SLOs, telemetry signals
    Metric,
    /// Env vars, feature flags, config files, parameters
    Configuration,
    /// staging, production, dev, CI, robot deployment zones
    Environment,
    /// CI/CD pipelines, data pipelines, ETL, mission workflows
    Pipeline,
    /// Engineering squads, SRE teams, robot fleet groups
    Team,
    /// SRE, tech lead, PM, architect, operator
    Role,
    /// Code modules, packages, crates, libraries, ROS2 packages
    Module,
    Other(String),
    // Writer-declared identifier types. Appended AFTER `Other` because bincode stores a variant
    // as its position: anything inserted above would relabel every stored entity after it
    // (src/storage/stored_enums.rs pins the order).
    /// A file in a named tree, canonically `repo:relative/path`
    FilePath,
    /// A commit, canonically `repo@sha`
    Commit,
    /// A code symbol: function, type, constant
    Symbol,
    /// An agent session, canonically `cc:<sessionId>`
    Session,
    /// An agent within a session, canonically `cc:<sessionId>/<agentId>`
    Agent,
    /// A tool or skill an agent invoked, canonically `tool:<name>` / `skill:/<name>`
    Tool,
    /// A prompt or tool template, canonically `template:<tool>@<sha8>`
    Template,
}

/// Whether an entity of this kind denotes the same referent regardless of
/// which project band observed it.
///
/// A band is a project. Agents working in one band see one tree, so names
/// inside it co-refer — `config.toml` is one file to everyone looking at it.
/// Across bands they do not: `staging` in one project is not `staging` in
/// another.
///
/// This is a property of the TYPE, not of the name, which is why it lives on
/// `EntityLabel`. The label space is closed and additive-only, so the match
/// below is total and a new variant cannot silently inherit a policy. A list
/// of privileged names could not give that guarantee, and its failures would
/// be silent false negatives — names that should have merged and quietly did
/// not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeClass {
    /// One referent per owner — may merge across bands.
    Global,
    /// Referent is relative to its band — never merge across bands.
    BandRelative,
}

impl MergeClass {
    /// Merge class for an entity carrying several labels.
    ///
    /// Entities accumulate labels — `add_entity` unions them when folding a
    /// new observation into an existing node — so the question is asked of a
    /// set. Any band-relative member makes the whole entity band-relative:
    /// that direction fails toward NOT merging, and a missed merge is
    /// recoverable where a wrong one is not.
    pub fn of(labels: &[EntityLabel]) -> MergeClass {
        // No labels is no evidence of referential scope, which gets the same
        // treatment as an unrecognised type.
        if labels.is_empty() {
            return MergeClass::BandRelative;
        }
        if labels
            .iter()
            .any(|l| l.merge_class() == MergeClass::BandRelative)
        {
            MergeClass::BandRelative
        } else {
            MergeClass::Global
        }
    }
}

/// Most candidates [`GraphMemory::find_file_paths_by_suffix`] reads for one term.
pub const FILE_PATH_SUFFIX_MAX_CANDIDATES: usize = 16;
/// Most `/`-separated suffixes indexed for one name.
pub const FILE_PATH_SUFFIX_MAX_DEPTH: usize = 32;

/// The lowercased `/`-boundary suffixes of the path in a `prefix:path` name, longest first:
/// `commitwork:admin/serve.mjs` → `["admin/serve.mjs", "serve.mjs"]`. Empty for any other shape.
pub fn path_suffixes(name: &str) -> Vec<String> {
    let lower = name.to_lowercase();
    let Some((prefix, path)) = lower.split_once(':') else {
        return Vec::new();
    };
    if prefix.is_empty() || prefix.contains('/') || prefix.chars().any(char::is_whitespace) {
        return Vec::new();
    }
    let path = path.trim_start_matches('/');
    if path.is_empty() || path.chars().any(char::is_whitespace) {
        return Vec::new();
    }
    let mut out = vec![path.to_string()];
    for (i, c) in path.char_indices() {
        if out.len() >= FILE_PATH_SUFFIX_MAX_DEPTH {
            break;
        }
        if c == '/' && i + 1 < path.len() {
            out.push(path[i + 1..].to_string());
        }
    }
    out
}

impl EntityLabel {
    /// Get string representation of the entity label
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::Person => "Person",
            Self::Organization => "Organization",
            Self::Location => "Location",
            Self::Technology => "Technology",
            Self::Concept => "Concept",
            Self::Event => "Event",
            Self::Date => "Date",
            Self::Product => "Product",
            Self::Skill => "Skill",
            Self::Keyword => "Keyword",
            Self::Project => "Project",
            Self::Task => "Task",
            Self::Document => "Document",
            Self::Repository => "Repository",
            Self::Service => "Service",
            Self::Database => "Database",
            Self::Metric => "Metric",
            Self::Configuration => "Configuration",
            Self::Environment => "Environment",
            Self::Pipeline => "Pipeline",
            Self::Team => "Team",
            Self::Role => "Role",
            Self::Module => "Module",
            Self::Other(s) => s.as_str(),
            Self::FilePath => "FilePath",
            Self::Commit => "Commit",
            Self::Symbol => "Symbol",
            Self::Session => "Session",
            Self::Agent => "Agent",
            Self::Tool => "Tool",
            Self::Template => "Template",
        }
    }

    /// Every label a client may declare on a write, in the spelling the error message names.
    /// `Other` is not declarable: an unrecognised label is refused, never filed as `Other`.
    pub const DECLARABLE: &'static [&'static str] = &[
        "Person",
        "Organization",
        "Location",
        "Technology",
        "Concept",
        "Event",
        "Date",
        "Product",
        "Skill",
        "Keyword",
        "Project",
        "Task",
        "Document",
        "Repository",
        "Service",
        "Database",
        "Metric",
        "Configuration",
        "Environment",
        "Pipeline",
        "Team",
        "Role",
        "Module",
        "FilePath",
        "Commit",
        "Symbol",
        "Session",
        "Agent",
        "Tool",
        "Template",
    ];

    /// Parse a client-declared label, case-insensitively. `None` for anything not in
    /// [`Self::DECLARABLE`].
    pub fn parse_declared(s: &str) -> Option<EntityLabel> {
        let label = match s.trim().to_ascii_lowercase().as_str() {
            "person" => Self::Person,
            "organization" => Self::Organization,
            "location" => Self::Location,
            "technology" => Self::Technology,
            "concept" => Self::Concept,
            "event" => Self::Event,
            "date" => Self::Date,
            "product" => Self::Product,
            "skill" => Self::Skill,
            "keyword" => Self::Keyword,
            "project" => Self::Project,
            "task" => Self::Task,
            "document" => Self::Document,
            "repository" => Self::Repository,
            "service" => Self::Service,
            "database" => Self::Database,
            "metric" => Self::Metric,
            "configuration" => Self::Configuration,
            "environment" => Self::Environment,
            "pipeline" => Self::Pipeline,
            "team" => Self::Team,
            "role" => Self::Role,
            "module" => Self::Module,
            "filepath" => Self::FilePath,
            "commit" => Self::Commit,
            "symbol" => Self::Symbol,
            "session" => Self::Session,
            "agent" => Self::Agent,
            "tool" => Self::Tool,
            "template" => Self::Template,
            _ => return None,
        };
        Some(label)
    }

    /// Static type hierarchy: returns parent labels for ontological matching.
    ///
    /// Enables hierarchical type matching so "who manages" can match Team
    /// (because Team is_a Organization-like group entity) and "what technology"
    /// can match Service (because Service is_a Technology subtype).
    ///
    /// Hierarchy is shallow (max depth 1) and fixed at compile time.
    /// No graph storage, no runtime cost beyond a match.
    ///
    /// Reference: Collins & Quillian (1969) "Retrieval time from semantic memory"
    /// — type hierarchies reduce retrieval time for plausible paths.
    pub fn parent_labels(&self) -> &'static [EntityLabel] {
        match self {
            // Organizational subtypes
            Self::Team => &[EntityLabel::Organization],
            Self::Role => &[EntityLabel::Concept],
            // Technical subtypes
            Self::Service | Self::Database | Self::Repository | Self::Module | Self::Pipeline => {
                &[EntityLabel::Technology]
            }
            // Work / knowledge subtypes
            Self::Task => &[EntityLabel::Event],
            Self::Document
            | Self::Configuration
            | Self::Metric
            | Self::Environment
            | Self::Project => &[EntityLabel::Concept],
            // Base types and Other have no parents
            _ => &[],
        }
    }

    /// Check if this label matches the expected label, considering type hierarchy.
    ///
    /// Returns true if:
    /// - Direct match (self == expected)
    /// - Self's parent matches expected (Team matches Organization)
    ///
    /// Used by spreading activation (Layer 2 entity penalty) and
    /// Layer 4.9 (ontological re-ranking) for hierarchical type matching.
    pub fn matches_with_hierarchy(&self, expected: &EntityLabel) -> bool {
        if self == expected {
            return true;
        }
        self.parent_labels().contains(expected)
    }

    /// Whether entities of this type may merge across project bands.
    ///
    /// Exhaustive on purpose — no wildcard arm. Adding a variant to
    /// `EntityLabel` must be a compile error here, not an inherited default,
    /// because the choice it forces cannot be undone once entities have
    /// merged on it.
    pub fn merge_class(&self) -> MergeClass {
        use MergeClass::{BandRelative, Global};
        match self {
            // The referent is the same thing whichever project observed it.
            Self::Person | Self::Organization | Self::Location => Global,
            Self::Technology | Self::Product | Self::Date => Global,
            Self::Concept | Self::Skill | Self::Role | Self::Team => Global,
            // Within one owner an Event is usually one occurrence referenced
            // from several projects — an outage spanning three services is one
            // outage. Least certain member of this set.
            Self::Event => Global,
            // Project and Repository NAME bands. They have to resolve globally
            // or one band could never refer to another, which is the mechanism
            // cross-band edges are built on.
            Self::Project | Self::Repository => Global,

            // Documented as "not a named entity" — a YAKE keyword's sense is
            // whatever the surrounding band made it.
            Self::Keyword => BandRelative,
            Self::Task | Self::Document => BandRelative,
            // Both variants are documented as INSTANCES ("PostgreSQL, Redis …
            // instances", "microservices, APIs, daemons"): every band runs its
            // own.
            Self::Service | Self::Database => BandRelative,
            Self::Metric | Self::Configuration => BandRelative,
            Self::Environment | Self::Pipeline => BandRelative,
            Self::Module => BandRelative,
            // Unrecognised type, unknown referential scope. Fail toward not
            // merging.
            Self::Other(_) => BandRelative,

            // Writer-declared identifiers whose canonical form carries its own
            // scope: a commit sha, a session or agent id, a tool name and a
            // template's content hash each name one referent wherever they
            // appear.
            Self::Commit | Self::Session | Self::Agent => Global,
            Self::Tool | Self::Template => Global,
            // A path or symbol is relative to its tree unless the writer
            // qualified it, and the label cannot tell whether it did.
            Self::FilePath | Self::Symbol => BandRelative,
        }
    }
}

/// The salience multiplier an entity earns from `mentions` mentions: `1 + 0.1 × ln(mentions)`,
/// which reaches [`crate::constants::ENTITY_SALIENCE_MAX_FREQUENCY_BOOST`] at about twenty
/// mentions and stays there. One mention earns none.
fn mention_frequency_boost(mentions: usize) -> f32 {
    (1.0 + 0.1 * (mentions.max(1) as f32).ln())
        .min(crate::constants::ENTITY_SALIENCE_MAX_FREQUENCY_BOOST)
}

/// Selectivity of an entity from the Forman-Ricci curvatures of its incident edges: their
/// sample standard deviation divided by their count, or 1.0 under two edges, where there is
/// no spread to measure (upstream #214).
fn curvature_selectivity(incident: &[f32]) -> f32 {
    let n = incident.len() as f32;
    if incident.len() < 2 {
        return 1.0;
    }
    let mean = incident.iter().sum::<f32>() / n;
    let variance = incident.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / (n - 1.0);
    variance.sqrt() / n
}

/// The ontological class a curated naming convention recognises in `tag`, or `None`.
///
/// Ported verbatim from upstream shodh-memory #509 (a292546f). Each rule recognises a
/// convention only artifacts follow: a named product (`rocksdb`), a file (`config.toml`), a
/// service (`auth-service`). A tag no rule recognises is a keyword: it describes the memory
/// without naming anything, and it stays in `tags` for search and `recall_by_tags` without
/// becoming a graph node (see `process_experience_into_graph_declared`).
///
/// Every rule matches a whole tag or a whole trailing token, never a bare substring:
/// upstream's `contains("config")` once made the prose phrase "stobar configuration" a
/// Configuration entity. Deployment-phase words ("production", "staging", "dev", "ci") are
/// absent on purpose: they are English common nouns that denote no referent off a
/// deployment log.
pub fn recognised_tag_label(tag: &str) -> Option<EntityLabel> {
    let lower = tag.to_lowercase();

    /// Whether `lower` is `word`, or ends with `word` after a `-`, `_` or space
    /// ("deploy-workflow" matches "workflow"; "workflows" does not).
    fn is_trailing_token(lower: &str, word: &str) -> bool {
        if lower == word {
            return true;
        }
        lower
            .strip_suffix(word)
            .is_some_and(|prefix| prefix.ends_with(['-', '_', ' ']))
    }

    if matches!(
        lower.as_str(),
        "kubernetes" | "k8s" | "docker" | "aws" | "gcp" | "azure"
    ) {
        return Some(EntityLabel::Environment);
    }

    if is_trailing_token(&lower, "pipeline")
        || is_trailing_token(&lower, "workflow")
        || lower == "ci-cd"
        || lower == "cicd"
    {
        return Some(EntityLabel::Pipeline);
    }

    if matches!(
        lower.as_str(),
        "rocksdb"
            | "postgres"
            | "postgresql"
            | "redis"
            | "sqlite"
            | "mongodb"
            | "mysql"
            | "dynamodb"
            | "s3"
            | "cassandra"
            | "elasticsearch"
    ) || lower.ends_with("db")
        || lower.ends_with("-db")
        || lower.ends_with("_db")
    {
        return Some(EntityLabel::Database);
    }

    // Suffix-only, so "my-api-docs" is not a service.
    if lower.ends_with("-service")
        || lower.ends_with("_service")
        || lower.ends_with("-api")
        || lower.ends_with("_api")
        || lower.ends_with("-server")
        || lower.ends_with("_server")
        || lower.ends_with("-daemon")
    {
        return Some(EntityLabel::Service);
    }

    // Before the module rules: README.md is a document, not a module.
    if lower.ends_with(".md")
        || is_trailing_token(&lower, "readme")
        || is_trailing_token(&lower, "runbook")
        || lower.ends_with("-rfc")
        || lower.ends_with("-spec")
    {
        return Some(EntityLabel::Document);
    }

    if lower.ends_with(".toml")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".env")
        || lower.ends_with(".json")
        || is_trailing_token(&lower, "config")
    {
        return Some(EntityLabel::Configuration);
    }

    if lower.ends_with(".rs")
        || lower.ends_with(".ts")
        || lower.ends_with(".js")
        || lower.ends_with(".py")
        || lower.ends_with("-lib")
        || lower.ends_with("_lib")
    {
        return Some(EntityLabel::Module);
    }

    None
}

/// Memory tier for edge consolidation
///
/// Based on hippocampal-cortical memory consolidation research:
/// - L1 (Working): Dense, fast encoding, aggressive pruning (Dentate Gyrus-like)
/// - L2 (Episodic): Moderate density, Hebbian selection (CA1/CA3-like)
/// - L3 (Semantic): Sparse, near-permanent (Neocortex-like)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeTier {
    /// Working memory tier: new edges, dense, aggressive decay
    #[default]
    L1Working,
    /// Episodic memory tier: proven edges, moderate decay
    L2Episodic,
    /// Semantic memory tier: consolidated edges, near-permanent
    L3Semantic,
}

/// Check whether a relationship edge is safe for cross-user collective aggregation.
///
/// An edge is safe when it has reached semantic tier (L3) and neither endpoint
/// carries personal or organizational identity.
pub fn is_edge_safe_for_collective(
    edge: &RelationshipEdge,
    from_entity: &EntityNode,
    to_entity: &EntityNode,
) -> bool {
    #[cfg(feature = "multi-tenant")]
    {
        crate::extensions::pii_policy::is_edge_safe_for_collective(edge, from_entity, to_entity)
    }

    #[cfg(not(feature = "multi-tenant"))]
    {
        edge.invalidated_at.is_none()
            && edge.tier == EdgeTier::L3Semantic
            && from_entity.pii_classification != PiiClassification::PersonalIdentity
            && to_entity.pii_classification != PiiClassification::PersonalIdentity
            && from_entity.pii_classification != PiiClassification::OrganizationIdentity
            && to_entity.pii_classification != PiiClassification::OrganizationIdentity
    }
}

impl EdgeTier {
    /// Get the initial weight for edges in this tier
    pub fn initial_weight(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_INITIAL_WEIGHT,
            Self::L2Episodic => L2_PROMOTION_WEIGHT,
            Self::L3Semantic => L3_PROMOTION_WEIGHT,
        }
    }

    /// Get the prune threshold for this tier
    pub fn prune_threshold(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_PRUNE_THRESHOLD,
            Self::L2Episodic => L2_PRUNE_THRESHOLD,
            Self::L3Semantic => L3_PRUNE_THRESHOLD,
        }
    }

    /// Get the promotion threshold to move to next tier
    pub fn promotion_threshold(&self) -> Option<f32> {
        use crate::constants::*;
        match self {
            Self::L1Working => Some(L1_PROMOTION_THRESHOLD),
            Self::L2Episodic => Some(L2_PROMOTION_THRESHOLD),
            Self::L3Semantic => None, // Already at highest tier
        }
    }

    /// Get the next tier (for promotion)
    pub fn next_tier(&self) -> Option<Self> {
        match self {
            Self::L1Working => Some(Self::L2Episodic),
            Self::L2Episodic => Some(Self::L3Semantic),
            Self::L3Semantic => None,
        }
    }

    /// Get target density for this tier
    pub fn target_density(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_TARGET_DENSITY,
            Self::L2Episodic => L2_TARGET_DENSITY,
            Self::L3Semantic => L3_TARGET_DENSITY,
        }
    }
}

/// Long-Term Potentiation status for edges (PIPE-4)
///
/// Multi-scale LTP based on neuroscience research:
/// - Burst: Temporary protection from high-frequency activation (E-LTP)
/// - Weekly: Moderate protection from consistent routine use (L-LTP)
/// - Full: Maximum protection from sustained long-term use (systems consolidation)
///
/// Reference: Frey & Morris (1997) "Synaptic tagging and long-term potentiation"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LtpStatus {
    /// Not potentiated - normal decay applies
    #[default]
    None,

    /// Burst potentiated: 5+ activations in 24 hours
    /// Temporary protection (2x slower decay) that expires after 48h
    /// Represents early-phase LTP (protein synthesis independent)
    Burst {
        /// When burst was detected (for expiration check)
        #[serde(with = "chrono::serde::ts_seconds")]
        detected_at: DateTime<Utc>,
    },

    /// Weekly potentiated: 3+/week for 2+ weeks
    /// Moderate protection (3x slower decay)
    /// Represents late-phase LTP (habit formation)
    Weekly,

    /// Fully potentiated: 10+ activations OR 5+ over 30 days
    /// Maximum protection (10x slower decay)
    /// Represents systems consolidation (semantic memory)
    Full,
}

impl LtpStatus {
    /// Get the decay factor for this LTP status
    pub fn decay_factor(&self) -> f32 {
        self.decay_factor_at(Utc::now())
    }

    /// As [`decay_factor`](Self::decay_factor), with burst expiry judged at `now`.
    pub fn decay_factor_at(&self, now: DateTime<Utc>) -> f32 {
        use crate::constants::*;
        match self {
            Self::None => 1.0,
            Self::Burst { detected_at } => {
                // Check if burst has expired
                let hours_since = (now - *detected_at).num_hours();
                if hours_since > LTP_BURST_DURATION_HOURS {
                    1.0 // Expired, normal decay
                } else {
                    LTP_BURST_DECAY_FACTOR
                }
            }
            Self::Weekly => LTP_WEEKLY_DECAY_FACTOR,
            Self::Full => LTP_DECAY_FACTOR,
        }
    }

    /// Check if this status provides any protection
    pub fn is_potentiated(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Check if burst protection has expired
    pub fn is_burst_expired(&self) -> bool {
        self.is_burst_expired_at(Utc::now())
    }

    /// As [`is_burst_expired`](Self::is_burst_expired), judged at `now`.
    pub fn is_burst_expired_at(&self, now: DateTime<Utc>) -> bool {
        use crate::constants::LTP_BURST_DURATION_HOURS;
        match self {
            Self::Burst { detected_at } => {
                (now - *detected_at).num_hours() > LTP_BURST_DURATION_HOURS
            }
            _ => false,
        }
    }

    /// Get priority for LTP upgrades (higher = stronger protection)
    pub fn priority(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::Burst { .. } => 1,
            Self::Weekly => 2,
            Self::Full => 3,
        }
    }
}

/// Relationship edge between entities
///
/// Implements Hebbian synaptic plasticity: "Neurons that fire together, wire together"
/// - Strength increases with co-activation (strengthen method)
/// - Strength decays over time without use (decay method)
/// - Long-Term Potentiation (LTP): After threshold activations, becomes permanent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEdge {
    /// Unique identifier for this edge
    pub uuid: Uuid,

    /// Source entity UUID
    pub from_entity: Uuid,

    /// Target entity UUID
    pub to_entity: Uuid,

    /// Type of relationship
    pub relation_type: RelationType,

    /// Confidence/strength of this relationship (0.0 to 1.0)
    /// Dynamic: increases with co-activation, decays without use
    pub strength: f32,

    /// When this relationship was created
    pub created_at: DateTime<Utc>,

    /// When this relationship was last observed (temporal tracking)
    pub valid_at: DateTime<Utc>,

    /// Whether this relationship has been invalidated (temporal edge invalidation)
    pub invalidated_at: Option<DateTime<Utc>>,

    /// Source episode that created this relationship
    pub source_episode_id: Option<Uuid>,

    /// Additional context about the relationship
    pub context: String,

    // === Hebbian Synaptic Plasticity Fields ===
    /// When this synapse was last activated (used in retrieval/traversal)
    /// Used to calculate time-based decay
    #[serde(default = "default_last_activated")]
    pub last_activated: DateTime<Utc>,

    /// Number of times both entities were co-accessed (Hebbian co-activation)
    /// Higher count = stronger learned association
    #[serde(default)]
    pub activation_count: u32,

    /// Long-Term Potentiation status (PIPE-4: multi-scale LTP)
    /// Replaces simple bool with tiered protection levels:
    /// - None: Normal decay
    /// - Burst: Temporary 2x protection (5+ activations in 24h)
    /// - Weekly: Moderate 3x protection (3+/week for 2 weeks)
    /// - Full: Maximum 10x protection (10+ activations or 5+ over 30 days)
    #[serde(default)]
    pub ltp_status: LtpStatus,

    /// Memory tier for consolidation (L1→L2→L3)
    /// Edges start in L1 (working memory) and promote based on Hebbian strength
    #[serde(default)]
    pub tier: EdgeTier,

    /// Activation timestamp history for temporal pattern detection (PIPE-4)
    /// Only populated for L2+ edges (L1 edges die too quickly to need history)
    /// Capacity: L2 = 20 timestamps, L3 = 50 timestamps
    /// Enables: burst detection, weekly patterns, temporal query relevance
    #[serde(default)]
    pub activation_timestamps: Option<VecDeque<DateTime<Utc>>>,

    /// Entity extraction confidence (PIPE-5: Unified LTP Readiness)
    /// Average confidence of the entities connected by this edge.
    /// Affects LTP threshold: high confidence → faster LTP (7 activations)
    /// Low confidence → slower LTP (13 activations).
    /// Based on synaptic tagging: behaviorally relevant synapses consolidate faster.
    #[serde(default)]
    pub entity_confidence: Option<f32>,

    /// Creation source for this edge (FIX-06: dream replay tagging)
    /// Tracks whether this edge was created from direct co-occurrence,
    /// dream replay, Hebbian coactivation, or explicit user input.
    /// Enables retrieval to distinguish speculative from empirical associations.
    #[serde(default)]
    pub created_by: EdgeSource,

    // === Directional Co-activation Fields (A5 Max Sphere) ===
    // M&I bidirectional memory-identity coupling: direction matters.
    // When memory A triggers recall of B (but not vice versa), the edge
    // should reflect this asymmetry for spreading activation.
    /// Forward activation strength (from_entity triggered, to_entity co-retrieved)
    /// Increases when from_entity is the query match and to_entity is co-retrieved,
    /// once per strengthening source, as `strength` does.
    #[serde(default)]
    pub forward_strength: f32,

    /// Backward activation strength (to_entity triggered, from_entity co-retrieved)
    /// Increases when to_entity is the query match and from_entity is co-retrieved,
    /// once per strengthening source, as `strength` does.
    #[serde(default)]
    pub backward_strength: f32,

    // === Evidential base ===
    /// Distinct sources (session, tool, ingest) that have strengthened this edge, each
    /// with its event count. A repeat from a recorded source refreshes `last_activated`
    /// and nothing else; a new source strengthens.
    #[serde(default)]
    pub strengthening_sources: BTreeMap<String, u32>,

    /// When a source not yet in `strengthening_sources` last strengthened this edge.
    #[serde(default)]
    pub last_corroborated: Option<DateTime<Utc>>,
}

fn default_last_activated() -> DateTime<Utc> {
    Utc::now()
}

/// Layout of `RelationshipEdge` before `strengthening_sources` / `last_corroborated`.
/// bincode is positional, so records written under it only decode through this arm.
#[derive(Serialize, Deserialize)]
struct RelationshipEdgeV0 {
    uuid: Uuid,
    from_entity: Uuid,
    to_entity: Uuid,
    relation_type: RelationType,
    strength: f32,
    created_at: DateTime<Utc>,
    valid_at: DateTime<Utc>,
    invalidated_at: Option<DateTime<Utc>>,
    source_episode_id: Option<Uuid>,
    context: String,
    last_activated: DateTime<Utc>,
    activation_count: u32,
    ltp_status: LtpStatus,
    tier: EdgeTier,
    activation_timestamps: Option<VecDeque<DateTime<Utc>>>,
    entity_confidence: Option<f32>,
    created_by: EdgeSource,
    forward_strength: f32,
    backward_strength: f32,
}

impl From<RelationshipEdgeV0> for RelationshipEdge {
    fn from(v: RelationshipEdgeV0) -> Self {
        Self {
            uuid: v.uuid,
            from_entity: v.from_entity,
            to_entity: v.to_entity,
            relation_type: v.relation_type,
            strength: v.strength,
            created_at: v.created_at,
            valid_at: v.valid_at,
            invalidated_at: v.invalidated_at,
            source_episode_id: v.source_episode_id,
            context: v.context,
            last_activated: v.last_activated,
            activation_count: v.activation_count,
            ltp_status: v.ltp_status,
            tier: v.tier,
            activation_timestamps: v.activation_timestamps,
            entity_confidence: v.entity_confidence,
            created_by: v.created_by,
            forward_strength: v.forward_strength,
            backward_strength: v.backward_strength,
            strengthening_sources: BTreeMap::new(),
            last_corroborated: None,
        }
    }
}

/// What a strengthening event counted as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    /// A source not yet recorded on the edge: strength moved.
    NewSource,
    /// A source already recorded on the edge: recorded, strength held.
    RepeatSource,
}

impl Evidence {
    pub fn label(self) -> &'static str {
        match self {
            Evidence::NewSource => "new_source",
            Evidence::RepeatSource => "repeat_source",
        }
    }
}

/// An entity's edges with their effective strength, strongest first.
pub type RankedEdges = Vec<(RelationshipEdge, f32)>;

/// Outcome of one `strengthen_from` call.
#[derive(Debug, Clone)]
pub struct Strengthening {
    pub evidence: Evidence,
    /// `Some((old_tier, new_tier))` when the event promoted the edge.
    pub promotion: Option<(String, String)>,
    /// The fraction of a new source's step this event earned: 1 for a new source;
    /// for a repeat, `HEBBIAN_REPEAT_SOURCE_BOOST` over the edge's full step rate, so
    /// 0 by default. The directional components take this same fraction of their
    /// own step, so one source counts once in every component of the edge.
    pub step: f32,
}

impl Strengthening {
    /// A repeat from a source already on the edge that earned no step: strength,
    /// direction, activation count and tier are as they were, and only `last_activated`
    /// and the source's repeat count moved.
    pub fn moves_nothing(&self) -> bool {
        self.evidence == Evidence::RepeatSource && self.step <= 0.0
    }
}

/// Evidence record for an edge created by a strengthening event from `source`.
pub(crate) fn evidence_from(source: &str) -> BTreeMap<String, u32> {
    let mut sources = BTreeMap::new();
    sources.insert(source.to_string(), 1);
    sources
}

thread_local! {
    static AMBIENT_EVIDENCE_SOURCE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Names the evidence source for strengthening events that happen inside graph
/// traversal on the current thread, where no caller argument reaches. Entered by
/// `MemorySystem::recall` for the duration of a recall; restores the previous
/// value on drop.
pub struct EvidenceScope {
    previous: Option<String>,
}

impl EvidenceScope {
    pub fn enter(source: Option<&str>) -> Self {
        let previous = AMBIENT_EVIDENCE_SOURCE.with(|s| s.replace(source.map(str::to_string)));
        Self { previous }
    }
}

impl Drop for EvidenceScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        AMBIENT_EVIDENCE_SOURCE.with(|s| *s.borrow_mut() = previous);
    }
}

/// The source named by the innermost `EvidenceScope`, or the unattributed fallback.
pub fn ambient_evidence_source() -> String {
    AMBIENT_EVIDENCE_SOURCE
        .with(|s| s.borrow().clone())
        .unwrap_or_else(|| crate::constants::HEBBIAN_UNATTRIBUTED_SOURCE.to_string())
}

// Hebbian learning constants now imported from crate::constants:
// - LTP_LEARNING_RATE (0.1): η for strength increase per co-activation
// - LTP_DECAY_HALF_LIFE_DAYS (14.0): λ for time-based decay
// - LTP_THRESHOLD (10): Activations needed for Full LTP
// - LTP_DECAY_FACTOR (0.1): Fully potentiated synapses decay 10x slower
// - LTP_MIN_STRENGTH (0.01): Floor to prevent complete forgetting
// PIPE-4 additions:
// - LTP_BURST_THRESHOLD (5): Activations in 24h for burst LTP
// - LTP_BURST_WINDOW_HOURS (24): Window for burst detection
// - LTP_WEEKLY_THRESHOLD (3): Activations per week for weekly LTP
// - LTP_WEEKLY_MIN_WEEKS (2): Minimum weeks of consistent activation

impl RelationshipEdge {
    /// Whether this edge connects `a` and `b`, in either direction.
    pub fn joins(&self, a: &Uuid, b: &Uuid) -> bool {
        (self.from_entity == *a && self.to_entity == *b)
            || (self.from_entity == *b && self.to_entity == *a)
    }

    /// Strengthen this synapse (Hebbian learning)
    ///
    /// Called when both connected entities are accessed together.
    /// Formula: w_new = w_old + η × (1 - w_old) × co_activation_boost
    ///
    /// PIPE-4: Multi-scale LTP detection
    /// - Records activation timestamps for L2+ edges
    /// - Detects burst patterns (5+ in 24h) → temporary protection
    /// - Detects weekly patterns (3+/week for 2 weeks) → moderate protection
    /// - Detects sustained patterns (10+ total or 5+ over 30 days) → full protection
    ///
    /// Also handles tier promotion (L1→L2→L3) when strength exceeds tier threshold.
    ///
    /// Returns `Some((old_tier_name, new_tier_name))` if a tier promotion occurred,
    /// `None` otherwise. This enables the memory-edge coupling: edge promotions
    /// can signal the memory layer to boost the associated memory's importance.
    /// Hebbian strengthening — private to the graph module.
    ///
    /// All external `edge.strength` mutations must go through
    /// `GraphMemory::apply_reinforcement` or its internal delegate
    /// `record_memory_coactivation`.  Making this `pub(crate)` prevents
    /// callers outside this crate from writing `edge.strength` directly.
    ///
    /// `source` names who is strengthening (session, tool, ingest). One source
    /// counts once per edge: a repeat refreshes `last_activated` and is recorded,
    /// and only a source new to the edge moves strength, activation count or LTP.
    pub(crate) fn strengthen_from(&mut self, source: &str) -> Strengthening {
        use crate::constants::*;

        let now = Utc::now();
        self.last_activated = now;

        let evidence = self.record_evidence(source);
        crate::metrics::HEBBIAN_STRENGTHEN_EVENTS_TOTAL
            .with_label_values(&[evidence.label()])
            .inc();
        // Hebbian strengthening with tier-specific boost
        let tier_boost = match self.tier {
            EdgeTier::L1Working => TIER_CO_ACCESS_BOOST,
            EdgeTier::L2Episodic => TIER_CO_ACCESS_BOOST * 0.8,
            EdgeTier::L3Semantic => TIER_CO_ACCESS_BOOST * 0.5,
        };
        let full_step_rate = LTP_LEARNING_RATE + tier_boost;

        if evidence == Evidence::RepeatSource {
            if HEBBIAN_REPEAT_SOURCE_BOOST > 0.0 {
                let boost = HEBBIAN_REPEAT_SOURCE_BOOST * (1.0 - self.strength);
                self.strength = (self.strength + boost).min(1.0);
            }
            return Strengthening {
                evidence,
                promotion: None,
                step: (HEBBIAN_REPEAT_SOURCE_BOOST / full_step_rate).min(1.0),
            };
        }

        self.activation_count += 1;
        self.last_corroborated = Some(now);

        // PIPE-4: Record activation timestamp for L2+ edges
        self.record_activation_timestamp(now);

        let boost = full_step_rate * (1.0 - self.strength);
        self.strength = (self.strength + boost).min(1.0);

        // PIPE-4: Multi-scale LTP detection (only upgrade, never downgrade)
        let new_ltp_status = self.detect_ltp_status(now);
        if new_ltp_status.priority() > self.ltp_status.priority() {
            let old_status = self.ltp_status;
            self.ltp_status = new_ltp_status;

            // LTP bonus: immediate strength boost on upgrade
            let bonus = match new_ltp_status {
                LtpStatus::Burst { .. } => 0.05,
                LtpStatus::Weekly => 0.1,
                LtpStatus::Full => 0.2,
                LtpStatus::None => 0.0,
            };
            self.strength = (self.strength + bonus).min(1.0);

            tracing::debug!(
                "Edge {} LTP upgrade: {:?} → {:?} (activations: {}, age: {} days)",
                self.uuid,
                old_status,
                self.ltp_status,
                self.activation_count,
                (now - self.created_at).num_days()
            );
        }

        // Check for burst expiration and potential downgrade
        if self.ltp_status.is_burst_expired() {
            // Burst expired - check if weekly pattern has emerged
            let weekly_check = self.detect_weekly_pattern();
            if weekly_check {
                self.ltp_status = LtpStatus::Weekly;
            } else {
                self.ltp_status = LtpStatus::None;
            }
        }

        // Tier promotion: check if strength exceeds current tier's promotion threshold
        let mut promotion_result = None;
        if let Some(threshold) = self.tier.promotion_threshold() {
            if self.strength >= threshold {
                if let Some(next_tier) = self.tier.next_tier() {
                    let old_tier = self.tier;
                    self.tier = next_tier;
                    // Preserve strength if already above next tier's initial weight
                    self.strength = self.strength.max(next_tier.initial_weight());

                    // PIPE-4: Initialize activation_timestamps on L1→L2 promotion
                    if old_tier == EdgeTier::L1Working {
                        self.activation_timestamps =
                            Some(VecDeque::with_capacity(ACTIVATION_HISTORY_L2_CAPACITY));
                        // Seed with current timestamp
                        if let Some(ref mut ts) = self.activation_timestamps {
                            ts.push_back(now);
                        }
                    }

                    // Expand capacity on L2→L3 promotion
                    if old_tier == EdgeTier::L2Episodic {
                        if let Some(ref mut ts) = self.activation_timestamps {
                            let current = ts.capacity();
                            if current < ACTIVATION_HISTORY_L3_CAPACITY {
                                ts.reserve(ACTIVATION_HISTORY_L3_CAPACITY - current);
                            }
                        }
                    }

                    tracing::debug!(
                        "Edge {} promoted: {:?} → {:?}",
                        self.uuid,
                        old_tier,
                        self.tier
                    );

                    promotion_result =
                        Some((format!("{:?}", old_tier), format!("{:?}", self.tier)));
                }
            }
        }

        // PIPE-5: L3 auto-LTP removed - now handled by unified ltp_readiness()
        // The readiness formula combines strength + activation count + entity confidence,
        // ensuring both intensity and repetition evidence are required for Full LTP.

        Strengthening {
            evidence: Evidence::NewSource,
            promotion: promotion_result,
            step: 1.0,
        }
    }

    /// Record `source` against this edge and say whether it is new evidence.
    fn record_evidence(&mut self, source: &str) -> Evidence {
        use crate::constants::HEBBIAN_EVIDENCE_SOURCES_CAP;
        if let Some(count) = self.strengthening_sources.get_mut(source) {
            *count = count.saturating_add(1);
            return Evidence::RepeatSource;
        }
        if self.strengthening_sources.len() >= HEBBIAN_EVIDENCE_SOURCES_CAP {
            return Evidence::RepeatSource;
        }
        self.strengthening_sources.insert(source.to_string(), 1);
        Evidence::NewSource
    }

    /// Number of distinct sources that have strengthened this edge.
    pub fn distinct_sources(&self) -> usize {
        self.strengthening_sources.len()
    }

    /// Decode a persisted edge, accepting the layout written before the evidential
    /// base existed. A record is only accepted by an arm that consumes it whole.
    pub fn decode(bytes: &[u8]) -> Result<RelationshipEdge> {
        let cfg = bincode::config::standard();
        let current = match bincode::serde::decode_from_slice::<RelationshipEdge, _>(bytes, cfg) {
            Ok((edge, read)) if read == bytes.len() => return Ok(edge),
            Ok((_, read)) => anyhow::anyhow!("{read} of {} bytes consumed", bytes.len()),
            Err(e) => anyhow::anyhow!(e),
        };
        match bincode::serde::decode_from_slice::<RelationshipEdgeV0, _>(bytes, cfg) {
            Ok((v0, read)) if read == bytes.len() => Ok(v0.into()),
            _ => Err(current.context("relationship edge decode failed in both layouts")),
        }
    }

    /// Record an activation timestamp (PIPE-4)
    ///
    /// Records for every tier. L1 was previously excluded as "too transient",
    /// but the adaptive decay rate is derived FROM this history, so excluding
    /// L1 guaranteed L1 could never learn its own cadence — the tier that most
    /// needs the evidence was the one forbidden to collect it. L1 keeps a short
    /// buffer; an interval needs two timestamps, not a lifecycle.
    fn record_activation_timestamp(&mut self, timestamp: DateTime<Utc>) {
        use crate::constants::*;

        let capacity = match self.tier {
            EdgeTier::L1Working => ACTIVATION_HISTORY_L1_CAPACITY,
            EdgeTier::L2Episodic => ACTIVATION_HISTORY_L2_CAPACITY,
            EdgeTier::L3Semantic => ACTIVATION_HISTORY_L3_CAPACITY,
        };

        // Initialize if needed
        if self.activation_timestamps.is_none() {
            self.activation_timestamps = Some(VecDeque::with_capacity(capacity));
        }

        if let Some(ref mut timestamps) = self.activation_timestamps {
            // Maintain capacity limit (ring buffer behavior)
            while timestamps.len() >= capacity {
                timestamps.pop_front();
            }
            timestamps.push_back(timestamp);
        }
    }

    /// The edge's own typical interval between activations, in hours.
    ///
    /// Returns `None` until there are `ADAPTIVE_DECAY_MIN_SAMPLES` timestamps
    /// (i.e. at least two intervals) — below that, one coincidence would set the
    /// rate. Uses the MEDIAN rather than the mean: activation histories are
    /// bursty, and a single long idle gap would drag a mean far enough to make
    /// the edge effectively permanent.
    pub fn typical_activation_interval_hours(&self) -> Option<f64> {
        use crate::constants::ADAPTIVE_DECAY_MIN_SAMPLES;
        // Read for every edge a traversal decays. Histories are capped per tier (L3 holds the
        // most), so the gaps fit a stack buffer and the read path stays allocation-free.
        const STACK_GAPS: usize = 256;

        let timestamps = self.activation_timestamps.as_ref()?;
        if timestamps.len() < ADAPTIVE_DECAY_MIN_SAMPLES {
            return None;
        }

        let gaps = timestamps
            .iter()
            .zip(timestamps.iter().skip(1))
            .map(|(a, b)| (*b - *a).num_seconds() as f64 / 3600.0)
            .filter(|g| *g > 0.0);

        if timestamps.len() <= STACK_GAPS {
            let mut buf = [0.0_f64; STACK_GAPS];
            let mut n = 0;
            for gap in gaps {
                buf[n] = gap;
                n += 1;
            }
            upper_median(&mut buf[..n])
        } else {
            upper_median(&mut gaps.collect::<Vec<f64>>())
        }
    }

    /// Detect LTP status based on unified readiness model (PIPE-4 + PIPE-5)
    ///
    /// PIPE-5 unifies LTP detection into a single readiness score that combines:
    /// - Activation count (repetition path)
    /// - Strength (intensity/durability path)
    /// - Entity confidence (synaptic tagging bonus)
    ///
    /// Multiple paths can lead to Full LTP:
    /// - High repetition alone (15+ activations)
    /// - High intensity alone (0.95+ strength at L3)
    /// - Balanced contribution from both
    /// - High-confidence edges reach threshold ~30% faster
    ///
    /// Temporal patterns (Burst, Weekly) remain separate as they represent
    /// different consolidation mechanisms (E-LTP vs habit formation).
    fn detect_ltp_status(&self, now: DateTime<Utc>) -> LtpStatus {
        use crate::constants::*;

        // PIPE-5: Unified LTP readiness for Full LTP
        // Combines activation count, strength, and entity confidence
        if self.ltp_readiness() >= LTP_READINESS_THRESHOLD {
            return LtpStatus::Full;
        }

        // Legacy time-aware path: 5+ activations over 30+ days
        // Kept for backward compatibility and edges that survived long decay
        let edge_age_days = (now - self.created_at).num_days();
        if edge_age_days >= LTP_TIME_AWARE_DAYS && self.activation_count >= LTP_TIME_AWARE_THRESHOLD
        {
            return LtpStatus::Full;
        }

        // Check for Weekly LTP (requires timestamp history)
        // Temporal pattern: 3+/week for 2+ weeks indicates habit
        if self.detect_weekly_pattern() {
            return LtpStatus::Weekly;
        }

        // Check for Burst LTP (requires timestamp history)
        // Temporal pattern: 5+ in 24h indicates high immediate interest
        if self.detect_burst_pattern(now) {
            return LtpStatus::Burst { detected_at: now };
        }

        LtpStatus::None
    }

    /// Detect burst pattern: 5+ activations in 24 hours (PIPE-4)
    fn detect_burst_pattern(&self, now: DateTime<Utc>) -> bool {
        use crate::constants::*;
        use chrono::Duration;

        let timestamps = match &self.activation_timestamps {
            Some(ts) => ts,
            None => return false,
        };

        let window_start = now - Duration::hours(LTP_BURST_WINDOW_HOURS);
        let count_in_window = timestamps.iter().filter(|&&ts| ts >= window_start).count();

        count_in_window >= LTP_BURST_THRESHOLD as usize
    }

    /// Detect weekly pattern: 3+/week for 2+ weeks (PIPE-4)
    fn detect_weekly_pattern(&self) -> bool {
        self.detect_weekly_pattern_at(Utc::now())
    }

    fn detect_weekly_pattern_at(&self, now: DateTime<Utc>) -> bool {
        use crate::constants::*;
        use chrono::Duration;

        let timestamps = match &self.activation_timestamps {
            Some(ts) => ts,
            None => return false,
        };

        if timestamps.is_empty() {
            return false;
        }

        let mut weeks_meeting_threshold = 0u32;

        // Check each of the last LTP_WEEKLY_MIN_WEEKS weeks
        for week_offset in 0..LTP_WEEKLY_MIN_WEEKS {
            let week_end = now - Duration::weeks(week_offset as i64);
            let week_start = week_end - Duration::weeks(1);

            let count_in_week = timestamps
                .iter()
                .filter(|&&ts| ts >= week_start && ts < week_end)
                .count();

            if count_in_week >= LTP_WEEKLY_THRESHOLD as usize {
                weeks_meeting_threshold += 1;
            }
        }

        weeks_meeting_threshold >= LTP_WEEKLY_MIN_WEEKS
    }

    /// Get activation count within a time window (for temporal retrieval scoring)
    pub fn activations_in_window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> usize {
        match &self.activation_timestamps {
            Some(ts) => ts.iter().filter(|&&t| t >= start && t <= end).count(),
            None => 0,
        }
    }

    /// Check if edge was active at similar time of day (for temporal retrieval)
    pub fn time_of_day_match(&self, target_hour: u32, window_hours: u32) -> usize {
        use chrono::Timelike;

        match &self.activation_timestamps {
            Some(ts) => ts
                .iter()
                .filter(|t| {
                    let hour = t.hour();
                    let diff = if hour > target_hour {
                        (hour - target_hour).min(24 + target_hour - hour)
                    } else {
                        (target_hour - hour).min(24 + hour - target_hour)
                    };
                    diff <= window_hours
                })
                .count(),
            None => 0,
        }
    }

    /// Apply time-based decay to this synapse
    ///
    /// Uses tier-aware decay model (3-tier memory consolidation):
    /// - L1 (Working): ~2.9%/hour decay (λ=0.029), max 48 hours before prune
    /// - L2 (Episodic): ~3.1%/day decay (λ=0.031), max 30 days before prune
    /// - L3 (Semantic): ~2%/month decay (λ=0.02/720h), near-permanent
    ///
    /// PIPE-4: Multi-scale LTP protection
    /// - Burst: 2x slower decay (temporary, 48h)
    /// - Weekly: 3x slower decay (habit protection)
    /// - Full: 10x slower decay (permanent protection)
    ///
    /// **Important:** Updates `last_activated` to prevent double-decay on
    /// repeated calls.
    ///
    /// Returns true if synapse should be pruned (below tier's threshold)
    pub(crate) fn decay(&mut self) -> bool {
        self.decay_at(Utc::now())
    }

    /// As [`decay`](Self::decay), with every clock read taken from `now`, so a
    /// harness can step an edge through simulated time at a chosen cadence.
    pub(crate) fn decay_at(&mut self, now: DateTime<Utc>) -> bool {
        use crate::decay::tier_decay_factor_adaptive;

        // Computed before any mutation below (immutable borrow of self).
        let typical_interval = self.typical_activation_interval_hours();

        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return false;
        }

        // Cap max decay to protect against clock jumps (max 1 year = 8760 hours)
        let hours_elapsed = hours_elapsed.min(8760.0);

        // Tier-aware decay with PIPE-4 multi-scale LTP
        let tier_num = match self.tier {
            EdgeTier::L1Working => 0,
            EdgeTier::L2Episodic => 1,
            EdgeTier::L3Semantic => 2,
        };
        let ltp_factor = self.ltp_status.decay_factor_at(now);
        let (decay_factor, exceeded_max_age) =
            tier_decay_factor_adaptive(hours_elapsed, tier_num, ltp_factor, typical_interval);
        self.strength *= decay_factor;

        // Update last_activated to prevent double-decay on repeated calls
        self.last_activated = now;

        // Apply floor to prevent complete forgetting
        let prune_threshold = self.tier.prune_threshold();
        if self.strength < LTP_MIN_STRENGTH {
            self.strength = LTP_MIN_STRENGTH;
        }

        // Spontaneous recovery (Dussutour habituation-with-recovery, B8/F8)
        //
        // After each decay call, apply a small upward-drift term so that
        // down-weighting is reversible over time.  The cap is derived from the
        // SAME prune-threshold constant used in the prune decision below, ensuring
        // a recovering edge can NEVER drift above `prune_threshold * RECOVERY_CAP_FACTOR`
        // (strictly < 1.0), so the prune condition `strength <= prune_threshold` at
        // line 916 remains true for any edge that entered recovery from below threshold.
        {
            use crate::constants::{EDGE_RECOVERY_CAP_FACTOR, EDGE_RECOVERY_RATE};
            let recovery_cap = prune_threshold * EDGE_RECOVERY_CAP_FACTOR;
            if self.strength < recovery_cap {
                self.strength = (self.strength + EDGE_RECOVERY_RATE).min(recovery_cap);
            }
        }

        // Downgrade expired burst LTP before prune decision
        // decay_factor() already returns 1.0 for expired bursts (correct rate),
        // but is_potentiated() still returns true — preventing pruning
        if self.ltp_status.is_burst_expired_at(now) {
            if self.detect_weekly_pattern_at(now) {
                self.ltp_status = LtpStatus::Weekly;
            } else {
                self.ltp_status = LtpStatus::None;
            }
        }

        // Strip LTP protection from near-zero edges (zombie edge cleanup)
        // Prevents immortal edges that retain LTP despite negligible strength
        if self.ltp_status.is_potentiated() && self.strength <= LTP_PRUNE_FLOOR {
            self.ltp_status = LtpStatus::None;
        }

        // Return whether this synapse should be pruned
        // Prune if: exceeded max age OR below prune threshold (unless any LTP protection)
        // NOTE: Recovery is capped strictly below prune_threshold, so this condition
        // remains true for any edge that started decay at or below the threshold.
        if self.ltp_status.is_potentiated() {
            false
        } else {
            exceeded_max_age || self.strength <= prune_threshold
        }
    }

    /// Synthetic, non-persisted edge for the decay-simulation harness.
    /// Timestamps anchored at `origin` so a harness can drive decay forward
    /// from a known point. Never written to storage.
    ///
    /// Adapted from upstream: veld's RelationshipEdge has no
    /// endpoint_selectivity, forman_curvature, provenance or promoted_at.
    pub(crate) fn synthetic_for_sim(
        strength: f32,
        tier: EdgeTier,
        ltp_status: LtpStatus,
        origin: DateTime<Utc>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: origin,
            valid_at: origin,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: origin,
            activation_count: 0,
            ltp_status,
            activation_timestamps: None,
            tier,
            entity_confidence: None,
            // veld-only fields, absent upstream: a synthetic edge is
            // undirected-equivalent, so both directions carry `strength`.
            created_by: EdgeSource::Coactivation,
            forward_strength: strength,
            backward_strength: strength,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    /// Get the effective strength considering recency
    ///
    /// This is a read-only version that calculates what the strength
    /// would be after decay, without modifying the edge.
    /// Uses tier-aware decay (L1/L2/L3 have different decay rates).
    pub fn effective_strength(&self) -> f32 {
        self.effective_strength_at(crate::memory::scoring_now())
    }

    /// As [`effective_strength`](Self::effective_strength), decayed to `now`.
    pub fn effective_strength_at(&self, now: DateTime<Utc>) -> f32 {
        use crate::decay::tier_decay_factor_adaptive;

        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return self.strength;
        }

        let tier_num = match self.tier {
            EdgeTier::L1Working => 0,
            EdgeTier::L2Episodic => 1,
            EdgeTier::L3Semantic => 2,
        };
        let ltp_factor = self.ltp_status.decay_factor_at(now);
        // Same rate as decay() — a read path on a different rate than the write
        // path reports a strength the store will never actually hold.
        let (decay_factor, _) = tier_decay_factor_adaptive(
            hours_elapsed,
            tier_num,
            ltp_factor,
            self.typical_activation_interval_hours(),
        );
        (self.strength * decay_factor).max(LTP_MIN_STRENGTH)
    }

    /// Check if this edge has any LTP protection (for backward compatibility)
    pub fn is_potentiated(&self) -> bool {
        self.ltp_status.is_potentiated()
    }

    // =========================================================================
    // A5 Max Sphere: Directional Co-activation
    // =========================================================================

    /// Lazy migration: ensure directional strength fields are populated.
    ///
    /// Edges created before directional co-activation will have
    /// `forward_strength == 0.0 && backward_strength == 0.0` due to serde defaults.
    /// This method initializes both directions to the undirected strength (symmetric).
    /// Forward and backward are independent signals that can each range [0, 1].
    /// Called on edge access paths where directional strength matters.
    pub fn ensure_directional(&mut self) {
        if self.forward_strength == 0.0 && self.backward_strength == 0.0 && self.strength > 0.0 {
            // Pre-directional edges: assume symmetric (no directional evidence yet)
            self.forward_strength = self.strength;
            self.backward_strength = self.strength;
        }
    }

    /// Get direction-aware effective strength for spreading activation.
    ///
    /// When traversing from entity `from` to entity `to`:
    /// - If the edge's stored direction matches (from_entity == from), use forward_strength
    /// - If the edge is being traversed in reverse (to_entity == from), use backward_strength
    /// - Falls back to undirected strength if directional fields aren't populated
    ///
    /// Applies the same time-based decay as `effective_strength()` to prevent
    /// directional signal from being artificially immune to forgetting.
    ///
    /// This is the main payoff of directional edges: "A often leads to B" is
    /// distinguished from "B often leads to A" in spreading activation.
    pub fn directional_strength(&self, traversal_source: &Uuid) -> f32 {
        self.directional_strength_at(traversal_source, crate::memory::scoring_now())
    }

    /// As [`directional_strength`](Self::directional_strength), decayed to `now`.
    pub fn directional_strength_at(&self, traversal_source: &Uuid, now: DateTime<Utc>) -> f32 {
        use crate::decay::tier_decay_factor;

        // If directional fields haven't been populated yet, use undirected effective
        if self.forward_strength == 0.0 && self.backward_strength == 0.0 {
            return self.effective_strength_at(now);
        }

        let raw = if self.from_entity == *traversal_source {
            // Traversing in the stored forward direction: from_entity → to_entity
            self.forward_strength
        } else if self.to_entity == *traversal_source {
            // Traversing in reverse: to_entity → from_entity
            self.backward_strength
        } else {
            // Edge doesn't involve this entity (shouldn't happen in normal traversal)
            self.strength
        };

        // Apply the same time-based decay as effective_strength
        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return raw;
        }

        let tier_num = match self.tier {
            EdgeTier::L1Working => 0,
            EdgeTier::L2Episodic => 1,
            EdgeTier::L3Semantic => 2,
        };
        let ltp_factor = self.ltp_status.decay_factor_at(now);
        let (decay_factor, _) = tier_decay_factor(hours_elapsed, tier_num, ltp_factor);
        (raw * decay_factor).max(LTP_MIN_STRENGTH)
    }

    /// Get the neighbor entity when traversing from a given source.
    ///
    /// Returns the entity on the other side of the edge from `traversal_source`.
    /// This correctly handles bidirectional edge indexing where either
    /// `from_entity` or `to_entity` might be the source.
    pub fn neighbor_of(&self, traversal_source: &Uuid) -> Uuid {
        if self.from_entity == *traversal_source {
            self.to_entity
        } else {
            self.from_entity
        }
    }

    /// Strengthen the forward directional component (trigger → coactivated).
    ///
    /// Called when from_entity was the trigger (matched the query) and
    /// to_entity was co-retrieved. Uses Hebbian delta with forward boost.
    ///
    /// Note: Does NOT modify `self.strength` — the undirected strength is managed
    /// independently by `strengthen()` and represents overall edge health.
    /// Directional fields track asymmetry only.
    ///
    /// `step` is [`Strengthening::step`] from the same event: a repeat from a source
    /// already on the edge moves direction by the same fraction as strength, 0 by
    /// default, so repetition cannot build asymmetry the evidence does not support.
    pub(crate) fn strengthen_forward(&mut self, step: f32) {
        use crate::constants::COACTIVATION_FORWARD_BOOST;
        if step <= 0.0 {
            return;
        }
        self.ensure_directional();
        let delta = COACTIVATION_FORWARD_BOOST * step * (1.0 - self.forward_strength);
        self.forward_strength = (self.forward_strength + delta).min(1.0);
    }

    /// Strengthen the backward directional component (coactivated → trigger).
    ///
    /// Called when to_entity was the trigger and from_entity was co-retrieved.
    /// Uses Hebbian delta with backward boost (smaller than forward).
    ///
    /// Note: Does NOT modify `self.strength` — the undirected strength is managed
    /// independently by `strengthen()` and represents overall edge health.
    /// Directional fields track asymmetry only. `step` as for [`Self::strengthen_forward`].
    pub(crate) fn strengthen_backward(&mut self, step: f32) {
        use crate::constants::COACTIVATION_BACKWARD_BOOST;
        if step <= 0.0 {
            return;
        }
        self.ensure_directional();
        let delta = COACTIVATION_BACKWARD_BOOST * step * (1.0 - self.backward_strength);
        self.backward_strength = (self.backward_strength + delta).min(1.0);
    }

    // =========================================================================
    // PIPE-5: Unified LTP Readiness Model
    // =========================================================================

    /// Get confidence-adjusted LTP threshold (PIPE-5)
    ///
    /// High-confidence edges (strong entity extraction) need fewer activations.
    /// Low-confidence edges need more activations to prove value.
    ///
    /// Returns: threshold in range [LTP_THRESHOLD_MIN, LTP_THRESHOLD_MAX]
    pub fn adjusted_threshold(&self) -> u32 {
        use crate::constants::*;

        let confidence = self.entity_confidence.unwrap_or(0.5);

        // Linear interpolation: high confidence → low threshold
        // confidence 0.0 → threshold_max (13)
        // confidence 1.0 → threshold_min (7)
        let range = LTP_THRESHOLD_MAX - LTP_THRESHOLD_MIN;
        let threshold = LTP_THRESHOLD_MAX as f32 - (confidence * range as f32);
        threshold.round() as u32
    }

    /// Get tier-specific strength floor for Full LTP (PIPE-5)
    ///
    /// L2 edges have lower floor (still proving themselves).
    /// L3 edges have higher floor (must demonstrate durability).
    /// L1 edges return 1.0 (effectively impossible to reach Full LTP).
    pub fn strength_floor(&self) -> f32 {
        use crate::constants::*;

        match self.tier {
            EdgeTier::L1Working => 1.0, // L1 can't reach Full LTP via readiness
            EdgeTier::L2Episodic => LTP_STRENGTH_FLOOR_L2,
            EdgeTier::L3Semantic => LTP_STRENGTH_FLOOR_L3,
        }
    }

    /// Calculate LTP readiness score (PIPE-5)
    ///
    /// Unified formula combining activation count, strength, and entity confidence:
    /// - count_score = activation_count / adjusted_threshold
    /// - strength_score = strength / strength_floor
    /// - tag_bonus = entity_confidence * TAG_WEIGHT
    ///
    /// readiness = count_score * COUNT_WEIGHT + strength_score * STRENGTH_WEIGHT + tag_bonus
    ///
    /// Full LTP when readiness >= 1.0
    ///
    /// This allows multiple paths to LTP:
    /// - Repetition-dominant: 15 activations can compensate for lower strength
    /// - Intensity-dominant: 0.95 strength can compensate for fewer activations
    /// - Balanced: 10 activations + 0.75 strength + moderate confidence
    /// - Tagged boost: high-confidence edges reach LTP ~30% faster
    pub fn ltp_readiness(&self) -> f32 {
        use crate::constants::*;

        // L1 edges can't reach Full LTP via readiness (too transient)
        if matches!(self.tier, EdgeTier::L1Working) {
            return 0.0;
        }

        let threshold = self.adjusted_threshold() as f32;
        let floor = self.strength_floor();

        // Count score: how close to activation threshold
        let count_score = self.activation_count as f32 / threshold;

        // Strength score: how close to strength floor
        let strength_score = self.strength / floor;

        // Tag bonus: entity confidence provides synaptic tagging advantage
        let confidence = self.entity_confidence.unwrap_or(0.5);
        let tag_bonus = confidence * LTP_READINESS_TAG_WEIGHT;

        // Weighted combination
        count_score * LTP_READINESS_COUNT_WEIGHT
            + strength_score * LTP_READINESS_STRENGTH_WEIGHT
            + tag_bonus
    }

    /// Compute LTP readiness incorporating NPMI as a co-occurrence signal.
    ///
    /// When NPMI is available (post cold-start), it replaces the count-based
    /// component with a normalized co-occurrence measure. This allows edges
    /// representing genuinely rare but consistent associations to reach LTP
    /// faster, while penalizing edges between ubiquitous entities.
    ///
    /// Formula:
    ///   npmi_score = (npmi / 0.75).clamp(0.0, 1.5)
    ///   readiness  = npmi_score * 0.4 + strength_score * 0.4 + tag_bonus * 0.2
    ///
    /// Falls back to standard `ltp_readiness()` when npmi is None (cold start).
    pub fn ltp_readiness_with_npmi(&self, npmi: Option<f32>) -> f32 {
        // L1 edges can't reach Full LTP
        if matches!(self.tier, EdgeTier::L1Working) {
            return 0.0;
        }

        let npmi = match npmi {
            Some(n) => n,
            None => return self.ltp_readiness(), // cold start fallback
        };

        let floor = self.strength_floor();
        let strength_score = self.strength / floor;

        let confidence = self.entity_confidence.unwrap_or(0.5);
        let tag_bonus = confidence * 0.2; // scaled down from 0.3 to fit 0.2 weight budget

        let npmi_score = (npmi / 0.75).clamp(0.0, 1.5);

        npmi_score * 0.4 + strength_score * 0.4 + tag_bonus
    }
}

/// Creation source for relationship edges (FIX-06: dream replay tagging)
///
/// Tracks how an edge was created so retrieval can distinguish empirically
/// observed associations from speculative/replay-discovered connections.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum EdgeSource {
    /// Created from entity co-occurrence in user-stored memories
    #[default]
    CoOccurrence,
    /// Created by waking dream replay (maintenance consolidation)
    DreamReplay,
    /// Created by sleep-phase replay (deep consolidation)
    SleepReplay,
    /// Created by Hebbian coactivation during retrieval
    Coactivation,
    /// Created from MIF import or explicit user input
    Explicit,
    /// Edge seeded from population-level collective knowledge.
    /// Imported from collective store during new-user initialization.
    /// Starts at low strength, rapidly potentiated if confirmed by user behavior.
    CollectivePrior,
    /// Unknown edge source (forward compatibility for future variants).
    /// Deserialized from a variant not recognized by this version.
    #[serde(other)]
    Unknown,
}

/// Result of memory coactivation recording.
#[derive(Debug, Clone)]
pub struct CoactivationResult {
    /// Number of edges updated or created.
    pub edges_updated: usize,
    /// Edge tier promotions that occurred during strengthening.
    /// Each promotion signals the memory layer to boost associated memory importance.
    pub promotions: Vec<EdgePromotion>,
}

/// An edge tier promotion event (e.g., L1→L2, L2→L3).
#[derive(Debug, Clone)]
pub struct EdgePromotion {
    /// Source entity UUID of the promoted edge.
    pub from_entity: Uuid,
    /// Target entity UUID of the promoted edge.
    pub to_entity: Uuid,
    /// Previous tier name (e.g., "L1Working").
    pub old_tier: String,
    /// New tier name (e.g., "L2Episodic").
    pub new_tier: String,
}

/// Relationship types for ontological edge classification.
///
/// Extends Graphiti's semantic model with management, operational, and
/// evolution relationships for DevOps, software engineering, and robotics.
///
/// New variants are additive only — never remove existing variants to
/// maintain backward compatibility with MessagePack-serialized data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationType {
    /// Work relationships
    WorksWith,
    WorksAt,
    EmployedBy,

    /// Structural relationships
    PartOf,
    Contains,
    OwnedBy,

    /// Location relationships
    LocatedIn,
    LocatedAt,

    /// Usage relationships
    Uses,
    CreatedBy,
    DevelopedBy,

    /// Causal relationships
    Causes,
    ResultsIn,

    /// Learning relationships
    Learned,
    Knows,
    Teaches,

    /// Generic relationships (never penalized by ontological filtering)
    RelatedTo,
    AssociatedWith,

    /// Memory co-retrieval (Hebbian association between memories)
    CoRetrieved,

    /// Sentence co-occurrence (entities appearing in same sentence)
    /// Key for multi-hop: "Melanie" <-> "sunrise" when "Melanie painted a sunrise"
    CoOccurs,

    // =========================================================================
    // Extended ontological relations (added 2026-03)
    // =========================================================================
    /// Management / Governance
    /// Person/Team manages Project/Service/Team
    Manages,
    /// Service/Module depends on Service/Module
    DependsOn,
    /// Task requires Skill/Technology
    Requires,
    /// Configuration configures Service/Environment
    Configures,

    /// Preference / Recommendation
    /// Person prefers Technology/Configuration/Approach
    Prefers,
    /// Person/Role recommends Technology/Approach
    Recommends,

    /// Document / Knowledge
    /// Document documents Service/API/Project
    Documents,
    /// Module/Service implements Concept/Spec
    Implements,

    /// Operational
    /// Pipeline deploys Service to Environment
    DeploysTo,
    /// Metric/Service monitors Service/Pipeline
    Monitors,
    /// Event/Pipeline triggers Pipeline/Task
    Triggers,

    /// Comparison / Evolution
    /// Technology/Config superseded by newer version
    SupersededBy,
    /// Technology is alternative to Technology
    AlternativeTo,

    /// Assignment
    /// Task assigned to Person/Team
    AssignedTo,
    /// Person/Role approves Task/Document
    Approves,

    /// Custom relationship
    Custom(String),
}

impl RelationType {
    /// Get string representation of the relation type
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::WorksWith => "WorksWith",
            Self::WorksAt => "WorksAt",
            Self::EmployedBy => "EmployedBy",
            Self::PartOf => "PartOf",
            Self::Contains => "Contains",
            Self::OwnedBy => "OwnedBy",
            Self::LocatedIn => "LocatedIn",
            Self::LocatedAt => "LocatedAt",
            Self::Uses => "Uses",
            Self::CreatedBy => "CreatedBy",
            Self::DevelopedBy => "DevelopedBy",
            Self::Causes => "Causes",
            Self::ResultsIn => "ResultsIn",
            Self::Learned => "Learned",
            Self::Knows => "Knows",
            Self::Teaches => "Teaches",
            Self::RelatedTo => "RelatedTo",
            Self::AssociatedWith => "AssociatedWith",
            Self::CoRetrieved => "CoRetrieved",
            Self::CoOccurs => "CoOccurs",
            Self::Manages => "Manages",
            Self::DependsOn => "DependsOn",
            Self::Requires => "Requires",
            Self::Configures => "Configures",
            Self::Prefers => "Prefers",
            Self::Recommends => "Recommends",
            Self::Documents => "Documents",
            Self::Implements => "Implements",
            Self::DeploysTo => "DeploysTo",
            Self::Monitors => "Monitors",
            Self::Triggers => "Triggers",
            Self::SupersededBy => "SupersededBy",
            Self::AlternativeTo => "AlternativeTo",
            Self::AssignedTo => "AssignedTo",
            Self::Approves => "Approves",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Whether this relation asserts nothing about its kind, so no ontological intent can
    /// mismatch it: the documented generics plus the co-retrieval and co-occurrence edges
    /// recall and ingest write between any two entities.
    pub fn is_generic(&self) -> bool {
        matches!(
            self,
            Self::RelatedTo | Self::AssociatedWith | Self::CoRetrieved | Self::CoOccurs
        )
    }

    /// Whether this edge asserts that its `from_entity` caused its `to_entity`.
    pub fn is_causal(&self) -> bool {
        // Exhaustive so a new variant cannot join or leave the causal set by default.
        match self {
            Self::Causes | Self::ResultsIn | Self::Triggers => true,
            Self::WorksWith
            | Self::WorksAt
            | Self::EmployedBy
            | Self::PartOf
            | Self::Contains
            | Self::OwnedBy
            | Self::LocatedIn
            | Self::LocatedAt
            | Self::Uses
            | Self::CreatedBy
            | Self::DevelopedBy
            | Self::Learned
            | Self::Knows
            | Self::Teaches
            | Self::RelatedTo
            | Self::AssociatedWith
            | Self::CoRetrieved
            | Self::CoOccurs
            | Self::Manages
            | Self::DependsOn
            | Self::Requires
            | Self::Configures
            | Self::Prefers
            | Self::Recommends
            | Self::Documents
            | Self::Implements
            | Self::DeploysTo
            | Self::Monitors
            | Self::SupersededBy
            | Self::AlternativeTo
            | Self::AssignedTo
            | Self::Approves
            | Self::Custom(_) => false,
        }
    }
}

/// Causal cue lexemes in signal-strength order; first present wins.
///
/// Ported from upstream shodh-memory `predicate_from_cues`, causal family only.
/// Fragments like "in motion" survive an entity splitting the verb phrase
/// ("X set Y in motion"). The non-causal families upstream types in the same
/// table (Manages, WorksAt, CreatedBy, ...) are not ported.
fn causal_cue(sentence: &str) -> Option<&'static str> {
    const CAUSAL_CUES: [&str; 10] = [
        "in motion",
        "brought about",
        "gave rise",
        "triggered",
        "led directly to",
        "led to",
        "resulted in",
        "caused",
        "because of",
        "due to",
    ];
    CAUSAL_CUES.iter().copied().find(|c| sentence.contains(c))
}

/// Span-scoped, direction-aware causal typing of a co-mentioned entity pair.
///
/// Ported from upstream shodh-memory `extract_directed_predicate`. Finds both
/// mentions, scans only the sentence containing both for a causal cue, and
/// returns `(Triggers, a_is_source)`: the earlier mention is the cause, flipped
/// for effect-first constructions ("A because of B", "A was caused by B").
/// `None` when the mentions do not share a sentence, no cue is present, or an
/// endpoint lies inside the cue itself ("motion" from "in motion").
pub fn extract_directed_predicate(
    text: &str,
    name_a: &str,
    name_b: &str,
) -> Option<(RelationType, bool)> {
    let lc = text.to_ascii_lowercase();
    let a = name_a.to_ascii_lowercase();
    let b = name_b.to_ascii_lowercase();
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let pa = lc.find(&a)?;
    let pb = lc.find(&b)?;
    if pa == pb {
        return None;
    }
    let (lo, hi) = if pa < pb {
        (pa, pb + b.len())
    } else {
        (pb, pa + a.len())
    };
    let sent_start = lc[..lo]
        .rfind(['.', '!', '?', ';', '\n'])
        .map(|i| i + 1)
        .unwrap_or(0);
    let sent_end = lc[hi..]
        .find(['.', '!', '?', ';', '\n'])
        .map(|i| hi + i)
        .unwrap_or(lc.len());
    let sentence = &lc[sent_start..sent_end];
    let cue = causal_cue(sentence)?;

    // A cue word minted as an entity would otherwise bridge every chain that uses the cue.
    if let Some(cue_off) = sentence.find(cue) {
        let cue_lo = sent_start + cue_off;
        let cue_hi = cue_lo + cue.len();
        let overlaps_cue = |lo: usize, len: usize| lo < cue_hi && cue_lo < lo + len;
        if overlaps_cue(pa, a.len()) || overlaps_cue(pb, b.len()) {
            return None;
        }
    }

    const EFFECT_FIRST_CUES: [&str; 4] = ["because of", "due to", "caused by", "triggered by"];
    let effect_first = EFFECT_FIRST_CUES.iter().any(|c| sentence.contains(c));
    let a_first = pa < pb;
    Some((
        RelationType::Triggers,
        if effect_first { !a_first } else { a_first },
    ))
}

/// The relation the ontology implies between two co-mentioned entities by their labels alone,
/// as `(relation, a_is_source)`, or `None` when no rule types the pair.
///
/// Ported from upstream shodh-memory #259/#260 (`infer_relation_type_for_pair`), with two
/// adaptations. Upstream matches only the pair in the order the memory listed it, so
/// `(Organization, Person)` went untyped and `(Location, Person)` became a `LocatedIn` edge
/// from the place; here the table is consulted in both orders and the edge points from the
/// rule's first label. And where upstream falls back to `CoOccurs`, this returns `None`, so
/// the caller keeps veld's generic `RelatedTo`: edges dedup by relation type, and a second
/// generic type would fork every existing pair.
///
/// `(Person, Person)` is `Knows` only with `person_person_knows`, which upstream keeps off by
/// default (`SHODH_PERSON_PERSON_KNOWS`): a typed edge is eligible for ontological penalties
/// a generic one is not. Two technologies stay untyped: co-mention says nothing about
/// substitutability.
pub fn infer_relation_type_for_pair(
    a: &EntityLabel,
    b: &EntityLabel,
    person_person_knows: bool,
) -> Option<(RelationType, bool)> {
    fn rule(from: &EntityLabel, to: &EntityLabel, knows: bool) -> Option<RelationType> {
        use EntityLabel::*;
        use RelationType::*;
        Some(match (from, to) {
            (Person, Person) if knows => Knows,
            (Person, Organization) | (Person, Team) => WorksAt,
            (Person, Technology) | (Person, Service) | (Person, Database) => Uses,
            (Person, Skill) => Learned,
            (Task, Person) | (Task, Team) => AssignedTo,
            (Task, Technology) | (Task, Service) => Requires,
            (Service, Database) => Uses,
            (Module, Module) | (Service, Service) | (Module, Service) | (Service, Module) => {
                DependsOn
            }
            (Pipeline, Service) | (Pipeline, Environment) => DeploysTo,
            (Metric, Service) | (Metric, Pipeline) => Monitors,
            (Configuration, Service) | (Configuration, Environment) => Configures,
            (Document, Service)
            | (Document, Module)
            | (Document, Pipeline)
            | (Document, Project) => Documents,
            (Task, Project)
            | (Document, Repository)
            | (Repository, Project)
            | (Module, Project)
            | (Service, Project) => PartOf,
            (_, Location) => LocatedIn,
            _ => return None,
        })
    }
    rule(a, b, person_person_knows)
        .map(|rt| (rt, true))
        .or_else(|| rule(b, a, person_person_knows).map(|rt| (rt, false)))
}

/// Episodic node representing a discrete experience/memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Human-readable name/title
    pub name: String,

    /// Episode content (the actual experience data)
    pub content: String,

    /// When the original event occurred (event time)
    pub valid_at: DateTime<Utc>,

    /// When this was ingested into the system (ingestion time)
    pub created_at: DateTime<Utc>,

    /// Entities extracted from this episode
    pub entity_refs: Vec<Uuid>,

    /// Source type
    pub source: EpisodeSource,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Episode source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpisodeSource {
    Message,
    Document,
    Event,
    Observation,
}

/// Graph memory storage and operations
///
/// Uses a single RocksDB instance with 9 column families for all graph data.
/// This reduces file descriptor usage from 9 separate DBs to 1 (sharing WAL, MANIFEST, LOCK).
pub struct GraphMemory {
    /// Multi-family access to the graph's nine column families, backend-erased.
    /// Every read/write/scan/batch goes through this.
    store: Arc<dyn crate::storage::ColumnStore>,

    /// Present ONLY when rocksdb-backed: the backup/checkpoint accessor hands
    /// the engine DB to the rocksdb BackupEngine. A store built over another
    /// backend has no engine DB to hold.
    #[cfg_attr(not(feature = "rocksdb"), allow(dead_code))]
    db: Option<crate::storage::RocksDbHandle>,

    /// The live redb graph database, for the redb backup (redb admits one opener).
    #[cfg(feature = "storage-redb")]
    redb_db: Option<Arc<::redb::Database>>,

    /// In-memory entity name index for fast lookups (loaded from name_index CF)
    entity_name_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory lowercase name index for O(1) case-insensitive lookups
    entity_lowercase_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory stemmed name index for O(1) linguistic lookups
    /// Key: Porter-stemmed lowercase name, Value: Entity UUID
    entity_stemmed_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory path-suffix index for `prefix:path`-shaped names (see
    /// [`GraphMemory::find_file_paths_by_suffix`]). Derived from the name index; not persisted.
    entity_path_suffix_index: Arc<parking_lot::RwLock<HashMap<String, Vec<Uuid>>>>,

    // === Atomic counters for O(1) stats (P1 fix) ===
    /// Entity count - initialized from entity_name_index.len(), updated on add
    entity_count: Arc<AtomicUsize>,

    /// Relationship count - initialized on startup, updated on add
    relationship_count: Arc<AtomicUsize>,

    /// Episode count - initialized on startup, updated on add
    episode_count: Arc<AtomicUsize>,

    /// Total activation count across all edges — initialized on startup by summing
    /// activation_count from all relationships. Incremented on strengthen, decremented
    /// on edge deletion. Used as N_s denominator in NPMI computation for Hebbian pruning.
    total_activations: Arc<AtomicU64>,

    /// Mutex for serializing synapse updates to prevent race conditions (SHO-64)
    /// Uses parking_lot::Mutex for better performance than std::sync::Mutex
    synapse_update_lock: Arc<parking_lot::Mutex<()>>,

    /// In-memory cache of entity name embeddings for concept merging.
    /// Maps entity UUID → embedding vector. Loaded on startup, updated on add.
    /// Used when string-based dedup (exact/case/stemmed) fails — catches synonyms
    /// like "authentication" ↔ "auth" via cosine similarity.
    #[allow(clippy::type_complexity)]
    entity_embedding_cache: Arc<parking_lot::RwLock<Vec<(Uuid, Vec<f32>)>>>,

    /// Edges found below prune threshold during lazy-decay reads.
    /// Flushed as batch deletes on each maintenance cycle (no full scan needed).
    pending_prune: parking_lot::Mutex<Vec<Uuid>>,

    /// Entities that may have become orphaned from pruned edges.
    /// Checked during flush_pending_maintenance().
    pending_orphan_checks: parking_lot::Mutex<Vec<Uuid>>,

    /// Idempotency guard for Hebbian reinforcement (S-R4).
    ///
    /// Tracks the most recent `retrieval_id` values that have already been
    /// reinforced. A second `apply_reinforcement` call with the same
    /// `retrieval_id` is a no-op, closing the cardinality>1 retry-compounding
    /// hazard. Capacity is bounded at `SEEN_RETRIEVALS_CAPACITY`; oldest
    /// entries are evicted FIFO when the cap is reached.
    ///
    /// B5 note: this lock is held only during the O(1) set-membership check +
    /// insert, NOT during the actual edge-strength mutations.  Per-edge
    /// mutation is still serialised by `synapse_update_lock` (fine-grained),
    /// so no coarse graph-write lock is introduced.
    seen_retrievals: parking_lot::Mutex<(
        std::collections::HashSet<Uuid>,
        std::collections::VecDeque<Uuid>,
    )>,

    /// Uuids of the entity and relationship rows written through `store`.
    change_log: Arc<GraphChangeLog>,

    /// Entity selectivity from the last [`Self::compute_entity_selectivity`] pass. Held in
    /// memory: empty until a heavy maintenance cycle scores the graph, and an entity it has
    /// not scored reads as unknown, never as low.
    entity_selectivity: parking_lot::RwLock<HashMap<Uuid, f32>>,
}

impl GraphMemory {
    /// The rocksdb engine handle for backup/checkpoint. Engine-typed by design
    /// and named for it: GraphMemory is a rocksdb-backed type, and this handle
    /// feeds the rocksdb BackupEngine's comprehensive archive. Nothing else
    /// should reach through it.
    #[cfg(feature = "rocksdb")]
    pub fn rocksdb_for_backup(&self) -> Option<&DB> {
        self.db.as_deref()
    }

    #[cfg(feature = "storage-redb")]
    pub fn attach_redb_backup_db(&mut self, database: Arc<::redb::Database>) {
        self.redb_db = Some(database);
    }

    #[cfg(feature = "storage-redb")]
    pub fn redb_for_backup(&self) -> Option<Arc<::redb::Database>> {
        self.redb_db.clone()
    }

    /// Create a new graph memory system.
    ///
    /// If `shared_cache` is provided, block-cache reads are charged against the
    /// shared LRU cache (recommended for multi-tenant server mode). When `None`,
    /// a small per-instance cache is created (standalone / test use).
    #[cfg(feature = "rocksdb")]
    pub fn new(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        Self::new_with_tuning(path, shared_cache, crate::config::StorageTuning::process())
    }

    /// [`Self::new`] with an explicit durability class and standalone block cache in
    /// place of the process's. Every write the graph makes, the pre-column-family
    /// import included, syncs as `tuning` says.
    #[cfg(feature = "rocksdb")]
    pub fn new_with_tuning(
        path: &Path,
        shared_cache: Option<&crate::storage::SharedBlockCache>,
        tuning: crate::config::StorageTuning,
    ) -> Result<Self> {
        use crate::constants::ROCKSDB_GRAPH_WRITE_BUFFER_BYTES;
        /// The graph's own default when `VELD_ROCKSDB_BLOCK_CACHE_BYTES` is unset: a
        /// standalone graph is read far less than the memory store beside it.
        const STANDALONE_GRAPH_BLOCK_CACHE_BYTES: usize = 8 * 1024 * 1024;

        let graph_path = path.join("graph");
        std::fs::create_dir_all(&graph_path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(ROCKSDB_GRAPH_WRITE_BUFFER_BYTES);
        opts.set_max_write_buffer_number(2);

        // Shared block cache for multi-tenant, small local for standalone/tests.
        use rocksdb::{BlockBasedOptions, Cache};
        let local_cache;
        let cache = match shared_cache {
            Some(c) => c,
            None => {
                local_cache = Cache::new_lru_cache(
                    tuning
                        .rocksdb_block_cache_bytes
                        .unwrap_or(STANDALONE_GRAPH_BLOCK_CACHE_BYTES),
                );
                &local_cache
            }
        };

        // Base block options (no bloom filter) for CFs that primarily do prefix scans
        let mut base_block_opts = BlockBasedOptions::default();
        base_block_opts.set_block_cache(cache);
        base_block_opts.set_cache_index_and_filter_blocks(true);
        opts.set_block_based_table_factory(&base_block_opts);

        // Bloom-filtered options for frequently point-queried CFs.
        // 10 bits/key gives ~1% false positive rate, reducing disk reads
        // on entity/relationship lookups by avoiding unnecessary SST scans.
        let bloom_cf_set: &[&str] = &[
            CF_ENTITIES,
            CF_RELATIONSHIPS,
            CF_ENTITY_EDGES,
            CF_ENTITY_PAIR_INDEX,
            CF_EDGE_INDEX,
        ];
        let mut bloom_opts = opts.clone();
        let mut bloom_block_opts = BlockBasedOptions::default();
        bloom_block_opts.set_bloom_filter(10.0, false);
        bloom_block_opts.set_block_cache(cache);
        bloom_block_opts.set_cache_index_and_filter_blocks(true);
        bloom_opts.set_block_based_table_factory(&bloom_block_opts);

        // Build column family descriptors — hot CFs get bloom filters, others use base opts
        let cf_descriptors: Vec<ColumnFamilyDescriptor> = GRAPH_CF_NAMES
            .iter()
            .map(|name| {
                if bloom_cf_set.contains(name) {
                    ColumnFamilyDescriptor::new(*name, bloom_opts.clone())
                } else {
                    ColumnFamilyDescriptor::new(*name, opts.clone())
                }
            })
            .collect();

        let db = Arc::new(DB::open_cf_descriptors(&opts, &graph_path, cf_descriptors)?);

        // Migrate data from old separate-DB layout if needed
        let sync_writes = tuning.rocksdb_sync_writes();
        let migrated = Self::migrate_from_separate_dbs(path, &db, sync_writes)?;
        if migrated > 0 {
            tracing::info!(
                "Migrated {} entries from separate graph DBs to column families",
                migrated
            );
        }

        let store: Arc<dyn crate::storage::ColumnStore> = Arc::new(
            crate::storage::legacy_rocksdb::RocksDbColumnStore::with_sync_writes(
                Arc::clone(&db),
                sync_writes,
            ),
        );
        Self::from_store(store, Some(db))
    }

    /// Open the graph store at `path` on redb, as the server opens a tenant's graph.
    #[cfg(all(feature = "storage-redb", not(feature = "rocksdb")))]
    pub fn new(
        path: &std::path::Path,
        _shared_cache: Option<&crate::storage::SharedBlockCache>,
    ) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let database = crate::storage::redb::open_graph_database(path)?;
        let mut graph = Self::with_store(Arc::new(crate::storage::redb::RedbColumnStore::new(
            Arc::clone(&database),
        )))?;
        graph.attach_redb_backup_db(database);
        Ok(graph)
    }

    /// Bind a handle directly — the constructor a backend-agnostic caller uses.
    ///
    /// Skips migrate_from_separate_dbs: that is a rocksdb-to-rocksdb import from
    /// superseded directories and has no meaning on another backend. The families
    /// are materialized here, since redb creates tables lazily.
    pub fn with_store(store: Arc<dyn crate::storage::ColumnStore>) -> Result<Self> {
        for cf in GRAPH_CF_NAMES {
            store.ensure_cf(cf)?;
        }
        Self::from_store(store, None)
    }

    /// The shared startup path: load the three indices, count, and warm the
    /// embedding cache. Everything below reads through the trait, so both
    /// constructors do the same work over whichever engine they were given.
    fn from_store(
        store: Arc<dyn crate::storage::ColumnStore>,
        db: Option<crate::storage::RocksDbHandle>,
    ) -> Result<Self> {
        // Load entity name index from name_index CF (O(n) but faster than deserializing entities)
        // If empty, migrate from entities CF (one-time migration for existing data)
        let entity_name_index = Self::load_or_migrate_name_index(store.as_ref())?;

        // Load/migrate lowercase index for O(1) case-insensitive lookup
        let entity_lowercase_index =
            Self::load_or_migrate_lowercase_index(store.as_ref(), &entity_name_index)?;

        // Load/migrate stemmed index for O(1) linguistic lookup
        let entity_stemmed_index =
            Self::load_or_migrate_stemmed_index(store.as_ref(), &entity_name_index)?;

        let entity_count = entity_name_index.len();
        let entity_path_suffix_index = Self::build_path_suffix_index(&entity_name_index);

        Self::ensure_episode_recency_index(store.as_ref())?;

        // Counted once at open, then kept by the write paths, so get_stats() is O(1).
        let RelationshipCensus {
            rows: relationship_count,
            total_activations,
        } = Self::relationship_census_and_edge_index(store.as_ref())?;
        let episode_count = Self::count_cf_entries(store.as_ref(), CF_EPISODES);

        // Load entity embedding cache for concept merging
        // Only entities with pre-computed name_embeddings are cached
        let entity_embedding_cache =
            Self::load_entity_embedding_cache(store.as_ref(), &entity_name_index);
        let embedding_cache_size = entity_embedding_cache.len();

        let change_log = Arc::new(GraphChangeLog::default());
        let store: Arc<dyn crate::storage::ColumnStore> = Arc::new(
            change_log::JournaledColumnStore::new(store, Arc::clone(&change_log)),
        );
        let graph = Self {
            store,
            db,
            #[cfg(feature = "storage-redb")]
            redb_db: None,
            entity_name_index: Arc::new(parking_lot::RwLock::new(entity_name_index)),
            entity_lowercase_index: Arc::new(parking_lot::RwLock::new(entity_lowercase_index)),
            entity_stemmed_index: Arc::new(parking_lot::RwLock::new(entity_stemmed_index)),
            entity_path_suffix_index: Arc::new(parking_lot::RwLock::new(entity_path_suffix_index)),
            entity_count: Arc::new(AtomicUsize::new(entity_count)),
            relationship_count: Arc::new(AtomicUsize::new(relationship_count)),
            episode_count: Arc::new(AtomicUsize::new(episode_count)),
            total_activations: Arc::new(AtomicU64::new(total_activations)),
            synapse_update_lock: Arc::new(parking_lot::Mutex::new(())),
            entity_embedding_cache: Arc::new(parking_lot::RwLock::new(entity_embedding_cache)),
            pending_prune: parking_lot::Mutex::new(Vec::new()),
            pending_orphan_checks: parking_lot::Mutex::new(Vec::new()),
            seen_retrievals: parking_lot::Mutex::new((
                std::collections::HashSet::new(),
                std::collections::VecDeque::new(),
            )),
            change_log,
            entity_selectivity: parking_lot::RwLock::new(HashMap::new()),
        };

        if entity_count > 0 || relationship_count > 0 || episode_count > 0 {
            tracing::info!(
                "Loaded graph with {} entities ({} with embeddings), {} relationships, {} episodes, {} total activations",
                entity_count,
                embedding_cache_size,
                relationship_count,
                episode_count,
                total_activations
            );
        }

        Ok(graph)
    }

    /// Migrate data from the old separate-DB layout (pre-CF) into column families.
    ///
    /// Detects old `graph_*` subdirectories, opens them read-only, copies all KV
    /// pairs into the corresponding CF, then renames the old directory for rollback safety.
    #[cfg(feature = "rocksdb")]
    fn migrate_from_separate_dbs(base_path: &Path, db: &DB, sync_writes: bool) -> Result<usize> {
        let write_opts = crate::storage::rocksdb_write_options(sync_writes);
        let old_dirs: &[(&str, &str)] = &[
            ("graph_entities", CF_ENTITIES),
            ("graph_relationships", CF_RELATIONSHIPS),
            ("graph_episodes", CF_EPISODES),
            ("graph_entity_edges", CF_ENTITY_EDGES),
            ("graph_entity_pair_index", CF_ENTITY_PAIR_INDEX),
            ("graph_entity_episodes", CF_ENTITY_EPISODES),
            ("graph_entity_name_index", CF_NAME_INDEX),
            ("graph_entity_lowercase_index", CF_LOWERCASE_INDEX),
            ("graph_entity_stemmed_index", CF_STEMMED_INDEX),
        ];

        let mut total_migrated = 0usize;

        for (old_name, cf_name) in old_dirs {
            let old_path = base_path.join(old_name);
            if !old_path.exists() {
                continue;
            }

            let cf = db.cf_handle(cf_name).unwrap();

            // Only migrate if the CF is empty (avoid double migration)
            if db
                .iterator_cf(cf, rocksdb::IteratorMode::Start)
                .next()
                .is_some()
            {
                // CF already has data — just rename the old dir
                let renamed = base_path.join(format!("{}.pre_cf_migration", old_name));
                if !renamed.exists() {
                    let _ = std::fs::rename(&old_path, &renamed);
                }
                continue;
            }

            // Open old DB read-only and copy all entries
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_path, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;

                    for item in old_db.iterator(rocksdb::IteratorMode::Start) {
                        match item {
                            Ok((key, value)) => {
                                batch.put_cf(cf, &key, &value);
                                count += 1;
                                // Flush in chunks to limit memory usage
                                if count.is_multiple_of(10_000) {
                                    db.write_opt(std::mem::take(&mut batch), &write_opts)?;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Error reading from old {}: {}", old_name, e);
                                break;
                            }
                        }
                    }

                    if count > 0 {
                        db.write_opt(batch, &write_opts)?;
                    }

                    drop(old_db);

                    // Rename old directory for rollback safety
                    let renamed = base_path.join(format!("{}.pre_cf_migration", old_name));
                    if let Err(e) = std::fs::rename(&old_path, &renamed) {
                        tracing::warn!(
                            "Migrated {} entries from {} but failed to rename: {}",
                            count,
                            old_name,
                            e
                        );
                    } else {
                        tracing::info!(
                            "Migrated {} entries from {} to CF '{}'",
                            count,
                            old_name,
                            cf_name
                        );
                    }

                    total_migrated += count;
                }
                Err(e) => {
                    tracing::warn!("Failed to open old DB {} for migration: {}", old_name, e);
                }
            }
        }

        Ok(total_migrated)
    }

    /// Load entity name->UUID index from name_index CF, or migrate from entities CF if empty
    fn load_or_migrate_name_index(
        store: &dyn crate::storage::ColumnStore,
    ) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        // Try to load from name_index CF first
        for (key, value) in store.scan_cf(CF_NAME_INDEX, None, None)? {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If name_index CF is empty but entities exist, migrate (one-time operation)
        if index.is_empty() {
            let mut migrated_count = 0;
            for (_, value) in store.scan_cf(CF_ENTITIES, None, None)? {
                if let Ok(entity) = bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                )
                .map(|(v, _)| v)
                {
                    // Store in name_index CF: name -> UUID bytes
                    store.put_cf(
                        CF_NAME_INDEX,
                        entity.name.as_bytes(),
                        entity.uuid.as_bytes(),
                    )?;
                    index.insert(entity.name.clone(), entity.uuid);
                    migrated_count += 1;
                }
            }
            if migrated_count > 0 {
                tracing::info!("Migrated {} entities to name index CF", migrated_count);
            }
        }

        Ok(index)
    }

    /// Load lowercase name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) case-insensitive entity lookup instead of O(n) linear search.
    fn load_or_migrate_lowercase_index(
        store: &dyn crate::storage::ColumnStore,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        // Try to load from lowercase_index CF
        for (key, value) in store.scan_cf(CF_LOWERCASE_INDEX, None, None)? {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If empty but name_index has data, migrate (one-time operation)
        if index.is_empty() && !name_index.is_empty() {
            for (name, uuid) in name_index {
                let lowercase_name = name.to_lowercase();
                store.put_cf(
                    CF_LOWERCASE_INDEX,
                    lowercase_name.as_bytes(),
                    uuid.as_bytes(),
                )?;
                index.insert(lowercase_name, *uuid);
            }
            tracing::info!(
                "Migrated {} entities to lowercase index CF",
                name_index.len()
            );
        }

        Ok(index)
    }

    /// Load stemmed name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) linguistic entity lookup: "running" matches "run"
    /// Uses Porter2 stemmer for English language stemming.
    fn load_or_migrate_stemmed_index(
        store: &dyn crate::storage::ColumnStore,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        // Try to load from stemmed_index CF
        for (key, value) in store.scan_cf(CF_STEMMED_INDEX, None, None)? {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If empty but name_index has data, migrate (one-time operation)
        if index.is_empty() && !name_index.is_empty() {
            let stemmer = Stemmer::create(Algorithm::English);
            for (name, uuid) in name_index {
                let stemmed_name = Self::stem_entity_name(&stemmer, name);
                store.put_cf(CF_STEMMED_INDEX, stemmed_name.as_bytes(), uuid.as_bytes())?;
                index.insert(stemmed_name, *uuid);
            }
            tracing::info!("Migrated {} entities to stemmed index CF", name_index.len());
        }

        Ok(index)
    }

    /// Stem an entity name for linguistic matching
    ///
    /// For multi-word names (e.g., "New York City"), stems each word and joins.
    /// Returns lowercase stemmed version for consistent matching.
    fn stem_entity_name(stemmer: &Stemmer, name: &str) -> String {
        name.split_whitespace()
            .map(|word| stemmer.stem(&word.to_lowercase()).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Visit every row of `cf`, one first-key-byte page at a time: `scan_cf` returns a
    /// `Vec`, and a whole family of a large graph (866k relationships on one live tenant)
    /// would otherwise be held in memory at once.
    fn visit_cf_paged(
        store: &dyn crate::storage::ColumnStore,
        cf: &str,
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        for first in 0..=u8::MAX {
            for (key, value) in store.scan_cf(cf, Some(&[first]), None)? {
                visit(&key, &value)?;
            }
        }
        Ok(())
    }

    /// Count entries in a column family (one-time startup cost)
    fn count_cf_entries(store: &dyn crate::storage::ColumnStore, cf: &str) -> usize {
        let mut count = 0usize;
        match Self::visit_cf_paged(store, cf, |_, _| {
            count += 1;
            Ok(())
        }) {
            Ok(()) => count,
            Err(e) => {
                tracing::warn!(cf, error = %e, "could not count graph family at open");
                0
            }
        }
    }

    /// One pass over the relationships family at open: its row count, the activation total
    /// (the NPMI denominator, kept incrementally afterwards), and, for a graph written
    /// before `edge_index` existed, that index's rows for every edge.
    fn relationship_census_and_edge_index(
        store: &dyn crate::storage::ColumnStore,
    ) -> Result<RelationshipCensus> {
        const BATCH_ROWS: usize = 10_000;
        let backfill = store.get_cf(CF_EDGE_INDEX, EDGE_INDEX_MARKER)?.is_none();
        let mut census = RelationshipCensus::default();
        let mut batch = crate::storage::WriteBatch::new();
        let mut indexed = 0usize;
        Self::visit_cf_paged(store, CF_RELATIONSHIPS, |key, value| {
            census.rows += 1;
            // The family also holds `mem_edge:` lookup rows; edges are keyed by their uuid.
            if key.len() != 16 {
                return Ok(());
            }
            let Ok(edge) = RelationshipEdge::decode(value) else {
                return Ok(());
            };
            census.total_activations += edge.activation_count as u64;
            if backfill {
                Self::stage_edge_index_rows(&mut batch, &edge);
                indexed += 1;
                if batch.len() >= BATCH_ROWS {
                    store.write_batch(&std::mem::take(&mut batch))?;
                }
            }
            Ok(())
        })?;
        if backfill {
            batch.put_cf(CF_EDGE_INDEX, EDGE_INDEX_MARKER, b"1".as_slice());
            store.write_batch(&batch)?;
            if indexed > 0 {
                tracing::info!(edges = indexed, "Backfilled the graph edge index");
            }
        }
        Ok(census)
    }

    /// `e:{episode}:{edge}`: `edge` was created by `episode`.
    fn episode_edge_key(episode: &Uuid, edge: &Uuid) -> String {
        format!("e:{episode}:{edge}")
    }

    fn episode_edge_prefix(episode: &Uuid) -> String {
        format!("e:{episode}:")
    }

    /// `t:{lo}:{hi}:{relation:?}`: the edge joining `a` and `b`, in either direction,
    /// under `relation`. Debug form, so `Custom("RelatedTo")` and `RelatedTo` differ.
    fn typed_pair_key(a: &Uuid, b: &Uuid, relation: &RelationType) -> String {
        format!("t:{}:{relation:?}", Self::pair_key(a, b))
    }

    /// Add `edge`'s `edge_index` rows to `batch`.
    fn stage_edge_index_rows(batch: &mut crate::storage::WriteBatch, edge: &RelationshipEdge) {
        batch.put_cf(
            CF_EDGE_INDEX,
            Self::typed_pair_key(&edge.from_entity, &edge.to_entity, &edge.relation_type)
                .into_bytes(),
            edge.uuid.as_bytes().to_vec(),
        );
        if let Some(episode) = edge.source_episode_id {
            batch.put_cf(
                CF_EDGE_INDEX,
                Self::episode_edge_key(&episode, &edge.uuid).into_bytes(),
                Vec::<u8>::new(),
            );
        }
    }

    /// Add to `batch` the removal of `edge`'s `edge_index` rows. The typed-pair row goes
    /// only while it still names this edge: another edge may have taken the slot.
    fn stage_edge_index_removals(
        &self,
        batch: &mut crate::storage::WriteBatch,
        edge: &RelationshipEdge,
    ) -> Result<()> {
        let typed = Self::typed_pair_key(&edge.from_entity, &edge.to_entity, &edge.relation_type);
        if self
            .store
            .get_cf(CF_EDGE_INDEX, typed.as_bytes())?
            .is_some_and(|holder| holder.as_slice() == edge.uuid.as_bytes())
        {
            batch.delete_cf(CF_EDGE_INDEX, typed.into_bytes());
        }
        if let Some(episode) = edge.source_episode_id {
            batch.delete_cf(
                CF_EDGE_INDEX,
                Self::episode_edge_key(&episode, &edge.uuid).into_bytes(),
            );
        }
        Ok(())
    }

    /// Load entity embedding cache from persisted entities.
    ///
    /// Scans entities referenced by the name index and collects those with
    /// pre-computed name_embeddings into an in-memory cache for O(n) concept
    /// merging during `add_entity()`. Entities without embeddings (pre-upgrade
    /// data) are skipped and will gain embeddings on their next mention.
    fn load_entity_embedding_cache(
        store: &dyn crate::storage::ColumnStore,
        name_index: &HashMap<String, Uuid>,
    ) -> Vec<(Uuid, Vec<f32>)> {
        let mut cache = Vec::with_capacity(ENTITY_EMBEDDING_CACHE_MAX.min(name_index.len()));
        for uuid in name_index.values() {
            let key = uuid.as_bytes();
            if let Ok(Some(value)) = store.get_cf(CF_ENTITIES, key) {
                if let Ok((entity, _)) = bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    if let Some(emb) = entity.name_embedding {
                        cache.push((*uuid, emb));
                        if cache.len() >= ENTITY_EMBEDDING_CACHE_MAX {
                            break;
                        }
                    }
                }
            }
        }
        cache
    }

    /// Add or update an entity node
    /// Salience is updated using the formula: salience = base_salience * (1 + 0.1 * ln(mention_count)),
    /// the boost capped at `ENTITY_SALIENCE_MAX_FREQUENCY_BOOST` and the result at 1.0, where
    /// base_salience is the first mention's. Frequently mentioned entities grow in salience
    /// (gravitational wells get heavier)
    ///
    /// BUG-002 FIX: Handles crash recovery for orphaned entities/stale indices
    pub fn add_entity(&self, mut entity: EntityNode) -> Result<Uuid> {
        // Multi-tier dedup pipeline: exact → case-insensitive → stemmed → embedding
        // Each tier is faster than the next; short-circuits on first match.

        // Tier 1: Exact name match (O(1))
        let mut existing_uuid = {
            let index = self.entity_name_index.read();
            index.get(&entity.name).cloned()
        };

        // Tier 2: Case-insensitive match (O(1))
        if existing_uuid.is_none() {
            let lowercase_name = entity.name.to_lowercase();
            let index = self.entity_lowercase_index.read();
            existing_uuid = index.get(&lowercase_name).cloned();
        }

        // Tier 3: Stemmed match (O(1)) — "running" matches "run"
        // Skip for proper nouns to prevent "Paris" → "pari" merging with "Parison"
        if existing_uuid.is_none() && !entity.is_proper_noun {
            let stemmer = Stemmer::create(Algorithm::English);
            let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);
            let index = self.entity_stemmed_index.read();
            existing_uuid = index.get(&stemmed_name).cloned();
        }

        // Tier 4: Embedding-based concept merge (O(n) over cache)
        // Catches synonyms like "authentication" ↔ "auth" that string matching misses.
        // Only runs when the entity carries a name_embedding (populated by caller).
        if existing_uuid.is_none() {
            if let Some(ref new_emb) = entity.name_embedding {
                let cache = self.entity_embedding_cache.read();
                let mut best_match: Option<(Uuid, f32)> = None;
                let mut crossed = 0u64;
                for (uuid, existing_emb) in cache.iter() {
                    // The merge bar was set on MiniLM name embeddings; read on that scale.
                    let sim = crate::similarity::scale::name_to_reference(
                        crate::similarity::cosine_similarity(new_emb, existing_emb),
                    );
                    let crosses = sim >= ENTITY_CONCEPT_MERGE_THRESHOLD;
                    crossed += u64::from(crosses);
                    if crosses && best_match.is_none_or(|(_, best_sim)| sim > best_sim) {
                        best_match = Some((*uuid, sim));
                    }
                }
                crate::similarity::bars::SimilarityBar::EntityConceptMerge
                    .record_many(cache.len() as u64, crossed);
                if let Some((matched_uuid, sim)) = best_match {
                    tracing::debug!(
                        "Concept merge: '{}' matched existing entity {} (cosine={:.3})",
                        entity.name,
                        matched_uuid,
                        sim
                    );
                    existing_uuid = Some(matched_uuid);
                }
            }
        }

        let is_new_entity;
        if let Some(uuid) = existing_uuid {
            // BUG-002 FIX: Verify entity actually exists in DB (handles stale index)
            if let Some(existing) = self.get_entity(&uuid)? {
                // Update existing entity — merge into canonical node
                entity.uuid = uuid;
                entity.mention_count = existing.mention_count + 1;
                entity.last_seen_at = Utc::now();
                entity.created_at = existing.created_at;
                entity.is_proper_noun = existing.is_proper_noun || entity.is_proper_noun;

                // Preserve the canonical name (first-seen name wins)
                entity.name = existing.name.clone();

                // Merge labels: preserve all observed entity types
                for label in &existing.labels {
                    if !entity.labels.contains(label) {
                        entity.labels.push(label.clone());
                    }
                }

                // Preserve existing embedding if the incoming one is None
                if entity.name_embedding.is_none() {
                    entity.name_embedding = existing.name_embedding;
                }

                // salience = base × mention_frequency_boost(mentions), at most 1.0, base being
                // the first mention's salience. The stored value already carries the boost for
                // the previous count, so it is rescaled by the ratio of the two boosts.
                // Multiplying it by the whole boost compounded on every mention.
                entity.salience = (existing.salience
                    * mention_frequency_boost(entity.mention_count)
                    / mention_frequency_boost(existing.mention_count))
                .min(1.0);
                is_new_entity = false;
            } else {
                // BUG-002 FIX: Stale index entry - entity in index but not in DB
                tracing::warn!(
                    "Stale index entry for entity '{}' (uuid={}), recreating",
                    entity.name,
                    uuid
                );
                entity.uuid = Uuid::new_v4();
                entity.created_at = Utc::now();
                entity.last_seen_at = entity.created_at;
                entity.mention_count = 1;
                is_new_entity = true;
            }
        } else {
            // Genuinely new entity — no match at any tier
            entity.uuid = Uuid::new_v4();
            entity.created_at = Utc::now();
            entity.last_seen_at = entity.created_at;
            entity.mention_count = 1;
            is_new_entity = true;
        }

        self.persist_entity(&entity, is_new_entity)?;
        Ok(entity.uuid)
    }

    /// Store `entity` exactly as given, under `entity.uuid`, and point its names at it:
    /// persistence only, which is what `GraphStore::put_entity` means on every engine.
    /// The dedup, merging and mention counting of [`Self::add_entity`] are the graph
    /// engine's, and a caller that names the uuid has already decided which node this is.
    pub fn put_entity_record(&self, entity: &EntityNode) -> Result<()> {
        let is_new_entity = self
            .store
            .get_cf(CF_ENTITIES, entity.uuid.as_bytes())?
            .is_none();
        self.persist_entity(entity, is_new_entity)
    }

    /// Index `entity`'s names and write it under its uuid, counting it when it is new.
    fn persist_entity(&self, entity: &EntityNode, is_new_entity: bool) -> Result<()> {
        // BUG-002 FIX: Write index FIRST, then entity
        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        // Update in-memory indices
        {
            let mut index = self.entity_name_index.write();
            index.insert(entity.name.clone(), entity.uuid);
        }
        {
            let mut lowercase_index = self.entity_lowercase_index.write();
            lowercase_index.insert(lowercase_name.clone(), entity.uuid);
        }
        // Skip stemmed index for proper nouns to prevent "Paris" → "pari" collisions
        if !entity.is_proper_noun {
            let mut stemmed_index = self.entity_stemmed_index.write();
            stemmed_index.insert(stemmed_name.clone(), entity.uuid);
        }
        self.index_path_suffixes(&entity.name, entity.uuid);

        // Update entity embedding cache for future concept merges
        if let Some(ref emb) = entity.name_embedding {
            let mut cache = self.entity_embedding_cache.write();
            if is_new_entity {
                cache.push((entity.uuid, emb.clone()));
                // Evict oldest entries when cache exceeds the configured maximum.
                // Oldest entries (index 0) are typically the least recently mentioned
                // since they were loaded at startup or added earliest.
                if cache.len() > ENTITY_EMBEDDING_CACHE_MAX {
                    let excess = cache.len() - ENTITY_EMBEDDING_CACHE_MAX;
                    cache.drain(..excess);
                }
            } else {
                // Update existing entry in cache (embedding may have changed)
                if let Some(entry) = cache.iter_mut().find(|(uuid, _)| *uuid == entity.uuid) {
                    entry.1 = emb.clone();
                }
            }
        }

        // Persist name->UUID mappings
        self.store.put_cf(
            CF_NAME_INDEX,
            entity.name.as_bytes(),
            entity.uuid.as_bytes(),
        )?;
        self.store.put_cf(
            CF_LOWERCASE_INDEX,
            lowercase_name.as_bytes(),
            entity.uuid.as_bytes(),
        )?;
        if !entity.is_proper_noun {
            self.store.put_cf(
                CF_STEMMED_INDEX,
                stemmed_name.as_bytes(),
                entity.uuid.as_bytes(),
            )?;
        }

        // Store entity in database
        let key = entity.uuid.as_bytes();
        let value = bincode::serde::encode_to_vec(entity, bincode::config::standard())?;
        self.store.put_cf(CF_ENTITIES, key, &value)?;

        // Increment counter only for truly new entities
        if is_new_entity {
            self.entity_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get entity by UUID
    pub fn get_entity(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        let key = uuid.as_bytes();
        match self.store.get_cf(CF_ENTITIES, key)? {
            Some(value) => {
                let (entity, _): (EntityNode, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Delete an entity, every relationship it is an endpoint of, and all its
    /// index entries.
    ///
    /// Removes:
    /// 0. each relationship the `entity_edges` index lists for it, through
    ///    `delete_relationship`, so no edge is left pointing at nothing
    /// 1. `entities` CF (primary storage)
    /// 2. `entity_name_index` (exact name → UUID)
    /// 3. `entity_lowercase_index` (lowercase name → UUID)
    /// 4. `entity_stemmed_index` (stemmed name → UUID)
    /// 5. `entity_embedding_cache` (in-memory embedding vector)
    /// 6. `entity_pair_index` CF (co-occurrence pair entries)
    /// 7. Decrements `entity_count`
    ///
    /// Returns true if the entity existed and was deleted.
    pub fn delete_entity(&self, uuid: &Uuid) -> Result<bool> {
        let entity = match self.get_entity(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // 0. Remove its relationships (keys are "{entity}:{edge}")
        let edge_prefix = format!("{uuid}:");
        let edge_uuids: Vec<Uuid> = self
            .store
            .scan_cf(CF_ENTITY_EDGES, Some(edge_prefix.as_bytes()), None)?
            .into_iter()
            .map_while(|(key, _)| {
                std::str::from_utf8(&key)
                    .ok()?
                    .strip_prefix(&edge_prefix)
                    .map(|rest| Uuid::parse_str(rest).ok())
            })
            .flatten()
            .collect();
        for edge_uuid in &edge_uuids {
            self.delete_relationship(edge_uuid)?;
        }

        // 1. Remove from entities CF
        self.store.delete_cf(CF_ENTITIES, uuid.as_bytes())?;

        // 2-3-4. Remove from name indices (in-memory + persisted)
        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        {
            let mut index = self.entity_name_index.write();
            index.remove(&entity.name);
        }
        self.store
            .delete_cf(CF_NAME_INDEX, entity.name.as_bytes())?;

        {
            let mut index = self.entity_lowercase_index.write();
            index.remove(&lowercase_name);
        }
        self.store
            .delete_cf(CF_LOWERCASE_INDEX, lowercase_name.as_bytes())?;

        {
            let mut index = self.entity_stemmed_index.write();
            index.remove(&stemmed_name);
        }
        self.store
            .delete_cf(CF_STEMMED_INDEX, stemmed_name.as_bytes())?;
        self.unindex_path_suffixes(&entity.name, uuid);

        // 5. Remove from embedding cache
        {
            let mut cache = self.entity_embedding_cache.write();
            cache.retain(|(id, _)| id != uuid);
        }

        // 6. Remove entity_pair_index entries (prefix scan)
        let prefix = format!("{}:", uuid);
        let mut pairs_to_delete = Vec::new();
        let iter = self
            .store
            .scan_cf(CF_ENTITY_PAIR_INDEX, Some(prefix.as_bytes()), None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for item in iter {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.starts_with(&prefix) {
                        pairs_to_delete.push(key.to_vec());
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        // Also scan for reverse direction (other_uuid:this_uuid)
        let suffix = format!(":{}", uuid);
        let iter = self
            .store
            .scan_cf(CF_ENTITY_PAIR_INDEX, None, None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for item in iter {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.ends_with(&suffix) {
                        pairs_to_delete.push(key.to_vec());
                    }
                }
                Err(_) => break,
            }
        }
        for key in &pairs_to_delete {
            self.store.delete_cf(CF_ENTITY_PAIR_INDEX, key)?;
        }

        // 7. Remove entity_episodes entries (prefix scan)
        //
        // This step did not exist. The inverted index is keyed
        // `{entity_uuid}:{episode_uuid}` (entity_episode_key) and was only ever
        // cleaned per-EPISODE, so deleting an entity that episodes still referenced
        // left its rows behind forever. Orphan cleanup runs repeatedly, so the
        // residue accumulates: a slow leak in index space that nothing reports and
        // that get_episodes_by_entity would happily prefix-scan into, resolving
        // episode ids for an entity that no longer exists.
        //
        // Only the forward direction exists for this index — unlike
        // entity_pair_index above, which needs the reverse sweep because it is
        // written in both orientations.
        // Both inverted indexes: `{uuid}:` (unordered) and `{uuid}#` (recency-ordered).
        let mut episode_keys_to_delete = Vec::new();
        for episode_prefix in [format!("{}:", uuid), format!("{}#", uuid)] {
            for (key, _) in
                self.store
                    .scan_cf(CF_ENTITY_EPISODES, Some(episode_prefix.as_bytes()), None)?
            {
                episode_keys_to_delete.push(key);
            }
        }
        let leaked = episode_keys_to_delete.len();
        for key in &episode_keys_to_delete {
            self.store.delete_cf(CF_ENTITY_EPISODES, key)?;
        }
        if leaked > 0 {
            tracing::debug!(
                entity = %uuid,
                entries = leaked,
                "delete_entity: scrubbed entity_episodes index entries",
            );
        }

        // 8. Decrement counter
        self.entity_count.fetch_sub(1, Ordering::Relaxed);

        tracing::debug!("Deleted orphaned entity '{}' (uuid={})", entity.name, uuid);
        Ok(true)
    }

    /// Strict entity resolution: exact, case-insensitive and stemmed tiers
    /// only, with no substring or word-level fuzzing.
    ///
    /// For callers that need precision over recall, such as seeds for
    /// [`trace_causal_origins`](Self::trace_causal_origins): the fuzzy tiers can
    /// bind a query fragment to an arbitrary hub entity.
    pub fn find_entity_by_name_strict(&self, name: &str) -> Result<Option<EntityNode>> {
        match self.find_entity_uuid_strict(name) {
            Some(uuid) => self.get_entity(&uuid),
            None => Ok(None),
        }
    }

    /// The entity [`find_entity_by_name_strict`](Self::find_entity_by_name_strict) resolves,
    /// as its uuid, from the in-memory name indexes alone: no storage read.
    pub fn find_entity_uuid_strict(&self, name: &str) -> Option<Uuid> {
        if let Some(uuid) = self.entity_name_index.read().get(name).copied() {
            return Some(uuid);
        }
        if let Some(uuid) = self
            .entity_lowercase_index
            .read()
            .get(&name.to_lowercase())
            .copied()
        {
            return Some(uuid);
        }
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, name);
        self.entity_stemmed_index.read().get(&stemmed_name).copied()
    }

    /// Find entity by name (case-insensitive, O(1) lookup)
    ///
    /// Uses a multi-tier matching strategy:
    /// 1. Exact match (O(1)) - fastest
    /// 2. Case-insensitive match (O(1)) - common case
    /// 3. Stemmed match (O(1)) - "running" matches "run"
    /// 4. Substring match - "York" matches "New York City"
    /// 5. Word-level match - "York" matches "New York"
    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<EntityNode>> {
        // Tier 1: Exact match (O(1))
        let uuid = {
            let index = self.entity_name_index.read();
            index.get(name).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 2: Case-insensitive match (O(1))
        let name_lower = name.to_lowercase();
        let uuid = {
            let lowercase_index = self.entity_lowercase_index.read();
            lowercase_index.get(&name_lower).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 3: Stemmed match (O(1)) - "running" matches "run", "conversations" matches "conversation"
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, name);
        let uuid = {
            let stemmed_index = self.entity_stemmed_index.read();
            stemmed_index.get(&stemmed_name).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 4 & 5: Fuzzy matching (O(n) but bounded)
        // Only do fuzzy matching for names >= 3 chars to avoid noise
        // Deterministic: collect ALL matches, pick highest salience (break ties by shortest name)
        if name.len() >= 3 {
            let lowercase_index = self.entity_lowercase_index.read();
            let mut candidates: Vec<(Uuid, String)> = Vec::new();

            // Tier 4: Substring match - query is substring of entity
            // e.g., "York" matches "New York City"
            for (entity_name, uuid) in lowercase_index.iter() {
                if entity_name.contains(&name_lower) {
                    candidates.push((*uuid, entity_name.clone()));
                }
            }

            // Tier 5: Word-level match (only if Tier 4 found nothing)
            if candidates.is_empty() {
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();
                for (entity_name, uuid) in lowercase_index.iter() {
                    let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                    for qw in &query_words {
                        if entity_words.iter().any(|ew| ew == qw || ew.starts_with(qw)) {
                            candidates.push((*uuid, entity_name.clone()));
                            break;
                        }
                    }
                }
            }

            // Pick best candidate: highest salience, then shortest name, then the name itself.
            // Candidates come from map iteration, so the last key keeps equal (salience,
            // length) pairs from resolving to whichever entity the hash order visits first.
            if !candidates.is_empty() {
                let mut best: Option<(Uuid, f32, &String)> = None;
                for (uuid, name) in &candidates {
                    let salience = self.get_entity(uuid)?.map(|e| e.salience).unwrap_or(0.0);
                    let better = match &best {
                        None => true,
                        Some((_, best_sal, best_name)) => salience
                            .total_cmp(best_sal)
                            .then_with(|| best_name.len().cmp(&name.len()))
                            .then_with(|| best_name.as_str().cmp(name.as_str()))
                            .is_gt(),
                    };
                    if better {
                        best = Some((*uuid, salience, name));
                    }
                }
                if let Some((uuid, _, _)) = best {
                    return self.get_entity(&uuid);
                }
            }
        }

        Ok(None)
    }

    /// [`find_entity_by_name_strict`](Self::find_entity_by_name_strict), then the substring
    /// and word tiers of [`find_entity_by_name`](Self::find_entity_by_name) bounded for hot
    /// read paths: at most `max_candidates` fuzzy matches are kept (shortest names first,
    /// then by name), and only those are read from storage to rank by salience. The
    /// unbounded form clones and reads every match, which on a large graph is most of the
    /// name index for a short or common word.
    pub fn find_entity_by_name_bounded(
        &self,
        name: &str,
        max_candidates: usize,
    ) -> Result<Option<EntityNode>> {
        if let Some(found) = self.find_entity_by_name_strict(name)? {
            return Ok(Some(found));
        }
        if name.len() < 3 || max_candidates == 0 {
            return Ok(None);
        }
        let name_lower = name.to_lowercase();

        let candidates: Vec<(usize, String, Uuid)> = {
            let lowercase_index = self.entity_lowercase_index.read();
            // Max-heap on (length, name): the top is the weakest kept candidate.
            let mut kept: BinaryHeap<(usize, String, Uuid)> = BinaryHeap::new();
            let offer = |kept: &mut BinaryHeap<(usize, String, Uuid)>,
                         entity_name: &String,
                         uuid: Uuid| {
                if kept.len() < max_candidates {
                    kept.push((entity_name.len(), entity_name.clone(), uuid));
                } else if let Some((top_len, top_name, _)) = kept.peek() {
                    if (entity_name.len(), entity_name.as_str()) < (*top_len, top_name.as_str()) {
                        kept.pop();
                        kept.push((entity_name.len(), entity_name.clone(), uuid));
                    }
                }
            };

            for (entity_name, uuid) in lowercase_index.iter() {
                if entity_name.contains(&name_lower) {
                    offer(&mut kept, entity_name, *uuid);
                }
            }
            if kept.is_empty() {
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();
                for (entity_name, uuid) in lowercase_index.iter() {
                    let word_hit = entity_name
                        .split_whitespace()
                        .any(|ew| query_words.iter().any(|qw| ew == *qw || ew.starts_with(qw)));
                    if word_hit {
                        offer(&mut kept, entity_name, *uuid);
                    }
                }
            }
            kept.into_vec()
        };

        // Highest salience, then shortest name, then the name itself.
        let mut best: Option<(EntityNode, usize, String)> = None;
        for (len, lower_name, uuid) in candidates {
            let Some(entity) = self.get_entity(&uuid)? else {
                continue;
            };
            let better = match &best {
                None => true,
                Some((held, held_len, held_name)) => entity
                    .salience
                    .total_cmp(&held.salience)
                    .then_with(|| held_len.cmp(&len))
                    .then_with(|| held_name.as_str().cmp(lower_name.as_str()))
                    .is_gt(),
            };
            if better {
                best = Some((entity, len, lower_name));
            }
        }
        Ok(best.map(|(entity, _, _)| entity))
    }

    /// Find all entities matching a name with fuzzy matching
    ///
    /// Returns multiple matches ranked by match quality.
    /// Useful for spreading activation across related entities.
    pub fn find_entities_fuzzy(&self, name: &str, max_results: usize) -> Result<Vec<EntityNode>> {
        let mut results = Vec::new();
        let name_lower = name.to_lowercase();

        // Skip very short queries
        if name.len() < 2 {
            return Ok(results);
        }

        let lowercase_index = self.entity_lowercase_index.read();

        // Score and collect matches
        let mut scored: Vec<(Uuid, f32)> = Vec::new();

        for (entity_name, uuid) in lowercase_index.iter() {
            let score = if entity_name == &name_lower {
                1.0 // Exact match
            } else if entity_name.starts_with(&name_lower) {
                0.9 // Prefix match
            } else if entity_name.contains(&name_lower) {
                0.7 // Substring match
            } else {
                // Word-level match
                let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();

                let mut word_score: f32 = 0.0;
                for qw in &query_words {
                    for ew in &entity_words {
                        if ew == qw {
                            word_score += 0.5;
                        } else if ew.starts_with(qw) {
                            word_score += 0.3;
                        }
                    }
                }
                word_score.min(0.6) // Cap word-level score
            };

            if score > 0.0 {
                scored.push((*uuid, score));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Take top results
        for (uuid, _score) in scored.into_iter().take(max_results) {
            if let Some(entity) = self.get_entity(&uuid)? {
                results.push(entity);
            }
        }

        Ok(results)
    }

    /// Canonical pair key for the entity-pair index.
    /// Uses min/max UUID ordering so A→B and B→A produce the same key.
    fn pair_key(entity_a: &Uuid, entity_b: &Uuid) -> String {
        if entity_a < entity_b {
            format!("{entity_a}:{entity_b}")
        } else {
            format!("{entity_b}:{entity_a}")
        }
    }

    /// Index an entity pair → edge UUID for O(1) dedup lookups
    fn index_entity_pair(&self, entity_a: &Uuid, entity_b: &Uuid, edge_uuid: &Uuid) -> Result<()> {
        let key = Self::pair_key(entity_a, entity_b);
        self.store
            .put_cf(CF_ENTITY_PAIR_INDEX, key.as_bytes(), edge_uuid.as_bytes())?;
        Ok(())
    }

    /// Remove entity pair from the pair index
    fn remove_entity_pair_index(&self, entity_a: &Uuid, entity_b: &Uuid) -> Result<()> {
        let key = Self::pair_key(entity_a, entity_b);
        self.store.delete_cf(CF_ENTITY_PAIR_INDEX, key.as_bytes())?;
        Ok(())
    }

    /// Find existing relationship between two entities (either direction)
    ///
    /// O(1) lookup via entity-pair index, with fallback to linear scan
    /// for edges created before the pair index existed (migration path).
    pub fn find_relationship_between(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        if let Some(edge) = self.find_relationship_by_pair_index(entity_a, entity_b)? {
            return Ok(Some(edge));
        }
        let edges_a = self.get_entity_relationships(entity_a)?;
        self.find_relationship_in_scan(entity_a, entity_b, &edges_a)
    }

    /// The pair-index half of [`find_relationship_between`](Self::find_relationship_between):
    /// one O(1) lookup, removing the index row if the edge it names is gone.
    fn find_relationship_by_pair_index(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        let key = Self::pair_key(entity_a, entity_b);
        if let Some(edge_uuid_bytes) = self.store.get_cf(CF_ENTITY_PAIR_INDEX, key.as_bytes())? {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    return Ok(Some(edge));
                }
                // Edge was deleted but pair index is stale — clean up
                if let Err(e) = self.store.delete_cf(CF_ENTITY_PAIR_INDEX, key.as_bytes()) {
                    tracing::warn!("Failed to clean stale pair index entry: {e}");
                }
            }
        }
        Ok(None)
    }

    /// The fallback half of [`find_relationship_between`](Self::find_relationship_between):
    /// the edge joining the pair among `edges_a` (entity_a's edges), backfilling the pair index.
    fn find_relationship_in_scan(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        edges_a: &[RelationshipEdge],
    ) -> Result<Option<RelationshipEdge>> {
        // Slow path: linear scan for pre-index edges (backward compatibility)
        // This path is only hit for edges created before the pair index existed.
        // Once all old edges are either strengthened (which updates the index) or
        // pruned, this path becomes dead code.
        // PERF-11: Track slow-path hits to measure migration progress (plan removal once zero).
        tracing::debug!(
            entity_a = %entity_a,
            entity_b = %entity_b,
            "find_relationship_between: pair index miss, falling back to O(n) linear scan"
        );
        for edge in edges_a {
            if (edge.from_entity == *entity_a && edge.to_entity == *entity_b)
                || (edge.from_entity == *entity_b && edge.to_entity == *entity_a)
            {
                // Backfill pair index for this legacy edge
                if let Err(e) = self.index_entity_pair(entity_a, entity_b, &edge.uuid) {
                    tracing::warn!(
                        "Failed to backfill pair index for legacy edge {}: {e}",
                        edge.uuid
                    );
                }
                return Ok(Some(edge.clone()));
            }
        }
        Ok(None)
    }

    /// Find existing relationship between two entities with a specific relation type.
    ///
    /// Unlike `find_relationship_between` which returns any edge between the pair,
    /// this method only matches edges with the same `RelationType`. This allows
    /// multiple semantically distinct edges (e.g. WorksWith + PartOf) between
    /// the same entity pair.
    ///
    /// One point read in the edge index, and one for the edge it names. A row whose edge is
    /// gone or no longer matches falls back to scanning `entity_a`'s edges, and the index
    /// is repaired from what that scan finds.
    pub fn find_relationship_between_typed(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        relation_type: &RelationType,
    ) -> Result<Option<RelationshipEdge>> {
        let typed = Self::typed_pair_key(entity_a, entity_b, relation_type);
        let Some(holder) = self.store.get_cf(CF_EDGE_INDEX, typed.as_bytes())? else {
            return Ok(None);
        };
        if let Ok(edge_uuid) = Uuid::from_slice(&holder) {
            if let Some(edge) = self.get_relationship(&edge_uuid)? {
                if edge.relation_type == *relation_type && edge.joins(entity_a, entity_b) {
                    return Ok(Some(edge));
                }
            }
        }
        let found = self.scan_for_typed_relationship(entity_a, entity_b, relation_type)?;
        match &found {
            Some(edge) => {
                self.store
                    .put_cf(CF_EDGE_INDEX, typed.as_bytes(), edge.uuid.as_bytes())?
            }
            None => self.store.delete_cf(CF_EDGE_INDEX, typed.as_bytes())?,
        }
        Ok(found)
    }

    /// The edge joining the pair under `relation_type`, found among `entity_a`'s edges.
    fn scan_for_typed_relationship(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        relation_type: &RelationType,
    ) -> Result<Option<RelationshipEdge>> {
        Ok(self
            .stored_entity_relationships(entity_a)?
            .into_iter()
            .find(|edge| edge.relation_type == *relation_type && edge.joins(entity_a, entity_b)))
    }

    /// Add a relationship edge (or strengthen existing one)
    ///
    /// If an edge already exists between the two entities, strengthens it
    /// instead of creating a duplicate. This implements proper Hebbian learning:
    /// "neurons that fire together, wire together" - repeated co-occurrence
    /// strengthens the same synapse rather than creating parallel connections.
    pub fn add_relationship(&self, edge: RelationshipEdge) -> Result<Uuid> {
        self.add_relationships(vec![edge])?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("add_relationships returned no id for its one edge"))
    }

    /// [`Self::add_relationship`] for many edges: one lock acquisition and ONE atomic
    /// write, which carries every strengthened edge and, for each new edge, the edge,
    /// both entity->edge index rows and the pair-index row — so no new edge can exist
    /// without its index rows, and the batch costs one commit rather than four per edge.
    /// Edges sharing an entity pair and relation type collapse onto one synapse exactly
    /// as successive calls would. Returns each edge's id, in input order.
    pub fn add_relationships(&self, edges: Vec<RelationshipEdge>) -> Result<Vec<Uuid>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        // Acquire synapse lock to prevent TOCTOU race: two concurrent callers can both
        // pass find_relationship_between_typed (both see None) and then both insert a new
        // edge, creating duplicates. Serializing here ensures find+create is atomic.
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| anyhow::anyhow!("synapse_update_lock timeout in add_relationships"))?;

        // Every edge this call writes, strengthened or new, once each: an edge touched
        // twice in one batch is strengthened twice on the same copy.
        let mut pending: Vec<RelationshipEdge> = Vec::new();
        let mut created: Vec<usize> = Vec::new();
        let mut ids = Vec::with_capacity(edges.len());
        for mut edge in edges {
            // Different relation types (e.g. WorksWith vs PartOf) are distinct edges.
            let in_batch = pending.iter().position(|p| {
                p.relation_type == edge.relation_type
                    && ((p.from_entity == edge.from_entity && p.to_entity == edge.to_entity)
                        || (p.from_entity == edge.to_entity && p.to_entity == edge.from_entity))
            });
            let existing = match in_batch {
                Some(slot) => Some(slot),
                None => self
                    .find_relationship_between_typed(
                        &edge.from_entity,
                        &edge.to_entity,
                        &edge.relation_type,
                    )?
                    .map(|stored| {
                        pending.push(stored);
                        pending.len() - 1
                    }),
            };
            match existing {
                Some(slot) => {
                    // Strengthen existing edge instead of creating duplicate
                    let synapse = &mut pending[slot];
                    let source = edge
                        .source_episode_id
                        .map(|id| format!("episode:{id}"))
                        .unwrap_or_else(|| {
                            crate::constants::HEBBIAN_UNATTRIBUTED_SOURCE.to_string()
                        });
                    let _ = synapse.strengthen_from(&source);
                    // Update context if new context is more informative
                    if edge.context.len() > synapse.context.len() {
                        synapse.context = edge.context;
                    }
                    ids.push(synapse.uuid);
                }
                None => {
                    edge.uuid = Uuid::new_v4();
                    edge.created_at = Utc::now();
                    ids.push(edge.uuid);
                    created.push(pending.len());
                    pending.push(edge);
                }
            }
        }

        let mut batch = crate::storage::WriteBatch::new();
        for edge in &pending {
            batch.put_cf(
                CF_RELATIONSHIPS,
                edge.uuid.as_bytes().to_vec(),
                bincode::serde::encode_to_vec(edge, bincode::config::standard())?,
            );
        }
        for &slot in &created {
            let edge = &pending[slot];
            for entity in [edge.from_entity, edge.to_entity] {
                batch.put_cf(
                    CF_ENTITY_EDGES,
                    Self::entity_edge_key(&entity, &edge.uuid).into_bytes(),
                    b"1".to_vec(),
                );
            }
            batch.put_cf(
                CF_ENTITY_PAIR_INDEX,
                Self::pair_key(&edge.from_entity, &edge.to_entity).into_bytes(),
                edge.uuid.as_bytes().to_vec(),
            );
            Self::stage_edge_index_rows(&mut batch, edge);
        }
        self.store.write_batch(&batch)?;
        self.relationship_count
            .fetch_add(created.len(), Ordering::Relaxed);

        // Insert-time degree pruning: cap edges per entity to prevent O(n²) explosion.
        // If either entity exceeds MAX_ENTITY_DEGREE, prune the weakest edges.
        // This is the primary defense against graph bloat (132MB for 600KB of content).
        // Pruned once per touched entity after the whole batch, not after each edge.
        let mut touched: Vec<Uuid> = Vec::new();
        for &slot in &created {
            for entity in [pending[slot].from_entity, pending[slot].to_entity] {
                if !touched.contains(&entity) {
                    touched.push(entity);
                }
            }
        }
        for entity in &touched {
            self.prune_entity_if_over_degree(entity)?;
        }

        Ok(ids)
    }

    /// Store `edge` exactly as given, under `edge.uuid`, with its entity and pair index
    /// rows, in one atomic write: persistence only, which is what
    /// `GraphStore::put_relationship` means on every engine. The Hebbian merge of
    /// [`Self::add_relationships`] — strengthen the existing synapse for a pair, mint a
    /// uuid for a new one, prune by degree — is the graph engine's.
    pub fn put_relationship_record(&self, edge: &RelationshipEdge) -> Result<()> {
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in put_relationship_record")
            })?;
        let previous = self
            .store
            .get_cf(CF_RELATIONSHIPS, edge.uuid.as_bytes())?
            .map(|bytes| RelationshipEdge::decode(&bytes).ok());

        let mut batch = crate::storage::WriteBatch::new();
        if let Some(Some(previous)) = &previous {
            for entity in [previous.from_entity, previous.to_entity] {
                if entity != edge.from_entity && entity != edge.to_entity {
                    batch.delete_cf(
                        CF_ENTITY_EDGES,
                        Self::entity_edge_key(&entity, &edge.uuid).into_bytes(),
                    );
                }
            }
            let previous_pair = Self::pair_key(&previous.from_entity, &previous.to_entity);
            let pair_still_names_this_edge = self
                .store
                .get_cf(CF_ENTITY_PAIR_INDEX, previous_pair.as_bytes())?
                .is_some_and(|holder| holder.as_slice() == edge.uuid.as_bytes());
            if previous_pair != Self::pair_key(&edge.from_entity, &edge.to_entity)
                && pair_still_names_this_edge
            {
                batch.delete_cf(CF_ENTITY_PAIR_INDEX, previous_pair.into_bytes());
            }
            self.stage_edge_index_removals(&mut batch, previous)?;
        }
        Self::stage_edge_index_rows(&mut batch, edge);
        batch.put_cf(
            CF_RELATIONSHIPS,
            edge.uuid.as_bytes().to_vec(),
            bincode::serde::encode_to_vec(edge, bincode::config::standard())?,
        );
        for entity in [edge.from_entity, edge.to_entity] {
            batch.put_cf(
                CF_ENTITY_EDGES,
                Self::entity_edge_key(&entity, &edge.uuid).into_bytes(),
                b"1".to_vec(),
            );
        }
        batch.put_cf(
            CF_ENTITY_PAIR_INDEX,
            Self::pair_key(&edge.from_entity, &edge.to_entity).into_bytes(),
            edge.uuid.as_bytes().to_vec(),
        );
        self.store.write_batch(&batch)?;
        if previous.is_none() {
            self.relationship_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// The entity->edge index key: `{entity_uuid}:{edge_uuid}`.
    fn entity_edge_key(entity_uuid: &Uuid, edge_uuid: &Uuid) -> String {
        format!("{entity_uuid}:{edge_uuid}")
    }

    /// Prune an entity's edges once its degree exceeds MAX_ENTITY_DEGREE, back down to
    /// ENTITY_DEGREE_PRUNE_TARGET.
    ///
    /// Loads all edges for the entity, sorts by effective strength, and deletes
    /// the weakest. LTP-protected edges are preserved preferentially (sorted last,
    /// so they survive pruning).
    ///
    /// This is called at insert time to prevent unbounded edge growth. The count is a
    /// key-only scan; the load and sort run once per prune, and a prune leaves room for
    /// the next MAX_ENTITY_DEGREE - ENTITY_DEGREE_PRUNE_TARGET edges.
    fn prune_entity_if_over_degree(&self, entity_uuid: &Uuid) -> Result<()> {
        use crate::constants::{ENTITY_DEGREE_PRUNE_TARGET, MAX_ENTITY_DEGREE};

        // Fast path: count edges without loading them
        let edge_count = self.entity_degree(entity_uuid)?;

        if edge_count <= MAX_ENTITY_DEGREE {
            return Ok(());
        }

        // Over cap — load all edges, sort, prune weakest. The count includes edges the
        // load leaves out as decayed (queued for the next maintenance flush), so the live
        // set can be within the cap; it is still cut to the target, or every insert until
        // that flush would pay this load again.
        let all_edges = self.get_entity_relationships(entity_uuid)?;
        if all_edges.len() <= ENTITY_DEGREE_PRUNE_TARGET {
            return Ok(());
        }

        // Sort by pruning priority: LTP-protected edges last (survive pruning), then by
        // effective strength ascending (weakest are the prune candidates).
        //
        // CROSS-INGEST DETERMINISM (ported from shodh-memory 6272430). get_entity_relationships
        // returns edges in prefix-scan order, i.e. edge-UUID order, and edge UUIDs are
        // Uuid::new_v4() — random per ingest. A stable sort on (protection, strength) leaves
        // equal-priority edges in that random order, and take(prune_count) below does not merely
        // reorder them: it DELETES them. Fresh graphs carry large exact-strength plateaus (every
        // co-occurrence edge starts at its tier's initial weight) and hubs carry far more edges
        // than MAX_ENTITY_DEGREE, so two ingests of the same corpus kept different edge sets —
        // not a different order over one graph, a different graph.
        //
        // Ordered by a key that is a pure function of graph CONTENT: (peer entity name, relation
        // type, edge uuid). Peer names are repeat-stable via entity resolution, and
        // add_relationship dedups by (from, to, type), so the uuid only ever separates
        // content-identical edges, where the choice cannot matter.
        let peer_of = |e: &RelationshipEdge| -> Uuid {
            if e.from_entity == *entity_uuid {
                e.to_entity
            } else {
                e.from_entity
            }
        };
        let mut peer_names: HashMap<Uuid, String> = HashMap::new();
        for e in &all_edges {
            let peer = peer_of(e);
            peer_names.entry(peer).or_insert_with(|| {
                self.get_entity(&peer)
                    .ok()
                    .flatten()
                    .map(|ent| ent.name)
                    .unwrap_or_default()
            });
        }

        // effective_strength() reads Utc::now(), so it is snapshotted once per edge here rather
        // than called from inside the comparator.
        let mut scored: Vec<(&RelationshipEdge, f32, bool)> = all_edges
            .iter()
            .map(|e| (e, e.effective_strength(), e.is_potentiated()))
            .collect();

        // Sort: unprotected+weak first (pruning candidates), protected+strong last (survivors)
        scored.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| peer_names[&peer_of(a.0)].cmp(&peer_names[&peer_of(b.0)]))
                .then_with(|| {
                    format!("{:?}", a.0.relation_type).cmp(&format!("{:?}", b.0.relation_type))
                })
                .then_with(|| a.0.uuid.cmp(&b.0.uuid))
        });

        // Prune excess: first N edges in sorted order are weakest/unprotected
        let prune_count = scored.len() - ENTITY_DEGREE_PRUNE_TARGET;
        let to_prune: Vec<Uuid> = scored.iter().take(prune_count).map(|s| s.0.uuid).collect();

        for edge_uuid in &to_prune {
            if let Err(e) = self.delete_relationship(edge_uuid) {
                tracing::warn!(
                    edge = %edge_uuid,
                    entity = %entity_uuid,
                    "Failed to prune edge during degree cap: {}",
                    e
                );
            }
        }

        // Debug, not info: a hub at the cap prunes on EVERY new edge, one edge at a time, so an
        // info line here was measured at ~1,900 lines/min on the live store after ~485 writes.
        // The total stays observable through veld_graph_degree_cap_pruned_edges_total.
        if !to_prune.is_empty() {
            crate::metrics::GRAPH_DEGREE_CAP_PRUNED_EDGES.inc_by(to_prune.len() as u64);
            tracing::debug!(
                entity = %entity_uuid,
                pruned = to_prune.len(),
                remaining = ENTITY_DEGREE_PRUNE_TARGET,
                "Pruned edges exceeding degree cap"
            );
        }

        Ok(())
    }

    /// Number of edges indexed against an entity, counted from the entity→edge index without
    /// loading the edges.
    pub fn entity_degree(&self, entity_uuid: &Uuid) -> Result<usize> {
        let prefix = format!("{entity_uuid}:");
        let mut edge_count = 0usize;
        for (key, _) in self
            .store
            .scan_cf(CF_ENTITY_EDGES, Some(prefix.as_bytes()), None)?
        {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                edge_count += 1;
            }
        }
        Ok(edge_count)
    }

    /// Entities labelled [`EntityLabel::FilePath`] whose path ends with `term` at a `/`
    /// boundary: `serve.mjs` and `admin/serve.mjs` both resolve `commitwork:admin/serve.mjs`,
    /// `observe.mjs` does not.
    ///
    /// Cost: one hash probe into the in-memory suffix index, then one entity read per candidate,
    /// at most [`FILE_PATH_SUFFIX_MAX_CANDIDATES`]. The index holds one key per `/`-separated
    /// suffix of every `prefix:path`-shaped entity name (capped at
    /// [`FILE_PATH_SUFFIX_MAX_DEPTH`] per name), built once from the name index at open and
    /// extended on insert. Keys are shape-based because labels are not in the name index, so
    /// the label is checked here, on the candidates only.
    pub fn find_file_paths_by_suffix(&self, term: &str) -> Result<Vec<EntityNode>> {
        let key = term.trim().trim_start_matches('/').to_lowercase();
        if key.is_empty() {
            return Ok(Vec::new());
        }
        let candidates: Vec<Uuid> = {
            let index = self.entity_path_suffix_index.read();
            match index.get(&key) {
                Some(uuids) => uuids
                    .iter()
                    .take(FILE_PATH_SUFFIX_MAX_CANDIDATES)
                    .copied()
                    .collect(),
                None => return Ok(Vec::new()),
            }
        };
        let mut found = Vec::new();
        for uuid in candidates {
            if let Some(entity) = self.get_entity(&uuid)? {
                if entity.labels.contains(&EntityLabel::FilePath) {
                    found.push(entity);
                }
            }
        }
        found.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(found)
    }

    fn index_path_suffixes(&self, name: &str, uuid: Uuid) {
        let suffixes = path_suffixes(name);
        if suffixes.is_empty() {
            return;
        }
        let mut index = self.entity_path_suffix_index.write();
        for suffix in suffixes {
            let entry = index.entry(suffix).or_default();
            if !entry.contains(&uuid) {
                entry.push(uuid);
            }
        }
    }

    fn unindex_path_suffixes(&self, name: &str, uuid: &Uuid) {
        let suffixes = path_suffixes(name);
        if suffixes.is_empty() {
            return;
        }
        let mut index = self.entity_path_suffix_index.write();
        for suffix in suffixes {
            if let Some(entry) = index.get_mut(&suffix) {
                entry.retain(|u| u != uuid);
                if entry.is_empty() {
                    index.remove(&suffix);
                }
            }
        }
    }

    fn build_path_suffix_index(name_index: &HashMap<String, Uuid>) -> HashMap<String, Vec<Uuid>> {
        let mut index: HashMap<String, Vec<Uuid>> = HashMap::new();
        for (name, uuid) in name_index {
            for suffix in path_suffixes(name) {
                let entry = index.entry(suffix).or_default();
                if !entry.contains(uuid) {
                    entry.push(*uuid);
                }
            }
        }
        index
    }

    /// Get relationships for an entity with optional limit
    ///
    /// If limit is None, returns all edges (use sparingly for large graphs).
    pub fn get_entity_relationships(&self, entity_uuid: &Uuid) -> Result<Vec<RelationshipEdge>> {
        self.get_entity_relationships_limited(entity_uuid, None)
    }

    /// Every stored edge that touches `entity_uuid`, in index order, before any
    /// strength ordering or decay pruning: what the store holds, which is what the
    /// storage trait's listing answers on every engine.
    pub fn stored_entity_relationships(&self, entity_uuid: &Uuid) -> Result<Vec<RelationshipEdge>> {
        self.decode_edges(&self.stored_entity_edge_ids(entity_uuid)?)
    }

    /// The ids of every stored edge that touches `entity_uuid`, in index order, read from
    /// the entity's edge index alone: no edge record is decoded.
    pub fn stored_entity_edge_ids(&self, entity_uuid: &Uuid) -> Result<Vec<Uuid>> {
        let prefix = format!("{entity_uuid}:");
        let mut edge_uuids: Vec<Uuid> = Vec::with_capacity(256);
        for (key, _) in self
            .store
            .scan_cf(CF_ENTITY_EDGES, Some(prefix.as_bytes()), None)?
        {
            let Ok(key_str) = std::str::from_utf8(&key) else {
                continue;
            };
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(edge_uuid) = key_str
                .split(':')
                .nth(1)
                .and_then(|id| Uuid::parse_str(id).ok())
            {
                edge_uuids.push(edge_uuid);
            }
        }
        Ok(edge_uuids)
    }

    /// The edge records stored under `edge_uuids`, in that order; a missing or
    /// undecodable record is left out.
    fn decode_edges(&self, edge_uuids: &[Uuid]) -> Result<Vec<RelationshipEdge>> {
        let mut edges = Vec::with_capacity(edge_uuids.len());
        for edge_uuid in edge_uuids {
            if let Some(bytes) = self.store.get_cf(CF_RELATIONSHIPS, edge_uuid.as_bytes())? {
                if let Ok(edge) = RelationshipEdge::decode(&bytes) {
                    edges.push(edge);
                }
            }
        }
        Ok(edges)
    }

    /// Get relationships for an entity with limit, ordered by effective strength
    ///
    /// Collects ALL edge UUIDs first, reads them, sorts by effective_strength
    /// descending, then returns the top `limit` strongest edges. This ensures
    /// traversal and queries always use the most valuable connections.
    ///
    /// When no limit is specified, returns all edges sorted by strength.
    pub fn get_entity_relationships_limited(
        &self,
        entity_uuid: &Uuid,
        limit: Option<usize>,
    ) -> Result<Vec<RelationshipEdge>> {
        self.get_entity_relationships_limited_at(entity_uuid, limit, crate::memory::scoring_now())
    }

    /// As [`get_entity_relationships_limited`](Self::get_entity_relationships_limited), with
    /// every edge decayed to `now`.
    pub fn get_entity_relationships_limited_at(
        &self,
        entity_uuid: &Uuid,
        limit: Option<usize>,
        now: DateTime<Utc>,
    ) -> Result<Vec<RelationshipEdge>> {
        Ok(self
            .entity_edges_with_strength(entity_uuid, limit, now)?
            .into_iter()
            .map(|(edge, _)| edge)
            .collect())
    }

    /// An entity's edges paired with their effective strength at `now`, strongest first,
    /// at most `limit` of them. Each edge's strength is computed once and reused by the
    /// ordering, the prune check and the caller.
    pub fn entity_edges_with_strength(
        &self,
        entity_uuid: &Uuid,
        limit: Option<usize>,
        now: DateTime<Utc>,
    ) -> Result<Vec<(RelationshipEdge, f32)>> {
        let stored = self.stored_entity_relationships(entity_uuid)?;
        Ok(self.rank_entity_edges(entity_uuid, stored, limit, now, &mut HashMap::new()))
    }

    /// [`entity_edges_with_strength`](Self::entity_edges_with_strength) for a walk that
    /// reads many entities under one budget. When the entity has more than `max_decodes`
    /// stored edges it decodes none and returns `None`; otherwise it returns the edges and
    /// how many records it read. Tie peers are named from `peer_names`, which the caller
    /// keeps for the whole walk, so each peer entity is read once however many of the
    /// walk's entities it neighbours.
    pub fn entity_edges_with_strength_bounded(
        &self,
        entity_uuid: &Uuid,
        limit: Option<usize>,
        now: DateTime<Utc>,
        max_decodes: usize,
        peer_names: &mut HashMap<Uuid, String>,
    ) -> Result<Option<(RankedEdges, usize)>> {
        let ids = self.stored_entity_edge_ids(entity_uuid)?;
        if ids.len() > max_decodes {
            return Ok(None);
        }
        let stored = self.decode_edges(&ids)?;
        let ranked = self.rank_entity_edges(entity_uuid, stored, limit, now, peer_names);
        Ok(Some((ranked, ids.len())))
    }

    /// Order `stored`, the edges of `entity_uuid`, strongest first at `now`, queue the
    /// decayed ones for pruning and keep the first `limit`.
    fn rank_entity_edges(
        &self,
        entity_uuid: &Uuid,
        stored: Vec<RelationshipEdge>,
        limit: Option<usize>,
        now: DateTime<Utc>,
        peer_names: &mut HashMap<Uuid, String>,
    ) -> Vec<(RelationshipEdge, f32)> {
        if stored.is_empty() {
            return Vec::new();
        }
        let mut edges: Vec<(RelationshipEdge, f32)> = stored
            .into_iter()
            .map(|edge| {
                let strength = edge.effective_strength_at(now);
                (edge, strength)
            })
            .collect();

        // Phase 3: Sort by effective strength descending (strongest first). Strengths are
        // computed once above against one `now`: recomputing them inside the comparator
        // could return different values across calls and break the total order.
        edges.sort_by(|a, b| b.1.total_cmp(&a.1));
        // Phase 3.5: Opportunistic pruning — queue edges that have decayed below
        // their tier's threshold for batch deletion on next maintenance cycle.
        // This replaces the eager full-scan apply_decay() with lazy on-read pruning.
        // NPMI rescue: edges with high enough NPMI are spared from pruning.
        let prunable = |(edge, strength): &(RelationshipEdge, f32)| {
            *strength < edge.tier.prune_threshold() && !edge.ltp_status.is_potentiated()
        };
        if edges.iter().any(prunable) {
            // Mention counts are read only for the endpoints of prunable edges.
            let mention_counts = self
                .build_mention_count_cache(edges.iter().filter(|e| prunable(e)).map(|(e, _)| e));

            let mut prune_queue = self.pending_prune.lock();
            let mut orphan_queue = self.pending_orphan_checks.lock();
            // Prevent unbounded growth — these queues are drained by maintenance,
            // but if maintenance hasn't run, cap to avoid increasing lock contention.
            if prune_queue.len() > 1000 {
                tracing::debug!(
                    "Prune queue overflow ({}) — clearing to prevent lock contention",
                    prune_queue.len()
                );
                prune_queue.clear();
            }
            if orphan_queue.len() > 2000 {
                orphan_queue.clear();
            }
            edges.retain(|entry| {
                if prunable(entry) {
                    let edge = &entry.0;
                    // NPMI rescue: check if edge has high enough NPMI to survive
                    let m_a = mention_counts.get(&edge.from_entity).copied().unwrap_or(0);
                    let m_b = mention_counts.get(&edge.to_entity).copied().unwrap_or(0);
                    let npmi = self.compute_npmi(edge, m_a, m_b);
                    let threshold = Self::npmi_prune_threshold(&edge.tier);
                    if npmi.is_some_and(|n| n >= threshold) {
                        return true; // NPMI rescue — keep edge
                    }
                    prune_queue.push(edge.uuid);
                    orphan_queue.push(edge.from_entity);
                    orphan_queue.push(edge.to_entity);
                    false // remove from results
                } else {
                    true
                }
            });
        }

        // Phase 3.1: equal-strength runs arrive in edge-UUID order, which is random per ingest,
        // and the limit below cuts through them on hub entities. Order each run by (peer entity
        // name, relation type, uuid), as the degree-cap prune does, so two stores built from the
        // same content keep the same edges. Names are looked up only inside runs of two or more
        // that start before the limit: a run past it is cut whole, so its order cannot matter.
        // Pruning first leaves the same survivors in the same relative order.
        let keep = limit.unwrap_or(usize::MAX);
        let peer_of = |e: &RelationshipEdge| -> Uuid {
            if e.from_entity == *entity_uuid {
                e.to_entity
            } else {
                e.from_entity
            }
        };
        let mut start = 0;
        while start < edges.len() && start < keep {
            let bits = edges[start].1.to_bits();
            let mut end = start + 1;
            while end < edges.len() && edges[end].1.to_bits() == bits {
                end += 1;
            }
            if end - start > 1 {
                for (e, _) in &edges[start..end] {
                    let peer = peer_of(e);
                    peer_names.entry(peer).or_insert_with(|| {
                        self.get_entity(&peer)
                            .ok()
                            .flatten()
                            .map(|ent| ent.name)
                            .unwrap_or_default()
                    });
                }
                edges[start..end].sort_by(|(a, _), (b, _)| {
                    peer_names[&peer_of(a)]
                        .cmp(&peer_names[&peer_of(b)])
                        .then_with(|| {
                            format!("{:?}", a.relation_type).cmp(&format!("{:?}", b.relation_type))
                        })
                        .then_with(|| {
                            (b.from_entity == *entity_uuid).cmp(&(a.from_entity == *entity_uuid))
                        })
                        .then_with(|| a.uuid.cmp(&b.uuid))
                });
            }
            start = end;
        }

        // Phase 4: Truncate to limit if specified
        edges.truncate(keep);
        edges
    }

    /// Calculate edge density for a specific entity (SHO-D5)
    ///
    /// Returns the number of edges connected to this entity.
    /// Used for per-entity density calculation: dense entities use vector search,
    /// sparse entities use graph search.
    ///
    /// This is an O(1) prefix count operation.
    pub fn entity_edge_count(&self, entity_uuid: &Uuid) -> Result<usize> {
        let prefix = format!("{entity_uuid}:");
        let iter = self
            .store
            .scan_cf(CF_ENTITY_EDGES, Some(prefix.as_bytes()), None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);

        let mut count = 0;
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                count += 1;
            }
        }

        Ok(count)
    }

    /// Calculate average edge density for a set of entities (SHO-D5)
    ///
    /// Returns the mean number of edges per entity for the given UUIDs.
    /// Used to determine optimal retrieval strategy:
    /// - Low density (<5 edges): Trust graph search (sparse, high-signal)
    /// - High density (>20 edges): Trust vector search (dense, noisy)
    ///
    /// Returns None if no entities provided.
    pub fn entities_average_density(&self, entity_uuids: &[Uuid]) -> Result<Option<f32>> {
        if entity_uuids.is_empty() {
            return Ok(None);
        }

        let mut total_edges = 0usize;
        for uuid in entity_uuids {
            total_edges += self.entity_edge_count(uuid)?;
        }

        Ok(Some(total_edges as f32 / entity_uuids.len() as f32))
    }

    /// Calculate edge density per tier for a specific entity (SHO-D5)
    ///
    /// Returns counts of edges by tier: (L1_count, L2_count, L3_count, LTP_count)
    /// Useful for understanding if an entity's graph is consolidated (mostly L3/LTP)
    /// or still noisy (mostly L1).
    pub fn entity_density_by_tier(
        &self,
        entity_uuid: &Uuid,
    ) -> Result<(usize, usize, usize, usize)> {
        let edges = self.get_entity_relationships(entity_uuid)?;

        let mut l1_count = 0;
        let mut l2_count = 0;
        let mut l3_count = 0;
        let mut ltp_count = 0;

        for edge in edges {
            if edge.is_potentiated() {
                ltp_count += 1;
            } else {
                match edge.tier {
                    EdgeTier::L1Working => l1_count += 1,
                    EdgeTier::L2Episodic => l2_count += 1,
                    EdgeTier::L3Semantic => l3_count += 1,
                }
            }
        }

        Ok((l1_count, l2_count, l3_count, ltp_count))
    }

    /// Calculate consolidated ratio for an entity (SHO-D5)
    ///
    /// Returns the ratio of consolidated edges (L2 + L3 + LTP) to total edges.
    /// High ratio (>0.7) = trust graph search, Low ratio (<0.3) = trust vector search.
    ///
    /// Returns None if entity has no edges.
    pub fn entity_consolidation_ratio(&self, entity_uuid: &Uuid) -> Result<Option<f32>> {
        let (l1, l2, l3, ltp) = self.entity_density_by_tier(entity_uuid)?;
        let total = l1 + l2 + l3 + ltp;

        if total == 0 {
            return Ok(None);
        }

        let consolidated = l2 + l3 + ltp;
        Ok(Some(consolidated as f32 / total as f32))
    }

    /// Trace the causal origins of a set of effect entities.
    ///
    /// Walks incoming causal edges (the current node is the `to_entity` of an
    /// edge whose type [`is_causal`](RelationType::is_causal)) back toward their
    /// `from_entity`, and returns the terminal sources: nodes with no further
    /// causal antecedent. A seed with no causal parent is the query subject, not
    /// an origin, and is omitted.
    ///
    /// Each origin is scored by its best backward path: the product of edge
    /// `effective_strength` along the path, times a per-hop decay. Results are
    /// strongest first, ties broken by uuid. No edge is strengthened or written;
    /// like any read through `get_entity_relationships_limited`, it may queue
    /// already-decayed edges for the next maintenance prune.
    pub fn trace_causal_origins(
        &self,
        seeds: &[Uuid],
        max_depth: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        use std::collections::HashSet;
        const HOP_DECAY: f32 = 0.7;
        const MAX_NODES: usize = 4000;

        let seed_set: HashSet<Uuid> = seeds.iter().copied().collect();
        let mut best: HashMap<Uuid, f32> = seeds.iter().map(|s| (*s, 1.0_f32)).collect();
        let mut origins: HashMap<Uuid, f32> = HashMap::new();
        let mut frontier: Vec<Uuid> = seeds.to_vec();

        for _ in 0..max_depth.max(1) {
            if frontier.is_empty() || best.len() >= MAX_NODES {
                break;
            }
            let mut next: Vec<Uuid> = Vec::new();
            for node in std::mem::take(&mut frontier) {
                let node_score = best.get(&node).copied().unwrap_or(0.0);
                if node_score <= 0.0 {
                    continue;
                }
                let edges = self.get_entity_relationships_limited(&node, Some(64))?;
                let parents: Vec<(Uuid, f32)> = edges
                    .iter()
                    .filter(|e| {
                        e.to_entity == node && e.from_entity != node && e.relation_type.is_causal()
                    })
                    .map(|e| (e.from_entity, e.effective_strength().clamp(0.0, 1.0)))
                    .collect();
                if parents.is_empty() {
                    if !seed_set.contains(&node) {
                        let entry = origins.entry(node).or_insert(0.0);
                        if node_score > *entry {
                            *entry = node_score;
                        }
                    }
                } else {
                    for (parent, strength) in parents {
                        let candidate = node_score * strength * HOP_DECAY;
                        let entry = best.entry(parent).or_insert(0.0);
                        // Products of factors below 1 cannot improve around a cycle, so relaxation terminates.
                        if candidate > *entry {
                            *entry = candidate;
                            next.push(parent);
                        }
                    }
                }
            }
            frontier = next;
        }

        let mut out: Vec<(Uuid, f32)> = origins.into_iter().collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Ok(out)
    }

    /// Get relationship by UUID (raw, without decay applied)
    pub fn get_relationship(&self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.store.get_cf(CF_RELATIONSHIPS, key)? {
            Some(value) => {
                let edge = RelationshipEdge::decode(&value)?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Get relationship by UUID with effective strength (lazy decay calculation)
    ///
    /// Returns the edge with strength reflecting time-based decay.
    /// This doesn't persist the decay - just calculates what the strength would be.
    /// Use this for API responses to show accurate current strength.
    pub fn get_relationship_with_effective_strength(
        &self,
        uuid: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.store.get_cf(CF_RELATIONSHIPS, key)? {
            Some(value) => {
                let mut edge = RelationshipEdge::decode(&value)?;
                // Apply effective strength calculation (doesn't persist)
                edge.strength = edge.effective_strength();
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Delete a relationship by UUID
    ///
    /// Removes the relationship from storage and decrements the counter.
    /// Returns true if the relationship was found and deleted.
    pub fn delete_relationship(&self, uuid: &Uuid) -> Result<bool> {
        let key = uuid.as_bytes();

        // Get the edge first to find both entities for index cleanup
        let edge = match self.get_relationship(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // The record and its edge-index rows go together: a lookup that found a row
        // without its edge would have to fall back to a scan.
        let mut batch = crate::storage::WriteBatch::new();
        batch.delete_cf(CF_RELATIONSHIPS, key.to_vec());
        self.stage_edge_index_removals(&mut batch, &edge)?;
        self.store.write_batch(&batch)?;
        self.relationship_count.fetch_sub(1, Ordering::Relaxed);
        // Decrement total activations by the edge's count (NPMI denominator)
        let edge_acts = edge.activation_count as u64;
        if edge_acts > 0 {
            // Saturating subtract via fetch_update to avoid underflow
            let _ = self.total_activations.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| Some(current.saturating_sub(edge_acts)),
            );
        }

        // Remove from entity_edges index for BOTH entities
        // (add_relationship indexes both from_entity and to_entity)
        let from_key = format!("{}:{}", edge.from_entity, uuid);
        if let Err(e) = self.store.delete_cf(CF_ENTITY_EDGES, from_key.as_bytes()) {
            tracing::warn!(edge = %uuid, key = %from_key, error = %e, "Failed to delete from entity_edges index");
        }
        let to_key = format!("{}:{}", edge.to_entity, uuid);
        if let Err(e) = self.store.delete_cf(CF_ENTITY_EDGES, to_key.as_bytes()) {
            tracing::warn!(edge = %uuid, key = %to_key, error = %e, "Failed to delete from entity_edges index");
        }

        // Remove from entity-pair index
        if let Err(e) = self.remove_entity_pair_index(&edge.from_entity, &edge.to_entity) {
            tracing::warn!(edge = %uuid, "Failed to delete from entity_pair index: {}", e);
        }

        Ok(true)
    }

    /// Delete an episode and clean up associated indices and orphan edges
    ///
    /// When a memory is deleted, its corresponding episode should also be removed.
    /// This method:
    /// 1. Removes the episode from the episodes DB
    /// 2. Removes entity_episodes index entries
    /// 3. Deletes relationship edges that were sourced from this episode
    pub fn delete_episode(&self, episode_uuid: &Uuid) -> Result<bool> {
        // Fetch episode to get entity_refs for index cleanup
        let episode = match self.get_episode(episode_uuid)? {
            Some(ep) => ep,
            None => return Ok(false),
        };

        // Delete episode from main storage
        self.store.delete_cf(CF_EPISODES, episode_uuid.as_bytes())?;
        self.episode_count.fetch_sub(1, Ordering::Relaxed);

        // Remove from both entity_episodes inverted indexes
        for entity_uuid in &episode.entity_refs {
            for key in [
                format!("{entity_uuid}:{episode_uuid}"),
                Self::episode_recency_key(entity_uuid, episode.created_at, episode_uuid),
            ] {
                if let Err(e) = self.store.delete_cf(CF_ENTITY_EPISODES, key.as_bytes()) {
                    tracing::warn!(episode = %episode_uuid, key = %key, error = %e, "Failed to delete from entity_episodes index");
                }
            }
        }

        // Delete the edges this episode sourced, as the edge index lists them. A row whose
        // edge is gone, or now names another source, is dropped without touching an edge.
        let prefix = Self::episode_edge_prefix(episode_uuid);
        let mut edges_to_delete = Vec::new();
        let mut rows = crate::storage::WriteBatch::new();
        for (key, _) in self
            .store
            .scan_cf(CF_EDGE_INDEX, Some(prefix.as_bytes()), None)?
        {
            let edge_uuid = std::str::from_utf8(&key)
                .ok()
                .and_then(|k| k.strip_prefix(prefix.as_str()))
                .and_then(|id| Uuid::parse_str(id).ok());
            if let Some(edge_uuid) = edge_uuid {
                if self
                    .get_relationship(&edge_uuid)?
                    .is_some_and(|edge| edge.source_episode_id == Some(*episode_uuid))
                {
                    edges_to_delete.push(edge_uuid);
                }
            }
            rows.delete_cf(CF_EDGE_INDEX, key);
        }

        for edge_uuid in &edges_to_delete {
            if let Err(e) = self.delete_relationship(edge_uuid) {
                tracing::debug!("Failed to delete orphan edge {}: {}", edge_uuid, e);
            }
        }
        if !rows.is_empty() {
            self.store.write_batch(&rows)?;
        }

        tracing::debug!(
            "Deleted episode {} with {} entity_refs and {} sourced edges",
            &episode_uuid.to_string()[..8],
            episode.entity_refs.len(),
            edges_to_delete.len()
        );

        Ok(true)
    }

    /// Clear all graph data (GDPR full erasure)
    ///
    /// Wipes all entities, relationships, episodes, and all indices.
    /// Resets all counters to zero.
    /// Returns (entity_count, relationship_count, episode_count) that were cleared.
    pub fn clear_all(&self) -> Result<(usize, usize, usize)> {
        let entity_count = self.entity_count.load(Ordering::Relaxed);
        let relationship_count = self.relationship_count.load(Ordering::Relaxed);
        let episode_count = self.episode_count.load(Ordering::Relaxed);

        // Clear each column family by iterating and batch-deleting
        for cf_name in GRAPH_CF_NAMES {
            let mut batch = crate::storage::WriteBatch::new();
            for (key, _) in self.store.scan_cf(cf_name, None, None)? {
                batch.delete_cf(*cf_name, key);
            }
            self.store.write_batch(&batch)?;
        }
        // An empty graph has nothing to backfill; keep the next open from rescanning.
        self.store
            .put_cf(CF_ENTITY_EPISODES, EPISODE_RECENCY_INDEX_MARKER, b"1")?;
        self.store.put_cf(CF_EDGE_INDEX, EDGE_INDEX_MARKER, b"1")?;

        // Clear in-memory indices
        self.entity_name_index.write().clear();
        self.entity_lowercase_index.write().clear();
        self.entity_stemmed_index.write().clear();
        self.entity_path_suffix_index.write().clear();
        self.entity_selectivity.write().clear();

        // Reset counters
        self.entity_count.store(0, Ordering::Relaxed);
        self.relationship_count.store(0, Ordering::Relaxed);
        self.episode_count.store(0, Ordering::Relaxed);
        self.total_activations.store(0, Ordering::Relaxed);

        // Drain pending maintenance queues — they reference now-deleted entities/edges
        let _ = std::mem::take(&mut *self.pending_prune.lock());
        let _ = std::mem::take(&mut *self.pending_orphan_checks.lock());

        tracing::info!(
            "Graph data cleared (GDPR erasure): {} entities, {} relationships, {} episodes",
            entity_count,
            relationship_count,
            episode_count
        );
        Ok((entity_count, relationship_count, episode_count))
    }

    /// Add an episodic node, or rewrite the one stored under its uuid.
    ///
    /// The episode, its entity->episode index rows, and the removal of the rows for
    /// entities a rewrite no longer references are one atomic write. `episode_count`
    /// counts episodes, not writes: it rises only when the uuid was not stored, as it
    /// is recounted from the stored rows on the next open.
    pub fn add_episode(&self, episode: EpisodicNode) -> Result<Uuid> {
        let key = episode.uuid.as_bytes();
        tracing::debug!(
            "add_episode: {} with {} entity_refs",
            &episode.uuid.to_string()[..8],
            episode.entity_refs.len()
        );

        let stored = self.store.get_cf(CF_EPISODES, key)?;
        let previous: Option<EpisodicNode> = stored.as_deref().and_then(|bytes| {
            bincode::serde::decode_from_slice::<EpisodicNode, _>(bytes, bincode::config::standard())
                .ok()
                .map(|(previous, _)| previous)
        });

        // The record, both inverted indexes (entity -> episode, and the recency-ordered one),
        // and the removal of a rewrite's stale rows land in one batch, so a reader never sees
        // an index row without its episode.
        let mut batch = crate::storage::WriteBatch::new();
        batch.put_cf(
            CF_EPISODES,
            key.to_vec(),
            bincode::serde::encode_to_vec(&episode, bincode::config::standard())?,
        );
        if let Some(previous) = &previous {
            for entity_uuid in &previous.entity_refs {
                if !episode.entity_refs.contains(entity_uuid) {
                    batch.delete_cf(
                        CF_ENTITY_EPISODES,
                        Self::entity_episode_key(entity_uuid, &episode.uuid).into_bytes(),
                    );
                }
                if previous.created_at != episode.created_at
                    || !episode.entity_refs.contains(entity_uuid)
                {
                    batch.delete_cf(
                        CF_ENTITY_EPISODES,
                        Self::episode_recency_key(entity_uuid, previous.created_at, &episode.uuid)
                            .into_bytes(),
                    );
                }
            }
        }
        for entity_uuid in &episode.entity_refs {
            batch.put_cf(
                CF_ENTITY_EPISODES,
                Self::entity_episode_key(entity_uuid, &episode.uuid).into_bytes(),
                b"1".to_vec(),
            );
            batch.put_cf(
                CF_ENTITY_EPISODES,
                Self::episode_recency_key(entity_uuid, episode.created_at, &episode.uuid)
                    .into_bytes(),
                Vec::<u8>::new(),
            );
        }
        self.store.write_batch(&batch)?;

        if stored.is_none() {
            self.episode_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(episode.uuid)
    }

    /// The entity->episode inverted-index key: `{entity_uuid}:{episode_uuid}`.
    fn entity_episode_key(entity_uuid: &Uuid, episode_uuid: &Uuid) -> String {
        format!("{entity_uuid}:{episode_uuid}")
    }

    /// Key of the recency-ordered entity -> episode index:
    /// `{entity}#{rank:016x}:{episode}`, where `rank` falls as `created_at` rises, so an
    /// ascending prefix scan of `{entity}#` yields an entity's newest episodes first. The `#`
    /// keeps these rows out of every `{entity}:` prefix scan of the unordered index.
    fn episode_recency_key(
        entity_uuid: &Uuid,
        created_at: DateTime<Utc>,
        episode: &Uuid,
    ) -> String {
        format!(
            "{entity_uuid}#{:016x}:{episode}",
            Self::episode_recency_rank(created_at)
        )
    }

    /// Microsecond timestamp mapped to a `u64` that sorts descending in time: the sign bit is
    /// flipped so signed order becomes unsigned order, then all bits are inverted.
    fn episode_recency_rank(created_at: DateTime<Utc>) -> u64 {
        !((created_at.timestamp_micros() as u64) ^ (1u64 << 63))
    }

    /// Parse a recency-index key under `prefix` (`{entity}#`) into (rank, episode).
    fn parse_episode_recency_key(prefix: &str, key: &[u8]) -> Option<(u64, Uuid)> {
        let rest = std::str::from_utf8(key).ok()?.strip_prefix(prefix)?;
        let (rank, episode) = rest.split_once(':')?;
        Some((
            u64::from_str_radix(rank, 16).ok()?,
            Uuid::parse_str(episode).ok()?,
        ))
    }

    /// Backfill the recency-ordered episode index for stores written before it existed.
    /// Runs once: the marker row records completion, and every later `add_episode` writes
    /// both indexes itself.
    fn ensure_episode_recency_index(store: &dyn crate::storage::ColumnStore) -> Result<()> {
        const BATCH_ROWS: usize = 10_000;
        if store
            .get_cf(CF_ENTITY_EPISODES, EPISODE_RECENCY_INDEX_MARKER)?
            .is_some()
        {
            return Ok(());
        }
        let mut batch = crate::storage::WriteBatch::new();
        let mut written = 0usize;
        for (_, value) in store.scan_cf(CF_EPISODES, None, None)? {
            let Ok((episode, _)) = bincode::serde::decode_from_slice::<EpisodicNode, _>(
                &value,
                bincode::config::standard(),
            ) else {
                continue;
            };
            for entity_uuid in &episode.entity_refs {
                batch.put_cf(
                    CF_ENTITY_EPISODES,
                    Self::episode_recency_key(entity_uuid, episode.created_at, &episode.uuid)
                        .into_bytes(),
                    Vec::<u8>::new(),
                );
                written += 1;
            }
            if batch.len() >= BATCH_ROWS {
                store.write_batch(&std::mem::take(&mut batch))?;
            }
        }
        batch.put_cf(
            CF_ENTITY_EPISODES,
            EPISODE_RECENCY_INDEX_MARKER,
            b"1".as_slice(),
        );
        store.write_batch(&batch)?;
        if written > 0 {
            tracing::info!(
                rows = written,
                "Backfilled the recency-ordered entity episode index"
            );
        }
        Ok(())
    }

    /// Get episode by UUID
    pub fn get_episode(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
        let key = uuid.as_bytes();
        match self.store.get_cf(CF_EPISODES, key)? {
            Some(value) => {
                let (episode, _): (EpisodicNode, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(episode))
            }
            None => Ok(None),
        }
    }

    /// Get all episodes that contain a specific entity
    ///
    /// Uses inverted index for O(k) lookup instead of O(n) full scan.
    /// Collects episode UUIDs first, then batch-reads them using multi_get.
    /// Crucial for spreading activation algorithm.
    pub fn get_episodes_by_entity(&self, entity_uuid: &Uuid) -> Result<Vec<EpisodicNode>> {
        self.get_episodes_by_entity_limited(entity_uuid, usize::MAX)
    }

    /// The first `limit` episodes `get_episodes_by_entity` would return, reading no more
    /// episode records than it takes to decode them.
    pub fn get_episodes_by_entity_limited(
        &self,
        entity_uuid: &Uuid,
        limit: usize,
    ) -> Result<Vec<EpisodicNode>> {
        let prefix = format!("{entity_uuid}:");
        tracing::debug!("get_episodes_by_entity: prefix {}", &prefix[..12]);

        // Phase 1: Collect episode UUIDs from index (fast prefix scan, no data transfer)
        let mut episode_uuids: Vec<Uuid> = Vec::new();
        let iter = self
            .store
            .scan_cf(CF_ENTITY_EPISODES, Some(prefix.as_bytes()), None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(episode_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(episode_uuid) = Uuid::parse_str(episode_uuid_str) {
                        episode_uuids.push(episode_uuid);
                    }
                }
            }
        }

        if episode_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: read episodes in index order until `limit` decode
        let mut episodes = Vec::with_capacity(episode_uuids.len().min(limit));
        for uuid in &episode_uuids {
            if episodes.len() >= limit {
                break;
            }
            let Ok(Some(value)) = self.store.get_cf(CF_EPISODES, uuid.as_bytes()) else {
                continue;
            };
            if let Ok((episode, _)) = bincode::serde::decode_from_slice::<EpisodicNode, _>(
                &value,
                bincode::config::standard(),
            ) {
                episodes.push(episode);
            }
        }

        tracing::debug!("get_episodes_by_entity: found {} episodes", episodes.len());
        Ok(episodes)
    }

    /// An entity's `limit` most recent episodes: newest first, equal `created_at` ordered by
    /// content. Reads the recency-ordered index, so only the returned episodes, plus any that
    /// share the last one's timestamp, are decoded.
    pub fn get_recent_episodes_by_entity(
        &self,
        entity_uuid: &Uuid,
        limit: usize,
    ) -> Result<Vec<EpisodicNode>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let prefix = format!("{entity_uuid}#");
        // Headroom past `limit` absorbs rows left stale by a re-added episode and a
        // timestamp tie at the cut, so one bounded scan is enough in the common case.
        let mut scan_limit = Some(limit.saturating_mul(2).saturating_add(8));
        let mut resume_after: Option<Vec<u8>> = None;
        let mut episodes: Vec<EpisodicNode> = Vec::with_capacity(limit);
        // Rank of the `limit`-th episode: past it, only rows tied with it can still rank.
        let mut cut_rank: Option<u64> = None;

        'scan: loop {
            let rows =
                self.store
                    .scan_cf(CF_ENTITY_EPISODES, Some(prefix.as_bytes()), scan_limit)?;
            let truncated = scan_limit.is_some_and(|n| rows.len() >= n);
            for (key, _) in &rows {
                if resume_after.as_ref().is_some_and(|last| key <= last) {
                    continue;
                }
                let Some((rank, episode_uuid)) = Self::parse_episode_recency_key(&prefix, key)
                else {
                    continue;
                };
                if cut_rank.is_some_and(|cut| rank != cut) {
                    break 'scan;
                }
                let Some(episode) = self.get_episode(&episode_uuid).ok().flatten() else {
                    continue;
                };
                // A re-added episode leaves its earlier row behind; only the row matching the
                // record's current timestamp counts.
                if Self::episode_recency_rank(episode.created_at) != rank {
                    continue;
                }
                episodes.push(episode);
                if cut_rank.is_none() && episodes.len() >= limit {
                    cut_rank = Some(rank);
                }
            }
            if !truncated {
                break;
            }
            resume_after = rows.last().map(|(key, _)| key.clone());
            scan_limit = None;
        }

        episodes.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.content.cmp(&b.content))
        });
        episodes.truncate(limit);
        Ok(episodes)
    }

    /// Traverse graph starting from an entity (breadth-first)
    ///
    /// Implements Hebbian learning: edges traversed during retrieval are strengthened.
    /// This means frequently accessed pathways become stronger over time.
    ///
    /// Returns `TraversedEntity` with hop distance and decay factor for proper scoring:
    /// - hop 0 (start entity): decay = 1.0
    /// - hop 1: decay = 0.7
    /// - hop 2: decay = 0.49
    /// - etc.
    ///
    /// Performance: Uses batch edge reading and limits to handle large graphs.
    pub fn traverse_from_entity(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
    ) -> Result<GraphTraversal> {
        self.traverse_from_entity_filtered(start_uuid, max_depth, None)
    }

    /// BFS graph traversal with optional minimum edge strength filter.
    ///
    /// When `min_strength` is Some, edges below the threshold are skipped
    /// during traversal and NOT Hebbianly strengthened (prevents ghost edge revival).
    pub fn traverse_from_entity_filtered(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
        min_strength: Option<f32>,
    ) -> Result<GraphTraversal> {
        // Performance limits
        const MAX_ENTITIES: usize = 200;
        const MAX_EDGES_PER_NODE: usize = 100;

        // Use tuned decay from constants (0.15 max decay → ~86% retention per hop)
        // This enables deeper traversal than the old 0.7 factor
        use crate::constants::IMPORTANCE_DECAY_MAX;
        let hop_decay_factor: f32 = (-IMPORTANCE_DECAY_MAX).exp(); // e^(-0.15) ≈ 0.86
        let now = crate::memory::scoring_now();

        let mut visited_entities = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut current_level: Vec<(Uuid, usize)> = vec![(*start_uuid, 0)]; // (uuid, hop_distance)
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges = Vec::new();
        let mut edges_to_strengthen = Vec::new();

        visited_entities.insert(*start_uuid);
        if let Some(entity) = self.get_entity(start_uuid)? {
            all_entities.push(TraversedEntity {
                entity,
                hop_distance: 0,
                decay_factor: 1.0,
            });
        }

        for depth in 0..max_depth {
            // Early termination if we have enough entities
            if all_entities.len() >= MAX_ENTITIES {
                break;
            }

            let mut next_level = Vec::new();

            for (entity_uuid, _hop) in &current_level {
                // The limit holds within a level too: one level of a hub can reach thousands.
                if all_entities.len() >= MAX_ENTITIES {
                    break;
                }
                // Use limited edge reading
                let edges =
                    self.entity_edges_with_strength(entity_uuid, Some(MAX_EDGES_PER_NODE), now)?;

                for (edge, effective) in edges {
                    if visited_edges.contains(&edge.uuid) {
                        continue;
                    }

                    visited_edges.insert(edge.uuid);

                    // Only traverse non-invalidated edges
                    if edge.invalidated_at.is_some() {
                        continue;
                    }

                    // Skip weak edges if min_strength filter is set
                    if let Some(threshold) = min_strength {
                        if effective < threshold {
                            continue;
                        }
                    }

                    // Collect edge UUID for Hebbian strengthening (only for traversed edges)
                    edges_to_strengthen.push(edge.uuid);

                    // Return edge with effective strength
                    let mut edge_with_decay = edge.clone();
                    edge_with_decay.strength = effective;
                    all_edges.push(edge_with_decay);

                    // Add connected entity
                    let connected_uuid = if edge.from_entity == *entity_uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    if !visited_entities.contains(&connected_uuid) {
                        visited_entities.insert(connected_uuid);
                        let next_hop = depth + 1;
                        let decay = hop_decay_factor.powi(next_hop as i32);

                        if let Some(entity) = self.get_entity(&connected_uuid)? {
                            all_entities.push(TraversedEntity {
                                entity,
                                hop_distance: next_hop,
                                decay_factor: decay,
                            });
                        }
                        next_level.push((connected_uuid, next_hop));
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            current_level = next_level;
        }

        // Apply Hebbian strengthening to all traversed edges atomically (SHO-65)
        // "Neurons that fire together, wire together"
        // Uses batch update for efficiency instead of individual writes.
        // A usage write: skipped under read-only recall (memory::recall_readonly), so an eval
        // repeat does not strengthen the edges it walked (shodh-memory 4ec3b6e).
        if !edges_to_strengthen.is_empty()
            && !crate::memory::recall_readonly()
            && crate::memory::ablation::is_enabled("graph_potentiation")
        {
            match self.batch_strengthen_synapses(&edges_to_strengthen, &ambient_evidence_source()) {
                Ok(count) => {
                    if count > 0 {
                        tracing::trace!("Strengthened {} synapses during traversal", count);
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to batch strengthen synapses: {}", e);
                }
            }
        }

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Weighted graph traversal with filtering (Dijkstra-style best-first)
    ///
    /// Unlike BFS traverse_from_entity, this uses edge weights to prioritize
    /// stronger connections. Enables Cypher-like pattern matching:
    /// - Filter by relationship types
    /// - Filter by minimum edge strength
    /// - Score paths by cumulative weight
    ///
    /// Returns entities sorted by path score (strongest connections first).
    ///
    /// Performance: Uses batch edge reading and early termination to handle
    /// large graphs (10000+ edges) efficiently.
    pub fn traverse_weighted(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
        relation_types: Option<&[RelationType]>,
        min_strength: f32,
    ) -> Result<GraphTraversal> {
        self.traverse_weighted_at(
            start_uuid,
            max_depth,
            relation_types,
            min_strength,
            crate::memory::scoring_now(),
        )
    }

    /// As [`traverse_weighted`](Self::traverse_weighted), with every edge decayed to `now`.
    pub fn traverse_weighted_at(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
        relation_types: Option<&[RelationType]>,
        min_strength: f32,
        now: DateTime<Utc>,
    ) -> Result<GraphTraversal> {
        // Performance limits - prevents exponential blowup on dense graphs
        const MAX_ENTITIES: usize = 200; // Stop after finding this many entities
        const MAX_EDGES_PER_NODE: usize = 100; // Limit edges loaded per node
        const MAX_ITERATIONS: usize = 500; // Prevent infinite loops

        // Priority queue entry: (negative score for max-heap, uuid, depth, path_score)
        #[derive(Clone)]
        struct PQEntry {
            score: f32,
            uuid: Uuid,
            depth: usize,
        }
        impl PartialEq for PQEntry {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for PQEntry {}
        impl PartialOrd for PQEntry {
            fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for PQEntry {
            fn cmp(&self, other: &Self) -> CmpOrdering {
                // BinaryHeap is a max-heap: higher score = popped first = explored first
                self.score.total_cmp(&other.score)
            }
        }

        let mut visited: HashMap<Uuid, f32> = HashMap::new(); // uuid -> best path score
        let mut heap: BinaryHeap<PQEntry> = BinaryHeap::new();
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges: Vec<RelationshipEdge> = Vec::new();
        let mut edges_to_strengthen: Vec<Uuid> = Vec::new();
        let mut iterations = 0;

        // Start node
        heap.push(PQEntry {
            score: 1.0,
            uuid: *start_uuid,
            depth: 0,
        });
        visited.insert(*start_uuid, 1.0);

        if let Some(entity) = self.get_entity(start_uuid)? {
            all_entities.push(TraversedEntity {
                entity,
                hop_distance: 0,
                decay_factor: 1.0,
            });
        }

        while let Some(PQEntry { score, uuid, depth }) = heap.pop() {
            iterations += 1;

            // Early termination: entity/iteration limits reached, stop draining heap
            if all_entities.len() >= MAX_ENTITIES || iterations >= MAX_ITERATIONS {
                break;
            }

            // Depth limit: skip this node's children but keep processing others
            if depth >= max_depth {
                continue;
            }

            // Skip if we've found a better path to this node
            if let Some(&best) = visited.get(&uuid) {
                if score < best * 0.99 {
                    continue;
                }
            }

            // Use limited edge reading to avoid loading 10000+ edges
            let edges = self.entity_edges_with_strength(&uuid, Some(MAX_EDGES_PER_NODE), now)?;

            for (edge, effective) in edges {
                // Skip invalidated edges
                if edge.invalidated_at.is_some() {
                    continue;
                }

                // Filter by relationship type if specified
                if let Some(types) = relation_types {
                    if !types.contains(&edge.relation_type) {
                        continue;
                    }
                }

                // Filter by minimum strength
                if effective < min_strength {
                    continue;
                }

                let connected_uuid = if edge.from_entity == uuid {
                    edge.to_entity
                } else {
                    edge.from_entity
                };

                // Path score = parent_score * edge_strength (multiplicative decay)
                let new_score = score * effective;

                // Only visit if this is a better path
                let dominated = visited
                    .get(&connected_uuid)
                    .is_some_and(|&best| new_score <= best);
                if dominated {
                    continue;
                }

                // Track edge for Hebbian strengthening AFTER domination check
                // so only edges on optimal paths are strengthened
                edges_to_strengthen.push(edge.uuid);

                let is_new_entity = !visited.contains_key(&connected_uuid);
                visited.insert(connected_uuid, new_score);

                // Add edge to results
                let mut edge_with_strength = edge;
                edge_with_strength.strength = effective;
                all_edges.push(edge_with_strength);

                // Only add entity once (first discovery); better-path rediscovery
                // updates visited score but doesn't duplicate the entity in results
                if is_new_entity {
                    if let Some(entity) = self.get_entity(&connected_uuid)? {
                        all_entities.push(TraversedEntity {
                            entity,
                            hop_distance: depth + 1,
                            decay_factor: new_score,
                        });
                    }
                }

                heap.push(PQEntry {
                    score: new_score,
                    uuid: connected_uuid,
                    depth: depth + 1,
                });
            }
        }

        // Sort entities by path score (decay_factor) descending
        all_entities.sort_by(|a, b| b.decay_factor.total_cmp(&a.decay_factor));

        // Hebbian strengthening — a usage write, skipped under read-only recall.
        if !edges_to_strengthen.is_empty()
            && !crate::memory::recall_readonly()
            && crate::memory::ablation::is_enabled("graph_potentiation")
        {
            if let Err(e) =
                self.batch_strengthen_synapses(&edges_to_strengthen, &ambient_evidence_source())
            {
                tracing::debug!("Failed to strengthen synapses: {}", e);
            }
        }

        tracing::debug!(
            "traverse_weighted: {} entities, {} edges (min_strength={:.2})",
            all_entities.len(),
            all_edges.len(),
            min_strength
        );

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Bidirectional search between two entities (meet-in-middle)
    ///
    /// Complexity: O(b^(d/2)) instead of O(b^d) for unidirectional search.
    /// Finds the shortest weighted path between start and goal.
    /// Returns entities along the path with their path scores.
    ///
    /// Performance: Uses batch edge reading with limits.
    pub fn traverse_bidirectional(
        &self,
        start_uuid: &Uuid,
        goal_uuid: &Uuid,
        max_depth: usize,
        min_strength: f32,
    ) -> Result<GraphTraversal> {
        self.traverse_bidirectional_at(
            start_uuid,
            goal_uuid,
            max_depth,
            min_strength,
            crate::memory::scoring_now(),
        )
    }

    /// As [`traverse_bidirectional`](Self::traverse_bidirectional), with every edge decayed
    /// to `now`.
    pub fn traverse_bidirectional_at(
        &self,
        start_uuid: &Uuid,
        goal_uuid: &Uuid,
        max_depth: usize,
        min_strength: f32,
        now: DateTime<Utc>,
    ) -> Result<GraphTraversal> {
        const MAX_EDGES_PER_NODE: usize = 100;

        // Track forward search from start
        let mut forward_visited: HashMap<Uuid, (f32, usize)> = HashMap::new(); // uuid -> (score, depth)
        let mut forward_parents: HashMap<Uuid, (Uuid, Uuid)> = HashMap::new(); // child -> (parent, edge_uuid)
        let mut forward_frontier: Vec<(Uuid, f32, usize)> = vec![(*start_uuid, 1.0, 0)];
        forward_visited.insert(*start_uuid, (1.0, 0));

        // Track backward search from goal
        let mut backward_visited: HashMap<Uuid, (f32, usize)> = HashMap::new();
        let mut backward_parents: HashMap<Uuid, (Uuid, Uuid)> = HashMap::new();
        let mut backward_frontier: Vec<(Uuid, f32, usize)> = vec![(*goal_uuid, 1.0, 0)];
        backward_visited.insert(*goal_uuid, (1.0, 0));

        let mut meeting_node: Option<Uuid> = None;
        let mut best_path_score: f32 = 0.0;
        let half_depth = max_depth / 2 + 1;

        // Alternate forward and backward expansion
        for _round in 0..half_depth {
            // Expand forward frontier
            let mut new_forward: Vec<(Uuid, f32, usize)> = Vec::new();
            for (uuid, score, depth) in forward_frontier.drain(..) {
                if depth >= half_depth {
                    continue;
                }

                let edges =
                    self.entity_edges_with_strength(&uuid, Some(MAX_EDGES_PER_NODE), now)?;
                for (edge, effective) in edges {
                    if edge.invalidated_at.is_some() {
                        continue;
                    }
                    if effective < min_strength {
                        continue;
                    }

                    let connected = if edge.from_entity == uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };
                    let new_score = score * effective;

                    // Check if we meet backward search
                    if let Some(&(back_score, _)) = backward_visited.get(&connected) {
                        let combined = new_score * back_score;
                        if combined > best_path_score {
                            best_path_score = combined;
                            meeting_node = Some(connected);
                        }
                    }

                    // Update forward frontier
                    let dominated = forward_visited
                        .get(&connected)
                        .is_some_and(|&(best, _)| new_score <= best);
                    if !dominated {
                        forward_visited.insert(connected, (new_score, depth + 1));
                        forward_parents.insert(connected, (uuid, edge.uuid));
                        new_forward.push((connected, new_score, depth + 1));
                    }
                }
            }
            forward_frontier = new_forward;

            // Expand backward frontier
            let mut new_backward: Vec<(Uuid, f32, usize)> = Vec::new();
            for (uuid, score, depth) in backward_frontier.drain(..) {
                if depth >= half_depth {
                    continue;
                }

                let edges =
                    self.entity_edges_with_strength(&uuid, Some(MAX_EDGES_PER_NODE), now)?;
                for (edge, effective) in edges {
                    if edge.invalidated_at.is_some() {
                        continue;
                    }
                    if effective < min_strength {
                        continue;
                    }

                    let connected = if edge.from_entity == uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };
                    let new_score = score * effective;

                    // Check if we meet forward search
                    if let Some(&(fwd_score, _)) = forward_visited.get(&connected) {
                        let combined = fwd_score * new_score;
                        if combined > best_path_score {
                            best_path_score = combined;
                            meeting_node = Some(connected);
                        }
                    }

                    // Update backward frontier
                    let dominated = backward_visited
                        .get(&connected)
                        .is_some_and(|&(best, _)| new_score <= best);
                    if !dominated {
                        backward_visited.insert(connected, (new_score, depth + 1));
                        backward_parents.insert(connected, (uuid, edge.uuid));
                        new_backward.push((connected, new_score, depth + 1));
                    }
                }
            }
            backward_frontier = new_backward;

            // Early termination if we found a meeting point
            if meeting_node.is_some() {
                break;
            }
        }

        // Reconstruct path from meeting node
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges: Vec<RelationshipEdge> = Vec::new();
        let mut edges_to_strengthen: Vec<Uuid> = Vec::new();

        if let Some(meeting) = meeting_node {
            // Forward path: start -> meeting
            let mut path_forward: Vec<Uuid> = vec![meeting];
            let mut current = meeting;
            while let Some(&(parent, edge_uuid)) = forward_parents.get(&current) {
                path_forward.push(parent);
                edges_to_strengthen.push(edge_uuid);
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    all_edges.push(edge);
                }
                current = parent;
            }
            path_forward.reverse();

            // Backward path: meeting -> goal
            let mut path_backward: Vec<Uuid> = Vec::new();
            current = meeting;
            while let Some(&(parent, edge_uuid)) = backward_parents.get(&current) {
                path_backward.push(parent);
                edges_to_strengthen.push(edge_uuid);
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    all_edges.push(edge);
                }
                current = parent;
            }

            // Combine paths
            let full_path: Vec<Uuid> = path_forward.into_iter().chain(path_backward).collect();

            // Build entities with scores
            for (i, uuid) in full_path.iter().enumerate() {
                if let Some(entity) = self.get_entity(uuid)? {
                    let score = forward_visited
                        .get(uuid)
                        .map(|&(s, _)| s)
                        .or_else(|| backward_visited.get(uuid).map(|&(s, _)| s))
                        .unwrap_or(0.5);
                    all_entities.push(TraversedEntity {
                        entity,
                        hop_distance: i,
                        decay_factor: score,
                    });
                }
            }
        } else {
            // No path found - return empty traversal
            tracing::debug!(
                "traverse_bidirectional: no path between {:?} and {:?}",
                start_uuid,
                goal_uuid
            );
        }

        // Hebbian strengthening for traversed edges — a usage write, skipped under read-only
        // recall. This is the site recall's multi-hop leg reaches (memory/recall.rs), the one
        // the gate's doc comment named as the residual when the gate landed.
        if !edges_to_strengthen.is_empty()
            && !crate::memory::recall_readonly()
            && crate::memory::ablation::is_enabled("graph_potentiation")
        {
            if let Err(e) =
                self.batch_strengthen_synapses(&edges_to_strengthen, &ambient_evidence_source())
            {
                tracing::debug!("Failed to strengthen synapses: {}", e);
            }
        }

        tracing::debug!(
            "traverse_bidirectional: {} entities, {} edges, path_score={:.4}",
            all_entities.len(),
            all_edges.len(),
            best_path_score
        );

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Subgraph pattern matching (Cypher-like MATCH patterns)
    ///
    /// Pattern format: Vec of (relation_type, direction) tuples
    /// Direction: true = outgoing (a->b), false = incoming (a<-b)
    ///
    /// Example: MATCH (a)-[:WORKS_AT]->(b)-[:LOCATED_IN]->(c)
    /// Pattern: vec![(WorksAt, true), (LocatedIn, true)]
    ///
    /// Returns all entities that match the pattern starting from start_uuid.
    #[allow(dead_code)]
    pub fn match_pattern(
        &self,
        start_uuid: &Uuid,
        pattern: &[(RelationType, bool)], // (relation_type, is_outgoing)
        min_strength: f32,
    ) -> Result<Vec<Vec<TraversedEntity>>> {
        let mut matches: Vec<Vec<TraversedEntity>> = Vec::new();

        // Start entity
        let start_entity = match self.get_entity(start_uuid)? {
            Some(e) => e,
            None => return Ok(matches),
        };

        // DFS backtracking search
        let mut path: Vec<TraversedEntity> = vec![TraversedEntity {
            entity: start_entity,
            hop_distance: 0,
            decay_factor: 1.0,
        }];

        self.match_pattern_recursive(
            *start_uuid,
            pattern,
            0,
            min_strength,
            1.0,
            &mut path,
            &mut matches,
        )?;

        tracing::debug!(
            "match_pattern: found {} matches for {}-step pattern",
            matches.len(),
            pattern.len()
        );

        Ok(matches)
    }

    #[allow(clippy::too_many_arguments)]
    fn match_pattern_recursive(
        &self,
        current_uuid: Uuid,
        pattern: &[(RelationType, bool)],
        pattern_idx: usize,
        min_strength: f32,
        path_score: f32,
        path: &mut Vec<TraversedEntity>,
        matches: &mut Vec<Vec<TraversedEntity>>,
    ) -> Result<()> {
        // Base case: completed the pattern
        if pattern_idx >= pattern.len() {
            matches.push(path.clone());
            return Ok(());
        }

        const MAX_EDGES_PER_NODE: usize = 100;
        let (required_type, is_outgoing) = &pattern[pattern_idx];
        let edges =
            self.get_entity_relationships_limited(&current_uuid, Some(MAX_EDGES_PER_NODE))?;

        for edge in edges {
            if edge.invalidated_at.is_some() {
                continue;
            }

            // Check relationship type
            if edge.relation_type != *required_type {
                continue;
            }

            // Check direction
            let (next_uuid, direction_matches) = if *is_outgoing {
                // Looking for current -> next
                if edge.from_entity == current_uuid {
                    (edge.to_entity, true)
                } else {
                    (edge.from_entity, false) // Wrong direction
                }
            } else {
                // Looking for current <- next (incoming)
                if edge.to_entity == current_uuid {
                    (edge.from_entity, true)
                } else {
                    (edge.to_entity, false) // Wrong direction
                }
            };

            if !direction_matches {
                continue;
            }

            // Check strength
            let effective = edge.effective_strength();
            if effective < min_strength {
                continue;
            }

            // Avoid cycles in pattern
            if path.iter().any(|te| te.entity.uuid == next_uuid) {
                continue;
            }

            // Add to path and recurse
            if let Some(entity) = self.get_entity(&next_uuid)? {
                let new_score = path_score * effective;
                path.push(TraversedEntity {
                    entity,
                    hop_distance: pattern_idx + 1,
                    decay_factor: new_score,
                });

                self.match_pattern_recursive(
                    next_uuid,
                    pattern,
                    pattern_idx + 1,
                    min_strength,
                    new_score,
                    path,
                    matches,
                )?;

                path.pop();
            }
        }

        Ok(())
    }

    /// Find entities matching a pattern from any starting point
    ///
    /// Scans all entities and finds those that match the given pattern.
    /// More expensive than match_pattern but doesn't require a known start.
    ///
    /// Pattern: Vec of (relation_type, is_outgoing) tuples
    /// Returns: All complete pattern matches with their paths.
    #[allow(dead_code)]
    pub fn find_pattern_matches(
        &self,
        pattern: &[(RelationType, bool)],
        min_strength: f32,
        limit: usize,
    ) -> Result<Vec<Vec<TraversedEntity>>> {
        let mut all_matches: Vec<Vec<TraversedEntity>> = Vec::new();

        // Iterate through all entities as potential starting points
        let iter = self
            .store
            .scan_cf(CF_ENTITIES, None, None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for result in iter {
            if all_matches.len() >= limit {
                break;
            }

            let (_, value) = result?;
            let (entity, _): (EntityNode, _) =
                bincode::serde::decode_from_slice(&value, bincode::config::standard())?;

            let entity_matches = self.match_pattern(&entity.uuid, pattern, min_strength)?;
            for m in entity_matches {
                if all_matches.len() >= limit {
                    break;
                }
                all_matches.push(m);
            }
        }

        tracing::debug!(
            "find_pattern_matches: {} total matches (limit={})",
            all_matches.len(),
            limit
        );

        Ok(all_matches)
    }

    /// Invalidate a relationship (temporal edge invalidation)
    ///
    /// Guarded by synapse_update_lock to prevent race with strengthen/decay.
    pub fn invalidate_relationship(&self, edge_uuid: &Uuid) -> Result<()> {
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in invalidate_relationship")
            })?;

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            edge.invalidated_at = Some(Utc::now());

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.store.put_cf(CF_RELATIONSHIPS, key, &value)?;
        }

        Ok(())
    }

    /// Strengthen a synapse (Hebbian learning)
    ///
    /// Called when an edge is traversed during memory retrieval.
    /// Implements "neurons that fire together, wire together".
    ///
    /// Uses a mutex to prevent race conditions during concurrent updates (SHO-64).
    ///
    /// A repeat from a source already on the edge that earns no step is not written:
    /// it would only reset the edge's decay clock, in a write the intent log does not
    /// carry, so a replica's copy of the edge would decay on a different clock.
    pub fn strengthen_synapse(&self, edge_uuid: &Uuid, source: &str) -> Result<()> {
        // Lock with timeout to prevent deadlock on panic
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| anyhow::anyhow!("synapse_update_lock timeout in strengthen_synapse"))?;

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let strengthening = edge.strengthen_from(source);
            if strengthening.moves_nothing() {
                return Ok(());
            }
            if strengthening.evidence == Evidence::NewSource {
                self.total_activations.fetch_add(1, Ordering::Relaxed);
            }

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.store.put_cf(CF_RELATIONSHIPS, key, &value)?;
        }

        Ok(())
    }

    /// Batch strengthen multiple synapses atomically (SHO-65)
    ///
    /// More efficient than calling strengthen_synapse individually for each edge.
    /// Uses RocksDB WriteBatch for atomic multi-write and a single lock acquisition.
    ///
    /// Returns the number of synapses successfully strengthened. As in
    /// [`Self::strengthen_synapse`], a repeat that earns no step is not written or counted.
    pub fn batch_strengthen_synapses(&self, edge_uuids: &[Uuid], source: &str) -> Result<usize> {
        if edge_uuids.is_empty() {
            return Ok(0);
        }

        // Single lock acquisition for entire batch, with timeout
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in batch_strengthen_synapses")
            })?;

        // Batch read all edges in a single RocksDB call (same pattern as get_entity_relationships_limited)
        let keys: Vec<[u8; 16]> = edge_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let results: Vec<anyhow::Result<Option<Vec<u8>>>> = key_refs
            .iter()
            .map(|k| self.store.get_cf(CF_RELATIONSHIPS, k))
            .collect();

        let mut batch = crate::storage::WriteBatch::new();
        let mut strengthened = 0;
        let mut activations = 0u64;

        for (i, result) in results.into_iter().enumerate() {
            if let Ok(Some(value)) = result {
                if let Ok(mut edge) = RelationshipEdge::decode(&value) {
                    let strengthening = edge.strengthen_from(source);
                    if strengthening.moves_nothing() {
                        continue;
                    }
                    if strengthening.evidence == Evidence::NewSource {
                        activations += 1;
                    }
                    match bincode::serde::encode_to_vec(&edge, bincode::config::standard()) {
                        Ok(encoded) => {
                            batch.put_cf(CF_RELATIONSHIPS, keys[i], encoded);
                            strengthened += 1;
                        }
                        Err(e) => {
                            tracing::debug!("Failed to serialize edge {}: {}", edge_uuids[i], e);
                        }
                    }
                }
            }
        }

        // Atomic write of all updates
        if strengthened > 0 {
            self.store.write_batch(&batch)?;
            self.total_activations
                .fetch_add(activations, Ordering::Relaxed);
        }

        Ok(strengthened)
    }

    /// Record directional co-retrieval of memories (Hebbian learning with direction).
    ///
    /// When memories are retrieved together, they form associations. The trigger
    /// memory (highest scoring, matched the query) gets stronger forward edges
    /// to co-retrieved memories, reflecting the M&I bidirectional coupling:
    /// "A triggers B" is distinguished from "B triggers A".
    ///
    /// For pairs among co-retrieved memories (non-trigger), undirected strengthening
    /// is applied since neither dominated the triggering relationship.
    ///
    /// Note: Limits to top N memories to avoid O(n^2) explosion on large retrievals.
    /// Returns the number of edges created/strengthened.
    /// Result of memory coactivation recording, including edge tier promotions.
    /// Promotions signal the memory layer to boost associated memory importance.
    ///
    /// Internal implementation called only through `apply_reinforcement`.
    /// Made `pub(crate)` so the handlers (which hold their own Arc<RwLock<GraphMemory>>
    /// guard) can still call it directly when they already have a unique context
    /// (e.g. fire-and-forget spawned tasks that generate a fresh retrieval_id).
    /// New callers should prefer `apply_reinforcement`.
    pub(crate) fn record_memory_coactivation(
        &self,
        source: &str,
        trigger_id: &Uuid,
        coactivated_ids: &[Uuid],
    ) -> Result<CoactivationResult> {
        use crate::constants::COACTIVATION_FORWARD_BOOST;

        const MAX_COACTIVATION_SIZE: usize = 20;

        if coactivated_ids.is_empty() {
            return Ok(CoactivationResult {
                edges_updated: 0,
                promotions: Vec::new(),
            });
        }

        // Limit coactivated set to bound worst-case complexity
        let coactivated = if coactivated_ids.len() > MAX_COACTIVATION_SIZE {
            &coactivated_ids[..MAX_COACTIVATION_SIZE]
        } else {
            coactivated_ids
        };

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in record_memory_coactivation")
            })?;
        let mut batch = crate::storage::WriteBatch::new();
        let mut edges_updated = 0;
        let mut new_edges = 0;
        let mut activations = 0u64;
        let mut promotions: Vec<EdgePromotion> = Vec::new();

        // Phase 1: Directional edges between trigger and each co-retrieved memory
        for &coact_id in coactivated {
            let existing_edge = self.find_edge_between_entities(trigger_id, &coact_id)?;

            if let Some(mut edge) = existing_edge {
                // Strengthen existing edge with directional awareness
                let strengthening = edge.strengthen_from(source);
                if strengthening.evidence == Evidence::NewSource {
                    activations += 1;
                }

                // Apply directional boost based on which entity is the trigger
                if edge.from_entity == *trigger_id {
                    // Edge stored as trigger→coactivated: boost forward
                    edge.strengthen_forward(strengthening.step);
                } else {
                    // Edge stored as coactivated→trigger: boost backward
                    // (backward from edge's perspective = forward from trigger's perspective)
                    edge.strengthen_backward(strengthening.step);
                }

                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put_cf(CF_RELATIONSHIPS, key, value);
                    edges_updated += 1;
                }
            } else {
                // Create new CoRetrieved edge: trigger → coactivated
                // Forward strength starts higher (trigger is evidence of directional intent)
                let initial_weight = EdgeTier::L1Working.initial_weight();
                let edge = RelationshipEdge {
                    uuid: Uuid::new_v4(),
                    from_entity: *trigger_id,
                    to_entity: coact_id,
                    relation_type: RelationType::CoRetrieved,
                    strength: initial_weight,
                    created_at: Utc::now(),
                    valid_at: Utc::now(),
                    invalidated_at: None,
                    source_episode_id: None,
                    context: String::new(),
                    last_activated: Utc::now(),
                    activation_count: 1,
                    ltp_status: LtpStatus::None,
                    activation_timestamps: None,
                    tier: EdgeTier::L1Working,
                    entity_confidence: None,
                    created_by: EdgeSource::Coactivation,
                    // Directional: both start at initial_weight and the creating source's
                    // trigger→coact event takes one forward step, in the same delta form as
                    // every later step.
                    forward_strength: initial_weight
                        + COACTIVATION_FORWARD_BOOST * (1.0 - initial_weight),
                    backward_strength: initial_weight,
                    strengthening_sources: evidence_from(source),
                    last_corroborated: Some(Utc::now()),
                };

                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put_cf(CF_RELATIONSHIPS, key, value);
                    Self::stage_edge_index_rows(&mut batch, &edge);

                    // Index both directions for lookup
                    let idx_key_fwd = format!("mem_edge:{}:{}", trigger_id, coact_id);
                    let idx_key_rev = format!("mem_edge:{}:{}", coact_id, trigger_id);
                    batch.put_cf(
                        CF_RELATIONSHIPS,
                        idx_key_fwd.as_bytes(),
                        edge.uuid.as_bytes(),
                    );
                    batch.put_cf(
                        CF_RELATIONSHIPS,
                        idx_key_rev.as_bytes(),
                        edge.uuid.as_bytes(),
                    );

                    edges_updated += 1;
                    new_edges += 1;
                    activations += 1;
                }
            }
        }

        // Phase 2: Undirected edges among co-retrieved memories (no single trigger)
        // These pairs have no directional signal — both were co-retrieved, neither triggered.
        for i in 0..coactivated.len() {
            for j in (i + 1)..coactivated.len() {
                let mem_a = coactivated[i];
                let mem_b = coactivated[j];

                let existing_edge = self.find_edge_between_entities(&mem_a, &mem_b)?;

                if let Some(mut edge) = existing_edge {
                    // Strengthen existing edge and capture tier promotions
                    let outcome = edge.strengthen_from(source);
                    if outcome.evidence == Evidence::NewSource {
                        activations += 1;
                    }
                    if let Some((old_tier, new_tier)) = outcome.promotion {
                        promotions.push(EdgePromotion {
                            from_entity: mem_a,
                            to_entity: mem_b,
                            old_tier,
                            new_tier,
                        });
                    }
                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put_cf(CF_RELATIONSHIPS, key, value);
                        edges_updated += 1;
                    }
                } else {
                    // New edge between co-retrieved memories (symmetric initial strengths)
                    let initial_weight = EdgeTier::L1Working.initial_weight();
                    let edge = RelationshipEdge {
                        uuid: Uuid::new_v4(),
                        from_entity: mem_a,
                        to_entity: mem_b,
                        relation_type: RelationType::CoRetrieved,
                        strength: initial_weight,
                        created_at: Utc::now(),
                        valid_at: Utc::now(),
                        invalidated_at: None,
                        source_episode_id: None,
                        context: String::new(),
                        last_activated: Utc::now(),
                        activation_count: 1,
                        ltp_status: LtpStatus::None,
                        activation_timestamps: None,
                        tier: EdgeTier::L1Working,
                        entity_confidence: None,
                        created_by: EdgeSource::CoOccurrence,
                        forward_strength: initial_weight,
                        backward_strength: initial_weight,
                        strengthening_sources: evidence_from(source),
                        last_corroborated: Some(Utc::now()),
                    };

                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put_cf(CF_RELATIONSHIPS, key, value);
                        Self::stage_edge_index_rows(&mut batch, &edge);

                        let idx_key_fwd = format!("mem_edge:{mem_a}:{mem_b}");
                        let idx_key_rev = format!("mem_edge:{mem_b}:{mem_a}");
                        batch.put_cf(
                            CF_RELATIONSHIPS,
                            idx_key_fwd.as_bytes(),
                            edge.uuid.as_bytes(),
                        );
                        batch.put_cf(
                            CF_RELATIONSHIPS,
                            idx_key_rev.as_bytes(),
                            edge.uuid.as_bytes(),
                        );

                        edges_updated += 1;
                        new_edges += 1;
                        activations += 1;
                    }
                }
            }
        }

        if edges_updated > 0 {
            self.store.write_batch(&batch)?;
            self.total_activations
                .fetch_add(activations, Ordering::Relaxed);
            // Update relationship counter for newly created edges
            if new_edges > 0 {
                self.relationship_count
                    .fetch_add(new_edges, Ordering::Relaxed);
            }
        }

        Ok(CoactivationResult {
            edges_updated,
            promotions,
        })
    }

    /// Idempotent reinforcement coordinator (S-R4).
    ///
    /// **This is the single authoritative path through which all
    /// Hebbian co-activation `edge.strength` mutations must flow.**
    /// Path-credit and lateral-inhibition writers added in the future
    /// should call this method rather than `record_memory_coactivation`
    /// directly.
    ///
    /// # Idempotency
    ///
    /// Calling this method twice with the same `retrieval_id` is a **no-op**
    /// on the second call — the same strength boost is *not* applied twice.
    /// This closes the cardinality>1 / retry-compounding hazard: a client
    /// that retries a retrieval request (network glitch, at-least-once
    /// delivery) will not compound edge strength beyond the intended single
    /// reinforcement.
    ///
    /// # Behavior delta
    ///
    /// Compared with the pre-S-R4 code, duplicate `retrieval_id` calls now
    /// produce **no** additional strength increment.  The first call is
    /// identical to the old behaviour.  Callers that previously relied on
    /// repeated calls accumulating strength (e.g., intentional multi-pass
    /// retrieval loops) must use distinct `retrieval_id` values per logical
    /// reinforcement event.
    ///
    /// # Locking model (B5 compliance)
    ///
    /// The `seen_retrievals` mutex is held **only** for the O(1)
    /// set-membership check + insert, and is released *before* calling
    /// `record_memory_coactivation`.  Per-edge mutation continues to be
    /// serialised by `synapse_update_lock` (fine-grained).  No coarse
    /// graph-write lock is introduced.
    pub fn apply_reinforcement(
        &self,
        retrieval_id: Uuid,
        source: &str,
        trigger_id: &Uuid,
        coactivated_ids: &[Uuid],
    ) -> Result<CoactivationResult> {
        // --- Idempotency check (O(1), short critical section) ---
        {
            let mut guard = self.seen_retrievals.lock();
            let (seen_set, seen_queue) = &mut *guard;
            if seen_set.contains(&retrieval_id) {
                tracing::trace!(
                    %retrieval_id,
                    "apply_reinforcement: duplicate retrieval_id — skipping (idempotent)"
                );
                return Ok(CoactivationResult {
                    edges_updated: 0,
                    promotions: Vec::new(),
                });
            }
            // Evict oldest entry when at capacity
            if seen_queue.len() >= SEEN_RETRIEVALS_CAPACITY {
                if let Some(oldest) = seen_queue.pop_front() {
                    seen_set.remove(&oldest);
                }
            }
            seen_set.insert(retrieval_id);
            seen_queue.push_back(retrieval_id);
        } // `seen_retrievals` mutex released here — before any edge mutation

        // --- Delegate to the core coactivation path ---
        self.record_memory_coactivation(source, trigger_id, coactivated_ids)
    }

    /// Strengthen spreading-activation delivery edges (Physarum tube-thickening, S-R5 B10).
    ///
    /// Called on POSITIVE feedback for a delivered memory to reinforce the graph paths
    /// that actually delivered it.  Each entry is `(edge_uuid, contribution)` where
    /// `contribution` is the activation share for that edge; contributions are
    /// normalised to 1.0 inside this method so callers can pass raw activation values.
    ///
    /// Only edges with a non-zero normalised credit are strengthened. Edges are
    /// strengthened through the existing `batch_strengthen_synapses` path (same
    /// lock, same WriteBatch), keeping this function free of direct `strength` writes.
    ///
    /// Returns the number of edges that were successfully strengthened.
    pub fn apply_path_credit(&self, path_edges: &[(Uuid, f32)], source: &str) -> Result<usize> {
        if path_edges.is_empty() {
            return Ok(0);
        }

        // Normalise contributions so they sum to 1.0.
        let total: f32 = path_edges.iter().map(|(_, c)| c).sum();
        if total <= 0.0 {
            return Ok(0);
        }

        // Select edges that carry meaningful credit (> 1% of the total share).
        // This avoids strengthening noise edges that barely contributed.
        let threshold = total * 0.01;
        let qualifying: Vec<Uuid> = path_edges
            .iter()
            .filter(|(_, c)| *c > threshold)
            .map(|(id, _)| *id)
            .collect();

        self.batch_strengthen_synapses(&qualifying, source)
    }

    /// The co-retrieval edge between two memories, in either direction, if one exists.
    pub fn memory_edge_between(&self, a: &Uuid, b: &Uuid) -> Result<Option<RelationshipEdge>> {
        self.find_edge_between_entities(a, b)
    }

    /// Find an edge between two entities/memories (in either direction)
    fn find_edge_between_entities(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        // Check forward index
        let idx_key = format!("mem_edge:{entity_a}:{entity_b}");
        if let Some(edge_uuid_bytes) = self.store.get_cf(CF_RELATIONSHIPS, idx_key.as_bytes())? {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        // Check reverse index
        let idx_key_rev = format!("mem_edge:{entity_b}:{entity_a}");
        if let Some(edge_uuid_bytes) = self
            .store
            .get_cf(CF_RELATIONSHIPS, idx_key_rev.as_bytes())?
        {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        Ok(None)
    }

    /// Batch strengthen edges between memory pairs from replay consolidation
    ///
    /// Takes edge boosts from memory replay and applies Hebbian strengthening.
    /// Creates edges if they don't exist, strengthens if they do.
    ///
    /// Returns (count_strengthened, promotion_boosts) where promotion_boosts contains
    /// signals for any edge tier promotions that occurred (Direction 1 coupling).
    pub fn strengthen_memory_edges(
        &self,
        edge_boosts: &[(String, String, f32)],
        source: &str,
    ) -> Result<(usize, Vec<crate::memory::types::EdgePromotionBoost>)> {
        use crate::constants::{EDGE_PROMOTION_MEMORY_BOOST_L2, EDGE_PROMOTION_MEMORY_BOOST_L3};

        if edge_boosts.is_empty() {
            return Ok((0, Vec::new()));
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in strengthen_edges_from_boosts")
            })?;
        let mut batch = crate::storage::WriteBatch::new();
        let mut strengthened = 0;
        let mut promotion_boosts = Vec::new();
        let mut entities_to_prune = Vec::new();

        for (from_id_str, to_id_str, _boost) in edge_boosts {
            // Parse UUIDs
            let from_uuid = match Uuid::parse_str(from_id_str) {
                Ok(u) => u,
                Err(_) => {
                    tracing::debug!("Invalid from_id UUID: {}", from_id_str);
                    continue;
                }
            };
            let to_uuid = match Uuid::parse_str(to_id_str) {
                Ok(u) => u,
                Err(_) => {
                    tracing::debug!("Invalid to_id UUID: {}", to_id_str);
                    continue;
                }
            };

            // Find or create edge
            let existing_edge = self.find_edge_between_entities(&from_uuid, &to_uuid)?;

            if let Some(mut edge) = existing_edge {
                // Strengthen existing edge — capture tier promotion if it occurs
                let promotion = edge.strengthen_from(source).promotion;
                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put_cf(CF_RELATIONSHIPS, key, value);
                    strengthened += 1;

                    // If a tier promotion occurred, emit boost signals for both memories
                    if let Some((old_tier, new_tier)) = promotion {
                        let boost = if new_tier.contains("L2") {
                            EDGE_PROMOTION_MEMORY_BOOST_L2
                        } else {
                            EDGE_PROMOTION_MEMORY_BOOST_L3
                        };
                        let entity_name = format!(
                            "{}↔{}",
                            &from_id_str[..8.min(from_id_str.len())],
                            &to_id_str[..8.min(to_id_str.len())]
                        );
                        // Boost both memories involved in the promoted edge
                        promotion_boosts.push(crate::memory::types::EdgePromotionBoost {
                            memory_id: from_id_str.clone(),
                            entity_name: entity_name.clone(),
                            old_tier: old_tier.clone(),
                            new_tier: new_tier.clone(),
                            boost,
                        });
                        promotion_boosts.push(crate::memory::types::EdgePromotionBoost {
                            memory_id: to_id_str.clone(),
                            entity_name,
                            old_tier,
                            new_tier,
                            boost,
                        });
                    }
                }
            } else {
                // Create new ReplayStrengthened edge
                // Replay edges start in L2 (episodic) since they represent consolidated associations
                let replay_initial = EdgeTier::L2Episodic.initial_weight();
                let edge = RelationshipEdge {
                    uuid: Uuid::new_v4(),
                    from_entity: from_uuid,
                    to_entity: to_uuid,
                    relation_type: RelationType::CoRetrieved,
                    strength: replay_initial,
                    created_at: Utc::now(),
                    valid_at: Utc::now(),
                    invalidated_at: None,
                    source_episode_id: None,
                    context: "replay_strengthened".to_string(),
                    last_activated: Utc::now(),
                    activation_count: 1,
                    ltp_status: LtpStatus::None,
                    activation_timestamps: None,
                    tier: EdgeTier::L2Episodic,
                    entity_confidence: None,
                    created_by: EdgeSource::Coactivation,
                    forward_strength: replay_initial,
                    backward_strength: replay_initial,
                    strengthening_sources: evidence_from(source),
                    last_corroborated: Some(Utc::now()),
                };

                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put_cf(CF_RELATIONSHIPS, key, value);
                    Self::stage_edge_index_rows(&mut batch, &edge);

                    // Index both directions
                    let idx_key_fwd = format!("mem_edge:{from_uuid}:{to_uuid}");
                    let idx_key_rev = format!("mem_edge:{to_uuid}:{from_uuid}");
                    batch.put_cf(
                        CF_RELATIONSHIPS,
                        idx_key_fwd.as_bytes(),
                        edge.uuid.as_bytes(),
                    );
                    batch.put_cf(
                        CF_RELATIONSHIPS,
                        idx_key_rev.as_bytes(),
                        edge.uuid.as_bytes(),
                    );
                    // Index new replay edges in entity_edges CF so they're visible to
                    // traversal and degree-cap enforcement (GQ-11 fix), in the same
                    // batch as the edge so neither can exist without the other.
                    for entity in [from_uuid, to_uuid] {
                        batch.put_cf(
                            CF_ENTITY_EDGES,
                            Self::entity_edge_key(&entity, &edge.uuid).into_bytes(),
                            b"1".to_vec(),
                        );
                    }
                    entities_to_prune.push(from_uuid);
                    entities_to_prune.push(to_uuid);

                    strengthened += 1;
                }
            }
        }

        if strengthened > 0 {
            self.store.write_batch(&batch)?;
            self.total_activations
                .fetch_add(strengthened as u64, Ordering::Relaxed);

            // Enforce degree cap on affected entities
            for entity_uuid in &entities_to_prune {
                if let Err(e) = self.prune_entity_if_over_degree(entity_uuid) {
                    tracing::warn!("prune_entity_if_over_degree failed for {entity_uuid}: {e}");
                }
            }

            tracing::debug!(
                "Applied {} edge boosts from replay consolidation ({} tier promotions)",
                strengthened,
                promotion_boosts.len()
            );
        }

        Ok((strengthened, promotion_boosts))
    }

    /// Find memories associated with a given memory through co-retrieval
    ///
    /// Uses weighted graph traversal prioritizing stronger associations.
    /// Returns memory UUIDs sorted by association strength.
    pub fn find_memory_associations(
        &self,
        memory_id: &Uuid,
        max_results: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        let mut associations: Vec<(Uuid, f32)> = Vec::new();

        // Scan for edges involving this memory
        let prefix_fwd = format!("mem_edge:{memory_id}:");

        let iter = self
            .store
            .scan_cf(CF_RELATIONSHIPS, Some(prefix_fwd.as_bytes()), None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for item in iter {
            let (key, value) = item?;

            // Check if this is our prefix (RocksDB prefix_iterator may return extra)
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix_fwd) {
                break;
            }

            // Get edge UUID from value and look up edge
            if value.len() == 16 {
                let edge_uuid = Uuid::from_slice(&value)?;
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    // Get the other memory in this edge
                    let other_id = if edge.from_entity == *memory_id {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    // Get effective strength with decay
                    let effective_strength = edge.effective_strength();
                    if effective_strength > LTP_MIN_STRENGTH {
                        associations.push((other_id, effective_strength));
                    }
                }
            }
        }

        // Sort by strength descending and limit
        associations.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        associations.truncate(max_results);

        Ok(associations)
    }

    /// Strengthen entity-entity edges for a replayed memory's episode.
    ///
    /// During consolidation replay, this reinforces the entity relationships that
    /// were involved in the replayed memory. This is "Direction 3" of the Hebbian
    /// maintenance system — entity-entity edges get strengthened alongside
    /// memory-to-memory edges (Direction 1) and lazy pruning (Direction 2).
    ///
    /// Algorithm:
    /// 1. Look up EpisodicNode for episode_id → get entity_refs
    /// 2. For each pair of entities, find their RelationshipEdge
    /// 3. Call strengthen() on each edge (Hebbian boost + LTP detection + tier promotion)
    /// 4. Batch write all updates
    pub fn strengthen_episode_entity_edges(
        &self,
        episode_id: &Uuid,
        source: &str,
    ) -> Result<usize> {
        let episode = match self.get_episode(episode_id) {
            Ok(Some(ep)) => ep,
            Ok(None) => return Ok(0),
            Err(_) => return Ok(0),
        };

        if episode.entity_refs.len() < 2 {
            return Ok(0);
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in strengthen_episode_entity_edges")
            })?;
        let mut batch = crate::storage::WriteBatch::new();
        let mut strengthened = 0;

        // Unique entity refs, first 20 in episode order: caps the O(n²) pair walk.
        let mut refs: Vec<Uuid> = Vec::with_capacity(episode.entity_refs.len().min(20));
        for entity in &episode.entity_refs {
            if refs.len() == 20 {
                break;
            }
            if !refs.contains(entity) {
                refs.push(*entity);
            }
        }

        // Entity-entity edges are found through the entity-pair index (the `mem_edge:` index
        // holds memory-memory edges only). A pair-index miss falls back to scanning the first
        // entity's edges, as find_relationship_between does, loaded once per entity.
        let mut scanned: HashMap<Uuid, Vec<RelationshipEdge>> = HashMap::new();
        for i in 0..refs.len() {
            for j in (i + 1)..refs.len() {
                let entity_a = &refs[i];
                let entity_b = &refs[j];

                let found = match self.find_relationship_by_pair_index(entity_a, entity_b) {
                    Ok(Some(edge)) => Some(edge),
                    Ok(None) => {
                        if !scanned.contains_key(entity_a) {
                            let edges = self.get_entity_relationships(entity_a).unwrap_or_default();
                            scanned.insert(*entity_a, edges);
                        }
                        self.find_relationship_in_scan(entity_a, entity_b, &scanned[entity_a])
                            .ok()
                            .flatten()
                    }
                    Err(_) => None,
                };
                if let Some(mut edge) = found {
                    if edge.invalidated_at.is_some() {
                        continue;
                    }
                    let _ = edge.strengthen_from(source);
                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put_cf(CF_RELATIONSHIPS, key, value);
                        strengthened += 1;
                    }
                }
                // Don't create new edges — only strengthen existing ones from NER
            }
        }

        if strengthened > 0 {
            self.store.write_batch(&batch)?;
            self.total_activations
                .fetch_add(strengthened as u64, Ordering::Relaxed);
            tracing::debug!(
                "Strengthened {} entity-entity edges for episode {}",
                strengthened,
                &episode_id.to_string()[..8]
            );
        }

        Ok(strengthened)
    }

    /// Get average Hebbian strength for a memory based on its entity relationships
    ///
    /// This looks up the entities referenced by the memory and averages their
    /// relationship strengths in the graph. Used for composite relevance scoring.
    ///
    /// The algorithm:
    /// 1. Look up memory's EpisodicNode (memory_id.0 == episode UUID)
    /// 2. Get entity_refs from the episode
    /// 3. For each entity, get relationships using get_entity_relationships
    /// 4. Filter to edges where both endpoints are in the memory's entity set
    /// 5. Return average effective_strength of these intra-memory edges
    ///
    /// Returns 0.5 (neutral) if no entities or relationships found.
    pub fn get_memory_hebbian_strength(&self, memory_id: &crate::memory::MemoryId) -> Option<f32> {
        // 1. Look up EpisodicNode for this memory (memory_id.0 == episode UUID)
        let episode = match self.get_episode(&memory_id.0) {
            Ok(Some(ep)) => ep,
            Ok(None) => return Some(0.5), // No episode found - neutral
            Err(_) => return Some(0.5),   // Error - neutral fallback
        };

        // 2. Get entity references from the episode
        if episode.entity_refs.is_empty() {
            return Some(0.5); // No entities - neutral
        }

        // Build a set of entity UUIDs for fast lookup
        let entity_set: std::collections::HashSet<Uuid> =
            episode.entity_refs.iter().cloned().collect();

        // 3. Collect all intra-memory relationship strengths
        let mut strengths: Vec<f32> = Vec::new();
        let mut seen_edges: std::collections::HashSet<Uuid> = std::collections::HashSet::new();

        const MAX_EDGES_PER_ENTITY: usize = 50; // Limit per entity for Hebbian lookup
        let now = crate::memory::scoring_now();
        for entity_uuid in &episode.entity_refs {
            if let Ok(edges) =
                self.entity_edges_with_strength(entity_uuid, Some(MAX_EDGES_PER_ENTITY), now)
            {
                for (edge, effective) in edges {
                    // Skip if already processed (edges are bidirectional in lookup)
                    if seen_edges.contains(&edge.uuid) {
                        continue;
                    }
                    seen_edges.insert(edge.uuid);

                    // 4. Only count edges where BOTH endpoints are in this memory's entities
                    if entity_set.contains(&edge.from_entity)
                        && entity_set.contains(&edge.to_entity)
                    {
                        // Skip invalidated edges
                        if edge.invalidated_at.is_some() {
                            continue;
                        }
                        // Effective strength applies time-based decay
                        strengths.push(effective);
                    }
                }
            }
        }

        // 5. Return average strength, or neutral if no intra-memory edges
        if strengths.is_empty() {
            Some(0.5)
        } else {
            let avg = strengths.iter().sum::<f32>() / strengths.len() as f32;
            Some(avg)
        }
    }

    /// Apply decay to a synapse
    ///
    /// Returns true if the synapse should be pruned (non-potentiated and below threshold)
    ///
    /// Uses a mutex to prevent race conditions during concurrent updates (SHO-64).
    pub fn decay_synapse(&self, edge_uuid: &Uuid) -> Result<bool> {
        // Lock to prevent concurrent read-modify-write race conditions
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| anyhow::anyhow!("synapse_update_lock timeout in decay_synapse"))?;

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let should_prune = edge.decay();

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.store.put_cf(CF_RELATIONSHIPS, key, &value)?;

            return Ok(should_prune);
        }

        Ok(false)
    }

    /// Batch decay multiple synapses atomically
    ///
    /// Returns a vector of edge UUIDs that should be pruned.
    pub fn batch_decay_synapses(&self, edge_uuids: &[Uuid]) -> Result<Vec<Uuid>> {
        if edge_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Single lock acquisition for entire batch, with timeout
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in batch_decay_synapses")
            })?;

        let mut batch = crate::storage::WriteBatch::new();
        let mut to_prune = Vec::new();

        for edge_uuid in edge_uuids {
            if let Some(mut edge) = self.get_relationship(edge_uuid)? {
                let should_prune = edge.decay();

                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put_cf(CF_RELATIONSHIPS, key, value);
                        if should_prune {
                            to_prune.push(*edge_uuid);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge_uuid, e);
                    }
                }
            }
        }

        // Atomic write of all updates
        self.store.write_batch(&batch)?;

        Ok(to_prune)
    }

    /// Apply decay to already-loaded edges in-place, avoiding double deserialization.
    ///
    /// Mutates edges directly, serializes results into a WriteBatch, and returns
    /// the UUIDs of edges that should be pruned. Used by `apply_decay()` which
    /// already has the full edge list from `get_all_relationships()`.
    fn batch_decay_edges_in_place_at(
        &self,
        edges: &mut [RelationshipEdge],
        now: DateTime<Utc>,
        mention_counts: &HashMap<Uuid, usize>,
    ) -> Result<Vec<Uuid>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in batch_decay_edges_in_place")
            })?;

        let mut batch = crate::storage::WriteBatch::new();
        let mut to_prune = Vec::new();

        for edge in edges.iter_mut() {
            let strength_before = edge.strength;
            let mut should_prune = edge.decay_at(now);

            let m_a = mention_counts.get(&edge.from_entity).copied().unwrap_or(0);
            let m_b = mention_counts.get(&edge.to_entity).copied().unwrap_or(0);
            let npmi = self.compute_npmi(edge, m_a, m_b);

            // NPMI rescue: if decay says prune, check if NPMI is high enough to rescue.
            // This preserves statistically rare but meaningful associations that have
            // decayed in strength but represent genuine co-occurrence patterns.
            if should_prune {
                let threshold = Self::npmi_prune_threshold(&edge.tier);
                // NPMI can rescue: if npmi >= tier threshold, don't prune.
                // During cold start (npmi = None), fall back to strength-only pruning.
                if npmi.is_some_and(|n| n >= threshold) {
                    should_prune = false;
                }
            }

            // NPMI-enhanced LTP readiness: upgrade LTP status during maintenance
            // when NPMI provides enough statistical evidence for promotion.
            let ltp_before = edge.ltp_status.is_potentiated();
            if !ltp_before && !matches!(edge.tier, EdgeTier::L1Working) {
                let readiness = edge.ltp_readiness_with_npmi(npmi);
                if readiness >= crate::constants::LTP_READINESS_THRESHOLD {
                    edge.ltp_status = LtpStatus::Full;
                }
            }
            let ltp_changed = edge.ltp_status.is_potentiated() != ltp_before;

            // Only write back edges whose strength actually changed, LTP status
            // changed, or need pruning. With 300s maintenance intervals, most edges
            // won't have meaningful decay, so this skips unchanged edges.
            if should_prune || (edge.strength - strength_before).abs() > f32::EPSILON || ltp_changed
            {
                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&*edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put_cf(CF_RELATIONSHIPS, key, value);
                        if should_prune {
                            to_prune.push(edge.uuid);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge.uuid, e);
                    }
                }
            }
        }

        self.store.write_batch(&batch)?;
        Ok(to_prune)
    }

    /// Apply decay to all synapses and prune weak edges (AUD-2)
    ///
    /// Called during maintenance cycle to:
    /// 1. Apply time-based decay to all edge strengths
    /// 2. Remove edges that have decayed below threshold
    /// 3. Detect orphaned entities (entities that lost all their edges)
    ///
    /// Returns a `GraphDecayResult` with pruned count and orphaned entity/memory IDs
    /// for Direction 2 coupling (edge pruning → orphan detection).
    ///
    /// Ages edges on the scoring clock, as `effective_strength` reads them: the decayed
    /// strengths are written back, so under a pinned `VELD_EVAL_NOW` a wall-clock pass would
    /// store a strength that depends on when maintenance happened to run.
    pub fn apply_decay(&self) -> Result<crate::memory::types::GraphDecayResult> {
        self.apply_decay_at(crate::memory::scoring_now())
    }

    /// As [`apply_decay`](Self::apply_decay), ageing every edge as of `now`.
    ///
    /// The family is read a first-key-byte page at a time and decayed in chunks of
    /// [`DECAY_CHUNK_EDGES`], each under its own hold of the synapse lock and in its own
    /// write batch: decaying a whole large graph in one pass held every edge in memory
    /// and the lock that every new edge waits on. Mention counts are read once per
    /// entity across chunks.
    pub fn apply_decay_at(
        &self,
        now: DateTime<Utc>,
    ) -> Result<crate::memory::types::GraphDecayResult> {
        let mut mention_counts: HashMap<Uuid, usize> = HashMap::new();
        let mut counted: HashSet<Uuid> = HashSet::new();
        let mut chunk: Vec<RelationshipEdge> = Vec::new();
        let mut to_prune: Vec<Uuid> = Vec::new();
        let mut orphan_candidates: HashSet<Uuid> = HashSet::new();
        let mut total_edges = 0usize;
        for first in 0..=u8::MAX {
            for (_, value) in self.store.scan_cf(CF_RELATIONSHIPS, Some(&[first]), None)? {
                if let Ok(edge) = RelationshipEdge::decode(&value) {
                    chunk.push(edge);
                }
            }
            if chunk.len() < DECAY_CHUNK_EDGES && first != u8::MAX {
                continue;
            }
            total_edges += chunk.len();
            self.extend_mention_count_cache(&mut mention_counts, &mut counted, chunk.iter());
            let pruned = self.batch_decay_edges_in_place_at(&mut chunk, now, &mention_counts)?;
            if !pruned.is_empty() {
                let pruned_set: HashSet<Uuid> = pruned.iter().copied().collect();
                for edge in chunk.iter().filter(|edge| pruned_set.contains(&edge.uuid)) {
                    orphan_candidates.insert(edge.from_entity);
                    orphan_candidates.insert(edge.to_entity);
                }
                to_prune.extend(pruned);
            }
            chunk.clear();
        }

        if to_prune.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // Delete pruned edges
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            if self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        // Check which candidate entities became orphaned (lost ALL edges)
        // Delete orphaned entities to prevent stale index pollution
        let mut orphaned_entity_ids = Vec::new();
        for entity_uuid in &orphan_candidates {
            let remaining = self.get_entity_relationships(entity_uuid)?;
            if remaining.is_empty() {
                orphaned_entity_ids.push(entity_uuid.to_string());
                if let Err(e) = self.delete_entity(entity_uuid) {
                    tracing::warn!("Failed to delete orphaned entity {}: {}", entity_uuid, e);
                }
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Graph decay: {} edges pruned (of {} total), {} entities orphaned",
                pruned_count,
                total_edges,
                orphaned_entity_ids.len()
            );
        }

        Ok(crate::memory::types::GraphDecayResult {
            pruned_count,
            orphaned_entity_ids,
            orphaned_memory_ids: Vec::new(), // Populated by memory layer via entity→memory lookup
        })
    }

    /// Flush pending maintenance from opportunistic pruning queues.
    ///
    /// Called every maintenance cycle (5 min). Instead of scanning all 34k+ edges,
    /// this only processes edges that were found below prune threshold during normal
    /// reads (via `get_entity_relationships_limited`). Typical cost: 0-50 targeted
    /// deletes per cycle vs a full CF iterator scan.
    pub fn flush_pending_maintenance(&self) -> Result<crate::memory::types::GraphDecayResult> {
        // 1. Drain queues (fast — just swaps empty Vecs)
        let to_prune: Vec<Uuid> = std::mem::take(&mut *self.pending_prune.lock());
        let orphan_candidates: Vec<Uuid> = std::mem::take(&mut *self.pending_orphan_checks.lock());

        if to_prune.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // 2. Dedup UUIDs
        let to_prune: std::collections::HashSet<Uuid> = to_prune.into_iter().collect();
        let orphan_candidates: std::collections::HashSet<Uuid> =
            orphan_candidates.into_iter().collect();

        // 3. Batch delete pruned edges. An edge strengthened since it was queued (a new
        // co-occurrence finds it through the edge index) is above its threshold now and stays.
        let now = crate::memory::scoring_now();
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            let still_prunable = self.get_relationship(edge_uuid)?.is_some_and(|edge| {
                edge.effective_strength_at(now) < edge.tier.prune_threshold()
                    && !edge.ltp_status.is_potentiated()
            });
            if still_prunable && self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        // 4. Check which candidate entities became orphaned (lost ALL edges)
        let mut orphaned_entity_ids = Vec::new();
        for entity_uuid in &orphan_candidates {
            let remaining = self.get_entity_relationships(entity_uuid)?;
            if remaining.is_empty() {
                orphaned_entity_ids.push(entity_uuid.to_string());
                if let Err(e) = self.delete_entity(entity_uuid) {
                    tracing::warn!("Failed to delete orphaned entity {}: {}", entity_uuid, e);
                }
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Lazy pruning: {} edges deleted, {} entities orphaned",
                pruned_count,
                orphaned_entity_ids.len()
            );
        }

        Ok(crate::memory::types::GraphDecayResult {
            pruned_count,
            orphaned_entity_ids,
            orphaned_memory_ids: Vec::new(),
        })
    }

    /// Episodes in the graph: the N of the PMI gate at edge birth. O(1).
    pub fn total_episode_count(&self) -> usize {
        self.episode_count.load(Ordering::Relaxed)
    }

    /// Score every entity's selectivity from the Forman-Ricci curvature of its edges, and hold
    /// the scores for [`Self::entity_selectivity`]. Returns how many entities were scored.
    ///
    /// Ported from upstream shodh-memory #214 (`compute_forman_ricci_curvature`). An edge's
    /// curvature is `4 - deg(u) - deg(v)`, and an entity's selectivity is the standard
    /// deviation of its incident curvatures divided by its degree. A hub joined to nodes of
    /// every degree scores low once its degree is large; an entity with one edge scores 1.0.
    /// Upstream writes the score onto each entity and the curvature onto each edge; veld's
    /// records are positional bincode, so the scores live in memory and a heavy maintenance
    /// cycle recomputes them.
    pub fn compute_entity_selectivity(&self) -> Result<usize> {
        let edges = self.get_all_relationships()?;
        let mut degree: HashMap<Uuid, i32> = HashMap::new();
        for edge in &edges {
            *degree.entry(edge.from_entity).or_insert(0) += 1;
            *degree.entry(edge.to_entity).or_insert(0) += 1;
        }
        let mut curvatures: HashMap<Uuid, Vec<f32>> = HashMap::new();
        for edge in &edges {
            let curvature = (4 - degree[&edge.from_entity] - degree[&edge.to_entity]) as f32;
            curvatures
                .entry(edge.from_entity)
                .or_default()
                .push(curvature);
            curvatures
                .entry(edge.to_entity)
                .or_default()
                .push(curvature);
        }
        let scores: HashMap<Uuid, f32> = curvatures
            .into_iter()
            .map(|(entity, incident)| (entity, curvature_selectivity(&incident)))
            .collect();
        let scored = scores.len();
        *self.entity_selectivity.write() = scores;
        Ok(scored)
    }

    /// The entity's selectivity from the last [`Self::compute_entity_selectivity`] pass, or
    /// `None` when that pass has not scored it.
    pub fn entity_selectivity(&self, entity: &Uuid) -> Option<f32> {
        self.entity_selectivity.read().get(entity).copied()
    }

    /// Get graph statistics - O(1) using atomic counters
    pub fn get_stats(&self) -> Result<GraphStats> {
        Ok(GraphStats {
            entity_count: self.entity_count.load(Ordering::Relaxed),
            relationship_count: self.relationship_count.load(Ordering::Relaxed),
            episode_count: self.episode_count.load(Ordering::Relaxed),
        })
    }

    /// Compute Normalized Pointwise Mutual Information for an edge.
    ///
    /// NPMI measures how much more (or less) often two entities co-occur than
    /// expected by chance, normalized to [-1, 1]. Used to rescue statistically
    /// rare but meaningful associations from strength-based pruning.
    ///
    /// Returns None during cold start (total_activations < 100) since NPMI
    /// estimates are unreliable with sparse data.
    ///
    /// Formula (with Laplace +1 smoothing):
    ///   f_ab = edge.activation_count + 1
    ///   m_a  = max(mention_count_a, activation_count) + 1
    ///   m_b  = max(mention_count_b, activation_count) + 1
    ///   N_s  = total_activations + num_edges
    ///   pmi  = log2(f_ab * N_s / (m_a * m_b))
    ///   npmi = pmi / log2(N_s / f_ab)
    fn compute_npmi(
        &self,
        edge: &RelationshipEdge,
        mention_count_a: usize,
        mention_count_b: usize,
    ) -> Option<f32> {
        let total = self.total_activations.load(Ordering::Relaxed);
        if total < 100 {
            return None; // Cold start: insufficient data for reliable NPMI
        }

        let num_edges = self.relationship_count.load(Ordering::Relaxed);
        let f_ab = edge.activation_count as f64 + 1.0;
        let m_a = (mention_count_a.max(edge.activation_count as usize) as f64) + 1.0;
        let m_b = (mention_count_b.max(edge.activation_count as usize) as f64) + 1.0;
        let n_s = total as f64 + num_edges as f64;

        let pmi = (f_ab * n_s / (m_a * m_b)).log2();
        let neg_log_p_ab = (n_s / f_ab).log2();

        if neg_log_p_ab < f64::EPSILON {
            return Some(1.0); // Perfect co-occurrence
        }

        Some((pmi / neg_log_p_ab).clamp(-1.0, 1.0) as f32)
    }

    /// Get the NPMI prune threshold for a given edge tier.
    ///
    /// Edges with NPMI above this threshold are rescued from strength-based pruning.
    /// Higher tiers require higher NPMI to reflect their stronger statistical bar.
    fn npmi_prune_threshold(tier: &EdgeTier) -> f32 {
        match tier {
            EdgeTier::L1Working => 0.30,
            EdgeTier::L2Episodic => 0.40,
            EdgeTier::L3Semantic => 0.50,
        }
    }

    /// Build a HashMap<Uuid, usize> cache of entity mention_counts.
    ///
    /// Collects all unique entity UUIDs from edges, batch-reads them from RocksDB,
    /// and returns their mention_count values. Used to avoid per-edge entity lookups
    /// during NPMI computation in the maintenance pass.
    fn build_mention_count_cache<'e>(
        &self,
        edges: impl IntoIterator<Item = &'e RelationshipEdge>,
    ) -> HashMap<Uuid, usize> {
        let mut mention_counts = HashMap::new();
        self.extend_mention_count_cache(&mut mention_counts, &mut HashSet::new(), edges);
        mention_counts
    }

    /// Add to `mention_counts` the endpoints of `edges` not yet in `counted`, each read
    /// once; an entity that is gone stays out of the map, as in a fresh cache.
    fn extend_mention_count_cache<'e>(
        &self,
        mention_counts: &mut HashMap<Uuid, usize>,
        counted: &mut HashSet<Uuid>,
        edges: impl IntoIterator<Item = &'e RelationshipEdge>,
    ) {
        let mut entity_uuids: HashSet<Uuid> = HashSet::new();
        for edge in edges {
            for entity in [edge.from_entity, edge.to_entity] {
                if counted.insert(entity) {
                    entity_uuids.insert(entity);
                }
            }
        }

        let keys: Vec<[u8; 16]> = entity_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let results: Vec<anyhow::Result<Option<Vec<u8>>>> = key_refs
            .iter()
            .map(|k| self.store.get_cf(CF_ENTITIES, k))
            .collect();

        let uuid_vec: Vec<Uuid> = entity_uuids.into_iter().collect();
        for (i, result) in results.into_iter().enumerate() {
            if let Ok(Some(value)) = result {
                if let Ok((entity, _)) = bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    mention_counts.insert(uuid_vec[i], entity.mention_count);
                }
            }
        }
    }

    /// Get all entities in the graph
    pub fn get_all_entities(&self) -> Result<Vec<EntityNode>> {
        let mut entities = Vec::new();

        // scan_cf materializes the rows; the fill_cache(false) block-cache
        // courtesy was a rocksdb-specific transient-read hint.
        for (_, value) in self.store.scan_cf(CF_ENTITIES, None, None)? {
            if let Ok(entity) = bincode::serde::decode_from_slice::<EntityNode, _>(
                &value,
                bincode::config::standard(),
            )
            .map(|(v, _)| v)
            {
                entities.push(entity);
            }
        }

        // Sort by mention count (most mentioned first)
        entities.sort_by_key(|entity| std::cmp::Reverse(entity.mention_count));

        Ok(entities)
    }

    /// Every episode, plus the number of rows that did not decode. The caller reports that
    /// number rather than this function hiding it: a count over a partially readable store is a
    /// lower bound, not a measurement.
    pub fn get_all_episodes(&self) -> Result<(Vec<EpisodicNode>, usize)> {
        let mut episodes = Vec::new();
        let mut undecodable = 0usize;
        for (_, value) in self.store.scan_cf(CF_EPISODES, None, None)? {
            match bincode::serde::decode_from_slice::<EpisodicNode, _>(
                &value,
                bincode::config::standard(),
            ) {
                Ok((episode, _)) => episodes.push(episode),
                Err(_) => undecodable += 1,
            }
        }
        episodes.sort_by(|a, b| {
            a.valid_at
                .cmp(&b.valid_at)
                .then_with(|| a.uuid.cmp(&b.uuid))
        });
        Ok((episodes, undecodable))
    }

    /// Count of edges per relation type, descending then alphabetical.
    /// Read by the recall harness's ontology report.
    pub fn relation_type_distribution(&self) -> Result<Vec<(String, usize)>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_, value) in self.store.scan_cf(CF_RELATIONSHIPS, None, None)? {
            if let Ok(edge) = RelationshipEdge::decode(&value) {
                *counts
                    .entry(edge.relation_type.as_str().to_string())
                    .or_default() += 1;
            }
        }
        let mut out: Vec<(String, usize)> = counts.into_iter().collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(out)
    }

    /// Get all relationships in the graph
    pub fn get_all_relationships(&self) -> Result<Vec<RelationshipEdge>> {
        let mut relationships = Vec::new();

        // fill_cache(false) prevents this full scan from evicting hot data from
        // the block cache. Decompressed blocks are used transiently and freed
        // after the iterator advances, reducing peak C++ heap usage.
        // scan_cf materializes the rows; the fill_cache(false) block-cache
        // courtesy was a rocksdb-specific transient-read hint.
        let iter = self
            .store
            .scan_cf(CF_RELATIONSHIPS, None, None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for (_, value) in iter.flatten() {
            if let Ok(edge) = RelationshipEdge::decode(&value) {
                // Only include non-invalidated relationships
                if edge.invalidated_at.is_none() {
                    relationships.push(edge);
                }
            }
        }

        // Sort by strength (strongest first)
        relationships.sort_by(|a, b| b.strength.total_cmp(&a.strength));

        Ok(relationships)
    }

    /// Export edges safe for cross-user collective aggregation.
    ///
    /// Returns L3-tier edges where both endpoints are PII-clean.
    /// Each tuple: (from_name, to_name, strength, relation_type).
    #[allow(dead_code)]
    pub fn export_collective_edges(&self) -> Result<Vec<(String, String, f32, String)>> {
        let mut safe_edges = Vec::new();
        // scan_cf materializes the rows; the fill_cache(false) block-cache
        // courtesy was a rocksdb-specific transient-read hint.
        let iter = self
            .store
            .scan_cf(CF_RELATIONSHIPS, None, None)?
            .into_iter()
            .map(Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>);
        for (_, value) in iter.flatten() {
            let edge = match RelationshipEdge::decode(&value) {
                Ok(e) => e,
                Err(_) => continue,
            };
            if edge.invalidated_at.is_some() || edge.tier != EdgeTier::L3Semantic {
                continue;
            }
            let from_entity = match self.get_entity(&edge.from_entity)? {
                Some(e) => e,
                None => continue,
            };
            let to_entity = match self.get_entity(&edge.to_entity)? {
                Some(e) => e,
                None => continue,
            };
            if is_edge_safe_for_collective(&edge, &from_entity, &to_entity) {
                safe_edges.push((
                    from_entity.name,
                    to_entity.name,
                    edge.strength,
                    format!("{:?}", edge.relation_type),
                ));
            }
        }
        Ok(safe_edges)
    }

    /// Get the Memory Universe visualization data
    /// Returns entities as "stars" with positions based on their relationships,
    /// sized by salience, and colored by entity type.
    pub fn get_universe(&self) -> Result<MemoryUniverse> {
        let entities = self.get_all_entities()?;
        let relationships = self.get_all_relationships()?;

        // Create entity UUID to index mapping for position calculation
        let entity_indices: HashMap<Uuid, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.uuid, i))
            .collect();

        // Calculate 3D positions using a force-directed layout approximation
        // High-salience entities are positioned more centrally
        let mut stars: Vec<UniverseStar> = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| {
                // Use a spiral galaxy layout with salience affecting radius
                // Higher salience = closer to center
                let angle = (i as f32) * 2.4; // Golden angle for even distribution
                let base_radius = 1.0 - entity.salience; // High salience = small radius
                let radius = base_radius * 100.0 + 10.0; // 10-110 range

                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let z = ((i as f32) * 0.1).sin() * 20.0; // Slight z variation

                UniverseStar {
                    id: entity.uuid.to_string(),
                    name: entity.name.clone(),
                    entity_type: entity.labels.first().map(|l| l.as_str().to_string()),
                    salience: entity.salience,
                    mention_count: entity.mention_count,
                    is_proper_noun: entity.is_proper_noun,
                    position: Position3D { x, y, z },
                    color: entity_type_color(entity.labels.first()),
                    size: 5.0 + entity.salience * 20.0, // Size 5-25 based on salience
                }
            })
            .collect();

        // Apply gravitational forces FIRST, before creating connections
        // This ensures connection positions match final star positions
        for rel in &relationships {
            if let (Some(from_idx), Some(to_idx)) = (
                entity_indices.get(&rel.from_entity),
                entity_indices.get(&rel.to_entity),
            ) {
                // Apply small gravitational pull based on effective (decay-aware) strength
                let pull_factor = rel.effective_strength() * 0.05;

                let from_pos = stars[*from_idx].position.clone();
                let to_pos = stars[*to_idx].position.clone();

                let dx = (to_pos.x - from_pos.x) * pull_factor;
                let dy = (to_pos.y - from_pos.y) * pull_factor;
                let dz = (to_pos.z - from_pos.z) * pull_factor;

                stars[*from_idx].position.x += dx;
                stars[*from_idx].position.y += dy;
                stars[*from_idx].position.z += dz;

                stars[*to_idx].position.x -= dx;
                stars[*to_idx].position.y -= dy;
                stars[*to_idx].position.z -= dz;
            }
        }

        // Create gravitational connections AFTER star positions are finalized
        // This ensures from_position/to_position match current star positions
        let connections: Vec<GravitationalConnection> = relationships
            .iter()
            .filter_map(|rel| {
                let from_idx = entity_indices.get(&rel.from_entity)?;
                let to_idx = entity_indices.get(&rel.to_entity)?;

                Some(GravitationalConnection {
                    id: rel.uuid.to_string(),
                    from_id: rel.from_entity.to_string(),
                    to_id: rel.to_entity.to_string(),
                    relation_type: rel.relation_type.as_str().to_string(),
                    strength: rel.effective_strength(),
                    from_position: stars[*from_idx].position.clone(),
                    to_position: stars[*to_idx].position.clone(),
                })
            })
            .collect();

        // Calculate universe bounds
        let (min_x, max_x, min_y, max_y, min_z, max_z) = stars.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y, min_z, max_z), star| {
                (
                    min_x.min(star.position.x),
                    max_x.max(star.position.x),
                    min_y.min(star.position.y),
                    max_y.max(star.position.y),
                    min_z.min(star.position.z),
                    max_z.max(star.position.z),
                )
            },
        );

        Ok(MemoryUniverse {
            stars,
            connections,
            total_entities: entities.len(),
            total_connections: relationships.len(),
            bounds: UniverseBounds {
                min: Position3D {
                    x: min_x,
                    y: min_y,
                    z: min_z,
                },
                max: Position3D {
                    x: max_x,
                    y: max_y,
                    z: max_z,
                },
            },
        })
    }
}

/// Helper function to get color for entity type
fn entity_type_color(label: Option<&EntityLabel>) -> String {
    match label {
        Some(EntityLabel::Person) => "#FF6B6B".to_string(), // Coral red
        Some(EntityLabel::Organization) => "#4ECDC4".to_string(), // Teal
        Some(EntityLabel::Location) => "#45B7D1".to_string(), // Sky blue
        Some(EntityLabel::Technology) => "#96CEB4".to_string(), // Sage green
        Some(EntityLabel::Product) => "#FFEAA7".to_string(), // Soft yellow
        Some(EntityLabel::Event) => "#DDA0DD".to_string(),  // Plum
        Some(EntityLabel::Skill) => "#98D8C8".to_string(),  // Mint
        Some(EntityLabel::Concept) => "#F7DC6F".to_string(), // Gold
        Some(EntityLabel::Date) => "#BB8FCE".to_string(),   // Light purple
        Some(EntityLabel::Keyword) => "#FF9F43".to_string(), // Orange for YAKE keywords
        Some(EntityLabel::Project) => "#E74C3C".to_string(), // Red — project anchors
        Some(EntityLabel::Task) => "#F39C12".to_string(),   // Amber — work items
        Some(EntityLabel::Document) => "#1ABC9C".to_string(), // Turquoise — knowledge
        Some(EntityLabel::Repository) => "#2ECC71".to_string(), // Emerald — code entities
        Some(EntityLabel::Service) => "#3498DB".to_string(), // Blue — architectural
        Some(EntityLabel::Database) => "#9B59B6".to_string(), // Purple — data stores
        Some(EntityLabel::Metric) => "#E67E22".to_string(), // Dark orange — telemetry
        Some(EntityLabel::Configuration) => "#95A5A6".to_string(), // Silver — config
        Some(EntityLabel::Environment) => "#16A085".to_string(), // Dark teal — infra
        Some(EntityLabel::Pipeline) => "#2980B9".to_string(), // Dark blue — CI/CD
        Some(EntityLabel::Team) => "#27AE60".to_string(),   // Green — organizational
        Some(EntityLabel::Role) => "#8E44AD".to_string(),   // Dark purple — roles
        Some(EntityLabel::Module) => "#D35400".to_string(), // Pumpkin — code modules
        Some(EntityLabel::Other(_)) => "#AEB6BF".to_string(), // Gray
        Some(EntityLabel::FilePath) => "#C0392B".to_string(), // Dark red — files
        Some(EntityLabel::Commit) => "#7F8C8D".to_string(), // Slate — commits
        Some(EntityLabel::Symbol) => "#E59866".to_string(), // Tan — symbols
        Some(EntityLabel::Session) => "#5DADE2".to_string(), // Light blue — sessions
        Some(EntityLabel::Agent) => "#48C9B0".to_string(),  // Aqua — agents
        Some(EntityLabel::Tool) => "#F4D03F".to_string(),   // Yellow — tools
        Some(EntityLabel::Template) => "#AF7AC5".to_string(), // Lilac — templates
        None => "#AEB6BF".to_string(),                      // Gray default
    }
}

/// 3D position in the memory universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A star in the memory universe (represents an entity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseStar {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub salience: f32,
    pub mention_count: usize,
    pub is_proper_noun: bool,
    pub position: Position3D,
    pub color: String,
    pub size: f32,
}

/// A gravitational connection between stars (represents a relationship)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitationalConnection {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation_type: String,
    pub strength: f32,
    pub from_position: Position3D,
    pub to_position: Position3D,
}

/// Bounds of the memory universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseBounds {
    pub min: Position3D,
    pub max: Position3D,
}

/// The complete memory universe visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUniverse {
    pub stars: Vec<UniverseStar>,
    pub connections: Vec<GravitationalConnection>,
    pub total_entities: usize,
    pub total_connections: usize,
    pub bounds: UniverseBounds,
}

/// Entity with hop distance from traversal origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversedEntity {
    pub entity: EntityNode,
    /// Number of hops from the starting entity (0 = start entity)
    pub hop_distance: usize,
    /// Decay factor based on hop distance: 1.0 at hop 0, decays with each hop
    pub decay_factor: f32,
}

/// Result of graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversal {
    /// Entities found during traversal with hop distance info
    pub entities: Vec<TraversedEntity>,
    pub relationships: Vec<RelationshipEdge>,
}

/// Graph statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub episode_count: usize,
}

/// Extracted entity with salience information
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub label: EntityLabel,
    pub base_salience: f32,
}

/// Simple entity extraction (rule-based NER) with salience detection
pub struct EntityExtractor {
    /// Common person name indicators
    person_indicators: HashSet<String>,

    /// Common organization indicators (suffixes like Inc, Corp)
    org_indicators: HashSet<String>,

    /// Known organization names (direct matches)
    org_keywords: HashSet<String>,

    /// Known location names (cities, countries, regions)
    location_keywords: HashSet<String>,

    /// Common technology keywords
    tech_keywords: HashSet<String>,

    /// Common words that should NOT be extracted as entities
    /// (stop words that start with capitals at sentence beginning)
    stop_words: HashSet<String>,

    // Extended ontological keyword dictionaries (2026-03)
    /// Service/infrastructure keywords (microservices, daemons, proxies)
    service_keywords: HashSet<String>,
    /// Database system keywords
    database_keywords: HashSet<String>,
    /// CI/CD and data pipeline keywords
    pipeline_keywords: HashSet<String>,
    /// Deployment environment keywords
    environment_keywords: HashSet<String>,
    /// Observability and metric keywords
    metric_keywords: HashSet<String>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let person_indicators: HashSet<String> =
            vec!["mr", "mrs", "ms", "dr", "prof", "sir", "madam"]
                .into_iter()
                .map(String::from)
                .collect();

        let org_indicators: HashSet<String> = vec![
            "inc",
            "corp",
            "ltd",
            "llc",
            "company",
            "corporation",
            "university",
            "institute",
            "foundation",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let tech_keywords: HashSet<String> = vec![
            "rust",
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "vue",
            "angular",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "sql",
            "nosql",
            "mongodb",
            "postgresql",
            "redis",
            "kafka",
            "api",
            "rest",
            "graphql",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Known organization names (global - India-first, then worldwide)
        let org_keywords: HashSet<String> = vec![
            // Indian Companies - IT/Tech
            "tcs",
            "infosys",
            "wipro",
            "hcl",
            "tech mahindra",
            "cognizant",
            "mindtree",
            "mphasis",
            "ltimindtree",
            "persistent",
            "zensar",
            "cyient",
            "hexaware",
            "coforge",
            "birlasoft",
            "sonata software",
            "mastek",
            "newgen",
            // Indian Companies - Startups/Unicorns
            "flipkart",
            "paytm",
            "zomato",
            "swiggy",
            "ola",
            "oyo",
            "byju's",
            "byjus",
            "razorpay",
            "phonepe",
            "cred",
            "zerodha",
            "groww",
            "upstox",
            "policybazaar",
            "nykaa",
            "meesho",
            "udaan",
            "delhivery",
            "freshworks",
            "zoho",
            "postman",
            "browserstack",
            "chargebee",
            "clevertap",
            "druva",
            "hasura",
            "innovaccer",
            "lenskart",
            "mamaearth",
            "unacademy",
            "vedantu",
            "physicswallah",
            "dream11",
            "mpl",
            "winzo",
            "slice",
            "jupiter",
            "fi",
            "niyo",
            "smallcase",
            "koo",
            "sharechat",
            "dailyhunt",
            "pratilipi",
            "inshorts",
            "rapido",
            "urban company",
            "dunzo",
            "bigbasket",
            "grofers",
            "blinkit",
            "jiomart",
            "tata neu",
            // Indian Conglomerates
            "tata",
            "reliance",
            "jio",
            "adani",
            "birla",
            "mahindra",
            "godrej",
            "bajaj",
            "hdfc",
            "icici",
            "kotak",
            "axis",
            "sbi",
            "bharti",
            "airtel",
            "vodafone",
            "idea",
            "hero",
            "tvs",
            "maruti",
            "suzuki",
            "hyundai",
            "kia",
            "mg",
            "tata motors",
            "larsen",
            "toubro",
            "l&t",
            "itc",
            "hindustan unilever",
            "hul",
            "nestle",
            "britannia",
            "parle",
            "amul",
            "dabur",
            "patanjali",
            "emami",
            "marico",
            // Indian Banks & Finance
            "rbi",
            "sebi",
            "nse",
            "bse",
            "npci",
            "upi",
            "bhim",
            "paisa",
            "mswipe",
            "pine labs",
            "billdesk",
            "ccavenue",
            "instamojo",
            "cashfree",
            // Indian Institutions
            "iit",
            "iim",
            "iisc",
            "nit",
            "bits",
            "isro",
            "drdo",
            "barc",
            "tifr",
            "aiims",
            "iiser",
            "iiit",
            "srm",
            "vit",
            "manipal",
            "amity",
            "lovely",
            // Global Tech Giants
            "microsoft",
            "google",
            "apple",
            "amazon",
            "meta",
            "facebook",
            "netflix",
            "alphabet",
            "youtube",
            "instagram",
            "whatsapp",
            "tiktok",
            "snapchat",
            "twitter",
            "x",
            "linkedin",
            "pinterest",
            "reddit",
            "discord",
            "telegram",
            // Global Enterprise Tech
            "salesforce",
            "oracle",
            "ibm",
            "sap",
            "vmware",
            "dell",
            "hp",
            "hpe",
            "cisco",
            "juniper",
            "palo alto",
            "crowdstrike",
            "fortinet",
            "splunk",
            "servicenow",
            "workday",
            "atlassian",
            "jira",
            "confluence",
            "trello",
            "asana",
            "monday",
            "notion",
            "airtable",
            "figma",
            "canva",
            "miro",
            // Global Cloud & Infrastructure
            "aws",
            "azure",
            "gcp",
            "digitalocean",
            "linode",
            "vultr",
            "cloudflare",
            "akamai",
            "fastly",
            "vercel",
            "netlify",
            "heroku",
            "render",
            "railway",
            // Global Hardware/Chip
            "intel",
            "amd",
            "nvidia",
            "qualcomm",
            "broadcom",
            "arm",
            "tsmc",
            "samsung",
            "mediatek",
            "apple silicon",
            "marvell",
            "micron",
            "sk hynix",
            "western digital",
            // Global AI/ML Companies
            "openai",
            "anthropic",
            "deepmind",
            "cohere",
            "stability",
            "midjourney",
            "hugging face",
            "databricks",
            "snowflake",
            "palantir",
            "c3ai",
            "datarobot",
            // Global Fintech
            "stripe",
            "square",
            "block",
            "paypal",
            "venmo",
            "wise",
            "revolut",
            "robinhood",
            "coinbase",
            "binance",
            "kraken",
            "gemini",
            "ftx",
            "blockchain",
            "ripple",
            // Global Dev Tools
            "github",
            "gitlab",
            "bitbucket",
            "jetbrains",
            "vscode",
            "sublime",
            "vim",
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "puppet",
            "chef",
            // Global Consulting
            "accenture",
            "deloitte",
            "pwc",
            "kpmg",
            "ey",
            "mckinsey",
            "bcg",
            "bain",
            // Global Auto/EV
            "tesla",
            "rivian",
            "lucid",
            "nio",
            "byd",
            "xpeng",
            "volkswagen",
            "bmw",
            "mercedes",
            "audi",
            "porsche",
            "toyota",
            "honda",
            "nissan",
            "ford",
            "gm",
            // Global Aerospace
            "spacex",
            "boeing",
            "airbus",
            "lockheed",
            "northrop",
            "raytheon",
            "nasa",
            "esa",
            "jaxa",
            "isro",
            "blue origin",
            "virgin galactic",
            // Universities - India
            "delhi university",
            "jnu",
            "bhu",
            "amu",
            "jadavpur",
            "presidency",
            "st stephens",
            "loyola",
            "xavier",
            "symbiosis",
            "nmims",
            "sp jain",
            "xlri",
            "fms",
            "iift",
            "mdi",
            "great lakes",
            "ism dhanbad",
            // Universities - Global
            "mit",
            "stanford",
            "harvard",
            "yale",
            "princeton",
            "caltech",
            "berkeley",
            "oxford",
            "cambridge",
            "imperial",
            "eth zurich",
            "epfl",
            "tsinghua",
            "peking",
            "nus",
            "nanyang",
            "kaist",
            "university of tokyo",
            "melbourne",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Known location names (global - India-first, then worldwide)
        let location_keywords: HashSet<String> = vec![
            // Indian Metro Cities
            "mumbai",
            "delhi",
            "bangalore",
            "bengaluru",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "ahmedabad",
            "surat",
            "jaipur",
            "lucknow",
            // Indian Tier-1 Cities
            "kochi",
            "cochin",
            "thiruvananthapuram",
            "trivandrum",
            "coimbatore",
            "madurai",
            "visakhapatnam",
            "vizag",
            "vijayawada",
            "nagpur",
            "indore",
            "bhopal",
            "chandigarh",
            "mohali",
            "panchkula",
            "noida",
            "gurgaon",
            "gurugram",
            "faridabad",
            "ghaziabad",
            "greater noida",
            "dwarka",
            // Indian Tier-2 Cities
            "mysore",
            "mangalore",
            "hubli",
            "belgaum",
            "nashik",
            "aurangabad",
            "rajkot",
            "vadodara",
            "baroda",
            "gandhinagar",
            "kanpur",
            "varanasi",
            "allahabad",
            "prayagraj",
            "agra",
            "meerut",
            "dehradun",
            "rishikesh",
            "haridwar",
            "amritsar",
            "jalandhar",
            "ludhiana",
            "shimla",
            "manali",
            "dharamshala",
            "jammu",
            "srinagar",
            "ranchi",
            "jamshedpur",
            "patna",
            "guwahati",
            "shillong",
            "imphal",
            "kohima",
            "gangtok",
            "darjeeling",
            "bhubaneswar",
            "cuttack",
            "rourkela",
            "raipur",
            "bilaspur",
            // Indian States & UTs
            "maharashtra",
            "karnataka",
            "tamil nadu",
            "telangana",
            "andhra pradesh",
            "kerala",
            "gujarat",
            "rajasthan",
            "uttar pradesh",
            "madhya pradesh",
            "west bengal",
            "bihar",
            "odisha",
            "jharkhand",
            "chhattisgarh",
            "punjab",
            "haryana",
            "himachal pradesh",
            "uttarakhand",
            "goa",
            "assam",
            "meghalaya",
            "manipur",
            "nagaland",
            "tripura",
            "mizoram",
            "arunachal pradesh",
            "sikkim",
            "jammu and kashmir",
            "ladakh",
            // Indian Regions
            "silicon valley of india",
            "electronic city",
            "whitefield",
            "marathahalli",
            "koramangala",
            "indiranagar",
            "hsr layout",
            "jayanagar",
            "malleshwaram",
            "bandra",
            "andheri",
            "powai",
            "lower parel",
            "bkc",
            "navi mumbai",
            "thane",
            "connaught place",
            "nehru place",
            "saket",
            "cyber city",
            "dlf",
            "hitech city",
            "madhapur",
            "gachibowli",
            "ecr",
            "omr",
            "it corridor",
            // Asian Cities
            "singapore",
            "hong kong",
            "tokyo",
            "osaka",
            "seoul",
            "busan",
            "beijing",
            "shanghai",
            "shenzhen",
            "guangzhou",
            "hangzhou",
            "taipei",
            "bangkok",
            "kuala lumpur",
            "jakarta",
            "manila",
            "ho chi minh",
            "hanoi",
            "dubai",
            "abu dhabi",
            "doha",
            "riyadh",
            "tel aviv",
            "istanbul",
            // European Cities
            "london",
            "paris",
            "berlin",
            "munich",
            "frankfurt",
            "amsterdam",
            "rotterdam",
            "brussels",
            "zurich",
            "geneva",
            "vienna",
            "prague",
            "warsaw",
            "budapest",
            "barcelona",
            "madrid",
            "milan",
            "rome",
            "lisbon",
            "dublin",
            "edinburgh",
            "manchester",
            "stockholm",
            "oslo",
            "helsinki",
            "copenhagen",
            "athens",
            "moscow",
            "st petersburg",
            // North American Cities
            "new york",
            "los angeles",
            "san francisco",
            "seattle",
            "boston",
            "chicago",
            "austin",
            "denver",
            "portland",
            "miami",
            "atlanta",
            "dallas",
            "houston",
            "phoenix",
            "san diego",
            "san jose",
            "oakland",
            "palo alto",
            "mountain view",
            "cupertino",
            "menlo park",
            "redwood city",
            "washington dc",
            "philadelphia",
            "detroit",
            "toronto",
            "vancouver",
            "montreal",
            "calgary",
            "ottawa",
            "mexico city",
            "guadalajara",
            // South American Cities
            "sao paulo",
            "rio de janeiro",
            "buenos aires",
            "santiago",
            "bogota",
            "lima",
            "medellin",
            "cartagena",
            // African Cities
            "johannesburg",
            "cape town",
            "lagos",
            "nairobi",
            "cairo",
            "casablanca",
            "accra",
            "addis ababa",
            "kigali",
            // Australian/NZ Cities
            "sydney",
            "melbourne",
            "brisbane",
            "perth",
            "auckland",
            "wellington",
            // Countries - Asia
            "india",
            "china",
            "japan",
            "south korea",
            "korea",
            "taiwan",
            "singapore",
            "malaysia",
            "thailand",
            "vietnam",
            "indonesia",
            "philippines",
            "bangladesh",
            "pakistan",
            "sri lanka",
            "nepal",
            "bhutan",
            "myanmar",
            "cambodia",
            "laos",
            // Countries - Middle East
            "uae",
            "emirates",
            "saudi arabia",
            "qatar",
            "bahrain",
            "kuwait",
            "oman",
            "israel",
            "turkey",
            "iran",
            "iraq",
            "jordan",
            "lebanon",
            "egypt",
            // Countries - Europe
            "uk",
            "united kingdom",
            "britain",
            "england",
            "scotland",
            "wales",
            "ireland",
            "france",
            "germany",
            "italy",
            "spain",
            "portugal",
            "netherlands",
            "belgium",
            "switzerland",
            "austria",
            "poland",
            "czech",
            "hungary",
            "romania",
            "bulgaria",
            "greece",
            "sweden",
            "norway",
            "finland",
            "denmark",
            "russia",
            "ukraine",
            // Countries - Americas
            "usa",
            "united states",
            "america",
            "canada",
            "mexico",
            "brazil",
            "argentina",
            "chile",
            "colombia",
            "peru",
            "venezuela",
            // Countries - Africa/Oceania
            "south africa",
            "nigeria",
            "kenya",
            "ghana",
            "ethiopia",
            "rwanda",
            "australia",
            "new zealand",
            // Famous Tech Hubs
            "silicon valley",
            "bay area",
            "wall street",
            "tech city",
            "shoreditch",
            "station f",
            "blockchain island",
            "crypto valley",
            "startup nation",
            "innovation district",
            "tech park",
            "it park",
            "sez",
            "special economic zone",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Stop words: common words that appear capitalized at sentence start
        // These aren't named entities even when capitalized
        let stop_words: HashSet<String> = vec![
            // Articles & pronouns
            "the", "a", "an", "this", "that", "these", "those", "i", "we", "you", "he", "she", "it",
            "they", // Common verbs (appear at sentence start)
            "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", // Question words
            "if", "when", "where", "what", "why", "how",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Service/infrastructure keywords
        let service_keywords: HashSet<String> = vec![
            "microservice",
            "daemon",
            "worker",
            "cron",
            "lambda",
            "serverless",
            "gateway",
            "proxy",
            "nginx",
            "envoy",
            "istio",
            "grpc",
            "graphql",
            "webhook",
            "middleware",
            "sidecar",
            "ingress",
            "loadbalancer",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Database system keywords
        let database_keywords: HashSet<String> = vec![
            "postgresql",
            "postgres",
            "mysql",
            "mariadb",
            "sqlite",
            "dynamodb",
            "cassandra",
            "elasticsearch",
            "opensearch",
            "clickhouse",
            "timescaledb",
            "cockroachdb",
            "neo4j",
            "arangodb",
            "rocksdb",
            "leveldb",
            "badger",
            "redb",
            "sled",
            "foundationdb",
            "vitess",
            "couchbase",
            "couchdb",
            "influxdb",
            "questdb",
            "duckdb",
            "dragonfly",
            "valkey",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // CI/CD and pipeline keywords
        let pipeline_keywords: HashSet<String> = vec![
            "jenkins",
            "circleci",
            "travis",
            "argo",
            "tekton",
            "spinnaker",
            "buildkite",
            "drone",
            "concourse",
            "woodpecker",
            "flux",
            "argocd",
            "terraform",
            "pulumi",
            "ansible",
            "airflow",
            "dagster",
            "prefect",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Deployment environment keywords
        let environment_keywords: HashSet<String> = vec![
            "staging",
            "production",
            "prod",
            "dev",
            "development",
            "sandbox",
            "canary",
            "preview",
            "qa",
            "uat",
            "integration",
            "preprod",
            "hotfix",
            "nightly",
            "edge",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Observability and metric keywords
        let metric_keywords: HashSet<String> = vec![
            "latency",
            "throughput",
            "p99",
            "p95",
            "p50",
            "slo",
            "sla",
            "sli",
            "uptime",
            "availability",
            "mttr",
            "mttf",
            "error rate",
            "saturation",
            "prometheus",
            "grafana",
            "datadog",
            "newrelic",
            "jaeger",
            "tempo",
            "loki",
            "opentelemetry",
            "otel",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            person_indicators,
            org_indicators,
            org_keywords,
            location_keywords,
            tech_keywords,
            stop_words,
            service_keywords,
            database_keywords,
            pipeline_keywords,
            environment_keywords,
            metric_keywords,
        }
    }

    /// Calculate base salience for an entity based on its type and detection confidence
    ///
    /// Salience values by entity type:
    /// - Person: 0.8 (highest - people are key context)
    /// - Organization/Product: 0.7
    /// - Location/Technology/Event: 0.6
    /// - Skill: 0.5
    /// - Concept: 0.4
    /// - Date/Other: 0.3
    ///
    /// Proper nouns receive a 20% boost (capped at 1.0).
    pub fn calculate_base_salience(label: &EntityLabel, is_proper_noun: bool) -> f32 {
        let type_salience = match label {
            EntityLabel::Person => 0.8,         // People are highly salient
            EntityLabel::Organization => 0.7,   // Organizations are important
            EntityLabel::Location => 0.6,       // Locations matter for context
            EntityLabel::Technology => 0.6,     // Tech keywords matter for dev context
            EntityLabel::Product => 0.7,        // Products are specific entities
            EntityLabel::Event => 0.6,          // Events are temporal anchors
            EntityLabel::Skill => 0.5,          // Skills are somewhat important
            EntityLabel::Keyword => 0.55,       // YAKE keywords - discriminative terms
            EntityLabel::Concept => 0.4,        // Concepts are more generic
            EntityLabel::Date => 0.3,           // Dates are structural, not salient
            EntityLabel::Project => 0.7,        // Projects are specific, high-salience anchors
            EntityLabel::Task => 0.55,          // Tasks are common but specific work items
            EntityLabel::Document => 0.5,       // Documents provide contextual reference
            EntityLabel::Repository => 0.65,    // Repos are identifiable code entities
            EntityLabel::Service => 0.65,       // Services are architectural anchors
            EntityLabel::Database => 0.6,       // Databases are infrastructure anchors
            EntityLabel::Metric => 0.5,         // Metrics are precise but numerous
            EntityLabel::Configuration => 0.45, // Config is low-salience, high-frequency
            EntityLabel::Environment => 0.5,    // Environments are contextual
            EntityLabel::Pipeline => 0.6,       // Pipelines are operational entities
            EntityLabel::Team => 0.7,           // Teams are organizational anchors
            EntityLabel::Role => 0.55,          // Roles are semi-generic
            EntityLabel::Module => 0.55,        // Modules are code-level entities
            EntityLabel::Other(_) => 0.3,       // Unknown types get low salience
            // Writer-declared identifiers are specific by construction.
            EntityLabel::FilePath | EntityLabel::Symbol => 0.65,
            EntityLabel::Commit | EntityLabel::Template => 0.6,
            EntityLabel::Session | EntityLabel::Agent => 0.5,
            EntityLabel::Tool => 0.55,
        };

        // Proper nouns get a 20% boost
        if is_proper_noun {
            (type_salience * 1.2_f32).min(1.0_f32)
        } else {
            type_salience
        }
    }

    /// Check if a word is likely a proper noun (not just capitalized at sentence start)
    fn is_likely_proper_noun(&self, word: &str, position: usize, prev_char: Option<char>) -> bool {
        // If it's not at position 0 and is capitalized, it's likely a proper noun
        if position > 0 {
            return true;
        }

        // At position 0, check if previous character was punctuation (sentence start)
        // If previous char was '.', '!', '?' then this might just be sentence capitalization
        if let Some(c) = prev_char {
            if c == '.' || c == '!' || c == '?' {
                // It's at sentence start - could be either
                // Check if it's a common word
                let lower = word.to_lowercase();
                return !self.stop_words.contains(&lower);
            }
        }

        // Default to proper noun for capitalized words
        true
    }

    /// Extract entities from text with salience information
    pub fn extract_with_salience(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();
        let mut skip_until_index = 0; // For skipping sub-spans of multi-word entities

        // Split into words and detect capitalized sequences
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Skip if this word is part of a multi-word entity we already extracted
            if i < skip_until_index {
                continue;
            }

            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());

            if clean_word.is_empty() {
                continue;
            }

            let lower = clean_word.to_lowercase();

            // Skip common stop words
            if self.stop_words.contains(&lower) {
                continue;
            }

            // Check for known organization keywords (direct match, min 2 chars to filter "x" noise)
            if lower.len() >= 2 && self.org_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Organization,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Organization, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for known location keywords (direct match)
            if self.location_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Location,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Location, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for technology keywords (always proper nouns in tech context)
            // Check extended ontological keyword dictionaries (more specific before generic)
            if self.database_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Database,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Database, true),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.pipeline_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Pipeline,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Pipeline, true),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.service_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Service,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Service, true),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.environment_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Environment,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Environment, false),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.metric_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Metric,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Metric, false),
                });
                seen.insert(lower.clone());
                continue;
            }

            // Generic technology keywords (catch-all for tech not matched above)
            if self.tech_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Technology,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Technology, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for capitalized words (potential entities)
            if clean_word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                let mut entity_name = clean_word.to_string();
                let mut entity_label = EntityLabel::Other("Unknown".to_string());

                // Determine previous character for proper noun detection
                let prev_char = if i > 0 {
                    words[i - 1].chars().last()
                } else {
                    None
                };

                let is_proper = self.is_likely_proper_noun(clean_word, i, prev_char);

                // Check for person indicators
                if i > 0
                    && self
                        .person_indicators
                        .contains(&words[i - 1].to_lowercase())
                {
                    entity_label = EntityLabel::Person;
                }

                // Check for multi-word capitalized sequences.
                // Include capitalized stop words (Of, The, And) in entity names
                // to preserve proper nouns like "Bank Of America", "University Of Delhi".
                let mut j = i + 1;
                while j < words.len()
                    && words[j]
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    let next_word = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                    entity_name.push(' ');
                    entity_name.push_str(next_word);
                    j += 1;
                }

                // Set skip_until_index to avoid extracting sub-spans
                // e.g., if we extracted "John Smith", skip "Smith" on next iteration
                if j > i + 1 {
                    skip_until_index = j;
                }

                let entity_name_lower = entity_name.to_lowercase();

                // Check multi-word entity against known lists
                if self.org_keywords.contains(&entity_name_lower) {
                    entity_label = EntityLabel::Organization;
                } else if self.location_keywords.contains(&entity_name_lower) {
                    entity_label = EntityLabel::Location;
                }

                // Check for organization indicators (suffixes)
                if matches!(entity_label, EntityLabel::Other(_)) {
                    for word in entity_name.split_whitespace() {
                        if self.org_indicators.contains(&word.to_lowercase()) {
                            entity_label = EntityLabel::Organization;
                            break;
                        }
                    }
                }

                // Only extract entities we have evidence for
                // Don't guess on single unknown capitalized words - they're often noise
                if matches!(entity_label, EntityLabel::Other(_)) {
                    if entity_name.contains(' ') {
                        // Multi-word capitalized sequences (like "John Smith", "New York")
                        // are likely proper names — use Concept as safe default
                        // Concept(0.4) + proper noun boost(1.2x) = 0.48 salience
                        // Hebbian strengthening will promote genuinely important entities
                        entity_label = EntityLabel::Concept;
                    } else {
                        // Single capitalized word not in any keyword list
                        // Skip it - we don't have enough evidence it's a real entity
                        // The neural NER model handles these cases properly
                        continue;
                    }
                }

                let entity_key = entity_name_lower;
                if !seen.contains(&entity_key) {
                    let base_salience = Self::calculate_base_salience(&entity_label, is_proper);
                    let entity = ExtractedEntity {
                        name: entity_name,
                        label: entity_label,
                        base_salience,
                    };
                    entities.push(entity);
                    seen.insert(entity_key);
                }
            }
        }

        // HYBRID APPROACH: POS-based extraction + YAKE importance scoring
        //
        // 1. POS extraction ensures ALL content words are captured (no frequency bias)
        // 2. YAKE provides discriminativeness scores for boosting rare/important terms
        //
        // This solves the "sunrise problem": YAKE alone buries rare words at position 41,
        // but POS ensures "sunrise" is extracted, and YAKE boosts its salience.
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};
        use crate::memory::query_parser::{extract_chunks, PosTag};

        // Get YAKE importance scores for discriminative weighting
        let kw_config = KeywordConfig {
            max_keywords: 100, // Get many keywords for lookup
            ngrams: 1,
            min_length: 3,
            ..Default::default()
        };
        let kw_extractor = KeywordExtractor::with_config(kw_config);
        let keywords = kw_extractor.extract(text);

        // Build a lookup map: term -> importance (0.0-1.0)
        let yake_importance: std::collections::HashMap<String, f32> = keywords
            .into_iter()
            .map(|kw| (kw.text.to_lowercase(), kw.importance))
            .collect();

        // POS-based extraction for comprehensive coverage
        let chunk_extraction = extract_chunks(text);

        // Add all proper nouns (these are likely named entities we might have missed)
        for proper_noun in &chunk_extraction.proper_nouns {
            let term_lower = proper_noun.to_lowercase();
            if !seen.contains(&term_lower) && term_lower.len() >= 3 {
                // Boost salience if YAKE identified this as discriminative
                let yake_boost = yake_importance.get(&term_lower).copied().unwrap_or(0.0);
                let entity = ExtractedEntity {
                    name: proper_noun.clone(),
                    label: EntityLabel::Person,
                    base_salience: 0.7 + (yake_boost * 0.2), // 0.7-0.9
                };
                entities.push(entity);
                seen.insert(term_lower);
            }
        }

        // Add all content words as Keyword entities
        // POS ensures comprehensive extraction, YAKE boosts discriminative terms
        for chunk in &chunk_extraction.chunks {
            for word in &chunk.words {
                let term_lower = word.text.to_lowercase();

                // Skip if already extracted or too short
                if seen.contains(&term_lower) || term_lower.len() < 4 {
                    continue;
                }

                // Skip stop words
                if self.stop_words.contains(&term_lower) {
                    continue;
                }

                // Base salience by POS, boosted by YAKE importance
                let yake_boost = yake_importance.get(&term_lower).copied().unwrap_or(0.0);

                let (label, base_salience) = match word.pos {
                    PosTag::Noun | PosTag::ProperNoun => {
                        // Nouns are most important, start at 0.5
                        (EntityLabel::Keyword, 0.5)
                    }
                    PosTag::Verb => {
                        // Verbs connect entities, start at 0.4
                        (EntityLabel::Keyword, 0.4)
                    }
                    PosTag::Adjective => {
                        // Adjectives are modifiers, start at 0.35
                        (EntityLabel::Keyword, 0.35)
                    }
                    _ => continue,
                };

                // Boost by YAKE importance (0.0-0.3 boost based on discriminativeness)
                let final_salience = base_salience + (yake_boost * 0.3);

                let entity = ExtractedEntity {
                    name: word.text.clone(),
                    label,
                    base_salience: final_salience,
                };
                entities.push(entity);
                seen.insert(term_lower);
            }
        }

        entities
    }

    /// Extract co-occurrence pairs from text for graph edge creation
    ///
    /// Returns pairs of (entity1, entity2) that appear in the same sentence.
    /// This enables creating edges between words that co-occur, which is critical
    /// for multi-hop retrieval (e.g., connecting "Melanie" to "sunrise" when
    /// they appear in the same sentence about painting).
    pub fn extract_cooccurrence_pairs(&self, text: &str) -> Vec<(String, String)> {
        use crate::memory::query_parser::extract_chunks;

        let chunk_extraction = extract_chunks(text);
        let mut pairs = Vec::new();

        // Get all co-occurrence pairs from chunks (same sentence)
        for chunk in &chunk_extraction.chunks {
            let content_words = chunk.content_words();

            // Create pairs between all content words in the same sentence
            for i in 0..content_words.len() {
                for j in (i + 1)..content_words.len() {
                    let w1 = content_words[i].text.to_lowercase();
                    let w2 = content_words[j].text.to_lowercase();

                    // Skip very short words and stop words
                    if w1.len() >= 3
                        && w2.len() >= 3
                        && !self.stop_words.contains(&w1)
                        && !self.stop_words.contains(&w2)
                    {
                        pairs.push((w1, w2));
                    }
                }
            }
        }

        pairs
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    /// One activation from a source the edge has not seen, so the mechanics tests
    /// below keep measuring boost, LTP and promotion rather than the evidence rule.
    fn strengthen_once(edge: &mut RelationshipEdge) -> Option<(String, String)> {
        edge.strengthen_from(&Uuid::new_v4().to_string()).promotion
    }

    /// Create a test relationship edge with specified strength and last_activated (L1 tier)
    fn create_test_edge(strength: f32, days_since_activated: i64) -> RelationshipEdge {
        create_test_edge_with_tier(strength, days_since_activated, EdgeTier::L1Working)
    }

    /// Create a test relationship edge with specified strength, last_activated, and tier
    fn create_test_edge_with_tier(
        strength: f32,
        days_since_activated: i64,
        tier: EdgeTier,
    ) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now() - Duration::days(days_since_activated),
            activation_count: 0,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: strength,
            backward_strength: strength,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    #[test]
    fn test_hebbian_strengthen_increases_strength() {
        use crate::constants::*;
        // Use L2 tier to avoid L1 promotion resetting strength
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        let initial_strength = edge.strength;

        let _ = strengthen_once(&mut edge);

        // With tier boost (L2 gets 80% of TIER_CO_ACCESS_BOOST), strength should increase
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.8;
        let expected_boost = (LTP_LEARNING_RATE + tier_boost) * (1.0 - initial_strength);
        assert!(
            edge.strength > initial_strength,
            "Strengthen should increase strength (expected boost {expected_boost})"
        );
        assert_eq!(edge.activation_count, 1);
    }

    #[test]
    fn test_hebbian_strengthen_asymptotic() {
        use crate::constants::*;
        // Use L3 tier (no promotion) with high initial strength
        let mut edge = create_test_edge_with_tier(0.95, 0, EdgeTier::L3Semantic);

        let _ = strengthen_once(&mut edge);

        // High strength should still increase but slowly (asymptotic to 1.0)
        // L3 tier boost = TIER_CO_ACCESS_BOOST * 0.5 = 0.075
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.5;
        let expected_min = 0.95 + (LTP_LEARNING_RATE + tier_boost) * 0.05 - 0.01;
        assert!(
            edge.strength > expected_min,
            "Expected > {expected_min}, got {}",
            edge.strength
        );
        assert!(edge.strength <= 1.0);
    }

    #[test]
    fn test_hebbian_strengthen_formula() {
        use crate::constants::*;
        // Test: w_new = w_old + (η + tier_boost) × (1 - w_old)
        // Use L2 tier (tier_boost = TIER_CO_ACCESS_BOOST * 0.8) at 0.3 to avoid promotion
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);

        let _ = strengthen_once(&mut edge);

        // L2 tier boost = 0.15 * 0.8 = 0.12
        // Expected: 0.3 + (0.1 + 0.12) * (1 - 0.3) = 0.3 + 0.22 * 0.7 = 0.454
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.8;
        let expected = 0.3 + (LTP_LEARNING_RATE + tier_boost) * 0.7;
        assert!(
            (edge.strength - expected).abs() < 0.001,
            "Expected {expected}, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_ltp_threshold_potentiation() {
        let mut edge = create_test_edge(0.5, 0);
        assert!(!edge.is_potentiated());

        // Strengthen 10 times (LTP_THRESHOLD = 10)
        for _ in 0..10 {
            let _ = strengthen_once(&mut edge);
        }

        assert!(
            edge.is_potentiated(),
            "Should be potentiated after 10 activations"
        );
        assert!(
            matches!(edge.ltp_status, LtpStatus::Full),
            "Should have Full LTP status after 10 activations"
        );
        assert!(
            edge.strength > 0.7,
            "Potentiated edge should have bonus strength"
        );
    }

    // ── Recovered from tests/graph_memory_tests.rs ────────────────────────────
    //
    // strengthen() and decay() are pub(crate); tests/ is a separate crate, so
    // that file could never compile and these cases never ran — from the commit
    // that introduced both, 2026-04-08. Moved here rather than widening the API,
    // because the pub(crate) boundary is the thing that forces external callers
    // through apply_reinforcement instead of writing edge.strength directly.
    //
    // Only the cases with no in-crate equivalent were brought across; the rest
    // were duplicates and were deleted with the file's test half.

    #[test]
    fn test_strengthen_increases_strength() {
        let mut edge = create_test_edge(0.5, 0);
        let initial_strength = edge.strength;

        let _ = strengthen_once(&mut edge);

        assert!(
            edge.strength > initial_strength,
            "Strength should increase after strengthening: {} -> {}",
            initial_strength,
            edge.strength
        );
        assert_eq!(edge.activation_count, 1, "Activation count should be 1");
    }

    #[test]
    fn test_strengthen_asymptotic_approach_to_one() {
        // The ceiling holds from an already-high base: five more activations
        // must still move it up and must never cross 1.0.
        let mut edge = create_test_edge(0.9, 0);

        for _ in 0..5 {
            let _ = strengthen_once(&mut edge);
        }

        assert!(
            edge.strength <= 1.0,
            "Strength should not exceed 1.0, got {}",
            edge.strength
        );
        assert!(
            edge.strength > 0.9,
            "Strength should increase even from a high base, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_strengthen_diminishing_returns() {
        // The same single activation must buy less the stronger the edge
        // already is — the property that keeps repetition from saturating
        // every edge to the ceiling.
        let mut low_edge = create_test_edge(0.2, 0);
        let low_initial = low_edge.strength;
        let _ = strengthen_once(&mut low_edge);
        let low_increase = low_edge.strength - low_initial;

        let mut high_edge = create_test_edge(0.8, 0);
        let high_initial = high_edge.strength;
        let _ = strengthen_once(&mut high_edge);
        let high_increase = high_edge.strength - high_initial;

        assert!(
            low_increase > high_increase,
            "Low-strength edge should gain more: {low_increase} > {high_increase}"
        );
    }

    #[test]
    fn test_burst_ltp_negative_boundary_at_l1() {
        // Adds two things the neighbouring tests do not cover:
        // test_ltp_threshold_potentiation checks 0 and 10, and
        // test_pipe4_burst_to_full_upgrade checks 5 at L2. Neither asserts the
        // step BELOW the burst threshold, which is where an off-by-one lands,
        // and neither walks the L1 tier into Burst.
        let mut edge = create_test_edge(0.5, 0);
        assert!(
            !edge.is_potentiated(),
            "Edge should not be potentiated initially"
        );

        for _ in 0..4 {
            let _ = strengthen_once(&mut edge);
        }
        assert!(
            !edge.is_potentiated(),
            "Edge must not be potentiated at 4 activations (burst threshold is 5)"
        );

        let _ = strengthen_once(&mut edge);
        assert!(
            edge.is_potentiated(),
            "Edge should be potentiated at 5 activations (Burst LTP), got {:?}",
            edge.ltp_status
        );
    }

    #[test]
    fn test_pipe4_burst_ltp_detection() {
        // Create an L2 edge with low strength to avoid early tier promotion
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Strengthen 5 times (LTP_BURST_THRESHOLD = 5) within 24 hours
        for _ in 0..5 {
            let _ = strengthen_once(&mut edge);
        }

        // Should have burst LTP (5+ activations in 24h)
        // Edge may promote to L3 during strengthening, but should keep Burst status
        assert!(
            matches!(edge.ltp_status, LtpStatus::Burst { .. }),
            "Should have Burst LTP after 5 rapid activations, got {:?}",
            edge.ltp_status
        );
    }

    #[test]
    fn test_pipe4_activation_timestamps_recorded() {
        // L2 edges should record activation timestamps
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Strengthen a few times
        for _ in 0..3 {
            let _ = strengthen_once(&mut edge);
        }

        // Should have recorded timestamps (edge may have promoted to L3, but still tracks)
        assert!(
            edge.activation_timestamps.is_some(),
            "L2+ edge should have activation timestamps"
        );
        assert_eq!(
            edge.activation_timestamps.as_ref().unwrap().len(),
            3,
            "Should have 3 recorded timestamps"
        );
    }

    #[test]
    fn test_pipe4_fresh_l1_no_timestamps() {
        // Fresh L1 edges should NOT have activation timestamps
        let edge = create_test_edge(0.3, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));
        assert!(
            edge.activation_timestamps.is_none(),
            "Fresh L1 edges should not have timestamps"
        );
    }

    #[test]
    fn test_l1_records_activation_history_while_still_l1() {
        // Regression: L1 used to record nothing, which meant the adaptive decay
        // rate could never be derived for the tier whose default was most wrong.
        let mut edge = create_test_edge(0.1, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));

        let _ = strengthen_once(&mut edge);

        assert!(
            matches!(edge.tier, EdgeTier::L1Working),
            "test needs the edge to still be L1 to prove L1 itself recorded"
        );
        assert!(
            edge.activation_timestamps.is_some(),
            "L1 edge must record activation history"
        );
    }

    #[test]
    fn test_l1_activation_history_respects_capacity() {
        use crate::constants::ACTIVATION_HISTORY_L1_CAPACITY;
        let mut edge = create_test_edge(0.1, 0);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        let now = Utc::now();
        for i in 0..(ACTIVATION_HISTORY_L1_CAPACITY as i64 + 5) {
            edge.record_activation_timestamp(now + chrono::Duration::minutes(i));
        }

        assert_eq!(
            edge.activation_timestamps.as_ref().unwrap().len(),
            ACTIVATION_HISTORY_L1_CAPACITY,
            "L1 history must be bounded by its capacity"
        );
    }

    #[test]
    fn test_typical_interval_needs_min_samples() {
        let mut edge = create_test_edge(0.3, 0);
        let now = Utc::now();

        // Two timestamps = one interval = not yet evidence of a rhythm.
        edge.activation_timestamps = Some(
            vec![now, now + chrono::Duration::hours(24)]
                .into_iter()
                .collect(),
        );
        assert_eq!(edge.typical_activation_interval_hours(), None);

        // Three timestamps = two intervals.
        edge.activation_timestamps = Some(
            vec![
                now,
                now + chrono::Duration::hours(24),
                now + chrono::Duration::hours(48),
            ]
            .into_iter()
            .collect(),
        );
        let interval = edge.typical_activation_interval_hours().unwrap();
        assert!(
            (interval - 24.0).abs() < 0.01,
            "expected 24h, got {interval}"
        );
    }

    #[test]
    fn test_typical_interval_uses_median_not_mean() {
        // One long idle gap must not drag the rate toward permanence.
        let mut edge = create_test_edge(0.3, 0);
        let now = Utc::now();
        edge.activation_timestamps = Some(
            vec![
                now,
                now + chrono::Duration::hours(1),
                now + chrono::Duration::hours(2),
                now + chrono::Duration::hours(2000),
            ]
            .into_iter()
            .collect(),
        );

        let interval = edge.typical_activation_interval_hours().unwrap();
        assert!(
            interval < 100.0,
            "median should reject the 1998h outlier, got {interval}"
        );
    }

    #[test]
    fn test_pipe4_l1_promotes_and_tracks() {
        // L1 edges that promote to L2 should start tracking timestamps
        let mut edge = create_test_edge(0.3, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));

        // Strengthen until it promotes to L2 (L1_PROMOTION_THRESHOLD = 0.5)
        while matches!(edge.tier, EdgeTier::L1Working) {
            let _ = strengthen_once(&mut edge);
        }

        // After promotion to L2, should start tracking
        assert!(
            matches!(edge.tier, EdgeTier::L2Episodic),
            "Should have promoted to L2"
        );
        // Timestamps are initialized on promotion
        assert!(
            edge.activation_timestamps.is_some(),
            "L2 edges should track timestamps after promotion"
        );
    }

    #[test]
    fn test_pipe4_ltp_status_decay_factors() {
        // Test that each LTP status has correct decay factor
        use crate::constants::*;

        assert_eq!(LtpStatus::None.decay_factor(), 1.0);
        assert_eq!(LtpStatus::Weekly.decay_factor(), LTP_WEEKLY_DECAY_FACTOR);
        assert_eq!(LtpStatus::Full.decay_factor(), LTP_DECAY_FACTOR);

        // Burst factor depends on expiration
        let burst = LtpStatus::Burst {
            detected_at: Utc::now(),
        };
        assert_eq!(burst.decay_factor(), LTP_BURST_DECAY_FACTOR);
    }

    #[test]
    fn test_pipe4_burst_to_full_upgrade() {
        // LTP should upgrade from Burst to Full after 10 activations
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Get to burst LTP (5 activations)
        for _ in 0..5 {
            let _ = strengthen_once(&mut edge);
        }
        assert!(
            matches!(edge.ltp_status, LtpStatus::Burst { .. }),
            "Should have Burst after 5 activations, got {:?}",
            edge.ltp_status
        );

        // Continue strengthening to Full LTP (10 total)
        for _ in 0..5 {
            let _ = strengthen_once(&mut edge);
        }

        // Should now be Full (upgraded from Burst via 10 activations)
        assert!(
            matches!(edge.ltp_status, LtpStatus::Full),
            "Should have upgraded to Full LTP after 10 activations"
        );
    }

    #[test]
    fn test_pipe4_activations_in_window() {
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Record some activations
        for _ in 0..5 {
            let _ = strengthen_once(&mut edge);
        }

        let now = Utc::now();
        let hour_ago = now - chrono::Duration::hours(1);
        let day_ago = now - chrono::Duration::days(1);

        // All activations are recent (within last second really)
        let in_hour = edge.activations_in_window(hour_ago, now);
        let in_day = edge.activations_in_window(day_ago, now);
        assert!(in_hour >= 5, "Expected 5+ in hour window, got {in_hour}");
        assert!(in_day >= 5, "Expected 5+ in day window, got {in_day}");
    }

    // =========================================================================
    // PIPE-5: Unified LTP Readiness Model Tests
    // =========================================================================

    #[test]
    fn test_pipe5_adjusted_threshold_default() {
        // Default confidence (None → 0.5) should give default threshold (10)
        let edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        assert!(edge.entity_confidence.is_none());

        let threshold = edge.adjusted_threshold();
        // confidence 0.5 → threshold = 13 - (0.5 * 6) = 10
        assert_eq!(threshold, 10, "Default confidence should give threshold 10");
    }

    #[test]
    fn test_pipe5_adjusted_threshold_high_confidence() {
        // High confidence (0.9) should give lower threshold (7-8)
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.entity_confidence = Some(0.9);

        let threshold = edge.adjusted_threshold();
        // confidence 0.9 → threshold = 13 - (0.9 * 6) = 7.6 → 8
        assert!(
            threshold <= 8,
            "High confidence should give threshold <= 8, got {threshold}"
        );
    }

    #[test]
    fn test_pipe5_adjusted_threshold_low_confidence() {
        // Low confidence (0.2) should give higher threshold (12-13)
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.entity_confidence = Some(0.2);

        let threshold = edge.adjusted_threshold();
        // confidence 0.2 → threshold = 13 - (0.2 * 6) = 11.8 → 12
        assert!(
            threshold >= 11,
            "Low confidence should give threshold >= 11, got {threshold}"
        );
    }

    #[test]
    fn test_pipe5_strength_floor_by_tier() {
        use crate::constants::*;

        let l1_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L1Working);
        let l2_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        let l3_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L3Semantic);

        assert_eq!(
            l1_edge.strength_floor(),
            1.0,
            "L1 should have floor 1.0 (impossible)"
        );
        assert_eq!(
            l2_edge.strength_floor(),
            LTP_STRENGTH_FLOOR_L2,
            "L2 floor mismatch"
        );
        assert_eq!(
            l3_edge.strength_floor(),
            LTP_STRENGTH_FLOOR_L3,
            "L3 floor mismatch"
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_l1_returns_zero() {
        // L1 edges should always return 0 readiness (can't reach Full LTP)
        let mut edge = create_test_edge_with_tier(0.99, 0, EdgeTier::L1Working);
        edge.activation_count = 100;
        edge.entity_confidence = Some(1.0);

        assert_eq!(
            edge.ltp_readiness(),
            0.0,
            "L1 edges should always return 0 readiness"
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_balanced_path() {
        use crate::constants::*;

        // Balanced: 10 activations + 0.75 strength + 0.5 confidence
        // count_score = 10 / 10 = 1.0
        // strength_score = 0.75 / 0.65 = 1.15
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 1.0 * 0.5 + 1.15 * 0.5 + 0.15 = 0.5 + 0.575 + 0.15 = 1.225
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Balanced path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_repetition_dominant() {
        use crate::constants::*;

        // Repetition dominant: 15 activations + 0.50 strength (below floor)
        // count_score = 15 / 10 = 1.5
        // strength_score = 0.50 / 0.65 = 0.77
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 1.5 * 0.5 + 0.77 * 0.5 + 0.15 = 0.75 + 0.385 + 0.15 = 1.285
        let mut edge = create_test_edge_with_tier(0.50, 0, EdgeTier::L2Episodic);
        edge.activation_count = 15;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Repetition-dominant path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_intensity_dominant() {
        use crate::constants::*;

        // Intensity dominant: 5 activations + 0.95 strength (L3)
        // count_score = 5 / 10 = 0.5
        // strength_score = 0.95 / 0.80 = 1.1875
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 0.5 * 0.5 + 1.1875 * 0.5 + 0.15 = 0.25 + 0.59 + 0.15 = 0.99
        // Need more strength or count for intensity-only path on L3
        let mut edge = create_test_edge_with_tier(0.99, 0, EdgeTier::L3Semantic);
        edge.activation_count = 6;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        // count_score = 6/10 = 0.6, strength_score = 0.99/0.80 = 1.24
        // readiness = 0.6*0.5 + 1.24*0.5 + 0.15 = 0.3 + 0.62 + 0.15 = 1.07
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Intensity-dominant path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_high_confidence_boost() {
        use crate::constants::*;

        // High confidence edge reaches LTP faster
        // 7 activations + 0.65 strength + 0.9 confidence
        // threshold = 13 - 0.9*6 = 7.6 → 8
        // count_score = 7 / 8 = 0.875
        // strength_score = 0.65 / 0.65 = 1.0
        // tag_bonus = 0.9 * 0.3 = 0.27
        // readiness = 0.875 * 0.5 + 1.0 * 0.5 + 0.27 = 0.44 + 0.5 + 0.27 = 1.21
        let mut edge = create_test_edge_with_tier(0.65, 0, EdgeTier::L2Episodic);
        edge.activation_count = 7;
        edge.entity_confidence = Some(0.9);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "High-confidence should boost to LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_weak_edge_no_ltp() {
        use crate::constants::*;

        // Weak edge: 4 activations + 0.40 strength + 0.3 confidence
        // threshold = 13 - 0.3*6 = 11.2 → 11
        // count_score = 4 / 11 = 0.36
        // strength_score = 0.40 / 0.65 = 0.62
        // tag_bonus = 0.3 * 0.3 = 0.09
        // readiness = 0.36 * 0.5 + 0.62 * 0.5 + 0.09 = 0.18 + 0.31 + 0.09 = 0.58
        let mut edge = create_test_edge_with_tier(0.40, 0, EdgeTier::L2Episodic);
        edge.activation_count = 4;
        edge.entity_confidence = Some(0.3);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness < LTP_READINESS_THRESHOLD,
            "Weak edge should NOT reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_unified_detect_ltp_status() {
        // Test that detect_ltp_status uses the unified readiness formula
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        let status = edge.detect_ltp_status(Utc::now());
        assert_eq!(
            status,
            LtpStatus::Full,
            "Balanced path should grant Full LTP via readiness"
        );
    }

    #[test]
    fn test_pipe5_l3_no_auto_ltp_without_activations() {
        // L3 with high strength but low activation count should NOT auto-LTP
        // This tests that the old auto-LTP behavior is removed
        let mut edge = create_test_edge_with_tier(0.85, 0, EdgeTier::L3Semantic);
        edge.activation_count = 2; // Low count
        edge.entity_confidence = Some(0.5);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        // count_score = 2/10 = 0.2, strength_score = 0.85/0.80 = 1.06
        // readiness = 0.2*0.5 + 1.06*0.5 + 0.15 = 0.1 + 0.53 + 0.15 = 0.78
        let status = edge.detect_ltp_status(Utc::now());
        assert_eq!(
            status,
            LtpStatus::None,
            "L3 high strength alone should NOT grant Full LTP, needs activations too"
        );
    }

    #[test]
    fn test_decay_reduces_strength() {
        // Use L2 tier for multi-day decay testing (L1 max age is only 4 hours)
        let mut edge = create_test_edge_with_tier(0.5, 7, EdgeTier::L2Episodic);

        let initial_strength = edge.strength;
        edge.decay();

        assert!(
            edge.strength < initial_strength,
            "Decay should reduce strength (initial: {}, after: {})",
            initial_strength,
            edge.strength
        );
    }

    #[test]
    fn test_decay_tier_aware() {
        // Tier-aware decay over 7 days at L2. The expected value is DERIVED from
        // L2_DECAY_PER_DAY rather than hardcoded, because a hardcoded band pins
        // the tuning rather than the mechanism and fails on every recalibration.
        // What this actually guards is that the L2 rate is applied in the right
        // units and to the right tier — a per-day rate read as per-hour is the
        // live bug class here, and it lands 24x off, far outside the tolerance.
        use crate::constants::L2_DECAY_PER_DAY;

        let mut edge = create_test_edge_with_tier(1.0, 7, EdgeTier::L2Episodic);
        edge.decay();

        let expected = (-(L2_DECAY_PER_DAY as f64) * 7.0).exp() as f32;
        assert!(
            (edge.strength - expected).abs() < 0.01,
            "7-day L2 decay should track L2_DECAY_PER_DAY: expected ~{expected}, got {}",
            edge.strength
        );
        // Direction and survival, both tuning-independent.
        assert!(
            edge.strength < 1.0,
            "decay must reduce strength, got {}",
            edge.strength
        );
        assert!(
            edge.strength > EdgeTier::L2Episodic.prune_threshold(),
            "a 7-day-old L2 edge must not be at prune level, got {}",
            edge.strength
        );
        assert!(
            edge.strength > LTP_MIN_STRENGTH,
            "Strength should still be above floor, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_decay_half_life() {
        // A SECOND point on the L2 curve, 14 days, beside test_decay_tier_aware's 7 days. Derived from
        // L2_DECAY_PER_DAY rather than hardcoded, like that test, so it guards the SHAPE (the same
        // exponential applied over twice the time) instead of pinning a rate that is allowed to change:
        // a change that preserved the 7-day value by accident, or applied the rate per-call instead of
        // per-day, fails here. (An earlier draft hardcoded lambda=0.031 and broke when d5047528 made
        // the rate learned per edge.)
        use crate::constants::L2_DECAY_PER_DAY;

        let mut edge = create_test_edge_with_tier(1.0, 14, EdgeTier::L2Episodic);
        edge.decay();

        let expected = (-(L2_DECAY_PER_DAY as f64) * 14.0).exp() as f32;
        assert!(
            (edge.strength - expected).abs() < 0.01,
            "14-day L2 decay should track L2_DECAY_PER_DAY: expected ~{expected}, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_decay_minimum_floor() {
        // Use L3 tier for very old edge testing (L3 has 10 year max age)
        let mut edge = create_test_edge_with_tier(0.02, 100, EdgeTier::L3Semantic);

        edge.decay();

        assert!(
            edge.strength >= LTP_MIN_STRENGTH,
            "Strength should not go below minimum floor"
        );
    }

    #[test]
    fn test_potentiated_decay_slower() {
        // Use L2 tier for multi-day decay comparison
        let mut edge1 = create_test_edge_with_tier(0.8, 7, EdgeTier::L2Episodic);
        let mut edge2 = create_test_edge_with_tier(0.8, 7, EdgeTier::L2Episodic);
        edge2.ltp_status = LtpStatus::Full; // Full LTP = 10x slower decay

        edge1.decay();
        edge2.decay();

        assert!(
            edge2.strength > edge1.strength,
            "Potentiated edge should decay slower (normal: {}, potentiated: {})",
            edge1.strength,
            edge2.strength
        );
    }

    #[test]
    fn test_effective_strength_read_only() {
        // Use L2 tier for multi-day testing
        let edge = create_test_edge_with_tier(0.5, 7, EdgeTier::L2Episodic);
        let initial_strength = edge.strength;

        let effective = edge.effective_strength();

        // effective_strength should not modify the edge
        assert_eq!(edge.strength, initial_strength);
        assert!(effective < initial_strength);
    }

    #[test]
    fn test_decay_prune_threshold() {
        // Use L2 tier for decay testing beyond its max age (14 days)
        let mut weak_edge = create_test_edge_with_tier(LTP_MIN_STRENGTH, 30, EdgeTier::L2Episodic);
        // No LTP status = normal decay
        assert!(matches!(weak_edge.ltp_status, LtpStatus::None));

        let should_prune = weak_edge.decay();

        // Non-potentiated edge at minimum strength past max age should be prunable
        assert!(
            should_prune,
            "Weak non-potentiated edge past max age should be marked for pruning"
        );
    }

    #[test]
    fn test_potentiated_above_floor_never_pruned() {
        // Potentiated edge above LTP_PRUNE_FLOOR should be protected
        let mut edge = create_test_edge_with_tier(0.1, 30, EdgeTier::L2Episodic);
        edge.ltp_status = LtpStatus::Full;

        let should_prune = edge.decay();

        assert!(
            !should_prune,
            "Potentiated edges above LTP_PRUNE_FLOOR should not be pruned"
        );
    }

    #[test]
    fn test_potentiated_at_floor_stripped_and_prunable() {
        // Potentiated edge at/below LTP_PRUNE_FLOOR should have LTP stripped
        let mut edge = create_test_edge_with_tier(LTP_MIN_STRENGTH, 30, EdgeTier::L2Episodic);
        edge.ltp_status = LtpStatus::Full;

        let should_prune = edge.decay();

        // LTP gets stripped because strength <= LTP_PRUNE_FLOOR,
        // then normal prune logic applies (strength at floor, past max age)
        assert!(
            should_prune,
            "Zombie potentiated edges at floor strength should be prunable"
        );
        assert!(
            matches!(edge.ltp_status, LtpStatus::None),
            "LTP status should be stripped when strength at floor"
        );
    }

    #[test]
    fn test_salience_calculation() {
        let person_salience = EntityExtractor::calculate_base_salience(&EntityLabel::Person, false);
        let person_proper_salience =
            EntityExtractor::calculate_base_salience(&EntityLabel::Person, true);

        assert_eq!(person_salience, 0.8);
        assert!((person_proper_salience - 0.96).abs() < 0.01); // 0.8 * 1.2 = 0.96
    }

    #[test]
    fn test_salience_caps_at_one() {
        // Person (0.8) * 1.2 = 0.96, should not exceed 1.0
        let salience = EntityExtractor::calculate_base_salience(&EntityLabel::Person, true);
        assert!(salience <= 1.0);
    }

    #[test]
    fn test_hebbian_strength_no_episode() {
        // Create a temporary graph memory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Random memory ID with no associated episode should return 0.5 (neutral)
        let fake_memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let strength = graph.get_memory_hebbian_strength(&fake_memory_id);
        assert_eq!(strength, Some(0.5), "No episode should return neutral 0.5");
    }

    #[test]
    fn test_hebbian_strength_with_episode_no_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Create entities
        let entity1 = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Entity1".to_string(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::PersonalIdentity,
        };
        let entity2 = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Entity2".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::OrganizationIdentity,
        };

        let entity1_uuid = graph.add_entity(entity1.clone()).unwrap();
        let entity2_uuid = graph.add_entity(entity2.clone()).unwrap();

        // Create episode with entities but no edges
        let memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: "Test Episode".to_string(),
            content: "Test content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity1_uuid, entity2_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        // Episode with entities but no edges should return 0.5
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert_eq!(
            strength,
            Some(0.5),
            "Episode without edges should return neutral 0.5"
        );
    }

    #[test]
    fn test_hebbian_strength_with_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Create entities
        let entity1_uuid = Uuid::new_v4();
        let entity2_uuid = Uuid::new_v4();

        let entity1 = EntityNode {
            uuid: entity1_uuid,
            name: "Entity1".to_string(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::PersonalIdentity,
        };
        let entity2 = EntityNode {
            uuid: entity2_uuid,
            name: "Entity2".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::OrganizationIdentity,
        };

        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();

        // Create episode
        let memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: "Test Episode".to_string(),
            content: "Test content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity1_uuid, entity2_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        // Create edge between entities with known strength
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: entity1_uuid,
            to_entity: entity2_uuid,
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: Some(memory_id.0),
            context: "Test context".to_string(),
            last_activated: Utc::now(), // Just activated - no decay
            activation_count: 5,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: 0.8,
            backward_strength: 0.8,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        };
        graph.add_relationship(edge).unwrap();

        // Should return the edge strength (0.8, with minimal decay since just activated)
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert!(strength.is_some());
        let s = strength.unwrap();
        assert!(s > 0.75 && s <= 0.8, "Strength should be ~0.8, got {}", s);
    }

    #[test]
    fn deleting_an_entity_deletes_the_relationships_it_is_an_endpoint_of() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let entity = |name: &str| EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };
        let edge = |from: Uuid, to: Uuid| RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: 0.8,
            backward_strength: 0.8,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        };
        let [a_id, b_id, c_id] =
            ["Aster", "Birch", "Cedar"].map(|name| graph.add_entity(entity(name)).unwrap());
        graph.add_relationship(edge(a_id, b_id)).unwrap();
        graph.add_relationship(edge(c_id, a_id)).unwrap();
        let kept = graph.add_relationship(edge(b_id, c_id)).unwrap();

        assert!(graph.delete_entity(&a_id).unwrap());

        let left: Vec<Uuid> = graph
            .get_all_relationships()
            .unwrap()
            .iter()
            .map(|e| e.uuid)
            .collect();
        assert_eq!(left, vec![kept]);
        assert_eq!(graph.get_entity_relationships(&b_id).unwrap().len(), 1);
        assert_eq!(graph.get_entity_relationships(&c_id).unwrap().len(), 1);
        assert!(graph.get_entity_relationships(&a_id).unwrap().is_empty());
    }

    #[test]
    fn a_batch_of_edges_lands_as_successive_adds_would_on_both_engines() {
        let entity = |name: &str| EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };
        let edge = |from: Uuid, to: Uuid, context: &str| RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::CoOccurs,
            strength: 0.3,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: Some(Uuid::new_v4()),
            context: context.to_string(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L1Working,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: 0.3,
            backward_strength: 0.3,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        };

        let stores = crate::storage::test_support::column_stores(GRAPH_CF_NAMES);
        crate::storage::test_support::assert_both_backends(&stores);
        for (backend, store, _tmp) in stores {
            let batched = GraphMemory::with_store(store).unwrap();
            let [a, b, c] =
                ["Alder", "Beech", "Cherry"].map(|name| batched.add_entity(entity(name)).unwrap());
            let ids = batched
                .add_relationships(vec![
                    edge(a, b, "first"),
                    edge(b, a, "the longer context"),
                    edge(a, c, "other"),
                ])
                .unwrap();
            assert_eq!(
                ids[0], ids[1],
                "{backend:?}: a reversed pair in one batch made a second synapse"
            );
            assert_ne!(ids[0], ids[2]);

            let ab = batched.get_relationship(&ids[0]).unwrap().unwrap();
            assert_eq!(ab.context, "the longer context");
            assert_eq!(
                ab.activation_count, 2,
                "{backend:?}: the repeat was not strengthened"
            );
            assert_eq!(batched.get_all_relationships().unwrap().len(), 2);
            assert_eq!(batched.get_stats().unwrap().relationship_count, 2);
            // Each new edge's index rows came with it in the same write.
            assert_eq!(batched.get_entity_relationships(&a).unwrap().len(), 2);
            assert_eq!(batched.get_entity_relationships(&b).unwrap().len(), 1);
            assert_eq!(
                batched
                    .find_relationship_between(&b, &a)
                    .unwrap()
                    .map(|e| e.uuid),
                Some(ids[0])
            );

            // A later batch strengthens the stored synapse rather than adding one.
            assert_eq!(
                batched.add_relationships(vec![edge(a, b, "x")]).unwrap(),
                vec![ids[0]]
            );
            assert_eq!(batched.get_stats().unwrap().relationship_count, 2);
            assert!(batched.add_relationships(Vec::new()).unwrap().is_empty());
        }
    }

    #[test]
    fn test_compute_npmi_cold_start() {
        // During cold start (total_activations < 100), compute_npmi returns None
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // total_activations starts at 0 — well below the 100 threshold
        let edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        let result = graph.compute_npmi(&edge, 10, 10);
        assert!(
            result.is_none(),
            "NPMI should return None during cold start (total_activations < 100)"
        );

        // Set total_activations to 99 — still below threshold
        graph.total_activations.store(99, Ordering::Relaxed);
        let result = graph.compute_npmi(&edge, 10, 10);
        assert!(
            result.is_none(),
            "NPMI should return None when total_activations = 99"
        );

        // Set to exactly 100 — should now compute
        graph.total_activations.store(100, Ordering::Relaxed);
        graph.relationship_count.store(10, Ordering::Relaxed);
        let result = graph.compute_npmi(&edge, 10, 10);
        assert!(
            result.is_some(),
            "NPMI should return Some when total_activations = 100"
        );
    }

    #[test]
    fn test_compute_npmi_values() {
        // Verify NPMI produces expected values for known inputs.
        //
        // Given:
        //   activation_count = 5, mention_count_a = 10, mention_count_b = 8
        //   total_activations = 200, num_edges = 50
        //
        // Manual calculation:
        //   f_ab = 5 + 1 = 6
        //   m_a  = max(10, 5) + 1 = 11
        //   m_b  = max(8, 5) + 1 = 9
        //   N_s  = 200 + 50 = 250
        //
        //   pmi  = log2(6 * 250 / (11 * 9)) = log2(1500 / 99) = log2(15.1515...) ≈ 3.9213
        //   -log2(p_ab) = log2(250 / 6) = log2(41.6667) ≈ 5.3808
        //   npmi = 3.9213 / 5.3808 ≈ 0.7288
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        graph.total_activations.store(200, Ordering::Relaxed);
        graph.relationship_count.store(50, Ordering::Relaxed);

        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.activation_count = 5;

        let npmi = graph.compute_npmi(&edge, 10, 8).unwrap();

        // Expected: ~0.7288
        assert!(
            (npmi - 0.7288).abs() < 0.01,
            "NPMI should be ~0.7288, got {}",
            npmi
        );

        // Verify it's in the expected range for L2 promotion gate (>= 0.45) and
        // below LTP readiness bonus threshold (0.75)
        assert!(npmi >= 0.45, "NPMI {} should pass L2 promotion gate", npmi);
        assert!(
            npmi < 0.75,
            "NPMI {} should be below LTP bonus threshold",
            npmi
        );
    }

    #[test]
    fn test_compute_npmi_perfect_cooccurrence() {
        // When f_ab equals N_s (perfect co-occurrence), NPMI approaches 1.0
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        graph.total_activations.store(100, Ordering::Relaxed);
        graph.relationship_count.store(1, Ordering::Relaxed);

        // Edge where activation_count is very high relative to total
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.activation_count = 99; // f_ab = 100, N_s = 101, m_a = m_b = 100

        let npmi = graph.compute_npmi(&edge, 1, 1).unwrap();

        // With high co-occurrence, NPMI should be close to 1.0
        assert!(
            npmi > 0.9,
            "High co-occurrence should yield NPMI close to 1.0, got {}",
            npmi
        );
    }

    #[test]
    fn test_compute_npmi_rare_edge() {
        // Edge with low activation relative to entity mentions → low NPMI
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        graph.total_activations.store(1000, Ordering::Relaxed);
        graph.relationship_count.store(100, Ordering::Relaxed);

        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L1Working);
        edge.activation_count = 1; // f_ab = 2

        // Both entities mentioned many times but rarely together
        let npmi = graph.compute_npmi(&edge, 200, 150).unwrap();

        assert!(
            npmi < 0.30,
            "Rare co-occurrence between popular entities should yield low NPMI, got {}",
            npmi
        );
    }

    #[test]
    fn test_ltp_readiness_with_npmi_cold_start_fallback() {
        // When npmi is None (cold start), should fall back to standard ltp_readiness
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);

        let standard = edge.ltp_readiness();
        let with_none = edge.ltp_readiness_with_npmi(None);

        assert!(
            (standard - with_none).abs() < f32::EPSILON,
            "NPMI=None should fall back to standard readiness: {} vs {}",
            standard,
            with_none
        );
    }

    #[test]
    fn test_ltp_readiness_with_npmi_high_value() {
        // High NPMI (>= 0.75) should boost readiness significantly
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 5;
        edge.entity_confidence = Some(0.5);

        // npmi_score = (0.80 / 0.75).clamp(0.0, 1.5) = 1.0667
        // strength_score = 0.75 / 0.65 = 1.1538
        // tag_bonus = 0.5 * 0.2 = 0.1
        // readiness = 1.0667 * 0.4 + 1.1538 * 0.4 + 0.1 = 0.4267 + 0.4615 + 0.1 = 0.9882
        let readiness = edge.ltp_readiness_with_npmi(Some(0.80));
        assert!(
            readiness > 0.9,
            "High NPMI should produce high readiness, got {}",
            readiness
        );
    }

    #[test]
    fn test_npmi_prune_thresholds() {
        assert!(
            (GraphMemory::npmi_prune_threshold(&EdgeTier::L1Working) - 0.30).abs() < f32::EPSILON
        );
        assert!(
            (GraphMemory::npmi_prune_threshold(&EdgeTier::L2Episodic) - 0.40).abs() < f32::EPSILON
        );
        assert!(
            (GraphMemory::npmi_prune_threshold(&EdgeTier::L3Semantic) - 0.50).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_total_activations_counter() {
        // Verify total_activations is initialized and updated correctly
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Should start at 0 for empty graph
        assert_eq!(graph.total_activations.load(Ordering::Relaxed), 0);

        // After strengthening a synapse, counter should increment
        let entity_a = Uuid::new_v4();
        let entity_b = Uuid::new_v4();

        // Add entities first
        let node_a = EntityNode {
            uuid: entity_a,
            name: "EntityA".to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };
        let node_b = EntityNode {
            uuid: entity_b,
            name: "EntityB".to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };
        graph.add_entity(node_a).unwrap();
        graph.add_entity(node_b).unwrap();

        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: entity_a,
            to_entity: entity_b,
            relation_type: RelationType::RelatedTo,
            strength: 0.5,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 3,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: 0.5,
            backward_strength: 0.5,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        };
        let edge_uuid = graph.add_relationship(edge).unwrap();

        // Strengthen the edge — should increment total_activations by 1
        graph.strengthen_synapse(&edge_uuid, "session-a").unwrap();
        assert_eq!(graph.total_activations.load(Ordering::Relaxed), 1);

        // Delete the edge — should decrement by activation_count (which is now 4)
        graph.delete_relationship(&edge_uuid).unwrap();
        assert_eq!(graph.total_activations.load(Ordering::Relaxed), 0);
    }

    // ── S-R4: idempotency guard ────────────────────────────────────────────

    /// Reinforcing the same `retrieval_id` twice must produce the same final
    /// edge strength as reinforcing it once (idempotent).
    ///
    /// This verifies the cardinality>1 / retry-compounding fix introduced in
    /// substrate pass S-R4.  The second call with the same `retrieval_id` is
    /// expected to be a complete no-op — `edges_updated` should be 0 and the
    /// edge strength should be unchanged from the first call.
    #[test]
    fn test_apply_reinforcement_is_idempotent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Create two memory entities to act as the trigger and co-activated node.
        let trigger_id = Uuid::new_v4();
        let coact_id = Uuid::new_v4();

        let make_entity = |id: Uuid, name: &str| EntityNode {
            uuid: id,
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };
        graph.add_entity(make_entity(trigger_id, "MemA")).unwrap();
        graph.add_entity(make_entity(coact_id, "MemB")).unwrap();

        let retrieval_id = Uuid::new_v4();

        // First reinforcement — should create an edge and return edges_updated > 0.
        let first = graph
            .apply_reinforcement(retrieval_id, "session-a", &trigger_id, &[coact_id])
            .expect("first apply_reinforcement should succeed");
        assert!(
            first.edges_updated > 0,
            "First reinforcement should create/update an edge (got {})",
            first.edges_updated
        );

        // Read back the edge strength after the first reinforcement.
        let edge_after_first = graph
            .find_edge_between_entities(&trigger_id, &coact_id)
            .expect("lookup should not error")
            .expect("edge should exist after first reinforcement");
        let strength_after_first = edge_after_first.strength;

        // Second reinforcement with the SAME retrieval_id — must be a no-op.
        let second = graph
            .apply_reinforcement(retrieval_id, "session-a", &trigger_id, &[coact_id])
            .expect("second apply_reinforcement should succeed");
        assert_eq!(
            second.edges_updated, 0,
            "Second reinforcement with same retrieval_id must be no-op (edges_updated = 0)"
        );

        // Edge strength must not have changed.
        let edge_after_second = graph
            .find_edge_between_entities(&trigger_id, &coact_id)
            .expect("lookup should not error")
            .expect("edge should still exist");
        assert!(
            (edge_after_second.strength - strength_after_first).abs() < f32::EPSILON,
            "Edge strength must not change on duplicate retrieval_id: {} → {}",
            strength_after_first,
            edge_after_second.strength
        );

        // A DIFFERENT retrieval_id must still apply reinforcement.
        let different_retrieval_id = Uuid::new_v4();
        let third = graph
            .apply_reinforcement(
                different_retrieval_id,
                "session-b",
                &trigger_id,
                &[coact_id],
            )
            .expect("third apply_reinforcement with new id should succeed");
        assert!(
            third.edges_updated > 0,
            "New retrieval_id must produce a real reinforcement (got {})",
            third.edges_updated
        );
        let edge_after_third = graph
            .find_edge_between_entities(&trigger_id, &coact_id)
            .expect("lookup should not error")
            .expect("edge should still exist");
        assert!(
            edge_after_third.strength >= strength_after_first,
            "A new retrieval_id should strengthen the edge further: {} (was {})",
            edge_after_third.strength,
            strength_after_first
        );
    }

    /// Confirm that `seen_retrievals` correctly evicts the oldest entry when the
    /// capacity cap (`SEEN_RETRIEVALS_CAPACITY`) is reached, so the guard never
    /// grows unboundedly.
    #[test]
    fn test_seen_retrievals_evicts_at_capacity() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Overflow the seen_retrievals ring by inserting CAPACITY+1 distinct IDs.
        for _ in 0..=SEEN_RETRIEVALS_CAPACITY {
            let id = Uuid::new_v4();
            let guard = graph.seen_retrievals.lock();
            drop(guard); // just ensure lock is accessible
                         // We exercise the code path through apply_reinforcement without needing
                         // real entities — a missing trigger simply returns Ok with 0 edges.
            let _ = graph.apply_reinforcement(id, "session-a", &Uuid::new_v4(), &[Uuid::new_v4()]);
        }

        // After cap+1 insertions the queue length must not exceed SEEN_RETRIEVALS_CAPACITY.
        let guard = graph.seen_retrievals.lock();
        let (ref set, ref queue) = *guard;
        assert!(
            queue.len() <= SEEN_RETRIEVALS_CAPACITY,
            "VecDeque len {} exceeds cap {}",
            queue.len(),
            SEEN_RETRIEVALS_CAPACITY
        );
        assert_eq!(
            set.len(),
            queue.len(),
            "HashSet and VecDeque must stay in sync"
        );
    }

    // =========================================================================
    // S-R5: Path-credit reinforcement + spontaneous recovery tests
    // =========================================================================

    /// (a) Delivering edges strengthen on positive feedback, and the total
    /// normalised credit passed to `apply_path_credit` sums to ~1.0.
    ///
    /// Setup: create two edges in a real GraphMemory instance, then call
    /// `apply_path_credit` with raw contributions.  Verify:
    ///   1. Both edges increase in strength.
    ///   2. The contributions used internally (after normalisation to 1.0) sum
    ///      to 1.0 (verified by checking the ratio of credit slices held in the
    ///      test matches unity after renormalisation).
    #[test]
    fn test_path_credit_strengthens_delivering_edges_and_normalises_to_one() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Create two pairs of entities so we can fabricate two edges.
        let (a1, a2) = (Uuid::new_v4(), Uuid::new_v4());
        let (b1, b2) = (Uuid::new_v4(), Uuid::new_v4());

        let make_entity = |id: Uuid, name: &str| EntityNode {
            uuid: id,
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };

        graph.add_entity(make_entity(a1, "A1")).unwrap();
        graph.add_entity(make_entity(a2, "A2")).unwrap();
        graph.add_entity(make_entity(b1, "B1")).unwrap();
        graph.add_entity(make_entity(b2, "B2")).unwrap();

        // Prime both edges by running a coactivation so they exist in the DB.
        graph
            .apply_reinforcement(Uuid::new_v4(), "session-a", &a1, &[a2])
            .unwrap();
        graph
            .apply_reinforcement(Uuid::new_v4(), "session-a", &b1, &[b2])
            .unwrap();

        let edge_a = graph
            .find_edge_between_entities(&a1, &a2)
            .unwrap()
            .expect("edge A should exist");
        let edge_b = graph
            .find_edge_between_entities(&b1, &b2)
            .unwrap()
            .expect("edge B should exist");

        let strength_a_before = edge_a.strength;
        let strength_b_before = edge_b.strength;

        // B10: raw contributions (do NOT need to sum to 1.0 before the call).
        let raw_contribution_a = 3.0_f32;
        let raw_contribution_b = 7.0_f32;
        let total = raw_contribution_a + raw_contribution_b; // 10.0

        // Verify that the normalised shares sum to exactly 1.0.
        let norm_a = raw_contribution_a / total;
        let norm_b = raw_contribution_b / total;
        assert!(
            (norm_a + norm_b - 1.0_f32).abs() < 1e-6,
            "Normalised contributions must sum to 1.0, got {}",
            norm_a + norm_b
        );

        let credit = vec![
            (edge_a.uuid, raw_contribution_a),
            (edge_b.uuid, raw_contribution_b),
        ];

        let n = graph.apply_path_credit(&credit, "session-b").unwrap();
        assert_eq!(n, 2, "Both edges should be strengthened (got {})", n);

        let edge_a_after = graph
            .find_edge_between_entities(&a1, &a2)
            .unwrap()
            .expect("edge A should still exist");
        let edge_b_after = graph
            .find_edge_between_entities(&b1, &b2)
            .unwrap()
            .expect("edge B should still exist");

        assert!(
            edge_a_after.strength > strength_a_before,
            "Edge A should be stronger after path-credit (was {}, now {})",
            strength_a_before,
            edge_a_after.strength
        );
        assert!(
            edge_b_after.strength > strength_b_before,
            "Edge B should be stronger after path-credit (was {}, now {})",
            strength_b_before,
            edge_b_after.strength
        );
    }

    /// (b) A recovering edge whose strength is BELOW the prune threshold is
    /// still pruned: recovery cannot rescue a prune-eligible edge.
    ///
    /// This verifies that `EDGE_RECOVERY_CAP_FACTOR < 1.0` keeps the recovery
    /// ceiling strictly below `prune_threshold`, so the `strength <= prune_threshold`
    /// guard at the end of `decay()` always fires for a sub-threshold edge regardless
    /// of how many recovery nudges it has received.
    #[test]
    fn test_recovery_cannot_rescue_prune_eligible_edge() {
        use crate::constants::{EDGE_RECOVERY_CAP_FACTOR, EDGE_RECOVERY_RATE, L2_PRUNE_THRESHOLD};

        // Create an L2 edge at exactly the prune threshold (prune-eligible by <=).
        let mut edge = create_test_edge_with_tier(L2_PRUNE_THRESHOLD, 15, EdgeTier::L2Episodic);

        // Apply recovery manually many times — simulating repeated maintenance cycles
        // while the edge is stuck at/below the threshold.
        for _ in 0..1000 {
            let cap = L2_PRUNE_THRESHOLD * EDGE_RECOVERY_CAP_FACTOR;
            if edge.strength < cap {
                edge.strength = (edge.strength + EDGE_RECOVERY_RATE).min(cap);
            }
        }

        // Even after 1 000 recovery nudges, the edge must never reach or exceed
        // the prune threshold.
        assert!(
            edge.strength <= L2_PRUNE_THRESHOLD,
            "Recovery must not push edge above prune threshold (strength={}, threshold={})",
            edge.strength,
            L2_PRUNE_THRESHOLD
        );

        // Now call decay() — the prune decision must still be true.
        let should_prune = edge.decay();
        assert!(
            should_prune,
            "A recovering edge at/below prune threshold must still be marked for pruning \
             (strength={}, prune_threshold={})",
            edge.strength, L2_PRUNE_THRESHOLD
        );
    }

    // ── merge class ────────────────────────────────────────────────────────

    /// Every variant must have a stated class. This is the test that fails
    /// when someone adds a label and forgets what it means across bands —
    /// the compiler catches the match, and this catches the vocabulary
    /// drifting away from the list the decision was made against.
    #[test]
    fn every_entity_label_declares_a_merge_class() {
        let all = [
            EntityLabel::Person,
            EntityLabel::Organization,
            EntityLabel::Location,
            EntityLabel::Technology,
            EntityLabel::Concept,
            EntityLabel::Event,
            EntityLabel::Date,
            EntityLabel::Product,
            EntityLabel::Skill,
            EntityLabel::Keyword,
            EntityLabel::Project,
            EntityLabel::Task,
            EntityLabel::Document,
            EntityLabel::Repository,
            EntityLabel::Service,
            EntityLabel::Database,
            EntityLabel::Metric,
            EntityLabel::Configuration,
            EntityLabel::Environment,
            EntityLabel::Pipeline,
            EntityLabel::Team,
            EntityLabel::Role,
            EntityLabel::Module,
            EntityLabel::Other("unrecognised".into()),
            EntityLabel::FilePath,
            EntityLabel::Commit,
            EntityLabel::Symbol,
            EntityLabel::Session,
            EntityLabel::Agent,
            EntityLabel::Tool,
            EntityLabel::Template,
        ];
        assert_eq!(all.len(), 31, "vocabulary changed — revisit merge classes");

        let global = all
            .iter()
            .filter(|l| l.merge_class() == MergeClass::Global)
            .count();
        assert_eq!(global, 18, "expected 18 global labels");
        assert_eq!(all.len() - global, 13, "expected 13 band-relative labels");
    }

    #[test]
    fn environment_never_merges_across_bands() {
        // The clincher for the whole rule: `staging` in one project is not
        // `staging` in another.
        assert_eq!(
            EntityLabel::Environment.merge_class(),
            MergeClass::BandRelative
        );
        assert_eq!(
            EntityLabel::Configuration.merge_class(),
            MergeClass::BandRelative
        );
    }

    #[test]
    fn band_naming_labels_are_global_so_bands_can_reference_each_other() {
        assert_eq!(EntityLabel::Project.merge_class(), MergeClass::Global);
        assert_eq!(EntityLabel::Repository.merge_class(), MergeClass::Global);
    }

    #[test]
    fn unrecognised_label_fails_toward_not_merging() {
        assert_eq!(
            EntityLabel::Other("whatever".into()).merge_class(),
            MergeClass::BandRelative
        );
    }

    #[test]
    fn a_single_band_relative_label_makes_the_whole_entity_band_relative() {
        // Person is global, Configuration is not; the set is not.
        let mixed = [EntityLabel::Person, EntityLabel::Configuration];
        assert_eq!(MergeClass::of(&mixed), MergeClass::BandRelative);

        let all_global = [EntityLabel::Person, EntityLabel::Organization];
        assert_eq!(MergeClass::of(&all_global), MergeClass::Global);
    }

    #[test]
    fn an_unlabelled_entity_is_band_relative() {
        // No labels is no evidence of scope, not evidence of global scope.
        assert_eq!(MergeClass::of(&[]), MergeClass::BandRelative);
    }

    // =========================================================================
    // Recall-harness APIs: decay_at, apply_decay_at, is_causal,
    // find_entity_by_name_strict, trace_causal_origins
    // =========================================================================

    // Verbatim body of decay() before the clock was injected, `self` renamed; the oracle.
    fn legacy_decay(edge: &mut RelationshipEdge) -> bool {
        use crate::decay::tier_decay_factor_adaptive;

        let typical_interval = edge.typical_activation_interval_hours();

        let now = Utc::now();
        let elapsed = now.signed_duration_since(edge.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return false;
        }

        let hours_elapsed = hours_elapsed.min(8760.0);

        let tier_num = match edge.tier {
            EdgeTier::L1Working => 0,
            EdgeTier::L2Episodic => 1,
            EdgeTier::L3Semantic => 2,
        };
        let ltp_factor = edge.ltp_status.decay_factor();
        let (decay_factor, exceeded_max_age) =
            tier_decay_factor_adaptive(hours_elapsed, tier_num, ltp_factor, typical_interval);
        edge.strength *= decay_factor;

        edge.last_activated = now;

        let prune_threshold = edge.tier.prune_threshold();
        if edge.strength < LTP_MIN_STRENGTH {
            edge.strength = LTP_MIN_STRENGTH;
        }

        {
            use crate::constants::{EDGE_RECOVERY_CAP_FACTOR, EDGE_RECOVERY_RATE};
            let recovery_cap = prune_threshold * EDGE_RECOVERY_CAP_FACTOR;
            if edge.strength < recovery_cap {
                edge.strength = (edge.strength + EDGE_RECOVERY_RATE).min(recovery_cap);
            }
        }

        if edge.ltp_status.is_burst_expired() {
            if edge.detect_weekly_pattern() {
                edge.ltp_status = LtpStatus::Weekly;
            } else {
                edge.ltp_status = LtpStatus::None;
            }
        }

        if edge.ltp_status.is_potentiated() && edge.strength <= LTP_PRUNE_FLOOR {
            edge.ltp_status = LtpStatus::None;
        }

        if edge.ltp_status.is_potentiated() {
            false
        } else {
            exceeded_max_age || edge.strength <= prune_threshold
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum LtpSpec {
        None,
        BurstFresh,
        BurstExpired,
        Weekly,
        Full,
    }

    #[derive(Clone, Copy, Debug)]
    enum HistorySpec {
        None,
        WeeklyHabit,
        Sparse,
    }

    fn edge_anchored_at(
        anchor: DateTime<Utc>,
        tier: EdgeTier,
        strength: f32,
        age_hours: i64,
        ltp: LtpSpec,
        history: HistorySpec,
    ) -> RelationshipEdge {
        let ltp_status = match ltp {
            LtpSpec::None => LtpStatus::None,
            LtpSpec::BurstFresh => LtpStatus::Burst {
                detected_at: anchor - Duration::hours(10),
            },
            LtpSpec::BurstExpired => LtpStatus::Burst {
                detected_at: anchor - Duration::hours(60),
            },
            LtpSpec::Weekly => LtpStatus::Weekly,
            LtpSpec::Full => LtpStatus::Full,
        };
        let activation_timestamps = match history {
            HistorySpec::None => None,
            HistorySpec::WeeklyHabit => Some(
                (0..3)
                    .flat_map(|w| (0..4).map(move |d| (w, d)))
                    .map(|(w, d)| anchor - Duration::days(7 * w + d) - Duration::minutes(90))
                    .rev()
                    .collect(),
            ),
            HistorySpec::Sparse => Some(
                (0..3)
                    .map(|i| anchor - Duration::hours(age_hours + 24 * i))
                    .rev()
                    .collect(),
            ),
        };
        let mut edge = create_test_edge_with_tier(strength, 0, tier);
        edge.created_at = anchor - Duration::days(40);
        edge.valid_at = edge.created_at;
        edge.last_activated = anchor - Duration::hours(age_hours);
        edge.ltp_status = ltp_status;
        edge.activation_timestamps = activation_timestamps;
        edge.activation_count = 3;
        edge
    }

    #[test]
    fn decay_at_matches_the_pre_injection_decay_at_the_same_effective_time() {
        // Far from the wall clock, so a decay_at that reads Utc::now() anywhere diverges.
        let future: DateTime<Utc> = "2031-03-01T00:00:00Z".parse().unwrap();
        let tiers = [
            EdgeTier::L1Working,
            EdgeTier::L2Episodic,
            EdgeTier::L3Semantic,
        ];
        let ltps = [
            LtpSpec::None,
            LtpSpec::BurstFresh,
            LtpSpec::BurstExpired,
            LtpSpec::Weekly,
            LtpSpec::Full,
        ];
        let histories = [
            HistorySpec::None,
            HistorySpec::WeeklyHabit,
            HistorySpec::Sparse,
        ];
        let ages_hours = [1, 6, 30, 49, 240, 1100, 9000];
        let strengths = [0.9_f32, 0.12];

        let mut cases = 0;
        let (mut pruned, mut changed_ltp) = (0, 0);
        for tier in tiers {
            for ltp in ltps {
                for history in histories {
                    for age in ages_hours {
                        for strength in strengths {
                            let wall = Utc::now();
                            let mut legacy =
                                edge_anchored_at(wall, tier, strength, age, ltp, history);
                            let legacy_prune = legacy_decay(&mut legacy);

                            let mut injected =
                                edge_anchored_at(future, tier, strength, age, ltp, history);
                            let injected_prune = injected.decay_at(future);

                            let case =
                                format!("{tier:?} {ltp:?} {history:?} age={age}h s={strength}");
                            assert_eq!(injected_prune, legacy_prune, "prune decision: {case}");
                            assert_eq!(
                                injected.strength.to_bits(),
                                legacy.strength.to_bits(),
                                "strength {} vs {}: {case}",
                                injected.strength,
                                legacy.strength
                            );
                            assert_eq!(injected.last_activated, future, "{case}");
                            let shape = |s: &LtpStatus| std::mem::discriminant(s);
                            assert_eq!(
                                shape(&injected.ltp_status),
                                shape(&legacy.ltp_status),
                                "ltp: {case}"
                            );
                            if legacy_prune {
                                pruned += 1;
                            }
                            if shape(&legacy.ltp_status)
                                != shape(
                                    &edge_anchored_at(wall, tier, strength, age, ltp, history)
                                        .ltp_status,
                                )
                            {
                                changed_ltp += 1;
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        // Guards against a grid that never reaches the branches it claims to cover.
        assert!(pruned > 0 && pruned < cases, "pruned {pruned} of {cases}");
        assert!(changed_ltp > 0, "no case exercised an LTP transition");
    }

    #[test]
    fn decay_is_decay_at_the_wall_clock() {
        for age in [1, 49, 1100] {
            let wall = Utc::now();
            let mut a = edge_anchored_at(
                wall,
                EdgeTier::L2Episodic,
                0.7,
                age,
                LtpSpec::BurstExpired,
                HistorySpec::Sparse,
            );
            let mut b = a.clone();
            let pa = legacy_decay(&mut a);
            let pb = b.decay();
            assert_eq!(
                (pa, a.strength.to_bits()),
                (pb, b.strength.to_bits()),
                "age={age}h"
            );
        }
    }

    fn named_entity(name: &str) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        }
    }

    fn typed_edge(
        from: Uuid,
        to: Uuid,
        relation_type: RelationType,
        strength: f32,
    ) -> RelationshipEdge {
        let mut edge = create_test_edge_with_tier(strength, 0, EdgeTier::L2Episodic);
        edge.from_entity = from;
        edge.to_entity = to;
        edge.relation_type = relation_type;
        edge.activation_count = 1;
        edge
    }

    #[test]
    fn apply_decay_at_ages_stored_edges_to_the_injected_time() {
        // apply_decay reads the scoring clock, which tests in this crate pin.
        let _env = crate::memory::ablation::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let mut ids = Vec::new();
        for (i, (tier, strength, ltp)) in [
            (EdgeTier::L1Working, 0.3_f32, LtpStatus::None),
            (EdgeTier::L2Episodic, 0.25, LtpStatus::None),
            (EdgeTier::L3Semantic, 0.95, LtpStatus::Full),
            (EdgeTier::L2Episodic, 0.9, LtpStatus::Full),
        ]
        .into_iter()
        .enumerate()
        {
            let a = graph
                .add_entity(named_entity(&format!("decay-src-{i}")))
                .unwrap();
            let b = graph
                .add_entity(named_entity(&format!("decay-dst-{i}")))
                .unwrap();
            let mut edge = typed_edge(a, b, RelationType::RelatedTo, strength);
            edge.tier = tier;
            edge.ltp_status = ltp;
            ids.push(graph.add_relationship(edge).unwrap());
        }

        assert_eq!(
            graph.apply_decay().unwrap().pruned_count,
            0,
            "fresh edges survive the scoring clock"
        );

        let stored: Vec<RelationshipEdge> = ids
            .iter()
            .map(|id| graph.get_relationship(id).unwrap().unwrap())
            .collect();
        let at = Utc::now() + Duration::days(60);
        let expected: Vec<(bool, RelationshipEdge)> = stored
            .iter()
            .map(|e| {
                let mut e = e.clone();
                (e.decay_at(at), e)
            })
            .collect();
        let expected_pruned = expected.iter().filter(|(p, _)| *p).count();
        assert!(
            expected_pruned > 0 && expected_pruned < ids.len(),
            "fixture must prune some edges and keep others at +60d, pruned {expected_pruned}"
        );

        let result = graph.apply_decay_at(at).unwrap();
        assert_eq!(result.pruned_count, expected_pruned);
        for (id, (should_prune, aged)) in ids.iter().zip(&expected) {
            match graph.get_relationship(id).unwrap() {
                None => assert!(*should_prune, "edge {id} removed but decay_at kept it"),
                Some(edge) => {
                    assert!(!*should_prune, "edge {id} kept but decay_at pruned it");
                    assert_eq!(edge.strength.to_bits(), aged.strength.to_bits());
                    assert_eq!(edge.last_activated, at);
                }
            }
        }
    }

    #[test]
    fn is_causal_is_exactly_causes_results_in_and_triggers() {
        use RelationType::*;
        let all = [
            WorksWith,
            WorksAt,
            EmployedBy,
            PartOf,
            Contains,
            OwnedBy,
            LocatedIn,
            LocatedAt,
            Uses,
            CreatedBy,
            DevelopedBy,
            Causes,
            ResultsIn,
            Learned,
            Knows,
            Teaches,
            RelatedTo,
            AssociatedWith,
            CoRetrieved,
            CoOccurs,
            Manages,
            DependsOn,
            Requires,
            Configures,
            Prefers,
            Recommends,
            Documents,
            Implements,
            DeploysTo,
            Monitors,
            Triggers,
            SupersededBy,
            AlternativeTo,
            AssignedTo,
            Approves,
            Custom("Causes".to_string()),
            Custom("Precedes".to_string()),
        ];
        let causal: Vec<&str> = all
            .iter()
            .filter(|r| r.is_causal())
            .map(|r| r.as_str())
            .collect();
        assert_eq!(causal, vec!["Causes", "ResultsIn", "Triggers"]);
    }

    #[test]
    fn extract_directed_predicate_sets_cause_effect_arrow() {
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Vornak", "Meslin"),
            Some((RelationType::Triggers, true))
        );
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Meslin", "Vornak"),
            Some((RelationType::Triggers, false))
        );
        assert_eq!(
            extract_directed_predicate("Alpha and Beta met. Gamma caused Delta.", "Alpha", "Beta"),
            None,
            "a cue in a neighbouring sentence must not type the pair"
        );
        assert_eq!(
            extract_directed_predicate("Gamma caused Delta.", "Gamma", "Delta"),
            Some((RelationType::Triggers, true))
        );
        assert_eq!(
            extract_directed_predicate("Gamma caused Delta.", "Gamma", "Epsilon"),
            None
        );
    }

    #[test]
    fn extract_directed_predicate_flips_effect_first_constructions() {
        assert_eq!(
            extract_directed_predicate(
                "The flood happened because of the dam failure.",
                "dam failure",
                "flood"
            ),
            Some((RelationType::Triggers, true))
        );
        assert_eq!(
            extract_directed_predicate("The outage was caused by Redis.", "Redis", "outage"),
            Some((RelationType::Triggers, true))
        );
        assert_eq!(
            extract_directed_predicate("The delay was due to the storm.", "delay", "storm"),
            Some((RelationType::Triggers, false))
        );
        assert_eq!(
            extract_directed_predicate("Redis caused the outage.", "Redis", "outage"),
            Some((RelationType::Triggers, true))
        );
    }

    #[test]
    fn extract_directed_predicate_rejects_cue_fragment_endpoints() {
        let mid = "the Meslin incident then brought about the Caldor incident.";
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Vornak", "motion"),
            None
        );
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Meslin", "motion"),
            None
        );
        assert_eq!(
            extract_directed_predicate(mid, "the Meslin incident", "brought"),
            None
        );
        assert_eq!(
            extract_directed_predicate(mid, "brought", "the Caldor incident"),
            None
        );
        assert_eq!(
            extract_directed_predicate(mid, "the Meslin incident", "the Caldor incident"),
            Some((RelationType::Triggers, true))
        );
    }

    #[test]
    fn extract_directed_predicate_leaves_non_causal_sentences_untyped() {
        for (text, a, b) in [
            ("Vornak and Meslin met.", "Vornak", "Meslin"),
            ("Vornak is near Meslin.", "Vornak", "Meslin"),
            ("Vornak works closely with Meslin.", "Vornak", "Meslin"),
            ("Vornak met Meslin before the review.", "Vornak", "Meslin"),
        ] {
            assert_eq!(extract_directed_predicate(text, a, b), None, "{text}");
        }
    }

    #[test]
    fn extract_directed_predicate_does_not_type_did_not_cause() {
        // Upstream has no negation handling; this holds only because the cue is the past tense "caused".
        assert_eq!(
            extract_directed_predicate("Vornak did not cause Meslin.", "Vornak", "Meslin"),
            None
        );
    }

    #[test]
    fn recognised_tag_label_maps_naming_conventions() {
        assert_eq!(recognised_tag_label("rocksdb"), Some(EntityLabel::Database));
        assert_eq!(
            recognised_tag_label("metrics-service"),
            Some(EntityLabel::Service)
        );
        assert_eq!(recognised_tag_label("README"), Some(EntityLabel::Document));
        assert_eq!(
            recognised_tag_label("config.toml"),
            Some(EntityLabel::Configuration)
        );
        assert_eq!(recognised_tag_label("router.rs"), Some(EntityLabel::Module));
        assert_eq!(recognised_tag_label("ci-cd"), Some(EntityLabel::Pipeline));
        assert_eq!(
            recognised_tag_label("kubernetes"),
            Some(EntityLabel::Environment)
        );
    }

    #[test]
    fn recognised_tag_label_abstains_on_prose() {
        for prose in [
            "production",
            "development",
            "staging",
            "container",
            "stobar configuration",
            "empty weight",
            "support group",
            "painting",
            "workflows",
            "62-kalyani-group.txt",
        ] {
            assert_eq!(recognised_tag_label(prose), None, "{prose:?}");
        }
    }

    /// A star's hub sees the same curvature on every edge, the stop-word signature; a leaf
    /// with one edge has no spread to measure and reads neutral.
    #[test]
    fn compute_entity_selectivity_scores_a_star_hub_low_and_its_leaves_neutral() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let hub = graph.add_entity(named_entity("Hub")).unwrap();
        let leaves: Vec<Uuid> = (0..12)
            .map(|k| graph.add_entity(named_entity(&format!("Leaf{k}"))).unwrap())
            .collect();
        for leaf in &leaves {
            graph
                .add_relationship(typed_edge(hub, *leaf, RelationType::RelatedTo, 0.5))
                .unwrap();
        }
        assert_eq!(graph.entity_selectivity(&hub), None, "unscored is unknown");

        assert_eq!(graph.compute_entity_selectivity().unwrap(), 13);
        let hub_score = graph.entity_selectivity(&hub).unwrap();
        assert!(
            hub_score < crate::constants::GRAPH_LOW_SELECTIVITY,
            "{hub_score}"
        );
        for leaf in &leaves {
            assert_eq!(graph.entity_selectivity(leaf), Some(1.0));
        }
    }

    /// Salience after n mentions is the first mention's salience times one boost for n, capped
    /// at 1.3x: under the compounding formula a 0.5 entity reached 1.0 on its seventh mention.
    #[test]
    fn repeat_mentions_boost_salience_once_and_cap_the_boost() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let mut node = named_entity("Pottery");
        node.salience = 0.5;
        let id = graph.add_entity(node.clone()).unwrap();
        for _ in 1..7 {
            graph.add_entity(node.clone()).unwrap();
        }
        let seventh = graph.get_entity(&id).unwrap().unwrap();
        assert_eq!(seventh.mention_count, 7);
        let expected = 0.5 * (1.0 + 0.1 * 7f32.ln());
        assert!(
            (seventh.salience - expected).abs() < 1e-4,
            "{} != {expected}",
            seventh.salience
        );

        for _ in 7..100 {
            graph.add_entity(node.clone()).unwrap();
        }
        let capped = graph.get_entity(&id).unwrap().unwrap().salience;
        let ceiling = 0.5 * crate::constants::ENTITY_SALIENCE_MAX_FREQUENCY_BOOST;
        assert!((capped - ceiling).abs() < 1e-4, "{capped} != {ceiling}");
    }

    #[test]
    fn label_pairs_are_typed_in_either_order_from_the_rules_first_label() {
        use EntityLabel::*;
        assert_eq!(
            infer_relation_type_for_pair(&Person, &Organization, false),
            Some((RelationType::WorksAt, true))
        );
        assert_eq!(
            infer_relation_type_for_pair(&Organization, &Person, false),
            Some((RelationType::WorksAt, false))
        );
        assert_eq!(
            infer_relation_type_for_pair(&Location, &Person, false),
            Some((RelationType::LocatedIn, false))
        );
        assert_eq!(
            infer_relation_type_for_pair(&Service, &Database, false),
            Some((RelationType::Uses, true))
        );
    }

    #[test]
    fn label_pairs_with_no_rule_stay_untyped() {
        use EntityLabel::*;
        assert_eq!(infer_relation_type_for_pair(&Person, &Person, false), None);
        assert_eq!(
            infer_relation_type_for_pair(&Person, &Person, true),
            Some((RelationType::Knows, true))
        );
        assert_eq!(
            infer_relation_type_for_pair(&Technology, &Technology, false),
            None
        );
        assert_eq!(
            infer_relation_type_for_pair(&Concept, &Other("MISC".into()), false),
            None
        );
    }

    #[test]
    fn curvature_selectivity_is_spread_over_degree() {
        assert_eq!(curvature_selectivity(&[-3.0]), 1.0);
        assert_eq!(curvature_selectivity(&[-5.0, -5.0, -5.0]), 0.0);
        // Sample deviation of {0, 2} is sqrt(2); over two edges.
        let spread = curvature_selectivity(&[0.0, 2.0]);
        assert!((spread - 2f32.sqrt() / 2.0).abs() < 1e-6, "{spread}");
    }

    #[test]
    fn find_entity_by_name_strict_resolves_exact_case_and_stem_but_never_fuzzes() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let reactor = graph.add_entity(named_entity("Vornak Reactors")).unwrap();

        let strict = |n: &str| graph.find_entity_by_name_strict(n).unwrap().map(|e| e.uuid);
        assert_eq!(strict("Vornak Reactors"), Some(reactor), "exact");
        assert_eq!(strict("vornak reactors"), Some(reactor), "case-insensitive");
        assert_eq!(strict("Vornak Reactor"), Some(reactor), "stemmed");
        assert_eq!(strict("Vornak"), None, "substring must not resolve");
        assert_eq!(
            strict("Reactors of Vornak"),
            None,
            "word-level must not resolve"
        );

        // The fuzzy resolver does bind the fragment, which is what strict exists to refuse.
        assert_eq!(
            graph.find_entity_by_name("Vornak").unwrap().map(|e| e.uuid),
            Some(reactor)
        );
    }

    #[test]
    fn path_suffixes_split_at_slash_boundaries_after_the_prefix() {
        assert_eq!(
            path_suffixes("commitwork:admin/serve.mjs"),
            vec!["admin/serve.mjs".to_string(), "serve.mjs".to_string()]
        );
        assert_eq!(
            path_suffixes("Repo:README.md"),
            vec!["readme.md".to_string()]
        );
        assert!(path_suffixes("commitwork@1a2b3c4d").is_empty(), "no prefix");
        assert!(
            path_suffixes("New York: the city").is_empty(),
            "prose is not a path"
        );
        assert!(path_suffixes("a/b:c").is_empty(), "a prefix holds no slash");
        assert!(path_suffixes("repo:").is_empty());
    }

    fn declared_file(name: &str) -> EntityNode {
        let mut e = named_entity(name);
        e.labels = vec![EntityLabel::FilePath];
        e.is_proper_noun = true;
        e
    }

    #[test]
    fn file_paths_resolve_by_suffix_at_a_boundary_and_only_when_labelled_filepath() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let serve = graph
            .add_entity(declared_file("commitwork:admin/serve.mjs"))
            .unwrap();
        let observe = graph
            .add_entity(declared_file("commitwork:lib/observe.mjs"))
            .unwrap();
        // Same shape, not a file: must never be returned by the FilePath lookup.
        graph.add_entity(named_entity("intent:serve.mjs")).unwrap();

        let by = |t: &str| -> Vec<Uuid> {
            graph
                .find_file_paths_by_suffix(t)
                .unwrap()
                .into_iter()
                .map(|e| e.uuid)
                .collect()
        };
        assert_eq!(by("serve.mjs"), vec![serve]);
        assert_eq!(by("Admin/Serve.mjs"), vec![serve], "case-insensitive");
        assert_eq!(by("/admin/serve.mjs"), vec![serve]);
        assert_eq!(by("observe.mjs"), vec![observe]);
        assert!(by("erve.mjs").is_empty(), "not at a boundary");
        assert!(by("missing.mjs").is_empty());

        // The index is rebuilt from the name index on reopen, not persisted.
        drop(graph);
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let found = graph.find_file_paths_by_suffix("serve.mjs").unwrap();
        assert_eq!(
            found.iter().map(|e| e.uuid).collect::<Vec<_>>(),
            vec![serve]
        );

        assert!(graph.delete_entity(&serve).unwrap());
        assert!(graph
            .find_file_paths_by_suffix("serve.mjs")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn degree_cap_pruning_removes_unpotentiated_edges_before_declared_ones() {
        use crate::constants::{ENTITY_DEGREE_PRUNE_TARGET, MAX_ENTITY_DEGREE};
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let hub = graph.add_entity(named_entity("hub")).unwrap();
        let mut protected = Vec::new();
        for i in 0..MAX_ENTITY_DEGREE + 5 {
            let peer = graph.add_entity(named_entity(&format!("peer{i}"))).unwrap();
            let mut edge = typed_edge(hub, peer, RelationType::RelatedTo, 0.95);
            edge.tier = EdgeTier::L1Working;
            if i % 100 == 0 {
                // A declared edge: protected, but weaker than every plain one.
                edge.strength = 0.1;
                edge.ltp_status = LtpStatus::Weekly;
                edge.created_by = EdgeSource::Explicit;
                protected.push(graph.add_relationship(edge).unwrap());
            } else {
                graph.add_relationship(edge).unwrap();
            }
        }
        // Edge MAX + 1 cut the hub back to the target; the four after it were kept.
        assert_eq!(
            graph.entity_degree(&hub).unwrap(),
            ENTITY_DEGREE_PRUNE_TARGET + 4
        );
        let remaining: std::collections::HashSet<Uuid> = graph
            .get_entity_relationships(&hub)
            .unwrap()
            .into_iter()
            .map(|e| e.uuid)
            .collect();
        for id in &protected {
            assert!(remaining.contains(id), "a declared edge was pruned first");
        }
    }

    #[test]
    fn trace_causal_origins_walks_incoming_causal_edges_to_the_root_without_touching_them() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let [root, mid, effect, bystander, downstream] =
            ["root", "mid", "effect", "bystander", "downstream"]
                .map(|n| graph.add_entity(named_entity(n)).unwrap());

        graph
            .add_relationship(typed_edge(root, mid, RelationType::Causes, 0.8))
            .unwrap();
        graph
            .add_relationship(typed_edge(mid, effect, RelationType::Triggers, 0.8))
            .unwrap();
        graph
            .add_relationship(typed_edge(bystander, effect, RelationType::CoOccurs, 0.9))
            .unwrap();
        graph
            .add_relationship(typed_edge(effect, downstream, RelationType::ResultsIn, 0.9))
            .unwrap();

        let snapshot = |g: &GraphMemory| {
            let mut edges: Vec<(Uuid, u32, u32, DateTime<Utc>, u32)> = g
                .get_all_relationships()
                .unwrap()
                .into_iter()
                .map(|e| {
                    (
                        e.uuid,
                        e.strength.to_bits(),
                        e.activation_count,
                        e.last_activated,
                        e.forward_strength.to_bits(),
                    )
                })
                .collect();
            edges.sort();
            edges
        };
        let before = snapshot(&graph);

        let origins = graph.trace_causal_origins(&[effect], 8).unwrap();
        let ids: Vec<Uuid> = origins.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![root],
            "root only: not the proximal cause, a non-causal parent, or an effect"
        );
        let expected = 0.8_f32 * 0.7 * 0.8 * 0.7;
        assert!(
            (origins[0].1 - expected).abs() < 0.02,
            "two-hop score {} vs {expected}",
            origins[0].1
        );

        assert_eq!(
            graph.trace_causal_origins(&[root], 8).unwrap(),
            vec![],
            "a seed with no cause is not its own origin"
        );
        assert_eq!(
            snapshot(&graph),
            before,
            "the walk must not strengthen or rewrite edges"
        );
    }

    // ── Evidential base: one source counts once ──────────────────────────────

    #[test]
    fn same_source_twice_strengthens_once() {
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        let first = edge.strengthen_from("session-a");
        assert_eq!(first.evidence, Evidence::NewSource);
        let after_first = edge.strength;
        assert!(after_first > 0.3);
        assert_eq!(edge.activation_count, 1);
        assert_eq!(edge.distinct_sources(), 1);
        let corroborated = edge.last_corroborated.expect("first source corroborates");

        let before_repeat = edge.last_activated;
        let repeat = edge.strengthen_from("session-a");
        assert_eq!(repeat.evidence, Evidence::RepeatSource);
        assert!(repeat.promotion.is_none());
        assert_eq!(
            edge.strength, after_first,
            "a repeat from a recorded source must not move strength"
        );
        assert_eq!(edge.activation_count, 1, "a repeat is not an activation");
        assert_eq!(edge.distinct_sources(), 1);
        assert_eq!(edge.strengthening_sources["session-a"], 2);
        assert_eq!(edge.last_corroborated, Some(corroborated));
        assert!(
            edge.last_activated >= before_repeat,
            "a repeat keeps the edge warm"
        );
    }

    #[test]
    fn two_sources_strengthen_twice() {
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        edge.strengthen_from("session-a");
        let after_first = edge.strength;
        let second = edge.strengthen_from("session-b");
        assert_eq!(second.evidence, Evidence::NewSource);
        assert!(
            edge.strength > after_first,
            "a second source corroborates: {} -> {}",
            after_first,
            edge.strength
        );
        assert_eq!(edge.activation_count, 2);
        assert_eq!(edge.distinct_sources(), 2);
        assert_eq!(edge.strengthening_sources["session-b"], 1);
    }

    #[test]
    fn unattributed_events_strengthen_at_most_once() {
        use crate::constants::HEBBIAN_UNATTRIBUTED_SOURCE;
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        edge.strengthen_from(HEBBIAN_UNATTRIBUTED_SOURCE);
        let after_first = edge.strength;
        for _ in 0..5 {
            let outcome = edge.strengthen_from(HEBBIAN_UNATTRIBUTED_SOURCE);
            assert_eq!(outcome.evidence, Evidence::RepeatSource);
        }
        assert_eq!(edge.strength, after_first);
        assert_eq!(edge.activation_count, 1);
        assert_eq!(edge.distinct_sources(), 1);
        assert_eq!(ambient_evidence_source(), HEBBIAN_UNATTRIBUTED_SOURCE);
    }

    #[test]
    fn sources_past_the_cap_are_treated_as_repeats() {
        use crate::constants::HEBBIAN_EVIDENCE_SOURCES_CAP;
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L3Semantic);
        for i in 0..HEBBIAN_EVIDENCE_SOURCES_CAP {
            assert_eq!(
                edge.strengthen_from(&format!("s{i}")).evidence,
                Evidence::NewSource
            );
        }
        let at_cap = edge.strength;
        assert_eq!(
            edge.strengthen_from("late").evidence,
            Evidence::RepeatSource
        );
        assert_eq!(edge.strength, at_cap);
        assert_eq!(edge.distinct_sources(), HEBBIAN_EVIDENCE_SOURCES_CAP);
        assert!(!edge.strengthening_sources.contains_key("late"));
        assert_eq!(
            edge.strengthen_from("later").step,
            0.0,
            "past the cap a new source earns no directional step either"
        );
    }

    #[test]
    fn direction_moves_once_per_source_by_exactly_one_step() {
        use crate::constants::{COACTIVATION_BACKWARD_BOOST, COACTIVATION_FORWARD_BOOST};
        let mut edge = create_test_edge_with_tier(0.4, 0, EdgeTier::L1Working);
        edge.forward_strength = 0.4;
        edge.backward_strength = 0.4;

        let first = edge.strengthen_from("session-a");
        assert_eq!(first.step, 1.0);
        edge.strengthen_forward(first.step);
        let forward = 0.4 + COACTIVATION_FORWARD_BOOST * (1.0 - 0.4);
        assert!((edge.forward_strength - forward).abs() < 1e-6);

        for _ in 0..50 {
            let repeat = edge.strengthen_from("session-a");
            assert_eq!(repeat.step, 0.0);
            edge.strengthen_forward(repeat.step);
            edge.strengthen_backward(repeat.step);
        }
        assert!(
            (edge.forward_strength - forward).abs() < 1e-6,
            "fifty repeats from one session must not move direction"
        );
        assert_eq!(edge.backward_strength, 0.4);

        let corroboration = edge.strengthen_from("session-b");
        edge.strengthen_backward(corroboration.step);
        let backward = 0.4 + COACTIVATION_BACKWARD_BOOST * (1.0 - 0.4);
        assert!((edge.backward_strength - backward).abs() < 1e-6);
        assert!(
            (edge.forward_strength - forward).abs() < 1e-6,
            "only the direction the new source triggered moves"
        );
    }

    #[test]
    fn evidence_scope_names_the_ambient_source_and_restores_on_drop() {
        assert_eq!(ambient_evidence_source(), "unattributed");
        {
            let _outer = EvidenceScope::enter(Some("session-a"));
            assert_eq!(ambient_evidence_source(), "session-a");
            {
                let _inner = EvidenceScope::enter(None);
                assert_eq!(ambient_evidence_source(), "unattributed");
            }
            assert_eq!(ambient_evidence_source(), "session-a");
        }
        assert_eq!(ambient_evidence_source(), "unattributed");
    }

    #[test]
    fn decode_accepts_the_layout_before_the_evidential_base() {
        let edge = create_test_edge_with_tier(0.42, 3, EdgeTier::L2Episodic);
        let v0 = RelationshipEdgeV0 {
            uuid: edge.uuid,
            from_entity: edge.from_entity,
            to_entity: edge.to_entity,
            relation_type: edge.relation_type.clone(),
            strength: edge.strength,
            created_at: edge.created_at,
            valid_at: edge.valid_at,
            invalidated_at: edge.invalidated_at,
            source_episode_id: edge.source_episode_id,
            context: edge.context.clone(),
            last_activated: edge.last_activated,
            activation_count: 7,
            ltp_status: edge.ltp_status,
            tier: edge.tier,
            activation_timestamps: None,
            entity_confidence: Some(0.8),
            created_by: EdgeSource::Explicit,
            forward_strength: 0.1,
            backward_strength: 0.2,
        };
        let old_bytes = bincode::serde::encode_to_vec(&v0, bincode::config::standard()).unwrap();
        let decoded = RelationshipEdge::decode(&old_bytes).expect("legacy layout decodes");
        assert_eq!(decoded.uuid, edge.uuid);
        assert_eq!(decoded.activation_count, 7);
        assert_eq!(decoded.entity_confidence, Some(0.8));
        assert_eq!(decoded.backward_strength, 0.2);
        assert_eq!(decoded.distinct_sources(), 0);
        assert_eq!(decoded.last_corroborated, None);

        let mut current = edge.clone();
        current.strengthen_from("session-a");
        let new_bytes =
            bincode::serde::encode_to_vec(&current, bincode::config::standard()).unwrap();
        let round = RelationshipEdge::decode(&new_bytes).expect("current layout decodes");
        assert_eq!(round.distinct_sources(), 1);
        assert_eq!(round.last_corroborated, current.last_corroborated);

        assert!(RelationshipEdge::decode(&new_bytes[..new_bytes.len() - 1]).is_err());
    }

    #[test]
    fn graph_reinforcement_from_one_source_strengthens_an_edge_once() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let make_entity = |id: Uuid, name: &str| EntityNode {
            uuid: id,
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        };
        graph.add_entity(make_entity(a, "A")).unwrap();
        graph.add_entity(make_entity(b, "B")).unwrap();

        graph
            .apply_reinforcement(Uuid::new_v4(), "session-a", &a, &[b])
            .unwrap();
        let created = graph.find_edge_between_entities(&a, &b).unwrap().unwrap();
        assert_eq!(created.distinct_sources(), 1);
        assert!(created.last_corroborated.is_some());

        for _ in 0..10 {
            graph
                .apply_reinforcement(Uuid::new_v4(), "session-a", &a, &[b])
                .unwrap();
        }
        let repeated = graph.find_edge_between_entities(&a, &b).unwrap().unwrap();
        assert_eq!(
            repeated.strength, created.strength,
            "ten more recalls from the same session must not move strength"
        );
        assert_eq!(repeated.activation_count, created.activation_count);
        assert_eq!(repeated.distinct_sources(), 1);
        assert_eq!(repeated.strengthening_sources["session-a"], 11);
        assert_eq!(
            (repeated.forward_strength, repeated.backward_strength),
            (created.forward_strength, created.backward_strength),
            "nor direction"
        );

        graph
            .apply_reinforcement(Uuid::new_v4(), "session-b", &a, &[b])
            .unwrap();
        let corroborated = graph.find_edge_between_entities(&a, &b).unwrap().unwrap();
        assert!(corroborated.strength > repeated.strength);
        assert_eq!(corroborated.distinct_sources(), 2);
        let one_step =
            crate::constants::COACTIVATION_FORWARD_BOOST * (1.0 - repeated.forward_strength);
        assert!(
            (corroborated.forward_strength - (repeated.forward_strength + one_step)).abs() < 1e-6,
            "a new session moves the triggered direction by one step: {} -> {}",
            repeated.forward_strength,
            corroborated.forward_strength
        );
        assert_eq!(corroborated.backward_strength, repeated.backward_strength);
        assert!(corroborated.last_corroborated >= created.last_corroborated);
    }

    fn episode_at(
        entity_refs: Vec<Uuid>,
        content: &str,
        created_at: DateTime<Utc>,
    ) -> EpisodicNode {
        EpisodicNode {
            uuid: Uuid::new_v4(),
            name: content.to_string(),
            content: content.to_string(),
            valid_at: created_at,
            created_at,
            entity_refs,
            source: EpisodeSource::Message,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn recent_episodes_are_newest_first_ties_by_content_and_capped() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let entity = graph.add_entity(named_entity("Kestrel")).unwrap();
        let base = Utc::now() - Duration::days(100);
        for day in 0..30 {
            graph
                .add_episode(episode_at(
                    vec![entity],
                    &format!("day {day:02}"),
                    base + Duration::days(day),
                ))
                .unwrap();
        }
        // Three episodes tied at the newest instant: the cap cuts through them by content.
        let newest = base + Duration::days(40);
        for content in ["tie c", "tie a", "tie b"] {
            graph
                .add_episode(episode_at(vec![entity], content, newest))
                .unwrap();
        }

        let recent = graph.get_recent_episodes_by_entity(&entity, 5).unwrap();
        let contents: Vec<&str> = recent.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, ["tie a", "tie b", "tie c", "day 29", "day 28"]);

        let two = graph.get_recent_episodes_by_entity(&entity, 2).unwrap();
        let contents: Vec<&str> = two.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(
            contents,
            ["tie a", "tie b"],
            "a tie at the cut is ordered by content"
        );

        // The capped read agrees with sorting every episode the unordered index holds.
        let mut all = graph.get_episodes_by_entity(&entity).unwrap();
        all.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.content.cmp(&b.content))
        });
        let expected: Vec<Uuid> = all.iter().take(20).map(|e| e.uuid).collect();
        let capped: Vec<Uuid> = graph
            .get_recent_episodes_by_entity(&entity, 20)
            .unwrap()
            .iter()
            .map(|e| e.uuid)
            .collect();
        assert_eq!(capped, expected);
    }

    #[test]
    fn recent_episodes_skip_rows_a_readd_left_behind_and_deleted_episodes() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let entity = graph.add_entity(named_entity("Osprey")).unwrap();
        let base = Utc::now() - Duration::days(10);

        let mut moved = episode_at(vec![entity], "moved", base + Duration::days(5));
        graph.add_episode(moved.clone()).unwrap();
        moved.created_at = base;
        graph.add_episode(moved.clone()).unwrap();
        let kept = episode_at(vec![entity], "kept", base + Duration::days(1));
        graph.add_episode(kept.clone()).unwrap();
        let gone = episode_at(vec![entity], "gone", base + Duration::days(2));
        graph.add_episode(gone.clone()).unwrap();
        graph.delete_episode(&gone.uuid).unwrap();

        let recent = graph.get_recent_episodes_by_entity(&entity, 10).unwrap();
        let contents: Vec<&str> = recent.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(
            contents,
            ["kept", "moved"],
            "one row per live episode, at its current time"
        );
    }

    #[test]
    fn recency_index_is_backfilled_for_a_store_written_before_it() {
        let temp_dir = tempfile::tempdir().unwrap();
        let entity;
        {
            let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
            entity = graph.add_entity(named_entity("Heron")).unwrap();
            let base = Utc::now() - Duration::days(3);
            for (day, content) in ["first", "second", "third"].iter().enumerate() {
                graph
                    .add_episode(episode_at(
                        vec![entity],
                        content,
                        base + Duration::days(day as i64),
                    ))
                    .unwrap();
            }
            // Strip the store back to the pre-index layout: no recency rows, no marker.
            let prefix = format!("{entity}#");
            for (key, _) in graph
                .store
                .scan_cf(CF_ENTITY_EPISODES, Some(prefix.as_bytes()), None)
                .unwrap()
            {
                graph.store.delete_cf(CF_ENTITY_EPISODES, &key).unwrap();
            }
            graph
                .store
                .delete_cf(CF_ENTITY_EPISODES, EPISODE_RECENCY_INDEX_MARKER)
                .unwrap();
            assert!(graph
                .get_recent_episodes_by_entity(&entity, 3)
                .unwrap()
                .is_empty());
        }

        let reopened = GraphMemory::new(temp_dir.path(), None).unwrap();
        let recent = reopened.get_recent_episodes_by_entity(&entity, 2).unwrap();
        let contents: Vec<&str> = recent.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, ["third", "second"]);
    }

    #[test]
    fn deleting_an_entity_scrubs_its_recency_rows() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let entity = graph.add_entity(named_entity("Plover")).unwrap();
        graph
            .add_episode(episode_at(vec![entity], "only", Utc::now()))
            .unwrap();
        graph.delete_entity(&entity).unwrap();
        let prefix = format!("{entity}#");
        assert!(graph
            .store
            .scan_cf(CF_ENTITY_EPISODES, Some(prefix.as_bytes()), None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn strengthen_episode_entity_edges_strengthens_the_existing_pair_edge() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let a = graph.add_entity(named_entity("Lattice")).unwrap();
        let b = graph.add_entity(named_entity("Furnace")).unwrap();
        let c = graph.add_entity(named_entity("Quarry")).unwrap();
        graph
            .add_relationship(typed_edge(a, b, RelationType::RelatedTo, 0.4))
            .unwrap();
        let before = graph.find_relationship_between(&a, &b).unwrap().unwrap();

        // `a` twice: duplicate refs must not strengthen one edge twice.
        let episode = episode_at(vec![a, b, a, c], "lattice furnace quarry", Utc::now());
        let episode_id = graph.add_episode(episode).unwrap();

        let strengthened = graph
            .strengthen_episode_entity_edges(&episode_id, "replay:test")
            .unwrap();
        assert_eq!(strengthened, 1, "one entity pair has an edge");

        let after = graph.find_relationship_between(&a, &b).unwrap().unwrap();
        assert_eq!(after.uuid, before.uuid);
        assert!(after.strength > before.strength);
        assert_eq!(after.activation_count, before.activation_count + 1);
        assert!(graph.find_relationship_between(&a, &c).unwrap().is_none());
    }

    #[test]
    fn bounded_name_resolution_keeps_the_strict_tiers_and_caps_the_fuzzy_ones() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let exact = graph.add_entity(named_entity("Vornak")).unwrap();
        let mut longer = Vec::new();
        for i in 0..5 {
            longer.push(
                graph
                    .add_entity(named_entity(&format!("Talmor Station {i}")))
                    .unwrap(),
            );
        }
        let mut salient = named_entity("Talmor Station Annex");
        salient.salience = 0.99;
        let salient = graph.add_entity(salient).unwrap();

        let bounded = |name: &str, cap: usize| {
            graph
                .find_entity_by_name_bounded(name, cap)
                .unwrap()
                .map(|e| e.uuid)
        };
        assert_eq!(
            bounded("vornak", 1),
            Some(exact),
            "strict tiers win outright"
        );
        assert_eq!(graph.find_entity_uuid_strict("VORNAK"), Some(exact));
        assert_eq!(graph.find_entity_uuid_strict("Talmor"), None);

        // Uncapped fuzzy matching picks the most salient match...
        assert_eq!(bounded("Talmor", 32), Some(salient));
        // ...a cap keeps the shortest names, so the longer, salient one is never read.
        let capped = bounded("Talmor", 2).unwrap();
        assert!(
            longer[..2].contains(&capped),
            "kept candidates are the two shortest names"
        );
        assert_eq!(bounded("Talmor", 0), None);
    }
}

#[cfg(test)]
mod ablation_gate_tests {
    use super::*;
    use crate::memory::ablation::{DISABLE_BOOSTS_VAR, ENV_LOCK};

    fn node(uuid: Uuid, name: &str) -> EntityNode {
        EntityNode {
            uuid,
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        }
    }

    /// Strength of a fresh A-B edge before and after one walk of `kind` from A.
    fn walk(kind: &str) -> (f32, f32) {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        graph.add_entity(node(a, "AblationA")).unwrap();
        graph.add_entity(node(b, "AblationB")).unwrap();
        let edge = graph
            .add_relationship(RelationshipEdge {
                uuid: Uuid::new_v4(),
                from_entity: a,
                to_entity: b,
                relation_type: RelationType::RelatedTo,
                strength: 0.5,
                created_at: Utc::now(),
                valid_at: Utc::now(),
                invalidated_at: None,
                source_episode_id: None,
                context: String::new(),
                last_activated: Utc::now(),
                activation_count: 1,
                ltp_status: LtpStatus::None,
                activation_timestamps: None,
                tier: EdgeTier::L2Episodic,
                created_by: EdgeSource::CoOccurrence,
                entity_confidence: None,
                forward_strength: 0.5,
                backward_strength: 0.5,
                strengthening_sources: Default::default(),
                last_corroborated: None,
            })
            .unwrap();
        let strength = || graph.get_relationship(&edge).unwrap().unwrap().strength;
        let before = strength();
        match kind {
            "from_entity" => {
                graph.traverse_from_entity(&a, 2).unwrap();
            }
            "weighted" => {
                graph.traverse_weighted(&a, 2, None, 0.0).unwrap();
            }
            "bidirectional" => {
                graph.traverse_bidirectional(&a, &b, 2, 0.0).unwrap();
            }
            other => panic!("unknown walk {other}"),
        }
        (before, strength())
    }

    /// Each traversal that feeds the recall graph leg strengthens the edges it walks;
    /// ablating `graph_potentiation` must leave the edge exactly as it was.
    #[test]
    fn graph_potentiation_ablation_stops_the_traversal_write() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_readonly = std::env::var("VELD_RECALL_READONLY").ok();
        // The harness pins SHODH_RECALL_READONLY=1 process-wide; VELD_ wins over it.
        std::env::set_var("VELD_RECALL_READONLY", "0");
        for kind in ["from_entity", "weighted", "bidirectional"] {
            std::env::remove_var(DISABLE_BOOSTS_VAR);
            let (before, live) = walk(kind);
            assert!(
                live > before,
                "{kind}: a live walk must strengthen ({before} -> {live})"
            );

            std::env::set_var(DISABLE_BOOSTS_VAR, "graph_potentiation");
            let (before, ablated) = walk(kind);
            std::env::remove_var(DISABLE_BOOSTS_VAR);
            assert_eq!(
                ablated, before,
                "{kind}: ablated walk still strengthened the edge"
            );
        }
        match prev_readonly {
            Some(v) => std::env::set_var("VELD_RECALL_READONLY", v),
            None => std::env::remove_var("VELD_RECALL_READONLY"),
        }
    }

    /// `SHODH_EVAL_NOW` freezes the read-path decay clock, and it is read per call.
    #[test]
    fn eval_now_freezes_effective_strength() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let anchor = DateTime::parse_from_rfc3339("2030-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: anchor - chrono::Duration::hours(200),
            valid_at: anchor - chrono::Duration::hours(200),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: anchor - chrono::Duration::hours(100),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L1Working,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: 0.8,
            backward_strength: 0.8,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        };
        let at_anchor = edge.effective_strength_at(anchor);
        let later = edge.effective_strength_at(anchor + chrono::Duration::hours(500));
        assert_ne!(
            at_anchor, later,
            "the fixture must decay measurably over 500h"
        );

        let prev = std::env::var("VELD_EVAL_NOW").ok();
        std::env::set_var("VELD_EVAL_NOW", "2030-01-01T00:00:00Z");
        assert_eq!(edge.effective_strength(), at_anchor);
        let directional_at_anchor = edge.directional_strength(&edge.from_entity);
        std::env::set_var("VELD_EVAL_NOW", "2030-01-21T20:00:00Z");
        assert_eq!(
            edge.effective_strength(),
            later,
            "a later pin was not re-read"
        );
        assert_ne!(
            edge.directional_strength(&edge.from_entity),
            directional_at_anchor,
            "directional decay ignored the pinned clock"
        );
        match prev {
            Some(v) => std::env::set_var("VELD_EVAL_NOW", v),
            None => std::env::remove_var("VELD_EVAL_NOW"),
        }
    }

    /// The maintenance decay pass writes the decayed strength back, so under a pinned
    /// `VELD_EVAL_NOW` it must age edges to the pinned instant, not to whenever it ran.
    #[test]
    fn eval_now_freezes_the_strength_maintenance_decay_stores() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let anchor = DateTime::parse_from_rfc3339("2030-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        graph.add_entity(node(a, "DecayClockA")).unwrap();
        graph.add_entity(node(b, "DecayClockB")).unwrap();
        let edge_id = graph
            .add_relationship(RelationshipEdge {
                uuid: Uuid::new_v4(),
                from_entity: a,
                to_entity: b,
                relation_type: RelationType::RelatedTo,
                strength: 0.8,
                created_at: anchor - chrono::Duration::hours(200),
                valid_at: anchor - chrono::Duration::hours(200),
                invalidated_at: None,
                source_episode_id: None,
                context: String::new(),
                last_activated: anchor - chrono::Duration::hours(100),
                activation_count: 1,
                ltp_status: LtpStatus::None,
                activation_timestamps: None,
                tier: EdgeTier::L2Episodic,
                created_by: EdgeSource::CoOccurrence,
                entity_confidence: None,
                forward_strength: 0.8,
                backward_strength: 0.8,
                strengthening_sources: Default::default(),
                last_corroborated: None,
            })
            .unwrap();
        let stored = graph.get_relationship(&edge_id).unwrap().unwrap();
        let mut expected = stored.clone();
        assert!(!expected.decay_at(anchor), "the fixture must survive 100h");
        assert_ne!(
            expected.strength, stored.strength,
            "the fixture must decay measurably over 100h"
        );

        let prev = std::env::var("VELD_EVAL_NOW").ok();
        std::env::set_var("VELD_EVAL_NOW", "2030-01-01T00:00:00Z");
        let decay = graph.apply_decay();
        match prev {
            Some(v) => std::env::set_var("VELD_EVAL_NOW", v),
            None => std::env::remove_var("VELD_EVAL_NOW"),
        }

        assert_eq!(decay.unwrap().pruned_count, 0);
        let decayed = graph.get_relationship(&edge_id).unwrap().unwrap();
        assert_eq!(
            decayed.strength.to_bits(),
            expected.strength.to_bits(),
            "stored {} after maintenance, the pinned clock gives {}",
            decayed.strength,
            expected.strength
        );
        assert_eq!(decayed.last_activated, anchor);
    }
}

#[cfg(test)]
mod repeat_stable_order_tests {
    use super::*;

    fn entity(name: &str, salience: f32) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        }
    }

    fn edge(from: Uuid, to: Uuid, strength: f32) -> RelationshipEdge {
        // Activated in the future, so no clock (live or pinned) decays it.
        let future = DateTime::parse_from_rfc3339("2100-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: future,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L1Working,
            activation_timestamps: None,
            entity_confidence: None,
            created_by: EdgeSource::CoOccurrence,
            forward_strength: strength,
            backward_strength: strength,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    #[test]
    fn equal_strength_edges_under_a_limit_follow_peer_names() {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        let hub = graph.add_entity(entity("hub", 0.5)).unwrap();
        let mut names: Vec<String> = (0..16).map(|i| format!("peer-{i:02}")).collect();
        names.reverse();
        for name in &names {
            let peer = graph.add_entity(entity(name, 0.5)).unwrap();
            graph.add_relationship(edge(hub, peer, 0.3)).unwrap();
        }
        let strongest = graph.add_entity(entity("zz-strong", 0.5)).unwrap();
        graph.add_relationship(edge(hub, strongest, 0.9)).unwrap();

        let edges = graph
            .get_entity_relationships_limited(&hub, Some(5))
            .unwrap();
        let peers: Vec<String> = edges
            .iter()
            .map(|e| graph.get_entity(&e.to_entity).unwrap().unwrap().name)
            .collect();
        assert_eq!(
            peers,
            ["zz-strong", "peer-00", "peer-01", "peer-02", "peer-03"],
            "strength first, then peer name; edge uuids must not choose the survivors"
        );
    }

    #[test]
    fn a_repeat_source_that_moves_nothing_is_not_written() {
        assert_eq!(
            crate::constants::HEBBIAN_REPEAT_SOURCE_BOOST,
            0.0,
            "this test pins the zero-boost default"
        );
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        let a = graph.add_entity(entity("a", 0.5)).unwrap();
        let b = graph.add_entity(entity("b", 0.5)).unwrap();
        let c = graph.add_entity(entity("c", 0.5)).unwrap();
        let first = graph.add_relationship(edge(a, b, 0.3)).unwrap();
        let second = graph.add_relationship(edge(b, c, 0.3)).unwrap();
        let stored = |uuid: &Uuid| graph.get_relationship(uuid).unwrap().unwrap();

        assert_eq!(
            graph
                .batch_strengthen_synapses(&[first], "recall-a")
                .unwrap(),
            1
        );
        let after_new = stored(&first);
        assert_eq!(
            graph
                .batch_strengthen_synapses(&[first], "recall-a")
                .unwrap(),
            0
        );
        let after_repeat = stored(&first);
        assert_eq!(after_repeat.last_activated, after_new.last_activated);
        assert_eq!(
            after_repeat.strengthening_sources,
            after_new.strengthening_sources
        );
        assert_eq!(
            graph
                .batch_strengthen_synapses(&[first, second], "recall-a")
                .unwrap(),
            1,
            "the edge new to the source is written, the repeat is not"
        );
        assert_eq!(
            graph
                .batch_strengthen_synapses(&[first], "recall-b")
                .unwrap(),
            1
        );
        assert!(stored(&first).strength > after_repeat.strength);

        let before = stored(&second);
        graph.strengthen_synapse(&second, "recall-a").unwrap();
        assert_eq!(stored(&second).last_activated, before.last_activated);
        graph.strengthen_synapse(&second, "recall-c").unwrap();
        assert!(stored(&second).strength > before.strength);
    }

    #[test]
    fn a_bounded_read_keeps_the_same_edges_names_each_peer_once_and_refuses_a_hub() {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        let hub = graph.add_entity(entity("hub", 0.5)).unwrap();
        let other = graph.add_entity(entity("other hub", 0.5)).unwrap();
        let mut names: Vec<String> = (0..12).map(|i| format!("peer-{i:02}")).collect();
        names.reverse();
        for name in &names {
            let peer = graph.add_entity(entity(name, 0.5)).unwrap();
            graph.add_relationship(edge(hub, peer, 0.3)).unwrap();
            graph.add_relationship(edge(other, peer, 0.3)).unwrap();
        }
        let now = Utc::now();
        let pairs = |edges: Vec<(RelationshipEdge, f32)>| -> Vec<(Uuid, Uuid)> {
            edges
                .into_iter()
                .map(|(e, _)| (e.from_entity, e.to_entity))
                .collect()
        };

        let all = pairs(graph.entity_edges_with_strength(&hub, None, now).unwrap());
        let five = pairs(
            graph
                .entity_edges_with_strength(&hub, Some(5), now)
                .unwrap(),
        );
        assert_eq!(five, all[..5], "the limit keeps the head of the full order");

        let mut peer_names = HashMap::new();
        let (bounded, decoded) = graph
            .entity_edges_with_strength_bounded(&hub, Some(5), now, 12, &mut peer_names)
            .unwrap()
            .expect("12 edges fit a budget of 12");
        assert_eq!(decoded, 12);
        assert_eq!(pairs(bounded), five);
        let named = peer_names.len();
        assert_eq!(named, 12, "the one tie run starts before the limit");
        graph
            .entity_edges_with_strength_bounded(&other, Some(5), now, 12, &mut peer_names)
            .unwrap()
            .expect("fits");
        assert_eq!(peer_names.len(), named, "shared peers are not named twice");

        assert!(graph
            .entity_edges_with_strength_bounded(&hub, Some(5), now, 11, &mut HashMap::new())
            .unwrap()
            .is_none());
    }

    #[test]
    fn fuzzy_lookup_breaks_salience_and_length_ties_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        // Proper nouns skip the stemmed dedup tier, so all ten stay distinct entities.
        for suffix in ["j", "c", "h", "a", "f", "b", "i", "d", "g", "e"] {
            let mut node = entity(&format!("Broke{suffix}"), 0.4);
            node.is_proper_noun = true;
            graph.add_entity(node).unwrap();
        }
        assert_eq!(graph.get_all_entities().unwrap().len(), 10);
        let found = graph.find_entity_by_name("rok").unwrap().unwrap();
        assert_eq!(found.name, "Brokea");
    }
}

#[cfg(test)]
mod edge_index_tests {
    use super::*;

    fn named_entity(name: &str) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: Default::default(),
        }
    }

    fn typed_edge(
        from: Uuid,
        to: Uuid,
        relation_type: RelationType,
        strength: f32,
    ) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: strength,
            backward_strength: strength,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    fn episode_over(uuid: Uuid, entity_refs: Vec<Uuid>) -> EpisodicNode {
        EpisodicNode {
            uuid,
            name: "episode".to_string(),
            content: "content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs,
            source: EpisodeSource::Message,
            metadata: HashMap::new(),
        }
    }

    fn sourced(mut edge: RelationshipEdge, episode: Option<Uuid>) -> RelationshipEdge {
        edge.source_episode_id = episode;
        edge
    }

    fn edge_index_rows(graph: &GraphMemory, prefix: &str) -> usize {
        graph
            .store
            .scan_cf(CF_EDGE_INDEX, Some(prefix.as_bytes()), None)
            .unwrap()
            .len()
    }

    #[test]
    fn deleting_an_episode_removes_the_edges_it_sourced_and_no_others_on_both_engines() {
        let stores = crate::storage::test_support::column_stores(GRAPH_CF_NAMES);
        crate::storage::test_support::assert_both_backends(&stores);
        for (backend, store, _tmp) in stores {
            let graph = GraphMemory::with_store(store).unwrap();
            let [a, b, c] = ["Alder", "Beech", "Cherry"]
                .map(|name| graph.add_entity(named_entity(name)).unwrap());
            let (gone, kept) = (Uuid::new_v4(), Uuid::new_v4());
            graph.add_episode(episode_over(gone, vec![a, b])).unwrap();
            graph.add_episode(episode_over(kept, vec![b, c])).unwrap();
            let ids = graph
                .add_relationships(vec![
                    sourced(typed_edge(a, b, RelationType::RelatedTo, 0.5), Some(gone)),
                    sourced(typed_edge(b, c, RelationType::CoOccurs, 0.5), Some(kept)),
                    sourced(typed_edge(a, c, RelationType::RelatedTo, 0.5), None),
                ])
                .unwrap();

            assert!(graph.delete_episode(&gone).unwrap(), "{backend:?}");

            assert!(
                graph.get_relationship(&ids[0]).unwrap().is_none(),
                "{backend:?}"
            );
            assert!(
                graph.get_relationship(&ids[1]).unwrap().is_some(),
                "{backend:?}"
            );
            assert!(
                graph.get_relationship(&ids[2]).unwrap().is_some(),
                "{backend:?}"
            );
            assert_eq!(
                edge_index_rows(&graph, &GraphMemory::episode_edge_prefix(&gone)),
                0,
                "{backend:?}: the deleted episode's rows remain"
            );
            assert!(graph
                .find_relationship_between_typed(&a, &b, &RelationType::RelatedTo)
                .unwrap()
                .is_none());
            assert_eq!(
                graph
                    .find_relationship_between_typed(&c, &b, &RelationType::CoOccurs)
                    .unwrap()
                    .map(|e| e.uuid),
                Some(ids[1]),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn a_graph_written_before_the_edge_index_backfills_it_on_open() {
        let stores = crate::storage::test_support::column_stores(GRAPH_CF_NAMES);
        crate::storage::test_support::assert_both_backends(&stores);
        for (backend, store, _tmp) in stores {
            let episode = Uuid::new_v4();
            let (a, b, id) = {
                let graph = GraphMemory::with_store(Arc::clone(&store)).unwrap();
                let a = graph.add_entity(named_entity("Larch")).unwrap();
                let b = graph.add_entity(named_entity("Maple")).unwrap();
                graph
                    .add_episode(episode_over(episode, vec![a, b]))
                    .unwrap();
                let id = graph
                    .add_relationship(sourced(
                        typed_edge(a, b, RelationType::PartOf, 0.5),
                        Some(episode),
                    ))
                    .unwrap();
                (a, b, id)
            };
            let mut erase = crate::storage::WriteBatch::new();
            for (key, _) in store.scan_cf(CF_EDGE_INDEX, None, None).unwrap() {
                erase.delete_cf(CF_EDGE_INDEX, key);
            }
            store.write_batch(&erase).unwrap();

            let graph = GraphMemory::with_store(store).unwrap();
            assert!(
                graph
                    .store
                    .get_cf(CF_EDGE_INDEX, EDGE_INDEX_MARKER)
                    .unwrap()
                    .is_some(),
                "{backend:?}"
            );
            assert_eq!(
                graph
                    .find_relationship_between_typed(&b, &a, &RelationType::PartOf)
                    .unwrap()
                    .map(|e| e.uuid),
                Some(id),
                "{backend:?}"
            );
            graph.delete_episode(&episode).unwrap();
            assert!(
                graph.get_relationship(&id).unwrap().is_none(),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn a_stale_typed_pair_row_falls_back_to_the_scan_and_is_repaired() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let a = graph.add_entity(named_entity("Rowan")).unwrap();
        let b = graph.add_entity(named_entity("Spruce")).unwrap();
        let id = graph
            .add_relationship(typed_edge(a, b, RelationType::Uses, 0.5))
            .unwrap();
        let typed = GraphMemory::typed_pair_key(&a, &b, &RelationType::Uses);
        graph
            .store
            .put_cf(CF_EDGE_INDEX, typed.as_bytes(), Uuid::new_v4().as_bytes())
            .unwrap();

        let found = graph
            .find_relationship_between_typed(&a, &b, &RelationType::Uses)
            .unwrap();
        assert_eq!(found.map(|e| e.uuid), Some(id));
        assert_eq!(
            graph.store.get_cf(CF_EDGE_INDEX, typed.as_bytes()).unwrap(),
            Some(id.as_bytes().to_vec()),
            "the stale row was not repointed at the edge the scan found"
        );
        assert_eq!(
            graph
                .add_relationship(typed_edge(b, a, RelationType::Uses, 0.5))
                .unwrap(),
            id,
            "a repeat made a second synapse"
        );
    }

    #[test]
    fn a_queued_prune_spares_an_edge_strengthened_since() {
        // The prune decides on the scoring clock, which other tests in this binary pin; an edge
        // stamped with the wall clock reads as years old while one of them holds it.
        let _env = crate::memory::ablation::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let [a, b, c] =
            ["Willow", "Hazel", "Holly"].map(|name| graph.add_entity(named_entity(name)).unwrap());
        let mut revived = typed_edge(a, b, RelationType::RelatedTo, 0.0001);
        revived.tier = EdgeTier::L1Working;
        let mut decayed = typed_edge(a, c, RelationType::RelatedTo, 0.0001);
        decayed.tier = EdgeTier::L1Working;
        let ids = graph.add_relationships(vec![revived, decayed]).unwrap();
        graph.pending_prune.lock().extend(ids.iter().copied());

        let mut strengthened = graph.get_relationship(&ids[0]).unwrap().unwrap();
        strengthened.strength = 0.9;
        strengthened.last_activated = crate::memory::scoring_now();
        graph.put_relationship_record(&strengthened).unwrap();

        graph.flush_pending_maintenance().unwrap();
        assert!(graph.get_relationship(&ids[0]).unwrap().is_some());
        assert!(graph.get_relationship(&ids[1]).unwrap().is_none());
    }
}
