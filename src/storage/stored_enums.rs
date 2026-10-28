//! Pins the variant index of every enum a bincode record stores.
//!
//! bincode writes a variant as its position, so a variant inserted anywhere but the end
//! re-labels every stored value after it, and nothing fails to decode: upstream lost 74% of a
//! live graph's nodes to one inserted `EntityLabel`. A new variant goes last, and so does its
//! line here (docs/upstream/PLAN.md, 1.6).

use serde::Serialize;

use crate::graph_memory::{
    EdgeSource, EdgeTier, EntityLabel, EpisodeSource, LtpStatus, PiiClassification, RelationType,
};
use crate::memory::compression::FactType;
use crate::memory::facets::{
    AgentRole, BoundaryCause, CausalKind, ContentKind, PlanStatus, RecordKind, TimePrecision,
};
use crate::memory::types::{
    ChangeType, ExperienceType, FileType, LearnedFrom, MediaType, MemoryTier, ProjectStatus,
    ProspectiveTaskStatus, RelationshipType, SourceType, TodoCommentType, TodoPriority, TodoStatus,
};

fn variant_index<T: Serialize>(value: &T) -> u32 {
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap();
    let (index, _): (u32, usize) =
        bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    index
}

/// Each listed variant must encode as its position in the list.
macro_rules! pinned {
    ($($variant:expr),+ $(,)?) => {{
        let found = [$(variant_index(&$variant)),+];
        let expected: Vec<u32> = (0..found.len() as u32).collect();
        assert_eq!(found.to_vec(), expected, "{}", stringify!($($variant),+));
    }};
}

#[test]
fn memory_record_enums_keep_their_positions() {
    pinned!(
        ExperienceType::Conversation,
        ExperienceType::Decision,
        ExperienceType::Error,
        ExperienceType::Learning,
        ExperienceType::Discovery,
        ExperienceType::Pattern,
        ExperienceType::Context,
        ExperienceType::Task,
        ExperienceType::CodeEdit,
        ExperienceType::FileAccess,
        ExperienceType::Search,
        ExperienceType::Command,
        ExperienceType::Observation,
        ExperienceType::Intention,
    );
    pinned!(
        MediaType::Image,
        MediaType::Audio,
        MediaType::Video,
        MediaType::Document,
    );
    pinned!(
        SourceType::User,
        SourceType::System,
        SourceType::ExternalApi,
        SourceType::File,
        SourceType::Web,
        SourceType::AiGenerated,
        SourceType::Inferred,
        SourceType::Unknown,
    );
    pinned!(
        RelationshipType::IsA,
        RelationshipType::HasA,
        RelationshipType::Uses,
        RelationshipType::RelatedTo,
        RelationshipType::Causes,
        RelationshipType::PartOf,
        RelationshipType::Similar,
        RelationshipType::Opposite,
    );
    pinned!(
        MemoryTier::Working,
        MemoryTier::Session,
        MemoryTier::LongTerm,
        MemoryTier::Archive,
    );
    pinned!(
        ChangeType::Created,
        ChangeType::ContentUpdated,
        ChangeType::StatusChanged,
        ChangeType::TagsUpdated,
        ChangeType::ImportanceAdjusted,
    );
}

#[test]
fn facet_enums_keep_their_positions() {
    pinned!(
        RecordKind::Memory,
        RecordKind::Plan,
        RecordKind::Prompt,
        RecordKind::Learning,
    );
    pinned!(
        PlanStatus::Draft,
        PlanStatus::Active,
        PlanStatus::Blocked,
        PlanStatus::Done,
        PlanStatus::Abandoned,
    );
    pinned!(
        AgentRole::SelfAgent,
        AgentRole::Author,
        AgentRole::Subject,
        AgentRole::Audience,
        AgentRole::Mentioned,
    );
    pinned!(
        CausalKind::Caused,
        CausalKind::Triggered,
        CausalKind::Enabled,
        CausalKind::Prevented,
        CausalKind::Motivated,
    );
    pinned!(
        BoundaryCause::GoalChange,
        BoundaryCause::PredictionError,
        BoundaryCause::LocationChange,
        BoundaryCause::AgentChange,
        BoundaryCause::TemporalGap,
    );
    pinned!(
        ContentKind::Observation,
        ContentKind::Decision,
        ContentKind::Learning,
        ContentKind::Action,
        ContentKind::Reflection,
        ContentKind::Conversation,
        ContentKind::Code,
        ContentKind::Document,
        ContentKind::Other,
    );
    pinned!(
        TimePrecision::Year,
        TimePrecision::Month,
        TimePrecision::Day,
        TimePrecision::Hour,
        TimePrecision::Minute,
        TimePrecision::Instant,
    );
}

#[test]
fn todo_project_and_file_enums_keep_their_positions() {
    pinned!(
        TodoStatus::Backlog,
        TodoStatus::Todo,
        TodoStatus::InProgress,
        TodoStatus::Blocked,
        TodoStatus::Done,
        TodoStatus::Cancelled,
    );
    pinned!(
        TodoPriority::Urgent,
        TodoPriority::High,
        TodoPriority::Medium,
        TodoPriority::Low,
        TodoPriority::None,
    );
    pinned!(
        TodoCommentType::Comment,
        TodoCommentType::Activity,
        TodoCommentType::Progress,
        TodoCommentType::Resolution,
    );
    pinned!(
        ProjectStatus::Active,
        ProjectStatus::OnHold,
        ProjectStatus::Completed,
        ProjectStatus::Archived,
    );
    pinned!(
        ProspectiveTaskStatus::Pending,
        ProspectiveTaskStatus::Triggered,
        ProspectiveTaskStatus::Dismissed,
        ProspectiveTaskStatus::Expired,
    );
    pinned!(
        FileType::Rust,
        FileType::TypeScript,
        FileType::JavaScript,
        FileType::Python,
        FileType::Go,
        FileType::Java,
        FileType::CSharp,
        FileType::Cpp,
        FileType::C,
        FileType::Ruby,
        FileType::Markdown,
        FileType::Json,
        FileType::Yaml,
        FileType::Toml,
        FileType::Html,
        FileType::Css,
        FileType::Sql,
        FileType::Shell,
        FileType::Other(String::new()),
    );
    pinned!(
        LearnedFrom::ManualIndex,
        LearnedFrom::ReadAccess,
        LearnedFrom::EditAccess,
        LearnedFrom::Mentioned,
    );
}

#[test]
fn graph_enums_keep_their_positions() {
    pinned!(
        PiiClassification::PersonalIdentity,
        PiiClassification::OrganizationIdentity,
        PiiClassification::Clean,
    );
    pinned!(
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
        EntityLabel::Other(String::new()),
        EntityLabel::FilePath,
        EntityLabel::Commit,
        EntityLabel::Symbol,
        EntityLabel::Session,
        EntityLabel::Agent,
        EntityLabel::Tool,
        EntityLabel::Template,
    );
    pinned!(
        EdgeTier::L1Working,
        EdgeTier::L2Episodic,
        EdgeTier::L3Semantic,
    );
    pinned!(
        LtpStatus::None,
        LtpStatus::Burst {
            detected_at: chrono::Utc::now()
        },
        LtpStatus::Weekly,
        LtpStatus::Full,
    );
    pinned!(
        EdgeSource::CoOccurrence,
        EdgeSource::DreamReplay,
        EdgeSource::SleepReplay,
        EdgeSource::Coactivation,
        EdgeSource::Explicit,
        EdgeSource::CollectivePrior,
        EdgeSource::Unknown,
    );
    pinned!(
        RelationType::WorksWith,
        RelationType::WorksAt,
        RelationType::EmployedBy,
        RelationType::PartOf,
        RelationType::Contains,
        RelationType::OwnedBy,
        RelationType::LocatedIn,
        RelationType::LocatedAt,
        RelationType::Uses,
        RelationType::CreatedBy,
        RelationType::DevelopedBy,
        RelationType::Causes,
        RelationType::ResultsIn,
        RelationType::Learned,
        RelationType::Knows,
        RelationType::Teaches,
        RelationType::RelatedTo,
        RelationType::AssociatedWith,
        RelationType::CoRetrieved,
        RelationType::CoOccurs,
        RelationType::Manages,
        RelationType::DependsOn,
        RelationType::Requires,
        RelationType::Configures,
        RelationType::Prefers,
        RelationType::Recommends,
        RelationType::Documents,
        RelationType::Implements,
        RelationType::DeploysTo,
        RelationType::Monitors,
        RelationType::Triggers,
        RelationType::SupersededBy,
        RelationType::AlternativeTo,
        RelationType::AssignedTo,
        RelationType::Approves,
        RelationType::Custom(String::new()),
    );
    pinned!(
        EpisodeSource::Message,
        EpisodeSource::Document,
        EpisodeSource::Event,
        EpisodeSource::Observation,
    );
}

#[test]
fn fact_enums_keep_their_positions() {
    pinned!(
        FactType::Preference,
        FactType::Capability,
        FactType::Relationship,
        FactType::Procedure,
        FactType::Definition,
        FactType::Pattern,
    );
}

/// An entity written before the declared labels existed decodes unchanged. The mirror below is
/// `EntityNode` with the label enum as it stood (23 named variants, then `Other`), so the bytes
/// are exactly what an older build stored.
#[test]
fn entity_records_written_before_the_declared_labels_still_decode() {
    use std::collections::HashMap;

    use chrono::{DateTime, Utc};
    use uuid::Uuid;

    use crate::graph_memory::EntityNode;

    #[allow(dead_code)]
    #[derive(Serialize)]
    enum OldEntityLabel {
        Person,
        Organization,
        Location,
        Technology,
        Concept,
        Event,
        Date,
        Product,
        Skill,
        Keyword,
        Project,
        Task,
        Document,
        Repository,
        Service,
        Database,
        Metric,
        Configuration,
        Environment,
        Pipeline,
        Team,
        Role,
        Module,
        Other(String),
    }

    #[derive(Serialize)]
    struct OldEntityNode {
        uuid: Uuid,
        name: String,
        labels: Vec<OldEntityLabel>,
        created_at: DateTime<Utc>,
        last_seen_at: DateTime<Utc>,
        mention_count: usize,
        summary: String,
        attributes: HashMap<String, String>,
        name_embedding: Option<Vec<f32>>,
        salience: f32,
        is_proper_noun: bool,
        pii_classification: PiiClassification,
    }

    let now = Utc::now();
    let old = OldEntityNode {
        uuid: Uuid::new_v4(),
        name: "veld".to_string(),
        labels: vec![
            OldEntityLabel::Person,
            OldEntityLabel::Module,
            OldEntityLabel::Other("MISC".to_string()),
        ],
        created_at: now,
        last_seen_at: now,
        mention_count: 3,
        summary: String::new(),
        attributes: HashMap::new(),
        name_embedding: None,
        salience: 0.7,
        is_proper_noun: true,
        pii_classification: PiiClassification::Clean,
    };
    let bytes = bincode::serde::encode_to_vec(&old, bincode::config::standard()).unwrap();
    let (decoded, used): (EntityNode, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(used, bytes.len());
    assert_eq!(decoded.name, "veld");
    assert_eq!(
        decoded.labels,
        vec![
            EntityLabel::Person,
            EntityLabel::Module,
            EntityLabel::Other("MISC".to_string()),
        ]
    );
    assert_eq!(decoded.mention_count, 3);
}
