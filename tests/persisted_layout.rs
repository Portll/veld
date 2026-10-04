//! bincode encodes by position, so any change to the field layout of a persisted type
//! makes every record written before it undecodable: `serde(default)` does nothing.
//! 4a54053 appended two fields to AgentSession and 24 live records stopped decoding.
//!
//! This snapshots the traced serde layout of every type reachable from `Memory`,
//! including types that are `None` in practice (`memory.json`), and of every other root
//! the stores bincode-encode (`roots.json`). A layout change fails here, at the commit
//! that makes it; the fix is a legacy decode arm for the old shape, then
//! `VELD_WRITE_LAYOUT=1 cargo test --test persisted_layout` to accept the new one.

use std::path::PathBuf;

use serde_reflection::{Registry, Samples, Tracer, TracerConfig};
use veld::memory::compression::PurgeReason;
use veld::memory::facets::{
    AgentRole, BoundaryCause, CausalRelation, ContentKind, Place, RecordKind, TimePrecision,
};
use veld::memory::types::{
    ChangeType, ExperienceType, MediaType, Memory, MemoryTier, RelationshipType, SourceType,
};

// trace_type walks only the variants of the type it is given, so every enum reachable
// from Memory is traced by name. A new one fails the trace, naming itself.
macro_rules! trace_enums {
    ($tracer:expr, $samples:expr, $($t:ty),* $(,)?) => {
        $( $tracer
            .trace_type::<$t>($samples)
            .unwrap_or_else(|e| panic!("tracing {}: {e}", stringify!($t))); )*
    };
}

fn tracer_config() -> TracerConfig {
    TracerConfig::default()
        .is_human_readable(false)
        // chrono parses its string form even in binary formats.
        .default_borrowed_str_value("2026-01-01T00:00:00Z")
        .default_string_value("2026-01-01T00:00:00Z".to_string())
        // uuid's binary form is exactly 16 bytes.
        .default_byte_buf_value(vec![0u8; 16])
        .default_borrowed_bytes_value(&[0u8; 16])
}

fn trace_memory_enums(tracer: &mut Tracer, samples: &Samples) {
    trace_enums!(
        tracer,
        samples,
        AgentRole,
        BoundaryCause,
        CausalRelation,
        ChangeType,
        ContentKind,
        ExperienceType,
        MediaType,
        MemoryTier,
        Place,
        RecordKind,
        RelationshipType,
        SourceType,
        TimePrecision,
    );
}

fn trace() -> Registry {
    let mut tracer = Tracer::new(tracer_config());
    let samples = Samples::new();
    trace_memory_enums(&mut tracer, &samples);
    tracer
        .trace_type::<Memory>(&samples)
        .unwrap_or_else(|e| panic!("tracing Memory: {e}"));
    tracer
        .registry()
        .unwrap_or_else(|e| panic!("incomplete trace: {e}"))
}

// Every type the stores bincode-encode, beyond Memory. A type bincode cannot decode
// (internally tagged, untagged, flatten: anything needing deserialize_any) encodes
// without error and never reads back; PurgeReason under SemanticFact did exactly that.
// HnswSnapshotOwned and vamana's MemoryEntry are private and guarded in their modules.
fn trace_all_roots() -> Registry {
    use veld::embeddings::alignment::AlignmentHeader;
    use veld::graph_memory::{
        EdgeSource, EdgeTier, EntityLabel, EpisodeSource, LtpStatus, PiiClassification,
        RelationType,
    };
    use veld::graph_memory::{EntityNode, EpisodicNode, RelationshipEdge};
    use veld::intent_log::payload::IntentPayload;
    use veld::memory::acl::MemoryAcl;
    use veld::memory::compression::FactType;
    use veld::memory::context_blocks::ContextBlock;
    use veld::memory::lineage::{CausalRelation as LineageCausalRelation, LineageSource};
    use veld::memory::lineage::{LineageBranch, LineageEdge};
    use veld::memory::sleep_time::graduation::GraduationRecord;
    use veld::memory::sleep_time::supersession::SupersessionRecord;
    use veld::memory::sleep_time::types::{BudgetState, GlobalBudgetState, QueueItem};
    use veld::memory::sleep_time::types::{SleepMode, SleepTimeTrigger};
    use veld::memory::storage::Modality;
    use veld::memory::storage::VectorMappingEntry;
    use veld::memory::types::Experience;
    use veld::memory::SemanticFact;
    use veld::storage::AuditLogEntry;
    use veld::user_auth::session::SessionRecord;
    use veld::user_auth::store::LoginThrottleRecord;
    use veld::user_auth::{UserRecord, UserRole};

    let mut tracer = Tracer::new(tracer_config());
    let samples = Samples::new();
    trace_memory_enums(&mut tracer, &samples);
    trace_enums!(
        tracer,
        &samples,
        PurgeReason,
        FactType,
        EdgeSource,
        EdgeTier,
        EntityLabel,
        EpisodeSource,
        LtpStatus,
        PiiClassification,
        RelationType,
        LineageCausalRelation,
        LineageSource,
        SleepMode,
        SleepTimeTrigger,
        Modality,
        UserRole,
    );
    trace_enums!(
        tracer,
        &samples,
        Memory,
        Experience,
        SemanticFact,
        RelationshipEdge,
        EntityNode,
        EpisodicNode,
        VectorMappingEntry,
        AuditLogEntry,
        IntentPayload,
        AlignmentHeader,
        QueueItem,
        BudgetState,
        GlobalBudgetState,
        GraduationRecord,
        SupersessionRecord,
        ContextBlock,
        LineageEdge,
        LineageBranch,
        MemoryAcl,
        UserRecord,
        LoginThrottleRecord,
        SessionRecord,
    );
    tracer.registry().unwrap_or_else(|e| {
        panic!("incomplete trace: {e}. Trace the named enum in the list above.")
    })
}

#[test]
fn every_bincode_persisted_root_is_decodable() {
    trace_all_roots();
}

// serde-reflection cannot trace TemporalFact: NaiveDate and DateTime both read the one
// default string, and no string parses as both. This witness round-trips every variant
// instead; the exhaustive destructure and matches stop compiling when a field or variant
// is added, and the coverage asserts fail until it has a sample here.
#[test]
fn temporal_fact_round_trips_through_bincode_in_every_variant() {
    use chrono::{NaiveDate, Utc};
    use veld::memory::temporal_facts::{EventType, ResolvedTime, TemporalFact};
    use veld::memory::types::MemoryId;

    const RESOLVED_TIME_VARIANTS: usize = 8;
    fn resolved_time_index(t: &ResolvedTime) -> usize {
        match t {
            ResolvedTime::ExactDate(_) => 0,
            ResolvedTime::MonthYear { .. } => 1,
            ResolvedTime::Year(_) => 2,
            ResolvedTime::RelativeDescription(_) => 3,
            ResolvedTime::Unknown => 4,
            ResolvedTime::After(_) => 5,
            ResolvedTime::Before(_) => 6,
            ResolvedTime::Around { .. } => 7,
        }
    }
    const EVENT_TYPE_VARIANTS: usize = 4;
    fn event_type_index(e: &EventType) -> usize {
        match e {
            EventType::Planned => 0,
            EventType::Occurred => 1,
            EventType::Historical => 2,
            EventType::PastIntent => 3,
        }
    }

    let d = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
    let times = vec![
        ResolvedTime::ExactDate(d),
        ResolvedTime::MonthYear {
            month: 1,
            year: 2026,
        },
        ResolvedTime::Year(2026),
        ResolvedTime::RelativeDescription("next week".into()),
        ResolvedTime::Unknown,
        ResolvedTime::After(d),
        ResolvedTime::Before(d),
        ResolvedTime::Around {
            centre: d,
            radius_days: 3,
        },
    ];
    let events = vec![
        EventType::Planned,
        EventType::Occurred,
        EventType::Historical,
        EventType::PastIntent,
    ];
    let covered: std::collections::BTreeSet<usize> =
        times.iter().map(resolved_time_index).collect();
    assert_eq!(
        covered.len(),
        RESOLVED_TIME_VARIANTS,
        "a ResolvedTime variant has no sample"
    );
    let covered: std::collections::BTreeSet<usize> = events.iter().map(event_type_index).collect();
    assert_eq!(
        covered.len(),
        EVENT_TYPE_VARIANTS,
        "an EventType variant has no sample"
    );

    let now = Utc::now();
    for (i, resolved_time) in times.into_iter().enumerate() {
        let fact = TemporalFact {
            id: "t".into(),
            entity: "e".into(),
            event: "ev".into(),
            event_stems: vec!["ev".into()],
            event_type: events[i % events.len()],
            relative_time: Some("then".into()),
            resolved_time,
            source_memory_id: MemoryId(uuid::Uuid::nil()),
            conversation_date: now,
            confidence: 0.5,
            source_text: "s".into(),
            valid_from: Some(now),
            valid_until: Some(now),
        };
        // No `..`: a new field stops this compiling until it is set to a non-default above.
        let TemporalFact {
            id: _,
            entity: _,
            event: _,
            event_stems: _,
            event_type: _,
            relative_time: _,
            resolved_time: _,
            source_memory_id: _,
            conversation_date: _,
            confidence: _,
            source_text: _,
            valid_from: _,
            valid_until: _,
        } = &fact;

        let bytes = bincode::serde::encode_to_vec(&fact, bincode::config::standard()).unwrap();
        let (back, read): (TemporalFact, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap_or_else(|e| panic!("TemporalFact {:?} does not decode: {e}", fact));
        assert_eq!(read, bytes.len());
        let again = bincode::serde::encode_to_vec(&back, bincode::config::standard()).unwrap();
        assert_eq!(
            again, bytes,
            "TemporalFact {:?} changed across a round trip",
            fact
        );
    }
}

fn snapshot_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/persisted_layout")
        .join(format!("{name}.json"))
}

#[test]
fn the_trace_reaches_nested_types_that_are_none_in_practice() {
    let registry = trace();
    for name in ["MemoryFlat", "Experience", "RichContext", "AgentSession"] {
        assert!(
            registry.contains_key(name),
            "trace never reached {name}: {:?}",
            registry.keys()
        );
    }
}

/// Compare `registry` with the committed snapshot `name`, or write it under VELD_WRITE_LAYOUT.
fn assert_layout_unchanged(name: &str, registry: &Registry) {
    let current = serde_json::to_string_pretty(registry).unwrap() + "\n";
    let path = snapshot_path(name);

    if std::env::var("VELD_WRITE_LAYOUT").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &current).unwrap();
        return;
    }

    let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e} (generate with VELD_WRITE_LAYOUT=1)",
            path.display()
        )
    });
    if committed == current {
        return;
    }

    let old: Registry = serde_json::from_str(&committed).unwrap();
    let new: Registry = serde_json::from_str(&current).unwrap();
    let changed: std::collections::BTreeSet<&String> = old
        .keys()
        .chain(new.keys())
        .filter(|k| old.get(*k) != new.get(*k))
        .collect();
    panic!(
        "persisted layout changed for {changed:?}. bincode is positional: records written \
         before this change will not decode. Add a legacy decode arm for the old shape, \
         test it on a record of that shape, then regenerate with VELD_WRITE_LAYOUT=1."
    );
}

#[test]
fn persisted_memory_layout_is_unchanged() {
    assert_layout_unchanged("memory", &trace());
}

/// Graph nodes and edges, facts, vector mappings, audit and journal entries, auth records:
/// every root the stores bincode-encode, beside Memory.
#[test]
fn every_persisted_root_layout_is_unchanged() {
    assert_layout_unchanged("roots", &trace_all_roots());
}
