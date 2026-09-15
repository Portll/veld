//! bincode encodes by position, so any change to the field layout of a persisted type
//! makes every record written before it undecodable: `serde(default)` does nothing.
//! 4a54053 appended two fields to AgentSession and 24 live records stopped decoding.
//!
//! This snapshots the traced serde layout of every type reachable from `Memory`,
//! including types that are `None` in practice. A layout change fails here, at the
//! commit that makes it; the fix is a legacy decode arm for the old shape, then
//! `VELD_WRITE_LAYOUT=1 cargo test --test persisted_layout` to accept the new one.

use std::path::PathBuf;

use serde_reflection::{Registry, Samples, Tracer, TracerConfig};
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

fn trace() -> Registry {
    let config = TracerConfig::default()
        .is_human_readable(false)
        // chrono parses its string form even in binary formats.
        .default_borrowed_str_value("2026-01-01T00:00:00Z")
        .default_string_value("2026-01-01T00:00:00Z".to_string())
        // uuid's binary form is exactly 16 bytes.
        .default_byte_buf_value(vec![0u8; 16])
        .default_borrowed_bytes_value(&[0u8; 16]);
    let mut tracer = Tracer::new(config);
    let samples = Samples::new();
    trace_enums!(
        tracer,
        &samples,
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
    tracer
        .trace_type::<Memory>(&samples)
        .unwrap_or_else(|e| panic!("tracing Memory: {e}"));
    tracer
        .registry()
        .unwrap_or_else(|e| panic!("incomplete trace: {e}"))
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/persisted_layout/memory.json")
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

#[test]
fn persisted_memory_layout_is_unchanged() {
    let current = serde_json::to_string_pretty(&trace()).unwrap() + "\n";
    let path = snapshot_path();

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
