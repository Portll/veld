//! bincode 2 with serde and `config::standard()` is positional: a record encoded with
//! an old struct shape does not decode into a shape with appended fields, whatever
//! `#[serde(default)]` says. This used to be a probe that failed to report the fact;
//! it now asserts it, and `tests/persisted_layout.rs` is the guard that acts on it.
//! If this ever starts failing, bincode's encoding changed and that guard's premise
//! needs re-examining.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct OldShape {
    id: String,
    confidence: f32,
    support: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NewShape {
    id: String,
    confidence: f32,
    support: usize,
    #[serde(default)]
    valid_from: Option<i64>,
    #[serde(default)]
    valid_until: Option<i64>,
    #[serde(default)]
    superseded_by: Option<String>,
}

#[test]
fn bincode_schema_evolution_probe() {
    let old = OldShape {
        id: "fact-1".to_string(),
        confidence: 0.8,
        support: 3,
    };
    let bytes =
        bincode::serde::encode_to_vec(&old, bincode::config::standard()).expect("encode OldShape");

    let result: Result<(NewShape, usize), _> =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard());

    assert!(
        result.is_err(),
        "bincode decoded old-shape bytes into a shape with appended fields: {result:?}. \
         Encoding is no longer positional, so tests/persisted_layout.rs may be stricter \
         than it needs to be."
    );
}
