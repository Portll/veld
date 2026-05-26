//! Empirical probe: does `bincode 2 + serde + config::standard()` decode a
//! record encoded with the OLD struct shape when the NEW struct adds trailing
//! `Option<T>` fields marked `#[serde(default)]`?

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

    let bytes = bincode::serde::encode_to_vec(&old, bincode::config::standard())
        .expect("encode OldShape");

    eprintln!("Encoded OldShape -> {} bytes: {:?}", bytes.len(), bytes);

    let result: Result<(NewShape, usize), _> =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard());

    match result {
        Ok((decoded, consumed)) => {
            eprintln!("DECODE SUCCEEDED");
            eprintln!("  consumed: {} of {} bytes", consumed, bytes.len());
            eprintln!("  decoded:  {:?}", decoded);
            assert_eq!(decoded.id, "fact-1");
            assert_eq!(decoded.confidence, 0.8);
            assert_eq!(decoded.support, 3);
            assert_eq!(decoded.valid_from, None);
            assert_eq!(decoded.valid_until, None);
            assert_eq!(decoded.superseded_by, None);
            eprintln!("VERDICT: serde(default) HANDLES trailing Options under bincode 2 std config.");
            eprintln!("STRATEGY: PR 1 B1 mitigation NOT required for additive Option fields.");
        }
        Err(e) => {
            eprintln!("DECODE FAILED — error: {}", e);
            eprintln!("VERDICT: bincode 2 IS positional; serde(default) does NOT cover trailing Options.");
            eprintln!("STRATEGY: PR 1 B1 mitigation REQUIRED — use versioned read shim.");
            panic!("bincode cannot decode old-shape bytes into new-shape struct: {}", e);
        }
    }
}
