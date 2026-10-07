//! Entities a writer declares on a memory, typed, as opposed to entities veld derives from the
//! content.
//!
//! The writer often knows exactly which file, commit, session or agent a memory is about, and
//! before this the only way to say so was a tag: every tag became a `Technology` entity at
//! salience 0.6 and then lost the per-memory top-N cut to NER noise at 0.7 and above. A declared
//! entity keeps its type, is exempt from that cut and from the generic-entity gate, and is linked
//! with a protected edge so degree-cap pruning removes it last.
//!
//! Wire shape, on `POST /api/remember`, `POST /api/upsert` and each item of
//! `POST /api/remember/batch`:
//!
//! ```json
//! "entities": [{ "name": "commitwork:admin/serve.mjs", "label": "FilePath", "weight": 0.9 }]
//! ```
//!
//! Validation fails closed: an unknown label is refused with the accepted list, never filed as
//! `Other`.
//!
//! Declared names are not stored on `Memory` with their labels — `Memory` is bincode-encoded and
//! positional, so a new field there makes every earlier record undecodable. The names travel as
//! the record's authored tags (labelled `declared` in the tag-source ledger); the labels and
//! weights live on the graph's entity nodes and edges, which already persist them.

use serde::Deserialize;

use crate::errors::AppError;
use crate::graph_memory::EntityLabel;

/// Most entities one memory may declare.
pub const MAX_DECLARED_ENTITIES: usize = 50;
/// Longest declared name, in characters, after trimming.
pub const MAX_DECLARED_NAME_CHARS: usize = 512;
/// Weight of a declared entity that names none.
pub const DEFAULT_DECLARED_WEIGHT: f32 = 0.9;

/// One declared entity as it arrives on the wire.
#[derive(Debug, Clone, Deserialize)]
pub struct DeclaredEntityInput {
    pub name: String,
    pub label: String,
    #[serde(default)]
    pub weight: Option<f64>,
}

/// A validated declared entity. Two declarations of one name (case-insensitively) fold into one,
/// keeping the first spelling, every label and the larger weight.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredEntity {
    pub name: String,
    pub labels: Vec<EntityLabel>,
    pub weight: f32,
}

fn invalid(field: String, reason: String) -> AppError {
    AppError::InvalidInput { field, reason }
}

/// Validate and normalise a request's declared entities.
pub fn validate(inputs: &[DeclaredEntityInput]) -> Result<Vec<DeclaredEntity>, AppError> {
    if inputs.len() > MAX_DECLARED_ENTITIES {
        return Err(invalid(
            "entities".to_string(),
            format!(
                "{} entities declared; at most {MAX_DECLARED_ENTITIES} are accepted per memory",
                inputs.len()
            ),
        ));
    }
    let mut out: Vec<DeclaredEntity> = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let name = input.name.trim();
        let chars = name.chars().count();
        if chars == 0 || chars > MAX_DECLARED_NAME_CHARS {
            return Err(invalid(
                format!("entities[{i}].name"),
                format!("name must be 1..={MAX_DECLARED_NAME_CHARS} characters after trimming, got {chars}"),
            ));
        }
        if name.chars().any(char::is_control) {
            return Err(invalid(
                format!("entities[{i}].name"),
                "name must not contain control characters".to_string(),
            ));
        }
        let Some(label) = EntityLabel::parse_declared(&input.label) else {
            return Err(invalid(
                format!("entities[{i}].label"),
                format!(
                    "Unknown entity label '{}'. Accepted labels (case-insensitive): {}",
                    input.label,
                    EntityLabel::DECLARABLE.join(", ")
                ),
            ));
        };
        let weight = match input.weight {
            None => DEFAULT_DECLARED_WEIGHT,
            Some(w) if w.is_finite() && (0.0..=1.0).contains(&w) => w as f32,
            Some(w) => {
                return Err(invalid(
                    format!("entities[{i}].weight"),
                    format!("weight must be within [0, 1], got {w}"),
                ));
            }
        };
        let key = name.to_lowercase();
        if let Some(existing) = out.iter_mut().find(|d| d.name.to_lowercase() == key) {
            if !existing.labels.contains(&label) {
                existing.labels.push(label);
            }
            existing.weight = existing.weight.max(weight);
        } else {
            out.push(DeclaredEntity {
                name: name.to_string(),
                labels: vec![label],
                weight,
            });
        }
    }
    Ok(out)
}

/// Drop declarations whose name the redaction lexicon would replace. The tag list still carries
/// the redaction label, as it does for any credential-shaped tag; what must not happen is a typed,
/// protected graph node minted from a credential, or one shared `redacted:*` node every such
/// declaration would fold into.
pub fn retain_unredacted(declared: &mut Vec<DeclaredEntity>) {
    declared.retain(|d| crate::redaction::classify(&d.name).is_none());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str, label: &str, weight: Option<f64>) -> DeclaredEntityInput {
        DeclaredEntityInput {
            name: name.to_string(),
            label: label.to_string(),
            weight,
        }
    }

    fn reason(err: AppError) -> (String, String) {
        match err {
            AppError::InvalidInput { field, reason } => (field, reason),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn labels_parse_case_insensitively_and_default_the_weight() {
        let got = validate(&[
            input(" commitwork:admin/serve.mjs ", "filepath", None),
            input("commitwork@1a2b3c4d5e6f", "COMMIT", Some(0.8)),
            input("Alice", "Person", Some(1.0)),
        ])
        .unwrap();
        assert_eq!(got[0].name, "commitwork:admin/serve.mjs");
        assert_eq!(got[0].labels, vec![EntityLabel::FilePath]);
        assert_eq!(got[0].weight, DEFAULT_DECLARED_WEIGHT);
        assert_eq!(got[1].labels, vec![EntityLabel::Commit]);
        assert!((got[1].weight - 0.8).abs() < 1e-6);
        assert_eq!(got[2].labels, vec![EntityLabel::Person]);
    }

    #[test]
    fn every_declarable_name_parses_and_other_does_not() {
        for name in EntityLabel::DECLARABLE {
            let label = EntityLabel::parse_declared(name)
                .unwrap_or_else(|| panic!("{name} is listed as declarable but does not parse"));
            assert_eq!(label.as_str(), *name, "listed spelling must round-trip");
        }
        assert_eq!(EntityLabel::parse_declared("Other"), None);
        assert_eq!(EntityLabel::parse_declared("MISC"), None);
    }

    #[test]
    fn an_unknown_label_is_refused_with_the_accepted_list() {
        let (field, why) = reason(validate(&[input("x", "Filepathh", None)]).unwrap_err());
        assert_eq!(field, "entities[0].label");
        assert!(why.contains("Filepathh"), "{why}");
        for name in EntityLabel::DECLARABLE {
            assert!(why.contains(name), "{name} missing from {why}");
        }
    }

    #[test]
    fn weight_out_of_range_is_refused() {
        for w in [1.5, -0.01, f64::NAN] {
            let (field, _) = reason(validate(&[input("x", "Tool", Some(w))]).unwrap_err());
            assert_eq!(field, "entities[0].weight", "weight {w}");
        }
        assert!(validate(&[input("x", "Tool", Some(0.0))]).is_ok());
        assert!(validate(&[input("x", "Tool", Some(1.0))]).is_ok());
    }

    #[test]
    fn names_are_bounded_and_free_of_control_characters() {
        let (field, _) = reason(validate(&[input("   ", "Tool", None)]).unwrap_err());
        assert_eq!(field, "entities[0].name");
        let long = "a".repeat(MAX_DECLARED_NAME_CHARS + 1);
        assert!(validate(&[input(&long, "Tool", None)]).is_err());
        let exact = "é".repeat(MAX_DECLARED_NAME_CHARS);
        assert!(
            validate(&[input(&exact, "Tool", None)]).is_ok(),
            "the bound is characters, not bytes"
        );
        assert!(validate(&[input("tool:\u{0007}Edit", "Tool", None)]).is_err());
        assert!(
            validate(&[input("tool:Edit\n", "Tool", None)]).is_ok(),
            "trimmed first"
        );
    }

    #[test]
    fn more_than_fifty_is_refused_and_fifty_is_not() {
        let fifty: Vec<_> = (0..MAX_DECLARED_ENTITIES)
            .map(|i| input(&format!("tool:t{i}"), "Tool", None))
            .collect();
        assert_eq!(validate(&fifty).unwrap().len(), MAX_DECLARED_ENTITIES);
        let mut fifty_one = fifty;
        fifty_one.push(input("tool:extra", "Tool", None));
        let (field, _) = reason(validate(&fifty_one).unwrap_err());
        assert_eq!(field, "entities");
    }

    #[test]
    fn a_repeated_name_folds_into_one_declaration() {
        let got = validate(&[
            input("cc:abc", "Session", Some(0.4)),
            input("CC:ABC", "Agent", Some(0.7)),
        ])
        .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "cc:abc");
        assert_eq!(
            got[0].labels,
            vec![EntityLabel::Session, EntityLabel::Agent]
        );
        assert!((got[0].weight - 0.7).abs() < 1e-6);
    }

    #[test]
    fn credential_shaped_declarations_do_not_become_typed_nodes() {
        let mut got = validate(&[
            input("ghp_abcdefghijklmnopqrstuvwxyz0123", "Tool", None),
            input("commitwork:admin/serve.mjs", "FilePath", None),
        ])
        .unwrap();
        retain_unredacted(&mut got);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "commitwork:admin/serve.mjs");
    }
}
