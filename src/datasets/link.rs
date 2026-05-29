//! Row-to-graph link types.
//!
//! A [`RowLink`] connects a specific row in a dataset (identified by its
//! primary key values) to either a knowledge-graph entity or a memory
//! record. The store side of links — listing, indexing, querying — lands
//! in a follow-up agent; this module defines the wire types only.

use serde::{Deserialize, Serialize};

use crate::datasets::store::DatasetRef;

/// Concrete primary-key value tuple identifying a single row.
///
/// `values` holds one JSON value per primary-key column, in the column
/// order declared by [`crate::datasets::schema::DatasetSchema::primary_key`].
/// JSON is used so that mixed-type composite keys round-trip cleanly across
/// the store boundary without a custom encoding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RowPk {
    pub values: Vec<serde_json::Value>,
}

/// What kind of graph object a [`RowLink`] points at.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LinkKind {
    /// Link target is a knowledge-graph entity (`target_id` is a UUID).
    Entity,
    /// Link target is a memory record (`target_id` is the memory id).
    Memory,
}

/// A directed association from a dataset row to a graph object.
///
/// `target_id` is stored as a `String` so the same link type can address
/// either an [`uuid::Uuid`] entity or a memory-id (which is also a UUID
/// today, but is treated as an opaque identifier at this layer).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RowLink {
    pub dataset: DatasetRef,
    pub row_pk: RowPk,
    pub kind: LinkKind,
    pub target_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_round_trips_through_json() {
        let link = RowLink {
            dataset: DatasetRef {
                user_id: "alice".to_string(),
                name: "events".to_string(),
                table: "alice__dataset__events".to_string(),
            },
            row_pk: RowPk {
                values: vec![serde_json::json!(42), serde_json::json!("abc")],
            },
            kind: LinkKind::Entity,
            target_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        };
        let json = serde_json::to_string(&link).expect("encode");
        let back: RowLink = serde_json::from_str(&json).expect("decode");
        assert_eq!(link, back);
    }
}
