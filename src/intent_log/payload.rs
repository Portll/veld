//! Typed payloads for the intent log.
//!
//! The intent log itself treats every record's payload as opaque bytes —
//! that lets the log primitive be reused for anything (W4 Postgres,
//! eventual fleet sync, …). For Veld's own use of the log, every payload
//! is bincoded from one of the [`IntentPayload`] variants below.
//!
//! ## Why bincode, not JSON
//!
//! Frame storage is on the hot write path. JSON would inflate the on-disk
//! footprint 2-3× and slow encode/decode. Bincode 2 is already Veld's
//! storage format for `Memory`/`Experience` (see `memory::storage`) so
//! using it here keeps the dependency surface and skill set narrow.
//!
//! ## Schema evolution
//!
//! `IntentPayload` is `#[non_exhaustive]` and each variant uses serde's
//! `default` attribute on every optional field. Adding a new variant or
//! field is forward-compatible — older binaries skip records they don't
//! recognise without crashing, and newer binaries deserialise older
//! records by filling in defaults. A breaking change (renaming a field,
//! changing a type) requires a `format_version` bump in the log header,
//! which we will add when the first such change comes up.

use serde::{Deserialize, Serialize};

use super::{IntentLog, IntentLogError, IntentRecord, Lsn};

/// Every state-changing operation that must be journaled before it lands
/// in any projection. The current variants cover the core memory CRUD; we
/// add more as additional operations are folded into the log.
///
/// All variants are tenant-scoped: `user_id` is required so a per-tenant
/// projection can filter on it without parsing the payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IntentPayload {
    /// A new memory was created. The full memory snapshot is captured so
    /// projections can rebuild from scratch by replaying the log.
    Remember {
        user_id: String,
        memory_id: String,
        /// Bincoded `Memory` snapshot at the moment of write. Carried as
        /// opaque bytes here so changes to the `Memory` schema don't
        /// require updating `IntentPayload`.
        memory_bincode: Vec<u8>,
    },
    /// An existing memory was deleted.
    Forget {
        user_id: String,
        memory_id: String,
    },
    /// An existing memory was edited in place. Same opaque-bytes pattern
    /// as `Remember`.
    Update {
        user_id: String,
        memory_id: String,
        memory_bincode: Vec<u8>,
    },
    /// An existing memory was anchored (decay-resistant boost).
    Anchor {
        user_id: String,
        memory_id: String,
        importance: f32,
    },
}

impl IntentPayload {
    /// Tenant the record belongs to. Used by per-tenant projections to
    /// skip records that aren't theirs without doing a full decode of the
    /// inner memory bytes.
    pub fn user_id(&self) -> &str {
        match self {
            IntentPayload::Remember { user_id, .. } => user_id,
            IntentPayload::Forget { user_id, .. } => user_id,
            IntentPayload::Update { user_id, .. } => user_id,
            IntentPayload::Anchor { user_id, .. } => user_id,
        }
    }

    /// Memory the record is about. Useful for cross-store dedupe and for
    /// projections that key by memory id (e.g. a "last-applied per id"
    /// idempotency table).
    pub fn memory_id(&self) -> &str {
        match self {
            IntentPayload::Remember { memory_id, .. } => memory_id,
            IntentPayload::Forget { memory_id, .. } => memory_id,
            IntentPayload::Update { memory_id, .. } => memory_id,
            IntentPayload::Anchor { memory_id, .. } => memory_id,
        }
    }
}

/// Errors raised when encoding/decoding typed payloads.
#[derive(Debug, thiserror::Error)]
pub enum PayloadError {
    #[error("bincode encode error: {0}")]
    Encode(String),
    #[error("bincode decode error: {0}")]
    Decode(String),
    #[error("intent log error: {0}")]
    Log(#[from] IntentLogError),
}

/// Bincode configuration used for every payload. Pinned so the same bytes
/// always come out — change this only behind a format-version bump.
fn bincode_cfg() -> bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

/// Encode a typed payload to bytes. Pure function — does not touch the log.
pub fn encode(payload: &IntentPayload) -> Result<Vec<u8>, PayloadError> {
    bincode::serde::encode_to_vec(payload, bincode_cfg())
        .map_err(|e| PayloadError::Encode(e.to_string()))
}

/// Decode bytes back into a typed payload.
pub fn decode(bytes: &[u8]) -> Result<IntentPayload, PayloadError> {
    let (payload, _consumed) = bincode::serde::decode_from_slice(bytes, bincode_cfg())
        .map_err(|e| PayloadError::Decode(e.to_string()))?;
    Ok(payload)
}

/// Append a typed payload to the intent log. Returns the LSN assigned.
/// Callers that need durability before continuing must follow with
/// `log.sync()` — leaving the sync decision to the caller lets a batch
/// path flush once for many appends.
pub fn append(log: &mut IntentLog, payload: &IntentPayload) -> Result<Lsn, PayloadError> {
    let bytes = encode(payload)?;
    Ok(log.append(&bytes)?)
}

/// Decode the inner payload of a raw `IntentRecord`. The `Lsn` is carried
/// over unchanged in the returned tuple so the caller can build their own
/// "(lsn, typed payload)" pair without a second decode.
pub fn decode_record(record: &IntentRecord) -> Result<(Lsn, IntentPayload), PayloadError> {
    let payload = decode(&record.payload)?;
    Ok((record.lsn, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("veld-intent-payload-{name}-{pid}-{stamp}.log"));
        p
    }

    #[test]
    fn remember_round_trip() {
        let original = IntentPayload::Remember {
            user_id: "alice".to_string(),
            memory_id: "mem-42".to_string(),
            memory_bincode: vec![0xde, 0xad, 0xbe, 0xef],
        };
        let bytes = encode(&original).unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn forget_round_trip() {
        let original = IntentPayload::Forget {
            user_id: "bob".to_string(),
            memory_id: "mem-99".to_string(),
        };
        let bytes = encode(&original).unwrap();
        assert_eq!(decode(&bytes).unwrap(), original);
    }

    #[test]
    fn anchor_round_trip() {
        let original = IntentPayload::Anchor {
            user_id: "carol".to_string(),
            memory_id: "mem-7".to_string(),
            importance: 0.83,
        };
        let bytes = encode(&original).unwrap();
        assert_eq!(decode(&bytes).unwrap(), original);
    }

    #[test]
    fn update_round_trip() {
        let original = IntentPayload::Update {
            user_id: "dan".to_string(),
            memory_id: "mem-11".to_string(),
            memory_bincode: vec![1, 2, 3, 4, 5],
        };
        let bytes = encode(&original).unwrap();
        assert_eq!(decode(&bytes).unwrap(), original);
    }

    #[test]
    fn accessors_return_correct_ids() {
        let p = IntentPayload::Remember {
            user_id: "alice".to_string(),
            memory_id: "mem-1".to_string(),
            memory_bincode: vec![],
        };
        assert_eq!(p.user_id(), "alice");
        assert_eq!(p.memory_id(), "mem-1");
    }

    #[test]
    fn append_and_iterate_through_log() {
        let path = tmp_path("append");
        {
            let mut log = IntentLog::open(&path).unwrap();
            append(
                &mut log,
                &IntentPayload::Remember {
                    user_id: "u".into(),
                    memory_id: "m-0".into(),
                    memory_bincode: b"first".to_vec(),
                },
            )
            .unwrap();
            append(
                &mut log,
                &IntentPayload::Forget {
                    user_id: "u".into(),
                    memory_id: "m-1".into(),
                },
            )
            .unwrap();
            append(
                &mut log,
                &IntentPayload::Anchor {
                    user_id: "u".into(),
                    memory_id: "m-0".into(),
                    importance: 0.95,
                },
            )
            .unwrap();
            log.sync().unwrap();
        }

        let log = IntentLog::open(&path).unwrap();
        let records: Vec<(Lsn, IntentPayload)> = log
            .iter()
            .unwrap()
            .map(|r| decode_record(&r.unwrap()).unwrap())
            .collect();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, Lsn(0));
        match &records[0].1 {
            IntentPayload::Remember { memory_id, memory_bincode, .. } => {
                assert_eq!(memory_id, "m-0");
                assert_eq!(memory_bincode, b"first");
            }
            other => panic!("expected Remember, got {other:?}"),
        }
        match &records[1].1 {
            IntentPayload::Forget { memory_id, .. } => assert_eq!(memory_id, "m-1"),
            other => panic!("expected Forget, got {other:?}"),
        }
        match &records[2].1 {
            IntentPayload::Anchor { importance, .. } => {
                assert!((importance - 0.95).abs() < 1e-6);
            }
            other => panic!("expected Anchor, got {other:?}"),
        }

        let _ = std::fs::remove_file(&path);
    }
}
