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
//! changing a type) requires both a `format_version` bump on the log
//! header (see [`super::header::IntentLogHeader`]) AND a per-record
//! `schema_version` bump, with a migration arm registered in
//! [`super::migrations::migrate_payload`].
//!
//! Every variant carries a `schema_version: Option<u16>`. A `None` value
//! is interpreted as version 0 (the pre-versioning shape this code was
//! born with). A `Some(v)` value that this binary does not recognise
//! returns a [`PayloadError::UnknownSchemaVersion`] — the caller is
//! expected to run the bytes through `migrate_payload` before trying to
//! decode again.

use serde::{Deserialize, Serialize};

use super::migrations::MigrationError;
use super::{IntentLog, IntentLogError, IntentRecord, Lsn};

/// Schema version assigned to a payload variant the *current* binary
/// produces. Encoded as `Option<u16>` on every variant; `None` and
/// `Some(0)` are wire-equivalent and mean "the original shape, before
/// per-record versioning existed".
///
/// Bumping this is a deliberate operator action: every existing
/// `IntentPayload` decode site continues to work because `Option`
/// defaults to `None`, but `migrate_payload` must learn the new arm
/// before the bump lands.
pub const CURRENT_PAYLOAD_SCHEMA_VERSION: u16 = 1;

/// Set of schema versions this binary can decode without migration.
/// Anything outside this set returns an [`PayloadError::UnknownSchemaVersion`]
/// from [`decode`]. The implicit `None` / `Some(0)` legacy version is
/// always accepted regardless of what is in this list.
pub const KNOWN_PAYLOAD_SCHEMA_VERSIONS: &[u16] = &[0, 1];

/// Resolve the effective version for an `Option<u16> schema_version`
/// field — `None` is the legacy pre-versioning shape (== 0).
fn effective_version(v: Option<u16>) -> u16 {
    v.unwrap_or(0)
}

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
        /// Schema version of the variant's wire shape. `None` is the
        /// pre-versioning shape (== 0). See module-level docs.
        #[serde(default)]
        schema_version: Option<u16>,
    },
    /// An existing memory was deleted.
    Forget {
        user_id: String,
        memory_id: String,
        #[serde(default)]
        schema_version: Option<u16>,
    },
    /// An existing memory was edited in place. Same opaque-bytes pattern
    /// as `Remember`.
    Update {
        user_id: String,
        memory_id: String,
        memory_bincode: Vec<u8>,
        #[serde(default)]
        schema_version: Option<u16>,
    },
    /// An existing memory was anchored (decay-resistant boost).
    Anchor {
        user_id: String,
        memory_id: String,
        importance: f32,
        #[serde(default)]
        schema_version: Option<u16>,
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

    /// Schema version of this variant's wire shape. `None` is the
    /// pre-versioning legacy shape; treat it as `0` for comparisons.
    pub fn schema_version(&self) -> Option<u16> {
        match self {
            IntentPayload::Remember { schema_version, .. } => *schema_version,
            IntentPayload::Forget { schema_version, .. } => *schema_version,
            IntentPayload::Update { schema_version, .. } => *schema_version,
            IntentPayload::Anchor { schema_version, .. } => *schema_version,
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
    /// The payload decoded into a known variant but carries a
    /// `schema_version` this binary does not recognise. The caller is
    /// expected to either upgrade the binary or pipe the bytes through
    /// [`super::migrations::migrate_payload`] before re-decoding.
    #[error(
        "unknown payload schema_version: {found} (this binary knows {known:?})"
    )]
    UnknownSchemaVersion {
        found: u16,
        known: &'static [u16],
    },
    /// A migration step failed while decoding an older payload shape.
    /// Wraps [`MigrationError`] verbatim so the caller can match on it.
    #[error("payload migration failed: {0}")]
    Migration(#[from] MigrationError),
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
///
/// After bincode produces the typed variant, this function checks the
/// `schema_version` field against [`KNOWN_PAYLOAD_SCHEMA_VERSIONS`]. If
/// the version is not in the set, returns
/// [`PayloadError::UnknownSchemaVersion`] — the caller is expected to
/// run the raw bytes through [`super::migrations::migrate_payload`] and
/// retry, or to surface the error to the operator if no migration path
/// exists yet.
pub fn decode(bytes: &[u8]) -> Result<IntentPayload, PayloadError> {
    let (payload, _consumed): (IntentPayload, usize) =
        bincode::serde::decode_from_slice(bytes, bincode_cfg())
            .map_err(|e| PayloadError::Decode(e.to_string()))?;
    let effective = effective_version(payload.schema_version());
    if !KNOWN_PAYLOAD_SCHEMA_VERSIONS.contains(&effective) {
        return Err(PayloadError::UnknownSchemaVersion {
            found: effective,
            known: KNOWN_PAYLOAD_SCHEMA_VERSIONS,
        });
    }
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
    fn remember_round_trip_with_schema_version_absent() {
        // `schema_version: None` is the wire-equivalent of the legacy
        // pre-versioning shape. Round-trip must reconstruct it identically.
        let original = IntentPayload::Remember {
            user_id: "alice".to_string(),
            memory_id: "mem-42".to_string(),
            memory_bincode: vec![0xde, 0xad, 0xbe, 0xef],
            schema_version: None,
        };
        let bytes = encode(&original).unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn remember_round_trip_with_schema_version_present() {
        // Caller asserts schema version 1; round-trip preserves that
        // assertion and decode accepts it (1 is in the known set).
        let original = IntentPayload::Remember {
            user_id: "alice".to_string(),
            memory_id: "mem-42".to_string(),
            memory_bincode: vec![0xde, 0xad, 0xbe, 0xef],
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        let bytes = encode(&original).unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(original, restored);
        assert_eq!(restored.schema_version(), Some(CURRENT_PAYLOAD_SCHEMA_VERSION));
    }

    #[test]
    fn forget_round_trip() {
        let original = IntentPayload::Forget {
            user_id: "bob".to_string(),
            memory_id: "mem-99".to_string(),
            schema_version: None,
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
            schema_version: None,
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
            schema_version: None,
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
            schema_version: None,
        };
        assert_eq!(p.user_id(), "alice");
        assert_eq!(p.memory_id(), "mem-1");
        assert_eq!(p.schema_version(), None);
    }

    #[test]
    fn unknown_schema_version_returns_structured_error() {
        // Construct a payload that *claims* schema_version 999, then
        // round-trip it through encode/decode. The decoder must surface
        // a structured UnknownSchemaVersion error rather than silently
        // accept the bytes.
        let claimed_future = IntentPayload::Anchor {
            user_id: "u".to_string(),
            memory_id: "m".to_string(),
            importance: 0.5,
            schema_version: Some(999),
        };
        let bytes = encode(&claimed_future).unwrap();
        let err = decode(&bytes).unwrap_err();
        match err {
            PayloadError::UnknownSchemaVersion { found, known } => {
                assert_eq!(found, 999);
                assert_eq!(known, KNOWN_PAYLOAD_SCHEMA_VERSIONS);
            }
            other => panic!("expected UnknownSchemaVersion, got {other:?}"),
        }
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
                    schema_version: None,
                },
            )
            .unwrap();
            append(
                &mut log,
                &IntentPayload::Forget {
                    user_id: "u".into(),
                    memory_id: "m-1".into(),
                    schema_version: None,
                },
            )
            .unwrap();
            append(
                &mut log,
                &IntentPayload::Anchor {
                    user_id: "u".into(),
                    memory_id: "m-0".into(),
                    importance: 0.95,
                    schema_version: None,
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
