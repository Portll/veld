//! Per-payload migration registry.
//!
//! The intent log stamps every typed payload with a `schema_version`
//! (see [`super::payload::IntentPayload`]). When a future change to the
//! payload shape lands — a new required field, a renamed enum variant,
//! a type widening — the change ships with:
//!
//!   1. A bump to [`super::payload::CURRENT_PAYLOAD_SCHEMA_VERSION`].
//!   2. A new arm in [`migrate_payload`] that turns the older byte
//!      shape into the current one (or signals "cannot upgrade — operator
//!      action required").
//!
//! Today there is nothing to migrate: `from == to` is the identity, and
//! every other combination is an error. Centralising this in one module
//! lets the project enumerate every shape transition in one place when
//! the time comes, rather than scattering ad-hoc `if version == ...`
//! branches across the codebase.
//!
//! ## Why a registry, not "decode + re-encode"
//!
//! A naive scheme is to "always decode with the current code, fill in
//! defaults". That is exactly what `#[serde(default)]` already buys us
//! for *additive* changes. The migration registry exists for the cases
//! that `#[serde(default)]` cannot handle:
//!
//! - A field's *meaning* changed (e.g. a flag whose semantics inverted).
//! - A variant was renamed or replaced.
//! - A type was widened (`u16` → `u32`) and the bincode wire shape moved
//!   accordingly.
//! - A bincode config change (endianness, fixint vs varint) shifted the
//!   byte layout. See the open question in the module-level docs.
//!
//! In each of these cases the older bytes are no longer a valid input
//! to the current `bincode::serde::decode`, and a deliberate byte-level
//! rewrite is required before decoding.

use std::fmt;

/// Errors raised by the migration registry.
#[derive(Debug)]
pub enum MigrationError {
    /// We do not know how to migrate from `from` to `to`. Either the
    /// `from` version was retired (forwards-incompatible — operator must
    /// roll forward on a previous binary first) or the `to` version is
    /// in the future (the file was written by a newer Veld than this
    /// binary knows about).
    UnsupportedMigration { from: u16, to: u16 },
    /// The migration was found but the underlying bytes failed to be
    /// rewritten. Carries a human-readable detail for logs.
    Failed { from: u16, to: u16, detail: String },
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MigrationError::UnsupportedMigration { from, to } => write!(
                f,
                "no migration path from payload schema_version {from} to {to}"
            ),
            MigrationError::Failed { from, to, detail } => write!(
                f,
                "migration from payload schema_version {from} to {to} failed: {detail}"
            ),
        }
    }
}

impl std::error::Error for MigrationError {}

/// Rewrite a payload's bytes from one schema version to another.
///
/// `from == to` is always the identity function (returns a copy of the
/// input). Anything else is an [`MigrationError::UnsupportedMigration`]
/// today; future Veld versions add arms here as breaking payload
/// changes ship.
///
/// The function operates on raw bytes deliberately — by the time
/// migration runs, the *current* decoder has already failed (or the
/// caller wouldn't be calling), so we cannot rely on the type system.
/// Each future migration writes a tiny ad-hoc decoder for its `from`
/// shape and re-encodes into the `to` shape with the current bincode
/// config.
pub fn migrate_payload(from: u16, to: u16, bytes: &[u8]) -> Result<Vec<u8>, MigrationError> {
    if from == to {
        return Ok(bytes.to_vec());
    }
    // No real migrations exist yet. The first breaking change will land
    // a new `match (from, to)` arm here.
    Err(MigrationError::UnsupportedMigration { from, to })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_for_equal_versions() {
        let bytes = vec![0xde, 0xad, 0xbe, 0xef];
        let migrated = migrate_payload(1, 1, &bytes).unwrap();
        assert_eq!(migrated, bytes);
    }

    #[test]
    fn identity_for_zero_version() {
        let bytes = vec![];
        let migrated = migrate_payload(0, 0, &bytes).unwrap();
        assert_eq!(migrated, bytes);
    }

    #[test]
    fn unknown_migration_returns_structured_error() {
        let err = migrate_payload(0, 1, &[1, 2, 3]).unwrap_err();
        match err {
            MigrationError::UnsupportedMigration { from, to } => {
                assert_eq!(from, 0);
                assert_eq!(to, 1);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn migration_error_display_is_informative() {
        let err = MigrationError::UnsupportedMigration { from: 1, to: 2 };
        let s = format!("{err}");
        assert!(s.contains("1"));
        assert!(s.contains("2"));
        assert!(s.contains("schema_version"));
    }
}
