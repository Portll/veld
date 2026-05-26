//! Fixed-size intent-log file header.
//!
//! Versioning the intent log means stamping a fixed, recognisable
//! preamble at byte 0 of every new file. The header carries:
//!
//! - a *magic* byte sequence so a new reader can tell a versioned log
//!   from a legacy (unversioned) one with one cheap read at offset 0;
//! - a `format_version` integer that gates the on-disk frame layout
//!   used in the rest of the file. Today the only known version is
//!   [`CURRENT_FORMAT_VERSION`] = `1`. A future breaking change to the
//!   frame shape bumps this number;
//! - a `created_unix_secs` timestamp purely for diagnostics — it costs
//!   nothing and a future operator can use it to correlate a log with
//!   the run that produced it;
//! - 12 reserved bytes, zeroed today. Reserved space lets a forwards-
//!   compatible *additive* change to the header (e.g. a tenant id) land
//!   without bumping `format_version`. Bytes added there must be parsed
//!   as zero by current readers, which they already are: today's parser
//!   ignores their content;
//! - a CRC32 over every preceding byte, so a torn write *during header
//!   creation* (process killed between the header write and any frame
//!   write) is detected as a corrupt header rather than silently
//!   accepted as a valid empty log.
//!
//! ## Wire layout
//!
//! ```text
//!   ┌────────────┬─────────────┬─────────────────┬─────────────┬──────────┐
//!   │ magic 8 B  │ version 4 B │ created 8 B     │ reserved 12 │ crc 4 B  │
//!   │ "VELDIL01" │   u32 LE    │   u64 LE secs   │   zero      │  u32 LE  │
//!   └────────────┴─────────────┴─────────────────┴─────────────┴──────────┘
//!     offset 0       8             12              20            32
//! ```
//!
//! Total on-disk size: [`HEADER_BYTES`] = 36. The CRC covers the first
//! 32 bytes (everything except the CRC itself).
//!
//! ## Why the magic literal
//!
//! `VELDIL01` = "Veld Intent Log, format 01". It is *not* the only
//! source of truth for the format — that's `format_version` — but it
//! lets a reader detect a file that has never been versioned (no magic
//! at offset 0) without having to interpret arbitrary leading bytes as
//! a frame header and risk misparsing.

use std::io::{self, Read};

/// Magic byte sequence that opens every versioned intent log file.
///
/// "VELDIL01" — Veld Intent Log, format-line tag 01. The trailing `01`
/// is a *brand* identifier, not the schema version; the actual schema
/// version follows in the four bytes after this. Bumping `format_version`
/// does *not* require changing this literal — the magic identifies the
/// file as "an intent log this codebase knows about", and the version
/// integer says which layout the frames after the header use.
pub const MAGIC: &[u8; 8] = b"VELDIL01";

/// Current intent-log format version. Bump when the frame layout in
/// `mod.rs` changes in a way that older readers cannot interpret.
///
/// Adding a new typed `IntentPayload` variant does NOT require a bump —
/// that is what `#[non_exhaustive]` + per-record `schema_version` are
/// for. The version here is the *frame* shape, not the *payload* shape.
pub const CURRENT_FORMAT_VERSION: u32 = 1;

/// Size of the full header on disk, including the trailing CRC.
pub const HEADER_BYTES: usize = 36;

/// Size of the portion of the header covered by the trailing CRC —
/// i.e. everything except the CRC itself.
const HEADER_PRECRC_BYTES: usize = HEADER_BYTES - 4;

/// Reserved-byte span. Filled with zero on write; readers ignore the
/// contents but the CRC covers them so a torn write is detected.
const RESERVED_BYTES: usize = 12;

/// Errors raised when reading or validating the intent log header.
#[derive(Debug, thiserror::Error)]
pub enum HeaderError {
    /// I/O failed during a header read or write.
    #[error("I/O error reading header: {0}")]
    Io(#[from] io::Error),
    /// The file is non-empty but the first bytes are not the magic. The
    /// caller is expected to treat this as "legacy unversioned log" and
    /// optionally invoke
    /// [`crate::intent_log::IntentLog::upgrade_unversioned_to_v1`] to
    /// stamp a header onto it.
    #[error("not a versioned intent log: magic mismatch (found {found:?})")]
    MagicMismatch { found: [u8; 8] },
    /// The header was readable but its `format_version` is not one this
    /// binary supports. Compare against [`CURRENT_FORMAT_VERSION`].
    #[error("unknown intent log format_version: {version} (this binary supports {supported})")]
    UnknownFormatVersion { version: u32, supported: u32 },
    /// The header's CRC did not match. Either the header was torn mid-
    /// write or the file was tampered with.
    #[error("intent log header CRC mismatch: stored={stored:#x} computed={computed:#x}")]
    CrcMismatch { stored: u32, computed: u32 },
    /// A header was *expected* (caller passed in a versioned file) but
    /// the file is too short to contain one.
    #[error("intent log file too short for header: have {have} bytes, need {need}")]
    TooShort { have: u64, need: u64 },
}

/// Parsed intent-log file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntentLogHeader {
    pub format_version: u32,
    pub created_unix_secs: u64,
}

impl IntentLogHeader {
    /// A fresh header stamped with the current format version and a
    /// wall-clock timestamp. `created_unix_secs` is best-effort —
    /// callers running before `UNIX_EPOCH` (clock skew) get `0`, which
    /// the parser treats as "unknown" without erroring.
    pub fn new_current() -> Self {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            format_version: CURRENT_FORMAT_VERSION,
            created_unix_secs: created,
        }
    }

    /// Serialise this header to its on-disk byte sequence. The returned
    /// `Vec` is exactly [`HEADER_BYTES`] bytes long and ready to be
    /// `write_all`-d at offset 0 of a new log file.
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut buf = [0u8; HEADER_BYTES];
        buf[0..8].copy_from_slice(MAGIC);
        buf[8..12].copy_from_slice(&self.format_version.to_le_bytes());
        buf[12..20].copy_from_slice(&self.created_unix_secs.to_le_bytes());
        // buf[20..32] stays zero — reserved.
        let crc = crc32fast::hash(&buf[..HEADER_PRECRC_BYTES]);
        buf[HEADER_PRECRC_BYTES..HEADER_BYTES].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Parse a header from `bytes`. `bytes.len()` must be at least
    /// [`HEADER_BYTES`]. Validates the magic, the CRC, and the version
    /// against [`CURRENT_FORMAT_VERSION`].
    pub fn parse(bytes: &[u8]) -> Result<Self, HeaderError> {
        if bytes.len() < HEADER_BYTES {
            return Err(HeaderError::TooShort {
                have: bytes.len() as u64,
                need: HEADER_BYTES as u64,
            });
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if &magic != MAGIC {
            return Err(HeaderError::MagicMismatch { found: magic });
        }
        let stored_crc = u32::from_le_bytes(bytes[HEADER_PRECRC_BYTES..HEADER_BYTES].try_into().unwrap());
        let computed_crc = crc32fast::hash(&bytes[..HEADER_PRECRC_BYTES]);
        if stored_crc != computed_crc {
            return Err(HeaderError::CrcMismatch {
                stored: stored_crc,
                computed: computed_crc,
            });
        }
        let format_version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if format_version != CURRENT_FORMAT_VERSION {
            return Err(HeaderError::UnknownFormatVersion {
                version: format_version,
                supported: CURRENT_FORMAT_VERSION,
            });
        }
        let created_unix_secs = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        // bytes[20..32] are reserved-and-zero. We do not enforce that
        // they are literally zero — future minor changes will write
        // additive fields there, and existing parsers must not refuse
        // those files. The CRC keeps the bytes honest either way.
        let _ = RESERVED_BYTES;
        Ok(Self {
            format_version,
            created_unix_secs,
        })
    }

    /// Read a header from the *start* of a freshly-opened file handle.
    /// Convenience over `parse` for the open path.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, HeaderError> {
        let mut buf = [0u8; HEADER_BYTES];
        match read_exact(reader, &mut buf) {
            Ok(()) => Self::parse(&buf),
            Err(e) => Err(HeaderError::Io(e)),
        }
    }
}

/// Drop-in `read_exact` that converts a short read into an explicit
/// `UnexpectedEof` — the std impl does the same, but spelling it out
/// keeps the error message stable for the upgrade path.
fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<()> {
    let mut read = 0;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "intent log file too short to contain a header",
                ));
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_bytes() {
        let h = IntentLogHeader {
            format_version: CURRENT_FORMAT_VERSION,
            created_unix_secs: 1_700_000_000,
        };
        let bytes = h.to_bytes();
        let back = IntentLogHeader::parse(&bytes).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn fresh_header_uses_current_version() {
        let h = IntentLogHeader::new_current();
        assert_eq!(h.format_version, CURRENT_FORMAT_VERSION);
    }

    #[test]
    fn magic_at_offset_zero() {
        let h = IntentLogHeader::new_current();
        let bytes = h.to_bytes();
        assert_eq!(&bytes[0..8], MAGIC);
    }

    #[test]
    fn total_header_length_is_36_bytes() {
        // Spec said "Total: 32 bytes" but the field arithmetic
        // (8 + 4 + 8 + 12 + 4) is 36. The 32 in the spec is the pre-CRC
        // payload length; the CRC adds 4 bytes after that. We pin both
        // here so a future refactor does not silently change either.
        assert_eq!(HEADER_BYTES, 36);
        assert_eq!(HEADER_PRECRC_BYTES, 32);
    }

    #[test]
    fn magic_mismatch_returns_legacy_signal() {
        let mut bytes = [0u8; HEADER_BYTES];
        // Garbage where the magic should be.
        bytes[..8].copy_from_slice(b"NOTAMAGI");
        let err = IntentLogHeader::parse(&bytes).unwrap_err();
        match err {
            HeaderError::MagicMismatch { found } => {
                assert_eq!(&found, b"NOTAMAGI");
            }
            other => panic!("expected MagicMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unknown_future_version_rejected() {
        let h = IntentLogHeader {
            format_version: 999,
            created_unix_secs: 0,
        };
        // Hand-build a header with version=999 but a correct CRC.
        let mut bytes = [0u8; HEADER_BYTES];
        bytes[0..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&h.format_version.to_le_bytes());
        bytes[12..20].copy_from_slice(&h.created_unix_secs.to_le_bytes());
        let crc = crc32fast::hash(&bytes[..HEADER_PRECRC_BYTES]);
        bytes[HEADER_PRECRC_BYTES..HEADER_BYTES].copy_from_slice(&crc.to_le_bytes());

        let err = IntentLogHeader::parse(&bytes).unwrap_err();
        match err {
            HeaderError::UnknownFormatVersion { version, supported } => {
                assert_eq!(version, 999);
                assert_eq!(supported, CURRENT_FORMAT_VERSION);
            }
            other => panic!("expected UnknownFormatVersion, got {other:?}"),
        }
    }

    #[test]
    fn crc_mismatch_detected() {
        let h = IntentLogHeader::new_current();
        let mut bytes = h.to_bytes();
        // Flip a bit in the version field; CRC will no longer match.
        bytes[8] ^= 0xff;
        let err = IntentLogHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, HeaderError::CrcMismatch { .. }));
    }

    #[test]
    fn too_short_buffer_rejected() {
        let bytes = vec![0u8; HEADER_BYTES - 1];
        let err = IntentLogHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, HeaderError::TooShort { .. }));
    }

    #[test]
    fn reserved_bytes_are_zero_on_fresh_header() {
        let h = IntentLogHeader::new_current();
        let bytes = h.to_bytes();
        assert!(bytes[20..32].iter().all(|b| *b == 0));
    }
}
