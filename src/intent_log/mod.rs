//! Durable, checksummed append-only intent log (W5 scaffold).
//!
//! ## Why
//!
//! Veld treats RocksDB as the source of truth and Vamana / BM25 / SQLite as
//! rebuildable projections of that truth. Today the sync between truth and
//! projections is best-effort: a write may land in RocksDB but fail to reach
//! the vector index, and there's no replayable record of what happened.
//!
//! The intent log replaces that ad-hoc sync. Every state-changing operation
//! appends an `IntentRecord` to this log *before* it is applied to any store.
//! Projections checkpoint how far they have applied, and on restart they
//! replay from their checkpoint to the head of the log. A torn write or
//! corrupted frame at the tail is detected by CRC mismatch and the log can
//! be truncated to the last good record without losing earlier history.
//!
//! W4 (Postgres / Supabase) depends on this — Postgres writes are idempotent
//! when keyed by LSN, so a replay re-applies them safely.
//!
//! ## File layout
//!
//! Every file starts with a fixed-size [`IntentLogHeader`] (36 bytes)
//! carrying a magic literal, a `format_version`, a creation timestamp,
//! a reserved span, and a header CRC. The header lets future versions
//! detect breaking changes to the frame layout below without having to
//! guess from the bytes. See [`header`] for the wire shape.
//!
//! Legacy logs written before the header existed are auto-detected (no
//! magic at offset 0) and read in "legacy unversioned" mode. They can
//! be upgraded in place to v1 via [`IntentLog::upgrade_unversioned_to_v1`].
//!
//! ## Frame format
//!
//! After the header, each record is stored as a frame:
//!
//! ```text
//!   ┌───────────┬───────────┬─────────────────┬───────────┐
//!   │ len: u32  │ lsn: u64  │ payload (len B) │ crc: u32  │
//!   │ (LE)      │ (LE)      │                 │ (LE)      │
//!   └───────────┴───────────┴─────────────────┴───────────┘
//!     4 bytes     8 bytes      `len` bytes       4 bytes
//! ```
//!
//! - `len` is the byte length of `payload` (NOT including the lsn/crc/len
//!   themselves).
//! - `lsn` is the monotonically-increasing log sequence number assigned at
//!   append time.
//! - `crc` is `crc32fast` over the concatenation `[len][lsn][payload]`.
//!
//! A frame is valid iff:
//!   1. We can read all 4+8+len+4 bytes from the current file offset, AND
//!   2. The recomputed crc matches the stored crc.
//!
//! Any frame failing either check is the *first corrupt frame* — the reader
//! stops there, and the file can be truncated to the offset where that frame
//! started. Everything before that offset is durable and verifiable.
//!
//! ## What this module is NOT
//!
//! This is the storage primitive only. The `Projection` trait, replay driver,
//! and per-store checkpointing wire on top of this in follow-up work.
//! Keeping the primitive isolated lets it be tested exhaustively before any
//! projection depends on it.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::metrics::{
    Timer, INTENT_LOG_APPEND_DURATION, INTENT_LOG_APPEND_TOTAL,
    INTENT_LOG_DURABLE_END_OFFSET_BYTES, INTENT_LOG_NEXT_LSN, INTENT_LOG_SYNC_DURATION,
    INTENT_LOG_SYNC_TOTAL, INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL,
};

pub mod checkpoint_store;
pub mod header;
pub mod journal;
pub mod migrations;
pub mod payload;
pub mod projection;
pub use checkpoint_store::{CheckpointStore, CheckpointStoreError};
pub use header::{
    HeaderError, IntentLogHeader, CURRENT_FORMAT_VERSION, HEADER_BYTES, MAGIC,
};
pub use journal::{ApplyError, JournalError, JournaledWriter, TypedProjection, WriteOutcome};
pub use migrations::{migrate_payload, MigrationError};
pub use payload::{IntentPayload, PayloadError};
pub use projection::{replay, Projection, ReplayError};

/// A log sequence number — the position assigned to a record at append time.
///
/// LSNs are monotonically increasing within one log file. They are *not*
/// byte offsets; they are abstract counters so that a future segment-rolling
/// implementation can keep LSNs continuous across files.
///
/// Serde-derived so projections that snapshot their derived state to the
/// [`checkpoint_store::CheckpointStore`] side-data slot can include an
/// `Lsn` field in their snapshot type without a manual impl per call
/// site. Wire format is the inner `u64` (serde's transparent treatment
/// of a single-field tuple struct).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct Lsn(pub u64);

impl Lsn {
    pub const ZERO: Lsn = Lsn(0);

    pub fn next(self) -> Lsn {
        Lsn(self.0 + 1)
    }
}

/// Errors raised by the intent log.
#[derive(Debug, thiserror::Error)]
pub enum IntentLogError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("payload exceeds frame size limit ({size} bytes; limit {limit})")]
    PayloadTooLarge { size: usize, limit: usize },
    #[error("frame CRC mismatch at offset {offset}: stored={stored:#x} computed={computed:#x}")]
    CrcMismatch {
        offset: u64,
        stored: u32,
        computed: u32,
    },
    #[error("truncated frame at offset {offset} (expected {expected} bytes, got {actual})")]
    TruncatedFrame {
        offset: u64,
        expected: usize,
        actual: usize,
    },
    /// The file's header could not be validated. Carries the underlying
    /// [`HeaderError`] so the caller can distinguish "legacy file with
    /// no header" (which is recoverable via `upgrade_unversioned_to_v1`)
    /// from "future version we cannot interpret" (which is not).
    #[error("intent log header error: {0}")]
    Header(#[from] HeaderError),
    /// Operation refused because the log is being treated as a legacy
    /// (header-less) file. The caller must `upgrade_unversioned_to_v1`
    /// before performing the operation.
    #[error("operation requires a versioned intent log; this file is in legacy mode")]
    LegacyModeUnsupported,
}

/// Hard cap on a single payload's size. 16 MiB is far larger than any
/// realistic memory write yet small enough to keep a torn write from
/// requesting a wild allocation if `len` is interpreted from random bytes.
pub const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

const FRAME_HEADER_BYTES: usize = 4 + 8; // len + lsn
const FRAME_TRAILER_BYTES: usize = 4; // crc

/// Append-only durable intent log.
///
/// One file per log. Opens in append mode; reads scan from the start.
/// `append` writes are buffered; call `sync` to force them to disk.
pub struct IntentLog {
    path: PathBuf,
    writer: BufWriter<File>,
    next_lsn: Lsn,
    /// File offset (in bytes from start) of the byte *after* the last
    /// successfully-written frame. Used by `truncate_corrupt_tail` and to
    /// answer "how big is the durable portion of this file" without an
    /// extra fstat. Includes any header bytes — `durable_end_offset` is
    /// always relative to byte 0 of the file.
    durable_end_offset: u64,
    /// Versioning mode this log file is in. `Versioned { .. }` means the
    /// file starts with a valid [`IntentLogHeader`]; `Legacy` means it
    /// was created before headers existed and frames start at offset 0.
    mode: LogMode,
}

/// Versioning mode for an open intent log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogMode {
    /// File starts with a valid header at offset 0; frames start at
    /// [`HEADER_BYTES`].
    Versioned {
        format_version: u32,
        created_unix_secs: u64,
    },
    /// File has no header (was created before headers existed). Frames
    /// start at offset 0. Reads still work; writes still work; the only
    /// difference from `Versioned` is the lack of a forwards-compat
    /// guard, which `upgrade_unversioned_to_v1` adds in place.
    Legacy,
}

impl LogMode {
    /// Byte offset at which frames start, relative to byte 0 of the file.
    pub fn frame_region_start(&self) -> u64 {
        match self {
            LogMode::Versioned { .. } => HEADER_BYTES as u64,
            LogMode::Legacy => 0,
        }
    }

    /// `true` iff the file currently has a versioned header.
    pub fn is_versioned(&self) -> bool {
        matches!(self, LogMode::Versioned { .. })
    }
}

impl IntentLog {
    /// Open (create if missing) the log file at `path`. Scans the existing
    /// file to compute the next LSN and the durable-end offset. If the file
    /// ends in a partial or CRC-bad frame, **does not** auto-truncate — the
    /// caller decides whether to `truncate_corrupt_tail()` or to bail. This
    /// avoids silently destroying data when the file is unexpectedly short
    /// (e.g. disk full mid-write that's about to be retried).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, IntentLogError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Probe the file shape: new / versioned / legacy. The probe also
        // tells us the byte offset at which frames begin (after a header,
        // for versioned files; at 0, for legacy files).
        let mode = probe_mode(&path)?;

        // Now scan frames from the appropriate offset to discover next
        // LSN + durable_end. A corrupt or truncated tail is reported via
        // the returned offset — open() does not truncate.
        let (next_lsn, durable_end) = scan_for_tail(&path, mode.frame_region_start())?;

        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;

        // If the file was *created just now* (zero length, no header yet)
        // stamp a fresh header on it before any frame can be written. We
        // detect "new" by combining the probe result with the file's
        // current length: a brand-new file is in `Versioned` mode (probe
        // returned that for an empty file) but its on-disk length is 0.
        let actual_len = file.metadata()?.len();
        let mode = if mode.is_versioned() && actual_len == 0 {
            let header = IntentLogHeader::new_current();
            file.write_all(&header.to_bytes())?;
            file.sync_data()?;
            LogMode::Versioned {
                format_version: header.format_version,
                created_unix_secs: header.created_unix_secs,
            }
        } else {
            mode
        };

        // Recompute durable_end if we just stamped a header onto a
        // newly-created file — the header occupies the first HEADER_BYTES
        // bytes and no frames exist yet.
        let durable_end = if mode.is_versioned() && actual_len == 0 {
            HEADER_BYTES as u64
        } else {
            durable_end
        };

        // Publish the head-of-log gauges so dashboards see a value immediately
        // after process start, without waiting for the first append. Runs
        // after the new-file header stamp so durable_end reflects the
        // post-stamp size.
        INTENT_LOG_NEXT_LSN.set(next_lsn.0 as i64);
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(durable_end as i64);

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            next_lsn,
            durable_end_offset: durable_end,
            mode,
        })
    }

    /// Versioning mode this log was opened in. Callers that need the
    /// `format_version` integer for diagnostics or compatibility checks
    /// read it from here.
    pub fn mode(&self) -> LogMode {
        self.mode
    }

    /// Rewrite a legacy (header-less) log so it starts with a v1
    /// [`IntentLogHeader`]. Atomic via write-temp-then-rename: if the
    /// process dies mid-upgrade, the original file is untouched. On
    /// success the log is reopened in versioned mode and `self.mode()`
    /// reports `Versioned { format_version: 1, .. }`.
    ///
    /// Returns `Ok(false)` (no-op) when the log is already versioned.
    /// Returns an error if the upgrade I/O failed; the on-disk file is
    /// guaranteed to be either the original legacy bytes (if the rename
    /// did not happen) or the upgraded bytes (if it did). There is no
    /// in-between persisted state.
    pub fn upgrade_unversioned_to_v1(&mut self) -> Result<bool, IntentLogError> {
        if self.mode.is_versioned() {
            return Ok(false);
        }

        // Flush any buffered appends so the bytes on disk reflect every
        // record the caller has written. Upgrade then reads the file
        // contents and prepends a header in a side file.
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;

        let header = IntentLogHeader::new_current();
        let header_bytes = header.to_bytes();
        let tmp_path = self.path.with_extension("intentlog.upgrade.tmp");

        // Stream the live file into the tmp with the header prepended.
        // Doing it as a stream keeps RAM usage flat for huge logs.
        {
            let mut tmp = BufWriter::new(File::create(&tmp_path)?);
            tmp.write_all(&header_bytes)?;
            let mut src = File::open(&self.path)?;
            io::copy(&mut src, &mut tmp)?;
            tmp.flush()?;
            tmp.get_ref().sync_data()?;
        }

        // Drop the live append handle before renaming — Windows refuses
        // to rename over an open file with an exclusive append handle.
        // We replace `self.writer` with a handle on the tmp file in the
        // interim, then swing it back after the rename completes.
        let staged = OpenOptions::new().read(true).append(true).open(&tmp_path)?;
        self.writer = BufWriter::new(staged);
        std::fs::rename(&tmp_path, &self.path)?;
        let live = OpenOptions::new().read(true).append(true).open(&self.path)?;
        self.writer = BufWriter::new(live);

        // The frame region just shifted by HEADER_BYTES, so durable_end
        // moves with it.
        self.durable_end_offset += HEADER_BYTES as u64;
        self.mode = LogMode::Versioned {
            format_version: header.format_version,
            created_unix_secs: header.created_unix_secs,
        };
        Ok(true)
    }

    /// Append a record. Returns the LSN assigned to it. Writes go through a
    /// `BufWriter` — call `sync` for durability before returning to a caller
    /// that requires the record to survive a crash.
    #[tracing::instrument(
        level = "info",
        name = "intent_log.append",
        skip(self, payload),
        fields(
            lsn = self.next_lsn.0,
            payload_len = payload.len(),
        ),
    )]
    pub fn append(&mut self, payload: &[u8]) -> Result<Lsn, IntentLogError> {
        let _timer = Timer::new(INTENT_LOG_APPEND_DURATION.clone());

        if payload.len() > MAX_PAYLOAD_BYTES {
            INTENT_LOG_APPEND_TOTAL
                .with_label_values(&["payload_too_large"])
                .inc();
            let err = IntentLogError::PayloadTooLarge {
                size: payload.len(),
                limit: MAX_PAYLOAD_BYTES,
            };
            tracing::error!(
                error = %err,
                payload_len = payload.len(),
                limit = MAX_PAYLOAD_BYTES,
                "intent log append rejected: payload too large",
            );
            return Err(err);
        }

        let lsn = self.next_lsn;
        let len = payload.len() as u32;

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&len.to_le_bytes());
        hasher.update(&lsn.0.to_le_bytes());
        hasher.update(payload);
        let crc = hasher.finalize();

        if let Err(e) = (|| -> io::Result<()> {
            self.writer.write_all(&len.to_le_bytes())?;
            self.writer.write_all(&lsn.0.to_le_bytes())?;
            self.writer.write_all(payload)?;
            self.writer.write_all(&crc.to_le_bytes())?;
            Ok(())
        })() {
            INTENT_LOG_APPEND_TOTAL
                .with_label_values(&["io_error"])
                .inc();
            tracing::error!(
                error = %e,
                lsn = lsn.0,
                "intent log append failed: I/O error",
            );
            return Err(IntentLogError::Io(e));
        }

        self.next_lsn = lsn.next();
        self.durable_end_offset += (FRAME_HEADER_BYTES + payload.len() + FRAME_TRAILER_BYTES) as u64;

        INTENT_LOG_APPEND_TOTAL.with_label_values(&["ok"]).inc();
        INTENT_LOG_NEXT_LSN.set(self.next_lsn.0 as i64);
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(self.durable_end_offset as i64);

        Ok(lsn)
    }

    /// Flush the BufWriter and fsync the file. After this returns, every
    /// frame `append`-ed since the last `sync` is durable.
    #[tracing::instrument(
        level = "info",
        name = "intent_log.sync",
        skip(self),
        fields(
            next_lsn = self.next_lsn.0,
            durable_end_offset = self.durable_end_offset,
        ),
    )]
    pub fn sync(&mut self) -> Result<(), IntentLogError> {
        let _timer = Timer::new(INTENT_LOG_SYNC_DURATION.clone());
        if let Err(e) = self.writer.flush() {
            INTENT_LOG_SYNC_TOTAL.with_label_values(&["io_error"]).inc();
            tracing::error!(error = %e, "intent log sync failed: flush error");
            return Err(IntentLogError::Io(e));
        }
        if let Err(e) = self.writer.get_ref().sync_data() {
            INTENT_LOG_SYNC_TOTAL.with_label_values(&["io_error"]).inc();
            tracing::error!(error = %e, "intent log sync failed: fsync error");
            return Err(IntentLogError::Io(e));
        }
        INTENT_LOG_SYNC_TOTAL.with_label_values(&["ok"]).inc();
        Ok(())
    }

    /// Path of the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// LSN that will be assigned to the next successful `append`.
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn
    }

    /// Byte offset immediately after the last good frame seen at open time
    /// or written since. Useful for `truncate_corrupt_tail` callers.
    pub fn durable_end_offset(&self) -> u64 {
        self.durable_end_offset
    }

    /// Iterate every frame from the start of the frame region. Stops at
    /// the first corrupt or truncated frame; returns an `IntentLogError`
    /// for that frame so the caller can decide whether to truncate. The
    /// iterator is independent of the writer — it opens its own read
    /// handle. The header (if any) is skipped before iteration begins.
    pub fn iter(&self) -> Result<IntentLogIter, IntentLogError> {
        IntentLogIter::open(&self.path, self.mode.frame_region_start())
    }

    /// Truncate the file so its length equals `durable_end_offset`. Used to
    /// recover from a torn-tail crash where the trailing bytes are partial
    /// or CRC-bad. After this returns, `iter()` will read cleanly to EOF.
    ///
    /// This is a destructive operation — callers must be certain the tail
    /// is genuinely corrupt (i.e. they ran `iter()` first and got an error
    /// at the very last frame).
    pub fn truncate_corrupt_tail(&mut self) -> Result<u64, IntentLogError> {
        self.writer.flush()?;
        let target = self.durable_end_offset;

        // On Windows, an append-mode handle lacks FILE_WRITE_DATA so
        // `set_len` returns ERROR_ACCESS_DENIED. Use a transient
        // read+write handle to perform the truncate, then resume our
        // own append handle. Cross-platform — POSIX behaves identically.
        let trunc_handle = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        trunc_handle.set_len(target)?;
        trunc_handle.sync_data()?;
        drop(trunc_handle);

        // Reopen so BufWriter's seek matches the new EOF.
        let file = OpenOptions::new().read(true).append(true).open(&self.path)?;
        self.writer = BufWriter::new(file);
        INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL.inc();
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(target as i64);
        tracing::info!(
            target_offset = target,
            "intent log truncated corrupt tail",
        );
        Ok(target)
    }

    /// Truncate the file so the last surviving frame is the one with LSN
    /// `max_lsn` (inclusive). All frames after that frame are discarded.
    /// Used by the backup point-in-time-restore (PITR) path to roll the log
    /// back to a known position.
    ///
    /// Returns the new file length in bytes.
    ///
    /// Errors:
    /// - Returns [`IntentLogError::Io`] if the underlying file can't be
    ///   read or resized.
    /// - Returns [`IntentLogError::CrcMismatch`] / [`IntentLogError::TruncatedFrame`]
    ///   if a corrupt frame is encountered *before* `max_lsn`. In that
    ///   case the file is left untouched — PITR refuses to truncate
    ///   through corruption because that would silently lose data the
    ///   caller didn't ask to drop.
    /// - Returns [`IntentLogError::Io`] (NotFound) if `max_lsn` is greater
    ///   than every LSN actually present in the file.
    pub fn truncate_to_lsn(&mut self, max_lsn: Lsn) -> Result<u64, IntentLogError> {
        // Flush any buffered append so the on-disk image is what we scan.
        self.writer.flush()?;

        // Walk frames in a scoped block so the scanner's read handle is
        // dropped before we attempt the truncate. Windows refuses
        // `set_len` while another read handle is open against the same
        // file, even though POSIX is happy with it.
        let (last_good_offset, _highest_seen) = {
            let mut iter = IntentLogIter::open_for_scan(&self.path)?;
            let mut last_good_offset = 0u64;
            let mut found = false;
            let mut highest_seen: Option<Lsn> = None;
            loop {
                match iter.next() {
                    None => break,
                    Some(Ok(rec)) => {
                        highest_seen = Some(rec.lsn);
                        let frame_end = iter.current_offset();
                        if rec.lsn == max_lsn {
                            last_good_offset = frame_end;
                            found = true;
                            break;
                        }
                        if rec.lsn.0 > max_lsn.0 {
                            // Walked past target without finding an exact match.
                            // The caller's max_lsn is between frames, which is
                            // not allowed — LSNs are dense.
                            return Err(IntentLogError::Io(io::Error::new(
                                io::ErrorKind::NotFound,
                                format!(
                                    "max_lsn={} is not present in intent log (highest seen so far: {})",
                                    max_lsn.0, rec.lsn.0
                                ),
                            )));
                        }
                    }
                    Some(Err(e)) => return Err(e),
                }
            }
            if !found {
                return Err(IntentLogError::Io(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "max_lsn={} not found in intent log (highest LSN present: {})",
                        max_lsn.0,
                        highest_seen.map(|l| l.0 as i128).unwrap_or(-1),
                    ),
                )));
            }
            (last_good_offset, highest_seen)
            // `iter` dropped here, releasing its read handle.
        };

        // Drop our own append-mode writer before truncating: on Windows
        // a handle opened with append(true) lacks FILE_WRITE_DATA, so
        // calling `set_len` on it returns ERROR_ACCESS_DENIED. Reopen
        // with read+write so the truncate succeeds, then reattach the
        // append handle afterwards.
        let trunc_handle = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        trunc_handle.set_len(last_good_offset)?;
        trunc_handle.sync_data()?;
        drop(trunc_handle);

        // Reopen so BufWriter's append position matches the new EOF, and
        // refresh the in-memory tail bookkeeping.
        let file = OpenOptions::new().read(true).append(true).open(&self.path)?;
        self.writer = BufWriter::new(file);
        self.durable_end_offset = last_good_offset;
        self.next_lsn = max_lsn.next();
        Ok(last_good_offset)
    }
}

/// One frame returned by [`IntentLogIter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentRecord {
    pub lsn: Lsn,
    pub payload: Vec<u8>,
}

/// Forward-only iterator over the frames in an intent log file. Stops at
/// EOF or the first corrupt/truncated frame; in the latter case yields one
/// `Err(...)` and then `None`. The byte offset reported in errors lets the
/// caller line up `truncate_corrupt_tail`.
pub struct IntentLogIter {
    file: File,
    offset: u64,
    finished: bool,
}

impl IntentLogIter {
    /// Open a read-only iterator over the frames in `path`. This is
    /// exposed so external tooling (e.g. the backup engine) can scan a
    /// freshly-restored log file end-to-end without needing to construct
    /// a full `IntentLog`. Use `IntentLog::iter()` for the in-process
    /// case where you already have a log handle open.
    ///
    /// Auto-detects header presence: a versioned log skips past the
    /// 36-byte header; a legacy log starts at offset 0.
    pub fn open_for_scan(path: &Path) -> Result<Self, IntentLogError> {
        let mode = probe_mode(path)?;
        Self::open(path, mode.frame_region_start())
    }

    fn open(path: &Path, start_offset: u64) -> Result<Self, IntentLogError> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(start_offset))?;
        Ok(Self {
            file,
            offset: start_offset,
            finished: false,
        })
    }

    /// Byte offset of the next frame to be read. After `next()` returns
    /// `Err`, this still points at the *start* of the bad frame — so a
    /// caller can `IntentLog::truncate_corrupt_tail` to that offset.
    pub fn current_offset(&self) -> u64 {
        self.offset
    }
}

impl Iterator for IntentLogIter {
    type Item = Result<IntentRecord, IntentLogError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let frame_start = self.offset;

        // Read header (12 bytes). A short read here = EOF or torn tail.
        let mut header = [0u8; FRAME_HEADER_BYTES];
        match read_exact_or_eof(&mut self.file, &mut header) {
            Ok(true) => {} // full read
            Ok(false) => {
                // EOF exactly at frame boundary — clean end.
                self.finished = true;
                return None;
            }
            Err(ReadOutcome::ShortRead { read }) => {
                self.finished = true;
                return Some(Err(IntentLogError::TruncatedFrame {
                    offset: frame_start,
                    expected: FRAME_HEADER_BYTES,
                    actual: read,
                }));
            }
            Err(ReadOutcome::Io(e)) => {
                self.finished = true;
                return Some(Err(IntentLogError::Io(e)));
            }
        }

        let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let lsn = u64::from_le_bytes(header[4..12].try_into().unwrap());

        // Reject obviously-wild lengths to avoid huge allocations from
        // misinterpreted garbage bytes.
        if len > MAX_PAYLOAD_BYTES {
            self.finished = true;
            return Some(Err(IntentLogError::TruncatedFrame {
                offset: frame_start,
                expected: len,
                actual: 0,
            }));
        }

        let mut payload = vec![0u8; len];
        if !payload.is_empty() {
            match read_exact_or_eof(&mut self.file, &mut payload) {
                Ok(true) => {}
                Ok(false) => {
                    self.finished = true;
                    return Some(Err(IntentLogError::TruncatedFrame {
                        offset: frame_start,
                        expected: FRAME_HEADER_BYTES + len,
                        actual: FRAME_HEADER_BYTES,
                    }));
                }
                Err(ReadOutcome::ShortRead { read }) => {
                    self.finished = true;
                    return Some(Err(IntentLogError::TruncatedFrame {
                        offset: frame_start,
                        expected: FRAME_HEADER_BYTES + len,
                        actual: FRAME_HEADER_BYTES + read,
                    }));
                }
                Err(ReadOutcome::Io(e)) => {
                    self.finished = true;
                    return Some(Err(IntentLogError::Io(e)));
                }
            }
        }

        let mut crc_bytes = [0u8; FRAME_TRAILER_BYTES];
        match read_exact_or_eof(&mut self.file, &mut crc_bytes) {
            Ok(true) => {}
            Ok(false) => {
                self.finished = true;
                return Some(Err(IntentLogError::TruncatedFrame {
                    offset: frame_start,
                    expected: FRAME_HEADER_BYTES + len + FRAME_TRAILER_BYTES,
                    actual: FRAME_HEADER_BYTES + len,
                }));
            }
            Err(ReadOutcome::ShortRead { read }) => {
                self.finished = true;
                return Some(Err(IntentLogError::TruncatedFrame {
                    offset: frame_start,
                    expected: FRAME_HEADER_BYTES + len + FRAME_TRAILER_BYTES,
                    actual: FRAME_HEADER_BYTES + len + read,
                }));
            }
            Err(ReadOutcome::Io(e)) => {
                self.finished = true;
                return Some(Err(IntentLogError::Io(e)));
            }
        }

        let stored_crc = u32::from_le_bytes(crc_bytes);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header[0..4]);
        hasher.update(&header[4..12]);
        hasher.update(&payload);
        let computed_crc = hasher.finalize();

        if stored_crc != computed_crc {
            self.finished = true;
            return Some(Err(IntentLogError::CrcMismatch {
                offset: frame_start,
                stored: stored_crc,
                computed: computed_crc,
            }));
        }

        self.offset += (FRAME_HEADER_BYTES + len + FRAME_TRAILER_BYTES) as u64;

        Some(Ok(IntentRecord {
            lsn: Lsn(lsn),
            payload,
        }))
    }
}

/// Distinguishes clean EOF from "read started but didn't finish".
enum ReadOutcome {
    ShortRead { read: usize },
    Io(io::Error),
}

/// Returns `Ok(true)` if `buf` was fully filled; `Ok(false)` if EOF was hit
/// before any bytes were read (clean frame boundary); `Err(ShortRead)` if
/// some bytes were read before EOF (torn tail).
fn read_exact_or_eof(file: &mut File, buf: &mut [u8]) -> Result<bool, ReadOutcome> {
    let mut read = 0;
    while read < buf.len() {
        match file.read(&mut buf[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(ReadOutcome::Io(e)),
        }
    }
    if read == buf.len() {
        Ok(true)
    } else if read == 0 {
        Ok(false)
    } else {
        Err(ReadOutcome::ShortRead { read })
    }
}

/// Walk an existing log file (if any) starting at `frame_region_start`
/// (immediately past the header, for versioned files; 0 for legacy) to
/// compute the next LSN and the byte offset immediately after the last
/// valid frame. A corrupt or truncated tail is reported via the returned
/// offset — open() does not truncate.
fn scan_for_tail(path: &Path, frame_region_start: u64) -> Result<(Lsn, u64), IntentLogError> {
    if !path.exists() {
        // No file yet → frames will start at frame_region_start once a
        // header is written. Report that as the durable end so the
        // caller's bookkeeping is consistent.
        return Ok((Lsn::ZERO, frame_region_start));
    }
    let file_len = std::fs::metadata(path)?.len();
    if file_len <= frame_region_start {
        // File exists but has nothing after the header — clean empty log.
        return Ok((Lsn::ZERO, frame_region_start));
    }
    let mut iter = IntentLogIter::open(path, frame_region_start)?;
    let mut last_lsn = None;
    let mut last_good_offset = frame_region_start;
    loop {
        let off = iter.current_offset();
        match iter.next() {
            None => break,
            Some(Ok(rec)) => {
                last_lsn = Some(rec.lsn);
                last_good_offset = iter.current_offset();
            }
            Some(Err(_)) => {
                // Stop at first corruption. `last_good_offset` is the
                // truncate target; `next_lsn` continues from the last
                // *good* lsn we saw.
                last_good_offset = off;
                break;
            }
        }
    }
    let next_lsn = match last_lsn {
        Some(lsn) => lsn.next(),
        None => Lsn::ZERO,
    };
    Ok((next_lsn, last_good_offset))
}

/// Probe a log file's versioning mode.
///
/// - File missing or empty → `LogMode::Versioned { format_version: 1, .. }`
///   so the caller stamps a fresh header on it.
/// - First 8 bytes match [`MAGIC`] → parse the header and return its
///   contents.
/// - First 8 bytes don't match → `LogMode::Legacy` (the caller will read
///   the file's frames from offset 0 as before).
/// - Magic matched but the header was malformed (CRC bad, version
///   unknown) → forwarded as an error; this is a real corruption /
///   future-version condition that callers must surface, not silently
///   recover from.
fn probe_mode(path: &Path) -> Result<LogMode, IntentLogError> {
    if !path.exists() {
        // Fresh file — the open() caller will write a header.
        return Ok(LogMode::Versioned {
            format_version: CURRENT_FORMAT_VERSION,
            // Placeholder; the real timestamp is stamped at write time.
            created_unix_secs: 0,
        });
    }
    let len = std::fs::metadata(path)?.len();
    if len == 0 {
        // Touched but never written — treat as fresh.
        return Ok(LogMode::Versioned {
            format_version: CURRENT_FORMAT_VERSION,
            created_unix_secs: 0,
        });
    }

    let mut file = File::open(path)?;
    let mut magic_buf = [0u8; 8];
    let n = match read_n_or_short(&mut file, &mut magic_buf) {
        Ok(n) => n,
        Err(e) => return Err(IntentLogError::Io(e)),
    };

    // Too short to even hold the magic — treat as legacy (whoever owns
    // this file can take it from there).
    if n < 8 {
        return Ok(LogMode::Legacy);
    }
    if &magic_buf != MAGIC {
        return Ok(LogMode::Legacy);
    }

    // Magic matches — parse the rest of the header. The file MUST be
    // long enough to contain the whole header; if it isn't, that is a
    // torn-write-during-create scenario which we surface as a header
    // error rather than silently downgrading to legacy mode.
    if len < HEADER_BYTES as u64 {
        return Err(IntentLogError::Header(HeaderError::TooShort {
            have: len,
            need: HEADER_BYTES as u64,
        }));
    }
    file.seek(SeekFrom::Start(0))?;
    let header = IntentLogHeader::read_from(&mut file)?;
    Ok(LogMode::Versioned {
        format_version: header.format_version,
        created_unix_secs: header.created_unix_secs,
    })
}

/// `read` up to `buf.len()` bytes, returning the number actually read.
/// Used by the probe path which is fine with partial reads (a 3-byte
/// file is unambiguously not a versioned log).
fn read_n_or_short(file: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut read = 0;
    while read < buf.len() {
        match file.read(&mut buf[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(read)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_log_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("veld-intent-log-{name}-{pid}-{stamp}.log"));
        p
    }

    #[test]
    fn empty_file_opens_clean() {
        let path = tmp_log_path("empty");
        let log = IntentLog::open(&path).unwrap();
        assert_eq!(log.next_lsn(), Lsn::ZERO);
        // Brand-new logs are versioned; the file already holds a header
        // (HEADER_BYTES) before any frame lands, so the durable end is
        // the post-header offset, not literally zero.
        assert_eq!(log.durable_end_offset(), HEADER_BYTES as u64);
        assert!(log.mode().is_versioned());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trip_three_records() {
        let path = tmp_log_path("rt3");
        {
            let mut log = IntentLog::open(&path).unwrap();
            assert_eq!(log.append(b"hello").unwrap(), Lsn(0));
            assert_eq!(log.append(b"world").unwrap(), Lsn(1));
            assert_eq!(log.append(b"!").unwrap(), Lsn(2));
            log.sync().unwrap();
        }
        // Reopen and iterate.
        let log = IntentLog::open(&path).unwrap();
        assert_eq!(log.next_lsn(), Lsn(3));
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].lsn, Lsn(0));
        assert_eq!(records[0].payload, b"hello");
        assert_eq!(records[1].payload, b"world");
        assert_eq!(records[2].payload, b"!");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_payload_is_valid() {
        let path = tmp_log_path("zerolen");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(&[]).unwrap();
            log.sync().unwrap();
        }
        let log = IntentLog::open(&path).unwrap();
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].payload.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncated_tail_detected_and_truncatable() {
        let path = tmp_log_path("torn");
        // Write two good frames, sync.
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"first").unwrap();
            log.append(b"second").unwrap();
            log.sync().unwrap();
        }
        // Append a half-frame manually (just a header, no payload + crc) to
        // simulate a torn write.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[7u8, 0, 0, 0]).unwrap(); // len=7
            f.write_all(&[42u8, 0, 0, 0, 0, 0, 0, 0]).unwrap(); // lsn=42
            // missing payload + crc
            f.sync_data().unwrap();
        }

        // Reopen — should see two good records and a TruncatedFrame error.
        let mut log = IntentLog::open(&path).unwrap();
        let mut iter = log.iter().unwrap();
        let r0 = iter.next().unwrap().unwrap();
        assert_eq!(r0.payload, b"first");
        let r1 = iter.next().unwrap().unwrap();
        assert_eq!(r1.payload, b"second");
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, IntentLogError::TruncatedFrame { .. }));
        assert!(iter.next().is_none());

        // durable_end_offset should point to end of the second good frame.
        let good_end = log.durable_end_offset();
        let truncated_to = log.truncate_corrupt_tail().unwrap();
        assert_eq!(truncated_to, good_end);

        // Re-iterate clean — no error.
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 2);

        // Next append continues from lsn=2.
        let lsn = log.append(b"third").unwrap();
        assert_eq!(lsn, Lsn(2));
        log.sync().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn crc_corruption_detected() {
        let path = tmp_log_path("crc");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"untouched").unwrap();
            log.append(b"will-be-flipped").unwrap();
            log.sync().unwrap();
        }
        // Flip one byte inside the second payload.
        {
            let mut f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
            // Layout: [HEADER_BYTES][first frame: 4+8+9+4=25][second frame…].
            // Second frame starts at HEADER_BYTES + 25; the first byte of
            // its payload is at HEADER_BYTES + 25 + 12.
            let target = (HEADER_BYTES + 25 + 12) as u64;
            f.seek(SeekFrom::Start(target)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xff;
            f.seek(SeekFrom::Start(target)).unwrap();
            f.write_all(&b).unwrap();
            f.sync_data().unwrap();
        }
        let log = IntentLog::open(&path).unwrap();
        let mut iter = log.iter().unwrap();
        // First frame still good.
        let r0 = iter.next().unwrap().unwrap();
        assert_eq!(r0.payload, b"untouched");
        // Second frame fails CRC.
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, IntentLogError::CrcMismatch { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn payload_too_large_rejected() {
        let path = tmp_log_path("toobig");
        let mut log = IntentLog::open(&path).unwrap();
        let huge = vec![0u8; MAX_PAYLOAD_BYTES + 1];
        let err = log.append(&huge).unwrap_err();
        assert!(matches!(err, IntentLogError::PayloadTooLarge { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncate_to_lsn_keeps_inclusive_frame() {
        let path = tmp_log_path("trunc_lsn");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"frame-0").unwrap(); // lsn 0
            log.append(b"frame-1").unwrap(); // lsn 1
            log.append(b"frame-2").unwrap(); // lsn 2
            log.append(b"frame-3").unwrap(); // lsn 3
            log.sync().unwrap();
        }
        let mut log = IntentLog::open(&path).unwrap();
        let new_len = log.truncate_to_lsn(Lsn(1)).unwrap();
        // Verify only frames 0 and 1 remain, in order.
        let records: Vec<_> = log
            .iter()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].lsn, Lsn(0));
        assert_eq!(records[0].payload, b"frame-0");
        assert_eq!(records[1].lsn, Lsn(1));
        assert_eq!(records[1].payload, b"frame-1");
        // File length matches reported truncation.
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), new_len);
        // Next append continues from lsn 2.
        let lsn = log.append(b"after-pitr").unwrap();
        assert_eq!(lsn, Lsn(2));
        log.sync().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncate_to_lsn_missing_target_errors() {
        let path = tmp_log_path("trunc_lsn_missing");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"only-frame").unwrap();
            log.sync().unwrap();
        }
        let mut log = IntentLog::open(&path).unwrap();
        // max_lsn beyond what's present must error.
        let err = log.truncate_to_lsn(Lsn(99)).unwrap_err();
        assert!(matches!(err, IntentLogError::Io(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reopen_continues_lsn_sequence() {
        let path = tmp_log_path("reopen");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"a").unwrap();
            log.append(b"b").unwrap();
            log.sync().unwrap();
        }
        {
            let mut log = IntentLog::open(&path).unwrap();
            assert_eq!(log.next_lsn(), Lsn(2));
            assert_eq!(log.append(b"c").unwrap(), Lsn(2));
            log.sync().unwrap();
        }
        let log = IntentLog::open(&path).unwrap();
        let lsns: Vec<_> = log
            .iter()
            .unwrap()
            .map(|r| r.unwrap().lsn)
            .collect();
        assert_eq!(lsns, vec![Lsn(0), Lsn(1), Lsn(2)]);
        let _ = std::fs::remove_file(&path);
    }

    // ========================================================================
    // Versioning tests (W: format_version header + per-record schema_version)
    // ========================================================================

    /// Hand-roll a legacy (header-less) log by writing raw frames to a fresh
    /// file. Used to seed the legacy-mode and upgrade tests.
    fn write_legacy_log(path: &Path, payloads: &[&[u8]]) {
        let mut f = File::create(path).unwrap();
        let mut next_lsn = 0u64;
        for payload in payloads {
            let len = payload.len() as u32;
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&len.to_le_bytes());
            hasher.update(&next_lsn.to_le_bytes());
            hasher.update(payload);
            let crc = hasher.finalize();
            f.write_all(&len.to_le_bytes()).unwrap();
            f.write_all(&next_lsn.to_le_bytes()).unwrap();
            f.write_all(payload).unwrap();
            f.write_all(&crc.to_le_bytes()).unwrap();
            next_lsn += 1;
        }
        f.sync_data().unwrap();
    }

    #[test]
    fn new_log_writes_a_v1_header_at_offset_zero() {
        let path = tmp_log_path("v1_header");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"hello").unwrap();
            log.sync().unwrap();
            // Mode reports the freshly-stamped header.
            match log.mode() {
                LogMode::Versioned { format_version, .. } => {
                    assert_eq!(format_version, CURRENT_FORMAT_VERSION);
                }
                LogMode::Legacy => panic!("expected versioned, got legacy"),
            }
        }
        // The on-disk file starts with the MAGIC and the file's first
        // HEADER_BYTES round-trip through the header parser.
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= HEADER_BYTES);
        assert_eq!(&bytes[0..8], MAGIC);
        let parsed = IntentLogHeader::parse(&bytes[..HEADER_BYTES]).unwrap();
        assert_eq!(parsed.format_version, CURRENT_FORMAT_VERSION);

        // Reopening sees the existing header (not a fresh one) and
        // iterates only the frame, not the header.
        let log2 = IntentLog::open(&path).unwrap();
        let records: Vec<_> = log2.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload, b"hello");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn legacy_unversioned_log_opens_and_iterates() {
        // A header-less file written before this feature existed should
        // still be readable end-to-end. open() puts it in Legacy mode and
        // iter() walks the frames from offset 0.
        let path = tmp_log_path("legacy_open");
        write_legacy_log(&path, &[b"alpha", b"beta", b"gamma"]);

        let log = IntentLog::open(&path).unwrap();
        assert_eq!(log.mode(), LogMode::Legacy);
        assert_eq!(log.next_lsn(), Lsn(3));
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].payload, b"alpha");
        assert_eq!(records[2].payload, b"gamma");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn upgrade_legacy_to_v1_preserves_all_frames() {
        let path = tmp_log_path("legacy_upgrade");
        write_legacy_log(&path, &[b"one", b"two", b"three"]);

        let mut log = IntentLog::open(&path).unwrap();
        assert_eq!(log.mode(), LogMode::Legacy);
        let upgraded = log.upgrade_unversioned_to_v1().unwrap();
        assert!(upgraded);
        assert!(log.mode().is_versioned());

        // After upgrade, the file's first HEADER_BYTES are a valid header
        // and the frames live immediately after.
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= HEADER_BYTES);
        let _ = IntentLogHeader::parse(&bytes[..HEADER_BYTES]).unwrap();

        // Reopen via the v1 reader; every original record survives, in
        // order, with the same LSNs as before.
        let log2 = IntentLog::open(&path).unwrap();
        assert!(log2.mode().is_versioned());
        let records: Vec<_> = log2.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].lsn, Lsn(0));
        assert_eq!(records[0].payload, b"one");
        assert_eq!(records[1].payload, b"two");
        assert_eq!(records[2].payload, b"three");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn upgrade_is_idempotent_on_versioned_log() {
        let path = tmp_log_path("upgrade_idempotent");
        let mut log = IntentLog::open(&path).unwrap();
        log.append(b"x").unwrap();
        log.sync().unwrap();
        assert!(log.mode().is_versioned());
        // Calling upgrade on a file that already has a header is a no-op.
        let upgraded = log.upgrade_unversioned_to_v1().unwrap();
        assert!(!upgraded, "upgrade on versioned log should report no-op");
        // Subsequent appends still work and frames are intact.
        log.append(b"y").unwrap();
        log.sync().unwrap();
        let log2 = IntentLog::open(&path).unwrap();
        let records: Vec<_> = log2.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn future_format_version_is_rejected_with_clear_error() {
        // Craft a file whose magic and CRC are valid but whose
        // format_version is 999 — i.e. a log written by some imaginary
        // future Veld. The current reader must refuse to open it with a
        // structured error, not panic.
        let path = tmp_log_path("future_version");
        {
            let mut bytes = [0u8; HEADER_BYTES];
            bytes[0..8].copy_from_slice(MAGIC);
            bytes[8..12].copy_from_slice(&999u32.to_le_bytes());
            bytes[12..20].copy_from_slice(&0u64.to_le_bytes());
            // reserved bytes stay zero
            let crc = crc32fast::hash(&bytes[..HEADER_BYTES - 4]);
            bytes[HEADER_BYTES - 4..HEADER_BYTES].copy_from_slice(&crc.to_le_bytes());
            std::fs::write(&path, bytes).unwrap();
        }

        // IntentLog does not implement Debug, so `unwrap_err()` is off
        // the table — pattern-match the Result directly instead.
        match IntentLog::open(&path) {
            Err(IntentLogError::Header(HeaderError::UnknownFormatVersion {
                version,
                supported,
            })) => {
                assert_eq!(version, 999);
                assert_eq!(supported, CURRENT_FORMAT_VERSION);
            }
            Err(other) => panic!("expected UnknownFormatVersion, got {other:?}"),
            Ok(_) => panic!("expected open() to fail for a future format_version"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn header_crc_mismatch_is_rejected() {
        let path = tmp_log_path("hdr_crc_bad");
        // Build a valid header, then flip a byte and leave the CRC alone.
        let mut bytes = IntentLogHeader::new_current().to_bytes();
        bytes[8] ^= 0xff; // tamper with format_version
        std::fs::write(&path, bytes).unwrap();
        match IntentLog::open(&path) {
            Err(IntentLogError::Header(HeaderError::CrcMismatch { .. })) => {}
            Err(other) => panic!("expected CrcMismatch, got {other:?}"),
            Ok(_) => panic!("expected open() to fail for a header CRC mismatch"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn header_only_too_short_after_magic_is_rejected() {
        // First eight bytes are the magic, then the file ends. Must
        // surface as a structured "too short" header error, not silently
        // get reinterpreted as legacy mode.
        let path = tmp_log_path("hdr_too_short");
        std::fs::write(&path, MAGIC).unwrap();
        match IntentLog::open(&path) {
            Err(IntentLogError::Header(HeaderError::TooShort { .. })) => {}
            Err(other) => panic!("expected TooShort, got {other:?}"),
            Ok(_) => panic!("expected open() to fail when file is shorter than the header"),
        }
        let _ = std::fs::remove_file(&path);
    }
}
