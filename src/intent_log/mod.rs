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
//! ## Frame offset index
//!
//! The log is append-only, so a reader that only needs the frames above some
//! LSN would otherwise walk every frame from the start. [`FrameIndex`] maps
//! LSNs to byte offsets: the latest frame plus one sparse entry every
//! [`FRAME_INDEX_STRIDE`] frames. The log keeps it current as it appends; the
//! journaled writer persists it as checkpoint-store side data so the next
//! open starts its tail scan at the latest frame and replay seeks straight to
//! the first frame it needs. An entry is trusted only after the frame at its
//! offset reads back with the recorded LSN and CRC; anything else falls back
//! to a scan from the start of the frame region, so a stale or corrupt index
//! costs time, never correctness.
//!
//! ## Segments
//!
//! A log is a head file (the path the log is opened at, e.g. `intent.log`)
//! and, once it has rotated, sealed segments beside it named
//! `<head>.seg.<base LSN, 20 digits>`. With a segment size set
//! ([`IntentLog::set_segment_bytes`]), a sync that leaves the head at least
//! that long renames it to a segment and opens a new head whose header names
//! the LSN its first frame will carry ([`IntentLogHeader::base_lsn`]) and keeps
//! the log's creation time, which identifies the log to replication followers.
//! Rotation happens only at a sync point, so no frame straddles two files and
//! a rolled-back write never reaches into a sealed one. Sealed segments are
//! never written again, which lets a backup hard-link them instead of copying.
//!
//! Readers ([`IntentLog::iter`], [`IntentLog::iter_from`],
//! [`IntentLog::iter_at`]) cross segments in LSN order and refuse a gap
//! between them. The [`FrameIndex`] covers the head file; positions recorded
//! before a rotation still resolve, in the segment that now holds them.
//!
//! ## What this module is NOT
//!
//! This is the storage primitive only. The `Projection` trait, replay driver,
//! and per-store checkpointing wire on top of this in follow-up work.
//! Keeping the primitive isolated lets it be tested exhaustively before any
//! projection depends on it.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::config::StorageDurability;
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
pub mod reseal;
pub use checkpoint_store::{CheckpointStore, CheckpointStoreError};
pub use header::{HeaderError, IntentLogHeader, CURRENT_FORMAT_VERSION, HEADER_BYTES, MAGIC};
pub use journal::{
    ApplyError, BackgroundFlush, DeferredProjection, JournalError, JournaledWriter, SharedJournal,
    TypedProjection, WriteOutcome,
};
pub use migrations::{migrate_payload, MigrationError};
pub use payload::{IntentPayload, PayloadError};
pub use projection::{replay, replay_all, Projection, ReplayError, ReplayTarget};

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
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
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
    /// The frames of consecutive log files do not continue each other: a
    /// segment is missing or was replaced.
    #[error("intent log segment {path:?} starts at LSN {found}, but the frames before it end at {expected}")]
    SegmentGap {
        path: PathBuf,
        expected: u64,
        found: u64,
    },
}

/// Hard cap on a single payload's size. 16 MiB is far larger than any
/// realistic memory write yet small enough to keep a torn write from
/// requesting a wild allocation if `len` is interpreted from random bytes.
pub const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

const FRAME_HEADER_BYTES: usize = 4 + 8; // len + lsn
const FRAME_TRAILER_BYTES: usize = 4; // crc

/// Frames between sparse [`FrameIndex`] entries: an entry is kept for every
/// LSN that is a multiple of this.
pub const FRAME_INDEX_STRIDE: u64 = 1024;

/// Read-buffer capacity for frame scans. Large enough that a scan of small
/// frames costs one `read` syscall per few hundred frames.
const READ_BUFFER_BYTES: usize = 256 * 1024;

const FRAME_INDEX_ENCODING_VERSION: u8 = 1;

/// Encoded size of one [`FramePosition`].
pub const FRAME_POSITION_BYTES: usize = 8 + 8 + 4;

/// Where one frame lives: its LSN, the byte offset its header starts at, and
/// its stored CRC. The CRC is what lets a reader confirm that the frame found
/// at `offset` is the very frame this position was recorded for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePosition {
    pub lsn: Lsn,
    pub offset: u64,
    pub crc: u32,
}

impl FramePosition {
    /// Wire form: `[lsn u64][offset u64][crc u32]`, little-endian.
    pub fn to_bytes(&self) -> [u8; FRAME_POSITION_BYTES] {
        let mut out = [0u8; FRAME_POSITION_BYTES];
        out[0..8].copy_from_slice(&self.lsn.0.to_le_bytes());
        out[8..16].copy_from_slice(&self.offset.to_le_bytes());
        out[16..20].copy_from_slice(&self.crc.to_le_bytes());
        out
    }

    /// Inverse of [`to_bytes`](Self::to_bytes); `None` unless `bytes` is
    /// exactly one encoded position.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != FRAME_POSITION_BYTES {
            return None;
        }
        Some(Self {
            lsn: Lsn(u64::from_le_bytes(bytes[0..8].try_into().ok()?)),
            offset: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
            crc: u32::from_le_bytes(bytes[16..20].try_into().ok()?),
        })
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bytes());
    }
}

/// LSN → byte-offset hints for one log file: the latest frame, and every
/// [`FRAME_INDEX_STRIDE`]-th frame. See the module docs for how the hints
/// are verified before use.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameIndex {
    sparse: Vec<FramePosition>,
    latest: Option<FramePosition>,
}

impl FrameIndex {
    /// Note a frame that was just appended or scanned. Positions must arrive
    /// in LSN order; an out-of-order position replaces nothing in the sparse
    /// list but still becomes the latest frame.
    pub fn record(&mut self, pos: FramePosition) {
        if pos.lsn.0.is_multiple_of(FRAME_INDEX_STRIDE)
            && self
                .sparse
                .last()
                .is_none_or(|last| last.lsn < pos.lsn && last.offset < pos.offset)
        {
            self.sparse.push(pos);
        }
        self.latest = Some(pos);
    }

    /// The most recently recorded frame.
    pub fn latest(&self) -> Option<FramePosition> {
        self.latest
    }

    /// Sparse entries, in LSN order.
    pub fn sparse(&self) -> &[FramePosition] {
        &self.sparse
    }

    /// The recorded frame with the greatest LSN at or below `target`.
    pub fn seek_hint(&self, target: Lsn) -> Option<FramePosition> {
        if let Some(latest) = self.latest {
            if latest.lsn <= target {
                return Some(latest);
            }
        }
        let below = self.sparse.partition_point(|p| p.lsn <= target);
        below.checked_sub(1).map(|i| self.sparse[i])
    }

    /// Forget every frame starting at or after `end_offset` (a truncation
    /// point, always a frame boundary). `last` becomes the latest frame when
    /// the caller knows which frame now ends the file.
    fn truncate_at(&mut self, end_offset: u64, last: Option<FramePosition>) {
        self.sparse.retain(|p| p.offset < end_offset);
        self.latest = match last {
            Some(pos) => Some(pos),
            None => self.latest.filter(|p| p.offset < end_offset),
        };
    }

    /// Keep only what lies at or below `pos`, which becomes the latest frame.
    fn keep_through(&mut self, pos: FramePosition) {
        self.sparse
            .retain(|p| (p.lsn < pos.lsn && p.offset < pos.offset) || *p == pos);
        self.latest = Some(pos);
    }

    /// Move every offset by `delta` bytes (a header was prepended).
    fn shift(&mut self, delta: u64) {
        for p in &mut self.sparse {
            p.offset += delta;
        }
        if let Some(p) = self.latest.as_mut() {
            p.offset += delta;
        }
    }

    /// Wire form: `[version u8][has_latest u8][latest?][count u32][sparse…]`,
    /// little-endian, each position `[lsn u64][offset u64][crc u32]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + FRAME_POSITION_BYTES * (self.sparse.len() + 1) + 4);
        out.push(FRAME_INDEX_ENCODING_VERSION);
        match self.latest {
            Some(pos) => {
                out.push(1);
                pos.encode_into(&mut out);
            }
            None => out.push(0),
        }
        out.extend_from_slice(&(self.sparse.len() as u32).to_le_bytes());
        for pos in &self.sparse {
            pos.encode_into(&mut out);
        }
        out
    }

    /// Inverse of [`encode`](Self::encode). `None` for anything malformed or
    /// of an unknown version — the caller then simply has no hints.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&version, rest) = bytes.split_first()?;
        if version != FRAME_INDEX_ENCODING_VERSION {
            return None;
        }
        let (&has_latest, mut rest) = rest.split_first()?;
        let latest = match has_latest {
            0 => None,
            1 => {
                let pos = FramePosition::from_bytes(rest.get(..FRAME_POSITION_BYTES)?)?;
                rest = &rest[FRAME_POSITION_BYTES..];
                Some(pos)
            }
            _ => return None,
        };
        let count = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) as usize;
        let rest = &rest[4..];
        if rest.len() != count.checked_mul(FRAME_POSITION_BYTES)? {
            return None;
        }
        let sparse: Vec<FramePosition> = rest
            .as_chunks::<FRAME_POSITION_BYTES>()
            .0
            .iter()
            .map(|chunk| FramePosition::from_bytes(chunk))
            .collect::<Option<_>>()?;
        if sparse.windows(2).any(|w| w[0].lsn >= w[1].lsn) {
            return None;
        }
        Some(Self { sparse, latest })
    }
}

/// Separates a sealed segment's file name from its head file's name.
const SEGMENT_INFIX: &str = ".seg.";
const SEGMENT_DIGITS: usize = 20;

/// A sealed segment: a former head file, renamed at rotation and never
/// written again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// LSN of its first frame.
    pub base: Lsn,
    pub path: PathBuf,
    /// Byte offset its frames start at: past the header, or 0 for a legacy
    /// (header-less) file.
    pub frame_start: u64,
}

/// The path a segment of the log at `log_path` whose first frame is `base` is
/// sealed under.
pub fn segment_path(log_path: &Path, base: Lsn) -> PathBuf {
    let mut name = log_path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(
        "{SEGMENT_INFIX}{:0width$}",
        base.0,
        width = SEGMENT_DIGITS
    ));
    log_path.with_file_name(name)
}

/// The base LSN `file_name` carries when it is a sealed segment of the log
/// whose head file is named `log_name`.
pub fn segment_base(log_name: &str, file_name: &str) -> Option<Lsn> {
    let digits = file_name
        .strip_prefix(log_name)?
        .strip_prefix(SEGMENT_INFIX)?;
    if digits.len() != SEGMENT_DIGITS || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok().map(Lsn)
}

/// Whether `file_name` holds frames of the log whose head file is named
/// `log_name`: the head file itself or one of its sealed segments.
pub fn is_log_file(log_name: &str, file_name: &str) -> bool {
    file_name == log_name || segment_base(log_name, file_name).is_some()
}

/// The sealed segments of the log at `log_path`, oldest first.
pub fn sealed_segments(log_path: &Path) -> Result<Vec<Segment>, IntentLogError> {
    let Some(log_name) = log_path.file_name().and_then(|n| n.to_str()) else {
        return Ok(Vec::new());
    };
    let dir = match log_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut segments = Vec::new();
    for entry in entries {
        let entry = entry?;
        let Some(base) = entry
            .file_name()
            .to_str()
            .and_then(|name| segment_base(log_name, name))
        else {
            continue;
        };
        let path = entry.path();
        let frame_start = probe_mode(&path)?.frame_region_start();
        segments.push(Segment {
            base,
            path,
            frame_start,
        });
    }
    segments.sort_by_key(|s| s.base);
    Ok(segments)
}

/// Every file holding frames of the log at `log_path`, in LSN order: its
/// sealed segments, then the head file when it exists.
pub fn log_files(log_path: &Path) -> Result<Vec<PathBuf>, IntentLogError> {
    let mut files: Vec<PathBuf> = sealed_segments(log_path)?
        .into_iter()
        .map(|s| s.path)
        .collect();
    if log_path.is_file() {
        files.push(log_path.to_path_buf());
    }
    Ok(files)
}

/// Bytes the log at `log_path` occupies across its files.
pub fn log_bytes(log_path: &Path) -> Result<u64, IntentLogError> {
    let mut total = 0;
    for file in log_files(log_path)? {
        total += std::fs::metadata(&file)?.len();
    }
    Ok(total)
}

/// Where a rotation stages the next head file before renaming it into place.
fn rotation_tmp_path(log_path: &Path) -> PathBuf {
    let mut name = log_path.file_name().unwrap_or_default().to_os_string();
    name.push(".rotate.tmp");
    log_path.with_file_name(name)
}

/// Make the bytes written to `file` as durable as `durability` promises.
///
/// `Immediate` waits for stable storage: `sync_data`, which on Apple platforms
/// is `F_FULLFSYNC`, a flush of the drive's whole cache costing milliseconds.
/// `Eventual` is the class whose stores may lose their latest commits to a
/// power loss, so the journal need not wait for the drive's cache either:
/// `F_BARRIERFSYNC` hands the bytes to the drive ordered before every later
/// write, as redb's `Eventual` commit does, so a power loss can only take a
/// suffix of the log. Elsewhere `sync_data` is `fdatasync`, which is already
/// that cost. A filesystem that refuses barriers gets the full sync instead.
fn sync_file(file: &File, durability: StorageDurability) -> io::Result<()> {
    #[cfg(target_vendor = "apple")]
    if durability == StorageDurability::Eventual {
        let fd = std::os::fd::AsRawFd::as_raw_fd(file);
        loop {
            // SAFETY: `F_BARRIERFSYNC` takes no argument beyond the descriptor,
            // which `file` keeps open for the duration of the call.
            if unsafe { libc::fcntl(fd, libc::F_BARRIERFSYNC) } != -1 {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            match error.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::ENOTSUP | libc::ENOTTY | libc::EINVAL) => break,
                _ => return Err(error),
            }
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    let _ = durability;
    file.sync_data()
}

/// Make a rename in `path`'s directory durable.
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir,
            _ => Path::new("."),
        };
        File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// The LSN after the last frame of `segment`: its base when it holds none.
fn segment_next_lsn(segment: &Segment) -> Result<Lsn, IntentLogError> {
    let (_, _, index) = scan_for_tail(&segment.path, segment.frame_start, None)?;
    Ok(index.latest().map_or(segment.base, |p| p.lsn.next()))
}

/// Append-only durable intent log.
///
/// A head file plus any sealed segments (see the module docs). Opens the head
/// in append mode; reads scan from the oldest segment. `append` writes are
/// buffered; call `sync` to force them to disk.
pub struct IntentLog {
    path: PathBuf,
    writer: BufWriter<LogFile>,
    next_lsn: Lsn,
    /// File offset (in bytes from start) of the byte *after* the last
    /// successfully-written frame. Used by `truncate_corrupt_tail` and to
    /// answer "how big is the durable portion of this file" without an
    /// extra fstat. Includes any header bytes — `durable_end_offset` is
    /// always relative to byte 0 of the file.
    durable_end_offset: u64,
    /// `next_lsn` and `durable_end_offset` as of the last successful `sync`
    /// (or open, or truncation). No caller has been told the frames past
    /// this point are durable, so a failed append or sync rolls back to it.
    synced_next_lsn: Lsn,
    synced_end_offset: u64,
    /// The index's latest frame as of the same sync, which a rollback restores.
    synced_index_latest: Option<FramePosition>,
    /// A failed write left bytes past `synced_end_offset` that could not be
    /// cut yet. `append` and `sync` retry the cut before writing anything,
    /// so no frame ever lands after a torn one.
    cut_pending: bool,
    /// Versioning mode this log file is in. `Versioned { .. }` means the
    /// file starts with a valid [`IntentLogHeader`]; `Legacy` means it
    /// was created before headers existed and frames start at offset 0.
    mode: LogMode,
    /// Optional per-frame payload encryptor (encryption-v2). When present, each
    /// frame's payload bytes are sealed on append and opened on read, so the
    /// journal never holds plaintext memory snapshots at rest. `None` = plaintext
    /// journal (backward compatible). Keystore-derived ("journal" purpose) like
    /// the slow store. The CRC is computed over the on-disk (sealed) bytes, so
    /// `scan_for_tail` validates frames without decrypting.
    cryptor: Option<crate::keystore::SecretCryptor>,
    /// Offset hints for the frames in this file, kept current by `append`
    /// and the truncation paths. See the module docs.
    index: FrameIndex,
    /// Sealed segments before the head file, oldest first.
    sealed: Vec<Segment>,
    /// LSN of the head file's first frame.
    base_lsn: Lsn,
    /// A sync that leaves the head file at least this long rotates it.
    /// `None` never rotates.
    segment_bytes: Option<u64>,
    /// After a failed rotation, the head length at which the next attempt is
    /// made, so a lasting failure is retried (and logged) every so often
    /// rather than at every sync.
    retry_rotation_at: u64,
    /// What `sync` waits for; see [`sync_file`].
    durability: StorageDurability,
}

/// The file under the log's `BufWriter`. A newtype rather than a bare `File`
/// so a test can make a write stop part-way through a frame, as ENOSPC does,
/// on a filesystem that has room.
struct LogFile {
    file: File,
    /// Bytes later writes may still put on disk before each one fails with
    /// `StorageFull`. `None` means no fault is armed.
    #[cfg(test)]
    write_budget: Option<usize>,
}

impl LogFile {
    fn new(file: File) -> Self {
        Self {
            file,
            #[cfg(test)]
            write_budget: None,
        }
    }
}

impl Write for LogFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[cfg(test)]
        if let Some(left) = self.write_budget {
            if left == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::StorageFull,
                    "injected: no space left on device",
                ));
            }
            let n = self.file.write(&buf[..buf.len().min(left)])?;
            self.write_budget = Some(left - n);
            return Ok(n);
        }
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// Open `path` for appending, behind the log's write buffer.
fn append_writer(path: &Path) -> io::Result<BufWriter<LogFile>> {
    let file = OpenOptions::new().read(true).append(true).open(path)?;
    Ok(BufWriter::new(LogFile::new(file)))
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
    /// The creation time a versioned header records; 0 for a legacy file.
    pub fn created_unix_secs(&self) -> u64 {
        match self {
            LogMode::Versioned {
                created_unix_secs, ..
            } => *created_unix_secs,
            LogMode::Legacy => 0,
        }
    }

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
    /// file to compute the next LSN and the durable-end offset.
    ///
    /// A torn tail (the partial or CRC-bad last frame a crash or a full disk
    /// leaves) is cut with `truncate_corrupt_tail()`, because the first
    /// append would otherwise land after it where no reader reaches. A
    /// corrupt frame with intact frames after it is left for an operator and
    /// reported by `iter()`: cutting there would drop acknowledged records.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, IntentLogError> {
        Self::open_with_cryptor(path, None)
    }

    /// Like [`open`](Self::open), but seals each frame's payload with `cryptor`
    /// on append and opens it on read (encryption-v2). `None` keeps the journal
    /// plaintext (backward compatible).
    pub fn open_with_cryptor<P: AsRef<Path>>(
        path: P,
        cryptor: Option<crate::keystore::SecretCryptor>,
    ) -> Result<Self, IntentLogError> {
        Self::open_with_index(path, cryptor, None)
    }

    /// Like [`open_with_cryptor`](Self::open_with_cryptor), starting the tail
    /// scan at `hint`'s latest frame instead of the first one when that frame
    /// still reads back with the recorded LSN and CRC. A hint that does not
    /// verify is ignored and the whole file is scanned, exactly as without
    /// one.
    pub fn open_with_index<P: AsRef<Path>>(
        path: P,
        cryptor: Option<crate::keystore::SecretCryptor>,
        hint: Option<FrameIndex>,
    ) -> Result<Self, IntentLogError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // A rotation that died before renaming its new head into place leaves
        // it staged; the head it would have replaced is intact.
        let staged = rotation_tmp_path(&path);
        if staged.exists() {
            std::fs::remove_file(&staged)?;
        }
        let mut sealed = sealed_segments(&path)?;

        // Probe the file shape: new / versioned / legacy. The probe also
        // tells us the byte offset at which frames begin (after a header,
        // for versioned files; at 0, for legacy files).
        let (mode, header_base) = probe_header(&path)?;
        if header_base > 0 {
            // Segments at or past the head's base are what an interrupted
            // truncation into a sealed segment was dropping. They are set
            // aside rather than deleted, and out of the way of the name the
            // head will be sealed under.
            let stale = sealed.partition_point(|s| s.base.0 < header_base);
            for segment in sealed.drain(stale..) {
                let mut aside = segment.path.as_os_str().to_os_string();
                aside.push(".orphaned");
                std::fs::rename(&segment.path, &aside)?;
                tracing::warn!(
                    path = %segment.path.display(),
                    head_base = header_base,
                    "intent log segment starts at or after the head; set aside as .orphaned",
                );
            }
        }

        // Now scan frames from the appropriate offset to discover next
        // LSN + durable_end. A corrupt or truncated tail is reported via
        // the returned offset — open() does not truncate.
        let (scanned_next, durable_end, index) =
            scan_for_tail(&path, mode.frame_region_start(), hint.as_ref())?;

        // The head's first LSN is what its header names. A head with base 0
        // after sealed segments is one a rotation never finished stamping (or
        // that a build without segments created): it continues the last one.
        // So does a head holding no frames yet, whatever its header says: a
        // rotation that failed part-way may have left frames in the segment
        // after the LSN the new head was stamped with.
        let base_lsn = match sealed.last() {
            Some(last) if header_base == 0 || index.latest().is_none() => {
                segment_next_lsn(last)?.max(Lsn(header_base))
            }
            _ => Lsn(header_base),
        };
        let next_lsn = if index.latest().is_some() {
            scanned_next
        } else {
            base_lsn
        };

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
        let mut actual_len = file.metadata()?.len();
        // An empty head whose header names another base than it continues is
        // stamped again, so later opens, once it holds frames, read it right.
        let restamp = mode.is_versioned()
            && actual_len == HEADER_BYTES as u64
            && index.latest().is_none()
            && base_lsn.0 != header_base;
        if restamp {
            OpenOptions::new().write(true).open(&path)?.set_len(0)?;
            actual_len = 0;
        }
        let mode = if mode.is_versioned() && actual_len == 0 {
            let header = match sealed.last() {
                Some(last) => IntentLogHeader::for_segment(
                    probe_mode(&last.path)?.created_unix_secs(),
                    base_lsn.0,
                ),
                None => IntentLogHeader::new_current(),
            };
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

        let mut log = Self {
            path,
            writer: BufWriter::new(LogFile::new(file)),
            next_lsn,
            durable_end_offset: durable_end,
            synced_next_lsn: next_lsn,
            synced_end_offset: durable_end,
            synced_index_latest: index.latest(),
            cut_pending: false,
            mode,
            cryptor,
            index,
            sealed,
            base_lsn,
            segment_bytes: None,
            retry_rotation_at: 0,
            durability: crate::config::StorageTuning::process().durability,
        };
        if actual_len > durable_end {
            if tail_is_torn(&log.path, durable_end, actual_len, next_lsn)? {
                tracing::warn!(
                    path = %log.path.display(),
                    torn_bytes = actual_len - durable_end,
                    "intent log ends in a torn frame; cutting it",
                );
                log.truncate_corrupt_tail()?;
            } else {
                tracing::error!(
                    path = %log.path.display(),
                    offset = durable_end,
                    "intent log has a corrupt frame with intact frames after it; left in place",
                );
            }
        }
        Ok(log)
    }

    /// Offset hints for the frames currently in the file.
    pub fn frame_index(&self) -> &FrameIndex {
        &self.index
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
        self.writer.get_ref().file.sync_data()?;

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
        self.writer = append_writer(&tmp_path)?;
        std::fs::rename(&tmp_path, &self.path)?;
        self.writer = append_writer(&self.path)?;

        // The frame region just shifted by HEADER_BYTES, so durable_end
        // moves with it.
        self.durable_end_offset += HEADER_BYTES as u64;
        self.index.shift(HEADER_BYTES as u64);
        self.synced_next_lsn = self.next_lsn;
        self.synced_end_offset = self.durable_end_offset;
        self.synced_index_latest = self.index.latest();
        self.mode = LogMode::Versioned {
            format_version: header.format_version,
            created_unix_secs: header.created_unix_secs,
        };
        Ok(true)
    }

    /// Append a record. Returns the LSN assigned to it. Writes go through a
    /// `BufWriter` — call `sync` for durability before returning to a caller
    /// that requires the record to survive a crash.
    ///
    /// If the write fails, every frame appended since the last successful
    /// `sync` is dropped from the buffer and cut from the file, and
    /// `next_lsn` rewinds to the first of them: the log ends at a frame
    /// boundary, so the next append is readable.
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
        if self.cut_pending {
            self.cut_unsynced()?;
        }

        let lsn = self.next_lsn;
        // Seal the payload at rest (encryption-v2) when a journal cryptor is
        // configured; len + CRC + the durable-offset accounting all use the
        // sealed (on-disk) bytes, so scan_for_tail validates frames without a key.
        let sealed: Vec<u8> = match &self.cryptor {
            Some(c) => c
                .encrypt_bytes(payload, b"")
                .map_err(|e| io::Error::other(format!("journal seal failed: {e}")))?,
            None => payload.to_vec(),
        };
        let len = sealed.len() as u32;

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&len.to_le_bytes());
        hasher.update(&lsn.0.to_le_bytes());
        hasher.update(&sealed);
        let crc = hasher.finalize();

        if let Err(e) = (|| -> io::Result<()> {
            self.writer.write_all(&len.to_le_bytes())?;
            self.writer.write_all(&lsn.0.to_le_bytes())?;
            self.writer.write_all(&sealed)?;
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
            self.roll_back_unsynced();
            return Err(IntentLogError::Io(e));
        }

        self.next_lsn = lsn.next();
        self.index.record(FramePosition {
            lsn,
            offset: self.durable_end_offset,
            crc,
        });
        self.durable_end_offset += (FRAME_HEADER_BYTES + sealed.len() + FRAME_TRAILER_BYTES) as u64;

        INTENT_LOG_APPEND_TOTAL.with_label_values(&["ok"]).inc();
        INTENT_LOG_NEXT_LSN.set(self.next_lsn.0 as i64);
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(self.durable_end_offset as i64);

        Ok(lsn)
    }

    /// Flush the BufWriter and sync the file at the log's durability (see
    /// [`Self::set_durability`]). After this returns, every frame
    /// `append`-ed since the last `sync` is durable. If it fails, those
    /// frames are rolled back as for a failed `append`, so a record the
    /// caller was told failed cannot become durable with a later sync.
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
        if self.cut_pending {
            self.cut_unsynced()?;
        }
        if let Err(e) = self.writer.flush() {
            INTENT_LOG_SYNC_TOTAL.with_label_values(&["io_error"]).inc();
            tracing::error!(error = %e, "intent log sync failed: flush error");
            self.roll_back_unsynced();
            return Err(IntentLogError::Io(e));
        }
        if let Err(e) = sync_file(&self.writer.get_ref().file, self.durability) {
            INTENT_LOG_SYNC_TOTAL.with_label_values(&["io_error"]).inc();
            tracing::error!(error = %e, "intent log sync failed: fsync error");
            self.roll_back_unsynced();
            return Err(IntentLogError::Io(e));
        }
        self.synced_next_lsn = self.next_lsn;
        self.synced_end_offset = self.durable_end_offset;
        self.synced_index_latest = self.index.latest();
        INTENT_LOG_SYNC_TOTAL.with_label_values(&["ok"]).inc();
        let due = self
            .segment_bytes
            .is_some_and(|limit| self.durable_end_offset >= limit.max(self.retry_rotation_at));
        if due && self.index.latest().is_some() {
            // The frames are durable either way; a head that could not rotate
            // keeps taking appends and is tried again once it has grown by
            // another eighth of a segment.
            if let Err(e) = self.rotate() {
                let limit = self.segment_bytes.unwrap_or_default();
                self.retry_rotation_at = self.durable_end_offset + (limit / 8).max(1);
                tracing::error!(
                    path = %self.path.display(),
                    bytes = self.durable_end_offset,
                    retry_at = self.retry_rotation_at,
                    error = %e,
                    "intent log rotation failed; appending to the current head",
                );
            }
        }
        Ok(())
    }

    /// Rotate the head file once a sync leaves it at least `limit` bytes
    /// long; `None` never rotates.
    pub fn set_segment_bytes(&mut self, limit: Option<u64>) {
        self.segment_bytes = limit;
    }

    /// What every later `sync` waits for. A log opens at the process's
    /// [`StorageTuning`](crate::config::StorageTuning) durability, the class
    /// the stores it journals for commit at.
    pub fn set_durability(&mut self, durability: StorageDurability) {
        self.durability = durability;
    }

    /// What `sync` waits for.
    pub fn durability(&self) -> StorageDurability {
        self.durability
    }

    /// Drop every frame appended since the last successful `sync`, as a
    /// failed append does. Lets a writer that appends several records as one
    /// unit take back the ones already appended when a later one fails.
    pub(crate) fn discard_unsynced(&mut self) {
        if self.next_lsn != self.synced_next_lsn {
            self.roll_back_unsynced();
        }
    }

    /// The sealed segments before the head file, oldest first.
    pub fn sealed(&self) -> &[Segment] {
        &self.sealed
    }

    /// LSN of the head file's first frame.
    pub fn base_lsn(&self) -> Lsn {
        self.base_lsn
    }

    /// Seal the head file as a segment and open a new one. Called only right
    /// after a successful sync, so everything written is durable and the
    /// buffer is empty. The new head is staged, and renamed into place only
    /// after the old one is sealed: a crash at any point leaves either the
    /// old head, or the sealed segment with a head (or none, which the next
    /// open creates) continuing it.
    fn rotate(&mut self) -> Result<(), IntentLogError> {
        let started = std::time::Instant::now();
        let sealed_path = segment_path(&self.path, self.base_lsn);
        if sealed_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("segment {} already exists", sealed_path.display()),
            )
            .into());
        }
        let created = match self.mode {
            LogMode::Versioned {
                created_unix_secs, ..
            } => created_unix_secs,
            LogMode::Legacy => IntentLogHeader::new_current().created_unix_secs,
        };
        let header = IntentLogHeader::for_segment(created, self.next_lsn.0);
        let staged = rotation_tmp_path(&self.path);
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged)?;
            file.write_all(&header.to_bytes())?;
            file.sync_data()?;
        }
        // Both handles follow their files through the renames (std opens them
        // shareable for delete, so Windows allows it): until the switch below,
        // every append still lands after the frames it follows, whatever the
        // old head is called by then.
        let next_writer = match append_writer(&staged) {
            Ok(writer) => writer,
            Err(e) => {
                let _ = std::fs::remove_file(&staged);
                return Err(e.into());
            }
        };
        if let Err(e) = std::fs::rename(&self.path, &sealed_path) {
            let _ = std::fs::remove_file(&staged);
            return Err(e.into());
        }
        let placed =
            std::fs::rename(&staged, &self.path).and_then(|()| sync_parent_dir(&self.path));
        if let Err(e) = placed {
            let restored = if staged.exists() {
                std::fs::rename(&sealed_path, &self.path)
            } else {
                std::fs::rename(&self.path, &staged)
                    .and_then(|()| std::fs::rename(&sealed_path, &self.path))
            };
            if let Err(undo) = restored {
                tracing::error!(
                    path = %self.path.display(),
                    sealed = %sealed_path.display(),
                    error = %undo,
                    "intent log rotation could not restore the head; the next open continues the sealed segment",
                );
            }
            let _ = std::fs::remove_file(&staged);
            return Err(e.into());
        }
        self.writer = next_writer;

        let sealed_bytes = self.durable_end_offset;
        self.sealed.push(Segment {
            base: self.base_lsn,
            path: sealed_path,
            frame_start: self.mode.frame_region_start(),
        });
        self.base_lsn = self.next_lsn;
        self.mode = LogMode::Versioned {
            format_version: header.format_version,
            created_unix_secs: header.created_unix_secs,
        };
        self.durable_end_offset = HEADER_BYTES as u64;
        self.synced_end_offset = self.durable_end_offset;
        self.index = FrameIndex::default();
        self.synced_index_latest = None;
        self.retry_rotation_at = 0;
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(self.durable_end_offset as i64);
        tracing::info!(
            path = %self.path.display(),
            sealed_bytes,
            segments = self.sealed.len(),
            next_lsn = self.next_lsn.0,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "intent log rotated its head into a sealed segment",
        );
        Ok(())
    }

    /// Forget every frame since the last successful sync and cut its bytes,
    /// after an append or sync failed part-way through them.
    fn roll_back_unsynced(&mut self) {
        self.next_lsn = self.synced_next_lsn;
        self.durable_end_offset = self.synced_end_offset;
        self.index
            .truncate_at(self.synced_end_offset, self.synced_index_latest);
        INTENT_LOG_NEXT_LSN.set(self.next_lsn.0 as i64);
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(self.durable_end_offset as i64);
        self.cut_pending = true;
        if let Err(e) = self.cut_unsynced() {
            tracing::error!(
                error = %e,
                end_offset = self.synced_end_offset,
                "intent log could not cut a failed write back to the last synced frame; \
                 appends are refused until it can",
            );
        }
    }

    fn cut_unsynced(&mut self) -> io::Result<()> {
        // Swap the handle before cutting so no later flush can write the
        // stale bytes: `into_parts` discards them, where dropping the
        // BufWriter would flush them.
        let stale = std::mem::replace(&mut self.writer, append_writer(&self.path)?);
        drop(stale.into_parts());
        // A transient read+write handle, as in `truncate_corrupt_tail`.
        let file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        if file.metadata()?.len() > self.synced_end_offset {
            file.set_len(self.synced_end_offset)?;
        }
        file.sync_data()?;
        self.cut_pending = false;
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

    /// Iterate every frame, from the oldest sealed segment through the head.
    /// Stops at the first corrupt or truncated frame; returns an
    /// `IntentLogError` for that frame so the caller can decide whether to
    /// truncate. The iterator is independent of the writer — it opens its
    /// own read handles. Headers are skipped.
    pub fn iter(&self) -> Result<IntentLogIter, IntentLogError> {
        self.iter_files(0, None)
    }

    /// An iterator over the sealed segments from `first` on, then the head,
    /// starting in the first of them at `offset` (its frame start when
    /// `None`).
    fn iter_files(
        &self,
        first: usize,
        offset: Option<u64>,
    ) -> Result<IntentLogIter, IntentLogError> {
        // Every file is opened now: a rotation while the iterator is in use
        // renames the head, and a handle follows its file where a path would
        // land on the new head.
        let mut files = VecDeque::new();
        for segment in &self.sealed[first.min(self.sealed.len())..] {
            files.push_back(NextFile::open(&segment.path, segment.frame_start)?);
        }
        files.push_back(NextFile::open(&self.path, self.mode.frame_region_start())?);
        let first = files.pop_front().expect("the head is always listed");
        let start = offset.unwrap_or(first.frame_start);
        let mut iter = IntentLogIter::from_file(first.file, start, self.cryptor.clone())?;
        iter.rest = files;
        Ok(iter)
    }

    /// The sealed segment holding `lsn`, when `lsn` precedes the head.
    fn sealed_holding(&self, lsn: Lsn) -> Option<usize> {
        if lsn >= self.base_lsn {
            return None;
        }
        self.sealed
            .partition_point(|s| s.base <= lsn)
            .checked_sub(1)
    }

    /// Like [`iter`](Self::iter), but positioned at the indexed frame with
    /// the greatest LSN at or below `target`, so a reader that needs only the
    /// frames from `target` on skips the bulk of the file. Frames below
    /// `target` may still be yielded (the index is sparse); callers filter by
    /// LSN. When no index entry applies, or the frame at the entry's offset
    /// does not carry the recorded LSN and CRC, iteration starts at the first
    /// frame.
    pub fn iter_from(&self, target: Lsn) -> Result<IntentLogIter, IntentLogError> {
        if target < self.base_lsn {
            // Sealed segments carry no offset hints: read the one holding
            // `target` from its first frame.
            return self.iter_files(self.sealed_holding(target).unwrap_or(0), None);
        }
        let region_start = self.mode.frame_region_start();
        if let Some(pos) = self.index.seek_hint(target) {
            if pos.offset > region_start && pos.offset < self.durable_end_offset {
                if let Some(iter) = self.iter_at(pos)? {
                    return Ok(iter);
                }
                tracing::warn!(
                    path = %self.path.display(),
                    lsn = pos.lsn.0,
                    offset = pos.offset,
                    "intent log offset hint did not verify; reading the head from its first frame",
                );
            }
        }
        IntentLogIter::open(&self.path, region_start, self.cryptor.clone())
    }

    /// An iterator positioned at `pos`, or `None` when the frame found there
    /// is not the one `pos` describes.
    pub fn iter_at(&self, pos: FramePosition) -> Result<Option<IntentLogIter>, IntentLogError> {
        let mut iter = if pos.lsn < self.base_lsn {
            let Some(at) = self.sealed_holding(pos.lsn) else {
                return Ok(None);
            };
            if pos.offset < self.sealed[at].frame_start {
                return Ok(None);
            }
            self.iter_files(at, Some(pos.offset))?
        } else {
            if pos.offset < self.mode.frame_region_start() || pos.offset >= self.durable_end_offset
            {
                return Ok(None);
            }
            IntentLogIter::open(&self.path, pos.offset, self.cryptor.clone())?
        };
        if iter.verify_and_rewind(pos)? {
            Ok(Some(iter))
        } else {
            Ok(None)
        }
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
        let trunc_handle = OpenOptions::new().read(true).write(true).open(&self.path)?;
        trunc_handle.set_len(target)?;
        trunc_handle.sync_data()?;
        drop(trunc_handle);

        // Reopen so BufWriter's seek matches the new EOF.
        self.writer = append_writer(&self.path)?;
        self.index.truncate_at(target, None);
        self.synced_next_lsn = self.next_lsn;
        self.synced_end_offset = target;
        self.synced_index_latest = self.index.latest();
        self.cut_pending = false;
        INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL.inc();
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(target as i64);
        tracing::info!(target_offset = target, "intent log truncated corrupt tail",);
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
        if let Some(at) = self.sealed_holding(max_lsn) {
            self.reopen_at_segment(at)?;
        }
        // Flush any buffered append so the on-disk image is what we scan.
        self.writer.flush()?;

        // Walk frames in a scoped block so the scanner's read handle is
        // dropped before we attempt the truncate. Windows refuses
        // `set_len` while another read handle is open against the same
        // file, even though POSIX is happy with it.
        let (last_good_offset, last_frame) = {
            let mut iter = IntentLogIter::open_for_scan(&self.path)?;
            let mut found: Option<(u64, FramePosition)> = None;
            let mut highest_seen: Option<Lsn> = None;
            while let Some(frame) = iter.read_frame()? {
                highest_seen = Some(frame.lsn);
                if frame.lsn == max_lsn {
                    found = Some((iter.current_offset(), frame.position()));
                    break;
                }
                if frame.lsn.0 > max_lsn.0 {
                    // Walked past target without finding an exact match.
                    // The caller's max_lsn is between frames, which is
                    // not allowed — LSNs are dense.
                    return Err(IntentLogError::Io(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "max_lsn={} is not present in intent log (highest seen so far: {})",
                            max_lsn.0, frame.lsn.0
                        ),
                    )));
                }
            }
            match found {
                Some(found) => found,
                None => {
                    return Err(IntentLogError::Io(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "max_lsn={} not found in intent log (highest LSN present: {})",
                            max_lsn.0,
                            highest_seen.map(|l| l.0 as i128).unwrap_or(-1),
                        ),
                    )));
                }
            }
            // `iter` dropped here, releasing its read handle.
        };

        // Drop our own append-mode writer before truncating: on Windows
        // a handle opened with append(true) lacks FILE_WRITE_DATA, so
        // calling `set_len` on it returns ERROR_ACCESS_DENIED. Reopen
        // with read+write so the truncate succeeds, then reattach the
        // append handle afterwards.
        let trunc_handle = OpenOptions::new().read(true).write(true).open(&self.path)?;
        trunc_handle.set_len(last_good_offset)?;
        trunc_handle.sync_data()?;
        drop(trunc_handle);

        // Reopen so BufWriter's append position matches the new EOF, and
        // refresh the in-memory tail bookkeeping.
        self.writer = append_writer(&self.path)?;
        self.durable_end_offset = last_good_offset;
        self.next_lsn = max_lsn.next();
        self.index.truncate_at(last_good_offset, Some(last_frame));
        self.synced_next_lsn = self.next_lsn;
        self.synced_end_offset = last_good_offset;
        self.synced_index_latest = self.index.latest();
        self.cut_pending = false;
        Ok(last_good_offset)
    }
}

impl IntentLog {
    /// Make sealed segment `at` the head again, dropping the head and every
    /// later segment, so a truncation can cut inside it. The segment is
    /// copied to a fresh file first: a backup may hold a hard link to it, and
    /// the cut must not reach that copy.
    fn reopen_at_segment(&mut self, at: usize) -> Result<(), IntentLogError> {
        self.writer.flush()?;
        let path = self.path.clone();
        let staged = rotation_tmp_path(&path);
        std::fs::copy(&self.sealed[at].path, &staged)?;
        OpenOptions::new().write(true).open(&staged)?.sync_data()?;
        std::fs::rename(&staged, &path)?;
        for later in self.sealed[at..].iter().rev() {
            std::fs::remove_file(&later.path)?;
        }
        sync_parent_dir(&path)?;
        let segment_bytes = self.segment_bytes;
        *self = Self::open_with_index(&path, self.cryptor.clone(), None)?;
        self.segment_bytes = segment_bytes;
        Ok(())
    }
}

/// One frame returned by [`IntentLogIter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentRecord {
    pub lsn: Lsn,
    pub payload: Vec<u8>,
}

/// Header facts of one CRC-verified frame read by
/// [`IntentLogIter::read_frame`]. The frame's payload, still sealed if the
/// journal is encrypted, stays in the iterator until the next read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    pub lsn: Lsn,
    /// Byte offset of the frame's header.
    pub offset: u64,
    /// Stored (and verified) CRC.
    pub crc: u32,
    /// On-disk payload length.
    pub len: usize,
}

impl FrameMeta {
    /// This frame's [`FramePosition`].
    pub fn position(&self) -> FramePosition {
        FramePosition {
            lsn: self.lsn,
            offset: self.offset,
            crc: self.crc,
        }
    }
}

/// Forward-only iterator over the frames in an intent log file. Stops at
/// EOF or the first corrupt/truncated frame; in the latter case yields one
/// `Err(...)` and then `None`. The byte offset reported in errors lets the
/// caller line up `truncate_corrupt_tail`.
///
/// Reads go through a buffered reader into one reused body buffer. Callers
/// that only need some frames use [`read_frame`](Self::read_frame), which
/// verifies a frame's CRC over its on-disk bytes without opening the payload,
/// and [`take_payload`](Self::take_payload) for the frames they keep.
pub struct IntentLogIter {
    reader: BufReader<File>,
    offset: u64,
    finished: bool,
    /// Opens (decrypts) each frame's sealed payload before yielding. `None` =
    /// payloads yielded as-is (plaintext journal, or a structural-only scan).
    cryptor: Option<crate::keystore::SecretCryptor>,
    /// Payload then CRC of the frame last read.
    body: Vec<u8>,
    /// Payload length of the frame in `body`.
    body_len: usize,
    /// Log files still to read after this one, oldest first.
    rest: VecDeque<NextFile>,
    /// LSN of the last frame read, which the first frame of the next file
    /// must follow.
    last_lsn: Option<Lsn>,
    /// The current file was entered from the previous one and its first
    /// frame has not been read yet.
    entered: Option<PathBuf>,
}

/// A log file an [`IntentLogIter`] reads after the current one.
struct NextFile {
    path: PathBuf,
    file: File,
    frame_start: u64,
}

impl NextFile {
    fn open(path: &Path, frame_start: u64) -> Result<Self, IntentLogError> {
        Ok(Self {
            path: path.to_path_buf(),
            file: File::open(path)?,
            frame_start,
        })
    }
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
        // Structural scan only — yields sealed payloads as-is (no key needed).
        Self::open(path, mode.frame_region_start(), None)
    }

    fn open(
        path: &Path,
        start_offset: u64,
        cryptor: Option<crate::keystore::SecretCryptor>,
    ) -> Result<Self, IntentLogError> {
        Self::from_file(File::open(path)?, start_offset, cryptor)
    }

    fn from_file(
        mut file: File,
        start_offset: u64,
        cryptor: Option<crate::keystore::SecretCryptor>,
    ) -> Result<Self, IntentLogError> {
        file.seek(SeekFrom::Start(start_offset))?;
        Ok(Self {
            reader: BufReader::with_capacity(READ_BUFFER_BYTES, file),
            offset: start_offset,
            finished: false,
            cryptor,
            body: Vec::new(),
            body_len: 0,
            rest: VecDeque::new(),
            last_lsn: None,
            entered: None,
        })
    }

    /// Move to the next log file, or report that there is none.
    fn enter_next_file(&mut self) -> Result<bool, IntentLogError> {
        let Some(NextFile {
            path,
            mut file,
            frame_start,
        }) = self.rest.pop_front()
        else {
            return Ok(false);
        };
        file.seek(SeekFrom::Start(frame_start))?;
        self.reader = BufReader::with_capacity(READ_BUFFER_BYTES, file);
        self.offset = frame_start;
        self.entered = Some(path);
        Ok(true)
    }

    /// Byte offset of the next frame to be read. After `next()` returns
    /// `Err`, this still points at the *start* of the bad frame — so a
    /// caller can `IntentLog::truncate_corrupt_tail` to that offset.
    pub fn current_offset(&self) -> u64 {
        self.offset
    }

    /// Read and CRC-check the next frame without opening its payload.
    /// `Ok(None)` at a clean end of file. After an error the iterator is
    /// finished and [`current_offset`](Self::current_offset) is the start of
    /// the bad frame.
    pub fn read_frame(&mut self) -> Result<Option<FrameMeta>, IntentLogError> {
        if self.finished {
            return Ok(None);
        }
        let result = self.read_frame_across_files();
        if !matches!(result, Ok(Some(_))) {
            self.finished = true;
        }
        result
    }

    fn read_frame_across_files(&mut self) -> Result<Option<FrameMeta>, IntentLogError> {
        loop {
            match self.read_frame_inner()? {
                Some(frame) => {
                    if let Some(path) = self.entered.take() {
                        let expected = self.last_lsn.map_or(frame.lsn, Lsn::next);
                        if frame.lsn != expected {
                            return Err(IntentLogError::SegmentGap {
                                path,
                                expected: expected.0,
                                found: frame.lsn.0,
                            });
                        }
                    }
                    self.last_lsn = Some(frame.lsn);
                    return Ok(Some(frame));
                }
                None if self.enter_next_file()? => continue,
                None => return Ok(None),
            }
        }
    }

    fn read_frame_inner(&mut self) -> Result<Option<FrameMeta>, IntentLogError> {
        let frame_start = self.offset;

        let mut header = [0u8; FRAME_HEADER_BYTES];
        let read = read_full(&mut self.reader, &mut header)?;
        if read == 0 {
            return Ok(None);
        }
        if read < FRAME_HEADER_BYTES {
            return Err(IntentLogError::TruncatedFrame {
                offset: frame_start,
                expected: FRAME_HEADER_BYTES,
                actual: read,
            });
        }

        let len = u32::from_le_bytes(header[0..4].try_into().expect("4-byte slice")) as usize;
        let lsn = u64::from_le_bytes(header[4..12].try_into().expect("8-byte slice"));

        // Reject obviously-wild lengths to avoid huge allocations from
        // misinterpreted garbage bytes.
        if len > MAX_PAYLOAD_BYTES {
            return Err(IntentLogError::TruncatedFrame {
                offset: frame_start,
                expected: len,
                actual: 0,
            });
        }

        self.body.resize(len + FRAME_TRAILER_BYTES, 0);
        self.body_len = len;
        let read = read_full(&mut self.reader, &mut self.body)?;
        if read < len {
            return Err(IntentLogError::TruncatedFrame {
                offset: frame_start,
                expected: FRAME_HEADER_BYTES + len,
                actual: FRAME_HEADER_BYTES + read,
            });
        }
        if read < len + FRAME_TRAILER_BYTES {
            return Err(IntentLogError::TruncatedFrame {
                offset: frame_start,
                expected: FRAME_HEADER_BYTES + len + FRAME_TRAILER_BYTES,
                actual: FRAME_HEADER_BYTES + read,
            });
        }

        let stored_crc = u32::from_le_bytes(self.body[len..].try_into().expect("4-byte slice"));
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header);
        hasher.update(&self.body[..len]);
        let computed_crc = hasher.finalize();
        if stored_crc != computed_crc {
            return Err(IntentLogError::CrcMismatch {
                offset: frame_start,
                stored: stored_crc,
                computed: computed_crc,
            });
        }

        self.offset += (FRAME_HEADER_BYTES + len + FRAME_TRAILER_BYTES) as u64;
        Ok(Some(FrameMeta {
            lsn: Lsn(lsn),
            offset: frame_start,
            crc: stored_crc,
            len,
        }))
    }

    /// On-disk (sealed, if encrypted) payload of the frame last read.
    pub fn sealed_payload(&self) -> &[u8] {
        &self.body[..self.body_len]
    }

    /// The plaintext payload of the frame last read, decrypting it when the
    /// iterator holds a journal key and the payload is sealed. The body
    /// buffer is handed over for plaintext payloads; give it back with
    /// [`recycle`](Self::recycle) to keep reusing it.
    pub fn take_payload(&mut self) -> Result<Vec<u8>, IntentLogError> {
        let frame_start = self
            .offset
            .saturating_sub((FRAME_HEADER_BYTES + self.body_len + FRAME_TRAILER_BYTES) as u64);
        let opened = match &self.cryptor {
            Some(c) if crate::keystore::SecretCryptor::is_encrypted(self.sealed_payload()) => {
                note_payload_decrypt();
                c.decrypt_bytes(self.sealed_payload(), b"")
            }
            _ => {
                let mut payload = std::mem::take(&mut self.body);
                payload.truncate(self.body_len);
                self.body_len = 0;
                return Ok(payload);
            }
        };
        opened.map_err(|e| {
            self.finished = true;
            IntentLogError::Io(io::Error::other(format!(
                "journal payload open failed at offset {frame_start}: {e}"
            )))
        })
    }

    /// Return a buffer from [`take_payload`](Self::take_payload) for reuse.
    pub fn recycle(&mut self, mut buf: Vec<u8>) {
        if buf.capacity() > self.body.capacity() {
            buf.clear();
            self.body = buf;
        }
    }

    /// Read the frame at the current offset and check it is the frame `pos`
    /// records; on success rewind so that frame is read again next.
    fn verify_and_rewind(&mut self, pos: FramePosition) -> Result<bool, IntentLogError> {
        // Only the file `pos` was recorded in: never the first frame of the next.
        let matches = match self.read_frame_inner() {
            Ok(Some(meta)) => meta.position() == pos,
            Ok(None) | Err(_) => false,
        };
        if matches {
            self.reader.seek(SeekFrom::Start(pos.offset))?;
            self.offset = pos.offset;
            self.finished = false;
        }
        Ok(matches)
    }
}

impl Iterator for IntentLogIter {
    type Item = Result<IntentRecord, IntentLogError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_frame() {
            Ok(None) => None,
            Ok(Some(meta)) => Some(self.take_payload().map(|payload| IntentRecord {
                lsn: meta.lsn,
                payload,
            })),
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
thread_local! {
    static PAYLOAD_DECRYPTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count of frame payloads this thread has decrypted, for tests that check a
/// reader never opens frames it does not need.
#[cfg(test)]
pub(crate) fn payload_decrypts_on_this_thread() -> u64 {
    PAYLOAD_DECRYPTS.with(|c| c.get())
}

#[inline]
fn note_payload_decrypt() {
    #[cfg(test)]
    PAYLOAD_DECRYPTS.with(|c| c.set(c.get() + 1));
}

/// Fill `buf` from `reader`, stopping early only at end of file. Returns the
/// number of bytes read: `buf.len()` for a full read, less at end of file.
fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut read = 0;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(read)
}

/// Walk an existing log file (if any) starting at `frame_region_start`
/// (immediately past the header, for versioned files; 0 for legacy) to
/// compute the next LSN, the byte offset immediately after the last valid
/// frame, and the frame index. A corrupt or truncated tail is reported via
/// the returned offset — open() does not truncate.
///
/// With a `hint` whose latest frame verifies, the walk starts there: the
/// frames before it were validated by the scan or append that recorded the
/// hint.
fn scan_for_tail(
    path: &Path,
    frame_region_start: u64,
    hint: Option<&FrameIndex>,
) -> Result<(Lsn, u64, FrameIndex), IntentLogError> {
    if !path.exists() {
        // No file yet → frames will start at frame_region_start once a
        // header is written. Report that as the durable end so the
        // caller's bookkeeping is consistent.
        return Ok((Lsn::ZERO, frame_region_start, FrameIndex::default()));
    }
    let file_len = std::fs::metadata(path)?.len();
    if file_len <= frame_region_start {
        // File exists but has nothing after the header — clean empty log.
        return Ok((Lsn::ZERO, frame_region_start, FrameIndex::default()));
    }

    // Tail scan only needs frame structure (lsn/offset/CRC), not payload content,
    // so it reads sealed payloads as-is (no key required).
    if let Some(pos) = hint.and_then(FrameIndex::latest) {
        if pos.offset >= frame_region_start && pos.offset < file_len {
            let mut iter = IntentLogIter::open(path, pos.offset, None)?;
            if iter.verify_and_rewind(pos)? {
                let mut index = hint.cloned().unwrap_or_default();
                index.keep_through(pos);
                return walk_to_tail(iter, index);
            }
            tracing::warn!(
                path = %path.display(),
                lsn = pos.lsn.0,
                offset = pos.offset,
                "intent log tail hint did not verify; scanning the whole log",
            );
        }
    }
    let iter = IntentLogIter::open(path, frame_region_start, None)?;
    walk_to_tail(iter, FrameIndex::default())
}

fn walk_to_tail(
    mut iter: IntentLogIter,
    mut index: FrameIndex,
) -> Result<(Lsn, u64, FrameIndex), IntentLogError> {
    let mut last_good_offset = iter.current_offset();
    loop {
        match iter.read_frame() {
            Ok(Some(frame)) => {
                index.record(frame.position());
                last_good_offset = iter.current_offset();
            }
            Ok(None) => break,
            // Stop at first corruption. `last_good_offset` is the truncate
            // target; `next_lsn` continues from the last *good* lsn we saw.
            Err(_) => break,
        }
    }
    let next_lsn = index.latest().map_or(Lsn::ZERO, |p| p.lsn.next());
    Ok((next_lsn, last_good_offset, index))
}

/// Whether the bytes past the last valid frame, `start..end`, are a torn
/// tail: what a write cut short by a crash or a full disk leaves. That is at
/// most one frame long and holds no complete frame. Anything else is a
/// corrupt frame with frames after it that callers were told are durable.
fn tail_is_torn(path: &Path, start: u64, end: u64, next_lsn: Lsn) -> io::Result<bool> {
    let len = end - start;
    if len > (FRAME_HEADER_BYTES + MAX_PAYLOAD_BYTES + FRAME_TRAILER_BYTES) as u64 {
        return Ok(false);
    }
    let mut tail = vec![0u8; len as usize];
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    file.read_exact(&mut tail)?;
    // A frame after the bad one carries a later LSN, and no more of them fit
    // than empty frames would; that window keeps the CRC checks to a few.
    let min_frame = (FRAME_HEADER_BYTES + FRAME_TRAILER_BYTES) as u64;
    let lsns = next_lsn.0..=next_lsn.0.saturating_add(len / min_frame);
    Ok(!(0..tail.len()).any(|at| frame_starts_at(&tail[at..], &lsns)))
}

/// Whether `bytes` begins with a complete, CRC-valid frame whose LSN is in `lsns`.
fn frame_starts_at(bytes: &[u8], lsns: &std::ops::RangeInclusive<u64>) -> bool {
    if bytes.len() < FRAME_HEADER_BYTES + FRAME_TRAILER_BYTES {
        return false;
    }
    let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let lsn = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let body = FRAME_HEADER_BYTES + len;
    if len > MAX_PAYLOAD_BYTES || !lsns.contains(&lsn) || bytes.len() < body + FRAME_TRAILER_BYTES {
        return false;
    }
    let stored = u32::from_le_bytes(bytes[body..body + FRAME_TRAILER_BYTES].try_into().unwrap());
    crc32fast::hash(&bytes[..body]) == stored
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
    probe_header(path).map(|(mode, _)| mode)
}

/// [`probe_mode`], with the base LSN the header names (0 when it names none).
fn probe_header(path: &Path) -> Result<(LogMode, u64), IntentLogError> {
    if !path.exists() {
        // Fresh file — the open() caller will write a header.
        return Ok((
            LogMode::Versioned {
                format_version: CURRENT_FORMAT_VERSION,
                // Placeholder; the real timestamp is stamped at write time.
                created_unix_secs: 0,
            },
            0,
        ));
    }
    let len = std::fs::metadata(path)?.len();
    if len == 0 {
        // Touched but never written — treat as fresh.
        return Ok((
            LogMode::Versioned {
                format_version: CURRENT_FORMAT_VERSION,
                created_unix_secs: 0,
            },
            0,
        ));
    }

    let mut file = File::open(path)?;
    let mut magic_buf = [0u8; 8];
    let n = read_full(&mut file, &mut magic_buf)?;

    // Too short to even hold the magic — treat as legacy (whoever owns
    // this file can take it from there).
    if n < 8 {
        return Ok((LogMode::Legacy, 0));
    }
    if &magic_buf != MAGIC {
        return Ok((LogMode::Legacy, 0));
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
    Ok((
        LogMode::Versioned {
            format_version: header.format_version,
            created_unix_secs: header.created_unix_secs,
        },
        header.base_lsn,
    ))
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
    fn sync_makes_frames_durable_at_either_durability() {
        let path = tmp_log_path("durability");
        let mut log = IntentLog::open(&path).unwrap();
        assert_eq!(
            log.durability(),
            crate::config::StorageTuning::process().durability
        );
        for durability in [StorageDurability::Eventual, StorageDurability::Immediate] {
            log.set_durability(durability);
            log.append(durability.as_str().as_bytes()).unwrap();
            log.sync().unwrap();
        }
        drop(log);
        let payloads: Vec<Vec<u8>> = IntentLog::open(&path)
            .unwrap()
            .iter()
            .unwrap()
            .map(|r| r.unwrap().payload)
            .collect();
        assert_eq!(payloads, vec![b"eventual".to_vec(), b"immediate".to_vec()]);
        let _ = std::fs::remove_file(&path);
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
    fn encrypted_journal_seals_on_disk_and_opens_on_read() {
        let path = tmp_log_path("enc");
        let cryptor = || {
            Some(crate::keystore::SecretCryptor::new(
                zeroize::Zeroizing::new([7u8; 32]),
            ))
        };
        const SECRET: &[u8] = b"sensitive-memory-snapshot-do-not-leak-into-the-journal";
        {
            let mut log = IntentLog::open_with_cryptor(&path, cryptor()).unwrap();
            log.append(SECRET).unwrap();
            log.append(b"second-frame").unwrap();
            log.sync().unwrap();
        }
        // On disk the plaintext must NOT appear and each frame payload is an
        // ENC\0 sealed blob.
        let raw = std::fs::read(&path).unwrap();
        assert!(
            !raw.windows(SECRET.len()).any(|w| w == SECRET),
            "plaintext payload leaked into the journal file"
        );
        assert!(
            raw.windows(4).any(|w| w == b"ENC\0"),
            "frame payload should be an ENC\\0 sealed blob"
        );
        // Reopen with the key: frames decrypt back to the original plaintext.
        let log = IntentLog::open_with_cryptor(&path, cryptor()).unwrap();
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].payload, SECRET);
        assert_eq!(records[1].payload, b"second-frame");
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

        // A scan sees two good records and a TruncatedFrame error.
        let mut iter = IntentLogIter::open_for_scan(&path).unwrap();
        let r0 = iter.next().unwrap().unwrap();
        assert_eq!(r0.payload, b"first");
        let r1 = iter.next().unwrap().unwrap();
        assert_eq!(r1.payload, b"second");
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, IntentLogError::TruncatedFrame { .. }));
        assert!(iter.next().is_none());
        drop(iter);

        // Open cuts the file to the end of the second good frame; cutting
        // again finds nothing more to remove.
        let mut log = IntentLog::open(&path).unwrap();
        let good_end = log.durable_end_offset();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good_end);
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

    /// The bytes `append` writes for `payload` at `lsn`, for hand-built logs.
    fn encode_frame(lsn: u64, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&lsn.to_le_bytes());
        frame.extend_from_slice(payload);
        let crc = crc32fast::hash(&frame);
        frame.extend_from_slice(&crc.to_le_bytes());
        frame
    }

    fn append_raw(path: &Path, bytes: &[u8]) {
        let mut f = OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_data().unwrap();
    }

    fn read_all(path: &Path) -> Vec<(u64, Vec<u8>)> {
        let log = IntentLog::open(path).unwrap();
        log.iter()
            .unwrap()
            .map(|r| r.map(|r| (r.lsn.0, r.payload)))
            .collect::<Result<_, _>>()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[test]
    fn a_failed_write_leaves_no_torn_frame_before_later_frames() {
        // Larger than the BufWriter's 8 KiB, so append itself reaches the file.
        const BIG: usize = 10_000;
        let frame = FRAME_HEADER_BYTES + BIG + FRAME_TRAILER_BYTES;
        // Where the disk fills: before the frame, inside and after len+lsn,
        // inside the payload, and before and inside the CRC. append buffers
        // the CRC, so the last two surface as a failed sync instead.
        let cuts = [
            0,
            5,
            FRAME_HEADER_BYTES,
            FRAME_HEADER_BYTES + 100,
            frame - FRAME_TRAILER_BYTES,
            frame - 1,
        ];
        for budget in cuts {
            let path = tmp_log_path(&format!("failed_write_{budget}"));
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"first").unwrap();
            log.sync().unwrap();

            log.writer.get_mut().write_budget = Some(budget);
            let failed = log.append(&[0xAB; BIG]).and_then(|_| log.sync());
            assert!(failed.is_err(), "cut at {budget}: the write should fail");
            log.writer.get_mut().write_budget = None;

            assert_eq!(log.append(b"second").unwrap(), Lsn(1), "cut at {budget}");
            log.sync().unwrap();
            assert_eq!(log.append(b"third").unwrap(), Lsn(2), "cut at {budget}");
            log.sync().unwrap();
            let end = log.durable_end_offset();
            drop(log);

            assert_eq!(
                read_all(&path),
                vec![
                    (0, b"first".to_vec()),
                    (1, b"second".to_vec()),
                    (2, b"third".to_vec())
                ],
                "cut at {budget}",
            );
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                end,
                "cut at {budget}"
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn a_failed_write_whose_cut_also_fails_blocks_appends_until_it_succeeds() {
        let path = tmp_log_path("cut_fails");
        let away = path.with_extension("away");
        let mut log = IntentLog::open(&path).unwrap();
        log.append(b"first").unwrap();
        log.sync().unwrap();

        // With the file moved, the writer's handle still reaches it but the
        // cut, which reopens the path, cannot.
        std::fs::rename(&path, &away).unwrap();
        log.writer.get_mut().write_budget = Some(5);
        assert!(log.append(&[0xAB; 10_000]).is_err());
        assert!(
            log.append(b"second").is_err(),
            "an append after an uncut failed write must not write"
        );
        assert!(log.sync().is_err());

        std::fs::rename(&away, &path).unwrap();
        assert_eq!(log.append(b"second").unwrap(), Lsn(1));
        log.sync().unwrap();
        drop(log);
        assert_eq!(
            read_all(&path),
            vec![(0, b"first".to_vec()), (1, b"second".to_vec())]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_cuts_a_torn_tail_so_the_next_append_is_readable() {
        // A frame cut short at each point: inside len+lsn, after them, and
        // inside the payload.
        for keep in [3, FRAME_HEADER_BYTES, FRAME_HEADER_BYTES + 4] {
            let path = tmp_log_path(&format!("torn_open_{keep}"));
            {
                let mut log = IntentLog::open(&path).unwrap();
                log.append(b"first").unwrap();
                log.sync().unwrap();
            }
            append_raw(&path, &encode_frame(1, b"never-finished")[..keep]);

            let mut log = IntentLog::open(&path).unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                log.durable_end_offset(),
                "keep {keep}: open should cut the torn frame",
            );
            assert_eq!(log.append(b"second").unwrap(), Lsn(1));
            log.sync().unwrap();
            drop(log);
            assert_eq!(
                read_all(&path),
                vec![(0, b"first".to_vec()), (1, b"second".to_vec())],
                "keep {keep}",
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn open_cuts_a_full_length_last_frame_that_never_reached_the_disk() {
        // After a crash the file can keep a frame's length while its blocks
        // read back as zeros; the frame is whole-sized but CRC-bad.
        let path = tmp_log_path("zeroed_tail");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"first").unwrap();
            log.sync().unwrap();
        }
        let mut zeroed = encode_frame(1, b"lost-in-the-crash");
        zeroed[FRAME_HEADER_BYTES..].fill(0);
        append_raw(&path, &zeroed);
        append_raw(&path, &[0u8; 40]);

        let mut log = IntentLog::open(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            log.durable_end_offset()
        );
        assert_eq!(log.append(b"second").unwrap(), Lsn(1));
        log.sync().unwrap();
        drop(log);
        assert_eq!(
            read_all(&path),
            vec![(0, b"first".to_vec()), (1, b"second".to_vec())]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_keeps_a_corrupt_frame_that_intact_frames_follow() {
        // What a failed write left before this was fixed: a torn frame, then
        // frames that were acknowledged as durable. Cutting at the torn frame
        // would drop them, so open must leave the file alone.
        let path = tmp_log_path("torn_mid_log");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"first").unwrap();
            log.sync().unwrap();
        }
        append_raw(
            &path,
            &encode_frame(1, &[0xAB; 10_000])[..FRAME_HEADER_BYTES],
        );
        append_raw(&path, &encode_frame(1, b"second"));
        append_raw(&path, &encode_frame(2, b"third"));
        let len_before = std::fs::metadata(&path).unwrap().len();

        let log = IntentLog::open(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len_before);
        let mut iter = log.iter().unwrap();
        assert_eq!(iter.next().unwrap().unwrap().payload, b"first");
        assert!(matches!(
            iter.next().unwrap().unwrap_err(),
            IntentLogError::TruncatedFrame { .. }
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn crc_corruption_detected() {
        let path = tmp_log_path("crc");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"untouched").unwrap();
            log.append(b"will-be-flipped").unwrap();
            log.append(b"after-the-flip").unwrap();
            log.sync().unwrap();
        }
        // Flip one byte inside the second payload.
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
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
        // An intact frame follows the flipped one, so open leaves the file
        // as it is.
        let len_before = std::fs::metadata(&path).unwrap().len();
        let log = IntentLog::open(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len_before);
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
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
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
        let lsns: Vec<_> = log.iter().unwrap().map(|r| r.unwrap().lsn).collect();
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

    // ========================================================================
    // Frame offset index
    // ========================================================================

    fn append_numbered(log: &mut IntentLog, count: u64) {
        for i in 0..count {
            log.append(&i.to_le_bytes()).unwrap();
        }
        log.sync().unwrap();
    }

    fn first_lsn(iter: &mut IntentLogIter) -> Lsn {
        iter.next().unwrap().unwrap().lsn
    }

    #[test]
    fn frame_index_encodes_round_trip_and_seeks_to_the_nearest_entry_below() {
        let mut index = FrameIndex::default();
        for lsn in [0u64, 1, 1023, 1024, 1500, 2048, 2050] {
            index.record(FramePosition {
                lsn: Lsn(lsn),
                offset: 100 + lsn * 10,
                crc: lsn as u32 ^ 0xabcd,
            });
        }
        let lsns: Vec<u64> = index.sparse().iter().map(|p| p.lsn.0).collect();
        assert_eq!(lsns, vec![0, 1024, 2048]);
        assert_eq!(index.latest().unwrap().lsn, Lsn(2050));
        assert_eq!(index.seek_hint(Lsn(1023)).unwrap().lsn, Lsn(0));
        assert_eq!(index.seek_hint(Lsn(1500)).unwrap().lsn, Lsn(1024));
        assert_eq!(index.seek_hint(Lsn(2049)).unwrap().lsn, Lsn(2048));
        assert_eq!(index.seek_hint(Lsn(9000)).unwrap().lsn, Lsn(2050));

        let decoded = FrameIndex::decode(&index.encode()).unwrap();
        assert_eq!(decoded, index);
        let mut truncated = index.encode();
        truncated.pop();
        assert!(FrameIndex::decode(&truncated).is_none());
        assert!(FrameIndex::decode(&[9, 0, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn iter_from_resumes_mid_log_at_the_indexed_frame() {
        let path = tmp_log_path("seek_mid");
        let mut log = IntentLog::open(&path).unwrap();
        append_numbered(&mut log, 2100);

        let mut iter = log.iter_from(Lsn(1500)).unwrap();
        assert_eq!(first_lsn(&mut iter), Lsn(1024), "seeks to the sparse entry");
        let mut iter = log.iter_from(Lsn(2099)).unwrap();
        let record = iter.next().unwrap().unwrap();
        assert_eq!(record.lsn, Lsn(2099), "seeks to the latest frame");
        assert_eq!(record.payload, 2099u64.to_le_bytes());
        assert!(iter.next().is_none());
        let mut iter = log.iter_from(Lsn(10)).unwrap();
        assert_eq!(first_lsn(&mut iter), Lsn(0));

        // The index rebuilt by a full scan on reopen matches the one the
        // appends built, and a reopen with it as the hint agrees too.
        let built = log.frame_index().clone();
        drop(log);
        let scanned = IntentLog::open(&path).unwrap();
        assert_eq!(scanned.frame_index(), &built);
        let hinted = IntentLog::open_with_index(&path, None, Some(built.clone())).unwrap();
        assert_eq!(hinted.next_lsn(), Lsn(2100));
        assert_eq!(hinted.durable_end_offset(), scanned.durable_end_offset());
        assert_eq!(hinted.frame_index(), &built);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn iter_from_falls_back_to_the_first_frame_on_a_corrupted_hint() {
        let path = tmp_log_path("seek_bad_hint");
        let mut log = IntentLog::open(&path).unwrap();
        append_numbered(&mut log, 2100);
        let good = log.index.seek_hint(Lsn(1500)).unwrap();

        for bad in [
            FramePosition {
                offset: good.offset + 1,
                ..good
            },
            FramePosition {
                crc: good.crc ^ 1,
                ..good
            },
            FramePosition {
                lsn: Lsn(1025),
                ..good
            },
        ] {
            let mut index = FrameIndex::default();
            index.record(bad);
            log.index = index;
            let lsns: Vec<u64> = log
                .iter_from(Lsn(1500))
                .unwrap()
                .map(|r| r.unwrap().lsn.0)
                .collect();
            assert_eq!(lsns, (0..2100).collect::<Vec<_>>(), "hint {bad:?}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_with_a_stale_hint_scans_the_whole_log() {
        let path = tmp_log_path("stale_hint");
        let mut log = IntentLog::open(&path).unwrap();
        append_numbered(&mut log, 2100);
        let stale = log.frame_index().clone();
        log.truncate_to_lsn(Lsn(1999)).unwrap();
        assert_eq!(log.frame_index().latest().unwrap().lsn, Lsn(1999));
        log.append(b"after-truncate").unwrap();
        log.sync().unwrap();
        let end = log.durable_end_offset();
        drop(log);

        let reopened = IntentLog::open_with_index(&path, None, Some(stale)).unwrap();
        assert_eq!(reopened.next_lsn(), Lsn(2001));
        assert_eq!(reopened.durable_end_offset(), end);
        assert_eq!(reopened.frame_index().latest().unwrap().lsn, Lsn(2000));
        let last = reopened
            .iter_from(Lsn(2000))
            .unwrap()
            .last()
            .unwrap()
            .unwrap();
        assert_eq!(last.payload, b"after-truncate");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn upgrade_shifts_the_frame_index_with_the_frames() {
        let path = tmp_log_path("upgrade_index");
        write_legacy_log(&path, &[b"one", b"two"]);
        let mut log = IntentLog::open(&path).unwrap();
        log.upgrade_unversioned_to_v1().unwrap();
        let latest = log.frame_index().latest().unwrap();
        let record = log
            .iter_at(latest)
            .unwrap()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(record.payload, b"two");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_frame_verifies_without_opening_sealed_payloads() {
        let path = tmp_log_path("raw_frames");
        let cryptor = || {
            Some(crate::keystore::SecretCryptor::new(
                zeroize::Zeroizing::new([3u8; 32]),
            ))
        };
        let mut log = IntentLog::open_with_cryptor(&path, cryptor()).unwrap();
        append_numbered(&mut log, 5);
        let before = payload_decrypts_on_this_thread();
        let mut iter = log.iter().unwrap();
        let mut seen = 0;
        while let Some(frame) = iter.read_frame().unwrap() {
            assert!(crate::keystore::SecretCryptor::is_encrypted(
                iter.sealed_payload()
            ));
            seen += 1;
            if frame.lsn == Lsn(3) {
                assert_eq!(iter.take_payload().unwrap(), 3u64.to_le_bytes());
            }
        }
        assert_eq!(seen, 5);
        assert_eq!(payload_decrypts_on_this_thread() - before, 1);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod segment_tests {
    use super::*;

    /// A log in its own directory that rotates whenever a sync leaves the head
    /// at `limit` bytes or more.
    fn rotating_log(dir: &tempfile::TempDir, limit: u64) -> IntentLog {
        let mut log = IntentLog::open(dir.path().join("intent.log")).unwrap();
        log.set_segment_bytes(Some(limit));
        log
    }

    fn append_synced(log: &mut IntentLog, count: u64) {
        for _ in 0..count {
            let lsn = log.next_lsn().0;
            log.append(format!("frame-{lsn}").as_bytes()).unwrap();
            log.sync().unwrap();
        }
    }

    fn lsns(iter: IntentLogIter) -> Vec<u64> {
        iter.map(|r| r.unwrap().lsn.0).collect()
    }

    /// Each frame here is 4 + 8 + len("frame-N") + 4 bytes; a 100-byte limit
    /// rotates every few frames.
    const LIMIT: u64 = 100;

    #[test]
    fn a_rotating_log_reads_back_every_frame_in_order_across_its_segments() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        let created = log.mode().created_unix_secs();
        append_synced(&mut log, 20);

        assert!(log.sealed().len() >= 3, "{} segments", log.sealed().len());
        let names: Vec<String> = log
            .sealed()
            .iter()
            .map(|s| s.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names[0], "intent.log.seg.00000000000000000000");
        assert_eq!(lsns(log.iter().unwrap()), (0..20).collect::<Vec<_>>());
        assert_eq!(
            log.mode().created_unix_secs(),
            created,
            "rotation keeps the log's identity"
        );

        drop(log);
        let reopened = IntentLog::open(dir.path().join("intent.log")).unwrap();
        assert_eq!(reopened.next_lsn(), Lsn(20));
        assert_eq!(reopened.mode().created_unix_secs(), created);
        assert_eq!(lsns(reopened.iter().unwrap()), (0..20).collect::<Vec<_>>());
        let records: Vec<IntentRecord> = reopened.iter().unwrap().map(Result::unwrap).collect();
        assert_eq!(records[7].payload, b"frame-7");
    }

    #[test]
    fn a_log_without_a_segment_size_never_rotates() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = IntentLog::open(dir.path().join("intent.log")).unwrap();
        append_synced(&mut log, 20);
        assert!(log.sealed().is_empty());
        assert_eq!(
            log_files(log.path()).unwrap(),
            vec![log.path().to_path_buf()]
        );
    }

    #[test]
    fn iter_from_and_iter_at_find_frames_in_sealed_segments() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        append_synced(&mut log, 3);
        let early = {
            let mut iter = log.iter().unwrap();
            iter.read_frame().unwrap();
            iter.read_frame().unwrap().unwrap().position()
        };
        append_synced(&mut log, 17);
        assert!(
            early.lsn < log.base_lsn(),
            "the position is now in a sealed segment"
        );

        let from: Vec<u64> = lsns(log.iter_from(Lsn(5)).unwrap())
            .into_iter()
            .filter(|lsn| *lsn >= 5)
            .collect();
        assert_eq!(from, (5..20).collect::<Vec<_>>());

        let at = log
            .iter_at(early)
            .unwrap()
            .expect("a position recorded before rotation still resolves");
        assert_eq!(lsns(at), (early.lsn.0..20).collect::<Vec<_>>());
    }

    #[test]
    fn a_missing_middle_segment_is_reported_not_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        append_synced(&mut log, 20);
        let middle = log.sealed()[1].path.clone();
        drop(log);
        std::fs::remove_file(&middle).unwrap();

        let log = IntentLog::open(dir.path().join("intent.log")).unwrap();
        let err = log
            .iter()
            .unwrap()
            .find_map(Result::err)
            .expect("the gap surfaces as an error");
        assert!(matches!(err, IntentLogError::SegmentGap { .. }), "{err}");
    }

    #[test]
    fn a_head_lost_between_the_renames_is_recreated_from_the_sealed_tail() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        append_synced(&mut log, 20);
        let created = log.mode().created_unix_secs();
        let head = log.path().to_path_buf();
        let (base, next) = (log.base_lsn(), log.next_lsn());
        drop(log);
        // What a crash between sealing the head and renaming the new one in
        // leaves: every frame sealed, and no head file.
        std::fs::rename(&head, segment_path(&head, base)).unwrap();

        let mut reopened = IntentLog::open(&head).unwrap();
        assert_eq!(reopened.next_lsn(), next);
        assert_eq!(reopened.base_lsn(), next);
        assert_eq!(reopened.mode().created_unix_secs(), created);
        reopened.append(b"after").unwrap();
        reopened.sync().unwrap();
        assert_eq!(
            lsns(reopened.iter().unwrap()),
            (0..=next.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn truncating_into_a_sealed_segment_leaves_a_hard_linked_copy_whole() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        append_synced(&mut log, 20);
        let first = log.sealed()[0].clone();
        let linked = dir.path().join("backup-of-first-segment");
        std::fs::hard_link(&first.path, &linked).unwrap();
        let linked_len = std::fs::metadata(&linked).unwrap().len();
        let target = Lsn(first.base.0 + 1);

        log.truncate_to_lsn(target).unwrap();

        assert!(log.sealed().is_empty());
        assert_eq!(log.next_lsn(), target.next());
        assert_eq!(
            lsns(log.iter().unwrap()),
            (0..=target.0).collect::<Vec<_>>()
        );
        assert!(sealed_segments(log.path()).unwrap().is_empty());
        assert_eq!(std::fs::metadata(&linked).unwrap().len(), linked_len);
        log.append(b"next").unwrap();
        log.sync().unwrap();
        assert_eq!(lsns(log.iter().unwrap()).last(), Some(&target.next().0));
    }

    #[test]
    fn a_rotation_that_fails_keeps_every_frame_and_waits_before_trying_again() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        let blocker = segment_path(log.path(), Lsn::ZERO);
        std::fs::write(&blocker, b"not a segment").unwrap();
        append_synced(&mut log, 6);
        assert!(
            log.sealed().is_empty(),
            "the name it would seal under is taken"
        );
        assert!(log.retry_rotation_at > log.durable_end_offset() - 30);

        std::fs::remove_file(&blocker).unwrap();
        append_synced(&mut log, 14);
        assert!(!log.sealed().is_empty(), "it rotates once the way is clear");
        assert_eq!(lsns(log.iter().unwrap()), (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn a_segment_left_past_the_head_is_set_aside_and_rotation_carries_on() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = rotating_log(&dir, LIMIT);
        append_synced(&mut log, 20);
        let target = Lsn(log.sealed()[1].base.0 + 1);
        let dropped = log.sealed()[2].path.clone();
        drop(log);
        // An interrupted truncation into segment 1: the head is already its
        // copy, the later segments still lie beside it.
        let head = dir.path().join("intent.log");
        std::fs::copy(segment_path(&head, Lsn(target.0 - 1)), &head).unwrap();

        let mut reopened = IntentLog::open(&head).unwrap();
        reopened.set_segment_bytes(Some(LIMIT));
        assert!(!dropped.exists());
        let mut aside = dropped.into_os_string();
        aside.push(".orphaned");
        assert!(std::path::Path::new(&aside).exists());
        assert!(reopened
            .sealed()
            .iter()
            .all(|s| s.base < reopened.base_lsn()));
        reopened.truncate_to_lsn(target).unwrap();
        append_synced(&mut reopened, 10);
        let read = lsns(reopened.iter().unwrap());
        assert_eq!(read, (0..=target.0 + 10).collect::<Vec<_>>());
    }

    #[test]
    fn segment_names_belong_to_their_log_only() {
        assert!(is_log_file("intent.log", "intent.log"));
        assert!(is_log_file(
            "intent.log",
            "intent.log.seg.00000000000000000042"
        ));
        assert_eq!(
            segment_base("intent.log", "intent.log.seg.00000000000000000042"),
            Some(Lsn(42))
        );
        for other in [
            "intent.log.pre_restore",
            "intent.log.seg.42",
            "intent.log.seg.00000000000000000042.sha256",
            "intent.log.rotate.tmp",
            "other.log.seg.00000000000000000042",
        ] {
            assert!(!is_log_file("intent.log", other), "{other}");
        }
    }
}
