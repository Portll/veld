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
//! ## Frame format
//!
//! Each record is stored as a frame:
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
pub mod journal;
pub mod payload;
pub mod projection;
pub use checkpoint_store::{CheckpointStore, CheckpointStoreError};
pub use journal::{ApplyError, JournalError, JournaledWriter, TypedProjection, WriteOutcome};
pub use payload::{IntentPayload, PayloadError};
pub use projection::{replay, Projection, ReplayError};

/// A log sequence number — the position assigned to a record at append time.
///
/// LSNs are monotonically increasing within one log file. They are *not*
/// byte offsets; they are abstract counters so that a future segment-rolling
/// implementation can keep LSNs continuous across files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// extra fstat.
    durable_end_offset: u64,
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

        // Discover next_lsn + durable_end by scanning whatever is already
        // there. A reader walks frames until EOF or a corruption signal.
        let (next_lsn, durable_end) = scan_for_tail(&path)?;

        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;

        // Publish the head-of-log gauges so dashboards see a value immediately
        // after process start, without waiting for the first append.
        INTENT_LOG_NEXT_LSN.set(next_lsn.0 as i64);
        INTENT_LOG_DURABLE_END_OFFSET_BYTES.set(durable_end as i64);

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            next_lsn,
            durable_end_offset: durable_end,
        })
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

    /// Iterate every frame from the start of the file. Stops at the first
    /// corrupt or truncated frame; returns an `IntentLogError` for that
    /// frame so the caller can decide whether to truncate. The iterator is
    /// independent of the writer — it opens its own read handle.
    pub fn iter(&self) -> Result<IntentLogIter, IntentLogError> {
        IntentLogIter::open(&self.path)
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
        self.writer.get_ref().set_len(target)?;
        self.writer.get_ref().sync_data()?;
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
    fn open(path: &Path) -> Result<Self, IntentLogError> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(0))?;
        Ok(Self {
            file,
            offset: 0,
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

/// Walk an existing log file (if any) to compute the next LSN and the byte
/// offset immediately after the last valid frame. A corrupt or truncated
/// tail is reported via the returned offset — open() does not truncate.
fn scan_for_tail(path: &Path) -> Result<(Lsn, u64), IntentLogError> {
    if !path.exists() {
        return Ok((Lsn::ZERO, 0));
    }
    let mut iter = IntentLogIter::open(path)?;
    let mut last_lsn = None;
    let mut last_good_offset = 0u64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

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
        assert_eq!(log.durable_end_offset(), 0);
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
            // First frame is 4 + 8 + 9 + 4 = 25 bytes.
            // Second frame starts at offset 25: len(4) + lsn(8) + payload(15) + crc(4).
            // Flip the first byte of the second payload, which is at offset 25 + 12 = 37.
            f.seek(SeekFrom::Start(37)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xff;
            f.seek(SeekFrom::Start(37)).unwrap();
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
}
