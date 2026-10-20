//! Durable per-projection checkpoint storage.
//!
//! A [`CheckpointStore`] maps a projection name to the highest LSN that
//! projection has successfully applied. The persisted form is a tiny
//! single-file format — one record per `set()` call, with the latest entry
//! winning. On `load_all()` we replay the file front-to-back to build the
//! current state. The file is checksummed per-record using the same
//! `crc32fast` machinery the intent log uses, so a torn write to the
//! checkpoint file is detected and ignored (the projection just appears to
//! be one step behind, and the replay driver re-applies the missing record
//! — idempotency makes that safe).
//!
//! ## Why a separate file (not the intent log)
//!
//! Checkpoints are *projection metadata*, not data. Storing them in the
//! intent log would mean every checkpoint update writes a frame that every
//! other projection has to skip on replay — quadratic noise. Keeping
//! checkpoints in their own file lets projections persist them on whatever
//! cadence makes sense without polluting the log.
//!
//! ## File format
//!
//! ```text
//!   Record: [name_len: u16][name bytes][lsn: u64][crc32: u32]
//!     - LE encoding throughout
//!     - crc covers `[name_len][name][lsn]`
//!     - name_len capped at 256 (sanity cap)
//! ```
//!
//! On `load_all()`, parse records until EOF or first corrupt frame; the
//! last valid record for each name wins. On corruption, the offset of the
//! bad frame is returned so callers can truncate if desired (analogous to
//! `IntentLog::truncate_corrupt_tail`).
//!
//! ## Side-data extension
//!
//! Projections sometimes own derived state that is expensive to recompute
//! from the log on every restart — Vamana's `memory_id → vector_id` map is
//! the motivating example. Replay-from-scratch will eventually rebuild
//! that map, but for large logs the constant factor is painful. The
//! checkpoint store therefore offers a second persistence channel,
//! [`CheckpointStore::set_side_data`] / [`CheckpointStore::get_side_data`],
//! keyed by `(projection_name, key)`.
//!
//! Side data lives in a sibling file next to the checkpoint file:
//! `{checkpoint_path}.sidedata`. The checkpoint file's frame layout is
//! unchanged — existing on-disk stores keep working without migration. The
//! side-data file is its own CRC-framed log, scanned the same way: latest
//! record per `(projection, key)` wins, and a torn tail stops the scan
//! without contaminating earlier good values.
//!
//! ### Side-data frame format
//!
//! ```text
//!   Record:
//!     [proj_len: u16][proj bytes]
//!     [key_len:  u16][key  bytes]
//!     [ckpt_lsn: u64]            ← projection checkpoint LSN at write time
//!     [bytes_len: u32][bytes ...]
//!     [crc32: u32]
//!
//!   crc covers everything before it.
//!   proj_len / key_len capped at MAX_NAME_BYTES.
//!   bytes_len capped at MAX_SIDE_DATA_BYTES.
//! ```
//!
//! The `ckpt_lsn` stamped at write time lets the projection do its own
//! staleness check on load: if the persisted checkpoint and the side
//! data's stamp disagree, the side data is treated as stale and the
//! projection falls back to a full replay. This is the "defence in depth"
//! the W5 plan calls for — the checkpoint LSN is the canonical truth, and
//! side data is opportunistic acceleration.
//!
//! ### Side-data compaction
//!
//! Every `set_side_data` appends a whole new blob, so the file grows by the
//! full payload on each write while only the latest record per
//! `(projection, key)` is live. [`CheckpointStore::compact_side_data`]
//! rewrites the file down to those live records (temp file, fsync, rename,
//! directory fsync), so a crash before the rename leaves the old file
//! intact. It runs on open when the file holds far more garbage than live
//! bytes, and again after every [`SIDE_DATA_COMPACT_EVERY_WRITES`] synced
//! side-data writes when the same threshold holds.
//!
//! ### The intent log's own side data
//!
//! The projection name [`INTENT_LOG_SIDE_DATA`] is reserved for data the
//! log keeps about itself; no projection may use it, and it never gets a
//! checkpoint record. Two keys live under it:
//!
//! - [`FRAME_INDEX_KEY`]: the log's [`FrameIndex`] (offset hints), stamped
//!   with the LSN of its latest frame.
//! - [`LIVE_MEMORY_IDS_KEY`]: the reconcile scan's live-id snapshot, stamped
//!   with the LSN of the last frame it covers (see
//!   [`super::projection::live_memory_ids_resumable`]).
//!
//! Both are pure accelerators, verified against the log before use. A store
//! written before they existed simply lacks them, and an older binary reads
//! them as side data of a projection it does not have.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::{FrameIndex, Lsn};

const MAX_NAME_BYTES: usize = 256;

/// Projection name reserved for the intent log's own side data.
pub const INTENT_LOG_SIDE_DATA: &str = "__intent_log";

/// Side-data key of the persisted [`FrameIndex`].
pub const FRAME_INDEX_KEY: &str = "frame_index";

/// Side-data key of the reconcile scan's live-id snapshot.
pub const LIVE_MEMORY_IDS_KEY: &str = "live_memory_ids";

/// Hard cap on a single side-data blob. 64 MiB is far larger than any
/// realistic per-projection map (Vamana's `memory_id → u32` map sits at
/// ~50 bytes/entry, so 64 MiB ≈ 1.3 million memories) and small enough
/// that a bit-flipped length field can't request a wild allocation.
pub const MAX_SIDE_DATA_BYTES: usize = 64 * 1024 * 1024;

/// Compact the side-data file once it is this many times its live bytes.
const SIDE_DATA_COMPACT_RATIO: u64 = 4;

/// Below this size the side-data file is never compacted: rewriting a small
/// file buys nothing and would churn on every open.
const SIDE_DATA_COMPACT_MIN_BYTES: u64 = 1024 * 1024;

/// Superseded bytes beyond which the file is compacted whatever the ratio,
/// so a store with one huge live blob still sheds its dead copies.
const SIDE_DATA_COMPACT_MAX_GARBAGE_BYTES: u64 = 256 * 1024 * 1024;

/// Synced side-data writes between compaction checks. Each check is a
/// `metadata` call; the rewrite itself only happens past the thresholds.
pub const SIDE_DATA_COMPACT_EVERY_WRITES: u32 = 64;

/// The checkpoint file is compacted on open once it is this many times its live
/// records and past [`CHECKPOINT_COMPACT_MIN_BYTES`]: every `set` appends one.
const CHECKPOINT_COMPACT_RATIO: u64 = 16;
const CHECKPOINT_COMPACT_MIN_BYTES: u64 = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("projection name too long ({size} bytes; cap {cap})")]
    NameTooLong { size: usize, cap: usize },
    #[error("side-data key too long ({size} bytes; cap {cap})")]
    KeyTooLong { size: usize, cap: usize },
    #[error("side-data payload too large ({size} bytes; cap {cap})")]
    SideDataTooLarge { size: usize, cap: usize },
    #[error("checkpoint file is corrupt at offset {offset}")]
    Corrupt { offset: u64 },
}

/// One persisted side-data record's metadata + bytes. Kept in-memory as
/// the most-recent value for `(projection, key)`. The `ckpt_lsn` is the
/// projection's checkpoint LSN at write time — projections compare this
/// against their own checkpoint on load to detect a stale side-data file.
#[derive(Debug, Clone)]
struct SideDataEntry {
    ckpt_lsn: Lsn,
    bytes: Vec<u8>,
}

/// Durable checkpoint store. One file per store; many projections share it
/// (one is fine — there's no benefit to per-projection files since reads
/// scan the whole file anyway).
pub struct CheckpointStore {
    path: PathBuf,
    writer: BufWriter<File>,
    in_memory: HashMap<String, Lsn>,
    /// Byte offset of the last valid record's end, used by callers who want
    /// to truncate a corrupt tail.
    durable_end_offset: u64,
    /// Sibling side-data file. Lives at `{path}.sidedata`; opened lazily
    /// the first time a `set_side_data` write happens (so stores with no
    /// side-data users don't pay the open cost).
    side_data_path: PathBuf,
    /// Buffered writer for the side-data file. `Some` once the side-data
    /// file has been opened (read on construction, write-opened lazily on
    /// first `set_side_data`).
    side_data_writer: Option<BufWriter<File>>,
    /// In-memory snapshot of side data, keyed by `(projection, key)`.
    side_data: HashMap<(String, String), SideDataEntry>,
    /// Byte offset of the last valid side-data record's end.
    side_data_durable_end: u64,
    /// Side-data records written since the last compaction (or open).
    side_data_writes_since_compact: u32,
    /// The checkpoint map and file end as of the last successful sync (or open, cut or
    /// compaction). No caller was told a later record is durable, so a failed write or
    /// sync rolls back to here and cuts the file to `synced_end`.
    synced_checkpoints: HashMap<String, Lsn>,
    synced_end: u64,
    /// A failed write left bytes past `synced_end` that could not be cut yet. Every
    /// write retries the cut first, so no record lands behind a torn one.
    cut_pending: bool,
    /// The side-data file's end as of its last successful sync, and the entries each
    /// unsynced write replaced, newest last, for the same rollback.
    side_data_synced_end: u64,
    side_data_unsynced: Vec<((String, String), Option<SideDataEntry>)>,
    side_data_cut_pending: bool,
}

impl CheckpointStore {
    /// Open (create if missing) the checkpoint store at `path`. Scans the
    /// existing file to populate the in-memory map and to compute the
    /// durable-end offset.
    ///
    /// Bytes past the last valid record (a torn tail from a crash or a full
    /// disk mid-write) are cut, in both files: appends land at the end of the
    /// file, and behind unreadable bytes no later scan would reach them, so
    /// every restart would load the pre-tear checkpoints. The cut can only
    /// drop records a scan already cannot read; a checkpoint is metadata, so
    /// losing one costs a replay, never data.
    ///
    /// The sibling side-data file at `{path}.sidedata` is also scanned if
    /// it exists. A missing side-data file is not an error — the store
    /// simply has no side data and the first `set_side_data` call will
    /// create the file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CheckpointStoreError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let checkpoint_tmp = checkpoint_compact_tmp_path(&path);
        if checkpoint_tmp.exists() {
            std::fs::remove_file(&checkpoint_tmp)?;
        }
        let (in_memory, durable_end_offset) = scan(&path)?;
        cut_torn_tail(&path, durable_end_offset, "checkpoint")?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;

        let side_data_path = side_data_path_for(&path);
        // A temp file left by a compaction that crashed before its rename is
        // garbage: the live file it was replacing is still intact.
        let stale_tmp = side_data_compact_tmp_path(&side_data_path);
        if stale_tmp.exists() {
            std::fs::remove_file(&stale_tmp)?;
        }
        let (side_data, side_data_durable_end) = scan_side_data(&side_data_path)?;
        cut_torn_tail(&side_data_path, side_data_durable_end, "side-data")?;

        let mut store = Self {
            path,
            writer: BufWriter::new(file),
            synced_checkpoints: in_memory.clone(),
            in_memory,
            durable_end_offset,
            side_data_path,
            side_data_writer: None,
            side_data,
            side_data_durable_end,
            side_data_writes_since_compact: 0,
            synced_end: durable_end_offset,
            cut_pending: false,
            side_data_synced_end: side_data_durable_end,
            side_data_unsynced: Vec::new(),
            side_data_cut_pending: false,
        };
        if store.checkpoints_need_compaction() {
            if let Err(e) = store.compact() {
                tracing::warn!(
                    path = %store.path.display(),
                    error = %e,
                    "checkpoint store: compaction failed; the uncompacted file is intact"
                );
            }
        }
        if store.side_data_needs_compaction()? {
            store.compact_side_data_logged("open");
        }
        Ok(store)
    }

    /// Look up the most recently persisted checkpoint for `projection_name`.
    /// Returns `None` if the projection has never persisted one.
    pub fn get(&self, projection_name: &str) -> Option<Lsn> {
        self.in_memory.get(projection_name).copied()
    }

    /// Persist `lsn` as the checkpoint for `projection_name`. The write is
    /// buffered — callers that need durability before returning to user
    /// code must follow with `sync()`. Returns the resulting on-disk
    /// position so callers can optionally compact later.
    pub fn set(&mut self, projection_name: &str, lsn: Lsn) -> Result<(), CheckpointStoreError> {
        let name_bytes = projection_name.as_bytes();
        if name_bytes.len() > MAX_NAME_BYTES {
            return Err(CheckpointStoreError::NameTooLong {
                size: name_bytes.len(),
                cap: MAX_NAME_BYTES,
            });
        }
        if self.cut_pending {
            self.cut_unsynced_checkpoints()?;
        }
        let record = checkpoint_record(projection_name, lsn);
        if let Err(e) = self.writer.write_all(&record) {
            self.roll_back_checkpoints();
            return Err(e.into());
        }

        self.in_memory.insert(projection_name.to_string(), lsn);
        self.durable_end_offset += record.len() as u64;
        Ok(())
    }

    /// Persist arbitrary projection side data keyed by
    /// `(projection_name, key)`. The blob is recorded with the current
    /// checkpoint LSN (read from the in-memory checkpoint map for
    /// `projection_name`), letting callers detect on load whether the
    /// side data is stale relative to the projection checkpoint.
    ///
    /// Each call writes a fresh CRC-framed record to the side-data file.
    /// The write is buffered — call [`sync`](Self::sync) to make it
    /// durable. The in-memory snapshot updates immediately so subsequent
    /// `get_side_data` calls see the new bytes without a re-read.
    ///
    /// Returns an error if the projection name, key, or payload exceeds
    /// their respective hard caps; returns an `Io` error on disk
    /// failure. Either way, the in-memory snapshot is NOT updated when
    /// the write fails — the next `get_side_data` returns the
    /// previously-persisted value (or `None`).
    pub fn set_side_data(
        &mut self,
        projection_name: &str,
        key: &str,
        bytes: &[u8],
    ) -> Result<(), CheckpointStoreError> {
        // Stamp the current checkpoint LSN for staleness detection.
        // `Lsn::ZERO` for projections that have never checkpointed — the
        // load-time staleness check still works: a zero stamp matches only
        // a zero (or unset) projection checkpoint.
        let ckpt_lsn = self
            .in_memory
            .get(projection_name)
            .copied()
            .unwrap_or(Lsn::ZERO);
        self.set_side_data_at(projection_name, key, ckpt_lsn, bytes)
    }

    /// [`set_side_data`](Self::set_side_data) with an explicit stamp. A
    /// projection uses this to write the snapshot for the checkpoint it is
    /// ABOUT to persist, so the snapshot can be made durable before the
    /// checkpoint record that vouches for it is written.
    pub fn set_side_data_at(
        &mut self,
        projection_name: &str,
        key: &str,
        ckpt_lsn: Lsn,
        bytes: &[u8],
    ) -> Result<(), CheckpointStoreError> {
        let proj_bytes = projection_name.as_bytes();
        if proj_bytes.len() > MAX_NAME_BYTES {
            return Err(CheckpointStoreError::NameTooLong {
                size: proj_bytes.len(),
                cap: MAX_NAME_BYTES,
            });
        }
        let key_bytes = key.as_bytes();
        if key_bytes.len() > MAX_NAME_BYTES {
            return Err(CheckpointStoreError::KeyTooLong {
                size: key_bytes.len(),
                cap: MAX_NAME_BYTES,
            });
        }
        if bytes.len() > MAX_SIDE_DATA_BYTES {
            return Err(CheckpointStoreError::SideDataTooLarge {
                size: bytes.len(),
                cap: MAX_SIDE_DATA_BYTES,
            });
        }

        if self.side_data_cut_pending {
            self.cut_unsynced_side_data()?;
        }
        // Lazy-open the writer on first use so plain-checkpoint workloads
        // don't pay for a second file handle.
        let writer = self.ensure_side_data_writer()?;
        let record_size =
            match write_side_data_record(writer, projection_name, key, ckpt_lsn, bytes) {
                Ok(size) => size,
                Err(e) => {
                    self.roll_back_side_data();
                    return Err(e);
                }
            };

        self.side_data_durable_end += record_size;
        self.side_data_writes_since_compact = self.side_data_writes_since_compact.saturating_add(1);
        let slot = (projection_name.to_string(), key.to_string());
        let replaced = self.side_data.insert(
            slot.clone(),
            SideDataEntry {
                ckpt_lsn,
                bytes: bytes.to_vec(),
            },
        );
        self.side_data_unsynced.push((slot, replaced));
        Ok(())
    }

    /// Look up the most recently persisted side-data bytes for
    /// `(projection_name, key)`. Returns `None` if no record has ever
    /// been written for that pair, or if the most recent record was
    /// rejected during scan due to a torn tail / CRC mismatch.
    pub fn get_side_data(&self, projection_name: &str, key: &str) -> Option<&[u8]> {
        self.side_data
            .get(&(projection_name.to_string(), key.to_string()))
            .map(|entry| entry.bytes.as_slice())
    }

    /// Checkpoint LSN stamped into the most recent side-data record for
    /// `(projection_name, key)`. Returns `None` when no such record
    /// exists. Projections compare this against their own checkpoint
    /// LSN to decide whether the side data is fresh enough to trust.
    pub fn side_data_checkpoint(&self, projection_name: &str, key: &str) -> Option<Lsn> {
        self.side_data
            .get(&(projection_name.to_string(), key.to_string()))
            .map(|entry| entry.ckpt_lsn)
    }

    /// The intent log's persisted offset hints, if any were written and
    /// still decode.
    pub fn frame_index(&self) -> Option<FrameIndex> {
        self.get_side_data(INTENT_LOG_SIDE_DATA, FRAME_INDEX_KEY)
            .and_then(FrameIndex::decode)
    }

    /// Persist `index` as the intent log's offset hints and make it durable.
    /// A no-op for an empty index or one identical to what is stored.
    pub fn persist_frame_index(&mut self, index: &FrameIndex) -> Result<(), CheckpointStoreError> {
        let Some(latest) = index.latest() else {
            return Ok(());
        };
        let encoded = index.encode();
        if self.get_side_data(INTENT_LOG_SIDE_DATA, FRAME_INDEX_KEY) == Some(encoded.as_slice()) {
            return Ok(());
        }
        self.set_side_data_at(INTENT_LOG_SIDE_DATA, FRAME_INDEX_KEY, latest.lsn, &encoded)?;
        self.sync_side_data()
    }

    /// Flush + fsync. After this returns, every `set` since the last `sync`
    /// is durable. Also flushes the side-data file if it has been opened.
    pub fn sync(&mut self) -> Result<(), CheckpointStoreError> {
        self.sync_checkpoints()?;
        self.sync_side_data()
    }

    /// Flush + fsync only the checkpoint file, for a caller that has already
    /// made its side data durable (or wrote none).
    ///
    /// If it fails, every record `set` since the last successful sync is rolled back,
    /// in memory and on disk, so a checkpoint the caller was told failed cannot become
    /// durable with a later sync.
    pub fn sync_checkpoints(&mut self) -> Result<(), CheckpointStoreError> {
        if self.cut_pending {
            self.cut_unsynced_checkpoints()?;
        }
        let synced = self
            .writer
            .flush()
            .and_then(|()| self.writer.get_ref().sync_data());
        if let Err(e) = synced {
            self.roll_back_checkpoints();
            return Err(e.into());
        }
        self.synced_end = self.durable_end_offset;
        self.synced_checkpoints.clone_from(&self.in_memory);
        Ok(())
    }

    /// Forget every checkpoint record since the last successful sync and cut its
    /// bytes, after a write or sync failed part-way through them.
    fn roll_back_checkpoints(&mut self) {
        self.in_memory.clone_from(&self.synced_checkpoints);
        self.durable_end_offset = self.synced_end;
        self.cut_pending = true;
        if let Err(e) = self.cut_unsynced_checkpoints() {
            tracing::error!(
                path = %self.path.display(),
                error = %e,
                end_offset = self.synced_end,
                "checkpoint store could not cut a failed write back to the last synced \
                 record; writes are refused until it can"
            );
        }
    }

    fn cut_unsynced_checkpoints(&mut self) -> Result<(), CheckpointStoreError> {
        // Swap the handle before cutting so no later flush writes the stale bytes:
        // `into_parts` discards them, where dropping the BufWriter would flush them.
        let fresh = BufWriter::new(
            OpenOptions::new()
                .read(true)
                .append(true)
                .open(&self.path)?,
        );
        drop(std::mem::replace(&mut self.writer, fresh).into_parts());
        cut_to(&self.path, self.synced_end)?;
        self.cut_pending = false;
        Ok(())
    }

    /// Flush + fsync only the side-data file. A projection calls this to make
    /// a snapshot durable before writing the checkpoint record that refers to
    /// it. Every [`SIDE_DATA_COMPACT_EVERY_WRITES`] writes it also checks
    /// whether the file is worth compacting; a failed compaction is logged and
    /// leaves the (still valid) uncompacted file in place.
    ///
    /// A failed sync rolls back the side data written since the last one, as
    /// [`sync_checkpoints`](Self::sync_checkpoints) does for checkpoints.
    pub fn sync_side_data(&mut self) -> Result<(), CheckpointStoreError> {
        if self.side_data_cut_pending {
            self.cut_unsynced_side_data()?;
        }
        if let Some(side) = self.side_data_writer.as_mut() {
            if let Err(e) = side.flush().and_then(|()| side.get_ref().sync_data()) {
                self.roll_back_side_data();
                return Err(e.into());
            }
        }
        self.side_data_synced_end = self.side_data_durable_end;
        self.side_data_unsynced.clear();
        if self.side_data_writes_since_compact >= SIDE_DATA_COMPACT_EVERY_WRITES {
            self.side_data_writes_since_compact = 0;
            if self.side_data_needs_compaction()? {
                self.compact_side_data_logged("write cadence");
            }
        }
        Ok(())
    }

    /// Restore the entries the unsynced side-data writes replaced and cut their bytes,
    /// after a write or sync failed part-way through them.
    fn roll_back_side_data(&mut self) {
        for (slot, replaced) in self.side_data_unsynced.drain(..).rev() {
            match replaced {
                Some(entry) => {
                    self.side_data.insert(slot, entry);
                }
                None => {
                    self.side_data.remove(&slot);
                }
            }
        }
        self.side_data_durable_end = self.side_data_synced_end;
        self.side_data_cut_pending = true;
        if let Err(e) = self.cut_unsynced_side_data() {
            tracing::error!(
                path = %self.side_data_path.display(),
                error = %e,
                end_offset = self.side_data_synced_end,
                "checkpoint store could not cut a failed side-data write back to the last \
                 synced record; side-data writes are refused until it can"
            );
        }
    }

    fn cut_unsynced_side_data(&mut self) -> Result<(), CheckpointStoreError> {
        // The stale buffer is discarded, not flushed; the next write reopens lazily.
        if let Some(stale) = self.side_data_writer.take() {
            drop(stale.into_parts());
        }
        if self.side_data_path.exists() {
            cut_to(&self.side_data_path, self.side_data_synced_end)?;
        }
        self.side_data_cut_pending = false;
        Ok(())
    }

    /// Whether the checkpoint file is mostly superseded records: one is appended per
    /// `set`, and only the latest per projection is live.
    fn checkpoints_need_compaction(&self) -> bool {
        let live: u64 = self
            .in_memory
            .keys()
            .map(|name| checkpoint_record_len(name))
            .sum();
        self.durable_end_offset > CHECKPOINT_COMPACT_MIN_BYTES
            && self.durable_end_offset > live.saturating_mul(CHECKPOINT_COMPACT_RATIO)
    }

    /// Bytes the side-data file would occupy holding only the latest record
    /// per `(projection, key)`.
    pub fn side_data_live_bytes(&self) -> u64 {
        self.side_data
            .iter()
            .map(|((proj, key), entry)| side_data_record_len(proj, key, entry.bytes.len()))
            .sum()
    }

    /// Whether the on-disk side-data file carries enough superseded records
    /// to be worth rewriting.
    fn side_data_needs_compaction(&self) -> Result<bool, CheckpointStoreError> {
        let file_len = match std::fs::metadata(&self.side_data_path) {
            Ok(meta) => meta.len(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        if file_len <= SIDE_DATA_COMPACT_MIN_BYTES {
            return Ok(false);
        }
        let live = self.side_data_live_bytes();
        Ok(file_len > live.saturating_mul(SIDE_DATA_COMPACT_RATIO)
            || file_len.saturating_sub(live) > SIDE_DATA_COMPACT_MAX_GARBAGE_BYTES)
    }

    fn compact_side_data_logged(&mut self, trigger: &'static str) {
        let before = std::fs::metadata(&self.side_data_path)
            .map(|m| m.len())
            .unwrap_or(0);
        match self.compact_side_data() {
            Ok(()) => tracing::info!(
                path = %self.side_data_path.display(),
                trigger,
                bytes_before = before,
                bytes_after = self.side_data_durable_end,
                "checkpoint store: side-data file compacted"
            ),
            Err(e) => tracing::warn!(
                path = %self.side_data_path.display(),
                trigger,
                error = %e,
                "checkpoint store: side-data compaction failed; the uncompacted file is intact"
            ),
        }
    }

    /// Rewrite the side-data file keeping only the latest record per
    /// `(projection, key)`, each with its original checkpoint stamp.
    ///
    /// The records come from the in-memory snapshot, which already holds
    /// exactly the latest record per key (it is what a scan of the file plus
    /// every write since would produce), so the rewrite needs no memory beyond
    /// that snapshot and never re-reads the superseded bytes. The new file is
    /// written to a temp path and fsynced, then renamed over the live file and
    /// the directory fsynced: a crash before the rename leaves the old file
    /// intact (the temp is removed on the next open), a crash after it leaves
    /// the compacted one.
    pub fn compact_side_data(&mut self) -> Result<(), CheckpointStoreError> {
        if self.side_data_cut_pending {
            self.cut_unsynced_side_data()?;
        }
        if let Some(w) = self.side_data_writer.as_mut() {
            w.flush()?;
        }
        if !self.side_data_path.exists() {
            return Ok(());
        }

        let tmp_path = side_data_compact_tmp_path(&self.side_data_path);
        let mut entries: Vec<(&(String, String), &SideDataEntry)> = self.side_data.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut written = 0u64;
        {
            let mut tmp = BufWriter::new(File::create(&tmp_path)?);
            for ((proj, key), entry) in entries {
                written +=
                    write_side_data_record(&mut tmp, proj, key, entry.ckpt_lsn, &entry.bytes)?;
            }
            tmp.flush()?;
            tmp.get_ref().sync_data()?;
        }

        // Close the append handle before the rename (Windows refuses to
        // rename over an open file); the next write reopens lazily.
        self.side_data_writer = None;
        std::fs::rename(&tmp_path, &self.side_data_path)?;
        sync_parent_dir(&self.side_data_path)?;

        // The rewrite made every in-memory entry durable, unsynced ones included.
        self.side_data_durable_end = written;
        self.side_data_synced_end = written;
        self.side_data_unsynced.clear();
        self.side_data_writes_since_compact = 0;
        Ok(())
    }

    /// Path of the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Path of the sibling side-data file. Useful for diagnostics; do not
    /// write to it directly — go through [`set_side_data`](Self::set_side_data).
    pub fn side_data_path(&self) -> &Path {
        &self.side_data_path
    }

    /// Byte offset of the durable end (after the last fully-valid record).
    pub fn durable_end_offset(&self) -> u64 {
        self.durable_end_offset
    }

    /// Byte offset of the durable end in the side-data file (after the
    /// last fully-valid side-data record). Zero when the side-data file
    /// does not exist or holds only torn-tail garbage.
    pub fn side_data_durable_end_offset(&self) -> u64 {
        self.side_data_durable_end
    }

    /// Truncate the file to `durable_end_offset`. Used to chop a torn-tail
    /// record that `open()` detected. Destructive — only call after
    /// `open()` flagged corruption.
    pub fn truncate_corrupt_tail(&mut self) -> Result<u64, CheckpointStoreError> {
        self.writer.flush()?;
        let target = self.durable_end_offset;
        self.writer.get_ref().set_len(target)?;
        self.writer.get_ref().sync_data()?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        self.synced_end = target;
        self.synced_checkpoints.clone_from(&self.in_memory);
        Ok(target)
    }

    /// Truncate the side-data file to its last fully-valid record.
    /// Destructive — call only after a scan reports a torn tail (i.e.
    /// the on-disk file is longer than [`side_data_durable_end_offset`]).
    pub fn truncate_corrupt_side_data_tail(&mut self) -> Result<u64, CheckpointStoreError> {
        // Nothing to truncate if no side data has ever been written.
        if !self.side_data_path.exists() {
            return Ok(0);
        }
        if let Some(w) = self.side_data_writer.as_mut() {
            w.flush()?;
        }
        let target = self.side_data_durable_end;
        let trunc = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.side_data_path)?;
        trunc.set_len(target)?;
        trunc.sync_data()?;
        drop(trunc);
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.side_data_path)?;
        self.side_data_writer = Some(BufWriter::new(file));
        self.side_data_synced_end = target;
        self.side_data_unsynced.clear();
        Ok(target)
    }

    /// Rewrite the file as a compacted snapshot — one record per projection,
    /// in arbitrary order. Runs on open once superseded records dominate the
    /// file. Atomic: temp file, fsync, rename, directory fsync. After this,
    /// `durable_end_offset` is reset and every in-memory checkpoint is durable.
    pub fn compact(&mut self) -> Result<(), CheckpointStoreError> {
        if self.cut_pending {
            self.cut_unsynced_checkpoints()?;
        }
        self.writer.flush()?;

        // Write the compacted snapshot to a side file, fsync it, then
        // close every handle on the live file before renaming (Windows
        // refuses to rename over a file with an open append handle).
        let tmp_path = checkpoint_compact_tmp_path(&self.path);
        {
            let mut tmp = BufWriter::new(File::create(&tmp_path)?);
            for (name, lsn) in &self.in_memory {
                tmp.write_all(&checkpoint_record(name, *lsn))?;
            }
            tmp.flush()?;
            tmp.get_ref().sync_data()?;
        }

        // Re-point `self.writer` at the tmp file so the old append handle
        // on `self.path` is dropped. Then rename can proceed on Windows.
        let staged = OpenOptions::new().read(true).append(true).open(&tmp_path)?;
        self.writer = BufWriter::new(staged);

        std::fs::rename(&tmp_path, &self.path)?;
        sync_parent_dir(&self.path)?;

        // Reopen `self.writer` so subsequent appends go to the renamed
        // file by its current path (the staged handle still works on
        // most platforms, but reopening keeps semantics consistent
        // across OS variants).
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);

        let (_, end) = scan(&self.path)?;
        self.durable_end_offset = end;
        self.synced_end = end;
        self.synced_checkpoints.clone_from(&self.in_memory);
        Ok(())
    }

    /// Lazy-open the side-data writer. Idempotent — subsequent calls
    /// return the existing handle. Creates the file if it does not yet
    /// exist.
    fn ensure_side_data_writer(&mut self) -> Result<&mut BufWriter<File>, CheckpointStoreError> {
        if self.side_data_writer.is_none() {
            if let Some(parent) = self.side_data_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = OpenOptions::new()
                .read(true)
                .append(true)
                .create(true)
                .open(&self.side_data_path)?;
            self.side_data_writer = Some(BufWriter::new(file));
        }
        Ok(self.side_data_writer.as_mut().expect("just initialised"))
    }
}

/// One encoded checkpoint record. The name's cap is the caller's to enforce.
fn checkpoint_record(projection_name: &str, lsn: Lsn) -> Vec<u8> {
    let name_bytes = projection_name.as_bytes();
    let name_len = (name_bytes.len() as u16).to_le_bytes();
    let lsn_bytes = lsn.0.to_le_bytes();
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&name_len);
    hasher.update(name_bytes);
    hasher.update(&lsn_bytes);
    let crc = hasher.finalize();

    let mut record = Vec::with_capacity(checkpoint_record_len(projection_name) as usize);
    record.extend_from_slice(&name_len);
    record.extend_from_slice(name_bytes);
    record.extend_from_slice(&lsn_bytes);
    record.extend_from_slice(&crc.to_le_bytes());
    record
}

fn checkpoint_record_len(projection_name: &str) -> u64 {
    (2 + projection_name.len() + 8 + 4) as u64
}

/// Temp file a checkpoint-file compaction writes before renaming it into place.
fn checkpoint_compact_tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".compact.tmp");
    PathBuf::from(s)
}

/// Cut `path` back to `end` and make the cut durable.
fn cut_to(path: &Path, end: u64) -> Result<(), CheckpointStoreError> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    file.set_len(end)?;
    file.sync_data()?;
    Ok(())
}

/// Cut the bytes a scan could not read past `valid_end`, if the file has any.
fn cut_torn_tail(path: &Path, valid_end: u64, file_kind: &str) -> Result<(), CheckpointStoreError> {
    let len = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    if len > valid_end {
        tracing::warn!(
            path = %path.display(),
            file_kind,
            torn_bytes = len - valid_end,
            "checkpoint store: cutting bytes past the last valid record"
        );
        cut_to(path, valid_end)?;
    }
    Ok(())
}

/// Encoded length of one side-data record.
fn side_data_record_len(projection_name: &str, key: &str, payload_len: usize) -> u64 {
    (2 + projection_name.len() + 2 + key.len() + 8 + 4 + payload_len + 4) as u64
}

/// Append one CRC-framed side-data record to `w`; returns its length. Caps
/// are the caller's to enforce.
fn write_side_data_record<W: Write>(
    w: &mut W,
    projection_name: &str,
    key: &str,
    ckpt_lsn: Lsn,
    bytes: &[u8],
) -> Result<u64, CheckpointStoreError> {
    let proj_bytes = projection_name.as_bytes();
    let key_bytes = key.as_bytes();
    let proj_len = proj_bytes.len() as u16;
    let key_len = key_bytes.len() as u16;
    let bytes_len = bytes.len() as u32;

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&proj_len.to_le_bytes());
    hasher.update(proj_bytes);
    hasher.update(&key_len.to_le_bytes());
    hasher.update(key_bytes);
    hasher.update(&ckpt_lsn.0.to_le_bytes());
    hasher.update(&bytes_len.to_le_bytes());
    hasher.update(bytes);
    let crc = hasher.finalize();

    w.write_all(&proj_len.to_le_bytes())?;
    w.write_all(proj_bytes)?;
    w.write_all(&key_len.to_le_bytes())?;
    w.write_all(key_bytes)?;
    w.write_all(&ckpt_lsn.0.to_le_bytes())?;
    w.write_all(&bytes_len.to_le_bytes())?;
    w.write_all(bytes)?;
    w.write_all(&crc.to_le_bytes())?;
    Ok(side_data_record_len(projection_name, key, bytes.len()))
}

/// Temp file a side-data compaction writes before renaming it into place.
fn side_data_compact_tmp_path(side_data_path: &Path) -> PathBuf {
    let mut s = side_data_path.as_os_str().to_owned();
    s.push(".compact.tmp");
    PathBuf::from(s)
}

/// Make a rename inside `path`'s directory durable. Directories cannot be
/// opened as files on Windows, where the rename is durable on return.
fn sync_parent_dir(path: &Path) -> Result<(), CheckpointStoreError> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            File::open(dir)?.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Build the canonical side-data path next to a checkpoint file.
/// Centralised so on-disk layout stays consistent across open/test/CI.
fn side_data_path_for(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".sidedata");
    PathBuf::from(s)
}

fn scan(path: &Path) -> Result<(HashMap<String, Lsn>, u64), CheckpointStoreError> {
    let mut map = HashMap::new();
    if !path.exists() {
        return Ok((map, 0));
    }
    let mut file = io::BufReader::new(File::open(path)?);

    let mut offset = 0u64;
    loop {
        let record_start = offset;

        let mut name_len_bytes = [0u8; 2];
        if !read_exact_or_eof(&mut file, &mut name_len_bytes)? {
            // Clean EOF at record boundary.
            return Ok((map, offset));
        }
        let name_len = u16::from_le_bytes(name_len_bytes) as usize;
        if name_len > MAX_NAME_BYTES {
            // Wild length — torn tail or bit-flip. Stop here.
            return Ok((map, record_start));
        }

        let mut name_buf = vec![0u8; name_len];
        if !read_exact_full(&mut file, &mut name_buf)? {
            return Ok((map, record_start));
        }

        let mut lsn_bytes = [0u8; 8];
        if !read_exact_full(&mut file, &mut lsn_bytes)? {
            return Ok((map, record_start));
        }
        let lsn = Lsn(u64::from_le_bytes(lsn_bytes));

        let mut crc_bytes = [0u8; 4];
        if !read_exact_full(&mut file, &mut crc_bytes)? {
            return Ok((map, record_start));
        }
        let stored_crc = u32::from_le_bytes(crc_bytes);

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&name_len_bytes);
        hasher.update(&name_buf);
        hasher.update(&lsn_bytes);
        let computed = hasher.finalize();
        if stored_crc != computed {
            // Bit-flip detected. Stop and let caller decide whether to
            // truncate. Don't insert the bad record.
            return Ok((map, record_start));
        }

        let name = match String::from_utf8(name_buf) {
            Ok(s) => s,
            Err(_) => return Ok((map, record_start)),
        };
        map.insert(name, lsn);
        offset += (2 + name_len + 8 + 4) as u64;
    }
}

/// Scan the side-data file front-to-back. Latest record per
/// `(projection, key)` wins. A torn or CRC-bad tail terminates the scan
/// at the start of the bad record — earlier good records survive.
#[allow(clippy::type_complexity)] // local scan accumulator: (projection,key) -> entry
fn scan_side_data(
    path: &Path,
) -> Result<(HashMap<(String, String), SideDataEntry>, u64), CheckpointStoreError> {
    let mut map: HashMap<(String, String), SideDataEntry> = HashMap::new();
    if !path.exists() {
        return Ok((map, 0));
    }
    let mut file = io::BufReader::new(File::open(path)?);

    let mut offset = 0u64;
    loop {
        let record_start = offset;

        let mut proj_len_bytes = [0u8; 2];
        if !read_exact_or_eof(&mut file, &mut proj_len_bytes)? {
            return Ok((map, offset));
        }
        let proj_len = u16::from_le_bytes(proj_len_bytes) as usize;
        if proj_len > MAX_NAME_BYTES {
            return Ok((map, record_start));
        }

        let mut proj_buf = vec![0u8; proj_len];
        if !read_exact_full(&mut file, &mut proj_buf)? {
            return Ok((map, record_start));
        }

        let mut key_len_bytes = [0u8; 2];
        if !read_exact_full(&mut file, &mut key_len_bytes)? {
            return Ok((map, record_start));
        }
        let key_len = u16::from_le_bytes(key_len_bytes) as usize;
        if key_len > MAX_NAME_BYTES {
            return Ok((map, record_start));
        }

        let mut key_buf = vec![0u8; key_len];
        if !read_exact_full(&mut file, &mut key_buf)? {
            return Ok((map, record_start));
        }

        let mut ckpt_bytes = [0u8; 8];
        if !read_exact_full(&mut file, &mut ckpt_bytes)? {
            return Ok((map, record_start));
        }
        let ckpt_lsn = Lsn(u64::from_le_bytes(ckpt_bytes));

        let mut bytes_len_bytes = [0u8; 4];
        if !read_exact_full(&mut file, &mut bytes_len_bytes)? {
            return Ok((map, record_start));
        }
        let bytes_len = u32::from_le_bytes(bytes_len_bytes) as usize;
        if bytes_len > MAX_SIDE_DATA_BYTES {
            return Ok((map, record_start));
        }

        let mut payload_buf = vec![0u8; bytes_len];
        if bytes_len > 0 && !read_exact_full(&mut file, &mut payload_buf)? {
            return Ok((map, record_start));
        }

        let mut crc_bytes = [0u8; 4];
        if !read_exact_full(&mut file, &mut crc_bytes)? {
            return Ok((map, record_start));
        }
        let stored_crc = u32::from_le_bytes(crc_bytes);

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&proj_len_bytes);
        hasher.update(&proj_buf);
        hasher.update(&key_len_bytes);
        hasher.update(&key_buf);
        hasher.update(&ckpt_bytes);
        hasher.update(&bytes_len_bytes);
        hasher.update(&payload_buf);
        let computed = hasher.finalize();
        if stored_crc != computed {
            return Ok((map, record_start));
        }

        let proj = match String::from_utf8(proj_buf) {
            Ok(s) => s,
            Err(_) => return Ok((map, record_start)),
        };
        let key = match String::from_utf8(key_buf) {
            Ok(s) => s,
            Err(_) => return Ok((map, record_start)),
        };

        map.insert(
            (proj, key),
            SideDataEntry {
                ckpt_lsn,
                bytes: payload_buf,
            },
        );

        offset += (2 + proj_len + 2 + key_len + 8 + 4 + bytes_len + 4) as u64;
    }
}

/// Returns `Ok(true)` if `buf` was fully filled; `Ok(false)` if EOF was
/// hit before the read completed (either clean — zero bytes read — or
/// torn — some bytes then EOF). Both forms mean "stop scanning here" for
/// our purposes, so they are folded into one return value.
fn read_exact_or_eof(file: &mut impl Read, buf: &mut [u8]) -> Result<bool, CheckpointStoreError> {
    read_exact_full(file, buf)
}

fn read_exact_full(file: &mut impl Read, buf: &mut [u8]) -> Result<bool, CheckpointStoreError> {
    let mut read = 0;
    while read < buf.len() {
        match file.read(&mut buf[read..]) {
            Ok(0) => return Ok(false),
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("veld-checkpoint-{name}-{pid}-{stamp}.bin"));
        p
    }

    #[test]
    fn empty_open_returns_no_checkpoints() {
        let path = tmp_path("empty");
        let store = CheckpointStore::open(&path).unwrap();
        assert!(store.get("anything").is_none());
        assert_eq!(store.durable_end_offset(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn set_get_round_trip() {
        let path = tmp_path("rt");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("vamana", Lsn(42)).unwrap();
            s.set("bm25", Lsn(100)).unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get("vamana"), Some(Lsn(42)));
        assert_eq!(s.get("bm25"), Some(Lsn(100)));
        assert_eq!(s.get("missing"), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn latest_record_for_a_name_wins() {
        let path = tmp_path("latest");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("vamana", Lsn(1)).unwrap();
            s.set("vamana", Lsn(2)).unwrap();
            s.set("vamana", Lsn(3)).unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get("vamana"), Some(Lsn(3)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn name_too_long_rejected() {
        let path = tmp_path("toolong");
        let mut s = CheckpointStore::open(&path).unwrap();
        let huge = "x".repeat(MAX_NAME_BYTES + 1);
        let err = s.set(&huge, Lsn(0)).unwrap_err();
        assert!(matches!(err, CheckpointStoreError::NameTooLong { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compact_yields_one_record_per_name() {
        let path = tmp_path("compact");
        let mut s = CheckpointStore::open(&path).unwrap();
        s.set("a", Lsn(1)).unwrap();
        s.set("a", Lsn(2)).unwrap();
        s.set("a", Lsn(3)).unwrap();
        s.set("b", Lsn(10)).unwrap();
        s.set("a", Lsn(4)).unwrap();
        s.sync().unwrap();

        let pre_size = std::fs::metadata(&path).unwrap().len();
        s.compact().unwrap();
        let post_size = std::fs::metadata(&path).unwrap().len();
        assert!(post_size < pre_size, "compact should shrink the file");

        // Values survive compaction.
        let s2 = CheckpointStore::open(&path).unwrap();
        assert_eq!(s2.get("a"), Some(Lsn(4)));
        assert_eq!(s2.get("b"), Some(Lsn(10)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_tail_does_not_corrupt_earlier_records() {
        let path = tmp_path("torn");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("vamana", Lsn(5)).unwrap();
            s.sync().unwrap();
        }
        // Append garbage that looks like a partial record.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[5u8, 0]).unwrap(); // name_len=5, but no payload
            f.sync_data().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        // Earlier record survives; corrupt tail ignored.
        assert_eq!(s.get("vamana"), Some(Lsn(5)));
        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // Side-data tests
    // ------------------------------------------------------------------

    #[test]
    fn side_data_round_trip() {
        let path = tmp_path("sd_rt");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("vamana-text-primary", Lsn(42)).unwrap();
            s.set_side_data("vamana-text-primary", "id_map", b"hello")
                .unwrap();
            s.set_side_data("vamana-image", "id_map", b"world").unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(
            s.get_side_data("vamana-text-primary", "id_map"),
            Some(&b"hello"[..])
        );
        assert_eq!(
            s.get_side_data("vamana-image", "id_map"),
            Some(&b"world"[..])
        );
        assert_eq!(s.get_side_data("vamana-image", "missing"), None);
        // Stamp recorded with the projection's current checkpoint LSN.
        assert_eq!(
            s.side_data_checkpoint("vamana-text-primary", "id_map"),
            Some(Lsn(42))
        );
        // vamana-image had no checkpoint yet → side data stamped with ZERO.
        assert_eq!(
            s.side_data_checkpoint("vamana-image", "id_map"),
            Some(Lsn::ZERO)
        );
        let side = s.side_data_path().to_path_buf();
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side);
    }

    #[test]
    fn side_data_latest_wins() {
        let path = tmp_path("sd_latest");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("p", Lsn(1)).unwrap();
            s.set_side_data("p", "k", b"v1").unwrap();
            s.set("p", Lsn(2)).unwrap();
            s.set_side_data("p", "k", b"v2").unwrap();
            s.set("p", Lsn(3)).unwrap();
            s.set_side_data("p", "k", b"v3").unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("p", "k"), Some(&b"v3"[..]));
        assert_eq!(s.side_data_checkpoint("p", "k"), Some(Lsn(3)));
        let side = s.side_data_path().to_path_buf();
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side);
    }

    #[test]
    fn side_data_corrupt_tail_does_not_corrupt_earlier_records() {
        let path = tmp_path("sd_torn");
        let side_path;
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("p", Lsn(7)).unwrap();
            s.set_side_data("p", "k1", b"good-one").unwrap();
            s.set_side_data("p", "k2", b"good-two").unwrap();
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();
        }
        // Append a truncated record to the side-data file: a valid-looking
        // proj_len header but no payload behind it.
        {
            let mut f = OpenOptions::new().append(true).open(&side_path).unwrap();
            // proj_len = 4, then 4 bytes of "proj" - but stop before key_len.
            f.write_all(&[4u8, 0]).unwrap();
            f.write_all(b"proj").unwrap();
            f.sync_data().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("p", "k1"), Some(&b"good-one"[..]));
        assert_eq!(s.get_side_data("p", "k2"), Some(&b"good-two"[..]));
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn side_data_crc_corruption_is_detected() {
        let path = tmp_path("sd_crc");
        let side_path;
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("p", Lsn(1)).unwrap();
            s.set_side_data("p", "k", b"untouched").unwrap();
            s.set_side_data("p", "k", b"will-be-flipped").unwrap();
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();
        }
        // Flip a byte deep inside the side-data file. The second record's
        // payload starts after the first record. Easier: flip the final
        // byte just before the trailing CRC of the *last* record. The CRC
        // covers the payload, so flipping the payload invalidates it.
        let len = std::fs::metadata(&side_path).unwrap().len();
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&side_path)
                .unwrap();
            // Position 5 bytes before EOF lands inside the last record's
            // payload (CRC is the last 4 bytes; payload sits above it).
            f.seek(SeekFrom::Start(len - 5)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xFF;
            f.seek(SeekFrom::Start(len - 5)).unwrap();
            f.write_all(&b).unwrap();
            f.sync_data().unwrap();
        }
        // On reopen the bad-CRC tail is dropped; the earlier good record
        // for the same key survives.
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("p", "k"), Some(&b"untouched"[..]));
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn side_data_too_long_rejected() {
        let path = tmp_path("sd_too_long");
        let mut s = CheckpointStore::open(&path).unwrap();
        let huge_key = "k".repeat(MAX_NAME_BYTES + 1);
        let err = s.set_side_data("p", &huge_key, b"v").unwrap_err();
        assert!(matches!(err, CheckpointStoreError::KeyTooLong { .. }));
        let huge_proj = "p".repeat(MAX_NAME_BYTES + 1);
        let err = s.set_side_data(&huge_proj, "k", b"v").unwrap_err();
        assert!(matches!(err, CheckpointStoreError::NameTooLong { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn side_data_persists_across_reopen_without_checkpoint() {
        // A projection might persist side data before ever calling `set`
        // (e.g. it stamps a bootstrap snapshot at startup). The store
        // must still round-trip the bytes — the ckpt_lsn stamp is just
        // Lsn::ZERO in that case.
        let path = tmp_path("sd_no_ckpt");
        let side_path;
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set_side_data("solo", "bootstrap", b"payload").unwrap();
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("solo", "bootstrap"), Some(&b"payload"[..]));
        assert_eq!(s.side_data_checkpoint("solo", "bootstrap"), Some(Lsn::ZERO));
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn compaction_keeps_the_latest_record_per_key_and_its_stamp() {
        let path = tmp_path("sd_compact");
        let side_path;
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            for lsn in 1..=10u64 {
                s.set_side_data_at("p", "map", Lsn(lsn), format!("map-{lsn}").as_bytes())
                    .unwrap();
                s.set_side_data_at("q", "map", Lsn(lsn * 100), b"q-blob")
                    .unwrap();
            }
            s.set_side_data_at("p", "other", Lsn(3), b"other").unwrap();
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();

            let before = std::fs::metadata(&side_path).unwrap().len();
            s.compact_side_data().unwrap();
            let after = std::fs::metadata(&side_path).unwrap().len();
            assert!(after < before, "compaction must shrink the file");
            assert_eq!(after, s.side_data_live_bytes());
            assert_eq!(after, s.side_data_durable_end_offset());

            // Writes after a compaction append to the compacted file.
            s.set_side_data_at("p", "map", Lsn(11), b"map-11").unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("p", "map"), Some(&b"map-11"[..]));
        assert_eq!(s.side_data_checkpoint("p", "map"), Some(Lsn(11)));
        assert_eq!(s.get_side_data("q", "map"), Some(&b"q-blob"[..]));
        assert_eq!(s.side_data_checkpoint("q", "map"), Some(Lsn(1000)));
        assert_eq!(s.get_side_data("p", "other"), Some(&b"other"[..]));
        assert_eq!(s.side_data_checkpoint("p", "other"), Some(Lsn(3)));
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn a_compaction_interrupted_before_its_rename_leaves_the_old_file() {
        let path = tmp_path("sd_compact_crash");
        let side_path;
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set_side_data_at("p", "map", Lsn(1), b"v1").unwrap();
            s.set_side_data_at("p", "map", Lsn(2), b"v2").unwrap();
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();
        }
        // What a crash between the temp write and the rename leaves behind.
        let tmp = side_data_compact_tmp_path(&side_path);
        std::fs::write(&tmp, b"half-written compaction").unwrap();

        let s = CheckpointStore::open(&path).unwrap();
        assert!(!tmp.exists(), "the stale temp is removed on open");
        assert_eq!(s.get_side_data("p", "map"), Some(&b"v2"[..]));
        assert_eq!(s.side_data_checkpoint("p", "map"), Some(Lsn(2)));
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn open_compacts_a_side_data_file_mostly_made_of_superseded_records() {
        let path = tmp_path("sd_compact_open");
        let side_path;
        let blob = vec![7u8; 128 * 1024];
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            // Fewer writes than the cadence check, so only open compacts.
            for lsn in 0..(SIDE_DATA_COMPACT_EVERY_WRITES as u64 / 2) {
                s.set_side_data_at("p", "map", Lsn(lsn), &blob).unwrap();
            }
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();
        }
        let grown = std::fs::metadata(&side_path).unwrap().len();
        assert!(grown > SIDE_DATA_COMPACT_MIN_BYTES);

        let s = CheckpointStore::open(&path).unwrap();
        let compacted = std::fs::metadata(&side_path).unwrap().len();
        assert_eq!(compacted, s.side_data_live_bytes());
        assert!(compacted * SIDE_DATA_COMPACT_RATIO < grown);
        assert_eq!(s.get_side_data("p", "map"), Some(&blob[..]));
        assert_eq!(
            s.side_data_checkpoint("p", "map"),
            Some(Lsn(SIDE_DATA_COMPACT_EVERY_WRITES as u64 / 2 - 1))
        );
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn write_cadence_compacts_without_a_reopen() {
        let path = tmp_path("sd_compact_cadence");
        let blob = vec![9u8; 64 * 1024];
        let mut s = CheckpointStore::open(&path).unwrap();
        for lsn in 0..SIDE_DATA_COMPACT_EVERY_WRITES as u64 {
            s.set_side_data_at("p", "map", Lsn(lsn), &blob).unwrap();
            s.sync().unwrap();
        }
        let side_path = s.side_data_path().to_path_buf();
        assert_eq!(
            std::fs::metadata(&side_path).unwrap().len(),
            s.side_data_live_bytes(),
            "the cadence check compacts once the file is mostly garbage"
        );
        assert_eq!(s.get_side_data("p", "map"), Some(&blob[..]));
        drop(s);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn a_torn_tail_is_cut_on_open_so_later_checkpoints_survive_a_reopen() {
        let path = tmp_path("torn_cut");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set("vamana", Lsn(5)).unwrap();
            s.sync().unwrap();
        }
        let valid_end = std::fs::metadata(&path).unwrap().len();
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[6u8, 0, b'v']).unwrap();
            f.sync_data().unwrap();
        }
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_end);
            s.set("vamana", Lsn(9)).unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(
            s.get("vamana"),
            Some(Lsn(9)),
            "a checkpoint written after the tear was appended behind it"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_torn_side_data_tail_is_cut_on_open() {
        let path = tmp_path("sd_torn_cut");
        let side_path;
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set_side_data("p", "k", b"before").unwrap();
            s.sync().unwrap();
            side_path = s.side_data_path().to_path_buf();
        }
        {
            let mut f = OpenOptions::new().append(true).open(&side_path).unwrap();
            f.write_all(&[1u8, 0, b'p']).unwrap();
            f.sync_data().unwrap();
        }
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            s.set_side_data("p", "k", b"after").unwrap();
            s.sync().unwrap();
        }
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("p", "k"), Some(&b"after"[..]));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn a_failed_sync_rolls_back_to_the_last_synced_checkpoints() {
        let path = tmp_path("rollback");
        let mut s = CheckpointStore::open(&path).unwrap();
        s.set("a", Lsn(1)).unwrap();
        s.sync().unwrap();
        let synced_len = std::fs::metadata(&path).unwrap().len();

        // The record reached the file, then the sync failed: what ENOSPC mid-flush leaves.
        s.set("a", Lsn(2)).unwrap();
        s.set("b", Lsn(7)).unwrap();
        s.writer.flush().unwrap();
        s.roll_back_checkpoints();

        assert_eq!(s.get("a"), Some(Lsn(1)));
        assert_eq!(s.get("b"), None);
        assert_eq!(s.durable_end_offset(), synced_len);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), synced_len);

        s.set("a", Lsn(3)).unwrap();
        s.sync().unwrap();
        drop(s);
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get("a"), Some(Lsn(3)));
        assert_eq!(s.get("b"), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_failed_side_data_sync_restores_the_replaced_entries() {
        let path = tmp_path("sd_rollback");
        let mut s = CheckpointStore::open(&path).unwrap();
        s.set_side_data("p", "kept", b"v1").unwrap();
        s.sync().unwrap();
        let side_path = s.side_data_path().to_path_buf();
        let synced_len = std::fs::metadata(&side_path).unwrap().len();

        s.set_side_data("p", "kept", b"v2").unwrap();
        s.set_side_data("p", "new", b"n1").unwrap();
        s.side_data_writer.as_mut().unwrap().flush().unwrap();
        s.roll_back_side_data();

        assert_eq!(s.get_side_data("p", "kept"), Some(&b"v1"[..]));
        assert_eq!(s.get_side_data("p", "new"), None);
        assert_eq!(std::fs::metadata(&side_path).unwrap().len(), synced_len);

        s.set_side_data("p", "kept", b"v3").unwrap();
        s.sync().unwrap();
        drop(s);
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get_side_data("p", "kept"), Some(&b"v3"[..]));
        assert_eq!(s.get_side_data("p", "new"), None);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&side_path);
    }

    #[test]
    fn open_compacts_a_checkpoint_file_of_superseded_records() {
        let path = tmp_path("ckpt_compact");
        {
            let mut s = CheckpointStore::open(&path).unwrap();
            for lsn in 0..10_000u64 {
                s.set("vamana-text-primary", Lsn(lsn)).unwrap();
            }
            s.set("bm25", Lsn(42)).unwrap();
            s.sync().unwrap();
        }
        let before = std::fs::metadata(&path).unwrap().len();
        let s = CheckpointStore::open(&path).unwrap();
        let after = std::fs::metadata(&path).unwrap().len();
        assert!(after < before / 100, "{before} -> {after}");
        assert_eq!(s.get("vamana-text-primary"), Some(Lsn(9_999)));
        assert_eq!(s.get("bm25"), Some(Lsn(42)));
        drop(s);
        let s = CheckpointStore::open(&path).unwrap();
        assert_eq!(s.get("vamana-text-primary"), Some(Lsn(9_999)));
        let _ = std::fs::remove_file(&path);
    }
}
