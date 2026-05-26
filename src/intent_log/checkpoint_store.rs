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

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::Lsn;

const MAX_NAME_BYTES: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("projection name too long ({size} bytes; cap {cap})")]
    NameTooLong { size: usize, cap: usize },
    #[error("checkpoint file is corrupt at offset {offset}")]
    Corrupt { offset: u64 },
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
}

impl CheckpointStore {
    /// Open (create if missing) the checkpoint store at `path`. Scans the
    /// existing file to populate the in-memory map and to compute the
    /// durable-end offset. A corrupt tail is *not* auto-truncated.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CheckpointStoreError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let (in_memory, durable_end_offset) = scan(&path)?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            in_memory,
            durable_end_offset,
        })
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
    pub fn set(
        &mut self,
        projection_name: &str,
        lsn: Lsn,
    ) -> Result<(), CheckpointStoreError> {
        let name_bytes = projection_name.as_bytes();
        if name_bytes.len() > MAX_NAME_BYTES {
            return Err(CheckpointStoreError::NameTooLong {
                size: name_bytes.len(),
                cap: MAX_NAME_BYTES,
            });
        }
        let name_len = name_bytes.len() as u16;

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&name_len.to_le_bytes());
        hasher.update(name_bytes);
        hasher.update(&lsn.0.to_le_bytes());
        let crc = hasher.finalize();

        self.writer.write_all(&name_len.to_le_bytes())?;
        self.writer.write_all(name_bytes)?;
        self.writer.write_all(&lsn.0.to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;

        self.in_memory.insert(projection_name.to_string(), lsn);
        self.durable_end_offset += (2 + name_bytes.len() + 8 + 4) as u64;
        Ok(())
    }

    /// Flush + fsync. After this returns, every `set` since the last `sync`
    /// is durable.
    pub fn sync(&mut self) -> Result<(), CheckpointStoreError> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    /// Path of the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Byte offset of the durable end (after the last fully-valid record).
    pub fn durable_end_offset(&self) -> u64 {
        self.durable_end_offset
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
        Ok(target)
    }

    /// Rewrite the file as a compacted snapshot — one record per projection,
    /// in arbitrary order. Useful when a long-lived store has accumulated
    /// many supersede-records for the same projection. Atomic via
    /// write-temp-then-rename. After this, `durable_end_offset` is reset.
    pub fn compact(&mut self) -> Result<(), CheckpointStoreError> {
        self.writer.flush()?;

        // Write the compacted snapshot to a side file, fsync it, then
        // close every handle on the live file before renaming (Windows
        // refuses to rename over a file with an open append handle).
        let tmp_path = self.path.with_extension("checkpoint.tmp");
        {
            let mut tmp = BufWriter::new(File::create(&tmp_path)?);
            for (name, lsn) in &self.in_memory {
                let bytes = name.as_bytes();
                let len = bytes.len() as u16;
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(&len.to_le_bytes());
                hasher.update(bytes);
                hasher.update(&lsn.0.to_le_bytes());
                let crc = hasher.finalize();
                tmp.write_all(&len.to_le_bytes())?;
                tmp.write_all(bytes)?;
                tmp.write_all(&lsn.0.to_le_bytes())?;
                tmp.write_all(&crc.to_le_bytes())?;
            }
            tmp.flush()?;
            tmp.get_ref().sync_data()?;
        }

        // Re-point `self.writer` at the tmp file so the old append handle
        // on `self.path` is dropped. Then rename can proceed on Windows.
        let staged = OpenOptions::new().read(true).append(true).open(&tmp_path)?;
        self.writer = BufWriter::new(staged);

        std::fs::rename(&tmp_path, &self.path)?;

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
        Ok(())
    }
}

fn scan(path: &Path) -> Result<(HashMap<String, Lsn>, u64), CheckpointStoreError> {
    let mut map = HashMap::new();
    if !path.exists() {
        return Ok((map, 0));
    }
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(0))?;

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

/// Returns `Ok(true)` if `buf` was fully filled; `Ok(false)` if EOF was
/// hit before the read completed (either clean — zero bytes read — or
/// torn — some bytes then EOF). Both forms mean "stop scanning here" for
/// our purposes, so they are folded into one return value.
fn read_exact_or_eof(file: &mut File, buf: &mut [u8]) -> Result<bool, CheckpointStoreError> {
    read_exact_full(file, buf)
}

fn read_exact_full(file: &mut File, buf: &mut [u8]) -> Result<bool, CheckpointStoreError> {
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
}
