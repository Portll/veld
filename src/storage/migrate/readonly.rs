//! Opening a store without writing to it.
//!
//! Both engines write on an ordinary open: rocksdb takes its LOCK file and may create
//! a store that was absent, and redb 2.x flips a recovery flag in the file header the
//! moment a database is opened and flips it back on drop. A migration source must
//! never be modified and a dry run must write nothing, so neither open is usable here.

use anyhow::{bail, Context, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// A rocksdb store opened read-only, with every column family it has.
pub(crate) struct RocksReadOnly {
    pub db: rocksdb::DB,
    pub column_families: Vec<String>,
}

/// `Ok(None)` only when the store is absent (no `CURRENT` file). Any other failure to
/// look is an error, never an empty store.
pub(crate) fn open_rocks(dir: &Path) -> Result<Option<RocksReadOnly>> {
    match std::fs::metadata(dir.join("CURRENT")) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("inspecting {}", dir.display())),
    }
    let opts = rocksdb::Options::default();
    let mut column_families = rocksdb::DB::list_cf(&opts, dir)
        .with_context(|| format!("listing column families of {}", dir.display()))?;
    column_families.sort();
    let db = rocksdb::DB::open_cf_for_read_only(&opts, dir, &column_families, false)
        .with_context(|| format!("opening {} read-only", dir.display()))?;
    Ok(Some(RocksReadOnly { db, column_families }))
}

#[cfg(feature = "storage-redb")]
pub(crate) use redb_read_only::open_redb;

#[cfg(feature = "storage-redb")]
mod redb_read_only {
    use super::*;
    use std::sync::Mutex;

    /// Field order matters: the database drops before the shared lock is released.
    pub(crate) struct RedbReadOnly {
        pub db: redb::Database,
        _lock: std::fs::File,
    }

    /// `Ok(None)` when the file is absent or zero-length (redb would initialise a new
    /// database in it). Holds a shared `flock` for the lifetime of the handle, so a
    /// writer that opens the file normally (redb takes `LOCK_EX`) is refused instead of
    /// mutating pages under the reader; a writer already holding it refuses this open.
    pub(crate) fn open_redb(path: &Path) -> Result<Option<RedbReadOnly>> {
        let lock = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
        };
        match lock.try_lock_shared() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => bail!(
                "{} is held open for writing by another process (a running server?) — \
                 refusing to read it while it can change underneath the read",
                path.display()
            ),
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("locking {}", path.display()))
            }
        }
        let len = lock.metadata()?.len();
        if len == 0 {
            return Ok(None);
        }
        let backend = OverlayBackend {
            file: Mutex::new(std::fs::File::open(path)?),
            state: Mutex::new(Overlay { len, file_visible: len, writes: Vec::new() }),
        };
        let db = redb::Database::builder()
            .create_with_backend(backend)
            .with_context(|| format!("opening {} read-only", path.display()))?;
        Ok(Some(RedbReadOnly { db, _lock: lock }))
    }

    #[derive(Debug)]
    struct Overlay {
        len: u64,
        file_visible: u64,
        writes: Vec<(u64, Vec<u8>)>,
    }

    /// Reads the file; every write lands in memory and is visible to later reads.
    #[derive(Debug)]
    struct OverlayBackend {
        file: Mutex<std::fs::File>,
        state: Mutex<Overlay>,
    }

    impl redb::StorageBackend for OverlayBackend {
        fn len(&self) -> std::io::Result<u64> {
            Ok(self.state.lock().unwrap_or_else(|e| e.into_inner()).len)
        }

        fn read(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let end = offset + len as u64;
            let mut buf = vec![0u8; len];
            if offset < state.file_visible {
                let n = (state.file_visible.min(end) - offset) as usize;
                let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
                file.seek(SeekFrom::Start(offset))?;
                file.read_exact(&mut buf[..n])?;
            }
            for (at, data) in &state.writes {
                let w_end = at + data.len() as u64;
                if w_end <= offset || *at >= end {
                    continue;
                }
                let from = (*at).max(offset);
                let to = w_end.min(end);
                buf[(from - offset) as usize..(to - offset) as usize]
                    .copy_from_slice(&data[(from - at) as usize..(to - at) as usize]);
            }
            Ok(buf)
        }

        fn set_len(&self, len: u64) -> std::io::Result<()> {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if len < state.len {
                state.file_visible = state.file_visible.min(len);
                for (at, data) in state.writes.iter_mut() {
                    if *at >= len {
                        data.clear();
                    } else if *at + data.len() as u64 > len {
                        data.truncate((len - *at) as usize);
                    }
                }
            }
            state.len = len;
            Ok(())
        }

        fn sync_data(&self, _eventual: bool) -> std::io::Result<()> {
            Ok(())
        }

        fn write(&self, offset: u64, data: &[u8]) -> std::io::Result<()> {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.writes.push((offset, data.to_vec()));
            state.len = state.len.max(offset + data.len() as u64);
            Ok(())
        }
    }
}
