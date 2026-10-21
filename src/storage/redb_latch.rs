//! The write latch for redb stores.
//!
//! **A redb handle is poisoned by its first I/O error.** Every backend call goes
//! through redb's `CheckedBackend`, which records the failure of any call to the
//! file (`tree_store/page_store/cached_file.rs:137-180` in redb 2.6.3) and from
//! then on answers every call with `StorageError::PreviousIo` without touching the
//! file (`cached_file.rs:129-135`). `Database::begin_write` checks that first
//! (`db.rs:1027`). The flag is set only at construction (`cached_file.rs:121`), so
//! nothing clears it but dropping the `Database` and opening the file again, which
//! is what redb's own `transient_io_error` test shows (`db.rs:1389-1435`). RocksDB's
//! flush+retry has nothing to work with here: a NoSpace failure latches the store
//! hard at once.
//!
//! **Detected at the file, not at each commit.** The page write that meets a full
//! disk is usually the commit's, but redb also writes pages from inside
//! `Table::insert` when its write buffer is over budget (`cached_file.rs:432-441`),
//! and either one poisons the handle. Every database the stores open goes through
//! [`open`] onto a [`LatchingBackend`], so whichever write meets ENOSPC latches the
//! file it was writing, keyed by that file's path.
//!
//! **Refused with the latch, not with `PreviousIo`.** A later write on the poisoned
//! handle is answered with the latch reason, which `AppError` maps to
//! `503 STORAGE_WRITE_LATCHED`. Any other poisoning still answers `PreviousIo`.
//!
//! **No reopen in place.** Every store built over a database holds its own clone of
//! the `Arc<redb::Database>`, and so do `MemorySystem` and `GraphMemory` for their
//! backups. There is no slot they all read through to swap, and redb's exclusive
//! `flock` on the file refuses a second open while any of them keeps the poisoned
//! handle. A restart is the reopen. A tenant evicted from the cache drops every
//! holder of its handles, so its next open is a reopen too. Every open writes and
//! syncs the header (`db.rs:926`, `page_manager.rs:371-376`), and that first sync
//! on the new handle clears the latch as a landed write clears a RocksDB one.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, Weak};

use anyhow::Result;
use parking_lot::Mutex;

use super::write_latch;

/// Why a hard latch on a redb store needs a restart, for the one ERROR line.
const REOPEN_ONLY: &str = "redb answers every write on a handle with PreviousIo after its \
     first I/O error until the database is reopened, and every store over the file holds \
     that handle";

/// The file one open handle writes.
#[derive(Debug)]
struct OpenFile {
    path: PathBuf,
    /// The NoSpace failure that poisoned this handle, once one has.
    no_space: OnceLock<String>,
}

/// redb's own file backend, reporting NoSpace into the write latch.
#[derive(Debug)]
struct LatchingBackend {
    file: redb::backends::FileBackend,
    open_file: Arc<OpenFile>,
}

impl LatchingBackend {
    /// A call that writes the file. A NoSpace failure latches the store here,
    /// before redb sees the error and poisons the handle.
    fn writing<T>(
        &self,
        op: impl FnOnce(&redb::backends::FileBackend) -> io::Result<T>,
    ) -> io::Result<T> {
        #[cfg(test)]
        let result = injected::refuse_if_armed(&self.open_file.path).and_then(|()| op(&self.file));
        #[cfg(not(test))]
        let result = op(&self.file);

        if let Err(err) = &result {
            if write_latch::is_enospc(&err.to_string()) {
                let err = anyhow::anyhow!("redb could not write its file: {err}");
                let _ = self.open_file.no_space.set(err.to_string());
                write_latch::latch_until_reopened(
                    write_latch::global(),
                    &self.open_file.path,
                    &err,
                    REOPEN_ONLY,
                );
            }
        }
        result
    }
}

impl redb::StorageBackend for LatchingBackend {
    fn len(&self) -> io::Result<u64> {
        self.file.len()
    }

    fn read(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.file.read(offset, len)
    }

    fn set_len(&self, len: u64) -> io::Result<()> {
        self.writing(|file| file.set_len(len))
    }

    fn sync_data(&self, eventual: bool) -> io::Result<()> {
        self.writing(|file| file.sync_data(eventual))?;
        // Every commit but a non-durable one ends in a sync, so a sync that
        // succeeds is redb's counterpart of a RocksDB write getting through.
        let latch = write_latch::global();
        if latch.is_latched() && self.open_file.no_space.get().is_none() {
            latch.note_write_ok(&self.open_file.path);
        }
        Ok(())
    }

    fn write(&self, offset: u64, data: &[u8]) -> io::Result<()> {
        self.writing(|file| file.write(offset, data))
    }
}

/// An open handle's file, found by the handle's address. redb gives no way from a
/// `Database` back to its backend or its path, and the handle travels as a bare
/// `Arc<redb::Database>` into modules that never see the path. The `Weak` keeps
/// the handle's allocation alive, so no other database can take its address while
/// the entry stands.
struct Registered {
    database: Weak<redb::Database>,
    open_file: Arc<OpenFile>,
}

fn registry() -> &'static Mutex<HashMap<usize, Registered>> {
    static HANDLES: OnceLock<Mutex<HashMap<usize, Registered>>> = OnceLock::new();
    HANDLES.get_or_init(Default::default)
}

fn identity(database: &redb::Database) -> usize {
    std::ptr::from_ref(database) as usize
}

/// Open or create the redb database at `path`, as `redb::Builder::create` does,
/// with the file's writes reporting into the write latch.
pub(crate) fn open(path: &Path, builder: &redb::Builder) -> Result<Arc<redb::Database>> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let open_file = Arc::new(OpenFile {
        path: path.to_path_buf(),
        no_space: OnceLock::new(),
    });
    let backend = LatchingBackend {
        file: redb::backends::FileBackend::new(file)?,
        open_file: Arc::clone(&open_file),
    };
    let database = Arc::new(builder.create_with_backend(backend)?);

    let mut handles = registry().lock();
    handles.retain(|_, handle| handle.database.strong_count() > 0);
    handles.insert(
        identity(&database),
        Registered {
            database: Arc::downgrade(&database),
            open_file,
        },
    );
    Ok(database)
}

/// `err`, from a write `database` failed, unless NoSpace has poisoned that handle:
/// then the write is refused with the latch, which the caller and the HTTP mapping
/// need to see in place of redb's `PreviousIo`.
pub(crate) fn refused(database: &redb::Database, err: anyhow::Error) -> anyhow::Error {
    let open_file = registry()
        .lock()
        .get(&identity(database))
        .map(|handle| Arc::clone(&handle.open_file));
    let Some(open_file) = open_file else {
        return err;
    };
    let Some(reason) = open_file.no_space.get() else {
        return err;
    };
    let refusal = err.context(format!(
        "write refused: this redb store is latched until it is reopened ({reason})"
    ));
    write_latch::latch_until_reopened(
        write_latch::global(),
        &open_file.path,
        &refusal,
        REOPEN_ONLY,
    );
    refusal
}

/// Per-thread NoSpace refusals at the file, so a test can drive a real redb write
/// into the latch: redb runs its I/O on the calling thread, and the refusal is the
/// same `io::Error` a full disk returns, so redb poisons the handle for real.
#[cfg(test)]
pub(crate) mod injected {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io;
    use std::path::{Path, PathBuf};

    thread_local! {
        /// The errno each armed file's writes fail with.
        static ARMED: RefCell<HashMap<PathBuf, i32>> = RefCell::new(HashMap::new());
    }

    /// Refuse every write, sync and resize of the file at `path` on this thread
    /// for want of space.
    pub(crate) fn arm(path: &Path) {
        arm_errno(path, libc::ENOSPC);
    }

    /// [`arm`] with another errno, for the failures the latch must leave alone.
    pub(crate) fn arm_errno(path: &Path, errno: i32) {
        ARMED.with(|armed| armed.borrow_mut().insert(path.to_path_buf(), errno));
    }

    pub(crate) fn disarm(path: &Path) {
        ARMED.with(|armed| armed.borrow_mut().remove(path));
    }

    pub(super) fn refuse_if_armed(path: &Path) -> io::Result<()> {
        match ARMED.with(|armed| armed.borrow().get(path).copied()) {
            Some(errno) => Err(io::Error::from_raw_os_error(errno)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::redb::{create_redb_database_for_test, RedbColumnStore};
    use crate::storage::write_latch::{chain_is_enospc, global, test_serial};
    use crate::storage::ColumnStore;

    /// A database at a fresh path with one committed row, so the refused write is
    /// not the first the file has seen.
    fn seeded(dir: &tempfile::TempDir) -> (PathBuf, Arc<redb::Database>) {
        let path = dir.path().join("storage.redb");
        let database = create_redb_database_for_test(&path).expect("open");
        RedbColumnStore::new(Arc::clone(&database))
            .put_cf("rows", b"seed", b"1")
            .expect("seed write");
        (path, database)
    }

    /// The premise everything here rests on, against the redb in Cargo.lock: after
    /// one failed commit the handle refuses writes that would otherwise succeed.
    #[test]
    fn a_failed_commit_poisons_the_handle_until_it_is_reopened() {
        let _latching = test_serial::latching();
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, database) = seeded(&dir);
        let rows = RedbColumnStore::new(Arc::clone(&database));

        injected::arm(&path);
        let failed = rows.put_cf("rows", b"k", b"v");
        injected::disarm(&path);
        assert!(chain_is_enospc(
            &failed.expect_err("the armed refusal reaches the caller")
        ));

        let raw = database
            .begin_write()
            .err()
            .expect("the disk has room again, yet the handle refuses");
        assert!(
            matches!(
                raw,
                redb::TransactionError::Storage(redb::StorageError::PreviousIo)
            ),
            "{raw}"
        );

        drop(rows);
        drop(database);
        let reopened = create_redb_database_for_test(&path).expect("reopen");
        RedbColumnStore::new(reopened)
            .put_cf("rows", b"k", b"v")
            .expect("a reopened handle writes");
        assert!(
            !global().is_latched_at(&path),
            "the reopen's sync clears it"
        );
    }

    #[test]
    fn no_space_latches_hard_and_later_writes_are_refused_with_the_latch() {
        let _latching = test_serial::latching();
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, database) = seeded(&dir);
        let rows = RedbColumnStore::new(Arc::clone(&database));

        injected::arm(&path);
        let failed = rows.put_cf("rows", b"k", b"v");
        injected::disarm(&path);
        assert!(chain_is_enospc(&failed.expect_err("refused")));

        let latched = global().snapshot_at(&path).expect("the file latched");
        assert!(latched.hard, "a poisoned handle has nothing to retry");
        assert_eq!(latched.recovery_attempts, 0);
        assert_eq!(latched.failed_writes, 1);

        let refusal = rows
            .put_cf("rows", b"k", b"v")
            .expect_err("the poisoned handle takes nothing");
        assert!(
            chain_is_enospc(&refusal),
            "refused with PreviousIo alone, which answers 500: {refusal:#}"
        );
        assert!(
            refusal.to_string().starts_with("write refused"),
            "{refusal:#}"
        );
        assert_eq!(
            global()
                .snapshot_at(&path)
                .expect("still latched")
                .failed_writes,
            2,
            "the refusal is counted against the store"
        );

        drop(rows);
        drop(database);
        create_redb_database_for_test(&path).expect("reopen");
        assert!(!global().is_latched_at(&path));
    }

    /// Any other I/O failure poisons the handle just the same, and keeps redb's
    /// own errors and no latch, as before.
    #[test]
    fn a_handle_poisoned_by_another_error_keeps_previous_io_and_no_latch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, database) = seeded(&dir);
        let rows = RedbColumnStore::new(Arc::clone(&database));

        injected::arm_errno(&path, libc::EIO);
        let failed = rows.put_cf("rows", b"k", b"v");
        injected::disarm(&path);
        let failed = failed.expect_err("the armed refusal reaches the caller");
        assert!(!chain_is_enospc(&failed), "{failed:#}");
        assert!(!global().is_latched_at(&path));

        let later = rows
            .put_cf("rows", b"k", b"v")
            .expect_err("the poisoned handle takes nothing");
        assert!(
            later
                .downcast_ref::<redb::TransactionError>()
                .is_some_and(|err| matches!(
                    err,
                    redb::TransactionError::Storage(redb::StorageError::PreviousIo)
                )),
            "{later:#}"
        );
        assert!(!global().is_latched_at(&path));
    }
}
