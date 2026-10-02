//! Refuse to boot a backend that is empty while the other backend holds data.
//!
//! On 2026-09-10 a redb build served an empty store as healthy over users whose
//! memories were all in rocksdb (ADR 0006). File presence cannot catch that: an
//! empty redb file is a fixed-size allocation, so this counts records.

use std::path::Path;

use anyhow::{bail, Result};

use crate::config::StorageBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreContent {
    Absent,
    Empty,
    NonEmpty,
}

pub fn user_store_content(user_dir: &Path, backend: StorageBackend) -> Result<StoreContent> {
    match backend {
        #[cfg(feature = "legacy-rocksdb")]
        StorageBackend::RocksDb => Ok(
            match crate::storage::legacy_rocksdb::user_has_memories(user_dir)? {
                None => StoreContent::Absent,
                Some(false) => StoreContent::Empty,
                Some(true) => StoreContent::NonEmpty,
            },
        ),
        #[cfg(not(feature = "legacy-rocksdb"))]
        StorageBackend::RocksDb => rocksdb_files(&user_dir.join("storage")),
        #[cfg(feature = "storage-redb")]
        StorageBackend::Redb => Ok(match crate::storage::redb::user_memory_count(user_dir)? {
            None => StoreContent::Absent,
            Some(0) => StoreContent::Empty,
            Some(_) => StoreContent::NonEmpty,
        }),
        #[cfg(not(feature = "storage-redb"))]
        StorageBackend::Redb => {
            bail!(
                "this build cannot read redb stores, so {} is unknown",
                user_dir.display()
            )
        }
    }
}

const fn other_backend(backend: StorageBackend) -> StorageBackend {
    match backend {
        StorageBackend::Redb => StorageBackend::RocksDb,
        StorageBackend::RocksDb => StorageBackend::Redb,
    }
}

/// Tenants holding at least one memory, per backend, measured at boot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendCensus {
    pub rocksdb: usize,
    /// `None` when this build cannot read redb, which is unknown rather than zero.
    pub redb: Option<usize>,
    pub measured_at: chrono::DateTime<chrono::Utc>,
}

/// A probe that fails refuses the boot too: an unreadable store is unknown, and
/// unknown is exactly what the outage served as healthy.
pub fn refuse_stranded_data(
    base_path: &Path,
    selected: StorageBackend,
    users: &[String],
) -> Result<BackendCensus> {
    let other = other_backend(selected);
    let can_read_other = other != StorageBackend::Redb || cfg!(feature = "storage-redb");
    if !can_read_other {
        tracing::warn!(
            "boot guard: this build cannot read redb, so data stranded there is unchecked"
        );
    }

    let mut with_data = [0usize; 2];
    let slot = |b: StorageBackend| match b {
        StorageBackend::RocksDb => 0,
        StorageBackend::Redb => 1,
    };
    let mut stranded = Vec::new();
    for user in users {
        let dir = base_path.join(user);
        let selected_has = user_store_content(&dir, selected)? == StoreContent::NonEmpty;
        let other_has =
            can_read_other && user_store_content(&dir, other)? == StoreContent::NonEmpty;
        with_data[slot(selected)] += usize::from(selected_has);
        with_data[slot(other)] += usize::from(other_has);
        if !selected_has && other_has {
            stranded.push(user.as_str());
        }
    }

    if !stranded.is_empty() {
        let uncounted = if cfg!(feature = "legacy-rocksdb") || other != StorageBackend::RocksDb {
            ""
        } else {
            " This build has no RocksDB code, so a RocksDB store with any table file or non-empty \
             write-ahead log counts as holding memories: a build with --features legacy-rocksdb \
             counts them, and a store it finds holds none can be moved aside."
        };
        bail!(
            "refusing to serve: the {selected} store is empty for {n} user(s) whose memories are \
             in {other}: {users}. Serving would report healthy over an empty store. Set \
             VELD_STORAGE_BACKEND={other} to serve the existing data, or copy it with \
             `veld migrate --from {other} --to {selected} --dry-run` and then without \
             --dry-run.{uncounted}",
            n = stranded.len(),
            users = stranded.join(", "),
        );
    }

    Ok(BackendCensus {
        rocksdb: with_data[0],
        redb: cfg!(feature = "storage-redb").then_some(with_data[1]),
        measured_at: chrono::Utc::now(),
    })
}

#[cfg(all(test, feature = "storage-redb", feature = "legacy-rocksdb"))]
mod tests {
    use super::*;
    use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId, MemoryTier};
    use crate::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore;
    use crate::storage::redb::RedbPrimaryMemoryStore;
    use crate::storage::PrimaryMemoryStore;

    fn memory(content: &str) -> Memory {
        let now = chrono::Utc::now();
        Memory::from_legacy(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                experience_type: ExperienceType::Observation,
                content: content.to_string(),
                ..Default::default()
            },
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }

    fn rocksdb_with_memory(user_dir: &Path) {
        let store = RocksDbPrimaryMemoryStore::open(user_dir, None).unwrap();
        store.put_memory(&memory("lives in rocksdb")).unwrap();
        store.flush().unwrap();
    }

    fn redb_empty(user_dir: &Path) {
        RedbPrimaryMemoryStore::open(user_dir).unwrap();
    }

    fn redb_with_memory(user_dir: &Path) {
        let store = RedbPrimaryMemoryStore::open(user_dir).unwrap();
        store.put_memory(&memory("lives in redb")).unwrap();
    }

    fn users(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn the_outage_shape_refuses_and_names_the_user() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("alice");
        rocksdb_with_memory(&dir);
        redb_empty(&dir);

        let err = refuse_stranded_data(base.path(), StorageBackend::Redb, &users(&["alice"]))
            .expect_err("empty redb over populated rocksdb must refuse");
        let msg = err.to_string();
        assert!(msg.contains("alice"), "{msg}");
        assert!(msg.contains("VELD_STORAGE_BACKEND=rocksdb"), "{msg}");
    }

    #[test]
    fn the_populated_backend_boots() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("alice");
        rocksdb_with_memory(&dir);
        redb_empty(&dir);

        refuse_stranded_data(base.path(), StorageBackend::RocksDb, &users(&["alice"])).unwrap();
    }

    #[test]
    fn data_only_in_redb_refuses_a_rocksdb_boot() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("bob");
        redb_with_memory(&dir);

        let err = refuse_stranded_data(base.path(), StorageBackend::RocksDb, &users(&["bob"]))
            .expect_err("absent rocksdb over populated redb must refuse");
        assert!(
            err.to_string().contains("VELD_STORAGE_BACKEND=redb"),
            "{err}"
        );
    }

    #[test]
    fn a_fresh_install_and_a_half_migrated_user_both_boot() {
        let base = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(base.path().join("fresh")).unwrap();
        let both = base.path().join("both");
        rocksdb_with_memory(&both);
        redb_with_memory(&both);

        for backend in [StorageBackend::Redb, StorageBackend::RocksDb] {
            refuse_stranded_data(base.path(), backend, &users(&["fresh", "both"])).unwrap();
        }
    }

    #[test]
    fn the_census_counts_each_backend_independently() {
        let base = tempfile::tempdir().unwrap();
        let both = base.path().join("both");
        rocksdb_with_memory(&both);
        redb_with_memory(&both);
        rocksdb_with_memory(&base.path().join("rocks_only"));
        redb_empty(&base.path().join("rocks_only"));
        std::fs::create_dir_all(base.path().join("fresh")).unwrap();

        let census = refuse_stranded_data(
            base.path(),
            StorageBackend::RocksDb,
            &users(&["both", "fresh", "rocks_only"]),
        )
        .unwrap();
        assert_eq!((census.rocksdb, census.redb), (2, Some(1)));
    }

    #[test]
    fn only_the_stranded_users_are_named() {
        let base = tempfile::tempdir().unwrap();
        rocksdb_with_memory(&base.path().join("stranded"));
        redb_with_memory(&base.path().join("migrated"));

        let msg = refuse_stranded_data(
            base.path(),
            StorageBackend::Redb,
            &users(&["migrated", "stranded"]),
        )
        .unwrap_err()
        .to_string();
        assert!(
            msg.contains("1 user(s)") && msg.contains("stranded"),
            "{msg}"
        );
        assert!(!msg.contains("migrated,"), "{msg}");
    }

    #[test]
    fn probing_an_absent_store_creates_nothing() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("ghost");
        std::fs::create_dir_all(&dir).unwrap();

        for backend in [StorageBackend::Redb, StorageBackend::RocksDb] {
            assert_eq!(
                user_store_content(&dir, backend).unwrap(),
                StoreContent::Absent
            );
        }
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "a probe must not create a store"
        );
    }

    #[test]
    fn an_empty_redb_file_reads_empty_not_absent() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("alloc");
        redb_empty(&dir);

        assert!(dir.join("storage.redb").metadata().unwrap().len() > 0);
        assert_eq!(
            user_store_content(&dir, StorageBackend::Redb).unwrap(),
            StoreContent::Empty
        );
    }

    #[test]
    fn an_unreadable_store_refuses_rather_than_passing() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("storage.redb"), b"not a redb file").unwrap();

        assert!(
            refuse_stranded_data(base.path(), StorageBackend::RocksDb, &users(&["corrupt"]))
                .is_err()
        );
    }
}

/// A RocksDB directory as a build that cannot open it sees it: absent without CURRENT, empty when
/// no table file and no write-ahead log holds anything, and otherwise taken to hold memories,
/// since only RocksDB can tell its memories from the default family's other keys.
#[cfg(not(feature = "legacy-rocksdb"))]
fn rocksdb_files(dir: &Path) -> Result<StoreContent> {
    let inspect =
        |e: std::io::Error| anyhow::Error::from(e).context(format!("inspecting {}", dir.display()));
    match std::fs::metadata(dir.join("CURRENT")) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(StoreContent::Absent),
        Err(e) => return Err(inspect(e)),
    }
    for entry in std::fs::read_dir(dir).map_err(inspect)? {
        let entry = entry.map_err(inspect)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let table = name.ends_with(".sst") || name.ends_with(".blob");
        let wal = name
            .strip_suffix(".log")
            .is_some_and(|number| !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()));
        if table || wal && entry.metadata().map_err(inspect)?.len() > 0 {
            return Ok(StoreContent::NonEmpty);
        }
    }
    Ok(StoreContent::Empty)
}

#[cfg(all(test, not(feature = "legacy-rocksdb")))]
mod without_rocksdb {
    use super::*;

    /// A RocksDB directory as RocksDB leaves an empty store, plus `extra` files of the given sizes.
    fn rocksdb_dir(user_dir: &Path, extra: &[(&str, usize)]) {
        let dir = user_dir.join("storage");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, size) in [
            ("CURRENT", 16),
            ("IDENTITY", 36),
            ("LOCK", 0),
            ("LOG", 19331),
            ("LOG.old.1785887307401456", 82050),
            ("MANIFEST-000005", 232),
            ("OPTIONS-000007", 17332),
            ("000004.log", 0),
        ]
        .into_iter()
        .chain(extra.iter().copied())
        {
            std::fs::write(dir.join(name), vec![b'x'; size]).unwrap();
        }
    }

    #[test]
    fn a_store_with_no_table_file_and_empty_logs_is_empty_and_boots_on_redb() {
        let base = tempfile::tempdir().unwrap();
        rocksdb_dir(&base.path().join("test"), &[]);
        assert_eq!(
            user_store_content(&base.path().join("test"), StorageBackend::RocksDb).unwrap(),
            StoreContent::Empty
        );
        let census =
            refuse_stranded_data(base.path(), StorageBackend::Redb, &["test".to_string()]).unwrap();
        assert_eq!((census.rocksdb, census.redb), (0, Some(0)));
    }

    #[test]
    fn a_table_file_or_a_written_log_counts_as_memories_and_the_refusal_says_why() {
        let base = tempfile::tempdir().unwrap();
        rocksdb_dir(&base.path().join("tables"), &[("000009.sst", 1322)]);
        rocksdb_dir(&base.path().join("logged"), &[("000016.log", 64)]);
        rocksdb_dir(&base.path().join("blobs"), &[("000010.blob", 10)]);
        let users: Vec<String> = ["tables", "logged", "blobs"].map(String::from).to_vec();
        let msg = refuse_stranded_data(base.path(), StorageBackend::Redb, &users)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("3 user(s)") && msg.contains("--features legacy-rocksdb counts them"),
            "{msg}"
        );
    }

    #[test]
    fn no_current_file_is_no_store() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("ghost");
        std::fs::create_dir_all(dir.join("storage")).unwrap();
        std::fs::write(dir.join("storage").join("000004.sst"), b"x").unwrap();
        assert_eq!(
            user_store_content(&dir, StorageBackend::RocksDb).unwrap(),
            StoreContent::Absent
        );
    }
}
