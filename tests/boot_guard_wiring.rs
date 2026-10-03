#![cfg(feature = "rocksdb")]
//! The boot guard is wired: the real manager refuses to start over the outage shape.

#![cfg(feature = "storage-redb")]

use tempfile::TempDir;
use veld::config::{ServerConfig, StorageBackend};
use veld::handlers::MultiUserMemoryManager;
use veld::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use veld::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore;
use veld::storage::redb::RedbPrimaryMemoryStore;
use veld::storage::PrimaryMemoryStore;

fn memory() -> Memory {
    Memory::new(
        MemoryId(veld::uuid::Uuid::new_v4()),
        Experience {
            experience_type: ExperienceType::Observation,
            content: "a memory that exists only in rocksdb".to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    )
}

fn manager(dir: &TempDir, backend: StorageBackend) -> anyhow::Result<MultiUserMemoryManager> {
    let cfg = ServerConfig {
        storage_path: dir.path().to_path_buf(),
        backup_enabled: false,
        effective_storage_backend: backend,
        ..ServerConfig::default()
    };
    MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg)
}

fn outage_shape() -> TempDir {
    let dir = TempDir::new().unwrap();
    let user = dir.path().join("portll");
    {
        let store = RocksDbPrimaryMemoryStore::open(&user, None).unwrap();
        store.put_memory(&memory()).unwrap();
        store.flush().unwrap();
    }
    RedbPrimaryMemoryStore::open(&user).unwrap();
    dir
}

#[test]
fn a_redb_server_over_rocksdb_data_refuses_to_start() {
    let dir = outage_shape();
    let err = match manager(&dir, StorageBackend::Redb) {
        Ok(_) => panic!("a redb server over rocksdb-only data must not start"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("portll") && err.contains("VELD_STORAGE_BACKEND=rocksdb"),
        "{err}"
    );
}

#[test]
fn the_same_data_starts_on_the_backend_that_holds_it() {
    let dir = outage_shape();
    let m =
        manager(&dir, StorageBackend::RocksDb).expect("rocksdb holds the data, so it must start");
    assert_eq!(
        (m.backend_census.rocksdb, m.backend_census.redb),
        (1, Some(0))
    );
}
