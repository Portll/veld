//! One store per backend, so a consumer's tests run the same assertions on each.
//!
//! Every handle here is built from a live database, never reopened by path —
//! the same constraint production construction obeys.

use std::sync::Arc;

use tempfile::TempDir;

use crate::config::StorageBackend;
use crate::storage::{AuditStore, BatchWriter, ColumnStore, GraphStore, KeyValueStore};

/// A fresh [`ColumnStore`] per backend with `families` available. The
/// `TempDir` must outlive the store.
pub(crate) fn column_stores(
    families: &[&str],
) -> Vec<(StorageBackend, Arc<dyn ColumnStore>, TempDir)> {
    let mut out: Vec<(StorageBackend, Arc<dyn ColumnStore>, TempDir)> = Vec::new();
    #[cfg(not(feature = "rocksdb"))]
    let _ = families;

    #[cfg(feature = "rocksdb")]
    {
        let tmp = TempDir::new().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let mut names = vec!["default"];
        names.extend(families.iter().copied());
        let db = rocksdb::DB::open_cf(&opts, tmp.path(), names).unwrap();
        out.push((
            StorageBackend::RocksDb,
            Arc::new(crate::storage::legacy_rocksdb::RocksDbColumnStore::new(
                Arc::new(db),
            )),
            tmp,
        ));
    }

    #[cfg(feature = "storage-redb")]
    {
        let tmp = TempDir::new().unwrap();
        let database =
            crate::storage::redb::create_redb_database_for_test(&tmp.path().join("t.redb"))
                .unwrap();
        out.push((
            StorageBackend::Redb,
            Arc::new(crate::storage::redb::RedbColumnStore::new(database)),
            tmp,
        ));
    }

    out
}

/// A [`KeyValueStore`] per backend, bound to one family.
///
/// For a store that prefixes its own keys the family name carries no meaning,
/// which is why it is the caller's to choose: what the handle has to answer is
/// the same set of calls on both engines. Production binds the DEFAULT keyspace
/// for several of these stores; a named family is used here because a rocksdb
/// `cf_handle("default")` is not guaranteed across crate versions, and the
/// prefixes the store writes are identical either way.
pub(crate) fn key_value_stores(
    family: &str,
) -> Vec<(
    StorageBackend,
    Arc<dyn crate::storage::KeyValueStore>,
    TempDir,
)> {
    column_stores(&[family])
        .into_iter()
        .map(|(backend, store, tmp)| {
            let kv: Arc<dyn crate::storage::KeyValueStore> =
                Arc::new(crate::storage::FamilyKeyValueStore::new(store, family).unwrap());
            (backend, kv, tmp)
        })
        .collect()
}

/// A fresh [`PrimaryMemoryStore`] per backend, each opened the way production opens it.
pub(crate) fn primary_stores() -> Vec<(
    StorageBackend,
    Arc<dyn crate::storage::PrimaryMemoryStore>,
    TempDir,
)> {
    let mut out: Vec<(
        StorageBackend,
        Arc<dyn crate::storage::PrimaryMemoryStore>,
        TempDir,
    )> = Vec::new();

    #[cfg(feature = "rocksdb")]
    {
        let tmp = TempDir::new().unwrap();
        let store =
            crate::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore::open(tmp.path(), None)
                .unwrap();
        out.push((StorageBackend::RocksDb, Arc::new(store), tmp));
    }

    #[cfg(feature = "storage-redb")]
    {
        let tmp = TempDir::new().unwrap();
        let store = crate::storage::redb::RedbPrimaryMemoryStore::open(tmp.path()).unwrap();
        out.push((StorageBackend::Redb, Arc::new(store), tmp));
    }

    out
}

/// A store under test and which implementation it is, so a disagreement names it.
pub(crate) struct NamedStore<T: ?Sized> {
    pub(crate) implementation: &'static str,
    pub(crate) store: Arc<T>,
}

/// Every [`KeyValueStore`] implementation per enabled backend, each over its own
/// fresh database. All of them are opaque byte maps, so all must answer the same
/// calls identically — two of one engine's own implementations disagreeing is as
/// much a defect as the engines disagreeing.
pub(crate) fn key_value_store_implementations(
    family: &str,
) -> Vec<(StorageBackend, NamedStore<dyn KeyValueStore>, TempDir)> {
    let mut out: Vec<(StorageBackend, NamedStore<dyn KeyValueStore>, TempDir)> =
        key_value_stores(family)
            .into_iter()
            .map(|(backend, store, tmp)| {
                let named = NamedStore {
                    implementation: "family-adapter",
                    store,
                };
                (backend, named, tmp)
            })
            .collect();

    #[cfg(feature = "rocksdb")]
    {
        let tmp = TempDir::new().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = Arc::new(rocksdb::DB::open_cf(&opts, tmp.path(), ["default", family]).unwrap());
        out.push((
            StorageBackend::RocksDb,
            NamedStore {
                implementation: "column-family",
                store: Arc::new(
                    crate::storage::legacy_rocksdb::RocksDbColumnFamilyStore::new(db, family),
                ),
            },
            tmp,
        ));

        let tmp = TempDir::new().unwrap();
        let db = Arc::new(rocksdb::DB::open_default(tmp.path()).unwrap());
        out.push((
            StorageBackend::RocksDb,
            NamedStore {
                implementation: "default-keyspace",
                store: Arc::new(
                    crate::storage::legacy_rocksdb::RocksDbDefaultKeyValueStore::new(db),
                ),
            },
            tmp,
        ));
    }

    #[cfg(feature = "storage-redb")]
    {
        let tmp = TempDir::new().unwrap();
        let store = crate::storage::redb::RedbKeyValueStore::open(tmp.path(), family).unwrap();
        out.push((
            StorageBackend::Redb,
            NamedStore {
                implementation: "namespace-file",
                store: Arc::new(store),
            },
            tmp,
        ));

        let tmp = TempDir::new().unwrap();
        let database =
            crate::storage::redb::create_redb_database_for_test(&tmp.path().join("t.redb"))
                .unwrap();
        out.push((
            StorageBackend::Redb,
            NamedStore {
                implementation: "shared-table",
                store: Arc::new(crate::storage::redb::RedbTableKeyValueStore::new(
                    database, "kv",
                )),
            },
            tmp,
        ));
    }

    out
}

/// A fresh [`GraphStore`] per backend, each opened the way the storage factory opens it.
pub(crate) fn graph_stores() -> Vec<(StorageBackend, Arc<dyn GraphStore>, TempDir)> {
    let mut out: Vec<(StorageBackend, Arc<dyn GraphStore>, TempDir)> = Vec::new();

    #[cfg(feature = "rocksdb")]
    {
        let tmp = TempDir::new().unwrap();
        let store =
            crate::storage::legacy_rocksdb::RocksDbGraphStore::open(tmp.path(), None).unwrap();
        out.push((StorageBackend::RocksDb, Arc::new(store), tmp));
    }

    #[cfg(feature = "storage-redb")]
    {
        let tmp = TempDir::new().unwrap();
        let store = crate::storage::redb::RedbGraphStore::open(tmp.path()).unwrap();
        out.push((StorageBackend::Redb, Arc::new(store), tmp));
    }

    out
}

/// A fresh [`AuditStore`] per backend. The rocksdb one lives in a named family of its
/// own database, as the shared store's audit family does in production.
pub(crate) fn audit_stores() -> Vec<(StorageBackend, Arc<dyn AuditStore>, TempDir)> {
    let mut out: Vec<(StorageBackend, Arc<dyn AuditStore>, TempDir)> = Vec::new();

    #[cfg(feature = "rocksdb")]
    {
        const AUDIT_FAMILY: &str = "audit";
        let tmp = TempDir::new().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = rocksdb::DB::open_cf(&opts, tmp.path(), ["default", AUDIT_FAMILY]).unwrap();
        out.push((
            StorageBackend::RocksDb,
            Arc::new(crate::storage::legacy_rocksdb::RocksDbAuditStore::new(
                Arc::new(db),
                AUDIT_FAMILY,
            )),
            tmp,
        ));
    }

    #[cfg(feature = "storage-redb")]
    {
        let tmp = TempDir::new().unwrap();
        let store = crate::storage::redb::RedbAuditStore::open(tmp.path()).unwrap();
        out.push((StorageBackend::Redb, Arc::new(store), tmp));
    }

    out
}

/// A [`BatchWriter`] and the [`ColumnStore`] that reads what it wrote, both over the
/// SAME live database — a writer is never opened by path (see [`BatchWriter`]).
pub(crate) struct BatchWriterUnderTest {
    pub(crate) writer: Arc<dyn BatchWriter>,
    pub(crate) columns: Arc<dyn ColumnStore>,
}

/// A fresh [`BatchWriterUnderTest`] per backend with `families` and the default
/// keyspace declared.
pub(crate) fn batch_writers(
    families: &[&str],
) -> Vec<(StorageBackend, BatchWriterUnderTest, TempDir)> {
    let mut out: Vec<(StorageBackend, BatchWriterUnderTest, TempDir)> = Vec::new();

    #[cfg(feature = "rocksdb")]
    {
        let tmp = TempDir::new().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let mut names = vec!["default"];
        names.extend(families.iter().copied());
        let db = Arc::new(rocksdb::DB::open_cf(&opts, tmp.path(), names).unwrap());
        out.push((
            StorageBackend::RocksDb,
            BatchWriterUnderTest {
                writer: Arc::new(crate::storage::legacy_rocksdb::RocksDbBatchWriter::new(
                    Arc::clone(&db),
                )),
                columns: Arc::new(crate::storage::legacy_rocksdb::RocksDbColumnStore::new(db)),
            },
            tmp,
        ));
    }

    #[cfg(feature = "storage-redb")]
    {
        let tmp = TempDir::new().unwrap();
        let database =
            crate::storage::redb::create_redb_database_for_test(&tmp.path().join("t.redb"))
                .unwrap();
        let columns = crate::storage::redb::RedbColumnStore::new(Arc::clone(&database));
        // rocksdb families exist from the moment the database opens; redb tables only
        // once created, and RedbBatchWriter refuses a missing one. Declaring them is
        // what every production `with_store` constructor does before its first batch.
        for family in std::iter::once("default").chain(families.iter().copied()) {
            columns.ensure_cf(family).unwrap();
        }
        out.push((
            StorageBackend::Redb,
            BatchWriterUnderTest {
                writer: Arc::new(crate::storage::redb::RedbBatchWriter::new(database)),
                columns: Arc::new(columns),
            },
            tmp,
        ));
    }

    out
}

/// Refuses a run that silently dropped a backend this build enables: a loop over
/// one store passes every assertion while proving nothing about the other.
pub(crate) fn assert_both_backends<T>(stores: &[(StorageBackend, T, TempDir)]) {
    let backends: Vec<StorageBackend> = stores.iter().map(|(b, _, _)| *b).collect();
    #[cfg(feature = "rocksdb")]
    assert!(
        backends.contains(&StorageBackend::RocksDb),
        "no rocksdb store under test"
    );
    #[cfg(feature = "storage-redb")]
    assert!(
        backends.contains(&StorageBackend::Redb),
        "no redb store under test"
    );
}
