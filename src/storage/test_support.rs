//! One store per backend, so a consumer's tests run the same assertions on each.
//!
//! Every handle here is built from a live database, never reopened by path —
//! the same constraint production construction obeys.

use std::sync::Arc;

use tempfile::TempDir;

use crate::config::StorageBackend;
use crate::storage::ColumnStore;

/// A fresh [`ColumnStore`] per backend with `families` available. The
/// `TempDir` must outlive the store.
pub(crate) fn column_stores(
    families: &[&str],
) -> Vec<(StorageBackend, Arc<dyn ColumnStore>, TempDir)> {
    let mut out: Vec<(StorageBackend, Arc<dyn ColumnStore>, TempDir)> = Vec::new();
    #[cfg(not(feature = "legacy-rocksdb"))]
    let _ = families;

    #[cfg(feature = "legacy-rocksdb")]
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

/// Refuses a run that silently dropped a backend this build enables: a loop over
/// one store passes every assertion while proving nothing about the other.
pub(crate) fn assert_both_backends<T>(stores: &[(StorageBackend, T, TempDir)]) {
    let backends: Vec<StorageBackend> = stores.iter().map(|(b, _, _)| *b).collect();
    #[cfg(feature = "legacy-rocksdb")]
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
