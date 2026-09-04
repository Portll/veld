//! Storage abstraction layer for backend-agnostic persistence.
//!
//! The live runtime still uses RocksDB through compatibility code, but new
//! backends should integrate at this seam rather than through direct database
//! types in recall, graph traversal, or server orchestration.

pub mod legacy_rocksdb;
pub mod portable;
#[cfg(feature = "storage-redb")]
pub mod redb;
pub mod relational;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::backup::BackupMetadata;
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, RelationshipEdge};
use crate::memory::storage::{SearchCriteria, StorageStats, VectorMappingEntry};
use crate::memory::{Memory, MemoryConfig, MemoryId};

/// Capabilities that differ across storage backends and should be modeled
/// explicitly rather than assumed by higher-level code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageCapabilities {
    pub embedded: bool,
    pub default_target: bool,
    pub legacy_compatibility: bool,
    pub supports_prefix_scan: bool,
    pub supports_transactional_batch: bool,
    pub supports_snapshots: bool,
    pub supports_migrate_in_place: bool,
    pub supports_shared_multi_tenant_store: bool,
}

impl StorageCapabilities {
    pub const fn for_backend(backend: StorageBackend) -> Self {
        match backend {
            StorageBackend::Redb => Self {
                embedded: true,
                default_target: true,
                legacy_compatibility: false,
                supports_prefix_scan: true,
                supports_transactional_batch: true,
                supports_snapshots: true,
                supports_migrate_in_place: true,
                supports_shared_multi_tenant_store: true,
            },
            StorageBackend::RocksDb => Self {
                embedded: true,
                default_target: false,
                legacy_compatibility: true,
                supports_prefix_scan: true,
                supports_transactional_batch: true,
                supports_snapshots: true,
                supports_migrate_in_place: true,
                supports_shared_multi_tenant_store: true,
            },
        }
    }
}

/// The one shared engine database behind the global stores, carried by ENGINE
/// rather than erased. The consumers that remain engine-typed — user-auth, the
/// GDPR purge's raw column scans, sleep-time, the comprehensive backup — must
/// SAY which engine they need and refuse on the other, instead of discovering
/// the mismatch at their first raw call.
#[derive(Clone)]
pub enum SharedEngineDb {
    RocksDb(Arc<rocksdb::DB>),
    #[cfg(feature = "storage-redb")]
    Redb(Arc<::redb::Database>),
}

impl SharedEngineDb {
    pub fn backend(&self) -> StorageBackend {
        match self {
            SharedEngineDb::RocksDb(_) => StorageBackend::RocksDb,
            #[cfg(feature = "storage-redb")]
            SharedEngineDb::Redb(_) => StorageBackend::Redb,
        }
    }

    pub fn rocksdb(&self) -> Option<&Arc<rocksdb::DB>> {
        match self {
            SharedEngineDb::RocksDb(db) => Some(db),
            #[cfg(feature = "storage-redb")]
            SharedEngineDb::Redb(_) => None,
        }
    }

    /// `what` names the CAPABILITY, so the refusal reads as a statement about
    /// the caller's requirement, not about the database.
    pub fn require_rocksdb(&self, what: &str) -> Result<&Arc<rocksdb::DB>> {
        self.rocksdb().ok_or_else(|| {
            anyhow::anyhow!(
                "{what} requires the rocksdb engine; this deployment runs {:?}",
                self.backend()
            )
        })
    }

    #[cfg(feature = "storage-redb")]
    pub fn redb(&self) -> Option<&Arc<::redb::Database>> {
        match self {
            SharedEngineDb::Redb(db) => Some(db),
            SharedEngineDb::RocksDb(_) => None,
        }
    }
}

/// Lightweight audit entry shape for storage-level interfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub memory_id: String,
    pub details: String,
}

/// A column family by NAME rather than by handle. The handle types are
/// backend-specific (`rocksdb::ColumnFamily` has no redb counterpart), so a
/// batch that has to survive the cutover can only carry names.
pub type CfName = String;

/// One deferred write. `cf: None` means the backend's default keyspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchOp {
    Put {
        cf: Option<CfName>,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        cf: Option<CfName>,
        key: Vec<u8>,
    },
}

/// A backend-agnostic atomic write set.
///
/// **DB-scoped, not store-scoped, and that was not the obvious answer.** Both
/// `KeyValueStore` implementations are already bound to a single column family
/// — `RocksDbColumnFamilyStore` routes every operation through its own `cf()`
/// and `RedbKeyValueStore` binds one table — so hanging `write_batch` off
/// `KeyValueStore` looks natural and needs no per-op `cf` field at all.
///
/// It cannot express what `src/memory/prospective.rs:298` and `:382` already
/// do: both write `index_cf` and `tasks_cf` in ONE batch. A store-scoped batch
/// would force those into two batches, turning an atomic cross-family update
/// into two updates with a window between them — and nothing at the call site
/// would look any different. So the handle is DB-scoped, and
/// `StorageCapabilities::supports_transactional_batch` means "atomic ACROSS
/// column families", because that is the property those sites depend on. A
/// version quietly meaning "atomic within one family" would be true, useless,
/// and indistinguishable from the real thing where it matters.
///
/// Measured 2026-09-01 over `src/`: 54 `put_cf` + 52 `delete_cf` against 15
/// `put` + 12 `delete`, so 80% of batch operations are column-family scoped.
/// Batch and column-family are not independent extensions that could be done in
/// either order — batch requires CF.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteBatch {
    ops: Vec<BatchOp>,
}

impl WriteBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&mut self, key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) {
        self.ops.push(BatchOp::Put {
            cf: None,
            key: key.into(),
            value: value.into(),
        });
    }

    pub fn put_cf(
        &mut self,
        cf: impl Into<CfName>,
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
    ) {
        self.ops.push(BatchOp::Put {
            cf: Some(cf.into()),
            key: key.into(),
            value: value.into(),
        });
    }

    pub fn delete(&mut self, key: impl Into<Vec<u8>>) {
        self.ops.push(BatchOp::Delete {
            cf: None,
            key: key.into(),
        });
    }

    pub fn delete_cf(&mut self, cf: impl Into<CfName>, key: impl Into<Vec<u8>>) {
        self.ops.push(BatchOp::Delete {
            cf: Some(cf.into()),
            key: key.into(),
        });
    }

    /// Covers the 7 `batch.is_empty()` call sites in `src/`.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn ops(&self) -> &[BatchOp] {
        &self.ops
    }

    /// Every distinct column family this batch touches, `None` excluded.
    /// Backends resolve these BEFORE applying anything — see `write_batch`.
    pub fn column_families(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for op in &self.ops {
            let cf = match op {
                BatchOp::Put { cf, .. } | BatchOp::Delete { cf, .. } => cf.as_deref(),
            };
            if let Some(name) = cf {
                if !seen.contains(&name) {
                    seen.push(name);
                }
            }
        }
        seen
    }
}

/// Applies a [`WriteBatch`] atomically across column families.
///
/// Separate from the per-store traits on purpose: the atomicity that call sites
/// depend on is a property of the DATABASE, and a trait hung off a single store
/// cannot promise it.
///
/// Implementors MUST resolve every column family named by the batch before
/// applying any operation. Resolving as they go would apply a prefix and then
/// fail on an unknown name, which is exactly the partial write the capability
/// flag says cannot happen.
///
/// **Deliberately absent from [`StorageFactory`], and this is a constraint
/// rather than an omission.** Every other store here is opened by path, and the
/// obvious next step is `open_batch_writer(&self, path: &Path)`. It would be
/// wrong: a batch has to be atomic *with respect to the stores it writes
/// through*, so it must share their already-open database handle, and opening
/// the path a second time does not merely duplicate a handle — it fails on the
/// store lock. Observed on this tree: a second opener dies on
/// `IO error: While lock file: …/shared/LOCK`. Construct a writer FROM a live
/// handle (`RocksDbBatchWriter::new(db)`, `RedbBatchWriter::new(database)`),
/// never from a path.
pub trait BatchWriter: Send + Sync {
    fn backend(&self) -> StorageBackend;

    /// Apply all ops or none. An unknown column family is an error and writes
    /// nothing.
    fn write_batch(&self, batch: &WriteBatch) -> Result<()>;
}

/// Multi-column-family access — the shape five secondary stores actually need.
///
/// `KeyValueStore` is bound to ONE family, which fits the stores that have one.
/// `ContextBlockStore`, `FileMemoryStore`, `ProspectiveStore`, `TodoStore` and
/// `FeedbackStore` each address several: measured 2026-09-02, 7 / 21 / 35 / 63 / 22
/// column-family operations respectively, plus `prefix_iterator_cf` and batch
/// `write`. None of them is expressible through a single-family handle, which is why
/// all five still hold an `Arc<DB>` and none is reachable through `StorageFactory` —
/// and why the storage traits reaching 66/69 did not make the bootstrap wireable.
///
/// This is [`BatchWriter`] one step further: that gave DB-scoped WRITES because two
/// call sites batch across families; this gives DB-scoped reads and scans for the
/// same reason. Same argument, same scope.
pub trait ColumnStore: Send + Sync {
    fn backend(&self) -> StorageBackend;

    fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()>;

    /// One scan for both uses a caller has. `prefix: None` walks the family.
    ///
    /// Results are in ASCENDING key order — part of the contract, not an
    /// implementation accident: prospective's due-index scans early-exit at the
    /// first future timestamp, which is only correct when zero-padded keys come
    /// back lexicographically. The conformance script asserts it.
    ///
    /// `limit` is explicit and NOT defaulted, so an unbounded scan of a family that
    /// may hold hundreds of thousands of rows is a visible choice at the call site
    /// rather than the shape of the API. The prefix bound is the implementation's:
    /// rocksdb's iterator runs past a prefix, and every caller ported so far carried
    /// its own `starts_with` break — three of which were removed tonight, all silent
    /// if dropped.
    fn scan_cf(
        &self,
        cf: &str,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    fn flush_cf(&self, cf: &str) -> Result<()>;

    /// Declare a family before use. rocksdb families exist only if the DB was
    /// opened with their descriptors, so this VERIFIES — a store fails at
    /// construction instead of at first use; redb tables are created lazily,
    /// so this MATERIALIZES the table so a later read or batched write never
    /// hits TableDoesNotExist (put_cf creates on demand, write_batch refuses —
    /// found when a store's first touch of its index family was a batch).
    /// `RedbKeyValueStore::open` already does this for its one table; this is
    /// the multi-family parity.
    fn ensure_cf(&self, cf: &str) -> Result<()>;

    /// Atomic across families — see [`WriteBatch`].
    fn write_batch(&self, batch: &WriteBatch) -> Result<()>;
}

/// Generic key-value operations for shared secondary stores.
pub trait KeyValueStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete_bytes(&self, key: &[u8]) -> Result<()>;
    fn scan_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn flush(&self) -> Result<()>;
    /// Atomic multi-put WITHIN THIS STORE'S KEYSPACE. Required, not defaulted:
    /// a default that looped put_bytes would silently trade away the atomicity
    /// callers batch FOR, so every implementor states its own. Deliberately not
    /// the DB-scoped [`WriteBatch`]: a cf on a single-keyspace handle is the
    /// exact ambiguity the `cf: None` bug came from.
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()>;
    /// Atomic multi-delete within this store's keyspace.
    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()>;
}

/// Domain interface for the primary memory store.
///
/// The surface below the first block is what `MemorySystem` actually calls on its
/// `long_term_memory` field: 24 distinct methods across 94 call sites in exactly
/// two files (`memory/mod.rs`, `memory/recall.rs`), measured 2026-09-01. They are
/// declared here so the field can become `Arc<dyn PrimaryMemoryStore>` — the
/// boundary the storage abstraction was designed for and stalled at.
///
/// **`get` is NOT expressible in terms of `get_memory`, and that is why both are
/// here.** `RocksDbPrimaryMemoryStore::get_memory` classifies not-found by
/// `err.to_string().contains("Memory not found")`, so routing `get` through it
/// would send every genuine failure through a string match and return `Ok(None)`
/// for any error whose message someone later rewords — a decode failure, a
/// corrupt value or a permission error rendered as "this memory does not exist".
/// `get` therefore delegates straight to the backend and keeps its own error.
pub trait PrimaryMemoryStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>>;
    fn put_memory(&self, memory: &Memory) -> Result<()>;
    fn update_memory(&self, memory: &Memory) -> Result<()>;
    fn delete_memory(&self, id: &MemoryId) -> Result<()>;
    fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>>;
    fn get_vector_mapping(&self, id: &MemoryId) -> Result<Option<VectorMappingEntry>>;
    /// Every mapping at once — the retrieval engine's index rebuild reads the
    /// whole set, and this was its last reason to hold the concrete store.
    fn get_all_vector_mappings(&self) -> Result<Vec<(MemoryId, VectorMappingEntry)>>;
    /// Replace the TEXT-modality vectors for one memory, preserving other
    /// modalities — the write half of the retrieval engine's index maintenance.
    fn update_vector_mapping(&self, memory_id: &MemoryId, vector_ids: Vec<u32>) -> Result<()>;
    fn put_vector_mapping(&self, id: &MemoryId, mapping: &VectorMappingEntry) -> Result<()>;
    fn delete_vector_mapping(&self, id: &MemoryId) -> Result<()>;
    fn flush(&self) -> Result<()>;

    // ── What MemorySystem calls, by frequency. Signatures are MemoryStorage's
    //    exactly: an alias that "tidies" one of these changes behaviour at a call
    //    site that will not look any different.
    fn get(&self, id: &MemoryId) -> Result<Memory>;
    fn store(&self, memory: &Memory) -> Result<()>;
    fn update(&self, memory: &Memory) -> Result<()>;
    fn delete(&self, id: &MemoryId) -> Result<()>;
    fn get_all(&self) -> Result<Vec<Memory>>;
    fn get_all_ids(&self) -> Result<Vec<MemoryId>>;
    fn increment_retrieval_count(&self) -> Result<usize>;
    fn get_stats(&self) -> Result<StorageStats>;
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>>;
    fn get_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>>;
    fn get_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>>;
    fn get_hierarchy_context(
        &self,
        memory_id: &MemoryId,
    ) -> Result<(Vec<Memory>, Memory, Vec<Memory>)>;
    fn get_uncompressed_older_than(&self, cutoff: DateTime<Utc>) -> Result<Vec<Memory>>;
    fn mark_forgotten_by_age(&self, cutoff: DateTime<Utc>) -> Result<Vec<MemoryId>>;
    fn save_interference_records(
        &self,
        memory_id: &str,
        records: &[crate::memory::replay::InterferenceRecord],
    ) -> Result<()>;
    fn delete_interference_records(&self, memory_id: &str) -> Result<()>;
    fn clear_all_interference_records(&self) -> Result<usize>;
    fn cleanup_corrupted(&self) -> Result<usize>;
    fn migrate_legacy(&self) -> Result<(usize, usize, usize)>;
    // `-> Result<usize>`, not `-> usize`. The bare count could not express failure,
    // so an unbuilt backend could only return 0 — making "merged nothing" and
    // "cannot merge" the same answer, which is the absence-as-success this codebase
    // refuses everywhere else. The redb side panicked to keep them distinguishable;
    // with an error channel it can simply say so.
    fn run_merge_pass(&self, user_id: &str, limit: usize) -> Result<usize>;
    fn gc_merge_tombstones(&self, user_id: &str) -> Result<usize>;

    /// Every key under `prefix`, values not loaded.
    ///
    /// Replaces `db().prefix_iterator(p)` at the one call site that reached past
    /// this trait for it. Two differences from the raw iterator, both of which the
    /// call site was compensating for by hand: rocksdb's prefix iterator may run
    /// PAST the prefix, so that code carried `if !key.starts_with(prefix) { break }`
    /// and a missing break would have deleted unrelated keys; and it yields values
    /// nobody there wanted. Implementations MUST return only matching keys.
    fn scan_prefix_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>>;

    /// Apply `batch` atomically against this store's database.
    ///
    /// Present on the store rather than reached through a separate handle because
    /// the batch has to be atomic with respect to THIS store's writes — see
    /// [`BatchWriter`], which explains why a writer cannot be opened by path.
    fn write_batch(&self, batch: &WriteBatch) -> Result<()>;

    // ── The eight methods the long_term_memory flip surfaced (2026-09-02). The
    //    handoff said the trait already carried every method the 94 call sites
    //    use; the compiler said otherwise — these existed only on concrete
    //    MemoryStorage. Signatures are MemoryStorage's exactly, per the
    //    discipline above: set_fact_watermark returns () and logs,
    //    get_by_content_hash returns Option with no error channel, because that
    //    is what the call sites already absorb — a stub there could not even
    //    fail loudly, so both backends must answer for real.
    fn get_by_content_hash(&self, content: &str) -> Option<MemoryId>;
    fn update_access(&self, id: &MemoryId) -> Result<()>;
    fn mark_forgotten_by_importance(&self, threshold: f32) -> Result<Vec<MemoryId>>;
    fn save_interference_event_count(&self, count: usize) -> Result<()>;
    fn get_fact_watermark(&self, user_id: &str) -> Option<i64>;
    fn set_fact_watermark(&self, user_id: &str, timestamp_millis: i64);
    /// Backend-agnostic KV handle over the general-purpose `kv` keyspace.
    fn key_value_store(&self) -> std::sync::Arc<dyn KeyValueStore>;
    /// KV handle over the DEFAULT keyspace — a DIFFERENT keyspace from `kv`;
    /// routing a default-keyspace consumer onto `kv` orphans every row it
    /// thought it wrote (the ACL store found this the hard way).
    fn default_key_value_store(&self) -> std::sync::Arc<dyn KeyValueStore>;
}

/// Domain interface for graph persistence.
pub trait GraphStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn get_entity(&self, uuid: &uuid::Uuid) -> Result<Option<EntityNode>>;
    fn put_entity(&self, entity: &EntityNode) -> Result<()>;
    fn get_relationship(&self, uuid: &uuid::Uuid) -> Result<Option<RelationshipEdge>>;
    fn put_relationship(&self, edge: &RelationshipEdge) -> Result<()>;
    fn list_relationships_for_entity(&self, entity_uuid: &uuid::Uuid) -> Result<Vec<RelationshipEdge>>;
    fn get_episode(&self, uuid: &uuid::Uuid) -> Result<Option<EpisodicNode>>;
    fn put_episode(&self, episode: &EpisodicNode) -> Result<()>;
    fn stats(&self) -> Result<GraphStats>;
    fn flush(&self) -> Result<()>;
}

/// Domain interface for audit storage.
pub trait AuditStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn append_event(&self, user_id: &str, event: &AuditLogEntry) -> Result<()>;
    fn list_events(&self, user_id: &str, limit: usize) -> Result<Vec<AuditLogEntry>>;
    fn rotate_events(
        &self,
        user_id: &str,
        max_entries: usize,
        retention_before: DateTime<Utc>,
    ) -> Result<usize>;
    fn flush(&self) -> Result<()>;
    /// Every user id with at least one audit row — the enumeration audit
    /// rotation needs. On the trait because the raw form (scanning the engine's
    /// keyspace for `{user}:` prefixes) was the last reason rotation reached
    /// past this abstraction to a concrete database handle.
    fn list_user_ids(&self) -> Result<Vec<String>>;
}

/// Backup surface that higher-level orchestration can use without embedding
/// backend-specific snapshot APIs.
pub trait BackupStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn create_backup(&self, user_id: &str) -> Result<BackupMetadata>;
    fn list_backups(&self, user_id: &str) -> Result<Vec<BackupMetadata>>;
    fn restore_backup(&self, user_id: &str, backup_id: u32, target_path: &Path) -> Result<()>;
}

/// Migration surface for portable export/import and backend cutover.
pub trait MigrationStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn export_portable(&self, user_id: &str, target: &Path) -> Result<()>;
    fn import_portable(&self, user_id: &str, source: &Path) -> Result<()>;
    fn migrate_in_place(
        &self,
        user_id: &str,
        target_backend: StorageBackend,
        target_path: &Path,
    ) -> Result<()>;
}

/// Factory for opening backend-specific store implementations behind the
/// backend-agnostic traits above.
pub trait StorageFactory: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn capabilities(&self) -> StorageCapabilities {
        StorageCapabilities::for_backend(self.backend())
    }
    fn open_primary_memory_store(
        &self,
        path: &Path,
        config: &MemoryConfig,
    ) -> Result<Arc<dyn PrimaryMemoryStore>>;
    fn open_graph_store(&self, path: &Path) -> Result<Arc<dyn GraphStore>>;
    fn open_shared_store(&self, path: &Path, namespace: &str) -> Result<Arc<dyn KeyValueStore>>;
    fn open_audit_store(&self, path: &Path) -> Result<Arc<dyn AuditStore>>;
    fn open_backup_store(&self, path: &Path) -> Result<Arc<dyn BackupStore>>;
    fn open_migration_store(&self, path: &Path) -> Result<Arc<dyn MigrationStore>>;
}

pub use legacy_rocksdb::{
    RocksDbBackupStore, RocksDbBatchWriter, RocksDbGraphStore, RocksDbPrimaryMemoryStore,
    RocksDbStorageFactory,
};
#[cfg(feature = "storage-redb")]
pub use redb::{RedbBatchWriter, RedbStorageFactory};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redb_is_the_default_target_backend() {
        let capabilities = StorageCapabilities::for_backend(StorageBackend::Redb);
        assert!(capabilities.default_target);
        assert!(!capabilities.legacy_compatibility);
        assert!(capabilities.supports_transactional_batch);
    }

    #[test]
    fn rocksdb_is_marked_legacy_compatibility() {
        let capabilities = StorageCapabilities::for_backend(StorageBackend::RocksDb);
        assert!(!capabilities.default_target);
        assert!(capabilities.legacy_compatibility);
        assert!(capabilities.supports_prefix_scan);
    }

    #[test]
    fn a_batch_carries_the_column_family_each_op_named() {
        let mut b = WriteBatch::new();
        b.put_cf("index", b"k1".to_vec(), b"v1".to_vec());
        b.delete_cf("tasks", b"k2".to_vec());
        b.put(b"k3".to_vec(), b"v3".to_vec());

        assert_eq!(b.len(), 3);
        assert!(!b.is_empty());
        // The default keyspace is not a column family and must not appear here,
        // or a backend would try to resolve a handle for it.
        assert_eq!(b.column_families(), vec!["index", "tasks"]);
    }

    #[test]
    fn an_empty_batch_is_empty_and_names_nothing() {
        let b = WriteBatch::new();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert!(b.column_families().is_empty());
        // Asserted because 7 call sites branch on is_empty() to skip the write
        // entirely; a batch that reported non-empty here would turn those into
        // writes of nothing.
    }

    #[test]
    fn the_multi_family_case_the_api_shape_exists_for() {
        // src/memory/prospective.rs:298 and :382 write index_cf and tasks_cf in
        // ONE batch. A store-scoped batch cannot express this, which is the
        // whole reason BatchWriter hangs off the database. Measured 2026-09-01:
        // 2 such blocks in src/, both in prospective.rs.
        let mut b = WriteBatch::new();
        b.put_cf("index", b"i".to_vec(), b"1".to_vec());
        b.put_cf("tasks", b"t".to_vec(), b"1".to_vec());
        assert_eq!(
            b.column_families().len(),
            2,
            "if this is ever 1, the batch has been collapsed onto a single \
             family and the atomic cross-family update has become two writes"
        );
    }

    #[test]
    fn column_families_deduplicates_but_keeps_first_appearance_order() {
        let mut b = WriteBatch::new();
        b.put_cf("b", b"1".to_vec(), b"x".to_vec());
        b.put_cf("a", b"2".to_vec(), b"x".to_vec());
        b.delete_cf("b", b"3".to_vec());
        // Deterministic: same input, same order out. Backends resolve handles
        // from this list, and a set with unstable iteration order would make a
        // resolution FAILURE report a different family run to run.
        assert_eq!(b.column_families(), vec!["b", "a"]);
    }

    #[test]
    fn rocksdb_factory_reports_backend_and_capabilities() {
        let factory = RocksDbStorageFactory::default();
        assert_eq!(factory.backend(), StorageBackend::RocksDb);
        assert!(factory.capabilities().legacy_compatibility);
    }

    #[cfg(feature = "storage-redb")]
    #[test]
    fn redb_factory_reports_backend_and_capabilities() {
        let factory = RedbStorageFactory;
        assert_eq!(factory.backend(), StorageBackend::Redb);
        assert!(factory.capabilities().default_target);
    }
}

#[cfg(test)]
mod backend_coverage_matrix {
    //! Executable coverage: every PrimaryMemoryStore method CALLED on both backends.
    //!
    //! A static count of `unbuilt(` in the source is one witness and a weak one — it
    //! cannot see a method that compiles, is reached, and fails for another reason,
    //! and it counts a refusal in a comment. This calls all of them and records what
    //! actually happens, which is a different predicate over the same question.
    use super::*;
    use crate::memory::types::{Experience, ExperienceType, MemoryTier};
    use crate::storage::PrimaryMemoryStore;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum Outcome { Works, Refuses, Panics }

    fn probe<T>(f: impl FnOnce() -> Result<T>) -> Outcome {
        match catch_unwind(AssertUnwindSafe(f)) {
            Err(_) => Outcome::Panics,
            Ok(Ok(_)) => Outcome::Works,
            Ok(Err(_)) => Outcome::Refuses,
        }
    }

    fn a_memory() -> Memory {
        let now = chrono::Utc::now();
        Memory::from_legacy(
            MemoryId(uuid::Uuid::new_v4()),
            Experience { experience_type: ExperienceType::Observation, content: "probe".into(), ..Default::default() },
            0.5, 0, now, now, false, MemoryTier::LongTerm, Vec::new(), 1.0,
            None, None, None, None, 0.0, None, None, 1, Vec::new(), Vec::new(),
        )
    }

    /// Calls all 43 trait methods and returns (name, outcome), sorted.
    fn matrix(s: &dyn PrimaryMemoryStore) -> Vec<(&'static str, Outcome)> {
        let m = a_memory();
        let id = m.id.clone();
        let cutoff = chrono::Utc::now();
        let mut out = vec![
            ("backend",                       probe(|| Ok::<_, anyhow::Error>(s.backend()))),
            ("put_memory",                    probe(|| s.put_memory(&m))),
            ("get_memory",                    probe(|| s.get_memory(&id))),
            ("update_memory",                 probe(|| s.update_memory(&m))),
            ("search",                        probe(|| s.search(crate::memory::storage::SearchCriteria::ByTags(vec!["x".into()])))),
            ("get_vector_mapping",            probe(|| s.get_vector_mapping(&id))),
            ("put_vector_mapping",            probe(|| s.put_vector_mapping(&id, &Default::default()))),
            ("delete_vector_mapping",         probe(|| s.delete_vector_mapping(&id))),
            ("flush",                         probe(|| s.flush())),
            ("get",                           probe(|| s.get(&id))),
            ("store",                         probe(|| s.store(&m))),
            ("update",                        probe(|| s.update(&m))),
            ("get_all",                       probe(|| s.get_all())),
            ("get_all_ids",                   probe(|| s.get_all_ids())),
            ("increment_retrieval_count",     probe(|| s.increment_retrieval_count())),
            ("get_stats",                     probe(|| s.get_stats())),
            ("find_by_external_id",           probe(|| s.find_by_external_id("nope"))),
            ("get_children",                  probe(|| s.get_children(&id))),
            ("get_ancestors",                 probe(|| s.get_ancestors(&id))),
            ("get_hierarchy_context",         probe(|| s.get_hierarchy_context(&id))),
            ("get_uncompressed_older_than",   probe(|| s.get_uncompressed_older_than(cutoff))),
            ("mark_forgotten_by_age",         probe(|| s.mark_forgotten_by_age(cutoff))),
            ("save_interference_records",     probe(|| s.save_interference_records("m", &[]))),
            ("delete_interference_records",   probe(|| s.delete_interference_records("m"))),
            ("clear_all_interference_records",probe(|| s.clear_all_interference_records())),
            ("cleanup_corrupted",             probe(|| s.cleanup_corrupted())),
            ("migrate_legacy",                probe(|| s.migrate_legacy())),
            ("scan_prefix_keys",              probe(|| s.scan_prefix_keys(b"x"))),
            ("write_batch",                   probe(|| s.write_batch(&crate::storage::WriteBatch::new()))),
            ("run_merge_pass",                probe(|| s.run_merge_pass("u", 1))),
            ("gc_merge_tombstones",           probe(|| s.gc_merge_tombstones("u"))),
            // The eight surfaced by the long_term_memory flip. The KV probes
            // exercise the handle they get back, not just its construction —
            // a handle over the wrong keyspace constructs fine.
            ("get_all_vector_mappings",       probe(|| s.get_all_vector_mappings())),
            ("update_vector_mapping",         probe(|| s.update_vector_mapping(&id, vec![1]))),
            ("get_by_content_hash",           probe(|| Ok::<_, anyhow::Error>(s.get_by_content_hash("probe")))),
            ("update_access",                 probe(|| s.update_access(&id))),
            ("mark_forgotten_by_importance",  probe(|| s.mark_forgotten_by_importance(0.0))),
            ("save_interference_event_count", probe(|| s.save_interference_event_count(1))),
            ("get_fact_watermark",            probe(|| Ok::<_, anyhow::Error>(s.get_fact_watermark("u")))),
            ("set_fact_watermark",            probe(|| Ok::<_, anyhow::Error>(s.set_fact_watermark("u", 1)))),
            ("key_value_store",               probe(|| {
                let kv = s.key_value_store();
                kv.put_bytes(b"matrix-probe", b"1")?;
                kv.get_bytes(b"matrix-probe")
            })),
            ("default_key_value_store",       probe(|| {
                let kv = s.default_key_value_store();
                kv.put_bytes(b"matrix-probe-default", b"1")?;
                kv.get_bytes(b"matrix-probe-default")
            })),
            ("delete_memory",                 probe(|| s.delete_memory(&id))),
            ("delete",                        probe(|| s.delete(&id))),
        ];
        out.sort_by_key(|(n, _)| *n);
        out
    }

    #[test]
    fn every_trait_method_is_called_on_both_backends_and_the_gap_is_exact() {
        let td = tempfile::tempdir().unwrap();
        let rocks = crate::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore::open(
            &td.path().join("rocks"), None).expect("rocks open");
        let redb = crate::storage::redb::RedbPrimaryMemoryStore::open(
            &td.path().join("redb")).expect("redb open");

        let r = matrix(&rocks);
        let b = matrix(&redb);
        assert_eq!(r.len(), 43, "the matrix must cover every trait method");
        assert_eq!(r.len(), b.len());

        let rocks_ok = r.iter().filter(|(_, o)| *o == Outcome::Works).count();
        let redb_ok  = b.iter().filter(|(_, o)| *o == Outcome::Works).count();
        let gap: Vec<&str> = r.iter().zip(&b)
            .filter(|((_, ro), (_, bo))| *ro == Outcome::Works && *bo != Outcome::Works)
            .map(|((n, _), _)| *n).collect();

        eprintln!("COVERAGE  rocksdb {rocks_ok}/43   redb {redb_ok}/43   gap {}", gap.len());
        for name in &gap { eprintln!("  redb missing: {name}"); }

        // The direction that matters: redb must never be ahead in a way that hides a
        // rocksdb regression, and the gap list must be exactly what is claimed.
        assert!(rocks_ok >= redb_ok, "redb reports more working methods than rocksdb — check the probe, not the backends");
        assert_eq!(gap.len(), 43 - redb_ok, "a redb method failed that rocksdb also fails — the gap list is not the whole story");
    }
}

#[cfg(test)]
mod column_store_conformance {
    //! The same script against both backends, compared.
    //!
    //! Two separate test suites prove each implementation works on its own terms,
    //! which is exactly what a divergence looks like from the inside — the `cf: None`
    //! bug in the batch API passed both sides' tests while landing in two different
    //! keyspaces. This runs ONE sequence and asserts the answers are equal, so a
    //! disagreement fails rather than being invisible until a cutover.
    use super::*;
    use crate::storage::legacy_rocksdb::RocksDbColumnStore;
    use crate::storage::redb::RedbColumnStore;

    const FAMILIES: [&str; 4] = ["default", "alpha", "beta", "gamma"];

    /// Every observable answer the script produces, as comparable values.
    fn script(s: &dyn ColumnStore) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut note = |k: &str, v: String| out.push((k.to_string(), v));

        note("backend-differs-on-purpose", format!("{:?}", s.backend()));

        s.put_cf("alpha", b"k1", b"v1").unwrap();
        s.put_cf("alpha", b"k2", b"v2").unwrap();
        s.put_cf("beta", b"k1", b"OTHER").unwrap();

        note("get alpha/k1", format!("{:?}", s.get_cf("alpha", b"k1").unwrap()));
        // Isolation: the same key in another family must not be visible here.
        note("get beta/k1", format!("{:?}", s.get_cf("beta", b"k1").unwrap()));
        note("get alpha/missing", format!("{:?}", s.get_cf("alpha", b"nope").unwrap()));

        s.put_cf("alpha", b"pre:a", b"1").unwrap();
        s.put_cf("alpha", b"pre:b", b"2").unwrap();
        s.put_cf("alpha", b"qqq", b"3").unwrap();

        // The prefix bound: `qqq` sorts after `pre:` and must NOT be returned.
        let pre = s.scan_cf("alpha", Some(b"pre:"), None).unwrap();
        note("scan prefix", format!("{:?}", pre.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>()));
        // Ascending order is contract, not accident — the due-index early exit
        // depends on it. A hard assert, not a note: two backends agreeing on a
        // wrong order would pass the equality comparison.
        assert!(
            pre.windows(2).all(|w| w[0].0 < w[1].0),
            "scan_cf must return keys in ascending order"
        );
        note("scan prefix limit 1", format!("{}", s.scan_cf("alpha", Some(b"pre:"), Some(1)).unwrap().len()));
        note("scan whole family", format!("{}", s.scan_cf("alpha", None, None).unwrap().len()));

        let mut b = WriteBatch::new();
        b.put_cf("alpha", b"batched".to_vec(), b"A".to_vec());
        b.put_cf("beta", b"batched".to_vec(), b"B".to_vec());
        b.delete_cf("alpha", b"k2".to_vec());
        s.write_batch(&b).unwrap();

        note("after batch alpha/batched", format!("{:?}", s.get_cf("alpha", b"batched").unwrap()));
        note("after batch beta/batched", format!("{:?}", s.get_cf("beta", b"batched").unwrap()));
        note("after batch alpha/k2 deleted", format!("{:?}", s.get_cf("alpha", b"k2").unwrap()));

        // A family whose FIRST touch is a batched write — the case a put-first
        // script structurally cannot compare: redb's put_cf creates tables on
        // demand while write_batch refuses missing ones, so every family the
        // script had already put to masked the divergence. ensure_cf is the
        // contract that makes the two backends answer identically here.
        s.ensure_cf("gamma").unwrap();
        let mut g = WriteBatch::new();
        g.put_cf("gamma", b"gk".to_vec(), b"gv".to_vec());
        s.write_batch(&g).unwrap();
        note("fresh family, first touch batched", format!("{:?}", s.get_cf("gamma", b"gk").unwrap()));

        s.delete_cf("alpha", b"k1").unwrap();
        note("after delete alpha/k1", format!("{:?}", s.get_cf("alpha", b"k1").unwrap()));
        // Deleting an absent key is not an error on either backend.
        note("delete absent is ok", format!("{:?}", s.delete_cf("alpha", b"never").is_ok()));
        note("flush", format!("{:?}", s.flush_cf("alpha").is_ok()));
        out
    }

    #[test]
    fn both_backends_answer_the_same_script_identically() {
        let tmp = tempfile::tempdir().unwrap();

        let rocks_db = {
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            opts.create_missing_column_families(true);
            let cfs: Vec<_> = FAMILIES
                .iter()
                .map(|n| rocksdb::ColumnFamilyDescriptor::new(*n, rocksdb::Options::default()))
                .collect();
            std::sync::Arc::new(rocksdb::DB::open_cf_descriptors(&opts, tmp.path().join("rocks"), cfs).unwrap())
        };
        let redb_db = crate::storage::redb::create_redb_database_for_test(&tmp.path().join("redb.redb")).unwrap();

        let r = script(&RocksDbColumnStore::new(rocks_db));
        let b = script(&RedbColumnStore::new(redb_db));

        assert_eq!(r.len(), b.len());
        let mut diffs = Vec::new();
        for ((rk, rv), (_, bv)) in r.iter().zip(&b) {
            // The backend name is the one answer that SHOULD differ.
            if rk == "backend-differs-on-purpose" {
                assert_ne!(rv, bv, "the two stores reported the same backend — the test built one twice");
                continue;
            }
            if rv != bv {
                diffs.push(format!("  {rk}:  rocksdb={rv}  redb={bv}"));
            }
        }
        assert!(diffs.is_empty(), "the backends disagree:\n{}", diffs.join("\n"));

        // NOT VACUOUS. Two backends that both returned None to everything would agree
        // perfectly, so the comparison above is only worth running if the script
        // actually observed something. These pin the answers that carry the meaning.
        let find = |k: &str| r.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).unwrap();
        assert_eq!(find("get alpha/k1"), "Some([118, 49])", "the round trip returned nothing to compare");
        assert_eq!(find("get beta/k1"), "Some([79, 84, 72, 69, 82])",
            "families are not isolated — the same key read back the other family's value");
        assert_eq!(find("get alpha/missing"), "None");
        assert_eq!(find("scan prefix"), "[[112, 114, 101, 58, 97], [112, 114, 101, 58, 98]]",
            "the prefix scan did not return exactly the two pre: keys — `qqq` sorts after them and \
             must not be included");
        assert_eq!(find("scan prefix limit 1"), "1");
        assert_eq!(find("after batch alpha/batched"), "Some([65])");
        assert_eq!(find("after batch beta/batched"), "Some([66])",
            "the cross-family batch did not reach the second family");
        assert_eq!(find("after batch alpha/k2 deleted"), "None");
        assert_eq!(find("after delete alpha/k1"), "None");
    }
}
