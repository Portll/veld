//! Storage abstraction layer for backend-agnostic persistence.
//!
//! The live runtime still uses RocksDB through compatibility code, but new
//! backends should integrate at this seam rather than through direct database
//! types in recall, graph traversal, or server orchestration.

pub mod legacy_rocksdb;
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

/// Generic key-value operations for shared secondary stores.
pub trait KeyValueStore: Send + Sync {
    fn backend(&self) -> StorageBackend;
    fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete_bytes(&self, key: &[u8]) -> Result<()>;
    fn scan_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn flush(&self) -> Result<()>;
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
    fn run_merge_pass(&self, user_id: &str, limit: usize) -> usize;
    fn gc_merge_tombstones(&self, user_id: &str) -> usize;
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
