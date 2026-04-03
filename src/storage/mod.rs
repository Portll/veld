//! Storage abstraction layer for backend-agnostic persistence.
//!
//! The live runtime still uses RocksDB through compatibility code, but new
//! backends should integrate at this seam rather than through direct database
//! types in recall, graph traversal, or server orchestration.

pub mod legacy_rocksdb;
#[cfg(feature = "storage-redb")]
pub mod redb;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::backup::BackupMetadata;
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, RelationshipEdge};
use crate::memory::storage::{SearchCriteria, VectorMappingEntry};
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
    RocksDbBackupStore, RocksDbGraphStore, RocksDbPrimaryMemoryStore, RocksDbStorageFactory,
};
#[cfg(feature = "storage-redb")]
pub use redb::RedbStorageFactory;

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
