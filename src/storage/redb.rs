//! redb scaffolding for the storage abstraction layer.
//!
//! These implementations intentionally establish the crate-level integration
//! surface without claiming that the backend is operational yet.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{
    AuditLogEntry, AuditStore, BackupStore, GraphStore, KeyValueStore, MigrationStore,
    PrimaryMemoryStore, StorageFactory,
};
use crate::backup::BackupMetadata;
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, RelationshipEdge};
use crate::memory::storage::{SearchCriteria, VectorMappingEntry};
use crate::memory::{Memory, MemoryConfig, MemoryId};

fn not_implemented(operation: &str, path: &Path) -> anyhow::Error {
    anyhow!(
        "redb {} is not implemented yet for {}",
        operation,
        path.display()
    )
}

fn create_redb_database(path: &Path) -> Result<Arc<redb::Database>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(Arc::new(redb::Database::create(path)?))
}

pub struct RedbPrimaryMemoryStore {
    _database: Arc<redb::Database>,
    path: PathBuf,
}

impl RedbPrimaryMemoryStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("storage.redb");
        Ok(Self {
            _database: create_redb_database(&database_path)?,
            path: database_path,
        })
    }
}

impl PrimaryMemoryStore for RedbPrimaryMemoryStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_memory(&self, _id: &MemoryId) -> Result<Option<Memory>> {
        Err(not_implemented("primary memory read", &self.path))
    }

    fn put_memory(&self, _memory: &Memory) -> Result<()> {
        Err(not_implemented("primary memory write", &self.path))
    }

    fn update_memory(&self, _memory: &Memory) -> Result<()> {
        Err(not_implemented("primary memory update", &self.path))
    }

    fn delete_memory(&self, _id: &MemoryId) -> Result<()> {
        Err(not_implemented("primary memory delete", &self.path))
    }

    fn search(&self, _criteria: SearchCriteria) -> Result<Vec<Memory>> {
        Err(not_implemented("primary memory search", &self.path))
    }

    fn get_vector_mapping(&self, _id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        Err(not_implemented("vector mapping read", &self.path))
    }

    fn put_vector_mapping(&self, _id: &MemoryId, _mapping: &VectorMappingEntry) -> Result<()> {
        Err(not_implemented("vector mapping write", &self.path))
    }

    fn delete_vector_mapping(&self, _id: &MemoryId) -> Result<()> {
        Err(not_implemented("vector mapping delete", &self.path))
    }

    fn flush(&self) -> Result<()> {
        Err(not_implemented("primary memory flush", &self.path))
    }
}

pub struct RedbGraphStore {
    _database: Arc<redb::Database>,
    path: PathBuf,
}

impl RedbGraphStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("graph.redb");
        Ok(Self {
            _database: create_redb_database(&database_path)?,
            path: database_path,
        })
    }
}

impl GraphStore for RedbGraphStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_entity(&self, _uuid: &uuid::Uuid) -> Result<Option<EntityNode>> {
        Err(not_implemented("graph entity read", &self.path))
    }

    fn put_entity(&self, _entity: &EntityNode) -> Result<()> {
        Err(not_implemented("graph entity write", &self.path))
    }

    fn get_relationship(&self, _uuid: &uuid::Uuid) -> Result<Option<RelationshipEdge>> {
        Err(not_implemented("graph relationship read", &self.path))
    }

    fn put_relationship(&self, _edge: &RelationshipEdge) -> Result<()> {
        Err(not_implemented("graph relationship write", &self.path))
    }

    fn list_relationships_for_entity(
        &self,
        _entity_uuid: &uuid::Uuid,
    ) -> Result<Vec<RelationshipEdge>> {
        Err(not_implemented("graph relationship scan", &self.path))
    }

    fn get_episode(&self, _uuid: &uuid::Uuid) -> Result<Option<EpisodicNode>> {
        Err(not_implemented("graph episode read", &self.path))
    }

    fn put_episode(&self, _episode: &EpisodicNode) -> Result<()> {
        Err(not_implemented("graph episode write", &self.path))
    }

    fn stats(&self) -> Result<GraphStats> {
        Err(not_implemented("graph stats", &self.path))
    }

    fn flush(&self) -> Result<()> {
        Err(not_implemented("graph flush", &self.path))
    }
}

pub struct RedbKeyValueStore {
    _database: Arc<redb::Database>,
    path: PathBuf,
    namespace: String,
}

impl RedbKeyValueStore {
    pub fn open(path: &Path, namespace: &str) -> Result<Self> {
        let database_path = path.join(format!("{}.redb", namespace));
        Ok(Self {
            _database: create_redb_database(&database_path)?,
            path: database_path,
            namespace: namespace.to_string(),
        })
    }
}

impl KeyValueStore for RedbKeyValueStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_bytes(&self, _key: &[u8]) -> Result<Option<Vec<u8>>> {
        Err(not_implemented(
            &format!("key-value read in namespace '{}'", self.namespace),
            &self.path,
        ))
    }

    fn put_bytes(&self, _key: &[u8], _value: &[u8]) -> Result<()> {
        Err(not_implemented(
            &format!("key-value write in namespace '{}'", self.namespace),
            &self.path,
        ))
    }

    fn delete_bytes(&self, _key: &[u8]) -> Result<()> {
        Err(not_implemented(
            &format!("key-value delete in namespace '{}'", self.namespace),
            &self.path,
        ))
    }

    fn scan_prefix(&self, _prefix: &[u8], _limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Err(not_implemented(
            &format!("key-value prefix scan in namespace '{}'", self.namespace),
            &self.path,
        ))
    }

    fn flush(&self) -> Result<()> {
        Err(not_implemented(
            &format!("key-value flush in namespace '{}'", self.namespace),
            &self.path,
        ))
    }
}

pub struct RedbAuditStore {
    _database: Arc<redb::Database>,
    path: PathBuf,
}

impl RedbAuditStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("audit.redb");
        Ok(Self {
            _database: create_redb_database(&database_path)?,
            path: database_path,
        })
    }
}

impl AuditStore for RedbAuditStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn append_event(&self, _user_id: &str, _event: &AuditLogEntry) -> Result<()> {
        Err(not_implemented("audit append", &self.path))
    }

    fn list_events(&self, _user_id: &str, _limit: usize) -> Result<Vec<AuditLogEntry>> {
        Err(not_implemented("audit list", &self.path))
    }

    fn rotate_events(
        &self,
        _user_id: &str,
        _max_entries: usize,
        _retention_before: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        Err(not_implemented("audit rotation", &self.path))
    }

    fn flush(&self) -> Result<()> {
        Err(not_implemented("audit flush", &self.path))
    }
}

pub struct RedbBackupStore {
    path: PathBuf,
}

impl RedbBackupStore {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl BackupStore for RedbBackupStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn create_backup(&self, _user_id: &str) -> Result<BackupMetadata> {
        Err(not_implemented("backup create", &self.path))
    }

    fn list_backups(&self, _user_id: &str) -> Result<Vec<BackupMetadata>> {
        Err(not_implemented("backup list", &self.path))
    }

    fn restore_backup(&self, _user_id: &str, _backup_id: u32, _target_path: &Path) -> Result<()> {
        Err(not_implemented("backup restore", &self.path))
    }
}

pub struct RedbMigrationStore {
    path: PathBuf,
}

impl RedbMigrationStore {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl MigrationStore for RedbMigrationStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn export_portable(&self, _user_id: &str, _target: &Path) -> Result<()> {
        Err(not_implemented("portable export", &self.path))
    }

    fn import_portable(&self, _user_id: &str, _source: &Path) -> Result<()> {
        Err(not_implemented("portable import", &self.path))
    }

    fn migrate_in_place(
        &self,
        _user_id: &str,
        _target_backend: StorageBackend,
        _target_path: &Path,
    ) -> Result<()> {
        Err(not_implemented("in-place migration", &self.path))
    }
}

#[derive(Default)]
pub struct RedbStorageFactory;

impl StorageFactory for RedbStorageFactory {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn open_primary_memory_store(
        &self,
        path: &Path,
        _config: &MemoryConfig,
    ) -> Result<Arc<dyn PrimaryMemoryStore>> {
        Ok(Arc::new(RedbPrimaryMemoryStore::open(path)?))
    }

    fn open_graph_store(&self, path: &Path) -> Result<Arc<dyn GraphStore>> {
        Ok(Arc::new(RedbGraphStore::open(path)?))
    }

    fn open_shared_store(&self, path: &Path, namespace: &str) -> Result<Arc<dyn KeyValueStore>> {
        Ok(Arc::new(RedbKeyValueStore::open(path, namespace)?))
    }

    fn open_audit_store(&self, path: &Path) -> Result<Arc<dyn AuditStore>> {
        Ok(Arc::new(RedbAuditStore::open(path)?))
    }

    fn open_backup_store(&self, path: &Path) -> Result<Arc<dyn BackupStore>> {
        Ok(Arc::new(RedbBackupStore::open(path)?))
    }

    fn open_migration_store(&self, path: &Path) -> Result<Arc<dyn MigrationStore>> {
        Ok(Arc::new(RedbMigrationStore::open(path)?))
    }
}