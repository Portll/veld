//! RocksDB compatibility adapters for the storage abstraction layer.
//!
//! These wrappers preserve current runtime behavior while moving call sites off
//! direct RocksDB types.

use anyhow::{anyhow, Context, Result};
use rocksdb::{ColumnFamily, Direction, IteratorMode, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;

use super::{
    AuditLogEntry, AuditStore, BackupStore, GraphStore, KeyValueStore, MigrationStore,
    PrimaryMemoryStore, StorageFactory,
};
use crate::backup::{BackupMetadata, VeldBackupEngine};
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphMemory, GraphStats, RelationshipEdge};
use crate::memory::storage::{MemoryStorage, SearchCriteria, VectorMappingEntry};
use crate::memory::{Memory, MemoryConfig, MemoryId};

/// Trait adapter for the current RocksDB-backed primary memory store.
pub struct RocksDbPrimaryMemoryStore {
    inner: MemoryStorage,
}

impl RocksDbPrimaryMemoryStore {
    pub fn open(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        Ok(Self {
            inner: MemoryStorage::new(path, shared_cache)?,
        })
    }

    pub fn from_inner(inner: MemoryStorage) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &MemoryStorage {
        &self.inner
    }

    pub fn into_inner(self) -> MemoryStorage {
        self.inner
    }
}

impl PrimaryMemoryStore for RocksDbPrimaryMemoryStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        match self.inner.get(id) {
            Ok(memory) => Ok(Some(memory)),
            Err(err) if err.to_string().contains("Memory not found") => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn put_memory(&self, memory: &Memory) -> Result<()> {
        self.inner.store(memory)
    }

    fn update_memory(&self, memory: &Memory) -> Result<()> {
        self.inner.update(memory)
    }

    fn delete_memory(&self, id: &MemoryId) -> Result<()> {
        self.inner.delete_with_vectors(id)
    }

    fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>> {
        self.inner.search(criteria)
    }

    fn get_vector_mapping(&self, id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        self.inner.get_vector_mapping(id)
    }

    fn put_vector_mapping(&self, id: &MemoryId, mapping: &VectorMappingEntry) -> Result<()> {
        let mapping_key = format!("vmapping:{}", id.0);
        let mapping_value =
            bincode::serde::encode_to_vec(mapping, bincode::config::standard())
                .context("Failed to serialize vector mapping for legacy RocksDB store")?;

        self.inner
            .db()
            .put(mapping_key.as_bytes(), mapping_value)
            .context("Failed to write vector mapping into legacy RocksDB store")?;

        Ok(())
    }

    fn delete_vector_mapping(&self, id: &MemoryId) -> Result<()> {
        self.inner.delete_vector_mapping(id)
    }

    fn flush(&self) -> Result<()> {
        self.inner.flush()
    }
}

/// Trait adapter for the current RocksDB-backed graph store.
pub struct RocksDbGraphStore {
    inner: GraphMemory,
}

impl RocksDbGraphStore {
    pub fn open(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        Ok(Self {
            inner: GraphMemory::new(path, shared_cache)?,
        })
    }

    pub fn from_inner(inner: GraphMemory) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &GraphMemory {
        &self.inner
    }

    pub fn into_inner(self) -> GraphMemory {
        self.inner
    }
}

impl GraphStore for RocksDbGraphStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn get_entity(&self, uuid: &uuid::Uuid) -> Result<Option<EntityNode>> {
        self.inner.get_entity(uuid)
    }

    fn put_entity(&self, entity: &EntityNode) -> Result<()> {
        self.inner.add_entity(entity.clone()).map(|_| ())
    }

    fn get_relationship(&self, uuid: &uuid::Uuid) -> Result<Option<RelationshipEdge>> {
        self.inner.get_relationship(uuid)
    }

    fn put_relationship(&self, edge: &RelationshipEdge) -> Result<()> {
        self.inner.add_relationship(edge.clone()).map(|_| ())
    }

    fn list_relationships_for_entity(
        &self,
        entity_uuid: &uuid::Uuid,
    ) -> Result<Vec<RelationshipEdge>> {
        self.inner.get_entity_relationships(entity_uuid)
    }

    fn get_episode(&self, uuid: &uuid::Uuid) -> Result<Option<EpisodicNode>> {
        self.inner.get_episode(uuid)
    }

    fn put_episode(&self, episode: &EpisodicNode) -> Result<()> {
        self.inner.add_episode(episode.clone()).map(|_| ())
    }

    fn stats(&self) -> Result<GraphStats> {
        self.inner.get_stats()
    }

    fn flush(&self) -> Result<()> {
        self.inner
            .get_db()
            .flush()
            .context("Failed to flush legacy RocksDB graph store")?;
        Ok(())
    }
}

/// Trait adapter for the current RocksDB backup engine.
pub struct RocksDbBackupStore {
    inner: VeldBackupEngine,
    primary_db: Arc<DB>,
}

impl RocksDbBackupStore {
    pub fn new(inner: VeldBackupEngine, primary_db: Arc<DB>) -> Self {
        Self { inner, primary_db }
    }

    pub fn inner(&self) -> &VeldBackupEngine {
        &self.inner
    }
}

impl BackupStore for RocksDbBackupStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn create_backup(&self, user_id: &str) -> Result<BackupMetadata> {
        self.inner.create_backup(&self.primary_db, user_id)
    }

    fn list_backups(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        self.inner.list_backups(user_id)
    }

    fn restore_backup(&self, user_id: &str, backup_id: u32, target_path: &Path) -> Result<()> {
        self.inner.restore_backup(user_id, Some(backup_id), target_path)
    }
}

/// Generic column-family adapter for shared RocksDB-backed stores.
pub struct RocksDbColumnFamilyStore {
    db: Arc<DB>,
    column_family: String,
}

impl RocksDbColumnFamilyStore {
    pub fn new(db: Arc<DB>, column_family: impl Into<String>) -> Self {
        Self {
            db,
            column_family: column_family.into(),
        }
    }

    fn cf(&self) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(&self.column_family)
            .with_context(|| format!("Missing RocksDB column family '{}'", self.column_family))
    }
}

impl KeyValueStore for RocksDbColumnFamilyStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.db.get_cf(self.cf()?, key)?)
    }

    fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put_cf(self.cf()?, key, value)?;
        Ok(())
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        self.db.delete_cf(self.cf()?, key)?;
        Ok(())
    }

    fn scan_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut rows = Vec::new();
        let cf = self.cf()?;

        for item in self
            .db
            .iterator_cf(cf, IteratorMode::From(prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix) {
                break;
            }
            rows.push((key.to_vec(), value.to_vec()));
            if limit.is_some_and(|max| rows.len() >= max) {
                break;
            }
        }

        Ok(rows)
    }

    fn flush(&self) -> Result<()> {
        self.db.flush_cf(self.cf()?)?;
        Ok(())
    }
}

/// Audit adapter for the shared RocksDB-backed audit log column family.
pub struct RocksDbAuditStore {
    db: Arc<DB>,
    column_family: String,
}

impl RocksDbAuditStore {
    pub fn new(db: Arc<DB>, column_family: impl Into<String>) -> Self {
        Self {
            db,
            column_family: column_family.into(),
        }
    }

    fn cf(&self) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(&self.column_family)
            .with_context(|| format!("Missing RocksDB audit column family '{}'", self.column_family))
    }
}

impl AuditStore for RocksDbAuditStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn append_event(&self, user_id: &str, event: &AuditLogEntry) -> Result<()> {
        let timestamp_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
        let key = format!("{user_id}:{timestamp_nanos:020}");
        let value = bincode::serde::encode_to_vec(event, bincode::config::standard())
            .context("Failed to serialize audit log entry for RocksDB audit store")?;

        self.db.put_cf(self.cf()?, key.as_bytes(), value)?;
        Ok(())
    }

    fn list_events(&self, user_id: &str, limit: usize) -> Result<Vec<AuditLogEntry>> {
        let prefix = format!("{user_id}:");
        let mut events = Vec::new();

        for item in self.db.prefix_iterator_cf(self.cf()?, prefix.as_bytes()) {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }

            let (event, _): (AuditLogEntry, _) =
                bincode::serde::decode_from_slice(&value, bincode::config::standard())
                    .context("Failed to deserialize audit log entry from RocksDB audit store")?;
            events.push(event);

            if events.len() >= limit {
                break;
            }
        }

        Ok(events)
    }

    fn rotate_events(
        &self,
        user_id: &str,
        max_entries: usize,
        retention_before: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        let prefix = format!("{user_id}:");
        let cutoff_nanos = retention_before.timestamp_nanos_opt().unwrap_or(0);
        let cf = self.cf()?;

        let mut keys = Vec::new();
        for item in self.db.prefix_iterator_cf(cf, prefix.as_bytes()) {
            let (key, _) = item?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            keys.push(key.to_vec());
        }

        let excess = keys.len().saturating_sub(max_entries);
        let mut batch = WriteBatch::default();
        let mut removed = 0usize;

        for (index, key) in keys.into_iter().enumerate() {
            let key_str = std::str::from_utf8(&key).unwrap_or_default();
            let ts = key_str
                .strip_prefix(&prefix)
                .and_then(|raw| raw.parse::<i64>().ok())
                .unwrap_or(0);

            if ts < cutoff_nanos || index < excess {
                batch.delete_cf(cf, &key);
                removed += 1;
            }
        }

        if removed > 0 {
            self.db.write(batch)?;
        }

        Ok(removed)
    }

    fn flush(&self) -> Result<()> {
        self.db.flush_cf(self.cf()?)?;
        Ok(())
    }
}

/// Factory for RocksDB-backed compatibility stores during the migration.
#[derive(Default)]
pub struct RocksDbStorageFactory {
    shared_cache: Option<rocksdb::Cache>,
}

impl RocksDbStorageFactory {
    pub fn new(shared_cache: Option<rocksdb::Cache>) -> Self {
        Self { shared_cache }
    }

    fn shared_cache(&self) -> Option<&rocksdb::Cache> {
        self.shared_cache.as_ref()
    }
}

impl StorageFactory for RocksDbStorageFactory {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn open_primary_memory_store(
        &self,
        path: &Path,
        _config: &MemoryConfig,
    ) -> Result<Arc<dyn PrimaryMemoryStore>> {
        Ok(Arc::new(RocksDbPrimaryMemoryStore::open(
            path,
            self.shared_cache(),
        )?))
    }

    fn open_graph_store(&self, path: &Path) -> Result<Arc<dyn GraphStore>> {
        Ok(Arc::new(RocksDbGraphStore::open(
            path,
            self.shared_cache(),
        )?))
    }

    fn open_shared_store(&self, _path: &Path, namespace: &str) -> Result<Arc<dyn KeyValueStore>> {
        Err(anyhow!(
            "RocksDB shared store '{}' must be opened from the live shared DB handle during bootstrap",
            namespace
        ))
    }

    fn open_audit_store(&self, _path: &Path) -> Result<Arc<dyn AuditStore>> {
        Err(anyhow!(
            "RocksDB audit store must be opened from the live shared DB handle during bootstrap"
        ))
    }

    fn open_backup_store(&self, path: &Path) -> Result<Arc<dyn BackupStore>> {
        Err(anyhow!(
            "RocksDB backup store requires a bound primary DB reference; use backup engine bootstrap for {:?}",
            path
        ))
    }

    fn open_migration_store(&self, path: &Path) -> Result<Arc<dyn MigrationStore>> {
        Err(anyhow!(
            "RocksDB migration store is not implemented yet for {:?}",
            path
        ))
    }
}