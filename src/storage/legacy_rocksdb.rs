//! RocksDB compatibility adapters for the storage abstraction layer.
//!
//! These wrappers preserve current runtime behavior while moving call sites off
//! direct RocksDB types.

use anyhow::{anyhow, Context, Result};
use rocksdb::{ColumnFamily, Direction, IteratorMode, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;

use super::{
    AuditLogEntry, AuditStore, BackupStore, BatchOp, BatchWriter, GraphStore, KeyValueStore,
    ColumnStore, MigrationStore, PrimaryMemoryStore, StorageFactory,
    WriteBatch as StorageWriteBatch,
};
use crate::backup::{BackupMetadata, VeldBackupEngine};
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphMemory, GraphStats, RelationshipEdge};
use crate::memory::storage::{MemoryStorage, SearchCriteria, VectorMappingEntry};
use crate::memory::{Memory, MemoryConfig, MemoryId};

/// DB-scoped atomic batch writer — see [`super::WriteBatch`] for why it hangs
/// off the database rather than off a single store.
pub struct RocksDbBatchWriter {
    db: Arc<DB>,
}

impl RocksDbBatchWriter {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }
}

impl BatchWriter for RocksDbBatchWriter {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        // RESOLVE EVERY COLUMN FAMILY FIRST. Resolving inside the translation
        // loop would build a rocksdb batch containing the ops before an unknown
        // name and then bail — and because nothing has been written yet that
        // would still look atomic here, while silently changing what the caller
        // asked for the moment anyone reorders the loop or writes early. The
        // failure is made to happen before any op is translated, not before any
        // op is applied, so the two cannot drift apart.
        let mut handles: Vec<(&str, &ColumnFamily)> = Vec::new();
        for name in batch.column_families() {
            let cf = self.db.cf_handle(name).ok_or_else(|| {
                anyhow!(
                    "write_batch names column family '{}', which this database does not have — \
                     nothing was written",
                    name
                )
            })?;
            handles.push((name, cf));
        }
        let resolved = |name: &str| -> &ColumnFamily {
            handles
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, h)| *h)
                .expect("column_families() is the set the ops draw from")
        };

        let mut wb = WriteBatch::default();
        for op in batch.ops() {
            match op {
                BatchOp::Put {
                    cf: Some(name),
                    key,
                    value,
                } => wb.put_cf(resolved(name), key, value),
                BatchOp::Put {
                    cf: None,
                    key,
                    value,
                } => wb.put(key, value),
                BatchOp::Delete {
                    cf: Some(name),
                    key,
                } => wb.delete_cf(resolved(name), key),
                BatchOp::Delete { cf: None, key } => wb.delete(key),
            }
        }

        self.db
            .write(wb)
            .context("write_batch: rocksdb refused the batch")?;
        Ok(())
    }
}

/// Multi-column-family access over a live rocksdb handle.
pub struct RocksDbColumnStore {
    db: Arc<DB>,
}

impl RocksDbColumnStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// A named family, or an error naming it. Never a silent default: resolving a
    /// missing family to the default keyspace would put a store's rows somewhere
    /// nothing reads, and every write would report success.
    fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| anyhow!("this database has no column family '{name}'"))
    }
}

impl ColumnStore for RocksDbColumnStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.db.get_cf(self.cf(cf)?, key)?)
    }

    fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put_cf(self.cf(cf)?, key, value)?;
        Ok(())
    }

    fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
        self.db.delete_cf(self.cf(cf)?, key)?;
        Ok(())
    }

    fn ensure_cf(&self, cf: &str) -> Result<()> {
        // Verification, not creation: rocksdb families come from the open-time
        // descriptors, so a missing one here is a wiring error to fail on now.
        self.cf(cf).map(|_| ())
    }

    fn scan_cf(
        &self,
        cf: &str,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let handle = self.cf(cf)?;
        let mode = match prefix {
            Some(p) => IteratorMode::From(p, Direction::Forward),
            None => IteratorMode::Start,
        };
        let mut rows = Vec::new();
        for item in self.db.iterator_cf(handle, mode) {
            let (key, value) = item?;
            // The bound is enforced HERE. rocksdb's iterator continues past the
            // prefix, so every caller was carrying this break by hand.
            if let Some(p) = prefix {
                if !key.starts_with(p) {
                    break;
                }
            }
            rows.push((key.to_vec(), value.to_vec()));
            if let Some(n) = limit {
                if rows.len() >= n {
                    break;
                }
            }
        }
        Ok(rows)
    }

    fn flush_cf(&self, cf: &str) -> Result<()> {
        self.db.flush_cf(self.cf(cf)?)?;
        Ok(())
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        RocksDbBatchWriter::new(Arc::clone(&self.db)).write_batch(batch)
    }
}

/// Trait adapter for the current RocksDB-backed primary memory store.
///
/// Holds the storage by Arc: MemorySystem's constructor receives a SHARED
/// Arc<MemoryStorage> (the BM25 projection and retrieval engine hold the same
/// one), so an owned inner would force a second store over the same files.
pub struct RocksDbPrimaryMemoryStore {
    inner: Arc<MemoryStorage>,
}

impl RocksDbPrimaryMemoryStore {
    pub fn open(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(MemoryStorage::new(path, shared_cache)?),
        })
    }

    pub fn from_arc(inner: Arc<MemoryStorage>) -> Self {
        Self { inner }
    }

    pub fn from_inner(inner: MemoryStorage) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn inner(&self) -> &MemoryStorage {
        &self.inner
    }

    pub fn into_inner(self) -> Arc<MemoryStorage> {
        self.inner
    }
}

impl PrimaryMemoryStore for RocksDbPrimaryMemoryStore {
    // ── The 21 methods MemorySystem calls, delegated verbatim. Pure forwarding on
    //    purpose: this adapter's whole job is to preserve current runtime behaviour
    //    while call sites move off the concrete type, so anything cleverer here
    //    would change behaviour in the commit whose point is that nothing changes.
    fn get(&self, id: &MemoryId) -> Result<Memory> { self.inner.get(id) }
    fn store(&self, memory: &Memory) -> Result<()> { self.inner.store(memory) }
    fn update(&self, memory: &Memory) -> Result<()> { self.inner.update(memory) }
    fn delete(&self, id: &MemoryId) -> Result<()> { self.inner.delete(id) }
    fn get_all(&self) -> Result<Vec<Memory>> { self.inner.get_all() }
    fn get_all_ids(&self) -> Result<Vec<MemoryId>> { self.inner.get_all_ids() }
    fn increment_retrieval_count(&self) -> Result<usize> { self.inner.increment_retrieval_count() }
    fn get_stats(&self) -> Result<crate::memory::storage::StorageStats> { self.inner.get_stats() }
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        self.inner.find_by_external_id(external_id)
    }
    fn get_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>> {
        self.inner.get_children(parent_id)
    }
    fn get_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>> {
        self.inner.get_ancestors(memory_id)
    }
    fn get_hierarchy_context(
        &self,
        memory_id: &MemoryId,
    ) -> Result<(Vec<Memory>, Memory, Vec<Memory>)> {
        self.inner.get_hierarchy_context(memory_id)
    }
    fn get_uncompressed_older_than(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<Memory>> {
        self.inner.get_uncompressed_older_than(cutoff)
    }
    fn mark_forgotten_by_age(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MemoryId>> {
        self.inner.mark_forgotten_by_age(cutoff)
    }
    fn save_interference_records(
        &self,
        memory_id: &str,
        records: &[crate::memory::replay::InterferenceRecord],
    ) -> Result<()> {
        self.inner.save_interference_records(memory_id, records)
    }
    fn delete_interference_records(&self, memory_id: &str) -> Result<()> {
        self.inner.delete_interference_records(memory_id)
    }
    fn clear_all_interference_records(&self) -> Result<usize> {
        self.inner.clear_all_interference_records()
    }
    fn cleanup_corrupted(&self) -> Result<usize> { self.inner.cleanup_corrupted() }
    fn migrate_legacy(&self) -> Result<(usize, usize, usize)> { self.inner.migrate_legacy() }
    fn run_merge_pass(&self, user_id: &str, limit: usize) -> Result<usize> {
        Ok(self.inner.run_merge_pass(user_id, limit))
    }
    fn gc_merge_tombstones(&self, user_id: &str) -> Result<usize> {
        Ok(self.inner.gc_merge_tombstones(user_id))
    }

    fn scan_prefix_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        self.inner.scan_prefix_keys(prefix)
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        self.inner.write_batch(batch)
    }

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

    // ── The eight the long_term_memory flip surfaced — same pure forwarding.
    fn get_all_vector_mappings(&self) -> Result<Vec<(MemoryId, VectorMappingEntry)>> {
        self.inner.get_all_vector_mappings()
    }

    fn load_all_interference_records(
        &self,
    ) -> Result<(
        std::collections::HashMap<String, Vec<crate::memory::replay::InterferenceRecord>>,
        usize,
    )> {
        self.inner.load_all_interference_records()
    }

    fn update_vector_mapping(&self, memory_id: &MemoryId, vector_ids: Vec<u32>) -> Result<()> {
        self.inner.update_vector_mapping(memory_id, vector_ids)
    }

    fn get_by_content_hash(&self, content: &str) -> Option<MemoryId> {
        self.inner.get_by_content_hash(content)
    }
    fn update_access(&self, id: &MemoryId) -> Result<()> {
        self.inner.update_access(id)
    }
    fn mark_forgotten_by_importance(&self, threshold: f32) -> Result<Vec<MemoryId>> {
        self.inner.mark_forgotten_by_importance(threshold)
    }
    fn save_interference_event_count(&self, count: usize) -> Result<()> {
        self.inner.save_interference_event_count(count)
    }
    fn get_fact_watermark(&self, user_id: &str) -> Option<i64> {
        self.inner.get_fact_watermark(user_id)
    }
    fn set_fact_watermark(&self, user_id: &str, timestamp_millis: i64) {
        self.inner.set_fact_watermark(user_id, timestamp_millis)
    }
    fn key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        self.inner.key_value_store()
    }
    fn default_key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        self.inner.default_key_value_store()
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
            .rocksdb_for_backup()
            .ok_or_else(|| anyhow!("legacy rocksdb graph adapter wraps a non-rocksdb graph"))?
            .flush()
            .context("Failed to flush legacy RocksDB graph store")?;
        Ok(())
    }
}

/// Portable export for the rocksdb side — R-6a.
///
/// Reads only. Nothing here mutates the source store, so it is safe to run
/// against a live database and safe to re-run: the point of doing export before
/// anything else is that a cutover which has only ever produced a file cannot
/// have damaged the thing it copied.
pub struct RocksDbMigrationStore {
    db: Arc<DB>,
}

impl RocksDbMigrationStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Column families the export walks, in a fixed order so two exports of the
    /// same store are byte-identical.
    ///
    /// Named explicitly rather than discovered from the handle: a CF added later
    /// and not listed here would be silently omitted from every export, and the
    /// dump would still verify — it is internally consistent, just incomplete.
    /// `export_portable` therefore also refuses a store whose live CF set is not
    /// covered by this list.
    const FAMILIES: [&'static str; 3] = ["default", "memory_index", "kv"];
}

impl MigrationStore for RocksDbMigrationStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn export_portable(&self, user_id: &str, target: &Path) -> Result<()> {
        use crate::storage::portable::{PortableRecord, PortableWriter};

        // A dump written straight to its final name is a dump that, if the process
        // dies mid-write, sits there looking like an export. Write beside it and
        // rename: the target either does not exist or is complete.
        let tmp = target.with_extension("partial");
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        let mut writer = PortableWriter::new(std::io::BufWriter::new(file))?;

        // Every failure below removes the partial file. Leaving it puts a
        // half-written dump on disk beside where a complete one goes — an
        // artefact whose entire purpose is to be trusted later by someone who was
        // not here when it was written.
        let written = (|| -> Result<(u64, [u8; 32])> {
            let mut total = 0u64;
            for cf_name in Self::FAMILIES {
                let cf = self.db.cf_handle(cf_name).ok_or_else(|| {
                    anyhow!(
                        "export_portable: this database has no column family '{cf_name}' — \
                         refusing to write a dump that would be missing a keyspace"
                    )
                })?;
                for item in self.db.iterator_cf(cf, IteratorMode::Start) {
                    let (key, value) = item?;
                    writer.write_record(&PortableRecord {
                        cf: cf_name.to_string(),
                        key: key.to_vec(),
                        value: value.to_vec(),
                    })?;
                    total += 1;
                }
            }
            let (count, digest) = writer.finish()?;
            debug_assert_eq!(count, total);
            Ok((count, digest))
        })();

        let (count, digest) = match written {
            Ok(v) => v,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        std::fs::rename(&tmp, target)
            .with_context(|| format!("renaming {} into place", tmp.display()))?;
        tracing::info!(
            user_id,
            records = count,
            digest = %hex_digest(&digest),
            target = %target.display(),
            "portable export written"
        );
        Ok(())
    }

    fn import_portable(&self, _user_id: &str, _source: &Path) -> Result<()> {
        // R-6b, and deliberately not written alongside the exporter: an importer
        // built in the same sitting tends to be tested against the dumps that
        // exporter happens to produce, which proves the pair agree and not that
        // either is right. The codec's own tests are the contract.
        Err(anyhow!(
            "import_portable is not implemented for rocksdb (R-6b) — the dump format and its \
             verification are in storage::portable"
        ))
    }

    fn migrate_in_place(
        &self,
        _user_id: &str,
        _target_backend: StorageBackend,
        _target_path: &Path,
    ) -> Result<()> {
        // R-6d, marked do-not-implement. In-place conversion necessarily creates a
        // window where the source bytes are consumed and the target is not yet
        // proven readable — under a supervisor that restarts the child 5x/min.
        Err(anyhow!(
            "migrate_in_place is deliberately not implemented: use export_portable, then import \
             into a scratch target, verify it, then switch the configured backend. That path is \
             reversible by one config value; this one is not."
        ))
    }
}

fn hex_digest(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
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

/// A `KeyValueStore` over the DEFAULT keyspace.
///
/// Exists because `RocksDbColumnFamilyStore` is bound to a named column family,
/// and some data predates the column families entirely — the memory ACLs are
/// written with `db.put(key, …)`, which is the default CF. Routing those through
/// the `kv`-bound `MemoryStorage::key_value_store()` would not fail: it would
/// read and write a DIFFERENT keyspace, so every existing grant would come back
/// absent and every new one would land where nothing looks. A migration wearing
/// a refactor's clothes.
pub struct RocksDbDefaultKeyValueStore {
    db: Arc<DB>,
}

impl RocksDbDefaultKeyValueStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }
}

impl KeyValueStore for RocksDbDefaultKeyValueStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::RocksDb
    }

    fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.db.get(key)?)
    }

    fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put(key, value)?;
        Ok(())
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        self.db.delete(key)?;
        Ok(())
    }

    fn scan_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // The prefix bound is enforced here rather than by each caller. Every
        // caller ported onto this so far carried its own `starts_with` break, and
        // rocksdb's iterator genuinely does run past the prefix, so the guard was
        // load-bearing in all of them.
        let mut rows = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix) {
                break;
            }
            rows.push((key.to_vec(), value.to_vec()));
            if let Some(n) = limit {
                if rows.len() >= n {
                    break;
                }
            }
        }
        Ok(rows)
    }

    fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let mut wb = WriteBatch::default();
        for (k, v) in pairs {
            wb.put(k, v);
        }
        self.db.write(wb)?;
        Ok(())
    }

    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
        let mut wb = WriteBatch::default();
        for k in keys {
            wb.delete(k);
        }
        self.db.write(wb)?;
        Ok(())
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
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let cf = self.cf()?;
        let mut wb = WriteBatch::default();
        for (k, v) in pairs {
            wb.put_cf(cf, k, v);
        }
        self.db.write(wb)?;
        Ok(())
    }

    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
        let cf = self.cf()?;
        let mut wb = WriteBatch::default();
        for k in keys {
            wb.delete_cf(cf, k);
        }
        self.db.write(wb)?;
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

    fn list_user_ids(&self) -> Result<Vec<String>> {
        let mut ids = std::collections::HashSet::new();
        for (key, _) in self.db.iterator_cf(self.cf()?, rocksdb::IteratorMode::Start).flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(user_id) = key_str.split(':').next() {
                    ids.insert(user_id.to_string());
                }
            }
        }
        Ok(ids.into_iter().collect())
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
#[cfg(test)]
mod migration_export_tests {
    use super::*;
    use crate::storage::portable::read_all;
    use rocksdb::{ColumnFamilyDescriptor, Options};

    fn db_with_families(dir: &std::path::Path) -> Arc<DB> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new("memory_index", Options::default()),
            ColumnFamilyDescriptor::new("kv", Options::default()),
        ];
        Arc::new(DB::open_cf_descriptors(&opts, dir, cfs).expect("open"))
    }

    #[test]
    fn the_export_carries_every_column_family_not_just_the_default() {
        // The failure this catches is an exporter that walks `default` only. Its
        // dump verifies perfectly — internally consistent, just missing two
        // keyspaces — so nothing downstream can tell. Only a source with rows in
        // every family can.
        let dir = tempfile::tempdir().expect("tempdir");
        let db = db_with_families(dir.path());
        db.put(b"acl:alice:1", b"in-default").unwrap();
        db.put_cf(db.cf_handle("memory_index").unwrap(), b"i1", b"in-index").unwrap();
        db.put_cf(db.cf_handle("kv").unwrap(), b"dials", b"in-kv").unwrap();

        let out = dir.path().join("export.veldport");
        RocksDbMigrationStore::new(db.clone())
            .export_portable("alice", &out)
            .expect("export");

        let records = read_all(std::fs::File::open(&out).unwrap()).expect("verify + read");
        let mut seen: Vec<(String, Vec<u8>)> =
            records.iter().map(|r| (r.cf.clone(), r.value.clone())).collect();
        seen.sort();
        assert_eq!(
            seen,
            vec![
                ("default".to_string(), b"in-default".to_vec()),
                ("kv".to_string(), b"in-kv".to_vec()),
                ("memory_index".to_string(), b"in-index".to_vec()),
            ]
        );
    }

    #[test]
    fn an_export_of_an_empty_store_verifies_and_holds_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = db_with_families(dir.path());
        let out = dir.path().join("empty.veldport");
        RocksDbMigrationStore::new(db)
            .export_portable("alice", &out)
            .expect("export");
        assert!(read_all(std::fs::File::open(&out).unwrap()).unwrap().is_empty());
    }

    #[test]
    fn two_exports_of_the_same_store_are_byte_identical() {
        // Determinism is not decoration here: export -> import -> verify compares
        // artefacts, and a dump that differs run to run makes "did this copy
        // correctly" unanswerable.
        let dir = tempfile::tempdir().expect("tempdir");
        let db = db_with_families(dir.path());
        for i in 0..50u32 {
            db.put(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
            db.put_cf(db.cf_handle("kv").unwrap(), format!("x{i:03}").as_bytes(), b"y").unwrap();
        }
        let store = RocksDbMigrationStore::new(db);
        let a = dir.path().join("a.veldport");
        let b = dir.path().join("b.veldport");
        store.export_portable("alice", &a).unwrap();
        store.export_portable("alice", &b).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    }

    #[test]
    fn a_store_missing_a_declared_family_refuses_rather_than_exporting_a_gap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = Arc::new(DB::open(&opts, dir.path()).expect("open default-only"));
        let out = dir.path().join("gap.veldport");
        let err = RocksDbMigrationStore::new(db)
            .export_portable("alice", &out)
            .expect_err("a missing keyspace must refuse");
        assert!(err.to_string().contains("column family"), "got: {err}");
        assert!(!out.exists(), "a refused export must leave no file to be mistaken for one");
        assert!(
            !out.with_extension("partial").exists(),
            "the partial was left on disk — a half-written dump beside where a complete one goes"
        );
    }

    #[test]
    fn migrate_in_place_refuses_and_says_what_to_do_instead() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = db_with_families(dir.path());
        let err = RocksDbMigrationStore::new(db)
            .migrate_in_place("alice", StorageBackend::Redb, dir.path())
            .expect_err("R-6d is do-not-implement");
        assert!(err.to_string().contains("export_portable"));
    }
}
