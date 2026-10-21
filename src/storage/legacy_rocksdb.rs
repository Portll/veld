//! RocksDB compatibility adapters for the storage abstraction layer.
//!
//! These wrappers preserve current runtime behavior while moving call sites off
//! direct RocksDB types.

use anyhow::{anyhow, Context, Result};
use rocksdb::{ColumnFamily, Direction, IteratorMode, WriteBatch, WriteOptions, DB};
use std::path::Path;
use std::sync::Arc;

use super::{
    AuditLogEntry, AuditStore, BackupStore, BatchOp, BatchWriter, ColumnStore, GraphStore,
    KeyValueStore, MigrationStore, PrimaryMemoryStore, StorageFactory,
    WriteBatch as StorageWriteBatch,
};
use crate::backup::{BackupMetadata, VeldBackupEngine};
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphMemory, GraphStats, RelationshipEdge};
use crate::memory::storage::{MemoryStorage, SearchCriteria, VectorMappingEntry};
use crate::memory::{Memory, MemoryConfig, MemoryId};

/// Options for one write: `sync` syncs the write-ahead log before the write returns,
/// which is what `StorageDurability::Immediate` means on RocksDB.
pub fn write_options(sync: bool) -> WriteOptions {
    let mut options = WriteOptions::default();
    options.set_sync(sync);
    options
}

/// The process's choice, for adapters built over a handle that carries none.
fn process_sync_writes() -> bool {
    crate::config::StorageTuning::process().rocksdb_sync_writes()
}

/// DB-scoped atomic batch writer — see [`super::WriteBatch`] for why it hangs
/// off the database rather than off a single store.
pub struct RocksDbBatchWriter {
    db: Arc<DB>,
    sync_writes: bool,
}

impl RocksDbBatchWriter {
    pub fn new(db: Arc<DB>) -> Self {
        Self::with_sync_writes(db, process_sync_writes())
    }

    pub fn with_sync_writes(db: Arc<DB>, sync_writes: bool) -> Self {
        Self { db, sync_writes }
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

        // Rebuildable, because `DB::write` consumes the batch and the write
        // latch's recovery path needs to replay the same ops.
        let build = || {
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
            wb
        };

        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || {
            self.db
                .write_opt(build(), &opts)
                .context("write_batch: rocksdb refused the batch")
        })
    }
}

/// Multi-column-family access over a live rocksdb handle.
pub struct RocksDbColumnStore {
    db: Arc<DB>,
    sync_writes: bool,
}

impl RocksDbColumnStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self::with_sync_writes(db, process_sync_writes())
    }

    pub fn with_sync_writes(db: Arc<DB>, sync_writes: bool) -> Self {
        Self { db, sync_writes }
    }

    /// A named family for a write, or an error naming it. Never a silent default:
    /// resolving a missing family to the default keyspace would put a store's rows
    /// somewhere nothing reads, and every write would report success. Reads do not
    /// come here: a family the database does not have holds no rows (see `ColumnStore`).
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
        match self.db.cf_handle(cf) {
            Some(handle) => Ok(self.db.get_cf(handle, key)?),
            None => Ok(None),
        }
    }

    fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let handle = self.cf(cf)?;
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || {
            Ok(self.db.put_cf_opt(handle, key, value, &opts)?)
        })
    }

    fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
        let handle = self.cf(cf)?;
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || {
            Ok(self.db.delete_cf_opt(handle, key, &opts)?)
        })
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
        let Some(handle) = self.db.cf_handle(cf) else {
            return Ok(Vec::new());
        };
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
        let handle = self.cf(cf)?;
        super::write_latch::guard(&self.db, || Ok(self.db.flush_cf(handle)?))
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        RocksDbBatchWriter::with_sync_writes(Arc::clone(&self.db), self.sync_writes)
            .write_batch(batch)
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

/// Whether a user's rocksdb primary store holds at least one memory. Opened
/// read-only, because a normal open creates the store it is asked about.
/// `None` when no store exists.
pub fn user_has_memories(user_dir: &Path) -> Result<Option<bool>> {
    let path = user_dir.join("storage");
    match std::fs::metadata(path.join("CURRENT")) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("probe rocksdb store {}", path.display())),
    }
    let db = DB::open_for_read_only(&rocksdb::Options::default(), &path, false)
        .with_context(|| format!("probe rocksdb store {}", path.display()))?;
    for item in db.iterator(IteratorMode::Start) {
        let (key, _) = item?;
        // Memories are the 16-byte uuid keys; the default family holds other keys too.
        if key.len() == 16 {
            return Ok(Some(true));
        }
    }
    Ok(Some(false))
}

/// Visits every default-family record of a rocksdb store opened read-only with all of its
/// column families, for offline diagnostics over a copied store. Nothing is written.
pub fn visit_records_read_only(path: &Path, mut visit: impl FnMut(&[u8], &[u8])) -> Result<()> {
    let opts = rocksdb::Options::default();
    let cfs = DB::list_cf(&opts, path)
        .with_context(|| format!("list column families {}", path.display()))?;
    let db = DB::open_cf_for_read_only(&opts, path, &cfs, false)
        .with_context(|| format!("open read-only {}", path.display()))?;
    for item in db.iterator(IteratorMode::Start) {
        let (key, value) = item?;
        visit(&key, &value);
    }
    Ok(())
}

impl RocksDbPrimaryMemoryStore {
    pub fn open(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        Self::open_with_tuning(path, shared_cache, crate::config::StorageTuning::process())
    }

    pub fn open_with_tuning(
        path: &Path,
        shared_cache: Option<&rocksdb::Cache>,
        tuning: crate::config::StorageTuning,
    ) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(MemoryStorage::new_with_tuning(path, shared_cache, tuning)?),
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
    fn get(&self, id: &MemoryId) -> Result<Memory> {
        self.inner.get(id)
    }
    fn store(&self, memory: &Memory) -> Result<()> {
        self.inner.store(memory)
    }
    fn update(&self, memory: &Memory) -> Result<()> {
        self.inner.update(memory)
    }
    fn update_many(&self, memories: &[Memory]) -> Result<()> {
        self.inner.store_many(memories)
    }
    fn delete(&self, id: &MemoryId) -> Result<()> {
        self.inner.delete(id)
    }
    fn get_all(&self) -> Result<Vec<Memory>> {
        self.inner.get_all()
    }
    fn get_all_ids(&self) -> Result<Vec<MemoryId>> {
        self.inner.get_all_ids()
    }
    fn increment_retrieval_count(&self) -> Result<usize> {
        self.inner.increment_retrieval_count()
    }
    fn get_stats(&self) -> Result<crate::memory::storage::StorageStats> {
        self.inner.get_stats()
    }
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
    fn cleanup_corrupted(&self) -> Result<usize> {
        self.inner.cleanup_corrupted()
    }
    fn migrate_legacy(&self) -> Result<(usize, usize, usize)> {
        self.inner.migrate_legacy()
    }
    fn run_merge_pass(&self, user_id: &str, limit: usize) -> Result<usize> {
        crate::memory::merge::run_merge_pass(self, user_id, limit)
    }
    fn gc_merge_tombstones(&self, user_id: &str) -> Result<usize> {
        crate::memory::merge::gc_merge_tombstones(self, user_id)
    }
    fn has_any_inbound_refs(&self, target: &MemoryId, user_id: &str) -> Result<bool> {
        self.inner.has_any_inbound_refs(target, user_id)
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
        self.inner.put_vector_mapping(id, mapping)
    }

    fn store_with_vectors(&self, memory: &Memory, vector_ids: Vec<u32>) -> Result<()> {
        self.inner.store_with_vectors(memory, vector_ids)
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
        self.inner.put_entity_record(entity)
    }

    fn get_relationship(&self, uuid: &uuid::Uuid) -> Result<Option<RelationshipEdge>> {
        self.inner.get_relationship(uuid)
    }

    fn put_relationship(&self, edge: &RelationshipEdge) -> Result<()> {
        self.inner.put_relationship_record(edge)
    }

    fn list_relationships_for_entity(
        &self,
        entity_uuid: &uuid::Uuid,
    ) -> Result<Vec<RelationshipEdge>> {
        self.inner.stored_entity_relationships(entity_uuid)
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
        let db = self
            .inner
            .rocksdb_for_backup()
            .ok_or_else(|| anyhow!("legacy rocksdb graph adapter wraps a non-rocksdb graph"))?;
        super::write_latch::guard(db, || {
            db.flush()
                .context("Failed to flush legacy RocksDB graph store")
        })
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
        let file =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
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
        self.inner
            .restore_backup(user_id, Some(backup_id), target_path)
    }
}

/// One-time import of a pre-column-family standalone DB into a shared-DB family; the
/// old directory is renamed to `<name>.pre_cf_migration` so a later boot finds nothing.
pub fn import_separate_db_into_cf(db: &DB, old_dir: &Path, cf_name: &str) -> Result<()> {
    if !old_dir.is_dir() {
        return Ok(());
    }
    let cf = db
        .cf_handle(cf_name)
        .ok_or_else(|| anyhow!("this database has no column family '{cf_name}'"))?;
    let old_db = match DB::open_for_read_only(&rocksdb::Options::default(), old_dir, false) {
        Ok(old_db) => old_db,
        Err(e) => {
            tracing::warn!(
                "Could not open old DB {} for migration: {e}",
                old_dir.display()
            );
            return Ok(());
        }
    };
    let write_opts = write_options(process_sync_writes());
    let mut batch = WriteBatch::default();
    let mut count = 0usize;
    for item in old_db.iterator(IteratorMode::Start) {
        let (key, value) = item.map_err(|e| {
            anyhow!(
                "RocksDB iterator error while importing {}: {e}",
                old_dir.display()
            )
        })?;
        batch.put_cf(cf, &key, &value);
        count += 1;
        if count.is_multiple_of(10_000) {
            super::write_latch::guard_batch(db, &batch, &write_opts)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        super::write_latch::guard_batch(db, &batch, &write_opts)?;
    }
    drop(old_db);
    tracing::info!(
        "  {}: migrated {count} entries to {cf_name} CF",
        old_dir.display()
    );

    let Some(name) = old_dir.file_name() else {
        return Ok(());
    };
    let backup = old_dir.with_file_name(format!("{}.pre_cf_migration", name.to_string_lossy()));
    if backup.exists() {
        let _ = std::fs::remove_dir_all(&backup);
    }
    if let Err(e) = std::fs::rename(old_dir, &backup) {
        tracing::warn!("Could not rename old dir {}: {e}", old_dir.display());
    }
    Ok(())
}

/// A standalone pre-column-family rocksdb directory holding `rows`, for import tests.
#[cfg(test)]
pub(crate) fn create_separate_db_for_test(path: &Path, rows: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
    let db = DB::open_default(path)?;
    for (key, value) in rows {
        db.put(key, value)?;
    }
    Ok(())
}

/// Generic column-family adapter for shared RocksDB-backed stores.
pub struct RocksDbColumnFamilyStore {
    db: Arc<DB>,
    column_family: String,
    sync_writes: bool,
}

impl RocksDbColumnFamilyStore {
    pub fn new(db: Arc<DB>, column_family: impl Into<String>) -> Self {
        Self::with_sync_writes(db, column_family, process_sync_writes())
    }

    pub fn with_sync_writes(
        db: Arc<DB>,
        column_family: impl Into<String>,
        sync_writes: bool,
    ) -> Self {
        Self {
            db,
            column_family: column_family.into(),
            sync_writes,
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
    sync_writes: bool,
}

impl RocksDbDefaultKeyValueStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self::with_sync_writes(db, process_sync_writes())
    }

    pub fn with_sync_writes(db: Arc<DB>, sync_writes: bool) -> Self {
        Self { db, sync_writes }
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
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || Ok(self.db.put_opt(key, value, &opts)?))
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || Ok(self.db.delete_opt(key, &opts)?))
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
        super::write_latch::guard(&self.db, || Ok(self.db.flush()?))
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let mut wb = WriteBatch::default();
        for (k, v) in pairs {
            wb.put(k, v);
        }
        super::write_latch::guard_batch(&self.db, &wb, &write_options(self.sync_writes))
    }

    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
        let mut wb = WriteBatch::default();
        for k in keys {
            wb.delete(k);
        }
        super::write_latch::guard_batch(&self.db, &wb, &write_options(self.sync_writes))
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
        let cf = self.cf()?;
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || {
            Ok(self.db.put_cf_opt(cf, key, value, &opts)?)
        })
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        let cf = self.cf()?;
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || Ok(self.db.delete_cf_opt(cf, key, &opts)?))
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
        let cf = self.cf()?;
        super::write_latch::guard(&self.db, || Ok(self.db.flush_cf(cf)?))
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let cf = self.cf()?;
        let mut wb = WriteBatch::default();
        for (k, v) in pairs {
            wb.put_cf(cf, k, v);
        }
        super::write_latch::guard_batch(&self.db, &wb, &write_options(self.sync_writes))
    }

    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
        let cf = self.cf()?;
        let mut wb = WriteBatch::default();
        for k in keys {
            wb.delete_cf(cf, k);
        }
        super::write_latch::guard_batch(&self.db, &wb, &write_options(self.sync_writes))
    }
}

/// Audit adapter for the shared RocksDB-backed audit log column family.
pub struct RocksDbAuditStore {
    db: Arc<DB>,
    column_family: String,
    sync_writes: bool,
}

impl RocksDbAuditStore {
    pub fn new(db: Arc<DB>, column_family: impl Into<String>) -> Self {
        Self {
            db,
            column_family: column_family.into(),
            sync_writes: process_sync_writes(),
        }
    }

    fn cf(&self) -> Result<&ColumnFamily> {
        self.db.cf_handle(&self.column_family).with_context(|| {
            format!(
                "Missing RocksDB audit column family '{}'",
                self.column_family
            )
        })
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

        let cf = self.cf()?;
        let opts = write_options(self.sync_writes);
        super::write_latch::guard(&self.db, || {
            Ok(self.db.put_cf_opt(cf, key.as_bytes(), &value, &opts)?)
        })
    }

    fn list_user_ids(&self) -> Result<Vec<String>> {
        let mut ids = std::collections::HashSet::new();
        for (key, _) in self
            .db
            .iterator_cf(self.cf()?, rocksdb::IteratorMode::Start)
            .flatten()
        {
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
            super::write_latch::guard_batch(&self.db, &batch, &write_options(self.sync_writes))?;
        }

        Ok(removed)
    }

    fn flush(&self) -> Result<()> {
        let cf = self.cf()?;
        super::write_latch::guard(&self.db, || Ok(self.db.flush_cf(cf)?))
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

    /// A read error part-way through the old DB must stop the import before the old directory
    /// is renamed aside; skipping the unreadable rows would retire the only copy of them.
    #[test]
    fn a_legacy_db_that_fails_to_read_is_not_renamed_away() {
        let dir = tempfile::tempdir().expect("tempdir");
        let old_dir = dir.path().join("legacy_kv");
        {
            let old = DB::open_default(&old_dir).expect("open legacy");
            for i in 0..500u32 {
                old.put(format!("key-{i:05}"), vec![i as u8; 128]).unwrap();
            }
            old.flush().unwrap();
        }
        for entry in std::fs::read_dir(&old_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "log") {
                // The WAL would replay every row into the memtable and hide the damaged table.
                std::fs::remove_file(&path).unwrap();
            } else if path.extension().is_some_and(|e| e == "sst") {
                let mut bytes = std::fs::read(&path).unwrap();
                for b in &mut bytes[64..256] {
                    *b ^= 0xA5;
                }
                std::fs::write(&path, bytes).unwrap();
            }
        }

        let target_dir = dir.path().join("target");
        let db = db_with_families(&target_dir);
        let result = import_separate_db_into_cf(&db, &old_dir, "kv");

        assert!(result.is_err(), "a damaged data block must fail the import");
        assert!(
            old_dir.is_dir(),
            "the old DB was renamed away after a failed read"
        );
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
        db.put_cf(db.cf_handle("memory_index").unwrap(), b"i1", b"in-index")
            .unwrap();
        db.put_cf(db.cf_handle("kv").unwrap(), b"dials", b"in-kv")
            .unwrap();

        let out = dir.path().join("export.veldport");
        RocksDbMigrationStore::new(db.clone())
            .export_portable("alice", &out)
            .expect("export");

        let records = read_all(std::fs::File::open(&out).unwrap()).expect("verify + read");
        let mut seen: Vec<(String, Vec<u8>)> = records
            .iter()
            .map(|r| (r.cf.clone(), r.value.clone()))
            .collect();
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
        assert!(read_all(std::fs::File::open(&out).unwrap())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn two_exports_of_the_same_store_are_byte_identical() {
        // Determinism is not decoration here: export -> import -> verify compares
        // artefacts, and a dump that differs run to run makes "did this copy
        // correctly" unanswerable.
        let dir = tempfile::tempdir().expect("tempdir");
        let db = db_with_families(dir.path());
        for i in 0..50u32 {
            db.put(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
            db.put_cf(
                db.cf_handle("kv").unwrap(),
                format!("x{i:03}").as_bytes(),
                b"y",
            )
            .unwrap();
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
        assert!(
            !out.exists(),
            "a refused export must leave no file to be mistaken for one"
        );
        assert!(
            !out.with_extension("partial").exists(),
            "the partial was left on disk — a half-written dump beside where a complete one goes"
        );
    }

    #[test]
    fn writes_sync_only_under_immediate_durability() {
        use crate::config::{StorageDurability, StorageTuning};
        let immediate = StorageTuning {
            durability: StorageDurability::Immediate,
            ..StorageTuning::default()
        };
        assert!(immediate.rocksdb_sync_writes());
        assert!(!StorageTuning::default().rocksdb_sync_writes());

        let dir = tempfile::tempdir().expect("tempdir");
        let db = db_with_families(dir.path());
        let mut batch = StorageWriteBatch::new();
        batch.put_cf("kv", b"synced".to_vec(), b"1".to_vec());
        RocksDbBatchWriter::with_sync_writes(Arc::clone(&db), true)
            .write_batch(&batch)
            .expect("a synced batch");
        let kv = RocksDbColumnFamilyStore::with_sync_writes(Arc::clone(&db), "kv", true);
        kv.put_bytes(b"also-synced", b"2").expect("a synced put");
        assert_eq!(kv.get_bytes(b"synced").unwrap().as_deref(), Some(&b"1"[..]));
        assert_eq!(
            kv.get_bytes(b"also-synced").unwrap().as_deref(),
            Some(&b"2"[..])
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

/// Behaviour both engines must share, asserted on each through the storage traits.
#[cfg(all(test, feature = "storage-redb"))]
mod engine_parity_tests {
    use super::*;
    use crate::backup::VeldBackupEngine;
    use crate::memory::context::ContextBuilder;
    use crate::memory::types::{EpisodeContext, Experience, ExperienceType, MemoryTier};
    use crate::storage::redb::{RedbBackupStore, RedbPrimaryMemoryStore};

    const MERGE_USER: &str = "parity-user";

    fn memory(content: &str, importance: f32) -> Memory {
        let now = chrono::Utc::now();
        Memory::from_legacy(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                experience_type: ExperienceType::Observation,
                content: content.to_string(),
                ..Default::default()
            },
            importance,
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

    /// A child with a row in every keyed index: external id, episode and sequence,
    /// tags, entities, and a parent that its inbound-reference row also names.
    fn fully_indexed_child(parent: &MemoryId, external_id: &str, content: &str) -> Memory {
        let mut child = memory(content, 0.5);
        child.experience.tags = vec!["Parity".to_string()];
        child.experience.entities = vec!["Tauranga".to_string()];
        child.experience.context = Some(
            ContextBuilder::new()
                .with_episode(EpisodeContext {
                    episode_id: Some(format!("episode-{external_id}")),
                    sequence_number: Some(2),
                    ..Default::default()
                })
                .build(),
        );
        child.parent_id = Some(parent.clone());
        child.external_id = Some(external_id.to_string());
        child
    }

    fn engines() -> Vec<(&'static str, Box<dyn PrimaryMemoryStore>, tempfile::TempDir)> {
        let rocks_dir = tempfile::tempdir().expect("tempdir");
        let rocks = RocksDbPrimaryMemoryStore::open(rocks_dir.path(), None).expect("rocksdb");
        let redb_dir = tempfile::tempdir().expect("tempdir");
        let redb = RedbPrimaryMemoryStore::open(redb_dir.path()).expect("redb");
        vec![
            ("rocksdb", Box::new(rocks), rocks_dir),
            ("redb", Box::new(redb), redb_dir),
        ]
    }

    fn ids(memories: Result<Vec<Memory>>) -> Vec<MemoryId> {
        let mut ids: Vec<MemoryId> = memories
            .expect("lookup")
            .into_iter()
            .map(|m| m.id)
            .collect();
        ids.sort();
        ids
    }

    /// What every lookup answers about a deleted child of `parent`.
    #[derive(Debug, PartialEq)]
    struct AfterDelete {
        get_memory: Option<MemoryId>,
        external_id: Option<MemoryId>,
        children: Vec<MemoryId>,
        by_parent: Vec<MemoryId>,
        by_tag: Vec<MemoryId>,
        by_entity: Vec<MemoryId>,
        by_episode: Vec<MemoryId>,
        by_episode_sequence: Vec<MemoryId>,
        content_hash: Option<MemoryId>,
        parent_still_cited: bool,
        vector_mapping: bool,
    }

    impl AfterDelete {
        const NOTHING: AfterDelete = AfterDelete {
            get_memory: None,
            external_id: None,
            children: Vec::new(),
            by_parent: Vec::new(),
            by_tag: Vec::new(),
            by_entity: Vec::new(),
            by_episode: Vec::new(),
            by_episode_sequence: Vec::new(),
            content_hash: None,
            parent_still_cited: false,
            vector_mapping: false,
        };

        fn observe(
            store: &dyn PrimaryMemoryStore,
            parent: &MemoryId,
            child: &Memory,
            external_id: &str,
        ) -> Self {
            let episode = format!("episode-{external_id}");
            Self {
                get_memory: store
                    .get_memory(&child.id)
                    .expect("get_memory")
                    .map(|m| m.id),
                external_id: store
                    .find_by_external_id(external_id)
                    .expect("find_by_external_id")
                    .map(|m| m.id),
                children: ids(store.get_children(parent)),
                by_parent: ids(store.search(SearchCriteria::ByParent(parent.clone()))),
                by_tag: ids(store.search(SearchCriteria::ByTags(vec!["parity".to_string()]))),
                by_entity: ids(store.search(SearchCriteria::ByEntity("tauranga".to_string()))),
                by_episode: ids(store.search(SearchCriteria::ByEpisode(episode.clone()))),
                by_episode_sequence: ids(store.search(SearchCriteria::ByEpisodeSequence {
                    episode_id: episode,
                    min_sequence: None,
                    max_sequence: None,
                })),
                content_hash: store.get_by_content_hash(&child.experience.content),
                parent_still_cited: store
                    .has_any_inbound_refs(parent, MERGE_USER)
                    .expect("has_any_inbound_refs"),
                vector_mapping: store
                    .get_vector_mapping(&child.id)
                    .expect("get_vector_mapping")
                    .is_some(),
            }
        }
    }

    #[test]
    fn memories_sharing_content_answer_the_lowest_live_id_on_both_engines() {
        let shared = "the same words, remembered three times";
        let with_id = |first_byte: u8| {
            let mut stored = memory(shared, 0.5);
            stored.id = MemoryId(uuid::Uuid::from_bytes([first_byte; 16]));
            stored
        };
        let (low, middle, high) = (with_id(0x10), with_id(0x80), with_id(0xF0));
        for (engine, store, _dir) in engines() {
            for stored in [&high, &low, &middle] {
                store.put_memory(stored).unwrap();
            }
            assert_eq!(
                store.get_by_content_hash(shared),
                Some(low.id.clone()),
                "{engine}: the lowest id answers, not the last stored"
            );

            store.delete_memory(&low.id).unwrap();
            assert_eq!(
                store.get_by_content_hash(shared),
                Some(middle.id.clone()),
                "{engine}: deleting one holder leaves the others findable"
            );

            let mut moved_on = middle.clone();
            moved_on.experience.content = "other words now".to_string();
            store.update_memory(&moved_on).unwrap();
            assert_eq!(
                store.get_by_content_hash(shared),
                Some(high.id.clone()),
                "{engine}: a rewritten holder stops answering for its old content"
            );
            assert_eq!(
                store.get_by_content_hash("other words now"),
                Some(middle.id.clone()),
                "{engine}"
            );

            store.delete_memory(&high.id).unwrap();
            assert_eq!(store.get_by_content_hash(shared), None, "{engine}");
        }
    }

    #[test]
    fn a_deleted_memory_answers_no_lookup_on_either_engine() {
        for (engine, store, _dir) in engines() {
            let parent = memory("the parent", 0.5);
            store.put_memory(&parent).unwrap();

            let by_delete_memory =
                fully_indexed_child(&parent.id, "linear:VELD-1", "removed by delete_memory");
            let by_delete = fully_indexed_child(&parent.id, "linear:VELD-2", "removed by delete");
            for child in [&by_delete_memory, &by_delete] {
                store.store_with_vectors(child, vec![5]).unwrap();
            }
            assert_eq!(
                AfterDelete::observe(
                    store.as_ref(),
                    &parent.id,
                    &by_delete_memory,
                    "linear:VELD-1"
                )
                .external_id,
                Some(by_delete_memory.id.clone()),
                "{engine}: the fixture is not findable before the delete"
            );

            store.delete_memory(&by_delete_memory.id).unwrap();
            store.delete(&by_delete.id).unwrap();

            for (child, external_id) in [
                (&by_delete_memory, "linear:VELD-1"),
                (&by_delete, "linear:VELD-2"),
            ] {
                assert_eq!(
                    AfterDelete::observe(store.as_ref(), &parent.id, child, external_id),
                    AfterDelete::NOTHING,
                    "{engine}: a lookup still answers for {external_id} after its delete"
                );
            }
        }
    }

    #[test]
    fn get_all_leaves_out_forgotten_memories_on_both_engines() {
        for (engine, store, _dir) in engines() {
            let kept = [memory("kept one", 0.9), memory("kept two", 0.9)];
            let forgotten = memory("forgotten", 0.1);
            for m in kept.iter().chain([&forgotten]) {
                store.put_memory(m).unwrap();
            }
            assert_eq!(
                store.mark_forgotten_by_importance(0.5).unwrap(),
                vec![forgotten.id.clone()],
                "{engine}"
            );

            let mut live: Vec<MemoryId> = kept.iter().map(|m| m.id.clone()).collect();
            live.sort();
            assert_eq!(ids(store.get_all()), live, "{engine}: get_all");

            let mut every = live.clone();
            every.push(forgotten.id.clone());
            every.sort();
            let mut all_ids = store.get_all_ids().unwrap();
            all_ids.sort();
            assert_eq!(
                all_ids, every,
                "{engine}: get_all_ids keeps forgotten records"
            );
            assert!(
                store.get(&forgotten.id).unwrap().is_forgotten(),
                "{engine}: a forgotten record is still readable by id"
            );
        }
    }

    #[test]
    fn backup_stores_list_newest_first_on_both_engines() {
        let rocks_dir = tempfile::tempdir().expect("tempdir");
        let rocks = RocksDbPrimaryMemoryStore::open(&rocks_dir.path().join("store"), None).unwrap();
        let rocks_backups = RocksDbBackupStore::new(
            VeldBackupEngine::new(rocks_dir.path().join("backups")).unwrap(),
            rocks.inner().db(),
        );
        let redb_dir = tempfile::tempdir().expect("tempdir");
        let redb = RedbPrimaryMemoryStore::open(redb_dir.path()).unwrap();
        let redb_backups = RedbBackupStore::from_database(redb.database(), redb_dir.path());

        let stores: [(&str, &dyn PrimaryMemoryStore, &dyn BackupStore); 2] = [
            ("rocksdb", &rocks, &rocks_backups),
            ("redb", &redb, &redb_backups),
        ];
        for (engine, primary, backups) in stores {
            let mut created = Vec::new();
            for round in 0..3 {
                primary
                    .put_memory(&memory(&format!("round {round}"), 0.5))
                    .unwrap();
                created.push(backups.create_backup(MERGE_USER).unwrap().backup_id);
            }
            created.reverse();
            let listed: Vec<u32> = backups
                .list_backups(MERGE_USER)
                .unwrap()
                .iter()
                .map(|metadata| metadata.backup_id)
                .collect();
            assert_eq!(
                listed, created,
                "{engine}: backups are not listed newest first"
            );
        }
    }

    #[test]
    fn an_empty_combined_search_finds_nothing_on_both_engines() {
        for (engine, store, _dir) in engines() {
            store.put_memory(&memory("a live memory", 0.5)).unwrap();
            assert!(
                store
                    .search(SearchCriteria::Combined(Vec::new()))
                    .unwrap()
                    .is_empty(),
                "{engine}: Combined([]) answered with memories"
            );
            assert!(
                store
                    .search(SearchCriteria::Combined(vec![
                        SearchCriteria::Combined(Vec::new()),
                        SearchCriteria::RootsOnly,
                    ]))
                    .unwrap()
                    .is_empty(),
                "{engine}: an empty Combined nested in another matched"
            );
        }
    }

    #[test]
    fn range_searches_answer_their_exact_bounds_on_both_engines() {
        let day = chrono::DateTime::parse_from_rfc3339("2026-03-04T00:00:00Z")
            .expect("a valid instant")
            .with_timezone(&chrono::Utc);
        let placed = |content: &str, importance: f32, hour: i64, reward: f32, at: (f64, f64)| {
            let mut placed = memory(content, importance);
            placed.created_at = day + chrono::Duration::hours(hour);
            placed.experience.reward = Some(reward);
            placed.experience.geo_location = Some([at.0, at.1, 0.0]);
            placed
        };
        let london = (51.5074, -0.1278);
        let low = placed("on the bucket edge", 0.30, 1, -0.30, london);
        let inside = placed("inside every bound", 0.34, 13, -0.26, (51.5074, 0.5900));
        let high = placed("past every bound", 0.39, 23, -0.21, (51.5074, 0.6400));
        let north = placed("49 km north", 0.10, 30, 0.9, (51.9474, -0.1278));

        for (engine, store, _dir) in engines() {
            for m in [&low, &inside, &high, &north] {
                store.put_memory(m).unwrap();
            }
            let found = |criteria: SearchCriteria| ids(store.search(criteria));
            assert_eq!(
                found(SearchCriteria::ByImportance {
                    min: 0.31,
                    max: 0.35
                }),
                vec![inside.id.clone()],
                "{engine}: importance answered from its tenth bucket"
            );
            assert_eq!(
                found(SearchCriteria::ByDate {
                    start: day + chrono::Duration::hours(12),
                    end: day + chrono::Duration::hours(14),
                }),
                vec![inside.id.clone()],
                "{engine}: a mid-day bound answered with the whole day"
            );
            assert_eq!(
                found(SearchCriteria::ByReward {
                    min: -0.28,
                    max: -0.24
                }),
                vec![inside.id.clone()],
                "{engine}: reward answered from its twentieth bucket"
            );
            let mut near_london = vec![low.id.clone(), inside.id.clone(), north.id.clone()];
            near_london.sort();
            assert_eq!(
                found(SearchCriteria::ByLocation {
                    lat: london.0,
                    lon: london.1,
                    radius_meters: 51_000.0,
                }),
                near_london,
                "{engine}: a radius search missed or overshot the circle"
            );
        }
    }

    #[test]
    fn update_many_leaves_what_successive_updates_leave_on_both_engines() {
        for (engine, store, _dir) in engines() {
            let mut first = memory("the ferry timetable changes in spring", 0.5);
            first.experience.tags = vec!["ferries".to_string()];
            let mut second = memory("the harbour crane is serviced in May", 0.5);
            second.experience.tags = vec!["cranes".to_string()];
            store.put_memory(&first).unwrap();
            store.put_memory(&second).unwrap();
            store.update_many(&[]).unwrap();

            let mut first_now = first.clone();
            first_now.experience.tags = vec!["timetables".to_string()];
            let mut second_now = second.clone();
            second_now.experience.content = "the harbour crane is serviced in June".to_string();
            let third = memory("the tide gauge reads high on Thursdays", 0.5);
            store
                .update_many(&[first_now.clone(), second_now.clone(), third.clone()])
                .unwrap();

            assert_eq!(
                store.get(&second.id).unwrap().experience.content,
                second_now.experience.content,
                "{engine}"
            );
            assert!(
                store.get(&third.id).is_ok(),
                "{engine}: a new id was not stored"
            );
            assert!(
                ids(store.search(SearchCriteria::ByTags(vec!["ferries".to_string()]))).is_empty(),
                "{engine}: the replaced tag still answers"
            );
            assert_eq!(
                ids(store.search(SearchCriteria::ByTags(vec!["timetables".to_string()]))),
                vec![first.id.clone()],
                "{engine}"
            );
            assert_eq!(
                ids(store.search(SearchCriteria::ByTags(vec!["cranes".to_string()]))),
                vec![second.id.clone()],
                "{engine}"
            );
        }
    }

    #[test]
    fn scan_prefix_keys_answers_memory_keys_only_on_both_engines() {
        for (engine, store, _dir) in engines() {
            let stored = memory("a memory to find by key", 0.5);
            store.put_memory(&stored).unwrap();
            // Never 16 bytes: RocksDB shares its default keyspace with memories and
            // takes every unprefixed 16-byte key there for a memory id.
            store
                .default_key_value_store()
                .put_bytes(b"dkv:not-a-memory-row", b"1")
                .unwrap();
            store
                .put_vector_mapping(&stored.id, &VectorMappingEntry::default())
                .unwrap();

            assert_eq!(
                store.scan_prefix_keys(b"").unwrap(),
                vec![stored.id.0.as_bytes().to_vec()],
                "{engine}: a scan of every key reached past the memories"
            );
            assert!(
                store.scan_prefix_keys(b"dkv:").unwrap().is_empty(),
                "{engine}: a default-keyspace row answered as a memory key"
            );
            assert_eq!(
                store
                    .scan_prefix_keys(&stored.id.0.as_bytes()[..1])
                    .unwrap(),
                vec![stored.id.0.as_bytes().to_vec()],
                "{engine}"
            );
        }
    }

    #[test]
    fn a_rewrite_over_a_live_id_leaves_no_lookup_for_the_old_version_on_both_engines() {
        type Write = fn(&dyn PrimaryMemoryStore, &Memory) -> Result<()>;
        let writes: [(&str, Write); 4] = [
            ("store", |store, memory| store.store(memory)),
            ("put_memory", |store, memory| store.put_memory(memory)),
            ("update", |store, memory| store.update(memory)),
            ("store_with_vectors", |store, memory| {
                store.store_with_vectors(memory, vec![4])
            }),
        ];
        for (path, write) in writes {
            for (engine, store, _dir) in engines() {
                let parent = memory("the old parent", 0.5);
                store.put_memory(&parent).unwrap();
                let original =
                    fully_indexed_child(&parent.id, "linear:VELD-9", "the original version");
                store.put_memory(&original).unwrap();

                let mut rewritten = memory("the rewritten version", 0.5);
                rewritten.id = original.id.clone();
                write(store.as_ref(), &rewritten).unwrap();

                let stale = AfterDelete {
                    get_memory: store.get_memory(&original.id).unwrap().map(|m| m.id),
                    vector_mapping: false,
                    ..AfterDelete::observe(store.as_ref(), &parent.id, &original, "linear:VELD-9")
                };
                let expected = AfterDelete {
                    get_memory: Some(original.id.clone()),
                    ..AfterDelete::NOTHING
                };
                assert_eq!(
                    stale, expected,
                    "{engine} {path}: a lookup still answers for the replaced version"
                );
                assert_eq!(
                    store.get_by_content_hash("the rewritten version"),
                    Some(original.id.clone()),
                    "{engine} {path}"
                );
            }
        }
    }

    #[test]
    fn a_batch_naming_kv_applies_to_a_fresh_store_on_both_engines() {
        for (engine, store, _dir) in engines() {
            let mut batch = StorageWriteBatch::new();
            batch.put_cf("kv", b"dials".to_vec(), b"1".to_vec());
            store
                .write_batch(&batch)
                .unwrap_or_else(|e| panic!("{engine}: {e}"));
            assert_eq!(
                store.key_value_store().get_bytes(b"dials").unwrap(),
                Some(b"1".to_vec()),
                "{engine}"
            );
        }
    }

    /// A record under a memory key that no decoder accepts.
    const UNDECODABLE: &[u8] = b"not a memory";

    #[test]
    fn an_undecodable_record_is_skipped_counted_and_alone_cleaned_up_on_both_engines() {
        let rocks_dir = tempfile::tempdir().expect("tempdir");
        let rocks = RocksDbPrimaryMemoryStore::open(rocks_dir.path(), None).unwrap();
        let redb_dir = tempfile::tempdir().expect("tempdir");
        let redb = RedbPrimaryMemoryStore::open(redb_dir.path()).unwrap();
        let corrupt = MemoryId(uuid::Uuid::new_v4());
        rocks
            .inner()
            .db()
            .put(corrupt.0.as_bytes(), UNDECODABLE)
            .unwrap();
        {
            let txn = redb.database().begin_write().unwrap();
            txn.open_table(::redb::TableDefinition::<&[u8], &[u8]>::new("memories"))
                .unwrap()
                .insert(corrupt.0.as_bytes().as_slice(), UNDECODABLE)
                .unwrap();
            txn.commit().unwrap();
        }

        let skipped = |engine: &str| match engine {
            "rocksdb" => rocks.inner().undecodable_records_skipped(),
            _ => redb.undecodable_records_skipped(),
        };
        let stores: [(&str, &dyn PrimaryMemoryStore); 2] = [("rocksdb", &rocks), ("redb", &redb)];
        for (engine, store) in stores {
            let live = memory("a decodable neighbour", 0.5);
            store.put_memory(&live).unwrap();
            let kv = store.default_key_value_store();
            kv.put_bytes(b"dkv:a", b"kept").unwrap();
            kv.put_bytes(b"merge_audit:parity-user:1", b"kept").unwrap();
            kv.put_bytes(b"merge_tombstone:parity-user:1", b"kept")
                .unwrap();

            assert_eq!(
                ids(store.get_all()),
                vec![live.id.clone()],
                "{engine}: get_all did not skip the undecodable record"
            );
            assert!(
                skipped(engine) >= 1,
                "{engine}: the skipped record was not counted"
            );
            assert_eq!(
                store.cleanup_corrupted().unwrap(),
                1,
                "{engine}: cleanup removed something other than the undecodable record"
            );
            assert_eq!(
                store.get_all_ids().unwrap(),
                vec![live.id.clone()],
                "{engine}"
            );
            for key in [
                &b"dkv:a"[..],
                b"merge_audit:parity-user:1",
                b"merge_tombstone:parity-user:1",
            ] {
                assert_eq!(
                    kv.get_bytes(key).unwrap(),
                    Some(b"kept".to_vec()),
                    "{engine}: cleanup deleted a default-keyspace row"
                );
            }
        }
    }

    #[test]
    fn an_undeclared_family_reads_empty_on_both_engines() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let rocks_db = Arc::new(DB::open(&opts, tmp.path().join("rocks")).unwrap());
        let redb_db =
            crate::storage::redb::create_redb_database_for_test(&tmp.path().join("t.redb"))
                .unwrap();
        let stores: [(&str, Arc<dyn ColumnStore>); 2] = [
            ("rocksdb", Arc::new(RocksDbColumnStore::new(rocks_db))),
            (
                "redb",
                Arc::new(crate::storage::redb::RedbColumnStore::new(redb_db)),
            ),
        ];
        for (engine, store) in stores {
            assert_eq!(store.get_cf("undeclared", b"k").unwrap(), None, "{engine}");
            assert!(
                store.scan_cf("undeclared", None, None).unwrap().is_empty(),
                "{engine}"
            );
            assert!(
                store
                    .scan_cf("undeclared", Some(b"k"), Some(1))
                    .unwrap()
                    .is_empty(),
                "{engine}"
            );
        }
    }

    fn graph_engines() -> Vec<(&'static str, Box<dyn GraphStore>, tempfile::TempDir)> {
        let rocks_dir = tempfile::tempdir().expect("tempdir");
        let rocks = RocksDbGraphStore::open(rocks_dir.path(), None).expect("rocksdb graph");
        let redb_dir = tempfile::tempdir().expect("tempdir");
        let redb = crate::storage::redb::RedbGraphStore::open(redb_dir.path()).expect("redb graph");
        vec![
            ("rocksdb", Box::new(rocks), rocks_dir),
            ("redb", Box::new(redb), redb_dir),
        ]
    }

    fn entity(name: &str) -> EntityNode {
        let now = chrono::Utc::now();
        EntityNode {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            labels: Vec::new(),
            created_at: now,
            last_seen_at: now,
            mention_count: 3,
            summary: "given by the caller".to_string(),
            attributes: Default::default(),
            name_embedding: None,
            salience: 0.4,
            is_proper_noun: true,
            pii_classification: Default::default(),
        }
    }

    fn edge(from: &EntityNode, to: &EntityNode) -> RelationshipEdge {
        let now = chrono::Utc::now();
        RelationshipEdge {
            uuid: uuid::Uuid::new_v4(),
            from_entity: from.uuid,
            to_entity: to.uuid,
            relation_type: crate::graph_memory::RelationType::WorksWith,
            strength: 0.3,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: None,
            context: "given by the caller".to_string(),
            last_activated: now,
            activation_count: 2,
            ltp_status: Default::default(),
            tier: Default::default(),
            activation_timestamps: None,
            entity_confidence: None,
            created_by: Default::default(),
            forward_strength: 0.0,
            backward_strength: 0.0,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    fn episode(refs: Vec<uuid::Uuid>) -> EpisodicNode {
        let now = chrono::Utc::now();
        EpisodicNode {
            uuid: uuid::Uuid::new_v4(),
            name: "an episode".to_string(),
            content: "the harbour survey".to_string(),
            valid_at: now,
            created_at: now,
            entity_refs: refs,
            source: crate::graph_memory::EpisodeSource::Observation,
            metadata: Default::default(),
        }
    }

    fn json<T: serde::Serialize>(value: &T) -> serde_json::Value {
        serde_json::to_value(value).expect("graph records serialize")
    }

    #[test]
    fn a_graph_record_reads_back_by_the_uuid_it_was_given_on_both_engines() {
        for (engine, store, _dir) in graph_engines() {
            let harbour = entity("harbour");
            let lantern = entity("lantern");
            for node in [&harbour, &lantern] {
                store.put_entity(node).unwrap();
            }
            let link = edge(&harbour, &lantern);
            store.put_relationship(&link).unwrap();

            assert_eq!(
                store.get_entity(&harbour.uuid).unwrap().as_ref().map(json),
                Some(json(&harbour)),
                "{engine}: the entity is not stored as given under its uuid"
            );
            assert_eq!(
                store
                    .get_relationship(&link.uuid)
                    .unwrap()
                    .as_ref()
                    .map(json),
                Some(json(&link)),
                "{engine}: the edge is not stored as given under its uuid"
            );
            let listed: Vec<serde_json::Value> = store
                .list_relationships_for_entity(&lantern.uuid)
                .unwrap()
                .iter()
                .map(json)
                .collect();
            assert_eq!(listed, vec![json(&link)], "{engine}");

            let mut rewritten = link.clone();
            rewritten.strength = 0.9;
            store.put_relationship(&rewritten).unwrap();
            assert_eq!(
                store
                    .get_relationship(&link.uuid)
                    .unwrap()
                    .as_ref()
                    .map(json),
                Some(json(&rewritten)),
                "{engine}: a rewrite is stored as given, not merged"
            );
            let stats = store.stats().unwrap();
            assert_eq!(
                (stats.entity_count, stats.relationship_count),
                (2, 1),
                "{engine}: a rewrite was counted as a new record"
            );
        }
    }

    #[test]
    fn rewriting_an_episode_does_not_count_it_twice_on_both_engines() {
        for (engine, store, _dir) in graph_engines() {
            let first = episode(Vec::new());
            store.put_episode(&first).unwrap();
            let mut rewritten = first.clone();
            rewritten.content = "the harbour survey, revised".to_string();
            store.put_episode(&rewritten).unwrap();
            store.put_episode(&episode(Vec::new())).unwrap();
            assert_eq!(store.stats().unwrap().episode_count, 2, "{engine}");
            assert_eq!(
                store.get_episode(&first.uuid).unwrap().as_ref().map(json),
                Some(json(&rewritten)),
                "{engine}"
            );
        }
    }

    #[test]
    fn graph_memory_counts_episodes_not_writes_over_either_column_store() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let graphs = [
            (
                "rocksdb",
                GraphMemory::new(&tmp.path().join("rocks"), None).unwrap(),
            ),
            (
                "redb",
                GraphMemory::with_store(Arc::new(crate::storage::redb::RedbColumnStore::new(
                    crate::storage::redb::create_redb_database_for_test(
                        &tmp.path().join("graph.redb"),
                    )
                    .unwrap(),
                )))
                .unwrap(),
            ),
        ];
        for (engine, graph) in graphs {
            let anchor = graph.add_entity(entity("anchor")).unwrap();
            let buoy = graph.add_entity(entity("buoy")).unwrap();
            let mut written = episode(vec![anchor, buoy]);
            graph.add_episode(written.clone()).unwrap();
            written.entity_refs = vec![buoy];
            graph.add_episode(written.clone()).unwrap();

            assert_eq!(graph.get_stats().unwrap().episode_count, 1, "{engine}");
            assert!(
                graph
                    .get_episodes_by_entity(&anchor)
                    .unwrap()
                    .iter()
                    .all(|found| found.uuid != written.uuid),
                "{engine}: the entity index still names an episode that dropped it"
            );
        }
    }
}
