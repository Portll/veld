//! redb implementations for the storage abstraction layer.
//!
//! The primary memory store, graph store, audit store, and key-value store are
//! operational: each persists `bincode(serde)`-encoded records in redb tables,
//! mirroring the codec the legacy RocksDB path uses (so unencrypted records are
//! byte-identical across both backends). Backup and migration remain
//! backend-cutover work — they need per-user snapshot semantics and a
//! cross-backend transfer path — and still return a clear not-implemented error.
//!
//! These trait impls are the forward integration seam. The live runtime still
//! constructs the concrete RocksDB-coupled engines (`MemoryStorage`,
//! `GraphMemory`, and the shared stores) directly; routing the whole runtime
//! through these traits — with entity-dedup/index parity and at-rest encryption
//! parity — is the remaining v0.9 cutover.

use anyhow::{anyhow, bail, Context, Result};
use redb::{ReadableTable, ReadableTableMetadata, TableHandle};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{
    AuditLogEntry, AuditStore, BackupStore, BatchOp, BatchWriter, GraphStore, KeyValueStore,
    ColumnStore, MigrationStore, PrimaryMemoryStore, StorageFactory,
    WriteBatch as StorageWriteBatch,
};
use crate::backup::BackupMetadata;
use crate::config::StorageBackend;
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, RelationshipEdge};
use crate::memory::storage::{SearchCriteria, VectorMappingEntry};
use crate::memory::{Memory, MemoryConfig, MemoryId};

/// The table an op with `cf: None` addresses.
///
/// **"default", not "kv", and the difference is a correctness bug this once had.**
/// It was defined as "kv" so that `KV_TABLE` could be derived from it, on the
/// reasoning that a default-keyspace batch op and a `KeyValueStore` put should
/// reach the same rows. That is true of `KeyValueStore`, which is bound to the
/// `kv` family — and false of a batch, because on rocksdb `cf: None` becomes
/// `db.put(...)`, which is the DEFAULT column family, a different keyspace from
/// `kv` entirely. The same `WriteBatch` therefore landed in two different places
/// depending on which backend applied it, which is precisely the divergence
/// `RedbBatchWriter::write_batch` carries a comment about refusing.
///
/// The exporter's family list is the arbiter: rocksdb has `default`,
/// `memory_index` and `kv`, and they are three distinct keyspaces.
pub(crate) const DEFAULT_TABLE_NAME: &str = "default";

/// DB-scoped atomic batch writer. See [`super::WriteBatch`] for why the handle
/// is the database rather than one store.
pub struct RedbBatchWriter {
    database: Arc<redb::Database>,
}

impl RedbBatchWriter {
    pub fn new(database: Arc<redb::Database>) -> Self {
        Self { database }
    }
}

impl BatchWriter for RedbBatchWriter {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        // EXISTENCE IS CHECKED, NOT ASSUMED, and the reason is a real
        // divergence between the backends rather than caution. RocksDB column
        // families are created when the database is opened, so naming an
        // unknown one is an error. redb tables are created LAZILY on first
        // write — `init_table` above exists solely to paper over that — so the
        // same batch would silently CREATE the table here and succeed. One
        // backend rejecting what the other quietly invents is precisely the
        // divergence that makes a shared trait a lie, so a read transaction
        // resolves every name first and an unknown one fails before the write
        // transaction is ever opened.
        {
            let read = self.database.begin_read()?;
            for name in batch.column_families() {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                    redb::TableDefinition::new(name);
                read.open_table(def).map_err(|e| {
                    anyhow!(
                        "write_batch names table '{}', which this database does not have \
                         ({e}) — nothing was written",
                        name
                    )
                })?;
            }
        }

        let txn = self.database.begin_write()?;
        {
            for op in batch.ops() {
                match op {
                    BatchOp::Put { cf, key, value } => {
                        let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                            redb::TableDefinition::new(cf.as_deref().unwrap_or(DEFAULT_TABLE_NAME));
                        let mut handle = txn.open_table(def)?;
                        handle.insert(key.as_slice(), value.as_slice())?;
                    }
                    BatchOp::Delete { cf, key } => {
                        let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                            redb::TableDefinition::new(cf.as_deref().unwrap_or(DEFAULT_TABLE_NAME));
                        let mut handle = txn.open_table(def)?;
                        // Deleting an absent key is not an error: rocksdb's
                        // delete_cf is equally indifferent, and a batch that
                        // failed on a missing row would diverge from it.
                        let _ = handle.remove(key.as_slice())?;
                    }
                }
            }
        }
        // Atomic here: one redb write transaction. Dropping it without commit
        // — which is what `?` above does on any failure — applies nothing.
        txn.commit()?;
        Ok(())
    }
}

/// Multi-column-family access over a live redb handle. A rocksdb column family is
/// a redb table, one for one — the same mapping the portable dump already uses.
pub struct RedbColumnStore {
    database: Arc<redb::Database>,
}

impl RedbColumnStore {
    pub fn new(database: Arc<redb::Database>) -> Self {
        Self { database }
    }
}

impl ColumnStore for RedbColumnStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let txn = self.database.begin_read()?;
        let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
        // A table that does not exist holds no rows — None, not an error. Reading is
        // the one direction where absence is a legitimate answer; every WRITE below
        // names its family and a redb write transaction creates it.
        match txn.open_table(def) {
            Ok(t) => Ok(t.get(key)?.map(|v| v.value().to_vec())),
            Err(_) => Ok(None),
        }
    }

    fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
            let mut table = txn.open_table(def)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn ensure_cf(&self, cf: &str) -> Result<()> {
        // Materialization, not verification: redb tables are lazy, and a family
        // whose first touch is a read or a batched write would otherwise hit
        // TableDoesNotExist. Same move RedbKeyValueStore::open makes for `kv`.
        let txn = self.database.begin_write()?;
        {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
            txn.open_table(def)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
            let mut table = txn.open_table(def)?;
            let _ = table.remove(key)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn scan_cf(
        &self,
        cf: &str,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let txn = self.database.begin_read()?;
        let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
        let table = match txn.open_table(def) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };
        let mut rows = Vec::new();
        // redb ranges are ordered, so a prefix scan stops at the first key that
        // leaves it rather than reading the table.
        let iter: Box<dyn Iterator<Item = _>> = match prefix {
            Some(p) => Box::new(table.range(p..)?),
            None => Box::new(table.iter()?),
        };
        for row in iter {
            let (k, v) = row?;
            let key = k.value().to_vec();
            if let Some(p) = prefix {
                if !key.starts_with(p) {
                    break;
                }
            }
            rows.push((key, v.value().to_vec()));
            if let Some(n) = limit {
                if rows.len() >= n {
                    break;
                }
            }
        }
        Ok(rows)
    }

    fn flush_cf(&self, _cf: &str) -> Result<()> {
        // redb commits are durable, so there is nothing buffered to flush. Ok(())
        // is the honest answer rather than a no-op standing in for one.
        Ok(())
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        RedbBatchWriter::new(Arc::clone(&self.database)).write_batch(batch)
    }
}

/// An operation the redb backend declares but has not built. Distinct from
/// `not_implemented`, which names a path; these are store-level and the store
/// holds only a database handle.
fn unbuilt(operation: &str) -> anyhow::Error {
    anyhow!("redb {operation} is declared on PrimaryMemoryStore but not implemented — refusing rather than returning an empty result")
}

fn not_implemented(operation: &str, path: &Path) -> anyhow::Error {
    anyhow!(
        "redb {} is not implemented yet for {}",
        operation,
        path.display()
    )
}

/// Test-only constructor so conformance tests can build a bare handle.
#[cfg(test)]
pub fn create_redb_database_for_test(path: &Path) -> Result<Arc<redb::Database>> {
    create_redb_database(path)
}

/// Open (or create) the SHARED redb database holding the global stores'
/// tables — the redb counterpart of `open_legacy_shared_db`. One file, one
/// opener: every store over it must be built FROM THIS HANDLE (`with_store`
/// constructors), never by path, or the second opener dies on the store lock.
pub fn open_shared_database(base_path: &Path) -> Result<Arc<redb::Database>> {
    create_redb_database(&base_path.join("shared.redb"))
}

/// Open (or create) a per-user GRAPH database — the redb counterpart of the
/// graph's own rocksdb directory. Separate file from the shared stores, matching
/// the rocksdb layout where the graph is its own database.
pub fn open_graph_database(graph_path: &Path) -> Result<Arc<redb::Database>> {
    create_redb_database(&graph_path.join("graph.redb"))
}

fn create_redb_database(path: &Path) -> Result<Arc<redb::Database>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(Arc::new(redb::Database::create(path)?))
}

// =============================================================================
// Shared redb helpers
//
// Every domain store below persists `bincode(serde)`-encoded records in a single
// `<&[u8], &[u8]>` table. Keys are the record's natural identity bytes (the UUID
// for memories/graph nodes; `{user_id}:{timestamp_nanos:020}` for audit). Reads
// are point lookups or ordered range scans; redb commits are durable, so every
// `flush` is a no-op.
// =============================================================================

/// The default keyspace. rocksdb writes several small records with bare `db.put`
/// — interference records, the retrieval counter — and those keys must land in the
/// SAME logical space on both backends or a portable dump round-trips them into a
/// table nothing reads. Named from DEFAULT_TABLE_NAME so it cannot drift from what
/// a `cf: None` batch op addresses.
const DEFAULT_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new(DEFAULT_TABLE_NAME);

/// Memories keyed by `MemoryId` bytes -> `bincode(Memory)`.
const MEMORIES_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("memories");
/// Vector-index mappings keyed by `MemoryId` bytes -> `bincode(VectorMappingEntry)`.
const VMAPPINGS_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("vector_mappings");
/// Graph entities keyed by UUID bytes -> `bincode(EntityNode)`.
const ENTITIES_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("entities");
/// Graph relationships keyed by UUID bytes -> `bincode(RelationshipEdge)`.
const RELATIONSHIPS_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("relationships");
/// Graph episodes keyed by UUID bytes -> `bincode(EpisodicNode)`.
const EPISODES_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("episodes");
/// Audit log keyed by `{user_id}:{timestamp_nanos:020}` -> `bincode(AuditLogEntry)`.
const AUDIT_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("audit");

/// Encode any serde value with the standard bincode config — the same codec the
/// legacy RocksDB store uses for unencrypted records.
fn encode<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| anyhow!("redb bincode encode failed: {e}"))
}

/// Decode a serde value previously written by [`encode`].
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map(|(value, _)| value)
        .map_err(|e| anyhow!("redb bincode decode failed: {e}"))
}

/// Materialize a table so later read transactions never hit `TableDoesNotExist`
/// on a freshly created database.
fn init_table(
    database: &redb::Database,
    table: redb::TableDefinition<'static, &[u8], &[u8]>,
) -> Result<()> {
    let txn = database.begin_write()?;
    txn.open_table(table)?;
    txn.commit()?;
    Ok(())
}

/// Point-write `key -> value` in `table` (one durable transaction).
fn table_put(
    database: &redb::Database,
    table: redb::TableDefinition<'static, &[u8], &[u8]>,
    key: &[u8],
    value: &[u8],
) -> Result<()> {
    let txn = database.begin_write()?;
    {
        let mut handle = txn.open_table(table)?;
        handle.insert(key, value)?;
    }
    txn.commit()?;
    Ok(())
}

/// Point-read `key` from `table`, returning the owned value bytes if present.
fn table_get(
    database: &redb::Database,
    table: redb::TableDefinition<'static, &[u8], &[u8]>,
    key: &[u8],
) -> Result<Option<Vec<u8>>> {
    let txn = database.begin_read()?;
    let handle = txn.open_table(table)?;
    Ok(handle.get(key)?.map(|value| value.value().to_vec()))
}

/// Remove `key` from `table`; returns whether a row existed.
fn table_remove(
    database: &redb::Database,
    table: redb::TableDefinition<'static, &[u8], &[u8]>,
    key: &[u8],
) -> Result<bool> {
    let txn = database.begin_write()?;
    let existed;
    {
        let mut handle = txn.open_table(table)?;
        existed = handle.remove(key)?.is_some();
    }
    txn.commit()?;
    Ok(existed)
}

/// Episode id attached to a memory, if any.
fn memory_episode_id(memory: &Memory) -> Option<&str> {
    memory
        .experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.episode_id.as_deref())
}

/// Sequence number of a memory within its episode, if any.
fn memory_sequence(memory: &Memory) -> Option<u32> {
    memory
        .experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.sequence_number)
}

/// Great-circle distance in metres (haversine).
fn haversine_metres(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;
    let (phi1, phi2) = (lat1.to_radians(), lat2.to_radians());
    let delta_phi = (lat2 - lat1).to_radians();
    let delta_lambda = (lon2 - lon1).to_radians();
    let a = (delta_phi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (delta_lambda / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().asin()
}

/// Whether a memory satisfies a [`SearchCriteria`].
///
/// This is the scan-time equivalent of the legacy RocksDB secondary indexes:
/// the redb primary store is the single source of truth, so structured search
/// scans memories and applies this predicate rather than maintaining parallel
/// index tables that could drift. `get_memory` (the hot path) stays an O(1)
/// point lookup; index acceleration for very large stores is a post-cutover
/// follow-up.
fn search_matches(memory: &Memory, criteria: &SearchCriteria) -> bool {
    match criteria {
        SearchCriteria::ByDate { start, end } => {
            memory.created_at >= *start && memory.created_at <= *end
        }
        SearchCriteria::ByType(experience_type) => {
            &memory.experience.experience_type == experience_type
        }
        SearchCriteria::ByImportance { min, max } => {
            let importance = memory.importance();
            importance >= *min && importance <= *max
        }
        SearchCriteria::ByEntity(entity) => {
            let needle = entity.to_lowercase();
            memory
                .experience
                .entities
                .iter()
                .any(|candidate| candidate.to_lowercase() == needle)
        }
        SearchCriteria::ByTags(tags) => {
            let owned: HashSet<String> = memory
                .experience
                .tags
                .iter()
                .map(|tag| tag.to_lowercase())
                .collect();
            tags.iter().any(|tag| owned.contains(&tag.to_lowercase()))
        }
        SearchCriteria::ByEpisode(episode_id) => {
            memory_episode_id(memory) == Some(episode_id.as_str())
        }
        SearchCriteria::ByEpisodeSequence {
            episode_id,
            min_sequence,
            max_sequence,
        } => {
            if memory_episode_id(memory) != Some(episode_id.as_str()) {
                return false;
            }
            match memory_sequence(memory) {
                Some(sequence) => {
                    min_sequence.is_none_or(|lo| sequence >= lo)
                        && max_sequence.is_none_or(|hi| sequence <= hi)
                }
                None => false,
            }
        }
        SearchCriteria::ByRobot(robot_id) => {
            memory.experience.resolved_robot_id() == Some(robot_id.as_str())
        }
        SearchCriteria::ByMission(mission_id) => {
            memory.experience.resolved_mission_id() == Some(mission_id.as_str())
        }
        SearchCriteria::ByLocation {
            lat,
            lon,
            radius_meters,
        } => match memory.experience.resolved_geo() {
            Some((mlat, mlon, _alt)) => haversine_metres(*lat, *lon, mlat, mlon) <= *radius_meters,
            None => false,
        },
        SearchCriteria::ByActionType(action_type) => {
            memory.experience.action_type.as_deref() == Some(action_type.as_str())
        }
        SearchCriteria::ByReward { min, max } => match memory.experience.reward {
            Some(reward) => reward >= *min && reward <= *max,
            None => false,
        },
        SearchCriteria::Combined(criterias) => criterias
            .iter()
            .all(|criteria| search_matches(memory, criteria)),
        SearchCriteria::ByParent(parent_id) => memory.parent_id.as_ref() == Some(parent_id),
        SearchCriteria::RootsOnly => memory.parent_id.is_none(),
    }
}

/// Memories in a user's redb primary store, opened without creating anything.
/// `None` when no store file exists.
pub fn user_memory_count(user_dir: &Path) -> Result<Option<u64>> {
    let path = user_dir.join("storage.redb");
    match std::fs::metadata(&path) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("probe redb store {}", path.display())),
    }
    let database = redb::Database::open(&path)
        .with_context(|| format!("probe redb store {}", path.display()))?;
    let txn = database.begin_read()?;
    match txn.open_table(MEMORIES_TABLE) {
        Ok(table) => Ok(Some(table.len()?)),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(Some(0)),
        Err(e) => Err(e.into()),
    }
}

pub struct RedbPrimaryMemoryStore {
    database: Arc<redb::Database>,
}

impl RedbPrimaryMemoryStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("storage.redb");
        let database = create_redb_database(&database_path)?;
        init_table(&database, MEMORIES_TABLE)?;
        init_table(&database, VMAPPINGS_TABLE)?;
        init_table(&database, DEFAULT_TABLE)?;
        Ok(Self { database })
    }

    /// The live database handle, for stores that must share it rather than open the
    /// path again — see RedbBackupStore. redb permits one opener per file.
    pub fn database(&self) -> Arc<redb::Database> {
        Arc::clone(&self.database)
    }

    /// Decode every stored memory. Backs structured `search`: the memories table
    /// is the single source of truth, scanned and filtered by [`search_matches`].
    fn scan_all_memories(&self) -> Result<Vec<Memory>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        let mut memories = Vec::new();
        for entry in table.iter()? {
            let (_key, value) = entry?;
            memories.push(decode::<Memory>(value.value())?);
        }
        Ok(memories)
    }
}

impl PrimaryMemoryStore for RedbPrimaryMemoryStore {
    // ── Declared but not built, and each one SAYS so rather than returning an
    //    empty success. redb is unreachable today (config::effective_storage_backend_
    //    for_current_build returns RocksDb from both arms), so nothing here can be hit
    //    in production; the value of the stubs is that a future implementer sees the
    //    list, and that wiring redb before filling them fails loudly instead of
    //    reporting a store with no memories in it.
    // These four are NOT unbuilt: redb's primary store is operational, and each of
    // them is the same operation this trait already carries under another name. On
    // redb `get_memory` is a plain table lookup returning Option, so composing `get`
    // from it is exact — unlike the rocksdb adapter, where `get_memory` recovers
    // not-found by matching an error STRING and composing the other way round would
    // launder real failures into `Ok(None)`. Same two names, opposite safe direction,
    // per backend.
    fn get(&self, id: &MemoryId) -> Result<Memory> {
        self.get_memory(id)?
            .ok_or_else(|| anyhow!("Memory not found: {}", id.0))
    }
    fn store(&self, memory: &Memory) -> Result<()> { self.put_memory(memory) }
    fn update(&self, memory: &Memory) -> Result<()> { self.update_memory(memory) }
    fn delete(&self, id: &MemoryId) -> Result<()> { self.delete_memory(id) }
    fn get_all(&self) -> Result<Vec<Memory>> {
        self.scan_all_memories()
    }
    fn get_all_ids(&self) -> Result<Vec<MemoryId>> {
        // rocksdb filters its default-CF scan to 16-byte keys, because that keyspace
        // also holds non-memory rows. The redb memories table holds only memories, so
        // the filter is not needed to be CORRECT here — it is kept because a stray
        // key of another length would otherwise become a garbage UUID rather than an
        // omission, and a wrong id is worse than a missing one.
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        let mut ids = Vec::new();
        for row in table.iter()? {
            let (k, _) = row?;
            let key = k.value();
            if key.len() == 16 {
                let bytes: [u8; 16] = key.try_into().expect("checked len");
                ids.push(MemoryId(uuid::Uuid::from_bytes(bytes)));
            }
        }
        Ok(ids)
    }
    fn increment_retrieval_count(&self) -> Result<usize> {
        // Same key and same encoding as rocksdb: `stats:total_retrievals` -> u64 LE.
        // Read-modify-write, which is NOT atomic under concurrent readers and loses
        // increments — rocksdb has the identical race. Matched rather than fixed:
        // diverging silently is worse than sharing a defect that can be repaired in
        // both places at once.
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        let current = table_get(&self.database, DEFAULT_TABLE, RETRIEVAL_KEY)?
            .and_then(|b| b.try_into().ok().map(usize::from_le_bytes))
            .unwrap_or(0);
        let next = current + 1;
        table_put(&self.database, DEFAULT_TABLE, RETRIEVAL_KEY, &next.to_le_bytes())?;
        Ok(next)
    }
    fn get_stats(&self) -> Result<crate::memory::storage::StorageStats> {
        // Forgotten memories are EXCLUDED from every figure, matching rocksdb —
        // total_count, size, compressed and importance all skip them, so a corpus
        // that has been forgotten down reports smaller rather than unchanged.
        let mut stats = crate::memory::storage::StorageStats::default();
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        for row in table.iter()? {
            let (k, v) = row?;
            if k.value().len() != 16 {
                continue;
            }
            match decode::<Memory>(v.value()) {
                Ok(memory) => {
                    if memory.is_forgotten() {
                        continue;
                    }
                    stats.total_count += 1;
                    stats.total_size_bytes += v.value().len();
                    if memory.compressed {
                        stats.compressed_count += 1;
                    }
                    stats.importance_sum += memory.importance();
                }
                // A record that will not decode is COUNTED as corrupt and skipped,
                // never silently dropped from the denominator as if it were absent.
                Err(e) => tracing::warn!(error = %e, "corrupted memory entry during get_stats"),
            }
        }
        if stats.total_count > 0 {
            stats.average_importance = stats.importance_sum / stats.total_count as f32;
        }
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        stats.total_retrievals = table_get(&self.database, DEFAULT_TABLE, RETRIEVAL_KEY)?
            .filter(|b| b.len() >= 8)
            .and_then(|b| b[..8].try_into().ok().map(usize::from_le_bytes))
            .unwrap_or(0);
        Ok(stats)
    }
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        // rocksdb answers this from a blinded index (`external:{blind_term}:{id}`).
        // redb has no such index, so this is a scan — same ANSWER, different cost.
        // Recorded rather than silently accepted: on a large store this is O(n) where
        // the rocksdb path is a seek, and it is a post-cutover index to build.
        Ok(self
            .scan_all_memories()?
            .into_iter()
            .find(|m| m.external_id.as_deref() == Some(external_id)))
    }
    fn get_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>> {
        Ok(self
            .scan_all_memories()?
            .into_iter()
            .filter(|m| m.parent_id.as_ref() == Some(parent_id))
            .collect())
    }
    fn get_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>> {
        // The 100-iteration bound is rocksdb's and is carried over deliberately: a
        // parent_id cycle is representable in the data, and without a bound this walk
        // does not terminate. Dropping a guard because the loop "obviously" ends at a
        // root is how the cycle case becomes a hang instead of a wrong answer.
        let mut ancestors = Vec::new();
        let mut current = memory_id.clone();
        for _ in 0..100 {
            let memory = self.get(&current)?;
            match &memory.parent_id {
                Some(parent_id) => {
                    let parent = self.get(parent_id)?;
                    ancestors.push(parent.clone());
                    current = parent_id.clone();
                }
                None => break,
            }
        }
        Ok(ancestors)
    }
    fn get_hierarchy_context(
        &self,
        memory_id: &MemoryId,
    ) -> Result<(Vec<Memory>, Memory, Vec<Memory>)> {
        let memory = self.get(memory_id)?;
        let ancestors = self.get_ancestors(memory_id)?;
        let children = self.get_children(memory_id)?;
        Ok((ancestors, memory, children))
    }
    fn get_uncompressed_older_than(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<Memory>> {
        // All three conditions, in rocksdb's order: not compressed, not forgotten,
        // older than the cutoff. Dropping the forgotten check would hand already-
        // discarded memories back to a compaction pass to compress.
        Ok(self
            .scan_all_memories()?
            .into_iter()
            .filter(|m| !m.compressed && !m.is_forgotten() && m.created_at < cutoff)
            .collect())
    }
    fn mark_forgotten_by_age(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MemoryId>> {
        // Soft delete: the record stays and gains two metadata keys. Both are
        // rocksdb's exact names — a reader looking for "forgotten" would not see a
        // memory marked under any other spelling, and is_forgotten() is what every
        // other method filters on.
        let now = chrono::Utc::now().to_rfc3339();
        let mut flagged = Vec::new();
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(MEMORIES_TABLE)?;
            let candidates: Vec<(Vec<u8>, Vec<u8>)> = table
                .iter()?
                .filter_map(|r| r.ok())
                .filter(|(k, _)| k.value().len() == 16)
                .map(|(k, v)| (k.value().to_vec(), v.value().to_vec()))
                .collect();
            for (key, value) in candidates {
                let mut memory = match decode::<Memory>(&value) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(error = %e,
                            "skipping a record that failed to deserialize during forget-by-age");
                        continue;
                    }
                };
                if memory.is_forgotten() || memory.created_at >= cutoff {
                    continue;
                }
                flagged.push(memory.id.clone());
                memory.experience.metadata.insert("forgotten".into(), "true".into());
                memory.experience.metadata.insert("forgotten_at".into(), now.clone());
                table.insert(key.as_slice(), encode(&memory)?.as_slice())?;
            }
        }
        // One transaction: rocksdb writes its whole batch or none of it, so a
        // partial forget-by-age is not a state either backend can produce.
        txn.commit()?;
        Ok(flagged)
    }
    fn save_interference_records(
        &self,
        memory_id: &str,
        records: &[crate::memory::replay::InterferenceRecord],
    ) -> Result<()> {
        // serde_json, not bincode — matching rocksdb, because these bytes have to be
        // readable by whichever backend is live after a cutover.
        let key = format!("interference:{memory_id}");
        let value = serde_json::to_vec(records)
            .context("Failed to serialize interference records")?;
        table_put(&self.database, DEFAULT_TABLE, key.as_bytes(), &value)
    }
    fn delete_interference_records(&self, memory_id: &str) -> Result<()> {
        let key = format!("interference:{memory_id}");
        table_remove(&self.database, DEFAULT_TABLE, key.as_bytes())?;
        Ok(())
    }
    fn clear_all_interference_records(&self) -> Result<usize> {
        // The prefix is `interference` with NO colon, which is rocksdb's, and it
        // therefore also removes `interference_meta:total` — the event counter — not
        // just the per-memory records. Replicated deliberately: whether that is
        // intended or a slip, the two backends must agree, and a divergence here
        // would show up as a counter that survives a clear on one engine only.
        let txn = self.database.begin_write()?;
        let mut count = 0usize;
        {
            let mut table = txn.open_table(DEFAULT_TABLE)?;
            let doomed: Vec<Vec<u8>> = table
                .iter()?
                .filter_map(|row| row.ok())
                .map(|(k, _)| k.value().to_vec())
                .filter(|k| k.starts_with(b"interference"))
                .collect();
            for key in doomed {
                let _ = table.remove(key.as_slice())?;
                count += 1;
            }
        }
        txn.commit()?;
        Ok(count)
    }
    fn cleanup_corrupted(&self) -> Result<usize> {
        // NO SKIP-PREFIX ALLOWLIST HERE, and that is a structural difference rather
        // than an omission. rocksdb walks the default column family, which fifteen
        // other subsystems share — facts, temporal facts, lineage, learning, geo,
        // vector mappings, interference — so it carries a 15-entry allowlist to
        // avoid deleting legitimate non-memory rows it cannot decode as a Memory.
        // redb keeps those in DEFAULT_TABLE and memories in MEMORIES_TABLE, so this
        // scan cannot reach them and an allowlist would protect nothing.
        //
        // The condition that makes it safe is therefore "MEMORIES_TABLE holds only
        // memories", not "this list is complete" — so it is asserted rather than
        // assumed: a key that is not a 16-byte UUID is left ALONE and reported,
        // because it is evidence the invariant has broken and deleting it would
        // destroy whatever put it there.
        let txn = self.database.begin_write()?;
        let mut deleted = 0usize;
        let mut foreign = 0usize;
        {
            let mut table = txn.open_table(MEMORIES_TABLE)?;
            let doomed: Vec<Vec<u8>> = table
                .iter()?
                .filter_map(|r| r.ok())
                .filter_map(|(k, v)| {
                    let key = k.value().to_vec();
                    if key.len() != 16 {
                        foreign += 1;
                        return None;
                    }
                    decode::<Memory>(v.value()).err().map(|_| key)
                })
                .collect();
            for key in doomed {
                let _ = table.remove(key.as_slice())?;
                deleted += 1;
            }
        }
        txn.commit()?;
        if foreign > 0 {
            tracing::error!(foreign,
                "MEMORIES_TABLE holds keys that are not 16-byte memory ids — left in place. \
                 Something is writing non-memory rows into the memories table, which is the \
                 invariant this cleanup relies on to be safe without an allowlist.");
        }
        Ok(deleted)
    }
    fn migrate_legacy(&self) -> Result<(usize, usize, usize)> {
        // redb has no legacy envelope: this backend was written against the current
        // format, so nothing here can be a v1 record needing re-encoding. Returning
        // (0, 0, 0) would be the easy answer and a false one — it reads as "no
        // records" rather than "no records needing migration". So it COUNTS, and
        // reports what it found: everything decodable is already current, and
        // anything that will not decode is reported as failed rather than ignored,
        // which is what cleanup_corrupted exists to act on.
        let mut already_current = 0usize;
        let mut failed = 0usize;
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        for row in table.iter()? {
            let (k, v) = row?;
            if k.value().len() != 16 {
                continue;
            }
            match decode::<Memory>(v.value()) {
                Ok(_) => already_current += 1,
                Err(_) => failed += 1,
            }
        }
        Ok((0, already_current, failed))
    }

    // These carried `-> usize` and had to PANIC to stay honest, because 0 would have
    // made "merged nothing" and "cannot merge" the same answer. The trait now returns
    // Result, so they refuse like everything else and a caller can handle it.
    fn run_merge_pass(&self, _user_id: &str, _limit: usize) -> Result<usize> {
        Err(unbuilt("run_merge_pass"))
    }
    fn gc_merge_tombstones(&self, _user_id: &str) -> Result<usize> {
        Err(unbuilt("gc_merge_tombstones"))
    }

    fn scan_prefix_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        // Ordered range scan over the memories table, stopping at the first key
        // that leaves the prefix. redb ranges are ordered, so this terminates
        // early rather than reading the table.
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        let mut keys = Vec::new();
        for row in table.range(prefix..)? {
            let (k, _) = row?;
            let key = k.value().to_vec();
            if !key.starts_with(prefix) {
                break;
            }
            keys.push(key);
        }
        Ok(keys)
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        RedbBatchWriter::new(self.database.clone()).write_batch(batch)
    }

    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        match table_get(&self.database, MEMORIES_TABLE, id.0.as_bytes())? {
            Some(bytes) => Ok(Some(decode::<Memory>(&bytes)?)),
            None => Ok(None),
        }
    }

    fn put_memory(&self, memory: &Memory) -> Result<()> {
        let bytes = encode(memory)?;
        table_put(&self.database, MEMORIES_TABLE, memory.id.0.as_bytes(), &bytes)
    }

    fn update_memory(&self, memory: &Memory) -> Result<()> {
        // redb insert is an upsert; update and put share semantics.
        self.put_memory(memory)
    }

    fn delete_memory(&self, id: &MemoryId) -> Result<()> {
        // Mirror RocksDB `delete_with_vectors`: drop the memory and its mapping.
        table_remove(&self.database, MEMORIES_TABLE, id.0.as_bytes())?;
        table_remove(&self.database, VMAPPINGS_TABLE, id.0.as_bytes())?;
        Ok(())
    }

    fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>> {
        let memories = self
            .scan_all_memories()?
            .into_iter()
            .filter(|memory| !memory.is_forgotten())
            .filter(|memory| search_matches(memory, &criteria))
            .collect();
        Ok(memories)
    }

    fn get_vector_mapping(&self, id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        match table_get(&self.database, VMAPPINGS_TABLE, id.0.as_bytes())? {
            Some(bytes) => Ok(Some(decode::<VectorMappingEntry>(&bytes)?)),
            None => Ok(None),
        }
    }

    fn put_vector_mapping(&self, id: &MemoryId, mapping: &VectorMappingEntry) -> Result<()> {
        let bytes = encode(mapping)?;
        table_put(&self.database, VMAPPINGS_TABLE, id.0.as_bytes(), &bytes)
    }

    fn delete_vector_mapping(&self, id: &MemoryId) -> Result<()> {
        table_remove(&self.database, VMAPPINGS_TABLE, id.0.as_bytes())?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // redb commits are durable per transaction — nothing buffered to flush.
        Ok(())
    }

    // ── The eight the long_term_memory flip surfaced. All eight answer for
    //    real: get_by_content_hash and set_fact_watermark have no error channel
    //    in their signatures, so an unbuilt() stub could not even fail loudly —
    //    it would BE the absence-as-success this file refuses everywhere else.

    fn get_all_vector_mappings(&self) -> Result<Vec<(MemoryId, VectorMappingEntry)>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(VMAPPINGS_TABLE)?;
        let mut mappings = Vec::new();
        for row in table.iter()? {
            let (k, v) = row?;
            let key = k.value();
            if key.len() == 16 {
                let bytes: [u8; 16] = key.try_into().expect("checked len");
                mappings.push((
                    MemoryId(uuid::Uuid::from_bytes(bytes)),
                    decode::<VectorMappingEntry>(v.value())?,
                ));
            }
        }
        Ok(mappings)
    }

    fn load_all_interference_records(
        &self,
    ) -> Result<(
        std::collections::HashMap<String, Vec<crate::memory::replay::InterferenceRecord>>,
        usize,
    )> {
        // Same default-keyspace rows as rocksdb: `interference:{memory_id}` JSON
        // arrays plus the `interference_meta:total` counter (which sorts after
        // the `interference:` prefix and is excluded by it, ':' < '_').
        let txn = self.database.begin_read()?;
        let table = txn.open_table(DEFAULT_TABLE)?;
        let mut history: std::collections::HashMap<
            String,
            Vec<crate::memory::replay::InterferenceRecord>,
        > = std::collections::HashMap::new();
        let mut total_events: usize = 0;
        let prefix: &[u8] = b"interference:";
        for row in table.range(prefix..)? {
            let (k, v) = row?;
            let key = k.value();
            if !key.starts_with(prefix) {
                break;
            }
            let key_str = String::from_utf8_lossy(key);
            if let Some(memory_id) = key_str.strip_prefix("interference:") {
                match serde_json::from_slice::<Vec<crate::memory::replay::InterferenceRecord>>(
                    v.value(),
                ) {
                    Ok(records) => {
                        total_events += records.len();
                        history.insert(memory_id.to_string(), records);
                    }
                    Err(e) => {
                        tracing::warn!(key = %key_str, error = %e,
                            "Failed to deserialize interference records, skipping");
                    }
                }
            }
        }
        let persisted_total = table
            .get(&b"interference_meta:total"[..])?
            .map(|v| v.value().to_vec())
            .and_then(|v| {
                if v.len() == 8 {
                    Some(u64::from_le_bytes(v[..8].try_into().unwrap()) as usize)
                } else {
                    None
                }
            })
            .unwrap_or(total_events);
        Ok((history, persisted_total.max(total_events)))
    }

    fn update_vector_mapping(&self, memory_id: &MemoryId, vector_ids: Vec<u32>) -> Result<()> {
        // Mirror of MemoryStorage::update_modality_vectors for Text: load the
        // existing entry so other modalities survive, then write it back.
        let mut entry = self.get_vector_mapping(memory_id)?.unwrap_or_default();
        entry.add_modality(crate::memory::storage::Modality::Text, vector_ids);
        self.put_vector_mapping(memory_id, &entry)
    }

    fn get_by_content_hash(&self, content: &str) -> Option<MemoryId> {
        // rocksdb answers from a content-hash index; redb has no such index yet,
        // so this scans and compares. Correct-if-slow, and the same answer:
        // "the memory whose content is exactly this". If remember-path dedupe
        // latency matters on redb, the index is the fix — not a None that
        // silently disables dedupe.
        let memories = self.scan_all_memories().ok()?;
        memories
            .into_iter()
            .find(|m| m.experience.content == content)
            .map(|m| m.id)
    }

    fn update_access(&self, id: &MemoryId) -> Result<()> {
        // Mirror of MemoryStorage::update_access: a missing memory is not an
        // error, and the metadata mutation goes through the same interior-
        // mutability path before the row is re-persisted.
        if let Some(memory) = self.get_memory(id)? {
            memory.update_access();
            self.update_memory(&memory)?;
        }
        Ok(())
    }

    fn mark_forgotten_by_importance(&self, threshold: f32) -> Result<Vec<MemoryId>> {
        // rocksdb flags all rows in one WriteBatch; redb updates per row, so a
        // crash mid-pass leaves a prefix flagged. The operation is idempotent
        // and re-runnable — already-forgotten rows are skipped — so the weaker
        // atomicity changes recovery, not outcome.
        let now = chrono::Utc::now().to_rfc3339();
        let mut flagged_ids = Vec::new();
        for mut memory in self.scan_all_memories()? {
            if memory.is_forgotten() {
                continue;
            }
            if memory.importance() < threshold {
                memory
                    .experience
                    .metadata
                    .insert("forgotten".to_string(), "true".to_string());
                memory
                    .experience
                    .metadata
                    .insert("forgotten_at".to_string(), now.clone());
                self.update_memory(&memory)?;
                flagged_ids.push(memory.id.clone());
            }
        }
        Ok(flagged_ids)
    }

    fn save_interference_event_count(&self, count: usize) -> Result<()> {
        // Same key and encoding as rocksdb's default keyspace, so an
        // export/import carries the row across backends unchanged.
        table_put(
            &self.database,
            DEFAULT_TABLE,
            b"interference_meta:total",
            &(count as u64).to_le_bytes(),
        )
    }

    fn get_fact_watermark(&self, user_id: &str) -> Option<i64> {
        let key = format!("_watermark:fact_extraction:{user_id}");
        match table_get(&self.database, DEFAULT_TABLE, key.as_bytes()) {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                Some(i64::from_le_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => None,
        }
    }

    fn set_fact_watermark(&self, user_id: &str, timestamp_millis: i64) {
        // Signature is MemoryStorage's exactly: returns () and logs, because
        // every call site treats the watermark as best-effort.
        let key = format!("_watermark:fact_extraction:{user_id}");
        if let Err(e) = table_put(
            &self.database,
            DEFAULT_TABLE,
            key.as_bytes(),
            &timestamp_millis.to_le_bytes(),
        ) {
            tracing::warn!("Failed to persist fact extraction watermark: {e}");
        }
    }

    fn key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        Arc::new(RedbTableKeyValueStore::new(
            Arc::clone(&self.database),
            "kv",
        ))
    }

    fn default_key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        Arc::new(RedbTableKeyValueStore::new(
            Arc::clone(&self.database),
            DEFAULT_TABLE_NAME,
        ))
    }
}

/// [`KeyValueStore`] view over one named table of an ALREADY-OPEN database.
///
/// By handle, never by path: redb permits one opener per file, so a store the
/// primary hands out must share the primary's database. Distinct from
/// [`RedbKeyValueStore`], which owns a whole `<namespace>.redb` file.
pub struct RedbTableKeyValueStore {
    database: Arc<redb::Database>,
    table: &'static str,
}

impl RedbTableKeyValueStore {
    pub fn new(database: Arc<redb::Database>, table: &'static str) -> Self {
        Self { database, table }
    }

    fn def(&self) -> redb::TableDefinition<'static, &'static [u8], &'static [u8]> {
        redb::TableDefinition::new(self.table)
    }
}

impl KeyValueStore for RedbTableKeyValueStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let txn = self.database.begin_read()?;
        let table = match txn.open_table(self.def()) {
            Ok(t) => t,
            // A table nothing has written yet reads as empty, matching a fresh
            // rocksdb family — not as an error.
            Err(_) => return Ok(None),
        };
        Ok(table.get(key)?.map(|v| v.value().to_vec()))
    }

    fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()> {
        table_put(&self.database, self.def(), key, value)
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        table_remove(&self.database, self.def(), key)?;
        Ok(())
    }

    fn scan_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let txn = self.database.begin_read()?;
        let table = match txn.open_table(self.def()) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };
        let mut out = Vec::new();
        for entry in table.range(prefix..)? {
            let (k, v) = entry?;
            let key_bytes = k.value();
            if !key_bytes.starts_with(prefix) {
                break;
            }
            out.push((key_bytes.to_vec(), v.value().to_vec()));
            if limit.is_some_and(|max| out.len() >= max) {
                break;
            }
        }
        Ok(out)
    }

    fn flush(&self) -> Result<()> {
        // Each write is committed durably in its own transaction.
        Ok(())
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(self.def())?;
            for (k, v) in pairs {
                table.insert(k.as_slice(), v.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(self.def())?;
            for k in keys {
                table.remove(k.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

}

pub struct RedbGraphStore {
    database: Arc<redb::Database>,
}

impl RedbGraphStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("graph.redb");
        let database = create_redb_database(&database_path)?;
        init_table(&database, ENTITIES_TABLE)?;
        init_table(&database, RELATIONSHIPS_TABLE)?;
        init_table(&database, EPISODES_TABLE)?;
        Ok(Self { database })
    }
}

impl GraphStore for RedbGraphStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_entity(&self, uuid: &uuid::Uuid) -> Result<Option<EntityNode>> {
        match table_get(&self.database, ENTITIES_TABLE, uuid.as_bytes())? {
            Some(bytes) => Ok(Some(decode::<EntityNode>(&bytes)?)),
            None => Ok(None),
        }
    }

    fn put_entity(&self, entity: &EntityNode) -> Result<()> {
        // Persistence only: entity de-duplication is the graph engine's
        // responsibility (it owns the name/embedding indexes), not the store's.
        let bytes = encode(entity)?;
        table_put(&self.database, ENTITIES_TABLE, entity.uuid.as_bytes(), &bytes)
    }

    fn get_relationship(&self, uuid: &uuid::Uuid) -> Result<Option<RelationshipEdge>> {
        match table_get(&self.database, RELATIONSHIPS_TABLE, uuid.as_bytes())? {
            Some(bytes) => Ok(Some(decode::<RelationshipEdge>(&bytes)?)),
            None => Ok(None),
        }
    }

    fn put_relationship(&self, edge: &RelationshipEdge) -> Result<()> {
        let bytes = encode(edge)?;
        table_put(
            &self.database,
            RELATIONSHIPS_TABLE,
            edge.uuid.as_bytes(),
            &bytes,
        )
    }

    fn list_relationships_for_entity(
        &self,
        entity_uuid: &uuid::Uuid,
    ) -> Result<Vec<RelationshipEdge>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(RELATIONSHIPS_TABLE)?;
        let mut edges = Vec::new();
        for entry in table.iter()? {
            let (_key, value) = entry?;
            let edge: RelationshipEdge = decode(value.value())?;
            if edge.from_entity == *entity_uuid || edge.to_entity == *entity_uuid {
                edges.push(edge);
            }
        }
        Ok(edges)
    }

    fn get_episode(&self, uuid: &uuid::Uuid) -> Result<Option<EpisodicNode>> {
        match table_get(&self.database, EPISODES_TABLE, uuid.as_bytes())? {
            Some(bytes) => Ok(Some(decode::<EpisodicNode>(&bytes)?)),
            None => Ok(None),
        }
    }

    fn put_episode(&self, episode: &EpisodicNode) -> Result<()> {
        let bytes = encode(episode)?;
        table_put(&self.database, EPISODES_TABLE, episode.uuid.as_bytes(), &bytes)
    }

    fn stats(&self) -> Result<GraphStats> {
        let txn = self.database.begin_read()?;
        let count = |table: redb::TableDefinition<'static, &[u8], &[u8]>| -> Result<usize> {
            Ok(txn.open_table(table)?.len()? as usize)
        };
        Ok(GraphStats {
            entity_count: count(ENTITIES_TABLE)?,
            relationship_count: count(RELATIONSHIPS_TABLE)?,
            episode_count: count(EPISODES_TABLE)?,
        })
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Single table backing each namespace's key-value `.redb` file. Keys and values
/// are opaque byte strings (the trait's contract); callers serialize above this.
/// The `kv` family. NOT derived from `DEFAULT_TABLE_NAME` — see that constant.
const KV_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> =
    redb::TableDefinition::new("kv");

/// redb-backed [`KeyValueStore`]. Each namespace is its own `<namespace>.redb` file
/// holding a single `kv` table. Operations are one transaction each (redb commits are
/// durable, so [`flush`](KeyValueStore::flush) is a no-op). Prefix scans use redb's
/// ordered byte-key range and stop at the first key that leaves the prefix.
pub struct RedbKeyValueStore {
    database: Arc<redb::Database>,
    #[allow(dead_code)]
    path: PathBuf,
    #[allow(dead_code)]
    namespace: String,
}

impl RedbKeyValueStore {
    pub fn open(path: &Path, namespace: &str) -> Result<Self> {
        let database_path = path.join(format!("{}.redb", namespace));
        let database = create_redb_database(&database_path)?;
        // Materialize the table up front so reads never hit `TableDoesNotExist`.
        {
            let txn = database.begin_write()?;
            txn.open_table(KV_TABLE)?;
            txn.commit()?;
        }
        Ok(Self {
            database,
            path: database_path,
            namespace: namespace.to_string(),
        })
    }
}

impl KeyValueStore for RedbKeyValueStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(KV_TABLE)?;
        Ok(table.get(key)?.map(|v| v.value().to_vec()))
    }

    fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(KV_TABLE)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(KV_TABLE)?;
            table.remove(key)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn scan_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(KV_TABLE)?;
        let mut out = Vec::new();
        for entry in table.range(prefix..)? {
            let (k, v) = entry?;
            let key_bytes = k.value();
            if !key_bytes.starts_with(prefix) {
                break;
            }
            out.push((key_bytes.to_vec(), v.value().to_vec()));
            if limit.is_some_and(|max| out.len() >= max) {
                break;
            }
        }
        Ok(out)
    }

    fn flush(&self) -> Result<()> {
        // Each write is committed durably in its own transaction — nothing to flush.
        Ok(())
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(KV_TABLE)?;
            for (k, v) in pairs {
                table.insert(k.as_slice(), v.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
        let txn = self.database.begin_write()?;
        {
            let mut table = txn.open_table(KV_TABLE)?;
            for k in keys {
                table.remove(k.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

}

#[cfg(test)]
mod redb_kv_tests {
    use super::*;
    use crate::storage::KeyValueStore;

    #[test]
    fn round_trip_overwrite_scan_delete_and_persist() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let kv = RedbKeyValueStore::open(tmp.path(), "test_ns").expect("open");

        // Missing key reads as None (table exists, key absent).
        assert_eq!(kv.get_bytes(b"missing").unwrap(), None);

        // Put / get.
        kv.put_bytes(b"dials:a", b"one").unwrap();
        kv.put_bytes(b"dials:b", b"two").unwrap();
        kv.put_bytes(b"other:c", b"three").unwrap();
        assert_eq!(kv.get_bytes(b"dials:a").unwrap().as_deref(), Some(&b"one"[..]));

        // Overwrite.
        kv.put_bytes(b"dials:a", b"ONE").unwrap();
        assert_eq!(kv.get_bytes(b"dials:a").unwrap().as_deref(), Some(&b"ONE"[..]));

        // Prefix scan (ordered, prefix-bounded) + limit.
        let scan = kv.scan_prefix(b"dials:", None).unwrap();
        assert_eq!(
            scan.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
            vec![b"dials:a".to_vec(), b"dials:b".to_vec()]
        );
        assert_eq!(kv.scan_prefix(b"dials:", Some(1)).unwrap().len(), 1);

        // Delete.
        kv.delete_bytes(b"dials:a").unwrap();
        assert_eq!(kv.get_bytes(b"dials:a").unwrap(), None);

        // Durability across reopen.
        drop(kv);
        let kv2 = RedbKeyValueStore::open(tmp.path(), "test_ns").expect("reopen");
        assert_eq!(kv2.get_bytes(b"dials:b").unwrap().as_deref(), Some(&b"two"[..]));
        assert_eq!(kv2.get_bytes(b"dials:a").unwrap(), None);
    }
}

pub struct RedbAuditStore {
    database: Arc<redb::Database>,
}

impl RedbAuditStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("audit.redb");
        let database = create_redb_database(&database_path)?;
        init_table(&database, AUDIT_TABLE)?;
        Ok(Self { database })
    }
}

impl AuditStore for RedbAuditStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn append_event(&self, user_id: &str, event: &AuditLogEntry) -> Result<()> {
        // Same key scheme as the RocksDB audit store: zero-padded nanos keep a
        // user's events in chronological order under the ordered byte index.
        let timestamp_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
        let key = format!("{user_id}:{timestamp_nanos:020}");
        let value = encode(event)?;
        table_put(&self.database, AUDIT_TABLE, key.as_bytes(), &value)
    }

    fn list_user_ids(&self) -> Result<Vec<String>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(AUDIT_TABLE)?;
        let mut ids = std::collections::HashSet::new();
        for row in table.iter()? {
            let (k, _) = row?;
            if let Ok(key_str) = std::str::from_utf8(k.value()) {
                if let Some(user_id) = key_str.split(':').next() {
                    ids.insert(user_id.to_string());
                }
            }
        }
        Ok(ids.into_iter().collect())
    }

    fn list_events(&self, user_id: &str, limit: usize) -> Result<Vec<AuditLogEntry>> {
        let prefix = format!("{user_id}:");
        let prefix_bytes = prefix.as_bytes();
        let txn = self.database.begin_read()?;
        let table = txn.open_table(AUDIT_TABLE)?;
        let mut events = Vec::new();
        for entry in table.range(prefix_bytes..)? {
            let (key, value) = entry?;
            if !key.value().starts_with(prefix_bytes) {
                break;
            }
            events.push(decode::<AuditLogEntry>(value.value())?);
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
        let prefix_bytes = prefix.as_bytes();
        let cutoff_nanos = retention_before.timestamp_nanos_opt().unwrap_or(0);

        // Collect this user's keys in chronological order (oldest first).
        let keys: Vec<Vec<u8>> = {
            let txn = self.database.begin_read()?;
            let table = txn.open_table(AUDIT_TABLE)?;
            let mut keys = Vec::new();
            for entry in table.range(prefix_bytes..)? {
                let (key, _value) = entry?;
                if !key.value().starts_with(prefix_bytes) {
                    break;
                }
                keys.push(key.value().to_vec());
            }
            keys
        };

        // Drop entries older than the retention cutoff, plus the oldest `excess`
        // beyond the count cap — mirrors the RocksDB audit rotation.
        let excess = keys.len().saturating_sub(max_entries);
        let mut removed = 0usize;
        {
            let txn = self.database.begin_write()?;
            {
                let mut table = txn.open_table(AUDIT_TABLE)?;
                for (index, key) in keys.iter().enumerate() {
                    let key_str = std::str::from_utf8(key).unwrap_or_default();
                    let ts = key_str
                        .strip_prefix(&prefix)
                        .and_then(|raw| raw.parse::<i64>().ok())
                        .unwrap_or(0);
                    if ts < cutoff_nanos || index < excess {
                        table.remove(key.as_slice())?;
                        removed += 1;
                    }
                }
            }
            txn.commit()?;
        }
        Ok(removed)
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod redb_domain_tests {
    use super::*;
    use crate::graph_memory::{EntityLabel, EpisodeSource, PiiClassification, RelationType};
    use crate::memory::types::{Experience, ExperienceType, MemoryTier};
    use std::collections::HashMap;

    fn memory_with(content: &str, entities: &[&str], tags: &[&str]) -> Memory {
        let now = chrono::Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        Memory::from_legacy(
            MemoryId(uuid::Uuid::new_v4()),
            experience,
            0.5,
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

    #[test]
    fn primary_store_round_trips_memory_and_vector_mapping() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");

        let memory = memory_with("hello world", &["Veld"], &["greeting"]);
        let id = memory.id.clone();

        assert!(store.get_memory(&id).unwrap().is_none());
        store.put_memory(&memory).unwrap();

        let fetched = store.get_memory(&id).unwrap().expect("memory present");
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.experience.content, "hello world");

        // Vector mapping round-trip.
        assert!(store.get_vector_mapping(&id).unwrap().is_none());
        let mapping = VectorMappingEntry::with_text(vec![7, 11]);
        store.put_vector_mapping(&id, &mapping).unwrap();
        assert!(store.get_vector_mapping(&id).unwrap().is_some());

        // delete_memory removes both the memory and its mapping.
        store.delete_memory(&id).unwrap();
        assert!(store.get_memory(&id).unwrap().is_none());
        assert!(store.get_vector_mapping(&id).unwrap().is_none());

        // Durability across reopen.
        let memory2 = memory_with("persisted", &[], &[]);
        let id2 = memory2.id.clone();
        store.put_memory(&memory2).unwrap();
        drop(store);
        let reopened = RedbPrimaryMemoryStore::open(tmp.path()).expect("reopen");
        assert_eq!(
            reopened.get_memory(&id2).unwrap().unwrap().experience.content,
            "persisted"
        );
    }

    #[test]
    fn primary_store_search_filters_by_entity_tag_and_type() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");

        let a = memory_with("rust async", &["Rust", "Tokio"], &["lang"]);
        let b = memory_with("python data", &["Python"], &["lang", "data"]);
        store.put_memory(&a).unwrap();
        store.put_memory(&b).unwrap();

        let by_entity = store
            .search(SearchCriteria::ByEntity("rust".to_string()))
            .unwrap();
        assert_eq!(by_entity.len(), 1);
        assert_eq!(by_entity[0].id, a.id);

        let by_tag = store
            .search(SearchCriteria::ByTags(vec!["data".to_string()]))
            .unwrap();
        assert_eq!(by_tag.len(), 1);
        assert_eq!(by_tag[0].id, b.id);

        let by_type = store
            .search(SearchCriteria::ByType(ExperienceType::Observation))
            .unwrap();
        assert_eq!(by_type.len(), 2);

        // Combined is an intersection.
        let combined = store
            .search(SearchCriteria::Combined(vec![
                SearchCriteria::ByTags(vec!["lang".to_string()]),
                SearchCriteria::ByEntity("python".to_string()),
            ]))
            .unwrap();
        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0].id, b.id);
    }

    fn entity_with(name: &str) -> EntityNode {
        let now = chrono::Utc::now();
        EntityNode {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Person],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            pii_classification: PiiClassification::Clean,
        }
    }

    fn edge_between(from: uuid::Uuid, to: uuid::Uuid) -> RelationshipEdge {
        let now = chrono::Utc::now();
        RelationshipEdge {
            uuid: uuid::Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::WorksWith,
            strength: 0.5,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: now,
            activation_count: 0,
            ltp_status: Default::default(),
            tier: Default::default(),
            activation_timestamps: None,
            entity_confidence: None,
            created_by: Default::default(),
            forward_strength: 0.0,
            backward_strength: 0.0,
        }
    }

    #[test]
    fn graph_store_round_trips_nodes_and_reports_stats() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbGraphStore::open(tmp.path()).expect("open");

        let alice = entity_with("Alice");
        let bob = entity_with("Bob");
        store.put_entity(&alice).unwrap();
        store.put_entity(&bob).unwrap();
        assert_eq!(
            store.get_entity(&alice.uuid).unwrap().unwrap().name,
            "Alice"
        );
        assert!(store.get_entity(&uuid::Uuid::new_v4()).unwrap().is_none());

        let edge = edge_between(alice.uuid, bob.uuid);
        store.put_relationship(&edge).unwrap();
        let for_alice = store.list_relationships_for_entity(&alice.uuid).unwrap();
        assert_eq!(for_alice.len(), 1);
        assert_eq!(for_alice[0].uuid, edge.uuid);
        assert_eq!(
            store.list_relationships_for_entity(&bob.uuid).unwrap().len(),
            1
        );

        let episode = EpisodicNode {
            uuid: uuid::Uuid::new_v4(),
            name: "ep".to_string(),
            content: "episode content".to_string(),
            valid_at: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            entity_refs: vec![alice.uuid],
            source: EpisodeSource::Observation,
            metadata: HashMap::new(),
        };
        store.put_episode(&episode).unwrap();
        assert_eq!(
            store.get_episode(&episode.uuid).unwrap().unwrap().name,
            "ep"
        );

        let stats = store.stats().unwrap();
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.relationship_count, 1);
        assert_eq!(stats.episode_count, 1);
    }

    #[test]
    fn audit_store_appends_lists_in_order_and_rotates() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbAuditStore::open(tmp.path()).expect("open");

        let base = chrono::Utc::now();
        for i in 0..5 {
            let event = AuditLogEntry {
                timestamp: base + chrono::Duration::seconds(i),
                event_type: "write".to_string(),
                memory_id: format!("m{i}"),
                details: format!("event {i}"),
            };
            store.append_event("alice", &event).unwrap();
        }
        // A different user's events must not bleed in.
        store
            .append_event(
                "bob",
                &AuditLogEntry {
                    timestamp: base,
                    event_type: "write".to_string(),
                    memory_id: "mb".to_string(),
                    details: "bob".to_string(),
                },
            )
            .unwrap();

        let listed = store.list_events("alice", 10).unwrap();
        assert_eq!(listed.len(), 5);
        // Chronological order (zero-padded nanos key).
        assert_eq!(listed[0].memory_id, "m0");
        assert_eq!(listed[4].memory_id, "m4");

        // Limit is honored.
        assert_eq!(store.list_events("alice", 2).unwrap().len(), 2);

        // Rotate down to the 2 most-recent; retention cutoff in the far past so
        // only the count cap applies.
        let removed = store
            .rotate_events("alice", 2, base - chrono::Duration::days(1))
            .unwrap();
        assert_eq!(removed, 3);
        let after = store.list_events("alice", 10).unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].memory_id, "m3");
        assert_eq!(after[1].memory_id, "m4");

        // Bob untouched.
        assert_eq!(store.list_events("bob", 10).unwrap().len(), 1);
    }
}

pub struct RedbBackupStore {
    path: PathBuf,
    /// The SAME handle the primary store uses, when there is one.
    ///
    /// redb allows one opener per database file. A backup store that opens by path
    /// therefore cannot run while the store it backs up is open — which is every
    /// moment that matters. Caught by its own tests with "Database already open.
    /// Cannot acquire lock.", after this file's own BatchWriter note already said a
    /// second opener dies on the lock. The rocksdb side reached the same conclusion
    /// and refuses at the factory: "requires a bound primary DB reference".
    database: Arc<redb::Database>,
}

impl RedbBackupStore {
    fn backup_dir(&self, user_id: &str) -> PathBuf {
        self.path.join("backups").join(user_id)
    }

    /// The next id, from a persisted high-water mark — NOT from what is on disk.
    ///
    /// Deriving it from the directory looked right and is wrong: delete a backup and
    /// the next one takes its number, so a restore that names id 2 resolves to
    /// different bytes than it did yesterday, silently. Its own test caught this.
    ///
    /// The counter is the source, and the directory is a FLOOR under it rather than
    /// the answer — if the marker is lost the ids still cannot go backwards past
    /// what exists. Two sources, and the higher wins, because the failure being
    /// prevented is only ever an id going DOWN.
    fn next_backup_id(&self, dir: &Path) -> Result<u32> {
        let marker = dir.join(".next_backup_id");
        let recorded: u32 = std::fs::read_to_string(&marker)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let mut on_disk = 0u32;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                if let Some(stem) = e.path().file_stem().and_then(|s| s.to_str()) {
                    if let Ok(n) = stem.trim_end_matches(".meta").parse::<u32>() {
                        on_disk = on_disk.max(n);
                    }
                }
            }
        }

        let next = recorded.max(on_disk) + 1;
        std::fs::write(&marker, next.to_string())
            .with_context(|| format!("recording the next backup id in {}", marker.display()))?;
        Ok(next)
    }

    /// Every table in the store, sorted. These are redb's column families.
    fn table_names(&self) -> Result<Vec<String>> {
        let read = self.database.begin_read()?;
        let mut names: Vec<String> = read.list_tables()?.map(|t| t.name().to_string()).collect();
        names.sort();
        Ok(names)
    }

    fn count_memories(&self) -> Result<usize> {
        let txn = self.database.begin_read()?;
        match txn.open_table(MEMORIES_TABLE) {
            Ok(t) => Ok(t.len()? as usize),
            Err(_) => Ok(0),
        }
    }

    /// Share a live handle — the constructor a running server must use.
    pub fn from_database(database: Arc<redb::Database>, path: &Path) -> Self {
        Self { path: path.to_path_buf(), database }
    }

    /// Open standalone. Only valid when nothing else holds the database: an offline
    /// tool, or a restore into a fresh target. Fails on the lock otherwise, loudly.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let database = create_redb_database(&path.join("storage.redb"))?;
        Ok(Self {
            database,
            path: path.to_path_buf(),
        })
    }
}

impl BackupStore for RedbBackupStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn create_backup(&self, user_id: &str) -> Result<BackupMetadata> {
        // A redb backup IS a portable dump. rocksdb uses its own BackupEngine, which
        // has no redb analogue — so rather than a second mechanism this reuses
        // export_portable, which means the same verification (trailer, count, digest)
        // guards a backup and a cutover.
        let dir = self.backup_dir(user_id);
        std::fs::create_dir_all(&dir)?;
        let backup_id = self.next_backup_id(&dir)?;
        let dump = dir.join(format!("{backup_id}.veldport"));

        RedbMigrationStore::from_database(Arc::clone(&self.database), &self.path)
            .export_portable(user_id, &dump)?;

        // Read the trailer back rather than trusting the writer's return: the file on
        // disk is what a restore will use, and this is the one moment it is cheap to
        // confirm the two agree.
        let (records, digest) = crate::storage::portable::read_trailer(&dump)?;
        let memory_count = self.count_memories()?;
        let tables = self.table_names()?;
        let meta = BackupMetadata {
            backup_id,
            created_at: chrono::Utc::now(),
            user_id: user_id.to_string(),
            backup_type: crate::backup::BackupType::Full,
            size_bytes: std::fs::metadata(&dump)?.len(),
            checksum: crate::storage::portable::hex(&digest),
            memory_count,

            // The rocksdb-shaped fields, filled honestly rather than with zeros that
            // would read as measurements.
            //
            // `sequence_number` is documented "RocksDB sequence number (for PITR)".
            // redb has no analogue and the field is not Option, so 0 is the only
            // available value — it means ABSENT here, not "sequence zero", and must
            // not be read as a point-in-time anchor for a redb backup.
            sequence_number: 0,
            // Not empty. A redb dump contains every table in one file, so the
            // secondary stores ARE captured — reporting none would be the exact shape
            // of 4d1ec01, "a backup that omitted a store said success", with the sign
            // flipped: a backup that included them claiming it did not.
            secondary_stores: tables.clone(),
            // ...and their bytes are inside `size_bytes` above, not in separate
            // archives, so there is no second figure to report.
            secondary_size_bytes: 0,
            // redb has no intent log and no journal, so there is no consistency cut to
            // name. `journal_cut_lsn` is Option precisely for this and takes None;
            // an empty Vec here is a true statement, not a missing measurement.
            intent_log_files: Vec::new(),
            journal_cut_lsn: None,
            // redb tables are this backend's column families, one for one.
            column_families: tables,
        };
        std::fs::write(
            dir.join(format!("{backup_id}.meta.json")),
            serde_json::to_vec_pretty(&meta)?,
        )?;
        tracing::info!(user_id, backup_id, records, memory_count, "redb backup written");
        Ok(meta)
    }

    fn list_backups(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        let dir = self.backup_dir(user_id);
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // ENOENT is the only absence that means "no backups yet".
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).context(format!("reading {}", dir.display())),
        };

        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let meta: BackupMetadata = match std::fs::read(&path)
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
            {
                Some(m) => m,
                None => {
                    tracing::error!(path = %path.display(), "unreadable backup metadata — not listed");
                    continue;
                }
            };
            // THE SIDECAR IS CROSS-CHECKED, NEVER TRUSTED. It is a separate file from
            // the dump it describes, so it can drift — or survive a dump that was
            // truncated after it was written. A listing is a promise that these can be
            // restored, so a backup whose trailer disagrees with its own metadata is
            // reported as broken rather than offered as usable.
            let dump = dir.join(format!("{}.veldport", meta.backup_id));
            match crate::storage::portable::read_trailer(&dump) {
                Ok((_, digest)) if crate::storage::portable::hex(&digest) == meta.checksum => {
                    out.push(meta)
                }
                Ok(_) => tracing::error!(backup_id = meta.backup_id,
                    "backup checksum does not match its metadata — NOT listed as restorable"),
                Err(e) => tracing::error!(backup_id = meta.backup_id, error = %e,
                    "backup dump is unreadable — NOT listed as restorable"),
            }
        }
        out.sort_by_key(|m| m.backup_id);
        Ok(out)
    }

    fn restore_backup(&self, user_id: &str, backup_id: u32, target_path: &Path) -> Result<()> {
        let dump = self.backup_dir(user_id).join(format!("{backup_id}.veldport"));
        if !dump.exists() {
            bail!("no backup {backup_id} for '{user_id}' at {}", dump.display());
        }
        // Into a TARGET, never over the source. import_portable refuses a target that
        // already holds rows, so a restore cannot merge into a live store — the
        // failure mode the whole export/import/verify/switch route exists to avoid.
        let target = RedbMigrationStore::open(target_path)?;
        target.import_portable(user_id, &dump)?;
        target.verify_portable(&dump)?;
        tracing::info!(user_id, backup_id, target = %target_path.display(),
            "redb restore imported and verified");
        Ok(())
    }
}

pub struct RedbMigrationStore {
    path: PathBuf,
    database: Arc<redb::Database>,
}

impl RedbMigrationStore {
    /// Share a live handle. Required whenever the store being exported is open —
    /// redb permits ONE opener per database file, so opening the path again fails
    /// with "Database already open. Cannot acquire lock." rather than degrading.
    ///
    /// This is not a redb quirk to route around; it is the property that makes a
    /// single-file embedded database safe. What it means for this layer is that
    /// `StorageFactory::open_*(path)` cannot be the constructor for a SECOND store
    /// over an already-open database — the rocksdb side reached the same place and
    /// made `open_backup_store` refuse outright.
    pub fn from_database(database: Arc<redb::Database>, path: &Path) -> Self {
        Self { path: path.to_path_buf(), database }
    }

    /// Open standalone. Valid only when nothing else holds the database: an offline
    /// tool, or a fresh import target.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let database = create_redb_database(&path.join("storage.redb"))?;
        Ok(Self { path: path.to_path_buf(), database })
    }
}

impl RedbMigrationStore {
    /// Prove an imported target matches the dump it came from — R-6c's verify step.
    ///
    /// Separate from `import_portable` and re-runnable on purpose: the point of
    /// export -> import -> VERIFY -> switch is that the verify can be repeated by
    /// someone who did not run the import, before anything is pointed at the new
    /// store. A check that only ever runs inside the write it is checking is a
    /// check the operator has to take on trust.
    ///
    /// BOTH DIRECTIONS. "every dump row is present in the target" passes over a
    /// target that also holds a thousand rows from somewhere else, and "the counts
    /// match" passes over a target with the right number of wrong rows. Neither
    /// alone is worth running.
    ///
    /// Reports counts only. The values are memory contents and must not reach a
    /// log line to prove they were copied.
    pub fn verify_portable(&self, source: &Path) -> Result<()> {
        use crate::storage::portable::read_all;

        let file = std::fs::File::open(source)
            .with_context(|| format!("opening {}", source.display()))?;
        let records = read_all(file)?;
        let database = Arc::clone(&self.database);
        let read = database.begin_read()?;

        // Direction 1: everything the dump holds is in the target, byte-for-byte.
        let mut missing = 0u64;
        let mut differing = 0u64;
        for rec in &records {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                redb::TableDefinition::new(rec.cf.as_str());
            match read.open_table(def) {
                Err(_) => missing += 1,
                Ok(table) => match table.get(rec.key.as_slice())? {
                    None => missing += 1,
                    Some(v) if v.value() != rec.value.as_slice() => differing += 1,
                    Some(_) => {}
                },
            }
        }

        // Direction 2: the target holds nothing the dump does not. This is the half
        // that catches an import into a store that was not empty.
        let mut in_target = 0u64;
        for handle in read.list_tables()? {
            let name = handle.name().to_string();
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(&name);
            if let Ok(t) = read.open_table(def) {
                in_target += t.len()?;
            }
        }
        let extra = in_target.saturating_sub(records.len() as u64);

        if missing > 0 || differing > 0 || in_target != records.len() as u64 {
            bail!(
                "verify_portable: {} does not match {} — {missing} missing, {differing} differing, \
                 {extra} unaccounted for in the target ({} in the dump, {in_target} in the store). \
                 Do NOT switch the configured backend.",
                self.path.display(),
                source.display(),
                records.len()
            );
        }

        tracing::info!(records = records.len(), target = %self.path.display(),
            "portable import verified against its dump, both directions");
        Ok(())
    }
}

impl MigrationStore for RedbMigrationStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn export_portable(&self, user_id: &str, target: &Path) -> Result<()> {
        use crate::storage::portable::{PortableRecord, PortableWriter};

        let database = Arc::clone(&self.database);

        // TABLES ARE DISCOVERED, NOT LISTED — the opposite of the rocksdb side, and
        // deliberately. rocksdb names its three families explicitly and refuses a
        // store missing one, because a column family added later and not listed would
        // be silently omitted from every export while the dump still verified.
        // redb can enumerate, so an unexpected table is INCLUDED rather than dropped.
        // The protection rocksdb gets from its fixed list is kept by asserting the
        // known tables are present: a missing one is still an error, an extra one is
        // still exported. Strictly the safer half of each.
        let read = database.begin_read()?;
        let mut names: Vec<String> = read
            .list_tables()?
            .map(|t| t.name().to_string())
            .collect();
        names.sort(); // fixed order, so two exports of one store are byte-identical
        for required in [DEFAULT_TABLE_NAME, "memories", "vector_mappings"] {
            if !names.iter().any(|n| n == required) {
                bail!(
                    "export_portable: this database has no table '{required}' — refusing to write \
                     a dump that would be missing a keyspace"
                );
            }
        }

        let tmp = target.with_extension("partial");
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        let mut writer = PortableWriter::new(std::io::BufWriter::new(file))?;

        let written = (|| -> Result<(u64, [u8; 32])> {
            for name in &names {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                    redb::TableDefinition::new(name.as_str());
                let table = read.open_table(def)?;
                for row in table.iter()? {
                    let (k, v) = row?;
                    writer.write_record(&PortableRecord {
                        cf: name.clone(),
                        key: k.value().to_vec(),
                        value: v.value().to_vec(),
                    })?;
                }
            }
            writer.finish()
        })();

        let (count, _digest) = match written {
            Ok(v) => v,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        std::fs::rename(&tmp, target)
            .with_context(|| format!("renaming {} into place", tmp.display()))?;
        tracing::info!(user_id, records = count, tables = names.len(),
            target = %target.display(), "portable export written (redb)");
        Ok(())
    }

    fn import_portable(&self, user_id: &str, source: &Path) -> Result<()> {
        use crate::storage::portable::read_all;

        let file = std::fs::File::open(source)
            .with_context(|| format!("opening {}", source.display()))?;
        // Verified in full before a single row is applied. read_all checks the
        // trailer, the digest and the count, so a truncated or corrupt dump fails
        // here rather than half-way through writing.
        let records = read_all(file)?;

        let database = Arc::clone(&self.database);

        // REFUSE A TARGET THAT IS NOT EMPTY. An import that merges into live data
        // is unrecoverable by config revert, which is the property the whole
        // export -> import -> verify -> switch route exists to keep. A mis-aimed
        // path is the likely mistake here, and it is silent: the rows land, the
        // count is right, and the store now holds two users' data.
        {
            let read = database.begin_read()?;
            for table in read.list_tables()? {
                let name = table.name().to_string();
                let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                    redb::TableDefinition::new(&name);
                if let Ok(t) = read.open_table(def) {
                    if t.len()? > 0 {
                        bail!(
                            "import_portable: {} already holds rows in table '{}' — refusing. \
                             Import into a scratch target, verify it, then switch the configured \
                             backend; merging into a populated store cannot be undone by reverting \
                             a config value.",
                            self.path.display(),
                            name
                        );
                    }
                }
            }
        }

        // One transaction for the whole dump: a partially-applied import is a
        // store that looks like a smaller one.
        let txn = database.begin_write()?;
        {
            for rec in &records {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                    redb::TableDefinition::new(rec.cf.as_str());
                let mut table = txn.open_table(def)?;
                table.insert(rec.key.as_slice(), rec.value.as_slice())?;
            }
        }
        txn.commit()?;

        // Count what landed and compare. The dump's own count was checked against
        // the dump; this checks the STORE against the dump, which is a different
        // claim and the one a cutover actually rests on.
        let mut landed = 0u64;
        {
            let read = database.begin_read()?;
            for table in read.list_tables()? {
                let name = table.name().to_string();
                let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                    redb::TableDefinition::new(&name);
                if let Ok(t) = read.open_table(def) {
                    landed += t.len()?;
                }
            }
        }
        if landed != records.len() as u64 {
            bail!(
                "import_portable: dump held {} records and the store now holds {landed} — \
                 refusing to report a completed import",
                records.len()
            );
        }

        tracing::info!(user_id, records = landed, target = %self.path.display(),
            "portable import verified");
        Ok(())
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
#[cfg(test)]
mod redb_batch_tests {
    use super::*;
    use crate::storage::WriteBatch as StorageWriteBatch;

    fn db(tmp: &tempfile::TempDir) -> Arc<redb::Database> {
        let path = tmp.path().join("batch.redb");
        create_redb_database(&path).expect("create db")
    }

    #[test]
    fn a_batch_spanning_two_tables_commits_both() {
        // This is the property `supports_transactional_batch: true` actually
        // claims — atomic ACROSS families, not merely within one. It is the
        // property src/memory/prospective.rs:298 and :382 depend on.
        let tmp = tempfile::tempdir().expect("tempdir");
        let database = db(&tmp);
        init_table(&database, MEMORIES_TABLE).expect("init memories");
        init_table(&database, AUDIT_TABLE).expect("init audit");

        let writer = RedbBatchWriter::new(database.clone());
        let mut batch = StorageWriteBatch::new();
        batch.put_cf("memories", b"m1".to_vec(), b"mv".to_vec());
        batch.put_cf("audit", b"a1".to_vec(), b"av".to_vec());
        writer.write_batch(&batch).expect("write_batch");

        assert_eq!(
            table_get(&database, MEMORIES_TABLE, b"m1").expect("get"),
            Some(b"mv".to_vec())
        );
        assert_eq!(
            table_get(&database, AUDIT_TABLE, b"a1").expect("get"),
            Some(b"av".to_vec())
        );
    }

    #[test]
    fn a_batch_whose_second_op_cannot_apply_leaves_the_first_unapplied() {
        // THE test that stops the capability flag being a declaration with
        // nothing behind it. Without it, `supports_transactional_batch: true`
        // is indistinguishable from a loop of point writes.
        let tmp = tempfile::tempdir().expect("tempdir");
        let database = db(&tmp);
        init_table(&database, MEMORIES_TABLE).expect("init memories");

        let writer = RedbBatchWriter::new(database.clone());
        let mut batch = StorageWriteBatch::new();
        batch.put_cf("memories", b"first".to_vec(), b"v".to_vec());
        batch.put_cf("no_such_table", b"second".to_vec(), b"v".to_vec());

        let err = writer.write_batch(&batch).expect_err("must refuse");
        assert!(
            err.to_string().contains("no_such_table"),
            "the error must name the table it could not resolve, got: {err}"
        );
        assert_eq!(
            table_get(&database, MEMORIES_TABLE, b"first").expect("get"),
            None,
            "the FIRST op was applied despite the batch failing — this is the \
             partial write the capability flag says cannot happen"
        );
    }

    #[test]
    fn an_unknown_table_is_refused_rather_than_created() {
        // redb opens-or-CREATES a table in a write transaction, so without an
        // explicit existence check this batch would silently succeed and invent
        // a table — while rocksdb, whose column families are fixed at open,
        // would reject the same batch. Asserted so the two backends cannot
        // quietly disagree about what the same WriteBatch means.
        let tmp = tempfile::tempdir().expect("tempdir");
        let database = db(&tmp);
        let writer = RedbBatchWriter::new(database.clone());

        let mut batch = StorageWriteBatch::new();
        batch.put_cf("invented", b"k".to_vec(), b"v".to_vec());
        assert!(writer.write_batch(&batch).is_err());

        let read = database.begin_read().expect("read txn");
        let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new("invented");
        assert!(
            read.open_table(def).is_err(),
            "the refused batch created the table anyway"
        );
    }

    #[test]
    fn an_empty_batch_writes_nothing_and_succeeds() {
        // 7 call sites in src/ branch on is_empty() to skip the write. This
        // makes the skip unnecessary rather than merely conventional.
        let tmp = tempfile::tempdir().expect("tempdir");
        let database = db(&tmp);
        let writer = RedbBatchWriter::new(database.clone());
        writer
            .write_batch(&StorageWriteBatch::new())
            .expect("an empty batch is a no-op, not an error");
    }
}

#[cfg(test)]
mod keyspace_agreement_tests {
    use super::*;
    use crate::storage::WriteBatch as StorageWriteBatch;

    #[test]
    fn cf_none_and_cf_default_address_the_same_rows() {
        // The two backends have to agree on what an unlabelled op means, and the
        // only way to state that here is that `None` and `Some("default")` are the
        // same keyspace — which is what rocksdb does, where `cf: None` is db.put
        // into the default column family. When this was "kv" the same batch landed
        // in two different places depending on the backend, and nothing said so.
        let tmp = tempfile::tempdir().expect("tempdir");
        let database = create_redb_database(&tmp.path().join("agree.redb")).expect("db");
        init_table(&database, redb::TableDefinition::<&[u8], &[u8]>::new(DEFAULT_TABLE_NAME))
            .expect("init default");

        let writer = RedbBatchWriter::new(database.clone());
        let mut unlabelled = StorageWriteBatch::new();
        unlabelled.put(b"k".to_vec(), b"v".to_vec());
        writer.write_batch(&unlabelled).expect("unlabelled write");

        // Read it back through the EXPLICIT name. If the two disagreed, this is
        // where a batch would appear to have written nothing.
        let read = database.begin_read().expect("read");
        let def: redb::TableDefinition<'_, &[u8], &[u8]> =
            redb::TableDefinition::new(DEFAULT_TABLE_NAME);
        let table = read.open_table(def).expect("open default");
        assert_eq!(
            table.get(b"k".as_slice()).unwrap().map(|v| v.value().to_vec()),
            Some(b"v".to_vec())
        );
    }

    #[test]
    fn the_default_keyspace_is_not_the_kv_family() {
        // Asserted separately and in the opposite direction: if someone re-derives
        // one constant from the other, the test above still passes while this one
        // fails, which is the signal they have been conflated again.
        assert_ne!(DEFAULT_TABLE_NAME, "kv");
        assert_eq!(DEFAULT_TABLE_NAME, "default");
    }
}

#[cfg(test)]
mod import_portable_tests {
    use super::*;
    use crate::storage::portable::{PortableRecord, PortableWriter};

    /// Dumps are built HERE, from records, not by calling the rocksdb exporter.
    /// An importer tested against its own exporter's output proves the pair agree
    /// and says nothing about whether either matches the format.
    fn dump_of(records: &[PortableRecord]) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut w = PortableWriter::new(&mut buf).unwrap();
        for r in records {
            w.write_record(r).unwrap();
        }
        w.finish().unwrap();
        buf
    }

    fn rec(cf: &str, k: &[u8], v: &[u8]) -> PortableRecord {
        PortableRecord { cf: cf.into(), key: k.to_vec(), value: v.to_vec() }
    }

    fn write_dump(dir: &Path, bytes: &[u8]) -> PathBuf {
        let p = dir.join("in.veldport");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn row_count(target: &Path) -> u64 {
        let db = create_redb_database(&target.join("storage.redb")).unwrap();
        let read = db.begin_read().unwrap();
        let mut n = 0;
        for t in read.list_tables().unwrap() {
            let name = t.name().to_string();
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(&name);
            if let Ok(tbl) = read.open_table(def) {
                n += tbl.len().unwrap();
            }
        }
        n
    }

    #[test]
    fn every_keyspace_in_the_dump_lands_in_its_own_table() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write_dump(
            tmp.path(),
            &dump_of(&[
                rec("default", b"acl:alice:1", b"a"),
                rec("memory_index", b"i", b"b"),
                rec("kv", b"dials", b"c"),
            ]),
        );
        let target = tmp.path().join("target");
        RedbMigrationStore::open(&target)
            .unwrap()
            .import_portable("alice", &src)
            .expect("import");

        let db = create_redb_database(&target.join("storage.redb")).unwrap();
        let read = db.begin_read().unwrap();
        for (cf, key, want) in [
            ("default", &b"acl:alice:1"[..], &b"a"[..]),
            ("memory_index", b"i", b"b"),
            ("kv", b"dials", b"c"),
        ] {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
            let t = read.open_table(def).unwrap_or_else(|e| panic!("table {cf} missing: {e}"));
            assert_eq!(t.get(key).unwrap().map(|v| v.value().to_vec()), Some(want.to_vec()));
        }
    }

    #[test]
    fn a_populated_target_is_refused_and_left_alone() {
        // The likely mistake is a mis-aimed path, and it is silent: the rows land,
        // the count is right, and the store now holds two users' data.
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("target");
        let first = write_dump(tmp.path(), &dump_of(&[rec("kv", b"already", b"here")]));
        RedbMigrationStore::open(&target).unwrap().import_portable("a", &first).unwrap();
        assert_eq!(row_count(&target), 1);

        let second = tmp.path().join("second.veldport");
        std::fs::write(&second, dump_of(&[rec("kv", b"new", b"row")])).unwrap();
        let err = RedbMigrationStore::open(&target)
            .unwrap()
            .import_portable("b", &second)
            .expect_err("a populated target must refuse");
        assert!(err.to_string().contains("already holds rows"), "got: {err}");
        assert_eq!(row_count(&target), 1, "the refused import still wrote");
    }

    #[test]
    fn a_truncated_dump_applies_nothing() {
        // Verification precedes application, so the target is untouched — not
        // partially filled with whatever preceded the cut.
        let tmp = tempfile::tempdir().unwrap();
        let full = dump_of(&[rec("kv", b"a", b"1"), rec("kv", b"b", b"2")]);
        let src = write_dump(tmp.path(), &full[..full.len() - 12]);
        let target = tmp.path().join("target");
        let err = RedbMigrationStore::open(&target)
            .unwrap()
            .import_portable("a", &src)
            .expect_err("a truncated dump must refuse");
        assert!(err.to_string().contains("trailer"), "got: {err}");
        assert_eq!(row_count(&target), 0, "a refused import left rows behind");
    }

    #[test]
    fn an_empty_dump_imports_as_an_empty_store() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write_dump(tmp.path(), &dump_of(&[]));
        let target = tmp.path().join("target");
        RedbMigrationStore::open(&target).unwrap().import_portable("a", &src).expect("import");
        assert_eq!(row_count(&target), 0);
    }
}

#[cfg(test)]
mod verify_portable_tests {
    use super::*;
    use crate::storage::portable::{PortableRecord, PortableWriter};

    fn rec(cf: &str, k: &[u8], v: &[u8]) -> PortableRecord {
        PortableRecord { cf: cf.into(), key: k.to_vec(), value: v.to_vec() }
    }

    fn dump_at(dir: &Path, records: &[PortableRecord]) -> PathBuf {
        let mut buf = Vec::new();
        let mut w = PortableWriter::new(&mut buf).unwrap();
        for r in records { w.write_record(r).unwrap(); }
        w.finish().unwrap();
        let p = dir.join("d.veldport");
        std::fs::write(&p, &buf).unwrap();
        p
    }

    fn imported(dir: &Path, records: &[PortableRecord]) -> (PathBuf, PathBuf) {
        let src = dump_at(dir, records);
        let target = dir.join("t");
        RedbMigrationStore::open(&target).unwrap().import_portable("u", &src).unwrap();
        (src, target)
    }

    #[test]
    fn a_faithful_import_verifies() {
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(tmp.path(), &[
            rec("default", b"a", b"1"),
            rec("kv", b"b", b"2"),
        ]);
        RedbMigrationStore::open(&target).unwrap().verify_portable(&src).expect("verify");
    }

    #[test]
    fn a_changed_value_is_caught() {
        // Counts alone would pass this: the right number of rows, one of them wrong.
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(tmp.path(), &[rec("kv", b"b", b"2")]);
        {
            let db = create_redb_database(&target.join("storage.redb")).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new("kv");
                let mut t = txn.open_table(def).unwrap();
                t.insert(b"b".as_slice(), b"tampered".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }
        let err = RedbMigrationStore::open(&target)
            .unwrap()
            .verify_portable(&src)
            .expect_err("a differing value must fail verification");
        assert!(err.to_string().contains("1 differing"), "got: {err}");
    }

    #[test]
    fn a_row_the_dump_does_not_have_is_caught() {
        // The direction that catches an import into a store that was not empty.
        // "every dump row is present" passes here; only the reverse check fails.
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(tmp.path(), &[rec("kv", b"b", b"2")]);
        {
            let db = create_redb_database(&target.join("storage.redb")).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new("kv");
                let mut t = txn.open_table(def).unwrap();
                t.insert(b"stowaway".as_slice(), b"x".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }
        let err = RedbMigrationStore::open(&target)
            .unwrap()
            .verify_portable(&src)
            .expect_err("an extra row must fail verification");
        assert!(err.to_string().contains("unaccounted for"), "got: {err}");
    }

    #[test]
    fn a_missing_row_is_caught() {
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(tmp.path(), &[rec("kv", b"b", b"2"), rec("kv", b"c", b"3")]);
        {
            let db = create_redb_database(&target.join("storage.redb")).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new("kv");
                let mut t = txn.open_table(def).unwrap();
                let _ = t.remove(b"c".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }
        let err = RedbMigrationStore::open(&target)
            .unwrap()
            .verify_portable(&src)
            .expect_err("a missing row must fail verification");
        assert!(err.to_string().contains("1 missing"), "got: {err}");
    }

    #[test]
    fn verification_is_re_runnable_and_does_not_mutate_the_target() {
        // R-6c's whole shape depends on this: the operator runs verify BEFORE
        // switching, possibly without having run the import.
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(tmp.path(), &[rec("kv", b"b", b"2")]);
        let store = RedbMigrationStore::open(&target).unwrap();
        store.verify_portable(&src).unwrap();
        store.verify_portable(&src).unwrap();
        store.verify_portable(&src).expect("verify must be idempotent");
    }
}

#[cfg(test)]
mod redb_hierarchy_tests {
    use super::*;
    use crate::memory::types::{Experience, ExperienceType, MemoryTier};

    fn mem(parent: Option<MemoryId>, external: Option<&str>) -> Memory {
        let now = chrono::Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "x".into(),
            ..Default::default()
        };
        let mut m = Memory::from_legacy(
            MemoryId(uuid::Uuid::new_v4()), experience, 0.5, 0, now, now, false,
            MemoryTier::LongTerm, Vec::new(), 1.0, None, None, None, None, 0.0, None, None, 1,
            Vec::new(), Vec::new(),
        );
        m.parent_id = parent;
        m.external_id = external.map(|s| s.to_string());
        m
    }

    fn store(tmp: &tempfile::TempDir) -> RedbPrimaryMemoryStore {
        RedbPrimaryMemoryStore::open(tmp.path()).expect("open")
    }

    #[test]
    fn get_all_and_get_all_ids_agree_on_the_same_population() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let ids: Vec<_> = (0..5).map(|_| { let m = mem(None, None); s.put_memory(&m).unwrap(); m.id }).collect();
        assert_eq!(s.get_all().unwrap().len(), 5);
        let mut got = s.get_all_ids().unwrap();
        let mut want = ids;
        got.sort_by_key(|i| i.0); want.sort_by_key(|i| i.0);
        assert_eq!(got, want, "get_all_ids must be the same set get_all returns, not a subset");
    }

    #[test]
    fn children_are_the_ones_naming_this_parent_and_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let parent = mem(None, None);
        s.put_memory(&parent).unwrap();
        let a = mem(Some(parent.id.clone()), None);
        let b = mem(Some(parent.id.clone()), None);
        let unrelated = mem(None, None);
        for m in [&a, &b, &unrelated] { s.put_memory(m).unwrap(); }

        let kids = s.get_children(&parent.id).unwrap();
        assert_eq!(kids.len(), 2);
        // Both directions: the two are present AND the unrelated one is not.
        assert!(kids.iter().any(|m| m.id == a.id) && kids.iter().any(|m| m.id == b.id));
        assert!(!kids.iter().any(|m| m.id == unrelated.id), "an unparented memory was returned as a child");
    }

    #[test]
    fn ancestors_walk_to_the_root_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let root = mem(None, None);
        let mid = mem(Some(root.id.clone()), None);
        let leaf = mem(Some(mid.id.clone()), None);
        for m in [&root, &mid, &leaf] { s.put_memory(m).unwrap(); }

        let anc = s.get_ancestors(&leaf.id).unwrap();
        assert_eq!(anc.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![mid.id.clone(), root.id.clone()],
            "ancestors must be nearest-first, matching the rocksdb walk");
        assert!(s.get_ancestors(&root.id).unwrap().is_empty(), "a root has no ancestors");
    }

    #[test]
    fn a_parent_cycle_terminates_instead_of_hanging() {
        // THE test the bound exists for. A cycle is representable in the data, and
        // without the 100-iteration guard this walk never returns — the failure is a
        // hung request, not a wrong answer, so no assertion downstream would catch it.
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let mut a = mem(None, None);
        let mut b = mem(None, None);
        a.parent_id = Some(b.id.clone());
        b.parent_id = Some(a.id.clone());
        s.put_memory(&a).unwrap();
        s.put_memory(&b).unwrap();

        let anc = s.get_ancestors(&a.id).unwrap();
        assert_eq!(anc.len(), 100, "the walk must stop at the rocksdb bound, not run forever");
    }

    #[test]
    fn external_id_lookup_finds_the_one_and_only_that_one() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let tagged = mem(None, Some("ext-42"));
        let other = mem(None, Some("ext-99"));
        let untagged = mem(None, None);
        for m in [&tagged, &other, &untagged] { s.put_memory(m).unwrap(); }

        assert_eq!(s.find_by_external_id("ext-42").unwrap().map(|m| m.id), Some(tagged.id));
        assert!(s.find_by_external_id("ext-nothing").unwrap().is_none(),
            "an unknown external id must be None, never the first memory scanned");
    }

    #[test]
    fn hierarchy_context_is_its_three_parts() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let root = mem(None, None);
        let target = mem(Some(root.id.clone()), None);
        let kid = mem(Some(target.id.clone()), None);
        for m in [&root, &target, &kid] { s.put_memory(m).unwrap(); }

        let (anc, me, kids) = s.get_hierarchy_context(&target.id).unwrap();
        assert_eq!(me.id, target.id);
        assert_eq!(anc.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![root.id]);
        assert_eq!(kids.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![kid.id]);
    }
}

#[cfg(test)]
mod redb_backup_tests {
    use super::*;
    use crate::memory::types::{Experience, ExperienceType, MemoryTier};

    fn seeded_store(dir: &Path, n: usize) -> RedbPrimaryMemoryStore {
        let s = RedbPrimaryMemoryStore::open(dir).expect("open");
        let now = chrono::Utc::now();
        for _ in 0..n {
            let m = Memory::from_legacy(
                MemoryId(uuid::Uuid::new_v4()),
                Experience { experience_type: ExperienceType::Observation, content: "x".into(), ..Default::default() },
                0.5, 0, now, now, false, MemoryTier::LongTerm, Vec::new(), 1.0,
                None, None, None, None, 0.0, None, None, 1, Vec::new(), Vec::new(),
            );
            s.put_memory(&m).unwrap();
        }
        s
    }

    #[test]
    fn create_list_restore_round_trips_and_the_target_holds_the_memories() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 4);

        let backups = RedbBackupStore::from_database(_s.database(), &store_dir);
        let meta = backups.create_backup("alice").expect("create");
        assert_eq!(meta.memory_count, 4);
        assert_eq!(meta.backup_id, 1);
        assert!(!meta.checksum.is_empty());

        let listed = backups.list_backups("alice").expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].checksum, meta.checksum);

        let target = tmp.path().join("restored");
        backups.restore_backup("alice", 1, &target).expect("restore");
        let restored = RedbPrimaryMemoryStore::open(&target).unwrap();
        assert_eq!(restored.get_all().unwrap().len(), 4, "the restore did not carry the memories");
    }

    #[test]
    fn a_backup_whose_dump_was_truncated_after_the_sidecar_is_not_listed() {
        // The reason the sidecar is cross-checked rather than trusted: it is a
        // separate file, so it survives damage to the dump it describes. Listing it
        // would promise a restore that cannot happen.
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 2);
        let backups = RedbBackupStore::from_database(_s.database(), &store_dir);
        backups.create_backup("alice").unwrap();

        let dump = store_dir.join("backups").join("alice").join("1.veldport");
        let bytes = std::fs::read(&dump).unwrap();
        std::fs::write(&dump, &bytes[..bytes.len() - 12]).unwrap();

        assert!(backups.list_backups("alice").unwrap().is_empty(),
            "a truncated dump was offered as restorable");
    }

    #[test]
    fn a_sidecar_that_disagrees_with_its_dump_is_not_listed() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 2);
        let backups = RedbBackupStore::from_database(_s.database(), &store_dir);
        let mut meta = backups.create_backup("alice").unwrap();

        meta.checksum = "0".repeat(64);
        std::fs::write(
            store_dir.join("backups").join("alice").join("1.meta.json"),
            serde_json::to_vec_pretty(&meta).unwrap(),
        ).unwrap();

        assert!(backups.list_backups("alice").unwrap().is_empty(),
            "a backup whose metadata disagrees with its bytes was listed as usable");
    }

    #[test]
    fn no_backups_yet_is_an_empty_list_and_not_an_error() {
        // ENOENT is the only absence that legitimately means "none".
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 1);
        assert!(RedbBackupStore::from_database(_s.database(), &store_dir).list_backups("nobody").unwrap().is_empty());
    }

    #[test]
    fn ids_advance_and_are_never_reused() {
        // A restore names an id. Reusing one makes an old reference resolve to
        // different bytes, silently.
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 1);
        let backups = RedbBackupStore::from_database(_s.database(), &store_dir);
        assert_eq!(backups.create_backup("alice").unwrap().backup_id, 1);
        assert_eq!(backups.create_backup("alice").unwrap().backup_id, 2);
        std::fs::remove_file(store_dir.join("backups").join("alice").join("2.veldport")).unwrap();
        std::fs::remove_file(store_dir.join("backups").join("alice").join("2.meta.json")).unwrap();
        assert_eq!(backups.create_backup("alice").unwrap().backup_id, 3,
            "id 2 was reused after its files were removed");
    }

    #[test]
    fn a_restore_refuses_a_target_that_already_holds_data() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 2);
        let backups = RedbBackupStore::from_database(_s.database(), &store_dir);
        backups.create_backup("alice").unwrap();

        let target = tmp.path().join("occupied");
        // Scoped so the target is CLOSED before the restore. redb permits one opener,
        // and a real restore target is not held open by anything — leaving it open
        // would have this assert on a lock error rather than on the refusal.
        {
            let _existing = seeded_store(&target, 1);
        }
        let err = backups.restore_backup("alice", 1, &target)
            .expect_err("restoring over a populated store must refuse");
        assert!(err.to_string().contains("already holds rows"), "got: {err}");
    }
}
