//! redb implementations for the storage abstraction layer.
//!
//! Every store persists `bincode(serde)`-encoded records in redb tables, the codec
//! the RocksDB path uses, so unencrypted records are byte-identical across both
//! engines. The primary store keeps the secondary indexes RocksDB keeps in
//! `memory_index` as `index_*` tables, written in the same transaction as the memory
//! they describe (see "Secondary indexes" below). In-place migration is not built;
//! `veld migrate` moves data between engines.
//!
//! Every database opens with the process's [`StorageTuning`] page cache, and every
//! write transaction commits at its [`StorageDurability`], set by [`begin_write`]:
//! redb starts each transaction at `Immediate` whatever the database was opened with.

use anyhow::{anyhow, bail, Context, Result};
use redb::{ReadableTable, ReadableTableMetadata, TableHandle};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use super::{
    AuditLogEntry, AuditStore, BackupStore, BatchOp, BatchWriter, ColumnStore, GraphStore,
    KeyValueStore, MigrationStore, PrimaryMemoryStore, StorageFactory,
    WriteBatch as StorageWriteBatch,
};
use crate::backup::BackupMetadata;
use crate::config::{RedbFileRole, StorageBackend, StorageDurability, StorageTuning};
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, RelationshipEdge};
use crate::memory::storage::{
    haversine_metres, SearchCriteria, VectorMappingEntry, EARTH_RADIUS_METRES,
};
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
    durability: StorageDurability,
}

impl RedbBatchWriter {
    pub fn new(database: Arc<redb::Database>) -> Self {
        Self::with_durability(database, StorageTuning::process().durability)
    }

    pub fn with_durability(database: Arc<redb::Database>, durability: StorageDurability) -> Self {
        Self {
            database,
            durability,
        }
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
            let read = self
                .database
                .begin_read()
                .map_err(|e| super::redb_latch::refused(&self.database, e.into()))?;
            for name in batch.column_families() {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(name);
                read.open_table(def).map_err(|e| {
                    super::redb_latch::refused(
                        &self.database,
                        anyhow!(
                            "write_batch names table '{}', which this database does not have \
                             ({e}) — nothing was written",
                            name
                        ),
                    )
                })?;
            }
        }

        // Each table is opened once for the batch, not once per op: opening one walks the
        // table tree, and a batch of thousands of rows in one family did that per row.
        // Grouping keeps every table's ops in batch order, which is all that decides the
        // final state: tables share no keys, and the commit below applies them together.
        let mut by_table: Vec<(&str, Vec<&BatchOp>)> = Vec::new();
        for op in batch.ops() {
            let (BatchOp::Put { cf, .. } | BatchOp::Delete { cf, .. }) = op;
            let name = cf.as_deref().unwrap_or(DEFAULT_TABLE_NAME);
            match by_table.iter_mut().find(|(table, _)| *table == name) {
                Some((_, ops)) => ops.push(op),
                None => by_table.push((name, vec![op])),
            }
        }
        let txn = begin_write(&self.database, self.durability)?;
        for (name, ops) in by_table {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(name);
            let mut handle = txn.open_table(def)?;
            for op in ops {
                match op {
                    BatchOp::Put { key, value, .. } => {
                        handle.insert(key.as_slice(), value.as_slice())?;
                    }
                    BatchOp::Delete { key, .. } => {
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
    durability: StorageDurability,
}

impl RedbColumnStore {
    pub fn new(database: Arc<redb::Database>) -> Self {
        Self::with_durability(database, StorageTuning::process().durability)
    }

    pub fn with_durability(database: Arc<redb::Database>, durability: StorageDurability) -> Self {
        Self {
            database,
            durability,
        }
    }
}

impl ColumnStore for RedbColumnStore {
    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let txn = self.database.begin_read()?;
        let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
        // A table that does not exist holds no rows — None, not an error, as
        // `ColumnStore::get_cf` documents for every engine.
        match txn.open_table(def) {
            Ok(t) => Ok(t.get(key)?.map(|v| v.value().to_vec())),
            Err(_) => Ok(None),
        }
    }

    fn multi_get_cf(&self, cf: &str, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>> {
        let txn = self.database.begin_read()?;
        let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
        let Ok(table) = txn.open_table(def) else {
            return Ok(vec![None; keys.len()]);
        };
        keys.iter()
            .map(|key| Ok(table.get(*key)?.map(|v| v.value().to_vec())))
            .collect()
    }

    fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let txn = begin_write(&self.database, self.durability)?;
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
        let txn = begin_write(&self.database, self.durability)?;
        {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
            txn.open_table(def)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
        let txn = begin_write(&self.database, self.durability)?;
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
        // A redb file has one commit history, so flushing one table flushes them all.
        make_durable(&self.database, self.durability)
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        RedbBatchWriter::with_durability(Arc::clone(&self.database), self.durability)
            .write_batch(batch)
    }
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
    create_redb_database(path, &StorageTuning::process(), RedbFileRole::Auxiliary)
}

/// Open (or create) the SHARED redb database holding the global stores'
/// tables — the redb counterpart of `open_legacy_shared_db`. One file, one
/// opener: every store over it must be built FROM THIS HANDLE (`with_store`
/// constructors), never by path, or the second opener dies on the store lock.
pub fn open_shared_database(base_path: &Path) -> Result<Arc<redb::Database>> {
    create_redb_database(
        &base_path.join("shared.redb"),
        &StorageTuning::process(),
        RedbFileRole::Shared,
    )
}

/// Open (or create) a per-user GRAPH database — the redb counterpart of the
/// graph's own rocksdb directory. Separate file from the shared stores, matching
/// the rocksdb layout where the graph is its own database.
pub fn open_graph_database(graph_path: &Path) -> Result<Arc<redb::Database>> {
    create_redb_database(
        &graph_path.join("graph.redb"),
        &StorageTuning::process(),
        RedbFileRole::Graph,
    )
}

/// Open (or create) the redb file at `path`, its page cache sized for `role` unless the
/// tuning names one size for every file.
pub(crate) fn create_redb_database(
    path: &Path,
    tuning: &StorageTuning,
    role: RedbFileRole,
) -> Result<Arc<redb::Database>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut builder = redb::Builder::new();
    builder.set_cache_size(tuning.redb_cache_bytes_for(role));
    super::redb_latch::open(path, &builder)
}

/// redb's durability for a configured class.
fn redb_durability(durability: StorageDurability) -> redb::Durability {
    match durability {
        StorageDurability::Immediate => redb::Durability::Immediate,
        StorageDurability::Eventual => redb::Durability::Eventual,
    }
}

/// A write transaction that commits at `durability`. Every write transaction this
/// module opens starts here: redb begins each one at `Immediate`, so a class set
/// anywhere else would not reach the transactions that bypass it. On a handle that
/// a full disk has poisoned it is refused with the write latch.
pub(crate) fn begin_write(
    database: &redb::Database,
    durability: StorageDurability,
) -> Result<redb::WriteTransaction> {
    let mut txn = database
        .begin_write()
        .map_err(|err| super::redb_latch::refused(database, err.into()))?;
    txn.set_durability(redb_durability(durability));
    Ok(txn)
}

/// Make every earlier commit survive a power loss: the `flush` of every store here.
/// An `Eventual` commit is persisted by the next `Immediate` one, which redb performs
/// in full even when the transaction wrote nothing.
pub(crate) fn make_durable(database: &redb::Database, durability: StorageDurability) -> Result<()> {
    if durability == StorageDurability::Eventual {
        begin_write(database, StorageDurability::Immediate)?.commit()?;
    }
    Ok(())
}

// =============================================================================
// Shared redb helpers
//
// Every domain store below persists `bincode(serde)`-encoded records in a single
// `<&[u8], &[u8]>` table. Keys are the record's natural identity bytes (the UUID
// for memories/graph nodes; `{user_id}:{timestamp_nanos:020}` for audit). Reads
// are point lookups or ordered range scans; every `flush` is `make_durable`.
// =============================================================================

/// Every table here maps opaque bytes to opaque bytes.
type BytesTable = redb::TableDefinition<'static, &'static [u8], &'static [u8]>;

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
    durability: StorageDurability,
    table: BytesTable,
) -> Result<()> {
    init_tables(database, durability, [table])
}

/// [`init_table`] for several tables, in one transaction.
fn init_tables(
    database: &redb::Database,
    durability: StorageDurability,
    tables: impl IntoIterator<Item = BytesTable>,
) -> Result<()> {
    let txn = begin_write(database, durability)?;
    for table in tables {
        txn.open_table(table)?;
    }
    txn.commit()?;
    Ok(())
}

/// Point-write `key -> value` in `table` (one transaction).
fn table_put(
    database: &redb::Database,
    durability: StorageDurability,
    table: redb::TableDefinition<'static, &[u8], &[u8]>,
    key: &[u8],
    value: &[u8],
) -> Result<()> {
    let txn = begin_write(database, durability)?;
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
    durability: StorageDurability,
    table: redb::TableDefinition<'static, &[u8], &[u8]>,
    key: &[u8],
) -> Result<bool> {
    let txn = begin_write(database, durability)?;
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

// =============================================================================
// Secondary indexes
//
// One table per index RocksDB keeps in its `memory_index` family. Every row is
// written in the same write transaction as the memory it describes, so a crash
// cannot separate the two. The tables are derived: `veld migrate` counts their rows
// rather than carrying them, and a store whose schema stamp is absent or stale
// rebuilds them on open.
//
// Key layouts. `id` is the 16 memory-id bytes; `term(t)` is t's byte length as a
// u32 BE followed by its UTF-8 bytes, so one term never prefix-matches a longer
// one; `ord(x)` is the big-endian, order-preserving encoding of a number; a term
// marked blinded is the keystore's HMAC token when the store has a keystore, as
// RocksDB blinds them.
//
//   index_date              ord(created_at µs) ++ id
//   index_type              term(experience type) ++ id
//   index_importance        ord(importance f32) ++ id
//   index_entity            term(blinded lowercase entity) ++ id
//   index_tag               term(blinded lowercase tag) ++ id
//   index_episode           term(blinded episode id) ++ id
//   index_episode_sequence  term(blinded episode id) ++ sequence u32 BE ++ id
//   index_robot             term(blinded robot id) ++ id
//   index_mission           term(blinded mission id) ++ id
//   index_geo               ord(latitude f64) ++ id  ->  latitude, longitude (f64 LE)
//   index_action_type       term(blinded action type) ++ id
//   index_reward            ord(reward f32) ++ id
//   index_content_hash      term(blinded SHA-256 hex of the content) ++ id
//   index_external_id       term(blinded external id) ++ id
//   index_parent            term(blinded parent id) ++ child id
//   index_inbound_ref       term(blinded referenced id) ++ holder id
//   index_memory_rows       id  ->  every (index, key, value) row the memory owns
//   index_meta              b"schema"  ->  schema version u32 LE ++ blinder fingerprint
//
// RocksDB buckets importance and reward and keys geo by geohash; exact values are
// kept here instead, so a range read is a superset of the scan's answer by
// construction rather than by bucket arithmetic.
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum IndexKind {
    Date = 0,
    ExperienceType = 1,
    Importance = 2,
    Entity = 3,
    Tag = 4,
    Episode = 5,
    EpisodeSequence = 6,
    Robot = 7,
    Mission = 8,
    Geo = 9,
    ActionType = 10,
    Reward = 11,
    ContentHash = 12,
    ExternalId = 13,
    Parent = 14,
    InboundRef = 15,
}

impl IndexKind {
    /// In discriminant order: `ALL[kind as usize] == kind`. The discriminants are
    /// stored in `index_memory_rows`, so they never change meaning.
    const ALL: [IndexKind; 16] = [
        IndexKind::Date,
        IndexKind::ExperienceType,
        IndexKind::Importance,
        IndexKind::Entity,
        IndexKind::Tag,
        IndexKind::Episode,
        IndexKind::EpisodeSequence,
        IndexKind::Robot,
        IndexKind::Mission,
        IndexKind::Geo,
        IndexKind::ActionType,
        IndexKind::Reward,
        IndexKind::ContentHash,
        IndexKind::ExternalId,
        IndexKind::Parent,
        IndexKind::InboundRef,
    ];

    const fn table_name(self) -> &'static str {
        match self {
            IndexKind::Date => "index_date",
            IndexKind::ExperienceType => "index_type",
            IndexKind::Importance => "index_importance",
            IndexKind::Entity => "index_entity",
            IndexKind::Tag => "index_tag",
            IndexKind::Episode => "index_episode",
            IndexKind::EpisodeSequence => "index_episode_sequence",
            IndexKind::Robot => "index_robot",
            IndexKind::Mission => "index_mission",
            IndexKind::Geo => "index_geo",
            IndexKind::ActionType => "index_action_type",
            IndexKind::Reward => "index_reward",
            IndexKind::ContentHash => "index_content_hash",
            IndexKind::ExternalId => "index_external_id",
            IndexKind::Parent => "index_parent",
            IndexKind::InboundRef => "index_inbound_ref",
        }
    }

    const fn table(self) -> BytesTable {
        redb::TableDefinition::new(self.table_name())
    }

    fn from_code(code: u8) -> Option<Self> {
        Self::ALL.get(usize::from(code)).copied()
    }
}

const INDEX_ROWS_TABLE_NAME: &str = "index_memory_rows";
const INDEX_META_TABLE_NAME: &str = "index_meta";
/// The reverse map: an update or delete removes exactly the rows the previous
/// version wrote, without decoding — or being able to decode — that version.
const INDEX_ROWS_TABLE: BytesTable = redb::TableDefinition::new(INDEX_ROWS_TABLE_NAME);
const INDEX_META_TABLE: BytesTable = redb::TableDefinition::new(INDEX_META_TABLE_NAME);

/// Every table [`RedbPrimaryMemoryStore`] derives from `memories`. `veld migrate`
/// counts their rows and does not carry them, as it does RocksDB's `memory_index`.
pub(crate) const DERIVED_INDEX_TABLES: [&str; 18] = [
    IndexKind::Date.table_name(),
    IndexKind::ExperienceType.table_name(),
    IndexKind::Importance.table_name(),
    IndexKind::Entity.table_name(),
    IndexKind::Tag.table_name(),
    IndexKind::Episode.table_name(),
    IndexKind::EpisodeSequence.table_name(),
    IndexKind::Robot.table_name(),
    IndexKind::Mission.table_name(),
    IndexKind::Geo.table_name(),
    IndexKind::ActionType.table_name(),
    IndexKind::Reward.table_name(),
    IndexKind::ContentHash.table_name(),
    IndexKind::ExternalId.table_name(),
    IndexKind::Parent.table_name(),
    IndexKind::InboundRef.table_name(),
    INDEX_ROWS_TABLE_NAME,
    INDEX_META_TABLE_NAME,
];

const INDEX_SCHEMA_KEY: &[u8] = b"schema";
/// Raised whenever a key layout or the set of indexed fields changes, so a store
/// stamped by an older build rebuilds on open.
const INDEX_SCHEMA_VERSION: u32 = 1;
/// Blinded into the schema stamp: a store reopened under another blinding key — a
/// keystore added to a plaintext store, or replaced — rebuilds rather than looking
/// terms up under a key its rows were never written with.
const INDEX_BLINDER_PROBE: &str = "veld.redb.index.blinder-fingerprint";

/// A memory's index rows: (index, key) -> value.
type IndexRows = BTreeMap<(IndexKind, Vec<u8>), Vec<u8>>;
type IndexRow = ((IndexKind, Vec<u8>), Vec<u8>);
/// Row writes to one index table: (key, value) pairs borrowed from an [`IndexRows`].
type RowWrites<'a> = Vec<(&'a [u8], &'a [u8])>;

fn term_key(term: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(4 + term.len() + 16);
    key.extend_from_slice(&(term.len() as u32).to_be_bytes());
    key.extend_from_slice(term.as_bytes());
    key
}

fn with_memory_id(mut key: Vec<u8>, id: &MemoryId) -> Vec<u8> {
    key.extend_from_slice(id.0.as_bytes());
    key
}

fn ordered_i64(value: i64) -> [u8; 8] {
    ((value as u64) ^ (1 << 63)).to_be_bytes()
}

/// -0.0 is folded into 0.0 because the two compare equal: a range read starting at
/// 0.0 has to find a memory whose value is -0.0.
fn ordered_f32(value: f32) -> [u8; 4] {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    let ordered = if bits & (1 << 31) != 0 {
        !bits
    } else {
        bits | (1 << 31)
    };
    ordered.to_be_bytes()
}

fn ordered_f64(value: f64) -> [u8; 8] {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    let ordered = if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits | (1 << 63)
    };
    ordered.to_be_bytes()
}

/// The inclusive key range of every memory id under each value prefix.
fn id_span(low_prefix: &[u8], high_prefix: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut low = low_prefix.to_vec();
    low.extend_from_slice(&[0x00; 16]);
    let mut high = high_prefix.to_vec();
    high.extend_from_slice(&[0xFF; 16]);
    (low, high)
}

fn trailing_memory_id(key: &[u8]) -> Option<MemoryId> {
    let start = key.len().checked_sub(16)?;
    uuid::Uuid::from_slice(&key[start..]).ok().map(MemoryId)
}

fn content_digest(content: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(content.as_bytes()))
}

fn encode_index_rows(rows: &IndexRows) -> Result<Vec<u8>> {
    let flat: Vec<(u8, &[u8], &[u8])> = rows
        .iter()
        .map(|((kind, key), value)| (*kind as u8, key.as_slice(), value.as_slice()))
        .collect();
    encode(&flat)
}

fn decode_index_rows(bytes: &[u8]) -> Result<IndexRows> {
    let flat: Vec<(u8, Vec<u8>, Vec<u8>)> = decode(bytes)?;
    flat.into_iter()
        .map(|(code, key, value)| -> Result<IndexRow> {
            let kind = IndexKind::from_code(code)
                .ok_or_else(|| anyhow!("index_memory_rows names index {code}, which is unknown"))?;
            Ok(((kind, key), value))
        })
        .collect()
}

/// Replace the rows `id` owns with `rows`, inside `txn`. Rows unchanged between the
/// two versions are not rewritten; an empty `rows` removes the memory from every index.
fn apply_index_rows(txn: &redb::WriteTransaction, id: &MemoryId, rows: &IndexRows) -> Result<()> {
    let id_key = id.0.as_bytes().as_slice();
    let previous = {
        let owned = txn.open_table(INDEX_ROWS_TABLE)?;
        let stored = owned.get(id_key)?.map(|v| v.value().to_vec());
        match stored {
            Some(bytes) => decode_index_rows(&bytes)?,
            None => IndexRows::new(),
        }
    };
    if previous == *rows {
        return Ok(());
    }

    let mut removals: BTreeMap<IndexKind, Vec<&[u8]>> = BTreeMap::new();
    let mut inserts: BTreeMap<IndexKind, RowWrites<'_>> = BTreeMap::new();
    for ((kind, key), value) in &previous {
        if rows.get(&(*kind, key.clone())) != Some(value) {
            removals.entry(*kind).or_default().push(key.as_slice());
        }
    }
    for ((kind, key), value) in rows {
        if previous.get(&(*kind, key.clone())) != Some(value) {
            inserts
                .entry(*kind)
                .or_default()
                .push((key.as_slice(), value.as_slice()));
        }
    }
    for kind in IndexKind::ALL {
        let doomed = removals.remove(&kind).unwrap_or_default();
        let added = inserts.remove(&kind).unwrap_or_default();
        if doomed.is_empty() && added.is_empty() {
            continue;
        }
        let mut table = txn.open_table(kind.table())?;
        // Removals first: a row whose value changed is both removed and re-added.
        for key in doomed {
            table.remove(key)?;
        }
        for (key, value) in added {
            table.insert(key, value)?;
        }
    }

    let mut owned = txn.open_table(INDEX_ROWS_TABLE)?;
    if rows.is_empty() {
        owned.remove(id_key)?;
    } else {
        owned.insert(id_key, encode_index_rows(rows)?.as_slice())?;
    }
    Ok(())
}

fn ids_with_prefix(
    txn: &redb::ReadTransaction,
    kind: IndexKind,
    prefix: &[u8],
) -> Result<Vec<MemoryId>> {
    let table = txn.open_table(kind.table())?;
    let mut ids = Vec::new();
    for row in table.range(prefix..)? {
        let (key, _) = row?;
        let key = key.value();
        if !key.starts_with(prefix) {
            break;
        }
        ids.extend(trailing_memory_id(key));
    }
    Ok(ids)
}

fn ids_in_span(
    txn: &redb::ReadTransaction,
    kind: IndexKind,
    (low, high): (Vec<u8>, Vec<u8>),
) -> Result<Vec<MemoryId>> {
    let table = txn.open_table(kind.table())?;
    let mut ids = Vec::new();
    for row in table.range(low.as_slice()..=high.as_slice())? {
        let (key, _) = row?;
        ids.extend(trailing_memory_id(key.value()));
    }
    Ok(ids)
}

/// Value-ordered candidates for `min..=max`; none when the range is empty or a bound
/// is NaN, which is when the scan's `>= min && <= max` matches nothing either.
fn ids_in_f32_range(
    txn: &redb::ReadTransaction,
    kind: IndexKind,
    min: f32,
    max: f32,
) -> Result<Vec<MemoryId>> {
    if min.is_nan() || max.is_nan() || min > max {
        return Ok(Vec::new());
    }
    ids_in_span(txn, kind, id_span(&ordered_f32(min), &ordered_f32(max)))
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

/// Rollback guard: refuse a keystore older than the last generation this store
/// saw; advance the sentinel otherwise.
fn check_keystore_generation(
    database: &redb::Database,
    durability: StorageDurability,
    generation: u64,
) -> Result<()> {
    let stored = table_get(database, DEFAULT_TABLE, KEYSTORE_GENERATION_KEY)?
        .and_then(|b| <[u8; 8]>::try_from(b.as_slice()).ok())
        .map(u64::from_le_bytes)
        .unwrap_or(0);
    if let Some(next) = crate::keystore::keystore_generation_sentinel(stored, generation)? {
        table_put(
            database,
            durability,
            DEFAULT_TABLE,
            KEYSTORE_GENERATION_KEY,
            &next.to_le_bytes(),
        )?;
    }
    Ok(())
}

/// The keystore beside a tenant's `storage.redb`. RocksDB keeps its own inside the
/// RocksDB directory, `storage/keystore.json`.
pub const KEYSTORE_FILE: &str = "keystore.json";

/// Rollback sentinel for the keystore generation, in the default table.
const KEYSTORE_GENERATION_KEY: &[u8] = crate::keystore::GENERATION_SENTINEL_KEY;

pub struct RedbPrimaryMemoryStore {
    database: Arc<redb::Database>,
    durability: StorageDurability,
    /// Record-level encryption (encryption-v2): with a keystore, every memory record
    /// is sealed in the same envelope RocksDB writes, bound to its memory id.
    cryptors: Option<crate::keystore::RecordCryptors>,
    /// With a keystore, exact-match index terms are stored as HMAC tokens, as RocksDB
    /// stores them, so the index tables do not hold what the sealed records hide.
    index_blinder: Option<crate::keystore::IndexBlinder>,
    /// Whether `index_meta` holds this store's schema stamp. A store opened empty is
    /// left without one — a row in it would make an untouched store look populated
    /// to `veld migrate` and `import_portable` — and the first memory write adds it.
    index_stamped: AtomicBool,
    /// See [`Self::undecodable_records_skipped`].
    undecodable_skipped: AtomicU64,
}

/// Content in the retired field-level format, opened. redb has no read-path
/// rewrite: the record is re-sealed by its next write.
fn open_legacy_field(mut memory: Memory) -> Result<Memory> {
    if let Some(plain) =
        crate::legacy_field_encryption::open_legacy_content(&memory.experience.content)?
    {
        memory.experience.content = plain;
    }
    Ok(memory)
}

impl RedbPrimaryMemoryStore {
    /// Whether this store seals its records (a keystore is unsealed).
    pub fn encrypts_records(&self) -> bool {
        self.cryptors.is_some()
    }

    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_tuning(path, StorageTuning::process())
    }

    /// [`Self::open`] with an explicit durability class and page cache in place of
    /// the process's.
    pub fn open_with_tuning(path: &Path, tuning: StorageTuning) -> Result<Self> {
        let database_path = path.join("storage.redb");
        let database = create_redb_database(&database_path, &tuning, RedbFileRole::Memories)?;
        let durability = tuning.durability;
        // `kv` with the rest: RocksDB opens its `kv` family with the database, and a
        // write_batch naming a table that does not exist yet is refused here.
        init_tables(
            &database,
            durability,
            [MEMORIES_TABLE, VMAPPINGS_TABLE, DEFAULT_TABLE, KV_TABLE]
                .into_iter()
                .chain(IndexKind::ALL.map(IndexKind::table))
                .chain([INDEX_ROWS_TABLE, INDEX_META_TABLE]),
        )?;
        let (cryptors, index_blinder) =
            match crate::keystore::open_record_keys(&path.join(KEYSTORE_FILE))? {
                Some(keys) => {
                    check_keystore_generation(&database, durability, keys.generation)?;
                    (Some(keys.cryptors), Some(keys.blinder))
                }
                None => (None, None),
            };
        crate::legacy_field_encryption::check_config(cryptors.is_some())?;
        let store = Self {
            database,
            durability,
            cryptors,
            index_blinder,
            index_stamped: AtomicBool::new(false),
            undecodable_skipped: AtomicU64::new(0),
        };
        store.ensure_indexes_current()?;
        Ok(store)
    }

    /// Memory records an enumeration has skipped because they would not open, since
    /// this store opened.
    pub fn undecodable_records_skipped(&self) -> u64 {
        self.undecodable_skipped.load(Ordering::Relaxed)
    }

    /// A record an enumeration read at `key`, opened; `None`, logged and counted, when
    /// it will not open. Skipped rather than failing the enumeration, as RocksDB skips
    /// it: one bad record would otherwise make every listing of the store fail, and
    /// `cleanup_corrupted` is the path that removes it.
    fn open_or_skip(&self, key: &[u8], bytes: &[u8], read: &'static str) -> Option<Memory> {
        match self.open_record(key, bytes) {
            Ok(memory) => Some(memory),
            Err(e) => {
                let skipped = self.undecodable_skipped.fetch_add(1, Ordering::Relaxed) + 1;
                let memory_id = uuid::Uuid::from_slice(key)
                    .map(|id| id.to_string())
                    .unwrap_or_else(|_| hex::encode(key));
                tracing::warn!(
                    memory_id = %memory_id,
                    read,
                    skipped_since_open = skipped,
                    error = %e,
                    "memory record does not decode; skipped"
                );
                None
            }
        }
    }

    /// Serialize a memory record, sealed when the store has a keystore.
    fn seal(&self, memory: &Memory) -> Result<Vec<u8>> {
        let plaintext = encode(memory)?;
        match &self.cryptors {
            Some(cryptors) => cryptors
                .active()
                .encrypt_record(&plaintext, memory.id.0.as_bytes())
                .context("Failed to encrypt memory record"),
            None => Ok(plaintext),
        }
    }

    /// Decode a memory record read at `key`. A sealed record is opened with its
    /// epoch's key; one that cannot be opened is an error, never a plaintext read.
    fn open_record(&self, key: &[u8], bytes: &[u8]) -> Result<Memory> {
        if !crate::keystore::is_encrypted_record(bytes) {
            return open_legacy_field(decode::<Memory>(bytes)?);
        }
        let cryptors = self.cryptors.as_ref().ok_or_else(|| {
            anyhow!("encountered an encrypted record but no keystore/key is configured")
        })?;
        let epoch = crate::keystore::record_epoch(bytes).unwrap_or(0);
        let cryptor = cryptors.for_epoch(epoch).ok_or_else(|| {
            anyhow!(
                "record epoch {epoch} has no DEK in the keystore (active epoch {}); \
                 the keystore may have been replaced or the epoch's key pruned",
                cryptors.active_epoch()
            )
        })?;
        let plaintext = cryptor
            .decrypt_record(bytes, key)
            .context("Failed to decrypt memory record")?;
        open_legacy_field(decode::<Memory>(&plaintext)?)
    }

    /// The live database handle, for stores that must share it rather than open the
    /// path again — see RedbBackupStore. redb permits one opener per file.
    pub fn database(&self) -> Arc<redb::Database> {
        Arc::clone(&self.database)
    }

    /// The class this store's write transactions commit at.
    pub fn durability(&self) -> StorageDurability {
        self.durability
    }

    /// Decode every stored memory, forgotten ones included: `get_all` and the `search`
    /// variants no index serves filter them out. A record that will not open is skipped
    /// ([`Self::open_or_skip`]).
    fn scan_all_memories(&self) -> Result<Vec<Memory>> {
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        let mut memories = Vec::new();
        for entry in table.iter()? {
            let (key, value) = entry?;
            memories.extend(self.open_or_skip(key.value(), value.value(), "memory scan"));
        }
        Ok(memories)
    }

    fn blinded_term_key(&self, term: &str) -> Vec<u8> {
        match &self.index_blinder {
            Some(blinder) => term_key(&blinder.blind(term)),
            None => term_key(term),
        }
    }

    /// The rows `memory`, stored under `id`, owns in every index.
    fn index_rows(&self, id: &MemoryId, memory: &Memory) -> IndexRows {
        let experience = &memory.experience;
        let mut rows = IndexRows::new();
        let mut add = |kind: IndexKind, key: Vec<u8>, value: Vec<u8>| {
            rows.insert((kind, with_memory_id(key, id)), value);
        };

        add(
            IndexKind::Date,
            ordered_i64(memory.created_at.timestamp_micros()).to_vec(),
            Vec::new(),
        );
        add(
            IndexKind::ExperienceType,
            term_key(&format!("{:?}", experience.experience_type)),
            Vec::new(),
        );
        add(
            IndexKind::Importance,
            ordered_f32(memory.importance()).to_vec(),
            Vec::new(),
        );
        for entity in &experience.entities {
            add(
                IndexKind::Entity,
                self.blinded_term_key(&entity.to_lowercase()),
                Vec::new(),
            );
        }
        for tag in &experience.tags {
            add(
                IndexKind::Tag,
                self.blinded_term_key(&tag.to_lowercase()),
                Vec::new(),
            );
        }
        if let Some(episode_id) = memory_episode_id(memory) {
            add(
                IndexKind::Episode,
                self.blinded_term_key(episode_id),
                Vec::new(),
            );
            if let Some(sequence) = memory_sequence(memory) {
                let mut key = self.blinded_term_key(episode_id);
                key.extend_from_slice(&sequence.to_be_bytes());
                add(IndexKind::EpisodeSequence, key, Vec::new());
            }
        }
        if let Some(robot_id) = experience.resolved_robot_id() {
            add(
                IndexKind::Robot,
                self.blinded_term_key(robot_id),
                Vec::new(),
            );
        }
        if let Some(mission_id) = experience.resolved_mission_id() {
            add(
                IndexKind::Mission,
                self.blinded_term_key(mission_id),
                Vec::new(),
            );
        }
        if let Some((lat, lon, _alt)) = experience.resolved_geo() {
            let mut coordinates = Vec::with_capacity(16);
            coordinates.extend_from_slice(&lat.to_le_bytes());
            coordinates.extend_from_slice(&lon.to_le_bytes());
            add(IndexKind::Geo, ordered_f64(lat).to_vec(), coordinates);
        }
        if let Some(action_type) = experience.action_type.as_deref() {
            add(
                IndexKind::ActionType,
                self.blinded_term_key(action_type),
                Vec::new(),
            );
        }
        if let Some(reward) = experience.reward {
            add(IndexKind::Reward, ordered_f32(reward).to_vec(), Vec::new());
        }
        add(
            IndexKind::ContentHash,
            self.blinded_term_key(&content_digest(&experience.content)),
            Vec::new(),
        );
        if let Some(external_id) = memory.external_id.as_deref() {
            add(
                IndexKind::ExternalId,
                self.blinded_term_key(external_id),
                Vec::new(),
            );
        }
        if let Some(parent_id) = &memory.parent_id {
            add(
                IndexKind::Parent,
                self.blinded_term_key(&parent_id.0.to_string()),
                Vec::new(),
            );
        }
        for target in memory.forward_ref_targets() {
            add(
                IndexKind::InboundRef,
                self.blinded_term_key(&target.0.to_string()),
                Vec::new(),
            );
        }
        rows
    }

    fn index_schema_stamp(&self) -> Vec<u8> {
        let mut stamp = INDEX_SCHEMA_VERSION.to_le_bytes().to_vec();
        if let Some(blinder) = &self.index_blinder {
            stamp.extend_from_slice(blinder.blind(INDEX_BLINDER_PROBE).as_bytes());
        }
        stamp
    }

    /// Run `body` in one write transaction and commit it, stamping the schema marker
    /// in the same transaction when the store has none yet. Every write to `memories`
    /// goes through here, so the indexes cannot fall behind the memories they describe.
    fn commit_indexed<T>(
        &self,
        body: impl FnOnce(&redb::WriteTransaction) -> Result<T>,
    ) -> Result<T> {
        let txn = begin_write(&self.database, self.durability)?;
        let out = body(&txn)?;
        let stamping = !self.index_stamped.load(Ordering::Acquire);
        if stamping {
            txn.open_table(INDEX_META_TABLE)?
                .insert(INDEX_SCHEMA_KEY, self.index_schema_stamp().as_slice())?;
        }
        txn.commit()?;
        if stamping {
            self.index_stamped.store(true, Ordering::Release);
        }
        Ok(out)
    }

    /// Seal and index `memory` before the write transaction that stores it begins.
    fn prepare(&self, memory: &Memory) -> Result<PreparedMemory> {
        Ok(PreparedMemory {
            id: memory.id.clone(),
            record: self.seal(memory)?,
            rows: self.index_rows(&memory.id, memory),
        })
    }

    /// Rebuild when the schema stamp is missing or names another layout or blinding
    /// key: a store written before the indexes existed, or reopened under a new keystore.
    fn ensure_indexes_current(&self) -> Result<()> {
        let stamp = self.index_schema_stamp();
        let stored = table_get(&self.database, INDEX_META_TABLE, INDEX_SCHEMA_KEY)?;
        if stored.as_deref() == Some(stamp.as_slice()) {
            self.index_stamped.store(true, Ordering::Release);
            return Ok(());
        }
        let indexed = self.rebuild_index()?;
        if indexed > 0 {
            tracing::info!(
                indexed,
                "redb secondary indexes rebuilt from the memories table"
            );
        }
        Ok(())
    }

    /// Clear every derived table and re-derive it from `memories`, in one write
    /// transaction. Idempotent; records are never touched. Returns the memories indexed.
    ///
    /// A record that cannot be opened is left out and logged: it is unreadable to
    /// every other path too, and the rebuild is how it gets indexed once its key
    /// is back.
    pub fn rebuild_index(&self) -> Result<usize> {
        let stamp = self.index_schema_stamp();
        let txn = begin_write(&self.database, self.durability)?;
        let (indexed, unreadable) = {
            let mut index_tables = Vec::with_capacity(IndexKind::ALL.len());
            for kind in IndexKind::ALL {
                let mut table = txn.open_table(kind.table())?;
                table.retain(|_, _| false)?;
                index_tables.push(table);
            }
            let mut owned = txn.open_table(INDEX_ROWS_TABLE)?;
            owned.retain(|_, _| false)?;

            let memories = txn.open_table(MEMORIES_TABLE)?;
            let mut indexed = 0usize;
            let mut unreadable = 0usize;
            for entry in memories.iter()? {
                let (key, value) = entry?;
                let key = key.value();
                if key.len() != 16 {
                    continue;
                }
                let Some(id) = trailing_memory_id(key) else {
                    continue;
                };
                match self.open_record(key, value.value()) {
                    Ok(memory) => {
                        let rows = self.index_rows(&id, &memory);
                        for ((kind, row_key), row_value) in &rows {
                            index_tables[*kind as usize]
                                .insert(row_key.as_slice(), row_value.as_slice())?;
                        }
                        owned.insert(key, encode_index_rows(&rows)?.as_slice())?;
                        indexed += 1;
                    }
                    Err(e) => {
                        unreadable += 1;
                        tracing::warn!(memory_id = %id.0, error = %e,
                            "memory record could not be opened; left out of the redb secondary indexes");
                    }
                }
            }

            let mut meta = txn.open_table(INDEX_META_TABLE)?;
            if indexed + unreadable > 0 {
                meta.insert(INDEX_SCHEMA_KEY, stamp.as_slice())?;
            } else {
                meta.remove(INDEX_SCHEMA_KEY)?;
            }
            (indexed, unreadable)
        };
        txn.commit()?;
        self.index_stamped
            .store(indexed + unreadable > 0, Ordering::Release);
        Ok(indexed)
    }

    /// The ids an index narrows `criteria` to, in the order `search` returns them;
    /// `None` for the variants RocksDB also answers by scanning.
    fn index_candidates(
        &self,
        txn: &redb::ReadTransaction,
        criteria: &SearchCriteria,
    ) -> Result<Option<Vec<MemoryId>>> {
        let ids = match criteria {
            SearchCriteria::ByDate { start, end } => {
                if start > end {
                    Vec::new()
                } else {
                    ids_in_span(
                        txn,
                        IndexKind::Date,
                        id_span(
                            &ordered_i64(start.timestamp_micros()),
                            &ordered_i64(end.timestamp_micros()),
                        ),
                    )?
                }
            }
            SearchCriteria::ByType(experience_type) => ids_with_prefix(
                txn,
                IndexKind::ExperienceType,
                &term_key(&format!("{experience_type:?}")),
            )?,
            SearchCriteria::ByImportance { min, max } => {
                ids_in_f32_range(txn, IndexKind::Importance, *min, *max)?
            }
            SearchCriteria::ByEntity(entity) => ids_with_prefix(
                txn,
                IndexKind::Entity,
                &self.blinded_term_key(&entity.to_lowercase()),
            )?,
            SearchCriteria::ByTags(tags) => {
                let mut union = BTreeSet::new();
                for tag in tags {
                    union.extend(ids_with_prefix(
                        txn,
                        IndexKind::Tag,
                        &self.blinded_term_key(&tag.to_lowercase()),
                    )?);
                }
                union.into_iter().collect()
            }
            SearchCriteria::ByEpisode(episode_id) => {
                ids_with_prefix(txn, IndexKind::Episode, &self.blinded_term_key(episode_id))?
            }
            SearchCriteria::ByEpisodeSequence {
                episode_id,
                min_sequence,
                max_sequence,
            } => {
                let low = min_sequence.unwrap_or(u32::MIN);
                let high = max_sequence.unwrap_or(u32::MAX);
                if low > high {
                    Vec::new()
                } else {
                    let prefix = self.blinded_term_key(episode_id);
                    let mut low_prefix = prefix.clone();
                    low_prefix.extend_from_slice(&low.to_be_bytes());
                    let mut high_prefix = prefix;
                    high_prefix.extend_from_slice(&high.to_be_bytes());
                    ids_in_span(
                        txn,
                        IndexKind::EpisodeSequence,
                        id_span(&low_prefix, &high_prefix),
                    )?
                }
            }
            SearchCriteria::ByRobot(robot_id) => {
                ids_with_prefix(txn, IndexKind::Robot, &self.blinded_term_key(robot_id))?
            }
            SearchCriteria::ByMission(mission_id) => {
                ids_with_prefix(txn, IndexKind::Mission, &self.blinded_term_key(mission_id))?
            }
            SearchCriteria::ByLocation {
                lat,
                lon,
                radius_meters,
            } => location_candidates(txn, *lat, *lon, *radius_meters)?,
            SearchCriteria::ByActionType(action_type) => ids_with_prefix(
                txn,
                IndexKind::ActionType,
                &self.blinded_term_key(action_type),
            )?,
            SearchCriteria::ByReward { min, max } => {
                ids_in_f32_range(txn, IndexKind::Reward, *min, *max)?
            }
            SearchCriteria::ByParent(parent_id) => ids_with_prefix(
                txn,
                IndexKind::Parent,
                &self.blinded_term_key(&parent_id.0.to_string()),
            )?,
            SearchCriteria::RootsOnly => return Ok(None),
            SearchCriteria::Combined(criterias) if criterias.is_empty() => Vec::new(),
            SearchCriteria::Combined(criterias) => {
                let mut served: Option<BTreeSet<MemoryId>> = None;
                for part in criterias {
                    if let Some(ids) = self.index_candidates(txn, part)? {
                        let ids: BTreeSet<MemoryId> = ids.into_iter().collect();
                        served = Some(match served {
                            None => ids,
                            Some(previous) => previous.intersection(&ids).cloned().collect(),
                        });
                    }
                }
                match served {
                    Some(ids) => ids.into_iter().collect(),
                    None => return Ok(None),
                }
            }
        };
        Ok(Some(ids))
    }

    /// Whether a memory still names `target` among its references, read from the
    /// inbound-reference index and confirmed against each holder's record.
    fn memory_cites(&self, target: &MemoryId) -> Result<bool> {
        let txn = self.database.begin_read()?;
        let memories = txn.open_table(MEMORIES_TABLE)?;
        let prefix = self.blinded_term_key(&target.0.to_string());
        for holder in ids_with_prefix(&txn, IndexKind::InboundRef, &prefix)? {
            let key = holder.0.as_bytes();
            if let Some(bytes) = memories.get(key.as_slice())? {
                let memory = self.open_record(key, bytes.value())?;
                if memory.forward_ref_targets().contains(target) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// The memories an index row names, in the row order, each confirmed by `keep`
    /// against its own record. A row whose memory is gone is skipped: the memory is
    /// the truth, and the row can only have been left by a write that bypassed this
    /// store.
    fn indexed_memories(
        &self,
        kind: IndexKind,
        prefix: &[u8],
        mut keep: impl FnMut(&Memory) -> bool,
        limit: Option<usize>,
    ) -> Result<Vec<Memory>> {
        let txn = self.database.begin_read()?;
        let memories = txn.open_table(MEMORIES_TABLE)?;
        let mut found = Vec::new();
        for id in ids_with_prefix(&txn, kind, prefix)? {
            let key = id.0.as_bytes();
            let Some(bytes) = memories.get(key.as_slice())? else {
                continue;
            };
            let memory = self.open_record(key, bytes.value())?;
            if keep(&memory) {
                found.push(memory);
                if limit.is_some_and(|max| found.len() >= max) {
                    break;
                }
            }
        }
        Ok(found)
    }

    /// Soft-delete every readable, unforgotten memory `select` picks. The scan and the
    /// writes share one write transaction: RocksDB writes the pass as one batch, so a
    /// partial pass is a state neither engine can produce, and no write can land
    /// between a record being read and its flagged copy replacing it.
    ///
    /// The two metadata keys are RocksDB's exact names; `is_forgotten` and every
    /// filter on it read `forgotten`.
    fn mark_forgotten_where(
        &self,
        pass: &str,
        select: impl Fn(&Memory) -> bool,
    ) -> Result<Vec<MemoryId>> {
        let now = chrono::Utc::now().to_rfc3339();
        self.commit_indexed(|txn| {
            let candidates: Vec<(Vec<u8>, Vec<u8>)> = {
                let table = txn.open_table(MEMORIES_TABLE)?;
                let mut rows = Vec::new();
                for row in table.iter()? {
                    let (k, v) = row?;
                    if k.value().len() == 16 {
                        rows.push((k.value().to_vec(), v.value().to_vec()));
                    }
                }
                rows
            };
            let mut flagged = Vec::new();
            for (key, value) in candidates {
                let mut memory = match self.open_record(&key, &value) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(error = %e, pass,
                            "skipping a record that failed to deserialize while marking memories forgotten");
                        continue;
                    }
                };
                if memory.is_forgotten() || !select(&memory) {
                    continue;
                }
                memory
                    .experience
                    .metadata
                    .insert("forgotten".into(), "true".into());
                memory
                    .experience
                    .metadata
                    .insert("forgotten_at".into(), now.clone());
                put_prepared(txn, &self.prepare(&memory)?)?;
                flagged.push(memory.id);
            }
            Ok(flagged)
        })
    }
}

/// A memory sealed and indexed, ready for its table writes. Built before the write
/// transaction where the caller can, since that transaction excludes every other writer.
struct PreparedMemory {
    id: MemoryId,
    record: Vec<u8>,
    rows: IndexRows,
}

fn put_prepared(txn: &redb::WriteTransaction, prepared: &PreparedMemory) -> Result<()> {
    txn.open_table(MEMORIES_TABLE)?.insert(
        prepared.id.0.as_bytes().as_slice(),
        prepared.record.as_slice(),
    )?;
    apply_index_rows(txn, &prepared.id, &prepared.rows)
}

/// Merge `vector_ids` into the TEXT modality of `id`'s mapping inside `txn`,
/// keeping its other modalities — `MemoryStorage::update_modality_vectors` for Text.
fn merge_text_vectors(
    txn: &redb::WriteTransaction,
    id: &MemoryId,
    vector_ids: Vec<u32>,
) -> Result<()> {
    let key = id.0.as_bytes().as_slice();
    let mut mappings = txn.open_table(VMAPPINGS_TABLE)?;
    let stored = mappings.get(key)?.map(|v| v.value().to_vec());
    let mut entry = match stored {
        Some(bytes) => decode::<VectorMappingEntry>(&bytes)?,
        None => VectorMappingEntry::default(),
    };
    entry.add_modality(crate::memory::storage::Modality::Text, vector_ids);
    mappings.insert(key, encode(&entry)?.as_slice())?;
    Ok(())
}

type KeySpan = (Bound<Vec<u8>>, Bound<Vec<u8>>);

/// Candidates for a radius search, each already within the radius by the same
/// haversine `SearchCriteria::matches` applies to the coordinates stored in its row.
///
/// The latitude band is a superset of the circle because great-circle distance is
/// never less than `R·|Δφ|` while both latitudes lie in [-90°, 90°]. Rows outside that
/// range break the bound, so they are always read, and a centre outside it reads
/// the whole index.
fn location_candidates(
    txn: &redb::ReadTransaction,
    lat: f64,
    lon: f64,
    radius_meters: f64,
) -> Result<Vec<MemoryId>> {
    if !lat.is_finite() || !lon.is_finite() || radius_meters.is_nan() || radius_meters < 0.0 {
        return Ok(Vec::new());
    }
    let table = txn.open_table(IndexKind::Geo.table())?;
    let half_band_degrees = (radius_meters / EARTH_RADIUS_METRES).to_degrees();
    let mut spans: Vec<KeySpan> = Vec::new();
    if (-90.0..=90.0).contains(&lat) && half_band_degrees < 180.0 {
        let slack = 1e-9 * half_band_degrees.max(1.0);
        let low = (lat - half_band_degrees - slack).max(-90.0);
        let high = (lat + half_band_degrees + slack).min(90.0);
        let (low_key, high_key) = id_span(&ordered_f64(low), &ordered_f64(high));
        spans.push((Bound::Included(low_key), Bound::Included(high_key)));
        // Every key below this holds a latitude under -90°, every key above the next
        // one a latitude over 90°.
        let south_pole = ordered_f64(-90.0).to_vec();
        let mut north_pole = ordered_f64(90.0).to_vec();
        north_pole.extend_from_slice(&[0xFF; 16]);
        spans.push((Bound::Unbounded, Bound::Excluded(south_pole)));
        spans.push((Bound::Excluded(north_pole), Bound::Unbounded));
    } else {
        spans.push((Bound::Unbounded, Bound::Unbounded));
    }

    let mut within = BTreeSet::new();
    for (low, high) in &spans {
        let span = (
            low.as_ref().map(Vec::as_slice),
            high.as_ref().map(Vec::as_slice),
        );
        for row in table.range::<&[u8]>(span)? {
            let (key, value) = row?;
            let coordinates = value.value();
            if coordinates.len() != 16 {
                continue;
            }
            let row_lat = f64::from_le_bytes(coordinates[..8].try_into().expect("8 bytes"));
            let row_lon = f64::from_le_bytes(coordinates[8..].try_into().expect("8 bytes"));
            if haversine_metres(lat, lon, row_lat, row_lon) <= radius_meters {
                within.extend(trailing_memory_id(key.value()));
            }
        }
    }
    Ok(within.into_iter().collect())
}

impl PrimaryMemoryStore for RedbPrimaryMemoryStore {
    // Production runs this store whenever VELD_STORAGE_BACKEND=redb (config::effective_
    // storage_backend_for_current_build); RocksDB stays the default under ADR 0007.
    //
    // These four are the same operations this trait carries under other names. On
    // redb `get_memory` is a plain table lookup returning Option, so composing `get`
    // from it is exact — unlike the rocksdb adapter, where `get_memory` recovers
    // not-found by matching an error STRING and composing the other way round would
    // launder real failures into `Ok(None)`. Same two names, opposite safe direction,
    // per backend.
    fn get(&self, id: &MemoryId) -> Result<Memory> {
        self.get_memory(id)?
            .ok_or_else(|| anyhow!("Memory not found: {}", id.0))
    }
    /// One read transaction for the whole batch. A transaction or table that will not open
    /// is answered id by id through `get`, so each id still carries the error `get` gives.
    fn get_many(&self, ids: &[MemoryId]) -> Vec<Result<Memory>> {
        let table = self
            .database
            .begin_read()
            .map_err(anyhow::Error::from)
            .and_then(|txn| Ok(txn.open_table(MEMORIES_TABLE)?));
        let Ok(table) = table else {
            return ids.iter().map(|id| self.get(id)).collect();
        };
        ids.iter()
            .map(|id| {
                let key = id.0.as_bytes().as_slice();
                match table.get(key)? {
                    Some(bytes) => self.open_record(key, bytes.value()),
                    None => Err(anyhow!("Memory not found: {}", id.0)),
                }
            })
            .collect()
    }
    fn store(&self, memory: &Memory) -> Result<()> {
        self.put_memory(memory)
    }
    fn update(&self, memory: &Memory) -> Result<()> {
        self.update_memory(memory)
    }
    fn update_many(&self, memories: &[Memory]) -> Result<()> {
        if memories.is_empty() {
            return Ok(());
        }
        let prepared = memories
            .iter()
            .map(|memory| self.prepare(memory))
            .collect::<Result<Vec<_>>>()?;
        self.commit_indexed(|txn| {
            for record in &prepared {
                put_prepared(txn, record)?;
            }
            Ok(())
        })
    }
    fn delete(&self, id: &MemoryId) -> Result<()> {
        self.delete_memory(id)
    }
    fn get_all(&self) -> Result<Vec<Memory>> {
        Ok(self
            .scan_all_memories()?
            .into_iter()
            .filter(|memory| !memory.is_forgotten())
            .collect())
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
        table_put(
            &self.database,
            self.durability,
            DEFAULT_TABLE,
            RETRIEVAL_KEY,
            &next.to_le_bytes(),
        )?;
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
            match self.open_record(k.value(), v.value()) {
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
        Ok(self
            .indexed_memories(
                IndexKind::ExternalId,
                &self.blinded_term_key(external_id),
                |m| m.external_id.as_deref() == Some(external_id),
                Some(1),
            )?
            .pop())
    }
    fn get_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>> {
        self.indexed_memories(
            IndexKind::Parent,
            &self.blinded_term_key(&parent_id.0.to_string()),
            |m| m.parent_id.as_ref() == Some(parent_id),
            None,
        )
    }
    fn get_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>> {
        // The 100-iteration bound is rocksdb's and is carried over deliberately: a
        // parent_id cycle is representable in the data, and without a bound this walk
        // does not terminate. Dropping a guard because the loop "obviously" ends at a
        // root is how the cycle case becomes a hang instead of a wrong answer.
        let mut ancestors = Vec::new();
        let mut parent_id = self.get(memory_id)?.parent_id;
        for _ in 0..100 {
            let Some(id) = parent_id else {
                break;
            };
            let parent = self.get(&id)?;
            parent_id = parent.parent_id.clone();
            ancestors.push(parent);
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
        self.mark_forgotten_where("forget-by-age", |m| m.created_at < cutoff)
    }
    fn save_interference_records(
        &self,
        memory_id: &str,
        records: &[crate::memory::replay::InterferenceRecord],
    ) -> Result<()> {
        // serde_json, not bincode — matching rocksdb, because these bytes have to be
        // readable by whichever backend is live after a cutover.
        let key = format!("interference:{memory_id}");
        let value =
            serde_json::to_vec(records).context("Failed to serialize interference records")?;
        table_put(
            &self.database,
            self.durability,
            DEFAULT_TABLE,
            key.as_bytes(),
            &value,
        )
    }
    fn delete_interference_records(&self, memory_id: &str) -> Result<()> {
        let key = format!("interference:{memory_id}");
        table_remove(
            &self.database,
            self.durability,
            DEFAULT_TABLE,
            key.as_bytes(),
        )?;
        Ok(())
    }
    fn clear_all_interference_records(&self) -> Result<usize> {
        // The prefix is `interference` with NO colon, which is rocksdb's, and it
        // therefore also removes `interference_meta:total` — the event counter — not
        // just the per-memory records. Replicated deliberately: whether that is
        // intended or a slip, the two backends must agree, and a divergence here
        // would show up as a counter that survives a clear on one engine only.
        let txn = begin_write(&self.database, self.durability)?;
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
        let mut foreign = 0usize;
        let deleted = self.commit_indexed(|txn| {
            let doomed: Vec<Vec<u8>> = txn
                .open_table(MEMORIES_TABLE)?
                .iter()?
                .filter_map(|r| r.ok())
                .filter_map(|(k, v)| {
                    let key = k.value().to_vec();
                    if key.len() != 16 {
                        foreign += 1;
                        return None;
                    }
                    // A sealed record that will not open points at the key, not the
                    // record; deleting it would destroy data a restored keystore opens.
                    if crate::keystore::is_encrypted_record(v.value()) {
                        return None;
                    }
                    decode::<Memory>(v.value()).err().map(|_| key)
                })
                .collect();
            for key in &doomed {
                txn.open_table(MEMORIES_TABLE)?.remove(key.as_slice())?;
                if let Some(id) = trailing_memory_id(key) {
                    apply_index_rows(txn, &id, &IndexRows::new())?;
                }
            }
            Ok(doomed.len())
        })?;
        if foreign > 0 {
            tracing::error!(
                foreign,
                "MEMORIES_TABLE holds keys that are not 16-byte memory ids — left in place. \
                 Something is writing non-memory rows into the memories table, which is the \
                 invariant this cleanup relies on to be safe without an allowlist."
            );
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
            match self.open_record(k.value(), v.value()) {
                Ok(_) => already_current += 1,
                Err(_) => failed += 1,
            }
        }
        Ok((0, already_current, failed))
    }

    fn run_merge_pass(&self, user_id: &str, limit: usize) -> Result<usize> {
        crate::memory::merge::run_merge_pass(self, user_id, limit)
    }
    fn gc_merge_tombstones(&self, user_id: &str) -> Result<usize> {
        crate::memory::merge::gc_merge_tombstones(self, user_id)
    }
    fn has_any_inbound_refs(&self, target: &MemoryId, user_id: &str) -> Result<bool> {
        if self.memory_cites(target)? {
            return Ok(true);
        }
        crate::memory::merge::fact_cites_memory(self, target, user_id)
    }

    fn scan_prefix_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        // Ordered range scan over the memories table, stopping at the first key
        // that leaves the prefix. redb ranges are ordered, so this terminates
        // early rather than reading the table. The length filter is get_all_ids':
        // a stray key is not a memory id on either engine.
        let txn = self.database.begin_read()?;
        let table = txn.open_table(MEMORIES_TABLE)?;
        let mut keys = Vec::new();
        for row in table.range(prefix..)? {
            let (k, _) = row?;
            let key = k.value();
            if !key.starts_with(prefix) {
                break;
            }
            if key.len() == 16 {
                keys.push(key.to_vec());
            }
        }
        Ok(keys)
    }

    fn write_batch(&self, batch: &StorageWriteBatch) -> Result<()> {
        RedbBatchWriter::with_durability(Arc::clone(&self.database), self.durability)
            .write_batch(batch)
    }

    fn backend(&self) -> StorageBackend {
        StorageBackend::Redb
    }

    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        match table_get(&self.database, MEMORIES_TABLE, id.0.as_bytes())? {
            Some(bytes) => Ok(Some(self.open_record(id.0.as_bytes(), &bytes)?)),
            None => Ok(None),
        }
    }

    fn put_memory(&self, memory: &Memory) -> Result<()> {
        let prepared = self.prepare(memory)?;
        self.commit_indexed(|txn| put_prepared(txn, &prepared))
    }

    fn update_memory(&self, memory: &Memory) -> Result<()> {
        // redb insert is an upsert, and the index rows are replaced from the reverse
        // map either way, so update and put share semantics.
        self.put_memory(memory)
    }

    fn store_with_vectors(&self, memory: &Memory, vector_ids: Vec<u32>) -> Result<()> {
        let prepared = self.prepare(memory)?;
        self.commit_indexed(|txn| {
            put_prepared(txn, &prepared)?;
            merge_text_vectors(txn, &memory.id, vector_ids)
        })
    }

    fn delete_memory(&self, id: &MemoryId) -> Result<()> {
        // Mirror RocksDB `delete_with_vectors`: the memory, its mapping and its index
        // rows go in one commit.
        let key = id.0.as_bytes().as_slice();
        self.commit_indexed(|txn| {
            txn.open_table(MEMORIES_TABLE)?.remove(key)?;
            txn.open_table(VMAPPINGS_TABLE)?.remove(key)?;
            apply_index_rows(txn, id, &IndexRows::new())
        })
    }

    fn ids_in_date_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MemoryId>> {
        let txn = self.database.begin_read()?;
        Ok(self
            .index_candidates(&txn, &SearchCriteria::ByDate { start, end })?
            .unwrap_or_default())
    }

    fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>> {
        let txn = self.database.begin_read()?;
        let Some(candidates) = self.index_candidates(&txn, &criteria)? else {
            return Ok(self
                .scan_all_memories()?
                .into_iter()
                .filter(|memory| !memory.is_forgotten() && criteria.matches(memory))
                .collect());
        };
        let memories = txn.open_table(MEMORIES_TABLE)?;
        let mut found = Vec::with_capacity(candidates.len());
        for id in candidates {
            let key = id.0.as_bytes();
            // The memory, not the row, is the truth: a row can outlive its memory only
            // through a write that bypassed this store.
            let Some(bytes) = memories.get(key.as_slice())? else {
                continue;
            };
            let Some(memory) = self.open_or_skip(key, bytes.value(), "search") else {
                continue;
            };
            if !memory.is_forgotten() && criteria.matches(&memory) {
                found.push(memory);
            }
        }
        Ok(found)
    }

    fn get_vector_mapping(&self, id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        match table_get(&self.database, VMAPPINGS_TABLE, id.0.as_bytes())? {
            Some(bytes) => Ok(Some(decode::<VectorMappingEntry>(&bytes)?)),
            None => Ok(None),
        }
    }

    fn put_vector_mapping(&self, id: &MemoryId, mapping: &VectorMappingEntry) -> Result<()> {
        let bytes = encode(mapping)?;
        table_put(
            &self.database,
            self.durability,
            VMAPPINGS_TABLE,
            id.0.as_bytes(),
            &bytes,
        )
    }

    fn delete_vector_mapping(&self, id: &MemoryId) -> Result<()> {
        table_remove(
            &self.database,
            self.durability,
            VMAPPINGS_TABLE,
            id.0.as_bytes(),
        )?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        make_durable(&self.database, self.durability)
    }

    // ── get_by_content_hash and set_fact_watermark have no error channel in their
    //    signatures, so each must answer for real: a refusal could not be heard.

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
        // The read and the write share a transaction, so a concurrent update of
        // another modality cannot be lost between them.
        let txn = begin_write(&self.database, self.durability)?;
        merge_text_vectors(&txn, memory_id, vector_ids)?;
        txn.commit()?;
        Ok(())
    }

    fn update_vector_mappings(&self, mappings: &[(MemoryId, Vec<u32>)]) -> Result<()> {
        if mappings.is_empty() {
            return Ok(());
        }
        let txn = begin_write(&self.database, self.durability)?;
        for (memory_id, vector_ids) in mappings {
            merge_text_vectors(&txn, memory_id, vector_ids.clone())?;
        }
        txn.commit()?;
        Ok(())
    }

    fn get_by_content_hash(&self, content: &str) -> Option<MemoryId> {
        // The memory whose content is exactly this, confirmed against the record so a
        // digest collision cannot answer for different content.
        self.indexed_memories(
            IndexKind::ContentHash,
            &self.blinded_term_key(&content_digest(content)),
            |m| m.experience.content == content,
            Some(1),
        )
        .ok()?
        .pop()
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
        self.mark_forgotten_where("forget-by-importance", |m| m.importance() < threshold)
    }

    fn save_interference_event_count(&self, count: usize) -> Result<()> {
        // Same key and encoding as rocksdb's default keyspace, so an
        // export/import carries the row across backends unchanged.
        table_put(
            &self.database,
            self.durability,
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
            self.durability,
            DEFAULT_TABLE,
            key.as_bytes(),
            &timestamp_millis.to_le_bytes(),
        ) {
            tracing::warn!("Failed to persist fact extraction watermark: {e}");
        }
    }

    fn key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        Arc::new(RedbTableKeyValueStore::with_durability(
            Arc::clone(&self.database),
            "kv",
            self.durability,
        ))
    }

    fn default_key_value_store(&self) -> Arc<dyn crate::storage::KeyValueStore> {
        Arc::new(RedbTableKeyValueStore::with_durability(
            Arc::clone(&self.database),
            DEFAULT_TABLE_NAME,
            self.durability,
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
    durability: StorageDurability,
}

impl RedbTableKeyValueStore {
    pub fn new(database: Arc<redb::Database>, table: &'static str) -> Self {
        Self::with_durability(database, table, StorageTuning::process().durability)
    }

    pub fn with_durability(
        database: Arc<redb::Database>,
        table: &'static str,
        durability: StorageDurability,
    ) -> Self {
        Self {
            database,
            table,
            durability,
        }
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
        table_put(&self.database, self.durability, self.def(), key, value)
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        table_remove(&self.database, self.durability, self.def(), key)?;
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
        make_durable(&self.database, self.durability)
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let txn = begin_write(&self.database, self.durability)?;
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
        let txn = begin_write(&self.database, self.durability)?;
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
    durability: StorageDurability,
}

impl RedbGraphStore {
    pub fn open(path: &Path) -> Result<Self> {
        let tuning = StorageTuning::process();
        let database = create_redb_database(&path.join("graph.redb"), &tuning, RedbFileRole::Graph)?;
        init_tables(
            &database,
            tuning.durability,
            [ENTITIES_TABLE, RELATIONSHIPS_TABLE, EPISODES_TABLE],
        )?;
        Ok(Self {
            database,
            durability: tuning.durability,
        })
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
        table_put(
            &self.database,
            self.durability,
            ENTITIES_TABLE,
            entity.uuid.as_bytes(),
            &bytes,
        )
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
            self.durability,
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
        table_put(
            &self.database,
            self.durability,
            EPISODES_TABLE,
            episode.uuid.as_bytes(),
            &bytes,
        )
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
        make_durable(&self.database, self.durability)
    }
}

/// Single table backing each namespace's key-value `.redb` file. Keys and values
/// are opaque byte strings (the trait's contract); callers serialize above this.
/// The `kv` family. NOT derived from `DEFAULT_TABLE_NAME` — see that constant.
const KV_TABLE: redb::TableDefinition<'static, &[u8], &[u8]> = redb::TableDefinition::new("kv");

/// redb-backed [`KeyValueStore`]. Each namespace is its own `<namespace>.redb` file
/// holding a single `kv` table. Operations are one transaction each, committed at the
/// store's durability. Prefix scans use redb's ordered byte-key range and stop at the
/// first key that leaves the prefix.
pub struct RedbKeyValueStore {
    database: Arc<redb::Database>,
    durability: StorageDurability,
    #[allow(dead_code)]
    path: PathBuf,
    #[allow(dead_code)]
    namespace: String,
}

impl RedbKeyValueStore {
    pub fn open(path: &Path, namespace: &str) -> Result<Self> {
        let tuning = StorageTuning::process();
        let database_path = path.join(format!("{}.redb", namespace));
        let database = create_redb_database(&database_path, &tuning, RedbFileRole::Auxiliary)?;
        // Materialize the table up front so reads never hit `TableDoesNotExist`.
        init_table(&database, tuning.durability, KV_TABLE)?;
        Ok(Self {
            database,
            durability: tuning.durability,
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
        let txn = begin_write(&self.database, self.durability)?;
        {
            let mut table = txn.open_table(KV_TABLE)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn delete_bytes(&self, key: &[u8]) -> Result<()> {
        let txn = begin_write(&self.database, self.durability)?;
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
        make_durable(&self.database, self.durability)
    }
    fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let txn = begin_write(&self.database, self.durability)?;
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
        let txn = begin_write(&self.database, self.durability)?;
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
        assert_eq!(
            kv.get_bytes(b"dials:a").unwrap().as_deref(),
            Some(&b"one"[..])
        );

        // Overwrite.
        kv.put_bytes(b"dials:a", b"ONE").unwrap();
        assert_eq!(
            kv.get_bytes(b"dials:a").unwrap().as_deref(),
            Some(&b"ONE"[..])
        );

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
        assert_eq!(
            kv2.get_bytes(b"dials:b").unwrap().as_deref(),
            Some(&b"two"[..])
        );
        assert_eq!(kv2.get_bytes(b"dials:a").unwrap(), None);
    }
}

pub struct RedbAuditStore {
    database: Arc<redb::Database>,
    durability: StorageDurability,
}

impl RedbAuditStore {
    pub fn open(path: &Path) -> Result<Self> {
        let tuning = StorageTuning::process();
        let database =
            create_redb_database(&path.join("audit.redb"), &tuning, RedbFileRole::Auxiliary)?;
        init_table(&database, tuning.durability, AUDIT_TABLE)?;
        Ok(Self {
            database,
            durability: tuning.durability,
        })
    }

    /// The live handle, for the redb backup: a second opener of audit.redb fails on its lock.
    pub fn database(&self) -> Arc<redb::Database> {
        Arc::clone(&self.database)
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
        table_put(
            &self.database,
            self.durability,
            AUDIT_TABLE,
            key.as_bytes(),
            &value,
        )
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
            let txn = begin_write(&self.database, self.durability)?;
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
        make_durable(&self.database, self.durability)
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

    fn found_by_tag(store: &RedbPrimaryMemoryStore, tag: &str) -> Vec<MemoryId> {
        store
            .search(SearchCriteria::ByTags(vec![tag.to_string()]))
            .expect("search")
            .into_iter()
            .map(|memory| memory.id)
            .collect()
    }

    #[test]
    fn a_store_opened_with_eventual_durability_reads_back_after_a_clean_close() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let eventual = StorageTuning {
            durability: StorageDurability::Eventual,
            redb_cache_bytes: Some(4 * 1024 * 1024),
            ..StorageTuning::default()
        };
        let unflushed = memory_with("committed, never flushed", &["Kiri"], &["unflushed"]);
        let flushed = memory_with("committed, then flushed", &["Kiri"], &["flushed"]);

        {
            let store = RedbPrimaryMemoryStore::open_with_tuning(tmp.path(), eventual)
                .expect("open with eventual durability");
            assert_eq!(store.durability(), StorageDurability::Eventual);
            store
                .store_with_vectors(&unflushed, vec![4, 2])
                .expect("store");
            store
                .key_value_store()
                .put_bytes(b"dials", b"tuned")
                .expect("kv put");
            store
                .save_interference_event_count(3)
                .expect("default-table put");
        }
        {
            let store =
                RedbPrimaryMemoryStore::open_with_tuning(tmp.path(), eventual).expect("reopen");
            assert_eq!(
                store.get(&unflushed.id).expect("get").experience.content,
                "committed, never flushed"
            );
            assert!(store
                .get_vector_mapping(&unflushed.id)
                .expect("mapping")
                .is_some());
            assert_eq!(
                store
                    .key_value_store()
                    .get_bytes(b"dials")
                    .expect("kv get")
                    .as_deref(),
                Some(&b"tuned"[..])
            );
            assert_eq!(store.load_all_interference_records().expect("load").1, 3);
            assert_eq!(
                found_by_tag(&store, "unflushed"),
                vec![unflushed.id.clone()]
            );

            store.put_memory(&flushed).expect("store");
            store.flush().expect("flush under eventual durability");
        }

        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("reopen at process tuning");
        assert_eq!(found_by_tag(&store, "flushed"), vec![flushed.id]);
        assert_eq!(found_by_tag(&store, "unflushed"), vec![unflushed.id]);
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
            reopened
                .get_memory(&id2)
                .unwrap()
                .unwrap()
                .experience
                .content,
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
            strengthening_sources: Default::default(),
            last_corroborated: None,
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
            store
                .list_relationships_for_entity(&bob.uuid)
                .unwrap()
                .len(),
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

    // ── Secondary indexes ────────────────────────────────────────────────────

    fn created_at(n: usize) -> chrono::DateTime<chrono::Utc> {
        // Eight hours apart, so several memories share each calendar day.
        chrono::DateTime::from_timestamp(1_700_000_000 + n as i64 * 28_800, 0).unwrap()
    }

    /// A memory whose every indexed field is a function of `n`, so a corpus over
    /// 0..N spreads across every index with collisions in each and gaps in each.
    fn indexed_memory(n: usize, parent: Option<&MemoryId>, cites: &[MemoryId]) -> Memory {
        let types = [
            ExperienceType::Observation,
            ExperienceType::Decision,
            ExperienceType::Learning,
        ];
        let mut context = crate::memory::context::ContextBuilder::new().build();
        context.episode.episode_id = (n % 4 != 3).then(|| format!("episode-{}", n % 3));
        context.episode.sequence_number = (!n.is_multiple_of(5)).then_some((n % 7) as u32);
        let experience = Experience {
            experience_type: types[n % types.len()].clone(),
            content: format!("content {}", n % 9),
            entities: vec![format!("Entity{}", n % 4), format!("ENTITY{}", (n + 1) % 4)],
            tags: vec![format!("Tag-{}", n % 5)],
            context: Some(context),
            robot_id: n.is_multiple_of(2).then(|| format!("robot-{}", n % 3)),
            mission_id: n.is_multiple_of(3).then(|| format!("mission-{}", n % 2)),
            geo_location: (!n.is_multiple_of(4)).then_some([
                (n as f64 * 7.3) % 180.0 - 90.0,
                (n as f64 * 31.7) % 360.0 - 180.0,
                0.0,
            ]),
            action_type: (n % 2 == 1).then(|| format!("action-{}", n % 3)),
            reward: (n % 3 != 1).then_some((n % 21) as f32 / 10.0 - 1.0),
            related_memories: cites.to_vec(),
            ..Default::default()
        };
        let mut memory = Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            experience,
            (n % 11) as f32 / 10.0,
            None,
            None,
            None,
            Some(created_at(n)),
        );
        memory.parent_id = parent.cloned();
        memory.external_id = (n % 3 == 2).then(|| format!("ext-{}", n % 4));
        if n % 6 == 5 {
            memory
                .experience
                .metadata
                .insert("forgotten".into(), "true".into());
        }
        memory
    }

    /// Sixty memories: every other one past the tenth parented, every third past
    /// the fifth citing an earlier one, every sixth forgotten.
    fn seed_corpus(store: &RedbPrimaryMemoryStore) -> Vec<Memory> {
        let mut corpus: Vec<Memory> = Vec::new();
        for n in 0..60usize {
            let parent = (n >= 10 && n.is_multiple_of(2)).then(|| corpus[n % 10].id.clone());
            let cites: Vec<MemoryId> = if n >= 5 && n.is_multiple_of(3) {
                vec![corpus[n - 5].id.clone()]
            } else {
                Vec::new()
            };
            let memory = indexed_memory(n, parent.as_ref(), &cites);
            store.put_memory(&memory).unwrap();
            corpus.push(memory);
        }
        corpus
    }

    /// Exercise the update and delete paths before anything is compared, so the
    /// comparison covers rows that were rewritten and rows that were removed.
    fn churn(store: &RedbPrimaryMemoryStore, corpus: &mut Vec<Memory>) {
        for index in [3, 14, 27] {
            let memory = &mut corpus[index];
            memory.experience.tags = vec!["churned".into()];
            memory.experience.entities = vec!["Entity9".into()];
            memory.external_id = Some("ext-churned".into());
            memory.parent_id = None;
            memory.experience.related_memories.clear();
            memory.set_importance(0.95);
            store.update_memory(memory).unwrap();
        }
        for index in [8, 33] {
            let memory = corpus.remove(index);
            store.delete_memory(&memory.id).unwrap();
        }
    }

    fn criteria_for(corpus: &[Memory]) -> Vec<SearchCriteria> {
        let located = corpus
            .iter()
            .find_map(|m| m.experience.resolved_geo())
            .expect("the corpus has located memories");
        vec![
            SearchCriteria::ByDate {
                start: created_at(3),
                end: created_at(20),
            },
            SearchCriteria::ByDate {
                start: created_at(5),
                end: created_at(5),
            },
            SearchCriteria::ByDate {
                start: created_at(20),
                end: created_at(3),
            },
            SearchCriteria::ByType(ExperienceType::Observation),
            SearchCriteria::ByType(ExperienceType::Decision),
            SearchCriteria::ByType(ExperienceType::Intention),
            SearchCriteria::ByImportance { min: 0.2, max: 0.6 },
            SearchCriteria::ByImportance { min: 0.0, max: 0.0 },
            SearchCriteria::ByImportance {
                min: -0.0,
                max: 0.1,
            },
            SearchCriteria::ByImportance { min: 0.7, max: 0.3 },
            SearchCriteria::ByImportance {
                min: f32::NAN,
                max: 1.0,
            },
            SearchCriteria::ByEntity("entity1".into()),
            SearchCriteria::ByEntity("ENTITY9".into()),
            SearchCriteria::ByEntity("missing".into()),
            SearchCriteria::ByTags(vec!["TAG-1".into(), "tag-3".into()]),
            SearchCriteria::ByTags(vec!["churned".into()]),
            SearchCriteria::ByTags(Vec::new()),
            SearchCriteria::ByEpisode("episode-1".into()),
            SearchCriteria::ByEpisode("episode-none".into()),
            SearchCriteria::ByEpisodeSequence {
                episode_id: "episode-0".into(),
                min_sequence: None,
                max_sequence: None,
            },
            SearchCriteria::ByEpisodeSequence {
                episode_id: "episode-1".into(),
                min_sequence: Some(2),
                max_sequence: Some(5),
            },
            SearchCriteria::ByEpisodeSequence {
                episode_id: "episode-2".into(),
                min_sequence: Some(3),
                max_sequence: None,
            },
            SearchCriteria::ByEpisodeSequence {
                episode_id: "episode-2".into(),
                min_sequence: Some(6),
                max_sequence: Some(1),
            },
            SearchCriteria::ByRobot("robot-0".into()),
            SearchCriteria::ByMission("mission-1".into()),
            SearchCriteria::ByLocation {
                lat: located.0,
                lon: located.1,
                radius_meters: 0.0,
            },
            SearchCriteria::ByLocation {
                lat: 0.0,
                lon: 0.0,
                radius_meters: 4_000_000.0,
            },
            SearchCriteria::ByLocation {
                lat: 89.5,
                lon: 0.0,
                radius_meters: 900_000.0,
            },
            SearchCriteria::ByLocation {
                lat: 10.0,
                lon: 179.5,
                radius_meters: 3_000_000.0,
            },
            SearchCriteria::ByLocation {
                lat: 0.0,
                lon: 0.0,
                radius_meters: f64::INFINITY,
            },
            SearchCriteria::ByLocation {
                lat: 135.0,
                lon: 0.0,
                radius_meters: 5_000_000.0,
            },
            SearchCriteria::ByLocation {
                lat: f64::NAN,
                lon: 0.0,
                radius_meters: 1_000.0,
            },
            SearchCriteria::ByLocation {
                lat: 0.0,
                lon: 0.0,
                radius_meters: -1.0,
            },
            SearchCriteria::ByActionType("action-1".into()),
            SearchCriteria::ByReward {
                min: -0.5,
                max: 0.5,
            },
            SearchCriteria::ByReward { min: 2.0, max: 3.0 },
            SearchCriteria::ByParent(corpus[0].id.clone()),
            SearchCriteria::ByParent(MemoryId(uuid::Uuid::new_v4())),
            SearchCriteria::RootsOnly,
            SearchCriteria::Combined(vec![
                SearchCriteria::ByTags(vec!["tag-1".into(), "tag-2".into()]),
                SearchCriteria::ByEntity("entity0".into()),
            ]),
            SearchCriteria::Combined(vec![
                SearchCriteria::RootsOnly,
                SearchCriteria::ByType(ExperienceType::Decision),
            ]),
            SearchCriteria::Combined(vec![SearchCriteria::RootsOnly]),
            SearchCriteria::Combined(Vec::new()),
            SearchCriteria::Combined(vec![
                SearchCriteria::Combined(vec![
                    SearchCriteria::ByRobot("robot-0".into()),
                    SearchCriteria::ByReward {
                        min: -1.0,
                        max: 1.0,
                    },
                ]),
                SearchCriteria::ByImportance { min: 0.0, max: 0.5 },
            ]),
        ]
    }

    /// What `search` must answer: every record, through the exact predicate.
    fn scanned(store: &RedbPrimaryMemoryStore, criteria: &SearchCriteria) -> Vec<MemoryId> {
        let mut ids: Vec<MemoryId> = store
            .scan_all_memories()
            .unwrap()
            .into_iter()
            .filter(|m| !m.is_forgotten() && criteria.matches(m))
            .map(|m| m.id)
            .collect();
        ids.sort();
        ids
    }

    fn searched(store: &RedbPrimaryMemoryStore, criteria: &SearchCriteria) -> Vec<MemoryId> {
        let mut ids: Vec<MemoryId> = store
            .search(criteria.clone())
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        ids.sort();
        ids
    }

    /// Every row of every derived table, for comparing two states of the indexes.
    fn derived_rows(store: &RedbPrimaryMemoryStore) -> Vec<(&'static str, Vec<u8>, Vec<u8>)> {
        let txn = store.database.begin_read().unwrap();
        let mut rows = Vec::new();
        for name in DERIVED_INDEX_TABLES {
            let table = txn.open_table(BytesTable::new(name)).unwrap();
            for row in table.iter().unwrap() {
                let (k, v) = row.unwrap();
                rows.push((name, k.value().to_vec(), v.value().to_vec()));
            }
        }
        rows
    }

    fn owned_rows(store: &RedbPrimaryMemoryStore, id: &MemoryId) -> IndexRows {
        let txn = store.database.begin_read().unwrap();
        let table = txn.open_table(INDEX_ROWS_TABLE).unwrap();
        let stored = table.get(id.0.as_bytes().as_slice()).unwrap();
        stored
            .map(|v| decode_index_rows(v.value()).unwrap())
            .unwrap_or_default()
    }

    fn index_holds(store: &RedbPrimaryMemoryStore, kind: IndexKind, key: &[u8]) -> bool {
        let txn = store.database.begin_read().unwrap();
        let table = txn.open_table(kind.table()).unwrap();
        table.get(key).unwrap().is_some()
    }

    fn rows_naming(store: &RedbPrimaryMemoryStore, id: &MemoryId) -> usize {
        let txn = store.database.begin_read().unwrap();
        let mut count = 0;
        for kind in IndexKind::ALL {
            let table = txn.open_table(kind.table()).unwrap();
            for row in table.iter().unwrap() {
                let (k, _) = row.unwrap();
                if trailing_memory_id(k.value()).as_ref() == Some(id) {
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn index_kinds_are_stored_by_a_stable_discriminant() {
        for (position, kind) in IndexKind::ALL.iter().enumerate() {
            assert_eq!(*kind as usize, position, "ALL is out of discriminant order");
            assert_eq!(IndexKind::from_code(*kind as u8), Some(*kind));
        }
        assert_eq!(IndexKind::from_code(IndexKind::ALL.len() as u8), None);
        let mut names = DERIVED_INDEX_TABLES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            DERIVED_INDEX_TABLES.len(),
            "two indexes share a table"
        );
    }

    #[test]
    fn index_rows_are_written_replaced_and_removed_with_their_memory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
        let parent = indexed_memory(0, None, &[]);
        let target = indexed_memory(1, None, &[]);
        store.put_memory(&parent).unwrap();
        store.put_memory(&target).unwrap();

        // n = 6 sets every optional indexed field but these two, set here.
        let mut memory = indexed_memory(6, Some(&parent.id), std::slice::from_ref(&target.id));
        memory.external_id = Some("ext-row".into());
        memory.experience.action_type = Some("action-row".into());
        store.put_memory(&memory).unwrap();

        let written = owned_rows(&store, &memory.id);
        for kind in IndexKind::ALL {
            assert!(
                written.keys().any(|(k, _)| *k == kind),
                "no {kind:?} row was written for a memory that carries the field"
            );
        }
        for (kind, key) in written.keys() {
            assert!(
                index_holds(&store, *kind, key),
                "{kind:?} row listed but absent"
            );
        }

        let old_tag = store.blinded_term_key("tag-1");
        let old_tag_key = with_memory_id(old_tag, &memory.id);
        assert!(index_holds(&store, IndexKind::Tag, &old_tag_key));

        memory.experience.tags = vec!["Renamed".into()];
        memory.external_id = None;
        memory.experience.related_memories.clear();
        store.update_memory(&memory).unwrap();

        let rewritten = owned_rows(&store, &memory.id);
        assert!(
            !index_holds(&store, IndexKind::Tag, &old_tag_key),
            "the old tag's row survived the update"
        );
        assert!(index_holds(
            &store,
            IndexKind::Tag,
            &with_memory_id(store.blinded_term_key("renamed"), &memory.id)
        ));
        assert!(!rewritten.keys().any(|(k, _)| *k == IndexKind::ExternalId));
        let cites = |cited: &MemoryId| {
            with_memory_id(store.blinded_term_key(&cited.0.to_string()), &memory.id)
        };
        assert!(
            !index_holds(&store, IndexKind::InboundRef, &cites(&target.id)),
            "the dropped reference's row survived the update"
        );
        assert!(
            index_holds(&store, IndexKind::InboundRef, &cites(&parent.id)),
            "the parent is still referenced and lost its row"
        );
        assert!(!store.has_any_inbound_refs(&target.id, "u").unwrap());
        assert_eq!(
            rows_naming(&store, &memory.id),
            rewritten.len(),
            "the index tables hold rows the reverse map does not list"
        );

        store
            .put_vector_mapping(&memory.id, &VectorMappingEntry::with_text(vec![1]))
            .unwrap();
        store.delete_memory(&memory.id).unwrap();
        assert!(store.get_memory(&memory.id).unwrap().is_none());
        assert!(store.get_vector_mapping(&memory.id).unwrap().is_none());
        assert!(owned_rows(&store, &memory.id).is_empty());
        assert_eq!(
            rows_naming(&store, &memory.id),
            0,
            "a deleted memory left rows in an index"
        );
        assert!(
            store.get_children(&parent.id).unwrap().is_empty(),
            "a deleted child is still listed under its parent"
        );
    }

    #[test]
    fn every_search_variant_answers_what_a_scan_answers() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
        let mut corpus = seed_corpus(&store);
        churn(&store, &mut corpus);
        store.mark_forgotten_by_importance(0.05).unwrap();

        let criteria = criteria_for(&corpus);
        let mut answered = 0;
        for criterion in &criteria {
            let expected = scanned(&store, criterion);
            assert_eq!(
                searched(&store, criterion),
                expected,
                "search disagrees with a scan for {criterion:?}"
            );
            if !expected.is_empty() {
                answered += 1;
            }
        }
        // Not vacuous: two empty answers agree too.
        assert!(
            answered >= 20,
            "only {answered} of {} criteria matched anything",
            criteria.len()
        );
    }

    #[test]
    fn range_indexes_return_their_own_order() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
        seed_corpus(&store);

        let by_sequence = store
            .search(SearchCriteria::ByEpisodeSequence {
                episode_id: "episode-1".into(),
                min_sequence: None,
                max_sequence: None,
            })
            .unwrap();
        assert!(by_sequence.len() > 2);
        let sequences: Vec<u32> = by_sequence
            .iter()
            .map(|m| memory_sequence(m).unwrap())
            .collect();
        assert!(
            sequences.windows(2).all(|w| w[0] <= w[1]),
            "episode search must come back in sequence order: {sequences:?}"
        );

        let by_date = store
            .search(SearchCriteria::ByDate {
                start: created_at(0),
                end: created_at(59),
            })
            .unwrap();
        assert!(by_date
            .windows(2)
            .all(|w| w[0].created_at <= w[1].created_at));
    }

    #[test]
    fn point_lookups_and_the_citation_check_follow_updates() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
        let parent = indexed_memory(0, None, &[]);
        let target = indexed_memory(1, None, &[]);
        let mut holder = indexed_memory(4, Some(&parent.id), std::slice::from_ref(&target.id));
        holder.external_id = Some("linear:SHO-1".into());
        holder.experience.content = "the holder's own words".into();
        for m in [&parent, &target, &holder] {
            store.put_memory(m).unwrap();
        }

        assert!(store.has_any_inbound_refs(&target.id, "u").unwrap());
        assert!(store.has_any_inbound_refs(&parent.id, "u").unwrap());
        assert!(!store.has_any_inbound_refs(&holder.id, "u").unwrap());
        assert_eq!(
            store
                .find_by_external_id("linear:SHO-1")
                .unwrap()
                .map(|m| m.id),
            Some(holder.id.clone())
        );
        assert_eq!(
            store.get_by_content_hash("the holder's own words"),
            Some(holder.id.clone())
        );
        assert_eq!(
            store
                .get_children(&parent.id)
                .unwrap()
                .into_iter()
                .map(|m| m.id)
                .collect::<Vec<_>>(),
            vec![holder.id.clone()]
        );

        holder.experience.related_memories.clear();
        holder.parent_id = None;
        holder.external_id = Some("linear:SHO-2".into());
        holder.experience.content = "words rewritten".into();
        store.update_memory(&holder).unwrap();

        assert!(!store.has_any_inbound_refs(&target.id, "u").unwrap());
        assert!(!store.has_any_inbound_refs(&parent.id, "u").unwrap());
        assert!(store.find_by_external_id("linear:SHO-1").unwrap().is_none());
        assert_eq!(
            store
                .find_by_external_id("linear:SHO-2")
                .unwrap()
                .map(|m| m.id),
            Some(holder.id.clone())
        );
        assert_eq!(store.get_by_content_hash("the holder's own words"), None);
        assert_eq!(
            store.get_by_content_hash("words rewritten"),
            Some(holder.id.clone())
        );
        assert!(store.get_children(&parent.id).unwrap().is_empty());
    }

    #[test]
    fn a_store_written_before_the_indexes_is_backfilled_on_open() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (criteria, expected) = {
            let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
            let mut corpus = seed_corpus(&store);
            churn(&store, &mut corpus);
            let criteria = criteria_for(&corpus);
            let expected: Vec<Vec<MemoryId>> =
                criteria.iter().map(|c| scanned(&store, c)).collect();
            (criteria, expected)
        };

        // What a store from before the indexes looks like: their tables never existed.
        {
            let database = create_redb_database_for_test(&tmp.path().join("storage.redb")).unwrap();
            let txn = database.begin_write().unwrap();
            for name in DERIVED_INDEX_TABLES {
                txn.delete_table(BytesTable::new(name)).unwrap();
            }
            txn.commit().unwrap();
        }

        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("reopen");
        let memory_count = store.get_all_ids().unwrap().len() as u64;
        {
            let txn = store.database.begin_read().unwrap();
            let owned = txn.open_table(INDEX_ROWS_TABLE).unwrap();
            assert_eq!(
                owned.len().unwrap(),
                memory_count,
                "the backfill skipped memories"
            );
        }
        for (criterion, expected) in criteria.iter().zip(&expected) {
            assert_eq!(
                &searched(&store, criterion),
                expected,
                "the backfilled index disagrees with the scan for {criterion:?}"
            );
        }

        let backfilled = derived_rows(&store);
        drop(store);
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("second reopen");
        assert_eq!(
            derived_rows(&store),
            backfilled,
            "a reopen of a stamped store changed its indexes"
        );

        // A stamp from another schema version rebuilds to the same rows.
        {
            let txn = store.database.begin_write().unwrap();
            txn.open_table(INDEX_META_TABLE)
                .unwrap()
                .insert(INDEX_SCHEMA_KEY, 0u32.to_le_bytes().as_slice())
                .unwrap();
            txn.open_table(IndexKind::Tag.table())
                .unwrap()
                .retain(|_, _| false)
                .unwrap();
            txn.commit().unwrap();
        }
        drop(store);
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("third reopen");
        assert_eq!(derived_rows(&store), backfilled);
    }

    #[test]
    fn an_opened_empty_store_holds_no_rows_until_its_first_memory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let row_total = |store: &RedbPrimaryMemoryStore| -> u64 {
            let txn = store.database.begin_read().unwrap();
            let mut total = 0;
            for table in txn.list_tables().unwrap() {
                let def: redb::TableDefinition<'_, &[u8], &[u8]> =
                    redb::TableDefinition::new(table.name());
                total += txn.open_table(def).unwrap().len().unwrap();
            }
            total
        };

        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
        // A row here would make an untouched store a populated migrate/import target.
        assert_eq!(row_total(&store), 0);
        drop(store);
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("reopen");
        assert_eq!(row_total(&store), 0);

        store.put_memory(&indexed_memory(2, None, &[])).unwrap();
        assert_eq!(
            table_get(&store.database, INDEX_META_TABLE, INDEX_SCHEMA_KEY).unwrap(),
            Some(store.index_schema_stamp()),
            "the first write did not stamp the schema"
        );
    }

    #[test]
    fn store_with_vectors_writes_the_memory_and_its_text_vectors_together() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = RedbPrimaryMemoryStore::open(tmp.path()).expect("open");
        let memory = memory_with("vectored", &["Veld"], &["paired"]);
        store
            .put_vector_mapping(
                &memory.id,
                &VectorMappingEntry::with_text(vec![1]).with_image(vec![9]),
            )
            .unwrap();

        store.store_with_vectors(&memory, vec![4, 5]).unwrap();

        assert!(store.get_memory(&memory.id).unwrap().is_some());
        let mapping = store.get_vector_mapping(&memory.id).unwrap().unwrap();
        assert_eq!(
            mapping.text_vectors().map(Vec::as_slice),
            Some([4, 5].as_slice())
        );
        assert!(
            mapping
                .all_vector_ids()
                .contains(&(crate::memory::storage::Modality::Image, 9)),
            "storing text vectors dropped another modality"
        );
        assert_eq!(
            searched(&store, &SearchCriteria::ByTags(vec!["paired".into()])),
            vec![memory.id.clone()]
        );
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
        Self {
            path: path.to_path_buf(),
            database,
        }
    }

    /// Open standalone. Only valid when nothing else holds the database: an offline
    /// tool, or a restore into a fresh target. Fails on the lock otherwise, loudly.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let database = create_redb_database(
            &path.join("storage.redb"),
            &StorageTuning::process(),
            RedbFileRole::Memories,
        )?;
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
        tracing::info!(
            user_id,
            backup_id,
            records,
            memory_count,
            "redb backup written"
        );
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
                Ok(_) => tracing::error!(
                    backup_id = meta.backup_id,
                    "backup checksum does not match its metadata — NOT listed as restorable"
                ),
                Err(e) => tracing::error!(backup_id = meta.backup_id, error = %e,
                    "backup dump is unreadable — NOT listed as restorable"),
            }
        }
        // Newest first, as every backup listing on either engine is: clients show the
        // most recent first and a restore without an id takes it.
        out.sort_by_key(|m| std::cmp::Reverse(m.backup_id));
        Ok(out)
    }

    fn restore_backup(&self, user_id: &str, backup_id: u32, target_path: &Path) -> Result<()> {
        let dump = self
            .backup_dir(user_id)
            .join(format!("{backup_id}.veldport"));
        if !dump.exists() {
            bail!(
                "no backup {backup_id} for '{user_id}' at {}",
                dump.display()
            );
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
        Self {
            path: path.to_path_buf(),
            database,
        }
    }

    /// Open standalone. Valid only when nothing else holds the database: an offline
    /// tool, or a fresh import target.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let database = create_redb_database(
            &path.join("storage.redb"),
            &StorageTuning::process(),
            RedbFileRole::Memories,
        )?;
        Ok(Self {
            path: path.to_path_buf(),
            database,
        })
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

        let file =
            std::fs::File::open(source).with_context(|| format!("opening {}", source.display()))?;
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
        let mut names: Vec<String> = read.list_tables()?.map(|t| t.name().to_string()).collect();
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
        let file =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
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

        let file =
            std::fs::File::open(source).with_context(|| format!("opening {}", source.display()))?;
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
        // store that looks like a smaller one. Immediate whatever the configured
        // class: a restore reports success once, and must still hold after a power loss.
        let txn = begin_write(&database, StorageDurability::Immediate)?;
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
        create_redb_database_for_test(&path).expect("create db")
    }

    #[test]
    fn a_batch_spanning_two_tables_commits_both() {
        // This is the property `supports_transactional_batch: true` actually
        // claims — atomic ACROSS families, not merely within one. It is the
        // property src/memory/prospective.rs:298 and :382 depend on.
        let tmp = tempfile::tempdir().expect("tempdir");
        let database = db(&tmp);
        init_tables(
            &database,
            StorageDurability::Immediate,
            [MEMORIES_TABLE, AUDIT_TABLE],
        )
        .expect("init memories and audit");

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
        init_table(&database, StorageDurability::Immediate, MEMORIES_TABLE).expect("init memories");

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
        let database = create_redb_database_for_test(&tmp.path().join("agree.redb")).expect("db");
        init_table(
            &database,
            StorageDurability::Immediate,
            redb::TableDefinition::<&[u8], &[u8]>::new(DEFAULT_TABLE_NAME),
        )
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
            table
                .get(b"k".as_slice())
                .unwrap()
                .map(|v| v.value().to_vec()),
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
        PortableRecord {
            cf: cf.into(),
            key: k.to_vec(),
            value: v.to_vec(),
        }
    }

    fn write_dump(dir: &Path, bytes: &[u8]) -> PathBuf {
        let p = dir.join("in.veldport");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn row_count(target: &Path) -> u64 {
        let db = create_redb_database_for_test(&target.join("storage.redb")).unwrap();
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

        let db = create_redb_database_for_test(&target.join("storage.redb")).unwrap();
        let read = db.begin_read().unwrap();
        for (cf, key, want) in [
            ("default", &b"acl:alice:1"[..], &b"a"[..]),
            ("memory_index", b"i", b"b"),
            ("kv", b"dials", b"c"),
        ] {
            let def: redb::TableDefinition<'_, &[u8], &[u8]> = redb::TableDefinition::new(cf);
            let t = read
                .open_table(def)
                .unwrap_or_else(|e| panic!("table {cf} missing: {e}"));
            assert_eq!(
                t.get(key).unwrap().map(|v| v.value().to_vec()),
                Some(want.to_vec())
            );
        }
    }

    #[test]
    fn a_populated_target_is_refused_and_left_alone() {
        // The likely mistake is a mis-aimed path, and it is silent: the rows land,
        // the count is right, and the store now holds two users' data.
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("target");
        let first = write_dump(tmp.path(), &dump_of(&[rec("kv", b"already", b"here")]));
        RedbMigrationStore::open(&target)
            .unwrap()
            .import_portable("a", &first)
            .unwrap();
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
        RedbMigrationStore::open(&target)
            .unwrap()
            .import_portable("a", &src)
            .expect("import");
        assert_eq!(row_count(&target), 0);
    }
}

#[cfg(test)]
mod verify_portable_tests {
    use super::*;
    use crate::storage::portable::{PortableRecord, PortableWriter};

    fn rec(cf: &str, k: &[u8], v: &[u8]) -> PortableRecord {
        PortableRecord {
            cf: cf.into(),
            key: k.to_vec(),
            value: v.to_vec(),
        }
    }

    fn dump_at(dir: &Path, records: &[PortableRecord]) -> PathBuf {
        let mut buf = Vec::new();
        let mut w = PortableWriter::new(&mut buf).unwrap();
        for r in records {
            w.write_record(r).unwrap();
        }
        w.finish().unwrap();
        let p = dir.join("d.veldport");
        std::fs::write(&p, &buf).unwrap();
        p
    }

    fn imported(dir: &Path, records: &[PortableRecord]) -> (PathBuf, PathBuf) {
        let src = dump_at(dir, records);
        let target = dir.join("t");
        RedbMigrationStore::open(&target)
            .unwrap()
            .import_portable("u", &src)
            .unwrap();
        (src, target)
    }

    #[test]
    fn a_faithful_import_verifies() {
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(
            tmp.path(),
            &[rec("default", b"a", b"1"), rec("kv", b"b", b"2")],
        );
        RedbMigrationStore::open(&target)
            .unwrap()
            .verify_portable(&src)
            .expect("verify");
    }

    #[test]
    fn a_changed_value_is_caught() {
        // Counts alone would pass this: the right number of rows, one of them wrong.
        let tmp = tempfile::tempdir().unwrap();
        let (src, target) = imported(tmp.path(), &[rec("kv", b"b", b"2")]);
        {
            let db = create_redb_database_for_test(&target.join("storage.redb")).unwrap();
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
            let db = create_redb_database_for_test(&target.join("storage.redb")).unwrap();
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
            let db = create_redb_database_for_test(&target.join("storage.redb")).unwrap();
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
        store
            .verify_portable(&src)
            .expect("verify must be idempotent");
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
        let ids: Vec<_> = (0..5)
            .map(|_| {
                let m = mem(None, None);
                s.put_memory(&m).unwrap();
                m.id
            })
            .collect();
        assert_eq!(s.get_all().unwrap().len(), 5);
        let mut got = s.get_all_ids().unwrap();
        let mut want = ids;
        got.sort_by_key(|i| i.0);
        want.sort_by_key(|i| i.0);
        assert_eq!(
            got, want,
            "get_all_ids must be the same set get_all returns, not a subset"
        );
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
        for m in [&a, &b, &unrelated] {
            s.put_memory(m).unwrap();
        }

        let kids = s.get_children(&parent.id).unwrap();
        assert_eq!(kids.len(), 2);
        // Both directions: the two are present AND the unrelated one is not.
        assert!(kids.iter().any(|m| m.id == a.id) && kids.iter().any(|m| m.id == b.id));
        assert!(
            !kids.iter().any(|m| m.id == unrelated.id),
            "an unparented memory was returned as a child"
        );
    }

    #[test]
    fn ancestors_walk_to_the_root_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let root = mem(None, None);
        let mid = mem(Some(root.id.clone()), None);
        let leaf = mem(Some(mid.id.clone()), None);
        for m in [&root, &mid, &leaf] {
            s.put_memory(m).unwrap();
        }

        let anc = s.get_ancestors(&leaf.id).unwrap();
        assert_eq!(
            anc.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            vec![mid.id.clone(), root.id.clone()],
            "ancestors must be nearest-first, matching the rocksdb walk"
        );
        assert!(
            s.get_ancestors(&root.id).unwrap().is_empty(),
            "a root has no ancestors"
        );
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
        assert_eq!(
            anc.len(),
            100,
            "the walk must stop at the rocksdb bound, not run forever"
        );
    }

    #[test]
    fn external_id_lookup_finds_the_one_and_only_that_one() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let tagged = mem(None, Some("ext-42"));
        let other = mem(None, Some("ext-99"));
        let untagged = mem(None, None);
        for m in [&tagged, &other, &untagged] {
            s.put_memory(m).unwrap();
        }

        assert_eq!(
            s.find_by_external_id("ext-42").unwrap().map(|m| m.id),
            Some(tagged.id)
        );
        assert!(
            s.find_by_external_id("ext-nothing").unwrap().is_none(),
            "an unknown external id must be None, never the first memory scanned"
        );
    }

    #[test]
    fn hierarchy_context_is_its_three_parts() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(&tmp);
        let root = mem(None, None);
        let target = mem(Some(root.id.clone()), None);
        let kid = mem(Some(target.id.clone()), None);
        for m in [&root, &target, &kid] {
            s.put_memory(m).unwrap();
        }

        let (anc, me, kids) = s.get_hierarchy_context(&target.id).unwrap();
        assert_eq!(me.id, target.id);
        assert_eq!(
            anc.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            vec![root.id]
        );
        assert_eq!(
            kids.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            vec![kid.id]
        );
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
                Experience {
                    experience_type: ExperienceType::Observation,
                    content: "x".into(),
                    ..Default::default()
                },
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
        backups
            .restore_backup("alice", 1, &target)
            .expect("restore");
        let restored = RedbPrimaryMemoryStore::open(&target).unwrap();
        assert_eq!(
            restored.get_all().unwrap().len(),
            4,
            "the restore did not carry the memories"
        );
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

        assert!(
            backups.list_backups("alice").unwrap().is_empty(),
            "a truncated dump was offered as restorable"
        );
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
        )
        .unwrap();

        assert!(
            backups.list_backups("alice").unwrap().is_empty(),
            "a backup whose metadata disagrees with its bytes was listed as usable"
        );
    }

    #[test]
    fn no_backups_yet_is_an_empty_list_and_not_an_error() {
        // ENOENT is the only absence that legitimately means "none".
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let _s = seeded_store(&store_dir, 1);
        assert!(RedbBackupStore::from_database(_s.database(), &store_dir)
            .list_backups("nobody")
            .unwrap()
            .is_empty());
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
        assert_eq!(
            backups.create_backup("alice").unwrap().backup_id,
            3,
            "id 2 was reused after its files were removed"
        );
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
        let err = backups
            .restore_backup("alice", 1, &target)
            .expect_err("restoring over a populated store must refuse");
        assert!(err.to_string().contains("already holds rows"), "got: {err}");
    }
}
