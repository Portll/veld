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

use anyhow::{anyhow, Result};
use redb::{ReadableTable, ReadableTableMetadata};
use std::collections::HashSet;
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

// =============================================================================
// Shared redb helpers
//
// Every domain store below persists `bincode(serde)`-encoded records in a single
// `<&[u8], &[u8]>` table. Keys are the record's natural identity bytes (the UUID
// for memories/graph nodes; `{user_id}:{timestamp_nanos:020}` for audit). Reads
// are point lookups or ordered range scans; redb commits are durable, so every
// `flush` is a no-op.
// =============================================================================

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

pub struct RedbPrimaryMemoryStore {
    database: Arc<redb::Database>,
}

impl RedbPrimaryMemoryStore {
    pub fn open(path: &Path) -> Result<Self> {
        let database_path = path.join("storage.redb");
        let database = create_redb_database(&database_path)?;
        init_table(&database, MEMORIES_TABLE)?;
        init_table(&database, VMAPPINGS_TABLE)?;
        Ok(Self { database })
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