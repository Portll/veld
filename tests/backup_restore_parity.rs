#![cfg(all(feature = "rocksdb", feature = "storage-redb"))]
//! ADR 0007 gate: "backup and point-in-time restore on redb produce the same restored store as
//! the RocksDB engine".
//!
//! The same deterministic corpus is written to a RocksDB tenant and a redb tenant through the
//! storage traits, backed up, verified, mutated, and restored into a fresh directory. Each
//! restored store is compared with the snapshot its own engine took before the mutations, so a
//! restore that hands back the live state fails as surely as one that loses a record, and the
//! two restored stores are compared with each other.
//!
//! Two backup surfaces are covered on each engine:
//! - the `BackupStore` trait, which on both engines captures the primary store alone;
//! - the tenant backup behind `/api/backup/*`: `VeldBackupEngine`'s comprehensive archive on
//!   RocksDB and `RedbBackupEngine` on redb, which also carry the graph and the audit log.
//!
//! Each surface runs a plaintext corpus and an encryption-v2 corpus. The snapshot, comparison and
//! artefact-scan helpers are `pub` so the other parity tests can take them over.
//!
//! Own test binary, because `VELD_MASTER_PASSPHRASE` is process-global.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{bail, ensure, Context, Result};
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde_json::{json, Map, Value};
use tempfile::TempDir;
use uuid::Uuid;

use veld::backup::{
    BackupMetadata, IntentLogBackupSpec, RestoreOptions, SecondaryStoreRef, VeldBackupEngine,
};
use veld::graph_memory::{
    EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, PiiClassification,
    RelationType, RelationshipEdge,
};
use veld::keystore::{KdfParams, Keystore};
use veld::memory::storage::{Modality, ModalityVectors, SearchCriteria, VectorMappingEntry};
use veld::memory::{
    ContextId, EpisodeContext, Experience, ExperienceType, InterferenceRecord, InterferenceType,
    Memory, MemoryId, MemoryTier, RichContext,
};
use veld::storage::legacy_rocksdb::{RocksDbAuditStore, RocksDbPrimaryMemoryStore};
use veld::storage::migrate::backup::RedbBackupEngine;
use veld::storage::migrate::LiveRedb;
use veld::storage::redb::{
    open_graph_database, open_shared_database, RedbAuditStore, RedbBackupStore, RedbColumnStore,
    RedbPrimaryMemoryStore, KEYSTORE_FILE,
};
use veld::storage::{
    AuditLogEntry, AuditStore, BackupStore, PrimaryMemoryStore, RocksDbBackupStore,
};

pub const TENANT: &str = "parity-tenant";
pub const NEIGHBOUR: &str = "parity-neighbour";
/// Written into every memory's content and nowhere else, so finding it in a file means a
/// memory's content reached that file in plaintext.
pub const CONTENT_MARKER: &str = "PARITY-CONTENT-MARKER-Q7X2M9K4W";

const KEYSTORE_PASSPHRASE: &str = "backup-parity-correct-horse-battery-staple-B7";
const KEY_VARIABLES: [&str; 4] = [
    "VELD_MASTER_PASSPHRASE",
    "VELD_KMS_WRAP_KEY",
    "VELD_ENCRYPTION_KEY",
    "VELD_ALLOW_KEYSTORE_ROLLBACK",
];

const PRIMARY_CORPUS_SIZE: u32 = 300;
const LISTING_CORPUS_SIZE: u32 = 24;
const GRAPH_ENTITY_COUNT: u32 = 24;
const GRAPH_EDGE_ATTEMPTS: u32 = 60;
const AUDIT_EVENT_COUNT: u32 = 50;
const AUDIT_KEPT_AFTER_ROTATION: usize = 45;
const DIFFERENCES_SHOWN: usize = 40;

const SHARED_STORE: &str = "shared";
const GRAPH_STORE: &str = "graph";
const AUDIT_FAMILY: &str = "audit";
const KV_ROW_PREFIX: &str = "parity:kv:";
const DEFAULT_ROW_PREFIX: &str = "parity:default:";

const MEMORY_NAMESPACE: u128 = 0x5eed_0001_0000_4000_8000_0000_0000_0000;
const CONTEXT_NAMESPACE: u128 = 0x5eed_0002_0000_4000_8000_0000_0000_0000;
const EPISODE_NAMESPACE: u128 = 0x5eed_0003_0000_4000_8000_0000_0000_0000;

pub const PRIMARY_MEMORIES: &str = "primary.memories";
pub const PRIMARY_VECTOR_MAPPINGS: &str = "primary.vector_mappings";
pub const PRIMARY_KV: &str = "primary.kv";
pub const PRIMARY_DEFAULT: &str = "primary.default";
pub const PRIMARY_INTERFERENCE: &str = "primary.interference";
pub const PRIMARY_FACT_WATERMARK: &str = "primary.fact_watermark";
pub const PRIMARY_STATS: &str = "primary.stats";
pub const PRIMARY_EXTERNAL_IDS: &str = "primary.external_ids";
pub const PRIMARY_CHILDREN: &str = "primary.children";
pub const PRIMARY_TAG_SEARCH: &str = "primary.tag_search";
pub const PRIMARY_EPISODE_SEARCH: &str = "primary.episode_search";
pub const GRAPH_ENTITIES: &str = "graph.entities";
pub const GRAPH_EDGES_BY_NODE: &str = "graph.edges_by_node";
pub const GRAPH_EPISODES: &str = "graph.episodes";
pub const GRAPH_STATS: &str = "graph.stats";
pub const AUDIT_EVENTS: &str = "audit.events";
pub const AUDIT_USERS: &str = "audit.users";

/// Every primary section the corpus mutations change.
const PRIMARY_SECTIONS: [&str; 11] = [
    PRIMARY_MEMORIES,
    PRIMARY_VECTOR_MAPPINGS,
    PRIMARY_KV,
    PRIMARY_DEFAULT,
    PRIMARY_INTERFERENCE,
    PRIMARY_FACT_WATERMARK,
    PRIMARY_STATS,
    PRIMARY_EXTERNAL_IDS,
    PRIMARY_CHILDREN,
    PRIMARY_TAG_SEARCH,
    PRIMARY_EPISODE_SEARCH,
];
const GRAPH_SECTIONS: [&str; 4] = [
    GRAPH_ENTITIES,
    GRAPH_EDGES_BY_NODE,
    GRAPH_EPISODES,
    GRAPH_STATS,
];

const EXPERIENCE_TYPES: [ExperienceType; 6] = [
    ExperienceType::Observation,
    ExperienceType::Decision,
    ExperienceType::Learning,
    ExperienceType::Discovery,
    ExperienceType::Task,
    ExperienceType::Conversation,
];
const TIERS: [MemoryTier; 4] = [
    MemoryTier::Working,
    MemoryTier::Session,
    MemoryTier::LongTerm,
    MemoryTier::Archive,
];
const ENTITY_LABELS: [EntityLabel; 3] = [
    EntityLabel::Concept,
    EntityLabel::Technology,
    EntityLabel::Project,
];
const RELATIONS: [RelationType; 6] = [
    RelationType::WorksWith,
    RelationType::PartOf,
    RelationType::Uses,
    RelationType::Causes,
    RelationType::Contains,
    RelationType::RelatedTo,
];
const EPISODE_SOURCES: [EpisodeSource; 4] = [
    EpisodeSource::Message,
    EpisodeSource::Document,
    EpisodeSource::Event,
    EpisodeSource::Observation,
];
const AUDIT_EVENT_TYPES: [&str; 4] = ["remember", "update", "forget", "recall"];

// ─── environment ────────────────────────────────────────────────────────────────

/// Whether a corpus is written in plaintext or sealed under encryption-v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusKind {
    Plaintext,
    Sealed,
}

impl CorpusKind {
    pub fn is_sealed(self) -> bool {
        self == CorpusKind::Sealed
    }
}

static KEY_VARIABLES_IN_USE: Mutex<()> = Mutex::new(());

/// Exclusive use of the process-global key variables for one test: cleared on entry and on
/// drop, and holding `VELD_MASTER_PASSPHRASE` for a sealed corpus, so a failed test cannot
/// leave a passphrase behind for a plaintext one.
pub struct KeyEnvironment {
    _exclusive: MutexGuard<'static, ()>,
}

impl KeyEnvironment {
    pub fn enter(kind: CorpusKind) -> Self {
        let exclusive = KEY_VARIABLES_IN_USE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        clear_key_variables();
        if kind.is_sealed() {
            env::set_var("VELD_MASTER_PASSPHRASE", KEYSTORE_PASSPHRASE);
        }
        Self {
            _exclusive: exclusive,
        }
    }
}

impl Drop for KeyEnvironment {
    fn drop(&mut self) {
        clear_key_variables();
    }
}

fn clear_key_variables() {
    for name in KEY_VARIABLES {
        env::remove_var(name);
    }
}

/// The lowest Argon2id cost the keystore accepts. The production cost is 256 MiB over three
/// passes and every sealed store open pays one derivation; the keystore format, the passphrase
/// unseal and the record envelope are the production ones.
fn minimum_cost_kdf() -> KdfParams {
    KdfParams {
        m_cost: 8 * 1024,
        t_cost: 1,
        ..KdfParams::production()
    }
}

/// Create the keystore a sealed store opens at `path`, under [`KEYSTORE_PASSPHRASE`].
pub fn install_keystore(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    Keystore::create(KEYSTORE_PASSPHRASE, minimum_cost_kdf())?.save_to_path(path)
}

/// Key material is held apart from every backup on both engines: neither the RocksDB backup
/// engine nor a redb dump carries `keystore.json`, and an in-place restore leaves the live
/// keystore where it is. A restore into a fresh directory therefore installs a copy of the live
/// keystore beside the restored store, as an operator restoring elsewhere must.
pub fn install_restored_keystore(
    engine: Engine,
    live_user_dir: &Path,
    restored_user_dir: &Path,
) -> Result<()> {
    let target = engine.keystore_path(restored_user_dir);
    if let Some(dir) = target.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::copy(engine.keystore_path(live_user_dir), &target)
        .with_context(|| format!("installing the keystore at {}", target.display()))?;
    Ok(())
}

// ─── snapshots ──────────────────────────────────────────────────────────────────

/// One engine's stores reduced to engine-neutral JSON, section by section. Records are compared
/// field by field through their serde form, so a difference names the field that differs.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    sections: BTreeMap<String, Value>,
}

impl Snapshot {
    pub fn insert(&mut self, section: &str, value: Value) {
        self.sections.insert(section.to_string(), value);
    }

    pub fn section(&self, section: &str) -> Option<&Value> {
        self.sections.get(section)
    }

    /// Entries in an object or array section; zero for an absent or scalar one.
    pub fn section_len(&self, section: &str) -> usize {
        match self.section(section) {
            Some(Value::Object(entries)) => entries.len(),
            Some(Value::Array(entries)) => entries.len(),
            _ => 0,
        }
    }

    pub fn merged(mut self, other: Snapshot) -> Snapshot {
        self.sections.extend(other.sections);
        self
    }

    pub fn normalised(&self, normalisations: &[Normalisation]) -> Snapshot {
        let mut snapshot = self.clone();
        for normalisation in normalisations {
            normalisation.apply(&mut snapshot);
        }
        snapshot
    }
}

/// What a primary snapshot asks the store beyond reading every record: the derived lookups
/// each engine answers from its own indexes.
#[derive(Debug, Clone)]
pub struct PrimaryProbes {
    pub user_id: String,
    pub default_prefix: Vec<u8>,
    pub tags: BTreeSet<String>,
    pub episodes: BTreeSet<String>,
    pub external_ids: BTreeSet<String>,
    pub parents: BTreeSet<MemoryId>,
}

/// Every memory (forgotten ones included) read back by id, the vector mappings, the whole `kv`
/// keyspace, the default keyspace under the probe prefix, the interference history, the fact
/// watermark, the storage stats, and the answers to the probe lookups.
///
/// Memories are enumerated with `get_all_ids` + `get` rather than `get_all`, because the
/// engines' `get_all` disagree on forgotten memories (RocksDB drops them, redb returns them),
/// and a backup has to carry soft-deleted records either way.
pub fn primary_snapshot(
    store: &dyn PrimaryMemoryStore,
    probes: &PrimaryProbes,
) -> Result<Snapshot> {
    let mut snapshot = Snapshot::default();

    let mut memories = Map::new();
    for id in store.get_all_ids()? {
        let memory = store
            .get(&id)
            .with_context(|| format!("reading memory {}", id.0))?;
        memories.insert(id.0.to_string(), serde_json::to_value(&memory)?);
    }
    snapshot.insert(PRIMARY_MEMORIES, Value::Object(memories));

    let mut mappings = Map::new();
    for (id, mapping) in store.get_all_vector_mappings()? {
        mappings.insert(id.0.to_string(), serde_json::to_value(&mapping)?);
    }
    snapshot.insert(PRIMARY_VECTOR_MAPPINGS, Value::Object(mappings));

    snapshot.insert(
        PRIMARY_KV,
        rows_value(store.key_value_store().scan_prefix(&[], None)?),
    );
    snapshot.insert(
        PRIMARY_DEFAULT,
        rows_value(
            store
                .default_key_value_store()
                .scan_prefix(&probes.default_prefix, None)?,
        ),
    );

    let (history, total_events) = store.load_all_interference_records()?;
    let history: BTreeMap<String, Vec<InterferenceRecord>> = history.into_iter().collect();
    snapshot.insert(
        PRIMARY_INTERFERENCE,
        json!({ "records": serde_json::to_value(&history)?, "total_events": total_events }),
    );
    snapshot.insert(
        PRIMARY_FACT_WATERMARK,
        json!(store.get_fact_watermark(&probes.user_id)),
    );
    snapshot.insert(PRIMARY_STATS, serde_json::to_value(store.get_stats()?)?);

    let mut external_ids = Map::new();
    for external_id in &probes.external_ids {
        let found = store
            .find_by_external_id(external_id)
            .with_context(|| format!("looking up external id {external_id}"))?;
        external_ids.insert(
            external_id.clone(),
            json!(found.map(|m| m.id.0.to_string())),
        );
    }
    snapshot.insert(PRIMARY_EXTERNAL_IDS, Value::Object(external_ids));

    let mut children = Map::new();
    for parent in &probes.parents {
        children.insert(
            parent.0.to_string(),
            sorted_ids(&store.get_children(parent)?),
        );
    }
    snapshot.insert(PRIMARY_CHILDREN, Value::Object(children));

    let mut tag_hits = Map::new();
    for tag in &probes.tags {
        let found = store.search(SearchCriteria::ByTags(vec![tag.clone()]))?;
        tag_hits.insert(tag.clone(), sorted_ids(&found));
    }
    snapshot.insert(PRIMARY_TAG_SEARCH, Value::Object(tag_hits));

    let mut episode_hits = Map::new();
    for episode in &probes.episodes {
        let found = store.search(SearchCriteria::ByEpisode(episode.clone()))?;
        episode_hits.insert(episode.clone(), sorted_ids(&found));
    }
    snapshot.insert(PRIMARY_EPISODE_SEARCH, Value::Object(episode_hits));

    Ok(snapshot)
}

/// Every entity, the edges each entity is an endpoint of (sorted by edge uuid), every episode,
/// and the graph's own counters.
pub fn graph_snapshot(graph: &GraphMemory) -> Result<Snapshot> {
    let mut snapshot = Snapshot::default();
    let mut entities = Map::new();
    let mut edges_by_node = Map::new();
    for entity in graph.get_all_entities()? {
        let mut edges = graph.get_entity_relationships(&entity.uuid)?;
        edges.sort_by_key(|edge| edge.uuid);
        edges_by_node.insert(entity.uuid.to_string(), serde_json::to_value(&edges)?);
        entities.insert(entity.uuid.to_string(), serde_json::to_value(&entity)?);
    }
    let (episodes, undecodable) = graph.get_all_episodes()?;
    ensure!(
        undecodable == 0,
        "{undecodable} graph episodes did not decode"
    );
    let episodes = episodes
        .iter()
        .map(|episode| Ok((episode.uuid.to_string(), serde_json::to_value(episode)?)))
        .collect::<Result<Map<String, Value>>>()?;

    snapshot.insert(GRAPH_ENTITIES, Value::Object(entities));
    snapshot.insert(GRAPH_EDGES_BY_NODE, Value::Object(edges_by_node));
    snapshot.insert(GRAPH_EPISODES, Value::Object(episodes));
    snapshot.insert(GRAPH_STATS, serde_json::to_value(graph.get_stats()?)?);
    Ok(snapshot)
}

/// Each listed user's audit events in the store's order, and the users the store knows of.
pub fn audit_snapshot(store: &dyn AuditStore, users: &[&str]) -> Result<Snapshot> {
    let mut snapshot = Snapshot::default();
    let mut events = Map::new();
    for user in users {
        events.insert(
            user.to_string(),
            serde_json::to_value(store.list_events(user, usize::MAX)?)?,
        );
    }
    let mut known_users = store.list_user_ids()?;
    known_users.sort();
    snapshot.insert(AUDIT_EVENTS, Value::Object(events));
    snapshot.insert(AUDIT_USERS, json!(known_users));
    Ok(snapshot)
}

fn rows_value(rows: Vec<(Vec<u8>, Vec<u8>)>) -> Value {
    Value::Object(
        rows.into_iter()
            .map(|(key, value)| {
                (
                    key.escape_ascii().to_string(),
                    Value::String(value.escape_ascii().to_string()),
                )
            })
            .collect(),
    )
}

fn sorted_ids(memories: &[Memory]) -> Value {
    let mut ids: Vec<String> = memories.iter().map(|m| m.id.0.to_string()).collect();
    ids.sort();
    json!(ids)
}

// ─── normalisation and comparison ───────────────────────────────────────────────

/// A difference between the engines that is legitimate, removed before they are compared.
/// Applied only across engines: a restore is compared with its own engine's pre-backup
/// snapshot exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalisation {
    /// `GraphMemory::add_entity` assigns every new entity a fresh `Uuid::new_v4()` and stamps
    /// `created_at`/`last_seen_at` with `Utc::now()`; `add_relationship` does the same for an
    /// edge's `uuid` and `created_at`. Two engines fed the same graph therefore hold different
    /// values there. Entities are keyed by name, edges by (from, relation, to) with their
    /// endpoints and episode `entity_refs` rewritten to entity names, and those insert-time
    /// fields are dropped.
    GraphIdentityByName,
}

impl Normalisation {
    pub fn apply(self, snapshot: &mut Snapshot) {
        match self {
            Normalisation::GraphIdentityByName => key_graph_by_name(snapshot),
        }
    }
}

fn key_graph_by_name(snapshot: &mut Snapshot) {
    let Some(Value::Object(entities)) = snapshot.section(GRAPH_ENTITIES).cloned() else {
        return;
    };
    let names: BTreeMap<String, String> = entities
        .iter()
        .filter_map(|(uuid, entity)| {
            entity
                .get("name")
                .and_then(Value::as_str)
                .map(|name| (uuid.clone(), name.to_string()))
        })
        .collect();
    let name_of = |uuid: String| names.get(&uuid).cloned().unwrap_or(uuid);

    let mut entities_by_name = Map::new();
    for (uuid, mut entity) in entities {
        strip_fields(&mut entity, &["uuid", "created_at", "last_seen_at"]);
        entities_by_name.insert(name_of(uuid), entity);
    }
    snapshot.insert(GRAPH_ENTITIES, Value::Object(entities_by_name));

    if let Some(Value::Object(by_node)) = snapshot.section(GRAPH_EDGES_BY_NODE).cloned() {
        let mut edges_by_name = Map::new();
        for (uuid, edges) in by_node {
            let mut edges = match edges {
                Value::Array(edges) => edges,
                other => vec![other],
            };
            for edge in &mut edges {
                strip_fields(edge, &["uuid", "created_at"]);
                for endpoint in ["from_entity", "to_entity"] {
                    if let Some(id) = edge.get(endpoint).cloned() {
                        edge[endpoint] = rename_uuid(&id, &names);
                    }
                }
            }
            edges.sort_by_key(edge_identity);
            edges_by_name.insert(name_of(uuid), Value::Array(edges));
        }
        snapshot.insert(GRAPH_EDGES_BY_NODE, Value::Object(edges_by_name));
    }

    if let Some(Value::Object(episodes)) = snapshot.section(GRAPH_EPISODES).cloned() {
        let mut renamed = Map::new();
        for (uuid, mut episode) in episodes {
            if let Some(Value::Array(refs)) = episode.get("entity_refs").cloned() {
                episode["entity_refs"] =
                    Value::Array(refs.iter().map(|r| rename_uuid(r, &names)).collect());
            }
            renamed.insert(uuid, episode);
        }
        snapshot.insert(GRAPH_EPISODES, Value::Object(renamed));
    }
}

fn strip_fields(value: &mut Value, fields: &[&str]) {
    if let Value::Object(map) = value {
        for field in fields {
            map.remove(*field);
        }
    }
}

fn rename_uuid(value: &Value, names: &BTreeMap<String, String>) -> Value {
    match value.as_str().and_then(|uuid| names.get(uuid)) {
        Some(name) => Value::String(name.clone()),
        None => value.clone(),
    }
}

fn edge_identity(edge: &Value) -> String {
    format!(
        "{}|{}|{}",
        edge["from_entity"], edge["relation_type"], edge["to_entity"]
    )
}

/// Every place two snapshots differ, as `section.path: detail`, sections and keys in order.
pub fn snapshot_differences(left: &Snapshot, right: &Snapshot) -> Vec<String> {
    let mut differences = Vec::new();
    let names: BTreeSet<&String> = left.sections.keys().chain(right.sections.keys()).collect();
    for name in names {
        match (left.sections.get(name), right.sections.get(name)) {
            (Some(l), Some(r)) => value_differences(name, l, r, &mut differences),
            (Some(_), None) => differences.push(format!("{name}: section only on the left")),
            (None, Some(_)) => differences.push(format!("{name}: section only on the right")),
            (None, None) => {}
        }
    }
    differences
}

fn value_differences(path: &str, left: &Value, right: &Value, out: &mut Vec<String>) {
    match (left, right) {
        (Value::Object(l), Value::Object(r)) => {
            let keys: BTreeSet<&String> = l.keys().chain(r.keys()).collect();
            for key in keys {
                let child = format!("{path}.{key}");
                match (l.get(key), r.get(key)) {
                    (Some(a), Some(b)) => value_differences(&child, a, b, out),
                    (Some(a), None) => out.push(format!("{child}: only on the left: {}", brief(a))),
                    (None, Some(b)) => {
                        out.push(format!("{child}: only on the right: {}", brief(b)))
                    }
                    (None, None) => {}
                }
            }
        }
        (Value::Array(l), Value::Array(r)) => {
            if l.len() != r.len() {
                out.push(format!(
                    "{path}: {} entries on the left, {} on the right",
                    l.len(),
                    r.len()
                ));
            }
            for (index, (a, b)) in l.iter().zip(r).enumerate() {
                value_differences(&format!("{path}[{index}]"), a, b, out);
            }
        }
        _ if left != right => out.push(format!("{path}: {} != {}", brief(left), brief(right))),
        _ => {}
    }
}

fn brief(value: &Value) -> String {
    const LIMIT: usize = 160;
    let text = value.to_string();
    if text.chars().count() <= LIMIT {
        text
    } else {
        format!("{}…", text.chars().take(LIMIT).collect::<String>())
    }
}

pub fn assert_snapshots_equal(
    left_label: &str,
    left: &Snapshot,
    right_label: &str,
    right: &Snapshot,
) {
    let differences = snapshot_differences(left, right);
    assert!(
        differences.is_empty(),
        "{left_label} and {right_label} differ in {} place(s), the first {}:\n  {}",
        differences.len(),
        DIFFERENCES_SHOWN.min(differences.len()),
        differences[..DIFFERENCES_SHOWN.min(differences.len())].join("\n  ")
    );
}

/// The mutations must reach every named section, or restoring them would prove nothing about
/// point-in-time semantics there.
pub fn assert_sections_changed(
    label: &str,
    before: &Snapshot,
    after: &Snapshot,
    sections: &[&str],
) {
    let unchanged: Vec<&str> = sections
        .iter()
        .copied()
        .filter(|section| before.section(section) == after.section(section))
        .collect();
    assert!(
        unchanged.is_empty(),
        "{label}: the mutations after the backup left {unchanged:?} unchanged"
    );
}

/// Backups as `(id, user, memory count)`, sorted by id. Both tenant engines list newest first
/// (`VeldBackupEngine::list_backups`, `RedbBackupEngine::list`); the order is asserted where a
/// listing is checked as listed, and normalised here so two listings compare by contents.
pub fn canonical_listing(listing: &[BackupMetadata]) -> Vec<ListedBackup> {
    let mut listed: Vec<ListedBackup> = listing.iter().map(ListedBackup::from).collect();
    listed.sort_by_key(|backup| backup.backup_id);
    listed
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedBackup {
    pub backup_id: u32,
    pub user_id: String,
    pub memory_count: usize,
}

impl From<&BackupMetadata> for ListedBackup {
    fn from(metadata: &BackupMetadata) -> Self {
        Self {
            backup_id: metadata.backup_id,
            user_id: metadata.user_id.clone(),
            memory_count: metadata.memory_count,
        }
    }
}

// ─── artefact scan ──────────────────────────────────────────────────────────────

/// Every file under `root`, recursively, whose bytes contain `needle`.
pub fn files_containing(root: &Path, needle: &[u8]) -> Result<Vec<PathBuf>> {
    let mut hits = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() {
                let bytes =
                    fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                if bytes.windows(needle.len()).any(|window| window == needle) {
                    hits.push(path);
                }
            }
        }
    }
    hits.sort();
    Ok(hits)
}

/// Where a content marker turned up: the live store, the backup artefact, the restored store.
#[derive(Debug, Clone)]
pub struct ArtefactScan {
    pub live: Vec<PathBuf>,
    pub backups: Vec<PathBuf>,
    pub restored: Vec<PathBuf>,
}

impl ArtefactScan {
    pub fn of(live: &Path, backups: &Path, restored: &Path, marker: &str) -> Result<Self> {
        Ok(Self {
            live: files_containing(live, marker.as_bytes())?,
            backups: files_containing(backups, marker.as_bytes())?,
            restored: files_containing(restored, marker.as_bytes())?,
        })
    }
}

fn memories_carrying(snapshot: &Snapshot, marker: &str) -> usize {
    match snapshot.section(PRIMARY_MEMORIES) {
        Some(Value::Object(memories)) => memories
            .values()
            .filter(|memory| {
                memory["experience"]["content"]
                    .as_str()
                    .is_some_and(|content| content.contains(marker))
            })
            .count(),
        _ => 0,
    }
}

// ─── corpus ─────────────────────────────────────────────────────────────────────

fn corpus_epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 9, 1, 8, 0, 0)
        .single()
        .expect("2026-09-01T08:00:00Z is one instant")
}

pub fn corpus_memory_id(ordinal: u32) -> MemoryId {
    MemoryId(Uuid::from_u128(MEMORY_NAMESPACE + u128::from(ordinal)))
}

fn corpus_external_id(ordinal: u32) -> String {
    format!("parity:ext:{ordinal:04}")
}

fn revised_external_id(ordinal: u32) -> String {
    format!("parity:rev:{ordinal:04}")
}

fn corpus_episode_id(slot: u32) -> String {
    format!("parity-episode-{slot:02}")
}

fn kv_key(index: u32) -> Vec<u8> {
    format!("{KV_ROW_PREFIX}{index:03}").into_bytes()
}

/// Never 16 bytes long: RocksDB reads every 16-byte key of its default family as a memory.
fn default_key(index: u32) -> Vec<u8> {
    format!("{DEFAULT_ROW_PREFIX}{index:03}").into_bytes()
}

fn episode_context(ordinal: u32, created_at: DateTime<Utc>) -> RichContext {
    RichContext {
        id: ContextId(Uuid::from_u128(CONTEXT_NAMESPACE + u128::from(ordinal))),
        conversation: Default::default(),
        user: Default::default(),
        project: Default::default(),
        temporal: Default::default(),
        semantic: Default::default(),
        code: Default::default(),
        document: Default::default(),
        environment: Default::default(),
        emotional: Default::default(),
        source: Default::default(),
        episode: EpisodeContext {
            episode_id: Some(corpus_episode_id(ordinal % 10)),
            sequence_number: Some(ordinal / 10),
            episode_type: Some("parity-session".to_string()),
            episode_start: Some(corpus_epoch()),
            is_episode_start: ordinal < 10,
            ..Default::default()
        },
        place: Default::default(),
        who: Default::default(),
        why: Default::default(),
        binding: Default::default(),
        session: Default::default(),
        parent: None,
        embeddings: None,
        decay_rate: 0.1,
        created_at,
        updated_at: created_at,
    }
}

/// Memory `ordinal` of the corpus: tags, entities, an episode for two in three, an embedding
/// for one in four, an external id for one in five, a parent for three in four, and a
/// soft-deleted (`forgotten`) flag for one in thirteen.
pub fn corpus_memory(ordinal: u32) -> Memory {
    let created_at = corpus_epoch() + Duration::minutes(i64::from(ordinal));
    let mut metadata = HashMap::from([
        ("corpus".to_string(), "backup-restore-parity".to_string()),
        ("ordinal".to_string(), ordinal.to_string()),
    ]);
    if ordinal.is_multiple_of(13) {
        metadata.insert("forgotten".to_string(), "true".to_string());
        metadata.insert("forgotten_at".to_string(), created_at.to_rfc3339());
    }
    let experience = Experience {
        experience_type: EXPERIENCE_TYPES[(ordinal % 6) as usize].clone(),
        content: format!(
            "{CONTENT_MARKER} memory {ordinal:04} of the backup-restore parity corpus"
        ),
        context: (!ordinal.is_multiple_of(3)).then(|| episode_context(ordinal, created_at)),
        entities: vec![format!("parity-subject-{:02}", ordinal % 11)],
        tags: vec!["parity".to_string(), format!("bucket-{}", ordinal % 7)],
        metadata,
        embeddings: ordinal.is_multiple_of(4).then(|| {
            (0..8u32)
                .map(|dimension| ((ordinal * 8 + dimension) % 97) as f32 / 97.0)
                .collect()
        }),
        related_memories: if ordinal % 6 == 5 {
            vec![corpus_memory_id(ordinal - 1)]
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    let mut memory = Memory::new(
        corpus_memory_id(ordinal),
        experience,
        0.15 + (ordinal % 8) as f32 * 0.1,
        None,
        None,
        None,
        Some(created_at),
    );
    memory.tier = TIERS[(ordinal % 4) as usize];
    memory.compressed = ordinal.is_multiple_of(17);
    memory.external_id = ordinal
        .is_multiple_of(5)
        .then(|| corpus_external_id(ordinal));
    memory.parent_id =
        (!ordinal.is_multiple_of(4)).then(|| corpus_memory_id(ordinal - ordinal % 4));
    memory.version = 1 + ordinal % 3;
    memory
}

/// The post-backup form of memory `ordinal`: soft-deleted when `ordinal % 9 == 4`, otherwise
/// rewritten with new content, new tags and a new external id.
fn revised_memory(ordinal: u32) -> Memory {
    let mut memory = corpus_memory(ordinal);
    if ordinal % 9 == 4 {
        memory
            .experience
            .metadata
            .insert("forgotten".to_string(), "true".to_string());
        memory.experience.metadata.insert(
            "forgotten_at".to_string(),
            (corpus_epoch() + Duration::days(30)).to_rfc3339(),
        );
    } else {
        memory.experience.content =
            format!("{CONTENT_MARKER} memory {ordinal:04}, revised after the backup");
        memory.experience.tags = vec!["parity".to_string(), "revised".to_string()];
        memory.external_id = Some(revised_external_id(ordinal));
    }
    memory.version += 1;
    memory
}

pub fn corpus_vector_mapping(ordinal: u32) -> VectorMappingEntry {
    let first = ordinal * 3;
    let chunk_ranges = if ordinal.is_multiple_of(10) {
        Some(vec![(0, 40), (40, 80)])
    } else {
        None
    };
    VectorMappingEntry {
        modalities: HashMap::from([(
            Modality::Text,
            ModalityVectors {
                vector_ids: vec![first, first + 1, first + 2],
                dimension: 384,
                chunk_ranges,
            },
        )]),
        created_at: corpus_epoch().timestamp_millis() + i64::from(ordinal),
        version: 1,
    }
}

fn interference_records(ordinal: u32) -> Vec<InterferenceRecord> {
    let at = corpus_epoch() + Duration::hours(i64::from(ordinal));
    vec![
        InterferenceRecord {
            interfering_memory_id: corpus_memory_id(ordinal + 1).0.to_string(),
            similarity: 0.5 + (ordinal % 4) as f32 * 0.1,
            interference_type: InterferenceType::Retroactive,
            strength_change: -0.05,
            timestamp: at,
        },
        InterferenceRecord {
            interfering_memory_id: corpus_memory_id(ordinal + 2).0.to_string(),
            similarity: 0.45,
            interference_type: InterferenceType::Proactive,
            strength_change: -0.02,
            timestamp: at + Duration::seconds(30),
        },
    ]
}

/// The primary-store corpus, written through `PrimaryMemoryStore` only.
pub struct PrimaryCorpus {
    pub size: u32,
    pub memories: Vec<Memory>,
    pub vector_mappings: Vec<(MemoryId, VectorMappingEntry)>,
    pub kv_rows: Vec<(Vec<u8>, Vec<u8>)>,
    pub default_rows: Vec<(Vec<u8>, Vec<u8>)>,
    pub interference: Vec<(String, Vec<InterferenceRecord>)>,
    pub interference_event_count: usize,
    pub fact_watermark: i64,
    pub retrievals: usize,
}

impl PrimaryCorpus {
    pub fn build(size: u32) -> Self {
        Self {
            size,
            memories: (0..size).map(corpus_memory).collect(),
            vector_mappings: (0..size)
                .step_by(2)
                .map(|ordinal| (corpus_memory_id(ordinal), corpus_vector_mapping(ordinal)))
                .collect(),
            kv_rows: (0..40)
                .map(|index| (kv_key(index), format!("kv-value-{index:03}").into_bytes()))
                .collect(),
            default_rows: (0..20)
                .map(|index| {
                    (
                        default_key(index),
                        format!("default-value-{index:03}").into_bytes(),
                    )
                })
                .collect(),
            interference: (0..8)
                .map(|slot| {
                    (
                        corpus_memory_id(slot * 3).0.to_string(),
                        interference_records(slot * 3),
                    )
                })
                .collect(),
            interference_event_count: 23,
            fact_watermark: corpus_epoch().timestamp_millis() + 12_345,
            retrievals: 3,
        }
    }

    pub fn write(&self, store: &dyn PrimaryMemoryStore, user: &str) -> Result<()> {
        for memory in &self.memories {
            store.put_memory(memory)?;
        }
        for (id, mapping) in &self.vector_mappings {
            store.put_vector_mapping(id, mapping)?;
        }
        store.key_value_store().put_many(&self.kv_rows)?;
        store
            .default_key_value_store()
            .put_many(&self.default_rows)?;
        for (memory_id, records) in &self.interference {
            store.save_interference_records(memory_id, records)?;
        }
        store.save_interference_event_count(self.interference_event_count)?;
        store.set_fact_watermark(user, self.fact_watermark);
        for _ in 0..self.retrievals {
            store.increment_retrieval_count()?;
        }
        Ok(())
    }

    /// Adds, revisions, soft deletes, hard deletes, remaps and keyspace edits, touching every
    /// primary section. Hard deletes reach memories with and without an external id, so the
    /// external-id probes see a deleted id answer `None` on both engines.
    pub fn mutations(&self) -> PrimaryMutations {
        let deleted: Vec<u32> = (0..self.size)
            .filter(|ordinal| ordinal % 25 == 7 || ordinal % 25 == 10)
            .collect();
        let added: Vec<u32> = (self.size..self.size + 12).collect();
        PrimaryMutations {
            added: added
                .iter()
                .map(|&ordinal| {
                    let mut memory = corpus_memory(ordinal);
                    memory.experience.tags.push("late-arrival".to_string());
                    memory
                })
                .collect(),
            mapped: added
                .iter()
                .map(|&ordinal| (corpus_memory_id(ordinal), corpus_vector_mapping(ordinal)))
                .collect(),
            revised: (0..self.size)
                .filter(|ordinal| {
                    (ordinal % 9 == 1 || ordinal % 9 == 4) && !deleted.contains(ordinal)
                })
                .map(revised_memory)
                .collect(),
            deleted: deleted
                .iter()
                .map(|&ordinal| corpus_memory_id(ordinal))
                .collect(),
            remapped: (0..self.size)
                .filter(|ordinal| ordinal % 10 == 2 && !deleted.contains(ordinal))
                .map(|ordinal| {
                    (
                        corpus_memory_id(ordinal),
                        vec![9_000 + ordinal, 9_001 + ordinal],
                    )
                })
                .collect(),
            kv_puts: (0..10)
                .map(|index| {
                    (
                        kv_key(index),
                        format!("kv-value-{index:03}-revised").into_bytes(),
                    )
                })
                .chain(
                    (40..45)
                        .map(|index| (kv_key(index), format!("kv-value-{index:03}").into_bytes())),
                )
                .collect(),
            kv_deletes: (30..40).map(kv_key).collect(),
            default_puts: (0..5)
                .map(|index| {
                    (
                        default_key(index),
                        format!("default-value-{index:03}-revised").into_bytes(),
                    )
                })
                .collect(),
            default_deletes: (15..20).map(default_key).collect(),
            interference_cleared: self
                .interference
                .iter()
                .take(2)
                .map(|(memory_id, _)| memory_id.clone())
                .collect(),
            interference_added: vec![(
                corpus_memory_id(40).0.to_string(),
                interference_records(40),
            )],
            interference_event_count: 31,
            fact_watermark: self.fact_watermark + 86_400_000,
            retrievals: 2,
        }
    }

    /// Every tag and episode the corpus and its mutations use, a third of the external ids and
    /// parents plus every one the mutations touch.
    pub fn probes(&self, mutations: &PrimaryMutations, user: &str) -> PrimaryProbes {
        let everything: Vec<&Memory> = self
            .memories
            .iter()
            .chain(&mutations.added)
            .chain(&mutations.revised)
            .collect();
        let tags = everything
            .iter()
            .flat_map(|memory| memory.experience.tags.iter().cloned())
            .collect();
        let episodes = everything
            .iter()
            .filter_map(|memory| {
                memory
                    .experience
                    .context
                    .as_ref()
                    .and_then(|context| context.episode.episode_id.clone())
            })
            .collect();
        let every_external_id: BTreeSet<String> = everything
            .iter()
            .filter_map(|memory| memory.external_id.clone())
            .collect();
        let mut external_ids: BTreeSet<String> =
            every_external_id.iter().step_by(3).cloned().collect();
        for revised in &mutations.revised {
            external_ids.extend(revised.external_id.clone());
            external_ids.extend(
                self.memories
                    .iter()
                    .find(|memory| memory.id == revised.id)
                    .and_then(|memory| memory.external_id.clone()),
            );
        }
        let every_parent: BTreeSet<MemoryId> = everything
            .iter()
            .filter_map(|memory| memory.parent_id.clone())
            .collect();
        let mut parents: BTreeSet<MemoryId> = every_parent.iter().step_by(3).cloned().collect();
        parents.extend(
            mutations
                .added
                .iter()
                .filter_map(|memory| memory.parent_id.clone()),
        );
        PrimaryProbes {
            user_id: user.to_string(),
            default_prefix: DEFAULT_ROW_PREFIX.as_bytes().to_vec(),
            tags,
            episodes,
            external_ids,
            parents,
        }
    }
}

/// What happens to the primary store after the backup, applied identically to both engines.
pub struct PrimaryMutations {
    pub added: Vec<Memory>,
    pub mapped: Vec<(MemoryId, VectorMappingEntry)>,
    pub revised: Vec<Memory>,
    pub deleted: Vec<MemoryId>,
    pub remapped: Vec<(MemoryId, Vec<u32>)>,
    pub kv_puts: Vec<(Vec<u8>, Vec<u8>)>,
    pub kv_deletes: Vec<Vec<u8>>,
    pub default_puts: Vec<(Vec<u8>, Vec<u8>)>,
    pub default_deletes: Vec<Vec<u8>>,
    pub interference_cleared: Vec<String>,
    pub interference_added: Vec<(String, Vec<InterferenceRecord>)>,
    pub interference_event_count: usize,
    pub fact_watermark: i64,
    pub retrievals: usize,
}

impl PrimaryMutations {
    pub fn apply(&self, store: &dyn PrimaryMemoryStore, user: &str) -> Result<()> {
        for memory in &self.added {
            store.put_memory(memory)?;
        }
        for (id, mapping) in &self.mapped {
            store.put_vector_mapping(id, mapping)?;
        }
        for memory in &self.revised {
            store.update_memory(memory)?;
        }
        for id in &self.deleted {
            store.delete_memory(id)?;
        }
        for (id, vector_ids) in &self.remapped {
            store.update_vector_mapping(id, vector_ids.clone())?;
        }
        store.key_value_store().put_many(&self.kv_puts)?;
        store.key_value_store().delete_many(&self.kv_deletes)?;
        store
            .default_key_value_store()
            .put_many(&self.default_puts)?;
        store
            .default_key_value_store()
            .delete_many(&self.default_deletes)?;
        for memory_id in &self.interference_cleared {
            store.delete_interference_records(memory_id)?;
        }
        for (memory_id, records) in &self.interference_added {
            store.save_interference_records(memory_id, records)?;
        }
        store.save_interference_event_count(self.interference_event_count)?;
        store.set_fact_watermark(user, self.fact_watermark);
        for _ in 0..self.retrievals {
            store.increment_retrieval_count()?;
        }
        Ok(())
    }
}

/// A primary corpus, its mutations and the lookups a snapshot asks of both.
pub struct PrimaryScenario {
    pub corpus: PrimaryCorpus,
    pub mutations: PrimaryMutations,
    pub probes: PrimaryProbes,
}

impl PrimaryScenario {
    pub fn build(size: u32) -> Self {
        let corpus = PrimaryCorpus::build(size);
        let mutations = corpus.mutations();
        let probes = corpus.probes(&mutations, TENANT);
        Self {
            corpus,
            mutations,
            probes,
        }
    }
}

fn graph_entity_name(ordinal: u32) -> String {
    format!("parity-entity-{ordinal:02}")
}

fn late_entity_name(ordinal: u32) -> String {
    format!("parity-entity-late-{ordinal}")
}

fn graph_episode_uuid(slot: u32) -> Uuid {
    Uuid::from_u128(EPISODE_NAMESPACE + u128::from(slot))
}

/// A proper noun, so `add_entity` never merges two corpus names through the stemmed index;
/// `uuid` and the timestamps are assigned by `add_entity` itself.
fn graph_entity(name: String, ordinal: u32) -> EntityNode {
    EntityNode {
        uuid: Uuid::nil(),
        name,
        labels: vec![ENTITY_LABELS[(ordinal % 3) as usize].clone()],
        created_at: corpus_epoch(),
        last_seen_at: corpus_epoch(),
        mention_count: 1,
        summary: format!("parity graph entity {ordinal:02}"),
        attributes: HashMap::from([("ordinal".to_string(), ordinal.to_string())]),
        name_embedding: None,
        salience: 0.25 + (ordinal % 5) as f32 * 0.1,
        is_proper_noun: true,
        pii_classification: PiiClassification::Clean,
    }
}

/// An edge between two entities by name. At most one edge joins any pair: `add_relationship`
/// strengthens an existing edge between the pair (in either direction) instead of adding one,
/// and that path stamps `last_activated` with the wall clock.
#[derive(Debug, Clone)]
pub struct EdgeSpec {
    pub from: String,
    pub to: String,
    pub relation: RelationType,
    pub strength: f32,
    pub at: DateTime<Utc>,
    pub episode: Option<Uuid>,
}

impl EdgeSpec {
    pub fn key(&self) -> String {
        format!("{}|{:?}|{}", self.from, self.relation, self.to)
    }

    fn relationship(&self, from: Uuid, to: Uuid) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::nil(),
            from_entity: from,
            to_entity: to,
            relation_type: self.relation.clone(),
            strength: self.strength,
            created_at: self.at,
            valid_at: self.at,
            invalidated_at: None,
            source_episode_id: self.episode,
            context: format!("parity edge {}", self.key()),
            last_activated: self.at,
            activation_count: 0,
            ltp_status: Default::default(),
            tier: Default::default(),
            activation_timestamps: None,
            entity_confidence: Some(0.8),
            created_by: Default::default(),
            forward_strength: 0.0,
            backward_strength: 0.0,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpisodeSpec {
    pub uuid: Uuid,
    pub name: String,
    pub at: DateTime<Utc>,
    pub entities: Vec<String>,
    pub source: EpisodeSource,
}

/// The uuids one engine's graph assigned to the corpus's entities and edges.
#[derive(Debug, Clone, Default)]
pub struct GraphHandles {
    pub entities: BTreeMap<String, Uuid>,
    pub edges: BTreeMap<String, Uuid>,
}

impl GraphHandles {
    fn entity(&self, name: &str) -> Result<Uuid> {
        self.entities
            .get(name)
            .copied()
            .with_context(|| format!("no graph entity named {name} was written"))
    }
}

/// The graph corpus, written through `GraphMemory`, the graph engine both backends run.
#[derive(Debug, Clone)]
pub struct GraphCorpus {
    pub entities: Vec<EntityNode>,
    pub episodes: Vec<EpisodeSpec>,
    pub edges: Vec<EdgeSpec>,
}

impl GraphCorpus {
    pub fn build() -> Self {
        let epoch = corpus_epoch();
        let entities = (0..GRAPH_ENTITY_COUNT)
            .map(|ordinal| graph_entity(graph_entity_name(ordinal), ordinal))
            .collect();
        let episodes = (0..4u32)
            .map(|slot| EpisodeSpec {
                uuid: graph_episode_uuid(slot),
                name: format!("parity-graph-episode-{slot}"),
                at: epoch + Duration::hours(i64::from(slot)),
                entities: (slot * 3..slot * 3 + 3).map(graph_entity_name).collect(),
                source: EPISODE_SOURCES[slot as usize].clone(),
            })
            .collect();
        let mut pairs = BTreeSet::new();
        let mut edges = Vec::new();
        for attempt in 0..GRAPH_EDGE_ATTEMPTS {
            let a = attempt % GRAPH_ENTITY_COUNT;
            let b = (attempt * 7 + 5 + attempt / GRAPH_ENTITY_COUNT) % GRAPH_ENTITY_COUNT;
            let pair = (a.min(b), a.max(b));
            if a == b || !pairs.insert(pair) {
                continue;
            }
            edges.push(EdgeSpec {
                from: graph_entity_name(pair.0),
                to: graph_entity_name(pair.1),
                relation: RELATIONS[(attempt % 6) as usize].clone(),
                strength: 0.2 + (attempt % 7) as f32 * 0.1,
                at: epoch + Duration::minutes(i64::from(attempt)),
                episode: attempt
                    .is_multiple_of(3)
                    .then(|| graph_episode_uuid(attempt % 4)),
            });
        }
        Self {
            entities,
            episodes,
            edges,
        }
    }

    pub fn write(&self, graph: &GraphMemory, handles: &mut GraphHandles) -> Result<()> {
        for entity in &self.entities {
            let uuid = graph.add_entity(entity.clone())?;
            ensure!(
                !handles.entities.values().any(|existing| *existing == uuid),
                "GraphMemory merged {} into an entity already in the corpus",
                entity.name
            );
            handles.entities.insert(entity.name.clone(), uuid);
        }
        for episode in &self.episodes {
            let entity_refs = episode
                .entities
                .iter()
                .map(|name| handles.entity(name))
                .collect::<Result<Vec<_>>>()?;
            graph.add_episode(EpisodicNode {
                uuid: episode.uuid,
                name: episode.name.clone(),
                content: format!("{} content", episode.name),
                valid_at: episode.at,
                created_at: episode.at,
                entity_refs,
                source: episode.source.clone(),
                metadata: HashMap::from([(
                    "corpus".to_string(),
                    "backup-restore-parity".to_string(),
                )]),
            })?;
        }
        for edge in &self.edges {
            let uuid = graph.add_relationship(
                edge.relationship(handles.entity(&edge.from)?, handles.entity(&edge.to)?),
            )?;
            handles.edges.insert(edge.key(), uuid);
        }
        Ok(())
    }
}

/// What happens to the graph after the backup: new entities, edges and an episode, and three
/// corpus edges deleted.
#[derive(Debug, Clone)]
pub struct GraphMutations {
    pub added: GraphCorpus,
    pub removed_edges: Vec<String>,
}

impl GraphMutations {
    pub fn after(corpus: &GraphCorpus) -> Self {
        let at = corpus_epoch() + Duration::days(7);
        let late_episode = graph_episode_uuid(10);
        Self {
            added: GraphCorpus {
                entities: (0..3)
                    .map(|ordinal| {
                        graph_entity(late_entity_name(ordinal), GRAPH_ENTITY_COUNT + ordinal)
                    })
                    .collect(),
                episodes: vec![EpisodeSpec {
                    uuid: late_episode,
                    name: "parity-graph-episode-late".to_string(),
                    at,
                    entities: (0..3).map(late_entity_name).collect(),
                    source: EpisodeSource::Event,
                }],
                edges: (0..3)
                    .map(|ordinal| EdgeSpec {
                        from: late_entity_name(ordinal),
                        to: graph_entity_name(ordinal * 5),
                        relation: RelationType::RelatedTo,
                        strength: 0.6,
                        at: at + Duration::minutes(i64::from(ordinal)),
                        episode: Some(late_episode),
                    })
                    .collect(),
            },
            removed_edges: corpus.edges.iter().take(3).map(EdgeSpec::key).collect(),
        }
    }

    pub fn apply(&self, graph: &GraphMemory, handles: &mut GraphHandles) -> Result<()> {
        self.added.write(graph, handles)?;
        for key in &self.removed_edges {
            let uuid = handles
                .edges
                .get(key)
                .copied()
                .with_context(|| format!("no edge {key} was written"))?;
            ensure!(
                graph.delete_relationship(&uuid)?,
                "edge {key} was not there to delete"
            );
        }
        Ok(())
    }
}

fn audit_entry(ordinal: u32) -> AuditLogEntry {
    AuditLogEntry {
        timestamp: corpus_epoch() + Duration::seconds(i64::from(ordinal)),
        event_type: AUDIT_EVENT_TYPES[(ordinal % 4) as usize].to_string(),
        memory_id: corpus_memory_id(ordinal).0.to_string(),
        details: format!("parity audit row {ordinal:03}"),
    }
}

/// Audit events for the tenant and a neighbour, written through `AuditStore`.
#[derive(Debug, Clone)]
pub struct AuditCorpus {
    pub events: Vec<(String, AuditLogEntry)>,
}

impl AuditCorpus {
    pub fn build() -> Self {
        Self {
            events: (0..AUDIT_EVENT_COUNT)
                .map(|ordinal| (TENANT.to_string(), audit_entry(ordinal)))
                .chain((0..5).map(|ordinal| (NEIGHBOUR.to_string(), audit_entry(1_000 + ordinal))))
                .collect(),
        }
    }

    pub fn write(&self, store: &dyn AuditStore) -> Result<()> {
        for (user, event) in &self.events {
            store.append_event(user, event)?;
        }
        Ok(())
    }
}

/// New tenant events after the backup, then a rotation down to the newest
/// [`AUDIT_KEPT_AFTER_ROTATION`], with a retention cut at the corpus epoch so only the count
/// cap removes anything.
#[derive(Debug, Clone)]
pub struct AuditMutations {
    pub appended: Vec<(String, AuditLogEntry)>,
    pub keep_newest: usize,
}

impl AuditMutations {
    pub fn after() -> Self {
        Self {
            appended: (AUDIT_EVENT_COUNT..AUDIT_EVENT_COUNT + 5)
                .map(|ordinal| (TENANT.to_string(), audit_entry(ordinal)))
                .collect(),
            keep_newest: AUDIT_KEPT_AFTER_ROTATION,
        }
    }

    pub fn apply(&self, store: &dyn AuditStore) -> Result<()> {
        for (user, event) in &self.appended {
            store.append_event(user, event)?;
        }
        let removed = store.rotate_events(TENANT, self.keep_newest, corpus_epoch())?;
        let expected = AUDIT_EVENT_COUNT as usize + self.appended.len() - self.keep_newest;
        ensure!(
            removed == expected,
            "audit rotation removed {removed} events, expected {expected}"
        );
        Ok(())
    }
}

/// One tenant's whole corpus: primary store, graph and audit log, with their mutations.
pub struct TenantScenario {
    pub primary: PrimaryScenario,
    pub graph: GraphCorpus,
    pub graph_mutations: GraphMutations,
    pub audit: AuditCorpus,
    pub audit_mutations: AuditMutations,
}

impl TenantScenario {
    pub fn build(size: u32) -> Self {
        let graph = GraphCorpus::build();
        let graph_mutations = GraphMutations::after(&graph);
        Self {
            primary: PrimaryScenario::build(size),
            graph,
            graph_mutations,
            audit: AuditCorpus::build(),
            audit_mutations: AuditMutations::after(),
        }
    }

    pub fn write(&self, tenant: &TenantStores, graph: &mut GraphHandles) -> Result<()> {
        self.primary.corpus.write(tenant.primary.store(), TENANT)?;
        self.graph.write(&tenant.graph, graph)?;
        self.audit.write(tenant.audit())
    }

    pub fn mutate(&self, tenant: &TenantStores, graph: &mut GraphHandles) -> Result<()> {
        self.primary
            .mutations
            .apply(tenant.primary.store(), TENANT)?;
        self.graph_mutations.apply(&tenant.graph, graph)?;
        self.audit_mutations.apply(tenant.audit())
    }
}

// ─── engines ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    RocksDb,
    Redb,
}

impl Engine {
    pub const BOTH: [Engine; 2] = [Engine::RocksDb, Engine::Redb];

    pub fn label(self) -> &'static str {
        match self {
            Engine::RocksDb => "rocksdb",
            Engine::Redb => "redb",
        }
    }

    /// RocksDB keeps a tenant's keystore inside its database directory, redb beside its file.
    pub fn keystore_path(self, user_dir: &Path) -> PathBuf {
        match self {
            Engine::RocksDb => user_dir.join("storage").join(KEYSTORE_FILE),
            Engine::Redb => user_dir.join(KEYSTORE_FILE),
        }
    }

    pub fn open_primary(self, user_dir: &Path) -> Result<PrimaryHandle> {
        Ok(match self {
            Engine::RocksDb => {
                PrimaryHandle::RocksDb(RocksDbPrimaryMemoryStore::open(user_dir, None)?)
            }
            Engine::Redb => PrimaryHandle::Redb(RedbPrimaryMemoryStore::open(user_dir)?),
        })
    }

    /// The `target_path` that makes `BackupStore::restore_backup` land where
    /// [`Engine::open_primary`] of `user_dir` reads. The engines name different things with it:
    /// RocksDB restores the database directory itself (`<user>/storage`), redb the directory a
    /// `storage.redb` is created in (`<user>`).
    pub fn backup_store_restore_target(self, user_dir: &Path) -> PathBuf {
        match self {
            Engine::RocksDb => user_dir.join("storage"),
            Engine::Redb => user_dir.to_path_buf(),
        }
    }
}

/// A tenant's primary store on one engine.
pub enum PrimaryHandle {
    RocksDb(RocksDbPrimaryMemoryStore),
    Redb(RedbPrimaryMemoryStore),
}

impl PrimaryHandle {
    pub fn store(&self) -> &dyn PrimaryMemoryStore {
        match self {
            PrimaryHandle::RocksDb(store) => store,
            PrimaryHandle::Redb(store) => store,
        }
    }

    pub fn encrypts_records(&self) -> bool {
        match self {
            PrimaryHandle::RocksDb(store) => store.inner().encrypts_records(),
            PrimaryHandle::Redb(store) => store.encrypts_records(),
        }
    }

    /// The engine's `BackupStore` over this store's live handle, keeping its backups under
    /// `root`.
    pub fn backup_store(&self, root: &Path) -> Result<PrimaryBackupStore> {
        Ok(match self {
            PrimaryHandle::RocksDb(store) => PrimaryBackupStore::RocksDb(RocksDbBackupStore::new(
                VeldBackupEngine::new(root.to_path_buf())?,
                store.inner().db(),
            )),
            PrimaryHandle::Redb(store) => {
                PrimaryBackupStore::Redb(RedbBackupStore::from_database(store.database(), root))
            }
        })
    }
}

/// One engine's `BackupStore`, with the integrity check that engine offers beside it.
pub enum PrimaryBackupStore {
    RocksDb(RocksDbBackupStore),
    Redb(RedbBackupStore),
}

impl PrimaryBackupStore {
    pub fn store(&self) -> &dyn BackupStore {
        match self {
            PrimaryBackupStore::RocksDb(store) => store,
            PrimaryBackupStore::Redb(store) => store,
        }
    }

    /// The trait has no verify. RocksDB's adapter exposes the engine's checksum check;
    /// redb's `list_backups` is its check, listing a backup only when the dump's trailer
    /// digest matches the checksum its metadata recorded.
    pub fn verify(&self, user: &str, backup_id: u32) -> Result<bool> {
        match self {
            PrimaryBackupStore::RocksDb(store) => store.inner().verify_backup(user, backup_id),
            PrimaryBackupStore::Redb(store) => Ok(store
                .list_backups(user)?
                .iter()
                .any(|metadata| metadata.backup_id == backup_id)),
        }
    }
}

enum TenantHandles {
    RocksDb {
        shared: Arc<rocksdb::DB>,
        audit: RocksDbAuditStore,
    },
    Redb {
        graph: Arc<::redb::Database>,
        shared: Arc<::redb::Database>,
        audit: RedbAuditStore,
    },
}

/// A tenant's stores laid out as the server lays them out: `<base>/<user>` holds the primary
/// store and `graph/`, `<base>` the shared store (whose `audit` family is the RocksDB audit
/// log) and, on redb, `audit.redb`.
pub struct TenantStores {
    pub primary: PrimaryHandle,
    pub graph: GraphMemory,
    handles: TenantHandles,
}

impl TenantStores {
    pub fn open(engine: Engine, base: &Path) -> Result<Self> {
        let user_dir = base.join(TENANT);
        let primary = engine.open_primary(&user_dir)?;
        Ok(match engine {
            Engine::RocksDb => {
                let shared = open_rocksdb_shared(&base.join(SHARED_STORE))?;
                Self {
                    primary,
                    graph: GraphMemory::new(&user_dir.join(GRAPH_STORE), None)?,
                    handles: TenantHandles::RocksDb {
                        audit: RocksDbAuditStore::new(Arc::clone(&shared), AUDIT_FAMILY),
                        shared,
                    },
                }
            }
            Engine::Redb => {
                let graph_database = open_graph_database(&user_dir.join(GRAPH_STORE))?;
                Self {
                    primary,
                    graph: GraphMemory::with_store(Arc::new(RedbColumnStore::new(Arc::clone(
                        &graph_database,
                    ))))?,
                    handles: TenantHandles::Redb {
                        graph: graph_database,
                        shared: open_shared_database(base)?,
                        audit: RedbAuditStore::open(base)?,
                    },
                }
            }
        })
    }

    pub fn audit(&self) -> &dyn AuditStore {
        match &self.handles {
            TenantHandles::RocksDb { audit, .. } => audit,
            TenantHandles::Redb { audit, .. } => audit,
        }
    }

    pub fn snapshot(&self, probes: &PrimaryProbes) -> Result<Snapshot> {
        Ok(primary_snapshot(self.primary.store(), probes)?
            .merged(graph_snapshot(&self.graph)?)
            .merged(audit_snapshot(self.audit(), &[TENANT, NEIGHBOUR])?))
    }
}

fn open_rocksdb_shared(path: &Path) -> Result<Arc<rocksdb::DB>> {
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    options.create_missing_column_families(true);
    let database = rocksdb::DB::open_cf(&options, path, ["default", AUDIT_FAMILY])
        .with_context(|| format!("opening the rocksdb shared store at {}", path.display()))?;
    Ok(Arc::new(database))
}

/// The tenant backup each engine's `/api/backup/*` handlers run.
pub enum TenantBackups {
    RocksDb(VeldBackupEngine),
    Redb(RedbBackupEngine),
}

impl TenantBackups {
    pub fn open(engine: Engine, root: &Path) -> Result<Self> {
        Ok(match engine {
            Engine::RocksDb => TenantBackups::RocksDb(VeldBackupEngine::new(root.to_path_buf())?),
            Engine::Redb => TenantBackups::Redb(RedbBackupEngine::new(root.to_path_buf())?),
        })
    }

    /// RocksDB: the primary database through the BackupEngine, the shared store (with the audit
    /// family) and the graph as checkpoints. redb: the tenant's and the shared stores' dumps.
    pub fn create(&self, tenant: &TenantStores, base: &Path) -> Result<BackupMetadata> {
        match (self, &tenant.primary, &tenant.handles) {
            (
                TenantBackups::RocksDb(engine),
                PrimaryHandle::RocksDb(primary),
                TenantHandles::RocksDb { shared, .. },
            ) => {
                let primary_db = primary.inner().db();
                let graph_db = tenant
                    .graph
                    .rocksdb_for_backup()
                    .context("the rocksdb tenant's graph holds no rocksdb handle")?;
                engine.create_comprehensive_backup_with_graph(
                    &primary_db,
                    TENANT,
                    &[SecondaryStoreRef {
                        name: SHARED_STORE,
                        db: shared,
                    }],
                    Some(graph_db),
                )
            }
            (
                TenantBackups::Redb(engine),
                PrimaryHandle::Redb(primary),
                TenantHandles::Redb {
                    graph,
                    shared,
                    audit,
                },
            ) => {
                let primary_db = primary.database();
                let audit_db = audit.database();
                engine.create(
                    base,
                    TENANT,
                    &LiveRedb {
                        primary: Some(&primary_db),
                        graph: Some(graph),
                        audit: Some(&audit_db),
                        shared: Some(shared),
                    },
                    &IntentLogBackupSpec::per_user(base.join(TENANT)),
                )
            }
            _ => bail!("the backup engine and the tenant's stores run on different engines"),
        }
    }

    pub fn verify(&self, user: &str, backup_id: u32) -> Result<bool> {
        match self {
            TenantBackups::RocksDb(engine) => engine.verify_backup(user, backup_id),
            TenantBackups::Redb(engine) => engine.verify(user, backup_id).map(|_| true),
        }
    }

    pub fn list(&self, user: &str) -> Result<Vec<BackupMetadata>> {
        match self {
            TenantBackups::RocksDb(engine) => engine.list_backups(user),
            TenantBackups::Redb(engine) => engine.list(user),
        }
    }

    pub fn purge(&self, user: &str, keep_count: usize) -> Result<usize> {
        match self {
            TenantBackups::RocksDb(engine) => engine.purge_old_backups(user, keep_count),
            TenantBackups::Redb(engine) => engine.purge(user, keep_count),
        }
    }

    /// Restore every captured store into `target_base`, laid out as [`TenantStores::open`]
    /// reads it, shared stores included: the server's own restore leaves them in place because
    /// they hold every tenant's rows, but a fresh directory has none to keep.
    pub fn restore(&self, user: &str, backup_id: u32, target_base: &Path) -> Result<()> {
        match self {
            TenantBackups::RocksDb(engine) => {
                let user_dir = target_base.join(user);
                fs::create_dir_all(&user_dir)?;
                let shared = target_base.join(SHARED_STORE);
                let graph = user_dir.join(GRAPH_STORE).join(GRAPH_STORE);
                let restored = engine.restore_comprehensive_backup(
                    user,
                    Some(backup_id),
                    &user_dir.join("storage"),
                    &[
                        (SHARED_STORE, shared.as_path()),
                        (GRAPH_STORE, graph.as_path()),
                    ],
                )?;
                for component in [SHARED_STORE, GRAPH_STORE] {
                    ensure!(
                        restored.iter().any(|name| name == component),
                        "rocksdb backup {backup_id} restored {restored:?}, without {component}"
                    );
                }
            }
            TenantBackups::Redb(engine) => {
                let report = engine.restore_into(
                    user,
                    backup_id,
                    target_base,
                    &RestoreOptions::default(),
                    true,
                )?;
                ensure!(
                    report.not_restored.is_empty() && report.imported.len() == 2,
                    "redb backup {backup_id} imported {} unit(s) and left {:?} unrestored",
                    report.imported.len(),
                    report.not_restored
                );
            }
        }
        Ok(())
    }
}

// ─── round trips ────────────────────────────────────────────────────────────────

/// One engine's backup, mutate, restore cycle.
pub struct RoundTrip {
    pub engine: Engine,
    pub before: Snapshot,
    pub after: Snapshot,
    pub restored: Snapshot,
    pub metadata: BackupMetadata,
    pub scan: ArtefactScan,
}

fn ensure_record_encryption(
    engine: Engine,
    primary: &PrimaryHandle,
    kind: CorpusKind,
    which: &str,
) -> Result<()> {
    ensure!(
        primary.encrypts_records() == kind.is_sealed(),
        "the {which} {} store opened with record encryption {}, the {kind:?} corpus needs {}",
        engine.label(),
        primary.encrypts_records(),
        kind.is_sealed()
    );
    Ok(())
}

fn backup_store_round_trip(
    engine: Engine,
    kind: CorpusKind,
    scenario: &PrimaryScenario,
) -> Result<RoundTrip> {
    let live = TempDir::new()?;
    let backups = TempDir::new()?;
    let target = TempDir::new()?;
    let user_dir = live.path().join(TENANT);
    let restored_dir = target.path().join(TENANT);

    if kind.is_sealed() {
        install_keystore(&engine.keystore_path(&user_dir))?;
    }
    let primary = engine.open_primary(&user_dir)?;
    ensure_record_encryption(engine, &primary, kind, "live")?;
    scenario.corpus.write(primary.store(), TENANT)?;
    let before = primary_snapshot(primary.store(), &scenario.probes)?;

    let backup_store = primary.backup_store(backups.path())?;
    let metadata = backup_store.store().create_backup(TENANT)?;
    ensure!(
        backup_store.verify(TENANT, metadata.backup_id)?,
        "{} BackupStore backup {} does not verify",
        engine.label(),
        metadata.backup_id
    );
    let listed: Vec<u32> = backup_store
        .store()
        .list_backups(TENANT)?
        .iter()
        .map(|listed| listed.backup_id)
        .collect();
    ensure!(
        listed == [metadata.backup_id],
        "{} BackupStore lists {listed:?} after creating backup {}",
        engine.label(),
        metadata.backup_id
    );

    scenario.mutations.apply(primary.store(), TENANT)?;
    let after = primary_snapshot(primary.store(), &scenario.probes)?;

    fs::create_dir_all(&restored_dir)?;
    backup_store.store().restore_backup(
        TENANT,
        metadata.backup_id,
        &engine.backup_store_restore_target(&restored_dir),
    )?;
    if kind.is_sealed() {
        install_restored_keystore(engine, &user_dir, &restored_dir)?;
    }
    let restored_primary = engine.open_primary(&restored_dir)?;
    ensure_record_encryption(engine, &restored_primary, kind, "restored")?;
    let restored = primary_snapshot(restored_primary.store(), &scenario.probes)?;

    let scan = ArtefactScan::of(live.path(), backups.path(), target.path(), CONTENT_MARKER)?;
    Ok(RoundTrip {
        engine,
        before,
        after,
        restored,
        metadata,
        scan,
    })
}

fn tenant_round_trip(
    engine: Engine,
    kind: CorpusKind,
    scenario: &TenantScenario,
) -> Result<RoundTrip> {
    let live = TempDir::new()?;
    let backups = TempDir::new()?;
    let target = TempDir::new()?;
    let user_dir = live.path().join(TENANT);
    let restored_dir = target.path().join(TENANT);

    if kind.is_sealed() {
        install_keystore(&engine.keystore_path(&user_dir))?;
    }
    let tenant = TenantStores::open(engine, live.path())?;
    ensure_record_encryption(engine, &tenant.primary, kind, "live")?;
    let mut graph = GraphHandles::default();
    scenario.write(&tenant, &mut graph)?;
    let before = tenant.snapshot(&scenario.primary.probes)?;

    let tenant_backups = TenantBackups::open(engine, backups.path())?;
    let metadata = tenant_backups.create(&tenant, live.path())?;
    ensure!(
        tenant_backups.verify(TENANT, metadata.backup_id)?,
        "{} tenant backup {} does not verify",
        engine.label(),
        metadata.backup_id
    );

    scenario.mutate(&tenant, &mut graph)?;
    let after = tenant.snapshot(&scenario.primary.probes)?;

    tenant_backups.restore(TENANT, metadata.backup_id, target.path())?;
    if kind.is_sealed() {
        install_restored_keystore(engine, &user_dir, &restored_dir)?;
    }
    let restored_tenant = TenantStores::open(engine, target.path())?;
    ensure_record_encryption(engine, &restored_tenant.primary, kind, "restored")?;
    let restored = restored_tenant.snapshot(&scenario.primary.probes)?;

    let scan = ArtefactScan::of(live.path(), backups.path(), target.path(), CONTENT_MARKER)?;
    Ok(RoundTrip {
        engine,
        before,
        after,
        restored,
        metadata,
        scan,
    })
}

/// Point-in-time semantics on one engine: the mutations changed every named section, and the
/// restore returned the pre-backup state exactly, with no normalisation.
fn assert_point_in_time(trip: &RoundTrip, kind: CorpusKind, mutated_sections: &[&str]) {
    let label = trip.engine.label();
    assert_sections_changed(
        &format!("{label} live store"),
        &trip.before,
        &trip.after,
        mutated_sections,
    );
    assert_snapshots_equal(
        &format!("{label} before the backup"),
        &trip.before,
        &format!("{label} restored"),
        &trip.restored,
    );
    assert_artefacts(trip, kind);
}

/// A plaintext corpus must show its marker in each engine's backup, or the scan cannot read that
/// backup format and a sealed corpus's clean scan would prove nothing. A sealed corpus must show
/// it nowhere, and every restored record must open under the keystore to its content.
fn assert_artefacts(trip: &RoundTrip, kind: CorpusKind) {
    let label = trip.engine.label();
    match kind {
        CorpusKind::Plaintext => assert!(
            !trip.scan.backups.is_empty(),
            "{label}: the plaintext corpus's content marker appears in no backup file"
        ),
        CorpusKind::Sealed => {
            assert!(
                trip.scan.live.is_empty(),
                "{label}: the live sealed store holds plaintext content in {:?}",
                trip.scan.live
            );
            assert!(
                trip.scan.backups.is_empty(),
                "{label}: the backup artefact holds plaintext content in {:?}",
                trip.scan.backups
            );
            assert!(
                trip.scan.restored.is_empty(),
                "{label}: the restored store holds plaintext content in {:?}",
                trip.scan.restored
            );
            assert_eq!(
                memories_carrying(&trip.restored, CONTENT_MARKER),
                trip.restored.section_len(PRIMARY_MEMORIES),
                "{label}: a restored sealed record did not open to its plaintext content"
            );
        }
    }
}

fn assert_engines_agree(rocks: &RoundTrip, redb: &RoundTrip, normalisations: &[Normalisation]) {
    assert_snapshots_equal(
        "rocksdb before the backup",
        &rocks.before.normalised(normalisations),
        "redb before the backup",
        &redb.before.normalised(normalisations),
    );
    assert_snapshots_equal(
        "rocksdb restored",
        &rocks.restored.normalised(normalisations),
        "redb restored",
        &redb.restored.normalised(normalisations),
    );
    assert_eq!(
        rocks.metadata.memory_count, redb.metadata.memory_count,
        "the engines' backup metadata count different memories"
    );
}

fn run_on_both_engines(
    surface: &str,
    round_trip: impl Fn(Engine) -> Result<RoundTrip>,
) -> [RoundTrip; 2] {
    Engine::BOTH.map(|engine| {
        round_trip(engine)
            .unwrap_or_else(|e| panic!("{} {surface} round trip failed: {e:#}", engine.label()))
    })
}

fn backup_store_parity(kind: CorpusKind) {
    let _keys = KeyEnvironment::enter(kind);
    let scenario = PrimaryScenario::build(PRIMARY_CORPUS_SIZE);
    let [rocks, redb] = run_on_both_engines("BackupStore", |engine| {
        backup_store_round_trip(engine, kind, &scenario)
    });
    let corpus_size = scenario.corpus.memories.len();
    for trip in [&rocks, &redb] {
        let label = trip.engine.label();
        assert_eq!(
            trip.before.section_len(PRIMARY_MEMORIES),
            corpus_size,
            "{label}: the corpus did not land"
        );
        assert_eq!(
            trip.metadata.memory_count, corpus_size,
            "{label}: the backup metadata miscounts the memories"
        );
        assert_point_in_time(trip, kind, &PRIMARY_SECTIONS);
    }
    assert_engines_agree(&rocks, &redb, &[]);
}

fn tenant_backup_parity(kind: CorpusKind) {
    let _keys = KeyEnvironment::enter(kind);
    let scenario = TenantScenario::build(PRIMARY_CORPUS_SIZE);
    let [rocks, redb] = run_on_both_engines("tenant backup", |engine| {
        tenant_round_trip(engine, kind, &scenario)
    });
    let mutated: Vec<&str> = PRIMARY_SECTIONS
        .iter()
        .chain(&GRAPH_SECTIONS)
        .chain(&[AUDIT_EVENTS])
        .copied()
        .collect();
    for trip in [&rocks, &redb] {
        let label = trip.engine.label();
        assert_eq!(
            trip.before.section_len(PRIMARY_MEMORIES),
            scenario.primary.corpus.memories.len(),
            "{label}: the primary corpus did not land"
        );
        assert_eq!(
            trip.before.section_len(GRAPH_ENTITIES),
            scenario.graph.entities.len(),
            "{label}: the graph entities did not land"
        );
        assert_eq!(
            trip.before
                .section(GRAPH_STATS)
                .and_then(|stats| stats["relationship_count"].as_u64()),
            Some(scenario.graph.edges.len() as u64),
            "{label}: the graph edges did not land"
        );
        assert_eq!(
            trip.before.section_len(GRAPH_EPISODES),
            scenario.graph.episodes.len(),
            "{label}: the graph episodes did not land"
        );
        assert_eq!(
            trip.before
                .section(AUDIT_EVENTS)
                .and_then(|events| events[TENANT].as_array())
                .map(Vec::len),
            Some(AUDIT_EVENT_COUNT as usize),
            "{label}: the audit events did not land"
        );
        assert_point_in_time(trip, kind, &mutated);
    }
    assert_engines_agree(&rocks, &redb, &[Normalisation::GraphIdentityByName]);
}

#[test]
fn backup_store_restores_the_same_primary_store_on_both_engines() {
    backup_store_parity(CorpusKind::Plaintext);
}

#[test]
fn backup_store_restores_a_sealed_primary_store_on_both_engines_without_leaking_plaintext() {
    backup_store_parity(CorpusKind::Sealed);
}

#[test]
fn tenant_backup_restores_the_same_primary_graph_and_audit_on_both_engines() {
    tenant_backup_parity(CorpusKind::Plaintext);
}

#[test]
fn sealed_tenant_backup_restores_the_same_stores_on_both_engines_without_leaking_plaintext() {
    tenant_backup_parity(CorpusKind::Sealed);
}

// ─── listing and retention ──────────────────────────────────────────────────────

/// What the listings and purges answered across the same sequence of backups.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RetentionTrace {
    backup_store_listing: Vec<ListedBackup>,
    tenant_created: Vec<ListedBackup>,
    tenant_listing_after_four: Vec<ListedBackup>,
    every_backup_verified: bool,
    purged_keeping_two: usize,
    tenant_listing_after_purge: Vec<ListedBackup>,
    purged_backup_verifies: bool,
    tenant_listing_after_fifth: Vec<ListedBackup>,
    purged_with_room_to_spare: usize,
    purge_to_zero_refused: bool,
}

/// Three `BackupStore` backups, then four tenant backups, a purge down to two, a fifth backup
/// and two purges that must remove nothing; one new memory before each backup.
fn retention_trace(engine: Engine, corpus: &PrimaryCorpus) -> Result<RetentionTrace> {
    let live = TempDir::new()?;
    let backup_store_root = TempDir::new()?;
    let tenant_backup_root = TempDir::new()?;
    let tenant = TenantStores::open(engine, live.path())?;
    corpus.write(tenant.primary.store(), TENANT)?;
    let mut next_ordinal = corpus.size;
    let mut add_memory = || -> Result<()> {
        tenant
            .primary
            .store()
            .put_memory(&corpus_memory(next_ordinal))?;
        next_ordinal += 1;
        Ok(())
    };

    let backup_store = tenant.primary.backup_store(backup_store_root.path())?;
    for _ in 0..3 {
        add_memory()?;
        backup_store.store().create_backup(TENANT)?;
    }
    let backup_store_listing = backup_store
        .store()
        .list_backups(TENANT)?
        .iter()
        .map(ListedBackup::from)
        .collect();

    let tenant_backups = TenantBackups::open(engine, tenant_backup_root.path())?;
    let mut tenant_created = Vec::new();
    for _ in 0..4 {
        add_memory()?;
        tenant_created.push(ListedBackup::from(
            &tenant_backups.create(&tenant, live.path())?,
        ));
    }
    let tenant_listing_after_four = canonical_listing(&tenant_backups.list(TENANT)?);
    let mut every_backup_verified = true;
    for backup in &tenant_created {
        every_backup_verified &= tenant_backups.verify(TENANT, backup.backup_id)?;
    }
    let purged_keeping_two = tenant_backups.purge(TENANT, 2)?;
    let tenant_listing_after_purge = canonical_listing(&tenant_backups.list(TENANT)?);
    let purged_backup_verifies = matches!(
        tenant_backups.verify(TENANT, tenant_created[0].backup_id),
        Ok(true)
    );

    add_memory()?;
    tenant_backups.create(&tenant, live.path())?;
    let tenant_listing_after_fifth = canonical_listing(&tenant_backups.list(TENANT)?);
    let purged_with_room_to_spare = tenant_backups.purge(TENANT, 10)?;
    let purge_to_zero_refused = tenant_backups.purge(TENANT, 0).is_err();

    Ok(RetentionTrace {
        backup_store_listing,
        tenant_created,
        tenant_listing_after_four,
        every_backup_verified,
        purged_keeping_two,
        tenant_listing_after_purge,
        purged_backup_verifies,
        tenant_listing_after_fifth,
        purged_with_room_to_spare,
        purge_to_zero_refused,
    })
}

#[test]
fn backup_listing_and_purge_retention_match_on_both_engines() {
    let _keys = KeyEnvironment::enter(CorpusKind::Plaintext);
    let corpus = PrimaryCorpus::build(LISTING_CORPUS_SIZE);
    let [rocks, redb] = Engine::BOTH.map(|engine| {
        retention_trace(engine, &corpus)
            .unwrap_or_else(|e| panic!("{} retention trace failed: {e:#}", engine.label()))
    });
    assert_eq!(
        rocks, redb,
        "the same backup, list and purge sequence left the engines holding different backups"
    );

    let base = corpus.memories.len();
    let listed = |backup_id: u32, added: usize| ListedBackup {
        backup_id,
        user_id: TENANT.to_string(),
        memory_count: base + added,
    };
    assert_eq!(
        rocks.backup_store_listing,
        vec![listed(3, 3), listed(2, 2), listed(1, 1)],
        "BackupStore lists its backups newest first, each counting the memories it captured"
    );
    assert_eq!(
        rocks.tenant_created,
        vec![listed(1, 4), listed(2, 5), listed(3, 6), listed(4, 7)]
    );
    assert!(
        rocks
            .tenant_created
            .iter()
            .all(|backup| backup.user_id == TENANT),
        "a tenant backup is recorded against another user"
    );
    assert!(
        rocks
            .tenant_created
            .windows(2)
            .all(|pair| pair[1].memory_count == pair[0].memory_count + 1),
        "each tenant backup must count the one memory added before it"
    );
    assert_eq!(rocks.tenant_listing_after_four, rocks.tenant_created);
    assert!(
        rocks.every_backup_verified,
        "a fresh tenant backup did not verify"
    );
    assert_eq!(rocks.purged_keeping_two, 2);
    assert_eq!(
        rocks.tenant_listing_after_purge,
        rocks.tenant_created[2..].to_vec(),
        "a purge keeps exactly the newest backups"
    );
    assert!(
        !rocks.purged_backup_verifies,
        "a purged backup still verifies"
    );
    assert_eq!(
        rocks.tenant_listing_after_fifth,
        vec![listed(3, 6), listed(4, 7), listed(5, 8)],
        "the backup after a purge takes the next id, never a purged one"
    );
    assert_eq!(rocks.purged_with_room_to_spare, 0);
    assert!(
        rocks.purge_to_zero_refused,
        "a purge to zero backups was accepted"
    );
}
