//! SQLite WAL slow store for relational gap queries
//!
//! The knowledge graph in RocksDB is optimized for fast key-value access and
//! spreading activation. But structural analysis — finding open triads, orbit gaps,
//! topological voids — requires relational queries that RocksDB cannot efficiently express.
//!
//! This module provides a SQLite backing store (WAL mode, no locking issues) that
//! mirrors the entity/edge graph and enables:
//! - Gap topology detection via SQL anti-joins and recursive CTEs
//! - Persistent storage of detected gaps and generated thoughts
//! - Multi-scale structural analysis across the knowledge graph
//!
//! Architecture: RocksDB (fast store) → periodic sync → SQLite (slow store, gap analysis)

mod embeddings;
mod projection;
mod queries;
mod relational_adapter;
mod storage;

pub use projection::SqliteProjection;
pub use queries::{EntityCluster, RawDiamondGap, RawOpenTriad, RawStarGap};
pub use relational_adapter::RelationalSlowStoreAdapter;
pub use storage::{StoredGap, StoredMemoryRow, StoredThought};

use super::gap_topology::GapStore;

use crate::graph_memory::{
    EdgeSource, EdgeTier, EntityNode, LtpStatus, RelationType, RelationshipEdge,
};
use crate::keystore::SecretCryptor;
use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OpenFlags, Transaction};
use std::collections::HashSet;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Statistics from a graph → SQLite sync operation
#[derive(Debug, Default)]
pub struct SyncStats {
    pub entities_upserted: usize,
    pub edges_upserted: usize,
    /// Co-retrieval edges between memories, which the entity mirror leaves out.
    pub edges_between_memories: usize,
    /// Other edges left out because an endpoint is not among the synced entities.
    pub edges_dangling: usize,
    pub duration_ms: u64,
}

/// Statistics from a retention cleanup operation
#[derive(Debug, Default)]
pub struct CleanupStats {
    pub gaps_deleted: usize,
    pub thoughts_deleted: usize,
}

pub const CURRENT_SCHEMA_VERSION: i32 = 2;

/// Per-column encryption toggles for the slow store (encryption-v2). Each is
/// independent and only takes effect when a keystore-derived encryptor is
/// configured. Read-side decryption is marker-based, so flipping a toggle is
/// safe for already-written rows (old rows still decode by their marker).
///   - `payload` (default ON): the `memory_bincode` blob.
///   - `graph` (default off): free-text content with no SQL readers — entity
///     labels/summary + edge context. Entity `name` and edge `relation_type`
///     stay plaintext because the gap-detection queries read and group on them
///     (encrypting those needs the gap subsystem to decrypt — a follow-up);
///     structural keys (uuid, from/to) stay plaintext for JOINs.
///   - `embeddings` (default off): entity embedding BLOBs (no semantic SQL use).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlowStoreEncrypt {
    pub payload: bool,
    pub graph: bool,
    pub embeddings: bool,
}

impl SlowStoreEncrypt {
    fn from_env() -> Self {
        let on = |k: &str, default: bool| {
            std::env::var(k)
                .map(|v| {
                    let v = v.trim().to_ascii_lowercase();
                    v == "true" || v == "1"
                })
                .unwrap_or(default)
        };
        Self {
            payload: on("VELD_SLOW_STORE_ENCRYPT_PAYLOAD", true),
            graph: on("VELD_SLOW_STORE_ENCRYPT_GRAPH", false),
            embeddings: on("VELD_SLOW_STORE_ENCRYPT_EMBEDDINGS", false),
        }
    }
}

/// Encrypt `bytes` iff an encryptor is present and `enabled`; else passthrough.
fn seal_if(
    enc: Option<&SecretCryptor>,
    bytes: &[u8],
    enabled: bool,
    aad_ctx: &[u8],
) -> Result<Vec<u8>> {
    match enc {
        Some(e) if enabled => e.encrypt_bytes(bytes, aad_ctx),
        _ => Ok(bytes.to_vec()),
    }
}

/// Seal a TEXT column to a SQLite value: a BLOB ciphertext when an encryptor is
/// present and `enabled`, otherwise the plaintext as TEXT (so storage class —
/// and any TEXT reader — is unchanged when the toggle is off).
fn seal_text_value(
    enc: Option<&SecretCryptor>,
    s: &str,
    enabled: bool,
    aad_ctx: &[u8],
) -> Result<rusqlite::types::Value> {
    match enc {
        Some(e) if enabled => Ok(rusqlite::types::Value::Blob(
            e.encrypt_bytes(s.as_bytes(), aad_ctx)?,
        )),
        _ => Ok(rusqlite::types::Value::Text(s.to_string())),
    }
}

/// Decrypt `bytes` iff it carries the encrypted-record marker — independent of
/// any toggle, so toggling a column off still reads previously-encrypted rows.
/// Errors if marked but no key is configured.
fn open_field(enc: Option<&SecretCryptor>, bytes: Vec<u8>, aad_ctx: &[u8]) -> Result<Vec<u8>> {
    match enc {
        Some(e) if SecretCryptor::is_encrypted(&bytes) => e.decrypt_bytes(&bytes, aad_ctx),
        None if SecretCryptor::is_encrypted(&bytes) => Err(anyhow::anyhow!(
            "slow store column is encrypted but no keystore/passphrase is configured \
             (VELD_KEYSTORE_PATH / VELD_MASTER_PASSPHRASE)"
        )),
        _ => Ok(bytes),
    }
}

/// SQLite-backed slow store for relational queries on the knowledge graph.
///
/// Runs in WAL mode: concurrent reads, single writer, no locking issues.
/// Periodically synced from the RocksDB fast store (GraphMemory).
pub struct SlowStore {
    pub(crate) conn: Mutex<Connection>,
    /// Pool of read-only connections. WAL allows many concurrent readers
    /// alongside the single writer, but a lone `Mutex<Connection>` serialized
    /// every read behind every write (and behind every other read). Reads now
    /// check a connection out of this pool, so a long gap-analysis query no
    /// longer blocks a live projection write — or another reader.
    read_pool: ReadPool,
    /// Optional purpose-bound encryptor (encryption-v2) keystore-derived
    /// (XChaCha20-Poly1305, "slow-store" purpose key) from `VELD_KEYSTORE_PATH` +
    /// `VELD_MASTER_PASSPHRASE`; `None` (unset) stores plaintext (backward
    /// compatible). Which columns it covers is governed by [`enc`](Self::enc).
    encryptor: Option<SecretCryptor>,
    /// Per-column encryption toggles (see [`SlowStoreEncrypt`]). Only meaningful
    /// when `encryptor` is `Some`.
    enc: SlowStoreEncrypt,
    path: PathBuf,
    last_sync: Mutex<Option<std::time::Instant>>,
}

/// Maximum reader connections kept idle in the pool. Bursts beyond this open
/// transient connections that are closed on return rather than pooled, so a
/// spike can't pin unbounded memory (each reader carries its own page cache).
const READ_POOL_MAX_IDLE: usize = 4;

/// A pool of read-only SQLite connections over the single WAL database.
///
/// Connections are opened lazily and reused. Each is `query_only` (writes
/// error) and `NO_MUTEX` (so it must be touched by one thread at a time — the
/// checkout/return discipline guarantees that). Readers are full WAL
/// participants: they observe a consistent snapshot concurrently with the
/// writer.
struct ReadPool {
    idle: Mutex<Vec<Connection>>,
    path: PathBuf,
}

impl ReadPool {
    fn new(path: PathBuf) -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
            path,
        }
    }

    /// Open a fresh reader. Opened `READ_WRITE` (not `READ_ONLY`) so it can
    /// initialise the WAL shared-memory index — a pure read-only handle can
    /// fail to attach to a WAL database — then pinned to `query_only` so it can
    /// never actually mutate.
    fn open_reader(path: &Path) -> Result<Connection> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("Failed to open slow store reader connection")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "cache_size", "-4096")?;
        conn.pragma_update(None, "query_only", "ON")?;
        Ok(conn)
    }

    /// Check out a reader: reuse an idle one or open a new one.
    fn checkout(&self) -> Result<PooledConn<'_>> {
        let reused = self.idle.lock().pop();
        let conn = match reused {
            Some(conn) => conn,
            None => Self::open_reader(&self.path)?,
        };
        Ok(PooledConn {
            pool: self,
            conn: Some(conn),
        })
    }

    /// Return a reader to the idle set, or drop it if the pool is full.
    fn put_back(&self, conn: Connection) {
        let mut idle = self.idle.lock();
        if idle.len() < READ_POOL_MAX_IDLE {
            idle.push(conn);
        }
    }
}

/// RAII handle to a pooled reader. Derefs to [`Connection`] and returns the
/// connection to the pool on drop.
pub(crate) struct PooledConn<'a> {
    pool: &'a ReadPool,
    conn: Option<Connection>,
}

impl Deref for PooledConn<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn
            .as_ref()
            .expect("pooled connection present until drop")
    }
}

impl Drop for PooledConn<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.put_back(conn);
        }
    }
}

/// One-off for `veld reseal-journal`: seal the plaintext memory payloads of the
/// slow store at `path` under `cryptor`. Returns (rows, plaintext rows).
pub fn reseal_plaintext_payloads(
    path: &Path,
    cryptor: SecretCryptor,
    dry_run: bool,
) -> Result<(u64, u64)> {
    SlowStore::open_with_encryptor(path, Some(cryptor))?.reseal_plaintext_payloads(dry_run)
}

impl SlowStore {
    /// Open or create the slow store database. Encryption of the memory payload is
    /// enabled iff `VELD_KEYSTORE_PATH` + `VELD_MASTER_PASSPHRASE` are set.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_encryptor(path, SecretCryptor::from_env("slow-store")?)
    }

    /// Open with an explicit encryptor; toggles come from the environment.
    /// Production opens via [`open`](Self::open) (which derives the key from the
    /// keystore); tests inject a key directly.
    pub(crate) fn open_with_encryptor(
        path: &Path,
        encryptor: Option<SecretCryptor>,
    ) -> Result<Self> {
        Self::open_with_encryptor_and_toggles(path, encryptor, SlowStoreEncrypt::from_env())
    }

    /// Open with an explicit encryptor AND explicit per-column toggles (tests use
    /// this to exercise graph/embedding encryption without touching process env).
    pub(crate) fn open_with_encryptor_and_toggles(
        path: &Path,
        encryptor: Option<SecretCryptor>,
        enc: SlowStoreEncrypt,
    ) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("Failed to open slow store SQLite database")?;

        // WAL mode: concurrent readers, no blocking
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // NORMAL sync: survives process crash, not power loss (matches veld's RocksDB async default)
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        // 8MB cache for gap analysis queries
        conn.pragma_update(None, "cache_size", "-8192")?;
        // Enable foreign keys
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // Busy timeout: wait up to 5s instead of failing immediately on contention
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        let store = Self {
            conn: Mutex::new(conn),
            read_pool: ReadPool::new(path.to_owned()),
            encryptor,
            enc,
            path: path.to_owned(),
            last_sync: Mutex::new(None),
        };
        store.create_schema()?;
        store.check_schema_version()?;
        Ok(store)
    }

    /// Check out a pooled read-only connection. All read paths use this instead
    /// of locking the write `conn`, so reads run concurrently with the writer
    /// (and with one another) under WAL rather than serializing on one mutex.
    pub(crate) fn read(&self) -> Result<PooledConn<'_>> {
        self.read_pool.checkout()
    }

    /// Encrypt a memory payload for at-rest storage when an encryptor is
    /// configured; otherwise pass the plaintext through unchanged.
    pub(crate) fn maybe_encrypt(&self, memory_id: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        seal_if(
            self.encryptor.as_ref(),
            plaintext,
            self.enc.payload,
            memory_id.as_bytes(),
        )
    }

    /// Inverse of [`maybe_encrypt`](Self::maybe_encrypt). Decrypts marked blobs,
    /// passes plaintext (pre-encryption) rows through, and errors if a blob is
    /// encrypted but no key is configured (key removed after encrypting).
    pub(crate) fn maybe_decrypt(&self, memory_id: &str, stored: Vec<u8>) -> Result<Vec<u8>> {
        open_field(self.encryptor.as_ref(), stored, memory_id.as_bytes())
    }

    fn create_schema(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            "
            -- Mirror of GraphMemory entities (synced periodically)
            CREATE TABLE IF NOT EXISTS entities (
                uuid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                labels TEXT NOT NULL DEFAULT '[]',
                salience REAL NOT NULL DEFAULT 0.5,
                mention_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                embedding BLOB
            );

            -- Mirror of GraphMemory relationship edges
            CREATE TABLE IF NOT EXISTS edges (
                uuid TEXT PRIMARY KEY,
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL NOT NULL,
                tier TEXT NOT NULL DEFAULT 'L1Working',
                ltp_status TEXT NOT NULL DEFAULT 'None',
                activation_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_activated TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (from_entity) REFERENCES entities(uuid),
                FOREIGN KEY (to_entity) REFERENCES entities(uuid)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_entity);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_entity);
            CREATE INDEX IF NOT EXISTS idx_edges_strength ON edges(strength DESC);

            -- Directional pair index (handles A→B as one entry)
            CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_pair
                ON edges(from_entity, to_entity);

            -- Detected gap topologies
            CREATE TABLE IF NOT EXISTS gap_topologies (
                id TEXT PRIMARY KEY,
                gap_type TEXT NOT NULL,
                shape_signature TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                missing_links_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                embedding_distance REAL,
                impact_score REAL NOT NULL DEFAULT 0.0,
                detected_at TEXT NOT NULL,
                last_verified TEXT NOT NULL,
                resolved_at TEXT,
                scope TEXT NOT NULL DEFAULT 'content'
            );

            CREATE INDEX IF NOT EXISTS idx_gaps_type ON gap_topologies(gap_type);
            CREATE INDEX IF NOT EXISTS idx_gaps_confidence
                ON gap_topologies(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_gaps_unresolved
                ON gap_topologies(resolved_at) WHERE resolved_at IS NULL;

            -- Generated thoughts from gap analysis
            CREATE TABLE IF NOT EXISTS thoughts (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT NOT NULL,
                hypothesis TEXT,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                impact_score REAL NOT NULL DEFAULT 0.0,
                entities_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                surfaced_count INTEGER NOT NULL DEFAULT 0,
                dismissed INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_thoughts_active
                ON thoughts(dismissed, confidence DESC)
                WHERE dismissed = 0;

            -- Projection of the intent log: one row per live memory per user.
            -- (user_id, memory_id) is the natural idempotency key — replay-safe
            -- because UPSERTs collapse duplicate applies. `lsn` is carried as a
            -- monotonic stamp so a concurrent replay vs. live write can resolve
            -- the conflict by keeping the higher LSN (write-skew guard).
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                lsn INTEGER NOT NULL,
                memory_bincode BLOB NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, memory_id)
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_lsn
                ON memories(lsn);

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            );
        ",
        )?;
        Ok(())
    }

    /// Check and apply schema migrations.
    ///
    /// On a fresh database, inserts version 1. On an existing database,
    /// runs any pending migrations up to CURRENT_SCHEMA_VERSION.
    fn check_schema_version(&self) -> Result<()> {
        let conn = self.conn.lock();

        let current_version: Option<i32> = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .context("Failed to query schema version")?;

        match current_version {
            None => {
                // New database — insert initial version
                let now = Utc::now().to_rfc3339();
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?1, ?2)",
                    params![CURRENT_SCHEMA_VERSION, now],
                )?;
                tracing::info!(
                    "SlowStore: initialized schema version {}",
                    CURRENT_SCHEMA_VERSION
                );
            }
            Some(v) if v < CURRENT_SCHEMA_VERSION => {
                // Run forward migrations. `create_schema` is idempotent
                // (every CREATE uses IF NOT EXISTS), so it has already
                // materialised any new tables/indexes added in v2+. The
                // migration here just records the version bump so future
                // boots stop trying to migrate again.
                let now = Utc::now().to_rfc3339();
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?1, ?2)",
                    params![CURRENT_SCHEMA_VERSION, now],
                )?;
                tracing::info!(
                    "SlowStore: migrated schema from version {} to {}",
                    v,
                    CURRENT_SCHEMA_VERSION
                );
            }
            Some(v) => {
                tracing::info!("SlowStore: schema version {} (current)", v);
            }
        }

        Ok(())
    }

    /// Delete old resolved gaps and dismissed thoughts beyond the retention window.
    ///
    /// This prevents the slow store from growing unboundedly with stale data.
    /// Only removes data that has already been acted on (resolved/dismissed).
    pub fn cleanup_old_data(&self, max_age_days: u32) -> Result<CleanupStats> {
        let cutoff = Utc::now() - Duration::days(max_age_days as i64);
        let cutoff_str = cutoff.to_rfc3339();
        let mut conn = self.conn.lock();

        // Both deletes in one transaction so a crash between them can't leave a
        // half-completed cleanup; either both retention sweeps land or neither.
        let tx = conn.transaction()?;
        let gaps_deleted = tx.execute(
            "DELETE FROM gap_topologies WHERE resolved_at IS NOT NULL AND detected_at < ?1",
            params![cutoff_str],
        )?;
        let thoughts_deleted = tx.execute(
            "DELETE FROM thoughts WHERE dismissed = 1 AND created_at < ?1",
            params![cutoff_str],
        )?;
        tx.commit()?;

        tracing::info!(
            "SlowStore cleanup: {} resolved gaps and {} dismissed thoughts removed (cutoff: {} days)",
            gaps_deleted,
            thoughts_deleted,
            max_age_days
        );

        Ok(CleanupStats {
            gaps_deleted,
            thoughts_deleted,
        })
    }

    /// Check if sync is needed based on TTL. Returns true if last sync was
    /// more than `ttl_secs` ago (or never synced).
    pub fn should_sync(&self, ttl_secs: u64) -> bool {
        let last = self.last_sync.lock();
        match *last {
            Some(instant) => instant.elapsed().as_secs() >= ttl_secs,
            None => true,
        }
    }

    /// Sync entity and edge data from GraphMemory into SQLite.
    ///
    /// Full-mirror semantics: every caller passes the complete graph
    /// (`get_all_entities` + `get_all_relationships`), so the mirror tables are
    /// cleared and rebuilt inside one transaction. Entities/edges removed from
    /// the live graph (decay, prune, forget) are purged here rather than
    /// lingering as stale rows — closing both an unbounded-growth leak and a
    /// stale-PII retention gap. Idempotent and atomic (rolls back on error).
    /// Designed to be called periodically (e.g., after consolidation cycles).
    pub fn sync_from_graph(
        &self,
        entities: &[EntityNode],
        edges: &[RelationshipEdge],
    ) -> Result<SyncStats> {
        let _span = tracing::info_span!(
            "slow_store.sync",
            entities = entities.len(),
            edges = edges.len(),
        )
        .entered();

        let start = std::time::Instant::now();
        let mut stats = SyncStats::default();
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;

        // Clear the mirror first so rows for graph-deleted entities/edges don't
        // survive. Edges before entities to respect the FK; the re-insert below
        // runs entities before edges, so referential integrity holds throughout.
        tx.execute("DELETE FROM edges", [])?;
        tx.execute("DELETE FROM entities", [])?;

        Self::sync_entities(
            &tx,
            entities,
            &mut stats,
            self.encryptor.as_ref(),
            &self.enc,
        )?;
        let synced: HashSet<Uuid> = entities.iter().map(|e| e.uuid).collect();
        Self::sync_edges(
            &tx,
            edges,
            &synced,
            &mut stats,
            self.encryptor.as_ref(),
            &self.enc,
        )?;

        tx.commit()?;
        stats.duration_ms = start.elapsed().as_millis() as u64;
        if stats.edges_dangling > 0 {
            tracing::warn!(
                dangling = stats.edges_dangling,
                "SlowStore sync left out edges whose endpoint is not in the graph's entities"
            );
        }

        // Record sync timestamp for TTL-based skip
        *self.last_sync.lock() = Some(std::time::Instant::now());

        tracing::info!(
            "SlowStore sync complete: {} entities, {} edges ({} memory co-retrieval edges left out) in {}ms",
            stats.entities_upserted,
            stats.edges_upserted,
            stats.edges_between_memories,
            stats.duration_ms
        );
        Ok(stats)
    }

    fn sync_entities(
        tx: &Transaction,
        entities: &[EntityNode],
        stats: &mut SyncStats,
        enc: Option<&SecretCryptor>,
        toggles: &SlowStoreEncrypt,
    ) -> Result<()> {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO entities (uuid, name, labels, salience, mention_count,
                                   created_at, last_seen_at, summary, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(uuid) DO UPDATE SET
                name = excluded.name,
                labels = excluded.labels,
                salience = excluded.salience,
                mention_count = excluded.mention_count,
                last_seen_at = excluded.last_seen_at,
                summary = excluded.summary,
                embedding = excluded.embedding",
        )?;

        for entity in entities {
            let labels_json =
                serde_json::to_string(&entity.labels).unwrap_or_else(|_| "[]".to_string());
            // Graph free-text content (labels, summary) is encrypted when the
            // graph toggle is on — these have no SQL readers. `entity.name` stays
            // PLAINTEXT: the gap-detection queries (find_star_gaps etc.) read and
            // group on it, so encrypting it would need those queries to decrypt
            // (tracked follow-up). `uuid` stays plaintext for JOINs.
            // AAD-bind the sealed columns to this entity's uuid.
            let entity_uuid = entity.uuid.to_string();
            let labels_col =
                seal_text_value(enc, &labels_json, toggles.graph, entity_uuid.as_bytes())?;
            let summary_col =
                seal_text_value(enc, &entity.summary, toggles.graph, entity_uuid.as_bytes())?;
            // Embedding BLOB: encrypted when the embeddings toggle is on.
            let embedding_blob: Option<Vec<u8>> = entity
                .name_embedding
                .as_ref()
                .map(|emb| -> Result<Vec<u8>> {
                    let raw: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
                    seal_if(enc, &raw, toggles.embeddings, entity_uuid.as_bytes())
                })
                .transpose()?;

            stmt.execute(params![
                entity.uuid.to_string(),
                entity.name,
                labels_col,
                entity.salience,
                entity.mention_count,
                entity.created_at.to_rfc3339(),
                entity.last_seen_at.to_rfc3339(),
                summary_col,
                embedding_blob,
            ])?;
            stats.entities_upserted += 1;
        }
        Ok(())
    }

    /// Mirrors the edges between synced entities. GraphMemory keeps memory
    /// co-retrieval edges (`CoRetrieved`, endpoints are memory ids; edges written
    /// before `created_by` existed read back as `CoOccurrence`) beside entity
    /// edges; those, and any edge to an entity the graph no longer returns, would
    /// break the entities foreign key and roll back the whole sync, so they are
    /// counted and left out.
    fn sync_edges(
        tx: &Transaction,
        edges: &[RelationshipEdge],
        synced: &HashSet<Uuid>,
        stats: &mut SyncStats,
        enc: Option<&SecretCryptor>,
        toggles: &SlowStoreEncrypt,
    ) -> Result<()> {
        // Conflict target is the directed pair, not `uuid`. The slow store is a
        // connectivity graph (one entry per A->B, per `idx_edge_pair`), but
        // GraphMemory legitimately holds several edges per pair (one per
        // relation type). Targeting `uuid` made the 2nd such edge violate the
        // pair-unique index — an unhandled conflict that aborted the whole sync
        // transaction. Collapsing on the pair keeps the first uuid/relation_type
        // and refreshes the mutable strength/tier/activation columns.
        let mut stmt = tx.prepare_cached(
            "INSERT INTO edges (uuid, from_entity, to_entity, relation_type, strength,
                                tier, ltp_status, activation_count, created_at,
                                last_activated, context)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(from_entity, to_entity) DO UPDATE SET
                strength = excluded.strength,
                tier = excluded.tier,
                ltp_status = excluded.ltp_status,
                activation_count = excluded.activation_count,
                last_activated = excluded.last_activated,
                context = excluded.context",
        )?;

        for edge in edges {
            if !synced.contains(&edge.from_entity) || !synced.contains(&edge.to_entity) {
                if edge.relation_type == RelationType::CoRetrieved
                    || edge.created_by == EdgeSource::Coactivation
                {
                    stats.edges_between_memories += 1;
                } else {
                    stats.edges_dangling += 1;
                }
                continue;
            }
            let tier_str = match edge.tier {
                EdgeTier::L1Working => "L1Working",
                EdgeTier::L2Episodic => "L2Episodic",
                EdgeTier::L3Semantic => "L3Semantic",
            };
            let ltp_str = match edge.ltp_status {
                LtpStatus::None => "None",
                LtpStatus::Burst { .. } => "Burst",
                LtpStatus::Weekly => "Weekly",
                LtpStatus::Full => "Full",
            };

            // Edge free-text `context` is encrypted when the graph toggle is on
            // (no SQL readers). `relation_type` stays PLAINTEXT — the gap queries
            // read/group on it; from/to/strength are structural.
            let edge_uuid = edge.uuid.to_string();
            let context_col =
                seal_text_value(enc, &edge.context, toggles.graph, edge_uuid.as_bytes())?;
            stmt.execute(params![
                edge.uuid.to_string(),
                edge.from_entity.to_string(),
                edge.to_entity.to_string(),
                edge.relation_type.as_str(),
                edge.strength,
                tier_str,
                ltp_str,
                edge.activation_count,
                edge.created_at.to_rfc3339(),
                edge.last_activated.to_rfc3339(),
                context_col,
            ])?;
            stats.edges_upserted += 1;
        }
        Ok(())
    }

    /// Database file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// SlowStore implements GapStore for SQL-backed gap detection.
impl GapStore for SlowStore {
    fn find_open_triads(
        &self,
        min_strength: f32,
        limit: usize,
    ) -> anyhow::Result<Vec<super::slow_store::RawOpenTriad>> {
        SlowStore::find_open_triads(self, min_strength, limit)
    }
    fn find_diamond_gaps(
        &self,
        min_strength: f32,
        limit: usize,
    ) -> anyhow::Result<Vec<super::slow_store::RawDiamondGap>> {
        SlowStore::find_diamond_gaps(self, min_strength, limit)
    }
    fn find_star_gaps(
        &self,
        min_spokes: usize,
        max_connectivity: f32,
        limit: usize,
    ) -> anyhow::Result<Vec<super::slow_store::RawStarGap>> {
        SlowStore::find_star_gaps(self, min_spokes, max_connectivity, limit)
    }
    fn get_adjacency_list(
        &self,
        min_strength: f32,
    ) -> anyhow::Result<std::collections::HashMap<String, Vec<String>>> {
        SlowStore::get_adjacency_list(self, min_strength)
    }
    fn get_entity_names(
        &self,
        uuids: &[&str],
    ) -> anyhow::Result<std::collections::HashMap<String, String>> {
        SlowStore::get_entity_names(self, uuids)
    }
    fn load_embeddings(
        &self,
        uuids: &[&str],
    ) -> anyhow::Result<std::collections::HashMap<String, Vec<f32>>> {
        SlowStore::load_embeddings(self, uuids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    #[test]
    fn test_slow_store_opens_and_creates_schema() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        assert_eq!(store.entity_count().unwrap(), 0);
        assert_eq!(store.edge_count().unwrap(), 0);
    }

    #[test]
    fn test_store_and_get_thought() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store
            .store_thought(
                "t-001",
                "missing_connection",
                "content",
                0.85,
                "A and C should connect",
                Some("B mediates"),
                "[]",
                0.7,
                "[]",
            )
            .unwrap();
        let thoughts = store.get_active_thoughts(10).unwrap();
        assert_eq!(thoughts.len(), 1);
        assert_eq!(thoughts[0].id, "t-001");
        assert!((thoughts[0].confidence - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_dismiss_thought() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store
            .store_thought(
                "t-002", "silo", "content", 0.5, "desc", None, "[]", 0.3, "[]",
            )
            .unwrap();
        assert_eq!(store.get_active_thoughts(10).unwrap().len(), 1);
        store.dismiss_thought("t-002").unwrap();
        assert_eq!(store.get_active_thoughts(10).unwrap().len(), 0);
    }

    #[test]
    fn test_store_and_resolve_gap() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store
            .store_gap(
                "g-001",
                "open_triad",
                "A-B-C",
                "[]",
                "[]",
                0.9,
                Some(0.45),
                0.8,
                "content",
            )
            .unwrap();
        assert_eq!(
            store.get_unresolved_gaps("open_triad", 10).unwrap().len(),
            1
        );
        store.resolve_gap("g-001").unwrap();
        assert!(store
            .get_unresolved_gaps("open_triad", 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_mark_thought_surfaced() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store
            .store_thought(
                "t-003", "golden", "content", 0.6, "desc", None, "[]", 0.5, "[]",
            )
            .unwrap();
        store.mark_thought_surfaced("t-003").unwrap();
        store.mark_thought_surfaced("t-003").unwrap();
        assert_eq!(store.get_active_thoughts(10).unwrap()[0].surfaced_count, 2);
    }

    #[test]
    fn test_empty_queries() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        assert!(store.load_all_embeddings().unwrap().is_empty());
        assert!(store.get_adjacency_list(0.0).unwrap().is_empty());
        assert!(store.load_all_edge_pairs().unwrap().is_empty());
        assert!(store.get_active_thoughts(10).unwrap().is_empty());
    }

    #[test]
    fn test_cleanup_old_data() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        // Create a thought and dismiss it
        store
            .store_thought(
                "t-cleanup",
                "silo",
                "content",
                0.5,
                "old dismissed thought",
                None,
                "[]",
                0.3,
                "[]",
            )
            .unwrap();
        store.dismiss_thought("t-cleanup").unwrap();

        // Verify the thought exists (dismissed, so not in active list, but in DB)
        {
            let conn = store.conn.lock();
            let count: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM thoughts WHERE id = 't-cleanup'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 1);
        }

        // Cleanup with 0 days cutoff — everything dismissed before now should be removed
        let stats = store.cleanup_old_data(0).unwrap();
        assert_eq!(stats.thoughts_deleted, 1);

        // Verify the thought is gone
        {
            let conn = store.conn.lock();
            let count: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM thoughts WHERE id = 't-cleanup'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 0);
        }
    }

    fn test_entity(name: &str) -> EntityNode {
        let now = Utc::now();
        EntityNode {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![crate::graph_memory::EntityLabel::Person],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            pii_classification: crate::graph_memory::PiiClassification::Clean,
        }
    }

    fn test_edge(
        from: uuid::Uuid,
        to: uuid::Uuid,
        rel: crate::graph_memory::RelationType,
    ) -> RelationshipEdge {
        let now = Utc::now();
        RelationshipEdge {
            uuid: uuid::Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: rel,
            strength: 0.7,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: now,
            activation_count: 0,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L1Working,
            activation_timestamps: None,
            entity_confidence: None,
            created_by: crate::graph_memory::EdgeSource::CoOccurrence,
            forward_strength: 0.0,
            backward_strength: 0.0,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    #[test]
    fn sync_collapses_multiple_relation_types_per_pair() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        let a = test_entity("Alice");
        let b = test_entity("Bob");
        // Two distinct edges between the same directed pair (different relation
        // types + uuids) — exactly what GraphMemory produces. Before the fix
        // the second edge violated idx_edge_pair, an unhandled conflict that
        // aborted the entire sync transaction.
        let e1 = test_edge(a.uuid, b.uuid, crate::graph_memory::RelationType::WorksWith);
        let e2 = test_edge(a.uuid, b.uuid, crate::graph_memory::RelationType::PartOf);

        let stats = store
            .sync_from_graph(&[a, b], &[e1, e2])
            .expect("sync must not abort on a multi-relation pair");
        assert_eq!(stats.entities_upserted, 2);
        assert_eq!(stats.edges_upserted, 2);
        // Collapsed to one connectivity entry for the directed pair.
        assert_eq!(store.edge_count().unwrap(), 1);
    }

    /// A store holding a pseudo-random graph whose strengths repeat, so ties
    /// are common, and the strength of each of its directed pairs.
    fn random_graph(
        seed: u64,
        nodes: usize,
        edges_per_node: usize,
    ) -> (NamedTempFile, SlowStore, HashMap<(String, String), f32>) {
        const STRENGTHS: [f32; 6] = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
        let mut state = seed | 1;
        let mut next = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let entities: Vec<EntityNode> = (0..nodes)
            .map(|i| {
                let mut e = test_entity(&format!("E{i}"));
                e.salience = STRENGTHS[next(STRENGTHS.len())];
                e
            })
            .collect();
        let mut strengths = HashMap::new();
        let mut edges = Vec::new();
        for _ in 0..nodes * edges_per_node {
            let (from, to) = (&entities[next(nodes)], &entities[next(nodes)]);
            let key = (from.uuid.to_string(), to.uuid.to_string());
            if strengths.contains_key(&key) {
                continue;
            }
            let mut edge = test_edge(
                from.uuid,
                to.uuid,
                crate::graph_memory::RelationType::RelatedTo,
            );
            edge.strength = STRENGTHS[next(STRENGTHS.len())];
            strengths.insert(key, edge.strength);
            edges.push(edge);
        }
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store.sync_from_graph(&entities, &edges).unwrap();
        (tmp, store, strengths)
    }

    #[test]
    fn open_triads_found_in_memory_match_the_sql_search() {
        for seed in 1..=12 {
            let (_tmp, store, strength) = random_graph(seed, 40, 4);
            let s = |a: &str, b: &str| strength.get(&(a.to_string(), b.to_string())).copied();
            for limit in [0, 1, 5, 20, 100_000] {
                let fast = store.find_open_triads(0.2, limit).unwrap();
                let sql = store.find_open_triads_sql(0.2, limit).unwrap();
                let sums = |t: &[RawOpenTriad]| -> Vec<f64> {
                    t.iter()
                        .map(|t| f64::from(t.ab_strength) + f64::from(t.bc_strength))
                        .collect()
                };
                assert_eq!(sums(&fast), sums(&sql), "seed {seed} limit {limit}");
                for t in &fast {
                    assert_eq!(s(&t.node_a, &t.node_b), Some(t.ab_strength));
                    assert_eq!(s(&t.node_b, &t.node_c), Some(t.bc_strength));
                    assert!(t.ab_strength >= 0.2 && t.bc_strength >= 0.2);
                    assert_ne!(t.node_a, t.node_c);
                    assert!(s(&t.node_a, &t.node_c).is_none() && s(&t.node_c, &t.node_a).is_none());
                }
                if limit == 100_000 {
                    let paths = |t: &[RawOpenTriad]| -> HashSet<(String, String, String)> {
                        t.iter()
                            .map(|t| (t.node_a.clone(), t.node_b.clone(), t.node_c.clone()))
                            .collect()
                    };
                    assert_eq!(paths(&fast), paths(&sql), "seed {seed}");
                }
            }
        }
    }

    #[test]
    fn diamond_gaps_found_in_memory_match_the_sql_search() {
        for seed in 1..=12 {
            let (_tmp, store, strength) = random_graph(seed, 30, 5);
            let s = |a: &str, b: &str| strength.get(&(a.to_string(), b.to_string())).copied();
            let sum = |d: &RawDiamondGap| -> f64 {
                [
                    (&d.top, &d.left),
                    (&d.top, &d.right),
                    (&d.left, &d.bottom),
                    (&d.right, &d.bottom),
                ]
                .iter()
                .map(|(a, b)| f64::from(s(a, b).expect("a diamond edge exists")))
                .sum()
            };
            for limit in [0, 1, 5, 20, 100_000] {
                let fast = store.find_diamond_gaps(0.2, limit).unwrap();
                let sql = store.find_diamond_gaps_sql(0.2, limit).unwrap();
                assert_eq!(
                    fast.iter().map(sum).collect::<Vec<_>>(),
                    sql.iter().map(sum).collect::<Vec<_>>(),
                    "seed {seed} limit {limit}"
                );
                for d in &fast {
                    assert!(d.left < d.right);
                    assert_ne!(d.top, d.bottom);
                    assert!(s(&d.left, &d.right).is_none() && s(&d.right, &d.left).is_none());
                    for (a, b) in [
                        (&d.top, &d.left),
                        (&d.top, &d.right),
                        (&d.left, &d.bottom),
                        (&d.right, &d.bottom),
                    ] {
                        assert!(s(a, b).is_some_and(|v| v >= 0.2));
                    }
                }
                if limit == 100_000 {
                    let shapes =
                        |g: &[RawDiamondGap]| -> HashSet<(String, String, String, String)> {
                            g.iter()
                                .map(|d| {
                                    (
                                        d.top.clone(),
                                        d.left.clone(),
                                        d.right.clone(),
                                        d.bottom.clone(),
                                    )
                                })
                                .collect()
                        };
                    assert_eq!(shapes(&fast), shapes(&sql), "seed {seed}");
                }
            }
        }
    }

    /// Times both searches, and a whole detection, on a copy of a live mirror:
    /// `SLOW_STORE_BENCH_DB=/path/to/slow_store.db cargo test --release -- --ignored gap_searches_on_a_mirror`.
    #[test]
    #[ignore]
    fn gap_searches_on_a_mirror() {
        let path = std::env::var("SLOW_STORE_BENCH_DB")
            .expect("SLOW_STORE_BENCH_DB names a slow_store.db");
        let store = SlowStore::open(Path::new(&path)).unwrap();
        let timed = |f: &dyn Fn() -> Vec<f64>| {
            let start = std::time::Instant::now();
            let sums = f();
            (start.elapsed(), sums)
        };
        let triad_sums = |t: Vec<RawOpenTriad>| -> Vec<f64> {
            t.iter()
                .map(|t| f64::from(t.ab_strength) + f64::from(t.bc_strength))
                .collect()
        };
        let (fast, fast_sums) = timed(&|| triad_sums(store.find_open_triads(0.2, 50).unwrap()));
        let (sql, sql_sums) = timed(&|| triad_sums(store.find_open_triads_sql(0.2, 50).unwrap()));
        assert_eq!(fast_sums, sql_sums);
        println!("open triads: in memory {fast:?}, SQL {sql:?}");
        let (fast, fast_count) =
            timed(&|| vec![store.find_diamond_gaps(0.2, 50).unwrap().len() as f64]);
        let (sql, sql_count) =
            timed(&|| vec![store.find_diamond_gaps_sql(0.2, 50).unwrap().len() as f64]);
        assert_eq!(fast_count, sql_count);
        println!("diamonds: in memory {fast:?}, SQL {sql:?}");
        let start = std::time::Instant::now();
        let result = crate::memory::gap_topology::GapDetector::detect(
            &store,
            &crate::memory::gap_topology::GapDetectionConfig::default(),
        )
        .unwrap();
        println!(
            "whole detection: {:?}, {} gaps {:?}",
            start.elapsed(),
            result.gaps.len(),
            result.type_counts
        );
    }

    #[test]
    fn sync_mirrors_only_edges_between_entities() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        let a = test_entity("Alice");
        let b = test_entity("Bob");
        let gone = test_entity("Dave");
        let kept = test_edge(a.uuid, b.uuid, crate::graph_memory::RelationType::WorksWith);
        let dangling = test_edge(
            a.uuid,
            gone.uuid,
            crate::graph_memory::RelationType::WorksWith,
        );
        // record_memory_coactivation stores memory ids in from_entity/to_entity;
        // an edge it wrote before created_by existed reads back as CoOccurrence.
        let mut co_retrieved = test_edge(
            uuid::Uuid::new_v4(),
            uuid::Uuid::new_v4(),
            crate::graph_memory::RelationType::CoRetrieved,
        );
        co_retrieved.created_by = EdgeSource::Coactivation;
        let older_co_retrieved = test_edge(
            uuid::Uuid::new_v4(),
            uuid::Uuid::new_v4(),
            crate::graph_memory::RelationType::CoRetrieved,
        );

        let stats = store
            .sync_from_graph(&[a, b], &[kept, dangling, co_retrieved, older_co_retrieved])
            .expect("an edge whose endpoint is not an entity must not roll back the sync");
        assert_eq!(stats.entities_upserted, 2);
        assert_eq!(stats.edges_upserted, 1);
        assert_eq!(stats.edges_between_memories, 2);
        assert_eq!(stats.edges_dangling, 1);
        assert_eq!(store.entity_count().unwrap(), 2);
        assert_eq!(store.edge_count().unwrap(), 1);
    }

    #[test]
    fn sync_purges_entities_dropped_from_the_graph() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        let a = test_entity("Alice");
        let b = test_entity("Bob");
        let c = test_entity("Carol");
        let carol_uuid = c.uuid.to_string();
        store
            .sync_from_graph(&[a.clone(), b.clone(), c], &[])
            .unwrap();
        assert_eq!(store.entity_count().unwrap(), 3);

        // Carol decayed out of the live graph; the next full sync omits her.
        store.sync_from_graph(&[a, b], &[]).unwrap();
        assert_eq!(store.entity_count().unwrap(), 2);
        assert!(
            store
                .get_entity_names(&[carol_uuid.as_str()])
                .unwrap()
                .is_empty(),
            "stale entity must be purged, not retained"
        );
    }

    #[test]
    fn find_star_gaps_counts_inter_spoke_connectivity() {
        use crate::graph_memory::RelationType;
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        let hub = test_entity("Hub");
        let s1 = test_entity("S1");
        let s2 = test_entity("S2");
        let s3 = test_entity("S3");
        // Hub connects to all three spokes; only s1<->s2 connect to each other.
        let edges = vec![
            test_edge(hub.uuid, s1.uuid, RelationType::WorksWith),
            test_edge(hub.uuid, s2.uuid, RelationType::WorksWith),
            test_edge(hub.uuid, s3.uuid, RelationType::WorksWith),
            test_edge(s1.uuid, s2.uuid, RelationType::WorksWith),
        ];
        store
            .sync_from_graph(&[hub.clone(), s1, s2, s3], &edges)
            .unwrap();

        let gaps = store.find_star_gaps(3, 0.9, 10).unwrap();
        assert_eq!(gaps.len(), 1, "the 1/3-connected hub is a star gap");
        let g = &gaps[0];
        assert_eq!(g.hub_name, "Hub");
        assert_eq!(g.spokes.len(), 3);
        assert_eq!(g.possible_edges, 3); // C(3,2)
        assert_eq!(g.missing_edges, 2); // only the s1-s2 pair is connected
    }

    #[test]
    fn entity_lookups_batch_and_decode() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();

        let a = test_entity("Alice");
        let b = test_entity("Bob");
        let mut c = test_entity("Carol");
        c.name_embedding = Some(vec![0.5, -0.25]);
        let (a_id, b_id, c_id) = (a.uuid.to_string(), b.uuid.to_string(), c.uuid.to_string());
        store.sync_from_graph(&[a, b, c], &[]).unwrap();

        // One batched IN query; missing UUIDs simply aren't returned.
        let names = store
            .get_entity_names(&[a_id.as_str(), b_id.as_str(), "missing"])
            .unwrap();
        assert_eq!(names.len(), 2);
        assert_eq!(names.get(&a_id).map(String::as_str), Some("Alice"));
        assert!(names.get("missing").is_none());

        // Only Carol carries an embedding; Alice has none, so it's absent.
        let embs = store
            .load_embeddings(&[c_id.as_str(), a_id.as_str()])
            .unwrap();
        assert_eq!(embs.len(), 1);
        assert_eq!(
            embs.get(&c_id).map(Vec::as_slice),
            Some(&[0.5f32, -0.25][..])
        );

        // Empty input short-circuits.
        assert!(store.get_entity_names(&[]).unwrap().is_empty());
        assert!(store.load_embeddings(&[]).unwrap().is_empty());
    }

    #[test]
    fn read_pool_serves_concurrent_readers() {
        use std::sync::Arc;
        let tmp = NamedTempFile::new().unwrap();
        let store = Arc::new(SlowStore::open(tmp.path()).unwrap());

        let a = test_entity("Alice");
        let b = test_entity("Bob");
        let a_id = a.uuid.to_string();
        store.sync_from_graph(&[a, b], &[]).unwrap();

        // Many readers hit the pool at once: none should deadlock or error, and
        // all observe the committed snapshot. Exercises checkout/return plus
        // several live reader connections concurrently (beyond MAX_IDLE).
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = Arc::clone(&store);
            let id = a_id.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let names = s.get_entity_names(&[id.as_str()]).unwrap();
                    assert_eq!(names.get(&id).map(String::as_str), Some("Alice"));
                    assert_eq!(s.entity_count().unwrap(), 2);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Rows written before the store had a key are sealed by the one-off, bound to
    /// their memory id exactly as a fresh write is; a dry run only counts them.
    #[test]
    fn plaintext_payloads_written_before_the_key_are_sealed_by_the_one_off() {
        let tmp = NamedTempFile::new().unwrap();
        let raw = |store: &SlowStore| -> Vec<u8> {
            store
                .conn
                .lock()
                .query_row(
                    "SELECT memory_bincode FROM memories WHERE user_id = 'u' AND memory_id = 'm'",
                    [],
                    |r| r.get(0),
                )
                .unwrap()
        };
        {
            let plain = SlowStore::open_with_encryptor(tmp.path(), None).unwrap();
            plain
                .upsert_memory("u", "m", 1, b"snapshot written before the key", 0.5)
                .unwrap();
        }
        let key = || crate::keystore::SecretCryptor::new(zeroize::Zeroizing::new([7u8; 32]));
        assert_eq!(
            reseal_plaintext_payloads(tmp.path(), key(), true).unwrap(),
            (1, 1)
        );
        let store = SlowStore::open_with_encryptor(tmp.path(), Some(key())).unwrap();
        assert_eq!(
            raw(&store),
            b"snapshot written before the key",
            "a dry run writes nothing"
        );
        drop(store);

        assert_eq!(
            reseal_plaintext_payloads(tmp.path(), key(), false).unwrap(),
            (1, 1)
        );
        let store = SlowStore::open_with_encryptor(tmp.path(), Some(key())).unwrap();
        assert!(crate::keystore::SecretCryptor::is_encrypted(&raw(&store)));
        assert_eq!(
            store.maybe_decrypt("m", raw(&store)).unwrap(),
            b"snapshot written before the key"
        );
        drop(store);
        assert_eq!(
            reseal_plaintext_payloads(tmp.path(), key(), false).unwrap(),
            (1, 0),
            "a second run is a no-op"
        );
    }

    #[test]
    fn memory_blob_encrypted_at_rest_when_key_configured() {
        let tmp = NamedTempFile::new().unwrap();
        let enc = crate::keystore::SecretCryptor::new(zeroize::Zeroizing::new([7u8; 32]));
        let store = SlowStore::open_with_encryptor(tmp.path(), Some(enc)).unwrap();

        let plaintext = b"super secret memory bincode payload".to_vec();
        store.upsert_memory("u", "m", 1, &plaintext, 0.5).unwrap();

        // On disk the blob is ENC-marked ciphertext, not the plaintext.
        {
            let conn = store.conn.lock();
            let raw: Vec<u8> = conn
                .query_row(
                    "SELECT memory_bincode FROM memories WHERE user_id = 'u' AND memory_id = 'm'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_ne!(raw, plaintext, "payload must be encrypted on disk");
            assert!(crate::keystore::SecretCryptor::is_encrypted(&raw));
        }

        // The read path decrypts transparently.
        let row = store.get_memory_blob("u", "m").unwrap().unwrap();
        assert_eq!(row.memory_bincode, plaintext);
    }

    #[test]
    fn memory_blob_plaintext_without_key() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open_with_encryptor(tmp.path(), None).unwrap();

        let plaintext = b"not encrypted".to_vec();
        store.upsert_memory("u", "m", 1, &plaintext, 0.5).unwrap();

        let conn = store.conn.lock();
        let raw: Vec<u8> = conn
            .query_row(
                "SELECT memory_bincode FROM memories WHERE user_id = 'u' AND memory_id = 'm'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw, plaintext, "no key configured -> stored verbatim");
    }

    #[test]
    fn graph_and_embedding_columns_encrypted_when_toggled() {
        let tmp = NamedTempFile::new().unwrap();
        let enc = crate::keystore::SecretCryptor::new(zeroize::Zeroizing::new([9u8; 32]));
        let toggles = SlowStoreEncrypt {
            payload: true,
            graph: true,
            embeddings: true,
        };
        let store =
            SlowStore::open_with_encryptor_and_toggles(tmp.path(), Some(enc), toggles).unwrap();

        let mut a = test_entity("alice-entity");
        a.summary = "alice private summary".to_string();
        a.name_embedding = Some(vec![0.25, -0.5, 0.75]);
        let b = test_entity("bob-entity");
        let mut edge = test_edge(a.uuid, b.uuid, crate::graph_memory::RelationType::WorksWith);
        edge.context = "edge private context".to_string();
        let a_uuid = a.uuid;

        store.sync_from_graph(&[a, b], &[edge]).unwrap();

        // On disk: summary + embedding (entity) and context (edge) are ENC-marked
        // ciphertext; name + relation_type stay plaintext (gap-query inputs).
        {
            let conn = store.conn.lock();
            let (summary_raw, name_raw, emb_raw): (Vec<u8>, String, Vec<u8>) = conn
                .query_row(
                    "SELECT summary, name, embedding FROM entities WHERE uuid = ?1",
                    [a_uuid.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            assert!(
                crate::keystore::SecretCryptor::is_encrypted(&summary_raw),
                "entity summary must be encrypted on disk"
            );
            assert!(
                crate::keystore::SecretCryptor::is_encrypted(&emb_raw),
                "entity embedding must be encrypted on disk"
            );
            assert_eq!(
                name_raw, "alice-entity",
                "entity name stays plaintext (gap-query input)"
            );

            let (ctx_raw, rel_raw): (Vec<u8>, String) = conn
                .query_row(
                    "SELECT context, relation_type FROM edges LIMIT 1",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert!(
                crate::keystore::SecretCryptor::is_encrypted(&ctx_raw),
                "edge context must be encrypted on disk"
            );
            assert!(
                !rel_raw.is_empty(),
                "relation_type stays plaintext + readable"
            );
        }

        // Read path: the embedding decrypts and the name round-trips.
        let all = store.load_all_embeddings().unwrap();
        let alice = all
            .iter()
            .find(|(_, n, _)| n == "alice-entity")
            .expect("alice present after sync");
        assert_eq!(
            alice.2,
            vec![0.25, -0.5, 0.75],
            "embedding round-trips through the encrypted column"
        );
    }
}
