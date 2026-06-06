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

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OpenFlags, Transaction};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use crate::graph_memory::{EdgeTier, EntityNode, LtpStatus, RelationshipEdge};
use crate::keystore::SecretCryptor;

/// Statistics from a graph → SQLite sync operation
#[derive(Debug, Default)]
pub struct SyncStats {
    pub entities_upserted: usize,
    pub edges_upserted: usize,
    pub duration_ms: u64,
}

/// Statistics from a retention cleanup operation
#[derive(Debug, Default)]
pub struct CleanupStats {
    pub gaps_deleted: usize,
    pub thoughts_deleted: usize,
}

pub const CURRENT_SCHEMA_VERSION: i32 = 2;

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
    /// Optional purpose-bound encryptor (encryption-v2) for the `memory_bincode`
    /// payload. Keystore-derived (XChaCha20-Poly1305, "slow-store" purpose key),
    /// sourced from `VELD_KEYSTORE_PATH` + `VELD_MASTER_PASSPHRASE`; `None` (unset)
    /// stores plaintext (backward compatible). Entity/edge/gap rows match the
    /// primary store's plaintext graph posture — the sensitive payload is the
    /// memory blob, encrypted at rest here when a keystore is configured.
    encryptor: Option<SecretCryptor>,
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

impl SlowStore {
    /// Open or create the slow store database. Encryption of the memory payload is
    /// enabled iff `VELD_KEYSTORE_PATH` + `VELD_MASTER_PASSPHRASE` are set.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_encryptor(path, SecretCryptor::from_env("slow-store")?)
    }

    /// Open with an explicit encryptor. Production opens via [`open`](Self::open)
    /// (which derives the key from the keystore); tests inject a key directly.
    pub(crate) fn open_with_encryptor(
        path: &Path,
        encryptor: Option<SecretCryptor>,
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
    pub(crate) fn maybe_encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        match &self.encryptor {
            Some(enc) => enc.encrypt_bytes(plaintext),
            None => Ok(plaintext.to_vec()),
        }
    }

    /// Inverse of [`maybe_encrypt`](Self::maybe_encrypt). Decrypts marked blobs,
    /// passes plaintext (pre-encryption) rows through, and errors if a blob is
    /// encrypted but no key is configured (key removed after encrypting).
    pub(crate) fn maybe_decrypt(&self, stored: Vec<u8>) -> Result<Vec<u8>> {
        match &self.encryptor {
            Some(enc) if SecretCryptor::is_encrypted(&stored) => enc.decrypt_bytes(&stored),
            Some(_) => Ok(stored),
            None if SecretCryptor::is_encrypted(&stored) => Err(anyhow::anyhow!(
                "slow store memory payload is encrypted but no keystore/passphrase is configured \
                 (VELD_KEYSTORE_PATH / VELD_MASTER_PASSPHRASE)"
            )),
            None => Ok(stored),
        }
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
            .query_row(
                "SELECT MAX(version) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .context("Failed to query schema version")?;

        match current_version {
            None => {
                // New database — insert initial version
                let now = Utc::now().to_rfc3339();
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?1, ?2)",
                    params![CURRENT_SCHEMA_VERSION, now],
                )?;
                tracing::info!("SlowStore: initialized schema version {}", CURRENT_SCHEMA_VERSION);
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

        Self::sync_entities(&tx, entities, &mut stats)?;
        Self::sync_edges(&tx, edges, &mut stats)?;

        tx.commit()?;
        stats.duration_ms = start.elapsed().as_millis() as u64;

        // Record sync timestamp for TTL-based skip
        *self.last_sync.lock() = Some(std::time::Instant::now());

        tracing::info!(
            "SlowStore sync complete: {} entities, {} edges in {}ms",
            stats.entities_upserted,
            stats.edges_upserted,
            stats.duration_ms
        );
        Ok(stats)
    }

    fn sync_entities(tx: &Transaction, entities: &[EntityNode], stats: &mut SyncStats) -> Result<()> {
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
            let labels_json = serde_json::to_string(&entity.labels)
                .unwrap_or_else(|_| "[]".to_string());
            let embedding_blob: Option<Vec<u8>> = entity
                .name_embedding
                .as_ref()
                .map(|emb| emb.iter().flat_map(|f| f.to_le_bytes()).collect());

            stmt.execute(params![
                entity.uuid.to_string(),
                entity.name,
                labels_json,
                entity.salience,
                entity.mention_count,
                entity.created_at.to_rfc3339(),
                entity.last_seen_at.to_rfc3339(),
                entity.summary,
                embedding_blob,
            ])?;
            stats.entities_upserted += 1;
        }
        Ok(())
    }

    fn sync_edges(tx: &Transaction, edges: &[RelationshipEdge], stats: &mut SyncStats) -> Result<()> {
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
                edge.context,
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
    fn find_open_triads(&self, min_strength: f32, limit: usize) -> anyhow::Result<Vec<super::slow_store::RawOpenTriad>> {
        SlowStore::find_open_triads(self, min_strength, limit)
    }
    fn find_diamond_gaps(&self, min_strength: f32, limit: usize) -> anyhow::Result<Vec<super::slow_store::RawDiamondGap>> {
        SlowStore::find_diamond_gaps(self, min_strength, limit)
    }
    fn find_star_gaps(&self, min_spokes: usize, max_connectivity: f32, limit: usize) -> anyhow::Result<Vec<super::slow_store::RawStarGap>> {
        SlowStore::find_star_gaps(self, min_spokes, max_connectivity, limit)
    }
    fn get_adjacency_list(&self, min_strength: f32) -> anyhow::Result<std::collections::HashMap<String, Vec<String>>> {
        SlowStore::get_adjacency_list(self, min_strength)
    }
    fn get_entity_names(&self, uuids: &[&str]) -> anyhow::Result<std::collections::HashMap<String, String>> {
        SlowStore::get_entity_names(self, uuids)
    }
    fn load_embeddings(&self, uuids: &[&str]) -> anyhow::Result<std::collections::HashMap<String, Vec<f32>>> {
        SlowStore::load_embeddings(self, uuids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        store.store_thought("t-001", "missing_connection", "content", 0.85,
            "A and C should connect", Some("B mediates"), "[]", 0.7, "[]").unwrap();
        let thoughts = store.get_active_thoughts(10).unwrap();
        assert_eq!(thoughts.len(), 1);
        assert_eq!(thoughts[0].id, "t-001");
        assert!((thoughts[0].confidence - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_dismiss_thought() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store.store_thought("t-002", "silo", "content", 0.5, "desc", None, "[]", 0.3, "[]").unwrap();
        assert_eq!(store.get_active_thoughts(10).unwrap().len(), 1);
        store.dismiss_thought("t-002").unwrap();
        assert_eq!(store.get_active_thoughts(10).unwrap().len(), 0);
    }

    #[test]
    fn test_store_and_resolve_gap() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store.store_gap("g-001", "open_triad", "A-B-C", "[]", "[]", 0.9, Some(0.45), 0.8, "content").unwrap();
        assert_eq!(store.get_unresolved_gaps("open_triad", 10).unwrap().len(), 1);
        store.resolve_gap("g-001").unwrap();
        assert!(store.get_unresolved_gaps("open_triad", 10).unwrap().is_empty());
    }

    #[test]
    fn test_mark_thought_surfaced() {
        let tmp = NamedTempFile::new().unwrap();
        let store = SlowStore::open(tmp.path()).unwrap();
        store.store_thought("t-003", "golden", "content", 0.6, "desc", None, "[]", 0.5, "[]").unwrap();
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
        let embs = store.load_embeddings(&[c_id.as_str(), a_id.as_str()]).unwrap();
        assert_eq!(embs.len(), 1);
        assert_eq!(embs.get(&c_id).map(Vec::as_slice), Some(&[0.5f32, -0.25][..]));

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
}
