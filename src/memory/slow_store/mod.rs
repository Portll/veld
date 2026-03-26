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
mod queries;
mod storage;

pub use queries::{EntityCluster, RawDiamondGap, RawOpenTriad, RawStarGap};
pub use storage::{StoredGap, StoredThought};

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OpenFlags, Transaction};
use std::path::{Path, PathBuf};
use crate::graph_memory::{EdgeTier, EntityNode, LtpStatus, RelationshipEdge};

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

pub const CURRENT_SCHEMA_VERSION: i32 = 1;

/// SQLite-backed slow store for relational queries on the knowledge graph.
///
/// Runs in WAL mode: concurrent reads, single writer, no locking issues.
/// Periodically synced from the RocksDB fast store (GraphMemory).
pub struct SlowStore {
    pub(crate) conn: Mutex<Connection>,
    path: PathBuf,
    last_sync: Mutex<Option<std::time::Instant>>,
}

impl SlowStore {
    /// Open or create the slow store database
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("Failed to open slow store SQLite database")?;

        // WAL mode: concurrent readers, no blocking
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // NORMAL sync: survives process crash, not power loss (matches shodh's RocksDB async default)
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        // 8MB cache for gap analysis queries
        conn.pragma_update(None, "cache_size", "-8192")?;
        // Enable foreign keys
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // Busy timeout: wait up to 5s instead of failing immediately on contention
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        let store = Self {
            conn: Mutex::new(conn),
            path: path.to_owned(),
            last_sync: Mutex::new(None),
        };
        store.create_schema()?;
        store.check_schema_version()?;
        Ok(store)
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
                // Run migrations from v+1 to CURRENT_SCHEMA_VERSION
                // No migrations yet — framework is in place for future versions
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
        let conn = self.conn.lock();

        let gaps_deleted = conn.execute(
            "DELETE FROM gap_topologies WHERE resolved_at IS NOT NULL AND detected_at < ?1",
            params![cutoff_str],
        )?;

        let thoughts_deleted = conn.execute(
            "DELETE FROM thoughts WHERE dismissed = 1 AND created_at < ?1",
            params![cutoff_str],
        )?;

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
    /// Uses upsert (INSERT OR REPLACE) for idempotent sync.
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
        let mut stmt = tx.prepare_cached(
            "INSERT INTO edges (uuid, from_entity, to_entity, relation_type, strength,
                                tier, ltp_status, activation_count, created_at,
                                last_activated, context)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(uuid) DO UPDATE SET
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
}
