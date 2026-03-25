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

use anyhow::{Context, Result};
use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OpenFlags, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::graph_memory::{EdgeTier, EntityNode, LtpStatus, RelationshipEdge};

/// Statistics from a graph → SQLite sync operation
#[derive(Debug, Default)]
pub struct SyncStats {
    pub entities_upserted: usize,
    pub edges_upserted: usize,
    pub duration_ms: u64,
}

/// An open triad (U-shape) found in the graph.
///
/// A → B → C where A and C have no direct edge.
/// The topology of the gap tells us what kind of connection is missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawOpenTriad {
    /// The first endpoint (no direct edge to node_c)
    pub node_a: String,
    pub node_a_name: String,
    /// The bridge node connecting both endpoints
    pub node_b: String,
    pub node_b_name: String,
    /// The second endpoint (no direct edge to node_a)
    pub node_c: String,
    pub node_c_name: String,
    /// Strength of the A→B edge
    pub ab_strength: f32,
    /// Strength of the B→C edge
    pub bc_strength: f32,
    /// Relationship type A→B
    pub ab_relation: String,
    /// Relationship type B→C
    pub bc_relation: String,
    /// Salience of node A
    pub a_salience: f32,
    /// Salience of node C
    pub c_salience: f32,
}

/// A star gap: multiple entities connected to a hub but not to each other.
/// The hub's spokes are isolated — they form a wheel without a rim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawStarGap {
    /// The hub entity UUID
    pub hub: String,
    pub hub_name: String,
    /// Spoke entities connected to the hub
    pub spokes: Vec<(String, String)>, // (uuid, name)
    /// Number of missing inter-spoke edges
    pub missing_edges: usize,
    /// Total possible inter-spoke edges
    pub possible_edges: usize,
    /// Average edge strength from hub to spokes
    pub avg_hub_strength: f32,
}

/// A diamond gap: A→B, A→C, B→D, C→D, but no B↔C link.
/// Two parallel paths converge without their intermediaries connecting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDiamondGap {
    pub top: String,
    pub top_name: String,
    pub left: String,
    pub left_name: String,
    pub right: String,
    pub right_name: String,
    pub bottom: String,
    pub bottom_name: String,
    /// Whether left→right edge is missing
    pub missing_left_right: bool,
}

/// Cluster information for orbit gap detection
#[derive(Debug, Clone)]
pub struct EntityCluster {
    pub cluster_id: usize,
    pub entities: Vec<(String, String)>, // (uuid, name)
    pub internal_edge_count: usize,
    pub avg_internal_strength: f32,
}

/// SQLite-backed slow store for relational queries on the knowledge graph.
///
/// Runs in WAL mode: concurrent reads, single writer, no locking issues.
/// Periodically synced from the RocksDB fast store (GraphMemory).
pub struct SlowStore {
    conn: Mutex<Connection>,
    path: PathBuf,
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

        let store = Self {
            conn: Mutex::new(conn),
            path: path.to_owned(),
        };
        store.create_schema()?;
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
        ",
        )?;
        Ok(())
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
        let start = std::time::Instant::now();
        let mut stats = SyncStats::default();
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;

        Self::sync_entities(&tx, entities, &mut stats)?;
        Self::sync_edges(&tx, edges, &mut stats)?;

        tx.commit()?;
        stats.duration_ms = start.elapsed().as_millis() as u64;
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

    // =========================================================================
    // GAP DETECTION QUERIES
    // =========================================================================

    /// Find all open triads (U-shapes) in the graph.
    ///
    /// An open triad is A→B→C where no edge exists between A and C (in either direction).
    /// These represent missing inferences: two things are both related to a third,
    /// but their mutual relationship hasn't been established.
    ///
    /// Returns raw triads sorted by combined edge strength (strongest gaps first).
    pub fn find_open_triads(&self, min_strength: f32, limit: usize) -> Result<Vec<RawOpenTriad>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT
                e1.from_entity, ea.name, ea.salience,
                e1.to_entity, eb.name,
                e2.to_entity, ec.name, ec.salience,
                e1.strength, e2.strength,
                e1.relation_type, e2.relation_type
             FROM edges e1
             JOIN edges e2 ON e1.to_entity = e2.from_entity
             JOIN entities ea ON e1.from_entity = ea.uuid
             JOIN entities eb ON e1.to_entity = eb.uuid
             JOIN entities ec ON e2.to_entity = ec.uuid
             WHERE e1.from_entity != e2.to_entity
               AND e1.strength >= ?1
               AND e2.strength >= ?1
               AND NOT EXISTS (
                   SELECT 1 FROM edges e3
                   WHERE (e3.from_entity = e1.from_entity AND e3.to_entity = e2.to_entity)
                      OR (e3.from_entity = e2.to_entity AND e3.to_entity = e1.from_entity)
               )
             ORDER BY (e1.strength + e2.strength) DESC
             LIMIT ?2",
        )?;

        let triads = stmt
            .query_map(params![min_strength, limit], |row| {
                Ok(RawOpenTriad {
                    node_a: row.get(0)?,
                    node_a_name: row.get(1)?,
                    a_salience: row.get(2)?,
                    node_b: row.get(3)?,
                    node_b_name: row.get(4)?,
                    node_c: row.get(5)?,
                    node_c_name: row.get(6)?,
                    c_salience: row.get(7)?,
                    ab_strength: row.get(8)?,
                    bc_strength: row.get(9)?,
                    ab_relation: row.get(10)?,
                    bc_relation: row.get(11)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(triads)
    }

    /// Find star gaps: hub entities whose spokes have few inter-connections.
    ///
    /// A star gap indicates a hub concept whose related entities exist in isolation
    /// from each other — the wheel has no rim. These represent missing community
    /// structure: things grouped by a common concept should often relate to each other.
    pub fn find_star_gaps(
        &self,
        min_spokes: usize,
        max_connectivity: f32,
        limit: usize,
    ) -> Result<Vec<RawStarGap>> {
        let conn = self.conn.lock();

        // Step 1: Find hub entities with enough connections
        let mut hub_stmt = conn.prepare_cached(
            "SELECT e_out.from_entity, ent.name,
                    COUNT(DISTINCT e_out.to_entity) as spoke_count,
                    AVG(e_out.strength) as avg_strength
             FROM edges e_out
             JOIN entities ent ON e_out.from_entity = ent.uuid
             GROUP BY e_out.from_entity
             HAVING spoke_count >= ?1
             ORDER BY spoke_count DESC
             LIMIT ?2",
        )?;

        let hubs: Vec<(String, String, usize, f32)> = hub_stmt
            .query_map(params![min_spokes, limit * 2], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get::<_, usize>(2)?, row.get(3)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Step 2: For each hub, get spokes and check inter-spoke connectivity
        let mut spoke_stmt = conn.prepare_cached(
            "SELECT e.to_entity, ent.name
             FROM edges e
             JOIN entities ent ON e.to_entity = ent.uuid
             WHERE e.from_entity = ?1
             ORDER BY e.strength DESC",
        )?;

        let mut inter_stmt = conn.prepare_cached(
            "SELECT COUNT(*) FROM edges
             WHERE from_entity = ?1 AND to_entity = ?2",
        )?;

        let mut results = Vec::new();
        for (hub_uuid, hub_name, _spoke_count, avg_strength) in &hubs {
            let spokes: Vec<(String, String)> = spoke_stmt
                .query_map(params![hub_uuid], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            if spokes.len() < min_spokes {
                continue;
            }

            // Count existing inter-spoke edges
            let mut existing_inter = 0usize;
            let possible = spokes.len() * (spokes.len() - 1) / 2;
            for i in 0..spokes.len() {
                for j in (i + 1)..spokes.len() {
                    let count_fwd: usize =
                        inter_stmt.query_row(params![&spokes[i].0, &spokes[j].0], |row| {
                            row.get(0)
                        })?;
                    let count_rev: usize =
                        inter_stmt.query_row(params![&spokes[j].0, &spokes[i].0], |row| {
                            row.get(0)
                        })?;
                    if count_fwd > 0 || count_rev > 0 {
                        existing_inter += 1;
                    }
                }
            }

            let connectivity = if possible > 0 {
                existing_inter as f32 / possible as f32
            } else {
                1.0
            };

            if connectivity <= max_connectivity {
                results.push(RawStarGap {
                    hub: hub_uuid.clone(),
                    hub_name: hub_name.clone(),
                    missing_edges: possible - existing_inter,
                    possible_edges: possible,
                    avg_hub_strength: *avg_strength,
                    spokes,
                });
            }

            if results.len() >= limit {
                break;
            }
        }

        Ok(results)
    }

    /// Find diamond gaps: A→B, A→C, B→D, C→D, but no B↔C.
    ///
    /// Two parallel paths from A to D through different intermediaries,
    /// where the intermediaries aren't connected. This often indicates
    /// two approaches to the same problem that haven't been reconciled.
    pub fn find_diamond_gaps(&self, min_strength: f32, limit: usize) -> Result<Vec<RawDiamondGap>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT
                e1.from_entity, ea.name,   -- A (top)
                e1.to_entity, eb.name,     -- B (left)
                e2.to_entity, ec.name,     -- C (right)
                e3.to_entity, ed.name      -- D (bottom)
             FROM edges e1
             JOIN edges e2 ON e1.from_entity = e2.from_entity  -- A→B, A→C (same source)
             JOIN edges e3 ON e1.to_entity = e3.from_entity    -- B→D
             JOIN edges e4 ON e2.to_entity = e4.from_entity    -- C→D
             JOIN entities ea ON e1.from_entity = ea.uuid
             JOIN entities eb ON e1.to_entity = eb.uuid
             JOIN entities ec ON e2.to_entity = ec.uuid
             JOIN entities ed ON e3.to_entity = ed.uuid
             WHERE e3.to_entity = e4.to_entity                 -- B→D and C→D (same target)
               AND e1.to_entity < e2.to_entity                 -- B < C to avoid duplicates
               AND e1.to_entity != e2.to_entity                -- B ≠ C
               AND e1.from_entity != e3.to_entity              -- A ≠ D
               AND e1.strength >= ?1
               AND e2.strength >= ?1
               AND e3.strength >= ?1
               AND e4.strength >= ?1
               AND NOT EXISTS (
                   SELECT 1 FROM edges ex
                   WHERE (ex.from_entity = e1.to_entity AND ex.to_entity = e2.to_entity)
                      OR (ex.from_entity = e2.to_entity AND ex.to_entity = e1.to_entity)
               )
             ORDER BY (e1.strength + e2.strength + e3.strength + e4.strength) DESC
             LIMIT ?2",
        )?;

        let diamonds = stmt
            .query_map(params![min_strength, limit], |row| {
                Ok(RawDiamondGap {
                    top: row.get(0)?,
                    top_name: row.get(1)?,
                    left: row.get(2)?,
                    left_name: row.get(3)?,
                    right: row.get(4)?,
                    right_name: row.get(5)?,
                    bottom: row.get(6)?,
                    bottom_name: row.get(7)?,
                    missing_left_right: true,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(diamonds)
    }

    /// Load embeddings for a set of entity UUIDs.
    ///
    /// Returns a map from UUID string to embedding vector.
    /// Used by gap topology scoring to compute cosine similarity between gap endpoints.
    pub fn load_embeddings(&self, uuids: &[&str]) -> Result<HashMap<String, Vec<f32>>> {
        let conn = self.conn.lock();
        let mut result = HashMap::new();

        let mut stmt = conn.prepare_cached(
            "SELECT uuid, embedding FROM entities WHERE uuid = ?1 AND embedding IS NOT NULL",
        )?;

        for uuid in uuids {
            if let Ok(row) = stmt.query_row(params![uuid], |row| {
                let uuid_str: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((uuid_str, blob))
            }) {
                // Decode f32 embedding from little-endian bytes
                let floats: Vec<f32> = row
                    .1
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                if !floats.is_empty() {
                    result.insert(row.0, floats);
                }
            }
        }

        Ok(result)
    }

    /// Load ALL entity embeddings from the slow store.
    ///
    /// Returns (uuid, name, embedding) triples for all entities that have embeddings.
    /// Used by Voronoi analysis to reason about the shape of knowledge space.
    pub fn load_all_embeddings(&self) -> Result<Vec<(String, String, Vec<f32>)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT uuid, name, embedding FROM entities WHERE embedding IS NOT NULL",
        )?;

        let results = stmt
            .query_map([], |row| {
                let uuid: String = row.get(0)?;
                let name: String = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                Ok((uuid, name, blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(uuid, name, blob)| {
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                if floats.is_empty() {
                    None
                } else {
                    Some((uuid, name, floats))
                }
            })
            .collect();

        Ok(results)
    }

    /// Load all directed edges (from_uuid, to_uuid) from the slow store.
    ///
    /// Used by Planet X detection to find where edges converge.
    pub fn load_all_edge_pairs(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare_cached("SELECT from_entity, to_entity FROM edges")?;

        let results = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(results)
    }

    /// Get entity count in the slow store
    pub fn entity_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize =
            conn.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Get edge count in the slow store
    pub fn edge_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Store a detected gap topology
    #[allow(clippy::too_many_arguments)]
    pub fn store_gap(
        &self,
        id: &str,
        gap_type: &str,
        shape_signature: &str,
        entities_json: &str,
        missing_links_json: &str,
        confidence: f32,
        embedding_distance: Option<f32>,
        impact_score: f32,
        scope: &str,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO gap_topologies
                (id, gap_type, shape_signature, entities_json, missing_links_json,
                 confidence, embedding_distance, impact_score, detected_at, last_verified, scope)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                embedding_distance = excluded.embedding_distance,
                impact_score = excluded.impact_score,
                last_verified = excluded.last_verified",
            params![
                id,
                gap_type,
                shape_signature,
                entities_json,
                missing_links_json,
                confidence,
                embedding_distance,
                impact_score,
                now,
                scope,
            ],
        )?;
        Ok(())
    }

    /// Store a generated thought
    #[allow(clippy::too_many_arguments)]
    pub fn store_thought(
        &self,
        id: &str,
        kind: &str,
        scope: &str,
        confidence: f32,
        description: &str,
        hypothesis: Option<&str>,
        evidence_json: &str,
        impact_score: f32,
        entities_json: &str,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO thoughts
                (id, kind, scope, confidence, description, hypothesis,
                 evidence_json, impact_score, entities_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                description = excluded.description,
                hypothesis = excluded.hypothesis,
                impact_score = excluded.impact_score",
            params![
                id,
                kind,
                scope,
                confidence,
                description,
                hypothesis,
                evidence_json,
                impact_score,
                entities_json,
                now,
            ],
        )?;
        Ok(())
    }

    /// Get active (non-dismissed) thoughts, ordered by confidence
    pub fn get_active_thoughts(&self, limit: usize) -> Result<Vec<StoredThought>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT id, kind, scope, confidence, description, hypothesis,
                    evidence_json, impact_score, entities_json, created_at,
                    surfaced_count
             FROM thoughts
             WHERE dismissed = 0
             ORDER BY impact_score DESC, confidence DESC
             LIMIT ?1",
        )?;

        let thoughts = stmt
            .query_map(params![limit], |row| {
                Ok(StoredThought {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    scope: row.get(2)?,
                    confidence: row.get(3)?,
                    description: row.get(4)?,
                    hypothesis: row.get(5)?,
                    evidence_json: row.get(6)?,
                    impact_score: row.get(7)?,
                    entities_json: row.get(8)?,
                    created_at: row.get(9)?,
                    surfaced_count: row.get(10)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(thoughts)
    }

    /// Mark a thought as surfaced (increment count)
    pub fn mark_thought_surfaced(&self, thought_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE thoughts SET surfaced_count = surfaced_count + 1 WHERE id = ?1",
            params![thought_id],
        )?;
        Ok(())
    }

    /// Dismiss a thought (user decided it's not useful)
    pub fn dismiss_thought(&self, thought_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE thoughts SET dismissed = 1 WHERE id = ?1",
            params![thought_id],
        )?;
        Ok(())
    }

    /// Mark a gap as resolved (an edge was created that closes it)
    pub fn resolve_gap(&self, gap_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE gap_topologies SET resolved_at = ?1 WHERE id = ?2",
            params![now, gap_id],
        )?;
        Ok(())
    }

    /// Get unresolved gaps by type
    pub fn get_unresolved_gaps(&self, gap_type: &str, limit: usize) -> Result<Vec<StoredGap>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT id, gap_type, shape_signature, entities_json, missing_links_json,
                    confidence, embedding_distance, impact_score, detected_at, scope
             FROM gap_topologies
             WHERE resolved_at IS NULL AND gap_type = ?1
             ORDER BY impact_score DESC, confidence DESC
             LIMIT ?2",
        )?;

        let gaps = stmt
            .query_map(params![gap_type, limit], |row| {
                Ok(StoredGap {
                    id: row.get(0)?,
                    gap_type: row.get(1)?,
                    shape_signature: row.get(2)?,
                    entities_json: row.get(3)?,
                    missing_links_json: row.get(4)?,
                    confidence: row.get(5)?,
                    embedding_distance: row.get(6)?,
                    impact_score: row.get(7)?,
                    detected_at: row.get(8)?,
                    scope: row.get(9)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(gaps)
    }

    /// Get all entity neighbors (both directions) for orbit detection
    pub fn get_adjacency_list(&self, min_strength: f32) -> Result<HashMap<String, Vec<String>>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT from_entity, to_entity FROM edges WHERE strength >= ?1",
        )?;

        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        let rows = stmt.query_map(params![min_strength], |row| {
            let from: String = row.get(0)?;
            let to: String = row.get(1)?;
            Ok((from, to))
        })?;

        for row in rows {
            let (from, to) = row?;
            adj.entry(from.clone()).or_default().push(to.clone());
            adj.entry(to).or_default().push(from);
        }

        Ok(adj)
    }

    /// Get entity names for a set of UUIDs
    pub fn get_entity_names(&self, uuids: &[&str]) -> Result<HashMap<String, String>> {
        let conn = self.conn.lock();
        let mut result = HashMap::new();
        let mut stmt = conn.prepare_cached("SELECT uuid, name FROM entities WHERE uuid = ?1")?;

        for uuid in uuids {
            if let Ok((id, name)) = stmt.query_row(params![uuid], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                result.insert(id, name);
            }
        }

        Ok(result)
    }

    /// Database file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// A stored gap topology record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredGap {
    pub id: String,
    pub gap_type: String,
    pub shape_signature: String,
    pub entities_json: String,
    pub missing_links_json: String,
    pub confidence: f32,
    pub embedding_distance: Option<f32>,
    pub impact_score: f32,
    pub detected_at: String,
    pub scope: String,
}

/// A stored thought record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredThought {
    pub id: String,
    pub kind: String,
    pub scope: String,
    pub confidence: f32,
    pub description: String,
    pub hypothesis: Option<String>,
    pub evidence_json: String,
    pub impact_score: f32,
    pub entities_json: String,
    pub created_at: String,
    pub surfaced_count: usize,
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
}
