//! Parallel slow-store path against the W4 `RelationalStore` trait.
//!
//! This module exists to prove the `RelationalStore` trait surface is
//! expressive enough to drive the SQL the existing rusqlite-backed
//! [`super::SlowStore`] runs. **It is deliberately not wired into any
//! call-site.** The existing [`super::SlowStore`] continues to handle every
//! production gap-analysis query unchanged.
//!
//! ## Why this lives here
//!
//! The W4 follow-up that ports the slow store onto the trait will face two
//! risks the codebase has not yet absorbed:
//!
//! 1. **Async/sync mismatch.** The current slow store exposes synchronous
//!    methods backed by a `parking_lot::Mutex<Connection>`. The trait is
//!    `async fn` because the production backends behind it (sqlx Postgres,
//!    sqlx Sqlite, Supabase HTTP) are all async-native.
//! 2. **`Param` shape vs. existing column types.** The DDL uses `TEXT` for
//!    timestamps and pre-serialised JSON, `REAL` for `f32`/`f64`, and stores
//!    nullable floats as `NULL` rows. The trait's `Param` has no
//!    `Timestamp`/`F32`/`Option` variants, so those have to funnel through
//!    `Param::Text` / `Param::F64` / `Param::Null` — the translations below
//!    pin those decisions in code.
//!
//! Proving the surface on a parallel adapter, with the schema and the most
//! representative queries from the slow store, lets the follow-up porting
//! agent diff this file against the existing implementation rather than
//! re-litigating the trait choices mid-port.
//!
//! ## Methods mirrored
//!
//! The adapter mirrors the gap-analysis CRUD primitives plus the
//! intent-log `memories` projection table — the latter is what the W6
//! query planner's [`crate::query_planner::adapters::RealRelationalQuerier`]
//! already *reads* through this same trait, so covering its writes here
//! proves the full round-trip ahead of the production cutover:
//!
//! | This adapter                          | Slow-store original                   |
//! |---------------------------------------|---------------------------------------|
//! | [`store_thought`](Self::store_thought)                       | [`super::SlowStore::store_thought`]          |
//! | [`get_active_thoughts`](Self::get_active_thoughts)           | [`super::SlowStore::get_active_thoughts`]    |
//! | [`store_gap`](Self::store_gap)                               | [`super::SlowStore::store_gap`]              |
//! | [`get_unresolved_gaps`](Self::get_unresolved_gaps)           | [`super::SlowStore::get_unresolved_gaps`]    |
//! | [`upsert_memory`](Self::upsert_memory)                       | [`super::SlowStore::upsert_memory`]          |
//! | [`anchor_memory_importance`](Self::anchor_memory_importance) | [`super::SlowStore::anchor_memory_importance`] |
//! | [`delete_memory`](Self::delete_memory)                       | [`super::SlowStore::delete_memory`]          |
//! | [`get_memory_blob`](Self::get_memory_blob)                   | [`super::SlowStore::get_memory_blob`]        |
//! | [`count_memories`](Self::count_memories)                     | [`super::SlowStore::count_memories`]         |
//!
//! Plus [`init_schema`](Self::init_schema), which runs the same DDL the
//! existing rusqlite path executes inside `SlowStore::create_schema` and
//! `check_schema_version`.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;

use crate::storage::relational::{Param, RelationalBackend, RelationalStore};

use super::storage::{StoredGap, StoredMemoryRow, StoredThought};

// ─── Dialect-aware `memories` projection SQL ─────────────────────────────
//
// The slow-store schema was authored for SQLite, but the W4 cutover runs
// the `memories` table against Postgres/Supabase/MSSQL too. Each backend
// translates `?` placeholders, but the *dialect* differs in three ways the
// memories path hits: column types (Postgres is strictly typed — `lsn` is
// bound i64→BIGINT, `importance` f64→DOUBLE PRECISION, blob→BYTEA; SQLite's
// INTEGER/REAL/BLOB would reject those binds), the upsert syntax (MSSQL has
// no `ON CONFLICT` — it needs `MERGE`), and `CREATE TABLE IF NOT EXISTS`
// (not valid T-SQL). These are selected per [`RelationalBackend`] below.

const MEMORIES_DDL_SQLITE: &str = "CREATE TABLE IF NOT EXISTS memories (
    user_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    lsn INTEGER NOT NULL,
    memory_bincode BLOB NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, memory_id)
)";

const MEMORIES_DDL_POSTGRES: &str = "CREATE TABLE IF NOT EXISTS memories (
    user_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    lsn BIGINT NOT NULL,
    memory_bincode BYTEA NOT NULL,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, memory_id)
)";

// No `CREATE TABLE IF NOT EXISTS` in T-SQL (guard with OBJECT_ID); no BLOB
// (VARBINARY(MAX)); the composite text PK must be bounded + NONCLUSTERED to
// fit the index key-size limit (user_id/memory_id are short ids / UUIDs).
const MEMORIES_DDL_MSSQL: &str = "IF OBJECT_ID(N'memories', N'U') IS NULL
CREATE TABLE memories (
    user_id NVARCHAR(255) NOT NULL,
    memory_id NVARCHAR(255) NOT NULL,
    lsn BIGINT NOT NULL,
    memory_bincode VARBINARY(MAX) NOT NULL,
    importance FLOAT NOT NULL DEFAULT 0.5,
    updated_at NVARCHAR(MAX) NOT NULL,
    CONSTRAINT pk_memories PRIMARY KEY NONCLUSTERED (user_id, memory_id)
)";

const MEMORIES_INDEXES_SQLITE_PG: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_lsn ON memories(lsn)",
];

const MEMORIES_INDEXES_MSSQL: &[&str] = &[
    "IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_memories_user' AND object_id = OBJECT_ID(N'memories')) CREATE INDEX idx_memories_user ON memories(user_id)",
    "IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_memories_lsn' AND object_id = OBJECT_ID(N'memories')) CREATE INDEX idx_memories_lsn ON memories(lsn)",
];

const UPSERT_MEMORY_ON_CONFLICT: &str = "INSERT INTO memories
    (user_id, memory_id, lsn, memory_bincode, importance, updated_at)
 VALUES (?, ?, ?, ?, ?, ?)
 ON CONFLICT(user_id, memory_id) DO UPDATE SET
    lsn = excluded.lsn,
    memory_bincode = excluded.memory_bincode,
    importance = excluded.importance,
    updated_at = excluded.updated_at
 WHERE excluded.lsn >= memories.lsn";

// MSSQL has no ON CONFLICT; the LSN-gated upsert becomes a MERGE. Same six
// bound params in the same order, so the call site is unchanged.
const UPSERT_MEMORY_MERGE: &str = "MERGE memories AS t
USING (SELECT ? AS user_id, ? AS memory_id, ? AS lsn, ? AS memory_bincode, ? AS importance, ? AS updated_at) AS s
ON t.user_id = s.user_id AND t.memory_id = s.memory_id
WHEN MATCHED AND s.lsn >= t.lsn THEN
    UPDATE SET lsn = s.lsn, memory_bincode = s.memory_bincode, importance = s.importance, updated_at = s.updated_at
WHEN NOT MATCHED THEN
    INSERT (user_id, memory_id, lsn, memory_bincode, importance, updated_at)
    VALUES (s.user_id, s.memory_id, s.lsn, s.memory_bincode, s.importance, s.updated_at);";

/// Dialect-correct `CREATE TABLE` for the `memories` projection.
fn memories_table_ddl(backend: &RelationalBackend) -> &'static str {
    match backend {
        RelationalBackend::Postgres | RelationalBackend::Supabase => MEMORIES_DDL_POSTGRES,
        RelationalBackend::Mssql => MEMORIES_DDL_MSSQL,
        _ => MEMORIES_DDL_SQLITE,
    }
}

/// Dialect-correct index DDL for the `memories` projection.
fn memories_index_ddls(backend: &RelationalBackend) -> &'static [&'static str] {
    match backend {
        RelationalBackend::Mssql => MEMORIES_INDEXES_MSSQL,
        // SQLite and Postgres/Supabase both accept `CREATE INDEX IF NOT EXISTS`.
        _ => MEMORIES_INDEXES_SQLITE_PG,
    }
}

/// Dialect-correct LSN-gated upsert for the `memories` projection.
fn upsert_memory_sql(backend: &RelationalBackend) -> &'static str {
    match backend {
        RelationalBackend::Mssql => UPSERT_MEMORY_MERGE,
        _ => UPSERT_MEMORY_ON_CONFLICT,
    }
}

/// W4 trait-backed mirror of the slow-store CRUD surface.
///
/// Only holds an `Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>`. No
/// connection management, no schema state — both live behind the trait.
///
/// **Not** a drop-in replacement for [`super::SlowStore`]. See the
/// module-level docs for the deliberate scope of this parallel path.
///
/// `Clone` is a cheap `Arc` bump — the `*_blocking` wrappers clone the
/// adapter into the `'static` future they hand to the bridge runtime.
#[derive(Clone)]
pub struct RelationalSlowStoreAdapter {
    store: Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>,
}

// The whole adapter is a deliberately-unwired parallel path (see the
// module docs): it proves the `RelationalStore` surface against the
// slow-store SQL ahead of the production cutover and is exercised only by
// its own tests. Until a call-site adopts it, every method reads as
// "unused" in a non-test build, so the allow lives at the impl level
// rather than being sprinkled per method.
#[allow(dead_code)]
impl RelationalSlowStoreAdapter {
    /// Wrap a `RelationalStore` so the slow-store query surface can be
    /// exercised against it.
    pub fn new(store: Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>>) -> Self {
        Self { store }
    }

    /// Borrow the underlying store. Used by tests and follow-up porting
    /// work that needs to issue ad-hoc statements.
    pub fn store(&self) -> &Arc<dyn RelationalStore<Error = crate::storage::relational::BoxError>> {
        &self.store
    }

    /// Apply the slow-store DDL idempotently.
    ///
    /// The statements below are the exact `CREATE TABLE IF NOT EXISTS` /
    /// `CREATE INDEX IF NOT EXISTS` set from `SlowStore::create_schema`,
    /// split into separate `execute` calls because the trait's `execute`
    /// surface is single-statement. The semantics are unchanged.
    pub async fn init_schema(&self) -> Result<()> {
        // Statements split out of the original `execute_batch` block. They
        // must stay in declaration order so the `edges.FOREIGN KEY` clause
        // sees `entities` before it runs.
        const DDL: &[&str] = &[
            // Mirror of GraphMemory entities (synced periodically).
            "CREATE TABLE IF NOT EXISTS entities (
                uuid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                labels TEXT NOT NULL DEFAULT '[]',
                salience REAL NOT NULL DEFAULT 0.5,
                mention_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                embedding BLOB
            )",
            // Mirror of GraphMemory relationship edges.
            "CREATE TABLE IF NOT EXISTS edges (
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
            )",
            "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_entity)",
            "CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_entity)",
            "CREATE INDEX IF NOT EXISTS idx_edges_strength ON edges(strength DESC)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_pair
                ON edges(from_entity, to_entity)",
            // Detected gap topologies.
            "CREATE TABLE IF NOT EXISTS gap_topologies (
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
            )",
            "CREATE INDEX IF NOT EXISTS idx_gaps_type ON gap_topologies(gap_type)",
            "CREATE INDEX IF NOT EXISTS idx_gaps_confidence
                ON gap_topologies(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_gaps_unresolved
                ON gap_topologies(resolved_at) WHERE resolved_at IS NULL",
            // Generated thoughts from gap analysis.
            "CREATE TABLE IF NOT EXISTS thoughts (
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
            )",
            "CREATE INDEX IF NOT EXISTS idx_thoughts_active
                ON thoughts(dismissed, confidence DESC)
                WHERE dismissed = 0",
            // Intent-log projection: one row per live memory per user. The
            // (user_id, memory_id) PK is the idempotency key; `lsn` is the
            // monotonic write-skew tie-breaker.
            "CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                lsn INTEGER NOT NULL,
                memory_bincode BLOB NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, memory_id)
            )",
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_lsn ON memories(lsn)",
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )",
        ];

        for stmt in DDL {
            self.store
                .execute(stmt, &[])
                .await
                .with_context(|| format!("slow-store DDL failed: {}", &stmt[..stmt.len().min(60)]))?;
        }

        // Bootstrap the schema_version row when missing. Matches the
        // `current_version is None` branch in `check_schema_version`.
        let existing = self
            .store
            .query("SELECT MAX(version) AS v FROM schema_version", &[])
            .await
            .context("query schema_version")?;
        let already_versioned = existing
            .first()
            .and_then(|r| r.get::<Option<i64>>(0).ok())
            .map(|opt| opt.is_some())
            .unwrap_or(false);
        if !already_versioned {
            let now = Utc::now().to_rfc3339();
            self.store
                .execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    &[
                        Param::I64(super::CURRENT_SCHEMA_VERSION as i64),
                        Param::Text(&now),
                    ],
                )
                .await
                .context("insert initial schema_version row")?;
        }

        Ok(())
    }

    /// Mirror of [`super::SlowStore::store_thought`].
    ///
    /// Same `ON CONFLICT(id) DO UPDATE` upsert as the rusqlite path. `f32`
    /// columns funnel through `Param::F64` because the trait has no `F32`
    /// variant — SQLite's NUMERIC affinity preserves the value.
    #[allow(clippy::too_many_arguments)]
    pub async fn store_thought(
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
        let now = Utc::now().to_rfc3339();
        let hypothesis_param = match hypothesis {
            Some(h) => Param::Text(h),
            None => Param::Null,
        };
        self.store
            .execute(
                "INSERT INTO thoughts
                    (id, kind, scope, confidence, description, hypothesis,
                     evidence_json, impact_score, entities_json, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(id) DO UPDATE SET
                    confidence = excluded.confidence,
                    description = excluded.description,
                    hypothesis = excluded.hypothesis,
                    impact_score = excluded.impact_score",
                &[
                    Param::Text(id),
                    Param::Text(kind),
                    Param::Text(scope),
                    Param::F64(confidence as f64),
                    Param::Text(description),
                    hypothesis_param,
                    Param::Text(evidence_json),
                    Param::F64(impact_score as f64),
                    Param::Text(entities_json),
                    Param::Text(&now),
                ],
            )
            .await
            .context("relational adapter: store_thought")?;
        Ok(())
    }

    /// Mirror of [`super::SlowStore::get_active_thoughts`].
    pub async fn get_active_thoughts(&self, limit: usize) -> Result<Vec<StoredThought>> {
        let rows = self
            .store
            .query(
                "SELECT id, kind, scope, confidence, description, hypothesis,
                        evidence_json, impact_score, entities_json, created_at,
                        surfaced_count
                 FROM thoughts
                 WHERE dismissed = 0
                 ORDER BY impact_score DESC, confidence DESC
                 LIMIT ?",
                &[Param::I64(limit as i64)],
            )
            .await
            .context("relational adapter: get_active_thoughts")?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let confidence: f64 = row.get(3).context("decode confidence")?;
            let impact_score: f64 = row.get(7).context("decode impact_score")?;
            let surfaced_count_i64: i64 = row.get(10).context("decode surfaced_count")?;
            out.push(StoredThought {
                id: row.get(0).context("decode id")?,
                kind: row.get(1).context("decode kind")?,
                scope: row.get(2).context("decode scope")?,
                confidence: confidence as f32,
                description: row.get(4).context("decode description")?,
                hypothesis: row.get(5).context("decode hypothesis")?,
                evidence_json: row.get(6).context("decode evidence_json")?,
                impact_score: impact_score as f32,
                entities_json: row.get(8).context("decode entities_json")?,
                created_at: row.get(9).context("decode created_at")?,
                surfaced_count: surfaced_count_i64 as usize,
            });
        }
        Ok(out)
    }

    /// Mirror of [`super::SlowStore::store_gap`].
    ///
    /// Exercises `Param::Null` for the nullable `embedding_distance` column
    /// — the trait surface's most awkward translation, since `Option<f32>`
    /// has no first-class variant.
    #[allow(clippy::too_many_arguments)]
    pub async fn store_gap(
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
        let now = Utc::now().to_rfc3339();
        let embedding_param = match embedding_distance {
            Some(v) => Param::F64(v as f64),
            None => Param::Null,
        };
        // The original SQL reused `?9` for both `detected_at` and
        // `last_verified`. The trait surface uses positional `?`
        // placeholders without index reuse, so we bind the timestamp twice.
        self.store
            .execute(
                "INSERT INTO gap_topologies
                    (id, gap_type, shape_signature, entities_json, missing_links_json,
                     confidence, embedding_distance, impact_score, detected_at, last_verified, scope)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(id) DO UPDATE SET
                    confidence = excluded.confidence,
                    embedding_distance = excluded.embedding_distance,
                    impact_score = excluded.impact_score,
                    last_verified = excluded.last_verified",
                &[
                    Param::Text(id),
                    Param::Text(gap_type),
                    Param::Text(shape_signature),
                    Param::Text(entities_json),
                    Param::Text(missing_links_json),
                    Param::F64(confidence as f64),
                    embedding_param,
                    Param::F64(impact_score as f64),
                    Param::Text(&now),
                    Param::Text(&now),
                    Param::Text(scope),
                ],
            )
            .await
            .context("relational adapter: store_gap")?;
        Ok(())
    }

    /// Mirror of [`super::SlowStore::get_unresolved_gaps`].
    pub async fn get_unresolved_gaps(
        &self,
        gap_type: &str,
        limit: usize,
    ) -> Result<Vec<StoredGap>> {
        let rows = self
            .store
            .query(
                "SELECT id, gap_type, shape_signature, entities_json, missing_links_json,
                        confidence, embedding_distance, impact_score, detected_at, scope
                 FROM gap_topologies
                 WHERE resolved_at IS NULL AND gap_type = ?
                 ORDER BY impact_score DESC, confidence DESC
                 LIMIT ?",
                &[Param::Text(gap_type), Param::I64(limit as i64)],
            )
            .await
            .context("relational adapter: get_unresolved_gaps")?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let confidence: f64 = row.get(5).context("decode confidence")?;
            let embedding_distance: Option<f64> =
                row.get(6).context("decode embedding_distance")?;
            let impact_score: f64 = row.get(7).context("decode impact_score")?;
            out.push(StoredGap {
                id: row.get(0).context("decode id")?,
                gap_type: row.get(1).context("decode gap_type")?,
                shape_signature: row.get(2).context("decode shape_signature")?,
                entities_json: row.get(3).context("decode entities_json")?,
                missing_links_json: row.get(4).context("decode missing_links_json")?,
                confidence: confidence as f32,
                embedding_distance: embedding_distance.map(|v| v as f32),
                impact_score: impact_score as f32,
                detected_at: row.get(8).context("decode detected_at")?,
                scope: row.get(9).context("decode scope")?,
            });
        }
        Ok(out)
    }

    /// Mirror of [`super::SlowStore::upsert_memory`].
    ///
    /// Same `(user_id, memory_id)` UPSERT gated on `excluded.lsn >=
    /// memories.lsn`, so a late replay can never overwrite a newer live
    /// write. `lsn` funnels through `Param::I64` (the column is `INTEGER`)
    /// and the bincode payload through `Param::Bytes`.
    pub async fn upsert_memory(
        &self,
        user_id: &str,
        memory_id: &str,
        lsn: u64,
        memory_bincode: &[u8],
        importance: f32,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.store
            .execute(
                upsert_memory_sql(&self.store.backend()),
                &[
                    Param::Text(user_id),
                    Param::Text(memory_id),
                    Param::I64(lsn as i64),
                    Param::Bytes(memory_bincode),
                    Param::F64(importance as f64),
                    Param::Text(&now),
                ],
            )
            .await
            .context("relational adapter: upsert_memory")?;
        Ok(())
    }

    /// Create just the `memories` projection table (+ its indexes) in the
    /// backend, with dialect-correct DDL for SQLite / Postgres / MSSQL.
    ///
    /// This is the only schema the W4 cutover needs: the gap-analysis tables
    /// (entities/edges/gaps/thoughts) stay in the rusqlite slow store, so —
    /// unlike [`Self::init_schema`], which mirrors the full SQLite schema for
    /// parity tests — they are intentionally NOT created here. Idempotent.
    pub async fn init_memories_schema(&self) -> Result<()> {
        let backend = self.store.backend();
        self.store
            .execute(memories_table_ddl(&backend), &[])
            .await
            .context("create memories table")?;
        for ddl in memories_index_ddls(&backend) {
            self.store
                .execute(ddl, &[])
                .await
                .with_context(|| format!("create memories index: {ddl}"))?;
        }
        Ok(())
    }

    /// Mirror of [`super::SlowStore::anchor_memory_importance`].
    ///
    /// Updates only `importance` (plus `lsn`/`updated_at`), gated on
    /// `lsn >= current`. The original rusqlite SQL reused the `?4` (lsn)
    /// placeholder in both the SET and the WHERE; the trait surface uses
    /// non-reusable positional `?`, so `lsn` is bound twice.
    pub async fn anchor_memory_importance(
        &self,
        user_id: &str,
        memory_id: &str,
        lsn: u64,
        importance: f32,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.store
            .execute(
                "UPDATE memories
                    SET importance = ?, lsn = ?, updated_at = ?
                  WHERE user_id = ? AND memory_id = ? AND ? >= lsn",
                &[
                    Param::F64(importance as f64),
                    Param::I64(lsn as i64),
                    Param::Text(&now),
                    Param::Text(user_id),
                    Param::Text(memory_id),
                    Param::I64(lsn as i64),
                ],
            )
            .await
            .context("relational adapter: anchor_memory_importance")?;
        Ok(())
    }

    /// Mirror of [`super::SlowStore::delete_memory`]. Idempotent — deleting
    /// a non-existent row succeeds with zero rows affected.
    pub async fn delete_memory(&self, user_id: &str, memory_id: &str) -> Result<()> {
        self.store
            .execute(
                "DELETE FROM memories WHERE user_id = ? AND memory_id = ?",
                &[Param::Text(user_id), Param::Text(memory_id)],
            )
            .await
            .context("relational adapter: delete_memory")?;
        Ok(())
    }

    /// Mirror of [`super::SlowStore::get_memory_blob`]. Returns `None` when
    /// the `(user_id, memory_id)` row is absent.
    pub async fn get_memory_blob(
        &self,
        user_id: &str,
        memory_id: &str,
    ) -> Result<Option<StoredMemoryRow>> {
        let rows = self
            .store
            .query(
                "SELECT user_id, memory_id, lsn, memory_bincode, importance, updated_at
                 FROM memories
                 WHERE user_id = ? AND memory_id = ?",
                &[Param::Text(user_id), Param::Text(memory_id)],
            )
            .await
            .context("relational adapter: get_memory_blob")?;
        let row = match rows.into_iter().next() {
            Some(r) => r,
            None => return Ok(None),
        };
        let lsn: i64 = row.get(2).context("decode lsn")?;
        let importance: f64 = row.get(4).context("decode importance")?;
        Ok(Some(StoredMemoryRow {
            user_id: row.get(0).context("decode user_id")?,
            memory_id: row.get(1).context("decode memory_id")?,
            lsn: lsn as u64,
            memory_bincode: row.get(3).context("decode memory_bincode")?,
            importance: importance as f32,
            updated_at: row.get(5).context("decode updated_at")?,
        }))
    }

    /// Mirror of [`super::SlowStore::count_memories`].
    pub async fn count_memories(&self, user_id: &str) -> Result<u64> {
        let rows = self
            .store
            .query(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?",
                &[Param::Text(user_id)],
            )
            .await
            .context("relational adapter: count_memories")?;
        let count: i64 = rows
            .first()
            .map(|r| r.get::<i64>(0))
            .transpose()
            .context("decode count")?
            .unwrap_or(0);
        Ok(count as u64)
    }

    // ─── Synchronous (bridged) wrappers ──────────────────────────────────
    //
    // The intent-log `SqliteProjection::apply` path is synchronous and runs
    // on a tokio worker thread, so it cannot `.await`. These wrappers run
    // the async write on the shared relational bridge runtime and block the
    // caller for the result — see [`crate::storage::relational::blocking`].
    // Each clones the adapter (cheap `Arc` bump) and owns its arguments so
    // the future handed to the bridge is `Send + 'static`.

    /// Synchronous, bridge-driven [`Self::upsert_memory`].
    pub fn upsert_memory_blocking(
        &self,
        user_id: &str,
        memory_id: &str,
        lsn: u64,
        memory_bincode: &[u8],
        importance: f32,
    ) -> Result<()> {
        let adapter = self.clone();
        let user_id = user_id.to_string();
        let memory_id = memory_id.to_string();
        let blob = memory_bincode.to_vec();
        crate::storage::relational::blocking::bridge_block_on(async move {
            adapter
                .upsert_memory(&user_id, &memory_id, lsn, &blob, importance)
                .await
        })
        .map_err(|e| anyhow::anyhow!("relational bridge dropped before completing: {e}"))?
    }

    /// Synchronous, bridge-driven [`Self::anchor_memory_importance`].
    pub fn anchor_memory_importance_blocking(
        &self,
        user_id: &str,
        memory_id: &str,
        lsn: u64,
        importance: f32,
    ) -> Result<()> {
        let adapter = self.clone();
        let user_id = user_id.to_string();
        let memory_id = memory_id.to_string();
        crate::storage::relational::blocking::bridge_block_on(async move {
            adapter
                .anchor_memory_importance(&user_id, &memory_id, lsn, importance)
                .await
        })
        .map_err(|e| anyhow::anyhow!("relational bridge dropped before completing: {e}"))?
    }

    /// Synchronous, bridge-driven [`Self::delete_memory`].
    pub fn delete_memory_blocking(&self, user_id: &str, memory_id: &str) -> Result<()> {
        let adapter = self.clone();
        let user_id = user_id.to_string();
        let memory_id = memory_id.to_string();
        crate::storage::relational::blocking::bridge_block_on(async move {
            adapter.delete_memory(&user_id, &memory_id).await
        })
        .map_err(|e| anyhow::anyhow!("relational bridge dropped before completing: {e}"))?
    }
}

#[cfg(test)]
mod tests {
    //! Tests use a small `BoxError`-erased newtype around
    //! [`SqliteRelationalStore`] because the adapter is generic over
    //! `Error = BoxError` but the in-tree SQLite backend yields
    //! `sqlx::Error`. The wrapper does the trivial
    //! `.map_err(BoxError::new)`.
    //!
    //! When a production backend lands with `Error = BoxError`, this shim
    //! can be deleted and the tests can wrap that backend directly.

    use super::*;
    use crate::storage::relational::{BoxError, Row, SqliteRelationalStore};
    use async_trait::async_trait;
    use std::sync::Arc;

    #[test]
    fn memories_ddl_is_dialect_correct() {
        // SQLite: BLOB + INTEGER + REAL.
        let lite = memories_table_ddl(&RelationalBackend::Sqlite);
        assert!(lite.contains("BLOB") && lite.contains("CREATE TABLE IF NOT EXISTS"));
        // Postgres: strict types — BYTEA / BIGINT / DOUBLE PRECISION.
        let pg = memories_table_ddl(&RelationalBackend::Postgres);
        assert!(pg.contains("BYTEA") && pg.contains("BIGINT") && pg.contains("DOUBLE PRECISION"));
        assert!(!pg.contains("BLOB"), "postgres DDL must not use BLOB");
        // Supabase shares the Postgres dialect.
        assert_eq!(memories_table_ddl(&RelationalBackend::Supabase), pg);
        // MSSQL: OBJECT_ID guard, VARBINARY(MAX), NONCLUSTERED PK.
        let ms = memories_table_ddl(&RelationalBackend::Mssql);
        assert!(ms.contains("VARBINARY(MAX)") && ms.contains("OBJECT_ID") && ms.contains("NONCLUSTERED"));
        assert!(!ms.contains("IF NOT EXISTS"), "T-SQL has no CREATE TABLE IF NOT EXISTS");
    }

    #[test]
    fn upsert_sql_is_dialect_correct() {
        assert!(upsert_memory_sql(&RelationalBackend::Sqlite).contains("ON CONFLICT"));
        assert!(upsert_memory_sql(&RelationalBackend::Postgres).contains("ON CONFLICT"));
        let ms = upsert_memory_sql(&RelationalBackend::Mssql);
        assert!(ms.contains("MERGE memories") && ms.contains("WHEN MATCHED AND s.lsn >= t.lsn"));
        assert!(!ms.contains("ON CONFLICT"), "T-SQL has no ON CONFLICT");
    }

    /// Thin newtype that re-erases `sqlx::Error` as [`BoxError`].
    struct BoxErrorSqlite(SqliteRelationalStore);

    #[async_trait]
    impl RelationalStore for BoxErrorSqlite {
        type Error = BoxError;

        async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, BoxError> {
            self.0
                .execute(sql, params)
                .await
                .map_err(BoxError::new)
        }

        async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
            self.0
                .query(sql, params)
                .await
                .map_err(BoxError::new)
        }

        fn backend(&self) -> crate::storage::relational::RelationalBackend {
            self.0.backend()
        }
    }

    async fn fresh_adapter() -> RelationalSlowStoreAdapter {
        let store = SqliteRelationalStore::in_memory()
            .await
            .expect("open in-memory sqlite");
        let erased: Arc<dyn RelationalStore<Error = BoxError>> = Arc::new(BoxErrorSqlite(store));
        RelationalSlowStoreAdapter::new(erased)
    }

    #[tokio::test]
    async fn adapter_init_schema_on_in_memory_sqlite_succeeds() {
        let adapter = fresh_adapter().await;
        adapter
            .init_schema()
            .await
            .expect("init_schema runs without error");

        // Calling init_schema twice must be idempotent — every DDL uses
        // `IF NOT EXISTS` and the schema_version bootstrap checks the
        // existing row before inserting.
        adapter
            .init_schema()
            .await
            .expect("init_schema is idempotent");

        // schema_version must have exactly one row at CURRENT_SCHEMA_VERSION.
        let rows = adapter
            .store
            .query("SELECT version FROM schema_version", &[])
            .await
            .expect("query schema_version");
        assert_eq!(rows.len(), 1, "schema_version should be initialised once");
        let v: i64 = rows[0].get(0).expect("decode version");
        assert_eq!(v, super::super::CURRENT_SCHEMA_VERSION as i64);
    }

    #[tokio::test]
    async fn adapter_round_trips_one_record() {
        let adapter = fresh_adapter().await;
        adapter.init_schema().await.expect("schema");

        adapter
            .store_thought(
                "t-rt-001",
                "missing_connection",
                "content",
                0.85,
                "A and C should connect",
                Some("B mediates"),
                "[\"e1\"]",
                0.7,
                "[\"a\",\"c\"]",
            )
            .await
            .expect("store_thought");

        let thoughts = adapter
            .get_active_thoughts(10)
            .await
            .expect("get_active_thoughts");
        assert_eq!(thoughts.len(), 1);
        let t = &thoughts[0];
        assert_eq!(t.id, "t-rt-001");
        assert_eq!(t.kind, "missing_connection");
        assert_eq!(t.scope, "content");
        assert!((t.confidence - 0.85).abs() < 1e-5);
        assert_eq!(t.description, "A and C should connect");
        assert_eq!(t.hypothesis.as_deref(), Some("B mediates"));
        assert_eq!(t.evidence_json, "[\"e1\"]");
        assert!((t.impact_score - 0.7).abs() < 1e-5);
        assert_eq!(t.entities_json, "[\"a\",\"c\"]");
        assert!(!t.created_at.is_empty(), "created_at must be populated");
        assert_eq!(t.surfaced_count, 0);

        // Gap path — also exercises the Param::Null branch for
        // embedding_distance.
        adapter
            .store_gap(
                "g-rt-001",
                "open_triad",
                "A-B-C",
                "[\"A\",\"B\",\"C\"]",
                "[[\"A\",\"C\"]]",
                0.9,
                None,
                0.8,
                "content",
            )
            .await
            .expect("store_gap");

        let gaps = adapter
            .get_unresolved_gaps("open_triad", 10)
            .await
            .expect("get_unresolved_gaps");
        assert_eq!(gaps.len(), 1);
        let g = &gaps[0];
        assert_eq!(g.id, "g-rt-001");
        assert_eq!(g.gap_type, "open_triad");
        assert_eq!(g.shape_signature, "A-B-C");
        assert_eq!(g.entities_json, "[\"A\",\"B\",\"C\"]");
        assert_eq!(g.missing_links_json, "[[\"A\",\"C\"]]");
        assert!((g.confidence - 0.9).abs() < 1e-5);
        assert!(g.embedding_distance.is_none());
        assert!((g.impact_score - 0.8).abs() < 1e-5);
        assert_eq!(g.scope, "content");
    }

    #[tokio::test]
    async fn adapter_get_missing_returns_none() {
        let adapter = fresh_adapter().await;
        adapter.init_schema().await.expect("schema");

        // No thoughts inserted — list must be empty.
        let thoughts = adapter
            .get_active_thoughts(10)
            .await
            .expect("get_active_thoughts on empty table");
        assert!(thoughts.is_empty(), "expected no thoughts on a fresh table");

        // No gaps of the requested type either.
        let gaps = adapter
            .get_unresolved_gaps("open_triad", 10)
            .await
            .expect("get_unresolved_gaps on empty table");
        assert!(gaps.is_empty(), "expected no gaps on a fresh table");
    }

    #[tokio::test]
    async fn adapter_memories_round_trip_lsn_gate_and_isolation() {
        let adapter = fresh_adapter().await;
        adapter.init_schema().await.expect("schema");

        // Fresh tenant — empty.
        assert_eq!(adapter.count_memories("alice").await.expect("count"), 0);
        assert!(adapter
            .get_memory_blob("alice", "m-1")
            .await
            .expect("get")
            .is_none());

        // Insert and read back the full row, including the BLOB payload.
        adapter
            .upsert_memory("alice", "m-1", 5, &[1u8, 2, 3], 0.4)
            .await
            .expect("upsert");
        let row = adapter
            .get_memory_blob("alice", "m-1")
            .await
            .expect("get")
            .expect("row present");
        assert_eq!(row.user_id, "alice");
        assert_eq!(row.memory_id, "m-1");
        assert_eq!(row.lsn, 5);
        assert_eq!(row.memory_bincode, vec![1u8, 2, 3]);
        assert!((row.importance - 0.4).abs() < 1e-5);
        assert_eq!(adapter.count_memories("alice").await.expect("count"), 1);

        // Higher LSN wins.
        adapter
            .upsert_memory("alice", "m-1", 9, &[9u8, 9], 0.8)
            .await
            .expect("upsert higher lsn");
        let row = adapter.get_memory_blob("alice", "m-1").await.unwrap().unwrap();
        assert_eq!(row.lsn, 9);
        assert_eq!(row.memory_bincode, vec![9u8, 9]);

        // Stale (lower) LSN must be a no-op, not an error or an overwrite.
        adapter
            .upsert_memory("alice", "m-1", 2, &[0u8], 0.1)
            .await
            .expect("stale upsert is a no-op");
        let row = adapter.get_memory_blob("alice", "m-1").await.unwrap().unwrap();
        assert_eq!(row.lsn, 9, "stale lsn must not overwrite the newer row");
        assert_eq!(row.memory_bincode, vec![9u8, 9]);

        // Anchor updates importance + lsn only (gated on lsn >= current).
        adapter
            .anchor_memory_importance("alice", "m-1", 12, 0.95)
            .await
            .expect("anchor");
        let row = adapter.get_memory_blob("alice", "m-1").await.unwrap().unwrap();
        assert!((row.importance - 0.95).abs() < 1e-5);
        assert_eq!(row.lsn, 12);
        assert_eq!(row.memory_bincode, vec![9u8, 9], "anchor leaves the blob untouched");

        // Tenant isolation — bob shares the table but sees none of alice's rows.
        assert_eq!(adapter.count_memories("bob").await.expect("count"), 0);

        // Delete is idempotent.
        adapter.delete_memory("alice", "m-1").await.expect("delete");
        assert!(adapter
            .get_memory_blob("alice", "m-1")
            .await
            .unwrap()
            .is_none());
        adapter
            .delete_memory("alice", "m-1")
            .await
            .expect("delete again is a no-op");
        assert_eq!(adapter.count_memories("alice").await.expect("count"), 0);
    }
}
