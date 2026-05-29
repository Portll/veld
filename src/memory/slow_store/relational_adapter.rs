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
//! The adapter intentionally targets the four most-called CRUD primitives
//! on the existing slow store:
//!
//! | This adapter                          | Slow-store original                   |
//! |---------------------------------------|---------------------------------------|
//! | [`store_thought`](Self::store_thought)                       | [`super::SlowStore::store_thought`]          |
//! | [`get_active_thoughts`](Self::get_active_thoughts)           | [`super::SlowStore::get_active_thoughts`]    |
//! | [`store_gap`](Self::store_gap)                               | [`super::SlowStore::store_gap`]              |
//! | [`get_unresolved_gaps`](Self::get_unresolved_gaps)           | [`super::SlowStore::get_unresolved_gaps`]    |
//!
//! Plus [`init_schema`](Self::init_schema), which runs the same DDL the
//! existing rusqlite path executes inside `SlowStore::create_schema` and
//! `check_schema_version`.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;

use crate::storage::relational::{Param, RelationalStore};

use super::storage::{StoredGap, StoredThought};

/// W4 trait-backed mirror of the slow-store CRUD surface.
///
/// Only holds an `Arc<dyn RelationalStore<Error = anyhow::Error>>`. No
/// connection management, no schema state — both live behind the trait.
///
/// **Not** a drop-in replacement for [`super::SlowStore`]. See the
/// module-level docs for the deliberate scope of this parallel path.
pub struct RelationalSlowStoreAdapter {
    store: Arc<dyn RelationalStore<Error = anyhow::Error>>,
}

impl RelationalSlowStoreAdapter {
    /// Wrap a `RelationalStore` so the slow-store query surface can be
    /// exercised against it.
    pub fn new(store: Arc<dyn RelationalStore<Error = anyhow::Error>>) -> Self {
        Self { store }
    }

    /// Borrow the underlying store. Used by tests and follow-up porting
    /// work that needs to issue ad-hoc statements.
    #[allow(dead_code)]
    pub fn store(&self) -> &Arc<dyn RelationalStore<Error = anyhow::Error>> {
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
}

#[cfg(test)]
mod tests {
    //! Tests use a small `anyhow`-erased newtype around
    //! [`SqliteRelationalStore`] because the adapter is generic over
    //! `Error = anyhow::Error` but the in-tree SQLite backend yields
    //! `sqlx::Error`. The wrapper does the trivial `.map_err(Into::into)`.
    //!
    //! When a production `Error = anyhow::Error` backend lands, this shim
    //! can be deleted and the tests can wrap that backend directly.

    use super::*;
    use crate::storage::relational::{Row, SqliteRelationalStore};
    use async_trait::async_trait;
    use std::sync::Arc;

    /// Thin newtype that re-erases `sqlx::Error` as `anyhow::Error`.
    struct AnyhowSqlite(SqliteRelationalStore);

    #[async_trait]
    impl RelationalStore for AnyhowSqlite {
        type Error = anyhow::Error;

        async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64> {
            self.0.execute(sql, params).await.map_err(Into::into)
        }

        async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>> {
            self.0.query(sql, params).await.map_err(Into::into)
        }

        fn backend(&self) -> crate::storage::relational::RelationalBackend {
            self.0.backend()
        }
    }

    async fn fresh_adapter() -> RelationalSlowStoreAdapter {
        let store = SqliteRelationalStore::in_memory()
            .await
            .expect("open in-memory sqlite");
        let erased: Arc<dyn RelationalStore<Error = anyhow::Error>> =
            Arc::new(AnyhowSqlite(store));
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
}
