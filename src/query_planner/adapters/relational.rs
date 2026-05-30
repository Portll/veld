//! Relational candidate source backed by the W4 `RelationalStore` trait.
//!
//! `RealRelationalQuerier` queries the slow-store `memories` table whose
//! schema is:
//!
//! ```text
//!   memories (
//!       user_id      TEXT NOT NULL,
//!       memory_id    TEXT NOT NULL,
//!       lsn          INTEGER NOT NULL,
//!       memory_bincode BLOB NOT NULL,
//!       importance   REAL NOT NULL,
//!       updated_at   TEXT NOT NULL,
//!       PRIMARY KEY (user_id, memory_id)
//!   )
//! ```
//!
//! ## SQL-injection posture
//!
//! Every parameter value is bound through the `RelationalStore::Param`
//! enum (never interpolated). Column names supplied by callers are
//! validated against the [`ALLOWED_RELATIONAL_COLUMNS`] whitelist before
//! they reach the SQL string — predicates referencing unknown columns
//! resolve to `Ok(vec![])` (scan) or `Ok(false)` (matches) rather than
//! producing a `WHERE` clause the backend would reject. This keeps the
//! adapter useful as a default-deny gate without dragging full SQL
//! validation into the planner.

use crate::query_planner::executor::RelationalQuerier;
use crate::query_planner::predicate::RelationalPredicate;
use crate::storage::relational::{BoxError, Param, RelationalStore};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::sync::Arc;

/// Columns the planner is allowed to reference in `RelationalPredicate`s.
///
/// Anything outside this list is treated as "unknown" — the adapter returns
/// empty / false rather than embedding the unknown identifier in SQL.
pub const ALLOWED_RELATIONAL_COLUMNS: &[&str] = &[
    "user_id",
    "memory_id",
    "lsn",
    "importance",
    "updated_at",
];

/// Adapter that translates `RelationalPredicate`s into SQL against the
/// slow-store `memories` table.
#[derive(Clone)]
pub struct RealRelationalQuerier {
    store: Arc<dyn RelationalStore<Error = BoxError>>,
}

impl RealRelationalQuerier {
    pub fn new(store: Arc<dyn RelationalStore<Error = BoxError>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl RelationalQuerier for RealRelationalQuerier {
    async fn scan(&self, user_id: &str, p: &RelationalPredicate) -> Result<Vec<String>> {
        let (sql, params) = match build_scan(user_id, p) {
            Some(pair) => pair,
            None => return Ok(Vec::new()),
        };
        let rows = self
            .store
            .query(&sql, &params)
            .await
            .map_err(|e| anyhow::anyhow!("relational scan failed: {e}"))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let memory_id: String = row
                .get(0)
                .map_err(|e| anyhow::anyhow!("decode memory_id: {e}"))?;
            out.push(memory_id);
        }
        Ok(out)
    }

    async fn matches(
        &self,
        user_id: &str,
        memory_id: &str,
        p: &RelationalPredicate,
    ) -> Result<bool> {
        // `matches` is the per-id probe; we re-use the scan filter and add
        // an extra `memory_id = ?` clause, projecting `COUNT(1)`.
        let (where_sql, mut params) = match build_where(user_id, p) {
            Some(pair) => pair,
            None => return Ok(false),
        };
        let sql = format!(
            "SELECT COUNT(1) FROM memories WHERE memory_id = ? AND {}",
            where_sql
        );
        // memory_id binds in front of the predicate's params (the format
        // string places it first), so push it to the head of the params
        // vector — Param does not impl Default so we build a new Vec.
        let mut full = Vec::with_capacity(params.len() + 1);
        full.push(Param::Text(memory_id));
        full.append(&mut params);
        let rows = self
            .store
            .query(&sql, &full)
            .await
            .map_err(|e| anyhow::anyhow!("relational matches failed: {e}"))?;
        let count: i64 = rows
            .first()
            .map(|r| r.get::<i64>(0))
            .transpose()
            .map_err(|e| anyhow::anyhow!("decode count: {e}"))?
            .unwrap_or(0);
        Ok(count > 0)
    }
}

/// Build the full `SELECT memory_id FROM memories WHERE …` statement plus
/// its bound parameters. Returns `None` when the predicate references a
/// column outside [`ALLOWED_RELATIONAL_COLUMNS`].
fn build_scan<'a>(
    user_id: &'a str,
    p: &'a RelationalPredicate,
) -> Option<(String, Vec<Param<'a>>)> {
    let (where_sql, params) = build_where(user_id, p)?;
    let sql = format!(
        "SELECT memory_id FROM memories WHERE {} ORDER BY updated_at DESC",
        where_sql
    );
    Some((sql, params))
}

/// Build only the WHERE-clause fragment + its bound parameters. Returns
/// `None` if the predicate's column is not whitelisted.
fn build_where<'a>(user_id: &'a str, p: &'a RelationalPredicate) -> Option<(String, Vec<Param<'a>>)> {
    // Every clause is tenant-scoped — the planner's tenant binding lands
    // here as the leading filter regardless of the predicate.
    let mut params: Vec<Param<'a>> = vec![Param::Text(user_id)];
    let predicate_sql = match p {
        RelationalPredicate::UserIdEquals(uid) => {
            // The leading user_id = ? above already covers the tenant
            // binding; this variant simply rejects when the request's
            // declared user_id disagrees.
            if uid != user_id {
                // Empty match — return a clause that never resolves to true.
                return Some(("user_id = ? AND 1 = 0".to_string(), params));
            }
            "1 = 1".to_string()
        }
        RelationalPredicate::Equals { column, value } => {
            if !ALLOWED_RELATIONAL_COLUMNS.contains(&column.as_str()) {
                return None;
            }
            push_json_param(&mut params, value);
            format!("{column} = ?")
        }
        RelationalPredicate::In { column, values } => {
            if !ALLOWED_RELATIONAL_COLUMNS.contains(&column.as_str()) {
                return None;
            }
            if values.is_empty() {
                return Some(("user_id = ? AND 1 = 0".to_string(), params));
            }
            let placeholders: Vec<&str> = std::iter::repeat_n("?", values.len()).collect();
            for v in values {
                push_json_param(&mut params, v);
            }
            format!("{column} IN ({})", placeholders.join(", "))
        }
        RelationalPredicate::Range { column, lo, hi } => {
            if !ALLOWED_RELATIONAL_COLUMNS.contains(&column.as_str()) {
                return None;
            }
            let mut clauses = Vec::new();
            if let Some(lo) = lo {
                push_json_param(&mut params, lo);
                clauses.push(format!("{column} >= ?"));
            }
            if let Some(hi) = hi {
                push_json_param(&mut params, hi);
                clauses.push(format!("{column} <= ?"));
            }
            if clauses.is_empty() {
                "1 = 1".to_string()
            } else {
                clauses.join(" AND ")
            }
        }
    };
    Some((format!("user_id = ? AND {predicate_sql}"), params))
}

/// Borrow a JSON-encoded predicate value as the closest matching [`Param`].
/// JSON's number type covers both int and float; we pick `I64` for whole
/// numbers in range and `F64` otherwise. Anything we can't represent as a
/// bound parameter (objects, arrays) binds as `Text` with the JSON
/// serialisation — backends that don't store JSON will simply fail to
/// match, which is the correct conservative behaviour for an unknown
/// shape.
///
/// The lifetime trick: we leak nothing; the caller owns the source
/// `JsonValue` and we borrow into it via `Param` variants. Strings borrow
/// directly; numbers and booleans are stored by value in `Param`.
fn push_json_param<'a>(params: &mut Vec<Param<'a>>, value: &'a JsonValue) {
    match value {
        JsonValue::Null => params.push(Param::Null),
        JsonValue::Bool(b) => params.push(Param::Bool(*b)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                params.push(Param::I64(i));
            } else if let Some(f) = n.as_f64() {
                params.push(Param::F64(f));
            } else {
                // u64 > i64::MAX falls through to F64; precision loss is
                // acceptable for filter predicates.
                params.push(Param::F64(n.as_f64().unwrap_or(0.0)));
            }
        }
        JsonValue::String(s) => params.push(Param::Text(s.as_str())),
        // Compound shapes go through as JSON-text. Backends that store JSON
        // can still match; backends that store the column as a scalar will
        // not — that mismatch surfaces as a "no rows" result rather than a
        // type error, which the planner already handles correctly.
        JsonValue::Array(_) | JsonValue::Object(_) => params.push(Param::Json(value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::relational::SqliteRelationalStore;

    /// Tiny wrapper that re-erases `sqlx::Error` to [`BoxError`] so the
    /// querier can hold an `Arc<dyn RelationalStore<Error = BoxError>>`.
    struct BoxErrorSqlite(SqliteRelationalStore);

    #[async_trait]
    impl RelationalStore for BoxErrorSqlite {
        type Error = BoxError;
        async fn execute(
            &self,
            sql: &str,
            params: &[Param<'_>],
        ) -> Result<u64, BoxError> {
            self.0
                .execute(sql, params)
                .await
                .map_err(BoxError::new)
        }
        async fn query(
            &self,
            sql: &str,
            params: &[Param<'_>],
        ) -> Result<Vec<crate::storage::relational::Row>, BoxError> {
            self.0
                .query(sql, params)
                .await
                .map_err(BoxError::new)
        }
        fn backend(&self) -> crate::storage::relational::RelationalBackend {
            self.0.backend()
        }
    }

    async fn fixture_querier() -> RealRelationalQuerier {
        let sqlite = SqliteRelationalStore::in_memory()
            .await
            .expect("open in-memory sqlite");
        let store: Arc<dyn RelationalStore<Error = BoxError>> = Arc::new(BoxErrorSqlite(sqlite));
        store
            .execute(
                "CREATE TABLE memories (
                    user_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    lsn INTEGER NOT NULL,
                    memory_bincode BLOB NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, memory_id)
                )",
                &[],
            )
            .await
            .expect("create memories table");
        // Seed three rows for user alice and one for bob.
        for (user, id, lsn, importance, ts) in &[
            ("alice", "m-1", 1i64, 0.4f64, "2026-01-01T00:00:00Z"),
            ("alice", "m-2", 2i64, 0.9f64, "2026-01-02T00:00:00Z"),
            ("alice", "m-3", 3i64, 0.5f64, "2026-01-03T00:00:00Z"),
            ("bob", "m-7", 4i64, 0.7f64, "2026-01-04T00:00:00Z"),
        ] {
            store
                .execute(
                    "INSERT INTO memories (user_id, memory_id, lsn, memory_bincode, importance, updated_at) VALUES (?, ?, ?, X'00', ?, ?)",
                    &[
                        Param::Text(user),
                        Param::Text(id),
                        Param::I64(*lsn),
                        Param::F64(*importance),
                        Param::Text(ts),
                    ],
                )
                .await
                .expect("seed insert");
        }
        RealRelationalQuerier::new(store)
    }

    #[tokio::test]
    async fn user_id_equals_returns_only_that_tenants_rows() {
        let q = fixture_querier().await;
        let p = RelationalPredicate::UserIdEquals("alice".into());
        let hits = q.scan("alice", &p).await.expect("scan");
        assert_eq!(hits.len(), 3);
        assert!(hits.contains(&"m-1".to_string()));
        assert!(!hits.contains(&"m-7".to_string()));
    }

    #[tokio::test]
    async fn equals_on_whitelisted_column_filters_correctly() {
        let q = fixture_querier().await;
        let p = RelationalPredicate::Equals {
            column: "memory_id".into(),
            value: JsonValue::String("m-2".into()),
        };
        let hits = q.scan("alice", &p).await.expect("scan");
        assert_eq!(hits, vec!["m-2".to_string()]);
    }

    #[tokio::test]
    async fn equals_on_unknown_column_returns_empty_rather_than_failing() {
        let q = fixture_querier().await;
        let p = RelationalPredicate::Equals {
            column: "totally_made_up".into(),
            value: JsonValue::String("anything".into()),
        };
        let hits = q.scan("alice", &p).await.expect("scan");
        assert!(hits.is_empty(), "unknown column should resolve to empty");
    }

    #[tokio::test]
    async fn range_filter_uses_both_bounds_when_present() {
        let q = fixture_querier().await;
        let p = RelationalPredicate::Range {
            column: "importance".into(),
            lo: Some(JsonValue::from(0.5_f64)),
            hi: Some(JsonValue::from(0.95_f64)),
        };
        let mut hits = q.scan("alice", &p).await.expect("scan");
        hits.sort();
        assert_eq!(hits, vec!["m-2".to_string(), "m-3".to_string()]);
    }

    #[tokio::test]
    async fn in_with_multiple_values_returns_intersection() {
        let q = fixture_querier().await;
        let p = RelationalPredicate::In {
            column: "memory_id".into(),
            values: vec![
                JsonValue::String("m-1".into()),
                JsonValue::String("m-3".into()),
                JsonValue::String("m-7".into()),
            ],
        };
        let mut hits = q.scan("alice", &p).await.expect("scan");
        hits.sort();
        // m-7 belongs to bob and is filtered by tenant scope.
        assert_eq!(hits, vec!["m-1".to_string(), "m-3".to_string()]);
    }

    #[tokio::test]
    async fn matches_returns_true_only_for_id_in_filter() {
        let q = fixture_querier().await;
        let p = RelationalPredicate::Range {
            column: "lsn".into(),
            lo: Some(JsonValue::from(2_i64)),
            hi: None,
        };
        assert!(q.matches("alice", "m-2", &p).await.expect("matches m-2"));
        assert!(q.matches("alice", "m-3", &p).await.expect("matches m-3"));
        assert!(!q.matches("alice", "m-1", &p).await.expect("matches m-1"));
        // Tenant boundary: bob's row never matches alice's scope.
        assert!(!q.matches("alice", "m-7", &p).await.expect("matches m-7"));
    }
}
