//! `SqliteRelationalStore` — `RelationalStore` adapter over `sqlx::SqlitePool`.
//!
//! This is the W4 foundation's SQLite backend. It binds [`Param`] values into
//! the sqlx query API and translates result rows into the backend-neutral
//! [`Row`] type.
//!
//! Connection management is delegated to `sqlx::SqlitePool` so callers get
//! pooling, concurrent access, and lifecycle management for free.

use async_trait::async_trait;
use sqlx::sqlite::{SqlitePool, SqliteRow};
use sqlx::{Column, Row as SqlxRow, TypeInfo, ValueRef};

use super::store::RelationalStore;
use super::types::{ColumnMeta, OwnedColumnValue, Param, RelationalBackend, Row};

/// SQLite-backed implementation of [`RelationalStore`].
///
/// Built around `sqlx::SqlitePool`. Use [`SqliteRelationalStore::open`] to
/// open a file-backed database and [`SqliteRelationalStore::in_memory`] for
/// a private in-memory database (typically for tests).
#[derive(Debug, Clone)]
pub struct SqliteRelationalStore {
    pool: SqlitePool,
}

impl SqliteRelationalStore {
    /// Open a SQLite database at `path`, creating it if it does not exist.
    ///
    /// `path` should be a sqlx connection string fragment — usually a plain
    /// filesystem path, but may also be `:memory:` or include sqlx query-string
    /// options. The store prepends `sqlite://` when not already present.
    pub async fn open(path: &str) -> Result<Self, sqlx::Error> {
        let url = if path.starts_with("sqlite:") {
            path.to_string()
        } else if path == ":memory:" {
            "sqlite::memory:".to_string()
        } else {
            format!("sqlite://{path}")
        };
        let pool = SqlitePool::connect(&url).await?;
        Ok(Self { pool })
    }

    /// Open a private in-memory SQLite database. Each call returns an isolated
    /// pool; rows do not leak between callers.
    pub async fn in_memory() -> Result<Self, sqlx::Error> {
        let pool = SqlitePool::connect(":memory:").await?;
        Ok(Self { pool })
    }

    /// Wrap an existing `SqlitePool`. Useful when an outer subsystem owns
    /// pool configuration (timeouts, max connections, pragmas).
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Borrow the underlying sqlx pool for advanced use.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

/// Translate one borrowed sqlx row into a backend-neutral [`Row`].
///
/// SQLite is dynamically typed at the storage level, so we inspect each
/// column's declared type (when available) and the runtime value to choose
/// a sensible [`OwnedColumnValue`] variant.
fn decode_sqlite_row(row: &SqliteRow) -> Result<Row, sqlx::Error> {
    let column_count = row.columns().len();
    let mut metas = Vec::with_capacity(column_count);
    let mut values = Vec::with_capacity(column_count);

    for (idx, col) in row.columns().iter().enumerate() {
        let sql_type = col.type_info().name().to_string();
        metas.push(ColumnMeta {
            name: col.name().to_string(),
            sql_type: sql_type.clone(),
        });

        // Probe NULL first via the raw ValueRef so we don't lose the variant to
        // type-coercion failures during fallback decoding.
        let raw = row.try_get_raw(idx)?;
        if raw.is_null() {
            values.push(OwnedColumnValue::Null);
            continue;
        }

        // Dispatch on SQLite's declared/runtime type. SQLite reports its core
        // storage classes as INTEGER, REAL, TEXT, BLOB, NULL; named affinities
        // (BOOLEAN, JSON, BIGINT, ...) map onto those at the engine level.
        let normalized = sql_type.to_ascii_uppercase();
        let owned = match normalized.as_str() {
            "BOOLEAN" | "BOOL" => OwnedColumnValue::Bool(row.try_get::<bool, _>(idx)?),
            "INTEGER" | "INT" | "BIGINT" | "INT8" | "SMALLINT" | "MEDIUMINT" | "TINYINT" => {
                OwnedColumnValue::I64(row.try_get::<i64, _>(idx)?)
            }
            "REAL" | "FLOAT" | "DOUBLE" | "DOUBLE PRECISION" | "NUMERIC" => {
                OwnedColumnValue::F64(row.try_get::<f64, _>(idx)?)
            }
            "BLOB" => OwnedColumnValue::Bytes(row.try_get::<Vec<u8>, _>(idx)?),
            "TEXT" | "CHAR" | "VARCHAR" | "CLOB" => {
                OwnedColumnValue::Text(row.try_get::<String, _>(idx)?)
            }
            "JSON" | "JSONB" => {
                // Stored as text; parse so downstream code receives structured JSON.
                let raw_text: String = row.try_get(idx)?;
                let parsed: serde_json::Value =
                    serde_json::from_str(&raw_text).map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
                OwnedColumnValue::Json(parsed)
            }
            _ => {
                // Unknown affinity. Try the SQLite storage classes in order
                // of specificity. This handles untyped expression columns
                // such as `SELECT 1` or `SELECT x'00'`.
                if let Ok(v) = row.try_get::<i64, _>(idx) {
                    OwnedColumnValue::I64(v)
                } else if let Ok(v) = row.try_get::<f64, _>(idx) {
                    OwnedColumnValue::F64(v)
                } else if let Ok(v) = row.try_get::<Vec<u8>, _>(idx) {
                    OwnedColumnValue::Bytes(v)
                } else {
                    OwnedColumnValue::Text(row.try_get::<String, _>(idx)?)
                }
            }
        };
        values.push(owned);
    }

    Ok(Row::new(metas, values))
}

/// Bind every [`Param`] in `params` onto an in-flight sqlx query builder.
fn bind_params<'q>(
    mut query: sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>>,
    params: &'q [Param<'q>],
) -> sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>> {
    for p in params {
        query = match p {
            Param::Null => query.bind(Option::<String>::None),
            Param::Bool(b) => query.bind(*b),
            Param::I64(i) => query.bind(*i),
            Param::F64(f) => query.bind(*f),
            Param::Text(s) => query.bind(*s),
            Param::Bytes(b) => query.bind(*b),
            // sqlx-sqlite has no first-class JSON encoder; persist as text so
            // round-trip works regardless of whether the column was declared JSON.
            Param::Json(v) => query.bind(v.to_string()),
        };
    }
    query
}

#[async_trait]
impl RelationalStore for SqliteRelationalStore {
    type Error = sqlx::Error;

    async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, Self::Error> {
        let query = sqlx::query(sql);
        let query = bind_params(query, params);
        let result = query.execute(&self.pool).await?;
        Ok(result.rows_affected())
    }

    async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, Self::Error> {
        let query = sqlx::query(sql);
        let query = bind_params(query, params);
        let raw_rows = query.fetch_all(&self.pool).await?;
        let mut out = Vec::with_capacity(raw_rows.len());
        for row in &raw_rows {
            out.push(decode_sqlite_row(row)?);
        }
        Ok(out)
    }

    fn backend(&self) -> RelationalBackend {
        RelationalBackend::Sqlite
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::Param;
    use super::*;
    use serde_json::json;

    /// Smoke test: open in-memory store, create a table, round-trip rows.
    #[tokio::test]
    async fn create_table_and_select_back() {
        let store = SqliteRelationalStore::in_memory()
            .await
            .expect("open in-memory sqlite");

        let affected = store
            .execute(
                "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL, qty INTEGER)",
                &[],
            )
            .await
            .expect("create table");
        // CREATE TABLE reports zero affected rows in SQLite — that's intentional.
        assert_eq!(affected, 0);

        let inserted = store
            .execute(
                "INSERT INTO items (id, name, qty) VALUES (?, ?, ?), (?, ?, ?)",
                &[
                    Param::I64(1),
                    Param::Text("alpha"),
                    Param::I64(10),
                    Param::I64(2),
                    Param::Text("beta"),
                    Param::I64(20),
                ],
            )
            .await
            .expect("insert rows");
        assert_eq!(inserted, 2);

        let rows = store
            .query("SELECT id, name, qty FROM items ORDER BY id", &[])
            .await
            .expect("select rows");
        assert_eq!(rows.len(), 2);

        let id0: i64 = rows[0].get(0).expect("decode id");
        let name0: String = rows[0].get(1).expect("decode name");
        let qty0: i64 = rows[0].get(2).expect("decode qty");
        assert_eq!(id0, 1);
        assert_eq!(name0, "alpha");
        assert_eq!(qty0, 10);

        let id1: i64 = rows[1].get(0).expect("decode id");
        let name1: String = rows[1].get(1).expect("decode name");
        assert_eq!(id1, 2);
        assert_eq!(name1, "beta");

        assert_eq!(store.backend(), RelationalBackend::Sqlite);
    }

    /// Verify NULL round-trips as `Option::None` when decoded.
    #[tokio::test]
    async fn null_round_trip() {
        let store = SqliteRelationalStore::in_memory().await.expect("open");
        store
            .execute("CREATE TABLE t (label TEXT)", &[])
            .await
            .expect("create");
        store
            .execute("INSERT INTO t (label) VALUES (?)", &[Param::Null])
            .await
            .expect("insert null");

        let rows = store
            .query("SELECT label FROM t", &[])
            .await
            .expect("select");
        assert_eq!(rows.len(), 1);
        let label: Option<String> = rows[0].get(0).expect("decode option");
        assert!(label.is_none(), "expected NULL → None, got {label:?}");
    }

    /// Every `Param` variant should encode and decode without loss.
    #[tokio::test]
    async fn each_param_variant_round_trips() {
        let store = SqliteRelationalStore::in_memory().await.expect("open");

        // One column per Param variant, with the appropriate affinity.
        store
            .execute(
                "CREATE TABLE all_kinds (
                    nullable TEXT,
                    flag BOOLEAN,
                    big INTEGER,
                    ratio REAL,
                    label TEXT,
                    payload BLOB,
                    doc JSON
                )",
                &[],
            )
            .await
            .expect("create");

        let json_value = json!({ "hello": "world", "n": 7 });
        let bytes: &[u8] = &[0x00, 0x01, 0xff, 0x42];

        store
            .execute(
                "INSERT INTO all_kinds VALUES (?, ?, ?, ?, ?, ?, ?)",
                &[
                    Param::Null,
                    Param::Bool(true),
                    Param::I64(-9_876_543_210),
                    Param::F64(std::f64::consts::PI),
                    Param::Text("veld"),
                    Param::Bytes(bytes),
                    Param::Json(&json_value),
                ],
            )
            .await
            .expect("insert all variants");

        let rows = store
            .query(
                "SELECT nullable, flag, big, ratio, label, payload, doc FROM all_kinds",
                &[],
            )
            .await
            .expect("select all variants");
        assert_eq!(rows.len(), 1);
        let r = &rows[0];

        assert!(r.get::<Option<String>>(0).expect("nullable").is_none());
        assert!(r.get::<bool>(1).expect("flag"));
        assert_eq!(r.get::<i64>(2).expect("big"), -9_876_543_210);
        let ratio: f64 = r.get(3).expect("ratio");
        assert!((ratio - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(r.get::<String>(4).expect("label"), "veld");
        assert_eq!(r.get::<Vec<u8>>(5).expect("payload"), bytes.to_vec());
        let doc: serde_json::Value = r.get(6).expect("doc");
        assert_eq!(doc, json_value);
    }

    /// Lookup by column name should find columns regardless of position.
    #[tokio::test]
    async fn column_by_name_lookup() {
        let store = SqliteRelationalStore::in_memory().await.expect("open");
        store
            .execute("CREATE TABLE kv (col1 TEXT, col2 INTEGER)", &[])
            .await
            .expect("create");
        store
            .execute(
                "INSERT INTO kv VALUES (?, ?)",
                &[Param::Text("named"), Param::I64(99)],
            )
            .await
            .expect("insert");

        let rows = store
            .query("SELECT col1, col2 FROM kv", &[])
            .await
            .expect("select");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get_by_name::<String>("col1").unwrap(), "named");
        assert_eq!(rows[0].get_by_name::<i64>("col2").unwrap(), 99);

        // Unknown column should report a helpful error rather than panic.
        let missing = rows[0].get_by_name::<String>("does_not_exist");
        assert!(missing.is_err(), "lookup of missing column must error");
    }

    /// Postgres / Supabase-specific surface area lands in a follow-up agent.
    /// Tracked here so future contributors see the placeholder.
    #[tokio::test]
    #[ignore = "postgres backend lands in follow-up W4 commit"]
    async fn postgres_backend_round_trip_placeholder() {}
}
