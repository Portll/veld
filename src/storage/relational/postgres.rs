//! `PostgresRelationalStore` — `RelationalStore` over `sqlx::PgPool`.
//!
//! The W4 Postgres backend, gated behind the `postgres` feature. Binds
//! [`Param`] values into sqlx-postgres queries and decodes result rows into
//! the backend-neutral [`Row`] type, exactly like the SQLite backend.
//!
//! ## Placeholder dialect
//!
//! The trait's SQL (slow-store adapter, W6 query planner) is written with
//! SQLite-style positional `?` placeholders. Postgres uses `$1, $2, …`, so
//! every statement is rewritten by [`translate_placeholders`] before it
//! reaches sqlx. The rewrite is single-quote-aware: a literal `?` inside a
//! string literal is left untouched.
//!
//! ## Known limitations (v1)
//!
//! - **Typed NULLs.** Postgres is strongly typed and sqlx binds `Param::Null`
//!   with a `TEXT` type OID, so a bare `Param::Null` into a non-text column
//!   (e.g. the gap table's `embedding_distance REAL`) needs an explicit cast
//!   in the SQL (`CAST(? AS REAL)`). The `memories` projection — the live
//!   cutover target — binds no `Param::Null`, so it is unaffected.
//! - **JSON / NUMERIC.** JSON is bound as TEXT and JSON/JSONB columns are
//!   decoded as text, matching the SQLite backend (the slow-store schema
//!   stores JSON in `TEXT` columns). A real `jsonb` column or a `NUMERIC`
//!   result would need the `sqlx/json` / `sqlx/rust_decimal` features.

use async_trait::async_trait;
use sqlx::postgres::{PgPool, PgRow};
use sqlx::{Column, Row as SqlxRow, TypeInfo, ValueRef};

use super::store::RelationalStore;
use super::types::{ColumnMeta, OwnedColumnValue, Param, RelationalBackend, Row};

/// Postgres-backed implementation of [`RelationalStore`], over `sqlx::PgPool`.
#[derive(Debug, Clone)]
pub struct PostgresRelationalStore {
    pool: PgPool,
}

impl PostgresRelationalStore {
    /// Connect to a Postgres server. `url` is a standard libpq/sqlx
    /// connection string, e.g. `postgres://user:pass@host:5432/db`.
    pub async fn connect(url: &str) -> Result<Self, sqlx::Error> {
        let pool = PgPool::connect(url).await?;
        Ok(Self { pool })
    }

    /// Wrap an existing `PgPool` (caller owns pool configuration — timeouts,
    /// max connections, TLS).
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Borrow the underlying sqlx pool for advanced use.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

/// Rewrite SQLite-style positional `?` placeholders into Postgres `$N`
/// placeholders, numbered left-to-right from 1.
///
/// Single-quote string literals are skipped, so a `?` inside `'…'` is
/// preserved verbatim. Quote handling toggles on every `'`; an escaped
/// quote pair (`''`) leaves the scanner "inside string" between the two
/// quotes, which keeps any `?` there literal — the conservative, correct
/// choice for the slow-store / planner SQL this backend runs.
fn translate_placeholders(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len() + 8);
    let mut in_string = false;
    let mut n: u32 = 0;
    for ch in sql.chars() {
        match ch {
            '\'' => {
                in_string = !in_string;
                out.push(ch);
            }
            '?' if !in_string => {
                n += 1;
                out.push('$');
                out.push_str(&n.to_string());
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Bind every [`Param`] onto an in-flight sqlx-postgres query.
fn bind_params<'q>(
    mut query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    params: &'q [Param<'q>],
) -> sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments> {
    for p in params {
        query = match p {
            // TEXT-typed NULL — see the module-level "typed NULLs" note.
            Param::Null => query.bind(Option::<String>::None),
            Param::Bool(b) => query.bind(*b),
            Param::I64(i) => query.bind(*i),
            Param::F64(f) => query.bind(*f),
            Param::Text(s) => query.bind(*s),
            Param::Bytes(b) => query.bind(*b),
            // Persist JSON as text (matches the SQLite backend).
            Param::Json(v) => query.bind(v.to_string()),
        };
    }
    query
}

/// Decode one borrowed sqlx-postgres row into a backend-neutral [`Row`].
///
/// Postgres is strongly typed, so each column is decoded with the exact Rust
/// type sqlx expects for its OID, then widened into the neutral
/// [`OwnedColumnValue`] (e.g. `INT4` → `i32` → `I64`).
fn decode_pg_row(row: &PgRow) -> Result<Row, sqlx::Error> {
    let mut metas = Vec::with_capacity(row.columns().len());
    let mut values = Vec::with_capacity(row.columns().len());

    for (idx, col) in row.columns().iter().enumerate() {
        let sql_type = col.type_info().name().to_string();
        metas.push(ColumnMeta {
            name: col.name().to_string(),
            sql_type: sql_type.clone(),
        });

        // Probe NULL via the raw ValueRef so the variant survives regardless
        // of the column's declared type.
        let raw = row.try_get_raw(idx)?;
        if raw.is_null() {
            values.push(OwnedColumnValue::Null);
            continue;
        }

        let owned = match sql_type.to_ascii_uppercase().as_str() {
            "BOOL" => OwnedColumnValue::Bool(row.try_get::<bool, _>(idx)?),
            "INT2" => OwnedColumnValue::I64(row.try_get::<i16, _>(idx)? as i64),
            "INT4" => OwnedColumnValue::I64(row.try_get::<i32, _>(idx)? as i64),
            "INT8" => OwnedColumnValue::I64(row.try_get::<i64, _>(idx)?),
            "FLOAT4" => OwnedColumnValue::F64(row.try_get::<f32, _>(idx)? as f64),
            "FLOAT8" => OwnedColumnValue::F64(row.try_get::<f64, _>(idx)?),
            "BYTEA" => OwnedColumnValue::Bytes(row.try_get::<Vec<u8>, _>(idx)?),
            // TEXT family, plus JSON-stored-as-text and unknown/inferred
            // types (e.g. a bare `SELECT $1` parameter), all decode to String.
            _ => OwnedColumnValue::Text(row.try_get::<String, _>(idx)?),
        };
        values.push(owned);
    }

    Ok(Row::new(metas, values))
}

#[async_trait]
impl RelationalStore for PostgresRelationalStore {
    type Error = sqlx::Error;

    async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, Self::Error> {
        let translated = translate_placeholders(sql);
        let query = sqlx::query(&translated);
        let query = bind_params(query, params);
        let result = query.execute(&self.pool).await?;
        Ok(result.rows_affected())
    }

    async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, Self::Error> {
        let translated = translate_placeholders(sql);
        let query = sqlx::query(&translated);
        let query = bind_params(query, params);
        let raw_rows = query.fetch_all(&self.pool).await?;
        let mut out = Vec::with_capacity(raw_rows.len());
        for row in &raw_rows {
            out.push(decode_pg_row(row)?);
        }
        Ok(out)
    }

    fn backend(&self) -> RelationalBackend {
        RelationalBackend::Postgres
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The live execute/query path needs a running Postgres and is exercised
    // in integration, not here. The placeholder translation is the pure,
    // backend-defining logic and is fully unit-tested below.

    #[test]
    fn translate_numbers_placeholders_left_to_right() {
        assert_eq!(
            translate_placeholders("SELECT * FROM t WHERE a = ? AND b = ?"),
            "SELECT * FROM t WHERE a = $1 AND b = $2"
        );
    }

    #[test]
    fn translate_skips_question_marks_inside_string_literals() {
        assert_eq!(
            translate_placeholders("SELECT '?' AS lit WHERE x = ?"),
            "SELECT '?' AS lit WHERE x = $1"
        );
        assert_eq!(
            translate_placeholders("UPDATE t SET note = 'why? because' WHERE id = ?"),
            "UPDATE t SET note = 'why? because' WHERE id = $1"
        );
    }

    #[test]
    fn translate_is_identity_with_no_placeholders() {
        let sql = "CREATE TABLE t (id INT8 PRIMARY KEY, name TEXT NOT NULL)";
        assert_eq!(translate_placeholders(sql), sql);
    }

    #[test]
    fn translate_handles_double_digit_placeholders() {
        assert_eq!(
            translate_placeholders("VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"),
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
        );
    }
}
