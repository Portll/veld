//! `MssqlRelationalStore` — `RelationalStore` over Microsoft SQL Server via
//! tiberius. Gated behind the `mssql` feature.
//!
//! ## Why tiberius, and the shape it forces
//!
//! sqlx has no maintained MSSQL driver, so this backend uses tiberius — the
//! de-facto MSSQL client. tiberius is *not* sqlx:
//!
//! - it owns a single `Client` whose `execute`/`query` take `&mut self` and
//!   has no built-in pool, so the store holds a `tokio::sync::Mutex<Client>`
//!   — one connection, serialised. The slow store is single-writer, so this
//!   is adequate (and matches the bridge's single-thread assumption);
//! - it uses `@P1, @P2` placeholders, so the trait's `?` placeholders are
//!   rewritten by [`translate_placeholders`];
//! - parameters bind through tiberius's `Query` builder, which owns the
//!   bound values and sidesteps the `&[&dyn ToSql]` lifetime juggling.
//!
//! ## Limitations (v1)
//!
//! - single connection: no pool, no reconnect-on-drop;
//! - typed NULLs bind as text (as with the Postgres backend);
//! - the row decoder probes Rust types in order (i64, i32, f64, f32, bool,
//!   bytes, str) — adequate for the slow-store schema; types outside that
//!   set decode to `Null`;
//! - **compile-verified only** in this repo: there is no live SQL Server to
//!   integration-test against here. The pure placeholder translation is
//!   unit-tested.

use async_trait::async_trait;
use tiberius::{Client, Config};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_util::compat::{Compat, TokioAsyncWriteCompatExt};

use super::store::RelationalStore;
use super::types::{ColumnMeta, OwnedColumnValue, Param, RelationalBackend, Row};
use super::BoxError;

/// Map a tiberius error to [`BoxError`] via its message. tiberius's error
/// type is not guaranteed `Sync`, so we don't store it behind the trait
/// object directly — the `Display` string is preserved in an
/// `io::Error::other`, which is `Send + Sync + 'static`.
fn mssql_err(e: tiberius::error::Error) -> BoxError {
    BoxError::new(std::io::Error::other(e.to_string()))
}

/// The concrete tiberius client type: TDS over a tokio TCP stream adapted to
/// the futures `AsyncRead`/`AsyncWrite` tiberius expects.
type MssqlClient = Client<Compat<TcpStream>>;

/// MSSQL-backed implementation of [`RelationalStore`].
pub struct MssqlRelationalStore {
    client: Mutex<MssqlClient>,
}

impl MssqlRelationalStore {
    /// Connect using an ADO-style connection string, e.g.
    /// `Server=tcp:host,1433;Database=db;User Id=sa;Password=pw;TrustServerCertificate=true`.
    pub async fn connect(ado_string: &str) -> Result<Self, tiberius::error::Error> {
        let config = Config::from_ado_string(ado_string)?;
        let tcp = TcpStream::connect(config.get_addr()).await?;
        tcp.set_nodelay(true)?;
        let client = Client::connect(config, tcp.compat_write()).await?;
        Ok(Self {
            client: Mutex::new(client),
        })
    }

    /// Wrap an already-connected tiberius client.
    pub fn from_client(client: MssqlClient) -> Self {
        Self {
            client: Mutex::new(client),
        }
    }
}

/// Rewrite SQLite-style positional `?` placeholders into MSSQL `@P1, @P2`
/// placeholders, numbered left-to-right from 1. Single-quote string literals
/// are skipped so a `?` inside `'…'` is preserved.
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
                out.push_str("@P");
                out.push_str(&n.to_string());
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Build a tiberius `Query` from translated SQL + bound params. Binds in
/// declaration order, which maps onto `@P1, @P2, …`.
fn build_query<'a>(sql: &'a str, params: &'a [Param<'a>]) -> tiberius::Query<'a> {
    let mut q = tiberius::Query::new(sql);
    for p in params {
        match p {
            // Text-typed NULL — see the module-level "typed NULLs" note.
            Param::Null => q.bind(Option::<&str>::None),
            Param::Bool(b) => q.bind(*b),
            Param::I64(i) => q.bind(*i),
            Param::F64(f) => q.bind(*f),
            Param::Text(s) => q.bind(*s),
            Param::Bytes(b) => q.bind(*b),
            Param::Json(v) => q.bind(v.to_string()),
        }
    }
    q
}

/// Decode one tiberius row into the backend-neutral [`Row`]. Probes Rust
/// types in order; the first that the column accepts wins, `None` → `Null`.
fn decode_row(row: &tiberius::Row) -> Row {
    let mut metas = Vec::with_capacity(row.columns().len());
    let mut values = Vec::with_capacity(row.columns().len());
    for (idx, col) in row.columns().iter().enumerate() {
        metas.push(ColumnMeta {
            name: col.name().to_string(),
            sql_type: format!("{:?}", col.column_type()),
        });
        values.push(decode_cell(row, idx));
    }
    Row::new(metas, values)
}

fn decode_cell(row: &tiberius::Row, idx: usize) -> OwnedColumnValue {
    // Order matters: i64 before i32 (BIGINT vs INT), bytes before str.
    if let Ok(v) = row.try_get::<i64, _>(idx) {
        return v.map_or(OwnedColumnValue::Null, OwnedColumnValue::I64);
    }
    if let Ok(v) = row.try_get::<i32, _>(idx) {
        return v.map_or(OwnedColumnValue::Null, |x| OwnedColumnValue::I64(x as i64));
    }
    if let Ok(v) = row.try_get::<f64, _>(idx) {
        return v.map_or(OwnedColumnValue::Null, OwnedColumnValue::F64);
    }
    if let Ok(v) = row.try_get::<f32, _>(idx) {
        return v.map_or(OwnedColumnValue::Null, |x| OwnedColumnValue::F64(x as f64));
    }
    if let Ok(v) = row.try_get::<bool, _>(idx) {
        return v.map_or(OwnedColumnValue::Null, OwnedColumnValue::Bool);
    }
    if let Ok(v) = row.try_get::<&[u8], _>(idx) {
        return v.map_or(OwnedColumnValue::Null, |x| OwnedColumnValue::Bytes(x.to_vec()));
    }
    if let Ok(v) = row.try_get::<&str, _>(idx) {
        return v.map_or(OwnedColumnValue::Null, |x| OwnedColumnValue::Text(x.to_string()));
    }
    OwnedColumnValue::Null
}

#[async_trait]
impl RelationalStore for MssqlRelationalStore {
    type Error = BoxError;

    async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, Self::Error> {
        let translated = translate_placeholders(sql);
        let query = build_query(&translated, params);
        let mut client = self.client.lock().await;
        let result = query.execute(&mut *client).await.map_err(mssql_err)?;
        Ok(result.total())
    }

    async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, Self::Error> {
        let translated = translate_placeholders(sql);
        let query = build_query(&translated, params);
        let mut client = self.client.lock().await;
        let stream = query.query(&mut *client).await.map_err(mssql_err)?;
        let rows = stream.into_first_result().await.map_err(mssql_err)?;
        Ok(rows.iter().map(decode_row).collect())
    }

    fn backend(&self) -> RelationalBackend {
        RelationalBackend::Mssql
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The live execute/query path needs a running SQL Server and is not
    // exercised here. The pure placeholder translation is unit-tested.

    #[test]
    fn translate_numbers_placeholders_as_named() {
        assert_eq!(
            translate_placeholders("SELECT * FROM t WHERE a = ? AND b = ?"),
            "SELECT * FROM t WHERE a = @P1 AND b = @P2"
        );
    }

    #[test]
    fn translate_skips_string_literals() {
        assert_eq!(
            translate_placeholders("SELECT '?' WHERE x = ?"),
            "SELECT '?' WHERE x = @P1"
        );
    }

    #[test]
    fn translate_identity_without_placeholders() {
        let sql = "CREATE TABLE t (id BIGINT PRIMARY KEY, name NVARCHAR(MAX))";
        assert_eq!(translate_placeholders(sql), sql);
    }
}
