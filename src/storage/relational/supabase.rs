//! `SupabaseRelationalStore` — a thin [`PostgresRelationalStore`] wrapper for
//! Supabase-hosted Postgres.
//!
//! Supabase is Postgres, so this delegates every query to an inner
//! [`PostgresRelationalStore`]; it exists to (a) build the Supabase
//! connection URL from a project ref, (b) force TLS (`sslmode=require`), and
//! (c) report [`RelationalBackend::Supabase`] so callers and metrics can tell
//! the deployment apart. Gated behind the `postgres` feature alongside the
//! Postgres backend it wraps.
//!
//! ## Credential
//!
//! [`SupabaseRelationalStore::connect`] takes the project ref and the
//! **database password** (Supabase dashboard → Project Settings → Database).
//! That is *not* the `anon` / `service_role` API keys — those are JWTs for
//! the PostgREST/Storage HTTP APIs and cannot authenticate a Postgres
//! connection. (The original W4 brief labelled this `service_role_key`; a
//! sqlx/libpq connection rejects that, so the parameter is the DB password.)

use async_trait::async_trait;

use super::postgres::PostgresRelationalStore;
use super::store::RelationalStore;
use super::types::{Param, RelationalBackend, Row};

/// Supabase-hosted Postgres backend. Wraps [`PostgresRelationalStore`].
#[derive(Debug, Clone)]
pub struct SupabaseRelationalStore {
    inner: PostgresRelationalStore,
}

impl SupabaseRelationalStore {
    /// Connect to a Supabase project's Postgres database over TLS.
    ///
    /// `project_ref` is the project reference (the `<ref>` in
    /// `db.<ref>.supabase.co`); `db_password` is the database password (see
    /// the module docs — not the API JWTs).
    pub async fn connect(project_ref: &str, db_password: &str) -> Result<Self, sqlx::Error> {
        let url = supabase_connection_url(project_ref, db_password);
        let inner = PostgresRelationalStore::connect(&url).await?;
        Ok(Self { inner })
    }

    /// Wrap an already-connected [`PostgresRelationalStore`] as a Supabase
    /// store (e.g. when the caller built the pool with custom options).
    pub fn from_postgres(inner: PostgresRelationalStore) -> Self {
        Self { inner }
    }

    /// Borrow the inner Postgres store.
    pub fn inner(&self) -> &PostgresRelationalStore {
        &self.inner
    }
}

/// Build the Supabase **direct** Postgres connection URL with TLS required.
///
/// Form: `postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres?sslmode=require`.
/// The password is percent-encoded so credentials containing `@`, `:`, `/`,
/// spaces, etc. don't corrupt the authority component.
pub fn supabase_connection_url(project_ref: &str, db_password: &str) -> String {
    format!(
        "postgresql://postgres:{pw}@db.{project_ref}.supabase.co:5432/postgres?sslmode=require",
        pw = percent_encode_userinfo(db_password),
    )
}

/// Percent-encode a connection-string credential per RFC 3986 userinfo
/// rules: everything outside the unreserved set (`A-Z a-z 0-9 - . _ ~`) is
/// escaped as `%XX`. Keeps a password with reserved characters from breaking
/// URL parsing.
fn percent_encode_userinfo(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[async_trait]
impl RelationalStore for SupabaseRelationalStore {
    type Error = sqlx::Error;

    async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, Self::Error> {
        self.inner.execute(sql, params).await
    }

    async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, Self::Error> {
        self.inner.query(sql, params).await
    }

    fn backend(&self) -> RelationalBackend {
        RelationalBackend::Supabase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_is_direct_connection_with_tls_required() {
        let url = supabase_connection_url("abcdefghijklmnop", "simplepass");
        assert_eq!(
            url,
            "postgresql://postgres:simplepass@db.abcdefghijklmnop.supabase.co:5432/postgres?sslmode=require"
        );
    }

    #[test]
    fn password_with_reserved_chars_is_percent_encoded() {
        let url = supabase_connection_url("proj", "p@ss:wo/rd word");
        assert_eq!(
            url,
            "postgresql://postgres:p%40ss%3Awo%2Frd%20word@db.proj.supabase.co:5432/postgres?sslmode=require"
        );
    }

    #[test]
    fn unreserved_chars_pass_through_unescaped() {
        assert_eq!(percent_encode_userinfo("Aa0-._~"), "Aa0-._~");
    }
}
