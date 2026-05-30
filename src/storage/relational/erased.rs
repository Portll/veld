//! Type-erase any [`RelationalStore`] to `Error = BoxError`.
//!
//! The dataset stores ([`crate::datasets::RelationalDatasetStore`],
//! [`crate::datasets::RelationalLinkStore`]), the slow-store adapter, and
//! the W6 query planner all consume
//! `Arc<dyn RelationalStore<Error = BoxError>>`. The concrete backends
//! (`SqliteRelationalStore`, `PostgresRelationalStore`,
//! `SupabaseRelationalStore`) all use `Error = sqlx::Error`.
//!
//! [`ErasedRelationalStore`] bridges the two: wrap a concrete backend and
//! it presents the same query surface with the error mapped to [`BoxError`]
//! at the boundary. This is the production realisation of the `Adapter<S>`
//! pattern sketched in the [`super`] module docs, replacing the
//! per-backend `BoxError`-shim newtypes that tests previously hand-rolled.

use async_trait::async_trait;

use super::{BoxError, Param, RelationalBackend, RelationalStore, Row};

/// Wraps any [`RelationalStore`] and erases its associated error to
/// [`BoxError`], so it can be held behind
/// `Arc<dyn RelationalStore<Error = BoxError>>`.
pub struct ErasedRelationalStore<S> {
    inner: S,
}

impl<S> ErasedRelationalStore<S> {
    /// Wrap `inner`, erasing its error type at the trait boundary.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped backend.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

#[async_trait]
impl<S> RelationalStore for ErasedRelationalStore<S>
where
    S: RelationalStore,
{
    // `S::Error` already satisfies `std::error::Error + Send + Sync +
    // 'static` (the `RelationalStore::Error` bound), so `BoxError::new`
    // accepts it directly.
    type Error = BoxError;

    async fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, BoxError> {
        self.inner.execute(sql, params).await.map_err(BoxError::new)
    }

    async fn query(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
        self.inner.query(sql, params).await.map_err(BoxError::new)
    }

    fn backend(&self) -> RelationalBackend {
        self.inner.backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::relational::SqliteRelationalStore;
    use std::sync::Arc;

    #[tokio::test]
    async fn erases_sqlite_error_to_boxerror_and_round_trips() {
        let sqlite = SqliteRelationalStore::in_memory()
            .await
            .expect("open in-memory sqlite");
        // The whole point: a concrete `Error = sqlx::Error` backend held
        // behind the `Error = BoxError` trait object.
        let store: Arc<dyn RelationalStore<Error = BoxError>> =
            Arc::new(ErasedRelationalStore::new(sqlite));

        store
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", &[])
            .await
            .expect("create");
        let n = store
            .execute(
                "INSERT INTO t (id, v) VALUES (?, ?)",
                &[Param::I64(1), Param::Text("hi")],
            )
            .await
            .expect("insert");
        assert_eq!(n, 1);

        let rows = store
            .query("SELECT v FROM t WHERE id = ?", &[Param::I64(1)])
            .await
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<String>(0).expect("v"), "hi");

        // A failing statement surfaces as Err (BoxError), not a panic.
        assert!(store.execute("NOT VALID SQL", &[]).await.is_err());
    }
}
