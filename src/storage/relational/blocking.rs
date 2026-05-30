//! Synchronous bridge to an async [`RelationalStore`].
//!
//! ## Why this exists
//!
//! The intent-log projection machinery is synchronous:
//! [`crate::memory::slow_store::SqliteProjection::apply`] runs on a tokio
//! worker thread because `MultiUserMemoryManager::journal_and_apply` is
//! called — un-awaited — from inside async handlers (`crud`, `remember`).
//! That makes the obvious `Handle::block_on` unusable: blocking the
//! current thread from within its own runtime panics with "Cannot start a
//! runtime from within a runtime".
//!
//! [`BlockingRelationalStore`] is the bridge that lets the synchronous
//! projection write the `memories` table through the async
//! [`RelationalStore`] trait (and therefore through any backend — SQLite,
//! Postgres, Supabase) without async-ifying the entire journal apply path.
//! It wraps an `Arc<dyn RelationalStore<Error = BoxError>>` and exposes
//! *synchronous* `execute_blocking` / `query_blocking`. Each call hands the
//! owned SQL + parameters to a dedicated, process-wide bridge runtime via
//! `spawn`, then blocks the calling thread on a `std::sync::mpsc` channel
//! until the result returns.
//!
//! Because the future runs on a *separate* runtime, the calling worker
//! only ever blocks on a plain OS channel — the same latency profile as
//! today's synchronous rusqlite write, with no "block within a runtime"
//! panic and no deadlock (the bridge runtime always makes progress
//! independently of the caller's runtime).
//!
//! This module is intentionally additive: it is the foundation for the
//! `memories`-projection cutover, but nothing wires it into the projection
//! yet — that lands in a follow-up so the bridge can be reviewed and tested
//! before it touches the safety-critical apply path.

use std::sync::{Arc, LazyLock};

use crate::storage::relational::{BoxError, Param, RelationalStore, Row};

/// Process-wide runtime that actually drives the async `RelationalStore`
/// futures handed to the bridge.
///
/// It is deliberately *separate* from the server's main runtime: a
/// synchronous caller already executing inside the main runtime can block
/// on a result produced here without tripping tokio's "cannot block the
/// current thread from within a runtime" guard. One worker thread is
/// enough — the slow store is single-writer and each call blocks the
/// caller until it completes, so there is no concurrency to exploit.
static BRIDGE_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("veld-relational-bridge")
        .enable_all()
        .build()
        .expect("build veld relational bridge runtime")
});

/// Owned mirror of [`Param`] so a borrowed parameter slice can be moved
/// into the `'static` future the bridge runtime drives, then reconstructed
/// back into borrowed `Param`s inside the spawned task.
enum OwnedParam {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Text(String),
    Bytes(Vec<u8>),
    Json(serde_json::Value),
}

impl OwnedParam {
    fn from_param(p: &Param<'_>) -> Self {
        match p {
            Param::Null => OwnedParam::Null,
            Param::Bool(b) => OwnedParam::Bool(*b),
            Param::I64(i) => OwnedParam::I64(*i),
            Param::F64(f) => OwnedParam::F64(*f),
            Param::Text(s) => OwnedParam::Text((*s).to_string()),
            Param::Bytes(b) => OwnedParam::Bytes((*b).to_vec()),
            Param::Json(j) => OwnedParam::Json((*j).clone()),
        }
    }

    fn as_param(&self) -> Param<'_> {
        match self {
            OwnedParam::Null => Param::Null,
            OwnedParam::Bool(b) => Param::Bool(*b),
            OwnedParam::I64(i) => Param::I64(*i),
            OwnedParam::F64(f) => Param::F64(*f),
            OwnedParam::Text(s) => Param::Text(s),
            OwnedParam::Bytes(b) => Param::Bytes(b),
            OwnedParam::Json(j) => Param::Json(j),
        }
    }
}

/// Synchronous wrapper over an async [`RelationalStore`]. Clone-cheap (just
/// an `Arc` bump); every clone shares the same backend and bridge runtime.
#[derive(Clone)]
pub struct BlockingRelationalStore {
    inner: Arc<dyn RelationalStore<Error = BoxError>>,
}

impl BlockingRelationalStore {
    /// Wrap an async `RelationalStore` for synchronous use.
    pub fn new(inner: Arc<dyn RelationalStore<Error = BoxError>>) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped async store, for callers that are already async
    /// and don't need the bridge.
    pub fn inner(&self) -> &Arc<dyn RelationalStore<Error = BoxError>> {
        &self.inner
    }

    /// Synchronous [`RelationalStore::execute`]. Safe to call from inside
    /// the main tokio runtime: the future runs on the dedicated bridge
    /// runtime and this thread blocks only on a plain channel.
    pub fn execute_blocking(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, BoxError> {
        let inner = self.inner.clone();
        let sql = sql.to_string();
        let owned: Vec<OwnedParam> = params.iter().map(OwnedParam::from_param).collect();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        BRIDGE_RUNTIME.spawn(async move {
            let bound: Vec<Param<'_>> = owned.iter().map(OwnedParam::as_param).collect();
            let result = inner.execute(&sql, &bound).await;
            // Receiver gone only if the caller was cancelled mid-call; the
            // dropped send is harmless.
            let _ = tx.send(result);
        });
        recv_bridge(rx)
    }

    /// Synchronous [`RelationalStore::query`]. Same bridging contract as
    /// [`Self::execute_blocking`].
    pub fn query_blocking(&self, sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
        let inner = self.inner.clone();
        let sql = sql.to_string();
        let owned: Vec<OwnedParam> = params.iter().map(OwnedParam::from_param).collect();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        BRIDGE_RUNTIME.spawn(async move {
            let bound: Vec<Param<'_>> = owned.iter().map(OwnedParam::as_param).collect();
            let result = inner.query(&sql, &bound).await;
            let _ = tx.send(result);
        });
        recv_bridge(rx)
    }

    /// Backend identity of the wrapped store.
    pub fn backend(&self) -> crate::storage::relational::RelationalBackend {
        self.inner.backend()
    }
}

/// Block on the bridge channel. A `RecvError` means the spawned task was
/// dropped before sending — only reachable if the bridge runtime is torn
/// down mid-call — which we surface as a [`BoxError`] rather than a panic.
fn recv_bridge<T>(rx: std::sync::mpsc::Receiver<Result<T, BoxError>>) -> Result<T, BoxError> {
    match rx.recv() {
        Ok(result) => result,
        Err(e) => Err(BoxError::new(e)),
    }
}

/// Drive an arbitrary future to completion on the dedicated bridge runtime
/// and block the calling thread until it finishes.
///
/// This is the generic form of [`BlockingRelationalStore::execute_blocking`]:
/// use it to bridge async helpers that wrap the trait (e.g. the slow-store
/// `RelationalSlowStoreAdapter`'s typed `*_memory` methods) from a
/// synchronous caller. Safe from inside the main runtime — the future runs
/// on a *separate* runtime and the caller only blocks on a channel — and
/// from a plain thread with no ambient runtime.
///
/// `fut` is moved onto the bridge runtime, so it must be `Send + 'static`;
/// callers pass owned data into an `async move` block. The output `T` is
/// usually itself a `Result`. Panics only if the spawned task panics before
/// producing a value (a genuine bug), surfaced as a dropped channel.
pub fn bridge_block_on<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    BRIDGE_RUNTIME.spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv()
        .expect("relational bridge runtime dropped before completing a bridged future")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::relational::{ColumnMeta, OwnedColumn, RelationalBackend};
    use async_trait::async_trait;
    use std::borrow::Cow;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Runtime-agnostic mock backend. Has a real `.await` point so the test
    /// proves the bridge runtime actually *drives* the future (not the
    /// caller's thread), and returns deterministic data so the round-trip
    /// across the channel can be asserted. SQL containing `FAIL` yields an
    /// error so error propagation is covered without a real driver.
    struct MockStore {
        executes: AtomicU64,
    }

    #[async_trait]
    impl RelationalStore for MockStore {
        type Error = BoxError;

        async fn execute(&self, sql: &str, _params: &[Param<'_>]) -> Result<u64, BoxError> {
            // Force a yield so the future cannot complete synchronously —
            // it must be polled to completion by the bridge runtime.
            tokio::task::yield_now().await;
            if sql.contains("FAIL") {
                return Err(BoxError::new(std::io::Error::other("mock failure")));
            }
            Ok(self.executes.fetch_add(1, Ordering::SeqCst) + 1)
        }

        async fn query(&self, _sql: &str, params: &[Param<'_>]) -> Result<Vec<Row>, BoxError> {
            tokio::task::yield_now().await;
            // Echo the number of bound params back as a single i64 row so the
            // test confirms params survived the owned round-trip.
            let columns = vec![ColumnMeta {
                name: "param_count".to_string(),
                sql_type: "BIGINT".to_string(),
            }];
            let values = vec![OwnedColumn::I64(params.len() as i64)];
            Ok(vec![Row::from_owned(columns, values)])
        }

        fn backend(&self) -> RelationalBackend {
            RelationalBackend::Custom(Cow::Borrowed("mock"))
        }
    }

    fn mock_store() -> BlockingRelationalStore {
        let inner: Arc<dyn RelationalStore<Error = BoxError>> =
            Arc::new(MockStore { executes: AtomicU64::new(0) });
        BlockingRelationalStore::new(inner)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_calls_from_runtime_worker_do_not_panic_or_deadlock() {
        let store = mock_store();
        // Synchronous calls made directly on a runtime worker thread — the
        // exact shape of SqliteProjection::apply. A naive Handle::block_on
        // here would panic; the bridge must not.
        let n1 = store.execute_blocking("INSERT 1", &[Param::I64(1)]).expect("execute 1");
        let n2 = store.execute_blocking("INSERT 2", &[Param::Text("x")]).expect("execute 2");
        assert_eq!((n1, n2), (1, 2), "counter advances across bridged calls");

        // Params survive the owned round-trip into the spawned future.
        let rows = store
            .query_blocking("SELECT ?, ?, ?", &[Param::I64(1), Param::Null, Param::Bool(true)])
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<i64>(0).expect("param_count"), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backend_error_propagates_as_boxerror() {
        let store = mock_store();
        let result = store.execute_blocking("FAIL please", &[]);
        assert!(result.is_err(), "backend error must surface as Err, not a panic");
        assert!(format!("{}", result.unwrap_err()).contains("mock failure"));
    }

    #[test]
    fn blocking_calls_work_outside_any_runtime() {
        // Called from a plain test thread with no ambient runtime at all —
        // the bridge runtime must still drive the work to completion.
        let store = mock_store();
        let n = store.execute_blocking("INSERT", &[]).expect("execute without ambient runtime");
        assert_eq!(n, 1);
    }
}
