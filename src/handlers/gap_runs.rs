//! Gap analysis runs by user, parameters and scope: one in flight per key on
//! the gap worker, the last result kept and served fresh, stale or not yet.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::sync::watch;

use crate::errors::AppError;
use crate::memory::gap_topology::{worker, GapRun, GapScope, GapWorkerConfig};

/// How long a result is served as fresh.
pub(super) const GAP_FRESH_SECS: i64 = 300;
/// Results kept at most, the oldest dropped first.
const GAP_RESULTS_KEPT: usize = 64;

/// What a run depends on; offset and limit only page its result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct GapKey {
    pub user_id: String,
    pub min_strength: u32,
    pub max_gaps: usize,
    pub min_embedding_similarity: Option<u32>,
    pub star_min_spokes: Option<usize>,
    pub orbit_min_cluster_size: Option<usize>,
    pub scope: GapScope,
}

#[derive(Clone)]
pub(super) struct Detected {
    pub run: Arc<GapRun>,
    pub at: chrono::DateTime<chrono::Utc>,
}

impl Detected {
    fn is_fresh(&self) -> bool {
        chrono::Utc::now() - self.at < chrono::Duration::seconds(GAP_FRESH_SECS)
    }
}

/// A run's end: its result, or why it failed.
type Outcome = Option<Result<Detected, String>>;

#[derive(Default)]
struct GapEntry {
    last: Option<Detected>,
    running: Option<watch::Receiver<Outcome>>,
}

/// Results by key, and the run under way for each.
static GAP_RUNS: LazyLock<Mutex<HashMap<GapKey, GapEntry>>> = LazyLock::new(Default::default);

pub(super) enum Served {
    /// Computed within the freshness window.
    Fresh(Detected),
    /// An older result, while a new run goes on or after it failed.
    Stale(Detected),
    /// No result yet; the run goes on.
    Running,
}

/// The run for `key`: a fresh result at once; otherwise the end of a run,
/// started here unless one is under way, if it comes within `wait`; otherwise
/// the last result while the run goes on. One run per key at a time, and a run
/// finishes and is kept whether or not anyone is still waiting for it.
pub(super) async fn detection(
    key: GapKey,
    wait: Duration,
    worker: GapWorkerConfig,
    job: impl FnOnce() -> anyhow::Result<GapRun> + Send + 'static,
) -> Result<Served, AppError> {
    let (mut run, last) = {
        let mut runs = GAP_RUNS.lock();
        let entry = runs.entry(key.clone()).or_default();
        if let Some(last) = entry.last.as_ref().filter(|d| d.is_fresh()) {
            return Ok(Served::Fresh(last.clone()));
        }
        // A run whose job died without an outcome has dropped its sender; start another.
        let run = match entry.running.as_ref().filter(|r| r.has_changed().is_ok()) {
            Some(run) => run.clone(),
            None => {
                let run = start(key, &worker, job);
                entry.running = Some(run.clone());
                run
            }
        };
        (run, entry.last.clone())
    };
    let ended = match tokio::time::timeout(wait, run.wait_for(Option::is_some)).await {
        Ok(Ok(outcome)) => Option::clone(&outcome),
        Ok(Err(_)) => Some(Err("the gap analysis run ended without a result".into())),
        Err(_) => None,
    };
    Ok(match ended {
        Some(Ok(detected)) => Served::Fresh(detected),
        Some(Err(error)) => match last {
            Some(last) => Served::Stale(last),
            None => return Err(AppError::Internal(anyhow::anyhow!(error))),
        },
        None => last.map_or(Served::Running, Served::Stale),
    })
}

/// Whether a run for `key` is under way.
pub(super) fn is_running(key: &GapKey) -> bool {
    GAP_RUNS
        .lock()
        .get(key)
        .and_then(|e| e.running.as_ref())
        .is_some_and(|r| r.has_changed().is_ok())
}

/// Queues `job` on the gap worker and keeps its result under `key`.
fn start(
    key: GapKey,
    worker: &GapWorkerConfig,
    job: impl FnOnce() -> anyhow::Result<GapRun> + Send + 'static,
) -> watch::Receiver<Outcome> {
    let (done, run) = watch::channel(None);
    let submitted = worker::submit(worker, move || {
        let outcome = match job() {
            Ok(result) => Ok(Detected {
                run: Arc::new(result),
                at: chrono::Utc::now(),
            }),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = &outcome {
            tracing::warn!(user_id = %key.user_id, error = %e, "gap analysis failed");
        }
        {
            let mut runs = GAP_RUNS.lock();
            let entry = runs.entry(key).or_default();
            entry.running = None;
            if let Ok(detected) = &outcome {
                entry.last = Some(detected.clone());
            }
            prune(&mut runs);
        }
        done.send_replace(Some(outcome));
    });
    if let Err(e) = submitted {
        tracing::error!(error = %e, "gap analysis not queued");
    }
    run
}

/// Drops results more than an hour old, then the oldest beyond the cap; a key
/// with a run under way is kept.
fn prune(runs: &mut HashMap<GapKey, GapEntry>) {
    let now = chrono::Utc::now();
    runs.retain(|_, e| {
        e.running.is_some()
            || e.last
                .as_ref()
                .is_some_and(|d| now - d.at < chrono::Duration::hours(1))
    });
    let excess = runs.len().saturating_sub(GAP_RESULTS_KEPT);
    if excess > 0 {
        let mut idle: Vec<(GapKey, chrono::DateTime<chrono::Utc>)> = runs
            .iter()
            .filter(|(_, e)| e.running.is_none())
            .filter_map(|(k, e)| e.last.as_ref().map(|d| (k.clone(), d.at)))
            .collect();
        idle.sort_by_key(|(_, at)| *at);
        for (k, _) in idle.into_iter().take(excess) {
            runs.remove(&k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::gap_topology::{GapDetectionResult, MirrorReport};
    use crate::memory::slow_store::SyncMode;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn key(user: &str) -> GapKey {
        GapKey {
            user_id: user.to_string(),
            min_strength: 0.2f32.to_bits(),
            max_gaps: 50,
            min_embedding_similarity: None,
            star_min_spokes: None,
            orbit_min_cluster_size: None,
            scope: GapScope::default(),
        }
    }

    fn result(duration_ms: u64) -> GapRun {
        GapRun {
            result: GapDetectionResult {
                gaps: Vec::new(),
                type_counts: HashMap::new(),
                duration_ms,
            },
            scope: GapScope::default().report(0, 0),
            mirror: MirrorReport {
                mode: SyncMode::Unchanged,
                best_effort: false,
                changed_during_copy: None,
                rows_written: 0,
                duration_ms: 0,
            },
            detect_ms: duration_ms,
            duration_ms,
        }
    }

    fn counted(
        runs: &Arc<AtomicUsize>,
        took: Duration,
        duration_ms: u64,
    ) -> impl FnOnce() -> anyhow::Result<GapRun> + Send + 'static {
        let runs = Arc::clone(runs);
        move || {
            runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(took);
            Ok(result(duration_ms))
        }
    }

    fn worker() -> GapWorkerConfig {
        GapWorkerConfig::default()
    }

    #[tokio::test]
    async fn concurrent_requests_share_one_run_and_a_fresh_result_needs_none() {
        let runs = Arc::new(AtomicUsize::new(0));
        let wait = Duration::from_secs(5);
        let took = Duration::from_millis(200);
        let (a, b) = tokio::join!(
            detection(key("gap-shared"), wait, worker(), counted(&runs, took, 7)),
            detection(key("gap-shared"), wait, worker(), counted(&runs, took, 7)),
        );
        let (Served::Fresh(a), Served::Fresh(b)) = (a.unwrap(), b.unwrap()) else {
            panic!("both requests get the run's result");
        };
        assert_eq!(a.at, b.at);
        assert_eq!(runs.load(Ordering::SeqCst), 1);

        let again = detection(key("gap-shared"), wait, worker(), counted(&runs, took, 7))
            .await
            .unwrap();
        assert!(matches!(again, Served::Fresh(d) if d.at == a.at));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_run_outlasting_the_wait_answers_running_and_its_result_is_kept() {
        let runs = Arc::new(AtomicUsize::new(0));
        let short = Duration::from_millis(50);
        let first = detection(
            key("gap-slow"),
            short,
            worker(),
            counted(&runs, Duration::from_millis(300), 1),
        )
        .await
        .unwrap();
        assert!(matches!(first, Served::Running));
        assert!(is_running(&key("gap-slow")));

        let later = detection(
            key("gap-slow"),
            Duration::from_secs(5),
            worker(),
            counted(&runs, Duration::ZERO, 2),
        )
        .await
        .unwrap();
        assert!(matches!(later, Served::Fresh(d) if d.run.result.duration_ms == 1));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(!is_running(&key("gap-slow")));
    }

    #[tokio::test]
    async fn a_failed_first_run_is_an_error_and_a_failed_refresh_serves_the_last_result() {
        let k = key("gap-fails");
        let wait = Duration::from_secs(5);
        assert!(
            detection(k.clone(), wait, worker(), || anyhow::bail!("no graph"))
                .await
                .is_err()
        );

        GAP_RUNS.lock().entry(k.clone()).or_default().last = Some(Detected {
            run: Arc::new(result(3)),
            at: chrono::Utc::now() - chrono::Duration::seconds(GAP_FRESH_SECS + 1),
        });
        let served = detection(k, wait, worker(), || anyhow::bail!("no graph"))
            .await
            .unwrap();
        assert!(matches!(served, Served::Stale(d) if d.run.result.duration_ms == 3));
    }

    #[tokio::test]
    async fn runs_with_different_scopes_are_kept_apart() {
        let runs = Arc::new(AtomicUsize::new(0));
        let wait = Duration::from_secs(5);
        let mut scoped = key("gap-scoped");
        scoped.scope = GapScope::new(Some("file".into()), None, vec![], Some(100));
        detection(
            key("gap-scoped"),
            wait,
            worker(),
            counted(&runs, Duration::ZERO, 1),
        )
        .await
        .unwrap();
        let served = detection(scoped, wait, worker(), counted(&runs, Duration::ZERO, 2))
            .await
            .unwrap();
        assert!(matches!(served, Served::Fresh(d) if d.run.result.duration_ms == 2));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }
}
