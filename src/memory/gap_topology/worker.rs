//! The thread gap analyses run on.
//!
//! One thread per process, started on first use: at lowered priority, paced to
//! a share of one core, and taking jobs one at a time in submission order. An
//! analysis never runs on the async runtime's workers or its blocking pool.

use anyhow::{anyhow, Result};
use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{self, Sender};
use std::sync::OnceLock;

use crate::memory::pacing::{self, PacingConfig};

/// How the worker thread runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GapWorkerConfig {
    /// Nice value, 0 to 19. On macOS the thread's scheduling priority drops by
    /// this much, down to the policy's floor.
    pub nice: i32,
    /// Rows of work between pauses.
    pub chunk_rows: usize,
    /// Share of one core the worker may take, 1 to 100.
    pub cpu_pct: u8,
}

impl Default for GapWorkerConfig {
    fn default() -> Self {
        Self {
            nice: 10,
            chunk_rows: 4096,
            cpu_pct: 50,
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

static WORKER: OnceLock<Sender<Job>> = OnceLock::new();

/// Queue `job` on the worker, starting it with `config` if it is not running.
/// A later call's config does not change a running worker.
pub fn submit(config: &GapWorkerConfig, job: impl FnOnce() + Send + 'static) -> Result<()> {
    WORKER
        .get_or_init(|| start(*config))
        .send(Box::new(job))
        .map_err(|_| anyhow!("the gap analysis worker is not running"))
}

/// Run `job` on the worker and wait for its result without blocking the runtime.
pub async fn run<T: Send + 'static>(
    config: &GapWorkerConfig,
    job: impl FnOnce() -> T + Send + 'static,
) -> Result<T> {
    let (done, result) = tokio::sync::oneshot::channel();
    submit(config, move || {
        let _ = done.send(job());
    })?;
    result
        .await
        .map_err(|_| anyhow!("the gap analysis job ended without a result"))
}

fn start(config: GapWorkerConfig) -> Sender<Job> {
    let (jobs, queue) = mpsc::channel::<Job>();
    let spawned = std::thread::Builder::new()
        .name("veld-gap-worker".into())
        .spawn(move || {
            lower_priority(config.nice);
            let _pacing = pacing::install(PacingConfig {
                chunk_rows: config.chunk_rows,
                cpu_pct: config.cpu_pct,
            });
            tracing::info!(
                nice = config.nice,
                chunk_rows = config.chunk_rows,
                cpu_pct = config.cpu_pct,
                "gap analysis worker started"
            );
            for job in queue {
                if std::panic::catch_unwind(AssertUnwindSafe(job)).is_err() {
                    tracing::error!("a gap analysis job panicked");
                }
            }
        });
    if let Err(e) = spawned {
        tracing::error!(error = %e, "gap analysis worker did not start");
    }
    jobs
}

#[cfg(target_os = "linux")]
fn lower_priority(nice: i32) {
    let nice = nice.clamp(0, 19);
    if nice == 0 {
        return;
    }
    // Linux keeps a nice value per thread.
    let tid = unsafe { libc::gettid() };
    if unsafe { libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, nice) } != 0 {
        tracing::warn!(nice, error = %std::io::Error::last_os_error(), "gap worker priority unchanged");
    }
}

#[cfg(target_os = "macos")]
fn lower_priority(nice: i32) {
    let nice = nice.clamp(0, 19);
    if nice == 0 {
        return;
    }
    // A lower QoS class would also throttle the thread's disk I/O, and request
    // writers wait on the mirror transactions it runs; only the scheduling
    // priority drops.
    let thread = unsafe { libc::pthread_self() };
    let mut policy = 0;
    let mut param: libc::sched_param = unsafe { std::mem::zeroed() };
    if unsafe { libc::pthread_getschedparam(thread, &mut policy, &mut param) } != 0 {
        tracing::warn!(nice, "gap worker priority unchanged");
        return;
    }
    param.sched_priority =
        (param.sched_priority - nice).max(unsafe { libc::sched_get_priority_min(policy) });
    if unsafe { libc::pthread_setschedparam(thread, policy, &param) } != 0 {
        tracing::warn!(nice, "gap worker priority unchanged");
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn lower_priority(nice: i32) {
    if nice > 0 {
        tracing::debug!(nice, "gap worker priority is not lowered on this platform");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn jobs_run_off_the_runtime_on_one_named_thread() {
        let config = GapWorkerConfig::default();
        let first = run(&config, || std::thread::current().name().map(String::from))
            .await
            .unwrap();
        let second = run(&config, || std::thread::current().id()).await.unwrap();
        let third = run(&config, || std::thread::current().id()).await.unwrap();
        assert_eq!(first.as_deref(), Some("veld-gap-worker"));
        assert_eq!(second, third);
    }

    #[tokio::test]
    async fn a_panicking_job_leaves_the_worker_running() {
        let config = GapWorkerConfig::default();
        assert!(run(&config, || -> u32 { panic!("job fails") })
            .await
            .is_err());
        assert_eq!(run(&config, || 7).await.unwrap(), 7);
    }
}
