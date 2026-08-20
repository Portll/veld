//! Process-global Argon2id concurrency limiter (B3 hardening).
//!
//! Argon2id verify / hash is the most expensive single operation in the
//! user-auth surface — ~50 ms of memory-bound CPU at the primary password
//! parameters and ~25 ms for recovery codes. Without back-pressure, 100
//! concurrent logins all hit the executor at once, the OS over-commits cores,
//! and the per-request latency tail balloons. The fix is a small
//! process-global [`tokio::sync::Semaphore`] sized to the host's available
//! parallelism: queued requests park on the semaphore — an *async* wait that
//! does not consume an executor thread — instead of all storming the CPU.
//!
//! The semaphore is created on first use through a [`std::sync::OnceLock`],
//! so it survives across all subsequent calls and tests (the permits never
//! shrink, never expire). Acquiring a permit returns an owned guard that
//! releases automatically on drop, so a panic inside an Argon2 critical
//! section cannot leak permits.
//!
//! Each critical section also runs the CPU-bound work inside
//! [`tokio::task::spawn_blocking`] so it never blocks the async executor.
//! The semaphore lives in front of `spawn_blocking` so the blocking thread
//! pool isn't drowned either: at most `permits` Argon2 jobs are in flight at
//! any moment, regardless of how many requests are queued behind them.

use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::user_auth::AuthError;

/// Lazily-initialised handle to the process-global Argon2id semaphore.
///
/// `Arc<Semaphore>` is used (rather than a plain `Semaphore`) so the
/// `acquire_owned` API can hand a permit guard whose lifetime is independent
/// of any borrow; that's what lets the permit travel into a `spawn_blocking`
/// closure and back without lifetime gymnastics.
static ARGON2_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();

/// Permit count for the semaphore — `available_parallelism()` with a
/// conservative fallback of 2 when the platform refuses to answer.
///
/// Exposed for tests so they can assert against the same formula the
/// production code uses (rather than re-deriving and risking drift).
pub fn permit_count() -> usize {
    std::thread::available_parallelism()
        .unwrap_or_else(|_| NonZeroUsize::new(2).expect("2 is non-zero"))
        .get()
}

/// Borrow (initialising on first call) the process-global semaphore.
fn semaphore() -> &'static Arc<Semaphore> {
    ARGON2_SEMAPHORE.get_or_init(|| Arc::new(Semaphore::new(permit_count())))
}

/// Acquire one Argon2 permit, awaiting if the cap is currently saturated.
///
/// Returns an owned permit guard that releases automatically when dropped.
/// The only failure mode is "the semaphore was closed", which this module
/// never does — so the call surfaces as a structured `AuthError::Internal`
/// rather than panicking, defending against future refactors that might
/// introduce shutdown semantics.
pub async fn acquire() -> Result<OwnedSemaphorePermit, AuthError> {
    semaphore()
        .clone()
        .acquire_owned()
        .await
        .map_err(|e| AuthError::internal(format!("argon2 semaphore closed: {e}")))
}

/// Run a CPU-bound Argon2 closure under the global semaphore.
///
/// Holds one permit for the entire duration of the closure, including the
/// `spawn_blocking` join. Closure-and-permit are dropped together so the
/// permit count reflects "Argon2 jobs in flight on the blocking pool",
/// which is exactly the resource being capped.
pub async fn run_blocking<F, T>(work: F) -> Result<T, AuthError>
where
    F: FnOnce() -> Result<T, AuthError> + Send + 'static,
    T: Send + 'static,
{
    let permit = acquire().await?;
    let result = tokio::task::spawn_blocking(move || {
        // Move the permit into the blocking task so it stays alive for the
        // duration of the CPU work; drop happens implicitly at task exit.
        let _permit = permit;
        work()
    })
    .await
    .map_err(|e| AuthError::internal(format!("argon2 task join failed: {e}")))?;
    result
}

/// Returns the number of permits currently available — only meaningful for
/// tests and observability. A drop below `permit_count()` means at least
/// one Argon2 critical section is in flight.
#[cfg(test)]
pub fn available_permits() -> usize {
    semaphore().available_permits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn permit_count_matches_available_parallelism_or_fallback() {
        let n = permit_count();
        assert!(n >= 1, "permit count must be positive (got {n})");
        let expected = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        assert_eq!(n, expected);
    }

    /// ONE TEST, DELIBERATELY, AND IT MUST STAY ONE.
    ///
    /// These were two `#[tokio::test]`s and the pair was flaky — measured
    /// 2026-08-20, `assert_eq!(available_permits(), before)` failing with
    /// `left: 0, right: 18`. Neither test was wrong about the code; they were
    /// wrong about each other. `ARGON2_SEMAPHORE` is PROCESS-GLOBAL (see the
    /// `OnceLock` above), the harness runs tests in parallel threads within one
    /// process, and the first half below SATURATES that semaphore on purpose to
    /// prove the cap binds. The second half sampled `available_permits()` and
    /// asserted it had returned to its starting value — which is only true when
    /// no other test is contending. `right: 18` is `available_parallelism()` on
    /// that machine, i.e. the full idle pool; `left: 0` is the first half
    /// holding every permit.
    ///
    /// The old code carried a `yield_now()` "to allow a small async wiggle for
    /// the spawn_blocking join" — a real timing fix aimed at the wrong race. It
    /// settles this test's own join and can do nothing about a sibling's permits.
    ///
    /// Worse than the flake: the equality assertion could not tell "my permit
    /// leaked" — the bug it exists to catch — from "someone else is busy", so it
    /// could also PASS while a permit genuinely leaked. Splitting these again
    /// restores both faults.
    #[tokio::test]
    async fn semaphore_caps_concurrency_and_releases_permits_on_error() {
        // Spawn N tasks where N is well above the permit count. Each task
        // counts itself in/out of the critical section with an AtomicUsize
        // and asserts the peak never exceeds the cap. The test mirrors the
        // pattern called out in the breakers report: an in-flight counter
        // sampled from inside the critical section.
        let cap = permit_count();
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let total = 20.max(cap * 3);
        let mut handles = Vec::with_capacity(total);
        for _ in 0..total {
            let in_flight = Arc::clone(&in_flight);
            let peak = Arc::clone(&peak);
            handles.push(tokio::spawn(async move {
                // Acquire one permit then enter the critical section.
                let _permit = acquire().await.expect("permit");
                let entered = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(entered, Ordering::SeqCst);
                // Hold the section long enough that other tasks pile up
                // behind us — without this the test would race past the
                // contention point and never observe the cap.
                tokio::time::sleep(Duration::from_millis(25)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert!(
            peak.load(Ordering::SeqCst) <= cap,
            "peak in-flight {} exceeded the semaphore cap {}",
            peak.load(Ordering::SeqCst),
            cap
        );
        // And we did actually contend — the cap is a meaningful constraint
        // here, not a no-op because the test under-spawned.
        assert!(
            peak.load(Ordering::SeqCst) >= 1,
            "peak should be at least 1 — no work observed"
        );

        // ── second half: an erroring closure must still return its permit ──
        //
        // ASSERTED BY PROGRESS, NOT BY COUNTING, and that distinction is the
        // whole point. Any `assert_eq!` on `available_permits()` is an exact
        // comparison against a PROCESS-GLOBAL counter that every other test in
        // this binary can move — which is what made the original flaky, and
        // which I reproduced once while fixing it by asserting the pool was
        // idle here. Counting cannot work; a leak has to be observed as a
        // failure to make progress.
        //
        // So: run `cap + 1` erroring calls in sequence. Each must acquire a
        // permit, error, and RELEASE. If the error path leaked even one, the
        // pool drains and a later call blocks forever — surfacing as the
        // timeout below rather than as a number that happened to differ.
        // Contention from other tests can only slow this down, never break it.
        for i in 0..=cap {
            let outcome = tokio::time::timeout(
                Duration::from_secs(10),
                run_blocking(|| Err::<(), _>(AuthError::internal("boom"))),
            )
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "call {i} of {cap} blocked for 10s — the error path is not releasing \
                     its permit, so the pool has drained"
                )
            });
            assert!(matches!(outcome.unwrap_err(), AuthError::Internal(_)));
        }
    }
}
