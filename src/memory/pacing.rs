//! Cooperative pacing for background work on the current thread.
//!
//! A thread that installs a pacer gives up the CPU after every `chunk_rows`
//! rows its loops report through [`tick`]: it sleeps long enough to hold its
//! share to `cpu_pct`, or yields at 100. On a thread without one, [`tick`] and
//! [`pause`] do nothing.

use std::cell::RefCell;
use std::time::{Duration, Instant};

/// The longest single pause.
const MAX_PAUSE: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacingConfig {
    /// Rows of work between pauses.
    pub chunk_rows: usize,
    /// Share of one core the work may take, 1 to 100.
    pub cpu_pct: u8,
}

struct Pacer {
    config: PacingConfig,
    rows: usize,
    busy_since: Instant,
    paused: Duration,
}

impl Pacer {
    fn pause(&mut self) {
        self.rows = 0;
        let pct = u32::from(self.config.cpu_pct.clamp(1, 100));
        if pct >= 100 {
            std::thread::yield_now();
        } else {
            let idle = self.busy_since.elapsed() * (100 - pct) / pct;
            let idle = idle.min(MAX_PAUSE);
            std::thread::sleep(idle);
            self.paused += idle;
        }
        self.busy_since = Instant::now();
    }
}

thread_local! {
    static PACER: RefCell<Option<Pacer>> = const { RefCell::new(None) };
}

/// Pace the current thread until the guard drops.
#[must_use = "pacing ends when the guard drops"]
pub fn install(config: PacingConfig) -> PacingGuard {
    let previous = PACER.with(|p| {
        p.borrow_mut().replace(Pacer {
            config: PacingConfig {
                chunk_rows: config.chunk_rows.max(1),
                cpu_pct: config.cpu_pct.clamp(1, 100),
            },
            rows: 0,
            busy_since: Instant::now(),
            paused: Duration::ZERO,
        })
    });
    PacingGuard { previous }
}

/// Restores the thread's previous pacer on drop.
pub struct PacingGuard {
    previous: Option<Pacer>,
}

impl Drop for PacingGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        PACER.with(|p| *p.borrow_mut() = previous);
    }
}

/// Report `rows` rows of work; pauses once a chunk's worth has run.
pub fn tick(rows: usize) {
    PACER.with(|p| {
        if let Some(pacer) = p.borrow_mut().as_mut() {
            pacer.rows += rows;
            if pacer.rows >= pacer.config.chunk_rows {
                pacer.pause();
            }
        }
    });
}

/// End a chunk of work now.
pub fn pause() {
    PACER.with(|p| {
        if let Some(pacer) = p.borrow_mut().as_mut() {
            pacer.pause();
        }
    });
}

/// Time the current thread's pacer has slept since it was installed.
pub fn paused() -> Duration {
    PACER.with(|p| {
        p.borrow()
            .as_ref()
            .map_or(Duration::ZERO, |pacer| pacer.paused)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spin(for_at_least: Duration) {
        let start = Instant::now();
        while start.elapsed() < for_at_least {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn a_paced_thread_sleeps_its_share_and_an_unpaced_one_never_does() {
        tick(1_000_000);
        assert_eq!(paused(), Duration::ZERO);
        let _guard = install(PacingConfig {
            chunk_rows: 10,
            cpu_pct: 50,
        });
        for _ in 0..5 {
            spin(Duration::from_millis(4));
            tick(10);
        }
        assert!(
            paused() >= Duration::from_millis(18),
            "half the time is spent paused: {:?}",
            paused()
        );
    }

    #[test]
    fn the_guard_restores_an_unpaced_thread() {
        {
            let _guard = install(PacingConfig {
                chunk_rows: 1,
                cpu_pct: 10,
            });
            tick(1);
        }
        assert_eq!(paused(), Duration::ZERO);
    }
}
