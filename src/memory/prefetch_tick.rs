//! Anticipatory prefetch tick state + planning helpers (R1 — Physarum-anticipation wiring).
//!
//! This module owns the *per-user side-state* and the *pure* planning logic for
//! the anticipatory-prefetch maintenance tick. The orchestrating method that
//! actually reads tiers, runs queries, and warms the session tier lives on
//! [`crate::memory::MemorySystem`] (see `run_prefetch_tick`).
//!
//! # Design constraints honored here
//!
//! - **B4 (perf)**: rhythm is a PER-USER AGGREGATE — exactly one FFT is run over
//!   the user's combined access stream per recompute. No per-memory FFT.
//! - **I9 (cache)**: [`LearnedScales`] are cached per-user with `computed_at` and
//!   recomputed only when stale or when the sample count grew by ≥ a threshold.
//! - **I4 + J5 (kill-switch)**: a hit-rate EMA drives a per-user kill-switch with
//!   a hysteresis band and a periodic re-probe (trial window), so a transient
//!   precision dip never disables a user permanently.
//! - **J5 (lifecycle)**: this state lives on `MemorySystem`, which is owned
//!   per-user in the `user_memories` cache and dropped on user eviction — no
//!   global per-user map that leaks.

use chrono::{DateTime, Utc};

use crate::constants::{
    PREFETCH_HITRATE_EMA_ALPHA, PREFETCH_PRECISION_DISABLE_FLOOR,
    PREFETCH_PRECISION_REENABLE_CEILING, PREFETCH_REPROBE_INTERVAL_TICKS,
    PREFETCH_SCALES_RECOMPUTE_SAMPLE_DELTA, PREFETCH_SCALES_STALE_HOURS,
};
use crate::decay_scales::LearnedScales;

/// Per-user anticipatory-prefetch side-state.
///
/// Held inside [`crate::memory::MemorySystem`] (one per user), so it is freed
/// when the user's memory is evicted (J5 lifecycle). All fields are runtime-only;
/// they are intentionally NOT serialized — on restart the scales recompute from
/// the (persisted) access histories on the first heavy tick, and the kill-switch
/// re-probes from the enabled state.
#[derive(Debug, Clone)]
pub struct PrefetchUserState {
    /// I9: cached learned decay scales (dominant rhythm periods). `None` until
    /// the first successful FFT.
    pub scales: Option<LearnedScales>,
    /// I4: smoothed prefetch precision (fraction of warmed candidates that were
    /// genuinely due / newly warmed). Drives the kill-switch.
    pub hitrate_ema: f64,
    /// I4/J5: whether prefetch is currently enabled for this user. Disabled when
    /// precision drops below the floor; re-enabled above the ceiling (hysteresis).
    pub enabled: bool,
    /// J5: number of heavy ticks elapsed since this user's prefetch was disabled.
    /// Used to schedule a periodic re-probe trial window.
    pub ticks_since_disabled: u32,
    /// Total prefetch ticks this user has run (for diagnostics / warm-up logic).
    pub ticks_run: u64,
}

impl Default for PrefetchUserState {
    fn default() -> Self {
        Self {
            scales: None,
            // Seed the EMA at the re-enable ceiling so a brand-new user starts
            // enabled and is not killed before it has produced any signal.
            hitrate_ema: PREFETCH_PRECISION_REENABLE_CEILING,
            enabled: true,
            ticks_since_disabled: 0,
            ticks_run: 0,
        }
    }
}

impl PrefetchUserState {
    /// I9: decide whether the cached [`LearnedScales`] must be recomputed.
    ///
    /// Recompute when there are no cached scales, when the cache is older than
    /// [`PREFETCH_SCALES_STALE_HOURS`], or when the access-sample count has grown
    /// by at least [`PREFETCH_SCALES_RECOMPUTE_SAMPLE_DELTA`] since the cache was
    /// built. Otherwise the cached scales are reused — the FFT does NOT run.
    pub fn scales_need_recompute(&self, current_sample_count: usize, now: DateTime<Utc>) -> bool {
        match &self.scales {
            None => true,
            Some(cached) => {
                let age = now.signed_duration_since(cached.computed_at);
                let stale = age.num_hours() >= PREFETCH_SCALES_STALE_HOURS;
                let grew = current_sample_count
                    .saturating_sub(cached.sample_count)
                    >= PREFETCH_SCALES_RECOMPUTE_SAMPLE_DELTA;
                stale || grew
            }
        }
    }

    /// I4/J5: should this disabled user run a re-probe (trial) tick now?
    ///
    /// Only meaningful when `!self.enabled`. Returns true once the user has been
    /// disabled for at least [`PREFETCH_REPROBE_INTERVAL_TICKS`] heavy ticks, so
    /// a transient dip cannot disable prefetch forever.
    pub fn should_reprobe(&self) -> bool {
        !self.enabled && self.ticks_since_disabled >= PREFETCH_REPROBE_INTERVAL_TICKS
    }

    /// Whether the tick body should execute this cycle: either prefetch is
    /// enabled, or it is disabled but a re-probe trial window has arrived.
    pub fn should_run_this_tick(&self) -> bool {
        self.enabled || self.should_reprobe()
    }

    /// I4: fold a measured precision sample into the EMA and apply the
    /// hysteresis kill-switch.
    ///
    /// `precision` is in `[0.0, 1.0]` — the fraction of this tick's warmed
    /// candidates that were genuinely new (not already warm). Updates
    /// `hitrate_ema`, then:
    /// - if enabled and EMA < disable-floor → disable (engage kill-switch);
    /// - if disabled and EMA ≥ re-enable-ceiling → re-enable.
    ///
    /// The band between floor and ceiling is the hysteresis region; nothing
    /// changes state while the EMA sits inside it.
    pub fn record_precision(&mut self, precision: f64) {
        let p = precision.clamp(0.0, 1.0);
        let a = PREFETCH_HITRATE_EMA_ALPHA;
        self.hitrate_ema = a * p + (1.0 - a) * self.hitrate_ema;

        if self.enabled {
            if self.hitrate_ema < PREFETCH_PRECISION_DISABLE_FLOOR {
                self.enabled = false;
                self.ticks_since_disabled = 0;
            }
        } else if self.hitrate_ema >= PREFETCH_PRECISION_REENABLE_CEILING {
            self.enabled = true;
            self.ticks_since_disabled = 0;
        }
    }

    /// Advance the disabled-tick counter. Called once per heavy tick for a user
    /// whose prefetch is currently disabled (and that did not run a re-probe).
    pub fn note_skipped_while_disabled(&mut self) {
        if !self.enabled {
            self.ticks_since_disabled = self.ticks_since_disabled.saturating_add(1);
        }
    }
}

/// Predict which memories are "due" for access soon, given a per-user dominant
/// rhythm period and each memory's last-access time.
///
/// B4: this consumes the SINGLE per-user aggregate period (`dominant_period_days`,
/// the smallest learned scale = fastest rhythm). It runs no FFT itself — it is a
/// cheap O(n) scan over `(memory_id, last_accessed)` pairs.
///
/// A memory is "due" when the time since its last access is within a tolerance
/// band around (a multiple of) the dominant period — i.e. the user's rhythm
/// predicts they are about to come back to it.
///
/// Returns memory ids sorted by how overdue they are (most overdue first),
/// truncated to `max_candidates`.
pub fn predict_due_memories(
    last_access: &[(crate::memory::MemoryId, DateTime<Utc>)],
    dominant_period_days: f64,
    now: DateTime<Utc>,
    max_candidates: usize,
) -> Vec<crate::memory::MemoryId> {
    if dominant_period_days <= 0.0 || last_access.is_empty() {
        return Vec::new();
    }

    // Tolerance: predict "due" within ±25% of the dominant period of the next
    // expected access. Bounds the band so very long periods don't swallow
    // everything.
    let tol = (dominant_period_days * 0.25).clamp(0.25, dominant_period_days);

    let mut scored: Vec<(crate::memory::MemoryId, f64)> = Vec::new();
    for (id, last) in last_access {
        let elapsed_days = now.signed_duration_since(*last).num_seconds() as f64 / 86_400.0;
        if elapsed_days < 0.0 {
            continue;
        }
        // How far past the nearest multiple of the period are we?
        let cycles = elapsed_days / dominant_period_days;
        let nearest = cycles.round();
        if nearest < 1.0 {
            // Hasn't completed even one period since last access — not due yet.
            continue;
        }
        let phase_days = (elapsed_days - nearest * dominant_period_days).abs();
        if phase_days <= tol {
            // Overdue score: prefer memories that are slightly past due (elapsed
            // just exceeds an integer number of periods).
            let overdue = elapsed_days - nearest * dominant_period_days;
            scored.push((id.clone(), overdue));
        }
    }

    // Most overdue first.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max_candidates);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::PREFETCH_REPROBE_INTERVAL_TICKS;
    use crate::memory::MemoryId;

    #[test]
    fn new_user_starts_enabled() {
        let st = PrefetchUserState::default();
        assert!(st.enabled);
        assert!(st.should_run_this_tick());
    }

    #[test]
    fn kill_switch_engages_below_floor_and_reprobes() {
        let mut st = PrefetchUserState::default();
        // Feed a run of zero-precision ticks; EMA decays below the floor.
        for _ in 0..50 {
            st.record_precision(0.0);
        }
        assert!(!st.enabled, "kill-switch should disable after sustained 0 precision");

        // While disabled it should not run until the re-probe window elapses.
        assert!(!st.should_run_this_tick());
        for _ in 0..PREFETCH_REPROBE_INTERVAL_TICKS {
            st.note_skipped_while_disabled();
        }
        assert!(st.should_reprobe(), "re-probe trial window must arrive");
        assert!(st.should_run_this_tick());
    }

    #[test]
    fn hysteresis_band_prevents_flapping() {
        let mut st = PrefetchUserState::default();
        for _ in 0..50 {
            st.record_precision(0.0);
        }
        assert!(!st.enabled);
        // A single mid-band sample must NOT re-enable (below the ceiling).
        st.record_precision(0.15);
        assert!(!st.enabled, "mid-band precision must not re-enable");
        // Sustained high precision climbs over the ceiling and re-enables.
        for _ in 0..20 {
            st.record_precision(1.0);
        }
        assert!(st.enabled, "sustained high precision re-enables");
    }

    #[test]
    fn scales_recompute_gate() {
        let now = Utc::now();
        let mut st = PrefetchUserState::default();
        assert!(st.scales_need_recompute(100, now), "no cache -> recompute");

        st.scales = Some(LearnedScales {
            scales: [7.0, 30.0, 365.0],
            confidence: 0.5,
            sample_count: 100,
            computed_at: now,
        });
        assert!(!st.scales_need_recompute(110, now), "small growth -> reuse cache");
        assert!(
            st.scales_need_recompute(200, now),
            "large sample growth -> recompute"
        );
        assert!(
            st.scales_need_recompute(100, now + chrono::Duration::hours(48)),
            "stale cache -> recompute"
        );
    }

    #[test]
    fn predict_due_picks_overdue_memories() {
        let now = Utc::now();
        let period = 7.0_f64;
        let due_id = MemoryId(uuid::Uuid::new_v4());
        let not_due_id = MemoryId(uuid::Uuid::new_v4());

        let last = vec![
            // Last accessed ~7 days ago -> one full period elapsed -> due.
            (due_id.clone(), now - chrono::Duration::days(7)),
            // Last accessed ~3 days ago -> mid-cycle -> not due.
            (not_due_id.clone(), now - chrono::Duration::days(3)),
        ];

        let due = predict_due_memories(&last, period, now, 10);
        assert!(due.contains(&due_id), "memory at one full period should be due");
        assert!(
            !due.contains(&not_due_id),
            "mid-cycle memory should not be due"
        );
    }

    #[test]
    fn predict_due_empty_on_degenerate_inputs() {
        let now = Utc::now();
        assert!(predict_due_memories(&[], 7.0, now, 10).is_empty());
        let one = vec![(MemoryId(uuid::Uuid::new_v4()), now)];
        assert!(predict_due_memories(&one, 0.0, now, 10).is_empty());
    }
}
