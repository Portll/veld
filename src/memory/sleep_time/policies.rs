//! Budget, lock, and debounce policy enforcement for sleep-time.
//!
//! Three concerns live here:
//!
//! 1. **Per-user budget** — hourly token + daily call caps (R12 / R20). State
//!    is persisted in the `sleep_time_budget` RocksDB CF so caps survive
//!    restart. A pre-flight `try_charge_*` call atomically (under a per-user
//!    Mutex) checks and reserves capacity; the caller settles with
//!    `record_actual_*` once the LLM response is in (R39 WAL-style ledger).
//!
//! 2. **Block lock state** — per-user list of locked block keys (R14). Locked
//!    blocks are skipped by the worker; they never auto-unlock (R22).
//!
//! 3. **Per-user debounce** — collapses repeated enqueues within a window.
//!    Lives in memory only; queue persistence (R31) is separate.
//!
//! A separate [`GlobalBudgetState`] tracks the all-users daily envelope (R33)
//! so a single tenant cannot exhaust the API key for everyone.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use parking_lot::Mutex;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, DB};
use std::collections::HashMap;
use std::sync::Arc;

use super::types::{BudgetState, GlobalBudgetState, SleepMode, SleepTimeError, SleepTimeResult};

/// Column family for per-user budget ledger.
pub const CF_SLEEP_TIME_BUDGET: &str = "sleep_time_budget";

/// Column family for global / all-users budget envelope (single key).
pub const CF_SLEEP_TIME_GLOBAL_BUDGET: &str = "sleep_time_global_budget";

const GLOBAL_KEY: &[u8] = b"global";

// =============================================================================
// Configuration
// =============================================================================

/// Tunable thresholds; defaults in `constants.rs` (V1 conservative values).
#[derive(Debug, Clone, Copy)]
pub struct PolicyLimits {
    pub tokens_per_hour: u32,
    pub calls_per_day: u32,
    pub global_tokens_per_day: u64,
    pub global_calls_per_day: u64,
    pub idle_threshold_secs: i64,
    pub debounce_secs: i64,
}

impl Default for PolicyLimits {
    fn default() -> Self {
        Self {
            tokens_per_hour: 10_000,
            calls_per_day: 50,
            global_tokens_per_day: 5_000_000,
            global_calls_per_day: 10_000,
            idle_threshold_secs: 90,
            debounce_secs: 300,
        }
    }
}

// =============================================================================
// Budget tracker
// =============================================================================

pub struct BudgetTracker {
    db: Arc<DB>,
    limits: PolicyLimits,
    /// Per-user mutexes so the read-check-write sequence for charges is
    /// linearised. Created lazily.
    user_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Global tracker is one row — single mutex is fine.
    global_lock: Mutex<()>,
}

impl BudgetTracker {
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        vec![
            ColumnFamilyDescriptor::new(CF_SLEEP_TIME_BUDGET, opts.clone()),
            ColumnFamilyDescriptor::new(CF_SLEEP_TIME_GLOBAL_BUDGET, opts),
        ]
    }

    pub fn new(db: Arc<DB>, limits: PolicyLimits) -> Self {
        Self {
            db,
            limits,
            user_locks: Mutex::new(HashMap::new()),
            global_lock: Mutex::new(()),
        }
    }

    fn cf(&self, name: &str) -> &ColumnFamily {
        self.db
            .cf_handle(name)
            .expect("sleep-time budget CF must exist in shared DB")
    }

    fn user_lock(&self, user_id: &str) -> Arc<Mutex<()>> {
        let mut map = self.user_locks.lock();
        map.entry(user_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Read current per-user state, rolling the windows if they have expired.
    pub fn user_state(&self, user_id: &str) -> Result<BudgetState> {
        let raw = self
            .db
            .get_cf(self.cf(CF_SLEEP_TIME_BUDGET), user_id.as_bytes())
            .context("read sleep_time_budget")?;
        let mut state: BudgetState = match raw {
            Some(bytes) => bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map(|(s, _)| s)
                .context("decode BudgetState")?,
            None => BudgetState::default(),
        };
        roll_windows(&mut state, Utc::now());
        Ok(state)
    }

    fn write_user_state(&self, user_id: &str, state: &BudgetState) -> Result<()> {
        let bytes = bincode::serde::encode_to_vec(state, bincode::config::standard())
            .context("encode BudgetState")?;
        self.db
            .put_cf(self.cf(CF_SLEEP_TIME_BUDGET), user_id.as_bytes(), bytes)
            .context("write sleep_time_budget")?;
        Ok(())
    }

    /// Read global state; rolls the daily window if expired.
    pub fn global_state(&self) -> Result<GlobalBudgetState> {
        let raw = self
            .db
            .get_cf(self.cf(CF_SLEEP_TIME_GLOBAL_BUDGET), GLOBAL_KEY)
            .context("read sleep_time_global_budget")?;
        let mut state: GlobalBudgetState = match raw {
            Some(bytes) => bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map(|(s, _)| s)
                .context("decode GlobalBudgetState")?,
            None => GlobalBudgetState::default(),
        };
        roll_global_window(&mut state, Utc::now());
        Ok(state)
    }

    fn write_global_state(&self, state: &GlobalBudgetState) -> Result<()> {
        let bytes = bincode::serde::encode_to_vec(state, bincode::config::standard())
            .context("encode GlobalBudgetState")?;
        self.db
            .put_cf(self.cf(CF_SLEEP_TIME_GLOBAL_BUDGET), GLOBAL_KEY, bytes)
            .context("write sleep_time_global_budget")?;
        Ok(())
    }

    /// Pre-flight check: would `projected_tokens` push the user past their
    /// hourly token cap or daily call cap, OR push the global pool past its
    /// daily envelope? If so, return [`SleepTimeError::BudgetExhausted`].
    /// On pass, reserve one call slot (does not yet debit tokens — token spend
    /// is settled in [`Self::record_actual_spend`]).
    pub fn try_charge_call(
        &self,
        user_id: &str,
        projected_tokens: u32,
    ) -> SleepTimeResult<()> {
        // Serialise per-user.
        let lock = self.user_lock(user_id);
        let _guard = lock.lock();

        let mut state = self.user_state(user_id)?;
        if state.calls_today >= self.limits.calls_per_day {
            return Err(SleepTimeError::BudgetExhausted {
                user_id: user_id.to_string(),
                what: format!("calls/day cap {}", self.limits.calls_per_day),
            });
        }
        if state.tokens_this_hour.saturating_add(projected_tokens)
            > self.limits.tokens_per_hour
        {
            return Err(SleepTimeError::BudgetExhausted {
                user_id: user_id.to_string(),
                what: format!(
                    "tokens/hour projection would exceed {}",
                    self.limits.tokens_per_hour
                ),
            });
        }

        // Global envelope check (R33).
        {
            let _g = self.global_lock.lock();
            let mut g = self.global_state()?;
            if g.calls_today >= self.limits.global_calls_per_day {
                return Err(SleepTimeError::BudgetExhausted {
                    user_id: user_id.to_string(),
                    what: format!("global calls/day cap {}", self.limits.global_calls_per_day),
                });
            }
            if g.tokens_today.saturating_add(projected_tokens as u64)
                > self.limits.global_tokens_per_day
            {
                return Err(SleepTimeError::BudgetExhausted {
                    user_id: user_id.to_string(),
                    what: format!(
                        "global tokens/day projection would exceed {}",
                        self.limits.global_tokens_per_day
                    ),
                });
            }
            // Reserve the call slot now; tokens settled later.
            g.calls_today = g.calls_today.saturating_add(1);
            if g.day_window_start.is_none() {
                g.day_window_start = Some(Utc::now());
            }
            self.write_global_state(&g)?;
        }

        state.calls_today = state.calls_today.saturating_add(1);
        if state.day_window_start.is_none() {
            state.day_window_start = Some(Utc::now());
        }
        self.write_user_state(user_id, &state)?;
        Ok(())
    }

    /// Settle actual token spend after an LLM call completes. Called even if
    /// the call failed mid-stream so the WAL-style ledger does not lose track
    /// of tokens already paid for (R39 / B9).
    pub fn record_actual_spend(&self, user_id: &str, actual_tokens: u32) -> Result<()> {
        let lock = self.user_lock(user_id);
        let _guard = lock.lock();

        let mut state = self.user_state(user_id)?;
        state.tokens_this_hour = state.tokens_this_hour.saturating_add(actual_tokens);
        if state.hour_window_start.is_none() {
            state.hour_window_start = Some(Utc::now());
        }
        self.write_user_state(user_id, &state)?;

        let _g = self.global_lock.lock();
        let mut g = self.global_state()?;
        g.tokens_today = g.tokens_today.saturating_add(actual_tokens as u64);
        if g.day_window_start.is_none() {
            g.day_window_start = Some(Utc::now());
        }
        self.write_global_state(&g)?;
        Ok(())
    }

    // ---- Lock state -------------------------------------------------------

    pub fn is_block_locked(&self, user_id: &str, block_key: &str) -> Result<bool> {
        let state = self.user_state(user_id)?;
        Ok(state.locked_blocks.iter().any(|k| k == block_key))
    }

    pub fn set_block_lock(&self, user_id: &str, block_key: &str, locked: bool) -> Result<()> {
        let lock = self.user_lock(user_id);
        let _guard = lock.lock();
        let mut state = self.user_state(user_id)?;
        let present = state.locked_blocks.iter().position(|k| k == block_key);
        match (present, locked) {
            (Some(_), true) => {} // already locked
            (None, false) => {}   // already unlocked
            (None, true) => state.locked_blocks.push(block_key.to_string()),
            (Some(idx), false) => {
                state.locked_blocks.swap_remove(idx);
            }
        }
        self.write_user_state(user_id, &state)?;
        Ok(())
    }
}

fn roll_windows(state: &mut BudgetState, now: DateTime<Utc>) {
    if let Some(start) = state.hour_window_start {
        if now - start >= Duration::hours(1) {
            state.tokens_this_hour = 0;
            state.hour_window_start = Some(now);
        }
    }
    if let Some(start) = state.day_window_start {
        if now - start >= Duration::days(1) {
            state.calls_today = 0;
            state.day_window_start = Some(now);
        }
    }
}

fn roll_global_window(state: &mut GlobalBudgetState, now: DateTime<Utc>) {
    if let Some(start) = state.day_window_start {
        if now - start >= Duration::days(1) {
            state.tokens_today = 0;
            state.calls_today = 0;
            state.day_window_start = Some(now);
        }
    }
}

// =============================================================================
// Debounce tracker
// =============================================================================

/// In-memory per-user debounce: collapses repeated enqueues within a window.
/// Persists nothing; on restart the next trigger always fires.
#[derive(Debug)]
pub struct DebounceTracker {
    inner: Mutex<HashMap<(String, SleepMode), DateTime<Utc>>>,
    window: Duration,
}

impl DebounceTracker {
    pub fn new(window_secs: i64) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            window: Duration::seconds(window_secs.max(0)),
        }
    }

    /// Returns true if the (user, mode) pair is within the debounce window
    /// since its last accepted trigger — caller should DROP the trigger.
    pub fn should_debounce(&self, user_id: &str, mode: SleepMode) -> bool {
        let now = Utc::now();
        let mut map = self.inner.lock();
        let key = (user_id.to_string(), mode);
        match map.get(&key) {
            Some(t) if now - *t < self.window => true,
            _ => {
                map.insert(key, now);
                false
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_test_db() -> (Arc<DB>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        let cfs = BudgetTracker::cf_descriptors();
        let db = DB::open_cf_descriptors(&db_opts, tmp.path(), cfs).unwrap();
        (Arc::new(db), tmp)
    }

    fn small_limits() -> PolicyLimits {
        PolicyLimits {
            tokens_per_hour: 1_000,
            calls_per_day: 3,
            global_tokens_per_day: 5_000,
            global_calls_per_day: 10,
            idle_threshold_secs: 60,
            debounce_secs: 5,
        }
    }

    #[test]
    fn try_charge_call_blocks_after_call_cap() {
        let (db, _tmp) = open_test_db();
        let t = BudgetTracker::new(db, small_limits());
        for _ in 0..3 {
            t.try_charge_call("u", 100).unwrap();
        }
        let err = t.try_charge_call("u", 100).unwrap_err();
        assert!(matches!(err, SleepTimeError::BudgetExhausted { .. }));
    }

    #[test]
    fn try_charge_call_blocks_when_projection_exceeds_hourly() {
        let (db, _tmp) = open_test_db();
        let t = BudgetTracker::new(db, small_limits());
        // First call: projected 600 tokens, then settled at 600.
        t.try_charge_call("u", 600).unwrap();
        t.record_actual_spend("u", 600).unwrap();
        // Second call: projected 500 → would exceed 1000/hour after 600.
        let err = t.try_charge_call("u", 500).unwrap_err();
        assert!(matches!(err, SleepTimeError::BudgetExhausted { .. }));
    }

    #[test]
    fn record_actual_spend_accumulates() {
        let (db, _tmp) = open_test_db();
        let t = BudgetTracker::new(db, small_limits());
        t.try_charge_call("u", 100).unwrap();
        t.record_actual_spend("u", 80).unwrap();
        let s = t.user_state("u").unwrap();
        assert_eq!(s.tokens_this_hour, 80);
        assert_eq!(s.calls_today, 1);
    }

    #[test]
    fn lock_state_set_and_check() {
        let (db, _tmp) = open_test_db();
        let t = BudgetTracker::new(db, small_limits());
        assert!(!t.is_block_locked("u", "persona").unwrap());
        t.set_block_lock("u", "persona", true).unwrap();
        assert!(t.is_block_locked("u", "persona").unwrap());
        t.set_block_lock("u", "persona", false).unwrap();
        assert!(!t.is_block_locked("u", "persona").unwrap());
    }

    #[test]
    fn lock_idempotent() {
        let (db, _tmp) = open_test_db();
        let t = BudgetTracker::new(db, small_limits());
        t.set_block_lock("u", "x", true).unwrap();
        t.set_block_lock("u", "x", true).unwrap(); // no-op
        let s = t.user_state("u").unwrap();
        assert_eq!(s.locked_blocks.len(), 1);
    }

    #[test]
    fn debounce_collapses_within_window() {
        let d = DebounceTracker::new(60);
        assert!(!d.should_debounce("u", SleepMode::Nrem));
        assert!(d.should_debounce("u", SleepMode::Nrem));
        // different mode is independent
        assert!(!d.should_debounce("u", SleepMode::Rem));
    }
}
