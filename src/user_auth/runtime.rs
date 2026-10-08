//! Glue layer between the user-auth store, the TOTP secret encryptor, and the
//! per-username login throttle.
//!
//! Holds the long-lived state every user-auth handler needs:
//!   - the persistent [`store::UserAuthStore`];
//!   - a per-username login [`LoginLimiter`] (token bucket, 5 attempts /
//!     15 min) — distinct from the existing peer-IP governance layer;
//!   - the optional [`SecretCryptor`] used to seal TOTP secrets at rest.
//!
//! The handlers do not own RocksDB or any cryptographic key; they consume
//! a [`UserAuthRuntime`] and call its methods. This keeps test wiring small
//! (a `UserAuthRuntime` is constructible from a single CF in a tempdir DB)
//! and makes the rate-limiter behaviour deterministic in tests by allowing
//! the burst quota to be tuned per construction call.

use std::sync::Arc;
use std::time::Duration;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use crate::keystore::SecretCryptor;
use crate::user_auth::store::UserAuthStore;

/// Default lockout window for the per-username login limiter: 5 attempts
/// per 15 minutes, leaky-bucket replenished by `1 attempt / 3 min`.
pub const DEFAULT_LOGIN_BURST: u32 = 5;
pub const DEFAULT_LOGIN_REPLENISH_PERIOD: Duration = Duration::from_secs(3 * 60);
const MAX_TRACKED_KEYS: usize = 10_000;

/// Per-key attempt throttle: a token bucket per lowercased key, refilled one token
/// per `replenish` up to `burst`. Hand-rolled rather than a governor keyed limiter
/// because an administrator must be able to clear one key, and governor has no
/// per-key reset.
#[derive(Clone)]
pub struct LoginLimiter {
    buckets: Arc<Mutex<HashMap<String, Bucket>>>,
    burst: u32,
    replenish: Duration,
}

#[derive(Clone, Copy)]
struct Bucket {
    tokens: u32,
    /// When the bucket last gained, or started counting toward, a token.
    since: Instant,
}

impl LoginLimiter {
    /// Build a fresh limiter with the given burst (max attempts before
    /// throttling) and replenish period (time to refill one attempt).
    pub fn new(burst: u32, replenish: Duration) -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
            burst: burst.max(1),
            replenish: if replenish.is_zero() {
                Duration::from_secs(1)
            } else {
                replenish
            },
        }
    }

    /// Convenience: default config (5 / 15 min).
    pub fn default_login_throttle() -> Self {
        Self::new(DEFAULT_LOGIN_BURST, DEFAULT_LOGIN_REPLENISH_PERIOD)
    }

    fn refilled(&self, bucket: Bucket, now: Instant) -> Bucket {
        let elapsed = now.saturating_duration_since(bucket.since);
        let gained = (elapsed.as_nanos() / self.replenish.as_nanos()) as u64;
        if gained == 0 {
            return bucket;
        }
        let tokens = (bucket.tokens as u64 + gained).min(self.burst as u64) as u32;
        let since = if tokens == self.burst {
            now
        } else {
            bucket.since + self.replenish * gained as u32
        };
        Bucket { tokens, since }
    }

    fn wait(&self, bucket: Bucket, now: Instant) -> u64 {
        let next = bucket.since + self.replenish;
        next.saturating_duration_since(now).as_secs().max(1)
    }

    /// Try to consume one login attempt for `username`.
    /// `Ok(())` on success; `Err(retry_after_secs)` on throttle.
    pub fn check(&self, username: &str) -> Result<(), u64> {
        let now = Instant::now();
        let mut buckets = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        // A full bucket carries no state, so dropping it is free; this bounds what a
        // flood of made-up usernames can make the map hold.
        if buckets.len() > MAX_TRACKED_KEYS {
            buckets.retain(|_, b| self.refilled(*b, now).tokens < self.burst);
        }
        let bucket = buckets.entry(username.to_lowercase()).or_insert(Bucket {
            tokens: self.burst,
            since: now,
        });
        let current = self.refilled(*bucket, now);
        if current.tokens == 0 {
            *bucket = current;
            return Err(self.wait(current, now));
        }
        *bucket = Bucket {
            tokens: current.tokens - 1,
            since: if current.tokens == self.burst {
                now
            } else {
                current.since
            },
        };
        Ok(())
    }

    /// `Err(retry_after_secs)` when `key` has no attempt left, without consuming one.
    pub fn peek(&self, key: &str) -> Result<(), u64> {
        let now = Instant::now();
        let buckets = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        match buckets.get(&key.to_lowercase()) {
            Some(bucket) => {
                let current = self.refilled(*bucket, now);
                if current.tokens == 0 {
                    Err(self.wait(current, now))
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }
    }

    /// Give `key` its full burst back.
    pub fn reset(&self, key: &str) {
        self.buckets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&key.to_lowercase());
    }
}

/// How long a successful step-up lets its session make further changes without
/// another code.
pub const STEP_UP_WINDOW: Duration = Duration::from_secs(5 * 60);

/// Sessions recently stepped up, by token hash. In memory: a restart asks again.
#[derive(Clone, Default)]
pub struct Elevations(Arc<Mutex<HashMap<[u8; 32], Instant>>>);

impl Elevations {
    pub fn elevate(&self, session: &[u8; 32]) {
        let now = Instant::now();
        let mut map = self.0.lock().unwrap_or_else(|p| p.into_inner());
        map.retain(|_, until| *until > now);
        map.insert(*session, now + STEP_UP_WINDOW);
    }

    pub fn is_elevated(&self, session: &[u8; 32]) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session)
            .is_some_and(|until| *until > Instant::now())
    }

    pub fn drop_session(&self, session: &[u8; 32]) {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(session);
    }
}

/// Aggregated long-lived state every user-auth handler needs.
#[derive(Clone)]
pub struct UserAuthRuntime {
    pub store: UserAuthStore,
    pub login_limiter: LoginLimiter,
    pub elevations: Elevations,
    /// Encryptor used to seal TOTP secrets at rest (encryption-v2, keystore-derived
    /// "totp" purpose key). `None` when `VELD_KEYSTORE_PATH` / `VELD_MASTER_PASSPHRASE`
    /// are unset; production mode then refuses 2FA enrollment (see
    /// `AuthError::TotpEncryptionRequired`).
    pub totp_encryptor: Option<SecretCryptor>,
}

impl UserAuthRuntime {
    pub fn new(store: UserAuthStore, totp_encryptor: Option<SecretCryptor>) -> Self {
        Self {
            store,
            login_limiter: LoginLimiter::default_login_throttle(),
            elevations: Elevations::default(),
            totp_encryptor,
        }
    }

    /// Test-only constructor with a configurable login limiter so unit
    /// tests can exercise throttling without 15-minute wall clocks.
    #[cfg(test)]
    pub fn with_limiter(
        store: UserAuthStore,
        login_limiter: LoginLimiter,
        totp_encryptor: Option<SecretCryptor>,
    ) -> Self {
        Self {
            store,
            login_limiter,
            elevations: Elevations::default(),
            totp_encryptor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_locks_out_after_burst_attempts() {
        // 3 attempts / 1s replenish: easier to test than the 15-minute default.
        let limiter = LoginLimiter::new(3, Duration::from_secs(1));
        assert!(limiter.check("alice").is_ok());
        assert!(limiter.check("alice").is_ok());
        assert!(limiter.check("alice").is_ok());
        // 4th attempt within the burst window must be throttled.
        let err = limiter.check("alice").unwrap_err();
        assert!(err >= 1, "expected positive retry-after, got {err}");
    }

    #[test]
    fn limiter_throttle_is_per_username() {
        let limiter = LoginLimiter::new(2, Duration::from_secs(1));
        // Drain alice's bucket.
        assert!(limiter.check("alice").is_ok());
        assert!(limiter.check("alice").is_ok());
        assert!(limiter.check("alice").is_err());
        // Bob's bucket is independent.
        assert!(limiter.check("bob").is_ok());
        assert!(limiter.check("bob").is_ok());
        assert!(limiter.check("bob").is_err());
    }

    #[test]
    fn peek_reports_without_spending_and_reset_restores_the_burst() {
        let limiter = LoginLimiter::new(2, Duration::from_secs(60));
        assert!(limiter.peek("alice").is_ok());
        assert!(limiter.peek("alice").is_ok(), "peeking spends nothing");
        limiter.check("alice").unwrap();
        limiter.check("alice").unwrap();
        assert!(limiter.peek("alice").is_err());
        assert!(limiter.check("alice").is_err());
        limiter.reset("ALICE");
        assert!(
            limiter.check("alice").is_ok(),
            "an admin unlock clears the key"
        );
    }

    #[test]
    fn a_spent_bucket_refills_one_attempt_per_period() {
        let limiter = LoginLimiter::new(1, Duration::from_millis(40));
        limiter.check("alice").unwrap();
        assert!(limiter.check("alice").is_err());
        std::thread::sleep(Duration::from_millis(100));
        assert!(limiter.check("alice").is_ok());
    }

    #[test]
    fn limiter_lowercases_username_keys() {
        let limiter = LoginLimiter::new(1, Duration::from_secs(60));
        // "ALICE" and "alice" share the same bucket.
        assert!(limiter.check("ALICE").is_ok());
        assert!(limiter.check("alice").is_err());
    }
}
