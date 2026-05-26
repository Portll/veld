//! Glue layer between the user-auth store, the field encryptor, and the
//! per-username login throttle.
//!
//! Holds the long-lived state every user-auth handler needs:
//!   - the persistent [`store::UserAuthStore`];
//!   - a per-username login [`LoginLimiter`] (governor::keyed, 5 attempts /
//!     15 min) — distinct from the existing peer-IP governance layer;
//!   - the optional [`FieldEncryptor`] used to seal TOTP secrets at rest.
//!
//! The handlers do not own RocksDB or any cryptographic key; they consume
//! a [`UserAuthRuntime`] and call its methods. This keeps test wiring small
//! (a `UserAuthRuntime` is constructible from a single CF in a tempdir DB)
//! and makes the rate-limiter behaviour deterministic in tests by allowing
//! the burst quota to be tuned per construction call.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use governor::{
    clock::{Clock, DefaultClock},
    middleware::NoOpMiddleware,
    state::keyed::DefaultKeyedStateStore,
    Quota, RateLimiter,
};

use crate::encryption::FieldEncryptor;
use crate::user_auth::store::UserAuthStore;

/// Default lockout window for the per-username login limiter: 5 attempts
/// per 15 minutes, leaky-bucket replenished by `1 attempt / 3 min`.
pub const DEFAULT_LOGIN_BURST: u32 = 5;
pub const DEFAULT_LOGIN_REPLENISH_PERIOD: Duration = Duration::from_secs(3 * 60);

type UsernameLimiter =
    RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock, NoOpMiddleware>;

/// Per-username login throttle.
///
/// Backed directly by `governor::RateLimiter` (the same library Veld's
/// existing peer-IP governance layer uses — we reuse the library, not the
/// existing limiter instance, because the existing one is keyed on
/// `IpAddr`).
#[derive(Clone)]
pub struct LoginLimiter {
    inner: Arc<UsernameLimiter>,
    cap_secs: u64,
}

impl LoginLimiter {
    /// Build a fresh limiter with the given burst (max attempts before
    /// throttling) and replenish period (time to refill one attempt).
    pub fn new(burst: u32, replenish: Duration) -> Self {
        let burst = NonZeroU32::new(burst.max(1)).expect("burst >= 1");
        let replenish = if replenish.is_zero() {
            Duration::from_secs(1)
        } else {
            replenish
        };
        let quota = Quota::with_period(replenish)
            .expect("non-zero replenish")
            .allow_burst(burst);
        // Approximate wait-time cap: burst * replenish secs, clamped to >=1.
        let cap_secs = ((burst.get() as u64) * replenish.as_secs()).max(1);
        Self {
            inner: Arc::new(RateLimiter::keyed(quota)),
            cap_secs,
        }
    }

    /// Convenience: default config (5 / 15 min).
    pub fn default_login_throttle() -> Self {
        Self::new(DEFAULT_LOGIN_BURST, DEFAULT_LOGIN_REPLENISH_PERIOD)
    }

    /// Try to consume one login attempt for `username`.
    /// `Ok(())` on success; `Err(retry_after_secs)` on throttle.
    pub fn check(&self, username: &str) -> Result<(), u64> {
        let key = username.to_lowercase();
        match self.inner.check_key(&key) {
            Ok(()) => Ok(()),
            Err(negative) => {
                let wait = negative
                    .wait_time_from(DefaultClock::default().now())
                    .as_secs();
                Err(wait.min(self.cap_secs).max(1))
            }
        }
    }
}

/// Aggregated long-lived state every user-auth handler needs.
#[derive(Clone)]
pub struct UserAuthRuntime {
    pub store: UserAuthStore,
    pub login_limiter: LoginLimiter,
    /// Encryptor used to seal TOTP secrets at rest. `None` when
    /// `VELD_ENCRYPTION_KEY` is unset; production mode then refuses 2FA
    /// enrollment (see `AuthError::TotpEncryptionRequired`).
    pub field_encryptor: Option<FieldEncryptor>,
}

impl UserAuthRuntime {
    pub fn new(store: UserAuthStore, field_encryptor: Option<FieldEncryptor>) -> Self {
        Self {
            store,
            login_limiter: LoginLimiter::default_login_throttle(),
            field_encryptor,
        }
    }

    /// Test-only constructor with a configurable login limiter so unit
    /// tests can exercise throttling without 15-minute wall clocks.
    #[cfg(test)]
    pub fn with_limiter(
        store: UserAuthStore,
        login_limiter: LoginLimiter,
        field_encryptor: Option<FieldEncryptor>,
    ) -> Self {
        Self {
            store,
            login_limiter,
            field_encryptor,
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
    fn limiter_lowercases_username_keys() {
        let limiter = LoginLimiter::new(1, Duration::from_secs(60));
        // "ALICE" and "alice" share the same bucket.
        assert!(limiter.check("ALICE").is_ok());
        assert!(limiter.check("alice").is_err());
    }
}
