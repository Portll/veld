//! CSRF-state allocator for the OAuth `state` parameter.
//!
//! The OAuth2 RFC requires the client to generate a random `state`,
//! include it in the authorize URL, and verify it on the callback.
//! `StateJar` materializes this: `allocate` returns a 32-byte
//! `OsRng`-derived base64url string and stashes the matching
//! [`LoopbackSession`] (user_id, PKCE verifier, creation timestamp);
//! `redeem` removes and returns the session if the state is fresh.
//!
//! # Bound + TTL
//!
//! - Up to [`MAX_ENTRIES`] sessions held at once. Overflow evicts the
//!   oldest entry (FIFO) to defend against a runaway login burst.
//! - Each entry has a 10-minute TTL — a callback that arrives later is
//!   treated as missing. The background `run_cleanup` task scrubs
//!   expired entries every minute; `redeem` also enforces the TTL
//!   on the hot path so a missed cleanup tick can't surface stale state.

use base64::Engine as _;
use parking_lot::Mutex;
use rand::rngs::OsRng;
use rand::RngCore;
use secrecy::SecretBox;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;

/// Maximum concurrent in-flight login sessions before FIFO eviction kicks in.
pub const MAX_ENTRIES: usize = 100;

/// How long an unredeemed state value remains valid.
pub const TTL: Duration = Duration::from_secs(600); // 10 minutes

/// Per-login in-memory state held until the loopback callback redeems
/// the matching `state` parameter.
pub struct LoopbackSession {
    pub user_id: String,
    /// PKCE verifier secret — wrapped so accidental `format!` won't leak.
    pub pkce_verifier: SecretBox<String>,
    /// Redirect URI we passed to the authorize step. Required again at
    /// `exchange_code` time so Google's exact-match accepts the
    /// request.
    pub redirect_uri: String,
    pub created_at: Instant,
}

/// Bounded CSRF-state jar with background cleanup.
pub struct StateJar {
    inner: Mutex<HashMap<String, LoopbackSession>>,
    /// Notified by [`Drop`] so the background cleanup task can exit.
    shutdown: Arc<Notify>,
}

impl Default for StateJar {
    fn default() -> Self {
        Self::new()
    }
}

impl StateJar {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Allocate a fresh `state` and stash the matching session. When the
    /// jar is at capacity, the oldest existing entry is evicted first.
    pub fn allocate(&self, session: LoopbackSession) -> String {
        let mut bytes = [0u8; 32];
        OsRng.fill_bytes(&mut bytes);
        let state = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);

        let mut jar = self.inner.lock();
        if jar.len() >= MAX_ENTRIES {
            // FIFO eviction by `created_at` — O(n) over a bounded n=100.
            if let Some(oldest_key) = jar
                .iter()
                .min_by_key(|(_, s)| s.created_at)
                .map(|(k, _)| k.clone())
            {
                jar.remove(&oldest_key);
            }
        }
        jar.insert(state.clone(), session);
        state
    }

    /// Look up `state`, remove it from the jar, and return the matching
    /// session iff it's still within [`TTL`]. Returns `None` for
    /// unknown or expired states (caller treats both as invalid).
    pub fn redeem(&self, state: &str) -> Option<LoopbackSession> {
        let mut jar = self.inner.lock();
        let removed = jar.remove(state)?;
        if removed.created_at.elapsed() <= TTL {
            Some(removed)
        } else {
            None
        }
    }

    /// Number of in-flight sessions. Test/diagnostics only.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Handle to the shutdown notifier — supplied to the background
    /// cleanup task so it can wake on `Drop` and exit cleanly.
    pub fn shutdown_handle(&self) -> Arc<Notify> {
        Arc::clone(&self.shutdown)
    }

    /// Spawn-able cleanup loop. Wakes once a minute to evict expired
    /// entries; exits when [`Drop`] notifies its shutdown handle.
    pub async fn run_cleanup(self: Arc<Self>) {
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        let shutdown = Arc::clone(&self.shutdown);
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    self.inner.lock().retain(|_, s| s.created_at.elapsed() <= TTL);
                }
                _ = shutdown.notified() => break,
            }
        }
    }
}

impl Drop for StateJar {
    fn drop(&mut self) {
        self.shutdown.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    fn dummy_session() -> LoopbackSession {
        LoopbackSession {
            user_id: "u".to_string(),
            pkce_verifier: SecretBox::new(Box::new("pkce-x".to_string())),
            redirect_uri: "http://127.0.0.1:0".to_string(),
            created_at: Instant::now(),
        }
    }

    #[test]
    fn allocate_then_redeem_returns_session() {
        let jar = StateJar::new();
        let s = jar.allocate(dummy_session());
        let got = jar.redeem(&s).expect("redeem fresh state");
        assert_eq!(got.user_id, "u");
        assert_eq!(got.pkce_verifier.expose_secret(), "pkce-x");
    }

    #[test]
    fn redeem_consumes_the_entry() {
        let jar = StateJar::new();
        let s = jar.allocate(dummy_session());
        assert!(jar.redeem(&s).is_some());
        assert!(jar.redeem(&s).is_none(), "second redeem should fail");
    }

    #[test]
    fn unknown_state_returns_none() {
        let jar = StateJar::new();
        assert!(jar.redeem("not-allocated").is_none());
    }

    #[test]
    fn expired_state_returns_none_on_redeem() {
        let jar = StateJar::new();
        let s = jar.allocate(LoopbackSession {
            user_id: "u".to_string(),
            pkce_verifier: SecretBox::new(Box::new("v".to_string())),
            redirect_uri: "http://127.0.0.1:0".to_string(),
            created_at: Instant::now() - TTL - Duration::from_secs(1),
        });
        assert!(jar.redeem(&s).is_none());
    }

    #[test]
    fn fifo_eviction_at_capacity() {
        let jar = StateJar::new();
        // Allocate MAX_ENTRIES + 1; the first (oldest) should be evicted.
        let mut states = Vec::with_capacity(MAX_ENTRIES + 1);
        for i in 0..MAX_ENTRIES {
            let mut s = dummy_session();
            // Stagger created_at so the FIFO order is deterministic.
            s.created_at = Instant::now() - Duration::from_secs((MAX_ENTRIES - i) as u64);
            s.user_id = format!("u{i}");
            states.push(jar.allocate(s));
        }
        assert_eq!(jar.len(), MAX_ENTRIES);
        states.push(jar.allocate(dummy_session()));
        assert_eq!(jar.len(), MAX_ENTRIES);
        // The oldest (u0) should have been evicted.
        assert!(jar.redeem(&states[0]).is_none(), "u0 should be evicted");
    }
}
