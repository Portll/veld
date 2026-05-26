//! Opaque session tokens for self-hosted user auth.
//!
//! Tokens are 32 cryptographically-random bytes encoded as URL-safe base64
//! without padding (43 chars). The plaintext token is shown to the client
//! exactly once at login; the server only ever persists a SHA-256 digest of
//! the bytes, so a database leak alone cannot resurrect a live session.
//!
//! Lifetime model:
//!   - Default expiry: 24 hours from issuance.
//!   - Refresh-on-use: every successful lookup of a not-yet-expired session
//!     extends `expires_at` to `now + 24h`. This gives an "idle-only" model
//!     — sessions die after 24h of inactivity, not 24h from initial login —
//!     which is what TUI/GUI clients want.
//!   - Hard cap: there is no maximum total lifetime today. If a user wants
//!     to invalidate a long-lived session they can `POST /logout` or, via
//!     password recovery, invalidate every session at once.
//!
//! Storage: the lookup key is the SHA-256 digest of the plaintext bytes.
//! That digest is non-secret (rainbow tables are useless: 32 random bytes ≠
//! a guessable password), so we can use it directly as a key without
//! per-row Argon2 work. The result: O(1) session resolution per request,
//! which is what an interactive API needs.

use crate::user_auth::AuthError;

use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Opaque session token wire shape (32 bytes, url-safe base64 no padding).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionToken(pub String);

/// Token byte length before encoding.
pub const TOKEN_BYTES: usize = 32;
/// Default token TTL (also the refresh window).
pub const DEFAULT_TTL_SECS: i64 = 24 * 60 * 60;

/// Persisted session record. The plaintext token never lives here — only
/// `token_hash` does (SHA-256 of the 32 random bytes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    pub user_id: Uuid,
    pub token_hash: [u8; 32],
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

impl SessionRecord {
    /// Has this session passed its expiry timestamp?
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.expires_at
    }

    /// Move the expiry to `now + DEFAULT_TTL_SECS`.
    pub fn refresh(&mut self, now: DateTime<Utc>) {
        self.expires_at = now + Duration::seconds(DEFAULT_TTL_SECS);
    }
}

/// Generate a fresh `(plaintext_token, record)` pair for the given user.
pub fn issue(user_id: Uuid, now: DateTime<Utc>) -> (SessionToken, SessionRecord) {
    let mut bytes = [0u8; TOKEN_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let token_str = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let token_hash = sha256(&bytes);
    let record = SessionRecord {
        user_id,
        token_hash,
        created_at: now,
        expires_at: now + Duration::seconds(DEFAULT_TTL_SECS),
    };
    (SessionToken(token_str), record)
}

/// Compute the lookup hash for a plaintext token presented by a client.
///
/// Returns `Err(AuthError::InvalidSession)` if the token is malformed
/// (wrong length after base64 decode, or non-base64 input).
pub fn hash_token(token: &str) -> Result<[u8; 32], AuthError> {
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token.as_bytes())
        .map_err(|_| AuthError::InvalidSession)?;
    if decoded.len() != TOKEN_BYTES {
        return Err(AuthError::InvalidSession);
    }
    Ok(sha256(&decoded))
}

fn sha256(input: &[u8]) -> [u8; 32] {
    Sha256::digest(input).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_then_hash_match() {
        let user = Uuid::new_v4();
        let now = Utc::now();
        let (SessionToken(token), record) = issue(user, now);
        assert_eq!(token.len(), 43, "base64 url-safe no-pad of 32 bytes = 43 chars");
        let derived = hash_token(&token).expect("hash plaintext");
        assert_eq!(derived, record.token_hash);
    }

    #[test]
    fn issue_produces_distinct_tokens() {
        let user = Uuid::new_v4();
        let now = Utc::now();
        let (a, _) = issue(user, now);
        let (b, _) = issue(user, now);
        assert_ne!(a, b);
    }

    #[test]
    fn expires_at_is_default_ttl_after_creation() {
        let user = Uuid::new_v4();
        let now = Utc::now();
        let (_, record) = issue(user, now);
        let delta = (record.expires_at - record.created_at).num_seconds();
        assert_eq!(delta, DEFAULT_TTL_SECS);
        assert!(!record.is_expired(now));
        assert!(!record.is_expired(now + Duration::seconds(DEFAULT_TTL_SECS - 1)));
        // At the boundary (now + ttl), `now >= expires_at` so it counts as
        // expired — clients have at most `TTL - 1` second of guaranteed
        // validity, which is fine.
        assert!(record.is_expired(now + Duration::seconds(DEFAULT_TTL_SECS)));
    }

    #[test]
    fn refresh_bumps_expiry_to_now_plus_ttl() {
        let user = Uuid::new_v4();
        let t0 = Utc::now();
        let (_, mut record) = issue(user, t0);
        let original_exp = record.expires_at;

        let t1 = t0 + Duration::seconds(5 * 60);
        record.refresh(t1);
        let new_exp = record.expires_at;

        assert!(new_exp > original_exp);
        assert_eq!((new_exp - t1).num_seconds(), DEFAULT_TTL_SECS);
    }

    #[test]
    fn malformed_token_rejected() {
        // Wrong charset
        assert!(matches!(
            hash_token("not base64!!!"),
            Err(AuthError::InvalidSession)
        ));
        // Right charset, wrong length
        let short = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0u8; 8]);
        assert!(matches!(hash_token(&short), Err(AuthError::InvalidSession)));
    }
}
