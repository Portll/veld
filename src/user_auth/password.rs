//! Argon2id password hashing for self-hosted user auth (Phase C).
//!
//! Wraps the `argon2` crate's high-level `PasswordHasher` / `PasswordVerifier`
//! traits, which internally use `subtle::ConstantTimeEq` for the digest
//! comparison and a random salt sourced from the OS RNG for every new hash.
//!
//! The encoded hash string is self-describing (algorithm, version, params,
//! salt, hash), so the cost parameters used at registration travel with the
//! stored hash. Future cost increases need only swap [`PASSWORD_PARAMS`] —
//! existing hashes keep verifying with the params encoded in their string.
//!
//! ## Concurrency
//!
//! [`hash_password`] / [`verify_password`] are synchronous helpers used by
//! the unit tests and by code paths that do not run on a Tokio executor.
//! Production async paths must call [`hash_password_async`] /
//! [`verify_password_async`], which acquire one slot from the process-global
//! [`crate::user_auth::argon2_pool`] semaphore before running the work on the
//! blocking-task pool. The semaphore caps in-flight Argon2 jobs at
//! `available_parallelism()` so a thundering herd of logins cannot saturate
//! every CPU core simultaneously — queued requests park on the semaphore
//! instead.

use crate::user_auth::argon2_pool;
use crate::user_auth::AuthError;

use argon2::password_hash::{
    rand_core::OsRng as PwOsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString,
};
use argon2::{Algorithm, Argon2, Params, Version};

/// Argon2id parameters for the primary password surface.
///
/// Tuned for an interactive login path on edge hardware: ~32 MiB memory,
/// 2 passes, single lane. Yields ~50–80 ms on a modern laptop core, which
/// is comfortably under the 200 ms login budget while still pricing
/// offline guessing at hundreds of dollars per million-attempt batch on
/// commodity GPUs (Argon2id is memory-hard, so GPU/ASIC speedups are
/// bounded by memory bandwidth, not raw FLOPs).
///
/// `Params::new(m_cost_kib, t_cost, parallelism, output_len)`.
fn password_params() -> Params {
    // 32 MiB = 32 * 1024 KiB. The `output_len = None` lets argon2 pick the
    // default 32-byte tag, which is what the encoded format expects.
    Params::new(32 * 1024, 2, 1, None)
        .expect("Argon2 password params are statically valid")
}

/// Build the Argon2id hasher used for password operations.
fn password_hasher() -> Argon2<'static> {
    Argon2::new(Algorithm::Argon2id, Version::V0x13, password_params())
}

/// Hash a plaintext password into an Argon2id encoded string.
///
/// Salt is generated freshly per call from the OS RNG (`OsRng`).
/// Returns the PHC string `$argon2id$v=19$m=...,t=...,p=...$<salt>$<hash>`.
pub fn hash_password(plaintext: &str) -> Result<String, AuthError> {
    if plaintext.is_empty() {
        return Err(AuthError::weak_password("password cannot be empty"));
    }
    let salt = SaltString::generate(&mut PwOsRng);
    let hasher = password_hasher();
    hasher
        .hash_password(plaintext.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| AuthError::internal(format!("argon2 hash failed: {e}")))
}

/// Verify a plaintext password against a previously-stored Argon2id hash.
///
/// Uses `argon2::PasswordVerifier::verify_password`, which internally calls
/// `subtle::ConstantTimeEq` on the recomputed digest — the comparison is
/// constant-time with respect to the secret bytes.
pub fn verify_password(plaintext: &str, stored_hash: &str) -> Result<bool, AuthError> {
    let parsed = PasswordHash::new(stored_hash)
        .map_err(|e| AuthError::internal(format!("malformed stored password hash: {e}")))?;
    let hasher = password_hasher();
    match hasher.verify_password(plaintext.as_bytes(), &parsed) {
        Ok(()) => Ok(true),
        Err(argon2::password_hash::Error::Password) => Ok(false),
        Err(other) => Err(AuthError::internal(format!(
            "argon2 verify failed: {other}"
        ))),
    }
}

/// Async wrapper around [`hash_password`] that runs the CPU-bound work on
/// the blocking-task pool *under* the process-global Argon2 semaphore. Use
/// from any handler reached through Axum — see the module-level
/// "Concurrency" note.
pub async fn hash_password_async(plaintext: &str) -> Result<String, AuthError> {
    let plain = plaintext.to_owned();
    argon2_pool::run_blocking(move || hash_password(&plain)).await
}

/// Async wrapper around [`verify_password`] with the same concurrency
/// semantics as [`hash_password_async`].
pub async fn verify_password_async(
    plaintext: &str,
    stored_hash: &str,
) -> Result<bool, AuthError> {
    let plain = plaintext.to_owned();
    let stored = stored_hash.to_owned();
    argon2_pool::run_blocking(move || verify_password(&plain, &stored)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_then_verify_succeeds() {
        let pw = "correct horse battery staple";
        let hash = hash_password(pw).expect("hash");
        assert!(hash.starts_with("$argon2id$"), "phc string shape: {hash}");
        assert!(verify_password(pw, &hash).expect("verify_ok"));
    }

    #[test]
    fn wrong_password_rejected() {
        let hash = hash_password("right-password").expect("hash");
        assert!(!verify_password("wrong-password", &hash).expect("verify"));
    }

    #[test]
    fn two_hashes_of_same_password_differ() {
        // Distinct random salts must produce distinct encoded hashes.
        let pw = "same-input";
        let h1 = hash_password(pw).expect("hash1");
        let h2 = hash_password(pw).expect("hash2");
        assert_ne!(h1, h2, "salts must differ between hash invocations");
        // Both still verify against the original plaintext.
        assert!(verify_password(pw, &h1).unwrap());
        assert!(verify_password(pw, &h2).unwrap());
    }

    #[test]
    fn malformed_hash_returns_internal_error() {
        let err = verify_password("anything", "not-a-phc-string").unwrap_err();
        // Internal mapping: it's a server-side data integrity issue, not a
        // user-visible bad password.
        assert!(matches!(err, AuthError::Internal(_)), "got {err:?}");
    }

    #[test]
    fn empty_password_rejected_at_hash_time() {
        let err = hash_password("").unwrap_err();
        assert!(
            matches!(err, AuthError::WeakPassword(_)),
            "expected WeakPassword, got {err:?}"
        );
    }

    #[tokio::test]
    async fn async_hash_and_verify_round_trip_through_the_pool() {
        // Smoke-test the async path so we know the semaphore wrapper still
        // produces the same answer the synchronous helpers would.
        let pw = "async-correct-horse";
        let hash = hash_password_async(pw).await.expect("hash_async");
        assert!(hash.starts_with("$argon2id$"));
        assert!(verify_password_async(pw, &hash).await.expect("verify_async"));
        assert!(!verify_password_async("nope", &hash).await.expect("mismatch"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_verifies_respect_the_semaphore_cap() {
        // Spawn 20 verifies concurrently. The Argon2 pool must keep
        // in-flight work below the permit cap; we sample the cap by reading
        // the available_permits counter from inside each task and capping
        // at `permit_count() - available`.
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let pw = "rate-limited-password";
        let hash = hash_password_async(pw).await.expect("hash_async");

        let cap = crate::user_auth::argon2_pool::permit_count();
        let peak = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(20);
        for _ in 0..20 {
            let pw = pw.to_owned();
            let hash = hash.clone();
            let peak = Arc::clone(&peak);
            handles.push(tokio::spawn(async move {
                let ok = verify_password_async(&pw, &hash).await.expect("verify");
                assert!(ok);
                // Sample the in-flight count from inside the spawned task.
                // available_permits is a lower bound (some tasks may have
                // already finished by the time we read it), so
                // `cap - available` upper-bounds the in-flight value.
                let avail = crate::user_auth::argon2_pool::available_permits();
                let in_flight = cap.saturating_sub(avail);
                peak.fetch_max(in_flight, Ordering::SeqCst);
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
    }
}
