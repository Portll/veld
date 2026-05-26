//! RFC 6238 TOTP verification, backed by the `totp-rs` crate.
//!
//! We use the SHA-1 / 30-second / 6-digit profile — the de-facto baseline
//! every modern authenticator app (Aegis, Google Authenticator, 1Password,
//! Bitwarden, Microsoft Authenticator, …) accepts out of the box. SHA-1 is
//! cryptographically suitable in HMAC mode (HMAC-SHA1 is still considered
//! safe per current NIST guidance for HMAC use); modern algorithms (SHA-256,
//! SHA-512) are interoperable in theory but break compatibility with the
//! authenticator-app long tail.
//!
//! Time-skew tolerance: we accept the previous and next 30-second windows in
//! addition to the current one (`±1` step), matching the RFC's allowance for
//! clients with mildly mis-synced clocks.

use crate::user_auth::AuthError;
use totp_rs::{Algorithm, Secret, TOTP};

/// TOTP digit count (RFC 6238 §5.3, recommended 6).
pub const TOTP_DIGITS: usize = 6;
/// TOTP step in seconds (RFC 6238 §5.2, baseline 30).
pub const TOTP_STEP_SECS: u64 = 30;
/// Time-skew tolerance, in steps either side of the current one.
pub const TOTP_SKEW_STEPS: u8 = 1;
/// Minimum recommended secret length for HMAC-SHA1 — 20 bytes is the RFC 4226
/// floor (one HMAC block). Authenticator apps universally accept this length.
pub const TOTP_SECRET_BYTES: usize = 20;

const ISSUER: &str = "Veld";

/// Generate a fresh 20-byte (160-bit) TOTP secret from the OS RNG.
pub fn generate_secret() -> Vec<u8> {
    use rand::RngCore;
    let mut bytes = vec![0u8; TOTP_SECRET_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    bytes
}

/// Build a `TOTP` instance from a raw secret. Returns an internal error if
/// the secret is malformed (e.g. shorter than the SHA-1 HMAC block size).
fn build_totp(secret: &[u8], account: &str) -> Result<TOTP, AuthError> {
    // `totp-rs` requires the secret as `Vec<u8>`; we own a copy so the
    // caller's reference outlives the construction.
    let raw = Secret::Raw(secret.to_vec()).to_bytes().map_err(|e| {
        AuthError::internal(format!("invalid TOTP secret encoding: {e:?}"))
    })?;

    TOTP::new(
        Algorithm::SHA1,
        TOTP_DIGITS,
        TOTP_SKEW_STEPS,
        TOTP_STEP_SECS,
        raw,
        Some(ISSUER.to_string()),
        account.to_string(),
    )
    .map_err(|e| AuthError::internal(format!("TOTP construction failed: {e:?}")))
}

/// Verify a candidate 6-digit code against the secret at `unix_timestamp`.
///
/// `unix_timestamp` is seconds since the UNIX epoch (use [`current_unix_time`]
/// for "now"). The verification window is `±TOTP_SKEW_STEPS` steps around the
/// supplied timestamp.
pub fn verify_code(
    secret: &[u8],
    candidate: &str,
    unix_timestamp: u64,
) -> Result<bool, AuthError> {
    if candidate.len() != TOTP_DIGITS || !candidate.chars().all(|c| c.is_ascii_digit()) {
        return Ok(false);
    }
    let totp = build_totp(secret, "verify")?;
    Ok(totp.check(candidate, unix_timestamp))
}

/// Build an `otpauth://totp/...` provisioning URI for QR-code enrollment.
///
/// `account` is the label shown in the authenticator app (typically the
/// username). The issuer is always `"Veld"`.
pub fn provisioning_uri(secret: &[u8], account: &str) -> Result<String, AuthError> {
    let totp = build_totp(secret, account)?;
    Ok(totp.get_url())
}

/// Current wall-clock time as UNIX seconds. Helper for callers that don't
/// want to thread a clock through the API. Falls back to 0 on negative
/// system clocks (which would indicate a configuration error worth logging
/// upstream — verification on a zero timestamp will simply fail).
pub fn current_unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_else(|_| {
            tracing::warn!("system clock is before UNIX epoch; TOTP using 0");
            0
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use totp_rs::{Algorithm, Secret, TOTP};

    /// RFC 6238 Appendix B test vector for SHA-1 with the seed
    /// `"12345678901234567890"` (20 ASCII bytes).
    ///
    /// At T = 59 s the expected truncated 8-digit code is `94287082`.
    /// We use the canonical 6-digit profile, which yields `287082` (the
    /// trailing 6 of the 8-digit output).
    #[test]
    fn rfc6238_known_vector_t59() {
        let secret = b"12345678901234567890".to_vec();
        // Build a TOTP matching RFC's expectation (8 digits / 30s / SHA-1)
        // so we can read the canonical RFC 8-digit value, then confirm the
        // production-profile (6-digit) tail at the same moment.
        let rfc = TOTP::new(
            Algorithm::SHA1,
            8,
            0,
            30,
            Secret::Raw(secret.clone()).to_bytes().unwrap(),
            None,
            "rfc".to_string(),
        )
        .unwrap();
        assert_eq!(rfc.generate(59), "94287082");

        // Production profile (6 digits) — same secret, same moment.
        assert!(
            verify_code(&secret, "287082", 59).unwrap(),
            "RFC vector at T=59s must verify under our 6-digit profile"
        );
    }

    #[test]
    fn rfc6238_known_vector_t1111111109_six_digit_tail() {
        // RFC Appendix B: at T = 1_111_111_109 the SHA-1 8-digit code is "07081804".
        let secret = b"12345678901234567890".to_vec();
        assert!(verify_code(&secret, "081804", 1_111_111_109).unwrap());
    }

    #[test]
    fn skew_tolerance_one_step_either_side() {
        // Generate a code at t=10000 (window centred on step 333) and
        // confirm verification at the previous and next step boundaries.
        let secret = generate_secret();
        let t = 10_000u64;
        let totp = build_totp(&secret, "skew").unwrap();
        let code = totp.generate(t);

        // Same step
        assert!(verify_code(&secret, &code, t).unwrap());
        // -1 step (t - 30s): must still verify under ±1 skew.
        assert!(verify_code(&secret, &code, t.saturating_sub(TOTP_STEP_SECS)).unwrap());
        // +1 step (t + 30s)
        assert!(verify_code(&secret, &code, t + TOTP_STEP_SECS).unwrap());
        // -2 steps must NOT verify (outside skew window).
        assert!(!verify_code(&secret, &code, t.saturating_sub(2 * TOTP_STEP_SECS)).unwrap());
        // +2 steps must NOT verify.
        assert!(!verify_code(&secret, &code, t + 2 * TOTP_STEP_SECS).unwrap());
    }

    #[test]
    fn non_numeric_or_wrong_length_rejected() {
        let secret = generate_secret();
        let now = 100_000u64;
        assert!(!verify_code(&secret, "abcdef", now).unwrap());
        assert!(!verify_code(&secret, "12345", now).unwrap()); // too short
        assert!(!verify_code(&secret, "1234567", now).unwrap()); // too long
        assert!(!verify_code(&secret, "", now).unwrap());
    }

    #[test]
    fn provisioning_uri_has_issuer_and_account() {
        let secret = generate_secret();
        let uri = provisioning_uri(&secret, "alice@example.com").unwrap();
        assert!(uri.starts_with("otpauth://totp/"), "shape: {uri}");
        assert!(uri.contains("Veld"), "issuer in label/query: {uri}");
        // The label is URL-encoded; `alice` is unambiguous before the `@`.
        assert!(
            uri.contains("alice"),
            "account label appears in uri: {uri}"
        );
        // Secret + issuer always appear as query params per RFC. totp-rs
        // 5.x omits default algorithm (SHA1) / digits (6) / period (30)
        // from the query string because authenticator apps already assume
        // those baselines — we only require `secret=` to be present.
        assert!(uri.contains("secret="), "secret query param: {uri}");
        assert!(uri.contains("issuer=Veld"), "issuer query param: {uri}");
    }

    #[test]
    fn generate_secret_is_twenty_bytes_and_high_entropy() {
        let a = generate_secret();
        let b = generate_secret();
        assert_eq!(a.len(), TOTP_SECRET_BYTES);
        assert_eq!(b.len(), TOTP_SECRET_BYTES);
        assert_ne!(a, b, "two fresh secrets should not collide");
    }
}
