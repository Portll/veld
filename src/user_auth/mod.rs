//! Self-hosted user authentication (Phase C of the auth roll-out).
//!
//! This module provides a parallel auth surface to Veld's existing API-key
//! mechanism. It is designed to back the planned Tauri GUI and TUI git
//! viewer: a human logs in with a username + password + TOTP and receives
//! an opaque bearer session token that authenticates subsequent calls.
//!
//! The entire surface is gated behind the `VELD_USER_AUTH_ENABLED=true`
//! environment variable — when disabled, routes 404 and no RocksDB column
//! family is created.
//!
//! ## Wire types
//!
//! - [`UserRecord`] — persistent per-user state (hashed credentials, role,
//!   2FA, recovery codes).
//! - [`UserRole`] — `Admin` | `User`.
//! - [`session::SessionToken`] — opaque 32-byte url-safe-base64 bearer token.
//! - [`AuthError`] — all failure modes for the user-auth surface; maps to
//!   HTTP via the impl in this module.
//!
//! ## Module map
//!
//! - [`password`] — Argon2id hashing + verify.
//! - [`totp`] — RFC 6238 verification + provisioning URI.
//! - [`recovery_codes`] — 10-code batches, Argon2id-hashed, redeem-once.
//! - [`session`] — opaque token issuance / refresh-on-use.
//! - [`store`] — RocksDB persistence under the `user_auth` CF.
//!
//! ## Encryption of TOTP secrets at rest
//!
//! TOTP secrets are encrypted with the existing [`crate::encryption::FieldEncryptor`]
//! when `VELD_ENCRYPTION_KEY` is set. If the env var is unset:
//!   - in development mode, the secret is stored as raw bytes with a WARN
//!     log so dev loops aren't blocked;
//!   - in production mode (`VELD_ENV=production`), 2FA enrollment is
//!     refused entirely. There is no scenario where production stores a
//!     2FA secret in plaintext.

pub mod password;
pub mod recovery_codes;
pub mod runtime;
pub mod session;
pub mod store;
pub mod totp;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::errors::ErrorResponse;

/// Axum extension wrapper for the [`UserRecord`] resolved by the session
/// middleware. Newtype so the type uniquely identifies the request-scope
/// extension (distinct from any other `UserRecord` value passed by value).
#[derive(Clone, Debug)]
pub struct SessionUser(pub UserRecord);

/// Axum extension wrapper for the raw bearer token string. Used by the
/// `logout` handler to look up the session it must invalidate without
/// re-parsing the header.
#[derive(Clone, Debug)]
pub struct SessionTokenExt(pub String);

/// Env var that gates the entire user-auth surface (column family, routes,
/// middleware). Default: disabled.
pub const FEATURE_FLAG_ENV: &str = "VELD_USER_AUTH_ENABLED";

/// Returns `true` if `VELD_USER_AUTH_ENABLED` is set to a truthy value.
/// Truthy values: `1`, `true`, `yes`, `on` (case-insensitive).
pub fn feature_enabled() -> bool {
    std::env::var(FEATURE_FLAG_ENV)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// User role for authorization decisions.
///
/// `Admin` is required to register new users after the first user has been
/// created. There is no self-service promotion to `Admin` — once an Admin
/// exists, only an existing Admin can mint another. This is intentional
/// (see `docs/user-auth.md` § "Admin bootstrap").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserRole {
    Admin,
    User,
}

impl UserRole {
    /// Stringy form used in API responses and audit logs.
    pub fn as_str(&self) -> &'static str {
        match self {
            UserRole::Admin => "admin",
            UserRole::User => "user",
        }
    }
}

/// Persistent per-user state. Stored as bincode in the `user_auth` CF.
///
/// `password_hash` is an Argon2id PHC string. `totp_secret_encrypted` is
/// `None` until 2FA is enrolled; once enrolled, it holds either an
/// AES-256-GCM-encrypted blob (production / VELD_ENCRYPTION_KEY set) or
/// the raw HMAC secret bytes (development fallback with WARN log).
/// `recovery_code_hashes` holds one Argon2id PHC string per unused
/// recovery code; consumed codes are removed from the vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserRecord {
    pub id: Uuid,
    pub username: String,
    pub password_hash: String,
    pub totp_secret_encrypted: Option<Vec<u8>>,
    /// True iff 2FA enrollment is mid-flight (secret issued, first valid
    /// TOTP not yet confirmed). While pending, login does NOT require TOTP;
    /// once confirmed this flips to false and TOTP becomes mandatory.
    pub totp_enrollment_pending: bool,
    pub recovery_code_hashes: Vec<String>,
    pub role: UserRole,
    pub created_at: DateTime<Utc>,
    pub last_login_at: Option<DateTime<Utc>>,
}

impl UserRecord {
    /// Has the user completed 2FA enrollment (secret stored AND confirmed)?
    pub fn has_active_totp(&self) -> bool {
        self.totp_secret_encrypted.is_some() && !self.totp_enrollment_pending
    }
}

/// All errors raised by the user-auth surface.
#[derive(Debug)]
pub enum AuthError {
    /// Submitted credentials don't match (wrong password, wrong TOTP, or
    /// non-existent user — all surfaced as a single response to deny user
    /// enumeration).
    InvalidCredentials,
    /// 2FA is enrolled on this account and the request did not include a
    /// valid TOTP code.
    TotpRequired,
    /// Caller asked to enroll 2FA but it's already active.
    TotpAlreadyEnrolled,
    /// Caller tried to confirm 2FA but there is no pending enrollment.
    TotpNoPendingEnrollment,
    /// Production mode + missing `VELD_ENCRYPTION_KEY` blocks 2FA enrollment.
    TotpEncryptionRequired,
    /// Submitted recovery code doesn't match any stored hash, or all codes
    /// have been consumed.
    InvalidRecoveryCode,
    /// Bearer token is missing, malformed, or has expired.
    InvalidSession,
    /// Non-admin caller tried to perform an admin-only operation.
    Forbidden,
    /// Registration: this username already maps to a user.
    UsernameTaken,
    /// Password is empty / fails policy (today: non-empty only).
    WeakPassword(String),
    /// Per-username login throttle tripped.
    TooManyAttempts,
    /// Admin operation references an unknown username.
    UserNotFound,
    /// Demote was asked to remove the last remaining Admin — refused to
    /// avoid locking the deployment out of every admin-only endpoint.
    LastAdmin,
    /// Feature flag is off — request hit a user-auth route despite the
    /// router 404-ing the surface; only reachable in tests.
    Disabled,
    /// Anything internal: encoding failures, missing CFs, etc.
    Internal(String),
}

impl AuthError {
    pub fn weak_password(msg: impl Into<String>) -> Self {
        Self::WeakPassword(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidCredentials => "INVALID_CREDENTIALS",
            Self::TotpRequired => "TOTP_REQUIRED",
            Self::TotpAlreadyEnrolled => "TOTP_ALREADY_ENROLLED",
            Self::TotpNoPendingEnrollment => "TOTP_NO_PENDING_ENROLLMENT",
            Self::TotpEncryptionRequired => "TOTP_ENCRYPTION_REQUIRED",
            Self::InvalidRecoveryCode => "INVALID_RECOVERY_CODE",
            Self::InvalidSession => "INVALID_SESSION",
            Self::Forbidden => "FORBIDDEN",
            Self::UsernameTaken => "USERNAME_TAKEN",
            Self::WeakPassword(_) => "WEAK_PASSWORD",
            Self::TooManyAttempts => "TOO_MANY_ATTEMPTS",
            Self::UserNotFound => "USER_NOT_FOUND",
            Self::LastAdmin => "LAST_ADMIN",
            Self::Disabled => "USER_AUTH_DISABLED",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidCredentials
            | Self::TotpRequired
            | Self::InvalidRecoveryCode
            | Self::InvalidSession => StatusCode::UNAUTHORIZED,
            Self::TotpAlreadyEnrolled
            | Self::TotpNoPendingEnrollment
            | Self::TotpEncryptionRequired
            | Self::WeakPassword(_) => StatusCode::BAD_REQUEST,
            Self::UsernameTaken => StatusCode::CONFLICT,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::TooManyAttempts => StatusCode::TOO_MANY_REQUESTS,
            Self::UserNotFound => StatusCode::NOT_FOUND,
            Self::LastAdmin => StatusCode::CONFLICT,
            Self::Disabled => StatusCode::NOT_FOUND,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::InvalidCredentials => "Invalid username, password, or TOTP".to_string(),
            Self::TotpRequired => "TOTP code required for this account".to_string(),
            Self::TotpAlreadyEnrolled => "2FA already enrolled on this account".to_string(),
            Self::TotpNoPendingEnrollment => {
                "No pending 2FA enrollment; call /2fa/enroll first".to_string()
            }
            Self::TotpEncryptionRequired => {
                "2FA enrollment in production mode requires VELD_ENCRYPTION_KEY to be set"
                    .to_string()
            }
            Self::InvalidRecoveryCode => "Recovery code is invalid or already used".to_string(),
            Self::InvalidSession => "Session is missing, malformed, or has expired".to_string(),
            Self::Forbidden => "Operation requires the Admin role".to_string(),
            Self::UsernameTaken => "Username is already registered".to_string(),
            Self::WeakPassword(reason) => format!("Password rejected: {reason}"),
            Self::TooManyAttempts => "Too many login attempts; try again later".to_string(),
            Self::UserNotFound => "No user matches the supplied username".to_string(),
            Self::LastAdmin => {
                "Refusing to demote the last remaining administrator".to_string()
            }
            Self::Disabled => "User auth is disabled on this server".to_string(),
            Self::Internal(detail) => {
                tracing::error!(detail = %detail, "user_auth internal error");
                "Internal server error".to_string()
            }
        }
    }
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for AuthError {}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ErrorResponse {
            code: self.code().to_string(),
            message: self.message(),
            details: None,
            request_id: None,
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_flag_defaults_off() {
        // Save and restore VELD_USER_AUTH_ENABLED so this test plays nicely
        // with the suite. A nested guard would be overkill — set/remove
        // is cheap and the assertion is local.
        let prior = std::env::var(FEATURE_FLAG_ENV).ok();
        std::env::remove_var(FEATURE_FLAG_ENV);
        assert!(!feature_enabled());
        std::env::set_var(FEATURE_FLAG_ENV, "true");
        assert!(feature_enabled());
        std::env::set_var(FEATURE_FLAG_ENV, "1");
        assert!(feature_enabled());
        std::env::set_var(FEATURE_FLAG_ENV, "no");
        assert!(!feature_enabled());
        // Restore prior state so concurrent tests aren't affected.
        match prior {
            Some(v) => std::env::set_var(FEATURE_FLAG_ENV, v),
            None => std::env::remove_var(FEATURE_FLAG_ENV),
        }
    }

    #[test]
    fn role_serde_is_lowercase_snake() {
        let admin = serde_json::to_string(&UserRole::Admin).unwrap();
        assert_eq!(admin, "\"admin\"");
        let user = serde_json::to_string(&UserRole::User).unwrap();
        assert_eq!(user, "\"user\"");
    }

    #[test]
    fn auth_error_status_codes_match_meaning() {
        assert_eq!(
            AuthError::InvalidCredentials.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(AuthError::Forbidden.status_code(), StatusCode::FORBIDDEN);
        assert_eq!(
            AuthError::UsernameTaken.status_code(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            AuthError::TooManyAttempts.status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(AuthError::Disabled.status_code(), StatusCode::NOT_FOUND);
    }
}
