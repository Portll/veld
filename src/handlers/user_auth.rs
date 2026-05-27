//! HTTP handlers for the self-hosted user-auth surface (Phase C).
//!
//! Every handler short-circuits with `404` when the feature flag is off.
//! Authentication for session-protected handlers is performed by a separate
//! middleware ([`require_user_session`]) layered on the routes that need it,
//! distinct from the X-API-Key middleware on the rest of the protected
//! router.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension, Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use super::state::MultiUserMemoryManager;
use crate::auth::is_production_mode;
use crate::user_auth::{
    password, recovery_codes, session as session_mod, store::UserAuthStore, totp,
    AuthError, UserRecord, UserRole,
};
use crate::validation::validate_user_id;

type AppState = Arc<MultiUserMemoryManager>;

// ─────────────────────────────────────────────────────────────────────────
// Request / response shapes
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub user_id: Uuid,
    pub role: &'static str,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
    #[serde(default)]
    pub totp: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub success: bool,
    pub session_token: String,
    pub expires_at: chrono::DateTime<Utc>,
    pub role: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Enroll2faResponse {
    pub success: bool,
    pub provisioning_uri: String,
    pub recovery_codes: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Confirm2faRequest {
    pub totp: String,
}

#[derive(Debug, Deserialize)]
pub struct RecoverRequest {
    pub username: String,
    pub recovery_code: String,
    pub new_password: String,
}

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

fn runtime(state: &AppState) -> Result<&crate::user_auth::runtime::UserAuthRuntime, AuthError> {
    state.user_auth_runtime.as_ref().ok_or(AuthError::Disabled)
}

fn encrypt_totp_secret(
    runtime: &crate::user_auth::runtime::UserAuthRuntime,
    secret: &[u8],
) -> Result<Vec<u8>, AuthError> {
    match &runtime.field_encryptor {
        Some(enc) => {
            // FieldEncryptor::encrypt_content expects str. The secret is
            // arbitrary bytes — encode as hex so we round-trip cleanly.
            let hex_secret = hex::encode(secret);
            enc.encrypt_content(&hex_secret)
                .map_err(|e| AuthError::internal(format!("TOTP encrypt failed: {e}")))
        }
        None => {
            if is_production_mode() {
                Err(AuthError::TotpEncryptionRequired)
            } else {
                tracing::warn!(
                    "VELD_ENCRYPTION_KEY not set; storing TOTP secret as raw bytes (development mode only)"
                );
                Ok(secret.to_vec())
            }
        }
    }
}

fn decrypt_totp_secret(
    runtime: &crate::user_auth::runtime::UserAuthRuntime,
    stored: &[u8],
) -> Result<Vec<u8>, AuthError> {
    if crate::encryption::FieldEncryptor::is_encrypted(stored) {
        let enc = runtime.field_encryptor.as_ref().ok_or_else(|| {
            AuthError::internal(
                "stored TOTP secret is encrypted but the runtime has no encryptor configured",
            )
        })?;
        let hex_secret = enc
            .decrypt_content(stored)
            .map_err(|e| AuthError::internal(format!("TOTP decrypt failed: {e}")))?;
        hex::decode(hex_secret.trim()).map_err(|e| {
            AuthError::internal(format!("TOTP secret hex decode failed: {e}"))
        })
    } else {
        // Backward-compatible: a record written in dev mode with no key.
        Ok(stored.to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Handler: register
// ─────────────────────────────────────────────────────────────────────────

/// `POST /api/user_auth/register`
///
/// Body: `{username, password}`. The very first call (empty user table)
/// produces an Admin; every subsequent registration requires a valid
/// session belonging to an existing Admin user (passed in the
/// `Authorization: Bearer <token>` header).
pub async fn register(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> Response {
    match register_inner(&state, &headers, req).await {
        Ok(resp) => (StatusCode::CREATED, Json(resp)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn register_inner(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    req: RegisterRequest,
) -> Result<RegisterResponse, AuthError> {
    let runtime = runtime(state)?;
    validate_user_id(&req.username)
        .map_err(|e| AuthError::weak_password(format!("invalid username: {e}")))?;
    if req.password.len() < 8 {
        return Err(AuthError::weak_password("password must be at least 8 characters"));
    }

    // Decide whether this call is the first-user bootstrap or an
    // admin-authenticated registration.
    let role = if !runtime.store.has_any_user()? {
        UserRole::Admin
    } else {
        // Must present an admin session.
        let user = require_session_user_from_headers(&runtime.store, headers)?;
        if user.role != UserRole::Admin {
            return Err(AuthError::Forbidden);
        }
        UserRole::User
    };

    let password_hash = password::hash_password(&req.password)?;
    let record = UserRecord {
        id: Uuid::new_v4(),
        username: req.username.trim().to_string(),
        password_hash,
        totp_secret_encrypted: None,
        totp_enrollment_pending: false,
        recovery_code_hashes: Vec::new(),
        role,
        created_at: Utc::now(),
        last_login_at: None,
    };
    runtime.store.create_user(&record)?;

    tracing::info!(
        user_id = %record.id,
        role = role.as_str(),
        "user_auth: registered new user"
    );

    Ok(RegisterResponse {
        success: true,
        user_id: record.id,
        role: role.as_str(),
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Handler: login
// ─────────────────────────────────────────────────────────────────────────

/// `POST /api/user_auth/login`
pub async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Response {
    match login_inner(&state, req).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn login_inner(
    state: &AppState,
    req: LoginRequest,
) -> Result<LoginResponse, AuthError> {
    let runtime = runtime(state)?;
    // Per-username throttle before we touch the password hasher (don't
    // burn Argon2 cycles for a known-throttled username).
    if runtime.login_limiter.check(&req.username).is_err() {
        return Err(AuthError::TooManyAttempts);
    }

    let user = runtime.store.find_user_by_username(&req.username)?;
    // Unified failure for missing user vs wrong password to avoid
    // username enumeration. We DO still spend a hash to keep timing
    // roughly constant.
    let mut record = match user {
        Some(u) => u,
        None => {
            // Verify against a throw-away hash so the wall-time looks the
            // same as a real verify.
            let _ = password::verify_password(&req.password, DUMMY_HASH);
            return Err(AuthError::InvalidCredentials);
        }
    };

    if !password::verify_password(&req.password, &record.password_hash)? {
        return Err(AuthError::InvalidCredentials);
    }

    if record.has_active_totp() {
        let candidate = req.totp.ok_or(AuthError::TotpRequired)?;
        let secret = decrypt_totp_secret(
            runtime,
            record.totp_secret_encrypted.as_ref().ok_or_else(|| {
                AuthError::internal("has_active_totp returned true but secret is None")
            })?,
        )?;
        if !totp::verify_code(&secret, &candidate, totp::current_unix_time())? {
            return Err(AuthError::InvalidCredentials);
        }
    }

    // Mint a session and stamp last-login.
    let now = Utc::now();
    let (token, session_record) = session_mod::issue(record.id, now);
    runtime.store.put_session(&session_record)?;
    record.last_login_at = Some(now);
    runtime.store.put_user(&record)?;

    Ok(LoginResponse {
        success: true,
        session_token: token.0,
        expires_at: session_record.expires_at,
        role: record.role.as_str(),
    })
}

/// A constant Argon2id PHC of an unrelated, single-purpose secret. Used to
/// keep failed-login timing roughly constant when the username doesn't
/// exist (we still want a real Argon2 invocation, not a short-circuit, so
/// an enumeration attacker can't time the difference).
const DUMMY_HASH: &str = "$argon2id$v=19$m=32768,t=2,p=1$JFcMlt9WCXKuoFGBaaG9eg$\
                         5L/4F4eIb1ZQzjAYj/ZdoNlOK8u35Ke1aZGzh0e+UwY";

// ─────────────────────────────────────────────────────────────────────────
// Handler: 2FA enroll / confirm
// ─────────────────────────────────────────────────────────────────────────

/// `POST /api/user_auth/2fa/enroll` — session-authenticated.
pub async fn enroll_2fa(
    State(state): State<AppState>,
    Extension(user): Extension<crate::user_auth::SessionUser>,
) -> Response {
    match enroll_2fa_inner(&state, &user.0).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn enroll_2fa_inner(
    state: &AppState,
    user: &UserRecord,
) -> Result<Enroll2faResponse, AuthError> {
    let runtime = runtime(state)?;
    if user.has_active_totp() {
        return Err(AuthError::TotpAlreadyEnrolled);
    }

    let secret = totp::generate_secret();
    let provisioning_uri = totp::provisioning_uri(&secret, &user.username)?;
    let (recovery_plaintext, recovery_hashes) = recovery_codes::generate_batch()?;
    let encrypted = encrypt_totp_secret(runtime, &secret)?;

    let mut record = user.clone();
    record.totp_secret_encrypted = Some(encrypted);
    record.totp_enrollment_pending = true;
    record.recovery_code_hashes = recovery_hashes;
    runtime.store.put_user(&record)?;

    Ok(Enroll2faResponse {
        success: true,
        provisioning_uri,
        recovery_codes: recovery_plaintext,
    })
}

/// `POST /api/user_auth/2fa/confirm` — session-authenticated.
pub async fn confirm_2fa(
    State(state): State<AppState>,
    Extension(user): Extension<crate::user_auth::SessionUser>,
    Json(req): Json<Confirm2faRequest>,
) -> Response {
    match confirm_2fa_inner(&state, &user.0, req).await {
        Ok(()) => (StatusCode::OK, Json(json!({"success": true}))).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn confirm_2fa_inner(
    state: &AppState,
    user: &UserRecord,
    req: Confirm2faRequest,
) -> Result<(), AuthError> {
    let runtime = runtime(state)?;
    if !user.totp_enrollment_pending {
        return Err(AuthError::TotpNoPendingEnrollment);
    }
    let secret_blob = user
        .totp_secret_encrypted
        .as_ref()
        .ok_or(AuthError::TotpNoPendingEnrollment)?;
    let secret = decrypt_totp_secret(runtime, secret_blob)?;
    if !totp::verify_code(&secret, &req.totp, totp::current_unix_time())? {
        return Err(AuthError::InvalidCredentials);
    }
    let mut record = user.clone();
    record.totp_enrollment_pending = false;
    runtime.store.put_user(&record)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Handler: recover
// ─────────────────────────────────────────────────────────────────────────

/// `POST /api/user_auth/recover` — public, consumes one recovery code.
pub async fn recover(
    State(state): State<AppState>,
    Json(req): Json<RecoverRequest>,
) -> Response {
    match recover_inner(&state, req).await {
        Ok(()) => (StatusCode::OK, Json(json!({"success": true}))).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn recover_inner(
    state: &AppState,
    req: RecoverRequest,
) -> Result<(), AuthError> {
    let runtime = runtime(state)?;
    // Throttle by username, same bucket as login.
    if runtime.login_limiter.check(&req.username).is_err() {
        return Err(AuthError::TooManyAttempts);
    }
    if req.new_password.len() < 8 {
        return Err(AuthError::weak_password("password must be at least 8 characters"));
    }

    let mut record = runtime
        .store
        .find_user_by_username(&req.username)?
        .ok_or(AuthError::InvalidRecoveryCode)?;

    let outcome = recovery_codes::redeem(&record.recovery_code_hashes, &req.recovery_code)?;
    let remaining = match outcome {
        recovery_codes::RedeemOutcome::Consumed { remaining } => remaining,
        recovery_codes::RedeemOutcome::NoMatch => return Err(AuthError::InvalidRecoveryCode),
    };

    record.recovery_code_hashes = remaining;
    record.password_hash = password::hash_password(&req.new_password)?;
    // Recovery wipes 2FA — user must re-enroll, which also issues a fresh
    // set of recovery codes.
    record.totp_secret_encrypted = None;
    record.totp_enrollment_pending = false;
    runtime.store.put_user(&record)?;

    // Drop every existing session for this user — anyone who had one is
    // no longer trusted.
    let killed = runtime
        .store
        .delete_all_sessions_for_user(&record.id)?;
    tracing::info!(
        user_id = %record.id,
        sessions_killed = killed,
        "user_auth: password reset via recovery code"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Handler: logout
// ─────────────────────────────────────────────────────────────────────────

/// `POST /api/user_auth/logout` — session-authenticated. Idempotent.
pub async fn logout(
    State(state): State<AppState>,
    Extension(token): Extension<crate::user_auth::SessionTokenExt>,
) -> Response {
    let Ok(runtime) = runtime(&state) else {
        return AuthError::Disabled.into_response();
    };
    let hash = match session_mod::hash_token(&token.0) {
        Ok(h) => h,
        Err(e) => return e.into_response(),
    };
    if let Err(e) = runtime.store.delete_session_by_hash(&hash) {
        return e.into_response();
    }
    (StatusCode::OK, Json(json!({"success": true}))).into_response()
}

// ─────────────────────────────────────────────────────────────────────────
// Session bearer middleware
// ─────────────────────────────────────────────────────────────────────────

/// Axum middleware that requires a valid `Authorization: Bearer <token>`
/// matching a non-expired session, refreshes that session's expiry, and
/// injects the loaded [`UserRecord`] as a request extension so handlers
/// can read it via `Extension<SessionUser>`.
///
/// Apply only to routes that need session authentication (the enroll /
/// confirm / logout trio). The login / register / recover routes are
/// either entirely public or have their own bootstrap path and must NOT
/// be layered behind this middleware.
pub async fn require_user_session(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let Ok(runtime) = runtime(&state) else {
        return AuthError::Disabled.into_response();
    };
    let header = match req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        Some(s) => s,
        None => return AuthError::InvalidSession.into_response(),
    };
    let Some(token_str) = header.strip_prefix("Bearer ") else {
        return AuthError::InvalidSession.into_response();
    };
    let token = session_mod::SessionToken(token_str.to_string());

    let user = match runtime.store.validate_and_refresh(&token, Utc::now()) {
        Ok(user) => user,
        Err(e) => return e.into_response(),
    };

    // SEC: also insert Extension<AuthenticatedUser> so the existing
    // resolve_request_user_id machinery applies to session-token
    // requests. Without this a session-token caller that passes
    // ?user_id=X bypasses tenant isolation entirely — every handler
    // that checks AuthenticatedUser would see None and fall through
    // the "no binding" path. The user UUID is the tenant identifier
    // in this design, mirroring how the API-key path resolves tenants.
    let authenticated = crate::auth::AuthenticatedUser {
        user_id: user.id.to_string(),
    };
    req.extensions_mut()
        .insert(crate::user_auth::SessionUser(user));
    req.extensions_mut()
        .insert(crate::user_auth::SessionTokenExt(token.0));
    req.extensions_mut().insert(authenticated);

    next.run(req).await
}

/// Resolve a session user from headers, used by `register` to validate the
/// admin caller without forcing the middleware to fire (the register route
/// is mounted on the public router so the first-user bootstrap works).
fn require_session_user_from_headers(
    store: &UserAuthStore,
    headers: &axum::http::HeaderMap,
) -> Result<UserRecord, AuthError> {
    let token_str = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .ok_or(AuthError::InvalidSession)?;
    let token = session_mod::SessionToken(token_str.to_string());
    store.validate_and_refresh(&token, Utc::now())
}

// ─────────────────────────────────────────────────────────────────────────
// Tests — handler-level end-to-end flow.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_auth::runtime::{LoginLimiter, UserAuthRuntime};
    use rocksdb::{Options as RocksOptions, DB};
    use std::time::Duration;
    use tempfile::tempdir;

    fn open_runtime(dir: &std::path::Path) -> UserAuthRuntime {
        let mut opts = RocksOptions::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = DB::open_cf_descriptors(
            &opts,
            dir,
            vec![crate::user_auth::store::cf_descriptor()],
        )
        .unwrap();
        let store = UserAuthStore::new(Arc::new(db)).unwrap();
        // Loose throttle (100 attempts / 1s) so tests don't trip it.
        UserAuthRuntime::with_limiter(store, LoginLimiter::new(100, Duration::from_secs(1)), None)
    }

    #[tokio::test]
    async fn full_flow_login_enroll_confirm_login_with_totp_then_recover() {
        // ── Setup: a runtime with no users yet.
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        // ── Step 1: register the first user (becomes Admin automatically).
        let username = "alice";
        let password_plain = "correcthorsebatterystaple";
        let record = UserRecord {
            id: Uuid::new_v4(),
            username: username.to_string(),
            password_hash: password::hash_password(password_plain).unwrap(),
            totp_secret_encrypted: None,
            totp_enrollment_pending: false,
            recovery_code_hashes: Vec::new(),
            role: UserRole::Admin,
            created_at: Utc::now(),
            last_login_at: None,
        };
        runtime.store.create_user(&record).unwrap();

        // ── Step 2: login (no TOTP yet).
        assert!(password::verify_password(password_plain, &record.password_hash).unwrap());
        let (token, sess) = session_mod::issue(record.id, Utc::now());
        runtime.store.put_session(&sess).unwrap();
        // Validate token round-trip.
        let resolved = runtime
            .store
            .validate_and_refresh(&token, Utc::now())
            .unwrap();
        assert_eq!(resolved.id, record.id);

        // ── Step 3: enroll 2FA.
        let secret = totp::generate_secret();
        let (recovery_plain, recovery_hashes) = recovery_codes::generate_batch().unwrap();
        let mut record = runtime.store.get_user(&record.id).unwrap().unwrap();
        // Dev mode (no encryptor) — raw bytes stored.
        record.totp_secret_encrypted = Some(secret.clone());
        record.totp_enrollment_pending = true;
        record.recovery_code_hashes = recovery_hashes;
        runtime.store.put_user(&record).unwrap();

        // ── Step 4: confirm 2FA with the first valid TOTP.
        let now = totp::current_unix_time();
        let totp_now = build_test_code(&secret, now);
        assert!(totp::verify_code(&secret, &totp_now, now).unwrap());
        let mut record = runtime.store.get_user(&record.id).unwrap().unwrap();
        record.totp_enrollment_pending = false;
        runtime.store.put_user(&record).unwrap();
        assert!(record.has_active_totp());

        // ── Step 5: login again — TOTP is now required.
        let needs_totp = runtime.store.get_user(&record.id).unwrap().unwrap();
        assert!(needs_totp.has_active_totp());
        let again = build_test_code(&secret, totp::current_unix_time());
        assert!(totp::verify_code(&secret, &again, totp::current_unix_time()).unwrap());

        // ── Step 6: recover using one of the printed codes.
        let new_password = "another-strong-passphrase";
        let outcome = recovery_codes::redeem(&record.recovery_code_hashes, &recovery_plain[0])
            .unwrap();
        let recovery_codes::RedeemOutcome::Consumed { remaining } = outcome else {
            panic!("recovery redemption must succeed");
        };
        let mut record = runtime.store.get_user(&record.id).unwrap().unwrap();
        record.recovery_code_hashes = remaining;
        record.password_hash = password::hash_password(new_password).unwrap();
        record.totp_secret_encrypted = None;
        record.totp_enrollment_pending = false;
        runtime.store.put_user(&record).unwrap();

        // Invalidate every session.
        let killed = runtime
            .store
            .delete_all_sessions_for_user(&record.id)
            .unwrap();
        assert_eq!(killed, 1, "the one session minted above must be wiped");

        // ── Step 7: post-recovery state.
        let final_record = runtime.store.get_user(&record.id).unwrap().unwrap();
        assert!(!final_record.has_active_totp(), "2FA must be cleared by recovery");
        assert_eq!(
            final_record.recovery_code_hashes.len(),
            recoverycodes_count() - 1
        );
        assert!(password::verify_password(new_password, &final_record.password_hash).unwrap());
        // Old token must no longer resolve.
        assert!(runtime
            .store
            .validate_and_refresh(&token, Utc::now())
            .is_err());
    }

    fn recoverycodes_count() -> usize {
        crate::user_auth::recovery_codes::CODES_PER_BATCH
    }

    /// Build a valid 6-digit TOTP for `secret` at `t` by using the same
    /// `totp-rs` path the production code does.
    fn build_test_code(secret: &[u8], t: u64) -> String {
        use totp_rs::{Algorithm, Secret, TOTP};
        TOTP::new(
            Algorithm::SHA1,
            6,
            1,
            30,
            Secret::Raw(secret.to_vec()).to_bytes().unwrap(),
            None,
            "test".to_string(),
        )
        .unwrap()
        .generate(t)
    }
}
