//! HTTP handlers for the self-hosted user-auth surface (Phase C).
//!
//! When `VELD_USER_AUTH_ENABLED=false`, the router (see
//! `crate::handlers::router::build_protected_routes`) replaces the live
//! handlers with [`disabled_fallback`], which returns `503 Service
//! Unavailable` with the documented body
//! `{"error":"user_auth_disabled","detail":"Set VELD_USER_AUTH_ENABLED=true to enable"}`.
//! Handlers in this module are therefore only invoked when the feature is
//! on; the in-handler `AuthError::Disabled` branches remain as a defence
//! in depth for the case where the router mounted the live routes but the
//! runtime failed to construct (e.g. the column family is missing).
//!
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

    let password_hash = password::hash_password_async(&req.password).await?;
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
    let now = Utc::now();

    // First line of defence: in-memory governor (cheap, no I/O).
    if runtime.login_limiter.check(&req.username).is_err() {
        return Err(AuthError::TooManyAttempts);
    }
    // Second line: persistent throttle row — survives process restart so an
    // attacker can't unstick themselves by waiting for a redeploy. Read
    // BEFORE touching the password hasher (don't burn Argon2 cycles for a
    // known-locked username).
    let throttle = runtime.store.get_login_throttle(&req.username)?;
    if throttle.is_locked(now) {
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
            // same as a real verify. Use the async wrapper so the dummy
            // verify also passes through the Argon2 semaphore — otherwise
            // the username-enumeration timing channel widens whenever the
            // semaphore is saturated.
            let _ = password::verify_password_async(&req.password, DUMMY_HASH).await;
            // We deliberately do NOT increment the persistent throttle for
            // unknown usernames — that would let an attacker grief-lock
            // legitimate users out by spamming logins with their username
            // (the in-memory governor + global rate-limit already cap this).
            return Err(AuthError::InvalidCredentials);
        }
    };

    if !password::verify_password_async(&req.password, &record.password_hash).await? {
        // Real user, wrong password — count it.
        record_failed_login(runtime, &req.username, now)?;
        return Err(AuthError::InvalidCredentials);
    }

    if record.has_active_totp() {
        let candidate = match req.totp {
            Some(c) => c,
            None => {
                // TOTP missing — treat as failed credential attempt so
                // brute-forcing TOTP also walks the throttle.
                record_failed_login(runtime, &req.username, now)?;
                return Err(AuthError::TotpRequired);
            }
        };
        let secret = decrypt_totp_secret(
            runtime,
            record.totp_secret_encrypted.as_ref().ok_or_else(|| {
                AuthError::internal("has_active_totp returned true but secret is None")
            })?,
        )?;
        if !totp::verify_code(&secret, &candidate, totp::current_unix_time())? {
            record_failed_login(runtime, &req.username, now)?;
            return Err(AuthError::InvalidCredentials);
        }
    }

    // Mint a session and stamp last-login.
    let (token, session_record) = session_mod::issue(record.id, now);
    runtime.store.put_session(&session_record)?;
    record.last_login_at = Some(now);
    runtime.store.put_user(&record)?;
    // Clear the persistent throttle row — fresh start.
    runtime.store.clear_login_throttle(&req.username)?;

    Ok(LoginResponse {
        success: true,
        session_token: token.0,
        expires_at: session_record.expires_at,
        role: record.role.as_str(),
    })
}

/// Bump the persistent throttle row for `username`. If the failure pushed
/// the row over [`crate::user_auth::store::THROTTLE_FAILURE_THRESHOLD`] we
/// emit an audit-tagged warn so a SIEM can correlate lockouts.
fn record_failed_login(
    runtime: &crate::user_auth::runtime::UserAuthRuntime,
    username: &str,
    now: chrono::DateTime<Utc>,
) -> Result<(), AuthError> {
    let row = runtime.store.record_login_failure(username, now)?;
    if row.locked_until.map(|t| t > now).unwrap_or(false) {
        tracing::warn!(
            audit = "login_lockout",
            username = %username,
            failed_attempts = row.failed_attempts,
            locked_until = ?row.locked_until,
            "user_auth: persistent login throttle locked username"
        );
    }
    Ok(())
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
    let (recovery_plaintext, recovery_hashes) = recovery_codes::generate_batch_async().await?;
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

    let outcome =
        recovery_codes::redeem_async(&record.recovery_code_hashes, &req.recovery_code).await?;
    let remaining = match outcome {
        recovery_codes::RedeemOutcome::Consumed { remaining } => remaining,
        recovery_codes::RedeemOutcome::NoMatch => return Err(AuthError::InvalidRecoveryCode),
    };

    record.recovery_code_hashes = remaining;
    record.password_hash = password::hash_password_async(&req.new_password).await?;
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
// Disabled-feature fallback (B7)
// ─────────────────────────────────────────────────────────────────────────

/// Catch-all handler mounted under `/api/user_auth/{*path}` when the
/// feature flag is **off** (`VELD_USER_AUTH_ENABLED=false`).
///
/// Returns `503 Service Unavailable` with a fixed JSON body shape so
/// clients probing the surface can distinguish "endpoint exists but
/// disabled" from "no such endpoint, are you on the right server?".
/// The body is deliberately a different shape from the live error
/// surface — the live surface uses [`crate::errors::ErrorResponse`]
/// (`code` / `message` / `details`), whereas the disabled surface uses
/// `{"error": "user_auth_disabled", "detail": "…"}` so an operator
/// reading logs / clients can pattern-match on the marker without
/// pulling in the full auth-error vocabulary.
pub async fn disabled_fallback() -> Response {
    let body = json!({
        "error": "user_auth_disabled",
        "detail": "Set VELD_USER_AUTH_ENABLED=true to enable",
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
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
// Admin-only: promote / demote
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct UsernameRequest {
    pub username: String,
}

/// `POST /api/user_auth/admin/promote` — admin-only.
///
/// Body: `{username}`. Sets the target user's role to `Admin`. Idempotent —
/// promoting an already-admin user is a no-op success.
pub async fn admin_promote(
    State(state): State<AppState>,
    Extension(caller): Extension<crate::user_auth::SessionUser>,
    Json(req): Json<UsernameRequest>,
) -> Response {
    let result = runtime(&state)
        .and_then(|rt| admin_set_role_with_runtime(rt, &caller.0, &req.username, UserRole::Admin));
    match result {
        Ok(()) => (StatusCode::OK, Json(json!({"success": true}))).into_response(),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/user_auth/admin/demote` — admin-only.
///
/// Body: `{username}`. Sets the target user's role to `User`. Refuses with
/// `409 LAST_ADMIN` if the demotion would leave zero administrators.
pub async fn admin_demote(
    State(state): State<AppState>,
    Extension(caller): Extension<crate::user_auth::SessionUser>,
    Json(req): Json<UsernameRequest>,
) -> Response {
    let result = runtime(&state)
        .and_then(|rt| admin_set_role_with_runtime(rt, &caller.0, &req.username, UserRole::User));
    match result {
        Ok(()) => (StatusCode::OK, Json(json!({"success": true}))).into_response(),
        Err(e) => e.into_response(),
    }
}

/// Inner role-change logic — broken out from the handlers so unit tests
/// can drive it with a freestanding [`crate::user_auth::runtime::UserAuthRuntime`]
/// without instantiating the full [`super::state::MultiUserMemoryManager`].
fn admin_set_role_with_runtime(
    runtime: &crate::user_auth::runtime::UserAuthRuntime,
    caller: &UserRecord,
    target_username: &str,
    new_role: UserRole,
) -> Result<(), AuthError> {
    if caller.role != UserRole::Admin {
        return Err(AuthError::Forbidden);
    }

    let mut target = runtime
        .store
        .find_user_by_username(target_username)?
        .ok_or(AuthError::UserNotFound)?;

    if target.role == new_role {
        // Idempotent. No write, no audit (nothing changed).
        return Ok(());
    }

    // Demote guard: if the target is currently Admin and we're moving them
    // to User, count remaining admins and refuse to zero out the set.
    if target.role == UserRole::Admin && new_role == UserRole::User {
        let admins = runtime.store.count_admins()?;
        if admins <= 1 {
            return Err(AuthError::LastAdmin);
        }
    }

    target.role = new_role;
    runtime.store.put_user(&target)?;

    let audit_action = match new_role {
        UserRole::Admin => "admin_promotion",
        UserRole::User => "admin_demotion",
    };
    tracing::warn!(
        audit = audit_action,
        promoter = %caller.id,
        target = %target.username,
        target_user_id = %target.id,
        "user_auth: role change"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Session revocation
// ─────────────────────────────────────────────────────────────────────────

/// `POST /api/user_auth/sessions/revoke_all` — admin-only.
///
/// Body: `{username}`. Deletes every active session for the named user.
/// Returns `{success: true, removed: <count>}`.
pub async fn revoke_all_sessions(
    State(state): State<AppState>,
    Extension(caller): Extension<crate::user_auth::SessionUser>,
    Json(req): Json<UsernameRequest>,
) -> Response {
    let result =
        runtime(&state).and_then(|rt| revoke_all_sessions_with_runtime(rt, &caller.0, &req.username));
    match result {
        Ok(removed) => (
            StatusCode::OK,
            Json(json!({"success": true, "removed": removed})),
        )
            .into_response(),
        Err(e) => e.into_response(),
    }
}

fn revoke_all_sessions_with_runtime(
    runtime: &crate::user_auth::runtime::UserAuthRuntime,
    caller: &UserRecord,
    target_username: &str,
) -> Result<usize, AuthError> {
    if caller.role != UserRole::Admin {
        return Err(AuthError::Forbidden);
    }
    let target = runtime
        .store
        .find_user_by_username(target_username)?
        .ok_or(AuthError::UserNotFound)?;
    let removed = runtime.store.delete_all_sessions_for_user(&target.id)?;
    tracing::warn!(
        audit = "session_revoke_all",
        admin = %caller.id,
        target = %target.username,
        target_user_id = %target.id,
        removed = removed,
        "user_auth: admin revoked all sessions for user"
    );
    Ok(removed)
}

/// `POST /api/user_auth/sessions/revoke_mine` — session-authenticated.
///
/// Deletes every session for the caller EXCEPT the one making this request
/// — i.e. "log out all other devices" from the active session.
pub async fn revoke_my_other_sessions(
    State(state): State<AppState>,
    Extension(caller): Extension<crate::user_auth::SessionUser>,
    Extension(token): Extension<crate::user_auth::SessionTokenExt>,
) -> Response {
    let result = runtime(&state)
        .and_then(|rt| revoke_my_other_sessions_with_runtime(rt, &caller.0, &token.0));
    match result {
        Ok(removed) => (
            StatusCode::OK,
            Json(json!({"success": true, "removed": removed})),
        )
            .into_response(),
        Err(e) => e.into_response(),
    }
}

fn revoke_my_other_sessions_with_runtime(
    runtime: &crate::user_auth::runtime::UserAuthRuntime,
    caller: &UserRecord,
    caller_token: &str,
) -> Result<usize, AuthError> {
    let keep_hash = session_mod::hash_token(caller_token)?;
    let removed = runtime
        .store
        .delete_other_sessions_for_user(&caller.id, &keep_hash)?;
    tracing::warn!(
        audit = "session_revoke_mine",
        user_id = %caller.id,
        username = %caller.username,
        removed = removed,
        "user_auth: user revoked their other sessions"
    );
    Ok(removed)
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

    // ─────────────────────────────────────────────────────────────────────
    // B7: disabled-feature fallback returns 503 with the documented body
    // ─────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn disabled_fallback_returns_503_with_documented_body_shape() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Method, Request};
        use axum::routing::any;
        use axum::Router;
        use tower::ServiceExt;

        let app: Router =
            Router::new().route("/api/user_auth/{*path}", any(disabled_fallback));

        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/user_auth/login")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"], "user_auth_disabled");
        assert_eq!(
            body["detail"],
            "Set VELD_USER_AUTH_ENABLED=true to enable"
        );

        let req = Request::builder()
            .method(Method::GET)
            .uri("/api/user_auth/2fa/enroll")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    // ── Helpers for admin / revoke / throttle tests ─────────────────────

    fn fixture_user(name: &str, role: UserRole) -> UserRecord {
        UserRecord {
            id: Uuid::new_v4(),
            username: name.to_string(),
            password_hash: password::hash_password("temp-pass-12345").unwrap(),
            totp_secret_encrypted: None,
            totp_enrollment_pending: false,
            recovery_code_hashes: Vec::new(),
            role,
            created_at: Utc::now(),
            last_login_at: None,
        }
    }

    // ── A) Admin promote / demote ───────────────────────────────────────

    #[test]
    fn admin_can_promote_user_to_admin() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        let admin = fixture_user("root", UserRole::Admin);
        let bob = fixture_user("bob", UserRole::User);
        runtime.store.create_user(&admin).unwrap();
        runtime.store.create_user(&bob).unwrap();

        admin_set_role_with_runtime(&runtime, &admin, "bob", UserRole::Admin).unwrap();

        let bob_after = runtime.store.find_user_by_username("bob").unwrap().unwrap();
        assert_eq!(bob_after.role, UserRole::Admin);
    }

    #[test]
    fn non_admin_promote_attempt_is_forbidden() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        let alice = fixture_user("alice", UserRole::User);
        let bob = fixture_user("bob", UserRole::User);
        runtime.store.create_user(&alice).unwrap();
        runtime.store.create_user(&bob).unwrap();

        let err = admin_set_role_with_runtime(&runtime, &alice, "bob", UserRole::Admin)
            .unwrap_err();
        assert!(matches!(err, AuthError::Forbidden), "{err:?}");
        // bob's role unchanged.
        let bob_after = runtime.store.find_user_by_username("bob").unwrap().unwrap();
        assert_eq!(bob_after.role, UserRole::User);
    }

    #[test]
    fn promote_unknown_username_returns_user_not_found() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());
        let admin = fixture_user("root", UserRole::Admin);
        runtime.store.create_user(&admin).unwrap();

        let err = admin_set_role_with_runtime(&runtime, &admin, "nobody", UserRole::Admin)
            .unwrap_err();
        assert!(matches!(err, AuthError::UserNotFound), "{err:?}");
    }

    #[test]
    fn demoting_only_admin_returns_last_admin_conflict() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        let admin = fixture_user("root", UserRole::Admin);
        let bob = fixture_user("bob", UserRole::User);
        runtime.store.create_user(&admin).unwrap();
        runtime.store.create_user(&bob).unwrap();

        // An admin self-demoting themselves while they're the only admin
        // must be refused.
        let err = admin_set_role_with_runtime(&runtime, &admin, "root", UserRole::User)
            .unwrap_err();
        assert!(matches!(err, AuthError::LastAdmin), "{err:?}");
        assert_eq!(err.status_code(), StatusCode::CONFLICT);
        assert_eq!(err.code(), "LAST_ADMIN");

        // root still has Admin role.
        let still = runtime.store.find_user_by_username("root").unwrap().unwrap();
        assert_eq!(still.role, UserRole::Admin);
    }

    #[test]
    fn demote_second_admin_succeeds() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        let admin = fixture_user("root", UserRole::Admin);
        let admin2 = fixture_user("root2", UserRole::Admin);
        runtime.store.create_user(&admin).unwrap();
        runtime.store.create_user(&admin2).unwrap();

        // Two admins — demoting root2 is allowed.
        admin_set_role_with_runtime(&runtime, &admin, "root2", UserRole::User).unwrap();
        let after = runtime
            .store
            .find_user_by_username("root2")
            .unwrap()
            .unwrap();
        assert_eq!(after.role, UserRole::User);

        // Now there's exactly one admin again — demoting root must fail.
        let err = admin_set_role_with_runtime(&runtime, &admin, "root", UserRole::User)
            .unwrap_err();
        assert!(matches!(err, AuthError::LastAdmin), "{err:?}");
    }

    // ── B) Persistent login throttle ────────────────────────────────────

    #[test]
    fn persistent_throttle_survives_runtime_drop_and_reopen() {
        // This is the spec acceptance test for deliverable B: 5 failed
        // logins, drop+reopen the store, the 6th attempt is still locked.
        let dir = tempdir().unwrap();
        let now = Utc::now();
        {
            let runtime = open_runtime(dir.path());
            // Sentinel real user so failures actually walk the throttle row.
            let bob = UserRecord {
                id: Uuid::new_v4(),
                username: "bob".to_string(),
                password_hash: password::hash_password("the-real-password-321").unwrap(),
                totp_secret_encrypted: None,
                totp_enrollment_pending: false,
                recovery_code_hashes: Vec::new(),
                role: UserRole::User,
                created_at: Utc::now(),
                last_login_at: None,
            };
            runtime.store.create_user(&bob).unwrap();

            for _ in 0..crate::user_auth::store::THROTTLE_FAILURE_THRESHOLD {
                record_failed_login(&runtime, "bob", now).unwrap();
            }
            let row = runtime.store.get_login_throttle("bob").unwrap();
            assert!(row.is_locked(now), "must lock at threshold");
        }
        // Process restart simulation: drop the store, reopen the same dir.
        let runtime = open_runtime(dir.path());
        let row = runtime.store.get_login_throttle("bob").unwrap();
        assert!(
            row.is_locked(now),
            "persistent lock must survive store reopen"
        );
    }

    // ── C) Session revocation ───────────────────────────────────────────

    #[test]
    fn admin_revoke_all_wipes_every_session() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        let admin = fixture_user("root", UserRole::Admin);
        let bob = fixture_user("bob", UserRole::User);
        runtime.store.create_user(&admin).unwrap();
        runtime.store.create_user(&bob).unwrap();

        let now = Utc::now();
        let (token_a, sess_a) = session_mod::issue(bob.id, now);
        let (token_b, sess_b) = session_mod::issue(bob.id, now);
        runtime.store.put_session(&sess_a).unwrap();
        runtime.store.put_session(&sess_b).unwrap();

        // Admin yanks every bob session.
        let removed = runtime.store.delete_all_sessions_for_user(&bob.id).unwrap();
        assert_eq!(removed, 2);
        // Both tokens are now dead.
        assert!(runtime
            .store
            .validate_and_refresh(&token_a, now)
            .is_err());
        assert!(runtime
            .store
            .validate_and_refresh(&token_b, now)
            .is_err());
    }

    #[test]
    fn revoke_mine_keeps_caller_token_only() {
        let dir = tempdir().unwrap();
        let runtime = open_runtime(dir.path());

        let bob = fixture_user("bob", UserRole::User);
        runtime.store.create_user(&bob).unwrap();

        let now = Utc::now();
        // token_a is the active call's bearer; token_b is the other device.
        let (token_a, sess_a) = session_mod::issue(bob.id, now);
        let (token_b, sess_b) = session_mod::issue(bob.id, now);
        runtime.store.put_session(&sess_a).unwrap();
        runtime.store.put_session(&sess_b).unwrap();

        let keep_hash = session_mod::hash_token(&token_a.0).unwrap();
        let removed = runtime
            .store
            .delete_other_sessions_for_user(&bob.id, &keep_hash)
            .unwrap();
        assert_eq!(removed, 1);

        // token_a still resolves.
        let still = runtime.store.validate_and_refresh(&token_a, now).unwrap();
        assert_eq!(still.id, bob.id);
        // token_b is gone.
        assert!(runtime
            .store
            .validate_and_refresh(&token_b, now)
            .is_err());
    }
}
