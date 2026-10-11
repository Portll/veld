//! Administering logins: users, their sessions and 2FA, the audit trail, and the
//! step-up check every change goes through.
//!
//! Reads need an Admin session. A change also needs step-up: a current TOTP code in
//! `X-Veld-Step-Up` from a user with 2FA enrolled. A code is accepted once, so a
//! captured request cannot be replayed inside the code's window, and it elevates
//! that one session for `STEP_UP_WINDOW`, so a run of changes asks once. Key
//! management (`handlers::keys`) goes through the same check.

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::state::MultiUserMemoryManager;
use super::user_auth::decrypt_totp_secret;
use crate::user_auth::{
    password, recovery_codes, runtime::UserAuthRuntime, session as session_mod, totp, AuditEvent,
    AuthError, SessionTokenExt, SessionUser, UserRecord, UserRole,
};

type AppState = Arc<MultiUserMemoryManager>;

pub const STEP_UP_HEADER: &str = "x-veld-step-up";

fn runtime(state: &AppState) -> Result<&UserAuthRuntime, AuthError> {
    state.user_auth_runtime.as_ref().ok_or(AuthError::Disabled)
}

fn respond<T: Serialize>(result: Result<T, AuthError>) -> Response {
    match result {
        Ok(body) => (StatusCode::OK, Json(body)).into_response(),
        Err(e) => e.into_response(),
    }
}

pub(crate) fn require_admin(caller: &UserRecord) -> Result<(), AuthError> {
    if caller.role == UserRole::Admin {
        Ok(())
    } else {
        Err(AuthError::Forbidden)
    }
}

/// Step-up for a request from `caller` on `session`: an elevated session passes; a
/// fresh code elevates it. A wrong code counts against the caller's own step-up
/// budget, so a stolen session cannot guess codes freely.
pub(crate) fn require_step_up(
    runtime: &UserAuthRuntime,
    caller: &UserRecord,
    session: &str,
    headers: &HeaderMap,
    now_unix: u64,
) -> Result<(), AuthError> {
    if !caller.has_active_totp() {
        return Err(AuthError::StepUpUnavailable);
    }
    let session = session_mod::hash_token(session)?;
    let presented = headers
        .get(STEP_UP_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|c| !c.trim().is_empty());
    if !presented && runtime.elevations.is_elevated(&session) {
        return Ok(());
    }
    let code = headers
        .get(STEP_UP_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .ok_or(AuthError::StepUpRequired)?;
    let budget = format!("step-up:{}", caller.id);
    if runtime.login_limiter.peek(&budget).is_err() {
        return Err(AuthError::TooManyAttempts);
    }
    let blob = caller
        .totp_secret_encrypted
        .as_ref()
        .ok_or_else(|| AuthError::internal("active TOTP without a secret"))?;
    let secret = decrypt_totp_secret(runtime, caller.id.as_bytes(), blob)?;
    let Some(step) = totp::matching_step(&secret, code, now_unix)? else {
        let _ = runtime.login_limiter.check(&budget);
        return Err(AuthError::StepUpRequired);
    };
    if runtime
        .store
        .last_step_up(&caller.id)?
        .is_some_and(|last| step <= last)
    {
        return Err(AuthError::StepUpReplayed);
    }
    runtime.store.put_last_step_up(&caller.id, step)?;
    runtime.elevations.elevate(&session);
    Ok(())
}

/// Admin, then step-up: the gate on every administrative change.
pub(crate) fn require_admin_step_up(
    runtime: &UserAuthRuntime,
    caller: &UserRecord,
    session: &str,
    headers: &HeaderMap,
) -> Result<(), AuthError> {
    require_admin(caller)?;
    require_step_up(runtime, caller, session, headers, totp::current_unix_time())
}

pub(crate) fn audit(
    runtime: &UserAuthRuntime,
    actor: &UserRecord,
    action: &str,
    target: Option<&str>,
    detail: serde_json::Value,
) -> Result<(), AuthError> {
    tracing::warn!(audit = action, actor = %actor.username, target = ?target, "user_auth: admin action");
    runtime.store.append_audit(&AuditEvent {
        at: Utc::now(),
        actor: actor.username.clone(),
        action: action.to_string(),
        target: target.map(str::to_string),
        detail: detail.to_string(),
    })
}

fn target_user(runtime: &UserAuthRuntime, username: &str) -> Result<UserRecord, AuthError> {
    runtime
        .store
        .find_user_by_username(username)?
        .ok_or(AuthError::UserNotFound)
}

/// Refuse an admin action aimed at the caller's own account: those go through the
/// self-service endpoints, which ask for the current password.
fn not_self(caller: &UserRecord, target: &UserRecord) -> Result<(), AuthError> {
    if caller.id == target.id {
        Err(AuthError::SelfAdministration)
    } else {
        Ok(())
    }
}

// ── Users ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct UserView {
    pub id: String,
    pub username: String,
    pub role: &'static str,
    pub created_at: DateTime<Utc>,
    pub last_login_at: Option<DateTime<Utc>>,
    /// `active`, `pending` or `none`.
    pub totp: &'static str,
    pub recovery_codes_remaining: usize,
    pub disabled_at: Option<DateTime<Utc>>,
    pub locked_until: Option<DateTime<Utc>>,
    pub failed_attempts: u32,
    pub sessions: usize,
}

fn view(runtime: &UserAuthRuntime, user: &UserRecord) -> Result<UserView, AuthError> {
    let now = Utc::now();
    let throttle = runtime.store.get_login_throttle(&user.username)?;
    Ok(UserView {
        id: user.id.to_string(),
        username: user.username.clone(),
        role: user.role.as_str(),
        created_at: user.created_at,
        last_login_at: user.last_login_at,
        totp: match (&user.totp_secret_encrypted, user.totp_enrollment_pending) {
            (Some(_), false) => "active",
            (Some(_), true) => "pending",
            (None, _) => "none",
        },
        recovery_codes_remaining: user.recovery_code_hashes.len(),
        disabled_at: user.disabled_at,
        locked_until: throttle.locked_until.filter(|t| *t > now),
        failed_attempts: throttle.failed_attempts,
        sessions: runtime.store.list_sessions_for_user(&user.id, now)?.len(),
    })
}

/// `GET /api/user_auth/admin/users`
pub async fn list_users(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin(&caller.0)?;
        rt.store
            .list_users()?
            .iter()
            .map(|u| view(rt, u))
            .collect::<Result<Vec<_>, _>>()
    }))
}

#[derive(Debug, Deserialize)]
pub struct UsernameBody {
    pub username: String,
}

pub(crate) fn set_disabled_with_runtime(
    runtime: &UserAuthRuntime,
    caller: &UserRecord,
    username: &str,
    disabled: bool,
) -> Result<UserView, AuthError> {
    let mut target = target_user(runtime, username)?;
    not_self(caller, &target)?;
    if disabled == target.disabled_at.is_some() {
        return view(runtime, &target);
    }
    target.disabled_at = disabled.then(Utc::now);
    runtime.store.put_user(&target)?;
    let killed = if disabled {
        runtime.store.delete_all_sessions_for_user(&target.id)?
    } else {
        0
    };
    audit(
        runtime,
        caller,
        if disabled {
            "user_disable"
        } else {
            "user_enable"
        },
        Some(&target.username),
        json!({ "sessions_revoked": killed }),
    )?;
    view(runtime, &target)
}

/// `POST /api/user_auth/admin/users/disable` — step-up. Revokes every session.
pub async fn disable_user(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<UsernameBody>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin_step_up(rt, &caller.0, &token.0, &headers)?;
        set_disabled_with_runtime(rt, &caller.0, &body.username, true)
    }))
}

/// `POST /api/user_auth/admin/users/enable` — step-up.
pub async fn enable_user(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<UsernameBody>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin_step_up(rt, &caller.0, &token.0, &headers)?;
        set_disabled_with_runtime(rt, &caller.0, &body.username, false)
    }))
}

/// A temporary password: 24 characters of base64url, 144 bits.
fn temporary_password() -> String {
    use base64::Engine;
    use rand::RngCore;
    let mut raw = [0u8; 18];
    rand::rngs::OsRng.fill_bytes(&mut raw);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw)
}

/// `POST /api/user_auth/admin/users/reset_password` — step-up. Returns a temporary
/// password once and revokes every session of the user.
pub async fn reset_password(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<UsernameBody>,
) -> Response {
    let result = async {
        let rt = runtime(&state)?;
        require_admin_step_up(rt, &caller.0, &token.0, &headers)?;
        let mut target = target_user(rt, &body.username)?;
        not_self(&caller.0, &target)?;
        let temporary = temporary_password();
        target.password_hash = password::hash_password_async(&temporary).await?;
        rt.store.put_user(&target)?;
        let killed = rt.store.delete_all_sessions_for_user(&target.id)?;
        rt.store.clear_login_throttle(&target.username)?;
        rt.login_limiter.reset(&target.username);
        audit(
            rt,
            &caller.0,
            "password_reset_by_admin",
            Some(&target.username),
            json!({ "sessions_revoked": killed }),
        )?;
        Ok(json!({ "username": target.username, "temporary_password": temporary }))
    }
    .await;
    respond(result)
}

pub(crate) fn reset_2fa_with_runtime(
    runtime: &UserAuthRuntime,
    caller: &UserRecord,
    username: &str,
) -> Result<UserView, AuthError> {
    let mut target = target_user(runtime, username)?;
    not_self(caller, &target)?;
    target.totp_secret_encrypted = None;
    target.totp_enrollment_pending = false;
    target.recovery_code_hashes.clear();
    runtime.store.put_user(&target)?;
    let killed = runtime.store.delete_all_sessions_for_user(&target.id)?;
    audit(
        runtime,
        caller,
        "totp_reset_by_admin",
        Some(&target.username),
        json!({ "sessions_revoked": killed }),
    )?;
    view(runtime, &target)
}

/// `POST /api/user_auth/admin/users/reset_2fa` — step-up. Clears 2FA and recovery
/// codes; the user signs in with the password alone and enrols again.
pub async fn reset_2fa(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<UsernameBody>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin_step_up(rt, &caller.0, &token.0, &headers)?;
        reset_2fa_with_runtime(rt, &caller.0, &body.username)
    }))
}

/// `POST /api/user_auth/admin/users/unlock` — step-up. Clears the login lockout.
pub async fn unlock_user(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<UsernameBody>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin_step_up(rt, &caller.0, &token.0, &headers)?;
        let target = target_user(rt, &body.username)?;
        rt.store.clear_login_throttle(&target.username)?;
        rt.login_limiter.reset(&target.username);
        audit(
            rt,
            &caller.0,
            "login_unlock",
            Some(&target.username),
            json!({}),
        )?;
        view(rt, &target)
    }))
}

// ── Sessions ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SessionView {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub current: bool,
}

fn session_views(
    runtime: &UserAuthRuntime,
    user: &UserRecord,
    current: Option<&str>,
) -> Result<Vec<SessionView>, AuthError> {
    let current_id = current
        .map(session_mod::hash_token)
        .transpose()?
        .map(|hash| session_mod::public_id(&hash));
    Ok(runtime
        .store
        .list_sessions_for_user(&user.id, Utc::now())?
        .into_iter()
        .map(|s| SessionView {
            current: current_id.as_deref() == Some(s.public_id().as_str()),
            id: s.public_id(),
            created_at: s.created_at,
            expires_at: s.expires_at,
        })
        .collect())
}

#[derive(Debug, Deserialize)]
pub struct SessionsQuery {
    pub username: String,
}

/// `GET /api/user_auth/admin/sessions?username=`
pub async fn list_sessions(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    Query(query): Query<SessionsQuery>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin(&caller.0)?;
        let target = target_user(rt, &query.username)?;
        session_views(rt, &target, Some(&token.0))
    }))
}

#[derive(Debug, Deserialize)]
pub struct RevokeSessionBody {
    pub username: String,
    pub session_id: String,
}

/// `POST /api/user_auth/admin/sessions/revoke` — step-up.
pub async fn revoke_session(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<RevokeSessionBody>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin_step_up(rt, &caller.0, &token.0, &headers)?;
        let target = target_user(rt, &body.username)?;
        if !rt
            .store
            .delete_session_by_public_id(&target.id, &body.session_id)?
        {
            return Err(AuthError::SessionNotFound);
        }
        audit(
            rt,
            &caller.0,
            "session_revoke",
            Some(&target.username),
            json!({ "session": body.session_id }),
        )?;
        Ok(json!({ "revoked": body.session_id }))
    }))
}

// ── Audit ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    pub limit: Option<usize>,
}

/// `GET /api/user_auth/admin/audit?limit=` — newest first, at most 1000.
pub async fn read_audit(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Query(query): Query<AuditQuery>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        require_admin(&caller.0)?;
        rt.store
            .recent_audit(query.limit.unwrap_or(100).clamp(1, 1000))
    }))
}

// ── Self-service ────────────────────────────────────────────────────────

/// `GET /api/user_auth/me`
pub async fn me(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        Ok(json!({
            "user": view(rt, &caller.0)?,
            "sessions": session_views(rt, &caller.0, Some(&token.0))?,
        }))
    }))
}

#[derive(Deserialize)]
pub struct ChangePasswordBody {
    pub current_password: String,
    pub new_password: String,
}

/// `POST /api/user_auth/password` — the current password, and step-up when 2FA is
/// enrolled. Every other session of the caller is revoked.
pub async fn change_password(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<ChangePasswordBody>,
) -> Response {
    let result = async {
        let rt = runtime(&state)?;
        let mut user = caller.0.clone();
        if body.new_password.len() < 8 {
            return Err(AuthError::weak_password(
                "password must be at least 8 characters",
            ));
        }
        if rt.login_limiter.check(&user.username).is_err() {
            return Err(AuthError::TooManyAttempts);
        }
        if !password::verify_password_async(&body.current_password, &user.password_hash).await? {
            return Err(AuthError::InvalidCredentials);
        }
        if user.has_active_totp() {
            require_step_up(rt, &user, &token.0, &headers, totp::current_unix_time())?;
        }
        user.password_hash = password::hash_password_async(&body.new_password).await?;
        rt.store.put_user(&user)?;
        let keep = session_mod::hash_token(&token.0)?;
        let killed = rt.store.delete_other_sessions_for_user(&user.id, &keep)?;
        audit(
            rt,
            &user,
            "password_change",
            Some(&user.username),
            json!({ "sessions_revoked": killed }),
        )?;
        Ok(json!({ "success": true, "sessions_revoked": killed }))
    }
    .await;
    respond(result)
}

/// `POST /api/user_auth/recovery_codes` — step-up. A fresh batch of ten codes,
/// shown once; the old batch stops working.
pub async fn regenerate_recovery_codes(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
) -> Response {
    let result = async {
        let rt = runtime(&state)?;
        require_step_up(rt, &caller.0, &token.0, &headers, totp::current_unix_time())?;
        let (codes, hashes) = recovery_codes::generate_batch_async().await?;
        let mut user = caller.0.clone();
        user.recovery_code_hashes = hashes;
        rt.store.put_user(&user)?;
        audit(
            rt,
            &user,
            "recovery_codes_regenerated",
            Some(&user.username),
            json!({}),
        )?;
        Ok(json!({ "recovery_codes": codes }))
    }
    .await;
    respond(result)
}

#[derive(Debug, Deserialize)]
pub struct OwnSessionBody {
    pub session_id: String,
}

/// `POST /api/user_auth/sessions/revoke` — one of the caller's own sessions.
pub async fn revoke_own_session(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    Json(body): Json<OwnSessionBody>,
) -> Response {
    respond(runtime(&state).and_then(|rt| {
        if !rt
            .store
            .delete_session_by_public_id(&caller.0.id, &body.session_id)?
        {
            return Err(AuthError::SessionNotFound);
        }
        Ok(json!({ "revoked": body.session_id }))
    }))
}

// These tests open the auth store on a RocksDB handle; a redb-only build has none to give them.
#[cfg(all(test, feature = "rocksdb"))]
mod tests {
    use super::*;
    use crate::user_auth::runtime::LoginLimiter;
    use crate::user_auth::store::UserAuthStore;
    use rocksdb::{Options as RocksOptions, DB};
    use std::time::Duration;
    use tempfile::tempdir;
    use uuid::Uuid;

    const SECRET: &[u8] = b"step-up-secret-twenty";

    fn open_runtime(dir: &std::path::Path, burst: u32) -> UserAuthRuntime {
        let mut opts = RocksOptions::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db =
            DB::open_cf_descriptors(&opts, dir, vec![crate::user_auth::store::cf_descriptor()])
                .unwrap();
        let store = UserAuthStore::new(Arc::new(db)).unwrap();
        UserAuthRuntime::with_limiter(
            store,
            LoginLimiter::new(burst, Duration::from_secs(60)),
            None,
        )
    }

    fn user(name: &str, role: UserRole, totp: bool) -> UserRecord {
        UserRecord {
            id: Uuid::new_v4(),
            username: name.to_string(),
            password_hash: password::hash_password("temp-pass-12345").unwrap(),
            // No encryptor in these tests: an unsealed secret reads as itself.
            totp_secret_encrypted: totp.then(|| SECRET.to_vec()),
            totp_enrollment_pending: false,
            recovery_code_hashes: vec!["h".into()],
            role,
            created_at: Utc::now(),
            last_login_at: None,
            disabled_at: None,
        }
    }

    fn session_for(rt: &UserAuthRuntime, u: &UserRecord) -> String {
        let (token, record) = session_mod::issue(u.id, Utc::now());
        rt.store.put_session(&record).unwrap();
        token.0
    }

    fn with_code(code: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(STEP_UP_HEADER, code.parse().unwrap());
        h
    }

    fn code(at: u64) -> String {
        totp::code_at(SECRET, at)
    }

    #[test]
    fn step_up_needs_2fa_and_a_code_then_elevates_only_that_session() {
        let dir = tempdir().unwrap();
        let rt = open_runtime(dir.path(), 10);
        let now = totp::current_unix_time();
        let admin = user("root", UserRole::Admin, true);
        rt.store.create_user(&admin).unwrap();
        let (a, b) = (session_for(&rt, &admin), session_for(&rt, &admin));

        let no_2fa = user("plain", UserRole::Admin, false);
        assert!(matches!(
            require_step_up(&rt, &no_2fa, &a, &with_code(&code(now)), now),
            Err(AuthError::StepUpUnavailable)
        ));
        assert!(matches!(
            require_step_up(&rt, &admin, &a, &HeaderMap::new(), now),
            Err(AuthError::StepUpRequired)
        ));
        require_step_up(&rt, &admin, &a, &with_code(&code(now)), now).unwrap();
        require_step_up(&rt, &admin, &a, &HeaderMap::new(), now)
            .expect("an elevated session needs no second code");
        assert!(
            matches!(
                require_step_up(&rt, &admin, &b, &HeaderMap::new(), now),
                Err(AuthError::StepUpRequired)
            ),
            "elevation belongs to the session that stepped up"
        );
        assert!(
            matches!(
                require_step_up(&rt, &admin, &b, &with_code(&code(now)), now),
                Err(AuthError::StepUpReplayed)
            ),
            "a code is accepted once"
        );
        require_step_up(&rt, &admin, &b, &with_code(&code(now + 30)), now + 30)
            .expect("the next code is fresh");
    }

    #[test]
    fn wrong_step_up_codes_spend_a_budget_that_then_refuses_even_a_right_one() {
        let dir = tempdir().unwrap();
        let rt = open_runtime(dir.path(), 2);
        let now = totp::current_unix_time();
        let admin = user("root", UserRole::Admin, true);
        rt.store.create_user(&admin).unwrap();
        let s = session_for(&rt, &admin);
        let window = [code(now - 30), code(now), code(now + 30)];
        let wrong = ["000000", "111111", "222222", "333333"]
            .into_iter()
            .find(|c| !window.iter().any(|w| w == c))
            .unwrap();
        for _ in 0..2 {
            assert!(matches!(
                require_step_up(&rt, &admin, &s, &with_code(wrong), now),
                Err(AuthError::StepUpRequired)
            ));
        }
        assert!(matches!(
            require_step_up(&rt, &admin, &s, &with_code(&code(now)), now),
            Err(AuthError::TooManyAttempts)
        ));
    }

    #[test]
    fn disabling_ends_sessions_blocks_the_next_request_and_is_audited() {
        let dir = tempdir().unwrap();
        let rt = open_runtime(dir.path(), 10);
        let admin = user("root", UserRole::Admin, true);
        let bob = user("bob", UserRole::User, false);
        rt.store.create_user(&admin).unwrap();
        rt.store.create_user(&bob).unwrap();
        let bob_session = session_for(&rt, &bob);

        assert!(matches!(
            set_disabled_with_runtime(&rt, &admin, "root", true),
            Err(AuthError::SelfAdministration)
        ));
        let view = set_disabled_with_runtime(&rt, &admin, "bob", true).unwrap();
        assert!(view.disabled_at.is_some());
        assert_eq!(view.sessions, 0);
        assert!(rt
            .store
            .validate_and_refresh(&session_mod::SessionToken(bob_session), Utc::now())
            .is_err());

        let again = session_for(&rt, &bob);
        assert!(
            matches!(
                rt.store
                    .validate_and_refresh(&session_mod::SessionToken(again), Utc::now()),
                Err(AuthError::AccountDisabled)
            ),
            "a session minted before the flag was read is refused too"
        );

        set_disabled_with_runtime(&rt, &admin, "bob", false).unwrap();
        let fresh = session_for(&rt, &bob);
        rt.store
            .validate_and_refresh(&session_mod::SessionToken(fresh), Utc::now())
            .unwrap();
        let actions: Vec<_> = rt
            .store
            .recent_audit(10)
            .unwrap()
            .into_iter()
            .map(|e| e.action)
            .collect();
        assert_eq!(actions, ["user_enable", "user_disable"], "newest first");
    }

    #[test]
    fn resetting_2fa_clears_the_secret_codes_and_sessions() {
        let dir = tempdir().unwrap();
        let rt = open_runtime(dir.path(), 10);
        let admin = user("root", UserRole::Admin, true);
        let carol = user("carol", UserRole::User, true);
        rt.store.create_user(&admin).unwrap();
        rt.store.create_user(&carol).unwrap();
        session_for(&rt, &carol);

        let view = reset_2fa_with_runtime(&rt, &admin, "carol").unwrap();
        assert_eq!(
            (view.totp, view.recovery_codes_remaining, view.sessions),
            ("none", 0, 0)
        );
        let stored = rt.store.find_user_by_username("carol").unwrap().unwrap();
        assert!(stored.totp_secret_encrypted.is_none());
    }

    #[test]
    fn a_session_is_revoked_by_its_public_id_only_for_its_own_user() {
        let dir = tempdir().unwrap();
        let rt = open_runtime(dir.path(), 10);
        let (alice, bob) = (
            user("alice", UserRole::User, false),
            user("bob", UserRole::User, false),
        );
        rt.store.create_user(&alice).unwrap();
        rt.store.create_user(&bob).unwrap();
        let token = session_for(&rt, &alice);
        let id = session_mod::public_id(&session_mod::hash_token(&token).unwrap());
        assert!(!rt.store.delete_session_by_public_id(&bob.id, &id).unwrap());
        assert!(rt
            .store
            .delete_session_by_public_id(&alice.id, &id)
            .unwrap());
        assert!(rt
            .store
            .list_sessions_for_user(&alice.id, Utc::now())
            .unwrap()
            .is_empty());
    }
}
