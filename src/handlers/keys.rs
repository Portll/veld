//! Key management over HTTP, for the Veld Monitor's Security tab.
//!
//! The same operations as `veld keys`, behind an Admin session. Every change also
//! needs step-up (`handlers::user_admin`) and opens the keyring with a factor in the
//! body: the passphrase, a recovery code, or a device secret. Failed unlocks share
//! one budget across the server.
//!
//! The bodies carry passphrases and key material, so a request is refused unless it
//! arrived over the TLS listener, the plain listener is on loopback, or VELD_TLS_ACK
//! says TLS terminates in front of the server.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Path as UrlPath, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    Extension, Json,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;
use zeroize::Zeroizing;

use super::state::MultiUserMemoryManager;
use super::user_admin::{audit, require_admin, require_admin_step_up};
use crate::keyring::{
    self, passkey, EpochStarted, InitReport, IssuedCode, KeyStatus, MasterKeysRotated, Rewrapped,
};
use crate::keystore::{Factor, FactorInfo, PROVIDER_PASSKEY, PROVIDER_SECURE_ENCLAVE};
use crate::tls::ArrivedOverTls;
use crate::user_auth::{
    runtime::UserAuthRuntime, AuthError, SessionTokenExt, SessionUser, UserRecord,
};

type AppState = Arc<MultiUserMemoryManager>;

const UNLOCK_BUDGET: &str = "keys:unlock";
const PASSKEY_TICKET_TTL: Duration = Duration::from_secs(300);

/// A factor as it arrives in a request body.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnlockBody {
    Passphrase(String),
    RecoveryCode(String),
    SecureEnclave { id: String, secret: String },
    Passkey { id: String, secret: String },
}

/// An owned factor, zeroized on drop.
pub struct Unlock {
    provider: &'static str,
    text: Zeroizing<String>,
    id: String,
    secret: Zeroizing<Vec<u8>>,
}

impl Unlock {
    fn from_body(body: UnlockBody) -> Result<Self, KeysError> {
        let device = |provider, id: String, secret: String| {
            let secret = Zeroizing::new(
                base64::engine::general_purpose::STANDARD
                    .decode(Zeroizing::new(secret).as_bytes())
                    .map_err(|_| KeysError::bad("the device secret is not base64"))?,
            );
            Ok(Self {
                provider,
                text: Zeroizing::new(String::new()),
                id,
                secret,
            })
        };
        match body {
            UnlockBody::Passphrase(p) => Ok(Self {
                provider: "passphrase",
                text: Zeroizing::new(p),
                id: String::new(),
                secret: Zeroizing::new(Vec::new()),
            }),
            UnlockBody::RecoveryCode(c) => Ok(Self {
                provider: "recovery",
                text: Zeroizing::new(c),
                id: String::new(),
                secret: Zeroizing::new(Vec::new()),
            }),
            UnlockBody::SecureEnclave { id, secret } => device(PROVIDER_SECURE_ENCLAVE, id, secret),
            UnlockBody::Passkey { id, secret } => device(PROVIDER_PASSKEY, id, secret),
        }
    }

    fn factor(&self) -> Factor<'_> {
        match self.provider {
            "passphrase" => Factor::Passphrase(&self.text),
            "recovery" => Factor::RecoveryCode(&self.text),
            provider => Factor::Secret {
                provider,
                id: &self.id,
                secret: &self.secret,
            },
        }
    }
}

/// Why a key request was refused, as an HTTP response.
pub enum KeysError {
    Auth(AuthError),
    Status(StatusCode, &'static str, String),
}

impl KeysError {
    fn bad(message: impl Into<String>) -> Self {
        Self::Status(StatusCode::BAD_REQUEST, "BAD_REQUEST", message.into())
    }
}

impl From<AuthError> for KeysError {
    fn from(e: AuthError) -> Self {
        Self::Auth(e)
    }
}

impl IntoResponse for KeysError {
    fn into_response(self) -> Response {
        match self {
            Self::Auth(e) => e.into_response(),
            Self::Status(status, code, message) => {
                (status, Json(json!({ "code": code, "message": message }))).into_response()
            }
        }
    }
}

/// The reply of `GET /api/user_auth/admin/keys/status`.
#[derive(Debug, Serialize)]
pub struct KeysStatusResponse {
    pub status: KeyStatus,
    /// The id of the ring key this server holds, `None` when it started without one.
    pub keyring_unsealed: Option<String>,
}

/// The reply of `POST /api/user_auth/admin/keys/passkey`: the ticket to poll and the
/// path of the page that finishes the enrolment.
#[derive(Debug, Serialize, Deserialize)]
pub struct PasskeyStartResponse {
    pub ticket: String,
    pub path: String,
}

/// The reply of `GET /api/user_auth/admin/keys/passkey/{ticket}`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PasskeyPollResponse {
    Waiting,
    Done { factor: FactorInfo },
    Failed { error: String },
}

fn runtime(state: &AppState) -> Result<&UserAuthRuntime, KeysError> {
    state
        .user_auth_runtime
        .as_ref()
        .ok_or(KeysError::Auth(AuthError::Disabled))
}

/// The TLS listener marks its requests [`ArrivedOverTls`]; a client cannot.
type OverTls = Option<Extension<ArrivedOverTls>>;

fn transport_is_private(plain_host: &str, over_tls: bool) -> bool {
    over_tls
        || crate::config::is_local_bind_host(plain_host)
        || std::env::var("VELD_TLS_ACK")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
}

fn require_private_transport(state: &AppState, over_tls: &OverTls) -> Result<(), KeysError> {
    if transport_is_private(&state.server_config.host, over_tls.is_some()) {
        Ok(())
    } else {
        Err(KeysError::Status(
            StatusCode::FORBIDDEN,
            "TLS_REQUIRED",
            "key management carries passphrases; call over the TLS listener (VELD_TLS_PORT), \
             bind the plain listener to loopback, or terminate TLS in front of veld and set \
             VELD_TLS_ACK=true"
                .into(),
        ))
    }
}

/// Run a keyring operation off the async runtime: they run Argon2 and write files.
/// An unlock that fails counts against the shared budget.
async fn keyring_op<T: Serialize + Send + 'static>(
    rt: &UserAuthRuntime,
    op: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> Result<T, KeysError> {
    if rt.login_limiter.peek(UNLOCK_BUDGET).is_err() {
        return Err(KeysError::Auth(AuthError::TooManyAttempts));
    }
    let outcome = tokio::task::spawn_blocking(op).await.map_err(|e| {
        KeysError::Auth(AuthError::internal(format!("key operation panicked: {e}")))
    })?;
    match outcome {
        // Serialised here as well as in the reply, so a report that cannot be (a path that
        // is not UTF-8) is refused before the handler writes its audit record.
        Ok(value) => serde_json::to_value(&value)
            .map(|_| value)
            .map_err(|e| KeysError::Auth(AuthError::internal(e.to_string()))),
        Err(e) => {
            let _ = rt.login_limiter.check(UNLOCK_BUDGET);
            Err(KeysError::Status(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KEY_OPERATION_REFUSED",
                format!("{e:#}"),
            ))
        }
    }
}

/// Private transport, an Admin session, and step-up.
fn gate(
    state: &AppState,
    over_tls: &OverTls,
    caller: &UserRecord,
    session: &str,
    headers: &HeaderMap,
) -> Result<(), KeysError> {
    require_private_transport(state, over_tls)?;
    require_admin_step_up(runtime(state)?, caller, session, headers)?;
    Ok(())
}

/// `GET /api/user_auth/admin/keys/status` — what is enrolled where; nothing is unsealed.
pub async fn status(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
) -> Result<Json<KeysStatusResponse>, KeysError> {
    require_private_transport(&state, &over_tls)?;
    runtime(&state)?;
    require_admin(&caller.0)?;
    let storage = state.base_path.clone();
    let status = tokio::task::spawn_blocking(move || keyring::status(&storage))
        .await
        .map_err(|e| KeysError::Auth(AuthError::internal(e.to_string())))?
        .map_err(|e| KeysError::Auth(AuthError::internal(format!("{e:#}"))))?;
    Ok(Json(KeysStatusResponse {
        status,
        keyring_unsealed: keyring::installed().map(|r| r.id().to_string()),
    }))
}

#[derive(Deserialize)]
pub struct InitBody {
    pub passphrase: String,
    pub label: Option<String>,
}

/// `POST /api/user_auth/admin/keys/init` — step-up. Create the keyring and the first kit.
pub async fn init(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<InitBody>,
) -> Result<Json<InitReport>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    let storage = state.base_path.clone();
    let passphrase = Zeroizing::new(body.passphrase);
    let report = keyring_op(rt, move || {
        keyring::init(&storage, &passphrase, body.label.as_deref())
    })
    .await?;
    audit(
        rt,
        &caller.0,
        "keyring_init",
        None,
        json!({ "ring": report.ring }),
    )?;
    Ok(Json(report))
}

#[derive(Deserialize)]
pub struct IssueBody {
    pub unlock: UnlockBody,
    pub label: Option<String>,
}

/// `POST /api/user_auth/admin/keys/recovery_codes` — step-up. A new recovery code and its kit,
/// shown once.
pub async fn add_recovery_code(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<IssueBody>,
) -> Result<Json<IssuedCode>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    let storage = state.base_path.clone();
    let unlock = Unlock::from_body(body.unlock)?;
    let issued = keyring_op(rt, move || {
        keyring::add_recovery_code(&storage, &unlock.factor(), body.label.as_deref())
    })
    .await?;
    audit(
        rt,
        &caller.0,
        "recovery_code_issued",
        Some(issued.id.as_str()),
        json!({ "label": issued.label }),
    )?;
    Ok(Json(issued))
}

#[derive(Deserialize)]
pub struct RevokeBody {
    pub unlock: UnlockBody,
    pub id: String,
}

/// `POST /api/user_auth/admin/keys/revoke` — step-up.
pub async fn revoke(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<RevokeBody>,
) -> Result<Json<FactorInfo>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    let storage = state.base_path.clone();
    let unlock = Unlock::from_body(body.unlock)?;
    let id = body.id.clone();
    let revoked = keyring_op(rt, move || keyring::revoke(&storage, &unlock.factor(), &id)).await?;
    audit(
        rt,
        &caller.0,
        "key_factor_revoked",
        Some(&body.id),
        json!({}),
    )?;
    Ok(Json(revoked))
}

#[derive(Deserialize)]
pub struct RotatePassphraseBody {
    pub old: String,
    pub new: String,
}

/// `POST /api/user_auth/admin/keys/rotate_passphrase` — step-up. The server keeps unsealing
/// with whatever its environment holds: set the new passphrase there before restart.
pub async fn rotate_passphrase(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<RotatePassphraseBody>,
) -> Result<Json<Rewrapped>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    let storage = state.base_path.clone();
    let (old, new) = (Zeroizing::new(body.old), Zeroizing::new(body.new));
    let rewrapped =
        keyring_op(rt, move || keyring::rotate_passphrase(&storage, &old, &new)).await?;
    audit(rt, &caller.0, "passphrase_rotated", None, json!({}))?;
    Ok(Json(rewrapped))
}

#[derive(Deserialize)]
pub struct UnlockOnlyBody {
    pub unlock: UnlockBody,
}

/// `POST /api/user_auth/admin/keys/rotate_data_keys` — step-up.
pub async fn rotate_data_keys(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<UnlockOnlyBody>,
) -> Result<Json<Vec<EpochStarted>>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    let storage = state.base_path.clone();
    let unlock = Unlock::from_body(body.unlock)?;
    let started = keyring_op(rt, move || {
        keyring::rotate_data_keys(&storage, &unlock.factor())
    })
    .await?;
    audit(rt, &caller.0, "data_keys_rotated", None, json!({}))?;
    Ok(Json(started))
}

/// `POST /api/user_auth/admin/keys/rotate_master_keys` — step-up. Drops every recovery code
/// and device factor and issues a new kit.
pub async fn rotate_master_keys(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<InitBody>,
) -> Result<Json<MasterKeysRotated>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    let storage = state.base_path.clone();
    let passphrase = Zeroizing::new(body.passphrase);
    let rotated = keyring_op(rt, move || {
        keyring::rotate_master_keys(&storage, &passphrase, body.label.as_deref())
    })
    .await?;
    audit(
        rt,
        &caller.0,
        "master_keys_rotated",
        None,
        json!({ "dropped": rotated.dropped }),
    )?;
    Ok(Json(rotated))
}

#[derive(Deserialize)]
pub struct EnrolBody {
    pub unlock: UnlockBody,
    /// `secure-enclave`: the Monitor ran `veld-enclave enrol` on this Mac and posts
    /// the secret it released.
    pub provider: String,
    pub id: String,
    pub label: Option<String>,
    pub secret: String,
}

/// `POST /api/user_auth/admin/keys/factors` — step-up. Enrol a device secret.
pub async fn enrol_factor(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<EnrolBody>,
) -> Result<Json<FactorInfo>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let rt = runtime(&state)?;
    if body.provider != PROVIDER_SECURE_ENCLAVE {
        return Err(KeysError::bad(
            "only secure-enclave secrets are posted; passkeys enrol through /api/user_auth/admin/keys/passkey",
        ));
    }
    let storage = state.base_path.clone();
    let unlock = Unlock::from_body(body.unlock)?;
    let secret = Zeroizing::new(
        base64::engine::general_purpose::STANDARD
            .decode(Zeroizing::new(body.secret).as_bytes())
            .map_err(|_| KeysError::bad("the secret is not base64"))?,
    );
    let (id, label) = (body.id.clone(), body.label.clone());
    let enrolled = keyring_op(rt, move || {
        keyring::enrol_secret(
            &storage,
            &unlock.factor(),
            PROVIDER_SECURE_ENCLAVE,
            &id,
            label.as_deref(),
            &secret,
        )
    })
    .await?;
    audit(
        rt,
        &caller.0,
        "key_factor_enrolled",
        Some(&body.id),
        json!({ "provider": body.provider }),
    )?;
    Ok(Json(enrolled))
}

// ── Passkeys through the server's own origin ────────────────────────────

enum TicketState {
    Waiting,
    Done(FactorInfo),
    Failed(String),
}

pub struct PasskeyTicket {
    created: Instant,
    rp_id: String,
    label: String,
    actor: UserRecord,
    unlock: Unlock,
    state: TicketState,
}

/// Tickets for passkey enrolments in progress, keyed by an unguessable id. A ticket
/// holds the unlock factor until the browser posts the passkey's secret or it expires.
#[derive(Default)]
pub struct PasskeyTickets(parking_lot::Mutex<std::collections::HashMap<String, PasskeyTicket>>);

impl PasskeyTickets {
    fn prune(map: &mut std::collections::HashMap<String, PasskeyTicket>) {
        map.retain(|_, t| t.created.elapsed() < PASSKEY_TICKET_TTL);
    }
}

/// The relying party for a passkey made through this server: the host the Monitor
/// reached it on, with loopback addresses named `localhost` (an IP is not a valid
/// relying party id).
fn relying_party(headers: &HeaderMap) -> String {
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let name = match host.rsplit_once(':') {
        Some((name, port)) if port.chars().all(|c| c.is_ascii_digit()) => name,
        _ => host,
    };
    let name = name.trim_start_matches('[').trim_end_matches(']');
    if name == "127.0.0.1" || name == "::1" || name.eq_ignore_ascii_case("localhost") {
        passkey::RP_LOCALHOST.to_string()
    } else {
        name.to_ascii_lowercase()
    }
}

#[derive(Deserialize)]
pub struct PasskeyStartBody {
    pub unlock: UnlockBody,
    pub label: Option<String>,
}

/// `POST /api/user_auth/admin/keys/passkey` — step-up. Returns the page to open in a browser,
/// which creates the passkey and posts its PRF secret back.
pub async fn passkey_start(
    State(state): State<AppState>,
    over_tls: OverTls,
    Extension(caller): Extension<SessionUser>,
    Extension(token): Extension<SessionTokenExt>,
    headers: HeaderMap,
    Json(body): Json<PasskeyStartBody>,
) -> Result<Json<PasskeyStartResponse>, KeysError> {
    gate(&state, &over_tls, &caller.0, &token.0, &headers)?;
    let unlock = Unlock::from_body(body.unlock)?;
    let mut raw = [0u8; 24];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut raw);
    let ticket = hex::encode(raw);
    let rp_id = relying_party(&headers);
    let mut tickets = state.passkey_tickets.0.lock();
    PasskeyTickets::prune(&mut tickets);
    tickets.insert(
        ticket.clone(),
        PasskeyTicket {
            created: Instant::now(),
            label: body.label.unwrap_or_else(|| format!("passkey ({rp_id})")),
            rp_id,
            actor: caller.0.clone(),
            unlock,
            state: TicketState::Waiting,
        },
    );
    let path = format!("/api/user_auth/keys/passkey/{ticket}");
    Ok(Json(PasskeyStartResponse { ticket, path }))
}

/// `GET /api/user_auth/admin/keys/passkey/{ticket}` — `waiting`, `done` with the factor, or
/// `failed` with why.
pub async fn passkey_poll(
    State(state): State<AppState>,
    Extension(caller): Extension<SessionUser>,
    UrlPath(ticket): UrlPath<String>,
) -> Result<Json<PasskeyPollResponse>, StatusCode> {
    let mut tickets = state.passkey_tickets.0.lock();
    PasskeyTickets::prune(&mut tickets);
    let reply = match tickets.get(&ticket) {
        Some(t) if t.actor.id == caller.0.id => match &t.state {
            TicketState::Waiting => PasskeyPollResponse::Waiting,
            TicketState::Done(factor) => PasskeyPollResponse::Done {
                factor: factor.clone(),
            },
            TicketState::Failed(error) => PasskeyPollResponse::Failed {
                error: error.clone(),
            },
        },
        _ => return Err(StatusCode::NOT_FOUND),
    };
    if !matches!(reply, PasskeyPollResponse::Waiting) {
        tickets.remove(&ticket);
    }
    Ok(Json(reply))
}

/// `GET /api/user_auth/keys/passkey/{ticket}` — the ceremony page. Unauthenticated: the ticket is
/// the capability, and it only ever enrols for the admin who started it.
///
/// The reply is an HTML page (or a plain-text 404 for an expired ticket), not JSON, so the
/// schema reference lists it as opaque. The page posts its result to
/// `POST /api/user_auth/keys/passkey/{ticket}/result`.
pub async fn passkey_page(
    State(state): State<AppState>,
    UrlPath(ticket): UrlPath<String>,
) -> Response {
    let tickets = state.passkey_tickets.0.lock();
    let Some(t) = tickets.get(&ticket).filter(|t| {
        t.created.elapsed() < PASSKEY_TICKET_TTL && matches!(t.state, TicketState::Waiting)
    }) else {
        return (StatusCode::NOT_FOUND, "this passkey link has expired").into_response();
    };
    let mut raw = [0u8; 16];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut raw);
    let script_nonce = hex::encode(raw);
    let html = passkey::page(
        &passkey::Ceremony::Enrol {
            label: t.label.clone(),
        },
        &t.rp_id,
        &format!("/api/user_auth/keys/passkey/{ticket}/result"),
        &script_nonce,
    );
    let csp = format!(
        "default-src 'none'; script-src 'nonce-{script_nonce}'; style-src 'unsafe-inline'; \
         connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    );
    (
        [
            (header::CACHE_CONTROL, "no-store".to_string()),
            (header::CONTENT_SECURITY_POLICY, csp),
        ],
        Html(html),
    )
        .into_response()
}

#[derive(Deserialize)]
pub struct PasskeyResultBody {
    pub credential_id: Option<String>,
    pub prf: Option<String>,
    pub error: Option<String>,
}

/// `POST /api/user_auth/keys/passkey/{ticket}/result` — the page posts the passkey's PRF secret;
/// the server enrols it with the unlock the ticket holds. The reply has no body: `204` when
/// the factor is enrolled, `422` when it is not (the poll says why), `404` for a ticket that
/// is unknown, expired or already used.
pub async fn passkey_result(
    State(state): State<AppState>,
    UrlPath(ticket): UrlPath<String>,
    Json(body): Json<PasskeyResultBody>,
) -> StatusCode {
    let taken = {
        let mut tickets = state.passkey_tickets.0.lock();
        match tickets.get_mut(&ticket) {
            Some(t)
                if t.created.elapsed() < PASSKEY_TICKET_TTL
                    && matches!(t.state, TicketState::Waiting) =>
            {
                let unlock = std::mem::replace(
                    &mut t.unlock,
                    Unlock {
                        provider: "passphrase",
                        text: Zeroizing::new(String::new()),
                        id: String::new(),
                        secret: Zeroizing::new(Vec::new()),
                    },
                );
                t.state = TicketState::Failed("in progress".into());
                (unlock, t.rp_id.clone(), t.label.clone(), t.actor.clone())
            }
            _ => return StatusCode::NOT_FOUND,
        }
    };
    let (unlock, rp_id, label, actor) = taken;
    let outcome = async {
        if let Some(error) = body.error {
            return Err(format!("the browser reported: {error}"));
        }
        let credential = body.credential_id.ok_or("the page posted no credential")?;
        let secret = Zeroizing::new(
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(
                    body.prf
                        .ok_or("the page posted no PRF secret")?
                        .trim_end_matches('='),
                )
                .map_err(|_| "the PRF secret is not base64url".to_string())?,
        );
        let id = passkey::factor_id(&rp_id, &credential);
        let storage = state.base_path.clone();
        let enrolled_id = id.clone();
        let factor = tokio::task::spawn_blocking(move || {
            keyring::enrol_secret(
                &storage,
                &unlock.factor(),
                PROVIDER_PASSKEY,
                &enrolled_id,
                Some(&label),
                &secret,
            )
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{e:#}"))?;
        if let Some(rt) = state.user_auth_runtime.as_ref() {
            let _ = audit(
                rt,
                &actor,
                "key_factor_enrolled",
                Some(&id),
                json!({ "provider": PROVIDER_PASSKEY }),
            );
        }
        Ok(factor)
    }
    .await;
    let status = match &outcome {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(_) => StatusCode::UNPROCESSABLE_ENTITY,
    };
    if let Some(t) = state.passkey_tickets.0.lock().get_mut(&ticket) {
        t.state = match outcome {
            Ok(factor) => TicketState::Done(factor),
            Err(error) => TicketState::Failed(error),
        };
    }
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_over_tls_is_private_whatever_the_plain_bind() {
        assert!(transport_is_private("0.0.0.0", true));
        assert!(transport_is_private("10.0.0.2", true));
        assert!(transport_is_private("127.0.0.1", false));
        if std::env::var_os("VELD_TLS_ACK").is_none() {
            assert!(!transport_is_private("0.0.0.0", false));
        }
    }

    #[test]
    fn typed_replies_keep_the_json_keys_they_replaced() {
        fn keys(value: impl Serialize) -> Vec<String> {
            let mut keys: Vec<String> = serde_json::to_value(value)
                .unwrap()
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            keys.sort();
            keys
        }
        let status = KeysStatusResponse {
            status: KeyStatus {
                keyring: None,
                keystores: Vec::new(),
            },
            keyring_unsealed: None,
        };
        assert_eq!(keys(&status), ["keyring_unsealed", "status"]);
        assert_eq!(
            keys(PasskeyStartResponse {
                ticket: "t".into(),
                path: "/p".into()
            }),
            ["path", "ticket"]
        );
        let waiting = serde_json::to_value(PasskeyPollResponse::Waiting).unwrap();
        assert_eq!(waiting, json!({ "state": "waiting" }));
        let factor = FactorInfo {
            provider: PROVIDER_PASSKEY.into(),
            id: "passkey:localhost:x".into(),
            label: None,
            created_at: None,
        };
        let done = serde_json::to_value(PasskeyPollResponse::Done {
            factor: factor.clone(),
        })
        .unwrap();
        assert_eq!(
            done,
            json!({ "state": "done", "factor": serde_json::to_value(&factor).unwrap() })
        );
        let failed =
            serde_json::to_value(PasskeyPollResponse::Failed { error: "no".into() }).unwrap();
        assert_eq!(failed, json!({ "state": "failed", "error": "no" }));
    }
}
