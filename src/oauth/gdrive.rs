//! GDrive OAuth provider against the `oauth2 = "5"` crate.
//!
//! # Configuration
//!
//! Reads `VELD_GDRIVE_CLIENT_ID` and `VELD_GDRIVE_CLIENT_SECRET` from
//! the environment. Returns [`OauthError::MissingClientCredentials`]
//! if either is unset when an authorize/refresh/revoke is requested.
//!
//! # Google specifics
//!
//! Google omits the `refresh_token` from authorize responses unless
//! BOTH `access_type=offline` AND `prompt=consent` are added as extra
//! params on the authorize URL. [`build_authorize_url`] sets both.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use oauth2::basic::BasicClient;
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken, PkceCodeChallenge,
    PkceCodeVerifier, RedirectUrl, RefreshToken, Scope, TokenResponse, TokenUrl,
};
use secrecy::{ExposeSecret, SecretBox};
use std::time::Duration;

use super::{OauthError, OauthProvider, TokenSet};

pub const GDRIVE_PROVIDER_NAME: &str = "gdrive";

/// Read-only Drive metadata + content scope.
pub const SCOPE_DRIVE_READONLY: &str = "https://www.googleapis.com/auth/drive.readonly";

const AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const REVOKE_URL: &str = "https://oauth2.googleapis.com/revoke";

/// Backoff steps used on transient (5xx) refresh failures.
const BACKOFF_STEPS_SECS: &[u64] = &[1, 2, 4];

/// Provider impl. The `oauth2 v5` typestate makes `BasicClient`
/// expensive to spell out in type signatures, so we recreate it per
/// call from the env-sourced credentials and a caller-supplied
/// redirect URL. The cost is four `Url::parse` calls per refresh —
/// negligible for agent-memory throughput.
pub struct GDriveOauthProvider {
    http: reqwest::Client,
    /// The `oauth2` v5 crate is async-only by default for `request_async`;
    /// we use the blocking variant for transparency with the rest of
    /// Veld's HTTP stack, but the `OauthProvider` trait surface stays
    /// `async fn`.
    blocking_http: reqwest::blocking::Client,
}

impl GDriveOauthProvider {
    pub fn new() -> Result<Self, OauthError> {
        let http = reqwest::Client::builder()
            .user_agent("veld-oauth/0.1")
            .timeout(Duration::from_secs(30))
            .build()?;
        let blocking_http = reqwest::blocking::Client::builder()
            .user_agent("veld-oauth/0.1")
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self { http, blocking_http })
    }

    fn credentials() -> Result<(String, String), OauthError> {
        let id = std::env::var("VELD_GDRIVE_CLIENT_ID")
            .map_err(|_| OauthError::MissingClientCredentials)?;
        let secret = std::env::var("VELD_GDRIVE_CLIENT_SECRET")
            .map_err(|_| OauthError::MissingClientCredentials)?;
        if id.trim().is_empty() || secret.trim().is_empty() {
            return Err(OauthError::MissingClientCredentials);
        }
        Ok((id, secret))
    }

    /// Build the typed oauth2 v5 client for a given redirect URI. The
    /// returned client has `Configured`-state endpoint typestates so
    /// `exchange_code` / `exchange_refresh_token` accept it cleanly.
    fn build_oauth_client(
        redirect_uri: &str,
    ) -> Result<
        oauth2::Client<
            oauth2::basic::BasicErrorResponse,
            oauth2::basic::BasicTokenResponse,
            oauth2::basic::BasicTokenIntrospectionResponse,
            oauth2::StandardRevocableToken,
            oauth2::basic::BasicRevocationErrorResponse,
            oauth2::EndpointSet,
            oauth2::EndpointNotSet,
            oauth2::EndpointNotSet,
            oauth2::EndpointNotSet,
            oauth2::EndpointSet,
        >,
        OauthError,
    > {
        let (id, secret) = Self::credentials()?;
        let auth = AuthUrl::new(AUTH_URL.to_string())
            .map_err(|e| OauthError::InvalidUrl(e.to_string()))?;
        let token = TokenUrl::new(TOKEN_URL.to_string())
            .map_err(|e| OauthError::InvalidUrl(e.to_string()))?;
        let redirect = RedirectUrl::new(redirect_uri.to_string())
            .map_err(|e| OauthError::InvalidUrl(e.to_string()))?;
        let client = BasicClient::new(ClientId::new(id))
            .set_client_secret(ClientSecret::new(secret))
            .set_auth_uri(auth)
            .set_token_uri(token)
            .set_redirect_uri(redirect);
        Ok(client)
    }

    /// Compose the user-facing authorize URL for the loopback flow.
    /// Returns the URL plus the CSRF state and the PKCE verifier (the
    /// caller stashes the verifier in `StateJar` keyed by the state).
    pub fn build_authorize_url(
        &self,
        redirect_uri: &str,
    ) -> Result<(oauth2::url::Url, CsrfToken, PkceCodeVerifier), OauthError> {
        let client = Self::build_oauth_client(redirect_uri)?;
        let (challenge, verifier) = PkceCodeChallenge::new_random_sha256();
        let (url, csrf) = client
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new(SCOPE_DRIVE_READONLY.to_string()))
            // Google specifics — both required for `refresh_token` issuance.
            .add_extra_param("access_type", "offline")
            .add_extra_param("prompt", "consent")
            .set_pkce_challenge(challenge)
            .url();
        Ok((url, csrf, verifier))
    }

    fn map_token_response(t: oauth2::basic::BasicTokenResponse) -> (SecretBox<String>, Option<SecretBox<String>>, DateTime<Utc>, Vec<String>) {
        let access = SecretBox::new(Box::new(t.access_token().secret().to_string()));
        let refresh = t
            .refresh_token()
            .map(|rt| SecretBox::new(Box::new(rt.secret().to_string())));
        let expires_at = t
            .expires_in()
            .and_then(|d| ChronoDuration::from_std(d).ok())
            .map(|d| Utc::now() + d)
            .unwrap_or_else(|| Utc::now() + ChronoDuration::hours(1));
        let scopes = t
            .scopes()
            .map(|ss| ss.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .unwrap_or_default();
        (access, refresh, expires_at, scopes)
    }
}

#[async_trait::async_trait]
impl OauthProvider for GDriveOauthProvider {
    fn name(&self) -> &str {
        GDRIVE_PROVIDER_NAME
    }

    async fn exchange_code(
        &self,
        code: String,
        pkce_verifier: SecretBox<String>,
        redirect_uri: String,
    ) -> Result<TokenSet, OauthError> {
        let client = Self::build_oauth_client(&redirect_uri)?;
        // The oauth2 v5 PkceCodeVerifier wraps a String. We have it as
        // SecretBox<String>; expose briefly to reconstruct the typed
        // wrapper at the crate boundary. The plaintext lives only for
        // the duration of this call.
        let verifier = PkceCodeVerifier::new(pkce_verifier.expose_secret().to_string());
        let client_clone = client;
        let http = self.blocking_http.clone();
        let response = tokio::task::spawn_blocking(move || {
            client_clone
                .exchange_code(AuthorizationCode::new(code))
                .set_pkce_verifier(verifier)
                .request(&http)
        })
        .await
        .map_err(|e| OauthError::Oauth2(format!("spawn_blocking join: {e}")))?
        .map_err(|e| classify_oauth_error(&e))?;
        let (access, refresh, expires_at, scopes) = Self::map_token_response(response);
        Ok(TokenSet {
            access_token: access,
            refresh_token: refresh,
            expires_at,
            scopes,
        })
    }

    async fn refresh(
        &self,
        refresh_token: &SecretBox<String>,
    ) -> Result<TokenSet, OauthError> {
        // The refresh path doesn't need a real redirect URI but the
        // typed client requires one. Use the registered base.
        let client = Self::build_oauth_client("http://127.0.0.1")?;
        let mut last_fatal: Option<OauthError> = None;
        for (attempt, secs) in BACKOFF_STEPS_SECS.iter().chain(std::iter::once(&0u64)).enumerate() {
            if attempt > 0 {
                let jitter_ms: u64 = rand::random::<u64>() % 500;
                tokio::time::sleep(Duration::from_secs(*secs) + Duration::from_millis(jitter_ms))
                    .await;
            }
            let token = RefreshToken::new(refresh_token.expose_secret().to_string());
            let http = self.blocking_http.clone();
            let client_clone = client.clone();
            let result = tokio::task::spawn_blocking(move || {
                client_clone
                    .exchange_refresh_token(&token)
                    .request(&http)
            })
            .await
            .map_err(|e| OauthError::Oauth2(format!("spawn_blocking join: {e}")))?;
            match result {
                Ok(t) => {
                    let (access, refresh, expires_at, scopes) =
                        Self::map_token_response(t);
                    return Ok(TokenSet {
                        access_token: access,
                        refresh_token: refresh,
                        expires_at,
                        scopes,
                    });
                }
                Err(e) => {
                    let mapped = classify_oauth_error(&e);
                    match &mapped {
                        OauthError::TokenEndpointTransient(_) => {
                            // try next backoff step
                            last_fatal = Some(mapped);
                            continue;
                        }
                        OauthError::RefreshTokenRevoked
                        | OauthError::TokenEndpointFatal(_)
                        | OauthError::Oauth2(_) => return Err(mapped),
                        _ => return Err(mapped),
                    }
                }
            }
        }
        Err(last_fatal.unwrap_or(OauthError::TokenEndpointTransient(0)))
    }

    async fn revoke(&self, access_token: &SecretBox<String>) -> Result<(), OauthError> {
        let token = access_token.expose_secret().to_string();
        let http = self.http.clone();
        let resp = http
            .post(REVOKE_URL)
            .form(&[("token", token.as_str())])
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(OauthError::TokenEndpointFatal(resp.status().as_u16()));
        }
        Ok(())
    }
}

/// Map an oauth2-crate error to our wider [`OauthError`] enum.
fn classify_oauth_error<RE, T>(err: &oauth2::RequestTokenError<RE, T>) -> OauthError
where
    RE: std::error::Error + 'static,
    T: oauth2::ErrorResponse + 'static,
{
    let s = err.to_string();
    // Google returns `invalid_grant` when the refresh token was
    // revoked or expired.
    if s.contains("invalid_grant") {
        return OauthError::RefreshTokenRevoked;
    }
    // Best-effort transient detection: 5xx codes show up in the
    // formatted error string.
    for code in [500u16, 502, 503, 504] {
        if s.contains(&code.to_string()) {
            return OauthError::TokenEndpointTransient(code);
        }
    }
    if s.contains("401") || s.contains("400") || s.contains("403") {
        let code = if s.contains("401") {
            401
        } else if s.contains("403") {
            403
        } else {
            400
        };
        return OauthError::TokenEndpointFatal(code);
    }
    OauthError::Oauth2(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_GUARD: Mutex<()> = Mutex::new(());

    fn with_creds<F: FnOnce()>(f: F) {
        let _g = ENV_GUARD.lock().unwrap();
        let prev_id = std::env::var("VELD_GDRIVE_CLIENT_ID").ok();
        let prev_secret = std::env::var("VELD_GDRIVE_CLIENT_SECRET").ok();
        std::env::set_var("VELD_GDRIVE_CLIENT_ID", "test-id");
        std::env::set_var("VELD_GDRIVE_CLIENT_SECRET", "test-secret");
        f();
        match prev_id {
            Some(v) => std::env::set_var("VELD_GDRIVE_CLIENT_ID", v),
            None => std::env::remove_var("VELD_GDRIVE_CLIENT_ID"),
        }
        match prev_secret {
            Some(v) => std::env::set_var("VELD_GDRIVE_CLIENT_SECRET", v),
            None => std::env::remove_var("VELD_GDRIVE_CLIENT_SECRET"),
        }
    }

    #[test]
    fn build_authorize_url_contains_required_params() {
        with_creds(|| {
            let provider = GDriveOauthProvider::new().expect("provider");
            let (url, _csrf, _verifier) = provider
                .build_authorize_url("http://127.0.0.1:8080")
                .expect("authorize url");
            let s = url.to_string();
            assert!(
                s.contains("access_type=offline"),
                "missing access_type=offline in {s}"
            );
            assert!(
                s.contains("prompt=consent"),
                "missing prompt=consent in {s}"
            );
            assert!(s.contains("code_challenge_method=S256"));
            assert!(s.contains("drive.readonly"));
        });
    }

    #[test]
    fn missing_credentials_returns_typed_error() {
        let _g = ENV_GUARD.lock().unwrap();
        let prev_id = std::env::var("VELD_GDRIVE_CLIENT_ID").ok();
        let prev_secret = std::env::var("VELD_GDRIVE_CLIENT_SECRET").ok();
        std::env::remove_var("VELD_GDRIVE_CLIENT_ID");
        std::env::remove_var("VELD_GDRIVE_CLIENT_SECRET");
        let provider = GDriveOauthProvider::new().expect("provider construct");
        let err = provider
            .build_authorize_url("http://127.0.0.1")
            .unwrap_err();
        assert!(
            matches!(err, OauthError::MissingClientCredentials),
            "expected MissingClientCredentials, got {err:?}"
        );
        if let Some(v) = prev_id {
            std::env::set_var("VELD_GDRIVE_CLIENT_ID", v);
        }
        if let Some(v) = prev_secret {
            std::env::set_var("VELD_GDRIVE_CLIENT_SECRET", v);
        }
    }
}
