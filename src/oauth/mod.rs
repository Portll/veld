//! OAuth2 connector substrate for Veld's external-content ingest path.
//!
//! Phase 1 ships the GDrive provider. The trait surface (`OauthProvider`),
//! storage (`TokenStore`), CSRF state allocator (`StateJar`), loopback
//! callback server (`loopback::run_loopback_once`), and policy gate
//! (`policy::is_production`) are shared infrastructure designed for the
//! GitHub OAuth follow-up to drop in as a sibling impl.
//!
//! # Storage layout
//!
//! All persisted state lives under the [`CF_OAUTH_TOKENS`] column family
//! in the shared RocksDB instance, mirroring the
//! [`crate::user_auth::store`] pattern:
//!
//! | Key shape                          | Value                         |
//! |------------------------------------|-------------------------------|
//! | `token:<user_id>:<provider>`       | bincode [`OauthRecord`]       |
//!
//! Token plaintext is never written — `access_token_ct` and
//! `refresh_token_ct` are AES-256-GCM ciphertext from
//! [`crate::encryption::FieldEncryptor`]. In-memory plaintext is wrapped
//! in [`secrecy::SecretBox<String>`], whose `Debug` / `Display` impls
//! print `[REDACTED]`; plaintext extraction requires the explicit
//! `.expose_secret()` call.
//!
//! # Bincode positional decode
//!
//! `bincode 2 + config::standard()` is positional — trailing
//! `Option<T>` fields with `#[serde(default)]` do NOT decode old records
//! (see `tests/bincode_schema_probe.rs`). When `OauthRecord` grows new
//! fields, add a `LegacyOauthRecordVN` shim and chain it into
//! [`decode_oauth_record_with_fallback`].

use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamilyDescriptor, Options};
use secrecy::SecretBox;
use serde::{Deserialize, Serialize};

pub mod gdrive;
pub mod loopback;
pub mod policy;
pub mod state_jar;
pub mod token_store;

pub use gdrive::{GDriveOauthProvider, GDRIVE_PROVIDER_NAME};
pub use policy::is_production;
pub use state_jar::{LoopbackSession, StateJar};
pub use token_store::{TokenStore, CF_OAUTH_TOKENS, KEY_PREFIX_TOKEN};

/// Build the column-family descriptor for `cf_oauth_tokens`. Caller is
/// responsible for handing this to `DB::open_cf_descriptors(...)` at
/// shared-DB bootstrap time (mirrors [`crate::user_auth::store::cf_descriptor`]).
pub fn cf_descriptor() -> ColumnFamilyDescriptor {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    ColumnFamilyDescriptor::new(CF_OAUTH_TOKENS, opts)
}

/// Persisted OAuth record. Token fields are ciphertext only —
/// in-memory plaintext is held by [`DecryptedOauthRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OauthRecord {
    /// Veld user_id that owns this token. Matches the prefix in the
    /// storage key for cheap key-only scans.
    pub veld_user_id: String,
    /// Lower-case provider identifier, e.g. `"gdrive"`.
    pub provider: String,
    /// `FieldEncryptor::encrypt_content(access_token_plaintext)`.
    pub access_token_ct: Vec<u8>,
    /// `FieldEncryptor::encrypt_content(refresh_token_plaintext)`.
    pub refresh_token_ct: Vec<u8>,
    /// Wall-clock expiry of the access token.
    pub expires_at: DateTime<Utc>,
    /// Granted scopes (string-typed; provider-defined).
    pub scopes: Vec<String>,
    /// When the record was originally established by `exchange_code`.
    pub obtained_at: DateTime<Utc>,
}

/// Decrypted in-memory view of an [`OauthRecord`].
///
/// Token fields are wrapped in [`SecretBox<String>`]: the type system
/// enforces that `format!("{record:?}")` redacts the values, and
/// `secrecy` deliberately omits a `Serialize` impl so JSON encoders
/// cannot accidentally exfiltrate plaintext at the network boundary.
pub struct DecryptedOauthRecord {
    pub veld_user_id: String,
    pub provider: String,
    pub access_token: SecretBox<String>,
    pub refresh_token: SecretBox<String>,
    pub expires_at: DateTime<Utc>,
    pub scopes: Vec<String>,
    pub obtained_at: DateTime<Utc>,
}

impl std::fmt::Debug for DecryptedOauthRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecryptedOauthRecord")
            .field("veld_user_id", &self.veld_user_id)
            .field("provider", &self.provider)
            .field("access_token", &"[REDACTED]")
            .field("refresh_token", &"[REDACTED]")
            .field("expires_at", &self.expires_at)
            .field("scopes", &self.scopes)
            .field("obtained_at", &self.obtained_at)
            .finish()
    }
}

/// Pre-v2 `OauthRecord` shape. Identical to v1 (current) at Phase-1
/// launch; lives here as a compile-time guard against forgetting the
/// shim pattern when the schema bumps. The first real bump produces
/// `LegacyOauthRecordV2` with `upgrade()` filling in defaults for any
/// added fields.
#[derive(Deserialize)]
struct LegacyOauthRecordV1 {
    veld_user_id: String,
    provider: String,
    access_token_ct: Vec<u8>,
    refresh_token_ct: Vec<u8>,
    expires_at: DateTime<Utc>,
    scopes: Vec<String>,
    obtained_at: DateTime<Utc>,
}

impl LegacyOauthRecordV1 {
    fn upgrade(self) -> OauthRecord {
        OauthRecord {
            veld_user_id: self.veld_user_id,
            provider: self.provider,
            access_token_ct: self.access_token_ct,
            refresh_token_ct: self.refresh_token_ct,
            expires_at: self.expires_at,
            scopes: self.scopes,
            obtained_at: self.obtained_at,
        }
    }
}

/// Decode an `OauthRecord` from RocksDB bytes with the legacy-v1
/// fallback chain. Returns `(record, is_legacy)` — when `is_legacy` is
/// true, the on-disk bytes were promoted from an older shape and the
/// caller may opt to re-write under the current schema to amortize
/// future reads.
pub fn decode_oauth_record_with_fallback(
    bytes: &[u8],
) -> Result<(OauthRecord, bool), OauthError> {
    match bincode::serde::decode_from_slice::<OauthRecord, _>(
        bytes,
        bincode::config::standard(),
    ) {
        Ok((rec, _)) => Ok((rec, false)),
        Err(_) => {
            let (legacy, _): (LegacyOauthRecordV1, _) = bincode::serde::decode_from_slice(
                bytes,
                bincode::config::standard(),
            )
            .map_err(|e| OauthError::Bincode(e.to_string()))?;
            tracing::debug!("Migrated OauthRecord from v1 shape");
            Ok((legacy.upgrade(), true))
        }
    }
}

/// `OauthProvider`-specific error variants are mapped through this
/// shared error type so handlers / CLI can pattern-match without
/// dragging in provider-specific dependencies.
#[derive(thiserror::Error, Debug)]
pub enum OauthError {
    #[error("VELD_ENCRYPTION_KEY missing in production environment")]
    EncryptionKeyMissing,
    #[error("VELD_GDRIVE_CLIENT_ID or VELD_GDRIVE_CLIENT_SECRET missing")]
    MissingClientCredentials,
    #[error("OAuth `state` parameter invalid or expired")]
    InvalidOrExpiredState,
    #[error("Refresh token revoked server-side; user must re-authenticate")]
    RefreshTokenRevoked,
    #[error("Token endpoint transient status {0}; backoff exhausted")]
    TokenEndpointTransient(u16),
    #[error("Token endpoint fatal status {0}")]
    TokenEndpointFatal(u16),
    #[error("Failed to bind loopback after 3 attempts")]
    LoopbackBindFailed,
    #[error("Browser open failed; auth URL must be opened manually")]
    BrowserOpenFailed,
    #[error("Loopback callback did not arrive within timeout")]
    CallbackTimeout,
    #[error("No OAuth record stored for user/provider")]
    NoRecord,
    #[error("Malformed OAuth record (empty ciphertext or corrupt header)")]
    MalformedRecord,
    #[error("rocksdb: {0}")]
    Rocksdb(String),
    #[error("bincode: {0}")]
    Bincode(String),
    #[error("encryption: {0}")]
    Encryption(String),
    #[error("reqwest: {0}")]
    Reqwest(String),
    #[error("oauth2: {0}")]
    Oauth2(String),
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    #[error("io: {0}")]
    Io(String),
}

impl From<rocksdb::Error> for OauthError {
    fn from(e: rocksdb::Error) -> Self {
        OauthError::Rocksdb(e.to_string())
    }
}

impl From<bincode::error::EncodeError> for OauthError {
    fn from(e: bincode::error::EncodeError) -> Self {
        OauthError::Bincode(e.to_string())
    }
}

impl From<bincode::error::DecodeError> for OauthError {
    fn from(e: bincode::error::DecodeError) -> Self {
        OauthError::Bincode(e.to_string())
    }
}

impl From<reqwest::Error> for OauthError {
    fn from(e: reqwest::Error) -> Self {
        OauthError::Reqwest(e.to_string())
    }
}

impl From<std::io::Error> for OauthError {
    fn from(e: std::io::Error) -> Self {
        OauthError::Io(e.to_string())
    }
}

impl From<anyhow::Error> for OauthError {
    fn from(e: anyhow::Error) -> Self {
        OauthError::Encryption(e.to_string())
    }
}

/// Newly-minted token set after exchange or refresh.
pub struct TokenSet {
    pub access_token: SecretBox<String>,
    /// `None` on a refresh that did not re-issue a refresh_token
    /// (Google sometimes omits it; caller should retain the previous one).
    pub refresh_token: Option<SecretBox<String>>,
    pub expires_at: DateTime<Utc>,
    pub scopes: Vec<String>,
}

/// Object-safe trait so multiple providers (gdrive, github, …) can be
/// stored behind `Arc<dyn OauthProvider>`.
#[async_trait::async_trait]
pub trait OauthProvider: Send + Sync {
    /// Lower-case provider name, e.g. `"gdrive"`. Used as the storage-key
    /// component and the route fragment.
    fn name(&self) -> &str;

    /// Exchange an authorization code for an initial token set.
    async fn exchange_code(
        &self,
        code: String,
        pkce_verifier: SecretBox<String>,
        redirect_uri: String,
    ) -> Result<TokenSet, OauthError>;

    /// Use a stored refresh token to obtain a fresh access token.
    async fn refresh(
        &self,
        refresh_token: &SecretBox<String>,
    ) -> Result<TokenSet, OauthError>;

    /// Tell the provider to invalidate the access token.
    async fn revoke(&self, access_token: &SecretBox<String>) -> Result<(), OauthError>;
}
