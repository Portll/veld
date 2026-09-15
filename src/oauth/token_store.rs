//! RocksDB-backed token persistence with entry-atomic refresh dedup.
//!
//! Records live in the `cf_oauth_tokens` column family (see
//! [`super::cf_descriptor`]) under the `token:<user_id>:<provider>`
//! prefix. On-disk tokens are encrypted via
//! [`crate::encryption::FieldEncryptor`]; in-memory plaintext is held by
//! [`super::DecryptedOauthRecord`] which uses
//! [`secrecy::SecretBox<String>`] for redaction.
//!
//! # Concurrent refresh
//!
//! `bearer_for` is the hot path. When the cached token is fresh
//! (expires > 60s from now) it returns immediately. Otherwise it
//! serializes refresh per `(user_id, provider)` key via
//! [`dashmap::DashMap`]'s entry-atomic API: two concurrent callers
//! find/insert the same `Arc<tokio::sync::Mutex<()>>`, one acquires the
//! lock and performs the HTTP refresh, the other waits and then re-reads
//! the fresh record.

use chrono::{Duration as ChronoDuration, Utc};
use dashmap::DashMap;
use rocksdb::{ColumnFamily, DB};
use secrecy::{ExposeSecret, SecretBox};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::encryption::FieldEncryptor;

use super::{
    decode_oauth_record_with_fallback, DecryptedOauthRecord, OauthError, OauthProvider,
    OauthRecord, TokenSet,
};

/// Column family the OAuth subsystem stores into.
pub const CF_OAUTH_TOKENS: &str = "oauth_tokens";

/// Storage-key prefix for per-user, per-provider records.
pub const KEY_PREFIX_TOKEN: &[u8] = b"token:";

/// How close to expiry a token must be before `bearer_for` triggers a
/// proactive refresh. Tightened from a 5-minute buffer (r2) to 60s
/// after analysis — most session lifetimes are short, so a longer buffer
/// wastes refreshes; 60s still covers small clock skew.
pub const PROACTIVE_REFRESH_BUFFER_SECS: i64 = 60;

/// Persistence + refresh-dedup substrate for OAuth tokens.
pub struct TokenStore {
    db: Arc<DB>,
    encryptor: Arc<FieldEncryptor>,
    /// Per-(user, provider) refresh mutexes. Entry-atomic via DashMap.
    refresh_locks: DashMap<(String, String), Arc<Mutex<()>>>,
}

impl TokenStore {
    /// Wrap a shared DB + encryptor handle. Verifies the
    /// [`CF_OAUTH_TOKENS`] CF is registered so misconfiguration shows
    /// up at construction time, not on the first put.
    pub fn new(db: Arc<DB>, encryptor: Arc<FieldEncryptor>) -> Result<Self, OauthError> {
        if db.cf_handle(CF_OAUTH_TOKENS).is_none() {
            return Err(OauthError::Rocksdb(format!(
                "shared DB is missing the '{}' column family",
                CF_OAUTH_TOKENS
            )));
        }
        Ok(Self {
            db,
            encryptor,
            refresh_locks: DashMap::new(),
        })
    }

    fn cf(&self) -> Result<&ColumnFamily, OauthError> {
        self.db
            .cf_handle(CF_OAUTH_TOKENS)
            .ok_or_else(|| OauthError::Rocksdb(format!("'{CF_OAUTH_TOKENS}' CF disappeared")))
    }

    fn key(user_id: &str, provider: &str) -> Vec<u8> {
        let mut k =
            Vec::with_capacity(KEY_PREFIX_TOKEN.len() + user_id.len() + 1 + provider.len());
        k.extend_from_slice(KEY_PREFIX_TOKEN);
        k.extend_from_slice(user_id.as_bytes());
        k.push(b':');
        k.extend_from_slice(provider.as_bytes());
        k
    }

    /// Read the on-disk record and decrypt the token fields. Returns
    /// [`OauthError::NoRecord`] if absent or [`OauthError::MalformedRecord`]
    /// if any ciphertext field is empty (defensive — empty ciphertext
    /// usually indicates a write that failed half-way).
    pub fn read_decrypted(
        &self,
        user_id: &str,
        provider: &str,
    ) -> Result<DecryptedOauthRecord, OauthError> {
        let cf = self.cf()?;
        let bytes = self
            .db
            .get_cf(cf, Self::key(user_id, provider))?
            .ok_or(OauthError::NoRecord)?;
        let (rec, _is_legacy) = decode_oauth_record_with_fallback(&bytes)?;
        if rec.access_token_ct.is_empty() || rec.refresh_token_ct.is_empty() {
            return Err(OauthError::MalformedRecord);
        }
        let access_plain = self.encryptor.decrypt_content(&rec.access_token_ct)?;
        let refresh_plain = self.encryptor.decrypt_content(&rec.refresh_token_ct)?;
        Ok(DecryptedOauthRecord {
            veld_user_id: rec.veld_user_id,
            provider: rec.provider,
            access_token: SecretBox::new(Box::new(access_plain)),
            refresh_token: SecretBox::new(Box::new(refresh_plain)),
            expires_at: rec.expires_at,
            scopes: rec.scopes,
            obtained_at: rec.obtained_at,
        })
    }

    /// Encrypt the token fields and persist the record under
    /// `token:<user_id>:<provider>` in [`CF_OAUTH_TOKENS`].
    pub fn write_encrypted(&self, rec: &DecryptedOauthRecord) -> Result<(), OauthError> {
        let cf = self.cf()?;
        let access_ct = self
            .encryptor
            .encrypt_content(rec.access_token.expose_secret())?;
        let refresh_ct = self
            .encryptor
            .encrypt_content(rec.refresh_token.expose_secret())?;
        let on_disk = OauthRecord {
            veld_user_id: rec.veld_user_id.clone(),
            provider: rec.provider.clone(),
            access_token_ct: access_ct,
            refresh_token_ct: refresh_ct,
            expires_at: rec.expires_at,
            scopes: rec.scopes.clone(),
            obtained_at: rec.obtained_at,
        };
        let bytes = bincode::serde::encode_to_vec(&on_disk, bincode::config::standard())?;
        self.db
            .put_cf(cf, Self::key(&rec.veld_user_id, &rec.provider), &bytes)?;
        Ok(())
    }

    /// Delete the stored record for `(user_id, provider)`. Used by the
    /// CLI `logout` flow after a successful provider-side revoke.
    pub fn delete(&self, user_id: &str, provider: &str) -> Result<(), OauthError> {
        let cf = self.cf()?;
        self.db.delete_cf(cf, Self::key(user_id, provider))?;
        // Drop the refresh-mutex entry too; it's a slow-growing leak otherwise.
        self.refresh_locks
            .remove(&(user_id.to_string(), provider.to_string()));
        Ok(())
    }

    /// Return a usable access token for `(user_id, provider)`. Refreshes
    /// transparently when the stored token has < `PROACTIVE_REFRESH_BUFFER_SECS`
    /// of life remaining. Concurrent callers for the same key are
    /// serialized: only one HTTP refresh hits the provider.
    pub async fn bearer_for(
        &self,
        user_id: &str,
        provider: &dyn OauthProvider,
    ) -> Result<SecretBox<String>, OauthError> {
        let provider_name = provider.name().to_string();
        let rec = self.read_decrypted(user_id, &provider_name)?;
        if !needs_refresh(&rec) {
            return Ok(rec.access_token);
        }

        // Get-or-create the per-key mutex via DashMap's entry-atomic API
        // so we cannot race in two threads creating different Arcs.
        let lock = self
            .refresh_locks
            .entry((user_id.to_string(), provider_name.clone()))
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = lock.lock().await;

        // Re-read under the lock — another caller may have refreshed
        // already while we were waiting.
        let rec = self.read_decrypted(user_id, &provider_name)?;
        if !needs_refresh(&rec) {
            return Ok(rec.access_token);
        }

        let new_tokens = provider.refresh(&rec.refresh_token).await?;
        // Google sometimes omits a new refresh_token on refresh — fall
        // back to the previous one. The SecretBox<String> can't be
        // cloned, so re-wrap a fresh String built from the previous
        // plaintext.
        let new_refresh = match new_tokens.refresh_token {
            Some(s) => s,
            None => SecretBox::new(Box::new(rec.refresh_token.expose_secret().to_string())),
        };
        let updated = DecryptedOauthRecord {
            veld_user_id: rec.veld_user_id,
            provider: rec.provider,
            access_token: SecretBox::new(Box::new(
                new_tokens.access_token.expose_secret().to_string(),
            )),
            refresh_token: new_refresh,
            expires_at: new_tokens.expires_at,
            scopes: new_tokens.scopes,
            obtained_at: Utc::now(),
        };
        self.write_encrypted(&updated)?;
        Ok(updated.access_token)
    }

    /// Persist a brand-new token set from a successful `exchange_code`.
    pub fn install_initial(
        &self,
        user_id: &str,
        provider: &str,
        tokens: TokenSet,
    ) -> Result<(), OauthError> {
        let refresh = tokens.refresh_token.ok_or_else(|| {
            OauthError::Oauth2(
                "exchange_code did not return a refresh_token — Google requires \
                 access_type=offline + prompt=consent"
                    .to_string(),
            )
        })?;
        let rec = DecryptedOauthRecord {
            veld_user_id: user_id.to_string(),
            provider: provider.to_string(),
            access_token: tokens.access_token,
            refresh_token: refresh,
            expires_at: tokens.expires_at,
            scopes: tokens.scopes,
            obtained_at: Utc::now(),
        };
        self.write_encrypted(&rec)
    }
}

fn needs_refresh(rec: &DecryptedOauthRecord) -> bool {
    let remaining = rec.expires_at - Utc::now();
    remaining <= ChronoDuration::seconds(PROACTIVE_REFRESH_BUFFER_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn make_store() -> (TokenStore, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let descriptors = vec![super::super::cf_descriptor()];
        let db = DB::open_cf_descriptors(&opts, dir.path(), descriptors).expect("open DB");
        // Fixed-bytes test key — never used outside this module.
        let key: [u8; 32] = *b"01234567890123456789012345678901";
        let encryptor = FieldEncryptor::new(&key);
        let store = TokenStore::new(Arc::new(db), Arc::new(encryptor)).expect("token store");
        (store, dir)
    }

    fn sample_record(user_id: &str, provider: &str) -> DecryptedOauthRecord {
        DecryptedOauthRecord {
            veld_user_id: user_id.to_string(),
            provider: provider.to_string(),
            access_token: SecretBox::new(Box::new("access-XYZ".to_string())),
            refresh_token: SecretBox::new(Box::new("refresh-XYZ".to_string())),
            expires_at: Utc::now() + ChronoDuration::hours(1),
            scopes: vec!["drive.readonly".to_string()],
            obtained_at: Utc::now(),
        }
    }

    #[test]
    fn round_trip_record_via_field_encryptor() {
        let (store, _dir) = make_store();
        let rec = sample_record("alice", "gdrive");
        store.write_encrypted(&rec).unwrap();
        let got = store.read_decrypted("alice", "gdrive").unwrap();
        assert_eq!(got.veld_user_id, "alice");
        assert_eq!(got.provider, "gdrive");
        assert_eq!(got.access_token.expose_secret(), "access-XYZ");
        assert_eq!(got.refresh_token.expose_secret(), "refresh-XYZ");
        assert_eq!(got.scopes, vec!["drive.readonly".to_string()]);
    }

    #[test]
    fn no_record_returns_no_record_error() {
        let (store, _dir) = make_store();
        let err = store.read_decrypted("bob", "gdrive").unwrap_err();
        assert!(
            matches!(err, OauthError::NoRecord),
            "expected NoRecord, got {err:?}"
        );
    }

    #[test]
    fn ciphertext_not_present_in_raw_bytes() {
        let (store, _dir) = make_store();
        let rec = sample_record("alice", "gdrive");
        store.write_encrypted(&rec).unwrap();
        let cf = store.cf().unwrap();
        let raw = store
            .db
            .get_cf(cf, TokenStore::key("alice", "gdrive"))
            .unwrap()
            .unwrap();
        let raw_str = String::from_utf8_lossy(&raw);
        assert!(
            !raw_str.contains("access-XYZ"),
            "access-XYZ plaintext leaked into RocksDB bytes"
        );
        assert!(
            !raw_str.contains("refresh-XYZ"),
            "refresh-XYZ plaintext leaked into RocksDB bytes"
        );
    }

    #[test]
    fn delete_removes_record_and_refresh_lock() {
        let (store, _dir) = make_store();
        let rec = sample_record("alice", "gdrive");
        store.write_encrypted(&rec).unwrap();
        store
            .refresh_locks
            .insert(("alice".into(), "gdrive".into()), Arc::new(Mutex::new(())));
        store.delete("alice", "gdrive").unwrap();
        assert!(matches!(
            store.read_decrypted("alice", "gdrive"),
            Err(OauthError::NoRecord)
        ));
        assert!(!store
            .refresh_locks
            .contains_key(&("alice".into(), "gdrive".into())));
    }

    #[test]
    fn malformed_record_when_ciphertext_empty() {
        let (store, _dir) = make_store();
        // Hand-craft a record with empty ciphertext and put it directly.
        let bad = OauthRecord {
            veld_user_id: "alice".into(),
            provider: "gdrive".into(),
            access_token_ct: Vec::new(),
            refresh_token_ct: Vec::new(),
            expires_at: Utc::now() + ChronoDuration::hours(1),
            scopes: vec![],
            obtained_at: Utc::now(),
        };
        let bytes =
            bincode::serde::encode_to_vec(&bad, bincode::config::standard()).unwrap();
        store
            .db
            .put_cf(store.cf().unwrap(), TokenStore::key("alice", "gdrive"), &bytes)
            .unwrap();
        let err = store.read_decrypted("alice", "gdrive").unwrap_err();
        assert!(matches!(err, OauthError::MalformedRecord));
    }
}
