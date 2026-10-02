//! Persistence for the user-auth substrate, over any [`ColumnStore`].
//!
//! Layout under the `user_auth` family:
//!
//! - `user:<uuid>`              → bincode-encoded [`UserRecord`]
//! - `username:<lowercase>`     → uuid bytes (16) — case-insensitive lookup
//! - `session:<token-hash-hex>` → bincode-encoded [`SessionRecord`]
//! - `throttle:<lowercase>`     → bincode-encoded [`LoginThrottleRecord`] —
//!   persistent per-username failed-login state. See the throttle section
//!   below for the schema and the lockout policy.
//!
//! The username index is kept in lock-step with the user record by writing
//! both keys in one `put_many`, which is a single atomic batch on both
//! engines. Sessions are stored under the SHA-256 hex of the plaintext bearer
//! token so that bearer-token lookup is O(1) and never has to scan.
//!
//! All bincode encoding uses `bincode::serde` with `bincode::config::standard()`
//! to match the rest of the codebase.
//!
//! ## Why one family instead of `cf_login_throttle`
//!
//! The throttle data is tiny (one row per active attacker), shares lifecycle
//! and backup semantics with the user records it gates, and never needs a
//! distinct compression/tuning profile. Putting it under a `throttle:` key
//! prefix in the existing family keeps the family count flat and makes restore
//! atomic across both halves of the auth state.

use crate::storage::{ColumnStore, FamilyKeyValueStore, KeyValueStore};
use crate::user_auth::session::{SessionRecord, SessionToken};
use crate::user_auth::{session, AuthError, UserRecord, UserRole};

use chrono::{DateTime, Utc};
#[cfg(feature = "legacy-rocksdb")]
use rocksdb::{ColumnFamilyDescriptor, Options};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Family name for the auth rows.
pub const CF_USER_AUTH: &str = "user_auth";

/// Build the `ColumnFamilyDescriptor` for `user_auth`. Used by the shared
/// RocksDB bootstrap until that bootstrap declares the family itself.
#[cfg(feature = "legacy-rocksdb")]
pub fn cf_descriptor() -> ColumnFamilyDescriptor {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    ColumnFamilyDescriptor::new(CF_USER_AUTH, opts)
}

const USER_PREFIX: &[u8] = b"user:";
const USERNAME_PREFIX: &[u8] = b"username:";
const SESSION_PREFIX: &[u8] = b"session:";
const THROTTLE_PREFIX: &[u8] = b"throttle:";

/// Window over which failed login attempts accumulate before a lockout.
pub const THROTTLE_WINDOW_MINUTES: i64 = 15;
/// Number of failed attempts within [`THROTTLE_WINDOW_MINUTES`] that triggers
/// a lockout.
pub const THROTTLE_FAILURE_THRESHOLD: u32 = 5;
/// Duration of a triggered lockout.
pub const THROTTLE_LOCKOUT_MINUTES: i64 = 15;

/// Per-username failed-login state, persisted under `throttle:<lowercase>` so
/// a process restart cannot reset an active lockout.
///
/// This is the *persistent* throttle layer; the in-memory
/// [`crate::user_auth::runtime::LoginLimiter`] sits in front of it as a
/// cheap rate-limit check that never reaches the store.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoginThrottleRecord {
    /// Count of consecutive failed login attempts since the last success or
    /// since the failure window rolled over.
    pub failed_attempts: u32,
    /// If set and in the future, login is rejected as `TooManyAttempts`.
    pub locked_until: Option<DateTime<Utc>>,
    /// Wall-clock time of the most recent attempt counted into
    /// `failed_attempts`. Used to drop the counter when the window expires.
    pub last_attempt_at: Option<DateTime<Utc>>,
}

impl LoginThrottleRecord {
    /// True iff a current lockout is in effect at `now`.
    pub fn is_locked(&self, now: DateTime<Utc>) -> bool {
        matches!(self.locked_until, Some(t) if t > now)
    }
}

fn user_key(id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(USER_PREFIX.len() + 36);
    k.extend_from_slice(USER_PREFIX);
    k.extend_from_slice(id.to_string().as_bytes());
    k
}

fn username_key(name: &str) -> Vec<u8> {
    let lowered = name.to_lowercase();
    let mut k = Vec::with_capacity(USERNAME_PREFIX.len() + lowered.len());
    k.extend_from_slice(USERNAME_PREFIX);
    k.extend_from_slice(lowered.as_bytes());
    k
}

fn session_key(token_hash: &[u8; 32]) -> Vec<u8> {
    let mut k = Vec::with_capacity(SESSION_PREFIX.len() + 64);
    k.extend_from_slice(SESSION_PREFIX);
    k.extend_from_slice(hex::encode(token_hash).as_bytes());
    k
}

fn throttle_key(username: &str) -> Vec<u8> {
    let lowered = username.to_lowercase();
    let mut k = Vec::with_capacity(THROTTLE_PREFIX.len() + lowered.len());
    k.extend_from_slice(THROTTLE_PREFIX);
    k.extend_from_slice(lowered.as_bytes());
    k
}

fn encode<T: Serialize>(value: &T, what: &str) -> Result<Vec<u8>, AuthError> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| AuthError::internal(format!("encode {what}: {e}")))
}

fn decode<T: DeserializeOwned>(raw: &[u8], what: &str) -> Result<T, AuthError> {
    bincode::serde::decode_from_slice(raw, bincode::config::standard())
        .map(|(value, _)| value)
        .map_err(|e| AuthError::internal(format!("decode {what}: {e}")))
}

fn storage_err(e: anyhow::Error) -> AuthError {
    AuthError::internal(format!("user_auth store: {e}"))
}

/// Persistence layer over the `user_auth` family of a shared [`ColumnStore`].
#[derive(Clone)]
pub struct UserAuthStore {
    kv: Arc<FamilyKeyValueStore>,
}

impl UserAuthStore {
    /// Bind to the `user_auth` family of a shared store. On RocksDB the family
    /// must have been declared when the DB was opened, and a missing one fails
    /// here rather than on the first write; on redb the table is created.
    pub fn with_store(store: Arc<dyn ColumnStore>) -> Result<Self, AuthError> {
        let kv = FamilyKeyValueStore::new(store, CF_USER_AUTH).map_err(storage_err)?;
        Ok(Self { kv: Arc::new(kv) })
    }

    /// Bind a raw rocksdb handle; equivalent to `with_store` over its column store.
    #[cfg(feature = "legacy-rocksdb")]
    pub fn new(db: Arc<rocksdb::DB>) -> Result<Self, AuthError> {
        Self::with_store(Arc::new(
            crate::storage::legacy_rocksdb::RocksDbColumnStore::new(db),
        ))
    }

    // ── Users ───────────────────────────────────────────────────────────

    /// Insert a brand-new user record + its username index entry atomically.
    /// Returns [`AuthError::UsernameTaken`] if the username already maps to
    /// any user.
    pub fn create_user(&self, record: &UserRecord) -> Result<(), AuthError> {
        // Not a true transaction: concurrent registrations of the SAME username
        // are serialised by the handler's per-username rate-limiter slot.
        if self
            .kv
            .get_bytes(&username_key(&record.username))
            .map_err(storage_err)?
            .is_some()
        {
            return Err(AuthError::UsernameTaken);
        }
        let encoded = encode(record, "UserRecord")?;
        self.kv
            .put_many(&[
                (user_key(&record.id), encoded),
                (
                    username_key(&record.username),
                    record.id.as_bytes().to_vec(),
                ),
            ])
            .map_err(storage_err)
    }

    /// Persist updates to an existing user record. Username remapping is
    /// NOT supported by this method (the username index is left untouched);
    /// the auth surface never changes a user's username after creation.
    pub fn put_user(&self, record: &UserRecord) -> Result<(), AuthError> {
        let encoded = encode(record, "UserRecord")?;
        self.kv
            .put_bytes(&user_key(&record.id), &encoded)
            .map_err(storage_err)
    }

    pub fn get_user(&self, id: &Uuid) -> Result<Option<UserRecord>, AuthError> {
        match self.kv.get_bytes(&user_key(id)).map_err(storage_err)? {
            Some(raw) => decode(&raw, "UserRecord").map(Some),
            None => Ok(None),
        }
    }

    pub fn find_user_by_username(&self, name: &str) -> Result<Option<UserRecord>, AuthError> {
        let bytes = match self
            .kv
            .get_bytes(&username_key(name))
            .map_err(storage_err)?
        {
            Some(b) => b,
            None => return Ok(None),
        };
        let arr: [u8; 16] = bytes.as_slice().try_into().map_err(|_| {
            AuthError::internal(format!(
                "corrupt username index: expected 16 byte uuid, got {}",
                bytes.len()
            ))
        })?;
        self.get_user(&Uuid::from_bytes(arr))
    }

    /// True if at least one user exists in the store.
    ///
    /// Used by the registration handler to decide whether the caller can
    /// bootstrap the very first admin without prior credentials.
    pub fn has_any_user(&self) -> Result<bool, AuthError> {
        Ok(!self
            .kv
            .scan_prefix(USER_PREFIX, Some(1))
            .map_err(storage_err)?
            .is_empty())
    }

    /// Count users whose role is `Admin`. Used by `demote` to prevent the
    /// last remaining administrator from being downgraded out of existence.
    pub fn count_admins(&self) -> Result<usize, AuthError> {
        let mut admins = 0usize;
        for (_, raw) in self
            .kv
            .scan_prefix(USER_PREFIX, None)
            .map_err(storage_err)?
        {
            let record: UserRecord = decode(&raw, "UserRecord in admin sweep")?;
            if record.role == UserRole::Admin {
                admins += 1;
            }
        }
        Ok(admins)
    }

    // ── Login throttle ──────────────────────────────────────────────────

    /// Read the persistent login throttle row for `username`. Returns the
    /// default (all-zero, never-locked) record when no row exists yet.
    pub fn get_login_throttle(&self, username: &str) -> Result<LoginThrottleRecord, AuthError> {
        match self
            .kv
            .get_bytes(&throttle_key(username))
            .map_err(storage_err)?
        {
            Some(raw) => decode(&raw, "LoginThrottleRecord"),
            None => Ok(LoginThrottleRecord::default()),
        }
    }

    /// Persist the throttle row for `username`. The row is keyed by the
    /// lowercased username so it stays in lock-step with the username index.
    pub fn put_login_throttle(
        &self,
        username: &str,
        record: &LoginThrottleRecord,
    ) -> Result<(), AuthError> {
        let encoded = encode(record, "LoginThrottleRecord")?;
        self.kv
            .put_bytes(&throttle_key(username), &encoded)
            .map_err(storage_err)
    }

    /// Delete the throttle row for `username` — called on successful login
    /// so the user starts fresh next time.
    pub fn clear_login_throttle(&self, username: &str) -> Result<(), AuthError> {
        self.kv
            .delete_bytes(&throttle_key(username))
            .map_err(storage_err)
    }

    /// Register one failed login attempt for `username` at `now`, applying
    /// the lockout policy in [`LoginThrottleRecord`].
    ///
    /// Counter behaviour:
    ///   - If the last attempt is older than [`THROTTLE_WINDOW_MINUTES`],
    ///     the counter resets to 1 (we're past the window).
    ///   - Otherwise it increments by 1.
    ///   - When it reaches [`THROTTLE_FAILURE_THRESHOLD`], `locked_until`
    ///     becomes `now + THROTTLE_LOCKOUT_MINUTES`.
    ///
    /// Returns the updated record so callers can decide what to surface.
    pub fn record_login_failure(
        &self,
        username: &str,
        now: DateTime<Utc>,
    ) -> Result<LoginThrottleRecord, AuthError> {
        let mut row = self.get_login_throttle(username)?;
        let window = chrono::Duration::minutes(THROTTLE_WINDOW_MINUTES);
        let within_window = row
            .last_attempt_at
            .map(|t| now.signed_duration_since(t) < window)
            .unwrap_or(false);
        if !within_window {
            row.failed_attempts = 0;
        }
        row.failed_attempts = row.failed_attempts.saturating_add(1);
        row.last_attempt_at = Some(now);
        if row.failed_attempts >= THROTTLE_FAILURE_THRESHOLD {
            row.locked_until = Some(now + chrono::Duration::minutes(THROTTLE_LOCKOUT_MINUTES));
        }
        self.put_login_throttle(username, &row)?;
        Ok(row)
    }

    // ── Sessions ────────────────────────────────────────────────────────

    pub fn put_session(&self, record: &SessionRecord) -> Result<(), AuthError> {
        let encoded = encode(record, "SessionRecord")?;
        self.kv
            .put_bytes(&session_key(&record.token_hash), &encoded)
            .map_err(storage_err)
    }

    pub fn get_session_by_hash(
        &self,
        token_hash: &[u8; 32],
    ) -> Result<Option<SessionRecord>, AuthError> {
        match self
            .kv
            .get_bytes(&session_key(token_hash))
            .map_err(storage_err)?
        {
            Some(raw) => decode(&raw, "SessionRecord").map(Some),
            None => Ok(None),
        }
    }

    pub fn delete_session_by_hash(&self, token_hash: &[u8; 32]) -> Result<(), AuthError> {
        self.kv
            .delete_bytes(&session_key(token_hash))
            .map_err(storage_err)
    }

    /// Drop every session belonging to `user_id` EXCEPT the one whose hash
    /// equals `keep`. Used by the "log out all other devices" flow so the
    /// caller's own bearer token survives. Returns the count removed.
    pub fn delete_other_sessions_for_user(
        &self,
        user_id: &Uuid,
        keep: &[u8; 32],
    ) -> Result<usize, AuthError> {
        self.delete_sessions_where(|s| s.user_id == *user_id && s.token_hash != *keep)
    }

    /// Drop every session belonging to `user_id`. Returns the count removed.
    ///
    /// Used by the recovery flow to wipe stale sessions after password
    /// reset, and (in future work) by an admin-driven "kick user" endpoint.
    pub fn delete_all_sessions_for_user(&self, user_id: &Uuid) -> Result<usize, AuthError> {
        self.delete_sessions_where(|s| s.user_id == *user_id)
    }

    /// One atomic batch, so a failure part-way leaves every session in place
    /// rather than some of them.
    fn delete_sessions_where(
        &self,
        matches: impl Fn(&SessionRecord) -> bool,
    ) -> Result<usize, AuthError> {
        let mut doomed = Vec::new();
        for (key, raw) in self
            .kv
            .scan_prefix(SESSION_PREFIX, None)
            .map_err(storage_err)?
        {
            let record: SessionRecord = decode(&raw, "session in sweep")?;
            if matches(&record) {
                doomed.push(key);
            }
        }
        if !doomed.is_empty() {
            self.kv.delete_many(&doomed).map_err(storage_err)?;
        }
        Ok(doomed.len())
    }

    /// Validate a presented bearer token: look it up by hash, reject if
    /// missing or expired, refresh expiry on hit. Returns the matching
    /// user record on success.
    pub fn validate_and_refresh(
        &self,
        token: &SessionToken,
        now: DateTime<Utc>,
    ) -> Result<UserRecord, AuthError> {
        let hash = session::hash_token(&token.0)?;
        let mut record = self
            .get_session_by_hash(&hash)?
            .ok_or(AuthError::InvalidSession)?;
        if record.is_expired(now) {
            // The auth decision does not depend on the delete succeeding.
            let _ = self.delete_session_by_hash(&hash);
            return Err(AuthError::InvalidSession);
        }
        record.refresh(now);
        self.put_session(&record)?;
        self.get_user(&record.user_id)?
            .ok_or(AuthError::InvalidSession)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageBackend;
    use crate::storage::test_support::{assert_both_backends, column_stores};
    use crate::user_auth::{UserRecord, UserRole};
    use tempfile::TempDir;

    fn stores() -> Vec<(StorageBackend, UserAuthStore, TempDir)> {
        let stores: Vec<_> = column_stores(&[CF_USER_AUTH])
            .into_iter()
            .map(|(backend, cs, tmp)| (backend, UserAuthStore::with_store(cs).unwrap(), tmp))
            .collect();
        assert_both_backends(&stores);
        stores
    }

    fn fixture_user(name: &str, role: UserRole) -> UserRecord {
        UserRecord {
            id: Uuid::new_v4(),
            username: name.to_string(),
            password_hash: "$argon2id$v=19$m=32768,t=2,p=1$abc$xyz".to_string(),
            totp_secret_encrypted: None,
            totp_enrollment_pending: false,
            recovery_code_hashes: Vec::new(),
            role,
            created_at: Utc::now(),
            last_login_at: None,
        }
    }

    #[test]
    fn create_and_fetch_roundtrip() {
        for (backend, store, _tmp) in stores() {
            let user = fixture_user("Alice", UserRole::Admin);
            store.create_user(&user).unwrap();

            let by_id = store.get_user(&user.id).unwrap().unwrap();
            assert_eq!(by_id.id, user.id, "{backend}");
            assert_eq!(by_id.username, "Alice", "{backend}");

            let by_name = store.find_user_by_username("ALICE").unwrap().unwrap();
            assert_eq!(
                by_name.id, user.id,
                "{backend}: lookup must be case-insensitive"
            );

            assert!(store.has_any_user().unwrap(), "{backend}");
        }
    }

    #[test]
    fn create_user_writes_record_and_index_together() {
        for (backend, store, _tmp) in stores() {
            let user = fixture_user("carol", UserRole::User);
            store.create_user(&user).unwrap();
            assert!(
                store.get_user(&user.id).unwrap().is_some(),
                "{backend}: record missing"
            );
            assert_eq!(
                store.find_user_by_username("carol").unwrap().map(|u| u.id),
                Some(user.id),
                "{backend}: index missing or pointing elsewhere"
            );
        }
    }

    #[test]
    fn duplicate_username_rejected() {
        for (backend, store, _tmp) in stores() {
            let u1 = fixture_user("bob", UserRole::User);
            let u2 = fixture_user("BOB", UserRole::User);
            store.create_user(&u1).unwrap();
            let err = store.create_user(&u2).unwrap_err();
            assert!(
                matches!(err, AuthError::UsernameTaken),
                "{backend}: {err:?}"
            );
        }
    }

    #[test]
    fn has_any_user_initially_false() {
        for (backend, store, _tmp) in stores() {
            assert!(!store.has_any_user().unwrap(), "{backend}");
        }
    }

    /// A row under a different prefix must not satisfy the user scan. The
    /// rocksdb iterator this replaced ran past its prefix and relied on a
    /// `starts_with` break to stop.
    #[test]
    fn has_any_user_ignores_sessions_and_throttle_rows() {
        for (backend, store, _tmp) in stores() {
            let now = Utc::now();
            let (_, orphan) = session::issue(Uuid::new_v4(), now);
            store.put_session(&orphan).unwrap();
            store.record_login_failure("nobody", now).unwrap();
            assert!(!store.has_any_user().unwrap(), "{backend}");
            assert_eq!(store.count_admins().unwrap(), 0, "{backend}");
        }
    }

    #[test]
    fn delete_all_sessions_only_removes_target_user() {
        for (backend, store, _tmp) in stores() {
            let alice = fixture_user("alice", UserRole::User);
            let bob = fixture_user("bob", UserRole::User);
            store.create_user(&alice).unwrap();
            store.create_user(&bob).unwrap();

            let now = Utc::now();
            let (_, a1) = session::issue(alice.id, now);
            let (_, a2) = session::issue(alice.id, now);
            let (_, b1) = session::issue(bob.id, now);
            store.put_session(&a1).unwrap();
            store.put_session(&a2).unwrap();
            store.put_session(&b1).unwrap();

            assert_eq!(
                store.delete_all_sessions_for_user(&alice.id).unwrap(),
                2,
                "{backend}"
            );
            assert!(
                store.get_session_by_hash(&b1.token_hash).unwrap().is_some(),
                "{backend}"
            );
            assert!(
                store.get_session_by_hash(&a1.token_hash).unwrap().is_none(),
                "{backend}"
            );
            assert!(
                store.get_session_by_hash(&a2.token_hash).unwrap().is_none(),
                "{backend}"
            );
        }
    }

    #[test]
    fn delete_other_sessions_keeps_caller_token() {
        for (backend, store, _tmp) in stores() {
            let alice = fixture_user("alice", UserRole::User);
            let bob = fixture_user("bob", UserRole::User);
            store.create_user(&alice).unwrap();
            store.create_user(&bob).unwrap();

            let now = Utc::now();
            let (_, a_keep) = session::issue(alice.id, now);
            let (_, a_drop) = session::issue(alice.id, now);
            let (_, b_keep) = session::issue(bob.id, now);
            store.put_session(&a_keep).unwrap();
            store.put_session(&a_drop).unwrap();
            store.put_session(&b_keep).unwrap();

            let n = store
                .delete_other_sessions_for_user(&alice.id, &a_keep.token_hash)
                .unwrap();
            assert_eq!(n, 1, "{backend}: only the second alice session should go");
            assert!(
                store
                    .get_session_by_hash(&a_keep.token_hash)
                    .unwrap()
                    .is_some(),
                "{backend}"
            );
            assert!(
                store
                    .get_session_by_hash(&a_drop.token_hash)
                    .unwrap()
                    .is_none(),
                "{backend}"
            );
            assert!(
                store
                    .get_session_by_hash(&b_keep.token_hash)
                    .unwrap()
                    .is_some(),
                "{backend}"
            );
        }
    }

    #[test]
    fn deleting_sessions_for_a_user_with_none_writes_nothing() {
        for (backend, store, _tmp) in stores() {
            let (_, other) = session::issue(Uuid::new_v4(), Utc::now());
            store.put_session(&other).unwrap();
            assert_eq!(
                store.delete_all_sessions_for_user(&Uuid::new_v4()).unwrap(),
                0,
                "{backend}"
            );
            assert!(
                store
                    .get_session_by_hash(&other.token_hash)
                    .unwrap()
                    .is_some(),
                "{backend}"
            );
        }
    }

    #[test]
    fn count_admins_walks_user_records() {
        for (backend, store, _tmp) in stores() {
            assert_eq!(store.count_admins().unwrap(), 0, "{backend}");
            store
                .create_user(&fixture_user("root", UserRole::Admin))
                .unwrap();
            store
                .create_user(&fixture_user("rooty", UserRole::Admin))
                .unwrap();
            store
                .create_user(&fixture_user("peon", UserRole::User))
                .unwrap();
            assert_eq!(store.count_admins().unwrap(), 2, "{backend}");
        }
    }

    #[test]
    fn login_throttle_default_is_unlocked() {
        for (backend, store, _tmp) in stores() {
            let row = store.get_login_throttle("ghost").unwrap();
            assert_eq!(row, LoginThrottleRecord::default(), "{backend}");
            assert!(!row.is_locked(Utc::now()), "{backend}");
        }
    }

    #[test]
    fn record_login_failure_locks_after_threshold() {
        for (backend, store, _tmp) in stores() {
            let now = Utc::now();
            for _ in 0..(THROTTLE_FAILURE_THRESHOLD - 1) {
                let row = store.record_login_failure("alice", now).unwrap();
                assert!(row.locked_until.is_none(), "{backend}: locked too early");
            }
            let final_row = store.record_login_failure("alice", now).unwrap();
            assert_eq!(
                final_row.failed_attempts, THROTTLE_FAILURE_THRESHOLD,
                "{backend}"
            );
            assert!(
                final_row.locked_until.expect("must be locked") > now,
                "{backend}"
            );
            assert!(final_row.is_locked(now), "{backend}");
        }
    }

    #[test]
    fn record_login_failure_resets_after_window() {
        for (backend, store, _tmp) in stores() {
            let t0 = Utc::now();
            store.record_login_failure("alice", t0).unwrap();
            store.record_login_failure("alice", t0).unwrap();
            assert_eq!(
                store.get_login_throttle("alice").unwrap().failed_attempts,
                2,
                "{backend}"
            );

            let t1 = t0 + chrono::Duration::minutes(THROTTLE_WINDOW_MINUTES + 1);
            let after = store.record_login_failure("alice", t1).unwrap();
            assert_eq!(after.failed_attempts, 1, "{backend}");
            assert!(after.locked_until.is_none(), "{backend}");
        }
    }

    #[test]
    fn clear_login_throttle_drops_row() {
        for (backend, store, _tmp) in stores() {
            store.record_login_failure("alice", Utc::now()).unwrap();
            assert_eq!(
                store.get_login_throttle("alice").unwrap().failed_attempts,
                1,
                "{backend}"
            );
            store.clear_login_throttle("alice").unwrap();
            assert_eq!(
                store.get_login_throttle("alice").unwrap(),
                LoginThrottleRecord::default(),
                "{backend}"
            );
        }
    }

    /// The lockout lives in the store, not in the `UserAuthStore` value, so a
    /// fresh value over the same handle — what a restart builds — still sees it.
    #[test]
    fn login_throttle_survives_a_rebuilt_store() {
        for (backend, cs, _tmp) in column_stores(&[CF_USER_AUTH]) {
            let now = Utc::now();
            {
                let store = UserAuthStore::with_store(cs.clone()).unwrap();
                for _ in 0..THROTTLE_FAILURE_THRESHOLD {
                    store.record_login_failure("alice", now).unwrap();
                }
            }
            let rebuilt = UserAuthStore::with_store(cs).unwrap();
            assert!(
                rebuilt.get_login_throttle("alice").unwrap().is_locked(now),
                "{backend}"
            );
        }
    }

    /// Reopening the redb FILE, not just the handle: the backend this store
    /// has never run on in production is the one whose durability is unproven.
    #[cfg(feature = "storage-redb")]
    #[test]
    fn login_throttle_persists_across_redb_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("auth.redb");
        let now = Utc::now();
        {
            let db = crate::storage::redb::create_redb_database_for_test(&path).unwrap();
            let store =
                UserAuthStore::with_store(Arc::new(crate::storage::redb::RedbColumnStore::new(db)))
                    .unwrap();
            for _ in 0..THROTTLE_FAILURE_THRESHOLD {
                store.record_login_failure("alice", now).unwrap();
            }
        }
        let db = crate::storage::redb::create_redb_database_for_test(&path).unwrap();
        let store =
            UserAuthStore::with_store(Arc::new(crate::storage::redb::RedbColumnStore::new(db)))
                .unwrap();
        assert!(
            store.get_login_throttle("alice").unwrap().is_locked(now),
            "lockout must survive restart"
        );
    }

    /// Fail-at-construction differs by engine by design: a RocksDB family must
    /// be declared at open, so a missing one is a configuration error; a redb
    /// table is created on first bind.
    #[test]
    fn with_store_fails_on_an_undeclared_rocksdb_family_and_creates_the_redb_table() {
        let stores = column_stores(&[]);
        assert_both_backends(&stores);
        for (backend, cs, _tmp) in stores {
            let bound = UserAuthStore::with_store(cs);
            match backend {
                StorageBackend::RocksDb => {
                    assert!(bound.is_err(), "rocksdb bound an undeclared family")
                }
                StorageBackend::Redb => assert!(bound.unwrap().has_any_user().is_ok()),
            }
        }
    }
}
