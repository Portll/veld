//! RocksDB-backed persistence for the user-auth substrate.
//!
//! Layout in the shared RocksDB instance under the `user_auth` column family:
//!
//! - `user:<uuid>`              → bincode-encoded [`UserRecord`]
//! - `username:<lowercase>`     → uuid bytes (16) — case-insensitive lookup
//! - `session:<token-hash-hex>` → bincode-encoded [`SessionRecord`]
//! - `throttle:<lowercase>`     → bincode-encoded [`LoginThrottleRecord`] —
//!   persistent per-username failed-login state. See the throttle section
//!   below for the schema and the lockout policy.
//!
//! The username index is kept in lock-step with the user record by writing
//! both keys in a single `WriteBatch`. Sessions are stored under the
//! SHA-256 hex of the plaintext bearer token so that bearer-token lookup is
//! O(1) and never has to scan.
//!
//! All bincode encoding uses `bincode::serde` with `bincode::config::standard()`
//! to match the rest of the codebase.
//!
//! ## Why one CF instead of `cf_login_throttle`
//!
//! The throttle data is tiny (one row per active attacker), shares lifecycle
//! and backup semantics with the user records it gates, and never needs a
//! distinct compression/tuning profile. Putting it under a `throttle:` key
//! prefix in the existing CF keeps the CF count flat, makes restore atomic
//! across both halves of the auth state, and avoids touching the shared-DB
//! bootstrap path in `MultiUserMemoryManager`.

use crate::user_auth::session::{SessionRecord, SessionToken};
use crate::user_auth::{session, AuthError, UserRecord, UserRole};

use anyhow::anyhow;
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, IteratorMode, Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Column family name in the shared RocksDB instance.
pub const CF_USER_AUTH: &str = "user_auth";

/// Build the `ColumnFamilyDescriptor` for `cf_user_auth`. Used by the shared
/// DB bootstrap in `MultiUserMemoryManager`.
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
/// cheap rate-limit check that never reaches RocksDB.
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

/// Persistence layer over the shared DB's `user_auth` CF.
#[derive(Clone)]
pub struct UserAuthStore {
    db: Arc<DB>,
}

impl UserAuthStore {
    /// Wrap a shared DB. The caller is responsible for having declared
    /// [`CF_USER_AUTH`] in the DB's column-family list at open time.
    pub fn new(db: Arc<DB>) -> Result<Self, AuthError> {
        // Eagerly verify the CF is present so misconfiguration is surfaced
        // at construction time rather than on the first write.
        if db.cf_handle(CF_USER_AUTH).is_none() {
            return Err(AuthError::internal(format!(
                "shared DB is missing the '{}' column family",
                CF_USER_AUTH
            )));
        }
        Ok(Self { db })
    }

    fn cf(&self) -> Result<&ColumnFamily, AuthError> {
        self.db.cf_handle(CF_USER_AUTH).ok_or_else(|| {
            AuthError::internal(format!("'{}' CF disappeared at runtime", CF_USER_AUTH))
        })
    }

    // ── Users ───────────────────────────────────────────────────────────

    /// Insert a brand-new user record + its username index entry atomically.
    /// Returns [`AuthError::UsernameTaken`] if the username already maps to
    /// any user.
    pub fn create_user(&self, record: &UserRecord) -> Result<(), AuthError> {
        let cf = self.cf()?;
        // Pre-check: username index. Not a true transaction, but the only
        // races we have to worry about are concurrent registrations of the
        // SAME username — and the higher-level handler already takes a
        // per-username lock around register/login (rate limiter slot).
        if self
            .db
            .get_cf(cf, username_key(&record.username))
            .map_err(rocks_err)?
            .is_some()
        {
            return Err(AuthError::UsernameTaken);
        }

        let encoded = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| AuthError::internal(format!("encode UserRecord: {e}")))?;

        let mut batch = WriteBatch::default();
        batch.put_cf(cf, user_key(&record.id), &encoded);
        batch.put_cf(cf, username_key(&record.username), record.id.as_bytes());
        self.db.write(batch).map_err(rocks_err)?;
        Ok(())
    }

    /// Persist updates to an existing user record. Username remapping is
    /// NOT supported by this method (the username index is left untouched);
    /// the auth surface never changes a user's username after creation.
    pub fn put_user(&self, record: &UserRecord) -> Result<(), AuthError> {
        let cf = self.cf()?;
        let encoded = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| AuthError::internal(format!("encode UserRecord: {e}")))?;
        self.db
            .put_cf(cf, user_key(&record.id), &encoded)
            .map_err(rocks_err)?;
        Ok(())
    }

    pub fn get_user(&self, id: &Uuid) -> Result<Option<UserRecord>, AuthError> {
        let cf = self.cf()?;
        let raw = match self.db.get_cf(cf, user_key(id)).map_err(rocks_err)? {
            Some(b) => b,
            None => return Ok(None),
        };
        let (record, _): (UserRecord, _) =
            bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                .map_err(|e| AuthError::internal(format!("decode UserRecord: {e}")))?;
        Ok(Some(record))
    }

    pub fn find_user_by_username(&self, name: &str) -> Result<Option<UserRecord>, AuthError> {
        let cf = self.cf()?;
        let bytes = match self
            .db
            .get_cf(cf, username_key(name))
            .map_err(rocks_err)?
        {
            Some(b) => b,
            None => return Ok(None),
        };
        if bytes.len() != 16 {
            return Err(AuthError::internal(format!(
                "corrupt username index: expected 16 byte uuid, got {}",
                bytes.len()
            )));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes);
        let id = Uuid::from_bytes(arr);
        self.get_user(&id)
    }

    /// True if at least one user exists in the store.
    ///
    /// Used by the registration handler to decide whether the caller can
    /// bootstrap the very first admin without prior credentials.
    pub fn has_any_user(&self) -> Result<bool, AuthError> {
        let cf = self.cf()?;
        let mut iter = self.db.iterator_cf(cf, IteratorMode::From(USER_PREFIX, rocksdb::Direction::Forward));
        if let Some(item) = iter.next() {
            let (k, _) = item.map_err(rocks_err)?;
            return Ok(k.starts_with(USER_PREFIX));
        }
        Ok(false)
    }

    /// Count users whose role is `Admin`. Used by `demote` to prevent the
    /// last remaining administrator from being downgraded out of existence.
    pub fn count_admins(&self) -> Result<usize, AuthError> {
        let cf = self.cf()?;
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(USER_PREFIX, rocksdb::Direction::Forward),
        );
        let mut admins = 0usize;
        for item in iter {
            let (k, v) = item.map_err(rocks_err)?;
            if !k.starts_with(USER_PREFIX) {
                break;
            }
            let (record, _): (UserRecord, _) =
                bincode::serde::decode_from_slice(&v, bincode::config::standard())
                    .map_err(|e| AuthError::internal(format!("decode UserRecord in admin sweep: {e}")))?;
            if record.role == UserRole::Admin {
                admins += 1;
            }
        }
        Ok(admins)
    }

    // ── Login throttle ──────────────────────────────────────────────────

    /// Read the persistent login throttle row for `username`. Returns the
    /// default (all-zero, never-locked) record when no row exists yet.
    pub fn get_login_throttle(
        &self,
        username: &str,
    ) -> Result<LoginThrottleRecord, AuthError> {
        let cf = self.cf()?;
        let raw = match self
            .db
            .get_cf(cf, throttle_key(username))
            .map_err(rocks_err)?
        {
            Some(b) => b,
            None => return Ok(LoginThrottleRecord::default()),
        };
        let (record, _): (LoginThrottleRecord, _) =
            bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                .map_err(|e| AuthError::internal(format!("decode LoginThrottleRecord: {e}")))?;
        Ok(record)
    }

    /// Persist the throttle row for `username`. The row is keyed by the
    /// lowercased username so it stays in lock-step with the username index.
    pub fn put_login_throttle(
        &self,
        username: &str,
        record: &LoginThrottleRecord,
    ) -> Result<(), AuthError> {
        let cf = self.cf()?;
        let encoded = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| AuthError::internal(format!("encode LoginThrottleRecord: {e}")))?;
        self.db
            .put_cf(cf, throttle_key(username), &encoded)
            .map_err(rocks_err)?;
        Ok(())
    }

    /// Delete the throttle row for `username` — called on successful login
    /// so the user starts fresh next time.
    pub fn clear_login_throttle(&self, username: &str) -> Result<(), AuthError> {
        let cf = self.cf()?;
        self.db
            .delete_cf(cf, throttle_key(username))
            .map_err(rocks_err)?;
        Ok(())
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
        let cf = self.cf()?;
        let encoded = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| AuthError::internal(format!("encode SessionRecord: {e}")))?;
        self.db
            .put_cf(cf, session_key(&record.token_hash), &encoded)
            .map_err(rocks_err)?;
        Ok(())
    }

    pub fn get_session_by_hash(
        &self,
        token_hash: &[u8; 32],
    ) -> Result<Option<SessionRecord>, AuthError> {
        let cf = self.cf()?;
        let raw = match self
            .db
            .get_cf(cf, session_key(token_hash))
            .map_err(rocks_err)?
        {
            Some(b) => b,
            None => return Ok(None),
        };
        let (record, _): (SessionRecord, _) =
            bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                .map_err(|e| AuthError::internal(format!("decode SessionRecord: {e}")))?;
        Ok(Some(record))
    }

    pub fn delete_session_by_hash(&self, token_hash: &[u8; 32]) -> Result<(), AuthError> {
        let cf = self.cf()?;
        self.db
            .delete_cf(cf, session_key(token_hash))
            .map_err(rocks_err)?;
        Ok(())
    }

    /// Drop every session belonging to `user_id` EXCEPT the one whose hash
    /// equals `keep`. Used by the "log out all other devices" flow so the
    /// caller's own bearer token survives. Returns the count removed.
    pub fn delete_other_sessions_for_user(
        &self,
        user_id: &Uuid,
        keep: &[u8; 32],
    ) -> Result<usize, AuthError> {
        let cf = self.cf()?;
        let mut removed = 0usize;
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(SESSION_PREFIX, rocksdb::Direction::Forward),
        );
        let mut batch = WriteBatch::default();
        for item in iter {
            let (k, v) = item.map_err(rocks_err)?;
            if !k.starts_with(SESSION_PREFIX) {
                break;
            }
            let (record, _): (SessionRecord, _) =
                bincode::serde::decode_from_slice(&v, bincode::config::standard())
                    .map_err(|e| AuthError::internal(format!("decode session in sweep: {e}")))?;
            if record.user_id == *user_id && record.token_hash != *keep {
                batch.delete_cf(cf, &k);
                removed += 1;
            }
        }
        if removed > 0 {
            self.db.write(batch).map_err(rocks_err)?;
        }
        Ok(removed)
    }

    /// Drop every session belonging to `user_id`. Returns the count removed.
    ///
    /// Used by the recovery flow to wipe stale sessions after password
    /// reset, and (in future work) by an admin-driven "kick user" endpoint.
    pub fn delete_all_sessions_for_user(&self, user_id: &Uuid) -> Result<usize, AuthError> {
        let cf = self.cf()?;
        let mut removed = 0usize;
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(SESSION_PREFIX, rocksdb::Direction::Forward),
        );
        let mut batch = WriteBatch::default();
        for item in iter {
            let (k, v) = item.map_err(rocks_err)?;
            if !k.starts_with(SESSION_PREFIX) {
                break;
            }
            let (record, _): (SessionRecord, _) =
                bincode::serde::decode_from_slice(&v, bincode::config::standard())
                    .map_err(|e| AuthError::internal(format!("decode session in sweep: {e}")))?;
            if record.user_id == *user_id {
                batch.delete_cf(cf, &k);
                removed += 1;
            }
        }
        if removed > 0 {
            self.db.write(batch).map_err(rocks_err)?;
        }
        Ok(removed)
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
            // Best-effort cleanup; ignore error because the auth decision
            // doesn't depend on the delete succeeding.
            let _ = self.delete_session_by_hash(&hash);
            return Err(AuthError::InvalidSession);
        }
        record.refresh(now);
        self.put_session(&record)?;
        self.get_user(&record.user_id)?
            .ok_or(AuthError::InvalidSession)
    }
}

fn rocks_err(e: rocksdb::Error) -> AuthError {
    AuthError::internal(anyhow!("rocksdb error: {e}").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_auth::{UserRecord, UserRole};
    use rocksdb::{Options as RocksOptions, DB};
    use tempfile::tempdir;

    fn open_store(dir: &std::path::Path) -> UserAuthStore {
        let mut opts = RocksOptions::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = DB::open_cf_descriptors(&opts, dir, vec![cf_descriptor()]).unwrap();
        UserAuthStore::new(Arc::new(db)).unwrap()
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
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        let user = fixture_user("Alice", UserRole::Admin);
        store.create_user(&user).unwrap();

        let by_id = store.get_user(&user.id).unwrap().unwrap();
        assert_eq!(by_id.id, user.id);
        assert_eq!(by_id.username, "Alice");

        // Case-insensitive username lookup.
        let by_name = store.find_user_by_username("ALICE").unwrap().unwrap();
        assert_eq!(by_name.id, user.id);

        // has_any_user is true after a single insert.
        assert!(store.has_any_user().unwrap());
    }

    #[test]
    fn duplicate_username_rejected() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        let u1 = fixture_user("bob", UserRole::User);
        let mut u2 = fixture_user("BOB", UserRole::User);
        // Different id, same case-insensitive username.
        u2.id = Uuid::new_v4();
        store.create_user(&u1).unwrap();
        let err = store.create_user(&u2).unwrap_err();
        assert!(matches!(err, AuthError::UsernameTaken), "{err:?}");
    }

    #[test]
    fn has_any_user_initially_false() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        assert!(!store.has_any_user().unwrap());
    }

    #[test]
    fn delete_all_sessions_only_removes_target_user() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());

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

        let n = store.delete_all_sessions_for_user(&alice.id).unwrap();
        assert_eq!(n, 2);
        // Bob's session survives.
        assert!(store
            .get_session_by_hash(&b1.token_hash)
            .unwrap()
            .is_some());
        // Alice's are gone.
        assert!(store
            .get_session_by_hash(&a1.token_hash)
            .unwrap()
            .is_none());
        assert!(store
            .get_session_by_hash(&a2.token_hash)
            .unwrap()
            .is_none());
    }

    #[test]
    fn delete_other_sessions_keeps_caller_token() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());

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
        assert_eq!(n, 1, "only the second alice session should be deleted");
        assert!(store
            .get_session_by_hash(&a_keep.token_hash)
            .unwrap()
            .is_some());
        assert!(store
            .get_session_by_hash(&a_drop.token_hash)
            .unwrap()
            .is_none());
        // Bob is untouched.
        assert!(store
            .get_session_by_hash(&b_keep.token_hash)
            .unwrap()
            .is_some());
    }

    #[test]
    fn count_admins_walks_user_records() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        assert_eq!(store.count_admins().unwrap(), 0);

        store
            .create_user(&fixture_user("root", UserRole::Admin))
            .unwrap();
        store
            .create_user(&fixture_user("rooty", UserRole::Admin))
            .unwrap();
        store
            .create_user(&fixture_user("peon", UserRole::User))
            .unwrap();

        assert_eq!(store.count_admins().unwrap(), 2);
    }

    #[test]
    fn login_throttle_default_is_unlocked() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        let row = store.get_login_throttle("ghost").unwrap();
        assert_eq!(row.failed_attempts, 0);
        assert!(row.locked_until.is_none());
        assert!(row.last_attempt_at.is_none());
        assert!(!row.is_locked(Utc::now()));
    }

    #[test]
    fn record_login_failure_locks_after_threshold() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        let now = Utc::now();
        for _ in 0..(THROTTLE_FAILURE_THRESHOLD - 1) {
            let row = store.record_login_failure("alice", now).unwrap();
            assert!(row.locked_until.is_none(), "locked too early");
        }
        let final_row = store.record_login_failure("alice", now).unwrap();
        assert_eq!(final_row.failed_attempts, THROTTLE_FAILURE_THRESHOLD);
        let lock_until = final_row.locked_until.expect("must be locked");
        assert!(lock_until > now);
        assert!(final_row.is_locked(now));
    }

    #[test]
    fn record_login_failure_resets_after_window() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        let t0 = Utc::now();
        store.record_login_failure("alice", t0).unwrap();
        store.record_login_failure("alice", t0).unwrap();
        let row = store.get_login_throttle("alice").unwrap();
        assert_eq!(row.failed_attempts, 2);

        // Past the window — counter starts fresh.
        let t1 = t0 + chrono::Duration::minutes(THROTTLE_WINDOW_MINUTES + 1);
        let after = store.record_login_failure("alice", t1).unwrap();
        assert_eq!(after.failed_attempts, 1);
        assert!(after.locked_until.is_none());
    }

    #[test]
    fn clear_login_throttle_drops_row() {
        let dir = tempdir().unwrap();
        let store = open_store(dir.path());
        let now = Utc::now();
        store.record_login_failure("alice", now).unwrap();
        assert_eq!(store.get_login_throttle("alice").unwrap().failed_attempts, 1);
        store.clear_login_throttle("alice").unwrap();
        let row = store.get_login_throttle("alice").unwrap();
        assert_eq!(row, LoginThrottleRecord::default());
    }

    #[test]
    fn login_throttle_persists_across_reopen() {
        let dir = tempdir().unwrap();
        let now = Utc::now();
        {
            let store = open_store(dir.path());
            for _ in 0..THROTTLE_FAILURE_THRESHOLD {
                store.record_login_failure("alice", now).unwrap();
            }
            let row = store.get_login_throttle("alice").unwrap();
            assert!(row.is_locked(now));
        }
        // Drop + reopen — simulates a process restart. Lock survives.
        let store = open_store(dir.path());
        let row = store.get_login_throttle("alice").unwrap();
        assert!(row.is_locked(now), "lockout must survive restart");
    }
}
