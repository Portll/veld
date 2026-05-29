//! Self-Editing Agent Context Blocks
//!
//! Named key-value blocks that agents can read and write. Unlike memories
//! (append-only with recall), context blocks are mutable state with a fixed
//! key that gets overwritten — inspired by Letta's core architecture.
//!
//! Storage uses the shared RocksDB with a dedicated column family, keyed as
//! `{user_id}:{block_key}` for efficient per-user prefix scans.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Column family name for context blocks
pub const CF_CONTEXT_BLOCKS: &str = "context_blocks";

/// Default maximum token budget for a single context block
const DEFAULT_MAX_TOKENS: usize = 2000;

/// A named, mutable context block that agents can read and write.
///
/// Unlike episodic memories which are append-only, context blocks are
/// keyed state that persists across sessions and can be freely overwritten.
/// Typical uses: persona, user profile, project state, running summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    /// Block identifier (e.g., "persona", "user_profile", "project_state")
    pub key: String,
    /// The block content (agent-editable)
    pub content: String,
    /// Size limit for this block (approximate token budget)
    pub max_tokens: usize,
    /// When this block was last updated
    pub updated_at: DateTime<Utc>,
    /// Monotonically increasing version, incremented on each write
    pub version: u32,
    /// User-pinned: when true, sleep-time will refuse to rewrite this block.
    /// Locks are explicit-only and never auto-expire (R14 + R22). The lock
    /// state itself is duplicated in the per-user budget ledger so
    /// sleep-time can check without loading every block; this field is the
    /// authoritative copy.
    #[serde(default)]
    pub locked: bool,
}

/// Outcome of an optimistic-concurrency write
/// ([`ContextBlockStore::set_with_version_check`]).
///
/// Used by the sleep-time worker to distinguish a successful OCC write from
/// a stale-version abort. The current live block is returned in the
/// conflict / locked cases so the caller can emit a
/// `SleepTimeRewriteAborted` event with rich diagnostics.
#[derive(Debug)]
pub enum OccOutcome {
    /// Write succeeded; the new block (with bumped version) is returned.
    Applied(ContextBlock),
    /// Live block version did not match `expected_version`. The current
    /// live block is returned (its `version` field is the live one).
    VersionConflict { current: ContextBlock },
    /// Live block has `locked = true`. The current live block is returned.
    Locked { current: ContextBlock },
    /// No such block exists at apply-time. Should not normally occur — the
    /// evidence pack will only carry blocks that existed at assembly time.
    /// Returned for completeness so callers can handle deletion races.
    Missing,
}

/// Storage for agent context blocks, backed by a shared RocksDB instance.
pub struct ContextBlockStore {
    db: Arc<DB>,
}

impl ContextBlockStore {
    /// Column family descriptors required by the ContextBlockStore.
    /// The caller must include these when opening the shared DB.
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut cf_opts = Options::default();
        cf_opts.create_if_missing(true);
        cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        vec![ColumnFamilyDescriptor::new(CF_CONTEXT_BLOCKS, cf_opts)]
    }

    /// Create a new context block store backed by the given shared DB.
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    fn cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_CONTEXT_BLOCKS)
            .expect("context_blocks CF must exist in shared DB")
    }

    /// Build the RocksDB key for a given user + block key.
    fn db_key(user_id: &str, block_key: &str) -> String {
        format!("{user_id}:{block_key}")
    }

    /// Retrieve a single context block by key.
    pub fn get(&self, user_id: &str, block_key: &str) -> Result<Option<ContextBlock>> {
        let key = Self::db_key(user_id, block_key);
        match self.db.get_cf(self.cf(), key.as_bytes())? {
            Some(data) => {
                let (block, _): (ContextBlock, _) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())
                        .context("Failed to deserialize context block")?;
                Ok(Some(block))
            }
            None => Ok(None),
        }
    }

    /// Create or update a context block. Returns the resulting block.
    ///
    /// If the block already exists, its content is replaced, version is
    /// incremented, and `updated_at` is set to now. The `locked` flag is
    /// preserved across writes (sleep-time guards against this separately;
    /// foreground agent edits never auto-unlock). If `max_tokens` is `None`
    /// on a new block, the default (2000) is used; on an existing block, the
    /// previous value is preserved.
    pub fn set(
        &self,
        user_id: &str,
        block_key: &str,
        content: &str,
        max_tokens: Option<usize>,
    ) -> Result<ContextBlock> {
        let key = Self::db_key(user_id, block_key);
        let now = Utc::now();

        let existing = self.get(user_id, block_key)?;

        let block = match existing {
            Some(prev) => ContextBlock {
                key: block_key.to_string(),
                content: content.to_string(),
                max_tokens: max_tokens.unwrap_or(prev.max_tokens),
                updated_at: now,
                version: prev.version.saturating_add(1),
                locked: prev.locked,
            },
            None => ContextBlock {
                key: block_key.to_string(),
                content: content.to_string(),
                max_tokens: max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
                updated_at: now,
                version: 1,
                locked: false,
            },
        };

        let value = bincode::serde::encode_to_vec(&block, bincode::config::standard())
            .context("Failed to serialize context block")?;
        self.db
            .put_cf(self.cf(), key.as_bytes(), &value)
            .context("Failed to write context block to RocksDB")?;

        tracing::debug!(
            user_id = user_id,
            block_key = block_key,
            version = block.version,
            content_len = block.content.len(),
            "Context block updated"
        );

        Ok(block)
    }

    /// Set with optimistic concurrency control (R1).
    ///
    /// Writes only if the live block's `version` equals `expected_version`.
    /// If the live version has advanced (foreground agent rewrote it since
    /// the evidence pack was assembled), returns
    /// [`OccOutcome::VersionConflict`] WITHOUT writing — the caller MUST
    /// abort and emit a `SleepTimeRewriteAborted` event.
    ///
    /// The `locked` flag is honoured: a locked block returns
    /// [`OccOutcome::Locked`] regardless of version match. This is a
    /// double-check on top of the orchestrator's pre-flight lock filter; if
    /// a user locks a block between evidence-assembly and apply-time, the
    /// rewrite still aborts cleanly.
    ///
    /// Successful applies preserve the existing `max_tokens` (sleep-time
    /// never resizes blocks — sizing is a foreground concern) and the
    /// existing `locked` flag.
    pub fn set_with_version_check(
        &self,
        user_id: &str,
        block_key: &str,
        new_content: &str,
        expected_version: u32,
    ) -> Result<OccOutcome> {
        let key = Self::db_key(user_id, block_key);
        let now = Utc::now();

        let Some(prev) = self.get(user_id, block_key)? else {
            return Ok(OccOutcome::Missing);
        };
        if prev.locked {
            return Ok(OccOutcome::Locked { current: prev });
        }
        if prev.version != expected_version {
            return Ok(OccOutcome::VersionConflict { current: prev });
        }

        let block = ContextBlock {
            key: block_key.to_string(),
            content: new_content.to_string(),
            max_tokens: prev.max_tokens,
            updated_at: now,
            version: prev.version.saturating_add(1),
            locked: prev.locked,
        };

        let value = bincode::serde::encode_to_vec(&block, bincode::config::standard())
            .context("Failed to serialize context block (OCC)")?;
        self.db
            .put_cf(self.cf(), key.as_bytes(), &value)
            .context("Failed to write context block to RocksDB (OCC)")?;

        tracing::debug!(
            user_id = user_id,
            block_key = block_key,
            old_version = expected_version,
            new_version = block.version,
            content_len = block.content.len(),
            "Context block updated via OCC"
        );

        Ok(OccOutcome::Applied(block))
    }

    /// Toggle the `locked` flag without touching content or version. Locks
    /// are explicit-only (R14 + R22): nothing else flips this.
    pub fn set_locked(&self, user_id: &str, block_key: &str, locked: bool) -> Result<()> {
        let Some(mut prev) = self.get(user_id, block_key)? else {
            anyhow::bail!("context block {block_key} for user {user_id} does not exist");
        };
        if prev.locked == locked {
            return Ok(());
        }
        prev.locked = locked;
        let key = Self::db_key(user_id, block_key);
        let value = bincode::serde::encode_to_vec(&prev, bincode::config::standard())
            .context("Failed to serialize context block lock toggle")?;
        self.db
            .put_cf(self.cf(), key.as_bytes(), &value)
            .context("Failed to write context block lock toggle")?;
        tracing::info!(
            user_id = user_id,
            block_key = block_key,
            locked,
            "Context block lock toggled"
        );
        Ok(())
    }

    /// List all context blocks for a user.
    pub fn list(&self, user_id: &str) -> Result<Vec<ContextBlock>> {
        let prefix = format!("{user_id}:");
        let mut blocks = Vec::new();

        let iter = self
            .db
            .prefix_iterator_cf(self.cf(), prefix.as_bytes());

        for item in iter {
            let (key, value) = item.context("Failed to iterate context blocks")?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            if let Ok((block, _)) = bincode::serde::decode_from_slice::<ContextBlock, _>(
                &value,
                bincode::config::standard(),
            ) {
                blocks.push(block);
            }
        }

        // Sort by key for deterministic ordering
        blocks.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(blocks)
    }

    /// Delete a context block. Returns `true` if the block existed.
    pub fn delete(&self, user_id: &str, block_key: &str) -> Result<bool> {
        let key = Self::db_key(user_id, block_key);
        let existed = self.db.get_cf(self.cf(), key.as_bytes())?.is_some();

        if existed {
            self.db
                .delete_cf(self.cf(), key.as_bytes())
                .context("Failed to delete context block from RocksDB")?;
            tracing::debug!(
                user_id = user_id,
                block_key = block_key,
                "Context block deleted"
            );
        }

        Ok(existed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_test_db() -> (Arc<DB>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let cfs = ContextBlockStore::cf_descriptors();
        let db = DB::open_cf_descriptors(&db_opts, tmp.path(), cfs).unwrap();
        (Arc::new(db), tmp)
    }

    #[test]
    fn test_set_and_get() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);

        let block = store.set("user1", "persona", "You are a helpful assistant.", None).unwrap();
        assert_eq!(block.key, "persona");
        assert_eq!(block.version, 1);
        assert_eq!(block.max_tokens, DEFAULT_MAX_TOKENS);

        let retrieved = store.get("user1", "persona").unwrap().unwrap();
        assert_eq!(retrieved.content, "You are a helpful assistant.");
        assert_eq!(retrieved.version, 1);
    }

    #[test]
    fn test_version_increment() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);

        store.set("user1", "state", "v1 content", None).unwrap();
        let block = store.set("user1", "state", "v2 content", None).unwrap();
        assert_eq!(block.version, 2);
        assert_eq!(block.content, "v2 content");

        let block = store.set("user1", "state", "v3 content", None).unwrap();
        assert_eq!(block.version, 3);
    }

    #[test]
    fn test_max_tokens_preserved() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);

        store.set("user1", "x", "content", Some(500)).unwrap();
        // Update without specifying max_tokens — should preserve 500
        let block = store.set("user1", "x", "new content", None).unwrap();
        assert_eq!(block.max_tokens, 500);

        // Explicit override
        let block = store.set("user1", "x", "newer", Some(1000)).unwrap();
        assert_eq!(block.max_tokens, 1000);
    }

    #[test]
    fn test_list() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);

        store.set("user1", "alpha", "a", None).unwrap();
        store.set("user1", "beta", "b", None).unwrap();
        store.set("user2", "gamma", "c", None).unwrap();

        let user1_blocks = store.list("user1").unwrap();
        assert_eq!(user1_blocks.len(), 2);
        assert_eq!(user1_blocks[0].key, "alpha");
        assert_eq!(user1_blocks[1].key, "beta");

        let user2_blocks = store.list("user2").unwrap();
        assert_eq!(user2_blocks.len(), 1);
    }

    #[test]
    fn test_delete() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);

        store.set("user1", "temp", "data", None).unwrap();
        assert!(store.delete("user1", "temp").unwrap());
        assert!(!store.delete("user1", "temp").unwrap());
        assert!(store.get("user1", "temp").unwrap().is_none());
    }

    #[test]
    fn test_get_nonexistent() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);

        assert!(store.get("user1", "missing").unwrap().is_none());
    }

    // -------------------------------------------------------------------------
    // R14 + R22: lock state
    // -------------------------------------------------------------------------

    #[test]
    fn new_block_starts_unlocked() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        let block = store.set("u", "persona", "x", None).unwrap();
        assert!(!block.locked);
    }

    #[test]
    fn set_locked_persists_across_get() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        store.set("u", "persona", "x", None).unwrap();
        store.set_locked("u", "persona", true).unwrap();
        let got = store.get("u", "persona").unwrap().unwrap();
        assert!(got.locked);
    }

    #[test]
    fn set_preserves_lock_across_content_update() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        store.set("u", "persona", "v1", None).unwrap();
        store.set_locked("u", "persona", true).unwrap();
        let updated = store.set("u", "persona", "v2", None).unwrap();
        assert!(
            updated.locked,
            "regular set() must not auto-unlock — only explicit set_locked(false) does (R22)"
        );
    }

    #[test]
    fn set_locked_idempotent() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        store.set("u", "persona", "x", None).unwrap();
        store.set_locked("u", "persona", true).unwrap();
        store.set_locked("u", "persona", true).unwrap();
        // No panic, no version churn (set_locked does not bump version).
        let got = store.get("u", "persona").unwrap().unwrap();
        assert_eq!(got.version, 1);
        assert!(got.locked);
    }

    #[test]
    fn set_locked_on_missing_block_errors() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        assert!(store.set_locked("u", "ghost", true).is_err());
    }

    // -------------------------------------------------------------------------
    // R1: optimistic concurrency control
    // -------------------------------------------------------------------------

    #[test]
    fn occ_applied_on_matching_version() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        let v1 = store.set("u", "persona", "v1", None).unwrap();
        let out = store
            .set_with_version_check("u", "persona", "v2", v1.version)
            .unwrap();
        match out {
            OccOutcome::Applied(b) => {
                assert_eq!(b.version, v1.version + 1);
                assert_eq!(b.content, "v2");
            }
            other => panic!("expected Applied, got {other:?}"),
        }
    }

    #[test]
    fn occ_version_conflict_does_not_write() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        let v1 = store.set("u", "persona", "v1", None).unwrap();
        // Foreground writer bumps version to 2.
        store.set("u", "persona", "v1.5", None).unwrap();
        // Sleep-time tries to apply with stale expected_version = 1.
        let out = store
            .set_with_version_check("u", "persona", "v2", v1.version)
            .unwrap();
        match out {
            OccOutcome::VersionConflict { current } => assert_eq!(current.version, 2),
            other => panic!("expected VersionConflict, got {other:?}"),
        }
        let live = store.get("u", "persona").unwrap().unwrap();
        assert_eq!(
            live.content, "v1.5",
            "OCC abort must not have overwritten the foreground edit"
        );
    }

    #[test]
    fn occ_locked_block_aborts() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        let v1 = store.set("u", "persona", "v1", None).unwrap();
        store.set_locked("u", "persona", true).unwrap();
        let out = store
            .set_with_version_check("u", "persona", "v2", v1.version)
            .unwrap();
        match out {
            OccOutcome::Locked { current } => {
                assert!(current.locked);
                assert_eq!(current.content, "v1");
            }
            other => panic!("expected Locked, got {other:?}"),
        }
    }

    #[test]
    fn occ_missing_block_returns_missing() {
        let (db, _tmp) = open_test_db();
        let store = ContextBlockStore::new(db);
        let out = store
            .set_with_version_check("u", "ghost", "anything", 1)
            .unwrap();
        assert!(matches!(out, OccOutcome::Missing));
    }
}
