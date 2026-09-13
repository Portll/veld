//! Persistent debounced work queue for sleep-time triggers.
//!
//! Queue items live in the `sleep_time_queue` column family so
//! triggers survive process restart. Workers claim items with a time-bounded
//! lease (R3) — if a worker dies mid-process the claim expires and the item
//! is re-claimable.
//!
//! Cold-start hygiene (R31 + R67):
//!   - On startup, [`Queue::cold_start_purge`] drops items older than the
//!     configured TTL. Avoids replaying week-old triggers after a long
//!     downtime.
//!   - Schema version (`QUEUE_ITEM_SCHEMA_VERSION`, R32) is checked on every
//!     decode; mismatched items are *skipped*, not panicked. The deploy that
//!     introduced the bump owns the migration.
//!
//! Key layout: `{user_id}:{iso8601_enqueued_at}:{item_id}` — sorts items
//! chronologically per-user via the store's lexicographic ordering, which makes
//! `claim_next_for_user` a cheap prefix scan that returns the oldest first.

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use std::sync::Arc;

use super::types::{QueueItem, SleepMode, SleepTimeTrigger, QUEUE_ITEM_SCHEMA_VERSION};

/// Column family for persistent queue items.
pub const CF_SLEEP_TIME_QUEUE: &str = "sleep_time_queue";

/// Default lease applied when a worker claims an item. Workers must finish
/// (success or failure) within this window or the claim expires and the item
/// becomes re-claimable.
pub const DEFAULT_CLAIM_LEASE_SECS: i64 = 600; // 10 minutes

pub struct Queue {
    /// One family, so a CF-bound handle rather than the multi-family one.
    store: Arc<dyn crate::storage::KeyValueStore>,
}

impl Queue {
    /// Bind a handle directly — the constructor a backend-agnostic caller uses.
    pub fn with_store(store: Arc<dyn crate::storage::KeyValueStore>) -> Self {
        Self { store }
    }

    /// Build the storage key for an item. The middle component is the
    /// enqueued_at timestamp in lexicographically-sortable form so `next_*`
    /// scans return oldest-first.
    fn key(item: &QueueItem) -> String {
        format!(
            "{}:{}:{}",
            item.user_id,
            // RFC3339 ISO8601 sorts lexicographically when zero-padded; chrono
            // produces e.g. `2026-05-28T12:34:56.789Z` — already sortable.
            item.enqueued_at.to_rfc3339(),
            item.id
        )
    }

    fn parse_user_from_key(key: &[u8]) -> Option<&str> {
        let s = std::str::from_utf8(key).ok()?;
        s.split(':').next()
    }

    /// Persist a new queue item. Returns the stored item (with its
    /// `schema_version` and key timestamp set).
    pub fn enqueue(&self, item: QueueItem) -> Result<QueueItem> {
        debug_assert_eq!(item.schema_version, QUEUE_ITEM_SCHEMA_VERSION);
        let key = Self::key(&item);
        let bytes = bincode::serde::encode_to_vec(&item, bincode::config::standard())
            .context("encode QueueItem")?;
        self.store
            .put_bytes(key.as_bytes(), &bytes)
            .context("write sleep_time_queue")?;
        tracing::debug!(
            user_id = %item.user_id,
            mode = item.mode.as_str(),
            id = %item.id,
            "sleep-time trigger enqueued"
        );
        Ok(item)
    }

    /// Enqueue collapsing on `(user_id, mode, trigger)` within `debounce`. If
    /// an unclaimed item already exists matching those keys whose
    /// `enqueued_at` is within the debounce window, returns `Ok(None)` (drop
    /// the duplicate). Otherwise returns `Ok(Some(stored))`.
    ///
    /// This is a best-effort prefix scan; it's safe for missed-deduplication
    /// to fall through (the worker idempotency will eventually reconcile).
    pub fn enqueue_debounced(
        &self,
        user_id: &str,
        mode: SleepMode,
        trigger: SleepTimeTrigger,
        debounce: Duration,
    ) -> Result<Option<QueueItem>> {
        let now = Utc::now();
        let cutoff = now - debounce;

        for it in self.scan_user(user_id)? {
            // Already-claimed items still count as "scheduled" for debounce
            // purposes — don't replace them.
            if it.mode == mode && it.trigger == trigger && it.enqueued_at >= cutoff {
                return Ok(None);
            }
        }

        let item = QueueItem::new(user_id.to_string(), trigger, mode);
        Ok(Some(self.enqueue(item)?))
    }

    /// Scan all items for a given user. Returns oldest-first by key ordering.
    pub fn scan_user(&self, user_id: &str) -> Result<Vec<QueueItem>> {
        let prefix = format!("{user_id}:");
        let mut out = Vec::new();
        // scan_prefix enforces the bound; the hand-written break is gone.
        for (k, v) in self.store.scan_prefix(prefix.as_bytes(), None)? {
            let k_str = String::from_utf8_lossy(&k);
            match decode_item(&v) {
                Ok(Some(item)) => out.push(item),
                Ok(None) => {
                    // Schema mismatch: skip (R32). Logged at warn level.
                    tracing::warn!(
                        key = %k_str,
                        "sleep_time_queue: skipping item with mismatched schema_version"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        key = %k_str,
                        error = %e,
                        "sleep_time_queue: failed to decode item"
                    );
                }
            }
        }
        Ok(out)
    }

    /// Find the next unclaimed (or claim-expired) item across all users. Used
    /// by the worker's per-user fairness loop together with [`Self::claim`].
    ///
    /// Returns oldest-first across the whole CF — callers are responsible for
    /// applying per-user fairness on top.
    pub fn next_unclaimed(&self) -> Result<Option<QueueItem>> {
        let now = Utc::now();
        // empty prefix = the whole family, ascending
        for (_k, v) in self.store.scan_prefix(b"", None)? {
            match decode_item(&v) {
                Ok(Some(item)) if item.claim_expired(now) => return Ok(Some(item)),
                Ok(Some(_)) => continue, // currently-claimed, skip
                Ok(None) => continue,    // schema mismatch, skip
                Err(_) => continue,
            }
        }
        Ok(None)
    }

    /// Find the next unclaimed item for a specific user (oldest first).
    pub fn next_unclaimed_for_user(&self, user_id: &str) -> Result<Option<QueueItem>> {
        let now = Utc::now();
        for item in self.scan_user(user_id)? {
            if item.claim_expired(now) {
                return Ok(Some(item));
            }
        }
        Ok(None)
    }

    /// Attempt to mark `item` as claimed by `worker_id` with the given lease.
    /// Returns the claimed item on success. If the item has been deleted or
    /// re-claimed since [`Self::next_unclaimed_for_user`], returns
    /// `Ok(None)` — caller should retry.
    pub fn claim(
        &self,
        item: &QueueItem,
        worker_id: &str,
        lease_secs: i64,
    ) -> Result<Option<QueueItem>> {
        let key = Self::key(item);
        let raw = self.store.get_bytes(key.as_bytes())?;
        let Some(bytes) = raw else { return Ok(None) };
        let current = match decode_item(&bytes) {
            Ok(Some(c)) => c,
            _ => return Ok(None),
        };
        // Race-check: someone else may have claimed in the gap between
        // next_unclaimed_for_user and here.
        if !current.claim_expired(Utc::now()) {
            return Ok(None);
        }
        let mut claimed = current;
        claimed.claimed_by = Some(worker_id.to_string());
        claimed.claim_expires_at = Some(Utc::now() + Duration::seconds(lease_secs));

        let new_bytes = bincode::serde::encode_to_vec(&claimed, bincode::config::standard())
            .context("encode claimed QueueItem")?;
        self.store
            .put_bytes(key.as_bytes(), &new_bytes)
            .context("write claimed sleep_time_queue")?;
        Ok(Some(claimed))
    }

    /// Release `item` after successful processing — deletes it.
    pub fn complete(&self, item: &QueueItem) -> Result<()> {
        let key = Self::key(item);
        self.store
            .delete_bytes(key.as_bytes())
            .context("delete completed sleep_time_queue")?;
        Ok(())
    }

    /// Release `item` after failed processing — clears the claim so another
    /// worker can retry. Callers that want bounded retries should track
    /// attempt count separately and call [`Self::complete`] to give up.
    pub fn release(&self, item: &QueueItem) -> Result<()> {
        let key = Self::key(item);
        let raw = self.store.get_bytes(key.as_bytes())?;
        let Some(bytes) = raw else { return Ok(()) };
        let Ok(Some(mut current)) = decode_item(&bytes) else {
            return Ok(());
        };
        current.claimed_by = None;
        current.claim_expires_at = None;
        let new_bytes = bincode::serde::encode_to_vec(&current, bincode::config::standard())
            .context("encode released QueueItem")?;
        self.store
            .put_bytes(key.as_bytes(), &new_bytes)
            .context("write released sleep_time_queue")?;
        Ok(())
    }

    /// Cold-start purge — drop items older than `ttl` (R31 + R67). Run once
    /// at startup *before* workers begin claiming. Returns the number purged
    /// for ops visibility.
    pub fn cold_start_purge(&self, ttl: Duration) -> Result<usize> {
        let cutoff = Utc::now() - ttl;
        let mut purged = 0usize;
        let mut to_delete: Vec<Vec<u8>> = Vec::new();
        for (k, v) in self.store.scan_prefix(b"", None)? {
            let decoded = decode_item(&v);
            let drop = match decoded {
                Ok(Some(item)) => item.enqueued_at < cutoff,
                Ok(None) => true, // schema mismatch — drop on cold start
                Err(_) => true,   // unreadable — drop on cold start
            };
            if drop {
                to_delete.push(k);
            }
        }
        for k in to_delete {
            self.store
                .delete_bytes(&k)
                .context("cold-start delete")?;
            purged += 1;
        }
        if purged > 0 {
            tracing::info!(purged, ?ttl, "sleep_time_queue cold-start purge complete");
        }
        Ok(purged)
    }

    /// Count of pending items for a user (claimed or not).
    pub fn pending_count(&self, user_id: &str) -> Result<usize> {
        Ok(self.scan_user(user_id)?.len())
    }

    /// List every user who has at least one queued item. Used by worker
    /// fairness scheduling.
    pub fn distinct_users(&self) -> Result<Vec<String>> {
        let mut users = std::collections::BTreeSet::new();
        for (k, _) in self.store.scan_prefix(b"", None)? {
            if let Some(u) = Self::parse_user_from_key(&k) {
                users.insert(u.to_string());
            }
        }
        Ok(users.into_iter().collect())
    }

    /// Test-helper: explicit item delete by id+user. Not exposed in the
    /// general API because in production the only ways an item should leave
    /// the queue are `complete` (success) or `cold_start_purge` (TTL).
    #[cfg(test)]
    pub fn delete_for_test(&self, item: &QueueItem) -> Result<()> {
        self.complete(item)
    }
}

/// Decode a queue item, returning `Ok(None)` on schema-version mismatch so
/// callers can `continue` past stale entries (R32). Hard decode failures
/// surface as `Err`.
fn decode_item(bytes: &[u8]) -> Result<Option<QueueItem>> {
    let item: QueueItem =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map(|(s, _)| s)
            .context("decode QueueItem")?;
    if item.schema_version != QUEUE_ITEM_SCHEMA_VERSION {
        return Ok(None);
    }
    Ok(Some(item))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageBackend;
    use crate::storage::test_support::{assert_both_backends, column_stores};
    use crate::storage::FamilyKeyValueStore;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn queues() -> Vec<(StorageBackend, Queue, TempDir)> {
        let out: Vec<_> = column_stores(&[CF_SLEEP_TIME_QUEUE])
            .into_iter()
            .map(|(backend, column, tmp)| {
                let kv = FamilyKeyValueStore::new(column, CF_SLEEP_TIME_QUEUE).unwrap();
                (backend, Queue::with_store(Arc::new(kv)), tmp)
            })
            .collect();
        assert_both_backends(&out);
        out
    }

    #[test]
    fn enqueue_then_scan_returns_item() {
        for (backend, q, _tmp) in queues() {
            let item = QueueItem::new("alice", SleepTimeTrigger::Idle, SleepMode::Nrem);
            q.enqueue(item.clone()).unwrap();
            let scanned = q.scan_user("alice").unwrap();
            assert_eq!(scanned.len(), 1, "{backend:?}");
            assert_eq!(scanned[0].id, item.id, "{backend:?}");
        }
    }

    #[test]
    fn enqueue_debounced_drops_duplicates() {
        for (backend, q, _tmp) in queues() {
            let first = q
                .enqueue_debounced("u", SleepMode::Nrem, SleepTimeTrigger::Idle, Duration::seconds(60))
                .unwrap();
            assert!(first.is_some(), "{backend:?}");
            let second = q
                .enqueue_debounced("u", SleepMode::Nrem, SleepTimeTrigger::Idle, Duration::seconds(60))
                .unwrap();
            assert!(second.is_none(), "{backend:?}: duplicate within window should be dropped");
            // Different mode bypasses the debounce.
            let third = q
                .enqueue_debounced("u", SleepMode::Rem, SleepTimeTrigger::Idle, Duration::seconds(60))
                .unwrap();
            assert!(third.is_some(), "{backend:?}");
        }
    }

    #[test]
    fn claim_then_release_makes_re_claimable() {
        for (backend, q, _tmp) in queues() {
            let item = QueueItem::new("u", SleepTimeTrigger::Manual, SleepMode::Nrem);
            let stored = q.enqueue(item).unwrap();

            let claimed = q.claim(&stored, "worker-1", 300).unwrap().unwrap();
            assert_eq!(claimed.claimed_by.as_deref(), Some("worker-1"), "{backend:?}");
            // Re-claim attempt should fail while leased.
            let racey = q.claim(&claimed, "worker-2", 300).unwrap();
            assert!(racey.is_none(), "{backend:?}: a leased item was claimed twice");

            q.release(&claimed).unwrap();
            let re_claimed = q.claim(&claimed, "worker-2", 300).unwrap().unwrap();
            assert_eq!(re_claimed.claimed_by.as_deref(), Some("worker-2"), "{backend:?}");
        }
    }

    #[test]
    fn complete_removes_item() {
        for (backend, q, _tmp) in queues() {
            let stored = q
                .enqueue(QueueItem::new("u", SleepTimeTrigger::Idle, SleepMode::Nrem))
                .unwrap();
            let claimed = q.claim(&stored, "w", 60).unwrap().unwrap();
            q.complete(&claimed).unwrap();
            assert_eq!(q.scan_user("u").unwrap().len(), 0, "{backend:?}");
        }
    }

    #[test]
    fn cold_start_purge_drops_old_items() {
        for (backend, q, _tmp) in queues() {
            let mut ancient = QueueItem::new("u", SleepTimeTrigger::Idle, SleepMode::Nrem);
            ancient.enqueued_at = Utc::now() - Duration::days(7);
            q.enqueue(ancient).unwrap();

            let fresh = QueueItem::new("u", SleepTimeTrigger::Idle, SleepMode::Rem);
            q.enqueue(fresh).unwrap();

            let purged = q.cold_start_purge(Duration::hours(2)).unwrap();
            assert_eq!(purged, 1, "{backend:?}");
            let left = q.scan_user("u").unwrap();
            assert_eq!(left.len(), 1, "{backend:?}");
            assert_eq!(left[0].mode, SleepMode::Rem, "{backend:?}");
        }
    }

    #[test]
    fn distinct_users_returns_unique_set() {
        for (backend, q, _tmp) in queues() {
            q.enqueue(QueueItem::new("alice", SleepTimeTrigger::Idle, SleepMode::Nrem))
                .unwrap();
            q.enqueue(QueueItem::new("alice", SleepTimeTrigger::Idle, SleepMode::Rem))
                .unwrap();
            q.enqueue(QueueItem::new("bob", SleepTimeTrigger::Manual, SleepMode::Nrem))
                .unwrap();
            let mut users = q.distinct_users().unwrap();
            users.sort();
            assert_eq!(users, vec!["alice".to_string(), "bob".to_string()], "{backend:?}");
        }
    }

    #[test]
    fn scan_user_does_not_leak_a_neighbouring_prefix() {
        // `al:` must not match `alice:` rows; the bound is the store's, not a hand break.
        for (backend, q, _tmp) in queues() {
            q.enqueue(QueueItem::new("alice", SleepTimeTrigger::Idle, SleepMode::Nrem))
                .unwrap();
            q.enqueue(QueueItem::new("al", SleepTimeTrigger::Idle, SleepMode::Nrem))
                .unwrap();
            assert_eq!(q.scan_user("al").unwrap().len(), 1, "{backend:?}");
            assert_eq!(q.pending_count("alice").unwrap(), 1, "{backend:?}");
        }
    }

    #[test]
    fn schema_version_mismatch_skipped_not_panicked() {
        // Hand-craft bytes for a wrong-version item and verify decode returns None.
        let bad = QueueItem {
            schema_version: 999,
            id: Uuid::new_v4(),
            user_id: "u".into(),
            trigger: SleepTimeTrigger::Idle,
            mode: SleepMode::Nrem,
            enqueued_at: Utc::now(),
            claim_expires_at: None,
            claimed_by: None,
        };
        let bytes = bincode::serde::encode_to_vec(&bad, bincode::config::standard()).unwrap();
        assert!(decode_item(&bytes).unwrap().is_none());
    }

    #[test]
    fn next_unclaimed_returns_oldest_first() {
        for (backend, q, _tmp) in queues() {
            // Insert in non-monotonic order; scan should still return oldest first.
            let mut newer = QueueItem::new("u", SleepTimeTrigger::Idle, SleepMode::Nrem);
            newer.enqueued_at = Utc::now();
            let mut older = QueueItem::new("u", SleepTimeTrigger::Manual, SleepMode::Rem);
            older.enqueued_at = Utc::now() - Duration::minutes(10);
            q.enqueue(newer).unwrap();
            q.enqueue(older.clone()).unwrap();
            let next = q.next_unclaimed().unwrap().unwrap();
            assert_eq!(next.id, older.id, "{backend:?}");
        }
    }
}
