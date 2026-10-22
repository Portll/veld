//! Memories left without a vector, and the drain that gives them one.
//!
//! A write succeeds while the embedder is down: the memory is stored, BM25 indexes it, and the
//! background embed path logs a warning and returns. Nothing retried. The vector index stayed
//! short of the store until the next restart's orphan repair, so recall was blind to everything
//! written during an outage. The backlog is the missing ledger: one row per memory that needs a
//! vector, written at every point that gives up, and reconciled against the index so a path that
//! never recorded its failure is caught too. The drain embeds oldest first and stops at the first
//! "embedder unavailable" rather than hammering a host that is down.
//!
//! A tenant's open queues here too, rather than repairing in place, every live memory the
//! loaded vector index lacks: the open serves at once and the drain indexes them on its tick.
//! Those rows are [`BacklogEntry::index_only`] when the memory already carries its vectors.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::memory::MemoryId;
use crate::storage::KeyValueStore;

const PREFIX: &str = "embed_backlog:";
pub const SCHEMA: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnqueueReason {
    /// The embedder refused or could not be reached while a write was being embedded.
    EmbedFailed,
    /// The vector index refused the memory after an embedding existed.
    IndexFailed,
    /// Found stored but not indexed by a reconcile pass; the failure was never recorded.
    Reconciled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BacklogEntry {
    pub v: u32,
    pub reason: EnqueueReason,
    pub enqueued_at: DateTime<Utc>,
    pub attempts: u32,
    #[serde(default)]
    pub last_attempt_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_error: Option<String>,
    /// Queued at open for a memory whose vector the saved index lost after the memory was
    /// embedded and processed: the drain indexes it from the vectors it carries, without the
    /// absorption and interference passes it already went through. Absent from rows that are
    /// not, so those serialize as before and an older build reads every row.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub index_only: bool,
}

/// What one attempt to embed and index a memory came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedOutcome {
    Indexed,
    /// The embedder did not answer. The drain stops here: the next memory would fail the same way.
    EmbedderUnavailable(String),
    /// The memory no longer exists or is forgotten; the row is dropped.
    MemoryGone,
    /// A failure specific to this memory (oversized, index refusal); the row stays, attempts grow.
    Failed(String),
    /// The tenant closed to writes for shutdown. The drain stops and leaves the row as it
    /// was: recording the attempt would be a write after the flush.
    WritesClosed,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BacklogReport {
    pub witness: &'static str,
    pub pending: usize,
    pub oldest_enqueued_at: Option<DateTime<Utc>>,
    pub max_attempts: u32,
    pub by_reason: BTreeMap<String, usize>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct ReconcileReport {
    pub stored: usize,
    pub indexed: usize,
    pub enqueued: usize,
    pub dropped_indexed: usize,
    pub dropped_gone: usize,
    pub pending: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct DrainReport {
    pub attempted: usize,
    pub drained: usize,
    pub dropped: usize,
    pub failed: usize,
    pub remaining: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
}

pub struct EmbeddingBacklogStore {
    kv: Arc<dyn KeyValueStore>,
}

fn key(id: &MemoryId) -> Vec<u8> {
    format!("{PREFIX}{}", id.0).into_bytes()
}

impl EmbeddingBacklogStore {
    pub fn with_kv(kv: Arc<dyn KeyValueStore>) -> Self {
        Self { kv }
    }

    pub fn get(&self, id: &MemoryId) -> Result<Option<BacklogEntry>> {
        match self
            .kv
            .get_bytes(&key(id))
            .context("read embedding backlog row")?
        {
            Some(bytes) => {
                Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
                    format!("decode embedding backlog row for {}", id.0)
                })?))
            }
            None => Ok(None),
        }
    }

    pub fn put(&self, id: &MemoryId, entry: &BacklogEntry) -> Result<()> {
        let bytes = serde_json::to_vec(entry).context("encode embedding backlog row")?;
        self.kv
            .put_bytes(&key(id), &bytes)
            .context("write embedding backlog row")
    }

    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        self.kv
            .delete_bytes(&key(id))
            .context("delete embedding backlog row")
    }

    /// Every row. A scan or decode error is an error, never an empty backlog.
    pub fn all(&self) -> Result<BTreeMap<MemoryId, BacklogEntry>> {
        let rows = self
            .kv
            .scan_prefix(PREFIX.as_bytes(), None)
            .context("scan embedding backlog")?;
        let mut out = BTreeMap::new();
        for (k, v) in rows {
            let k = String::from_utf8(k).context("embedding backlog key is not utf-8")?;
            let id_str = k
                .strip_prefix(PREFIX)
                .context("embedding backlog key without prefix")?;
            let id = MemoryId(
                uuid::Uuid::parse_str(id_str)
                    .with_context(|| format!("embedding backlog key {k} is not a memory id"))?,
            );
            let entry: BacklogEntry = serde_json::from_slice(&v)
                .with_context(|| format!("decode embedding backlog row for {id_str}"))?;
            out.insert(id, entry);
        }
        Ok(out)
    }

    /// Record that a memory needs a vector. An existing row keeps its enqueue time and reason;
    /// the attempt count and last error move.
    pub fn record(&self, id: &MemoryId, reason: EnqueueReason, error: Option<&str>) -> Result<()> {
        let now = Utc::now();
        let entry = match self.get(id)? {
            Some(mut existing) => {
                existing.attempts += 1;
                existing.last_attempt_at = Some(now);
                existing.last_error = error.map(str::to_string);
                existing
            }
            None => BacklogEntry {
                v: SCHEMA,
                reason,
                enqueued_at: now,
                attempts: u32::from(error.is_some()),
                last_attempt_at: error.map(|_| now),
                last_error: error.map(str::to_string),
                index_only: false,
            },
        };
        self.put(id, &entry)
    }

    /// Queue `ids`, live memories a tenant's open found missing from its vector index, as
    /// [`BacklogEntry::index_only`] rows, in one atomic write. An id that already has a row
    /// keeps it: whatever queued it first knows more about what it needs. Returns how many
    /// rows were added.
    pub fn enqueue_unindexed(&self, ids: &[MemoryId]) -> Result<usize> {
        let existing = self.all()?;
        let now = Utc::now();
        let pairs = ids
            .iter()
            .filter(|id| !existing.contains_key(*id))
            .map(|id| {
                let entry = BacklogEntry {
                    v: SCHEMA,
                    reason: EnqueueReason::Reconciled,
                    enqueued_at: now,
                    attempts: 0,
                    last_attempt_at: None,
                    last_error: None,
                    index_only: true,
                };
                Ok((
                    key(id),
                    serde_json::to_vec(&entry).context("encode embedding backlog row")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if !pairs.is_empty() {
            self.kv
                .put_many(&pairs)
                .context("write embedding backlog rows")?;
        }
        Ok(pairs.len())
    }

    pub fn report(&self) -> Result<BacklogReport> {
        let rows = self.all()?;
        let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
        let mut max_attempts = 0u32;
        let mut oldest: Option<DateTime<Utc>> = None;
        let mut last_error = None;
        let mut last_error_at: Option<DateTime<Utc>> = None;
        for entry in rows.values() {
            *by_reason
                .entry(
                    serde_json::to_value(entry.reason)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_string))
                        .unwrap_or_default(),
                )
                .or_default() += 1;
            max_attempts = max_attempts.max(entry.attempts);
            if oldest.is_none_or(|o| entry.enqueued_at < o) {
                oldest = Some(entry.enqueued_at);
            }
            if let (Some(err), Some(at)) = (&entry.last_error, entry.last_attempt_at) {
                if last_error_at.is_none_or(|t| at > t) {
                    last_error_at = Some(at);
                    last_error = Some(err.clone());
                }
            }
        }
        Ok(BacklogReport {
            witness: "embed_backlog rows in the user's KV store",
            pending: rows.len(),
            oldest_enqueued_at: oldest,
            max_attempts,
            by_reason,
            last_error,
        })
    }

    /// Bring the rows in line with the store and the index. `stored` is every live memory id,
    /// `indexed` every id the vector index holds. A stored id the index lacks gets a row; a row
    /// whose id is indexed or no longer stored is dropped.
    pub fn reconcile(
        &self,
        stored: &[MemoryId],
        indexed: &std::collections::HashSet<MemoryId>,
    ) -> Result<ReconcileReport> {
        let rows = self.all()?;
        let stored_set: std::collections::HashSet<&MemoryId> = stored.iter().collect();
        let mut report = ReconcileReport {
            stored: stored.len(),
            indexed: indexed.len(),
            ..Default::default()
        };
        for id in rows.keys() {
            if indexed.contains(id) {
                self.delete(id)?;
                report.dropped_indexed += 1;
            } else if !stored_set.contains(id) {
                self.delete(id)?;
                report.dropped_gone += 1;
            }
        }
        for id in stored {
            if !indexed.contains(id) && !rows.contains_key(id) {
                self.record(id, EnqueueReason::Reconciled, None)?;
                report.enqueued += 1;
            }
        }
        report.pending = self.all()?.len();
        Ok(report)
    }

    /// Embed oldest first, at most `limit`, through `embed_one`, which is given each row with
    /// its id. Stops at the first `EmbedderUnavailable` so a down host is asked once per
    /// drain, not once per row.
    pub fn drain_with(
        &self,
        limit: usize,
        embed_one: impl FnMut(&MemoryId, &BacklogEntry) -> EmbedOutcome,
    ) -> Result<DrainReport> {
        self.drain_where(limit, |_| true, embed_one)
    }

    /// [`Self::drain_with`] over the rows `select` accepts; the others are left as they are
    /// and still counted in `remaining`.
    pub fn drain_where(
        &self,
        limit: usize,
        select: impl Fn(&BacklogEntry) -> bool,
        mut embed_one: impl FnMut(&MemoryId, &BacklogEntry) -> EmbedOutcome,
    ) -> Result<DrainReport> {
        let rows = self.all()?;
        let mut ordered: Vec<(&MemoryId, &BacklogEntry)> =
            rows.iter().filter(|(_, entry)| select(entry)).collect();
        ordered.sort_by(|a, b| {
            a.1.enqueued_at
                .cmp(&b.1.enqueued_at)
                .then_with(|| a.0.cmp(b.0))
        });
        let mut report = DrainReport::default();
        for (id, entry) in ordered.into_iter().take(limit) {
            report.attempted += 1;
            match embed_one(id, entry) {
                EmbedOutcome::Indexed => {
                    self.delete(id)?;
                    report.drained += 1;
                }
                EmbedOutcome::MemoryGone => {
                    self.delete(id)?;
                    report.dropped += 1;
                }
                EmbedOutcome::Failed(err) => {
                    self.record(id, EnqueueReason::EmbedFailed, Some(&err))?;
                    report.failed += 1;
                }
                EmbedOutcome::EmbedderUnavailable(err) => {
                    self.record(id, EnqueueReason::EmbedFailed, Some(&err))?;
                    report.stopped = Some(err);
                    break;
                }
                EmbedOutcome::WritesClosed => {
                    report.stopped = Some(crate::memory::write_gate::WritesClosed.to_string());
                    break;
                }
            }
        }
        report.remaining = self.all()?.len();
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemKv {
        map: Mutex<BTreeMap<Vec<u8>, Vec<u8>>>,
        poison: Mutex<bool>,
    }

    impl KeyValueStore for MemKv {
        fn backend(&self) -> crate::config::StorageBackend {
            crate::config::StorageBackend::Redb
        }
        fn get_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
            Ok(self.map.lock().unwrap().get(key).cloned())
        }
        fn put_bytes(&self, key: &[u8], value: &[u8]) -> Result<()> {
            self.map
                .lock()
                .unwrap()
                .insert(key.to_vec(), value.to_vec());
            Ok(())
        }
        fn delete_bytes(&self, key: &[u8]) -> Result<()> {
            self.map.lock().unwrap().remove(key);
            Ok(())
        }
        fn scan_prefix(
            &self,
            prefix: &[u8],
            _limit: Option<usize>,
        ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            if *self.poison.lock().unwrap() {
                anyhow::bail!("scan failed");
            }
            Ok(self
                .map
                .lock()
                .unwrap()
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect())
        }
        fn flush(&self) -> Result<()> {
            Ok(())
        }
        fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
            for (k, v) in pairs {
                self.put_bytes(k, v)?;
            }
            Ok(())
        }
        fn delete_many(&self, keys: &[Vec<u8>]) -> Result<()> {
            for k in keys {
                self.delete_bytes(k)?;
            }
            Ok(())
        }
    }

    fn store() -> (Arc<MemKv>, EmbeddingBacklogStore) {
        let kv = Arc::new(MemKv::default());
        (kv.clone(), EmbeddingBacklogStore::with_kv(kv))
    }

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn record_keeps_enqueue_time_and_counts_attempts() {
        let (_, s) = store();
        s.record(
            &id(1),
            EnqueueReason::EmbedFailed,
            Some("connection refused"),
        )
        .unwrap();
        let first = s.get(&id(1)).unwrap().unwrap();
        assert_eq!(first.attempts, 1);
        assert_eq!(first.reason, EnqueueReason::EmbedFailed);
        s.record(&id(1), EnqueueReason::Reconciled, Some("timeout"))
            .unwrap();
        let second = s.get(&id(1)).unwrap().unwrap();
        assert_eq!(second.attempts, 2);
        assert_eq!(
            second.reason,
            EnqueueReason::EmbedFailed,
            "the first reason is kept"
        );
        assert_eq!(second.enqueued_at, first.enqueued_at);
        assert_eq!(second.last_error.as_deref(), Some("timeout"));
    }

    #[test]
    fn a_reconciled_row_starts_with_zero_attempts() {
        let (_, s) = store();
        s.record(&id(2), EnqueueReason::Reconciled, None).unwrap();
        assert_eq!(s.get(&id(2)).unwrap().unwrap().attempts, 0);
    }

    #[test]
    fn reconcile_enqueues_stored_but_unindexed_and_drops_the_rest() {
        let (_, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, Some("x"))
            .unwrap();
        s.record(&id(9), EnqueueReason::EmbedFailed, Some("x"))
            .unwrap();
        let stored = vec![id(1), id(2), id(3)];
        let indexed: HashSet<MemoryId> = [id(1), id(3)].into_iter().collect();
        let r = s.reconcile(&stored, &indexed).unwrap();
        assert_eq!(r.dropped_indexed, 1, "id 1 is indexed now");
        assert_eq!(r.dropped_gone, 1, "id 9 is no longer stored");
        assert_eq!(r.enqueued, 1, "id 2 is stored and unindexed with no row");
        assert_eq!(r.pending, 1);
        assert_eq!(
            s.get(&id(2)).unwrap().unwrap().reason,
            EnqueueReason::Reconciled
        );
    }

    #[test]
    fn drain_embeds_oldest_first_and_stops_when_the_embedder_is_down() {
        let (_, s) = store();
        for n in [3u128, 1, 2] {
            s.record(&id(n), EnqueueReason::EmbedFailed, Some("x"))
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let mut calls = Vec::new();
        let r = s
            .drain_with(10, |m, _| {
                calls.push(m.clone());
                if *m == id(2) {
                    EmbedOutcome::EmbedderUnavailable("connection refused".into())
                } else {
                    EmbedOutcome::Indexed
                }
            })
            .unwrap();
        assert_eq!(
            calls,
            vec![id(3), id(1), id(2)],
            "enqueue order, not id order"
        );
        assert_eq!(r.drained, 2);
        assert_eq!(r.stopped.as_deref(), Some("connection refused"));
        assert_eq!(r.remaining, 1);
        assert_eq!(s.get(&id(2)).unwrap().unwrap().attempts, 2);
    }

    #[test]
    fn drain_drops_gone_memories_and_keeps_per_memory_failures() {
        let (_, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, None).unwrap();
        s.record(&id(2), EnqueueReason::EmbedFailed, None).unwrap();
        let r = s
            .drain_with(10, |m, _| {
                if *m == id(1) {
                    EmbedOutcome::MemoryGone
                } else {
                    EmbedOutcome::Failed("oversized".into())
                }
            })
            .unwrap();
        assert_eq!((r.dropped, r.failed, r.remaining), (1, 1, 1));
        assert!(s.get(&id(1)).unwrap().is_none());
        assert_eq!(
            s.get(&id(2)).unwrap().unwrap().last_error.as_deref(),
            Some("oversized")
        );
    }

    #[test]
    fn a_tenant_closed_for_shutdown_stops_the_drain_and_records_nothing() {
        let (_, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, Some("x"))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        s.record(&id(2), EnqueueReason::EmbedFailed, Some("x"))
            .unwrap();
        let before = s.get(&id(1)).unwrap().unwrap();

        let mut calls = 0;
        let r = s
            .drain_with(10, |_, _| {
                calls += 1;
                EmbedOutcome::WritesClosed
            })
            .unwrap();

        assert_eq!(calls, 1, "the next row would be refused the same way");
        assert_eq!((r.drained, r.failed, r.remaining), (0, 0, 2));
        assert!(r.stopped.is_some());
        assert_eq!(s.get(&id(1)).unwrap().unwrap(), before);
    }

    #[test]
    fn limit_bounds_a_drain_and_remaining_is_reported() {
        let (_, s) = store();
        for n in 1..=5u128 {
            s.record(&id(n), EnqueueReason::Reconciled, None).unwrap();
        }
        let r = s.drain_with(2, |_, _| EmbedOutcome::Indexed).unwrap();
        assert_eq!((r.attempted, r.drained, r.remaining), (2, 2, 3));
    }

    #[test]
    fn report_summarises_without_hiding_the_last_error() {
        let (_, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, Some("refused"))
            .unwrap();
        s.record(&id(2), EnqueueReason::Reconciled, None).unwrap();
        let r = s.report().unwrap();
        assert_eq!(r.pending, 2);
        assert_eq!(r.by_reason["embed-failed"], 1);
        assert_eq!(r.by_reason["reconciled"], 1);
        assert_eq!(r.max_attempts, 1);
        assert_eq!(r.last_error.as_deref(), Some("refused"));
        assert!(r.oldest_enqueued_at.is_some());
    }

    #[test]
    fn a_scan_error_is_an_error_not_an_empty_backlog() {
        let (kv, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, None).unwrap();
        *kv.poison.lock().unwrap() = true;
        assert!(s.all().is_err());
        assert!(s.report().is_err());
        assert!(s.drain_with(1, |_, _| EmbedOutcome::Indexed).is_err());
    }

    #[test]
    fn rows_queued_at_open_are_index_only_and_leave_existing_rows_alone() {
        let (_, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, Some("refused"))
            .unwrap();
        let before = s.get(&id(1)).unwrap().unwrap();

        assert_eq!(s.enqueue_unindexed(&[id(1), id(2), id(3)]).unwrap(), 2);
        assert_eq!(s.enqueue_unindexed(&[id(2)]).unwrap(), 0, "already queued");

        assert_eq!(s.get(&id(1)).unwrap().unwrap(), before);
        let queued = s.get(&id(2)).unwrap().unwrap();
        assert!(queued.index_only);
        assert_eq!(queued.reason, EnqueueReason::Reconciled);
        assert_eq!(queued.attempts, 0);

        let mut seen = Vec::new();
        s.drain_with(10, |m, entry| {
            seen.push((m.clone(), entry.index_only));
            EmbedOutcome::Indexed
        })
        .unwrap();
        assert_eq!(seen, vec![(id(1), false), (id(2), true), (id(3), true)]);
    }

    #[test]
    fn a_selective_drain_leaves_the_other_rows_queued() {
        let (_, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, Some("refused"))
            .unwrap();
        s.enqueue_unindexed(&[id(2)]).unwrap();
        let mut seen = Vec::new();
        let r = s
            .drain_where(
                10,
                |entry| entry.index_only,
                |m, _| {
                    seen.push(m.clone());
                    EmbedOutcome::Indexed
                },
            )
            .unwrap();
        assert_eq!(seen, vec![id(2)]);
        assert_eq!((r.attempted, r.drained, r.remaining), (1, 1, 1));
        assert!(s.get(&id(1)).unwrap().is_some());
    }

    /// The flag is left out of rows that do not carry it, so a row reads the same to a build
    /// that predates it; one that does carry it is still a row such a build can read.
    #[test]
    fn rows_without_the_flag_serialize_as_before() {
        let (kv, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, None).unwrap();
        s.enqueue_unindexed(&[id(2)]).unwrap();
        let raw = |n: u128| {
            let bytes = kv.get_bytes(&key(&id(n))).unwrap().unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };
        assert!(raw(1).get("index_only").is_none());
        assert_eq!(raw(2)["index_only"], serde_json::Value::Bool(true));

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct RowBeforeTheFlag {
            v: u32,
            reason: EnqueueReason,
            enqueued_at: DateTime<Utc>,
            attempts: u32,
            #[serde(default)]
            last_attempt_at: Option<DateTime<Utc>>,
            #[serde(default)]
            last_error: Option<String>,
        }
        let bytes = kv.get_bytes(&key(&id(2))).unwrap().unwrap();
        assert!(serde_json::from_slice::<RowBeforeTheFlag>(&bytes).is_ok());
    }
}
