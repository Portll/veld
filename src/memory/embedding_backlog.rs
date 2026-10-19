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
//! A row targets a space. A row under `embed_backlog:<id>` targets the primary index's space
//! at drain time, the only target this ledger had before spaces were named. A row under
//! `embed_backlog_space:<id>/<space>` targets that space, today the secondary index's: the
//! separate prefix keeps a binary that predates it from reading a key it cannot parse.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::embeddings::space::SpaceId;
use crate::memory::MemoryId;
use crate::storage::KeyValueStore;

const PREFIX: &str = "embed_backlog:";
const SPACE_PREFIX: &str = "embed_backlog_space:";
const UUID_LEN: usize = 36;
pub const SCHEMA: u32 = 2;

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
    /// The space the vector this row waits for must be in. Absent on rows written before
    /// spaces were named, which target the primary index.
    #[serde(default)]
    pub target_space: Option<SpaceId>,
    /// The space of the vector the memory already stores, when an index refused it for being
    /// in another space than the index's.
    #[serde(default)]
    pub stored_space: Option<SpaceId>,
}

/// Which row: the memory and the space its vector is wanted in. `space: None` is the primary
/// index's space at drain time.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BacklogKey {
    pub memory: MemoryId,
    pub space: Option<SpaceId>,
}

impl BacklogKey {
    pub fn primary(memory: &MemoryId) -> Self {
        Self {
            memory: memory.clone(),
            space: None,
        }
    }

    pub fn in_space(memory: &MemoryId, space: &SpaceId) -> Self {
        Self {
            memory: memory.clone(),
            space: Some(space.clone()),
        }
    }

    fn encode(&self) -> Vec<u8> {
        match &self.space {
            None => format!("{PREFIX}{}", self.memory.0).into_bytes(),
            Some(space) => format!("{SPACE_PREFIX}{}/{space}", self.memory.0).into_bytes(),
        }
    }

    /// A key read back from the store. The uuid has a fixed width, so a space id may hold any
    /// character, `/` included.
    fn decode(raw: &[u8]) -> Result<Self> {
        let key = std::str::from_utf8(raw).context("embedding backlog key is not utf-8")?;
        let (rest, spaced) = match key.strip_prefix(SPACE_PREFIX) {
            Some(rest) => (rest, true),
            None => (
                key.strip_prefix(PREFIX)
                    .context("embedding backlog key without prefix")?,
                false,
            ),
        };
        let id_part = rest.get(..UUID_LEN).unwrap_or(rest);
        let memory = MemoryId(
            uuid::Uuid::parse_str(id_part)
                .with_context(|| format!("embedding backlog key {key} is not a memory id"))?,
        );
        let tail = &rest[id_part.len()..];
        let space = match (spaced, tail.strip_prefix('/')) {
            (false, _) if tail.is_empty() => None,
            (true, Some(space)) => Some(
                SpaceId::parse(space)
                    .with_context(|| format!("embedding backlog key {key} names no space"))?,
            ),
            _ => anyhow::bail!("embedding backlog key {key} is malformed"),
        };
        Ok(Self { memory, space })
    }
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
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BacklogReport {
    pub witness: &'static str,
    pub pending: usize,
    pub oldest_enqueued_at: Option<DateTime<Utc>>,
    pub max_attempts: u32,
    pub by_reason: BTreeMap<String, usize>,
    pub last_error: Option<String>,
    /// Rows by the space they wait for; `primary` for rows that name none.
    pub by_target_space: BTreeMap<String, usize>,
    /// Rows whose memory stores a vector in another space than the index that refused it.
    pub space_mismatches: usize,
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

impl EmbeddingBacklogStore {
    pub fn with_kv(kv: Arc<dyn KeyValueStore>) -> Self {
        Self { kv }
    }

    pub fn get(&self, id: &MemoryId) -> Result<Option<BacklogEntry>> {
        self.get_key(&BacklogKey::primary(id))
    }

    pub fn get_key(&self, key: &BacklogKey) -> Result<Option<BacklogEntry>> {
        match self
            .kv
            .get_bytes(&key.encode())
            .context("read embedding backlog row")?
        {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
                format!("decode embedding backlog row for {}", key.memory.0)
            })?)),
            None => Ok(None),
        }
    }

    pub fn put(&self, id: &MemoryId, entry: &BacklogEntry) -> Result<()> {
        self.put_key(&BacklogKey::primary(id), entry)
    }

    fn put_key(&self, key: &BacklogKey, entry: &BacklogEntry) -> Result<()> {
        let bytes = serde_json::to_vec(entry).context("encode embedding backlog row")?;
        self.kv
            .put_bytes(&key.encode(), &bytes)
            .context("write embedding backlog row")
    }

    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        self.delete_key(&BacklogKey::primary(id))
    }

    pub fn delete_key(&self, key: &BacklogKey) -> Result<()> {
        self.kv
            .delete_bytes(&key.encode())
            .context("delete embedding backlog row")
    }

    /// Every row targeting the primary index. A scan or decode error is an error, never an
    /// empty backlog.
    pub fn all(&self) -> Result<BTreeMap<MemoryId, BacklogEntry>> {
        Ok(self
            .scan(PREFIX)?
            .into_iter()
            .map(|(key, entry)| (key.memory, entry))
            .collect())
    }

    /// Every row, whatever space it targets.
    pub fn rows(&self) -> Result<BTreeMap<BacklogKey, BacklogEntry>> {
        let mut rows = self.scan(PREFIX)?;
        rows.extend(self.scan(SPACE_PREFIX)?);
        Ok(rows)
    }

    fn scan(&self, prefix: &str) -> Result<BTreeMap<BacklogKey, BacklogEntry>> {
        let rows = self
            .kv
            .scan_prefix(prefix.as_bytes(), None)
            .context("scan embedding backlog")?;
        let mut out = BTreeMap::new();
        for (k, v) in rows {
            let key = BacklogKey::decode(&k)?;
            let entry: BacklogEntry = serde_json::from_slice(&v)
                .with_context(|| format!("decode embedding backlog row for {}", key.memory.0))?;
            out.insert(key, entry);
        }
        Ok(out)
    }

    /// Record that a memory needs a vector in the primary index. An existing row keeps its
    /// enqueue time and reason; the attempt count and last error move.
    pub fn record(&self, id: &MemoryId, reason: EnqueueReason, error: Option<&str>) -> Result<()> {
        self.record_key(&BacklogKey::primary(id), reason, None, None, error)
    }

    /// Record that an index refused a memory's stored vector for being in `stored`, not the
    /// index's `target`. The row waits for a vector in `target`.
    pub fn record_mismatch(
        &self,
        key: &BacklogKey,
        stored: &SpaceId,
        target: &SpaceId,
    ) -> Result<()> {
        let error = format!("stored vector is in {stored}; the index holds {target}");
        self.record_key(
            key,
            EnqueueReason::IndexFailed,
            Some(target),
            Some(stored),
            Some(&error),
        )
    }

    /// Record a row under `key`. Spaces given here fill a row that lacks them; a known space
    /// is never replaced, so a row's first diagnosis survives retries.
    pub fn record_key(
        &self,
        key: &BacklogKey,
        reason: EnqueueReason,
        target_space: Option<&SpaceId>,
        stored_space: Option<&SpaceId>,
        error: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now();
        let entry = match self.get_key(key)? {
            Some(mut existing) => {
                existing.attempts += 1;
                existing.last_attempt_at = Some(now);
                existing.last_error = error.map(str::to_string);
                if existing.target_space.is_none() {
                    existing.target_space = target_space.cloned();
                }
                if existing.stored_space.is_none() {
                    existing.stored_space = stored_space.cloned();
                }
                existing
            }
            None => BacklogEntry {
                v: SCHEMA,
                reason,
                enqueued_at: now,
                attempts: u32::from(error.is_some()),
                last_attempt_at: error.map(|_| now),
                last_error: error.map(str::to_string),
                target_space: target_space.cloned(),
                stored_space: stored_space.cloned(),
            },
        };
        self.put_key(key, &entry)
    }

    pub fn report(&self) -> Result<BacklogReport> {
        let rows = self.rows()?;
        let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
        let mut by_target_space: BTreeMap<String, usize> = BTreeMap::new();
        let mut space_mismatches = 0usize;
        let mut max_attempts = 0u32;
        let mut oldest: Option<DateTime<Utc>> = None;
        let mut last_error = None;
        let mut last_error_at: Option<DateTime<Utc>> = None;
        for (key, entry) in &rows {
            let target = key
                .space
                .as_ref()
                .or(entry.target_space.as_ref())
                .map_or_else(|| "primary".to_string(), SpaceId::to_string);
            *by_target_space.entry(target).or_default() += 1;
            if entry.stored_space.is_some() {
                space_mismatches += 1;
            }
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
            by_target_space,
            space_mismatches,
        })
    }

    /// Bring the rows in line with the store and the index. `stored` is every live memory id,
    /// `indexed` every id the primary vector index holds. A stored id the index lacks gets a
    /// row; a primary row whose id is indexed, and any row whose id is no longer stored, is
    /// dropped. A row for another space stays until its drain: `indexed` says nothing of it.
    pub fn reconcile(
        &self,
        stored: &[MemoryId],
        indexed: &std::collections::HashSet<MemoryId>,
    ) -> Result<ReconcileReport> {
        let rows = self.rows()?;
        let stored_set: std::collections::HashSet<&MemoryId> = stored.iter().collect();
        let mut report = ReconcileReport {
            stored: stored.len(),
            indexed: indexed.len(),
            ..Default::default()
        };
        for key in rows.keys() {
            if !stored_set.contains(&key.memory) {
                self.delete_key(key)?;
                report.dropped_gone += 1;
            } else if key.space.is_none() && indexed.contains(&key.memory) {
                self.delete_key(key)?;
                report.dropped_indexed += 1;
            }
        }
        let rows: BTreeMap<MemoryId, ()> = rows
            .into_keys()
            .filter(|key| key.space.is_none())
            .map(|key| (key.memory, ()))
            .collect();
        for id in stored {
            if !indexed.contains(id) && !rows.contains_key(id) {
                self.record(id, EnqueueReason::Reconciled, None)?;
                report.enqueued += 1;
            }
        }
        report.pending = self.rows()?.len();
        Ok(report)
    }

    /// Embed oldest first, at most `limit`, through `embed_one`. Stops at the first
    /// `EmbedderUnavailable` so a down host is asked once per drain, not once per row.
    pub fn drain_with(
        &self,
        limit: usize,
        mut embed_one: impl FnMut(&BacklogKey) -> EmbedOutcome,
    ) -> Result<DrainReport> {
        let rows = self.rows()?;
        let mut ordered: Vec<(&BacklogKey, &BacklogEntry)> = rows.iter().collect();
        ordered.sort_by(|a, b| {
            a.1.enqueued_at
                .cmp(&b.1.enqueued_at)
                .then_with(|| a.0.cmp(b.0))
        });
        let mut report = DrainReport::default();
        for (key, _) in ordered.into_iter().take(limit) {
            report.attempted += 1;
            match embed_one(key) {
                EmbedOutcome::Indexed => {
                    self.delete_key(key)?;
                    report.drained += 1;
                }
                EmbedOutcome::MemoryGone => {
                    self.delete_key(key)?;
                    report.dropped += 1;
                }
                EmbedOutcome::Failed(err) => {
                    self.record_key(key, EnqueueReason::EmbedFailed, None, None, Some(&err))?;
                    report.failed += 1;
                }
                EmbedOutcome::EmbedderUnavailable(err) => {
                    self.record_key(key, EnqueueReason::EmbedFailed, None, None, Some(&err))?;
                    report.stopped = Some(err);
                    break;
                }
            }
        }
        report.remaining = self.rows()?.len();
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
            .drain_with(10, |key| {
                let m = &key.memory;
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
            .drain_with(10, |key| {
                if key.memory == id(1) {
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
    fn limit_bounds_a_drain_and_remaining_is_reported() {
        let (_, s) = store();
        for n in 1..=5u128 {
            s.record(&id(n), EnqueueReason::Reconciled, None).unwrap();
        }
        let r = s.drain_with(2, |_| EmbedOutcome::Indexed).unwrap();
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

    fn space(raw: &str) -> SpaceId {
        SpaceId::parse(raw).unwrap()
    }

    #[test]
    fn a_key_round_trips_with_any_space_id() {
        let primary = BacklogKey::primary(&id(7));
        assert_eq!(BacklogKey::decode(&primary.encode()).unwrap(), primary);
        let spaced = BacklogKey::in_space(&id(7), &space("nomic-ai/nomic-embed-text-v1.5@768"));
        assert_eq!(BacklogKey::decode(&spaced.encode()).unwrap(), spaced);
        assert!(BacklogKey::decode(b"embed_backlog_space:not-a-uuid/x@1").is_err());
        assert!(
            BacklogKey::decode(format!("{PREFIX}{}/minilm-l6-v2@384", id(7).0).as_bytes()).is_err()
        );
    }

    #[test]
    fn a_space_row_is_invisible_to_the_primary_scan_an_older_binary_runs() {
        let (_, s) = store();
        let nomic = space("nomic-embed-text-v1.5@768");
        let minilm = space("minilm-l6-v2@384");
        s.record_mismatch(&BacklogKey::in_space(&id(1), &minilm), &nomic, &minilm)
            .unwrap();
        assert!(s.all().unwrap().is_empty());
        assert_eq!(s.rows().unwrap().len(), 1);
    }

    #[test]
    fn a_mismatch_row_names_both_spaces_and_keeps_its_first_diagnosis() {
        let (_, s) = store();
        let stored = space("unrecorded@384");
        let target = space("nomic-embed-text-v1.5@768");
        s.record_mismatch(&BacklogKey::primary(&id(1)), &stored, &target)
            .unwrap();
        s.record(&id(1), EnqueueReason::EmbedFailed, Some("timeout"))
            .unwrap();
        let row = s.get(&id(1)).unwrap().unwrap();
        assert_eq!(row.reason, EnqueueReason::IndexFailed);
        assert_eq!(row.stored_space, Some(stored));
        assert_eq!(row.target_space, Some(target.clone()));
        assert_eq!(row.attempts, 2);
        s.record(&id(2), EnqueueReason::EmbedFailed, Some("refused"))
            .unwrap();
        let r = s.report().unwrap();
        assert_eq!(r.space_mismatches, 1);
        assert_eq!(r.by_target_space[target.as_str()], 1);
        assert_eq!(r.by_target_space["primary"], 1);
    }

    #[test]
    fn reconcile_keeps_a_space_row_until_its_memory_is_gone() {
        let (_, s) = store();
        let minilm = space("minilm-l6-v2@384");
        let key = BacklogKey::in_space(&id(1), &minilm);
        s.record_key(&key, EnqueueReason::IndexFailed, Some(&minilm), None, None)
            .unwrap();
        let indexed: HashSet<MemoryId> = [id(1)].into_iter().collect();
        s.reconcile(&[id(1)], &indexed).unwrap();
        assert!(
            s.get_key(&key).unwrap().is_some(),
            "the primary index holding the memory says nothing of the secondary"
        );
        let r = s.reconcile(&[], &HashSet::new()).unwrap();
        assert_eq!(r.dropped_gone, 1);
        assert!(s.get_key(&key).unwrap().is_none());
    }

    #[test]
    fn a_drain_reaches_space_rows_and_clears_them() {
        let (_, s) = store();
        let minilm = space("minilm-l6-v2@384");
        let key = BacklogKey::in_space(&id(1), &minilm);
        s.record_key(&key, EnqueueReason::IndexFailed, Some(&minilm), None, None)
            .unwrap();
        let mut seen = Vec::new();
        let r = s
            .drain_with(10, |k| {
                seen.push(k.clone());
                EmbedOutcome::Indexed
            })
            .unwrap();
        assert_eq!(seen, vec![key]);
        assert_eq!((r.drained, r.remaining), (1, 0));
    }

    #[test]
    fn a_scan_error_is_an_error_not_an_empty_backlog() {
        let (kv, s) = store();
        s.record(&id(1), EnqueueReason::EmbedFailed, None).unwrap();
        *kv.poison.lock().unwrap() = true;
        assert!(s.all().is_err());
        assert!(s.report().is_err());
        assert!(s.drain_with(1, |_| EmbedOutcome::Indexed).is_err());
    }
}
