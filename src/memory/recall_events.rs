//! One row per recall, including recalls that returned nothing.
//!
//! The LTR trace store cannot answer "what was asked": it writes one row per RETURNED memory, so a
//! recall that found nothing leaves no row at all, its rows share no recall id, and feedback moves
//! them between prefixes. This log is the question side: what was asked, by which episode, through
//! which endpoint, and what came back — so "asked but not known" is a query over it.
//!
//! Query text is redacted token by token with the entity redactor before it is stored. Rows live in
//! the user's own KV keyspace under a time-ordered key and are bounded by a ring cap.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::storage::KeyValueStore;

pub(crate) const RECALL_EVENT_PREFIX: &str = "recall_event:";
pub const RECALL_EVENT_MAX: usize = 20_000;
/// The cap is enforced every this many writes, not on each: enforcement scans the prefix.
const ENFORCE_EVERY: u64 = 256;
pub const RECALL_EVENTS_SCHEMA: u32 = 1;
/// Proactive context sends whole prompts; the log keeps the head of each.
pub const RECALL_QUERY_MAX_CHARS: usize = 500;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallHit {
    pub memory_id: String,
    pub rank: usize,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallEvent {
    pub v: u32,
    pub recall_id: String,
    pub at: DateTime<Utc>,
    pub endpoint: String,
    pub query: String,
    /// Length of the query as asked, before truncation to `RECALL_QUERY_MAX_CHARS`.
    pub query_chars: usize,
    /// Tokens in the query replaced by a credential class label.
    pub query_redactions: usize,
    pub episode_id: Option<String>,
    pub result_count: usize,
    pub best_score: Option<f32>,
    pub hits: Vec<RecallHit>,
}

impl RecallEvent {
    pub fn new(
        endpoint: &str,
        query: &str,
        episode_id: Option<String>,
        hits: Vec<RecallHit>,
    ) -> Self {
        let query_chars = query.chars().count();
        let (query, query_redactions) = redact_query(query);
        let query: String = query.chars().take(RECALL_QUERY_MAX_CHARS).collect();
        let best_score = hits.iter().map(|h| h.score).reduce(f32::max);
        Self {
            v: RECALL_EVENTS_SCHEMA,
            recall_id: uuid::Uuid::new_v4().to_string(),
            at: Utc::now(),
            endpoint: endpoint.to_string(),
            query,
            query_chars,
            query_redactions,
            episode_id: episode_id.filter(|e| !e.trim().is_empty()),
            result_count: hits.len(),
            best_score,
            hits,
        }
    }

    pub fn hits_from_scored<'a>(
        scored: impl IntoIterator<Item = (&'a str, f32)>,
    ) -> Vec<RecallHit> {
        scored
            .into_iter()
            .enumerate()
            .map(|(rank, (id, score))| RecallHit {
                memory_id: id.to_string(),
                rank,
                score,
            })
            .collect()
    }
}

fn redact_query(query: &str) -> (String, usize) {
    let mut count = 0;
    let redacted = query
        .split_whitespace()
        .map(|token| match crate::redaction::classify(token) {
            Some(class) => {
                count += 1;
                class.label()
            }
            None => token,
        })
        .collect::<Vec<_>>()
        .join(" ");
    (redacted, count)
}

pub struct RecallEventStore {
    kv: Arc<dyn KeyValueStore>,
    writes: AtomicU64,
}

impl RecallEventStore {
    pub fn with_kv(kv: Arc<dyn KeyValueStore>) -> Self {
        Self {
            kv,
            writes: AtomicU64::new(0),
        }
    }

    fn key(event: &RecallEvent) -> Vec<u8> {
        let millis = event.at.timestamp_millis().max(0);
        format!("{RECALL_EVENT_PREFIX}{millis:020}:{}", event.recall_id).into_bytes()
    }

    pub fn record(&self, event: &RecallEvent) -> Result<()> {
        let bytes = serde_json::to_vec(event).context("encode recall event")?;
        self.kv.put_bytes(&Self::key(event), &bytes)?;
        if self.writes.fetch_add(1, Ordering::Relaxed) % ENFORCE_EVERY == ENFORCE_EVERY - 1 {
            self.enforce_bound(RECALL_EVENT_MAX)?;
        }
        Ok(())
    }

    /// Oldest rows go first. Keys are time-ordered, so the scan order is the eviction order.
    pub fn enforce_bound(&self, max: usize) -> Result<usize> {
        let rows = self
            .kv
            .scan_prefix(RECALL_EVENT_PREFIX.as_bytes(), None)
            .context("scan recall events for bound")?;
        if rows.len() <= max {
            return Ok(0);
        }
        let mut keys: Vec<Vec<u8>> = rows.into_iter().map(|(k, _)| k).collect();
        keys.sort();
        let evicted = keys.len() - max;
        let excess: Vec<Vec<u8>> = keys.into_iter().take(evicted).collect();
        self.kv.delete_many(&excess)?;
        Ok(evicted)
    }

    /// Events at or after `since`, oldest first. A scan error or an undecodable row is an error:
    /// an empty log has to mean nobody asked, not that the read failed.
    pub fn since(&self, since: DateTime<Utc>) -> Result<Vec<RecallEvent>> {
        let rows = self
            .kv
            .scan_prefix(RECALL_EVENT_PREFIX.as_bytes(), None)
            .context("scan recall events")?;
        let mut out = Vec::with_capacity(rows.len());
        for (key, value) in rows {
            let event: RecallEvent = serde_json::from_slice(&value).with_context(|| {
                format!("decode recall event {}", String::from_utf8_lossy(&key))
            })?;
            if event.at >= since {
                out.push(event);
            }
        }
        out.sort_by(|a, b| a.at.cmp(&b.at).then_with(|| a.recall_id.cmp(&b.recall_id)));
        Ok(out)
    }

    /// The oldest retained event, which bounds what `since` can answer.
    pub fn horizon(&self) -> Result<Option<DateTime<Utc>>> {
        Ok(self.since(DateTime::<Utc>::MIN_UTC)?.first().map(|e| e.at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::test_support::key_value_stores;

    #[test]
    fn an_empty_recall_is_recorded_as_a_row() {
        let event = RecallEvent::new("/api/recall", "who owns the billing service", None, vec![]);
        assert_eq!(event.result_count, 0);
        assert_eq!(event.best_score, None);
        for (backend, kv, _tmp) in key_value_stores("recall_events_test") {
            let store = RecallEventStore::with_kv(kv);
            store.record(&event).unwrap();
            let got = store.since(DateTime::<Utc>::MIN_UTC).unwrap();
            assert_eq!(got, vec![event.clone()], "{backend:?}");
        }
    }

    #[test]
    fn a_credential_in_the_query_never_reaches_the_log() {
        let event = RecallEvent::new(
            "/api/relevant",
            "why does ghp_abcdefghijklmnopqrstuvwxyz0123 fail",
            Some("transcript-1".into()),
            vec![],
        );
        assert!(!event.query.contains("ghp_"), "{}", event.query);
        assert_eq!(event.query_redactions, 1);
        assert_eq!(event.episode_id.as_deref(), Some("transcript-1"));
    }

    #[test]
    fn a_long_query_is_truncated_after_redaction_and_its_length_kept() {
        let long = "word ".repeat(400);
        let event = RecallEvent::new("/api/proactive_context", &long, None, vec![]);
        assert_eq!(event.query.chars().count(), RECALL_QUERY_MAX_CHARS);
        assert_eq!(event.query_chars, long.chars().count());
    }

    #[test]
    fn hits_carry_rank_and_best_score() {
        let hits = RecallEvent::hits_from_scored([("a", 0.4), ("b", 0.9)]);
        let event = RecallEvent::new("/api/recall", "q", Some("  ".into()), hits);
        assert_eq!(event.hits[1].rank, 1);
        assert_eq!(event.best_score, Some(0.9));
        assert_eq!(event.episode_id, None, "a blank episode is no episode");
    }

    #[test]
    fn bound_evicts_oldest_and_since_filters_and_corruption_is_an_error() {
        for (backend, kv, _tmp) in key_value_stores("recall_events_bound") {
            let store = RecallEventStore::with_kv(kv.clone());
            let mut events = Vec::new();
            for i in 0..5 {
                let mut e = RecallEvent::new("/api/recall", &format!("q{i}"), None, vec![]);
                e.at = DateTime::<Utc>::from_timestamp(1_700_000_000 + i, 0).unwrap();
                store.record(&e).unwrap();
                events.push(e);
            }
            assert_eq!(store.enforce_bound(3).unwrap(), 2, "{backend:?}");
            let left = store.since(DateTime::<Utc>::MIN_UTC).unwrap();
            assert_eq!(left, events[2..].to_vec(), "{backend:?}: oldest evicted");
            assert_eq!(store.horizon().unwrap(), Some(events[2].at), "{backend:?}");
            assert_eq!(store.since(events[4].at).unwrap().len(), 1, "{backend:?}");

            kv.put_bytes(format!("{RECALL_EVENT_PREFIX}zz").as_bytes(), b"{")
                .unwrap();
            assert!(
                store.since(DateTime::<Utc>::MIN_UTC).is_err(),
                "{backend:?}"
            );
        }
    }
}
