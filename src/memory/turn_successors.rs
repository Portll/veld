//! Persisted successor index for conversation turns: prior turn → the turns that followed it.
//!
//! A turn names the turn it follows (`preceding_memory_id` on its episode context), so
//! walking back from a turn is a read of the record. Walking forward is not: nothing on a
//! turn names the turn that answered it. Recall's neighbour expansion needs the forward
//! direction, because an anaphoric reply ("I've had them for 3 years") shares no word with
//! the question about it while the turn before it does. So when `remember` resolves the
//! prior turn of a new turn (from either source in [`super::index_text::PriorSource`]), it
//! writes one row here, in the user's own KV keyspace, and the index survives a restart on
//! either engine.
//!
//! Rows are `turn_successor:<prior>:<successor>`, one per pair, so a prior's successors are
//! one prefix scan and two writers threading the same predecessor each keep their row
//! without a read-modify-write. Forgetting a memory leaves its rows in place; readers skip
//! a successor that no longer resolves to a live memory.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::memory::types::MemoryId;
use crate::storage::KeyValueStore;

pub(crate) const TURN_SUCCESSOR_PREFIX: &str = "turn_successor:";

/// Row value: the row-format version. The key carries the whole pair.
const ROW_V1: &[u8] = &[1];

/// Most rows read for one prior turn. A turn is normally answered once; the bound keeps
/// a turn that many writers threaded onto from turning one lookup into a long scan.
pub const SUCCESSOR_SCAN_LIMIT: usize = 16;

pub struct TurnSuccessorStore {
    kv: Arc<dyn KeyValueStore>,
}

impl TurnSuccessorStore {
    pub fn with_kv(kv: Arc<dyn KeyValueStore>) -> Self {
        Self { kv }
    }

    fn prefix(prior: &MemoryId) -> String {
        format!("{TURN_SUCCESSOR_PREFIX}{}:", prior.0)
    }

    fn key(prior: &MemoryId, successor: &MemoryId) -> Vec<u8> {
        format!("{}{}", Self::prefix(prior), successor.0).into_bytes()
    }

    /// Record that `successor` follows `prior`. Idempotent; a turn cannot follow itself.
    pub fn record(&self, prior: &MemoryId, successor: &MemoryId) -> Result<()> {
        if prior == successor {
            return Ok(());
        }
        self.kv
            .put_bytes(&Self::key(prior, successor), ROW_V1)
            .with_context(|| format!("record turn successor {} -> {}", prior.0, successor.0))
    }

    /// The turns recorded as following `prior`, in key order, at most
    /// [`SUCCESSOR_SCAN_LIMIT`]. Liveness is the caller's to check. A row whose key does
    /// not end in a memory id is skipped with a warning rather than failing the read: the
    /// index ranks candidates, and one bad row must not hide the good ones.
    pub fn successors(&self, prior: &MemoryId) -> Result<Vec<MemoryId>> {
        let prefix = Self::prefix(prior);
        let rows = self
            .kv
            .scan_prefix(prefix.as_bytes(), Some(SUCCESSOR_SCAN_LIMIT))
            .with_context(|| format!("scan turn successors of {}", prior.0))?;
        let mut out = Vec::with_capacity(rows.len());
        for (key, _) in rows {
            let parsed = std::str::from_utf8(&key)
                .ok()
                .and_then(|k| k.strip_prefix(prefix.as_str()))
                .and_then(|id| uuid::Uuid::parse_str(id).ok());
            match parsed {
                Some(id) => out.push(MemoryId(id)),
                None => tracing::warn!(
                    prior = %prior.0,
                    key = %String::from_utf8_lossy(&key),
                    "turn successor row with an unreadable key skipped"
                ),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::test_support::{assert_both_backends, key_value_stores};

    fn id() -> MemoryId {
        MemoryId(uuid::Uuid::new_v4())
    }

    #[test]
    fn successors_round_trip_per_prior_on_both_engines() {
        let stores = key_value_stores("turn_successors_test");
        assert_both_backends(&stores);
        for (backend, kv, _tmp) in stores {
            let store = TurnSuccessorStore::with_kv(kv);
            let (question, reply, other_prior, other_reply) = (id(), id(), id(), id());
            assert!(
                store.successors(&question).unwrap().is_empty(),
                "{backend:?}: nothing recorded yet"
            );

            store.record(&question, &reply).unwrap();
            store.record(&question, &reply).unwrap();
            store.record(&other_prior, &other_reply).unwrap();
            assert_eq!(
                store.successors(&question).unwrap(),
                vec![reply.clone()],
                "{backend:?}: one row per pair, and only this prior's"
            );
            assert_eq!(
                store.successors(&other_prior).unwrap(),
                vec![other_reply],
                "{backend:?}"
            );
            assert!(
                store.successors(&reply).unwrap().is_empty(),
                "{backend:?}: the index runs forward only"
            );
        }
    }

    #[test]
    fn a_turn_threaded_twice_keeps_both_successors() {
        for (backend, kv, _tmp) in key_value_stores("turn_successors_test") {
            let store = TurnSuccessorStore::with_kv(kv);
            let (question, first, second) = (id(), id(), id());
            store.record(&question, &first).unwrap();
            store.record(&question, &second).unwrap();
            let mut got = store.successors(&question).unwrap();
            got.sort();
            let mut want = vec![first, second];
            want.sort();
            assert_eq!(got, want, "{backend:?}");
        }
    }

    #[test]
    fn a_turn_never_follows_itself() {
        for (backend, kv, _tmp) in key_value_stores("turn_successors_test") {
            let store = TurnSuccessorStore::with_kv(kv);
            let turn = id();
            store.record(&turn, &turn).unwrap();
            assert!(store.successors(&turn).unwrap().is_empty(), "{backend:?}");
        }
    }

    #[test]
    fn an_unreadable_row_is_skipped_and_the_rest_still_read() {
        for (backend, kv, _tmp) in key_value_stores("turn_successors_test") {
            let store = TurnSuccessorStore::with_kv(kv.clone());
            let (question, reply) = (id(), id());
            store.record(&question, &reply).unwrap();
            kv.put_bytes(
                format!("{TURN_SUCCESSOR_PREFIX}{}:not-a-uuid", question.0).as_bytes(),
                ROW_V1,
            )
            .unwrap();
            assert_eq!(
                store.successors(&question).unwrap(),
                vec![reply],
                "{backend:?}"
            );
        }
    }

    #[test]
    fn the_scan_is_bounded() {
        for (backend, kv, _tmp) in key_value_stores("turn_successors_test") {
            let store = TurnSuccessorStore::with_kv(kv);
            let hub = id();
            for _ in 0..SUCCESSOR_SCAN_LIMIT + 4 {
                store.record(&hub, &id()).unwrap();
            }
            assert_eq!(
                store.successors(&hub).unwrap().len(),
                SUCCESSOR_SCAN_LIMIT,
                "{backend:?}"
            );
        }
    }
}
