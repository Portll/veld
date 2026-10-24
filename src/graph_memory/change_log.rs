//! Change log of the graph's entity and relationship rows.
//!
//! [`JournaledColumnStore`] wraps the graph's column store and records the uuid
//! of every entity or relationship row a write puts or deletes, after the write
//! succeeds. A reader holding a [`ChangeCursor`] asks for the rows changed since
//! it; the log keeps the last [`CHANGE_LOG_CAPACITY`] changes, and a cursor older
//! than that, or from another log, gets `None`.

use anyhow::Result;
use parking_lot::Mutex;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use super::{CF_ENTITIES, CF_RELATIONSHIPS};
use crate::config::StorageBackend;
use crate::storage::{BatchOp, ColumnStore, WriteBatch};

/// Changes kept; older ones are dropped first.
pub const CHANGE_LOG_CAPACITY: usize = 1 << 18;

/// Source of log epochs: each log of this process has its own.
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Entity,
    Relationship,
}

/// A position in one change log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChangeCursor {
    pub epoch: u64,
    pub seq: u64,
}

/// Rows changed between a cursor and the log's head, each uuid once.
#[derive(Debug, Default, Clone)]
pub struct GraphChanges {
    pub entities: Vec<Uuid>,
    pub relationships: Vec<Uuid>,
    /// The head when read; every change before it is included.
    pub upto: ChangeCursor,
}

impl GraphChanges {
    pub fn len(&self) -> usize {
        self.entities.len() + self.relationships.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct Ring {
    /// Sequence number of `entries[0]`.
    first: u64,
    entries: VecDeque<(Family, Uuid)>,
}

/// Bounded log of the uuids of changed entity and relationship rows.
pub struct GraphChangeLog {
    epoch: u64,
    capacity: usize,
    ring: Mutex<Ring>,
}

impl Default for GraphChangeLog {
    fn default() -> Self {
        Self::with_capacity(CHANGE_LOG_CAPACITY)
    }
}

impl GraphChangeLog {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            epoch: NEXT_EPOCH.fetch_add(1, Ordering::Relaxed),
            capacity: capacity.max(1),
            ring: Mutex::new(Ring {
                first: 0,
                entries: VecDeque::new(),
            }),
        }
    }

    /// The position after the last recorded change.
    pub fn head(&self) -> ChangeCursor {
        let ring = self.ring.lock();
        ChangeCursor {
            epoch: self.epoch,
            seq: ring.first + ring.entries.len() as u64,
        }
    }

    /// Rows changed since `cursor`; `None` when the log no longer holds every
    /// change since it, or the cursor belongs to another log.
    pub fn changes_since(&self, cursor: ChangeCursor) -> Option<GraphChanges> {
        if cursor.epoch != self.epoch {
            return None;
        }
        let (entries, head) = {
            let ring = self.ring.lock();
            let head = ring.first + ring.entries.len() as u64;
            if cursor.seq < ring.first || cursor.seq > head {
                return None;
            }
            let skip = (cursor.seq - ring.first) as usize;
            let entries: Vec<(Family, Uuid)> = ring.entries.range(skip..).copied().collect();
            (entries, head)
        };
        let mut changes = GraphChanges {
            upto: ChangeCursor {
                epoch: self.epoch,
                seq: head,
            },
            ..GraphChanges::default()
        };
        let (mut seen_entities, mut seen_edges) = (HashSet::new(), HashSet::new());
        for (family, uuid) in entries {
            match family {
                Family::Entity if seen_entities.insert(uuid) => changes.entities.push(uuid),
                Family::Relationship if seen_edges.insert(uuid) => changes.relationships.push(uuid),
                _ => {}
            }
        }
        Some(changes)
    }

    fn record(&self, changed: impl IntoIterator<Item = (Family, Uuid)>) {
        let mut ring = self.ring.lock();
        for entry in changed {
            if ring.entries.len() == self.capacity {
                ring.entries.pop_front();
                ring.first += 1;
            }
            ring.entries.push_back(entry);
        }
    }
}

/// The logged family and uuid of a row key, if the row is an entity or relationship.
fn logged_row(cf: Option<&str>, key: &[u8]) -> Option<(Family, Uuid)> {
    let family = match cf? {
        CF_ENTITIES => Family::Entity,
        CF_RELATIONSHIPS => Family::Relationship,
        _ => return None,
    };
    // Relationship lookup rows (`mem_edge:…`) are not 16 bytes.
    let uuid = Uuid::from_slice(key).ok()?;
    Some((family, uuid))
}

/// A column store that records its entity and relationship writes in a change log.
pub struct JournaledColumnStore {
    inner: Arc<dyn ColumnStore>,
    log: Arc<GraphChangeLog>,
}

impl JournaledColumnStore {
    pub fn new(inner: Arc<dyn ColumnStore>, log: Arc<GraphChangeLog>) -> Self {
        Self { inner, log }
    }
}

impl ColumnStore for JournaledColumnStore {
    fn backend(&self) -> StorageBackend {
        self.inner.backend()
    }

    fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.get_cf(cf, key)
    }

    fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.put_cf(cf, key, value)?;
        self.log.record(logged_row(Some(cf), key));
        Ok(())
    }

    fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
        self.inner.delete_cf(cf, key)?;
        self.log.record(logged_row(Some(cf), key));
        Ok(())
    }

    fn scan_cf(
        &self,
        cf: &str,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.inner.scan_cf(cf, prefix, limit)
    }

    fn flush_cf(&self, cf: &str) -> Result<()> {
        self.inner.flush_cf(cf)
    }

    fn ensure_cf(&self, cf: &str) -> Result<()> {
        self.inner.ensure_cf(cf)
    }

    fn write_batch(&self, batch: &WriteBatch) -> Result<()> {
        self.inner.write_batch(batch)?;
        self.log
            .record(batch.ops().iter().filter_map(|op| match op {
                BatchOp::Put { cf, key, .. } | BatchOp::Delete { cf, key } => {
                    logged_row(cf.as_deref(), key)
                }
            }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_with(capacity: usize, rows: &[(Family, Uuid)]) -> GraphChangeLog {
        let log = GraphChangeLog::with_capacity(capacity);
        log.record(rows.iter().copied());
        log
    }

    #[test]
    fn changes_since_a_cursor_are_deduplicated_and_end_at_the_head() {
        let (a, b, e) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let log = log_with(8, &[(Family::Entity, a)]);
        let cursor = log.head();
        log.record([
            (Family::Entity, b),
            (Family::Relationship, e),
            (Family::Entity, b),
            (Family::Relationship, e),
        ]);
        let changes = log.changes_since(cursor).unwrap();
        assert_eq!(changes.entities, vec![b]);
        assert_eq!(changes.relationships, vec![e]);
        assert_eq!(changes.upto, log.head());
        assert!(log.changes_since(log.head()).unwrap().is_empty());
    }

    #[test]
    fn a_cursor_the_ring_has_dropped_or_from_another_log_gets_none() {
        let log = log_with(2, &[]);
        let start = log.head();
        log.record((0..3).map(|_| (Family::Entity, Uuid::new_v4())));
        assert!(log.changes_since(start).is_none(), "one change was dropped");
        let other = GraphChangeLog::with_capacity(2);
        assert!(other.changes_since(log.head()).is_none());
    }

    #[test]
    fn only_entity_and_relationship_rows_are_logged() {
        let id = Uuid::new_v4();
        assert_eq!(
            logged_row(Some(CF_RELATIONSHIPS), id.as_bytes()),
            Some((Family::Relationship, id))
        );
        assert_eq!(logged_row(Some(CF_RELATIONSHIPS), b"mem_edge:a:b"), None);
        assert_eq!(logged_row(Some("entity_edges"), id.as_bytes()), None);
        assert_eq!(logged_row(None, id.as_bytes()), None);
    }
}
