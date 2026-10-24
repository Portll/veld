//! Keeps a tenant's SQLite graph mirror in step with its graph.
//!
//! Reads take the graph's read lock one key-prefix chunk at a time, so a
//! writer waits for at most one chunk. A full copy notes the change log's head
//! before it starts and afterwards applies every row written since, so the
//! mirror reflects every graph write logged before the copy ended; writes
//! after that are applied by the next sync. When the log has dropped some of
//! those writes, the copy is marked best-effort.

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use super::pacing;
use super::slow_store::{GraphLookup, MirrorSource, SlowStore, SyncMode, SyncStats};
use crate::graph_memory::{EntityNode, GraphMemory, RelationshipEdge};

/// An incremental sync handles at most this fraction of the graph's rows;
/// more changes than that rebuild the mirror instead.
const INCREMENTAL_MAX_SHARE: f64 = 0.25;

/// Longest wait for the graph's read lock in a point read; below SQLite's
/// five-second busy timeout.
const LOOKUP_LOCK_WAIT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy)]
pub struct MirrorSyncOptions {
    /// A full rebuild runs at most once per `ttl`. Incremental syncs have no TTL.
    pub ttl: Duration,
    /// Rows read per hold of the graph's read lock, roughly.
    pub chunk_rows: usize,
}

/// Bring `store`'s mirror up to `graph`: the rows changed since the mirror's
/// cursor, or a full chunked copy when it has no usable cursor or too much
/// changed. One sync per store at a time.
pub fn sync_mirror(
    graph: &RwLock<GraphMemory>,
    store: &SlowStore,
    options: &MirrorSyncOptions,
) -> Result<SyncStats> {
    let start = std::time::Instant::now();
    let (log, rows) = {
        let guard = graph.read();
        let stats = guard.get_stats()?;
        (
            Arc::clone(guard.change_log()),
            stats.entity_count + stats.relationship_count,
        )
    };
    let mut feed = GraphFeed::new(graph, options.chunk_rows);
    let mut between_chunks = pacing::pause;

    let changes = store.mirror_cursor().and_then(|c| log.changes_since(c));
    if let Some(changes) = changes {
        if changes.is_empty() {
            return Ok(SyncStats {
                mode: SyncMode::Unchanged,
                ..SyncStats::default()
            });
        }
        if (changes.len() as f64) <= (rows.max(1) as f64) * INCREMENTAL_MAX_SHARE {
            let Some(mut token) = store.begin_incremental() else {
                return Ok(skipped());
            };
            // Another sync may have moved the cursor before the slot was ours.
            let Some(changes) = store.mirror_cursor().and_then(|c| log.changes_since(c)) else {
                return Ok(skipped());
            };
            let mut stats = token.apply_changes(&changes, &mut feed, &mut between_chunks)?;
            token.set_cursor(Some(changes.upto));
            stats.duration_ms = start.elapsed().as_millis() as u64;
            return Ok(stats);
        }
    }

    let Some(mut token) = store.begin_sync(options.ttl) else {
        tracing::debug!(
            "graph mirror sync skipped: a sync is in flight or the last rebuild is within the TTL"
        );
        return Ok(skipped());
    };
    let copy_start = log.head();
    let mut stats = token.rebuild_from(&mut feed, &mut between_chunks)?;
    match log.changes_since(copy_start) {
        Some(changes) => {
            let repair = token.apply_changes(&changes, &mut feed, &mut between_chunks)?;
            stats.entities_upserted += repair.entities_upserted;
            stats.edges_upserted += repair.edges_upserted;
            stats.entities_removed += repair.entities_removed;
            stats.edges_removed += repair.edges_removed;
            stats.changed_during_copy = Some(changes.len());
            token.set_cursor(Some(changes.upto));
        }
        None => {
            tracing::warn!(
                "graph mirror copy is best-effort: the change log dropped writes made during it"
            );
            stats.changed_during_copy = None;
            token.set_cursor(None);
        }
    }
    stats.mode = SyncMode::Full;
    stats.duration_ms = start.elapsed().as_millis() as u64;
    Ok(stats)
}

fn skipped() -> SyncStats {
    SyncStats {
        mode: SyncMode::Skipped,
        skipped: true,
        ..SyncStats::default()
    }
}

/// Key prefixes that split a family of uuid-keyed rows into chunks.
struct Prefixes {
    width: usize,
    next: u32,
}

impl Prefixes {
    /// One-byte prefixes, or two-byte ones when a 256th of `rows` exceeds `chunk_rows`.
    fn for_rows(rows: usize, chunk_rows: usize) -> Self {
        Self {
            width: if rows / 256 > chunk_rows { 2 } else { 1 },
            next: 0,
        }
    }

    fn next(&mut self) -> Option<Vec<u8>> {
        if self.next >= 1 << (8 * self.width) {
            return None;
        }
        let prefix = self.next.to_be_bytes()[4 - self.width..].to_vec();
        self.next += 1;
        Some(prefix)
    }
}

/// The graph read for the mirror: chunked full passes and point reads, each
/// under its own hold of the read lock.
pub struct GraphFeed<'a> {
    graph: &'a RwLock<GraphMemory>,
    chunk_rows: usize,
    entities: Option<Prefixes>,
    relationships: Option<Prefixes>,
}

impl<'a> GraphFeed<'a> {
    fn read_for_lookup(&self) -> Result<parking_lot::RwLockReadGuard<'a, GraphMemory>> {
        self.graph.try_read_for(LOOKUP_LOCK_WAIT).ok_or_else(|| {
            anyhow::anyhow!("graph lock busy for {LOOKUP_LOCK_WAIT:?}; mirror sync abandoned")
        })
    }

    pub fn new(graph: &'a RwLock<GraphMemory>, chunk_rows: usize) -> Self {
        Self {
            graph,
            chunk_rows: chunk_rows.max(1),
            entities: None,
            relationships: None,
        }
    }

    /// Read whole prefixes until a chunk's worth of rows is in hand.
    fn next_chunk<T>(
        graph: &RwLock<GraphMemory>,
        chunk_rows: usize,
        prefixes: &mut Prefixes,
        read: impl Fn(&GraphMemory, &[u8]) -> Result<Vec<T>>,
    ) -> Result<Option<Vec<T>>> {
        let mut rows = Vec::new();
        let mut any = false;
        while rows.len() < chunk_rows {
            let Some(prefix) = prefixes.next() else {
                break;
            };
            any = true;
            let chunk = read(&graph.read(), &prefix)?;
            pacing::tick(chunk.len().max(1));
            rows.extend(chunk);
        }
        Ok(any.then_some(rows))
    }
}

impl MirrorSource for GraphFeed<'_> {
    fn next_entities(&mut self) -> Result<Option<Vec<EntityNode>>> {
        let (graph, chunk_rows) = (self.graph, self.chunk_rows);
        let prefixes = self.entities.get_or_insert_with(|| {
            let rows = graph.read().get_stats().map_or(0, |s| s.entity_count);
            Prefixes::for_rows(rows, chunk_rows)
        });
        Self::next_chunk(graph, chunk_rows, prefixes, |g, p| {
            g.entities_with_key_prefix(p)
        })
    }

    fn next_relationships(&mut self) -> Result<Option<Vec<RelationshipEdge>>> {
        let (graph, chunk_rows) = (self.graph, self.chunk_rows);
        let prefixes = self.relationships.get_or_insert_with(|| {
            let rows = graph.read().get_stats().map_or(0, |s| s.relationship_count);
            Prefixes::for_rows(rows, chunk_rows)
        });
        Self::next_chunk(graph, chunk_rows, prefixes, |g, p| {
            g.relationships_with_key_prefix(p)
        })
    }
}

/// Point reads may run inside a mirror transaction. They give up on the graph
/// lock before SQLite's busy timeout, so a writer that waits on the mirror while
/// a graph writer waits on the read lock cannot hold both up; the sync fails,
/// and the next one re-applies the same changes.
impl GraphLookup for GraphFeed<'_> {
    fn entities(&mut self, uuids: &[Uuid]) -> Result<Vec<Option<EntityNode>>> {
        let mut found = Vec::with_capacity(uuids.len());
        for chunk in uuids.chunks(self.chunk_rows) {
            let graph = self.read_for_lookup()?;
            for uuid in chunk {
                found.push(graph.get_entity(uuid)?);
            }
        }
        Ok(found)
    }

    fn relationships(&mut self, uuids: &[Uuid]) -> Result<Vec<Option<RelationshipEdge>>> {
        let mut found = Vec::with_capacity(uuids.len());
        for chunk in uuids.chunks(self.chunk_rows) {
            let graph = self.read_for_lookup()?;
            for uuid in chunk {
                found.push(graph.live_relationship(uuid)?);
            }
        }
        Ok(found)
    }

    fn relationships_of(&mut self, entity: &Uuid) -> Result<Vec<RelationshipEdge>> {
        self.read_for_lookup()?.live_relationships_of(entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_cover_every_key_once() {
        let mut one = Prefixes::for_rows(1000, 4096);
        let mut n = 0;
        while let Some(p) = one.next() {
            assert_eq!(p, vec![n as u8]);
            n += 1;
        }
        assert_eq!(n, 256);
        let mut two = Prefixes::for_rows(256 * 4097, 4096);
        assert_eq!(two.next(), Some(vec![0, 0]));
        assert_eq!(two.next(), Some(vec![0, 1]));
        let rest = std::iter::from_fn(|| two.next()).count();
        assert_eq!(rest + 2, 65_536);
    }
}
