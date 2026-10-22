//! The graph reads a mirror sync makes, and the incremental sync.
//!
//! An incremental sync re-reads each row the change log names and writes what
//! the graph holds now. A directed pair keeps its strongest live edge: a changed
//! edge replaces the pair's row when it ranks above it, and when the row's own
//! edge weakened, moved or went, the pair is recomputed from every live edge of
//! the pair. Graph reads and the decisions they feed happen outside any write
//! transaction; only the writes run under the write turn.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use super::mirror_tx::{Step, MIRROR_STEP_ROWS};
use super::{
    count_left_out, edge_upsert_sql, SlowStore, SyncMode, SyncStats, DELTA_CHUNK_ROWS,
    MIRROR_CHUNK_ROWS,
};
use crate::graph_memory::{EntityNode, GraphChanges, RelationshipEdge};

/// A full pass over the graph, a chunk at a time: every entity, then every
/// live relationship.
pub trait MirrorSource {
    fn next_entities(&mut self) -> Result<Option<Vec<EntityNode>>>;
    fn next_relationships(&mut self) -> Result<Option<Vec<RelationshipEdge>>>;
}

/// Point reads of the graph as it is now.
pub trait GraphLookup {
    /// Each entity, `None` once deleted.
    fn entities(&mut self, uuids: &[Uuid]) -> Result<Vec<Option<EntityNode>>>;
    /// Each relationship, `None` once deleted or invalidated.
    fn relationships(&mut self, uuids: &[Uuid]) -> Result<Vec<Option<RelationshipEdge>>>;
    /// Every live relationship with `entity` at either end.
    fn relationships_of(&mut self, entity: &Uuid) -> Result<Vec<RelationshipEdge>>;
}

/// A [`MirrorSource`] over slices already in memory.
pub(super) struct SliceSource<'a> {
    entities: &'a [EntityNode],
    edges: &'a [RelationshipEdge],
}

impl<'a> SliceSource<'a> {
    pub(super) fn new(entities: &'a [EntityNode], edges: &'a [RelationshipEdge]) -> Self {
        Self { entities, edges }
    }
}

fn take_chunk<T: Clone>(rows: &mut &[T]) -> Option<Vec<T>> {
    if rows.is_empty() {
        return None;
    }
    let (chunk, rest) = rows.split_at(rows.len().min(MIRROR_CHUNK_ROWS));
    *rows = rest;
    Some(chunk.to_vec())
}

impl MirrorSource for SliceSource<'_> {
    fn next_entities(&mut self) -> Result<Option<Vec<EntityNode>>> {
        Ok(take_chunk(&mut self.entities))
    }

    fn next_relationships(&mut self) -> Result<Option<Vec<RelationshipEdge>>> {
        Ok(take_chunk(&mut self.edges))
    }
}

type Pair = (Uuid, Uuid);

/// Orders edges as the mirror ranks them: stronger first, then the lower uuid.
fn outranks(strength: f32, uuid: &Uuid, other_strength: f32, other_uuid: &Uuid) -> bool {
    match strength.total_cmp(&other_strength) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => uuid < other_uuid,
    }
}

fn best_edge(edges: impl IntoIterator<Item = RelationshipEdge>) -> Option<RelationshipEdge> {
    edges.into_iter().reduce(|best, edge| {
        if outranks(edge.strength, &edge.uuid, best.strength, &best.uuid) {
            edge
        } else {
            best
        }
    })
}

fn parse_uuid(text: String) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(&text).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

/// The mirrored row of a directed pair: its edge uuid and strength.
fn row_at(conn: &Connection, pair: &Pair) -> Result<Option<(Uuid, f32)>> {
    Ok(conn
        .prepare_cached(
            "SELECT uuid, strength FROM edges WHERE from_entity = ?1 AND to_entity = ?2",
        )?
        .query_row(params![pair.0.to_string(), pair.1.to_string()], |row| {
            Ok((parse_uuid(row.get(0)?)?, row.get::<_, f32>(1)?))
        })
        .optional()?)
}

/// The pair whose row holds edge `uuid`.
fn pair_holding(conn: &Connection, uuid: &Uuid) -> Result<Option<Pair>> {
    Ok(conn
        .prepare_cached("SELECT from_entity, to_entity FROM edges WHERE uuid = ?1")?
        .query_row([uuid.to_string()], |row| {
            Ok((parse_uuid(row.get(0)?)?, parse_uuid(row.get(1)?)?))
        })
        .optional()?)
}

fn delete_row_at(conn: &Connection, pair: &Pair) -> Result<usize> {
    Ok(conn
        .prepare_cached("DELETE FROM edges WHERE from_entity = ?1 AND to_entity = ?2")?
        .execute(params![pair.0.to_string(), pair.1.to_string()])?)
}

fn is_mirrored(conn: &Connection, entity: &Uuid) -> Result<bool> {
    Ok(conn
        .prepare_cached("SELECT 1 FROM entities WHERE uuid = ?1")?
        .query_row([entity.to_string()], |_| Ok(()))
        .optional()?
        .is_some())
}

/// A pair's row to replace with `best`, or to remove when there is none or
/// an endpoint is not mirrored.
struct PairWrite {
    pair: Pair,
    best: Option<RelationshipEdge>,
    mirrored: bool,
}

/// A chunk's live edges by pair.
type ByPair = HashMap<Pair, Vec<RelationshipEdge>>;

/// What one incremental sync knows of the graph.
struct Delta<'a> {
    lookup: &'a mut dyn GraphLookup,
    /// Every relationship the change log named.
    changed: HashSet<Uuid>,
    /// Relationships read so far: the current edge, `None` once gone.
    fresh: HashMap<Uuid, Option<RelationshipEdge>>,
    /// Live relationships of an entity, read at most once per chunk.
    of_entity: HashMap<Uuid, Vec<RelationshipEdge>>,
}

impl Delta<'_> {
    fn fresh(&mut self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        if let Some(edge) = self.fresh.get(uuid) {
            return Ok(edge.clone());
        }
        let edge = self
            .lookup
            .relationships(std::slice::from_ref(uuid))?
            .pop()
            .flatten();
        self.fresh.insert(*uuid, edge.clone());
        Ok(edge)
    }

    fn live_edges_of_pair(&mut self, pair: &Pair) -> Result<Vec<RelationshipEdge>> {
        if !self.of_entity.contains_key(&pair.0) {
            let edges = self.lookup.relationships_of(&pair.0)?;
            self.of_entity.insert(pair.0, edges);
        }
        Ok(self.of_entity[&pair.0]
            .iter()
            .filter(|e| e.from_entity == pair.0 && e.to_entity == pair.1)
            .cloned()
            .collect())
    }

    /// Read the current value of each relationship in a chunk not read yet.
    fn read_chunk(&mut self, uuids: &[Uuid]) -> Result<()> {
        self.of_entity.clear();
        let unread: Vec<Uuid> = uuids
            .iter()
            .filter(|u| !self.fresh.contains_key(u))
            .copied()
            .collect();
        for (uuid, edge) in unread.iter().zip(self.lookup.relationships(&unread)?) {
            self.fresh.insert(*uuid, edge);
        }
        Ok(())
    }

    /// The pairs a chunk of changed relationships touches, where their rows are
    /// and where the edges are now, and the chunk's live edges by pair.
    fn pairs_of(&self, conn: &Connection, uuids: &[Uuid]) -> Result<(Vec<Pair>, ByPair)> {
        let mut by_pair = ByPair::new();
        let mut pairs = Vec::new();
        for uuid in uuids {
            if let Some(old) = pair_holding(conn, uuid)? {
                pairs.push(old);
            }
            if let Some(Some(edge)) = self.fresh.get(uuid) {
                let pair = (edge.from_entity, edge.to_entity);
                by_pair.entry(pair).or_default().push(edge.clone());
                pairs.push(pair);
            }
        }
        Ok((pairs, by_pair))
    }

    /// Decide the row of `pair`: the pair's strongest live edge, or `None` when
    /// the row stands. The candidates are `changed`, the pair's changed edges
    /// read in this chunk, and the row's edge; `recompute`, or a row whose edge
    /// weakened, moved or went, reads every live edge of the pair instead.
    fn plan_pair(
        &mut self,
        conn: &Connection,
        pair: Pair,
        mut recompute: bool,
        changed: &[RelationshipEdge],
    ) -> Result<Option<PairWrite>> {
        let mut candidates: Vec<RelationshipEdge> = changed.to_vec();
        let mut kept_row = None;
        if let Some((row_uuid, row_strength)) = row_at(conn, &pair)? {
            if self.changed.contains(&row_uuid) || self.fresh.contains_key(&row_uuid) {
                match self.fresh(&row_uuid)? {
                    Some(edge)
                        if (edge.from_entity, edge.to_entity) == pair
                            && edge.strength >= row_strength =>
                    {
                        candidates.push(edge)
                    }
                    _ => recompute = true,
                }
            } else {
                kept_row = Some((row_uuid, row_strength));
            }
        }
        if recompute {
            candidates = self.live_edges_of_pair(&pair)?;
            kept_row = None;
        }
        let best = best_edge(candidates);
        if let Some((row_uuid, row_strength)) = kept_row {
            if best
                .as_ref()
                .is_none_or(|b| !outranks(b.strength, &b.uuid, row_strength, &row_uuid))
            {
                return Ok(None);
            }
        }
        let mirrored = best.is_some() && is_mirrored(conn, &pair.0)? && is_mirrored(conn, &pair.1)?;
        Ok(Some(PairWrite {
            pair,
            best,
            mirrored,
        }))
    }
}

impl SlowStore {
    /// Write one pair's row. Returns the pair whose row held the winning edge
    /// before it moved here; that row is removed and the pair is to be redone.
    fn apply_plan(
        &self,
        conn: &Connection,
        plan: PairWrite,
        stats: &mut SyncStats,
    ) -> Result<Option<Pair>> {
        let PairWrite {
            pair,
            best,
            mirrored,
        } = plan;
        let removed = delete_row_at(conn, &pair)?;
        let Some(best) = best else {
            stats.edges_removed += removed;
            return Ok(None);
        };
        if !mirrored {
            stats.edges_removed += removed;
            count_left_out(&best, stats);
            return Ok(None);
        }
        let moved_from = pair_holding(conn, &best.uuid)?;
        if let Some(other) = &moved_from {
            delete_row_at(conn, other)?;
        }
        let mut stmt = conn.prepare_cached(&edge_upsert_sql("edges"))?;
        Self::insert_edge(&mut stmt, &best, self.encryptor.as_ref(), &self.enc)?;
        stats.edges_upserted += 1;
        Ok(moved_from)
    }
}

/// Which of `uuids` the entity mirror holds.
fn mirrored_entities(conn: &Connection, uuids: &[Uuid]) -> Result<HashSet<Uuid>> {
    let mut found = HashSet::new();
    for uuid in uuids {
        if is_mirrored(conn, uuid)? {
            found.insert(*uuid);
        }
    }
    Ok(found)
}

/// Remove an entity and every mirrored edge touching it.
fn remove_entity(conn: &Connection, uuid: &Uuid, stats: &mut SyncStats) -> Result<()> {
    let id = uuid.to_string();
    stats.edges_removed += conn
        .prepare_cached("DELETE FROM edges WHERE from_entity = ?1 OR to_entity = ?1")?
        .execute([&id])?;
    stats.entities_removed += conn
        .prepare_cached("DELETE FROM entities WHERE uuid = ?1")?
        .execute([&id])?;
    Ok(())
}

/// Take up to [`MIRROR_STEP_ROWS`] from the front of `rest`.
pub(super) fn next_step<'r, T>(rest: &mut &'r [T]) -> (&'r [T], Step) {
    let (step, tail) = rest.split_at(rest.len().min(MIRROR_STEP_ROWS));
    *rest = tail;
    (
        step,
        if tail.is_empty() {
            Step::Done
        } else {
            Step::More
        },
    )
}

impl SlowStore {
    /// Apply the rows `changes` names to the live mirror tables, entities first.
    /// An entity new to the mirror brings its live edges along, as edges left
    /// out while it was missing are not in the log.
    pub(super) fn apply_graph_changes(
        &self,
        changes: &GraphChanges,
        lookup: &mut dyn GraphLookup,
        between_chunks: &mut dyn FnMut(),
    ) -> Result<SyncStats> {
        let _span = tracing::info_span!(
            "slow_store.sync_delta",
            entities = changes.entities.len(),
            relationships = changes.relationships.len(),
        )
        .entered();
        let start = std::time::Instant::now();
        let mut stats = SyncStats {
            mode: SyncMode::Incremental,
            ..SyncStats::default()
        };
        let mut guard = self.mirror_conn.lock();
        if guard.is_none() {
            *guard = Some(Self::open_mirror_writer(&self.path)?);
        }
        let conn = guard.as_mut().expect("mirror connection opened above");
        let _paused = self.pause_projection_checkpoints()?;
        let mut delta = Delta {
            lookup,
            changed: changes.relationships.iter().copied().collect(),
            fresh: HashMap::new(),
            of_entity: HashMap::new(),
        };
        let mut edge_uuids = changes.relationships.clone();

        for chunk in changes.entities.chunks(DELTA_CHUNK_ROWS) {
            let current = delta.lookup.entities(chunk)?;
            let mirrored = mirrored_entities(conn, chunk)?;
            let work: Vec<(Uuid, Option<EntityNode>)> = chunk
                .iter()
                .copied()
                .zip(current)
                .filter(|(uuid, entity)| entity.is_some() || mirrored.contains(uuid))
                .collect();
            let mut rest = work.as_slice();
            if !rest.is_empty() {
                self.budgeted_writes(conn, between_chunks, |tx| {
                    let (step, more) = next_step(&mut rest);
                    let present: Vec<EntityNode> =
                        step.iter().filter_map(|(_, e)| e.clone()).collect();
                    Self::sync_entities(
                        tx,
                        "entities",
                        &present,
                        &mut stats,
                        self.encryptor.as_ref(),
                        &self.enc,
                    )?;
                    for (uuid, entity) in step {
                        if entity.is_none() {
                            remove_entity(tx, uuid, &mut stats)?;
                        }
                    }
                    Ok(more)
                })?;
            }
            for (_, entity) in &work {
                let Some(entity) = entity.as_ref().filter(|e| !mirrored.contains(&e.uuid)) else {
                    continue;
                };
                for edge in delta.lookup.relationships_of(&entity.uuid)? {
                    if !delta.fresh.contains_key(&edge.uuid) {
                        if !delta.changed.contains(&edge.uuid) {
                            edge_uuids.push(edge.uuid);
                        }
                        delta.fresh.insert(edge.uuid, Some(edge));
                    }
                }
            }
        }

        for chunk in edge_uuids.chunks(DELTA_CHUNK_ROWS) {
            delta.read_chunk(chunk)?;
            let (pairs, by_pair) = delta.pairs_of(conn, chunk)?;
            let mut settled: HashSet<Pair> = HashSet::new();
            let mut queue: Vec<(Pair, bool)> = pairs.into_iter().map(|p| (p, false)).collect();
            while !queue.is_empty() {
                let mut plans = Vec::new();
                for (pair, recompute) in std::mem::take(&mut queue) {
                    if !recompute && !settled.insert(pair) {
                        continue;
                    }
                    let changed = by_pair.get(&pair).map_or(&[][..], Vec::as_slice);
                    plans.extend(delta.plan_pair(conn, pair, recompute, changed)?);
                }
                if plans.is_empty() {
                    continue;
                }
                let mut pending = plans.into_iter().peekable();
                self.budgeted_writes(conn, between_chunks, |tx| {
                    for plan in pending.by_ref().take(MIRROR_STEP_ROWS) {
                        if let Some(other) = self.apply_plan(tx, plan, &mut stats)? {
                            queue.push((other, true));
                        }
                    }
                    Ok(if pending.peek().is_none() {
                        Step::Done
                    } else {
                        Step::More
                    })
                })?;
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            entities_upserted = stats.entities_upserted,
            entities_removed = stats.entities_removed,
            edges_upserted = stats.edges_upserted,
            edges_removed = stats.edges_removed,
            duration_ms = stats.duration_ms,
            "SlowStore incremental sync complete"
        );
        Ok(stats)
    }
}
