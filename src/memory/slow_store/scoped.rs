//! A gap store over the part of the mirror a [`GapScope`] selects, held in
//! memory and read from the mirror in one transaction.

use anyhow::Result;
use rusqlite::types::ValueRef;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use super::queries::{diamond_gaps, open_triads, EdgeGraph, EdgeGraphBuilder};
use super::{open_field, RawDiamondGap, RawOpenTriad, RawStarGap, SlowStore};
use crate::graph_memory::EntityLabel;
use crate::memory::gap_topology::{GapScope, GapStore, ScopeReport};
use crate::memory::pacing;

/// Scopes up to this size read their edges by index; larger ones scan.
const INDEXED_EDGE_READ_MAX: usize = 20_000;
/// Uuids bound per `IN (…)` query.
const IN_BATCH: usize = 500;

/// The entities in a scope and the edges between them.
pub struct ScopedGraph<'a> {
    store: &'a SlowStore,
    uuids: Vec<String>,
    names: Vec<String>,
    salience: Vec<f32>,
    index: HashMap<String, u32>,
    /// From, to, strength, relation.
    edges: Vec<(u32, u32, f32, String)>,
    report: ScopeReport,
}

struct Candidate {
    uuid: String,
    name: String,
    salience: f32,
    mentions: i64,
}

impl SlowStore {
    /// The entities `scope` selects from the mirror, sampled down to its
    /// `max_nodes`, and the edges between them.
    pub fn scoped_graph(&self, scope: &GapScope) -> Result<ScopedGraph<'_>> {
        let _span = tracing::info_span!("slow_store.scoped_graph").entered();
        let conn = self.read()?;
        let tx = conn.unchecked_transaction()?;
        let namespace = scope.namespace.as_ref().map(|n| format!("{n}:"));
        let project_tag = scope.project.as_ref().map(|p| format!("project:{p}"));
        let mut tags = Vec::new();
        let mut candidates = Vec::new();
        let mut stmt =
            tx.prepare_cached("SELECT uuid, name, salience, mention_count, labels FROM entities")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            pacing::tick(1);
            let uuid: String = row.get(0)?;
            let name: String = row.get(1)?;
            let lower = name.to_lowercase();
            if project_tag.as_deref() == Some(lower.as_str()) {
                tags.push(uuid);
                continue;
            }
            if namespace
                .as_ref()
                .is_some_and(|n| !lower.starts_with(n.as_str()))
            {
                continue;
            }
            if !scope.entity_types.is_empty()
                && !self
                    .labels_of(&uuid, row.get_ref(4)?)?
                    .iter()
                    .any(|l| scope.entity_types.contains(l))
            {
                continue;
            }
            candidates.push(Candidate {
                uuid,
                name,
                salience: row.get(2)?,
                mentions: row.get(3)?,
            });
        }
        drop(rows);
        drop(stmt);

        if scope.project.is_some() {
            let mut linked: HashSet<String> = HashSet::new();
            let mut stmt = tx.prepare_cached(
                "SELECT to_entity FROM edges WHERE from_entity = ?1
                 UNION SELECT from_entity FROM edges WHERE to_entity = ?1",
            )?;
            for tag in &tags {
                for uuid in stmt.query_map([tag], |row| row.get::<_, String>(0))? {
                    linked.insert(uuid?);
                }
            }
            candidates.retain(|c| linked.contains(&c.uuid));
        }

        let in_scope = candidates.len();
        if let Some(max) = scope.max_nodes.filter(|&max| candidates.len() > max) {
            candidates.sort_by(|a, b| {
                b.salience
                    .total_cmp(&a.salience)
                    .then(b.mentions.cmp(&a.mentions))
                    .then_with(|| a.uuid.cmp(&b.uuid))
            });
            candidates.truncate(max);
        }

        let mut graph = ScopedGraph {
            store: self,
            uuids: Vec::with_capacity(candidates.len()),
            names: Vec::with_capacity(candidates.len()),
            salience: Vec::with_capacity(candidates.len()),
            index: HashMap::with_capacity(candidates.len()),
            edges: Vec::new(),
            report: scope.report(in_scope, candidates.len()),
        };
        for c in candidates {
            graph.index.insert(c.uuid.clone(), graph.uuids.len() as u32);
            graph.uuids.push(c.uuid);
            graph.names.push(c.name);
            graph.salience.push(c.salience);
        }
        graph.read_edges(&tx)?;
        Ok(graph)
    }

    /// An entity's labels, lowercased; a sealed column is opened first.
    fn labels_of(&self, uuid: &str, column: ValueRef<'_>) -> Result<Vec<String>> {
        let json = match column {
            ValueRef::Text(text) => text.to_vec(),
            ValueRef::Blob(blob) => {
                open_field(self.encryptor.as_ref(), blob.to_vec(), uuid.as_bytes())?
            }
            _ => return Ok(Vec::new()),
        };
        Ok(serde_json::from_slice::<Vec<EntityLabel>>(&json)
            .unwrap_or_default()
            .iter()
            .map(|l| l.as_str().to_lowercase())
            .collect())
    }
}

impl ScopedGraph<'_> {
    pub fn report(&self) -> &ScopeReport {
        &self.report
    }

    fn read_edges(&mut self, tx: &rusqlite::Transaction<'_>) -> Result<()> {
        let mut keep = |from: &str, to: &str, strength: f32, relation: &str| {
            pacing::tick(1);
            if let (Some(&f), Some(&t)) = (self.index.get(from), self.index.get(to)) {
                self.edges.push((f, t, strength, relation.to_string()));
            }
        };
        let columns = "SELECT from_entity, to_entity, strength, relation_type FROM edges";
        if self.uuids.len() <= INDEXED_EDGE_READ_MAX {
            for batch in self.uuids.clone().chunks(IN_BATCH) {
                let sql = format!(
                    "{columns} WHERE from_entity IN ({})",
                    vec!["?"; batch.len()].join(",")
                );
                let mut stmt = tx.prepare(&sql)?;
                let mut rows = stmt.query(rusqlite::params_from_iter(batch))?;
                while let Some(row) = rows.next()? {
                    keep(
                        row.get_ref(0)?.as_str()?,
                        row.get_ref(1)?.as_str()?,
                        row.get(2)?,
                        row.get_ref(3)?.as_str()?,
                    );
                }
            }
        } else {
            let mut stmt = tx.prepare(columns)?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                keep(
                    row.get_ref(0)?.as_str()?,
                    row.get_ref(1)?.as_str()?,
                    row.get(2)?,
                    row.get_ref(3)?.as_str()?,
                );
            }
        }
        Ok(())
    }

    fn edge_graph(&self, min_strength: f32) -> EdgeGraph {
        let mut builder = EdgeGraphBuilder::new(min_strength);
        for i in 0..self.uuids.len() {
            builder.entity(
                self.uuids[i].clone(),
                self.names[i].clone(),
                self.salience[i],
            );
        }
        for (f, t, strength, relation) in &self.edges {
            builder.edge(
                &self.uuids[*f as usize],
                &self.uuids[*t as usize],
                *strength,
                relation,
            );
        }
        builder.finish()
    }
}

impl GapStore for ScopedGraph<'_> {
    fn find_open_triads(&self, min_strength: f32, limit: usize) -> Result<Vec<RawOpenTriad>> {
        Ok(open_triads(&self.edge_graph(min_strength), limit))
    }

    fn find_diamond_gaps(&self, min_strength: f32, limit: usize) -> Result<Vec<RawDiamondGap>> {
        Ok(diamond_gaps(&self.edge_graph(min_strength), limit))
    }

    /// As [`SlowStore::find_star_gaps`]: hubs by distinct out-neighbours, any
    /// strength, most first; spokes strongest first; a hub qualifies when at
    /// most `max_connectivity` of its spoke pairs are linked either way.
    fn find_star_gaps(
        &self,
        min_spokes: usize,
        max_connectivity: f32,
        limit: usize,
    ) -> Result<Vec<RawStarGap>> {
        let mut out: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.uuids.len()];
        for (f, t, strength, _) in &self.edges {
            out[*f as usize].push((*t, *strength));
        }
        let mut hubs: Vec<(usize, usize, f32)> = out
            .iter()
            .enumerate()
            .filter_map(|(hub, spokes)| {
                let distinct: HashSet<u32> = spokes.iter().map(|s| s.0).collect();
                let avg = spokes.iter().map(|s| s.1).sum::<f32>() / spokes.len().max(1) as f32;
                (distinct.len() >= min_spokes).then_some((hub, distinct.len(), avg))
            })
            .collect();
        hubs.sort_by_key(|&(hub, count, _)| (Reverse(count), self.uuids[hub].clone()));
        hubs.truncate(limit.saturating_mul(2));

        let mut results = Vec::new();
        for (hub, _, avg_strength) in hubs {
            pacing::tick(1);
            let mut spokes = out[hub].clone();
            spokes.sort_by(|a, b| b.1.total_cmp(&a.1));
            let members: HashSet<u32> = spokes.iter().map(|s| s.0).collect();
            let mut linked: HashSet<(u32, u32)> = HashSet::new();
            for &s in &members {
                for &(t, _) in &out[s as usize] {
                    if t != s && members.contains(&t) {
                        linked.insert((s.min(t), s.max(t)));
                    }
                }
            }
            let possible = spokes.len() * spokes.len().saturating_sub(1) / 2;
            let connectivity = if possible > 0 {
                linked.len() as f32 / possible as f32
            } else {
                1.0
            };
            if connectivity <= max_connectivity {
                results.push(RawStarGap {
                    hub: self.uuids[hub].clone(),
                    hub_name: self.names[hub].clone(),
                    spokes: spokes
                        .iter()
                        .map(|&(s, _)| {
                            (
                                self.uuids[s as usize].clone(),
                                self.names[s as usize].clone(),
                            )
                        })
                        .collect(),
                    missing_edges: possible.saturating_sub(linked.len()),
                    possible_edges: possible,
                    avg_hub_strength: avg_strength,
                });
            }
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    fn get_adjacency_list(&self, min_strength: f32) -> Result<HashMap<String, Vec<String>>> {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (f, t, strength, _) in &self.edges {
            if *strength >= min_strength {
                let (from, to) = (&self.uuids[*f as usize], &self.uuids[*t as usize]);
                adj.entry(from.clone()).or_default().push(to.clone());
                adj.entry(to.clone()).or_default().push(from.clone());
            }
        }
        Ok(adj)
    }

    fn get_entity_names(&self, uuids: &[&str]) -> Result<HashMap<String, String>> {
        Ok(uuids
            .iter()
            .filter_map(|u| {
                self.index
                    .get(*u)
                    .map(|&i| (u.to_string(), self.names[i as usize].clone()))
            })
            .collect())
    }

    fn load_embeddings(&self, uuids: &[&str]) -> Result<HashMap<String, Vec<f32>>> {
        self.store.load_embeddings(uuids)
    }
}
