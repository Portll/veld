//! Gap detection queries and raw structural types.
//!
//! SQL-based structural analysis of the knowledge graph: open triads,
//! star gaps, diamond gaps, adjacency lists, and entity metadata queries.

use anyhow::Result;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use super::SlowStore;

/// The entities and edges the structural searches walk in memory: names and
/// salience by index, every directed pair at any strength, and for each node its
/// edges at or above the minimum strength, strongest first.
struct EdgeGraph {
    uuids: Vec<String>,
    names: Vec<String>,
    salience: Vec<f32>,
    relations: Vec<String>,
    /// (to, strength, relation) at or above the minimum, strongest first.
    out: Vec<Vec<(u32, f32, u32)>>,
    /// The strength of each pair at or above the minimum.
    strong: HashMap<(u32, u32), f32>,
    /// Every directed pair, at any strength.
    pairs: HashSet<(u32, u32)>,
}

impl EdgeGraph {
    fn linked(&self, a: u32, b: u32) -> bool {
        self.pairs.contains(&(a, b)) || self.pairs.contains(&(b, a))
    }

    /// Each node's strongest outgoing strength, and the strongest of all.
    fn best_out(&self) -> (Vec<f64>, f64) {
        let best: Vec<f64> = self
            .out
            .iter()
            .map(|l| l.first().map_or(f64::NEG_INFINITY, |e| f64::from(e.1)))
            .collect();
        let global = best.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (best, global)
    }
}

struct Scored<T>(f64, T);

impl<T> PartialEq for Scored<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}
impl<T> Eq for Scored<T> {}
impl<T> PartialOrd for Scored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Scored<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The `k` highest-scoring items. A score that only ties the lowest kept one is
/// not taken, as `ORDER BY … LIMIT` leaves the choice among ties open.
struct TopK<T> {
    k: usize,
    heap: BinaryHeap<Reverse<Scored<T>>>,
}

impl<T> TopK<T> {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::new(),
        }
    }

    /// Whether an item scoring `score` would be kept.
    fn admits(&self, score: f64) -> bool {
        self.k > 0
            && (self.heap.len() < self.k
                || self.heap.peek().is_some_and(|Reverse(low)| score > low.0))
    }

    fn push(&mut self, score: f64, item: T) {
        if !self.admits(score) {
            return;
        }
        if self.heap.len() == self.k {
            self.heap.pop();
        }
        self.heap.push(Reverse(Scored(score, item)));
    }

    /// The kept items, highest score first.
    fn into_sorted(self) -> Vec<T> {
        let mut kept: Vec<Scored<T>> = self.heap.into_iter().map(|Reverse(s)| s).collect();
        kept.sort_by(|a, b| b.cmp(a));
        kept.into_iter().map(|s| s.1).collect()
    }
}

/// An open triad (U-shape) found in the graph.
///
/// A → B → C where A and C have no direct edge.
/// The topology of the gap tells us what kind of connection is missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawOpenTriad {
    /// The first endpoint (no direct edge to node_c)
    pub node_a: String,
    pub node_a_name: String,
    /// The bridge node connecting both endpoints
    pub node_b: String,
    pub node_b_name: String,
    /// The second endpoint (no direct edge to node_a)
    pub node_c: String,
    pub node_c_name: String,
    /// Strength of the A→B edge
    pub ab_strength: f32,
    /// Strength of the B→C edge
    pub bc_strength: f32,
    /// Relationship type A→B
    pub ab_relation: String,
    /// Relationship type B→C
    pub bc_relation: String,
    /// Salience of node A
    pub a_salience: f32,
    /// Salience of node C
    pub c_salience: f32,
}

/// A star gap: multiple entities connected to a hub but not to each other.
/// The hub's spokes are isolated — they form a wheel without a rim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawStarGap {
    /// The hub entity UUID
    pub hub: String,
    pub hub_name: String,
    /// Spoke entities connected to the hub
    pub spokes: Vec<(String, String)>, // (uuid, name)
    /// Number of missing inter-spoke edges
    pub missing_edges: usize,
    /// Total possible inter-spoke edges
    pub possible_edges: usize,
    /// Average edge strength from hub to spokes
    pub avg_hub_strength: f32,
}

/// A diamond gap: A→B, A→C, B→D, C→D, but no B↔C link.
/// Two parallel paths converge without their intermediaries connecting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDiamondGap {
    pub top: String,
    pub top_name: String,
    pub left: String,
    pub left_name: String,
    pub right: String,
    pub right_name: String,
    pub bottom: String,
    pub bottom_name: String,
    /// Whether left→right edge is missing
    pub missing_left_right: bool,
}

/// Cluster information for orbit gap detection
#[derive(Debug, Clone)]
pub struct EntityCluster {
    pub cluster_id: usize,
    pub entities: Vec<(String, String)>, // (uuid, name)
    pub internal_edge_count: usize,
    pub avg_internal_strength: f32,
}

impl SlowStore {
    /// The entities and edges of the mirror, for the in-memory structural searches.
    fn edge_graph(&self, min_strength: f32) -> Result<EdgeGraph> {
        let conn = self.read()?;
        let mut index: HashMap<String, u32> = HashMap::new();
        let (mut uuids, mut names, mut salience) = (Vec::new(), Vec::new(), Vec::new());
        let mut stmt = conn.prepare_cached("SELECT uuid, name, salience FROM entities")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f32>(2)?,
            ))
        })?;
        for row in rows {
            let (uuid, name, s) = row?;
            index.insert(uuid.clone(), uuids.len() as u32);
            uuids.push(uuid);
            names.push(name);
            salience.push(s);
        }

        let mut relation_index: HashMap<String, u32> = HashMap::new();
        let mut relations = Vec::new();
        let mut out = vec![Vec::new(); uuids.len()];
        let mut strong = HashMap::new();
        let mut pairs = HashSet::new();
        let mut stmt = conn
            .prepare_cached("SELECT from_entity, to_entity, strength, relation_type FROM edges")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f32>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        for row in rows {
            let (from, to, strength, relation) = row?;
            let (Some(&f), Some(&t)) = (index.get(&from), index.get(&to)) else {
                continue;
            };
            pairs.insert((f, t));
            if strength >= min_strength {
                let r = match relation_index.get(&relation) {
                    Some(&r) => r,
                    None => {
                        let r = relations.len() as u32;
                        relation_index.insert(relation.clone(), r);
                        relations.push(relation);
                        r
                    }
                };
                out[f as usize].push((t, strength, r));
                strong.insert((f, t), strength);
            }
        }
        for list in &mut out {
            list.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
        Ok(EdgeGraph {
            uuids,
            names,
            salience,
            relations,
            out,
            strong,
            pairs,
        })
    }

    /// Find all open triads (U-shapes) in the graph.
    ///
    /// An open triad is A→B→C where no edge exists between A and C (in either direction).
    /// These represent missing inferences: two things are both related to a third,
    /// but their mutual relationship hasn't been established.
    ///
    /// Returns raw triads sorted by combined edge strength (strongest gaps first).
    /// Edges are taken strongest first and a path stops being extended once its
    /// best possible sum cannot displace the weakest triad kept.
    pub fn find_open_triads(&self, min_strength: f32, limit: usize) -> Result<Vec<RawOpenTriad>> {
        let g = self.edge_graph(min_strength)?;
        let (best_out, global_best) = g.best_out();
        let mut first: Vec<(u32, u32, f32, u32)> = g
            .out
            .iter()
            .enumerate()
            .flat_map(|(a, l)| l.iter().map(move |&(b, s, r)| (a as u32, b, s, r)))
            .collect();
        first.sort_by(|x, y| y.2.total_cmp(&x.2));

        let mut top = TopK::new(limit);
        for &(a, b, s1, r1) in &first {
            let s1f = f64::from(s1);
            if !top.admits(s1f + global_best) {
                break;
            }
            if !top.admits(s1f + best_out[b as usize]) {
                continue;
            }
            for &(c, s2, r2) in &g.out[b as usize] {
                let sum = s1f + f64::from(s2);
                if !top.admits(sum) {
                    break;
                }
                if c == a || g.linked(a, c) {
                    continue;
                }
                top.push(sum, (a, b, c, s1, s2, r1, r2));
            }
        }

        Ok(top
            .into_sorted()
            .into_iter()
            .map(|(a, b, c, s1, s2, r1, r2)| {
                let (a, b, c) = (a as usize, b as usize, c as usize);
                RawOpenTriad {
                    node_a: g.uuids[a].clone(),
                    node_a_name: g.names[a].clone(),
                    a_salience: g.salience[a],
                    node_b: g.uuids[b].clone(),
                    node_b_name: g.names[b].clone(),
                    node_c: g.uuids[c].clone(),
                    node_c_name: g.names[c].clone(),
                    c_salience: g.salience[c],
                    ab_strength: s1,
                    bc_strength: s2,
                    ab_relation: g.relations[r1 as usize].clone(),
                    bc_relation: g.relations[r2 as usize].clone(),
                }
            })
            .collect())
    }

    /// The SQL form of [`find_open_triads`](Self::find_open_triads), kept to check it against.
    #[cfg(test)]
    pub(crate) fn find_open_triads_sql(
        &self,
        min_strength: f32,
        limit: usize,
    ) -> Result<Vec<RawOpenTriad>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare_cached(
            "SELECT
                e1.from_entity, ea.name, ea.salience,
                e1.to_entity, eb.name,
                e2.to_entity, ec.name, ec.salience,
                e1.strength, e2.strength,
                e1.relation_type, e2.relation_type
             FROM edges e1
             JOIN edges e2 ON e1.to_entity = e2.from_entity
             JOIN entities ea ON e1.from_entity = ea.uuid
             JOIN entities eb ON e1.to_entity = eb.uuid
             JOIN entities ec ON e2.to_entity = ec.uuid
             WHERE e1.from_entity != e2.to_entity
               AND e1.strength >= ?1
               AND e2.strength >= ?1
               AND NOT EXISTS (
                   SELECT 1 FROM edges e3
                   WHERE (e3.from_entity = e1.from_entity AND e3.to_entity = e2.to_entity)
                      OR (e3.from_entity = e2.to_entity AND e3.to_entity = e1.from_entity)
               )
             ORDER BY (e1.strength + e2.strength) DESC
             LIMIT ?2",
        )?;

        let triads = stmt
            .query_map(params![min_strength, limit], |row| {
                Ok(RawOpenTriad {
                    node_a: row.get(0)?,
                    node_a_name: row.get(1)?,
                    a_salience: row.get(2)?,
                    node_b: row.get(3)?,
                    node_b_name: row.get(4)?,
                    node_c: row.get(5)?,
                    node_c_name: row.get(6)?,
                    c_salience: row.get(7)?,
                    ab_strength: row.get(8)?,
                    bc_strength: row.get(9)?,
                    ab_relation: row.get(10)?,
                    bc_relation: row.get(11)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(triads)
    }

    /// Find star gaps: hub entities whose spokes have few inter-connections.
    ///
    /// A star gap indicates a hub concept whose related entities exist in isolation
    /// from each other — the wheel has no rim. These represent missing community
    /// structure: things grouped by a common concept should often relate to each other.
    pub fn find_star_gaps(
        &self,
        min_spokes: usize,
        max_connectivity: f32,
        limit: usize,
    ) -> Result<Vec<RawStarGap>> {
        let conn = self.read()?;

        // Step 1: Find hub entities with enough connections
        let mut hub_stmt = conn.prepare_cached(
            "SELECT e_out.from_entity, ent.name,
                    COUNT(DISTINCT e_out.to_entity) as spoke_count,
                    AVG(e_out.strength) as avg_strength
             FROM edges e_out
             JOIN entities ent ON e_out.from_entity = ent.uuid
             GROUP BY e_out.from_entity
             HAVING spoke_count >= ?1
             ORDER BY spoke_count DESC
             LIMIT ?2",
        )?;

        let hubs: Vec<(String, String, usize, f32)> = hub_stmt
            .query_map(params![min_spokes, limit * 2], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get::<_, usize>(2)?,
                    row.get(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Step 2: For each hub, get spokes and check inter-spoke connectivity
        let mut spoke_stmt = conn.prepare_cached(
            "SELECT e.to_entity, ent.name
             FROM edges e
             JOIN entities ent ON e.to_entity = ent.uuid
             WHERE e.from_entity = ?1
             ORDER BY e.strength DESC",
        )?;

        let mut results = Vec::new();
        for (hub_uuid, hub_name, _spoke_count, avg_strength) in &hubs {
            let spokes: Vec<(String, String)> = spoke_stmt
                .query_map(params![hub_uuid], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            if spokes.len() < min_spokes {
                continue;
            }

            let possible = spokes.len() * (spokes.len() - 1) / 2;

            // Count distinct connected spoke pairs (edge present in either
            // direction) in ONE query instead of O(spokes^2) round-trips. The
            // canonicalized lo/hi pair de-dupes A->B and B->A. Placeholders are
            // generated for the two IN clauses; the spoke UUIDs are *bound*,
            // never interpolated, so this stays injection-safe.
            let existing_inter: usize = if spokes.len() >= 2 {
                let spoke_ids: Vec<&str> = spokes.iter().map(|(id, _)| id.as_str()).collect();
                let placeholders = vec!["?"; spoke_ids.len()].join(",");
                let inter_sql = format!(
                    "SELECT COUNT(*) FROM (
                        SELECT DISTINCT
                            CASE WHEN from_entity < to_entity THEN from_entity ELSE to_entity END AS lo,
                            CASE WHEN from_entity < to_entity THEN to_entity ELSE from_entity END AS hi
                        FROM edges
                        WHERE from_entity IN ({ph}) AND to_entity IN ({ph})
                          AND from_entity <> to_entity
                    )",
                    ph = placeholders
                );
                let mut bind: Vec<&str> = Vec::with_capacity(spoke_ids.len() * 2);
                bind.extend_from_slice(&spoke_ids);
                bind.extend_from_slice(&spoke_ids);
                conn.query_row(&inter_sql, rusqlite::params_from_iter(bind), |row| {
                    row.get(0)
                })?
            } else {
                0
            };

            let connectivity = if possible > 0 {
                existing_inter as f32 / possible as f32
            } else {
                1.0
            };

            if connectivity <= max_connectivity {
                results.push(RawStarGap {
                    hub: hub_uuid.clone(),
                    hub_name: hub_name.clone(),
                    missing_edges: possible - existing_inter,
                    possible_edges: possible,
                    avg_hub_strength: *avg_strength,
                    spokes,
                });
            }

            if results.len() >= limit {
                break;
            }
        }

        Ok(results)
    }

    /// Find diamond gaps: A→B, A→C, B→D, C→D, but no B↔C.
    ///
    /// Two parallel paths from A to D through different intermediaries,
    /// where the intermediaries aren't connected. This often indicates
    /// two approaches to the same problem that haven't been reconciled.
    ///
    /// Returns diamonds sorted by the sum of their four edge strengths. Each pair
    /// of A's edges, and each B→D after it, is skipped once its best possible sum
    /// cannot displace the weakest diamond kept.
    pub fn find_diamond_gaps(&self, min_strength: f32, limit: usize) -> Result<Vec<RawDiamondGap>> {
        let g = self.edge_graph(min_strength)?;
        let (best_out, global_best) = g.best_out();
        let mut top = TopK::new(limit);
        for (a, list) in g.out.iter().enumerate() {
            let a = a as u32;
            for (i, &(x, sx, _)) in list.iter().enumerate() {
                let sx = f64::from(sx);
                if !top.admits(sx + sx + global_best + global_best) {
                    break;
                }
                for &(y, sy, _) in &list[i + 1..] {
                    let sy = f64::from(sy);
                    if !top.admits(sx + sy + best_out[x as usize] + global_best) {
                        break;
                    }
                    if !top.admits(sx + sy + best_out[x as usize] + best_out[y as usize]) {
                        continue;
                    }
                    // B is the intermediary whose uuid sorts first, as `e1.to_entity < e2.to_entity`.
                    let ((b, sab), (c, sac)) = if g.uuids[x as usize] < g.uuids[y as usize] {
                        ((x, sx), (y, sy))
                    } else {
                        ((y, sy), (x, sx))
                    };
                    if g.linked(b, c) {
                        continue;
                    }
                    for &(d, sbd, _) in &g.out[b as usize] {
                        let sbd = f64::from(sbd);
                        if !top.admits(sab + sac + sbd + best_out[c as usize]) {
                            break;
                        }
                        if d == a {
                            continue;
                        }
                        let Some(&scd) = g.strong.get(&(c, d)) else {
                            continue;
                        };
                        top.push(sab + sac + sbd + f64::from(scd), (a, b, c, d));
                    }
                }
            }
        }

        Ok(top
            .into_sorted()
            .into_iter()
            .map(|(a, b, c, d)| {
                let (a, b, c, d) = (a as usize, b as usize, c as usize, d as usize);
                RawDiamondGap {
                    top: g.uuids[a].clone(),
                    top_name: g.names[a].clone(),
                    left: g.uuids[b].clone(),
                    left_name: g.names[b].clone(),
                    right: g.uuids[c].clone(),
                    right_name: g.names[c].clone(),
                    bottom: g.uuids[d].clone(),
                    bottom_name: g.names[d].clone(),
                    missing_left_right: true,
                }
            })
            .collect())
    }

    /// The SQL form of [`find_diamond_gaps`](Self::find_diamond_gaps), kept to check it against.
    #[cfg(test)]
    pub(crate) fn find_diamond_gaps_sql(
        &self,
        min_strength: f32,
        limit: usize,
    ) -> Result<Vec<RawDiamondGap>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare_cached(
            "SELECT
                e1.from_entity, ea.name,   -- A (top)
                e1.to_entity, eb.name,     -- B (left)
                e2.to_entity, ec.name,     -- C (right)
                e3.to_entity, ed.name      -- D (bottom)
             FROM edges e1
             JOIN edges e2 ON e1.from_entity = e2.from_entity  -- A→B, A→C (same source)
             JOIN edges e3 ON e1.to_entity = e3.from_entity    -- B→D
             JOIN edges e4 ON e2.to_entity = e4.from_entity    -- C→D
             JOIN entities ea ON e1.from_entity = ea.uuid
             JOIN entities eb ON e1.to_entity = eb.uuid
             JOIN entities ec ON e2.to_entity = ec.uuid
             JOIN entities ed ON e3.to_entity = ed.uuid
             WHERE e3.to_entity = e4.to_entity                 -- B→D and C→D (same target)
               AND e1.to_entity < e2.to_entity                 -- B < C to avoid duplicates
               AND e1.to_entity != e2.to_entity                -- B ≠ C
               AND e1.from_entity != e3.to_entity              -- A ≠ D
               AND e1.strength >= ?1
               AND e2.strength >= ?1
               AND e3.strength >= ?1
               AND e4.strength >= ?1
               AND NOT EXISTS (
                   SELECT 1 FROM edges ex
                   WHERE (ex.from_entity = e1.to_entity AND ex.to_entity = e2.to_entity)
                      OR (ex.from_entity = e2.to_entity AND ex.to_entity = e1.to_entity)
               )
             ORDER BY (e1.strength + e2.strength + e3.strength + e4.strength) DESC
             LIMIT ?2",
        )?;

        let diamonds = stmt
            .query_map(params![min_strength, limit], |row| {
                Ok(RawDiamondGap {
                    top: row.get(0)?,
                    top_name: row.get(1)?,
                    left: row.get(2)?,
                    left_name: row.get(3)?,
                    right: row.get(4)?,
                    right_name: row.get(5)?,
                    bottom: row.get(6)?,
                    bottom_name: row.get(7)?,
                    missing_left_right: true,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(diamonds)
    }

    /// Get all entity neighbors (both directions) for orbit detection
    pub fn get_adjacency_list(&self, min_strength: f32) -> Result<HashMap<String, Vec<String>>> {
        let conn = self.read()?;
        let mut stmt =
            conn.prepare_cached("SELECT from_entity, to_entity FROM edges WHERE strength >= ?1")?;

        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        let rows = stmt.query_map(params![min_strength], |row| {
            let from: String = row.get(0)?;
            let to: String = row.get(1)?;
            Ok((from, to))
        })?;

        for row in rows {
            let (from, to) = row?;
            adj.entry(from.clone()).or_default().push(to.clone());
            adj.entry(to).or_default().push(from);
        }

        Ok(adj)
    }

    /// Get entity names for a set of UUIDs.
    ///
    /// One `IN (...)` query rather than a point lookup per UUID. Placeholders
    /// are generated for the bind count; UUIDs are bound, never interpolated.
    pub fn get_entity_names(&self, uuids: &[&str]) -> Result<HashMap<String, String>> {
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let conn = self.read()?;
        let placeholders = vec!["?"; uuids.len()].join(",");
        let sql = format!("SELECT uuid, name FROM entities WHERE uuid IN ({placeholders})");
        let mut stmt = conn.prepare(&sql)?;

        let mut result = HashMap::new();
        let rows = stmt.query_map(rusqlite::params_from_iter(uuids.iter().copied()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (id, name) = row?;
            result.insert(id, name);
        }

        Ok(result)
    }

    /// Get entity count in the slow store
    pub fn entity_count(&self) -> Result<usize> {
        let conn = self.read()?;
        let count: usize = conn.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Get edge count in the slow store
    pub fn edge_count(&self) -> Result<usize> {
        let conn = self.read()?;
        let count: usize = conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        Ok(count)
    }
}
