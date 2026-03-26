//! Gap detection queries and raw structural types.
//!
//! SQL-based structural analysis of the knowledge graph: open triads,
//! star gaps, diamond gaps, adjacency lists, and entity metadata queries.

use anyhow::Result;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::SlowStore;

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
    /// Find all open triads (U-shapes) in the graph.
    ///
    /// An open triad is A→B→C where no edge exists between A and C (in either direction).
    /// These represent missing inferences: two things are both related to a third,
    /// but their mutual relationship hasn't been established.
    ///
    /// Returns raw triads sorted by combined edge strength (strongest gaps first).
    pub fn find_open_triads(&self, min_strength: f32, limit: usize) -> Result<Vec<RawOpenTriad>> {
        let conn = self.conn.lock();
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
        let conn = self.conn.lock();

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
                Ok((row.get(0)?, row.get(1)?, row.get::<_, usize>(2)?, row.get(3)?))
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

        let mut inter_stmt = conn.prepare_cached(
            "SELECT COUNT(*) FROM edges
             WHERE from_entity = ?1 AND to_entity = ?2",
        )?;

        let mut results = Vec::new();
        for (hub_uuid, hub_name, _spoke_count, avg_strength) in &hubs {
            let spokes: Vec<(String, String)> = spoke_stmt
                .query_map(params![hub_uuid], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            if spokes.len() < min_spokes {
                continue;
            }

            // Count existing inter-spoke edges
            let mut existing_inter = 0usize;
            let possible = spokes.len() * (spokes.len() - 1) / 2;
            for i in 0..spokes.len() {
                for j in (i + 1)..spokes.len() {
                    let count_fwd: usize =
                        inter_stmt.query_row(params![&spokes[i].0, &spokes[j].0], |row| {
                            row.get(0)
                        })?;
                    let count_rev: usize =
                        inter_stmt.query_row(params![&spokes[j].0, &spokes[i].0], |row| {
                            row.get(0)
                        })?;
                    if count_fwd > 0 || count_rev > 0 {
                        existing_inter += 1;
                    }
                }
            }

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
    pub fn find_diamond_gaps(&self, min_strength: f32, limit: usize) -> Result<Vec<RawDiamondGap>> {
        let conn = self.conn.lock();
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
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT from_entity, to_entity FROM edges WHERE strength >= ?1",
        )?;

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

    /// Get entity names for a set of UUIDs
    pub fn get_entity_names(&self, uuids: &[&str]) -> Result<HashMap<String, String>> {
        let conn = self.conn.lock();
        let mut result = HashMap::new();
        let mut stmt = conn.prepare_cached("SELECT uuid, name FROM entities WHERE uuid = ?1")?;

        for uuid in uuids {
            if let Ok((id, name)) = stmt.query_row(params![uuid], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                result.insert(id, name);
            }
        }

        Ok(result)
    }

    /// Get entity count in the slow store
    pub fn entity_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize =
            conn.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Get edge count in the slow store
    pub fn edge_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        Ok(count)
    }
}
