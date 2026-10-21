//! Gap detection algorithms for structural gap analysis.
//!
//! Contains the detection logic for each gap type:
//! open triads, diamond gaps, star gaps, and orbit gaps.

use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::similarity::cosine_similarity;

use super::{
    GapDetectionConfig, GapEntity, GapRole, GapStore, GapTopology, GapType, MissingLink,
    ShapeSignature,
};

/// Detect open triads (U-shapes) and score by embedding similarity.
pub fn detect_open_triads(
    store: &dyn GapStore,
    config: &GapDetectionConfig,
) -> Result<Vec<GapTopology>> {
    let raw_triads = store.find_open_triads(config.min_edge_strength, config.max_gaps_per_type)?;

    // Collect UUIDs for embedding lookup
    let uuids: Vec<&str> = raw_triads
        .iter()
        .flat_map(|t| [t.node_a.as_str(), t.node_c.as_str()])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let embeddings = store.load_embeddings(&uuids)?;

    let mut gaps = Vec::new();
    for triad in &raw_triads {
        // Score: how similar are the endpoints in embedding space?
        let emb_sim = match (embeddings.get(&triad.node_a), embeddings.get(&triad.node_c)) {
            (Some(a), Some(c)) => Some(cosine_similarity(a, c)),
            _ => None,
        };

        // Skip low-similarity gaps (endpoints aren't semantically related)
        if let Some(sim) = emb_sim {
            if sim < config.min_embedding_similarity {
                continue;
            }
        }

        // Confidence: strong edges + high embedding similarity = confident gap
        let edge_confidence = (triad.ab_strength + triad.bc_strength) / 2.0;
        let emb_confidence = emb_sim.unwrap_or(0.5);
        let salience_factor = (triad.a_salience + triad.c_salience) / 2.0;
        let confidence =
            (edge_confidence * 0.4 + emb_confidence * 0.4 + salience_factor * 0.2).clamp(0.0, 1.0);

        let gap_id = format!(
            "triad:{}:{}:{}",
            &triad.node_a[..8.min(triad.node_a.len())],
            &triad.node_b[..8.min(triad.node_b.len())],
            &triad.node_c[..8.min(triad.node_c.len())]
        );

        gaps.push(GapTopology {
            id: gap_id,
            gap_type: GapType::OpenTriad,
            shape: ShapeSignature::open_triad(&triad.node_b_name),
            entities: vec![
                GapEntity {
                    uuid: triad.node_a.clone(),
                    name: triad.node_a_name.clone(),
                    role: GapRole::Endpoint,
                },
                GapEntity {
                    uuid: triad.node_b.clone(),
                    name: triad.node_b_name.clone(),
                    role: GapRole::Bridge,
                },
                GapEntity {
                    uuid: triad.node_c.clone(),
                    name: triad.node_c_name.clone(),
                    role: GapRole::Endpoint,
                },
            ],
            missing_links: vec![MissingLink {
                from_uuid: triad.node_a.clone(),
                from_name: triad.node_a_name.clone(),
                to_uuid: triad.node_c.clone(),
                to_name: triad.node_c_name.clone(),
                evidence: format!(
                    "Both linked to '{}' ({}: {:.2}, {}: {:.2}) but not to each other",
                    triad.node_b_name,
                    triad.ab_relation,
                    triad.ab_strength,
                    triad.bc_relation,
                    triad.bc_strength
                ),
            }],
            confidence,
            embedding_similarity: emb_sim,
            impact_score: 0.0, // computed later
        });
    }

    Ok(gaps)
}

/// Detect diamond gaps and score them.
pub fn detect_diamond_gaps(
    store: &dyn GapStore,
    config: &GapDetectionConfig,
) -> Result<Vec<GapTopology>> {
    let raw_diamonds =
        store.find_diamond_gaps(config.min_edge_strength, config.max_gaps_per_type)?;

    let uuids: Vec<&str> = raw_diamonds
        .iter()
        .flat_map(|d| [d.left.as_str(), d.right.as_str()])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let embeddings = store.load_embeddings(&uuids)?;

    let mut gaps = Vec::new();
    for diamond in &raw_diamonds {
        let emb_sim = match (
            embeddings.get(&diamond.left),
            embeddings.get(&diamond.right),
        ) {
            (Some(l), Some(r)) => Some(cosine_similarity(l, r)),
            _ => None,
        };

        if let Some(sim) = emb_sim {
            if sim < config.min_embedding_similarity {
                continue;
            }
        }

        let confidence = emb_sim.unwrap_or(0.5) * 0.6 + 0.4; // diamonds are inherently strong signals

        let gap_id = format!(
            "diamond:{}:{}:{}:{}",
            &diamond.top[..8.min(diamond.top.len())],
            &diamond.left[..8.min(diamond.left.len())],
            &diamond.right[..8.min(diamond.right.len())],
            &diamond.bottom[..8.min(diamond.bottom.len())]
        );

        gaps.push(GapTopology {
            id: gap_id,
            gap_type: GapType::DiamondGap,
            shape: ShapeSignature::diamond(),
            entities: vec![
                GapEntity {
                    uuid: diamond.top.clone(),
                    name: diamond.top_name.clone(),
                    role: GapRole::Apex,
                },
                GapEntity {
                    uuid: diamond.left.clone(),
                    name: diamond.left_name.clone(),
                    role: GapRole::Lateral,
                },
                GapEntity {
                    uuid: diamond.right.clone(),
                    name: diamond.right_name.clone(),
                    role: GapRole::Lateral,
                },
                GapEntity {
                    uuid: diamond.bottom.clone(),
                    name: diamond.bottom_name.clone(),
                    role: GapRole::Apex,
                },
            ],
            missing_links: vec![MissingLink {
                from_uuid: diamond.left.clone(),
                from_name: diamond.left_name.clone(),
                to_uuid: diamond.right.clone(),
                to_name: diamond.right_name.clone(),
                evidence: format!(
                    "Both reachable from '{}' and converge at '{}' via parallel paths, but not directly connected",
                    diamond.top_name, diamond.bottom_name
                ),
            }],
            confidence,
            embedding_similarity: emb_sim,
            impact_score: 0.0,
        });
    }

    Ok(gaps)
}

/// Detect star gaps (hub with disconnected spokes).
pub fn detect_star_gaps(
    store: &dyn GapStore,
    config: &GapDetectionConfig,
) -> Result<Vec<GapTopology>> {
    let raw_stars = store.find_star_gaps(
        config.star_min_spokes,
        config.star_max_connectivity,
        config.max_gaps_per_type,
    )?;

    let mut gaps = Vec::new();
    for star in &raw_stars {
        let confidence = (1.0
            - (star.possible_edges as f32 - star.missing_edges as f32)
                / star.possible_edges.max(1) as f32)
            * star.avg_hub_strength;

        let gap_id = format!(
            "star:{}:spokes={}",
            &star.hub[..8.min(star.hub.len())],
            star.spokes.len()
        );

        let mut entities = vec![GapEntity {
            uuid: star.hub.clone(),
            name: star.hub_name.clone(),
            role: GapRole::Hub,
        }];
        entities.extend(star.spokes.iter().map(|(uuid, name)| GapEntity {
            uuid: uuid.clone(),
            name: name.clone(),
            role: GapRole::Spoke,
        }));

        // Generate missing links between spokes (sample — don't enumerate all N^2)
        let max_links = 10;
        let mut missing_links = Vec::new();
        'outer: for i in 0..star.spokes.len() {
            for j in (i + 1)..star.spokes.len() {
                missing_links.push(MissingLink {
                    from_uuid: star.spokes[i].0.clone(),
                    from_name: star.spokes[i].1.clone(),
                    to_uuid: star.spokes[j].0.clone(),
                    to_name: star.spokes[j].1.clone(),
                    evidence: format!(
                        "Both connected to hub '{}' but not to each other",
                        star.hub_name
                    ),
                });
                if missing_links.len() >= max_links {
                    break 'outer;
                }
            }
        }

        gaps.push(GapTopology {
            id: gap_id,
            gap_type: GapType::StarGap,
            shape: ShapeSignature::star(
                &star.hub_name,
                star.spokes.len(),
                star.missing_edges,
                star.possible_edges,
            ),
            entities,
            missing_links,
            confidence,
            embedding_similarity: None,
            impact_score: 0.0,
        });
    }

    Ok(gaps)
}

/// Detect orbit gaps using label propagation clustering.
///
/// Two clusters that share "attractor" entities (common neighbors outside both clusters)
/// but have no direct cross-links represent knowledge silos that should be connected.
pub fn detect_orbit_gaps(
    store: &dyn GapStore,
    config: &GapDetectionConfig,
) -> Result<Vec<GapTopology>> {
    let adj = store.get_adjacency_list(config.min_edge_strength)?;
    if adj.is_empty() {
        return Ok(Vec::new());
    }

    // Label propagation clustering
    let clusters = label_propagation(&adj, 20);

    // Filter to meaningful clusters
    let meaningful_clusters: Vec<(usize, Vec<String>)> = clusters
        .into_iter()
        .filter(|(_, members)| members.len() >= config.orbit_min_cluster_size)
        .collect();

    if meaningful_clusters.len() < 2 {
        return Ok(Vec::new());
    }

    // Find cluster pairs that share attractors but have no direct cross-links
    let links = cluster_links(&adj, &meaningful_clusters);
    let mut gaps = Vec::new();
    for i in 0..meaningful_clusters.len() {
        for j in (i + 1)..meaningful_clusters.len() {
            let (_, members_a) = &meaningful_clusters[i];
            let (_, members_b) = &meaningful_clusters[j];
            let Some(&(shared_attractors, cross_links)) = links.get(&(i, j)) else {
                continue;
            };
            if shared_attractors == 0 {
                continue;
            }

            // If few or no cross-links relative to shared attractors, it's an orbit gap
            let cross_ratio =
                cross_links as f32 / (members_a.len() * members_b.len()).max(1) as f32;
            if cross_ratio < 0.1 {
                let confidence =
                    (shared_attractors as f32 / 5.0).clamp(0.0, 1.0) * (1.0 - cross_ratio);

                // Named by each cluster's first member, as triads are by their
                // nodes: a label is a position among all entities and moves
                // whenever an entity sorting before it comes or goes.
                let (first_a, first_b) = (&members_a[0], &members_b[0]);
                let gap_id = format!(
                    "orbit:{}:{}",
                    &first_a[..8.min(first_a.len())],
                    &first_b[..8.min(first_b.len())]
                );

                // Get names for entities
                let all_uuids: Vec<&str> = members_a
                    .iter()
                    .chain(members_b.iter())
                    .map(|s| s.as_str())
                    .take(20) // limit for name lookup
                    .collect();
                let names = store.get_entity_names(&all_uuids).unwrap_or_default();

                let entities: Vec<GapEntity> = members_a
                    .iter()
                    .take(5)
                    .chain(members_b.iter().take(5))
                    .map(|uuid| GapEntity {
                        uuid: uuid.clone(),
                        name: names
                            .get(uuid.as_str())
                            .cloned()
                            .unwrap_or_else(|| uuid[..8.min(uuid.len())].to_string()),
                        role: GapRole::ClusterMember,
                    })
                    .collect();

                // Sample missing links
                let mut missing_links = Vec::new();
                for a in members_a.iter().take(3) {
                    for b in members_b.iter().take(3) {
                        let a_name = names
                            .get(a.as_str())
                            .cloned()
                            .unwrap_or_else(|| a[..8.min(a.len())].to_string());
                        let b_name = names
                            .get(b.as_str())
                            .cloned()
                            .unwrap_or_else(|| b[..8.min(b.len())].to_string());
                        missing_links.push(MissingLink {
                            from_uuid: a.clone(),
                            from_name: a_name,
                            to_uuid: b.clone(),
                            to_name: b_name,
                            evidence: format!(
                                "Clusters share {} attractor(s) but have only {} cross-link(s)",
                                shared_attractors, cross_links
                            ),
                        });
                    }
                }

                gaps.push(GapTopology {
                    id: gap_id,
                    gap_type: GapType::OrbitGap,
                    shape: ShapeSignature::orbit(
                        members_a.len(),
                        members_b.len(),
                        shared_attractors,
                    ),
                    entities,
                    missing_links,
                    confidence,
                    embedding_similarity: None,
                    impact_score: 0.0,
                });
            }
        }
    }

    Ok(gaps)
}

/// For each pair of clusters `(i, j)`, `i < j` by position: how many entities
/// outside both have a neighbor in each (shared attractors), and how many
/// adjacency entries run from cluster `i` into cluster `j` (cross-links). One
/// pass over the adjacency; testing every pair against every entity took a pass
/// per pair.
fn cluster_links(
    adj: &HashMap<String, Vec<String>>,
    clusters: &[(usize, Vec<String>)],
) -> HashMap<(usize, usize), (usize, usize)> {
    let cluster_of: HashMap<&str, usize> = clusters
        .iter()
        .enumerate()
        .flat_map(|(i, (_, members))| members.iter().map(move |m| (m.as_str(), i)))
        .collect();
    let mut links: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for (entity, neighbors) in adj {
        let own = cluster_of.get(entity.as_str()).copied();
        let mut near: Vec<usize> = neighbors
            .iter()
            .filter_map(|n| cluster_of.get(n.as_str()).copied())
            .collect();
        if let Some(p) = own {
            for &q in &near {
                if p < q {
                    links.entry((p, q)).or_default().1 += 1;
                }
            }
        }
        near.sort_unstable();
        near.dedup();
        near.retain(|&c| Some(c) != own);
        for (k, &p) in near.iter().enumerate() {
            for &q in &near[k + 1..] {
                links.entry((p, q)).or_default().0 += 1;
            }
        }
    }
    links
}

/// Label propagation for community detection.
///
/// Each node starts with its own label, then adopts the most common label
/// among its neighbors. Converges to clusters of densely connected nodes.
/// Returns `(label, members)` per cluster, ordered by label, members sorted.
///
/// The result depends only on the graph, not on the order `adj` was built or
/// iterates in: ids are interned to `u32` in sorted order, each sweep visits
/// nodes in that order updating labels in place, and a tie between neighbour
/// labels goes to the smallest. A node also keeps its label while that label
/// ties for most common. That rule is what makes the sweeps settle: every change
/// then strictly raises the number of edges whose ends share a label, so there
/// can be at most one change per edge, where adopting any tied label could
/// trade the same labels back and forth until `max_iterations` cut it off.
pub fn label_propagation(
    adj: &HashMap<String, Vec<String>>,
    max_iterations: usize,
) -> Vec<(usize, Vec<String>)> {
    let mut ids: Vec<&str> = adj.keys().map(String::as_str).collect();
    ids.sort_unstable();
    let node_count =
        u32::try_from(ids.len()).expect("a knowledge graph mirror holds fewer than 2^32 entities");
    let index: HashMap<&str, u32> = ids.iter().copied().zip(0..node_count).collect();

    // Neighbours as one flat array, node `v`'s at `offsets[v]..offsets[v + 1]`.
    // A neighbour that is not itself a key has no label and is skipped.
    let mut offsets: Vec<usize> = Vec::with_capacity(ids.len() + 1);
    let mut neighbours: Vec<u32> = Vec::new();
    offsets.push(0);
    for id in &ids {
        neighbours.extend(
            adj[*id]
                .iter()
                .filter_map(|n| index.get(n.as_str()).copied()),
        );
        offsets.push(neighbours.len());
    }

    let mut labels: Vec<u32> = (0..node_count).collect();
    // Votes per label for the node being swept, and which labels got any, so
    // only those are read and reset.
    let mut votes: Vec<u32> = vec![0; ids.len()];
    let mut voted: Vec<u32> = Vec::new();
    for _ in 0..max_iterations {
        let mut changed = false;
        for node in 0..ids.len() {
            for &neighbour in &neighbours[offsets[node]..offsets[node + 1]] {
                let label = labels[neighbour as usize];
                if votes[label as usize] == 0 {
                    voted.push(label);
                }
                votes[label as usize] += 1;
            }
            let most = voted
                .iter()
                .map(|&label| votes[label as usize])
                .max()
                .unwrap_or(0);
            if votes[labels[node] as usize] < most {
                if let Some(best) = voted
                    .iter()
                    .copied()
                    .filter(|&label| votes[label as usize] == most)
                    .min()
                {
                    labels[node] = best;
                    changed = true;
                }
            }
            for &label in &voted {
                votes[label as usize] = 0;
            }
            voted.clear();
        }
        if !changed {
            break;
        }
    }

    // Nodes are visited in sorted order, so each member list comes out sorted.
    let mut clusters: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for (node, id) in ids.iter().enumerate() {
        clusters
            .entry(labels[node])
            .or_default()
            .push((*id).to_string());
    }
    clusters
        .into_iter()
        .map(|(label, members)| (label as usize, members))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::slow_store::{RawDiamondGap, RawOpenTriad, RawStarGap};

    /// The adjacency `get_adjacency_list` builds from `edges`, in their order.
    /// Every `HashMap::new()` draws its own random hash seed, so two maps built
    /// from one edge list iterate in different orders as well.
    fn adjacency<'a>(
        edges: impl Iterator<Item = &'a (String, String)>,
    ) -> HashMap<String, Vec<String>> {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (a, b) in edges {
            adj.entry(a.clone()).or_default().push(b.clone());
            adj.entry(b.clone()).or_default().push(a.clone());
        }
        adj
    }

    fn clique(names: &[&str]) -> Vec<(String, String)> {
        let mut edges = Vec::new();
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                edges.push((a.to_string(), b.to_string()));
            }
        }
        edges
    }

    /// A store that only has an adjacency list, for orbit detection alone.
    struct AdjacencyOnly(HashMap<String, Vec<String>>);

    impl GapStore for AdjacencyOnly {
        fn find_open_triads(&self, _: f32, _: usize) -> Result<Vec<RawOpenTriad>> {
            Ok(Vec::new())
        }
        fn find_diamond_gaps(&self, _: f32, _: usize) -> Result<Vec<RawDiamondGap>> {
            Ok(Vec::new())
        }
        fn find_star_gaps(&self, _: usize, _: f32, _: usize) -> Result<Vec<RawStarGap>> {
            Ok(Vec::new())
        }
        fn get_adjacency_list(&self, _: f32) -> Result<HashMap<String, Vec<String>>> {
            Ok(self.0.clone())
        }
        fn get_entity_names(&self, _: &[&str]) -> Result<HashMap<String, String>> {
            Ok(HashMap::new())
        }
        fn load_embeddings(&self, _: &[&str]) -> Result<HashMap<String, Vec<f32>>> {
            Ok(HashMap::new())
        }
    }

    #[test]
    fn label_propagation_ignores_insertion_order() {
        for seed in 1..=10u64 {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut next = move |n: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % n as u64) as usize
            };
            let names: Vec<String> = (0..80).map(|i| format!("n{i:02}")).collect();
            let edges: Vec<(String, String)> = (0..200)
                .map(|_| (names[next(80)].clone(), names[next(80)].clone()))
                .collect();

            let forward = label_propagation(&adjacency(edges.iter()), 20);
            let backward = label_propagation(&adjacency(edges.iter().rev()), 20);

            assert_eq!(forward, backward, "seed {seed}");
            assert!(forward.windows(2).all(|w| w[0].0 < w[1].0), "seed {seed}");
            for (_, members) in &forward {
                assert!(members.windows(2).all(|w| w[0] < w[1]), "seed {seed}");
            }
        }
    }

    #[test]
    fn orbit_gaps_ignore_insertion_order() {
        // Two 4-cliques P and Q with no edge between them, and a triangle H whose
        // members each touch one node of P and one of Q: H is a third cluster,
        // and all three of its members are attractors shared by P and Q.
        let mut edges = clique(&["p0", "p1", "p2", "p3"]);
        edges.extend(clique(&["q0", "q1", "q2", "q3"]));
        edges.extend(clique(&["h0", "h1", "h2"]));
        for (h, p, q) in [("h0", "p3", "q3"), ("h1", "p2", "q2"), ("h2", "p1", "q1")] {
            edges.push((h.to_string(), p.to_string()));
            edges.push((h.to_string(), q.to_string()));
        }
        let config = GapDetectionConfig::default();

        let forward = detect_orbit_gaps(&AdjacencyOnly(adjacency(edges.iter())), &config).unwrap();
        let backward =
            detect_orbit_gaps(&AdjacencyOnly(adjacency(edges.iter().rev())), &config).unwrap();

        assert_eq!(
            serde_json::to_string(&forward).unwrap(),
            serde_json::to_string(&backward).unwrap()
        );
        assert_eq!(forward.len(), 1);
        let gap = &forward[0];
        assert_eq!(gap.id, "orbit:p0:q0");
        assert_eq!(gap.shape.canonical, "orbit:a=4,b=4,shared_attractors=3");
        let uuids: Vec<&str> = gap.entities.iter().map(|e| e.uuid.as_str()).collect();
        assert_eq!(uuids, ["p0", "p1", "p2", "p3", "q0", "q1", "q2", "q3"]);
    }

    #[test]
    fn cluster_links_counts_what_the_per_pair_scan_counted() {
        for seed in 1..=20u64 {
            let mut state = seed | 1;
            let mut next = move |n: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % n as u64) as usize
            };
            let names: Vec<String> = (0..60).map(|i| format!("n{i}")).collect();
            let mut adj: HashMap<String, Vec<String>> = HashMap::new();
            for _ in 0..180 {
                let (a, b) = (&names[next(60)], &names[next(60)]);
                adj.entry(a.clone()).or_default().push(b.clone());
                adj.entry(b.clone()).or_default().push(a.clone());
            }
            let mut clusters: Vec<(usize, Vec<String>)> =
                (0..5).map(|c| (c * 7, Vec::new())).collect();
            for name in adj.keys() {
                // Some entities stay outside every cluster, as small clusters are dropped.
                if let Some(c) = Some(next(7)).filter(|&c| c < 5) {
                    clusters[c].1.push(name.clone());
                }
            }

            let links = cluster_links(&adj, &clusters);
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let set_a: HashSet<&str> = clusters[i].1.iter().map(String::as_str).collect();
                    let set_b: HashSet<&str> = clusters[j].1.iter().map(String::as_str).collect();
                    let shared = adj
                        .iter()
                        .filter(|(e, n)| {
                            !set_a.contains(e.as_str())
                                && !set_b.contains(e.as_str())
                                && n.iter().any(|x| set_a.contains(x.as_str()))
                                && n.iter().any(|x| set_b.contains(x.as_str()))
                        })
                        .count();
                    let cross: usize = clusters[i]
                        .1
                        .iter()
                        .filter_map(|m| adj.get(m))
                        .map(|n| n.iter().filter(|x| set_b.contains(x.as_str())).count())
                        .sum();
                    let got = links.get(&(i, j)).copied().unwrap_or((0, 0));
                    assert_eq!(got, (shared, cross), "seed {seed} pair {i},{j}");
                }
            }
        }
    }
}
