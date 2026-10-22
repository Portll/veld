//! Gap detection algorithms for structural gap analysis.
//!
//! Contains the detection logic for each gap type:
//! open triads, diamond gaps, star gaps, and orbit gaps.

use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::memory::pacing;
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

    // Cluster pairs that share attractors but have few or no cross-links; the
    // most confident `max_gaps_per_type` become gaps.
    let links = cluster_links(&adj, &meaningful_clusters);
    let mut candidates: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
    for (&(i, j), &(shared_attractors, cross_links)) in &links {
        pacing::tick(1);
        if shared_attractors == 0 {
            continue;
        }
        let (size_a, size_b) = (
            meaningful_clusters[i].1.len(),
            meaningful_clusters[j].1.len(),
        );
        let cross_ratio = cross_links as f32 / (size_a * size_b).max(1) as f32;
        if cross_ratio < 0.1 {
            let confidence = (shared_attractors as f32 / 5.0).clamp(0.0, 1.0) * (1.0 - cross_ratio);
            candidates.push((i, j, shared_attractors, cross_links, confidence));
        }
    }
    candidates.sort_by(|a, b| b.4.total_cmp(&a.4).then((a.0, a.1).cmp(&(b.0, b.1))));
    candidates.truncate(config.max_gaps_per_type);

    let mut gaps = Vec::new();
    for (i, j, shared_attractors, cross_links, confidence) in candidates {
        let (id_a, members_a) = &meaningful_clusters[i];
        let (id_b, members_b) = &meaningful_clusters[j];
        let gap_id = format!("orbit:{}:{}", id_a, id_b);

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
            shape: ShapeSignature::orbit(members_a.len(), members_b.len(), shared_attractors),
            entities,
            missing_links,
            confidence,
            embedding_similarity: None,
            impact_score: 0.0,
        });
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
        pacing::tick(1);
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

/// Simple label propagation for community detection.
///
/// Each node starts with its own label, then adopts the most common label
/// among its neighbors. Converges to clusters of densely connected nodes.
pub fn label_propagation(
    adj: &HashMap<String, Vec<String>>,
    max_iterations: usize,
) -> Vec<(usize, Vec<String>)> {
    let nodes: Vec<&String> = adj.keys().collect();
    let mut labels: HashMap<&str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        labels.insert(node.as_str(), i);
    }

    for _ in 0..max_iterations {
        let mut changed = false;
        for node in &nodes {
            pacing::tick(1);
            if let Some(neighbors) = adj.get(node.as_str()) {
                if neighbors.is_empty() {
                    continue;
                }
                // Count neighbor labels
                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for neighbor in neighbors {
                    if let Some(&label) = labels.get(neighbor.as_str()) {
                        *label_counts.entry(label).or_default() += 1;
                    }
                }
                // Adopt most common neighbor label
                if let Some((&best_label, _)) = label_counts.iter().max_by_key(|(_, count)| *count)
                {
                    let current = labels.get(node.as_str()).copied().unwrap_or(0);
                    if best_label != current {
                        labels.insert(node.as_str(), best_label);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Group by label
    let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
    for (node, label) in &labels {
        clusters.entry(*label).or_default().push(node.to_string());
    }

    clusters.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
