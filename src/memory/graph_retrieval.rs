//! Graph-Aware Retrieval using Spreading Activation
//!
//! Based on:
//! - Anderson & Pirolli (1984): "Spread of Activation"
//! - Collins & Loftus (1975): "Spreading-activation theory of semantic processing"
//! - Xiong et al. (2017): "Explicit Semantic Ranking via Knowledge Graph Embedding"
//! - GraphRAG Survey (arXiv 2408.08921): Hybrid KG-Vector improves 13.1%
//! - spreadr R package (Siew, 2019): Importance-weighted decay
//! - ACT-R cognitive architecture: Multi-source activation with intersection boost
//!
//! Implements spreading activation algorithm for memory retrieval:
//! 1. Extract entities from query (using linguistic analysis)
//! 2. Activate entities in knowledge graph (salience-weighted, ACT-R inspired)
//! 3. Spread activation through graph relationships (importance-weighted decay)
//!    - PIPE-7: Bidirectional spreading for 2+ entities (meet-in-middle)
//! 4. Retrieve episodic memories connected to activated entities
//! 5. Score using hybrid method (density-dependent graph + semantic + linguistic)
//!
//! SHO-26 Enhancements:
//! - Density-dependent hybrid weights: Graph trust scales with learned associations
//! - Importance-weighted decay: Important memories decay slower, preserve signal
//!
//! PIPE-7 Enhancements:
//! - Bidirectional spreading: When 2+ focal entities, split and spread from both ends
//! - Intersection boost: Entities reached from both directions get 1.5× activation
//! - Complexity reduction: O(b^d) → O(2 × b^(d/2)) for multi-entity queries

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

use crate::constants::{
    BIDIRECTIONAL_DENSITY_DENSE, BIDIRECTIONAL_DENSITY_SPARSE, BIDIRECTIONAL_HOPS_DENSE,
    BIDIRECTIONAL_HOPS_MEDIUM, BIDIRECTIONAL_HOPS_SPARSE, BIDIRECTIONAL_INTERSECTION_BOOST,
    BIDIRECTIONAL_INTERSECTION_MIN, BIDIRECTIONAL_MIN_ENTITIES, DENSITY_GRAPH_WEIGHT_MAX,
    DENSITY_GRAPH_WEIGHT_MIN, DENSITY_LINGUISTIC_WEIGHT, DENSITY_THRESHOLD_MAX,
    DENSITY_THRESHOLD_MIN, EDGE_TIER_TRUST_L1, EDGE_TIER_TRUST_L2, EDGE_TIER_TRUST_L3,
    EDGE_TIER_TRUST_LTP, HYBRID_GRAPH_WEIGHT, HYBRID_LINGUISTIC_WEIGHT, HYBRID_SEMANTIC_WEIGHT,
    IMPORTANCE_DECAY_MAX, IMPORTANCE_DECAY_MIN, MEMORY_TIER_GRAPH_MULT_ARCHIVE,
    MEMORY_TIER_GRAPH_MULT_LONGTERM, MEMORY_TIER_GRAPH_MULT_SESSION,
    MEMORY_TIER_GRAPH_MULT_WORKING, ONTOLOGICAL_DENSITY_THRESHOLD, ONTOLOGICAL_ENTITY_PENALTY,
    ONTOLOGICAL_MIN_CONFIDENCE, ONTOLOGICAL_RELATION_PENALTY, QUERY_TYPE_ATTRIBUTE_GRAPH,
    QUERY_TYPE_ATTRIBUTE_LINGUISTIC, QUERY_TYPE_ATTRIBUTE_SEMANTIC, QUERY_TYPE_EXPLORATORY_GRAPH,
    QUERY_TYPE_EXPLORATORY_LINGUISTIC, QUERY_TYPE_EXPLORATORY_SEMANTIC, QUERY_TYPE_TEMPORAL_GRAPH,
    QUERY_TYPE_TEMPORAL_LINGUISTIC, QUERY_TYPE_TEMPORAL_SEMANTIC, SALIENCE_BOOST_FACTOR,
    SPREADING_ACTIVATION_THRESHOLD, SPREADING_DEGREE_NORMALIZATION,
    SPREADING_EARLY_TERMINATION_CANDIDATES, SPREADING_EARLY_TERMINATION_RATIO, SPREADING_MAX_HOPS,
    SPREADING_MIN_CANDIDATES, SPREADING_MIN_HOPS, SPREADING_NORMALIZATION_FACTOR,
    SPREADING_RELAXED_THRESHOLD,
};
use crate::embeddings::Embedder;
use crate::graph_memory::{EdgeTier, EntityLabel, EpisodicNode, GraphMemory, RelationType};
use crate::memory::query_parser::{classify_query, infer_ontological_intent, OntologicalIntent};
use crate::memory::types::MemoryTier;
// Note: compute_relevance removed - using unified density-weighted scoring directly
use crate::memory::query_parser::{analyze_query, QueryAnalysis, QueryType};
use crate::memory::types::{Memory, Query, RetrievalStats, SharedMemory};
use crate::similarity::cosine_similarity;

/// Memory with activation score
#[derive(Debug, Clone)]
pub struct ActivatedMemory {
    pub memory: SharedMemory,
    #[allow(dead_code)] // Useful for debugging score breakdown
    pub activation_score: f32,
    #[allow(dead_code)] // Useful for debugging score breakdown
    pub semantic_score: f32,
    #[allow(dead_code)] // Useful for debugging score breakdown
    pub linguistic_score: f32,
    pub final_score: f32,
}

/// Calculate density-dependent graph weight (SHO-26, corrected)
///
/// SPARSE = TRUST GRAPH (edges that survived Hebbian decay are high-quality)
/// DENSE = TRUST VECTOR (too many noisy edges, use semantic similarity)
///
/// Graph weight scales INVERSELY with density:
/// - density < 0.5: weight = 0.5 (sparse graph, trust high-signal associations)
/// - density > 2.0: weight = 0.1 (dense graph, noisy, trust vector search)
/// - in between: linear interpolation
///
/// Reference: GraphRAG Survey (arXiv 2408.08921) + Hebbian learning insight
/// How much of a memory's GRAPH evidence counts, given its memory tier (SHO-D2).
///
/// Working memories are dense and noisy, so their associations are discounted; `LongTerm`
/// is the undiscounted reference. The value scales the graph activation ONLY — see
/// [`fuse_hybrid_score`] for why that distinction is load-bearing.
///
/// The per-tier values are veld's own constants, unchanged by this port. Upstream
/// (shodh-memory 653dc00) also held `Session` at the `Working` value and retired `Archive`
/// after measuring both on its LoCoMo gate; veld has not yet produced a gate run of its own
/// (docs/HANDOFF-2026-09-07.md §A3/A4), so those two ranking decisions are NOT imported here.
/// Make them on a measured veld run, in this function.
pub fn memory_tier_graph_trust(tier: MemoryTier) -> f32 {
    match tier {
        MemoryTier::Working => MEMORY_TIER_GRAPH_MULT_WORKING,
        MemoryTier::Session => MEMORY_TIER_GRAPH_MULT_SESSION,
        MemoryTier::LongTerm => MEMORY_TIER_GRAPH_MULT_LONGTERM,
        MemoryTier::Archive => MEMORY_TIER_GRAPH_MULT_ARCHIVE,
    }
}

/// Blend the three retrieval legs into one comparable score.
///
/// `semantic_weight`/`graph_weight`/`linguistic_weight` come from [`calculate_density_weights`]
/// and are a property of the QUERY, identical for every candidate. `tier_graph_trust` is a
/// property of the CANDIDATE and discounts its graph evidence.
///
/// # Why the trust factor multiplies the signal, not the weight
///
/// This used to fold the tier factor into the weight vector and renormalise per candidate:
///
/// ```text
///   S' = w_sem + t·w_graph + w_ling
///   score = sem·(w_sem/S') + graph·(t·w_graph/S') + ling·(w_ling/S')
/// ```
///
/// `S'` depends on `t`, so every tier produced a DIFFERENT weight mix, and those scores were
/// ranked against each other. Discounting a memory's graph leg silently INFLATED its semantic
/// and linguistic legs: whenever semantic evidence dominated (the common case) the memory the
/// system trusted LESS scored higher for that reason alone, inverting the ordering the tier
/// ladder exists to express. Ported from shodh-memory 653dc00, where the change was a strict
/// improvement on its gate (multi_hop ndcg 0.2516 → 0.2652, no category regressed).
pub fn fuse_hybrid_score(
    semantic_score: f32,
    graph_activation: f32,
    linguistic_score: f32,
    semantic_weight: f32,
    graph_weight: f32,
    linguistic_weight: f32,
    tier_graph_trust: f32,
) -> f32 {
    let weight_sum = semantic_weight + graph_weight + linguistic_weight;
    if weight_sum <= 0.0 {
        return 0.0;
    }
    semantic_score * (semantic_weight / weight_sum)
        + graph_activation * tier_graph_trust * (graph_weight / weight_sum)
        + linguistic_score * (linguistic_weight / weight_sum)
}

pub fn calculate_density_weights(graph_density: f32) -> (f32, f32, f32) {
    // INVERTED logic: sparse graphs have higher graph weight
    let graph_weight = if graph_density <= DENSITY_THRESHOLD_MIN {
        DENSITY_GRAPH_WEIGHT_MAX // Sparse = trust graph more
    } else if graph_density >= DENSITY_THRESHOLD_MAX {
        DENSITY_GRAPH_WEIGHT_MIN // Dense = trust vector more
    } else {
        // Linear interpolation: higher density = lower graph weight
        let ratio = (graph_density - DENSITY_THRESHOLD_MIN)
            / (DENSITY_THRESHOLD_MAX - DENSITY_THRESHOLD_MIN);
        DENSITY_GRAPH_WEIGHT_MAX - ratio * (DENSITY_GRAPH_WEIGHT_MAX - DENSITY_GRAPH_WEIGHT_MIN)
    };

    let linguistic_weight = DENSITY_LINGUISTIC_WEIGHT;
    let semantic_weight = 1.0 - graph_weight - linguistic_weight;

    (semantic_weight, graph_weight, linguistic_weight)
}

/// Apply query-type modulation to density-based RRF weights (Plan 001 §2.2)
///
/// Rationale: different query types benefit from different retrieval channels.
/// - Temporal queries: BM25/linguistic weight↑ (keywords like "week"/"yesterday" are strong signals)
/// - Attribute queries: graph weight↑ (entity traversal finds attribute-bearing memories)
/// - Exploratory queries: keep density-based defaults (semantic similarity works best)
///
/// The shift is additive (±0.10) and renormalized to sum to 1.0, preserving the
/// density-based baseline while tilting toward the query-relevant channel.
pub fn apply_query_type_weights(
    base: (f32, f32, f32),
    query_type: &crate::memory::query_parser::QueryType,
) -> (f32, f32, f32) {
    let (mut semantic_w, mut graph_w, mut linguistic_w) = base;

    match query_type {
        crate::memory::query_parser::QueryType::Temporal => {
            // Temporal queries: boost linguistic (BM25 catches temporal keywords),
            // slight semantic boost (temporal embeddings are weak but non-zero)
            linguistic_w += 0.10;
            semantic_w += 0.02;
            graph_w -= 0.12;
        }
        crate::memory::query_parser::QueryType::Attribute(_) => {
            // Attribute queries: boost graph (entity traversal finds attribute memories)
            graph_w += 0.10;
            semantic_w -= 0.10;
        }
        crate::memory::query_parser::QueryType::Exploratory => {
            // Exploratory: density-based defaults are already semantic-favoring, no shift
        }
    }

    // Normalize to sum to 1.0, then enforce a minimum floor of 5% per channel.
    // If any channel is below floor after normalization, steal from the largest.
    const FLOOR: f32 = 0.05;
    semantic_w = semantic_w.max(0.0);
    graph_w = graph_w.max(0.0);
    linguistic_w = linguistic_w.max(0.0);

    let total = semantic_w + graph_w + linguistic_w;
    if total > 0.0 {
        semantic_w /= total;
        graph_w /= total;
        linguistic_w /= total;
    }

    // Enforce floor: redistribute deficit from the largest channel
    let mut weights = [semantic_w, graph_w, linguistic_w];
    for _ in 0..3 {
        let mut deficit = 0.0_f32;
        for w in weights.iter_mut() {
            if *w < FLOOR {
                deficit += FLOOR - *w;
                *w = FLOOR;
            }
        }
        if deficit > 0.0 {
            // Find largest channel and subtract deficit
            let max_idx = weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            weights[max_idx] = (weights[max_idx] - deficit).max(FLOOR);
        }
    }

    (weights[0], weights[1], weights[2])
}

/// Calculate importance-weighted decay for spreading activation (SHO-26)
///
/// Important memories (decisions, learnings) decay slower to preserve signal.
/// Weak memories (observations, context) decay faster for exploration.
///
/// Formula: decay = DECAY_MIN + (1.0 - importance) * (DECAY_MAX - DECAY_MIN)
/// - importance = 1.0: decay = 0.1 (high-value, preserve)
/// - importance = 0.0: decay = 0.4 (low-value, explore)
///
/// Reference: spreadr R package (Siew, 2019)
pub fn calculate_importance_weighted_decay(importance: f32) -> f32 {
    let clamped_importance = importance.clamp(0.0, 1.0);
    IMPORTANCE_DECAY_MIN
        + (1.0 - clamped_importance) * (IMPORTANCE_DECAY_MAX - IMPORTANCE_DECAY_MIN)
}

// =============================================================================
// PIPE-7: BIDIRECTIONAL SPREADING ACTIVATION
// =============================================================================
//
// Based on:
// - Collins & Loftus (1975): Spreading activation in semantic memory
// - ACT-R cognitive architecture: Multi-source activation with intersection boost
// - Meet-in-middle search: O(b^d) → O(2 × b^(d/2)) complexity reduction
//
// Key insight: When query has multiple focal entities (e.g., "Rust" and "database"),
// spreading from both ends and boosting intersection entities finds the "bridge"
// concepts that connect the query terms - these are often the most relevant.
//
// Density-adaptive hop count:
// - Dense (fresh) graphs: fewer hops to avoid noise from L1 edges
// - Sparse (mature) graphs: more hops to find connections through curated L2/L3 edges

/// Calculate density-adaptive hop count for bidirectional spreading
///
/// Graph lifecycle: Dense → Sparse as decay prunes weak edges over time.
/// - Fresh system (dense): many noisy L1 edges → fewer hops (2)
/// - Mature system (sparse): curated L2/L3 edges → more hops (4)
///
/// Returns: number of hops per direction
pub fn calculate_adaptive_hops(graph_density: Option<f32>) -> usize {
    match graph_density {
        Some(density) if density > BIDIRECTIONAL_DENSITY_DENSE => {
            // Dense graph: limit exploration to avoid noise
            BIDIRECTIONAL_HOPS_DENSE
        }
        Some(density) if density < BIDIRECTIONAL_DENSITY_SPARSE => {
            // Sparse graph: explore deeper through quality edges
            BIDIRECTIONAL_HOPS_SPARSE
        }
        Some(_) => {
            // Medium density: balanced exploration
            BIDIRECTIONAL_HOPS_MEDIUM
        }
        None => {
            // No density info: use medium as safe default
            BIDIRECTIONAL_HOPS_MEDIUM
        }
    }
}

/// Spread activation from a set of seed entities for a fixed number of hops
///
/// This is a single-direction spread used by bidirectional algorithm.
/// Returns the activation map, traversed edge UUIDs, and path-credit contributions.
fn spread_single_direction(
    seeds: &[(Uuid, f32)],
    graph: &GraphMemory,
    max_hops: usize,
    threshold: f32,
    ontological_intent: Option<&OntologicalIntent>,
    entity_label_cache: &mut HashMap<Uuid, Option<Vec<EntityLabel>>>,
) -> Result<(HashMap<Uuid, f32>, Vec<Uuid>, Vec<(Uuid, f32)>)> {
    let mut activation_map: HashMap<Uuid, f32> = seeds.iter().cloned().collect();
    let mut traversed_edges: Vec<Uuid> = Vec::new();
    // Path-credit: raw (edge_uuid, spread_amount) contributions across all hops.
    let mut credit_edges: Vec<(Uuid, f32)> = Vec::new();

    for hop in 1..=max_hops {
        let current_activated: Vec<(Uuid, f32)> =
            activation_map.iter().map(|(id, act)| (*id, *act)).collect();

        for (entity_uuid, source_activation) in current_activated {
            if source_activation < threshold {
                continue;
            }

            const MAX_EDGES_PER_SPREAD: usize = 100;
            let edges =
                graph.get_entity_relationships_limited(&entity_uuid, Some(MAX_EDGES_PER_SPREAD))?;

            // Degree normalization: prevent hub nodes from flooding the network.
            // Divides outgoing activation by sqrt(1 + degree), matching the fan effect
            // in ACT-R spreading activation (Anderson & Reder 1999).
            let degree_norm = if SPREADING_DEGREE_NORMALIZATION {
                1.0 / (1.0 + edges.len() as f32).sqrt()
            } else {
                1.0
            };

            for edge in edges {
                // Resolve the actual neighbor (handles bidirectional edge indexing)
                let target_uuid = edge.neighbor_of(&entity_uuid);

                // Edge-tier trust weight
                let tier_trust = if edge.is_potentiated() {
                    EDGE_TIER_TRUST_LTP
                } else {
                    match edge.tier {
                        EdgeTier::L3Semantic => EDGE_TIER_TRUST_L3,
                        EdgeTier::L2Episodic => EDGE_TIER_TRUST_L2,
                        EdgeTier::L1Working => EDGE_TIER_TRUST_L1,
                    }
                };

                // Direction-aware strength: use the directional component
                // matching the traversal direction (A5 Max Sphere)
                let directional = edge.directional_strength(&entity_uuid);
                let effective = edge.effective_strength();
                // Blend: directional strength modulates the time-decayed undirected strength
                // When directional data exists, it biases the effective strength
                let blended = if directional > 0.0 && (directional - effective).abs() > f32::EPSILON
                {
                    // Weight directional signal at 40%, undirected at 60%
                    // This prevents old edges (pre-directional) from losing signal
                    effective * 0.6 + directional * 0.4
                } else {
                    effective
                };

                let decay_rate = calculate_importance_weighted_decay(blended);
                let decay = (-decay_rate * hop as f32).exp();

                let base_spread = source_activation * decay * effective * tier_trust * degree_norm;

                // Ontological type penalty: dampen activation through wrong-type edges/entities.
                // CoOccurs, RelatedTo, CoRetrieved are generic bridges — never penalized.
                let spread_amount = if let Some(intent) = ontological_intent {
                    let mut penalty = 1.0_f32;

                    // Relation type penalty
                    if !intent.relation_types.is_empty()
                        && !matches!(
                            edge.relation_type,
                            RelationType::CoOccurs
                                | RelationType::RelatedTo
                                | RelationType::CoRetrieved
                        )
                        && !intent.relation_types.contains(&edge.relation_type)
                    {
                        penalty *= ONTOLOGICAL_RELATION_PENALTY;
                    }

                    // Entity type penalty (target entity label mismatch)
                    // Uses hierarchical matching: Team matches Organization, Service matches Technology
                    // Cached: each entity is read from RocksDB at most once across all hops.
                    if !intent.expected_labels.is_empty() {
                        let cached_labels =
                            entity_label_cache.entry(target_uuid).or_insert_with(|| {
                                graph
                                    .get_entity(&target_uuid)
                                    .ok()
                                    .flatten()
                                    .map(|e| e.labels)
                            });
                        if let Some(labels) = cached_labels {
                            let type_match = labels.iter().any(|l| {
                                intent
                                    .expected_labels
                                    .iter()
                                    .any(|exp| l.matches_with_hierarchy(exp))
                            });
                            if !type_match {
                                penalty *= ONTOLOGICAL_ENTITY_PENALTY;
                            }
                        }
                    }

                    base_spread * penalty
                } else {
                    base_spread
                };

                let new_activation = activation_map.entry(target_uuid).or_insert(0.0);
                *new_activation += spread_amount;

                if spread_amount > 0.01 {
                    traversed_edges.push(edge.uuid);
                    // Record (edge, contribution) for path-credit normalisation.
                    credit_edges.push((edge.uuid, spread_amount));
                }
            }
        }

        // Normalize to prevent unbounded growth
        let max_activation = activation_map
            .values()
            .cloned()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.0);

        if max_activation > SPREADING_NORMALIZATION_FACTOR {
            let scale = SPREADING_NORMALIZATION_FACTOR / max_activation;
            for activation in activation_map.values_mut() {
                *activation *= scale;
            }
        }

        // Prune weak activations
        activation_map.retain(|_, activation| *activation > threshold);
    }

    Ok((activation_map, traversed_edges, credit_edges))
}

/// Bidirectional spreading activation with intersection boost (PIPE-7)
///
/// Splits focal entities into forward/backward sets, spreads from each,
/// and boosts entities found at the intersection.
///
/// # Arguments
/// - `entity_data`: Focal entities with (uuid, name, ic_weight, salience)
/// - `graph`: The knowledge graph
/// - `total_salience`: Sum of entity saliences for normalization
/// - `hops_per_direction`: Density-adaptive hop count (2-4)
///
/// Returns: (combined_activation_map, traversed_edges, path_credit_edges, intersection_count)
fn bidirectional_spread(
    entity_data: &[(Uuid, String, f32, f32)], // (uuid, name, ic_weight, salience)
    graph: &GraphMemory,
    total_salience: f32,
    hops_per_direction: usize,
    ontological_intent: Option<&OntologicalIntent>,
) -> Result<(HashMap<Uuid, f32>, Vec<Uuid>, Vec<(Uuid, f32)>, usize)> {
    // Split entities into forward/backward sets (alternating assignment)
    // This distributes entities evenly regardless of count
    let mut forward_seeds: Vec<(Uuid, f32)> = Vec::new();
    let mut backward_seeds: Vec<(Uuid, f32)> = Vec::new();

    for (i, (uuid, _name, ic_weight, salience)) in entity_data.iter().enumerate() {
        let normalized_salience = salience / total_salience;
        let salience_boost = SALIENCE_BOOST_FACTOR * normalized_salience;
        let initial_activation = ic_weight * (1.0 + salience_boost);

        if i % 2 == 0 {
            forward_seeds.push((*uuid, initial_activation));
        } else {
            backward_seeds.push((*uuid, initial_activation));
        }
    }

    // If odd number of entities, backward might be empty - add last entity to both
    if backward_seeds.is_empty() && !forward_seeds.is_empty() {
        backward_seeds.push(forward_seeds[forward_seeds.len() - 1]);
    }

    tracing::debug!(
        "🔀 Bidirectional spread: {} forward seeds, {} backward seeds",
        forward_seeds.len(),
        backward_seeds.len()
    );

    // Spread from both directions with density-adaptive hops
    // Shared entity label cache: each entity is read from RocksDB at most once
    // across both forward and backward passes.
    let mut entity_label_cache: HashMap<Uuid, Option<Vec<EntityLabel>>> = HashMap::new();
    let threshold = SPREADING_ACTIVATION_THRESHOLD;
    let (forward_map, forward_edges, forward_credit) = spread_single_direction(
        &forward_seeds,
        graph,
        hops_per_direction,
        threshold,
        ontological_intent,
        &mut entity_label_cache,
    )?;

    let (backward_map, backward_edges, backward_credit) = spread_single_direction(
        &backward_seeds,
        graph,
        hops_per_direction,
        threshold,
        ontological_intent,
        &mut entity_label_cache,
    )?;

    // Combine maps with intersection boost
    let mut combined_map: HashMap<Uuid, f32> = HashMap::new();
    let mut intersection_count = 0;

    // Collect all entities from both directions
    let all_entities: std::collections::HashSet<Uuid> = forward_map
        .keys()
        .chain(backward_map.keys())
        .cloned()
        .collect();

    for entity_uuid in all_entities {
        let forward_activation = forward_map.get(&entity_uuid).cloned().unwrap_or(0.0);
        let backward_activation = backward_map.get(&entity_uuid).cloned().unwrap_or(0.0);

        // Check if this is an intersection entity (meaningful activation from both)
        let is_intersection = forward_activation >= BIDIRECTIONAL_INTERSECTION_MIN
            && backward_activation >= BIDIRECTIONAL_INTERSECTION_MIN;

        let combined_activation = if is_intersection {
            intersection_count += 1;
            // Intersection boost: these are "bridge" concepts
            (forward_activation + backward_activation) * BIDIRECTIONAL_INTERSECTION_BOOST
        } else {
            // Non-intersection: just sum the activations
            forward_activation + backward_activation
        };

        combined_map.insert(entity_uuid, combined_activation);
    }

    // Combine traversed edges and path-credit contributions
    let mut all_edges = forward_edges;
    all_edges.extend(backward_edges);
    let mut all_credit = forward_credit;
    all_credit.extend(backward_credit);

    tracing::debug!(
        "🔀 Bidirectional result: {} entities ({} intersections), {} edges",
        combined_map.len(),
        intersection_count,
        all_edges.len()
    );

    Ok((combined_map, all_edges, all_credit, intersection_count))
}

/// Spreading activation retrieval (legacy - uses fixed weights)
///
/// This is the core algorithm implementing Anderson & Pirolli (1984)
/// spreading activation model adapted for episodic memory retrieval.
///
/// For SHO-26 density-dependent weights, use `spreading_activation_retrieve_with_stats`
pub fn spreading_activation_retrieve(
    query_text: &str,
    query: &Query,
    graph: &GraphMemory,
    embedder: &dyn Embedder,
    episode_to_memory_fn: impl Fn(&EpisodicNode) -> Result<Option<SharedMemory>>,
) -> Result<Vec<ActivatedMemory>> {
    // Delegate to enhanced version with default density (uses legacy weights)
    let (memories, _stats) = spreading_activation_retrieve_with_stats(
        query_text,
        query,
        graph,
        embedder,
        None, // No density = use legacy fixed weights
        None, // No pre-computed query type = classify internally
        None, // No pre-computed intent = compute internally
        episode_to_memory_fn,
    )?;
    Ok(memories)
}

/// Spreading activation retrieval with density-dependent weights (SHO-26)
///
/// Enhanced version that:
/// - Uses density-dependent hybrid weights (graph trust scales with associations)
/// - Uses query-type-dependent weights when density is unavailable
/// - Applies importance-weighted decay (important memories decay slower)
/// - Tracks per-signal contribution variance for observability
/// - Returns RetrievalStats for observability
///
/// # Arguments
/// - `query_text`: The search query
/// - `query`: Query parameters including max_results
/// - `graph`: The knowledge graph for spreading activation
/// - `embedder`: Embedding model for semantic scoring
/// - `graph_density`: Optional density (edges/memories). If None, uses query-type or fixed weights.
/// - `query_type`: Optional query classification. Used for weight selection when density is None.
/// - `ontological_intent`: Pre-computed ontological intent. When `Some`, avoids recomputing.
/// - `episode_to_memory_fn`: Function to convert episodes to memories
///
/// # Returns
/// (Vec<ActivatedMemory>, RetrievalStats)
#[allow(clippy::too_many_arguments)]
pub fn spreading_activation_retrieve_with_stats(
    query_text: &str,
    query: &Query,
    graph: &GraphMemory,
    embedder: &dyn Embedder,
    graph_density: Option<f32>,
    query_type: Option<&QueryType>,
    ontological_intent: Option<&OntologicalIntent>,
    episode_to_memory_fn: impl Fn(&EpisodicNode) -> Result<Option<SharedMemory>>,
) -> Result<(Vec<ActivatedMemory>, RetrievalStats)> {
    let start_time = Instant::now();
    let mut stats = RetrievalStats::default();

    let computed_query_type;
    let resolved_query_type = match query_type {
        Some(query_type) => Some(query_type),
        None => {
            computed_query_type = classify_query(query_text);
            Some(&computed_query_type)
        }
    };

    // Determine weights: density takes precedence, then query type, then defaults.
    let (semantic_weight, graph_weight, linguistic_weight) = if let Some(density) = graph_density {
        stats.mode = "associative".to_string();
        stats.graph_density = density;
        stats.weight_profile = "density".to_string();
        calculate_density_weights(density)
    } else if let Some(query_type) = resolved_query_type {
        stats.mode = "hybrid".to_string();
        stats.graph_density = 0.0;
        match query_type {
            QueryType::Attribute(_) => {
                stats.weight_profile = "query_type:attribute".to_string();
                (
                    QUERY_TYPE_ATTRIBUTE_SEMANTIC,
                    QUERY_TYPE_ATTRIBUTE_GRAPH,
                    QUERY_TYPE_ATTRIBUTE_LINGUISTIC,
                )
            }
            QueryType::Temporal => {
                stats.weight_profile = "query_type:temporal".to_string();
                (
                    QUERY_TYPE_TEMPORAL_SEMANTIC,
                    QUERY_TYPE_TEMPORAL_GRAPH,
                    QUERY_TYPE_TEMPORAL_LINGUISTIC,
                )
            }
            QueryType::Exploratory => {
                stats.weight_profile = "query_type:exploratory".to_string();
                (
                    QUERY_TYPE_EXPLORATORY_SEMANTIC,
                    QUERY_TYPE_EXPLORATORY_GRAPH,
                    QUERY_TYPE_EXPLORATORY_LINGUISTIC,
                )
            }
        }
    } else {
        stats.mode = "hybrid".to_string();
        stats.graph_density = 0.0;
        stats.weight_profile = "default".to_string();
        (
            HYBRID_SEMANTIC_WEIGHT,
            HYBRID_GRAPH_WEIGHT,
            HYBRID_LINGUISTIC_WEIGHT,
        )
    };

    stats.semantic_weight = semantic_weight;
    stats.graph_weight = graph_weight;
    stats.linguistic_weight = linguistic_weight;

    // Step 1: Linguistic query analysis (Lioma & Ounis 2006)
    let analysis = analyze_query(query_text);

    // Ontological intent: use pre-computed if provided, otherwise infer from query structure
    let computed_intent;
    let ontological_intent_resolved = match ontological_intent {
        Some(intent) => intent,
        None => {
            computed_intent = infer_ontological_intent(query_text, &analysis);
            &computed_intent
        }
    };
    let use_ontology = ontological_intent_resolved.confidence >= ONTOLOGICAL_MIN_CONFIDENCE
        && !graph_density.is_some_and(|d| d >= ONTOLOGICAL_DENSITY_THRESHOLD);
    let intent_ref = if use_ontology {
        Some(ontological_intent_resolved)
    } else {
        None
    };

    if use_ontology {
        tracing::info!(
            "Ontological intent: labels={:?}, relations={:?}, confidence={:.2}",
            ontological_intent_resolved.expected_labels,
            ontological_intent_resolved
                .relation_types
                .iter()
                .map(|r| r.as_str())
                .collect::<Vec<_>>(),
            ontological_intent_resolved.confidence
        );
    }

    tracing::info!("🔍 Query Analysis:");
    tracing::info!(
        "  Focal Entities: {:?}",
        analysis
            .focal_entities
            .iter()
            .map(|e| &e.text)
            .collect::<Vec<_>>()
    );
    tracing::info!(
        "  Modifiers: {:?}",
        analysis
            .discriminative_modifiers
            .iter()
            .map(|m| &m.text)
            .collect::<Vec<_>>()
    );
    tracing::info!(
        "  Relations: {:?}",
        analysis
            .relational_context
            .iter()
            .map(|r| &r.text)
            .collect::<Vec<_>>()
    );
    tracing::info!(
        "  Weights: semantic={:.2}, graph={:.2}, linguistic={:.2}",
        semantic_weight,
        graph_weight,
        linguistic_weight
    );

    // Step 2: Initialize activation map from focal entities (nouns)
    // ACT-R inspired: weight initial activation by entity salience (attention budget)
    let mut activation_map: HashMap<Uuid, f32> = HashMap::new();

    // First pass: collect entities with their salience values
    let mut entity_data: Vec<(Uuid, String, f32, f32)> = Vec::new(); // (uuid, name, ic_weight, salience)

    for entity in &analysis.focal_entities {
        if let Some(entity_node) = graph.find_entity_by_name(&entity.text)? {
            entity_data.push((
                entity_node.uuid,
                entity.text.clone(),
                entity.ic_weight,
                entity_node.salience,
            ));
            // FIX-09: Track entity resolution success
            stats
                .entity_resolution
                .push(crate::memory::types::EntityResolution {
                    name: entity.text.clone(),
                    status: "found".to_string(),
                    resolved_to: Some(entity_node.uuid.to_string()),
                });
        } else {
            tracing::debug!("  ✗ Entity '{}' not found in graph", entity.text);
            // FIX-09: Track entity resolution failure
            stats
                .entity_resolution
                .push(crate::memory::types::EntityResolution {
                    name: entity.text.clone(),
                    status: "not_found".to_string(),
                    resolved_to: None,
                });
        }
    }

    // Calculate total salience for normalization (attention budget)
    let total_salience: f32 = entity_data.iter().map(|(_, _, _, s)| s).sum();
    let total_salience = total_salience.max(0.1); // Prevent division by zero

    // Second pass: apply salience-weighted activation
    let mut total_boost = 0.0_f32;
    for (uuid, name, ic_weight, salience) in &entity_data {
        // Normalize salience across query entities (ACT-R attention budget)
        let normalized_salience = salience / total_salience;

        // ACT-R inspired: activation = IC_weight × (1 + boost_factor × normalized_salience)
        let salience_boost = SALIENCE_BOOST_FACTOR * normalized_salience;
        let initial_activation = ic_weight * (1.0 + salience_boost);

        activation_map.insert(*uuid, initial_activation);
        stats.entities_activated += 1;
        total_boost += salience_boost;

        tracing::debug!(
            "  ✓ Activated '{}' (IC={:.2}, salience={:.2}, norm={:.2}, boost={:.2}, activation={:.2})",
            name,
            ic_weight,
            salience,
            normalized_salience,
            salience_boost,
            initial_activation
        );
    }

    // Track average salience boost for observability
    stats.avg_salience_boost = if !entity_data.is_empty() {
        total_boost / entity_data.len() as f32
    } else {
        0.0
    };

    if activation_map.is_empty() {
        tracing::warn!("No entities found in graph, falling back to semantic search");
        stats.retrieval_time_us = start_time.elapsed().as_micros() as u64;
        return Ok((Vec::new(), stats)); // Caller should fall back to semantic search
    }

    // Step 3: Spread activation through graph
    // PIPE-7: Use bidirectional spreading when 2+ focal entities, else unidirectional
    let graph_start = Instant::now();
    let mut traversed_edges: Vec<Uuid>;
    // Path-credit raw contributions (edge_uuid, spread_amount) — normalised later.
    let raw_credit_edges: Vec<(Uuid, f32)>;

    if entity_data.len() >= BIDIRECTIONAL_MIN_ENTITIES {
        // PIPE-7: Bidirectional spreading activation
        // Split entities into forward/backward sets, spread from each, boost intersections
        // Density-adaptive hops: dense (fresh) graphs use fewer hops, sparse (mature) use more
        let adaptive_hops = calculate_adaptive_hops(graph_density);

        tracing::info!(
            "🔀 Using bidirectional spreading ({} focal entities, {} hops/direction, density={:.2})",
            entity_data.len(),
            adaptive_hops,
            graph_density.unwrap_or(0.0)
        );

        let (bidirectional_map, edges, credit, intersection_count) = bidirectional_spread(
            &entity_data,
            graph,
            total_salience,
            adaptive_hops,
            intent_ref,
        )?;

        activation_map = bidirectional_map;
        traversed_edges = edges;
        raw_credit_edges = credit;
        stats.entities_activated = activation_map.len();
        stats.graph_hops = adaptive_hops * 2; // Effective depth (forward + backward)

        tracing::info!(
            "🔀 Bidirectional complete: {} entities, {} intersections",
            activation_map.len(),
            intersection_count
        );
    } else {
        // Standard unidirectional spreading (Anderson & Pirolli 1984)
        // With importance-weighted decay (SHO-26) and adaptive limits
        tracing::info!(
            "📊 Using unidirectional spreading ({} focal entity)",
            entity_data.len()
        );

        let mut edges_collected: Vec<Uuid> = Vec::new();
        let mut credit_collected: Vec<(Uuid, f32)> = Vec::new();
        let mut current_threshold = SPREADING_ACTIVATION_THRESHOLD;
        // Entity label cache: each entity is read from RocksDB at most once across all hops.
        let mut entity_label_cache: HashMap<Uuid, Option<Vec<EntityLabel>>> = HashMap::new();

        for hop in 1..=SPREADING_MAX_HOPS {
            stats.graph_hops = hop;
            let count_before = activation_map.len();

            tracing::debug!(
                "📊 Spreading activation (hop {}/{}, threshold={:.4})",
                hop,
                SPREADING_MAX_HOPS,
                current_threshold
            );

            // Clone to avoid borrow issues
            let current_activated: Vec<(Uuid, f32)> =
                activation_map.iter().map(|(id, act)| (*id, *act)).collect();

            for (entity_uuid, source_activation) in current_activated {
                // Only spread from entities with sufficient activation
                if source_activation < current_threshold {
                    continue;
                }

                // Get relationships from this entity (limited to prevent blowup)
                const MAX_EDGES_PER_SPREAD: usize = 100;
                let edges = graph
                    .get_entity_relationships_limited(&entity_uuid, Some(MAX_EDGES_PER_SPREAD))?;

                for edge in edges {
                    // Resolve the real neighbour, as the bidirectional path does.
                    //
                    // This was `edge.to_entity` unconditionally. Edges are indexed
                    // under BOTH endpoints (index_entity_edge writes two keys), so
                    // get_entity_relationships_limited returns edges where
                    // entity_uuid is sometimes the `to` side — and on those, taking
                    // `to_entity` spread activation from a node back onto ITSELF
                    // instead of to its neighbour. A self-loop that inflates the
                    // source's own score and never reaches the entity the edge was
                    // recording a relationship to.
                    //
                    // Still divergent from the bidirectional path, deliberately and
                    // not silently: that path also blends directional_strength
                    // (effective*0.6 + directional*0.4), so learned edge direction
                    // affects scoring there and not here. That is a scoring change
                    // rather than a correctness fix, so it is left for a change that
                    // can be measured on its own.
                    let target_uuid = edge.neighbor_of(&entity_uuid);

                    // Edge-tier trust weight (SHO-D1, PIPE-4)
                    let tier_trust = if edge.is_potentiated() {
                        EDGE_TIER_TRUST_LTP
                    } else {
                        match edge.tier {
                            EdgeTier::L3Semantic => EDGE_TIER_TRUST_L3,
                            EdgeTier::L2Episodic => EDGE_TIER_TRUST_L2,
                            EdgeTier::L1Working => EDGE_TIER_TRUST_L1,
                        }
                    };

                    // SHO-26: Importance-weighted decay (using effective_strength for time-aware decay)
                    let effective = edge.effective_strength();
                    let decay_rate = calculate_importance_weighted_decay(effective);
                    let decay = (-decay_rate * hop as f32).exp();

                    let base_spread = source_activation * decay * effective * tier_trust;

                    // Ontological type penalty (same as bidirectional path)
                    let spread_amount = if let Some(intent) = intent_ref {
                        let mut penalty = 1.0_f32;
                        if !intent.relation_types.is_empty()
                            && !matches!(
                                edge.relation_type,
                                RelationType::CoOccurs
                                    | RelationType::RelatedTo
                                    | RelationType::CoRetrieved
                            )
                            && !intent.relation_types.contains(&edge.relation_type)
                        {
                            penalty *= ONTOLOGICAL_RELATION_PENALTY;
                        }
                        if !intent.expected_labels.is_empty() {
                            let cached_labels =
                                entity_label_cache.entry(target_uuid).or_insert_with(|| {
                                    graph
                                        .get_entity(&target_uuid)
                                        .ok()
                                        .flatten()
                                        .map(|e| e.labels)
                                });
                            if let Some(labels) = cached_labels {
                                let type_match = labels.iter().any(|l| {
                                    intent
                                        .expected_labels
                                        .iter()
                                        .any(|exp| l.matches_with_hierarchy(exp))
                                });
                                if !type_match {
                                    penalty *= ONTOLOGICAL_ENTITY_PENALTY;
                                }
                            }
                        }
                        base_spread * penalty
                    } else {
                        base_spread
                    };

                    let new_activation = activation_map.entry(target_uuid).or_insert(0.0);
                    *new_activation += spread_amount;

                    if spread_amount > 0.01 {
                        edges_collected.push(edge.uuid);
                        // Record contribution for path-credit normalisation (B10).
                        credit_collected.push((edge.uuid, spread_amount));
                    }

                    if *new_activation >= current_threshold
                        && *new_activation - spread_amount < current_threshold
                    {
                        stats.entities_activated += 1;
                    }
                }
            }

            // Normalize activations to prevent unbounded growth
            let max_activation = activation_map
                .values()
                .cloned()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(1.0);

            if max_activation > SPREADING_NORMALIZATION_FACTOR {
                let scale = SPREADING_NORMALIZATION_FACTOR / max_activation;
                for activation in activation_map.values_mut() {
                    *activation *= scale;
                }
            }

            // Prune weak activations (ACT-R model)
            activation_map.retain(|_, activation| *activation > current_threshold);

            let count_after = activation_map.len();
            let new_activations = count_after.saturating_sub(count_before);

            tracing::debug!(
                "  Activated entities: {} (+{} new)",
                count_after,
                new_activations
            );

            // Adaptive threshold relaxation
            if count_after < SPREADING_MIN_CANDIDATES
                && current_threshold > SPREADING_RELAXED_THRESHOLD
            {
                current_threshold = SPREADING_RELAXED_THRESHOLD;
                tracing::debug!(
                    "  Relaxing threshold to {:.4} (only {} candidates)",
                    current_threshold,
                    count_after
                );
            }

            // Early termination checks (only after minimum hops)
            if hop >= SPREADING_MIN_HOPS {
                let new_ratio = if count_after > 0 {
                    new_activations as f32 / count_after as f32
                } else {
                    0.0
                };

                if new_ratio < SPREADING_EARLY_TERMINATION_RATIO && count_after > 0 {
                    tracing::debug!(
                        "  Early termination: activation saturated ({:.1}% new)",
                        new_ratio * 100.0
                    );
                    break;
                }

                if count_after >= SPREADING_EARLY_TERMINATION_CANDIDATES {
                    tracing::debug!(
                        "  Early termination: sufficient coverage ({} candidates)",
                        count_after
                    );
                    break;
                }
            }
        }

        traversed_edges = edges_collected;
        raw_credit_edges = credit_collected;
    }

    stats.graph_time_us = graph_start.elapsed().as_micros() as u64;
    tracing::info!("📊 Final activated entities: {}", activation_map.len());

    // Step 4: Retrieve episodic memories connected to activated entities
    let mut activated_memories: HashMap<Uuid, (f32, EpisodicNode)> = HashMap::new();

    for (entity_uuid, entity_activation) in &activation_map {
        let episodes = graph.get_episodes_by_entity(entity_uuid)?;

        for episode in episodes {
            // Accumulate activation for each episode (might be connected to multiple entities)
            let current = activated_memories
                .entry(episode.uuid)
                .or_insert((0.0, episode.clone()));

            current.0 += entity_activation;
        }
    }

    stats.graph_candidates = activated_memories.len();
    tracing::info!(
        "📊 Retrieved {} episodic memories via graph",
        activated_memories.len()
    );

    // Step 5: Convert episodes to memories and calculate scores using UNIFIED scoring
    let mut scored_memories = Vec::new();
    let mut total_semantic_contrib = 0.0_f32;
    let mut total_graph_contrib = 0.0_f32;
    let mut total_linguistic_contrib = 0.0_f32;
    let mut variance_sample_count = 0_usize;

    // Generate query embedding once (for semantic scoring)
    let embedding_start = Instant::now();
    let query_embedding = embedder.encode_for_query(query_text)?;
    stats.embedding_time_us = embedding_start.elapsed().as_micros() as u64;

    let now = chrono::Utc::now();

    for (_episode_uuid, (graph_activation, episode)) in activated_memories {
        // Convert episode to memory
        if let Some(memory) = episode_to_memory_fn(&episode)? {
            // Calculate semantic similarity (still needed for ActivatedMemory debug fields)
            let semantic_score = if let Some(mem_emb) = &memory.experience.embeddings {
                cosine_similarity(&query_embedding, mem_emb)
            } else {
                0.0
            };

            // Calculate linguistic match score (normalized to 0.0-1.0)
            let linguistic_raw = calculate_linguistic_match(&memory, &analysis);
            let linguistic_score = linguistic_raw; // Already normalized in calculate_linguistic_match

            // Unified scoring using density-dependent weights (calculated at function start),
            // with the memory-tier trust discount applied to the graph EVIDENCE and nothing
            // else. See `memory_tier_graph_trust` / `fuse_hybrid_score`.
            let tier_graph_trust = memory_tier_graph_trust(memory.tier);
            let hybrid_score = fuse_hybrid_score(
                semantic_score,
                graph_activation,
                linguistic_score,
                semantic_weight,
                graph_weight,
                linguistic_weight,
                tier_graph_trust,
            );

            if hybrid_score > 0.0 {
                // The weight mix is tier-independent; only the graph term carries the trust.
                let weight_sum = semantic_weight + graph_weight + linguistic_weight;
                let norm_semantic = semantic_weight / weight_sum;
                let norm_graph = graph_weight / weight_sum;
                let norm_linguistic = linguistic_weight / weight_sum;
                total_semantic_contrib += (semantic_score * norm_semantic) / hybrid_score;
                total_graph_contrib +=
                    (graph_activation * tier_graph_trust * norm_graph) / hybrid_score;
                total_linguistic_contrib += (linguistic_score * norm_linguistic) / hybrid_score;
                variance_sample_count += 1;
            }

            // Recency decay (10% contribution) - recent memories get boost
            // λ = 0.01 means ~50% at 70 hours, ~25% at 140 hours
            const RECENCY_DECAY_RATE: f32 = 0.01;
            let hours_old = (now - memory.created_at).num_hours().max(0) as f32;
            let recency_boost = (-RECENCY_DECAY_RATE * hours_old).exp() * 0.1;

            // Emotional arousal boost: high arousal = more salient (5% contribution)
            let arousal_boost = memory
                .experience
                .context
                .as_ref()
                .map(|c| c.emotional.arousal * 0.05)
                .unwrap_or(0.0);

            // Source credibility boost (5% contribution)
            let credibility_boost = memory
                .experience
                .context
                .as_ref()
                .map(|c| (c.source.credibility - 0.5).max(0.0) * 0.1)
                .unwrap_or(0.0);

            let final_score = hybrid_score + recency_boost + arousal_boost + credibility_boost;

            scored_memories.push(ActivatedMemory {
                memory,
                activation_score: graph_activation,
                semantic_score,
                linguistic_score,
                final_score,
            });
        }
    }

    if variance_sample_count > 0 {
        let n = variance_sample_count as f32;
        stats.semantic_variance_pct = total_semantic_contrib / n;
        stats.graph_variance_pct = total_graph_contrib / n;
        stats.linguistic_variance_pct = total_linguistic_contrib / n;

        crate::metrics::RETRIEVAL_VARIANCE_SEMANTIC.observe(stats.semantic_variance_pct as f64);
        crate::metrics::RETRIEVAL_VARIANCE_GRAPH.observe(stats.graph_variance_pct as f64);
        crate::metrics::RETRIEVAL_VARIANCE_LINGUISTIC.observe(stats.linguistic_variance_pct as f64);
    }

    // Step 6: Sort by final score (descending), memory id as the tie-break: the truncate below
    // makes any tie a coin toss over WHICH memories are returned (shodh-memory 6272430).
    scored_memories.sort_by(|a, b| {
        b.final_score
            .total_cmp(&a.final_score)
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });

    // Step 7: Apply limit
    scored_memories.truncate(query.max_results);

    stats.retrieval_time_us = start_time.elapsed().as_micros() as u64;

    // Deduplicate traversed edges (same edge may be traversed multiple times across hops)
    traversed_edges.sort();
    traversed_edges.dedup();
    stats.traversed_edges = traversed_edges;

    // Path-credit normalisation (B10): merge duplicate edge entries by summing their
    // contributions, then store raw sums in stats.  GraphMemory::apply_path_credit will
    // perform the final normalisation to 1.0 when called at feedback time.
    {
        let mut credit_map: std::collections::HashMap<Uuid, f32> = std::collections::HashMap::new();
        for (edge_uuid, contrib) in raw_credit_edges {
            *credit_map.entry(edge_uuid).or_insert(0.0) += contrib;
        }
        stats.path_credit_edges = credit_map.into_iter().collect();
    }

    // FIX-08: Populate retrieval mode recommendation based on graph state
    let density = stats.graph_density;
    let entity_count = stats.entities_activated;
    if density > 0.0 && density < 1.0 && entity_count >= 2 {
        stats.recommended_mode = Some("associative".to_string());
        stats.mode_rationale = Some(format!(
            "sparse graph (density={density:.2}) + {entity_count} entities: graph edges are high-signal"
        ));
    } else if density >= 2.0 || entity_count == 0 {
        stats.recommended_mode = Some("semantic".to_string());
        stats.mode_rationale = Some(format!(
            "dense/no-entity graph (density={density:.2}, entities={entity_count}): trust vector similarity"
        ));
    } else {
        stats.recommended_mode = Some("hybrid".to_string());
        stats.mode_rationale = Some("balanced density: use combined signals".to_string());
    }

    tracing::info!(
        "🎯 Returning {} memories (top scores: {:?}), {} edges traversed",
        scored_memories.len(),
        scored_memories
            .iter()
            .take(3)
            .map(|m| m.final_score)
            .collect::<Vec<_>>(),
        stats.traversed_edges.len()
    );

    Ok((scored_memories, stats))
}

/// Calculate linguistic feature match score
///
/// Based on IC-weighted matching:
/// - Focal entities (nouns): 1.0 point each
/// - Modifiers (adjectives): 0.5 points each
/// - Relations (verbs): 0.2 points each
fn calculate_linguistic_match(memory: &Memory, analysis: &QueryAnalysis) -> f32 {
    let content_lower = memory.experience.content.to_lowercase();
    let mut score = 0.0;

    // Entity matches (nouns) - highest weight
    for entity in &analysis.focal_entities {
        if content_lower.contains(&entity.text.to_lowercase()) {
            score += 1.0;
        }
    }

    // Modifier matches (adjectives) - medium weight
    for modifier in &analysis.discriminative_modifiers {
        if content_lower.contains(&modifier.text.to_lowercase()) {
            score += 0.5;
        }
    }

    // Relation matches (verbs) - low weight (they're "bus stops")
    for relation in &analysis.relational_context {
        if content_lower.contains(&relation.text.to_lowercase()) {
            score += 0.2;
        }
    }

    // Normalize by total possible score
    let max_possible = analysis.focal_entities.len() as f32 * 1.0
        + analysis.discriminative_modifiers.len() as f32 * 0.5
        + analysis.relational_context.len() as f32 * 0.2;

    if max_possible > 0.0 {
        score / max_possible
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_weights_sparse() {
        // Sparse graph: density < 0.5 -> MAX graph weight (trust graph)
        let (semantic, graph, linguistic) = calculate_density_weights(0.3);
        assert!((graph - DENSITY_GRAPH_WEIGHT_MAX).abs() < 0.001);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_weights_dense() {
        // Dense graph: density > 2.0 -> MIN graph weight (trust vector)
        let (semantic, graph, linguistic) = calculate_density_weights(2.5);
        assert!((graph - DENSITY_GRAPH_WEIGHT_MIN).abs() < 0.001);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_weights_interpolation() {
        // Medium density: should interpolate
        let (semantic, graph, linguistic) = calculate_density_weights(1.25);
        assert!(graph > DENSITY_GRAPH_WEIGHT_MIN);
        assert!(graph < DENSITY_GRAPH_WEIGHT_MAX);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_importance_weighted_decay_high() {
        // High importance -> low decay (preserve signal)
        let decay = calculate_importance_weighted_decay(1.0);
        assert!((decay - IMPORTANCE_DECAY_MIN).abs() < 0.001);
    }

    #[test]
    fn test_importance_weighted_decay_low() {
        // Low importance -> high decay (explore)
        let decay = calculate_importance_weighted_decay(0.0);
        assert!((decay - IMPORTANCE_DECAY_MAX).abs() < 0.001);
    }

    #[test]
    fn test_importance_weighted_decay_mid() {
        // Medium importance -> intermediate decay
        let decay = calculate_importance_weighted_decay(0.5);
        let expected = IMPORTANCE_DECAY_MIN + 0.5 * (IMPORTANCE_DECAY_MAX - IMPORTANCE_DECAY_MIN);
        assert!((decay - expected).abs() < 0.001);
    }

    #[test]
    fn test_activation_decay() {
        // Test that importance-weighted decay varies correctly
        let initial_activation = 1.0;

        // High importance = slow decay
        let high_importance_decay = calculate_importance_weighted_decay(0.9);
        let high_importance_final = initial_activation * (-high_importance_decay).exp();

        // Low importance = fast decay
        let low_importance_decay = calculate_importance_weighted_decay(0.1);
        let low_importance_final = initial_activation * (-low_importance_decay).exp();

        // High importance should retain more activation
        assert!(high_importance_final > low_importance_final);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_adaptive_constants_valid() {
        use crate::constants::*;

        // Relaxed threshold must be lower than strict threshold
        assert!(SPREADING_RELAXED_THRESHOLD < SPREADING_ACTIVATION_THRESHOLD);

        // Min hops must be <= max hops
        assert!(SPREADING_MIN_HOPS <= SPREADING_MAX_HOPS);

        // Early termination ratio must be in (0, 1)
        assert!(SPREADING_EARLY_TERMINATION_RATIO > 0.0);
        assert!(SPREADING_EARLY_TERMINATION_RATIO < 1.0);

        // Normalization factor must be positive
        assert!(SPREADING_NORMALIZATION_FACTOR > 0.0);

        // Min candidates for relaxation should be reasonable
        assert!(SPREADING_MIN_CANDIDATES > 0);
        assert!(SPREADING_MIN_CANDIDATES < SPREADING_EARLY_TERMINATION_CANDIDATES);
    }

    #[test]
    fn test_normalization_prevents_explosion() {
        use crate::constants::SPREADING_NORMALIZATION_FACTOR;

        // Simulate activation accumulation over hops
        let mut activations: Vec<f32> = vec![1.0, 0.8, 0.5, 0.3];

        // Simulate 5 hops of accumulation (each hop adds to existing)
        for _ in 0..5 {
            for activation in &mut activations {
                *activation += *activation * 0.5; // 50% growth per hop
            }

            // Normalize
            let max_activation = activations
                .iter()
                .cloned()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(1.0);

            if max_activation > SPREADING_NORMALIZATION_FACTOR {
                let scale = SPREADING_NORMALIZATION_FACTOR / max_activation;
                for activation in &mut activations {
                    *activation *= scale;
                }
            }
        }

        // After normalization, max should never exceed factor
        let final_max = activations
            .iter()
            .cloned()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        assert!(final_max <= SPREADING_NORMALIZATION_FACTOR + 0.001);
    }

    #[test]
    fn test_early_termination_ratio() {
        use crate::constants::SPREADING_EARLY_TERMINATION_RATIO;

        // Simulate saturation detection
        let total_before = 50;
        let total_after = 52; // Only 2 new activations

        let new_activations = total_after - total_before;
        let new_ratio = new_activations as f32 / total_after as f32;

        // Should trigger early termination (only ~4% new)
        assert!(new_ratio < SPREADING_EARLY_TERMINATION_RATIO);

        // Simulate rapid growth - should not terminate
        let growing_before = 10;
        let growing_after = 25; // 15 new activations

        let growing_new = growing_after - growing_before;
        let growing_ratio = growing_new as f32 / growing_after as f32;

        // Should NOT trigger early termination (60% new)
        assert!(growing_ratio >= SPREADING_EARLY_TERMINATION_RATIO);
    }

    // =========================================================================
    // PIPE-7: Bidirectional Spreading Activation Tests
    // =========================================================================

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_bidirectional_constants_valid() {
        // Minimum entities must be at least 2 for bidirectional to make sense
        assert!(BIDIRECTIONAL_MIN_ENTITIES >= 2);

        // Intersection boost must be positive
        assert!(BIDIRECTIONAL_INTERSECTION_BOOST > 1.0);

        // Intersection minimum must be below activation threshold
        assert!(BIDIRECTIONAL_INTERSECTION_MIN < SPREADING_ACTIVATION_THRESHOLD);

        // Density thresholds must be ordered correctly
        assert!(BIDIRECTIONAL_DENSITY_SPARSE < BIDIRECTIONAL_DENSITY_DENSE);

        // Hop counts must be ordered: dense < medium < sparse
        assert!(BIDIRECTIONAL_HOPS_DENSE < BIDIRECTIONAL_HOPS_MEDIUM);
        assert!(BIDIRECTIONAL_HOPS_MEDIUM < BIDIRECTIONAL_HOPS_SPARSE);

        // Medium hops × 2 should approximate max hops for unidirectional
        assert!(BIDIRECTIONAL_HOPS_MEDIUM * 2 >= SPREADING_MAX_HOPS);
    }

    #[test]
    fn test_adaptive_hops_dense_graph() {
        // Dense graph (fresh system): fewer hops to avoid noise
        let hops = calculate_adaptive_hops(Some(3.0)); // Above DENSE threshold
        assert_eq!(hops, BIDIRECTIONAL_HOPS_DENSE);
        assert_eq!(hops, 2);
    }

    #[test]
    fn test_adaptive_hops_sparse_graph() {
        // Sparse graph (mature system): more hops for quality edges
        let hops = calculate_adaptive_hops(Some(0.3)); // Below SPARSE threshold
        assert_eq!(hops, BIDIRECTIONAL_HOPS_SPARSE);
        assert_eq!(hops, 4);
    }

    #[test]
    fn test_adaptive_hops_medium_graph() {
        // Medium density: balanced exploration
        let hops = calculate_adaptive_hops(Some(1.0)); // Between thresholds
        assert_eq!(hops, BIDIRECTIONAL_HOPS_MEDIUM);
        assert_eq!(hops, 3);
    }

    #[test]
    fn test_adaptive_hops_no_density() {
        // No density info: use medium as safe default
        let hops = calculate_adaptive_hops(None);
        assert_eq!(hops, BIDIRECTIONAL_HOPS_MEDIUM);
    }

    #[test]
    fn test_adaptive_hops_lifecycle() {
        // Simulate graph lifecycle: dense → medium → sparse
        let fresh_hops = calculate_adaptive_hops(Some(2.5)); // Fresh system
        let mid_hops = calculate_adaptive_hops(Some(1.0)); // 6 months in
        let mature_hops = calculate_adaptive_hops(Some(0.3)); // Mature system

        // Hops should increase as graph matures (gets sparser)
        assert!(fresh_hops <= mid_hops);
        assert!(mid_hops <= mature_hops);

        // Verify actual values
        assert_eq!(fresh_hops, 2);
        assert_eq!(mid_hops, 3);
        assert_eq!(mature_hops, 4);
    }

    #[test]
    fn test_intersection_boost_calculation() {
        // Test that intersection entities get boosted
        let forward_activation = 0.5;
        let backward_activation = 0.3;

        // Both exceed minimum threshold
        assert!(forward_activation >= BIDIRECTIONAL_INTERSECTION_MIN);
        assert!(backward_activation >= BIDIRECTIONAL_INTERSECTION_MIN);

        // Intersection boost calculation
        let boosted = (forward_activation + backward_activation) * BIDIRECTIONAL_INTERSECTION_BOOST;
        let unboosted = forward_activation + backward_activation;

        // Boosted should be higher
        assert!(boosted > unboosted);

        // Boosted should be exactly 1.5× unboosted (with default constants)
        let expected_ratio = BIDIRECTIONAL_INTERSECTION_BOOST;
        assert!((boosted / unboosted - expected_ratio).abs() < 0.001);
    }

    #[test]
    fn test_non_intersection_no_boost() {
        // Test that non-intersection entities don't get boosted
        let forward_activation = 0.5;
        let backward_activation = 0.0; // Below threshold

        // Backward doesn't meet threshold
        assert!(backward_activation < BIDIRECTIONAL_INTERSECTION_MIN);

        // Non-intersection: just sum
        let combined = forward_activation + backward_activation;

        // Should equal just forward activation
        assert!((combined - forward_activation).abs() < 0.001);
    }

    #[test]
    fn test_bidirectional_entity_split() {
        // Test alternating assignment distributes evenly
        let entities = [
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity3".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity4".to_string(), 1.0, 0.5),
        ];

        // With 4 entities, split should be 2-2
        let mut forward_count = 0;
        let mut backward_count = 0;

        for (i, _) in entities.iter().enumerate() {
            if i % 2 == 0 {
                forward_count += 1;
            } else {
                backward_count += 1;
            }
        }

        assert_eq!(forward_count, 2);
        assert_eq!(backward_count, 2);
    }

    #[test]
    fn test_bidirectional_odd_entities() {
        // Test odd number of entities doesn't leave backward empty
        let entities = [
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity3".to_string(), 1.0, 0.5),
        ];

        // With 3 entities: indices 0,2 go forward, index 1 goes backward
        let mut forward_seeds = Vec::new();
        let mut backward_seeds = Vec::new();

        for (i, entity) in entities.iter().enumerate() {
            if i % 2 == 0 {
                forward_seeds.push(entity.0);
            } else {
                backward_seeds.push(entity.0);
            }
        }

        // 2 forward, 1 backward
        assert_eq!(forward_seeds.len(), 2);
        assert_eq!(backward_seeds.len(), 1);

        // Both sets are non-empty (the actual function duplicates if backward would be empty)
        assert!(!forward_seeds.is_empty());
        assert!(!backward_seeds.is_empty());
    }

    #[test]
    fn test_bidirectional_threshold_triggers() {
        // Test when bidirectional is triggered vs unidirectional

        // 1 entity: unidirectional
        let single_entity = [(Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5)];
        assert!(single_entity.len() < BIDIRECTIONAL_MIN_ENTITIES);

        // 2 entities: bidirectional
        let two_entities = [
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
        ];
        assert!(two_entities.len() >= BIDIRECTIONAL_MIN_ENTITIES);

        // 5 entities: bidirectional
        let many_entities = [
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity3".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity4".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity5".to_string(), 1.0, 0.5),
        ];
        assert!(many_entities.len() >= BIDIRECTIONAL_MIN_ENTITIES);
    }

    #[test]
    fn test_complexity_improvement() {
        // Theoretical complexity comparison
        // Let b = branching factor, d = depth

        // Unidirectional: O(b^d)
        // Bidirectional: O(2 × b^(d/2))

        // For b=10, d=6:
        // Unidirectional: 10^6 = 1,000,000
        // Bidirectional: 2 × 10^3 = 2,000

        let b: f64 = 10.0;
        let d: f64 = 6.0;

        let unidirectional = b.powf(d);
        let bidirectional = 2.0 * b.powf(d / 2.0);

        // Bidirectional should be significantly smaller
        assert!(bidirectional < unidirectional);

        // Ratio should be ~500× improvement
        let improvement = unidirectional / bidirectional;
        assert!(improvement > 100.0);
    }

    #[test]
    fn test_intersection_detection_threshold() {
        // Test the minimum threshold for intersection detection
        let min_threshold = BIDIRECTIONAL_INTERSECTION_MIN;

        // Should be half of activation threshold
        let expected = SPREADING_ACTIVATION_THRESHOLD / 2.0;
        assert!((min_threshold - expected).abs() < 0.001);

        // Should be positive
        assert!(min_threshold > 0.0);

        // Should be less than typical initial activation
        // (IC_NOUN * (1 + SALIENCE_BOOST_FACTOR) = 2.3 * 2 = 4.6 max)
        assert!(min_threshold < 1.0);
    }

    // =========================================================================
    // Query-type weight modulation tests (Plan 001 §2.2)
    // =========================================================================

    #[test]
    fn test_query_type_weights_temporal_boosts_linguistic() {
        let base = (0.5, 0.35, 0.15); // semantic, graph, linguistic
        let (s, g, l) =
            apply_query_type_weights(base, &crate::memory::query_parser::QueryType::Temporal);
        // Linguistic should increase, graph should decrease
        assert!(l > 0.15, "Temporal should boost linguistic: got {l}");
        assert!(g < 0.35, "Temporal should reduce graph: got {g}");
        // Weights must sum to 1.0
        assert!(((s + g + l) - 1.0).abs() < 0.001, "Weights must sum to 1.0");
    }

    #[test]
    fn test_query_type_weights_attribute_boosts_graph() {
        let base = (0.5, 0.35, 0.15);
        let attr = crate::memory::query_parser::AttributeQuery {
            entity: "Alice".to_string(),
            attribute: "age".to_string(),
            attribute_synonyms: vec![],
            original_query: "How old is Alice?".to_string(),
        };
        let (s, g, l) = apply_query_type_weights(
            base,
            &crate::memory::query_parser::QueryType::Attribute(attr),
        );
        // Graph should increase, semantic should decrease
        assert!(g > 0.35, "Attribute should boost graph: got {g}");
        assert!(s < 0.5, "Attribute should reduce semantic: got {s}");
        assert!(((s + g + l) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_query_type_weights_exploratory_unchanged() {
        let base = (0.5, 0.35, 0.15);
        let (s, g, l) =
            apply_query_type_weights(base, &crate::memory::query_parser::QueryType::Exploratory);
        // Should be unchanged after renormalization
        assert!(((s + g + l) - 1.0).abs() < 0.001);
        // Proportions should be preserved (no shift applied)
        assert!((s - 0.5).abs() < 0.01);
        assert!((g - 0.35).abs() < 0.01);
        assert!((l - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_query_type_weights_no_channel_zeroed() {
        // Even with extreme base weights, no channel should go below 0.05
        let extreme = (0.9, 0.05, 0.05);
        let (s, g, l) =
            apply_query_type_weights(extreme, &crate::memory::query_parser::QueryType::Temporal);
        assert!(s >= 0.05, "Semantic should not go below floor: got {s}");
        assert!(g >= 0.05, "Graph should not go below floor: got {g}");
        assert!(l >= 0.05, "Linguistic should not go below floor: got {l}");
        assert!(((s + g + l) - 1.0).abs() < 0.001);
    }
}

#[cfg(test)]
mod fusion_tests {
    // Memory-tier graph trust must attenuate GRAPH EVIDENCE and nothing else. It used to be
    // folded into the weight vector and renormalised per candidate, so a candidate's tier
    // changed the weight on its SEMANTIC and LINGUISTIC legs too, and those scores were ranked
    // against each other. Ported from shodh-memory 653dc00.
    use super::{fuse_hybrid_score, memory_tier_graph_trust};
    use crate::constants::MEMORY_TIER_GRAPH_MULT_LONGTERM;
    use crate::memory::types::MemoryTier;

    /// The shipped density weights at the sparse end: semantic 0.5, graph 0.3, linguistic 0.2.
    const W_SEM: f32 = 0.5;
    const W_GRAPH: f32 = 0.3;
    const W_LING: f32 = 0.2;

    #[test]
    fn tier_trust_leaves_the_semantic_and_linguistic_legs_untouched() {
        // Identical evidence, different trust, zero graph activation: no graph term, so the
        // scores must be exactly equal. Under the renormalising form they were not
        // (0.5·0.5/0.79 = 0.3165 vs 0.5·0.5/1.0 = 0.25, a 27% spread driven purely by tier).
        let low = fuse_hybrid_score(0.5, 0.0, 0.4, W_SEM, W_GRAPH, W_LING, 0.3);
        let high = fuse_hybrid_score(0.5, 0.0, 0.4, W_SEM, W_GRAPH, W_LING, 1.0);
        assert_eq!(
            low, high,
            "with no graph evidence, tier trust must not move the score"
        );
        assert!((low - (0.5 * 0.5 + 0.4 * 0.2)).abs() < 1e-6, "got {low}");
    }

    #[test]
    fn tier_trust_scales_the_graph_term_linearly() {
        let base = fuse_hybrid_score(0.2, 0.0, 0.1, W_SEM, W_GRAPH, W_LING, 0.3);
        let at_03 = fuse_hybrid_score(0.2, 0.8, 0.1, W_SEM, W_GRAPH, W_LING, 0.3);
        let at_06 = fuse_hybrid_score(0.2, 0.8, 0.1, W_SEM, W_GRAPH, W_LING, 0.6);
        let at_10 = fuse_hybrid_score(0.2, 0.8, 0.1, W_SEM, W_GRAPH, W_LING, 1.0);
        assert!((at_03 - base - 0.8 * 0.3 * 0.3).abs() < 1e-6);
        assert!(
            ((at_06 - base) - 2.0 * (at_03 - base)).abs() < 1e-6,
            "graph contribution must be linear in trust"
        );
        assert!(((at_10 - base) - 0.8 * 1.0 * 0.3).abs() < 1e-6);
    }

    #[test]
    fn a_more_trusted_memory_is_never_outranked_on_identical_evidence() {
        // The failure shape this fix exists to prevent: same evidence, one LongTerm, one
        // Working. The renormalising form ranked the Working candidate ABOVE whenever
        // semantic evidence dominated, because discounting its graph leg inflated its semantic leg.
        let (sem, graph, ling) = (0.72_f32, 0.05_f32, 0.30_f32);
        let working = fuse_hybrid_score(
            sem,
            graph,
            ling,
            W_SEM,
            W_GRAPH,
            W_LING,
            memory_tier_graph_trust(MemoryTier::Working),
        );
        let longterm = fuse_hybrid_score(
            sem,
            graph,
            ling,
            W_SEM,
            W_GRAPH,
            W_LING,
            memory_tier_graph_trust(MemoryTier::LongTerm),
        );
        assert!(
            longterm >= working,
            "equal evidence: longterm={longterm} working={working}"
        );

        // The old form really did invert it, so this pins a regression and not a tautology.
        let legacy = |trust: f32| {
            let sum = W_SEM + W_GRAPH * trust + W_LING;
            sem * (W_SEM / sum) + graph * (W_GRAPH * trust / sum) + ling * (W_LING / sum)
        };
        assert!(
            legacy(0.3) > legacy(1.0),
            "the renormalising form is expected to invert the ordering here"
        );
    }

    #[test]
    fn longterm_is_the_undiscounted_reference() {
        assert_eq!(
            memory_tier_graph_trust(MemoryTier::LongTerm),
            MEMORY_TIER_GRAPH_MULT_LONGTERM
        );
        assert!(
            memory_tier_graph_trust(MemoryTier::Working)
                < memory_tier_graph_trust(MemoryTier::LongTerm)
        );
    }

    #[test]
    fn degenerate_weights_do_not_produce_nan() {
        // A NaN would poison the sort comparator rather than merely rank badly.
        assert_eq!(fuse_hybrid_score(0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 1.0), 0.0);
    }
}
