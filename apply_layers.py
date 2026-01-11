#!/usr/bin/env python3
"""Apply layered architecture changes to semantic_retrieve"""

import re

with open('src/memory/mod.rs', 'r', encoding='utf-8') as f:
    content = f.read()

# Layer 1+2 insertion after embedding cache
layer12 = '''
        // ===========================================================================
        // LAYER 1: TEMPORAL PRE-FILTER (Episode Coherence)
        // ===========================================================================
        let episode_candidates: Option<HashSet<MemoryId>> = if let Some(episode_id) = &query.episode_id {
            match self.long_term_memory.search(SearchCriteria::ByEpisode(episode_id.clone())) {
                Ok(ep) if !ep.is_empty() => {
                    tracing::debug!("Layer 1: {} candidates in episode {}", ep.len(), episode_id);
                    Some(ep.into_iter().map(|m| m.id).collect())
                }
                _ => { tracing::debug!("Layer 1: global search"); None }
            }
        } else { None };

        // ===========================================================================
        // LAYER 2: GRAPH EXPANSION (Knowledge Graph Traversal)
        // ===========================================================================
        let (graph_results, graph_density): (Vec<(MemoryId, f32, f32)>, Option<f32>) = {
            if let Some(graph) = &self.graph_memory {
                let g = graph.read();
                let a = query_parser::analyze_query(query_text);
                let d = g.get_stats().ok().and_then(|s| if s.entity_count > 0 { Some(s.relationship_count as f32 / s.entity_count as f32) } else { None });
                let mut ids = Vec::new();
                for e in a.focal_entities.iter().map(|e| e.text.as_str()).chain(a.discriminative_modifiers.iter().map(|m| m.text.as_str())) {
                    if let Ok(Some(ent)) = g.find_entity_by_name(e) {
                        if let Ok(t) = g.traverse_from_entity(&ent.uuid, 2) {
                            for tr in &t.entities {
                                if let Ok(eps) = g.get_episodes_by_entity(&tr.entity.uuid) {
                                    for ep in eps {
                                        let mid = MemoryId(ep.uuid);
                                        if episode_candidates.as_ref().map_or(true, |c| c.contains(&mid)) {
                                            ids.push((mid, tr.entity.salience * tr.decay_factor, tr.decay_factor));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let mut seen: std::collections::HashMap<MemoryId, (f32, f32)> = std::collections::HashMap::new();
                for (id, act, heb) in ids { seen.entry(id).and_modify(|(a,h)| { *a = a.max(act); *h = h.max(heb); }).or_insert((act, heb)); }
                let r: Vec<_> = seen.into_iter().map(|(id, (a, h))| (id, a, h)).collect();
                if !r.is_empty() { tracing::debug!("Layer 2: {} graph results", r.len()); }
                (r, d)
            } else { (Vec::new(), None) }
        };

'''

# Insert Layer 1+2 after embedding block
content = re.sub(
    r'(embedding\n        \};\n\n)(        // Create a modified query)',
    r'\1' + layer12 + r'\2',
    content
)

# Add episode_id and prospective_signals to Query
content = re.sub(
    r'(offset: query\.offset,\n)(        \};)',
    r'\1            episode_id: query.episode_id.clone(),\n            prospective_signals: query.prospective_signals.clone(),\n\2',
    content
)

# Replace vector search with Layer 3+4
old_vs = '''        // Get memory IDs from vector search (fast HNSW search)
        let vector_results = self
            .retriever
            .search_ids(&vector_query, query.max_results * 2)?; // Get more for hybrid fusion

        // HYBRID SEARCH: Combine BM25 (keyword) + Vector (semantic) with RRF fusion
        // This improves recall for both exact keyword matches and semantic similarity
        let memory_ids = {
            // Get content for reranking
            let get_content = |id: &MemoryId| -> Option<String> {
                // Try caches first, then storage
                if let Some(m) = self.working_memory.read().get(id) {
                    return Some(m.experience.content.clone());
                }
                if let Some(m) = self.session_memory.read().get(id) {
                    return Some(m.experience.content.clone());
                }
                self.long_term_memory
                    .get(id)
                    .ok()
                    .map(|m| m.experience.content.clone())
            };

            // Run hybrid search (BM25 + RRF + optional reranking)
            match self
                .hybrid_search
                .search(query_text, vector_results.clone(), get_content)
            {
                Ok(hybrid_results) => {
                    // Convert HybridSearchResult to (MemoryId, score) pairs
                    hybrid_results
                        .into_iter()
                        .take(query.max_results)
                        .map(|r| (r.memory_id, r.score))
                        .collect::<Vec<_>>()
                }
                Err(e) => {
                    // Fallback to vector-only if hybrid fails
                    tracing::warn!("Hybrid search failed, falling back to vector: {}", e);
                    vector_results
                }
            }
        };'''

new_vs = '''        // ===========================================================================
        // LAYER 3: VECTOR SEARCH (Vamana Index)
        // ===========================================================================
        let vr = self.retriever.search_ids(&vector_query, query.max_results * 3)?;
        let vector_results: Vec<(MemoryId, f32)> = if let Some(ref c) = episode_candidates {
            vr.into_iter().filter(|(id, _)| c.contains(id)).collect()
        } else { vr };
        tracing::debug!("Layer 3: {} vector results", vector_results.len());

        // ===========================================================================
        // LAYER 4: BM25 + RRF FUSION
        // ===========================================================================
        let (memory_ids, hebbian_scores): (Vec<(MemoryId, f32)>, std::collections::HashMap<MemoryId, f32>) = {
            let get_content = |id: &MemoryId| -> Option<String> {
                self.working_memory.read().get(id).map(|m| m.experience.content.clone())
                    .or_else(|| self.session_memory.read().get(id).map(|m| m.experience.content.clone()))
                    .or_else(|| self.long_term_memory.get(id).ok().map(|m| m.experience.content.clone()))
            };
            let hybrid_ids = self.hybrid_search.search(query_text, vector_results.clone(), get_content)
                .map(|r| r.into_iter().map(|x| (x.memory_id, x.score)).collect::<Vec<_>>())
                .unwrap_or(vector_results);

            const K: f32 = 60.0;
            let mut fused: std::collections::HashMap<MemoryId, f32> = std::collections::HashMap::new();
            let mut heb: std::collections::HashMap<MemoryId, f32> = std::collections::HashMap::new();
            let boost = graph_density.map(|d| 1.1 + d.min(2.0) * 0.7).unwrap_or(1.5);
            for (r, (id, _, h)) in graph_results.iter().enumerate() {
                *fused.entry(id.clone()).or_insert(0.0) += boost / (K + r as f32);
                heb.insert(id.clone(), *h);
            }
            for (r, (id, _)) in hybrid_ids.iter().enumerate() {
                *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (K + r as f32);
            }
            let mut res: Vec<_> = fused.into_iter().collect();
            res.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            res.truncate(query.max_results);
            tracing::debug!("Layer 4: {} fused results", res.len());
            (res, heb)
        };'''

content = content.replace(old_vs, new_vs)

with open('src/memory/mod.rs', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
