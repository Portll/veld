//! Memory Replay and Interference Module (SHO-105, SHO-106)
//!
//! This module implements biologically-inspired memory consolidation mechanisms:
//!
//! ## Memory Replay (SHO-105)
//! Based on Rasch & Born (2013) - sleep consolidation research:
//! - Hippocampus replays recent experiences during rest/sleep
//! - Co-activation strengthens related memories and their associations
//! - High-value memories (important + recent + emotional) get priority
//!
//! ## Memory Interference (SHO-106)
//! Based on Anderson & Neely (1996) - retrieval competition:
//! - Retroactive interference: new learning disrupts old memories
//! - Proactive interference: old memories interfere with new learning
//! - Similar memories compete during retrieval

use crate::constants::{
    INTERFERENCE_COMPETITION_FACTOR, INTERFERENCE_MAX_TRACKED, INTERFERENCE_PROACTIVE_DECAY,
    INTERFERENCE_PROACTIVE_THRESHOLD, INTERFERENCE_RETROACTIVE_DECAY,
    INTERFERENCE_SEVERE_THRESHOLD, INTERFERENCE_SIMILARITY_THRESHOLD,
    INTERFERENCE_VULNERABILITY_HOURS, REPLAY_AROUSAL_THRESHOLD, REPLAY_BATCH_SIZE,
    REPLAY_EDGE_BOOST, REPLAY_IMPORTANCE_THRESHOLD, REPLAY_MAX_AGE_DAYS, REPLAY_STRENGTH_BOOST,
};
use crate::memory::introspection::{ConsolidationEvent, InterferenceType};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Candidate memory for replay, scored by priority
#[derive(Debug, Clone)]
pub struct ReplayCandidate {
    pub memory_id: String,
    pub content_preview: String,
    pub importance: f32,
    pub arousal: f32,
    pub age_days: f64,
    pub connection_count: usize,
    pub priority_score: f32,
    pub connected_memory_ids: Vec<String>,
}

/// Result of a replay cycle
#[derive(Debug, Clone, Default)]
pub struct ReplayCycleResult {
    pub memories_replayed: usize,
    pub edges_strengthened: usize,
    pub total_priority_score: f32,
    pub events: Vec<ConsolidationEvent>,
    /// Edge boosts: (from_memory_id, to_memory_id, boost_value)
    /// To be applied via GraphMemory at API layer
    pub edge_boosts: Vec<(String, String, f32)>,
    /// Memory IDs that were replayed — used for entity-entity edge strengthening
    pub replay_memory_ids: Vec<String>,
}

/// Manager for memory replay during consolidation
///
/// Implements sleep-like consolidation by:
/// 1. Identifying high-value memories for replay
/// 2. Simulating co-activation during replay
/// 3. Strengthening both memories and their associations
pub struct ReplayManager {
    /// Last replay cycle timestamp
    last_replay: DateTime<Utc>,
    /// Minimum interval between replay cycles (hours)
    replay_interval_hours: i64,
    /// Replay statistics
    total_replays: usize,
}

impl Default for ReplayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayManager {
    pub fn new() -> Self {
        Self {
            last_replay: Utc::now() - Duration::hours(24), // Allow immediate first replay
            replay_interval_hours: 1,                      // Replay every hour during active use
            total_replays: 0,
        }
    }

    /// Check if replay cycle should run
    pub fn should_replay(&self) -> bool {
        let elapsed = Utc::now() - self.last_replay;
        elapsed.num_hours() >= self.replay_interval_hours
    }

    /// Identify memories eligible for replay
    ///
    /// Selection criteria:
    /// - Recent (within REPLAY_MAX_AGE_DAYS)
    /// - Important (above REPLAY_IMPORTANCE_THRESHOLD)
    /// - Connected (at least REPLAY_MIN_CONNECTIONS)
    /// - Optionally: high emotional arousal for priority
    #[allow(clippy::type_complexity)]
    pub fn identify_replay_candidates(
        &self,
        memories: &[(String, f32, f32, DateTime<Utc>, Vec<String>, String)], // (id, importance, arousal, created_at, connections, content_preview)
    ) -> Vec<ReplayCandidate> {
        let now = Utc::now();
        let mut candidates: Vec<ReplayCandidate> = memories
            .iter()
            .filter_map(
                |(id, importance, arousal, created_at, connections, preview)| {
                    let age = now - *created_at;
                    let age_days = age.num_hours() as f64 / 24.0;

                    // Check eligibility
                    if age_days > REPLAY_MAX_AGE_DAYS as f64 {
                        return None;
                    }
                    if *importance < REPLAY_IMPORTANCE_THRESHOLD {
                        return None;
                    }
                    // REPLAY_MIN_CONNECTIONS=0: importance alone qualifies for replay.
                    // Connections still boost priority via connectivity_factor below.

                    // Calculate priority score
                    // Priority = importance × recency_factor × (1 + arousal_boost) × connectivity_factor
                    let recency_factor = 1.0 - (age_days / REPLAY_MAX_AGE_DAYS as f64) as f32;
                    let arousal_boost = if *arousal > REPLAY_AROUSAL_THRESHOLD {
                        (*arousal - REPLAY_AROUSAL_THRESHOLD) * 0.5
                    } else {
                        0.0
                    };
                    let connectivity_factor = 1.0 + (connections.len() as f32 / 10.0).min(0.5); // Max 50% boost

                    let priority =
                        importance * recency_factor * (1.0 + arousal_boost) * connectivity_factor;

                    Some(ReplayCandidate {
                        memory_id: id.clone(),
                        content_preview: preview.clone(),
                        importance: *importance,
                        arousal: *arousal,
                        age_days,
                        connection_count: connections.len(),
                        priority_score: priority,
                        connected_memory_ids: connections.clone(),
                    })
                },
            )
            .collect();

        // Sort by priority (highest first)
        candidates.sort_by(|a, b| b.priority_score.total_cmp(&a.priority_score));

        // Take top REPLAY_BATCH_SIZE candidates
        candidates.truncate(REPLAY_BATCH_SIZE);
        candidates
    }

    /// Execute replay for a batch of candidates
    ///
    /// Returns strength boosts to apply to memories and edges
    #[allow(clippy::type_complexity)]
    pub fn execute_replay(
        &mut self,
        candidates: &[ReplayCandidate],
    ) -> (
        Vec<(String, f32)>,
        Vec<(String, String, f32)>,
        Vec<ConsolidationEvent>,
    ) {
        // (memory_id, boost), (from_id, to_id, boost), events
        let mut memory_boosts: Vec<(String, f32)> = Vec::new();
        let mut edge_boosts: Vec<(String, String, f32)> = Vec::new();
        let mut events: Vec<ConsolidationEvent> = Vec::new();
        let now = Utc::now();

        // Track replayed memories to avoid duplicate boosts
        let mut replayed: HashSet<String> = HashSet::new();

        for candidate in candidates {
            if replayed.contains(&candidate.memory_id) {
                continue;
            }

            // Boost the primary memory
            memory_boosts.push((candidate.memory_id.clone(), REPLAY_STRENGTH_BOOST));
            replayed.insert(candidate.memory_id.clone());

            // Co-activate connected memories
            let mut connected_replayed = 0;
            for connected_id in &candidate.connected_memory_ids {
                if !replayed.contains(connected_id) {
                    // Boost connected memory (slightly less than primary)
                    memory_boosts.push((connected_id.clone(), REPLAY_STRENGTH_BOOST * 0.5));
                    replayed.insert(connected_id.clone());
                }

                // Strengthen the edge between them
                edge_boosts.push((
                    candidate.memory_id.clone(),
                    connected_id.clone(),
                    REPLAY_EDGE_BOOST,
                ));
                connected_replayed += 1;
            }

            // Create replay event
            events.push(ConsolidationEvent::MemoryReplayed {
                memory_id: candidate.memory_id.clone(),
                content_preview: candidate.content_preview.clone(),
                activation_before: candidate.importance,
                activation_after: (candidate.importance + REPLAY_STRENGTH_BOOST).min(1.0),
                replay_priority: candidate.priority_score,
                connected_memories_replayed: connected_replayed,
                timestamp: now,
            });
        }

        self.last_replay = now;
        self.total_replays += replayed.len();

        (memory_boosts, edge_boosts, events)
    }

    /// Get replay statistics
    pub fn stats(&self) -> (usize, DateTime<Utc>) {
        (self.total_replays, self.last_replay)
    }
}

// =============================================================================
// MEMORY INTERFERENCE (SHO-106)
// =============================================================================

/// Record of an interference event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceRecord {
    pub interfering_memory_id: String,
    pub similarity: f32,
    pub interference_type: InterferenceType,
    pub strength_change: f32,
    pub timestamp: DateTime<Utc>,
}

/// Result of interference check during memory storage
#[derive(Debug, Clone, Default)]
pub struct InterferenceCheckResult {
    /// Retroactive interference: old memories to weaken
    pub retroactive_targets: Vec<(String, f32, f32)>, // (memory_id, similarity, decay_amount)
    /// Proactive interference: strength reduction for new memory
    pub proactive_decay: f32,
    /// Whether memories are duplicates (should merge instead of interfere)
    pub is_duplicate: bool,
    /// Events generated
    pub events: Vec<ConsolidationEvent>,
}

/// What one retrieval competition decides, before it is reported or recorded.
struct RetrievalCompetitionOutcome {
    winners: Vec<(String, f32)>,
    suppressed: Vec<String>,
    /// `(affected, winner, score_ratio, strength_change)` per interference record, in the
    /// order the candidates were ranked.
    interference: Vec<(String, String, f32, f32)>,
}

impl RetrievalCompetitionOutcome {
    /// Stronger memories suppress weaker ones: a competitor within 10% of the winner loses
    /// score in proportion to how close it came, and is suppressed when that leaves it at
    /// 0.1 or below. Close survivors and suppressed memories both leave an interference
    /// record (RIF feedback for Layer 4.6, PIPE-3), the survivors a milder one.
    fn of(candidates: &[(String, f32, f32)]) -> Self {
        let mut outcome = Self {
            winners: Vec::new(),
            suppressed: Vec::new(),
            interference: Vec::new(),
        };
        if candidates.len() <= 1 {
            outcome.winners = candidates
                .iter()
                .map(|(id, score, _)| (id.clone(), *score))
                .collect();
            return outcome;
        }

        let mut scores: Vec<(String, f32)> = candidates
            .iter()
            .map(|(id, score, _)| (id.clone(), *score))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let Some((winner_id, winner_score)) = scores.first().cloned() else {
            return outcome;
        };
        outcome.winners.push((winner_id.clone(), winner_score));

        // A zero-scoring winner means every remaining score is zero too: no competition.
        if winner_score <= 0.0 {
            outcome.winners.extend(scores.into_iter().skip(1));
            return outcome;
        }

        for (id, score) in scores.into_iter().skip(1) {
            let score_ratio = score / winner_score;
            if score_ratio > 0.9 {
                let suppression = INTERFERENCE_COMPETITION_FACTOR * (1.0 - score_ratio) * 10.0;
                let new_score = (score - suppression).max(0.0);
                if new_score > 0.1 {
                    outcome.interference.push((
                        id.clone(),
                        winner_id.clone(),
                        score_ratio,
                        suppression * 0.3,
                    ));
                    outcome.winners.push((id, new_score));
                } else {
                    outcome.interference.push((
                        id.clone(),
                        winner_id.clone(),
                        score_ratio,
                        suppression,
                    ));
                    outcome.suppressed.push(id);
                }
            } else {
                outcome.winners.push((id, score));
            }
        }
        outcome
    }
}

/// Result of retrieval competition
#[derive(Debug, Clone)]
pub struct CompetitionResult {
    /// Memory IDs that won (survive suppression)
    pub winners: Vec<(String, f32)>, // (memory_id, final_score)
    /// Memory IDs that were suppressed
    pub suppressed: Vec<String>,
    /// Competition factor applied
    pub competition_factor: f32,
    /// Event generated
    pub event: Option<ConsolidationEvent>,
}

/// Detector for memory interference effects
pub struct InterferenceDetector {
    /// Tracked interference records per memory
    interference_history: HashMap<String, Vec<InterferenceRecord>>,
    /// Total interference events
    total_interference_events: usize,
}

impl Default for InterferenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl InterferenceDetector {
    pub fn new() -> Self {
        Self {
            interference_history: HashMap::new(),
            total_interference_events: 0,
        }
    }

    /// Check for interference when storing a new memory
    ///
    /// Compares the new memory's embedding against existing memories
    /// and determines interference effects.
    pub fn check_interference(
        &mut self,
        new_memory_id: &str,
        new_memory_importance: f32,
        _new_memory_created: DateTime<Utc>,
        similar_memories: &[(String, f32, f32, DateTime<Utc>, String)], // (id, similarity, importance, created_at, content_preview)
    ) -> InterferenceCheckResult {
        let mut result = InterferenceCheckResult::default();
        let now = Utc::now();

        for (old_id, similarity, old_importance, old_created, old_preview) in similar_memories {
            // Skip self
            if old_id == new_memory_id {
                continue;
            }

            // Check if similarity exceeds threshold
            if *similarity < INTERFERENCE_SIMILARITY_THRESHOLD {
                continue;
            }

            // Check for duplicates (very high similarity)
            if *similarity >= INTERFERENCE_SEVERE_THRESHOLD {
                result.is_duplicate = true;
                // Return early - should merge, not interfere
                return result;
            }

            // Calculate interference effects
            let age_hours = (now - *old_created).num_hours();
            let is_vulnerable = age_hours < INTERFERENCE_VULNERABILITY_HOURS;

            // Retroactive interference: new memory weakens old
            if is_vulnerable || *old_importance < new_memory_importance {
                // Stronger interference for more similar memories
                let interference_strength = (*similarity - INTERFERENCE_SIMILARITY_THRESHOLD)
                    / (1.0 - INTERFERENCE_SIMILARITY_THRESHOLD);

                let decay = INTERFERENCE_RETROACTIVE_DECAY * interference_strength;
                result
                    .retroactive_targets
                    .push((old_id.clone(), *similarity, decay));

                // Record event
                result
                    .events
                    .push(ConsolidationEvent::InterferenceDetected {
                        new_memory_id: new_memory_id.to_string(),
                        old_memory_id: old_id.clone(),
                        similarity: *similarity,
                        interference_type: InterferenceType::Retroactive,
                        timestamp: now,
                    });

                result.events.push(ConsolidationEvent::MemoryWeakened {
                    memory_id: old_id.clone(),
                    content_preview: old_preview.clone(),
                    activation_before: *old_importance,
                    activation_after: (*old_importance - decay).max(0.05),
                    interfering_memory_id: new_memory_id.to_string(),
                    interference_type: InterferenceType::Retroactive,
                    timestamp: now,
                });

                // Track in history
                self.record_interference(
                    old_id,
                    new_memory_id,
                    *similarity,
                    InterferenceType::Retroactive,
                    decay,
                );
            }

            // Proactive interference: strong old memory suppresses new
            if *old_importance > INTERFERENCE_PROACTIVE_THRESHOLD {
                let interference_strength = (*similarity - INTERFERENCE_SIMILARITY_THRESHOLD)
                    / (1.0 - INTERFERENCE_SIMILARITY_THRESHOLD);

                let decay = INTERFERENCE_PROACTIVE_DECAY
                    * interference_strength
                    * (*old_importance - INTERFERENCE_PROACTIVE_THRESHOLD);

                result.proactive_decay += decay;

                result
                    .events
                    .push(ConsolidationEvent::InterferenceDetected {
                        new_memory_id: new_memory_id.to_string(),
                        old_memory_id: old_id.clone(),
                        similarity: *similarity,
                        interference_type: InterferenceType::Proactive,
                        timestamp: now,
                    });

                self.record_interference(
                    new_memory_id,
                    old_id,
                    *similarity,
                    InterferenceType::Proactive,
                    decay,
                );
            }
        }

        self.total_interference_events += result.events.len();
        result
    }

    /// Apply retrieval competition between similar memories
    ///
    /// When multiple similar memories are retrieved, they compete
    /// for activation. Stronger memories suppress weaker ones.
    /// `record_state` — whether the interference produced here is ACCUMULATED into the
    /// detector's history. Suppression is computed in every mode (it is per-query ranking);
    /// accumulation is a usage write that feeds later queries' boosts, so read-only recall
    /// (eval repeats) passes `false`. Ported from shodh-memory 4ec3b6e.
    pub fn apply_retrieval_competition(
        &mut self,
        candidates: &[(String, f32, f32)], // (memory_id, relevance_score, similarity_to_query)
        query_preview: &str,
        record_state: bool,
    ) -> CompetitionResult {
        let result = Self::compete_for_retrieval(candidates, query_preview);
        self.record_retrieval_competition(candidates, record_state);
        result
    }

    /// The ranking half of [`Self::apply_retrieval_competition`]: winners, suppressed and
    /// the event, from the candidates alone. Reads no detector state, so a recall can rank
    /// without the detector's write lock.
    pub fn compete_for_retrieval(
        candidates: &[(String, f32, f32)],
        query_preview: &str,
    ) -> CompetitionResult {
        let outcome = RetrievalCompetitionOutcome::of(candidates);
        let competition_factor = if candidates.len() <= 1 {
            0.0
        } else {
            INTERFERENCE_COMPETITION_FACTOR
        };
        let event = if !outcome.suppressed.is_empty() {
            Some(ConsolidationEvent::RetrievalCompetition {
                query_preview: query_preview.to_string(),
                winner_memory_id: outcome
                    .winners
                    .first()
                    .map(|(id, _)| id.clone())
                    .unwrap_or_default(),
                suppressed_memory_ids: outcome.suppressed.clone(),
                competition_factor: INTERFERENCE_COMPETITION_FACTOR,
                timestamp: Utc::now(),
            })
        } else {
            None
        };

        CompetitionResult {
            winners: outcome.winners,
            suppressed: outcome.suppressed,
            competition_factor,
            event,
        }
    }

    /// The state half of [`Self::apply_retrieval_competition`]: the interference the same
    /// competition leaves, recorded in its order when `record_state`, and the suppressed
    /// count added to the event total either way, as the combined call always did.
    pub fn record_retrieval_competition(
        &mut self,
        candidates: &[(String, f32, f32)],
        record_state: bool,
    ) {
        let outcome = RetrievalCompetitionOutcome::of(candidates);
        if record_state {
            for (id, winner_id, score_ratio, strength) in &outcome.interference {
                self.record_interference(
                    id,
                    winner_id,
                    *score_ratio,
                    InterferenceType::RetrievalCompetition,
                    *strength,
                );
            }
        }
        self.total_interference_events += outcome.suppressed.len();
    }

    /// Record an interference event
    pub(crate) fn record_interference(
        &mut self,
        affected_memory_id: &str,
        interfering_memory_id: &str,
        similarity: f32,
        interference_type: InterferenceType,
        strength_change: f32,
    ) {
        let record = InterferenceRecord {
            interfering_memory_id: interfering_memory_id.to_string(),
            similarity,
            interference_type,
            strength_change,
            timestamp: Utc::now(),
        };

        let history = self
            .interference_history
            .entry(affected_memory_id.to_string())
            .or_default();

        history.push(record);

        // Limit history size
        if history.len() > INTERFERENCE_MAX_TRACKED {
            history.remove(0);
        }
    }

    /// Get interference history for a memory
    /// Number of interference records held across all memories — what a usage write adds to.
    pub fn interference_history_len(&self) -> usize {
        self.interference_history.values().map(Vec::len).sum()
    }

    pub fn get_history(&self, memory_id: &str) -> Option<&Vec<InterferenceRecord>> {
        self.interference_history.get(memory_id)
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, usize) {
        (
            self.total_interference_events,
            self.interference_history.len(),
        )
    }

    /// Clear history for a deleted memory
    pub fn clear_memory(&mut self, memory_id: &str) {
        self.interference_history.remove(memory_id);
    }

    // =========================================================================
    // PIPE-3: INTERFERENCE-AWARE RETRIEVAL SCORING
    // =========================================================================
    //
    // Research basis:
    // - Anderson & Neely (1996): "Interference and inhibition in memory retrieval"
    // - Anderson et al. (1994): Retrieval-induced forgetting (RIF)
    // - Postman & Underwood (1973): "Critical issues in interference theory"
    //
    // Key insight: Retrieval is a competitive process where:
    // 1. Memories that frequently "lose" competitions become harder to retrieve
    // 2. Memories that survive despite competition become STRONGER (robust encoding)
    //
    // The "fan effect" (Anderson 1974): Memories with many competing associations
    // are harder to retrieve, BUT memories that maintain strength despite fans
    // are extra-reliable.
    // =========================================================================

    /// Calculate retrieval score adjustment based on interference history
    ///
    /// This implements Anderson's retrieval-induced forgetting (RIF) theory:
    /// - Memories with high interference that maintained activation → BOOST (survivors)
    /// - Memories with high interference and low activation → SUPPRESS (chronic losers)
    ///
    /// # Arguments
    /// * `memory_id` - The memory to score
    /// * `current_activation` - Current importance/activation level (0.0-1.0)
    ///
    /// # Returns
    /// Score adjustment factor:
    /// - > 1.0: boost (multiply score)
    /// - < 1.0: suppress (multiply score)
    /// - 1.0: no adjustment
    ///
    /// # Research Reference
    /// Anderson, M.C. & Neely, J.H. (1996). Interference and inhibition in
    /// memory retrieval. In E.L. Bjork & R.A. Bjork (Eds.), Memory (pp. 237-313).
    pub fn calculate_retrieval_adjustment(&self, memory_id: &str, current_activation: f32) -> f32 {
        let history = match self.interference_history.get(memory_id) {
            Some(h) if !h.is_empty() => h,
            _ => return 1.0, // No interference history → no adjustment
        };

        // Calculate interference metrics
        let interference_count = history.len();
        let total_strength_lost: f32 = history.iter().map(|r| r.strength_change.abs()).sum();
        let avg_similarity: f32 =
            history.iter().map(|r| r.similarity).sum::<f32>() / interference_count as f32;

        // Normalize interference intensity (0-1 scale)
        // Combines: event count + similarity + cumulative damage
        // More events + higher similarity + more damage = more intense competition history
        let count_factor = (interference_count as f32 / INTERFERENCE_MAX_TRACKED as f32).min(1.0);
        let damage_factor = (total_strength_lost / 0.5).min(1.0); // 0.5 total loss = max damage
        let interference_intensity = (count_factor * 0.5 + damage_factor * 0.5) * avg_similarity;

        // The key insight from Anderson's RIF research:
        // - High interference + high activation = SURVIVOR (boost)
        // - High interference + low activation = CHRONIC LOSER (suppress)
        //
        // Formula: adjustment = 1.0 + intensity * (2 * activation - 1)
        // - When activation = 1.0: adjustment = 1.0 + intensity (boost up to 2x)
        // - When activation = 0.5: adjustment = 1.0 (neutral)
        // - When activation = 0.0: adjustment = 1.0 - intensity (suppress down to 0x)

        let activation_factor = 2.0 * current_activation - 1.0; // Maps [0,1] to [-1,1]
        let adjustment = 1.0 + interference_intensity * activation_factor * 0.5;

        // Clamp to reasonable bounds (0.5x to 1.5x)
        adjustment.clamp(0.5, 1.5)
    }

    /// Batch calculate retrieval adjustments for multiple memories
    ///
    /// Efficient for scoring entire result sets.
    ///
    /// # Arguments
    /// * `memories` - Vec of (memory_id, current_activation)
    ///
    /// # Returns
    /// HashMap of memory_id → adjustment factor
    pub fn batch_retrieval_adjustments(&self, memories: &[(String, f32)]) -> HashMap<String, f32> {
        memories
            .iter()
            .map(|(id, activation)| {
                (
                    id.clone(),
                    self.calculate_retrieval_adjustment(id, *activation),
                )
            })
            .collect()
    }

    /// Check if a memory has significant interference history
    ///
    /// Useful for deciding whether to apply interference adjustments.
    pub fn has_significant_interference(&self, memory_id: &str) -> bool {
        self.interference_history
            .get(memory_id)
            .map(|h| h.len() >= 2) // At least 2 interference events
            .unwrap_or(false)
    }

    // =========================================================================
    // PERSISTENCE HELPERS
    // =========================================================================

    /// Bulk load interference history from persistent storage on startup
    ///
    /// Replaces the in-memory HashMap with persisted data. Called once during
    /// MemorySystem initialization.
    pub fn load_history(
        &mut self,
        history: HashMap<String, Vec<InterferenceRecord>>,
        total_events: usize,
    ) {
        self.interference_history = history;
        self.total_interference_events = total_events;
        tracing::info!(
            memories_tracked = self.interference_history.len(),
            total_events = self.total_interference_events,
            "Loaded interference history from persistent storage"
        );
    }

    /// Get memory IDs affected by a storage interference check
    ///
    /// Returns IDs that had interference records modified, for targeted persistence.
    pub fn get_affected_ids_from_check(
        &self,
        new_memory_id: &str,
        result: &InterferenceCheckResult,
    ) -> Vec<String> {
        let mut ids: Vec<String> = result
            .retroactive_targets
            .iter()
            .map(|(id, _, _)| id.clone())
            .collect();

        if result.proactive_decay > 0.0 {
            ids.push(new_memory_id.to_string());
        }

        ids
    }

    /// Get memory IDs affected by retrieval competition
    ///
    /// Returns IDs of all memories that had interference recorded
    /// (both suppressed and close survivors with score_ratio > 0.9).
    pub fn get_affected_ids_from_competition(&self, result: &CompetitionResult) -> Vec<String> {
        let mut ids = result.suppressed.clone();

        // Include close survivors that had mild interference recorded
        if let Some((_, winner_score)) = result.winners.first() {
            if *winner_score > 0.0 {
                for (id, score) in result.winners.iter().skip(1) {
                    let score_ratio = score / winner_score;
                    if score_ratio > 0.9 {
                        ids.push(id.clone());
                    }
                }
            }
        }

        ids
    }

    /// Get interference records for specific memory IDs (for targeted persistence)
    pub fn get_records_for_ids<'a>(
        &'a self,
        ids: &'a [String],
    ) -> Vec<(&'a str, &'a Vec<InterferenceRecord>)> {
        ids.iter()
            .filter_map(|id| {
                self.interference_history
                    .get(id)
                    .map(|records| (id.as_str(), records))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_candidate_identification() {
        let manager = ReplayManager::new();
        let now = Utc::now();

        // Create test memories
        let memories = vec![
            (
                "mem-1".to_string(),
                0.8,                                            // High importance
                0.7,                                            // High arousal
                now - Duration::hours(12),                      // Recent
                vec!["mem-2".to_string(), "mem-3".to_string()], // Connected
                "Important memory".to_string(),
            ),
            (
                "mem-2".to_string(),
                0.2, // Low importance - should be excluded
                0.3,
                now - Duration::hours(6),
                vec!["mem-1".to_string()],
                "Unimportant memory".to_string(),
            ),
            (
                "mem-3".to_string(),
                0.6,
                0.4,
                now - Duration::days(15), // Too old (>14 day max) - should be excluded
                vec!["mem-1".to_string(), "mem-4".to_string()],
                "Old memory".to_string(),
            ),
        ];

        let candidates = manager.identify_replay_candidates(&memories);

        // Only mem-1 should be eligible
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].memory_id, "mem-1");
        assert!(candidates[0].priority_score > 0.0);
    }

    #[test]
    fn test_replay_execution() {
        let mut manager = ReplayManager::new();
        let _now = Utc::now();

        let candidates = vec![ReplayCandidate {
            memory_id: "mem-1".to_string(),
            content_preview: "Test memory".to_string(),
            importance: 0.7,
            arousal: 0.6,
            age_days: 1.0,
            connection_count: 2,
            priority_score: 0.8,
            connected_memory_ids: vec!["mem-2".to_string(), "mem-3".to_string()],
        }];

        let (memory_boosts, edge_boosts, events) = manager.execute_replay(&candidates);

        // Primary memory should get a boost
        assert!(memory_boosts.iter().any(|(id, _)| id == "mem-1"));

        // Connected memories should get boosts
        assert!(memory_boosts.iter().any(|(id, _)| id == "mem-2"));
        assert!(memory_boosts.iter().any(|(id, _)| id == "mem-3"));

        // Edges should be strengthened
        assert_eq!(edge_boosts.len(), 2);

        // Event should be generated
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_interference_detection() {
        let mut detector = InterferenceDetector::new();
        let now = Utc::now();

        // Test retroactive interference
        let similar_memories = vec![(
            "old-mem".to_string(),
            0.90,                      // High similarity
            0.5,                       // Moderate importance
            now - Duration::hours(12), // Recent, vulnerable
            "Old memory content".to_string(),
        )];

        let result = detector.check_interference(
            "new-mem",
            0.7, // Higher importance than old
            now,
            &similar_memories,
        );

        // Should detect retroactive interference
        assert!(!result.retroactive_targets.is_empty());
        assert!(!result.events.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let mut detector = InterferenceDetector::new();
        let now = Utc::now();

        // Very similar memory (near duplicate)
        let similar_memories = vec![(
            "existing-mem".to_string(),
            0.98, // Very high similarity - duplicate
            0.5,
            now - Duration::hours(1),
            "Existing content".to_string(),
        )];

        let result = detector.check_interference("new-mem", 0.6, now, &similar_memories);

        // Should detect as duplicate
        assert!(result.is_duplicate);
        // No interference events for duplicates
        assert!(result.events.is_empty());
    }

    #[test]
    fn test_retrieval_competition() {
        let mut detector = InterferenceDetector::new();

        let candidates = vec![
            ("mem-1".to_string(), 0.9, 0.85),  // Winner
            ("mem-2".to_string(), 0.88, 0.82), // Close competitor
            ("mem-3".to_string(), 0.5, 0.70),  // Lower, should survive
        ];

        let result = detector.apply_retrieval_competition(&candidates, "test query", true);

        // Winner should be first
        assert_eq!(result.winners[0].0, "mem-1");
        // Close competitor may be suppressed depending on competition factor
        assert!(!result.winners.is_empty());
    }

    /// Read-only recall still competes (same winners, same suppression) but leaves no
    /// interference record behind, so a repeat measures the same detector.
    #[test]
    fn test_retrieval_competition_readonly_computes_but_does_not_record() {
        let candidates = vec![
            ("mem-1".to_string(), 0.9, 0.85),
            ("mem-2".to_string(), 0.88, 0.82),
            ("mem-3".to_string(), 0.5, 0.70),
        ];

        let mut recording = InterferenceDetector::new();
        let recorded = recording.apply_retrieval_competition(&candidates, "q", true);
        let recorded_history = recording.interference_history_len();

        let mut readonly = InterferenceDetector::new();
        let silent = readonly.apply_retrieval_competition(&candidates, "q", false);

        assert_eq!(
            recorded.winners, silent.winners,
            "ranking must not depend on the flag"
        );
        assert_eq!(recorded.suppressed, silent.suppressed);
        assert!(
            recorded_history > 0,
            "the recording run must have written something for the test to mean anything"
        );
        assert_eq!(
            readonly.interference_history_len(),
            0,
            "read-only recall taught the detector"
        );
    }

    /// Ranking without the detector and recording afterwards (a background writer's split)
    /// gives the combined call's result and leaves the same history and event total.
    #[test]
    fn competing_then_recording_matches_the_combined_call() {
        let candidates = vec![
            ("mem-1".to_string(), 0.9, 0.85),
            ("mem-2".to_string(), 0.895, 0.82),
            ("mem-3".to_string(), 0.86, 0.80),
            ("mem-4".to_string(), 0.5, 0.70),
        ];

        let mut combined = InterferenceDetector::new();
        let together = combined.apply_retrieval_competition(&candidates, "q", true);

        let mut split = InterferenceDetector::new();
        let ranked = InterferenceDetector::compete_for_retrieval(&candidates, "q");
        split.record_retrieval_competition(&candidates, true);

        assert_eq!(together.winners, ranked.winners);
        assert_eq!(together.suppressed, ranked.suppressed);
        assert_eq!(together.competition_factor, ranked.competition_factor);
        assert_eq!(together.event.is_some(), ranked.event.is_some());
        assert_eq!(
            combined.interference_history_len(),
            split.interference_history_len()
        );
        assert!(split.interference_history_len() > 0);
        assert_eq!(combined.stats().0, split.stats().0);
        for id in ["mem-2", "mem-3"] {
            let a: Vec<f32> = combined
                .get_history(id)
                .unwrap()
                .iter()
                .map(|r| r.strength_change)
                .collect();
            let b: Vec<f32> = split
                .get_history(id)
                .unwrap()
                .iter()
                .map(|r| r.strength_change)
                .collect();
            assert_eq!(a, b);
        }

        let single = vec![("only".to_string(), 0.7, 0.6)];
        let alone = InterferenceDetector::compete_for_retrieval(&single, "q");
        assert_eq!(alone.winners, vec![("only".to_string(), 0.7)]);
        assert_eq!(alone.competition_factor, 0.0);
    }

    // =========================================================================
    // PIPE-3: Interference-Aware Retrieval Tests
    // =========================================================================

    #[test]
    fn test_retrieval_adjustment_no_history() {
        let detector = InterferenceDetector::new();

        // Memory with no interference history should get neutral adjustment
        let adjustment = detector.calculate_retrieval_adjustment("unknown-mem", 0.8);
        assert_eq!(
            adjustment, 1.0,
            "No history should return neutral adjustment"
        );
    }

    #[test]
    fn test_retrieval_adjustment_survivor_boost() {
        let mut detector = InterferenceDetector::new();
        let now = Utc::now();

        // Simulate interference history for a "survivor" memory
        // (high interference but maintained high activation)
        let _similar_memories = [(
            "survivor-mem".to_string(),
            0.90,
            0.85, // High importance maintained despite interference
            now - Duration::hours(12),
            "Survivor content".to_string(),
        )];

        // Record multiple interference events
        for i in 0..5 {
            detector.record_interference(
                "survivor-mem",
                &format!("interferer-{}", i),
                0.88,
                InterferenceType::Retroactive,
                0.05,
            );
        }

        // High activation (0.9) + interference history = BOOST
        let adjustment = detector.calculate_retrieval_adjustment("survivor-mem", 0.9);
        assert!(
            adjustment > 1.0,
            "Survivor (high activation despite interference) should be boosted: {}",
            adjustment
        );
    }

    #[test]
    fn test_retrieval_adjustment_chronic_loser_suppress() {
        let mut detector = InterferenceDetector::new();

        // Simulate interference history for a "chronic loser" memory
        // (high interference and low activation = weak memory)
        for i in 0..5 {
            detector.record_interference(
                "loser-mem",
                &format!("winner-{}", i),
                0.88,
                InterferenceType::Retroactive,
                0.1,
            );
        }

        // Low activation (0.2) + interference history = SUPPRESS
        let adjustment = detector.calculate_retrieval_adjustment("loser-mem", 0.2);
        assert!(
            adjustment < 1.0,
            "Chronic loser (low activation with interference) should be suppressed: {}",
            adjustment
        );
    }

    #[test]
    fn test_retrieval_adjustment_neutral_midpoint() {
        let mut detector = InterferenceDetector::new();

        // Record some interference
        for i in 0..3 {
            detector.record_interference(
                "neutral-mem",
                &format!("other-{}", i),
                0.87,
                InterferenceType::Retroactive,
                0.05,
            );
        }

        // Medium activation (0.5) should be near neutral
        let adjustment = detector.calculate_retrieval_adjustment("neutral-mem", 0.5);
        assert!(
            (adjustment - 1.0).abs() < 0.1,
            "Medium activation should be near neutral: {}",
            adjustment
        );
    }

    #[test]
    fn test_batch_retrieval_adjustments() {
        let mut detector = InterferenceDetector::new();

        // Set up different interference histories
        for i in 0..3 {
            detector.record_interference(
                "mem-with-history",
                &format!("interferer-{}", i),
                0.88,
                InterferenceType::Retroactive,
                0.05,
            );
        }

        let memories = vec![
            ("mem-with-history".to_string(), 0.9), // Has history, high activation
            ("mem-no-history".to_string(), 0.9),   // No history
        ];

        let adjustments = detector.batch_retrieval_adjustments(&memories);

        // Memory with history and high activation should be boosted
        assert!(adjustments.get("mem-with-history").unwrap() > &1.0);
        // Memory without history should be neutral
        assert_eq!(adjustments.get("mem-no-history").unwrap(), &1.0);
    }
}
