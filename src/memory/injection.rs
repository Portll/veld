//! Proactive Context Injection System
//!
//! Implements truly proactive memory injection - surfacing relevant memories
//! without explicit agent action. Based on multi-signal relevance scoring
//! with user-adaptive thresholds.
//!
//! # Enhanced Relevance Model (MEMO-1)
//!
//! ```text
//! R(m, c) = α·semantic + β·recency + γ·strength + δ·entity_overlap + ε·type_boost + ζ·file_match - η·suppression
//! ```
//!
//! Where:
//! - semantic: cosine similarity between memory and context embeddings
//! - recency: exponential decay based on memory age
//! - strength: Hebbian edge weight from knowledge graph
//! - entity_overlap: Jaccard similarity of entities between memory and context
//! - type_boost: Weight bonus based on memory type (Decision > Learning > Context)
//! - file_match: Boost when memory mentions files in current context
//! - suppression: Penalty for memories with negative feedback momentum
//!
//! # Feedback Loop
//!
//! The system learns from implicit feedback:
//! - Positive: injected memory referenced in next turn
//! - Negative: user indicates irrelevance
//! - Neutral: memory ignored (no adjustment)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::types::MemoryId;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Weights for composite relevance scoring
///
/// Enhanced with entity_overlap, type_boost, file_match, and suppression (MEMO-1).
/// New fields default to 0.0 for backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceWeights {
    /// Weight for semantic similarity (cosine distance)
    pub semantic: f32,
    /// Weight for recency (exponential decay)
    pub recency: f32,
    /// Weight for Hebbian strength from graph
    pub strength: f32,
    /// Weight for entity overlap between memory and context (MEMO-1)
    #[serde(default)]
    pub entity_overlap: f32,
    /// Weight for memory type boost (Decision > Learning > Context) (MEMO-1)
    #[serde(default)]
    pub type_boost: f32,
    /// Weight for file path matching (MEMO-1)
    #[serde(default)]
    pub file_match: f32,
    /// Weight for negative feedback suppression (MEMO-1)
    #[serde(default)]
    pub suppression: f32,
    /// Weight for episode coherence boost (SHO-temporal)
    /// Memories from the same episode as the query get boosted
    #[serde(default)]
    pub episode_coherence: f32,
    /// Weight for graph activation from spreading activation traversal
    /// Higher activation = stronger association in knowledge graph
    #[serde(default)]
    pub graph_activation: f32,
    /// Weight for linguistic score from query analysis
    /// Focal entity matches, modifier matches, etc.
    #[serde(default)]
    pub linguistic_score: f32,
}

impl Default for RelevanceWeights {
    fn default() -> Self {
        Self {
            semantic: 0.40,          // Primary signal - semantic similarity
            recency: 0.08,           // Recent memories get boost
            strength: 0.08,          // Hebbian edge weight (from graph)
            entity_overlap: 0.08,    // Entity Jaccard similarity
            type_boost: 0.06,        // Decision/Learning type boost
            file_match: 0.04,        // File path matching
            suppression: 0.02,       // Negative feedback penalty
            episode_coherence: 0.06, // Same-episode boost (prevents bleeding)
            graph_activation: 0.10,  // Spreading activation from graph traversal
            linguistic_score: 0.08,  // Query analysis (focal entities, modifiers)
        }
    }
}

impl RelevanceWeights {
    /// Legacy weights for backwards compatibility (original 3-signal model)
    pub fn legacy() -> Self {
        Self {
            semantic: 0.5,
            recency: 0.3,
            strength: 0.2,
            entity_overlap: 0.0,
            type_boost: 0.0,
            file_match: 0.0,
            suppression: 0.0,
            episode_coherence: 0.0,
            graph_activation: 0.0,
            linguistic_score: 0.0,
        }
    }
}

/// Configuration for proactive injection behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Minimum relevance score to trigger injection (0.0 - 1.0)
    pub min_relevance: f32,

    /// Maximum memories to inject per message
    pub max_per_message: usize,

    /// Cooldown in seconds before re-injecting same memory
    pub cooldown_seconds: u64,

    /// Weights for relevance score components
    pub weights: RelevanceWeights,

    /// Decay rate for recency calculation (λ in e^(-λt))
    /// Higher = faster decay. Default 0.01 means ~50% at 70 hours
    pub recency_decay_rate: f32,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            min_relevance: 0.50, // Raised from 0.35 - require stronger semantic match
            max_per_message: 3,
            cooldown_seconds: 180,
            weights: RelevanceWeights::default(),
            recency_decay_rate: 0.01,
        }
    }
}

impl InjectionConfig {
    /// Legacy config for backwards compatibility
    pub fn legacy() -> Self {
        Self {
            min_relevance: 0.70,
            max_per_message: 3,
            cooldown_seconds: 180,
            weights: RelevanceWeights::legacy(),
            recency_decay_rate: 0.01,
        }
    }
}

// Note: RelevanceInput and compute_relevance removed - using unified 5-layer pipeline

// =============================================================================
// INJECTION ENGINE
// =============================================================================

/// Candidate memory for injection with computed relevance
#[derive(Debug, Clone)]
pub struct InjectionCandidate {
    pub memory_id: MemoryId,
    pub relevance_score: f32,
}

/// Engine that decides which memories to inject
pub struct InjectionEngine {
    config: InjectionConfig,
    /// Tracks last injection time per memory for cooldown
    cooldowns: HashMap<MemoryId, Instant>,
}

impl InjectionEngine {
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            config,
            cooldowns: HashMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(InjectionConfig::default())
    }

    /// Check if a memory is on cooldown
    fn on_cooldown(&self, memory_id: &MemoryId) -> bool {
        if let Some(last) = self.cooldowns.get(memory_id) {
            last.elapsed().as_secs() < self.config.cooldown_seconds
        } else {
            false
        }
    }

    /// Select memories for injection from candidates
    ///
    /// Filters by:
    /// 1. Minimum relevance threshold
    /// 2. Cooldown (recently injected memories excluded)
    /// 3. Max count limit
    ///
    /// Returns memory IDs sorted by relevance (highest first)
    pub fn select_for_injection(
        &mut self,
        mut candidates: Vec<InjectionCandidate>,
    ) -> Vec<MemoryId> {
        // Sort by relevance descending
        candidates.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected: Vec<MemoryId> = candidates
            .into_iter()
            .filter(|c| {
                c.relevance_score >= self.config.min_relevance && !self.on_cooldown(&c.memory_id)
            })
            .take(self.config.max_per_message)
            .map(|c| c.memory_id)
            .collect();

        // Record injection time for cooldown
        let now = Instant::now();
        for id in &selected {
            self.cooldowns.insert(id.clone(), now);
        }

        selected
    }

    /// Clear expired cooldowns to prevent memory leak
    pub fn cleanup_cooldowns(&mut self) {
        let threshold = self.config.cooldown_seconds;
        self.cooldowns
            .retain(|_, last| last.elapsed().as_secs() < threshold * 2);
    }

    /// Get current configuration
    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: InjectionConfig) {
        self.config = config;
    }
}

// =============================================================================
// INJECTION TRACKING (for feedback loop)
// =============================================================================

/// Record of an injection for feedback tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionRecord {
    pub memory_id: MemoryId,
    pub injected_at: DateTime<Utc>,
    pub relevance_score: f32,
    pub context_signature: u64,
}

/// Tracks injections for feedback learning
#[derive(Debug, Default)]
pub struct InjectionTracker {
    /// Recent injections awaiting feedback
    pending: Vec<InjectionRecord>,
    /// Max pending records to keep
    max_pending: usize,
}

impl InjectionTracker {
    pub fn new(max_pending: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_pending,
        }
    }

    /// Record a new injection
    pub fn record_injection(
        &mut self,
        memory_id: MemoryId,
        relevance_score: f32,
        context_signature: u64,
    ) {
        let record = InjectionRecord {
            memory_id,
            injected_at: Utc::now(),
            relevance_score,
            context_signature,
        };

        self.pending.push(record);

        // Trim old records
        if self.pending.len() > self.max_pending {
            self.pending.remove(0);
        }
    }

    /// Get pending injections for feedback analysis
    pub fn pending_injections(&self) -> &[InjectionRecord] {
        &self.pending
    }

    /// Clear injections older than given duration
    pub fn clear_old(&mut self, max_age_seconds: i64) {
        let cutoff = Utc::now() - chrono::Duration::seconds(max_age_seconds);
        self.pending.retain(|r| r.injected_at > cutoff);
    }

    /// Remove specific injection after feedback processed
    pub fn mark_processed(&mut self, memory_id: &MemoryId) {
        self.pending.retain(|r| &r.memory_id != memory_id);
    }
}

// =============================================================================
// USER PROFILE (adaptive thresholds)
// =============================================================================

/// Feedback signal type for learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSignal {
    /// Memory was referenced/used - lower threshold
    Positive,
    /// Memory was explicitly rejected - raise threshold
    Negative,
    /// Memory was ignored - no change
    Neutral,
}

/// Per-user adaptive injection profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInjectionProfile {
    pub user_id: String,
    /// Effective threshold (starts at default, adapts over time)
    pub effective_threshold: f32,
    /// Count of positive signals received
    pub positive_signals: u32,
    /// Count of negative signals received
    pub negative_signals: u32,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl UserInjectionProfile {
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            effective_threshold: InjectionConfig::default().min_relevance,
            positive_signals: 0,
            negative_signals: 0,
            updated_at: Utc::now(),
        }
    }

    /// Adjust threshold based on feedback signal
    ///
    /// - Positive: lower threshold by 0.01 (min 0.50)
    /// - Negative: raise threshold by 0.02 (max 0.90)
    /// - Neutral: no change
    ///
    /// Asymmetric adjustment: we're more cautious about noise
    pub fn adjust(&mut self, signal: FeedbackSignal) {
        match signal {
            FeedbackSignal::Positive => {
                self.positive_signals += 1;
                self.effective_threshold = (self.effective_threshold - 0.01).max(0.50);
            }
            FeedbackSignal::Negative => {
                self.negative_signals += 1;
                self.effective_threshold = (self.effective_threshold + 0.02).min(0.90);
            }
            FeedbackSignal::Neutral => {}
        }
        self.updated_at = Utc::now();
    }

    /// Get signal ratio (positive / total)
    pub fn signal_ratio(&self) -> f32 {
        let total = self.positive_signals + self.negative_signals;
        if total == 0 {
            0.5 // No data yet
        } else {
            self.positive_signals as f32 / total as f32
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_injection_engine_filtering() {
        let mut engine = InjectionEngine::with_default_config();

        let candidates = vec![
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.85,
            },
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.45, // Below threshold (0.50)
            },
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.75,
            },
        ];

        let selected = engine.select_for_injection(candidates);

        assert_eq!(selected.len(), 2); // Only 0.85 and 0.75 pass threshold (0.50)
    }

    #[test]
    fn test_user_profile_adjustment() {
        let mut profile = UserInjectionProfile::new("test-user".to_string());

        assert_eq!(profile.effective_threshold, 0.50);

        profile.adjust(FeedbackSignal::Positive);
        assert!((profile.effective_threshold - 0.49).abs() < 0.01);

        profile.adjust(FeedbackSignal::Negative);
        assert!((profile.effective_threshold - 0.51).abs() < 0.01);

        // Many negatives should cap at 0.90
        for _ in 0..20 {
            profile.adjust(FeedbackSignal::Negative);
        }
        assert_eq!(profile.effective_threshold, 0.90);
    }
}
