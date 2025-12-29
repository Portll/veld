//! Proactive Context Injection System
//!
//! Implements truly proactive memory injection - surfacing relevant memories
//! without explicit agent action. Based on multi-signal relevance scoring
//! with user-adaptive thresholds.
//!
//! # Relevance Model
//!
//! ```text
//! R(m, c) = α·semantic(m, c) + β·recency(m) + γ·strength(m)
//! ```
//!
//! Where:
//! - semantic: cosine similarity between memory and context embeddings
//! - recency: exponential decay based on memory age
//! - strength: Hebbian edge weight from knowledge graph
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceWeights {
    /// Weight for semantic similarity (cosine distance)
    pub semantic: f32,
    /// Weight for recency (exponential decay)
    pub recency: f32,
    /// Weight for Hebbian strength from graph
    pub strength: f32,
}

impl Default for RelevanceWeights {
    fn default() -> Self {
        Self {
            semantic: 0.5,
            recency: 0.3,
            strength: 0.2,
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
            min_relevance: 0.70,
            max_per_message: 3,
            cooldown_seconds: 180,
            weights: RelevanceWeights::default(),
            recency_decay_rate: 0.01,
        }
    }
}

// =============================================================================
// RELEVANCE SCORING
// =============================================================================

/// Input data for computing relevance of a memory
#[derive(Debug, Clone)]
pub struct RelevanceInput {
    /// Memory's embedding vector
    pub memory_embedding: Vec<f32>,
    /// Memory creation timestamp
    pub created_at: DateTime<Utc>,
    /// Hebbian strength from knowledge graph (0.0 - 1.0)
    pub hebbian_strength: f32,
}

/// Compute composite relevance score for a memory
///
/// # Arguments
/// * `input` - Memory data for scoring
/// * `context_embedding` - Current context embedding
/// * `now` - Current timestamp
/// * `config` - Injection configuration with weights
///
/// # Returns
/// Relevance score in range [0.0, 1.0]
pub fn compute_relevance(
    input: &RelevanceInput,
    context_embedding: &[f32],
    now: DateTime<Utc>,
    config: &InjectionConfig,
) -> f32 {
    let semantic = cosine_similarity(&input.memory_embedding, context_embedding);

    let hours_old = (now - input.created_at).num_hours().max(0) as f32;
    let recency = (-config.recency_decay_rate * hours_old).exp();

    let strength = input.hebbian_strength;

    let w = &config.weights;
    let score = w.semantic * semantic + w.recency * recency + w.strength * strength;

    // Clamp to [0, 1]
    score.clamp(0.0, 1.0)
}

/// Cosine similarity between two vectors (public for use by streaming module)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

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
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_compute_relevance() {
        let config = InjectionConfig::default();
        let now = Utc::now();

        let input = RelevanceInput {
            memory_embedding: vec![1.0, 0.0, 0.0],
            created_at: now,
            hebbian_strength: 0.8,
        };

        let context = vec![1.0, 0.0, 0.0]; // Perfect match
        let score = compute_relevance(&input, &context, now, &config);

        // semantic=1.0, recency=1.0 (just created), strength=0.8
        // 0.5*1.0 + 0.3*1.0 + 0.2*0.8 = 0.5 + 0.3 + 0.16 = 0.96
        assert!(score > 0.9);
    }

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
                relevance_score: 0.60, // Below threshold
            },
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.75,
            },
        ];

        let selected = engine.select_for_injection(candidates);

        assert_eq!(selected.len(), 2); // Only 0.85 and 0.75 pass threshold
    }

    #[test]
    fn test_user_profile_adjustment() {
        let mut profile = UserInjectionProfile::new("test-user".to_string());

        assert_eq!(profile.effective_threshold, 0.70);

        profile.adjust(FeedbackSignal::Positive);
        assert_eq!(profile.effective_threshold, 0.69);

        profile.adjust(FeedbackSignal::Negative);
        assert_eq!(profile.effective_threshold, 0.71);

        // Many negatives should cap at 0.90
        for _ in 0..20 {
            profile.adjust(FeedbackSignal::Negative);
        }
        assert_eq!(profile.effective_threshold, 0.90);
    }
}
