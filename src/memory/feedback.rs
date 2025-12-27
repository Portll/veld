//! Implicit Feedback System for Memory Reinforcement
//!
//! Extracts feedback signals from agent behavior without explicit ratings.
//! Uses entity overlap, semantic similarity, and user corrections to
//! determine memory usefulness. Implements momentum-based updates with
//! type-dependent inertia to prevent noise from destabilizing useful memories.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::memory::types::{ExperienceType, MemoryId};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum number of recent signals to keep for trend detection
const MAX_RECENT_SIGNALS: usize = 20;

/// Maximum context fingerprints per memory
const MAX_CONTEXT_FINGERPRINTS: usize = 100;

/// Entity overlap thresholds
const OVERLAP_STRONG_THRESHOLD: f32 = 0.5;
const OVERLAP_WEAK_THRESHOLD: f32 = 0.2;

/// Signal value multipliers
const SIGNAL_STRONG_MULTIPLIER: f32 = 0.8;
const SIGNAL_WEAK_MULTIPLIER: f32 = 0.3;
const SIGNAL_NO_OVERLAP_PENALTY: f32 = -0.1;
const SIGNAL_NEGATIVE_KEYWORD_PENALTY: f32 = -0.5;

/// Stability adjustment rates
const STABILITY_INCREMENT: f32 = 0.05;
const STABILITY_DECREMENT_MULTIPLIER: f32 = 0.1;

/// Trend detection thresholds
const TREND_IMPROVING_THRESHOLD: f32 = 0.1;
const TREND_DECLINING_THRESHOLD: f32 = -0.1;

/// Negative keywords indicating correction/failure
const NEGATIVE_KEYWORDS: &[&str] = &[
    "no",
    "wrong",
    "incorrect",
    "not what i meant",
    "that's not right",
    "i already said",
    "i told you",
    "not correct",
    "that's wrong",
    "nope",
];

// =============================================================================
// SIGNAL TYPES
// =============================================================================

/// What triggered a feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalTrigger {
    /// Entity overlap between memory and agent response
    EntityOverlap { overlap_ratio: f32 },

    /// Semantic similarity between memory and response
    SemanticSimilarity { similarity: f32 },

    /// Negative keywords detected in user's followup
    NegativeKeywords { keywords: Vec<String> },

    /// User repeated the same question (retrieval failed)
    UserRepetition { similarity: f32 },

    /// Topic changed successfully (task completed)
    TopicChange,
}

/// A single feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRecord {
    /// When the signal was recorded
    pub timestamp: DateTime<Utc>,

    /// Signal value: -1.0 (misleading) to +1.0 (helpful)
    pub value: f32,

    /// Confidence in this signal (0.0 to 1.0)
    pub confidence: f32,

    /// What triggered this signal
    pub trigger: SignalTrigger,
}

impl SignalRecord {
    pub fn new(value: f32, confidence: f32, trigger: SignalTrigger) -> Self {
        Self {
            timestamp: Utc::now(),
            value: value.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            trigger,
        }
    }

    /// Create signal from entity overlap ratio
    pub fn from_entity_overlap(overlap_ratio: f32) -> Self {
        let (value, confidence) = if overlap_ratio >= OVERLAP_STRONG_THRESHOLD {
            (SIGNAL_STRONG_MULTIPLIER * overlap_ratio, 0.9)
        } else if overlap_ratio >= OVERLAP_WEAK_THRESHOLD {
            (SIGNAL_WEAK_MULTIPLIER * overlap_ratio, 0.6)
        } else {
            (SIGNAL_NO_OVERLAP_PENALTY, 0.4)
        };

        Self::new(
            value,
            confidence,
            SignalTrigger::EntityOverlap { overlap_ratio },
        )
    }

    /// Create signal from negative keyword detection
    pub fn from_negative_keywords(keywords: Vec<String>) -> Self {
        Self::new(
            SIGNAL_NEGATIVE_KEYWORD_PENALTY,
            0.95, // High confidence - explicit correction
            SignalTrigger::NegativeKeywords { keywords },
        )
    }
}

// =============================================================================
// TREND DETECTION
// =============================================================================

/// Trend direction for a memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    /// Memory is becoming more useful over time
    Improving,
    /// Memory usefulness is stable
    Stable,
    /// Memory is becoming less useful (possibly outdated)
    Declining,
    /// Not enough data to determine trend
    Insufficient,
}

impl Trend {
    /// Calculate trend from recent signals using linear regression
    pub fn from_signals(signals: &VecDeque<SignalRecord>) -> Self {
        if signals.len() < 3 {
            return Trend::Insufficient;
        }

        let n = signals.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, signal) in signals.iter().enumerate() {
            let x = i as f32;
            let y = signal.value;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        // Linear regression slope: (n*Σxy - Σx*Σy) / (n*Σxx - Σx²)
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return Trend::Stable;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        if slope > TREND_IMPROVING_THRESHOLD {
            Trend::Improving
        } else if slope < TREND_DECLINING_THRESHOLD {
            Trend::Declining
        } else {
            Trend::Stable
        }
    }
}

// =============================================================================
// CONTEXT FINGERPRINT
// =============================================================================

/// Fingerprint of a context for pattern detection
/// Tracks which contexts a memory was helpful vs misleading in
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFingerprint {
    /// Top entities in the context
    pub entities: Vec<String>,

    /// Compressed embedding signature (top 16 components)
    pub embedding_signature: [f32; 16],

    /// When this context occurred
    pub timestamp: DateTime<Utc>,

    /// Was the memory helpful in this context?
    pub was_helpful: bool,
}

impl ContextFingerprint {
    pub fn new(entities: Vec<String>, embedding: &[f32], was_helpful: bool) -> Self {
        // Compress embedding to 16 components by taking evenly spaced samples
        let mut signature = [0.0f32; 16];
        if !embedding.is_empty() {
            let step = embedding.len() / 16;
            for (i, sig) in signature.iter_mut().enumerate() {
                let idx = (i * step).min(embedding.len() - 1);
                *sig = embedding[idx];
            }
        }

        Self {
            entities,
            embedding_signature: signature,
            timestamp: Utc::now(),
            was_helpful,
        }
    }

    /// Calculate similarity to another fingerprint
    pub fn similarity(&self, other: &ContextFingerprint) -> f32 {
        // Entity Jaccard similarity
        let self_set: HashSet<_> = self.entities.iter().collect();
        let other_set: HashSet<_> = other.entities.iter().collect();
        let intersection = self_set.intersection(&other_set).count() as f32;
        let union = self_set.union(&other_set).count() as f32;
        let entity_sim = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };

        // Embedding cosine similarity
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..16 {
            dot += self.embedding_signature[i] * other.embedding_signature[i];
            norm_a += self.embedding_signature[i] * self.embedding_signature[i];
            norm_b += other.embedding_signature[i] * other.embedding_signature[i];
        }
        let embed_sim = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            0.0
        };

        // Weighted combination
        entity_sim * 0.6 + embed_sim * 0.4
    }
}

// =============================================================================
// FEEDBACK MOMENTUM
// =============================================================================

/// Tracks feedback history for a single memory
/// Implements momentum-based updates with type-dependent inertia
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackMomentum {
    /// Memory this momentum belongs to
    pub memory_id: MemoryId,

    /// Memory type (for inertia calculation)
    pub memory_type: ExperienceType,

    /// Exponential moving average of feedback signals
    /// Range: -1.0 (always misleading) to +1.0 (always helpful)
    pub ema: f32,

    /// How many feedback signals have we received?
    pub signal_count: u32,

    /// Stability score: how consistent is the feedback?
    /// High stability = resistant to change
    pub stability: f32,

    /// When did we first evaluate this memory?
    pub first_signal_at: Option<DateTime<Utc>>,

    /// When was the last signal?
    pub last_signal_at: Option<DateTime<Utc>>,

    /// Recent signals for trend detection
    pub recent_signals: VecDeque<SignalRecord>,

    /// Contexts where this memory was helpful
    pub helpful_contexts: Vec<ContextFingerprint>,

    /// Contexts where this memory was misleading
    pub misleading_contexts: Vec<ContextFingerprint>,
}

impl FeedbackMomentum {
    pub fn new(memory_id: MemoryId, memory_type: ExperienceType) -> Self {
        Self {
            memory_id,
            memory_type,
            ema: 0.0,
            signal_count: 0,
            stability: 0.5, // Start neutral
            first_signal_at: None,
            last_signal_at: None,
            recent_signals: VecDeque::with_capacity(MAX_RECENT_SIGNALS),
            helpful_contexts: Vec::new(),
            misleading_contexts: Vec::new(),
        }
    }

    /// Get base inertia for memory type
    /// Higher inertia = more resistant to change
    pub fn base_inertia(&self) -> f32 {
        match self.memory_type {
            ExperienceType::Learning => 0.95,
            ExperienceType::Decision => 0.90,
            ExperienceType::Pattern => 0.85,
            ExperienceType::Discovery => 0.75,
            ExperienceType::Context => 0.60,
            ExperienceType::Task => 0.50,
            ExperienceType::Observation => 0.40,
            ExperienceType::Conversation => 0.30,
            ExperienceType::Error => 0.20,
            // Others default to medium
            ExperienceType::CodeEdit => 0.50,
            ExperienceType::FileAccess => 0.40,
            ExperienceType::Search => 0.35,
            ExperienceType::Command => 0.35,
            ExperienceType::Intention => 0.60,
        }
    }

    /// Calculate age factor for inertia
    /// Older memories are more stable
    pub fn age_factor(&self) -> f32 {
        let age_days = self
            .first_signal_at
            .map(|first| {
                let duration = Utc::now() - first;
                duration.num_days() as f32
            })
            .unwrap_or(0.0);

        if age_days < 1.0 {
            0.8 // New, still malleable
        } else if age_days < 7.0 {
            0.9 // Consolidating
        } else if age_days < 30.0 {
            1.0 // Consolidated
        } else {
            1.1 // Deeply encoded
        }
    }

    /// Calculate history factor for inertia
    /// More evaluations = more confidence = more inertia
    pub fn history_factor(&self) -> f32 {
        match self.signal_count {
            0..=2 => 0.7,   // Not enough data
            3..=9 => 0.9,   // Some history
            10..=49 => 1.0, // Good history
            _ => 1.1,       // Very well tested
        }
    }

    /// Calculate stability factor for inertia
    /// Consistent history = resist change
    pub fn stability_factor(&self) -> f32 {
        // Map stability 0.0-1.0 to factor 0.8-1.2
        0.8 + (self.stability * 0.4)
    }

    /// Calculate effective inertia combining all factors
    pub fn effective_inertia(&self) -> f32 {
        let inertia =
            self.base_inertia() * self.age_factor() * self.history_factor() * self.stability_factor();

        // Clamp to valid range - never fully frozen, never fully fluid
        inertia.clamp(0.5, 0.99)
    }

    /// Calculate recency weight for a signal
    pub fn recency_weight(&self, signal_time: DateTime<Utc>) -> f32 {
        let time_since_last = self
            .last_signal_at
            .map(|last| signal_time - last)
            .unwrap_or_else(|| Duration::zero());

        if time_since_last < Duration::hours(1) {
            1.0
        } else if time_since_last < Duration::days(1) {
            0.9
        } else if time_since_last < Duration::days(7) {
            0.7
        } else {
            0.5
        }
    }

    /// Update momentum with a new signal
    pub fn update(&mut self, signal: SignalRecord) {
        let now = signal.timestamp;

        // Initialize first signal time if needed
        if self.first_signal_at.is_none() {
            self.first_signal_at = Some(now);
        }

        // Calculate effective inertia before update
        let effective_inertia = self.effective_inertia();
        let recency = self.recency_weight(now);

        // Alpha = how much new signal affects EMA
        // High inertia = low alpha = resistant to change
        let alpha = (1.0 - effective_inertia) * recency * signal.confidence;

        // Store old EMA for stability calculation
        let old_ema = self.ema;

        // Update EMA
        self.ema = old_ema * (1.0 - alpha) + signal.value * alpha;

        // Update stability
        let direction_matches =
            (signal.value > 0.0) == (old_ema > 0.0) || old_ema.abs() < 0.1;

        if direction_matches {
            // Consistent feedback: increase stability
            self.stability = (self.stability + STABILITY_INCREMENT).min(1.0);
        } else {
            // Contradictory feedback: decrease stability
            let contradiction_strength = (signal.value - old_ema).abs();
            self.stability =
                (self.stability - STABILITY_DECREMENT_MULTIPLIER * contradiction_strength).max(0.0);
        }

        // Record signal
        self.recent_signals.push_back(signal);
        if self.recent_signals.len() > MAX_RECENT_SIGNALS {
            self.recent_signals.pop_front();
        }

        self.signal_count += 1;
        self.last_signal_at = Some(now);
    }

    /// Get current trend
    pub fn trend(&self) -> Trend {
        Trend::from_signals(&self.recent_signals)
    }

    /// Add context fingerprint
    pub fn add_context(&mut self, fingerprint: ContextFingerprint) {
        let target = if fingerprint.was_helpful {
            &mut self.helpful_contexts
        } else {
            &mut self.misleading_contexts
        };

        target.push(fingerprint);

        // Trim to max size, keeping most recent
        if target.len() > MAX_CONTEXT_FINGERPRINTS {
            target.remove(0);
        }
    }

    /// Check if current context matches helpful pattern
    pub fn matches_helpful_pattern(&self, current: &ContextFingerprint) -> Option<f32> {
        self.helpful_contexts
            .iter()
            .map(|fp| fp.similarity(current))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Check if current context matches misleading pattern
    pub fn matches_misleading_pattern(&self, current: &ContextFingerprint) -> Option<f32> {
        self.misleading_contexts
            .iter()
            .map(|fp| fp.similarity(current))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// =============================================================================
// PENDING FEEDBACK
// =============================================================================

/// Information about a surfaced memory awaiting feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacedMemoryInfo {
    pub id: MemoryId,
    pub entities: HashSet<String>,
    pub content_preview: String,
    pub score: f32,
}

/// Pending feedback for a user - tracks what was surfaced, awaiting response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingFeedback {
    pub user_id: String,
    pub surfaced_at: DateTime<Utc>,
    pub surfaced_memories: Vec<SurfacedMemoryInfo>,
    pub context: String,
    pub context_embedding: Vec<f32>,
}

impl PendingFeedback {
    pub fn new(
        user_id: String,
        context: String,
        context_embedding: Vec<f32>,
        memories: Vec<SurfacedMemoryInfo>,
    ) -> Self {
        Self {
            user_id,
            surfaced_at: Utc::now(),
            surfaced_memories: memories,
            context,
            context_embedding,
        }
    }

    /// Check if this pending feedback has expired (older than 1 hour)
    pub fn is_expired(&self) -> bool {
        Utc::now() - self.surfaced_at > Duration::hours(1)
    }
}

// =============================================================================
// SIGNAL EXTRACTION
// =============================================================================

/// Extract entities from text using simple word extraction
/// TODO: Use NER model for better extraction
pub fn extract_entities_simple(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| word.len() > 2)
        .map(|s| s.to_string())
        .collect()
}

/// Calculate entity overlap between memory entities and response entities
pub fn calculate_entity_overlap(
    memory_entities: &HashSet<String>,
    response_entities: &HashSet<String>,
) -> f32 {
    if memory_entities.is_empty() {
        return 0.0;
    }

    let intersection = memory_entities.intersection(response_entities).count() as f32;
    intersection / memory_entities.len() as f32
}

/// Detect negative keywords in user's followup message
pub fn detect_negative_keywords(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    NEGATIVE_KEYWORDS
        .iter()
        .filter(|&&kw| lower.contains(kw))
        .map(|&s| s.to_string())
        .collect()
}

/// Process feedback for surfaced memories based on agent response
pub fn process_implicit_feedback(
    pending: &PendingFeedback,
    response_text: &str,
    user_followup: Option<&str>,
) -> Vec<(MemoryId, SignalRecord)> {
    let response_entities = extract_entities_simple(response_text);
    let mut signals = Vec::new();

    // Calculate entity overlap signals for each memory
    for memory in &pending.surfaced_memories {
        let overlap = calculate_entity_overlap(&memory.entities, &response_entities);
        let mut signal = SignalRecord::from_entity_overlap(overlap);

        // Apply negative keyword penalty if detected in followup
        if let Some(followup) = user_followup {
            let negative = detect_negative_keywords(followup);
            if !negative.is_empty() {
                signal.value += SIGNAL_NEGATIVE_KEYWORD_PENALTY;
                signal.value = signal.value.clamp(-1.0, 1.0);
            }
        }

        signals.push((memory.id.clone(), signal));
    }

    signals
}

// =============================================================================
// FEEDBACK STORE
// =============================================================================

/// In-memory store for feedback momentum and pending feedback
#[derive(Debug, Default)]
pub struct FeedbackStore {
    /// Momentum per memory: memory_id -> FeedbackMomentum
    momentum: HashMap<MemoryId, FeedbackMomentum>,

    /// Pending feedback per user: user_id -> PendingFeedback
    pending: HashMap<String, PendingFeedback>,
}

impl FeedbackStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create momentum for a memory
    pub fn get_or_create_momentum(
        &mut self,
        memory_id: MemoryId,
        memory_type: ExperienceType,
    ) -> &mut FeedbackMomentum {
        self.momentum
            .entry(memory_id.clone())
            .or_insert_with(|| FeedbackMomentum::new(memory_id, memory_type))
    }

    /// Get momentum for a memory (if exists)
    pub fn get_momentum(&self, memory_id: &MemoryId) -> Option<&FeedbackMomentum> {
        self.momentum.get(memory_id)
    }

    /// Set pending feedback for a user
    pub fn set_pending(&mut self, pending: PendingFeedback) {
        self.pending.insert(pending.user_id.clone(), pending);
    }

    /// Take pending feedback for a user (removes from store)
    pub fn take_pending(&mut self, user_id: &str) -> Option<PendingFeedback> {
        self.pending.remove(user_id)
    }

    /// Get pending feedback for a user (without removing)
    pub fn get_pending(&self, user_id: &str) -> Option<&PendingFeedback> {
        self.pending.get(user_id)
    }

    /// Clean up expired pending feedback
    pub fn cleanup_expired(&mut self) {
        self.pending.retain(|_, p| !p.is_expired());
    }

    /// Get statistics
    pub fn stats(&self) -> FeedbackStoreStats {
        FeedbackStoreStats {
            total_momentum_entries: self.momentum.len(),
            total_pending: self.pending.len(),
            avg_ema: if self.momentum.is_empty() {
                0.0
            } else {
                self.momentum.values().map(|m| m.ema).sum::<f32>() / self.momentum.len() as f32
            },
            avg_stability: if self.momentum.is_empty() {
                0.0
            } else {
                self.momentum.values().map(|m| m.stability).sum::<f32>()
                    / self.momentum.len() as f32
            },
        }
    }
}

/// Statistics about the feedback store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStoreStats {
    pub total_momentum_entries: usize,
    pub total_pending: usize,
    pub avg_ema: f32,
    pub avg_stability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_signal_from_entity_overlap() {
        // Strong overlap
        let signal = SignalRecord::from_entity_overlap(0.7);
        assert!(signal.value > 0.5);
        assert!(signal.confidence > 0.8);

        // Weak overlap
        let signal = SignalRecord::from_entity_overlap(0.3);
        assert!(signal.value > 0.0);
        assert!(signal.value < 0.5);

        // No overlap
        let signal = SignalRecord::from_entity_overlap(0.1);
        assert!(signal.value < 0.0);
    }

    #[test]
    fn test_momentum_inertia_by_type() {
        let learning = FeedbackMomentum::new(
            MemoryId(Uuid::new_v4()),
            ExperienceType::Learning,
        );
        let conversation = FeedbackMomentum::new(
            MemoryId(Uuid::new_v4()),
            ExperienceType::Conversation,
        );

        assert!(learning.base_inertia() > conversation.base_inertia());
        assert!(learning.base_inertia() >= 0.9);
        assert!(conversation.base_inertia() <= 0.4);
    }

    #[test]
    fn test_momentum_update_with_inertia() {
        let mut momentum = FeedbackMomentum::new(
            MemoryId(Uuid::new_v4()),
            ExperienceType::Learning, // High inertia
        );

        // Apply positive signal
        momentum.update(SignalRecord::new(
            1.0,
            1.0,
            SignalTrigger::EntityOverlap { overlap_ratio: 1.0 },
        ));

        // EMA should move slowly due to high inertia
        assert!(momentum.ema > 0.0);
        assert!(momentum.ema < 0.5); // Not too fast

        // Apply many positive signals
        for _ in 0..20 {
            momentum.update(SignalRecord::new(
                1.0,
                1.0,
                SignalTrigger::EntityOverlap { overlap_ratio: 1.0 },
            ));
        }

        // Now EMA should be higher
        assert!(momentum.ema > 0.5);
        // Stability should be high after consistent signals
        assert!(momentum.stability > 0.7);
    }

    #[test]
    fn test_trend_detection() {
        let mut signals = VecDeque::new();

        // Not enough data
        assert_eq!(Trend::from_signals(&signals), Trend::Insufficient);

        // Add improving signals (steeper slope > 0.1 threshold)
        for i in 0..10 {
            signals.push_back(SignalRecord::new(
                i as f32 * 0.15, // 0, 0.15, 0.3, ... gives slope ~0.15
                1.0,
                SignalTrigger::TopicChange,
            ));
        }
        assert_eq!(Trend::from_signals(&signals), Trend::Improving);

        // Add declining signals (steeper slope < -0.1 threshold)
        signals.clear();
        for i in (0..10).rev() {
            signals.push_back(SignalRecord::new(
                i as f32 * 0.15, // 1.35, 1.2, ... 0 gives slope ~-0.15
                1.0,
                SignalTrigger::TopicChange,
            ));
        }
        assert_eq!(Trend::from_signals(&signals), Trend::Declining);
    }

    #[test]
    fn test_entity_overlap() {
        let memory: HashSet<String> = ["rust", "async", "tokio"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let response: HashSet<String> = ["rust", "tokio", "spawn"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let overlap = calculate_entity_overlap(&memory, &response);
        assert!((overlap - 0.666).abs() < 0.01); // 2/3
    }

    #[test]
    fn test_negative_keyword_detection() {
        let text = "No, that's not what I meant";
        let keywords = detect_negative_keywords(text);
        assert!(keywords.contains(&"no".to_string()));
        assert!(keywords.contains(&"not what i meant".to_string()));
    }

    #[test]
    fn test_feedback_store_pending() {
        let mut store = FeedbackStore::new();
        let user_id = "test-user";

        // Initially no pending
        assert!(store.get_pending(user_id).is_none());

        // Set pending feedback
        let pending = PendingFeedback::new(
            user_id.to_string(),
            "test context".to_string(),
            vec![0.1; 384],
            vec![SurfacedMemoryInfo {
                id: MemoryId(Uuid::new_v4()),
                entities: ["rust", "memory"].iter().map(|s| s.to_string()).collect(),
                content_preview: "Test memory".to_string(),
                score: 0.8,
            }],
        );
        store.set_pending(pending);

        // Should have pending now
        assert!(store.get_pending(user_id).is_some());
        assert_eq!(store.get_pending(user_id).unwrap().surfaced_memories.len(), 1);

        // Take should remove it
        let taken = store.take_pending(user_id);
        assert!(taken.is_some());
        assert!(store.get_pending(user_id).is_none());
    }

    #[test]
    fn test_feedback_store_momentum() {
        let mut store = FeedbackStore::new();
        let memory_id = MemoryId(Uuid::new_v4());

        // Get or create momentum
        let momentum = store.get_or_create_momentum(
            memory_id.clone(),
            ExperienceType::Context,
        );
        assert_eq!(momentum.signal_count, 0);
        assert_eq!(momentum.ema, 0.0);

        // Update it
        momentum.update(SignalRecord::new(
            0.8,
            1.0,
            SignalTrigger::EntityOverlap { overlap_ratio: 0.8 },
        ));
        assert!(momentum.ema > 0.0);
        assert_eq!(momentum.signal_count, 1);

        // Get should return existing
        let momentum2 = store.get_momentum(&memory_id);
        assert!(momentum2.is_some());
        assert_eq!(momentum2.unwrap().signal_count, 1);
    }

    #[test]
    fn test_process_implicit_feedback_full() {
        let memory_id1 = MemoryId(Uuid::new_v4());
        let memory_id2 = MemoryId(Uuid::new_v4());

        let pending = PendingFeedback::new(
            "user1".to_string(),
            "How do I use async in Rust?".to_string(),
            vec![0.1; 384],
            vec![
                SurfacedMemoryInfo {
                    id: memory_id1.clone(),
                    entities: ["rust", "async", "tokio"].iter().map(|s| s.to_string()).collect(),
                    content_preview: "Rust async with tokio".to_string(),
                    score: 0.9,
                },
                SurfacedMemoryInfo {
                    id: memory_id2.clone(),
                    entities: ["python", "django"].iter().map(|s| s.to_string()).collect(),
                    content_preview: "Python Django web".to_string(),
                    score: 0.3,
                },
            ],
        );

        // Response that uses Rust async terminology
        let response = "To use async in Rust, you can use tokio runtime. Here is an example with async await.";
        let signals = process_implicit_feedback(&pending, response, None);

        assert_eq!(signals.len(), 2);

        // First memory should have positive signal (high entity overlap)
        let (id1, sig1) = &signals[0];
        assert_eq!(id1, &memory_id1);
        assert!(sig1.value > 0.0);

        // Second memory should have negative/low signal (no overlap)
        let (id2, sig2) = &signals[1];
        assert_eq!(id2, &memory_id2);
        assert!(sig2.value <= 0.0);
    }

    #[test]
    fn test_process_implicit_feedback_with_negative_keywords() {
        let memory_id = MemoryId(Uuid::new_v4());

        let pending = PendingFeedback::new(
            "user1".to_string(),
            "How do I use async?".to_string(),
            vec![0.1; 384],
            vec![SurfacedMemoryInfo {
                id: memory_id.clone(),
                entities: ["async", "code"].iter().map(|s| s.to_string()).collect(),
                content_preview: "Async code".to_string(),
                score: 0.9,
            }],
        );

        // Response uses entities
        let response = "Here is the async code pattern";

        // Process without negative keywords
        let signals1 = process_implicit_feedback(&pending, response, None);
        let value_without = signals1[0].1.value;

        // Process with negative keywords in followup
        let signals2 = process_implicit_feedback(&pending, response, Some("No, that is wrong!"));
        let value_with = signals2[0].1.value;

        // Negative keywords should decrease the signal
        assert!(value_with < value_without);
    }

    #[test]
    fn test_context_fingerprint_similarity() {
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let fp1 = ContextFingerprint::new(
            vec!["rust".to_string(), "memory".to_string()],
            &embedding,
            true,
        );
        let fp2 = ContextFingerprint::new(
            vec!["rust".to_string(), "async".to_string()],
            &embedding,
            false,
        );
        let different_embedding: Vec<f32> = (0..384).map(|i| 1.0 - (i as f32) * 0.01).collect();
        let fp3 = ContextFingerprint::new(
            vec!["python".to_string(), "django".to_string()],
            &different_embedding,
            true,
        );

        // fp1 and fp2 share "rust" entity and same embedding
        let sim12 = fp1.similarity(&fp2);
        // fp1 and fp3 have no entity overlap and different embedding
        let sim13 = fp1.similarity(&fp3);

        assert!(sim12 > sim13);
    }

    #[test]
    fn test_feedback_store_stats() {
        let mut store = FeedbackStore::new();

        // Empty stats
        let stats = store.stats();
        assert_eq!(stats.total_momentum_entries, 0);
        assert_eq!(stats.total_pending, 0);

        // Add some momentum entries
        for i in 0..5 {
            let mut momentum = FeedbackMomentum::new(
                MemoryId(Uuid::new_v4()),
                ExperienceType::Context,
            );
            momentum.ema = i as f32 * 0.2; // 0, 0.2, 0.4, 0.6, 0.8
            store.momentum.insert(momentum.memory_id.clone(), momentum);
        }

        let stats = store.stats();
        assert_eq!(stats.total_momentum_entries, 5);
        assert!((stats.avg_ema - 0.4).abs() < 0.01); // (0+0.2+0.4+0.6+0.8)/5 = 0.4
    }
}
