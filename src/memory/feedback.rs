//! Implicit Feedback System for Memory Reinforcement
//!
//! Extracts feedback signals from agent behavior without explicit ratings.
//! Uses entity overlap, semantic similarity, and user corrections to
//! determine memory usefulness. Implements momentum-based updates with
//! type-dependent inertia to prevent noise from destabilizing useful memories.

use chrono::{DateTime, Duration, Timelike, Utc};
#[cfg(feature = "rocksdb")]
use rocksdb::{ColumnFamilyDescriptor, Options, DB};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::memory::types::{ExperienceType, MemoryId};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Column family name for feedback data in the shared RocksDB instance
#[cfg_attr(not(feature = "rocksdb"), allow(dead_code))]
pub(crate) const CF_FEEDBACK: &str = "feedback";

/// Maximum number of recent signals to keep for trend detection
const MAX_RECENT_SIGNALS: usize = 20;

/// Maximum context fingerprints per memory
const MAX_CONTEXT_FINGERPRINTS: usize = 100;

/// Entity overlap thresholds
/// FBK-4: Lowered thresholds so weak signals (0.06-0.15 range) actually affect learning
const OVERLAP_STRONG_THRESHOLD: f32 = 0.4;
const OVERLAP_WEAK_THRESHOLD: f32 = 0.1;

/// Semantic similarity thresholds
/// FBK-4: Lowered to catch more meaningful signals
const SEMANTIC_STRONG_THRESHOLD: f32 = 0.6;
const SEMANTIC_WEAK_THRESHOLD: f32 = 0.3;

/// Signal value multipliers (ACT-R inspired)
const SIGNAL_STRONG_MULTIPLIER: f32 = 0.8;
const SIGNAL_WEAK_MULTIPLIER: f32 = 0.3;
const SIGNAL_NO_OVERLAP_PENALTY: f32 = -0.2; // Strengthened: was -0.1 (FBK-3)
const SIGNAL_NEGATIVE_KEYWORD_PENALTY: f32 = -0.5;

/// Action-based signals (FBK-1, FBK-2)
const SIGNAL_REPETITION_PENALTY: f32 = -0.4; // User asked again = memories failed
const SIGNAL_TOPIC_CHANGE_BOOST: f32 = 0.2; // User moved on = task might be complete
const SIGNAL_IGNORED_PENALTY: f32 = -0.2; // Memory shown but completely unused

/// Weights for combining entity and semantic signals
const ENTITY_WEIGHT: f32 = 0.4;
const SEMANTIC_WEIGHT: f32 = 0.6;

/// Tool-usage attribution constants
/// Minimum Jaccard token overlap between memory content and tool action inputs.
/// Lower than entity overlap (0.1) because tool inputs are short with high
/// information density per token (file paths, commands, coordinates).
const TOOL_USAGE_MIN_OVERLAP: f32 = 0.08;
/// Above this Jaccard overlap, signal gets high confidence (0.9).
const TOOL_USAGE_STRONG_THRESHOLD: f32 = 0.25;
/// Positive signal when tool action matches memory and succeeds.
/// Stronger than SIGNAL_STRONG_MULTIPLIER (0.8) because tool usage is
/// a concrete behavioral signal, not just word overlap.
const TOOL_USAGE_SUCCESS_SIGNAL: f32 = 0.7;
/// Negative signal when tool action matches memory but fails.
const TOOL_USAGE_FAILURE_SIGNAL: f32 = -0.4;
/// Blend weight for tool signal vs entity+semantic.
/// When a tool action matches, 35% of final signal comes from tool attribution.
const TOOL_USAGE_WEIGHT: f32 = 0.35;

/// Information-theoretic attribution constants
/// Uses vector projection to factor out query-shared information from memory↔response
/// similarity, isolating the memory's unique causal contribution.
/// Minimum residual-cosine score to count as a positive signal.
/// Below this, memory's unique content wasn't reflected in the response.
const INFO_ATTRIBUTION_MIN: f32 = 0.05;
/// Strong attribution — memory clearly influenced the response beyond query overlap.
const INFO_ATTRIBUTION_STRONG: f32 = 0.25;
/// Signal value for strong attribution (scaled by score).
const INFO_ATTRIBUTION_STRONG_SIGNAL: f32 = 0.85;
/// Signal value for weak-but-present attribution (scaled by score).
const INFO_ATTRIBUTION_WEAK_SIGNAL: f32 = 0.3;
/// Penalty when memory was surfaced but its unique content is absent from response.
const INFO_ATTRIBUTION_NO_SIGNAL: f32 = -0.15;
/// Weight in combined signal when info attribution is available.
/// Takes 35% from the combination, reducing semantic from 60% to 35%.
const INFO_ATTRIBUTION_WEIGHT: f32 = 0.35;
/// Adjusted entity weight when info attribution available (was 0.4).
const ENTITY_WEIGHT_WITH_INFO: f32 = 0.30;
/// Adjusted semantic weight when info attribution available (was 0.6).
const SEMANTIC_WEIGHT_WITH_INFO: f32 = 0.35;

/// Stability adjustment rates
const STABILITY_INCREMENT: f32 = 0.05;
const STABILITY_DECREMENT_MULTIPLIER: f32 = 0.1;

/// Trend detection thresholds
const TREND_IMPROVING_THRESHOLD: f32 = 0.1;
const TREND_DECLINING_THRESHOLD: f32 = -0.1;

/// Time decay constants for momentum (AUD-6)
/// Momentum should decay towards 0 when not reinforced
const DECAY_HALF_LIFE_DAYS: f32 = 14.0; // Half-life of 14 days

/// Negative keywords indicating correction/failure
/// Multi-word phrases checked first (contains match on lowercased text)
const NEGATIVE_KEYWORDS: &[&str] = &[
    // Direct negation / correction
    "wrong",
    "incorrect",
    "not correct",
    "nope",
    // Frustration / repetition
    "not what i meant",
    "that's not right",
    "that's wrong",
    "i already said",
    "i told you",
    "i already told",
    "already mentioned",
    // Irrelevance / unhelpfulness
    "not helpful",
    "not relevant",
    "not useful",
    "irrelevant",
    "useless",
    "doesn't help",
    "didn't help",
    "not related",
    // Failure / broken
    "doesn't work",
    "didn't work",
    "broken",
    "still broken",
    "that failed",
    // Explicit rejection
    "forget that",
    "ignore that",
    "disregard",
    "stop suggesting",
    "don't show",
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
    /// Action: user asked again → memories didn't help
    UserRepetition { similarity: f32 },

    /// Topic changed successfully (task completed)
    /// Action: user moved on → memories may have helped
    TopicChange { similarity: f32 },

    /// Memory was surfaced but completely ignored
    /// Action: response has no relation to memory
    Ignored { overlap_ratio: f32 },

    /// FBK-8: Entity flow tracking
    /// Measures how response builds on memory entities
    /// - derived_ratio: proportion of response entities that came from memory
    /// - novel_ratio: proportion of response entities that are new
    EntityFlow {
        derived_ratio: f32,
        novel_ratio: f32,
        memory_entities_used: usize,
        response_entities_total: usize,
    },

    /// Tool/actuator action matched surfaced memory content.
    /// The agent performed a concrete action aligned with the memory's guidance.
    /// Covers both Claude Code tools (Read, Edit, Bash) and robot actuators
    /// (navigate, grasp, sense).
    ToolUsage {
        /// Jaccard token overlap between memory content and tool inputs (0.0-1.0)
        content_overlap: f32,
        /// Tool or actuator name
        tool_name: String,
        /// Whether the tool action succeeded
        success: bool,
    },

    /// Information-theoretic attribution: measures unique information
    /// the memory contributed beyond what the query already provided.
    /// Uses vector projection to factor out query-shared components.
    InformationAttribution {
        /// Cosine similarity between memory and response residuals (query projected out)
        attribution_score: f32,
        /// Raw cosine similarity before projection (for comparison/diagnostics)
        raw_similarity: f32,
    },
}

/// A tool or actuator action performed between feedback cycles.
///
/// Unified abstraction covering Claude Code tools (Read, Edit, Write, Bash)
/// and robot actuator commands (navigate, grasp, sense). The feedback system
/// matches action inputs/outputs against surfaced memory content to determine
/// whether a memory influenced a concrete action.
///
/// For Claude Code: collected by hooks between proactive_context calls.
/// For robotics: constructed from Experience fields when remember() follows recall().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAction {
    /// Tool or actuator name: "Edit", "Bash", "navigate", "grasp"
    pub tool_name: String,

    /// Key-value inputs.
    /// Claude Code: {"file_path": "/src/main.rs", "command": "cargo build"}.
    /// Robotics: {"target": "waypoint_7", "speed": "0.5"}.
    #[serde(default)]
    pub inputs: HashMap<String, String>,

    /// Whether the action succeeded.
    /// Claude Code: true unless tool_output contains error markers.
    /// Robotics: derived from outcome_type (success/partial = true).
    pub success: bool,

    /// First 200 chars of output. Used for content matching.
    #[serde(default)]
    pub output_snippet: Option<String>,

    /// Reward signal (robotics only, -1.0 to 1.0). None for Claude Code tools.
    #[serde(default)]
    pub reward: Option<f32>,
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
        let inertia = self.base_inertia()
            * self.age_factor()
            * self.history_factor()
            * self.stability_factor();

        // Clamp to valid range - never fully frozen, never fully fluid
        inertia.clamp(0.5, 0.99)
    }

    /// Calculate recency weight for a signal
    pub fn recency_weight(&self, signal_time: DateTime<Utc>) -> f32 {
        let time_since_last = self
            .last_signal_at
            .map(|last| signal_time - last)
            .unwrap_or_else(Duration::zero);

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

        // FIX-04: Bidirectional feedback acceleration
        // When explicit strong negative signal detected (high confidence + strong negative value),
        // bypass gradual EMA accumulation and apply immediate momentum override.
        // This cuts unlearning latency from 5-15 cycles to 1-2 cycles for explicit rejections
        // like "wrong", "not helpful", "that's incorrect" (detected as NegativeKeywords trigger).
        if signal.confidence > 0.8 && signal.value < -0.3 {
            // Direct override: jump EMA to signal value (no inertia dampening)
            self.ema = signal.value;
            // Crash stability to signal strong contradiction
            self.stability = (self.stability * 0.3).max(0.0);
        } else {
            // Standard EMA update for all other signals
            self.ema = old_ema * (1.0 - alpha) + signal.value * alpha;
        }

        // Update stability
        let direction_matches =
            (signal.value > 0.0) == (old_ema > 0.0) || old_ema.abs() < f32::EPSILON;

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
            .max_by(|a, b| a.total_cmp(b))
    }

    /// Check if current context matches misleading pattern
    pub fn matches_misleading_pattern(&self, current: &ContextFingerprint) -> Option<f32> {
        self.misleading_contexts
            .iter()
            .map(|fp| fp.similarity(current))
            .max_by(|a, b| a.total_cmp(b))
    }

    /// Apply time-based decay to momentum (AUD-6)
    /// Returns the decayed EMA value without mutating the struct.
    /// Momentum decays towards 0 when not reinforced by feedback.
    pub fn ema_with_decay(&self) -> f32 {
        let days_since_last = self
            .last_signal_at
            .map(|last| {
                let duration = Utc::now() - last;
                duration.num_hours() as f32 / 24.0
            })
            .unwrap_or(0.0);

        if days_since_last < 0.1 {
            // Very recent signal, no decay
            return self.ema;
        }

        // Exponential decay with half-life
        // decay_factor = 0.5^(days / half_life)
        let decay_factor = 0.5_f32.powf(days_since_last / DECAY_HALF_LIFE_DAYS);

        // Decay towards 0
        self.ema * decay_factor
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
    /// Memory embedding for semantic similarity feedback
    #[serde(default)]
    pub embedding: Vec<f32>,
}

/// Pending feedback for a user - tracks what was surfaced, awaiting response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingFeedback {
    pub user_id: String,
    pub surfaced_at: DateTime<Utc>,
    pub surfaced_memories: Vec<SurfacedMemoryInfo>,
    pub context: String,
    pub context_embedding: Vec<f32>,
    /// Tool/actuator actions performed after memories were surfaced.
    /// Claude Code: collected by hooks between proactive_context calls.
    /// Robotics: populated from action-outcome Experience fields.
    #[serde(default)]
    pub tool_actions: Vec<ToolAction>,
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
            tool_actions: Vec::new(),
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

/// Extract entities from text using heuristic word extraction.
/// Splits on non-alphanumeric boundaries and returns words longer than 2 characters.
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

/// Calculate cosine similarity between two embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Compute information-theoretic attribution using vector projection.
///
/// Projects out the query component from both memory and response embeddings,
/// then measures cosine similarity of the residuals. This isolates what the
/// memory uniquely contributed vs what was already in the query.
///
/// Returns `(attribution_score, raw_similarity)` or None if embeddings are
/// empty or have mismatched dimensions.
fn compute_information_attribution(
    query_emb: &[f32],
    memory_emb: &[f32],
    response_emb: &[f32],
) -> Option<(f32, f32)> {
    if query_emb.is_empty()
        || memory_emb.len() != query_emb.len()
        || response_emb.len() != query_emb.len()
    {
        return None;
    }

    // query · query (projection denominator)
    let query_dot_query: f32 = query_emb.iter().map(|x| x * x).sum();
    if query_dot_query < 1e-10 {
        return None; // degenerate query embedding
    }

    // Scalar projections onto query direction
    let mem_dot_query: f32 = memory_emb.iter().zip(query_emb).map(|(m, q)| m * q).sum();
    let resp_dot_query: f32 = response_emb.iter().zip(query_emb).map(|(r, q)| r * q).sum();

    let mem_proj_scale = mem_dot_query / query_dot_query;
    let resp_proj_scale = resp_dot_query / query_dot_query;

    // Residuals: original vector minus its projection onto query
    let mem_residual: Vec<f32> = memory_emb
        .iter()
        .zip(query_emb)
        .map(|(m, q)| m - mem_proj_scale * q)
        .collect();
    let resp_residual: Vec<f32> = response_emb
        .iter()
        .zip(query_emb)
        .map(|(r, q)| r - resp_proj_scale * q)
        .collect();

    let attribution = cosine_similarity(&mem_residual, &resp_residual).max(0.0);
    let raw_similarity = cosine_similarity(memory_emb, response_emb);

    Some((attribution, raw_similarity))
}

/// Create signal from semantic similarity
fn signal_from_semantic_similarity(similarity: f32) -> (f32, f32) {
    if similarity >= SEMANTIC_STRONG_THRESHOLD {
        (SIGNAL_STRONG_MULTIPLIER * similarity, 0.9)
    } else if similarity >= SEMANTIC_WEAK_THRESHOLD {
        (SIGNAL_WEAK_MULTIPLIER * similarity, 0.6)
    } else {
        (SIGNAL_NO_OVERLAP_PENALTY * 0.5, 0.3) // Lighter penalty for semantic
    }
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

/// FBK-8: Calculate entity flow between memory and response
///
/// Tracks how the response builds on memory entities:
/// - derived_ratio: How many response entities came from the memory (0.0 to 1.0)
/// - novel_ratio: How many response entities are new/not from memory (0.0 to 1.0)
///
/// High derived_ratio = response uses memory knowledge = positive signal
/// High novel_ratio with low derived = memory might not have been relevant
pub fn calculate_entity_flow(
    memory_entities: &HashSet<String>,
    response_entities: &HashSet<String>,
) -> (f32, f32, usize, usize) {
    if response_entities.is_empty() {
        return (0.0, 0.0, 0, 0);
    }

    // Count how many response entities came from the memory
    let derived: HashSet<_> = response_entities
        .intersection(memory_entities)
        .cloned()
        .collect();
    let derived_count = derived.len();

    // Count novel entities (in response but not in memory)
    let novel_count = response_entities.len() - derived_count;

    let derived_ratio = derived_count as f32 / response_entities.len() as f32;
    let novel_ratio = novel_count as f32 / response_entities.len() as f32;

    (
        derived_ratio,
        novel_ratio,
        derived_count,
        response_entities.len(),
    )
}

/// FBK-8: Create signal from entity flow analysis
pub fn signal_from_entity_flow(
    derived_ratio: f32,
    novel_ratio: f32,
    memory_entities_used: usize,
    response_entities_total: usize,
) -> SignalRecord {
    // Signal value based on how much the response builds on memory
    // High derived ratio = memory was useful
    // Low derived ratio with high novel = memory might be irrelevant
    let value = if derived_ratio >= 0.5 {
        // Response heavily uses memory entities - strong positive
        0.6 + (derived_ratio - 0.5) * 0.4
    } else if derived_ratio >= 0.2 {
        // Response somewhat uses memory entities - weak positive
        derived_ratio * 1.5
    } else if novel_ratio >= 0.8 {
        // Response mostly novel, memory barely used - slight negative
        -0.1
    } else {
        // Mixed - neutral
        0.0
    };

    let confidence = if response_entities_total >= 3 {
        0.8 // Good sample size
    } else {
        0.5 // Small sample, lower confidence
    };

    SignalRecord::new(
        value,
        confidence,
        SignalTrigger::EntityFlow {
            derived_ratio,
            novel_ratio,
            memory_entities_used,
            response_entities_total,
        },
    )
}

/// Process feedback for surfaced memories based on agent response
/// Uses both entity overlap and semantic similarity for more accurate signals
pub fn process_implicit_feedback(
    pending: &PendingFeedback,
    response_text: &str,
    user_followup: Option<&str>,
) -> Vec<(MemoryId, SignalRecord)> {
    // For backwards compatibility, call enhanced version with no response embedding
    process_implicit_feedback_with_semantics(pending, response_text, user_followup, None)
}

/// Enhanced feedback processing using both entity overlap and semantic similarity
///
/// When response_embedding is provided, combines entity overlap (40%) with
/// semantic similarity (60%) for a more robust feedback signal. This helps
/// detect when a memory was genuinely useful vs just sharing some words.
pub fn process_implicit_feedback_with_semantics(
    pending: &PendingFeedback,
    response_text: &str,
    user_followup: Option<&str>,
    response_embedding: Option<&[f32]>,
) -> Vec<(MemoryId, SignalRecord)> {
    let response_entities = extract_entities_simple(response_text);
    let mut signals = Vec::new();

    // Calculate combined signals for each memory
    for memory in &pending.surfaced_memories {
        // Entity overlap signal
        let entity_overlap = calculate_entity_overlap(&memory.entities, &response_entities);
        let (entity_value, entity_conf) = if entity_overlap >= OVERLAP_STRONG_THRESHOLD {
            (SIGNAL_STRONG_MULTIPLIER * entity_overlap, 0.9)
        } else if entity_overlap >= OVERLAP_WEAK_THRESHOLD {
            (SIGNAL_WEAK_MULTIPLIER * entity_overlap, 0.6)
        } else {
            (SIGNAL_NO_OVERLAP_PENALTY, 0.4)
        };

        // Semantic similarity signal (if embeddings available)
        let (semantic_value, semantic_conf, has_semantic) =
            if let Some(resp_emb) = response_embedding {
                if !memory.embedding.is_empty() {
                    let similarity = cosine_similarity(&memory.embedding, resp_emb);
                    let (val, conf) = signal_from_semantic_similarity(similarity);
                    (val, conf, true)
                } else {
                    (0.0, 0.0, false)
                }
            } else {
                (0.0, 0.0, false)
            };

        // Combine signals with weights
        // When info-theoretic attribution is available (all 3 embeddings present),
        // use 3-signal combination that isolates the memory's unique causal contribution.
        // Otherwise fall back to entity+semantic weighted sum.
        let (combined_value, combined_confidence, trigger) = if has_semantic {
            if let Some((attr_score, raw_sim)) = response_embedding.and_then(|resp_emb| {
                compute_information_attribution(
                    &pending.context_embedding,
                    &memory.embedding,
                    resp_emb,
                )
            }) {
                // Three-signal combination: entity + semantic + info attribution
                let (info_value, info_conf) = if attr_score >= INFO_ATTRIBUTION_STRONG {
                    (INFO_ATTRIBUTION_STRONG_SIGNAL * attr_score.min(1.0), 0.9)
                } else if attr_score >= INFO_ATTRIBUTION_MIN {
                    (INFO_ATTRIBUTION_WEAK_SIGNAL * attr_score, 0.65)
                } else {
                    (INFO_ATTRIBUTION_NO_SIGNAL, 0.5)
                };

                let value = (ENTITY_WEIGHT_WITH_INFO * entity_value)
                    + (SEMANTIC_WEIGHT_WITH_INFO * semantic_value)
                    + (INFO_ATTRIBUTION_WEIGHT * info_value);
                let confidence = (ENTITY_WEIGHT_WITH_INFO * entity_conf)
                    + (SEMANTIC_WEIGHT_WITH_INFO * semantic_conf)
                    + (INFO_ATTRIBUTION_WEIGHT * info_conf);

                (
                    value,
                    confidence,
                    SignalTrigger::InformationAttribution {
                        attribution_score: attr_score,
                        raw_similarity: raw_sim,
                    },
                )
            } else {
                // Fallback: entity + semantic only (context_embedding empty or dim mismatch)
                let value = (ENTITY_WEIGHT * entity_value) + (SEMANTIC_WEIGHT * semantic_value);
                let confidence = (ENTITY_WEIGHT * entity_conf) + (SEMANTIC_WEIGHT * semantic_conf);
                let similarity = response_embedding
                    .map(|resp_emb| cosine_similarity(&memory.embedding, resp_emb))
                    .unwrap_or(0.0);
                (
                    value,
                    confidence,
                    SignalTrigger::SemanticSimilarity { similarity },
                )
            }
        } else {
            // No embeddings at all — entity-only
            (
                entity_value,
                entity_conf,
                SignalTrigger::EntityOverlap {
                    overlap_ratio: entity_overlap,
                },
            )
        };

        // Tool-usage attribution: blend if tool actions matched this memory
        let (combined_value, combined_confidence, trigger) =
            if let Some((tool_val, tool_conf, tool_name, tool_overlap)) =
                compute_tool_usage_signal(memory, &pending.tool_actions)
            {
                let blended_value =
                    (TOOL_USAGE_WEIGHT * tool_val) + ((1.0 - TOOL_USAGE_WEIGHT) * combined_value);
                let blended_conf = tool_conf.max(combined_confidence);
                (
                    blended_value,
                    blended_conf,
                    SignalTrigger::ToolUsage {
                        content_overlap: tool_overlap,
                        tool_name,
                        success: tool_val > 0.0,
                    },
                )
            } else {
                (combined_value, combined_confidence, trigger)
            };

        let mut signal = SignalRecord::new(combined_value, combined_confidence, trigger);

        // Apply negative keyword penalty if detected in followup
        if let Some(followup) = user_followup {
            let negative = detect_negative_keywords(followup);
            if !negative.is_empty() {
                signal.value += SIGNAL_NEGATIVE_KEYWORD_PENALTY;
                signal.value = signal.value.clamp(-1.0, 1.0);
                signal.confidence = 0.95; // High confidence on explicit correction
            }
        }

        signals.push((memory.id.clone(), signal));
    }

    signals
}

/// Compute tool-usage attribution signal for a single memory.
///
/// Checks whether any tool action's inputs or output contain content
/// from the surfaced memory. Uses normalized Jaccard token overlap
/// because tool inputs are short and keyword-heavy (file paths, commands,
/// coordinates, waypoints).
///
/// Returns `(signal_value, confidence, tool_name, overlap)` for the best
/// matching tool action, or None if no match above threshold.
pub fn compute_tool_usage_signal(
    memory: &SurfacedMemoryInfo,
    tool_actions: &[ToolAction],
) -> Option<(f32, f32, String, f32)> {
    if tool_actions.is_empty() {
        return None;
    }

    let memory_tokens: HashSet<&str> = memory
        .content_preview
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.' && c != '/')
        .filter(|w| w.len() >= 3)
        .collect();

    if memory_tokens.is_empty() {
        return None;
    }

    let mut best_overlap = 0.0f32;
    let mut best_tool = String::new();
    let mut best_success = false;
    let mut best_reward: Option<f32> = None;

    for action in tool_actions {
        let mut action_text = String::new();
        for value in action.inputs.values() {
            action_text.push(' ');
            action_text.push_str(value);
        }
        if let Some(ref snippet) = action.output_snippet {
            action_text.push(' ');
            action_text.push_str(snippet);
        }

        let action_tokens: HashSet<&str> = action_text
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.' && c != '/')
            .filter(|w| w.len() >= 3)
            .collect();

        if action_tokens.is_empty() {
            continue;
        }

        let intersection = memory_tokens.intersection(&action_tokens).count() as f32;
        let union = memory_tokens.union(&action_tokens).count() as f32;
        let overlap = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };

        if overlap > best_overlap {
            best_overlap = overlap;
            best_tool = action.tool_name.clone();
            best_success = action.success;
            best_reward = action.reward;
        }
    }

    if best_overlap < TOOL_USAGE_MIN_OVERLAP {
        return None;
    }

    // For robotics actions with explicit reward, use the reward directly
    let base_value = if let Some(reward) = best_reward {
        reward * best_overlap
    } else if best_success {
        TOOL_USAGE_SUCCESS_SIGNAL * best_overlap
    } else {
        TOOL_USAGE_FAILURE_SIGNAL * best_overlap
    };

    let confidence = if best_overlap >= TOOL_USAGE_STRONG_THRESHOLD {
        0.9
    } else {
        0.65
    };

    Some((base_value, confidence, best_tool, best_overlap))
}

/// Apply context pattern signals (repetition/topic change) to existing signals
///
/// This function modifies signal values based on detected user actions:
/// - Repetition (user asked same thing again): negative signal (memories failed)
/// - Topic change (user moved on): positive signal (task might be complete)
/// - Ignored (memory shown but no overlap): negative signal
///
/// # Arguments
/// - `signals`: Existing signals from process_implicit_feedback
/// - `is_repetition`: User is asking the same question again
/// - `is_topic_change`: User has moved to a different topic
/// - `context_similarity`: Similarity between current and previous context
pub fn apply_context_pattern_signals(
    signals: &mut [(MemoryId, SignalRecord)],
    is_repetition: bool,
    is_topic_change: bool,
    _context_similarity: f32,
) {
    for (memory_id, signal) in signals.iter_mut() {
        if is_repetition {
            // User asked the same thing again - memories didn't help
            // Apply penalty proportional to how irrelevant the memory was
            // FBK-4: Lowered threshold from 0.3 to 0.15 so more signals affect learning
            if signal.value < 0.15 {
                // Memory wasn't used in response AND user is re-asking
                signal.value += SIGNAL_REPETITION_PENALTY;
                signal.value = signal.value.clamp(-1.0, 1.0);
                signal.trigger = SignalTrigger::UserRepetition {
                    similarity: _context_similarity,
                };
                signal.confidence = 0.85; // High confidence - clear action signal
                tracing::debug!(
                    "Repetition detected for memory {:?}: applied penalty",
                    memory_id
                );
            }
        } else if is_topic_change {
            // User moved on to different topic - task might be complete
            // Apply boost to memories that were used in the response
            // FBK-4: Lowered threshold from 0.1 to 0.05 so more signals affect learning
            if signal.value > 0.05 {
                // Memory was somewhat used - boost it
                signal.value += SIGNAL_TOPIC_CHANGE_BOOST;
                signal.value = signal.value.clamp(-1.0, 1.0);
                signal.trigger = SignalTrigger::TopicChange {
                    similarity: _context_similarity,
                };
                signal.confidence = 0.7; // Moderate confidence
                tracing::debug!(
                    "Topic change detected for memory {:?}: applied boost",
                    memory_id
                );
            }
        }

        // Apply ignored penalty for memories with very low overlap
        // regardless of repetition/topic change
        if signal.value < -0.05 && signal.value > -0.3 {
            // Memory was surfaced but not used - strengthen the penalty
            signal.value = SIGNAL_IGNORED_PENALTY.min(signal.value);
            if !matches!(signal.trigger, SignalTrigger::UserRepetition { .. }) {
                signal.trigger = SignalTrigger::Ignored {
                    overlap_ratio: match &signal.trigger {
                        SignalTrigger::EntityOverlap { overlap_ratio } => *overlap_ratio,
                        _ => 0.0,
                    },
                };
            }
        }
    }
}

// =============================================================================
// FEEDBACK STORE
// =============================================================================

/// Previous context for a user - used for repetition/topic change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviousContext {
    /// The query/context text
    pub context: String,
    /// Embedding of the context for similarity comparison
    pub embedding: Vec<f32>,
    /// When this context was recorded
    pub timestamp: DateTime<Utc>,
    /// Memory IDs that were surfaced for this context
    pub surfaced_memory_ids: Vec<MemoryId>,
}

// =============================================================================
// METACOGNITIVE MEMORY: RETRIEVAL QUALITY TRACKER (A1)
// Reference: Bai et al. (2026) §3.2.3 — Reflexion, Memento, LightSearcher
// =============================================================================

/// Aggregated retrieval quality tracker (A1: Metacognitive Memory)
///
/// Tracks success/failure rates by entity cluster and query type,
/// enabling the system to observe its own retrieval performance and
/// identify areas where consolidation should be prioritized.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalQualityTracker {
    /// Per-entity retrieval quality: entity_name → stats
    pub entity_stats: HashMap<String, EntityRetrievalStats>,
    /// Per-query-type retrieval quality: type_name → stats
    pub query_type_stats: HashMap<String, QueryTypeRetrievalStats>,
    /// Time-windowed quality (hourly buckets, last 7 days)
    pub time_windows: VecDeque<TimeWindowStats>,
    /// Overall counters
    pub total_retrievals: u64,
    pub total_successful: u64,
    pub last_updated: Option<DateTime<Utc>>,
}

/// Per-entity retrieval quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRetrievalStats {
    pub retrieval_count: u32,
    pub success_count: u32,
    pub avg_signal: f32,
    pub last_retrieval: DateTime<Utc>,
}

/// Per-query-type retrieval quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTypeRetrievalStats {
    pub retrieval_count: u32,
    pub success_count: u32,
    pub avg_confidence: f32,
}

/// Hourly time window for quality tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindowStats {
    pub window_start: DateTime<Utc>,
    pub retrieval_count: u32,
    pub success_count: u32,
}

/// Maximum hourly windows to retain (7 days × 24 hours)
const QUALITY_TRACKER_MAX_WINDOWS: usize = 168;

impl RetrievalQualityTracker {
    /// Record a retrieval outcome for quality tracking
    pub fn record_retrieval(
        &mut self,
        entity_names: &[String],
        query_type: Option<&str>,
        signal_value: f32,
        confidence: f32,
    ) {
        let now = chrono::Utc::now();
        let is_success = signal_value > 0.0;

        self.total_retrievals += 1;
        if is_success {
            self.total_successful += 1;
        }
        self.last_updated = Some(now);

        // Update per-entity stats
        for entity in entity_names {
            let stats =
                self.entity_stats
                    .entry(entity.clone())
                    .or_insert_with(|| EntityRetrievalStats {
                        retrieval_count: 0,
                        success_count: 0,
                        avg_signal: 0.0,
                        last_retrieval: now,
                    });
            stats.retrieval_count += 1;
            if is_success {
                stats.success_count += 1;
            }
            // Exponential moving average of signal value
            let alpha = 0.2;
            stats.avg_signal = stats.avg_signal * (1.0 - alpha) + signal_value * alpha;
            stats.last_retrieval = now;
        }

        // Update per-query-type stats
        if let Some(qt) = query_type {
            let stats = self
                .query_type_stats
                .entry(qt.to_string())
                .or_insert_with(|| QueryTypeRetrievalStats {
                    retrieval_count: 0,
                    success_count: 0,
                    avg_confidence: 0.0,
                });
            stats.retrieval_count += 1;
            if is_success {
                stats.success_count += 1;
            }
            let alpha = 0.2;
            stats.avg_confidence = stats.avg_confidence * (1.0 - alpha) + confidence * alpha;
        }

        // Update time window
        let window_start = now
            .date_naive()
            .and_hms_opt(now.hour(), 0, 0)
            .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
            .unwrap_or(now);

        if let Some(last) = self.time_windows.back_mut() {
            if last.window_start == window_start {
                last.retrieval_count += 1;
                if is_success {
                    last.success_count += 1;
                }
            } else {
                self.time_windows.push_back(TimeWindowStats {
                    window_start,
                    retrieval_count: 1,
                    success_count: if is_success { 1 } else { 0 },
                });
            }
        } else {
            self.time_windows.push_back(TimeWindowStats {
                window_start,
                retrieval_count: 1,
                success_count: if is_success { 1 } else { 0 },
            });
        }

        // Prune old windows
        while self.time_windows.len() > QUALITY_TRACKER_MAX_WINDOWS {
            self.time_windows.pop_front();
        }
    }

    /// Get overall success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_retrievals == 0 {
            return 0.0;
        }
        self.total_successful as f32 / self.total_retrievals as f32
    }

    /// Get entities with lowest success rates (weak areas needing consolidation)
    pub fn weakest_entities(&self, top_n: usize) -> Vec<(String, f32)> {
        let mut entity_rates: Vec<(String, f32)> = self
            .entity_stats
            .iter()
            .filter(|(_, s)| s.retrieval_count >= 3) // Minimum retrievals for significance
            .map(|(name, s)| {
                let rate = s.success_count as f32 / s.retrieval_count as f32;
                (name.clone(), rate)
            })
            .collect();
        entity_rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        entity_rates.truncate(top_n);
        entity_rates
    }
}

/// Persistent store for feedback momentum with in-memory cache
#[derive(Default)]
pub struct FeedbackStore {
    /// In-memory cache: memory_id -> FeedbackMomentum
    pub momentum: HashMap<MemoryId, FeedbackMomentum>,

    /// Metacognitive quality tracker (A1: aggregated retrieval quality signals)
    pub quality_tracker: RetrievalQualityTracker,

    /// Pending feedback per user: user_id -> PendingFeedback (in-memory only)
    pending: HashMap<String, PendingFeedback>,

    /// Previous context per user: for repetition/topic change detection
    /// Tracks what the user asked last time to detect patterns
    previous_context: HashMap<String, PreviousContext>,

    /// Persistent storage for momentum data
    /// A handle bound to CF_FEEDBACK, not a database.

    ///

    /// Optional because this store works in memory with no persistence at all. That

    /// made the old shape a DOUBLE option — `(Some(db), Some(cf))` at every call site —

    /// where the second arm was a CF lookup that could fail independently of whether

    /// persistence was configured. One handle collapses it: bound or absent, and a

    /// missing column family becomes an error at construction rather than a silent

    /// no-op at each write.
    store: Option<Arc<dyn crate::storage::KeyValueStore>>,

    /// Track dirty entries that need persistence
    dirty: HashSet<MemoryId>,
}

impl std::fmt::Debug for FeedbackStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedbackStore")
            .field("momentum_count", &self.momentum.len())
            .field("pending_count", &self.pending.len())
            .field("previous_context_count", &self.previous_context.len())
            .field("has_store", &self.store.is_some())
            .field("dirty_count", &self.dirty.len())
            .finish()
    }
}

impl FeedbackStore {
    /// Create in-memory only store (no persistence)
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a handle directly; loads now because pending and previous context are served from memory only.
    pub fn with_store(
        store: Arc<dyn crate::storage::KeyValueStore>,
        base_path: &Path,
    ) -> anyhow::Result<Self> {
        // Not state this store keeps; the pre-column-family import runs in the rocksdb boot path.
        let _ = base_path;
        Self::load_from(store)
    }

    #[cfg_attr(not(feature = "rocksdb"), allow(dead_code))]
    /// Pre-column-family standalone DB under the data root, imported into
    /// CF_FEEDBACK by the rocksdb boot path before the store is bound.
    pub(crate) const LEGACY_SEPARATE_DBS: &'static [(&'static str, &'static str)] =
        &[("feedback", CF_FEEDBACK)];

    fn load_from(store: Arc<dyn crate::storage::KeyValueStore>) -> anyhow::Result<Self> {
        let mut momentum = HashMap::new();
        for (key, value) in store.scan_prefix(b"momentum:", None)? {
            if std::str::from_utf8(&key).is_err() {
                continue;
            }
            if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&value) {
                momentum.insert(m.memory_id.clone(), m);
            }
        }

        let mut pending = HashMap::new();
        for (key, value) in store.scan_prefix(b"pending:", None)? {
            if std::str::from_utf8(&key).is_err() {
                continue;
            }
            if let Ok(p) = serde_json::from_slice::<PendingFeedback>(&value) {
                if !p.is_expired() {
                    pending.insert(p.user_id.clone(), p);
                } else {
                    let _ = store.delete_bytes(&key);
                }
            }
        }

        let mut previous_context = HashMap::new();
        for (key, value) in store.scan_prefix(b"prev_ctx:", None)? {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Ok(ctx) = serde_json::from_slice::<PreviousContext>(&value) {
                    let user_id = key_str.strip_prefix("prev_ctx:").unwrap_or("");
                    previous_context.insert(user_id.to_string(), ctx);
                }
            }
        }

        tracing::info!(
            "Loaded {} momentum, {} pending, {} previous context from feedback store",
            momentum.len(),
            pending.len(),
            previous_context.len()
        );

        Ok(Self {
            momentum,
            quality_tracker: RetrievalQualityTracker::default(),
            pending,
            previous_context,
            store: Some(store),
            dirty: HashSet::new(),
        })
    }

    /// Create persistent store with its own standalone RocksDB instance.
    ///
    /// Primarily useful for tests and standalone operation. In production the
    /// rocksdb boot path binds [`with_store`](Self::with_store) to the shared DB.
    #[cfg(feature = "rocksdb")]
    pub fn with_persistence<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let cfs = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new(CF_FEEDBACK, {
                let mut cf_opts = Options::default();
                cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                cf_opts
            }),
        ];
        let db = DB::open_cf_descriptors(&opts, path.as_ref(), cfs)?;
        let db = Arc::new(db);
        Self::load_from(Arc::new(
            crate::storage::legacy_rocksdb::RocksDbColumnFamilyStore::new(db, CF_FEEDBACK),
        ))
    }

    /// Get or create momentum for a memory
    pub fn get_or_create_momentum(
        &mut self,
        memory_id: MemoryId,
        memory_type: ExperienceType,
    ) -> &mut FeedbackMomentum {
        // Check if we need to load from disk
        if !self.momentum.contains_key(&memory_id) {
            if let Some(store) = &self.store {
                let key = format!("momentum:{}", memory_id.0);
                if let Ok(Some(data)) = store.get_bytes(key.as_bytes()) {
                    if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&data) {
                        self.momentum.insert(memory_id.clone(), m);
                    }
                }
            }
        }

        self.momentum.entry(memory_id.clone()).or_insert_with(|| {
            self.dirty.insert(memory_id.clone());
            FeedbackMomentum::new(memory_id, memory_type)
        })
    }

    /// Get momentum for a memory (if exists in-memory), with disk fallback.
    /// Checks the in-memory HashMap first, then falls back to RocksDB.
    pub fn get_momentum(&self, memory_id: &MemoryId) -> Option<FeedbackMomentum> {
        if let Some(m) = self.momentum.get(memory_id) {
            return Some(m.clone());
        }
        // Fall back to disk lookup
        if let Some(store) = &self.store {
            let key = format!("momentum:{}", memory_id.0);
            if let Ok(Some(data)) = store.get_bytes(key.as_bytes()) {
                if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&data) {
                    return Some(m);
                }
            }
        }
        None
    }

    /// Mark a memory as dirty (needs persistence)
    pub fn mark_dirty(&mut self, memory_id: &MemoryId) {
        self.dirty.insert(memory_id.clone());
    }

    /// Set pending feedback for a user (also persists to disk)
    pub fn set_pending(&mut self, pending: PendingFeedback) {
        let user_id = pending.user_id.clone();
        self.pending.insert(user_id.clone(), pending.clone());

        // Persist to disk
        if let Some(store) = &self.store {
            let key = format!("pending:{}", user_id);
            if let Ok(value) = serde_json::to_vec(&pending) {
                if let Err(e) = store.put_bytes(key.as_bytes(), &value) {
                    tracing::warn!("Failed to persist pending feedback: {}", e);
                }
            }
        }
    }

    /// Take pending feedback for a user (removes from store and disk)
    pub fn take_pending(&mut self, user_id: &str) -> Option<PendingFeedback> {
        let result = self.pending.remove(user_id);

        // Remove from disk
        if let Some(store) = &self.store {
            let key = format!("pending:{}", user_id);
            let _ = store.delete_bytes(key.as_bytes());
        }

        result
    }

    /// Get pending feedback for a user (without removing)
    pub fn get_pending(&self, user_id: &str) -> Option<&PendingFeedback> {
        self.pending.get(user_id)
    }

    /// Clean up expired pending feedback
    pub fn cleanup_expired(&mut self) {
        self.pending.retain(|_, p| !p.is_expired());
    }

    /// Set previous context for a user (for repetition/topic change detection)
    /// Called when memories are surfaced to track what the user asked
    pub fn set_previous_context(
        &mut self,
        user_id: &str,
        context: String,
        embedding: Vec<f32>,
        surfaced_memory_ids: Vec<MemoryId>,
    ) {
        let prev_ctx = PreviousContext {
            context,
            embedding,
            timestamp: Utc::now(),
            surfaced_memory_ids,
        };

        self.previous_context
            .insert(user_id.to_string(), prev_ctx.clone());

        // Persist to disk
        if let Some(store) = &self.store {
            let key = format!("prev_ctx:{}", user_id);
            if let Ok(value) = serde_json::to_vec(&prev_ctx) {
                if let Err(e) = store.put_bytes(key.as_bytes(), &value) {
                    tracing::warn!("Failed to persist previous context: {}", e);
                }
            }
        }
    }

    /// Get previous context for a user
    pub fn get_previous_context(&self, user_id: &str) -> Option<&PreviousContext> {
        self.previous_context.get(user_id)
    }

    /// Compare current context to previous and detect action patterns
    /// Returns: (is_repetition, is_topic_change, similarity)
    /// - Repetition: similarity > 0.8 means user is asking same thing again (memories failed)
    /// - Topic change: similarity < 0.3 means user moved on (task might be complete)
    pub fn detect_context_pattern(
        &self,
        user_id: &str,
        current_embedding: &[f32],
    ) -> Option<(bool, bool, f32)> {
        let prev = self.previous_context.get(user_id)?;

        if prev.embedding.is_empty() || current_embedding.is_empty() {
            return None;
        }

        let similarity = crate::similarity::scale::to_reference(cosine_similarity(
            &prev.embedding,
            current_embedding,
        ));

        // ACT-R inspired thresholds
        let is_repetition = similarity > 0.8; // High similarity = re-asking
        let is_topic_change = similarity < 0.3; // Low similarity = moved on

        Some((is_repetition, is_topic_change, similarity))
    }

    /// Flush dirty entries to disk and ensure WAL is persisted
    pub fn flush(&mut self) -> anyhow::Result<usize> {
        let Some(store) = self.store.clone() else {
            return Ok(0);
        };

        // Drain dirty set first so the mutable borrow is released before we
        // take shared references to self.momentum / self.pending below.
        let dirty: Vec<MemoryId> = self.dirty.drain().collect();

        let mut flushed = 0;
        for memory_id in &dirty {
            if let Some(momentum) = self.momentum.get(memory_id) {
                let key = format!("momentum:{}", memory_id.0);
                let value = serde_json::to_vec(momentum)?;
                store.put_bytes(key.as_bytes(), &value)?;
                flushed += 1;
            }
        }

        // Also persist any pending feedback entries
        for (user_id, pending) in &self.pending {
            let key = format!("pending:{}", user_id);
            let value = serde_json::to_vec(pending)?;
            store.put_bytes(key.as_bytes(), &value)?;
        }

        // Persist before returning — the comment this replaces called it critical for
        // graceful shutdown, and the error was being DISCARDED. A flush that fails on
        // the way down is exactly the one a caller needs to hear about, so it now
        // propagates. The old call was CF-scoped with set_wait(true); the trait's
        // flush is whole-database and synchronous, so the wait is kept and the scope
        // widens to a superset.
        store.flush()?;

        if flushed > 0 {
            tracing::debug!("Flushed {} feedback momentum entries to disk", flushed);
        }

        Ok(flushed)
    }

    /// Whether this store persists at all.
    ///
    /// Replaces `database() -> Option<&Arc<DB>>`, which handed the raw engine handle
    /// out for "backup" — the same leak `MemorySystem::get_db()` is, one store down.
    /// A caller needing to back this up goes through BackupStore, which is the thing
    /// that knows how.
    pub fn is_persistent(&self) -> bool {
        self.store.is_some()
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

// =============================================================================
// R2 — RETRIEVAL TRACE STORE (LTR shadow feature/label capture)
// =============================================================================
//
// SHADOW-ONLY. This store records, per (recall, memory), the full unified-scoring
// feature vector + base/final score + query/rank at recall time, then JOINS the
// eventual feedback label and graduates the row to a TRAINING set for a future
// learned ranker (R3). It is a side-effect-only observer: nothing here is read by
// the live scoring path. It writes ONLY to its own per-user KV namespace.
//
// Two-stage buffer (J3 — fixes silent label loss):
//   - PENDING: un-labeled rows. TTL >= realistic feedback latency. Bounded by a
//     reservoir cap. Evicting a PENDING row before its label arrives is a LOST
//     join → increments `LTR_TRACE_LABEL_JOIN_MISS` (B9; never a silent drop).
//   - TRAINING: labeled+joined rows (the compacted store). Bounded by a ring cap.
//
// Per-user by construction: the store is handed the user's own `KeyValueStore`
// (one RocksDB CF per user), so trace rows never cross tenants.

/// Key prefix for un-labeled (PENDING) trace rows in the per-user KV store.
pub(crate) const TRACE_PENDING_PREFIX: &str = "ltr_trace_pending:";
/// Key prefix for labeled (TRAINING) trace rows in the per-user KV store.
pub(crate) const TRACE_TRAIN_PREFIX: &str = "ltr_trace_train:";

/// TTL for a PENDING (un-labeled) trace row. Must be >= a realistic feedback
/// latency window (feedback lags recall by minutes to hours). Rows older than this
/// with no label are evicted and counted as a join miss — never dropped silently.
pub(crate) const TRACE_PENDING_TTL_HOURS: i64 = 48;
/// Reservoir cap on PENDING rows. When exceeded, the OLDEST pending rows are
/// evicted (by created_at, not by recency of access) — i.e. rows closest to TTL
/// expiry go first, so we never evict a fresh row whose label may still be coming.
/// Each such eviction is a join miss (B9).
pub(crate) const TRACE_PENDING_MAX: usize = 50_000;
/// Recalls between full PENDING bound checks when the live row count is under the
/// cap. TTL expiry is time-driven, so it cannot be detected from the count alone;
/// the check still has to run, just not on every recall.
pub(crate) const TRACE_PENDING_CHECK_INTERVAL: usize = 64;
/// Ring cap on TRAINING rows. When exceeded, the oldest labeled rows are dropped
/// (a labeled row has already served its purpose for the join; pure-recency
/// eviction here is fine because the (features,label) pair is preserved-or-trained).
pub(crate) const TRACE_TRAIN_MAX: usize = 200_000;

/// A single retrieval-trace row (R2 / LTR).
///
/// I1: keyed in storage by the surrogate `trace_id`, NOT `memory_id` — so the row
/// survives a future merge that remaps memory ids (`memory_id` is a COLUMN here).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalTrace {
    /// Surrogate key (UUID). Storage key = prefix + trace_id.
    pub trace_id: String,
    /// The memory this row describes — a COLUMN, joinable, merge-remappable.
    pub memory_id: MemoryId,
    /// The query text that produced this retrieval.
    #[serde(default)]
    pub query_text: String,
    /// 0-based rank of this memory in the FINAL returned, ordered result set.
    #[serde(default)]
    pub rank: usize,
    /// Full unified-scoring feature vector (read out of the score path, never written to it).
    #[serde(default)]
    pub features: crate::memory::types::ScoringFeatures,
    /// Fused base score (closure input).
    #[serde(default)]
    pub base_score: f32,
    /// FINAL ordered score of the returned memory (`m.score` at return time).
    #[serde(default)]
    pub final_score: f32,
    /// The joined feedback label. `None` while PENDING; `Some(_)` once graduated.
    #[serde(default)]
    pub outcome: Option<crate::memory::retrieval::RetrievalOutcome>,
    /// Recall timestamp (drives the PENDING TTL).
    pub created_at: DateTime<Utc>,
    /// Feedback-join timestamp (set on graduation to TRAINING).
    #[serde(default)]
    pub labeled_at: Option<DateTime<Utc>>,
}

impl RetrievalTrace {
    /// Build an un-labeled (PENDING) trace row at recall time.
    pub fn new_pending(
        memory_id: MemoryId,
        query_text: String,
        rank: usize,
        features: crate::memory::types::ScoringFeatures,
        base_score: f32,
        final_score: f32,
    ) -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            memory_id,
            query_text,
            rank,
            features,
            base_score,
            final_score,
            outcome: None,
            created_at: Utc::now(),
            labeled_at: None,
        }
    }

    fn is_expired(&self, now: DateTime<Utc>) -> bool {
        self.outcome.is_none() && (now - self.created_at).num_hours() >= TRACE_PENDING_TTL_HOURS
    }
}

/// Per-user retrieval-trace store backing the LTR shadow capture (R2).
///
/// Persists through a per-user [`crate::storage::KeyValueStore`] handle; when none
/// is configured it operates as a no-op observer (capture is best-effort and must
/// never affect the request).
#[derive(Clone)]
pub struct RetrievalTraceStore {
    kv: Option<Arc<dyn crate::storage::KeyValueStore>>,
    /// Live PENDING row count. Seeded from a key scan at open, then maintained by
    /// `record_pending`, graduation and eviction, so a recall can compare it to the
    /// cap without scanning and decoding every row. Shared across clones.
    pending_rows: Arc<std::sync::atomic::AtomicUsize>,
    /// Recalls since the last full bound check.
    recalls_since_check: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for RetrievalTraceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetrievalTraceStore")
            .field("persistent", &self.kv.is_some())
            .finish()
    }
}

impl Default for RetrievalTraceStore {
    fn default() -> Self {
        Self {
            kv: None,
            pending_rows: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            recalls_since_check: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

impl RetrievalTraceStore {
    /// In-memory/no-op store (no persistence).
    pub fn new() -> Self {
        Self::default()
    }

    /// Persistent store over a per-user KV handle.
    pub fn with_kv(kv: Arc<dyn crate::storage::KeyValueStore>) -> Self {
        let seeded = match kv.scan_prefix(TRACE_PENDING_PREFIX.as_bytes(), None) {
            Ok(rows) => rows.len(),
            Err(e) => {
                tracing::warn!(error = %e, "LTR trace: failed to count pending rows at open");
                0
            }
        };
        Self {
            kv: Some(kv),
            pending_rows: Arc::new(std::sync::atomic::AtomicUsize::new(seeded)),
            ..Self::default()
        }
    }

    fn sub_pending(&self, n: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let _ = self
            .pending_rows
            .fetch_update(Relaxed, Relaxed, |c| Some(c.saturating_sub(n)));
    }

    fn pending_key(trace_id: &str) -> Vec<u8> {
        format!("{TRACE_PENDING_PREFIX}{trace_id}").into_bytes()
    }

    fn train_key(trace_id: &str) -> Vec<u8> {
        format!("{TRACE_TRAIN_PREFIX}{trace_id}").into_bytes()
    }

    /// Capture one PENDING trace row (one per (recall, memory)). Best-effort:
    /// failures are logged, never propagated to the recall path.
    ///
    /// Bound enforcement (B9 — TTL eviction + reservoir cap) is intentionally
    /// NOT called here. The recall write-site emits one row per result in a
    /// tight loop; calling `enforce_pending_bound` per insert would run a full
    /// KV prefix scan per row. Callers MUST call `enforce_pending_bound` once
    /// after the loop (batch enforcement), not once per insert.
    pub fn record_pending(&self, trace: &RetrievalTrace) {
        let Some(kv) = &self.kv else { return };
        match serde_json::to_vec(trace) {
            Ok(bytes) => match kv.put_bytes(&Self::pending_key(&trace.trace_id), &bytes) {
                Ok(()) => {
                    self.pending_rows
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LTR trace: failed to persist pending row")
                }
            },
            Err(e) => tracing::warn!(error = %e, "LTR trace: failed to serialize pending row"),
        }
    }

    /// Read all PENDING rows (un-labeled), newest-irrelevant ordering.
    pub fn load_pending(&self) -> Vec<RetrievalTrace> {
        let Some(kv) = &self.kv else {
            return Vec::new();
        };
        match kv.scan_prefix(TRACE_PENDING_PREFIX.as_bytes(), None) {
            Ok(rows) => rows
                .into_iter()
                .filter_map(|(_, v)| serde_json::from_slice::<RetrievalTrace>(&v).ok())
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "LTR trace: failed to scan pending rows");
                Vec::new()
            }
        }
    }

    /// Read all TRAINING rows (labeled + joined). Used by tests and a future R3.
    pub fn load_training(&self) -> Vec<RetrievalTrace> {
        let Some(kv) = &self.kv else {
            return Vec::new();
        };
        match kv.scan_prefix(TRACE_TRAIN_PREFIX.as_bytes(), None) {
            Ok(rows) => rows
                .into_iter()
                .filter_map(|(_, v)| serde_json::from_slice::<RetrievalTrace>(&v).ok())
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "LTR trace: failed to scan training rows");
                Vec::new()
            }
        }
    }

    /// J3/JOIN: a feedback label arrived for these memories. Graduate the matching
    /// PENDING rows to TRAINING with the label. Joins by `memory_id` (feedback
    /// correlates by memory id, not retrieval id), taking the MOST RECENT pending
    /// row per memory. Returns the number of rows graduated.
    pub fn join_feedback(
        &self,
        memory_ids: &[MemoryId],
        outcome: crate::memory::retrieval::RetrievalOutcome,
    ) -> usize {
        let Some(kv) = &self.kv else { return 0 };
        if memory_ids.is_empty() {
            return 0;
        }
        let wanted: HashSet<&MemoryId> = memory_ids.iter().collect();

        // Pick the most-recent un-expired pending row per wanted memory_id.
        let now = Utc::now();
        let mut best: HashMap<MemoryId, RetrievalTrace> = HashMap::new();
        for t in self.load_pending() {
            if t.outcome.is_some() || t.is_expired(now) {
                continue;
            }
            if !wanted.contains(&t.memory_id) {
                continue;
            }
            match best.get(&t.memory_id) {
                Some(existing) if existing.created_at >= t.created_at => {}
                _ => {
                    best.insert(t.memory_id.clone(), t);
                }
            }
        }

        let mut graduated = 0;
        for (_mid, mut trace) in best {
            trace.outcome = Some(outcome);
            trace.labeled_at = Some(now);
            // Write to TRAINING, remove from PENDING (atomic enough: training row is
            // the commit; a crash between writes only re-graduates idempotently).
            match serde_json::to_vec(&trace) {
                Ok(bytes) => {
                    if kv
                        .put_bytes(&Self::train_key(&trace.trace_id), &bytes)
                        .is_ok()
                    {
                        let _ = kv.delete_bytes(&Self::pending_key(&trace.trace_id));
                        self.sub_pending(1);
                        graduated += 1;
                        crate::metrics::LTR_TRACE_GRADUATED_TOTAL.inc();
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LTR trace: failed to serialize graduated row")
                }
            }
        }
        if graduated > 0 {
            self.enforce_training_bound();
        }
        graduated
    }

    /// B9/J3: bound the PENDING stage. Evict TTL-expired rows AND, if still over the
    /// reservoir cap, the oldest rows (closest to expiry) — each eviction of an
    /// un-labeled row is a LOST join, counted, never silent. Returns evicted count.
    pub fn enforce_pending_bound(&self) -> usize {
        let Some(kv) = &self.kv else { return 0 };
        let now = Utc::now();
        let mut rows = self.load_pending();

        let mut evicted = 0;
        // 1) TTL eviction.
        rows.retain(|t| {
            if t.is_expired(now) {
                let _ = kv.delete_bytes(&Self::pending_key(&t.trace_id));
                crate::metrics::LTR_TRACE_LABEL_JOIN_MISS.inc();
                evicted += 1;
                false
            } else {
                true
            }
        });

        // 2) Reservoir cap: evict OLDEST first (closest to TTL), so fresh rows whose
        // label may still arrive are preserved (do NOT evict by pure recency ahead
        // of TTL — J3).
        let mut cap_evicted = 0;
        if rows.len() > TRACE_PENDING_MAX {
            rows.sort_by(|a, b| a.created_at.cmp(&b.created_at));
            let overflow = rows.len() - TRACE_PENDING_MAX;
            cap_evicted = overflow;
            for t in rows.iter().take(overflow) {
                let _ = kv.delete_bytes(&Self::pending_key(&t.trace_id));
                crate::metrics::LTR_TRACE_LABEL_JOIN_MISS.inc();
                evicted += 1;
            }
        }

        // `rows` was scanned fresh above, so this also corrects any drift in the counter.
        self.pending_rows.store(
            rows.len() - cap_evicted,
            std::sync::atomic::Ordering::Relaxed,
        );

        if evicted > 0 {
            tracing::info!(
                evicted,
                "LTR trace: evicted pending rows before label (join_miss counted)"
            );
        }
        evicted
    }

    /// `enforce_pending_bound` for the recall path: a full scan only when the live
    /// count is over the cap or every `TRACE_PENDING_CHECK_INTERVAL` calls (which is
    /// what catches TTL expiry). Returns the number of rows evicted by this call.
    pub fn enforce_pending_bound_amortized(&self) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        if self.kv.is_none() {
            return 0;
        }
        let over_cap = self.pending_rows.load(Relaxed) > TRACE_PENDING_MAX;
        let due =
            self.recalls_since_check.fetch_add(1, Relaxed) + 1 >= TRACE_PENDING_CHECK_INTERVAL;
        if !over_cap && !due {
            return 0;
        }
        self.recalls_since_check.store(0, Relaxed);
        self.enforce_pending_bound()
    }

    /// Bound the TRAINING stage (ring): drop oldest labeled rows over the cap.
    /// Labeled rows are not join-misses — the (features,label) pair already exists.
    fn enforce_training_bound(&self) -> usize {
        let Some(kv) = &self.kv else { return 0 };
        let mut rows = self.load_training();
        if rows.len() <= TRACE_TRAIN_MAX {
            return 0;
        }
        rows.sort_by(|a, b| {
            a.labeled_at
                .unwrap_or(a.created_at)
                .cmp(&b.labeled_at.unwrap_or(b.created_at))
        });
        let overflow = rows.len() - TRACE_TRAIN_MAX;
        let mut evicted = 0;
        for t in rows.iter().take(overflow) {
            if kv.delete_bytes(&Self::train_key(&t.trace_id)).is_ok() {
                evicted += 1;
            }
        }
        evicted
    }

    /// Count of PENDING rows (test/diagnostic).
    pub fn pending_count(&self) -> usize {
        self.load_pending().len()
    }

    /// Count of TRAINING rows (test/diagnostic).
    pub fn training_count(&self) -> usize {
        self.load_training().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_signal_from_entity_overlap() {
        // Strong overlap (>= 0.4 after FBK-4 threshold adjustment)
        let signal = SignalRecord::from_entity_overlap(0.7);
        assert!(signal.value > 0.5);
        assert!(signal.confidence > 0.8);

        // Weak overlap (>= 0.1 after FBK-4 threshold adjustment)
        let signal = SignalRecord::from_entity_overlap(0.3);
        assert!(signal.value > 0.0);
        assert!(signal.value < 0.5);

        // No overlap (< 0.1 after FBK-4 threshold adjustment)
        let signal = SignalRecord::from_entity_overlap(0.05);
        assert!(signal.value < 0.0);
    }

    #[test]
    fn test_momentum_inertia_by_type() {
        let learning = FeedbackMomentum::new(MemoryId(Uuid::new_v4()), ExperienceType::Learning);
        let conversation =
            FeedbackMomentum::new(MemoryId(Uuid::new_v4()), ExperienceType::Conversation);

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
                SignalTrigger::TopicChange { similarity: 0.2 },
            ));
        }
        assert_eq!(Trend::from_signals(&signals), Trend::Improving);

        // Add declining signals (steeper slope < -0.1 threshold)
        signals.clear();
        for i in (0..10).rev() {
            signals.push_back(SignalRecord::new(
                i as f32 * 0.15, // 1.35, 1.2, ... 0 gives slope ~-0.15
                1.0,
                SignalTrigger::TopicChange { similarity: 0.2 },
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
        // Multi-word phrase detection
        let text = "No, that's not what I meant";
        let keywords = detect_negative_keywords(text);
        assert!(keywords.contains(&"not what i meant".to_string()));

        // Irrelevance signals
        let text2 = "That's not helpful at all, it's irrelevant";
        let keywords2 = detect_negative_keywords(text2);
        assert!(keywords2.contains(&"not helpful".to_string()));
        assert!(keywords2.contains(&"irrelevant".to_string()));

        // Explicit rejection
        let text3 = "Please forget that, it doesn't work";
        let keywords3 = detect_negative_keywords(text3);
        assert!(keywords3.contains(&"forget that".to_string()));
        assert!(keywords3.contains(&"doesn't work".to_string()));

        // No false positives on neutral text
        let text4 = "Can you help me debug this function?";
        let keywords4 = detect_negative_keywords(text4);
        assert!(keywords4.is_empty());
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
                embedding: Vec::new(),
            }],
        );
        store.set_pending(pending);

        // Should have pending now
        assert!(store.get_pending(user_id).is_some());
        assert_eq!(
            store.get_pending(user_id).unwrap().surfaced_memories.len(),
            1
        );

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
        let momentum = store.get_or_create_momentum(memory_id.clone(), ExperienceType::Context);
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
                    entities: ["rust", "async", "tokio"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    content_preview: "Rust async with tokio".to_string(),
                    score: 0.9,
                    embedding: Vec::new(),
                },
                SurfacedMemoryInfo {
                    id: memory_id2.clone(),
                    entities: ["python", "django"].iter().map(|s| s.to_string()).collect(),
                    content_preview: "Python Django web".to_string(),
                    score: 0.3,
                    embedding: Vec::new(),
                },
            ],
        );

        // Response that uses Rust async terminology
        let response =
            "To use async in Rust, you can use tokio runtime. Here is an example with async await.";
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
                embedding: Vec::new(),
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
            let mut momentum =
                FeedbackMomentum::new(MemoryId(Uuid::new_v4()), ExperienceType::Context);
            momentum.ema = i as f32 * 0.2; // 0, 0.2, 0.4, 0.6, 0.8
            store.momentum.insert(momentum.memory_id.clone(), momentum);
        }

        let stats = store.stats();
        assert_eq!(stats.total_momentum_entries, 5);
        assert!((stats.avg_ema - 0.4).abs() < 0.01); // (0+0.2+0.4+0.6+0.8)/5 = 0.4
    }

    #[test]
    fn test_process_feedback_with_semantic_similarity() {
        let memory_id1 = MemoryId(Uuid::new_v4());
        let memory_id2 = MemoryId(Uuid::new_v4());

        // Create embeddings: similar embeddings for related content
        let rust_embedding: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let python_embedding: Vec<f32> = (0..384).map(|i| 1.0 - (i as f32) * 0.01).collect();

        let pending = PendingFeedback::new(
            "user1".to_string(),
            "How do I use async in Rust?".to_string(),
            vec![0.1; 384],
            vec![
                SurfacedMemoryInfo {
                    id: memory_id1.clone(),
                    entities: ["rust", "async", "tokio"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    content_preview: "Rust async with tokio".to_string(),
                    score: 0.9,
                    embedding: rust_embedding.clone(),
                },
                SurfacedMemoryInfo {
                    id: memory_id2.clone(),
                    entities: ["python", "django"].iter().map(|s| s.to_string()).collect(),
                    content_preview: "Python Django web".to_string(),
                    score: 0.3,
                    embedding: python_embedding.clone(),
                },
            ],
        );

        // Response embedding similar to rust_embedding
        let response = "Here is how to use async/await in Rust with tokio runtime.";
        let response_embedding = rust_embedding; // Similar to memory 1

        // Process without semantic (backwards compat)
        let signals_entity_only = process_implicit_feedback(&pending, response, None);

        // Process with semantic similarity
        let signals_with_semantic = process_implicit_feedback_with_semantics(
            &pending,
            response,
            None,
            Some(&response_embedding),
        );

        // First memory should score higher with semantic (response embedding matches memory embedding)
        let (id1, _sig1_entity) = &signals_entity_only[0];
        let (_, sig1_semantic) = &signals_with_semantic[0];
        assert_eq!(id1, &memory_id1);

        // With context_embedding available, should use InformationAttribution trigger
        // (projects out query component to isolate memory's unique contribution)
        match &sig1_semantic.trigger {
            SignalTrigger::InformationAttribution {
                attribution_score,
                raw_similarity,
            } => {
                // Raw similarity is high (response_emb == memory_emb)
                assert!(*raw_similarity > 0.9);
                // Attribution score measures residual after projecting out query
                assert!(*attribution_score >= 0.0);
            }
            SignalTrigger::SemanticSimilarity { similarity } => {
                // Fallback if context_embedding is degenerate
                assert!(*similarity > 0.9);
            }
            _ => panic!("Expected InformationAttribution or SemanticSimilarity trigger"),
        }

        // Second memory (python) should have low score since embedding is different
        let (id2, sig2_semantic) = &signals_with_semantic[1];
        assert_eq!(id2, &memory_id2);
        match &sig2_semantic.trigger {
            SignalTrigger::InformationAttribution { raw_similarity, .. } => {
                assert!(*raw_similarity < 0.5); // Different embeddings
            }
            SignalTrigger::SemanticSimilarity { similarity } => {
                assert!(*similarity < 0.5);
            }
            _ => panic!("Expected InformationAttribution or SemanticSimilarity trigger"),
        }
    }

    #[test]
    fn test_cosine_similarity_basic() {
        // Identical vectors = 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors = 0.0
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        // Opposite vectors = -1.0
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);

        // Empty vectors = 0.0
        assert!((cosine_similarity(&[], &[]) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_entity_flow() {
        use std::collections::HashSet;

        // Case 1: Response heavily derived from memory
        let memory_entities: HashSet<String> = ["rust", "async", "tokio", "futures"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let response_entities: HashSet<String> = ["rust", "async", "tokio", "runtime"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let (derived_ratio, novel_ratio, derived_count, total) =
            calculate_entity_flow(&memory_entities, &response_entities);

        assert_eq!(derived_count, 3); // rust, async, tokio
        assert_eq!(total, 4);
        assert!((derived_ratio - 0.75).abs() < 0.01);
        assert!((novel_ratio - 0.25).abs() < 0.01);

        // Case 2: Response mostly novel (memory not used)
        let response_novel: HashSet<String> = ["python", "django", "flask", "web"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let (derived_ratio2, novel_ratio2, derived_count2, _) =
            calculate_entity_flow(&memory_entities, &response_novel);

        assert_eq!(derived_count2, 0);
        assert!((derived_ratio2 - 0.0).abs() < 0.01);
        assert!((novel_ratio2 - 1.0).abs() < 0.01);

        // Case 3: Empty response
        let empty: HashSet<String> = HashSet::new();
        let (dr, nr, dc, total) = calculate_entity_flow(&memory_entities, &empty);
        assert_eq!(dc, 0);
        assert_eq!(total, 0);
        assert!((dr - 0.0).abs() < 0.01);
        assert!((nr - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_from_entity_flow() {
        // Case 1: High derived ratio (>=0.5) = strong positive
        let sig1 = signal_from_entity_flow(0.75, 0.25, 3, 4);
        assert!(sig1.value > 0.5); // Strong positive
        assert!((sig1.confidence - 0.8).abs() < 0.01); // Good sample size

        // Case 2: Medium derived ratio (0.2 to 0.5) = weak positive
        let sig2 = signal_from_entity_flow(0.3, 0.7, 1, 4);
        assert!(sig2.value > 0.0 && sig2.value <= 0.5); // Weak positive
        assert!((sig2.confidence - 0.8).abs() < 0.01);

        // Case 3: Low derived, high novel = slight negative
        let sig3 = signal_from_entity_flow(0.1, 0.9, 0, 4);
        assert!(sig3.value < 0.0); // Negative
        assert!((sig3.value - (-0.1)).abs() < 0.01);

        // Case 4: Small sample size = lower confidence
        let sig4 = signal_from_entity_flow(0.5, 0.5, 1, 2);
        assert!((sig4.confidence - 0.5).abs() < 0.01);

        // Verify trigger variant
        match sig1.trigger {
            SignalTrigger::EntityFlow {
                derived_ratio,
                novel_ratio,
                memory_entities_used,
                response_entities_total,
            } => {
                assert!((derived_ratio - 0.75).abs() < 0.01);
                assert!((novel_ratio - 0.25).abs() < 0.01);
                assert_eq!(memory_entities_used, 3);
                assert_eq!(response_entities_total, 4);
            }
            _ => panic!("Expected EntityFlow trigger"),
        }
    }

    // =====================================================================
    // R2 (LTR) retrieval-trace store tests.
    //
    // These exercise the trace store directly over an in-test HashMap-backed
    // KeyValueStore (no RocksDB/redb files needed), so the PENDING→TRAINING
    // join, the J3 bounded-eviction join-miss accounting, and per-user
    // isolation are deterministic and fast. The SHADOW-INVARIANT guarantee
    // (capture never perturbs final_score or result order) is asserted in a
    // separate integration test in recall.rs where the full scoring path runs.
    // =====================================================================

    use crate::config::StorageBackend;
    use crate::memory::retrieval::RetrievalOutcome;
    use crate::storage::KeyValueStore;
    use parking_lot::Mutex as PlMutex;
    use std::collections::BTreeMap;

    /// Minimal in-memory KeyValueStore for trace-store tests. A BTreeMap gives
    /// ordered prefix scans for free; Mutex makes it Send+Sync.
    #[derive(Default)]
    struct MemKv {
        map: PlMutex<BTreeMap<Vec<u8>, Vec<u8>>>,
    }

    impl KeyValueStore for MemKv {
        fn backend(&self) -> StorageBackend {
            StorageBackend::Redb
        }
        fn get_bytes(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(self.map.lock().get(key).cloned())
        }
        fn put_bytes(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
            self.map.lock().insert(key.to_vec(), value.to_vec());
            Ok(())
        }
        fn delete_bytes(&self, key: &[u8]) -> anyhow::Result<()> {
            self.map.lock().remove(key);
            Ok(())
        }
        fn scan_prefix(
            &self,
            prefix: &[u8],
            limit: Option<usize>,
        ) -> anyhow::Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let guard = self.map.lock();
            let mut out: Vec<(Vec<u8>, Vec<u8>)> = guard
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if let Some(n) = limit {
                out.truncate(n);
            }
            Ok(out)
        }
        fn flush(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> anyhow::Result<()> {
            let mut map = self.map.lock();
            for (k, v) in pairs {
                map.insert(k.clone(), v.clone());
            }
            Ok(())
        }
        fn delete_many(&self, keys: &[Vec<u8>]) -> anyhow::Result<()> {
            let mut map = self.map.lock();
            for k in keys {
                map.remove(k);
            }
            Ok(())
        }
    }

    fn mem_store() -> RetrievalTraceStore {
        RetrievalTraceStore::with_kv(Arc::new(MemKv::default()))
    }

    fn pending_trace(memory_id: MemoryId, rank: usize) -> RetrievalTrace {
        RetrievalTrace::new_pending(
            memory_id,
            "what are our priorities".to_string(),
            rank,
            crate::memory::types::ScoringFeatures {
                base_score: 0.5,
                unified_final: 0.9,
                ..Default::default()
            },
            0.5,
            0.9,
        )
    }

    fn live_rows(store: &RetrievalTraceStore) -> usize {
        store
            .pending_rows
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The atomic count follows record, graduation and eviction, and a reopened
    /// store seeds it from the persisted rows.
    #[test]
    fn test_pending_counter_tracks_rows() {
        let kv = Arc::new(MemKv::default());
        let store = RetrievalTraceStore::with_kv(kv.clone());
        assert_eq!(live_rows(&store), 0);

        let graduating = MemoryId(Uuid::new_v4());
        store.record_pending(&pending_trace(graduating.clone(), 0));
        store.record_pending(&pending_trace(MemoryId(Uuid::new_v4()), 1));
        let mut stale = pending_trace(MemoryId(Uuid::new_v4()), 2);
        stale.created_at = Utc::now() - chrono::Duration::hours(TRACE_PENDING_TTL_HOURS + 1);
        store.record_pending(&stale);
        assert_eq!(live_rows(&store), 3);
        assert_eq!(live_rows(&store), store.pending_count());

        store.join_feedback(&[graduating], RetrievalOutcome::Helpful);
        assert_eq!(live_rows(&store), 2);

        assert_eq!(store.enforce_pending_bound(), 1);
        assert_eq!(live_rows(&store), 1);
        assert_eq!(live_rows(&store), store.pending_count());

        let reopened = RetrievalTraceStore::with_kv(kv);
        assert_eq!(live_rows(&reopened), 1, "seeded from persisted rows");
    }

    /// Under the cap, the full scan runs once per check interval, not per recall;
    /// the interval is what retires TTL-expired rows.
    #[test]
    fn test_amortized_enforcement_runs_once_per_interval() {
        let store = mem_store();
        let mut stale = pending_trace(MemoryId(Uuid::new_v4()), 0);
        stale.created_at = Utc::now() - chrono::Duration::hours(TRACE_PENDING_TTL_HOURS + 1);
        store.record_pending(&stale);

        for _ in 0..TRACE_PENDING_CHECK_INTERVAL - 1 {
            assert_eq!(store.enforce_pending_bound_amortized(), 0);
        }
        assert_eq!(
            store.pending_count(),
            1,
            "no scan before the interval elapses"
        );

        assert_eq!(store.enforce_pending_bound_amortized(), 1);
        assert_eq!(store.pending_count(), 0);
        assert_eq!(live_rows(&store), 0);
    }

    /// PENDING → feedback → TRAINING join graduates exactly the matching row,
    /// stamps the outcome label + labeled_at, and removes it from PENDING.
    #[test]
    fn test_trace_join_graduates_to_training() {
        let store = mem_store();
        let mid = MemoryId(Uuid::new_v4());
        store.record_pending(&pending_trace(mid.clone(), 0));
        // An unrelated pending row that must NOT graduate.
        let other = MemoryId(Uuid::new_v4());
        store.record_pending(&pending_trace(other.clone(), 1));

        assert_eq!(store.pending_count(), 2);
        assert_eq!(store.training_count(), 0);

        let graduated = store.join_feedback(&[mid.clone()], RetrievalOutcome::Helpful);
        assert_eq!(graduated, 1, "exactly the matched memory graduates");

        assert_eq!(store.pending_count(), 1, "graduated row left PENDING");
        assert_eq!(store.training_count(), 1, "graduated row is in TRAINING");

        let train = store.load_training();
        assert_eq!(train.len(), 1);
        assert_eq!(train[0].memory_id, mid);
        assert_eq!(train[0].outcome, Some(RetrievalOutcome::Helpful));
        assert!(train[0].labeled_at.is_some(), "label timestamp stamped");

        // The non-matched row is still pending and un-labeled.
        let pend = store.load_pending();
        assert_eq!(pend.len(), 1);
        assert_eq!(pend[0].memory_id, other);
        assert!(pend[0].outcome.is_none());
    }

    /// A negative outcome (Abstained/Ignored — I8) is a valid training LABEL: the
    /// row still graduates, carrying its negative label for the future ranker.
    #[test]
    fn test_trace_join_carries_negative_label() {
        let store = mem_store();
        let mid = MemoryId(Uuid::new_v4());
        store.record_pending(&pending_trace(mid.clone(), 0));

        let graduated = store.join_feedback(&[mid.clone()], RetrievalOutcome::Ignored);
        assert_eq!(graduated, 1);
        let train = store.load_training();
        assert_eq!(train[0].outcome, Some(RetrievalOutcome::Ignored));
    }

    /// One (recall, memory) → one PENDING row: re-recording the same memory across
    /// two recalls makes two rows, and the join graduates the MOST RECENT one.
    #[test]
    fn test_trace_join_picks_most_recent_pending() {
        let store = mem_store();
        let mid = MemoryId(Uuid::new_v4());

        let mut older = pending_trace(mid.clone(), 5);
        older.created_at = Utc::now() - chrono::Duration::hours(1);
        older.final_score = 0.10;
        store.record_pending(&older);

        let mut newer = pending_trace(mid.clone(), 0);
        newer.final_score = 0.90;
        store.record_pending(&newer);

        assert_eq!(
            store.pending_count(),
            2,
            "one row per recall for same memory"
        );

        let graduated = store.join_feedback(&[mid.clone()], RetrievalOutcome::Helpful);
        assert_eq!(graduated, 1);
        let train = store.load_training();
        assert_eq!(train.len(), 1);
        assert!(
            (train[0].final_score - 0.90).abs() < 1e-6,
            "the most-recent pending row is the one graduated"
        );
        // The older row remains PENDING (it will later TTL-evict as a join miss).
        assert_eq!(store.pending_count(), 1);
    }

    /// J3: a PENDING row past its TTL with no label is EVICTED and counted as a
    /// label-join miss — never silently dropped. Enforcement is now BATCHED:
    /// `record_pending` no longer triggers a full scan per insert; eviction fires
    /// on the first `enforce_pending_bound` call after the batch of inserts.
    #[test]
    fn test_trace_ttl_eviction_counts_join_miss() {
        let store = mem_store();

        // A fresh row (must survive).
        let fresh = MemoryId(Uuid::new_v4());
        store.record_pending(&pending_trace(fresh.clone(), 0));

        let stale = MemoryId(Uuid::new_v4());
        let mut stale_row = pending_trace(stale.clone(), 1);
        stale_row.created_at = Utc::now() - chrono::Duration::hours(TRACE_PENDING_TTL_HOURS + 1);
        store.record_pending(&stale_row);

        // After the batch of inserts the expired row is still in KV — eviction
        // has NOT fired yet (no per-insert scan). Both rows are visible.
        assert_eq!(
            store.pending_count(),
            2,
            "expired row not yet evicted — enforcement is deferred to batch call"
        );

        // Now simulate the batch-enforcement call (one per recall, not per insert).
        let before = crate::metrics::LTR_TRACE_LABEL_JOIN_MISS.get();
        let evicted = store.enforce_pending_bound();
        let after = crate::metrics::LTR_TRACE_LABEL_JOIN_MISS.get();

        assert_eq!(evicted, 1, "enforce evicts exactly the one expired row");
        // The expired row's lost label is COUNTED, not silently dropped (J3).
        // >= before+1: counter is process-global; parallel tests may also increment.
        assert!(
            after >= before + 1,
            "the lost label is COUNTED as a join miss, not silently dropped \
             (before={before}, after={after})"
        );

        // A second enforce is now a no-op: only the fresh row remains.
        assert_eq!(
            store.enforce_pending_bound(),
            0,
            "second enforce is a no-op; fresh row survives"
        );

        // Fresh row preserved; expired row gone.
        let pend = store.load_pending();
        assert_eq!(pend.len(), 1);
        assert_eq!(pend[0].memory_id, fresh);
    }

    /// Confirms that a single insert does NOT trigger a full scan (no per-insert
    /// enforcement), and that calling `enforce_pending_bound` once after N inserts
    /// still correctly applies the TTL and reservoir cap.
    #[test]
    fn test_pending_bound_batch_enforced_not_per_insert() {
        let store = mem_store();

        // Insert N fresh rows — none should be evicted by record_pending itself.
        const N: usize = 5;
        let fresh_ids: Vec<MemoryId> = (0..N).map(|_| MemoryId(Uuid::new_v4())).collect();
        for (rank, mid) in fresh_ids.iter().enumerate() {
            store.record_pending(&pending_trace(mid.clone(), rank));
        }

        // Also insert two expired rows.
        let expired_ids: Vec<MemoryId> = (0..2).map(|_| MemoryId(Uuid::new_v4())).collect();
        for (rank, mid) in expired_ids.iter().enumerate() {
            let mut row = pending_trace(mid.clone(), N + rank);
            row.created_at = Utc::now() - chrono::Duration::hours(TRACE_PENDING_TTL_HOURS + 1);
            store.record_pending(&row);
        }

        // No enforcement has fired yet — all N+2 rows are in KV.
        assert_eq!(
            store.pending_count(),
            N + 2,
            "no per-insert enforcement; all rows present before batch call"
        );

        // One batch enforce: only the 2 expired rows should be evicted.
        let evicted = store.enforce_pending_bound();
        assert_eq!(evicted, 2, "batch enforce evicts exactly the expired rows");
        assert_eq!(
            store.pending_count(),
            N,
            "fresh rows survive the batch enforce"
        );

        // Confirm the surviving rows are the originally-fresh ones.
        let pend_ids: std::collections::HashSet<MemoryId> = store
            .load_pending()
            .into_iter()
            .map(|t| t.memory_id)
            .collect();
        for mid in &fresh_ids {
            assert!(pend_ids.contains(mid), "fresh row {mid:?} must survive");
        }
        for mid in &expired_ids {
            assert!(
                !pend_ids.contains(mid),
                "expired row {mid:?} must be evicted"
            );
        }
    }

    /// An expired PENDING row can no longer be joined (it is past TTL): the join
    /// returns 0 and does not resurrect it into TRAINING.
    #[test]
    fn test_trace_join_skips_expired_pending() {
        let store = mem_store();
        let mid = MemoryId(Uuid::new_v4());
        let mut expired = pending_trace(mid.clone(), 0);
        expired.created_at = Utc::now() - chrono::Duration::hours(TRACE_PENDING_TTL_HOURS + 1);
        store.record_pending(&expired);

        let graduated = store.join_feedback(&[mid.clone()], RetrievalOutcome::Helpful);
        assert_eq!(graduated, 0, "expired pending rows are not joinable");
        assert_eq!(store.training_count(), 0);
    }

    /// Per-user isolation: two stores over DISTINCT KV handles (the per-user CF
    /// model) never see each other's rows. A join on store A graduates nothing in
    /// store B, and B's pending row is untouched.
    #[test]
    fn test_trace_per_user_isolation() {
        let store_a = mem_store();
        let store_b = mem_store();

        let shared_mid = MemoryId(Uuid::new_v4());
        store_a.record_pending(&pending_trace(shared_mid.clone(), 0));
        store_b.record_pending(&pending_trace(shared_mid.clone(), 0));

        // Feedback arrives "for user A" — joins only A's store.
        let graduated_a = store_a.join_feedback(&[shared_mid.clone()], RetrievalOutcome::Helpful);
        assert_eq!(graduated_a, 1);

        assert_eq!(store_a.training_count(), 1);
        assert_eq!(store_a.pending_count(), 0);

        // B is completely untouched by A's feedback.
        assert_eq!(store_b.training_count(), 0, "user B sees no training row");
        assert_eq!(store_b.pending_count(), 1, "user B's pending row intact");
    }

    /// A no-op (no KV) store is a silent observer: capture and join never panic
    /// and never persist — the recall path is never affected when unconfigured.
    #[test]
    fn test_trace_noop_store_is_inert() {
        let store = RetrievalTraceStore::new();
        let mid = MemoryId(Uuid::new_v4());
        store.record_pending(&pending_trace(mid.clone(), 0));
        assert_eq!(store.pending_count(), 0);
        assert_eq!(store.join_feedback(&[mid], RetrievalOutcome::Helpful), 0);
        assert_eq!(store.enforce_pending_bound(), 0);
    }
}
