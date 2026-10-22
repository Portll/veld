//! Compression pipeline for memory optimization

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use lz4;
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::types::*;
use crate::constants::{
    COMPRESSION_ACCESS_THRESHOLD, COMPRESSION_AGE_DAYS, COMPRESSION_IMPORTANCE_HIGH,
    COMPRESSION_IMPORTANCE_LOW, CONSOLIDATION_CLUSTER_SIZE_CAP, CONSOLIDATION_JACCARD_THRESHOLD,
    CONSOLIDATION_MAX_CANDIDATES_PER_MEMORY, CONSOLIDATION_MIN_AGE_DAYS, CONSOLIDATION_MIN_SUPPORT,
    CONSOLIDATION_MIN_SUPPORT_LARGE, CONSOLIDATION_MIN_SUPPORT_MEDIUM,
    CONSOLIDATION_MIN_SUPPORT_SMALL, CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS,
    MAX_COMPRESSION_RATIO, MAX_DECOMPRESSED_SIZE,
};

/// Compression strategy for memories
#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    None,
    Lz4,           // Fast compression
    Summarization, // Semantic compression
    Hybrid,        // Combination of methods
}

/// Compressed memory representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    pub id: MemoryId,
    pub summary: String,
    pub keywords: Vec<String>,
    pub importance: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub compression_ratio: f32,
    pub original_size: usize,
    pub compressed_data: Vec<u8>,
    pub strategy: String,
}

/// Metadata a restore from a `compressed_data` blob removes from what it returns.
const BLOB_RESTORE_REMOVES: [&str; 3] = [
    "compressed_data",
    "compression_ratio",
    "compression_strategy",
];

/// Compression pipeline for optimizing memory storage
pub struct CompressionPipeline {
    keyword_extractor: KeywordExtractor,
}

impl Default for CompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionPipeline {
    pub fn new() -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
        }
    }

    /// Compress a memory based on its characteristics
    pub fn compress(&self, memory: &Memory) -> Result<Memory> {
        // Don't compress if already compressed or very recent
        if memory.compressed {
            return Ok(memory.clone());
        }

        let strategy = self.select_strategy(memory);

        match strategy {
            CompressionStrategy::None => Ok(memory.clone()),
            CompressionStrategy::Lz4 => self.compress_lz4(memory),
            CompressionStrategy::Summarization => self.compress_semantic(memory),
            CompressionStrategy::Hybrid => self.compress_hybrid(memory),
        }
    }

    /// Select compression strategy based on memory characteristics
    fn select_strategy(&self, memory: &Memory) -> CompressionStrategy {
        // Persistence is checked FIRST and outranks every heuristic below. Importance, age and
        // access count are all PROXIES for "is this worth keeping"; `persistent` is the caller
        // saying so directly, and a proxy must never overrule a statement. Default is true, so
        // the lossy path is now opt-out rather than the fate of anything nobody scored highly.
        if memory.persistent {
            return CompressionStrategy::Lz4;
        }

        // High importance memories get lighter compression (lossless LZ4)
        if memory.importance() > COMPRESSION_IMPORTANCE_HIGH {
            return CompressionStrategy::Lz4;
        }

        // Frequently accessed memories stay uncompressed
        if memory.access_count() > COMPRESSION_ACCESS_THRESHOLD {
            return CompressionStrategy::None;
        }

        // Old, low-importance memories get aggressive compression (lossy semantic)
        let age = chrono::Utc::now() - memory.created_at;
        if age.num_days() > COMPRESSION_AGE_DAYS && memory.importance() < COMPRESSION_IMPORTANCE_LOW
        {
            return CompressionStrategy::Summarization;
        }

        // Default to LOSSLESS. Hybrid runs compress_semantic first, so defaulting to it made
        // content destruction the fate of every ordinary memory — including ones that fell
        // through the age/importance test directly above precisely because they were NOT old
        // and low-importance. Lossy compression is the documented exception; it must be reached
        // by the branch that states its conditions, never by the fallthrough.
        CompressionStrategy::Lz4
    }

    /// Lossless compression: mark the memory compressed and keep its full experience.
    ///
    /// The bytes are saved by the storage engine's own block compression where it has one
    /// (LZ4 blocks on RocksDB). Records written before this also carried a second copy of the
    /// whole experience, embeddings included, LZ4-encoded into `compressed_data` metadata,
    /// which roughly doubled them; [`Self::decompress_lz4`] still restores those from the copy.
    fn compress_lz4(&self, memory: &Memory) -> Result<Memory> {
        let mut compressed_memory = memory.clone();
        compressed_memory.compressed = true;
        compressed_memory
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "lz4".to_string());
        Ok(compressed_memory)
    }

    /// Semantic compression - attach an additive summary WITHOUT discarding content.
    ///
    /// Ported from shodh-memory `0cefb8c` (#391). The full `experience.content` is preserved
    /// byte-for-byte; the extractive summary and keywords are stored as ADDITIONAL metadata
    /// (`summary`, `keywords`) as a retrieval-layer view, never as a replacement for the body.
    /// Semantic compression is therefore lossless; byte savings come only from the LZ4 layer.
    ///
    /// The `summary` metadata key doubles as the on-disk marker that separates new-format
    /// (content-preserving) records from legacy records whose bodies were destroyed.
    fn compress_semantic(&self, memory: &Memory) -> Result<Memory> {
        let mut compressed_memory = memory.clone();

        // Keywords and an extractive summary, both as retrieval-layer views.
        let keywords = self.keyword_extractor.extract(&memory.experience.content);
        let summary = self.create_summary(&memory.experience.content, 50);

        // Full content stays intact — the summary is additive, not a replacement.
        compressed_memory
            .experience
            .metadata
            .insert("summary".to_string(), summary);
        compressed_memory
            .experience
            .metadata
            .insert("keywords".to_string(), keywords.join(","));
        compressed_memory
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "semantic".to_string());
        compressed_memory.compressed = true;

        Ok(compressed_memory)
    }

    /// Hybrid compression - additive semantic summary + lossless LZ4 of the FULL body.
    ///
    /// The semantic step attaches summary/keywords with the body intact, then LZ4 compresses
    /// the full experience, so the original is fully recoverable through `decompress`.
    fn compress_hybrid(&self, memory: &Memory) -> Result<Memory> {
        // First attach the additive semantic view (content preserved).
        let semantic = self.compress_semantic(memory)?;

        // Then LZ4-compress the full experience.
        let mut hybrid = self.compress_lz4(&semantic)?;

        // Re-tag. compress_lz4 writes "lz4" unconditionally; before the semantic pass became
        // additive that left every hybrid record — whose original was ALREADY destroyed —
        // advertising the lossless strategy, so is_lossless() said true and decompress() handed
        // back the 50-word summary AS the original. The label has to survive the last writer,
        // and it is also what lets is_legacy_lossy() recognise those records today.
        hybrid
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "hybrid".to_string());

        Ok(hybrid)
    }

    /// Decompress a memory, restoring its full original content.
    ///
    /// Everything the current pipeline produces is lossless:
    /// - `"lz4"` / `"hybrid"` restore the full experience from the LZ4 blob.
    /// - `"semantic"` records already hold the full body in `experience.content` (the summary
    ///   is additive metadata), so the restore is in place.
    ///
    /// # Errors
    /// - The LZ4 blob is missing or corrupt, or the strategy is unknown.
    /// - The record is a legacy one whose body was destroyed at storage time (see
    ///   [`Self::is_legacy_lossy`]); that fails honestly instead of returning a summary as the
    ///   original.
    pub fn decompress(&self, memory: &Memory) -> Result<Memory> {
        if !memory.compressed {
            return Ok(memory.clone());
        }

        if Self::is_legacy_lossy(memory) {
            return Err(anyhow!(
                "Memory '{}' was written by the pre-fix lossy compressor: its full content was \
                 replaced by a summary at storage time and is not recoverable. Only the \
                 surviving summary and keywords remain in place.",
                memory.id.0
            ));
        }

        let strategy = memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        match strategy {
            // Both LZ4 and hybrid store the full experience in the LZ4 blob.
            "lz4" | "hybrid" => self.decompress_lz4(memory),
            "semantic" => Ok(Self::restore_in_place(memory)),
            unknown => Err(anyhow!(
                "Unknown compression strategy '{}' for memory '{}'. Cannot decompress.",
                unknown,
                memory.id.0
            )),
        }
    }

    /// Detect legacy records whose content was destructively replaced.
    ///
    /// The pre-fix semantic pass overwrote `experience.content` with a summary ending in `...`,
    /// wrote `keywords`, and never wrote a `summary` key. The new-format pass always writes
    /// `summary` alongside `keywords` and keeps the body. So a compressed record that carries
    /// `keywords` but no `summary`, with a `...`-terminated body, is an unrecoverable legacy
    /// record — whatever strategy label it carries. In veld those records are labelled `lz4`
    /// (the fallthrough-hybrid bug fixed in `edfb43f` re-tagged its LAST writer) as well as
    /// `hybrid` and `semantic`, so the label alone cannot identify them; the key set can.
    fn is_legacy_lossy(memory: &Memory) -> bool {
        let meta = &memory.experience.metadata;
        memory.compressed
            && meta.contains_key("keywords")
            && !meta.contains_key("summary")
            && memory.experience.content.trim_end().ends_with("...")
    }

    /// Check if a memory's compression is lossless (can be fully restored).
    ///
    /// Everything the current pipeline produces is lossless; the only lossy case that can
    /// still exist on disk is a legacy record (see [`Self::is_legacy_lossy`]).
    pub fn is_lossless(&self, memory: &Memory) -> bool {
        if !memory.compressed {
            return true;
        }
        if Self::is_legacy_lossy(memory) {
            return false;
        }
        let strategy = memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        matches!(strategy, "lz4" | "hybrid" | "semantic")
    }

    /// Get the compression strategy used for a memory
    pub fn get_strategy<'a>(&self, memory: &'a Memory) -> Option<&'a str> {
        if !memory.compressed {
            return None;
        }
        memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
    }

    /// A record that holds its full experience: clear the compression markers.
    fn restore_in_place(memory: &Memory) -> Memory {
        let mut restored = memory.clone();
        restored.compressed = false;
        restored.experience.metadata.remove("compression_ratio");
        restored.experience.metadata.remove("compression_strategy");
        restored
    }

    /// Restore an `lz4` or `hybrid` record: from its `compressed_data` copy when it was
    /// written with one, otherwise in place.
    fn decompress_lz4(&self, memory: &Memory) -> Result<Memory> {
        let Some(compressed_b64) = memory.experience.metadata.get("compressed_data") else {
            return Ok(Self::restore_in_place(memory));
        };
        let experience = Self::decode_blob(compressed_b64)?;

        // Restore the memory
        let mut restored = memory.clone();
        restored.experience = experience;
        restored.compressed = false;
        for key in BLOB_RESTORE_REMOVES {
            restored.experience.metadata.remove(key);
        }

        Ok(restored)
    }

    /// `memory` without its `compressed_data` blob and the `compression_ratio` describing
    /// it, when the blob decodes to the experience the record already holds. Then
    /// [`Self::decompress`] restores the same memory from either copy, and the blob is only a
    /// second copy of the whole experience, embeddings included, written before
    /// compress_lz4 stopped writing one.
    ///
    /// `None` when there is no blob, it will not decode, or it holds anything else: an edit
    /// made after compression, which a restore from the blob would undo, so removing the
    /// blob would change what a restore returns.
    pub fn without_redundant_blob(memory: &Memory) -> Option<Memory> {
        let blob = memory.experience.metadata.get("compressed_data")?;
        let from_blob = Self::decode_blob(blob).ok()?;
        // Compared as JSON values: metadata is a HashMap, whose encoding order differs from
        // one map to another holding the same pairs. The keys a restore drops are left out of
        // both sides, since neither copy's restore keeps them.
        let comparable = |experience: &Experience| {
            let mut experience = experience.clone();
            for key in BLOB_RESTORE_REMOVES {
                experience.metadata.remove(key);
            }
            serde_json::to_value(&experience).ok()
        };
        let blob_view = comparable(&from_blob)?;
        if Some(blob_view) != comparable(&memory.experience) {
            return None;
        }
        let mut stripped = memory.clone();
        stripped.experience.metadata.remove("compressed_data");
        stripped.experience.metadata.remove("compression_ratio");
        Some(stripped)
    }

    /// The experience a `compressed_data` blob holds, decoded under the size and ratio
    /// bounds that keep a hostile blob from expanding without limit.
    fn decode_blob(compressed_b64: &str) -> Result<Experience> {
        let compressed = general_purpose::STANDARD.decode(compressed_b64)?;

        // Zip bomb protection: Check compression ratio before decompressing
        // A small payload claiming to decompress to MAX_DECOMPRESSED_SIZE is suspicious
        let compressed_size = compressed.len();
        let max_expected_decompressed = compressed_size.saturating_mul(MAX_COMPRESSION_RATIO);

        if max_expected_decompressed > MAX_DECOMPRESSED_SIZE as usize {
            // The compressed size is so small that even at MAX_COMPRESSION_RATIO
            // it would exceed our limit - this is suspicious
            return Err(anyhow!(
                "Suspicious compression ratio: compressed size {} bytes with max ratio {} \
                 would allow {} bytes decompressed, which exceeds limit of {} bytes. \
                 This may indicate a zip bomb attack.",
                compressed_size,
                MAX_COMPRESSION_RATIO,
                max_expected_decompressed,
                MAX_DECOMPRESSED_SIZE
            ));
        }

        // Limit decompression size to prevent DoS attacks
        let decompressed = lz4::block::decompress(&compressed, Some(MAX_DECOMPRESSED_SIZE))?;

        // Post-decompression ratio check for additional safety
        let actual_ratio = decompressed.len().checked_div(compressed_size).unwrap_or(0);
        if actual_ratio > MAX_COMPRESSION_RATIO {
            return Err(anyhow!(
                "Decompression ratio {} exceeds maximum allowed ratio of {}. \
                 Compressed: {} bytes, Decompressed: {} bytes. \
                 This may indicate a zip bomb attack.",
                actual_ratio,
                MAX_COMPRESSION_RATIO,
                compressed_size,
                decompressed.len()
            ));
        }

        let (experience, _): (Experience, _) =
            bincode::serde::decode_from_slice(&decompressed, bincode::config::standard())?;
        Ok(experience)
    }

    /// Create a summary of content (extractive - takes first N words)
    fn create_summary(&self, content: &str, max_words: usize) -> String {
        // Simple extractive summary - take first N words
        // In production, this would use NLP/LLM
        let words: Vec<&str> = content.split_whitespace().collect();
        let summary_words = &words[..words.len().min(max_words)];
        format!("{}...", summary_words.join(" "))
    }
}

/// Keyword extraction for semantic compression
struct KeywordExtractor {
    stop_words: HashSet<String>,
}

impl KeywordExtractor {
    fn new() -> Self {
        let stop_words = Self::load_stop_words();
        Self { stop_words }
    }

    fn extract(&self, text: &str) -> Vec<String> {
        // Simple TF-IDF style extraction
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        for word in text.split_whitespace() {
            let clean_word = word
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();

            if clean_word.len() >= 2 && !self.stop_words.contains(&clean_word) {
                *word_freq.entry(clean_word).or_insert(0) += 1;
            }
        }

        // Sort by frequency and take top keywords
        let mut keywords: Vec<(String, usize)> = word_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));

        keywords
            .into_iter()
            .take(10)
            .map(|(word, _)| word)
            .collect()
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(word)
    }

    fn load_stop_words() -> HashSet<String> {
        let words = vec![
            "the",
            "is",
            "at",
            "which",
            "on",
            "and",
            "a",
            "an",
            "as",
            "are",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "there",
            "these",
            "those",
            "this",
            "that",
            "it",
            "its",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "where",
            "when",
            "why",
            "how",
            "all",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "but",
            "or",
            "if",
            // Pronouns and possessives
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "our",
            "their",
            // Common adverbs and adjectives that aren't meaningful entities
            "also",
            "always",
            "never",
            "often",
            "sometimes",
            "usually",
            "really",
            "actually",
            "basically",
            "currently",
            "recently",
            "already",
            "still",
            "yet",
            "now",
            "here",
            "carefully",
            "quickly",
            "easily",
            "well",
            "much",
            "many",
            "new",
            "old",
            "good",
            "great",
            "best",
            "like",
            "even",
            "also",
            "get",
            "got",
            "set",
            "use",
            "using",
            "used",
            "make",
            "made",
            "being",
        ];

        words.into_iter().map(String::from).collect()
    }
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_compressed: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub average_compression_ratio: f32,
    pub strategies_used: HashMap<String, usize>,
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            total_compressed: 0,
            total_original_size: 0,
            total_compressed_size: 0,
            average_compression_ratio: 1.0,
            strategies_used: HashMap::new(),
        }
    }
}

// ============================================================================
// SEMANTIC CONSOLIDATION - Extract durable facts from episodic memories
// ============================================================================

/// A semantic fact extracted from episodic memories
///
/// As memories age, specific episodes ("yesterday I debugged the auth module")
/// consolidate into semantic knowledge ("the auth module uses JWT tokens").
/// This mimics how human memory transitions from episodic to semantic.
///
/// # Bi-temporal validity
///
/// `valid_from` / `valid_until` track world-time validity. A fact is "currently
/// valid" when `valid_until` is `None` or in the future. Contradicting facts
/// invalidate older facts by setting `valid_until` on the old fact and recording
/// the lineage in `superseded_by` / `supersedes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Unique identifier
    pub id: String,
    /// The factual statement
    pub fact: String,
    /// Confidence in this fact (0.0 - 1.0)
    pub confidence: f32,
    /// How many episodic memories support this fact
    pub support_count: usize,
    /// Source memory IDs that contributed to this fact
    pub source_memories: Vec<MemoryId>,
    /// Keywords/entities this fact relates to
    pub related_entities: Vec<String>,
    /// When this fact was first extracted
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// When this fact was last reinforced
    pub last_reinforced: chrono::DateTime<chrono::Utc>,
    /// Category of fact (preference, capability, relationship, procedure)
    pub fact_type: FactType,
    /// When this fact became true in the world. `None` is treated as
    /// `created_at` by the store at first read. Set explicitly when the
    /// observation has a knowable start time.
    #[serde(default)]
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// When this fact stopped being true. `None` = currently valid.
    /// Set by `detect_and_resolve_contradictions` when a newer contradicting
    /// fact supersedes this one.
    #[serde(default)]
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
    /// ID of the fact that superseded this one (back-pointer for audit).
    /// Only meaningful when `valid_until` is `Some`.
    #[serde(default)]
    pub superseded_by: Option<String>,
    /// IDs of facts that this fact replaced (forward-pointer for audit).
    /// Populated on a *new* fact when it invalidates one or more older facts.
    #[serde(default)]
    pub supersedes: Vec<String>,
    /// When this fact was administratively purged. `None` = never purged.
    ///
    /// ORTHOGONAL to `valid_until` / `superseded_by` which track world-truth
    /// invalidation. `purged_at` records intentional administrative removal:
    /// pattern match, confidence floor, user request. A purged fact is hidden
    /// from all active queries (`SemanticFactStore` reader methods filter via
    /// [`is_active`]) regardless of whether it was "true" at any time.
    ///
    /// Reaper retention uses `purged_at` precedence over `valid_until`: when
    /// both are set, the purged window governs hard-delete timing.
    #[serde(default)]
    pub purged_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Why this fact was purged. `None` when `purged_at` is `None`.
    ///
    /// Internal metadata — NOT exposed to agent-facing MCP/HTTP responses.
    /// Operators reach it via audit log + direct store queries.
    #[serde(default)]
    pub purge_reason: Option<PurgeReason>,
}

impl SemanticFact {
    /// Ends belief in this fact at `at` in favour of `winner_id`. The first supersession
    /// stands: a fact already closed keeps its window and its winner, so the record says when
    /// belief stopped. The window never closes before the fact's own `valid_from`.
    pub fn supersede(&mut self, winner_id: Option<&str>, at: chrono::DateTime<chrono::Utc>) {
        if self.valid_until.is_none_or(|until| until > at) {
            self.valid_until = Some(at.max(self.valid_from.unwrap_or(self.created_at)));
        }
        if self.superseded_by.is_none() {
            self.superseded_by = winner_id.map(str::to_string);
        }
    }

    /// Records, on the fact that stays believed, a fact it displaced.
    pub fn record_superseded(&mut self, loser_id: &str) {
        if !self.supersedes.iter().any(|id| id == loser_id) {
            self.supersedes.push(loser_id.to_string());
        }
    }
}

/// Types of semantic facts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum FactType {
    /// User preference: "prefers concise code"
    Preference,
    /// System capability: "can handle 10k requests/sec"
    Capability,
    /// Relationship: "auth module depends on JWT library"
    Relationship,
    /// Procedure: "to deploy, run cargo build --release"
    Procedure,
    /// Definition: "MemoryId is a UUID wrapper"
    Definition,
    /// Pattern: "errors often occur after deployment"
    #[default]
    Pattern,
}

/// Reason a fact was administratively purged. Closed enum (no `Other` escape
/// hatch) to keep `purge_reason` non-leaky: free-form operator notes belong
/// in the audit log, not on the fact record.
#[derive(Debug, Clone, PartialEq)]
pub enum PurgeReason {
    /// Explicit user request to remove specific content.
    UserRequest,
    /// Operator-initiated cleanup not tied to a particular pattern.
    AdminCleanup,
    /// Matched a substring pattern. Pattern is stored as a SHA-256 hash to
    /// avoid leaking content into the record; the raw pattern (if forensics
    /// needs it) lives in the audit log keyed by purge event id.
    PatternMatch { pattern_hash: String },
    /// Calibrated confidence fell below the given threshold.
    ConfidenceFloor { threshold: f32 },
}

// bincode cannot decode an internally tagged enum (it needs deserialize_any), so binary
// formats get the externally tagged form. JSON keeps the `kind` tag. Records written
// in the tagged layout are read by facts.rs's LegacySemanticFactV3 arm.
#[derive(Serialize, Deserialize)]
#[serde(remote = "PurgeReason", tag = "kind", rename_all = "snake_case")]
enum PurgeReasonTagged {
    UserRequest,
    AdminCleanup,
    PatternMatch { pattern_hash: String },
    ConfidenceFloor { threshold: f32 },
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "PurgeReason", rename_all = "snake_case")]
enum PurgeReasonBinary {
    UserRequest,
    AdminCleanup,
    PatternMatch { pattern_hash: String },
    ConfidenceFloor { threshold: f32 },
}

impl Serialize for PurgeReason {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            PurgeReasonTagged::serialize(self, serializer)
        } else {
            PurgeReasonBinary::serialize(self, serializer)
        }
    }
}

impl<'de> Deserialize<'de> for PurgeReason {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            PurgeReasonTagged::deserialize(deserializer)
        } else {
            PurgeReasonBinary::deserialize(deserializer)
        }
    }
}

/// Returns `true` if the fact is currently active: NOT bi-temporally expired
/// AND NOT administratively purged. This is the universal read-path filter —
/// every `SemanticFactStore` reader method that returns "currently true" facts
/// must use this helper rather than checking the two fields independently.
///
/// Centralized so that adding a future state dimension (e.g. `quarantined_at`)
/// flows through one call site rather than 9+ enumerated reader methods.
pub fn is_active(fact: &SemanticFact, now: chrono::DateTime<chrono::Utc>) -> bool {
    if fact.purged_at.is_some() {
        return false;
    }
    match fact.valid_until {
        Some(until) => until > now,
        None => true,
    }
}

/// A cluster of related facts grouped by shared entities, with a template-
/// generated narrative and heuristic causal chain detection. Ported from
/// `shodh-memory`'s fact narrative subsystem; clustering algorithm is
/// document-frequency-based topic discovery (lowest-DF non-hub entity = topic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCluster {
    /// Most common entity in the cluster (used as topic heading)
    pub topic: String,
    /// All entities appearing in the cluster's facts (sorted, deduped)
    pub entities: Vec<String>,
    /// Facts in this cluster, sorted by created_at ascending
    pub facts: Vec<SemanticFact>,
    /// Template-generated narrative summary (no LLM required)
    pub narrative: String,
    /// Average confidence across facts in the cluster
    pub avg_confidence: f32,
    /// Sum of support_count across all facts (drives cluster ranking)
    pub total_support: usize,
    /// Detected causal relationships between facts in this cluster
    pub causal_chains: Vec<CausalFactLink>,
}

/// A heuristic-detected causal relationship between two facts.
///
/// Detected via keyword analysis on temporally ordered facts sharing entities.
/// Relations: `led_to` (default), `superseded_by` (replaced/deprecated),
/// `resolved_by` (fixed/resolved), `informed_by` (derived/based-on).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalFactLink {
    pub from_fact_id: String,
    pub to_fact_id: String,
    /// The source fact statement (snapshot at detection time)
    pub from_fact: String,
    /// The target fact statement (snapshot at detection time)
    pub to_fact: String,
    /// Relationship type
    pub relation: String,
}

/// Result of consolidation operation
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of memories processed
    pub memories_processed: usize,
    /// Memories that passed the age gate and went through extraction. Fewer than
    /// `memories_processed` means part of the corpus is still aging in and a later cycle must
    /// extract again even if nothing new is written.
    pub memories_eligible: usize,
    /// Number of new facts extracted
    pub facts_extracted: usize,
    /// Number of existing facts reinforced
    pub facts_reinforced: usize,
    /// IDs of newly created facts
    pub new_fact_ids: Vec<String>,
    /// Newly extracted semantic facts (ready for storage)
    pub new_facts: Vec<SemanticFact>,
}

/// Semantic consolidation engine
///
/// Extracts durable semantic facts from episodic memories using:
/// - Multi-extractor pipeline: all extractors run on every memory
/// - Stemmed-token Jaccard clustering: groups semantically similar patterns
/// - Sentence-level extraction: facts are real content, not synthetic strings
pub struct SemanticConsolidator {
    keyword_extractor: KeywordExtractor,
    /// Minimum times a pattern must appear to become a fact
    min_support: usize,
    /// Minimum age in days before consolidation
    min_age_days: i64,
    stemmer: Stemmer,
    /// Optional LLM hook for augmenting pattern-based facts with
    /// LLM-extracted ones. When `None`, consolidation stays fully offline
    /// (Veld's default). When set via [`SemanticConsolidator::with_llm`],
    /// callers can opt into [`SemanticConsolidator::augment_with_llm`] for
    /// the long-tail patterns the regex catalog misses.
    llm: Option<std::sync::Arc<dyn crate::memory::llm::LlmConsolidator>>,
}

/// A cluster of semantically similar pattern candidates
struct PatternCluster {
    /// Stemmed tokens of the cluster's first member; every later member is compared to these.
    stem_set: HashSet<String>,
    /// Negation polarity of the first member. A negation is often one word away from its
    /// claim and shares most of its stems, so without this a claim and its correction merge
    /// into one cluster and the correction never reaches arbitration.
    polarity: bool,
    /// All candidate entries: (pattern_text, memory_id, confidence)
    members: Vec<(String, MemoryId, f32)>,
}

impl Default for SemanticConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute heat score for a memory (A2: MemoryOS-inspired consolidation trigger)
///
/// Heat = access_count × ln(1 + context_richness) × recency_factor
///
/// - access_count: how often retrieved (proxy for engagement)
/// - context_richness: count of populated context sub-fields (proxy for information density)
/// - recency_factor: decaying boost so recent hot memories consolidate sooner
fn compute_heat_score(memory: &Memory, age_days: i64) -> f32 {
    let access_count = memory.access_count() as f32;
    let context_richness = memory.context_richness() as f32;
    let recency_factor = 1.0 / (1.0 + age_days as f32 * 0.1);
    access_count * (1.0 + context_richness).ln() * recency_factor
}

impl SemanticConsolidator {
    pub fn new() -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
            min_support: CONSOLIDATION_MIN_SUPPORT,
            min_age_days: CONSOLIDATION_MIN_AGE_DAYS,
            stemmer: Stemmer::create(Algorithm::English),
            llm: None,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(min_support: usize, min_age_days: i64) -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
            min_support,
            min_age_days,
            stemmer: Stemmer::create(Algorithm::English),
            llm: None,
        }
    }

    /// Attach an LLM hook. Caller must explicitly invoke
    /// [`SemanticConsolidator::augment_with_llm`] after [`consolidate`]
    /// to opt into LLM augmentation — `consolidate` itself stays offline.
    #[must_use]
    pub fn with_llm(
        mut self,
        llm: std::sync::Arc<dyn crate::memory::llm::LlmConsolidator>,
    ) -> Self {
        self.llm = Some(llm);
        self
    }

    /// True when an LLM hook is attached.
    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
    }

    /// Augment a pattern-based consolidation result with facts extracted by
    /// the attached LLM hook. No-op when no LLM is attached or when the
    /// memory list is empty.
    ///
    /// Strategy: send the raw content of the last N memories to the LLM
    /// and append every distinct claim it returns as a new `SemanticFact`.
    /// Distinctness is checked by case-insensitive fact-text equality
    /// against the already-extracted patterns — caller-side dedup via
    /// `SemanticFactStore::find_similar` is still recommended on store.
    pub fn augment_with_llm(
        &self,
        memories: &[Memory],
        result: &mut ConsolidationResult,
    ) -> anyhow::Result<usize> {
        let Some(llm) = self.llm.as_ref() else {
            return Ok(0);
        };
        if memories.is_empty() {
            return Ok(0);
        }

        let snippets: Vec<String> = memories
            .iter()
            .map(|m| m.experience.content.clone())
            .collect();
        let llm_facts = llm.extract_facts(&snippets)?;
        let llm_version = llm.llm_version();

        let existing: std::collections::HashSet<String> = result
            .new_facts
            .iter()
            .map(|f| f.fact.to_lowercase())
            .collect();

        let now = chrono::Utc::now();
        let mut added = 0usize;
        for llm_fact in llm_facts {
            if llm_fact.fact.trim().is_empty() || existing.contains(&llm_fact.fact.to_lowercase()) {
                continue;
            }
            let fact_type = self.classify_fact(&llm_fact.fact);
            // Provenance tag: persisted in `related_entities` with a
            // `_llm_provenance:` prefix so it survives the round-trip to
            // RocksDB without needing a SemanticFact schema bump. Reader
            // tooling can grep for the prefix to recover provenance.
            // Prefer the per-fact version when populated (in case the
            // impl tagged individual facts differently), falling back to
            // the consolidator-wide version.
            let version_tag = if llm_fact.llm_version.is_empty() {
                llm_version.clone()
            } else {
                llm_fact.llm_version.clone()
            };
            let mut related = llm_fact.entities;
            related.push(format!("_llm_provenance:{version_tag}"));
            let new_fact = SemanticFact {
                id: uuid::Uuid::new_v4().to_string(),
                fact: llm_fact.fact,
                confidence: llm_fact.confidence.clamp(0.0, 1.0),
                support_count: 1,
                source_memories: memories.iter().map(|m| m.id.clone()).collect(),
                related_entities: related,
                created_at: now,
                last_reinforced: now,
                fact_type,
                valid_from: Some(now),
                valid_until: None,
                superseded_by: None,
                supersedes: Vec::new(),
                purged_at: None,
                purge_reason: None,
            };
            result.new_fact_ids.push(new_fact.id.clone());
            result.new_facts.push(new_fact);
            result.facts_extracted += 1;
            added += 1;
        }

        tracing::info!(
            llm_name = llm.name(),
            llm_version = %llm_version,
            llm_facts_added = added,
            "augment_with_llm appended LLM-extracted facts"
        );
        Ok(added)
    }

    /// Extract semantic facts from a set of memories
    ///
    /// Pipeline:
    /// 1. Filter memories by age threshold
    /// 2. Run multi-extractor on each eligible memory
    /// 3. Cluster candidates by stemmed-token Jaccard similarity
    /// 4. Convert qualifying clusters (>= min_support) into facts
    pub fn consolidate(&self, memories: &[Memory]) -> ConsolidationResult {
        let mut result = ConsolidationResult {
            memories_processed: memories.len(),
            ..Default::default()
        };

        if memories.is_empty() {
            return result;
        }

        let now = chrono::Utc::now();
        let eligible: Vec<&Memory> = memories
            .iter()
            .filter(|m| {
                let age_days = (now - m.created_at).num_days();
                // Hard floor: never consolidate memories younger than HEAT_SCORE_MIN_AGE_DAYS
                if age_days < crate::constants::HEAT_SCORE_MIN_AGE_DAYS {
                    return false;
                }
                // Legacy gate: always consolidate memories older than min_age_days
                if age_days >= self.min_age_days {
                    return true;
                }
                // Heat-score fast-track for memories between floor and legacy gate
                compute_heat_score(m, age_days) >= crate::constants::HEAT_SCORE_THRESHOLD
            })
            .collect();

        result.memories_eligible = eligible.len();
        if eligible.is_empty() {
            return result;
        }

        let created_by_id: HashMap<&MemoryId, chrono::DateTime<chrono::Utc>> =
            eligible.iter().map(|m| (&m.id, m.created_at)).collect();
        let created_of = |id: &MemoryId| {
            created_by_id
                .get(id)
                .copied()
                .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC)
        };

        // Phase 1: Extract candidates using multi-extractor pipeline
        let mut all_candidates: Vec<(String, MemoryId, f32)> = Vec::new();
        for memory in &eligible {
            let extracted = self.extract_fact_candidates(memory);
            for (pattern, confidence) in extracted {
                all_candidates.push((pattern, memory.id.clone(), confidence));
            }
        }

        if all_candidates.is_empty() {
            return result;
        }

        // Greedy clustering depends on order, and memories arrive in storage order (random
        // ids). Ordering by evidence age, then text, makes a corpus cluster the same way every
        // time, with each cluster anchored on the earliest phrasing of its claim.
        all_candidates.sort_by(|a, b| {
            created_of(&a.1)
                .cmp(&created_of(&b.1))
                .then_with(|| a.0.cmp(&b.0))
        });

        // Phase 2: Group by stemmed-token Jaccard similarity
        let clusters =
            self.group_candidates_by_similarity(&all_candidates, CONSOLIDATION_JACCARD_THRESHOLD);

        // Phase 3: Convert qualifying clusters to facts
        let effective_min_support = if eligible.len() <= 100 {
            CONSOLIDATION_MIN_SUPPORT_SMALL
        } else if eligible.len() <= 1000 {
            CONSOLIDATION_MIN_SUPPORT_MEDIUM
        } else {
            CONSOLIDATION_MIN_SUPPORT_LARGE
        }
        .max(self.min_support);

        for cluster in clusters {
            if cluster.members.len() >= effective_min_support {
                let representative = Self::select_representative(&cluster.members);
                let avg_confidence = cluster.members.iter().map(|(_, _, c)| c).sum::<f32>()
                    / cluster.members.len() as f32;

                let source_ids: Vec<MemoryId> = cluster
                    .members
                    .iter()
                    .map(|(_, id, _)| id.clone())
                    .collect();
                let entities = self.keyword_extractor.extract(representative);
                let fact_type = self.classify_fact(representative);

                let fact = SemanticFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    fact: representative.to_string(),
                    confidence: avg_confidence.min(1.0),
                    // Cluster size sets confidence; support counts consolidation cycles that
                    // confirm the fact, so a new fact starts at one.
                    support_count: 1,
                    source_memories: source_ids,
                    related_entities: entities,
                    created_at: now,
                    last_reinforced: now,
                    fact_type,
                    valid_from: Some(now),
                    valid_until: None,
                    superseded_by: None,
                    supersedes: Vec::new(),
                    purged_at: None,
                    purge_reason: None,
                };

                result.new_fact_ids.push(fact.id.clone());
                result.new_facts.push(fact);
                result.facts_extracted += 1;
            }
        }

        // Oldest evidence first: callers ingest in this order, and arbitration gives a support
        // tie to the incoming fact, so the claim backed by the newest evidence wins, as a
        // later correction should. Text breaks ties within a cohort written at one instant.
        let newest_evidence = |fact: &SemanticFact| {
            fact.source_memories
                .iter()
                .map(&created_of)
                .max()
                .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC)
        };
        result.new_facts.sort_by(|a, b| {
            newest_evidence(a)
                .cmp(&newest_evidence(b))
                .then_with(|| a.fact.cmp(&b.fact))
        });
        result.new_fact_ids = result.new_facts.iter().map(|f| f.id.clone()).collect();

        result
    }

    // ── Clustering ──────────────────────────────────────────────────────────

    /// Tokenize text into stemmed tokens, removing stop words and punctuation
    fn stemmed_tokens(&self, text: &str) -> HashSet<String> {
        text.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|w| w.len() >= 2 && !self.keyword_extractor.is_stop_word(w))
            .map(|w| self.stemmer.stem(&w).to_string())
            .collect()
    }

    /// Jaccard similarity between two token sets: |A ∩ B| / |A ∪ B|
    fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.union(b).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Group candidates into clusters by Jaccard similarity to each cluster's first member.
    ///
    /// The comparison set never grows, so a chain of near matches cannot carry one cluster
    /// from "Rust compilation" to "code formatting". Full clusters take no more members.
    fn group_candidates_by_similarity(
        &self,
        candidates: &[(String, MemoryId, f32)],
        threshold: f32,
    ) -> Vec<PatternCluster> {
        let mut clusters: Vec<PatternCluster> = Vec::new();

        for (pattern, memory_id, confidence) in candidates {
            let tokens = self.stemmed_tokens(pattern);
            if tokens.is_empty() {
                continue;
            }
            let polarity = super::facts::detect_polarity(&pattern.to_lowercase());

            let mut best_idx = None;
            let mut best_sim = 0.0f32;
            for (i, cluster) in clusters.iter().enumerate() {
                if cluster.members.len() >= CONSOLIDATION_CLUSTER_SIZE_CAP
                    || cluster.polarity != polarity
                {
                    continue;
                }
                let sim = Self::jaccard_similarity(&tokens, &cluster.stem_set);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }

            if best_sim >= threshold {
                let idx = best_idx.unwrap();
                clusters[idx]
                    .members
                    .push((pattern.clone(), memory_id.clone(), *confidence));
            } else {
                clusters.push(PatternCluster {
                    stem_set: tokens,
                    polarity,
                    members: vec![(pattern.clone(), memory_id.clone(), *confidence)],
                });
            }
        }

        clusters
    }

    /// Select the representative fact text from a cluster (highest confidence)
    fn select_representative(members: &[(String, MemoryId, f32)]) -> &str {
        members
            .iter()
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(text, _, _)| text.as_str())
            .unwrap_or("")
    }

    // ── Multi-Extractor Pipeline ────────────────────────────────────────────

    /// Extract fact candidates from a single memory.
    ///
    /// Operational traces (Context, Command, CodeEdit, FileAccess, Search) produce no
    /// candidates. The four specialist extractors each look for one rhetorical shape and run
    /// only on the experience types that carry it. The declarative extractor runs on every
    /// other type: a plain factual sentence has no marker to gate on and belongs to no single
    /// type. Importance scales a candidate's confidence, not whether it exists; corroboration
    /// across distinct memories decides what becomes a fact. Every candidate must pass
    /// `is_fact_shaped`.
    fn extract_fact_candidates(&self, memory: &Memory) -> Vec<(String, f32)> {
        let mut candidates = Vec::new();
        let content = &memory.experience.content;
        let importance = memory.importance();
        let exp_type = &memory.experience.experience_type;

        if matches!(
            exp_type,
            ExperienceType::Context
                | ExperienceType::Command
                | ExperienceType::CodeEdit
                | ExperienceType::FileAccess
                | ExperienceType::Search
        ) {
            return candidates;
        }

        if matches!(
            exp_type,
            ExperienceType::Decision | ExperienceType::Learning | ExperienceType::Task
        ) {
            if let Some(fact) = self.extract_procedure(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Decision {
                        1.0
                    } else {
                        0.7
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        if matches!(
            exp_type,
            ExperienceType::Learning | ExperienceType::Discovery | ExperienceType::Pattern
        ) {
            if let Some(fact) = self.extract_definition(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Learning
                        || *exp_type == ExperienceType::Discovery
                    {
                        1.2
                    } else {
                        0.8
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        if matches!(
            exp_type,
            ExperienceType::Error
                | ExperienceType::Learning
                | ExperienceType::Discovery
                | ExperienceType::Pattern
        ) {
            if let Some(fact) = self.extract_pattern(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Error {
                        1.1
                    } else {
                        0.7
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        if *exp_type == ExperienceType::Decision
            || (*exp_type == ExperienceType::Conversation && importance >= 0.5)
        {
            if let Some(fact) = self.extract_preference(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Conversation {
                        1.0
                    } else {
                        0.6
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        if let Some(fact) = self.extract_salient_statement(content, &memory.experience.entities) {
            if Self::is_fact_shaped(&fact) {
                candidates.push((fact, importance * 0.6));
            }
        }

        candidates.retain(|(text, _)| Self::is_knowledge_worthy(text));

        // Deduplicate overlapping extractions from the same memory
        candidates = self.dedup_within_memory(candidates);

        // Cap per-memory candidates
        candidates.truncate(CONSOLIDATION_MAX_CANDIDATES_PER_MEMORY);

        candidates
    }

    /// Remove overlapping extractions from the same memory (keep highest confidence)
    fn dedup_within_memory(&self, mut candidates: Vec<(String, f32)>) -> Vec<(String, f32)> {
        if candidates.len() <= 1 {
            return candidates;
        }

        // Sort by confidence descending so we keep higher-confidence versions
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut kept: Vec<(String, f32, HashSet<String>)> = Vec::new();
        for (text, conf) in candidates {
            let tokens = self.stemmed_tokens(&text);
            let overlaps = kept
                .iter()
                .any(|(_, _, t)| Self::jaccard_similarity(&tokens, t) > 0.8);
            if !overlaps {
                kept.push((text, conf, tokens));
            }
        }

        kept.into_iter()
            .map(|(text, conf, _)| (text, conf))
            .collect()
    }

    // ── Individual Extractors ───────────────────────────────────────────────

    /// Extract a procedure: an action word at the start of a sentence or after instructional
    /// punctuation ("Step 1: use cargo build"), not one inside prose ("we use Rust").
    fn extract_procedure(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        // to_lowercase() can change byte lengths for non-ASCII chars (e.g. İ→i̇),
        // making byte offsets from `lower.find()` invalid for indexing into `content`.
        let lower = content.to_ascii_lowercase();
        let action_markers = [
            "run ",
            "execute ",
            "use ",
            "call ",
            "invoke ",
            "create ",
            "build ",
            "deploy ",
            "install ",
            "configure ",
            "start ",
            "stop ",
            "restart ",
            "set up ",
            "update ",
            "remove ",
            "delete ",
            "add ",
            "import ",
            "export ",
            "migrate ",
        ];

        for marker in action_markers {
            if let Some(pos) = lower.find(marker) {
                if pos > 0 {
                    let prev_char = content[..pos].trim_end().chars().last().unwrap_or(' ');
                    if !matches!(prev_char, '.' | ':' | '—' | '-' | '\n' | '!' | '•' | '*') {
                        continue;
                    }
                }
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// Extract a definition. The subject must be a name (capitalised, or carrying `_` or `.`)
    /// of three or more characters and not a pronoun, and the definition ten or more.
    fn extract_definition(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();
        let def_markers = [
            " is ",
            " are ",
            " means ",
            " refers to ",
            " represents ",
            " denotes ",
            " describes ",
            " consists of ",
            " defined as ",
            " known as ",
            " stands for ",
            " equivalent to ",
        ];
        const PRONOUN_SUBJECTS: &[&str] = &[
            "it", "this", "that", "there", "here", "which", "what", "they", "these", "those",
        ];

        for marker in def_markers {
            if let Some(pos) = lower.find(marker) {
                let subject_start = content[..pos]
                    .char_indices()
                    .rev()
                    .find(|(_, c)| !c.is_alphanumeric() && *c != '_')
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(0);
                let subject = &content[subject_start..pos];

                if subject.len() < 3
                    || PRONOUN_SUBJECTS
                        .iter()
                        .any(|p| subject.eq_ignore_ascii_case(p))
                {
                    continue;
                }
                let first_char = subject.chars().next().unwrap_or(' ');
                if !first_char.is_uppercase() && !subject.contains('_') && !subject.contains('.') {
                    continue;
                }

                let def_end = content[pos + marker.len()..].find(['.', '!', '?', ',']);
                let def_end = def_end.map(|i| pos + marker.len() + i).unwrap_or_else(|| {
                    let cap = content.len().min(pos + marker.len() + 100);
                    content.floor_char_boundary(cap)
                });

                let definition = &content[pos + marker.len()..def_end];
                if definition.trim().len() >= 10 {
                    return Some(format!("{}{}{}", subject, marker, definition.trim()));
                }
            }
        }
        None
    }

    /// Extract a pattern from error content (returns the actual sentence)
    fn extract_pattern(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();
        let pattern_markers = [
            "error",
            "failed",
            "crash",
            "bug",
            "issue",
            "problem",
            "exception",
            "warning",
            "panic",
            "timeout",
            "overflow",
            "deadlock",
            "leak",
            "corrupt",
        ];

        for marker in pattern_markers {
            if let Some(pos) = lower.find(marker) {
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// Extract a preference: "always"/"never" anywhere, other markers only with a first-person
    /// subject shortly before them, and "like" never as a simile ("looks like").
    fn extract_preference(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();
        const IMPERATIVE_MARKERS: &[&str] = &["always", "never"];
        const SUBJECT_MARKERS: &[&str] = &[
            "prefer",
            "like",
            "want",
            "better",
            "should",
            "dislike",
            "avoid",
            "recommend",
            "favorite",
            "rather",
            "instead of",
            "opt for",
        ];
        const FIRST_PERSON: &[&str] = &["i ", "we ", "my ", "our ", "user "];
        const SIMILE_PREFIXES: &[&str] = &["looks ", "seems ", "feels ", "sounds ", "acts "];

        for marker in IMPERATIVE_MARKERS {
            if let Some(pos) = lower.find(marker) {
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }

        for marker in SUBJECT_MARKERS {
            if let Some(pos) = lower.find(marker) {
                if *marker == "like" && SIMILE_PREFIXES.iter().any(|p| lower[..pos].ends_with(p)) {
                    continue;
                }
                let lookback_start = pos.saturating_sub(30);
                let before_marker = &lower[lookback_start..pos];
                if FIRST_PERSON.iter().any(|s| before_marker.contains(s)) {
                    if let Some(sentence) = Self::extract_sentence(content, pos) {
                        return Some(sentence);
                    }
                }
            }
        }
        None
    }

    /// Extract the most information-dense declarative sentence.
    ///
    /// Entity mentions rank sentences; they do not gate them. `entities` comes from NER, which
    /// is empty for whole classes of records, and a gate on it would extract nothing whenever
    /// NER is silent.
    fn extract_salient_statement(&self, content: &str, entities: &[String]) -> Option<String> {
        let sentences = Self::split_sentences(content);
        let entity_lower: Vec<String> = entities.iter().map(|e| e.to_lowercase()).collect();

        let mut best: Option<(String, f32)> = None;

        for sentence in sentences {
            let trimmed = sentence.trim();
            if trimmed.len() < 20 || trimmed.len() > 200 {
                continue;
            }

            let lower = trimmed.to_lowercase();

            // Score: count of non-stop-words (content density)
            let content_words: usize = lower
                .split_whitespace()
                .map(|w| {
                    w.chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect::<String>()
                })
                .filter(|w| !w.is_empty() && !self.keyword_extractor.is_stop_word(w))
                .count();

            if content_words < CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS {
                continue;
            }

            let entity_match_count = entity_lower
                .iter()
                .filter(|e| lower.contains(e.as_str()))
                .count();
            let entity_bonus = entity_match_count as f32 * 2.0;
            let score = content_words as f32 + entity_bonus;

            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((trimmed.to_string(), score));
            }
        }

        best.map(|(s, _)| s)
    }

    // ── Structural Gate ─────────────────────────────────────────────────────

    /// Rejects text that is structurally not a statement of fact: questions, truncated
    /// fragments, mid-sentence fragments, code or config, log lines and path listings.
    fn is_fact_shaped(text: &str) -> bool {
        let trimmed = text.trim();

        if trimmed.ends_with('?') || trimmed.ends_with("..") {
            return false;
        }

        if let Some(first) = trimmed.chars().next() {
            if first.is_lowercase() {
                const LOWERCASE_STARTERS: &[&str] = &[
                    "the ", "a ", "an ", "in ", "on ", "at ", "for ", "to ", "if ", "when ",
                    "but ", "and ", "or ", "so ", "as ", "by ", "with ",
                ];
                let lower_start = trimmed.to_ascii_lowercase();
                if !LOWERCASE_STARTERS
                    .iter()
                    .any(|s| lower_start.starts_with(s))
                {
                    return false;
                }
            }
        }

        let special_count = trimmed
            .chars()
            .filter(|c| {
                matches!(
                    c,
                    '{' | '}'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | ';'
                        | '='
                        | '<'
                        | '>'
                        | '|'
                        | '&'
                        | '#'
                        | '$'
                        | '`'
                        | '\\'
                )
            })
            .count();
        let alpha_count = trimmed.chars().filter(|c| c.is_alphanumeric()).count();
        if alpha_count > 0 && special_count as f32 / alpha_count as f32 > 0.25 {
            return false;
        }

        const LOG_PATTERNS: &[&str] = &[
            "[info]",
            "[warn]",
            "[error]",
            "[debug]",
            "[trace]",
            "info:",
            "warn:",
            "error:",
            "debug:",
            "trace:",
            " at 0x",
            "thread '",
            "stack trace",
        ];
        let lower = trimmed.to_ascii_lowercase();
        if LOG_PATTERNS.iter().any(|p| lower.contains(p)) {
            return false;
        }

        let slash_count = trimmed.chars().filter(|c| *c == '/' || *c == '\\').count();
        if slash_count >= 2 && trimmed.split_whitespace().count() < 8 {
            return false;
        }

        true
    }

    // ── Quality Gate ────────────────────────────────────────────────────────

    /// Rejects operational noise: session lifecycle, tool invocations, todo status chatter
    /// and bare file paths are not domain knowledge.
    fn is_knowledge_worthy(text: &str) -> bool {
        let lower = text.to_lowercase();
        if text.trim().len() < 25 {
            return false;
        }

        const SESSION_NOISE: &[&str] = &[
            "session started",
            "session ended",
            "session summary",
            "context compressed",
            "context window",
            "token budget",
            "tokens used",
            "proactive_context",
            "auto_ingest",
            "memory surfaced",
            "memories surfaced",
            "memory stored",
            "memories stored",
        ];
        const TOOL_NOISE: &[&str] = &[
            "tool call",
            "tool_name",
            "mcp tool",
            "called remember",
            "called recall",
            "called forget",
            "hook triggered",
            "hook fired",
        ];
        const TODO_NOISE: &[&str] = &[
            "todo created",
            "todo completed",
            "todo updated",
            "task created",
            "task completed",
            "task updated",
            "marked as done",
            "marked as complete",
            "moved to backlog",
            "moved to in_progress",
        ];
        // Protocol output that reaches memories verbatim: task notifications, agent output
        // tags, crash-recovery lines.
        const SYSTEM_OUTPUT_NOISE: &[&str] = &[
            "<task-notification>",
            "</output-file>",
            "<status>killed</status>",
            "<status>completed</status>",
            "[assistant: codeedit]",
            "[assistant: observation]",
            "[assistant: search]",
            "[assistant: decision]",
            "[assistant: pattern]",
            "reconnected after",
            "oom crash",
            "surfaced 5 memories",
            "surfaced 3 memories",
            "agent \"",
        ];
        if [SESSION_NOISE, TOOL_NOISE, TODO_NOISE, SYSTEM_OUTPUT_NOISE]
            .iter()
            .any(|list| list.iter().any(|s| lower.contains(s)))
        {
            return false;
        }

        // A short text with a path separator is a file reference, not a statement.
        let path_chars = lower.chars().filter(|c| *c == '/' || *c == '\\').count();
        if path_chars >= 1 && text.split_whitespace().count() < 6 {
            return false;
        }

        true
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Extract the sentence containing a character position
    fn extract_sentence(content: &str, pos: usize) -> Option<String> {
        let start = content[..pos].rfind(['.', '!', '?']);
        let start = start.map(|i| i + 1).unwrap_or(0);

        let end = content[pos..].find(['.', '!', '?']);
        let end = end.map(|i| pos + i).unwrap_or(content.len());

        let sentence = content[start..end].trim();
        if sentence.len() >= 20 && sentence.len() < 200 {
            Some(sentence.to_string())
        } else {
            None
        }
    }

    /// Split content into sentences by sentence-ending punctuation
    fn split_sentences(content: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in content.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let sentence = &content[start..i];
                if !sentence.trim().is_empty() {
                    sentences.push(sentence.trim());
                }
                start = i + c.len_utf8();
            }
        }

        // Trailing content without sentence-ending punctuation
        let remaining = content[start..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }

    /// Classify what type of fact this is
    fn classify_fact(&self, pattern: &str) -> FactType {
        let lower = pattern.to_lowercase();

        if lower.contains("prefer")
            || lower.contains("like")
            || lower.contains("always")
            || lower.contains("never")
            || lower.contains("favorite")
        {
            FactType::Preference
        } else if lower.contains("can ") || lower.contains("able to") || lower.contains("supports")
        {
            FactType::Capability
        } else if lower.contains("relates to")
            || lower.contains("depends on")
            || lower.contains("connects")
        {
            FactType::Relationship
        } else if lower.contains("to ")
            || lower.contains("run ")
            || lower.contains("execute")
            || lower.contains("deploy")
        {
            FactType::Procedure
        } else if lower.contains(" is ") || lower.contains(" are ") || lower.contains("means") {
            FactType::Definition
        } else {
            FactType::Pattern
        }
    }

    /// Reinforce an existing fact with new evidence
    ///
    /// Called when a memory matches an existing fact, strengthening confidence.
    pub fn reinforce_fact(&self, fact: &mut SemanticFact, memory: &Memory) {
        fact.support_count += 1;
        fact.last_reinforced = chrono::Utc::now();

        // Increase confidence with diminishing returns
        let boost = 0.1 * (1.0 - fact.confidence);
        fact.confidence = (fact.confidence + boost).min(1.0);

        // Add source if not already present
        if !fact.source_memories.contains(&memory.id) {
            fact.source_memories.push(memory.id.clone());
        }

        // Add any new entities (filter noise: short tokens, stop words)
        for entity in &memory.experience.entities {
            let lower = entity.to_lowercase();
            if lower.len() >= 3
                && !self.keyword_extractor.is_stop_word(&lower)
                && !fact.related_entities.contains(entity)
            {
                fact.related_entities.push(entity.clone());
            }
        }
    }

    /// Check if a fact would decay below deletion threshold (0.1 confidence)
    ///
    /// Uses exponential half-life model matching `decay_facts_for_all_users()`:
    /// 90-day grace period, then half-life = 180 + (30 × support_count) days.
    pub fn should_decay_fact(&self, fact: &SemanticFact) -> bool {
        use crate::constants::{
            FACT_DECAY_GRACE_DAYS, FACT_DECAY_HALF_LIFE_BASE_DAYS,
            FACT_DECAY_HALF_LIFE_PER_SUPPORT_DAYS,
        };
        let days_since = (chrono::Utc::now() - fact.last_reinforced).num_days();
        if days_since <= FACT_DECAY_GRACE_DAYS {
            return false;
        }
        let elapsed = (days_since - FACT_DECAY_GRACE_DAYS) as f64;
        let half_life = FACT_DECAY_HALF_LIFE_BASE_DAYS
            + (fact.support_count as f64 * FACT_DECAY_HALF_LIFE_PER_SUPPORT_DAYS);
        let projected = fact.confidence * (0.5_f64).powf(elapsed / half_life) as f32;
        projected < 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn purge_reason_round_trips_through_bincode_and_json() {
        let reasons = [
            PurgeReason::UserRequest,
            PurgeReason::AdminCleanup,
            PurgeReason::PatternMatch {
                pattern_hash: "abc123".to_string(),
            },
            PurgeReason::ConfidenceFloor { threshold: 0.25 },
        ];
        for reason in reasons {
            let wrapped = (Some(reason.clone()), 7u32);
            let bytes = bincode::serde::encode_to_vec(&wrapped, bincode::config::standard())
                .expect("encode");
            let (back, read): ((Option<PurgeReason>, u32), _) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                    .expect("bincode decode");
            assert_eq!(back, wrapped, "bincode round-trip of {reason:?}");
            assert_eq!(read, bytes.len(), "bincode must consume the whole record");

            let json = serde_json::to_value(&reason).expect("json encode");
            assert!(json.get("kind").is_some(), "JSON stays internally tagged");
            let back: PurgeReason = serde_json::from_value(json).expect("json decode");
            assert_eq!(back, reason);
        }
    }

    #[test]
    fn purge_reason_json_keeps_its_kind_tag() {
        let json = serde_json::to_value(PurgeReason::PatternMatch {
            pattern_hash: "abc".to_string(),
        })
        .unwrap();
        assert_eq!(
            json,
            serde_json::json!({"kind": "pattern_match", "pattern_hash": "abc"})
        );
        assert_eq!(
            serde_json::to_value(PurgeReason::UserRequest).unwrap(),
            serde_json::json!({"kind": "user_request"})
        );
    }

    fn create_test_memory(content: &str, importance: f32) -> Memory {
        let experience = Experience {
            content: content.to_string(),
            experience_type: ExperienceType::Observation,
            entities: vec!["test".to_string()],
            ..Default::default()
        };

        let created_at = Some(chrono::Utc::now() - chrono::Duration::days(60));

        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            importance,
            None, // agent_id
            None, // run_id
            None, // actor_id
            created_at,
        )
    }

    #[test]
    fn test_compression_pipeline_default() {
        let pipeline = CompressionPipeline::default();
        assert!(pipeline.keyword_extractor.stop_words.contains("the"));
    }

    #[test]
    fn test_lz4_compress_decompress() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("This is a test memory content for compression", 0.9);

        let compressed = pipeline.compress(&memory).unwrap();
        assert!(compressed.compressed);
        assert_eq!(
            compressed
                .experience
                .metadata
                .get("compression_strategy")
                .unwrap(),
            "lz4"
        );

        let decompressed = pipeline.decompress(&compressed).unwrap();
        assert!(!decompressed.compressed);
        assert_eq!(decompressed.experience.content, memory.experience.content);
    }

    /// The `compressed_data` value records carried before compress_lz4 stopped writing a
    /// second copy: base64 of the LZ4 block of the bincode experience.
    fn legacy_compressed_data(memory: &Memory) -> String {
        let encoded =
            bincode::serde::encode_to_vec(&memory.experience, bincode::config::standard()).unwrap();
        general_purpose::STANDARD.encode(lz4::block::compress(&encoded, None, false).unwrap())
    }

    /// The old blob copied the whole experience, embeddings included, next to the original,
    /// so a "compressed" record was about twice its uncompressed size.
    #[test]
    fn lz4_compression_keeps_one_copy_of_the_experience() {
        let pipeline = CompressionPipeline::new();
        let mut memory = create_test_memory(&make_body(200), 0.9);
        memory.experience.embeddings = Some(vec![0.25; 384]);
        let before = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
            .unwrap()
            .len();

        let compressed = pipeline.compress(&memory).unwrap();
        let after = bincode::serde::encode_to_vec(&compressed, bincode::config::standard())
            .unwrap()
            .len();

        assert!(!compressed
            .experience
            .metadata
            .contains_key("compressed_data"));
        assert!(
            after < before + 64,
            "compression grew the record from {before} to {after} bytes"
        );
        let restored = pipeline.decompress(&compressed).unwrap();
        assert!(!restored.compressed);
        assert_eq!(restored.experience.content, memory.experience.content);
        assert_eq!(restored.experience.embeddings, memory.experience.embeddings);
        assert!(!restored
            .experience
            .metadata
            .contains_key("compression_strategy"));
    }

    /// Records already on disk carry the blob; they still decompress from it.
    #[test]
    fn a_record_written_with_the_old_blob_still_decompresses_from_it() {
        let pipeline = CompressionPipeline::new();
        let original = create_test_memory("the body the blob was taken from", 0.9);
        let mut stored = original.clone();
        stored.compressed = true;
        stored.experience.content = "stale copy that the blob overrides".to_string();
        let meta = &mut stored.experience.metadata;
        meta.insert("compression_strategy".to_string(), "lz4".to_string());
        meta.insert("compression_ratio".to_string(), "0.5".to_string());
        meta.insert(
            "compressed_data".to_string(),
            legacy_compressed_data(&original),
        );

        let restored = pipeline.decompress(&stored).unwrap();
        assert!(!restored.compressed);
        assert_eq!(restored.experience.content, original.experience.content);
        assert!(!restored.experience.metadata.contains_key("compressed_data"));
        assert!(!restored
            .experience
            .metadata
            .contains_key("compression_ratio"));
    }

    /// A record as compress_lz4 wrote it before 0e79f6c3: the experience, plus its blob.
    fn legacy_lz4_record(original: &Memory) -> Memory {
        let mut stored = original.clone();
        stored.compressed = true;
        let blob = legacy_compressed_data(original);
        let meta = &mut stored.experience.metadata;
        meta.insert("compressed_data".to_string(), blob);
        meta.insert("compression_ratio".to_string(), "0.41".to_string());
        meta.insert("compression_strategy".to_string(), "lz4".to_string());
        stored
    }

    fn json(memory: &Memory) -> serde_json::Value {
        serde_json::to_value(memory).unwrap()
    }

    /// A blob that copies the record's experience goes, and the record restores to exactly
    /// what it restored to from the blob.
    #[test]
    fn a_blob_that_copies_the_record_is_stripped_and_the_restore_is_unchanged() {
        let pipeline = CompressionPipeline::new();
        let mut original = create_test_memory(&make_body(120), 0.9);
        original.experience.embeddings = Some(vec![0.125; 384]);
        let stored = legacy_lz4_record(&original);

        let stripped = CompressionPipeline::without_redundant_blob(&stored).expect("redundant");
        let meta = &stripped.experience.metadata;
        assert!(!meta.contains_key("compressed_data"));
        assert!(!meta.contains_key("compression_ratio"));
        assert_eq!(
            meta.get("compression_strategy").map(String::as_str),
            Some("lz4")
        );
        assert!(stripped.compressed);
        assert_eq!(
            json(&pipeline.decompress(&stripped).unwrap()),
            json(&pipeline.decompress(&stored).unwrap())
        );
        assert!(
            CompressionPipeline::without_redundant_blob(&stripped).is_none(),
            "nothing left to strip"
        );
    }

    /// The semantic step ran before the blob was taken, so a hybrid blob carries the summary,
    /// the keywords and a `semantic` label the record later relabelled.
    #[test]
    fn a_hybrid_blob_that_copies_the_record_is_stripped() {
        let pipeline = CompressionPipeline::new();
        let original = create_test_memory(&make_body(80), 0.2);
        let mut semantic = original.clone();
        let meta = &mut semantic.experience.metadata;
        meta.insert("summary".to_string(), "a summary...".to_string());
        meta.insert("keywords".to_string(), "alpha,beta".to_string());
        meta.insert("compression_strategy".to_string(), "semantic".to_string());
        let mut stored = legacy_lz4_record(&semantic);
        stored
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "hybrid".to_string());

        let stripped = CompressionPipeline::without_redundant_blob(&stored).expect("redundant");
        assert_eq!(
            json(&pipeline.decompress(&stripped).unwrap()),
            json(&pipeline.decompress(&stored).unwrap())
        );
    }

    /// A record edited after compression restores from its blob to the version before the
    /// edit; stripping would change that, so the blob stays.
    #[test]
    fn a_blob_that_differs_from_the_record_is_kept() {
        let original = create_test_memory("the body when it was compressed", 0.9);
        let mut stored = legacy_lz4_record(&original);
        stored.experience.content = "the body as edited afterwards".to_string();
        assert!(CompressionPipeline::without_redundant_blob(&stored).is_none());

        let mut tagged = legacy_lz4_record(&original);
        tagged.experience.tags.push("added-later".to_string());
        assert!(CompressionPipeline::without_redundant_blob(&tagged).is_none());

        let mut corrupt = legacy_lz4_record(&original);
        corrupt.experience.metadata.insert(
            "compressed_data".to_string(),
            "not base64 at all!".to_string(),
        );
        assert!(CompressionPipeline::without_redundant_blob(&corrupt).is_none());

        assert!(CompressionPipeline::without_redundant_blob(&original).is_none());
    }

    #[test]
    fn test_already_compressed_memory() {
        let pipeline = CompressionPipeline::new();
        let mut memory = create_test_memory("Test content", 0.9);
        memory.compressed = true;

        let result = pipeline.compress(&memory).unwrap();
        assert!(result.compressed);
    }

    /// Build a deterministic multi-word body of the requested length.
    fn make_body(words: usize) -> String {
        (0..words)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Ported from shodh-memory `0cefb8c`: the semantic pass is additive. The body survives
    /// byte-for-byte and the summary rides alongside as metadata.
    #[test]
    fn semantic_compression_preserves_full_content() {
        let pipeline = CompressionPipeline::new();
        let body = make_body(300);
        let memory = create_test_memory(&body, 0.1);

        let compressed = pipeline.compress_semantic(&memory).unwrap();
        assert!(compressed.compressed);
        assert_eq!(compressed.experience.content, body);
        assert!(compressed.experience.metadata.contains_key("summary"));
        assert!(compressed.experience.metadata.contains_key("keywords"));

        let restored = pipeline.decompress(&compressed).unwrap();
        assert!(!restored.compressed);
        assert_eq!(restored.experience.content, body);
    }

    /// The hybrid label still has to survive the last writer (compress_lz4 writes "lz4"), and a
    /// hybrid record is now lossless because the semantic pass no longer destroys the body.
    #[test]
    fn test_hybrid_is_labelled_hybrid_not_lz4() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory(
            "A long-form document whose whole value is that every word of it survives storage",
            0.3,
        );

        let hybrid = pipeline.compress_hybrid(&memory).unwrap();
        assert_eq!(pipeline.get_strategy(&hybrid), Some("hybrid"));
        assert!(pipeline.is_lossless(&hybrid));
    }

    /// Decompressing a hybrid record returns every byte of the original, not the summary.
    #[test]
    fn hybrid_compression_round_trips_full_content() {
        let pipeline = CompressionPipeline::new();
        let body = make_body(300);
        let memory = create_test_memory(&body, 0.6);

        let hybrid = pipeline.compress_hybrid(&memory).unwrap();
        let restored = pipeline.decompress(&hybrid).unwrap();
        assert!(!restored.compressed);
        assert_eq!(restored.experience.content, body);
    }

    /// The exact entrypoint tier promotion invokes is compress(); drive it end to end for the
    /// two strategies that used to truncate (persistence waived so the lossy branches are reachable).
    #[test]
    fn promotion_to_longterm_never_truncates() {
        let pipeline = CompressionPipeline::new();
        let body = make_body(300);

        let mut low = create_test_memory(&body, 0.1);
        low.persistent = false;
        low.created_at = chrono::Utc::now() - chrono::Duration::days(100);
        let c_low = pipeline.compress(&low).unwrap();
        assert_eq!(pipeline.get_strategy(&c_low), Some("semantic"));
        assert_eq!(c_low.experience.content, body);
        assert_eq!(
            pipeline.decompress(&c_low).unwrap().experience.content,
            body
        );
    }

    /// A record the pre-fix compressor wrote: body replaced by a `...` summary, `keywords` present,
    /// no `summary` key. In veld those carry the "lz4" label (the last-writer bug), so the
    /// detector must key on the metadata shape, not the label — and must refuse to decompress
    /// them into a confident wrong answer.
    #[test]
    fn legacy_lossy_record_is_detected_under_any_label_and_refuses_to_decompress() {
        let pipeline = CompressionPipeline::new();
        for label in ["lz4", "hybrid", "semantic"] {
            let mut legacy =
                create_test_memory("first fifty words of a document that is gone...", 0.3);
            legacy.compressed = true;
            let meta = &mut legacy.experience.metadata;
            meta.insert("keywords".to_string(), "document,gone".to_string());
            meta.insert("compression_strategy".to_string(), label.to_string());
            if label != "semantic" {
                // A blob is present on the old lz4/hybrid records; it holds the summary experience.
                let data = legacy_compressed_data(&create_test_memory(
                    "first fifty words of a document that is gone...",
                    0.3,
                ));
                legacy
                    .experience
                    .metadata
                    .insert("compressed_data".to_string(), data);
            }

            assert!(
                !pipeline.is_lossless(&legacy),
                "label {label}: a destroyed body is not lossless"
            );
            let err = pipeline.decompress(&legacy).unwrap_err().to_string();
            assert!(err.contains("not recoverable"), "label {label}: got {err}");
        }
    }

    /// The detector must not fire on an intact record whose content merely ends with "...".
    #[test]
    fn intact_lz4_record_ending_in_ellipsis_is_not_legacy() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("To be continued...", 0.9);
        let compressed = pipeline.compress_lz4(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed));
        assert_eq!(
            pipeline.decompress(&compressed).unwrap().experience.content,
            "To be continued..."
        );
    }

    /// The fallthrough default decided the fate of every ordinary memory. Hybrid meant lossy.
    #[test]
    fn test_default_strategy_is_lossless() {
        let pipeline = CompressionPipeline::new();
        // Recent and mid-importance: misses the lz4 branch, the access branch and the
        // old-AND-low-importance branch, so it lands on the default.
        let memory = create_test_memory(
            "Content that must survive: not old, not unimportant, not frequently accessed",
            0.6,
        );

        assert!(matches!(
            pipeline.select_strategy(&memory),
            CompressionStrategy::Lz4
        ));

        let compressed = pipeline.compress(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed));

        let restored = pipeline.decompress(&compressed).unwrap();
        assert_eq!(
            restored.experience.content, memory.experience.content,
            "a default-path round trip must return every byte it was given"
        );
    }

    /// The exception still has to work — but ONLY once persistence is waived.
    #[test]
    fn test_old_low_importance_selects_lossy_only_when_persistence_is_waived() {
        let pipeline = CompressionPipeline::new();
        let mut memory = create_test_memory("Old and unimportant content here", 0.1);
        memory.created_at = chrono::Utc::now() - chrono::Duration::days(100);

        // Default: persistent, so even old + unimportant stays lossless.
        assert!(memory.persistent, "persistence defaults ON");
        assert!(matches!(
            pipeline.select_strategy(&memory),
            CompressionStrategy::Lz4
        ));

        memory.persistent = false;
        assert!(matches!(
            pipeline.select_strategy(&memory),
            CompressionStrategy::Summarization
        ));
    }

    /// Persistence outranks the heuristics. Importance/age/access are proxies for "worth keeping";
    /// the flag is the caller saying it outright, and a proxy must not overrule a statement.
    #[test]
    fn test_persistent_outranks_every_lossy_heuristic() {
        let pipeline = CompressionPipeline::new();
        let original = "A document whose entire value is that every word of it survives storage \
                        long past the point where any heuristic would still rate it interesting";
        let mut memory = create_test_memory(original, 0.0);
        memory.created_at = chrono::Utc::now() - chrono::Duration::days(3650);

        assert!(matches!(
            pipeline.select_strategy(&memory),
            CompressionStrategy::Lz4
        ));

        let compressed = pipeline.compress(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed));
        let restored = pipeline.decompress(&compressed).unwrap();
        assert_eq!(restored.experience.content, original);
    }

    /// A record serialized before the flag existed must come back PERSISTENT, not lossy-eligible.
    #[test]
    fn test_absent_persistent_flag_deserializes_as_true() {
        let memory = create_test_memory("Written before the flag existed", 0.1);
        let mut json: serde_json::Value = serde_json::to_value(&memory).unwrap();
        json.as_object_mut().unwrap().remove("persistent");
        assert!(
            json.get("persistent").is_none(),
            "field removed for the test"
        );

        let restored: Memory = serde_json::from_value(json).unwrap();
        assert!(
            restored.persistent,
            "absent must mean persistent — false-by-omission would make every legacy record lossy-eligible"
        );
    }

    #[test]
    fn test_is_lossless() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("Test content", 0.9);

        assert!(pipeline.is_lossless(&memory));

        let compressed_lz4 = pipeline.compress_lz4(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed_lz4));

        // Semantic compression keeps the body; the summary is additive.
        let compressed_semantic = pipeline.compress_semantic(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed_semantic));
    }

    #[test]
    fn test_get_strategy() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("Test content", 0.9);

        assert!(pipeline.get_strategy(&memory).is_none());

        let compressed = pipeline.compress_lz4(&memory).unwrap();
        assert_eq!(pipeline.get_strategy(&compressed), Some("lz4"));
    }

    #[test]
    fn test_keyword_extraction() {
        let extractor = KeywordExtractor::new();
        let text = "Rust programming language memory management ownership borrowing";
        let keywords = extractor.extract(text);

        assert!(!keywords.is_empty());
        assert!(keywords.contains(&"rust".to_string()));
        assert!(keywords.contains(&"memory".to_string()));
        assert!(!keywords.contains(&"the".to_string()));
    }

    #[test]
    fn test_stop_words_filtered() {
        let extractor = KeywordExtractor::new();
        let text = "the is at which on and a an as are was were";
        let keywords = extractor.extract(text);

        assert!(keywords.is_empty());
    }

    #[test]
    fn test_semantic_consolidator_empty() {
        let consolidator = SemanticConsolidator::new();
        let result = consolidator.consolidate(&[]);

        assert_eq!(result.memories_processed, 0);
        assert_eq!(result.facts_extracted, 0);
    }

    #[test]
    fn test_semantic_consolidator_with_thresholds() {
        let consolidator = SemanticConsolidator::with_thresholds(2, 7);
        assert_eq!(consolidator.min_support, 2);
        assert_eq!(consolidator.min_age_days, 7);
    }

    #[test]
    fn test_fact_type_classification() {
        let consolidator = SemanticConsolidator::new();

        assert_eq!(
            consolidator.classify_fact("preference: concise code"),
            FactType::Preference
        );
        assert_eq!(
            consolidator.classify_fact("system can handle 10k requests"),
            FactType::Capability
        );
        assert_eq!(
            consolidator.classify_fact("auth relates to jwt"),
            FactType::Relationship
        );
        assert_eq!(
            consolidator.classify_fact("to deploy, run cargo build"),
            FactType::Procedure
        );
        assert_eq!(
            consolidator.classify_fact("MemoryId is a UUID wrapper"),
            FactType::Definition
        );
    }

    #[test]
    fn test_reinforce_fact() {
        let consolidator = SemanticConsolidator::new();
        let mut fact = SemanticFact {
            id: "test-fact".to_string(),
            fact: "test fact content".to_string(),
            confidence: 0.5,
            support_count: 1,
            source_memories: vec![],
            related_entities: vec![],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now() - chrono::Duration::days(10),
            fact_type: FactType::Pattern,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        };
        let memory = create_test_memory("reinforcing memory", 0.7);

        let old_confidence = fact.confidence;
        consolidator.reinforce_fact(&mut fact, &memory);

        assert!(fact.confidence > old_confidence);
        assert_eq!(fact.support_count, 2);
        assert!(fact.source_memories.contains(&memory.id));
    }

    #[test]
    fn test_fact_decay_threshold() {
        let consolidator = SemanticConsolidator::new();

        let recent_fact = SemanticFact {
            id: "recent".to_string(),
            fact: "recent fact".to_string(),
            confidence: 0.8,
            support_count: 5,
            source_memories: vec![],
            related_entities: vec![],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        };
        assert!(!consolidator.should_decay_fact(&recent_fact));

        let old_fact = SemanticFact {
            id: "old".to_string(),
            fact: "old fact".to_string(),
            confidence: 0.1,
            support_count: 1,
            source_memories: vec![],
            related_entities: vec![],
            created_at: chrono::Utc::now() - chrono::Duration::days(365),
            last_reinforced: chrono::Utc::now() - chrono::Duration::days(100),
            fact_type: FactType::Pattern,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        };
        assert!(consolidator.should_decay_fact(&old_fact));
    }

    #[test]
    fn test_compression_stats_default() {
        let stats = CompressionStats::default();

        assert_eq!(stats.total_compressed, 0);
        assert_eq!(stats.average_compression_ratio, 1.0);
        assert!(stats.strategies_used.is_empty());
    }

    #[test]
    fn test_create_summary() {
        let pipeline = CompressionPipeline::new();
        let content = "This is a long piece of content that should be summarized into fewer words";
        let summary = pipeline.create_summary(content, 5);

        assert!(summary.ends_with("..."));
        assert!(summary.len() < content.len());
    }

    #[test]
    fn test_consolidation_result_default() {
        let result = ConsolidationResult::default();

        assert_eq!(result.memories_processed, 0);
        assert_eq!(result.facts_extracted, 0);
        assert!(result.new_facts.is_empty());
    }

    #[test]
    fn test_fact_type_default() {
        let fact_type = FactType::default();
        assert_eq!(fact_type, FactType::Pattern);
    }

    fn create_typed_memory(
        content: &str,
        importance: f32,
        exp_type: ExperienceType,
        entities: Vec<String>,
    ) -> Memory {
        let experience = Experience {
            content: content.to_string(),
            experience_type: exp_type,
            entities,
            ..Default::default()
        };

        let created_at = Some(chrono::Utc::now() - chrono::Duration::days(60));

        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            importance,
            None,
            None,
            None,
            created_at,
        )
    }

    #[test]
    fn test_jaccard_similarity_helper() {
        let a: HashSet<String> = ["rust", "memory", "safety"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let b: HashSet<String> = ["rust", "memory", "performance"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let c: HashSet<String> = ["python", "web", "django"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let ab = SemanticConsolidator::jaccard_similarity(&a, &b);
        assert!(ab > 0.4, "rust+memory overlap should give ~0.5, got {ab}");
        assert!(ab < 0.6);

        let ac = SemanticConsolidator::jaccard_similarity(&a, &c);
        assert!(ac < 0.01, "disjoint sets should give 0.0, got {ac}");

        let aa = SemanticConsolidator::jaccard_similarity(&a, &a);
        assert!((aa - 1.0).abs() < 0.001, "identical sets should give 1.0");

        let empty: HashSet<String> = HashSet::new();
        assert_eq!(
            SemanticConsolidator::jaccard_similarity(&empty, &empty),
            0.0
        );
    }

    #[test]
    fn test_stemmed_tokens_removes_stop_words() {
        let consolidator = SemanticConsolidator::new();
        let tokens = consolidator.stemmed_tokens("The Rust programming language is very fast");

        assert!(!tokens.is_empty());
        // "the", "is", "very" should be filtered as stop words
        assert!(!tokens.contains("the"));
        assert!(!tokens.contains("is"));
        assert!(!tokens.contains("very"));
        // "rust", "programming", "language", "fast" should survive (stemmed)
        assert!(tokens.contains("rust"));
        assert!(tokens.contains("fast"));
    }

    #[test]
    fn test_similarity_grouping_clusters_similar_patterns() {
        let consolidator = SemanticConsolidator::with_thresholds(2, 0);

        let m1 = create_typed_memory(
            "Rust is a systems programming language with memory safety and performance guarantees",
            0.8,
            ExperienceType::Learning,
            vec!["Rust".to_string(), "memory safety".to_string()],
        );
        let m2 = create_typed_memory(
            "Rust is a compiled language providing memory safety with great performance",
            0.7,
            ExperienceType::Learning,
            vec!["Rust".to_string(), "memory safety".to_string()],
        );
        let m3 = create_typed_memory(
            "Python is a dynamic language great for data science and machine learning workflows",
            0.6,
            ExperienceType::Learning,
            vec!["Python".to_string(), "data science".to_string()],
        );

        let result = consolidator.consolidate(&[m1, m2, m3]);

        assert!(
            result.facts_extracted >= 1,
            "Similar memories about Rust should cluster into at least 1 fact, got {}",
            result.facts_extracted
        );
        let has_rust_fact = result
            .new_facts
            .iter()
            .any(|f| f.fact.to_lowercase().contains("rust"));
        assert!(has_rust_fact, "Should have a fact about Rust");
        assert!(
            result.new_facts.iter().all(|f| f.support_count == 1),
            "a new fact has been confirmed by one consolidation cycle"
        );
    }

    #[test]
    fn test_multi_extractor_produces_multiple_candidates() {
        let consolidator = SemanticConsolidator::new();

        // Learning runs the definition, pattern and procedure extractors.
        let memory = create_typed_memory(
            "RocksDB is a high-performance embedded key-value database engine. If you see a corruption error, run the repair tool.",
            0.8,
            ExperienceType::Learning,
            vec!["RocksDB".to_string()],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);

        assert!(
            candidates.len() >= 2,
            "Multi-extractor should produce >=2 candidates, got {}",
            candidates.len()
        );
    }

    #[test]
    fn operational_types_produce_no_candidates() {
        let consolidator = SemanticConsolidator::new();
        for exp_type in [
            ExperienceType::Context,
            ExperienceType::Command,
            ExperienceType::CodeEdit,
            ExperienceType::FileAccess,
            ExperienceType::Search,
        ] {
            let memory = create_typed_memory(
                "RocksDB is a high-performance embedded key-value database engine used for storage.",
                0.9,
                exp_type.clone(),
                vec!["RocksDB".to_string()],
            );
            assert!(
                consolidator.extract_fact_candidates(&memory).is_empty(),
                "{exp_type:?} is an execution trace"
            );
        }
    }

    #[test]
    fn fact_shape_gate_rejects_questions_fragments_code_and_logs() {
        for text in [
            "Should the scheduler drain one worker at a time?",
            "The migration finished but the index was still rebuilding...",
            "continues the earlier thought about the retry policy",
            "let x = foo(bar[0]); if (x > 1) { return y; }",
            "[ERROR] connection refused while opening the replica stream",
            "src/a/b.rs src/c/d.rs src/e.rs",
        ] {
            assert!(!SemanticConsolidator::is_fact_shaped(text), "{text}");
        }
        assert!(SemanticConsolidator::is_fact_shaped(
            "The scheduler drains one worker at a time during rolling restarts."
        ));
        assert!(SemanticConsolidator::is_fact_shaped(
            "the scheduler drains one worker at a time during rolling restarts."
        ));
    }

    #[test]
    fn definition_extraction_slices_on_char_boundaries() {
        let consolidator = SemanticConsolidator::new();
        let dash_before_subject =
            "Background \u{2014}Vamana is a graph index for nearest neighbours";
        assert_eq!(
            consolidator
                .extract_definition(dash_before_subject)
                .as_deref(),
            Some("Vamana is a graph index for nearest neighbours")
        );
        let long_definition = format!("Vamana is {}\u{2014}{}", "x".repeat(98), "y".repeat(20));
        let extracted = consolidator.extract_definition(&long_definition).unwrap();
        assert!(extracted.starts_with("Vamana is xxx"));
        assert!(!extracted.contains('\u{2014}'));
    }

    #[test]
    fn extractors_reject_pronoun_subjects_similes_and_mid_sentence_actions() {
        let consolidator = SemanticConsolidator::new();
        assert_eq!(
            consolidator.extract_definition("It is a long-running background process."),
            None
        );
        assert!(consolidator
            .extract_definition("Vamana is a graph index for approximate nearest neighbours.")
            .is_some());
        assert_eq!(
            consolidator.extract_preference("The dashboard looks like a spreadsheet from 1998."),
            None
        );
        assert!(consolidator
            .extract_preference("We prefer explicit error types over anyhow in library code.")
            .is_some());
        assert_eq!(
            consolidator.extract_procedure("In this project we use Rust for the storage engine."),
            None
        );
        assert!(consolidator
            .extract_procedure("Step 1: run cargo fmt before every commit to the repository.")
            .is_some());
    }

    #[test]
    fn clusters_compare_to_their_first_member_and_stop_at_the_cap() {
        let consolidator = SemanticConsolidator::new();
        let id = || MemoryId(Uuid::new_v4());
        // The third candidate clears the threshold against the union of the first two
        // (3 of 6 stems) but not against the first alone (2 of 6).
        let candidates = vec![
            ("alpha beta gamma delta".to_string(), id(), 0.5),
            ("alpha beta gamma epsilon".to_string(), id(), 0.5),
            ("beta gamma epsilon zeta".to_string(), id(), 0.5),
        ];
        let clusters = consolidator.group_candidates_by_similarity(&candidates, 0.45);
        assert_eq!(
            clusters.len(),
            2,
            "the third candidate starts its own cluster"
        );

        let same: Vec<(String, MemoryId, f32)> = (0..CONSOLIDATION_CLUSTER_SIZE_CAP + 5)
            .map(|_| ("alpha beta gamma delta".to_string(), id(), 0.5))
            .collect();
        let clusters = consolidator.group_candidates_by_similarity(&same, 0.45);
        assert_eq!(clusters[0].members.len(), CONSOLIDATION_CLUSTER_SIZE_CAP);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_generic_fallback_produces_real_sentence() {
        let consolidator = SemanticConsolidator::new();

        let memory = create_typed_memory(
            "The deployment pipeline uses Docker containers for isolation. Each service runs independently.",
            0.7,
            ExperienceType::Observation,
            vec!["Docker".to_string()],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);

        // Should NOT produce "involves: ..." synthetic patterns
        let has_involves = candidates
            .iter()
            .any(|(text, _)| text.starts_with("involves:"));
        assert!(
            !has_involves,
            "Should not produce synthetic 'involves:' patterns"
        );

        // Should have at least one real sentence
        assert!(
            !candidates.is_empty(),
            "Should extract at least one candidate"
        );
    }

    const BRIDGE_REPORT: &str =
        "Initial reports said four crew members were injured in the bridge collapse.";

    #[test]
    fn declarative_sentence_yields_a_candidate_with_no_ner_entities() {
        let consolidator = SemanticConsolidator::new();
        let memory = create_typed_memory(BRIDGE_REPORT, 0.3, ExperienceType::Observation, vec![]);

        let candidates = consolidator.extract_fact_candidates(&memory);
        assert!(
            candidates
                .iter()
                .any(|(t, _)| t.contains("four crew members were injured")),
            "a plain declarative sentence must mint a candidate without NER entities, got \
             {candidates:?}"
        );
    }

    #[test]
    fn declarative_extractor_runs_on_every_non_operational_type() {
        let consolidator = SemanticConsolidator::new();
        for exp_type in [
            ExperienceType::Observation,
            ExperienceType::Conversation,
            ExperienceType::Task,
            ExperienceType::Intention,
            ExperienceType::Decision,
            ExperienceType::Learning,
            ExperienceType::Discovery,
            ExperienceType::Pattern,
            ExperienceType::Error,
        ] {
            let memory = create_typed_memory(BRIDGE_REPORT, 0.3, exp_type.clone(), vec![]);
            assert!(
                !consolidator.extract_fact_candidates(&memory).is_empty(),
                "{exp_type:?} must produce a declarative candidate"
            );
        }
    }

    #[test]
    fn importance_weights_a_declarative_candidate_instead_of_gating_it() {
        let consolidator = SemanticConsolidator::new();
        let low = create_typed_memory(BRIDGE_REPORT, 0.23, ExperienceType::Observation, vec![]);
        let low = consolidator.extract_fact_candidates(&low);
        assert!(
            !low.is_empty(),
            "importance below 0.5 must not erase the candidate"
        );

        let high = create_typed_memory(BRIDGE_REPORT, 0.9, ExperienceType::Observation, vec![]);
        let high = consolidator.extract_fact_candidates(&high);
        assert!(
            high[0].1 > low[0].1,
            "importance must still rank candidates"
        );
    }

    #[test]
    fn entity_mentions_rank_sentences_rather_than_gating_them() {
        let consolidator = SemanticConsolidator::new();
        // Identical tails: the second sentence has exactly one more content word, so density
        // alone picks it, and one entity hit (+2.0) flips the choice to the first.
        let content = "The Dali blocked the shipping channel for weeks. \
                       The salvage barge blocked the shipping channel for weeks.";

        let with_entity = create_typed_memory(
            content,
            0.6,
            ExperienceType::Observation,
            vec!["Dali".to_string()],
        );
        let picked = consolidator.extract_fact_candidates(&with_entity);
        assert!(
            picked.iter().any(|(t, _)| t.contains("Dali")),
            "an entity mention must outrank raw density, got {picked:?}"
        );

        let without_entity = create_typed_memory(content, 0.6, ExperienceType::Observation, vec![]);
        let fallback = consolidator.extract_fact_candidates(&without_entity);
        assert!(
            fallback.iter().any(|(t, _)| t.contains("salvage barge")),
            "without an entity signal, density decides; got {fallback:?}"
        );
    }

    #[test]
    fn content_word_floor_separates_a_caption_from_a_clause() {
        assert_eq!(CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS, 4);
        let consolidator = SemanticConsolidator::new();
        for w in ["the", "and", "in"] {
            assert!(
                consolidator.keyword_extractor.is_stop_word(w),
                "fixture assumes '{w}' is a stop word"
            );
        }
        for w in ["barge", "tugboat", "harbor", "sank"] {
            assert!(
                !consolidator.keyword_extractor.is_stop_word(w),
                "fixture assumes '{w}' is a content word"
            );
        }

        let caption = "The barge and the tugboat in the harbor";
        assert!(consolidator
            .extract_salient_statement(caption, &[])
            .is_none());
        let clause = "The barge and the tugboat in the harbor sank";
        assert_eq!(
            consolidator.extract_salient_statement(clause, &[]),
            Some(clause.to_string())
        );
    }

    #[test]
    fn repeated_declarative_mentions_reach_the_support_threshold() {
        let consolidator = SemanticConsolidator::with_thresholds(2, 0);
        let m1 = create_typed_memory(BRIDGE_REPORT, 0.3, ExperienceType::Observation, vec![]);
        let m2 = create_typed_memory(
            "Early reports said four crew members were injured in the bridge collapse.",
            0.3,
            ExperienceType::Observation,
            vec![],
        );

        let single = consolidator.consolidate(std::slice::from_ref(&m1));
        assert_eq!(single.facts_extracted, 0, "one mention is an anecdote");

        let paired = consolidator.consolidate(&[m1, m2]);
        assert!(
            paired
                .new_facts
                .iter()
                .any(|f| f.fact.contains("crew members were injured")),
            "two corroborating memories must mint the sentence, got {:?}",
            paired.new_facts
        );
    }

    fn observation_aged(content: &str, days: i64) -> Memory {
        Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience {
                content: content.to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            0.8,
            None,
            None,
            None,
            Some(chrono::Utc::now() - chrono::Duration::days(days)),
        )
    }

    const CLAIM: &str =
        "Initial reports said four crew members were injured in the bridge collapse.";
    const CLAIM_ECHO: &str =
        "Early reports said four crew members were injured in the bridge collapse.";
    const CORRECTION: &str =
        "Corrected reports confirm no crew members were injured in the bridge collapse.";
    const CORRECTION_ECHO: &str =
        "Later reports confirm no crew members were injured in the bridge collapse.";

    #[test]
    fn contradicting_statements_never_share_a_cluster() {
        let consolidator = SemanticConsolidator::new();
        let memories: Vec<Memory> = [CLAIM, CLAIM_ECHO, CORRECTION, CORRECTION_ECHO]
            .iter()
            .map(|content| observation_aged(content, 8))
            .collect();

        let result = consolidator.consolidate(&memories);
        assert_eq!(
            result.facts_extracted, 2,
            "claim and correction mint as separate facts: {:?}",
            result.new_facts
        );
        assert!(result
            .new_facts
            .iter()
            .any(|f| f.fact.contains("four crew members were injured")));
        assert!(result
            .new_facts
            .iter()
            .any(|f| f.fact.contains("no crew members were injured")));
    }

    #[test]
    fn minted_facts_are_ordered_by_evidence_recency() {
        let consolidator = SemanticConsolidator::new();
        let memories = vec![
            observation_aged(CORRECTION, 20),
            observation_aged(CLAIM, 40),
            observation_aged(CORRECTION_ECHO, 20),
            observation_aged(CLAIM_ECHO, 40),
        ];

        let result = consolidator.consolidate(&memories);
        assert_eq!(result.facts_extracted, 2);
        assert!(
            result.new_facts[0]
                .fact
                .contains("four crew members were injured"),
            "the older claim is ingested first: {:?}",
            result.new_facts
        );
        assert!(result.new_facts[1]
            .fact
            .contains("no crew members were injured"));
        assert_eq!(
            result.new_fact_ids,
            result
                .new_facts
                .iter()
                .map(|f| f.id.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn consolidation_does_not_depend_on_input_order() {
        let consolidator = SemanticConsolidator::new();
        let memories: Vec<Memory> = [
            CLAIM,
            CORRECTION,
            CLAIM_ECHO,
            CORRECTION_ECHO,
            "The salvage barge cleared the shipping channel after six weeks of work.",
            "After six weeks of work the salvage barge cleared the shipping channel.",
        ]
        .iter()
        .enumerate()
        .map(|(i, content)| observation_aged(content, 10 + i as i64))
        .collect();
        let forward: Vec<String> = consolidator
            .consolidate(&memories)
            .new_facts
            .into_iter()
            .map(|f| f.fact)
            .collect();
        let mut reversed = memories.clone();
        reversed.reverse();
        let backward: Vec<String> = consolidator
            .consolidate(&reversed)
            .new_facts
            .into_iter()
            .map(|f| f.fact)
            .collect();
        assert!(!forward.is_empty());
        assert_eq!(forward, backward);
    }

    /// Two different zeros: a fresh corpus mints nothing because the age gate looked at
    /// nothing, while an aged corpus of unique statements mints nothing because none is
    /// corroborated. `memories_eligible` tells them apart.
    #[test]
    fn an_age_gated_zero_is_distinguishable_from_a_corroboration_zero() {
        let unique = [
            "MSC Brianna completed cargo operations at Seagirt berth 3 and departed on the evening tide.",
            "Crane STS-07 at Seagirt completed its 4,000-hour preventive maintenance.",
            "Fog delayed pilot boardings in the Craighill Channel for two hours.",
        ];

        let fresh: Vec<Memory> = unique.iter().map(|c| observation_aged(c, 0)).collect();
        let gated = SemanticConsolidator::new().consolidate(&fresh);
        assert_eq!(gated.memories_eligible, 0);
        assert_eq!(gated.facts_extracted, 0);

        // Past the consolidator's three-day floor, with the age gate lowered to zero.
        let aged: Vec<Memory> = unique.iter().map(|c| observation_aged(c, 4)).collect();
        let eager = SemanticConsolidator::with_thresholds(CONSOLIDATION_MIN_SUPPORT, 0);
        let result = eager.consolidate(&aged);
        assert_eq!(result.memories_eligible, aged.len());
        assert_eq!(
            result.facts_extracted, 0,
            "unique statements are not corroborated"
        );
    }

    #[test]
    fn test_no_relates_to_patterns() {
        let consolidator = SemanticConsolidator::new();
        let memory = create_typed_memory(
            "Testing entity extraction without template patterns",
            0.7,
            ExperienceType::Observation,
            vec!["JWT".to_string(), "Auth".to_string(), "Token".to_string()],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);
        assert!(
            !candidates.iter().any(|(t, _)| t.contains("relates to")),
            "entity co-occurrence belongs to the graph, not to 'X relates to Y' facts"
        );
    }

    #[test]
    fn test_quality_gate_rejects_session_noise() {
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Session started at 2:15 PM with token budget of 200k"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Context compressed after reaching 80% of token budget"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "3 memories surfaced via proactive_context for the current query"
        ));
    }

    #[test]
    fn test_quality_gate_rejects_todo_noise() {
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Todo created for implementing the new authentication module"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Task completed after fixing the database connection pooling issue"
        ));
    }

    #[test]
    fn test_quality_gate_rejects_system_output() {
        for text in [
            "<task-notification> background build finished with status ok",
            "[Assistant: CodeEdit] rewrote the retry loop in the uploader module",
            "Reconnected after the server restarted at midnight on the build host",
            "Agent \"reviewer\" completed its pass over the storage module changes",
        ] {
            assert!(!SemanticConsolidator::is_knowledge_worthy(text), "{text}");
        }
    }

    #[test]
    fn test_quality_gate_rejects_short_text() {
        assert!(!SemanticConsolidator::is_knowledge_worthy("short"));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "too short to be fact"
        ));
    }

    #[test]
    fn test_quality_gate_rejects_bare_file_paths() {
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "src/memory/compression.rs:919"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "hooks/memory-hook.ts file"
        ));
    }

    #[test]
    fn test_quality_gate_accepts_genuine_knowledge() {
        for text in [
            "Rust provides memory safety without garbage collection through its ownership system",
            "The Vamana index automatically switches to SPANN when the dataset exceeds 100k vectors",
            "Hebbian learning uses additive boost and multiplicative decay for asymmetric strengthening",
            "RocksDB column families reduce file descriptor usage by consolidating multiple stores",
        ] {
            assert!(SemanticConsolidator::is_knowledge_worthy(text), "{text}");
        }
    }

    #[test]
    fn test_quality_gate_accepts_file_paths_with_prose() {
        assert!(SemanticConsolidator::is_knowledge_worthy(
            "The router configuration in src/handlers/router.rs defines all API endpoint routes for the server"
        ));
    }
}
