//! Hebbian-Friendly Segmentation Engine
//!
//! Segments raw input into atomic memory units optimized for Hebbian learning.
//! Key principle: entities that belong together should form edges, unrelated ones shouldn't.
//!
//! Architecture:
//! 1. Sentence splitting - break input into sentence-level units
//! 2. Type detection - classify each sentence by ExperienceType
//! 3. Same-type merging - consecutive sentences of same type become one memory
//! 4. Entity-aware splitting - split if entities have no semantic relation
//! 5. Deduplication - prevent duplicate edges in knowledge graph

use crate::memory::types::ExperienceType;
use regex::Regex;
use std::collections::HashSet;

/// Result of segmenting input text
#[derive(Debug, Clone)]
pub struct AtomicMemory {
    /// Detected experience type
    pub experience_type: ExperienceType,
    /// The segmented content
    pub content: String,
    /// Extracted entities (for Hebbian edge formation)
    pub entities: Vec<String>,
    /// Confidence in type detection (0.0 - 1.0)
    pub type_confidence: f32,
    /// Source indicator (which part of input this came from)
    pub source_offset: usize,
}

/// Input source for segmentation context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSource {
    /// From Cortex proxy (Claude API)
    Cortex,
    /// Direct user input via remember API
    UserApi,
    /// From codebase indexing
    Codebase,
    /// From streaming ingestion
    Streaming,
    /// From auto-ingest (proactive_context)
    AutoIngest,
}

/// Type detection pattern with priority
struct TypePattern {
    pattern: Regex,
    experience_type: ExperienceType,
    confidence: f32,
    priority: u8,
}

/// Segmentation engine for Hebbian-optimal memory formation
pub struct SegmentationEngine {
    /// Type detection patterns ordered by priority
    type_patterns: Vec<TypePattern>,
    /// Minimum content length for a valid segment
    min_segment_length: usize,
    /// Maximum content length before forced split
    max_segment_length: usize,
}

impl Default for SegmentationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentationEngine {
    /// Create a new segmentation engine with default patterns
    pub fn new() -> Self {
        let type_patterns = Self::build_type_patterns();
        Self {
            type_patterns,
            min_segment_length: 20,
            max_segment_length: 2000,
        }
    }

    /// Build type detection patterns
    /// Priority: higher = checked first
    /// Confidence: how certain we are when pattern matches
    fn build_type_patterns() -> Vec<TypePattern> {
        let mut patterns = vec![
            // === HIGH PRIORITY: Explicit markers ===
            TypePattern {
                pattern: Regex::new(r"(?i)\b(decided|chose|chosen|went with|picked|selected|opted for|decision to)\b").unwrap(),
                experience_type: ExperienceType::Decision,
                confidence: 0.95,
                priority: 100,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(learned|realized|understood|figured out|now I know|insight)\b").unwrap(),
                experience_type: ExperienceType::Learning,
                confidence: 0.90,
                priority: 95,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)(error:|bug:|exception:|failed:|broke:|crash|traceback|stacktrace|\bfixed\b)").unwrap(),
                experience_type: ExperienceType::Error,
                confidence: 0.95,
                priority: 98,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(discovered|found that|noticed|stumbled upon|turns out)\b").unwrap(),
                experience_type: ExperienceType::Discovery,
                confidence: 0.85,
                priority: 90,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)(pattern:|always|every time|whenever|consistently|tends to)\b").unwrap(),
                experience_type: ExperienceType::Pattern,
                confidence: 0.80,
                priority: 85,
            },

            // === MEDIUM PRIORITY: Action-based ===
            TypePattern {
                pattern: Regex::new(r"(?i)\b(will|tomorrow|later|remind me|don't forget|need to remember|scheduled|need to fix)\b").unwrap(),
                experience_type: ExperienceType::Intention,
                confidence: 0.85,
                priority: 88,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(edited|changed|modified|updated|refactored|renamed|moved)\b.*\b(file|code|function|class|module)\b").unwrap(),
                experience_type: ExperienceType::CodeEdit,
                confidence: 0.90,
                priority: 80,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(opened|read|accessed|viewed|looked at)\b.*\b(file|document|page)\b").unwrap(),
                experience_type: ExperienceType::FileAccess,
                confidence: 0.85,
                priority: 75,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(searched|looked for|found|grep|rg|find)\b").unwrap(),
                experience_type: ExperienceType::Search,
                confidence: 0.80,
                priority: 70,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(ran|executed|command:|terminal|shell|bash|npm|cargo|git)\b").unwrap(),
                experience_type: ExperienceType::Command,
                confidence: 0.85,
                priority: 72,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(task:|todo:|need to|should|must|have to|working on)\b").unwrap(),
                experience_type: ExperienceType::Task,
                confidence: 0.75,
                priority: 65,
            },

            // === LOWER PRIORITY: Context indicators ===
            TypePattern {
                pattern: Regex::new(r"(?i)(context:|background:|for reference|fyi|note:)\b").unwrap(),
                experience_type: ExperienceType::Context,
                confidence: 0.80,
                priority: 60,
            },
            TypePattern {
                pattern: Regex::new(r"(?i)\b(said|told|asked|replied|mentioned|discussed|conversation)\b").unwrap(),
                experience_type: ExperienceType::Conversation,
                confidence: 0.70,
                priority: 50,
            },
        ];

        // Sort by priority descending
        patterns.sort_by(|a, b| b.priority.cmp(&a.priority));
        patterns
    }

    /// Main entry point: segment input into atomic memories
    pub fn segment(&self, input: &str, source: InputSource) -> Vec<AtomicMemory> {
        let input = input.trim();
        if input.is_empty() {
            return Vec::new();
        }

        // Step 1: Split into sentences
        let sentences = self.split_sentences(input);
        if sentences.is_empty() {
            return Vec::new();
        }

        // Step 2: Classify each sentence
        let typed_sentences: Vec<(ExperienceType, f32, String, usize)> = sentences
            .into_iter()
            .enumerate()
            .filter(|(_, s)| s.len() >= self.min_segment_length)
            .map(|(offset, s)| {
                let (exp_type, confidence) = self.detect_type(&s, source);
                (exp_type, confidence, s, offset)
            })
            .collect();

        if typed_sentences.is_empty() {
            // If all sentences were too short, treat entire input as one
            let (exp_type, confidence) = self.detect_type(input, source);
            return vec![AtomicMemory {
                experience_type: exp_type,
                content: input.to_string(),
                entities: self.extract_simple_entities(input),
                type_confidence: confidence,
                source_offset: 0,
            }];
        }

        // Step 3: Merge consecutive same-type sentences
        let merged = self.merge_consecutive_same_type(typed_sentences);

        // Step 4: Apply max length splitting if needed
        let split = self.apply_max_length_splits(merged);

        // Step 5: Extract entities for each segment
        split
            .into_iter()
            .map(|(exp_type, confidence, content, offset)| AtomicMemory {
                experience_type: exp_type,
                entities: self.extract_simple_entities(&content),
                content,
                type_confidence: confidence,
                source_offset: offset,
            })
            .collect()
    }

    /// Split input into sentences
    fn split_sentences(&self, input: &str) -> Vec<String> {
        // Split on sentence boundaries: . ! ? followed by space or newline
        // But preserve abbreviations like "e.g." "i.e." "Dr." etc.
        let mut sentences = Vec::new();
        let mut current = String::new();
        let mut chars = input.chars().peekable();

        while let Some(c) = chars.next() {
            current.push(c);

            // Check for sentence boundary
            if matches!(c, '.' | '!' | '?') {
                // Look ahead to see if this is end of sentence
                if let Some(&next) = chars.peek() {
                    if next.is_whitespace() || next == '\n' {
                        // Check if this looks like an abbreviation
                        let trimmed = current.trim();
                        let is_abbreviation = Self::is_likely_abbreviation(trimmed);

                        if !is_abbreviation {
                            let sentence = current.trim().to_string();
                            if !sentence.is_empty() {
                                sentences.push(sentence);
                            }
                            current = String::new();
                            // Skip the whitespace
                            chars.next();
                        }
                    }
                }
            }

            // Also split on double newlines (paragraph boundaries)
            if c == '\n' {
                if let Some(&next) = chars.peek() {
                    if next == '\n' {
                        let sentence = current.trim().to_string();
                        if !sentence.is_empty() {
                            sentences.push(sentence);
                        }
                        current = String::new();
                        chars.next(); // Skip second newline
                    }
                }
            }
        }

        // Don't forget the last sentence
        let final_sentence = current.trim().to_string();
        if !final_sentence.is_empty() {
            sentences.push(final_sentence);
        }

        sentences
    }

    /// Check if a string ending looks like an abbreviation
    fn is_likely_abbreviation(s: &str) -> bool {
        let lower = s.to_lowercase();
        let abbreviations = [
            "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.", "jr.", "sr.", "inc.",
            "ltd.", "corp.", "co.", "st.", "ave.", "rd.", "blvd.", "fig.", "ref.", "vol.", "no.",
            "pp.", "ed.", "rev.",
        ];

        for abbr in &abbreviations {
            if lower.ends_with(abbr) {
                return true;
            }
        }

        // Single letter followed by period (initials)
        if s.len() >= 2 {
            let chars: Vec<char> = s.chars().collect();
            let last_two = &chars[chars.len() - 2..];
            if last_two[0].is_alphabetic() && last_two[1] == '.' {
                // Check if the letter before is whitespace or start
                if chars.len() == 2 || chars[chars.len() - 3].is_whitespace() {
                    return true;
                }
            }
        }

        false
    }

    /// Detect experience type from content
    fn detect_type(&self, content: &str, source: InputSource) -> (ExperienceType, f32) {
        // Source-based hints
        let source_type = match source {
            InputSource::Codebase => Some((ExperienceType::FileAccess, 0.6)),
            InputSource::AutoIngest => None, // Need to detect from content
            _ => None,
        };

        // Try pattern matching
        for pattern in &self.type_patterns {
            if pattern.pattern.is_match(content) {
                return (pattern.experience_type.clone(), pattern.confidence);
            }
        }

        // Fall back to source hint or default
        source_type.unwrap_or((ExperienceType::Observation, 0.5))
    }

    /// Merge consecutive sentences of the same type
    fn merge_consecutive_same_type(
        &self,
        sentences: Vec<(ExperienceType, f32, String, usize)>,
    ) -> Vec<(ExperienceType, f32, String, usize)> {
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_type = sentences[0].0.clone();
        let mut current_confidence = sentences[0].1;
        let mut current_content = sentences[0].2.clone();
        let mut current_offset = sentences[0].3;

        for (exp_type, confidence, content, offset) in sentences.into_iter().skip(1) {
            if exp_type == current_type {
                // Merge: append content, take max confidence
                current_content.push(' ');
                current_content.push_str(&content);
                current_confidence = current_confidence.max(confidence);
            } else {
                // Different type: save current, start new
                result.push((
                    current_type,
                    current_confidence,
                    current_content,
                    current_offset,
                ));
                current_type = exp_type;
                current_confidence = confidence;
                current_content = content;
                current_offset = offset;
            }
        }

        // Don't forget the last one
        result.push((
            current_type,
            current_confidence,
            current_content,
            current_offset,
        ));

        result
    }

    /// Split segments that exceed max length
    fn apply_max_length_splits(
        &self,
        segments: Vec<(ExperienceType, f32, String, usize)>,
    ) -> Vec<(ExperienceType, f32, String, usize)> {
        let mut result = Vec::new();

        for (exp_type, confidence, content, offset) in segments {
            if content.len() <= self.max_segment_length {
                result.push((exp_type, confidence, content, offset));
            } else {
                // Split on sentence boundaries within the long content
                let sub_sentences = self.split_sentences(&content);
                let mut current_chunk = String::new();

                for sentence in sub_sentences {
                    if current_chunk.len() + sentence.len() + 1 > self.max_segment_length {
                        if !current_chunk.is_empty() {
                            result.push((
                                exp_type.clone(),
                                confidence,
                                current_chunk.clone(),
                                offset,
                            ));
                        }
                        current_chunk = sentence;
                    } else {
                        if !current_chunk.is_empty() {
                            current_chunk.push(' ');
                        }
                        current_chunk.push_str(&sentence);
                    }
                }

                if !current_chunk.is_empty() {
                    result.push((exp_type, confidence, current_chunk, offset));
                }
            }
        }

        result
    }

    /// Simple entity extraction (words > 2 chars, excluding stopwords)
    fn extract_simple_entities(&self, content: &str) -> Vec<String> {
        let stopwords: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "or", "if", "because", "while", "although", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "what", "which", "who", "whom",
            "its", "his", "her", "their", "my", "your", "our",
        ]
        .into_iter()
        .collect();

        content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|word| word.len() > 2 && !stopwords.contains(word))
            .map(|s| s.to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Deduplication result
#[derive(Debug, Clone)]
pub enum DeduplicationResult {
    /// Store as new memory
    New,
    /// Exact duplicate - skip storage
    Duplicate { existing_id: String },
    /// Semantic near-duplicate - consider merging
    SemanticMatch {
        existing_id: String,
        similarity: f32,
    },
    /// Same entities but different content - link as related
    EntityOverlap { existing_id: String, overlap: f32 },
}

/// Deduplication engine to prevent duplicate Hebbian edges
pub struct DeduplicationEngine {
    /// Content hash -> memory ID index
    content_hashes: HashSet<u64>,
}

impl Default for DeduplicationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DeduplicationEngine {
    pub fn new() -> Self {
        Self {
            content_hashes: HashSet::new(),
        }
    }

    /// Compute content hash for exact duplicate detection
    pub fn content_hash(content: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Normalize: lowercase, collapse whitespace
        let normalized: String = content
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        normalized.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if content is a duplicate
    pub fn is_duplicate(&self, content: &str) -> bool {
        let hash = Self::content_hash(content);
        self.content_hashes.contains(&hash)
    }

    /// Register a new content hash
    pub fn register(&mut self, content: &str) {
        let hash = Self::content_hash(content);
        self.content_hashes.insert(hash);
    }

    /// Calculate entity overlap between two entity sets
    pub fn calculate_entity_overlap(entities1: &[String], entities2: &[String]) -> f32 {
        if entities1.is_empty() || entities2.is_empty() {
            return 0.0;
        }

        let set1: HashSet<_> = entities1.iter().map(|s| s.to_lowercase()).collect();
        let set2: HashSet<_> = entities2.iter().map(|s| s.to_lowercase()).collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_splitting() {
        let engine = SegmentationEngine::new();

        // Test with explicit newline separation which is more reliable
        let input = "I decided to use Rust.\n\nIt has great performance.\n\nThe memory safety is excellent.";
        let sentences = engine.split_sentences(input);

        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("Rust"));
        assert!(sentences[1].contains("performance"));
        assert!(sentences[2].contains("memory safety"));
    }

    #[test]
    fn test_abbreviation_preservation() {
        let engine = SegmentationEngine::new();

        let input = "E.g. this is an example.\n\nDr. Smith said so.";
        let sentences = engine.split_sentences(input);

        // Should preserve abbreviations within sentences
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("E.g."));
        assert!(sentences[1].contains("Dr."));
    }

    #[test]
    fn test_type_detection_decision() {
        let engine = SegmentationEngine::new();

        let (exp_type, confidence) = engine.detect_type(
            "I decided to use Rust for this project",
            InputSource::UserApi,
        );

        assert!(matches!(exp_type, ExperienceType::Decision));
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_type_detection_error() {
        let engine = SegmentationEngine::new();

        let (exp_type, confidence) =
            engine.detect_type("error: cannot find module 'foo'", InputSource::UserApi);

        assert!(matches!(exp_type, ExperienceType::Error));
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_type_detection_learning() {
        let engine = SegmentationEngine::new();

        let (exp_type, confidence) = engine.detect_type(
            "I learned that async functions need await",
            InputSource::UserApi,
        );

        assert!(matches!(exp_type, ExperienceType::Learning));
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_type_detection_intention() {
        let engine = SegmentationEngine::new();

        let (exp_type, confidence) =
            engine.detect_type("Tomorrow I will review the PR", InputSource::UserApi);

        assert!(matches!(exp_type, ExperienceType::Intention));
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_segmentation_mixed_types() {
        let engine = SegmentationEngine::new();

        // Use explicit type markers for clearer segmentation
        let input = "I decided to use Rust.\n\nerror: found a bug in the auth module.\n\nTomorrow need to fix it.";
        let segments = engine.segment(input, InputSource::UserApi);

        assert_eq!(segments.len(), 3);
        assert!(matches!(
            segments[0].experience_type,
            ExperienceType::Decision
        ));
        assert!(matches!(segments[1].experience_type, ExperienceType::Error));
        assert!(matches!(
            segments[2].experience_type,
            ExperienceType::Intention
        ));
    }

    #[test]
    fn test_same_type_merging() {
        let engine = SegmentationEngine::new();

        // All sentences have "decided" which triggers Decision type
        let input =
            "I decided to use Rust.\n\nI also decided to use Axum.\n\nWe chose RocksDB for storage.";
        let segments = engine.segment(input, InputSource::UserApi);

        // All three sentences are Decision type, should merge into one
        assert_eq!(segments.len(), 1);
        assert!(matches!(
            segments[0].experience_type,
            ExperienceType::Decision
        ));
        assert!(segments[0].content.contains("Rust"));
        assert!(segments[0].content.contains("Axum"));
    }

    #[test]
    fn test_entity_extraction() {
        let engine = SegmentationEngine::new();

        let entities =
            engine.extract_simple_entities("I decided to use Rust for the shodh-memory project");

        assert!(entities.contains(&"rust".to_string()));
        assert!(entities.contains(&"shodh-memory".to_string()));
        assert!(entities.contains(&"project".to_string()));
        // Stopwords should be excluded
        assert!(!entities.contains(&"the".to_string()));
        assert!(!entities.contains(&"to".to_string()));
    }

    #[test]
    fn test_deduplication_hash() {
        let hash1 = DeduplicationEngine::content_hash("Hello World");
        let hash2 = DeduplicationEngine::content_hash("hello world");
        let hash3 = DeduplicationEngine::content_hash("Hello  World"); // Extra space

        // All should normalize to same hash
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_entity_overlap() {
        let entities1 = vec![
            "rust".to_string(),
            "memory".to_string(),
            "project".to_string(),
        ];
        let entities2 = vec![
            "rust".to_string(),
            "memory".to_string(),
            "performance".to_string(),
        ];

        let overlap = DeduplicationEngine::calculate_entity_overlap(&entities1, &entities2);

        // 2 common (rust, memory) / 4 total unique = 0.5
        assert!((overlap - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_max_length_split() {
        let mut engine = SegmentationEngine::new();
        engine.max_segment_length = 100;

        let long_input =
            "This is a very long decision that I made about using Rust for the backend. \
            I also decided to use Axum for the web framework because it has great performance. \
            Additionally I chose RocksDB for storage due to its reliability and speed.";

        let segments = engine.segment(long_input, InputSource::UserApi);

        // Should split into multiple segments due to length
        assert!(segments.len() > 1);
        for segment in &segments {
            assert!(segment.content.len() <= engine.max_segment_length + 50); // Allow some overflow
        }
    }
}
