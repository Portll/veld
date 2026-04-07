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

use crate::memory::types::{ExperienceType, MemoryId};
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use uuid::Uuid;

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

/// A single turn in a conversation (role + content)
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// Speaker role (e.g., "User", "Assistant", "System") or name
    pub role: String,
    /// Content of this turn
    pub content: String,
}

/// Result of conversation-aware segmentation
///
/// Extends AtomicMemory with conversation-level structure:
/// parent_id linking for Q&A pairs, speaker entity refs, and topic tags.
#[derive(Debug, Clone)]
pub struct ConversationMemory {
    /// The underlying atomic memory
    pub memory: AtomicMemory,
    /// Unique ID assigned during segmentation (used for parent_id linking)
    pub id: MemoryId,
    /// Parent memory ID - links an answer to its question
    pub parent_id: Option<MemoryId>,
    /// Speaker identity extracted from the turn role
    pub speaker: String,
    /// Entity reference for the speaker (name + relation)
    pub speaker_entity: SpeakerEntity,
    /// Topic threading tags shared across related consecutive turns
    pub topic_tags: Vec<String>,
}

/// Speaker entity extracted from a conversation turn
#[derive(Debug, Clone)]
pub struct SpeakerEntity {
    /// Normalized speaker name (lowercased, trimmed)
    pub name: String,
    /// Relation type for entity ref ("speaker", "questioner", "responder")
    pub relation: String,
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

    /// Segment a conversation into linked ConversationMemory entries
    ///
    /// This is the conversation-aware counterpart to `segment()`. It takes structured
    /// turns (role + content) and produces memories with:
    /// - Q&A pair linking via parent_id (question -> answer)
    /// - Speaker attribution as entity references
    /// - Topic threading via shared tags on consecutive related turns
    pub fn segment_conversation(
        &self,
        turns: &[ConversationTurn],
        source: InputSource,
    ) -> Vec<ConversationMemory> {
        if turns.is_empty() {
            return Vec::new();
        }

        // Phase 1: Produce one AtomicMemory per turn, stripping role prefixes from content
        let mut turn_memories: Vec<(AtomicMemory, String)> = Vec::with_capacity(turns.len());
        for (i, turn) in turns.iter().enumerate() {
            let cleaned = Self::strip_role_prefix(&turn.content);
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                continue;
            }

            let (exp_type, confidence) = self.detect_type(trimmed, source);
            let entities = self.extract_simple_entities(trimmed);

            let memory = AtomicMemory {
                experience_type: exp_type,
                content: trimmed.to_string(),
                entities,
                type_confidence: confidence,
                source_offset: i,
            };
            turn_memories.push((memory, turn.role.clone()));
        }

        if turn_memories.is_empty() {
            return Vec::new();
        }

        // Phase 2: Assign IDs to every turn memory
        let ids: Vec<MemoryId> = (0..turn_memories.len())
            .map(|_| MemoryId(Uuid::new_v4()))
            .collect();

        // Phase 3: Detect Q&A pairs and assign parent_id links
        let parent_ids = self.detect_qa_pairs(&turn_memories);

        // Phase 4: Compute topic threading tags
        let topic_tags = self.compute_topic_threads(&turn_memories);

        // Phase 5: Assemble ConversationMemory entries
        let mut results = Vec::with_capacity(turn_memories.len());
        for (i, (memory, role)) in turn_memories.into_iter().enumerate() {
            let speaker_name = Self::normalize_speaker(&role);
            let relation = if parent_ids[i].is_some() {
                "responder".to_string()
            } else if self.content_ends_with_question(&memory.content) {
                "questioner".to_string()
            } else {
                "speaker".to_string()
            };

            results.push(ConversationMemory {
                memory,
                id: ids[i].clone(),
                parent_id: parent_ids[i].as_ref().map(|idx| ids[*idx].clone()),
                speaker: speaker_name.clone(),
                speaker_entity: SpeakerEntity {
                    name: speaker_name,
                    relation,
                },
                topic_tags: topic_tags[i].clone(),
            });
        }

        results
    }

    /// Detect Q&A pairs: if turn[i] ends with `?` and turn[i+1] doesn't, link them.
    /// Returns a Vec<Option<usize>> where each entry is the index of the parent (question) turn,
    /// or None if this turn is not an answer.
    fn detect_qa_pairs(&self, turns: &[(AtomicMemory, String)]) -> Vec<Option<usize>> {
        let mut parent_ids = vec![None; turns.len()];

        for i in 0..turns.len().saturating_sub(1) {
            let current_is_question = self.content_ends_with_question(&turns[i].0.content);
            let next_is_question = self.content_ends_with_question(&turns[i + 1].0.content);

            // Link answer to question: current ends with ?, next doesn't
            if current_is_question && !next_is_question {
                parent_ids[i + 1] = Some(i);
            }
        }

        parent_ids
    }

    /// Check if content ends with a question mark (ignoring trailing whitespace)
    fn content_ends_with_question(&self, content: &str) -> bool {
        content.trim_end().ends_with('?')
    }

    /// Compute topic threading tags for consecutive turns.
    ///
    /// Two consecutive turns share a topic tag if their stemmed-token Jaccard
    /// similarity >= 0.3. The tag is formatted as "topic:{stem1}+{stem2}+..." using
    /// the intersection stems (alphabetically sorted).
    fn compute_topic_threads(
        &self,
        turns: &[(AtomicMemory, String)],
    ) -> Vec<Vec<String>> {
        let stemmer = Stemmer::create(Algorithm::English);
        let mut tags_per_turn: Vec<Vec<String>> = vec![Vec::new(); turns.len()];

        // Pre-compute stemmed token sets for each turn
        let stem_sets: Vec<HashSet<String>> = turns
            .iter()
            .map(|(memory, _)| Self::stemmed_tokens(&stemmer, &memory.content))
            .collect();

        for i in 0..turns.len().saturating_sub(1) {
            let jaccard = Self::jaccard_similarity(&stem_sets[i], &stem_sets[i + 1]);

            if jaccard >= 0.3 {
                // Build a topic tag from intersection stems
                let intersection: Vec<String> = stem_sets[i]
                    .intersection(&stem_sets[i + 1])
                    .cloned()
                    .collect::<Vec<_>>();
                let mut sorted = intersection;
                sorted.sort();
                // Limit tag length to avoid pathologically long tags
                let tag_stems: Vec<&str> = sorted.iter().take(5).map(|s| s.as_str()).collect();
                let tag = format!("topic:{}", tag_stems.join("+"));

                // Add to both turns (they share this topic)
                if !tags_per_turn[i].contains(&tag) {
                    tags_per_turn[i].push(tag.clone());
                }
                if !tags_per_turn[i + 1].contains(&tag) {
                    tags_per_turn[i + 1].push(tag);
                }
            }
        }

        tags_per_turn
    }

    /// Compute stemmed tokens from content, excluding stopwords
    fn stemmed_tokens(stemmer: &Stemmer, content: &str) -> HashSet<String> {
        let stopwords: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "must", "shall", "can", "need", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "or", "if", "because", "while", "although", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
            "who", "whom", "its", "his", "her", "their", "my", "your", "our",
        ]
        .into_iter()
        .collect();

        content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2 && !stopwords.contains(w))
            .map(|w| stemmer.stem(w).to_string())
            .collect()
    }

    /// Jaccard similarity between two sets
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

    /// Strip common role prefixes from content (e.g., "User: hello" -> "hello")
    fn strip_role_prefix(content: &str) -> String {
        // Match patterns like "User:", "Assistant:", "System:", or "Name:" at start of content
        let trimmed = content.trim_start();
        if let Some(colon_pos) = trimmed.find(':') {
            // Only strip if the prefix is short (< 30 chars) and looks like a role
            if colon_pos < 30 {
                let prefix = &trimmed[..colon_pos];
                let is_role = prefix
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == ' ');
                if is_role && !prefix.is_empty() {
                    return trimmed[colon_pos + 1..].trim_start().to_string();
                }
            }
        }
        content.to_string()
    }

    /// Normalize a speaker name: trim, lowercase, collapse whitespace
    fn normalize_speaker(role: &str) -> String {
        role.trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
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
            engine.extract_simple_entities("I decided to use Rust for the veld project");

        assert!(entities.contains(&"rust".to_string()));
        assert!(entities.contains(&"veld".to_string()));
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

    // =========================================================================
    // Conversation-level segmentation tests
    // =========================================================================

    #[test]
    fn test_conversation_qa_pair_detection() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "What database should we use for this project?".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "I recommend RocksDB because it has excellent write performance and embedded key-value storage.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::Cortex);

        assert_eq!(results.len(), 2);
        // First turn (question) has no parent
        assert!(results[0].parent_id.is_none());
        // Second turn (answer) links to the question
        assert!(results[1].parent_id.is_some());
        assert_eq!(results[1].parent_id.as_ref().unwrap(), &results[0].id);
    }

    #[test]
    fn test_conversation_no_qa_when_both_questions() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "What database should we use?".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "Do you need SQL or NoSQL compatibility?".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::Cortex);

        assert_eq!(results.len(), 2);
        // Both are questions, no Q&A linking
        assert!(results[0].parent_id.is_none());
        assert!(results[1].parent_id.is_none());
    }

    #[test]
    fn test_conversation_speaker_attribution() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "Alice".to_string(),
                content: "I think we should refactor the authentication module.".to_string(),
            },
            ConversationTurn {
                role: "Bob".to_string(),
                content: "Agreed, the current auth code has too many edge cases.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::UserApi);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].speaker, "alice");
        assert_eq!(results[1].speaker, "bob");
        assert_eq!(results[0].speaker_entity.name, "alice");
        assert_eq!(results[1].speaker_entity.name, "bob");
        assert_eq!(results[0].speaker_entity.relation, "speaker");
        assert_eq!(results[1].speaker_entity.relation, "speaker");
    }

    #[test]
    fn test_conversation_speaker_questioner_responder() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "How does the Hebbian learning work in veld?".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "Hebbian learning in veld strengthens edges between co-activated entities.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::Cortex);

        assert_eq!(results[0].speaker_entity.relation, "questioner");
        assert_eq!(results[1].speaker_entity.relation, "responder");
    }

    #[test]
    fn test_conversation_topic_threading() {
        let engine = SegmentationEngine::new();

        // Two turns with high stemmed-token overlap (both about Rust memory performance)
        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "Rust memory performance looks impressive for our storage system.".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "Rust memory performance benchmarks confirm impressive storage system throughput.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::UserApi);

        assert_eq!(results.len(), 2);
        // Both should share at least one topic tag
        assert!(!results[0].topic_tags.is_empty());
        assert!(!results[1].topic_tags.is_empty());
        // They should share the same tag
        let shared: Vec<_> = results[0]
            .topic_tags
            .iter()
            .filter(|t| results[1].topic_tags.contains(t))
            .collect();
        assert!(!shared.is_empty());
        // Topic tags start with "topic:"
        assert!(shared[0].starts_with("topic:"));
    }

    #[test]
    fn test_conversation_no_topic_threading_unrelated() {
        let engine = SegmentationEngine::new();

        // Two completely unrelated turns
        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "The weather forecast says heavy rain tomorrow afternoon.".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "The Rust compiler caught three lifetime errors in the authentication module.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::UserApi);

        assert_eq!(results.len(), 2);
        // No shared topic tags between unrelated turns
        let shared: Vec<_> = results[0]
            .topic_tags
            .iter()
            .filter(|t| results[1].topic_tags.contains(t))
            .collect();
        assert!(shared.is_empty());
    }

    #[test]
    fn test_conversation_role_prefix_stripping() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "User: What is the capital of France?".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "Assistant: The capital of France is Paris.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::Cortex);

        assert_eq!(results.len(), 2);
        // Content should have role prefix stripped
        assert!(results[0].memory.content.starts_with("What"));
        assert!(results[1].memory.content.starts_with("The"));
    }

    #[test]
    fn test_conversation_empty_turns() {
        let engine = SegmentationEngine::new();

        let turns: Vec<ConversationTurn> = Vec::new();
        let results = engine.segment_conversation(&turns, InputSource::UserApi);
        assert!(results.is_empty());
    }

    #[test]
    fn test_conversation_single_turn() {
        let engine = SegmentationEngine::new();

        let turns = vec![ConversationTurn {
            role: "User".to_string(),
            content: "I decided to use RocksDB for the persistent storage layer.".to_string(),
        }];

        let results = engine.segment_conversation(&turns, InputSource::UserApi);

        assert_eq!(results.len(), 1);
        assert!(results[0].parent_id.is_none());
        assert_eq!(results[0].speaker, "user");
        assert!(results[0].topic_tags.is_empty()); // No consecutive turn to thread with
    }

    #[test]
    fn test_conversation_multi_turn_qa_chain() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "What embedding model should we use?".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "I recommend MiniLM for its balance of quality and speed in our embeddings pipeline.".to_string(),
            },
            ConversationTurn {
                role: "User".to_string(),
                content: "How many dimensions does MiniLM produce?".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "MiniLM produces 384-dimensional embedding vectors.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::Cortex);

        assert_eq!(results.len(), 4);
        // Turn 0 (Q) -> no parent
        assert!(results[0].parent_id.is_none());
        // Turn 1 (A) -> parent is turn 0
        assert_eq!(results[1].parent_id.as_ref().unwrap(), &results[0].id);
        // Turn 2 (Q) -> no parent
        assert!(results[2].parent_id.is_none());
        // Turn 3 (A) -> parent is turn 2
        assert_eq!(results[3].parent_id.as_ref().unwrap(), &results[2].id);
    }

    #[test]
    fn test_conversation_blank_content_skipped() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "What is Hebbian learning in the context of memory systems?".to_string(),
            },
            ConversationTurn {
                role: "System".to_string(),
                content: "   ".to_string(), // Blank content
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "Hebbian learning strengthens connections between neurons that fire together.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::Cortex);

        // Blank turn should be skipped
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_strip_role_prefix() {
        assert_eq!(
            SegmentationEngine::strip_role_prefix("User: hello world"),
            "hello world"
        );
        assert_eq!(
            SegmentationEngine::strip_role_prefix("Assistant: the answer is 42"),
            "the answer is 42"
        );
        // Don't strip if no colon
        assert_eq!(
            SegmentationEngine::strip_role_prefix("hello world"),
            "hello world"
        );
        // Don't strip long prefixes (> 30 chars)
        let long_prefix = format!("{}: content", "a".repeat(35));
        assert_eq!(
            SegmentationEngine::strip_role_prefix(&long_prefix),
            long_prefix
        );
    }

    #[test]
    fn test_normalize_speaker() {
        assert_eq!(SegmentationEngine::normalize_speaker("User"), "user");
        assert_eq!(
            SegmentationEngine::normalize_speaker("  Alice  Bob  "),
            "alice bob"
        );
        assert_eq!(
            SegmentationEngine::normalize_speaker("ASSISTANT"),
            "assistant"
        );
    }

    #[test]
    fn test_stemmed_tokens() {
        let stemmer = Stemmer::create(Algorithm::English);
        let tokens = SegmentationEngine::stemmed_tokens(&stemmer, "Running performances are impressive");
        // "running" -> "run", "performances" -> "perform", "impressive" -> "impress"
        // "are" is a stopword
        assert!(tokens.contains("run") || tokens.contains("running"));
        assert!(!tokens.contains("are")); // stopword excluded
        assert!(tokens.len() >= 2); // at least some non-stopword tokens
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: HashSet<String> = ["rust", "memory", "performance"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let b: HashSet<String> = ["rust", "memory", "safety"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let sim = SegmentationEngine::jaccard_similarity(&a, &b);
        // 2 intersection / 4 union = 0.5
        assert!((sim - 0.5).abs() < 0.01);

        // Empty sets
        let empty: HashSet<String> = HashSet::new();
        assert_eq!(SegmentationEngine::jaccard_similarity(&empty, &empty), 0.0);
    }

    #[test]
    fn test_conversation_unique_ids() {
        let engine = SegmentationEngine::new();

        let turns = vec![
            ConversationTurn {
                role: "User".to_string(),
                content: "First message about the Rust programming language.".to_string(),
            },
            ConversationTurn {
                role: "Assistant".to_string(),
                content: "Second message about memory safety in Rust systems.".to_string(),
            },
            ConversationTurn {
                role: "User".to_string(),
                content: "Third message about performance benchmarks for the system.".to_string(),
            },
        ];

        let results = engine.segment_conversation(&turns, InputSource::UserApi);

        // All IDs should be unique
        let ids: HashSet<_> = results.iter().map(|r| r.id.0).collect();
        assert_eq!(ids.len(), results.len());
    }
}
