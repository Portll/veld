//! Text Chunking for Long-Content Embeddings
//!
//! MiniLM is configured with a 256 token window, and this module splits longer content so each
//! piece is separately searchable rather than only the opening of a memory being findable.
//!
//! The header used to end "Content beyond this is silently dropped." That was true when written
//! and is no longer: `minilm.rs` embeds the whole input by pooling across windows, so an
//! over-length chunk is now blurred rather than cut off. **The sentence is corrected rather than
//! deleted because it described a real defect that this module did not actually prevent** — see
//! the measurement on `chunk_size` below.
//!
//! This module implements two chunking strategies:
//!
//! ## 1. Fixed-Size Overlapping Chunking (`chunk_text`)
//! - Split text into ~200 token chunks (leaving room for special tokens)
//! - 50 token overlap between chunks for context continuity
//! - Good for general prose and documents
//!
//! ## 2. Semantic Chunking (`semantic_chunk_text`)
//! - Splits on natural boundaries (paragraphs, dialogue turns, sections)
//! - Preserves conversational context - never splits mid-turn
//! - Better for dialogue, structured text, logs, and multi-speaker content
//!
//! # Example (Fixed-Size)
//!
//! A 1000-token memory becomes ~6 chunks:
//! ```text
//! [0-200] [150-350] [300-500] [450-650] [600-800] [750-1000]
//! ```
//!
//! # Example (Semantic)
//!
//! A dialogue becomes natural chunks preserving speaker turns:
//! ```text
//! [Alice: Hi / Bob: Hello] [Alice: How are you? / Bob: Great!]
//! ```

use regex::Regex;
use std::sync::LazyLock;

/// Pattern to detect dialogue turns (e.g., "Alice:", "User:", "Speaker 1:")
static DIALOGUE_TURN_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^([A-Z][a-zA-Z0-9_\- ]{0,30})\s*:").unwrap());

/// Pattern to detect section headers or timestamps
static SECTION_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(?:\[.*?\]|#{1,3}\s+\w|Session \d+|---+)").unwrap());

/// Chunk configuration for fixed-size chunking
pub struct ChunkConfig {
    /// Target chunk size in characters (approximate tokens * 4)
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Minimum chunk size (don't create tiny trailing chunks)
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            // DERIVED FROM MEASUREMENT, not from an assumed characters-per-token ratio.
            //
            // This was 800, commented "~200 tokens * 4 chars/token ... leave headroom for tokenizer
            // differences". The headroom ran the wrong way: the assumption fails toward chunks that
            // are too BIG, because dense text has FEWER chars per token, not more. This corpus is
            // paths, JSON, SHAs and slugs, and its densest chunks run under 2 chars/token.
            //
            // `chunk-audit --sweep` over 300 real memories against the real tokenizer, counting how
            // many chunks exceed the 256-token window:
            //
            //     chunk_size   chunks   over   max tokens
            //            400     2431      0          248
            //            480     2046      2          269
            //            640     1564     10          322
            //            800     1278     58          412   <- the old value
            //
            // 400 is the only value with zero overflow, at 1.9x the vectors. 480 costs 1.6x and
            // leaves 2 chunks in 2,046 (0.1%) spanning two windows. Since `minilm.rs` now pools
            // across windows rather than truncating, an over-length chunk is blurred and not lost,
            // so 0.1% buys a third of the index back. That trade is why this is 480 and not 400,
            // and the table is here so the next person can pick differently with the same numbers
            // instead of re-deriving them.
            chunk_size: 480,
            // Held at a quarter of chunk_size, the ratio the 800/200 pair used.
            overlap: 120,
            // Don't create chunks smaller than ~30 tokens
            min_chunk_size: 120,
        }
    }
}

/// Result of chunking a text
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The chunked text segments
    pub chunks: Vec<String>,
    /// Original text length
    pub original_length: usize,
    /// Whether chunking was needed (content exceeded single chunk)
    pub was_chunked: bool,
    /// Bytes of the original actually present in some chunk, MEASURED rather than assumed.
    ///
    /// The chunk loop can leave a gap: it breaks on `start >= end`, and it force-advances `start`
    /// when the overlap arithmetic fails to make progress. Neither is hypothetical enough to leave
    /// unmeasured, and a coverage number that cannot go below 1.0 cannot report either.
    pub covered_length: usize,
}

impl ChunkResult {
    /// Fraction of the original text present in the chunks.
    ///
    /// This used to be `1.0`, unconditionally, with the comment "With chunking, we cover
    /// everything". Its only caller was a test asserting it equals 1.0 — a metric, and a test of
    /// that metric, that agreed with each other by construction and could not have disagreed with
    /// anything else. The value it now returns is derived from the spans the chunker actually
    /// emitted, so a gap shows up as a number below 1.0 instead of as nothing at all.
    pub fn coverage_ratio(&self) -> f32 {
        if self.chunks.is_empty() || self.original_length == 0 {
            return 0.0;
        }
        (self.covered_length as f32 / self.original_length as f32).min(1.0)
    }
}

/// Find the nearest valid char boundary at or before the given byte index
#[inline]
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    // Walk backwards to find a valid char boundary
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Find the nearest valid char boundary at or after the given byte index
#[inline]
fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    // Walk forwards to find a valid char boundary
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Chunk text into overlapping segments for embedding
///
/// Uses sentence-aware splitting to avoid breaking mid-sentence when possible.
pub fn chunk_text(text: &str, config: &ChunkConfig) -> ChunkResult {
    let text = text.trim();
    let original_length = text.len();

    // If text fits in a single chunk, no need to split
    if original_length <= config.chunk_size {
        return ChunkResult {
            chunks: vec![text.to_string()],
            original_length,
            was_chunked: false,
            covered_length: original_length,
        };
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    // Union of the [start, end) spans emitted, walked in order. `frontier` is the furthest byte
    // any chunk has reached; anything the loop skips past never gets counted.
    let mut covered_length = 0usize;
    let mut frontier = 0usize;

    while start < original_length {
        // Calculate end position for this chunk, ensuring valid char boundary
        let mut end = floor_char_boundary(text, (start + config.chunk_size).min(original_length));

        // If we're not at the end, try to break at a sentence boundary
        if end < original_length {
            end = find_break_point(text, start, end, config.min_chunk_size);
            // Ensure the break point is on a valid char boundary
            end = floor_char_boundary(text, end);
        }

        // Ensure start is on a valid char boundary
        start = ceil_char_boundary(text, start);

        // Safety check: ensure we don't have start >= end
        if start >= end {
            break;
        }

        // Extract chunk (now safe - both start and end are valid char boundaries)
        if end > frontier {
            covered_length += end - frontier.max(start);
            frontier = end;
        }
        let chunk = text[start..end].trim();
        if chunk.len() >= config.min_chunk_size || chunks.is_empty() {
            chunks.push(chunk.to_string());
        } else if let Some(last) = chunks.last_mut() {
            // Append tiny trailing chunk to previous
            last.push(' ');
            last.push_str(chunk);
        }

        // Move start position, accounting for overlap
        if end >= original_length {
            break;
        }
        // The overlap must never carry the next chunk back to this one's start: a break point
        // exactly `overlap` bytes in, on a slice too short to emit, otherwise repeats forever while
        // the last chunk grows without bound.
        let next = ceil_char_boundary(text, end.saturating_sub(config.overlap));
        start = if next <= start
            || next <= chunks.len().saturating_sub(1) * (config.chunk_size - config.overlap)
        {
            ceil_char_boundary(text, end)
        } else {
            next
        };
    }

    ChunkResult {
        chunks,
        original_length,
        was_chunked: true,
        covered_length,
    }
}

/// Find a good break point for chunking (sentence or word boundary)
fn find_break_point(text: &str, start: usize, ideal_end: usize, min_size: usize) -> usize {
    let chunk = &text[start..ideal_end];

    // Try to find sentence boundary (. ! ?) followed by space or end
    let sentence_boundaries: Vec<usize> = chunk
        .char_indices()
        .filter_map(|(byte_offset, c)| {
            if (c == '.' || c == '!' || c == '?') && byte_offset >= min_size {
                // Check if followed by space or end of chunk
                let after = byte_offset + c.len_utf8();
                let next_char = chunk[after..].chars().next();
                if next_char.is_none_or(|nc| nc.is_whitespace()) {
                    return Some(start + after);
                }
            }
            None
        })
        .collect();

    // Use the last sentence boundary if available
    if let Some(&boundary) = sentence_boundaries.last() {
        return boundary;
    }

    // Fall back to word boundary
    let word_boundaries: Vec<usize> = chunk
        .char_indices()
        .filter_map(|(i, c)| {
            if c.is_whitespace() && i >= min_size {
                Some(start + i)
            } else {
                None
            }
        })
        .collect();

    if let Some(&boundary) = word_boundaries.last() {
        return boundary;
    }

    // No good boundary found, use ideal_end
    ideal_end
}

/// Estimate token count for text (improved approximation)
///
/// Uses word-based estimation with adjustment for BPE subword tokenization.
/// More accurate than simple character division for mixed content (prose, code, numbers).
///
/// Accuracy: ~85-90% for English prose, ~75-85% for code/mixed content.
/// For exact counts, use a proper tokenizer like tiktoken.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let words = text.split_whitespace().count();
    if words == 0 {
        // Text with no whitespace (e.g., single long token or CJK)
        // Fall back to character-based estimate
        return text.chars().count().div_ceil(4);
    }

    // BPE tokenization typically splits words into ~1.3 subword tokens on average
    // Code and technical content have more splits (camelCase, snake_case, etc.)
    let base_tokens = (words as f64 * 1.3).ceil() as usize;

    // Add tokens for punctuation and special characters not attached to words
    // These often become separate tokens
    let special_chars = text
        .chars()
        .filter(|c| c.is_ascii_punctuation() || *c == '\n')
        .count();
    let punct_tokens = special_chars / 3; // ~3 punct chars per token on average

    base_tokens + punct_tokens
}

/// Configuration for semantic chunking
pub struct SemanticChunkConfig {
    /// Target chunk size in characters
    pub target_size: usize,
    /// Maximum chunk size (hard limit)
    pub max_size: usize,
    /// Minimum chunk size (merge smaller segments)
    pub min_size: usize,
    /// Whether to preserve dialogue turns intact
    pub preserve_dialogue_turns: bool,
    /// Whether to split on paragraph boundaries
    pub split_on_paragraphs: bool,
}

impl Default for SemanticChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 800,
            max_size: 1200,
            min_size: 100,
            preserve_dialogue_turns: true,
            split_on_paragraphs: true,
        }
    }
}

/// A semantic segment with metadata
#[derive(Debug, Clone)]
struct SemanticSegment {
    text: String,
    #[allow(dead_code)]
    segment_type: SegmentType,
}

#[derive(Debug, Clone, PartialEq)]
enum SegmentType {
    DialogueTurn,
    Paragraph,
    Section,
    Text,
}

/// Semantic chunking: splits text on natural boundaries (dialogue turns, paragraphs, sections)
/// and groups related content together.
///
/// This is better for conversational content, logs, and structured text than fixed-size chunking.
pub fn semantic_chunk_text(text: &str, config: &SemanticChunkConfig) -> ChunkResult {
    let text = text.trim();
    let original_length = text.len();

    // If text fits in a single chunk, no need to split
    if original_length <= config.target_size {
        return ChunkResult {
            chunks: vec![text.to_string()],
            original_length,
            was_chunked: false,
            covered_length: original_length,
        };
    }

    // Step 1: Split into semantic segments
    let segments = split_into_segments(text, config);

    // Step 2: Group segments into chunks respecting size constraints
    let chunks = group_segments_into_chunks(segments, config);

    // Semantic chunking groups whole segments rather than slicing spans, so there are no
    // [start, end) offsets to union the way `chunk_text` does. Summing the emitted bytes is a
    // WEAKER guarantee, and the weakness is worth naming: it counts content, so a dropped segment
    // shows up, but it cannot tell a gap from a gap that an overlap elsewhere happens to offset.
    // A lower bound that can fall is still worth more than a 1.0 that cannot.
    let covered_length = chunks
        .iter()
        .map(String::len)
        .sum::<usize>()
        .min(original_length);

    ChunkResult {
        chunks,
        original_length,
        was_chunked: true,
        covered_length,
    }
}

/// Split text into semantic segments based on structure
fn split_into_segments(text: &str, config: &SemanticChunkConfig) -> Vec<SemanticSegment> {
    let mut segments = Vec::new();

    // Check if this looks like dialogue (has speaker patterns)
    let is_dialogue = config.preserve_dialogue_turns && DIALOGUE_TURN_PATTERN.is_match(text);

    if is_dialogue {
        // Split by dialogue turns
        let turn_starts: Vec<usize> = DIALOGUE_TURN_PATTERN
            .find_iter(text)
            .map(|m| m.start())
            .collect();

        // Add any text before the first turn
        if !turn_starts.is_empty() && turn_starts[0] > 0 {
            let pre_text = text[..turn_starts[0]].trim();
            if !pre_text.is_empty() {
                segments.push(SemanticSegment {
                    text: pre_text.to_string(),
                    segment_type: SegmentType::Text,
                });
            }
        }

        for (i, &start) in turn_starts.iter().enumerate() {
            let end = if i + 1 < turn_starts.len() {
                turn_starts[i + 1]
            } else {
                text.len()
            };

            let turn_text = text[start..end].trim();
            if !turn_text.is_empty() {
                segments.push(SemanticSegment {
                    text: turn_text.to_string(),
                    segment_type: SegmentType::DialogueTurn,
                });
            }
        }
    } else if config.split_on_paragraphs {
        // Split by paragraphs (double newlines) or section markers
        let paragraph_pattern = Regex::new(r"\n\s*\n").unwrap();
        let mut last_end = 0;

        for mat in paragraph_pattern.find_iter(text) {
            if mat.start() > last_end {
                let para_text = text[last_end..mat.start()].trim();
                if !para_text.is_empty() {
                    let seg_type = if SECTION_PATTERN.is_match(para_text) {
                        SegmentType::Section
                    } else {
                        SegmentType::Paragraph
                    };
                    segments.push(SemanticSegment {
                        text: para_text.to_string(),
                        segment_type: seg_type,
                    });
                }
            }
            last_end = mat.end();
        }

        // Add remaining text
        if last_end < text.len() {
            let remaining = text[last_end..].trim();
            if !remaining.is_empty() {
                segments.push(SemanticSegment {
                    text: remaining.to_string(),
                    segment_type: SegmentType::Paragraph,
                });
            }
        }
    } else {
        // Fall back to sentence-based splitting
        segments = split_by_sentences(text);
    }

    // If no segments found, treat entire text as one segment
    if segments.is_empty() {
        segments.push(SemanticSegment {
            text: text.to_string(),
            segment_type: SegmentType::Text,
        });
    }

    segments
}

/// Split text by sentences for fallback
fn split_by_sentences(text: &str) -> Vec<SemanticSegment> {
    let sentence_pattern = Regex::new(r"[.!?]+\s+").unwrap();
    let mut segments = Vec::new();
    let mut last_end = 0;

    for mat in sentence_pattern.find_iter(text) {
        let sentence = text[last_end..mat.end()].trim();
        if !sentence.is_empty() {
            segments.push(SemanticSegment {
                text: sentence.to_string(),
                segment_type: SegmentType::Text,
            });
        }
        last_end = mat.end();
    }

    // Add remaining text
    if last_end < text.len() {
        let remaining = text[last_end..].trim();
        if !remaining.is_empty() {
            segments.push(SemanticSegment {
                text: remaining.to_string(),
                segment_type: SegmentType::Text,
            });
        }
    }

    segments
}

/// Group segments into chunks respecting size constraints
fn group_segments_into_chunks(
    segments: Vec<SemanticSegment>,
    config: &SemanticChunkConfig,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for segment in segments {
        let segment_len = segment.text.len();

        // If segment alone exceeds max size, we need to split it
        if segment_len > config.max_size {
            // Flush current chunk first
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }

            // Split the large segment using fixed-size chunking
            let fixed_config = ChunkConfig {
                chunk_size: config.target_size,
                overlap: config.min_size / 2,
                min_chunk_size: config.min_size,
            };
            let sub_chunks = chunk_text(&segment.text, &fixed_config);
            chunks.extend(sub_chunks.chunks);
            continue;
        }

        // Check if adding this segment would exceed target
        let new_len = current_chunk.len() + segment_len + 1; // +1 for newline

        if new_len > config.target_size && !current_chunk.is_empty() {
            // Flush current chunk
            chunks.push(current_chunk.trim().to_string());
            current_chunk = String::new();
        }

        // Add segment to current chunk
        if !current_chunk.is_empty() {
            current_chunk.push('\n');
        }
        current_chunk.push_str(&segment.text);
    }

    // Flush remaining chunk
    if !current_chunk.is_empty() {
        let trimmed = current_chunk.trim().to_string();
        // Merge tiny trailing chunk with previous if too small
        if trimmed.len() < config.min_size && !chunks.is_empty() {
            let last = chunks.pop().unwrap_or_default();
            chunks.push(format!("{last}\n{trimmed}"));
        } else {
            chunks.push(trimmed);
        }
    }

    chunks
}

/// Detect if text appears to be dialogue/conversation format
pub fn is_dialogue_format(text: &str) -> bool {
    DIALOGUE_TURN_PATTERN.is_match(text)
}

/// Auto-select the best chunking strategy based on content
pub fn auto_chunk_text(text: &str) -> ChunkResult {
    if is_dialogue_format(text) {
        semantic_chunk_text(text, &SemanticChunkConfig::default())
    } else {
        chunk_text(text, &ChunkConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_advances_past_a_break_exactly_one_overlap_in() {
        // From byte 1 the only whitespace at or past `min_chunk_size` is the newline 120 bytes in,
        // the trimmed slice before it is too short to emit, and the overlap lands back on byte 1.
        let text = format!("y{}   \n{}", "x".repeat(117), "z".repeat(361));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(chunk_text(&text, &ChunkConfig::default()));
        });
        let result = rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("chunk_text did not terminate");
        assert!(result.chunks.len() <= 3, "{} chunks", result.chunks.len());
        assert_eq!(result.covered_length, result.original_length);
    }

    #[test]
    fn test_short_text_no_chunking() {
        let config = ChunkConfig::default();
        let result = chunk_text("This is a short text.", &config);

        assert_eq!(result.chunks.len(), 1);
        assert!(!result.was_chunked);
        assert_eq!(result.chunks[0], "This is a short text.");
    }

    #[test]
    fn test_long_text_chunking() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 20,
            min_chunk_size: 30,
        };

        // Create text longer than chunk_size
        let text = "This is sentence one. This is sentence two. This is sentence three. \
                   This is sentence four. This is sentence five. This is sentence six. \
                   This is sentence seven. This is sentence eight.";

        let result = chunk_text(text, &config);

        assert!(result.was_chunked);
        assert!(result.chunks.len() > 1);

        // Verify each chunk has meaningful content
        for chunk in &result.chunks {
            assert!(
                chunk.len() >= config.min_chunk_size,
                "Chunk too small: '{}' (len={})",
                chunk,
                chunk.len()
            );
        }

        // Verify total chunked content is at least as long as original (with overlaps)
        let total_len: usize = result.chunks.iter().map(|c| c.len()).sum();
        assert!(
            total_len >= result.original_length,
            "Total chunk length {} < original {}",
            total_len,
            result.original_length
        );
    }

    #[test]
    fn test_sentence_boundary_respected() {
        let config = ChunkConfig {
            chunk_size: 50,
            overlap: 10,
            min_chunk_size: 20,
        };

        let text = "First sentence here. Second sentence follows. Third sentence ends.";
        let result = chunk_text(text, &config);

        // Chunks should end at sentence boundaries when possible
        for chunk in &result.chunks {
            let trimmed = chunk.trim();
            if !trimmed.is_empty() && result.chunks.len() > 1 {
                // Most chunks should end with sentence-ending punctuation
                let last_char = trimmed.chars().last().unwrap();
                // Allow some flexibility - not all chunks will end at sentence boundary
                assert!(
                    last_char == '.'
                        || last_char == '!'
                        || last_char == '?'
                        || chunk == result.chunks.last().unwrap(),
                    "Chunk '{chunk}' doesn't end at sentence boundary"
                );
            }
        }
    }

    #[test]
    fn test_overlap_exists() {
        let config = ChunkConfig {
            chunk_size: 60,
            overlap: 20,
            min_chunk_size: 20,
        };

        let text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ KKKK LLLL MMMM";
        let result = chunk_text(text, &config);

        if result.chunks.len() >= 2 {
            // Check that consecutive chunks have overlapping content
            for i in 0..result.chunks.len() - 1 {
                let chunk1 = &result.chunks[i];
                let chunk2 = &result.chunks[i + 1];

                // Find common words
                let words1: std::collections::HashSet<_> = chunk1.split_whitespace().collect();
                let words2: std::collections::HashSet<_> = chunk2.split_whitespace().collect();
                let common: Vec<_> = words1.intersection(&words2).collect();

                // Should have some overlap
                assert!(
                    !common.is_empty() || chunk1.len() < config.overlap,
                    "No overlap between chunks {} and {}",
                    i,
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_token_estimation() {
        // Empty string
        assert_eq!(estimate_tokens(""), 0);

        // Single word: 1 * 1.3 = 2 (ceil)
        assert_eq!(estimate_tokens("test"), 2);

        // Two words: 2 * 1.3 = 3 (ceil)
        assert_eq!(estimate_tokens("hello world"), 3);

        // Sentence with punctuation: 5 words * 1.3 = 7 (ceil) + 1 punct token (3 punct chars / 3)
        assert_eq!(estimate_tokens("Hello, world! How are you?"), 8);

        // Code-like content with more punctuation
        let code = "fn main() { println!(\"hello\"); }";
        let tokens = estimate_tokens(code);
        assert!((5..=15).contains(&tokens), "Code tokens: {}", tokens);

        // No whitespace (falls back to char-based)
        assert_eq!(estimate_tokens("abcdefgh"), 2); // 8 chars / 4 = 2
    }

    #[test]
    fn test_very_long_content() {
        let config = ChunkConfig::default();

        // Create 10KB of content
        let long_text = "This is a test sentence. ".repeat(400);
        let result = chunk_text(&long_text, &config);

        assert!(result.was_chunked);
        assert!(result.chunks.len() > 10); // Should have many chunks

        // This assertion used to be the ONLY caller of coverage_ratio(), and coverage_ratio()
        // returned a hardcoded 1.0 — so it agreed with the function by construction and could not
        // have caught anything. It now checks a measured value, and the two assertions below make
        // sure it is a measurement rather than a constant that happens to read 1.0.
        assert_eq!(result.coverage_ratio(), 1.0);
        assert_eq!(
            result.covered_length, result.original_length,
            "chunking left a gap: {} of {} bytes reached a chunk",
            result.covered_length, result.original_length
        );

        // Non-vacuity: the value must be able to FALL. A metric whose only observed output is 1.0
        // is indistinguishable from one that cannot report a problem, which is what this was.
        let partial = ChunkResult {
            chunks: vec!["half".to_string()],
            original_length: 100,
            was_chunked: true,
            covered_length: 40,
        };
        assert_eq!(
            partial.coverage_ratio(),
            0.4,
            "coverage_ratio must report incomplete coverage, not round it up to 1.0"
        );

        // Verify no chunk exceeds config size significantly
        for chunk in &result.chunks {
            assert!(
                chunk.len() <= config.chunk_size + 100,
                "Chunk too large: {} chars",
                chunk.len()
            );
        }
    }

    #[test]
    fn test_chunking_quality_unique_content_searchable() {
        let config = ChunkConfig::default();

        // Create content with UNIQUE markers at beginning, middle, and end
        // These markers should each appear in at least one chunk
        let beginning = "ALPHA_BEGINNING_MARKER is a unique identifier at the start.";
        let middle_padding = "This is filler content to push things apart. ".repeat(30);
        let middle = "BETA_MIDDLE_MARKER represents content in the center of the document.";
        let end_padding = "More filler content for separation between sections. ".repeat(30);
        let end = "GAMMA_END_MARKER signifies the conclusion of this memory content.";

        let full_text = format!("{beginning} {middle_padding} {middle} {end_padding} {end}");

        let result = chunk_text(&full_text, &config);

        // Content should be chunked
        assert!(result.was_chunked, "Content should require chunking");
        assert!(result.chunks.len() >= 3, "Should have multiple chunks");

        // Verify each unique marker appears in at least one chunk
        let has_alpha = result.chunks.iter().any(|c| c.contains("ALPHA_BEGINNING"));
        let has_beta = result.chunks.iter().any(|c| c.contains("BETA_MIDDLE"));
        let has_gamma = result.chunks.iter().any(|c| c.contains("GAMMA_END"));

        assert!(has_alpha, "ALPHA marker (beginning) not found in any chunk");
        assert!(has_beta, "BETA marker (middle) not found in any chunk");
        assert!(has_gamma, "GAMMA marker (end) not found in any chunk");

        // Log chunk info for debugging
        println!("Total chunks: {}", result.chunks.len());
        println!("Original length: {} chars", result.original_length);
        for (i, chunk) in result.chunks.iter().enumerate() {
            let markers: Vec<&str> = vec![
                if chunk.contains("ALPHA") { "ALPHA" } else { "" },
                if chunk.contains("BETA") { "BETA" } else { "" },
                if chunk.contains("GAMMA") { "GAMMA" } else { "" },
            ]
            .into_iter()
            .filter(|m| !m.is_empty())
            .collect();
            println!(
                "  Chunk {}: {} chars {}",
                i,
                chunk.len(),
                if markers.is_empty() {
                    String::new()
                } else {
                    format!("[contains: {}]", markers.join(", "))
                }
            );
        }
    }

    #[test]
    fn test_chunking_coverage_no_content_lost() {
        let config = ChunkConfig {
            chunk_size: 200,
            overlap: 50,
            min_chunk_size: 50,
        };

        // Create text with numbered sentences for easy tracking
        let sentences: Vec<String> = (1..=20)
            .map(|i| format!("Sentence number {i} contains unique information. "))
            .collect();
        let text = sentences.join("");

        let result = chunk_text(&text, &config);

        // Every sentence number should appear in at least one chunk
        for i in 1..=20 {
            let marker = format!("number {i}");
            let found = result.chunks.iter().any(|c| c.contains(&marker));
            assert!(
                found,
                "Sentence {i} not found in any chunk! Coverage gap detected."
            );
        }
    }
}
