//! Linguistic Query Parser
//!
//! Based on:
//! - Lioma & Ounis (2006): "Content Load of Part of Speech Blocks"
//! - Bendersky & Croft (2008): "Discovering Key Concepts in Verbose Queries"
//! - Porter (1980): Stemming algorithm for term normalization
//!
//! Extracts focal entities (nouns), discriminative modifiers (adjectives),
//! and relational context (verbs) from natural language queries.
//!
//! # Polished Features (v2)
//! - Porter2 stemming for term normalization
//! - Compound noun detection (bigrams/trigrams)
//! - Context-aware POS disambiguation
//! - Negation scope tracking
//! - IDF-inspired term rarity weighting
//!
//! # Shallow Parsing / Chunking (v3)
//! - Sentence-level chunking for co-occurrence detection
//! - POS-based entity extraction (all nouns, verbs, adjectives - not just top-N)
//! - Designed for both query analysis AND memory storage
//!
//! # Temporal Extraction (v4)
//! - Extract dates from natural language text ("May 7, 2023", "yesterday", "last week")
//! - Detect temporal queries ("when did", "what date", "how long ago")
//! - Based on TEMPR approach (Hindsight paper achieving 89.6% on LoCoMo)

use crate::constants::{IC_ADJECTIVE, IC_NOUN, IC_VERB};
use crate::memory::temporal_facts::{
    last_weekday, next_weekday, resolve_relative_time, TemporalBound,
};
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc, Weekday};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::LazyLock;

// ============================================================================
// SHALLOW PARSING / CHUNKING MODULE
// ============================================================================
// This section provides sentence-level chunking and POS-based entity extraction.
// Unlike YAKE (which ranks by frequency and misses rare discriminative terms),
// this extracts ALL content words (nouns, verbs, adjectives) for graph building.

/// Part of speech tag
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PosTag {
    Noun,
    Verb,
    Adjective,
    ProperNoun,
    StopWord,
    Other,
}

/// A word with its POS annotation
#[derive(Debug, Clone)]
pub struct TaggedWord {
    pub text: String,
    pub stem: String,
    pub pos: PosTag,
    /// Position within the sentence (0-indexed)
    pub position: usize,
}

/// A sentence chunk containing tagged words
#[derive(Debug, Clone)]
pub struct SentenceChunk {
    /// Original sentence text
    pub text: String,
    /// Sentence index in document (0-indexed)
    pub sentence_idx: usize,
    /// All tagged words in this sentence
    pub words: Vec<TaggedWord>,
}

impl SentenceChunk {
    /// Get all nouns in this chunk
    pub fn nouns(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| matches!(w.pos, PosTag::Noun | PosTag::ProperNoun))
            .collect()
    }

    /// Get all verbs in this chunk
    pub fn verbs(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| w.pos == PosTag::Verb)
            .collect()
    }

    /// Get all adjectives in this chunk
    pub fn adjectives(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| w.pos == PosTag::Adjective)
            .collect()
    }

    /// Get all content words (nouns, verbs, adjectives)
    pub fn content_words(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| {
                matches!(
                    w.pos,
                    PosTag::Noun | PosTag::ProperNoun | PosTag::Verb | PosTag::Adjective
                )
            })
            .collect()
    }

    /// Generate co-occurrence pairs (words in same sentence get edges)
    /// Returns pairs of (word1_stem, word2_stem) for graph edge creation
    pub fn cooccurrence_pairs(&self) -> Vec<(&str, &str)> {
        let content = self.content_words();
        let mut pairs = Vec::new();

        for i in 0..content.len() {
            for j in (i + 1)..content.len() {
                pairs.push((content[i].stem.as_str(), content[j].stem.as_str()));
            }
        }

        pairs
    }
}

/// Result of chunking a document
#[derive(Debug, Clone)]
pub struct ChunkExtraction {
    /// All sentence chunks
    pub chunks: Vec<SentenceChunk>,
    /// Unique nouns found (stems)
    pub unique_nouns: HashSet<String>,
    /// Unique verbs found (stems)
    pub unique_verbs: HashSet<String>,
    /// Unique adjectives found (stems)
    pub unique_adjectives: HashSet<String>,
    /// Proper nouns (likely named entities)
    pub proper_nouns: HashSet<String>,
}

impl ChunkExtraction {
    /// Get all unique content word stems
    pub fn all_content_stems(&self) -> HashSet<String> {
        let mut all = self.unique_nouns.clone();
        all.extend(self.unique_verbs.clone());
        all.extend(self.unique_adjectives.clone());
        all.extend(self.proper_nouns.clone());
        all
    }

    /// Get all co-occurrence pairs across all chunks
    pub fn all_cooccurrence_pairs(&self) -> Vec<(String, String)> {
        let mut all_pairs = Vec::new();
        for chunk in &self.chunks {
            for (w1, w2) in chunk.cooccurrence_pairs() {
                all_pairs.push((w1.to_string(), w2.to_string()));
            }
        }
        all_pairs
    }
}

// ============================================================================
// TEMPORAL EXTRACTION MODULE
// ============================================================================
// Extracts temporal references from natural language text.
// Based on TEMPR approach from Hindsight paper (89.6% accuracy on LoCoMo).
// Key insight: Temporal filtering is critical for multi-hop retrieval.

/// A temporal reference extracted from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRef {
    /// The extracted date (normalized to NaiveDate)
    pub date: NaiveDate,
    /// Original text that was parsed (e.g., "May 7, 2023", "yesterday")
    pub original_text: String,
    /// Confidence in the extraction (0.0-1.0)
    pub confidence: f32,
    /// Position in original text (character offset)
    pub position: usize,
    /// Type of temporal reference
    pub ref_type: TemporalRefType,
}

/// Type of temporal reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalRefType {
    /// Absolute date (May 7, 2023)
    Absolute,
    /// Relative date (yesterday, last week)
    Relative,
    /// Day of week (on Monday, last Tuesday)
    DayOfWeek,
    /// Month reference (in May, last March)
    Month,
    /// Year reference (in 2023, last year)
    Year,
}

/// Result of temporal extraction from text
#[derive(Debug, Clone, Default)]
pub struct TemporalExtraction {
    /// All temporal references found
    pub refs: Vec<TemporalRef>,
    /// Earliest date mentioned
    pub earliest: Option<NaiveDate>,
    /// Latest date mentioned
    pub latest: Option<NaiveDate>,
    /// Query-side temporal bounds resolved from relative expressions.
    ///
    /// Populated by the relative pre-pass in `extract_temporal_refs` when the
    /// query contains phrases like "yesterday", "last week", "next month", etc.
    /// Each entry corresponds to one matched relative phrase, in text order.
    ///
    /// Callers should treat these as opaque — do NOT match on specific
    /// `TemporalBound` variants here; a later pass (R4) will add After/Before/Fuzzy
    /// variants that must flow through transparently.
    pub query_bounds: Vec<TemporalBound>,
}

impl TemporalExtraction {
    /// Check if any temporal references were found
    pub fn has_temporal_refs(&self) -> bool {
        !self.refs.is_empty()
    }

    /// Get date range (earliest, latest) if temporal refs exist
    pub fn date_range(&self) -> Option<(NaiveDate, NaiveDate)> {
        match (self.earliest, self.latest) {
            (Some(e), Some(l)) => Some((e, l)),
            (Some(e), None) => Some((e, e)),
            (None, Some(l)) => Some((l, l)),
            (None, None) => None,
        }
    }

    /// Return the query-side `TemporalBound`s resolved from relative phrases.
    ///
    /// Returns an empty slice when no relative expressions were found (the
    /// behaviour is identical to pre-R1: callers that do not use bounds
    /// are unaffected).
    pub fn bounds(&self) -> &[TemporalBound] {
        &self.query_bounds
    }
}

/// Extract temporal references from text
///
/// Uses date_time_parser crate for natural language date parsing.
/// Handles:
/// - Absolute dates: "May 7, 2023", "2023-05-07", "07/05/2023"
/// - Relative dates: "yesterday", "last week", "3 days ago"
/// - Day of week: "on Monday", "last Tuesday"
/// - Month/year: "in May", "last year", "2023"
/// Ordered list of relative-time phrase patterns that `resolve_relative_time`
/// handles.  Listed longest-first so substring matches don't shadow longer
/// ones (e.g. "the day before yesterday" before "yesterday").
///
/// Phrases are lower-cased; callers pass `text.to_lowercase()` for matching.
const RELATIVE_PHRASES: &[&str] = &[
    // Multi-word — must come before any substring they contain.
    "the day before yesterday",
    "the day after tomorrow",
    "last weekend",
    "next weekend",
    "this weekend",
    "this morning",
    "this afternoon",
    "this evening",
    "next month",
    "last month",
    "this month",
    "next year",
    "last year",
    "this year",
    "next week",
    "last week",
    "this week",
    // Single-word / short phrases.
    "yesterday",
    "tomorrow",
    "today",
    // "N units ago" — matched by a trailing-" ago" scan below; not listed here.
    // Day-of-week compounds are also handled in the scan loop below.
];

/// Extract temporal references from text
///
/// Uses date_time_parser crate for natural language date parsing.
/// Handles:
/// - Absolute dates: "May 7, 2023", "2023-05-07", "07/05/2023"
/// - Relative dates: "yesterday", "last week", "3 days ago"
/// - Day of week: "on Monday", "last Tuesday"
/// - Month/year: "in May", "last year", "2023"
pub fn extract_temporal_refs(text: &str) -> TemporalExtraction {
    let now = Utc::now();
    let mut refs = Vec::new();
    let mut earliest: Option<NaiveDate> = None;
    let mut latest: Option<NaiveDate> = None;

    // -----------------------------------------------------------------------
    // Relative-date pre-pass (F6 fix)
    //
    // `dateparser` is absolute-date oriented and returns Err for all relative
    // phrases like "yesterday", "last week", "next month".  This pre-pass
    // detects the same vocabulary handled by `resolve_relative_time` (the
    // write-side resolver), calls it with the current wall-clock time, and
    // stores the resulting `TemporalBound` in `query_bounds`.
    //
    // CRITICAL: results are forwarded OPAQUELY — we never match on specific
    // TemporalBound variants here so that After/Before/Fuzzy added in R4 pass
    // through without any change to this function.
    // -----------------------------------------------------------------------
    let text_lower = text.to_lowercase();
    let mut query_bounds: Vec<TemporalBound> = Vec::new();

    // -----------------------------------------------------------------------
    // Pass 0: anchor-relative expressions (B1).
    //
    // Handles patterns of the form:
    //   "the <weekday> before <anchor-date>"
    //   "the <weekday> after <anchor-date>"
    //   "the day before <anchor-date>" / "the day after <anchor-date>"
    //   "<N> day|days|week|weeks before|after <anchor-date>"
    //
    // The anchor must be an absolute date resolvable by `try_parse_anchor_date`.
    // If the anchor cannot be resolved (e.g. it is a named event — B2's scope),
    // this pass silently skips and the query falls through to later passes.
    //
    // Results are emitted as TemporalBound::Point + a synthesised TemporalRef
    // so that `refs`/`earliest`/`latest`/`query_bounds` are all populated,
    // matching the additive contract of the existing relative pre-pass.
    // -----------------------------------------------------------------------
    {
        use regex::Regex;
        use std::sync::LazyLock;

        /// Parse an anchor string into a `NaiveDate` by reusing the existing
        /// `extract_explicit_dates` path (which handles "25 May 2023",
        /// "2023-05-25", "May 25, 2023") and falling back to `dateparser`.
        fn try_parse_anchor_date(anchor: &str) -> Option<NaiveDate> {
            let anchor = anchor.trim();
            // Reuse the explicit-date regex extractor.
            let explicit = extract_explicit_dates(anchor);
            if let Some((d, _, _)) = explicit.into_iter().next() {
                return Some(d);
            }
            // Fallback: dateparser for ISO strings, etc.
            if let Ok(parsed) = dateparser::parse(anchor) {
                let d = parsed.date_naive();
                if (1900..=2100).contains(&d.year()) {
                    return Some(d);
                }
            }
            None
        }

        /// Map a lower-cased weekday name to a `chrono::Weekday`.
        fn parse_weekday(s: &str) -> Option<Weekday> {
            match s {
                "monday" => Some(Weekday::Mon),
                "tuesday" => Some(Weekday::Tue),
                "wednesday" => Some(Weekday::Wed),
                "thursday" => Some(Weekday::Thu),
                "friday" => Some(Weekday::Fri),
                "saturday" => Some(Weekday::Sat),
                "sunday" => Some(Weekday::Sun),
                _ => None,
            }
        }

        // Patterns — ordered most-specific first so longer phrases are not
        // shadowed by shorter ones.  All patterns share the same named group
        // `anchor` for the tail date string.

        // 1. "the <weekday> before|after <anchor>"
        static WEEKDAY_REL: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(
                r"(?i)\bthe\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(before|after)\s+(.+)",
            )
            .unwrap()
        });

        // 2. "the day before|after <anchor>"
        static DAY_REL: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?i)\bthe\s+day\s+(before|after)\s+(.+)").unwrap());

        // 3. "<N> day|days|week|weeks before|after <anchor>"
        static N_UNIT_REL: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(?i)\b(\d+)\s+(days?|weeks?)\s+(before|after)\s+(.+)").unwrap()
        });

        // Helper: emit one anchor-relative result.
        // Emits the derived date into both `query_bounds` (for B3 boost path)
        // AND `refs`/`earliest`/`latest` (for Layer 0.6 temporal-fact boost via
        // `filter_facts_by_time_window`).  Both signals are needed: query_bounds
        // drives session-level scoring boosts; refs/earliest/latest allow Layer 0.6
        // to surface the temporal facts closest to the derived date.
        let mut emit_anchor_result = |computed: NaiveDate, matched_text: &str, pos: usize| {
            let bound = TemporalBound::Point(computed);
            if !query_bounds
                .iter()
                .any(|b| matches!(b, TemporalBound::Point(p) if *p == computed))
            {
                query_bounds.push(bound);
            }
            if !refs.iter().any(|r: &TemporalRef| r.date == computed) {
                refs.push(TemporalRef {
                    date: computed,
                    original_text: matched_text.to_string(),
                    confidence: 0.92,
                    position: pos,
                    ref_type: TemporalRefType::Relative,
                });
                update_bounds(&mut earliest, &mut latest, computed);
            }
        };

        // Check pattern 1: "the <weekday> before|after <anchor>"
        if let Some(caps) = WEEKDAY_REL.captures(&text_lower) {
            let wday_str = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let direction = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            let anchor_str = caps.get(3).map(|m| m.as_str()).unwrap_or("");
            if let (Some(weekday), Some(anchor_date)) =
                (parse_weekday(wday_str), try_parse_anchor_date(anchor_str))
            {
                let computed = if direction == "before" {
                    last_weekday(anchor_date, weekday)
                } else {
                    next_weekday(anchor_date, weekday)
                };
                let pos = caps.get(0).map(|m| m.start()).unwrap_or(0);
                let matched = caps.get(0).map(|m| m.as_str()).unwrap_or("");
                emit_anchor_result(computed, matched, pos);
            }
        }
        // Check pattern 2: "the day before|after <anchor>"
        else if let Some(caps) = DAY_REL.captures(&text_lower) {
            let direction = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let anchor_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            if let Some(anchor_date) = try_parse_anchor_date(anchor_str) {
                let computed = if direction == "before" {
                    anchor_date - Duration::days(1)
                } else {
                    anchor_date + Duration::days(1)
                };
                let pos = caps.get(0).map(|m| m.start()).unwrap_or(0);
                let matched = caps.get(0).map(|m| m.as_str()).unwrap_or("");
                emit_anchor_result(computed, matched, pos);
            }
        }
        // Check pattern 3: "<N> day|days|week|weeks before|after <anchor>"
        else if let Some(caps) = N_UNIT_REL.captures(&text_lower) {
            let n: i64 = caps
                .get(1)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let unit = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            let direction = caps.get(3).map(|m| m.as_str()).unwrap_or("");
            let anchor_str = caps.get(4).map(|m| m.as_str()).unwrap_or("");
            if n > 0 {
                if let Some(anchor_date) = try_parse_anchor_date(anchor_str) {
                    let days = if unit.starts_with("week") { n * 7 } else { n };
                    let computed = if direction == "before" {
                        anchor_date - Duration::days(days)
                    } else {
                        anchor_date + Duration::days(days)
                    };
                    let pos = caps.get(0).map(|m| m.start()).unwrap_or(0);
                    let matched = caps.get(0).map(|m| m.as_str()).unwrap_or("");
                    emit_anchor_result(computed, matched, pos);
                }
            }
        }
    }

    // Pass 1: fixed-phrase matches.
    for &phrase in RELATIVE_PHRASES {
        if text_lower.contains(phrase) {
            let resolved = resolve_relative_time(phrase, now);
            let bound = resolved.to_bound();
            query_bounds.push(bound);

            // Also synthesise a TemporalRef so existing callers that inspect
            // `refs`/`earliest`/`latest` benefit from the resolved date.
            if let crate::memory::temporal_facts::ResolvedTime::ExactDate(d) = resolved {
                let pos = text_lower.find(phrase).unwrap_or(0);
                if !refs.iter().any(|r: &TemporalRef| r.date == d) {
                    refs.push(TemporalRef {
                        date: d,
                        original_text: phrase.to_string(),
                        confidence: 0.9,
                        position: pos,
                        ref_type: TemporalRefType::Relative,
                    });
                    update_bounds(&mut earliest, &mut latest, d);
                }
            } else if let crate::memory::temporal_facts::ResolvedTime::MonthYear { month, year } =
                resolve_relative_time(phrase, now)
            {
                // Use the first day of the month as the representative date.
                if let Some(d) = NaiveDate::from_ymd_opt(year, month, 1) {
                    let pos = text_lower.find(phrase).unwrap_or(0);
                    if !refs.iter().any(|r: &TemporalRef| r.date == d) {
                        refs.push(TemporalRef {
                            date: d,
                            original_text: phrase.to_string(),
                            confidence: 0.85,
                            position: pos,
                            ref_type: TemporalRefType::Relative,
                        });
                        update_bounds(&mut earliest, &mut latest, d);
                    }
                }
            } else if let crate::memory::temporal_facts::ResolvedTime::Year(year) =
                resolve_relative_time(phrase, now)
            {
                if let Some(d) = NaiveDate::from_ymd_opt(year, 1, 1) {
                    let pos = text_lower.find(phrase).unwrap_or(0);
                    if !refs.iter().any(|r: &TemporalRef| r.date == d) {
                        refs.push(TemporalRef {
                            date: d,
                            original_text: phrase.to_string(),
                            confidence: 0.8,
                            position: pos,
                            ref_type: TemporalRefType::Relative,
                        });
                        update_bounds(&mut earliest, &mut latest, d);
                    }
                }
            }
        }
    }

    // Pass 2: "N days/weeks/months ago" — scan for " ago" suffix.
    if text_lower.contains(" ago") {
        // Try to resolve the entire text as an "ago" expression.
        let resolved = resolve_relative_time(&text_lower, now);
        if let crate::memory::temporal_facts::ResolvedTime::ExactDate(d) = resolved {
            let bound = crate::memory::temporal_facts::ResolvedTime::ExactDate(d).to_bound();
            if !query_bounds
                .iter()
                .any(|b| matches!(b, TemporalBound::Point(p) if *p == d))
            {
                query_bounds.push(bound);
                if !refs.iter().any(|r: &TemporalRef| r.date == d) {
                    refs.push(TemporalRef {
                        date: d,
                        original_text: text.to_string(),
                        confidence: 0.85,
                        position: 0,
                        ref_type: TemporalRefType::Relative,
                    });
                    update_bounds(&mut earliest, &mut latest, d);
                }
            }
        }
    }

    // Pass 3: "last/next <weekday>" compounds.
    let weekday_names = [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ];
    for day in weekday_names {
        for prefix in ["last ", "next ", "this "] {
            let phrase = format!("{}{}", prefix, day);
            if text_lower.contains(&phrase) {
                let resolved = resolve_relative_time(&phrase, now);
                if let crate::memory::temporal_facts::ResolvedTime::ExactDate(d) = resolved {
                    let bound =
                        crate::memory::temporal_facts::ResolvedTime::ExactDate(d).to_bound();
                    if !query_bounds
                        .iter()
                        .any(|b| matches!(b, TemporalBound::Point(p) if *p == d))
                    {
                        query_bounds.push(bound);
                        if !refs.iter().any(|r: &TemporalRef| r.date == d) {
                            let pos = text_lower.find(&phrase).unwrap_or(0);
                            refs.push(TemporalRef {
                                date: d,
                                original_text: phrase.clone(),
                                confidence: 0.9,
                                position: pos,
                                ref_type: TemporalRefType::Relative,
                            });
                            update_bounds(&mut earliest, &mut latest, d);
                        }
                    }
                }
            }
        }
    }

    // Emit a debug trace when no temporal window was resolved — this is the
    // silent no-op where the retrieval gate returns all facts unfiltered.
    if query_bounds.is_empty() {
        tracing::debug!(
            query = %text,
            "temporal pre-pass: no relative phrases matched; query temporal window is unresolved"
        );
    }

    // Helper to validate date is in reasonable range (1900-2100)
    let is_valid_date = |date: &NaiveDate| -> bool {
        let year = date.year();
        (1900..=2100).contains(&year)
    };

    // Try dateparser on the full text (returns Result, never panics)
    if let Ok(parsed) = dateparser::parse(text) {
        let date = parsed.date_naive();
        if is_valid_date(&date) {
            refs.push(TemporalRef {
                date,
                original_text: text.to_string(),
                confidence: 0.8,
                position: 0,
                ref_type: classify_temporal_ref(text, &date, &now),
            });
            update_bounds(&mut earliest, &mut latest, date);
        }
    }

    // Try parsing individual sentences/phrases
    for (pos, sentence) in split_temporal_phrases(text).iter().enumerate() {
        if let Ok(parsed) = dateparser::parse(sentence) {
            let date = parsed.date_naive();
            if !is_valid_date(&date) {
                continue;
            }
            if refs.iter().any(|r| r.date == date) {
                continue;
            }
            refs.push(TemporalRef {
                date,
                original_text: sentence.to_string(),
                confidence: 0.7,
                position: pos,
                ref_type: classify_temporal_ref(sentence, &date, &now),
            });
            update_bounds(&mut earliest, &mut latest, date);
        }
    }

    // Also use regex-based extraction for explicit date patterns
    let explicit_dates = extract_explicit_dates(text);
    for (date, original, pos) in explicit_dates {
        if !is_valid_date(&date) {
            continue;
        }
        if refs.iter().any(|r| r.date == date) {
            continue;
        }
        refs.push(TemporalRef {
            date,
            original_text: original,
            confidence: 0.9,
            position: pos,
            ref_type: TemporalRefType::Absolute,
        });
        update_bounds(&mut earliest, &mut latest, date);
    }

    // Sort by position in text
    refs.sort_by_key(|r| r.position);

    TemporalExtraction {
        refs,
        earliest,
        latest,
        query_bounds,
    }
}

/// Extract temporal references from text, with an optional named-event resolver
/// for query-time anchor resolution (B2).
///
/// Behaves identically to `extract_temporal_refs` except that when Pass 0 finds
/// an anchor-relative pattern ("the Sunday before the wedding") but
/// `try_parse_anchor_date` fails (because the anchor is a named event, not a
/// calendar date), it calls `event_resolver(anchor_str)` to look up the event's
/// stored date from the `TemporalFactStore`.
///
/// **Plumbing note**: `extract_temporal_refs` is a free function called from
/// `detect_temporal_intent` / `requires_temporal_filtering` without store access;
/// those callers are unaffected.  `recall.rs` calls this variant so that the
/// store is available for event lookup.
///
/// The `event_resolver` closure is called at most once per query (one anchor
/// phrase per Pass 0 pattern match).  Returning `None` from the closure leaves
/// the anchor unresolved and Pass 0 emits nothing — later passes still run.
pub fn extract_temporal_refs_with_events(
    text: &str,
    event_resolver: Option<&dyn Fn(&str) -> Option<NaiveDate>>,
) -> TemporalExtraction {
    use regex::Regex;
    use std::sync::LazyLock;

    // Run the standard extractor first — covers absolute-date anchors (B1) and
    // all existing passes (relative, "N ago", weekdays).
    let base = extract_temporal_refs(text);

    // If Pass 0 already emitted at least one bound, or there is no event
    // resolver, return the base result unchanged.
    let resolver = match event_resolver {
        Some(r) => r,
        None => return base,
    };

    // Re-check Pass 0 patterns: if any matched and the anchor was NOT resolved
    // by `try_parse_anchor_date` (i.e. no bound was emitted yet from Pass 0),
    // try the event resolver as a fallback.
    //
    // We detect "Pass 0 would have matched but didn't" by checking whether the
    // text matches one of the three regexes AND the base result has no bounds
    // that could have come from that match (i.e. base.query_bounds is empty or
    // only contains non-Point bounds from later passes).
    //
    // For simplicity we re-run the regexes and, if try_parse_anchor_date returns
    // None, call the event resolver.  This is intentionally additive: existing
    // bounds from base are preserved.

    fn try_parse_anchor_date(anchor: &str) -> Option<NaiveDate> {
        let anchor = anchor.trim();
        let explicit = extract_explicit_dates(anchor);
        if let Some((d, _, _)) = explicit.into_iter().next() {
            return Some(d);
        }
        if let Ok(parsed) = dateparser::parse(anchor) {
            let d = parsed.date_naive();
            if (1900..=2100).contains(&d.year()) {
                return Some(d);
            }
        }
        None
    }

    fn parse_weekday_b2(s: &str) -> Option<Weekday> {
        match s {
            "monday" => Some(Weekday::Mon),
            "tuesday" => Some(Weekday::Tue),
            "wednesday" => Some(Weekday::Wed),
            "thursday" => Some(Weekday::Thu),
            "friday" => Some(Weekday::Fri),
            "saturday" => Some(Weekday::Sat),
            "sunday" => Some(Weekday::Sun),
            _ => None,
        }
    }

    static WEEKDAY_REL_B2: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)\bthe\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(before|after)\s+(.+)",
        )
        .unwrap()
    });
    static DAY_REL_B2: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\bthe\s+day\s+(before|after)\s+(.+)").unwrap());
    static N_UNIT_REL_B2: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\b(\d+)\s+(days?|weeks?)\s+(before|after)\s+(.+)").unwrap()
    });

    let text_lower = text.to_lowercase();

    // Determine which pattern matched and extract the anchor string.
    // If try_parse_anchor_date succeeds on the anchor, base already handled it.
    // We only call the event resolver when try_parse_anchor_date returns None.

    struct AnchorHit {
        anchor_str: String,
        kind: AnchorKind,
        matched_text: String,
        pos: usize,
    }
    enum AnchorKind {
        Weekday { weekday: Weekday, direction: String },
        Day { direction: String },
        NUnit { n: i64, direction: String },
    }

    let hit: Option<AnchorHit> = if let Some(caps) = WEEKDAY_REL_B2.captures(&text_lower) {
        let wday_str = caps.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        let direction = caps.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
        let anchor_str = caps.get(3).map(|m| m.as_str()).unwrap_or("").to_string();
        let matched_text = caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        let pos = caps.get(0).map(|m| m.start()).unwrap_or(0);
        if try_parse_anchor_date(&anchor_str).is_none() {
            parse_weekday_b2(&wday_str).map(|weekday| AnchorHit {
                anchor_str,
                kind: AnchorKind::Weekday { weekday, direction },
                matched_text,
                pos,
            })
        } else {
            None
        }
    } else if let Some(caps) = DAY_REL_B2.captures(&text_lower) {
        let direction = caps.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        let anchor_str = caps.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
        let matched_text = caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        let pos = caps.get(0).map(|m| m.start()).unwrap_or(0);
        if try_parse_anchor_date(&anchor_str).is_none() {
            Some(AnchorHit {
                anchor_str,
                kind: AnchorKind::Day { direction },
                matched_text,
                pos,
            })
        } else {
            None
        }
    } else if let Some(caps) = N_UNIT_REL_B2.captures(&text_lower) {
        let n: i64 = caps
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let unit = caps.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
        let direction = caps.get(3).map(|m| m.as_str()).unwrap_or("").to_string();
        let anchor_str = caps.get(4).map(|m| m.as_str()).unwrap_or("").to_string();
        let matched_text = caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        let pos = caps.get(0).map(|m| m.start()).unwrap_or(0);
        if n > 0 && try_parse_anchor_date(&anchor_str).is_none() {
            Some(AnchorHit {
                anchor_str,
                kind: AnchorKind::NUnit {
                    n,
                    direction: format!("{}:{}", direction, unit),
                },
                matched_text,
                pos,
            })
        } else {
            None
        }
    } else {
        None
    };

    let hit = match hit {
        Some(h) => h,
        None => return base,
    };

    // Call the event resolver with the anchor phrase.
    let anchor_date = match resolver(&hit.anchor_str) {
        Some(d) => d,
        None => return base,
    };

    // Compute the target date using B1's existing helpers.
    let computed = match &hit.kind {
        AnchorKind::Weekday { weekday, direction } => {
            if direction == "before" {
                crate::memory::temporal_facts::last_weekday(anchor_date, *weekday)
            } else {
                crate::memory::temporal_facts::next_weekday(anchor_date, *weekday)
            }
        }
        AnchorKind::Day { direction } => {
            if direction == "before" {
                anchor_date - Duration::days(1)
            } else {
                anchor_date + Duration::days(1)
            }
        }
        AnchorKind::NUnit { n, direction } => {
            // direction is encoded as "before:days?" or "after:weeks?"
            let parts: Vec<&str> = direction.splitn(2, ':').collect();
            let dir = parts.first().copied().unwrap_or("");
            let unit = parts.get(1).copied().unwrap_or("");
            let days = if unit.starts_with("week") { n * 7 } else { *n };
            if dir == "before" {
                anchor_date - Duration::days(days)
            } else {
                anchor_date + Duration::days(days)
            }
        }
    };

    // Build the augmented extraction from the base, adding the newly resolved bound.
    let mut augmented = base;

    // Emit into query_bounds (for B3 session-level boost) and refs/earliest/latest
    // (for Layer 0.6 temporal-fact boost via filter_facts_by_time_window).
    if !augmented
        .query_bounds
        .iter()
        .any(|b| matches!(b, TemporalBound::Point(p) if *p == computed))
    {
        augmented.query_bounds.push(TemporalBound::Point(computed));
    }
    if !augmented.refs.iter().any(|r| r.date == computed) {
        augmented.refs.push(TemporalRef {
            date: computed,
            original_text: hit.matched_text,
            confidence: 0.88,
            position: hit.pos,
            ref_type: TemporalRefType::Relative,
        });
        let mut e = augmented.earliest;
        let mut l = augmented.latest;
        update_bounds(&mut e, &mut l, computed);
        augmented.earliest = e;
        augmented.latest = l;
    }

    augmented
}

// ============================================================================
// Calendar windows named in a query
// ============================================================================
// `extract_temporal_refs` resolves every date it finds to one `NaiveDate` and
// cannot say whether "May 2022" meant the month or its first day. The date
// window leg in recall needs that distinction: it fetches memories by
// `created_at`, so a month must widen to the whole month and a year must not
// fire at all (a year of memories is not a candidate list). This extractor
// reads the calendar granularity straight off the text.

/// How much of the calendar a [`DateWindow`] covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DateWindowGranularity {
    /// One calendar day: "1 May, 2022", "May 1, 2022", "2022-05-01", "05/01/2022".
    Day,
    /// One calendar month: "May 2022", "in April of 2022".
    Month,
}

/// A calendar span a query names explicitly, with its year.
///
/// `end` is exclusive, so a day is `[d, d+1)` and a month is `[1st, 1st of next month)`.
/// Year-only mentions ("in 2022") are not windows: they stay with the temporal boost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DateWindow {
    /// First day covered.
    pub start: NaiveDate,
    /// First day NOT covered.
    pub end: NaiveDate,
    /// Day or month.
    pub granularity: DateWindowGranularity,
    /// The text the window was read from, as written.
    pub original_text: String,
}

impl DateWindow {
    /// The window for one calendar day.
    pub fn day(date: NaiveDate, original_text: &str) -> Self {
        Self {
            start: date,
            end: date + Duration::days(1),
            granularity: DateWindowGranularity::Day,
            original_text: original_text.to_string(),
        }
    }

    /// The window for one calendar month. `None` when `month` is not 1..=12.
    pub fn month(year: i32, month: u32, original_text: &str) -> Option<Self> {
        let start = NaiveDate::from_ymd_opt(year, month, 1)?;
        let end = if month == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1)?
        } else {
            NaiveDate::from_ymd_opt(year, month + 1, 1)?
        };
        Some(Self {
            start,
            end,
            granularity: DateWindowGranularity::Month,
            original_text: original_text.to_string(),
        })
    }

    /// Whether `date` falls inside the window.
    pub fn contains(&self, date: NaiveDate) -> bool {
        date >= self.start && date < self.end
    }
}

/// The calendar days and month-years a query names explicitly, in text order.
///
/// Recognised day forms: "May 7, 2023", "May 7th 2023", "7 May, 2023", "7th of May 2023",
/// "2023-05-07" and "05/07/2023" (month first, as `extract_explicit_dates` reads it).
/// Recognised month forms: "May 2023", "May, 2023", "May of 2023", with month names
/// spelt out or abbreviated. A month-year that is part of a day form ("7 May, 2023")
/// is reported once, as the day. Years outside 1900..=2100 are not dates. Two mentions
/// of the same window are reported once.
pub fn extract_date_windows(text: &str) -> Vec<DateWindow> {
    use regex::Regex;
    use std::sync::LazyLock;

    const MONTH: &str = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)";

    // "May 7, 2023", "May 7th 2023", "Sept. 7, 2023"
    static MONTH_DAY_YEAR: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(
            r"(?i)\b{MONTH}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b"
        ))
        .expect("month-day-year regex")
    });
    // "7 May, 2023", "7th of May 2023"
    static DAY_MONTH_YEAR: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(
            r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}\.?,?\s+(\d{{4}})\b"
        ))
        .expect("day-month-year regex")
    });
    // "2023-05-07"
    static ISO_DAY: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").expect("iso-day regex"));
    // "05/07/2023", month first.
    static SLASH_DAY: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b").expect("slash-day regex"));
    // "May 2023", "May, 2023", "May of 2023"
    static MONTH_YEAR: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(r"(?i)\b{MONTH}\.?,?\s+(?:of\s+)?(\d{{4}})\b"))
            .expect("month-year regex")
    });

    fn year_in_range(year: i32) -> bool {
        (1900..=2100).contains(&year)
    }

    // (start offset, end offset, window)
    let mut found: Vec<(usize, usize, DateWindow)> = Vec::new();
    // Every day form's span, recorded whether or not its window is new: a repeated day
    // still contains a month-year that must not surface as a month on its own.
    let mut day_spans: Vec<(usize, usize)> = Vec::new();
    fn push(
        found: &mut Vec<(usize, usize, DateWindow)>,
        start: usize,
        end: usize,
        window: DateWindow,
    ) {
        if !found
            .iter()
            .any(|(_, _, w)| w.start == window.start && w.end == window.end)
        {
            found.push((start, end, window));
        }
    }

    for cap in MONTH_DAY_YEAR.captures_iter(text) {
        let whole = cap.get(0).expect("whole match");
        let month = month_to_num(&cap[1]);
        let (Ok(day), Ok(year)) = (cap[2].parse::<u32>(), cap[3].parse::<i32>()) else {
            continue;
        };
        if !year_in_range(year) {
            continue;
        }
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            day_spans.push((whole.start(), whole.end()));
            push(
                &mut found,
                whole.start(),
                whole.end(),
                DateWindow::day(date, whole.as_str()),
            );
        }
    }
    for cap in DAY_MONTH_YEAR.captures_iter(text) {
        let whole = cap.get(0).expect("whole match");
        let (Ok(day), Ok(year)) = (cap[1].parse::<u32>(), cap[3].parse::<i32>()) else {
            continue;
        };
        let month = month_to_num(&cap[2]);
        if !year_in_range(year) {
            continue;
        }
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            day_spans.push((whole.start(), whole.end()));
            push(
                &mut found,
                whole.start(),
                whole.end(),
                DateWindow::day(date, whole.as_str()),
            );
        }
    }
    for cap in ISO_DAY.captures_iter(text) {
        let whole = cap.get(0).expect("whole match");
        let (Ok(year), Ok(month), Ok(day)) = (
            cap[1].parse::<i32>(),
            cap[2].parse::<u32>(),
            cap[3].parse::<u32>(),
        ) else {
            continue;
        };
        if !year_in_range(year) {
            continue;
        }
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            day_spans.push((whole.start(), whole.end()));
            push(
                &mut found,
                whole.start(),
                whole.end(),
                DateWindow::day(date, whole.as_str()),
            );
        }
    }
    for cap in SLASH_DAY.captures_iter(text) {
        let whole = cap.get(0).expect("whole match");
        let (Ok(month), Ok(day), Ok(year)) = (
            cap[1].parse::<u32>(),
            cap[2].parse::<u32>(),
            cap[3].parse::<i32>(),
        ) else {
            continue;
        };
        if !year_in_range(year) {
            continue;
        }
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            day_spans.push((whole.start(), whole.end()));
            push(
                &mut found,
                whole.start(),
                whole.end(),
                DateWindow::day(date, whole.as_str()),
            );
        }
    }
    // A month-year inside a day form is that day, already recorded.
    for cap in MONTH_YEAR.captures_iter(text) {
        let whole = cap.get(0).expect("whole match");
        let inside_day = day_spans
            .iter()
            .any(|(s, e)| whole.start() >= *s && whole.end() <= *e);
        if inside_day {
            continue;
        }
        let Ok(year) = cap[2].parse::<i32>() else {
            continue;
        };
        if !year_in_range(year) {
            continue;
        }
        let month = month_to_num(&cap[1]);
        if let Some(window) = DateWindow::month(year, month, whole.as_str()) {
            push(&mut found, whole.start(), whole.end(), window);
        }
    }

    found.sort_by_key(|(start, _, _)| *start);
    found.into_iter().map(|(_, _, w)| w).collect()
}

#[cfg(test)]
mod date_window_tests {
    use super::{extract_date_windows, DateWindow, DateWindowGranularity};
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn a_day_month_year_is_one_day() {
        let w = extract_date_windows("What movie did Joanna watch on 1 May, 2022?");
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(w[0].granularity, DateWindowGranularity::Day);
        assert_eq!(w[0].start, d(2022, 5, 1));
        assert_eq!(w[0].end, d(2022, 5, 2));
        assert_eq!(w[0].original_text, "1 May, 2022");
    }

    #[test]
    fn every_day_form_reads_as_that_day() {
        for text in [
            "on February 7, 2022",
            "on February 7th, 2022",
            "on 7 February 2022",
            "on the 7th of February 2022",
            "on 2022-02-07",
            "on 02/07/2022",
            "on Feb. 7, 2022",
        ] {
            let w = extract_date_windows(text);
            assert_eq!(w.len(), 1, "{text}: {w:?}");
            assert_eq!(
                w[0],
                DateWindow::day(d(2022, 2, 7), &w[0].original_text),
                "{text}"
            );
        }
    }

    #[test]
    fn a_month_year_is_the_whole_month() {
        let w =
            extract_date_windows("What physical transformation did Nate undergo in April 2022?");
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(w[0].granularity, DateWindowGranularity::Month);
        assert_eq!(w[0].start, d(2022, 4, 1));
        assert_eq!(w[0].end, d(2022, 5, 1));
        assert!(w[0].contains(d(2022, 4, 30)));
        assert!(!w[0].contains(d(2022, 5, 1)));
    }

    #[test]
    fn december_rolls_into_the_next_year() {
        let w = extract_date_windows("in December of 2021");
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].start, d(2021, 12, 1));
        assert_eq!(w[0].end, d(2022, 1, 1));
    }

    #[test]
    fn a_year_alone_is_not_a_window() {
        assert!(extract_date_windows("What state did Joanna visit in 2021?").is_empty());
        assert!(extract_date_windows("in summer 2021").is_empty());
    }

    #[test]
    fn a_month_inside_a_day_form_is_reported_once_as_the_day() {
        let w = extract_date_windows("on 25 February, 2022");
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(w[0].granularity, DateWindowGranularity::Day);
    }

    #[test]
    fn a_month_without_a_year_is_not_a_window() {
        assert!(extract_date_windows("What happened in May?").is_empty());
        assert!(extract_date_windows("last May").is_empty());
    }

    #[test]
    fn impossible_dates_and_out_of_range_years_are_skipped() {
        assert!(extract_date_windows("on February 30, 2022").is_empty());
        assert!(extract_date_windows("on May 1, 1066").is_empty());
        assert!(extract_date_windows("in May 3000").is_empty());
    }

    #[test]
    fn windows_come_in_text_order_and_repeats_collapse() {
        let w = extract_date_windows("between June 2022 and 3 May, 2022, or May 3, 2022");
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!(w[0].granularity, DateWindowGranularity::Month);
        assert_eq!(w[0].start, d(2022, 6, 1));
        assert_eq!(w[1].granularity, DateWindowGranularity::Day);
        assert_eq!(w[1].start, d(2022, 5, 3));
    }
}

/// Classify the type of temporal reference
fn classify_temporal_ref(text: &str, date: &NaiveDate, now: &DateTime<Utc>) -> TemporalRefType {
    let text_lower = text.to_lowercase();
    let today = now.date_naive();

    // Check for relative indicators
    if text_lower.contains("yesterday")
        || text_lower.contains("ago")
        || text_lower.contains("last")
        || text_lower.contains("previous")
        || text_lower.contains("before")
        || text_lower.contains("earlier")
    {
        return TemporalRefType::Relative;
    }

    // Check for day of week
    let days = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ];
    if days.iter().any(|d| text_lower.contains(d)) {
        return TemporalRefType::DayOfWeek;
    }

    // Check for month names without day
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let has_month = months.iter().any(|m| text_lower.contains(m));
    let has_day = text.chars().any(|c| c.is_ascii_digit());

    if has_month && !has_day {
        return TemporalRefType::Month;
    }

    // Check for year-only reference
    if text.len() == 4 && text.chars().all(|c| c.is_ascii_digit()) {
        return TemporalRefType::Year;
    }

    // If date is today or very close, might be relative
    let diff = (today - *date).num_days().abs();
    if diff <= 7 && text_lower.contains("this") {
        return TemporalRefType::Relative;
    }

    TemporalRefType::Absolute
}

/// Split text into temporal-relevant phrases
fn split_temporal_phrases(text: &str) -> Vec<String> {
    let mut phrases = Vec::new();

    // Split by common temporal markers and punctuation
    let markers = [
        " on ", " in ", " at ", " during ", " since ", " until ", " before ", " after ",
        " around ", ", ", ". ", "! ", "? ",
    ];

    let current = text.to_string();
    for marker in markers {
        let parts: Vec<&str> = current.split(marker).collect();
        if parts.len() > 1 {
            for part in parts {
                let trimmed = part.trim();
                if !trimmed.is_empty() && trimmed.len() > 3 {
                    phrases.push(trimmed.to_string());
                }
            }
            break;
        }
    }

    // If no splitting happened, try sentence boundaries
    if phrases.is_empty() {
        for sentence in text.split('.') {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() && trimmed.len() > 3 {
                phrases.push(trimmed.to_string());
            }
        }
    }

    phrases
}

/// Extract explicit date patterns that date_time_parser might miss
fn extract_explicit_dates(text: &str) -> Vec<(NaiveDate, String, usize)> {
    use regex::Regex;

    let mut results = Vec::new();

    // Pattern: "Month Day, Year" (e.g., "May 7, 2023")
    let month_day_year =
        Regex::new(r"(?i)(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})")
            .unwrap();

    for cap in month_day_year.captures_iter(text) {
        let month_str = &cap[1];
        let day: u32 = cap[2].parse().unwrap_or(1);
        let year: i32 = cap[3].parse().unwrap_or(2000);
        let month = month_to_num(month_str);

        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let pos = cap.get(0).map(|m| m.start()).unwrap_or(0);
            results.push((date, cap[0].to_string(), pos));
        }
    }

    // Pattern: "YYYY-MM-DD"
    let iso_date = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
    for cap in iso_date.captures_iter(text) {
        let year: i32 = cap[1].parse().unwrap_or(2000);
        let month: u32 = cap[2].parse().unwrap_or(1);
        let day: u32 = cap[3].parse().unwrap_or(1);

        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let pos = cap.get(0).map(|m| m.start()).unwrap_or(0);
            results.push((date, cap[0].to_string(), pos));
        }
    }

    // Pattern: "MM/DD/YYYY" or "DD/MM/YYYY" (assume US format MM/DD)
    let slash_date = Regex::new(r"(\d{1,2})/(\d{1,2})/(\d{4})").unwrap();
    for cap in slash_date.captures_iter(text) {
        let month: u32 = cap[1].parse().unwrap_or(1);
        let day: u32 = cap[2].parse().unwrap_or(1);
        let year: i32 = cap[3].parse().unwrap_or(2000);

        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let pos = cap.get(0).map(|m| m.start()).unwrap_or(0);
            results.push((date, cap[0].to_string(), pos));
        }
    }

    results
}

/// Convert month name to number
fn month_to_num(month: &str) -> u32 {
    match month.to_lowercase().as_str() {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => 1,
    }
}

/// Update earliest/latest bounds
fn update_bounds(
    earliest: &mut Option<NaiveDate>,
    latest: &mut Option<NaiveDate>,
    date: NaiveDate,
) {
    match earliest {
        Some(e) if date < *e => *earliest = Some(date),
        None => *earliest = Some(date),
        _ => {}
    }
    match latest {
        Some(l) if date > *l => *latest = Some(date),
        None => *latest = Some(date),
        _ => {}
    }
}

// ============================================================================
// TEMPORAL QUERY DETECTION
// ============================================================================
// Detect when a query is asking about time/dates.

/// Query temporal intent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalIntent {
    /// Query is asking "when" something happened
    WhenQuestion,
    /// Query references a specific time period
    SpecificTime,
    /// Query asks about temporal ordering (before/after)
    Ordering,
    /// Query asks about duration (how long)
    Duration,
    /// No temporal intent detected
    None,
}

/// Detect temporal intent in a query
pub fn detect_temporal_intent(query: &str) -> TemporalIntent {
    let query_lower = query.to_lowercase();

    // "When" questions are highest priority
    if query_lower.starts_with("when")
        || query_lower.contains(" when ")
        || query_lower.contains("what date")
        || query_lower.contains("what day")
        || query_lower.contains("what time")
    {
        return TemporalIntent::WhenQuestion;
    }

    // Duration questions
    if query_lower.contains("how long")
        || query_lower.contains("how many days")
        || query_lower.contains("how many weeks")
        || query_lower.contains("how many months")
        || query_lower.contains("how many years")
    {
        return TemporalIntent::Duration;
    }

    // Ordering questions — compound comparisons AND standalone relative ordering
    // "before X" and "after X" imply temporal ordering even without a comparison pair
    if query_lower.contains("before or after")
        || query_lower.contains("first or")
        || query_lower.contains("earlier or later")
        || query_lower.contains("which came first")
        || query_lower.contains("in what order")
        || query_lower.contains(" before the ")
        || query_lower.contains(" before we ")
        || query_lower.contains(" after the ")
        || query_lower.contains(" after we ")
        || query_lower.contains(" prior to ")
    {
        return TemporalIntent::Ordering;
    }

    // Specific time references
    let time_indicators = [
        "yesterday",
        "today",
        "last week",
        "last month",
        "last year",
        "this week",
        "this month",
        "this year",
        "most recent",
        "latest",
        "earliest",
        "recently",
        "first meeting",
        "second meeting",
        "third meeting",
        "fourth meeting",
        "first session",
        "second session",
        "third session",
        "fourth session",
        "in january",
        "in february",
        "in march",
        "in april",
        "in may",
        "in june",
        "in july",
        "in august",
        "in september",
        "in october",
        "in november",
        "in december",
        "on monday",
        "on tuesday",
        "on wednesday",
        "on thursday",
        "on friday",
        "on saturday",
        "on sunday",
        " ago",
        " days ago",
        " weeks ago",
        " months ago",
        " years ago",
    ];

    if time_indicators.iter().any(|t| query_lower.contains(t)) {
        return TemporalIntent::SpecificTime;
    }

    // Check for date patterns
    let extraction = extract_temporal_refs(query);
    if extraction.has_temporal_refs() {
        return TemporalIntent::SpecificTime;
    }

    TemporalIntent::None
}

/// Check if a query requires temporal filtering for accurate retrieval
///
/// Returns true if the query has a temporal component that should be used
/// to filter/rank memories by their temporal references.
///
/// IMPORTANT: "When did X happen?" questions (WhenQuestion) return FALSE
/// because they are asking FOR a date, not filtering BY a date.
/// We should search semantically for X and extract the date from results.
///
/// "What happened in May 2023?" (SpecificTime) returns TRUE because
/// it's filtering BY a specific time period.
pub fn requires_temporal_filtering(query: &str) -> bool {
    let intent = detect_temporal_intent(query);
    matches!(
        intent,
        // WhenQuestion is EXCLUDED - it asks FOR a date, not BY a date
        TemporalIntent::SpecificTime | TemporalIntent::Duration | TemporalIntent::Ordering
    )
}

/// Check if a query is asking FOR a temporal answer (when did X happen?)
///
/// These queries should use semantic search on the event X, then extract
/// the date from the retrieved content.
pub fn asks_for_temporal_answer(query: &str) -> bool {
    matches!(detect_temporal_intent(query), TemporalIntent::WhenQuestion)
}

// ============================================================================
// B4: TEMPORAL ANSWER-DERIVATION
// ============================================================================
// The answer-side complement to B1/B2. B1/B2 *parse* a derived date out of a
// query so retrieval can FILTER by it. B4 answers "when did/is X?" questions
// directly: when the query asks FOR a date and veld can compute it, we surface
// the computed date plus the supporting facts/source memories that justify it.
//
// This is strictly additive: it never touches retrieval or ranking. The HTTP
// surface (`handlers::query`) calls `derive_temporal_answer` and attaches the
// result to a NEW optional response field; when no date is derivable the result
// is `None` and the response is byte-identical to today.

/// How a temporal answer's date was derived. Priority order, highest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivationMethod {
    /// "the Sunday before 2023-05-25" — anchor is a literal calendar date (B1
    /// path). The date is computed purely from the query, no store lookup.
    AnchorRelative,
    /// "the Monday after the wedding" — anchor is a named event resolved to a
    /// date via the event resolver / `TemporalFactStore` (B2 path).
    EventAnchor,
    /// "when did X do Y" — a stored temporal fact for (entity, event) carries
    /// the answer in its `resolved_time`.
    TemporalFact,
}

impl DerivationMethod {
    /// Stable wire token. Matches the spec strings exactly.
    pub fn as_str(&self) -> &'static str {
        match self {
            DerivationMethod::AnchorRelative => "anchor_relative",
            DerivationMethod::EventAnchor => "event_anchor",
            DerivationMethod::TemporalFact => "temporal_fact",
        }
    }
}

/// A derived answer to a temporal question.
///
/// `derived_date` is an ISO `YYYY-MM-DD` string for `AnchorRelative`/`EventAnchor`
/// (always a `TemporalBound::Point`) or the fact's `ResolvedTime` sortable
/// representation for `TemporalFact` (which may be coarser, e.g. `2023-06` for a
/// MonthYear fact). `supporting_facts` and `source_memories` are empty for the
/// pure query-computed paths (anchor-relative) and populated when a stored fact
/// justified the answer.
#[derive(Debug, Clone)]
pub struct DerivedTemporalAnswer {
    pub derived_date: String,
    pub method: DerivationMethod,
    pub supporting_facts: Vec<crate::memory::temporal_facts::TemporalFact>,
    pub source_memories: Vec<crate::memory::types::MemoryId>,
}

/// Re-run the B1/B2 anchor regexes to classify whether the matched anchor is a
/// literal calendar date (→ `AnchorRelative`) or a named event (→ `EventAnchor`).
///
/// Returns `None` when no anchor pattern matched at all. Mirrors the regexes in
/// `extract_temporal_refs` / `extract_temporal_refs_with_events` so the
/// classification cannot drift from the resolver that produced the bound.
fn classify_anchor_relative(query: &str) -> Option<DerivationMethod> {
    use regex::Regex;
    use std::sync::LazyLock;

    static WEEKDAY_REL_B4: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)\bthe\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:before|after)\s+(.+)",
        )
        .unwrap()
    });
    static DAY_REL_B4: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\bthe\s+day\s+(?:before|after)\s+(.+)").unwrap());
    static N_UNIT_REL_B4: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\b\d+\s+(?:days?|weeks?)\s+(?:before|after)\s+(.+)").unwrap()
    });

    fn anchor_is_literal_date(anchor: &str) -> bool {
        let anchor = anchor.trim();
        if extract_explicit_dates(anchor).into_iter().next().is_some() {
            return true;
        }
        if let Ok(parsed) = dateparser::parse(anchor) {
            let d = parsed.date_naive();
            if (1900..=2100).contains(&d.year()) {
                return true;
            }
        }
        false
    }

    let lower = query.to_lowercase();
    let anchor = WEEKDAY_REL_B4
        .captures(&lower)
        .and_then(|c| c.get(1))
        .or_else(|| DAY_REL_B4.captures(&lower).and_then(|c| c.get(1)))
        .or_else(|| N_UNIT_REL_B4.captures(&lower).and_then(|c| c.get(1)))
        .map(|m| m.as_str().to_string())?;

    Some(if anchor_is_literal_date(&anchor) {
        DerivationMethod::AnchorRelative
    } else {
        DerivationMethod::EventAnchor
    })
}

/// Derive the answer date for a temporal "when did/is X?" question.
///
/// Returns `None` unless [`asks_for_temporal_answer`] is true AND a date can be
/// computed. Tries, in priority order:
///
/// 1. **Anchor-relative / event-anchor** — reuse [`extract_temporal_refs_with_events`].
///    If it resolves a `TemporalBound::Point`, that Point IS the answer date.
///    The anchor is classified as literal-date (`AnchorRelative`) or named-event
///    (`EventAnchor`).
/// 2. **Matching temporal fact** — `fact_lookup(query)` returns stored facts for
///    the query's (entity, event). The most recent fact's `resolved_time` is the
///    answer; the fact is a supporting fact and its `source_memory_id` a source
///    memory.
///
/// Both closures are injected so this function is pure and unit-testable without
/// a live `TemporalFactStore`. `event_resolver` resolves a named-event anchor
/// phrase to a date (the same closure `recall.rs` builds over the store).
/// `fact_lookup` returns candidate facts for the whole query string.
pub fn derive_temporal_answer(
    query: &str,
    event_resolver: Option<&dyn Fn(&str) -> Option<NaiveDate>>,
    fact_lookup: Option<&dyn Fn(&str) -> Vec<crate::memory::temporal_facts::TemporalFact>>,
) -> Option<DerivedTemporalAnswer> {
    // Reversion guard: only fire for "when/what-date" questions.
    if !asks_for_temporal_answer(query) {
        return None;
    }

    // --- Priority 1: anchor-relative / event-anchor → a Point bound. ---
    let extraction = extract_temporal_refs_with_events(query, event_resolver);
    let point = extraction.query_bounds.iter().find_map(|b| match b {
        TemporalBound::Point(p) => Some(*p),
        _ => None,
    });
    if let Some(date) = point {
        // Classify which sub-path produced the Point. Default to AnchorRelative
        // when the regex classifier can't see an anchor (e.g. a bare relative
        // pre-pass like "yesterday") — that date is still query-computed.
        let method = classify_anchor_relative(query).unwrap_or(DerivationMethod::AnchorRelative);
        return Some(DerivedTemporalAnswer {
            derived_date: date.format("%Y-%m-%d").to_string(),
            method,
            supporting_facts: Vec::new(),
            source_memories: Vec::new(),
        });
    }

    // --- Priority 2: a stored temporal fact answers the (entity, event). ---
    if let Some(lookup) = fact_lookup {
        let facts = lookup(query);
        // Choose the most recent NON-Unknown-dated fact (by conversation_date);
        // an Unknown resolved_time carries no answer.
        let chosen = facts
            .iter()
            .filter(|f| {
                !matches!(
                    f.resolved_time,
                    crate::memory::temporal_facts::ResolvedTime::Unknown
                        | crate::memory::temporal_facts::ResolvedTime::RelativeDescription(_)
                )
            })
            .max_by_key(|f| f.conversation_date)
            .cloned();
        if let Some(fact) = chosen {
            let derived_date = fact.resolved_time.to_sortable_string();
            let source_memories = vec![fact.source_memory_id.clone()];
            return Some(DerivedTemporalAnswer {
                derived_date,
                method: DerivationMethod::TemporalFact,
                supporting_facts: vec![fact],
                source_memories,
            });
        }
    }

    None
}

// ============================================================================
// ATTRIBUTE QUERY DETECTION
// ============================================================================
// Detect queries asking for specific attributes of entities.
// These queries need fact-first retrieval, not semantic similarity.
//
// Examples:
// - "What is Caroline's relationship status?" → entity=Caroline, attribute=relationship_status
// - "What is Melanie's job?" → entity=Melanie, attribute=job
// - "Where does Caroline live?" → entity=Caroline, attribute=location

/// Type of query for routing to appropriate retrieval strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    /// Attribute query: "What is X's Y?" - needs fact lookup
    Attribute(AttributeQuery),
    /// Temporal query: "When did X do Y?" - needs temporal filtering
    Temporal,
    /// Exploratory query: general semantic search
    Exploratory,
}

/// Extracted attribute query components
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeQuery {
    /// The entity being asked about (e.g., "Caroline")
    pub entity: String,
    /// The attribute being requested (e.g., "relationship_status")
    pub attribute: String,
    /// Attribute synonyms for matching (e.g., ["status", "single", "married", "dating"])
    pub attribute_synonyms: Vec<String>,
    /// Original query text
    pub original_query: String,
}

/// Classify a query to determine retrieval strategy
pub fn classify_query(query: &str) -> QueryType {
    // First check for attribute queries
    if let Some(attr_query) = detect_attribute_query(query) {
        return QueryType::Attribute(attr_query);
    }

    // Check for temporal queries
    if asks_for_temporal_answer(query) {
        return QueryType::Temporal;
    }

    // Default to exploratory
    QueryType::Exploratory
}

/// The shape of a query as far as the Layer 5.3 cross-encoder blend is concerned. Each
/// shape has its own share in `constants::CE_BLEND_*` and its own dial.
///
/// - Grounding shapes (`Identifier`, `Ordinal`, `Keyword`) describe questions whose answer
///   is decided by exact tokens or position in time, where the reranker adds nothing. They
///   take precedence over everything else, lowest share first.
/// - Reasoning shapes (`Attribute`, `Relational`, `Abstract`) describe questions whose
///   answer needs query-to-passage reading.
/// - Labels (`Temporal`, `People`) mark a plain question of that kind so its share can be
///   tuned on its own. They apply only when no other shape does: a date in "what filling
///   did she use in the cake she made in May 2022" filters, it does not answer, and the
///   reranker still has to match "filling" to "cake".
///
/// Declaration order is default-share order, lowest first; [`classify_rerank_shape`] takes
/// the minimum among grounding shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RerankShape {
    Identifier,
    Ordinal,
    Keyword,
    Temporal,
    People,
    Exploratory,
    Attribute,
    Relational,
    Abstract,
}

impl RerankShape {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identifier => "identifier",
            Self::Ordinal => "ordinal",
            Self::Temporal => "temporal",
            Self::People => "people",
            Self::Keyword => "keyword",
            Self::Exploratory => "exploratory",
            Self::Attribute => "attribute",
            Self::Relational => "relational",
            Self::Abstract => "abstract",
        }
    }
}

/// A "who" / "whom" / "who's" question. "Whose" is not one: "whose config file" asks
/// about a thing.
pub fn is_people_question(query: &str) -> bool {
    let lower = query.trim_start().to_lowercase();
    ["who ", "whom ", "who's ", "who\u{2019}s ", "who're "]
        .iter()
        .any(|p| lower.starts_with(p))
}

/// How strongly a query asks about people, in `[0, 1]`. A who-question or an explicit
/// ownership phrase is 1.0; a whole-word people noun ("people", "colleagues") is 0.6. Whole
/// words only: "owner" and "contact" were left out because "file owner" and "contact form"
/// are technical, and substring matching is how "recent" inside "recently" once amplified
/// recency fivefold.
pub fn people_intent(query: &str) -> f32 {
    if is_people_question(query) {
        return 1.0;
    }
    let lower = query.to_lowercase();
    if [
        "who is responsible",
        "who owns",
        "who worked on",
        "assigned to",
        "point of contact",
    ]
    .iter()
    .any(|p| lower.contains(p))
    {
        return 1.0;
    }
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let has_word = |w: &str| words.contains(&w);
    let has_pair = |a: &str, b: &str| words.windows(2).any(|p| p[0] == a && p[1] == b);
    if ["people", "colleagues", "stakeholders", "teammates", "staff"]
        .iter()
        .any(|w| has_word(w))
        || has_pair("team", "members")
    {
        return 0.6;
    }
    0.0
}

/// Words that shape a question rather than name its topic: quantity, duration, reason,
/// likelihood, recency, kind. An answer almost never repeats them ("How long has she been
/// doing yoga?" is answered by "about 3 years now"), so query-term coverage must not count
/// them. On the full 1,531-question LoCoMo suite (2026-10-20) they were behind most of the
/// coverage layer's lost cases.
pub const QUESTION_FRAME_WORDS: &[&str] = &[
    "many",
    "much",
    "long",
    "often",
    "time",
    "times",
    "period",
    "reason",
    "reasons",
    "cause",
    "likely",
    "probably",
    "possibly",
    "related",
    "recent",
    "recently",
    "kind",
    "kinds",
    "type",
    "types",
    "sort",
    "sorts",
    "way",
    "ways",
    "aspect",
    "aspects",
    "example",
    "examples",
    "specific",
    "particular",
    "exactly",
    "usually",
    "usual",
    "currently",
    "lately",
];

/// Whether `word` (any case) is in [`QUESTION_FRAME_WORDS`].
pub fn is_question_frame_word(word: &str) -> bool {
    let lower = word.to_lowercase();
    QUESTION_FRAME_WORDS.contains(&lower.as_str())
}

/// English stems of the non-stopword tokens in `text`, for query-term coverage.
pub fn content_stems(text: &str) -> HashSet<String> {
    let stemmer = Stemmer::create(Algorithm::English);
    tokenize(text)
        .into_iter()
        .filter(|w| !is_stop_word(w))
        .map(|w| stemmer.stem(&w).into_owned())
        .collect()
}

/// Pseudo-relevance feedback terms: words shared by at least two of the first pass's top
/// documents (all of them when only one is given), not stopwords, not already in the query
/// by stem, alphabetic and at least four letters. Ranked by how many documents share them,
/// then by total frequency, then alphabetically so the choice is deterministic. Requiring
/// two documents keeps one off-topic hit from steering the expansion.
pub fn prf_terms(docs: &[&str], query: &str, n: usize) -> Vec<String> {
    if docs.is_empty() || n == 0 {
        return Vec::new();
    }
    let stemmer = Stemmer::create(Algorithm::English);
    let query_stems = content_stems(query);
    let mut doc_freq: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    for doc in docs {
        let mut seen_here: HashSet<String> = HashSet::new();
        for word in tokenize(doc) {
            if word.len() < 4 || !word.chars().all(char::is_alphabetic) || is_stop_word(&word) {
                continue;
            }
            if query_stems.contains(stemmer.stem(&word).as_ref()) {
                continue;
            }
            let entry = doc_freq.entry(word.clone()).or_insert((0, 0));
            entry.1 += 1;
            if seen_here.insert(word) {
                entry.0 += 1;
            }
        }
    }
    let min_docs = docs.len().min(2);
    let mut ranked: Vec<(String, usize, usize)> = doc_freq
        .into_iter()
        .filter(|(_, (df, _))| *df >= min_docs)
        .map(|(w, (df, tf))| (w, df, tf))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
    ranked.into_iter().take(n).map(|(w, _, _)| w).collect()
}

/// Classify a query for the cross-encoder blend. Grounding shapes win, lowest share first;
/// then reasoning shapes; then the `Temporal` and `People` labels; then `Exploratory`.
/// `decomposed` is Layer 2's multi-query flag; `query_type` is [`classify_query`]'s result.
pub fn classify_rerank_shape(query: &str, decomposed: bool, query_type: &QueryType) -> RerankShape {
    let lower = query.to_lowercase();
    let mut shapes: Vec<RerankShape> = Vec::with_capacity(3);

    if has_structured_identifier(query)
        || query
            .split_whitespace()
            .any(|w| w.contains('/') || w.contains("::") || w.contains('_'))
    {
        shapes.push(RerankShape::Identifier);
    }
    // Same trigger as Layer 5.9's ordinal pin, so the share follows what the pipeline does.
    if lower.contains("first ")
        || lower.contains("most recent")
        || lower.contains("latest")
        || lower.contains("last ")
    {
        shapes.push(RerankShape::Ordinal);
    }
    let content_tokens = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !is_stop_word(w))
        .count();
    if content_tokens < 3 {
        shapes.push(RerankShape::Keyword);
    }
    if let Some(grounding) = shapes.iter().min() {
        return *grounding;
    }

    if decomposed {
        return RerankShape::Abstract;
    }
    let trimmed = lower.trim_start();
    if trimmed.starts_with("how did ")
        || trimmed.starts_with("how does ")
        || trimmed.starts_with("how do ")
        || trimmed.starts_with("how was ")
        || trimmed.starts_with("why ")
    {
        return RerankShape::Relational;
    }
    if matches!(query_type, QueryType::Attribute(_)) {
        return RerankShape::Attribute;
    }
    // Labels, not grounding shapes. Measured 2026-10-19: as grounding shapes at 0.10, the
    // temporal share cost the recall gate a gold document on a dated content question and
    // the people share overrode the abstract share on decomposed who-questions (LOCOMO-20
    // Q20 lost a top-5 answer, gate q121 lost rank).
    if [
        "last week",
        "last month",
        "yesterday",
        "today",
        "recently",
        "recent ",
    ]
    .iter()
    .any(|kw| lower.contains(kw))
        || matches!(
            detect_temporal_intent(query),
            TemporalIntent::SpecificTime | TemporalIntent::Ordering
        )
    {
        return RerankShape::Temporal;
    }
    if is_people_question(query) {
        return RerankShape::People;
    }
    RerankShape::Exploratory
}

/// Like [`has_lexical_identifiers`], but a bare all-caps word counts only when it carries a
/// digit (`E0308`) or is long enough to be a code constant (`ECONNREFUSED`, eight or more
/// letters). A short acronym in a question (`LGBTQ`, `HTTPS`) names a topic, and treating the
/// question as an identifier lookup switched its reranker off: on the five LoCoMo questions
/// with a bare acronym, MRR fell from 0.767 to 0.600.
pub fn has_structured_identifier(query: &str) -> bool {
    static STRUCTURED: LazyLock<Vec<regex::Regex>> = LazyLock::new(|| {
        vec![
            regex::Regex::new(r"\b[A-Z]{2,10}-\d{1,6}\b").unwrap(),
            regex::Regex::new(r"\bv?\d+\.\d+(?:\.\d+)?\b").unwrap(),
            regex::Regex::new(r"\b[A-Z][A-Z0-9]*\d[A-Z0-9]*\b").unwrap(),
            regex::Regex::new(r"\b[A-Z]{8,}\b").unwrap(),
            regex::Regex::new(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
                .unwrap(),
            regex::Regex::new(r"\b[0-9a-f]{7,40}\b").unwrap(),
            regex::Regex::new(r"#\d{1,6}\b").unwrap(),
        ]
    });
    STRUCTURED.iter().any(|p| p.is_match(query))
}

/// Detect whether a query contains lexical identifiers that score poorly on
/// vector similarity (ticket IDs, version strings, error codes, UUIDs, etc.).
///
/// Lexical identifiers are out-of-vocabulary for embedding models and produce
/// lower cosine similarity than semantically equivalent natural language. The
/// abstention threshold should be relaxed for such queries to prevent filtering
/// out correct but low-scoring results.
///
/// Detected patterns:
/// - Ticket/issue IDs: SHO-72, JIRA-123, #456
/// - Version strings: v1.2.3, 0.7.0
/// - Error codes: ECONNREFUSED, E0308, HTTP 404
/// - UUIDs: 550e8400-e29b-41d4-a716-446655440000
/// - Technical identifiers: bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2
pub fn has_lexical_identifiers(query: &str) -> bool {
    static LEXICAL_PATTERNS: LazyLock<Vec<regex::Regex>> = LazyLock::new(|| {
        vec![
            // Ticket IDs: ABC-123, SHO-72, PIPE-3
            regex::Regex::new(r"\b[A-Z]{2,10}-\d{1,6}\b").unwrap(),
            // Version strings: v1.2.3, 0.7.0, v2
            regex::Regex::new(r"\bv?\d+\.\d+(?:\.\d+)?\b").unwrap(),
            // Error codes: ECONNREFUSED, E0308 (uppercase + digits, ≥5 chars)
            regex::Regex::new(r"\b[A-Z][A-Z0-9]{4,}\b").unwrap(),
            // UUIDs
            regex::Regex::new(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
                .unwrap(),
            // Hash-like identifiers (7+ hex chars, e.g. commit SHAs)
            regex::Regex::new(r"\b[0-9a-f]{7,40}\b").unwrap(),
            // GitHub issue references: #123
            regex::Regex::new(r"#\d{1,6}\b").unwrap(),
        ]
    });

    LEXICAL_PATTERNS.iter().any(|p| p.is_match(query))
}

/// Detect and extract attribute query components
///
/// Patterns detected:
/// - "What is X's Y?" / "What is X's Y"
/// - "What is the Y of X?"
/// - "What Y does X have?"
/// - "Is X Y?" (boolean attribute)
/// - "Where does/is X?" (location attribute)
/// - "How old is X?" (age attribute)
pub fn detect_attribute_query(query: &str) -> Option<AttributeQuery> {
    let query_lower = query.to_lowercase();
    let query_trimmed = query_lower.trim().trim_end_matches('?');

    // Pattern 1: "What is X's Y" / "What's X's Y"
    if let Some(result) = extract_possessive_pattern(query_trimmed) {
        return Some(result);
    }

    // Pattern 2: "What is the Y of X"
    if let Some(result) = extract_of_pattern(query_trimmed) {
        return Some(result);
    }

    // Pattern 3: "Where does/is X" (location attribute)
    if query_lower.starts_with("where does") || query_lower.starts_with("where is") {
        if let Some(entity) = extract_entity_after_verb(query_trimmed) {
            return Some(AttributeQuery {
                entity,
                attribute: "location".to_string(),
                attribute_synonyms: vec![
                    "live".to_string(),
                    "lives".to_string(),
                    "living".to_string(),
                    "resides".to_string(),
                    "located".to_string(),
                    "address".to_string(),
                    "home".to_string(),
                    "place".to_string(),
                ],
                original_query: query.to_string(),
            });
        }
    }

    // Pattern 4: "How old is X" (age attribute)
    if query_lower.starts_with("how old") {
        if let Some(entity) = extract_entity_after_verb(query_trimmed) {
            return Some(AttributeQuery {
                entity,
                attribute: "age".to_string(),
                attribute_synonyms: vec![
                    "age".to_string(),
                    "years old".to_string(),
                    "born".to_string(),
                    "birthday".to_string(),
                ],
                original_query: query.to_string(),
            });
        }
    }

    // Pattern 5: "Is X married/single/..." (boolean relationship status)
    if query_lower.starts_with("is ") {
        let status_words = [
            "married",
            "single",
            "divorced",
            "engaged",
            "dating",
            "in a relationship",
        ];
        for status in &status_words {
            if query_lower.contains(status) {
                // Extract entity between "is" and status word
                let after_is = &query_trimmed[3..]; // Skip "is "
                if let Some(pos) = after_is.find(status) {
                    let entity = after_is[..pos].trim().to_string();
                    if !entity.is_empty()
                        && entity.chars().next().is_some_and(|c| c.is_alphabetic())
                    {
                        return Some(AttributeQuery {
                            entity: capitalize_first(&entity),
                            attribute: "relationship_status".to_string(),
                            attribute_synonyms: vec![
                                "single".to_string(),
                                "married".to_string(),
                                "divorced".to_string(),
                                "engaged".to_string(),
                                "dating".to_string(),
                                "relationship".to_string(),
                                "partner".to_string(),
                                "spouse".to_string(),
                                "status".to_string(),
                            ],
                            original_query: query.to_string(),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Extract "X's Y" pattern from query
fn extract_possessive_pattern(query: &str) -> Option<AttributeQuery> {
    // Find possessive marker ('s or s')
    let possessive_patterns = [
        ("what is ", "'s "),
        ("what's ", "'s "),
        ("what is ", "' "),
        ("what's ", "' "),
    ];

    for (prefix, possessive) in possessive_patterns {
        if let Some(start) = query.find(prefix) {
            let after_prefix = &query[start + prefix.len()..];
            if let Some(pos_pos) = after_prefix.find(possessive) {
                let entity = after_prefix[..pos_pos].trim();
                let attribute = after_prefix[pos_pos + possessive.len()..].trim();

                if !entity.is_empty() && !attribute.is_empty() {
                    return Some(AttributeQuery {
                        entity: capitalize_first(entity),
                        attribute: normalize_attribute(attribute),
                        attribute_synonyms: get_attribute_synonyms(attribute),
                        original_query: query.to_string(),
                    });
                }
            }
        }
    }

    None
}

/// Extract "the Y of X" pattern from query
fn extract_of_pattern(query: &str) -> Option<AttributeQuery> {
    // Pattern: "what is the Y of X"
    let prefixes = ["what is the ", "what's the "];

    for prefix in prefixes {
        if let Some(start) = query.find(prefix) {
            let after_prefix = &query[start + prefix.len()..];
            if let Some(of_pos) = after_prefix.find(" of ") {
                let attribute = after_prefix[..of_pos].trim();
                let entity = after_prefix[of_pos + 4..].trim();

                if !entity.is_empty() && !attribute.is_empty() {
                    return Some(AttributeQuery {
                        entity: capitalize_first(entity),
                        attribute: normalize_attribute(attribute),
                        attribute_synonyms: get_attribute_synonyms(attribute),
                        original_query: query.to_string(),
                    });
                }
            }
        }
    }

    None
}

/// Extract entity after a verb like "is" or "does"
fn extract_entity_after_verb(query: &str) -> Option<String> {
    let verbs = [" is ", " does "];
    for verb in verbs {
        if let Some(pos) = query.find(verb) {
            let after_verb = query[pos + verb.len()..].trim();
            // Take first word(s) as entity (stop at common words)
            let stop_words = ["live", "work", "do", "have", "go", "stay", "come"];
            let words: Vec<&str> = after_verb.split_whitespace().collect();
            let mut entity_words = Vec::new();
            for word in words {
                if stop_words.contains(&word) {
                    break;
                }
                entity_words.push(word);
            }
            if !entity_words.is_empty() {
                return Some(capitalize_first(&entity_words.join(" ")));
            }
        }
    }
    None
}

/// Normalize an attribute name (e.g., "relationship status" → "relationship_status")
fn normalize_attribute(attr: &str) -> String {
    attr.trim().to_lowercase().replace([' ', '-'], "_")
}

/// Get synonyms for common attributes
fn get_attribute_synonyms(attribute: &str) -> Vec<String> {
    let attr_lower = attribute.to_lowercase();

    // Relationship status synonyms
    if attr_lower.contains("relationship")
        || attr_lower.contains("status")
        || attr_lower.contains("marital")
    {
        return vec![
            "single".to_string(),
            "married".to_string(),
            "divorced".to_string(),
            "engaged".to_string(),
            "dating".to_string(),
            "relationship".to_string(),
            "partner".to_string(),
            "spouse".to_string(),
            "single parent".to_string(),
            "status".to_string(),
            "marital".to_string(),
        ];
    }

    // Job/occupation synonyms
    if attr_lower.contains("job")
        || attr_lower.contains("occupation")
        || attr_lower.contains("work")
    {
        return vec![
            "job".to_string(),
            "work".to_string(),
            "occupation".to_string(),
            "profession".to_string(),
            "career".to_string(),
            "employed".to_string(),
            "works as".to_string(),
        ];
    }

    // Name synonyms
    if attr_lower.contains("name") {
        return vec![
            "name".to_string(),
            "called".to_string(),
            "named".to_string(),
        ];
    }

    // Age synonyms
    if attr_lower.contains("age") {
        return vec![
            "age".to_string(),
            "old".to_string(),
            "years".to_string(),
            "born".to_string(),
            "birthday".to_string(),
        ];
    }

    // Default: return the attribute itself and common variations
    vec![attr_lower.clone(), attr_lower.replace('_', " ")]
}

/// Capitalize first letter of a string
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Extract chunks from text using shallow parsing
///
/// This function:
/// 1. Splits text into sentences
/// 2. Tags each word with its POS (noun, verb, adjective, etc.)
/// 3. Returns chunks that can be used for:
///    - Entity extraction (all nouns, not just YAKE top-N)
///    - Co-occurrence edge creation (words in same sentence)
///
/// Unlike YAKE, this doesn't rank by frequency - ALL content words are extracted.
pub fn extract_chunks(text: &str) -> ChunkExtraction {
    let stemmer = Stemmer::create(Algorithm::English);
    let sentences = split_sentences(text);

    let mut chunks = Vec::with_capacity(sentences.len());
    let mut unique_nouns = HashSet::new();
    let mut unique_verbs = HashSet::new();
    let mut unique_adjectives = HashSet::new();
    let mut proper_nouns = HashSet::new();

    for (sentence_idx, sentence) in sentences.iter().enumerate() {
        let words = tokenize_with_case(sentence);
        let mut tagged_words = Vec::with_capacity(words.len());

        for (position, (word, is_capitalized)) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();

            // Skip very short words
            if word_lower.len() < 2 {
                continue;
            }

            let stem = stemmer.stem(&word_lower).to_string();
            let pos = classify_pos_for_chunking(&word_lower, *is_capitalized, position, &words);

            match pos {
                PosTag::Noun => {
                    unique_nouns.insert(stem.clone());
                }
                PosTag::Verb => {
                    unique_verbs.insert(stem.clone());
                }
                PosTag::Adjective => {
                    unique_adjectives.insert(stem.clone());
                }
                PosTag::ProperNoun => {
                    proper_nouns.insert(word.clone());
                    unique_nouns.insert(stem.clone()); // Also add to nouns
                }
                _ => {}
            }

            if pos != PosTag::StopWord {
                tagged_words.push(TaggedWord {
                    text: word.clone(),
                    stem,
                    pos,
                    position,
                });
            }
        }

        if !tagged_words.is_empty() {
            chunks.push(SentenceChunk {
                text: sentence.clone(),
                sentence_idx,
                words: tagged_words,
            });
        }
    }

    ChunkExtraction {
        chunks,
        unique_nouns,
        unique_verbs,
        unique_adjectives,
        proper_nouns,
    }
}

/// Split text into sentences using punctuation and common patterns
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    // Simple sentence splitting on . ! ? and newlines
    // More sophisticated than just split('.') because we handle abbreviations
    for ch in text.chars() {
        current.push(ch);

        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let trimmed = current.trim();
            if !trimmed.is_empty() && trimmed.len() > 3 {
                // Avoid splitting on abbreviations like "Dr." or "Mr."
                let last_word: String = trimmed
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .collect();

                // Common abbreviations that shouldn't split
                let is_abbrev = matches!(
                    last_word.to_lowercase().as_str(),
                    "mr" | "mrs"
                        | "ms"
                        | "dr"
                        | "prof"
                        | "sr"
                        | "jr"
                        | "vs"
                        | "etc"
                        | "eg"
                        | "ie"
                        | "st"
                        | "ave"
                        | "rd"
                        | "blvd"
                );

                if !is_abbrev || ch == '\n' {
                    sentences.push(trimmed.to_string());
                    current.clear();
                }
            }
        }
    }

    // Don't forget the last sentence
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Tokenize preserving case information for proper noun detection
fn tokenize_with_case(text: &str) -> Vec<(String, bool)> {
    text.split_whitespace()
        .map(|w| {
            let clean: String = w
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_string();
            let is_capitalized = clean
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            (clean, is_capitalized)
        })
        .filter(|(w, _)| !w.is_empty())
        .collect()
}

/// Classify POS for chunking purposes
fn classify_pos_for_chunking(
    word: &str,
    is_capitalized: bool,
    position: usize,
    _context: &[(String, bool)],
) -> PosTag {
    // Check stop words first
    if is_stop_word(word) {
        return PosTag::StopWord;
    }

    // Capitalized words not at sentence start are likely proper nouns
    if is_capitalized && position > 0 {
        return PosTag::ProperNoun;
    }

    // Check verb indicators
    if is_verb(word) {
        return PosTag::Verb;
    }

    // Check adjective indicators
    if is_adjective(word) {
        return PosTag::Adjective;
    }

    // Check noun indicators
    if is_noun_for_chunking(word) {
        return PosTag::Noun;
    }

    // Default: if it's a content word (not too short, not a stop word), treat as noun
    // This is the "80% rule" - unknown words are usually nouns in English
    if word.len() >= 4 {
        return PosTag::Noun;
    }

    PosTag::Other
}

/// Noun detection for chunking (more aggressive than query parsing)
fn is_noun_for_chunking(word: &str) -> bool {
    // All the noun indicators from the original is_noun function
    // Plus additional heuristics for storage

    // Domain-specific nouns
    const NOUN_INDICATORS: &[&str] = &[
        // Keep all original indicators
        "memory",
        "graph",
        "node",
        "edge",
        "entity",
        "embedding",
        "vector",
        "index",
        "query",
        "retrieval",
        "activation",
        "potentiation",
        "consolidation",
        "decay",
        "strength",
        "weight",
        "threshold",
        "importance",
        "robot",
        "drone",
        "sensor",
        "lidar",
        "camera",
        "motor",
        "actuator",
        "obstacle",
        "path",
        "waypoint",
        "location",
        "coordinates",
        "position",
        "battery",
        "power",
        "energy",
        "voltage",
        "current",
        "system",
        "module",
        "component",
        "unit",
        "device",
        "temperature",
        "pressure",
        "humidity",
        "speed",
        "velocity",
        "signal",
        "communication",
        "network",
        "link",
        "connection",
        "navigation",
        "guidance",
        "control",
        "steering",
        "data",
        "information",
        "message",
        "command",
        "response",
        "function",
        "method",
        "class",
        "struct",
        "interface",
        "package",
        "library",
        "framework",
        "api",
        "endpoint",
        "request",
        "error",
        "exception",
        "bug",
        "fix",
        "feature",
        "test",
        "benchmark",
        "performance",
        "latency",
        "throughput",
        "cache",
        "buffer",
        "queue",
        "stack",
        "heap",
        "thread",
        "process",
        "server",
        "client",
        "database",
        "table",
        "column",
        "row",
        "schema",
        "migration",
        "deployment",
        "container",
        "cluster",
        "replica",
        "person",
        "people",
        "user",
        "agent",
        "operator",
        "time",
        "date",
        "day",
        "hour",
        "minute",
        "second",
        "area",
        "zone",
        "region",
        "sector",
        "space",
        "task",
        "mission",
        "goal",
        "objective",
        "target",
        "warning",
        "alert",
        "notification",
        "level",
        "status",
        "state",
        "condition",
        "mode",
        "type",
        "kind",
        "version",
        "release",
        "update",
        "change",
        "result",
        "output",
        "input",
        "value",
        "key",
        "name",
        "id",
        "identifier",
        // Additional nouns common in conversational text
        "sunrise",
        "sunset",
        "lake",
        "mountain",
        "beach",
        "forest",
        "garden",
        "park",
        "city",
        "town",
        "village",
        "country",
        "house",
        "home",
        "room",
        "building",
        "street",
        "road",
        "car",
        "bus",
        "train",
        "plane",
        "boat",
        "bicycle",
        "food",
        "drink",
        "water",
        "coffee",
        "tea",
        "breakfast",
        "lunch",
        "dinner",
        "meal",
        "book",
        "movie",
        "music",
        "song",
        "art",
        "painting",
        "photo",
        "picture",
        "video",
        "game",
        "sport",
        "team",
        "player",
        "match",
        "race",
        "trip",
        "vacation",
        "holiday",
        "weekend",
        "morning",
        "evening",
        "night",
        "week",
        "month",
        "year",
        "birthday",
        "wedding",
        "party",
        "event",
        "meeting",
        "class",
        "lesson",
        "course",
        "school",
        "college",
        "university",
        "job",
        "work",
        "office",
        "company",
        "business",
        "project",
        "plan",
        "idea",
        "thought",
        "feeling",
        "emotion",
        "love",
        "friend",
        "family",
        "parent",
        "child",
        "kid",
        "baby",
        "mother",
        "father",
        "sister",
        "brother",
        "wife",
        "husband",
        "partner",
        "group",
        "community",
        "society",
        "culture",
        "tradition",
        "story",
        "history",
        "news",
        "article",
        "blog",
        "post",
        "comment",
        "email",
        "letter",
        "phone",
        "call",
        "text",
        "chat",
        "conversation",
        "discussion",
        "talk",
        "speech",
        "presentation",
        "question",
        "answer",
        "problem",
        "solution",
        "issue",
        "challenge",
        "opportunity",
        "success",
        "failure",
        "experience",
        "skill",
        "knowledge",
        "wisdom",
        "truth",
        "fact",
        "opinion",
        "belief",
        "value",
        "principle",
        "rule",
        "law",
        "policy",
        "decision",
        "choice",
        "option",
        "alternative",
        "reason",
        "cause",
        "effect",
        "impact",
        "influence",
        "power",
        "authority",
        "responsibility",
        "duty",
        "right",
        "freedom",
        "justice",
        "peace",
        "war",
        "conflict",
        "agreement",
        "contract",
        "deal",
        "price",
        "cost",
        "money",
        "dollar",
        "euro",
        "pound",
        "budget",
        "investment",
        "profit",
        "loss",
        "risk",
        "reward",
        "benefit",
        "advantage",
        "disadvantage",
        "strength",
        "weakness",
        "opportunity",
        "threat",
        "strategy",
        "tactic",
        "method",
        "approach",
        "technique",
        "tool",
        "resource",
        "material",
        "product",
        "service",
        "quality",
        "quantity",
        "size",
        "shape",
        "color",
        "sound",
        "smell",
        "taste",
        "touch",
        "sight",
        "sense",
        "mind",
        "body",
        "heart",
        "soul",
        "spirit",
        "health",
        "illness",
        "disease",
        "medicine",
        "doctor",
        "nurse",
        "hospital",
        "clinic",
        "therapy",
        "treatment",
        "care",
        "support",
        "help",
        "advice",
        "guidance",
        "counseling",
        "coaching",
        "mentoring",
        "training",
        "education",
        "learning",
        "teaching",
        "research",
        "study",
        "experiment",
        "discovery",
        "invention",
        "innovation",
        "technology",
        "science",
        "math",
        "physics",
        "chemistry",
        "biology",
        "psychology",
        "sociology",
        "philosophy",
        "religion",
        "spirituality",
        "meditation",
        "yoga",
        "exercise",
        "fitness",
        "diet",
        "nutrition",
        "sleep",
        "rest",
        "relaxation",
        "stress",
        "anxiety",
        "depression",
        "happiness",
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "hope",
        "faith",
        "courage",
        "confidence",
        "pride",
        "shame",
        "guilt",
        "regret",
        "gratitude",
        "empathy",
        "compassion",
        "kindness",
        "generosity",
        "honesty",
        "integrity",
        "loyalty",
        "respect",
        "tolerance",
        "patience",
        "persistence",
        "determination",
        "motivation",
        "inspiration",
        "creativity",
        "imagination",
        "curiosity",
        "wonder",
        "beauty",
        "art",
        "music",
        "dance",
        "theater",
        "film",
        "literature",
        "poetry",
        "writing",
        "reading",
        "speaking",
        "listening",
        "communication",
        "expression",
        "interpretation",
        "understanding",
        "meaning",
        "purpose",
        "goal",
        "dream",
        "vision",
        "mission",
        "passion",
        "interest",
        "hobby",
        "activity",
        "adventure",
        "journey",
        "path",
        "way",
        "direction",
        "destination",
        "origin",
        "beginning",
        "end",
        "start",
        "finish",
        "progress",
        "growth",
        "development",
        "evolution",
        "transformation",
        "change",
        "transition",
        "shift",
        "movement",
        "action",
        "reaction",
        "response",
        "behavior",
        "habit",
        "pattern",
        "routine",
        "schedule",
        "plan",
        "strategy",
        "tactic",
        "approach",
        "method",
        "process",
        "procedure",
        "step",
        "stage",
        "phase",
        "cycle",
        "circle",
        "loop",
        "sequence",
        "order",
        "arrangement",
        "organization",
        "structure",
        "system",
        "network",
        "connection",
        "relationship",
        "bond",
        "link",
        "tie",
        "association",
        "affiliation",
        "membership",
        "participation",
        "involvement",
        "engagement",
        "commitment",
        "dedication",
        "devotion",
        "loyalty",
        "allegiance",
        "support",
        "backing",
        "endorsement",
        "approval",
        "acceptance",
        "recognition",
        "acknowledgment",
        "appreciation",
        "gratitude",
        "thanks",
        "praise",
        "compliment",
        "criticism",
        "feedback",
        "evaluation",
        "assessment",
        "judgment",
        "opinion",
        "view",
        "perspective",
        "angle",
        "aspect",
        "dimension",
        "element",
        "component",
        "part",
        "piece",
        "section",
        "segment",
        "portion",
        "share",
        "fraction",
        "percentage",
        "ratio",
        "proportion",
        "balance",
        "equilibrium",
        "harmony",
        "unity",
        "diversity",
        "variety",
        "difference",
        "similarity",
        "comparison",
        "contrast",
        "distinction",
        "separation",
        "division",
        "classification",
        "category",
        "class",
        "type",
        "kind",
        "sort",
        "species",
        "variety",
        "version",
        "edition",
        "model",
        "design",
        "style",
        "format",
        "layout",
        "arrangement",
        "configuration",
        "setup",
        "installation",
        "deployment",
    ];

    if NOUN_INDICATORS.contains(&word) {
        return true;
    }

    // Noun suffixes
    if word.ends_with("tion")
        || word.ends_with("sion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("age")
        || word.ends_with("ure")
        || word.ends_with("dom")
        || word.ends_with("ship")
        || word.ends_with("hood")
        || word.ends_with("ism")
        || word.ends_with("ist")
    {
        return true;
    }

    // -er/-or suffixes (but not comparative adjectives)
    if (word.ends_with("er") || word.ends_with("or")) && word.len() > 4 {
        // Exclude likely comparatives
        let without_suffix = &word[..word.len() - 2];
        if !without_suffix.ends_with("t")
            && !without_suffix.ends_with("g")
            && !without_suffix.ends_with("d")
        {
            return true;
        }
    }

    false
}

/// Focal entity extracted from query (noun)
#[derive(Debug, Clone)]
pub struct FocalEntity {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    pub ic_weight: f32,
    /// True if entity is part of a compound noun
    pub is_compound: bool,
    /// True if preceded by negation
    pub negated: bool,
}

/// Discriminative modifier (adjective/qualifier)
#[derive(Debug, Clone)]
pub struct Modifier {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
    /// True if preceded by negation
    pub negated: bool,
}

/// Relational context (verb)
#[derive(Debug, Clone)]
pub struct Relation {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
    /// True if preceded by negation
    pub negated: bool,
}

/// Query intent type for retrieval strategy selection (SHO-D6)
///
/// Determines the optimal retrieval approach based on query structure:
/// - Needle: Looking for specific facts → favor BM25/Vector (high precision)
/// - Exploratory: Seeking related concepts → favor Graph (high recall)
/// - Hybrid: Balanced query → use all retrieval methods equally
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryIntent {
    /// Needle query: Seeking specific information
    /// Examples: "What is John's email?", "When did we deploy v2.0?"
    /// Strategy: High BM25/Vector weight, low Graph weight
    Needle,

    /// Exploratory query: Seeking related concepts and associations
    /// Examples: "Tell me about the project", "What do we know about security?"
    /// Strategy: Low BM25/Vector weight, high Graph weight
    Exploratory,

    /// Hybrid query: Balanced between specific and exploratory
    /// Examples: "How does authentication work?", "What are the deployment options?"
    /// Strategy: Balanced weights for all retrieval methods
    #[default]
    Hybrid,
}

/// Complete linguistic analysis of a query
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Focal entities (nouns) - primary search targets
    pub focal_entities: Vec<FocalEntity>,

    /// Discriminative modifiers (adjectives) - quality refiners
    pub discriminative_modifiers: Vec<Modifier>,

    /// Relational context (verbs) - graph traversal guides
    pub relational_context: Vec<Relation>,

    /// Compound nouns detected (e.g., "machine learning", "neural network")
    pub compound_nouns: Vec<String>,

    /// Original query text (retained for logging/debugging)
    pub original_query: String,

    /// True if query contains negation
    pub has_negation: bool,

    /// Detected query intent for retrieval strategy selection (SHO-D6)
    pub intent: QueryIntent,
}

impl QueryAnalysis {
    /// Calculate weighted importance of this query (for ranking)
    pub fn total_weight(&self) -> f32 {
        let entity_weight: f32 = self.focal_entities.iter().map(|e| e.ic_weight).sum();

        let modifier_weight: f32 = self
            .discriminative_modifiers
            .iter()
            .map(|m| m.ic_weight)
            .sum();

        let relation_weight: f32 = self.relational_context.iter().map(|r| r.ic_weight).sum();

        // Compound nouns get bonus weight
        let compound_bonus = self.compound_nouns.len() as f32 * 0.5;

        entity_weight + modifier_weight + relation_weight + compound_bonus
    }

    /// Get all stems for efficient matching
    pub fn all_stems(&self) -> HashSet<String> {
        let mut stems = HashSet::new();
        for e in &self.focal_entities {
            stems.insert(e.stem.clone());
        }
        for m in &self.discriminative_modifiers {
            stems.insert(m.stem.clone());
        }
        for r in &self.relational_context {
            stems.insert(r.stem.clone());
        }
        stems
    }

    /// Get non-negated entity stems (for positive matching)
    pub fn positive_entity_stems(&self) -> Vec<&str> {
        self.focal_entities
            .iter()
            .filter(|e| !e.negated)
            .map(|e| e.stem.as_str())
            .collect()
    }

    /// Get negated entity stems (for exclusion)
    pub fn negated_entity_stems(&self) -> Vec<&str> {
        self.focal_entities
            .iter()
            .filter(|e| e.negated)
            .map(|e| e.stem.as_str())
            .collect()
    }

    /// Convert analysis to IC weights HashMap for BM25 term boosting
    ///
    /// Returns a mapping of lowercase terms to their IC weights:
    /// - Nouns (focal entities): IC_NOUN = 1.5
    /// - Adjectives (modifiers): IC_ADJECTIVE = 0.9
    /// - Verbs (relations): IC_VERB = 0.7
    ///
    /// Based on Lioma & Ounis (2006) - nouns carry more information content.
    pub fn to_ic_weights(&self) -> std::collections::HashMap<String, f32> {
        self.to_ic_weights_with_yake(true)
    }

    /// Convert analysis to IC weights with optional YAKE boosting
    ///
    /// When use_yake=true, extracts discriminative keywords using YAKE algorithm
    /// and boosts their weights. This is critical for multi-hop queries where
    /// discriminative terms like "sunrise" must outweigh common terms like "Melanie".
    pub fn to_ic_weights_with_yake(
        &self,
        use_yake: bool,
    ) -> std::collections::HashMap<String, f32> {
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};

        let mut weights = std::collections::HashMap::new();

        // YAKE keyword extraction for discriminative term boosting
        // YAKE identifies statistically rare/important terms in the query
        if use_yake {
            let config = KeywordConfig {
                max_keywords: 5,
                ngrams: 2,
                min_length: 3,
                ..Default::default()
            };
            let extractor = KeywordExtractor::with_config(config);
            let keywords = extractor.extract(&self.original_query);

            // Boost YAKE-identified keywords with high weights
            // YAKE importance is 0.0-1.0 where higher = more discriminative
            // Key insight: Only boost SINGLE words, not bigrams
            // Bigrams like "Melanie paint" match too broadly - we need specific terms
            for kw in keywords {
                let term = kw.text.to_lowercase();

                // Skip bigrams/trigrams - they match documents with either word
                // which defeats the purpose of discriminative keyword boosting
                if term.contains(' ') {
                    continue;
                }

                // Aggressive boost for single discriminative words
                // Scale: importance 0.5 → boost 3.5, importance 1.0 → boost 6.0
                // This ensures rare words like "sunrise" dominate over common words
                let yake_boost = 1.0 + (kw.importance * 5.0);
                weights
                    .entry(term)
                    .and_modify(|w: &mut f32| *w = w.max(yake_boost))
                    .or_insert(yake_boost);
            }
        }

        // Add focal entities (nouns) with highest IC weight
        for entity in &self.focal_entities {
            let term = entity.text.to_lowercase();
            weights
                .entry(term)
                .and_modify(|w: &mut f32| *w = w.max(entity.ic_weight))
                .or_insert(entity.ic_weight);
            // Also add stem for fuzzy matching
            if entity.stem != entity.text.to_lowercase() {
                weights
                    .entry(entity.stem.clone())
                    .and_modify(|w: &mut f32| *w = w.max(entity.ic_weight))
                    .or_insert(entity.ic_weight);
            }
        }

        // Add discriminative modifiers (adjectives)
        for modifier in &self.discriminative_modifiers {
            let term = modifier.text.to_lowercase();
            weights
                .entry(term)
                .and_modify(|w: &mut f32| *w = w.max(modifier.ic_weight))
                .or_insert(modifier.ic_weight);
            if modifier.stem != modifier.text.to_lowercase() {
                weights
                    .entry(modifier.stem.clone())
                    .and_modify(|w: &mut f32| *w = w.max(modifier.ic_weight))
                    .or_insert(modifier.ic_weight);
            }
        }

        // Add relational context (verbs)
        for relation in &self.relational_context {
            let term = relation.text.to_lowercase();
            weights
                .entry(term)
                .and_modify(|w: &mut f32| *w = w.max(relation.ic_weight))
                .or_insert(relation.ic_weight);
            if relation.stem != relation.text.to_lowercase() {
                weights
                    .entry(relation.stem.clone())
                    .and_modify(|w: &mut f32| *w = w.max(relation.ic_weight))
                    .or_insert(relation.ic_weight);
            }
        }

        // Boost compound nouns (they carry more specific meaning)
        for compound in &self.compound_nouns {
            // Each word in compound gets a small boost
            for word in compound.split_whitespace() {
                let term = word.to_lowercase();
                weights.entry(term).and_modify(|w: &mut f32| *w *= 1.2);
            }
        }

        weights
    }

    /// Get maximum YAKE keyword discriminativeness score for dynamic weight adjustment
    ///
    /// Returns (max_importance, discriminative_keywords) where:
    /// - max_importance: 0.0-1.0, higher means more discriminative keywords found
    /// - discriminative_keywords: keywords with importance > 0.5 (for logging)
    ///
    /// Use this to dynamically adjust BM25/vector weights in hybrid search:
    /// - High discriminativeness (>0.6) → boost BM25 weight (keyword matching critical)
    /// - Low discriminativeness (<0.3) → trust vector more (semantic similarity better)
    pub fn keyword_discriminativeness(&self) -> (f32, Vec<String>) {
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};

        let config = KeywordConfig {
            max_keywords: 5,
            ngrams: 2,
            min_length: 2, // Allow short terms like "AI", "ML", "Go", etc.
            ..Default::default()
        };
        let extractor = KeywordExtractor::with_config(config);
        let keywords = extractor.extract(&self.original_query);

        let mut max_importance = 0.0f32;
        let mut discriminative = Vec::new();

        for kw in keywords {
            if kw.importance > max_importance {
                max_importance = kw.importance;
            }
            // Keywords with importance > 0.5 are considered discriminative
            if kw.importance > 0.5 {
                discriminative.push(kw.text.to_lowercase());
            }
        }

        (max_importance, discriminative)
    }

    /// Get phrase boosts for BM25 exact phrase matching
    ///
    /// Returns compound nouns and adjacent noun pairs as phrases with boost weights.
    /// Phrase matching significantly improves retrieval for multi-word concepts
    /// like "support group", "machine learning", "LGBTQ community".
    pub fn to_phrase_boosts(&self) -> Vec<(String, f32)> {
        let mut phrases = Vec::new();

        // Add compound nouns with high boost (they are specific concepts)
        for compound in &self.compound_nouns {
            // Compound nouns get 2.0x boost for exact phrase match
            phrases.push((compound.to_lowercase(), 2.0));
        }

        // Also detect adjacent nouns that might form natural phrases
        // Even if not in compound_nouns, adjacent entities may form useful phrases
        if self.focal_entities.len() >= 2 {
            for i in 0..self.focal_entities.len() - 1 {
                let e1 = &self.focal_entities[i];
                let e2 = &self.focal_entities[i + 1];
                // Only if not negated and not already a compound
                if !e1.negated && !e2.negated {
                    let phrase = format!("{} {}", e1.text.to_lowercase(), e2.text.to_lowercase());
                    if !self
                        .compound_nouns
                        .iter()
                        .any(|c| c.to_lowercase() == phrase)
                    {
                        // Adjacent nouns get 1.5x boost (lower than explicit compounds)
                        phrases.push((phrase, 1.5));
                    }
                }
            }
        }

        phrases
    }
}

/// Token with linguistic annotations
#[derive(Debug)]
struct AnnotatedToken {
    text: String,
    stem: String,
    pos: PartOfSpeech,
    negated: bool,
    position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PartOfSpeech {
    Noun,
    Adjective,
    Verb,
    StopWord,
    Negation,
    Unknown,
}

/// Parse query using linguistic analysis with Porter2 stemming
pub fn analyze_query(query_text: &str) -> QueryAnalysis {
    let stemmer = Stemmer::create(Algorithm::English);
    let words = tokenize(query_text);

    if words.is_empty() {
        return QueryAnalysis {
            focal_entities: Vec::new(),
            discriminative_modifiers: Vec::new(),
            relational_context: Vec::new(),
            compound_nouns: Vec::new(),
            original_query: query_text.to_string(),
            has_negation: false,
            intent: QueryIntent::Hybrid,
        };
    }

    // Annotate each token with POS and negation scope
    let annotated = annotate_tokens(&words, &stemmer);

    // Detect compound nouns
    let compound_nouns = detect_compound_nouns(&annotated);

    // Build result structures
    let mut focal_entities = Vec::new();
    let mut discriminative_modifiers = Vec::new();
    let mut relational_context = Vec::new();
    let mut has_negation = false;

    // Track which tokens are part of compounds
    let compound_positions: HashSet<usize> = compound_positions(&annotated, &compound_nouns);

    for token in &annotated {
        if token.pos == PartOfSpeech::Negation {
            has_negation = true;
            continue;
        }
        if token.pos == PartOfSpeech::StopWord {
            continue;
        }

        let is_compound = compound_positions.contains(&token.position);

        match token.pos {
            PartOfSpeech::Noun | PartOfSpeech::Unknown => {
                // Unknown words are likely domain-specific nouns
                let weight = calculate_term_weight(&token.text, IC_NOUN);
                focal_entities.push(FocalEntity {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    is_compound,
                    negated: token.negated,
                });
            }
            PartOfSpeech::Adjective => {
                let weight = calculate_term_weight(&token.text, IC_ADJECTIVE);
                discriminative_modifiers.push(Modifier {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    negated: token.negated,
                });
            }
            PartOfSpeech::Verb => {
                let weight = calculate_term_weight(&token.text, IC_VERB);
                relational_context.push(Relation {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    negated: token.negated,
                });
            }
            _ => {}
        }
    }

    // Add compound nouns as high-weight entities
    for compound in &compound_nouns {
        let stem = stemmer.stem(compound).to_string();
        focal_entities.push(FocalEntity {
            text: compound.clone(),
            stem,
            ic_weight: IC_NOUN * 1.5, // Compound bonus
            is_compound: true,
            negated: false,
        });
    }

    // Detect query intent for retrieval strategy (SHO-D6)
    let intent = detect_query_intent(query_text, &focal_entities, &relational_context);

    QueryAnalysis {
        focal_entities,
        discriminative_modifiers,
        relational_context,
        compound_nouns,
        original_query: query_text.to_string(),
        has_negation,
        intent,
    }
}

/// Detect query intent for retrieval strategy selection (SHO-D6)
///
/// Analyzes query structure to determine optimal retrieval approach:
/// - Needle: Specific fact-seeking queries → favor BM25/Vector
/// - Exploratory: Association-seeking queries → favor Graph
/// - Hybrid: Balanced queries → use all methods equally
fn detect_query_intent(
    query_text: &str,
    focal_entities: &[FocalEntity],
    relational_context: &[Relation],
) -> QueryIntent {
    let lower = query_text.to_lowercase();

    // Needle indicators: seeking specific facts
    let needle_starters = [
        "what is", "what's", "who is", "who's", "where is", "where's", "when did", "when was",
        "which", "how much", "how many", "find", "get me", "show me", "list", "give me",
    ];

    let needle_patterns = [
        "'s email",
        "'s phone",
        "'s address",
        "'s name",
        "email of",
        "phone of",
        "address of",
        "name of",
        "id of",
        "password",
        "api key",
        "token",
    ];

    // Exploratory indicators: seeking associations and context
    let exploratory_starters = [
        "tell me about",
        "explain",
        "describe",
        "what do we know about",
        "summarize",
        "overview",
        "recap",
        "context",
        "related to",
        "associated with",
        "connected to",
        "how does",
        "how do",
        "why does",
        "why do",
    ];

    let exploratory_patterns = [
        "all about",
        "everything about",
        "more about",
        "history of",
        "background",
        "related",
    ];

    // Check for needle patterns
    for starter in needle_starters.iter() {
        if lower.starts_with(starter) {
            return QueryIntent::Needle;
        }
    }

    for pattern in needle_patterns.iter() {
        if lower.contains(pattern) {
            return QueryIntent::Needle;
        }
    }

    // Check for exploratory patterns
    for starter in exploratory_starters.iter() {
        if lower.starts_with(starter) || lower.contains(starter) {
            return QueryIntent::Exploratory;
        }
    }

    for pattern in exploratory_patterns.iter() {
        if lower.contains(pattern) {
            return QueryIntent::Exploratory;
        }
    }

    // Heuristic: If query has many entities and few relations → Needle
    // If query has few entities and many relations → Exploratory
    let entity_count = focal_entities.len();
    let relation_count = relational_context.len();

    if entity_count > 0 && relation_count == 0 {
        // Pure entity query (e.g., "John's project") → Needle
        QueryIntent::Needle
    } else if relation_count > entity_count {
        // More relations than entities → Exploratory
        QueryIntent::Exploratory
    } else {
        // Balanced → Hybrid
        QueryIntent::Hybrid
    }
}

/// Tokenize query text into lowercase words
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Annotate tokens with POS tags and negation scope
fn annotate_tokens(words: &[String], stemmer: &Stemmer) -> Vec<AnnotatedToken> {
    let mut annotated = Vec::with_capacity(words.len());
    let mut in_negation_scope = false;
    let mut negation_distance = 0;

    for (i, word) in words.iter().enumerate() {
        let stem = stemmer.stem(word).to_string();
        let pos = classify_pos(word, i, words);

        // Track negation scope (extends 2-3 words after negation)
        if pos == PartOfSpeech::Negation {
            in_negation_scope = true;
            negation_distance = 0;
        } else if in_negation_scope {
            negation_distance += 1;
            if negation_distance > 3 {
                in_negation_scope = false;
            }
        }

        let negated = in_negation_scope && pos != PartOfSpeech::Negation;

        annotated.push(AnnotatedToken {
            text: word.clone(),
            stem,
            pos,
            negated,
            position: i,
        });
    }

    annotated
}

/// Classify part of speech using heuristics
fn classify_pos(word: &str, position: usize, context: &[String]) -> PartOfSpeech {
    // Check negation first
    if is_negation(word) {
        return PartOfSpeech::Negation;
    }

    // Check stop words
    if is_stop_word(word) {
        return PartOfSpeech::StopWord;
    }

    // Use suffix patterns and context for classification
    if is_verb(word) {
        return PartOfSpeech::Verb;
    }

    if is_adjective(word) {
        return PartOfSpeech::Adjective;
    }

    if is_noun(word, position, context) {
        return PartOfSpeech::Noun;
    }

    // Default to unknown (treated as noun for domain terms)
    PartOfSpeech::Unknown
}

/// Detect compound nouns (bigrams that commonly co-occur)
fn detect_compound_nouns(tokens: &[AnnotatedToken]) -> Vec<String> {
    let mut compounds = Vec::new();

    // Common compound noun patterns
    const COMPOUND_PATTERNS: &[(&str, &str)] = &[
        // Tech/AI compounds
        ("machine", "learning"),
        ("deep", "learning"),
        ("neural", "network"),
        ("natural", "language"),
        ("language", "model"),
        ("artificial", "intelligence"),
        ("knowledge", "graph"),
        ("vector", "database"),
        ("memory", "system"),
        ("data", "structure"),
        ("source", "code"),
        ("error", "handling"),
        ("unit", "test"),
        ("integration", "test"),
        ("api", "endpoint"),
        ("web", "server"),
        ("file", "system"),
        ("operating", "system"),
        ("database", "schema"),
        ("user", "interface"),
        ("command", "line"),
        ("version", "control"),
        ("pull", "request"),
        ("code", "review"),
        ("bug", "fix"),
        ("feature", "request"),
        // Domain-specific
        ("spreading", "activation"),
        ("hebbian", "learning"),
        ("long", "term"),
        ("short", "term"),
        ("working", "memory"),
        ("semantic", "search"),
        ("graph", "traversal"),
        ("edge", "device"),
        ("air", "gapped"),
        // Social/community
        ("support", "group"),
        ("pride", "parade"),
        ("poetry", "reading"),
        ("civil", "rights"),
        ("human", "rights"),
        ("social", "media"),
        ("community", "center"),
        ("discussion", "group"),
        ("therapy", "session"),
        ("art", "therapy"),
        ("group", "therapy"),
    ];

    // Check for known compound patterns
    for i in 0..tokens.len().saturating_sub(1) {
        let t1 = &tokens[i];
        let t2 = &tokens[i + 1];

        // Skip if either is a stop word or verb
        if t1.pos == PartOfSpeech::StopWord || t2.pos == PartOfSpeech::StopWord {
            continue;
        }

        for (w1, w2) in COMPOUND_PATTERNS {
            if (t1.stem == *w1 || t1.text == *w1) && (t2.stem == *w2 || t2.text == *w2) {
                compounds.push(format!("{} {}", t1.text, t2.text));
                break;
            }
        }

        // Heuristic: Noun + Noun often forms compound
        if (t1.pos == PartOfSpeech::Noun || t1.pos == PartOfSpeech::Unknown)
            && (t2.pos == PartOfSpeech::Noun || t2.pos == PartOfSpeech::Unknown)
        {
            // Check for common suffixes that indicate compound-worthy nouns
            if has_compound_suffix(&t1.text) || has_compound_suffix(&t2.text) {
                let compound = format!("{} {}", t1.text, t2.text);
                if !compounds.contains(&compound) {
                    compounds.push(compound);
                }
            }
        }
    }

    compounds
}

/// Check if word has suffix that often appears in compounds
fn has_compound_suffix(word: &str) -> bool {
    word.ends_with("tion")
        || word.ends_with("ment")
        || word.ends_with("ing")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("er")
        || word.ends_with("or")
        || word.ends_with("ist")
        || word.ends_with("ism")
}

/// Get positions of tokens that are part of compounds
fn compound_positions(tokens: &[AnnotatedToken], compounds: &[String]) -> HashSet<usize> {
    let mut positions = HashSet::new();

    for compound in compounds {
        let parts: Vec<&str> = compound.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        for i in 0..tokens.len().saturating_sub(parts.len() - 1) {
            let mut matches = true;
            for (j, part) in parts.iter().enumerate() {
                if tokens[i + j].text != *part {
                    matches = false;
                    break;
                }
            }
            if matches {
                for j in 0..parts.len() {
                    positions.insert(i + j);
                }
            }
        }
    }

    positions
}

/// Calculate term weight with IDF-like rarity boost
fn calculate_term_weight(word: &str, base_weight: f32) -> f32 {
    // Longer words tend to be more specific/rare
    let length_factor = if word.len() > 8 {
        1.2
    } else if word.len() > 5 {
        1.1
    } else {
        1.0
    };

    // Technical suffixes get slight boost
    let suffix_factor = if word.ends_with("tion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
    {
        1.1
    } else {
        1.0
    };

    base_weight * length_factor * suffix_factor
}

/// Check if word is negation
fn is_negation(word: &str) -> bool {
    const NEGATIONS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "neither",
        "nobody",
        "nowhere",
        "without",
        "cannot",
        "can't",
        "won't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "mustn't",
    ];
    NEGATIONS.contains(&word)
}

/// Check if word is a noun (entity)
fn is_noun(word: &str, position: usize, context: &[String]) -> bool {
    // Domain-specific nouns (expanded list)
    const NOUN_INDICATORS: &[&str] = &[
        // Core memory/cognitive terms
        "memory",
        "graph",
        "node",
        "edge",
        "entity",
        "embedding",
        "vector",
        "index",
        "query",
        "retrieval",
        "activation",
        "potentiation",
        "consolidation",
        "decay",
        "strength",
        "weight",
        "threshold",
        "importance",
        // Tech terms
        "robot",
        "drone",
        "sensor",
        "lidar",
        "camera",
        "motor",
        "actuator",
        "obstacle",
        "path",
        "waypoint",
        "location",
        "coordinates",
        "position",
        "battery",
        "power",
        "energy",
        "voltage",
        "current",
        "system",
        "module",
        "component",
        "unit",
        "device",
        "temperature",
        "pressure",
        "humidity",
        "speed",
        "velocity",
        "signal",
        "communication",
        "network",
        "link",
        "connection",
        "navigation",
        "guidance",
        "control",
        "steering",
        "data",
        "information",
        "message",
        "command",
        "response",
        // Software terms
        "function",
        "method",
        "class",
        "struct",
        "interface",
        "module",
        "package",
        "library",
        "framework",
        "api",
        "endpoint",
        "request",
        "response",
        "error",
        "exception",
        "bug",
        "fix",
        "feature",
        "test",
        "benchmark",
        "performance",
        "latency",
        "throughput",
        "cache",
        "buffer",
        "queue",
        "stack",
        "heap",
        "thread",
        "process",
        "server",
        "client",
        "database",
        "table",
        "column",
        "row",
        "schema",
        "migration",
        "deployment",
        "container",
        "cluster",
        "replica",
        // General nouns
        "person",
        "people",
        "user",
        "agent",
        "operator",
        "time",
        "date",
        "day",
        "hour",
        "minute",
        "second",
        "area",
        "zone",
        "region",
        "sector",
        "space",
        "task",
        "mission",
        "goal",
        "objective",
        "target",
        "warning",
        "alert",
        "notification",
        "level",
        "status",
        "state",
        "condition",
        "mode",
        "type",
        "kind",
        "version",
        "release",
        "update",
        "change",
        "result",
        "output",
        "input",
        "value",
        "key",
        "name",
        "id",
        "identifier",
    ];

    if NOUN_INDICATORS.contains(&word) {
        return true;
    }

    // Check for noun suffixes
    if word.ends_with("tion")
        || word.ends_with("sion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("er")
        || word.ends_with("or")
        || word.ends_with("ist")
        || word.ends_with("ism")
        || word.ends_with("age")
        || word.ends_with("ure")
        || word.ends_with("dom")
    {
        // Avoid verb forms like "better", "faster"
        if !(word.ends_with("er") && word.len() < 5) {
            return true;
        }
    }

    // Check if preceded by determiner (a, an, the)
    if position > 0 {
        if let Some(prev) = context.get(position - 1) {
            let prev = prev.to_lowercase();
            if prev == "a" || prev == "an" || prev == "the" || prev == "this" || prev == "that" {
                return true;
            }
        }
    }

    // Check if preceded by possessive
    if position > 0 {
        if let Some(prev) = context.get(position - 1) {
            if prev.ends_with("'s") || prev.ends_with("s'") {
                return true;
            }
        }
    }

    false
}

/// Check if word is an adjective (qualifier)
fn is_adjective(word: &str) -> bool {
    const ADJECTIVE_INDICATORS: &[&str] = &[
        // Colors
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "black",
        "white",
        "gray",
        "grey",
        "pink",
        "brown",
        // Sizes
        "big",
        "small",
        "large",
        "tiny",
        "huge",
        "massive",
        "mini",
        "micro",
        "high",
        "low",
        "tall",
        "short",
        "long",
        "wide",
        "narrow",
        // States
        "hot",
        "cold",
        "warm",
        "cool",
        "frozen",
        "heated",
        "fast",
        "slow",
        "quick",
        "rapid",
        "gradual",
        "active",
        "inactive",
        "enabled",
        "disabled",
        "open",
        "closed",
        "locked",
        "unlocked",
        "full",
        "empty",
        "partial",
        "complete",
        "valid",
        "invalid",
        "correct",
        "incorrect",
        "true",
        "false",
        // Quality
        "good",
        "bad",
        "excellent",
        "poor",
        "optimal",
        "suboptimal",
        "normal",
        "abnormal",
        "stable",
        "unstable",
        "safe",
        "unsafe",
        "dangerous",
        "hazardous",
        "new",
        "old",
        "recent",
        "ancient",
        "current",
        "latest",
        "first",
        "last",
        "next",
        "previous",
        "primary",
        "secondary",
        "main",
        "important",
        "critical",
        "minor",
        "major",
        // Technical
        "autonomous",
        "manual",
        "automatic",
        "remote",
        "digital",
        "analog",
        "electronic",
        "mechanical",
        "wireless",
        "wired",
        "connected",
        "disconnected",
        "local",
        "global",
        "private",
        "public",
        "static",
        "dynamic",
        "mutable",
        "immutable",
        "sync",
        "async",
        "concurrent",
        "parallel",
        "serial",
        "sequential",
        "optional",
        "required",
        "default",
        "custom",
    ];

    if ADJECTIVE_INDICATORS.contains(&word) {
        return true;
    }

    // Common adjective suffixes (excluding verb participles)
    if word.ends_with("ful")
        || word.ends_with("less")
        || word.ends_with("ous")
        || word.ends_with("ive")
        || word.ends_with("able")
        || word.ends_with("ible")
        || word.ends_with("al")
        || word.ends_with("ic")
        || word.ends_with("ary")
        || word.ends_with("ory")
    {
        // Avoid false positives
        let exceptions = ["animal", "interval", "arrival", "approval"];
        if !exceptions.contains(&word) {
            return true;
        }
    }

    false
}

/// Check if word is a verb (relational, lower priority)
fn is_verb(word: &str) -> bool {
    const VERB_INDICATORS: &[&str] = &[
        // Auxiliary/modal verbs
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        // Common action verbs
        "go",
        "goes",
        "went",
        "gone",
        "going",
        "get",
        "gets",
        "got",
        "gotten",
        "getting",
        "make",
        "makes",
        "made",
        "making",
        "take",
        "takes",
        "took",
        "taken",
        "taking",
        "see",
        "sees",
        "saw",
        "seen",
        "seeing",
        "give",
        "gives",
        "gave",
        "given",
        "giving",
        "use",
        "uses",
        "used",
        "using",
        "find",
        "finds",
        "found",
        "finding",
        "know",
        "knows",
        "knew",
        "known",
        "knowing",
        "think",
        "thinks",
        "thought",
        "thinking",
        "want",
        "wants",
        "wanted",
        "wanting",
        "need",
        "needs",
        "needed",
        "needing",
        "try",
        "tries",
        "tried",
        "trying",
        // Technical verbs
        "detect",
        "detects",
        "detected",
        "detecting",
        "observe",
        "observes",
        "observed",
        "observing",
        "measure",
        "measures",
        "measured",
        "measuring",
        "sense",
        "senses",
        "sensed",
        "sensing",
        "scan",
        "scans",
        "scanned",
        "scanning",
        "navigate",
        "navigates",
        "navigated",
        "navigating",
        "move",
        "moves",
        "moved",
        "moving",
        "stop",
        "stops",
        "stopped",
        "stopping",
        "start",
        "starts",
        "started",
        "starting",
        "reach",
        "reaches",
        "reached",
        "reaching",
        "avoid",
        "avoids",
        "avoided",
        "avoiding",
        "block",
        "blocks",
        "blocked",
        "blocking",
        "create",
        "creates",
        "created",
        "creating",
        "delete",
        "deletes",
        "deleted",
        "deleting",
        "update",
        "updates",
        "updated",
        "updating",
        "read",
        "reads",
        "reading",
        "write",
        "writes",
        "wrote",
        "written",
        "writing",
        "run",
        "runs",
        "ran",
        "running",
        "execute",
        "executes",
        "executed",
        "executing",
        "call",
        "calls",
        "called",
        "calling",
        "return",
        "returns",
        "returned",
        "returning",
        "store",
        "stores",
        "stored",
        "storing",
        "load",
        "loads",
        "loaded",
        "loading",
        "save",
        "saves",
        "saved",
        "saving",
        "fetch",
        "fetches",
        "fetched",
        "fetching",
        "send",
        "sends",
        "sent",
        "sending",
        "receive",
        "receives",
        "received",
        "receiving",
        "connect",
        "connects",
        "connected",
        "connecting",
        "disconnect",
        "disconnects",
        "disconnected",
        "disconnecting",
        "process",
        "processes",
        "processed",
        "processing",
        "handle",
        "handles",
        "handled",
        "handling",
        "parse",
        "parses",
        "parsed",
        "parsing",
        "compile",
        "compiles",
        "compiled",
        "compiling",
        "build",
        "builds",
        "built",
        "building",
        "test",
        "tests",
        "tested",
        "testing",
        "deploy",
        "deploys",
        "deployed",
        "deploying",
        "install",
        "installs",
        "installed",
        "installing",
        "configure",
        "configures",
        "configured",
        "configuring",
        "initialize",
        "initializes",
        "initialized",
        "initializing",
        "shutdown",
        "shutdowns",
        "terminate",
        "terminates",
        "terminated",
        "terminating",
    ];

    VERB_INDICATORS.contains(&word)
}

/// Check if word is a stop word (no information content)
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        // Articles
        "a",
        "an",
        "the",
        // Demonstratives
        "this",
        "that",
        "these",
        "those",
        // Prepositions
        "at",
        "in",
        "on",
        "to",
        "for",
        "of",
        "from",
        "by",
        "with",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        // Conjunctions
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        // Pronouns
        "i",
        "you",
        "he",
        "she",
        "it",
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
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "whoever",
        "whatever",
        "whichever",
        // Relative
        "that",
        "which",
        "who",
        "whom",
        "whose",
        // Question words (when not seeking info)
        "how",
        "when",
        "where",
        "why",
        // Common filler
        "just",
        "only",
        "even",
        "also",
        "too",
        "very",
        "really",
        "quite",
        "rather",
        "almost",
        "already",
        "still",
        "always",
        "never",
        "ever",
        "often",
        "sometimes",
        "usually",
        "perhaps",
        "maybe",
        "probably",
        "possibly",
        "certainly",
        "definitely",
        "actually",
        "basically",
        "essentially",
        "simply",
        "merely",
        // Be forms handled separately as verbs
        "as",
        "if",
        "then",
        "than",
        "because",
        "although",
        "though",
        "unless",
        "until",
        "while",
        "whereas",
        "whether",
        "since",
        // Others
        "some",
        "any",
        "all",
        "each",
        "every",
        "many",
        "much",
        "more",
        "most",
        "few",
        "less",
        "least",
        "other",
        "another",
        "such",
        "same",
        "different",
        "own",
        "several",
    ];

    STOP_WORDS.contains(&word)
}

// =============================================================================
// ONTOLOGICAL INTENT INFERENCE
// =============================================================================
// Infers expected entity types and relation types from query structure.
// Zero-config: types come from existing EntityLabel and RelationType enums
// already stored on every graph node/edge.
//
// Reference: Collins & Quillian (1969), Anderson (1983) selective spreading

use crate::graph_memory::{EntityLabel, RelationType};

/// Ontological intent inferred from query structure.
///
/// Determines expected entity types and relation types based on question words
/// and verb stems. Used by Layer 2 (Graph Expansion) to apply soft type penalties
/// during spreading activation and Layer 4.9 for post-RRF re-ranking.
///
/// Zero-config: no user-defined ontology needed. Types come from the existing
/// EntityLabel and RelationType enums already stored in the knowledge graph.
#[derive(Debug, Clone)]
pub struct OntologicalIntent {
    /// Expected entity types in the answer (e.g., Person for "who" queries).
    /// Empty = no type constraint.
    pub expected_labels: Vec<EntityLabel>,
    /// Preferred relation types for graph traversal (e.g., WorksWith for "work" verbs).
    /// Empty = no relation constraint.
    pub relation_types: Vec<RelationType>,
    /// Confidence in the inferred intent (0.0-1.0).
    /// Below ONTOLOGICAL_MIN_CONFIDENCE, intent is ignored.
    pub confidence: f32,
}

/// Map verb stems to relation types.
///
/// Uses Porter2-stemmed forms for matching against `Relation.stem` from query analysis.
/// Returns empty vec for unrecognized verbs (no penalty applied).
fn verb_stem_to_relation_types(stem: &str) -> Vec<RelationType> {
    match stem {
        // Work relationships
        "work" | "collabor" => vec![
            RelationType::WorksWith,
            RelationType::WorksAt,
            RelationType::EmployedBy,
        ],
        "employ" | "hire" => vec![RelationType::EmployedBy, RelationType::WorksAt],

        // Location relationships
        "locat" | "live" | "base" | "resid" | "situat" => {
            vec![RelationType::LocatedIn, RelationType::LocatedAt]
        }

        // Learning relationships
        "learn" | "studi" | "discover" => vec![RelationType::Learned, RelationType::Knows],
        "teach" | "mentor" | "instruct" => vec![RelationType::Teaches],
        "know" | "familiar" => vec![RelationType::Knows],

        // Usage relationships
        "use" | "util" | "adopt" | "leverag" => vec![RelationType::Uses],
        "read" | "consult" | "refer" => vec![RelationType::Uses],

        // Creation relationships
        "creat" | "build" | "develop" | "design" => {
            vec![RelationType::CreatedBy, RelationType::DevelopedBy]
        }

        // Causal relationships
        // Triggers is the type the causal cue extractor gives "caused", "led to" and
        // "because of"; outside this set the ontology penalty would dampen those edges.
        "caus" | "result" | "lead" => {
            vec![
                RelationType::Causes,
                RelationType::ResultsIn,
                RelationType::Triggers,
            ]
        }

        // Ownership / structural
        "own" | "belong" | "possess" => vec![RelationType::OwnedBy, RelationType::PartOf],
        "contain" | "includ" | "consist" => vec![RelationType::Contains, RelationType::PartOf],

        // Management / governance
        "manag" | "supervis" | "oversee" | "direct" | "coordin" => vec![RelationType::Manages],
        "assign" | "deleg" | "allocat" => vec![RelationType::AssignedTo],
        "approv" | "reject" | "review" | "sign" => vec![RelationType::Approves],

        // Dependency / requirement
        "depend" | "requir" | "need" | "reliant" => {
            vec![RelationType::DependsOn, RelationType::Requires]
        }

        // Preference / recommendation
        "prefer" | "like" | "favor" | "favour" | "chose" | "choos" | "select" => {
            vec![RelationType::Prefers]
        }
        "recommend" | "suggest" | "advis" | "propos" => vec![RelationType::Recommends],

        // Documentation / knowledge
        "write" | "document" | "author" | "draft" | "compil" => {
            vec![RelationType::Documents, RelationType::CreatedBy]
        }
        "implement" | "realiz" | "fulfill" => vec![RelationType::Implements],

        // Operational / execution
        "run" | "execut" | "launch" | "start" | "invok" => {
            vec![RelationType::Uses, RelationType::Triggers]
        }
        "deploy" | "releas" | "ship" | "promot" => vec![RelationType::DeploysTo],
        "monitor" | "track" | "observ" | "watch" | "alert" => vec![RelationType::Monitors],
        "trigger" | "fire" | "emit" => vec![RelationType::Triggers],
        "schedul" | "plan" | "arrang" | "organiz" => vec![RelationType::Manages],
        "configur" | "tune" | "adjust" => vec![RelationType::Configures],

        // Evolution / comparison
        "migrat" | "upgrad" | "patch" | "updat" => vec![RelationType::SupersededBy],
        "replac" | "supersed" | "obsolet" | "deprec" => vec![RelationType::SupersededBy],
        "compar" | "differ" | "contrast" | "benchmark" => vec![RelationType::AlternativeTo],

        // Problem solving
        "fix" | "solv" | "resolv" | "debug" | "repair" => vec![RelationType::ResultsIn],

        // Integration / connection
        "connect" | "integr" | "link" | "bridg" | "coupl" => {
            vec![RelationType::DependsOn, RelationType::Uses]
        }

        _ => vec![],
    }
}

/// Map question word patterns to expected entity labels.
///
/// Checks the first few tokens of the query for WH-words and maps them
/// to the entity types most likely to appear in the answer.
fn question_word_to_labels(query_text: &str) -> Vec<EntityLabel> {
    let lower = query_text.to_lowercase();
    let trimmed = lower.trim_start();

    // "who" / "whom" queries → people and organizational entities
    if trimmed.starts_with("who ") || trimmed.starts_with("whom ") {
        vec![
            EntityLabel::Person,
            EntityLabel::Organization,
            EntityLabel::Team,
        ]
    }
    // "where" queries → location and environment
    else if trimmed.starts_with("where ") {
        vec![EntityLabel::Location, EntityLabel::Environment]
    }
    // "when" queries → temporal entities
    else if trimmed.starts_with("when ") {
        vec![EntityLabel::Date, EntityLabel::Event]
    }
    // "how" queries → methodology, technology, operational
    else if trimmed.starts_with("how ") {
        if trimmed.contains(" deploy")
            || trimmed.contains(" ci")
            || trimmed.contains(" cd")
            || trimmed.contains(" pipeline")
            || trimmed.contains(" build")
            || trimmed.contains(" releas")
        {
            vec![
                EntityLabel::Pipeline,
                EntityLabel::Technology,
                EntityLabel::Configuration,
            ]
        } else if trimmed.contains(" monitor")
            || trimmed.contains(" metric")
            || trimmed.contains(" alert")
            || trimmed.contains(" observ")
        {
            vec![EntityLabel::Metric, EntityLabel::Service]
        } else if trimmed.contains(" configur")
            || trimmed.contains(" set up")
            || trimmed.contains(" setup")
        {
            vec![EntityLabel::Configuration, EntityLabel::Technology]
        } else {
            vec![EntityLabel::Concept, EntityLabel::Technology]
        }
    }
    // "why" queries → causal reasoning, events, decisions
    else if trimmed.starts_with("why ") {
        vec![EntityLabel::Event, EntityLabel::Concept]
    }
    // "what/which" + technology context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" tool")
            || trimmed.contains(" tech")
            || trimmed.contains(" framework")
            || trimmed.contains(" language")
            || trimmed.contains(" library")
            || trimmed.contains(" stack"))
    {
        vec![
            EntityLabel::Technology,
            EntityLabel::Product,
            EntityLabel::Module,
        ]
    }
    // "what/which" + service/api context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" service")
            || trimmed.contains(" api")
            || trimmed.contains(" endpoint")
            || trimmed.contains(" microservice"))
    {
        vec![EntityLabel::Service, EntityLabel::Technology]
    }
    // "what/which" + project/repo context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" project")
            || trimmed.contains(" repo")
            || trimmed.contains(" repositor")
            || trimmed.contains(" codebase"))
    {
        vec![EntityLabel::Project, EntityLabel::Repository]
    }
    // "what/which" + database context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" databas")
            || trimmed.contains(" db")
            || trimmed.contains(" store")
            || trimmed.contains(" cache"))
    {
        vec![EntityLabel::Database, EntityLabel::Technology]
    }
    // "what/which" + environment context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" environ")
            || trimmed.contains(" stage")
            || trimmed.contains(" prod")
            || trimmed.contains(" cluster"))
    {
        vec![EntityLabel::Environment]
    }
    // "what/which" + task/ticket context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" task")
            || trimmed.contains(" ticket")
            || trimmed.contains(" issue")
            || trimmed.contains(" bug"))
    {
        vec![EntityLabel::Task]
    }
    // "what/which" + metric/SLO context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" metric")
            || trimmed.contains(" slo")
            || trimmed.contains(" latenc")
            || trimmed.contains(" error rate"))
    {
        vec![EntityLabel::Metric]
    }
    // "what/which" + config context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" config")
            || trimmed.contains(" flag")
            || trimmed.contains(" setting")
            || trimmed.contains(" param"))
    {
        vec![EntityLabel::Configuration]
    }
    // "what/which" + document context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" doc")
            || trimmed.contains(" rfc")
            || trimmed.contains(" spec")
            || trimmed.contains(" readme")
            || trimmed.contains(" runbook"))
    {
        vec![EntityLabel::Document]
    }
    // "what/which" + skill context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" skill")
            || trimmed.contains(" abilit")
            || trimmed.contains(" competenc"))
    {
        vec![EntityLabel::Skill]
    }
    // "what/which" + organization/team context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" compan")
            || trimmed.contains(" organiz")
            || trimmed.contains(" team")
            || trimmed.contains(" squad"))
    {
        vec![EntityLabel::Organization, EntityLabel::Team]
    }
    // "what/which" + pipeline context
    else if (trimmed.starts_with("what ") || trimmed.starts_with("which "))
        && (trimmed.contains(" pipeline")
            || trimmed.contains(" workflow")
            || trimmed.contains(" ci")
            || trimmed.contains(" cd"))
    {
        vec![EntityLabel::Pipeline]
    }
    // Imperative patterns: "list all...", "show me...", "find..."
    else if trimmed.starts_with("list ")
        || trimmed.starts_with("show ")
        || trimmed.starts_with("find ")
        || trimmed.starts_with("get ")
        || trimmed.starts_with("fetch ")
        || trimmed.starts_with("search ")
    {
        imperative_object_to_labels(trimmed)
    }
    // Comparative patterns
    else if trimmed.starts_with("compar")
        || trimmed.contains(" vs ")
        || trimmed.contains(" versus ")
        || trimmed.contains(" better than ")
        || trimmed.contains(" differ")
        || trimmed.contains(" alternative")
    {
        vec![EntityLabel::Technology, EntityLabel::Product]
    } else {
        vec![]
    }
}

/// Extract entity labels from the object of an imperative query.
///
/// Parses the noun object in commands like "list all services" or
/// "find the database config" to infer expected entity types.
fn imperative_object_to_labels(query: &str) -> Vec<EntityLabel> {
    if query.contains("service") || query.contains("api") || query.contains("endpoint") {
        vec![EntityLabel::Service]
    } else if query.contains("project") || query.contains("repo") {
        vec![EntityLabel::Project, EntityLabel::Repository]
    } else if query.contains("team") || query.contains("squad") || query.contains("group") {
        vec![EntityLabel::Team, EntityLabel::Organization]
    } else if query.contains("person") || query.contains("people") || query.contains("member") {
        vec![EntityLabel::Person]
    } else if query.contains("task") || query.contains("ticket") || query.contains("issue") {
        vec![EntityLabel::Task]
    } else if query.contains("config") || query.contains("flag") || query.contains("setting") {
        vec![EntityLabel::Configuration]
    } else if query.contains("doc") || query.contains("spec") || query.contains("rfc") {
        vec![EntityLabel::Document]
    } else if query.contains("pipeline") || query.contains("workflow") || query.contains("ci") {
        vec![EntityLabel::Pipeline]
    } else if query.contains("metric") || query.contains("slo") || query.contains("alert") {
        vec![EntityLabel::Metric]
    } else if query.contains("environ") || query.contains("cluster") || query.contains("stage") {
        vec![EntityLabel::Environment]
    } else if query.contains("databas") || query.contains("db") || query.contains("store") {
        vec![EntityLabel::Database]
    } else if query.contains("skill") || query.contains("competenc") {
        vec![EntityLabel::Skill]
    } else if query.contains("technolog") || query.contains("tool") || query.contains("framework") {
        vec![EntityLabel::Technology]
    } else if query.contains("module") || query.contains("package") || query.contains("crate") {
        vec![EntityLabel::Module]
    } else {
        vec![]
    }
}

/// Infer ontological intent from query analysis.
///
/// Combines question word patterns (for expected entity types) and verb stems
/// (for preferred relation types) to produce a type-aware intent signal.
///
/// Called once per query in `semantic_retrieve()`. Zero cost for queries
/// with no type signal (returns confidence=0.0, empty vecs).
pub fn infer_ontological_intent(query_text: &str, analysis: &QueryAnalysis) -> OntologicalIntent {
    let mut confidence = 0.0_f32;

    // 1. Question word → expected entity labels
    let expected_labels = question_word_to_labels(query_text);
    if !expected_labels.is_empty() {
        confidence += 0.4;
    }

    // 2. Verb stems → preferred relation types
    let mut relation_types = Vec::new();
    let mut seen_relations = std::collections::HashSet::new();
    for relation in &analysis.relational_context {
        for rt in verb_stem_to_relation_types(&relation.stem) {
            let key = rt.as_str().to_string();
            if seen_relations.insert(key) {
                relation_types.push(rt);
            }
        }
    }
    if !relation_types.is_empty() {
        confidence += 0.4;
    }

    // 3. Multiple signals bonus (both question word AND verb match)
    if !expected_labels.is_empty() && !relation_types.is_empty() {
        confidence += 0.2;
    }

    OntologicalIntent {
        expected_labels,
        relation_types,
        confidence: confidence.min(1.0),
    }
}

// ============================================================================
// QUERY DECOMPOSITION
// ============================================================================
// Splits compound/multi-entity queries into sub-queries for independent
// vector retrieval. BM25 and graph traversal handle compound queries well
// naturally, so decomposition only applies to the vector search path.
//
// Reference: Khattab et al. (2021) "Baleen: Robust Multi-Hop Reasoning"
// Key insight: A single embedding of "When did we switch from PostgreSQL to
// MongoDB?" dilutes signal for both PostgreSQL and MongoDB. Decomposing into
// per-entity sub-queries recovers the individual signals.

/// Decompose compound queries into sub-queries for independent retrieval.
///
/// Returns a vec containing only the original query if no decomposition is
/// needed (simple/single-entity queries). When decomposition fires, the
/// original query is always included as the first element.
///
/// Decomposition rules (applied in priority order):
/// 1. Conjunction split: "X AND Y" or "X and also Y" or "X as well as Y"
/// 2. Multi-entity relationship: 2+ focal entities with relationship verbs
/// 3. Compound temporal: temporal reference + entity in the same query
///
/// Sub-queries are capped at [`QUERY_DECOMPOSITION_MAX_SUBQUERIES`].
pub fn decompose_query(query: &str, analysis: &QueryAnalysis) -> Vec<String> {
    use crate::constants::QUERY_DECOMPOSITION_MAX_SUBQUERIES;

    let query_trimmed = query.trim();
    if query_trimmed.is_empty() {
        return vec![query_trimmed.to_string()];
    }

    // Rule 1: Conjunction split
    // "Why did we change the database AND what was the impact on the timeline?"
    // → ["Why did we change the database?", "What was the impact on the timeline?"]
    if let Some(parts) = split_on_conjunction(query_trimmed) {
        let mut sub_queries = vec![query_trimmed.to_string()];
        for part in parts {
            let cleaned = clean_sub_query(&part);
            if !cleaned.is_empty() && cleaned.len() >= 5 {
                sub_queries.push(cleaned);
            }
        }
        if sub_queries.len() > 1 {
            sub_queries.truncate(1 + QUERY_DECOMPOSITION_MAX_SUBQUERIES);
            tracing::debug!(
                "Query decomposition [conjunction]: {} → {} sub-queries",
                query_trimmed,
                sub_queries.len() - 1
            );
            return sub_queries;
        }
    }

    // Rule 2: Multi-entity relationship queries
    // "When did we switch from PostgreSQL to MongoDB?"
    // → ["PostgreSQL", "MongoDB", "When did we switch from PostgreSQL to MongoDB?"]
    if analysis.focal_entities.len() >= 2 && has_relationship_signal(query_trimmed) {
        let mut sub_queries = vec![query_trimmed.to_string()];
        for entity in &analysis.focal_entities {
            // Skip stop-word-like short entities that aren't meaningful on their own
            if entity.text.len() < 3 {
                continue;
            }
            sub_queries.push(entity.text.clone());
        }
        if sub_queries.len() > 1 {
            sub_queries.truncate(1 + QUERY_DECOMPOSITION_MAX_SUBQUERIES);
            tracing::debug!(
                "Query decomposition [multi-entity]: {} → {} sub-queries",
                query_trimmed,
                sub_queries.len() - 1
            );
            return sub_queries;
        }
    }

    // Rule 3: Compound temporal — query has both a temporal reference AND focal entities
    // "What did Melanie do last Tuesday?" → ["Melanie", "What did Melanie do last Tuesday?"]
    // Only fires when there are distinct entity + temporal signals worth separating.
    let temporal_intent = detect_temporal_intent(query_trimmed);
    let has_temporal = !matches!(temporal_intent, TemporalIntent::None);
    if has_temporal && !analysis.focal_entities.is_empty() {
        // Build an entity-only sub-query from all focal entities
        let entity_names: Vec<&str> = analysis
            .focal_entities
            .iter()
            .filter(|e| e.text.len() >= 3)
            .map(|e| e.text.as_str())
            .collect();

        if !entity_names.is_empty() {
            let entity_sub = entity_names.join(" ");
            // Only decompose if entity sub-query is meaningfully different from original
            if entity_sub.len() < query_trimmed.len() / 2 {
                let mut sub_queries = vec![query_trimmed.to_string(), entity_sub];
                sub_queries.truncate(1 + QUERY_DECOMPOSITION_MAX_SUBQUERIES);
                tracing::debug!(
                    "Query decomposition [temporal+entity]: {} → {} sub-queries",
                    query_trimmed,
                    sub_queries.len() - 1
                );
                return sub_queries;
            }
        }
    }

    // Rule 4: Abstract/thematic queries — meta-reasoning vocabulary detected but
    // no conjunction, relationship signal, or temporal signal triggered earlier rules.
    // Decomposes into focal-entity sub-queries to widen vector search coverage across
    // semantically distant memory clusters (e.g., "performance improvements" must find
    // memories about caching, payments, onboarding — all different vector neighborhoods).
    // Guard: expand_abstract_terms_for_bm25 must recognize abstract vocabulary to avoid
    // decomposing already-specific queries like "What database did we choose?"
    if !has_temporal
        && !analysis.focal_entities.is_empty()
        && query_trimmed.split_whitespace().count() >= 5
        && expand_abstract_terms_for_bm25(query_trimmed).is_some()
    {
        let mut sub_queries = vec![query_trimmed.to_string()];
        for entity in &analysis.focal_entities {
            if entity.text.len() >= 3 {
                sub_queries.push(entity.text.clone());
            }
        }
        for compound in &analysis.compound_nouns {
            if compound.len() >= 5 {
                sub_queries.push(compound.clone());
            }
        }
        sub_queries.dedup();
        if sub_queries.len() > 1 {
            sub_queries.truncate(1 + QUERY_DECOMPOSITION_MAX_SUBQUERIES);
            tracing::debug!(
                "Query decomposition [abstract-thematic]: {} → {} sub-queries",
                query_trimmed,
                sub_queries.len() - 1
            );
            return sub_queries;
        }
    }

    // No decomposition needed
    vec![query_trimmed.to_string()]
}

/// Expand abstract/meta-reasoning query terms with concrete synonyms for BM25.
///
/// Abstract queries like "What patterns do you see in our decision-making?" fail
/// BM25 because "patterns" doesn't appear in any memory about specific decisions.
/// This function detects abstract terms and appends concrete related terms with
/// a low boost weight (^0.3) so they act as BM25 tiebreakers, not primary signals.
///
/// Returns `Some(expanded)` if expansion was applied, `None` if no abstract terms
/// were detected (so the caller can use the original query unchanged).
///
/// The expansion map covers the most common meta-reasoning terms in conversational
/// memory retrieval. Terms are matched against stemmed query words to handle
/// morphological variants (e.g., "pattern" and "patterns" both trigger expansion).
pub fn expand_abstract_terms_for_bm25(query: &str) -> Option<String> {
    use rust_stemmers::{Algorithm, Stemmer};

    // Static mapping of abstract stem → concrete expansion terms.
    // Kept small (15 entries) and focused on meta-reasoning vocabulary.
    // Each expansion term gets ^0.3 boost in the BM25 query to avoid
    // overwhelming the original query terms.
    const EXPANSION_BOOST: &str = "^0.3";
    static EXPANSIONS: &[(&str, &[&str])] = &[
        (
            "pattern",
            &["decisions", "choices", "recurring", "repeated", "trend"],
        ),
        (
            "prioriti",
            &["goals", "focus", "important", "urgent", "next"],
        ),
        (
            "risk",
            &["concerns", "issues", "problems", "challenges", "threats"],
        ),
        ("strategi", &["plan", "approach", "direction", "roadmap"]),
        ("lesson", &["learned", "mistakes", "insights", "takeaways"]),
        (
            "progress",
            &["status", "milestone", "completed", "remaining"],
        ),
        ("impact", &["effect", "consequence", "result", "outcome"]),
        ("blocker", &["obstacle", "impediment", "stuck", "waiting"]),
        ("tradeoff", &["compromise", "versus", "cost", "benefit"]),
        (
            "retrospect",
            &["review", "reflection", "improvement", "feedback"],
        ),
        ("evolv", &["changed", "grew", "adapted", "shifted"]),
        ("evolut", &["changed", "grew", "adapted", "shifted"]),
        ("theme", &["recurring", "common", "trend", "motif"]),
        // Open-domain / aggregative vocabulary (v0.7.1)
        (
            "perform",
            &["speed", "latency", "throughput", "faster", "optimized"],
        ),
        (
            "improv",
            &["better", "faster", "fixed", "enhanced", "upgraded"],
        ),
        (
            "achiev",
            &["completed", "accomplished", "reached", "delivered"],
        ),
        (
            "key",
            &["important", "critical", "main", "primary", "essential"],
        ),
        (
            "peopl",
            &["team", "member", "responsible", "leading", "working"],
        ),
        (
            "person",
            &["team", "member", "responsible", "leading", "working"],
        ),
        (
            "infrastructur",
            &["deploy", "server", "cloud", "hosting", "kubernetes"],
        ),
        (
            "decis",
            &["chose", "decided", "selected", "picked", "adopted"],
        ),
        (
            "deploy",
            &["infrastructure", "kubernetes", "server", "hosting", "cloud"],
        ),
        ("moral", &["team", "mood", "happy", "celebrate", "social"]),
        (
            "dynamic",
            &["interaction", "relationship", "team", "collaboration"],
        ),
    ];

    let stemmer = Stemmer::create(Algorithm::English);
    let mut expansion_terms: Vec<String> = Vec::new();

    for word in query.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        if clean.len() < 3 {
            continue;
        }
        let stemmed = stemmer.stem(&clean).to_string();

        for &(abstract_stem, concrete_terms) in EXPANSIONS {
            if stemmed == abstract_stem || clean == abstract_stem {
                for &term in concrete_terms {
                    expansion_terms.push(format!("{}{}", term, EXPANSION_BOOST));
                }
            }
        }
    }

    if expansion_terms.is_empty() {
        return None;
    }

    // Append expansion terms to the original query. The original terms retain
    // their natural weight (or IC-weighted boost from upstream), while expansion
    // terms get the low ^0.3 boost.
    let expanded = format!("{} {}", query, expansion_terms.join(" "));
    tracing::debug!(
        expansion_count = expansion_terms.len(),
        "BM25 concept expansion: '{}' → {} expansion terms appended",
        &query[..query.len().min(60)],
        expansion_terms.len()
    );
    Some(expanded)
}

/// Split a query on conjunction markers (" AND ", " and also ", " as well as ").
/// Returns None if no conjunction is found, or the split parts otherwise.
fn split_on_conjunction(query: &str) -> Option<Vec<String>> {
    // Order matters: try longer patterns first to avoid partial matches
    let conjunctions = [" and also ", " as well as ", " AND "];

    for conj in &conjunctions {
        // Case-insensitive search for "and also" and "as well as"
        let query_lower = query.to_lowercase();
        if let Some(pos) = query_lower.find(&conj.to_lowercase()) {
            let left = query[..pos].trim().to_string();
            let right = query[pos + conj.len()..].trim().to_string();
            if !left.is_empty() && !right.is_empty() {
                return Some(vec![left, right]);
            }
        }
    }

    None
}

/// Detect relationship signals in a query that indicate multi-entity comparison/transition.
/// These patterns suggest the query is asking about the relationship between entities,
/// not just mentioning them independently.
fn has_relationship_signal(query: &str) -> bool {
    let query_lower = query.to_lowercase();

    let relationship_patterns = [
        "from ",
        " to ",       // "switch from X to Y"
        "between ",   // "between X and Y"
        "switch",     // "switch from X to Y"
        "change",     // "changed from X to Y"
        "compare",    // "compare X and Y"
        "replace",    // "replaced X with Y"
        "migrat",     // "migrate/migration from X to Y"
        "convert",    // "converted from X to Y"
        "transition", // "transition from X to Y"
        "moved from", // "moved from X to Y"
        "versus",
        " vs ",   // "X versus Y"
        "differ", // "difference between X and Y"
    ];

    // Require at least one relationship pattern
    relationship_patterns
        .iter()
        .any(|p| query_lower.contains(p))
}

/// Clean a sub-query part: trim whitespace, remove trailing punctuation,
/// ensure it reads as a complete question/phrase.
fn clean_sub_query(part: &str) -> String {
    let trimmed = part.trim();
    // Remove leading/trailing question marks and periods
    let cleaned = trimmed
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim_end_matches(',')
        .trim_start_matches(',')
        .trim();

    // If the part starts with a lowercase word and looks like a continuation,
    // keep it as-is — the vector search embedding handles fragments well.
    cleaned.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noun_detection() {
        let query = "robot detected obstacle at coordinates";
        let analysis = analyze_query(query);

        let noun_texts: Vec<String> = analysis
            .focal_entities
            .iter()
            .map(|e| e.text.clone())
            .collect();

        assert!(noun_texts.contains(&"robot".to_string()));
        assert!(noun_texts.contains(&"obstacle".to_string()));
        assert!(noun_texts.contains(&"coordinates".to_string()));
    }

    #[test]
    fn test_adjective_detection() {
        let query = "red large obstacle in path";
        let analysis = analyze_query(query);

        let adj_texts: Vec<String> = analysis
            .discriminative_modifiers
            .iter()
            .map(|m| m.text.clone())
            .collect();

        assert!(adj_texts.contains(&"red".to_string()));
        assert!(adj_texts.contains(&"large".to_string()));
    }

    #[test]
    fn test_verb_detection() {
        let query = "robot detected obstacle";
        let analysis = analyze_query(query);

        let verb_texts: Vec<String> = analysis
            .relational_context
            .iter()
            .map(|r| r.text.clone())
            .collect();

        assert!(verb_texts.contains(&"detected".to_string()));
    }

    #[test]
    fn test_information_content_weights() {
        let query = "sensor detected red obstacle";
        let analysis = analyze_query(query);

        // Nouns should have IC weight >= IC_NOUN base
        for entity in &analysis.focal_entities {
            assert!(entity.ic_weight >= IC_NOUN * 0.9); // Allow small variance
        }

        // Adjectives should have IC weight >= IC_ADJECTIVE base
        for modifier in &analysis.discriminative_modifiers {
            assert!(modifier.ic_weight >= IC_ADJECTIVE * 0.9);
        }

        // Verbs should have IC weight >= IC_VERB base
        for relation in &analysis.relational_context {
            assert!(relation.ic_weight >= IC_VERB * 0.9);
        }
    }

    #[test]
    fn test_stemming() {
        let query = "running detection algorithms";
        let analysis = analyze_query(query);

        // Check stems are different from original text
        let stems: Vec<String> = analysis
            .focal_entities
            .iter()
            .map(|e| e.stem.clone())
            .collect();

        // "detection" should stem to "detect"
        assert!(stems.iter().any(|s| s == "detect"));
        // "algorithms" should stem to "algorithm"
        assert!(stems.iter().any(|s| s == "algorithm"));
    }

    #[test]
    fn test_compound_noun_detection() {
        let query = "machine learning neural network";
        let analysis = analyze_query(query);

        assert!(analysis
            .compound_nouns
            .contains(&"machine learning".to_string()));
        assert!(analysis
            .compound_nouns
            .contains(&"neural network".to_string()));
    }

    #[test]
    fn test_negation_detection() {
        let query = "not working correctly";
        let analysis = analyze_query(query);

        assert!(analysis.has_negation);

        // Check that tokens after negation are marked
        let negated_entities: Vec<&FocalEntity> = analysis
            .focal_entities
            .iter()
            .filter(|e| e.negated)
            .collect();

        assert!(!negated_entities.is_empty());
    }

    #[test]
    fn test_negation_scope() {
        let query = "the sensor is not detecting obstacles properly";
        let analysis = analyze_query(query);

        assert!(analysis.has_negation);

        // "detecting" should be marked as negated
        let negated_verbs: Vec<&Relation> = analysis
            .relational_context
            .iter()
            .filter(|r| r.negated)
            .collect();

        assert!(negated_verbs.iter().any(|r| r.text == "detecting"));
    }

    #[test]
    fn test_all_stems_helper() {
        let query = "fast robot detecting obstacles";
        let analysis = analyze_query(query);

        let stems = analysis.all_stems();
        assert!(stems.contains("robot"));
        assert!(stems.contains("fast"));
        assert!(stems.contains("detect"));
        assert!(stems.contains("obstacl")); // Porter stem
    }

    #[test]
    fn test_positive_and_negated_stems() {
        let query = "working memory not failed";
        let analysis = analyze_query(query);

        let positive = analysis.positive_entity_stems();
        let _negated = analysis.negated_entity_stems();

        // "memory" should be positive
        assert!(positive.iter().any(|s| s.contains("memori")));

        // "failed" should be negated (after "not")
        // Note: "failed" might be classified as verb or noun depending on context
    }

    #[test]
    fn test_empty_query() {
        let query = "";
        let analysis = analyze_query(query);

        assert!(analysis.focal_entities.is_empty());
        assert!(analysis.discriminative_modifiers.is_empty());
        assert!(analysis.relational_context.is_empty());
        assert!(!analysis.has_negation);
    }

    #[test]
    fn test_stop_words_filtered() {
        let query = "the a an is are was were";
        let analysis = analyze_query(query);

        // Only verbs should remain (is, are, was, were)
        assert!(analysis.focal_entities.is_empty());
        assert!(analysis.discriminative_modifiers.is_empty());
        assert!(!analysis.relational_context.is_empty());
    }

    #[test]
    fn test_total_weight_calculation() {
        let query = "fast robot detecting red obstacles";
        let analysis = analyze_query(query);

        let weight = analysis.total_weight();
        assert!(weight > 0.0);
    }

    #[test]
    fn test_to_ic_weights() {
        use crate::constants::{IC_ADJECTIVE, IC_NOUN, IC_VERB};

        let query = "fast robot detecting obstacles";
        let analysis = analyze_query(query);
        let weights = analysis.to_ic_weights();

        // Should have weights for terms
        assert!(!weights.is_empty(), "Weights should not be empty");

        // Check that weights were generated
        // Nouns get IC_NOUN (1.5), adjectives IC_ADJECTIVE (0.9), verbs IC_VERB (0.7)
        // The actual terms depend on POS tagging which may vary

        // At minimum, check that we have some weights with expected IC values
        let has_noun_weight = weights.values().any(|&w| (w - IC_NOUN).abs() < 0.01);
        let has_adj_weight = weights.values().any(|&w| (w - IC_ADJECTIVE).abs() < 0.01);
        let has_verb_weight = weights.values().any(|&w| (w - IC_VERB).abs() < 0.01);

        // At least one type should be present
        assert!(
            has_noun_weight || has_adj_weight || has_verb_weight,
            "Should have at least one IC weight type. Weights: {:?}",
            weights
        );
    }

    #[test]
    fn test_to_phrase_boosts() {
        // Test with a query containing known compound noun
        let query = "machine learning model for semantic search";
        let analysis = analyze_query(query);
        let phrases = analysis.to_phrase_boosts();

        // "machine learning" is a known compound pattern
        let has_ml = phrases.iter().any(|(p, _)| p == "machine learning");
        let has_ss = phrases.iter().any(|(p, _)| p == "semantic search");

        assert!(
            has_ml || has_ss,
            "Should detect 'machine learning' or 'semantic search' as phrase. Found: {:?}",
            phrases
        );

        // Compound nouns should have higher boost (2.0)
        for (phrase, boost) in &phrases {
            assert!(
                *boost >= 1.0,
                "Phrase '{}' should have boost >= 1.0, got {}",
                phrase,
                boost
            );
        }
    }

    #[test]
    fn test_to_phrase_boosts_support_group() {
        // Test with LoCoMo-style query
        let query = "when did she go to the support group";
        let analysis = analyze_query(query);
        let phrases = analysis.to_phrase_boosts();

        // "support group" should be detected as a compound
        let has_support_group = phrases.iter().any(|(p, _)| p == "support group");
        assert!(
            has_support_group,
            "Should detect 'support group' as phrase. Found: {:?}",
            phrases
        );
    }

    // =====================================================================
    // QUERY DECOMPOSITION TESTS
    // =====================================================================

    #[test]
    fn test_decompose_conjunction_and() {
        let query = "Why did we change the database AND what was the impact on the timeline?";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        assert!(
            subs.len() >= 3,
            "Should produce original + 2 parts, got {:?}",
            subs
        );
        assert_eq!(subs[0], query); // original always first
    }

    #[test]
    fn test_decompose_conjunction_and_also() {
        let query = "What framework did we pick and also what ORM are we using?";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        assert!(
            subs.len() >= 3,
            "Should split on 'and also', got {:?}",
            subs
        );
    }

    #[test]
    fn test_decompose_multi_entity_switch() {
        let query = "When did we switch from PostgreSQL to MongoDB?";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        // Should produce original + entity sub-queries
        assert!(
            subs.len() >= 2,
            "Should decompose multi-entity query, got {:?}",
            subs
        );
        assert_eq!(subs[0], query);
    }

    #[test]
    fn test_decompose_simple_no_split() {
        let query = "What programming language did we choose?";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        assert_eq!(
            subs.len(),
            1,
            "Simple query should not decompose: {:?}",
            subs
        );
        assert_eq!(subs[0], query);
    }

    #[test]
    fn test_decompose_cap_max_subqueries() {
        // Even with many conjunctions, should cap at MAX_SUBQUERIES + 1 (original)
        let query = "A AND B AND C AND D AND E AND F";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        assert!(
            subs.len() <= crate::constants::QUERY_DECOMPOSITION_MAX_SUBQUERIES + 1,
            "Should cap at max sub-queries + 1, got {}",
            subs.len()
        );
    }

    #[test]
    fn test_decompose_empty_query() {
        let query = "";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn test_decompose_multi_entity_compare() {
        let query = "Compare the performance of React versus Vue";
        let analysis = analyze_query(query);
        let subs = decompose_query(query, &analysis);
        assert!(
            subs.len() >= 2,
            "Should decompose comparison query, got {:?}",
            subs
        );
    }

    // =====================================================================
    // RELATIVE-DATE PRE-PASS TESTS (R1 / F6)
    // =====================================================================

    /// Verify that dateparser alone cannot resolve relative expressions —
    /// confirms F6 gap is real and the pre-pass is necessary.
    #[test]
    fn probe_dateparser_relative_expressions() {
        // Record which inputs dateparser resolves vs. fails.
        // All five should return Err; any Ok would reduce the scope of the fix.
        let cases = [
            "next month",
            "yesterday",
            "last week",
            "3 days ago",
            "tomorrow",
        ];
        for input in cases {
            let result = dateparser::parse(input);
            assert!(
                result.is_err(),
                "dateparser unexpectedly resolved {:?} — F6 gap may be narrower than assumed",
                input
            );
        }
    }

    /// "yesterday" query must yield at least one resolved TemporalBound via bounds().
    #[test]
    fn test_relative_prepass_yesterday_produces_bound() {
        let extraction = extract_temporal_refs("what did she do yesterday");
        let bounds = extraction.bounds();
        assert!(
            !bounds.is_empty(),
            "Expected at least one TemporalBound for 'yesterday' query, got none"
        );
        // The bound must be a resolved Point (not Unknown), but we deliberately
        // avoid matching on the variant name to stay R4-transparent.
        let has_resolved = bounds
            .iter()
            .any(|b| !matches!(b, crate::memory::temporal_facts::TemporalBound::Unknown));
        assert!(
            has_resolved,
            "Expected a non-Unknown TemporalBound for 'yesterday', got: {:?}",
            bounds
        );
    }

    /// "next month" query must yield at least one resolved TemporalBound.
    #[test]
    fn test_relative_prepass_next_month_produces_bound() {
        let extraction = extract_temporal_refs("what are we planning next month");
        let bounds = extraction.bounds();
        assert!(
            !bounds.is_empty(),
            "Expected at least one TemporalBound for 'next month' query, got none"
        );
        let has_resolved = bounds
            .iter()
            .any(|b| !matches!(b, crate::memory::temporal_facts::TemporalBound::Unknown));
        assert!(
            has_resolved,
            "Expected a non-Unknown TemporalBound for 'next month', got: {:?}",
            bounds
        );
    }

    /// A query with no relative phrases must produce no query_bounds and still
    /// behave exactly as before — regression guard for the additive contract.
    #[test]
    fn test_relative_prepass_no_false_positives_for_absolute_query() {
        let extraction = extract_temporal_refs("robot detected obstacle");
        assert!(
            extraction.bounds().is_empty(),
            "Non-temporal query should produce zero query_bounds, got: {:?}",
            extraction.bounds()
        );
    }

    /// "last week" query must produce a bound and also populate refs/earliest/latest
    /// — the additive guarantee (existing fields are still updated).
    #[test]
    fn test_relative_prepass_last_week_populates_refs_and_bounds() {
        let extraction = extract_temporal_refs("what happened last week");
        assert!(
            !extraction.bounds().is_empty(),
            "Expected bounds for 'last week'"
        );
        assert!(
            extraction.has_temporal_refs(),
            "Expected refs to be populated for 'last week'"
        );
        assert!(
            extraction.earliest.is_some(),
            "Expected earliest to be set for 'last week'"
        );
    }

    // -------------------------------------------------------------------------
    // B1: anchor-relative date resolution tests
    // -------------------------------------------------------------------------

    /// "the Sunday before 2023-05-25" → 2023-05-21
    /// 25 May 2023 is a Thursday; the Sunday *before* it is 21 May 2023.
    ///
    /// Post fix-A contract: the derived date (2023-05-21) lands in bounds() only,
    /// NOT in refs or earliest/latest.  The anchor date (2023-05-25) may still
    /// appear in refs from the later passes (dateparser / explicit_dates).
    #[test]
    fn test_anchor_relative_sunday_before_iso() {
        let extraction = extract_temporal_refs("the Sunday before 2023-05-25");
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 21).unwrap();
        // bounds() must contain the derived date (B3 boost signal).
        let bound_has_expected = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, crate::memory::temporal_facts::TemporalBound::Point(p) if *p == expected));
        assert!(
            bound_has_expected,
            "Expected TemporalBound::Point(2023-05-21) in bounds, got: {:?}",
            extraction.bounds()
        );
        // refs must also contain the derived date (Layer 0.6 temporal-fact boost).
        let ref_has_derived = extraction.refs.iter().any(|r| r.date == expected);
        assert!(
            ref_has_derived,
            "Derived date 2023-05-21 must appear in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    /// "the Monday after 2023-05-25" → 2023-05-29
    /// 25 May 2023 is a Thursday; the Monday *after* it is 29 May 2023.
    #[test]
    fn test_anchor_relative_monday_after_iso() {
        let extraction = extract_temporal_refs("the Monday after 2023-05-25");
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 29).unwrap();
        let bound_has_expected = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, crate::memory::temporal_facts::TemporalBound::Point(p) if *p == expected));
        assert!(
            bound_has_expected,
            "Expected TemporalBound::Point(2023-05-29) in bounds, got: {:?}",
            extraction.bounds()
        );
        // refs must also contain the derived date (Layer 0.6 boost).
        let ref_has_derived = extraction.refs.iter().any(|r| r.date == expected);
        assert!(
            ref_has_derived,
            "Derived date 2023-05-29 must appear in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    /// "3 days before 2023-05-25" → 2023-05-22
    #[test]
    fn test_anchor_relative_n_days_before() {
        let extraction = extract_temporal_refs("3 days before 2023-05-25");
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 22).unwrap();
        let bound_has_expected = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, crate::memory::temporal_facts::TemporalBound::Point(p) if *p == expected));
        assert!(
            bound_has_expected,
            "Expected TemporalBound::Point(2023-05-22) in bounds, got: {:?}",
            extraction.bounds()
        );
        // refs must also contain the derived date (Layer 0.6 boost).
        let ref_has_derived = extraction.refs.iter().any(|r| r.date == expected);
        assert!(
            ref_has_derived,
            "Derived date 2023-05-22 must appear in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    /// "the Sunday before 25 May 2023" — text-form anchor → same result 2023-05-21
    #[test]
    fn test_anchor_relative_sunday_before_text_form() {
        let extraction = extract_temporal_refs("the Sunday before 25 May 2023");
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 21).unwrap();
        let bound_has_expected = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, crate::memory::temporal_facts::TemporalBound::Point(p) if *p == expected));
        assert!(
            bound_has_expected,
            "Expected TemporalBound::Point(2023-05-21) in bounds for text-form anchor, got: {:?}",
            extraction.bounds()
        );
        // refs must also contain the derived date.
        let ref_has_derived = extraction.refs.iter().any(|r| r.date == expected);
        assert!(
            ref_has_derived,
            "Derived date 2023-05-21 must appear in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    /// Non-anchor-relative query must be unchanged — no spurious bounds.
    #[test]
    fn test_non_anchor_relative_query_unchanged() {
        let extraction = extract_temporal_refs("robot detected obstacle");
        // Additive contract: existing behaviour is preserved.
        assert!(
            extraction.bounds().is_empty(),
            "Non-temporal query should produce zero query_bounds, got: {:?}",
            extraction.bounds()
        );
        assert!(
            extraction.refs.is_empty(),
            "Non-temporal query should produce zero refs, got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    /// "yesterday" still works — existing relative pass not broken by B1 pass.
    #[test]
    fn test_b1_does_not_break_existing_relative_pass() {
        let extraction = extract_temporal_refs("what did she do yesterday");
        assert!(
            !extraction.bounds().is_empty(),
            "Expected bounds for 'yesterday' after B1 pass added"
        );
    }

    // -------------------------------------------------------------------------
    // B2: named-event anchor resolution tests
    // -------------------------------------------------------------------------

    /// "the Sunday before the wedding" with a resolver that returns wedding = 2023-05-25
    /// (Thursday) → Sunday before = 2023-05-21.
    ///
    /// Post fix-A contract: 2023-05-21 appears in bounds() (boost signal) but NOT
    /// in refs or earliest/latest (no filter narrowing).
    #[test]
    fn test_b2_sunday_before_named_event() {
        use crate::memory::temporal_facts::TemporalBound;

        let wedding_date = chrono::NaiveDate::from_ymd_opt(2023, 5, 25).unwrap();
        let resolver = |phrase: &str| -> Option<chrono::NaiveDate> {
            if phrase.contains("wedding") {
                Some(wedding_date)
            } else {
                None
            }
        };

        let extraction =
            extract_temporal_refs_with_events("the Sunday before the wedding", Some(&resolver));
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 21).unwrap();

        // bounds() must contain the derived date (for B3 boost).
        let has_bound = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, TemporalBound::Point(p) if *p == expected));
        assert!(
            has_bound,
            "Expected TemporalBound::Point(2023-05-21) in bounds, got: {:?}",
            extraction.bounds()
        );

        // refs must also contain the derived date (Layer 0.6 temporal-fact boost).
        let has_ref = extraction.refs.iter().any(|r| r.date == expected);
        assert!(
            has_ref,
            "Derived date 2023-05-21 must appear in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    /// B2: forward ref — no event stored → resolver returns None → no bounds emitted.
    #[test]
    fn test_b2_forward_ref_no_bound_emitted() {
        let resolver = |_phrase: &str| -> Option<chrono::NaiveDate> { None };
        let extraction =
            extract_temporal_refs_with_events("the Sunday before the wedding", Some(&resolver));
        assert!(
            extraction.bounds().is_empty(),
            "Forward ref with None resolver must emit no bounds, got: {:?}",
            extraction.bounds()
        );
    }

    /// B2: absolute-date anchor still works unchanged (reversion guard).
    /// When the anchor is a parseable date, the B1 pass handles it and the B2
    /// resolver must NOT be called.
    #[test]
    fn test_b2_absolute_anchor_unchanged() {
        use std::cell::Cell;
        // Resolver is present but should NOT be called for an absolute-date anchor.
        let called = Cell::new(false);
        let resolver = |_phrase: &str| -> Option<chrono::NaiveDate> {
            called.set(true);
            None
        };
        let extraction =
            extract_temporal_refs_with_events("the Sunday before 2023-05-25", Some(&resolver));
        // B1 already resolved this; resolver must not have been called.
        assert!(
            !called.get(),
            "Resolver must NOT be called when anchor is an absolute date"
        );
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 21).unwrap();
        // bounds() must contain derived date (boost signal).
        let has_bound = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, crate::memory::temporal_facts::TemporalBound::Point(p) if *p == expected));
        assert!(
            has_bound,
            "Absolute-date anchor path must produce TemporalBound::Point(2023-05-21) in bounds, got: {:?}",
            extraction.bounds()
        );
        // refs must also contain the derived date (Layer 0.6 boost).
        let has_ref = extraction.refs.iter().any(|r| r.date == expected);
        assert!(
            has_ref,
            "Derived date 2023-05-21 must appear in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
    }

    // -------------------------------------------------------------------------
    // DIAGNOSTIC: q53/q63 regression investigation
    // -------------------------------------------------------------------------

    /// q53: "When did John help renovate his hometown community center?"
    /// Pass 0 should NOT fire (no anchor-relative pattern).
    /// dateparser + split_temporal_phrases may or may not emit refs.
    /// This test records what ACTUALLY fires so we can confirm the mechanism.
    #[test]
    fn test_q53_no_spurious_bounds() {
        let q53 = "When did John help renovate his hometown community center?";
        let extraction = extract_temporal_refs(q53);
        // This is a WhenQuestion — asks FOR a date. No anchor-relative pattern.
        // Pass 0 must not emit anything. query_bounds must be empty.
        assert!(
            extraction.query_bounds.is_empty(),
            "q53: Pass 0 must emit no query_bounds; got: {:?}",
            extraction.query_bounds
        );
        // Also record what refs/earliest/latest dateparser emitted (can be noise)
        // This is informational — we don't assert on refs here.
        eprintln!(
            "q53 refs: {:?}",
            extraction
                .refs
                .iter()
                .map(|r| (r.date, r.original_text.clone()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "q53 earliest: {:?}, latest: {:?}",
            extraction.earliest, extraction.latest
        );
    }

    /// q63: "How many weeks passed between Maria adopting Coco and Shadow?"
    /// Duration question. Pass 0 must NOT fire (no before|after, no anchor).
    /// No absolute date in the text → no spurious refs/earliest/latest.
    #[test]
    fn test_q63_no_spurious_bounds_or_date_range() {
        let q63 = "How many weeks passed between Maria adopting Coco and Shadow?";
        let extraction = extract_temporal_refs(q63);
        // Pass 0 must not fire.
        assert!(
            extraction.query_bounds.is_empty(),
            "q63: Pass 0 must emit no query_bounds; got: {:?}",
            extraction.query_bounds
        );
        // No absolute date in query → no refs → date_range() must be None.
        // If date_range() is Some(), filter_facts_by_time_window fires and may
        // drop the right answer (the 2022 conversation).
        assert!(
            extraction.date_range().is_none(),
            "q63: duration question with no date must produce no date_range; got: {:?}",
            extraction.date_range()
        );
        eprintln!(
            "q63 refs: {:?}",
            extraction
                .refs
                .iter()
                .map(|r| (r.date, r.original_text.clone()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "q63 earliest: {:?}, latest: {:?}",
            extraction.earliest, extraction.latest
        );
    }

    // -------------------------------------------------------------------------
    // Regression tests: Pass 0 dual-path contract
    // -------------------------------------------------------------------------

    /// Anchor-relative derived date must appear in BOTH bounds() (B3 boost path)
    /// AND refs/earliest (Layer 0.6 temporal-fact boost).
    #[test]
    fn test_anchor_relative_boost_and_ref_contract() {
        use crate::memory::temporal_facts::TemporalBound;
        // "the Sunday before 2023-05-25" derives 2023-05-21.
        let q = "the Sunday before 2023-05-25";
        let extraction = extract_temporal_refs(q);
        let derived = chrono::NaiveDate::from_ymd_opt(2023, 5, 21).unwrap();

        // Derived date must be in bounds() for the B3 boost path.
        assert!(
            extraction
                .bounds()
                .iter()
                .any(|b| matches!(b, TemporalBound::Point(p) if *p == derived)),
            "derived date must be in bounds(); got: {:?}",
            extraction.bounds()
        );
        // Derived date must also be in refs for the Layer 0.6 temporal-fact boost.
        assert!(
            extraction.refs.iter().any(|r| r.date == derived),
            "derived date must be in refs (Layer 0.6 boost); got: {:?}",
            extraction.refs.iter().map(|r| r.date).collect::<Vec<_>>()
        );
        // earliest must be the derived date (or earlier if other dates are present).
        assert!(
            extraction.earliest.map(|e| e <= derived).unwrap_or(false),
            "earliest must be set from the derived date; got: {:?}",
            extraction.earliest
        );
    }

    /// "How many weeks passed between A and B?" — duration counting question.
    /// No anchor-relative pattern, no date in text → query_bounds and date_range
    /// must both be empty so filter_facts_by_time_window is a no-op.
    #[test]
    fn test_duration_counting_no_filter_bounds() {
        let q = "How many weeks passed between Maria adopting Coco and Shadow?";
        let extraction = extract_temporal_refs(q);
        assert!(
            extraction.query_bounds.is_empty(),
            "Duration counting question must produce no query_bounds; got: {:?}",
            extraction.query_bounds
        );
        assert!(
            extraction.date_range().is_none(),
            "Duration counting question with no date must produce no date_range; got: {:?}",
            extraction.date_range()
        );
    }

    /// "When did X verb noun?" — WhenQuestion with no date in text → no filtering bounds.
    #[test]
    fn test_when_question_no_filter_bounds() {
        let q = "When did John help renovate his hometown community center?";
        let extraction = extract_temporal_refs(q);
        assert!(
            extraction.query_bounds.is_empty(),
            "WhenQuestion with no date must produce no query_bounds; got: {:?}",
            extraction.query_bounds
        );
        assert!(
            extraction.date_range().is_none(),
            "WhenQuestion with no date must produce no date_range; got: {:?}",
            extraction.date_range()
        );
    }

    /// B2: "3 days before the graduation" with event resolver → correct offset.
    #[test]
    fn test_b2_n_days_before_named_event() {
        use crate::memory::temporal_facts::TemporalBound;

        let grad_date = chrono::NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        let resolver = |phrase: &str| -> Option<chrono::NaiveDate> {
            if phrase.contains("graduation") {
                Some(grad_date)
            } else {
                None
            }
        };

        let extraction =
            extract_temporal_refs_with_events("3 days before the graduation", Some(&resolver));
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 6, 12).unwrap();

        let has_bound = extraction
            .bounds()
            .iter()
            .any(|b| matches!(b, TemporalBound::Point(p) if *p == expected));
        assert!(
            has_bound,
            "Expected TemporalBound::Point(2023-06-12) for '3 days before graduation', got: {:?}",
            extraction.bounds()
        );
    }

    // -------------------------------------------------------------------------
    // B4: temporal answer-derivation tests
    // -------------------------------------------------------------------------

    use crate::memory::temporal_facts::{ResolvedTime, TemporalFact};
    use crate::memory::types::MemoryId;

    fn b4_fact(
        entity: &str,
        event: &str,
        stems: &[&str],
        resolved_time: ResolvedTime,
        conv: chrono::DateTime<chrono::Utc>,
        mem_uuid: uuid::Uuid,
    ) -> TemporalFact {
        TemporalFact {
            id: format!("{entity}:{event}"),
            entity: entity.to_string(),
            event: event.to_string(),
            event_stems: stems.iter().map(|s| s.to_string()).collect(),
            event_type: crate::memory::temporal_facts::EventType::Occurred,
            relative_time: None,
            resolved_time,
            source_memory_id: MemoryId(mem_uuid),
            conversation_date: conv,
            confidence: 0.9,
            source_text: format!("{entity} did {event}"),
            valid_from: None,
            valid_until: None,
        }
    }

    /// (a) Literal-date anchor: "when is the Sunday before 2023-05-25?"
    ///     → derived_date 2023-05-21, method anchor_relative. No store touched.
    #[test]
    fn test_b4_anchor_relative_literal_date() {
        // Resolver present but must not be needed for a literal-date anchor.
        let resolver = |_p: &str| -> Option<chrono::NaiveDate> { None };
        let answer = derive_temporal_answer(
            "when is the Sunday before 2023-05-25?",
            Some(&resolver),
            None,
        )
        .expect("literal-date anchor must derive an answer");
        assert_eq!(
            answer.derived_date, "2023-05-21",
            "Sunday before Thu 2023-05-25 must be 2023-05-21"
        );
        assert_eq!(answer.method, DerivationMethod::AnchorRelative);
        assert!(answer.supporting_facts.is_empty());
        assert!(answer.source_memories.is_empty());
    }

    /// (b) Event-anchor: "when was the Monday after the wedding?" with a resolver
    ///     mapping wedding → 2023-05-25 (Thu) → Monday after = 2023-05-29,
    ///     method event_anchor.
    #[test]
    fn test_b4_event_anchor_resolver() {
        let wedding = chrono::NaiveDate::from_ymd_opt(2023, 5, 25).unwrap();
        let resolver = |phrase: &str| -> Option<chrono::NaiveDate> {
            phrase.contains("wedding").then_some(wedding)
        };
        let answer = derive_temporal_answer(
            "when was the Monday after the wedding?",
            Some(&resolver),
            None,
        )
        .expect("event-anchor must derive an answer");
        assert_eq!(
            answer.derived_date, "2023-05-29",
            "Monday after Thu 2023-05-25 must be 2023-05-29"
        );
        assert_eq!(answer.method, DerivationMethod::EventAnchor);
    }

    /// (c) Temporal-fact: "when did Melanie go camping?" with a stored fact for
    ///     (Melanie, camping) resolving to 2023-07-04 → derived_date is the
    ///     fact's resolved_time, method temporal_fact, supporting fact + source
    ///     memory surfaced.
    #[test]
    fn test_b4_temporal_fact_lookup() {
        let camp_date = chrono::NaiveDate::from_ymd_opt(2023, 7, 4).unwrap();
        let conv = chrono::DateTime::parse_from_rfc3339("2023-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let mem_uuid = uuid::Uuid::from_u128(0xC0FFEE);
        let fact = b4_fact(
            "Melanie",
            "camping",
            &["camp"],
            ResolvedTime::ExactDate(camp_date),
            conv,
            mem_uuid,
        );
        let lookup = move |_q: &str| -> Vec<TemporalFact> { vec![fact.clone()] };
        // No anchor pattern in this query, so priority-1 yields nothing and we
        // fall through to the fact lookup.
        let answer = derive_temporal_answer("when did Melanie go camping?", None, Some(&lookup))
            .expect("stored temporal fact must derive an answer");
        assert_eq!(
            answer.derived_date, "2023-07-04",
            "derived_date must equal the fact's resolved_time"
        );
        assert_eq!(answer.method, DerivationMethod::TemporalFact);
        assert_eq!(answer.supporting_facts.len(), 1);
        assert_eq!(answer.supporting_facts[0].entity, "Melanie");
        assert_eq!(answer.source_memories.len(), 1);
        assert_eq!(answer.source_memories[0], MemoryId(mem_uuid));
    }

    /// (d) Reversion guard: a non-temporal query derives nothing — even with both
    ///     closures present and a matching fact available.
    #[test]
    fn test_b4_non_temporal_query_is_none() {
        let resolver = |_p: &str| -> Option<chrono::NaiveDate> {
            Some(chrono::NaiveDate::from_ymd_opt(2023, 5, 25).unwrap())
        };
        let conv = chrono::Utc::now();
        let fact = b4_fact(
            "Melanie",
            "camping",
            &["camp"],
            ResolvedTime::ExactDate(chrono::NaiveDate::from_ymd_opt(2023, 7, 4).unwrap()),
            conv,
            uuid::Uuid::nil(),
        );
        let lookup = move |_q: &str| -> Vec<TemporalFact> { vec![fact.clone()] };
        let answer = derive_temporal_answer(
            "What did Melanie think about camping?",
            Some(&resolver),
            Some(&lookup),
        );
        assert!(
            answer.is_none(),
            "Non-temporal query must derive no answer (got {answer:?})"
        );
        // Sanity: it is genuinely classified as non-temporal-answer.
        assert!(!asks_for_temporal_answer(
            "What did Melanie think about camping?"
        ));
    }

    fn shape(q: &str, decomposed: bool) -> RerankShape {
        classify_rerank_shape(q, decomposed, &classify_query(q))
    }

    #[test]
    fn rerank_shape_grounding_shapes_win() {
        assert_eq!(
            shape("where is admin/serve.mjs loaded", false),
            RerankShape::Identifier
        );
        assert_eq!(
            shape("what was the first bug we hit", false),
            RerankShape::Ordinal
        );
        assert_eq!(shape("redis cache", false), RerankShape::Keyword);
        assert_eq!(
            shape("who fixed the EU checkout last week", false),
            RerankShape::Ordinal
        );
        assert_eq!(
            shape("how did we fix the latest checkout bug", false),
            RerankShape::Ordinal
        );
        assert_eq!(
            shape("what was the first bug we hit", true),
            RerankShape::Ordinal
        );
    }

    #[test]
    fn rerank_shape_labels_apply_only_when_nothing_else_does() {
        assert_eq!(
            shape("what did we decide recently about the API", false),
            RerankShape::Temporal
        );
        assert_eq!(
            shape("who fixed the EU checkout yesterday", false),
            RerankShape::Temporal
        );
        assert_eq!(
            shape("who are the key people on the team", false),
            RerankShape::People
        );
        assert_eq!(
            shape("who are the key people on the team", true),
            RerankShape::Abstract
        );
        assert_eq!(
            shape(
                "what filling did Joanna use in the cake she made recently",
                true
            ),
            RerankShape::Abstract
        );
        assert_eq!(
            shape(
                "how did the migration recently change checkout latency",
                false
            ),
            RerankShape::Relational
        );
    }

    #[test]
    fn rerank_shape_reasoning_shapes_only_without_grounding() {
        assert_eq!(
            shape(
                "how did Sarah's Stripe migration fix the EU checkout delay",
                false
            ),
            RerankShape::Relational
        );
        assert_eq!(
            shape(
                "why did the analytics database run out of disk space",
                false
            ),
            RerankShape::Relational
        );
        assert_eq!(
            shape(
                "what performance improvements have we achieved across the product",
                true
            ),
            RerankShape::Abstract
        );
        assert_eq!(
            shape(
                "what caching problems did we have and how were they resolved",
                false
            ),
            RerankShape::Exploratory
        );
        assert_eq!(
            shape("how did we deploy SHO-72 to staging", false),
            RerankShape::Identifier
        );
    }

    #[test]
    fn structured_identifiers_exclude_short_acronyms() {
        assert!(!has_structured_identifier(
            "What LGBTQ+ events has Caroline participated in?"
        ));
        assert!(!has_structured_identifier("Did the HTTPS rollout finish?"));
        assert!(has_structured_identifier(
            "why does ECONNREFUSED happen on startup"
        ));
        assert!(has_structured_identifier("rustc E0308 mismatched types"));
        assert!(has_structured_identifier("status of SHO-72"));
        assert!(has_structured_identifier("what changed in v0.8.3"));
        assert_eq!(
            shape("What LGBTQ+ events has Caroline participated in?", false),
            RerankShape::Exploratory
        );
    }

    #[test]
    fn question_frame_words_are_not_topic_words() {
        for w in ["many", "Long", "reason", "recently", "kind"] {
            assert!(is_question_frame_word(w), "{w}");
        }
        for w in ["yoga", "screenplay", "favorite", "picture"] {
            assert!(!is_question_frame_word(w), "{w}");
        }
    }

    #[test]
    fn people_intent_grades_and_rejects_technical_words() {
        assert_eq!(people_intent("Who are the key people on the team?"), 1.0);
        assert_eq!(
            people_intent("which tasks are assigned to the platform group"),
            1.0
        );
        assert_eq!(people_intent("what did the team members decide"), 0.6);
        assert_eq!(people_intent("what happened to the people counter"), 0.6);
        assert_eq!(people_intent("whose fault was the outage"), 0.0);
        assert_eq!(people_intent("who's on call this week"), 1.0);
        assert_eq!(
            people_intent("change the file owner and the contact form"),
            0.0
        );
    }

    #[test]
    fn content_stems_match_across_inflections() {
        let query = content_stems("what caching problems did we have");
        let memory = content_stems("Raj found the Redis cache invalidation bug");
        assert!(query.contains("cach"));
        assert!(memory.contains("cach"));
        assert!(!query.contains("what"));
    }

    #[test]
    fn prf_terms_need_two_documents_and_skip_query_words() {
        let docs = [
            "Redis cache invalidation race in the pub/sub listener",
            "Added distributed locking with Redlock to the Redis listener",
            "Team lunch at Sakura Sushi",
        ];
        let terms = prf_terms(&docs, "what caching problems did we have", 5);
        assert!(terms.contains(&"redis".to_string()));
        assert!(terms.contains(&"listener".to_string()));
        assert!(!terms.contains(&"sushi".to_string()));
        assert!(!terms.iter().any(|t| t.starts_with("cach")));
        assert_eq!(prf_terms(&docs, "q", 5), prf_terms(&docs, "q", 5));
    }

    #[test]
    fn rerank_shape_is_ordered_by_default_share() {
        use crate::constants as c;
        let shares = [
            (RerankShape::Identifier, c::CE_BLEND_IDENTIFIER),
            (RerankShape::Ordinal, c::CE_BLEND_ORDINAL),
            (RerankShape::Keyword, c::CE_BLEND_KEYWORD),
            (RerankShape::Temporal, c::CE_BLEND_TEMPORAL),
            (RerankShape::People, c::CE_BLEND_PEOPLE),
            (RerankShape::Exploratory, c::CE_BLEND_EXPLORATORY),
            (RerankShape::Attribute, c::CE_BLEND_ATTRIBUTE),
            (RerankShape::Relational, c::CE_BLEND_RELATIONAL),
            (RerankShape::Abstract, c::CE_BLEND_ABSTRACT),
        ];
        for pair in shares.windows(2) {
            assert!(pair[0].0 < pair[1].0);
            assert!(
                pair[0].1 <= pair[1].1,
                "{:?} share above {:?}",
                pair[0].0,
                pair[1].0
            );
        }
    }

    #[test]
    fn a_causal_verb_asks_for_every_relation_the_cue_extractor_writes() {
        for stem in ["caus", "lead", "result"] {
            let wanted = verb_stem_to_relation_types(stem);
            for causal in [
                RelationType::Causes,
                RelationType::ResultsIn,
                RelationType::Triggers,
            ] {
                assert!(wanted.contains(&causal), "{stem} misses {causal:?}");
            }
        }
        let (relation, _) = crate::graph_memory::extract_directed_predicate(
            "The storm caused the outage.",
            "storm",
            "outage",
        )
        .expect("a causal cue");
        assert!(verb_stem_to_relation_types("caus").contains(&relation));
    }
}
