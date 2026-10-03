//! Temporal Fact Extraction and Storage
//!
//! Extracts and stores temporal facts from conversations for multi-hop reasoning.
//! Key insight: Multi-hop temporal queries like "When is X planning Y?" require:
//! 1. Finding the FIRST/PLANNING mention, not any mention
//! 2. Resolving relative dates ("next month", "last Saturday") to absolute dates
//! 3. Indexing by entity + event for fast lookup
//!
//! Storage schema:
//! - `temporal_facts:{user_id}:{fact_id}` - Primary storage
//! - `temporal_by_entity:{user_id}:{entity}:{fact_id}` - Entity index
//! - `temporal_by_event:{user_id}:{event_stem}:{fact_id}` - Event index

use anyhow::Result;
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc, Weekday};
#[cfg(feature = "rocksdb")]
use rocksdb::DB;
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use super::types::MemoryId;

/// Type of temporal event
// bincode: append-only — never reorder or insert; new variants go last
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Event was planned/scheduled ("We're planning to go camping next month")
    Planned,
    /// Event occurred/happened ("I ran a charity race last Saturday")
    Occurred,
    /// Event was mentioned in past tense referring to history ("I painted that in 2022")
    Historical,
    /// A past intention whose fulfilment is unknown ("I was going to visit her
    /// last week but couldn't", "I wanted to go camping"). NOT a confident
    /// future (the intent is past) and NOT a confident occurrence (the act may
    /// never have happened). Appended LAST — bincode tag 3.
    PastIntent,
}

/// A temporal fact extracted from conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFact {
    /// Unique identifier
    pub id: String,
    /// Entity involved (e.g., "Melanie", "Caroline")
    pub entity: String,
    /// Event/action (e.g., "camping", "charity race", "painted sunrise")
    pub event: String,
    /// Stemmed event keywords for matching
    pub event_stems: Vec<String>,
    /// Type of event (planned, occurred, historical)
    pub event_type: EventType,
    /// Original relative time expression ("next month", "last Saturday")
    pub relative_time: Option<String>,
    /// Resolved absolute time
    pub resolved_time: ResolvedTime,
    /// Source memory ID
    pub source_memory_id: MemoryId,
    /// Conversation timestamp (used to resolve relative dates)
    pub conversation_date: DateTime<Utc>,
    /// Confidence in extraction (0.0-1.0)
    pub confidence: f32,
    /// Original sentence fragment
    pub source_text: String,
    /// When this fact became true (defaults to conversation_date if not set)
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
    /// When this fact stopped being true (None = still valid).
    /// Set automatically by contradiction detection when a newer fact supersedes this one.
    #[serde(default)]
    pub valid_until: Option<DateTime<Utc>>,
}

/// Resolved time representation
// bincode: append-only — never reorder or insert; new variants go last
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolvedTime {
    /// Exact date (e.g., "May 7, 2023")
    ExactDate(NaiveDate),
    /// Month and year (e.g., "June 2023")
    MonthYear { month: u32, year: i32 },
    /// Just year (e.g., "2022")
    Year(i32),
    /// Relative to conversation ("the week before", "next month")
    RelativeDescription(String),
    /// Unknown/couldn't resolve
    Unknown,
    // --- INFERENTIAL VARIANTS — appended after Unknown; never reorder ---
    /// Open lower bound: "after <X>" / "since <X>" / "from <X>"
    After(NaiveDate),
    /// Open upper bound: "before <X>" / "prior to <X>" / "until <X>"
    Before(NaiveDate),
    /// Fuzzy window centred on a date: "around <X>" / "~<X>"
    Around { centre: NaiveDate, radius_days: u32 },
}

/// A resolved temporal value projected into an interval for overlap testing.
///
/// This is the comparison form used by the retrieval validation gate
/// (`filter_facts_by_time_window`). Discrete/bounding indicators resolve to
/// `Point` or `Closed`; anything that cannot be resolved becomes `Unknown`,
/// which signals the gate to fall back to `conversation_date`.
///
/// Inferential/open variants (After/Before/Fuzzy) are produced by the
/// inferential resolver (R4) and flow through the query gate transparently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBound {
    /// Zero-width: a single day (yesterday, an exact date, last Saturday).
    Point(NaiveDate),
    /// A closed interval [start, end] (next week, last weekend, a month, a year).
    Closed { start: NaiveDate, end: NaiveDate },
    /// Could not be resolved to an interval — caller uses conversation_date fallback.
    Unknown,
    /// Open lower bound: matches anything on or after `anchor`.
    After(NaiveDate),
    /// Open upper bound: matches anything on or before `anchor`.
    Before(NaiveDate),
    /// Fuzzy closed interval: [centre − radius_days, centre + radius_days].
    Fuzzy { centre: NaiveDate, radius_days: u32 },
}

impl TemporalBound {
    /// Returns true if this bound overlaps the inclusive window [start, end].
    ///
    /// `Unknown` never overlaps — the gate handles it via the conversation_date
    /// fallback rather than this method. Inverted/empty inputs are guarded (J3).
    ///
    /// Inferential variants:
    /// - `After(x)`:  overlaps iff the window extends at or past `x` (we >= x).
    /// - `Before(x)`: overlaps iff the window starts at or before `x` (ws <= x).
    /// - `Fuzzy{c,r}`: the fuzzy interval [c-r, c+r] intersects [ws, we].
    pub fn overlaps_range(&self, window_start: NaiveDate, window_end: NaiveDate) -> bool {
        // Guard inverted window — treat as no overlap rather than silently matching.
        if window_start > window_end {
            return false;
        }
        match self {
            TemporalBound::Point(d) => *d >= window_start && *d <= window_end,
            TemporalBound::Closed { start, end } => {
                // Guard a malformed interval.
                if start > end {
                    return false;
                }
                // Standard inclusive interval intersection.
                *start <= window_end && window_start <= *end
            }
            TemporalBound::Unknown => false,
            // Open lower bound: the "after X" interval extends to +infinity.
            // It intersects [ws, we] iff we >= x.
            TemporalBound::After(anchor) => window_end >= *anchor,
            // Open upper bound: the "before X" interval extends to -infinity.
            // It intersects [ws, we] iff ws <= x.
            TemporalBound::Before(anchor) => window_start <= *anchor,
            // Fuzzy interval [c-r, c+r] intersects [ws, we] via standard overlap.
            TemporalBound::Fuzzy {
                centre,
                radius_days,
            } => {
                let r = Duration::days(*radius_days as i64);
                let fuzzy_start = *centre - r;
                let fuzzy_end = *centre + r;
                fuzzy_start <= window_end && window_start <= fuzzy_end
            }
        }
    }
}

impl ResolvedTime {
    /// Convert to a sortable string for comparison.
    ///
    /// Existing arms (ExactDate/MonthYear/Year/RelativeDescription/Unknown) are
    /// byte-identical to the pre-R4 output — do NOT change them.
    ///
    /// New arms use a namespace prefix that cannot collide with existing arms:
    ///   After(d)      → "after:YYYY-MM-DD"
    ///   Before(d)     → "before:YYYY-MM-DD"
    ///   Around{c,r}   → "around:YYYY-MM-DD:Nd"
    /// Existing arms never start with "after:", "before:", or "around:", so
    /// contradiction-dedup will never conflate an inferential fact with an exact one.
    pub fn to_sortable_string(&self) -> String {
        match self {
            ResolvedTime::ExactDate(d) => d.format("%Y-%m-%d").to_string(),
            ResolvedTime::MonthYear { month, year } => format!("{:04}-{:02}", year, month),
            ResolvedTime::Year(y) => format!("{:04}", y),
            ResolvedTime::RelativeDescription(s) => s.clone(),
            ResolvedTime::Unknown => "unknown".to_string(),
            // Inferential — namespaced to prevent collision with exact-date arms.
            ResolvedTime::After(d) => format!("after:{}", d.format("%Y-%m-%d")),
            ResolvedTime::Before(d) => format!("before:{}", d.format("%Y-%m-%d")),
            ResolvedTime::Around {
                centre,
                radius_days,
            } => {
                format!("around:{}:{}d", centre.format("%Y-%m-%d"), radius_days)
            }
        }
    }

    /// Project this resolved time into a `TemporalBound` for interval-overlap
    /// validation against a query window.
    ///
    /// Discrete points (exact dates, weekdays) become `Point`; spans
    /// (month, year) become `Closed`. Unresolved values (`RelativeDescription`,
    /// `Unknown`) become `TemporalBound::Unknown`, signalling the gate to fall
    /// back to `conversation_date`.
    ///
    /// Date construction is checked (`from_ymd_opt`); corrupt stored data
    /// (e.g. a `MonthYear { month: 13 }`) degrades to `Unknown` rather than
    /// panicking the retrieval path.
    pub fn to_bound(&self) -> TemporalBound {
        match self {
            ResolvedTime::ExactDate(d) => TemporalBound::Point(*d),
            ResolvedTime::MonthYear { month, year } => {
                match (
                    NaiveDate::from_ymd_opt(*year, *month, 1),
                    last_day_of_month(*year, *month),
                ) {
                    (Some(start), Some(end)) => TemporalBound::Closed { start, end },
                    _ => TemporalBound::Unknown,
                }
            }
            ResolvedTime::Year(y) => {
                match (
                    NaiveDate::from_ymd_opt(*y, 1, 1),
                    NaiveDate::from_ymd_opt(*y, 12, 31),
                ) {
                    (Some(start), Some(end)) => TemporalBound::Closed { start, end },
                    _ => TemporalBound::Unknown,
                }
            }
            ResolvedTime::RelativeDescription(_) | ResolvedTime::Unknown => TemporalBound::Unknown,
            // Inferential → corresponding open/fuzzy bound.
            ResolvedTime::After(d) => TemporalBound::After(*d),
            ResolvedTime::Before(d) => TemporalBound::Before(*d),
            ResolvedTime::Around {
                centre,
                radius_days,
            } => TemporalBound::Fuzzy {
                centre: *centre,
                radius_days: *radius_days,
            },
        }
    }
}

/// Last calendar day of `month` in `year`, computed by stepping to the first of
/// the next month and subtracting a day. Returns `None` on invalid input.
fn last_day_of_month(year: i32, month: u32) -> Option<NaiveDate> {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    NaiveDate::from_ymd_opt(next_year, next_month, 1).map(|d| d - Duration::days(1))
}

/// How a temporal fact was invalidated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContradictionType {
    /// Same entity + event_type + stem overlap; newer fact replaces older
    Superseded,
    /// Semantic opposition detected between source texts
    Contradicted,
}

/// Returns `true` if the fact has been invalidated (valid_until is set and in the past).
fn is_expired(fact: &TemporalFact) -> bool {
    match fact.valid_until {
        Some(until) => until < Utc::now(),
        None => false,
    }
}

/// Compute word-overlap cosine similarity between two texts.
///
/// Tokenizes both texts into lowercased words, builds term-frequency vectors,
/// and returns the cosine similarity (0.0..1.0). Words shorter than 2 chars
/// are excluded as noise.
pub fn text_similarity(a: &str, b: &str) -> f32 {
    use std::collections::HashMap;

    fn term_freq(text: &str) -> HashMap<String, f32> {
        let mut freq = HashMap::new();
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let w = word.to_lowercase();
            if w.len() >= 2 {
                *freq.entry(w).or_insert(0.0) += 1.0;
            }
        }
        freq
    }

    let tf_a = term_freq(a);
    let tf_b = term_freq(b);

    if tf_a.is_empty() || tf_b.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (word, &count_a) in &tf_a {
        norm_a += count_a * count_a;
        if let Some(&count_b) = tf_b.get(word) {
            dot += count_a * count_b;
        }
    }
    for &count_b in tf_b.values() {
        norm_b += count_b * count_b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Detect semantic opposition between two text fragments using heuristics.
///
/// Checks for:
/// 1. Negation inversion: one text says "X is Y" and the other says "X is not Y"
///    (or "isn't", "doesn't", "won't", "can't", etc.)
/// 2. Keyword opposition: one text contains a term and the other contains its antonym
///    from a curated set of common binary oppositions.
pub fn detect_semantic_opposition(text_a: &str, text_b: &str) -> bool {
    let a_lower = text_a.to_lowercase();
    let b_lower = text_b.to_lowercase();

    // Strategy 1: Negation asymmetry.
    // If one sentence contains a negation marker and the other does not,
    // and they share significant vocabulary (caller already verified similarity > 0.85),
    // that's a strong signal of contradiction.
    let negation_markers = [
        "not ",
        "n't ",
        "no ",
        "never ",
        "doesn't ",
        "don't ",
        "didn't ",
        "isn't ",
        "aren't ",
        "wasn't ",
        "weren't ",
        "won't ",
        "wouldn't ",
        "can't ",
        "cannot ",
        "couldn't ",
        "shouldn't ",
        "hasn't ",
        "haven't ",
        "hadn't ",
    ];

    let a_negated = negation_markers.iter().any(|m| a_lower.contains(m));
    let b_negated = negation_markers.iter().any(|m| b_lower.contains(m));

    if a_negated != b_negated {
        return true;
    }

    // Strategy 2: Keyword opposition pairs.
    // If text A contains one side of a binary pair and text B contains the other,
    // that indicates a factual flip.
    let opposition_pairs: &[(&str, &str)] = &[
        ("yes", "no"),
        ("true", "false"),
        ("enabled", "disabled"),
        ("active", "inactive"),
        ("uses", "doesn't use"),
        ("supports", "doesn't support"),
        ("likes", "dislikes"),
        ("allow", "disallow"),
        ("accept", "reject"),
        ("include", "exclude"),
        ("open", "closed"),
        ("start", "stop"),
        ("enable", "disable"),
        ("connect", "disconnect"),
        ("agree", "disagree"),
        ("available", "unavailable"),
        ("valid", "invalid"),
        ("correct", "incorrect"),
        ("possible", "impossible"),
        ("complete", "incomplete"),
    ];

    for &(pos, neg) in opposition_pairs {
        let a_has_pos = a_lower.contains(pos);
        let a_has_neg = a_lower.contains(neg);
        let b_has_pos = b_lower.contains(pos);
        let b_has_neg = b_lower.contains(neg);

        // One text has the positive form while the other has the negative form
        if (a_has_pos && b_has_neg && !a_has_neg && !b_has_pos)
            || (a_has_neg && b_has_pos && !a_has_pos && !b_has_neg)
        {
            return true;
        }
    }

    false
}

#[cfg(feature = "rocksdb")]
/// Minimal deserializer for the GC-path scanner in `storage.rs`.
///
/// Decodes a raw RocksDB value into a [`TemporalFact`] so that
/// `storage::MemoryStorage::temporal_fact_cites_memory` can inspect
/// `source_memory_id` without depending on the full `TemporalFactStore` API.
/// `pub(super)` — visible only within the `memory` module subtree.
pub(super) fn decode_for_gc(data: &[u8]) -> anyhow::Result<TemporalFact> {
    let (fact, _): (TemporalFact, _) =
        bincode::serde::decode_from_slice(data, bincode::config::standard())?;
    Ok(fact)
}

/// Storage for temporal facts
pub struct TemporalFactStore {
    /// Default-keyspace rows under `temporal_facts:`/`temporal_by_*:` prefixes,
    /// so this binds the DEFAULT keyspace handle — the same keyspace the raw
    /// `db.put` calls used, never the `kv` family.
    store: Arc<dyn crate::storage::KeyValueStore>,
}

impl TemporalFactStore {
    #[cfg(feature = "rocksdb")]
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            store: Arc::new(crate::storage::legacy_rocksdb::RocksDbDefaultKeyValueStore::new(db)),
        }
    }

    /// Bind a handle directly — the constructor a backend-agnostic caller uses.
    pub fn with_store(store: Arc<dyn crate::storage::KeyValueStore>) -> Self {
        Self { store }
    }

    /// Store a temporal fact
    pub fn store(&self, user_id: &str, fact: &TemporalFact) -> Result<()> {
        // Primary storage
        let key = format!("temporal_facts:{}:{}", user_id, fact.id);
        let value = bincode::serde::encode_to_vec(fact, bincode::config::standard())?;
        self.store.put_bytes(key.as_bytes(), &value)?;

        // Entity index
        let entity_key = format!(
            "temporal_by_entity:{}:{}:{}",
            user_id,
            fact.entity.to_lowercase(),
            fact.id
        );
        self.store
            .put_bytes(entity_key.as_bytes(), fact.id.as_bytes())?;

        // Event index (by each stem)
        for stem in &fact.event_stems {
            let event_key = format!("temporal_by_event:{}:{}:{}", user_id, stem, fact.id);
            self.store
                .put_bytes(event_key.as_bytes(), fact.id.as_bytes())?;
        }

        Ok(())
    }

    /// Store multiple facts
    pub fn store_batch(&self, user_id: &str, facts: &[TemporalFact]) -> Result<usize> {
        let mut stored = 0;
        for fact in facts {
            if self.store(user_id, fact).is_ok() {
                stored += 1;
            }
        }
        Ok(stored)
    }

    /// Get a fact by ID
    pub fn get(&self, user_id: &str, fact_id: &str) -> Result<Option<TemporalFact>> {
        let key = format!("temporal_facts:{}:{}", user_id, fact_id);
        match self.store.get_bytes(key.as_bytes())? {
            Some(data) => {
                let (fact, _): (TemporalFact, _) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
                Ok(Some(fact))
            }
            None => Ok(None),
        }
    }

    /// Check if a new fact contradicts existing facts about the same entity and event type.
    /// If so, set `valid_until` on the old fact to the new fact's `valid_from` timestamp,
    /// effectively invalidating the superseded fact.
    ///
    /// Two detection passes:
    /// 1. **Superseded** — same entity + event_type + stem overlap, content differs
    /// 2. **Contradicted** — same entity (any event_type), high text similarity but
    ///    opposing semantic content detected via negation/opposition heuristics
    ///
    /// Returns the IDs and contradiction types of facts that were invalidated.
    pub fn detect_and_resolve_contradictions(
        &self,
        user_id: &str,
        new_fact: &TemporalFact,
    ) -> Result<Vec<String>> {
        // Find existing valid facts for the same entity
        let existing = self.find_by_entity_filtered(user_id, &new_fact.entity, 500, false)?;

        let mut invalidated_ids = Vec::new();
        let mut already_invalidated = HashSet::new();
        let now = new_fact.valid_from.unwrap_or_else(Utc::now);

        // Pass 1: Superseded — same entity + event_type + stem overlap
        for old_fact in &existing {
            if old_fact.valid_until.is_some() || old_fact.id == new_fact.id {
                continue;
            }

            if old_fact.event_type != new_fact.event_type {
                continue;
            }

            let has_stem_overlap = old_fact
                .event_stems
                .iter()
                .any(|s| new_fact.event_stems.contains(s));

            if !has_stem_overlap {
                continue;
            }

            let content_differs = old_fact.source_text != new_fact.source_text
                || old_fact.resolved_time.to_sortable_string()
                    != new_fact.resolved_time.to_sortable_string();

            if content_differs {
                self.invalidate_fact(user_id, old_fact, now, ContradictionType::Superseded)?;

                tracing::info!(
                    user_id = user_id,
                    old_fact_id = %old_fact.id,
                    new_fact_id = %new_fact.id,
                    entity = %new_fact.entity,
                    event_type = ?new_fact.event_type,
                    contradiction_type = "superseded",
                    "Temporal fact superseded — old fact invalidated"
                );

                already_invalidated.insert(old_fact.id.clone());
                invalidated_ids.push(old_fact.id.clone());
            }
        }

        // Pass 2: Contradicted — same entity, different event_type, semantic opposition
        for old_fact in &existing {
            if old_fact.valid_until.is_some()
                || old_fact.id == new_fact.id
                || already_invalidated.contains(&old_fact.id)
            {
                continue;
            }

            // Require high text similarity (shared vocabulary) before checking opposition
            let similarity = text_similarity(&old_fact.source_text, &new_fact.source_text);
            if similarity < 0.85 {
                continue;
            }

            if detect_semantic_opposition(&old_fact.source_text, &new_fact.source_text) {
                self.invalidate_fact(user_id, old_fact, now, ContradictionType::Contradicted)?;

                tracing::info!(
                    user_id = user_id,
                    old_fact_id = %old_fact.id,
                    new_fact_id = %new_fact.id,
                    entity = %new_fact.entity,
                    similarity = similarity,
                    contradiction_type = "contradicted",
                    "Semantic contradiction detected — old fact invalidated"
                );

                invalidated_ids.push(old_fact.id.clone());
            }
        }

        Ok(invalidated_ids)
    }

    /// Invalidate a single fact by setting valid_until and persisting.
    fn invalidate_fact(
        &self,
        user_id: &str,
        old_fact: &TemporalFact,
        valid_until: DateTime<Utc>,
        _contradiction_type: ContradictionType,
    ) -> Result<()> {
        let mut updated = old_fact.clone();
        updated.valid_until = Some(valid_until);

        let key = format!("temporal_facts:{}:{}", user_id, updated.id);
        let value = bincode::serde::encode_to_vec(&updated, bincode::config::standard())?;
        self.store.put_bytes(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Find facts by entity, with expired-fact filtering.
    pub fn find_by_entity(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
    ) -> Result<Vec<TemporalFact>> {
        self.find_by_entity_filtered(user_id, entity, limit, false)
    }

    /// Find facts by entity, optionally including expired facts.
    pub fn find_by_entity_filtered(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<TemporalFact>> {
        let prefix = format!("temporal_by_entity:{}:{}:", user_id, entity.to_lowercase());
        self.find_by_prefix(&prefix, user_id, limit, include_expired)
    }

    /// Find facts by event keyword, with expired-fact filtering.
    pub fn find_by_event(
        &self,
        user_id: &str,
        event: &str,
        limit: usize,
    ) -> Result<Vec<TemporalFact>> {
        self.find_by_event_filtered(user_id, event, limit, false)
    }

    /// Find facts by event keyword, optionally including expired facts.
    pub fn find_by_event_filtered(
        &self,
        user_id: &str,
        event: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<TemporalFact>> {
        let stemmer = Stemmer::create(Algorithm::English);
        let stem = stemmer.stem(&event.to_lowercase()).to_string();
        let prefix = format!("temporal_by_event:{}:{}:", user_id, stem);
        self.find_by_prefix(&prefix, user_id, limit, include_expired)
    }

    /// Find facts matching entity AND event
    pub fn find_by_entity_and_event(
        &self,
        user_id: &str,
        entity: &str,
        event_keywords: &[&str],
        event_type: Option<EventType>,
    ) -> Result<Vec<TemporalFact>> {
        self.find_by_entity_and_event_filtered(user_id, entity, event_keywords, event_type, false)
    }

    /// Find facts matching entity AND event, optionally including expired facts.
    pub fn find_by_entity_and_event_filtered(
        &self,
        user_id: &str,
        entity: &str,
        event_keywords: &[&str],
        event_type: Option<EventType>,
        include_expired: bool,
    ) -> Result<Vec<TemporalFact>> {
        // Get facts by entity (already filtered by include_expired)
        let entity_facts = self.find_by_entity_filtered(user_id, entity, 100, include_expired)?;

        // Filter by event keywords
        let stemmer = Stemmer::create(Algorithm::English);
        let event_stems: HashSet<String> = event_keywords
            .iter()
            .map(|kw| stemmer.stem(&kw.to_lowercase()).to_string())
            .collect();

        let mut matching: Vec<TemporalFact> = entity_facts
            .into_iter()
            .filter(|f| {
                // Check if any event stem matches
                let has_event_match = f.event_stems.iter().any(|s| event_stems.contains(s));
                // Check event type if specified
                let type_matches = event_type.is_none_or(|t| f.event_type == t);
                has_event_match && type_matches
            })
            .collect();

        // Sort by conversation date (earliest first for "planning" queries)
        matching.sort_by_key(|f| f.conversation_date);

        Ok(matching)
    }

    fn find_by_prefix(
        &self,
        prefix: &str,
        user_id: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<TemporalFact>> {
        let mut facts = Vec::new();
        let mut seen_ids = HashSet::new();

        // scan_prefix enforces the bound; the hand-written break is gone.
        for (_key, value) in self.store.scan_prefix(prefix.as_bytes(), None)? {
            let fact_id = String::from_utf8_lossy(&value);
            if seen_ids.insert(fact_id.to_string()) {
                if let Some(fact) = self.get(user_id, &fact_id)? {
                    if include_expired || !is_expired(&fact) {
                        facts.push(fact);
                        if facts.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        Ok(facts)
    }

    /// List all temporal facts for a user, with expired-fact filtering.
    pub fn list(&self, user_id: &str, limit: usize) -> Result<Vec<TemporalFact>> {
        self.list_filtered(user_id, limit, false)
    }

    /// List all temporal facts for a user, optionally including expired facts.
    pub fn list_filtered(
        &self,
        user_id: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<TemporalFact>> {
        let prefix = format!("temporal_facts:{}:", user_id);
        let mut facts = Vec::new();

        // scan_prefix enforces the bound; the hand-written break is gone.
        for (_key, value) in self.store.scan_prefix(prefix.as_bytes(), None)? {
            if let Ok((fact, _)) = bincode::serde::decode_from_slice::<TemporalFact, _>(
                &value,
                bincode::config::standard(),
            ) {
                if include_expired || !is_expired(&fact) {
                    facts.push(fact);
                    if facts.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(facts)
    }

    /// Return temporal facts that were valid at the given instant `at`
    /// (point-in-time query). A fact is valid at `at` when its `valid_from`
    /// (or `conversation_date` if `valid_from` is `None`) is at or before
    /// `at`, and its `valid_until` is `None` or strictly after `at`.
    pub fn as_of(
        &self,
        user_id: &str,
        at: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<TemporalFact>> {
        let prefix = format!("temporal_facts:{}:", user_id);
        let mut facts = Vec::new();

        // scan_prefix enforces the bound; the hand-written break is gone.
        for (_key, value) in self.store.scan_prefix(prefix.as_bytes(), None)? {
            if let Ok((fact, _)) = bincode::serde::decode_from_slice::<TemporalFact, _>(
                &value,
                bincode::config::standard(),
            ) {
                let from = fact.valid_from.unwrap_or(fact.conversation_date);
                if from > at {
                    continue;
                }
                let still_valid = match fact.valid_until {
                    Some(until) => until > at,
                    None => true,
                };
                if !still_valid {
                    continue;
                }
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        Ok(facts)
    }

    // =========================================================================
    // R5: named-event second-hop resolution (write-time, cycle-safe, B2/B3)
    // =========================================================================

    /// Resolve the `resolved_time` of a single fact whose relative time
    /// expression references a NAMED EVENT (e.g. "after the wedding") rather
    /// than a calendar phrase.
    ///
    /// **B2 — write-time only.**  Resolution is performed once, at store time,
    /// against facts that ALREADY exist in the database.  If the referenced
    /// event is not yet stored (forward reference), the bound stays `Unknown`.
    /// No back-fill cascade is ever triggered when later events arrive.
    ///
    /// **B3 — depth-bounded + cycle-safe.**  The hop depth is capped at 2 and a
    /// `visited` set of fact-ids prevents infinite loops such as
    /// "A after B" + "B after A".
    ///
    /// Returns an updated copy of `fact` if the hop succeeded, or the original
    /// unchanged fact if: (a) the resolved_time is already concrete, (b) the
    /// relative_time contains no recognised after/before preposition, (c) the
    /// stripped object resolves to a date (not a named event), or (d) no
    /// matching stored fact is found.
    pub fn resolve_named_event_hop(
        &self,
        user_id: &str,
        fact: &TemporalFact,
        visited: &mut HashSet<String>,
        depth: u8,
    ) -> TemporalFact {
        // Hard depth limit.
        if depth >= 2 {
            return fact.clone();
        }

        // Only attempt resolution when the current resolved_time is Unknown or
        // RelativeDescription — if it is already concrete we have nothing to do.
        match &fact.resolved_time {
            ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {}
            _ => return fact.clone(),
        }

        // Extract the relative_time string — nothing to do without one.
        let rel = match &fact.relative_time {
            Some(r) => r.clone(),
            None => return fact.clone(),
        };
        let rel_lower = rel.to_lowercase();

        // Detect a leading preposition and strip it.
        // Returns (is_after: bool, object_phrase: &str) or None.
        let (is_after, object_phrase) = {
            let after_prefixes = ["since ", "from ", "after "];
            let before_prefixes = ["prior to ", "before ", "until "];

            if let Some(rest) = after_prefixes
                .iter()
                .find_map(|p| rel_lower.strip_prefix(p))
            {
                (true, rest)
            } else if let Some(rest) = before_prefixes
                .iter()
                .find_map(|p| rel_lower.strip_prefix(p))
            {
                (false, rest)
            } else {
                // No recognised preposition — not a named-event phrase.
                return fact.clone();
            }
        };

        // Guard: if the object phrase already resolves to a concrete date via
        // `resolve_relative_time`, the R4 path already handled it correctly and
        // we must not override.
        let probe = resolve_relative_time(object_phrase, fact.conversation_date);
        match &probe {
            ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {
                // Object is a named event — continue with second-hop lookup.
            }
            _ => {
                // The object is a date/relative-date phrase; R4 would have handled
                // it, but if it still arrived here as Unknown the probe now gives
                // us a concrete anchor — use it directly.
                let resolved = if is_after {
                    match &probe {
                        ResolvedTime::ExactDate(d) => ResolvedTime::After(*d),
                        ResolvedTime::MonthYear { month, year } => {
                            match last_day_of_month(*year, *month) {
                                Some(d) => ResolvedTime::After(d),
                                None => return fact.clone(),
                            }
                        }
                        ResolvedTime::Year(y) => match NaiveDate::from_ymd_opt(*y, 12, 31) {
                            Some(d) => ResolvedTime::After(d),
                            None => return fact.clone(),
                        },
                        other => other.clone(),
                    }
                } else {
                    match &probe {
                        ResolvedTime::ExactDate(d) => ResolvedTime::Before(*d),
                        ResolvedTime::MonthYear { month, year } => {
                            match NaiveDate::from_ymd_opt(*year, *month, 1) {
                                Some(d) => ResolvedTime::Before(d),
                                None => return fact.clone(),
                            }
                        }
                        ResolvedTime::Year(y) => match NaiveDate::from_ymd_opt(*y, 1, 1) {
                            Some(d) => ResolvedTime::Before(d),
                            None => return fact.clone(),
                        },
                        other => other.clone(),
                    }
                };
                let mut updated = fact.clone();
                updated.resolved_time = resolved;
                return updated;
            }
        }

        // Delegate to the shared event-lookup helper.
        // We insert the current fact's id into the visited set *before* the
        // lookup so that the helper's own cycle guard cannot re-enter this fact.
        visited.insert(fact.id.clone());

        // Use resolve_event_to_date for the lookup, but we need direction-aware
        // date selection (latest for "after", earliest for "before"), which
        // requires direct access to the candidates.  We call the helper only for
        // simple ExactDate anchors; for MonthYear/Year we stay with the same
        // date-selection logic as before.
        //
        // Implementation: re-use resolve_event_to_date to get a representative
        // NaiveDate, then adjust for directionality via the stored resolved_time
        // of the winning candidate.
        let stemmer = Stemmer::create(Algorithm::English);
        let stopwords: HashSet<&str> = [
            "the", "a", "an", "of", "in", "at", "on", "to", "for", "and", "or",
        ]
        .iter()
        .copied()
        .collect();

        let keywords: Vec<String> = object_phrase
            .split_whitespace()
            .filter(|w| w.len() > 1 && !stopwords.contains(*w))
            .map(|w| stemmer.stem(w).to_string())
            .collect();

        if keywords.is_empty() {
            return fact.clone();
        }

        // Search by each keyword stem; collect all candidate fact ids.
        let mut candidates: Vec<TemporalFact> = Vec::new();
        for kw in &keywords {
            if let Ok(found) = self.find_by_event(user_id, kw, 10) {
                for f in found {
                    if !candidates.iter().any(|c: &TemporalFact| c.id == f.id) {
                        candidates.push(f);
                    }
                }
            }
        }

        if candidates.is_empty() {
            // Forward reference — no matching event stored yet.  B2: leave Unknown.
            return fact.clone();
        }

        // Pick the most recent candidate by conversation_date.
        // For "after X" we want the event's latest date to anchor the lower bound.
        // For "before X" we want the event's earliest date to anchor the upper bound.
        let anchor_fact = if is_after {
            candidates
                .iter()
                .max_by_key(|f| f.conversation_date)
                .unwrap()
        } else {
            candidates
                .iter()
                .min_by_key(|f| f.conversation_date)
                .unwrap()
        };

        // Cycle guard: if anchor fact's id is already in the visited set, stop.
        if visited.contains(&anchor_fact.id) {
            return fact.clone();
        }

        // Recursively resolve the anchor fact if it is itself unresolved,
        // within the depth budget (depth+1).
        let anchor_resolved: ResolvedTime = match &anchor_fact.resolved_time {
            ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {
                // Recurse one level deeper to resolve the anchor.
                let resolved_anchor =
                    self.resolve_named_event_hop(user_id, anchor_fact, visited, depth + 1);
                resolved_anchor.resolved_time
            }
            other => other.clone(),
        };

        // Extract the anchor date from the resolved time.
        let anchor_date: Option<NaiveDate> = if is_after {
            // For "after X": use the LATEST day covered by the anchor's resolved time.
            match &anchor_resolved {
                ResolvedTime::ExactDate(d) => Some(*d),
                ResolvedTime::MonthYear { month, year } => last_day_of_month(*year, *month),
                ResolvedTime::Year(y) => NaiveDate::from_ymd_opt(*y, 12, 31),
                ResolvedTime::After(d) | ResolvedTime::Before(d) => Some(*d),
                ResolvedTime::Around { centre, .. } => Some(*centre),
                _ => None,
            }
        } else {
            // For "before X": use the EARLIEST day covered by the anchor's resolved time.
            match &anchor_resolved {
                ResolvedTime::ExactDate(d) => Some(*d),
                ResolvedTime::MonthYear { month, year } => {
                    NaiveDate::from_ymd_opt(*year, *month, 1)
                }
                ResolvedTime::Year(y) => NaiveDate::from_ymd_opt(*y, 1, 1),
                ResolvedTime::After(d) | ResolvedTime::Before(d) => Some(*d),
                ResolvedTime::Around { centre, .. } => Some(*centre),
                _ => None,
            }
        };

        match anchor_date {
            Some(d) => {
                let mut updated = fact.clone();
                updated.resolved_time = if is_after {
                    ResolvedTime::After(d)
                } else {
                    ResolvedTime::Before(d)
                };
                updated
            }
            None => fact.clone(),
        }
    }

    // =========================================================================
    // B2: shared event-to-date resolver (query-time + write-time)
    // =========================================================================

    /// Resolve a named-event phrase (e.g. "the wedding", "graduation") to a
    /// concrete `NaiveDate` by looking up the stored event in the database.
    ///
    /// This is the shared core used by **both** the write-time second-hop path
    /// (`resolve_named_event_hop`, R5) and the query-time anchor resolver (B2).
    ///
    /// **Algorithm**
    /// 1. Strip common stop-words and stem the remaining keywords.
    /// 2. Call `find_by_event` for each keyword; collect deduplicated candidate facts.
    /// 3. Pick the most recent candidate (by `conversation_date`) and extract its
    ///    date using `extract_anchor_date_from_resolved`.
    /// 4. If the candidate itself has an unresolved `resolved_time`, recurse once
    ///    (depth ≤ 2, `visited` cycle guard) — identical to the write-time logic.
    ///
    /// Returns `None` when:
    /// - No matching facts are stored (forward reference).
    /// - The cycle guard fires (`visited` already contains the candidate id).
    /// - The depth budget is exhausted.
    /// - The matched fact's date cannot be extracted.
    ///
    /// **B2 note**: The `visited` set must be caller-supplied so that call chains
    /// across write-time and query-time code share the same cycle boundary.
    pub fn resolve_event_to_date(
        &self,
        user_id: &str,
        event_phrase: &str,
        visited: &mut HashSet<String>,
        depth: u8,
    ) -> Option<NaiveDate> {
        if depth >= 2 {
            return None;
        }

        let stemmer = Stemmer::create(Algorithm::English);
        let stopwords: HashSet<&str> = [
            "the", "a", "an", "of", "in", "at", "on", "to", "for", "and", "or",
        ]
        .iter()
        .copied()
        .collect();

        let keywords: Vec<String> = event_phrase
            .split_whitespace()
            .filter(|w| w.len() > 1 && !stopwords.contains(*w))
            .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
            .collect();

        if keywords.is_empty() {
            return None;
        }

        // Collect deduplicated candidates via the event index.
        let mut candidates: Vec<TemporalFact> = Vec::new();
        for kw in &keywords {
            if let Ok(found) = self.find_by_event(user_id, kw, 10) {
                for f in found {
                    if !candidates.iter().any(|c: &TemporalFact| c.id == f.id) {
                        candidates.push(f);
                    }
                }
            }
        }

        if candidates.is_empty() {
            return None; // Forward reference — no matching event stored yet.
        }

        // Pick the most recent candidate by conversation_date.
        let anchor_fact = candidates.iter().max_by_key(|f| f.conversation_date)?;

        // Cycle guard.
        if visited.contains(&anchor_fact.id) {
            return None;
        }
        visited.insert(anchor_fact.id.clone());

        // If the anchor fact is itself unresolved, recurse one level deeper.
        let anchor_resolved: ResolvedTime = match &anchor_fact.resolved_time {
            ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {
                // Recurse: try to resolve the anchor's event phrase.
                let anchor_rel = anchor_fact.relative_time.as_deref().unwrap_or("");
                // Strip leading preposition if present (mirrors resolve_named_event_hop).
                let object = {
                    let lower = anchor_rel.to_lowercase();
                    let after_prefixes = ["since ", "from ", "after "];
                    let before_prefixes = ["prior to ", "before ", "until "];
                    if let Some(rest) = after_prefixes.iter().find_map(|p| lower.strip_prefix(p)) {
                        rest.to_string()
                    } else if let Some(rest) =
                        before_prefixes.iter().find_map(|p| lower.strip_prefix(p))
                    {
                        rest.to_string()
                    } else {
                        anchor_rel.to_string()
                    }
                };
                if let Some(d) = self.resolve_event_to_date(user_id, &object, visited, depth + 1) {
                    ResolvedTime::ExactDate(d)
                } else {
                    return None;
                }
            }
            other => other.clone(),
        };

        // Extract a single NaiveDate from the resolved time (take the earliest
        // for "before" anchors but for simple date-lookup callers we just use the
        // representative date — the exact day for ExactDate, first-of-period
        // for MonthYear/Year, centre for Around).
        let date = match &anchor_resolved {
            ResolvedTime::ExactDate(d) => Some(*d),
            ResolvedTime::MonthYear { month, year } => NaiveDate::from_ymd_opt(*year, *month, 1),
            ResolvedTime::Year(y) => NaiveDate::from_ymd_opt(*y, 7, 1),
            ResolvedTime::After(d) | ResolvedTime::Before(d) => Some(*d),
            ResolvedTime::Around { centre, .. } => Some(*centre),
            ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => None,
        };

        date
    }

    /// Apply named-event second-hop resolution to a batch of newly extracted
    /// facts (before they are stored).
    ///
    /// Each fact is resolved independently; the call is idempotent on facts
    /// that already have a concrete `resolved_time`.  The visited set is
    /// per-fact (not shared across facts) so facts in the batch do not
    /// influence each other's cycle detection.
    ///
    /// Facts that are already indexed (from prior store calls) may also be used
    /// as anchors — this is intentional and correct.  Facts in `new_facts` are
    /// NOT used as anchors for each other (forward references within the same
    /// batch stay `Unknown`) because they have not yet been indexed.
    pub fn resolve_named_event_hops(
        &self,
        user_id: &str,
        new_facts: Vec<TemporalFact>,
    ) -> Vec<TemporalFact> {
        new_facts
            .into_iter()
            .map(|fact| {
                let mut visited = HashSet::new();
                self.resolve_named_event_hop(user_id, &fact, &mut visited, 0)
            })
            .collect()
    }
}

// ============================================================================
// WORD-BOUNDARY MATCHING + INTENT INVENTORY  (precision layer)
// ============================================================================
//
// Raw `contains` matching silently misroutes: "ran" hits *orange*, "got" hits
// *forgot*, "soon" hits *monsoon*, "did" hits *candid*, "met" hits *helmet*.
// Everything below matches against whole tokens / token-windows so a phrase
// only fires when its words appear as words, never as substrings.

/// Tokenise a lowercased sentence into alphanumeric word tokens, preserving the
/// apostrophe so contractions (`don't`, `i'm`, `can't`) survive intact.
fn sentence_tokens(sentence_lower: &str) -> Vec<&str> {
    sentence_lower
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|t| !t.is_empty())
        .collect()
}

/// Return the token index at which `phrase` (one or more space-separated words)
/// appears as a contiguous run of whole tokens, or `None`. Word-boundary safe:
/// `contains_phrase(tokens("orange juice"), "ran")` is `None`.
fn contains_phrase(tokens: &[&str], phrase: &str) -> Option<usize> {
    let needle: Vec<&str> = phrase.split_whitespace().collect();
    if needle.is_empty() || needle.len() > tokens.len() {
        return None;
    }
    tokens
        .windows(needle.len())
        .position(|w| w == needle.as_slice())
}

/// True if any phrase in `phrases` appears as whole tokens.
fn any_phrase(tokens: &[&str], phrases: &[&str]) -> bool {
    phrases.iter().any(|p| contains_phrase(tokens, p).is_some())
}

/// Tense class of a matched intent construction. Drives bucket routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntentTense {
    /// want(s) to, will, planning to, going to (present aux), about to … → Planned.
    PresentFuture,
    /// wanted to, was going to, planned to, hoped to … → PastIntent bucket.
    PastIntent,
    /// decided/agreed/promised to … → decision is past, act usually future → Planned (lower conf).
    DecisionPast,
}

/// Intent-verb heads whose PAST simple/progressive form routes to PastIntent.
/// Mapped present→past so we can detect the past form by token match.
/// (head present form, head past form)
const PAST_INTENT_FORMS: &[&str] = &[
    "wanted", "hoped", "planned", "intended", "wished", "expected", "meant",
];

/// Decision-class heads: the decision is past but the act is usually future.
/// `decided to adopt a puppy` → a plan, not (yet) an occurrence.
const DECISION_FORMS: &[&str] = &["decided", "agreed", "promised"];

/// Auxiliaries that, when sitting ≤2 tokens before a `going to`/`gonna`/
/// `-ing` intent head, force the PAST-INTENT reading.
const PAST_AUX: &[&str] = &["was", "were", "wasn't", "weren't"];

/// Negators that kill an intent fact when within 3 tokens before the head.
/// `can't wait to` is whitelisted upstream (positive intent containing `can't`).
const NEGATORS: &[&str] = &[
    "not", "n't", "never", "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "cannot",
    "can't", "refuse", "refused", "no",
];

/// Subordinators that turn `get to` into the arrival sense
/// (`when I get to the office`) — skip the intent reading.
const ARRIVAL_SUBORDINATORS: &[&str] = &["when", "once", "after", "until", "before"];

/// Determiners following a commitment verb signal a noun complement
/// (`agreed to the terms`), not an event — skip.
const DETERMINERS: &[&str] = &[
    "the", "a", "an", "my", "his", "her", "their", "our", "this", "that",
];

/// Genuine future-intent phrases (multi-word, matched word-boundary). Present /
/// future tense → Planned. The PAST and DECISION forms are handled separately so
/// they route to the PastIntent / DecisionPast classes instead of confident-future.
const INTENT_PHRASES: &[&str] = &[
    // canonical plan/intent
    "planning to",
    "planning on",
    "plan to",
    "plans to",
    "going to",
    "gonna",
    "want to",
    "wants to",
    "wanting to",
    "wanna",
    "hope to",
    "hopes to",
    "hoping to",
    "intend to",
    "intends to",
    "intending to",
    "aim to",
    "aims to",
    "aiming to",
    "wish to",
    "wishes to",
    "wishing to",
    "expect to",
    "expects to",
    "expecting to",
    "thinking about",
    "thinking of",
    "looking forward to",
    "look forward to",
    "looks forward to",
    "excited to",
    "excited about",
    // polite / strong anticipation
    "would like to",
    "'d like to",
    "would love to",
    "'d love to",
    "can't wait to",
    "cannot wait to",
    // the requested addition + imminent/scheduled
    "get to",
    "gets to",
    "getting to",
    "about to",
    "set to",
    "scheduled to",
    "slated to",
    // deliberation
    "considering",
    "decide to",
    "decides to",
    "deciding to",
    "agree to",
    "agrees to",
    "promise to",
    "promises to",
    "promising to",
];

/// Future time markers (word-boundary). Presence alone (verb-less) is a weak
/// future signal kept for back-compat at lowered confidence.
const FUTURE_TIME_MARKERS: &[&str] = &[
    "next month",
    "next week",
    "next year",
    "next weekend",
    "this weekend",
    "tomorrow",
    "soon",
];

/// Phrases that look like intent but must NEVER trigger a Planned fact.
/// Checked and rejected before any INTENT_PHRASES match is honoured.
const INTENT_TRAPS: &[&str] = &[
    "due to",  // causal preposition ("cancelled due to rain")
    "used to", // past habitual ("used to run marathons")
    "have to",
    "has to",
    "had to", // obligation
    "need to",
    "needs to",
    "needed to",
    "got to",
    "gotta",       // obligation / past-opportunity
    "able to",     // capability
    "supposed to", // expectation (v1: exclude)
    "in order to",
    "so as to", // purpose adjunct
    "like to",
    "love to", // bare habitual preference
    "try to",
    "trying to",
    "tried to",
    "ought to",
    "go to",
    "happy to",
    "glad to",
    "willing to",
    "ready to",
];

/// Query-side symmetry: does a (lowercased) query phrase signal a *planned*
/// intent? Reuses the same INTENT_PHRASES inventory as extraction so the two
/// lists can never drift apart. Used by recall.rs to map queries to
/// `EventType::Planned`.
pub fn query_signals_planned_intent(query_lower: &str) -> bool {
    let tokens = sentence_tokens(query_lower);
    // Explicit planning words + the full intent inventory.
    any_phrase(&tokens, &["planning", "will"]) || any_phrase(&tokens, INTENT_PHRASES)
}

/// Classify the tense of an intent match given the sentence tokens.
/// Looks at past-form heads, decision heads, and a ≤2-token aux lookbehind for
/// `going to`/`gonna`/`-ing` heads.
fn classify_intent_tense(tokens: &[&str]) -> IntentTense {
    // Past-simple intent heads ("wanted to go", "planned to visit").
    if PAST_INTENT_FORMS
        .iter()
        .any(|f| contains_phrase(tokens, f).is_some())
    {
        return IntentTense::PastIntent;
    }
    // Decision class ("decided to", "agreed to", "promised to").
    if DECISION_FORMS
        .iter()
        .any(|f| contains_phrase(tokens, f).is_some())
    {
        return IntentTense::DecisionPast;
    }
    // Past progressive: `was/were … going to / gonna / planning / hoping …`.
    // Find a progressive/periphrastic head and check ≤2 tokens before it.
    let progressive_heads = ["going", "gonna", "planning", "hoping", "looking", "aiming"];
    for (idx, tok) in tokens.iter().enumerate() {
        if progressive_heads.contains(tok) {
            let lo = idx.saturating_sub(2);
            if tokens[lo..idx].iter().any(|t| PAST_AUX.contains(t)) {
                return IntentTense::PastIntent;
            }
        }
    }
    IntentTense::PresentFuture
}

/// Guard: is the intent head negated? Scans up to 3 tokens before the *first*
/// intent-phrase head. `can't wait to` is treated as positive upstream.
fn intent_is_negated(tokens: &[&str], head_idx: usize) -> bool {
    let lo = head_idx.saturating_sub(3);
    tokens[lo..head_idx].iter().any(|t| NEGATORS.contains(t))
}

/// Guard: arrival-sense `get to` (`when I get to the office`). True ⇒ skip.
fn is_arrival_get_to(tokens: &[&str]) -> bool {
    // Only relevant when a get-to family phrase is present.
    let head = ["get", "gets", "getting"];
    for (idx, tok) in tokens.iter().enumerate() {
        if head.contains(tok) && tokens.get(idx + 1) == Some(&"to") {
            let lo = idx.saturating_sub(2);
            if tokens[lo..idx]
                .iter()
                .any(|t| ARRIVAL_SUBORDINATORS.contains(t))
            {
                return true;
            }
        }
    }
    false
}

/// Guard: commitment verb followed by a determiner ⇒ noun complement, not an
/// event (`agreed to the terms`). True ⇒ skip.
fn is_noun_complement(tokens: &[&str]) -> bool {
    let commit = [
        "agree", "agrees", "agreed", "promise", "promises", "promised",
    ];
    for (idx, tok) in tokens.iter().enumerate() {
        if commit.contains(tok)
            && tokens.get(idx + 1) == Some(&"to")
            && tokens.get(idx + 2).is_some_and(|t| DETERMINERS.contains(t))
        {
            return true;
        }
    }
    false
}

/// Guard: perfect-infinitive complement (`excited to have met you`) references a
/// PAST event — do not treat as a future plan. True ⇒ skip the Planned reading.
fn is_perfect_infinitive(tokens: &[&str]) -> bool {
    tokens.windows(2).any(|w| w == ["to", "have"])
}

/// Guard: a second-person question (`Do you want to come hiking?`). The sentence
/// terminator is needed; passed in separately. True ⇒ skip (a question to the
/// entity is not a fact about them).
fn is_second_person_question(tokens: &[&str], terminator: char) -> bool {
    terminator == '?' && tokens.contains(&"you")
}

// ============================================================================
// TEMPORAL FACT EXTRACTION
// ============================================================================

/// Extract temporal facts from memory content
pub fn extract_temporal_facts(
    content: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    entities: &[String],
) -> Vec<TemporalFact> {
    let mut facts = Vec::new();
    let stemmer = Stemmer::create(Algorithm::English);

    // Split into sentences, RETAINING the terminator so question detection is
    // possible downstream (a `?` sentence directed at the entity is not a fact).
    let mut sentences: Vec<(&str, char)> = Vec::new();
    {
        let mut start = 0usize;
        for (idx, ch) in content.char_indices() {
            if ch == '.' || ch == '!' || ch == '?' {
                let seg = &content[start..idx];
                if !seg.trim().is_empty() {
                    sentences.push((seg, ch));
                }
                start = idx + ch.len_utf8();
            }
        }
        if start < content.len() {
            let seg = &content[start..];
            if !seg.trim().is_empty() {
                sentences.push((seg, '.'));
            }
        }
    }

    for (sentence, terminator) in sentences {
        let sentence_lower = sentence.to_lowercase();

        // Find which entity is mentioned in this sentence
        let mentioned_entity = entities
            .iter()
            .find(|e| sentence_lower.contains(&e.to_lowercase()));

        // GAP-3: If no NER entity matched, fall back to first-person/second-person
        // pronoun subjects when a clear time+event signal is present.  The
        // pronoun check is conservative: we only activate it when the sentence
        // contains a recognisable temporal expression AND a past/planning verb
        // form, to avoid creating noisy facts from unrelated pronoun sentences.
        let entity: String = if let Some(e) = mentioned_entity {
            e.clone()
        } else {
            // Require a time signal in the sentence.
            let has_time_signal = !extract_time_expression(&sentence_lower).is_empty()
                || parse_in_n_units(&sentence_lower).is_some()
                || extract_explicit_date_from_sentence(sentence, conversation_date).is_some();

            if !has_time_signal {
                continue;
            }

            // Require a past or planning verb form.
            let past_verbs = [
                "went",
                "ran",
                "did",
                "took",
                "made",
                "had",
                "got",
                "saw",
                "met",
                "was",
                "were",
                "attended",
                "visited",
                "started",
                "finished",
                "completed",
                "joined",
                "left",
            ];
            let planning_verbs = [
                "planning", "going", "thinking", "want", "hope", "will", "would", "am", "are", "is",
            ];
            let has_verb = past_verbs.iter().any(|v| {
                // word-boundary check: surrounded by non-alpha on both sides
                let lc = &sentence_lower;
                lc.split_whitespace()
                    .any(|w| w.trim_matches(|c: char| !c.is_alphabetic()) == *v)
            }) || planning_verbs.iter().any(|v| sentence_lower.contains(v));

            if !has_verb {
                continue;
            }

            // Map pronoun to generic entity label.
            if sentence_lower.split_whitespace().any(|w| {
                let w = w.trim_matches(|c: char| !c.is_alphabetic());
                w == "i" || w == "i've" || w == "i'm" || w == "i'd" || w == "i'll"
            }) {
                "speaker".to_string()
            } else if sentence_lower.split_whitespace().any(|w| {
                let w = w.trim_matches(|c: char| !c.is_alphabetic());
                w == "we" || w == "we've" || w == "we're" || w == "we'll"
            }) {
                "group".to_string()
            } else if sentence_lower.split_whitespace().any(|w| {
                let w = w.trim_matches(|c: char| !c.is_alphabetic());
                w == "you" || w == "you've" || w == "you're"
            }) {
                "listener".to_string()
            } else {
                continue;
            }
        };

        // Try to extract temporal facts using patterns
        if let Some(fact) = extract_planning_fact(
            sentence,
            terminator,
            &entity,
            memory_id,
            conversation_date,
            &stemmer,
        ) {
            facts.push(fact);
        } else if let Some(fact) =
            extract_occurred_fact(sentence, &entity, memory_id, conversation_date, &stemmer)
        {
            facts.push(fact);
        } else if let Some(fact) =
            extract_historical_fact(sentence, &entity, memory_id, conversation_date, &stemmer)
        {
            facts.push(fact);
        }
    }

    facts
}

/// Extract "planning"/intent facts (future events + past-intent).
///
/// Precision-first: word-boundary matching, an expanded intent inventory with
/// explicit trap exclusion, tense routing (past-intent away from confident
/// future), and a battery of deterministic guards.
fn extract_planning_fact(
    sentence: &str,
    terminator: char,
    entity: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    stemmer: &Stemmer,
) -> Option<TemporalFact> {
    let sentence_lower = sentence.to_lowercase();
    let tokens = sentence_tokens(&sentence_lower);

    // ---- Trap rejection (before any intent match is honoured) ---------------
    // due to / used to / obligation / capability / purpose / habitual-preference.
    // Whitelist `can't wait to` and `would like/love to`, which legitimately
    // contain trapped substrings, by checking traps as whole phrases only.
    let is_positive_wait = any_phrase(&tokens, &["can't wait to", "cannot wait to"]);
    if !is_positive_wait && any_phrase(&tokens, INTENT_TRAPS) {
        // A trap phrase is present. Only bail if NO clean intent phrase also
        // fires — e.g. "I want to, but I have to leave" should still extract.
        // For v1 we are conservative: a trap present and no future marker → drop.
        let clean_intent = any_phrase(&tokens, INTENT_PHRASES);
        if !clean_intent {
            return None;
        }
    }

    // ---- Intent / future detection (word-boundary) --------------------------
    let head_idx = INTENT_PHRASES
        .iter()
        .filter_map(|p| contains_phrase(&tokens, p))
        .min();
    let has_past_intent_form = PAST_INTENT_FORMS
        .iter()
        .any(|f| contains_phrase(&tokens, f).is_some())
        || DECISION_FORMS
            .iter()
            .any(|f| contains_phrase(&tokens, f).is_some());
    let has_future_marker = any_phrase(&tokens, FUTURE_TIME_MARKERS);
    // GAP-2: "in N days/weeks/months" is also a future indicator.
    let has_in_n = parse_in_n_units(&sentence_lower).is_some();

    let has_intent_verb = head_idx.is_some() || has_past_intent_form;
    if !has_intent_verb && !has_future_marker && !has_in_n {
        return None;
    }

    // ---- Guards (deterministic, O(n)) ---------------------------------------
    // Question to the entity is not a fact about them.
    if is_second_person_question(&tokens, terminator) {
        return None;
    }
    // Arrival-sense `get to` / commitment noun-complement → not an event.
    if is_arrival_get_to(&tokens) || is_noun_complement(&tokens) {
        // Only skip when the *only* intent signal was that construction.
        // If another clean intent phrase exists, fall through.
        let other = head_idx.is_some()
            && !any_phrase(&tokens, &["get to", "gets to", "getting to"])
            && !is_noun_complement(&tokens);
        if !other {
            return None;
        }
    }
    // Negation kills the intent (unless positive `can't wait to`).
    if let Some(idx) = head_idx {
        if !is_positive_wait && intent_is_negated(&tokens, idx) {
            return None;
        }
    }

    // ---- Tense routing ------------------------------------------------------
    // Perfect infinitive ("excited to have met you") → past reference: not a
    // confident future plan. Route to PastIntent so it is not surfaced as a plan.
    let perfect = is_perfect_infinitive(&tokens) && head_idx.is_some();
    let tense = if perfect {
        IntentTense::PastIntent
    } else {
        classify_intent_tense(&tokens)
    };

    // Extract event and time (word boundaries already applied to detection).
    let (event, relative_time) = extract_event_and_time(sentence, INTENT_PHRASES);
    if event.is_empty() {
        return None;
    }

    // Marker direction: does the resolved marker land in the past?
    let marker_resolved = if relative_time.is_empty() {
        ResolvedTime::Unknown
    } else {
        resolve_relative_time(&relative_time, conversation_date)
    };
    let conv_naive = conversation_date.date_naive();
    let marker_is_past = matches!(
        marker_resolved.to_bound(),
        TemporalBound::Point(d) if d < conv_naive
    ) || matches!(
        marker_resolved.to_bound(),
        TemporalBound::Closed { end, .. } if end < conv_naive
    ) || matches!(marker_resolved, ResolvedTime::Before(_));
    let has_marker = !matches!(marker_resolved.to_bound(), TemporalBound::Unknown);

    // Routing matrix (subset of plan §b — the load-bearing rows).
    let (event_type, resolved_time, confidence) = match tense {
        IntentTense::PresentFuture => {
            if has_marker && marker_is_past {
                // conflict: marker wins → Occurred ("want to mention I went yesterday").
                (EventType::Occurred, marker_resolved.clone(), 0.6)
            } else if has_marker {
                (EventType::Planned, marker_resolved.clone(), 0.85)
            } else {
                // Marker-less plan: forward of utterance (replaces junk empty desc).
                (EventType::Planned, ResolvedTime::After(conv_naive), 0.7)
            }
        }
        IntentTense::DecisionPast => {
            if has_marker && marker_is_past {
                (EventType::Occurred, marker_resolved.clone(), 0.65)
            } else if has_marker {
                (EventType::Planned, marker_resolved.clone(), 0.65)
            } else {
                (EventType::Planned, ResolvedTime::After(conv_naive), 0.65)
            }
        }
        IntentTense::PastIntent => {
            // Past intention, fulfilment unknown. Never confident-future.
            let rt = if has_marker {
                marker_resolved.clone()
            } else {
                ResolvedTime::Before(conv_naive)
            };
            (EventType::PastIntent, rt, 0.4)
        }
    };

    // Verb-less future-marker-only sentences get a lowered confidence.
    let confidence = if !has_intent_verb { 0.6 } else { confidence };

    let entity_stem = stemmer.stem(&entity.to_lowercase()).to_string();
    let event_stems: Vec<String> = event
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .filter(|s| s != &entity_stem)
        .collect();

    Some(TemporalFact {
        id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
        entity: entity.to_string(),
        event: event.clone(),
        event_stems,
        event_type,
        relative_time: if relative_time.is_empty() {
            None
        } else {
            Some(relative_time)
        },
        resolved_time,
        source_memory_id: memory_id.clone(),
        conversation_date,
        confidence,
        source_text: sentence.to_string(),
        valid_from: Some(conversation_date),
        valid_until: None,
    })
}

/// Extract "occurred" facts (past events)
fn extract_occurred_fact(
    sentence: &str,
    entity: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    stemmer: &Stemmer,
) -> Option<TemporalFact> {
    let sentence_lower = sentence.to_lowercase();

    // Past event patterns
    let occurred_patterns = [
        "last saturday",
        "last sunday",
        "last monday",
        "last tuesday",
        "last wednesday",
        "last thursday",
        "last friday",
        "last week",
        "last weekend",
        "last month",
        "yesterday",
        "this morning",
        "earlier today",
        "recently",
        "just",
        "ago",
        "two weeks ago",
        "a few days ago",
    ];

    // Check for past tense verbs combined with time patterns. Word-boundary
    // matching: "ran" must not fire on *orange*, "got" not on *forgot*, "did"
    // not on *candid*, "met" not on *helmet*, "had" not on *shadow*.
    let tokens = sentence_tokens(&sentence_lower);
    let past_verbs = [
        "ran", "went", "did", "took", "made", "had", "got", "saw", "met",
    ];
    // The occurred time-markers are multi-word/safe phrases except "just" and
    // "ago" which leak as substrings (adjust/justice, agony) — match those by
    // whole token too via contains_phrase.
    let has_past_event = occurred_patterns
        .iter()
        .any(|p| contains_phrase(&tokens, p).is_some())
        || past_verbs
            .iter()
            .any(|v| contains_phrase(&tokens, v).is_some());

    if !has_past_event {
        return None;
    }

    // Extract event and time
    let (event, relative_time) = extract_event_and_time(sentence, &occurred_patterns);
    if event.is_empty() {
        return None;
    }

    // Resolve relative time
    let resolved_time = resolve_relative_time(&relative_time, conversation_date);

    let entity_stem = stemmer.stem(&entity.to_lowercase()).to_string();
    let event_stems: Vec<String> = event
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .filter(|s| s != &entity_stem)
        .collect();

    Some(TemporalFact {
        id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
        entity: entity.to_string(),
        event: event.clone(),
        event_stems,
        event_type: EventType::Occurred,
        relative_time: if relative_time.is_empty() {
            None
        } else {
            Some(relative_time)
        },
        resolved_time,
        source_memory_id: memory_id.clone(),
        conversation_date,
        confidence: 0.75,
        source_text: sentence.to_string(),
        valid_from: Some(conversation_date),
        valid_until: None,
    })
}

/// Extract "historical" facts (events in distant past)
fn extract_historical_fact(
    sentence: &str,
    entity: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    stemmer: &Stemmer,
) -> Option<TemporalFact> {
    let sentence_lower = sentence.to_lowercase();

    // GAP-1: Try explicit date patterns first (full/partial dates → ExactDate/MonthYear).
    // This covers "May 8, 2023", "8 May 2023", "2023-05-08", "May 2023", "May 8" (conv year),
    // "8 May" (conv year).  Dates in the conversation year are treated as Occurred-class facts
    // here; dates strictly before the conversation year remain Historical.
    if let Some((resolved_time, time_text)) =
        extract_explicit_date_from_sentence(sentence, conversation_date)
    {
        // Only proceed if the date is not in the future relative to the conversation.
        let is_future = match &resolved_time {
            ResolvedTime::ExactDate(d) => *d > conversation_date.date_naive(),
            ResolvedTime::MonthYear { month, year } => {
                let conv_naive = conversation_date.date_naive();
                (*year > conv_naive.year())
                    || (*year == conv_naive.year() && *month > conv_naive.month())
            }
            _ => false,
        };
        if !is_future {
            let event = extract_event_from_sentence(sentence);
            if !event.is_empty() {
                let entity_stem = stemmer.stem(&entity.to_lowercase()).to_string();
                let event_stems: Vec<String> = event
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
                    .filter(|s| s != &entity_stem)
                    .collect();
                return Some(TemporalFact {
                    id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
                    entity: entity.to_string(),
                    event: event.clone(),
                    event_stems,
                    event_type: EventType::Historical,
                    relative_time: Some(time_text),
                    resolved_time,
                    source_memory_id: memory_id.clone(),
                    conversation_date,
                    confidence: 0.90,
                    source_text: sentence.to_string(),
                    valid_from: Some(conversation_date),
                    valid_until: None,
                });
            }
        }
    }

    // Fallback: bare 4-digit year reference (original behaviour — unchanged).
    let year_pattern = regex::Regex::new(r"\b(19|20)\d{2}\b").ok()?;
    let year_match = year_pattern.find(&sentence_lower)?;
    let year: i32 = year_match.as_str().parse().ok()?;

    // Only treat as historical if year is before conversation year
    if year >= conversation_date.year() {
        return None;
    }

    // Extract what happened
    let event = extract_event_from_sentence(sentence);
    if event.is_empty() {
        return None;
    }

    let entity_stem = stemmer.stem(&entity.to_lowercase()).to_string();
    let event_stems: Vec<String> = event
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .filter(|s| s != &entity_stem)
        .collect();

    Some(TemporalFact {
        id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
        entity: entity.to_string(),
        event: event.clone(),
        event_stems,
        event_type: EventType::Historical,
        relative_time: Some(year.to_string()),
        resolved_time: ResolvedTime::Year(year),
        source_memory_id: memory_id.clone(),
        conversation_date,
        confidence: 0.85,
        source_text: sentence.to_string(),
        valid_from: Some(conversation_date),
        valid_until: None,
    })
}

/// Extract event description and time expression from sentence
fn extract_event_and_time(sentence: &str, patterns: &[&str]) -> (String, String) {
    let sentence_lower = sentence.to_lowercase();

    // Find which pattern matched (used to verify a pattern exists)
    let _matched_pattern = patterns
        .iter()
        .find(|p| sentence_lower.contains(*p))
        .copied()
        .unwrap_or("");

    // Extract time expression
    let relative_time = extract_time_expression(&sentence_lower);

    // Extract event (words around the pattern)
    let event = extract_event_from_sentence(sentence);

    (event, relative_time)
}

/// Parse "in N days/weeks/months" future expressions.
///
/// Returns `Some((n, unit_in_days_or_none))` where:
/// - `unit_in_days_or_none = Some(days)` for day/week expressions,
/// - `unit_in_days_or_none = None` for month expressions (calendar-correct).
///
/// Returns `None` for unrecognised or malformed expressions.
fn parse_in_n_units(expr: &str) -> Option<(i64, InNUnit)> {
    // Match "in (a|an|N) (day|days|week|weeks|month|months)"
    let re = regex::Regex::new(
        r"(?i)\bin\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)\s+(day|days|week|weeks|month|months)\b",
    )
    .ok()?;
    let cap = re.captures(expr)?;

    let n: i64 = match cap[1].to_lowercase().as_str() {
        "a" | "an" | "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        s => s.parse().ok()?,
    };

    let unit = match cap[2].to_lowercase().as_str() {
        "day" | "days" => InNUnit::DaysOrWeeks(n),
        "week" | "weeks" => InNUnit::DaysOrWeeks(n * 7),
        "month" | "months" => InNUnit::Months,
        _ => return None,
    };

    Some((n, unit))
}

/// Unit classification returned by `parse_in_n_units`.
enum InNUnit {
    /// Calendar-month addition: resolve to `MonthYear`.
    Months,
    /// Day/week: exact day-count, resolve to `ExactDate`.
    DaysOrWeeks(i64),
}

/// Parse explicit absolute-date patterns from a sentence.
///
/// Handles (all case-insensitive):
///   - "Month Day, Year"  → ExactDate  ("May 8, 2023")
///   - "Day Month Year"   → ExactDate  ("8 May 2023")
///   - "YYYY-MM-DD"       → ExactDate
///   - "Month Year"       → MonthYear  ("May 2023")
///   - "Month Day" / "Day Month" (no year) → ExactDate anchored to conv year
///
/// Returns the *most precise* match found, or `None`.
fn extract_explicit_date_from_sentence(
    sentence: &str,
    conversation_date: DateTime<Utc>,
) -> Option<(ResolvedTime, String)> {
    let months = [
        ("january", 1u32),
        ("jan", 1),
        ("february", 2),
        ("feb", 2),
        ("march", 3),
        ("mar", 3),
        ("april", 4),
        ("apr", 4),
        ("may", 5),
        ("june", 6),
        ("jun", 6),
        ("july", 7),
        ("jul", 7),
        ("august", 8),
        ("aug", 8),
        ("september", 9),
        ("sep", 9),
        ("sept", 9),
        ("october", 10),
        ("oct", 10),
        ("november", 11),
        ("nov", 11),
        ("december", 12),
        ("dec", 12),
    ];

    let conv_year = conversation_date.year();

    // -- ISO: YYYY-MM-DD --
    if let Some(cap) = regex::Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b")
        .ok()?
        .captures(sentence)
    {
        let year: i32 = cap[1].parse().ok()?;
        let month: u32 = cap[2].parse().ok()?;
        let day: u32 = cap[3].parse().ok()?;
        if (1900..=2100).contains(&year) {
            if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                return Some((ResolvedTime::ExactDate(d), cap[0].to_string()));
            }
        }
    }

    let s_lower = sentence.to_lowercase();

    // -- Month Day, Year  OR  Month Day Year  ("May 8, 2023" / "May 8 2023") --
    for &(month_name, month_num) in &months {
        // "May 8, 2023" or "May 8 2023"
        let pat = format!(
            r"(?i)\b{}\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
            regex::escape(month_name)
        );
        if let Ok(re) = regex::Regex::new(&pat) {
            if let Some(cap) = re.captures(&s_lower) {
                let day: u32 = cap[1].parse().ok()?;
                let year: i32 = cap[2].parse().ok()?;
                if (1900..=2100).contains(&year) {
                    if let Some(d) = NaiveDate::from_ymd_opt(year, month_num, day) {
                        return Some((ResolvedTime::ExactDate(d), cap[0].to_string()));
                    }
                }
            }
        }
        // "8 May 2023" or "8 May, 2023"
        let pat2 = format!(
            r"(?i)\b(\d{{1,2}})\s+{},?\s+(\d{{4}})\b",
            regex::escape(month_name)
        );
        if let Ok(re) = regex::Regex::new(&pat2) {
            if let Some(cap) = re.captures(&s_lower) {
                let day: u32 = cap[1].parse().ok()?;
                let year: i32 = cap[2].parse().ok()?;
                if (1900..=2100).contains(&year) {
                    if let Some(d) = NaiveDate::from_ymd_opt(year, month_num, day) {
                        return Some((ResolvedTime::ExactDate(d), cap[0].to_string()));
                    }
                }
            }
        }
        // "May 2023" (month + year, no day)
        let pat3 = format!(r"(?i)\b{}\s+(\d{{4}})\b", regex::escape(month_name));
        if let Ok(re) = regex::Regex::new(&pat3) {
            if let Some(cap) = re.captures(&s_lower) {
                let year: i32 = cap[1].parse().ok()?;
                if (1900..=2100).contains(&year) {
                    return Some((
                        ResolvedTime::MonthYear {
                            month: month_num,
                            year,
                        },
                        cap[0].to_string(),
                    ));
                }
            }
        }
        // "May 8" or "8 May" (no year — anchor to conversation year)
        let pat4 = format!(r"(?i)\b{}\s+(\d{{1,2}})\b", regex::escape(month_name));
        if let Ok(re) = regex::Regex::new(&pat4) {
            if let Some(cap) = re.captures(&s_lower) {
                let day: u32 = cap[1].parse().ok()?;
                if let Some(d) = NaiveDate::from_ymd_opt(conv_year, month_num, day) {
                    return Some((ResolvedTime::ExactDate(d), cap[0].to_string()));
                }
            }
        }
        let pat5 = format!(r"(?i)\b(\d{{1,2}})\s+{}\b", regex::escape(month_name));
        if let Ok(re) = regex::Regex::new(&pat5) {
            if let Some(cap) = re.captures(&s_lower) {
                let day: u32 = cap[1].parse().ok()?;
                if let Some(d) = NaiveDate::from_ymd_opt(conv_year, month_num, day) {
                    return Some((ResolvedTime::ExactDate(d), cap[0].to_string()));
                }
            }
        }
    }

    None
}

/// Extract time expression from sentence
fn extract_time_expression(sentence: &str) -> String {
    // Ordered most-specific-first: the loop returns the first substring match,
    // so longer phrases that CONTAIN a shorter one must precede it. In
    // particular "* weekend" must precede "* week" (it contains it), and the
    // multi-word day offsets / "earlier today" must precede the bare
    // "yesterday"/"tomorrow"/"today" tokens.
    //
    // Inferential prepositional patterns are placed before the phrases they
    // contain, e.g. "after next week" before "next week".
    let time_patterns = [
        // multi-word day offsets (contain "yesterday"/"tomorrow")
        "the day before yesterday",
        "the day after tomorrow",
        "earlier today",
        // --- inferential: "prior to" must precede "before" (substring) ---
        "prior to next year",
        "prior to last year",
        "prior to this year",
        "prior to next month",
        "prior to last month",
        "prior to this month",
        "prior to next week",
        "prior to last week",
        "prior to yesterday",
        "prior to tomorrow",
        "prior to today",
        // inferential: "since/from/after" + standard phrases
        "since last year",
        "since last month",
        "since last week",
        "since yesterday",
        "after next year",
        "after last year",
        "after next month",
        "after last month",
        "after next week",
        "after last week",
        "after yesterday",
        "after tomorrow",
        "after today",
        "from last year",
        "from last month",
        "from last week",
        "from yesterday",
        "from today",
        // inferential: "before" + standard phrases (must come after "prior to")
        "before next year",
        "before last year",
        "before next month",
        "before last month",
        "before next week",
        "before last week",
        "before yesterday",
        "before tomorrow",
        "before today",
        // inferential: "until" + standard phrases
        "until next year",
        "until last year",
        "until next month",
        "until last month",
        "until next week",
        "until last week",
        "until yesterday",
        "until tomorrow",
        "until today",
        // inferential: "around" + standard phrases
        "around next year",
        "around last year",
        "around next month",
        "around last month",
        "around next week",
        "around last week",
        "around yesterday",
        "around tomorrow",
        "around today",
        // weekends (contain "week")
        "last weekend",
        "next weekend",
        "this weekend",
        // months
        "next month",
        "last month",
        "this month",
        // years
        "next year",
        "last year",
        "this year",
        // weekdays
        "last saturday",
        "last sunday",
        "last monday",
        "last tuesday",
        "last wednesday",
        "last thursday",
        "last friday",
        // weeks
        "next week",
        "last week",
        // time-of-day (anchor to conversation day)
        "this morning",
        "this afternoon",
        "this evening",
        "tonight",
        // bare day tokens
        "yesterday",
        "today",
        "tomorrow",
        // explicit "ago" phrases
        "two weeks ago",
        "a few days ago",
        "a week ago",
        // bare "recently"
        "recently",
        // near-future fuzzy marker (resolves to Around conv_date+7d, radius 7)
        // so a "soon" plan stops producing a junk RelativeDescription("").
        "soon",
    ];

    // Word-boundary match: every pattern here is whole words, so a boundary
    // check is strictly safer and stops "soon" matching *monsoon*, "today"
    // matching *todays*, etc., without changing any genuine phrase match.
    let pat_tokens = sentence_tokens(sentence);
    for pattern in time_patterns {
        if contains_phrase(&pat_tokens, pattern).is_some() {
            return pattern.to_string();
        }
    }

    // Check for "X ago" pattern
    if let Some(idx) = sentence.find(" ago") {
        let before = &sentence[..idx];
        let words: Vec<&str> = before.split_whitespace().collect();
        if words.len() >= 2 {
            let last_two = words[words.len() - 2..].join(" ");
            return format!("{} ago", last_two);
        }
    }

    // GAP-2: "in N days/weeks/months" future expressions
    if let Some(cap) = regex::Regex::new(
        r"(?i)\bin\s+(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)\s+(?:day|days|week|weeks|month|months)\b",
    )
    .ok()
    .and_then(|re| re.find(sentence))
    {
        return cap.as_str().to_string();
    }

    String::new()
}

/// Extract main event/action from sentence
fn extract_event_from_sentence(sentence: &str) -> String {
    let stopwords: HashSet<&str> = [
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "it",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
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
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
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
        "about",
        "really",
        "also",
        "that",
        "this",
        "these",
        "those",
        "am",
        "going",
        "planning",
        "thinking",
        "want",
        "hope",
        "looking",
        "forward",
        "excited",
        // expanded intent heads (S7) — keep these out of event_stems so the new
        // inventory does not pollute matching with the verb itself.
        "wanted",
        "hoped",
        "planned",
        "intend",
        "intends",
        "intended",
        "aim",
        "aims",
        "aiming",
        "wish",
        "wishes",
        "wished",
        "expect",
        "expects",
        "expecting",
        "expected",
        "decide",
        "decides",
        "decided",
        "agree",
        "agrees",
        "agreed",
        "promise",
        "promises",
        "promised",
        "gonna",
        "wanna",
        "considering",
        "supposed",
        "scheduled",
        "slated",
        "get",
        "gets",
        "getting",
        "set",
        "meant",
        "last",
        "next",
        "week",
        "month",
        "year",
        "saturday",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "weekend",
        "ago",
        "recently",
    ]
    .into_iter()
    .collect();

    // Extract content words
    let words: Vec<&str> = sentence
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let w_lower = w.to_lowercase();
            w.len() > 2 && !stopwords.contains(w_lower.as_str())
        })
        .take(5)
        .collect();

    words.join(" ")
}

// ============================================================================
// RELATIVE DATE RESOLUTION
// ============================================================================

/// Resolve relative time expression to absolute time
pub fn resolve_relative_time(relative: &str, conversation_date: DateTime<Utc>) -> ResolvedTime {
    let relative_lower = relative.to_lowercase();
    let conv_date = conversation_date.date_naive();

    // -----------------------------------------------------------------------
    // INFERENTIAL PREPOSITIONS — checked before all other branches.
    //
    // Recognise a leading preposition, strip it, recursively resolve the
    // remainder, then wrap in the appropriate inferential variant.
    //
    // Ordering: longer/more-specific prefixes must precede shorter ones that
    // are substrings (e.g. "prior to " before "to ", "since " before bare).
    //
    // If the recursive resolution yields `Unknown` the whole expression is
    // `Unknown` — no degenerate half-formed bounds are emitted.
    // -----------------------------------------------------------------------

    // "since <X>" / "from <X>" / "after <X>"
    for prefix in &["since ", "from ", "after "] {
        if let Some(rest) = relative_lower.strip_prefix(prefix) {
            let inner = resolve_relative_time(rest, conversation_date);
            return match &inner {
                ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {
                    ResolvedTime::Unknown
                }
                ResolvedTime::ExactDate(d) => ResolvedTime::After(*d),
                ResolvedTime::MonthYear { month, year } => {
                    // Take the LATEST day of the month as the anchor.
                    match last_day_of_month(*year, *month) {
                        Some(d) => ResolvedTime::After(d),
                        None => ResolvedTime::Unknown,
                    }
                }
                ResolvedTime::Year(y) => {
                    // Latest day of the year.
                    match NaiveDate::from_ymd_opt(*y, 12, 31) {
                        Some(d) => ResolvedTime::After(d),
                        None => ResolvedTime::Unknown,
                    }
                }
                // Propagate inferential — After(After(x)) collapses to After(x).
                ResolvedTime::After(d) => ResolvedTime::After(*d),
                ResolvedTime::Before(d) => ResolvedTime::Before(*d),
                ResolvedTime::Around {
                    centre,
                    radius_days,
                } => ResolvedTime::Around {
                    centre: *centre,
                    radius_days: *radius_days,
                },
            };
        }
    }

    // "prior to <X>" / "before <X>" / "until <X>"
    for prefix in &["prior to ", "before ", "until "] {
        if let Some(rest) = relative_lower.strip_prefix(prefix) {
            let inner = resolve_relative_time(rest, conversation_date);
            return match &inner {
                ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {
                    ResolvedTime::Unknown
                }
                ResolvedTime::ExactDate(d) => ResolvedTime::Before(*d),
                ResolvedTime::MonthYear { month, year } => {
                    // Take the EARLIEST day of the month as the anchor.
                    match NaiveDate::from_ymd_opt(*year, *month, 1) {
                        Some(d) => ResolvedTime::Before(d),
                        None => ResolvedTime::Unknown,
                    }
                }
                ResolvedTime::Year(y) => {
                    // Earliest day of the year.
                    match NaiveDate::from_ymd_opt(*y, 1, 1) {
                        Some(d) => ResolvedTime::Before(d),
                        None => ResolvedTime::Unknown,
                    }
                }
                ResolvedTime::After(d) => ResolvedTime::After(*d),
                ResolvedTime::Before(d) => ResolvedTime::Before(*d),
                ResolvedTime::Around {
                    centre,
                    radius_days,
                } => ResolvedTime::Around {
                    centre: *centre,
                    radius_days: *radius_days,
                },
            };
        }
    }

    // "around <X>" / "~<X>"
    for prefix in &["around ", "~"] {
        if let Some(rest) = relative_lower.strip_prefix(prefix) {
            let inner = resolve_relative_time(rest, conversation_date);
            let centre = match &inner {
                ResolvedTime::ExactDate(d) => Some(*d),
                ResolvedTime::MonthYear { month, year } => {
                    NaiveDate::from_ymd_opt(*year, *month, 1)
                }
                ResolvedTime::Year(y) => NaiveDate::from_ymd_opt(*y, 7, 1),
                _ => None,
            };
            return match centre {
                Some(c) => ResolvedTime::Around {
                    centre: c,
                    radius_days: 7,
                },
                None => ResolvedTime::Unknown,
            };
        }
    }

    // "recently" → fuzzy centred on the conversation date, ±14 days.
    if relative_lower.trim() == "recently" {
        return ResolvedTime::Around {
            centre: conv_date,
            radius_days: 14,
        };
    }

    // "soon" → fuzzy near-future window centred a week ahead, ±7 days.
    // Replaces the prior junk RelativeDescription("") for "soon" plans.
    if relative_lower.trim() == "soon" {
        return ResolvedTime::Around {
            centre: conv_date + Duration::days(7),
            radius_days: 7,
        };
    }

    // Multi-word day offsets — MUST precede the bare "yesterday"/"tomorrow"
    // branches below, which they contain as substrings.
    if relative_lower.contains("the day before yesterday") {
        return ResolvedTime::ExactDate(conv_date - Duration::days(2));
    }
    if relative_lower.contains("the day after tomorrow") {
        return ResolvedTime::ExactDate(conv_date + Duration::days(2));
    }

    // "next month"
    if relative_lower.contains("next month") {
        let next_month = if conv_date.month() == 12 {
            (1, conv_date.year() + 1)
        } else {
            (conv_date.month() + 1, conv_date.year())
        };
        return ResolvedTime::MonthYear {
            month: next_month.0,
            year: next_month.1,
        };
    }

    // "last month"
    if relative_lower.contains("last month") {
        let last_month = if conv_date.month() == 1 {
            (12, conv_date.year() - 1)
        } else {
            (conv_date.month() - 1, conv_date.year())
        };
        return ResolvedTime::MonthYear {
            month: last_month.0,
            year: last_month.1,
        };
    }

    // "this month"
    if relative_lower.contains("this month") {
        return ResolvedTime::MonthYear {
            month: conv_date.month(),
            year: conv_date.year(),
        };
    }

    // "next year" / "last year" / "this year"
    if relative_lower.contains("next year") {
        return ResolvedTime::Year(conv_date.year() + 1);
    }
    if relative_lower.contains("last year") {
        return ResolvedTime::Year(conv_date.year() - 1);
    }
    if relative_lower.contains("this year") {
        return ResolvedTime::Year(conv_date.year());
    }

    // "next week"
    if relative_lower.contains("next week") {
        let next_week = conv_date + Duration::days(7);
        return ResolvedTime::ExactDate(next_week);
    }

    // "last week"
    if relative_lower.contains("last week") {
        let last_week = conv_date - Duration::days(7);
        return ResolvedTime::ExactDate(last_week);
    }

    // "yesterday"
    if relative_lower.contains("yesterday") {
        return ResolvedTime::ExactDate(conv_date - Duration::days(1));
    }

    // "today"
    if relative_lower.contains("today") {
        return ResolvedTime::ExactDate(conv_date);
    }

    // "tomorrow"
    if relative_lower.contains("tomorrow") {
        return ResolvedTime::ExactDate(conv_date + Duration::days(1));
    }

    // "last saturday", "last sunday", etc.
    let weekdays = [
        ("sunday", Weekday::Sun),
        ("monday", Weekday::Mon),
        ("tuesday", Weekday::Tue),
        ("wednesday", Weekday::Wed),
        ("thursday", Weekday::Thu),
        ("friday", Weekday::Fri),
        ("saturday", Weekday::Sat),
    ];

    for (name, weekday) in weekdays {
        if relative_lower.contains(&format!("last {}", name)) {
            let resolved = last_weekday(conv_date, weekday);
            return ResolvedTime::ExactDate(resolved);
        }
        if relative_lower.contains(&format!("this {}", name))
            || relative_lower.contains(&format!("next {}", name))
        {
            let resolved = next_weekday(conv_date, weekday);
            return ResolvedTime::ExactDate(resolved);
        }
    }

    // "N months/years ago" → calendar-aware MonthYear / Year (F8).
    // "N days/weeks ago"   → ExactDate (fine as before).
    if relative_lower.contains("ago") {
        if let Some((n, unit)) = parse_ago_unit(&relative_lower) {
            match unit {
                AgoUnit::Months => {
                    let total_months = conv_date.month() as i64 - n;
                    let (year, month) = if total_months <= 0 {
                        let years_back = ((-total_months) / 12) + 1;
                        let month = ((total_months + years_back * 12) % 12) as u32;
                        let month = if month == 0 { 12 } else { month };
                        (conv_date.year() - years_back as i32, month)
                    } else {
                        (conv_date.year(), total_months as u32)
                    };
                    return ResolvedTime::MonthYear { month, year };
                }
                AgoUnit::Years => {
                    return ResolvedTime::Year(conv_date.year() - n as i32);
                }
                AgoUnit::DaysOrWeeks(days) => {
                    return ResolvedTime::ExactDate(conv_date - Duration::days(days));
                }
            }
        }
    }

    // "last weekend" — Saturday of the previous weekend.
    if relative_lower.contains("last weekend") {
        return ResolvedTime::ExactDate(last_weekday(conv_date, Weekday::Sat));
    }

    // "next weekend" — Saturday of the upcoming weekend.
    if relative_lower.contains("next weekend") {
        return ResolvedTime::ExactDate(next_weekday(conv_date, Weekday::Sat));
    }

    // "this weekend"
    if relative_lower.contains("this weekend") {
        let days_until_saturday = (Weekday::Sat.num_days_from_monday() as i64
            - conv_date.weekday().num_days_from_monday() as i64
            + 7)
            % 7;
        let saturday = conv_date + Duration::days(days_until_saturday);
        return ResolvedTime::ExactDate(saturday);
    }

    // Time-of-day phrases anchor to the conversation day (sub-day precision is
    // not modeled). "earlier today" is already covered by the "today" branch.
    if relative_lower.contains("this morning")
        || relative_lower.contains("this afternoon")
        || relative_lower.contains("this evening")
        || relative_lower.contains("tonight")
    {
        return ResolvedTime::ExactDate(conv_date);
    }

    // Year only (e.g., "2022")
    if let Ok(year) = relative_lower.trim().parse::<i32>() {
        if (1900..2100).contains(&year) {
            return ResolvedTime::Year(year);
        }
    }

    // GAP-2: "in N days/weeks/months" — future offsets from conversation_date
    if relative_lower.contains("in ") {
        if let Some((n, unit)) = parse_in_n_units(&relative_lower) {
            match unit {
                InNUnit::Months => {
                    let total_months = conv_date.month() as i64 + n;
                    let extra_years = (total_months - 1) / 12;
                    let month = ((total_months - 1) % 12 + 1) as u32;
                    let year = conv_date.year() + extra_years as i32;
                    return ResolvedTime::MonthYear { month, year };
                }
                InNUnit::DaysOrWeeks(days) => {
                    return ResolvedTime::ExactDate(conv_date + Duration::days(days));
                }
            }
        }
    }

    ResolvedTime::RelativeDescription(relative.to_string())
}

/// Find the last occurrence of a weekday before the given date
pub fn last_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let current_weekday = from.weekday();
    let days_back =
        (current_weekday.num_days_from_monday() as i64 - target.num_days_from_monday() as i64 + 7)
            % 7;
    let days_back = if days_back == 0 { 7 } else { days_back };
    from - Duration::days(days_back)
}

/// Find the next occurrence of a weekday after the given date
pub fn next_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let current_weekday = from.weekday();
    let days_forward =
        (target.num_days_from_monday() as i64 - current_weekday.num_days_from_monday() as i64 + 7)
            % 7;
    let days_forward = if days_forward == 0 { 7 } else { days_forward };
    from + Duration::days(days_forward)
}

/// Unit classification returned by `parse_ago_unit`.
enum AgoUnit {
    /// Calendar-month subtraction: resolve to `MonthYear`.
    Months,
    /// Calendar-year subtraction: resolve to `Year`.
    Years,
    /// Day/week: already an exact day-count, resolve to `ExactDate`.
    DaysOrWeeks(i64),
}

/// Parse "N days/weeks/months/years ago" and return the numeric count and unit.
///
/// Month and year expressions are returned as `AgoUnit::Months` / `AgoUnit::Years`
/// so callers can do calendar-correct subtraction (F8). Day and week expressions
/// are folded into a day count (`AgoUnit::DaysOrWeeks`).
///
/// Returns `None` for unrecognised or malformed expressions.
fn parse_ago_unit(expr: &str) -> Option<(i64, AgoUnit)> {
    let words: Vec<&str> = expr.split_whitespace().collect();

    let ago_pos = words.iter().position(|w| *w == "ago")?;
    if ago_pos < 2 {
        return None;
    }

    let unit = words[ago_pos - 1];
    let num_str = words[ago_pos - 2];

    let n: i64 = match num_str {
        "a" | "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "few" => 3,
        "couple" => 2,
        _ => num_str.parse().ok()?,
    };

    let ago_unit = match unit {
        "day" | "days" => AgoUnit::DaysOrWeeks(n),
        "week" | "weeks" => AgoUnit::DaysOrWeeks(n * 7),
        "month" | "months" => AgoUnit::Months,
        "year" | "years" => AgoUnit::Years,
        _ => return None,
    };

    Some((n, ago_unit))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a store test on every backend.
    ///
    /// These tests built a rocksdb DB directly, so the point-in-time recall this
    /// module exists for had never executed against redb even though
    /// `with_store` has accepted either engine for months. A body that takes the
    /// store by value cannot leak state between the two runs.
    fn on_both_backends(body: fn(TemporalFactStore)) {
        let stores = crate::storage::test_support::key_value_stores("temporal_facts_test");
        crate::storage::test_support::assert_both_backends(&stores);
        for (_backend, kv, _keyspace) in stores {
            body(TemporalFactStore::with_store(kv));
        }
    }

    #[test]
    fn test_resolve_next_month() {
        let conv_date = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let resolved = resolve_relative_time("next month", conv_date);

        match resolved {
            ResolvedTime::MonthYear { month, year } => {
                assert_eq!(month, 6);
                assert_eq!(year, 2023);
            }
            _ => panic!("Expected MonthYear"),
        }
    }

    fn conv() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    /// I1: indicators that were previously extracted but unresolved must now
    /// resolve to a concrete value, not RelativeDescription/Unknown.
    #[test]
    fn test_resolve_newly_supported_indicators() {
        // "last year" / "next year" / "this year"
        assert!(matches!(
            resolve_relative_time("last year", conv()),
            ResolvedTime::Year(2022)
        ));
        assert!(matches!(
            resolve_relative_time("next year", conv()),
            ResolvedTime::Year(2024)
        ));
        assert!(matches!(
            resolve_relative_time("this year", conv()),
            ResolvedTime::Year(2023)
        ));
        // "last weekend" / "next weekend" resolve to a concrete date, not a
        // RelativeDescription (the old behaviour).
        assert!(matches!(
            resolve_relative_time("last weekend", conv()),
            ResolvedTime::ExactDate(_)
        ));
        assert!(matches!(
            resolve_relative_time("next weekend", conv()),
            ResolvedTime::ExactDate(_)
        ));
        // time-of-day anchors to the conversation day
        assert!(matches!(
            resolve_relative_time("this morning", conv()),
            ResolvedTime::ExactDate(d) if d == conv().date_naive()
        ));
        // multi-word day offsets must NOT be swallowed by yesterday/tomorrow
        assert!(matches!(
            resolve_relative_time("the day before yesterday", conv()),
            ResolvedTime::ExactDate(d) if d == conv().date_naive() - Duration::days(2)
        ));
        assert!(matches!(
            resolve_relative_time("the day after tomorrow", conv()),
            ResolvedTime::ExactDate(d) if d == conv().date_naive() + Duration::days(2)
        ));
    }

    /// Boy-scout: "last weekend" must extract as itself, not be shadowed by the
    /// "last week" substring (which precedes it in older orderings).
    #[test]
    fn test_extract_weekend_not_shadowed_by_week() {
        assert_eq!(
            extract_time_expression("we went hiking last weekend"),
            "last weekend"
        );
    }

    /// I4/B1: to_bound projects each ResolvedTime variant into a TemporalBound,
    /// and never panics on a corrupt MonthYear/Year.
    #[test]
    fn test_to_bound_projection() {
        let d = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        assert_eq!(
            ResolvedTime::ExactDate(d).to_bound(),
            TemporalBound::Point(d)
        );

        match (ResolvedTime::MonthYear {
            month: 6,
            year: 2023,
        })
        .to_bound()
        {
            TemporalBound::Closed { start, end } => {
                assert_eq!(start, NaiveDate::from_ymd_opt(2023, 6, 1).unwrap());
                assert_eq!(end, NaiveDate::from_ymd_opt(2023, 6, 30).unwrap());
            }
            other => panic!("expected Closed, got {other:?}"),
        }
        match ResolvedTime::Year(2022).to_bound() {
            TemporalBound::Closed { start, end } => {
                assert_eq!(start, NaiveDate::from_ymd_opt(2022, 1, 1).unwrap());
                assert_eq!(end, NaiveDate::from_ymd_opt(2022, 12, 31).unwrap());
            }
            other => panic!("expected Closed, got {other:?}"),
        }
        assert_eq!(
            ResolvedTime::RelativeDescription("soon".into()).to_bound(),
            TemporalBound::Unknown
        );
        // B1: corrupt month must degrade to Unknown, not panic.
        assert_eq!(
            (ResolvedTime::MonthYear {
                month: 13,
                year: 2023
            })
            .to_bound(),
            TemporalBound::Unknown
        );
    }

    /// J3 + F5 refutation: a Point overlaps a window; a December Closed spans
    /// the year boundary correctly; inverted windows never match.
    #[test]
    fn test_overlaps_range() {
        let jun = |d| NaiveDate::from_ymd_opt(2023, 6, d).unwrap();
        let p = TemporalBound::Point(jun(15));
        assert!(p.overlaps_range(jun(10), jun(20)));
        assert!(!p.overlaps_range(jun(1), jun(10)));
        // inverted window guarded
        assert!(!p.overlaps_range(jun(20), jun(10)));

        // December MonthYear last-day computation crosses into next year.
        match (ResolvedTime::MonthYear {
            month: 12,
            year: 2023,
        })
        .to_bound()
        {
            TemporalBound::Closed { end, .. } => {
                assert_eq!(end, NaiveDate::from_ymd_opt(2023, 12, 31).unwrap());
            }
            other => panic!("expected Closed, got {other:?}"),
        }

        // F5 refutation: a "next week" fact (single ExactDate point) is still
        // caught by a week-scale query window thanks to the ±7 widening.
        let next_week = match resolve_relative_time("next week", conv()).to_bound() {
            TemporalBound::Point(d) => d,
            other => panic!("expected Point, got {other:?}"),
        };
        let qstart = next_week - Duration::days(3);
        let qend = next_week + Duration::days(3);
        assert!(TemporalBound::Point(next_week).overlaps_range(qstart, qend));
    }

    #[test]
    fn test_resolve_last_saturday() {
        // May 25, 2023 is a Thursday
        let conv_date = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let resolved = resolve_relative_time("last saturday", conv_date);

        match resolved {
            ResolvedTime::ExactDate(d) => {
                // Last Saturday before May 25 (Thursday) is May 20
                assert_eq!(d.to_string(), "2023-05-20");
            }
            _ => panic!("Expected ExactDate"),
        }
    }

    #[test]
    fn test_extract_event() {
        let sentence = "We're thinking about going camping next month";
        let event = extract_event_from_sentence(sentence);
        assert!(event.to_lowercase().contains("camping"));
    }

    /// Point-in-time recall over TemporalFacts: a fact invalidated at T2
    /// must be visible via `as_of(T_between)` but absent via `as_of(now)`.
    #[test]
    fn test_temporal_fact_as_of_returns_historical_state() {
        on_both_backends(test_temporal_fact_as_of_returns_historical_state_body);
    }

    fn test_temporal_fact_as_of_returns_historical_state_body(store: TemporalFactStore) {
        let now = Utc::now();
        let t1 = now - chrono::Duration::days(10);
        let t2 = now - chrono::Duration::days(5);
        let t_between = now - chrono::Duration::days(7);

        let memory_id = MemoryId(uuid::Uuid::new_v4());

        // Fact A: planned camping next month — valid from t1, invalidated at t2
        let fact_a = TemporalFact {
            id: "tf-a".to_string(),
            entity: "alice".to_string(),
            event: "camping".to_string(),
            event_stems: vec!["camp".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("next month".to_string()),
            resolved_time: ResolvedTime::Year(2025),
            source_memory_id: memory_id.clone(),
            conversation_date: t1,
            confidence: 0.8,
            source_text: "Alice is planning to go camping next month".to_string(),
            valid_from: Some(t1),
            valid_until: Some(t2),
        };
        store.store("user-1", &fact_a).unwrap();

        // Fact B: cancellation — valid from t2 onward
        let fact_b = TemporalFact {
            id: "tf-b".to_string(),
            entity: "alice".to_string(),
            event: "cancelled camping".to_string(),
            event_stems: vec!["cancel".to_string(), "camp".to_string()],
            event_type: EventType::Occurred,
            relative_time: None,
            resolved_time: ResolvedTime::Year(2025),
            source_memory_id: memory_id,
            conversation_date: t2,
            confidence: 0.85,
            source_text: "Alice cancelled the camping trip".to_string(),
            valid_from: Some(t2),
            valid_until: None,
        };
        store.store("user-1", &fact_b).unwrap();

        let at_between = store.as_of("user-1", t_between, 100).unwrap();
        assert_eq!(at_between.len(), 1, "between t1 and t2: only A");
        assert_eq!(at_between[0].id, "tf-a");

        let at_now = store.as_of("user-1", now, 100).unwrap();
        assert_eq!(at_now.len(), 1, "now: only B");
        assert_eq!(at_now[0].id, "tf-b");

        let before_t1 = store
            .as_of("user-1", t1 - chrono::Duration::days(1), 100)
            .unwrap();
        assert!(before_t1.is_empty(), "before t1: nothing yet");
    }

    // -------------------------------------------------------------------------
    // F8: "N months ago" → MonthYear (calendar-correct, no 30-day drift)
    // -------------------------------------------------------------------------

    /// "2 months ago" from May 2023 → MonthYear { month: 3, year: 2023 }
    #[test]
    fn test_two_months_ago_resolves_to_month_year() {
        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        match resolve_relative_time("2 months ago", conv) {
            ResolvedTime::MonthYear { month, year } => {
                assert_eq!(month, 3, "2 months before May is March");
                assert_eq!(year, 2023);
            }
            other => panic!("expected MonthYear, got {other:?}"),
        }
    }

    /// "2 months ago" crossing a year boundary: Feb 2023 → Dec 2022
    #[test]
    fn test_months_ago_year_wraparound() {
        let conv = DateTime::parse_from_rfc3339("2023-01-15T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        match resolve_relative_time("2 months ago", conv) {
            ResolvedTime::MonthYear { month, year } => {
                assert_eq!(month, 11, "2 months before Jan is Nov");
                assert_eq!(year, 2022);
            }
            other => panic!("expected MonthYear, got {other:?}"),
        }
    }

    /// "3 years ago" → Year(conv_year - 3)
    #[test]
    fn test_three_years_ago_resolves_to_year() {
        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        match resolve_relative_time("3 years ago", conv) {
            ResolvedTime::Year(y) => assert_eq!(y, 2020),
            other => panic!("expected Year, got {other:?}"),
        }
    }

    /// Days/weeks "ago" still produces ExactDate (no regression).
    #[test]
    fn test_days_and_weeks_ago_still_exact_date() {
        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(matches!(
            resolve_relative_time("3 days ago", conv),
            ResolvedTime::ExactDate(_)
        ));
        assert!(matches!(
            resolve_relative_time("2 weeks ago", conv),
            ResolvedTime::ExactDate(_)
        ));
    }

    // -------------------------------------------------------------------------
    // F9: entity name must not appear in event_stems
    // -------------------------------------------------------------------------

    /// When extract_planning_fact processes "Melanie is planning to go camping
    /// next month", the stem of "Melanie" must not be in event_stems.
    #[test]
    fn test_event_stems_do_not_contain_entity_name() {
        use super::super::types::MemoryId;
        use rust_stemmers::{Algorithm, Stemmer};

        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let memory_id = MemoryId(uuid::Uuid::new_v4());
        let stemmer = Stemmer::create(Algorithm::English);
        let entity = "Melanie";

        let fact = extract_planning_fact(
            "Melanie is planning to go camping next month",
            '.',
            entity,
            &memory_id,
            conv,
            &stemmer,
        )
        .expect("should extract a planning fact");

        let entity_stem = stemmer.stem(&entity.to_lowercase()).to_string();
        assert!(
            !fact.event_stems.contains(&entity_stem),
            "entity stem '{entity_stem}' must not appear in event_stems: {:?}",
            fact.event_stems
        );
    }

    // =========================================================================
    // R4: Inferential After/Before/Around — resolution, to_bound, overlaps_range
    // =========================================================================

    // R4 tests reuse the conv() helper defined earlier in this test module.

    // -------------------------------------------------------------------------
    // R4-1: phrase → ResolvedTime variant
    // -------------------------------------------------------------------------

    /// "after yesterday" → After(conv_date - 1)
    #[test]
    fn test_inferential_after_yesterday() {
        let expected = conv().date_naive() - Duration::days(1);
        match resolve_relative_time("after yesterday", conv()) {
            ResolvedTime::After(d) => assert_eq!(d, expected),
            other => panic!("expected After, got {other:?}"),
        }
    }

    /// "since last week" → After(conv_date - 7)
    #[test]
    fn test_inferential_since_last_week() {
        let expected = conv().date_naive() - Duration::days(7);
        match resolve_relative_time("since last week", conv()) {
            ResolvedTime::After(d) => assert_eq!(d, expected),
            other => panic!("expected After, got {other:?}"),
        }
    }

    /// "from last month" → After(last day of April 2023 = Apr 30)
    #[test]
    fn test_inferential_from_last_month() {
        // conv is May 25, 2023 → last month = April 2023 → last day = Apr 30
        let expected = NaiveDate::from_ymd_opt(2023, 4, 30).unwrap();
        match resolve_relative_time("from last month", conv()) {
            ResolvedTime::After(d) => assert_eq!(d, expected),
            other => panic!("expected After, got {other:?}"),
        }
    }

    /// "before next month" → Before(first day of June 2023 = Jun 1)
    #[test]
    fn test_inferential_before_next_month() {
        let expected = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        match resolve_relative_time("before next month", conv()) {
            ResolvedTime::Before(d) => assert_eq!(d, expected),
            other => panic!("expected Before, got {other:?}"),
        }
    }

    /// "prior to last year" → Before(Jan 1 of 2022)
    #[test]
    fn test_inferential_prior_to_last_year() {
        let expected = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();
        match resolve_relative_time("prior to last year", conv()) {
            ResolvedTime::Before(d) => assert_eq!(d, expected),
            other => panic!("expected Before, got {other:?}"),
        }
    }

    /// "until tomorrow" → Before(conv_date + 1)
    #[test]
    fn test_inferential_until_tomorrow() {
        let expected = conv().date_naive() + Duration::days(1);
        match resolve_relative_time("until tomorrow", conv()) {
            ResolvedTime::Before(d) => assert_eq!(d, expected),
            other => panic!("expected Before, got {other:?}"),
        }
    }

    /// "around yesterday" → Around { centre: conv_date-1, radius_days: 7 }
    #[test]
    fn test_inferential_around_yesterday() {
        let expected_centre = conv().date_naive() - Duration::days(1);
        match resolve_relative_time("around yesterday", conv()) {
            ResolvedTime::Around {
                centre,
                radius_days,
            } => {
                assert_eq!(centre, expected_centre);
                assert_eq!(radius_days, 7);
            }
            other => panic!("expected Around, got {other:?}"),
        }
    }

    /// "recently" → Around { centre: conv_date, radius_days: 14 }
    #[test]
    fn test_inferential_recently() {
        match resolve_relative_time("recently", conv()) {
            ResolvedTime::Around {
                centre,
                radius_days,
            } => {
                assert_eq!(centre, conv().date_naive());
                assert_eq!(radius_days, 14);
            }
            other => panic!("expected Around for 'recently', got {other:?}"),
        }
    }

    /// "after <unknown>" → Unknown (no degenerate bound emitted)
    #[test]
    fn test_inferential_after_unknown_gives_unknown() {
        match resolve_relative_time("after something-unparseable-xyz", conv()) {
            ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_) => {} // pass
            other => panic!("expected Unknown/RelativeDescription, got {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // R4-2: to_bound mapping
    // -------------------------------------------------------------------------

    #[test]
    fn test_to_bound_after() {
        let d = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        assert_eq!(ResolvedTime::After(d).to_bound(), TemporalBound::After(d));
    }

    #[test]
    fn test_to_bound_before() {
        let d = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        assert_eq!(ResolvedTime::Before(d).to_bound(), TemporalBound::Before(d));
    }

    #[test]
    fn test_to_bound_around() {
        let c = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        assert_eq!(
            ResolvedTime::Around {
                centre: c,
                radius_days: 7
            }
            .to_bound(),
            TemporalBound::Fuzzy {
                centre: c,
                radius_days: 7
            }
        );
    }

    // -------------------------------------------------------------------------
    // R4-3: overlaps_range for new variants including boundary cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_overlaps_range_after() {
        let anchor = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        let bound = TemporalBound::After(anchor);
        let jun = |d: u32| NaiveDate::from_ymd_opt(2023, 6, d).unwrap();

        // window entirely after anchor → overlaps
        assert!(bound.overlaps_range(jun(5), jun(20)));
        // window ending exactly on anchor → overlaps
        assert!(bound.overlaps_range(jun(1), jun(1)));
        // window straddling anchor → overlaps
        assert!(bound.overlaps_range(NaiveDate::from_ymd_opt(2023, 5, 25).unwrap(), jun(5)));
        // window entirely before anchor → no overlap
        assert!(!bound.overlaps_range(
            NaiveDate::from_ymd_opt(2023, 5, 1).unwrap(),
            NaiveDate::from_ymd_opt(2023, 5, 31).unwrap()
        ));
        // inverted window → no overlap
        assert!(!bound.overlaps_range(jun(20), jun(5)));
    }

    #[test]
    fn test_overlaps_range_before() {
        let anchor = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        let bound = TemporalBound::Before(anchor);
        let may = |d: u32| NaiveDate::from_ymd_opt(2023, 5, d).unwrap();
        let jun = |d: u32| NaiveDate::from_ymd_opt(2023, 6, d).unwrap();

        // window entirely before anchor → overlaps
        assert!(bound.overlaps_range(may(1), may(31)));
        // window starting exactly on anchor → overlaps
        assert!(bound.overlaps_range(jun(1), jun(1)));
        // window entirely after anchor → no overlap
        assert!(!bound.overlaps_range(jun(2), jun(30)));
        // inverted window → no overlap
        assert!(!bound.overlaps_range(may(31), may(1)));
    }

    #[test]
    fn test_overlaps_range_fuzzy() {
        // centre = Jun 15, radius = 7 → fuzzy window [Jun 8, Jun 22]
        let centre = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        let bound = TemporalBound::Fuzzy {
            centre,
            radius_days: 7,
        };
        let jun = |d: u32| NaiveDate::from_ymd_opt(2023, 6, d).unwrap();

        // entirely inside fuzzy window
        assert!(bound.overlaps_range(jun(10), jun(20)));
        // touching left boundary
        assert!(bound.overlaps_range(jun(1), jun(8)));
        // touching right boundary
        assert!(bound.overlaps_range(jun(22), jun(30)));
        // entirely to the left of fuzzy window
        assert!(!bound.overlaps_range(jun(1), jun(7)));
        // entirely to the right of fuzzy window
        assert!(!bound.overlaps_range(jun(23), jun(30)));
        // inverted window → no overlap
        assert!(!bound.overlaps_range(jun(20), jun(10)));
    }

    // -------------------------------------------------------------------------
    // R4-4: sortable-string non-collision
    // -------------------------------------------------------------------------

    /// After(June 1) and ExactDate(June 1) MUST produce different sortable strings.
    #[test]
    fn test_sortable_string_no_collision_after_vs_exact() {
        let d = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        let exact_str = ResolvedTime::ExactDate(d).to_sortable_string();
        let after_str = ResolvedTime::After(d).to_sortable_string();
        assert_ne!(
            exact_str, after_str,
            "After({d}) and ExactDate({d}) must not produce the same sortable string"
        );
        assert!(
            after_str.starts_with("after:"),
            "After sortable string must be namespaced: {after_str}"
        );
        assert_eq!(exact_str, "2023-06-01");
        assert_eq!(after_str, "after:2023-06-01");
    }

    /// Before(June 1) must also not collide with ExactDate(June 1).
    #[test]
    fn test_sortable_string_no_collision_before_vs_exact() {
        let d = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        let exact_str = ResolvedTime::ExactDate(d).to_sortable_string();
        let before_str = ResolvedTime::Before(d).to_sortable_string();
        assert_ne!(exact_str, before_str);
        assert!(before_str.starts_with("before:"));
        assert_eq!(before_str, "before:2023-06-01");
    }

    /// Around sortable string format is correct.
    #[test]
    fn test_sortable_string_around_format() {
        let c = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        let s = ResolvedTime::Around {
            centre: c,
            radius_days: 7,
        }
        .to_sortable_string();
        assert_eq!(s, "around:2023-06-15:7d");
    }

    // -------------------------------------------------------------------------
    // R4-5: bincode round-trip for new variants + old-fact decode safety
    // -------------------------------------------------------------------------

    fn make_fact(resolved_time: ResolvedTime, conversation_date: DateTime<Utc>) -> TemporalFact {
        use super::super::types::MemoryId;
        TemporalFact {
            id: "test-fact-id".to_string(),
            entity: "alice".to_string(),
            event: "test event".to_string(),
            event_stems: vec!["test".to_string()],
            event_type: EventType::Planned,
            relative_time: None,
            resolved_time,
            source_memory_id: MemoryId(uuid::Uuid::nil()),
            conversation_date,
            confidence: 0.9,
            source_text: "test source text".to_string(),
            valid_from: None,
            valid_until: None,
        }
    }

    /// Encode and decode each new ResolvedTime variant; decoded value must equal encoded.
    #[test]
    fn test_bincode_roundtrip_new_variants() {
        let conv = conv();
        let d = NaiveDate::from_ymd_opt(2023, 6, 1).unwrap();
        let centre = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();

        let cases: Vec<ResolvedTime> = vec![
            ResolvedTime::After(d),
            ResolvedTime::Before(d),
            ResolvedTime::Around {
                centre,
                radius_days: 7,
            },
        ];

        for rt in cases {
            let fact = make_fact(rt, conv);
            let encoded = bincode::serde::encode_to_vec(&fact, bincode::config::standard())
                .expect("encode must succeed");
            let (decoded, _): (TemporalFact, _) =
                bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
                    .expect("decode must succeed");

            // Verify the sortable string round-trips correctly.
            assert_eq!(
                fact.resolved_time.to_sortable_string(),
                decoded.resolved_time.to_sortable_string(),
                "sortable string must be identical after round-trip"
            );
            assert_eq!(fact.id, decoded.id);
            assert_eq!(fact.entity, decoded.entity);
        }
    }

    /// Old-fact variants (ExactDate, MonthYear, Year, RelativeDescription, Unknown)
    /// must still decode correctly — verifies append-only bincode safety.
    #[test]
    fn test_bincode_old_fact_decode_unchanged() {
        let conv = conv();
        let d = NaiveDate::from_ymd_opt(2023, 5, 20).unwrap();

        let old_variants: Vec<ResolvedTime> = vec![
            ResolvedTime::ExactDate(d),
            ResolvedTime::MonthYear {
                month: 5,
                year: 2023,
            },
            ResolvedTime::Year(2022),
            ResolvedTime::RelativeDescription("some relative phrase".to_string()),
            ResolvedTime::Unknown,
        ];

        for rt in old_variants {
            let fact = make_fact(rt, conv);
            let encoded = bincode::serde::encode_to_vec(&fact, bincode::config::standard())
                .expect("encode must succeed");
            let (decoded, _): (TemporalFact, _) =
                bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
                    .expect("old-fact decode must still succeed after R4 additions");
            assert_eq!(
                fact.resolved_time.to_sortable_string(),
                decoded.resolved_time.to_sortable_string(),
                "old variant sortable string must be unchanged after round-trip"
            );
        }
    }

    // =========================================================================
    // R5: named-event second-hop resolution
    // =========================================================================
    //
    // All three tests below use the TempDir+DB+TemporalFactStore pattern from
    // `test_temporal_fact_as_of_returns_historical_state`.
    //
    // Design constraints verified:
    //   B2 — write-time only: forward-ref (event stored AFTER the referencing
    //        fact) stays Unknown (test b).
    //   B3 — cycle-safe: A "after B" + B "after A" terminates without
    //        hang/overflow (test c).

    /// (a) A "wedding" fact is stored first (ExactDate).  A second fact with
    /// relative_time "after the wedding" resolves to After(wedding_date).
    #[test]
    fn test_r5_named_event_resolves_after_wedding() {
        on_both_backends(test_r5_named_event_resolves_after_wedding_body);
    }

    fn test_r5_named_event_resolves_after_wedding_body(store: TemporalFactStore) {
        let conv = conv();
        let wedding_date = NaiveDate::from_ymd_opt(2023, 3, 15).unwrap();
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        // Store the wedding fact first.
        let wedding_fact = TemporalFact {
            id: "tf-wedding".to_string(),
            entity: "alice".to_string(),
            event: "wedding".to_string(),
            event_stems: vec!["wed".to_string()],
            event_type: EventType::Occurred,
            relative_time: Some("2023-03-15".to_string()),
            resolved_time: ResolvedTime::ExactDate(wedding_date),
            source_memory_id: memory_id.clone(),
            conversation_date: conv,
            confidence: 0.9,
            source_text: "Alice had her wedding on March 15, 2023".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };
        store.store("user-r5a", &wedding_fact).unwrap();

        // Now create a fact that references "after the wedding" (Unknown resolved).
        let after_fact = TemporalFact {
            id: "tf-after-wedding".to_string(),
            entity: "alice".to_string(),
            event: "honeymoon".to_string(),
            event_stems: vec!["honeymoon".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the wedding".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id,
            conversation_date: conv,
            confidence: 0.8,
            source_text: "Alice is planning the honeymoon after the wedding".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        let mut visited = std::collections::HashSet::new();
        let resolved = store.resolve_named_event_hop("user-r5a", &after_fact, &mut visited, 0);

        match resolved.resolved_time {
            ResolvedTime::After(d) => {
                assert_eq!(d, wedding_date, "should anchor on wedding date");
            }
            other => panic!("expected After(wedding_date), got {other:?}"),
        }
    }

    /// (b) Forward reference: the referencing fact is resolved BEFORE the
    /// wedding fact exists.  The result must stay Unknown (B2: no cascade).
    #[test]
    fn test_r5_forward_ref_stays_unknown() {
        on_both_backends(test_r5_forward_ref_stays_unknown_body);
    }

    fn test_r5_forward_ref_stays_unknown_body(store: TemporalFactStore) {
        let conv = conv();
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        // No wedding fact stored — this is the forward-reference scenario.
        let after_fact = TemporalFact {
            id: "tf-after-wedding-fwd".to_string(),
            entity: "alice".to_string(),
            event: "honeymoon".to_string(),
            event_stems: vec!["honeymoon".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the wedding".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id,
            conversation_date: conv,
            confidence: 0.8,
            source_text: "Alice is planning the honeymoon after the wedding".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        let mut visited = std::collections::HashSet::new();
        let resolved = store.resolve_named_event_hop("user-r5b", &after_fact, &mut visited, 0);

        // Must remain Unknown — no cascade, no back-fill.
        assert!(
            matches!(
                resolved.resolved_time,
                ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_)
            ),
            "forward ref must stay Unknown, got {:?}",
            resolved.resolved_time
        );
    }

    // =========================================================================
    // B2: resolve_event_to_date — shared helper tests
    // =========================================================================

    /// (a) "wedding" fact stored at a known date → query-time resolver returns
    /// that date; "the Sunday before the wedding" resolves to the Sunday before
    /// the stored wedding date.
    #[test]
    fn test_b2_resolve_event_to_date_wedding() {
        on_both_backends(test_b2_resolve_event_to_date_wedding_body);
    }

    fn test_b2_resolve_event_to_date_wedding_body(store: TemporalFactStore) {
        let conv = conv();
        // Wedding on May 25, 2023 (Thursday).
        let wedding_date = NaiveDate::from_ymd_opt(2023, 5, 25).unwrap();
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        let wedding_fact = TemporalFact {
            id: "tf-b2-wedding".to_string(),
            entity: "alice".to_string(),
            event: "wedding".to_string(),
            event_stems: vec!["wed".to_string()],
            event_type: EventType::Occurred,
            relative_time: Some("2023-05-25".to_string()),
            resolved_time: ResolvedTime::ExactDate(wedding_date),
            source_memory_id: memory_id.clone(),
            conversation_date: conv,
            confidence: 0.9,
            source_text: "Alice had her wedding on May 25, 2023".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };
        store.store("user-b2a", &wedding_fact).unwrap();

        // Resolve "the wedding" event phrase.
        let mut visited = std::collections::HashSet::new();
        let result = store.resolve_event_to_date("user-b2a", "the wedding", &mut visited, 0);

        assert_eq!(
            result,
            Some(wedding_date),
            "resolve_event_to_date must return the wedding date"
        );

        // Verify "the Sunday before the wedding" gives 2023-05-21.
        let sunday_before = last_weekday(wedding_date, chrono::Weekday::Sun);
        assert_eq!(
            sunday_before,
            NaiveDate::from_ymd_opt(2023, 5, 21).unwrap(),
            "Sunday before 2023-05-25 must be 2023-05-21"
        );
    }

    /// (b) Forward reference: no wedding fact stored → resolve_event_to_date → None.
    #[test]
    fn test_b2_forward_ref_returns_none() {
        on_both_backends(test_b2_forward_ref_returns_none_body);
    }

    fn test_b2_forward_ref_returns_none_body(store: TemporalFactStore) {
        let mut visited = std::collections::HashSet::new();
        let result = store.resolve_event_to_date("user-b2b", "the wedding", &mut visited, 0);

        assert_eq!(
            result, None,
            "Forward reference must return None from resolve_event_to_date"
        );
    }

    /// (c) Cycle guard: two events referencing each other must terminate and
    /// yield None (depth cap + visited set).
    #[test]
    fn test_b2_cycle_terminates_in_resolve_event_to_date() {
        on_both_backends(test_b2_cycle_terminates_in_resolve_event_to_date_body);
    }

    fn test_b2_cycle_terminates_in_resolve_event_to_date_body(store: TemporalFactStore) {
        let conv = conv();
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        // Store fact A (event "move", resolved Unknown, relative "after the party")
        let fact_a = TemporalFact {
            id: "tf-b2c-a".to_string(),
            entity: "bob".to_string(),
            event: "move".to_string(),
            event_stems: vec!["move".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the party".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id.clone(),
            conversation_date: conv,
            confidence: 0.7,
            source_text: "Bob is moving after the party".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        // Store fact B (event "party", resolved Unknown, relative "after the move")
        let fact_b = TemporalFact {
            id: "tf-b2c-b".to_string(),
            entity: "bob".to_string(),
            event: "party".to_string(),
            event_stems: vec!["parti".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the move".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id,
            conversation_date: conv,
            confidence: 0.7,
            source_text: "Bob will have a party after the move".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        store.store("user-b2c", &fact_a).unwrap();
        store.store("user-b2c", &fact_b).unwrap();

        // Both lookups must terminate without panic.
        let mut vis_a = std::collections::HashSet::new();
        let r_a = store.resolve_event_to_date("user-b2c", "the party", &mut vis_a, 0);
        let mut vis_b = std::collections::HashSet::new();
        let r_b = store.resolve_event_to_date("user-b2c", "the move", &mut vis_b, 0);

        // Both should return None (cycle terminates — no concrete date).
        assert!(
            r_a.is_none(),
            "Cycle: lookup for 'the party' must terminate and return None, got {r_a:?}"
        );
        assert!(
            r_b.is_none(),
            "Cycle: lookup for 'the move' must terminate and return None, got {r_b:?}"
        );
    }

    /// (d) R5 behavior is unchanged after helper extraction: "after the wedding"
    /// still resolves to After(wedding_date) in resolve_named_event_hop.
    #[test]
    fn test_b2_r5_behavior_unchanged_after_helper_extraction() {
        on_both_backends(test_b2_r5_behavior_unchanged_after_helper_extraction_body);
    }

    fn test_b2_r5_behavior_unchanged_after_helper_extraction_body(store: TemporalFactStore) {
        let conv = conv();
        let wedding_date = NaiveDate::from_ymd_opt(2023, 3, 15).unwrap();
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        let wedding_fact = TemporalFact {
            id: "tf-b2d-wedding".to_string(),
            entity: "alice".to_string(),
            event: "wedding".to_string(),
            event_stems: vec!["wed".to_string()],
            event_type: EventType::Occurred,
            relative_time: Some("2023-03-15".to_string()),
            resolved_time: ResolvedTime::ExactDate(wedding_date),
            source_memory_id: memory_id.clone(),
            conversation_date: conv,
            confidence: 0.9,
            source_text: "Alice had her wedding on March 15, 2023".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };
        store.store("user-b2d", &wedding_fact).unwrap();

        let after_fact = TemporalFact {
            id: "tf-b2d-honeymoon".to_string(),
            entity: "alice".to_string(),
            event: "honeymoon".to_string(),
            event_stems: vec!["honeymoon".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the wedding".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id,
            conversation_date: conv,
            confidence: 0.8,
            source_text: "Alice is planning the honeymoon after the wedding".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        let mut visited = std::collections::HashSet::new();
        let resolved = store.resolve_named_event_hop("user-b2d", &after_fact, &mut visited, 0);

        match resolved.resolved_time {
            ResolvedTime::After(d) => {
                assert_eq!(d, wedding_date, "R5 must still anchor on wedding_date");
            }
            other => {
                panic!("R5 behavior must be unchanged: expected After(wedding_date), got {other:?}")
            }
        }
    }

    // =========================================================================
    // GAP-1: Explicit date extraction — ExactDate / MonthYear
    // =========================================================================

    /// "Caroline attended on 8 May 2023" with conv date 2023-05-25 →
    /// Historical fact with ExactDate(2023-05-08) (full date) or MonthYear(5,2023).
    #[test]
    fn test_gap1_explicit_date_full() {
        use super::super::types::MemoryId;
        use rust_stemmers::{Algorithm, Stemmer};

        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let memory_id = MemoryId(uuid::Uuid::new_v4());
        let stemmer = Stemmer::create(Algorithm::English);
        let entity = "Caroline";

        let fact = extract_historical_fact(
            "Caroline attended on 8 May 2023",
            entity,
            &memory_id,
            conv,
            &stemmer,
        )
        .expect("should extract a fact for '8 May 2023'");

        assert!(
            matches!(
                fact.resolved_time,
                ResolvedTime::ExactDate(d) if d == NaiveDate::from_ymd_opt(2023, 5, 8).unwrap()
            ),
            "expected ExactDate(2023-05-08), got {:?}",
            fact.resolved_time
        );
    }

    /// "Caroline attended on May 2023" → MonthYear { month: 5, year: 2023 }.
    #[test]
    fn test_gap1_explicit_date_month_year() {
        use super::super::types::MemoryId;
        use rust_stemmers::{Algorithm, Stemmer};

        let conv = DateTime::parse_from_rfc3339("2023-07-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let memory_id = MemoryId(uuid::Uuid::new_v4());
        let stemmer = Stemmer::create(Algorithm::English);
        let entity = "Caroline";

        let fact = extract_historical_fact(
            "Caroline attended the event in May 2023",
            entity,
            &memory_id,
            conv,
            &stemmer,
        )
        .expect("should extract a fact for 'May 2023'");

        assert!(
            matches!(
                fact.resolved_time,
                ResolvedTime::MonthYear {
                    month: 5,
                    year: 2023
                }
            ),
            "expected MonthYear(5, 2023), got {:?}",
            fact.resolved_time
        );
    }

    // =========================================================================
    // GAP-2: "in N units" future extraction
    // =========================================================================

    /// "the conference is in 3 weeks" with conv date 2023-06-01 →
    /// planning fact with ExactDate(~2023-06-22).
    #[test]
    fn test_gap2_in_n_weeks_future() {
        use super::super::types::MemoryId;
        use rust_stemmers::{Algorithm, Stemmer};

        let conv = DateTime::parse_from_rfc3339("2023-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let memory_id = MemoryId(uuid::Uuid::new_v4());
        let stemmer = Stemmer::create(Algorithm::English);
        let entity = "conference";

        let fact = extract_planning_fact(
            "the conference is in 3 weeks",
            '.',
            entity,
            &memory_id,
            conv,
            &stemmer,
        )
        .expect("should extract a planning fact for 'in 3 weeks'");

        let expected = NaiveDate::from_ymd_opt(2023, 6, 22).unwrap();
        assert!(
            matches!(fact.resolved_time, ResolvedTime::ExactDate(d) if d == expected),
            "expected ExactDate(2023-06-22), got {:?}",
            fact.resolved_time
        );
    }

    /// resolve_relative_time: "in a month" from 2023-06-01 → MonthYear { month: 7, year: 2023 }.
    #[test]
    fn test_gap2_in_a_month_resolves() {
        let conv = DateTime::parse_from_rfc3339("2023-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        match resolve_relative_time("in a month", conv) {
            ResolvedTime::MonthYear { month, year } => {
                assert_eq!(month, 7, "one month after June is July");
                assert_eq!(year, 2023);
            }
            other => panic!("expected MonthYear(7, 2023), got {other:?}"),
        }
    }

    /// resolve_relative_time: "in 2 days" from 2023-06-01 → ExactDate(2023-06-03).
    #[test]
    fn test_gap2_in_n_days_resolves() {
        let conv = DateTime::parse_from_rfc3339("2023-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let expected = NaiveDate::from_ymd_opt(2023, 6, 3).unwrap();
        match resolve_relative_time("in 2 days", conv) {
            ResolvedTime::ExactDate(d) => assert_eq!(d, expected),
            other => panic!("expected ExactDate(2023-06-03), got {other:?}"),
        }
    }

    // =========================================================================
    // GAP-3: Pronoun-entity fallback
    // =========================================================================

    /// "I got my first two turtles three years ago" with NO 'speaker' in entities
    /// → still creates a fact via pronoun fallback with entity = "speaker",
    /// resolved to Year(conv_year - 3).
    #[test]
    fn test_gap3_pronoun_fallback_no_entity() {
        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        // Empty entity list — no NER match possible.
        let facts = extract_temporal_facts(
            "I got my first two turtles three years ago",
            &memory_id,
            conv,
            &[],
        );

        assert!(
            !facts.is_empty(),
            "pronoun fallback must produce at least one fact"
        );
        let fact = &facts[0];
        assert_eq!(fact.entity, "speaker", "fallback entity must be 'speaker'");
        assert!(
            matches!(fact.resolved_time, ResolvedTime::Year(2020)),
            "3 years before 2023 = Year(2020), got {:?}",
            fact.resolved_time
        );
    }

    /// GAP-3 does not fire on a sentence with NO time signal (no noise).
    #[test]
    fn test_gap3_no_spurious_fact_without_time_signal() {
        let conv = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        let facts = extract_temporal_facts(
            "I really enjoy hiking in the mountains",
            &memory_id,
            conv,
            &[],
        );

        assert!(
            facts.is_empty(),
            "no time signal → no pronoun-fallback fact; got {facts:?}"
        );
    }

    /// (c) Cycle: A references "after B event", B references "after A event".
    /// Must terminate (no stack overflow / hang) and both end up Unknown.
    #[test]
    fn test_r5_cycle_terminates() {
        on_both_backends(test_r5_cycle_terminates_body);
    }

    fn test_r5_cycle_terminates_body(store: TemporalFactStore) {
        let conv = conv();
        let memory_id = super::super::types::MemoryId(uuid::Uuid::new_v4());

        // Fact A: event "party", relative "after the move".
        let fact_a = TemporalFact {
            id: "tf-cycle-a".to_string(),
            entity: "bob".to_string(),
            event: "party".to_string(),
            event_stems: vec!["parti".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the move".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id.clone(),
            conversation_date: conv,
            confidence: 0.7,
            source_text: "Bob will have a party after the move".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        // Fact B: event "move", relative "after the party".
        let fact_b = TemporalFact {
            id: "tf-cycle-b".to_string(),
            entity: "bob".to_string(),
            event: "move".to_string(),
            event_stems: vec!["move".to_string()],
            event_type: EventType::Planned,
            relative_time: Some("after the party".to_string()),
            resolved_time: ResolvedTime::Unknown,
            source_memory_id: memory_id,
            conversation_date: conv,
            confidence: 0.7,
            source_text: "Bob is moving after the party".to_string(),
            valid_from: Some(conv),
            valid_until: None,
        };

        // Store both facts so each lookup finds the other.
        store.store("user-r5c", &fact_a).unwrap();
        store.store("user-r5c", &fact_b).unwrap();

        // Attempt to resolve fact_a — must not hang or overflow.
        let mut visited_a = std::collections::HashSet::new();
        let resolved_a = store.resolve_named_event_hop("user-r5c", &fact_a, &mut visited_a, 0);

        // Attempt to resolve fact_b — must not hang or overflow.
        let mut visited_b = std::collections::HashSet::new();
        let resolved_b = store.resolve_named_event_hop("user-r5c", &fact_b, &mut visited_b, 0);

        // Both must remain Unknown (cycle broken by depth/visited guards).
        assert!(
            matches!(
                resolved_a.resolved_time,
                ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_)
            ),
            "cycle fact A must remain Unknown, got {:?}",
            resolved_a.resolved_time
        );
        assert!(
            matches!(
                resolved_b.resolved_time,
                ResolvedTime::Unknown | ResolvedTime::RelativeDescription(_)
            ),
            "cycle fact B must remain Unknown, got {:?}",
            resolved_b.resolved_time
        );
    }

    // ========================================================================
    // INTENT-VERB EXPANSION + WORD-BOUNDARY PRECISION + TENSE ROUTING
    // ========================================================================

    fn iv_conv() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn extract(content: &str) -> Vec<TemporalFact> {
        let mid = super::super::types::MemoryId(uuid::Uuid::new_v4());
        extract_temporal_facts(content, &mid, iv_conv(), &["Melanie".to_string()])
    }

    // ---- Word-boundary precision: the load-bearing FP fixes -----------------

    #[test]
    fn test_wordboundary_orange_creates_no_fact() {
        // "ran" must NOT match inside "orange". No time signal, no real verb.
        let facts = extract("Melanie likes orange juice");
        assert!(
            facts.is_empty(),
            "'orange' must not trip the 'ran' past-verb match: {facts:?}"
        );
    }

    #[test]
    fn test_wordboundary_forgot_creates_no_fact() {
        // "got" must NOT match inside "forgot".
        let facts = extract("Melanie forgot her keys");
        assert!(
            facts.is_empty(),
            "'forgot' must not trip the 'got' past-verb match: {facts:?}"
        );
    }

    #[test]
    fn test_wordboundary_monsoon_creates_no_fact() {
        // "soon" must NOT match inside "monsoon" (no intent verb present).
        let facts = extract("Melanie loves the monsoon");
        assert!(
            facts.is_empty(),
            "'monsoon' must not trip the 'soon' future marker: {facts:?}"
        );
    }

    #[test]
    fn test_wordboundary_keeps_genuine_matches() {
        // The boundary change must NOT drop real matches.
        let ran = extract("Melanie ran a marathon yesterday");
        assert!(!ran.is_empty(), "'ran ... yesterday' must still extract");
        let soon = extract("Melanie is planning to visit soon");
        assert!(
            !soon.is_empty(),
            "'planning to ... soon' must still extract"
        );
    }

    // ---- New intent families DO create planned facts ------------------------

    #[test]
    fn test_get_to_creates_planned_fact() {
        let facts = extract("Melanie gets to see her sister next week");
        assert_eq!(facts.len(), 1, "get-to family must extract: {facts:?}");
        assert_eq!(facts[0].event_type, EventType::Planned);
    }

    #[test]
    fn test_decide_to_creates_planned_fact() {
        let facts = extract("Melanie decided to adopt a puppy");
        assert_eq!(facts.len(), 1, "decide-to must extract: {facts:?}");
        assert_eq!(
            facts[0].event_type,
            EventType::Planned,
            "decision-past with no past marker routes Planned"
        );
    }

    #[test]
    fn test_about_to_creates_planned_fact() {
        let facts = extract("Melanie is about to start a pottery class");
        assert_eq!(facts.len(), 1, "about-to must extract: {facts:?}");
        assert_eq!(facts[0].event_type, EventType::Planned);
    }

    // ---- Traps do NOT create planned facts ----------------------------------

    #[test]
    fn test_due_to_creates_no_planned_fact() {
        let facts = extract("Melanie cancelled the picnic due to rain");
        // No future plan asserted (causal "due to"); may surface nothing.
        assert!(
            !facts.iter().any(|f| f.event_type == EventType::Planned),
            "'due to' is causal, never a plan: {facts:?}"
        );
    }

    #[test]
    fn test_used_to_creates_no_planned_fact() {
        let facts = extract("Melanie used to run marathons");
        assert!(
            !facts.iter().any(|f| f.event_type == EventType::Planned),
            "'used to' is past-habitual, never a plan: {facts:?}"
        );
    }

    // ---- Tense routing: past-intent away from confident-future --------------

    #[test]
    fn test_was_going_to_last_week_is_past_intent_not_future() {
        let facts = extract("Melanie was going to visit her last week but couldn't");
        assert!(!facts.is_empty(), "must extract something: {facts:?}");
        let f = &facts[0];
        assert_ne!(
            f.event_type,
            EventType::Planned,
            "past-aux 'was going to' must NOT route confident-future Planned: {f:?}"
        );
        assert_eq!(
            f.event_type,
            EventType::PastIntent,
            "unfulfilled past intent → PastIntent bucket: {f:?}"
        );
        assert!(
            f.confidence <= 0.5,
            "past intent must not be high-confidence, got {}",
            f.confidence
        );
    }

    #[test]
    fn test_wanted_to_is_past_intent() {
        let facts = extract("Melanie wanted to go camping last week");
        assert!(!facts.is_empty());
        assert_eq!(
            facts[0].event_type,
            EventType::PastIntent,
            "'wanted to' is past-intent: {:?}",
            facts[0]
        );
    }

    // ---- Negation + question guards -----------------------------------------

    #[test]
    fn test_negated_intent_creates_no_fact() {
        let facts = extract("Melanie doesn't want to go to the reunion");
        assert!(
            !facts.iter().any(|f| f.event_type == EventType::Planned),
            "negated intent must not assert a plan: {facts:?}"
        );
    }

    #[test]
    fn test_second_person_question_is_not_a_fact() {
        let facts = extract("Do you want to come hiking next week?");
        assert!(
            facts.is_empty(),
            "a 2nd-person question is not a fact about the entity: {facts:?}"
        );
    }

    #[test]
    fn test_cant_wait_to_is_positive_intent() {
        // Contains "can't" but is positive anticipation — must survive negation guard.
        let facts = extract("Melanie can't wait to try the new restaurant");
        assert_eq!(facts.len(), 1, "can't-wait-to must extract: {facts:?}");
        assert_eq!(facts[0].event_type, EventType::Planned);
    }

    // ---- "soon" no longer junk ----------------------------------------------

    #[test]
    fn test_soon_resolves_to_window_not_junk() {
        match resolve_relative_time("soon", iv_conv()) {
            ResolvedTime::Around { radius_days, .. } => {
                assert_eq!(radius_days, 7, "soon → ±7d window");
            }
            other => panic!("'soon' must resolve to an Around window, got {other:?}"),
        }
    }

    // ---- Query-side symmetry ------------------------------------------------

    #[test]
    fn test_query_signals_planned_intent_mirrors_inventory() {
        assert!(query_signals_planned_intent(
            "when is melanie planning to travel"
        ));
        assert!(query_signals_planned_intent(
            "what does she want to do next week"
        ));
        assert!(query_signals_planned_intent("when does he get to visit"));
        assert!(!query_signals_planned_intent(
            "the orange monsoon forgot nothing"
        ));
    }
}
