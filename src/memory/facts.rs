//! Semantic Fact Storage
//!
//! Persistent storage for semantic facts extracted from episodic memories.
//! Facts represent durable knowledge distilled from multiple experiences.
//!
//! Storage schema:
//! - `facts:{user_id}:{fact_id}` - Primary fact storage
//! - `facts_by_entity:{user_id}:{entity}:{fact_id}` - Entity index for fast lookup
//! - `facts_by_type:{user_id}:{type}:{fact_id}` - Type index
//! - `facts_embedding:{user_id}:{fact_id}` - Pre-computed embedding vector (384-dim)

use anyhow::Result;
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::compression::{FactType, SemanticFact};
use super::types::MemoryId;

/// Response for fact queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactQueryResponse {
    pub facts: Vec<SemanticFact>,
    pub total: usize,
}

/// Statistics about semantic facts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactStats {
    pub total_facts: usize,
    pub by_type: std::collections::HashMap<String, usize>,
    pub avg_confidence: f32,
    pub avg_support: f32,
}

/// Pre-bi-temporal `SemanticFact` shape (the OLDEST on-disk format). Used as
/// a fallback when decoding records written before `valid_from` / `valid_until`
/// / `superseded_by` / `supersedes` were appended to the struct.
///
/// Decoder chain on read: current shape → [`LegacySemanticFactV2`] (bi-temporal
/// but pre-purge) → `LegacySemanticFactV1` (pre-bi-temporal).
///
/// bincode 2 with `config::standard()` is positional — `#[serde(default)]`
/// does NOT cover trailing fields. We must explicitly decode against the
/// old shape, then promote the result via [`LegacySemanticFactV1::upgrade`].
#[derive(Deserialize)]
struct LegacySemanticFactV1 {
    id: String,
    fact: String,
    confidence: f32,
    support_count: usize,
    source_memories: Vec<MemoryId>,
    related_entities: Vec<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    last_reinforced: chrono::DateTime<chrono::Utc>,
    fact_type: FactType,
}

impl LegacySemanticFactV1 {
    fn upgrade(self) -> SemanticFact {
        SemanticFact {
            id: self.id,
            fact: self.fact,
            confidence: self.confidence,
            support_count: self.support_count,
            source_memories: self.source_memories,
            related_entities: self.related_entities,
            created_at: self.created_at,
            last_reinforced: self.last_reinforced,
            fact_type: self.fact_type,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        }
    }
}

/// Bi-temporal-but-pre-purge `SemanticFact` shape. Used as a fallback for
/// records written between the bi-temporal landing and the
/// `purged_at` / `purge_reason` schema bump.
#[derive(Deserialize)]
struct LegacySemanticFactV2 {
    id: String,
    fact: String,
    confidence: f32,
    support_count: usize,
    source_memories: Vec<MemoryId>,
    related_entities: Vec<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    last_reinforced: chrono::DateTime<chrono::Utc>,
    fact_type: FactType,
    #[serde(default)]
    valid_from: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    valid_until: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    superseded_by: Option<String>,
    #[serde(default)]
    supersedes: Vec<String>,
}

impl LegacySemanticFactV2 {
    fn upgrade(self) -> SemanticFact {
        SemanticFact {
            id: self.id,
            fact: self.fact,
            confidence: self.confidence,
            support_count: self.support_count,
            source_memories: self.source_memories,
            related_entities: self.related_entities,
            created_at: self.created_at,
            last_reinforced: self.last_reinforced,
            fact_type: self.fact_type,
            valid_from: self.valid_from,
            valid_until: self.valid_until,
            superseded_by: self.superseded_by,
            supersedes: self.supersedes,
            purged_at: None,
            purge_reason: None,
        }
    }
}

/// Decode a `SemanticFact` from RocksDB bytes with a two-step fallback chain:
/// current shape → V2 (bi-temporal, no purge) → V1 (pre-bi-temporal). Returns
/// `(fact, is_legacy)` — when `is_legacy` is true, callers may opt to re-write
/// the record in the current shape to amortize future reads. Matches the
/// [crate::memory::storage::deserialize_with_fallback] pattern on `Memory`.
fn decode_semantic_fact_with_fallback(data: &[u8]) -> Result<(SemanticFact, bool)> {
    match bincode::serde::decode_from_slice::<SemanticFact, _>(
        data,
        bincode::config::standard(),
    ) {
        Ok((fact, _)) => Ok((fact, false)),
        Err(_) => {
            if let Ok((v2, _)) = bincode::serde::decode_from_slice::<LegacySemanticFactV2, _>(
                data,
                bincode::config::standard(),
            ) {
                tracing::debug!("Migrated SemanticFact from pre-purge v2 shape");
                return Ok((v2.upgrade(), true));
            }
            let (legacy, _): (LegacySemanticFactV1, _) =
                bincode::serde::decode_from_slice(data, bincode::config::standard())?;
            tracing::debug!("Migrated SemanticFact from pre-bi-temporal v1 shape");
            Ok((legacy.upgrade(), true))
        }
    }
}

/// Universal active-fact predicate used by every "currently-true" reader path
/// in [`SemanticFactStore`]. Excludes BOTH bi-temporally expired AND
/// administratively purged facts. Centralized so that adding future state
/// dimensions flows through one call site.
///
/// Direct delegate to [`super::compression::is_active`] kept here for ergonomic
/// access from this module without an additional import at every call site.
fn fact_is_active(fact: &SemanticFact, now: chrono::DateTime<chrono::Utc>) -> bool {
    super::compression::is_active(fact, now)
}

/// Returns `true` if the fact was valid at the given instant `at`:
/// `valid_from <= at` (or `valid_from` is None, treated as `created_at`)
/// AND (`valid_until` is None OR `valid_until > at`).
fn fact_is_valid_at(fact: &SemanticFact, at: chrono::DateTime<chrono::Utc>) -> bool {
    let from = fact.valid_from.unwrap_or(fact.created_at);
    if from > at {
        return false;
    }
    match fact.valid_until {
        Some(until) => until > at,
        None => true,
    }
}

/// Storage for semantic facts with indexing
pub struct SemanticFactStore {
    db: Arc<DB>,
}

impl SemanticFactStore {
    /// Create a new fact store backed by RocksDB
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Get references to all RocksDB databases for backup
    pub fn databases(&self) -> Vec<(&str, &Arc<DB>)> {
        vec![("semantic_facts", &self.db)]
    }

    /// Store a semantic fact
    pub fn store(&self, user_id: &str, fact: &SemanticFact) -> Result<()> {
        // Primary storage
        let key = format!("facts:{}:{}", user_id, fact.id);
        let value = bincode::serde::encode_to_vec(fact, bincode::config::standard())?;
        self.db.put(key.as_bytes(), &value)?;

        // Entity index - index by each related entity
        for entity in &fact.related_entities {
            let entity_key = format!(
                "facts_by_entity:{}:{}:{}",
                user_id,
                entity.to_lowercase(),
                fact.id
            );
            self.db.put(entity_key.as_bytes(), fact.id.as_bytes())?;
        }

        // Type index
        let type_name = format!("{:?}", fact.fact_type);
        let type_key = format!("facts_by_type:{}:{}:{}", user_id, type_name, fact.id);
        self.db.put(type_key.as_bytes(), fact.id.as_bytes())?;

        Ok(())
    }

    /// Store multiple facts in a batch
    pub fn store_batch(&self, user_id: &str, facts: &[SemanticFact]) -> Result<usize> {
        let mut stored = 0;
        for fact in facts {
            if self.store(user_id, fact).is_ok() {
                stored += 1;
            }
        }
        Ok(stored)
    }

    /// Get a fact by ID
    pub fn get(&self, user_id: &str, fact_id: &str) -> Result<Option<SemanticFact>> {
        let key = format!("facts:{}:{}", user_id, fact_id);
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let (fact, _is_legacy) = decode_semantic_fact_with_fallback(&data)?;
                Ok(Some(fact))
            }
            None => Ok(None),
        }
    }

    /// Update an existing fact (for reinforcement)
    pub fn update(&self, user_id: &str, fact: &SemanticFact) -> Result<()> {
        // Simply overwrite - indices stay valid since ID doesn't change
        let key = format!("facts:{}:{}", user_id, fact.id);
        let value = bincode::serde::encode_to_vec(fact, bincode::config::standard())?;
        self.db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Delete a fact
    pub fn delete(&self, user_id: &str, fact_id: &str) -> Result<bool> {
        // Get fact first to clean up indices
        if let Some(fact) = self.get(user_id, fact_id)? {
            // Delete entity indices
            for entity in &fact.related_entities {
                let entity_key = format!(
                    "facts_by_entity:{}:{}:{}",
                    user_id,
                    entity.to_lowercase(),
                    fact_id
                );
                self.db.delete(entity_key.as_bytes())?;
            }

            // Delete type index
            let type_name = format!("{:?}", fact.fact_type);
            let type_key = format!("facts_by_type:{}:{}:{}", user_id, type_name, fact_id);
            self.db.delete(type_key.as_bytes())?;

            // Delete primary record
            let key = format!("facts:{}:{}", user_id, fact_id);
            self.db.delete(key.as_bytes())?;

            // Delete embedding if present
            let _ = self.delete_embedding(user_id, fact_id);

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all currently-active facts for a user (excludes expired AND purged).
    pub fn list(&self, user_id: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        self.list_filtered(user_id, limit, false)
    }

    /// List facts for a user. When `include_inactive` is true the result also
    /// contains bi-temporally expired AND administratively purged facts —
    /// reserved for forensic / MIF-export / restore-replay paths. Active
    /// reader paths should call [`Self::list`].
    pub fn list_filtered(
        &self,
        user_id: &str,
        limit: usize,
        include_inactive: bool,
    ) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts:{}:", user_id);
        let mut facts = Vec::new();
        let now = chrono::Utc::now();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            // Skip index keys (they contain extra colons)
            if key_str.matches(':').count() > 2 {
                continue;
            }

            if let Ok((fact, _is_legacy)) = decode_semantic_fact_with_fallback(&value) {
                if !include_inactive && !fact_is_active(&fact, now) {
                    continue;
                }
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        // Sort by confidence (highest first)
        facts.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        Ok(facts)
    }

    /// Find currently-active facts by related entity (excludes expired AND purged).
    pub fn find_by_entity(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        self.find_by_entity_filtered(user_id, entity, limit, false)
    }

    /// Find facts by related entity. When `include_inactive` is true the
    /// result also contains expired AND purged facts (forensic / MIF paths).
    pub fn find_by_entity_filtered(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
        include_inactive: bool,
    ) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts_by_entity:{}:{}:", user_id, entity.to_lowercase());
        let mut facts = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        let now = chrono::Utc::now();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if seen_ids.insert(fact_id.to_string()) {
                if let Some(fact) = self.get(user_id, &fact_id)? {
                    if !include_inactive && !fact_is_active(&fact, now) {
                        continue;
                    }
                    facts.push(fact);
                    if facts.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(facts)
    }

    /// Find currently-active facts by type (excludes expired AND purged).
    pub fn find_by_type(
        &self,
        user_id: &str,
        fact_type: FactType,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        self.find_by_type_filtered(user_id, fact_type, limit, false)
    }

    /// Find facts by type. When `include_inactive` is true the result also
    /// contains expired AND purged facts (forensic / MIF paths).
    pub fn find_by_type_filtered(
        &self,
        user_id: &str,
        fact_type: FactType,
        limit: usize,
        include_inactive: bool,
    ) -> Result<Vec<SemanticFact>> {
        let type_name = format!("{:?}", fact_type);
        let prefix = format!("facts_by_type:{}:{}:", user_id, type_name);
        let mut facts = Vec::new();
        let now = chrono::Utc::now();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if let Some(fact) = self.get(user_id, &fact_id)? {
                if !include_inactive && !fact_is_active(&fact, now) {
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

    /// Search currently-active facts by keyword (excludes expired AND purged).
    pub fn search(&self, user_id: &str, query: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        self.search_filtered(user_id, query, limit, false)
    }

    /// Search facts by keyword. When `include_inactive` is true the result
    /// also contains expired AND purged facts (forensic / MIF paths).
    pub fn search_filtered(
        &self,
        user_id: &str,
        query: &str,
        limit: usize,
        include_inactive: bool,
    ) -> Result<Vec<SemanticFact>> {
        let query_lower = query.to_lowercase();
        let all_facts = self.list_filtered(user_id, 1000, include_inactive)?;

        let mut matching: Vec<SemanticFact> = all_facts
            .into_iter()
            .filter(|f| f.fact.to_lowercase().contains(&query_lower))
            .collect();

        matching.truncate(limit);
        Ok(matching)
    }

    /// Return facts that were valid at the given instant `at` (point-in-time
    /// query). A fact is valid at `at` when `valid_from <= at` and
    /// `valid_until` is `None` or strictly after `at`. Facts without an
    /// explicit `valid_from` use `created_at` as the start of their window.
    ///
    /// **Administrative-purge override** (security): facts with `purged_at`
    /// set are ALWAYS excluded, regardless of whether `at` predates the
    /// purge timestamp. Time-travel queries must not become an oracle for
    /// purged content (see `evaluations/breakers-revised-plan-p1-...json`
    /// TIME-TRAVEL-LEAK).
    pub fn as_of(
        &self,
        user_id: &str,
        at: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts:{}:", user_id);
        let mut facts = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }
            if key_str.matches(':').count() > 2 {
                continue;
            }

            if let Ok((fact, _is_legacy)) = decode_semantic_fact_with_fallback(&value) {
                if fact.purged_at.is_some() {
                    continue;
                }
                if fact_is_valid_at(&fact, at) {
                    facts.push(fact);
                    if facts.len() >= limit {
                        break;
                    }
                }
            }
        }

        facts.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(facts)
    }

    /// Get statistics about stored facts
    pub fn stats(&self, user_id: &str) -> Result<FactStats> {
        let facts = self.list(user_id, 10000)?;

        if facts.is_empty() {
            return Ok(FactStats::default());
        }

        let mut by_type: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut total_confidence: f32 = 0.0;
        let mut total_support: usize = 0;

        for fact in &facts {
            let type_name = format!("{:?}", fact.fact_type);
            *by_type.entry(type_name).or_insert(0) += 1;
            total_confidence += fact.confidence;
            total_support += fact.support_count;
        }

        let count = facts.len();
        Ok(FactStats {
            total_facts: count,
            by_type,
            avg_confidence: total_confidence / count as f32,
            avg_support: total_support as f32 / count as f32,
        })
    }

    /// Find the creation timestamp of the most recent fact for a user.
    ///
    /// Used at startup to initialize the fact extraction watermark when no
    /// persisted watermark exists. Returns None if user has no facts.
    pub fn latest_fact_created_at(&self, user_id: &str) -> Option<i64> {
        let prefix = format!("facts:{user_id}:");
        let mut max_millis: Option<i64> = None;

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Skip index keys (entity/type sub-keys have extra colons)
            if key_str.matches(':').count() > 2 {
                continue;
            }
            if let Ok((fact, _is_legacy)) = decode_semantic_fact_with_fallback(&value) {
                let millis = fact.created_at.timestamp_millis();
                max_millis = Some(max_millis.map_or(millis, |cur| cur.max(millis)));
            }
        }

        max_millis
    }

    /// Find facts that should decay (no reinforcement for too long)
    pub fn find_decaying_facts(
        &self,
        user_id: &str,
        max_age_days: i64,
    ) -> Result<Vec<SemanticFact>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days);
        let all_facts = self.list(user_id, 10000)?;

        let decaying: Vec<SemanticFact> = all_facts
            .into_iter()
            .filter(|f| f.last_reinforced < cutoff)
            .collect();

        Ok(decaying)
    }

    /// Check if a similar fact already exists (hybrid dedup)
    ///
    /// Multi-gate pipeline when embedding is provided:
    /// 1. Entity gate: at least 1 shared entity, OR both have zero entities
    /// 2. Polarity gate: same negation polarity (prevents merging contradictions)
    /// 3. Cosine gate: embedding similarity >= FACT_DEDUP_COSINE_THRESHOLD
    /// 4. Jaccard floor: word overlap >= FACT_DEDUP_JACCARD_FLOOR
    ///
    /// Falls back to pure Jaccard (0.70) if no embedding is provided.
    pub fn find_similar(
        &self,
        user_id: &str,
        fact_content: &str,
        fact_entities: &[String],
        new_embedding: Option<&[f32]>,
    ) -> Result<Option<SemanticFact>> {
        use crate::constants::{
            FACT_DEDUP_COSINE_THRESHOLD, FACT_DEDUP_JACCARD_FALLBACK, FACT_DEDUP_JACCARD_FLOOR,
        };
        use crate::similarity::cosine_similarity;

        let facts = self.list(user_id, 1000)?;
        let query_lower = fact_content.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
        let new_polarity = detect_polarity(&query_lower);
        let new_entity_set: std::collections::HashSet<&str> =
            fact_entities.iter().map(|s| s.as_str()).collect();

        let use_hybrid = new_embedding.is_some();
        let mut best_match: Option<(f32, SemanticFact)> = None;

        for fact in facts {
            let fact_lower = fact.fact.to_lowercase();
            let fact_words: std::collections::HashSet<&str> =
                fact_lower.split_whitespace().collect();

            // Compute Jaccard (needed in both modes)
            let intersection = query_words.intersection(&fact_words).count();
            let union = query_words.union(&fact_words).count();
            let jaccard = if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };

            if use_hybrid {
                // Gate 1: Entity overlap — at least 1 shared entity, or both empty
                let existing_entity_set: std::collections::HashSet<&str> =
                    fact.related_entities.iter().map(|s| s.as_str()).collect();
                let both_empty = new_entity_set.is_empty() && existing_entity_set.is_empty();
                let has_overlap = !new_entity_set.is_disjoint(&existing_entity_set);
                if !both_empty && !has_overlap {
                    continue;
                }

                // Gate 2: Polarity match — prevents merging contradictions
                let existing_polarity = detect_polarity(&fact_lower);
                if new_polarity != existing_polarity {
                    continue;
                }

                // Gate 3: Cosine similarity
                let new_emb = new_embedding.unwrap();
                match self.get_embedding(user_id, &fact.id) {
                    Ok(Some(existing_emb)) => {
                        let cosine = cosine_similarity(new_emb, &existing_emb);
                        if cosine < FACT_DEDUP_COSINE_THRESHOLD {
                            continue;
                        }

                        // Gate 4: Jaccard sanity floor
                        if jaccard < FACT_DEDUP_JACCARD_FLOOR {
                            continue;
                        }

                        // Passed all gates — rank by cosine
                        if best_match.as_ref().is_none_or(|(s, _)| cosine > *s) {
                            best_match = Some((cosine, fact));
                        }
                    }
                    _ => {
                        // No stored embedding — fall back to Jaccard-only for this candidate
                        if jaccard >= FACT_DEDUP_JACCARD_FALLBACK
                            && best_match.as_ref().is_none_or(|(s, _)| jaccard > *s)
                        {
                            best_match = Some((jaccard, fact));
                        }
                    }
                }
            } else {
                // Fallback: pure Jaccard (legacy behavior when embedder unavailable)
                if jaccard >= FACT_DEDUP_JACCARD_FALLBACK {
                    return Ok(Some(fact));
                }
            }
        }

        Ok(best_match.map(|(_, fact)| fact))
    }

    // =========================================================================
    // BI-TEMPORAL INVALIDATION
    // =========================================================================

    /// Detect facts that contradict `new_fact` and return their invalidated
    /// snapshots (with `valid_until` and `superseded_by` set) without writing
    /// them. Caller is expected to persist via [`store_with_invalidation`]
    /// for atomicity.
    ///
    /// Contradiction criteria (mirrors the existing polarity-aware dedup path):
    /// 1. Shares ≥1 `related_entity` with `new_fact`
    /// 2. Has a stored embedding with cosine ≥ [`FACT_DEDUP_COSINE_THRESHOLD`]
    /// 3. Opposite polarity (detected via [`detect_polarity`])
    ///
    /// Skips silently if `new_embedding` is `None` — there's no robust way to
    /// detect a semantic contradiction without an embedding, and falling back
    /// to Jaccard is too aggressive for an invalidation decision (logs at
    /// `debug!` for observability).
    pub fn find_contradictions(
        &self,
        user_id: &str,
        new_fact: &SemanticFact,
        new_embedding: Option<&[f32]>,
    ) -> Result<Vec<SemanticFact>> {
        use crate::constants::FACT_DEDUP_COSINE_THRESHOLD;
        use crate::similarity::cosine_similarity;

        let Some(new_emb) = new_embedding else {
            tracing::debug!(
                user_id = user_id,
                fact_id = %new_fact.id,
                "No embedding for new fact — skipping contradiction detection"
            );
            return Ok(Vec::new());
        };

        let new_polarity = detect_polarity(&new_fact.fact.to_lowercase());
        let new_from = new_fact
            .valid_from
            .unwrap_or(new_fact.created_at);

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut invalidated: Vec<SemanticFact> = Vec::new();

        for entity in &new_fact.related_entities {
            let candidates = self.find_by_entity_filtered(user_id, entity, 200, false)?;
            for candidate in candidates {
                if candidate.id == new_fact.id || !seen.insert(candidate.id.clone()) {
                    continue;
                }

                let Ok(Some(existing_emb)) = self.get_embedding(user_id, &candidate.id) else {
                    continue;
                };

                let cosine = cosine_similarity(new_emb, &existing_emb);
                if cosine < FACT_DEDUP_COSINE_THRESHOLD {
                    continue;
                }

                let candidate_polarity = detect_polarity(&candidate.fact.to_lowercase());
                if candidate_polarity == new_polarity {
                    continue;
                }

                // Clamp valid_until to candidate.valid_from (or its created_at)
                // so contradictions from clock-skewed writers never produce
                // valid_until < valid_from windows.
                let candidate_from = candidate.valid_from.unwrap_or(candidate.created_at);
                let clamped = if new_from < candidate_from {
                    candidate_from
                } else {
                    new_from
                };

                let mut superseded = candidate.clone();
                superseded.valid_until = Some(clamped);
                superseded.superseded_by = Some(new_fact.id.clone());
                invalidated.push(superseded);
            }
        }

        Ok(invalidated)
    }

    /// Atomically store a new fact, invalidate any contradicting older facts,
    /// and stamp `supersedes` on the new fact. Uses a RocksDB `WriteBatch` so
    /// the invalidation half cannot land without the new fact landing.
    ///
    /// Returns the list of invalidated fact IDs. Equivalent to `store` when
    /// no contradictions are detected (and when `new_embedding` is `None`).
    pub fn store_with_invalidation(
        &self,
        user_id: &str,
        new_fact: &SemanticFact,
        new_embedding: Option<&[f32]>,
    ) -> Result<Vec<String>> {
        let mut to_invalidate = self.find_contradictions(user_id, new_fact, new_embedding)?;
        let invalidated_ids: Vec<String> =
            to_invalidate.iter().map(|f| f.id.clone()).collect();

        // Stamp the new fact with the forward-pointer to the facts it supersedes.
        let mut new_fact_owned = new_fact.clone();
        if !invalidated_ids.is_empty() {
            let mut combined = new_fact_owned.supersedes.clone();
            combined.extend(invalidated_ids.iter().cloned());
            combined.sort();
            combined.dedup();
            new_fact_owned.supersedes = combined;
        }

        let mut batch = rocksdb::WriteBatch::default();

        // Re-encode each invalidated fact with valid_until + superseded_by set.
        for old in to_invalidate.drain(..) {
            let key = format!("facts:{}:{}", user_id, old.id);
            let value = bincode::serde::encode_to_vec(&old, bincode::config::standard())?;
            batch.put(key.as_bytes(), &value);

            tracing::info!(
                user_id = user_id,
                old_fact_id = %old.id,
                new_fact_id = %new_fact_owned.id,
                "SemanticFact superseded — old fact invalidated"
            );
        }

        // Primary record for the new fact.
        let new_key = format!("facts:{}:{}", user_id, new_fact_owned.id);
        let new_value =
            bincode::serde::encode_to_vec(&new_fact_owned, bincode::config::standard())?;
        batch.put(new_key.as_bytes(), &new_value);

        // Entity indices for the new fact.
        for entity in &new_fact_owned.related_entities {
            let entity_key = format!(
                "facts_by_entity:{}:{}:{}",
                user_id,
                entity.to_lowercase(),
                new_fact_owned.id
            );
            batch.put(entity_key.as_bytes(), new_fact_owned.id.as_bytes());
        }

        // Type index for the new fact.
        let type_name = format!("{:?}", new_fact_owned.fact_type);
        let type_key =
            format!("facts_by_type:{}:{}:{}", user_id, type_name, new_fact_owned.id);
        batch.put(type_key.as_bytes(), new_fact_owned.id.as_bytes());

        // Optional embedding for the new fact.
        if let Some(emb) = new_embedding {
            let emb_key = format!("facts_embedding:{}:{}", user_id, new_fact_owned.id);
            let emb_value =
                bincode::serde::encode_to_vec(emb, bincode::config::standard())?;
            batch.put(emb_key.as_bytes(), &emb_value);
        }

        self.db.write(batch)?;
        Ok(invalidated_ids)
    }

    // =========================================================================
    // EMBEDDING PERSISTENCE
    // =========================================================================

    /// Store pre-computed embedding vector for a fact
    ///
    /// Key format: `facts_embedding:{user_id}:{fact_id}` → bincode Vec<f32>
    /// Stored separately from SemanticFact struct for backward compatibility.
    pub fn store_embedding(&self, user_id: &str, fact_id: &str, embedding: &[f32]) -> Result<()> {
        let key = format!("facts_embedding:{user_id}:{fact_id}");
        let value = bincode::serde::encode_to_vec(embedding, bincode::config::standard())?;
        self.db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Get pre-computed embedding vector for a fact
    pub fn get_embedding(&self, user_id: &str, fact_id: &str) -> Result<Option<Vec<f32>>> {
        let key = format!("facts_embedding:{user_id}:{fact_id}");
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let (embedding, _): (Vec<f32>, _) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    /// Delete embedding for a fact (called during fact deletion)
    pub fn delete_embedding(&self, user_id: &str, fact_id: &str) -> Result<()> {
        let key = format!("facts_embedding:{user_id}:{fact_id}");
        self.db.delete(key.as_bytes())?;
        Ok(())
    }

    /// List all unique user IDs that have facts
    pub fn list_users(&self, limit: usize) -> Result<Vec<String>> {
        let prefix = "facts:";
        let mut users = std::collections::HashSet::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(prefix) {
                break;
            }

            // Key format: facts:{user_id}:{fact_id}
            // Skip index keys (facts_by_entity, facts_by_type)
            if key_str.starts_with("facts_by_") {
                continue;
            }

            // Extract user_id from key
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() >= 2 {
                users.insert(parts[1].to_string());
                if users.len() >= limit {
                    break;
                }
            }
        }

        Ok(users.into_iter().collect())
    }
}

/// Detect negation polarity of a fact statement.
///
/// Returns `true` for positive polarity (even negation count, including 0),
/// `false` for negative polarity (odd negation count).
/// Handles double-negation: "not unlike" = positive.
fn detect_polarity(text_lower: &str) -> bool {
    use crate::constants::FACT_NEGATION_MARKERS;
    let words: Vec<&str> = text_lower.split_whitespace().collect();
    let negation_count = words
        .iter()
        .filter(|w| FACT_NEGATION_MARKERS.iter().any(|marker| *w == marker))
        .count();
    negation_count % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (SemanticFactStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        (SemanticFactStore::new(db), temp_dir)
    }

    fn create_test_fact(id: &str, content: &str) -> SemanticFact {
        SemanticFact {
            id: id.to_string(),
            fact: content.to_string(),
            confidence: 0.8,
            support_count: 3,
            source_memories: vec![],
            related_entities: vec!["rust".to_string(), "memory".to_string()],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
            valid_from: None,
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        }
    }

    #[test]
    fn test_store_and_get() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Rust is a systems programming language");

        store.store("user-1", &fact).unwrap();
        let retrieved = store.get("user-1", "fact-1").unwrap();

        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().fact,
            "Rust is a systems programming language"
        );
    }

    #[test]
    fn test_find_by_entity() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Rust has efficient memory management");

        store.store("user-1", &fact).unwrap();
        let results = store.find_by_entity("user-1", "rust", 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "fact-1");
    }

    #[test]
    fn test_find_by_type() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Pattern detected in codebase");

        store.store("user-1", &fact).unwrap();
        let results = store.find_by_type("user-1", FactType::Pattern, 10).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_delete() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Test fact");

        store.store("user-1", &fact).unwrap();
        assert!(store.get("user-1", "fact-1").unwrap().is_some());

        store.delete("user-1", "fact-1").unwrap();
        assert!(store.get("user-1", "fact-1").unwrap().is_none());

        // Entity index should also be cleaned up
        let by_entity = store.find_by_entity("user-1", "rust", 10).unwrap();
        assert!(by_entity.is_empty());
    }

    #[test]
    fn test_stats() {
        let (store, _dir) = create_test_store();

        store
            .store("user-1", &create_test_fact("fact-1", "Fact one"))
            .unwrap();
        store
            .store("user-1", &create_test_fact("fact-2", "Fact two"))
            .unwrap();

        let stats = store.stats("user-1").unwrap();
        assert_eq!(stats.total_facts, 2);
        assert!(stats.avg_confidence > 0.0);
    }

    // =========================================================================
    // BI-TEMPORAL TESTS
    // =========================================================================

    /// Hand-crafted v1-shape blob (no bi-temporal fields) must decode through
    /// `decode_semantic_fact_with_fallback` and upgrade cleanly. This is the
    /// backup/restore regression test for the schema-evolution shim.
    #[test]
    fn test_semantic_fact_legacy_v1_decode_upgrades() {
        // Encode in the OLD shape (mirror of LegacySemanticFactV1 fields).
        #[derive(serde::Serialize)]
        struct V1 {
            id: String,
            fact: String,
            confidence: f32,
            support_count: usize,
            source_memories: Vec<MemoryId>,
            related_entities: Vec<String>,
            created_at: chrono::DateTime<chrono::Utc>,
            last_reinforced: chrono::DateTime<chrono::Utc>,
            fact_type: FactType,
        }

        let now = chrono::Utc::now();
        let v1 = V1 {
            id: "legacy-fact".to_string(),
            fact: "An old fact from before bi-temporal".to_string(),
            confidence: 0.7,
            support_count: 2,
            source_memories: vec![],
            related_entities: vec!["legacy".to_string()],
            created_at: now,
            last_reinforced: now,
            fact_type: FactType::Pattern,
        };

        let v1_bytes = bincode::serde::encode_to_vec(&v1, bincode::config::standard())
            .expect("encode v1 shape");

        // Put the v1 bytes directly into RocksDB under the canonical key,
        // then read via the store — must transparently upgrade.
        let (store, _dir) = create_test_store();
        let key = format!("facts:{}:{}", "user-1", "legacy-fact");
        store.db.put(key.as_bytes(), &v1_bytes).unwrap();

        let got = store
            .get("user-1", "legacy-fact")
            .unwrap()
            .expect("legacy fact decodes");

        assert_eq!(got.id, "legacy-fact");
        assert_eq!(got.fact, "An old fact from before bi-temporal");
        assert_eq!(got.confidence, 0.7);
        assert_eq!(got.support_count, 2);
        assert_eq!(got.valid_from, None);
        assert_eq!(got.valid_until, None);
        assert_eq!(got.superseded_by, None);
        assert!(got.supersedes.is_empty());
    }

    /// Manual `valid_until` in the past must hide the fact from default
    /// `list()` but keep it visible under `list_filtered(include_expired=true)`.
    #[test]
    fn test_semantic_fact_list_excludes_expired() {
        let (store, _dir) = create_test_store();
        let now = chrono::Utc::now();

        let mut alive = create_test_fact("alive", "Currently true");
        alive.valid_from = Some(now - chrono::Duration::days(2));
        store.store("user-1", &alive).unwrap();

        let mut dead = create_test_fact("dead", "No longer true");
        dead.valid_from = Some(now - chrono::Duration::days(5));
        dead.valid_until = Some(now - chrono::Duration::days(1));
        dead.superseded_by = Some("alive".to_string());
        store.store("user-1", &dead).unwrap();

        let visible = store.list("user-1", 100).unwrap();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, "alive");

        let all = store.list_filtered("user-1", 100, true).unwrap();
        assert_eq!(all.len(), 2);
    }

    /// Point-in-time query returns the fact valid at `at` and excludes facts
    /// that hadn't yet started or had already been invalidated.
    #[test]
    fn test_semantic_fact_as_of_returns_historical_state() {
        let (store, _dir) = create_test_store();
        let now = chrono::Utc::now();
        let t1 = now - chrono::Duration::days(10);
        let t2 = now - chrono::Duration::days(5);
        let t_between = now - chrono::Duration::days(7);

        // Fact A: valid from t1 to t2 (then invalidated by B)
        let mut a = create_test_fact("fact-a", "Postgres v15 is supported");
        a.valid_from = Some(t1);
        a.valid_until = Some(t2);
        a.superseded_by = Some("fact-b".to_string());
        store.store("user-1", &a).unwrap();

        // Fact B: valid from t2 onward
        let mut b = create_test_fact("fact-b", "Postgres v16 is supported");
        b.valid_from = Some(t2);
        b.supersedes = vec!["fact-a".to_string()];
        store.store("user-1", &b).unwrap();

        let at_t_between = store.as_of("user-1", t_between, 100).unwrap();
        assert_eq!(at_t_between.len(), 1, "between t1 and t2: only A");
        assert_eq!(at_t_between[0].id, "fact-a");

        let at_now = store.as_of("user-1", now, 100).unwrap();
        assert_eq!(at_now.len(), 1, "now: only B (A invalidated)");
        assert_eq!(at_now[0].id, "fact-b");

        let before_t1 = store
            .as_of("user-1", t1 - chrono::Duration::days(1), 100)
            .unwrap();
        assert!(before_t1.is_empty(), "before t1: nothing yet");
    }

    /// `store_with_invalidation` with opposite-polarity high-cosine fact must
    /// invalidate the older fact and stamp `supersedes`/`superseded_by` links.
    #[test]
    fn test_semantic_fact_store_with_invalidation_supersedes_old() {
        let (store, _dir) = create_test_store();
        let user = "user-1";

        // Embedding is the same vector for both, so cosine = 1.0 (above 0.80
        // threshold). Polarity differs because of the "not".
        let emb: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let mut old = create_test_fact("old", "Postgres v15 is supported");
        old.valid_from = Some(chrono::Utc::now() - chrono::Duration::hours(2));
        store.store(user, &old).unwrap();
        store.store_embedding(user, &old.id, &emb).unwrap();

        let new = SemanticFact {
            id: "new".to_string(),
            fact: "Postgres v15 is not supported".to_string(),
            confidence: 0.9,
            support_count: 1,
            source_memories: vec![],
            related_entities: vec!["rust".to_string(), "memory".to_string()],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
            valid_from: Some(chrono::Utc::now()),
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        };

        let invalidated = store
            .store_with_invalidation(user, &new, Some(&emb))
            .unwrap();
        assert_eq!(invalidated, vec!["old".to_string()]);

        // Old fact is no longer in the default view.
        let visible = store.list(user, 100).unwrap();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, "new");

        // Old fact still readable directly, with the invalidation stamped.
        let old_back = store.get(user, "old").unwrap().unwrap();
        assert!(old_back.valid_until.is_some());
        assert_eq!(old_back.superseded_by.as_deref(), Some("new"));

        // New fact has the forward-pointer.
        let new_back = store.get(user, "new").unwrap().unwrap();
        assert_eq!(new_back.supersedes, vec!["old".to_string()]);
    }

    /// `find_contradictions` must skip silently when no embedding is provided.
    #[test]
    fn test_semantic_fact_no_embedding_skips_contradiction() {
        let (store, _dir) = create_test_store();
        let user = "user-1";

        let mut old = create_test_fact("old", "Postgres v15 is supported");
        old.valid_from = Some(chrono::Utc::now() - chrono::Duration::hours(2));
        store.store(user, &old).unwrap();

        let new = SemanticFact {
            id: "new".to_string(),
            fact: "Postgres v15 is not supported".to_string(),
            confidence: 0.9,
            support_count: 1,
            source_memories: vec![],
            related_entities: vec!["rust".to_string(), "memory".to_string()],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
            valid_from: Some(chrono::Utc::now()),
            valid_until: None,
            superseded_by: None,
            supersedes: Vec::new(),
            purged_at: None,
            purge_reason: None,
        };

        // No embedding on either side — must not invalidate.
        let invalidated = store.store_with_invalidation(user, &new, None).unwrap();
        assert!(invalidated.is_empty());

        // Both visible.
        let visible = store.list(user, 100).unwrap();
        assert_eq!(visible.len(), 2);
    }
}
