//! Read-only aggregations for the integrated GUI (spine/gui).
//!
//! Every response is one envelope: `{schema, user_id, binding, measured_at, witness, data}`, and
//! every section inside `data` is a [`Field`] whose `state` says whether it was measured, could not
//! be measured (with the reason), or is undetermined. A read that fails is `unmeasured`, never an
//! empty list: an empty list has to mean "nothing recorded".
//!
//! These routes bind the user from the credential, not from the request. A tenant-bound key wins;
//! otherwise the server must name the one user its UI may read in `VELD_UI_USER_ID`. With neither,
//! the routes refuse — on a stock build any key can name any `user_id`, and these routes expose
//! recall queries and whole-store aggregates.

use std::collections::{BTreeMap, HashMap, HashSet};

use axum::{
    extract::{Extension, Query, State},
    response::Json,
};
use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};

use super::health::AppState;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::AppError;
use crate::memory::recall_events::RecallEvent;
use crate::memory::tag_sources::{TagSource, TagSources};
use crate::memory::types::MemoryId;
use crate::memory::{ExperienceType, Memory};
use crate::validation;

pub const UI_SCHEMA: u32 = 1;
pub const UI_USER_ENV: &str = "VELD_UI_USER_ID";
/// Comma-separated additions to the plumbing list; an entry ending in `:` is a namespace.
pub const UI_PLUMBING_EXTRA_ENV: &str = "VELD_UI_PLUMBING_EXTRA";

const PLUMBING_NAMESPACES: &[&str] = &[
    "scope:",
    "source:",
    "kind:",
    "agent-type:",
    "spawned-by:",
    "model:",
    "type:",
    "role:",
    "format:",
    "redacted:",
];
const PLUMBING_BARE: &[&str] = &[
    "automated",
    "hook",
    "verified",
    "born",
    "derived",
    "session-memory",
    "session-close",
    "handoff",
    "overlook-snapshot",
];
const WEEKS: usize = 12;
/// p90 of measured revisit intervals (9.9 days); a memory younger than this has not had its chance.
const NEVER_ASKED_MIN_AGE_HOURS: i64 = 238;
const DECAY_IMPORTANCE_FLOOR: f32 = 0.05;
/// A memory whose decay estimate is under this many days is on the about-to-forget shelf.
const ABOUT_TO_FORGET_DAYS: f32 = 30.0;
const SHELF_ITEMS: usize = 20;

// ============================================================================
// ENVELOPE
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldState {
    Measured,
    Unmeasured,
    Undetermined,
}

#[derive(Debug, Clone, Serialize)]
pub struct Field<T> {
    pub state: FieldState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub witness: &'static str,
}

impl<T> Field<T> {
    pub fn measured(value: T, witness: &'static str) -> Self {
        Self {
            state: FieldState::Measured,
            value: Some(value),
            reason: None,
            witness,
        }
    }

    pub fn unmeasured(reason: impl Into<String>, witness: &'static str) -> Self {
        Self {
            state: FieldState::Unmeasured,
            value: None,
            reason: Some(reason.into()),
            witness,
        }
    }

    pub fn undetermined(reason: impl Into<String>, witness: &'static str) -> Self {
        Self {
            state: FieldState::Undetermined,
            value: None,
            reason: Some(reason.into()),
            witness,
        }
    }

    pub fn from_result(result: anyhow::Result<T>, witness: &'static str) -> Self {
        match result {
            Ok(v) => Self::measured(v, witness),
            Err(e) => Self::unmeasured(format!("{e:#}"), witness),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Binding {
    TenantKey,
    UiUserEnv,
}

#[derive(Debug, Serialize)]
pub struct UiEnvelope<T> {
    pub schema: u32,
    pub user_id: String,
    pub binding: Binding,
    pub measured_at: DateTime<Utc>,
    pub witness: &'static str,
    pub data: T,
}

fn envelope<T>(user_id: String, binding: Binding, witness: &'static str, data: T) -> UiEnvelope<T> {
    UiEnvelope {
        schema: UI_SCHEMA,
        user_id,
        binding,
        measured_at: Utc::now(),
        witness,
        data,
    }
}

/// The user these routes may read. Read at call time so a test or an operator can set it after
/// start-up.
pub fn resolve_ui_user(
    requested: Option<&str>,
    authenticated: Option<&AuthenticatedUser>,
) -> Result<(String, Binding), AppError> {
    if let Some(auth) = authenticated {
        return resolve_request_user_id(requested, Some(auth)).map(|u| (u, Binding::TenantKey));
    }
    let bound = std::env::var(UI_USER_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: format!(
                "UI routes need a bound user: use a tenant-bound key or set {UI_USER_ENV}"
            ),
        })?;
    if let Some(requested) = requested {
        if requested != bound {
            tracing::warn!(
                audit = "ui_user_rejected",
                bound = %bound,
                requested = %requested,
                "UI route asked for a user other than the bound one"
            );
            return Err(AppError::InvalidInput {
                field: "user_id".to_string(),
                reason: "does not match the UI's bound user".to_string(),
            });
        }
    }
    validation::validate_user_id(&bound).map_err(|e| AppError::InvalidInput {
        field: "user_id".to_string(),
        reason: e.to_string(),
    })?;
    Ok((bound, Binding::UiUserEnv))
}

// ============================================================================
// TAG CLASSIFICATION
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TagClass {
    Visible,
    Plumbing,
    Malformed,
    Timestamp,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlumbingRule {
    pub namespaces: Vec<String>,
    pub bare: Vec<String>,
    pub from_env: Vec<String>,
}

impl PlumbingRule {
    pub fn from_env() -> Self {
        let from_env: Vec<String> = std::env::var(UI_PLUMBING_EXTRA_ENV)
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        Self::with_extra(from_env)
    }

    pub fn with_extra(from_env: Vec<String>) -> Self {
        let mut namespaces: Vec<String> =
            PLUMBING_NAMESPACES.iter().map(|s| s.to_string()).collect();
        let mut bare: Vec<String> = PLUMBING_BARE.iter().map(|s| s.to_string()).collect();
        for e in &from_env {
            if e.ends_with(':') {
                namespaces.push(e.clone());
            } else {
                bare.push(e.clone());
            }
        }
        Self {
            namespaces,
            bare,
            from_env,
        }
    }

    pub fn classify(&self, tag: &str) -> TagClass {
        if tag.contains('"') || tag.contains('{') || tag.contains('}') {
            return TagClass::Malformed;
        }
        if is_timestamp_shaped(tag) {
            return TagClass::Timestamp;
        }
        let lower = tag.to_lowercase();
        if self.bare.contains(&lower)
            || self
                .namespaces
                .iter()
                .any(|n| lower.starts_with(n.as_str()))
        {
            return TagClass::Plumbing;
        }
        TagClass::Visible
    }
}

/// `YYYY-MM-DD`, optionally followed by `T` and up to a full time.
fn is_timestamp_shaped(tag: &str) -> bool {
    let b = tag.as_bytes();
    if b.len() < 10 {
        return false;
    }
    let date = b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit);
    if !date {
        return false;
    }
    b.len() == 10
        || (b[10] == b'T'
            && b[11..]
                .iter()
                .all(|c| c.is_ascii_digit() || matches!(c, b':' | b'.' | b'Z' | b'+' | b'-')))
}

// ============================================================================
// TAG AGGREGATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct TagInput {
    pub id: MemoryId,
    pub created_at: DateTime<Utc>,
    pub tags: Vec<String>,
}

impl From<&Memory> for TagInput {
    fn from(m: &Memory) -> Self {
        Self {
            id: m.id.clone(),
            created_at: m.created_at,
            tags: m.experience.tags.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct OriginCounts {
    pub authored: usize,
    pub derived: usize,
    pub undetermined: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Producer {
    pub name: String,
    pub share: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoTag {
    pub tag: String,
    pub memories: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagRow {
    pub tag: String,
    pub class: TagClass,
    /// Memories carrying the tag (document frequency).
    pub memories: usize,
    /// `memories × ln(N / memories)`: a tag on every memory weighs nothing.
    pub weight: f32,
    pub origin: OriginCounts,
    pub sources: BTreeMap<&'static str, usize>,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    /// Memories created per week, oldest first, the last entry being the current week.
    pub weekly: Vec<usize>,
    pub producer: Producer,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster: Option<usize>,
    pub co_occurring: Vec<CoTag>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HiddenCount {
    pub tags: usize,
    pub occurrences: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Cluster {
    pub id: usize,
    pub tags: Vec<String>,
    pub memories: usize,
    pub producer: Producer,
}

#[derive(Debug, Clone, Serialize)]
pub struct Population {
    pub memories: usize,
    pub with_tag_sources: usize,
    pub undetermined: usize,
    pub distinct_tags: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagsReport {
    pub population: Population,
    pub hidden: BTreeMap<TagClass, HiddenCount>,
    pub tags: Vec<TagRow>,
    pub clusters: Vec<Cluster>,
    pub rule: PlumbingRule,
    pub include_hidden: bool,
    pub limit: usize,
    pub week_starts: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct TagsParams {
    pub include_hidden: bool,
    pub limit: usize,
    /// Only tags at or above this many memories are clustered: small sets coincide by chance.
    pub cluster_min_memories: usize,
    pub cluster_min_tags: usize,
    pub co_occurring_per_tag: usize,
}

impl Default for TagsParams {
    fn default() -> Self {
        Self {
            include_hidden: false,
            limit: 300,
            cluster_min_memories: 10,
            cluster_min_tags: 5,
            co_occurring_per_tag: 5,
        }
    }
}

fn producer_of(tags: &[String]) -> String {
    tags.iter()
        .find_map(|t| {
            let l = t.to_lowercase();
            (l.starts_with("scope:") || l.starts_with("source:")).then(|| t.clone())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn week_start(t: DateTime<Utc>) -> DateTime<Utc> {
    let d = t.date_naive();
    let monday = d - Duration::days(d.weekday().num_days_from_monday() as i64);
    monday.and_hms_opt(0, 0, 0).unwrap_or_default().and_utc()
}

/// Aggregate the whole population. `sources` is `None` when the tag-source ledger could not be
/// read; origin counts are then all undetermined and the caller marks the field unmeasured.
pub fn aggregate_tags(
    memories: &[TagInput],
    sources: Option<&HashMap<MemoryId, TagSources>>,
    rule: &PlumbingRule,
    params: &TagsParams,
    now: DateTime<Utc>,
) -> TagsReport {
    struct Acc {
        class: TagClass,
        members: Vec<usize>,
        origin: OriginCounts,
        sources: BTreeMap<&'static str, usize>,
        first: DateTime<Utc>,
        last: DateTime<Utc>,
    }

    let this_week = week_start(now);
    let week_starts: Vec<DateTime<Utc>> = (0..WEEKS)
        .rev()
        .map(|i| this_week - Duration::weeks(i as i64))
        .collect();
    let n = memories.len();
    let producers: Vec<String> = memories.iter().map(|m| producer_of(&m.tags)).collect();

    let mut acc: HashMap<String, Acc> = HashMap::new();
    let mut with_sources = 0usize;
    for (idx, m) in memories.iter().enumerate() {
        let row = sources.and_then(|s| s.get(&m.id));
        if row.is_some() {
            with_sources += 1;
        }
        let by_tag: HashMap<&str, TagSource> = row
            .map(|r| r.tags.iter().map(|t| (t.tag.as_str(), t.source)).collect())
            .unwrap_or_default();
        let mut seen_here: HashSet<&str> = HashSet::new();
        for tag in &m.tags {
            if !seen_here.insert(tag.as_str()) {
                continue;
            }
            let a = acc.entry(tag.clone()).or_insert_with(|| Acc {
                class: rule.classify(tag),
                members: Vec::new(),
                origin: OriginCounts::default(),
                sources: BTreeMap::new(),
                first: m.created_at,
                last: m.created_at,
            });
            a.members.push(idx);
            a.first = a.first.min(m.created_at);
            a.last = a.last.max(m.created_at);
            match by_tag.get(tag.as_str()) {
                Some(src) => {
                    if src.is_authored() {
                        a.origin.authored += 1;
                    } else {
                        a.origin.derived += 1;
                    }
                    *a.sources.entry(src.as_str()).or_insert(0) += 1;
                }
                None => a.origin.undetermined += 1,
            }
        }
    }

    let mut hidden: BTreeMap<TagClass, HiddenCount> = BTreeMap::new();
    for a in acc.values() {
        if a.class != TagClass::Visible {
            let h = hidden.entry(a.class).or_default();
            h.tags += 1;
            h.occurrences += a.members.len();
        }
    }

    let weight = |df: usize| -> f32 {
        if df == 0 || n == 0 {
            0.0
        } else {
            df as f32 * (n as f32 / df as f32).ln()
        }
    };
    let producer_for = |members: &[usize]| -> Producer {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for &i in members {
            *counts.entry(producers[i].as_str()).or_insert(0) += 1;
        }
        let (name, count) = counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
            .unwrap_or(("unknown", 0));
        Producer {
            name: name.to_string(),
            share: if members.is_empty() {
                0.0
            } else {
                count as f32 / members.len() as f32
            },
        }
    };

    let mut ranked: Vec<(&String, &Acc)> = acc
        .iter()
        .filter(|(_, a)| params.include_hidden || a.class == TagClass::Visible)
        .collect();
    ranked.sort_by(|a, b| {
        weight(b.1.members.len())
            .total_cmp(&weight(a.1.members.len()))
            .then_with(|| b.1.members.len().cmp(&a.1.members.len()))
            .then_with(|| a.0.cmp(b.0))
    });
    ranked.truncate(params.limit);

    // Tags stamped on exactly the same set of memories, many at once, are one producer's
    // template rather than many topics.
    let mut by_set: HashMap<&[usize], Vec<&String>> = HashMap::new();
    for (tag, a) in &acc {
        if a.class == TagClass::Visible && a.members.len() >= params.cluster_min_memories {
            by_set.entry(a.members.as_slice()).or_default().push(tag);
        }
    }
    let mut cluster_sets: Vec<(&[usize], Vec<&String>)> = by_set
        .into_iter()
        .filter(|(_, tags)| tags.len() >= params.cluster_min_tags)
        .collect();
    cluster_sets.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));
    let mut cluster_of: HashMap<&str, usize> = HashMap::new();
    let clusters: Vec<Cluster> = cluster_sets
        .into_iter()
        .enumerate()
        .map(|(id, (members, mut tags))| {
            tags.sort();
            for t in &tags {
                cluster_of.insert(t.as_str(), id);
            }
            Cluster {
                id,
                tags: tags.into_iter().cloned().collect(),
                memories: members.len(),
                producer: producer_for(members),
            }
        })
        .collect();

    let ranked_set: HashSet<&str> = ranked.iter().map(|(t, _)| t.as_str()).collect();
    let mut co: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
    for m in memories {
        let present: Vec<&str> = {
            let mut v: Vec<&str> = m
                .tags
                .iter()
                .map(String::as_str)
                .filter(|t| ranked_set.contains(t))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        for (i, a) in present.iter().enumerate() {
            for b in &present[i + 1..] {
                *co.entry(a).or_default().entry(b).or_insert(0) += 1;
                *co.entry(b).or_default().entry(a).or_insert(0) += 1;
            }
        }
    }

    let tags = ranked
        .into_iter()
        .map(|(tag, a)| {
            let mut weekly = vec![0usize; WEEKS];
            for &i in &a.members {
                let ws = week_start(memories[i].created_at);
                if let Some(pos) = week_starts.iter().position(|w| *w == ws) {
                    weekly[pos] += 1;
                }
            }
            let mut co_occurring: Vec<CoTag> = co
                .get(tag.as_str())
                .map(|m| {
                    m.iter()
                        .map(|(t, c)| CoTag {
                            tag: t.to_string(),
                            memories: *c,
                        })
                        .collect()
                })
                .unwrap_or_default();
            co_occurring
                .sort_by(|x, y| y.memories.cmp(&x.memories).then_with(|| x.tag.cmp(&y.tag)));
            co_occurring.truncate(params.co_occurring_per_tag);
            TagRow {
                tag: tag.clone(),
                class: a.class,
                memories: a.members.len(),
                weight: weight(a.members.len()),
                origin: a.origin.clone(),
                sources: a.sources.clone(),
                first_seen: a.first,
                last_seen: a.last,
                weekly,
                producer: producer_for(&a.members),
                cluster: cluster_of.get(tag.as_str()).copied(),
                co_occurring,
            }
        })
        .collect();

    TagsReport {
        population: Population {
            memories: n,
            with_tag_sources: with_sources,
            undetermined: n - with_sources,
            distinct_tags: acc.len(),
        },
        hidden,
        tags,
        clusters,
        rule: rule.clone(),
        include_hidden: params.include_hidden,
        limit: params.limit,
        week_starts,
    }
}

// ============================================================================
// RECALL LOG SUMMARY
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct MissedQuery {
    pub query: String,
    pub times: usize,
    pub last_asked: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallLogReport {
    /// Oldest retained event. Nothing before it can be answered.
    pub horizon: Option<DateTime<Utc>>,
    pub since: DateTime<Utc>,
    pub total: usize,
    pub misses: usize,
    pub attributed: usize,
    pub unattributed: usize,
    pub by_endpoint: BTreeMap<String, usize>,
    pub top_misses: Vec<MissedQuery>,
    /// Newest first.
    pub events: Vec<RecallEvent>,
}

pub fn summarise_recalls(
    events: Vec<RecallEvent>,
    horizon: Option<DateTime<Utc>>,
    since: DateTime<Utc>,
    limit: usize,
) -> RecallLogReport {
    let total = events.len();
    let misses = events.iter().filter(|e| e.result_count == 0).count();
    let attributed = events.iter().filter(|e| e.episode_id.is_some()).count();
    let mut by_endpoint: BTreeMap<String, usize> = BTreeMap::new();
    let mut missed: HashMap<&str, (usize, DateTime<Utc>)> = HashMap::new();
    for e in &events {
        *by_endpoint.entry(e.endpoint.clone()).or_insert(0) += 1;
        if e.result_count == 0 {
            let entry = missed.entry(e.query.as_str()).or_insert((0, e.at));
            entry.0 += 1;
            entry.1 = entry.1.max(e.at);
        }
    }
    let mut top_misses: Vec<MissedQuery> = missed
        .into_iter()
        .map(|(q, (times, last))| MissedQuery {
            query: q.to_string(),
            times,
            last_asked: last,
        })
        .collect();
    top_misses.sort_by(|a, b| {
        b.times
            .cmp(&a.times)
            .then_with(|| b.last_asked.cmp(&a.last_asked))
            .then_with(|| a.query.cmp(&b.query))
    });
    top_misses.truncate(SHELF_ITEMS);
    let mut events = events;
    events.reverse();
    events.truncate(limit);
    RecallLogReport {
        horizon,
        since,
        total,
        misses,
        attributed,
        unattributed: total - attributed,
        by_endpoint,
        top_misses,
        events,
    }
}

// ============================================================================
// THINGS VELD KNOWS
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct MemoryCard {
    pub id: String,
    pub head: String,
    pub experience_type: String,
    pub created_at: DateTime<Utc>,
    pub importance: f32,
    pub access_count: u32,
    pub anchored: bool,
    /// Tags whose source is caller or intent; `None` when the record has no tag-source row.
    pub authored_tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_days_to_decay: Option<f32>,
    /// Times this memory came back in a recall, counted from the recall event log.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recalled: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Shelf<T> {
    pub count: usize,
    pub items: Vec<T>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FactCard {
    pub id: String,
    pub fact: String,
    pub confidence: f32,
    pub support_count: usize,
    pub source_memories: Vec<String>,
    pub related_entities: Vec<String>,
    pub fact_type: String,
    pub created_at: DateTime<Utc>,
    pub last_reinforced: DateTime<Utc>,
    pub supersedes: Vec<String>,
    /// A single supporting memory: shown, but labelled as one memory's claim.
    pub single_source: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityCard {
    pub name: String,
    pub labels: Vec<String>,
    pub mention_count: usize,
    pub first_learned: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssociationCard {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub strength: f32,
    pub activation_count: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CountCard {
    pub key: String,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct Knows {
    pub settled_facts: Field<Shelf<FactCard>>,
    pub changed_its_mind: Field<Shelf<FactCard>>,
    pub contradictions: Field<Shelf<FactCard>>,
    pub rulings: Field<Shelf<MemoryCard>>,
    pub scars: Field<Shelf<MemoryCard>>,
    pub who_and_what: Field<Shelf<EntityCard>>,
    pub strongest_associations: Field<Shelf<AssociationCard>>,
    pub about_to_forget: Field<Shelf<MemoryCard>>,
    pub pinned: Field<Shelf<MemoryCard>>,
    pub known_but_never_asked: Field<Shelf<MemoryCard>>,
    pub most_asked: Field<Shelf<MemoryCard>>,
    pub asked_but_not_known: Field<Shelf<MissedQuery>>,
    pub suspected_unknowns: Field<Shelf<String>>,
    pub this_week: Field<Shelf<CountCard>>,
    pub where_learned: Field<Shelf<CountCard>>,
    pub sense_of_self: Field<Shelf<MemoryCard>>,
    pub undetermined: Field<Shelf<CountCard>>,
}

fn head(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

/// `(importance / floor)^2`, the same estimate `/api/memory/{id}/health` reports.
pub fn estimated_days_to_decay(importance: f32) -> Option<f32> {
    (importance > DECAY_IMPORTANCE_FLOOR).then(|| (importance / DECAY_IMPORTANCE_FLOOR).powf(2.0))
}

fn memory_card(m: &Memory, sources: Option<&HashMap<MemoryId, TagSources>>) -> MemoryCard {
    let authored_tags = sources.and_then(|s| s.get(&m.id)).map(|row| {
        row.tags
            .iter()
            .filter(|t| t.source.is_authored())
            .map(|t| t.tag.clone())
            .collect()
    });
    MemoryCard {
        id: m.id.0.to_string(),
        head: head(&m.experience.content, 240),
        experience_type: format!("{:?}", m.experience.experience_type),
        created_at: m.created_at,
        importance: m.importance(),
        access_count: m.access_count(),
        anchored: m.is_anchored(),
        authored_tags,
        estimated_days_to_decay: estimated_days_to_decay(m.importance()),
        recalled: None,
    }
}

fn shelf<T>(mut items: Vec<T>, keep: usize) -> Shelf<T> {
    let count = items.len();
    items.truncate(keep);
    Shelf { count, items }
}

fn fact_card(f: &crate::memory::compression::SemanticFact) -> FactCard {
    FactCard {
        id: f.id.clone(),
        fact: f.fact.clone(),
        confidence: f.confidence,
        support_count: f.support_count,
        source_memories: f.source_memories.iter().map(|m| m.0.to_string()).collect(),
        related_entities: f.related_entities.clone(),
        fact_type: format!("{:?}", f.fact_type),
        created_at: f.created_at,
        last_reinforced: f.last_reinforced,
        supersedes: f.supersedes.clone(),
        single_source: f.support_count <= 1,
    }
}

const W_MEMORIES: &str = "MemorySystem::get_all_memories";
const W_FACTS: &str = "FactStore::list (active facts; skips rows it cannot decode)";
const W_ENTITIES: &str = "GraphMemory::get_all_entities (skips rows it cannot decode)";
const W_EDGES: &str = "GraphMemory::get_all_relationships, tier L3, not invalidated";
const W_RECALLS: &str = "RecallEventStore::since";
/// `access_count` is updated in memory on recall and persisted only when something else rewrites
/// the record, so it reads zero after a restart; recall hits are counted from the log instead.
const W_RECALL_HITS: &str = "RecallEventStore hits since the log's horizon";
const W_SOURCES: &str = "TagSourceStore::all";

pub struct KnowsInputs<'a> {
    pub memories: &'a anyhow::Result<Vec<std::sync::Arc<Memory>>>,
    pub sources: &'a anyhow::Result<HashMap<MemoryId, TagSources>>,
    pub facts: anyhow::Result<Vec<crate::memory::compression::SemanticFact>>,
    pub entities: anyhow::Result<Vec<crate::graph_memory::EntityNode>>,
    pub edges: anyhow::Result<Vec<crate::graph_memory::RelationshipEdge>>,
    pub recalls: anyhow::Result<Vec<RecallEvent>>,
    pub now: DateTime<Utc>,
}

pub fn build_knows(i: KnowsInputs<'_>) -> Knows {
    let sources = i.sources.as_ref().ok();
    let mems: Option<&Vec<std::sync::Arc<Memory>>> = i.memories.as_ref().ok();
    let mem_err = || {
        i.memories
            .as_ref()
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default()
    };

    let by_type = |types: &[ExperienceType]| -> Field<Shelf<MemoryCard>> {
        match mems {
            None => Field::unmeasured(mem_err(), W_MEMORIES),
            Some(ms) => {
                let mut v: Vec<&Memory> = ms
                    .iter()
                    .map(|m| m.as_ref())
                    .filter(|m| types.contains(&m.experience.experience_type))
                    .collect();
                v.sort_by_key(|m| std::cmp::Reverse(m.created_at));
                Field::measured(
                    shelf(
                        v.into_iter().map(|m| memory_card(m, sources)).collect(),
                        SHELF_ITEMS,
                    ),
                    W_MEMORIES,
                )
            }
        }
    };
    let memory_shelf = |filter: &dyn Fn(&Memory) -> bool,
                        order: &dyn Fn(&Memory, &Memory) -> std::cmp::Ordering|
     -> Field<Shelf<MemoryCard>> {
        match mems {
            None => Field::unmeasured(mem_err(), W_MEMORIES),
            Some(ms) => {
                let mut v: Vec<&Memory> = ms
                    .iter()
                    .map(|m| m.as_ref())
                    .filter(|m| filter(m))
                    .collect();
                v.sort_by(|a, b| order(a, b));
                Field::measured(
                    shelf(
                        v.into_iter().map(|m| memory_card(m, sources)).collect(),
                        SHELF_ITEMS,
                    ),
                    W_MEMORIES,
                )
            }
        }
    };

    let (settled_facts, changed_its_mind) = match &i.facts {
        Err(e) => (
            Field::unmeasured(format!("{e:#}"), W_FACTS),
            Field::unmeasured(format!("{e:#}"), W_FACTS),
        ),
        Ok(facts) => {
            let mut settled: Vec<&_> = facts
                .iter()
                .filter(|f| f.superseded_by.is_none() && f.confidence >= 0.7)
                .collect();
            settled.sort_by(|a, b| {
                b.support_count
                    .cmp(&a.support_count)
                    .then_with(|| b.confidence.total_cmp(&a.confidence))
            });
            let mut changed: Vec<&_> = facts.iter().filter(|f| !f.supersedes.is_empty()).collect();
            changed.sort_by_key(|f| std::cmp::Reverse(f.created_at));
            (
                Field::measured(
                    shelf(settled.into_iter().map(fact_card).collect(), SHELF_ITEMS),
                    W_FACTS,
                ),
                Field::measured(
                    shelf(changed.into_iter().map(fact_card).collect(), SHELF_ITEMS),
                    W_FACTS,
                ),
            )
        }
    };

    let who_and_what = match &i.entities {
        Err(e) => Field::unmeasured(format!("{e:#}"), W_ENTITIES),
        Ok(entities) => {
            let mut v: Vec<&_> = entities.iter().collect();
            v.sort_by(|a, b| {
                b.mention_count
                    .cmp(&a.mention_count)
                    .then_with(|| a.name.cmp(&b.name))
            });
            Field::measured(
                shelf(
                    v.into_iter()
                        .map(|e| EntityCard {
                            name: e.name.clone(),
                            labels: e.labels.iter().map(|l| format!("{l:?}")).collect(),
                            mention_count: e.mention_count,
                            first_learned: e.created_at,
                            last_seen: e.last_seen_at,
                        })
                        .collect(),
                    SHELF_ITEMS,
                ),
                W_ENTITIES,
            )
        }
    };

    let strongest_associations = match (&i.edges, &i.entities) {
        (Err(e), _) => Field::unmeasured(format!("{e:#}"), W_EDGES),
        (_, Err(e)) => Field::unmeasured(format!("entity names: {e:#}"), W_EDGES),
        (Ok(edges), Ok(entities)) => {
            let names: HashMap<uuid::Uuid, &str> =
                entities.iter().map(|e| (e.uuid, e.name.as_str())).collect();
            let mut v: Vec<&_> = edges
                .iter()
                .filter(|e| {
                    e.invalidated_at.is_none()
                        && matches!(e.tier, crate::graph_memory::EdgeTier::L3Semantic)
                })
                .collect();
            v.sort_by(|a, b| b.strength.total_cmp(&a.strength));
            let name = |id: &uuid::Uuid| {
                names
                    .get(id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| id.to_string())
            };
            Field::measured(
                shelf(
                    v.into_iter()
                        .map(|e| AssociationCard {
                            from: name(&e.from_entity),
                            to: name(&e.to_entity),
                            relation: format!("{:?}", e.relation_type),
                            strength: e.strength,
                            activation_count: e.activation_count,
                        })
                        .collect(),
                    SHELF_ITEMS,
                ),
                W_EDGES,
            )
        }
    };

    let now = i.now;
    let never_age = Duration::hours(NEVER_ASKED_MIN_AGE_HOURS);

    let (most_asked, known_but_never_asked) = match (&i.recalls, mems) {
        (Err(e), _) => (
            Field::unmeasured(format!("{e:#}"), W_RECALL_HITS),
            Field::unmeasured(format!("{e:#}"), W_RECALL_HITS),
        ),
        (_, None) => (
            Field::unmeasured(mem_err(), W_MEMORIES),
            Field::unmeasured(mem_err(), W_MEMORIES),
        ),
        (Ok(events), Some(ms)) => {
            let horizon = events.first().map(|e| e.at);
            let mut hits: HashMap<&str, usize> = HashMap::new();
            for e in events {
                for h in &e.hits {
                    *hits.entry(h.memory_id.as_str()).or_insert(0) += 1;
                }
            }
            let since = horizon
                .map(|h| format!("since {}", h.to_rfc3339()))
                .unwrap_or_else(|| "the recall log is empty".into());
            let card = |m: &Memory| {
                let mut c = memory_card(m, sources);
                c.recalled = Some(hits.get(m.id.0.to_string().as_str()).copied().unwrap_or(0));
                c
            };
            let mut asked: Vec<&Memory> = ms
                .iter()
                .map(|m| m.as_ref())
                .filter(|m| hits.contains_key(m.id.0.to_string().as_str()))
                .collect();
            asked.sort_by_key(|m| {
                std::cmp::Reverse(hits.get(m.id.0.to_string().as_str()).copied().unwrap_or(0))
            });
            let mut most = Field::measured(
                shelf(asked.into_iter().map(card).collect(), SHELF_ITEMS),
                W_RECALL_HITS,
            );
            most.reason = Some(since.clone());

            let never = match horizon {
                Some(h) if now - h >= never_age => {
                    let mut v: Vec<&Memory> = ms
                        .iter()
                        .map(|m| m.as_ref())
                        .filter(|m| {
                            m.created_at <= h && !hits.contains_key(m.id.0.to_string().as_str())
                        })
                        .collect();
                    v.sort_by_key(|m| m.created_at);
                    let mut f = Field::measured(
                        shelf(v.into_iter().map(card).collect(), SHELF_ITEMS),
                        W_RECALL_HITS,
                    );
                    f.reason = Some(since);
                    f
                }
                _ => Field::undetermined(
                    format!(
                        "the recall log has to span {NEVER_ASKED_MIN_AGE_HOURS}h (the p90 revisit \
                         interval) before an unrecalled memory means anything; {since}"
                    ),
                    W_RECALL_HITS,
                ),
            };
            (most, never)
        }
    };

    let asked_but_not_known = match &i.recalls {
        Err(e) => Field::unmeasured(format!("{e:#}"), W_RECALLS),
        Ok(events) => {
            let r = summarise_recalls(
                events.clone(),
                events.first().map(|e| e.at),
                DateTime::<Utc>::MIN_UTC,
                0,
            );
            Field::measured(
                Shelf {
                    count: r.misses,
                    items: r.top_misses,
                },
                W_RECALLS,
            )
        }
    };

    let this_week = match mems {
        None => Field::unmeasured(mem_err(), W_MEMORIES),
        Some(ms) => {
            let since = now - Duration::days(7);
            let mut by: BTreeMap<String, usize> = BTreeMap::new();
            for m in ms.iter().filter(|m| m.created_at >= since) {
                *by.entry(format!("{:?}", m.experience.experience_type))
                    .or_insert(0) += 1;
            }
            let mut items: Vec<CountCard> = by
                .into_iter()
                .map(|(key, count)| CountCard { key, count })
                .collect();
            items.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.key.cmp(&b.key)));
            Field::measured(shelf(items, SHELF_ITEMS), W_MEMORIES)
        }
    };

    let where_learned = match mems {
        None => Field::unmeasured(mem_err(), W_MEMORIES),
        Some(ms) => {
            let mut by: BTreeMap<String, usize> = BTreeMap::new();
            for m in ms.iter() {
                let project = m.experience.tags.iter().find_map(|t| {
                    t.strip_prefix("project:")
                        .or_else(|| t.strip_prefix("veld-project:"))
                        .map(str::to_lowercase)
                });
                *by.entry(project.unwrap_or_else(|| "(no project tag)".into()))
                    .or_insert(0) += 1;
            }
            let mut items: Vec<CountCard> = by
                .into_iter()
                .map(|(key, count)| CountCard { key, count })
                .collect();
            items.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.key.cmp(&b.key)));
            Field::measured(shelf(items, SHELF_ITEMS), W_MEMORIES)
        }
    };

    let undetermined = match (mems, sources) {
        (None, _) => Field::unmeasured(mem_err(), W_MEMORIES),
        (Some(_), None) => Field::unmeasured(
            i.sources
                .as_ref()
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default(),
            W_SOURCES,
        ),
        (Some(ms), Some(s)) => {
            let without = ms.iter().filter(|m| !s.contains_key(&m.id)).count();
            Field::measured(
                Shelf {
                    count: without,
                    items: vec![CountCard {
                        key: "memories without a tag-source row".into(),
                        count: without,
                    }],
                },
                W_SOURCES,
            )
        }
    };

    Knows {
        settled_facts,
        changed_its_mind,
        contradictions: Field::undetermined(
            "no contradiction detector exists; conflicting values need value semantics",
            "none",
        ),
        rulings: by_type(&[ExperienceType::Decision]),
        scars: by_type(&[
            ExperienceType::Error,
            ExperienceType::Learning,
            ExperienceType::Pattern,
        ]),
        who_and_what,
        strongest_associations,
        about_to_forget: memory_shelf(
            &|m| {
                !m.is_anchored()
                    && estimated_days_to_decay(m.importance())
                        .is_none_or(|d| d < ABOUT_TO_FORGET_DAYS)
            },
            &|a, b| a.importance().total_cmp(&b.importance()),
        ),
        pinned: memory_shelf(&|m| m.is_anchored(), &|a, b| {
            b.created_at.cmp(&a.created_at)
        }),
        known_but_never_asked,
        most_asked,
        asked_but_not_known,
        suspected_unknowns: Field::unmeasured(
            "Planet-X candidates are computed by spine-skills get_planet_x, not by veld",
            "none",
        ),
        this_week,
        where_learned,
        sense_of_self: memory_shelf(
            &|m| m.experience.content.starts_with("OVERLOOK_CONCEPTUAL_SELF"),
            &|a, b| b.created_at.cmp(&a.created_at),
        ),
        undetermined,
    }
}

// ============================================================================
// HANDLERS
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct TagsQuery {
    pub user_id: Option<String>,
    #[serde(default)]
    pub include_hidden: bool,
    pub limit: Option<usize>,
}

/// GET /api/ui/tags — whole-population tag aggregation with plumbing hidden by default.
pub async fn ui_tags(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(q): Query<TagsQuery>,
) -> Result<Json<UiEnvelope<Field<TagsReport>>>, AppError> {
    let (user_id, binding) = resolve_ui_user(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    let memory = state
        .get_user_memory_async(&user_id)
        .await
        .map_err(AppError::Internal)?;
    let params = TagsParams {
        include_hidden: q.include_hidden,
        limit: q.limit.unwrap_or(300).clamp(1, 5_000),
        ..TagsParams::default()
    };
    let rule = PlumbingRule::from_env();
    let report = tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        let memories = guard.get_all_memories()?;
        let inputs: Vec<TagInput> = memories
            .iter()
            .map(|m| TagInput::from(m.as_ref()))
            .collect();
        let sources = guard.tag_source_store().all();
        let report = aggregate_tags(&inputs, sources.as_ref().ok(), &rule, &params, Utc::now());
        Ok::<_, anyhow::Error>((report, sources.err().map(|e| format!("{e:#}"))))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("tags task panicked: {e}")))?;

    let field = match report {
        Err(e) => Field::unmeasured(format!("{e:#}"), W_MEMORIES),
        Ok((report, None)) => Field::measured(report, W_MEMORIES),
        // The counts are real; only the origin split is unknown, so say exactly that.
        Ok((report, Some(source_err))) => Field {
            state: FieldState::Measured,
            value: Some(report),
            reason: Some(format!(
                "origin counts unmeasured — tag-source ledger unreadable: {source_err}"
            )),
            witness: W_MEMORIES,
        },
    };
    Ok(Json(envelope(user_id, binding, "veld /api/ui/tags", field)))
}

#[derive(Debug, Deserialize)]
pub struct RecallLogQuery {
    pub user_id: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

/// GET /api/ui/recall-log — what was asked, misses included.
pub async fn ui_recall_log(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(q): Query<RecallLogQuery>,
) -> Result<Json<UiEnvelope<Field<RecallLogReport>>>, AppError> {
    let (user_id, binding) = resolve_ui_user(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    let memory = state
        .get_user_memory_async(&user_id)
        .await
        .map_err(AppError::Internal)?;
    let since = q.since.unwrap_or(DateTime::<Utc>::MIN_UTC);
    let limit = q.limit.unwrap_or(200).clamp(1, 5_000);
    let result = tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        let store = guard.recall_event_store();
        let horizon = store.horizon()?;
        let events = store.since(since)?;
        Ok::<_, anyhow::Error>(summarise_recalls(events, horizon, since, limit))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("recall-log task panicked: {e}")))?;
    Ok(Json(envelope(
        user_id,
        binding,
        "veld /api/ui/recall-log",
        Field::from_result(result, W_RECALLS),
    )))
}

#[derive(Debug, Deserialize)]
pub struct KnowsQuery {
    pub user_id: Option<String>,
}

/// GET /api/ui/knows — the almanac. Each shelf states its own source and state.
pub async fn ui_knows(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(q): Query<KnowsQuery>,
) -> Result<Json<UiEnvelope<Knows>>, AppError> {
    let (user_id, binding) = resolve_ui_user(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    let memory = state
        .get_user_memory_async(&user_id)
        .await
        .map_err(AppError::Internal)?;
    let graph = state.get_user_graph_async(&user_id).await;
    let uid = user_id.clone();
    let knows = tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        let memories = guard.get_all_memories();
        let sources = guard.tag_source_store().all();
        let facts = guard.get_facts(&uid, 100_000);
        let recalls = guard.recall_event_store().since(DateTime::<Utc>::MIN_UTC);
        let (entities, edges) = match &graph {
            Ok(g) => {
                let g = g.read();
                (g.get_all_entities(), g.get_all_relationships())
            }
            Err(e) => (
                Err(anyhow::anyhow!("graph unavailable: {e:#}")),
                Err(anyhow::anyhow!("graph unavailable: {e:#}")),
            ),
        };
        build_knows(KnowsInputs {
            memories: &memories,
            sources: &sources,
            facts,
            entities,
            edges,
            recalls,
            now: Utc::now(),
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("knows task panicked: {e}")))?;
    Ok(Json(envelope(
        user_id,
        binding,
        "veld /api/ui/knows",
        knows,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::tag_sources::SourcedTag;
    use crate::redaction::RedactionReport;

    fn input(tags: &[&str], days_ago: i64, now: DateTime<Utc>) -> TagInput {
        TagInput {
            id: MemoryId(uuid::Uuid::new_v4()),
            created_at: now - Duration::days(days_ago),
            tags: tags.iter().map(|t| t.to_string()).collect(),
        }
    }

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-10-07T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn plumbing_malformed_and_timestamps_are_classified() {
        let rule = PlumbingRule::with_extra(vec![]);
        for t in [
            "scope:claude-code",
            "automated",
            "hook",
            "Source:x",
            "redacted:key",
            "session-close",
        ] {
            assert_eq!(rule.classify(t), TagClass::Plumbing, "{t}");
        }
        assert_eq!(
            rule.classify(r#"classification":"needs-human"#),
            TagClass::Malformed
        );
        assert_eq!(rule.classify("2026-10-07T04:"), TagClass::Timestamp);
        assert_eq!(rule.classify("2026-09-17"), TagClass::Timestamp);
        for t in ["project:veld", "rust", "PL/I", "2026-plan", "hooks"] {
            assert_eq!(rule.classify(t), TagClass::Visible, "{t}");
        }
    }

    #[test]
    fn extra_plumbing_from_the_environment_is_added_not_replacing() {
        let rule = PlumbingRule::with_extra(vec!["bifocal-probe".into(), "fleet:".into()]);
        assert_eq!(rule.classify("bifocal-probe"), TagClass::Plumbing);
        assert_eq!(rule.classify("fleet:commitwork-93"), TagClass::Plumbing);
        assert_eq!(rule.classify("hook"), TagClass::Plumbing);
    }

    #[test]
    fn hidden_by_default_counted_and_weighted_by_rarity() {
        let n = now();
        let mems = vec![
            input(&["scope:claude-code", "rust", "veld"], 1, n),
            input(&["scope:claude-code", "rust"], 2, n),
            input(&["scope:claude-code", "cobol"], 3, n),
            input(&["scope:claude-code", "2026-10-07T04:", "x\"y"], 30, n),
        ];
        let rule = PlumbingRule::with_extra(vec![]);
        let r = aggregate_tags(
            &mems,
            Some(&HashMap::new()),
            &rule,
            &TagsParams::default(),
            n,
        );
        assert!(r.tags.iter().all(|t| t.class == TagClass::Visible));
        assert_eq!(r.hidden[&TagClass::Plumbing].tags, 1);
        assert_eq!(r.hidden[&TagClass::Plumbing].occurrences, 4);
        assert_eq!(r.hidden[&TagClass::Timestamp].tags, 1);
        assert_eq!(r.hidden[&TagClass::Malformed].tags, 1);
        assert_eq!(r.population.undetermined, 4);
        let rust = r.tags.iter().find(|t| t.tag == "rust").unwrap();
        assert_eq!(rust.memories, 2);
        assert!((rust.weight - 2.0 * 2f32.ln()).abs() < 1e-5);
        assert_eq!(rust.weekly.iter().sum::<usize>(), 2);
        assert_eq!(rust.co_occurring[0].tag, "veld");

        let all = aggregate_tags(
            &mems,
            None,
            &rule,
            &TagsParams {
                include_hidden: true,
                ..TagsParams::default()
            },
            n,
        );
        let scope = all
            .tags
            .iter()
            .find(|t| t.tag == "scope:claude-code")
            .unwrap();
        assert_eq!(scope.weight, 0.0, "a tag on every memory weighs nothing");
        assert_eq!(scope.class, TagClass::Plumbing);
    }

    #[test]
    fn origin_comes_from_the_ledger_and_absence_is_undetermined() {
        let n = now();
        let a = input(&["veld", "Alice"], 1, n);
        let b = input(&["veld"], 1, n);
        let mut sources = HashMap::new();
        sources.insert(
            a.id.clone(),
            TagSources {
                v: 1,
                tags: vec![
                    SourcedTag {
                        tag: "veld".into(),
                        source: TagSource::Caller,
                        redacted: false,
                    },
                    SourcedTag {
                        tag: "Alice".into(),
                        source: TagSource::Ner,
                        redacted: false,
                    },
                ],
                redacted: (&RedactionReport::default()).into(),
            },
        );
        let r = aggregate_tags(
            &[a, b],
            Some(&sources),
            &PlumbingRule::with_extra(vec![]),
            &TagsParams::default(),
            n,
        );
        let veld = r.tags.iter().find(|t| t.tag == "veld").unwrap();
        assert_eq!(
            veld.origin,
            OriginCounts {
                authored: 1,
                derived: 0,
                undetermined: 1
            }
        );
        assert_eq!(veld.sources["caller"], 1);
        let alice = r.tags.iter().find(|t| t.tag == "Alice").unwrap();
        assert_eq!(alice.origin.derived, 1);
        assert_eq!(r.population.with_tag_sources, 1);
    }

    #[test]
    fn a_template_stamped_on_the_same_memories_is_one_cluster_with_its_producer() {
        let n = now();
        let template = ["findings", "CVEs", "checks", "repos", "slice", "lifecycle"];
        let mut mems = Vec::new();
        for i in 0..12 {
            let mut tags = vec!["scope:commitwork-sweep"];
            tags.extend(template);
            mems.push(input(&tags, i, n));
        }
        mems.push(input(&["scope:claude-code", "findings"], 1, n));
        let r = aggregate_tags(
            &mems,
            None,
            &PlumbingRule::with_extra(vec![]),
            &TagsParams::default(),
            n,
        );
        assert_eq!(
            r.clusters.len(),
            1,
            "findings is on one more memory, so it is not in the set"
        );
        assert_eq!(r.clusters[0].tags.len(), 5);
        assert_eq!(r.clusters[0].memories, 12);
        assert_eq!(r.clusters[0].producer.name, "scope:commitwork-sweep");
        assert_eq!(r.clusters[0].producer.share, 1.0);
        let cves = r.tags.iter().find(|t| t.tag == "CVEs").unwrap();
        assert_eq!(cves.cluster, Some(0));
    }

    #[test]
    fn recall_summary_counts_misses_and_attribution() {
        let mk = |q: &str, hits: usize, ep: Option<&str>| {
            RecallEvent::new(
                "/api/recall",
                q,
                ep.map(String::from),
                RecallEvent::hits_from_scored((0..hits).map(|_| ("m", 0.5))),
            )
        };
        let events = vec![
            mk("a", 0, None),
            mk("a", 0, Some("t1")),
            mk("b", 2, Some("t1")),
        ];
        let r = summarise_recalls(events, None, DateTime::<Utc>::MIN_UTC, 1);
        assert_eq!(r.total, 3);
        assert_eq!(r.misses, 2);
        assert_eq!(r.attributed, 2);
        assert_eq!(r.top_misses[0].query, "a");
        assert_eq!(r.top_misses[0].times, 2);
        assert_eq!(r.events.len(), 1);
        assert_eq!(r.events[0].query, "b", "newest first");
    }

    #[test]
    fn ui_user_binding_refuses_without_a_binding_and_rejects_other_users() {
        // Env is process-global; this test owns UI_USER_ENV and restores it.
        let previous = std::env::var(UI_USER_ENV).ok();
        std::env::remove_var(UI_USER_ENV);
        assert!(
            resolve_ui_user(Some("portll"), None).is_err(),
            "no binding, no read"
        );

        std::env::set_var(UI_USER_ENV, "portll");
        assert_eq!(
            resolve_ui_user(None, None).unwrap(),
            ("portll".into(), Binding::UiUserEnv)
        );
        assert_eq!(resolve_ui_user(Some("portll"), None).unwrap().0, "portll");
        assert!(resolve_ui_user(Some("someone-else"), None).is_err());

        let tenant = AuthenticatedUser {
            user_id: "tenant-a".into(),
        };
        assert_eq!(
            resolve_ui_user(None, Some(&tenant)).unwrap(),
            ("tenant-a".into(), Binding::TenantKey),
            "a tenant-bound key outranks the env binding"
        );
        assert!(resolve_ui_user(Some("portll"), Some(&tenant)).is_err());

        match previous {
            Some(v) => std::env::set_var(UI_USER_ENV, v),
            None => std::env::remove_var(UI_USER_ENV),
        }
    }

    #[test]
    fn asked_shelves_come_from_the_recall_log_and_wait_for_it_to_span_the_revisit_interval() {
        let memories: anyhow::Result<Vec<std::sync::Arc<Memory>>> = Ok(vec![]);
        let sources: anyhow::Result<HashMap<MemoryId, TagSources>> = Ok(HashMap::new());
        let at = |hours_ago: i64| {
            let mut e = RecallEvent::new("/api/recall", "q", None, vec![]);
            e.at = now() - Duration::hours(hours_ago);
            e
        };
        let knows = |recalls: Vec<RecallEvent>| {
            build_knows(KnowsInputs {
                memories: &memories,
                sources: &sources,
                facts: Ok(vec![]),
                entities: Ok(vec![]),
                edges: Ok(vec![]),
                recalls: Ok(recalls),
                now: now(),
            })
        };

        let young = knows(vec![at(5)]);
        assert_eq!(young.known_but_never_asked.state, FieldState::Undetermined);
        assert!(young
            .known_but_never_asked
            .reason
            .as_deref()
            .unwrap()
            .contains("238h"));
        assert_eq!(young.most_asked.state, FieldState::Measured);
        assert!(young
            .most_asked
            .reason
            .as_deref()
            .unwrap()
            .starts_with("since "));

        let old = knows(vec![at(NEVER_ASKED_MIN_AGE_HOURS + 1)]);
        assert_eq!(old.known_but_never_asked.state, FieldState::Measured);

        let empty = knows(vec![]);
        assert_eq!(empty.known_but_never_asked.state, FieldState::Undetermined);
        assert_eq!(
            empty.most_asked.reason.as_deref(),
            Some("the recall log is empty")
        );
    }

    #[test]
    fn decay_estimate_matches_the_health_route() {
        assert_eq!(estimated_days_to_decay(0.05), None);
        assert_eq!(estimated_days_to_decay(0.5), Some(100.0));
    }

    #[test]
    fn a_failed_source_is_unmeasured_never_an_empty_shelf() {
        let memories: anyhow::Result<Vec<std::sync::Arc<Memory>>> =
            Err(anyhow::anyhow!("scan failed"));
        let sources: anyhow::Result<HashMap<MemoryId, TagSources>> = Ok(HashMap::new());
        let k = build_knows(KnowsInputs {
            memories: &memories,
            sources: &sources,
            facts: Err(anyhow::anyhow!("facts down")),
            entities: Ok(vec![]),
            edges: Ok(vec![]),
            recalls: Ok(vec![]),
            now: now(),
        });
        assert_eq!(k.rulings.state, FieldState::Unmeasured);
        assert!(k.rulings.reason.as_deref().unwrap().contains("scan failed"));
        assert_eq!(k.settled_facts.state, FieldState::Unmeasured);
        assert_eq!(k.who_and_what.state, FieldState::Measured);
        assert_eq!(
            k.who_and_what.value.as_ref().unwrap().count,
            0,
            "measured zero is a real zero"
        );
        assert_eq!(k.contradictions.state, FieldState::Undetermined);
        assert_eq!(k.suspected_unknowns.state, FieldState::Unmeasured);
    }
}
