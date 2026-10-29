//! Entity sentiment over HTTP.
//!
//! `POST /api/entity/sentiment` aggregates a user's stance toward one entity from the
//! episodes that mention it, recomputed on every call (see [`crate::memory::sentiment`]).
//! `POST /api/sentiment/analyze` reads a piece of text and touches no store.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{extract::State, response::Json};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::sentiment::{
    self, Holder, LikertProjection, Modifier, Observation, Opinion, StanceSummary, LIKERT_SCALES,
};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

const DEFAULT_SCALE: u8 = 5;
const DEFAULT_HALF_LIFE_DAYS: f32 = 180.0;
const DEFAULT_MAX_EPISODES: usize = 1_000;
const MAX_EPISODES_CAP: usize = 10_000;
const DEFAULT_EVIDENCE: usize = 5;
const MAX_EVIDENCE: usize = 50;
const MAX_HOLDERS: usize = 10;
const SNIPPET_CHARS: usize = 240;

fn default_scale() -> u8 {
    DEFAULT_SCALE
}

fn default_half_life_days() -> f32 {
    DEFAULT_HALF_LIFE_DAYS
}

fn default_max_episodes() -> usize {
    DEFAULT_MAX_EPISODES
}

fn default_evidence() -> usize {
    DEFAULT_EVIDENCE
}

fn validate_scale(scale: u8) -> Result<(), AppError> {
    if LIKERT_SCALES.contains(&scale) {
        return Ok(());
    }
    Err(AppError::InvalidInput {
        field: "scale".to_string(),
        reason: format!(
            "Likert scale must be {}..={}, got {scale}",
            LIKERT_SCALES.start(),
            LIKERT_SCALES.end()
        ),
    })
}

fn snippet(text: &str, opinion: &Opinion) -> String {
    let clause = text.get(opinion.clause.clone()).unwrap_or(text).trim();
    match clause.char_indices().nth(SNIPPET_CHARS) {
        Some((cut, _)) => format!("{}…", &clause[..cut]),
        None => clause.to_string(),
    }
}

#[derive(Debug, Deserialize)]
pub struct EntitySentimentRequest {
    pub user_id: String,
    pub entity: String,
    /// Likert width, 3..=11.
    #[serde(default = "default_scale")]
    pub scale: u8,
    /// Opinions lose half their weight every this many days; 0 turns decay off.
    #[serde(default = "default_half_life_days")]
    pub half_life_days: f32,
    /// Episodes read, up to 10 000.
    #[serde(default = "default_max_episodes")]
    pub max_episodes: usize,
    /// Strongest opinions returned as evidence, up to 50.
    #[serde(default = "default_evidence")]
    pub evidence: usize,
    /// Restrict `stance` to one holder: `"speaker"`, `"other"`, or a name such as a
    /// transcript speaker.
    pub holder: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EntitySentimentResponse {
    /// The entity's canonical name.
    pub entity: String,
    /// Names searched for in episode text: the canonical name and its aliases.
    pub surface_forms: Vec<String>,
    pub episodes_scanned: usize,
    /// Episodes whose text names the entity.
    pub episodes_mentioning: usize,
    /// Mentions with no opinion attached.
    pub neutral_mentions: usize,
    /// All opinions, or the requested holder's; `None` when there are none.
    pub stance: Option<StanceSummary>,
    /// Stance per holder, heaviest evidence first.
    pub by_holder: Vec<HolderStance>,
    pub evidence: Vec<SentimentEvidence>,
}

#[derive(Debug, Serialize)]
pub struct HolderStance {
    pub holder: Holder,
    pub stance: StanceSummary,
}

#[derive(Debug, Serialize)]
pub struct SentimentEvidence {
    pub episode_id: String,
    pub at: DateTime<Utc>,
    pub holder: Holder,
    pub cue: String,
    pub valence: f32,
    pub weight: f32,
    pub modifiers: Vec<Modifier>,
    pub snippet: String,
}

struct Found<'a> {
    opinion: Opinion,
    observation: Observation,
    episode_id: String,
    content: &'a str,
}

impl Found<'_> {
    fn strength(&self) -> f32 {
        self.opinion.valence.abs() * self.opinion.weight
    }
}

fn holder_matches(holder: &Holder, wanted: &str) -> bool {
    match holder {
        Holder::Speaker => wanted.eq_ignore_ascii_case("speaker"),
        Holder::Named(name) => name.eq_ignore_ascii_case(wanted),
        Holder::Other => wanted.eq_ignore_ascii_case("other"),
    }
}

fn surface_forms(canonical: &str, aliases: Option<&String>, requested: &str) -> Vec<String> {
    let mut forms: Vec<String> = Vec::new();
    let candidates = std::iter::once(canonical)
        .chain(aliases.into_iter().flat_map(|a| a.split(", ")))
        .chain(std::iter::once(requested));
    for form in candidates.map(str::trim).filter(|f| !f.is_empty()) {
        if !forms.iter().any(|known| known.eq_ignore_ascii_case(form)) {
            forms.push(form.to_string());
        }
    }
    forms
}

/// POST /api/entity/sentiment — a user's stance toward one entity, read from every
/// episode that mentions it and placed on a Likert scale.
pub async fn entity_sentiment(
    State(state): State<AppState>,
    Json(req): Json<EntitySentimentRequest>,
) -> Result<Json<EntitySentimentResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.entity).map_validation_err("entity")?;
    validate_scale(req.scale)?;
    if !req.half_life_days.is_finite() || req.half_life_days < 0.0 {
        return Err(AppError::InvalidInput {
            field: "half_life_days".to_string(),
            reason: "must be a finite number of days, 0 or more".to_string(),
        });
    }

    let graph_memory = state
        .get_user_graph_async(&req.user_id)
        .await
        .map_err(AppError::Internal)?;

    tokio::task::spawn_blocking(move || {
        let graph = graph_memory.read();
        let node = graph
            .find_entity_by_name(&req.entity)
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::InvalidInput {
                field: "entity".to_string(),
                reason: format!("Entity not found: {}", req.entity),
            })?;
        let episodes = graph
            .get_episodes_by_entity_limited(&node.uuid, req.max_episodes.min(MAX_EPISODES_CAP))
            .map_err(AppError::Internal)?;
        drop(graph);

        let forms = surface_forms(&node.name, node.attributes.get("aliases"), &req.entity);
        let targets: Vec<&str> = forms.iter().map(String::as_str).collect();

        let mut episodes_mentioning = 0;
        let mut neutral_mentions = 0;
        let mut found: Vec<Found> = Vec::new();
        for episode in &episodes {
            let read = sentiment::analyze(&episode.content, &targets);
            if read.mentions == 0 {
                continue;
            }
            episodes_mentioning += 1;
            neutral_mentions += read.mentions.saturating_sub(read.opinions.len());
            found.extend(read.opinions.into_iter().map(|opinion| Found {
                observation: Observation {
                    valence: opinion.valence,
                    weight: opinion.weight,
                    at: episode.valid_at,
                },
                opinion,
                episode_id: episode.uuid.to_string(),
                content: &episode.content,
            }));
        }

        let now = Utc::now();
        let mut grouped: BTreeMap<&Holder, Vec<Observation>> = BTreeMap::new();
        for f in &found {
            grouped
                .entry(&f.opinion.holder)
                .or_default()
                .push(f.observation);
        }
        let mut by_holder: Vec<HolderStance> = grouped
            .into_iter()
            .filter_map(|(holder, observations)| {
                sentiment::summarize(&observations, now, req.half_life_days, req.scale).map(
                    |stance| HolderStance {
                        holder: holder.clone(),
                        stance,
                    },
                )
            })
            .collect();
        by_holder.sort_by(|a, b| {
            b.stance
                .effective_weight
                .total_cmp(&a.stance.effective_weight)
        });
        by_holder.truncate(MAX_HOLDERS);

        let mut selected: Vec<&Found> = found
            .iter()
            .filter(|f| {
                req.holder
                    .as_deref()
                    .is_none_or(|wanted| holder_matches(&f.opinion.holder, wanted))
            })
            .collect();
        let observations: Vec<Observation> = selected.iter().map(|f| f.observation).collect();
        let stance = sentiment::summarize(&observations, now, req.half_life_days, req.scale);

        selected.sort_by(|a, b| {
            b.strength()
                .total_cmp(&a.strength())
                .then_with(|| b.observation.at.cmp(&a.observation.at))
        });
        let evidence = selected
            .into_iter()
            .take(req.evidence.min(MAX_EVIDENCE))
            .map(|f| SentimentEvidence {
                episode_id: f.episode_id.clone(),
                at: f.observation.at,
                holder: f.opinion.holder.clone(),
                cue: f.opinion.cue.clone(),
                valence: f.opinion.valence,
                weight: f.opinion.weight,
                modifiers: f.opinion.modifiers.clone(),
                snippet: snippet(f.content, &f.opinion),
            })
            .collect();

        Ok(Json(EntitySentimentResponse {
            entity: node.name.clone(),
            surface_forms: forms.clone(),
            episodes_scanned: episodes.len(),
            episodes_mentioning,
            neutral_mentions,
            stance,
            by_holder,
            evidence,
        }))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Entity sentiment task panicked: {e}")))?
}

#[derive(Debug, Deserialize)]
pub struct AnalyzeSentimentRequest {
    pub text: String,
    /// Entities to read stance toward; matched on whole words, case-insensitively.
    #[serde(default)]
    pub targets: Vec<String>,
    /// Likert width, 3..=11.
    #[serde(default = "default_scale")]
    pub scale: u8,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeSentimentResponse {
    /// Whole-text valence in (-1, 1).
    pub valence: f32,
    pub likert: LikertProjection,
    /// Target mentions located, with or without an opinion.
    pub mentions: usize,
    pub opinions: Vec<AnalyzedOpinion>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzedOpinion {
    #[serde(flatten)]
    pub opinion: Opinion,
    pub likert: LikertProjection,
    pub snippet: String,
}

/// POST /api/sentiment/analyze — whole-text valence and per-target opinions for one text.
pub async fn analyze_sentiment(
    Json(req): Json<AnalyzeSentimentRequest>,
) -> Result<Json<AnalyzeSentimentResponse>, AppError> {
    validation::validate_content(&req.text, false).map_validation_err("text")?;
    validation::validate_entities(&req.targets).map_validation_err("targets")?;
    validate_scale(req.scale)?;

    let targets: Vec<&str> = req.targets.iter().map(String::as_str).collect();
    let read = sentiment::analyze(&req.text, &targets);
    let opinions = read
        .opinions
        .into_iter()
        .map(|opinion| AnalyzedOpinion {
            likert: sentiment::project(opinion.valence, req.scale),
            snippet: snippet(&req.text, &opinion),
            opinion,
        })
        .collect();

    Ok(Json(AnalyzeSentimentResponse {
        valence: read.valence,
        likert: sentiment::project(read.valence, req.scale),
        mentions: read.mentions,
        opinions,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_forms_keep_the_canonical_name_first_and_drop_duplicates() {
        let aliases = "Rustlang, rust, ".to_string();
        let forms = surface_forms("Rust", Some(&aliases), "RUST");
        assert_eq!(forms, vec!["Rust".to_string(), "Rustlang".to_string()]);
    }

    #[test]
    fn holder_filter_names_the_speaker_or_a_person() {
        assert!(holder_matches(&Holder::Speaker, "Speaker"));
        assert!(holder_matches(
            &Holder::Named("Caroline".into()),
            "caroline"
        ));
        assert!(!holder_matches(
            &Holder::Named("Caroline".into()),
            "speaker"
        ));
    }

    #[test]
    fn scale_outside_the_likert_range_is_refused() {
        assert!(validate_scale(5).is_ok());
        assert!(validate_scale(2).is_err());
        assert!(validate_scale(12).is_err());
    }

    #[tokio::test]
    async fn analyze_places_each_opinion_on_the_requested_scale() {
        let Json(reply) = analyze_sentiment(Json(AnalyzeSentimentRequest {
            text: "I absolutely love Rust but I can't stand Java!".to_string(),
            targets: vec!["Rust".to_string(), "Java".to_string()],
            scale: 7,
        }))
        .await
        .unwrap();
        assert_eq!(reply.mentions, 2);
        let of = |t: &str| {
            reply
                .opinions
                .iter()
                .find(|o| o.opinion.target == t)
                .unwrap()
        };
        assert!(of("Rust").likert.rounded >= 6);
        assert!(of("Java").likert.rounded <= 2);
        assert_eq!(of("Java").snippet, "I can't stand Java");
    }
}
