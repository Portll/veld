//! Index text for conversation turns: what the embedder and the BM25 index see.
//!
//! A conversation turn on its own is often unretrievable: "I've had them for 3
//! years" carries nothing that a query like "how long has Nate had his turtles"
//! can match, and a date that lives only in `created_at` never reaches either
//! index. This module derives the text the two indexes receive from the turn,
//! its `created_at` and the turn that preceded it in the same conversation.
//!
//! Index text is derived from content and never stored as content: the record's
//! `content` is unchanged, there is no schema change, and a re-index recomputes
//! the same text from the same inputs (ADR 0009).
//!
//! The two indexes get different text, because they match differently:
//!
//! - the **embedding** gets the date and the prior turn's text, so the
//!   semantic neighbourhood of the turn includes what it was answering;
//! - the **BM25 document** gets the date (ISO and spelled out, so "1 May 2022"
//!   matches) and the prior turn's entities, which are the tokens a question
//!   about the turn would use; the prior turn's full text would make every
//!   turn a lexical match for the turn before it.
//!
//! The prior turn comes from `preceding_memory_id` when the writer threaded the
//! episode; otherwise it is the latest conversation memory sharing the same
//! conversation key (an `episode_id` or a `conv-*` tag) within the session gap
//! (two hours), ordered by sequence number where both turns carry one.
//! Resolution lives on [`crate::memory::MemorySystem`]; this module is the pure
//! text derivation plus the chain bookkeeping it needs.

use chrono::{DateTime, Datelike, Utc};

use super::types::{
    CodeContext, ContextId, ConversationContext, DocumentContext, EmotionalContext,
    EnvironmentContext, EpisodeContext, Experience, ExperienceType, Memory, MemoryId,
    ProjectContext, RichContext, SemanticContext, SourceContext, TemporalContext, UserContext,
};

/// Tag prefix that names a conversation when no `episode_id` is set.
pub const CONVERSATION_TAG_PREFIX: &str = "conv-";

/// Longest slice of the prior turn carried into the embedding text, in characters.
/// MiniLM truncates at 256 word pieces; the current turn must keep the first claim
/// on the model's attention, so the prior turn is capped well under that.
pub const PRIOR_TURN_MAX_CHARS: usize = 320;

/// Most prior-turn entities appended to the BM25 document.
pub const PRIOR_ENTITY_MAX: usize = 12;

/// Inter-turn gap beyond which two turns are not the same session, in seconds.
/// Matches `wavelet_sessions::detect_sessions`'s fixed threshold.
pub const SESSION_GAP_SECS: i64 = 7_200;

/// Where the prior turn was found. Recorded in `veld_turn_context_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorSource {
    /// The writer sent `preceding_memory_id`.
    PrecedingId,
    /// Resolved from the latest turn sharing the conversation key.
    ConversationKey,
}

impl PriorSource {
    pub fn label(self) -> &'static str {
        match self {
            PriorSource::PrecedingId => "preceding_id",
            PriorSource::ConversationKey => "conversation_key",
        }
    }
}

/// The turn that preceded the one being indexed.
#[derive(Debug, Clone)]
pub struct PriorTurn {
    pub id: MemoryId,
    pub content: String,
    pub entities: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub sequence_number: Option<u32>,
    pub source: PriorSource,
}

impl PriorTurn {
    pub fn from_memory(memory: &Memory, source: PriorSource) -> Self {
        Self {
            id: memory.id.clone(),
            content: memory.experience.content.clone(),
            entities: memory.experience.entities.clone(),
            created_at: memory.created_at,
            sequence_number: sequence_number_of(&memory.experience),
            source,
        }
    }
}

/// The text each index receives for one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnIndexText {
    /// Fed to the embedder (primary and secondary).
    pub embedding: String,
    /// The BM25 `content` field.
    pub bm25_content: String,
    /// The BM25 `entities` field: the turn's own entities plus the prior turn's.
    pub bm25_entities: Vec<String>,
}

/// What ingest resolved for one turn: the text its indexes receive and the turn it
/// was found to follow, which `remember` records in the successor index.
#[derive(Debug, Clone)]
pub struct ContextualisedTurn {
    pub text: TurnIndexText,
    pub prior: Option<MemoryId>,
}

/// Tail of the per-conversation chain: the latest turn seen for a conversation key.
#[derive(Debug, Clone)]
pub struct ChainTail {
    pub id: MemoryId,
    pub created_at: DateTime<Utc>,
    pub sequence_number: Option<u32>,
}

/// True when this experience is a conversation turn that gets contextualised.
pub fn applies(experience: &Experience) -> bool {
    matches!(experience.experience_type, ExperienceType::Conversation)
}

/// The key that groups turns into one conversation: the episode id when set,
/// else the first `conv-*` tag.
pub fn conversation_key(experience: &Experience) -> Option<String> {
    if let Some(id) = experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.episode_id.as_deref())
        .filter(|id| !id.trim().is_empty())
    {
        return Some(id.to_string());
    }
    experience
        .tags
        .iter()
        .chain(experience.entities.iter())
        .find(|tag| is_conversation_tag(tag))
        .cloned()
}

/// The explicit predecessor, when the writer threaded the episode.
pub fn preceding_memory_id(experience: &Experience) -> Option<MemoryId> {
    experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.preceding_memory_id.as_deref())
        .and_then(|raw| uuid::Uuid::parse_str(raw.trim()).ok())
        .map(MemoryId)
}

/// The turn's position in its episode, when the writer numbered it.
pub fn sequence_number_of(experience: &Experience) -> Option<u32> {
    experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.sequence_number)
}

fn is_conversation_tag(tag: &str) -> bool {
    tag.len() > CONVERSATION_TAG_PREFIX.len()
        && tag
            .get(..CONVERSATION_TAG_PREFIX.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(CONVERSATION_TAG_PREFIX))
}

/// Whether `tail` can precede a turn stored at `created_at` with `sequence_number`.
///
/// Turns are ordered by sequence number, not arrival: when both carry one, the
/// tail must come strictly before. Without numbers, the tail must fall inside the
/// session gap on either side of the new turn (clock skew between writers is
/// tolerated up to the same gap), and never after it by more than that.
pub fn chain_tail_precedes(
    tail: &ChainTail,
    created_at: DateTime<Utc>,
    sequence_number: Option<u32>,
) -> bool {
    if let (Some(tail_seq), Some(seq)) = (tail.sequence_number, sequence_number) {
        if tail_seq >= seq {
            return false;
        }
    }
    let gap = (created_at - tail.created_at).num_seconds();
    gap.abs() <= SESSION_GAP_SECS
}

/// Pick the better of two chain tails for the same conversation: the higher
/// sequence number when both carry one, else the later `created_at`, and on an
/// exact timestamp tie the later arrival (`candidate`). Turns of one session often
/// share a timestamp (the LoCoMo corpus stamps a whole day's session alike), and
/// the turn written last is the one the next turn answers.
pub fn later_tail(current: Option<ChainTail>, candidate: ChainTail) -> ChainTail {
    match current {
        None => candidate,
        Some(held) => {
            let candidate_wins = match (held.sequence_number, candidate.sequence_number) {
                (Some(a), Some(b)) => b > a,
                _ => candidate.created_at >= held.created_at,
            };
            if candidate_wins {
                candidate
            } else {
                held
            }
        }
    }
}

/// The context given to a turn that arrived without one, so the resolved prior turn
/// can be recorded on it as `preceding_memory_id`.
///
/// Nothing in it is detected. `session_detect` describes the process that runs the
/// detection, which here is the veld server, not the writer: its cwd, git branch and
/// editor window would be stamped onto someone else's turn. The session stays default
/// and both timestamps are the turn's own `created_at`, not the wall clock at ingest.
pub fn threaded_context(created_at: DateTime<Utc>) -> RichContext {
    RichContext {
        id: ContextId(uuid::Uuid::new_v4()),
        conversation: ConversationContext::default(),
        user: UserContext::default(),
        project: ProjectContext::default(),
        temporal: TemporalContext::default(),
        semantic: SemanticContext::default(),
        code: CodeContext::default(),
        document: DocumentContext::default(),
        environment: EnvironmentContext::default(),
        emotional: EmotionalContext::default(),
        source: SourceContext::default(),
        episode: EpisodeContext::default(),
        place: Default::default(),
        who: Default::default(),
        why: Default::default(),
        binding: Default::default(),
        session: Default::default(),
        parent: None,
        embeddings: None,
        decay_rate: 1.0,
        created_at,
        updated_at: created_at,
    }
}

/// `[2022-05-01]` — the ISO date prefix shared by both index texts.
pub fn iso_date_prefix(created_at: DateTime<Utc>) -> String {
    format!("[{}]", created_at.format("%Y-%m-%d"))
}

/// `1 May 2022` — the date as a question would phrase it, for the BM25 document.
pub fn spelled_date(created_at: DateTime<Utc>) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let month = MONTHS[(created_at.month0() as usize).min(11)];
    format!("{} {} {}", created_at.day(), month, created_at.year())
}

/// Derive the index text for one turn.
///
/// `entities` are the turn's own entities (the BM25 entities field it would have
/// had anyway); the prior turn's entities are appended after them, deduplicated
/// case-insensitively, conversation tags excluded, capped at [`PRIOR_ENTITY_MAX`].
pub fn build(
    content: &str,
    created_at: DateTime<Utc>,
    entities: &[String],
    prior: Option<&PriorTurn>,
) -> TurnIndexText {
    let date = iso_date_prefix(created_at);
    let content = content.trim();

    let mut embedding = String::with_capacity(content.len() + PRIOR_TURN_MAX_CHARS + 32);
    embedding.push_str(&date);
    embedding.push(' ');
    embedding.push_str(content);
    if let Some(prior) = prior {
        let prior_text = truncate_chars(prior.content.trim(), PRIOR_TURN_MAX_CHARS);
        if !prior_text.is_empty() {
            embedding.push_str("\n(previous: ");
            embedding.push_str(&prior_text);
            embedding.push(')');
        }
    }

    let mut bm25_entities: Vec<String> = entities.to_vec();
    let mut seen: std::collections::HashSet<String> =
        bm25_entities.iter().map(|e| e.to_lowercase()).collect();
    let mut prior_entities: Vec<String> = Vec::new();
    if let Some(prior) = prior {
        for entity in &prior.entities {
            if is_conversation_tag(entity) || entity.trim().is_empty() {
                continue;
            }
            if seen.insert(entity.to_lowercase()) {
                prior_entities.push(entity.clone());
                if prior_entities.len() >= PRIOR_ENTITY_MAX {
                    break;
                }
            }
        }
    }

    let mut bm25_content = String::with_capacity(content.len() + 64);
    bm25_content.push_str(&date);
    bm25_content.push(' ');
    bm25_content.push_str(&spelled_date(created_at));
    bm25_content.push(' ');
    bm25_content.push_str(content);
    if !prior_entities.is_empty() {
        bm25_content.push_str("\n(previous: ");
        bm25_content.push_str(&prior_entities.join(", "));
        bm25_content.push(')');
    }
    bm25_entities.extend(prior_entities);

    TurnIndexText {
        embedding,
        bm25_content,
        bm25_entities,
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut out: String = text.chars().take(max_chars).collect();
    if out.len() < text.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{EpisodeContext, RichContext};
    use chrono::TimeZone;

    fn at(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 19, 31, 0).unwrap()
    }

    fn turn(content: &str, tags: &[&str]) -> Experience {
        Experience {
            experience_type: ExperienceType::Conversation,
            content: content.to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            entities: tags.iter().map(|t| t.to_string()).collect(),
            ..Default::default()
        }
    }

    fn prior(content: &str, entities: &[&str]) -> PriorTurn {
        PriorTurn {
            id: MemoryId(uuid::Uuid::new_v4()),
            content: content.to_string(),
            entities: entities.iter().map(|e| e.to_string()).collect(),
            created_at: at(2022, 5, 1),
            sequence_number: None,
            source: PriorSource::ConversationKey,
        }
    }

    #[test]
    fn the_embedding_carries_the_date_and_the_prior_turn() {
        let p = prior("Nate: I got two turtles last spring.", &["Nate", "turtles"]);
        let text = build(
            "Nate: I've had them for 3 years.",
            at(2022, 5, 1),
            &["conv-42".into(), "Nate".into()],
            Some(&p),
        );
        assert_eq!(
            text.embedding,
            "[2022-05-01] Nate: I've had them for 3 years.\n(previous: Nate: I got two turtles last spring.)"
        );
    }

    #[test]
    fn the_bm25_document_carries_the_date_and_the_prior_entities_not_its_text() {
        let p = prior(
            "Nate: I got two turtles last spring.",
            &["conv-42", "Nate", "turtles"],
        );
        let text = build(
            "Nate: I've had them for 3 years.",
            at(2022, 5, 1),
            &["conv-42".into(), "Nate".into()],
            Some(&p),
        );
        assert_eq!(
            text.bm25_content,
            "[2022-05-01] 1 May 2022 Nate: I've had them for 3 years.\n(previous: turtles)"
        );
        assert!(!text.bm25_content.contains("last spring"));
        assert_eq!(text.bm25_entities, vec!["conv-42", "Nate", "turtles"]);
    }

    #[test]
    fn a_turn_with_no_prior_still_gets_its_date() {
        let text = build("Joanna: Hey Nate!", at(2022, 1, 21), &[], None);
        assert_eq!(text.embedding, "[2022-01-21] Joanna: Hey Nate!");
        assert_eq!(
            text.bm25_content,
            "[2022-01-21] 21 January 2022 Joanna: Hey Nate!"
        );
        assert!(text.bm25_entities.is_empty());
    }

    #[test]
    fn the_prior_turn_is_capped_so_the_current_turn_keeps_the_attention() {
        let long = "x".repeat(PRIOR_TURN_MAX_CHARS * 2);
        let p = prior(&long, &[]);
        let text = build("short", at(2022, 5, 1), &[], Some(&p));
        let prior_part = text.embedding.split("(previous: ").nth(1).unwrap();
        assert_eq!(prior_part.chars().count(), PRIOR_TURN_MAX_CHARS + 2);
        assert!(prior_part.ends_with("…)"));
    }

    #[test]
    fn prior_entities_are_capped_and_deduplicated() {
        let many: Vec<String> = (0..30).map(|i| format!("e{i}")).collect();
        let mut p = prior("prior", &[]);
        p.entities = many;
        p.entities.insert(0, "Nate".into());
        let text = build("turn", at(2022, 5, 1), &["nate".into()], Some(&p));
        assert_eq!(text.bm25_entities.len(), 1 + PRIOR_ENTITY_MAX);
        assert_eq!(text.bm25_entities[0], "nate");
        assert!(!text.bm25_entities[1..].iter().any(|e| e == "Nate"));
    }

    #[test]
    fn the_conversation_key_prefers_the_episode_id_over_the_tag() {
        let mut t = turn("hi", &["conv-42", "Nate"]);
        assert_eq!(conversation_key(&t).as_deref(), Some("conv-42"));
        let mut ctx = crate::memory::session_detect::rich_context_with_session();
        ctx.episode = EpisodeContext {
            episode_id: Some("episode-7".into()),
            ..Default::default()
        };
        t.context = Some(ctx);
        assert_eq!(conversation_key(&t).as_deref(), Some("episode-7"));
        let bare = turn("hi", &["Nate"]);
        assert_eq!(conversation_key(&bare), None);
    }

    #[test]
    fn only_conversation_turns_apply() {
        assert!(applies(&turn("hi", &[])));
        let learning = Experience {
            experience_type: ExperienceType::Learning,
            content: "hi".into(),
            ..Default::default()
        };
        assert!(!applies(&learning));
    }

    #[test]
    fn the_chain_orders_by_sequence_number_not_arrival() {
        let earlier = ChainTail {
            id: MemoryId(uuid::Uuid::new_v4()),
            created_at: at(2022, 5, 1),
            sequence_number: Some(3),
        };
        let later_arrival = ChainTail {
            id: MemoryId(uuid::Uuid::new_v4()),
            created_at: at(2022, 5, 1),
            sequence_number: Some(2),
        };
        let kept = later_tail(Some(earlier.clone()), later_arrival);
        assert_eq!(kept.id, earlier.id);
        assert!(!chain_tail_precedes(&kept, at(2022, 5, 1), Some(3)));
        assert!(chain_tail_precedes(&kept, at(2022, 5, 1), Some(4)));
        assert!(chain_tail_precedes(&kept, at(2022, 5, 1), None));
    }

    #[test]
    fn an_exact_timestamp_tie_goes_to_the_later_arrival() {
        let first = ChainTail {
            id: MemoryId(uuid::Uuid::new_v4()),
            created_at: at(2022, 5, 1),
            sequence_number: None,
        };
        let second = ChainTail {
            id: MemoryId(uuid::Uuid::new_v4()),
            created_at: at(2022, 5, 1),
            sequence_number: None,
        };
        let kept = later_tail(Some(first), second.clone());
        assert_eq!(kept.id, second.id);
    }

    #[test]
    fn the_chain_breaks_at_the_session_gap() {
        let tail = ChainTail {
            id: MemoryId(uuid::Uuid::new_v4()),
            created_at: at(2022, 5, 1),
            sequence_number: None,
        };
        let same_session = at(2022, 5, 1) + chrono::Duration::seconds(SESSION_GAP_SECS);
        let next_day = at(2022, 5, 2);
        assert!(chain_tail_precedes(&tail, same_session, None));
        assert!(!chain_tail_precedes(&tail, next_day, None));
    }

    #[test]
    fn a_threaded_context_carries_no_detected_session_and_the_turns_own_time() {
        let ctx = threaded_context(at(2022, 5, 1));
        assert_eq!(ctx.session, crate::memory::facets::AgentSession::default());
        assert_eq!(ctx.created_at, at(2022, 5, 1));
        assert_eq!(ctx.updated_at, at(2022, 5, 1));
        assert!(ctx.episode.preceding_memory_id.is_none());
    }

    #[test]
    fn the_preceding_id_must_be_a_uuid() {
        let mut t = turn("hi", &[]);
        let mut ctx: RichContext = crate::memory::session_detect::rich_context_with_session();
        ctx.episode.preceding_memory_id = Some("not-a-uuid".into());
        t.context = Some(ctx.clone());
        assert!(preceding_memory_id(&t).is_none());
        let id = uuid::Uuid::new_v4();
        ctx.episode.preceding_memory_id = Some(id.to_string());
        t.context = Some(ctx);
        assert_eq!(preceding_memory_id(&t), Some(MemoryId(id)));
    }
}
