//! Ingest-time contextualisation of conversation turns (`memory::index_text`).
//!
//! A reply like "I've had them for 3 years" is only findable by a question about
//! turtles when the turn it answers reaches the indexes. These tests write a
//! short conversation through the same path the recall harness uses and ask the
//! question the LoCoMo gate asks.

use chrono::{TimeZone, Utc};
use tempfile::TempDir;
use veld::memory::index_text;
use veld::memory::types::{Experience, ExperienceType, Query};
use veld::memory::{CognitiveDials, MemoryConfig, MemorySystem};

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        collective_store_dir: None,
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.3,
    };
    let system = MemorySystem::new(config, None).expect("memory system");
    (system, temp_dir)
}

fn turn(content: &str, speaker: &str) -> Experience {
    Experience {
        experience_type: ExperienceType::Conversation,
        content: content.to_string(),
        tags: vec!["conv-42".to_string(), speaker.to_string()],
        entities: vec!["conv-42".to_string(), speaker.to_string()],
        ..Default::default()
    }
}

/// The LoCoMo failure the taxonomy calls `anaphora`: the gold turn names neither
/// the pet nor the question's subject.
#[test]
fn an_anaphoric_reply_is_retrieved_by_a_question_about_the_prior_turn() {
    let (system, _dir) = create_test_system();
    let at = Utc.with_ymd_and_hms(2023, 5, 20, 14, 3, 0).unwrap();

    let distractors = [
        (
            "Joanna: I finally finished the mural at the community centre.",
            "Joanna",
        ),
        (
            "Nate: My team lost the quarter-final, but the new map was fun.",
            "Nate",
        ),
        (
            "Joanna: The bakery on 5th started doing sourdough on Sundays.",
            "Joanna",
        ),
        (
            "Nate: I'm thinking of repainting the kitchen a deep green.",
            "Nate",
        ),
    ];
    for (content, speaker) in distractors {
        system
            .remember(turn(content, speaker), Some(at))
            .expect("distractor stored");
    }
    system
        .remember(
            turn("Joanna: How are your turtles doing, Nate?", "Joanna"),
            Some(at),
        )
        .expect("question stored");
    let reply_id = system
        .remember(
            turn("Nate: They're great! I've had them for 3 years.", "Nate"),
            Some(at),
        )
        .expect("reply stored");

    let query = Query {
        query_text: Some("How long has Nate had his turtles?".to_string()),
        max_results: 3,
        ..Default::default()
    };
    let hits = system.recall(&query).expect("recall");
    assert!(
        hits.iter().any(|m| m.id == reply_id),
        "the reply must be in the top {} for the turtles question; got: {:?}",
        query.max_results,
        hits.iter()
            .map(|m| m.experience.content.clone())
            .collect::<Vec<_>>()
    );
}

/// Without `preceding_memory_id`, the prior turn is found by the conversation
/// tag and the episode is threaded with its id; stored content is untouched.
#[test]
fn the_prior_turn_threads_the_episode_and_leaves_content_alone() {
    let (system, _dir) = create_test_system();
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();

    let first = system
        .remember(
            turn("Joanna: What movie did you watch last night?", "Joanna"),
            Some(at),
        )
        .expect("first stored");
    let second = system
        .remember(
            turn("Nate: Everything Everywhere All at Once. Loved it.", "Nate"),
            Some(at + chrono::Duration::minutes(2)),
        )
        .expect("second stored");

    let stored = system.get_memory(&second).expect("second readable");
    assert_eq!(
        stored.experience.content,
        "Nate: Everything Everywhere All at Once. Loved it."
    );
    let preceding = stored
        .experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.preceding_memory_id.clone());
    assert_eq!(preceding.as_deref(), Some(first.0.to_string().as_str()));
}

/// The context created to hold the threaded id describes the turn, not the server:
/// session detection would stamp the veld process's own cwd, branch and editor onto
/// it, and the wall clock would replace the turn's own time.
#[test]
fn a_threaded_turns_stored_context_carries_a_default_session() {
    let (system, _dir) = create_test_system();
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();
    let reply_at = at + chrono::Duration::minutes(2);

    system
        .remember(
            turn("Joanna: Did you get the new controller?", "Joanna"),
            Some(at),
        )
        .expect("first stored");
    let second = system
        .remember(
            turn("Nate: Yes, it arrived this morning.", "Nate"),
            Some(reply_at),
        )
        .expect("second stored");

    let stored = system.get_memory(&second).expect("second readable");
    let ctx = stored
        .experience
        .context
        .as_ref()
        .expect("a threaded turn carries a context");
    assert!(ctx.episode.preceding_memory_id.is_some());
    assert_eq!(ctx.session, veld::memory::facets::AgentSession::default());
    assert_eq!(ctx.created_at, reply_at);
    assert_eq!(ctx.updated_at, reply_at);
}

/// A turn more than the session gap after the last one starts a new chain.
#[test]
fn a_turn_after_the_session_gap_has_no_prior() {
    let (system, _dir) = create_test_system();
    let day_one = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();
    let day_two = Utc.with_ymd_and_hms(2022, 5, 2, 9, 0, 0).unwrap();

    system
        .remember(turn("Joanna: See you tomorrow!", "Joanna"), Some(day_one))
        .expect("day one stored");
    let next = system
        .remember(turn("Nate: Morning! Slept well?", "Nate"), Some(day_two))
        .expect("day two stored");

    let stored = system.get_memory(&next).expect("readable");
    let preceding = stored
        .experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.preceding_memory_id.clone());
    assert_eq!(preceding, None);
}

/// With the dial off the record is written exactly as before: no threading.
#[test]
fn the_dial_turns_contextualisation_off() {
    let (system, _dir) = create_test_system();
    system.set_cognitive_dials(CognitiveDials {
        turn_context: Some(false),
        ..Default::default()
    });
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();

    system
        .remember(
            turn("Joanna: Any plans for the weekend?", "Joanna"),
            Some(at),
        )
        .expect("first stored");
    let second = system
        .remember(
            turn("Nate: Hiking, if the rain holds off.", "Nate"),
            Some(at),
        )
        .expect("second stored");

    let stored = system.get_memory(&second).expect("readable");
    assert!(stored
        .experience
        .context
        .as_ref()
        .and_then(|ctx| ctx.episode.preceding_memory_id.clone())
        .is_none());
}

/// The derivation itself, pinned so a re-index produces the same text.
#[test]
fn the_index_text_is_derived_from_content_and_the_prior_turn() {
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();
    let prior = index_text::PriorTurn {
        id: veld::memory::types::MemoryId(uuid::Uuid::new_v4()),
        content: "Joanna: How are your turtles doing, Nate?".to_string(),
        entities: vec!["conv-42".into(), "Joanna".into(), "turtles".into()],
        created_at: at,
        sequence_number: None,
        source: index_text::PriorSource::ConversationKey,
    };
    let text = index_text::build(
        "Nate: They're great! I've had them for 3 years.",
        at,
        &["conv-42".into(), "Nate".into()],
        Some(&prior),
    );
    assert_eq!(
        text.embedding,
        "[2022-05-01] Nate: They're great! I've had them for 3 years.\n(previous: Joanna: How are your turtles doing, Nate?)"
    );
    assert_eq!(
        text.bm25_content,
        "[2022-05-01] 1 May 2022 Nate: They're great! I've had them for 3 years.\n(previous: Joanna, turtles)"
    );
    assert_eq!(
        text.bm25_entities,
        vec!["conv-42", "Nate", "Joanna", "turtles"]
    );
}
