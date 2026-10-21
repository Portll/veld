//! The turn successor index and recall's neighbour expansion.
//!
//! An anaphoric reply ("I've had them for 3 years now!") shares no word with the
//! question about it, while the turn it answers does. `remember` records which turn
//! followed which; recall brings the turn after a well-ranked turn into the candidate
//! pool. These tests write conversations through the same path the recall harness uses.

use std::collections::HashSet;

use chrono::{TimeZone, Utc};
use tempfile::TempDir;
use veld::memory::types::{Experience, ExperienceType, ForgetCriteria, MemoryId, Query};
use veld::memory::{CognitiveDials, MemoryConfig, MemorySystem};

fn test_config(dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        storage_path: dir.path().to_path_buf(),
        collective_store_dir: None,
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.3,
    }
}

fn turn(content: &str, conversation: &str, speaker: &str) -> Experience {
    Experience {
        experience_type: ExperienceType::Conversation,
        content: content.to_string(),
        tags: vec![conversation.to_string(), speaker.to_string()],
        entities: vec![conversation.to_string(), speaker.to_string()],
        ..Default::default()
    }
}

/// The index is persisted: a turn ranked after a restart still finds its reply.
#[test]
fn a_turns_successor_is_found_after_a_restart() {
    let dir = TempDir::new().expect("temp dir");
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();

    let (question, reply) = {
        let system = MemorySystem::new(test_config(&dir), None).expect("memory system");
        let question = system
            .remember(
                turn(
                    "Joanna: Did the new controller arrive?",
                    "conv-42",
                    "Joanna",
                ),
                Some(at),
            )
            .expect("question stored");
        let reply = system
            .remember(
                turn("Nate: Yes, this morning!", "conv-42", "Nate"),
                Some(at + chrono::Duration::minutes(1)),
            )
            .expect("reply stored");
        assert_eq!(system.turn_successors(&question), vec![reply.clone()]);
        (question, reply)
    };

    let system = MemorySystem::new(test_config(&dir), None).expect("memory system reopened");
    assert_eq!(system.turn_successors(&question), vec![reply.clone()]);
    assert!(
        system.turn_successors(&reply).is_empty(),
        "the last turn has no successor"
    );
}

/// A writer-threaded `preceding_memory_id` records the successor as well.
#[test]
fn a_writer_threaded_turn_is_recorded_as_the_successor() {
    let dir = TempDir::new().expect("temp dir");
    let system = MemorySystem::new(test_config(&dir), None).expect("memory system");
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();

    let question = system
        .remember(
            turn("Joanna: Which map did you play?", "conv-42", "Joanna"),
            Some(at),
        )
        .expect("question stored");
    system
        .remember(
            turn("Joanna: Was it the desert one?", "conv-42", "Joanna"),
            Some(at + chrono::Duration::minutes(1)),
        )
        .expect("follow-up stored");

    let mut answer = turn("Nate: The harbour map, we won 16-9.", "conv-42", "Nate");
    let mut ctx = veld::memory::index_text::threaded_context(at);
    ctx.episode.preceding_memory_id = Some(question.0.to_string());
    answer.context = Some(ctx);
    let answer_id = system
        .remember(answer, Some(at + chrono::Duration::minutes(2)))
        .expect("answer stored");

    assert!(system.turn_successors(&question).contains(&answer_id));
}

/// A successor forgotten after it was recorded is skipped on read.
#[test]
fn a_forgotten_successor_is_skipped() {
    let dir = TempDir::new().expect("temp dir");
    let system = MemorySystem::new(test_config(&dir), None).expect("memory system");
    let at = Utc.with_ymd_and_hms(2022, 5, 1, 19, 31, 0).unwrap();

    let question = system
        .remember(
            turn("Joanna: Are you coming on Saturday?", "conv-42", "Joanna"),
            Some(at),
        )
        .expect("question stored");
    let reply = system
        .remember(
            turn("Nate: Wouldn't miss it.", "conv-42", "Nate"),
            Some(at + chrono::Duration::minutes(1)),
        )
        .expect("reply stored");
    assert_eq!(system.turn_successors(&question), vec![reply.clone()]);

    system
        .forget(ForgetCriteria::ById(reply))
        .expect("reply forgotten");
    assert!(system.turn_successors(&question).is_empty());
}

/// A turn labelled by its tags alone. The entities are left empty on purpose: for a
/// duration question ("how long ...") Layer 4.54 demotes the older of any two memories
/// sharing more than 40% of their entities, and Layer 4.525 adds 0.10 for each query
/// word an entity matches. Both outweigh the RRF differences between these turns, so
/// with entities the test would measure entity overlap and age, not the expansion leg.
fn tagged_turn(content: &str, conversation: &str, speaker: &str) -> Experience {
    Experience {
        experience_type: ExperienceType::Conversation,
        content: content.to_string(),
        tags: vec![conversation.to_string(), speaker.to_string()],
        ..Default::default()
    }
}

const QUERY: &str = "How long has Nate had his turtles?";

/// What Nate "had" in the 24 questions that match the query better than the reply.
const OTHER_THINGS: [&str; 24] = [
    "bike",
    "guitar",
    "laptop",
    "car",
    "apartment",
    "gaming chair",
    "keyboard",
    "headset",
    "couch",
    "camera",
    "skateboard",
    "backpack",
    "watch",
    "jacket",
    "phone",
    "desk",
    "monitor",
    "drone",
    "tent",
    "kayak",
    "piano",
    "telescope",
    "scooter",
    "printer",
];

/// Writes 24 questions that share every query word but "turtles", then the turtle
/// question and the reply to it, and returns `(question, reply)`.
///
/// The reply shares only "Nate" and "had" with the query, so BM25 and the embedding
/// rank it below all 25 questions: past the 20 candidates the scoring layer fetches,
/// which is the LoCoMo failure the miss taxonomy calls anaphora. The turtle question is
/// the best match, the turn neighbour expansion starts from.
fn write_turtle_conversation(system: &MemorySystem) -> (MemoryId, MemoryId) {
    let at = Utc.with_ymd_and_hms(2023, 5, 20, 14, 0, 0).unwrap();
    for (i, thing) in OTHER_THINGS.iter().enumerate() {
        system
            .remember(
                tagged_turn(
                    &format!("Joanna: How long has Nate had his {thing}?"),
                    "conv-7",
                    "Joanna",
                ),
                Some(at + chrono::Duration::minutes(i as i64)),
            )
            .expect("other question stored");
    }
    let asked = at + chrono::Duration::hours(1);
    let question = system
        .remember(
            tagged_turn(
                "Joanna: How long have you had your turtles, Nate?",
                "conv-42",
                "Joanna",
            ),
            Some(asked),
        )
        .expect("turtle question stored");
    let reply = system
        .remember(
            tagged_turn("Nate: I've had them for 3 years now!", "conv-42", "Nate"),
            Some(asked + chrono::Duration::minutes(1)),
        )
        .expect("reply stored");
    (question, reply)
}

/// The cross-encoder is off in both arms. With it on, the result would turn on how one
/// model scores one reply against 24 questions, which the recall gate measures; this
/// test pins the leg that puts the reply in front of it.
fn system_with(neighbour_expansion: bool) -> (MemorySystem, TempDir) {
    let dir = TempDir::new().expect("temp dir");
    let system = MemorySystem::new(test_config(&dir), None).expect("memory system");
    system.set_cognitive_dials(CognitiveDials {
        neighbour_expansion: Some(neighbour_expansion),
        use_reranking: Some(false),
        ..Default::default()
    });
    (system, dir)
}

/// One recall of [`QUERY`] with the gold funnel armed on `gold`.
struct Recalled {
    ids: Vec<MemoryId>,
    contents: Vec<String>,
    /// Best rank of `gold` in the neighbour leg's own list, `None` when it is absent.
    neighbour_stage: Option<usize>,
}

fn recall_turtles(system: &MemorySystem, gold: &MemoryId) -> Recalled {
    let query = Query {
        query_text: Some(QUERY.to_string()),
        max_results: 10,
        ..Default::default()
    };
    veld::memory::gold_funnel::begin(HashSet::from([gold.clone()]));
    let hits = system.recall(&query).expect("recall");
    let funnel = veld::memory::gold_funnel::take().expect("funnel armed");
    let neighbour_stage = funnel
        .iter()
        .find(|(stage, _)| stage == "neighbour")
        .unwrap_or_else(|| panic!("no neighbour stage in {funnel:?}"))
        .1;
    Recalled {
        ids: hits.iter().map(|m| m.id.clone()).collect(),
        contents: hits.iter().map(|m| m.experience.content.clone()).collect(),
        neighbour_stage,
    }
}

#[test]
fn a_reply_that_shares_no_word_with_the_question_is_recalled_through_its_prior_turn() {
    let (system, _dir) = system_with(true);
    let (question, reply) = write_turtle_conversation(&system);
    assert_eq!(system.turn_successors(&question), vec![reply.clone()]);

    let recalled = recall_turtles(&system, &reply);
    assert!(
        recalled.ids.contains(&reply),
        "the reply must be in the top 10 for {QUERY:?}; got {:#?}",
        recalled.contents
    );
    assert!(
        recalled.neighbour_stage.is_some(),
        "the reply must come from the neighbour leg"
    );
}

#[test]
fn without_neighbour_expansion_the_reply_is_not_recalled() {
    let (system, _dir) = system_with(false);
    let (_, reply) = write_turtle_conversation(&system);

    let recalled = recall_turtles(&system, &reply);
    assert!(
        !recalled.ids.contains(&reply),
        "with the dial off nothing brings the reply in; got {:#?}",
        recalled.contents
    );
    assert_eq!(recalled.neighbour_stage, None);
}
