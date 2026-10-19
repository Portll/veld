//! The turn successor index and recall's neighbour expansion.
//!
//! An anaphoric reply ("I've had them for 3 years now!") shares no word with the
//! question about it, while the turn it answers does. `remember` records which turn
//! followed which; recall brings the turn after a well-ranked turn into the candidate
//! pool. These tests write conversations through the same path the recall harness uses.

use chrono::{TimeZone, Utc};
use tempfile::TempDir;
use veld::memory::types::{Experience, ExperienceType, ForgetCriteria};
use veld::memory::{MemoryConfig, MemorySystem};

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
