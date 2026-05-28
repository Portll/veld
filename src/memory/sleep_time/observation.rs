//! ObservationDraft → Memory persistence pipeline.
//!
//! V1 path: persist sleep-time observations into the user's working memory
//! tier with `ExperienceType::Observation` and origin metadata. The existing
//! synchronous maintenance loop (`maintenance.rs`) will tier-promote them
//! naturally based on importance + age.
//!
//! V2 (R27) introduces explicit *graduation* — a foreground-access-driven
//! transition where observations earn entry into the semantic fact corpus.
//! That is **not** implemented here; graduation logic lives with the
//! observer/orchestrator in V2.
//!
//! Embedder discipline (R38 / B6): we check `Embedder::is_healthy()` before
//! persisting. If the embedder is degraded, the draft is returned to the
//! caller as a *pending* observation; the worker can retry on the next cycle.
//! We never persist an observation with a hash-fallback embedding.

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::Arc;
use uuid::Uuid;

use super::types::{MemoryOrigin, ObservationDraft, SleepTimeError, SleepTimeResult};
use crate::embeddings::Embedder;
use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId, WorkingMemory};

/// Importance assigned to freshly-persisted observations. Picked to be just
/// above the working→session promotion threshold so observations that survive
/// one maintenance cycle without negative feedback graduate naturally.
const DEFAULT_OBSERVATION_IMPORTANCE: f32 = 0.55;

/// Outcome of a persistence attempt.
pub enum PersistOutcome {
    /// Observation was stored; this is its assigned memory id.
    Stored(MemoryId),
    /// Embedder unhealthy; draft was not stored. Caller should hold the draft
    /// and retry on the next sleep-time cycle (R38).
    DeferredEmbedderUnhealthy,
}

/// Persist an [`ObservationDraft`] into the user's working memory tier.
///
/// Side effects:
///   1. (R38) Aborts persistence if `embedder.is_healthy() == false`.
///   2. Generates the content embedding via the supplied [`Embedder`].
///   3. Stamps the embedder version on the draft for later quality-gate use.
///   4. Constructs a `Memory` (origin metadata in `experience.metadata`).
///   5. Adds to working memory under `user_id` actor (multi-tenant routing).
///
/// Note: this function does not write any consolidation events — the worker
/// is responsible for emitting `SleepTimeObservationEmitted` so the event
/// and the persistence can share a single audit batch (R1 spirit).
pub fn persist_observation(
    draft: &mut ObservationDraft,
    embedder: &dyn Embedder,
    working_memory: &Arc<RwLock<WorkingMemory>>,
    user_id: &str,
) -> SleepTimeResult<PersistOutcome> {
    // R15: stamp the embedder identity we're about to embed against.
    draft.embedder_version = embedder.model_id().to_string();

    // R38 / B6: refuse to persist with a degraded fallback embedding.
    // `encode_with_status` returns (vec, is_degraded); the circuit-breaker
    // wrapper sets is_degraded=true when the underlying model is failing.
    let (embedding, degraded) = embedder
        .encode_with_status(&draft.content)
        .map_err(|e| SleepTimeError::Other(anyhow::anyhow!("embed observation: {e}")))?;
    if degraded {
        tracing::debug!(
            user_id = %user_id,
            "embedder degraded — deferring sleep-time observation persistence"
        );
        return Ok(PersistOutcome::DeferredEmbedderUnhealthy);
    }

    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: draft.content.clone(),
        entities: draft.entity_refs.clone(),
        metadata: observation_metadata(draft),
        embeddings: Some(embedding),
        embedding_degraded: false,
        ..Default::default()
    };

    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        experience,
        DEFAULT_OBSERVATION_IMPORTANCE,
        None,                       // agent_id — sleep-time has no foreground agent
        None,                       // run_id
        Some(user_id.to_string()),  // actor_id carries the user
        Some(draft.created_at),
    );
    let memory_id = memory.id.clone();

    {
        let mut wm = working_memory.write();
        wm.add(memory)
            .map_err(|e| SleepTimeError::Other(anyhow::anyhow!("working_memory.add: {e}")))?;
    }

    Ok(PersistOutcome::Stored(memory_id))
}

/// Build the metadata HashMap attached to the `Experience`. Includes the
/// fields that downstream feedback/retrieval code needs to recognize this
/// memory as a sleep-time observation (without requiring schema changes to
/// `Experience`).
fn observation_metadata(draft: &ObservationDraft) -> std::collections::HashMap<String, String> {
    let mut m = std::collections::HashMap::new();
    m.insert("origin".to_string(), draft.origin.as_str().to_string());
    m.insert("sleep_mode".to_string(), draft.mode.as_str().to_string());
    m.insert("embedder_version".to_string(), draft.embedder_version.clone());
    m.insert(
        "priority".to_string(),
        match draft.priority {
            super::types::ObservationPriority::High => "high".into(),
            super::types::ObservationPriority::Medium => "medium".into(),
            super::types::ObservationPriority::Low => "low".into(),
        },
    );
    m.insert("confidence".to_string(), format!("{:.3}", draft.confidence));
    if let Some(rid) = draft.source_rewrite_id {
        m.insert("source_rewrite_id".to_string(), rid.to_string());
    }
    if let Some(s) = &draft.relative_at_anchor {
        m.insert("relative_at_anchor".to_string(), s.clone());
    }
    if let Some(t) = &draft.referenced_at {
        m.insert("referenced_at".to_string(), t.to_rfc3339());
    }
    if let Some(sup) = &draft.supersedes {
        m.insert("supersedes".to_string(), sup.0.to_string());
    }
    if !draft.source_memory_ids.is_empty() {
        m.insert(
            "source_memory_ids".to_string(),
            draft
                .source_memory_ids
                .iter()
                .map(|id| id.0.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    m
}

// =============================================================================
// Public re-export for the orchestrator to thread origin metadata into the
// foreground retrieval/feedback path. Provides a stable accessor that does
// not require downstream code to know the metadata key names.
// =============================================================================

/// Read the `MemoryOrigin` recorded in an Experience's metadata. Returns
/// [`MemoryOrigin::ForegroundUser`] (the safe default) if the metadata key
/// is absent or unrecognised — never panics. Downstream evidence-assembly
/// code uses this to enforce the visibility rules in
/// [`MemoryOrigin::visible_to_evidence`].
pub fn origin_of(experience: &Experience) -> MemoryOrigin {
    match experience.metadata.get("origin").map(|s| s.as_str()) {
        Some("background_sleep_time_observation") => {
            MemoryOrigin::BackgroundSleepTimeObservation
        }
        Some("background_sleep_time_graduated") => MemoryOrigin::BackgroundSleepTimeGraduated,
        Some("foreground_agent") => MemoryOrigin::ForegroundAgent,
        Some("background_maintenance") => MemoryOrigin::BackgroundMaintenance,
        Some("operator_action") => MemoryOrigin::OperatorAction,
        _ => MemoryOrigin::ForegroundUser,
    }
}

/// Convenience: was this memory authored by sleep-time (either tier)?
pub fn is_sleep_time_authored(experience: &Experience) -> bool {
    matches!(
        origin_of(experience),
        MemoryOrigin::BackgroundSleepTimeObservation
            | MemoryOrigin::BackgroundSleepTimeGraduated
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_draft() -> ObservationDraft {
        let mut d = ObservationDraft::new(
            "user prefers concise replies",
            crate::memory::sleep_time::types::SleepMode::Nrem,
            "v1",
        );
        d.entity_refs = vec!["user".to_string()];
        d.confidence = 0.7;
        d.priority = super::super::types::ObservationPriority::High;
        d
    }

    #[test]
    fn metadata_captures_origin_and_mode() {
        let draft = make_draft();
        let md = observation_metadata(&draft);
        assert_eq!(
            md.get("origin").map(|s| s.as_str()),
            Some("background_sleep_time_observation")
        );
        assert_eq!(md.get("sleep_mode").map(|s| s.as_str()), Some("nrem"));
        assert_eq!(md.get("priority").map(|s| s.as_str()), Some("high"));
        assert!(md.get("confidence").is_some());
    }

    #[test]
    fn origin_of_returns_observation_when_metadata_set() {
        let mut exp = Experience::default();
        exp.metadata.insert(
            "origin".to_string(),
            "background_sleep_time_observation".to_string(),
        );
        assert_eq!(
            origin_of(&exp),
            MemoryOrigin::BackgroundSleepTimeObservation
        );
        assert!(is_sleep_time_authored(&exp));
    }

    #[test]
    fn origin_of_returns_foreground_user_when_metadata_absent() {
        let exp = Experience::default();
        assert_eq!(origin_of(&exp), MemoryOrigin::ForegroundUser);
        assert!(!is_sleep_time_authored(&exp));
    }
}
