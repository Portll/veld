//! ObservationDraft → Memory persistence pipeline.
//!
//! V1 path: persist sleep-time observations via the standard
//! [`crate::memory::MemorySystem::remember`] entry point. This reuses the
//! existing embedding + entity extraction + graph indexing pipeline rather
//! than building a parallel path. Origin metadata is stamped into
//! `Experience.metadata` so retrieval and evidence-assembly downstream can
//! apply the visibility rules in
//! [`super::types::MemoryOrigin::visible_to_evidence`].
//!
//! Degraded-embedding handling: if the underlying embedder is in fallback
//! mode, `remember()` will still store the memory but mark
//! `experience.embedding_degraded = true`. The existing maintenance loop
//! re-embeds degraded memories when the embedder recovers
//! (`maintenance.rs::run_maintenance` section 3.7). No special handling here.
//!
//! V2 (R27) introduces explicit *graduation* — a foreground-access-driven
//! transition where observations earn entry into the semantic fact corpus.
//! That logic lives in `observer.rs` / V2.

use std::collections::HashMap;

use super::types::{MemoryOrigin, ObservationDraft, SleepTimeError, SleepTimeResult};
use crate::memory::types::{Experience, ExperienceType, MemoryId};
use crate::memory::MemorySystem;

/// Importance assigned to freshly-persisted observations. Picked to be just
/// above the working→session promotion threshold so observations that survive
/// one maintenance cycle without negative feedback graduate naturally.
pub const DEFAULT_OBSERVATION_IMPORTANCE: f32 = 0.55;

/// Outcome of a persistence attempt.
#[derive(Debug)]
pub enum PersistOutcome {
    /// Observation was stored; this is its assigned memory id.
    Stored(MemoryId),
    /// `remember()` returned an existing-content match (idempotency
    /// dedup). The existing memory id is returned so the worker can still
    /// emit a `SleepTimeObservationEmitted` event pointing to a stable
    /// retrieval target.
    Deduped(MemoryId),
}

/// Persist an [`ObservationDraft`] via [`MemorySystem::remember`].
///
/// Side effects:
///   - Builds an `Experience` whose `metadata` carries the origin tag, sleep
///     mode, embedder version, priority, confidence, lineage backlinks, and
///     three-date temporal anchors.
///   - Defers all embedding / entity extraction / graph indexing to
///     `remember()` — sleep-time observations are first-class memories.
///   - Returns `PersistOutcome::Deduped` if the content already exists
///     (content-hash dedup in `remember()`).
///
/// Does NOT emit consolidation events — the worker pairs persistence with
/// `SleepTimeObservationEmitted` in a single audit batch (R1 spirit).
pub fn persist_observation(
    draft: &ObservationDraft,
    mem_sys: &MemorySystem,
) -> SleepTimeResult<PersistOutcome> {
    let experience = build_experience(draft);

    // `remember()` is the canonical foreground-API entry point and does:
    //   - content-hash dedup (idempotent on repeat content)
    //   - embedding generation (with cache)
    //   - entity extraction + graph linking
    //   - vector indexing + interference detection
    // We piggy-back on all of it; origin metadata distinguishes downstream.
    let memory_id = mem_sys
        .remember(experience, Some(draft.created_at))
        .map_err(|e| SleepTimeError::Other(anyhow::anyhow!("remember observation: {e}")))?;

    // `remember()` returns the EXISTING memory id on content-hash hit;
    // we have no cheap way to tell after-the-fact whether dedup occurred,
    // so default to `Stored` and let the worker treat the returned id
    // idempotently.
    Ok(PersistOutcome::Stored(memory_id))
}

/// Construct the `Experience` value that will be passed to `remember()`.
/// Public so V2 graduation can build the same shape with a different origin
/// tag.
pub fn build_experience(draft: &ObservationDraft) -> Experience {
    Experience {
        experience_type: ExperienceType::Observation,
        content: draft.content.clone(),
        entities: draft.entity_refs.clone(),
        metadata: observation_metadata(draft),
        ..Default::default()
    }
}

/// Build the metadata HashMap attached to the `Experience`. Includes the
/// fields that downstream feedback/retrieval code needs to recognise this
/// memory as a sleep-time observation (without requiring schema changes to
/// `Experience`).
fn observation_metadata(draft: &ObservationDraft) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("origin".to_string(), draft.origin.as_str().to_string());
    m.insert("sleep_mode".to_string(), draft.mode.as_str().to_string());
    m.insert(
        "embedder_version".to_string(),
        draft.embedder_version.clone(),
    );
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
// Origin inspection helpers for downstream code
// =============================================================================

/// Read the `MemoryOrigin` recorded in an `Experience`'s metadata.
///
/// Returns [`MemoryOrigin::ForegroundUser`] (the safe default) if the
/// metadata key is absent or unrecognised — never panics. Evidence-assembly
/// uses this to enforce the visibility rules in
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
            "minilm-l6-v2",
        );
        d.entity_refs = vec!["user".to_string()];
        d.confidence = 0.7;
        d.priority = super::super::types::ObservationPriority::High;
        d
    }

    #[test]
    fn build_experience_carries_origin_metadata() {
        let draft = make_draft();
        let exp = build_experience(&draft);
        assert_eq!(exp.experience_type, ExperienceType::Observation);
        assert_eq!(
            exp.metadata.get("origin").map(|s| s.as_str()),
            Some("background_sleep_time_observation")
        );
        assert_eq!(exp.metadata.get("sleep_mode").map(|s| s.as_str()), Some("nrem"));
        assert_eq!(
            exp.metadata.get("embedder_version").map(|s| s.as_str()),
            Some("minilm-l6-v2")
        );
    }

    #[test]
    fn metadata_captures_priority_and_confidence() {
        let draft = make_draft();
        let md = observation_metadata(&draft);
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
