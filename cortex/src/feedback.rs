//! Feedback Module - Detect outcomes and reinforce memories
//!
//! Like Hebbian learning ("neurons that fire together wire together"),
//! this module detects whether an interaction was helpful or not,
//! and sends reinforcement signals to strengthen or weaken memories.

use crate::perception::{detect_followup_signal, FollowupSignal};
use crate::types::{ReinforceRequest, ReinforceResponse, Session};
use tracing::{debug, info, warn};

/// Outcome of an interaction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    /// The response was helpful
    Helpful,
    /// The response was misleading or wrong
    Misleading,
    /// Outcome is unclear
    Neutral,
}

impl Outcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Helpful => "helpful",
            Outcome::Misleading => "misleading",
            Outcome::Neutral => "neutral",
        }
    }
}

/// Detect outcome from user's followup message
///
/// Analyzes the user's message to infer whether the previous
/// response was helpful, misleading, or neutral.
pub fn detect_outcome(user_message: &str, session: &Session) -> Outcome {
    // Check for explicit followup signals
    if let Some(signal) = detect_followup_signal(user_message) {
        return match signal {
            FollowupSignal::Positive => Outcome::Helpful,
            FollowupSignal::Negative => Outcome::Misleading,
            FollowupSignal::Correction => Outcome::Misleading,
            FollowupSignal::Continuation => Outcome::Helpful,
        };
    }

    // Check for abandonment signals
    if is_topic_change(user_message, session) {
        // Topic change after response could indicate unhelpfulness
        return Outcome::Neutral;
    }

    // Check for repetition (user asking same thing again = unhelpful)
    if is_repetition(user_message, session) {
        return Outcome::Misleading;
    }

    // Default to neutral if no clear signal
    Outcome::Neutral
}

/// Check if the user changed topic (potential abandonment)
fn is_topic_change(user_message: &str, session: &Session) -> bool {
    let Some(ref last_message) = session.last_user_message else {
        return false;
    };

    // Simple heuristic: low word overlap indicates topic change
    let last_lower = last_message.to_lowercase();
    let last_words: std::collections::HashSet<_> = last_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    let current_lower = user_message.to_lowercase();
    let current_words: std::collections::HashSet<_> = current_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    if last_words.is_empty() || current_words.is_empty() {
        return false;
    }

    let overlap = last_words.intersection(&current_words).count();
    let max_size = last_words.len().max(current_words.len());

    // Less than 10% word overlap suggests topic change
    (overlap as f32 / max_size as f32) < 0.1
}

/// Check if the user is repeating their request
fn is_repetition(user_message: &str, session: &Session) -> bool {
    let Some(ref last_message) = session.last_user_message else {
        return false;
    };

    // Normalize messages
    let normalize = |s: &str| -> String {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect()
    };

    let last_normalized = normalize(last_message);
    let current_normalized = normalize(user_message);

    // High similarity suggests repetition
    if last_normalized == current_normalized {
        return true;
    }

    // Check for "again" or "try again" patterns
    let lower = user_message.to_lowercase();
    if lower.contains("again") || lower.contains("retry") || lower.contains("one more time") {
        return true;
    }

    false
}

/// Send reinforcement signal to the brain
///
/// Tells the brain whether the surfaced memories were helpful,
/// enabling Hebbian learning to strengthen or weaken associations.
pub async fn reinforce_memories(
    http_client: &reqwest::Client,
    brain_url: &str,
    brain_api_key: &str,
    user_id: &str,
    memory_ids: &[String],
    outcome: Outcome,
) -> bool {
    if memory_ids.is_empty() {
        return true; // Nothing to reinforce
    }

    // Don't send neutral signals (no learning signal)
    if outcome == Outcome::Neutral {
        debug!("Skipping reinforcement for neutral outcome");
        return true;
    }

    let request = ReinforceRequest {
        user_id: user_id.to_string(),
        ids: memory_ids.to_vec(),
        outcome: outcome.as_str().to_string(),
    };

    info!(
        user_id = %user_id,
        outcome = %outcome.as_str(),
        memory_count = memory_ids.len(),
        "Sending reinforcement to brain"
    );

    match http_client
        .post(format!("{}/api/reinforce", brain_url))
        .header("Content-Type", "application/json")
        .header("X-API-Key", brain_api_key)
        .json(&request)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<ReinforceResponse>().await {
                Ok(data) => {
                    info!(
                        "Reinforced {} memories with outcome: {}",
                        data.memories_processed,
                        outcome.as_str()
                    );
                    true
                }
                Err(e) => {
                    warn!("Failed to parse reinforcement response: {}", e);
                    false
                }
            }
        }
        Ok(resp) => {
            warn!("Brain returned error for reinforcement: {}", resp.status());
            false
        }
        Err(e) => {
            warn!("Failed to send reinforcement: {}", e);
            false
        }
    }
}

/// Process feedback for the previous interaction
///
/// Called at the start of a new request to evaluate how helpful
/// the previous response was based on the user's followup.
pub async fn process_feedback(
    http_client: &reqwest::Client,
    brain_url: &str,
    brain_api_key: &str,
    user_id: &str,
    user_message: &str,
    session: &Session,
) -> Option<Outcome> {
    // Skip if no previous memories to reinforce
    if session.last_memory_ids.is_empty() {
        return None;
    }

    // Detect outcome from user's message
    let outcome = detect_outcome(user_message, session);

    // Send reinforcement signal
    reinforce_memories(
        http_client,
        brain_url,
        brain_api_key,
        user_id,
        &session.last_memory_ids,
        outcome,
    )
    .await;

    Some(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(last_message: Option<&str>) -> Session {
        Session {
            last_response: Some("Previous response".to_string()),
            last_memory_ids: vec!["mem1".to_string()],
            last_tool_uses: vec![],
            interaction_count: 1,
            last_user_message: last_message.map(|s| s.to_string()),
        }
    }

    #[test]
    fn test_detect_outcome_positive() {
        let session = make_session(Some("Help me with X"));
        assert_eq!(detect_outcome("Thanks, that works!", &session), Outcome::Helpful);
    }

    #[test]
    fn test_detect_outcome_negative() {
        let session = make_session(Some("Help me with X"));
        assert_eq!(detect_outcome("No, that's wrong", &session), Outcome::Misleading);
    }

    #[test]
    fn test_detect_outcome_repetition() {
        let session = make_session(Some("List the files"));
        assert_eq!(detect_outcome("List the files again", &session), Outcome::Misleading);
    }

    #[test]
    fn test_detect_outcome_continuation() {
        let session = make_session(Some("Add a function"));
        assert_eq!(detect_outcome("Now add the tests", &session), Outcome::Helpful);
    }

    #[test]
    fn test_is_topic_change() {
        let session = make_session(Some("Help me fix this Rust error"));
        assert!(is_topic_change("What's the weather like?", &session));
    }

    #[test]
    fn test_not_topic_change() {
        let session = make_session(Some("Help me fix this Rust error"));
        assert!(!is_topic_change("I still get the Rust error", &session));
    }
}
