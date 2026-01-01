//! Perception Module - Extract full context from Claude API requests
//!
//! Like the sensory cortex, this module perceives everything that Claude sees:
//! - System prompt (instructions, context)
//! - All messages (conversation history)
//! - Tool uses (what Claude did)
//! - Tool results (what happened)
//!
//! The brain needs to see everything to learn properly.

use crate::types::{ClaudeRequest, FullContext, MessageSummary};
use axum::http::HeaderMap;

/// Extract user ID from request headers
pub fn extract_user_id(headers: &HeaderMap) -> String {
    headers
        .get("x-shodh-user-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "claude-code".to_string())
}

/// Extract agent ID from request headers (for sub-agent tracking)
pub fn extract_agent_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-shodh-agent-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Extract parent agent ID from request headers (for hierarchical tracking)
pub fn extract_parent_agent_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-shodh-parent-agent-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Extract run ID from request headers (groups requests within one execution)
pub fn extract_run_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-shodh-run-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Extract full context from a Claude API request
///
/// This is the "eyes and ears" of the brain - it sees everything Claude sees.
pub fn extract_full_context(request: &ClaudeRequest, headers: &HeaderMap) -> FullContext {
    let user_id = extract_user_id(headers);
    let agent_id = extract_agent_id(headers);
    let parent_agent_id = extract_parent_agent_id(headers);
    let run_id = extract_run_id(headers);

    // Extract system prompt
    let system_prompt = request.system.as_ref().map(|s| s.as_text());

    // Extract all messages with summaries
    let mut messages = Vec::new();
    let mut tool_uses = Vec::new();
    let mut tool_results = Vec::new();

    for msg in &request.messages {
        // Create message summary
        let content = msg.content.as_text();
        let has_tool_use = msg.content.has_tool_use();
        let has_tool_result = !msg.content.get_tool_results().is_empty();

        messages.push(MessageSummary {
            role: msg.role.clone(),
            content,
            has_tool_use,
            has_tool_result,
        });

        // Extract tool uses from assistant messages
        if msg.role == "assistant" {
            tool_uses.extend(msg.content.get_tool_uses());
        }

        // Extract tool results from user messages
        if msg.role == "user" {
            tool_results.extend(msg.content.get_tool_results());
        }
    }

    // Extract available tool names
    let available_tools: Vec<String> = request
        .tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
        .collect();

    FullContext {
        system_prompt,
        messages,
        tool_uses,
        tool_results,
        model: request.model.clone(),
        user_id,
        agent_id,
        parent_agent_id,
        run_id,
        available_tools,
    }
}

/// Detect if the user's message indicates followup behavior
/// This helps the brain understand if the previous response was helpful
pub fn detect_followup_signal(user_message: &str) -> Option<FollowupSignal> {
    let lower = user_message.to_lowercase();

    // Positive signals
    if lower.contains("thanks")
        || lower.contains("thank you")
        || lower.contains("perfect")
        || lower.contains("great")
        || lower.contains("awesome")
        || lower.contains("that works")
        || lower.contains("exactly")
    {
        return Some(FollowupSignal::Positive);
    }

    // Negative signals
    if lower.contains("no,")
        || lower.contains("wrong")
        || lower.contains("that's not")
        || lower.contains("incorrect")
        || lower.contains("don't")
        || lower.contains("stop")
        || lower.starts_with("no ")
        || lower.contains("try again")
        || lower.contains("not what i")
    {
        return Some(FollowupSignal::Negative);
    }

    // Correction signals (user is providing the right answer)
    if lower.contains("actually,")
        || lower.contains("i meant")
        || lower.contains("instead,")
        || lower.contains("should be")
    {
        return Some(FollowupSignal::Correction);
    }

    // Continuation signals (task continues normally)
    if lower.contains("now ")
        || lower.contains("next,")
        || lower.contains("also,")
        || lower.contains("and then")
        || lower.contains("continue")
    {
        return Some(FollowupSignal::Continuation);
    }

    None
}

/// Type of followup signal detected from user message
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FollowupSignal {
    /// User expressed satisfaction
    Positive,
    /// User indicated something was wrong
    Negative,
    /// User is correcting the assistant
    Correction,
    /// User is continuing with the task
    Continuation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_positive_followup() {
        assert_eq!(
            detect_followup_signal("Thanks, that works!"),
            Some(FollowupSignal::Positive)
        );
        assert_eq!(
            detect_followup_signal("Perfect, exactly what I needed"),
            Some(FollowupSignal::Positive)
        );
    }

    #[test]
    fn test_detect_negative_followup() {
        assert_eq!(
            detect_followup_signal("No, that's not right"),
            Some(FollowupSignal::Negative)
        );
        assert_eq!(
            detect_followup_signal("Wrong, try again"),
            Some(FollowupSignal::Negative)
        );
    }

    #[test]
    fn test_detect_correction() {
        assert_eq!(
            detect_followup_signal("Actually, I meant the other file"),
            Some(FollowupSignal::Correction)
        );
    }

    #[test]
    fn test_detect_continuation() {
        assert_eq!(
            detect_followup_signal("Now let's add the tests"),
            Some(FollowupSignal::Continuation)
        );
    }

    #[test]
    fn test_no_signal() {
        assert_eq!(detect_followup_signal("What is the weather?"), None);
    }
}
