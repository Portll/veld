//! Encoding Module - Store interactions to the brain
//!
//! Like the hippocampus encoding experiences, this module stores
//! each interaction (user message + assistant response) to the brain
//! for long-term memory formation.

use crate::types::{FullContext, RememberRequest, RememberResponse, RESPONSE_PREVIEW_CHARS};
use tracing::{debug, info, warn};

/// Encode an interaction to the brain
///
/// Stores the user's message and Claude's response as a memory,
/// enabling the brain to learn from the interaction.
pub async fn encode_interaction(
    http_client: &reqwest::Client,
    brain_url: &str,
    brain_api_key: &str,
    context: &FullContext,
    response_text: &str,
    tool_uses: &[String],
) -> Option<String> {
    // Skip encoding if response is empty (likely an error or tool-only response)
    if response_text.is_empty() && tool_uses.is_empty() {
        debug!("Skipping encoding: no response text or tools");
        return None;
    }

    // Get the last user message
    let user_message = context.last_user_message().unwrap_or("(no message)");

    // Format the interaction content
    let content = format_interaction(user_message, response_text, tool_uses);

    // Determine memory type based on interaction
    let memory_type = determine_memory_type(user_message, response_text, tool_uses);

    // Generate tags
    let tags = generate_tags(context, tool_uses);

    // Estimate emotional valence from content with context awareness
    let emotional_valence = estimate_valence_contextual(user_message, response_text, tool_uses);

    let request = RememberRequest {
        user_id: context.user_id.clone(),
        content,
        tags,
        memory_type: Some(memory_type.to_string()),
        emotional_valence,
        agent_id: context.agent_id.clone(),
        parent_agent_id: context.parent_agent_id.clone(),
        run_id: context.run_id.clone(),
    };

    debug!(
        user_id = %context.user_id,
        memory_type = %memory_type,
        "Encoding interaction to brain"
    );

    match http_client
        .post(format!("{}/api/remember", brain_url))
        .header("Content-Type", "application/json")
        .header("X-API-Key", brain_api_key)
        .json(&request)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<RememberResponse>().await {
                Ok(data) if data.success => {
                    info!("Encoded interaction as memory {}", data.id);
                    Some(data.id)
                }
                Ok(_) => {
                    warn!("Brain returned success=false for encoding");
                    None
                }
                Err(e) => {
                    warn!("Failed to parse encoding response: {}", e);
                    None
                }
            }
        }
        Ok(resp) => {
            warn!("Brain returned error for encoding: {}", resp.status());
            None
        }
        Err(e) => {
            warn!("Failed to encode to brain: {}", e);
            None
        }
    }
}

/// Format an interaction for storage with smart truncation
fn format_interaction(user_message: &str, response: &str, tool_uses: &[String]) -> String {
    let mut parts = Vec::new();

    // User's message (truncated at sentence boundary if too long)
    let user_preview = smart_truncate(user_message, 500);
    parts.push(format!("User: {}", user_preview));

    // Tools used with count
    if !tool_uses.is_empty() {
        let tool_summary = if tool_uses.len() <= 3 {
            tool_uses.join(", ")
        } else {
            format!(
                "{}, ... (+{} more)",
                tool_uses[..3].join(", "),
                tool_uses.len() - 3
            )
        };
        parts.push(format!("Tools: {}", tool_summary));
    }

    // Response (truncated at sentence boundary if too long)
    let response_preview = smart_truncate(response, RESPONSE_PREVIEW_CHARS);
    parts.push(format!("Assistant: {}", response_preview));

    parts.join("\n")
}

/// Smart truncation that tries to preserve sentence boundaries
fn smart_truncate(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    // Take max_chars, then find last sentence boundary
    let truncated: String = text.chars().take(max_chars).collect();

    // Try to find a good break point (sentence ending)
    if let Some(pos) = truncated.rfind(|c| c == '.' || c == '!' || c == '?' || c == '\n') {
        if pos > max_chars / 2 {
            // Only use sentence boundary if it's past halfway
            return truncated[..=pos].to_string();
        }
    }

    // Fall back to word boundary
    if let Some(pos) = truncated.rfind(char::is_whitespace) {
        if pos > max_chars * 3 / 4 {
            return format!("{}...", &truncated[..pos]);
        }
    }

    format!("{}...", truncated)
}

/// Determine memory type based on interaction content
fn determine_memory_type(user_message: &str, response: &str, tool_uses: &[String]) -> &'static str {
    let user_lower = user_message.to_lowercase();
    let response_lower = response.to_lowercase();

    // Check for decision-related content
    if user_lower.contains("should i")
        || user_lower.contains("which one")
        || user_lower.contains("decide")
        || user_lower.contains("choose between")
        || response_lower.contains("i recommend")
        || response_lower.contains("i suggest")
        || response_lower.contains("the better option")
    {
        return "Decision";
    }

    // Check for learning-related content
    if user_lower.contains("how do")
        || user_lower.contains("what is")
        || user_lower.contains("explain")
        || user_lower.contains("learn")
        || user_lower.contains("understand")
        || response_lower.contains("this means")
        || response_lower.contains("in other words")
        || response_lower.contains("the concept")
    {
        return "Learning";
    }

    // Check for error-related content
    if user_lower.contains("error")
        || user_lower.contains("bug")
        || user_lower.contains("fix")
        || user_lower.contains("wrong")
        || user_lower.contains("doesn't work")
        || user_lower.contains("broken")
        || response_lower.contains("the issue")
        || response_lower.contains("the problem")
        || response_lower.contains("the fix")
    {
        return "Error";
    }

    // Check for task-related content (tool usage)
    if !tool_uses.is_empty() {
        // Code editing
        if tool_uses.iter().any(|t| t == "Edit" || t == "Write") {
            return "Task";
        }
        // Reading/exploring
        if tool_uses
            .iter()
            .any(|t| t == "Read" || t == "Glob" || t == "Grep")
        {
            return "Discovery";
        }
        // Commands
        if tool_uses.iter().any(|t| t == "Bash") {
            return "Task";
        }
    }

    // Default to Conversation
    "Conversation"
}

/// Generate tags for the interaction
fn generate_tags(context: &FullContext, tool_uses: &[String]) -> Vec<String> {
    let mut tags = Vec::new();

    // Add model tag
    tags.push(format!("model:{}", context.model));

    // Add agent tag if present
    if let Some(ref agent_id) = context.agent_id {
        tags.push(format!("agent:{}", agent_id));
    }

    // Add parent agent tag if present (for hierarchy tracking)
    if let Some(ref parent_id) = context.parent_agent_id {
        tags.push(format!("parent_agent:{}", parent_id));
    }

    // Add run tag if present (for grouping)
    if let Some(ref run_id) = context.run_id {
        tags.push(format!("run:{}", run_id));
    }

    // Add tool tags (deduplicated)
    let mut seen_tools = std::collections::HashSet::new();
    for tool in tool_uses {
        if seen_tools.insert(tool.clone()) {
            tags.push(format!("tool:{}", tool));
        }
    }

    // Add cortex tag to identify auto-encoded memories
    tags.push("source:cortex".to_string());

    tags
}

/// Estimate emotional valence with context awareness
///
/// This version considers:
/// - Tool success/failure context
/// - Conversation flow (is this resolving an issue?)
/// - Semantic patterns beyond just keywords
fn estimate_valence_contextual(
    user_message: &str,
    response: &str,
    tool_uses: &[String],
) -> Option<f32> {
    let user_lower = user_message.to_lowercase();
    let response_lower = response.to_lowercase();

    // Context modifiers
    let mut positive_score: f32 = 0.0;
    let mut negative_score: f32 = 0.0;

    // Strong positive indicators
    let strong_positive = [
        "thanks",
        "thank you",
        "perfect",
        "excellent",
        "exactly what i needed",
        "that works",
        "awesome",
        "great job",
    ];

    // Moderate positive indicators
    let moderate_positive = [
        "works",
        "success",
        "done",
        "fixed",
        "solved",
        "helpful",
        "good",
        "nice",
    ];

    // Strong negative indicators
    let strong_negative = [
        "completely wrong",
        "terrible",
        "hate",
        "worst",
        "useless",
        "frustrated",
    ];

    // Moderate negative indicators
    let moderate_negative = [
        "error",
        "bug",
        "wrong",
        "fail",
        "broken",
        "issue",
        "problem",
        "confus",
        "stuck",
        "crash",
    ];

    // Check user message for strong signals
    for word in &strong_positive {
        if user_lower.contains(word) {
            positive_score += 2.0;
        }
    }
    for word in &strong_negative {
        if user_lower.contains(word) {
            negative_score += 2.0;
        }
    }

    // Check for moderate signals in both
    let combined = format!("{} {}", user_lower, response_lower);
    for word in &moderate_positive {
        if combined.contains(word) {
            positive_score += 1.0;
        }
    }
    for word in &moderate_negative {
        if combined.contains(word) {
            negative_score += 1.0;
        }
    }

    // Context: If tools were used successfully, that's positive
    if !tool_uses.is_empty() {
        // If response contains "completed" or "done" with tools, boost positive
        if response_lower.contains("completed")
            || response_lower.contains("done")
            || response_lower.contains("created")
            || response_lower.contains("updated")
        {
            positive_score += 1.0;
        }
    }

    // Context: Error + fix pattern is ultimately positive (resolution)
    if (user_lower.contains("error") || user_lower.contains("bug"))
        && (response_lower.contains("fixed") || response_lower.contains("the issue was"))
    {
        positive_score += 1.5; // Resolution is net positive
    }

    // Calculate final valence
    let total = positive_score + negative_score;
    if total < 1.0 {
        return None; // Not enough signal
    }

    let valence = (positive_score - negative_score) / total;
    Some(valence.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_memory_type_decision() {
        assert_eq!(
            determine_memory_type("Should I use Rust or Python?", "I recommend Rust", &[]),
            "Decision"
        );
    }

    #[test]
    fn test_determine_memory_type_learning() {
        assert_eq!(
            determine_memory_type(
                "What is a mutex?",
                "A mutex is a synchronization primitive",
                &[]
            ),
            "Learning"
        );
    }

    #[test]
    fn test_determine_memory_type_error() {
        assert_eq!(
            determine_memory_type("I got an error", "The issue is with the type", &[]),
            "Error"
        );
    }

    #[test]
    fn test_determine_memory_type_task() {
        assert_eq!(
            determine_memory_type("Update the config", "Done", &["Edit".to_string()]),
            "Task"
        );
    }

    #[test]
    fn test_estimate_valence_positive() {
        let valence = estimate_valence_contextual("Thanks, that's great!", "You're welcome", &[]);
        assert!(valence.is_some());
        assert!(valence.unwrap() > 0.0);
    }

    #[test]
    fn test_estimate_valence_negative() {
        let valence =
            estimate_valence_contextual("This is broken", "There's an error in the code", &[]);
        assert!(valence.is_some());
        assert!(valence.unwrap() < 0.0);
    }

    #[test]
    fn test_estimate_valence_error_resolution() {
        // Error that got fixed should be net positive
        let valence = estimate_valence_contextual(
            "I got an error with the database",
            "The issue was a missing connection. I've fixed it.",
            &["Edit".to_string()],
        );
        assert!(valence.is_some());
        assert!(valence.unwrap() > 0.0, "Error resolution should be positive");
    }

    #[test]
    fn test_estimate_valence_neutral() {
        let valence = estimate_valence_contextual("List the files", "Here are the files", &[]);
        assert!(valence.is_none());
    }

    #[test]
    fn test_smart_truncate_short() {
        let text = "Hello world";
        assert_eq!(smart_truncate(text, 100), "Hello world");
    }

    #[test]
    fn test_smart_truncate_sentence() {
        let text = "First sentence. Second sentence. Third sentence is longer.";
        let truncated = smart_truncate(text, 40);
        assert!(truncated.ends_with(".") || truncated.ends_with("..."));
    }
}
