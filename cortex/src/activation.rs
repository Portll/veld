//! Activation Module - Retrieve relevant memories from the brain
//!
//! Like spreading activation in neural networks, this module queries
//! the brain with the current context and receives relevant memories
//! that should be activated (brought into working memory).

use crate::types::{
    FullContext, ProactiveContextRequest, ProactiveContextResponse, Session, SurfacedMemory,
};
use tracing::{debug, info, warn};

/// Result of memory activation
#[derive(Debug)]
pub struct ActivationResult {
    /// Memories that were activated (surfaced)
    pub memories: Vec<SurfacedMemory>,

    /// Memory IDs for later reinforcement
    pub memory_ids: Vec<String>,

    /// Whether feedback was processed from previous interaction
    #[allow(dead_code)]
    pub feedback_processed: bool,

    /// Whether there was an error contacting the brain
    pub brain_error: bool,
}

impl ActivationResult {
    /// Create an empty result (used when brain is unavailable)
    pub fn empty() -> Self {
        Self {
            memories: vec![],
            memory_ids: vec![],
            feedback_processed: false,
            brain_error: false,
        }
    }
}

/// Activate relevant memories from the brain
///
/// Sends the full context to the brain and receives memories that
/// are relevant to the current situation (spreading activation).
pub async fn activate_memories(
    http_client: &reqwest::Client,
    brain_url: &str,
    brain_api_key: &str,
    context: &FullContext,
    session: &Session,
    max_memories: usize,
) -> ActivationResult {
    // Build the context string from full context
    let context_string = context.to_context_string();

    // Prepare request to brain
    let request = ProactiveContextRequest {
        user_id: context.user_id.clone(),
        context: context_string,
        max_results: max_memories,
        auto_ingest: false, // Don't auto-ingest - encoding handles this
        previous_response: session.last_response.clone(),
        user_followup: context.last_user_message().map(|s| s.to_string()),
    };

    debug!(
        user_id = %context.user_id,
        context_len = request.context.len(),
        max_memories = max_memories,
        has_previous = session.last_response.is_some(),
        "Activating memories from brain"
    );

    // Call brain's proactive_context endpoint
    let (response, brain_error) = match http_client
        .post(format!("{}/api/proactive_context", brain_url))
        .header("Content-Type", "application/json")
        .header("X-API-Key", brain_api_key)
        .json(&request)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<ProactiveContextResponse>().await {
                Ok(data) => (data, false),
                Err(e) => {
                    warn!("Failed to parse brain response: {}", e);
                    (ProactiveContextResponse::default(), true)
                }
            }
        }
        Ok(resp) => {
            warn!("Brain returned error status: {}", resp.status());
            (ProactiveContextResponse::default(), true)
        }
        Err(e) => {
            warn!("Failed to contact brain: {}", e);
            (ProactiveContextResponse::default(), true)
        }
    };

    // Extract memory IDs for later reinforcement
    let memory_ids: Vec<String> = response.memories.iter().map(|m| m.id.clone()).collect();

    // Log activation
    if !response.memories.is_empty() {
        info!(
            "Activated {} memories for user {}",
            response.memories.len(),
            context.user_id
        );
        for (i, mem) in response.memories.iter().enumerate() {
            debug!(
                "  [{}] {}: {:.0}% - {}...",
                i + 1,
                mem.memory_type,
                mem.score * 100.0,
                mem.content.chars().take(60).collect::<String>()
            );
        }
    }

    // Check if feedback was processed
    let feedback_processed = response
        .feedback_processed
        .map(|f| f.memories_evaluated > 0)
        .unwrap_or(false);

    if feedback_processed {
        debug!("Brain processed feedback from previous interaction");
    }

    ActivationResult {
        memories: response.memories,
        memory_ids,
        feedback_processed,
        brain_error,
    }
}

/// Activate memories with retry on failure
#[allow(dead_code)]
pub async fn activate_memories_with_retry(
    http_client: &reqwest::Client,
    brain_url: &str,
    brain_api_key: &str,
    context: &FullContext,
    session: &Session,
    max_memories: usize,
    max_retries: u32,
) -> ActivationResult {
    let mut last_result = ActivationResult::empty();

    for attempt in 0..max_retries {
        last_result = activate_memories(
            http_client,
            brain_url,
            brain_api_key,
            context,
            session,
            max_memories,
        )
        .await;

        if !last_result.memories.is_empty() || !last_result.brain_error || attempt == max_retries - 1
        {
            break;
        }

        debug!("Retry {} for memory activation", attempt + 1);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    last_result
}
