//! Cortex Types - Shared data structures for the transparent memory layer
//!
//! These types represent the full context that flows through Cortex,
//! enabling the brain to see everything Claude sees.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Configuration Constants
// ============================================================================

/// Maximum characters for system prompt preview in context string
pub const SYSTEM_PROMPT_PREVIEW_CHARS: usize = 2000;

/// Maximum characters for message preview in context string
pub const MESSAGE_PREVIEW_CHARS: usize = 1000;

/// Maximum characters for response preview in encoding
pub const RESPONSE_PREVIEW_CHARS: usize = 2000;

/// Maximum number of recent messages to include in context string
pub const MAX_RECENT_MESSAGES: usize = 20;

/// Session TTL in seconds (1 hour)
pub const SESSION_TTL_SECS: u64 = 3600;

/// Default max memories to retrieve
pub const DEFAULT_MAX_MEMORIES: usize = 5;

/// Brain API timeout in seconds
pub const BRAIN_TIMEOUT_SECS: u64 = 5;


// ============================================================================
// Claude API Types
// ============================================================================

/// Claude API message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// Message content - can be simple text or structured blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// Extract plain text from message content
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    ContentBlock::ToolResult { content, .. } => {
                        content.as_ref().map(|c| c.as_text())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Check if content contains tool use
    pub fn has_tool_use(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Blocks(blocks) => {
                blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }))
            }
        }
    }

    /// Extract tool uses from content
    pub fn get_tool_uses(&self) -> Vec<ToolUseInfo> {
        match self {
            MessageContent::Text(_) => vec![],
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, name, input } => Some(ToolUseInfo {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract tool results from content
    pub fn get_tool_results(&self) -> Vec<ToolResultInfo> {
        match self {
            MessageContent::Text(_) => vec![],
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    } => Some(ToolResultInfo {
                        tool_use_id: tool_use_id.clone(),
                        content: content.as_ref().map(|c| c.as_text()).unwrap_or_default(),
                        is_error: is_error.unwrap_or(false),
                    }),
                    _ => None,
                })
                .collect(),
        }
    }
}

/// Content block types in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Tool result content - can be text or blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

impl ToolResultContent {
    pub fn as_text(&self) -> String {
        match self {
            ToolResultContent::Text(s) => s.clone(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ToolResultBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

/// System prompt - can be text or blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

impl SystemContent {
    /// Extract plain text from system content
    pub fn as_text(&self) -> String {
        match self {
            SystemContent::Text(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    SystemBlock::Text { text, .. } => text.clone(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Convert to blocks and append memory injection, preserving cache_control
    pub fn into_blocks_with_injection(self, memory_block: &str) -> Vec<SystemBlock> {
        let mut blocks = match self {
            SystemContent::Text(s) => vec![SystemBlock::Text {
                text: s,
                cache_control: None,
            }],
            SystemContent::Blocks(b) => b,
        };

        // Find if any existing block has cache_control to preserve caching behavior
        let has_cached_block = blocks.iter().any(|b| match b {
            SystemBlock::Text { cache_control, .. } => cache_control.is_some(),
        });

        // Add memory block - if there's caching, don't cache the dynamic memories
        blocks.push(SystemBlock::Text {
            text: memory_block.to_string(),
            // Memory block should NOT be cached (it changes every request)
            // But we preserve existing blocks' cache_control
            cache_control: if has_cached_block {
                None // Explicitly no cache for dynamic content
            } else {
                None
            },
        });
        blocks
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SystemBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Full Claude API request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemContent>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Extracted Information
// ============================================================================

/// Tool use information extracted from messages
#[derive(Debug, Clone, Serialize)]
pub struct ToolUseInfo {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool result information extracted from messages
#[derive(Debug, Clone, Serialize)]
pub struct ToolResultInfo {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Full context extracted from a request - what the brain sees
#[derive(Debug, Clone, Serialize)]
pub struct FullContext {
    /// The complete system prompt
    pub system_prompt: Option<String>,

    /// All messages in the conversation
    pub messages: Vec<MessageSummary>,

    /// Tool uses from assistant messages
    pub tool_uses: Vec<ToolUseInfo>,

    /// Tool results from user messages
    pub tool_results: Vec<ToolResultInfo>,

    /// Model being used
    pub model: String,

    /// User identifier
    pub user_id: String,

    /// Agent identifier (for sub-agent tracking)
    pub agent_id: Option<String>,

    /// Parent agent identifier (for hierarchical tracking)
    pub parent_agent_id: Option<String>,

    /// Run identifier (groups requests within one execution)
    pub run_id: Option<String>,

    /// Available tools
    pub available_tools: Vec<String>,
}

impl FullContext {
    /// Convert to a string representation for the brain
    /// Uses smart truncation that preserves sentence boundaries
    pub fn to_context_string(&self) -> String {
        let mut parts = Vec::new();

        // System prompt summary - truncate at sentence boundary
        if let Some(ref sys) = self.system_prompt {
            let preview = smart_truncate(sys, SYSTEM_PROMPT_PREVIEW_CHARS);
            if !preview.is_empty() {
                parts.push(format!("[System]: {}", preview));
            }
        }

        // Recent messages - more of them, smarter truncation
        let recent: Vec<_> = self
            .messages
            .iter()
            .rev()
            .take(MAX_RECENT_MESSAGES)
            .rev()
            .collect();

        for msg in recent {
            let preview = smart_truncate(&msg.content, MESSAGE_PREVIEW_CHARS);
            if !preview.is_empty() {
                parts.push(format!("[{}]: {}", msg.role, preview));
            }
        }

        // Tool activity summary with more detail
        if !self.tool_uses.is_empty() {
            let tools: Vec<String> = self
                .tool_uses
                .iter()
                .map(|t| {
                    // Include brief input summary for context
                    let input_preview = t.input.to_string();
                    let input_short: String = input_preview.chars().take(100).collect();
                    format!("{}({})", t.name, input_short)
                })
                .collect();
            parts.push(format!("[Tools]: {}", tools.join(", ")));
        }

        // Tool results summary (especially errors)
        let errors: Vec<_> = self
            .tool_results
            .iter()
            .filter(|r| r.is_error)
            .collect();
        if !errors.is_empty() {
            let error_summaries: Vec<String> = errors
                .iter()
                .map(|e| smart_truncate(&e.content, 200))
                .collect();
            parts.push(format!("[Errors]: {}", error_summaries.join("; ")));
        }

        parts.join("\n")
    }

    /// Get the last user message
    pub fn last_user_message(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
    }
}

/// Smart truncation that tries to preserve sentence boundaries
fn smart_truncate(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    // Take max_chars, then find last sentence boundary
    let truncated: String = text.chars().take(max_chars).collect();

    // Try to find a good break point
    if let Some(pos) = truncated.rfind(|c| c == '.' || c == '!' || c == '?' || c == '\n') {
        if pos > max_chars / 2 {
            // Only use sentence boundary if it's past halfway
            return format!("{}", &truncated[..=pos]);
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

/// Simplified message for context
#[derive(Debug, Clone, Serialize)]
pub struct MessageSummary {
    pub role: String,
    pub content: String,
    pub has_tool_use: bool,
    pub has_tool_result: bool,
}

// ============================================================================
// Brain API Types
// ============================================================================

/// Request to brain's proactive_context endpoint
#[derive(Debug, Serialize)]
pub struct ProactiveContextRequest {
    pub user_id: String,
    pub context: String,
    pub max_results: usize,
    pub auto_ingest: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_followup: Option<String>,
}

/// Memory from brain
#[derive(Debug, Clone, Deserialize)]
pub struct SurfacedMemory {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub score: f32,
}

/// Response from brain's proactive_context
#[derive(Debug, Deserialize, Default)]
pub struct ProactiveContextResponse {
    #[serde(default)]
    pub memories: Vec<SurfacedMemory>,
    #[serde(default)]
    pub feedback_processed: Option<FeedbackProcessed>,
}

#[derive(Debug, Deserialize)]
pub struct FeedbackProcessed {
    pub memories_evaluated: usize,
    #[serde(default)]
    #[allow(dead_code)]
    pub reinforced: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub weakened: Vec<String>,
}

/// Request to brain's remember endpoint
#[derive(Debug, Serialize)]
pub struct RememberRequest {
    pub user_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotional_valence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

/// Response from brain's remember endpoint
#[derive(Debug, Deserialize)]
pub struct RememberResponse {
    pub id: String,
    pub success: bool,
}

/// Request to brain's reinforce endpoint
#[derive(Debug, Serialize)]
pub struct ReinforceRequest {
    pub user_id: String,
    pub ids: Vec<String>,
    pub outcome: String,
}

/// Response from brain's reinforce endpoint
#[derive(Debug, Deserialize)]
pub struct ReinforceResponse {
    #[serde(default)]
    pub memories_processed: usize,
}

// ============================================================================
// Session State
// ============================================================================

/// Per-session state for tracking interactions
#[derive(Debug, Clone)]
pub struct Session {
    /// Last response from Claude (for feedback loop)
    pub last_response: Option<String>,

    /// Memory IDs from last proactive context (for reinforcement)
    pub last_memory_ids: Vec<String>,

    /// Tools used in last response
    pub last_tool_uses: Vec<String>,

    /// Interaction count in this session
    pub interaction_count: u32,

    /// Last user message (for followup detection)
    pub last_user_message: Option<String>,

    /// When this session was last accessed
    pub last_accessed: Instant,

    /// When this session was created
    pub created_at: Instant,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            last_response: None,
            last_memory_ids: Vec::new(),
            last_tool_uses: Vec::new(),
            interaction_count: 0,
            last_user_message: None,
            last_accessed: Instant::now(),
            created_at: Instant::now(),
        }
    }
}

impl Session {
    /// Check if this session has expired
    pub fn is_expired(&self) -> bool {
        self.last_accessed.elapsed().as_secs() > SESSION_TTL_SECS
    }

    /// Touch the session to update last_accessed
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

// ============================================================================
// Cortex Configuration
// ============================================================================

/// LLM API format
#[derive(Debug, Clone, PartialEq)]
pub enum LlmFormat {
    /// Anthropic Messages API (system as separate field)
    Anthropic,
    /// OpenAI Chat Completions API (system as first message)
    OpenAI,
}

impl LlmFormat {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "openai" | "gpt" | "mistral" | "ollama" | "local" => LlmFormat::OpenAI,
            _ => LlmFormat::Anthropic,
        }
    }
}

/// Cortex runtime configuration
#[derive(Debug, Clone)]
pub struct CortexConfig {
    /// Port to listen on
    pub port: u16,

    /// Brain API URL
    pub brain_url: String,

    /// Brain API key
    pub brain_api_key: String,

    /// Max memories to retrieve
    pub max_memories: usize,

    /// Whether to auto-ingest context
    pub auto_ingest: bool,

    /// Brain API timeout in seconds
    pub brain_timeout_secs: u64,

    /// Whether brain is available (circuit breaker state)
    pub brain_available: bool,

    /// Upstream LLM URL (e.g., api.anthropic.com, api.openai.com, localhost:11434)
    pub upstream_url: String,

    /// Upstream LLM API format
    pub upstream_format: LlmFormat,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self {
            port: 3031,
            brain_url: "http://127.0.0.1:3030".to_string(),
            brain_api_key: "sk-shodh-dev-local-testing-key".to_string(),
            max_memories: DEFAULT_MAX_MEMORIES,
            auto_ingest: false, // Disabled to avoid double-storing
            brain_timeout_secs: BRAIN_TIMEOUT_SECS,
            brain_available: true,
            upstream_url: "https://api.anthropic.com".to_string(),
            upstream_format: LlmFormat::Anthropic,
        }
    }
}

impl CortexConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        let upstream_url = std::env::var("UPSTREAM_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

        // Auto-detect format from URL if not explicitly set
        let upstream_format = std::env::var("UPSTREAM_FORMAT")
            .map(|f| LlmFormat::from_str(&f))
            .unwrap_or_else(|_| {
                if upstream_url.contains("openai.com")
                    || upstream_url.contains("localhost")
                    || upstream_url.contains("127.0.0.1:11434") // Ollama
                    || upstream_url.contains("mistral")
                {
                    LlmFormat::OpenAI
                } else {
                    LlmFormat::Anthropic
                }
            });

        Self {
            port: std::env::var("CORTEX_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(3031),
            brain_url: std::env::var("SHODH_API_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string()),
            brain_api_key: std::env::var("SHODH_API_KEY")
                .unwrap_or_else(|_| "sk-shodh-dev-local-testing-key".to_string()),
            max_memories: std::env::var("CORTEX_MAX_MEMORIES")
                .ok()
                .and_then(|m| m.parse().ok())
                .unwrap_or(DEFAULT_MAX_MEMORIES),
            auto_ingest: std::env::var("CORTEX_AUTO_INGEST")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            brain_timeout_secs: std::env::var("CORTEX_BRAIN_TIMEOUT")
                .ok()
                .and_then(|t| t.parse().ok())
                .unwrap_or(BRAIN_TIMEOUT_SECS),
            brain_available: true,
            upstream_url,
            upstream_format,
        }
    }
}
