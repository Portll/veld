//! Unified shodh binary - MCP server + Claude Code hooks
//!
//! Usage:
//!   shodh serve              - Run as MCP server (stdio transport)
//!   shodh hook session-start - Output session start hook JSON
//!   shodh hook prompt <msg>  - Output prompt submit hook JSON
//!
//! Both modes use the same core memory functionality, ready for future MCP push.

use anyhow::Result;
use clap::{Parser, Subcommand};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Content, ErrorCode, Implementation, ProtocolVersion, ServerCapabilities,
        ServerInfo,
    },
    schemars, tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, sync::Arc};

// =============================================================================
// CLI STRUCTURE
// =============================================================================

#[derive(Parser)]
#[command(name = "shodh")]
#[command(about = "Shodh Memory - MCP server and Claude Code hooks")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run as MCP server (stdio transport)
    Serve {
        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for memory operations
        #[arg(long, env = "SHODH_USER_ID", default_value = "claude-code")]
        user_id: String,
    },

    /// Output Claude Code hook JSON
    Hook {
        #[command(subcommand)]
        hook_type: HookType,
    },
}

#[derive(Subcommand)]
enum HookType {
    /// Session start hook - restore context
    SessionStart {
        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for memory operations
        #[arg(long, env = "SHODH_USER_ID", default_value = "claude-code")]
        user_id: String,

        /// Project directory (from CLAUDE_PROJECT_DIR)
        #[arg(long, env = "CLAUDE_PROJECT_DIR")]
        project_dir: Option<String>,
    },

    /// User prompt submit hook - inject relevant context
    Prompt {
        /// The user's message
        message: String,

        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for memory operations
        #[arg(long, env = "SHODH_USER_ID", default_value = "claude-code")]
        user_id: String,
    },
}

// =============================================================================
// API CLIENT
// =============================================================================

/// HTTP client for the shodh-memory API (async version for MCP tools)
#[derive(Clone, Debug)]
struct AsyncApiClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    user_id: String,
}

impl AsyncApiClient {
    fn new(base_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            user_id,
        }
    }

    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, endpoint);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("API error {}: {}", status, text);
        }

        Ok(resp.json().await?)
    }
}

/// HTTP client for the shodh-memory API (blocking version for hooks)
#[derive(Clone, Debug)]
struct BlockingApiClient {
    client: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
    user_id: String,
}

impl BlockingApiClient {
    fn new(base_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url,
            api_key,
            user_id,
        }
    }

    fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, endpoint);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("API error {}: {}", status, text);
        }

        Ok(resp.json()?)
    }
}

// =============================================================================
// API REQUEST/RESPONSE TYPES
// =============================================================================

#[derive(Serialize)]
struct ProactiveContextRequest {
    user_id: String,
    context: String,
    max_results: u32,
    auto_ingest: bool,
}

#[derive(Deserialize)]
struct ProactiveContextResponse {
    memories: Vec<SurfacedMemory>,
}

#[derive(Deserialize)]
struct SurfacedMemory {
    id: String,
    content: String,
    memory_type: String,
    relevance_score: f32,
}

#[derive(Serialize)]
struct ListTodosRequest {
    user_id: String,
    status: Vec<String>,
}

#[derive(Deserialize)]
struct ListTodosResponse {
    todos: Vec<Todo>,
}

#[derive(Deserialize)]
struct Todo {
    id: String,
    content: String,
    status: String,
    priority: Option<String>,
    #[allow(dead_code)]
    project: Option<String>,
}

#[derive(Serialize)]
struct RememberRequest {
    user_id: String,
    content: String,
    memory_type: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct RememberResponse {
    id: String,
    message: String,
}

#[derive(Serialize)]
struct RecallRequest {
    user_id: String,
    query: String,
    limit: Option<u32>,
    mode: Option<String>,
}

#[derive(Deserialize)]
struct RecallResponse {
    memories: Vec<RecalledMemory>,
}

#[derive(Deserialize)]
struct RecalledMemory {
    id: String,
    content: String,
    memory_type: String,
    similarity: f32,
    #[allow(dead_code)]
    tags: Vec<String>,
}

// =============================================================================
// HOOK OUTPUT
// =============================================================================

#[derive(Serialize)]
struct HookOutput {
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: HookSpecificOutput,
}

#[derive(Serialize)]
struct HookSpecificOutput {
    #[serde(rename = "hookEventName")]
    hook_event_name: String,
    #[serde(rename = "additionalContext")]
    additional_context: String,
}

fn output_hook(event_name: &str, context: &str) {
    let output = HookOutput {
        hook_specific_output: HookSpecificOutput {
            hook_event_name: event_name.to_string(),
            additional_context: context.to_string(),
        },
    };
    println!("{}", serde_json::to_string(&output).unwrap());
}

// =============================================================================
// HOOK HANDLERS
// =============================================================================

fn handle_session_start(api_url: &str, api_key: &str, user_id: &str, project_dir: Option<&str>) {
    let client = BlockingApiClient::new(
        api_url.to_string(),
        api_key.to_string(),
        user_id.to_string(),
    );

    let dir_name = project_dir
        .and_then(|p| std::path::Path::new(p).file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Get proactive context
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: format!("Starting session in {}", dir_name),
            max_results: 3,
            auto_ingest: false,
        },
    );

    // Get pending todos
    let todos_result: Result<ListTodosResponse> = client.post(
        "/api/todos",
        &ListTodosRequest {
            user_id: user_id.to_string(),
            status: vec!["todo".to_string(), "in_progress".to_string()],
        },
    );

    // Build context string
    let mut context_parts = vec!["## Shodh Memory Context Restored\n".to_string()];

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("### Relevant Memories:".to_string());
            for mem in ctx.memories.iter().take(3) {
                context_parts.push(format!(
                    "- [{}] {}: {}",
                    mem.memory_type,
                    &mem.id[..8.min(mem.id.len())],
                    mem.content.chars().take(200).collect::<String>()
                ));
            }
            context_parts.push(String::new());
        }
    }

    if let Ok(todos) = todos_result {
        let in_progress: Vec<_> = todos
            .todos
            .iter()
            .filter(|t| t.status == "in_progress")
            .collect();
        let pending: Vec<_> = todos.todos.iter().filter(|t| t.status == "todo").collect();

        if !in_progress.is_empty() || !pending.is_empty() {
            context_parts.push("### Pending Todos:".to_string());

            if !in_progress.is_empty() {
                context_parts.push("**In Progress:**".to_string());
                for todo in in_progress.iter().take(5) {
                    context_parts.push(format!("- ⏳ {}", todo.content));
                }
            }

            if !pending.is_empty() {
                context_parts.push("**Todo:**".to_string());
                for todo in pending.iter().take(5) {
                    let priority = todo.priority.as_deref().unwrap_or("");
                    let prefix = match priority {
                        "urgent" => "🔴",
                        "high" => "🟠",
                        "medium" => "🟡",
                        _ => "⚪",
                    };
                    context_parts.push(format!("- {} {}", prefix, todo.content));
                }
            }
        }
    }

    output_hook("SessionStart", &context_parts.join("\n"));
}

fn handle_prompt_submit(api_url: &str, api_key: &str, user_id: &str, message: &str) {
    let client = BlockingApiClient::new(
        api_url.to_string(),
        api_key.to_string(),
        user_id.to_string(),
    );

    // Get proactive context based on user message
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: message.to_string(),
            max_results: 5,
            auto_ingest: true, // Store the context for implicit feedback
        },
    );

    let mut context_parts = Vec::new();

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("## Relevant Memories (auto-surfaced)\n".to_string());
            for mem in ctx.memories.iter() {
                let relevance = (mem.relevance_score * 100.0) as u32;
                context_parts.push(format!(
                    "- [{}%] **{}**: {}",
                    relevance,
                    mem.memory_type,
                    mem.content.chars().take(300).collect::<String>()
                ));
            }
        }
    }

    // Only output if we have relevant context
    if !context_parts.is_empty() {
        output_hook("UserPromptSubmit", &context_parts.join("\n"));
    } else {
        // Output empty hook (no context to inject)
        output_hook("UserPromptSubmit", "");
    }
}

// =============================================================================
// MCP TOOL PARAMETER TYPES
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RememberParams {
    /// The content to remember
    content: String,
    /// Type of memory (Observation, Decision, Learning, etc.)
    #[serde(rename = "type")]
    memory_type: Option<String>,
    /// Optional tags for categorization
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RecallParams {
    /// Natural language search query
    query: String,
    /// Maximum number of results (default: 5)
    limit: Option<u32>,
    /// Retrieval mode: semantic, associative, or hybrid
    mode: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ProactiveContextParams {
    /// Current conversation context
    context: String,
    /// Maximum memories to surface (default: 5)
    max_results: Option<u32>,
    /// Auto-store context for feedback (default: true)
    auto_ingest: Option<bool>,
}

// =============================================================================
// LINEAGE MCP TOOL PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageTraceParams {
    /// Memory ID to trace lineage from
    memory_id: String,
    /// Direction: "backward" (find causes), "forward" (find effects), "both"
    direction: Option<String>,
    /// Maximum depth to traverse (default: 10)
    max_depth: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageConfirmParams {
    /// ID of the inferred edge to confirm
    edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageRejectParams {
    /// ID of the inferred edge to reject
    edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageLinkParams {
    /// Source memory ID (the cause/origin)
    from_memory_id: String,
    /// Target memory ID (the effect/result)
    to_memory_id: String,
    /// Relation type: Caused, ResolvedBy, InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo
    relation: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageStatsParams {
    /// Optional - leave empty to get stats for current user
    #[serde(default)]
    _placeholder: Option<String>,
}

// Lineage API request types (for API calls)
#[derive(Serialize)]
struct LineageTraceRequest {
    user_id: String,
    memory_id: String,
    direction: String,
    max_depth: u32,
}

#[derive(Serialize)]
struct LineageEdgeRequest {
    user_id: String,
    edge_id: String,
}

#[derive(Serialize)]
struct LineageAddEdgeRequest {
    user_id: String,
    from_memory_id: String,
    to_memory_id: String,
    relation: String,
}

#[derive(Serialize)]
struct LineageStatsRequest {
    user_id: String,
}

// Lineage API response types
#[derive(Deserialize)]
struct LineageTraceResponse {
    root: String,
    direction: String,
    edges: Vec<LineageEdgeInfo>,
    path: Vec<String>,
    depth: usize,
}

#[derive(Deserialize)]
struct LineageEdgeInfo {
    id: String,
    from: String,
    to: String,
    relation: String,
    confidence: f32,
    source: String,
}

#[derive(Deserialize)]
struct LineageConfirmResponse {
    message: String,
    edge_id: String,
}

#[derive(Deserialize)]
struct LineageRejectResponse {
    message: String,
    deleted: bool,
}

#[derive(Deserialize)]
struct LineageAddResponse {
    message: String,
    edge_id: String,
}

#[derive(Deserialize)]
struct LineageStatsResponse {
    total_edges: usize,
    inferred_edges: usize,
    confirmed_edges: usize,
    explicit_edges: usize,
    total_branches: usize,
    active_branches: usize,
    edges_by_relation: std::collections::HashMap<String, usize>,
    avg_confidence: f32,
}

// =============================================================================
// MCP SERVER
// =============================================================================

#[derive(Debug, Clone)]
struct ShodhMcpServer {
    client: Arc<AsyncApiClient>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ShodhMcpServer {
    fn new(api_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: Arc::new(AsyncApiClient::new(api_url, api_key, user_id)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a memory for future recall. Use this to remember important information, decisions, user preferences, project context, or anything you want to recall later."
    )]
    async fn remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RememberResponse> = self
            .client
            .post(
                "/api/remember",
                &RememberRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    memory_type: params.memory_type,
                    tags: params.tags,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Stored memory: {} ({})",
                resp.id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Search memories using semantic, associative, or hybrid retrieval. Modes: 'semantic' (vector similarity), 'associative' (graph traversal), 'hybrid' (combines both)."
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RecallResponse> = self
            .client
            .post(
                "/api/recall",
                &RecallRequest {
                    user_id: self.client.user_id.clone(),
                    query: params.query,
                    limit: params.limit,
                    mode: params.mode,
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Found {} memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "**[{}]** {} (similarity: {:.0}%)\n{}\n\n",
                        mem.memory_type,
                        &mem.id[..8.min(mem.id.len())],
                        mem.similarity * 100.0,
                        mem.content
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "REQUIRED: Call this to surface relevant memories based on current context. Enables automatic memory surfacing and implicit feedback learning."
    )]
    async fn proactive_context(
        &self,
        Parameters(params): Parameters<ProactiveContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ProactiveContextResponse> = self
            .client
            .post(
                "/api/proactive_context",
                &ProactiveContextRequest {
                    user_id: self.client.user_id.clone(),
                    context: params.context,
                    max_results: params.max_results.unwrap_or(5),
                    auto_ingest: params.auto_ingest.unwrap_or(true),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Surfaced {} relevant memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "- [{}%] **{}**: {}\n",
                        (mem.relevance_score * 100.0) as u32,
                        mem.memory_type,
                        mem.content.chars().take(200).collect::<String>()
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    // =========================================================================
    // LINEAGE TOOLS - Causal Memory Tracking
    // =========================================================================

    #[tool(
        description = "Trace the causal lineage of a memory. Find what caused it (backward), what it led to (forward), or both. Useful for understanding 'why' something happened."
    )]
    async fn lineage_trace(
        &self,
        Parameters(params): Parameters<LineageTraceParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageTraceResponse> = self
            .client
            .post(
                "/api/lineage/trace",
                &LineageTraceRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    direction: params.direction.unwrap_or_else(|| "backward".to_string()),
                    max_depth: params.max_depth.unwrap_or(10),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!(
                    "**Lineage Trace** ({})\n\nRoot: {}\nDepth: {}\n\n",
                    resp.direction, resp.root, resp.depth
                );

                if resp.edges.is_empty() {
                    output.push_str("No causal connections found.\n");
                } else {
                    output.push_str("**Causal Chain:**\n");
                    for edge in &resp.edges {
                        let confidence = (edge.confidence * 100.0) as u32;
                        let source_icon = match edge.source.as_str() {
                            "Confirmed" => "✓",
                            "Explicit" => "⚡",
                            _ => "?",
                        };
                        output.push_str(&format!(
                            "  {} --[{} {}% {}]--> {}\n",
                            &edge.from[..8.min(edge.from.len())],
                            edge.relation,
                            confidence,
                            source_icon,
                            &edge.to[..8.min(edge.to.len())]
                        ));
                    }

                    output.push_str(&format!("\n**Path:** {}\n", resp.path.join(" → ")));
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Confirm an inferred causal relationship between memories. This improves the system's confidence and learning."
    )]
    async fn lineage_confirm(
        &self,
        Parameters(params): Parameters<LineageConfirmParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageConfirmResponse> = self
            .client
            .post(
                "/api/lineage/confirm",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✓ Confirmed edge: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Reject an incorrectly inferred causal relationship. This helps the system learn better inference patterns."
    )]
    async fn lineage_reject(
        &self,
        Parameters(params): Parameters<LineageRejectParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageRejectResponse> = self
            .client
            .post(
                "/api/lineage/reject",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✗ Rejected edge: {}",
                resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Create an explicit causal link between two memories. Relations: Caused (Error→Todo), ResolvedBy (Todo→Learning), InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo."
    )]
    async fn lineage_link(
        &self,
        Parameters(params): Parameters<LineageLinkParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageAddResponse> = self
            .client
            .post(
                "/api/lineage/link",
                &LineageAddEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    from_memory_id: params.from_memory_id,
                    to_memory_id: params.to_memory_id,
                    relation: params.relation,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "⚡ Created link: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Get statistics about the causal lineage graph - edge counts, relation types, confidence distribution."
    )]
    async fn lineage_stats(
        &self,
        Parameters(_params): Parameters<LineageStatsParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageStatsResponse> = self
            .client
            .post(
                "/api/lineage/stats",
                &LineageStatsRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = "**Lineage Graph Statistics**\n\n".to_string();
                output.push_str(&format!("**Edges:** {}\n", resp.total_edges));
                output.push_str(&format!("  ✓ Confirmed: {}\n", resp.confirmed_edges));
                output.push_str(&format!("  ? Inferred: {}\n", resp.inferred_edges));
                output.push_str(&format!("  ⚡ Explicit: {}\n", resp.explicit_edges));
                output.push_str(&format!(
                    "Average Confidence: {:.1}%\n\n",
                    resp.avg_confidence * 100.0
                ));
                output.push_str(&format!(
                    "**Branches:** {} total, {} active\n\n",
                    resp.total_branches, resp.active_branches
                ));

                if !resp.edges_by_relation.is_empty() {
                    output.push_str("**By Relation Type:**\n");
                    let mut relations: Vec<_> = resp.edges_by_relation.iter().collect();
                    relations.sort_by(|a, b| b.1.cmp(a.1));
                    for (relation, count) in relations {
                        output.push_str(&format!("  {}: {}\n", relation, count));
                    }
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }
}

#[tool_handler]
impl ServerHandler for ShodhMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Shodh Memory - persistent cognitive memory with causal reasoning. \
                 Use proactive_context at session start to surface relevant memories. \
                 Use remember to store decisions, learnings, errors. \
                 Use recall to search memories. \
                 Use lineage_trace to understand 'why' - trace causal chains backward/forward. \
                 Use lineage_link to explicitly connect cause→effect memories. \
                 Use lineage_confirm/reject to improve inference accuracy."
                    .to_string(),
            ),
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            api_url,
            api_key,
            user_id,
        } => {
            eprintln!("Starting shodh MCP server...");
            eprintln!("  API URL: {}", api_url);
            eprintln!("  User ID: {}", user_id);

            let server = ShodhMcpServer::new(api_url, api_key, user_id);
            let service = server.serve(rmcp::transport::stdio()).await?;
            service.waiting().await?;
        }

        Commands::Hook { hook_type } => match hook_type {
            HookType::SessionStart {
                api_url,
                api_key,
                user_id,
                project_dir,
            } => {
                handle_session_start(&api_url, &api_key, &user_id, project_dir.as_deref());
            }

            HookType::Prompt {
                message,
                api_url,
                api_key,
                user_id,
            } => {
                handle_prompt_submit(&api_url, &api_key, &user_id, &message);
            }
        },
    }

    Ok(())
}
