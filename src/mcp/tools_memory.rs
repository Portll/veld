//! MCP tools over memories, indexes, backups, facts, token accounting and seeding: the
//! half of the TypeScript server's tool set that `tools.rs` did not carry. Same contract
//! rule: tool and parameter names are part of the MCP surface and do not change.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Utc};
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    schemars, tool, tool_router, ErrorData as McpError,
};
use serde::{Deserialize, Serialize};

use super::tools::{mcp_error, url_encode_component_shared};
use super::VeldMcpServer;

/// Upper bound on a caller-supplied result limit, as in the TypeScript server.
const MAX_LIMIT: u32 = 250;
const RULE_30: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
const RULE_42: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
const RULE_52: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

// =============================================================================
// TOKEN ACCOUNTING (mirrors mcp-server/token-tracking.ts)
// =============================================================================

/// Session token accounting fed by every tool call's text output. Estimates at four
/// characters per token, like the TypeScript tracker; exact counts are not the point,
/// the alert at the budget's threshold is.
#[derive(Debug)]
pub(crate) struct TokenTracker {
    budget: u64,
    alert_threshold: f64,
    state: Mutex<TokenState>,
}

#[derive(Debug)]
struct TokenState {
    session_tokens: u64,
    session_start: Instant,
    per_tool: HashMap<String, u64>,
}

pub(crate) struct TokenStatus {
    pub tokens: u64,
    pub budget: u64,
    pub percent: f64,
    pub alert: Option<String>,
}

impl TokenTracker {
    /// Budget from `VELD_TOKEN_BUDGET` (default 100 000) and threshold from
    /// `VELD_ALERT_THRESHOLD` (default 0.9), the TypeScript server's variables.
    pub(crate) fn from_env() -> Self {
        let budget = std::env::var("VELD_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(100_000);
        let alert_threshold = std::env::var("VELD_ALERT_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.9);
        Self::new(budget, alert_threshold)
    }

    pub(crate) fn new(budget: u64, alert_threshold: f64) -> Self {
        Self {
            budget,
            alert_threshold,
            state: Mutex::new(TokenState {
                session_tokens: 0,
                session_start: Instant::now(),
                per_tool: HashMap::new(),
            }),
        }
    }

    pub(crate) fn estimate_tokens(text: &str) -> u64 {
        (text.chars().count() as u64).div_ceil(4)
    }

    /// Record the tokens of one tool's output under the tool's name.
    pub(crate) fn track(&self, tool: &str, text: &str) -> u64 {
        let count = Self::estimate_tokens(text);
        let mut s = self.lock();
        s.session_tokens += count;
        *s.per_tool.entry(tool.to_string()).or_insert(0) += count;
        count
    }

    pub(crate) fn status(&self) -> TokenStatus {
        let s = self.lock();
        let percent = if self.budget == 0 {
            0.0
        } else {
            s.session_tokens as f64 / self.budget as f64
        };
        TokenStatus {
            tokens: s.session_tokens,
            budget: self.budget,
            percent: (percent * 100.0).round() / 100.0,
            alert: (percent >= self.alert_threshold)
                .then(|| format!("context_{}_percent", (self.alert_threshold * 100.0).round())),
        }
    }

    pub(crate) fn session_minutes(&self) -> u64 {
        self.lock().session_start.elapsed().as_secs() / 60
    }

    pub(crate) fn session_tokens(&self) -> u64 {
        self.lock().session_tokens
    }

    pub(crate) fn reset(&self) {
        let mut s = self.lock();
        s.session_tokens = 0;
        s.session_start = Instant::now();
        s.per_tool.clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TokenState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn progress_bar(percent_used: u64) -> String {
    let clamped = percent_used.min(100) as usize;
    let filled = (clamped * 20 + 50) / 100;
    format!("{}{}", "█".repeat(filled), "░".repeat(20 - filled))
}

/// `1234567` → `1,234,567`, the shape `toLocaleString` gives the TypeScript output.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn format_token_status(status: &TokenStatus, session_minutes: u64) -> String {
    let remaining = status.budget.saturating_sub(status.tokens);
    let percent_used = (status.percent * 100.0).round() as u64;
    let mut out = String::from("🌾 Token Status\n");
    out.push_str(&format!(
        "{RULE_42}\n{} {percent_used}%\n{RULE_42}\n",
        progress_bar(percent_used)
    ));
    out.push_str(&format!("Used: {} tokens\n", thousands(status.tokens)));
    out.push_str(&format!("Budget: {} tokens\n", thousands(status.budget)));
    out.push_str(&format!("Remaining: {} tokens\n", thousands(remaining)));
    out.push_str(&format!("Session: {session_minutes} min\n{RULE_42}\n"));
    if status.alert.is_some() {
        out.push_str(&format!(
            "⚠️ ALERT: {percent_used}% used - Consider new session"
        ));
    } else {
        out.push_str("✓ Context window healthy");
    }
    out
}

// =============================================================================
// SHARED WIRE SHAPES
// =============================================================================

/// A memory as the recall-by-tags route serialises it: the full record, of which the
/// text view needs the id, tier, content, type and tags. Top-level `content` and
/// `memory_type` are the older flat shape some routes still emit.
#[derive(Debug, Deserialize)]
struct WireMemory {
    id: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    experience: Option<WireExperience>,
}

#[derive(Debug, Deserialize)]
struct WireExperience {
    #[serde(default)]
    content: String,
    #[serde(default)]
    experience_type: Option<String>,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

impl WireMemory {
    fn content(&self) -> &str {
        self.content
            .as_deref()
            .filter(|c| !c.is_empty())
            .or(self.experience.as_ref().map(|e| e.content.as_str()))
            .unwrap_or("")
    }

    fn memory_type(&self) -> &str {
        self.memory_type
            .as_deref()
            .or(self
                .experience
                .as_ref()
                .and_then(|e| e.memory_type.as_deref()))
            .or(self
                .experience
                .as_ref()
                .and_then(|e| e.experience_type.as_deref()))
            .unwrap_or("Observation")
    }

    fn tags(&self) -> &[String] {
        self.experience
            .as_ref()
            .map(|e| e.tags.as_slice())
            .unwrap_or(&[])
    }
}

/// One row of `POST /api/memories`.
#[derive(Debug, Deserialize)]
struct ListMemoryItem {
    id: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    memory_type: String,
    #[serde(default)]
    tier: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ListMemoriesResponse {
    #[serde(default)]
    memories: Vec<ListMemoryItem>,
}

#[derive(Debug, Serialize)]
struct ListMemoriesRequest {
    user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<u32>,
}

#[derive(Debug, Serialize)]
struct UserIdRequest {
    user_id: String,
}

fn type_icon(memory_type: &str) -> &'static str {
    match memory_type {
        "Decision" => "📋",
        "Learning" => "💡",
        "Context" => "📁",
        "Pattern" => "🔄",
        "Error" => "⚠️",
        "Observation" => "👁️",
        "Discovery" => "🔍",
        "Task" => "✅",
        "CodeEdit" => "📝",
        "FileAccess" => "📄",
        "Search" => "🔎",
        "Command" => "⚡",
        "Conversation" => "💬",
        _ => "📦",
    }
}

/// First `max` characters with an ellipsis when the text is longer, as the
/// TypeScript `slice(0, n) + "..."` does; char-safe, never splits a code point.
fn preview(text: &str, max: usize) -> String {
    let mut out: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        out.push_str("...");
    }
    out
}

fn local_time(at: &DateTime<Utc>) -> String {
    at.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

fn mb(bytes: u64) -> String {
    format!("{:.2}", bytes as f64 / (1024.0 * 1024.0))
}

fn clamp_limit(raw: Option<u32>, default: u32) -> u32 {
    raw.unwrap_or(default).clamp(1, MAX_LIMIT)
}

fn text_error(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(message.into())])
}

// =============================================================================
// PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RecallByTagsParams {
    /// Tags to search for (returns memories matching ANY of these tags)
    pub tags: Vec<String>,
    /// Maximum number of results (default: 50)
    pub limit: Option<u32>,
    /// Also match tags veld derived from the content (names, keywords), not only the writer's own (default: false)
    pub include_derived: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ContextSummaryParams {
    /// Include recent decisions (default: true)
    pub include_decisions: Option<bool>,
    /// Include recent learnings (default: true)
    pub include_learnings: Option<bool>,
    /// Include project context (default: true)
    pub include_context: Option<bool>,
    /// Maximum items per category (default: 5)
    pub max_items: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ListMemoriesParams {
    /// Maximum number of results
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ForgetParams {
    /// The ID of the memory to delete
    pub id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct NoParams {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct BackupIdParams {
    /// The backup ID to verify
    pub backup_id: u32,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct BackupRestoreParams {
    /// The backup ID to restore (from backup_list)
    pub backup_id: u32,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct BackupPurgeParams {
    /// Number of backups to keep (default: 7)
    pub keep_count: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ConsolidationReportParams {
    /// Start of report period (ISO 8601 format). Defaults to 24 hours ago.
    pub since: Option<String>,
    /// End of report period (ISO 8601 format). Defaults to now.
    pub until: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct FactNarrativesParams {
    /// Maximum clusters to return. Clamped to [1, 50] server-side. Default: 20.
    pub limit: Option<u32>,
    /// Optional: narrow the candidate fact set to those related to a specific entity/topic. Substring match (not regex).
    pub entity_filter: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct FactsPreviewPurgeParams {
    /// Substring to match in fact content (case-insensitive, minimum 3 characters).
    pub pattern: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ReadMemoryParams {
    /// The memory ID (full UUID or short prefix like '5581cd02')
    pub memory_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct SeedProjectParams {
    /// Absolute path to the project directory to scan
    pub project_path: String,
    /// Maximum number of files to process (default: 50)
    pub max_files: Option<u32>,
}

// =============================================================================
// REQUEST AND RESPONSE BODIES
// =============================================================================

#[derive(Debug, Serialize)]
struct RecallByTagsRequest {
    user_id: String,
    tags: Vec<String>,
    limit: u32,
    include_derived: bool,
}

#[derive(Debug, Deserialize)]
struct RecallByTagsResponse {
    #[serde(default)]
    memories: Vec<WireMemory>,
}

#[derive(Debug, Serialize)]
struct ForgetByIdRequest {
    user_id: String,
    memory_id: String,
}

#[derive(Debug, Deserialize)]
struct DeleteMemoryResponse {
    id: String,
}

/// `GET /api/users/{id}/stats`: the server's `MemoryStats`, read loosely so a
/// field the server adds or drops does not break the tool.
#[derive(Debug, Deserialize)]
struct MemoryStatsWire {
    #[serde(default)]
    total_memories: u64,
    #[serde(default)]
    memory_types: HashMap<String, u64>,
    #[serde(default)]
    average_importance: Option<f32>,
    #[serde(default)]
    avg_importance: Option<f32>,
    #[serde(default)]
    graph_nodes: u64,
    #[serde(default)]
    graph_edges: u64,
    #[serde(default)]
    vector_index_count: Option<u64>,
    #[serde(default)]
    indexed_vectors: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct IndexIntegrityReport {
    total_storage: u64,
    total_indexed: u64,
    orphaned_count: u64,
    is_healthy: bool,
}

#[derive(Debug, Deserialize)]
struct RepairIndexResponse {
    success: bool,
    total_storage: u64,
    total_indexed: u64,
    repaired: u64,
    failed: u64,
    is_healthy: bool,
}

#[derive(Debug, Deserialize)]
struct BackupMetadata {
    backup_id: u32,
    created_at: DateTime<Utc>,
    backup_type: String,
    size_bytes: u64,
    checksum: String,
    memory_count: u64,
    #[serde(default)]
    secondary_stores: Vec<String>,
    #[serde(default)]
    secondary_size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct BackupResponse {
    success: bool,
    #[serde(default)]
    backup: Option<BackupMetadata>,
    #[serde(default)]
    message: String,
}

#[derive(Debug, Deserialize)]
struct ListBackupsResponse {
    #[serde(default)]
    backups: Vec<BackupMetadata>,
    #[serde(default)]
    count: u64,
}

#[derive(Debug, Serialize)]
struct BackupIdRequest {
    user_id: String,
    backup_id: u32,
}

#[derive(Debug, Deserialize)]
struct VerifyBackupResponse {
    is_valid: bool,
    #[serde(default)]
    message: String,
}

#[derive(Debug, Serialize)]
struct PurgeBackupsRequest {
    user_id: String,
    keep_count: u32,
}

#[derive(Debug, Deserialize)]
struct PurgeBackupsResponse {
    purged_count: u64,
}

#[derive(Debug, Deserialize)]
struct RestoreBackupResponse {
    success: bool,
    #[serde(default)]
    message: String,
    #[serde(default)]
    restored_stores: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ConsolidationReportRequest {
    user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    since: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    until: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReportPeriod {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct ConsolidationStats {
    #[serde(default)]
    total_memories: u64,
    #[serde(default)]
    memories_strengthened: u64,
    #[serde(default)]
    memories_decayed: u64,
    #[serde(default)]
    memories_at_risk: u64,
    #[serde(default)]
    edges_formed: u64,
    #[serde(default)]
    edges_strengthened: u64,
    #[serde(default)]
    edges_potentiated: u64,
    #[serde(default)]
    edges_pruned: u64,
    #[serde(default)]
    facts_extracted: u64,
    #[serde(default)]
    facts_reinforced: u64,
    #[serde(default)]
    maintenance_cycles: u64,
    #[serde(default)]
    total_maintenance_duration_ms: u64,
}

/// Only the event counts are rendered, so each list is read as opaque values.
#[derive(Debug, Deserialize)]
struct ConsolidationReport {
    period: ReportPeriod,
    #[serde(default)]
    strengthened_memories: Vec<serde_json::Value>,
    #[serde(default)]
    decayed_memories: Vec<serde_json::Value>,
    #[serde(default)]
    formed_associations: Vec<serde_json::Value>,
    #[serde(default)]
    strengthened_associations: Vec<serde_json::Value>,
    #[serde(default)]
    potentiated_associations: Vec<serde_json::Value>,
    #[serde(default)]
    pruned_associations: Vec<serde_json::Value>,
    #[serde(default)]
    extracted_facts: Vec<serde_json::Value>,
    #[serde(default)]
    reinforced_facts: Vec<serde_json::Value>,
    statistics: ConsolidationStats,
}

#[derive(Debug, Serialize)]
struct FactNarrativesRequest {
    user_id: String,
    limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    entity_filter: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CausalFactLink {
    from_fact: String,
    to_fact: String,
    relation: String,
}

#[derive(Debug, Deserialize)]
struct FactCluster {
    topic: String,
    #[serde(default)]
    facts: Vec<serde_json::Value>,
    narrative: String,
    avg_confidence: f32,
    total_support: u64,
    #[serde(default)]
    causal_chains: Vec<CausalFactLink>,
}

#[derive(Debug, Deserialize)]
struct FactNarrativesResponse {
    success: bool,
    #[serde(default)]
    clusters: Vec<FactCluster>,
    #[serde(default)]
    total_facts: u64,
    #[serde(default)]
    total_clusters: u64,
}

#[derive(Debug, Serialize)]
struct FactsPreviewPurgeRequest {
    user_id: String,
    pattern: String,
}

#[derive(Debug, Deserialize)]
struct FactsPreviewPurgeResponse {
    match_bucket: String,
    total_scanned: u64,
    #[serde(default)]
    audit_recorded: bool,
}

/// `GET /api/memory/{id}`: the flattened record plus its hierarchy.
#[derive(Debug, Deserialize)]
struct MemoryWithHierarchy {
    id: String,
    experience: ReadExperience,
    #[serde(default)]
    importance: Option<f32>,
    created_at: DateTime<Utc>,
    #[serde(default)]
    tier: Option<String>,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    children_ids: Vec<String>,
    #[serde(default)]
    children_count: u64,
}

#[derive(Debug, Deserialize)]
struct ReadExperience {
    #[serde(default)]
    content: String,
    #[serde(default)]
    experience_type: Option<String>,
    #[serde(default)]
    entities: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SeedRequest {
    user_id: String,
    project_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_files: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct SeedResponse {
    memories_created: u64,
    files_scanned: u64,
    project_name: String,
}

// =============================================================================
// TOOLS
// =============================================================================

#[tool_router(router = memory_tool_router, vis = "pub(crate)")]
impl VeldMcpServer {
    #[tool(
        description = "Find memories by tags, newest first. Returns memories whose writer gave them ANY of the provided tags, e.g. 'decision', 'learning', 'error', 'project:veld'."
    )]
    async fn recall_by_tags(
        &self,
        Parameters(params): Parameters<RecallByTagsParams>,
    ) -> Result<CallToolResult, McpError> {
        let tags: Vec<String> = params
            .tags
            .into_iter()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if tags.is_empty() {
            return Ok(text_error(
                "Error: 'tags' is required and must contain at least one tag",
            ));
        }
        let result: Result<RecallByTagsResponse> = self
            .client
            .post(
                "/api/recall/tags",
                &RecallByTagsRequest {
                    user_id: self.client.user_id.clone(),
                    tags: tags.clone(),
                    limit: clamp_limit(params.limit, 50),
                    include_derived: params.include_derived.unwrap_or(false),
                },
            )
            .await;
        let resp = result.map_err(mcp_error)?;
        let joined = tags.join(", ");
        if resp.memories.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No memories found matching tags: {joined}"
            ))]));
        }
        let mut out = format!(
            "Recall by tags: {joined}\nFound {} memories\n\n",
            resp.memories.len()
        );
        for (i, m) in resp.memories.iter().enumerate() {
            out.push_str(&format!(
                "{:>2}. {}\n    {} | tags: [{}] | {}\n\n",
                i + 1,
                preview(m.content(), 150),
                m.memory_type(),
                m.tags().join(", "),
                m.id
            ));
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(
        description = "Get a condensed summary of recent learnings, decisions, and context. Use this at the start of a session to quickly understand what you've learned before."
    )]
    async fn context_summary(
        &self,
        Parameters(params): Parameters<ContextSummaryParams>,
    ) -> Result<CallToolResult, McpError> {
        let include_decisions = params.include_decisions.unwrap_or(true);
        let include_learnings = params.include_learnings.unwrap_or(true);
        let include_context = params.include_context.unwrap_or(true);
        let max_items = params.max_items.unwrap_or(5).max(1) as usize;

        let result: Result<ListMemoriesResponse> = self
            .client
            .post(
                "/api/memories",
                &ListMemoriesRequest {
                    user_id: self.client.user_id.clone(),
                    limit: None,
                },
            )
            .await;
        let memories = result.map_err(mcp_error)?.memories;

        if memories.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "🌾 Context Summary\n{RULE_42}\nNo memories stored yet.\nStart remembering to build context!"
            ))]));
        }

        let of_type = |t: &str| -> Vec<&ListMemoryItem> {
            memories.iter().filter(|m| m.memory_type == t).collect()
        };
        let decisions = of_type("Decision");
        let learnings = of_type("Learning");
        let context = of_type("Context");
        let patterns = of_type("Pattern");
        let errors = of_type("Error");

        let mut out = format!(
            "🌾 Context Summary\n{RULE_52}\nTotal: {} memories │ 📋 {} │ 💡 {} │ 📁 {} │ 🔄 {} │ ⚠️ {}\n{RULE_52}\n\n",
            memories.len(),
            decisions.len(),
            learnings.len(),
            context.len(),
            patterns.len(),
            errors.len()
        );
        let mut section = |title: &str, items: &[&ListMemoryItem], cap: usize| {
            if items.is_empty() {
                return;
            }
            out.push_str(title);
            out.push('\n');
            for m in items.iter().take(cap) {
                out.push_str(&format!("   • {}\n", preview(&m.content, 70)));
            }
            out.push('\n');
        };
        if include_context {
            section("📁 PROJECT CONTEXT", &context, max_items);
        }
        if include_decisions {
            section("📋 DECISIONS", &decisions, max_items);
        }
        if include_learnings {
            section("💡 LEARNINGS", &learnings, max_items);
        }
        section("🔄 PATTERNS", &patterns, max_items);
        section("⚠️ ERRORS TO AVOID", &errors, max_items.min(3));

        if decisions.is_empty() && learnings.is_empty() && context.is_empty() {
            out.push_str(
                "ℹ️  Tip: Use types like Decision, Learning, Context when remembering\n   to build richer context summaries.",
            );
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(description = "List all stored memories")]
    async fn list_memories(
        &self,
        Parameters(params): Parameters<ListMemoriesParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = clamp_limit(params.limit, 20);
        let result: Result<ListMemoriesResponse> = self
            .client
            .post(
                "/api/memories",
                &ListMemoriesRequest {
                    user_id: self.client.user_id.clone(),
                    limit: Some(limit),
                },
            )
            .await;
        let mut memories = result.map_err(mcp_error)?.memories;
        memories.truncate(limit as usize);

        if memories.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "🌾 Memory List\n{RULE_42}\nNo memories stored yet."
            ))]));
        }

        let mut type_counts: Vec<(String, usize)> = Vec::new();
        for m in &memories {
            match type_counts.iter_mut().find(|(t, _)| *t == m.memory_type) {
                Some((_, n)) => *n += 1,
                None => type_counts.push((m.memory_type.clone(), 1)),
            }
        }
        let types = type_counts
            .iter()
            .map(|(t, n)| format!("{t}({n})"))
            .collect::<Vec<_>>()
            .join(" │ ");

        let mut out = format!(
            "🌾 Memory List\n{RULE_52}\nShowing {} memories\nTypes: {types}\n{RULE_52}\n\n",
            memories.len()
        );
        for (i, m) in memories.iter().enumerate() {
            let tier = m
                .tier
                .as_deref()
                .map(|t| format!(" │ {t}"))
                .unwrap_or_default();
            out.push_str(&format!(
                "{:>2}. {} {}\n    ┗━ {}{tier} │ {}\n",
                i + 1,
                type_icon(&m.memory_type),
                preview(&m.content, 150),
                m.memory_type,
                m.id
            ));
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(description = "Delete a specific memory by ID")]
    async fn forget(
        &self,
        Parameters(params): Parameters<ForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        let id = params.id.trim().to_string();
        if id.is_empty() {
            return Ok(text_error("Error: 'id' is required"));
        }
        let result: Result<DeleteMemoryResponse> = self
            .client
            .post(
                "/api/forget",
                &ForgetByIdRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: id,
                },
            )
            .await;
        let resp = result.map_err(mcp_error)?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "🌾 Memory Deleted\n{RULE_30}\n✓ Removed: {}",
            resp.id
        ))]))
    }

    #[tool(description = "Get statistics about stored memories")]
    async fn memory_stats(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let path = format!(
            "/api/users/{}/stats",
            url_encode_component_shared(&self.client.user_id)
        );
        let result: Result<MemoryStatsWire> = self.client.get(&path).await;
        let s = result.map_err(mcp_error)?;
        let indexed = s.vector_index_count.or(s.indexed_vectors).unwrap_or(0);
        let avg = s.average_importance.or(s.avg_importance).unwrap_or(0.0);

        let mut out = format!(
            "🌾 Memory Statistics\n{RULE_52}\nTotal Memories: {}\nGraph: {} nodes │ {} edges\nIndexed Vectors: {indexed}\nAvg Importance: {avg:.2}\n{RULE_52}\n",
            s.total_memories, s.graph_nodes, s.graph_edges
        );
        if !s.memory_types.is_empty() {
            out.push_str("\nBy Type:\n");
            let mut types: Vec<_> = s.memory_types.iter().collect();
            types.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let total = s.total_memories.max(1) as f64;
            for (t, n) in types {
                let bar_len = ((*n as f64 / total) * 20.0).round().min(20.0) as usize;
                out.push_str(&format!(
                    "   {} {:<12} {} {n}\n",
                    type_icon(t),
                    t,
                    "█".repeat(bar_len)
                ));
            }
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(
        description = "Verify vector index integrity - diagnose orphaned memories that are stored but not searchable. Returns health status and count of orphaned memories."
    )]
    async fn verify_index(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<IndexIntegrityReport> = self
            .client
            .post(
                "/api/index/verify",
                &UserIdRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let (icon, health) = if r.is_healthy {
            ("✓", "HEALTHY")
        } else {
            ("⚠️", "UNHEALTHY")
        };
        let mut out = format!(
            "🌾 Index Verification\n{RULE_42}\n{icon} {health}\n{RULE_42}\nStorage: {} memories\nIndexed: {} vectors\nOrphaned: {}\n",
            r.total_storage, r.total_indexed, r.orphaned_count
        );
        if r.orphaned_count > 0 {
            out.push_str(&format!(
                "{RULE_42}\n⚠️ Run repair_index to fix orphaned memories"
            ));
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "Repair vector index by re-indexing orphaned memories. Use this when verify_index shows unhealthy status. Returns count of repaired memories."
    )]
    async fn repair_index(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RepairIndexResponse> = self
            .client
            .post(
                "/api/index/repair",
                &UserIdRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let icon = if r.is_healthy { "✓" } else { "⚠️" };
        let status = if r.success { "SUCCESS" } else { "PARTIAL" };
        let mut out = format!(
            "🌾 Index Repair\n{RULE_42}\n{icon} {status}\n{RULE_42}\nStorage: {} memories\nIndexed: {} vectors\nRepaired: {}\nFailed: {}\n",
            r.total_storage, r.total_indexed, r.repaired, r.failed
        );
        if r.failed > 0 {
            out.push_str(&format!("{RULE_42}\n⚠️ {} could not be repaired", r.failed));
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "Create a backup of all memories. Returns backup metadata including ID, size, and checksum. Backups are stored locally and can be restored later."
    )]
    async fn backup_create(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<BackupResponse> = self
            .client
            .post(
                "/api/backup/create",
                &UserIdRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let mut out = format!("🌾 Backup Created\n{RULE_42}\n");
        match (r.success, r.backup) {
            (true, Some(b)) => {
                out.push_str(&format!(
                    "✓ Backup ID: {}\n{RULE_42}\nType: {}\nMemories: {}\nSize: {} MB\n",
                    b.backup_id,
                    b.backup_type,
                    b.memory_count,
                    mb(b.size_bytes)
                ));
                if !b.secondary_stores.is_empty() {
                    out.push_str(&format!(
                        "Secondary stores: {} ({} MB)\n  Includes: {}\n",
                        b.secondary_stores.len(),
                        mb(b.secondary_size_bytes),
                        b.secondary_stores.join(", ")
                    ));
                }
                out.push_str(&format!(
                    "Checksum: {}...\nCreated: {}\n",
                    b.checksum.chars().take(16).collect::<String>(),
                    local_time(&b.created_at)
                ));
            }
            _ => {
                let msg = if r.message.is_empty() {
                    "Unknown backup creation error"
                } else {
                    r.message.as_str()
                };
                out.push_str(&format!("✗ Failed: {msg}\n"));
            }
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "List all available backups for this user. Returns backup history with IDs, timestamps, and sizes."
    )]
    async fn backup_list(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ListBackupsResponse> = self
            .client
            .post(
                "/api/backups",
                &UserIdRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let mut out = format!("🌾 Available Backups\n{RULE_52}\n");
        if r.backups.is_empty() {
            out.push_str("No backups available.\nUse backup_create to create your first backup.");
        } else {
            out.push_str(&format!("Found: {} backup(s)\n{RULE_52}\n\n", r.count));
            for b in &r.backups {
                out.push_str(&format!(
                    "📦 Backup #{}\n   Type: {} │ Memories: {} │ Size: {} MB\n   Created: {}\n\n",
                    b.backup_id,
                    b.backup_type,
                    b.memory_count,
                    mb(b.size_bytes),
                    local_time(&b.created_at)
                ));
            }
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(
        description = "Verify backup integrity using SHA-256 checksum. Use to check if a backup is corrupted before restoring."
    )]
    async fn backup_verify(
        &self,
        Parameters(params): Parameters<BackupIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<VerifyBackupResponse> = self
            .client
            .post(
                "/api/backup/verify",
                &BackupIdRequest {
                    user_id: self.client.user_id.clone(),
                    backup_id: params.backup_id,
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let (icon, status) = if r.is_valid {
            ("✓", "VALID")
        } else {
            ("✗", "INVALID")
        };
        let message = if r.message.is_empty() {
            "No verification details provided"
        } else {
            r.message.as_str()
        };
        Ok(CallToolResult::success(vec![Content::text(format!(
            "🌾 Backup Verification\n{RULE_42}\n{icon} Backup #{}: {status}\n{RULE_42}\n{message}",
            params.backup_id
        ))]))
    }

    #[tool(
        description = "Purge old backups, keeping only the most recent N. Useful for managing disk space."
    )]
    async fn backup_purge(
        &self,
        Parameters(params): Parameters<BackupPurgeParams>,
    ) -> Result<CallToolResult, McpError> {
        let keep_count = params.keep_count.unwrap_or(7);
        let result: Result<PurgeBackupsResponse> = self
            .client
            .post(
                "/api/backups/purge",
                &PurgeBackupsRequest {
                    user_id: self.client.user_id.clone(),
                    keep_count,
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let body = if r.purged_count == 0 {
            format!("No backups purged (keeping {keep_count}, none exceeded limit)")
        } else {
            format!(
                "✓ Purged {} old backup(s)\nKept {keep_count} most recent backup(s)",
                r.purged_count
            )
        };
        Ok(CallToolResult::success(vec![Content::text(format!(
            "🌾 Backup Purge\n{RULE_42}\n{body}"
        ))]))
    }

    #[tool(
        description = "Restore a previously created backup by ID. This replaces all current data for the user with the backup contents. Server restart is recommended after restore."
    )]
    async fn backup_restore(
        &self,
        Parameters(params): Parameters<BackupRestoreParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RestoreBackupResponse> = self
            .client
            .post(
                "/api/backup/restore",
                &BackupIdRequest {
                    user_id: self.client.user_id.clone(),
                    backup_id: params.backup_id,
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let mut out = format!("🔄 Backup Restore\n{RULE_42}\n");
        if r.success {
            out.push_str(&format!(
                "✓ Backup #{} restored successfully\n",
                params.backup_id
            ));
            if !r.restored_stores.is_empty() {
                out.push_str(&format!(
                    "Restored stores: {}\n",
                    r.restored_stores.join(", ")
                ));
            }
            let message = if r.message.is_empty() {
                "Restore completed with no additional details"
            } else {
                r.message.as_str()
            };
            out.push_str(&format!("\n⚠️ {message}"));
        } else {
            let message = if r.message.is_empty() {
                "Unknown restore error"
            } else {
                r.message.as_str()
            };
            out.push_str(&format!("✗ Restore failed: {message}"));
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "Get a report of what the memory system has been learning. Shows memory strengthening/decay events, edge formation, fact extraction, and maintenance cycles. Use this to understand how your memories are evolving."
    )]
    async fn consolidation_report(
        &self,
        Parameters(params): Parameters<ConsolidationReportParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ConsolidationReport> = self
            .client
            .post(
                "/api/consolidation/report",
                &ConsolidationReportRequest {
                    user_id: self.client.user_id.clone(),
                    since: params.since,
                    until: params.until,
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let s = &r.statistics;
        let event_count = r.strengthened_memories.len()
            + r.decayed_memories.len()
            + r.formed_associations.len()
            + r.strengthened_associations.len()
            + r.potentiated_associations.len()
            + r.pruned_associations.len()
            + r.extracted_facts.len()
            + r.reinforced_facts.len();

        let mut out = format!(
            "🌾 Consolidation Report\n{RULE_52}\nPeriod: {} → {}\nEvents: {event_count} │ Memories: {}\n{RULE_52}\n\n",
            local_time(&r.period.start),
            local_time(&r.period.end),
            s.total_memories
        );
        if s.memories_strengthened > 0 || s.memories_decayed > 0 || s.memories_at_risk > 0 {
            out.push_str("🧠 MEMORY DYNAMICS\n");
            if s.memories_strengthened > 0 {
                out.push_str(&format!("   ↑ {} strengthened\n", s.memories_strengthened));
            }
            if s.memories_decayed > 0 {
                out.push_str(&format!("   ↓ {} decayed\n", s.memories_decayed));
            }
            if s.memories_at_risk > 0 {
                out.push_str(&format!("   ⚠️ {} at risk\n", s.memories_at_risk));
            }
            out.push('\n');
        }
        if s.edges_formed > 0
            || s.edges_strengthened > 0
            || s.edges_potentiated > 0
            || s.edges_pruned > 0
        {
            out.push_str("🔗 ASSOCIATIONS (Hebbian)\n");
            if s.edges_formed > 0 {
                out.push_str(&format!("   + {} formed\n", s.edges_formed));
            }
            if s.edges_strengthened > 0 {
                out.push_str(&format!("   ↑ {} strengthened\n", s.edges_strengthened));
            }
            if s.edges_potentiated > 0 {
                out.push_str(&format!("   ★ {} permanent (LTP)\n", s.edges_potentiated));
            }
            if s.edges_pruned > 0 {
                out.push_str(&format!("   ✂ {} pruned\n", s.edges_pruned));
            }
            out.push('\n');
        }
        if s.facts_extracted > 0 || s.facts_reinforced > 0 {
            out.push_str("📚 FACTS\n");
            if s.facts_extracted > 0 {
                out.push_str(&format!("   + {} extracted\n", s.facts_extracted));
            }
            if s.facts_reinforced > 0 {
                out.push_str(&format!("   ↑ {} reinforced\n", s.facts_reinforced));
            }
            out.push('\n');
        }
        if s.maintenance_cycles > 0 {
            out.push_str(&format!(
                "⚙️ MAINTENANCE: {} cycles ({:.2}s)\n",
                s.maintenance_cycles,
                s.total_maintenance_duration_ms as f64 / 1000.0
            ));
        }
        if event_count == 0 {
            out.push_str(
                "ℹ️ No consolidation activity in this period.\n   Store and access memories to trigger learning.",
            );
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(
        description = "Get synthesized narratives from accumulated semantic facts, clustered by shared entities with confidence levels and heuristic causal chains. Shows what the system has learned, organized into coherent themes. Read-only; never modifies stored facts. Purged and time-invalidated facts are automatically excluded."
    )]
    async fn fact_narratives(
        &self,
        Parameters(params): Parameters<FactNarrativesParams>,
    ) -> Result<CallToolResult, McpError> {
        let entity_filter = params
            .entity_filter
            .map(|f| f.trim().to_string())
            .filter(|f| !f.is_empty());
        let result: Result<FactNarrativesResponse> = self
            .client
            .post(
                "/api/facts/narratives",
                &FactNarrativesRequest {
                    user_id: self.client.user_id.clone(),
                    limit: params.limit.unwrap_or(20).clamp(1, 50),
                    entity_filter: entity_filter.clone(),
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        if !r.success || r.total_clusters == 0 {
            let scope = entity_filter
                .map(|f| format!(" related to \"{f}\""))
                .unwrap_or_default();
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "ℹ️ No fact narratives found{scope}. Store more memories to build the fact corpus."
            ))]));
        }
        let mut out = format!(
            "📚 FACT NARRATIVES ({} clusters from {} facts)\n\n",
            r.total_clusters, r.total_facts
        );
        for c in &r.clusters {
            out.push_str(&format!(
                "── {} ({} facts · avg confidence {:.0}% · support {}) ──\n{}\n",
                c.topic,
                c.facts.len(),
                c.avg_confidence * 100.0,
                c.total_support,
                c.narrative
            ));
            if !c.causal_chains.is_empty() {
                out.push_str("\nCausal links:\n");
                for link in &c.causal_chains {
                    out.push_str(&format!(
                        "  • {} → [{}] → {}\n",
                        link.from_fact, link.relation, link.to_fact
                    ));
                }
            }
            out.push('\n');
        }
        Ok(CallToolResult::success(vec![Content::text(
            out.trim_end().to_string(),
        )]))
    }

    #[tool(
        description = "Preview the blast radius of a fact-purge pattern WITHOUT deleting anything. Returns a bucketed match count (none / few / some / many) — not an exact count — so the preview cannot become an oracle for fact existence. Use this BEFORE any destructive purge: agents should always preview-check a pattern, surface the bucket to the user, and only escalate to the destructive `/api/facts/purge` route after explicit confirmation. Pattern is substring match (not regex), case-insensitive, minimum 3 characters."
    )]
    async fn facts_preview_purge(
        &self,
        Parameters(params): Parameters<FactsPreviewPurgeParams>,
    ) -> Result<CallToolResult, McpError> {
        let pattern = params.pattern.trim().to_string();
        if pattern.chars().count() < 3 {
            return Ok(text_error("Error: 'pattern' must be at least 3 characters"));
        }
        let result: Result<FactsPreviewPurgeResponse> = self
            .client
            .post(
                "/api/facts/preview-purge",
                &FactsPreviewPurgeRequest {
                    user_id: self.client.user_id.clone(),
                    pattern,
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        let description = match r.match_bucket.as_str() {
            "none" => "ZERO facts match — pattern is safe to purge (nothing would be removed).",
            "few" => "1-5 facts match — small, targeted purge.",
            "some" => "6-50 facts match — moderate purge; review pattern before escalating.",
            "many" => "51+ facts match — LARGE purge; refine the pattern before invoking the destructive endpoint.",
            _ => "",
        };
        let mut out = format!(
            "🔍 PURGE PREVIEW (dry-run only — nothing deleted)\n\nMatch bucket: {}\n{description}\n\nScanned {} active facts.\n",
            r.match_bucket.to_uppercase(),
            r.total_scanned
        );
        if !r.audit_recorded {
            out.push_str("⚠️ Audit entry did NOT persist — check server logs.\n");
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "Get current token usage status for this session. Returns tokens used, budget remaining, and percentage consumed. Use this to check context window health."
    )]
    async fn token_status(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let status = self.tokens.status();
        Ok(CallToolResult::success(vec![Content::text(
            format_token_status(&status, self.tokens.session_minutes()),
        )]))
    }

    #[tool(
        description = "Reset the token counter for a new session. Call this when starting a new conversation or after context has been compressed/summarized."
    )]
    async fn reset_token_session(
        &self,
        Parameters(_params): Parameters<NoParams>,
    ) -> Result<CallToolResult, McpError> {
        let previous = self.tokens.session_tokens();
        self.tokens.reset();
        let budget = self.tokens.status().budget;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "🌾 Token Session Reset\n{RULE_30}\nPrevious: {} tokens\nCurrent: 0 tokens\nBudget: {} tokens\n{RULE_30}\n✓ Counter cleared",
            thousands(previous),
            thousands(budget)
        ))]))
    }

    #[tool(
        description = "Read the FULL content of a specific memory by ID. Use this when you need to see the complete text of a memory that was truncated in search results."
    )]
    async fn read_memory(
        &self,
        Parameters(params): Parameters<ReadMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let memory_id = params.memory_id.trim().to_string();
        if memory_id.is_empty() {
            return Ok(text_error(
                "Error: 'memory_id' is required. Pass the full UUID or 8+ character prefix from recall results.",
            ));
        }
        let path = format!(
            "/api/memory/{}?user_id={}",
            url_encode_component_shared(&memory_id),
            url_encode_component_shared(&self.client.user_id)
        );
        let result: Result<MemoryWithHierarchy> = self.client.get(&path).await;
        let m = match result {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[Memory] Failed to fetch memory {memory_id}: {e}");
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Memory not found: {memory_id}"
                ))]));
            }
        };
        let tags = if m.experience.entities.is_empty() {
            "none".to_string()
        } else {
            m.experience.entities.join(", ")
        };
        let importance = m
            .importance
            .map(|i| format!("{:.0}%", i * 100.0))
            .unwrap_or_else(|| "n/a".to_string());
        let mut out = format!(
            "Memory: {}\n{RULE_52}\nType: {} | Tags: {tags}\nTier: {} | Created: {} | Importance: {importance}\n",
            m.id,
            m.experience.experience_type.as_deref().unwrap_or("Unknown"),
            m.tier.as_deref().unwrap_or("Unknown"),
            local_time(&m.created_at)
        );
        if let Some(parent) = &m.parent_id {
            out.push_str(&format!("Parent: {parent}\n"));
        }
        if m.children_count > 0 {
            out.push_str(&format!(
                "Children: {} ({})\n",
                m.children_count,
                m.children_ids.join(", ")
            ));
        }
        out.push_str(&format!("{RULE_52}\n\n{}", m.experience.content));
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "Scan a project directory and create foundational memories from config files, README, and source code. Enables rapid bootstrapping of memory for a new project. Idempotent: re-running deletes old seed memories first."
    )]
    async fn seed_project(
        &self,
        Parameters(params): Parameters<SeedProjectParams>,
    ) -> Result<CallToolResult, McpError> {
        let project_path = params.project_path.trim().to_string();
        if project_path.is_empty() {
            return Ok(text_error(
                "Error: 'project_path' is required and must be an absolute path to a directory",
            ));
        }
        let result: Result<SeedResponse> = self
            .client
            .post(
                "/api/seed",
                &SeedRequest {
                    user_id: self.client.user_id.clone(),
                    project_path,
                    max_files: params.max_files,
                },
            )
            .await;
        let r = result.map_err(mcp_error)?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Seeded project \"{}\": {} memories created from {} files scanned.",
            r.project_name, r.memories_created, r.files_scanned
        ))]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_four_chars_rounded_up() {
        assert_eq!(TokenTracker::estimate_tokens(""), 0);
        assert_eq!(TokenTracker::estimate_tokens("abcd"), 1);
        assert_eq!(TokenTracker::estimate_tokens("abcde"), 2);
    }

    #[test]
    fn tracker_accumulates_and_alerts_at_threshold() {
        let t = TokenTracker::new(100, 0.9);
        t.track("recall", &"x".repeat(356));
        assert!(t.status().alert.is_none(), "89 of 100 is under the 90% threshold");
        t.track("recall", &"x".repeat(8));
        let s = t.status();
        assert_eq!(s.tokens, 91);
        assert_eq!(s.alert.as_deref(), Some("context_90_percent"));
        t.reset();
        assert_eq!(t.session_tokens(), 0);
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn progress_bar_is_twenty_cells() {
        assert_eq!(progress_bar(0), "░".repeat(20));
        assert_eq!(progress_bar(100), "█".repeat(20));
        assert_eq!(
            progress_bar(50),
            format!("{}{}", "█".repeat(10), "░".repeat(10))
        );
        assert_eq!(progress_bar(1).chars().count(), 20);
    }

    #[test]
    fn preview_never_splits_a_code_point() {
        assert_eq!(preview("héllo", 2), "hé...");
        assert_eq!(preview("hi", 5), "hi");
    }
}
