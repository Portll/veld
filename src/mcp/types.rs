//! Request, response, and parameter types for the MCP server.

use rmcp::schemars;
use serde::{Deserialize, Serialize};

// =============================================================================
// MCP TOOL PARAMETER TYPES (schemars-generated JSON schema)
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RememberParams {
    /// The content to remember
    pub content: String,
    /// Type of memory (Observation, Decision, Learning, etc.)
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    /// Optional tags for categorization
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RecallParams {
    /// Natural language search query
    pub query: String,
    /// Maximum number of results (default: 5)
    pub limit: Option<u32>,
    /// Retrieval mode: semantic, associative, or hybrid
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ProactiveContextParams {
    /// Current conversation context
    pub context: String,
    /// Maximum memories to surface (default: 5)
    pub max_results: Option<u32>,
    /// Auto-store context for feedback (default: true)
    pub auto_ingest: Option<bool>,
}

// =============================================================================
// LINEAGE MCP TOOL PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct LineageTraceParams {
    /// Memory ID to trace lineage from
    pub memory_id: String,
    /// Direction: "backward" (find causes), "forward" (find effects), "both"
    pub direction: Option<String>,
    /// Maximum depth to traverse (default: 10)
    pub max_depth: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct LineageConfirmParams {
    /// ID of the inferred edge to confirm
    pub edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct LineageRejectParams {
    /// ID of the inferred edge to reject
    pub edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct LineageLinkParams {
    /// Source memory ID (the cause/origin)
    pub from_memory_id: String,
    /// Target memory ID (the effect/result)
    pub to_memory_id: String,
    /// Relation type: Caused, ResolvedBy, InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo
    pub relation: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct LineageStatsParams {
    /// Optional - leave empty to get stats for current user
    #[serde(default)]
    pub _placeholder: Option<String>,
}

// =============================================================================
// AGENT SELF-PAGING (TIER + PIN) MCP TOOL PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct PinMemoryParams {
    /// Memory ID (UUID string) to pin or unpin
    pub memory_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct MoveTierParams {
    /// Memory ID (UUID string) to move
    pub memory_id: String,
    /// Target tier: "working" (hottest), "session", "longterm", "archive" (coldest)
    pub target_tier: String,
}

#[derive(Serialize)]
pub(crate) struct AnchorRequest {
    pub user_id: String,
    pub memory_id: String,
    pub anchor: bool,
}

#[derive(Deserialize)]
pub(crate) struct AnchorResponse {
    pub success: bool,
    pub memory_id: String,
    pub anchored: bool,
}

#[derive(Serialize)]
pub(crate) struct TierMoveRequest {
    pub user_id: String,
    pub memory_id: String,
    pub target_tier: String,
}

#[derive(Deserialize)]
pub(crate) struct TierMoveResponse {
    pub success: bool,
    pub memory_id: String,
    pub previous_tier: String,
    pub current_tier: String,
}

// =============================================================================
// CLOSED-LOOP FEEDBACK MCP TOOL PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ReinforceParams {
    /// Memory IDs (UUID strings) that were surfaced. Mark them with the
    /// outcome — the feedback store updates each memory's momentum so future
    /// rankings prefer helpful memories and suppress misleading ones.
    pub memory_ids: Vec<String>,
    /// "helpful" — memories that helped solve the task (boost score)
    /// "misleading" — memories that misled or wasted time (suppress score)
    /// "neutral" — neither obviously helped nor hurt (mild access bump)
    pub outcome: String,
}

#[derive(Serialize)]
pub(crate) struct ReinforceRequest {
    pub user_id: String,
    pub ids: Vec<String>,
    pub outcome: String,
}

#[derive(Deserialize)]
pub(crate) struct ReinforceResponse {
    pub memories_processed: usize,
    pub associations_strengthened: usize,
    pub importance_boosts: usize,
    pub importance_decays: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct GapMapperParams {
    #[schemars(description = "Filter function: 'centroid_distance' (default), 'density', 'eccentricity', 'neighbor_distance', or 'embedding_pc1'")]
    pub filter: Option<String>,
    #[schemars(description = "Number of intervals in the cover (default 10)")]
    pub num_intervals: Option<usize>,
    #[schemars(description = "Overlap percentage between intervals, 0.0-1.0 (default 0.3)")]
    pub overlap: Option<f32>,
}

// =============================================================================
// API REQUEST TYPES (sent to the veld HTTP server)
// =============================================================================

#[derive(Serialize)]
pub(crate) struct ProactiveContextRequest {
    pub user_id: String,
    pub context: String,
    pub max_results: u32,
    pub auto_ingest: bool,
}

#[derive(Serialize)]
pub(crate) struct ListTodosRequest {
    pub user_id: String,
    pub status: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct RememberRequest {
    pub user_id: String,
    pub content: String,
    pub memory_type: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Serialize)]
pub(crate) struct RecallRequest {
    pub user_id: String,
    pub query: String,
    pub limit: Option<u32>,
    pub mode: Option<String>,
}

// Lineage API request types
#[derive(Serialize)]
pub(crate) struct LineageTraceRequest {
    pub user_id: String,
    pub memory_id: String,
    pub direction: String,
    pub max_depth: u32,
}

#[derive(Serialize)]
pub(crate) struct LineageEdgeRequest {
    pub user_id: String,
    pub edge_id: String,
}

#[derive(Serialize)]
pub(crate) struct LineageAddEdgeRequest {
    pub user_id: String,
    pub from_memory_id: String,
    pub to_memory_id: String,
    pub relation: String,
}

#[derive(Serialize)]
pub(crate) struct LineageStatsRequest {
    pub user_id: String,
}

// Gap analysis request types
#[derive(Serialize)]
pub(crate) struct GapAnalyzeRequest {
    pub user_id: String,
}

#[derive(Serialize)]
pub(crate) struct GapMapperRequest {
    pub user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_intervals: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap: Option<f32>,
}

// =============================================================================
// API RESPONSE TYPES (received from the veld HTTP server)
// =============================================================================

#[derive(Deserialize)]
pub(crate) struct ProactiveContextResponse {
    pub memories: Vec<SurfacedMemory>,
}

#[derive(Deserialize)]
pub(crate) struct SurfacedMemory {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub relevance_score: f32,
}

#[derive(Deserialize)]
pub(crate) struct ListTodosResponse {
    pub todos: Vec<Todo>,
}

#[derive(Deserialize)]
pub(crate) struct Todo {
    #[allow(dead_code)]
    pub id: String,
    pub content: String,
    pub status: String,
    pub priority: Option<String>,
    #[allow(dead_code)]
    pub project: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct RememberResponse {
    pub id: String,
    pub message: String,
}

#[derive(Deserialize)]
pub(crate) struct RecallResponse {
    pub memories: Vec<RecalledMemory>,
}

#[derive(Deserialize)]
pub(crate) struct RecalledMemory {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub similarity: f32,
    #[allow(dead_code)]
    pub tags: Vec<String>,
}

// Lineage API response types
#[derive(Deserialize)]
pub(crate) struct LineageTraceResponse {
    pub root: String,
    pub direction: String,
    pub edges: Vec<LineageEdgeInfo>,
    pub path: Vec<String>,
    pub depth: usize,
}

#[derive(Deserialize)]
pub(crate) struct LineageEdgeInfo {
    #[allow(dead_code)]
    pub id: String,
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
    pub source: String,
}

#[derive(Deserialize)]
pub(crate) struct LineageConfirmResponse {
    pub message: String,
    pub edge_id: String,
}

#[derive(Deserialize)]
pub(crate) struct LineageRejectResponse {
    pub message: String,
    #[allow(dead_code)]
    pub deleted: bool,
}

#[derive(Deserialize)]
pub(crate) struct LineageAddResponse {
    pub message: String,
    pub edge_id: String,
}

#[derive(Deserialize)]
pub(crate) struct LineageStatsResponse {
    pub total_edges: usize,
    pub inferred_edges: usize,
    pub confirmed_edges: usize,
    pub explicit_edges: usize,
    pub total_branches: usize,
    pub active_branches: usize,
    pub edges_by_relation: std::collections::HashMap<String, usize>,
    pub avg_confidence: f32,
}

// Gap analysis response types
#[derive(Deserialize)]
pub(crate) struct GapAnalyzeResponse {
    pub gaps: Vec<GapSummaryMcp>,
    pub type_counts: std::collections::HashMap<String, usize>,
    pub duration_ms: u64,
}

#[derive(Deserialize)]
pub(crate) struct GapSummaryMcp {
    pub id: String,
    pub gap_type: String,
    pub confidence: f32,
    pub impact_score: f32,
    pub entity_names: Vec<String>,
}

#[derive(Deserialize)]
pub(crate) struct GapMapperResponse {
    pub nodes: Vec<GapMapperNode>,
    #[allow(dead_code)]
    pub edges: Vec<GapMapperEdge>,
    pub num_components: usize,
    pub num_loops: usize,
    pub flare_count: usize,
    pub branch_count: usize,
    pub filter: String,
    pub stats: GapMapperStats,
}

#[derive(Deserialize)]
pub(crate) struct GapMapperNode {
    pub id: usize,
    pub member_names: Vec<String>,
    pub size: usize,
    pub avg_filter_value: f32,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub(crate) struct GapMapperEdge {
    pub from: usize,
    pub to: usize,
    pub weight: usize,
}

#[derive(Deserialize)]
pub(crate) struct GapMapperStats {
    pub entity_count: usize,
    #[allow(dead_code)]
    pub interval_count: usize,
    pub cluster_count: usize,
    pub edge_count: usize,
    pub duration_ms: u64,
}

// =============================================================================
// HOOK OUTPUT TYPES
// =============================================================================

#[derive(Serialize)]
pub(crate) struct HookOutput {
    #[serde(rename = "hookSpecificOutput")]
    pub hook_specific_output: HookSpecificOutput,
}

#[derive(Serialize)]
pub(crate) struct HookSpecificOutput {
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    #[serde(rename = "additionalContext")]
    pub additional_context: String,
}
