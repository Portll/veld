//! MCP tools over todos, subtasks, comments, projects and reminders: the work-tracking
//! half of the TypeScript server's tool set. Tool and parameter names are part of the
//! MCP surface and do not change.

use anyhow::Result;
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars, tool, tool_router, ErrorData as McpError,
};
use serde::{Deserialize, Serialize};

use super::tools::{mcp_error, url_encode_component_shared as encode};
use super::VeldMcpServer;

/// The TypeScript server's cap on content fields, mirrored so both servers refuse the same input.
const MAX_CONTENT_LENGTH: usize = 100_000;
/// The TypeScript server's cap on list sizes.
const MAX_LIMIT: u64 = 250;

// =============================================================================
// ENUMS SHARED BY THE SCHEMAS (serde names are the wire and schema strings)
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReminderTriggerType {
    Time,
    Duration,
    Context,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReminderStatusFilter {
    Pending,
    Triggered,
    Dismissed,
    All,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NewTodoStatus {
    Backlog,
    Todo,
    InProgress,
    Blocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TodoStatus {
    Backlog,
    Todo,
    InProgress,
    Blocked,
    Done,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TodoPriority {
    Urgent,
    High,
    Medium,
    Low,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TodoPriorityFilter {
    Urgent,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DueFilter {
    Today,
    Overdue,
    ThisWeek,
    All,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Recurrence {
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReorderDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CommentType {
    Comment,
    Progress,
    Resolution,
    Activity,
}

// =============================================================================
// REMINDER PARAMETERS, REQUESTS AND RESPONSES
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct SetReminderParams {
    /// What to remember/remind about
    pub content: String,
    /// When to trigger: 'time' (at specific ISO timestamp), 'duration' (after N seconds), 'context' (when keywords match)
    pub trigger_type: ReminderTriggerType,
    /// ISO 8601 timestamp for 'time' trigger (e.g., '2025-12-23T18:00:00Z')
    pub trigger_at: Option<String>,
    /// Seconds from now for 'duration' trigger
    pub after_seconds: Option<u64>,
    /// Keywords for 'context' trigger - reminder surfaces when any keyword appears
    pub keywords: Option<Vec<String>>,
    /// Priority 1-5 (5 = highest, default: 3)
    pub priority: Option<u8>,
    /// Optional tags for categorization
    pub tags: Option<Vec<String>>,
    /// Semantic similarity threshold for 'context' trigger (0.0-1.0, default: 0.7). Lower values match more broadly, higher values require closer semantic match.
    pub threshold: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ListRemindersParams {
    /// Filter by status (default: pending)
    pub status: Option<ReminderStatusFilter>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct DismissReminderParams {
    /// ID of the reminder to dismiss
    pub reminder_id: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReminderTrigger {
    Time {
        at: String,
    },
    Duration {
        after_seconds: u64,
    },
    Context {
        keywords: Vec<String>,
        threshold: f32,
    },
}

#[derive(Serialize)]
struct CreateReminderRequest {
    user_id: String,
    content: String,
    trigger: ReminderTrigger,
    priority: u8,
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct CreateReminderResponse {
    id: String,
    #[serde(default)]
    due_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Serialize)]
struct ListRemindersRequest {
    user_id: String,
    status: Option<String>,
}

#[derive(Deserialize)]
struct ReminderItem {
    id: String,
    content: String,
    trigger_type: String,
    status: String,
    #[serde(default)]
    due_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    priority: u8,
    #[serde(default)]
    overdue_seconds: Option<i64>,
}

#[derive(Deserialize)]
struct ListRemindersResponse {
    reminders: Vec<ReminderItem>,
    count: usize,
}

#[derive(Serialize)]
struct UserOnlyRequest {
    user_id: String,
}

#[derive(Deserialize)]
struct ActionResponse {
    success: bool,
    #[serde(default)]
    message: String,
}

// =============================================================================
// TODO, PROJECT AND COMMENT PARAMETERS, REQUESTS AND RESPONSES
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct AddTodoParams {
    /// What needs to be done
    pub content: String,
    /// Initial status (default: todo)
    pub status: Option<NewTodoStatus>,
    /// Priority level (default: medium)
    pub priority: Option<TodoPriority>,
    /// Project name (created if doesn't exist)
    pub project: Option<String>,
    /// Contexts like @computer, @phone, @errands
    pub contexts: Option<Vec<String>>,
    /// Due date - ISO format or 'today', 'tomorrow', 'monday', etc.
    pub due_date: Option<String>,
    /// Optional tags for categorization
    pub tags: Option<Vec<String>>,
    /// Who/what you're waiting on (sets status to blocked)
    pub blocked_on: Option<String>,
    /// Additional notes
    pub notes: Option<String>,
    /// Recurrence pattern for repeating tasks
    pub recurrence: Option<Recurrence>,
    /// Structured dependencies: list of todo short IDs (e.g. ['SHO-1', 'BOLT-3']). This todo cannot start until all listed todos are Done.
    pub depends_on: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ListTodosParams {
    /// Semantic search query - when provided, uses vector similarity to find matching todos instead of listing all
    pub query: Option<String>,
    /// Filter by status(es)
    pub status: Option<Vec<TodoStatus>>,
    /// Filter by project name
    pub project: Option<String>,
    /// Filter by context (e.g., @computer)
    pub context: Option<String>,
    /// Filter by priority
    pub priority: Option<TodoPriorityFilter>,
    /// Filter by due date
    pub due: Option<DueFilter>,
    /// Maximum results (default: 50)
    pub limit: Option<u64>,
    /// Skip first N items for pagination (default: 0)
    pub offset: Option<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct UpdateTodoParams {
    /// Todo ID or short prefix
    pub todo_id: String,
    /// New content
    pub content: Option<String>,
    /// New status
    pub status: Option<TodoStatus>,
    /// New priority
    pub priority: Option<TodoPriority>,
    /// New project name
    pub project: Option<String>,
    /// New contexts
    pub contexts: Option<Vec<String>>,
    /// New due date
    pub due_date: Option<String>,
    /// Who/what you're waiting on
    pub blocked_on: Option<String>,
    /// Additional notes
    pub notes: Option<String>,
    /// New tags
    pub tags: Option<Vec<String>>,
    /// Parent todo ID or short prefix to make this a subtask. Pass empty string to remove parent.
    pub parent_id: Option<String>,
    /// Structured dependencies: list of todo short IDs. Replaces existing deps. Pass [] to clear.
    pub depends_on: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct TodoIdParams {
    /// Todo ID or short prefix
    pub todo_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ReorderTodoParams {
    /// Todo ID or short prefix
    pub todo_id: String,
    /// Direction to move the todo
    pub direction: ReorderDirection,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct AddProjectParams {
    /// Project name
    pub name: String,
    /// Custom prefix for todo IDs (e.g., 'BOLT', 'MEM'). Auto-derived from name if not provided.
    pub prefix: Option<String>,
    /// Project description
    pub description: Option<String>,
    /// Parent project name or ID to create a sub-project
    pub parent: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ArchiveProjectParams {
    /// Project name or ID to archive
    pub project: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct DeleteProjectParams {
    /// Project name or ID to delete
    pub project: String,
    /// Also delete all todos in this project (default: false)
    pub delete_todos: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ListSubtasksParams {
    /// Parent todo ID or short prefix
    pub parent_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct AddTodoCommentParams {
    /// Todo ID or short prefix (e.g., 'BOLT-1', 'MEM-2')
    pub todo_id: String,
    /// Comment content (supports markdown)
    pub content: String,
    /// Type of comment: comment (default), progress (updates), resolution (fix details), activity (system)
    pub comment_type: Option<CommentType>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ListTodoCommentsParams {
    /// Todo ID or short prefix (e.g., 'BOLT-1', 'MEM-2')
    pub todo_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct UpdateTodoCommentParams {
    /// Todo ID or short prefix
    pub todo_id: String,
    /// Comment ID (UUID)
    pub comment_id: String,
    /// New comment content
    pub content: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct DeleteTodoCommentParams {
    /// Todo ID or short prefix
    pub todo_id: String,
    /// Comment ID (UUID)
    pub comment_id: String,
}

#[derive(Serialize)]
struct CreateTodoRequest {
    user_id: String,
    content: String,
    status: NewTodoStatus,
    priority: TodoPriority,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    contexts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    due_date: Option<String>,
    tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked_on: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recurrence: Option<Recurrence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depends_on: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ListTodosRequest {
    user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<Vec<TodoStatus>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<TodoPriorityFilter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    due: Option<DueFilter>,
    limit: u64,
    offset: u64,
}

#[derive(Serialize)]
struct UpdateTodoRequest {
    user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<TodoStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<TodoPriority>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    contexts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    due_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked_on: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depends_on: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ReorderTodoRequest {
    user_id: String,
    direction: ReorderDirection,
}

#[derive(Serialize)]
struct CreateProjectRequest {
    user_id: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
}

#[derive(Serialize)]
struct ArchiveProjectRequest {
    user_id: String,
    status: &'static str,
}

#[derive(Serialize)]
struct DeleteProjectRequest {
    user_id: String,
    delete_todos: bool,
}

#[derive(Serialize)]
struct AddCommentRequest {
    user_id: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    comment_type: Option<CommentType>,
}

#[derive(Serialize)]
struct UpdateCommentRequest {
    user_id: String,
    content: String,
}

/// Every todo, project and comment endpoint renders its own text; the tools pass it through.
#[derive(Deserialize)]
struct FormattedResponse {
    formatted: String,
}

// =============================================================================
// HELPERS
// =============================================================================

fn text_ok(text: impl Into<String>) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}

/// The TypeScript server answers input errors as an error result, not a protocol error.
fn text_error(text: impl Into<String>) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(text)]))
}

fn formatted(result: Result<FormattedResponse>) -> Result<CallToolResult, McpError> {
    match result {
        Ok(resp) => text_ok(resp.formatted),
        Err(e) => Err(mcp_error(e)),
    }
}

fn content_error(content: &str) -> Option<&'static str> {
    if content.is_empty() {
        Some("Error: 'content' is required and cannot be empty")
    } else if content.len() > MAX_CONTENT_LENGTH {
        Some("Error: 'content' exceeds maximum length of 100000 characters")
    } else {
        None
    }
}

fn human_duration(seconds: i64) -> String {
    let mins = (seconds as f64 / 60.0).round() as i64;
    if mins > 60 {
        format!("{}h", (mins as f64 / 60.0).round() as i64)
    } else {
        format!("{mins}m")
    }
}

fn stars(priority: u8) -> String {
    let filled = usize::from(priority.min(5));
    format!("{}{}", "★".repeat(filled), "☆".repeat(5 - filled))
}

fn local_time(at: &chrono::DateTime<chrono::Utc>) -> String {
    at.with_timezone(&chrono::Local)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

// =============================================================================
// TOOLS
// =============================================================================

#[tool_router(router = work_tool_router, vis = "pub(crate)")]
impl VeldMcpServer {
    #[tool(
        description = "Set a reminder for the future. Triggers on time (at specific time or after duration) or context match (when keywords appear in conversation). Reminders will surface automatically when conditions are met."
    )]
    async fn set_reminder(
        &self,
        Parameters(params): Parameters<SetReminderParams>,
    ) -> Result<CallToolResult, McpError> {
        if let Some(message) = content_error(&params.content) {
            return text_error(message);
        }
        let priority = params.priority.unwrap_or(3);
        if !(1..=5).contains(&priority) {
            return text_error("Error: 'priority' must be between 1 and 5");
        }

        let trigger = match params.trigger_type {
            ReminderTriggerType::Time => match params.trigger_at {
                Some(at) if !at.is_empty() => ReminderTrigger::Time { at },
                _ => return text_error("Error: 'trigger_at' is required for time-based reminders"),
            },
            ReminderTriggerType::Duration => match params.after_seconds {
                Some(after_seconds) if after_seconds > 0 => {
                    ReminderTrigger::Duration { after_seconds }
                }
                _ => {
                    return text_error(
                        "Error: 'after_seconds' must be positive for duration-based reminders",
                    )
                }
            },
            ReminderTriggerType::Context => match params.keywords {
                Some(keywords) if !keywords.is_empty() => {
                    let threshold = params
                        .threshold
                        .filter(|t| (0.0..=1.0).contains(t))
                        .unwrap_or(0.7);
                    ReminderTrigger::Context {
                        keywords,
                        threshold,
                    }
                }
                _ => {
                    return text_error("Error: 'keywords' is required for context-based reminders")
                }
            },
        };

        let trigger_summary = match &trigger {
            ReminderTrigger::Time { .. } => None,
            ReminderTrigger::Duration { after_seconds } => Some(format!(
                " (in {})",
                human_duration(i64::try_from(*after_seconds).unwrap_or(i64::MAX))
            )),
            ReminderTrigger::Context { keywords, .. } => {
                Some(format!(" (keywords: {})", keywords.join(", ")))
            }
        };

        let result: Result<CreateReminderResponse> = self
            .client
            .post(
                "/api/remind",
                &CreateReminderRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content.clone(),
                    trigger,
                    priority,
                    tags: params.tags.unwrap_or_default(),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = String::from("🌾 Reminder Set\n");
                output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                output.push_str(&format!("ID: {}\n", resp.id));
                output.push_str(&format!("Content: {}\n", params.content));
                let trigger_name = match params.trigger_type {
                    ReminderTriggerType::Time => "time",
                    ReminderTriggerType::Duration => "duration",
                    ReminderTriggerType::Context => "context",
                };
                output.push_str(&format!("Trigger: {trigger_name}"));
                match (&params.trigger_type, &resp.due_at, trigger_summary) {
                    (ReminderTriggerType::Time, Some(due_at), _) => {
                        output.push_str(&format!(" ({})", local_time(due_at)));
                    }
                    (_, _, Some(summary)) => output.push_str(&summary),
                    _ => {}
                }
                output.push('\n');
                if priority != 3 {
                    output.push_str(&format!("Priority: {}\n", stars(priority)));
                }
                text_ok(output)
            }
            Err(e) => Err(mcp_error(e)),
        }
    }

    #[tool(description = "List all pending reminders. Use to check what reminders are scheduled.")]
    async fn list_reminders(
        &self,
        Parameters(params): Parameters<ListRemindersParams>,
    ) -> Result<CallToolResult, McpError> {
        let status = params.status.unwrap_or(ReminderStatusFilter::Pending);
        let status_name = match status {
            ReminderStatusFilter::Pending => "pending",
            ReminderStatusFilter::Triggered => "triggered",
            ReminderStatusFilter::Dismissed => "dismissed",
            ReminderStatusFilter::All => "all",
        };
        let result: Result<ListRemindersResponse> = self
            .client
            .post(
                "/api/reminders",
                &ListRemindersRequest {
                    user_id: self.client.user_id.clone(),
                    status: match status {
                        ReminderStatusFilter::All => None,
                        _ => Some(status_name.to_string()),
                    },
                },
            )
            .await;

        match result {
            Ok(resp) => {
                if resp.count == 0 {
                    let label = match status {
                        ReminderStatusFilter::All => String::new(),
                        _ => format!("{status_name} "),
                    };
                    return text_ok(format!("No {label}reminders found."));
                }
                let mut output = format!("🌾 VELD REMINDERS ({})\n", resp.count);
                output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                for r in &resp.reminders {
                    let overdue = r.overdue_seconds.filter(|s| *s > 0);
                    let icon = if overdue.is_some() { "⏰" } else { "📌" };
                    let badge = if r.status == "triggered" {
                        " [TRIGGERED]"
                    } else {
                        ""
                    };
                    let preview: String = r.content.chars().take(50).collect();
                    let ellipsis = if r.content.chars().count() > 50 {
                        "..."
                    } else {
                        ""
                    };
                    output.push_str(&format!("{icon} {preview}{ellipsis}{badge}\n"));
                    output.push_str(&format!(
                        "   Type: {} | Priority: {} | ID: {}\n",
                        r.trigger_type,
                        "★".repeat(usize::from(r.priority)),
                        r.id
                    ));
                    if let Some(due_at) = &r.due_at {
                        output.push_str(&format!("   Due: {}\n", local_time(due_at)));
                    }
                    if let Some(seconds) = overdue {
                        output.push_str(&format!("   ⚠️ Overdue by {}\n", human_duration(seconds)));
                    }
                    output.push('\n');
                }
                text_ok(output)
            }
            Err(e) => Err(mcp_error(e)),
        }
    }

    #[tool(
        description = "Dismiss/acknowledge a triggered reminder. Call this after you've handled a reminder."
    )]
    async fn dismiss_reminder(
        &self,
        Parameters(params): Parameters<DismissReminderParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/reminders/{}/dismiss", encode(&params.reminder_id));
        let result: Result<ActionResponse> = self
            .client
            .post(
                &endpoint,
                &UserOnlyRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;

        match result {
            Ok(resp) if resp.success => {
                text_ok(format!("✓ Reminder dismissed: {}", params.reminder_id))
            }
            Ok(resp) => text_ok(format!(
                "⚠️ {}",
                if resp.message.is_empty() {
                    "No message returned".to_string()
                } else {
                    resp.message
                }
            )),
            Err(e) => Err(mcp_error(e)),
        }
    }

    #[tool(
        description = "Add a task to your todo list. Supports GTD workflow with projects, contexts (@computer, @phone), priorities, due dates, and subtasks (via parent_id)."
    )]
    async fn add_todo(
        &self,
        Parameters(params): Parameters<AddTodoParams>,
    ) -> Result<CallToolResult, McpError> {
        if let Some(message) = content_error(&params.content) {
            return text_error(message);
        }
        let result: Result<FormattedResponse> = self
            .client
            .post(
                "/api/todos/add",
                &CreateTodoRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    status: params.status.unwrap_or(NewTodoStatus::Todo),
                    priority: params.priority.unwrap_or(TodoPriority::Medium),
                    project: params.project,
                    contexts: params.contexts.unwrap_or_default(),
                    due_date: params.due_date,
                    tags: params.tags.unwrap_or_default(),
                    blocked_on: params.blocked_on,
                    notes: params.notes,
                    recurrence: params.recurrence,
                    depends_on: params.depends_on,
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "List or search todos. Supports semantic search via query parameter, or GTD-style filtering. Returns Linear-style formatted output grouped by status."
    )]
    async fn list_todos(
        &self,
        Parameters(params): Parameters<ListTodosParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<FormattedResponse> = self
            .client
            .post(
                "/api/todos/list",
                &ListTodosRequest {
                    user_id: self.client.user_id.clone(),
                    query: params.query,
                    status: params.status,
                    project: params.project,
                    context: params.context,
                    priority: params.priority,
                    due: params.due,
                    limit: params.limit.unwrap_or(50).clamp(1, MAX_LIMIT),
                    offset: params.offset.unwrap_or(0),
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "Update a todo's properties. Use short ID prefix (e.g., SHO-1a2b) or full ID."
    )]
    async fn update_todo(
        &self,
        Parameters(params): Parameters<UpdateTodoParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/todos/{}/update", encode(&params.todo_id));
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &UpdateTodoRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    status: params.status,
                    priority: params.priority,
                    project: params.project,
                    contexts: params.contexts,
                    due_date: params.due_date,
                    blocked_on: params.blocked_on,
                    notes: params.notes,
                    tags: params.tags,
                    parent_id: params.parent_id,
                    depends_on: params.depends_on,
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "Mark a todo as complete. For recurring tasks, automatically creates the next occurrence."
    )]
    async fn complete_todo(
        &self,
        Parameters(params): Parameters<TodoIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/todos/{}/complete", encode(&params.todo_id));
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &UserOnlyRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        formatted(result)
    }

    #[tool(description = "Delete a todo permanently.")]
    async fn delete_todo(
        &self,
        Parameters(params): Parameters<TodoIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!(
            "/api/todos/{}?user_id={}",
            encode(&params.todo_id),
            encode(&self.client.user_id)
        );
        let result: Result<FormattedResponse> = self.client.delete(&endpoint).await;
        formatted(result)
    }

    #[tool(
        description = "Move a todo up or down within its status group. Use to prioritize tasks manually."
    )]
    async fn reorder_todo(
        &self,
        Parameters(params): Parameters<ReorderTodoParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/todos/{}/reorder", encode(&params.todo_id));
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &ReorderTodoRequest {
                    user_id: self.client.user_id.clone(),
                    direction: params.direction,
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "Create a new project to group todos. Use parent to create a sub-project under another project."
    )]
    async fn add_project(
        &self,
        Parameters(params): Parameters<AddProjectParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<FormattedResponse> = self
            .client
            .post(
                "/api/projects",
                &CreateProjectRequest {
                    user_id: self.client.user_id.clone(),
                    name: params.name,
                    prefix: params.prefix,
                    description: params.description,
                    parent: params.parent,
                },
            )
            .await;
        formatted(result)
    }

    #[tool(description = "List all projects with todo counts and status breakdown.")]
    async fn list_projects(&self) -> Result<CallToolResult, McpError> {
        let result: Result<FormattedResponse> = self
            .client
            .post(
                "/api/projects/list",
                &UserOnlyRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "Archive a project. Archived projects are hidden by default but can be restored."
    )]
    async fn archive_project(
        &self,
        Parameters(params): Parameters<ArchiveProjectParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/projects/{}/update", encode(&params.project));
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &ArchiveProjectRequest {
                    user_id: self.client.user_id.clone(),
                    status: "archived",
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "Permanently delete a project. Use delete_todos=true to also delete all todos in the project."
    )]
    async fn delete_project(
        &self,
        Parameters(params): Parameters<DeleteProjectParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/projects/{}/delete", encode(&params.project));
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &DeleteProjectRequest {
                    user_id: self.client.user_id.clone(),
                    delete_todos: params.delete_todos.unwrap_or(false),
                },
            )
            .await;
        formatted(result)
    }

    #[tool(description = "Get statistics about your todos - counts by status, overdue items, etc.")]
    async fn todo_stats(&self) -> Result<CallToolResult, McpError> {
        let result: Result<FormattedResponse> = self
            .client
            .post(
                "/api/todos/stats",
                &UserOnlyRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;
        formatted(result)
    }

    #[tool(
        description = "List subtasks of a parent todo. Use add_todo with parent_id to create subtasks."
    )]
    async fn list_subtasks(
        &self,
        Parameters(params): Parameters<ListSubtasksParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!(
            "/api/todos/{}/subtasks?user_id={}",
            encode(&params.parent_id),
            encode(&self.client.user_id)
        );
        let result: Result<FormattedResponse> = self.client.get(&endpoint).await;
        formatted(result)
    }

    #[tool(
        description = "Add a comment to a todo. Use to track progress, notes, or resolution details."
    )]
    async fn add_todo_comment(
        &self,
        Parameters(params): Parameters<AddTodoCommentParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!("/api/todos/{}/comments", encode(&params.todo_id));
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &AddCommentRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    comment_type: params.comment_type,
                },
            )
            .await;
        formatted(result)
    }

    #[tool(description = "List all comments and activity history for a specific todo.")]
    async fn list_todo_comments(
        &self,
        Parameters(params): Parameters<ListTodoCommentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!(
            "/api/todos/{}/comments?user_id={}",
            encode(&params.todo_id),
            encode(&self.client.user_id)
        );
        let result: Result<FormattedResponse> = self.client.get(&endpoint).await;
        formatted(result)
    }

    #[tool(description = "Update an existing comment on a todo.")]
    async fn update_todo_comment(
        &self,
        Parameters(params): Parameters<UpdateTodoCommentParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!(
            "/api/todos/{}/comments/{}/update",
            encode(&params.todo_id),
            encode(&params.comment_id)
        );
        let result: Result<FormattedResponse> = self
            .client
            .post(
                &endpoint,
                &UpdateCommentRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                },
            )
            .await;
        formatted(result)
    }

    #[tool(description = "Delete a comment from a todo.")]
    async fn delete_todo_comment(
        &self,
        Parameters(params): Parameters<DeleteTodoCommentParams>,
    ) -> Result<CallToolResult, McpError> {
        let endpoint = format!(
            "/api/todos/{}/comments/{}?user_id={}",
            encode(&params.todo_id),
            encode(&params.comment_id),
            encode(&self.client.user_id)
        );
        let result: Result<FormattedResponse> = self.client.delete(&endpoint).await;
        formatted(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_wire_names_match_the_typescript_schema() {
        assert_eq!(
            serde_json::to_string(&TodoStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&DueFilter::ThisWeek).unwrap(),
            "\"this_week\""
        );
        assert_eq!(
            serde_json::to_string(&TodoPriority::None).unwrap(),
            "\"none\""
        );
    }

    #[test]
    fn reminder_trigger_serialises_as_the_tagged_shape_the_server_reads() {
        let time = serde_json::to_value(ReminderTrigger::Time {
            at: "2025-12-23T18:00:00Z".to_string(),
        })
        .unwrap();
        assert_eq!(time["type"], "time");
        assert_eq!(time["at"], "2025-12-23T18:00:00Z");

        let context = serde_json::to_value(ReminderTrigger::Context {
            keywords: vec!["deploy".to_string()],
            threshold: 0.7,
        })
        .unwrap();
        assert_eq!(context["type"], "context");
        assert_eq!(context["keywords"][0], "deploy");
    }

    #[test]
    fn durations_and_stars_render_like_the_typescript_server() {
        assert_eq!(human_duration(90), "2m");
        assert_eq!(human_duration(7200), "2h");
        assert_eq!(human_duration(3600), "60m");
        assert_eq!(stars(4), "★★★★☆");
    }

    #[test]
    fn content_limits_match() {
        assert!(content_error("").is_some());
        assert!(content_error("x").is_none());
        assert!(content_error(&"x".repeat(MAX_CONTENT_LENGTH + 1)).is_some());
    }
}
