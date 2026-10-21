//! Todo, Reminder, and Project Handlers
//!
//! GTD-style task management with:
//! - Prospective memory (reminders with time/duration/context triggers)
//! - Todo CRUD with semantic search
//! - Project hierarchy with nested projects
//! - Todo comments and activity tracking

use axum::{
    extract::{Extension, Path, Query, State},
    response::Json,
};
use chrono::Datelike;
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::sessions::SessionEvent;
use crate::memory::todo_formatter;
use crate::memory::{Experience, ExperienceType};
use crate::memory::{
    MemoryId, Project, ProjectId, ProjectStats, ProjectStatus, ProspectiveTask, ProspectiveTaskId,
    ProspectiveTaskStatus, ProspectiveTrigger, Recurrence, Todo, TodoComment, TodoCommentId,
    TodoCommentType, TodoId, TodoPriority, TodoStatus, UserTodoStats,
};
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// REMINDER REQUEST/RESPONSE TYPES
// =============================================================================

/// Request to create a new reminder
#[derive(Debug, Deserialize)]
pub struct CreateReminderRequest {
    pub user_id: String,
    pub content: String,
    pub trigger: ReminderTriggerRequest,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_reminder_priority")]
    pub priority: u8,
}

fn default_reminder_priority() -> u8 {
    3
}

/// Trigger configuration for reminder creation
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReminderTriggerRequest {
    Time {
        at: chrono::DateTime<chrono::Utc>,
    },
    Duration {
        after_seconds: u64,
    },
    Context {
        keywords: Vec<String>,
        #[serde(default = "default_context_threshold")]
        threshold: f32,
    },
}

fn default_context_threshold() -> f32 {
    0.7
}

/// Response for reminder creation
#[derive(Debug, Serialize)]
pub struct CreateReminderResponse {
    pub id: String,
    pub content: String,
    pub trigger_type: String,
    pub due_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Request to list reminders
#[derive(Debug, Deserialize)]
pub struct ListRemindersRequest {
    pub user_id: String,
    pub status: Option<String>,
}

/// Individual reminder in list response
#[derive(Debug, Serialize)]
pub struct ReminderItem {
    pub id: String,
    pub content: String,
    pub trigger_type: String,
    pub status: String,
    pub due_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub triggered_at: Option<chrono::DateTime<chrono::Utc>>,
    pub dismissed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub priority: u8,
    pub tags: Vec<String>,
    pub overdue_seconds: Option<i64>,
}

/// Response for listing reminders
#[derive(Debug, Serialize)]
pub struct ListRemindersResponse {
    pub reminders: Vec<ReminderItem>,
    pub count: usize,
}

/// Request to get due reminders
#[derive(Debug, Deserialize)]
pub struct GetDueRemindersRequest {
    pub user_id: String,
    #[serde(default = "default_true")]
    pub mark_triggered: bool,
}

fn default_true() -> bool {
    true
}

/// Response for due reminders
#[derive(Debug, Serialize)]
pub struct DueRemindersResponse {
    pub reminders: Vec<ReminderItem>,
    pub count: usize,
}

/// Request to check context-triggered reminders
#[derive(Debug, Deserialize)]
pub struct CheckContextRemindersRequest {
    pub user_id: String,
    pub context: String,
    #[serde(default = "default_true")]
    pub mark_triggered: bool,
}

/// Request to dismiss a reminder
#[derive(Debug, Deserialize)]
pub struct DismissReminderRequest {
    pub user_id: String,
}

/// Response for dismiss/delete operations
#[derive(Debug, Serialize)]
pub struct ReminderActionResponse {
    pub success: bool,
    pub message: String,
}

/// Query for delete reminder
#[derive(Debug, Deserialize)]
pub struct DeleteReminderQuery {
    pub user_id: String,
}

// =============================================================================
// TODO REQUEST/RESPONSE TYPES
// =============================================================================

/// Request to create a new todo
#[derive(Debug, Deserialize)]
pub struct CreateTodoRequest {
    pub user_id: String,
    pub content: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub priority: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    #[serde(default)]
    pub due_date: Option<String>,
    #[serde(default)]
    pub blocked_on: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub recurrence: Option<String>,
    #[serde(default)]
    pub external_id: Option<String>,
    /// Structured dependencies: list of todo short IDs (e.g. ["SHO-1", "BOLT-3"]) or UUIDs.
    /// This todo cannot start until all listed todos are Done. `blocked_by` is upstream's name
    /// for the same list. An unknown reference is a 400.
    #[serde(default, alias = "blocked_by")]
    pub depends_on: Option<Vec<String>>,
    /// Memory UUIDs that motivated this todo, verified to exist.
    #[serde(default)]
    pub related_memory_ids: Option<Vec<String>>,
}

/// Wire form of a todo: the stored record without its embedding. The embedding cannot be
/// skipped on the field itself, because TodoStore persists todos through that same Serialize
/// impl and the stored embedding is what semantic todo search reads. No client reads it, and it
/// was most of every response's bytes.
mod todo_wire {
    use super::Todo;
    use serde::ser::SerializeSeq;
    use serde::{Serialize, Serializer};

    fn stripped(todo: &Todo) -> Todo {
        let mut todo = todo.clone();
        todo.embedding = None;
        todo
    }

    pub fn opt<S: Serializer>(todo: &Option<Todo>, ser: S) -> Result<S::Ok, S::Error> {
        match todo {
            Some(todo) => stripped(todo).serialize(ser),
            None => ser.serialize_none(),
        }
    }

    pub fn list<S: Serializer>(todos: &[Todo], ser: S) -> Result<S::Ok, S::Error> {
        let mut seq = ser.serialize_seq(Some(todos.len()))?;
        for todo in todos {
            seq.serialize_element(&stripped(todo))?;
        }
        seq.end()
    }
}

/// Response for todo operations
#[derive(Debug, Serialize)]
pub struct TodoResponse {
    pub success: bool,
    #[serde(serialize_with = "todo_wire::opt")]
    pub todo: Option<Todo>,
    pub project: Option<Project>,
    pub formatted: String,
}

/// Response for todo list operations
#[derive(Debug, Serialize)]
pub struct TodoListResponse {
    pub success: bool,
    pub count: usize,
    #[serde(serialize_with = "todo_wire::list")]
    pub todos: Vec<Todo>,
    pub projects: Vec<Project>,
    pub formatted: String,
}

/// Response for todo complete with potential next recurrence
#[derive(Debug, Serialize)]
pub struct TodoCompleteResponse {
    pub success: bool,
    #[serde(serialize_with = "todo_wire::opt")]
    pub todo: Option<Todo>,
    #[serde(serialize_with = "todo_wire::opt")]
    pub next_recurrence: Option<Todo>,
    /// Todos whose dependencies are all closed now that this one is done.
    #[serde(serialize_with = "todo_wire::list")]
    pub unblocked: Vec<Todo>,
    pub formatted: String,
}

/// Request to list todos with filters
#[derive(Debug, Deserialize)]
pub struct ListTodosRequest {
    pub user_id: String,
    #[serde(default)]
    pub status: Option<Vec<String>>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub include_completed: Option<bool>,
    #[serde(default)]
    pub due: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub priority: Option<String>,
}

/// Request to update a todo
#[derive(Debug, Deserialize)]
pub struct UpdateTodoRequest {
    pub user_id: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub priority: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    #[serde(default)]
    pub due_date: Option<String>,
    #[serde(default)]
    pub blocked_on: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub sort_order: Option<i32>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub external_id: Option<String>,
    /// Structured dependencies: list of todo short IDs (e.g. ["SHO-1", "BOLT-3"]) or UUIDs.
    /// Replaces the set; an empty list clears it. Unknown references and cycles are a 400.
    #[serde(default, alias = "blocked_by")]
    pub depends_on: Option<Vec<String>>,
    /// Replaces the set of linked memory UUIDs; an empty list clears it.
    #[serde(default)]
    pub related_memory_ids: Option<Vec<String>>,
    /// The target todo, named in the body by the flat `/api/todos/update` alias; the path-style
    /// route takes it from the URL.
    #[serde(default)]
    pub todo_id: Option<String>,
}

/// Request to reorder a todo
#[derive(Debug, Deserialize)]
pub struct ReorderTodoRequest {
    pub user_id: String,
    pub direction: String,
    /// The target todo, named in the body by the flat `/api/todos/reorder` alias.
    #[serde(default)]
    pub todo_id: Option<String>,
}

/// Body of the flat `/api/todos/complete` and `/api/todos/delete` aliases, which have no path
/// capture and name their target here.
#[derive(Debug, Deserialize)]
pub struct FlatTodoRequest {
    pub user_id: String,
    pub todo_id: String,
}

/// Request to get due todos
#[derive(Debug, Deserialize)]
pub struct DueTodosRequest {
    pub user_id: String,
    #[serde(default = "default_include_overdue")]
    pub include_overdue: bool,
}

fn default_include_overdue() -> bool {
    true
}

/// Query params for single todo operations
#[derive(Debug, Deserialize)]
pub struct TodoQuery {
    pub user_id: String,
}

/// Request for todo stats
#[derive(Debug, Deserialize)]
pub struct TodoStatsRequest {
    pub user_id: String,
}

/// Response for todo stats
#[derive(Debug, Serialize)]
pub struct TodoStatsResponse {
    pub success: bool,
    pub stats: UserTodoStats,
    pub formatted: String,
}

// =============================================================================
// COMMENT REQUEST/RESPONSE TYPES
// =============================================================================

/// Request to add a comment to a todo
#[derive(Debug, Deserialize)]
pub struct AddCommentRequest {
    pub user_id: String,
    pub content: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub comment_type: Option<String>,
}

/// Request to update a comment
#[derive(Debug, Deserialize)]
pub struct UpdateCommentRequest {
    pub user_id: String,
    pub content: String,
}

/// Response for comment operations
#[derive(Debug, Serialize)]
pub struct CommentResponse {
    pub success: bool,
    pub comment: Option<TodoComment>,
    pub formatted: String,
}

/// Response for listing comments
#[derive(Debug, Serialize)]
pub struct CommentListResponse {
    pub success: bool,
    pub count: usize,
    pub comments: Vec<TodoComment>,
    pub formatted: String,
}

// =============================================================================
// PROJECT REQUEST/RESPONSE TYPES
// =============================================================================

/// Request to create a project
#[derive(Debug, Deserialize)]
pub struct CreateProjectRequest {
    pub user_id: String,
    pub name: String,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
}

/// Response for project operations
#[derive(Debug, Serialize)]
pub struct ProjectResponse {
    pub success: bool,
    pub project: Option<Project>,
    pub stats: Option<ProjectStats>,
    pub formatted: String,
}

/// Response for project list
#[derive(Debug, Serialize)]
pub struct ProjectListResponse {
    pub success: bool,
    pub count: usize,
    pub projects: Vec<(Project, ProjectStats)>,
    pub formatted: String,
}

/// Request to update a project
#[derive(Debug, Deserialize)]
pub struct UpdateProjectRequest {
    pub user_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub status: Option<ProjectStatus>,
    #[serde(default)]
    pub color: Option<Option<String>>,
}

/// Request to delete a project
#[derive(Debug, Deserialize)]
pub struct DeleteProjectRequest {
    pub user_id: String,
    #[serde(default)]
    pub delete_todos: bool,
}

/// Request to list projects
#[derive(Debug, Deserialize)]
pub struct ListProjectsRequest {
    pub user_id: String,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Parse recurrence string to Recurrence enum
fn parse_recurrence(s: &str) -> Result<Recurrence, AppError> {
    match s.to_lowercase().as_str() {
        "daily" => Ok(Recurrence::Daily),
        "weekly" => Ok(Recurrence::Weekly {
            days: vec![1, 2, 3, 4, 5],
        }),
        "monthly" => Ok(Recurrence::Monthly { day: 1 }),
        _ => Err(AppError::InvalidInput {
            field: "recurrence".to_string(),
            reason: format!("Unknown recurrence '{s}'. Valid values: daily, weekly, monthly"),
        }),
    }
}

fn parse_todo_status(s: &str) -> Result<TodoStatus, AppError> {
    TodoStatus::from_str_loose(s).ok_or_else(|| AppError::InvalidInput {
        field: "status".to_string(),
        reason: format!(
            "Unknown todo status '{s}'. Valid values: backlog, todo, in_progress, blocked, done, \
             cancelled"
        ),
    })
}

fn parse_todo_priority(s: &str) -> Result<TodoPriority, AppError> {
    TodoPriority::from_str_loose(s).ok_or_else(|| AppError::InvalidInput {
        field: "priority".to_string(),
        reason: format!("Unknown priority '{s}'. Valid values: urgent, high, medium, low, none"),
    })
}

/// Resolves todo references (short keys like "SHO-3", sequence numbers or UUIDs) to ids. A
/// dependency on a todo that does not exist is a data error, not something to drop silently.
fn resolve_todo_refs(
    state: &AppState,
    user_id: &str,
    refs: &[String],
    field: &str,
) -> Result<Vec<TodoId>, AppError> {
    let mut resolved = Vec::with_capacity(refs.len());
    for r in refs {
        let todo = state
            .todo_store
            .find_todo_by_prefix(user_id, r)
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::InvalidInput {
                field: field.to_string(),
                reason: format!("No todo found matching '{r}'"),
            })?;
        if !resolved.contains(&todo.id) {
            resolved.push(todo.id);
        }
    }
    Ok(resolved)
}

/// Parses memory UUIDs and checks each exists, so a todo never links a memory that is not
/// there.
async fn verify_memory_ids(
    state: &AppState,
    user_id: &str,
    ids: &[String],
) -> Result<Vec<MemoryId>, AppError> {
    let mut parsed = Vec::with_capacity(ids.len());
    for id_str in ids {
        let uuid = uuid::Uuid::parse_str(id_str).map_err(|_| AppError::InvalidInput {
            field: "related_memory_ids".to_string(),
            reason: format!("'{id_str}' is not a valid memory UUID"),
        })?;
        let mid = MemoryId(uuid);
        if !parsed.contains(&mid) {
            parsed.push(mid);
        }
    }
    if parsed.is_empty() {
        return Ok(parsed);
    }

    let memory_system = state.get_user_memory(user_id).map_err(AppError::Internal)?;
    let to_check = parsed.clone();
    let missing = tokio::task::spawn_blocking(move || {
        let guard = memory_system.read();
        to_check
            .into_iter()
            .find(|mid| guard.get_memory(mid).is_err())
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Memory verification task panicked: {e}")))?;
    if let Some(mid) = missing {
        return Err(AppError::InvalidInput {
            field: "related_memory_ids".to_string(),
            reason: format!("No memory found with id '{}'", mid.0),
        });
    }
    Ok(parsed)
}

/// Every todo under a project and its sub-projects: the set TodoStore::delete_project
/// cascades over.
fn collect_project_todos(
    state: &AppState,
    user_id: &str,
    project_id: &ProjectId,
    out: &mut Vec<Todo>,
) -> Result<(), AppError> {
    out.extend(
        state
            .todo_store
            .list_todos_by_project(user_id, project_id)
            .map_err(AppError::Internal)?,
    );
    for sub in state
        .todo_store
        .list_subprojects(user_id, project_id)
        .map_err(AppError::Internal)?
    {
        collect_project_todos(state, user_id, &sub.id, out)?;
    }
    Ok(())
}

/// Writes the memory half of the todo-memory link: `add` memories gain a back-link to the
/// todo, `remove` memories lose it, so a replaced link set stays consistent on both sides. A
/// back-link is bookkeeping: a memory deleted since is logged and skipped, not a failed write.
async fn sync_memory_back_links(
    state: &AppState,
    user_id: &str,
    todo_id: &TodoId,
    add: Vec<MemoryId>,
    remove: Vec<MemoryId>,
) -> Result<(), AppError> {
    if add.is_empty() && remove.is_empty() {
        return Ok(());
    }
    let memory_system = state.get_user_memory(user_id).map_err(AppError::Internal)?;
    let todo_id = *todo_id;
    tokio::task::spawn_blocking(move || {
        let guard = memory_system.read();
        for memory_id in remove {
            if let Err(e) = guard.unlink_related_todo(&memory_id, &todo_id) {
                tracing::warn!(memory_id = %memory_id.0, todo_id = %todo_id, "Failed to remove todo back-link: {e}");
            }
        }
        for memory_id in add {
            if let Err(e) = guard.link_related_todo(&memory_id, todo_id) {
                tracing::warn!(memory_id = %memory_id.0, todo_id = %todo_id, "Failed to write todo back-link: {e}");
            }
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Back-link task panicked: {e}")))
}

/// The target of a flat alias, named in its body. A missing id is a 400, never the Path
/// extractor's 500.
fn flat_todo_id(todo_id: Option<String>) -> Result<String, AppError> {
    match todo_id {
        Some(id) if !id.trim().is_empty() => Ok(id),
        _ => Err(AppError::InvalidInput {
            field: "todo_id".to_string(),
            reason: "This endpoint takes the target todo in the request body; pass \"todo_id\", \
                     or use the path-style route (/api/todos/{todo_id}/...)."
                .to_string(),
        }),
    }
}

/// Links a memory the todo flow just stored back to its todo, so the todo's provenance can be
/// walked with the memory tools.
fn link_todo_memory(state: &AppState, user_id: &str, todo_id: &TodoId, memory_id: &MemoryId) {
    if let Err(e) = state
        .todo_store
        .add_related_memory(user_id, todo_id, memory_id.clone())
    {
        tracing::debug!("Failed to link memory {} to todo: {e}", memory_id.0);
    }
}

// =============================================================================
// REMINDER HANDLERS
// =============================================================================

/// Create a new reminder (prospective memory)
pub async fn create_reminder(
    State(state): State<AppState>,
    Json(req): Json<CreateReminderRequest>,
) -> Result<Json<CreateReminderResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Reminder content cannot be empty".to_string(),
        });
    }

    let trigger = match req.trigger {
        ReminderTriggerRequest::Time { at } => {
            validation::validate_reminder_timestamp(&at).map_validation_err("trigger_at")?;
            ProspectiveTrigger::AtTime { at }
        }
        ReminderTriggerRequest::Duration { after_seconds } => {
            if after_seconds > 5 * 365 * 24 * 3600 {
                return Err(AppError::InvalidInput {
                    field: "after_seconds".to_string(),
                    reason: "Duration cannot exceed 5 years".to_string(),
                });
            }
            ProspectiveTrigger::AfterDuration {
                seconds: after_seconds,
                from: chrono::Utc::now(),
            }
        }
        ReminderTriggerRequest::Context {
            keywords,
            threshold,
        } => {
            if keywords.is_empty() {
                return Err(AppError::InvalidInput {
                    field: "keywords".to_string(),
                    reason: "Context trigger requires at least one keyword".to_string(),
                });
            }
            validation::validate_weight("threshold", threshold).map_validation_err("threshold")?;
            ProspectiveTrigger::OnContext {
                keywords,
                threshold,
            }
        }
    };

    let mut task = ProspectiveTask::new(req.user_id.clone(), req.content.clone(), trigger);
    task.tags = req.tags;
    task.priority = req.priority.clamp(1, 5);

    // Cache embedding at creation time for context triggers (avoids recomputation on every check)
    if matches!(task.trigger, ProspectiveTrigger::OnContext { .. }) {
        if let Ok(memory_system) = state.get_user_memory(&req.user_id) {
            let content_for_embed = task.content.clone();
            let memory_clone = memory_system.clone();
            if let Ok(Ok(embedding)) = tokio::task::spawn_blocking(move || {
                let guard = memory_clone.read();
                guard.compute_embedding(&content_for_embed)
            })
            .await
            {
                task.embedding = Some(embedding);
            }
        }
    }

    let trigger_type = match &task.trigger {
        ProspectiveTrigger::AtTime { .. } => "time",
        ProspectiveTrigger::AfterDuration { .. } => "duration",
        ProspectiveTrigger::OnContext { .. } => "context",
    };

    let due_at = task.trigger.due_at();

    state
        .prospective_store
        .store(&task)
        .map_err(AppError::Internal)?;

    tracing::info!(
        user_id = %req.user_id,
        reminder_id = %task.id,
        trigger_type = trigger_type,
        "Created prospective memory (reminder)"
    );

    state.log_event(
        &req.user_id,
        "REMINDER_CREATE",
        &task.id.to_string(),
        &format!(
            "Created reminder trigger={}: '{}'",
            trigger_type,
            req.content.chars().take(50).collect::<String>()
        ),
    );

    Ok(Json(CreateReminderResponse {
        id: task.id.to_string(),
        content: task.content,
        trigger_type: trigger_type.to_string(),
        due_at,
        created_at: task.created_at,
    }))
}

/// List reminders for a user
pub async fn list_reminders(
    State(state): State<AppState>,
    Json(req): Json<ListRemindersRequest>,
) -> Result<Json<ListRemindersResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let status_filter = match req.status.as_deref() {
        Some("pending") => Some(ProspectiveTaskStatus::Pending),
        Some("triggered") => Some(ProspectiveTaskStatus::Triggered),
        Some("dismissed") => Some(ProspectiveTaskStatus::Dismissed),
        Some("expired") => Some(ProspectiveTaskStatus::Expired),
        Some("all") | None => None,
        Some(unknown) => {
            return Err(AppError::InvalidInput {
                field: "status".to_string(),
                reason: format!(
                    "Unknown reminder status '{unknown}'. Valid values: pending, triggered, \
                     dismissed, expired, all"
                ),
            });
        }
    };

    let tasks = state
        .prospective_store
        .list_for_user(&req.user_id, status_filter)
        .map_err(AppError::Internal)?;

    let reminders: Vec<ReminderItem> = tasks
        .into_iter()
        .map(|t| {
            let overdue_seconds = t.overdue_seconds();
            ReminderItem {
                id: t.id.to_string(),
                content: t.content,
                trigger_type: match &t.trigger {
                    ProspectiveTrigger::AtTime { .. } => "time".to_string(),
                    ProspectiveTrigger::AfterDuration { .. } => "duration".to_string(),
                    ProspectiveTrigger::OnContext { .. } => "context".to_string(),
                },
                status: format!("{:?}", t.status).to_lowercase(),
                due_at: t.trigger.due_at(),
                created_at: t.created_at,
                triggered_at: t.triggered_at,
                dismissed_at: t.dismissed_at,
                priority: t.priority,
                tags: t.tags,
                overdue_seconds,
            }
        })
        .collect();

    let count = reminders.len();

    Ok(Json(ListRemindersResponse { reminders, count }))
}

/// Get due time-based reminders
pub async fn get_due_reminders(
    State(state): State<AppState>,
    Json(req): Json<GetDueRemindersRequest>,
) -> Result<Json<DueRemindersResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let due_tasks = state
        .prospective_store
        .get_due_tasks(&req.user_id)
        .map_err(AppError::Internal)?;

    // Mark triggered and re-read actual state from DB (fixes C1: timestamp mismatch,
    // C2: stale snapshot, C3: silent error swallowing)
    let tasks_for_response: Vec<ProspectiveTask> = if req.mark_triggered {
        let mut result = Vec::with_capacity(due_tasks.len());
        for task in &due_tasks {
            match state
                .prospective_store
                .mark_triggered(&req.user_id, &task.id)
            {
                Ok(true) => {
                    // Re-read to get the actual DB state with correct triggered_at timestamp
                    match state.prospective_store.get(&req.user_id, &task.id) {
                        Ok(Some(updated)) => result.push(updated),
                        _ => result.push(task.clone()),
                    }
                }
                Ok(false) => {
                    // Already triggered by concurrent call (race) — skip
                    tracing::debug!(task_id = %task.id, "Reminder already triggered (concurrent)");
                }
                Err(e) => {
                    tracing::warn!(task_id = %task.id, error = %e, "Failed to mark reminder triggered");
                    result.push(task.clone());
                }
            }
        }
        result
    } else {
        due_tasks
    };

    let reminders: Vec<ReminderItem> = tasks_for_response
        .into_iter()
        .map(|t| {
            let overdue_seconds = t.overdue_seconds();
            ReminderItem {
                id: t.id.to_string(),
                content: t.content,
                trigger_type: match &t.trigger {
                    ProspectiveTrigger::AtTime { .. } => "time".to_string(),
                    ProspectiveTrigger::AfterDuration { .. } => "duration".to_string(),
                    ProspectiveTrigger::OnContext { .. } => "context".to_string(),
                },
                status: format!("{:?}", t.status).to_lowercase(),
                due_at: t.trigger.due_at(),
                created_at: t.created_at,
                triggered_at: t.triggered_at,
                dismissed_at: t.dismissed_at,
                priority: t.priority,
                tags: t.tags,
                overdue_seconds,
            }
        })
        .collect();

    let count = reminders.len();

    if count > 0 {
        tracing::debug!(
            user_id = %req.user_id,
            count = count,
            "Returning due reminders"
        );
    }

    Ok(Json(DueRemindersResponse { reminders, count }))
}

/// Check for context-triggered reminders
pub async fn check_context_reminders(
    State(state): State<AppState>,
    Json(req): Json<CheckContextRemindersRequest>,
) -> Result<Json<DueRemindersResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.context.trim().is_empty() {
        return Ok(Json(DueRemindersResponse {
            reminders: vec![],
            count: 0,
        }));
    }

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let context_for_embed = req.context.clone();
    let memory_for_embedding = memory_system.clone();
    let context_embedding: Vec<f32> = tokio::task::spawn_blocking(move || {
        let memory_guard = memory_for_embedding.read();
        memory_guard
            .compute_embedding(&context_for_embed)
            .unwrap_or_else(|_| vec![0.0; 384])
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?;

    let user_id = req.user_id.clone();
    let context_for_triggers = req.context.clone();
    let memory_for_task_embed = memory_system.clone();
    let prospective = state.prospective_store.clone();
    let mark_triggered = req.mark_triggered;

    let matched_tasks: Vec<(crate::memory::types::ProspectiveTask, f32)> =
        tokio::task::spawn_blocking(move || {
            let embed_fn = |text: &str| -> Option<Vec<f32>> {
                let memory_guard = memory_for_task_embed.read();
                memory_guard.compute_embedding(text).ok()
            };

            prospective
                .check_context_triggers_semantic(
                    &user_id,
                    &context_for_triggers,
                    &context_embedding,
                    embed_fn,
                )
                .unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?;

    // Mark triggered and re-read actual state (same C1+C2+C3 fixes as get_due_reminders)
    let tasks_with_scores: Vec<(ProspectiveTask, f32)> = if mark_triggered {
        let mut result = Vec::with_capacity(matched_tasks.len());
        for (task, score) in &matched_tasks {
            match state
                .prospective_store
                .mark_triggered(&req.user_id, &task.id)
            {
                Ok(true) => match state.prospective_store.get(&req.user_id, &task.id) {
                    Ok(Some(updated)) => result.push((updated, *score)),
                    _ => result.push((task.clone(), *score)),
                },
                Ok(false) => {
                    tracing::debug!(task_id = %task.id, "Context reminder already triggered (concurrent)");
                }
                Err(e) => {
                    tracing::warn!(task_id = %task.id, error = %e, "Failed to mark context reminder triggered");
                    result.push((task.clone(), *score));
                }
            }
        }
        result
    } else {
        matched_tasks
    };

    let reminders: Vec<ReminderItem> = tasks_with_scores
        .into_iter()
        .map(|(t, score)| ReminderItem {
            id: t.id.to_string(),
            content: t.content,
            trigger_type: format!("context (score: {:.2})", score),
            status: format!("{:?}", t.status).to_lowercase(),
            due_at: None,
            created_at: t.created_at,
            triggered_at: t.triggered_at,
            dismissed_at: t.dismissed_at,
            priority: t.priority,
            tags: t.tags,
            overdue_seconds: None,
        })
        .collect();

    let count = reminders.len();

    if count > 0 {
        tracing::debug!(
            user_id = %req.user_id,
            count = count,
            context_preview = %req.context.chars().take(50).collect::<String>(),
            "Context-triggered reminders matched"
        );
    }

    Ok(Json(DueRemindersResponse { reminders, count }))
}

/// Dismiss (acknowledge) a triggered reminder
pub async fn dismiss_reminder(
    State(state): State<AppState>,
    Path(reminder_id): Path<String>,
    Json(req): Json<DismissReminderRequest>,
) -> Result<Json<ReminderActionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let task_id = if let Ok(uuid) = uuid::Uuid::parse_str(&reminder_id) {
        ProspectiveTaskId(uuid)
    } else {
        let task = state
            .prospective_store
            .find_by_prefix(&req.user_id, &reminder_id)
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::InvalidInput {
                field: "reminder_id".to_string(),
                reason: format!("No reminder found with ID prefix '{}'", reminder_id),
            })?;
        task.id
    };

    let success = state
        .prospective_store
        .mark_dismissed(&req.user_id, &task_id)
        .map_err(AppError::Internal)?;

    if success {
        tracing::info!(
            user_id = %req.user_id,
            reminder_id = %task_id.0,
            "Dismissed reminder"
        );
    }

    Ok(Json(ReminderActionResponse {
        success,
        message: if success {
            "Reminder dismissed".to_string()
        } else {
            "Reminder not found".to_string()
        },
    }))
}

/// Delete (cancel) a reminder
pub async fn delete_reminder(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(reminder_id): Path<String>,
    Query(query): Query<DeleteReminderQuery>,
) -> Result<Json<ReminderActionResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let task_id = if let Ok(uuid) = uuid::Uuid::parse_str(&reminder_id) {
        ProspectiveTaskId(uuid)
    } else {
        let task = state
            .prospective_store
            .find_by_prefix(&user_id, &reminder_id)
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::InvalidInput {
                field: "reminder_id".to_string(),
                reason: format!("No reminder found with ID prefix '{}'", reminder_id),
            })?;
        task.id
    };

    let success = state
        .prospective_store
        .delete(&user_id, &task_id)
        .map_err(AppError::Internal)?;

    if success {
        tracing::info!(
            user_id = %user_id,
            reminder_id = %task_id.0,
            "Deleted reminder"
        );
    }

    Ok(Json(ReminderActionResponse {
        success,
        message: if success {
            "Reminder deleted".to_string()
        } else {
            "Reminder not found".to_string()
        },
    }))
}

// =============================================================================
// TODO HANDLERS
// =============================================================================

/// POST /api/todos - Create a new todo
pub async fn create_todo(
    State(state): State<AppState>,
    Json(req): Json<CreateTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Content cannot be empty".to_string(),
        });
    }

    let mut todo = Todo::new(req.user_id.clone(), req.content.clone());

    if let Some(ref status_str) = req.status {
        todo.status = parse_todo_status(status_str)?;
    }

    if let Some(ref priority_str) = req.priority {
        todo.priority = parse_todo_priority(priority_str)?;
    }

    let mut project_name = None;
    if let Some(ref proj_name) = req.project {
        let project = state
            .todo_store
            .find_or_create_project(&req.user_id, proj_name)
            .map_err(AppError::Internal)?;
        todo.project_id = Some(project.id.clone());
        project_name = Some(project.name.clone());
    }

    if let Some(contexts) = req.contexts {
        todo.contexts = contexts;
    } else {
        todo.contexts = todo_formatter::extract_contexts(&req.content);
    }

    if let Some(ref due_str) = req.due_date {
        todo.due_date = todo_formatter::parse_due_date(due_str);
    }

    todo.blocked_on = req.blocked_on;

    if let Some(ref parent_str) = req.parent_id {
        if let Some(parent) = state
            .todo_store
            .find_todo_by_prefix(&req.user_id, parent_str)
            .map_err(AppError::Internal)?
        {
            todo.parent_id = Some(parent.id);
            if todo.project_id.is_none() {
                todo.project_id = parent.project_id;
                if let Some(ref proj_id) = todo.project_id {
                    if let Ok(Some(proj)) = state.todo_store.get_project(&req.user_id, proj_id) {
                        project_name = Some(proj.name.clone());
                    }
                }
            }
        }
    }

    todo.tags = req.tags.unwrap_or_default();
    todo.notes = req.notes;
    todo.external_id = req.external_id;

    if let Some(ref recurrence_str) = req.recurrence {
        todo.recurrence = Some(parse_recurrence(recurrence_str)?);
    }

    // A new todo has no dependents yet, so its dependencies cannot form a cycle.
    if let Some(ref dep_ids) = req.depends_on {
        for dep_id in resolve_todo_refs(&state, &req.user_id, dep_ids, "depends_on")? {
            todo.add_dependency(dep_id);
        }
    }

    if let Some(ref memory_ids) = req.related_memory_ids {
        todo.related_memory_ids = verify_memory_ids(&state, &req.user_id, memory_ids).await?;
    }

    // Compute embedding for semantic search
    let embedding_text = format!(
        "{} {} {}",
        todo.content,
        todo.notes.as_deref().unwrap_or(""),
        todo.tags.join(" ")
    );

    if let Ok(memory_system) = state.get_user_memory(&req.user_id) {
        let memory_clone = memory_system.clone();
        let embedding_text_clone = embedding_text.clone();

        if let Ok(embedding) = tokio::task::spawn_blocking(move || {
            let memory_guard = memory_clone.read();
            memory_guard.compute_embedding(&embedding_text_clone)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?
        {
            todo.embedding = Some(embedding);
        }
    }

    let todo = state
        .todo_store
        .store_todo(&todo)
        .map_err(AppError::Internal)?;

    sync_memory_back_links(
        &state,
        &req.user_id,
        &todo.id,
        todo.related_memory_ids.clone(),
        Vec::new(),
    )
    .await?;

    let activity_msg = if let Some(ref proj) = project_name {
        format!("Created in project '{}'", proj)
    } else {
        "Created".to_string()
    };
    let _ = state
        .todo_store
        .add_activity(&req.user_id, &todo.id, activity_msg);

    // Create memory from todo
    let memory_content = if let Some(ref proj) = project_name {
        format!(
            "[{}] Todo created in {}: {}",
            todo.short_id(),
            proj,
            todo.content
        )
    } else {
        format!("[{}] Todo created: {}", todo.short_id(), todo.content)
    };

    let mut tags = vec![
        format!("todo:{}", todo.short_id()),
        "todo-created".to_string(),
    ];
    if let Some(ref proj) = project_name {
        tags.push(format!("project:{}", proj));
    }

    let experience = Experience {
        content: memory_content,
        experience_type: ExperienceType::Task,
        tags,
        ..Default::default()
    };

    if let Ok(memory) = state.get_user_memory(&req.user_id) {
        let memory_clone = memory.clone();
        let exp_clone = experience.clone();
        let state_clone = state.clone();
        let user_id = req.user_id.clone();
        let todo_id = todo.id;

        state.task_tracker.spawn(async move {
            let memory_result = tokio::task::spawn_blocking(move || {
                let memory_guard = memory_clone.read();
                memory_guard.remember(exp_clone, None)
            })
            .await;

            if let Ok(Ok(memory_id)) = memory_result {
                if let Err(e) =
                    state_clone.process_experience_into_graph(&user_id, &experience, &memory_id)
                {
                    tracing::debug!(
                        "Graph processing failed for todo memory {}: {}",
                        memory_id.0,
                        e
                    );
                }
                link_todo_memory(&state_clone, &user_id, &todo_id, &memory_id);
                tracing::debug!(memory_id = %memory_id.0, "Todo creation stored as memory");
            }
        });
    }

    let formatted = todo_formatter::format_todo_created(&todo, project_name.as_deref());

    state.emit_event(MemoryEvent {
        event_type: "TODO_CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(todo.id.0.to_string()),
        content_preview: Some(todo.content.clone()),
        memory_type: Some(format!("{:?}", todo.status)),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    let session_id = state.session_store.get_or_create_session(&req.user_id);
    state.session_store.add_event(
        &session_id,
        SessionEvent::TodoCreated {
            timestamp: chrono::Utc::now(),
            todo_id: todo.id.0.to_string(),
            content: todo.content.chars().take(100).collect(),
            project: project_name.clone(),
        },
    );

    tracing::info!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        seq_num = todo.seq_num,
        content = %req.content,
        "Created todo"
    );

    state.log_event(
        &req.user_id,
        "TODO_CREATE",
        &todo.id.0.to_string(),
        &format!(
            "Created todo [{}] project={}: '{}'",
            todo.short_id(),
            project_name.as_deref().unwrap_or("none"),
            req.content.chars().take(50).collect::<String>()
        ),
    );

    Ok(Json(TodoResponse {
        success: true,
        todo: Some(todo),
        project: None,
        formatted,
    }))
}

/// POST /api/todos/list - List todos with filters
pub async fn list_todos(
    State(state): State<AppState>,
    Json(req): Json<ListTodosRequest>,
) -> Result<Json<TodoListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let status_filter: Option<Vec<TodoStatus>> = req
        .status
        .as_ref()
        .map(|statuses| statuses.iter().map(|s| parse_todo_status(s)).collect())
        .transpose()?;

    let mut todos = if let Some(ref query) = req.query {
        if query.trim().is_empty() {
            Vec::new()
        } else {
            // Semantic and lexical: an exact word match surfaces even when the embedder is
            // down or the todo predates embeddings.
            let query_embedding: Option<Vec<f32>> =
                if let Ok(memory_system) = state.get_user_memory(&req.user_id) {
                    let query_clone = query.clone();
                    tokio::task::spawn_blocking(move || {
                        let memory_guard = memory_system.read();
                        memory_guard.compute_query_embedding(&query_clone).ok()
                    })
                    .await
                    .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding failed: {e}")))?
                } else {
                    None
                };

            let limit = req.limit.unwrap_or(50);
            state
                .todo_store
                .search_todos(&req.user_id, query, query_embedding.as_deref(), limit * 2)
                .map_err(AppError::Internal)?
                .into_iter()
                .map(|(todo, _score)| todo)
                .collect()
        }
    } else if let Some(ref statuses) = status_filter {
        state
            .todo_store
            .list_todos_for_user(&req.user_id, Some(statuses))
            .map_err(AppError::Internal)?
    } else {
        let include_completed = req.include_completed.unwrap_or(false);
        let all_todos = state
            .todo_store
            .list_todos_for_user(&req.user_id, None)
            .map_err(AppError::Internal)?;

        if include_completed {
            all_todos
        } else {
            all_todos
                .into_iter()
                .filter(|t| t.status != TodoStatus::Done && t.status != TodoStatus::Cancelled)
                .collect()
        }
    };

    // Apply status filter for semantic search results
    if req.query.is_some() {
        if let Some(ref statuses) = status_filter {
            todos.retain(|t| statuses.contains(&t.status));
        } else if !req.include_completed.unwrap_or(false) {
            todos.retain(|t| t.status != TodoStatus::Done && t.status != TodoStatus::Cancelled);
        }
    }

    // Filter by project
    if let Some(ref proj_name) = req.project {
        if let Some(project) = state
            .todo_store
            .find_project_by_name(&req.user_id, proj_name)
            .map_err(AppError::Internal)?
        {
            todos.retain(|t| t.project_id.as_ref() == Some(&project.id));
        }
    }

    // Filter by context
    if let Some(ref ctx) = req.context {
        let ctx_lower = ctx.to_lowercase();
        todos.retain(|t| t.contexts.iter().any(|c| c.to_lowercase() == ctx_lower));
    }

    // Filter by parent_id
    if let Some(ref parent_str) = req.parent_id {
        if let Some(parent) = state
            .todo_store
            .find_todo_by_prefix(&req.user_id, parent_str)
            .map_err(AppError::Internal)?
        {
            todos.retain(|t| t.parent_id.as_ref() == Some(&parent.id));
        }
    }

    // Filter by due date
    if let Some(ref due_filter) = req.due {
        let now = chrono::Utc::now();
        let end_of_today = now
            .date_naive()
            .and_hms_opt(23, 59, 59)
            .map(|t| t.and_utc())
            .unwrap_or(now);
        let end_of_week =
            now + chrono::Duration::days(7 - now.weekday().num_days_from_monday() as i64);

        match due_filter.to_lowercase().as_str() {
            "today" => {
                todos.retain(|t| {
                    t.due_date
                        .as_ref()
                        .map(|d| *d <= end_of_today || *d < now)
                        .unwrap_or(false)
                });
            }
            "overdue" => {
                todos.retain(|t| t.is_overdue());
            }
            "this_week" => {
                todos.retain(|t| {
                    t.due_date
                        .as_ref()
                        .map(|d| *d <= end_of_week)
                        .unwrap_or(false)
                });
            }
            "all" => {}
            _ => {
                return Err(AppError::InvalidInput {
                    field: "due".to_string(),
                    reason: format!(
                        "Unknown due filter '{due_filter}'. Valid values: today, overdue, this_week, \
                         all"
                    ),
                });
            }
        }
    }

    // Filter by priority
    if let Some(ref priority_str) = req.priority {
        let target_priority = parse_todo_priority(priority_str)?;
        todos.retain(|t| t.priority == target_priority);
    }

    // Apply pagination
    let total_count = todos.len();
    let offset = req.offset.unwrap_or(0);
    let limit = req.limit.unwrap_or(100);

    if offset > 0 && offset < todos.len() {
        todos = todos.into_iter().skip(offset).collect();
    } else if offset >= total_count {
        todos.clear();
    }

    if todos.len() > limit {
        todos.truncate(limit);
    }

    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_todo_list_with_total(&todos, &projects, total_count);

    Ok(Json(TodoListResponse {
        success: true,
        count: total_count,
        todos,
        projects,
        formatted,
    }))
}

/// POST /api/todos/due - List due/overdue todos
pub async fn list_due_todos(
    State(state): State<AppState>,
    Json(req): Json<DueTodosRequest>,
) -> Result<Json<TodoListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let todos = state
        .todo_store
        .list_due_todos(&req.user_id, req.include_overdue)
        .map_err(AppError::Internal)?;

    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_due_todos(&todos);

    Ok(Json(TodoListResponse {
        success: true,
        count: todos.len(),
        todos,
        projects,
        formatted,
    }))
}

/// GET /api/todos/{todo_id} - Get a single todo
pub async fn get_todo(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let project_name = if let Some(ref pid) = todo.project_id {
        state
            .todo_store
            .get_project(&user_id, pid)
            .map_err(AppError::Internal)?
            .map(|p| p.name)
    } else {
        None
    };

    let mut formatted = todo_formatter::format_todo_line(&todo, project_name.as_deref(), true);
    if !todo.depends_on.is_empty() {
        let deps: Vec<String> = todo
            .depends_on
            .iter()
            .map(|id| match state.todo_store.get_todo(&user_id, id) {
                Ok(Some(dep)) => {
                    format!(
                        "{} ({})",
                        dep.short_id(),
                        format!("{:?}", dep.status).to_lowercase()
                    )
                }
                _ => format!("{} (deleted)", id.short()),
            })
            .collect();
        formatted.push_str(&format!("\n  Depends on: {}", deps.join(", ")));
    }
    if !todo.related_memory_ids.is_empty() {
        let ids: Vec<String> = todo
            .related_memory_ids
            .iter()
            .map(|m| m.0.to_string())
            .collect();
        formatted.push_str(&format!("\n  Linked memories: {}", ids.join(", ")));
    }

    Ok(Json(TodoResponse {
        success: true,
        todo: Some(todo),
        project: None,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/update - Update a todo
pub async fn update_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<UpdateTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    update_todo_core(state, todo_id, req).await
}

/// POST /api/todos/update - Update a todo named in the body
pub async fn update_todo_flat(
    State(state): State<AppState>,
    Json(req): Json<UpdateTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    let todo_id = flat_todo_id(req.todo_id.clone())?;
    update_todo_core(state, todo_id, req).await
}

async fn update_todo_core(
    state: AppState,
    todo_id: String,
    req: UpdateTodoRequest,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let mut todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;
    let links_before = todo.related_memory_ids.clone();

    if let Some(ref content) = req.content {
        todo.content = content.clone();
    }
    if let Some(ref status_str) = req.status {
        todo.status = parse_todo_status(status_str)?;
    }
    if let Some(ref priority_str) = req.priority {
        todo.priority = parse_todo_priority(priority_str)?;
    }
    if let Some(ref contexts) = req.contexts {
        todo.contexts = contexts.clone();
    }
    if let Some(ref due_str) = req.due_date {
        todo.due_date = todo_formatter::parse_due_date(due_str);
    }
    if let Some(ref blocked) = req.blocked_on {
        todo.blocked_on = Some(blocked.clone());
    }
    if let Some(ref notes) = req.notes {
        todo.notes = Some(notes.clone());
    }
    if let Some(ref tags) = req.tags {
        todo.tags = tags.clone();
    }
    if let Some(ref external_id) = req.external_id {
        todo.external_id = Some(external_id.clone());
    }
    if let Some(ref parent_id_str) = req.parent_id {
        if parent_id_str.is_empty() {
            todo.parent_id = None;
        } else if let Ok(Some(parent)) = state
            .todo_store
            .find_todo_by_prefix(&req.user_id, parent_id_str)
        {
            todo.parent_id = Some(parent.id);
        }
    }

    // Resolve structured dependencies: replace semantics, unknown references and cycles
    // rejected.
    if let Some(ref dep_refs) = req.depends_on {
        let resolved = resolve_todo_refs(&state, &req.user_id, dep_refs, "depends_on")?;
        todo.depends_on.clear();
        for (dep_ref, dep_id) in dep_refs.iter().zip(resolved) {
            if state
                .todo_store
                .would_create_cycle(&req.user_id, &todo.id, &dep_id)
                .map_err(AppError::Internal)?
            {
                return Err(AppError::InvalidInput {
                    field: "depends_on".to_string(),
                    reason: format!("Adding dependency on {dep_ref} would create a cycle"),
                });
            }
            todo.add_dependency(dep_id);
        }
    }

    if let Some(ref memory_ids) = req.related_memory_ids {
        todo.related_memory_ids = verify_memory_ids(&state, &req.user_id, memory_ids).await?;
    }

    let mut project_name = None;
    if let Some(ref proj_name) = req.project {
        let project = state
            .todo_store
            .find_or_create_project(&req.user_id, proj_name)
            .map_err(AppError::Internal)?;
        todo.project_id = Some(project.id.clone());
        project_name = Some(project.name.clone());
    }

    todo.updated_at = chrono::Utc::now();

    // The embedding is recomputed before the one write, so the todo is stored once with it.
    if req.content.is_some() || req.notes.is_some() || req.tags.is_some() {
        let embedding_text = format!(
            "{} {} {}",
            todo.content,
            todo.notes.as_deref().unwrap_or(""),
            todo.tags.join(" ")
        );

        if let Ok(memory_system) = state.get_user_memory(&req.user_id) {
            let memory_clone = memory_system.clone();
            let embedding_text_clone = embedding_text.clone();

            if let Ok(embedding) = tokio::task::spawn_blocking(move || {
                let memory_guard = memory_clone.read();
                memory_guard.compute_embedding(&embedding_text_clone)
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?
            {
                todo.embedding = Some(embedding);
            }
        }
    }

    state
        .todo_store
        .update_todo(&todo)
        .map_err(AppError::Internal)?;

    let added: Vec<MemoryId> = todo
        .related_memory_ids
        .iter()
        .filter(|id| !links_before.contains(id))
        .cloned()
        .collect();
    let removed: Vec<MemoryId> = links_before
        .iter()
        .filter(|id| !todo.related_memory_ids.contains(id))
        .cloned()
        .collect();
    sync_memory_back_links(&state, &req.user_id, &todo.id, added, removed).await?;

    let update_description = {
        let mut changes = Vec::new();
        if req.status.is_some() {
            changes.push(format!("status → {:?}", todo.status));
        }
        if req.priority.is_some() {
            changes.push(format!("priority → {:?}", todo.priority));
        }
        if req.content.is_some() {
            changes.push("content updated".to_string());
        }
        if req.project.is_some() {
            changes.push(format!(
                "project → {}",
                project_name.as_deref().unwrap_or("none")
            ));
        }
        if req.blocked_on.is_some() {
            changes.push(format!(
                "blocked on: {}",
                todo.blocked_on.as_deref().unwrap_or("cleared")
            ));
        }
        if req.depends_on.is_some() {
            let dep_display: Vec<String> = todo
                .depends_on
                .iter()
                .filter_map(|id| {
                    state
                        .todo_store
                        .get_todo(&req.user_id, id)
                        .ok()
                        .flatten()
                        .map(|t| t.short_id())
                })
                .collect();
            if dep_display.is_empty() {
                changes.push("dependencies cleared".to_string());
            } else {
                changes.push(format!("depends on: {}", dep_display.join(", ")));
            }
        }
        changes.join(", ")
    };

    if !update_description.is_empty() {
        let _ = state.todo_store.add_activity(
            &req.user_id,
            &todo.id,
            format!("Updated: {}", update_description),
        );
    }

    if !update_description.is_empty() {
        let memory_content = format!(
            "[{}] Todo updated ({}): {}",
            todo.short_id(),
            update_description,
            todo.content
        );

        let mut tags = vec![
            format!("todo:{}", todo.short_id()),
            "todo-updated".to_string(),
        ];
        if let Some(ref proj) = project_name {
            tags.push(format!("project:{}", proj));
        }
        if req.status.is_some() {
            tags.push(format!("status:{:?}", todo.status).to_lowercase());
        }

        let experience = Experience {
            content: memory_content,
            experience_type: ExperienceType::Context,
            tags,
            ..Default::default()
        };

        if let Ok(memory) = state.get_user_memory(&req.user_id) {
            let memory_clone = memory.clone();
            let exp_clone = experience.clone();
            let state_clone = state.clone();
            let user_id = req.user_id.clone();
            let todo_id = todo.id;

            state.task_tracker.spawn(async move {
                let memory_result = tokio::task::spawn_blocking(move || {
                    let memory_guard = memory_clone.read();
                    memory_guard.remember(exp_clone, None)
                })
                .await;

                if let Ok(Ok(memory_id)) = memory_result {
                    if let Err(e) =
                        state_clone.process_experience_into_graph(&user_id, &experience, &memory_id)
                    {
                        tracing::debug!(
                            "Graph processing failed for todo update memory {}: {}",
                            memory_id.0,
                            e
                        );
                    }
                    link_todo_memory(&state_clone, &user_id, &todo_id, &memory_id);
                    tracing::debug!(memory_id = %memory_id.0, "Todo update stored as memory");
                }
            });
        }
    }

    let formatted = todo_formatter::format_todo_updated(&todo, project_name.as_deref());

    state.emit_event(MemoryEvent {
        event_type: "TODO_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(todo.id.0.to_string()),
        content_preview: Some(todo.content.clone()),
        memory_type: Some(format!("{:?}", todo.status)),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        "Updated todo"
    );

    state.log_event(
        &req.user_id,
        "TODO_UPDATE",
        &todo.id.0.to_string(),
        &format!(
            "Updated todo [{}]: {}",
            todo.short_id(),
            if update_description.is_empty() {
                "no changes"
            } else {
                &update_description
            }
        ),
    );

    Ok(Json(TodoResponse {
        success: true,
        todo: Some(todo),
        project: None,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/complete - Mark todo as complete
pub async fn complete_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<TodoQuery>,
) -> Result<Json<TodoCompleteResponse>, AppError> {
    complete_todo_core(state, todo_id, req.user_id).await
}

/// POST /api/todos/complete - Complete a todo named in the body
pub async fn complete_todo_flat(
    State(state): State<AppState>,
    Json(req): Json<FlatTodoRequest>,
) -> Result<Json<TodoCompleteResponse>, AppError> {
    let todo_id = flat_todo_id(Some(req.todo_id))?;
    complete_todo_core(state, todo_id, req.user_id).await
}

async fn complete_todo_core(
    state: AppState,
    todo_id: String,
    user_id: String,
) -> Result<Json<TodoCompleteResponse>, AppError> {
    let req = TodoQuery { user_id };
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let result = state
        .todo_store
        .complete_todo(&req.user_id, &todo.id)
        .map_err(AppError::Internal)?;

    if result.is_some() {
        let days_taken = (chrono::Utc::now() - todo.created_at).num_hours() as f64 / 24.0;
        let activity_msg = format!("Marked complete after {:.1} days", days_taken);
        let _ = state
            .todo_store
            .add_activity(&req.user_id, &todo.id, activity_msg.clone());

        let memory_content = format!(
            "[{}] Todo completed: {} (took {:.1} days)",
            todo.short_id(),
            todo.content,
            days_taken
        );

        let mut tags = vec![
            format!("todo:{}", todo.short_id()),
            "todo-completed".to_string(),
            "completion".to_string(),
        ];
        if let Some(ref project_id) = todo.project_id {
            if let Ok(Some(project)) = state.todo_store.get_project(&req.user_id, project_id) {
                tags.push(format!("project:{}", project.name));
            }
        }

        let experience = Experience {
            content: memory_content,
            experience_type: ExperienceType::Task,
            tags,
            ..Default::default()
        };

        if let Ok(memory) = state.get_user_memory(&req.user_id) {
            let memory_clone = memory.clone();
            let exp_clone = experience.clone();
            let state_clone = state.clone();
            let user_id = req.user_id.clone();
            let todo_id = todo.id;

            state.task_tracker.spawn(async move {
                let memory_result = tokio::task::spawn_blocking(move || {
                    let memory_guard = memory_clone.read();
                    memory_guard.remember(exp_clone, None)
                })
                .await;

                if let Ok(Ok(memory_id)) = memory_result {
                    if let Err(e) =
                        state_clone.process_experience_into_graph(&user_id, &experience, &memory_id)
                    {
                        tracing::debug!(
                            "Graph processing failed for todo completion memory {}: {}",
                            memory_id.0,
                            e
                        );
                    }
                    link_todo_memory(&state_clone, &user_id, &todo_id, &memory_id);
                    tracing::debug!(memory_id = %memory_id.0, "Todo completion stored as searchable memory");
                }
            });
        }
    }

    match result {
        Some((completed, next)) => {
            // The next occurrence of a recurring todo copies its memory links, so its own id
            // needs the back-links too.
            if let Some(ref next_todo) = next {
                sync_memory_back_links(
                    &state,
                    &req.user_id,
                    &next_todo.id,
                    next_todo.related_memory_ids.clone(),
                    Vec::new(),
                )
                .await?;
            }
            let unblocked = state
                .todo_store
                .unblocked_by_completion(&req.user_id, &completed.id)
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "Failed to compute unblocked todos");
                    Vec::new()
                });
            let mut formatted = todo_formatter::format_todo_completed(&completed, next.as_ref());
            if !unblocked.is_empty() {
                let ids: Vec<String> = unblocked.iter().map(|t| t.short_id()).collect();
                formatted.push_str(&format!("\n\n  → Unblocked: {}", ids.join(", ")));
            }

            state.emit_event(MemoryEvent {
                event_type: "TODO_COMPLETE".to_string(),
                timestamp: chrono::Utc::now(),
                user_id: req.user_id.clone(),
                memory_id: Some(completed.id.0.to_string()),
                content_preview: Some(completed.content.clone()),
                memory_type: Some("Done".to_string()),
                importance: None,
                count: None,
                entities: None,
                results: None,
            });

            let session_id = state.session_store.get_or_create_session(&req.user_id);
            state.session_store.add_event(
                &session_id,
                SessionEvent::TodoCompleted {
                    timestamp: chrono::Utc::now(),
                    todo_id: completed.id.0.to_string(),
                },
            );

            tracing::info!(
                user_id = %req.user_id,
                todo_id = %completed.id,
                has_next = next.is_some(),
                "Completed todo"
            );

            state.log_event(
                &req.user_id,
                "TODO_COMPLETE",
                &completed.id.0.to_string(),
                &format!(
                    "Completed todo [{}]: '{}' (recurrence={})",
                    completed.short_id(),
                    completed.content.chars().take(40).collect::<String>(),
                    next.is_some()
                ),
            );

            Ok(Json(TodoCompleteResponse {
                success: true,
                todo: Some(completed),
                next_recurrence: next,
                unblocked,
                formatted,
            }))
        }
        None => Err(AppError::TodoNotFound(todo_id)),
    }
}

/// DELETE /api/todos/{todo_id} - Delete a todo
pub async fn delete_todo(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    delete_todo_core(state, todo_id, user_id).await
}

/// POST /api/todos/delete - Delete a todo named in the body
pub async fn delete_todo_flat(
    State(state): State<AppState>,
    Json(req): Json<FlatTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    let todo_id = flat_todo_id(Some(req.todo_id))?;
    delete_todo_core(state, todo_id, req.user_id).await
}

async fn delete_todo_core(
    state: AppState,
    todo_id: String,
    user_id: String,
) -> Result<Json<TodoResponse>, AppError> {
    let query = TodoQuery { user_id };
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&query.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let success = state
        .todo_store
        .delete_todo(&query.user_id, &todo.id)
        .map_err(AppError::Internal)?;
    if success {
        sync_memory_back_links(
            &state,
            &query.user_id,
            &todo.id,
            Vec::new(),
            todo.related_memory_ids.clone(),
        )
        .await?;
    }

    let formatted = if success {
        todo_formatter::format_todo_deleted(&todo.short_id())
    } else {
        "Todo not found".to_string()
    };

    if success {
        state.emit_event(MemoryEvent {
            event_type: "TODO_DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: query.user_id.clone(),
            memory_id: Some(todo.id.0.to_string()),
            content_preview: Some(todo.content.clone()),
            memory_type: None,
            importance: None,
            count: None,
            entities: None,
            results: None,
        });

        tracing::info!(
            user_id = %query.user_id,
            todo_id = %todo.id,
            "Deleted todo"
        );
    }

    Ok(Json(TodoResponse {
        success,
        todo: None,
        project: None,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/reorder - Move todo up/down
pub async fn reorder_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<ReorderTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    reorder_todo_core(state, todo_id, req).await
}

/// POST /api/todos/reorder - Reorder a todo named in the body
pub async fn reorder_todo_flat(
    State(state): State<AppState>,
    Json(req): Json<ReorderTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    let todo_id = flat_todo_id(req.todo_id.clone())?;
    reorder_todo_core(state, todo_id, req).await
}

async fn reorder_todo_core(
    state: AppState,
    todo_id: String,
    req: ReorderTodoRequest,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    if req.direction != "up" && req.direction != "down" {
        return Err(AppError::InvalidInput {
            field: "direction".to_string(),
            reason: format!(
                "Unknown direction '{}'. Valid values: up, down",
                req.direction
            ),
        });
    }

    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let result = state
        .todo_store
        .reorder_todo(&req.user_id, &todo.id, &req.direction)
        .map_err(AppError::Internal)?;

    match result {
        Some(updated) => {
            let formatted = format!("Moved {} {}", updated.short_id(), req.direction);

            state.emit_event(MemoryEvent {
                event_type: "TODO_REORDER".to_string(),
                timestamp: chrono::Utc::now(),
                user_id: req.user_id.clone(),
                memory_id: Some(updated.id.0.to_string()),
                content_preview: Some(updated.content.clone()),
                memory_type: Some(format!("{:?}", updated.status)),
                importance: None,
                count: None,
                entities: None,
                results: None,
            });

            tracing::debug!(
                user_id = %req.user_id,
                todo_id = %updated.id,
                direction = %req.direction,
                "Reordered todo"
            );

            Ok(Json(TodoResponse {
                success: true,
                todo: Some(updated),
                project: None,
                formatted,
            }))
        }
        None => Err(AppError::TodoNotFound(todo_id)),
    }
}

/// GET /api/todos/{todo_id}/subtasks - List subtasks of a parent todo
pub async fn list_subtasks(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoListResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let parent = state
        .todo_store
        .find_todo_by_prefix(&user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let subtasks = state
        .todo_store
        .list_subtasks(&parent.id)
        .map_err(AppError::Internal)?;

    let projects = state
        .todo_store
        .list_projects(&user_id)
        .map_err(AppError::Internal)?;

    let formatted = if subtasks.is_empty() {
        format!("No subtasks for {}", parent.short_id())
    } else {
        let mut output = format!(
            "🌾━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
             ┃  SUBTASKS OF {}  ┃\n\
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n",
            parent.short_id()
        );
        output.push_str(&todo_formatter::format_todo_list(&subtasks, &projects));
        output
    };

    tracing::debug!(
        user_id = %user_id,
        parent_id = %parent.id,
        count = subtasks.len(),
        "Listed subtasks"
    );

    Ok(Json(TodoListResponse {
        success: true,
        count: subtasks.len(),
        todos: subtasks,
        projects,
        formatted,
    }))
}

/// POST /api/todos/ready - List todos that are ready to work on (all deps satisfied)
pub async fn list_ready_todos(
    State(state): State<AppState>,
    Json(req): Json<ListTodosRequest>,
) -> Result<Json<TodoListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let ready = state
        .todo_store
        .list_ready_todos(&req.user_id)
        .map_err(AppError::Internal)?;

    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = if ready.is_empty() {
        "No ready todos (all tasks have unresolved dependencies)".to_string()
    } else {
        let mut output = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
             ┃  READY (all dependencies satisfied)  ┃\n\
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            .to_string();
        output.push_str(&todo_formatter::format_todo_list(&ready, &projects));
        output
    };

    Ok(Json(TodoListResponse {
        success: true,
        count: ready.len(),
        todos: ready,
        projects,
        formatted,
    }))
}

/// GET /api/todos/{todo_id}/dependents - List todos that depend on this todo
pub async fn list_dependents(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoListResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let dependents = state
        .todo_store
        .list_dependents(&todo.id)
        .map_err(AppError::Internal)?;

    let projects = state
        .todo_store
        .list_projects(&user_id)
        .map_err(AppError::Internal)?;

    let formatted = if dependents.is_empty() {
        format!("Nothing depends on {}", todo.short_id())
    } else {
        let mut output = format!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
             ┃  DEPENDENTS OF {}  ┃\n\
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n",
            todo.short_id()
        );
        output.push_str(&todo_formatter::format_todo_list(&dependents, &projects));
        output
    };

    Ok(Json(TodoListResponse {
        success: true,
        count: dependents.len(),
        todos: dependents,
        projects,
        formatted,
    }))
}

/// Params for dependency chain walk
#[derive(Debug, Deserialize)]
pub struct DependencyChainQuery {
    pub user_id: String,
    #[serde(default = "default_direction")]
    pub direction: String,
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
}

fn default_direction() -> String {
    "forward".to_string()
}
fn default_max_depth() -> usize {
    10
}

/// GET /api/todos/{todo_id}/dependency_chain - Walk the dependency graph
pub async fn dependency_chain(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(todo_id): Path<String>,
    Query(query): Query<DependencyChainQuery>,
) -> Result<Json<TodoListResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let chain = if query.direction == "backward" {
        state
            .todo_store
            .dependency_chain_backward(&user_id, &todo.id, query.max_depth)
            .map_err(AppError::Internal)?
    } else {
        state
            .todo_store
            .dependency_chain_forward(&user_id, &todo.id, query.max_depth)
            .map_err(AppError::Internal)?
    };

    let chain_todos: Vec<Todo> = chain.into_iter().map(|(t, _)| t).collect();

    let projects = state
        .todo_store
        .list_projects(&user_id)
        .map_err(AppError::Internal)?;

    let label = if query.direction == "backward" {
        "BLOCKS (upstream)"
    } else {
        "UNBLOCKS (downstream)"
    };

    let formatted = if chain_todos.is_empty() {
        format!(
            "No {} dependencies for {}",
            query.direction,
            todo.short_id()
        )
    } else {
        let mut output = format!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
             ┃  {} — {}  ┃\n\
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n",
            label,
            todo.short_id()
        );
        output.push_str(&todo_formatter::format_todo_list(&chain_todos, &projects));
        output
    };

    Ok(Json(TodoListResponse {
        success: true,
        count: chain_todos.len(),
        todos: chain_todos,
        projects,
        formatted,
    }))
}

/// POST /api/todos/stats - Get todo statistics
pub async fn get_todo_stats(
    State(state): State<AppState>,
    Json(req): Json<TodoStatsRequest>,
) -> Result<Json<TodoStatsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let stats = state
        .todo_store
        .get_user_stats(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_user_stats(&stats);

    Ok(Json(TodoStatsResponse {
        success: true,
        stats,
        formatted,
    }))
}

// =============================================================================
// COMMENT HANDLERS
// =============================================================================

/// POST /api/todos/{todo_id}/comments - Add a comment to a todo
pub async fn add_todo_comment(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<AddCommentRequest>,
) -> Result<Json<CommentResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Comment content cannot be empty".to_string(),
        });
    }

    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let comment_type = match req.comment_type.as_deref() {
        None => None,
        Some(ct) => Some(match ct.to_lowercase().as_str() {
            "comment" => TodoCommentType::Comment,
            "progress" => TodoCommentType::Progress,
            "resolution" => TodoCommentType::Resolution,
            "activity" => TodoCommentType::Activity,
            _ => {
                return Err(AppError::InvalidInput {
                    field: "comment_type".to_string(),
                    reason: format!(
                        "Unknown comment type '{ct}'. Valid values: comment, progress, \
                         resolution, activity"
                    ),
                });
            }
        }),
    };

    let author = req.author.unwrap_or_else(|| req.user_id.clone());

    let comment = state
        .todo_store
        .add_comment(
            &req.user_id,
            &todo.id,
            author.clone(),
            req.content.clone(),
            comment_type.clone(),
        )
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let experience_type = match comment_type.as_ref().unwrap_or(&TodoCommentType::Comment) {
        TodoCommentType::Comment => ExperienceType::Observation,
        TodoCommentType::Progress => ExperienceType::Learning,
        TodoCommentType::Resolution => ExperienceType::Learning,
        TodoCommentType::Activity => ExperienceType::Context,
    };

    let memory_content = format!(
        "[{}] {} ({}): {}",
        todo.short_id(),
        match comment_type.as_ref().unwrap_or(&TodoCommentType::Comment) {
            TodoCommentType::Comment => "Comment",
            TodoCommentType::Progress => "Progress",
            TodoCommentType::Resolution => "Resolution",
            TodoCommentType::Activity => "Activity",
        },
        todo.content,
        req.content
    );

    let mut tags = vec![
        format!("todo:{}", todo.short_id()),
        format!("todo-comment:{:?}", comment.comment_type).to_lowercase(),
    ];
    if let Some(ref project_id) = todo.project_id {
        if let Ok(Some(project)) = state.todo_store.get_project(&req.user_id, project_id) {
            tags.push(format!("project:{}", project.name));
        }
    }

    let experience = Experience {
        content: memory_content,
        experience_type,
        tags,
        ..Default::default()
    };

    if let Ok(memory) = state.get_user_memory(&req.user_id) {
        let memory_clone = memory.clone();
        let exp_clone = experience.clone();
        let memory_result = tokio::task::spawn_blocking(move || {
            let memory_guard = memory_clone.read();
            memory_guard.remember(exp_clone, None)
        })
        .await;

        if let Ok(Ok(memory_id)) = memory_result {
            if let Err(e) =
                state.process_experience_into_graph(&req.user_id, &experience, &memory_id)
            {
                tracing::debug!(
                    "Graph processing failed for todo comment memory {}: {}",
                    memory_id.0,
                    e
                );
            }

            link_todo_memory(&state, &req.user_id, &todo.id, &memory_id);
            tracing::debug!(
                memory_id = %memory_id.0,
                todo_id = %todo.id,
                "Todo comment stored as memory"
            );
        }
    }

    let formatted = todo_formatter::format_comment_added(&todo.short_id(), &comment);

    state.emit_event(MemoryEvent {
        event_type: "TODO_COMMENT_ADD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(comment.id.0.to_string()),
        content_preview: Some(format!(
            "[{}] {}",
            todo.short_id(),
            req.content.chars().take(80).collect::<String>()
        )),
        memory_type: Some(format!("{:?}", comment.comment_type)),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    tracing::debug!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        comment_id = %comment.id.0,
        "Added comment to todo"
    );

    Ok(Json(CommentResponse {
        success: true,
        comment: Some(comment),
        formatted,
    }))
}

/// GET /api/todos/{todo_id}/comments - List comments for a todo
pub async fn list_todo_comments(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<CommentListResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let comments = state
        .todo_store
        .get_comments(&user_id, &todo.id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_comment_list(&todo.short_id(), &comments);

    tracing::debug!(
        user_id = %user_id,
        todo_id = %todo.id,
        count = comments.len(),
        "Listed todo comments"
    );

    Ok(Json(CommentListResponse {
        success: true,
        count: comments.len(),
        comments,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/comments/{comment_id}/update - Update a comment
pub async fn update_todo_comment(
    State(state): State<AppState>,
    Path((todo_id, comment_id)): Path<(String, String)>,
    Json(req): Json<UpdateCommentRequest>,
) -> Result<Json<CommentResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Comment content cannot be empty".to_string(),
        });
    }

    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let cid = uuid::Uuid::parse_str(&comment_id).map_err(|_| AppError::InvalidInput {
        field: "comment_id".to_string(),
        reason: "Invalid comment ID format".to_string(),
    })?;
    let comment_id_typed = TodoCommentId(cid);

    let comment = state
        .todo_store
        .update_comment(
            &req.user_id,
            &todo.id,
            &comment_id_typed,
            req.content.clone(),
        )
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::InvalidInput {
            field: "comment_id".to_string(),
            reason: "Comment not found".to_string(),
        })?;

    let formatted = format!(
        "✓ Updated comment on {}\n\n  Updated content:\n  {}",
        todo.short_id(),
        req.content
    );

    tracing::debug!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        comment_id = %comment_id_typed.0,
        "Updated todo comment"
    );

    Ok(Json(CommentResponse {
        success: true,
        comment: Some(comment),
        formatted,
    }))
}

/// DELETE /api/todos/{todo_id}/comments/{comment_id} - Delete a comment
pub async fn delete_todo_comment(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path((todo_id, comment_id)): Path<(String, String)>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<CommentResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let cid = uuid::Uuid::parse_str(&comment_id).map_err(|_| AppError::InvalidInput {
        field: "comment_id".to_string(),
        reason: "Invalid comment ID format".to_string(),
    })?;
    let comment_id_typed = TodoCommentId(cid);

    let success = state
        .todo_store
        .delete_comment(&user_id, &todo.id, &comment_id_typed)
        .map_err(AppError::Internal)?;

    let formatted = if success {
        format!("✓ Deleted comment from {}", todo.short_id())
    } else {
        "Comment not found".to_string()
    };

    if success {
        state.emit_event(MemoryEvent {
            event_type: "TODO_COMMENT_DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: user_id.clone(),
            memory_id: Some(comment_id.to_string()),
            content_preview: Some(format!("[{}] comment deleted", todo.short_id())),
            memory_type: None,
            importance: None,
            count: None,
            entities: None,
            results: None,
        });
    }

    tracing::debug!(
        user_id = %user_id,
        todo_id = %todo.id,
        comment_id = %comment_id,
        success = success,
        "Deleted todo comment"
    );

    Ok(Json(CommentResponse {
        success,
        comment: None,
        formatted,
    }))
}

// =============================================================================
// PROJECT HANDLERS
// =============================================================================

/// POST /api/projects - Create a new project
pub async fn create_project(
    State(state): State<AppState>,
    Json(req): Json<CreateProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.name.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "name".to_string(),
            reason: "Project name cannot be empty".to_string(),
        });
    }

    let parent_id = if let Some(ref parent_ref) = req.parent {
        if let Ok(uuid) = uuid::Uuid::parse_str(parent_ref) {
            let pid = ProjectId(uuid);
            state
                .todo_store
                .get_project(&req.user_id, &pid)
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::ProjectNotFound(parent_ref.clone()))?;
            Some(pid)
        } else {
            let parent = state
                .todo_store
                .find_project_by_name(&req.user_id, parent_ref)
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::ProjectNotFound(parent_ref.clone()))?;
            Some(parent.id)
        }
    } else {
        None
    };

    let mut project = Project::new(req.user_id.clone(), req.name.clone());
    if let Some(ref custom_prefix) = req.prefix {
        let clean = custom_prefix.trim().to_uppercase();
        if !clean.is_empty() {
            project.prefix = Some(clean);
        }
    }
    project.description = req.description;
    project.color = req.color;
    project.parent_id = parent_id;

    state
        .todo_store
        .store_project(&project)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_project_created(&project);

    state.emit_event(MemoryEvent {
        event_type: "PROJECT_CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(project.id.0.to_string()),
        content_preview: Some(project.name.clone()),
        memory_type: Some("Project".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        name = %req.name,
        parent = ?project.parent_id,
        "Created project"
    );

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(project),
        stats: None,
        formatted,
    }))
}

/// POST /api/projects/list - List projects
pub async fn list_projects(
    State(state): State<AppState>,
    Json(req): Json<ListProjectsRequest>,
) -> Result<Json<ProjectListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    let mut project_stats = Vec::new();
    for project in projects {
        let stats = state
            .todo_store
            .get_project_stats(&req.user_id, &project.id)
            .map_err(AppError::Internal)?;
        project_stats.push((project, stats));
    }

    let formatted = todo_formatter::format_project_list(&project_stats);

    Ok(Json(ProjectListResponse {
        success: true,
        count: project_stats.len(),
        projects: project_stats,
        formatted,
    }))
}

/// GET /api/projects/{project_id} - Get a project with stats
pub async fn get_project(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(project_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<ProjectResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let project = state
        .todo_store
        .find_project_by_name(&user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let stats = state
        .todo_store
        .get_project_stats(&user_id, &project.id)
        .map_err(AppError::Internal)?;

    let todos = state
        .todo_store
        .list_todos_by_project(&user_id, &project.id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_project_todos(&project, &todos, &stats);

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(project),
        stats: Some(stats),
        formatted,
    }))
}

/// POST /api/projects/{project_id}/update - Update a project
pub async fn update_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<UpdateProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let updated = state
        .todo_store
        .update_project(
            &req.user_id,
            &project.id,
            req.name,
            req.prefix,
            req.description,
            req.status,
            req.color,
        )
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let formatted = todo_formatter::format_project_updated(&updated);

    state.emit_event(MemoryEvent {
        event_type: "PROJECT_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(updated.id.0.to_string()),
        content_preview: Some(updated.name.clone()),
        memory_type: Some("Project".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %updated.id.0,
        status = ?updated.status,
        "Updated project"
    );

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(updated),
        stats: None,
        formatted,
    }))
}

/// DELETE /api/projects/{project_id} - Delete a project
pub async fn delete_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<DeleteProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    // delete_project cascades into sub-projects, so their todos are counted and lose their
    // memory back-links too.
    let doomed_todos = if req.delete_todos {
        let mut collected = Vec::new();
        collect_project_todos(&state, &req.user_id, &project.id, &mut collected)?;
        collected
    } else {
        Vec::new()
    };
    let todos_count = doomed_todos.len();

    let deleted = state
        .todo_store
        .delete_project(&req.user_id, &project.id, req.delete_todos)
        .map_err(AppError::Internal)?;

    if !deleted {
        return Err(AppError::ProjectNotFound(project_id));
    }

    for todo in &doomed_todos {
        sync_memory_back_links(
            &state,
            &req.user_id,
            &todo.id,
            Vec::new(),
            todo.related_memory_ids.clone(),
        )
        .await?;
    }

    let formatted = todo_formatter::format_project_deleted(&project, todos_count);

    state.emit_event(MemoryEvent {
        event_type: "PROJECT_DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(project.id.0.to_string()),
        content_preview: Some(project.name.clone()),
        memory_type: Some("Project".to_string()),
        importance: None,
        count: Some(todos_count),
        entities: None,
        results: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        delete_todos = %req.delete_todos,
        todos_deleted = %todos_count,
        "Deleted project"
    );

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(project),
        stats: None,
        formatted,
    }))
}
