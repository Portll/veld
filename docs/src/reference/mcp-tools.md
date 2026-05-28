<!-- GENERATED FILE — do not edit by hand.
     Source: mcp-server/index.ts
     Generator: docs/generators/src/bin/gen-mcp-tools.rs
     Regenerate: cd docs/generators && cargo run --bin gen-mcp-tools -->

# MCP Tools

The TypeScript MCP server (`@veld/memory-mcp`) exposes **46** tools over the HTTP API. The Rust binary (`veld serve`) exposes the same tools via stdio MCP using `rmcp`.

Tools are listed alphabetically. For full parameter schemas, see [mcp-server/index.ts](https://github.com/Portll/veld/blob/main/mcp-server/index.ts).

| Tool | Description |
|---|---|
| `add_project` | Create a new project to group todos. Use parent to create a sub-project under another project. |
| `add_todo` | Add a task to your todo list. Supports GTD workflow with projects, contexts (@computer, @phone), priorities, due dates, and subtasks (via parent_id). |
| `add_todo_comment` | Add a comment to a todo. Use to track progress, notes, or resolution details. |
| `archive_project` | Archive a project. Archived projects are hidden by default but can be restored. |
| `backup_create` | Create a backup of all memories. Returns backup metadata including ID, size, and checksum. Backups are stored locally and can be restored later. |
| `backup_list` | List all available backups for this user. Returns backup history with IDs, timestamps, and sizes. |
| `backup_purge` | Purge old backups, keeping only the most recent N. Useful for managing disk space. |
| `backup_restore` | Restore a previously created backup by ID. This replaces all current data for the user with the backup contents. Server restart is recommended after restore. |
| `backup_verify` | Verify backup integrity using SHA-256 checksum. Use to check if a backup is corrupted before restoring. |
| `complete_todo` | Mark a todo as complete. For recurring tasks, automatically creates the next occurrence. |
| `consolidation_report` | Get a report of what the memory system has been learning. Shows memory strengthening/decay events, edge formation, fact extraction, and maintenance cycles. Use this to understand how your memories are evolving. |
| `context_summary` | Get a condensed summary of recent learnings, decisions, and context. Use this at the start of a session to quickly understand what you've learned before. |
| `count` | Number of memories (default: 10) |
| `delete_project` | Permanently delete a project. Use delete_todos=true to also delete all todos in the project. |
| `delete_todo` | Delete a todo permanently. |
| `delete_todo_comment` | Delete a comment from a todo. |
| `dismiss_reminder` | Dismiss/acknowledge a triggered reminder. Call this after you've handled a reminder. |
| `forget` | Delete a specific memory by ID |
| `list_memories` | List all stored memories |
| `list_projects` | List all projects with todo counts and status breakdown. |
| `list_reminders` | List all pending reminders. Use to check what reminders are scheduled. |
| `list_subtasks` | List subtasks of a parent todo. Use add_todo with parent_id to create subtasks. |
| `list_todo_comments` | List all comments and activity history for a specific todo. |
| `list_todos` | List or search todos. Supports semantic search via query parameter, or GTD-style filtering. Returns Linear-style formatted output grouped by status. |
| `memory_health` | Check memory system status and statistics |
| `memory_stats` | Get statistics about stored memories |
| `pending_work` | Show todos and incomplete tasks |
| `proactive_context` | REQUIRED: Call this tool with EVERY user message to surface relevant memories and build conversation history. Pass the user's message as context. This enables: (1) retrieving memories relevant to what the user is asking, (2) building persistent memory of the conversation for future sessions. The system analyzes entities, semantic similarity, and recency to find contextually appropriate memories. Auto-ingest stores the context automatically. USAGE: Always call this FIRST when you receive a user message, passing their message as the context parameter. |
| `query` | What to search for in memories |
| `quick_recall` | Search your memories for relevant context |
| `read_memory` | Read the FULL content of a specific memory by ID. Use this when you need to see the complete text of a memory that was truncated in search results. |
| `recall` | Search memories AND todos using semantic similarity. Returns both relevant memories and matching todos. Use this to find past experiences, decisions, context, or pending work. Modes: 'semantic' (vector similarity), 'associative' (graph traversal), 'hybrid' (combined). |
| `recent_memories` | Show recently created memories |
| `reorder_todo` | Move a todo up or down within its status group. Use to prioritize tasks manually. |
| `repair_index` | Repair vector index by re-indexing orphaned memories. Use this when verify_index shows unhealthy status. Returns count of repaired memories. |
| `reset_token_session` | Reset the token counter for a new session. Call this when starting a new conversation or after context has been compressed/summarized. |
| `seed_project` | Scan a project directory and create foundational memories from config files, README, and source code. Enables rapid bootstrapping of memory for a new project. Idempotent: re-running deletes old seed memories first. |
| `session_summary` | Get a summary of recent learnings, decisions, and context |
| `set_reminder` | Set a reminder for the future. Triggers on time (at specific time or after duration) or context match (when keywords appear in conversation). Reminders will surface automatically when conditions are met. |
| `todo_stats` | Get statistics about your todos - counts by status, overdue items, etc. |
| `token_status` | Get current token usage status for this session. Returns tokens used, budget remaining, and percentage consumed. Use this to check context window health. |
| `topic` | The topic to explore |
| `update_todo` | Update a todo's properties. Use short ID prefix (e.g., SHO-1a2b) or full ID. |
| `update_todo_comment` | Update an existing comment on a todo. |
| `verify_index` | Verify vector index integrity - diagnose orphaned memories that are stored but not searchable. Returns health status and count of orphaned memories. |
| `what_i_know` | Surface everything related to a topic |

---

*To use these from Claude Code or VS Code Copilot, see the [Claude Code integration guide](../guides/claude-code-integration.md) or [VS Code Copilot guide](../guides/vscode-copilot.md).*
