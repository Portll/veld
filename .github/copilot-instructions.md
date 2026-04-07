# Veld - Agentic Memory: Workspace Instructions

You have access to Veld's persistent memory system via MCP tools. Memory persists across sessions — you are not starting fresh.

## Memory Tools — Registered Switches

### Core Memory (always use)
| Tool | When | Switch |
|------|------|--------|
| `proactive_context` | **Every first message** — surfaces relevant memories, todos, facts, entities | ON |
| `remember` | After decisions, learnings, errors, discoveries, patterns | ON |
| `recall` | When searching past context by meaning (modes: semantic, associative, hybrid) | ON |
| `forget` | To suppress/correct a stored memory | ON |
| `context_summary` | Quick overview of recent learnings/decisions/context | ON |
| `read_memory` | Read full content of a specific memory by ID | ON |
| `list_memories` | Browse all stored memories | ON |
| `memory_stats` | Check memory count, type distribution, health | ON |

### Todo System (GTD task management)
| Tool | When | Switch |
|------|------|--------|
| `add_todo` | Track work items with priority, project, context, dependencies | ON |
| `list_todos` | Query todos by status/project/priority/due date | ON |
| `update_todo` | Change status, priority, blocked_on, or any field | ON |
| `complete_todo` | Mark done (auto-creates next for recurring) | ON |
| `delete_todo` | Remove a todo | ON |
| `reorder_todo` | Move up/down in list | ON |
| `add_project` | Create project with auto-generated prefix | ON |
| `list_projects` | See all projects with todo counts | ON |
| `archive_project` | Archive completed project | ON |
| `delete_project` | Remove project | ON |
| `todo_stats` | Counts by status, overdue items | ON |
| `list_subtasks` | Get subtasks of a parent todo | ON |
| `add_todo_comment` | Add comment/progress/resolution to a todo | ON |
| `list_todo_comments` | Read comments on a todo | ON |
| `update_todo_comment` | Edit a comment | ON |
| `delete_todo_comment` | Remove a comment | ON |

### Reminders (Prospective Memory)
| Tool | When | Switch |
|------|------|--------|
| `set_reminder` | Time-based, duration-based, or context-triggered reminders | ON |
| `list_reminders` | See pending/triggered/dismissed reminders | ON |
| `dismiss_reminder` | Dismiss a triggered reminder | ON |

### Index & Maintenance
| Tool | When | Switch |
|------|------|--------|
| `verify_index` | Check for orphaned memories or index corruption | ON |
| `repair_index` | Fix orphaned memories | ON |
| `consolidation_report` | See memory strengthening, decay, edge formation | ON |
| `backup_create` | Create a backup of all memory data | ON |
| `backup_list` | List available backups | ON |
| `backup_verify` | Verify backup integrity (SHA-256) | ON |
| `backup_restore` | Restore from a backup | ON |
| `backup_purge` | Remove old backups (keeps N most recent) | ON |

### Lineage (Causal Chains)
| Tool | When | Switch |
|------|------|--------|
| `lineage_trace` | Trace causal chains backward/forward from a memory | ON |
| `lineage_link` | Explicitly connect cause→effect memories | ON |
| `lineage_confirm` | Confirm an inferred causal edge | ON |
| `lineage_reject` | Reject an incorrect inferred edge | ON |
| `lineage_stats` | Total edges, confidence, relation types | ON |

### Session Management
| Tool | When | Switch |
|------|------|--------|
| `token_status` | Check token budget usage | ON |
| `reset_token_session` | Reset token counter for new session | ON |
| `seed_project` | Bootstrap memory from codebase files | ON |

## Skills

### veld
Persistent cognitive memory best practices. Activated when storing decisions, learnings, errors, or context.

### orchestrate
Multi-agent parallel execution with todo-driven task graphs. Activated for "orchestrate", "break down work", "coordinate subtasks", "multi-agent" requests.

## Workflow

1. **Start of conversation**: Call `proactive_context` with the user's first message to load relevant context.
2. **During work**: Use `remember` for important decisions, learnings, errors, and discoveries. Use `recall` when you need past context.
3. **Task tracking**: Use todos for multi-step work. Use projects for grouping related tasks.
4. **End of significant work**: Store a summary memory with tags for future retrieval.

## Memory Types

Use the right type for proper importance weighting:
- `Decision` — architectural choices, user preferences (high importance, slow decay)
- `Learning` — new knowledge gained (high importance)
- `Error` — bugs and fixes (high importance, slow decay)
- `Discovery` — insights, aha moments
- `Pattern` — recurring behaviors
- `Context` — background information (lower importance, faster decay)
- `Task` — work in progress
- `Observation` — general notes (lowest importance)
- `CodeEdit` — file modifications (auto-captured by hooks)
- `Command` — shell command results
- `Conversation` — dialogue context

## Recall Modes

| Mode | Use When |
|------|----------|
| `semantic` | Pure meaning-based search |
| `associative` | Follow learned graph connections |
| `hybrid` | Best of both (default, recommended) |

## This Codebase

Veld is its own memory system. Key structure:
- `src/` — Rust core (memory engine, API server, embeddings, graph, Hebbian learning)
- `mcp-server/` — TypeScript MCP server (47 tools, streaming ingestion)
- `tui/` — Rust TUI dashboard
- `hooks/` — Claude Code hooks for automatic memory capture
- `skills/` — Agent discovery modules (orchestrate, veld)
- `python/` — Python bindings (maturin/PyO3)

Architecture: RocksDB + HNSW vector search + knowledge graph with Hebbian learning.
