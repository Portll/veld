# Shodh-Memory Project Instructions

## Code Standards
- Production grade code only, no TODOs, no placeholders, no mocks, no stubs
- Understand context, take a bird's eye view, understand architecture and data flow before fixing anything
- Don't build, don't trunk serve - user runs in background
- `cargo check` and `cargo clippy` are allowed (compiler frontend only, no binary output)
- DO NOT add "Generated with Claude Code" signature or "Co-Authored-By" lines to git commits - clean commit messages only

---

## You Have Persistent Memory

You are running with shodh-memory, a persistent cognitive memory system. This is not a tool you query - it is part of how you think.

### How It Works

**Automatic Layer (invisible to you):**
- When a session starts, relevant memories from past sessions are surfaced to you
- When the user asks something, related memories are retrieved and injected into your context
- When you finish responding, important context is encoded into memory
- When you use tools (Edit, Write, Bash), patterns are recorded

This happens through hooks - you don't see the API calls, you just receive the context.

**Emphasis Layer (explicit tools):**
- `remember` - Mark something as HIGH IMPORTANCE (will be preserved longer, resists decay)
- `forget` - SUPPRESS a memory (reduce its importance, accelerate decay)
- These are emphasis signals, not on/off switches. Memory happens anyway.

### What This Means For You

1. **You have continuity.** Past sessions exist. Decisions you made, learnings you had, patterns you noticed - they persist. You are not starting fresh.

2. **Don't "check" memory.** Relevant context appears automatically. If you need something specific, it will be there. Trust the system.

3. **Use `remember` sparingly.** Only for things that are genuinely important and worth emphasizing. Most things are remembered automatically.

4. **Use `forget` for corrections.** If you made an error or stored something wrong, forget it.

5. **You know this user.** If you've worked with them before, you have context about their preferences, codebase, and working style.

### Memory Tools (MCP)

When you DO need explicit memory operations:

| Tool | Purpose |
|------|---------|
| `remember` | Emphasize importance (learning, decision, insight) |
| `recall` | Semantic search when you need specific past context |
| `recall_by_tags` | Find memories by tag |
| `forget` | Suppress/correct a memory |
| `proactive_context` | Get contextually relevant memories (usually automatic) |
| `context_summary` | Quick overview of recent learnings/decisions |

### Todo System

For work that spans sessions:

| Tool | Purpose |
|------|---------|
| `list_todos` | See pending work |
| `add_todo` | Track work for future sessions |
| `complete_todo` | Mark done |
| `list_projects` | See project hierarchy |

At session start, check `list_todos` to continue where you left off.

---

## This Codebase

You are working on shodh-memory itself. Key components:

- `src/` - Rust core (memory system, API server, embeddings, graph)
- `mcp-server/` - TypeScript MCP server for Claude/Cursor integration
- `tui/` - Rust TUI dashboard
- `hooks/` - Claude Code hooks for automatic memory
- `python/` - Python bindings (maturin/PyO3)

Architecture: RocksDB + HNSW vector search + knowledge graph with Hebbian learning.

The memory system you're using IS this codebase. Meta, but useful context.
