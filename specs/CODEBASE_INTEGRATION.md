# Veld - Agentic Memory Codebase Integration Specification

**Status:** Draft
**Author:** Claude + Varun
**Date:** 2025-12-30
**Version:** 0.1

---

## Overview

Integrate codebase awareness into Veld - Agentic Memory, allowing projects to have associated codebases that the AI learns about through usage. Files are treated as contextual knowledge that supplements (never replaces) decision/learning memories.

### Design Principles

1. **Brain, not IDE** - Learn organically through usage, not upfront indexing
2. **Supplement, not replace** - Files are context, decisions are knowledge
3. **Clean separation** - File Explorer is a separate popup window
4. **Backwards compatible** - Existing functionality untouched
5. **Invisible by default** - Works automatically, manual exploration optional

---

## 1. Data Model

### 1.1 Project Extension

```rust
pub struct Project {
    // Existing fields...
    pub id: ProjectId,
    pub name: String,
    pub description: Option<String>,
    pub status: ProjectStatus,
    pub parent_id: Option<ProjectId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    // NEW: Codebase integration
    pub codebase_path: Option<String>,      // Absolute path to codebase root
    pub codebase_indexed: bool,             // Has initial index been run?
    pub codebase_indexed_at: Option<DateTime<Utc>>,
    pub codebase_file_count: usize,         // Number of files tracked
}
```

### 1.2 FileMemory (New)

```rust
/// Learned knowledge about a file in a codebase
/// Stored with key prefix: `filemem:{user_id}:{project_id}:{path_hash}`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMemory {
    pub id: FileMemoryId,
    pub project_id: ProjectId,
    pub user_id: String,

    // File identification
    pub path: String,               // Relative path from codebase root
    pub absolute_path: String,      // Full path for access
    pub file_hash: String,          // SHA256 of content (for change detection)

    // Learned content
    pub summary: String,            // AI-generated summary
    pub key_items: Vec<String>,     // Functions, classes, exports
    pub purpose: Option<String>,    // What this file does
    pub connections: Vec<String>,   // Related files (imports, etc.)

    // Metadata
    pub file_type: FileType,        // Rust, TypeScript, Python, etc.
    pub line_count: usize,
    pub size_bytes: u64,

    // Usage tracking
    pub access_count: u32,          // Times accessed by AI
    pub last_accessed: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    // Learning source
    pub learned_from: LearnedFrom,  // Manual index vs organic usage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Markdown,
    Json,
    Toml,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearnedFrom {
    ManualIndex,    // User triggered indexing
    ReadAccess,     // AI read the file
    EditAccess,     // AI edited the file
    Mentioned,      // File mentioned in conversation
}
```

### 1.3 Storage Schema

```
RocksDB Keys:

# FileMemory storage
filemem:{user_id}:{file_id}                    → FileMemory (serialized)

# Indexes
filemem_by_project:{user_id}:{project_id}:{file_id}  → file_id
filemem_by_path:{user_id}:{path_hash}                → file_id
filemem_by_type:{user_id}:{file_type}:{file_id}      → file_id
```

---

## 2. Search & Retrieval Behavior

### 2.1 Separation Principle

**FileMemories are NEVER mixed with regular memories in default search.**

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Search Space                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────┐    ┌─────────────────────────┐   │
│   │  Primary Memories   │    │    File Memories        │   │
│   │  ─────────────────  │    │    ──────────────       │   │
│   │  • Decision         │    │    • FileMemory         │   │
│   │  • Learning         │    │                         │   │
│   │  • Error            │    │    Searched separately  │   │
│   │  • Discovery        │    │    Only when:           │   │
│   │  • Pattern          │    │    • File path mentioned│   │
│   │  • Context          │    │    • Editing a file     │   │
│   │  • Task             │    │    • Explicit request   │   │
│   └─────────────────────┘    └─────────────────────────┘   │
│            ▲                            ▲                   │
│            │                            │                   │
│      recall()                   recall_files()              │
│      proactive_context          (explicit only)             │
│      (memories section)         proactive_context           │
│                                 (files section)             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 recall() Behavior (Unchanged)

```rust
// Default: searches memories only, excludes FileMemory
pub fn recall(query: &str, limit: usize) -> Vec<Memory>

// FileMemory is ExperienceType::FileAccess
// By default, recall excludes FileAccess type
```

### 2.3 recall_files() (New)

```rust
/// Search file memories for a project
pub fn recall_files(
    project_id: &ProjectId,
    query: &str,
    limit: usize,
) -> Vec<FileMemory>
```

### 2.4 proactive_context() Changes

**Response structure becomes:**

```json
{
  "success": true,
  "memories": [
    {
      "id": "...",
      "type": "Decision",
      "content": "Use JWT tokens with 24h expiry",
      "relevance": 0.92
    },
    {
      "id": "...",
      "type": "Learning",
      "content": "Session tokens caused mobile issues",
      "relevance": 0.87
    }
  ],
  "relevant_files": [
    {
      "path": "src/auth.rs",
      "summary": "Authentication module - JWT validation",
      "relevance": 0.78,
      "last_accessed": "2h ago"
    }
  ],
  "relevant_todos": [...],
  "triggered_reminders": [...]
}
```

**Key changes:**
- `memories` - Regular memories (Decision, Learning, etc.)
- `relevant_files` - FileMemories, separate section
- Files only included when contextually relevant

### 2.5 When Files Surface

| Trigger | Files Surface? | Example |
|---------|---------------|---------|
| General question | No | "What did we decide about auth?" |
| File path mentioned | Yes | "What does src/auth.rs do?" |
| Currently editing file | Yes | Context: editing auth.rs |
| Code-related query | Maybe | "How does authentication work?" |
| Explicit request | Yes | "Show me relevant files" |

### 2.6 Relevance Scoring

**For mixed queries (when files are included):**

```rust
fn calculate_relevance(item: &SearchResult) -> f32 {
    let base_score = semantic_similarity;

    let type_multiplier = match item.type {
        Decision | Learning => 1.3,    // Boosted - primary knowledge
        Error | Discovery => 1.1,      // Slightly boosted
        Pattern | Context => 1.0,      // Neutral
        Task => 0.9,                   // Slightly suppressed
        FileAccess => 0.6,             // Suppressed - contextual only
        Conversation => 0.5,           // Background
    };

    base_score * type_multiplier * recency_factor * access_factor
}
```

---

## 3. API Design

### 3.1 Project Endpoints (Modified)

```
PATCH /api/projects/{project_id}
Request:
{
  "codebase_path": "/absolute/path/to/codebase"
}

Response:
{
  "success": true,
  "project": { ... }
}
```

### 3.2 File Endpoints (New)

```
# List files for a project
GET /api/projects/{project_id}/files?limit=50&offset=0&sort=access_count

Response:
{
  "success": true,
  "files": [
    {
      "id": "...",
      "path": "src/main.rs",
      "summary": "API server with 50+ routes",
      "access_count": 47,
      "last_accessed": "2024-12-30T10:00:00Z"
    }
  ],
  "total": 142
}

# Get single file details
GET /api/projects/{project_id}/files/{file_id}

Response:
{
  "success": true,
  "file": {
    "id": "...",
    "path": "src/main.rs",
    "summary": "API server binary...",
    "key_items": ["AppState", "remember()", "recall()"],
    "purpose": "Main HTTP server entry point",
    "connections": ["src/memory/mod.rs", "src/mcp.rs"],
    "access_count": 47,
    "recent_edits": [
      { "when": "2h ago", "what": "Added auto-classification" }
    ]
  }
}

# Trigger indexing
POST /api/projects/{project_id}/index
Request:
{
  "force": false,           // Re-index even if already indexed
  "include_patterns": ["*.rs", "*.ts"],
  "exclude_patterns": ["target/", "node_modules/"]
}

Response:
{
  "success": true,
  "indexed": 142,
  "skipped": 15,
  "errors": []
}

# Search files
POST /api/projects/{project_id}/files/search
Request:
{
  "query": "authentication",
  "limit": 10
}

Response:
{
  "success": true,
  "files": [...]
}
```

### 3.3 Auto-Learning Endpoint (Internal)

```
# Called when AI reads/edits a file
POST /api/files/learn
Request:
{
  "user_id": "...",
  "project_id": "...",        // Optional, auto-detected from path
  "path": "/absolute/path",
  "action": "read" | "edit",
  "content_hash": "sha256...",
  "summary": "Optional AI-generated summary"
}
```

---

## 4. TUI Integration

### 4.1 Projects View Changes

```
PROJECTS 9
────────────────────────────────
📂 Veld               4 left · 82%
   📁 ~/veld  ●            ← Codebase row (NEW)
   └ tui              ✓ done       ← Subproject (existing)
   └ Implicit Feed... ✓ done

📂 Bolt              53 left · 0%
   📁 ~/bolt          ○            ← Red = not indexed
   └ bolt-api         10 left

📂 Personal           2 left
   (no codebase linked)           ← No codebase
```

**Visual indicators:**
- `📁` icon for codebase row
- `●` green = indexed
- `○` red = not indexed
- Shows path (can be truncated)

### 4.2 Keybindings

| Key | Action | Context |
|-----|--------|---------|
| `Enter` | Open File Explorer popup | On codebase row |
| `i` | Index/re-index codebase | On codebase row |
| `l` | Link codebase to project | On project with no codebase |

### 4.3 Status Line

```
Before:
337 memories │ 8404 edges │ 26 recalls

After:
337 memories │ 8404 edges │ 26 recalls │ 📁 Veld (142)
                                         └── Current project's codebase
```

---

## 5. File Explorer (Popup Window)

### 5.1 Launch Method

```bash
# Spawned from main TUI when Enter pressed on codebase row
veld-files --project-id <id> --api-url <url> --api-key <key>

# Or standalone
veld-files ~/path/to/codebase
```

### 5.2 Layout

```
┌─ 📁 Veld ──────────────────────────────────────────────────────────────────────┐
│  ~/veld                                    142 files │ indexed 2h ago  │
├──────────────────────────────────┬─────────────────────────────────────────────┤
│  TREE                            │  DETAILS                                    │
│  ────                            │  ───────                                    │
│  ▾ src/                    38    │  src/main.rs                                │
│    ▸ main.rs              ●●●    │  ─────────────────────────────────────────  │
│      memory/                     │                                             │
│        mod.rs             ●●     │  API server binary. Axum-based HTTP server  │
│        lineage.rs         ●      │  with 50+ routes for memory operations.     │
│        todos.rs           ●●     │                                             │
│      embeddings/                 │  Purpose:                                   │
│        minilm.rs                 │  Main entry point for the Veld server        │
│  ▾ tui/                    12    │                                             │
│      src/                        │  Key Items:                                 │
│        main.rs            ●●●    │  ├─ AppState      shared application state  │
│  ▾ mcp-server/              4    │  ├─ remember()    store new memories        │
│      index.ts             ●●     │  ├─ recall()      semantic search           │
│                                  │  └─ proactive_context()                     │
│                                  │                                             │
│                                  │  Connections:                               │
│                                  │  → src/memory/mod.rs                        │
│                                  │  → src/mcp.rs                               │
│                                  │                                             │
│                                  │  Recent Activity:                           │
│                                  │  • 2h ago - Added auto-classification       │
│                                  │  • 1d ago - Fixed method call               │
├──────────────────────────────────┴─────────────────────────────────────────────┤
│  ↑↓ navigate   Enter expand   / search   i index   r refresh   q close         │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Features

- **Tree view** - Collapsible directory structure
- **Heat indicators** - `●●●` for frequently accessed files
- **Detail panel** - Summary, key items, connections
- **Search** - `/` to filter files
- **Refresh** - `r` to update from API
- **Index** - `i` to trigger re-indexing

### 5.4 Keybindings

| Key | Action |
|-----|--------|
| `↑/↓` | Navigate tree |
| `Enter` | Expand/collapse directory |
| `Tab` | Switch focus tree ↔ details |
| `/` | Search files |
| `i` | Index codebase |
| `r` | Refresh |
| `q/Esc` | Close window |

---

## 6. Auto-Learning Flow

### 6.1 When AI Reads a File

```
Claude reads src/main.rs
        │
        ▼
┌───────────────────────────────────────────┐
│ 1. Check if FileMemory exists for path    │
│ 2. If not, create stub FileMemory         │
│ 3. Increment access_count                 │
│ 4. Update last_accessed                   │
│ 5. If content changed, mark for re-learn  │
└───────────────────────────────────────────┘
        │
        ▼
FileMemory updated in background
```

### 6.2 When AI Edits a File

```
Claude edits src/main.rs (adds function)
        │
        ▼
┌───────────────────────────────────────────┐
│ 1. Update FileMemory.file_hash            │
│ 2. Create CodeEdit memory (existing)      │
│ 3. Link CodeEdit to FileMemory            │
│ 4. Queue summary regeneration             │
└───────────────────────────────────────────┘
```

### 6.3 Summary Generation

**Option A: On-demand (lazy)**
- Generate summary when file is viewed in explorer
- Uses local LLM or deferred to user's LLM

**Option B: Background (eager)**
- Queue files for summarization
- Process during idle time
- Requires local summarization capability

**Recommended: Option A** - simpler, no extra inference cost

---

## 7. Migration & Compatibility

### 7.1 Existing Data

- **Projects** - Gain optional `codebase_path`, default None
- **Memories** - Unchanged
- **Todos** - Unchanged
- **RocksDB** - New `filemem:` prefix, no conflicts

### 7.2 Version Compatibility

```rust
// Project deserialization handles missing fields
#[serde(default)]
pub codebase_path: Option<String>,

#[serde(default)]
pub codebase_indexed: bool,
```

### 7.3 Rollback

If feature disabled:
- `codebase_path` ignored
- FileMemories stay in DB but unused
- No functional impact

---

## 8. Configuration

### 8.1 Server Config

```toml
[codebase]
enabled = true
max_files_per_project = 1000          # Hard limit - most devs work with <1000 files
max_file_size_for_embedding = 524288  # 500KB - larger files chunked or summarized
auto_learn_on_access = true
require_explicit_indexing = true      # Indexing only starts when user/LLM okays it

default_exclude_patterns = [
  "target/",
  "node_modules/",
  ".git/",
  "__pycache__/",
  "dist/",
  "build/",
  "*.lock",
  "*.min.js",
  "*.min.css",
  "*.map"
]

# Binary/non-code files always skipped
skip_binary = true
```

### 8.2 Per-Project Config

```json
{
  "codebase_path": "/path/to/code",
  "include_patterns": ["*.rs", "*.ts", "*.py"],
  "exclude_patterns": ["tests/fixtures/"]
}
```

---

## 9. Implementation Phases

### Phase 1: Foundation (6h)
- [ ] Add `codebase_path` to Project struct
- [ ] Add FileMemory struct and storage
- [ ] Basic API endpoints (list, get, update)
- [ ] Tests

### Phase 2: Main TUI (3h)
- [ ] Codebase row in Projects view
- [ ] Green/red indicator
- [ ] Spawn file explorer command
- [ ] Status line update

### Phase 3: File Explorer (12h)
- [ ] New binary: `veld-files`
- [ ] Tree view component
- [ ] Detail panel component
- [ ] Navigation and search
- [ ] API integration

### Phase 4: Intelligence (6h)
- [ ] Auto-learn on file access
- [ ] Access counting
- [ ] proactive_context integration
- [ ] Relevance scoring

### Phase 5: Polish (3h)
- [ ] Edge cases
- [ ] Error handling
- [ ] Documentation
- [ ] Performance testing

---

## 10. Indexing Flow

### 10.1 Explicit Initialization Required

Indexing NEVER starts automatically. User or LLM must explicitly approve:

```
1. User links codebase path to project
2. System shows: "Codebase linked. Ready to index 847 files. Index now? [y/n]"
3. User/LLM confirms
4. Indexing begins in background
5. User can continue working while indexing runs
```

### 10.2 Progressive Background Indexing

Once approved, indexing runs in background without blocking:

```
┌────────────────────────────────────────────────────────────────┐
│  Indexing Progress                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━░░░░░░░░░░░░░ 67% (567/847)        │
│  Currently: src/memory/storage.rs                               │
│  Estimated: ~30s remaining                                      │
└────────────────────────────────────────────────────────────────┘
```

**Stages:**
1. **Scan** (fast) - Count files, apply exclusion patterns
2. **Embed** (slow) - Generate embeddings for each file
3. **Store** - Save FileMemory to RocksDB
4. **Index** - Add to vector index for search

**Priority order during indexing:**
- Files accessed by LLM while indexing → immediate priority
- Recently modified files → higher priority
- Deep directories → lower priority

### 10.3 Hard Limits

| Limit | Value | Behavior when exceeded |
|-------|-------|----------------------|
| Max files per project | 1,000 | Stop indexing, warn user |
| Max file size | 500 KB | Chunk into sections, embed each |
| Total scan time | 5 minutes | Abort, partial index available |

**Large file handling:**
- Files 0-50KB: Embed full content
- Files 50-500KB: Extract key sections (functions, classes, exports) + embed
- Files >500KB: Skip embedding, store path + metadata only (can be fetched on demand)

If codebase has >1000 source files (after exclusions), user must specify include patterns to narrow scope.

---

## 11. Open Questions

1. **Summary generation** - Local LLM, API call, or manual?
2. ~~**Large codebases** - How to handle 10k+ files?~~ **RESOLVED: 1000 file limit**
3. **Monorepos** - Multiple projects in one codebase?
4. **Remote codebases** - SSH paths, git URLs?
5. **Real-time sync** - Watch for file changes?

---

## 12. Success Criteria

1. FileMemories never crowd out Decision/Learning in search
2. File Explorer opens in <500ms
3. 1000 files scanned in <5s, embedded in <60s (background)
4. Zero impact on existing memory/todo functionality
5. Clean separation - can disable feature entirely

---

## Appendix A: Example Flows

### A.1 Link Codebase to Project

```
1. User in TUI, selects "Veld" project
2. Presses 'l' (link codebase)
3. Dialog: "Enter codebase path: [~/veld]"
4. Confirms
5. API: PATCH /api/projects/{id} { codebase_path: "..." }
6. System scans directory (fast): "Found 847 source files"
7. Prompt: "Index 847 files now? [y/n/later]"
8. User confirms 'y'
9. Background indexing starts, progress bar shown
10. User can continue working (switch screens, etc.)
11. When complete: 📁 ~/veld ● (green indicator)
```

### A.2 Explore Files

```
1. User selects the Veld codebase row
2. Presses Enter
3. New terminal window spawns with veld-files
4. User navigates tree, views file details
5. Presses 'q' to close
6. Back to main TUI
```

### A.3 AI Learns About File

```
1. Claude reads src/main.rs (via Read tool)
2. Hook detects file access
3. Checks: Is path under a project's codebase_path?
4. Yes → Create/update FileMemory
5. FileMemory: { path: "src/main.rs", access_count: 1 }
6. Next time user asks "what files handle routing?"
7. proactive_context includes src/main.rs in relevant_files
```
