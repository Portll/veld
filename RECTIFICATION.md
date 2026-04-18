# Veld Rectification Plan

Generated from /portll-security + /breakers audits, 2026-04-08.

## Agent Assignment

Five agents run in parallel. Each agent owns a non-overlapping set of files.
No agent touches another agent's files. All agents target `c:\Repositories\Portll\veld`.

---

## Agent 1: TUI UTF-8 Safety

**Model**: sonnet
**Files owned**: `tui/src/widgets.rs`, `tui/src/types.rs`
**Estimated items**: 9 fixes

All `&str[..n]` byte-index slices on user-derived strings must be replaced with
char-boundary-safe truncation. These are CRITICAL — any non-ASCII text crashes the TUI.

### Findings

| # | Severity | File:Line | Finding | Fix |
|---|----------|-----------|---------|-----|
| 1 | CRITICAL | `tui/src/widgets.rs:681` | `&result.id[..8]` byte-slice on memory ID | Use `result.id.chars().take(8).collect::<String>()` or `result.id.get(..8).unwrap_or(&result.id)` after validating ASCII |
| 2 | CRITICAL | `tui/src/widgets.rs:1203` | `&path[path.len() - visible_width + 3..]` byte-slice on user file path | Use `path.char_indices()` to find safe split point |
| 3 | CRITICAL | `tui/src/widgets.rs:1625` | `&line[..popup_width - line_num_width - 11]` byte-slice on source code lines | Use `line.char_indices().take_while(...)` for safe truncation |
| 4 | CRITICAL | `tui/src/widgets.rs:4399` | `&todo.content[..max_content_len]` on user todo text | Safe char-boundary truncation |
| 5 | CRITICAL | `tui/src/widgets.rs:4408` | `&project[..12]` on project name | Safe char-boundary truncation |
| 6 | CRITICAL | `tui/src/widgets.rs:4428` | `&blocked[..12]` on blocked text | Safe char-boundary truncation |
| 7 | CRITICAL | `tui/src/types.rs:2097-2101` | `[..8.min(len)]` byte-slice on event from_id/to_id | Safe char-boundary truncation |
| 8 | CRITICAL | `tui/src/types.rs:2226` | `id[..8].to_string()` on graph node ID | Safe char-boundary truncation |
| 9 | CRITICAL | `tui/src/types.rs:2238` | `&content[..37]` on memory content (user-provided arbitrary text) | Safe char-boundary truncation |

### Helper pattern to use everywhere

```rust
fn truncate_safe(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
```

Add this as a utility function in `tui/src/types.rs` or `tui/src/widgets.rs` and replace all byte-index slices.

Also fix test files:
- `tests/benchmark_evaluation.rs:568` — `&eval.query_text[..67]`
- `tests/cognitive_stress_test.rs:706` — `&r.experience.content[..60.min(len)]`

---

## Agent 2: Auth & Credential Hardening

**Model**: sonnet
**Files owned**: `src/auth.rs`, `src/encryption.rs`, `src/config.rs`, `hooks/session-start.sh`, `hooks/user-prompt.sh`
**Estimated items**: 7 fixes

### Findings

| # | Severity | File:Line | Finding | Fix |
|---|----------|-----------|---------|-----|
| 1 | HIGH | `src/auth.rs:141-143, 153-155` | Dev API key returned in full in HTTP 401 error responses | Never return key in response body. Replace with `"Unauthorized. See server logs for the dev API key."` |
| 2 | HIGH | `src/auth.rs:31-32` | `tracing::warn!("Auto-generated dev API key: {key}")` logs full key | Truncate: `tracing::warn!("Auto-generated dev API key: {}...", &key[..12])` |
| 3 | MEDIUM | `src/encryption.rs` | `FieldEncryptor` does not zeroize key material on drop | Add `zeroize` crate dependency. Implement `Drop` for `FieldEncryptor` to zero key bytes. At minimum wrap the key in `Zeroizing<Vec<u8>>`. |
| 4 | MEDIUM | `hooks/session-start.sh:6` | Hardcoded fallback `sk-veld-dev-local-testing-key` — known credential in public repo | Remove hardcoded fallback. Require `VELD_API_KEY` env var or fail with actionable error. Pattern: `API_KEY="${VELD_API_KEY:?Set VELD_API_KEY}"` |
| 5 | MEDIUM | `hooks/user-prompt.sh:6` | Same hardcoded fallback key | Same fix as above |
| 6 | MEDIUM | `src/config.rs:537-539` | Rate limiting disabled by default on localhost | Add comment documenting the risk. Consider minimum rate limit of 100/s even in dev. |
| 7 | LOW | `src/auth.rs:175-199` | Constant-time comparison is correct but document the timing invariant | Add a brief comment noting the max_len iteration strategy. No code change needed. |

---

## Agent 3: Graph Concurrency & Silent Failures

**Model**: sonnet
**Files owned**: `src/graph_memory.rs`, `src/handlers/consolidation.rs`, `src/handlers/recall.rs`
**Estimated items**: 8 fixes

### Findings

| # | Severity | File:Line | Finding | Fix |
|---|----------|-----------|---------|-----|
| 1 | CRITICAL | `src/graph_memory.rs:2484-2549` | TOCTOU race in `add_relationship`: find-then-insert without acquiring `synapse_update_lock` | Acquire `synapse_update_lock` around the find_relationship_between_typed + add_relationship block |
| 2 | MEDIUM | `src/handlers/consolidation.rs:44-55` | Fire-and-forget consolidation spawn — concurrent runs on same user can double-strengthen edges | Add a per-user consolidation lock (e.g., `DashMap<String, Arc<Mutex<()>>>`) — skip if already running |
| 3 | MEDIUM | `src/graph_memory.rs:2431` | `let _ =` silently discards edge index update errors | Replace with `if let Err(e) = ... { tracing::warn!("edge index update failed: {e}"); }` |
| 4 | MEDIUM | `src/graph_memory.rs:2447` | `let _ =` silently discards edge strengthen errors | Same pattern — log at warn |
| 5 | MEDIUM | `src/graph_memory.rs:2493` | `let _ =` silently discards relationship write errors | Same pattern — log at warn |
| 6 | MEDIUM | `src/graph_memory.rs:2535` | `let _ =` silently discards graph operation errors | Same pattern — log at warn |
| 7 | MEDIUM | `src/handlers/consolidation.rs:44,124,156,415` | Multiple `let _ =` on operational results | Log failures at warn level |
| 8 | LOW | `src/handlers/recall.rs:816` | `let _ = guard.update_memory(&mem)` discards reconsolidation importance boost failure | Log at debug level |

### Performance items (lower priority, same files)

| # | Severity | File:Line | Finding | Fix |
|---|----------|-----------|---------|-----|
| 9 | MEDIUM | `src/graph_memory.rs:2044-2076` | Four separate write lock acquisitions for entity add | Consider combining three name indices into single struct behind one lock |
| 10 | LOW | `src/graph_memory.rs:2060-2075` | Entity embedding cache uses O(n) linear scan | Replace Vec with HashMap keyed by UUID |
| 11 | MEDIUM | `src/graph_memory.rs:2437-2450` | Linear scan fallback for pre-index edges (backward compat) | Add metric/log to track how often this path fires; plan removal |

---

## Agent 4: MCP Server & Hooks Hardening

**Model**: sonnet
**Files owned**: `mcp-server/index.ts`, `mcp-server/security-utils.ts`, `hooks/memory-hook.ts`
**Estimated items**: 10 fixes

### Findings

| # | Severity | File:Line | Finding | Fix |
|---|----------|-----------|---------|-----|
| 1 | HIGH | `hooks/memory-hook.ts:599` | `new URL(url).hostname` throws on invalid URLs from tool input | Wrap in try/catch: `try { new URL(url).hostname } catch { url }` |
| 2 | HIGH | `hooks/memory-hook.ts:446-508` | Tool recordings store secrets (API keys, env vars) as memories | Add secret pattern regex filter before `callBrain`: reject content matching `/(sk-\|ghp_\|AKIA\|Bearer \|password=)/i` |
| 3 | MEDIUM | `mcp-server/index.ts:1573-1574` | Health check HTTP request on every single MCP tool call | Cache health status with 5s TTL: `let lastHealthCheck = 0; let isHealthy = false;` |
| 4 | MEDIUM | `mcp-server/index.ts:1580-1581` | Internal server URL leaked in error: `"Memory server unavailable at ${API_URL}"` | Replace with `"Memory server unavailable. Check that veld is running."` |
| 5 | MEDIUM | `mcp-server/index.ts:553-558` | Raw upstream error details passed to MCP client | Wrap: `throw new Error("Connection failed. Is the veld server running?")` and `console.error` the raw error |
| 6 | MEDIUM | `mcp-server/index.ts:528` | Raw API error text exposed: `"API error ${status}: ${errorText}"` | Sanitize: include status code but not raw body |
| 7 | MEDIUM | `hooks/memory-hook.ts:244-273` | Proactive context creates self-reinforcing memory feedback loop | Cap implicit reinforcement: track per-memory-ID boost count per session, skip after 3 |
| 8 | MEDIUM | `hooks/memory-hook.ts:265-282` | Stale context served during backend outage, then auto-ingested as ground truth on recovery | Disable `auto_ingest` when serving from stale cache |
| 9 | LOW | `hooks/memory-hook.ts:889-892` | Top-level catch completely silent — hook failures invisible | Add `console.error("Hook error:", e)` |
| 10 | LOW | `mcp-server/index.ts` various | Magic numbers hardcoded (`STREAM_MIN_CONTENT_LENGTH=50`, `MAX_BUFFER_SIZE=100`, etc.) | Extract to `const CONFIG = { ... }` block with env var overrides |

---

## Agent 5: Error Handling, Audit & HTTP Response Scrubbing

**Model**: sonnet
**Files owned**: `src/errors.rs`, `src/handlers/users.rs`, `src/handlers/router.rs`, `src/server.rs`, `src/middleware.rs`, `src/handlers/state.rs`
**Estimated items**: 8 fixes

### Findings

| # | Severity | File:Line | Finding | Fix |
|---|----------|-----------|---------|-----|
| 1 | MEDIUM | `src/errors.rs:196-205` | Internal errors (StorageError, DatabaseError, LockPoisoned) include raw error messages with filesystem paths and RocksDB details in HTTP responses | Map internal errors to generic 500 messages: `"Internal server error"`. Log full detail server-side with `tracing::error!` |
| 2 | MEDIUM | `src/errors.rs:170-208` | Memory IDs passed verbatim in error text — potential log injection | Sanitize/truncate IDs in error messages |
| 3 | MEDIUM | `src/handlers/users.rs:88-89` | `list_users()` returns all user IDs without authorization — any authenticated user can enumerate tenants | Add admin-only gate: check for admin API key or restrict to specific key |
| 4 | MEDIUM | `src/handlers/router.rs:138` | `list_users` and `delete_user` have no audit logging | Add `tracing::info!("audit: list_users called by {user_id}")` for sensitive operations |
| 5 | MEDIUM | `src/server.rs:582-590` | `std::process::exit(1)` on shutdown timeout skips destructors, may corrupt RocksDB WAL | Replace with longer timeout + `abort()` after explicit flush, or remove the hard exit |
| 6 | LOW | `src/handlers/router.rs:53` | `/graph/view` HTML endpoint on public (unauthenticated) route | Move behind auth middleware or document intentionally public |
| 7 | LOW | `src/middleware.rs:139` | `req.method().to_string()` allocates per request | Use `req.method().as_str()` instead |
| 8 | LOW | `src/server.rs:119` | `tracing_setup::init_tracing().expect(...)` panics on tracing init failure | Acceptable but document |

---

## Verification

After all 5 agents complete:
1. `cargo check` must pass
2. `cargo clippy` must pass with no new warnings
3. No `&str[..n]` patterns on user-derived strings remain in tui/
4. No API keys appear in HTTP response bodies
5. No `let _ =` on write operations in graph_memory.rs
6. `grep -rn 'sk-veld-dev-local-testing-key' hooks/` returns nothing
