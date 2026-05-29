#!/usr/bin/env bun
/**
 * Veld Hook - Native Claude Code Integration
 *
 * Aggressive proactive context surfacing at every opportunity.
 * Memory should be woven into every interaction - the AI thinks with memory.
 *
 * Events: SessionStart, SessionEnd, UserPromptSubmit, PreToolUse, PostToolUse, SubagentStop, Stop
 */

import { existsSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";

const VELD_API_URL = process.env.VELD_API_URL || "http://127.0.0.1:3030";
const VELD_API_KEY = resolveApiKey();
const VELD_USER_ID = process.env.VELD_USER_ID || "claude-code";

/**
 * Resolve the Veld API key, in order:
 *   1. VELD_API_KEY env var (if non-empty)
 *   2. `api_key = "..."` line in the platform's veld config.toml
 *   3. empty string (server may reject; backendDown will trip)
 *
 * Exported for testing.
 */
export function resolveApiKey(): string {
  const envKey = process.env.VELD_API_KEY;
  if (envKey && envKey.trim()) return envKey.trim();

  const path = veldConfigPath();
  if (!existsSync(path)) return "";

  try {
    const text = readFileSync(path, "utf-8");
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const eq = line.indexOf("=");
      if (eq === -1) continue;
      if (line.slice(0, eq).trim() !== "api_key") continue;
      const value = line
        .slice(eq + 1)
        .split("#")[0]
        .trim()
        .replace(/^["']|["']$/g, "");
      if (value) return value;
    }
  } catch {
    // unreadable / corrupt config — fall through
  }
  return "";
}

/** Exported for testing. */
export function veldConfigPath(): string {
  const xdg = process.env.XDG_CONFIG_HOME;
  if (xdg) return join(xdg, "veld", "config.toml");
  if (process.platform === "win32") {
    const appdata = process.env.APPDATA;
    if (appdata) return join(appdata, "veld", "config.toml");
  } else if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", "veld", "config.toml");
  }
  return join(homedir(), ".config", "veld", "config.toml");
}

interface HookInput {
  hook_event_name: string;
  session_id?: string;
  transcript_path?: string;
  cwd?: string;
  // UserPromptSubmit
  prompt?: string;
  // PreToolUse / PostToolUse
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: string;
  tool_response?: unknown;
  // Stop
  stop_reason?: string;
  // SubagentStop
  agent_id?: string;
  agent_type?: string;
  agent_transcript_path?: string;
  // Legacy field names (backward compat)
  subagent_type?: string;
  subagent_result?: string;
}

interface SurfacedMemory {
  id: string;
  content: string;
  memory_type: string;
  score: number;
  importance: number;
  created_at: string;
  tags: string[];
  relevance_reason: string;
  matched_entities: string[];
}

interface ProactiveContextResponse {
  memories: SurfacedMemory[];
  due_reminders: unknown[];
  context_reminders: unknown[];
  memory_count: number;
  reminder_count: number;
  ingested_memory_id: string | null;
  feedback_processed: { memories_evaluated: number; reinforced: string[]; weakened: string[] } | null;
  relevant_todos: { id: string; short_id: string; content: string; status: string; priority: string; project: string | null; due_date: string | null; relevance_reason: string }[];
  todo_count: number;
  relevant_facts: { id: string; fact: string; confidence: number; support_count: number; related_entities: string[] }[];
  latency_ms: number;
  detected_entities: { name: string; entity_type: string }[];
}

const HOOK_TIMEOUT_MS = 5000;

/** Regex for detecting secrets that must not be stored in memory */
const SECRET_PATTERN = /(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36,}|AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._\-]{20,}|password\s*[=:]\s*\S+)/i;

function containsSecrets(text: string): boolean {
  return SECRET_PATTERN.test(text);
}

/** Per-session boost counter: skip implicit reinforcement after 3 boosts per memory ID */
const sessionBoostCount = new Map<string, number>();
const MAX_IMPLICIT_BOOSTS = 3;

/** Backend availability tracking for degradation signaling */
let backendDown = false;
let lastSuccessfulContext: string | null = null;
let lastContextTimestamp: number = 0;

/** Tool actions collected since last proactive_context call for feedback attribution */
const pendingToolActions: { tool_name: string; inputs: Record<string, string>; success: boolean; output_snippet?: string }[] = [];

// =============================================================================
// EPISODE THREADING (auto-generated per session)
// =============================================================================

/** Unique episode ID for this session — enables temporal clustering & ordinal boost */
const sessionEpisodeId = `ep-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;

/** Monotonic sequence counter for memory ordering within this session */
let episodeSequenceNumber = 0;

/** ID of the last memory we stored (for preceding_memory_id chains) */
let lastStoredMemoryId: string | null = null;

// =============================================================================
// FEEDBACK LOOP (track surfaced memories for relevance signaling)
// =============================================================================

/** IDs of memories surfaced in the most recent proactive_context call */
let lastSurfacedMemoryIds: string[] = [];

/** Formatted output from the last proactive_context (for implicit feedback via previous_response) */
let lastProactiveOutput: string | null = null;

// =============================================================================
// EMOTIONAL CLASSIFICATION
// =============================================================================

interface EmotionalSignal {
  emotional_valence: number;  // -1.0 (negative) to 1.0 (positive)
  emotional_arousal: number;  // 0.0 (calm) to 1.0 (highly aroused)
  emotion: string;
}

/** Classify emotional context from event type and outcome */
function classifyEmotion(eventType: string, success: boolean, content: string): EmotionalSignal {
  // Error events: high arousal, negative valence
  if (eventType === "Error" || (!success && content.includes("error"))) {
    return { emotional_valence: -0.6, emotional_arousal: 0.8, emotion: "frustration" };
  }

  // Task/orchestration completions: positive, medium arousal
  if (eventType === "Task" && success) {
    return { emotional_valence: 0.5, emotional_arousal: 0.5, emotion: "satisfaction" };
  }

  // File modifications: neutral, low arousal (routine work)
  if (eventType === "FileAccess") {
    return { emotional_valence: 0.1, emotional_arousal: 0.2, emotion: "focus" };
  }

  // Discovery/learning: positive, medium-high arousal
  if (eventType === "Discovery" || eventType === "Learning") {
    return { emotional_valence: 0.6, emotional_arousal: 0.6, emotion: "curiosity" };
  }

  // Decisions: neutral-positive, medium arousal
  if (eventType === "Decision") {
    return { emotional_valence: 0.3, emotional_arousal: 0.4, emotion: "resolve" };
  }

  // Default: neutral observation
  return { emotional_valence: 0.0, emotional_arousal: 0.15, emotion: "neutral" };
}

async function callBrain(endpoint: string, body: Record<string, unknown>): Promise<unknown> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
    const response = await fetch(`${VELD_API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": VELD_API_KEY,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      backendDown = true;
      return null;
    }
    backendDown = false;
    return await response.json();
  } catch {
    backendDown = true;
    return null;
  }
}

export function formatRelativeTime(isoDate: string): string {
  const d = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function formatMemoriesForContext(memories: SurfacedMemory[]): string {
  if (!memories.length) return "";

  return memories
    .map((m) => {
      const time = formatRelativeTime(m.created_at);
      const score = Math.round(m.score * 100);
      return `• [${score}%] (${time}) ${m.content.slice(0, 120)}${m.content.length > 120 ? "..." : ""}`;
    })
    .join("\n");
}

export function isErrorOutput(toolOutput: string): boolean {
  return (
    toolOutput.includes("error") ||
    toolOutput.includes("Error") ||
    toolOutput.includes("failed") ||
    toolOutput.includes("FAILED")
  );
}

export function buildPreToolContext(toolName: string, toolInput: Record<string, unknown>): string {
  if (toolName === "Edit" || toolName === "Write" || toolName === "MultiEdit") {
    const filePath = toolInput.file_path as string;
    if (filePath) {
      return `Editing file: ${filePath}`;
    }
    const edits = toolInput.edits as Array<{ file_path?: string }> | undefined;
    if (edits?.length) {
      return `Multi-editing ${edits.length} file(s): ${edits.map((e) => e.file_path).filter(Boolean).slice(0, 3).join(", ")}`;
    }
  } else if (toolName === "Bash") {
    const command = toolInput.command as string;
    if (command) {
      return `Running command: ${command.slice(0, 100)}`;
    }
  } else if (toolName === "Read") {
    const filePath = toolInput.file_path as string;
    if (filePath) {
      return `Reading file: ${filePath}`;
    }
  } else if (toolName === "Task") {
    const prompt = toolInput.prompt as string;
    if (prompt) {
      return `Spawning agent: ${prompt.slice(0, 100)}`;
    }
  } else if (toolName === "Search" || toolName === "Grep" || toolName === "Glob") {
    const query = (toolInput.query as string) || (toolInput.pattern as string) || "";
    if (query) {
      return `Searching: ${query.slice(0, 100)}`;
    }
  } else if (toolName === "WebFetch") {
    const url = toolInput.url as string;
    if (url) {
      return `Fetching: ${url}`;
    }
  } else if (toolName === "NotebookEdit") {
    const notebook = (toolInput.notebook as string) || (toolInput.file_path as string) || "";
    if (notebook) {
      return `Editing notebook: ${notebook}`;
    }
  }

  return `About to use ${toolName}`;
}

/** Debounce window (ms) for proactive_context calls — keyed by context prefix.
 *  Reduces server-side embedding load when parallel agents fan out identical
 *  tool calls. Tunable via VELD_HOOK_DEBOUNCE_MS. */
const PROACTIVE_DEBOUNCE_MS = (() => {
  const raw = process.env.VELD_HOOK_DEBOUNCE_MS;
  const n = raw ? Number.parseInt(raw, 10) : 500;
  return Number.isFinite(n) && n >= 0 ? n : 500;
})();

interface ProactiveCacheEntry {
  at: number;
  result: string | null;
}
/** Last (timestamp, result) per context-key. Bounded; oldest entries evicted. */
const proactiveDebounceCache = new Map<string, ProactiveCacheEntry>();
const PROACTIVE_DEBOUNCE_MAX_KEYS = 64;

function proactiveDebounceKey(context: string, maxResults: number): string {
  // 80-char prefix lets "Editing file: a.rs" and "Editing file: b.rs" debounce
  // independently, while two Edits of the same file within the window coalesce.
  return `${maxResults}|${context.slice(0, 80)}`;
}

async function surfaceProactiveContext(context: string, maxResults = 3, autoIngest = false): Promise<string | null> {
  // P2 debounce: parallel agents firing identical proactive_context queries
  // hammer the Nomic embedder for no extra information. Skip the round-trip
  // if we answered the same question within VELD_HOOK_DEBOUNCE_MS.
  //
  // Two important carve-outs:
  // - `autoIngest`: UserPromptSubmit relies on this to *record* the prompt;
  //   debouncing would drop ingestion. Pass through always.
  // - `backendDown`: when the server is recovering we want every call to try,
  //   so the offline → online transition is visible quickly.
  if (PROACTIVE_DEBOUNCE_MS > 0 && !autoIngest && !backendDown) {
    const key = proactiveDebounceKey(context, maxResults);
    const cached = proactiveDebounceCache.get(key);
    const now = Date.now();
    if (cached && now - cached.at < PROACTIVE_DEBOUNCE_MS) {
      return cached.result;
    }
  }

  // Drain pending tool actions for feedback attribution
  const toolActions = pendingToolActions.splice(0, pendingToolActions.length);

  // Build feedback payload from previous cycle, capping per-memory implicit boosts
  const feedbackPayload: Record<string, unknown> = {};
  if (lastProactiveOutput) {
    feedbackPayload.previous_response = lastProactiveOutput;
    feedbackPayload.user_followup = context.slice(0, 1000);
  }
  if (lastSurfacedMemoryIds.length > 0) {
    // Only include IDs that haven't been reinforced too many times this session
    const eligibleIds = lastSurfacedMemoryIds.filter((id) => {
      const count = sessionBoostCount.get(id) ?? 0;
      return count < MAX_IMPLICIT_BOOSTS;
    });
    if (eligibleIds.length > 0) {
      feedbackPayload.surfaced_memory_ids = eligibleIds;
      // Increment boost counters for these IDs
      for (const id of eligibleIds) {
        sessionBoostCount.set(id, (sessionBoostCount.get(id) ?? 0) + 1);
      }
    }
  }

  // When the backend is down, disable auto_ingest — stale context must not be
  // re-ingested as ground truth when the backend recovers.
  const effectiveAutoIngest = autoIngest && !backendDown;

  const response = (await callBrain("/api/proactive_context", {
    user_id: VELD_USER_ID,
    context,
    max_results: maxResults,
    semantic_threshold: 0.6,
    entity_match_weight: 0.3,
    recency_weight: 0.2,
    auto_ingest: effectiveAutoIngest,
    // FIX-05: Propagate episode + emotional context to auto-ingested memories
    episode_id: sessionEpisodeId,
    sequence_number: episodeSequenceNumber,
    emotional_valence: 0.0,
    emotional_arousal: 0.15,
    ...(toolActions.length > 0 ? { tool_actions: toolActions } : {}),
    ...feedbackPayload,
  })) as ProactiveContextResponse | null;

  if (!response) {
    // Backend unreachable — serve stale cache with age warning.
    // Do NOT update lastProactiveOutput so stale content is never fed back as feedback.
    if (lastSuccessfulContext && lastContextTimestamp > 0) {
      const ageMinutes = Math.round((Date.now() - lastContextTimestamp) / 60_000);
      return `⚠️ Memory offline (${ageMinutes}m stale cache):\n${lastSuccessfulContext}`;
    }
    return null;
  }

  const hasMemories = response.memories?.length > 0;
  const hasFacts = response.relevant_facts?.length > 0;
  const hasTodos = response.relevant_todos?.length > 0;
  const hasReminders = (response.due_reminders?.length || 0) + (response.context_reminders?.length || 0) > 0;

  if (!hasMemories && !hasFacts && !hasTodos && !hasReminders) return null;

  const now = new Date();
  const header = `📅 ${now.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })} ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

  let output = header;
  if (hasMemories) {
    output += `\n${formatMemoriesForContext(response.memories)}`;
  }
  if (hasFacts) {
    output += `\n🧠 Facts:`;
    for (const f of response.relevant_facts.slice(0, 3)) {
      output += `\n• (${Math.round(f.confidence * 100)}%) ${f.fact}`;
    }
  }
  if (hasTodos) {
    output += `\n📋 Todos:`;
    for (const t of response.relevant_todos.slice(0, 3)) {
      const icon = t.status === "in_progress" ? "🔄" : "☐";
      output += `\n${icon} ${t.content.slice(0, 80)}`;
    }
  }
  if (hasReminders) {
    const allReminders = [...(response.due_reminders || []), ...(response.context_reminders || [])];
    output += `\n⏰ ${allReminders.length} reminder(s) active`;
  }

  // Cache successful context for stale-serve on future failures
  lastSuccessfulContext = output;
  lastContextTimestamp = Date.now();

  // Track surfaced memories for feedback loop on next call
  lastSurfacedMemoryIds = (response.memories || []).map((m) => m.id);
  lastProactiveOutput = output;

  // P2 debounce: record the result so the next identical call within the
  // window can answer locally. Cache only success paths — backend-down
  // returns intentionally bypass the cache so recovery is visible quickly.
  if (PROACTIVE_DEBOUNCE_MS > 0 && !autoIngest) {
    const key = proactiveDebounceKey(context, maxResults);
    // LRU-ish eviction: when full, drop the oldest entry. Maps preserve
    // insertion order, so the first key() is the oldest.
    if (proactiveDebounceCache.size >= PROACTIVE_DEBOUNCE_MAX_KEYS) {
      const oldest = proactiveDebounceCache.keys().next().value;
      if (oldest !== undefined) proactiveDebounceCache.delete(oldest);
    }
    proactiveDebounceCache.set(key, { at: Date.now(), result: output });
  }

  return output;
}

// =============================================================================
// AGENT SESSION MARKER FILE
// =============================================================================
//
// On SessionStart we write `.veld-agent-session.<pid>` into the current
// working directory so external tooling (notably the upcoming agent-session
// detection helper) can identify which chat brand is driving this process.
//
// Format (JSON):
//   {
//     "agent_id":   "Claude",          // chat brand — always "Claude" for this hook
//     "started_at": ISO-8601 UTC,
//     "pid":        <process pid>,
//     "binary":     basename of process.argv[0] (best-effort, diagnostics only)
//   }
//
// Written atomically (write to .tmp then rename) and idempotent (overwrites
// on every SessionStart). Removed on SessionEnd best-effort.

export interface AgentSessionMarker {
  agent_id: string;
  started_at: string;
  pid: number;
  binary: string;
}

/** Absolute path to the agent-session marker file for the current process. */
export function agentSessionMarkerPath(cwd: string = process.cwd(), pid: number = process.pid): string {
  return join(cwd, `.veld-agent-session.${pid}`);
}

/**
 * Best-effort detection of the launcher binary. Returns the basename of
 * `process.argv[0]` (e.g. "bun.exe", "node", "claude-code"). Falls back to
 * a marker string if argv[0] is missing or empty. Used by downstream tooling
 * for diagnostics; not load-bearing.
 */
export function detectBinary(): string {
  if (process.env.CLAUDE_DESKTOP) return "claude-desktop";
  const argv0 = process.argv[0];
  if (!argv0) return "unknown";
  return basename(argv0) || "unknown";
}

/** Build the marker payload. Pure for testability. */
export function buildAgentSessionMarker(now: Date = new Date()): AgentSessionMarker {
  return {
    agent_id: "Claude",
    started_at: now.toISOString(),
    pid: process.pid,
    binary: detectBinary(),
  };
}

/**
 * Write the marker file atomically. Writes to `<path>.tmp` then renames over
 * the target so readers never observe a half-written file.
 * Errors are logged and swallowed — the hook must never crash the host.
 */
export function writeAgentSessionMarker(path: string = agentSessionMarkerPath()): AgentSessionMarker | null {
  const marker = buildAgentSessionMarker();
  const tmp = `${path}.tmp`;
  try {
    writeFileSync(tmp, JSON.stringify(marker, null, 2), { encoding: "utf-8" });
    renameSync(tmp, path);
    return marker;
  } catch (e) {
    console.error(`[veld] agent-session marker write failed at ${path}:`, e);
    try { unlinkSync(tmp); } catch { /* tmp may not exist */ }
    return null;
  }
}

/**
 * Delete the marker file. Best-effort — a missing file is not an error.
 * Logs a debug message on unexpected failure (e.g. permission denied).
 */
export function removeAgentSessionMarker(path: string = agentSessionMarkerPath()): void {
  try {
    unlinkSync(path);
  } catch (e: unknown) {
    const code = (e as NodeJS.ErrnoException | undefined)?.code;
    if (code === "ENOENT") return; // already gone — nothing to do
    console.error(`[veld] agent-session marker delete failed at ${path}:`, e);
  }
}

async function handleSessionStart(): Promise<void> {
  // Drop the agent-session marker file first so downstream chat-brand
  // detection works even if the memory backend is unreachable below.
  writeAgentSessionMarker();

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const projectName = projectDir.split(/[/\\]/).pop() || "unknown";

  const context = `Starting session in project: ${projectName}`;
  const memoryContext = await surfaceProactiveContext(context, 5);

  const memoryFile = `${projectDir}/.claude/memory-context.md`;
  if (memoryContext) {
    console.error(`[veld] Session context loaded`);

    try {
      await Bun.write(memoryFile, `# Veld Context\n\n${memoryContext}\n`);
    } catch {
      // Directory might not exist
    }
  } else if (backendDown) {
    console.error(`[veld] Memory system unreachable — operating without persistent context`);
    try {
      await Bun.write(memoryFile, `# Veld Context\n\n⚠️ Memory system offline. No persistent context available.\n`);
    } catch {
      // Directory might not exist
    }
  }
}

async function handleUserPrompt(input: HookInput): Promise<void> {
  const prompt = input.prompt;
  if (!prompt || prompt.length < 10) return;

  // Single call: surface memories AND ingest the prompt in one pipeline pass
  const memoryContext = await surfaceProactiveContext(prompt.slice(0, 1000), 3, true);

  if (memoryContext) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "UserPromptSubmit",
          additionalContext: `\n<veld>\n${memoryContext}\n</veld>`,
        },
      })
    );
  } else if (backendDown) {
    // Inject a single-line degradation warning so the model knows memory is offline
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "UserPromptSubmit",
          additionalContext: `\n<veld-warning>Memory system offline. No persistent context for this prompt.</veld-warning>`,
        },
      })
    );
  }
}

async function handlePreToolUse(input: HookInput): Promise<void> {
  const toolName = input.tool_name;
  const toolInput = input.tool_input;
  if (!toolName || !toolInput) return;

  const context = buildPreToolContext(toolName, toolInput);

  // Surface relevant context BEFORE the tool runs
  const memoryContext = await surfaceProactiveContext(context, 2);

  if (memoryContext) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          additionalContext: `\n<veld context="pre-${toolName.toLowerCase()}">\n${memoryContext}\n</veld>`,
        },
      })
    );
  }
}

async function handlePostToolUse(input: HookInput): Promise<void> {
  const toolName = input.tool_name;
  const toolInput = input.tool_input;
  const toolOutput = input.tool_output;

  if (!toolName) return;

  // Record tool action for feedback attribution (before any early returns)
  if (toolName !== "Task") {
    const actionRecord: (typeof pendingToolActions)[number] = {
      tool_name: toolName,
      inputs: {},
      success: true,
    };
    if (toolInput) {
      for (const [k, v] of Object.entries(toolInput)) {
        if (typeof v === "string") {
          actionRecord.inputs[k] = v.slice(0, 500);
        }
      }
    }
    if (toolOutput) {
      actionRecord.success = !isErrorOutput(toolOutput);
      actionRecord.output_snippet = toolOutput.slice(0, 200);
    }
    pendingToolActions.push(actionRecord);
    if (pendingToolActions.length > 50) {
      pendingToolActions.splice(0, pendingToolActions.length - 50);
    }
  }

  // Orchestration: handle Task tool completions
  if (toolName === "Task") {
    await handlePostToolUseTask(input);
    return;
  }

  // Store significant tool uses with emotional context + episode threading
  if (toolName === "Edit" || toolName === "Write") {
    const filePath = toolInput?.file_path as string;
    if (filePath) {
      // Build descriptive content based on tool type (FIX-01: structured reasoning capture)
      let content: string;
      if (toolName === "Edit") {
        const oldStr = (toolInput?.old_string as string) || "";
        const newStr = (toolInput?.new_string as string) || "";
        content = `Edited ${filePath}: replaced ${oldStr.slice(0, 80)} → ${newStr.slice(0, 80)}`;
      } else {
        const fileContent = (toolInput?.content as string) || "";
        content = `Created/wrote ${filePath} (${fileContent.length} chars)`;
      }

      if (containsSecrets(content)) return;

      const emotion = classifyEmotion("FileAccess", true, filePath);
      const seq = ++episodeSequenceNumber;
      const resp = await callBrain("/api/remember", {
        user_id: VELD_USER_ID,
        content,
        memory_type: "FileAccess",
        tags: [`tool:${toolName}`, `file:${filePath.split(/[/\\]/).pop()}`],
        source_type: "system",
        credibility: 0.8,
        episode_id: sessionEpisodeId,
        sequence_number: seq,
        ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
        ...emotion,
      }) as { id?: string } | null;
      if (resp?.id) lastStoredMemoryId = resp.id;
    }
  } else if (toolName === "Bash" && toolOutput) {
    const command = toolInput?.command as string;

    // Store errors/failures for learning
    if (isErrorOutput(toolOutput)) {
      const bashContent = `Command failed: ${command?.slice(0, 100)} → ${toolOutput.slice(0, 200)}`;
      if (containsSecrets(bashContent)) return;
      const emotion = classifyEmotion("Error", false, toolOutput);
      const seq = ++episodeSequenceNumber;
      const resp = await callBrain("/api/remember", {
        user_id: VELD_USER_ID,
        content: bashContent,
        memory_type: "Error",
        tags: ["tool:Bash", "error"],
        source_type: "system",
        credibility: 0.9,
        episode_id: sessionEpisodeId,
        sequence_number: seq,
        ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
        ...emotion,
      }) as { id?: string } | null;
      if (resp?.id) lastStoredMemoryId = resp.id;

      // Surface past errors for this type of command
      const memoryContext = await surfaceProactiveContext(
        `Error with command: ${command?.slice(0, 100)}`,
        2
      );
      if (memoryContext) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<veld context="similar-errors">\n${memoryContext}\n</veld>`,
            },
          })
        );
      }
    }
  } else if (toolName === "Read") {
    const filePath = toolInput?.file_path as string;
    if (filePath) {
      // Surface what we know about this file
      const memoryContext = await surfaceProactiveContext(
        `Reading file: ${filePath}`,
        2
      );
      if (memoryContext) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<veld context="file-context">\n${memoryContext}\n</veld>`,
            },
          })
        );
      }
    }
  } else if (toolName === "MultiEdit") {
    // MultiEdit: batch file modifications — record each affected file
    const edits = toolInput?.edits as Array<{ file_path?: string }> | undefined;
    if (edits?.length) {
      const files = edits.map((e) => e.file_path).filter(Boolean).slice(0, 5);
      const content = `Multi-edited ${edits.length} file(s): ${files.join(", ")}`;
      const emotion = classifyEmotion("FileAccess", true, content);
      const seq = ++episodeSequenceNumber;
      const resp = await callBrain("/api/remember", {
        user_id: VELD_USER_ID,
        content,
        memory_type: "FileAccess",
        tags: ["tool:MultiEdit", ...files.map((f) => `file:${f!.split(/[/\\]/).pop()}`)],
        source_type: "system",
        credibility: 0.8,
        episode_id: sessionEpisodeId,
        sequence_number: seq,
        ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
        ...emotion,
      }) as { id?: string } | null;
      if (resp?.id) lastStoredMemoryId = resp.id;
    }
  } else if (toolName === "Search" || toolName === "Grep" || toolName === "Glob") {
    // Search tools: surface related memories for the query
    const query = (toolInput?.query as string) || (toolInput?.pattern as string) || "";
    if (query) {
      const memoryContext = await surfaceProactiveContext(
        `Searching codebase: ${query.slice(0, 200)}`,
        2
      );
      if (memoryContext) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<veld context="search-context">\n${memoryContext}\n</veld>`,
            },
          })
        );
      }
    }
  } else if (toolName === "ListFiles" || toolName === "LS") {
    // Directory listing: surface what we know about the directory
    const dirPath = (toolInput?.path as string) || (toolInput?.directory as string) || "";
    if (dirPath) {
      const memoryContext = await surfaceProactiveContext(
        `Exploring directory: ${dirPath}`,
        2
      );
      if (memoryContext) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<veld context="directory-context">\n${memoryContext}\n</veld>`,
            },
          })
        );
      }
    }
  } else if (toolName === "WebFetch") {
    // Web fetch: store fetched URL as context for future reference
    const url = (toolInput?.url as string) || "";
    if (url && toolOutput) {
      const content = `Fetched ${url}: ${toolOutput.slice(0, 200)}`;
      if (containsSecrets(content)) return;
      let urlHostname: string;
      try { urlHostname = new URL(url).hostname; } catch { urlHostname = url.slice(0, 100); }
      const emotion = classifyEmotion("Discovery", true, content);
      const seq = ++episodeSequenceNumber;
      const resp = await callBrain("/api/remember", {
        user_id: VELD_USER_ID,
        content,
        memory_type: "Context",
        tags: ["tool:WebFetch", `url:${urlHostname}`],
        source_type: "system",
        credibility: 0.6,
        episode_id: sessionEpisodeId,
        sequence_number: seq,
        ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
        ...emotion,
      }) as { id?: string } | null;
      if (resp?.id) lastStoredMemoryId = resp.id;
    }
  } else if (toolName === "NotebookEdit") {
    // Notebook edits: record like file edits
    const notebook = (toolInput?.notebook as string) || (toolInput?.file_path as string) || "";
    if (notebook) {
      const content = `Edited notebook ${notebook}`;
      const emotion = classifyEmotion("FileAccess", true, content);
      const seq = ++episodeSequenceNumber;
      const resp = await callBrain("/api/remember", {
        user_id: VELD_USER_ID,
        content,
        memory_type: "FileAccess",
        tags: ["tool:NotebookEdit", `file:${notebook.split(/[/\\]/).pop()}`],
        source_type: "system",
        credibility: 0.7,
        episode_id: sessionEpisodeId,
        sequence_number: seq,
        ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
        ...emotion,
      }) as { id?: string } | null;
      if (resp?.id) lastStoredMemoryId = resp.id;
    }
  }
}

// --- Orchestration: PostToolUse(Task) handler ---

const ORCH_TAG_RE = /\[ORCH-TODO:([A-Z]+-\d+)\]/;

async function callBrainGet(endpoint: string): Promise<unknown> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
    const response = await fetch(`${VELD_API_URL}${endpoint}`, {
      headers: { "X-API-Key": VELD_API_KEY },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

async function unblockDependents(completedShortId: string): Promise<void> {
  const dashIdx = completedShortId.lastIndexOf("-");
  if (dashIdx < 0) return;

  // List all projects to find the one matching this prefix
  // API returns [project, stats] tuples in the projects array
  const projectsResp = (await callBrain("/api/projects/list", {
    user_id: VELD_USER_ID,
  })) as { projects?: Array<[{ id: string; name: string; prefix?: string }, unknown]> } | null;

  if (!projectsResp?.projects?.length) return;

  const prefix = completedShortId.substring(0, dashIdx).toUpperCase();
  const projectEntry = projectsResp.projects.find(
    (entry) => {
      const p = Array.isArray(entry) ? entry[0] : entry;
      return (p.prefix || "").toUpperCase() === prefix;
    }
  );
  if (!projectEntry) return;
  const project = Array.isArray(projectEntry) ? projectEntry[0] : projectEntry;

  // List blocked todos in this project
  const todosResp = (await callBrain("/api/todos/list", {
    user_id: VELD_USER_ID,
    project: project.name,
    status: ["blocked"],
  })) as { todos?: Array<{ id: string; seq_num?: number; project_prefix?: string; blocked_on?: string }> } | null;

  if (!todosResp?.todos?.length) return;

  for (const todo of todosResp.todos) {
    if (!todo.blocked_on) continue;

    const blockers = todo.blocked_on.split(",").map((s) => s.trim());
    const remaining = blockers.filter((b) => b !== completedShortId);
    // Construct short_id from project_prefix + seq_num, fall back to UUID
    const todoId = todo.project_prefix && todo.seq_num != null
      ? `${todo.project_prefix}-${todo.seq_num}`
      : todo.id;

    if (remaining.length === 0) {
      // All blockers resolved — unblock
      await callBrain(`/api/todos/${todoId}/update`, {
        user_id: VELD_USER_ID,
        status: "todo",
        blocked_on: "",
      });
      await callBrain(`/api/todos/${todoId}/comments`, {
        user_id: VELD_USER_ID,
        content: `Unblocked: dependency ${completedShortId} completed`,
        comment_type: "activity",
      });
    } else {
      // Some blockers remain — update the list
      await callBrain(`/api/todos/${todoId}/update`, {
        user_id: VELD_USER_ID,
        blocked_on: remaining.join(","),
      });
    }
  }
}

async function handlePostToolUseTask(input: HookInput): Promise<void> {
  const toolInput = input.tool_input;
  const toolResult = input.tool_output ?? input.tool_response;

  if (!toolInput) return;

  const prompt = (toolInput.prompt as string) || "";
  const resultText = typeof toolResult === "string"
    ? toolResult
    : toolResult != null
      ? JSON.stringify(toolResult)
      : "";

  // Check for orchestration tag
  const tagMatch = prompt.match(ORCH_TAG_RE);

  if (!tagMatch) {
    // Not an orchestration task — store as generic memory
    if (resultText) {
      const emotion = classifyEmotion("Task", true, resultText);
      const seq = ++episodeSequenceNumber;
      const resp = await callBrain("/api/remember", {
        user_id: VELD_USER_ID,
        content: `Task agent completed: ${resultText.slice(0, 300)}`,
        memory_type: "Task",
        tags: ["subagent:task", "source:hook"],
        source_type: "system",
        credibility: 0.7,
        episode_id: sessionEpisodeId,
        sequence_number: seq,
        ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
        ...emotion,
      }) as { id?: string } | null;
      if (resp?.id) lastStoredMemoryId = resp.id;
    }
    return;
  }

  const todoShortId = tagMatch[1];

  // 1. Add Resolution comment with agent result
  if (resultText) {
    await callBrain(`/api/todos/${todoShortId}/comments`, {
      user_id: VELD_USER_ID,
      content: resultText.slice(0, 4000),
      comment_type: "resolution",
    });
  }

  // 2. Complete the todo (path-based endpoint)
  await callBrain(`/api/todos/${todoShortId}/complete`, {
    user_id: VELD_USER_ID,
  });

  // 3. Unblock dependents — retry once on failure to prevent orchestration deadlocks
  try {
    await unblockDependents(todoShortId);
  } catch (e) {
    console.error(`[veld] unblockDependents failed for ${todoShortId}, retrying: ${e}`);
    try {
      await unblockDependents(todoShortId);
    } catch (e2) {
      console.error(`[veld] unblockDependents retry failed for ${todoShortId}: ${e2}`);
    }
  }

  // 4. Store memory of orchestration completion with emotional + episode context
  {
    const emotion = classifyEmotion("Task", true, resultText);
    const seq = ++episodeSequenceNumber;
    const resp = await callBrain("/api/remember", {
      user_id: VELD_USER_ID,
      content: `Orchestration task ${todoShortId} completed: ${resultText.slice(0, 200)}`,
      memory_type: "Task",
      tags: ["orchestration", `todo:${todoShortId}`, "source:hook"],
      source_type: "system",
      credibility: 0.75,
      episode_id: sessionEpisodeId,
      sequence_number: seq,
      ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
      ...emotion,
    }) as { id?: string } | null;
    if (resp?.id) lastStoredMemoryId = resp.id;
  }

  // 5. Surface orchestration status
  const memoryContext = await surfaceProactiveContext(
    `Orchestration: task ${todoShortId} completed, checking for unblocked work`,
    2
  );
  if (memoryContext) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          additionalContext: `\n<veld context="orchestration">\n${memoryContext}\n</veld>`,
        },
      })
    );
  }
}

async function handleSubagentStop(input: HookInput): Promise<void> {
  const agentType = input.agent_type || input.subagent_type;
  const agentId = input.agent_id;
  const result = input.subagent_result;

  if (!agentType) return;

  const content = result
    ? `${agentType} agent completed: ${result.slice(0, 300)}`
    : `${agentType} agent (${agentId || "unknown"}) completed`;

  const emotion = classifyEmotion("Task", true, content);
  const seq = ++episodeSequenceNumber;
  const resp = await callBrain("/api/remember", {
    user_id: VELD_USER_ID,
    content,
    memory_type: "Task",
    tags: [`subagent:${agentType}`, "source:hook"],
    source_type: "system",
    credibility: 0.6,
    episode_id: sessionEpisodeId,
    sequence_number: seq,
    ...(lastStoredMemoryId ? { preceding_memory_id: lastStoredMemoryId } : {}),
    ...emotion,
  }) as { id?: string } | null;
  if (resp?.id) lastStoredMemoryId = resp.id;
}

async function handleStop(_input: HookInput): Promise<void> {
  // Session end is tracked implicitly by memory timestamps and decay.
  // Storing explicit "Session ended" memories creates noise in the activity log
  // and gets re-ingested by proactive_context auto-ingest, causing duplicate events.
}

async function handleSessionEnd(_input: HookInput): Promise<void> {
  // Remove the agent-session marker file written at SessionStart.
  // Best-effort — leaks are not critical: the file is gitignored and lives in
  // the worktree, so the next SessionStart will overwrite it anyway.
  removeAgentSessionMarker();
}

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();

  let input: HookInput;
  try {
    input = JSON.parse(inputText);
  } catch {
    const eventType = process.argv[2];
    input = { hook_event_name: eventType || "SessionStart" };
  }

  const eventName = input.hook_event_name;

  switch (eventName) {
    case "SessionStart":
      await handleSessionStart();
      break;
    case "UserPromptSubmit":
      await handleUserPrompt(input);
      break;
    case "PreToolUse":
      await handlePreToolUse(input);
      break;
    case "PostToolUse":
      await handlePostToolUse(input);
      break;
    case "SubagentStop":
      await handleSubagentStop(input);
      break;
    case "Stop":
      await handleStop(input);
      break;
    case "SessionEnd":
      await handleSessionEnd(input);
      break;
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    // Log but do not rethrow — hooks must not crash Claude Code
    console.error("[veld] Hook error:", e);
  }
}
