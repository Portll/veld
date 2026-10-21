#!/usr/bin/env bun
/**
 * Veld Hook - Native Claude Code Integration
 *
 * Surfaces relevant memories into the agent's context (SessionStart, UserPromptSubmit,
 * PreToolUse) and records the captures worth keeping: command failures, subagent
 * completions and orchestration progress.
 *
 * Claude Code spawns a fresh process for every event, so state that must span events
 * (episode threading, feedback loop, debounce cache, pending tool actions) lives in a
 * per-session JSON file under VELD_HOOK_STATE_DIR (default: <tmpdir>/veld-hook).
 *
 * Events: SessionStart, SessionEnd, UserPromptSubmit, PreToolUse, PostToolUse,
 * PostToolUseFailure, SubagentStop, Stop
 */

import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join } from "node:path";

const VELD_API_URL = process.env.VELD_API_URL || "http://127.0.0.1:3030";
const VELD_API_KEY = resolveApiKey();
const VELD_USER_ID = process.env.VELD_USER_ID || "claude-code";

/**
 * Resolve the Veld API key, in order:
 *   1. VELD_API_KEY env var (if non-empty)
 *   2. `api_key = "..."` line in the platform's veld config.toml
 *   3. empty string (the server will answer 401, reported as a memory-system error)
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
  // PreToolUse / PostToolUse / PostToolUseFailure
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_response?: unknown;
  error?: string;
  is_interrupt?: boolean;
  // SubagentStop
  agent_id?: string;
  agent_type?: string;
  agent_transcript_path?: string;
  last_assistant_message?: string;
  // Legacy field names (backward compat)
  tool_output?: unknown;
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

function envMs(name: string, fallback: number): number {
  const raw = process.env[name];
  const n = raw ? Number.parseInt(raw, 10) : Number.NaN;
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/** Per-request budget on tool events. Kept under the 5 s hook `timeout` in the settings files. */
const TOOL_TIMEOUT_MS = envMs("VELD_HOOK_TIMEOUT_MS", 4000);

/** Budget for SessionStart and UserPromptSubmit, where context matters most. Kept under their 10 s hook `timeout`. */
const PROMPT_TIMEOUT_MS = envMs("VELD_HOOK_PROMPT_TIMEOUT_MS", 8000);

/** Regex for detecting secrets that must not be stored in memory */
const SECRET_PATTERN = /(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36,}|AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._\-]{20,}|password\s*[=:]\s*\S+)/i;

function containsSecrets(text: string): boolean {
  return SECRET_PATTERN.test(text);
}

const MAX_IMPLICIT_BOOSTS = 3;

/** Claude Code renamed the subagent tool from `Task` to `Agent`; older clients still send `Task`. */
const SUBAGENT_TOOLS = new Set(["Agent", "Task"]);
const SHELL_TOOLS = new Set(["Bash", "PowerShell"]);

export function isSubagentTool(toolName: string): boolean {
  return SUBAGENT_TOOLS.has(toolName);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// =============================================================================
// SESSION STATE (persisted per session_id across hook processes)
// =============================================================================

interface ToolActionRecord {
  tool_name: string;
  inputs: Record<string, string>;
  success: boolean;
  output_snippet?: string;
}

export interface SessionState {
  /** Episode ID for this session — enables temporal clustering & ordinal boost */
  episodeId: string;
  /** Monotonic sequence counter for memory ordering within the episode */
  sequence: number;
  /** ID of the last memory stored (for preceding_memory_id chains) */
  lastStoredMemoryId: string | null;
  /** IDs and formatted output of the most recent surfacing, sent back as implicit feedback */
  lastSurfacedMemoryIds: string[];
  lastProactiveOutput: string | null;
  /** Implicit reinforcements sent per memory ID, capped at MAX_IMPLICIT_BOOSTS */
  boostCounts: Record<string, number>;
  /** Tool actions since the last proactive_context call, for feedback attribution */
  pendingToolActions: ToolActionRecord[];
  /** Last result per debounce key */
  debounce: Record<string, { at: number; result: string | null }>;
  /** Agent-session marker written at SessionStart; SessionEnd runs in another process */
  markerPath: string | null;
  /** Repository slug sent as `repo` on every call; null when the working directory is not a repo */
  repo?: string | null;
}

// =============================================================================
// REPOSITORY SCOPE
// =============================================================================

/** "Owner/Repo" from an https, ssh or scp-style remote URL; null when it has fewer than two path parts. */
export function repoSlugFromRemote(url: string): string | null {
  const path = url
    .trim()
    .replace(/^[a-z+]+:\/\/[^/]+\//i, "")
    .replace(/^[^@/:]+@[^:/]+:/, "")
    .replace(/\.git\/?$/, "")
    .replace(/\/+$/, "");
  const parts = path.split("/").filter(Boolean);
  return parts.length >= 2 ? parts.slice(-2).join("/") : null;
}

function git(dir: string, args: string[]): string | null {
  try {
    return execFileSync("git", ["-C", dir, ...args], {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 1500,
    }).trim() || null;
  } catch {
    return null;
  }
}

/**
 * The repository a session works in, the same for every worktree of it: VELD_REPO if set, else
 * the origin remote's "Owner/Repo", else the main checkout's directory name.
 */
export function detectRepo(dir: string): string | null {
  const declared = process.env.VELD_REPO?.trim();
  if (declared) return declared;
  const remote = git(dir, ["remote", "get-url", "origin"]);
  const fromRemote = remote ? repoSlugFromRemote(remote) : null;
  if (fromRemote) return fromRemote;
  const commonDir = git(dir, ["rev-parse", "--git-common-dir"]);
  if (!commonDir) return null;
  const absolute = isAbsolute(commonDir) ? commonDir : join(dir, commonDir);
  return basename(dirname(absolute)) || null;
}

const STATE_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const MAX_PENDING_TOOL_ACTIONS = 20;
const MAX_BOOST_ENTRIES = 500;

export function freshSessionState(): SessionState {
  return {
    episodeId: `ep-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    sequence: 0,
    lastStoredMemoryId: null,
    lastSurfacedMemoryIds: [],
    lastProactiveOutput: null,
    boostCounts: {},
    pendingToolActions: [],
    debounce: {},
    markerPath: null,
  };
}

export function hookStateDir(): string {
  return process.env.VELD_HOOK_STATE_DIR || join(tmpdir(), "veld-hook");
}

export function sessionStatePath(sessionId: string, dir: string = hookStateDir()): string {
  const safe = sessionId.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 128);
  return join(dir, `${safe}.json`);
}

function isSessionState(value: unknown): value is SessionState {
  if (!isRecord(value)) return false;
  return (
    typeof value.episodeId === "string" &&
    typeof value.sequence === "number" &&
    (value.lastStoredMemoryId === null || typeof value.lastStoredMemoryId === "string") &&
    Array.isArray(value.lastSurfacedMemoryIds) &&
    (value.lastProactiveOutput === null || typeof value.lastProactiveOutput === "string") &&
    isRecord(value.boostCounts) &&
    Array.isArray(value.pendingToolActions) &&
    isRecord(value.debounce) &&
    (value.markerPath === null || typeof value.markerPath === "string") &&
    (value.repo === undefined || value.repo === null || typeof value.repo === "string")
  );
}

/** Load the session's state. Any read or parse failure yields fresh state: the hook must not fail on it. */
export function loadSessionState(sessionId: string | undefined, dir: string = hookStateDir()): SessionState {
  if (!sessionId) return freshSessionState();
  try {
    const parsed = JSON.parse(readFileSync(sessionStatePath(sessionId, dir), "utf-8"));
    if (isSessionState(parsed)) return parsed;
  } catch {
    // missing, unreadable or corrupt — start fresh
  }
  return freshSessionState();
}

function compactSessionState(state: SessionState, now: number): void {
  for (const [key, entry] of Object.entries(state.debounce)) {
    if (now - entry.at >= PROACTIVE_DEBOUNCE_MS) delete state.debounce[key];
  }
  if (state.pendingToolActions.length > MAX_PENDING_TOOL_ACTIONS) {
    state.pendingToolActions.splice(0, state.pendingToolActions.length - MAX_PENDING_TOOL_ACTIONS);
  }
  const boostKeys = Object.keys(state.boostCounts);
  for (const key of boostKeys.slice(0, Math.max(0, boostKeys.length - MAX_BOOST_ENTRIES))) {
    delete state.boostCounts[key];
  }
}

/** Write state atomically (tmp + rename). Errors are logged and swallowed. */
export function saveSessionState(sessionId: string | undefined, state: SessionState, dir: string = hookStateDir()): void {
  if (!sessionId) return;
  compactSessionState(state, Date.now());
  const path = sessionStatePath(sessionId, dir);
  // Per-pid tmp name: parallel hooks for one session must not interleave writes into one tmp file.
  const tmp = `${path}.${process.pid}.tmp`;
  try {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    writeFileSync(tmp, JSON.stringify(state), { encoding: "utf-8", mode: 0o600 });
    renameSync(tmp, path);
  } catch (e) {
    console.error(`[veld] session state write failed at ${path}:`, e);
    try { unlinkSync(tmp); } catch { /* tmp may not exist */ }
  }
}

export function removeSessionState(sessionId: string | undefined, dir: string = hookStateDir()): void {
  if (!sessionId) return;
  try { unlinkSync(sessionStatePath(sessionId, dir)); } catch { /* already gone */ }
}

/** Delete state files (and stranded tmp files) untouched for more than a day. */
export function pruneSessionStates(dir: string = hookStateDir(), now: number = Date.now()): void {
  let names: string[];
  try {
    names = readdirSync(dir);
  } catch {
    return;
  }
  for (const name of names) {
    if (!name.endsWith(".json") && !name.endsWith(".tmp")) continue;
    const path = join(dir, name);
    try {
      if (now - statSync(path).mtimeMs > STATE_MAX_AGE_MS) unlinkSync(path);
    } catch {
      // another hook removed or replaced it first
    }
  }
}

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

// =============================================================================
// HTTP
// =============================================================================

/**
 * `timeout`: no answer within the budget — the server may be busy, not down.
 * `unreachable`: the request never got a response (connection refused, DNS, reset).
 * `error`: the server answered with a non-2xx status or an unreadable body.
 */
export type BrainOutcome = "ok" | "timeout" | "unreachable" | "error";

interface BrainResult {
  outcome: BrainOutcome;
  status?: number;
  body: unknown;
}

async function requestBrain(endpoint: string, body: Record<string, unknown>, timeoutMs: number): Promise<BrainResult> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  let status: number | undefined;
  try {
    const response = await fetch(`${VELD_API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": VELD_API_KEY,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    status = response.status;
    if (!response.ok) return { outcome: "error", status, body: null };
    return { outcome: "ok", status, body: await response.json() };
  } catch {
    const outcome: BrainOutcome = timedOut ? "timeout" : status === undefined ? "unreachable" : "error";
    return { outcome, status, body: null };
  } finally {
    clearTimeout(timer);
  }
}

async function callBrain(endpoint: string, body: Record<string, unknown>): Promise<unknown> {
  return (await requestBrain(endpoint, body, TOOL_TIMEOUT_MS)).body;
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

function blockText(block: unknown): string {
  if (typeof block === "string") return block;
  if (isRecord(block) && typeof block.text === "string") return block.text;
  return "";
}

/**
 * Flatten a PostToolUse `tool_response` to text. Its shape depends on the tool: a string,
 * an array of content blocks, `{ content: [...] }` for Agent, `{ stdout, stderr, ... }`
 * for Bash, or another structured object.
 */
export function toolResponseText(response: unknown): string {
  if (response == null) return "";
  if (typeof response === "string") return response;
  if (Array.isArray(response)) return response.map(blockText).filter(Boolean).join("\n");
  if (isRecord(response)) {
    if (Array.isArray(response.content)) return toolResponseText(response.content);
    const streams = [response.stdout, response.stderr].filter(
      (s): s is string => typeof s === "string" && s.length > 0
    );
    if (streams.length) return streams.join("\n");
  }
  try {
    return JSON.stringify(response) ?? "";
  } catch {
    return "";
  }
}

/**
 * Error text from a shell tool's PostToolUse `tool_response`, or null. On the object shape
 * only stderr counts: stdout of a successful command (grep, logs) mentions "error" too often.
 */
export function shellErrorText(response: unknown): string | null {
  if (typeof response === "string") return isErrorOutput(response) ? response : null;
  if (isRecord(response) && typeof response.stderr === "string" && isErrorOutput(response.stderr)) {
    return response.stderr;
  }
  return null;
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
  } else if (SHELL_TOOLS.has(toolName)) {
    const command = toolInput.command as string;
    if (command) {
      return `Running command: ${command.slice(0, 100)}`;
    }
  } else if (toolName === "Read") {
    const filePath = toolInput.file_path as string;
    if (filePath) {
      return `Reading file: ${filePath}`;
    }
  } else if (isSubagentTool(toolName)) {
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
 *  Reduces server-side embedding load when the same question repeats in quick
 *  succession. Tunable via VELD_HOOK_DEBOUNCE_MS (0 disables). */
const PROACTIVE_DEBOUNCE_MS = (() => {
  const raw = process.env.VELD_HOOK_DEBOUNCE_MS;
  const n = raw ? Number.parseInt(raw, 10) : 500;
  return Number.isFinite(n) && n >= 0 ? n : 500;
})();

function proactiveDebounceKey(context: string, maxResults: number): string {
  // 80-char prefix lets "Editing file: a.rs" and "Editing file: b.rs" debounce
  // independently, while two Edits of the same file within the window coalesce.
  return `${maxResults}|${context.slice(0, 80)}`;
}

interface SurfaceResult {
  text: string | null;
  outcome: BrainOutcome;
  status?: number;
}

interface SurfaceOptions {
  autoIngest?: boolean;
  timeoutMs?: number;
}

async function surfaceProactiveContext(
  state: SessionState,
  context: string,
  maxResults: number,
  { autoIngest = false, timeoutMs = TOOL_TIMEOUT_MS }: SurfaceOptions = {}
): Promise<SurfaceResult> {
  // UserPromptSubmit relies on autoIngest to record the prompt; a debounced answer would drop it.
  const debounceKey = proactiveDebounceKey(context, maxResults);
  if (PROACTIVE_DEBOUNCE_MS > 0 && !autoIngest) {
    const cached = state.debounce[debounceKey];
    if (cached && Date.now() - cached.at < PROACTIVE_DEBOUNCE_MS) {
      return { text: cached.result, outcome: "ok" };
    }
  }

  const toolActions = state.pendingToolActions.splice(0, state.pendingToolActions.length);

  const feedbackPayload: Record<string, unknown> = {};
  if (state.lastProactiveOutput) {
    feedbackPayload.previous_response = state.lastProactiveOutput;
    feedbackPayload.user_followup = context.slice(0, 1000);
  }
  const eligibleIds = state.lastSurfacedMemoryIds.filter(
    (id) => (state.boostCounts[id] ?? 0) < MAX_IMPLICIT_BOOSTS
  );
  if (eligibleIds.length > 0) {
    feedbackPayload.surfaced_memory_ids = eligibleIds;
    for (const id of eligibleIds) {
      state.boostCounts[id] = (state.boostCounts[id] ?? 0) + 1;
    }
  }

  if (autoIngest) state.sequence += 1;

  const result = await requestBrain(
    "/api/proactive_context",
    {
      user_id: VELD_USER_ID,
      context,
      max_results: maxResults,
      semantic_threshold: 0.6,
      entity_match_weight: 0.3,
      recency_weight: 0.2,
      auto_ingest: autoIngest,
      // FIX-05: Propagate episode + emotional context to auto-ingested memories
      episode_id: state.episodeId,
      sequence_number: state.sequence,
      emotional_valence: 0.0,
      emotional_arousal: 0.15,
      ...(state.repo ? { repo: state.repo } : {}),
      ...(toolActions.length > 0 ? { tool_actions: toolActions } : {}),
      ...feedbackPayload,
    },
    timeoutMs
  );

  if (result.outcome !== "ok") {
    return { text: null, outcome: result.outcome, status: result.status };
  }

  const response = (result.body ?? {}) as Partial<ProactiveContextResponse>;
  const memories = response.memories ?? [];
  const facts = response.relevant_facts ?? [];
  const todos = response.relevant_todos ?? [];
  const reminderCount = (response.due_reminders?.length || 0) + (response.context_reminders?.length || 0);

  let output: string | null = null;
  if (memories.length || facts.length || todos.length || reminderCount) {
    const now = new Date();
    output = `📅 ${now.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })} ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    if (memories.length) {
      output += `\n${formatMemoriesForContext(memories)}`;
    }
    if (facts.length) {
      output += `\n🧠 Facts:`;
      for (const f of facts.slice(0, 3)) {
        output += `\n• (${Math.round(f.confidence * 100)}%) ${f.fact}`;
      }
    }
    if (todos.length) {
      output += `\n📋 Todos:`;
      for (const t of todos.slice(0, 3)) {
        const icon = t.status === "in_progress" ? "🔄" : "☐";
        output += `\n${icon} ${t.content.slice(0, 80)}`;
      }
    }
    if (reminderCount) {
      output += `\n⏰ ${reminderCount} reminder(s) active`;
    }
  }

  state.lastSurfacedMemoryIds = memories.map((m) => m.id);
  state.lastProactiveOutput = output;

  if (PROACTIVE_DEBOUNCE_MS > 0 && !autoIngest) {
    state.debounce[debounceKey] = { at: Date.now(), result: output };
  }

  return { text: output, outcome: "ok" };
}

/** One-line notice for a server that refused or failed. A timeout gets none: slow is not down. */
function memoryOfflineNotice(result: SurfaceResult): string | null {
  if (result.outcome === "unreachable") {
    return `Memory system unreachable at ${VELD_API_URL}. No persistent context available.`;
  }
  if (result.outcome === "error") {
    return `Memory system returned ${result.status ? `HTTP ${result.status}` : "an unreadable response"}. No persistent context available.`;
  }
  return null;
}

function emitContext(hookEventName: string, additionalContext: string): void {
  console.log(JSON.stringify({ hookSpecificOutput: { hookEventName, additionalContext } }));
}

async function rememberInEpisode(
  state: SessionState,
  memory: { content: string; memory_type: string; tags: string[]; credibility: number },
  emotion: EmotionalSignal
): Promise<void> {
  state.sequence += 1;
  const resp = (await callBrain("/api/remember", {
    user_id: VELD_USER_ID,
    ...memory,
    source_type: "system",
    episode_id: state.episodeId,
    sequence_number: state.sequence,
    ...(state.lastStoredMemoryId ? { preceding_memory_id: state.lastStoredMemoryId } : {}),
    ...(state.repo ? { repo: state.repo } : {}),
    ...emotion,
  })) as { id?: string } | null;
  if (resp?.id) state.lastStoredMemoryId = resp.id;
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
// on every SessionStart). Removed on SessionEnd best-effort, via the path kept
// in session state: SessionEnd runs in a different process with a different pid.

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

// =============================================================================
// EVENT HANDLERS
// =============================================================================

async function handleSessionStart(state: SessionState): Promise<void> {
  pruneSessionStates();

  // Drop the agent-session marker file first so downstream chat-brand
  // detection works even if the memory backend is unreachable below.
  const markerPath = agentSessionMarkerPath();
  if (writeAgentSessionMarker(markerPath)) {
    if (state.markerPath && state.markerPath !== markerPath) removeAgentSessionMarker(state.markerPath);
    state.markerPath = markerPath;
  }

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const projectName = projectDir.split(/[/\\]/).pop() || "unknown";

  const result = await surfaceProactiveContext(state, `Starting session in project: ${projectName}`, 5, {
    timeoutMs: PROMPT_TIMEOUT_MS,
  });

  if (result.text) {
    emitContext("SessionStart", `\n<veld>\n${result.text}\n</veld>`);
    return;
  }
  const notice = memoryOfflineNotice(result);
  if (notice) {
    console.error(`[veld] ${notice}`);
    emitContext("SessionStart", `\n<veld-warning>${notice}</veld-warning>`);
  }
}

async function handleUserPrompt(input: HookInput, state: SessionState): Promise<void> {
  const prompt = input.prompt;
  if (!prompt || prompt.length < 10) return;

  // Single call: surface memories AND ingest the prompt in one pipeline pass
  const result = await surfaceProactiveContext(state, prompt.slice(0, 1000), 3, {
    autoIngest: true,
    timeoutMs: PROMPT_TIMEOUT_MS,
  });

  if (result.text) {
    emitContext("UserPromptSubmit", `\n<veld>\n${result.text}\n</veld>`);
    return;
  }
  const notice = memoryOfflineNotice(result);
  if (notice) {
    emitContext("UserPromptSubmit", `\n<veld-warning>${notice}</veld-warning>`);
  }
}

const ORCH_TAG_RE = /\[ORCH-TODO:([A-Z]+-\d+)\]/;

function orchestrationTodoId(toolInput: Record<string, unknown>): string | null {
  const prompt = toolInput.prompt;
  if (typeof prompt !== "string") return null;
  return prompt.match(ORCH_TAG_RE)?.[1] ?? null;
}

/**
 * Surfacing for tool use happens here, before the tool runs, and nowhere else — with one
 * exception: an orchestration subagent surfaces after it completes, when unblocked work exists.
 */
async function handlePreToolUse(input: HookInput, state: SessionState): Promise<void> {
  const toolName = input.tool_name;
  const toolInput = input.tool_input;
  if (!toolName || !toolInput) return;
  if (isSubagentTool(toolName) && orchestrationTodoId(toolInput)) return;

  const result = await surfaceProactiveContext(state, buildPreToolContext(toolName, toolInput), 2);
  if (result.text) {
    emitContext("PreToolUse", `\n<veld context="pre-${toolName.toLowerCase()}">\n${result.text}\n</veld>`);
  }
}

function recordToolAction(
  state: SessionState,
  toolName: string,
  toolInput: Record<string, unknown>,
  success: boolean,
  output: string
): void {
  const inputs: Record<string, string> = {};
  for (const [k, v] of Object.entries(toolInput)) {
    if (typeof v === "string" && !containsSecrets(v)) {
      inputs[k] = v.slice(0, 500);
    }
  }
  const snippet = output.slice(0, 200);
  state.pendingToolActions.push({
    tool_name: toolName,
    inputs,
    success,
    ...(snippet && !containsSecrets(snippet) ? { output_snippet: snippet } : {}),
  });
}

async function rememberCommandFailure(
  state: SessionState,
  toolName: string,
  toolInput: Record<string, unknown>,
  errorText: string
): Promise<void> {
  const command = typeof toolInput.command === "string" ? toolInput.command : "";
  const content = `Command failed: ${command.slice(0, 100)} → ${errorText.slice(0, 200)}`;
  if (containsSecrets(content)) return;
  await rememberInEpisode(
    state,
    { content, memory_type: "Error", tags: [`tool:${toolName}`, "error"], credibility: 0.9 },
    classifyEmotion("Error", false, errorText)
  );
}

/**
 * PostToolUse fires only for tools that succeeded. It records the action for feedback
 * attribution and stores nothing per edit or read: those echoes were most of the store's noise.
 */
async function handlePostToolUse(input: HookInput, state: SessionState): Promise<void> {
  const toolName = input.tool_name;
  if (!toolName) return;
  const toolInput = input.tool_input ?? {};
  const response = input.tool_response ?? input.tool_output;

  if (isSubagentTool(toolName)) {
    await handleSubagentToolResult(toolInput, response, state);
    return;
  }

  const errorText = SHELL_TOOLS.has(toolName) ? shellErrorText(response) : null;
  recordToolAction(state, toolName, toolInput, errorText === null, errorText ?? toolResponseText(response));
  if (errorText) {
    await rememberCommandFailure(state, toolName, toolInput, errorText);
  }
}

async function handlePostToolUseFailure(input: HookInput, state: SessionState): Promise<void> {
  const toolName = input.tool_name;
  if (!toolName || isSubagentTool(toolName)) return;
  const toolInput = input.tool_input ?? {};
  const error = typeof input.error === "string" ? input.error : "";

  recordToolAction(state, toolName, toolInput, false, error);
  if (error && !input.is_interrupt && SHELL_TOOLS.has(toolName)) {
    await rememberCommandFailure(state, toolName, toolInput, error);
  }
}

// --- Orchestration: PostToolUse(Agent) handler ---

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

/**
 * Only orchestration-tagged subagents are handled here. Plain subagent completions are
 * stored once, at SubagentStop, which also covers background subagents.
 */
async function handleSubagentToolResult(
  toolInput: Record<string, unknown>,
  response: unknown,
  state: SessionState
): Promise<void> {
  const todoShortId = orchestrationTodoId(toolInput);
  if (!todoShortId) return;
  // A background launch returns at once; the subagent has not done the work yet.
  if (isRecord(response) && response.status === "async_launched") return;

  const resultText = toolResponseText(response);

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
  const content = `Orchestration task ${todoShortId} completed: ${resultText.slice(0, 200)}`;
  if (!containsSecrets(content)) {
    await rememberInEpisode(
      state,
      { content, memory_type: "Task", tags: ["orchestration", `todo:${todoShortId}`, "source:hook"], credibility: 0.75 },
      classifyEmotion("Task", true, resultText)
    );
  }

  // 5. Surface orchestration status
  const result = await surfaceProactiveContext(
    state,
    `Orchestration: task ${todoShortId} completed, checking for unblocked work`,
    2
  );
  if (result.text) {
    emitContext("PostToolUse", `\n<veld context="orchestration">\n${result.text}\n</veld>`);
  }
}

async function handleSubagentStop(input: HookInput, state: SessionState): Promise<void> {
  const agentType = input.agent_type || input.subagent_type;
  const result = input.last_assistant_message || input.subagent_result;
  if (!agentType || !result) return;

  const content = `${agentType} agent completed: ${result.slice(0, 300)}`;
  if (containsSecrets(content)) return;
  await rememberInEpisode(
    state,
    { content, memory_type: "Task", tags: [`subagent:${agentType}`, "source:hook"], credibility: 0.6 },
    classifyEmotion("Task", true, content)
  );
}

async function handleStop(_input: HookInput): Promise<void> {
  // Session end is tracked implicitly by memory timestamps and decay.
  // Storing explicit "Session ended" memories creates noise in the activity log
  // and gets re-ingested by proactive_context auto-ingest, causing duplicate events.
}

function handleSessionEnd(input: HookInput): void {
  const state = loadSessionState(input.session_id);
  removeAgentSessionMarker(state.markerPath ?? agentSessionMarkerPath());
  removeSessionState(input.session_id);
}

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();

  let input: HookInput;
  try {
    input = JSON.parse(inputText);
  } catch {
    input = { hook_event_name: process.argv[2] || "SessionStart" };
  }

  const eventName = input.hook_event_name || process.argv[2];
  if (eventName === "SessionEnd") {
    handleSessionEnd(input);
    return;
  }

  const state = loadSessionState(input.session_id);
  if (state.repo === undefined) {
    state.repo = detectRepo(input.cwd || process.env.CLAUDE_PROJECT_DIR || process.cwd());
  }

  switch (eventName) {
    case "SessionStart":
      await handleSessionStart(state);
      break;
    case "UserPromptSubmit":
      await handleUserPrompt(input, state);
      break;
    case "PreToolUse":
      await handlePreToolUse(input, state);
      break;
    case "PostToolUse":
      await handlePostToolUse(input, state);
      break;
    case "PostToolUseFailure":
      await handlePostToolUseFailure(input, state);
      break;
    case "SubagentStop":
      await handleSubagentStop(input, state);
      break;
    case "Stop":
      await handleStop(input);
      return;
    default:
      return;
  }

  saveSessionState(input.session_id, state);
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    // Log but do not rethrow — hooks must not crash Claude Code
    console.error("[veld] Hook error:", e);
  }
}
