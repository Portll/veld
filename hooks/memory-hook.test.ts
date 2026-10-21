import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "bun:test";
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  agentSessionMarkerPath,
  buildAgentSessionMarker,
  buildPreToolContext,
  detectBinary,
  formatRelativeTime,
  formatMemoriesForContext,
  freshSessionState,
  isErrorOutput,
  isSubagentTool,
  loadSessionState,
  pruneSessionStates,
  removeAgentSessionMarker,
  saveSessionState,
  sessionStatePath,
  shellErrorText,
  toolResponseText,
  writeAgentSessionMarker,
} from "./memory-hook";

describe("formatRelativeTime", () => {
  it("returns today for current date", () => {
    const nowIso = new Date().toISOString();
    expect(formatRelativeTime(nowIso)).toBe("today");
  });

  it("returns yesterday for one day old date", () => {
    const d = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(d)).toBe("yesterday");
  });

  it("returns Xd ago for dates under a week", () => {
    const d = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(d)).toBe("3d ago");
  });

  it("returns calendar date for older memories", () => {
    const d = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString();
    const value = formatRelativeTime(d);
    expect(value).not.toContain("ago");
    expect(value).not.toBe("today");
    expect(value).not.toBe("yesterday");
  });
});

describe("formatMemoriesForContext", () => {
  it("returns empty string for empty input", () => {
    expect(formatMemoriesForContext([])).toBe("");
  });

  it("formats score, relative time, and content", () => {
    const memories = [
      {
        id: "m1",
        content: "Remember to review deployment checklist before release",
        memory_type: "Task",
        score: 0.83,
        importance: 0.7,
        created_at: new Date().toISOString(),
        tags: ["deploy"],
        relevance_reason: "matches query",
        matched_entities: ["release"],
      },
    ];

    const out = formatMemoriesForContext(memories);
    expect(out).toContain("[83%]");
    expect(out).toContain("(today)");
    expect(out).toContain("Remember to review deployment checklist");
  });

  it("truncates long memory content at 120 chars with ellipsis", () => {
    const longContent = "x".repeat(130);
    const memories = [
      {
        id: "m2",
        content: longContent,
        memory_type: "Observation",
        score: 0.5,
        importance: 0.5,
        created_at: new Date().toISOString(),
        tags: [],
        relevance_reason: "test",
        matched_entities: [],
      },
    ];

    const out = formatMemoriesForContext(memories);
    expect(out).toContain("...");
    expect(out.length).toBeGreaterThan(120);
  });

  it("formats multiple memories as multiple bullet lines", () => {
    const now = new Date().toISOString();
    const out = formatMemoriesForContext([
      {
        id: "m1",
        content: "A",
        memory_type: "Observation",
        score: 0.2,
        importance: 0.1,
        created_at: now,
        tags: [],
        relevance_reason: "x",
        matched_entities: [],
      },
      {
        id: "m2",
        content: "B",
        memory_type: "Observation",
        score: 0.3,
        importance: 0.1,
        created_at: now,
        tags: [],
        relevance_reason: "x",
        matched_entities: [],
      },
    ]);
    expect((out.match(/•/g) || []).length).toBe(2);
  });
});

describe("buildPreToolContext", () => {
  it("builds edit context", () => {
    expect(buildPreToolContext("Edit", { file_path: "src/main.ts" })).toBe("Editing file: src/main.ts");
  });

  it("builds bash context and truncates command", () => {
    const cmd = "x".repeat(140);
    const value = buildPreToolContext("Bash", { command: cmd });
    expect(value.startsWith("Running command: ")).toBe(true);
    expect(value.length).toBeLessThanOrEqual("Running command: ".length + 100);
  });

  it("falls back to generic context", () => {
    expect(buildPreToolContext("Read", {})).toBe("About to use Read");
  });

  it("treats Agent and the legacy Task name as the subagent tool", () => {
    expect(buildPreToolContext("Agent", { prompt: "Find all API endpoints" })).toBe("Spawning agent: Find all API endpoints");
    expect(buildPreToolContext("Task", { prompt: "Find all API endpoints" })).toBe("Spawning agent: Find all API endpoints");
  });

  it("treats PowerShell like Bash", () => {
    expect(buildPreToolContext("PowerShell", { command: "Get-ChildItem" })).toBe("Running command: Get-ChildItem");
  });
});

describe("isSubagentTool", () => {
  it("accepts Agent and Task only", () => {
    expect(isSubagentTool("Agent")).toBe(true);
    expect(isSubagentTool("Task")).toBe(true);
    expect(isSubagentTool("Bash")).toBe(false);
    expect(isSubagentTool("TaskCreate")).toBe(false);
  });
});

describe("toolResponseText", () => {
  it("passes strings through and flattens the documented object shapes", () => {
    expect(toolResponseText("plain")).toBe("plain");
    expect(toolResponseText({ stdout: "out", stderr: "err", interrupted: false, isImage: false })).toBe("out\nerr");
    expect(
      toolResponseText({ status: "completed", agentId: "a1", content: [{ type: "text", text: "Found 12 endpoints" }] })
    ).toBe("Found 12 endpoints");
    expect(toolResponseText([{ type: "text", text: "a" }, { type: "text", text: "b" }])).toBe("a\nb");
    expect(toolResponseText({ filePath: "/x.ts", type: "create" })).toBe('{"filePath":"/x.ts","type":"create"}');
    expect(toolResponseText(undefined)).toBe("");
  });
});

describe("shellErrorText", () => {
  it("reads stderr from the Bash object shape", () => {
    expect(shellErrorText({ stdout: "", stderr: "error: linker failed", interrupted: false })).toBe("error: linker failed");
  });

  it("ignores error words on stdout of a command that succeeded", () => {
    expect(shellErrorText({ stdout: "3 errors in log.txt", stderr: "", interrupted: false })).toBeNull();
  });

  it("still accepts a string response", () => {
    expect(shellErrorText("Error: boom")).toBe("Error: boom");
    expect(shellErrorText("all good")).toBeNull();
  });
});

describe("session state file", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "veld-hook-state-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("round-trips through an atomic write with no tmp file left behind", () => {
    const state = freshSessionState();
    state.sequence = 7;
    state.lastStoredMemoryId = "mem-7";
    saveSessionState("sess-1", state, dir);

    expect(readdirSync(dir)).toEqual(["sess-1.json"]);
    expect(statSync(sessionStatePath("sess-1", dir)).mode & 0o777).toBe(0o600);
    const loaded = loadSessionState("sess-1", dir);
    expect(loaded.episodeId).toBe(state.episodeId);
    expect(loaded.sequence).toBe(7);
    expect(loaded.lastStoredMemoryId).toBe("mem-7");
  });

  it("keeps the session id from escaping the state directory", () => {
    const path = sessionStatePath("../../etc/passwd", dir);
    expect(path.startsWith(dir)).toBe(true);
    expect(path.slice(dir.length + 1).includes("/")).toBe(false);
  });

  it("starts fresh on a corrupt or wrongly shaped file", () => {
    writeFileSync(sessionStatePath("bad", dir), "{not json");
    expect(loadSessionState("bad", dir).sequence).toBe(0);
    writeFileSync(sessionStatePath("odd", dir), JSON.stringify({ episodeId: 5, sequence: "x" }));
    expect(typeof loadSessionState("odd", dir).episodeId).toBe("string");
  });

  it("does not throw when the state directory cannot be written", () => {
    const blocker = join(dir, "not-a-dir");
    writeFileSync(blocker, "");
    expect(() => saveSessionState("sess", freshSessionState(), blocker)).not.toThrow();
  });

  it("does not persist without a session id", () => {
    saveSessionState(undefined, freshSessionState(), dir);
    expect(readdirSync(dir)).toEqual([]);
  });

  it("prunes files untouched for more than a day", () => {
    saveSessionState("old", freshSessionState(), dir);
    saveSessionState("new", freshSessionState(), dir);
    const twoDaysAgo = (Date.now() - 2 * 24 * 60 * 60 * 1000) / 1000;
    utimesSync(sessionStatePath("old", dir), twoDaysAgo, twoDaysAgo);
    pruneSessionStates(dir);
    expect(readdirSync(dir)).toEqual(["new.json"]);
  });
});

describe("hook settings files", () => {
  const files = [
    join(import.meta.dir, "claude-settings.json"),
    join(import.meta.dir, "..", ".claude", "settings.json.example"),
  ];

  for (const file of files) {
    it(`${file.split("/").slice(-2).join("/")} uses string matchers and timeouts in seconds`, () => {
      const settings = JSON.parse(readFileSync(file, "utf-8"));
      const groups = Object.values(settings.hooks as Record<string, Array<{ matcher?: unknown; hooks: Array<{ timeout: number }> }>>).flat();
      for (const group of groups) {
        if (group.matcher !== undefined) expect(typeof group.matcher).toBe("string");
        for (const hook of group.hooks) {
          expect(hook.timeout).toBeGreaterThan(0);
          expect(hook.timeout).toBeLessThanOrEqual(60);
        }
      }
      expect(settings.hooks.PreToolUse[0].matcher.split("|")).toContain("Agent");
      expect(settings.hooks.PostToolUse.map((g: { matcher: string }) => g.matcher).join("|").split("|")).toContain("Agent");
      expect(settings.hooks.PostToolUseFailure[0].matcher.split("|")).toContain("Bash");
    });
  }
});

describe("hook process against a mock veld server", () => {
  const HOOK_PATH = join(import.meta.dir, "memory-hook.ts");
  const SURFACED = {
    id: "mem-surfaced",
    content: "Past note about cargo builds",
    memory_type: "Learning",
    score: 0.8,
    importance: 0.6,
    created_at: new Date().toISOString(),
    tags: [],
    relevance_reason: "match",
    matched_entities: [],
  };

  type Responder = (path: string) => Response | Promise<Response>;
  let server: ReturnType<typeof Bun.serve>;
  let requests: Array<{ path: string; body: Record<string, any> }> = [];
  let rememberCount = 0;
  let responder: Responder;
  let workDir: string;

  const defaultResponder: Responder = (path) => {
    if (path === "/api/proactive_context") {
      return Response.json({ memories: [SURFACED], due_reminders: [], context_reminders: [], relevant_todos: [], relevant_facts: [] });
    }
    if (path === "/api/remember") {
      rememberCount += 1;
      return Response.json({ id: `mem-${rememberCount}`, success: true });
    }
    return Response.json({});
  };

  beforeAll(() => {
    server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        const body = (await req.json().catch(() => ({}))) as Record<string, any>;
        requests.push({ path, body });
        return responder(path);
      },
    });
  });

  afterAll(() => {
    server.stop(true);
  });

  beforeEach(() => {
    requests = [];
    rememberCount = 0;
    responder = defaultResponder;
    workDir = mkdtempSync(join(tmpdir(), "veld-hook-run-"));
  });

  afterEach(() => {
    rmSync(workDir, { recursive: true, force: true });
  });

  async function runHook(payload: Record<string, unknown>, env: Record<string, string> = {}) {
    const proc = Bun.spawn([process.execPath, "run", HOOK_PATH, String(payload.hook_event_name)], {
      cwd: workDir,
      env: {
        ...process.env,
        VELD_API_URL: `http://127.0.0.1:${server.port}`,
        VELD_API_KEY: "test-key",
        VELD_HOOK_STATE_DIR: join(workDir, "state"),
        VELD_HOOK_DEBOUNCE_MS: "0",
        CLAUDE_PROJECT_DIR: workDir,
        XDG_CONFIG_HOME: workDir,
        ...env,
      },
      stdin: new Blob([JSON.stringify(payload)]),
      stdout: "pipe",
      stderr: "pipe",
    });
    const [stdout, stderr] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text()]);
    await proc.exited;
    return { stdout: stdout.trim(), stderr };
  }

  const to = (path: string) => requests.filter((r) => r.path === path);

  it("stores an error from stderr in a PostToolUse Bash tool_response, without surfacing", async () => {
    const { stdout } = await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-bash",
      tool_name: "Bash",
      tool_input: { command: "cargo build" },
      tool_response: { stdout: "", stderr: "error[E0425]: cannot find value `x`", interrupted: false, isImage: false },
    });
    const stored = to("/api/remember");
    expect(stored.length).toBe(1);
    expect(stored[0].body.memory_type).toBe("Error");
    expect(stored[0].body.content).toContain("Command failed: cargo build");
    expect(stored[0].body.content).toContain("E0425");
    expect(to("/api/proactive_context").length).toBe(0);
    expect(stdout).toBe("");
  });

  it("stores nothing for a PostToolUse Bash whose stdout merely mentions errors", async () => {
    await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-bash-ok",
      tool_name: "Bash",
      tool_input: { command: "grep -c error log.txt" },
      tool_response: { stdout: "3 errors", stderr: "", interrupted: false, isImage: false },
    });
    expect(to("/api/remember").length).toBe(0);
  });

  it("stores a PostToolUseFailure shell error and skips an interrupt", async () => {
    await runHook({
      hook_event_name: "PostToolUseFailure",
      session_id: "s-fail",
      tool_name: "Bash",
      tool_input: { command: "npm test" },
      error: "Exit code 1\nError: Cannot find module 'express'",
      is_interrupt: false,
    });
    await runHook({
      hook_event_name: "PostToolUseFailure",
      session_id: "s-fail",
      tool_name: "Bash",
      tool_input: { command: "sleep 100" },
      error: "Interrupted",
      is_interrupt: true,
    });
    const stored = to("/api/remember");
    expect(stored.length).toBe(1);
    expect(stored[0].body.content).toContain("Command failed: npm test → Exit code 1");
  });

  it("persists episode threading across invocations with the same session_id", async () => {
    const failure = (session_id: string) => ({
      hook_event_name: "PostToolUseFailure",
      session_id,
      tool_name: "Bash",
      tool_input: { command: "make" },
      error: "Exit code 2\nmake: *** [all] Error 1",
    });
    await runHook(failure("s-thread"));
    await runHook(failure("s-thread"));
    await runHook(failure("s-other"));

    const [first, second, other] = to("/api/remember").map((r) => r.body);
    expect(first.sequence_number).toBe(1);
    expect(second.sequence_number).toBe(2);
    expect(second.episode_id).toBe(first.episode_id);
    expect(first.preceding_memory_id).toBeUndefined();
    expect(second.preceding_memory_id).toBe("mem-1");
    expect(other.sequence_number).toBe(1);
    expect(other.episode_id).not.toBe(first.episode_id);
    expect(existsSync(join(workDir, "state", "s-thread.json"))).toBe(true);
  });

  it("surfaces once per Read: before the tool, never after", async () => {
    const read = { session_id: "s-read", tool_name: "Read", tool_input: { file_path: "/src/main.rs" } };
    const pre = await runHook({ hook_event_name: "PreToolUse", ...read });
    const post = await runHook({
      hook_event_name: "PostToolUse",
      ...read,
      tool_response: { type: "text", file: { filePath: "/src/main.rs", content: "fn main() {}" } },
    });
    expect(to("/api/proactive_context").length).toBe(1);
    expect(JSON.parse(pre.stdout).hookSpecificOutput.additionalContext).toContain("Past note about cargo builds");
    expect(post.stdout).toBe("");
    expect(to("/api/remember").length).toBe(0);
  });

  it("stores no echo for an edit and hands the action to the next surfacing as feedback", async () => {
    await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-edit",
      tool_name: "Edit",
      tool_input: { file_path: "/src/lib.rs", old_string: "a", new_string: "b" },
      tool_response: { filePath: "/src/lib.rs" },
    });
    expect(to("/api/remember").length).toBe(0);

    await runHook({ hook_event_name: "PreToolUse", session_id: "s-edit", tool_name: "Read", tool_input: { file_path: "/src/lib.rs" } });
    const [surfacing] = to("/api/proactive_context");
    expect(surfacing.body.tool_actions).toHaveLength(1);
    expect(surfacing.body.tool_actions[0].tool_name).toBe("Edit");
    expect(surfacing.body.tool_actions[0].inputs.file_path).toBe("/src/lib.rs");
    expect(surfacing.body.tool_actions[0].success).toBe(true);
  });

  it("reinforces a surfaced memory at most three times per session", async () => {
    for (let i = 0; i < 5; i++) {
      await runHook({ hook_event_name: "PreToolUse", session_id: "s-boost", tool_name: "Read", tool_input: { file_path: `/f${i}.rs` } });
    }
    const boosted = to("/api/proactive_context").filter((r) => (r.body.surfaced_memory_ids ?? []).includes("mem-surfaced"));
    expect(boosted.length).toBe(3);
    expect(to("/api/proactive_context")[1].body.previous_response).toContain("Past note about cargo builds");
  });

  it("answers a repeated question from the persisted debounce cache", async () => {
    const pre = { hook_event_name: "PreToolUse", session_id: "s-debounce", tool_name: "Read", tool_input: { file_path: "/same.rs" } };
    const first = await runHook(pre, { VELD_HOOK_DEBOUNCE_MS: "60000" });
    const second = await runHook(pre, { VELD_HOOK_DEBOUNCE_MS: "60000" });
    expect(to("/api/proactive_context").length).toBe(1);
    expect(second.stdout).toBe(first.stdout);
  });

  it("handles Agent as the subagent tool, completing an orchestration todo after the run", async () => {
    const toolInput = { prompt: "[ORCH-TODO:ORCH-7] wire the endpoint", subagent_type: "general-purpose" };
    const pre = await runHook({ hook_event_name: "PreToolUse", session_id: "s-orch", tool_name: "Agent", tool_input: toolInput });
    expect(pre.stdout).toBe("");

    const post = await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-orch",
      tool_name: "Agent",
      tool_input: toolInput,
      tool_response: { status: "completed", agentId: "a1", content: [{ type: "text", text: "Endpoint wired and tested" }] },
    });
    const comment = to("/api/todos/ORCH-7/comments");
    expect(comment.length).toBe(1);
    expect(comment[0].body.content).toBe("Endpoint wired and tested");
    expect(to("/api/todos/ORCH-7/complete").length).toBe(1);
    expect(to("/api/remember")[0].body.content).toContain("Orchestration task ORCH-7 completed: Endpoint wired and tested");
    expect(to("/api/proactive_context").length).toBe(1);
    expect(JSON.parse(post.stdout).hookSpecificOutput.hookEventName).toBe("PostToolUse");
  });

  it("does not complete an orchestration todo when the subagent was only launched in the background", async () => {
    await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-orch-bg",
      tool_name: "Agent",
      tool_input: { prompt: "[ORCH-TODO:ORCH-8] long job" },
      tool_response: { status: "async_launched", agentId: "a2", description: "long job", prompt: "[ORCH-TODO:ORCH-8] long job" },
    });
    expect(to("/api/todos/ORCH-8/complete").length).toBe(0);
  });

  it("still accepts the legacy Task tool name", async () => {
    await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-task",
      tool_name: "Task",
      tool_input: { prompt: "[ORCH-TODO:ORCH-9] legacy" },
      tool_response: "Legacy client result",
    });
    expect(to("/api/todos/ORCH-9/complete").length).toBe(1);
  });

  it("stores a plain subagent completion once, at SubagentStop", async () => {
    await runHook({
      hook_event_name: "PostToolUse",
      session_id: "s-sub",
      tool_name: "Agent",
      tool_input: { prompt: "Find all API endpoints" },
      tool_response: { status: "completed", content: [{ type: "text", text: "Found 12 endpoints" }] },
    });
    await runHook({
      hook_event_name: "SubagentStop",
      session_id: "s-sub",
      agent_id: "a3",
      agent_type: "Explore",
      last_assistant_message: "Found 12 endpoints",
    });
    const stored = to("/api/remember");
    expect(stored.length).toBe(1);
    expect(stored[0].body.content).toBe("Explore agent completed: Found 12 endpoints");
  });

  it("returns SessionStart context as additionalContext JSON, and SessionEnd cleans up", async () => {
    const start = await runHook({ hook_event_name: "SessionStart", session_id: "s-start", source: "startup" });
    const output = JSON.parse(start.stdout);
    expect(output.hookSpecificOutput.hookEventName).toBe("SessionStart");
    expect(output.hookSpecificOutput.additionalContext).toContain("Past note about cargo builds");
    expect(existsSync(join(workDir, ".claude", "memory-context.md"))).toBe(false);

    const markers = readdirSync(workDir).filter((n) => n.startsWith(".veld-agent-session."));
    expect(markers.length).toBe(1);
    expect(existsSync(join(workDir, "state", "s-start.json"))).toBe(true);

    await runHook({ hook_event_name: "SessionEnd", session_id: "s-start", reason: "other" });
    expect(readdirSync(workDir).filter((n) => n.startsWith(".veld-agent-session."))).toEqual([]);
    expect(existsSync(join(workDir, "state", "s-start.json"))).toBe(false);
  });

  it("returns UserPromptSubmit context as additionalContext JSON and ingests the prompt", async () => {
    const { stdout } = await runHook({ hook_event_name: "UserPromptSubmit", session_id: "s-prompt", prompt: "How do the cargo builds work here?" });
    expect(JSON.parse(stdout).hookSpecificOutput.hookEventName).toBe("UserPromptSubmit");
    const [surfacing] = to("/api/proactive_context");
    expect(surfacing.body.auto_ingest).toBe(true);
    expect(surfacing.body.sequence_number).toBe(1);
  });

  it("injects nothing on a timeout and does not treat the backend as down", async () => {
    responder = async (path) => {
      await Bun.sleep(1000);
      return defaultResponder(path);
    };
    const env = { VELD_HOOK_PROMPT_TIMEOUT_MS: "150", VELD_HOOK_TIMEOUT_MS: "150" };
    const start = await runHook({ hook_event_name: "SessionStart", session_id: "s-slow" }, env);
    const prompt = await runHook({ hook_event_name: "UserPromptSubmit", session_id: "s-slow", prompt: "a prompt long enough" }, env);
    expect(start.stdout).toBe("");
    expect(prompt.stdout).toBe("");

    responder = defaultResponder;
    const recovered = await runHook({ hook_event_name: "UserPromptSubmit", session_id: "s-slow", prompt: "a prompt long enough" }, env);
    const context = JSON.parse(recovered.stdout).hookSpecificOutput.additionalContext;
    expect(context).toContain("Past note about cargo builds");
    expect(context).not.toContain("veld-warning");
  });

  it("warns that memory is offline when the connection is refused", async () => {
    const closed = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() });
    const port = closed.port;
    closed.stop(true);
    const { stdout } = await runHook(
      { hook_event_name: "UserPromptSubmit", session_id: "s-down", prompt: "a prompt long enough" },
      { VELD_API_URL: `http://127.0.0.1:${port}` }
    );
    const context = JSON.parse(stdout).hookSpecificOutput.additionalContext;
    expect(context).toContain("<veld-warning>");
    expect(context).toContain("unreachable");
  });

  it("names the HTTP status when the server rejects the request", async () => {
    responder = () => new Response("unauthorized", { status: 401 });
    const { stdout } = await runHook({ hook_event_name: "SessionStart", session_id: "s-401" });
    expect(JSON.parse(stdout).hookSpecificOutput.additionalContext).toContain("HTTP 401");
  });
});

describe("isErrorOutput", () => {
  it("detects error/failure strings", () => {
    expect(isErrorOutput("fatal error occurred")).toBe(true);
    expect(isErrorOutput("Operation FAILED quickly")).toBe(true);
    expect(isErrorOutput("This failed with code 1")).toBe(true);
  });

  it("returns false on clean output", () => {
    expect(isErrorOutput("Command completed successfully")).toBe(false);
  });
});

describe("agent-session marker", () => {
  it("agentSessionMarkerPath uses cwd and pid", () => {
    const p = agentSessionMarkerPath("/tmp/proj", 4242);
    // join() normalises separators per-platform; basename check is portable.
    expect(p.endsWith(".veld-agent-session.4242")).toBe(true);
    expect(p).toContain("proj");
  });

  it("buildAgentSessionMarker produces the documented shape", () => {
    const now = new Date("2026-05-27T05:00:00.000Z");
    const marker = buildAgentSessionMarker(now);
    expect(marker.agent_id).toBe("Claude");
    // started_at is generated via new Date().toISOString() at write time, not
    // from the argument — verify it's a valid ISO-8601 UTC timestamp.
    expect(marker.started_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/);
    expect(marker.pid).toBe(process.pid);
    expect(typeof marker.binary).toBe("string");
    expect(marker.binary.length).toBeGreaterThan(0);
  });

  it("detectBinary returns a non-empty basename string", () => {
    const b = detectBinary();
    expect(typeof b).toBe("string");
    expect(b.length).toBeGreaterThan(0);
    // basename must not contain path separators
    expect(b.includes("/")).toBe(false);
    expect(b.includes("\\")).toBe(false);
  });

  it("writeAgentSessionMarker writes valid JSON atomically and removeAgentSessionMarker cleans up", () => {
    const dir = mkdtempSync(join(tmpdir(), "veld-marker-"));
    try {
      const path = agentSessionMarkerPath(dir, process.pid);
      const written = writeAgentSessionMarker(path);
      expect(written).not.toBeNull();
      expect(existsSync(path)).toBe(true);

      // Read back and assert the JSON shape.
      const raw = readFileSync(path, "utf-8");
      const parsed = JSON.parse(raw);
      expect(parsed.agent_id).toBe("Claude");
      expect(parsed.pid).toBe(process.pid);
      expect(typeof parsed.started_at).toBe("string");
      expect(parsed.started_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/);
      expect(typeof parsed.binary).toBe("string");

      // Temp file should be cleaned up after atomic rename.
      expect(existsSync(`${path}.tmp`)).toBe(false);

      // Idempotent overwrite.
      const second = writeAgentSessionMarker(path);
      expect(second).not.toBeNull();
      expect(existsSync(path)).toBe(true);

      removeAgentSessionMarker(path);
      expect(existsSync(path)).toBe(false);

      // Calling remove on a missing file is a no-op.
      removeAgentSessionMarker(path);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
