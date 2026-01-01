#!/usr/bin/env bun
/**
 * Shodh Memory Hook - Native Claude Code Integration
 *
 * This replaces Cortex entirely. Instead of a proxy, we hook directly
 * into Claude Code's lifecycle events.
 *
 * Usage: Called by Claude Code hooks system
 * Events: SessionStart, UserPromptSubmit, Stop, PostToolUse
 */

const SHODH_API_URL = process.env.SHODH_API_URL || "http://127.0.0.1:3030";
const SHODH_API_KEY = process.env.SHODH_API_KEY || "sk-shodh-dev-local-testing-key";
const SHODH_USER_ID = process.env.SHODH_USER_ID || "claude-code";

interface HookInput {
  hook_event_name: string;
  prompt?: string;
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: string;
  stop_reason?: string;
  session_id?: string;
}

interface Memory {
  id: string;
  content: string;
  memory_type: string;
  score: number;
}

interface ProactiveResponse {
  memories: Memory[];
  memory_count: number;
}

interface RecallResponse {
  results: Memory[];
}

async function callBrain(endpoint: string, body: Record<string, unknown>): Promise<unknown> {
  try {
    const response = await fetch(`${SHODH_API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": SHODH_API_KEY,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      console.error(`Brain error: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (e) {
    // Brain unavailable - fail silently, don't block Claude
    return null;
  }
}

async function handleSessionStart(): Promise<void> {
  // Get project context
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const projectName = projectDir.split("/").pop() || "unknown";

  // Build context string
  let context = `Starting session in: ${projectName}`;

  // Query proactive context
  const response = await callBrain("/api/proactive_context", {
    user_id: SHODH_USER_ID,
    context,
    max_results: 5,
    auto_ingest: true, // NOW WE ENABLE IT - brain sees session start
  }) as ProactiveResponse | null;

  if (response?.memories?.length) {
    console.error(`[shodh] Loaded ${response.memories.length} memories`);

    // Format for Claude's context
    const memoryBlock = response.memories
      .map((m, i) => `${i + 1}. [${m.memory_type}] ${m.content.slice(0, 200)}...`)
      .join("\n");

    // Write to project memory file for auto-loading
    const memoryFile = `${projectDir}/.claude/memory-context.md`;
    await Bun.write(memoryFile, `# Session Memory\n\n${memoryBlock}\n`);
  }
}

async function handleUserPrompt(input: HookInput): Promise<void> {
  const prompt = input.prompt;
  if (!prompt || prompt.length < 10) return;

  // Semantic search for relevant memories
  const response = await callBrain("/api/recall", {
    user_id: SHODH_USER_ID,
    query: prompt.slice(0, 500),
    limit: 3,
  }) as RecallResponse | null;

  if (response?.results?.length) {
    // Output additional context for Claude to see
    const context = response.results
      .map(m => `[${m.memory_type}] ${m.content.slice(0, 100)}`)
      .join("; ");

    // Return hook output - Claude Code will inject this
    console.log(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: `Relevant memories: ${context}`,
      },
    }));
  }
}

async function handleStop(input: HookInput): Promise<void> {
  // Store that this session ended - useful for session boundary detection
  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content: `Session ended: ${input.stop_reason || "user_stop"}`,
    memory_type: "Context",
    tags: ["session:end", `reason:${input.stop_reason || "unknown"}`],
  });
}

async function handlePostToolUse(input: HookInput): Promise<void> {
  const toolName = input.tool_name;
  const toolOutput = input.tool_output;

  // Only store significant tool uses
  if (!toolName || !toolOutput) return;

  // Skip noisy tools
  const skipTools = ["Glob", "Grep", "Read", "LSP"];
  if (skipTools.includes(toolName)) return;

  // Store tool usage pattern
  const content = `Used ${toolName}: ${JSON.stringify(input.tool_input || {}).slice(0, 200)}`;

  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content,
    memory_type: "Task",
    tags: [`tool:${toolName}`, "source:hook"],
  });
}

async function main(): Promise<void> {
  // Read input from stdin
  const inputText = await Bun.stdin.text();

  let input: HookInput;
  try {
    input = JSON.parse(inputText);
  } catch {
    // No input or invalid JSON - check for event type from args
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
    case "Stop":
      await handleStop(input);
      break;
    case "PostToolUse":
      await handlePostToolUse(input);
      break;
    default:
      // Unknown event - ignore
      break;
  }
}

main().catch(console.error);
