export interface TokenStatusShape {
  tokens: number;
  budget: number;
  percent: number;
  alert: string | null;
}

const PROACTIVE_EXCLUDED_TOOLS = new Set([
  "remember",
  "recall",
  "forget",
  "list_memories",
  "proactive_context",
  "context_summary",
  "memory_stats",
]);

// An echo of a read returns what veld already holds as a new memory, so every
// recall would surface the recall that found it. Writes that store their own
// record (remember) or remove one (forget) would echo noise too.
const UNSTREAMED_TOOLS = new Set([
  "remember",
  "forget",
  "recall",
  "recall_by_tags",
  "read_memory",
  "list_memories",
  "context_summary",
  "proactive_context",
  "memory_stats",
  "verify_index",
  "consolidation_report",
  "fact_narratives",
  "facts_preview_purge",
  "token_status",
  "get_dials",
  "backup_list",
  "backup_verify",
  "list_reminders",
  "list_todos",
  "list_subtasks",
  "list_todo_comments",
  "list_projects",
  "todo_stats",
  "dataset_list",
  "dataset_query",
]);

export function shouldStreamToolCall(toolName: string): boolean {
  return !UNSTREAMED_TOOLS.has(toolName);
}

export function buildProgressBar(percentUsed: number, barLength = 20): string {
  const clamped = Math.max(0, Math.min(100, Math.round(percentUsed)));
  const filledLength = Math.round((clamped / 100) * barLength);
  return "█".repeat(filledLength) + "░".repeat(barLength - filledLength);
}

export function formatTokenStatusText(
  status: TokenStatusShape,
  sessionDurationMins: number,
): string {
  const remaining = status.budget - status.tokens;
  const percentUsed = Math.round(status.percent * 100);
  const bar = buildProgressBar(percentUsed);

  let response = `🌾 Token Status\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `${bar} ${percentUsed}%\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `Used: ${status.tokens.toLocaleString()} tokens\n`;
  response += `Budget: ${status.budget.toLocaleString()} tokens\n`;
  response += `Remaining: ${remaining.toLocaleString()} tokens\n`;
  response += `Session: ${sessionDurationMins} min\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += status.alert
    ? `⚠️ ALERT: ${percentUsed}% used - Consider new session`
    : `✓ Context window healthy`;
  return response;
}

export function formatResetTokenSessionText(
  previousTokens: number,
  budget: number,
): string {
  let response = `🌾 Token Session Reset\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `Previous: ${previousTokens.toLocaleString()} tokens\n`;
  response += `Current: 0 tokens\n`;
  response += `Budget: ${budget.toLocaleString()} tokens\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `✓ Counter cleared`;
  return response;
}

export function normalizeLimit(
  value: number | undefined,
  maxLimit = 250,
): number {
  const raw = value ?? 10;
  return Math.max(1, Math.min(maxLimit, Math.floor(raw)));
}

export function shouldAppendProactiveContext(toolName: string): boolean {
  return !PROACTIVE_EXCLUDED_TOOLS.has(toolName);
}

export function extractStringContextFromArgs(
  args: unknown,
  minLength = 10,
  maxLen = 1000,
): string {
  const contextParts: string[] = [];

  if (args && typeof args === "object") {
    for (const value of Object.values(args as Record<string, unknown>)) {
      if (typeof value === "string" && value.length > minLength) {
        contextParts.push(value);
      }
    }
  }

  return contextParts.join(" ").slice(0, maxLen);
}

const AUTO_STREAM_EXCLUDED_TOOLS = new Set([
  "remember",
  "proactive_context",
  "streaming_status",
  "token_status",
  "reset_token_session",
]);

const AUTO_STREAM_MIN_LENGTH = 20;

/**
 * The text a tool call contributes to the memory stream, or null when it contributes nothing.
 *
 * A question is not knowledge. Streaming `args.query` stored every recall as a memory of its
 * own, and because the stored text matched the next identical question better than any answer,
 * the question came back as the top result. `remember` is excluded because it stores its content
 * itself; the meta tools because they carry no conversation content.
 */
export function autoStreamText(
  toolName: string,
  args: Record<string, unknown> | undefined,
): string | null {
  if (AUTO_STREAM_EXCLUDED_TOOLS.has(toolName) || !args) return null;
  if (typeof args.query === "string") return null;
  const text =
    typeof args.content === "string"
      ? args.content
      : typeof args.context === "string"
        ? args.context
        : "";
  return text.length >= AUTO_STREAM_MIN_LENGTH ? text : null;
}
