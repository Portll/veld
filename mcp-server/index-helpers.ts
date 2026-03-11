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

  let response = `🐘 Token Status\n`;
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
  let response = `🐘 Token Session Reset\n`;
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
