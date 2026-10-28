import type { RecallMemory } from "../src/types.js";

export const BASE_URL = "http://veld.test:3030";

export interface RecordedCall {
  url: string;
  path: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
  signal: AbortSignal | undefined;
}

/** A fetch that records every call and answers from `handler`; nothing leaves the process. */
export function fakeFetch(handler: (call: RecordedCall) => Response | Promise<Response>) {
  const calls: RecordedCall[] = [];
  const fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const call: RecordedCall = {
      url,
      path: url.startsWith(BASE_URL) ? url.slice(BASE_URL.length) : url,
      method: init?.method ?? "GET",
      headers: (init?.headers ?? {}) as Record<string, string>,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
      signal: init?.signal ?? undefined,
    };
    calls.push(call);
    return handler(call);
  }) as typeof globalThis.fetch;
  return { fetch, calls };
}

export function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function recallMemory(
  id: string,
  content: string,
  extra: Partial<RecallMemory> = {},
): RecallMemory {
  return {
    id,
    experience: { content, memory_type: "Decision", tags: [] },
    importance: 0.7,
    created_at: "2026-10-28T09:15:30.123456789+00:00",
    score: 0.9,
    tier: "Working",
    ...extra,
  };
}

/** A V4 model result carrying text, for MockLanguageModelV4. */
export function textResult(text: string) {
  return {
    content: [{ type: "text" as const, text }],
    finishReason: { unified: "stop" as const, raw: "stop" },
    usage: {
      inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 1, text: 1, reasoning: 0 },
    },
    warnings: [],
  };
}

/** A V4 model result carrying one tool call, for MockLanguageModelV4. */
export function toolCallResult(toolName: string, input: unknown) {
  return {
    content: [
      {
        type: "tool-call" as const,
        toolCallId: "call-1",
        toolName,
        input: JSON.stringify(input),
      },
    ],
    finishReason: { unified: "tool-calls" as const, raw: "tool_calls" },
    usage: {
      inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 1, text: 1, reasoning: 0 },
    },
    warnings: [],
  };
}
