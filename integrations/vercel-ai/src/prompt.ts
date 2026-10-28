import type { ModelMessage, SystemModelMessage } from "ai";
import type { VeldClient } from "./client.js";
import { formatMemories } from "./format.js";
import type { RetrievalMode } from "./types.js";

/** The AI SDK's `instructions` value: a string or system messages. */
export type Instructions = string | SystemModelMessage | SystemModelMessage[];

/** The prompt fields of a `generateText`/`streamText` call. */
export interface PromptFields {
  instructions?: Instructions;
  /** Deprecated in the AI SDK in favour of `instructions`; moved there. */
  system?: Instructions;
  prompt?: string | ModelMessage[];
  messages?: ModelMessage[];
}

export interface MemoryPromptOptions {
  /** Text to recall with. Default: the prompt string, else the last user message's text. */
  query?: string;
  /** Memories to recall. Default: 5. */
  limit?: number;
  mode?: RetrievalMode;
  /** Recall only memories carrying at least one of these tags. */
  tags?: string[];
  /** Keep memories veld flags as suspected prompt injection. Default: false. */
  includeSuspected?: boolean;
  signal?: AbortSignal;
}

export type WithMemories<T extends PromptFields> = Omit<T, "system" | "instructions"> & {
  instructions?: Instructions;
};

function messageText(message: ModelMessage): string {
  if (typeof message.content === "string") return message.content;
  return message.content
    .map((part) => (part.type === "text" ? part.text : ""))
    .filter((text) => text.length > 0)
    .join("\n");
}

/** The text a prompt is about: the prompt string, else the last user message's text. */
export function queryFromPrompt(fields: PromptFields): string {
  if (typeof fields.prompt === "string") return fields.prompt.trim();
  const messages = fields.messages ?? (Array.isArray(fields.prompt) ? fields.prompt : []);
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message && message.role === "user") return messageText(message).trim();
  }
  return "";
}

function appendInstructions(existing: Instructions | undefined, block: string): Instructions {
  if (existing === undefined) return block;
  if (typeof existing === "string") return existing ? `${existing}\n\n${block}` : block;
  const messages = Array.isArray(existing) ? existing : [existing];
  return [...messages, { role: "system", content: block }];
}

/**
 * Recall memories relevant to a prompt and add them to its `instructions`.
 * Spread the result into `generateText` or `streamText`:
 *
 *     const result = await generateText({
 *       model,
 *       ...(await withVeldMemories(veld, { instructions: "You are...", prompt })),
 *     });
 *
 * A deprecated `system` field is moved to `instructions`. With no query, or
 * when nothing relevant is recalled, the instructions are returned unchanged.
 * A failed recall throws `VeldError`.
 */
export async function withVeldMemories<T extends PromptFields>(
  client: VeldClient,
  fields: T,
  options: MemoryPromptOptions = {},
): Promise<WithMemories<T>> {
  const { system, instructions, ...rest } = fields;
  const original = instructions ?? system;
  const result = { ...rest, ...(original !== undefined && { instructions: original }) };

  const query = (options.query ?? queryFromPrompt(fields)).trim();
  if (!query) return result;

  const reply = await client.recall(
    {
      query,
      limit: options.limit ?? 5,
      ...(options.mode && { mode: options.mode }),
      ...(options.tags && { tags: options.tags }),
    },
    options.signal,
  );
  const block = formatMemories(reply.memories, options);
  if (!block) return result;
  return { ...rest, instructions: appendInstructions(original, block) };
}
