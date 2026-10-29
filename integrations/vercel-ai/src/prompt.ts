import type { ModelMessage, SystemModelMessage, UserContent } from "ai";
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
  /**
   * Use memories whose source veld does not trust, labelled `trust="untrusted"`.
   * Default: true. `false` is strict mode: they are withheld. Veld records every
   * memory written over its HTTP API as untrusted, including the ones
   * `veld_remember` stores, so strict mode withholds all of those.
   */
  includeUntrusted?: boolean;
  /**
   * Keep memories veld flags as suspected prompt injection, labelled. Default:
   * false. Strict mode withholds them regardless.
   */
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

function prependText(content: UserContent, block: string): UserContent {
  if (typeof content === "string") {
    return [
      { type: "text", text: block },
      { type: "text", text: content },
    ];
  }
  return [{ type: "text", text: block }, ...content];
}

/**
 * The messages with the block in front of the last user message's content, or,
 * when there is no user message, in a user message of its own at the start.
 * The input array and its messages are not modified.
 */
function withBlockInMessages(messages: readonly ModelMessage[], block: string): ModelMessage[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message?.role === "user") {
      const copy = [...messages];
      copy[i] = { ...message, content: prependText(message.content, block) };
      return copy;
    }
  }
  return [{ role: "user", content: block }, ...messages];
}

/**
 * Recall memories relevant to a prompt and put them in front of the user's
 * turn. Spread the result into `generateText` or `streamText`:
 *
 *     const result = await generateText({
 *       model,
 *       ...(await withVeldMemories(veld, { instructions: "You are...", prompt })),
 *     });
 *
 * The memories go in the user turn, never in `instructions`: recalled text is
 * data, and the system channel is for the developer's instructions. A string
 * prompt gets the memory block before it; with messages, the block becomes a
 * text part at the start of the last user message. The block fences the
 * memories, escapes their text and labels each one's trust (see
 * `formatMemories`).
 *
 * Untrusted memories are used by default, labelled, as veld's contract asks:
 * untrusted content is data at prompt assembly. Suspected prompt injection is
 * withheld unless `includeSuspected`; `includeUntrusted: false` (strict mode)
 * withholds every untrusted memory.
 *
 * A deprecated `system` field is moved to `instructions`; the instructions are
 * otherwise returned unchanged. With no query, or when no memory is left to
 * use, the prompt is returned unchanged. A failed recall throws `VeldError`.
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

  if (typeof fields.prompt === "string") {
    return { ...result, prompt: `${block}\n\n${fields.prompt}` };
  }
  if (Array.isArray(fields.prompt)) {
    return { ...result, prompt: withBlockInMessages(fields.prompt, block) };
  }
  return { ...result, messages: withBlockInMessages(fields.messages ?? [], block) };
}
