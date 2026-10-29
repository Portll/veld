/**
 * Veld - Agentic Memory + Vercel AI SDK Example
 *
 * A terminal chat that remembers across runs. Each turn recalls memories
 * relevant to your message into your turn of the prompt, and the model can
 * store new ones with veld_remember. Tell it something about yourself, quit,
 * run it again and ask.
 *
 * The model is any OpenAI-compatible endpoint; the default is a local Ollama,
 * so nothing leaves the machine.
 *
 * Setup:
 *   1. Start a veld server: veld server   (listens on http://127.0.0.1:3030)
 *   2. export VELD_API_KEY=<a key the server accepts>
 *   3. Start a model server, e.g. ollama pull qwen3:8b
 *   4. npm install && npm start
 */

import { stdin, stdout } from "node:process";
import { createInterface } from "node:readline";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { VeldClient, veldTools, withVeldMemories } from "@veld/vercel-ai";

const provider = createOpenAICompatible({
  name: "local",
  baseURL: process.env.LLM_BASE_URL ?? "http://127.0.0.1:11434/v1",
  ...(process.env.LLM_API_KEY && { apiKey: process.env.LLM_API_KEY }),
});
const model = provider(process.env.LLM_MODEL ?? "qwen3:8b");

// VELD_API_URL and VELD_API_KEY come from the environment.
const veld = new VeldClient({ userId: process.env.VELD_USER_ID ?? "vercel-ai-example" });
// Veld records every memory written over its HTTP API as untrusted, including the
// ones this chat's model stores, and recall withholds untrusted memories by default.
// This chat reads back what its model stored, so it opts in: each such memory
// reaches the model fenced and labelled trust="untrusted".
const includeUntrusted = true;
const tools = veldTools(veld, { tags: ["vercel-ai-example"], includeUntrusted });

const instructions =
  "You are a helpful assistant with long-term memory. When the user tells you something " +
  "worth keeping, such as a preference, a fact about them or a decision, store it with " +
  "veld_remember. Use veld_recall when you need something the recalled memories do not cover.";

const messages: ModelMessage[] = [];
const terminal = createInterface({ input: stdin, output: stdout, prompt: "> " });
// Input can end while a reply is still being generated; prompting after that throws.
let inputOpen = true;
terminal.on("close", () => {
  inputOpen = false;
});

console.log("Chat with memory. Ctrl+D to quit.");
terminal.prompt();
for await (const line of terminal) {
  if (line.trim()) {
    messages.push({ role: "user", content: line });
    const result = await generateText({
      model,
      tools,
      stopWhen: stepCountIs(5),
      ...(await withVeldMemories(veld, { instructions, messages }, { includeUntrusted })),
    });
    messages.push(...result.responseMessages);
    console.log(result.text);
  }
  if (inputOpen) terminal.prompt();
}
