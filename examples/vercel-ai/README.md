# Veld - Agentic Memory + Vercel AI SDK Integration

Give a Vercel AI SDK (v7) app persistent memory with
[`@veld/vercel-ai`](../../integrations/vercel-ai): veld remember and recall as tools, and
recalled memories added to each call's instructions.

## Installation

The package is not on npm yet; build it from this repository first:

```bash
cd integrations/vercel-ai && npm install && npm run build && cd -
cd examples/vercel-ai && npm install
```

## Quick Start

```ts
import { generateText, stepCountIs } from "ai";
import { VeldClient, veldTools, withVeldMemories } from "@veld/vercel-ai";

const veld = new VeldClient({ userId: "alice" }); // VELD_API_URL, VELD_API_KEY from the env

const result = await generateText({
  model,
  tools: veldTools(veld),
  stopWhen: stepCountIs(5),
  ...(await withVeldMemories(veld, {
    instructions: "You are a helpful assistant with long-term memory.",
    messages,
  })),
});
```

- `veldTools(veld)` gives the model `veld_remember` and `veld_recall`.
- `withVeldMemories(veld, fields)` recalls with the last user message and adds the
  memories to `instructions`, fenced and marked as data rather than instructions.

## Available Tools

| Tool | Description |
|------|-------------|
| `veld_remember` | Store a memory with an optional type (Decision, Learning, ...) and tags |
| `veld_recall` | Search memory in natural language; most relevant first |

## Example

`chat.ts` is a terminal chat that remembers across runs. It uses any OpenAI-compatible
model server and defaults to a local Ollama, so the whole loop can run offline.

```bash
veld server &
export VELD_API_KEY=your-api-key
ollama pull qwen3:8b
npm start
```

Tell it your name, press Ctrl+D, run `npm start` again and ask. Environment:

| Variable | Default | Meaning |
|---|---|---|
| `VELD_API_URL` | `http://127.0.0.1:3030` | Veld server |
| `VELD_API_KEY` | (required) | Veld API key |
| `VELD_USER_ID` | `vercel-ai-example` | Veld user the chat's memories belong to |
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, ...) |
| `LLM_MODEL` | `qwen3:8b` | Model name on that endpoint; it must support tool calls |
| `LLM_API_KEY` | none | Key for the endpoint, if it needs one |

Requires Node 22.18 or later, which runs `chat.ts` directly.

## Documentation

- [`@veld/vercel-ai` README](../../integrations/vercel-ai/README.md): options, errors, client
- [Veld Documentation](https://github.com/Portll/veld)
- [AI SDK tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling)
