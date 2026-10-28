# @veld/vercel-ai

Veld memory for the [Vercel AI SDK](https://ai-sdk.dev) (v7): `veld_remember` and
`veld_recall` as AI SDK tools, and a helper that recalls memories relevant to a prompt
and adds them to its instructions.

It calls only the veld server you configure. It sends no telemetry and makes no other
network calls.

## Install

The package is not on npm yet. Build it from this directory and install it from the path:

```bash
cd integrations/vercel-ai
npm install
npm run build
cd /your/app
npm install ai zod /path/to/veld/integrations/vercel-ai
```

`ai` (^7) and `zod` (^3.25.76 or ^4) are peer dependencies. Node 22 or later.

## Configure

```ts
import { VeldClient } from "@veld/vercel-ai";

const veld = new VeldClient({
  baseUrl: "http://127.0.0.1:3030", // default: VELD_API_URL, else http://127.0.0.1:3030
  apiKey: process.env.VELD_API_KEY, // default: VELD_API_KEY; required
  userId: "alice",                  // default: VELD_USER_ID, else "default"
});
```

`userId` is the memory namespace. In a multi-user app, make one client per end user.
The API key goes in the `X-API-Key` header; the client warns when it would be sent over
plain HTTP to a server other than localhost (set `VELD_ALLOW_HTTP=true` to silence it).

## Tools

```ts
import { generateText, stepCountIs } from "ai";
import { veldTools } from "@veld/vercel-ai";

const result = await generateText({
  model,
  tools: veldTools(veld),
  stopWhen: stepCountIs(5),
  prompt: "Remember that I'm allergic to peanuts, then suggest a snack.",
});
```

| Tool | Input | Output |
|---|---|---|
| `veld_remember` | `content`, optional `memoryType` (Observation, Decision, Learning, Error, Discovery, Pattern, Context, Task, Conversation), optional `tags` | `{ id, stored: true }` |
| `veld_recall` | `query`, optional `limit` (1 to 50) | `{ memories: [{ id, content, type, createdAt, score }], count }` |

Options: `veldTools(veld, { recallLimit, mode, tags, includeSuspected })`. `tags` are
added to every stored memory; `mode` is `"semantic"`, `"associative"` or `"hybrid"`.

Memories the model stores are recorded with `source_type: "ai_generated"`, so veld scans
them for prompt injection when they are recalled.

## Memories in the prompt

`withVeldMemories` recalls with the prompt text (or the last user message) and adds the
memories to `instructions`, fenced and marked as reference data:

```ts
import { generateText } from "ai";
import { withVeldMemories } from "@veld/vercel-ai";

const result = await generateText({
  model,
  ...(await withVeldMemories(veld, {
    instructions: "You are a helpful assistant.",
    messages,
  })),
});
```

Options: `{ query, limit, mode, tags, includeSuspected, signal }`. A deprecated `system`
field is moved to `instructions`. When nothing relevant is recalled, the fields come back
unchanged. `formatMemories(reply.memories)` renders the same block from a recall you
made yourself.

## Client

`VeldClient` also exposes the routes directly:

| Method | Route |
|---|---|
| `remember({ content, memoryType?, tags?, sourceType? })` | `POST /api/remember` |
| `recall({ query, limit?, mode?, tags? })` | `POST /api/recall` |
| `list({ limit?, offset?, type?, query?, tags? })` | `POST /api/memories` |
| `forget(id)` | `DELETE /api/memory/{id}` |

Request and response types follow veld's
[HTTP API schemas](../../docs/src/reference/schemas.md) and are exported (`RecallMemory`,
`RecallResponse`, `ListResponse`, ...).

## Errors

Every failed call throws `VeldError` with:

- `status`: the HTTP status, or `undefined` when no reply arrived.
- `code`: the server's error code (`INVALID_API_KEY`, `INVALID_INPUT`, `MEMORY_NOT_FOUND`,
  ...), or `TIMEOUT`, `CONNECTION_FAILED`, `INVALID_RESPONSE`, `NOT_STORED`, `HTTP_ERROR`.

Inside a tool, the AI SDK reports the error to the model as a tool error.
`withVeldMemories` throws, so catch it if a reply should go ahead without memory. Calls
are not retried; a caller's `AbortSignal` is passed through. The default timeout is
10 seconds (`timeoutMs`).

## Suspected prompt injection

Veld flags recalled memories from untrusted sources whose content looks like
instructions (`injection_suspected`). Both the recall tool and `withVeldMemories` drop
them by default. Pass `includeSuspected: true` to keep them; they are then labelled.

## Develop

```bash
npm install
npm run typecheck
npm test
```

The tests stub `fetch` and the language model, so they need no veld server and no model
provider.
