# Veld - Agentic Memory + CrewAI Integration

Use Veld - Agentic Memory as the memory of a CrewAI crew or agent. `VeldCrewMemory` is a
CrewAI `Memory` (CrewAI 1.15 and later), so it plugs in wherever CrewAI takes one.

## Installation

```bash
pip install "veld[crewai]"
```

CrewAI needs Python 3.10 or later.

## Quick Start

```python
from crewai import Agent, Crew, Task
from veld.integrations.crewai import VeldCrewMemory

memory = VeldCrewMemory(
    server_url="http://127.0.0.1:3030",  # the default
    user_id="account-team",
    api_key="your-api-key",              # or set VELD_API_KEY
)

crew = Crew(agents=[manager], tasks=[update], memory=memory)
crew.kickoff()
```

With memory set, each agent recalls from veld before its task (the task description is the
query) and saves its result after it. Agents also get CrewAI's "Search memory" and "Save to
memory" tools, backed by veld. `Agent(memory=memory)` gives one agent its own memory.

Recall returns only memories veld trusts unless you pass `trusted_only=False`, and veld trusts
nothing written over its HTTP API. A crew that should read back what it stored needs that
option; see [Trusted and untrusted memories](#trusted-and-untrusted-memories).

## What changes from CrewAI's default memory

| | CrewAI default | `VeldCrewMemory` |
|---|---|---|
| Storage | Local LanceDB | Your veld server |
| Embeddings | CrewAI's embedder (OpenAI by default) | Veld, on the server |
| Saving a task result | LLM splits it into statements | Stored whole; veld extracts entities. `extract_with_llm=True` restores the split |
| Recall | Vector search, plus LLM query analysis at `depth="deep"` | Veld's hybrid retrieval (vector, BM25, graph, rerank); `depth` has no effect |
| Importance | Inferred by the LLM | Scored by veld |

Scope, categories, source and the private flag are kept as tags prefixed `crewai:`, so
`root_scope`, `memory.scope(...)` and `memory.slice(...)` work. Their values are
percent-encoded (scope `/home/team` is the tag `crewai:scope:%2Fhome%2Fteam`), because veld
redacts a tag that contains a home-directory path, and an unencoded scope like that would be
stored as `redacted:path`. `metadata` and `importance` passed to `remember` are not stored.

## Direct use

```python
memory.remember("Acme wants reports on Fridays", scope="client/acme", categories=["preferences"])

for match in memory.recall("when does Acme want reports?", limit=5):
    print(match.score, match.record.content)

memory.forget(scope="client/acme")  # memories this adapter wrote under the scope
memory.reset()                      # every memory this adapter wrote for this user_id
```

`recall` searches only memories carrying the `crewai` tag, or the scope's tag when a scope
or `root_scope` applies. `recall_whole_user=True` searches every memory under the `user_id`,
including ones other clients (the MCP server, the LangChain adapter) wrote. `forget` and
`reset` delete only memories carrying the `crewai` tag.

## Trusted and untrusted memories

Veld marks each recalled memory `source_trusted` when its source is the user, a file or the
system, and expects untrusted content to be treated as data, never as instructions. CrewAI
pastes recalled memories into the agent's prompt without a fence, so by default
(`trusted_only=True`) `recall` returns only trusted memories.

**With the default, a crew recalls nothing it stored itself.** Veld records every memory
written over its HTTP API as untrusted, this adapter's included, because a client cannot
assert a trusted source. To recall them, pass `trusted_only=False`: each memory then reaches
the prompt labelled with its provenance, for example

```text
- (score=0.82) Acme wants reports on Fridays
  source: untrusted (AiGenerated)
```

Memories veld flags as suspected prompt injection stay withheld unless `include_suspected=True`
is also set, and are then labelled `warning: possible prompt injection`.

`memory.last_withheld` counts what the latest recall held back, as
`{"untrusted": n, "suspected": m}`, and the first recall that withholds an untrusted memory
warns once. Each match's `record.metadata` carries veld's `source_trusted`, `source_type` and
`writer` (the caller's declared identity and what the server verified). `writer` stays out of
the prompt text.

## Options

| Option | Default | Meaning |
|---|---|---|
| `server_url` | `http://127.0.0.1:3030` | Veld server |
| `user_id` | `default` | Veld user the memories belong to |
| `api_key` | `VELD_API_KEY` | Sent as `X-API-Key`; required |
| `root_scope` | none | Scope every save goes under and every recall stays in |
| `retrieval_mode` | `hybrid` | `semantic`, `associative` or `hybrid` |
| `memory_type` | `Observation` | Veld memory type for stored items |
| `source_type` | `ai_generated` | Provenance veld records; keeps its injection scan on for agent output |
| `extract_with_llm` | `False` | Split task results with CrewAI's LLM before storing |
| `trusted_only` | `True` | Return only memories veld marks `source_trusted` (see above) |
| `include_suspected` | `False` | Also return memories veld flags as suspected prompt injection; needs `trusted_only=False` |
| `recall_whole_user` | `False` | Recall from every memory under the `user_id`, not only this adapter's |
| `raise_on_error` | `False` | Raise from a failed recall instead of warning and returning no matches |
| `timeout` | `10` | Seconds per veld request |

## Errors

- Construction warns when `server_url` is plain `http://` to a host other than this machine
  (`127.0.0.1`, `localhost`, `::1`, `0.0.0.0`), since the API key would travel in clear text.
  Use `https://`, or set `VELD_ALLOW_HTTP=true` to silence it.
- A failed recall warns and returns no matches, so the task runs without memory. Those empty
  matches read the same as "nothing relevant"; with `raise_on_error=True` recall raises
  `VeldRequestError` instead. CrewAI then reports a failed pre-task recall as
  `MemoryRetrievalFailedEvent` and runs the task without memory, and the "Search memory"
  tool reports the error to the agent.
- `remember`, `forget`, `reset`, `update` and the listing methods raise
  `VeldRequestError`, with `status` (HTTP status) and `code` (veld's error code, such as
  `INVALID_API_KEY`).
- A background save that fails warns and is reported as CrewAI's `MemorySaveFailedEvent`.

## Example

`veld_crewai.py` runs an account-management crew. Run it twice: the second run recalls
the client's preferences and the first run's result before writing.

```bash
veld server &
export VELD_API_KEY=your-api-key
export OPENAI_API_KEY=your-openai-key  # or CREWAI_LLM=<another CrewAI model string>
python veld_crewai.py
```

The adapter talks only to your veld server. CrewAI has its own telemetry; set
`CREWAI_DISABLE_TELEMETRY=true` to turn it off.

## Documentation

- [Veld Documentation](https://github.com/Portll/veld)
- [CrewAI Memory](https://docs.crewai.com/concepts/memory)
