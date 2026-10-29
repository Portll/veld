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

`recall` searches every memory under the `user_id`, including ones other clients (the MCP
server, the LangChain adapter) wrote. `forget` and `reset` delete only memories carrying the
`crewai` tag.

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
| `include_suspected` | `False` | Return memories veld flags as suspected prompt injection |
| `timeout` | `10` | Seconds per veld request |

## Errors

- Construction warns when `server_url` is plain `http://` to a host other than this machine
  (`127.0.0.1`, `localhost`, `::1`, `0.0.0.0`), since the API key would travel in clear text.
  Use `https://`, or set `VELD_ALLOW_HTTP=true` to silence it.
- A failed recall warns and returns no matches, so the task runs without memory.
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
