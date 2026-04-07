# Veld - Agentic Memory + LangChain Integration

Use Veld - Agentic Memory as a persistent memory backend for LangChain agents.

## Installation

```bash
pip install langchain langchain-mcp-adapters langchain-anthropic
```

## Quick Start

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

async with MultiServerMCPClient({
    "veld": {
        "command": "npx",
        "args": ["-y", "@veld/memory-mcp"],
    }
}) as client:
    tools = client.get_tools()
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    agent = create_react_agent(model, tools)

    # Agent now has persistent memory!
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Remember that I prefer TypeScript"}]
    })
```

## Available Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with type and tags |
| `recall` | Semantic search across memories |
| `proactive_context` | Auto-surface relevant memories (recommended) |
| `recall_by_tags` | Filter memories by tags |
| `context_summary` | Get overview of recent learnings |
| `memory_stats` | System health and statistics |

## Examples

See `veld_langchain.py` for complete examples including:
- Basic memory storage and retrieval
- Stateful sessions
- HTTP transport for remote servers

## Documentation

- [Veld Documentation](https://www.veld-rag.com/memory)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
