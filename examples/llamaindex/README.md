# Veld - Agentic Memory + LlamaIndex Integration

Use Veld - Agentic Memory as a persistent memory backend for LlamaIndex agents.

## Installation

```bash
pip install llama-index llama-index-llms-anthropic shodh-memory
```

## Quick Start

```python
from shodh_memory import Memory

# Option 1: Use the Python SDK directly
memory = Memory(user_id="my-agent")
memory.remember("User prefers TypeScript", memory_type="Pattern")
results = memory.recall("programming preferences")

# Option 2: Use as a LlamaIndex memory block (see example)
from examples.llamaindex.shodh_memory_llamaindex import VeldProactiveMemory

memory = VeldProactiveMemory(user_id="my-agent")
```

## Custom Memory Blocks

The example provides two memory block implementations:

### VeldMemoryBlock

Basic memory block with semantic search:

```python
from llama_index.core.memory import BaseMemory
from shodh_memory import Memory

class VeldMemoryBlock(BaseMemory):
    def get(self, input: str) -> List[ChatMessage]:
        # Retrieves relevant memories for current input

    def put(self, message: ChatMessage) -> None:
        # Stores message with auto-inferred memory type
```

### VeldProactiveMemory (Recommended)

Enhanced memory that proactively surfaces context:

```python
memory = VeldProactiveMemory(
    user_id="my-agent",
    max_memories=5,
)

# Automatically:
# - Retrieves semantically relevant memories
# - Stores current context as Conversation memory
# - Builds association graphs over time
```

## Creating an Agent

```python
from llama_index.llms.anthropic import Anthropic
from llama_index.core.agent import FunctionCallingAgent

memory = VeldProactiveMemory(user_id="my-agent")
llm = Anthropic(model="claude-sonnet-4-20250514")

agent = FunctionCallingAgent.from_tools(
    tools=[],  # Your tools
    llm=llm,
    memory=memory,
    verbose=True,
)

# Agent now has persistent memory across sessions!
response = agent.chat("What do you remember about our project?")
```

## Memory Type Inference

The memory block automatically infers types from content:

| Keywords | Memory Type |
|----------|-------------|
| decided, chose, will use | Decision |
| learned, discovered, realized | Learning |
| error, bug, fixed | Error |
| pattern, always, prefers | Pattern |
| (default) | Context |

## Documentation

- [Veld Documentation](https://www.shodh-rag.com/memory)
- [LlamaIndex Memory Guide](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/)
