"""
Veld Integrations

Optional adapters for popular LLM frameworks:
- LangChain: VeldMemory class for ConversationChain, agents
- LlamaIndex: VeldLlamaMemory for chat engines
- OpenAI Agents SDK: VeldTools + VeldSession for agents

Install extras:
    pip install veld[langchain]
    pip install veld[llamaindex]
    pip install veld[openai-agents]
    pip install veld[all]
"""

# Lazy imports to avoid requiring dependencies
def get_langchain_memory():
    """Get LangChain VeldMemory class (requires langchain installed)"""
    from .langchain import VeldMemory
    return VeldMemory

def get_llamaindex_memory():
    """Get LlamaIndex VeldLlamaMemory class (requires llama-index installed)"""
    from .llamaindex import VeldLlamaMemory
    return VeldLlamaMemory

def get_openai_agents_tools():
    """Get OpenAI Agents SDK VeldTools class (requires openai-agents installed)"""
    from .openai_agents import VeldTools
    return VeldTools

def get_openai_agents_session():
    """Get OpenAI Agents SDK VeldSession class (requires openai-agents installed)"""
    from .openai_agents import VeldSession
    return VeldSession

__all__ = [
    "get_langchain_memory",
    "get_llamaindex_memory",
    "get_openai_agents_tools",
    "get_openai_agents_session",
]
