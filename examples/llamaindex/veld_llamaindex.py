"""
Veld - Agentic Memory + LlamaIndex Integration Example

This example shows how to use Veld - Agentic Memory as a custom memory
backend for LlamaIndex agents.

Requirements:
    pip install llama-index llama-index-llms-anthropic veld

Setup:
    1. Install veld Python SDK: pip install veld
    2. Or start server: npx -y @veld/memory-mcp
    3. Run this script

Documentation: https://www.veld-rag.com/memory
"""

import os
from typing import Any, List, Optional
from dataclasses import dataclass

from llama_index.core.memory import BaseMemory
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.anthropic import Anthropic
from llama_index.core.agent import FunctionCallingAgent

# Option 1: Use the Python SDK directly
try:
    from veld import Memory
    USE_SDK = True
except ImportError:
    USE_SDK = False
    import httpx  # Fallback to HTTP API


class VeldMemoryBlock(BaseMemory):
    """
    Custom LlamaIndex memory block backed by Veld - Agentic Memory.

    Features:
    - Persistent storage across sessions
    - Hebbian learning (associations strengthen with use)
    - Semantic search for relevant memories
    - Tag-based organization
    """

    user_id: str = Field(default="llamaindex-agent", description="User ID for memory isolation")
    api_url: str = Field(default="http://localhost:3030", description="Veld server URL")
    api_key: Optional[str] = Field(default=None, description="API key if authentication enabled")
    max_memories: int = Field(default=10, description="Max memories to retrieve per query")

    _memory: Any = None  # SDK instance
    _client: Any = None  # HTTP client fallback

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if USE_SDK:
            self._memory = Memory(user_id=self.user_id)
        else:
            self._client = httpx.Client(
                base_url=self.api_url,
                headers={"X-API-Key": self.api_key} if self.api_key else {},
            )

    def get(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        """
        Retrieve relevant memories for the current input.

        Uses semantic search to find memories related to the input,
        then formats them as system messages for context.
        """
        if not input:
            return []

        memories = self._recall(input)

        if not memories:
            return []

        # Format memories as a system message
        memory_context = "Relevant memories from previous sessions:\n\n"
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", mem.get("experience", {}).get("content", ""))
            mem_type = mem.get("memory_type", mem.get("experience", {}).get("experience_type", ""))
            memory_context += f"{i}. [{mem_type}] {content}\n"

        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=memory_context,
            )
        ]

    def put(self, message: ChatMessage) -> None:
        """
        Store a message in long-term memory.

        Automatically determines memory type based on content.
        """
        content = message.content
        if not content:
            return

        # Determine memory type from content
        memory_type = self._infer_memory_type(content)

        self._remember(content, memory_type)

    def set(self, messages: List[ChatMessage]) -> None:
        """Store multiple messages in memory."""
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        """Clear all memories for this user (use with caution)."""
        if USE_SDK:
            # SDK doesn't have a clear-all method, would need to iterate
            pass
        else:
            # HTTP API: would need forget endpoint
            pass

    def _recall(self, query: str) -> List[dict]:
        """Retrieve memories matching the query."""
        if USE_SDK:
            results = self._memory.recall(query, limit=self.max_memories)
            return results.get("memories", [])
        else:
            response = self._client.post(
                "/api/recall",
                json={
                    "user_id": self.user_id,
                    "query": query,
                    "limit": self.max_memories,
                },
            )
            response.raise_for_status()
            return response.json().get("memories", [])

    def _remember(self, content: str, memory_type: str = "Context") -> None:
        """Store a new memory."""
        if USE_SDK:
            self._memory.remember(content, memory_type=memory_type)
        else:
            response = self._client.post(
                "/api/remember",
                json={
                    "user_id": self.user_id,
                    "content": content,
                    "memory_type": memory_type,
                },
            )
            response.raise_for_status()

    def _infer_memory_type(self, content: str) -> str:
        """Infer the appropriate memory type from content."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["decided", "chose", "will use", "going with"]):
            return "Decision"
        elif any(word in content_lower for word in ["learned", "discovered", "found out", "realized"]):
            return "Learning"
        elif any(word in content_lower for word in ["error", "bug", "fixed", "issue"]):
            return "Error"
        elif any(word in content_lower for word in ["pattern", "always", "usually", "prefers"]):
            return "Pattern"
        else:
            return "Context"


class VeldProactiveMemory(VeldMemoryBlock):
    """
    Enhanced memory block that proactively surfaces relevant context.

    Uses Veld's proactive_context endpoint which:
    - Retrieves semantically relevant memories
    - Stores the current context as a Conversation memory
    - Builds association graphs over time
    """

    def get(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        """
        Proactively surface relevant memories and store current context.

        This is the recommended way to use Veld - Agentic Memory - call this
        at the start of every conversation turn.
        """
        if not input:
            return []

        memories = self._proactive_context(input)

        if not memories:
            return []

        memory_context = "Context from your persistent memory:\n\n"
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            relevance = mem.get("relevance_score", 0)
            memory_context += f"{i}. (relevance: {relevance:.2f}) {content}\n"

        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=memory_context,
            )
        ]

    def _proactive_context(self, context: str) -> List[dict]:
        """Call proactive_context endpoint."""
        if USE_SDK:
            results = self._memory.proactive_context(context, max_results=self.max_memories)
            return results.get("memories", [])
        else:
            response = self._client.post(
                "/api/proactive_context",
                json={
                    "user_id": self.user_id,
                    "context": context,
                    "max_results": self.max_memories,
                },
            )
            response.raise_for_status()
            return response.json().get("memories", [])


def create_agent_with_veld_memory():
    """
    Create a LlamaIndex agent with Veld - Agentic Memory for persistent context.
    """
    # Initialize memory
    memory = VeldProactiveMemory(
        user_id="my-llamaindex-agent",
        max_memories=5,
    )

    # Initialize LLM
    llm = Anthropic(model="claude-sonnet-4-20250514")

    # Create agent with memory
    agent = FunctionCallingAgent.from_tools(
        tools=[],  # Add your tools here
        llm=llm,
        memory=memory,
        verbose=True,
    )

    return agent


def main():
    """Example usage of Veld - Agentic Memory with LlamaIndex."""

    print("=== Veld - Agentic Memory + LlamaIndex Integration ===\n")

    # Create memory block
    memory = VeldProactiveMemory(
        user_id="example-agent",
        max_memories=5,
    )

    # Example 1: Store some memories
    print("--- Storing memories ---")
    memory.put(ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Decided to use FastAPI for the backend API framework.",
    ))
    memory.put(ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Learned that the client prefers detailed error messages in API responses.",
    ))
    memory.put(ChatMessage(
        role=MessageRole.ASSISTANT,
        content="The project uses Python 3.11 with type hints throughout.",
    ))
    print("Stored 3 memories\n")

    # Example 2: Retrieve relevant memories
    print("--- Retrieving memories for 'API development' ---")
    relevant = memory.get("API development best practices")
    for msg in relevant:
        print(msg.content)

    # Example 3: Use with an agent
    print("\n--- Creating agent with persistent memory ---")
    try:
        agent = create_agent_with_veld_memory()
        print("Agent created successfully!")

        # The agent now has persistent memory across sessions
        # response = agent.chat("What do you remember about our API project?")
        # print(response)
    except Exception as e:
        print(f"Note: Full agent requires Anthropic API key. Error: {e}")


if __name__ == "__main__":
    main()
