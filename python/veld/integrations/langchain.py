"""
LangChain Integration for Veld

Provides a drop-in memory class that works with LangChain chains and agents.
Uses Hebbian learning for association strengthening - no LLM calls for memory ops.

Usage:
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI
    from shodh_memory.integrations.langchain import ShodhMemory

    memory = ShodhMemory(
        server_url="http://localhost:3030",
        user_id="user-123",
        api_key="your-api-key"
    )

    chain = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory
    )

    response = chain.invoke({"input": "Hello!"})
"""

from typing import Any, Dict, List, Optional
import os

try:
    from langchain_core.memory import BaseMemory
    from pydantic import Field, PrivateAttr
except ImportError:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install with: pip install shodh-memory[langchain] "
        "or: pip install langchain-core"
    )

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ShodhMemory(BaseMemory):
    """LangChain-compatible memory backed by Veld.

    Features:
    - Sub-millisecond retrieval (no LLM calls for memory ops)
    - Hebbian learning: associations strengthen through use
    - Persists across sessions
    - Supports semantic, associative, and hybrid retrieval

    Attributes:
        server_url: URL of the Veld server
        user_id: Unique identifier for this user
        api_key: API key for authentication
        memory_key: Key used in chain's memory dict (default: "shodh_context")
        input_key: Key for input in chain (default: "input")
        output_key: Key for output in chain (default: "output")
        return_messages: Whether to return as messages (default: False)
        max_memories: Maximum memories to retrieve per query (default: 5)
        retrieval_mode: One of "semantic", "associative", "hybrid" (default: "hybrid")
    """

    server_url: str = Field(default="http://localhost:3030")
    user_id: str = Field(default="default")
    api_key: Optional[str] = Field(default=None)
    memory_key: str = Field(default="shodh_context")
    input_key: str = Field(default="input")
    output_key: str = Field(default="output")
    return_messages: bool = Field(default=False)
    max_memories: int = Field(default=5)
    retrieval_mode: str = Field(default="hybrid")
    store_interactions: bool = Field(default=True)

    _session: Any = PrivateAttr(default=None)
    _headers: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get API key from param or environment
        if self.api_key is None:
            self.api_key = os.environ.get("SHODH_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key parameter or set SHODH_API_KEY env var."
            )

        # Setup HTTP session with retry logic
        self._session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables (keys in the returned dict)."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memories based on the input.

        Called by LangChain before invoking the chain.
        Retrieves memories semantically similar to the input.
        """
        # Extract query from input
        query = inputs.get(self.input_key, "")
        if not query:
            # Try other common keys
            query = inputs.get("question", inputs.get("query", ""))

        if not query:
            return {self.memory_key: ""}

        # Recall relevant memories
        try:
            response = self._session.post(
                f"{self.server_url}/api/recall",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "query": query,
                    "limit": self.max_memories,
                    "mode": self.retrieval_mode,
                },
                timeout=10,
            )
            response.raise_for_status()

            memories = response.json().get("memories", [])

            if not memories:
                return {self.memory_key: ""}

            # Format memories as context string
            context_parts = []
            for mem in memories:
                exp = mem.get("experience", {})
                content = exp.get("content", mem.get("content", ""))
                mem_type = exp.get("memory_type", "")
                score = mem.get("score", 0)

                if content:
                    if mem_type:
                        context_parts.append(f"[{mem_type}] {content}")
                    else:
                        context_parts.append(content)

            context = "\n".join(context_parts)

            if self.return_messages:
                # Return as message format for chat models
                from langchain_core.messages import SystemMessage
                return {self.memory_key: [SystemMessage(content=f"Relevant context:\n{context}")]}

            return {self.memory_key: context}

        except requests.exceptions.RequestException as e:
            # Log error but don't fail the chain
            import warnings
            warnings.warn(f"Veld memory retrieval failed: {e}")
            return {self.memory_key: ""}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the interaction to memory.

        Called by LangChain after the chain produces output.
        Stores the Q&A pair as a new memory.
        """
        if not self.store_interactions:
            return

        user_input = inputs.get(self.input_key, inputs.get("question", ""))
        ai_output = outputs.get(self.output_key, outputs.get("response", outputs.get("text", "")))

        if not user_input or not ai_output:
            return

        # Store as conversation memory
        content = f"User: {user_input}\nAssistant: {ai_output}"

        try:
            self._session.post(
                f"{self.server_url}/api/remember",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "content": content,
                    "memory_type": "Conversation",
                    "tags": ["langchain", "chat"],
                },
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            import warnings
            warnings.warn(f"Veld memory save failed: {e}")

    def clear(self) -> None:
        """Clear all memories for this user.

        Use with caution - this is irreversible.
        """
        try:
            self._session.delete(
                f"{self.server_url}/api/users/{self.user_id}",
                headers=self._headers,
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            import warnings
            warnings.warn(f"Veld memory clear failed: {e}")

    # Additional convenience methods

    def add_memory(
        self,
        content: str,
        memory_type: str = "Observation",
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Manually add a memory.

        Args:
            content: The content to remember
            memory_type: Type (Observation, Decision, Learning, Error, etc.)
            tags: Optional categorization tags

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            response = self._session.post(
                f"{self.server_url}/api/remember",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "content": content,
                    "memory_type": memory_type,
                    "tags": tags or [],
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("id")
        except requests.exceptions.RequestException:
            return None

    def search(
        self,
        query: str,
        limit: int = 5,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """Search memories directly.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            mode: Retrieval mode (semantic, associative, hybrid)

        Returns:
            List of matching memories
        """
        try:
            response = self._session.post(
                f"{self.server_url}/api/recall",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "query": query,
                    "limit": limit,
                    "mode": mode,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("memories", [])
        except requests.exceptions.RequestException:
            return []

    def get_context_summary(self, max_items: int = 5) -> Dict[str, Any]:
        """Get a summary of stored memories organized by type.

        Useful for session bootstrap or debugging.
        """
        try:
            response = self._session.post(
                f"{self.server_url}/api/context_summary",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "include_decisions": True,
                    "include_learnings": True,
                    "include_context": True,
                    "max_items": max_items,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}
