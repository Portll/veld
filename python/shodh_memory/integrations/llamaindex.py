"""
LlamaIndex Integration for Veld

Provides a memory class compatible with LlamaIndex chat engines and agents.
Uses Hebbian learning for association strengthening - no LLM calls for memory ops.

Usage:
    from llama_index.core.chat_engine import SimpleChatEngine
    from shodh_memory.integrations.llamaindex import ShodhLlamaMemory

    memory = ShodhLlamaMemory(
        server_url="http://localhost:3030",
        user_id="user-123",
        api_key="your-api-key"
    )

    # Use with chat engine
    chat_engine = SimpleChatEngine.from_defaults(memory=memory)
    response = chat_engine.chat("Hello!")

    # Or use directly
    memory.put("User prefers Python", memory_type="Decision")
    results = memory.get("programming preferences")
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class ShodhLlamaMemory:
    """LlamaIndex-compatible memory backed by Veld.

    This class implements a simple memory interface that can be used
    with LlamaIndex chat engines and agents. It provides:

    - Sub-millisecond retrieval (no LLM calls for memory ops)
    - Hebbian learning: associations strengthen through use
    - Persistent storage across sessions
    - Semantic, associative, and hybrid retrieval modes

    Attributes:
        server_url: URL of the Veld server
        user_id: Unique identifier for this user
        api_key: API key for authentication
        max_memories: Maximum memories to retrieve per query
        retrieval_mode: One of "semantic", "associative", "hybrid"
    """

    server_url: str = "http://localhost:3030"
    user_id: str = "default"
    api_key: Optional[str] = None
    max_memories: int = 5
    retrieval_mode: str = "hybrid"

    _session: Any = field(default=None, repr=False, init=False)
    _headers: Dict[str, str] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
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

    def get(
        self,
        query: str,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query.

        Args:
            query: Natural language search query
            limit: Maximum results (default: self.max_memories)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            List of memory dicts with content, type, score, etc.
        """
        try:
            response = self._session.post(
                f"{self.server_url}/api/recall",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "query": query,
                    "limit": limit or self.max_memories,
                    "mode": self.retrieval_mode,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("memories", [])
        except requests.exceptions.RequestException as e:
            import warnings
            warnings.warn(f"Veld memory get failed: {e}")
            return []

    def put(
        self,
        content: str,
        memory_type: str = "Observation",
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        """Store a memory.

        Args:
            content: The content to remember
            memory_type: Type (Observation, Decision, Learning, Error, etc.)
            tags: Optional categorization tags
            **kwargs: Additional arguments (ignored for compatibility)

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
        except requests.exceptions.RequestException as e:
            import warnings
            warnings.warn(f"Veld memory put failed: {e}")
            return None

    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all memories for this user.

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of all memories
        """
        try:
            response = self._session.post(
                f"{self.server_url}/api/memories",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "limit": limit,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("memories", [])
        except requests.exceptions.RequestException:
            return []

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: UUID of the memory to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            response = self._session.delete(
                f"{self.server_url}/api/memory/{memory_id}",
                headers=self._headers,
                params={"user_id": self.user_id},
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def reset(self) -> bool:
        """Clear all memories for this user.

        Returns:
            True if cleared, False otherwise
        """
        try:
            response = self._session.delete(
                f"{self.server_url}/api/users/{self.user_id}",
                headers=self._headers,
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def get_context(self, query: str) -> str:
        """Get formatted context string for a query.

        Convenience method that retrieves memories and formats them
        as a string suitable for injection into prompts.

        Args:
            query: Natural language query

        Returns:
            Formatted context string
        """
        memories = self.get(query)

        if not memories:
            return ""

        parts = []
        for mem in memories:
            exp = mem.get("experience", {})
            content = exp.get("content", mem.get("content", ""))
            mem_type = exp.get("memory_type", "")

            if content:
                if mem_type:
                    parts.append(f"[{mem_type}] {content}")
                else:
                    parts.append(content)

        return "\n".join(parts)

    def get_context_summary(self, max_items: int = 5) -> Dict[str, Any]:
        """Get a summary of stored memories organized by type.

        Useful for session bootstrap.

        Args:
            max_items: Maximum items per category

        Returns:
            Dict with categorized memories
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

    def surface_relevant(
        self,
        context: str,
        semantic_threshold: float = 0.65,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Proactively surface relevant memories based on context.

        Uses entity matching + semantic similarity for <30ms retrieval.

        Args:
            context: Current conversation context
            semantic_threshold: Minimum similarity score (0-1)
            max_results: Maximum memories to surface

        Returns:
            Dict with surfaced memories and metadata
        """
        try:
            response = self._session.post(
                f"{self.server_url}/api/relevant",
                headers=self._headers,
                json={
                    "user_id": self.user_id,
                    "context": context,
                    "config": {
                        "semantic_threshold": semantic_threshold,
                        "max_results": max_results,
                    },
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {"memories": [], "latency_ms": 0}


# Alias for compatibility with different LlamaIndex versions
ShodhMemory = ShodhLlamaMemory
