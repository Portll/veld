"""
CrewAI Integration for Veld

Makes a veld server the memory of a CrewAI crew or agent. CrewAI 1.x routes all
memory through its unified ``Memory`` class, and ``Crew(memory=...)`` and
``Agent(memory=...)`` accept any ``Memory`` instance. ``VeldCrewMemory`` is one
whose saves and recalls go to veld. Veld embeds, extracts entities and ranks on
the server, so the crew needs no embedder, no analysis LLM and no local vector
store for its memory.

Targets CrewAI 1.15 (``crewai.memory.unified_memory.Memory``).

Usage:
    from crewai import Agent, Crew, Task
    from veld.integrations.crewai import VeldCrewMemory

    memory = VeldCrewMemory(
        server_url="http://127.0.0.1:3030",
        user_id="research-crew",
        api_key="your-api-key",  # or set VELD_API_KEY
    )

    crew = Crew(agents=[researcher], tasks=[research_task], memory=memory)
    crew.kickoff()

    # Or use it directly
    memory.remember("The client prefers weekly reports", scope="/client/acme")
    matches = memory.recall("how often does the client want reports?")
"""

from __future__ import annotations

import os
import re
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit

try:
    from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo
    from crewai.memory.unified_memory import Memory
    from crewai.memory.utils import join_scope_paths, normalize_scope_path
    from pydantic import Field, PrivateAttr, field_validator
except ImportError:
    raise ImportError(
        "CrewAI is required for this integration. "
        "Install with: pip install veld[crewai] "
        "or: pip install crewai"
    )

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Every memory this adapter writes carries TAG. Scope, category, source and
# privacy ride in prefixed tags, because veld's remember route has no other
# field for them and tags are what its list, recall and forget routes filter on.
#
# The value after each prefix is percent-encoded (every character but letters,
# digits and "_.-~"). Veld redacts tags that look like credentials or home
# directories before storing them: a raw "crewai:scope:/home/team" matched its
# path rule (/(?:Users|home)/...) and came back as "redacted:path", so recall
# and reset by that scope missed the memory. Encoded, "/" is "%2F" and none of
# the redactor's shapes can match a value. It also keeps commas out of tags,
# which the list route would split its filter on.
TAG = "crewai"
_SCOPE_PREFIX = "crewai:scope:"
_CATEGORY_PREFIX = "crewai:category:"
_SOURCE_PREFIX = "crewai:source:"
_PRIVATE_TAG = "crewai:private"

_MEMORY_TYPES = {
    "observation", "decision", "learning", "error", "discovery", "pattern",
    "context", "task", "codeedit", "code_edit", "fileaccess", "file_access",
    "search", "command", "conversation", "intention",
}
_MAX_RECALL_LIMIT = 10_000
_BATCH_LIMIT = 1_000
_LIST_PAGE = 1_000
# Recall asks veld for this many times the requested limit, so that dropping
# private, suspected or out-of-category matches still leaves `limit` results.
_RECALL_OVERSAMPLE = 2
_FRACTION_PAST_MICROS = re.compile(r"(\.\d{6})\d+")
# Hosts where plain HTTP keeps the API key on this machine (the Vercel AI SDK
# client's isLocalHost list).
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _sends_key_in_clear(server_url: str) -> bool:
    """Whether requests to ``server_url`` carry the API key unencrypted off this machine."""
    parts = urlsplit(server_url)
    return parts.scheme.lower() == "http" and (parts.hostname or "") not in _LOCAL_HOSTS


class VeldRequestError(Exception):
    """A veld HTTP call failed: the server refused it or could not be reached.

    Attributes:
        status: HTTP status, or None when no response arrived.
        code: The server's machine-readable error code (e.g. ``INVALID_API_KEY``).
    """

    def __init__(self, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


class _VeldHTTP:
    """Session, headers and error mapping shared by the memory and its store."""

    def __init__(self, server_url: str, api_key: str, timeout: float):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
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
            "X-API-Key": api_key,
        }

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        try:
            response = self._session.request(
                method,
                f"{self.server_url}{path}",
                headers=self._headers,
                json=payload,
                params=params,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise VeldRequestError(f"veld {method} {path} failed: {e}") from e

        if not response.ok:
            code, message = _error_fields(response)
            raise VeldRequestError(
                f"veld {method} {path} returned {response.status_code} {code}: {message}",
                status=response.status_code,
                code=code,
            )
        try:
            return response.json()
        except ValueError as e:
            raise VeldRequestError(
                f"veld {method} {path} returned a body that is not JSON",
                status=response.status_code,
            ) from e

    def close(self) -> None:
        self._session.close()


def _error_fields(response: Any) -> Tuple[str, str]:
    """Read veld's ``{code, message}`` error body, tolerating plain-text bodies."""
    try:
        body = response.json()
    except ValueError:
        return "HTTP_ERROR", response.text or ""
    if isinstance(body, dict):
        return (
            str(body.get("code", "HTTP_ERROR")),
            str(body.get("message", body.get("error", ""))),
        )
    return "HTTP_ERROR", str(body)


def _utcnow() -> datetime:
    # CrewAI stores naive UTC datetimes; aware ones would fail its comparisons.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_time(value: Any) -> datetime:
    """Parse veld's RFC 3339 timestamps, which carry up to nanosecond precision."""
    if not isinstance(value, str):
        raise VeldRequestError(f"veld returned created_at {value!r}, which is not RFC 3339")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = _FRACTION_PAST_MICROS.sub(r"\1", text)
    try:
        return _naive_utc(datetime.fromisoformat(text))
    except ValueError as e:
        raise VeldRequestError(f"veld returned created_at {value!r}, which is not RFC 3339") from e


def _tag_value(value: str) -> str:
    return quote(value, safe="")


def _scope_tags(scope: Optional[str]) -> List[str]:
    """Tags for a scope and each of its ancestors, so an exact-match tag filter
    on any ancestor selects the whole subtree."""
    path = normalize_scope_path(scope or "/")
    if path == "/":
        return []
    parts = path.strip("/").split("/")
    return [_SCOPE_PREFIX + _tag_value("/" + "/".join(parts[: i + 1])) for i in range(len(parts))]


def _subtree_tag(scope_prefix: Optional[str]) -> str:
    """The one tag every adapter-written memory under ``scope_prefix`` carries."""
    tags = _scope_tags(scope_prefix)
    return tags[-1] if tags else TAG


def _record_tags(record: MemoryRecord) -> List[str]:
    tags = [TAG, *_scope_tags(record.scope)]
    tags.extend(_CATEGORY_PREFIX + _tag_value(c) for c in record.categories)
    if record.source:
        tags.append(_SOURCE_PREFIX + _tag_value(record.source))
    if record.private:
        tags.append(_PRIVATE_TAG)
    return tags


def _build_record(
    memory_id: Any,
    content: Any,
    tags: Any,
    importance: Any,
    created_at: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> MemoryRecord:
    """Rebuild a CrewAI record from a veld memory and the tags this adapter wrote.

    Tags veld derived itself (entities, keywords) are not categories and are left out.
    """
    scopes: List[str] = []
    categories: List[str] = []
    source: Optional[str] = None
    private = False
    for tag in tags if isinstance(tags, list) else []:
        if not isinstance(tag, str):
            continue
        if tag.startswith(_SCOPE_PREFIX):
            scopes.append(unquote(tag[len(_SCOPE_PREFIX):]))
        elif tag.startswith(_CATEGORY_PREFIX):
            categories.append(unquote(tag[len(_CATEGORY_PREFIX):]))
        elif tag.startswith(_SOURCE_PREFIX):
            source = unquote(tag[len(_SOURCE_PREFIX):])
        elif tag == _PRIVATE_TAG:
            private = True
    created = _parse_time(created_at)
    try:
        weight = min(1.0, max(0.0, float(importance)))
    except (TypeError, ValueError):
        weight = 0.5
    return MemoryRecord(
        id=str(memory_id),
        content=str(content or ""),
        scope=max(scopes, key=len) if scopes else "/",
        categories=categories,
        metadata=metadata or {},
        importance=weight,
        created_at=created,
        last_accessed=created,
        source=source,
        private=private,
    )


def _memories(body: Any) -> List[Dict[str, Any]]:
    if not isinstance(body, dict) or not isinstance(body.get("memories"), list):
        raise VeldRequestError("veld returned a response without a memories list")
    return [m for m in body["memories"] if isinstance(m, dict)]


class _VeldStore:
    """The record operations CrewAI's ``Memory`` delegates to its storage backend,
    carried out over veld's HTTP API.

    Ranking is not here. Veld embeds and ranks on the server, so
    ``VeldCrewMemory.recall`` sends the query text to ``/api/recall``; ``search``,
    which CrewAI calls with a vector from its own embedder, is refused.
    """

    def __init__(self, http: _VeldHTTP, user_id: str, memory_type: str, source_type: str):
        self._http = http
        self._user_id = user_id
        self._memory_type = memory_type
        self._source_type = source_type

    def _item(self, record: MemoryRecord) -> Dict[str, Any]:
        return {
            "content": record.content,
            "memory_type": self._memory_type,
            "tags": _record_tags(record),
            "source_type": self._source_type,
        }

    def save(self, records: List[MemoryRecord]) -> None:
        """Store records, replacing each record's id with the one veld assigned."""
        if len(records) == 1:
            body = self._http.request(
                "POST", "/api/remember", {"user_id": self._user_id, **self._item(records[0])}
            )
            if not isinstance(body, dict) or not body.get("success") or not body.get("id"):
                raise VeldRequestError(f"veld did not store the memory: {body!r}")
            records[0].id = str(body["id"])
            return
        for start in range(0, len(records), _BATCH_LIMIT):
            chunk = records[start:start + _BATCH_LIMIT]
            body = self._http.request(
                "POST",
                "/api/remember/batch",
                {"user_id": self._user_id, "memories": [self._item(r) for r in chunk]},
            )
            if not isinstance(body, dict):
                raise VeldRequestError(f"veld returned {body!r} for a batch remember")
            errors = body.get("errors") or []
            ids = body.get("memory_ids") or []
            if errors or len(ids) != len(chunk):
                detail = "; ".join(
                    f"item {start + int(e.get('index', 0))}: {e.get('error', '')}"
                    for e in errors
                    if isinstance(e, dict)
                )
                raise VeldRequestError(
                    f"veld stored {len(ids)} of {len(chunk)} memories"
                    + (f": {detail}" if detail else "")
                )
            for record, memory_id in zip(chunk, ids):
                record.id = str(memory_id)

    def search(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[tuple]:
        raise NotImplementedError(
            "veld ranks with its own embeddings and cannot search by a vector from "
            "CrewAI's embedder; VeldCrewMemory.recall sends the query text instead"
        )

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        try:
            body = self._http.request(
                "GET",
                f"/api/memory/{quote(record_id, safe='')}",
                params={"user_id": self._user_id, "decompress": "1"},
            )
        except VeldRequestError as e:
            if e.status == 404:
                return None
            raise
        experience = body.get("experience") if isinstance(body, dict) else None
        if not isinstance(experience, dict):
            raise VeldRequestError(f"veld returned memory {record_id} without its experience")
        return _build_record(
            body.get("id", record_id),
            experience.get("content", ""),
            experience.get("entities", experience.get("tags", [])),
            body.get("importance", 0.5),
            body.get("created_at"),
        )

    def _scan(self, scope_prefix: Optional[str]) -> List[Tuple[MemoryRecord, bool]]:
        """Every adapter-written record under ``scope_prefix``, each paired with
        whether the list route cut its content to a preview."""
        # An encoded scope tag holds no comma, so the list route, which splits its
        # tag filter on commas, can filter by it directly.
        wanted = _subtree_tag(scope_prefix)
        rows: List[Tuple[MemoryRecord, bool]] = []
        offset = 0
        while True:
            body = self._http.request(
                "POST",
                "/api/memories",
                {"user_id": self._user_id, "tags": wanted, "limit": _LIST_PAGE, "offset": offset},
            )
            items = _memories(body)
            for item in items:
                record = _build_record(
                    item.get("id"),
                    item.get("content", ""),
                    item.get("tags", []),
                    item.get("importance", 0.5),
                    item.get("created_at"),
                )
                rows.append((record, bool(item.get("content_truncated"))))
            offset += len(items)
            if not items or offset >= int(body.get("total", 0)):
                return rows

    def _records(self, scope_prefix: Optional[str]) -> List[MemoryRecord]:
        return [record for record, _ in self._scan(scope_prefix)]

    def _delete_one(self, record_id: str) -> bool:
        try:
            self._http.request(
                "DELETE",
                f"/api/memory/{quote(record_id, safe='')}",
                params={"user_id": self._user_id},
            )
        except VeldRequestError as e:
            if e.status == 404:
                return False
            raise
        return True

    def _forget_tag(self, tag: str) -> int:
        body = self._http.request(
            "POST", "/api/forget/tags", {"user_id": self._user_id, "tags": [tag]}
        )
        return int(body.get("deleted_count", 0)) if isinstance(body, dict) else 0

    def delete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Delete matching records and return how many went.

        Without ``record_ids``, only memories this adapter wrote are candidates.
        """
        if metadata_filter:
            raise ValueError(
                "veld does not store CrewAI metadata, so it cannot forget by metadata_filter"
            )
        if record_ids and not (categories or older_than):
            return sum(1 for record_id in record_ids if self._delete_one(record_id))
        if not (categories or older_than or record_ids):
            return self._forget_tag(_subtree_tag(scope_prefix))

        cutoff = _naive_utc(older_than) if older_than else None
        ids = set(record_ids or [])
        doomed = [
            r.id
            for r in self._records(scope_prefix)
            if (not categories or any(c in r.categories for c in categories))
            and (cutoff is None or r.created_at < cutoff)
            and (not ids or r.id in ids)
        ]
        return sum(1 for record_id in doomed if self._delete_one(record_id))

    def update(self, record: MemoryRecord) -> None:
        """Rewrite a record's content; veld re-embeds it on the server."""
        self._http.request(
            "PUT",
            f"/api/memory/{quote(record.id, safe='')}",
            {"user_id": self._user_id, "content": record.content},
        )

    def list_records(
        self, scope_prefix: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[MemoryRecord]:
        """Records under ``scope_prefix``, newest first, with their full content."""
        if limit <= 0 or offset < 0:
            return []
        rows = sorted(self._scan(scope_prefix), key=lambda row: row[0].created_at, reverse=True)
        page: List[MemoryRecord] = []
        for record, truncated in rows[offset:offset + limit]:
            full = self.get_record(record.id) if truncated else record
            if full is not None:
                page.append(full)
        return page

    def get_scope_info(self, scope: str) -> ScopeInfo:
        path = normalize_scope_path(scope)
        records = self._records(path)
        child_prefix = "/" if path == "/" else path + "/"
        children = set()
        categories = set()
        for record in records:
            categories.update(record.categories)
            if record.scope.startswith(child_prefix):
                first = record.scope[len(child_prefix):].split("/", 1)[0]
                if first:
                    children.add(child_prefix + first)
        created = [r.created_at for r in records]
        return ScopeInfo(
            path=path,
            record_count=len(records),
            categories=sorted(categories),
            oldest_record=min(created) if created else None,
            newest_record=max(created) if created else None,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> List[str]:
        return self.get_scope_info(parent).child_scopes

    def list_categories(self, scope_prefix: Optional[str] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self._records(scope_prefix):
            for category in record.categories:
                counts[category] = counts.get(category, 0) + 1
        return counts

    def count(self, scope_prefix: Optional[str] = None) -> int:
        return len(self._scan(scope_prefix))

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        """Delete every memory this adapter wrote under ``scope_prefix``.

        Memories other clients wrote under the same user_id are left alone.
        """
        self._forget_tag(_subtree_tag(scope_prefix))

    async def asave(self, records: List[MemoryRecord]) -> None:
        self.save(records)

    async def asearch(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[tuple]:
        return self.search(query_embedding, scope_prefix, categories, metadata_filter, limit, min_score)

    async def adelete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self.delete(scope_prefix, categories, record_ids, older_than, metadata_filter)

    def close(self) -> None:
        self._http.close()


class VeldCrewMemory(Memory):
    """CrewAI memory backed by a veld server.

    Pass it as ``Crew(memory=...)`` or ``Agent(memory=...)``. Agents recall from
    veld before each task and save their results after it; the crew's "Search
    memory" and "Save to memory" tools use it too.

    What differs from CrewAI's default memory:

    - No CrewAI LLM or embedder is called. Veld embeds, extracts entities and
      scores importance on the server. ``extract_with_llm=True`` restores CrewAI's
      LLM split of task results into separate statements before they are stored.
    - ``recall`` runs veld's retrieval pipeline once; ``depth`` is accepted and
      has no effect.
    - Scope, categories, source and the private flag are stored as tags prefixed
      ``crewai:``, their values percent-encoded so that veld's tag redactor
      leaves them intact (scope ``/home/team`` is ``crewai:scope:%2Fhome%2Fteam``).
      ``metadata`` and ``importance`` passed to ``remember`` are not
      stored: veld's remember route has no free-form field and scores
      importance itself.
    - ``recall`` searches the whole user_id, including memories other clients
      wrote; ``forget`` and ``reset`` only delete memories this adapter wrote.
    - Matches veld flags as suspected prompt injection are dropped from recall
      unless ``include_suspected=True``, and then carry
      ``metadata["injection_suspected"] = True``.

    Errors: a failed recall warns and returns no matches, so a task still runs
    without memory. ``remember`` raises ``VeldRequestError``. A failed background
    save from ``remember_many`` warns and is reported as CrewAI's
    ``MemorySaveFailedEvent``. ``forget``, ``reset``, ``update`` and the listing
    methods raise ``VeldRequestError``.

    Construction warns when ``server_url`` is plain ``http://`` to a host other
    than this machine, since the API key would travel in clear text; set
    VELD_ALLOW_HTTP=true to silence it.

    Attributes:
        server_url: URL of the veld server.
        user_id: Veld user the memories belong to.
        api_key: API key; falls back to the VELD_API_KEY environment variable.
        retrieval_mode: "semantic", "associative" or "hybrid".
        memory_type: Veld memory type for stored items (Observation, Decision,
            Learning, Error, Discovery, Pattern, Context, Task, Conversation, ...).
        source_type: Provenance veld records for stored items. Agent output is
            model-written, so the default is "ai_generated", which keeps veld's
            injection scan on for it.
        timeout: Seconds to wait for each veld request.
        extract_with_llm: Split task results with CrewAI's analysis LLM first.
        include_suspected: Return matches veld flags as suspected injection.
    """

    server_url: str = Field(default="http://127.0.0.1:3030", description="URL of the veld server.")
    user_id: str = Field(default="default", description="Veld user the memories belong to.")
    api_key: Optional[str] = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Veld API key; falls back to VELD_API_KEY.",
    )
    retrieval_mode: Literal["semantic", "associative", "hybrid"] = Field(
        default="hybrid", description="Veld retrieval mode."
    )
    memory_type: str = Field(default="Observation", description="Veld memory type for stored items.")
    source_type: str = Field(
        default="ai_generated", description="Provenance veld records for stored items."
    )
    timeout: float = Field(default=10.0, gt=0, description="Seconds to wait for each veld request.")
    extract_with_llm: bool = Field(
        default=False,
        description="Split task results into statements with CrewAI's analysis LLM before storing.",
    )
    include_suspected: bool = Field(
        default=False,
        description="Return matches veld flags as suspected prompt injection.",
    )
    storage: Any = Field(
        default=None,
        exclude=True,
        description="Always veld. Set by VeldCrewMemory; not a parameter.",
    )

    _http: Optional[_VeldHTTP] = PrivateAttr(default=None)

    @field_validator("storage", mode="before")
    @classmethod
    def _storage_is_veld(cls, value: Any) -> Any:
        if value is not None:
            raise ValueError("VeldCrewMemory stores memories in veld; storage cannot be set")
        return value

    @field_validator("memory_type")
    @classmethod
    def _known_memory_type(cls, value: str) -> str:
        if value.lower() not in _MEMORY_TYPES:
            raise ValueError(f"unknown veld memory type {value!r}")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("VELD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key parameter or set VELD_API_KEY env var."
            )
        self.server_url = self.server_url.rstrip("/")
        if _sends_key_in_clear(self.server_url) and os.environ.get("VELD_ALLOW_HTTP") != "true":
            warnings.warn(
                "Using HTTP for a non-localhost veld server sends the API key in clear text. "
                "Use an https:// URL, or set VELD_ALLOW_HTTP=true to suppress this warning."
            )
        self._http = _VeldHTTP(self.server_url, self.api_key, self.timeout)
        self.storage = _VeldStore(self._http, self.user_id, self.memory_type, self.source_type)
        super().model_post_init(__context)

    def _new_record(
        self,
        content: str,
        scope: Optional[str],
        categories: Optional[List[str]],
        importance: Optional[float],
        source: Optional[str],
        private: bool,
        root_scope: Optional[str],
    ) -> MemoryRecord:
        now = _utcnow()
        return MemoryRecord(
            content=content,
            scope=join_scope_paths(root_scope, scope),
            categories=list(categories or []),
            importance=importance if importance is not None else self.default_importance,
            created_at=now,
            last_accessed=now,
            source=source,
            private=private,
        )

    def _effective_scope(self, scope: Optional[str]) -> Optional[str]:
        if scope is None:
            return self.root_scope
        if self.root_scope:
            return join_scope_paths(self.root_scope, scope)
        return scope

    def remember(
        self,
        content: str,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        source: Optional[str] = None,
        private: bool = False,
        agent_role: Optional[str] = None,
        root_scope: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        """Store one item in veld and return its record, carrying veld's id.

        ``metadata`` and ``importance`` are not stored (see the class docstring).
        Returns None when the memory is read-only.

        Raises:
            VeldRequestError: veld refused the item or could not be reached.
        """
        if self.read_only:
            return None
        effective_root = root_scope if root_scope is not None else self.root_scope
        record = self._new_record(content, scope, categories, importance, source, private, effective_root)
        self._submit_save(self._storage.save, [record]).result()
        return record

    def remember_many(
        self,
        contents: List[str],
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        source: Optional[str] = None,
        private: bool = False,
        agent_role: Optional[str] = None,
        root_scope: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Queue items for one batch write to veld and return immediately.

        Like CrewAI's own memory, this returns an empty list; ``recall`` and
        ``drain_writes`` wait for queued writes to land.
        """
        if self.read_only:
            return []
        effective_root = root_scope if root_scope is not None else self.root_scope
        records = [
            self._new_record(c, scope, categories, importance, source, private, effective_root)
            for c in contents
            if c and c.strip()
        ]
        if records:
            self._submit_save(self._save_in_background, records)
        return []

    def _save_in_background(self, records: List[MemoryRecord]) -> None:
        try:
            self._storage.save(records)
        except VeldRequestError as e:
            warnings.warn(f"Veld memory save failed: {e}")
            raise

    def extract_memories(self, content: str) -> List[str]:
        """Return what to store from a task result.

        By default the content is stored whole: veld extracts entities and facts
        from it on the server, with no LLM call. With ``extract_with_llm=True``,
        CrewAI's analysis LLM splits it into statements first.
        """
        if self.extract_with_llm:
            return super().extract_memories(content)
        text = content.strip()
        return [text] if text else []

    def recall(
        self,
        query: str,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: int = 10,
        depth: Literal["shallow", "deep"] = "deep",
        source: Optional[str] = None,
        include_private: bool = False,
    ) -> List[MemoryMatch]:
        """Recall memories relevant to ``query`` with veld's retrieval pipeline.

        A scope narrows the search to memories this adapter wrote under it; with
        no scope and no ``root_scope`` the whole user_id is searched. ``depth`` is
        accepted for compatibility and has no effect. On failure, warns and
        returns an empty list.
        """
        self.drain_writes()
        if limit <= 0 or not query or not query.strip():
            return []

        payload: Dict[str, Any] = {
            "user_id": self.user_id,
            "query": query,
            "limit": min(limit * _RECALL_OVERSAMPLE, _MAX_RECALL_LIMIT),
            "mode": self.retrieval_mode,
        }
        scope_tags = _scope_tags(self._effective_scope(scope))
        if scope_tags:
            payload["tags"] = [scope_tags[-1]]

        try:
            items = _memories(self._http.request("POST", "/api/recall", payload))
            matches: List[MemoryMatch] = []
            for item in items:
                suspected = bool(item.get("injection_suspected"))
                if suspected and not self.include_suspected:
                    continue
                experience = item.get("experience") if isinstance(item.get("experience"), dict) else {}
                record = _build_record(
                    item.get("id"),
                    experience.get("content", ""),
                    experience.get("tags", []),
                    item.get("importance", 0.5),
                    item.get("created_at"),
                    {"injection_suspected": True} if suspected else None,
                )
                if categories and not any(c in record.categories for c in categories):
                    continue
                if record.private and not include_private and (source is None or record.source != source):
                    continue
                matches.append(
                    MemoryMatch(
                        record=record,
                        score=float(item.get("score", 0.0)),
                        match_reasons=[self.retrieval_mode],
                    )
                )
                if len(matches) >= limit:
                    break
            return matches
        except VeldRequestError as e:
            warnings.warn(f"Veld memory recall failed: {e}")
            return []

    def update(
        self,
        record_id: str,
        content: Optional[str] = None,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> MemoryRecord:
        """Replace a record's content; veld re-embeds it on the server.

        Veld's update route rewrites content only. To change scope, categories,
        metadata or importance, forget the record and remember it again.

        Raises:
            ValueError: the record does not exist, or a field other than content was given.
            VeldRequestError: veld refused the update or could not be reached.
        """
        if any(v is not None for v in (scope, categories, metadata, importance)):
            raise ValueError(
                "veld updates content only; forget the record and remember it again "
                "to change its scope, categories, metadata or importance"
            )
        existing = self._storage.get_record(record_id)
        if existing is None:
            raise ValueError(f"Record not found: {record_id}")
        if self.read_only or content is None:
            return existing
        updated = existing.model_copy(update={"content": content, "last_accessed": _utcnow()})
        self._storage.update(updated)
        return updated


__all__ = ["VeldCrewMemory", "VeldRequestError", "TAG"]
