"""
mem0 OSS as the retrieval system of locomo_qa_eval.py (--system mem0).

Ingest: one mem0 user per LoCoMo conversation (locomo_<conv-id>). Each session (or each run of
--mem0-batch-turns turns of it) is one Memory.add() with infer on, so mem0's own LLM extracts and
stores facts. Every turn is a "user" message "<speaker>: <text>", the same line veld stores, so
both speakers' facts are extracted and stay named; mem0 1.x extracts only from user messages, and
mem0's own LoCoMo run gave each speaker a separate user, which this harness does not. The chunk
opens with a system message giving the session's date and time, the only observation date
mem0 1.x can be given; 2.x also gets it as custom instructions, since its extraction prompt
otherwise grounds relative times on today. Each memory's created_at is the session time (plus one
second per turn of offset), passed as metadata.

Retrieval: Memory.search over the conversation's user, top-k. mem0 keeps no link from a fact to
the turns it came from, so evidence hit@k is not measured, and a fact is listed under the speaker
it names first (under the first speaker when it names neither; counted as unattributed).

Every mem0 LLM call carries reasoning_effort (default none): with reasoning on, LM Studio returns
structured output in the reasoning channel and leaves content empty. LM Studio also refuses
response_format json_object, which mem0 sends, so on --provider lmstudio each call gets the
json_schema of the reply mem0 parses (2.x extraction; 1.x facts, then memory actions): with a
bare object schema the model answers {} and mem0 stores nothing. Every LLM and embedding call is
counted; truncated, unparseable or empty extraction replies (mem0 drops those facts silently) and
replies that extract nothing are reported.

mem0 runs in its own venv (benchmarks/requirements-locomo-qa-mem0.txt). Telemetry and mem0's
remote notice fetch are off (MEM0_TELEMETRY=False), and MEM0_DIR is a scratch directory removed
with the vector store and history database when the run ends.
"""

import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta

STRINGS = {"type": "array", "items": {"type": "string"}}
V2_EXTRACTION = {"type": "object", "required": ["memory"], "properties": {"memory": {
    "type": "array", "items": {"type": "object", "required": ["id", "text", "attributed_to"],
                               "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                              "attributed_to": {"type": "string",
                                                                "enum": ["user", "assistant"]},
                                              "linked_memory_ids": STRINGS}}}}}
V1_FACTS = {"type": "object", "required": ["facts"], "properties": {"facts": STRINGS}}
V1_ACTIONS = {"type": "object", "required": ["memory"], "properties": {"memory": {
    "type": "array", "items": {"type": "object", "required": ["id", "text", "event"],
                               "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                              "event": {"type": "string", "enum": [
                                                  "ADD", "UPDATE", "DELETE", "NONE"]},
                                              "old_memory": {"type": "string"}}}}}}
ADD_ATTEMPTS = 4
ADD_BACKOFF_S = 15


def rfc3339(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def session_chunks(turns: list, start: datetime, batch_turns: int, content) -> list[dict]:
    """The add() calls for one session: messages, observation time and created_at per chunk."""
    size = batch_turns if batch_turns > 0 else max(len(turns), 1)
    chunks = []
    for offset in range(0, len(turns), size):
        when = start + timedelta(seconds=offset)
        observed = when.strftime("%-I:%M %p on %-d %B %Y")
        messages = [{"role": "system",
                     "content": f"These messages were exchanged at {observed}."}]
        messages += [{"role": "user", "content": content(t)} for t in turns[offset:offset + size]]
        chunks.append({
            "messages": messages,
            "created_at": rfc3339(when),
            "dia_ids": [t["dia_id"] for t in turns[offset:offset + size]],
            "instructions": (f"Observation Date of these messages: {when.date().isoformat()}. "
                             "Ground relative time references in them to that date, not to the "
                             "Current Date."),
        })
    return chunks


def attribute_speaker(text: str, speakers: tuple[str, str]) -> tuple[str, bool]:
    """(speaker, attributed): the speaker the text names first, else the first speaker."""
    found = []
    for speaker in speakers:
        match = re.search(rf"\b{re.escape(speaker)}\b", text, re.IGNORECASE)
        if match:
            found.append((match.start(), speaker))
    if not found:
        return speakers[0], False
    return min(found)[1], True


def major_version(version: str) -> int:
    return int(version.split(".")[0])


def lmstudio_format(version: str, messages: list) -> dict:
    """The json_schema response format for one mem0 LLM call. mem0 1.x extracts facts with a
    system prompt, then decides memory actions in a single user message."""
    if major_version(version) >= 2:
        name, schema = "mem0_extraction", V2_EXTRACTION
    elif messages and messages[0].get("role") == "system":
        name, schema = "mem0_facts", V1_FACTS
    else:
        name, schema = "mem0_actions", V1_ACTIONS
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


class CallLog:
    """Counts mem0's LLM and embedding calls; mem0 1.x calls from worker threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.llm = {"calls": 0, "errors": 0, "truncated": 0, "unparseable": 0, "empty": 0,
                    "nothing_extracted": 0, "input_tokens": 0, "output_tokens": 0,
                    "latency_s": 0.0, "last_errors": []}
        self.embed = {"calls": 0, "errors": 0, "inputs": 0, "latency_s": 0.0}

    def llm_reply(self, response, latency: float):
        choice = response.choices[0] if response.choices else None
        content = (choice.message.content or "") if choice else ""
        usage = response.usage
        with self.lock:
            s = self.llm
            s["calls"] += 1
            s["latency_s"] += latency
            s["input_tokens"] += (usage.prompt_tokens or 0) if usage else 0
            s["output_tokens"] += (usage.completion_tokens or 0) if usage else 0
            if choice is not None and choice.finish_reason == "length":
                s["truncated"] += 1
            if not content.strip():
                s["empty"] += 1
            else:
                try:
                    reply = json.loads(content)
                except json.JSONDecodeError:
                    s["unparseable"] += 1
                else:
                    if isinstance(reply, dict) and not (reply.get("memory") or reply.get("facts")):
                        s["nothing_extracted"] += 1

    def llm_error(self, error: Exception, latency: float):
        with self.lock:
            self.llm["calls"] += 1
            self.llm["errors"] += 1
            self.llm["latency_s"] += latency
            self.llm["last_errors"] = (self.llm["last_errors"]
                                       + [f"{type(error).__name__}: {str(error)[:200]}"])[-5:]

    def embedding(self, inputs: int, latency: float, failed: bool):
        with self.lock:
            self.embed["calls"] += 1
            self.embed["inputs"] += inputs
            self.embed["latency_s"] += latency
            self.embed["errors"] += failed

    def snapshot(self) -> dict:
        with self.lock:
            llm = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.llm.items()}
            embed = {k: (round(v, 1) if isinstance(v, float) else v)
                     for k, v in self.embed.items()}
        return {"llm": llm, "embedding": embed}


class _Completions:
    def __init__(self, inner, log: CallLog, extra_body: dict, formatter=None):
        self._inner = inner
        self._log = log
        self._extra = extra_body
        self._formatter = formatter

    def create(self, **params):
        params["extra_body"] = {**self._extra, **(params.get("extra_body") or {})}
        if self._formatter is not None and "response_format" in params:
            params["response_format"] = self._formatter(params.get("messages") or [])
        started = time.perf_counter()
        try:
            response = self._inner.create(**params)
        except Exception as e:
            self._log.llm_error(e, time.perf_counter() - started)
            raise
        self._log.llm_reply(response, time.perf_counter() - started)
        return response


class _Embeddings:
    def __init__(self, inner, log: CallLog):
        self._inner = inner
        self._log = log

    def create(self, **params):
        inputs = params.get("input")
        count = len(inputs) if isinstance(inputs, list) else 1
        started = time.perf_counter()
        try:
            response = self._inner.create(**params)
        except Exception:
            self._log.embedding(count, time.perf_counter() - started, True)
            raise
        self._log.embedding(count, time.perf_counter() - started, False)
        return response


class _Namespace:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class AccountingClient:
    """An OpenAI client whose chat completions and embeddings are counted, the rest passed on."""

    def __init__(self, client, log: CallLog, extra_body: dict, formatter=None):
        self._client = client
        self.chat = _Namespace(completions=_Completions(client.chat.completions, log, extra_body,
                                                        formatter))
        self.embeddings = _Embeddings(client.embeddings, log)

    def __getattr__(self, name):
        return getattr(self._client, name)


def installed_features() -> dict:
    """mem0 2.x search adds BM25 when fastembed imports and entity boosts when spaCy does."""
    features = {}
    for name, module in (("bm25_keyword_search", "fastembed"), ("entity_linking", "spacy")):
        try:
            importlib.metadata.version(module)
            features[name] = True
        except importlib.metadata.PackageNotFoundError:
            features[name] = False
    return features


class Mem0System:
    def __init__(self, *, provider: str, base_url: str, api_key: str | None, llm_model: str,
                 llm_effort: str | None, llm_max_tokens: int, embedder_model: str,
                 embedding_dims: int, batch_turns: int, memory=None, version: str | None = None,
                 sleep=time.sleep):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.llm_effort = llm_effort
        self.llm_max_tokens = llm_max_tokens
        self.embedder_model = embedder_model
        self.embedding_dims = embedding_dims
        self.batch_turns = batch_turns
        self.version = version or importlib.metadata.version("mem0ai")
        self.log = CallLog()
        self.sleep = sleep
        self.workdir = None
        self._memory = memory

    def config(self, workdir: str) -> dict:
        lmstudio = self.provider == "lmstudio"
        url_key = "lmstudio_base_url" if lmstudio else "openai_base_url"
        llm = {"model": self.llm_model, "temperature": 0, "max_tokens": self.llm_max_tokens,
               "api_key": self.api_key or "lm-studio", url_key: self.base_url}
        embedder = {"model": self.embedder_model, "embedding_dims": self.embedding_dims,
                    "api_key": self.api_key or "lm-studio", url_key: self.base_url}
        provider = "lmstudio" if lmstudio else "openai"
        return {
            "llm": {"provider": provider, "config": llm},
            "embedder": {"provider": provider, "config": embedder},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "locomo", "path": os.path.join(workdir, "qdrant"),
                "embedding_model_dims": self.embedding_dims, "on_disk": True}},
            "history_db_path": os.path.join(workdir, "history.db"),
        }

    def memory(self):
        if self._memory is None:
            self.workdir = tempfile.mkdtemp(prefix="veld-bench-mem0-")
            os.environ["MEM0_TELEMETRY"] = "False"
            os.environ["MEM0_DIR"] = os.path.join(self.workdir, "home")
            from mem0 import Memory

            memory = Memory.from_config(self.config(self.workdir))
            extra = {"reasoning_effort": self.llm_effort} if self.llm_effort else {}
            formatter = None
            if self.provider == "lmstudio":
                version = self.version
                formatter = lambda messages: lmstudio_format(version, messages)  # noqa: E731
            memory.llm.client = AccountingClient(memory.llm.client, self.log, extra, formatter)
            memory.embedding_model.client = AccountingClient(memory.embedding_model.client,
                                                             self.log, {})
            self._memory = memory
        return self._memory

    def describe(self) -> dict:
        return {
            "name": "mem0", "package": "mem0ai", "version": self.version,
            "source": "PyPI", "provider": self.provider, "base_url": self.base_url,
            "llm_model": self.llm_model, "llm_reasoning_effort": self.llm_effort,
            "llm_max_tokens": self.llm_max_tokens, "llm_temperature": 0,
            "embedder_model": self.embedder_model, "embedding_dims": self.embedding_dims,
            "vector_store": "qdrant (local, scratch)", "batch_turns": self.batch_turns,
            "infer": True, "features": installed_features(),
            "response_format": ("json_schema of mem0's reply" if self.provider == "lmstudio"
                                else "json_object (mem0 default)"),
            "observation_date": ("system message per chunk"
                                 + (" and custom instructions" if major_version(self.version) >= 2
                                    else "")),
        }

    def add(self, chunk: dict, user_id: str, session: int) -> dict:
        kwargs = {"user_id": user_id,
                  "metadata": {"created_at": chunk["created_at"], "locomo_session": session}}
        if major_version(self.version) >= 2:
            kwargs["prompt"] = chunk["instructions"]
        for attempt in range(ADD_ATTEMPTS):
            try:
                return self.memory().add(chunk["messages"], **kwargs)
            except Exception as e:
                if attempt == ADD_ATTEMPTS - 1:
                    raise RuntimeError(f"{user_id} session {session}: mem0 add failed "
                                       f"{ADD_ATTEMPTS} times; last: {type(e).__name__}: {e}")
                self.sleep(ADD_BACKOFF_S * 2 ** attempt)
        raise AssertionError("unreachable")

    def ingest(self, user_id: str, sessions: list, content, progress=lambda line: None) -> dict:
        """sessions: [(number, turns, start)] as locomo_qa_eval.conversation_sessions returns;
        progress gets one line per session, since a session's extraction can take minutes."""
        before = self.log.snapshot()
        started = time.perf_counter()
        stats = {"user_id": user_id, "sessions": 0, "turns": 0, "adds": 0, "events": {}}
        for number, turns, start in sessions:
            for chunk in session_chunks(turns, start, self.batch_turns, content):
                result = self.add(chunk, user_id, number)
                for item in (result or {}).get("results", []):
                    event = item.get("event", "ADD")
                    stats["events"][event] = stats["events"].get(event, 0) + 1
                stats["adds"] += 1
            stats["sessions"] += 1
            stats["turns"] += len(turns)
            progress(f"  {user_id}: session {number} ({stats['sessions']} of {len(sessions)}) "
                     f"added, {sum(stats['events'].values())} memory events so far, "
                     f"{round(time.perf_counter() - started)}s")
        stats["ingest_s"] = round(time.perf_counter() - started, 2)
        stats["stored_memories"] = self.count(user_id)
        after = self.log.snapshot()
        stats["calls"] = {kind: {k: (after[kind][k] - before[kind][k]
                                     if isinstance(after[kind][k], (int, float)) else after[kind][k])
                                 for k in after[kind]} for kind in after}
        return stats

    def count(self, user_id: str) -> int:
        memory = self.memory()
        if major_version(self.version) >= 2:
            found = memory.get_all(filters={"user_id": user_id}, top_k=100_000)
        else:
            found = memory.get_all(user_id=user_id, limit=100_000)
        return len(found.get("results", []) if isinstance(found, dict) else found)

    def search(self, user_id: str, query: str, k: int, speakers: tuple[str, str]) -> dict:
        memory = self.memory()
        started = time.perf_counter()
        if major_version(self.version) >= 2:
            found = memory.search(query, filters={"user_id": user_id}, top_k=k)
        else:
            found = memory.search(query, user_id=user_id, limit=k)
        search_ms = (time.perf_counter() - started) * 1000
        retrieved = []
        unattributed = 0
        for m in (found.get("results", []) if isinstance(found, dict) else found)[:k]:
            text = m.get("memory", "")
            speaker, attributed = attribute_speaker(text, speakers)
            unattributed += not attributed
            retrieved.append({"id": m.get("id"), "dia_ids": [], "speaker": speaker,
                              "attributed": attributed, "created_at": m.get("created_at"),
                              "score": m.get("score"), "content": text})
        return {"retrieved": retrieved, "retrieved_ids": [r["id"] for r in retrieved],
                "evidence_hit": None, "unattributed": unattributed,
                "recall_ms": round(search_ms, 2)}

    def close(self):
        memory, self._memory = self._memory, None
        if memory is not None:
            client = getattr(getattr(memory, "vector_store", None), "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
        workdir, self.workdir = self.workdir, None
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
