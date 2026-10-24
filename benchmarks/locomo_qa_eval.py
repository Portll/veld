#!/usr/bin/env python3
"""
LoCoMo end-to-end QA accuracy (J, LLM-judged) for veld.

Protocol:
  1. Dataset: snap-research LoCoMo locomo10.json, cached in ~/.cache/veld-bench/; its sha256 is
     recorded in the results.
  2. Server: a scratch `veld server` spawned from --binary on a free port, with a fresh temporary
     data dir and a random API key. Of ~/.config/veld/server.env only the embedder settings are
     copied, so retrieval embeds as production does; the storage backend, replication and admin
     keys are not. The child gets no VELD_/SHODH_ variable from the calling shell. The server is
     stopped and its temp dir removed as soon as retrieval is done, and on any exit, Ctrl-C
     included; its log is copied next to --output first.
  3. Ingest: one veld user per conversation (locomo_<conv-id>). One Conversation memory per turn,
     "<speaker>: <text>", plus " [shares an image: <blip_caption>]" when the turn has a caption;
     created_at is the session's date and time in UTC (as tests/recall/corpora/locomo.jsonl) plus
     one second per turn, tags = [speaker], one POST /api/remember/batch per session. Then wait
     until GET /api/embeddings/backlog reports nothing pending and GET /api/health/index reports
     the vector and lexical indexes covering every stored memory.
  4. Retrieval: POST /api/recall {user_id, query, limit: k} once per question; context lines are
     "[<created_at date>] <content>".
  5. Reader: Mem0's ANSWER_PROMPT. Judge: Mem0's ACCURACY_PROMPT (J). Both run on claude-opus-5-5
     (reader effort medium, judge effort low, thinking always on, server-side refusal fallbacks
     on). --runs repeats reader + judge over the same retrieved contexts. J is the share of judged
     answers labelled CORRECT; mean and sample sd are over runs.

Categories (LoCoMo code -> name, the mapping benchmarks/locomo_to_harness.py in shodh-memory used
to build tests/recall/locomo_cases.jsonl): 1 multi_hop, 2 temporal, 3 open_domain, 4 single_hop,
5 adversarial. Adversarial questions are excluded, as Mem0 does.

Comparability: claude-opus-5-5 is reader and judge here. Published LoCoMo J figures (Mem0 and the
systems its paper compares) use gpt-4o-mini for both, so these numbers are not comparable with
them; compare runs of this harness with each other.

Setup (once):
    /opt/homebrew/bin/python3.13 -m venv ~/.venvs/veld-bench
    ~/.venvs/veld-bench/bin/pip install -r benchmarks/requirements-longmemeval.txt
Credentials: anthropic.Anthropic() reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
`ant auth login` profile.

Usage (from the repository root):
    # Dry run: spawn, ingest, retrieve, print assembled prompts and a token/cost projection.
    # Makes no API calls.
    ~/.venvs/veld-bench/bin/python benchmarks/locomo_qa_eval.py --dry-run \\
        --conversations conv-26 --limit-questions 5

    # Pilot: one conversation, one run
    ~/.venvs/veld-bench/bin/python benchmarks/locomo_qa_eval.py \\
        --conversations conv-26 --runs 1 \\
        --output ~/.cache/veld-bench/results/locomo_qa_pilot_conv-26.json

    # Full: all ten conversations, three runs
    ~/.venvs/veld-bench/bin/python benchmarks/locomo_qa_eval.py --runs 3 \\
        --output ~/.cache/veld-bench/results/locomo_qa_full.json

    # After a crash or Ctrl-C: the same command plus --resume. Stored retrievals are reused and
    # only unjudged (question, run) pairs are sent again.

    # Against an already-running scratch server instead of spawning one:
    ... --no-spawn --url http://127.0.0.1:4100 --api-key <key>
"""

import argparse
import atexit
import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CACHE_DIR = Path.home() / ".cache" / "veld-bench"
SERVER_ENV_PATH = Path.home() / ".config" / "veld" / "server.env"
DEFAULT_BINARY = Path.home() / ".local" / "bin" / "veld"
LIVE_PORTS = {3030, 3031}

MODEL = "claude-opus-5-5"
READER_EFFORT = "medium"
JUDGE_EFFORT = "low"
READER_MAX_TOKENS = 8000
JUDGE_MAX_TOKENS = 4000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
PRICE_PER_MTOK = {"input_tokens": 4.00, "output_tokens": 20.00, "cache_read_input_tokens": 0.20}

CHARS_PER_TOKEN = 3.5
EST_READER_OUTPUT_TOKENS = 1500
EST_JUDGE_OUTPUT_TOKENS = 300
EST_ANSWER_CHARS = 400

CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
EXCLUDED_CATEGORIES = {5}
CATEGORY_ORDER = ["single_hop", "multi_hop", "temporal", "open_domain"]

EMBEDDER_ENV_NAMES = {
    "VELD_EMBEDDER",
    "VELD_SECONDARY_EMBEDDER",
    "VELD_LIVE_EMBEDDER",
    "VELD_EMBED_TIMEOUT_MS",
    "VELD_NOMIC_EMBED_TIMEOUT_MS",
    "VELD_ALLOW_INSECURE_REMOTE_EMBEDDER",
    "VELD_ALLOW_SIMPLIFIED_EMBEDDINGS",
}
EMBEDDER_ENV_PREFIX = "VELD_EMBEDDING_"

COMPARABILITY_NOTE = (
    f"Not comparable with published LoCoMo J figures: this run uses {MODEL} as both reader and "
    f"judge (reader effort {READER_EFFORT}, judge effort {JUDGE_EFFORT}), while every published "
    "J figure (Mem0 and the systems its paper compares) uses gpt-4o-mini for both. Compare only "
    "runs of this harness made with the same model settings."
)

MEM0_COMMIT = "b3ede5b7c0ac0e847b03786a603c107ac943b3ee"
ANSWER_PROMPT_SOURCE = f"https://github.com/mem0ai/mem0/blob/{MEM0_COMMIT}/evaluation/prompts.py"
ACCURACY_PROMPT_SOURCE = (
    f"https://github.com/mem0ai/mem0/blob/{MEM0_COMMIT}/evaluation/metrics/llm_judge.py"
)

# Verbatim from https://github.com/mem0ai/mem0/blob/b3ede5b7c0ac0e847b03786a603c107ac943b3ee/evaluation/prompts.py
# (Mem0 retired evaluation/ from main on 2026-06-13; this is the last commit that carried it.)
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """

# Verbatim from https://github.com/mem0ai/mem0/blob/b3ede5b7c0ac0e847b03786a603c107ac943b3ee/evaluation/metrics/llm_judge.py
ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["CORRECT", "WRONG"]}},
    "required": ["label"],
    "additionalProperties": False,
}

SESSION_KEY = re.compile(r"^session_(\d+)$")
DATETIME_FORMATS = ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y")
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locomo(cache_dir: Path) -> tuple[list, Path, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "locomo10.json"
    if not path.exists():
        print(f"Downloading {DATASET_URL}")
        resp = requests.get(DATASET_URL, timeout=120)
        resp.raise_for_status()
        part = path.with_suffix(".json.part")
        part.write_bytes(resp.content)
        os.replace(part, path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, path, sha256_file(path)


def parse_session_datetime(raw: str) -> datetime:
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unparseable LoCoMo session datetime: {raw!r}")


def rfc3339(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def turn_content(turn: dict) -> str:
    content = f"{turn['speaker']}: {turn.get('text', '')}".rstrip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        content += f" [shares an image: {caption}]"
    return content


def conversation_sessions(conv: dict) -> list[tuple[int, list, datetime]]:
    c = conv["conversation"]
    sessions = []
    for key, turns in c.items():
        match = SESSION_KEY.match(key)
        if match and isinstance(turns, list):
            start = parse_session_datetime(c[f"{key}_date_time"])
            sessions.append((int(match.group(1)), turns, start))
    return sorted(sessions, key=lambda s: s[0])


def session_batches(conv: dict) -> list[tuple[int, list[tuple[str, dict]]]]:
    batches = []
    for number, turns, start in conversation_sessions(conv):
        items = []
        for offset, turn in enumerate(turns):
            items.append((turn["dia_id"], {
                "content": turn_content(turn),
                "memory_type": "Conversation",
                "tags": [turn["speaker"]],
                "created_at": rfc3339(start + timedelta(seconds=offset)),
            }))
        batches.append((number, items))
    return batches


def conversation_questions(conv: dict) -> list[dict]:
    questions = []
    for index, qa in enumerate(conv["qa"]):
        code = qa.get("category")
        if code in EXCLUDED_CATEGORIES:
            continue
        questions.append({
            "qid": f"{conv['sample_id']}_q{index}",
            "conversation": conv["sample_id"],
            "qa_index": index,
            "category_code": code,
            "category": CATEGORY[code],
            "question": qa["question"],
            "gold": str(qa["answer"]),
            "evidence": [e for e in qa.get("evidence") or [] if isinstance(e, str)],
        })
    return questions


def user_for(conv_id: str) -> str:
    return f"locomo_{conv_id}"


def read_env_file(path: Path) -> dict:
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    return values


def is_embedder_setting(name: str) -> bool:
    return name in EMBEDDER_ENV_NAMES or name.startswith(EMBEDDER_ENV_PREFIX)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ScratchServer:
    def __init__(self, binary, port=None, server_env=SERVER_ENV_PATH,
                 storage_backend="rocksdb", startup_timeout=180, extra_env=()):
        self.binary = str(Path(binary).expanduser())
        self.port = port or free_port()
        if self.port in LIVE_PORTS:
            raise SystemExit(f"Port {self.port} belongs to the live veld server or its replica; "
                             "pick another --port.")
        refused = [n for n in extra_env if n.startswith(("VELD_REPLICATE_", "VELD_ADMIN_"))]
        if refused:
            raise SystemExit(f"Refusing to copy replication or admin settings: {refused}")
        self.extra_env = set(extra_env)
        self.server_env = Path(server_env).expanduser()
        self.storage_backend = storage_backend
        self.startup_timeout = startup_timeout
        self.api_key = f"sk-veld-scratch-{secrets.token_hex(16)}"
        self.url = f"http://127.0.0.1:{self.port}"
        self.tmpdir = None
        self.proc = None
        self.copied_env = []
        self.version = None
        self.health = None

    def environment(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VELD_", "SHODH_"))}
        self.copied_env = []
        if self.server_env.exists():
            for key, value in read_env_file(self.server_env).items():
                if is_embedder_setting(key) or key in self.extra_env:
                    env[key] = value
                    self.copied_env.append(key)
        # veld bridges the live config.toml into any of these left unset, which would hand
        # the scratch server the live API key, port or store.
        env.update({
            "VELD_HOST": "127.0.0.1",
            "VELD_PORT": str(self.port),
            "VELD_MEMORY_PATH": str(Path(self.tmpdir) / "data"),
            "VELD_STORAGE_BACKEND": self.storage_backend,
            "VELD_API_KEYS": self.api_key,
            "VELD_API_KEY": self.api_key,
            "VELD_INSTANCE_LABEL": f"bench-scratch-{self.port}",
        })
        return env

    def start(self):
        if not os.access(self.binary, os.X_OK):
            raise SystemExit(f"veld binary not found or not executable: {self.binary}")
        self.tmpdir = tempfile.mkdtemp(prefix="veld-bench-scratch-")
        atexit.register(self.stop)
        env = self.environment()
        self.log_path = Path(self.tmpdir) / "server.log"
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen([self.binary, "server"], stdout=log,
                                         stderr=subprocess.STDOUT, env=env, cwd=self.tmpdir,
                                         start_new_session=True)
        print(f"Spawned scratch veld pid={self.proc.pid} at {self.url}, data in {self.tmpdir}")
        print(f"  settings copied from {self.server_env}: "
              f"{', '.join(sorted(self.copied_env)) or 'none'}")
        self.wait_healthy()
        self.version = f"{self.health.get('version')} (build {self.health.get('build')})"
        print(f"  veld {self.version}, backend {self.health.get('effective_storage_backend')}")
        return self

    def log_tail(self, lines=40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def wait_healthy(self):
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            code = self.proc.poll()
            if code is not None:
                tail = self.log_tail()
                self.stop()
                raise SystemExit(f"Scratch veld exited with code {code}.\n{tail}")
            try:
                resp = requests.get(f"{self.url}/health", timeout=2)
                if resp.status_code == 200:
                    self.health = resp.json()
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)
        tail = self.log_tail()
        self.stop()
        raise SystemExit(f"Scratch veld not healthy within {self.startup_timeout}s.\n{tail}")

    def stop(self, keep_log=None):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        if keep_log and self.tmpdir and self.log_path.exists():
            Path(keep_log).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.log_path, keep_log)
        tmpdir, self.tmpdir = self.tmpdir, None
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"Stopped scratch veld at {self.url}; removed {tmpdir}")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class VeldClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "X-API-Key": api_key})

    def health(self) -> dict:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def remember_batch(self, user_id: str, memories: list[dict]) -> dict:
        resp = self.session.post(f"{self.base_url}/api/remember/batch",
                                 json={"user_id": user_id, "memories": memories}, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def recall(self, user_id: str, query: str, limit: int) -> list[dict]:
        resp = self.session.post(f"{self.base_url}/api/recall",
                                 json={"user_id": user_id, "query": query, "limit": limit},
                                 timeout=60)
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def optional_get(self, path: str, params: dict):
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def index_coverage(self, user_id: str):
        body = self.optional_get("/api/health/index", {"user_id": user_id})
        return None if body is None else body.get("index_coverage")

    def embedding_backlog(self, user_id: str):
        return self.optional_get("/api/embeddings/backlog", {"user_id": user_id})


def ingest_conversation(client: VeldClient, conv: dict) -> tuple[dict, dict]:
    user_id = user_for(conv["sample_id"])
    stats = {"user_id": user_id, "sessions": 0, "turns": 0, "created": 0, "failed": 0,
             "errors": []}
    memory_turns = {}
    started = time.perf_counter()
    for number, items in session_batches(conv):
        result = client.remember_batch(user_id, [memory for _, memory in items])
        errors = result.get("errors", [])
        failed = {e["index"] for e in errors}
        stored = [(dia_id, memory) for i, (dia_id, memory) in enumerate(items) if i not in failed]
        ids = result.get("memory_ids", [])
        if len(stored) != len(ids):
            raise RuntimeError(f"{user_id} session {number}: {len(ids)} ids for "
                               f"{len(stored)} accepted turns")
        for (dia_id, memory), memory_id in zip(stored, ids):
            entry = memory_turns.setdefault(memory_id,
                                            {"dia_ids": [], "speaker": memory["tags"][0]})
            entry["dia_ids"].append(dia_id)
        for e in errors:
            stats["errors"].append({"session": number, "dia_id": items[e["index"]][0],
                                    "error": e["error"]})
        stats["sessions"] += 1
        stats["turns"] += len(items)
        stats["created"] += result.get("created", len(ids))
        stats["failed"] += len(errors)
    stats["unique_memories"] = len(memory_turns)
    stats["ingest_s"] = round(time.perf_counter() - started, 2)
    return stats, memory_turns


def wait_for_index(client: VeldClient, user_id: str, expected: int, timeout_s: float) -> dict:
    started = time.perf_counter()
    polls = 0
    while True:
        polls += 1
        backlog = client.embedding_backlog(user_id)
        coverage = client.index_coverage(user_id)
        if backlog is None and coverage is None:
            raise RuntimeError("the server exposes neither /api/embeddings/backlog nor "
                               "/api/health/index; cannot tell when ingest is indexed")
        pending = backlog.get("pending", 0) if backlog else 0
        # The lexical count is reported but not waited on: the BM25 projection commits its
        # last partial batch at the next write or search, so an idle store stays short until
        # the first recall. retrieval_phase reads coverage again after retrieving.
        covered = coverage is None or (
            coverage["stored"] >= expected
            and coverage["vector_indexed"] >= coverage["stored"])
        waited = time.perf_counter() - started
        if pending == 0 and covered:
            return {"waited_s": round(waited, 2), "polls": polls, "pending": pending,
                    "coverage": coverage, "backlog": backlog}
        if waited > timeout_s:
            raise RuntimeError(f"{user_id}: index not complete after {timeout_s:.0f}s "
                               f"(pending={pending}, coverage={coverage})")
        time.sleep(2.0)


def memory_date(created_at) -> str:
    try:
        return datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return str(created_at)[:10]


def retrieve(client: VeldClient, question: dict, k: int, memory_turns: dict) -> dict:
    started = time.perf_counter()
    memories = client.recall(user_for(question["conversation"]), question["question"], k)
    recall_ms = (time.perf_counter() - started) * 1000
    retrieved = []
    for m in memories:
        turn = memory_turns.get(m.get("id"), {})
        content = (m.get("experience") or {}).get("content", m.get("content", ""))
        retrieved.append({
            "id": m.get("id"),
            "dia_ids": turn.get("dia_ids", []),
            "speaker": turn.get("speaker") or content.partition(":")[0],
            "created_at": m.get("created_at"),
            "score": m.get("score"),
            "content": content,
        })
    retrieved_dia = {d for item in retrieved for d in item["dia_ids"]}
    return {
        "retrieved": retrieved,
        "retrieved_ids": [item["id"] for item in retrieved],
        "evidence_hit": bool(retrieved_dia & set(question["evidence"]))
        if question["evidence"] else None,
        "recall_ms": round(recall_ms, 2),
    }


def reader_prompt(row: dict, speakers: tuple[str, str]) -> str:
    first, second = speakers
    lines = {first: [], second: []}
    for item in row["retrieved"]:
        if item["speaker"] not in lines:
            raise ValueError(f"{row['qid']}: retrieved memory {item['id']} has speaker "
                             f"{item['speaker']!r}, not {first} or {second}")
        lines[item["speaker"]].append(f"[{memory_date(item['created_at'])}] {item['content']}")
    values = {
        "speaker_1_user_id": first,
        "speaker_2_user_id": second,
        "speaker_1_memories": json.dumps(lines[first], indent=4),
        "speaker_2_memories": json.dumps(lines[second], indent=4),
        "question": row["question"],
    }
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], ANSWER_PROMPT)


def judge_prompt(row: dict, answer: str) -> str:
    return ACCURACY_PROMPT.format(question=row["question"], gold_answer=row["gold"],
                                  generated_answer=answer)


def make_client():
    import anthropic

    return anthropic.Anthropic(max_retries=5)


def call_claude(client, prompt: str, effort: str, max_tokens: int, schema=None,
                model: str = MODEL) -> tuple[dict, str | None]:
    import anthropic

    output_config = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    record = {"model": model, "served_model": None, "request_id": None, "stop_reason": None,
              "fallback": False, "input_tokens": 0, "output_tokens": 0,
              "cache_read_input_tokens": 0, "latency_ms": None, "error": None, "fatal": False}
    text = None
    started = time.perf_counter()
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.RateLimitError as e:
        record.update(error=f"HTTP 429 after retries: {e.message}", request_id=e.request_id)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
            anthropic.NotFoundError, anthropic.BadRequestError) as e:
        record.update(error=f"HTTP {e.status_code}: {e.message}", request_id=e.request_id,
                      fatal=True)
    except anthropic.APIStatusError as e:
        record.update(error=f"HTTP {e.status_code} after retries: {e.message}",
                      request_id=e.request_id)
    except anthropic.APIConnectionError as e:
        record.update(error=f"connection error after retries: {e}")
    else:
        usage = response.usage
        record.update(
            request_id=response._request_id,
            served_model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            fallback=any(e.type == "fallback_message" for e in (usage.iterations or [])),
        )
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            record["error"] = f"refusal (category {category})"
        elif response.stop_reason == "max_tokens":
            record["error"] = f"max_tokens: output truncated at {max_tokens} tokens"
        elif response.stop_reason not in ("end_turn", "stop_sequence"):
            record["error"] = f"unexpected stop_reason {response.stop_reason}"
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if not text:
                record["error"] = "response carried no text block"
                text = None
    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return record, text


def call_cost(record: dict) -> float:
    return sum(record.get(key, 0) * price for key, price in PRICE_PER_MTOK.items()) / 1e6


def run_unit(client, row: dict, run: dict, prompt: str,
             abort: threading.Event) -> tuple[dict, bool]:
    if abort.is_set():
        return run, False
    if run["answer"] is None:
        record, text = call_claude(client, prompt, READER_EFFORT, READER_MAX_TOKENS)
        run["calls"].append({"kind": "reader", **record})
        if record["error"]:
            return run, record["fatal"]
        run["answer"] = text
        run["answer_fallback"] = record["fallback"]
    record, text = call_claude(client, judge_prompt(row, run["answer"]), JUDGE_EFFORT,
                               JUDGE_MAX_TOKENS, schema=JUDGE_SCHEMA)
    call = {"kind": "judge", **record}
    run["calls"].append(call)
    if record["error"]:
        return run, record["fatal"]
    try:
        label = json.loads(text)["label"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        call["error"] = f"judge output is not the label schema: {e}: {text[:200]!r}"
        return run, False
    if label not in ("CORRECT", "WRONG"):
        call["error"] = f"judge label {label!r} is neither CORRECT nor WRONG"
        return run, False
    run["label"] = label
    run["judge_fallback"] = record["fallback"]
    return run, False


def percentile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(results: dict, qids: list[str], runs: int) -> dict:
    rows = [results["rows"][q] for q in qids]
    categories = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in rows)]
    keys = categories + ["overall"]
    n = {c: sum(1 for r in rows if r["category"] == c) for c in categories}
    n["overall"] = len(rows)

    per_run = []
    for r in range(runs):
        tally = {c: [0, 0] for c in keys}
        unjudged = 0
        for row in rows:
            runs_done = row.get("runs", [])
            label = runs_done[r]["label"] if r < len(runs_done) else None
            if label is None:
                unjudged += 1
                continue
            for c in (row["category"], "overall"):
                tally[c][1] += 1
                tally[c][0] += label == "CORRECT"
        per_run.append({
            "j": {c: (t[0] / t[1] if t[1] else None) for c, t in tally.items()},
            "judged": {c: t[1] for c, t in tally.items()},
            "unjudged": unjudged,
        })

    j = {}
    for c in keys:
        values = [pr["j"][c] for pr in per_run if pr["j"][c] is not None]
        j[c] = {
            "n": n[c],
            "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "per_run": [pr["j"][c] for pr in per_run],
            "judged_per_run": [pr["judged"][c] for pr in per_run],
        }

    evidence = {}
    for c in keys:
        hits = [r["evidence_hit"] for r in rows
                if r.get("evidence_hit") is not None and c in (r["category"], "overall")]
        evidence[c] = sum(hits) / len(hits) if hits else None

    latencies = [r["recall_ms"] for r in rows if r.get("recall_ms") is not None]
    calls = [call for row in rows for run in row.get("runs", []) for call in run["calls"]]
    totals = {key: sum(call.get(key, 0) for call in calls) for key in PRICE_PER_MTOK}
    by_kind = {}
    for kind in ("reader", "judge"):
        mine = [call for call in calls if call["kind"] == kind]
        by_kind[kind] = {
            "calls": len(mine),
            "errors": sum(1 for call in mine if call["error"]),
            "refusals": sum(1 for call in mine if call["stop_reason"] == "refusal"),
            "truncated": sum(1 for call in mine if call["stop_reason"] == "max_tokens"),
            "fallback_served": sum(1 for call in mine if call["fallback"]),
        }
    answers_by_fallback = sum(1 for row in rows for run in row.get("runs", [])
                              if run.get("label") and run.get("answer_fallback"))
    return {
        "categories": keys,
        "j": j,
        "unjudged_per_run": [pr["unjudged"] for pr in per_run],
        "evidence_hit_at_k": evidence,
        "recall_latency_ms": {"p50": percentile(latencies, 0.50),
                              "p95": percentile(latencies, 0.95), "n": len(latencies)},
        "calls": by_kind,
        "judged_answers_served_by_fallback": answers_by_fallback,
        "tokens": totals,
        "estimated_cost_usd": round(sum(call_cost(call) for call in calls), 4),
    }


def fmt(value, pattern="{:.3f}"):
    return "-" if value is None else pattern.format(value)


def print_summary(results: dict, agg: dict):
    cfg = results["config"]
    print("\n## LoCoMo QA accuracy (J)\n")
    print(f"veld {results['veld'].get('binary_version')} (repo {results['veld'].get('git')}), "
          f"top-k {cfg['top_k']}, reader and judge {cfg['model']} "
          f"(effort {cfg['reader_effort']} / {cfg['judge_effort']}), runs {cfg['runs']}\n")
    print(f"> {COMPARABILITY_NOTE}\n")
    print("| Category | n | J mean | J sd | J per run | evidence hit@k |")
    print("|---|---:|---:|---:|---|---:|")
    for c in agg["categories"]:
        row = agg["j"][c]
        per_run = ", ".join(fmt(v) for v in row["per_run"])
        print(f"| {c} | {row['n']} | {fmt(row['mean'])} | {fmt(row['sd'])} | {per_run} | "
              f"{fmt(agg['evidence_hit_at_k'][c])} |")
    lat = agg["recall_latency_ms"]
    print(f"\nRecall latency: p50 {fmt(lat['p50'], '{:.1f}')} ms, "
          f"p95 {fmt(lat['p95'], '{:.1f}')} ms (n={lat['n']})")
    for kind, c in agg["calls"].items():
        print(f"{kind.capitalize()} calls: {c['calls']}, errors {c['errors']} "
              f"(refusals {c['refusals']}, truncated {c['truncated']}), "
              f"served by a fallback model {c['fallback_served']}")
    print(f"Judged answers written by a fallback model, not {MODEL}: "
          f"{agg['judged_answers_served_by_fallback']}")
    if any(agg["unjudged_per_run"]):
        print(f"Unjudged questions per run: {agg['unjudged_per_run']} (excluded from J; "
              "rerun with --resume to retry them)")
    t = agg["tokens"]
    print(f"Tokens: input {t['input_tokens']:,}, output {t['output_tokens']:,}, "
          f"cache read {t['cache_read_input_tokens']:,}; estimated cost "
          f"${agg['estimated_cost_usd']:.2f} at $4 / $20 / $0.20 per MTok")


def save_results(results: dict, path: Path):
    results["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def git_describe() -> str:
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.check_output(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(here), "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def refuse_live_url(url: str):
    parsed = urlparse(url)
    if parsed.hostname in ("127.0.0.1", "localhost", "::1") and parsed.port in LIVE_PORTS:
        raise SystemExit(f"{url} is the live veld server or its replica; point --url at a "
                         "scratch server.")


def project(rows: list[dict], speakers: dict, runs: int, full_questions: int):
    reader_chars = [len(reader_prompt(r, speakers[r["conversation"]])) for r in rows]
    judge_chars = [len(judge_prompt(r, "x" * EST_ANSWER_CHARS)) for r in rows]
    reader_in = statistics.mean(reader_chars) / CHARS_PER_TOKEN
    judge_in = statistics.mean(judge_chars) / CHARS_PER_TOKEN
    per_question = {
        "input_tokens": reader_in + judge_in,
        "output_tokens": EST_READER_OUTPUT_TOKENS + EST_JUDGE_OUTPUT_TOKENS,
    }
    cost_per_question = call_cost(per_question)
    print("\n## Projection (no API calls made)\n")
    print(f"Mean reader prompt {statistics.mean(reader_chars):,.0f} chars "
          f"(~{reader_in:,.0f} tokens), judge prompt {statistics.mean(judge_chars):,.0f} chars "
          f"(~{judge_in:,.0f} tokens) at {CHARS_PER_TOKEN} chars/token, over {len(rows)} "
          "questions.")
    print(f"Assumed, not measured: reader output {EST_READER_OUTPUT_TOKENS} tokens and judge "
          f"output {EST_JUDGE_OUTPUT_TOKENS} tokens per question-run, thinking included; the "
          f"judge sees a {EST_ANSWER_CHARS}-char reader answer.")
    for label, count in (("selected", len(rows)), ("full LoCoMo", full_questions)):
        calls = count * runs
        input_cost = call_cost({"input_tokens": per_question["input_tokens"] * calls})
        print(f"- {label}: {count} questions x {runs} runs = {2 * calls} calls, "
              f"~{per_question['input_tokens'] * calls / 1e6:.2f} M input + "
              f"~{per_question['output_tokens'] * calls / 1e6:.2f} M output tokens, "
              f"~${cost_per_question * calls:.2f} (input alone ~${input_cost:.2f}; every 100 "
              f"output tokens per question-run adds "
              f"~${call_cost({'output_tokens': 100 * calls}):.2f})")
    return {"chars_per_token": CHARS_PER_TOKEN,
            "mean_reader_prompt_chars": statistics.mean(reader_chars),
            "mean_judge_prompt_chars": statistics.mean(judge_chars),
            "est_reader_output_tokens": EST_READER_OUTPUT_TOKENS,
            "est_judge_output_tokens": EST_JUDGE_OUTPUT_TOKENS,
            "cost_per_question_run_usd": cost_per_question,
            "full_locomo_questions": full_questions, "runs": runs,
            "full_locomo_cost_usd": cost_per_question * full_questions * runs}


def retrieval_phase(args, data_by_id: dict, results: dict, todo: list[dict], out_path: Path):
    convs = sorted({q["conversation"] for q in todo})
    server = None
    if args.no_spawn:
        if not (args.url and args.api_key):
            raise SystemExit("--no-spawn needs --url and --api-key")
        refuse_live_url(args.url)
        base_url, api_key = args.url, args.api_key
    else:
        server = ScratchServer(args.binary, args.port, args.server_env, args.storage_backend,
                               args.startup_timeout, args.copy_env)
        server.start()
        base_url, api_key = server.url, server.api_key
        results["veld"].update(binary=server.binary, binary_version=server.version,
                               binary_sha256=sha256_file(server.binary),
                               health=server.health, copied_env=sorted(server.copied_env))
    try:
        client = VeldClient(base_url, api_key)
        turns_by_conv = {}
        for cid in convs:
            user_id = user_for(cid)
            existing = client.index_coverage(user_id)
            if existing and existing.get("stored", 0) > 0:
                raise SystemExit(f"{user_id} already holds {existing['stored']} memories on "
                                 f"{base_url}; use a fresh scratch server.")
            stats, memory_turns = ingest_conversation(client, data_by_id[cid])
            stats["index"] = wait_for_index(client, user_id, stats["unique_memories"],
                                            args.index_timeout)
            dataset_turns = sum(len(t) for _, t, _ in conversation_sessions(data_by_id[cid]))
            stats["dataset_turns"] = dataset_turns
            results["ingest"][cid] = stats
            turns_by_conv[cid] = memory_turns
            cov = stats["index"]["coverage"] or {}
            print(f"  {cid}: {dataset_turns} turns in {stats['sessions']} sessions -> created "
                  f"{stats['created']}, failed {stats['failed']}, unique memories "
                  f"{stats['unique_memories']} ({stats['ingest_s']}s); indexed after "
                  f"{stats['index']['waited_s']}s: backlog pending {stats['index']['pending']}, "
                  f"stored {cov.get('stored')}, vector {cov.get('vector_indexed')}, lexical "
                  f"{cov.get('lexically_indexed')}")
            for e in stats["errors"]:
                print(f"    not stored: {e['dia_id']} ({e['error']})")
        embedder = client.health().get("embedder")
        results["veld"]["embedder"] = embedder
        print(f"  primary embedder after ingest: {json.dumps(embedder)}")
        for q in todo:
            row = results["rows"].setdefault(q["qid"], {**q, "runs": []})
            row.update(retrieve(client, q, args.top_k, turns_by_conv[q["conversation"]]))
        print(f"Retrieved top-{args.top_k} for {len(todo)} questions.")
        for cid in convs:
            cov = client.index_coverage(user_for(cid))
            results["ingest"][cid]["coverage_after_retrieval"] = cov
            if cov:
                print(f"  {cid} after retrieval: stored {cov['stored']}, vector "
                      f"{cov['vector_indexed']}, lexical {cov['lexically_indexed']}")
    finally:
        if server is not None:
            log_copy = out_path.with_name(out_path.stem + ".server.log")
            server.stop(keep_log=log_copy)
            print(f"  scratch server log kept at {log_copy}")


def api_phase(args, results: dict, qids: list[str], speakers: dict, out_path: Path):
    rows = results["rows"]
    units = []
    for qid in qids:
        row = rows[qid]
        while len(row["runs"]) < args.runs:
            row["runs"].append({"run": len(row["runs"]), "answer": None, "label": None,
                                "calls": []})
        units.extend((qid, r) for r in range(args.runs) if row["runs"][r]["label"] is None)
    if not units:
        print("Every (question, run) pair is already judged.")
        return
    print(f"Running reader + judge for {len(units)} (question, run) pairs on {MODEL}, "
          f"concurrency {args.concurrency}")
    client = make_client()
    abort = threading.Event()
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {}
    done = 0
    last_save = time.time()
    try:
        for qid, r in units:
            row = rows[qid]
            prompt = reader_prompt(row, speakers[row["conversation"]])
            futures[pool.submit(run_unit, client, row, copy.deepcopy(row["runs"][r]), prompt,
                                abort)] = (qid, r)
        for future in as_completed(futures):
            qid, r = futures[future]
            previous = len(rows[qid]["runs"][r]["calls"])
            run, fatal = future.result()
            rows[qid]["runs"][r] = run
            for call in run["calls"][previous:]:
                if call["error"]:
                    print(f"  {qid} run {r} {call['kind']}: {call['error']} "
                          f"(request_id {call['request_id']})", file=sys.stderr)
            if fatal and not abort.is_set():
                abort.set()
                print("Stopping: a non-retryable API error means every remaining call would "
                      "fail the same way. Fix it, then rerun with --resume.", file=sys.stderr)
            done += 1
            if done % 50 == 0 or time.time() - last_save > 60:
                save_results(results, out_path)
                last_save = time.time()
                spent = aggregate(results, qids, args.runs)["estimated_cost_usd"]
                print(f"  {done}/{len(units)} done, estimated spend ${spent:.2f}")
    except KeyboardInterrupt:
        abort.set()
        print("\nInterrupted: dropping queued calls and waiting for the in-flight ones "
              "(Ctrl-C again to abandon them).", file=sys.stderr)
        pool.shutdown(wait=True, cancel_futures=True)
        for future, (qid, r) in futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                rows[qid]["runs"][r] = future.result()[0]
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        save_results(results, out_path)


def default_output(dry_run: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "dryrun" if dry_run else "run"
    return CACHE_DIR / "results" / f"locomo_qa_{kind}_{stamp}.json"


def main():
    ap = argparse.ArgumentParser(description="LoCoMo end-to-end QA accuracy (J) for veld")
    ap.add_argument("--conversations", default=None,
                    help="Comma-separated sample ids, e.g. conv-26,conv-30 (default: all 10)")
    ap.add_argument("--limit-questions", type=int, default=0,
                    help="First N non-adversarial questions per conversation (0 = all)")
    ap.add_argument("--top-k", type=int, default=10, help="Memories recalled per question")
    ap.add_argument("--runs", type=int, default=3,
                    help="Reader + judge repetitions over the same retrieved contexts")
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel API calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="Spawn, ingest, retrieve and print the assembled prompts and a cost "
                         "projection; no API calls")
    ap.add_argument("--show-prompts", type=int, default=2,
                    help="Dry run: questions whose prompts are printed")
    ap.add_argument("--output", default=None,
                    help="Results JSON (default ~/.cache/veld-bench/results/locomo_qa_*.json)")
    ap.add_argument("--resume", action="store_true",
                    help="Continue --output: reuse its retrievals and judged runs")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY), help="veld binary to spawn")
    ap.add_argument("--port", type=int, default=None, help="Scratch server port (default: free)")
    ap.add_argument("--server-env", default=str(SERVER_ENV_PATH),
                    help="Env file whose embedder settings the scratch server copies")
    ap.add_argument("--copy-env", action="append", default=[], metavar="NAME",
                    help="Also copy this variable from --server-env (repeatable), e.g. "
                         "VELD_DIALS_DISABLE_SIGNALS to match production's retrieval dials")
    ap.add_argument("--storage-backend", default="rocksdb", choices=["rocksdb", "redb"])
    ap.add_argument("--startup-timeout", type=float, default=180)
    ap.add_argument("--index-timeout", type=float, default=900,
                    help="Seconds to wait per conversation for the embedding backlog to drain")
    ap.add_argument("--no-spawn", action="store_true", help="Use an existing scratch server")
    ap.add_argument("--url", default=None, help="With --no-spawn: server base URL")
    ap.add_argument("--api-key", default=None, help="With --no-spawn: server API key")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))

    data, dataset_path, dataset_sha = load_locomo(Path(args.cache_dir).expanduser())
    data_by_id = {conv["sample_id"]: conv for conv in data}
    wanted = args.conversations.split(",") if args.conversations else sorted(data_by_id)
    unknown = [c for c in wanted if c not in data_by_id]
    if unknown:
        raise SystemExit(f"Unknown conversations {unknown}; known: {sorted(data_by_id)}")

    questions = []
    for cid in wanted:
        qs = conversation_questions(data_by_id[cid])
        questions.extend(qs[:args.limit_questions] if args.limit_questions > 0 else qs)
    full_questions = sum(len(conversation_questions(conv)) for conv in data)
    speakers = {cid: (data_by_id[cid]["conversation"]["speaker_a"],
                      data_by_id[cid]["conversation"]["speaker_b"]) for cid in data_by_id}
    print(f"LoCoMo {dataset_path} sha256 {dataset_sha}: {len(questions)} questions selected "
          f"from {len(wanted)} conversations ({full_questions} non-adversarial in the dataset)")

    out_path = Path(args.output).expanduser() if args.output else default_output(args.dry_run)
    if out_path.exists() and not args.resume:
        raise SystemExit(f"{out_path} exists; pass --resume to continue it or choose another "
                         "--output.")
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        if results["dataset"]["sha256"] != dataset_sha:
            raise SystemExit(f"{out_path} was made from a different locomo10.json")
        if results["config"]["top_k"] != args.top_k:
            raise SystemExit(f"{out_path} retrieved top-{results['config']['top_k']}; "
                             f"rerun with --top-k {results['config']['top_k']}")
        results["config"]["runs"] = max(results["config"]["runs"], args.runs)
        results["dry_run"] = results.get("dry_run", False) and args.dry_run
        print(f"Resuming {out_path}")
    else:
        results = {
            "benchmark": "LoCoMo end-to-end QA accuracy (J)",
            "comparability_note": COMPARABILITY_NOTE,
            "dataset": {"url": DATASET_URL, "path": str(dataset_path), "sha256": dataset_sha},
            "config": {
                "model": MODEL, "reader_effort": READER_EFFORT, "judge_effort": JUDGE_EFFORT,
                "reader_max_tokens": READER_MAX_TOKENS, "judge_max_tokens": JUDGE_MAX_TOKENS,
                "fallbacks": f"default ({FALLBACK_BETA})", "top_k": args.top_k,
                "recall_mode": "hybrid (server default)", "runs": args.runs,
                "categories": {str(k): v for k, v in CATEGORY.items()},
                "excluded_categories": sorted(EXCLUDED_CATEGORIES),
                "answer_prompt": ANSWER_PROMPT_SOURCE, "judge_prompt": ACCURACY_PROMPT_SOURCE,
                "session_timezone": "UTC",
            },
            "veld": {"git": git_describe()},
            "ingest": {},
            "rows": {},
            "dry_run": args.dry_run,
        }
    results["config"]["conversations"] = wanted
    results["config"]["limit_questions"] = args.limit_questions
    qids = [q["qid"] for q in questions]

    todo = [q for q in questions if results["rows"].get(q["qid"], {}).get("retrieved") is None]
    if todo:
        retrieval_phase(args, data_by_id, results, todo, out_path)
        save_results(results, out_path)
    else:
        print("All selected questions already carry their retrieval; no server needed.")

    if args.dry_run:
        rows = [results["rows"][q] for q in qids]
        for row in rows[:args.show_prompts]:
            print(f"\n----- reader prompt, {row['qid']} ({row['category']}) -----")
            print(reader_prompt(row, speakers[row["conversation"]]))
            print(f"----- judge prompt, {row['qid']} (answer placeholder) -----")
            print(judge_prompt(row, "<reader answer>"))
        results["projection"] = project(rows, speakers, args.runs, full_questions)
        save_results(results, out_path)
        print(f"\nDry run written to {out_path}")
        return

    api_phase(args, results, qids, speakers, out_path)
    results["aggregate"] = aggregate(results, qids, results["config"]["runs"])
    save_results(results, out_path)
    print_summary(results, results["aggregate"])
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
