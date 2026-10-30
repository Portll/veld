#!/usr/bin/env python3
"""
LoCoMo end-to-end QA accuracy (J, LLM-judged) for veld, and for mem0 OSS on the same protocol.

Protocol (--system veld, the default):
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
     "[<created_at date>] <content>". With --event-dates they are
     "[mention <created_at date>, occurred <date>, planned <date>] <content>", from the recall
     response's occurred and planned fields (dates the server resolves from the content at read
     time). --retrieval-from reuses another results file's retrieval unchanged, so two runs can
     differ only in how the same memories are shown. --eval-now pins the server's scoring clock
     (VELD_EVAL_NOW); unset, recency is scored against the wall clock.
  5. Context: each question's retrieved lines are counted in tokens (tiktoken o200k_base, the
     same count for every system). --context-budget N keeps the longest prefix of the ranked
     lines that fits N tokens, so systems can be compared at one budget; unset, all k are kept.
     Evidence hit@k is measured on the lines the reader sees.
  6. Reader: Mem0's ANSWER_PROMPT. Judge: Mem0's ACCURACY_PROMPT (J). --runs repeats reader +
     judge over the same retrieved contexts. J is the share of judged answers labelled CORRECT;
     mean and sample sd are over runs, and a 95% Wilson interval is over questions (the mean J
     over runs and the mean judged count per run). A judge reply without a CORRECT/WRONG label is
     an error row, left unjudged, never counted WRONG.
     - --provider anthropic (default): claude-opus-5-5 for both (reader effort medium, judge
       effort low, thinking always on, server-side refusal fallbacks on).
     - --provider openai-compatible: chat.completions at --base-url with --model (or
       --reader-model / --judge-model), temperature 0, the prompt as the only user message; the
       judge runs in JSON mode. The answer is message.content alone; reasoning is never read.
       Known Groq models carry default reasoning efforts (MODEL_DEFAULTS) and list prices.
     - --provider lmstudio: the same against an LM Studio host (spine reaches the M4 through its
       ssh tunnel). LM Studio refuses JSON mode, so the judge gets the label as a json_schema;
       reasoning effort defaults to none for both, since structured output comes back empty with
       reasoning on; tokens cost nothing.
  7. Errors: a failed (question, run) is sent again, up to --attempts times in all, unless the
     error is fatal (bad key or request, a spent quota that will not refill within the wait cap),
     truncation or refusal, which repeat at temperature 0. A daily-quota 429 that names a wait
     under the cap is waited out, not fatal. Whatever is still unjudged is counted by reason; the
     results carry status "incomplete" and the process exits 3. An incomplete run is not a
     result: rerun with --resume.

--system mem0: the same dataset, questions, reader, judge, runs and resume, with mem0 OSS (PyPI
mem0ai, version recorded) as the memory: one mem0 user per conversation, every turn ingested
through mem0's own LLM extraction at the session's real time, the extraction LLM and embedder on
the reader's host. See benchmarks/locomo_qa_mem0.py for the ingest and retrieval details. Needs
--provider lmstudio or openai-compatible, and mem0's own venv.

Categories (LoCoMo code -> name, the mapping benchmarks/locomo_to_harness.py in shodh-memory used
to build tests/recall/locomo_cases.jsonl): 1 multi_hop, 2 temporal, 3 open_domain, 4 single_hop,
5 adversarial. Adversarial questions are excluded, as Mem0 does, leaving 1,540.

Question count: the retrieval harness (tests/recall/locomo_cases.jsonl) has 1,531 cases. It keeps
a question only when an evidence id names a turn exactly, which drops 9: four with no evidence
(conv-26_q30, conv-26_q46, conv-50_q39, conv-50_q42), four with several ids in one string
(conv-26_q37 "D8:6; D9:17", conv-49_q31, conv-49_q38, conv-49_q46) and one zero-padded id
(conv-50_q69 "D30:05"). This harness judges all 1,540, reads the packed and padded ids as the
turns they name, marks each row in_retrieval_set, and reports J on those rows too.

Comparability: published LoCoMo J figures (Mem0 and the systems its paper compares) use
gpt-4o-mini as reader and judge. Any other reader or judge makes these numbers incomparable with
them; compare runs of this harness made with the same models and settings.

Setup (once):
    /opt/homebrew/bin/python3.13 -m venv ~/.venvs/veld-bench
    ~/.venvs/veld-bench/bin/pip install -r benchmarks/requirements-longmemeval.txt tiktoken
    # mem0 2.x, and the last 1.x, each in its own venv:
    /opt/homebrew/bin/python3.13 -m venv ~/.venvs/veld-bench-mem0
    ~/.venvs/veld-bench-mem0/bin/pip install -r benchmarks/requirements-locomo-qa-mem0.txt
    /opt/homebrew/bin/python3.13 -m venv ~/.venvs/veld-bench-mem0-v1
    ~/.venvs/veld-bench-mem0-v1/bin/pip install -r benchmarks/requirements-locomo-qa-mem0-v1.txt
Credentials: anthropic.Anthropic() reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
`ant auth login` profile. openai-compatible and lmstudio: --api-key-keychain NAME reads the macOS
keychain item NAME at run time, --api-key-env NAME an environment variable; neither, and the SDK
reads OPENAI_API_KEY.

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

    # Groq, one model as reader and judge, production retrieval dials, scoring clock pinned
    ~/.venvs/veld-bench/bin/python benchmarks/locomo_qa_eval.py --runs 3 \\
        --provider openai-compatible --base-url https://api.groq.com/openai/v1 \\
        --model openai/gpt-oss-120b --api-key-keychain GROQ_API_KEY \\
        --copy-env VELD_DIALS_DISABLE_SIGNALS --eval-now auto \\
        --output ~/.cache/veld-bench/results/locomo_qa_full_gpt-oss-120b.json

    # The M4 through spine's tunnel, mem0 2.x against the same reader and judge, 1,000-token
    # contexts. Spine's locomo-qa lane job runs these (spine bin/eval-act.mjs).
    ~/.venvs/veld-bench-mem0/bin/python benchmarks/locomo_qa_eval.py --system mem0 \\
        --conversations conv-26 --runs 1 --provider lmstudio \\
        --base-url http://127.0.0.1:11234/v1 --model qwen/qwen3.8-27b \\
        --api-key-keychain LMSTUDIO_M4PRO_API_KEY --top-k 20 --context-budget 1000 \\
        --output ~/.cache/veld-bench/results/locomo_qa_mem0_conv-26.json

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
import math
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

OPENAI_MAX_RETRIES = 6
RATE_LIMIT_ROUNDS = 4
RATE_LIMIT_MAX_WAIT_S = 900
# Groq list prices per MTok, read 2026-10-22 from https://console.groq.com/docs/models and
# https://console.groq.com/docs/model/<model id> (https://groq.com/pricing redirects to the home
# page). Groq bills cached gpt-oss input at $0.075; it is costed here at the full input price.
# Efforts per https://console.groq.com/docs/reasoning: gpt-oss takes low/medium/high and refuses
# reasoning_format; qwen3.8's none is instruct mode, and unless reasoning_format is parsed or
# hidden it writes its reasoning into message.content.
MODEL_DEFAULTS = {
    "openai/gpt-oss-120b": {
        "price": {"input_tokens": 0.15, "output_tokens": 0.60},
        "reader_effort": "medium", "judge_effort": "low", "reasoning_format": None,
    },
    "qwen/qwen3.8-27b": {
        "price": {"input_tokens": 0.80, "output_tokens": 4.00},
        "reader_effort": "medium", "judge_effort": "none", "reasoning_format": "parsed",
    },
}
PRICE_SOURCE = {
    "anthropic": "Anthropic list price for claude-opus-5-5",
    "openai-compatible": "https://console.groq.com/docs/models (read 2026-10-22)",
    "lmstudio": "a local LM Studio host: no per-token charge",
}
LMSTUDIO_DEFAULTS = {"price": {"input_tokens": 0.0, "output_tokens": 0.0},
                     "reader_effort": "none", "judge_effort": "none", "reasoning_format": None}
DEFAULT_ATTEMPTS = 3
RETRY_PAUSE_S = 20
EXIT_INCOMPLETE = 3
# An error that would recur on a resend at temperature 0, or that every later call would hit.
NOT_RETRIED = re.compile(r"^(length|max_tokens|refusal)\b")
QUOTA_WAIT = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")

TOKENIZER = "o200k_base"
TOKEN_CACHE_DIR = Path.home() / ".cache" / "veld-bench" / "tiktoken"
WILSON_Z = 1.959964
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "reasoning_tokens",
              "cached_input_tokens")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

CHARS_PER_TOKEN = 3.5
EST_READER_OUTPUT_TOKENS = 1500
EST_JUDGE_OUTPUT_TOKENS = 300
EST_ANSWER_CHARS = 400

CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
EXCLUDED_CATEGORIES = {5}
# Set from --event-dates: context lines carry the dates a memory's content resolves to.
RENDER_EVENT_DATES = False
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
EVIDENCE_ID = re.compile(r"\bD(\d+):0*(\d+)\b")
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


def evidence_ids(raw: list) -> list[str]:
    """The turn ids a question's evidence names, in order: LoCoMo packs several into one string
    for a few questions ("D8:6; D9:17") and zero-pads one ("D30:05")."""
    ids = []
    for item in raw:
        if not isinstance(item, str):
            continue
        for session, turn in EVIDENCE_ID.findall(item):
            dia_id = f"D{session}:{turn}"
            if dia_id not in ids:
                ids.append(dia_id)
    return ids


def in_retrieval_set(raw: list, turn_ids: set) -> bool:
    """Whether tests/recall/locomo_cases.jsonl holds the question: an evidence entry names a turn
    exactly."""
    return any(isinstance(item, str) and item in turn_ids for item in raw)


def conversation_questions(conv: dict) -> list[dict]:
    turn_ids = {turn["dia_id"] for _, turns, _ in conversation_sessions(conv) for turn in turns}
    questions = []
    for index, qa in enumerate(conv["qa"]):
        code = qa.get("category")
        if code in EXCLUDED_CATEGORIES:
            continue
        raw = qa.get("evidence") or []
        questions.append({
            "qid": f"{conv['sample_id']}_q{index}",
            "conversation": conv["sample_id"],
            "qa_index": index,
            "category_code": code,
            "category": CATEGORY[code],
            "question": qa["question"],
            "gold": str(qa["answer"]),
            "evidence": evidence_ids(raw),
            "evidence_raw": [e for e in raw if isinstance(e, str)],
            "in_retrieval_set": in_retrieval_set(raw, turn_ids),
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
                 storage_backend="rocksdb", startup_timeout=180, extra_env=(), eval_now=None):
        self.binary = str(Path(binary).expanduser())
        self.port = port or free_port()
        if self.port in LIVE_PORTS:
            raise SystemExit(f"Port {self.port} belongs to the live veld server or its replica; "
                             "pick another --port.")
        refused = [n for n in extra_env if n.startswith(("VELD_REPLICATE_", "VELD_ADMIN_"))]
        if refused:
            raise SystemExit(f"Refusing to copy replication or admin settings: {refused}")
        if "VELD_EVAL_NOW" in extra_env:
            raise SystemExit("Pin the scoring clock with --eval-now, not --copy-env VELD_EVAL_NOW")
        self.extra_env = set(extra_env)
        self.eval_now = eval_now
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
        if self.eval_now:
            env["VELD_EVAL_NOW"] = self.eval_now
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
        print(f"  scoring clock: {f'VELD_EVAL_NOW={self.eval_now}' if self.eval_now else 'live'}")
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
            "occurred": m.get("occurred") or [],
            "planned": m.get("planned") or [],
        })



    return {
        "retrieved": retrieved,
        "retrieved_ids": [item["id"] for item in retrieved],
        "recall_ms": round(recall_ms, 2),
    }


def context_line(item: dict) -> str:
    date = memory_date(item["created_at"])
    if not RENDER_EVENT_DATES:
        return f"[{date}] {item['content']}"
    stamp = [f"mention {date}"]
    stamp += [f"occurred {d}" for d in item.get("occurred") or []]
    stamp += [f"planned {d}" for d in item.get("planned") or []]
    return f"[{', '.join(stamp)}] {item['content']}"


RETRIEVAL_FIELDS = ("retrieved", "retrieved_ids", "recall_ms")


def seed_retrieval(results: dict, source: dict, qids: list[str], source_path: Path):
    """Copy source's retrieval for qids into results, with the server and ingest it came from,
    so this run reads and judges exactly the memories that run retrieved."""
    missing = [q for q in qids if (source["rows"].get(q) or {}).get("retrieved") is None]
    if missing:
        raise SystemExit(f"{source_path} holds no retrieval for {len(missing)} selected "
                         f"questions, e.g. {missing[:3]}")
    if RENDER_EVENT_DATES and any("occurred" not in item for q in qids
                                  for item in source["rows"][q]["retrieved"]):
        raise SystemExit(f"{source_path} was retrieved before the harness recorded occurred and "
                         "planned dates; --event-dates needs a run that did")
    for qid in qids:
        row = results["rows"].setdefault(qid, {k: v for k, v in source["rows"][qid].items()
                                                if k not in ("runs", "context_n")})
        row.update({k: source["rows"][qid][k] for k in RETRIEVAL_FIELDS})
        row["runs"] = []
        row.pop("context_n", None)
    for key in ("veld", "ingest", "system"):
        if key in source:
            results[key] = source[key]


def token_counter():
    """One token count for every system's context: tiktoken's o200k_base. Its encoding file is
    fetched once into TOKEN_CACHE_DIR, and tiktoken checks it against its own hash."""
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TOKEN_CACHE_DIR))
    import tiktoken

    encoding = tiktoken.get_encoding(TOKENIZER)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def fit_context(items: list[dict], budget: int | None, count) -> list[int]:
    """Token counts of the longest prefix of the ranked lines that fits budget (all, when None).
    A prefix, not the best-fitting subset: the reader sees what the system ranked first."""
    sizes = []
    for item in items:
        n = count(context_line(item))
        if budget is not None and sum(sizes) + n > budget:
            break
        sizes.append(n)
    return sizes


def settle_context(row: dict, budget: int | None, count, maps_turns: bool):
    """The lines the reader will see, their tokens, and whether they hold an evidence turn
    (None when the system's memories do not map to turns, or the question names none)."""
    sizes = fit_context(row["retrieved"], budget, count)
    row["context_n"] = len(sizes)
    row["context_tokens"] = sum(sizes)
    row["retrieved_tokens"] = sum(count(context_line(item)) for item in row["retrieved"])
    shown = {d for item in row["retrieved"][:len(sizes)] for d in item["dia_ids"]}
    row["evidence_hit"] = (bool(shown & set(row["evidence"]))
                           if maps_turns and row["evidence"] else None)


def visible(row: dict) -> list[dict]:
    return row["retrieved"][:row.get("context_n", len(row["retrieved"]))]


def wilson(p: float, n: float, z: float = WILSON_Z) -> list[float] | None:
    """Wilson score interval for a share p over n questions; n may be a mean over runs."""
    if not n:
        return None
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def reader_prompt(row: dict, speakers: tuple[str, str]) -> str:
    first, second = speakers
    lines = {first: [], second: []}
    for item in visible(row):
        if item["speaker"] not in lines:
            raise ValueError(f"{row['qid']}: retrieved memory {item['id']} has speaker "
                             f"{item['speaker']!r}, not {first} or {second}")
        lines[item["speaker"]].append(context_line(item))
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


def comparability_note(llm: dict, system: dict | None = None) -> str:
    reader, judge = llm["reader"], llm["judge"]
    if reader["model"] == judge["model"]:
        models = (f"{reader['model']} as both reader and judge (reader effort "
                  f"{reader['effort']}, judge effort {judge['effort']})")
    else:
        models = (f"{reader['model']} as reader (effort {reader['effort']}) and "
                  f"{judge['model']} as judge (effort {judge['effort']})")
    memory = ""
    if system and system.get("name") == "mem0":
        memory = (f" mem0 {system['version']} extracted facts with {system['llm_model']} "
                  f"(reasoning {system['llm_reasoning_effort']}) and embedded with "
                  f"{system['embedder_model']}.")
    elif system and system.get("name") == "veld":
        memory = f" veld embedded with {system.get('embedder') or 'its configured embedder'}."
    return (f"Not comparable with published LoCoMo J figures: this run uses {models}, while "
            "every published J figure (Mem0 and the systems its paper compares) uses gpt-4o-mini "
            f"for both.{memory} Compare only runs of this harness made with the same model "
            "settings.")


def keychain_secret(name: str) -> str:
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", name, "-w"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"Keychain item {name!r} is not readable: {e}")
    return out.stdout.strip()


def llm_api_key(args) -> str | None:
    if args.api_key_keychain:
        return keychain_secret(args.api_key_keychain)
    if args.api_key_env:
        value = os.environ.get(args.api_key_env)
        if not value:
            raise SystemExit(f"${args.api_key_env} is not set")
        return value
    return None


RATE_LIMITED = threading.local()


def count_rate_limited(response):
    if response.status_code == 429:
        RATE_LIMITED.count = getattr(RATE_LIMITED, "count", 0) + 1


def make_client(llm: dict, api_key: str | None):
    options = {k: v for k, v in (("base_url", llm["base_url"]), ("api_key", api_key)) if v}
    if llm["provider"] == "anthropic":
        import anthropic

        return anthropic.Anthropic(max_retries=5, **options)
    import openai

    return openai.OpenAI(max_retries=OPENAI_MAX_RETRIES, http_client=openai.DefaultHttpxClient(
        event_hooks={"response": [count_rate_limited]}), **options)


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


def retry_after_s(error) -> float | None:
    headers = error.response.headers if error.response is not None else {}
    for name, scale in (("retry-after-ms", 1000), ("retry-after", 1)):
        try:
            return float(headers[name]) / scale
        except (KeyError, ValueError):
            continue
    return None


def quota_wait_s(message: str) -> float | None:
    """The wait a 429 body names ("Please try again in 3m18.72s"), in seconds."""
    match = QUOTA_WAIT.search(message)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def rate_limit_wait_s(error) -> float | None:
    wait = retry_after_s(error)
    return wait if wait is not None else quota_wait_s(error.message)


def rate_limit_terminal(error) -> str | None:
    wait = rate_limit_wait_s(error)
    if "Request too large" in error.message:
        return ("max tokens exceed a per-minute limit; lower --reader-max-tokens or "
                "--judge-max-tokens")
    # Groq's daily token limits refill continuously, so a spent day usually names a wait of
    # minutes; only an unnamed or long wait ends the run.
    if ("per day" in error.message or "(TPD)" in error.message) and wait is None:
        return "a per-day quota is spent; rerun with --resume once it refills"
    if wait is not None and wait > RATE_LIMIT_MAX_WAIT_S:
        return f"retry-after {wait:.0f}s exceeds the {RATE_LIMIT_MAX_WAIT_S}s wait cap"
    return None


def create_completion(client, request: dict):
    import openai

    for attempt in range(RATE_LIMIT_ROUNDS):
        try:
            return client.chat.completions.create(**request)
        except openai.RateLimitError as e:
            if rate_limit_terminal(e):
                raise
            # The SDK's retries run out under contention, and it gives up on any retry-after
            # above its own ceiling of one or two minutes.
            wait = rate_limit_wait_s(e)
            time.sleep(wait + 1 if wait is not None else 5 * 2 ** attempt)
    return client.chat.completions.create(**request)


def judge_format(provider: str) -> dict:
    if provider == "lmstudio":
        return {"type": "json_schema",
                "json_schema": {"name": "judge_label", "strict": True, "schema": JUDGE_SCHEMA}}
    return {"type": "json_object"}


def call_openai_compatible(client, prompt: str, role: dict,
                           response_format: dict | None) -> tuple[dict, str | None]:
    import openai

    extra = {"reasoning_effort": role["effort"], "reasoning_format": role["reasoning_format"]}
    request = {
        "model": role["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": role["max_tokens"],
        "extra_body": {k: v for k, v in extra.items() if v is not None},
    }
    json_mode = response_format is not None
    if json_mode:
        request["response_format"] = response_format
    record = {"model": role["model"], "served_model": None, "request_id": None,
              "stop_reason": None, "fallback": False, "input_tokens": 0, "output_tokens": 0,
              "reasoning_tokens": 0, "cached_input_tokens": 0, "rate_limited": 0,
              "latency_ms": None, "error": None, "fatal": False}
    text = None
    RATE_LIMITED.count = 0
    started = time.perf_counter()
    try:
        response = create_completion(client, request)
    except openai.RateLimitError as e:
        terminal = rate_limit_terminal(e)
        record.update(error=f"HTTP 429, {terminal or 'still limited after retries'}: {e.message}",
                      request_id=e.request_id, fatal=terminal is not None)
    except openai.BadRequestError as e:
        parse_failure = json_mode and e.code == "json_validate_failed"
        record.update(error=f"HTTP {e.status_code} {e.code}: {e.message}",
                      request_id=e.request_id, fatal=not parse_failure)
        if parse_failure:
            record["parse_failure"] = True
    except (openai.AuthenticationError, openai.PermissionDeniedError,
            openai.NotFoundError) as e:
        record.update(error=f"HTTP {e.status_code}: {e.message}", request_id=e.request_id,
                      fatal=True)
    except openai.APIStatusError as e:
        record.update(error=f"HTTP {e.status_code} after retries: {e.message}",
                      request_id=e.request_id, fatal=e.status_code == 413)
    except openai.APIConnectionError as e:
        record.update(error=f"connection error after retries: {e}")
    else:
        usage = response.usage
        output_details = getattr(usage, "completion_tokens_details", None)
        input_details = getattr(usage, "prompt_tokens_details", None)
        choice = response.choices[0] if response.choices else None
        record.update(
            request_id=getattr(response, "_request_id", None),
            served_model=response.model,
            stop_reason=choice.finish_reason if choice else None,
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
            reasoning_tokens=(output_details.reasoning_tokens or 0) if output_details else 0,
            cached_input_tokens=(input_details.cached_tokens or 0) if input_details else 0,
        )
        if choice is None:
            record["error"] = "response carried no choice"
        elif choice.finish_reason == "length":
            record["error"] = f"length: output truncated at {role['max_tokens']} tokens"
        elif choice.finish_reason == "content_filter":
            record["error"] = "refusal (content_filter)"
        elif choice.finish_reason != "stop":
            record["error"] = f"unexpected finish_reason {choice.finish_reason}"
        else:
            text = THINK_BLOCK.sub("", choice.message.content or "").strip()
            if "<think>" in text:
                record["error"] = "reasoning is not separated from the answer"
                text = None
            elif not text:
                record["error"] = "response carried no content"
                text = None
    record["rate_limited"] = RATE_LIMITED.count
    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return record, text


def call_model(client, llm: dict, kind: str, prompt: str) -> tuple[dict, str | None]:
    role = llm[kind]
    if llm["provider"] == "anthropic":
        return call_claude(client, prompt, role["effort"], role["max_tokens"],
                           schema=JUDGE_SCHEMA if kind == "judge" else None, model=role["model"])
    return call_openai_compatible(client, prompt, role,
                                  judge_format(llm["provider"]) if kind == "judge" else None)


def call_cost(record: dict, prices: dict) -> float:
    price = prices.get(record.get("model"), {})
    return sum(record.get(key, 0) * p for key, p in price.items()) / 1e6


def run_unit(client, llm: dict, row: dict, run: dict, prompt: str,
             abort: threading.Event) -> tuple[dict, bool]:
    if abort.is_set():
        return run, False
    if run["answer"] is None:
        record, text = call_model(client, llm, "reader", prompt)
        run["calls"].append({"kind": "reader", **record})
        if record["error"]:
            return run, record["fatal"]
        run["answer"] = text
        run["answer_fallback"] = record["fallback"]
    record, text = call_model(client, llm, "judge", judge_prompt(row, run["answer"]))
    call = {"kind": "judge", **record}
    run["calls"].append(call)
    if record["error"]:
        return run, record["fatal"]
    try:
        label = json.loads(text)["label"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        call.update(error=f"judge output is not the label schema: {e}: {text[:200]!r}",
                    parse_failure=True)
        return run, False
    if label not in ("CORRECT", "WRONG"):
        call.update(error=f"judge label {label!r} is neither CORRECT nor WRONG",
                    parse_failure=True)
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
        judged = [pr["judged"][c] for pr in per_run if pr["j"][c] is not None]
        mean = statistics.mean(values) if values else None
        j[c] = {
            "n": n[c],
            "mean": mean,
            "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "ci95": wilson(mean, statistics.mean(judged)) if values else None,
            "per_run": [pr["j"][c] for pr in per_run],
            "judged_per_run": [pr["judged"][c] for pr in per_run],
        }

    reasons = {}
    examples = []
    for row in rows:
        for run in row.get("runs", [])[:runs]:
            state = unit_state(run)
            if state == "judged":
                continue
            reasons[state] = reasons.get(state, 0) + 1
            if len(examples) < 20:
                last = run["calls"][-1] if run["calls"] else {}
                examples.append({"qid": row["qid"], "run": run["run"], "state": state,
                                 "kind": last.get("kind"), "error": last.get("error")})
    missing_runs = sum(max(0, runs - len(row.get("runs", []))) for row in rows)
    if missing_runs:
        reasons["not_attempted"] = reasons.get("not_attempted", 0) + missing_runs

    context = [r["context_tokens"] for r in rows if r.get("context_tokens") is not None]
    retrieved = [r["retrieved_tokens"] for r in rows if r.get("retrieved_tokens") is not None]
    lines = [r["context_n"] for r in rows if r.get("context_n") is not None]
    context_tokens = {
        "tokenizer": TOKENIZER, "n": len(context),
        "mean": statistics.mean(context) if context else None,
        "p50": percentile(context, 0.50), "p95": percentile(context, 0.95),
        "max": max(context) if context else None,
        "mean_lines": statistics.mean(lines) if lines else None,
        "mean_retrieved_tokens": statistics.mean(retrieved) if retrieved else None,
    }

    evidence = {}
    for c in keys:
        hits = [r["evidence_hit"] for r in rows
                if r.get("evidence_hit") is not None and c in (r["category"], "overall")]
        evidence[c] = sum(hits) / len(hits) if hits else None

    latencies = [r["recall_ms"] for r in rows if r.get("recall_ms") is not None]
    calls = [call for row in rows for run in row.get("runs", []) for call in run["calls"]]
    totals = {key: sum(call.get(key, 0) for call in calls) for key in TOKEN_KEYS}
    by_kind = {}
    for kind in ("reader", "judge"):
        mine = [call for call in calls if call["kind"] == kind]
        by_kind[kind] = {
            "calls": len(mine),
            "errors": sum(1 for call in mine if call["error"]),
            "refusals": sum(1 for call in mine
                            if call["stop_reason"] in ("refusal", "content_filter")),
            "truncated": sum(1 for call in mine if call["stop_reason"] in ("max_tokens", "length")),
            "parse_failures": sum(1 for call in mine if call.get("parse_failure")),
            "rate_limited_responses": sum(call.get("rate_limited", 0) for call in mine),
            "fallback_served": sum(1 for call in mine if call["fallback"]),
        }
    answers_by_fallback = sum(1 for row in rows for run in row.get("runs", [])
                              if run.get("label") and run.get("answer_fallback"))
    return {
        "categories": keys,
        "j": j,
        "unjudged_per_run": [pr["unjudged"] for pr in per_run],
        "unjudged_reasons": reasons,
        "unjudged_examples": examples,
        "context_tokens": context_tokens,
        "evidence_hit_at_k": evidence,
        "recall_latency_ms": {"p50": percentile(latencies, 0.50),
                              "p95": percentile(latencies, 0.95), "n": len(latencies)},
        "calls": by_kind,
        "judged_answers_served_by_fallback": answers_by_fallback,
        "tokens": totals,
        "estimated_cost_usd": round(sum(call_cost(call, results["config"]["price_per_mtok"])
                                        for call in calls), 4),
    }


def fmt(value, pattern="{:.3f}"):
    return "-" if value is None else pattern.format(value)


def price_text(prices: dict) -> str:
    return "; ".join(
        f"{model} " + " / ".join(f"{key.removesuffix('_tokens')} ${usd:g}"
                                 for key, usd in price.items())
        for model, price in prices.items()) + " per MTok"


def summarise(results: dict, qids: list[str], runs: int) -> dict:
    """The aggregate, J on the retrieval harness's 1,531-question set, and the run's status."""
    agg = aggregate(results, qids, runs)
    subset = [q for q in qids if results["rows"][q].get("in_retrieval_set", True)]
    excluded = [q for q in qids if not results["rows"][q].get("in_retrieval_set", True)]
    agg["retrieval_set"] = {"questions": len(subset), "excluded": excluded,
                            "j": aggregate(results, subset, runs)["j"].get("overall")
                            if excluded and subset else agg["j"]["overall"]}
    unjudged = sum(agg["unjudged_reasons"].values())
    results["status"] = "incomplete" if unjudged else "complete"
    results["unjudged"] = unjudged
    return agg


def system_line(results: dict) -> str:
    system = results.get("system") or {"name": "veld"}
    if system["name"] == "mem0":
        return (f"mem0 {system['version']} ({system['llm_model']} extraction, reasoning "
                f"{system['llm_reasoning_effort']}; {system['embedder_model']} embeddings; "
                f"features {json.dumps(system['features'])})")
    return f"veld {results['veld'].get('binary_version')} (repo {results['veld'].get('git')})"


def print_summary(results: dict, agg: dict):
    cfg = results["config"]
    print("\n## LoCoMo QA accuracy (J)\n")
    budget = cfg.get("context_budget")
    print(f"{system_line(results)}, top-k {cfg['top_k']}, context budget "
          f"{f'{budget} tokens' if budget else 'none'}, {cfg['provider']} reader "
          f"{cfg['reader_model']} (effort {cfg['reader_effort']}), judge {cfg['judge_model']} "
          f"(effort {cfg['judge_effort']}), runs {cfg['runs']}, scoring clock "
          f"{cfg.get('eval_now') or 'live'}\n")
    print(f"> {results['comparability_note']}\n")
    print("| Category | n | J mean | 95% CI | J sd | J per run | evidence hit@k |")
    print("|---|---:|---:|---|---:|---|---:|")
    for c in agg["categories"]:
        row = agg["j"][c]
        per_run = ", ".join(fmt(v) for v in row["per_run"])
        ci = "-" if not row.get("ci95") else f"{row['ci95'][0]:.3f}-{row['ci95'][1]:.3f}"
        print(f"| {c} | {row['n']} | {fmt(row['mean'])} | {ci} | {fmt(row['sd'])} | {per_run} | "
              f"{fmt(agg['evidence_hit_at_k'][c])} |")
    subset = agg.get("retrieval_set")
    if subset and subset["excluded"]:
        print(f"\nOn the retrieval harness's question set ({subset['questions']} here; "
              f"{len(subset['excluded'])} have no evidence id naming a turn: "
              f"{', '.join(subset['excluded'])}): J {fmt((subset['j'] or {}).get('mean'))}")
    ctx = agg["context_tokens"]
    print(f"\nContext per question ({ctx['tokenizer']} tokens): mean {fmt(ctx['mean'], '{:.0f}')}, "
          f"p50 {fmt(ctx['p50'], '{:.0f}')}, p95 {fmt(ctx['p95'], '{:.0f}')}, max "
          f"{fmt(ctx['max'], '{:.0f}')}, mean lines {fmt(ctx['mean_lines'], '{:.1f}')} "
          f"(retrieved before the budget: {fmt(ctx['mean_retrieved_tokens'], '{:.0f}')})")
    lat = agg["recall_latency_ms"]
    print(f"Recall latency: p50 {fmt(lat['p50'], '{:.1f}')} ms, "
          f"p95 {fmt(lat['p95'], '{:.1f}')} ms (n={lat['n']})")
    for kind, c in agg["calls"].items():
        print(f"{kind.capitalize()} calls: {c['calls']}, errors {c['errors']} "
              f"(refusals {c['refusals']}, truncated {c['truncated']}, unparseable "
              f"{c['parse_failures']}), HTTP 429 responses {c['rate_limited_responses']}, "
              f"served by a fallback model {c['fallback_served']}")
    if cfg["provider"] == "anthropic":
        print(f"Judged answers written by a fallback model, not {cfg['reader_model']}: "
              f"{agg['judged_answers_served_by_fallback']}")
    t = agg["tokens"]
    print(f"Tokens: input {t['input_tokens']:,} (cached {t['cached_input_tokens']:,}), output "
          f"{t['output_tokens']:,} (reasoning {t['reasoning_tokens']:,}), cache read "
          f"{t['cache_read_input_tokens']:,}; estimated cost ${agg['estimated_cost_usd']:.2f} at "
          f"{price_text(cfg['price_per_mtok'])}"
          + (", cached input at the full input price" if t["cached_input_tokens"] else ""))
    if agg["unjudged_reasons"]:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(agg["unjudged_reasons"].items()))
        print(f"\nINCOMPLETE: {sum(agg['unjudged_reasons'].values())} (question, run) pairs "
              f"unjudged ({reasons}); J above is over judged pairs only and is not a result. "
              "Rerun with --resume.")
        for e in agg["unjudged_examples"][:5]:
            print(f"  {e['qid']} run {e['run']} {e['state']} {e['kind'] or ''}: "
                  f"{(e['error'] or 'never sent')[:160]}")


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


def project(rows: list[dict], speakers: dict, runs: int, full_questions: int, llm: dict):
    reader_chars = [len(reader_prompt(r, speakers[r["conversation"]])) for r in rows]
    judge_chars = [len(judge_prompt(r, "x" * EST_ANSWER_CHARS)) for r in rows]
    reader_in = statistics.mean(reader_chars) / CHARS_PER_TOKEN
    judge_in = statistics.mean(judge_chars) / CHARS_PER_TOKEN
    per_question = {
        "input_tokens": reader_in + judge_in,
        "output_tokens": EST_READER_OUTPUT_TOKENS + EST_JUDGE_OUTPUT_TOKENS,
    }

    def cost(kind: str, **tokens) -> float:
        return call_cost({"model": llm[kind]["model"], **tokens}, llm["prices"])

    cost_per_question = (cost("reader", input_tokens=reader_in,
                              output_tokens=EST_READER_OUTPUT_TOKENS)
                         + cost("judge", input_tokens=judge_in,
                                output_tokens=EST_JUDGE_OUTPUT_TOKENS))
    print("\n## Projection (no API calls made)\n")
    print(f"{llm['provider']}: reader {llm['reader']['model']} (effort "
          f"{llm['reader']['effort']}), judge {llm['judge']['model']} (effort "
          f"{llm['judge']['effort']}), at {price_text(llm['prices'])}.")
    print(f"Mean reader prompt {statistics.mean(reader_chars):,.0f} chars "
          f"(~{reader_in:,.0f} tokens), judge prompt {statistics.mean(judge_chars):,.0f} chars "
          f"(~{judge_in:,.0f} tokens) at {CHARS_PER_TOKEN} chars/token, over {len(rows)} "
          "questions.")
    print(f"Assumed, not measured: reader output {EST_READER_OUTPUT_TOKENS} tokens and judge "
          f"output {EST_JUDGE_OUTPUT_TOKENS} tokens per question-run, thinking included; the "
          f"judge sees a {EST_ANSWER_CHARS}-char reader answer.")
    for label, count in (("selected", len(rows)), ("full LoCoMo", full_questions)):
        calls = count * runs
        input_cost = (cost("reader", input_tokens=reader_in * calls)
                      + cost("judge", input_tokens=judge_in * calls))
        print(f"- {label}: {count} questions x {runs} runs = {2 * calls} calls, "
              f"~{per_question['input_tokens'] * calls / 1e6:.2f} M input + "
              f"~{per_question['output_tokens'] * calls / 1e6:.2f} M output tokens, "
              f"~${cost_per_question * calls:.2f} (input alone ~${input_cost:.2f}; every 100 "
              f"reader output tokens per question-run adds "
              f"~${cost('reader', output_tokens=100 * calls):.2f})")
    return {"chars_per_token": CHARS_PER_TOKEN,
            "mean_reader_prompt_chars": statistics.mean(reader_chars),
            "mean_judge_prompt_chars": statistics.mean(judge_chars),
            "est_reader_output_tokens": EST_READER_OUTPUT_TOKENS,
            "est_judge_output_tokens": EST_JUDGE_OUTPUT_TOKENS,
            "cost_per_question_run_usd": cost_per_question,
            "full_locomo_questions": full_questions, "runs": runs,
            "full_locomo_cost_usd": cost_per_question * full_questions * runs}


def retrieval_phase(args, data_by_id: dict, results: dict, todo: list[dict], out_path: Path,
                    llm: dict, count, api_key: str | None = None):
    if args.system == "mem0":
        return mem0_retrieval_phase(args, data_by_id, results, todo, out_path, llm, count,
                                    api_key)
    convs = sorted({q["conversation"] for q in todo})
    server = None
    if args.no_spawn:
        if not (args.url and args.api_key):
            raise SystemExit("--no-spawn needs --url and --api-key")
        refuse_live_url(args.url)
        base_url, api_key = args.url, args.api_key
    else:
        server = ScratchServer(args.binary, args.port, args.server_env, args.storage_backend,
                               args.startup_timeout, args.copy_env, results["config"]["eval_now"])
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
        results["system"] = {"name": "veld", "embedder": embedder.get("model")
                             if isinstance(embedder, dict) else embedder}
        print(f"  primary embedder after ingest: {json.dumps(embedder)}")
        for q in todo:
            row = results["rows"].setdefault(q["qid"], {**q, "runs": []})
            row.update(retrieve(client, q, args.top_k, turns_by_conv[q["conversation"]]))
            settle_context(row, args.context_budget, count, maps_turns=True)
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


def mem0_retrieval_phase(args, data_by_id: dict, results: dict, todo: list[dict], out_path: Path,
                         llm: dict, count, api_key: str | None, system=None):
    if system is None:
        from locomo_qa_mem0 import Mem0System

        system = Mem0System(
            provider=llm["provider"], base_url=llm["base_url"], api_key=api_key,
            llm_model=args.mem0_llm_model or llm["reader"]["model"],
            llm_effort=None if args.mem0_llm_effort == "default" else args.mem0_llm_effort,
            llm_max_tokens=args.mem0_llm_max_tokens, embedder_model=args.mem0_embedder_model,
            embedding_dims=args.mem0_embedding_dims, batch_turns=args.mem0_batch_turns)
    stored = results.get("system") or {}
    described = system.describe()
    if stored.get("version") not in (None, described["version"]):
        raise SystemExit(f"{out_path} retrieved with mem0 {stored['version']}; this venv has "
                         f"{described['version']}")
    results["system"] = described
    print(f"mem0 {described['version']}: extraction {described['llm_model']} (reasoning "
          f"{described['llm_reasoning_effort']}), embedder {described['embedder_model']} at "
          f"{described['base_url']}, features {json.dumps(described['features'])}")
    try:
        for cid in sorted({q["conversation"] for q in todo}):
            conv = data_by_id[cid]
            c = conv["conversation"]
            speakers = (c["speaker_a"], c["speaker_b"])
            user_id = user_for(cid)
            stats = system.ingest(user_id, conversation_sessions(conv), turn_content, print)
            results["ingest"][cid] = stats
            calls = stats["calls"]["llm"]
            print(f"  {cid}: {stats['turns']} turns in {stats['sessions']} sessions, "
                  f"{stats['adds']} adds -> {stats['stored_memories']} memories "
                  f"({stats['ingest_s']}s); extraction calls {calls['calls']}, errors "
                  f"{calls['errors']}, truncated {calls['truncated']}, unparseable "
                  f"{calls['unparseable']}, empty {calls['empty']}, nothing extracted "
                  f"{calls['nothing_extracted']}")
            if calls["truncated"] or calls["unparseable"] or calls["empty"]:
                print(f"    mem0 dropped the facts of {calls['truncated'] + calls['unparseable']}"
                      f" + {calls['empty']} extraction replies; raise --mem0-llm-max-tokens if "
                      "they were truncated", file=sys.stderr)
            unattributed = 0
            mine = [q for q in todo if q["conversation"] == cid]
            for q in mine:
                row = results["rows"].setdefault(q["qid"], {**q, "runs": []})
                row.update(system.search(user_id, q["question"], args.top_k, speakers))
                unattributed += row["unattributed"]
                settle_context(row, args.context_budget, count, maps_turns=False)
            stats["unattributed_memories"] = unattributed
            print(f"  {cid}: retrieved top-{args.top_k} for {len(mine)} questions; "
                  f"{unattributed} retrieved facts named neither speaker")
            save_results(results, out_path)
    finally:
        system.close()


def unit_state(run: dict) -> str:
    """judged, not_attempted (a fatal error stopped the run first), fatal, persistent (would
    recur on a resend: truncation, refusal) or retryable."""
    if run["label"] is not None:
        return "judged"
    if not run["calls"]:
        return "not_attempted"
    last = run["calls"][-1]
    if last.get("fatal"):
        return "fatal"
    if NOT_RETRIED.match(last.get("error") or ""):
        return "persistent"
    return "retryable"


def run_round(args, results: dict, qids: list[str], speakers: dict, out_path: Path, client,
              llm: dict, units: list, abort: threading.Event) -> int:
    rows = results["rows"]
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {}
    done = 0
    last_save = time.time()
    try:
        for qid, r in units:
            row = rows[qid]
            prompt = reader_prompt(row, speakers[row["conversation"]])
            futures[pool.submit(run_unit, client, llm, row, copy.deepcopy(row["runs"][r]),
                                prompt, abort)] = (qid, r)
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
    return done


def api_phase(args, results: dict, qids: list[str], speakers: dict, out_path: Path, client,
              llm: dict, pause=time.sleep):
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
    print(f"Running reader + judge for {len(units)} (question, run) pairs: reader "
          f"{llm['reader']['model']}, judge {llm['judge']['model']}, concurrency "
          f"{args.concurrency}, up to {args.attempts} attempts each")
    abort = threading.Event()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    done = 0
    rounds = 0
    pending = units
    try:
        while pending and rounds < args.attempts and not abort.is_set():
            if rounds:
                wait = RETRY_PAUSE_S * rounds
                print(f"Attempt {rounds + 1} of {args.attempts}: {len(pending)} (question, run) "
                      f"pairs failed with a retryable error; sending them again in {wait}s")
                pause(wait)
            done += run_round(args, results, qids, speakers, out_path, client, llm, pending,
                              abort)
            rounds += 1
            pending = [(qid, r) for qid, r in pending
                       if unit_state(rows[qid]["runs"][r]) == "retryable"]
    finally:
        wall_s = round(time.perf_counter() - started, 1)
        results.setdefault("api_phase", []).append({
            "started_at": started_at, "wall_s": wall_s, "units": len(units),
            "completed": done, "rounds": rounds, "concurrency": args.concurrency})
        print(f"API phase: {done} sends over {rounds} round(s) for {len(units)} (question, run) "
              f"pairs in {wall_s}s")
        save_results(results, out_path)


def default_output(dry_run: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "dryrun" if dry_run else "run"
    return CACHE_DIR / "results" / f"locomo_qa_{kind}_{stamp}.json"


def build_llm(args) -> dict:
    anthropic = args.provider == "anthropic"
    if not anthropic and not (args.model or (args.reader_model and args.judge_model)):
        raise SystemExit(f"--provider {args.provider} needs --model, or both --reader-model "
                         "and --judge-model")
    if args.provider == "lmstudio" and not args.base_url:
        raise SystemExit("--provider lmstudio needs --base-url, e.g. spine's tunnel "
                         "http://127.0.0.1:11234/v1")
    llm = {"provider": args.provider, "base_url": args.base_url, "prices": {}}
    for kind, model, effort, max_tokens in (
            ("reader", args.reader_model, args.reader_effort, args.reader_max_tokens),
            ("judge", args.judge_model, args.judge_effort, args.judge_max_tokens)):
        model = model or args.model or MODEL
        if anthropic:
            known = {}
        elif args.provider == "lmstudio":
            known = LMSTUDIO_DEFAULTS
        else:
            known = MODEL_DEFAULTS.get(model, {})
        if anthropic:
            default_effort = READER_EFFORT if kind == "reader" else JUDGE_EFFORT
        else:
            default_effort = known.get(f"{kind}_effort")
        llm[kind] = {"model": model, "effort": effort or default_effort,
                     "max_tokens": max_tokens, "reasoning_format": known.get("reasoning_format")}
        price = dict(PRICE_PER_MTOK if anthropic else known.get("price", {}))
        if args.price_in is not None:
            price["input_tokens"] = args.price_in
        if args.price_out is not None:
            price["output_tokens"] = args.price_out
        if not {"input_tokens", "output_tokens"} <= price.keys():
            raise SystemExit(f"No list price known for {model}; pass --price-in and --price-out "
                             "(USD per MTok)")
        llm["prices"][model] = price
    return llm


def llm_config(llm: dict, price_overridden: bool) -> dict:
    config = {
        "provider": llm["provider"], "base_url": llm["base_url"],
        "reader_model": llm["reader"]["model"], "judge_model": llm["judge"]["model"],
        "reader_effort": llm["reader"]["effort"], "judge_effort": llm["judge"]["effort"],
        "reader_max_tokens": llm["reader"]["max_tokens"],
        "judge_max_tokens": llm["judge"]["max_tokens"],
        "price_per_mtok": llm["prices"],
        "price_source": ("--price-in / --price-out" if price_overridden
                         else PRICE_SOURCE[llm["provider"]]),
    }
    if llm["provider"] == "anthropic":
        config["fallbacks"] = f"default ({FALLBACK_BETA})"
    else:
        config.update(temperature=0, judge_response_format=judge_format(llm["provider"])["type"],
                      reader_reasoning_format=llm["reader"]["reasoning_format"],
                      judge_reasoning_format=llm["judge"]["reasoning_format"],
                      max_retries=OPENAI_MAX_RETRIES)
    return config


LLM_IDENTITY = ("provider", "reader_model", "judge_model", "reader_effort", "judge_effort")


def stored_llm_identity(config: dict) -> tuple:
    stored = {"provider": "anthropic", "reader_model": config.get("model"),
              "judge_model": config.get("model"), **config}
    return tuple(stored.get(key) for key in LLM_IDENTITY)


def resolve_eval_now(raw: str | None, convs: list[dict]) -> str | None:
    if raw is None:
        return None
    if raw == "auto":
        latest = max(start + timedelta(seconds=len(turns) - 1)
                     for conv in convs for _, turns, start in conversation_sessions(conv))
        return rfc3339(latest + timedelta(days=1))
    try:
        pinned = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"--eval-now {raw!r} is neither ISO 8601 nor auto")
    if pinned.tzinfo is None:
        pinned = pinned.replace(tzinfo=timezone.utc)
    return rfc3339(pinned.astimezone(timezone.utc))


def main():
    ap = argparse.ArgumentParser(description="LoCoMo end-to-end QA accuracy (J) for veld and mem0")
    ap.add_argument("--system", default="veld", choices=["veld", "mem0"],
                    help="Memory system under test (mem0 runs from its own venv)")
    ap.add_argument("--label", default=None,
                    help="Free text recorded with the results, e.g. before or after a fix")
    ap.add_argument("--conversations", default=None,
                    help="Comma-separated sample ids, e.g. conv-26,conv-30 (default: all 10)")
    ap.add_argument("--limit-questions", type=int, default=0,
                    help="First N non-adversarial questions per conversation (0 = all)")
    ap.add_argument("--categories", default=None,
                    help=f"Comma-separated categories to ask, of {','.join(CATEGORY_ORDER)} "
                         "(default: all)")
    ap.add_argument("--event-dates", action="store_true",
                    help="Show each memory's mention date and the dates its content resolves "
                         "to (the recall response's occurred and planned)")
    ap.add_argument("--retrieval-from", default=None, metavar="RESULTS",
                    help="Reuse this results file's retrieval instead of spawning a server")
    ap.add_argument("--top-k", type=int, default=10, help="Memories recalled per question")
    ap.add_argument("--context-budget", type=int, default=None, metavar="TOKENS",
                    help=f"Keep the ranked prefix of the retrieved lines that fits this many "
                         f"{TOKENIZER} tokens (default: keep all top-k)")
    ap.add_argument("--runs", type=int, default=3,
                    help="Reader + judge repetitions over the same retrieved contexts")
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel API calls")
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                    help="Sends per (question, run) before a retryable error is left unjudged")
    ap.add_argument("--dry-run", action="store_true",
                    help="Spawn, ingest, retrieve and print the assembled prompts and a cost "
                         "projection; no reader or judge calls (mem0 ingest still calls its LLM)")
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
    ap.add_argument("--eval-now", default=None, metavar="ISO8601|auto",
                    help="Pin the scratch server's scoring clock (VELD_EVAL_NOW); auto is one "
                         "day after the latest session of the selected conversations (default: "
                         "unset, the live clock)")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai-compatible", "lmstudio"],
                    help="Reader and judge API (and, for mem0, its extraction LLM and embedder)")
    ap.add_argument("--base-url", default=None,
                    help="API base URL, e.g. https://api.groq.com/openai/v1")
    ap.add_argument("--model", default=None,
                    help=f"Reader and judge model (anthropic default {MODEL})")
    ap.add_argument("--reader-model", default=None, help="Reader model, overriding --model")
    ap.add_argument("--judge-model", default=None, help="Judge model, overriding --model")
    ap.add_argument("--reader-effort", default=None,
                    help="Reader effort / reasoning_effort (default per model)")
    ap.add_argument("--judge-effort", default=None,
                    help="Judge effort / reasoning_effort (default per model)")
    ap.add_argument("--reader-max-tokens", type=int, default=READER_MAX_TOKENS)
    ap.add_argument("--judge-max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    key_source = ap.add_mutually_exclusive_group()
    key_source.add_argument("--api-key-keychain", default=None, metavar="NAME",
                            help="Read the model API key from macOS keychain item NAME")
    key_source.add_argument("--api-key-env", default=None, metavar="NAME",
                            help="Read the model API key from environment variable NAME")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per MTok of input, for every model (default: list price)")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per MTok of output, for every model (default: list price)")
    mem0 = ap.add_argument_group("--system mem0")
    mem0.add_argument("--mem0-llm-model", default=None,
                      help="mem0's extraction LLM on the reader's host (default: the reader)")
    mem0.add_argument("--mem0-llm-effort", default="none",
                      help="reasoning_effort sent with every mem0 LLM call; 'default' sends none")
    mem0.add_argument("--mem0-llm-max-tokens", type=int, default=6000)
    mem0.add_argument("--mem0-embedder-model", default="text-embedding-nomic-embed-text-v1.5",
                      help="mem0's embedder on the reader's host (default: nomic-embed-text "
                           "v1.5, the model veld embeds with on this machine)")
    mem0.add_argument("--mem0-embedding-dims", type=int, default=768)
    mem0.add_argument("--mem0-batch-turns", type=int, default=0,
                      help="Turns per mem0 add() within a session (0 = the whole session)")
    args = ap.parse_args()
    llm = build_llm(args)
    if args.eval_now and args.no_spawn:
        raise SystemExit("--eval-now sets the spawned server's clock; with --no-spawn, start "
                         "that server with VELD_EVAL_NOW instead")
    if args.system == "mem0" and args.provider == "anthropic":
        raise SystemExit("--system mem0 points mem0's extraction LLM and embedder at the "
                         "reader's host; use --provider lmstudio or openai-compatible")
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    if args.system == "mem0" and (args.event_dates or args.retrieval_from):
        raise SystemExit("--event-dates and --retrieval-from read veld recall responses")
    categories = args.categories.split(",") if args.categories else list(CATEGORY_ORDER)
    unknown_categories = sorted(set(categories) - set(CATEGORY_ORDER))
    if unknown_categories:
        raise SystemExit(f"Unknown categories {unknown_categories}; known: {CATEGORY_ORDER}")
    global RENDER_EVENT_DATES
    RENDER_EVENT_DATES = args.event_dates

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
        qs = [q for q in qs if q["category"] in categories]
        questions.extend(qs[:args.limit_questions] if args.limit_questions > 0 else qs)
    every = [q for conv in data for q in conversation_questions(conv)]
    full_questions = len(every)
    speakers = {cid: (data_by_id[cid]["conversation"]["speaker_a"],
                      data_by_id[cid]["conversation"]["speaker_b"]) for cid in data_by_id}
    print(f"LoCoMo {dataset_path} sha256 {dataset_sha}: {len(questions)} questions selected "
          f"from {len(wanted)} conversations ({full_questions} non-adversarial in the dataset, "
          f"{sum(q['in_retrieval_set'] for q in every)} of them in the retrieval harness's set)")
    eval_now = resolve_eval_now(args.eval_now, [data_by_id[c] for c in wanted])

    out_path = Path(args.output).expanduser() if args.output else default_output(args.dry_run)
    if out_path.exists() and not args.resume:
        raise SystemExit(f"{out_path} exists; pass --resume to continue it or choose another "
                         "--output.")
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        cfg = results["config"]
        if results["dataset"]["sha256"] != dataset_sha:
            raise SystemExit(f"{out_path} was made from a different locomo10.json")
        if cfg["top_k"] != args.top_k:
            raise SystemExit(f"{out_path} retrieved top-{cfg['top_k']}; "
                             f"rerun with --top-k {cfg['top_k']}")
        if cfg.get("system", "veld") != args.system:
            raise SystemExit(f"{out_path} measured --system {cfg.get('system', 'veld')}")
        if cfg.get("context_budget") != args.context_budget:
            raise SystemExit(f"{out_path} kept a context budget of {cfg.get('context_budget')}; "
                             f"rerun with --context-budget {cfg.get('context_budget')}")
        stored = stored_llm_identity(cfg)
        current = stored_llm_identity(llm_config(llm, False))
        if stored != current:
            raise SystemExit(f"{out_path} was judged with {dict(zip(LLM_IDENTITY, stored))}; "
                             f"this command asks for {dict(zip(LLM_IDENTITY, current))}")
        if cfg.get("eval_now") != eval_now:
            raise SystemExit(f"{out_path} retrieved with --eval-now {cfg.get('eval_now')}; this "
                             f"command resolves to {eval_now}")
        if cfg.get("event_dates", False) != args.event_dates:
            raise SystemExit(f"{out_path} was read with --event-dates "
                             f"{cfg.get('event_dates', False)}; rerun to match")
        cfg["runs"] = max(cfg["runs"], args.runs)
        results["dry_run"] = results.get("dry_run", False) and args.dry_run
        results.setdefault("system", {"name": "veld"})
        print(f"Resuming {out_path}")
    else:
        results = {
            "benchmark": "LoCoMo end-to-end QA accuracy (J)",
            "label": args.label,
            "comparability_note": comparability_note(llm),
            "dataset": {"url": DATASET_URL, "path": str(dataset_path), "sha256": dataset_sha},
            "config": {
                "system": args.system, "top_k": args.top_k,
                "context_budget": args.context_budget, "tokenizer": TOKENIZER,
                "recall_mode": ("hybrid (server default)" if args.system == "veld"
                                else "Memory.search (mem0 default)"),
                "runs": args.runs,
                "categories": {str(k): v for k, v in CATEGORY.items()},
                "excluded_categories": sorted(EXCLUDED_CATEGORIES),
                "answer_prompt": ANSWER_PROMPT_SOURCE, "judge_prompt": ACCURACY_PROMPT_SOURCE,
                "session_timezone": "UTC",
                "event_dates": args.event_dates,
            },
            "system": {"name": args.system},
            "veld": {"git": git_describe()},
            "ingest": {},
            "rows": {},
            "dry_run": args.dry_run,
        }
    results["config"].update(llm_config(llm, args.price_in is not None
                                        or args.price_out is not None))
    results["config"].update(eval_now=eval_now, eval_now_arg=args.eval_now,
                             attempts=args.attempts, conversations=wanted,
                             limit_questions=args.limit_questions, categories_asked=categories)
    if args.retrieval_from and not results["rows"]:
        source_path = Path(args.retrieval_from).expanduser()
        with open(source_path, encoding="utf-8") as f:
            source = json.load(f)
        source_cfg = source["config"]
        if source["dataset"]["sha256"] != dataset_sha:
            raise SystemExit(f"{source_path} was made from a different locomo10.json")
        for key, mine in (("top_k", args.top_k), ("eval_now", eval_now),
                          ("system", args.system)):
            if source_cfg.get(key) != mine:
                raise SystemExit(f"{source_path} has {key} {source_cfg.get(key)!r}, this run "
                                 f"{mine!r}")
        seed_retrieval(results, source, [q["qid"] for q in questions], source_path)
        results["config"]["retrieval_from"] = {"path": str(source_path),
                                               "label": source.get("label")}
        print(f"Retrieval reused from {source_path}")
    if args.label is not None:
        results["label"] = args.label
    needs_key = not args.dry_run or args.system == "mem0"
    api_key = llm_api_key(args) if needs_key else None
    client = None if args.dry_run else make_client(llm, api_key)
    qids = [q["qid"] for q in questions]
    count = token_counter()

    for q in questions:
        row = results["rows"].get(q["qid"])
        if row is None:
            continue
        row.update({k: q[k] for k in ("evidence", "evidence_raw", "in_retrieval_set")})
        if row.get("retrieved") is not None and "context_n" not in row:
            settle_context(row, args.context_budget, count, maps_turns=args.system == "veld")

    todo = [q for q in questions if results["rows"].get(q["qid"], {}).get("retrieved") is None]
    if todo:
        retrieval_phase(args, data_by_id, results, todo, out_path, llm, count, api_key)
        save_results(results, out_path)
    else:
        print("All selected questions already carry their retrieval; no server needed.")
    results["comparability_note"] = comparability_note(llm, results.get("system"))

    if args.dry_run:
        rows = [results["rows"][q] for q in qids]
        for row in rows[:args.show_prompts]:
            print(f"\n----- reader prompt, {row['qid']} ({row['category']}) -----")
            print(reader_prompt(row, speakers[row["conversation"]]))
            print(f"----- judge prompt, {row['qid']} (answer placeholder) -----")
            print(judge_prompt(row, "<reader answer>"))
        results["projection"] = project(rows, speakers, args.runs, full_questions, llm)
        save_results(results, out_path)
        print(f"\nDry run written to {out_path}")
        return

    api_phase(args, results, qids, speakers, out_path, client, llm)
    results["aggregate"] = summarise(results, qids, results["config"]["runs"])
    save_results(results, out_path)
    print_summary(results, results["aggregate"])
    print(f"\nResults: {out_path}")
    if results["status"] != "complete":
        sys.exit(EXIT_INCOMPLETE)


if __name__ == "__main__":
    main()
