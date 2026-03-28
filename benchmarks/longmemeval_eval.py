#!/usr/bin/env python3
"""
LongMemEval Benchmark Evaluation for shodh-memory

Runs the official LongMemEval benchmark (ICLR 2025, 500 questions, 6 types)
against shodh-memory's retrieval pipeline. Produces reproducible, publishable
results with content hash verification.

Protocol:
  1. Download dataset from HuggingFace (cached)
  2. For each question: ingest sessions -> retrieve -> generate answer -> judge
  3. Report retrieval metrics (Recall@k, NDCG@k) and QA accuracy per type

Requirements:
    pip install -r requirements-longmemeval.txt
    # shodh-memory server must be running on localhost:3030

Usage:
    # Quick smoke test
    python longmemeval_eval.py --split s --limit 5

    # Full evaluation
    python longmemeval_eval.py --split s --answer-model gpt-4o

    # With Claude for answers, GPT-4o for judge
    python longmemeval_eval.py --split s \\
        --answer-provider anthropic --answer-model claude-sonnet-4-20250514

Reference: https://github.com/xiaowu0162/LongMemEval
Dataset:   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# Load .env file if present (for API keys)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import backoff
    HAS_BACKOFF = True
except ImportError:
    HAS_BACKOFF = False

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
SPLITS = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
]

RECALL_K_VALUES = [1, 3, 5, 10, 30, 50]

# Map question_id prefixes to canonical types for metrics
TASK2TYPE = {
    "single_hop": "single-session-user",
    "assistant_previnfo": "single-session-assistant",
    "two_hop": "multi-session",
    "multi_session_synthesis": "multi-session",
    "knowledge_update": "knowledge-update",
    "temp_reasoning_explicit": "temporal-reasoning",
    "temp_reasoning_implicit": "temporal-reasoning",
    "implicit_preference_v2": "single-session-preference",
}


# =============================================================================
# SHODH-MEMORY CLIENT
# =============================================================================

class ShodhMemoryClient:
    def __init__(self, base_url="http://127.0.0.1:3030",
                 api_key="sk-shodh-dev-local-testing-key"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        })

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def remember_batch(self, user_id: str, memories: list,
                       extract_entities: bool = False) -> dict:
        chunks = [memories[i:i+900] for i in range(0, len(memories), 900)]
        total_created = 0
        all_ids = []
        all_errors = []
        for chunk in chunks:
            payload = {
                "user_id": user_id,
                "memories": chunk,
                "extract_entities": extract_entities,
                "create_edges": False,
            }
            r = self.session.post(f"{self.base_url}/api/remember/batch",
                                 json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            total_created += data.get("created", 0)
            all_ids.extend(data.get("memory_ids", []))
            all_errors.extend(data.get("errors", []))
        return {"created": total_created, "memory_ids": all_ids, "errors": all_errors}

    def recall(self, user_id: str, query: str, limit: int = 50,
               mode: str = "hybrid") -> list:
        payload = {"user_id": user_id, "query": query,
                   "limit": limit, "mode": mode}
        r = self.session.post(f"{self.base_url}/api/recall",
                              json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("memories", [])

    def delete_user(self, user_id: str) -> bool:
        try:
            r = self.session.delete(f"{self.base_url}/api/users/{user_id}",
                                    timeout=30)
            return r.status_code in (200, 204, 404)
        except Exception:
            return False


# =============================================================================
# LLM PROVIDERS
# =============================================================================

def llm_complete(provider: str, model: str, prompt: str,
                 temperature: float = 0, max_tokens: int = 500,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None) -> str:
    if provider == "openai" or provider == "openai-compatible":
        return _openai_complete(model, prompt, temperature, max_tokens,
                                api_key, api_base)
    elif provider == "anthropic":
        return _anthropic_complete(model, prompt, temperature, max_tokens,
                                   api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _openai_complete(model, prompt, temperature, max_tokens,
                     api_key=None, api_base=None):
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=api_base,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _anthropic_complete(model, prompt, temperature, max_tokens, api_key=None):
    import anthropic
    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _llm_with_retry(provider, model, prompt, temperature=0, max_tokens=500,
                     api_key=None, api_base=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            return llm_complete(provider, model, prompt, temperature,
                                max_tokens, api_key, api_base)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt * 2, 60)
            print(f"  LLM error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


# =============================================================================
# DATASET LOADING
# =============================================================================

def download_dataset(split: str = "s", cache_dir: str = None) -> Path:
    filename = SPLITS[split]
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/longmemeval")
    os.makedirs(cache_dir, exist_ok=True)
    cached = Path(cache_dir) / filename

    if cached.exists():
        print(f"Using cached dataset: {cached}")
        return cached

    print(f"Downloading {filename} from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            local_dir=cache_dir,
        )
        return Path(path)
    except ImportError:
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{filename}"
        print(f"  huggingface_hub not installed, downloading via requests...")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(cached, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved to {cached}")
        return cached


def load_dataset(path: Path) -> list:
    print(f"Loading dataset from {path} ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} questions")
    return data


def parse_longmemeval_date(date_str: str) -> str:
    """Parse 'YYYY/MM/DD (Day) HH:MM' to ISO 8601."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y/%m/%d (%a) %H:%M")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        try:
            dt = datetime.strptime(date_str.strip()[:16], "%Y/%m/%d (%a) %")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            return datetime.now(timezone.utc).isoformat()


def get_question_type(entry: dict) -> str:
    """Resolve canonical question type from entry."""
    if "question_type" in entry and entry["question_type"]:
        return entry["question_type"]
    qid = entry.get("question_id", "")
    for prefix, qtype in TASK2TYPE.items():
        if qid.startswith(prefix):
            return qtype
    return "unknown"


def is_abstention(entry: dict) -> bool:
    qid = entry.get("question_id", "")
    return qid.endswith("_abs") or "_abs_" in qid


# =============================================================================
# INGESTION
# =============================================================================

def ingest_question(client: ShodhMemoryClient, entry: dict,
                    user_id: str, granularity: str = "turn") -> dict:
    """Ingest all haystack sessions for one question into shodh-memory."""
    client.delete_user(user_id)
    time.sleep(0.1)  # Brief pause for cleanup

    sessions = entry.get("haystack_sessions", [])
    session_ids = entry.get("haystack_session_ids", [])
    dates = entry.get("haystack_dates", [])

    memories = []
    corpus_id_map = {}  # memory_index -> corpus_id

    for s_idx, session_turns in enumerate(sessions):
        sid = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"
        date_str = dates[s_idx] if s_idx < len(dates) else ""
        created_at = parse_longmemeval_date(date_str) if date_str else None

        if granularity == "turn":
            # Ingest user+assistant turn pairs for conversational context.
            # Each memory = one user turn with its preceding/following assistant
            # response, giving the LLM full dialogue context at retrieval.
            user_turn_counter = 0
            for t_idx, turn in enumerate(session_turns):
                role = turn.get("role", "human")
                content = turn.get("content", "")
                if not content.strip():
                    continue
                if role in ("human", "user"):
                    # Include the assistant response that follows (if any)
                    parts = [f"User: {content}"]
                    if t_idx + 1 < len(session_turns):
                        next_turn = session_turns[t_idx + 1]
                        if next_turn.get("role") in ("assistant", "gpt"):
                            next_content = next_turn.get("content", "")
                            if next_content.strip():
                                # Truncate long assistant responses
                                if len(next_content) > 500:
                                    next_content = next_content[:500] + "..."
                                parts.append(f"Assistant: {next_content}")

                    corpus_id = f"{sid}_turn_{user_turn_counter}"
                    mem = {
                        "content": f"[{date_str}]\n" + "\n".join(parts),
                        "memory_type": "Conversation",
                        "tags": [sid, f"turn_{user_turn_counter}",
                                 f"tidx_{t_idx}"],
                    }
                    if created_at:
                        mem["created_at"] = created_at
                    corpus_id_map[len(memories)] = corpus_id
                    memories.append(mem)
                    user_turn_counter += 1
        elif granularity == "session":
            # Concatenate full dialogue (both roles) for session-level retrieval
            all_turns = []
            for t in session_turns:
                role = "User" if t.get("role") in ("human", "user") else "Assistant"
                content = t.get("content", "").strip()
                if content:
                    if role == "Assistant" and len(content) > 500:
                        content = content[:500] + "..."
                    all_turns.append(f"{role}: {content}")
            if not all_turns:
                continue
            content = f"[{date_str}]\n" + "\n".join(all_turns)
            corpus_id = sid
            mem = {
                "content": content,
                "memory_type": "Conversation",
                "tags": [sid],
            }
            if created_at:
                mem["created_at"] = created_at
            corpus_id_map[len(memories)] = corpus_id
            memories.append(mem)

    if not memories:
        return {"count": 0, "corpus_id_map": {}, "memory_ids": []}

    t0 = time.time()
    result = client.remember_batch(user_id, memories, extract_entities=False)
    elapsed = (time.time() - t0) * 1000

    # Build memory_id -> corpus_id map
    mid_to_cid = {}
    for i, mid in enumerate(result.get("memory_ids", [])):
        if i in corpus_id_map:
            mid_to_cid[mid] = corpus_id_map[i]

    return {
        "count": result["created"],
        "corpus_id_map": mid_to_cid,
        "memory_ids": result.get("memory_ids", []),
        "ingest_ms": elapsed,
        "errors": result.get("errors", []),
    }


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

def dcg(relevances, k):
    r = np.asarray(relevances[:k], dtype=float)
    if r.size == 0:
        return 0.0
    if r.size == 1:
        return float(r[0])
    return float(r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))))


def ndcg_score(retrieved_ids, correct_ids, k):
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in retrieved_ids[:k]]
    actual = dcg(relevances, k)
    ideal_rels = sorted([1.0] * min(len(correct_ids), k) +
                        [0.0] * max(0, k - len(correct_ids)), reverse=True)
    ideal = dcg(ideal_rels, k)
    return actual / ideal if ideal > 0 else 0.0


def compute_retrieval_metrics(retrieved_corpus_ids: list,
                              answer_session_ids: list,
                              granularity: str = "turn") -> dict:
    """Compute Recall@k and NDCG@k for one question."""
    if granularity == "turn":
        # Aggregate turn-level IDs to session-level
        retrieved_sessions = []
        seen = set()
        for cid in retrieved_corpus_ids:
            sid = cid.rsplit("_turn_", 1)[0] if "_turn_" in cid else cid
            if sid not in seen:
                seen.add(sid)
                retrieved_sessions.append(sid)
    else:
        retrieved_sessions = retrieved_corpus_ids

    correct = set(answer_session_ids)
    metrics = {}
    for k in RECALL_K_VALUES:
        top_k = set(retrieved_sessions[:k])
        recall_any = float(bool(top_k & correct))
        recall_all = float(correct.issubset(top_k)) if correct else 1.0
        ndcg = ndcg_score(retrieved_sessions, correct, k)
        metrics[f"recall_any@{k}"] = recall_any
        metrics[f"recall_all@{k}"] = recall_all
        metrics[f"ndcg_any@{k}"] = ndcg

    return metrics


# =============================================================================
# ANSWER GENERATION
# =============================================================================

ANSWER_PROMPT = (
    "I will give you several history chats between you and a user. "
    "Please answer the question based on the relevant chat history.\n\n"
    "Important: Pay close attention to the question word.\n"
    "- 'Where' questions need a specific PLACE, STORE, or LOCATION name.\n"
    "- 'When' questions need a specific DATE, TIME, or DAY.\n"
    "- 'Who' questions need a specific PERSON name.\n"
    "- 'How long/much/many' questions need a specific NUMBER or QUANTITY.\n\n"
    "Extract the most specific, concrete answer from the chat history. "
    "If multiple pieces of context are relevant, combine them.\n\n"
    "History Chats:\n\n{history}\n\n"
    "Current Date: {question_date}\n"
    "Question: {question}\n"
    "Answer:"
)


def _get_mem_content(mem: dict) -> str:
    """Extract content from recall response (handles nested experience)."""
    if "experience" in mem:
        return mem["experience"].get("content", "")
    return mem.get("content", "")


def _get_mem_tags(mem: dict) -> list:
    """Extract tags from recall response (handles nested experience)."""
    if "experience" in mem:
        return mem["experience"].get("tags", [])
    return mem.get("tags", [])


def format_retrieved_context(memories: list, max_tokens: int = 12000) -> str:
    """Format retrieved memories as session-grouped context."""
    # Group by session tag (session IDs vary: session_*, answer_*, ultrachat_*, sharegpt_*, etc.)
    sessions = defaultdict(list)
    for mem in memories:
        tags = _get_mem_tags(mem)
        # The first tag is always the session_id (set during ingestion)
        # Skip tags that are clearly turn markers or extracted entities
        sid = "unknown"
        for t in tags:
            if t.startswith("turn_") or t.startswith("tidx_"):
                continue
            # First non-turn tag is the session ID
            sid = t
            break
        sessions[sid].append(mem)

    parts = []
    total_chars = 0
    char_limit = max_tokens * 4  # rough chars-to-tokens

    for i, (sid, mems) in enumerate(sessions.items()):
        section = f"### Chat Session {i+1}:\n"
        for mem in mems:
            content = _get_mem_content(mem)
            section += content + "\n"
        if total_chars + len(section) > char_limit:
            break
        parts.append(section)
        total_chars += len(section)

    return "\n".join(parts)


def generate_answer(provider: str, model: str, question: str,
                    question_date: str, memories: list,
                    api_key: str = None, api_base: str = None) -> str:
    context = format_retrieved_context(memories)
    prompt = ANSWER_PROMPT.format(
        history=context,
        question_date=question_date,
        question=question,
    )
    return _llm_with_retry(provider, model, prompt, temperature=0,
                           max_tokens=500, api_key=api_key, api_base=api_base)


# =============================================================================
# LLM-AS-JUDGE (Official LongMemEval prompts)
# =============================================================================

JUDGE_PROMPT_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_PROMPT_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_PROMPT_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. If the response contains some previous information "
    "along with an updated answer, the response should be considered as correct "
    "as long as the updated answer is the required answer.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_PROMPT_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\n"
    "Desired Response Rubric: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Does the model response satisfy the rubric? Answer yes or no only."
)

JUDGE_PROMPT_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not. If the model gives a concrete answer (even if it is wrong), "
    "answer no.\n\n"
    "Unanswerable Question: {question}\n\n"
    "Explanation: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def get_judge_prompt(question_type: str, abstention: bool,
                     question: str, answer: str, response: str) -> str:
    if abstention:
        template = JUDGE_PROMPT_ABSTENTION
    elif question_type == "temporal-reasoning":
        template = JUDGE_PROMPT_TEMPORAL
    elif question_type == "knowledge-update":
        template = JUDGE_PROMPT_KNOWLEDGE_UPDATE
    elif question_type == "single-session-preference":
        template = JUDGE_PROMPT_PREFERENCE
    else:
        template = JUDGE_PROMPT_STANDARD

    return template.format(question=question, answer=answer, response=response)


def judge_answer(question_type: str, abstention: bool,
                 question: str, gold_answer: str, hypothesis: str,
                 judge_model: str = "gpt-4o",
                 api_key: str = None) -> tuple:
    """Returns (correct: bool, raw_judge_response: str)."""
    prompt = get_judge_prompt(question_type, abstention,
                              question, gold_answer, hypothesis)
    response = _llm_with_retry("openai", judge_model, prompt,
                                temperature=0, max_tokens=10,
                                api_key=api_key)
    correct = "yes" in response.lower()
    return correct, response


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def run_evaluation(args):
    # Download and load dataset
    dataset_path = download_dataset(args.split, args.cache_dir)
    dataset = load_dataset(dataset_path)

    if args.limit and args.limit > 0:
        dataset = dataset[:args.limit]

    # Load checkpoint if resuming
    completed = {}
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            checkpoint = json.load(f)
        for r in checkpoint.get("per_question", []):
            completed[r["question_id"]] = r
        print(f"Resuming from checkpoint: {len(completed)} questions already done")

    # Health check
    client = ShodhMemoryClient(args.shodh_url, args.api_key)
    if not client.health():
        print(f"ERROR: shodh-memory server not reachable at {args.shodh_url}")
        print("Start with: cargo run -- serve")
        sys.exit(1)
    print(f"shodh-memory server OK at {args.shodh_url}")

    # Run evaluation
    results = []
    for entry in tqdm(dataset, desc="LongMemEval"):
        qid = entry.get("question_id", f"q_{len(results)}")
        if qid in completed:
            results.append(completed[qid])
            continue

        question = entry["question"]
        gold_answer = entry["answer"]
        question_date = entry.get("question_date", "")
        question_type = get_question_type(entry)
        abstention = is_abstention(entry)
        answer_session_ids = entry.get("answer_session_ids", [])
        user_id = f"longmemeval_{qid}"

        # Phase 1: Ingest
        t_start = time.time()
        ingest_result = ingest_question(client, entry, user_id,
                                        args.granularity)
        t_ingest = (time.time() - t_start) * 1000

        # Phase 2: Retrieve
        t_start = time.time()
        memories = client.recall(user_id, question,
                                 limit=args.retrieval_limit, mode="hybrid")
        t_recall = (time.time() - t_start) * 1000

        # Map retrieved memories to corpus IDs for retrieval metrics
        mid_to_cid = ingest_result["corpus_id_map"]
        retrieved_corpus_ids = []
        for mem in memories:
            mid = mem.get("id", "")
            if mid in mid_to_cid:
                retrieved_corpus_ids.append(mid_to_cid[mid])
            else:
                tags = _get_mem_tags(mem)
                # First non-turn tag is the session ID
                for t in tags:
                    if not t.startswith("turn_") and not t.startswith("tidx_"):
                        retrieved_corpus_ids.append(t)
                        break

        # Retrieval metrics (skip for abstention)
        ret_metrics = {}
        if not abstention and answer_session_ids:
            ret_metrics = compute_retrieval_metrics(
                retrieved_corpus_ids, answer_session_ids, args.granularity)

        # Phase 3: Generate answer
        t_start = time.time()
        try:
            hypothesis = generate_answer(
                args.answer_provider, args.answer_model,
                question, question_date, memories,
                api_key=args.answer_api_key, api_base=args.answer_api_base)
        except Exception as e:
            print(f"  Answer generation failed for {qid}: {e}")
            hypothesis = "I don't have enough information to answer this question."
        t_generate = (time.time() - t_start) * 1000

        # Phase 4: Judge
        t_start = time.time()
        try:
            correct, judge_raw = judge_answer(
                question_type, abstention, question, gold_answer, hypothesis,
                judge_model=args.judge_model, api_key=args.judge_api_key)
        except Exception as e:
            print(f"  Judge failed for {qid}: {e}")
            correct, judge_raw = False, f"ERROR: {e}"
        t_judge = (time.time() - t_start) * 1000

        result = {
            "question_id": qid,
            "question_type": question_type,
            "abstention": abstention,
            "question": question,
            "answer": gold_answer,
            "hypothesis": hypothesis,
            "correct": correct,
            "judge_raw": judge_raw,
            "retrieval_metrics": ret_metrics,
            "num_ingested": ingest_result["count"],
            "num_retrieved": len(memories),
            "latency": {
                "ingest_ms": round(t_ingest, 1),
                "recall_ms": round(t_recall, 1),
                "generate_ms": round(t_generate, 1),
                "judge_ms": round(t_judge, 1),
            },
        }
        results.append(result)

        # Cleanup
        client.delete_user(user_id)

        # Checkpoint every 25 questions
        if len(results) % 25 == 0:
            _save_checkpoint(results, args)

    # Aggregate and output
    output = aggregate_results(results, args)
    output_path = args.output or _default_output_path(args)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"Content hash: {output['content_hash']}")

    print_summary(output)


# =============================================================================
# AGGREGATION AND OUTPUT
# =============================================================================

def aggregate_results(results: list, args) -> dict:
    # Per-type accuracy
    by_type = defaultdict(list)
    abstention_results = []
    for r in results:
        if r["abstention"]:
            abstention_results.append(r["correct"])
        else:
            by_type[r["question_type"]].append(r["correct"])

    type_metrics = {}
    for qt in QUESTION_TYPES:
        vals = by_type.get(qt, [])
        type_metrics[qt] = {
            "accuracy": float(np.mean(vals)) if vals else 0.0,
            "n": len(vals),
            "correct": sum(vals),
        }

    non_abs = [r["correct"] for r in results if not r["abstention"]]
    overall_accuracy = float(np.mean(non_abs)) if non_abs else 0.0
    task_averaged = float(np.mean([m["accuracy"] for m in type_metrics.values()
                                   if m["n"] > 0]))
    abstention_accuracy = float(np.mean(abstention_results)) if abstention_results else 0.0

    # Retrieval metrics (aggregate)
    ret_agg = defaultdict(list)
    for r in results:
        if not r["abstention"]:
            for k, v in r.get("retrieval_metrics", {}).items():
                ret_agg[k].append(v)
    retrieval_metrics = {k: float(np.mean(v)) for k, v in ret_agg.items()}

    # Content hash
    hash_input = json.dumps(
        [{"qid": r["question_id"], "correct": r["correct"]} for r in results],
        sort_keys=True,
    )
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    git_hash = "unknown"
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    return {
        "benchmark": "LongMemEval",
        "version": "0.5.0",
        "build": git_hash,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "dataset_split": args.split,
        "content_hash": content_hash,
        "config": {
            "retrieval_limit": args.retrieval_limit,
            "retrieval_mode": "hybrid",
            "granularity": args.granularity,
            "answer_provider": args.answer_provider,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
        },
        "qa_metrics": {
            "overall_accuracy": round(overall_accuracy, 4),
            "task_averaged_accuracy": round(task_averaged, 4),
            "abstention_accuracy": round(abstention_accuracy, 4),
            "total_questions": len(results),
            "total_non_abstention": len(non_abs),
            "total_abstention": len(abstention_results),
            "by_type": type_metrics,
        },
        "retrieval_metrics": retrieval_metrics,
        "per_question": results,
    }


def _save_checkpoint(results, args):
    path = args.output or _default_output_path(args)
    checkpoint_path = path.replace(".json", "_checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump({"per_question": results}, f)


def _default_output_path(args):
    git_hash = "unknown"
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return f"evaluations/longmemeval_v0.5.0_{args.split}_{git_hash}.json"


def print_summary(output: dict):
    qa = output["qa_metrics"]
    ret = output["retrieval_metrics"]

    print("\n" + "=" * 70)
    print("  LongMemEval Results — shodh-memory v0.5.0")
    print("=" * 70)

    print(f"\n  Overall Accuracy:       {qa['overall_accuracy']:.1%}")
    print(f"  Task-Averaged Accuracy: {qa['task_averaged_accuracy']:.1%}")
    print(f"  Abstention Accuracy:    {qa['abstention_accuracy']:.1%}")

    print(f"\n  {'Type':<30} {'Acc':>8} {'N':>6}")
    print("  " + "-" * 46)
    for qt in QUESTION_TYPES:
        m = qa["by_type"].get(qt, {})
        acc = m.get("accuracy", 0)
        n = m.get("n", 0)
        print(f"  {qt:<30} {acc:>7.1%} {n:>6}")

    if ret:
        print(f"\n  Retrieval Metrics:")
        for k in sorted(ret.keys()):
            print(f"    {k:<20} {ret[k]:.3f}")

    print(f"\n  Content Hash: {output['content_hash']}")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark for shodh-memory")
    parser.add_argument("--split", default="s", choices=["s", "m", "oracle"],
                        help="Dataset split (s=small ~40 sessions, m=medium ~500)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of questions (0=all)")
    parser.add_argument("--granularity", default="turn",
                        choices=["turn", "session"],
                        help="Ingestion granularity")
    parser.add_argument("--retrieval-limit", type=int, default=50,
                        help="Number of memories to retrieve per query")

    parser.add_argument("--answer-provider", default="openai",
                        choices=["openai", "anthropic", "openai-compatible"])
    parser.add_argument("--answer-model", default="gpt-4o")
    parser.add_argument("--answer-api-key", default=None)
    parser.add_argument("--answer-api-base", default=None)

    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-api-key", default=None)

    parser.add_argument("--shodh-url", default="http://127.0.0.1:3030")
    parser.add_argument("--api-key", default="sk-shodh-dev-local-testing-key")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint JSON")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
