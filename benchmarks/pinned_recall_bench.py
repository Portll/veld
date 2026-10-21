#!/usr/bin/env python3
"""Pinned LoCoMo-MC10 evidence-recall harness for veld.

This is the STABLE, no-noise recall harness used for signal-ablation and
embedder/coverage work. Every config must run this identical setup so that
cross-config deltas are comparable against the measured noise band (~0.34pp
ingest nondeterminism, ~0.68pp band).

Pinned protocol
---------------
* Dataset: HuggingFace `Percena/locomo-mc10`, snapshot
  `7d59a0463d83f97b042684310c0b3d17553004cd` (1,986 questions over exactly
  10 conversations). The snapshot hash is pinned below; a different snapshot
  triggers a loud warning.
* Corpus: the 10 conversations sorted by ID
  (conv-26, conv-30, conv-41, conv-42, conv-43, conv-44, conv-47, conv-48,
  conv-49, conv-50 — i.e. the whole MC10 set), each ingested under its own
  per-conversation user_id (`locomo_<conv-id>`).
* Ingest (mirrors benchmarks/locomo_mc10_eval.py exactly): per session,
  one summary memory (`Context`) + dialogue-turn-aware ~800-char chunks
  (`Conversation`), each prefixed with
  `[Conversation on <Month DD, YYYY at HH:MM AM/PM>] Session N:` derived from
  `haystack_session_datetimes`. Turns are never split mid-turn.
* Gold sessions per question: the sessions of the question's own LoCoMo
  evidence turns (`relevant` in tests/recall/locomo_cases.jsonl, D<n> = session
  n). Before scoring, the run checks that each evidence turn's text appears in
  the dialogue it stored for that session, and records the mismatches.
* Metric population: LoCoMo's temporal category, 320 questions, every one
  locatable. MC10 labels these questions `multi_hop`; the harness used that
  label until 2026-10-21.

Gold labels before 2026-10-21 were sessions whose dialogue contained every
answer token as a substring. Date answers matched the `[Conversation on ...]`
header of whichever session carried the date, so only 87 of the 147
"locatable" questions shared a session with their real evidence, and 174 of
321 counted as unlocatable. Numbers measured with those labels (the ~0.76
anchor below) do not compare with numbers measured with these.
* Warmup: 5 queries (conv-26 q0/q6/q10/q12/q16) issued and DISCARDED before
  any scored query.
* Score: evidence recall@10 — a question is a hit when any retrieved memory's
  `Session (\\d+)` header intersects the question's gold sessions.
  Recall mode=hybrid, k=10.

Superseded baseline (answer-token labels, multi_hop/temporal population of 147):
real MiniLM-L6-384 primary embedder, rerank OFF (`use_reranking: false`) →
evidence recall ~0.76 +/- 0.02. Not comparable with evidence labels.

Usage
-----
    # 1. Emit the durable gold labels (from tests/recall/locomo_cases.jsonl, no server
    #    and no dataset download):
    python3 benchmarks/pinned_recall_bench.py --emit-gold

    # 2. Full run, spawning a fresh server on a disposable store:
    VELD_API_KEYS=sk-veld-dev-local-testing-key \\
    ORT_DYLIB_PATH=... VELD_MODEL_PATH=...  \\
    python3 benchmarks/pinned_recall_bench.py \\
        --binary target/release/veld --port 4001 \\
        --store /tmp/veld-pinned-bench \\
        --dials '{"use_reranking": false}' \\
        --output /tmp/pinned_recall_results.json

    # Against an already-running server (no spawn):
    python3 benchmarks/pinned_recall_bench.py --port 4001 --no-spawn
"""

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

PINNED_SNAPSHOT = "7d59a0463d83f97b042684310c0b3d17553004cd"
TARGET_CHUNK_SIZE = 800
RECALL_K = 10
RECALL_MODE = "hybrid"
METRIC_CATEGORY = "temporal"
WARMUP_QIDS = ["conv-26_q0", "conv-26_q6", "conv-26_q10", "conv-26_q12", "conv-26_q16"]
SESSION_RE = re.compile(r"Session (\d+)")
DEFAULT_GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "pinned_gold_labels.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCOMO_CASES_PATH = os.path.join(REPO_ROOT, "tests", "recall", "locomo_cases.jsonl")
LOCOMO_CORPUS_PATH = os.path.join(REPO_ROOT, "tests", "recall", "corpora", "locomo.jsonl")
EVIDENCE_SESSION_RE = re.compile(r"^conv-\d+:D(\d+):\d+$")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def find_mc10_dataset() -> str:
    """Locate the pinned LoCoMo-MC10 snapshot. Order: LOCOMO_MC10_PATH env,
    then the pinned snapshot in the HF cache, then any snapshot (with a warning)."""
    override = os.environ.get("LOCOMO_MC10_PATH")
    if override and os.path.exists(override):
        return override
    roots = [os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(os.environ["HUGGINGFACE_HUB_CACHE"])
    fallback = None
    for root in roots:
        pinned = os.path.join(root, "datasets--Percena--locomo-mc10", "snapshots",
                              PINNED_SNAPSHOT, "data", "locomo_mc10.json")
        if os.path.exists(pinned):
            return pinned
        hits = sorted(glob.glob(os.path.join(
            root, "datasets--Percena--locomo-mc10", "snapshots", "*", "data",
            "locomo_mc10.json")))
        if hits and not fallback:
            fallback = hits[-1]
    if fallback:
        print(f"WARNING: pinned snapshot {PINNED_SNAPSHOT} not found; "
              f"using {fallback} — results are NOT comparable to the pinned baseline.",
              file=sys.stderr)
        return fallback
    sys.exit("LoCoMo-MC10 dataset not found (set LOCOMO_MC10_PATH or download "
             "Percena/locomo-mc10 into the HF cache).")


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def conv_id(question_id: str) -> str:
    return question_id.split("_q")[0]


def conv_haystacks(rows: list[dict]) -> dict[str, dict]:
    """One representative row (haystack) per conversation, keyed by conv id.
    Every row of a conversation carries the identical haystack."""
    convs: dict[str, dict] = {}
    for r in rows:
        convs.setdefault(conv_id(r["question_id"]), r)
    return convs


# ---------------------------------------------------------------------------
# Chunking (must mirror locomo_mc10_eval.py store_conversations EXACTLY)
# ---------------------------------------------------------------------------

def dt_prefix_for(session_dt: str | None) -> str:
    if not session_dt:
        return ""
    try:
        dt = datetime.fromisoformat(session_dt.replace("Z", "+00:00"))
        return f"[Conversation on {dt.strftime('%B %d, %Y at %I:%M %p')}] "
    except Exception:
        return f"[{session_dt}] "


def build_session_memories(item: dict) -> dict[int, list[tuple[str, str]]]:
    """Deterministic stored-memory texts per session: {session_no: [(kind, text)]}
    where kind is 'summary' or 'dialogue'. This is the single source of truth
    for BOTH ingest payloads and gold-label derivation."""
    sessions = item.get("haystack_sessions", [])
    summaries = item.get("haystack_session_summaries", [])
    datetimes = item.get("haystack_session_datetimes", [])
    out: dict[int, list[tuple[str, str]]] = {}

    for i, summary in enumerate(summaries):
        session = sessions[i] if i < len(sessions) else []
        session_dt = datetimes[i] if i < len(datetimes) else None
        dt_prefix = dt_prefix_for(session_dt)
        mems: list[tuple[str, str]] = []

        if summary and isinstance(summary, str) and summary.strip():
            mems.append(("summary", f"{dt_prefix}Session {i + 1} Summary: {summary[:2000]}"))

        if session and isinstance(session, list):
            current_chunk: list[str] = []
            current_len = 0
            for turn in session:
                if isinstance(turn, dict):
                    content = turn.get("content", "").strip()
                    if not content:
                        continue
                    turn_len = len(content)
                    if current_len > 0 and current_len + turn_len > TARGET_CHUNK_SIZE:
                        chunk_text = "\n".join(current_chunk)
                        mems.append(("dialogue", f"{dt_prefix}Session {i + 1}:\n{chunk_text}"))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(content)
                    current_len += turn_len
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                mems.append(("dialogue", f"{dt_prefix}Session {i + 1}:\n{chunk_text}"))

        out[i + 1] = mems
    return out


# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------

def evidence_session(corpus_item_id: str) -> int:
    match = EVIDENCE_SESSION_RE.match(corpus_item_id)
    if not match:
        sys.exit(f"unrecognised LoCoMo evidence id {corpus_item_id!r}")
    return int(match.group(1))


def derive_gold() -> dict[str, dict]:
    """{qid: {gold_sessions, evidence, locatable, question}} for every question in
    METRIC_CATEGORY, in id order, from LoCoMo's own evidence labels."""
    gold: dict[str, dict] = {}
    with open(LOCOMO_CASES_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    for case in sorted(cases, key=lambda c: (conv_id(c["id"]), int(c["id"].split("_q")[1]))):
        if case["category"] != METRIC_CATEGORY:
            continue
        evidence = [item["corpus_item_id"] for item in case["relevant"]]
        gold[case["id"]] = {
            "gold_sessions": sorted({evidence_session(e) for e in evidence}),
            "evidence": evidence,
            "locatable": bool(evidence),
            "question": case["query"],
        }
    return gold


def evidence_alignment(gold: dict[str, dict], convs: dict[str, dict]) -> list[str]:
    """Evidence turns whose text is not in the dialogue stored for their session.
    MC10 numbers sessions by position; this is the check that the numbering matches
    LoCoMo's D<n>."""
    turn_text: dict[str, str] = {}
    with open(LOCOMO_CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                turn_text[item["id"]] = item["content"].split(": ", 1)[-1].strip().lower()[:60]
    blobs = {
        cid: {sn: " \n ".join(t for kind, t in texts if kind == "dialogue").lower()
              for sn, texts in build_session_memories(convs[cid]).items()}
        for cid in convs
    }
    missing = []
    for qid, g in gold.items():
        for evidence in g["evidence"]:
            text = turn_text.get(evidence, "")
            session = blobs.get(conv_id(qid), {}).get(evidence_session(evidence), "")
            if not text or text not in session:
                missing.append(evidence)
    return missing


# ---------------------------------------------------------------------------
# Veld HTTP client
# ---------------------------------------------------------------------------

class VeldClient:
    def __init__(self, base_url: str, api_key: str):
        # Imported here so --emit-gold runs without the HTTP dependency installed.
        import requests

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json",
                                     "X-API-Key": api_key})

    def health(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except Exception:
            return False

    def remember(self, user_id: str, content: str, memory_type: str, tags: list[str]):
        resp = self.session.post(f"{self.base_url}/api/remember", json={
            "user_id": user_id, "content": content,
            "memory_type": memory_type, "tags": tags})
        resp.raise_for_status()
        return resp.json()

    def recall(self, user_id: str, query: str, limit: int, mode: str) -> list[dict]:
        resp = self.session.post(f"{self.base_url}/api/recall", json={
            "user_id": user_id, "query": query, "limit": limit, "mode": mode})
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def set_dials(self, user_id: str, overrides: dict) -> dict:
        """GET-merge-PUT (the dials endpoint is full-replace)."""
        cur = self.session.get(f"{self.base_url}/api/dials",
                               params={"user_id": user_id})
        cur.raise_for_status()
        dials = cur.json()
        dials.update(overrides)
        resp = self.session.put(f"{self.base_url}/api/dials",
                                json={"user_id": user_id, "dials": dials})
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def spawn_server(binary: str, port: int, store: str) -> subprocess.Popen:
    os.makedirs(store, exist_ok=True)
    log_path = os.path.join(store, "server.log")
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        [binary, "server", "--port", str(port), "--storage", store],
        stdout=log, stderr=log, env=os.environ.copy())
    print(f"Spawned veld server pid={proc.pid} port={port} store={store} "
          f"(log: {log_path})")
    return proc


def wait_health(client: VeldClient, proc: subprocess.Popen | None, timeout_s: int = 120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            sys.exit(f"veld server exited early with code {proc.returncode}")
        if client.health():
            return
        time.sleep(1.0)
    sys.exit(f"veld server did not become healthy within {timeout_s}s")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def user_for(cid: str) -> str:
    return f"locomo_{cid}"


def ingest_corpus(client: VeldClient, convs: dict[str, dict]) -> dict[str, int]:
    counts = {}
    for cid in sorted(convs):
        uid = user_for(cid)
        mems = build_session_memories(convs[cid])
        n = 0
        t0 = time.perf_counter()
        for sn in sorted(mems):
            for kind, text in mems[sn]:
                if kind == "summary":
                    client.remember(uid, text, "Context", [f"session_{sn}", "summary"])
                else:
                    client.remember(uid, text, "Conversation", [f"session_{sn}", "dialogue"])
                n += 1
        counts[cid] = n
        print(f"  ingested {cid}: {n} memories in {time.perf_counter() - t0:.1f}s")
    return counts


def retrieved_sessions(memories: list[dict]) -> set[int]:
    hits: set[int] = set()
    for m in memories:
        content = m.get("experience", {}).get("content", "") or m.get("content", "")
        hits.update(int(s) for s in SESSION_RE.findall(content))
    return hits


def run_bench(args):
    rows = load_dataset(find_mc10_dataset())
    convs = conv_haystacks(rows)

    with open(args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)
    n_loc = sum(1 for g in gold.values() if g["locatable"])
    print(f"Gold labels: {len(gold)} LoCoMo {METRIC_CATEGORY} questions, n_loc={n_loc}")
    misaligned = evidence_alignment(gold, convs)
    evidence_total = sum(len(g["evidence"]) for g in gold.values())
    print(f"Evidence turns not found in their labelled session: "
          f"{len(misaligned)}/{evidence_total}")
    if evidence_total and len(misaligned) * 2 > evidence_total:
        sys.exit("Most evidence turns are missing from their labelled session: MC10's session "
                 "numbering does not match LoCoMo's D<n>, so these labels cannot score this run.")

    dials = json.loads(args.dials) if args.dials else {}
    base_url = f"http://127.0.0.1:{args.port}"
    api_key = os.environ.get("VELD_API_KEYS", os.environ.get(
        "VELD_API_KEY", "sk-veld-dev-local-testing-key")).split(",")[0]
    client = VeldClient(base_url, api_key)

    proc = None
    if not args.no_spawn:
        if not args.store:
            sys.exit("--store is required when spawning a server (or pass --no-spawn)")
        proc = spawn_server(args.binary, args.port, args.store)
    try:
        wait_health(client, proc)
        print("Server healthy. Ingesting pinned corpus (10 convs, sorted by ID)...")
        counts = ingest_corpus(client, convs)

        if dials:
            for cid in sorted(convs):
                applied = client.set_dials(user_for(cid), dials)
                relevant = {k: applied.get(k) for k in dials}
            print(f"Applied dials to all users: {relevant}")

        # Warmup (discarded)
        by_qid = {r["question_id"]: r for r in rows}
        for qid in WARMUP_QIDS:
            q = by_qid[qid]
            client.recall(user_for(conv_id(qid)), q["question"], RECALL_K, RECALL_MODE)
        print(f"Warmup done ({len(WARMUP_QIDS)} queries discarded).")

        # Scored pass: METRIC_CATEGORY, pinned conv order
        results = []
        hits = 0
        latencies = []
        for cid in sorted(convs):
            for r in rows:
                qid = r["question_id"]
                if conv_id(qid) != cid or qid not in gold:
                    continue
                g = gold[qid]
                if not g["locatable"]:
                    continue
                t0 = time.perf_counter()
                mems = client.recall(user_for(cid), r["question"], RECALL_K, RECALL_MODE)
                lat = (time.perf_counter() - t0) * 1000
                latencies.append(lat)
                got = retrieved_sessions(mems)
                hit = bool(got & set(g["gold_sessions"]))
                hits += hit
                results.append({"qid": qid, "hit": hit,
                                "gold_sessions": g["gold_sessions"],
                                "retrieved_sessions": sorted(got),
                                "latency_ms": round(lat, 1)})
        recall_at_k = hits / len(results) if results else 0.0
        latencies.sort()
        out = {
            "protocol": "pinned-locomo-mc10-evidence-recall/evidence-labels",
            "population": f"locomo {METRIC_CATEGORY}",
            "evidence_turns_misaligned": len(misaligned),
            "snapshot": PINNED_SNAPSHOT,
            "mode": RECALL_MODE, "k": RECALL_K,
            "dials": dials,
            "n_loc": len(results),
            "hits": hits,
            "evidence_recall_at_10": recall_at_k,
            "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
            "memories_ingested": counts,
            "results": results,
        }
        print(f"\nEvidence recall@{RECALL_K}: {recall_at_k:.4f} "
              f"({hits}/{len(results)}), p50 latency "
              f"{out['latency_ms_p50']:.0f}ms")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Results written to {args.output}")
        return out
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("Server stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pinned LoCoMo-MC10 evidence-recall harness")
    ap.add_argument("--emit-gold", action="store_true",
                    help="Derive gold labels from LoCoMo's evidence (tests/recall/locomo_cases.jsonl) "
                         "and write them to --gold (no server or dataset needed), then exit")
    ap.add_argument("--gold", default=DEFAULT_GOLD_PATH,
                    help=f"Gold labels path (default: {DEFAULT_GOLD_PATH})")
    ap.add_argument("--port", type=int, default=4001, help="Server port (default 4001)")
    ap.add_argument("--store", default=None,
                    help="Server storage dir (disposable; required unless --no-spawn)")
    ap.add_argument("--binary", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "target", "release", "veld"),
        help="veld binary to spawn (default: <repo>/target/release/veld)")
    ap.add_argument("--no-spawn", action="store_true",
                    help="Use an already-running server at --port instead of spawning")
    ap.add_argument("--dials", default=None,
                    help="JSON dial overrides applied per-user before scoring, "
                         "e.g. '{\"use_reranking\": false}' for the rerank-off baseline")
    ap.add_argument("--output", default=None, help="Results JSON output path")
    args = ap.parse_args()

    if args.emit_gold:
        gold = derive_gold()
        n_loc = sum(1 for g in gold.values() if g["locatable"])
        with open(args.gold, "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {len(gold)} LoCoMo {METRIC_CATEGORY} gold labels to {args.gold} "
              f"(n_loc={n_loc})")
        return

    run_bench(args)


if __name__ == "__main__":
    main()
