"""Concurrent HTTP load driver for one benchmark phase. Prints one JSON object on stdout."""
import argparse
import http.client
import json
import random
import sys
import threading
import time

USER_ID = "bench"


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return round(sorted_values[position], 3)


def run_phase(host, port, api_key, path, bodies, concurrency):
    cursor = [0]
    cursor_lock = threading.Lock()
    latencies = []
    errors = [0]
    first_error = [None]
    result_lock = threading.Lock()
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    def worker():
        connection = http.client.HTTPConnection(host, port, timeout=120)
        local = []
        local_errors = 0
        local_first = None
        while True:
            with cursor_lock:
                position = cursor[0]
                cursor[0] += 1
            if position >= len(bodies):
                break
            started = time.perf_counter()
            try:
                connection.request("POST", path, body=bodies[position], headers=headers)
                response = connection.getresponse()
                payload = response.read()
                if response.status >= 300:
                    local_errors += 1
                    local_first = local_first or "HTTP %d %s" % (response.status, payload[:200])
                    continue
            except (OSError, http.client.HTTPException) as exc:
                local_errors += 1
                local_first = local_first or repr(exc)
                connection.close()
                connection = http.client.HTTPConnection(host, port, timeout=120)
                continue
            local.append((time.perf_counter() - started) * 1000.0)
        connection.close()
        with result_lock:
            latencies.extend(local)
            errors[0] += local_errors
            first_error[0] = first_error[0] or local_first

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    wall_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - wall_start
    latencies.sort()
    return {
        "requests": len(bodies),
        "ok": len(latencies),
        "errors": errors[0],
        "first_error": first_error[0],
        "wall_seconds": round(wall, 3),
        "req_per_s": round(len(latencies) / wall, 2) if wall > 0 else None,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
    }


def read_corpus(path, count):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) == count:
                break
    if len(records) < count:
        sys.exit("corpus has %d records, %d requested" % (len(records), count))
    return records


def query_text(record):
    return " ".join(record["content"].split()[:14])


def main():
    parser = argparse.ArgumentParser(description="Run one load phase against a veld server.")
    parser.add_argument("phase", choices=["remember", "recall", "proactive"],
                        help="remember posts corpus records; recall and proactive query from them")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--corpus", required=True, help="JSON lines file from corpus.py")
    parser.add_argument("--count", type=int, required=True,
                        help="records to seed (remember) or size of the seeded set to draw queries from")
    parser.add_argument("--queries", type=int, default=0, help="query count for recall and proactive")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7, help="query selection seed")
    parser.add_argument("--limit", type=int, default=10, help="recall result limit")
    args = parser.parse_args()

    records = read_corpus(args.corpus, args.count)
    if args.phase == "remember":
        path = "/api/remember"
        bodies = [json.dumps({"user_id": USER_ID, "content": r["content"], "tags": r["tags"],
                              "episode_id": r["episode_id"], "sequence_number": r["sequence_number"]})
                  for r in records]
    else:
        rng = random.Random(args.seed)
        picks = [records[rng.randrange(len(records))] for _ in range(args.queries)]
        if args.phase == "recall":
            path = "/api/recall"
            bodies = [json.dumps({"user_id": USER_ID, "query": query_text(r), "limit": args.limit})
                      for r in picks]
        else:
            path = "/api/proactive_context"
            bodies = [json.dumps({"user_id": USER_ID, "context": query_text(r), "auto_ingest": False})
                      for r in picks]
    result = run_phase(args.host, args.port, args.api_key, path, bodies, args.concurrency)
    result["phase"] = args.phase
    result["concurrency"] = args.concurrency
    print(json.dumps(result))


if __name__ == "__main__":
    main()
