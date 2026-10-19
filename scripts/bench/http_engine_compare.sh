#!/usr/bin/env bash
# Compares veld storage engines through the real HTTP API: seed, recall cold/warm,
# proactive_context, server RSS and data dir size. Needs bash, curl, python3, ps, du.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${VELD_BIN:-target/release/veld}"
ENGINES="rocksdb redb"
MEMORIES=10000
QUERIES=2000
PROACTIVE_QUERIES=200
CONCURRENCY=16
RESULTS_ROOT="./bench-results"
API_KEY="bench-key-0123456789abcdef0123456789abcdef"
CORPUS_SEED=20261018
READY_TIMEOUT_SECONDS=60

# Passed to the server untouched when set in the caller's environment. The embedder
# variables matter because the embedder, not the engine, dominates remember and cold recall.
PASS_THROUGH_ENV=(
  VELD_STORAGE_DURABILITY
  VELD_REDB_CACHE_BYTES
  VELD_ROCKSDB_WRITE_BUFFER_BYTES
  VELD_ROCKSDB_BLOCK_CACHE_BYTES
  VELD_EMBEDDER
  VELD_EMBEDDING_API_URL
  VELD_EMBEDDING_API_MODEL
  VELD_ONNX_THREADS
  VELD_OFFLINE
  ORT_DYLIB_PATH
)

usage() {
  cat <<'EOF'
Usage: http_engine_compare.sh [options]

Starts `veld server` once per engine on a fresh data dir, seeds memories through
POST /api/remember, runs recall twice (cold, warm) and proactive_context, records
server RSS and data dir size, and prints a side-by-side markdown table.

Options:
  --engines "A B"          Engines to compare, space separated (default: "rocksdb redb")
  --memories N             Memories to seed (default: 10000; use 100000 for the large run)
  --queries N              Recall queries per pass, drawn from the seeded content (default: 2000)
  --proactive-queries N    proactive_context queries, auto_ingest disabled (default: 200)
  --concurrency C          Concurrent client connections for every phase (default: 16)
  --results-dir DIR        Root for results; files go in DIR/<timestamp>/ (default: ./bench-results)
  --bin PATH               veld binary (default: $VELD_BIN, else target/release/veld)
  --api-key KEY            API key given to the server and the client (default: fixed bench key)
  --corpus-seed N          Corpus RNG seed (default: 20261018)
  -h, --help               Show this help

Environment passed through to the server when set:
  VELD_STORAGE_DURABILITY (immediate|eventual), VELD_REDB_CACHE_BYTES,
  VELD_ROCKSDB_WRITE_BUFFER_BYTES, VELD_ROCKSDB_BLOCK_CACHE_BYTES,
  VELD_EMBEDDER, VELD_EMBEDDING_API_URL, VELD_EMBEDDING_API_MODEL,
  VELD_ONNX_THREADS, VELD_OFFLINE, ORT_DYLIB_PATH

Caveat: no deterministic or network-free embedder exists, so remember throughput
and cold recall include embedding cost. Warm recall hits the server's query
embedding cache and is the storage-bound number. Use the same embedder for both
engines and compare the engines with each other, not with other machines.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engines) ENGINES="${2:?--engines needs a value}"; shift 2 ;;
    --memories) MEMORIES="${2:?--memories needs a value}"; shift 2 ;;
    --queries) QUERIES="${2:?--queries needs a value}"; shift 2 ;;
    --proactive-queries) PROACTIVE_QUERIES="${2:?--proactive-queries needs a value}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:?--concurrency needs a value}"; shift 2 ;;
    --results-dir) RESULTS_ROOT="${2:?--results-dir needs a value}"; shift 2 ;;
    --bin) BIN="${2:?--bin needs a value}"; shift 2 ;;
    --api-key) API_KEY="${2:?--api-key needs a value}"; shift 2 ;;
    --corpus-seed) CORPUS_SEED="${2:?--corpus-seed needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

for tool in curl python3 ps du; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool not found: $tool"
done
[[ -x "$BIN" ]] || die "veld binary not found or not executable: $BIN (set VELD_BIN or --bin)"
for value in "$MEMORIES" "$QUERIES" "$PROACTIVE_QUERIES" "$CONCURRENCY"; do
  [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || die "numeric flags must be positive integers, got: $value"
done

RESULTS_DIR="$RESULTS_ROOT/$(date +%Y%m%d-%H%M%S)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/veld-bench.XXXXXX")"
mkdir -p "$RESULTS_DIR"
SERVER_PID=""

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$SERVER_PID" 2>/dev/null && [[ $waited -lt 60 ]]; do
      sleep 0.5
      waited=$((waited + 1))
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "warning: server $SERVER_PID ignored SIGTERM for 30s, sending SIGKILL" >&2
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  stop_server
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

free_port() {
  python3 -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

rss_kb() {
  ps -o rss= -p "$SERVER_PID" | tr -d ' '
}

dir_bytes() {
  local kilobytes
  kilobytes="$(du -sk "$1" | cut -f1)"
  echo $((kilobytes * 1024))
}

wait_for_live() {
  local port="$1" log="$2" waited=0
  while [[ $waited -lt $READY_TIMEOUT_SECONDS ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before becoming live; log tail:" >&2
      tail -n 40 "$log" >&2
      exit 1
    fi
    if curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:$port/health/live" 2>/dev/null; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "server not live after ${READY_TIMEOUT_SECONDS}s; log tail:" >&2
  tail -n 40 "$log" >&2
  exit 1
}

run_phase() {
  local name="$1" port="$2" out_file="$3"
  shift 3
  python3 -I "$SCRIPT_DIR/load.py" "$name" --port "$port" --api-key "$API_KEY" \
    --corpus "$CORPUS_FILE" --count "$MEMORIES" --concurrency "$CONCURRENCY" "$@" >"$out_file"
  python3 -I -c 'import json, sys
r = json.load(open(sys.argv[1]))
print("  %-10s %9.1f req/s  p50 %8.1f  p95 %8.1f  p99 %8.1f ms  errors %d" % (
    sys.argv[2], r["req_per_s"] or 0, r["p50_ms"] or 0, r["p95_ms"] or 0, r["p99_ms"] or 0, r["errors"]))
if r["first_error"]:
    print("  first error: " + r["first_error"], file=sys.stderr)' "$out_file" "$name"
}

CORPUS_FILE="$WORK_DIR/corpus.jsonl"
python3 -I "$SCRIPT_DIR/corpus.py" --count "$MEMORIES" --seed "$CORPUS_SEED" --out "$CORPUS_FILE"

run_engine() {
  local engine="$1"
  local data_dir="$WORK_DIR/data-$engine" phases_dir="$WORK_DIR/phases-$engine"
  local log="$WORK_DIR/server-$engine.log" port
  mkdir -p "$data_dir" "$phases_dir"
  port="$(free_port)"

  local server_env=(
    "VELD_STORAGE_BACKEND=$engine"
    "VELD_MEMORY_PATH=$data_dir"
    "VELD_HOST=127.0.0.1"
    "VELD_PORT=$port"
    "VELD_API_KEYS=$API_KEY"
  )
  local tuning=()
  local name
  for name in "${PASS_THROUGH_ENV[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      server_env+=("$name=${!name}")
      tuning+=("$name=${!name}")
    fi
  done

  echo "== $engine: starting on port $port"
  env "${server_env[@]}" "$BIN" server >"$log" 2>&1 &
  SERVER_PID=$!
  wait_for_live "$port" "$log"

  echo "== $engine: seeding $MEMORIES memories"
  run_phase remember "$port" "$phases_dir/remember.json"
  local rss_seed
  rss_seed="$(rss_kb)"

  echo "== $engine: recall cold, recall warm, proactive_context"
  run_phase recall "$port" "$phases_dir/recall_cold.json" --queries "$QUERIES"
  run_phase recall "$port" "$phases_dir/recall_warm.json" --queries "$QUERIES"
  run_phase proactive "$port" "$phases_dir/proactive.json" --queries "$PROACTIVE_QUERIES"
  local rss_recall data_bytes
  rss_recall="$(rss_kb)"
  data_bytes="$(dir_bytes "$data_dir")"

  stop_server

  python3 -I "$SCRIPT_DIR/report.py" assemble --engine "$engine" --phases-dir "$phases_dir" \
    --memories "$MEMORIES" --concurrency "$CONCURRENCY" \
    --rss-seed-kb "$rss_seed" --rss-recall-kb "$rss_recall" --data-bytes "$data_bytes" \
    --tuning "${tuning[@]+"${tuning[@]}"}" --out "$RESULTS_DIR/$engine.json"
}

for engine in $ENGINES; do
  run_engine "$engine"
done

echo
echo "Results: $RESULTS_DIR"
python3 -I "$SCRIPT_DIR/report.py" table "$RESULTS_DIR"
