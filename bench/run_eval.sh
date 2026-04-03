#!/usr/bin/env bash
# Run sleight evaluations across multiple LLMs and collect Overlook traces.
# Traces are saved to traces/ for consumption by calibrate.py.
#
# Prerequisites:
#   - shodh-memory running on localhost:3301
#   - LM Studio running with target models loaded
#   - Megablast model router accessible
#
# Usage:
#   ./run_eval.sh                    # All models
#   ./run_eval.sh --model qwq-32b   # Single model
#   ./run_eval.sh --tasks coding     # Task subset

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACES_DIR="${SCRIPT_DIR}/traces"
PROFILES_DIR="${SCRIPT_DIR}/profiles"
SHODH_URL="${SHODH_URL:-http://localhost:3301}"
MEGABLAST_URL="${MEGABLAST_URL:-http://localhost:3302}"

mkdir -p "$TRACES_DIR" "$PROFILES_DIR"

# ── Model list (extend as needed) ──
MODELS=(
    "opus-4"
    "sonnet-4"
    "qwq-32b"
    "deepseek-r1"
    "mistral-7b"
)

# ── Standard evaluation tasks ──
TASK_SETS=(
    "coding:Implement a rate limiter with sliding window"
    "coding:Fix a race condition in concurrent HashMap access"
    "review:Security audit of an OAuth2 implementation"
    "review:Evaluate error handling in async Rust"
    "planning:Design a migration from monolith to microservices"
    "planning:Architect a real-time notification system"
    "question:Explain the trade-offs of eventual consistency"
    "question:Compare Raft vs Paxos for consensus"
)

# ── Parse args ──
TARGET_MODEL=""
TARGET_TASKS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) TARGET_MODEL="$2"; shift 2 ;;
        --tasks) TARGET_TASKS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Health checks ──
echo "Checking shodh-memory..."
curl -sf "${SHODH_URL}/health" > /dev/null || { echo "shodh-memory not running at ${SHODH_URL}"; exit 1; }
echo "  ✓ shodh-memory healthy"

# ── Run evaluations ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for model in "${MODELS[@]}"; do
    [[ -n "$TARGET_MODEL" && "$model" != "$TARGET_MODEL" ]] && continue
    echo ""
    echo "═══ Evaluating: ${model} ═══"

    for task_entry in "${TASK_SETS[@]}"; do
        IFS=':' read -r task_type task_desc <<< "$task_entry"
        [[ -n "$TARGET_TASKS" && "$task_type" != "$TARGET_TASKS" ]] && continue

        task_id="${task_type}_$(echo "$task_desc" | tr ' ' '_' | tr '[:upper:]' '[:lower:]' | cut -c1-40)"
        trace_file="${TRACES_DIR}/${model}_${task_id}_${TIMESTAMP}.json"

        echo "  ├─ ${task_type}: ${task_desc}"

        # Run evaluation via sleight CLI with Overlook trace capture
        # This assumes sleight has a CLI mode that accepts --model and --trace-output
        # Adapt to your actual sleight CLI interface
        if command -v sleight-eval &> /dev/null; then
            sleight-eval \
                --model "$model" \
                --task-type "$task_type" \
                --task "$task_desc" \
                --shodh-url "$SHODH_URL" \
                --trace-output "$trace_file" \
                2>&1 | sed 's/^/  │  /'
        else
            # Fallback: generate trace via Python bridge
            python3 "${SCRIPT_DIR}/eval_bridge.py" \
                --model "$model" \
                --task-type "$task_type" \
                --task "$task_desc" \
                --shodh-url "$SHODH_URL" \
                --output "$trace_file" \
                2>&1 | sed 's/^/  │  /'
        fi

        if [[ -f "$trace_file" ]]; then
            echo "  │  ✓ Trace saved: $(basename "$trace_file")"
        else
            echo "  │  ✗ No trace generated"
        fi
    done
done

# ── Calibrate ──
TRACE_COUNT=$(find "$TRACES_DIR" -name "*.json" | wc -l)
echo ""
echo "═══ Calibration ═══"
echo "  Traces: ${TRACE_COUNT}"

if [[ "$TRACE_COUNT" -gt 0 ]]; then
    python3 "${SCRIPT_DIR}/calibrate.py" \
        --traces-dir "$TRACES_DIR" \
        --output-dir "$PROFILES_DIR" \
        --visualise
    echo ""
    echo "  Profiles saved to ${PROFILES_DIR}/"
    ls -la "$PROFILES_DIR"/*.json 2>/dev/null | sed 's/^/  /'
else
    echo "  No traces to calibrate. Run evaluations first."
fi
