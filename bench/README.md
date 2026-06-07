# bench/

Calibration and evaluation tooling for sleight / veld.

## recall_latency_guard.py

**Purpose:** Lightweight regression guard for evidence-recall quality and
recall-path latency.  Run this before merging any change that touches the
recall pipeline (`memory/recall.rs`, BM25 weights, vector index parameters,
or embedding model config).

**What it does:**

1. Stores 20 fixed conversational memories into a throwaway veld user
   namespace (`guard_<timestamp>`).
2. Runs 5 multi-hop recall queries (the hardest category — requires the index
   to surface multiple semantically related memories simultaneously).
3. Asserts `multi_hop_R@10 >= FLOOR` and `p99_recall_latency <= CEILING`.
4. Exits 0 if both pass; exits 1 and prints the offending metric if either
   fails.

**No LLM judge required.**  Scoring is pure retrieval: whether the expected
memory IDs appear in the top-10 results.  This makes the guard fast (~15-30 s
with a warm veld server) and CI-safe without model credentials.

### Thresholds and their derivation

| Metric | Threshold | Source |
|---|---|---|
| `multi_hop_R@10` | **>= 0.60** | V-4/V-5 baseline ~0.78, minus 10 pp regression margin, rounded down to 0.60 for env/machine variance |
| `p99_recall_latency` | **<= 3 000 ms** | Baseline p99 = 1 225 ms (locomo_results.json @ 8d3ddc9), doubled to 2 450 ms, rounded up to 3 000 ms for CI runner overhead |

The contamination floor mentioned in V-4 analysis (51.7 % on the MC10
multiple-choice judge task) does not directly map to the R@10 metric here;
the 0.60 floor is derived from the V-5 observed R@10, not from the judge
task.

Override thresholds without editing the file:

```sh
MULTI_HOP_R10_FLOOR=0.70 P99_LATENCY_CEILING_MS=2500 python bench/recall_latency_guard.py
```

### Quick run

```sh
# veld server must be running on localhost:3030
python bench/recall_latency_guard.py

# or with explicit env
VELD_URL=http://127.0.0.1:3030 VELD_API_KEY=dev-key-local python bench/recall_latency_guard.py
```

### CI integration

See `.github/workflows/recall-guard.yml`.  The job is `workflow_dispatch`
(opt-in) because it requires a live veld server with the embedding model
loaded — GitHub hosted runners cannot provide that.  Run it locally or on a
self-hosted runner before merging recall-path changes.

To run on a self-hosted runner:

1. Ensure the runner has `ORT_DYLIB_PATH` and `VELD_MODEL_PATH` set (see
   project CLAUDE.md).
2. Add the `self-hosted` label to the job's `runs-on` field in
   `.github/workflows/recall-guard.yml`.
3. The veld server must be started as a pre-step (or run as a systemd service
   on the runner).

## run_eval.sh

Full sleight evaluation sweep across multiple LLM models.  Requires
`sleight-eval` CLI or the Python bridge, plus veld + Megablast running.

## calibrate.py

Post-processes evaluation traces from `run_eval.sh` into per-model
performance profiles in `profiles/`.
