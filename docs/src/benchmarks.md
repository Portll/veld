# Benchmarks

Retrieval-quality benchmarks are tracked in
[BENCHMARKS.md](https://github.com/Portll/veld/blob/main/BENCHMARKS.md) and
[benchmark_report.json](https://github.com/Portll/veld/blob/main/benchmark_report.json)
in the repo. This page summarises the most recent snapshot.

## Most recent measurement: LoCoMo, 1,531 questions

Measured 2026-10-20 at `3faec074` on 1,531 questions from the LoCoMo dataset's 10 conversations
(single-hop, multi-hop, temporal and open-domain; the adversarial category is left out), with the
MiniLM embedder and the mxbai-rerank-xsmall-v1 cross-encoder reranking the top 20.

| Metric | Value |
|---|---|
| **p@1** (gold memory ranked first) | 0.406 |
| **MRR** (Mean Reciprocal Rank) | 0.492 |
| **recall@10** | 0.589 |
| **nDCG@10** | 0.493 |

By category, nDCG@10 is 0.551 single-hop (841 questions), 0.593 temporal (320), 0.294
multi-hop (281) and 0.221 open-domain (89).

These are retrieval metrics: whether the gold turn was ranked, and where. They are not
end-to-end answer accuracy. Published LoCoMo figures for Mem0, Zep, Letta and others are
LLM-judged answer accuracy ("J"), so the two are on different scales and do not compare. The
1,531 questions also include the 100-question gate (conversation conv-42) that recall changes are
tuned against, so the run is not a fully held-out measurement; a held-out set without conv-42,
with confidence intervals, is task 7 of plan `veld-review-remediation-20261020`.

## Withdrawn figures

Earlier pages quoted MRR 0.925 for v0.7.2, and a changelog entry quotes MRR 0.917 (`6dae51f6`).
Both came from `benchmarks/locomo_api_benchmark.py`, which stores 20 synthetic memories and asks
20 questions about them. That is not the LoCoMo dataset, and a store of 20 memories leaves a
retriever very few candidates to rank, so the figures do not describe veld's retrieval. They are
withdrawn; the table above replaces them.

## Not yet measured on the full suite

Main has gained these since `3faec074`; the 100-question LoCoMo gate measured each one on its
own, and a full-suite run of all of them together is part of the v0.8 track:

- Recall embeds the query, and each sub-query, on the embedder's query side.
- Layer 3.7: a date-window candidate leg for questions that name a day or a month and year.
- Reranker dials: which cross-encoder (`rerank_model`), how many candidates it sees
  (`rerank_pool`) and how many it scores (`rerank_count`).
- Faster cross-encoding: pairs batched by length, a second session under concurrent load, and a
  cache of pair scores.
- snowflake-arctic-embed-xs as an optional embedder.

## Running benchmarks locally

The benchmark harness lives in `benches/` (Criterion) and
[evaluations/](https://github.com/Portll/veld/tree/main/evaluations).

```sh
cargo bench --workspace                  # Criterion benches
cargo run --bin alignment-eval --release # alignment retrieval eval
```

Each `cargo bench` produces HTML reports under `target/criterion/`.

## Throughput

Throughput numbers are platform-dependent and the current focus is on
retrieval *quality*, not throughput. Rough order-of-magnitude on a modern
laptop (M2 Pro / Ryzen 7950X equivalent) with default config:

| Operation | Throughput |
|---|---|
| `POST /api/remember` (single) | ~500-2000 req/s |
| `POST /api/recall` (single, cold) | ~50-200 req/s |
| `POST /api/recall` (single, cached embedder) | ~200-800 req/s |
| `POST /api/proactive_context` | ~50-200 req/s |

Cross-encoder reranking is the dominant cost in recall; reducing the
rerank top-K trades quality for latency.

Rate limit defaults: 4000 req/s per API key, 200 concurrent requests. See
[Configuration reference](reference/config.md) for tuning.

## Engine comparison (RocksDB vs redb)

Not yet measured on this branch. The table below is empty on purpose: run the commands, then paste the script output into it. Do not quote numbers from other machines or from the older sections of this document for this comparison.

### Commands

End-to-end through the HTTP API, one run per memory count. Build the release binary first, then:

```sh
VELD_BIN=target/release/veld scripts/bench/http_engine_compare.sh --memories 10000
VELD_BIN=target/release/veld scripts/bench/http_engine_compare.sh --memories 100000
```

The script starts `veld server` once per engine (default `rocksdb redb`) on a fresh temporary data dir and a free port, seeds the memories through `POST /api/remember`, runs `POST /api/recall` twice over the same deterministic queries (cold, then warm), runs `POST /api/proactive_context` with `auto_ingest` off, and records server RSS after seeding and after recall plus the data dir size. It writes one JSON file per engine to `bench-results/<timestamp>/` and prints the side-by-side table.

Flags: `--engines "A B"`, `--memories N` (default 10000), `--queries N` (default 2000), `--proactive-queries N` (default 200), `--concurrency C` (default 16), `--results-dir DIR`, `--bin PATH`, `--api-key KEY`, `--corpus-seed N`. Run `scripts/bench/http_engine_compare.sh --help` for the full text.

Tuning variables set in the calling environment are passed to the server unchanged: `VELD_STORAGE_DURABILITY` (`immediate` or `eventual`), `VELD_REDB_CACHE_BYTES`, `VELD_ROCKSDB_WRITE_BUFFER_BYTES`, `VELD_ROCKSDB_BLOCK_CACHE_BYTES`, plus the embedder selection `VELD_EMBEDDER` and `VELD_EMBEDDING_API_URL`. Record the values you used next to the results. The values are also saved in each JSON file.

Micro and cold-start benches, which isolate the engines from the HTTP and embedding layers:

```sh
cargo bench --bench storage_engine_benchmarks
cargo bench --bench cold_start_benchmarks
```

### What the numbers include

No deterministic or network-free embedder exists, so remember throughput and cold recall include the cost of embedding each request, and both engines pay it equally. Warm recall repeats the same queries and is served from the query embedding cache, so it is the closest end-to-end figure to storage and index cost. Use one embedder for every run and compare engines with each other on the same machine, not against other hardware. The Criterion benches above are the storage-only numbers.

### Results

Paste the script table for each run below. Columns are filled per engine and memory count.

| Metric | RocksDB 10k | redb 10k | RocksDB 100k | redb 100k |
|---|---|---|---|---|
| remember req/s | | | | |
| remember p95 | | | | |
| recall cold req/s | | | | |
| recall warm req/s | | | | |
| recall p95 warm | | | | |
| proactive_context req/s | | | | |
| RSS after seed | | | | |
| RSS after recall | | | | |
| data dir size | | | | |

## See also

- [Tuning retrieval](guides/tuning-retrieval.md)
- [Retrieval pipeline](architecture/retrieval.md)
- [PROGRESS.md](https://github.com/Portll/veld/blob/main/PROGRESS.md)
- [BENCHMARKS.md](https://github.com/Portll/veld/blob/main/BENCHMARKS.md)
