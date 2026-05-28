# Benchmarks

Retrieval-quality benchmarks are tracked in
[BENCHMARKS.md](https://github.com/Portll/veld/blob/main/BENCHMARKS.md) and
[benchmark_report.json](https://github.com/Portll/veld/blob/main/benchmark_report.json)
in the repo. This page summarises the most recent snapshot.

## Most recent benchmarked snapshot — v0.7.2

| Metric | Value |
|---|---|
| **MRR** (Mean Reciprocal Rank) | 0.925 |
| **Temporal MRR** | 0.900 |
| **Single-hop accuracy** | 1.000 |
| **Composite score** | 78.4% |

Source: [PROGRESS.md](https://github.com/Portll/veld/blob/main/PROGRESS.md)
v0.7.2 retrieval benchmark.

## What changed since the snapshot

The retrieval pipeline has shipped several improvements since v0.7.2 that
have not yet been benchmarked publicly:

- Layer 3.5: dual-embedder max-score merge (MiniLM + Nomic)
- Layer 4.527: BM25 specificity discount
- Layer 4.92: interference detection
- Layer 5.3: cross-encoder reranking wired in (18% blend, top-20 budget)
- Layer 5.85: linguistic boost reordering
- Layer 5.87: calendar-aware temporal range demotion
- Cross-embedder alignment (Procrustes + Ridge) — see
  [alignment](architecture/alignment.md)

A v0.8 benchmark pass is part of the stabilization track.

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

## See also

- [Tuning retrieval](guides/tuning-retrieval.md)
- [Retrieval pipeline](architecture/retrieval.md)
- [PROGRESS.md](https://github.com/Portll/veld/blob/main/PROGRESS.md)
- [BENCHMARKS.md](https://github.com/Portll/veld/blob/main/BENCHMARKS.md)
