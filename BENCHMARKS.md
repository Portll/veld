# Veld - Agentic Memory Performance Benchmarks

Comprehensive performance measurements for Veld - Agentic Memory v0.1.5+.

**Test Environment:** Windows 11, Intel i7-1355U (10 cores, 12 threads), Rust 1.75, Release build with LTO

---

## Quick Summary

| Component | Latency | Verdict |
|-----------|---------|---------|
| Entity lookup (1000 entities) | **763 ns** | ⚡ Sub-microsecond |
| Graph stats | **26 ns - 7 μs** | ⚡ O(1) |
| Vector search (1000 vectors) | **2-5 ms** | ✅ Fast |
| Memory store (with embedding) | **150-250 ms** | 🟡 ONNX bottleneck |
| Hebbian edge update | **<1 ms** | ⚡ Negligible overhead |

---

## 1. Hebbian Learning (December 2025)

### Graph Stats (O(1) cache reads)

| Operation | Time |
|-----------|------|
| Empty graph | **26 ns** |
| 10 memories | **148 ns** |
| 50 memories | **446 ns** |
| 100 memories | **604 ns** |
| 500 memories | **7.1 μs** |

### Edge Formation (includes embedding generation)

| Memories | Time | Edges Formed |
|----------|------|--------------|
| 2 | **37.6 ms** | 1 edge |
| 5 | **47.5 ms** | 10 edges |
| 10 | **49.6 ms** | 45 edges |
| 20 | **52.3 ms** | 190 edges |

### Edge Strengthening (LTP pathway)

| Coactivations | Time |
|---------------|------|
| 1x | **43 ms** |
| 3x | **56.5 ms** |
| 5x | **139.5 ms** |
| 10x | **44.9 ms** |

### Graph Persistence (save + reload)

| Edges | Time |
|-------|------|
| 10 | **86.7 ms** |
| 50 | **92.4 ms** |
| 100 | **103.3 ms** |

### Key Operations

| Operation | Time |
|-----------|------|
| Associative retrieval | **107 ms** |
| LTP threshold (5 coact) | **48 ms** |

**Conclusion:** Hebbian learning adds **<1ms overhead**. Embedding generation (ONNX) is the bottleneck.

---

## 2. Graph Memory (Knowledge Graph)

### Entity Operations

| Operation | Scale | Latency | Throughput |
|-----------|-------|---------|------------|
| Entity Get | 1000 entities | **763 ns** | ~1.3M ops/sec |
| Entity Search | 1000 entities | **775 ns** | ~1.3M ops/sec |
| Entity Add | 100 entities | 23.6 ms | ~4 ops/sec |

### Relationship Operations

| Operation | Scale | Latency |
|-----------|-------|---------|
| Query | 1000e/500r | **2.2 μs** |
| Add | 100 rels | 13.5 ms |

### Graph Traversal

| Depth | Latency | Notes |
|-------|---------|-------|
| 1-hop | **10 μs** | Direct neighbors |
| 2-hop | **20 μs** | Friends of friends |
| 3-hop | **30 μs** | Extended network |

### Salience & Hebbian Plasticity

| Operation | Scale | Latency |
|-----------|-------|---------|
| Hebbian Strengthen | 10 entities | **5.70 μs** |
| Hebbian Strengthen | 100 entities | **7.54 μs** |
| Hebbian Strengthen | 500 entities | **6.03 μs** |
| Salience Update | 1 entity | **8.49 μs** |
| Salience Update | 100 entities | **894 μs** |

---

## 3. Vector Search (Vamana/DiskANN)

| Query Type | Graph Size | Latency |
|------------|------------|---------|
| Top-5 ANN | 100 vectors | **~2 ms** |
| Top-5 ANN | 1000 vectors | **~5 ms** |
| Top-10 ANN | 1000 vectors | **~8 ms** |

### Embedding Generation (MiniLM-L6)

| Text Length | Latency |
|-------------|---------|
| Short (~10 words) | 15-25 ms |
| Medium (~50 words) | 20-30 ms |
| Batch (10 texts) | 50-80 ms |

---

## 4. Associative Retrieval (Density-Dependent Hybrid)

### Cross-Domain Association Discovery

| Scenario | Latency |
|----------|---------|
| coffee → AWS | **176 μs** |
| AWS → coffee | **189 μs** |
| Jeff → Seattle | **194 μs** |
| Temporal chain | **140 μs** |
| Team discovery | **227 μs** |

### Density-Dependent Weights

| Graph Density | Graph Weight | Latency |
|---------------|--------------|---------|
| Sparse (d=0.5) | 10% | **80 μs** |
| Medium (d=2.0) | 44% | **504 μs** |
| Dense (d=5.0) | ~50% | **2.73 ms** |

---

## 5. Proactive Memory Surfacing

Target: **<30ms** for real-time use

### Entity Extraction (Rule-based NER)

| Context Length | Time |
|----------------|------|
| Short (~6 words) | **1.33 μs** |
| Medium (~15 words) | **2.83 μs** |
| Long (~50 words) | **10.68 μs** |

### Full Relevance Pipeline

| Database Size | Latency | vs 30ms Target |
|---------------|---------|----------------|
| 10 memories | **61 μs** | ✅ 492x under |
| 50 memories | **630 μs** | ✅ 48x under |
| 100 memories | **374 μs** | ✅ 80x under |
| 200 memories | **660 μs** | ✅ 45x under |

---

## 6. Streaming Memory Extraction

### Content Hashing (Optimized)

| Benchmark | Latency |
|-----------|---------|
| hash_short | **36.58 ns** |
| hash_medium | **330.80 ns** |
| hash_long | **280.46 ns** |
| dedup_check_100_items | **37.70 ns** |

### Importance Calculation

| Benchmark | Latency |
|-----------|---------|
| short_neutral | **215.43 ns** |
| short_important | **59.20 ns** |
| with_entities | **77.25 ns** |
| error_content | **63.22 ns** |

---

## 7. External Integrations (Linear & GitHub)

### Linear Integration

| Operation | Latency |
|-----------|---------|
| Payload Parsing | **2.34 μs** |
| HMAC-SHA256 Verification | **864 ns** |
| Content Transformation | **903 ns** |
| Tag Extraction | **305 ns** |
| **Full Pipeline** | **6.06 μs** |

### GitHub Integration

| Operation | Latency |
|-----------|---------|
| Payload Parsing | **1.86 μs** |
| HMAC-SHA256 Verification | **842 ns** |
| Issue Transformation | **615 ns** |
| PR Transformation | **1.00 μs** |
| **Full Pipeline** | **4.40 μs** |

**Throughput:** ~150K/sec (Linear), ~225K/sec (GitHub) per core

---

## 8. Competitive Comparison

### vs. Neo4j

| Metric | Veld | Neo4j | Winner |
|--------|--------------|-------|--------|
| Entity Lookup | **763 ns** | 1-5 ms | Veld (local graph runtime) |
| Memory Footprint | ~50 MB | 1-4 GB | Veld |
| Cold Start | ~200 ms | 5-30 s | Veld |

## Product Comparison Note

Canonical product positioning now lives in the root `README.md` so it stays in one place.

This benchmark document keeps measured runtime and systems numbers. Product-level comparisons against Mem0, Cognee, Pinecone, and Zep are intentionally not duplicated here because those products are not all like-for-like benchmark targets.

---

## Store-Size Latency Lane

`bench/store_size_latency_probe.py`, run by `.github/workflows/store-size-latency.yml` (nightly and on demand; not a PR gate).

The 20-memory recall guard (`bench/recall_latency_guard.py`) cannot see store-size scaling: a path that costs O(N) per request costs the same on 20 memories before and after a fix. The lane seeds one tenant to N memories (default 10 000, through `POST /api/remember/batch`), creates a fresh tenant in the same run, and measures p50/p95 over 20 calls per route on each:

| Route | Large-tenant p95 limit |
|---|---|
| `POST /api/remember` | 2.0 s |
| `POST /api/recall` | 2.0 s |
| `GET /api/memory/{id}` | 0.2 s |
| `POST /api/relevant` | 1.0 s |
| `GET /api/users/{id}/stats` | 0.2 s |
| `GET /api/health/index` (per tenant) and `GET /health/index` (public) | 0.2 s |

The job fails when any large-tenant p95 exceeds its limit, when the large/fresh p95 ratio exceeds 5 (`--max-ratio`; fresh p95 is floored at 50 ms in the ratio), or when any call errors or hits the 60 s timeout. A failed call is charged at 60 s in the percentiles, never dropped. Limits can be overridden with `--thresholds <json>`. The JSON result records the commit sha, store size and timestamps and is uploaded as the `store-size-latency` artifact.

Reference point: a 15.6k-memory tenant measured `POST /api/remember` timing out at 60 s and `POST /api/recall` at 8-10 s, against 1.1 s for a write on a fresh tenant.

---

## Engine comparison (RocksDB vs redb)

Not yet measured on this branch. The table below is empty on purpose: run the commands, then paste the script output into it. Do not quote numbers from other machines or from the older sections of this document for this comparison.

### Commands

End-to-end through the HTTP API, one run per memory count. Build the release binary first, then:

```bash
VELD_BIN=target/release/veld scripts/bench/http_engine_compare.sh --memories 10000
VELD_BIN=target/release/veld scripts/bench/http_engine_compare.sh --memories 100000
```

The script starts `veld server` once per engine (default `rocksdb redb`) on a fresh temporary data dir and a free port, seeds the memories through `POST /api/remember`, runs `POST /api/recall` twice over the same deterministic queries (cold, then warm), runs `POST /api/proactive_context` with `auto_ingest` off, and records server RSS after seeding and after recall plus the data dir size. It writes one JSON file per engine to `bench-results/<timestamp>/` and prints the side-by-side table.

Flags: `--engines "A B"`, `--memories N` (default 10000), `--queries N` (default 2000), `--proactive-queries N` (default 200), `--concurrency C` (default 16), `--results-dir DIR`, `--bin PATH`, `--api-key KEY`, `--corpus-seed N`. Run `scripts/bench/http_engine_compare.sh --help` for the full text.

Tuning variables set in the calling environment are passed to the server unchanged: `VELD_STORAGE_DURABILITY` (`immediate` or `eventual`), `VELD_REDB_CACHE_BYTES`, `VELD_ROCKSDB_WRITE_BUFFER_BYTES`, `VELD_ROCKSDB_BLOCK_CACHE_BYTES`, plus the embedder selection `VELD_EMBEDDER` and `VELD_EMBEDDING_API_URL`. Record the values you used next to the results. The values are also saved in each JSON file.

Micro and cold-start benches, which isolate the engines from the HTTP and embedding layers:

```bash
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

---

## Running Benchmarks

```bash
# All benchmarks
ORT_DYLIB_PATH=/path/to/onnxruntime.dll cargo bench

# Specific benchmark
cargo bench --bench hebbian_benchmarks
cargo bench --bench memory_benchmarks
cargo bench --bench graph_benchmarks
```

---

*Last updated: December 16, 2025*
