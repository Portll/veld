<!-- GENERATED FILE — do not edit by hand.
     Source: src/**/*.rs (register_* macro calls)
     Generator: docs/generators/src/bin/gen-metrics.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-metrics`) -->

# Metrics Reference

Veld exposes **39** Prometheus metrics on the `/metrics` endpoint. Visibility is configured by `VELD_METRICS_PUBLIC` — when `true`, `/metrics` is unauthenticated; otherwise it requires an API key.

| Name | Kind | Help | Source |
|---|---|---|---|
| `veld_batch_store_duration_seconds` | Histogram | Batch memory store operation duration | `src/metrics.rs` |
| `veld_batch_store_size` | Histogram | Number of memories in batch store operations | `src/metrics.rs` |
| `veld_checkpoint_persist_total` | IntCounterVec | Total projection checkpoint persist operations | `src/metrics.rs` |
| `veld_consolidate_duration_seconds` | Histogram | Memory consolidation operation duration | `src/metrics.rs` |
| `veld_consolidate_total` | IntCounterVec | Total memory consolidation operations | `src/metrics.rs` |
| `veld_embed_background_duration_seconds` | Histogram | Background embed_and_index operation duration | `src/metrics.rs` |
| `veld_embedding_cache_content_total` | IntCounterVec | Content embedding cache operations | `src/metrics.rs` |
| `veld_embedding_cache_query_total` | IntCounter | Query embedding cache operations | `src/metrics.rs` |
| `veld_embedding_generate_duration_seconds` | HistogramVec | Embedding generation duration | `src/metrics.rs` |
| `veld_embedding_generate_total` | IntCounterVec | Total embedding generations | `src/metrics.rs` |
| `veld_errors_total` | IntCounterVec | Total errors by type | `src/metrics.rs` |
| `veld_hebbian_reinforce_duration_seconds` | HistogramVec | Hebbian reinforcement operation duration | `src/metrics.rs` |
| `veld_hebbian_reinforce_total` | IntGauge | Total Hebbian reinforcement operations | `src/metrics.rs` |
| `veld_http_request_duration_seconds` | HistogramVec | HTTP request duration in seconds | `src/metrics.rs` |
| `veld_http_requests_total` | IntCounterVec | Total HTTP requests | `src/metrics.rs` |
| `veld_intent_log_append_duration_seconds` | Histogram | Intent log append duration | `src/metrics.rs` |
| `veld_intent_log_append_total` | IntCounterVec | Total intent log append operations | `src/metrics.rs` |
| `veld_intent_log_sync_duration_seconds` | Histogram | Intent log sync (fsync) duration | `src/metrics.rs` |
| `veld_intent_log_sync_total` | IntCounterVec | Total intent log sync (fsync) operations | `src/metrics.rs` |
| `veld_legacy_fallback_branch_total` | HistogramVec | Total fallback deserialization branch hits | `src/metrics.rs` |
| `veld_memories_by_tier` | IntGauge | Total memories by tier | `src/metrics.rs` |
| `veld_ontological_intent_confidence` | Histogram | Distribution of inferred ontological intent confidence scores | `src/metrics.rs` |
| `veld_ontological_rerank_boost` | Histogram | Distribution of ontological re-rank boost values applied to memories | `src/metrics.rs` |
| `veld_projection_apply_duration_seconds` | HistogramVec | Projection apply duration (labelled per projection) | `src/metrics.rs` |
| `veld_projection_apply_total` | IntGauge | Total projection apply operations | `src/metrics.rs` |
| `veld_projection_checkpoint_lsn` | IntGaugeVec | Last persisted checkpoint LSN per projection | `src/metrics.rs` |
| `veld_projection_lag_records` | IntGaugeVec | How far a projection lags the head of the intent log, in records | `src/metrics.rs` |
| `veld_projection_replay_records_total` | IntCounterVec | Total records applied to a projection during replay | `src/metrics.rs` |
| `veld_resource_limit_rejections` | IntCounterVec | Requests rejected due to resource limits | `src/metrics.rs` |
| `veld_retrieval_variance_graph` | Histogram | Average graph signal contribution to final retrieval score | `src/metrics.rs` |
| `veld_retrieval_variance_linguistic` | Histogram | Average linguistic signal contribution to final retrieval score | `src/metrics.rs` |
| `veld_retrieval_variance_semantic` | IntGauge | Average semantic signal contribution to final retrieval score | `src/metrics.rs` |
| `veld_retrieve_duration_seconds` | HistogramVec | Memory retrieve operation duration | `src/metrics.rs` |
| `veld_retrieve_results` | HistogramVec | Number of results returned per query | `src/metrics.rs` |
| `veld_retrieve_total` | IntCounterVec | Total memory retrieve operations | `src/metrics.rs` |
| `veld_rocksdb_ops_total` | Histogram | Total RocksDB operations | `src/metrics.rs` |
| `veld_store_duration_seconds` | Histogram | Memory store operation duration | `src/metrics.rs` |
| `veld_store_total` | IntCounterVec | Total memory store operations | `src/metrics.rs` |
| `veld_vector_search_total` | IntGauge | Total vector search operations | `src/metrics.rs` |

---

*Metric kind is the suffix of the macro: `counter`, `gauge`, `histogram`, `int_counter`, `int_counter_vec`, `histogram_vec`, etc.*
