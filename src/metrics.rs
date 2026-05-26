//! Production-grade metrics with Prometheus
//!
//! Exposes key operational metrics for monitoring and alerting:
//! - Request rates and latencies
//! - Memory usage and resource consumption
//! - Vector index performance
//! - Error rates and types
//!
//! NOTE: We intentionally avoid user_id in metric labels to prevent
//! high-cardinality explosion that can crash Prometheus.

use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    Registry,
};
use std::sync::{LazyLock, OnceLock};

/// Metrics initialization result
static METRICS_INIT: OnceLock<Result<(), MetricsError>> = OnceLock::new();

/// Error type for metrics initialization
#[derive(Debug, Clone)]
pub struct MetricsError {
    pub message: String,
}

impl std::fmt::Display for MetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metrics initialization failed: {}", self.message)
    }
}

impl std::error::Error for MetricsError {}

/// Create histogram opts with standard latency buckets
fn latency_histogram_opts(name: &str, help: &str) -> HistogramOpts {
    HistogramOpts::new(name, help).buckets(vec![
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    ])
}

/// Create histogram opts for fast operations (sub-millisecond)
fn fast_histogram_opts(name: &str, help: &str) -> HistogramOpts {
    HistogramOpts::new(name, help).buckets(vec![
        0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
    ])
}

/// Global metrics registry
pub static METRICS_REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

// ============================================================================
// Request Metrics
// ============================================================================

/// HTTP request duration in seconds
pub static HTTP_REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        latency_histogram_opts(
            "veld_http_request_duration_seconds",
            "HTTP request duration in seconds",
        ),
        &["method", "endpoint", "status"],
    )
    .expect("HTTP_REQUEST_DURATION metric must be valid at compile time")
});

/// Total HTTP requests
pub static HTTP_REQUESTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("veld_http_requests_total", "Total HTTP requests"),
        &["method", "endpoint", "status"],
    )
    .expect("HTTP_REQUESTS_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Memory Operation Metrics
// NOTE: No user_id in labels to prevent cardinality explosion
// ============================================================================

/// Memory store operations (record)
pub static MEMORY_STORE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("veld_store_total", "Total memory store operations"),
        &["result"],
    )
    .expect("MEMORY_STORE_TOTAL metric must be valid at compile time")
});

/// Memory store duration
pub static MEMORY_STORE_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_store_duration_seconds",
            "Memory store operation duration",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]),
    )
    .expect("MEMORY_STORE_DURATION metric must be valid at compile time")
});

/// Memory retrieve operations
pub static MEMORY_RETRIEVE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_retrieve_total",
            "Total memory retrieve operations",
        ),
        &["retrieval_mode", "result"],
    )
    .expect("MEMORY_RETRIEVE_TOTAL metric must be valid at compile time")
});

/// Memory retrieve duration
pub static MEMORY_RETRIEVE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "veld_retrieve_duration_seconds",
            "Memory retrieve operation duration",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["retrieval_mode"],
    )
    .expect("MEMORY_RETRIEVE_DURATION metric must be valid at compile time")
});

/// Results returned per query
pub static MEMORY_RETRIEVE_RESULTS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "veld_retrieve_results",
            "Number of results returned per query",
        )
        .buckets(vec![0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        &["retrieval_mode"],
    )
    .expect("MEMORY_RETRIEVE_RESULTS metric must be valid at compile time")
});

// ============================================================================
// Ontological Retrieval Metrics
// ============================================================================

/// Ontological intent confidence distribution per query
pub static ONTOLOGICAL_INTENT_CONFIDENCE: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_ontological_intent_confidence",
            "Distribution of inferred ontological intent confidence scores",
        )
        .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )
    .expect("ONTOLOGICAL_INTENT_CONFIDENCE metric must be valid at compile time")
});

/// Ontological re-rank boost applied to individual memories (Layer 4.9)
pub static ONTOLOGICAL_RERANK_BOOST_APPLIED: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_ontological_rerank_boost",
            "Distribution of ontological re-rank boost values applied to memories",
        )
        .buckets(vec![0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25]),
    )
    .expect("ONTOLOGICAL_RERANK_BOOST_APPLIED metric must be valid at compile time")
});

/// Queries where ontological intent was below confidence threshold (fallback to unfiltered)
pub static ONTOLOGICAL_FALLBACK_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_ontological_fallback_total",
        "Queries where ontological intent was below confidence threshold",
    )
    .expect("ONTOLOGICAL_FALLBACK_TOTAL metric must be valid at compile time")
});

/// Queries where ontological filtering was disabled due to high graph density
pub static ONTOLOGICAL_DENSITY_SKIP_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_ontological_density_skip_total",
        "Queries where ontological filtering was disabled due to high graph density",
    )
    .expect("ONTOLOGICAL_DENSITY_SKIP_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Embedding Metrics (P1.2: Instrument embed operations)
// ============================================================================

/// Embedding generation operations
pub static EMBEDDING_GENERATE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_embedding_generate_total",
            "Total embedding generations",
        ),
        &["mode", "result"], // mode: "onnx" or "simplified"
    )
    .expect("EMBEDDING_GENERATE_TOTAL metric must be valid at compile time")
});

/// Embedding generation duration
pub static EMBEDDING_GENERATE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "veld_embedding_generate_duration_seconds",
            "Embedding generation duration",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0,
        ]),
        &["mode"],
    )
    .expect("EMBEDDING_GENERATE_DURATION metric must be valid at compile time")
});

/// Background embed_and_index duration (deferred embedding path)
pub static EMBED_BACKGROUND_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_embed_background_duration_seconds",
            "Background embed_and_index operation duration",
        )
        .buckets(vec![0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]),
    )
    .expect("EMBED_BACKGROUND_DURATION metric must be valid at compile time")
});

/// Embedding timeout count
pub static EMBEDDING_TIMEOUT_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_embedding_timeout_total",
        "Total embedding generation timeouts",
    )
    .expect("EMBEDDING_TIMEOUT_TOTAL metric must be valid at compile time")
});

/// NER session lock timeout count
pub static NER_LOCK_TIMEOUT_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_ner_lock_timeout_total",
        "Total NER session lock timeouts (degraded entity extraction)",
    )
    .expect("NER_LOCK_TIMEOUT_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Memory Usage Metrics (aggregate, no per-user to avoid cardinality)
// ============================================================================

/// Active users in cache
pub static ACTIVE_USERS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_active_users",
        "Number of users with active memory sessions",
    )
    .expect("ACTIVE_USERS metric must be valid at compile time")
});

/// Total memories stored by tier (aggregate across all users)
pub static MEMORIES_BY_TIER: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new("veld_memories_by_tier", "Total memories by tier"),
        &["tier"], // tier: "working", "session", "longterm"
    )
    .expect("MEMORIES_BY_TIER metric must be valid at compile time")
});

/// Total memory system heap usage (estimated, aggregate)
pub static MEMORY_HEAP_BYTES_TOTAL: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_heap_bytes_total",
        "Total estimated heap usage across all users",
    )
    .expect("MEMORY_HEAP_BYTES_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Vector Index Metrics (aggregate)
// ============================================================================

/// Total vector index size (number of vectors across all users)
pub static VECTOR_INDEX_SIZE_TOTAL: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_vector_index_size_total",
        "Total number of vectors in all indices",
    )
    .expect("VECTOR_INDEX_SIZE_TOTAL metric must be valid at compile time")
});

/// Vector search operations
pub static VECTOR_SEARCH_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_vector_search_total",
            "Total vector search operations",
        ),
        &["result"],
    )
    .expect("VECTOR_SEARCH_TOTAL metric must be valid at compile time")
});

/// Vector search duration
pub static VECTOR_SEARCH_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(fast_histogram_opts(
        "veld_vector_search_duration_seconds",
        "Vector search duration",
    ))
    .expect("VECTOR_SEARCH_DURATION metric must be valid at compile time")
});

// ============================================================================
// Storage Metrics
// ============================================================================

/// RocksDB operations
pub static ROCKSDB_OPS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("veld_rocksdb_ops_total", "Total RocksDB operations"),
        &["operation", "result"], // operation: "get", "put", "delete"
    )
    .expect("ROCKSDB_OPS_TOTAL metric must be valid at compile time")
});

/// RocksDB operation duration
pub static ROCKSDB_OPS_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        fast_histogram_opts(
            "veld_rocksdb_ops_duration_seconds",
            "RocksDB operation duration",
        ),
        &["operation"],
    )
    .expect("ROCKSDB_OPS_DURATION metric must be valid at compile time")
});

/// Fallback deserialization branch hits (legacy migration observability)
pub static LEGACY_FALLBACK_BRANCH_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_legacy_fallback_branch_total",
            "Total fallback deserialization branch hits",
        ),
        &["branch"],
    )
    .expect("LEGACY_FALLBACK_BRANCH_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Intent Log Metrics (W5)
// ============================================================================

/// Intent log append operations
pub static INTENT_LOG_APPEND_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_intent_log_append_total",
            "Total intent log append operations",
        ),
        &["result"], // result: "ok", "payload_too_large", "io_error"
    )
    .expect("INTENT_LOG_APPEND_TOTAL metric must be valid at compile time")
});

/// Intent log append duration
pub static INTENT_LOG_APPEND_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(HistogramOpts::new(
        "veld_intent_log_append_duration_seconds",
        "Intent log append duration",
    ).buckets(vec![
        0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    ]))
    .expect("INTENT_LOG_APPEND_DURATION metric must be valid at compile time")
});

/// Intent log sync (fsync) operations
pub static INTENT_LOG_SYNC_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_intent_log_sync_total",
            "Total intent log sync (fsync) operations",
        ),
        &["result"], // result: "ok", "io_error"
    )
    .expect("INTENT_LOG_SYNC_TOTAL metric must be valid at compile time")
});

/// Intent log sync duration
pub static INTENT_LOG_SYNC_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(HistogramOpts::new(
        "veld_intent_log_sync_duration_seconds",
        "Intent log sync (fsync) duration",
    ).buckets(vec![
        0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    ]))
    .expect("INTENT_LOG_SYNC_DURATION metric must be valid at compile time")
});

/// Corrupt-tail truncation events
pub static INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_intent_log_truncate_corrupt_tail_total",
        "Total intent log corrupt-tail truncation events",
    )
    .expect("INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL metric must be valid at compile time")
});

/// Next LSN that will be assigned by the intent log
pub static INTENT_LOG_NEXT_LSN: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_intent_log_next_lsn",
        "LSN that will be assigned to the next intent log append",
    )
    .expect("INTENT_LOG_NEXT_LSN metric must be valid at compile time")
});

/// Durable end offset of the intent log (file size in bytes)
pub static INTENT_LOG_DURABLE_END_OFFSET_BYTES: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_intent_log_durable_end_offset_bytes",
        "Byte offset after the last durable intent log frame",
    )
    .expect("INTENT_LOG_DURABLE_END_OFFSET_BYTES metric must be valid at compile time")
});

/// Projection apply operations
pub static PROJECTION_APPLY_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_projection_apply_total",
            "Total projection apply operations",
        ),
        &["projection", "result"], // result: "ok", "error"
    )
    .expect("PROJECTION_APPLY_TOTAL metric must be valid at compile time")
});

/// Projection apply duration (labelled per projection)
pub static PROJECTION_APPLY_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "veld_projection_apply_duration_seconds",
            "Projection apply duration (labelled per projection)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        ]),
        &["projection"],
    )
    .expect("PROJECTION_APPLY_DURATION metric must be valid at compile time")
});

/// Records applied during replay (per projection)
pub static PROJECTION_REPLAY_RECORDS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_projection_replay_records_total",
            "Total records applied to a projection during replay",
        ),
        &["projection"],
    )
    .expect("PROJECTION_REPLAY_RECORDS_TOTAL metric must be valid at compile time")
});

/// Checkpoint persist operations (per projection)
pub static CHECKPOINT_PERSIST_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_checkpoint_persist_total",
            "Total projection checkpoint persist operations",
        ),
        &["projection", "result"], // result: "ok", "error"
    )
    .expect("CHECKPOINT_PERSIST_TOTAL metric must be valid at compile time")
});

/// Last persisted checkpoint LSN per projection
pub static PROJECTION_CHECKPOINT_LSN: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "veld_projection_checkpoint_lsn",
            "Last persisted checkpoint LSN per projection",
        ),
        &["projection"],
    )
    .expect("PROJECTION_CHECKPOINT_LSN metric must be valid at compile time")
});

/// Projection replay lag in records (next_lsn - checkpoint_lsn)
pub static PROJECTION_LAG_RECORDS: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "veld_projection_lag_records",
            "How far a projection lags the head of the intent log, in records",
        ),
        &["projection"],
    )
    .expect("PROJECTION_LAG_RECORDS metric must be valid at compile time")
});

// ============================================================================
// Error Metrics
// ============================================================================

/// Total errors by type
pub static ERRORS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("veld_errors_total", "Total errors by type"),
        &["error_type", "endpoint"],
    )
    .expect("ERRORS_TOTAL metric must be valid at compile time")
});

/// Resource limit rejections
pub static RESOURCE_LIMIT_REJECTIONS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_resource_limit_rejections",
            "Requests rejected due to resource limits",
        ),
        &["resource"],
    )
    .expect("RESOURCE_LIMIT_REJECTIONS metric must be valid at compile time")
});

// ============================================================================
// Concurrency Metrics (P0.8)
// ============================================================================

/// Current concurrent requests
pub static CONCURRENT_REQUESTS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_concurrent_requests",
        "Current number of concurrent requests",
    )
    .expect("CONCURRENT_REQUESTS metric must be valid at compile time")
});

/// Request queue size (if queuing implemented)
pub static REQUEST_QUEUE_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new("veld_request_queue_size", "Number of queued requests")
        .expect("REQUEST_QUEUE_SIZE metric must be valid at compile time")
});

// ============================================================================
// Hebbian Learning Metrics
// ============================================================================

/// Hebbian reinforcement operations
pub static HEBBIAN_REINFORCE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_hebbian_reinforce_total",
            "Total Hebbian reinforcement operations",
        ),
        &["outcome", "result"], // outcome: "helpful", "misleading", "neutral"
    )
    .expect("HEBBIAN_REINFORCE_TOTAL metric must be valid at compile time")
});

/// Hebbian reinforcement duration
pub static HEBBIAN_REINFORCE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "veld_hebbian_reinforce_duration_seconds",
            "Hebbian reinforcement operation duration",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]),
        &["outcome"],
    )
    .expect("HEBBIAN_REINFORCE_DURATION metric must be valid at compile time")
});

// ============================================================================
// Consolidation Metrics
// ============================================================================

/// Memory consolidation operations
pub static CONSOLIDATE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_consolidate_total",
            "Total memory consolidation operations",
        ),
        &["result"],
    )
    .expect("CONSOLIDATE_TOTAL metric must be valid at compile time")
});

/// Memory consolidation duration
pub static CONSOLIDATE_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_consolidate_duration_seconds",
            "Memory consolidation operation duration",
        )
        .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
    )
    .expect("CONSOLIDATE_DURATION metric must be valid at compile time")
});

// ============================================================================
// Batch Operation Metrics
// ============================================================================

/// Batch store duration
pub static BATCH_STORE_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_batch_store_duration_seconds",
            "Batch memory store operation duration",
        )
        .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]),
    )
    .expect("BATCH_STORE_DURATION metric must be valid at compile time")
});

/// Batch store size
pub static BATCH_STORE_SIZE: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_batch_store_size",
            "Number of memories in batch store operations",
        )
        .buckets(vec![
            1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
        ]),
    )
    .expect("BATCH_STORE_SIZE metric must be valid at compile time")
});

// ============================================================================
// Write Gate Metrics (Predictive Coding)
// ============================================================================

/// Memories absorbed by the write gate (redundant content detected)
pub static WRITE_GATE_ABSORBED: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_write_gate_absorbed_total",
        "Memories absorbed by write gate due to semantic redundancy",
    )
    .expect("WRITE_GATE_ABSORBED metric must be valid at compile time")
});

// ============================================================================
// Dream Replay Metrics (Consolidation Discovery)
// ============================================================================

/// Edges discovered during dream replay phase
pub static DREAM_REPLAY_EDGES_CREATED: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_dream_replay_edges_created_total",
        "Graph edges discovered during dream replay consolidation",
    )
    .expect("DREAM_REPLAY_EDGES_CREATED metric must be valid at compile time")
});

/// Memory pairs evaluated during dream replay
pub static DREAM_REPLAY_PAIRS_EVALUATED: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_dream_replay_pairs_evaluated_total",
        "Memory pairs compared during dream replay",
    )
    .expect("DREAM_REPLAY_PAIRS_EVALUATED metric must be valid at compile time")
});

// ============================================================================
// Reconsolidation Metrics
// ============================================================================

/// Memories reconsolidated (importance boosted on retrieval)
pub static RECONSOLIDATION_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_reconsolidation_total",
        "Memories reconsolidated (importance updated on retrieval)",
    )
    .expect("RECONSOLIDATION_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Embedding Cache Metrics (SHO-68)
// ============================================================================

/// Embedding cache operations (query cache)
pub static EMBEDDING_CACHE_QUERY: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_embedding_cache_query_total",
            "Query embedding cache operations",
        ),
        &["result"], // result: "hit" or "miss"
    )
    .expect("EMBEDDING_CACHE_QUERY metric must be valid at compile time")
});

/// Embedding cache operations (content cache)
pub static EMBEDDING_CACHE_CONTENT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "veld_embedding_cache_content_total",
            "Content embedding cache operations",
        ),
        &["result"], // result: "hit" or "miss"
    )
    .expect("EMBEDDING_CACHE_CONTENT metric must be valid at compile time")
});

/// Current cache size (query cache)
pub static EMBEDDING_CACHE_QUERY_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_embedding_cache_query_size",
        "Current number of entries in query embedding cache",
    )
    .expect("EMBEDDING_CACHE_QUERY_SIZE metric must be valid at compile time")
});

/// Current cache size (content cache)
pub static EMBEDDING_CACHE_CONTENT_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "veld_embedding_cache_content_size",
        "Current number of entries in content embedding cache",
    )
    .expect("EMBEDDING_CACHE_CONTENT_SIZE metric must be valid at compile time")
});

// ============================================================================
// Retrieval Signal Variance Metrics
// ============================================================================

pub static RETRIEVAL_VARIANCE_SEMANTIC: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_retrieval_variance_semantic",
            "Average semantic signal contribution to final retrieval score",
        )
        .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )
    .expect("RETRIEVAL_VARIANCE_SEMANTIC metric must be valid at compile time")
});

pub static RETRIEVAL_VARIANCE_GRAPH: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_retrieval_variance_graph",
            "Average graph signal contribution to final retrieval score",
        )
        .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )
    .expect("RETRIEVAL_VARIANCE_GRAPH metric must be valid at compile time")
});

pub static RETRIEVAL_VARIANCE_LINGUISTIC: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "veld_retrieval_variance_linguistic",
            "Average linguistic signal contribution to final retrieval score",
        )
        .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )
    .expect("RETRIEVAL_VARIANCE_LINGUISTIC metric must be valid at compile time")
});

pub static SUPPRESSOR_DETECTIONS_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "veld_suppressor_detections_total",
        "Total suppressor events where multi-signal competition removed or demoted a candidate",
    )
    .expect("SUPPRESSOR_DETECTIONS_TOTAL metric must be valid at compile time")
});

/// Register all metrics with the global registry
///
/// # Returns
/// - `Ok(())` if all metrics registered successfully
/// - `Err(MetricsError)` if any metric fails to register
///
/// # Behavior
/// - Registration is idempotent - calling multiple times is safe
/// - On failure, server should log warning and continue (degraded mode)
/// - Prometheus scraping will simply return empty metrics if registration failed
pub fn register_metrics() -> Result<(), MetricsError> {
    // Check if already initialized
    if let Some(result) = METRICS_INIT.get() {
        return result.clone();
    }

    let result = do_register_metrics();
    let _ = METRICS_INIT.set(result.clone());
    result
}

fn do_register_metrics() -> Result<(), MetricsError> {
    let mut errors = Vec::new();

    // Helper macro to reduce boilerplate
    macro_rules! register {
        ($metric:expr, $name:expr) => {
            if let Err(e) = METRICS_REGISTRY.register(Box::new($metric.clone())) {
                errors.push(format!("{}: {}", $name, e));
            }
        };
    }

    // Request metrics
    register!(HTTP_REQUEST_DURATION, "HTTP_REQUEST_DURATION");
    register!(HTTP_REQUESTS_TOTAL, "HTTP_REQUESTS_TOTAL");

    // Memory operation metrics
    register!(MEMORY_STORE_TOTAL, "MEMORY_STORE_TOTAL");
    register!(MEMORY_STORE_DURATION, "MEMORY_STORE_DURATION");
    register!(MEMORY_RETRIEVE_TOTAL, "MEMORY_RETRIEVE_TOTAL");
    register!(MEMORY_RETRIEVE_DURATION, "MEMORY_RETRIEVE_DURATION");
    register!(MEMORY_RETRIEVE_RESULTS, "MEMORY_RETRIEVE_RESULTS");

    // Ontological retrieval metrics
    register!(
        ONTOLOGICAL_INTENT_CONFIDENCE,
        "ONTOLOGICAL_INTENT_CONFIDENCE"
    );
    register!(
        ONTOLOGICAL_RERANK_BOOST_APPLIED,
        "ONTOLOGICAL_RERANK_BOOST_APPLIED"
    );
    register!(ONTOLOGICAL_FALLBACK_TOTAL, "ONTOLOGICAL_FALLBACK_TOTAL");
    register!(
        ONTOLOGICAL_DENSITY_SKIP_TOTAL,
        "ONTOLOGICAL_DENSITY_SKIP_TOTAL"
    );

    // Embedding metrics
    register!(EMBEDDING_GENERATE_TOTAL, "EMBEDDING_GENERATE_TOTAL");
    register!(EMBEDDING_GENERATE_DURATION, "EMBEDDING_GENERATE_DURATION");
    register!(EMBEDDING_TIMEOUT_TOTAL, "EMBEDDING_TIMEOUT_TOTAL");
    register!(EMBED_BACKGROUND_DURATION, "EMBED_BACKGROUND_DURATION");
    register!(NER_LOCK_TIMEOUT_TOTAL, "NER_LOCK_TIMEOUT_TOTAL");

    // Memory usage metrics (aggregate)
    register!(ACTIVE_USERS, "ACTIVE_USERS");
    register!(MEMORIES_BY_TIER, "MEMORIES_BY_TIER");
    register!(MEMORY_HEAP_BYTES_TOTAL, "MEMORY_HEAP_BYTES_TOTAL");

    // Vector index metrics (aggregate)
    register!(VECTOR_INDEX_SIZE_TOTAL, "VECTOR_INDEX_SIZE_TOTAL");
    register!(VECTOR_SEARCH_TOTAL, "VECTOR_SEARCH_TOTAL");
    register!(VECTOR_SEARCH_DURATION, "VECTOR_SEARCH_DURATION");

    // Storage metrics
    register!(ROCKSDB_OPS_TOTAL, "ROCKSDB_OPS_TOTAL");
    register!(ROCKSDB_OPS_DURATION, "ROCKSDB_OPS_DURATION");
    register!(LEGACY_FALLBACK_BRANCH_TOTAL, "LEGACY_FALLBACK_BRANCH_TOTAL");

    // Intent log metrics (W5)
    register!(INTENT_LOG_APPEND_TOTAL, "INTENT_LOG_APPEND_TOTAL");
    register!(INTENT_LOG_APPEND_DURATION, "INTENT_LOG_APPEND_DURATION");
    register!(INTENT_LOG_SYNC_TOTAL, "INTENT_LOG_SYNC_TOTAL");
    register!(INTENT_LOG_SYNC_DURATION, "INTENT_LOG_SYNC_DURATION");
    register!(
        INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL,
        "INTENT_LOG_TRUNCATE_CORRUPT_TAIL_TOTAL"
    );
    register!(INTENT_LOG_NEXT_LSN, "INTENT_LOG_NEXT_LSN");
    register!(
        INTENT_LOG_DURABLE_END_OFFSET_BYTES,
        "INTENT_LOG_DURABLE_END_OFFSET_BYTES"
    );
    register!(PROJECTION_APPLY_TOTAL, "PROJECTION_APPLY_TOTAL");
    register!(PROJECTION_APPLY_DURATION, "PROJECTION_APPLY_DURATION");
    register!(
        PROJECTION_REPLAY_RECORDS_TOTAL,
        "PROJECTION_REPLAY_RECORDS_TOTAL"
    );
    register!(CHECKPOINT_PERSIST_TOTAL, "CHECKPOINT_PERSIST_TOTAL");
    register!(PROJECTION_CHECKPOINT_LSN, "PROJECTION_CHECKPOINT_LSN");
    register!(PROJECTION_LAG_RECORDS, "PROJECTION_LAG_RECORDS");

    // Error metrics
    register!(ERRORS_TOTAL, "ERRORS_TOTAL");
    register!(RESOURCE_LIMIT_REJECTIONS, "RESOURCE_LIMIT_REJECTIONS");

    // Concurrency metrics
    register!(CONCURRENT_REQUESTS, "CONCURRENT_REQUESTS");
    register!(REQUEST_QUEUE_SIZE, "REQUEST_QUEUE_SIZE");

    // Hebbian learning metrics
    register!(HEBBIAN_REINFORCE_TOTAL, "HEBBIAN_REINFORCE_TOTAL");
    register!(HEBBIAN_REINFORCE_DURATION, "HEBBIAN_REINFORCE_DURATION");

    // Consolidation metrics
    register!(CONSOLIDATE_TOTAL, "CONSOLIDATE_TOTAL");
    register!(CONSOLIDATE_DURATION, "CONSOLIDATE_DURATION");

    // Batch operation metrics
    register!(BATCH_STORE_DURATION, "BATCH_STORE_DURATION");
    register!(BATCH_STORE_SIZE, "BATCH_STORE_SIZE");

    // Write gate metrics
    register!(WRITE_GATE_ABSORBED, "WRITE_GATE_ABSORBED");

    // Dream replay metrics
    register!(DREAM_REPLAY_EDGES_CREATED, "DREAM_REPLAY_EDGES_CREATED");
    register!(DREAM_REPLAY_PAIRS_EVALUATED, "DREAM_REPLAY_PAIRS_EVALUATED");

    // Reconsolidation metrics
    register!(RECONSOLIDATION_TOTAL, "RECONSOLIDATION_TOTAL");

    // Embedding cache metrics (SHO-68)
    register!(EMBEDDING_CACHE_QUERY, "EMBEDDING_CACHE_QUERY");
    register!(EMBEDDING_CACHE_CONTENT, "EMBEDDING_CACHE_CONTENT");
    register!(EMBEDDING_CACHE_QUERY_SIZE, "EMBEDDING_CACHE_QUERY_SIZE");
    register!(EMBEDDING_CACHE_CONTENT_SIZE, "EMBEDDING_CACHE_CONTENT_SIZE");
    register!(RETRIEVAL_VARIANCE_SEMANTIC, "RETRIEVAL_VARIANCE_SEMANTIC");
    register!(RETRIEVAL_VARIANCE_GRAPH, "RETRIEVAL_VARIANCE_GRAPH");
    register!(
        RETRIEVAL_VARIANCE_LINGUISTIC,
        "RETRIEVAL_VARIANCE_LINGUISTIC"
    );
    register!(SUPPRESSOR_DETECTIONS_TOTAL, "SUPPRESSOR_DETECTIONS_TOTAL");

    if errors.is_empty() {
        Ok(())
    } else {
        Err(MetricsError {
            message: errors.join("; "),
        })
    }
}

/// Helper to time operations with histogram (RAII pattern)
/// Usage: let _timer = Timer::new(SOME_HISTOGRAM.clone());
pub struct Timer {
    histogram: Histogram,
    start: std::time::Instant,
}

impl Timer {
    /// Create timer that records duration to histogram on drop
    pub fn new(histogram: Histogram) -> Self {
        Self {
            histogram,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();
        self.histogram.observe(duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::core::Metric;

    #[test]
    fn test_metrics_registration_is_idempotent() {
        // First registration should succeed
        let result1 = register_metrics();
        // Second registration should also succeed (returns cached result)
        let result2 = register_metrics();

        // Both should have same result
        assert_eq!(result1.is_ok(), result2.is_ok());
    }

    #[test]
    fn test_timer_records_duration() {
        // Create a test histogram
        let histogram = Histogram::with_opts(HistogramOpts::new(
            "test_timer_histogram",
            "Test histogram for timer",
        ))
        .unwrap();

        {
            let _timer = Timer::new(histogram.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Histogram should have recorded one observation
        let metric = histogram.metric();
        assert_eq!(metric.get_histogram().get_sample_count(), 1);
        // Duration should be at least 10ms
        assert!(metric.get_histogram().get_sample_sum() >= 0.01);
    }
}
