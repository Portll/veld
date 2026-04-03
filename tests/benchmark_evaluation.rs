//! Shodh-Memory Retrieval Benchmark
//!
//! Evaluates retrieval quality across 4 query types:
//! - Single-hop: direct fact lookup
//! - Temporal: time-sensitive queries
//! - Multi-hop: queries requiring connecting multiple memories
//! - Open-domain: broad topical queries
//!
//! Scoring metrics:
//! - Precision@3: of top-3 results, how many are relevant?
//! - Precision@5: of top-5 results, how many are relevant?
//! - Recall@5: of all relevant memories, how many appear in top-5?
//! - MRR: mean reciprocal rank of first relevant result
//! - Latency: wall-clock time per query
//!
//! Run: cargo test --test benchmark_evaluation -- --nocapture

use std::time::Instant;

use veld::memory::types::{Experience, ExperienceType, Query};
use veld::memory::{MemoryConfig, MemoryId, MemorySystem};
use tempfile::TempDir;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct BenchmarkMemory {
    content: &'static str,
    memory_type: ExperienceType,
    tags: Vec<&'static str>,
    importance: f32,
}

struct BenchmarkQuery {
    query: &'static str,
    query_type: &'static str,
    expected_memory_indices: Vec<usize>,
    expected_absent: Vec<usize>,
}

struct EvaluationResult {
    query_type: String,
    precision_at_3: f32,
    precision_at_5: f32,
    recall_at_5: f32,
    mrr: f32,
    latency_ms: u64,
}

struct QueryEvaluation {
    query_text: String,
    query_type: String,
    precision_at_3: f32,
    precision_at_5: f32,
    recall_at_5: f32,
    reciprocal_rank: f32,
    latency_ms: u64,
    retrieved_indices: Vec<usize>,
    expected_indices: Vec<usize>,
}

// =============================================================================
// TEST DATA
// =============================================================================

fn benchmark_memories() -> Vec<BenchmarkMemory> {
    vec![
        // 0: Architecture decision — database choice
        BenchmarkMemory {
            content: "Architecture decision: we chose PostgreSQL as the primary database for the \
                      payment service because it provides strong ACID guarantees, mature JSON \
                      support via jsonb columns, and proven horizontal read-scaling with \
                      streaming replication. The team evaluated DynamoDB and CockroachDB but \
                      rejected them due to vendor lock-in and operational complexity respectively.",
            memory_type: ExperienceType::Decision,
            tags: vec!["architecture", "database", "postgresql"],
            importance: 0.9,
        },
        // 1: Bug fix — race condition
        BenchmarkMemory {
            content: "Fixed a critical race condition in the order processing pipeline last week. \
                      The bug manifested as duplicate charge events when two concurrent requests \
                      hit the idempotency check before either committed. Root cause was a missing \
                      SELECT FOR UPDATE lock on the idempotency_keys table. Added pessimistic \
                      locking and a unique constraint as defense-in-depth.",
            memory_type: ExperienceType::Error,
            tags: vec!["bug", "race-condition", "payments"],
            importance: 0.85,
        },
        // 2: Team preference — Sarah's code review style
        BenchmarkMemory {
            content: "Sarah prefers small, focused pull requests (under 300 lines) and insists on \
                      separate commits for refactoring vs. feature changes. She reviews fastest \
                      on Tuesday and Wednesday mornings. She is particularly thorough about error \
                      handling paths and edge cases in payment-related code.",
            memory_type: ExperienceType::Observation,
            tags: vec!["team", "code-review", "sarah"],
            importance: 0.6,
        },
        // 3: Code pattern — retry with exponential backoff
        BenchmarkMemory {
            content: "Learned that our retry strategy for external API calls should use \
                      exponential backoff with jitter starting at 100ms base delay, capped at \
                      30 seconds. The payment gateway returns 429 status codes under load and \
                      a fixed 1-second retry was causing thundering herd effects. Implemented \
                      a RetryPolicy trait with configurable max_attempts and backoff_factor.",
            memory_type: ExperienceType::Learning,
            tags: vec!["pattern", "retry", "api"],
            importance: 0.8,
        },
        // 4: Error encounter — OOM in staging
        BenchmarkMemory {
            content: "The staging environment ran out of memory during load testing because the \
                      in-memory cache had no eviction policy. The analytics aggregation service \
                      was caching every unique user session without bounds. Added an LRU eviction \
                      strategy with a 512MB cap and 60-second TTL for session aggregates.",
            memory_type: ExperienceType::Error,
            tags: vec!["error", "oom", "cache", "staging"],
            importance: 0.75,
        },
        // 5: Tool configuration — CI/CD pipeline
        BenchmarkMemory {
            content: "Configured the CI/CD pipeline to run unit tests in parallel using 4 workers, \
                      integration tests sequentially to avoid port conflicts, and deploy to staging \
                      automatically on merge to main. The GitHub Actions workflow uses a matrix \
                      strategy for Rust 1.75 and 1.76 toolchains. Build caching via sccache \
                      reduced CI time from 12 minutes to 4 minutes.",
            memory_type: ExperienceType::Task,
            tags: vec!["ci-cd", "github-actions", "tooling"],
            importance: 0.7,
        },
        // 6: Architecture decision — event sourcing
        BenchmarkMemory {
            content: "Architecture decision: adopted event sourcing for the inventory domain \
                      because the business needs a complete audit trail of stock movements. \
                      Every stock change is recorded as an immutable event (StockReceived, \
                      StockAllocated, StockShipped, StockAdjusted). Projections rebuild \
                      current state. Chose not to use CQRS because read patterns are simple \
                      enough that a single read model suffices.",
            memory_type: ExperienceType::Decision,
            tags: vec!["architecture", "event-sourcing", "inventory"],
            importance: 0.85,
        },
        // 7: Bug fix — timezone handling
        BenchmarkMemory {
            content: "Fixed a subtle timezone bug in the reporting module yesterday. The daily \
                      revenue aggregation was using server-local time (UTC+0) instead of the \
                      merchant's configured timezone, causing transactions near midnight to \
                      appear in the wrong day's report. Switched all date boundaries to use \
                      the merchant's tz from their profile settings.",
            memory_type: ExperienceType::Error,
            tags: vec!["bug", "timezone", "reporting"],
            importance: 0.7,
        },
        // 8: Team preference — Marcus's deployment process
        BenchmarkMemory {
            content: "Marcus, the SRE lead, requires all production deployments to include a \
                      rollback plan documented in the deployment ticket. He prefers blue-green \
                      deployments over rolling updates for database migration releases. He \
                      monitors Datadog dashboards for 30 minutes post-deploy before signing off.",
            memory_type: ExperienceType::Observation,
            tags: vec!["team", "deployment", "marcus", "sre"],
            importance: 0.65,
        },
        // 9: Code pattern — structured logging
        BenchmarkMemory {
            content: "Established a structured logging convention across all Rust services: use \
                      tracing with span-based context propagation, include request_id and user_id \
                      in every span, log at INFO for request lifecycle events, WARN for \
                      recoverable errors, and ERROR only for actionable failures. Switched from \
                      env_logger to tracing-subscriber with JSON output for production.",
            memory_type: ExperienceType::Learning,
            tags: vec!["pattern", "logging", "observability"],
            importance: 0.75,
        },
        // 10: Performance optimization — query plan
        BenchmarkMemory {
            content: "Discovered that the product search endpoint was doing a sequential scan on \
                      the 2M-row products table because the GIN index on the tsvector column was \
                      not being used when combined with a category filter. Adding a composite \
                      index on (category_id, textsearch_vector) brought p99 latency from 800ms \
                      to 12ms. EXPLAIN ANALYZE confirmed the index-only scan.",
            memory_type: ExperienceType::Discovery,
            tags: vec!["performance", "postgresql", "indexing"],
            importance: 0.85,
        },
        // 11: Security finding — JWT validation
        BenchmarkMemory {
            content: "Security review finding: the authentication middleware was not validating \
                      the JWT audience claim, which meant tokens issued for the admin portal \
                      could be used against the customer API. Added aud claim validation to \
                      all service endpoints and rotated the signing keys as a precaution. \
                      Filed as a P1 security incident.",
            memory_type: ExperienceType::Error,
            tags: vec!["security", "jwt", "authentication"],
            importance: 0.95,
        },
        // 12: Testing strategy — property-based testing
        BenchmarkMemory {
            content: "Introduced property-based testing with proptest for the serialization \
                      layer. Found 3 edge cases in the first week: a panic on empty strings \
                      in the tag parser, integer overflow in the pagination offset calculation, \
                      and a Unicode normalization inconsistency in search indexing. Property \
                      tests now run in CI alongside unit tests.",
            memory_type: ExperienceType::Learning,
            tags: vec!["testing", "proptest", "quality"],
            importance: 0.8,
        },
        // 13: Infrastructure — Kubernetes resource limits
        BenchmarkMemory {
            content: "Set Kubernetes resource limits for the API gateway pods: requests of \
                      256Mi memory and 250m CPU, limits of 512Mi memory and 500m CPU. The \
                      HPA scales between 3 and 12 replicas based on CPU utilization at 70% \
                      target. Pod disruption budget ensures at least 2 replicas during \
                      rolling updates.",
            memory_type: ExperienceType::Task,
            tags: vec!["infrastructure", "kubernetes", "scaling"],
            importance: 0.7,
        },
        // 14: Architecture decision — GraphQL vs REST
        BenchmarkMemory {
            content: "Architecture decision: chose REST over GraphQL for the public merchant \
                      API because our consumers are backend integrations, not frontend apps. \
                      REST offers simpler caching with HTTP semantics, better tooling support \
                      for webhook signatures, and lower complexity for the integration team. \
                      GraphQL remains an option for the internal dashboard BFF layer.",
            memory_type: ExperienceType::Decision,
            tags: vec!["architecture", "api-design", "rest"],
            importance: 0.8,
        },
        // 15: Bug fix — connection pool exhaustion
        BenchmarkMemory {
            content: "Diagnosed and fixed a connection pool exhaustion issue this week in the \
                      notification service. The service was acquiring a database connection \
                      before calling the external email provider, holding it for the full \
                      HTTP timeout (30s). Refactored to release the connection before the \
                      external call and reacquire after, reducing peak pool usage from 48 \
                      to 6 connections.",
            memory_type: ExperienceType::Error,
            tags: vec!["bug", "connection-pool", "performance"],
            importance: 0.8,
        },
        // 16: Code pattern — error handling convention
        BenchmarkMemory {
            content: "Adopted thiserror for domain error types and anyhow for application-level \
                      error propagation. Each service defines a ServiceError enum with variants \
                      for NotFound, Conflict, Validation, and Internal. The HTTP handler maps \
                      these to status codes. Never expose internal error messages to API callers; \
                      log the full context server-side and return a correlation ID.",
            memory_type: ExperienceType::Learning,
            tags: vec!["pattern", "error-handling", "rust"],
            importance: 0.75,
        },
        // 17: Monitoring — alerting thresholds
        BenchmarkMemory {
            content: "Configured alerting thresholds in Datadog: error rate above 1% for 5 \
                      minutes triggers a warning, above 5% triggers a page. P99 latency \
                      above 500ms for 10 minutes triggers investigation. Memory usage above \
                      80% of limit triggers a warning. These thresholds were calibrated from \
                      3 months of production baseline data.",
            memory_type: ExperienceType::Task,
            tags: vec!["monitoring", "alerting", "datadog"],
            importance: 0.7,
        },
        // 18: Data migration — user table restructure
        BenchmarkMemory {
            content: "Completed a zero-downtime migration of the users table to split the \
                      monolithic profile blob into normalized columns. Used the expand-contract \
                      pattern: added new columns, deployed dual-write code, backfilled 1.2M \
                      rows in batches of 5000, validated checksums, then dropped the old \
                      column. Total migration took 3 days including validation.",
            memory_type: ExperienceType::Task,
            tags: vec!["migration", "database", "users"],
            importance: 0.75,
        },
        // 19: Team process — incident response
        BenchmarkMemory {
            content: "After the payment outage last month, established a formal incident \
                      response process: declare severity (SEV1-3), assign incident commander, \
                      communicate via dedicated Slack channel, post status updates every 15 \
                      minutes for SEV1. Blameless post-mortems within 48 hours. Action items \
                      tracked in Linear with a 2-week SLA for SEV1 remediations.",
            memory_type: ExperienceType::Decision,
            tags: vec!["process", "incident-response", "team"],
            importance: 0.8,
        },
    ]
}

fn benchmark_queries() -> Vec<BenchmarkQuery> {
    vec![
        // =====================================================================
        // SINGLE-HOP: Direct fact lookup
        // =====================================================================
        BenchmarkQuery {
            query: "What database does the project use for the payment service?",
            query_type: "single_hop",
            expected_memory_indices: vec![0],
            expected_absent: vec![6, 13],
        },
        BenchmarkQuery {
            query: "What are the Kubernetes resource limits for the API gateway?",
            query_type: "single_hop",
            expected_memory_indices: vec![13],
            expected_absent: vec![0, 1],
        },
        BenchmarkQuery {
            query: "What CI/CD build caching tool is used and how much did it improve build time?",
            query_type: "single_hop",
            expected_memory_indices: vec![5],
            expected_absent: vec![13, 17],
        },
        BenchmarkQuery {
            query: "What error handling libraries do we use in Rust services?",
            query_type: "single_hop",
            expected_memory_indices: vec![16],
            expected_absent: vec![4, 11],
        },
        BenchmarkQuery {
            query: "What JWT vulnerability was found in the security review?",
            query_type: "single_hop",
            expected_memory_indices: vec![11],
            expected_absent: vec![3, 5],
        },
        // =====================================================================
        // TEMPORAL: Time-sensitive queries
        // =====================================================================
        BenchmarkQuery {
            query: "What bugs were fixed recently this week?",
            query_type: "temporal",
            expected_memory_indices: vec![1, 7, 15],
            expected_absent: vec![0, 6],
        },
        BenchmarkQuery {
            query: "What happened during the payment outage last month?",
            query_type: "temporal",
            expected_memory_indices: vec![19],
            expected_absent: vec![5, 13],
        },
        BenchmarkQuery {
            query: "What was the most recent database migration we completed?",
            query_type: "temporal",
            expected_memory_indices: vec![18],
            expected_absent: vec![5, 13],
        },
        BenchmarkQuery {
            query: "What timezone issue was fixed yesterday in reporting?",
            query_type: "temporal",
            expected_memory_indices: vec![7],
            expected_absent: vec![1, 15],
        },
        BenchmarkQuery {
            query: "What performance problems were discovered recently in staging?",
            query_type: "temporal",
            expected_memory_indices: vec![4, 10],
            expected_absent: vec![2, 8],
        },
        // =====================================================================
        // MULTI-HOP: Queries requiring connecting multiple memories
        // =====================================================================
        BenchmarkQuery {
            query: "Based on the architecture decisions and the performance issues, what database \
                    indexing strategies should we apply?",
            query_type: "multi_hop",
            expected_memory_indices: vec![0, 10],
            expected_absent: vec![2, 8],
        },
        BenchmarkQuery {
            query: "Considering the connection pool exhaustion and the retry strategy, how should \
                    we handle external service calls?",
            query_type: "multi_hop",
            expected_memory_indices: vec![3, 15],
            expected_absent: vec![2, 13],
        },
        BenchmarkQuery {
            query: "What error handling and logging patterns should a new Rust service follow?",
            query_type: "multi_hop",
            expected_memory_indices: vec![9, 16],
            expected_absent: vec![1, 13],
        },
        BenchmarkQuery {
            query: "How do our monitoring thresholds and incident response process work together?",
            query_type: "multi_hop",
            expected_memory_indices: vec![17, 19],
            expected_absent: vec![0, 5],
        },
        BenchmarkQuery {
            query: "What testing approaches and quality measures have we adopted for our codebase?",
            query_type: "multi_hop",
            expected_memory_indices: vec![5, 12],
            expected_absent: vec![0, 8],
        },
        // =====================================================================
        // OPEN-DOMAIN: Broad topical queries
        // =====================================================================
        BenchmarkQuery {
            query: "What have I learned about testing and code quality?",
            query_type: "open_domain",
            expected_memory_indices: vec![2, 12],
            expected_absent: vec![13, 17],
        },
        BenchmarkQuery {
            query: "What do I know about our team members and their preferences?",
            query_type: "open_domain",
            expected_memory_indices: vec![2, 8],
            expected_absent: vec![0, 10],
        },
        BenchmarkQuery {
            query: "What architecture decisions have been made for the system?",
            query_type: "open_domain",
            expected_memory_indices: vec![0, 6, 14],
            expected_absent: vec![1, 4],
        },
        BenchmarkQuery {
            query: "What production reliability and operational practices do we follow?",
            query_type: "open_domain",
            expected_memory_indices: vec![8, 17, 19],
            expected_absent: vec![3, 12],
        },
        BenchmarkQuery {
            query: "What security and authentication related work has been done?",
            query_type: "open_domain",
            expected_memory_indices: vec![11],
            expected_absent: vec![4, 10],
        },
    ]
}

// =============================================================================
// SCORING
// =============================================================================

/// Compute precision@k: of the top-k results, what fraction is relevant?
fn precision_at_k(retrieved: &[usize], relevant: &[usize], k: usize) -> f32 {
    let top_k: Vec<usize> = retrieved.iter().take(k).copied().collect();
    if top_k.is_empty() {
        return 0.0;
    }
    let hits = top_k.iter().filter(|r| relevant.contains(r)).count();
    hits as f32 / k as f32
}

/// Compute recall@k: of all relevant memories, what fraction appears in top-k?
fn recall_at_k(retrieved: &[usize], relevant: &[usize], k: usize) -> f32 {
    if relevant.is_empty() {
        return 1.0; // vacuously true
    }
    let top_k: Vec<usize> = retrieved.iter().take(k).copied().collect();
    let hits = relevant.iter().filter(|r| top_k.contains(r)).count();
    hits as f32 / relevant.len() as f32
}

/// Compute reciprocal rank: 1/rank of the first relevant result (0 if none found).
fn reciprocal_rank(retrieved: &[usize], relevant: &[usize]) -> f32 {
    for (rank, idx) in retrieved.iter().enumerate() {
        if relevant.contains(idx) {
            return 1.0 / (rank as f32 + 1.0);
        }
    }
    0.0
}

/// Map retrieved MemoryIds back to their original benchmark indices.
fn map_results_to_indices(
    results: &[std::sync::Arc<veld::memory::types::Memory>],
    stored_ids: &[MemoryId],
) -> Vec<usize> {
    results
        .iter()
        .filter_map(|mem| stored_ids.iter().position(|id| *id == mem.id))
        .collect()
}

/// Aggregate per-query evaluations into per-type summary.
fn aggregate_by_type(evaluations: &[QueryEvaluation]) -> Vec<EvaluationResult> {
    let types = ["single_hop", "temporal", "multi_hop", "open_domain"];
    types
        .iter()
        .map(|qt| {
            let matching: Vec<&QueryEvaluation> =
                evaluations.iter().filter(|e| e.query_type == *qt).collect();
            let n = matching.len() as f32;
            if n == 0.0 {
                return EvaluationResult {
                    query_type: qt.to_string(),
                    precision_at_3: 0.0,
                    precision_at_5: 0.0,
                    recall_at_5: 0.0,
                    mrr: 0.0,
                    latency_ms: 0,
                };
            }
            EvaluationResult {
                query_type: qt.to_string(),
                precision_at_3: matching.iter().map(|e| e.precision_at_3).sum::<f32>() / n,
                precision_at_5: matching.iter().map(|e| e.precision_at_5).sum::<f32>() / n,
                recall_at_5: matching.iter().map(|e| e.recall_at_5).sum::<f32>() / n,
                mrr: matching.iter().map(|e| e.reciprocal_rank).sum::<f32>() / n,
                latency_ms: (matching.iter().map(|e| e.latency_ms).sum::<u64>() as f32 / n) as u64,
            }
        })
        .collect()
}

// =============================================================================
// DISPLAY
// =============================================================================

fn print_summary_table(type_results: &[EvaluationResult], overall: &EvaluationResult) {
    println!();
    println!("Shodh-Memory Retrieval Benchmark");
    println!(
        "{}",
        "\u{2501}".repeat(73)
    );
    println!(
        "{:<16}| {:<8}| {:<8}| {:<8}| {:<8}| Latency",
        "Query Type", "P@3", "P@5", "R@5", "MRR"
    );
    println!(
        "{}",
        "\u{2500}".repeat(73)
    );
    for r in type_results {
        println!(
            "{:<16}| {:<8.2}| {:<8.2}| {:<8.2}| {:<8.2}| {}ms",
            r.query_type, r.precision_at_3, r.precision_at_5, r.recall_at_5, r.mrr, r.latency_ms
        );
    }
    println!(
        "{}",
        "\u{2501}".repeat(73)
    );
    println!(
        "{:<16}| {:<8.2}| {:<8.2}| {:<8.2}| {:<8.2}| {}ms",
        "OVERALL",
        overall.precision_at_3,
        overall.precision_at_5,
        overall.recall_at_5,
        overall.mrr,
        overall.latency_ms
    );
    println!();
}

fn print_per_query_detail(evaluations: &[QueryEvaluation]) {
    println!("Per-Query Detail");
    println!("{}", "\u{2500}".repeat(73));
    for eval in evaluations {
        let hit_marker = if eval.reciprocal_rank > 0.0 {
            "HIT"
        } else {
            "MISS"
        };
        println!(
            "  [{:<10}] [{}] RR={:.2}  P@3={:.2}  R@5={:.2}  {}ms",
            eval.query_type,
            hit_marker,
            eval.reciprocal_rank,
            eval.precision_at_3,
            eval.recall_at_5,
            eval.latency_ms,
        );
        println!(
            "    Q: \"{}\"",
            if eval.query_text.len() > 70 {
                format!("{}...", &eval.query_text[..67])
            } else {
                eval.query_text.clone()
            }
        );
        println!(
            "    expected={:?}  retrieved={:?}",
            eval.expected_indices, eval.retrieved_indices
        );
    }
    println!();
}

// =============================================================================
// BENCHMARK TEST
// =============================================================================

#[test]
fn benchmark_retrieval_quality() {
    // 1. Setup
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.1,
    };
    let system = MemorySystem::new(config, None).expect("Failed to create memory system");

    // 2. Store all benchmark memories
    let memories = benchmark_memories();
    let mut stored_ids: Vec<MemoryId> = Vec::with_capacity(memories.len());

    let store_start = Instant::now();
    for mem in &memories {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("importance_hint".to_string(), mem.importance.to_string());
        let experience = Experience {
            content: mem.content.to_string(),
            experience_type: mem.memory_type.clone(),
            entities: mem.tags.iter().map(|t| t.to_string()).collect(),
            metadata,
            ..Default::default()
        };
        let id = system
            .remember(experience, None)
            .expect("Failed to store benchmark memory");
        stored_ids.push(id);
    }
    let store_duration = store_start.elapsed();
    println!(
        "\nStored {} memories in {}ms",
        stored_ids.len(),
        store_duration.as_millis()
    );

    // 3. Run queries and score
    let queries = benchmark_queries();
    let mut evaluations: Vec<QueryEvaluation> = Vec::with_capacity(queries.len());

    for bq in &queries {
        let query = Query {
            query_text: Some(bq.query.to_string()),
            max_results: 5,
            ..Default::default()
        };

        let query_start = Instant::now();
        let results = system.recall(&query).expect("Recall failed");
        let latency = query_start.elapsed();

        let retrieved_indices = map_results_to_indices(&results, &stored_ids);

        let p3 = precision_at_k(&retrieved_indices, &bq.expected_memory_indices, 3);
        let p5 = precision_at_k(&retrieved_indices, &bq.expected_memory_indices, 5);
        let r5 = recall_at_k(&retrieved_indices, &bq.expected_memory_indices, 5);
        let rr = reciprocal_rank(&retrieved_indices, &bq.expected_memory_indices);

        evaluations.push(QueryEvaluation {
            query_text: bq.query.to_string(),
            query_type: bq.query_type.to_string(),
            precision_at_3: p3,
            precision_at_5: p5,
            recall_at_5: r5,
            reciprocal_rank: rr,
            latency_ms: latency.as_millis() as u64,
            retrieved_indices,
            expected_indices: bq.expected_memory_indices.clone(),
        });
    }

    // 4. Aggregate and display
    let type_results = aggregate_by_type(&evaluations);

    let n = evaluations.len() as f32;
    let overall = EvaluationResult {
        query_type: "OVERALL".to_string(),
        precision_at_3: evaluations.iter().map(|e| e.precision_at_3).sum::<f32>() / n,
        precision_at_5: evaluations.iter().map(|e| e.precision_at_5).sum::<f32>() / n,
        recall_at_5: evaluations.iter().map(|e| e.recall_at_5).sum::<f32>() / n,
        mrr: evaluations.iter().map(|e| e.reciprocal_rank).sum::<f32>() / n,
        latency_ms: (evaluations.iter().map(|e| e.latency_ms).sum::<u64>() as f32 / n) as u64,
    };

    print_per_query_detail(&evaluations);
    print_summary_table(&type_results, &overall);

    // 5. Check absence constraints (items that should NOT appear in results)
    let mut absence_violations = 0;
    for (i, bq) in queries.iter().enumerate() {
        for absent_idx in &bq.expected_absent {
            if evaluations[i].retrieved_indices.contains(absent_idx) {
                println!(
                    "  ABSENCE VIOLATION: query \"{}\" retrieved memory {} which should be absent",
                    bq.query, absent_idx
                );
                absence_violations += 1;
            }
        }
    }
    if absence_violations > 0 {
        println!(
            "  Total absence violations: {}/{}",
            absence_violations,
            queries.iter().map(|q| q.expected_absent.len()).sum::<usize>()
        );
    }

    // 6. Assertions: minimum quality thresholds
    //
    // Threshold rationale:
    // - MRR >= 0.70: the first relevant result should appear in the top-2 on average.
    //   This is the most important metric — does the system surface what you need?
    // - Recall@5 >= 0.70: at least 70% of relevant memories should appear in top-5.
    //   Validates that the system doesn't miss critical context.
    // - Single-hop MRR >= 0.80: direct fact lookups should almost always rank first.
    // - Latency < 500ms: any individual query must complete under half a second.
    //
    // Note: Precision@k is structurally bounded by (relevant_count / k). Single-hop
    // queries with 1 relevant memory can achieve at most P@3 = 0.33. We therefore
    // do NOT threshold on precision; it is reported for analysis only.

    let single_hop = type_results
        .iter()
        .find(|r| r.query_type == "single_hop")
        .expect("single_hop results missing");

    assert!(
        single_hop.mrr >= 0.80,
        "Single-hop MRR too low: {:.2} (threshold: 0.80)",
        single_hop.mrr
    );

    assert!(
        overall.mrr >= 0.70,
        "Overall MRR too low: {:.2} (threshold: 0.70)",
        overall.mrr
    );

    assert!(
        overall.recall_at_5 >= 0.70,
        "Overall Recall@5 too low: {:.2} (threshold: 0.70)",
        overall.recall_at_5
    );

    let max_latency = evaluations.iter().map(|e| e.latency_ms).max().unwrap_or(0);
    assert!(
        max_latency < 500,
        "Max query latency too high: {}ms (threshold: 500ms)",
        max_latency
    );

    println!("All quality thresholds passed.");
}
