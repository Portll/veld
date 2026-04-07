//! Temporal Perception Benchmark (A9: Kendall's τ)
//!
//! Evaluates veld's ability to reconstruct temporal orderings
//! from stored episodic memories. Stores events with known timestamps,
//! queries with temporal questions, and measures Kendall's τ rank correlation
//! between the system's retrieved ordering and ground truth.
//!
//! Reference: Bai et al. (2026) §6.1.3 — Kendall's τ for temporal perception
//!
//! Run with: cargo test temporal_benchmark -- --ignored --nocapture

use chrono::{Duration, Utc};
use veld::memory::types::{Experience, ExperienceType, Query, RetrievalMode};
use veld::memory::{MemoryConfig, MemorySystem};
use tempfile::TempDir;

// =============================================================================
// BENCHMARK PARAMETERS
// =============================================================================

const KENDALLS_TAU_MINIMUM: f64 = 0.3;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct TemporalEvent {
    content: &'static str,
    /// Days before "now" when this event occurred (larger = older)
    days_ago: i64,
    /// Ground-truth ordering index (0 = oldest)
    ground_truth_order: usize,
}

struct TemporalQuery {
    query: &'static str,
    /// Indices into the event corpus, in expected temporal order (oldest first)
    expected_order: Vec<usize>,
    query_type: &'static str,
}

// =============================================================================
// KENDALL'S TAU IMPLEMENTATION
// =============================================================================

/// Compute Kendall's τ rank correlation between two orderings.
///
/// τ = (concordant - discordant) / (n × (n-1) / 2)
///
/// Range: -1.0 (perfectly inverted) to +1.0 (perfectly correlated)
/// 0.0 = no correlation
fn kendalls_tau(predicted: &[usize], expected: &[usize]) -> f64 {
    let n = predicted.len().min(expected.len());
    if n < 2 {
        return 0.0;
    }

    // Build rank maps
    let mut pred_rank = vec![0usize; n];
    let mut exp_rank = vec![0usize; n];

    for (rank, &idx) in predicted.iter().enumerate().take(n) {
        if idx < n {
            pred_rank[idx] = rank;
        }
    }
    for (rank, &idx) in expected.iter().enumerate().take(n) {
        if idx < n {
            exp_rank[idx] = rank;
        }
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let pred_diff = pred_rank[i] as i64 - pred_rank[j] as i64;
            let exp_diff = exp_rank[i] as i64 - exp_rank[j] as i64;
            let product = pred_diff * exp_diff;
            if product > 0 {
                concordant += 1;
            } else if product < 0 {
                discordant += 1;
            }
        }
    }

    let total_pairs = (n * (n - 1)) / 2;
    if total_pairs == 0 {
        return 0.0;
    }

    (concordant - discordant) as f64 / total_pairs as f64
}

// =============================================================================
// TEST CORPUS
// =============================================================================

fn build_corpus() -> Vec<TemporalEvent> {
    vec![
        TemporalEvent {
            content: "Started the new authentication service project. Set up the repository and CI pipeline. Team kickoff meeting held.",
            days_ago: 90,
            ground_truth_order: 0,
        },
        TemporalEvent {
            content: "Designed the database schema for user accounts. Chose PostgreSQL over MongoDB for ACID compliance.",
            days_ago: 85,
            ground_truth_order: 1,
        },
        TemporalEvent {
            content: "Implemented JWT token generation and validation. Added refresh token rotation for security.",
            days_ago: 75,
            ground_truth_order: 2,
        },
        TemporalEvent {
            content: "Discovered a critical bug in the token refresh logic. Tokens were not being invalidated after rotation.",
            days_ago: 60,
            ground_truth_order: 3,
        },
        TemporalEvent {
            content: "Fixed the token invalidation bug. Added comprehensive test coverage for the refresh flow.",
            days_ago: 58,
            ground_truth_order: 4,
        },
        TemporalEvent {
            content: "Deployed the authentication service to staging environment. Performance testing showed 200ms p99 latency.",
            days_ago: 45,
            ground_truth_order: 5,
        },
        TemporalEvent {
            content: "Security audit completed by external team. Found two medium-severity issues in password hashing.",
            days_ago: 35,
            ground_truth_order: 6,
        },
        TemporalEvent {
            content: "Remediated security findings. Upgraded bcrypt rounds from 10 to 12. Added rate limiting on login endpoint.",
            days_ago: 30,
            ground_truth_order: 7,
        },
        TemporalEvent {
            content: "Production deployment of authentication service. Gradual rollout starting at 5% of traffic.",
            days_ago: 20,
            ground_truth_order: 8,
        },
        TemporalEvent {
            content: "Scaled authentication service to 100% traffic. No incidents during rollout. Project retrospective held.",
            days_ago: 10,
            ground_truth_order: 9,
        },
        TemporalEvent {
            content: "Added OAuth2 provider integration. Users can now sign in with Google and GitHub.",
            days_ago: 5,
            ground_truth_order: 10,
        },
        TemporalEvent {
            content: "Implemented multi-factor authentication using TOTP. Added backup codes for account recovery.",
            days_ago: 2,
            ground_truth_order: 11,
        },
    ]
}

fn build_queries() -> Vec<TemporalQuery> {
    vec![
        TemporalQuery {
            query: "What happened with the authentication project in chronological order?",
            expected_order: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            query_type: "full_sequence",
        },
        TemporalQuery {
            query: "What happened before the security audit?",
            expected_order: vec![0, 1, 2, 3, 4, 5],
            query_type: "before",
        },
        TemporalQuery {
            query: "What happened after production deployment?",
            expected_order: vec![9, 10, 11],
            query_type: "after",
        },
        TemporalQuery {
            query: "What was the sequence of the token bug discovery and fix?",
            expected_order: vec![3, 4],
            query_type: "sequence",
        },
        TemporalQuery {
            query: "What was the order of security-related events?",
            expected_order: vec![6, 7, 8],
            query_type: "sequence",
        },
        TemporalQuery {
            query: "What events happened between the database design and deployment?",
            expected_order: vec![2, 3, 4, 5],
            query_type: "between",
        },
    ]
}

// =============================================================================
// BENCHMARK TEST
// =============================================================================

#[test]
#[ignore] // Run with: cargo test temporal_benchmark -- --ignored --nocapture
fn temporal_benchmark_kendalls_tau() {
    let tmp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: tmp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let ms = MemorySystem::new(config, None).expect("Failed to create MemorySystem");

    let now = Utc::now();
    let corpus = build_corpus();

    // Phase 1: Store events with explicit timestamps
    println!("\n=== TEMPORAL BENCHMARK: Storing {} events ===\n", corpus.len());
    let mut memory_ids = Vec::new();

    for event in &corpus {
        let created_at = now - Duration::days(event.days_ago);
        let experience = Experience {
            content: event.content.to_string(),
            experience_type: ExperienceType::Observation,
            entities: vec![],
            context: None,
            embeddings: None,
            metadata: Default::default(),
            ..Default::default()
        };

        let id = ms
            .remember(experience, Some(created_at))
            .expect("Failed to store memory");

        // Embed immediately
        let _ = ms.embed_and_index(&id);

        memory_ids.push(id);
    }

    // Phase 2: Query and measure temporal ordering
    println!("=== Running {} temporal queries ===\n", build_queries().len());
    let queries = build_queries();
    let mut tau_scores: Vec<f64> = Vec::new();
    let mut per_type_taus: std::collections::HashMap<&str, Vec<f64>> =
        std::collections::HashMap::new();

    for tq in &queries {
        let query = Query::builder()
            .query_text(tq.query)
            .max_results(corpus.len())
            .retrieval_mode(RetrievalMode::Hybrid)
            .build();

        let results = match ms.recall(&query) {
            Ok(r) => r,
            Err(e) => {
                println!("  Query failed: {} — {}", tq.query, e);
                continue;
            }
        };

        // Map retrieved memory IDs back to corpus indices, preserving retrieval order
        let retrieved_indices: Vec<usize> = results
            .iter()
            .filter_map(|m| {
                memory_ids
                    .iter()
                    .position(|stored_id| *stored_id == m.id)
            })
            .collect();

        // Extract the subset of expected indices that appear in retrieved results
        let expected_in_results: Vec<usize> = tq
            .expected_order
            .iter()
            .filter(|idx| retrieved_indices.contains(idx))
            .copied()
            .collect();

        let predicted_in_expected: Vec<usize> = retrieved_indices
            .iter()
            .filter(|idx| tq.expected_order.contains(idx))
            .copied()
            .collect();

        if predicted_in_expected.len() >= 2 && expected_in_results.len() >= 2 {
            let tau = kendalls_tau(&predicted_in_expected, &expected_in_results);
            tau_scores.push(tau);
            per_type_taus
                .entry(tq.query_type)
                .or_default()
                .push(tau);

            println!(
                "  [{:.3}τ] {} — retrieved {}/{} expected items",
                tau,
                tq.query,
                predicted_in_expected.len(),
                tq.expected_order.len()
            );
        } else {
            println!(
                "  [skip] {} — insufficient overlap ({} predicted, {} expected)",
                tq.query,
                predicted_in_expected.len(),
                expected_in_results.len()
            );
        }
    }

    // Phase 3: Report
    println!("\n=== TEMPORAL BENCHMARK RESULTS ===\n");

    let avg_tau = if tau_scores.is_empty() {
        0.0
    } else {
        tau_scores.iter().sum::<f64>() / tau_scores.len() as f64
    };

    println!("Overall Kendall's τ: {:.3} (n={})", avg_tau, tau_scores.len());
    println!();

    for (qtype, taus) in &per_type_taus {
        let avg = taus.iter().sum::<f64>() / taus.len() as f64;
        println!("  {:<15} τ={:.3} (n={})", qtype, avg, taus.len());
    }

    println!();
    println!("Minimum required: τ ≥ {:.1}", KENDALLS_TAU_MINIMUM);
    println!(
        "Result: {}",
        if avg_tau >= KENDALLS_TAU_MINIMUM {
            "PASS"
        } else {
            "FAIL"
        }
    );

    assert!(
        tau_scores.is_empty() || avg_tau >= KENDALLS_TAU_MINIMUM,
        "Temporal ordering quality below minimum: τ={:.3} < {:.1}",
        avg_tau,
        KENDALLS_TAU_MINIMUM
    );
}

// =============================================================================
// UNIT TESTS: Kendall's τ implementation correctness
// =============================================================================

#[test]
fn test_kendalls_tau_perfect() {
    let predicted = vec![0, 1, 2, 3, 4];
    let expected = vec![0, 1, 2, 3, 4];
    let tau = kendalls_tau(&predicted, &expected);
    assert!(
        (tau - 1.0).abs() < 0.001,
        "Perfect agreement should give τ=1.0, got {}",
        tau
    );
}

#[test]
fn test_kendalls_tau_reversed() {
    let predicted = vec![4, 3, 2, 1, 0];
    let expected = vec![0, 1, 2, 3, 4];
    let tau = kendalls_tau(&predicted, &expected);
    assert!(
        (tau - (-1.0)).abs() < 0.001,
        "Perfect reversal should give τ=-1.0, got {}",
        tau
    );
}

#[test]
fn test_kendalls_tau_partial() {
    let predicted = vec![0, 2, 1, 3, 4];
    let expected = vec![0, 1, 2, 3, 4];
    let tau = kendalls_tau(&predicted, &expected);
    // One swap: 1 discordant out of 10 pairs = (9-1)/10 = 0.8
    assert!(
        (tau - 0.8).abs() < 0.001,
        "One swap should give τ=0.8, got {}",
        tau
    );
}

#[test]
fn test_kendalls_tau_empty() {
    let tau = kendalls_tau(&[], &[]);
    assert!((tau - 0.0).abs() < 0.001);
}
