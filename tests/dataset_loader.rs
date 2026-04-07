//! Dataset loader for the JSON benchmark format.
//!
//! Loads `nexus_benchmark_v2.json` and provides typed access to memories,
//! queries, persons, sessions, and scoring configuration. Includes validation
//! to catch ground truth errors at test time.

use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

// =============================================================================
// SCHEMA TYPES
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BenchmarkDataset {
    pub version: String,
    pub name: String,
    pub description: String,
    pub created: String,
    pub review_status: String,
    #[serde(default)]
    pub reviewers: Vec<String>,
    pub conventions: Conventions,
    pub persons: Vec<Person>,
    pub sessions: Vec<Session>,
    pub memories: Vec<DatasetMemory>,
    pub queries: Vec<DatasetQuery>,
    pub scoring: ScoringConfig,
    #[serde(default)]
    pub challenge_tag_vocabulary: Vec<String>,
    #[serde(default)]
    pub query_type_vocabulary: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Conventions {
    pub who_is_person: String,
    pub negation_semantics: String,
    pub temporal_resolution: String,
    pub contradiction_handling: String,
}

#[derive(Debug, Deserialize)]
pub struct Person {
    pub id: String,
    pub full_name: String,
    pub role: String,
}

#[derive(Debug, Deserialize)]
pub struct Session {
    pub id: u8,
    pub name: String,
    pub temporal_label: String,
    pub week_offset: i32,
}

#[derive(Debug, Deserialize)]
pub struct DatasetMemory {
    pub id: usize,
    pub content: String,
    pub experience_type: String,
    pub tags: Vec<String>,
    pub person_mentions: Vec<String>,
    pub importance: f32,
    pub session: u8,
    #[serde(default)]
    pub narrative_notes: String,
}

#[derive(Debug, Deserialize)]
pub struct DatasetQuery {
    pub id: usize,
    pub query: String,
    pub query_type: String,
    pub difficulty: String,
    #[serde(default)]
    pub challenge_tags: Vec<String>,
    pub expected_memory_ids: Vec<usize>,
    pub acceptable_memory_ids: Vec<usize>,
    pub absence_memory_ids: Vec<usize>,
    pub who_entity: Option<String>,
    pub expected_answer_sketch: String,
    pub rationale: String,
    pub review_status: String,
    #[serde(default)]
    pub reviewer_notes: String,
}

#[derive(Debug, Deserialize)]
pub struct ScoringConfig {
    pub composite_weights: CompositeWeights,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub reference_scores: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
pub struct CompositeWeights {
    pub mrr: f32,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub precision_at_5: f32,
    pub absence_compliance: f32,
}

// =============================================================================
// VALIDATION
// =============================================================================

#[derive(Debug)]
pub struct ValidationError {
    pub query_id: Option<usize>,
    pub memory_id: Option<usize>,
    pub severity: &'static str,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let location = match (self.query_id, self.memory_id) {
            (Some(q), _) => format!("query[{}]", q),
            (_, Some(m)) => format!("memory[{}]", m),
            _ => "dataset".to_string(),
        };
        write!(f, "[{}] {}: {}", self.severity, location, self.message)
    }
}

impl BenchmarkDataset {
    /// Load a dataset from a JSON file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let dataset: Self = serde_json::from_str(&content)?;
        Ok(dataset)
    }

    /// Validate dataset integrity. Returns errors found.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let max_memory_id = self.memories.iter().map(|m| m.id).max().unwrap_or(0);
        let memory_ids: HashSet<usize> = self.memories.iter().map(|m| m.id).collect();
        let person_ids: HashSet<&str> = self.persons.iter().map(|p| p.id.as_str()).collect();
        let session_ids: HashSet<u8> = self.sessions.iter().map(|s| s.id).collect();
        let valid_types: HashSet<&str> = self.query_type_vocabulary.iter().map(|s| s.as_str()).collect();
        let valid_challenges: HashSet<&str> = self.challenge_tag_vocabulary.iter().map(|s| s.as_str()).collect();

        // Validate memories
        for mem in &self.memories {
            if !session_ids.contains(&mem.session) {
                errors.push(ValidationError {
                    query_id: None,
                    memory_id: Some(mem.id),
                    severity: "ERROR",
                    message: format!("references non-existent session {}", mem.session),
                });
            }
            for person in &mem.person_mentions {
                if !person_ids.contains(person.as_str()) {
                    errors.push(ValidationError {
                        query_id: None,
                        memory_id: Some(mem.id),
                        severity: "ERROR",
                        message: format!("references unknown person '{}'", person),
                    });
                }
            }
        }

        // Validate queries
        for query in &self.queries {
            // Check memory index bounds
            for idx in &query.expected_memory_ids {
                if *idx > max_memory_id || !memory_ids.contains(idx) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "ERROR",
                        message: format!("expected_memory_ids contains invalid index {}", idx),
                    });
                }
            }
            for idx in &query.acceptable_memory_ids {
                if *idx > max_memory_id || !memory_ids.contains(idx) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "ERROR",
                        message: format!("acceptable_memory_ids contains invalid index {}", idx),
                    });
                }
            }
            for idx in &query.absence_memory_ids {
                if *idx > max_memory_id || !memory_ids.contains(idx) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "ERROR",
                        message: format!("absence_memory_ids contains invalid index {}", idx),
                    });
                }
            }

            // Expected must be subset of acceptable
            let expected: HashSet<usize> = query.expected_memory_ids.iter().copied().collect();
            let acceptable: HashSet<usize> = query.acceptable_memory_ids.iter().copied().collect();
            let absence: HashSet<usize> = query.absence_memory_ids.iter().copied().collect();

            for idx in &expected {
                if !acceptable.contains(idx) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "ERROR",
                        message: format!(
                            "expected_memory_ids[{}] not in acceptable_memory_ids",
                            idx
                        ),
                    });
                }
            }

            // No overlap between expected/acceptable and absence
            for idx in &absence {
                if expected.contains(idx) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "ERROR",
                        message: format!(
                            "memory {} appears in both expected and absence sets",
                            idx
                        ),
                    });
                }
                if acceptable.contains(idx) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "ERROR",
                        message: format!(
                            "memory {} appears in both acceptable and absence sets",
                            idx
                        ),
                    });
                }
            }

            // WHO queries must have person_mentions in expected memories
            if query.query_type == "who_person" {
                if let Some(ref who) = query.who_entity {
                    // Check that at least one expected memory mentions this person
                    let mentions_person = query.expected_memory_ids.iter().any(|idx| {
                        self.memories
                            .iter()
                            .find(|m| m.id == *idx)
                            .map(|m| m.person_mentions.iter().any(|p| p == who))
                            .unwrap_or(false)
                    });
                    if !mentions_person && !query.expected_memory_ids.is_empty() {
                        errors.push(ValidationError {
                            query_id: Some(query.id),
                            memory_id: None,
                            severity: "WARN",
                            message: format!(
                                "who_person query with who_entity='{}' but no expected memory mentions this person",
                                who
                            ),
                        });
                    }
                }
            }

            // Single-hop queries should have exactly 1 expected memory
            if query.query_type == "single_hop" && query.expected_memory_ids.len() != 1 {
                errors.push(ValidationError {
                    query_id: Some(query.id),
                    memory_id: None,
                    severity: "WARN",
                    message: format!(
                        "single_hop query has {} expected memories (expected 1)",
                        query.expected_memory_ids.len()
                    ),
                });
            }

            // Validate query type against vocabulary
            if !valid_types.is_empty() && !valid_types.contains(query.query_type.as_str()) {
                errors.push(ValidationError {
                    query_id: Some(query.id),
                    memory_id: None,
                    severity: "WARN",
                    message: format!("unknown query_type '{}'", query.query_type),
                });
            }

            // Validate challenge tags against vocabulary
            for tag in &query.challenge_tags {
                if !valid_challenges.is_empty() && !valid_challenges.contains(tag.as_str()) {
                    errors.push(ValidationError {
                        query_id: Some(query.id),
                        memory_id: None,
                        severity: "WARN",
                        message: format!("unknown challenge_tag '{}'", tag),
                    });
                }
            }
        }

        // Validate scoring weights sum to ~1.0
        let weights = &self.scoring.composite_weights;
        let total = weights.mrr
            + weights.recall_at_5
            + weights.recall_at_10
            + weights.precision_at_5
            + weights.absence_compliance;
        if (total - 1.0).abs() > 0.01 {
            errors.push(ValidationError {
                query_id: None,
                memory_id: None,
                severity: "ERROR",
                message: format!("composite weights sum to {:.3}, expected 1.0", total),
            });
        }

        errors
    }

    /// Get query type distribution as a summary.
    pub fn query_type_summary(&self) -> Vec<(String, usize)> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for q in &self.queries {
            *counts.entry(q.query_type.clone()).or_default() += 1;
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Get challenge tag distribution.
    pub fn challenge_tag_summary(&self) -> Vec<(String, usize)> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for q in &self.queries {
            for tag in &q.challenge_tags {
                *counts.entry(tag.clone()).or_default() += 1;
            }
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_and_validate_dataset() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("datasets")
            .join("nexus_benchmark_v2.json");

        let dataset = BenchmarkDataset::load(&path)
            .unwrap_or_else(|e| panic!("Failed to load dataset: {}", e));

        assert_eq!(dataset.version, "2.0.0");
        assert!(!dataset.memories.is_empty(), "Dataset has no memories");
        assert!(!dataset.queries.is_empty(), "Dataset has no queries");

        let errors = dataset.validate();
        let hard_errors: Vec<_> = errors.iter().filter(|e| e.severity == "ERROR").collect();

        if !hard_errors.is_empty() {
            for err in &hard_errors {
                eprintln!("  {}", err);
            }
            panic!("{} validation errors found in dataset", hard_errors.len());
        }

        let warnings: Vec<_> = errors.iter().filter(|e| e.severity == "WARN").collect();
        if !warnings.is_empty() {
            eprintln!("\nDataset warnings ({}):", warnings.len());
            for w in &warnings {
                eprintln!("  {}", w);
            }
        }

        // Print summary
        println!("\n=== Dataset Summary ===");
        println!("Memories: {}", dataset.memories.len());
        println!("Queries:  {}", dataset.queries.len());
        println!("Persons:  {}", dataset.persons.len());
        println!("Sessions: {}", dataset.sessions.len());

        println!("\nQuery types:");
        for (qt, count) in dataset.query_type_summary() {
            println!("  {:15} {}", qt, count);
        }

        println!("\nChallenge tags:");
        for (tag, count) in dataset.challenge_tag_summary() {
            println!("  {:25} {}", tag, count);
        }
    }
}
