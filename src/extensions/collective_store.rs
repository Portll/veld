//! Collective knowledge store for cross-user weight aggregation.
//!
//! Manages population-level learned parameters derived from individual user
//! retrieval feedback. New users bootstrap from the collective prior instead
//! of cold-starting with static defaults.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationWeights {
    pub bm25: f32,
    pub vector: f32,
    pub graph: f32,
    pub contributing_users: usize,
    pub total_feedback_events: u64,
    pub last_aggregated: DateTime<Utc>,
}

impl Default for PopulationWeights {
    fn default() -> Self {
        Self {
            bm25: 0.35,
            vector: 0.45,
            graph: 0.20,
            contributing_users: 0,
            total_feedback_events: 0,
            last_aggregated: Utc::now(),
        }
    }
}

impl PopulationWeights {
    pub fn confidence(&self) -> f32 {
        (self.contributing_users as f32 / 10.0).min(1.0)
    }

    pub fn blend_with_defaults(
        &self,
        default_bm25: f32,
        default_vector: f32,
        default_graph: f32,
    ) -> (f32, f32, f32) {
        let confidence = self.confidence();
        let bm25 = (1.0 - confidence) * default_bm25 + confidence * self.bm25;
        let vector = (1.0 - confidence) * default_vector + confidence * self.vector;
        let graph = (1.0 - confidence) * default_graph + confidence * self.graph;

        let sum = bm25 + vector + graph;
        if sum > 0.0 {
            (bm25 / sum, vector / sum, graph / sum)
        } else {
            (default_bm25, default_vector, default_graph)
        }
    }
}

pub struct CollectiveStore {
    dir: PathBuf,
    weights: RwLock<PopulationWeights>,
}

impl CollectiveStore {
    pub fn open(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let weights_path = dir.join("population_weights.json");
        let weights = if weights_path.exists() {
            let data = std::fs::read_to_string(&weights_path)?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            PopulationWeights::default()
        };

        Ok(Self {
            dir,
            weights: RwLock::new(weights),
        })
    }

    pub fn population_weights(&self) -> PopulationWeights {
        self.weights.read().clone()
    }

    pub fn update_weights(&self, new_weights: PopulationWeights) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&new_weights)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

        let tmp_path = self.dir.join("population_weights.json.tmp");
        let final_path = self.dir.join("population_weights.json");
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &final_path)?;

        *self.weights.write() = new_weights;
        Ok(())
    }

    pub fn bootstrap_user_weights(
        &self,
        default_bm25: f32,
        default_vector: f32,
        default_graph: f32,
    ) -> (f32, f32, f32) {
        self.weights
            .read()
            .blend_with_defaults(default_bm25, default_vector, default_graph)
    }
}