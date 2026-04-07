//! Maintenance routines for collective knowledge and cache persistence.

use std::path::Path;

use crate::extensions::collective_store::PopulationWeights;

pub fn aggregate_population_weights(user_weights: &[(f32, f32, f32)]) -> Option<PopulationWeights> {
    if user_weights.is_empty() {
        return None;
    }

    let n = user_weights.len() as f32;
    let bm25 = user_weights.iter().map(|weights| weights.0).sum::<f32>() / n;
    let vector = user_weights.iter().map(|weights| weights.1).sum::<f32>() / n;
    let graph = user_weights.iter().map(|weights| weights.2).sum::<f32>() / n;
    let sum = bm25 + vector + graph;
    let (bm25, vector, graph) = if sum > 0.0 {
        (bm25 / sum, vector / sum, graph / sum)
    } else {
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    };

    Some(PopulationWeights {
        bm25,
        vector,
        graph,
        contributing_users: user_weights.len(),
        total_feedback_events: 0,
        last_aggregated: chrono::Utc::now(),
    })
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LearnedWeightsSnapshot {
    bm25: f32,
    vector: f32,
    graph: f32,
    update_count: u64,
}

pub fn persist_learned_weights(
    path: &Path,
    bm25: f32,
    vector: f32,
    graph: f32,
    update_count: u64,
) -> std::io::Result<()> {
    if update_count == 0 {
        return Ok(());
    }

    let snapshot = LearnedWeightsSnapshot {
        bm25,
        vector,
        graph,
        update_count,
    };
    let json = serde_json::to_string_pretty(&snapshot)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

    std::fs::create_dir_all(path)?;
    let tmp = path.join("learned_weights.json.tmp");
    let final_path = path.join("learned_weights.json");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, &final_path)?;
    Ok(())
}

pub fn load_learned_weights(path: &Path) -> Option<(f32, f32, f32, u64)> {
    let weights_path = path.join("learned_weights.json");
    let data = std::fs::read_to_string(&weights_path).ok()?;
    let snapshot: LearnedWeightsSnapshot = serde_json::from_str(&data).ok()?;
    Some((snapshot.bm25, snapshot.vector, snapshot.graph, snapshot.update_count))
}

pub fn run_maintenance_cycle(
    collective_dir: &Path,
    user_weights: &[(f32, f32, f32)],
    total_feedback: u64,
) -> std::io::Result<()> {
    if user_weights.is_empty() {
        return Ok(());
    }

    if let Some(mut population_weights) = aggregate_population_weights(user_weights) {
        population_weights.total_feedback_events = total_feedback;
        let store = crate::extensions::collective_store::CollectiveStore::open(collective_dir)?;
        store.update_weights(population_weights)?;
    }

    Ok(())
}