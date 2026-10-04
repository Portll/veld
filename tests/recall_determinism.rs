//! Cross-repeat determinism of the recall harness on the LoCoMo gate suite.
//!
//! The recall gate (`.github/workflows/recall.yml`) ingests the gate corpus into a fresh
//! store per repeat and fails as "infrastructure" when any case's top-10 differs between
//! repeats. This test runs the same harness entry point the `recall-eval` binary uses, with
//! the same process settings, so a divergence shows up under `cargo test` rather than only
//! in CI.
//!
//! Run:
//!   cargo test --release --test recall_determinism -- --ignored --nocapture
//!
//! `RECALL_DET_LAYERS=all` runs every cumulative layer mode, so the first mode that diverges
//! names the pipeline stage. `RECALL_DET_REPEATS` sets the repeat count (default 2), and
//! `RECALL_DET_STORAGE` keeps the repeat stores at that path for inspection.

use std::collections::BTreeMap;
use std::path::PathBuf;

use veld::memory::types::LayerMode;
use veld::recall_harness::fixtures;
use veld::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs, HARNESS_CLOCK_ANCHOR};

/// The settings `recall-eval` pins in `main`, plus the CI job's environment, plus what keeps a
/// developer machine from reaching services the CI runner does not have.
fn pin_gate_environment() {
    let pins: [(&str, &str); 7] = [
        ("VELD_ONNX_THREADS", "1"),
        ("RAYON_NUM_THREADS", "1"),
        ("VELD_RECALL_READONLY", "1"),
        ("VELD_EVAL_NOW", HARNESS_CLOCK_ANCHOR),
        ("SHODH_ALLOW_FALLBACK_NER", "1"),
        ("VELD_EMBEDDER", "minilm"),
        // Nothing listens on the discard port, so no HTTP embedder is chosen as a secondary.
        ("VELD_EMBEDDING_API_URL", "http://127.0.0.1:9"),
    ];
    for (key, value) in pins {
        if std::env::var_os(key).is_none() {
            // SAFETY: this is the only test in the binary, and it pins before any other thread
            // or the harness reads the environment.
            unsafe { std::env::set_var(key, value) };
        }
    }
}

#[test]
#[ignore = "ingests the 629-turn LoCoMo gate corpus once per repeat; run with --ignored --release"]
fn locomo_gate_repeats_return_identical_rank_lists() {
    pin_gate_environment();

    let layer_modes = match std::env::var("RECALL_DET_LAYERS").as_deref() {
        Ok("all") => LayerMode::ALL.to_vec(),
        _ => vec![LayerMode::Full],
    };
    let repeats = std::env::var("RECALL_DET_REPEATS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(2);
    let keep = std::env::var_os("RECALL_DET_STORAGE").map(PathBuf::from);
    let storage = keep.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "veld-recall-determinism-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ))
    });

    let inputs = RunInputs {
        storage_path: storage.clone(),
        corpus_path: Some(fixtures::manifest_path(fixtures::LOCOMO_GATE_CORPUS_PATH)),
        cases_path: Some(fixtures::manifest_path(fixtures::LOCOMO_GATE_CASES_PATH)),
        suite: "locomo-gate".to_string(),
        git_sha: "test".to_string(),
        repeats,
        layer_modes: layer_modes.clone(),
        age_days: 0.0,
    };

    let started = std::time::Instant::now();
    let result = run_smoke_suite_with_ranks(&inputs).expect("the harness run completes");
    let report = result.report;
    eprintln!(
        "recall_determinism: {} cases x {} repeats x {} modes in {:.1}s",
        report.case_count,
        repeats,
        layer_modes.len(),
        started.elapsed().as_secs_f64()
    );

    let divergences: Vec<&str> = report
        .failures
        .iter()
        .filter(|f| f.kind == "infrastructure")
        .map(|f| f.detail.as_str())
        .collect();

    let mut by_mode: BTreeMap<&str, usize> = layer_modes
        .iter()
        .map(|m| (m.report_key(), 0usize))
        .collect();
    for detail in &divergences {
        for mode in &layer_modes {
            if detail.contains(&format!("[mode={}]", mode.report_key())) {
                *by_mode.entry(mode.report_key()).or_default() += 1;
            }
        }
    }
    for mode in &layer_modes {
        let layer = &report.layers[mode.report_key()];
        eprintln!(
            "recall_determinism: mode={:<12} divergent_case_repeats={:<4} ndcg@10={:.4} recall@10={:.4}",
            mode.report_key(),
            by_mode[mode.report_key()],
            layer.ndcg_at_10,
            layer.recall_at_10
        );
    }
    for detail in divergences.iter().take(5) {
        eprintln!("recall_determinism: {detail}");
    }

    if keep.is_none() && divergences.is_empty() {
        let _ = std::fs::remove_dir_all(&storage);
    }
    assert!(
        divergences.is_empty(),
        "{} case rank lists diverged across {repeats} repeats (per mode: {by_mode:?}); storage kept at {}",
        divergences.len(),
        storage.display()
    );
}
