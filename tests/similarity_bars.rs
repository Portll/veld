//! Every similarity bar counts its comparisons and its fires.
//!
//! The counters are process-wide, so these tests run in their own binary and take one lock:
//! nothing else in the process moves a bar, and each delta is exact.

use std::sync::Mutex;

use chrono::{Duration, Utc};
use prometheus::Encoder;
use veld::memory::replay::InterferenceDetector;
use veld::similarity::bars::{SimilarityBar, WritePath};

static SERIAL: Mutex<()> = Mutex::new(());

fn delta(bar: SimilarityBar, before: (u64, u64)) -> (u64, u64) {
    let (evaluated, fired) = bar.totals();
    (evaluated - before.0, fired - before.1)
}

#[test]
fn a_comparison_above_and_one_below_each_bar_move_its_counters() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for bar in SimilarityBar::ALL {
        let before = bar.totals();
        assert!(bar.record(true), "{bar:?}: record must return what fired");
        assert!(!bar.record(false), "{bar:?}: record must return what fired");
        assert_eq!(delta(bar, before), (2, 1), "{bar:?}");

        let before = bar.totals();
        bar.record_many(5, 2);
        bar.record_many(0, 0);
        assert_eq!(delta(bar, before), (5, 2), "{bar:?}: record_many");
    }
}

#[test]
fn interference_counts_under_the_write_path_that_ran_it() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let now = Utc::now();
    let neighbour = |id: &str, similarity: f32| {
        (
            id.to_string(),
            similarity,
            0.5,
            now - Duration::hours(1),
            String::new(),
        )
    };
    let watched = [
        SimilarityBar::InterferenceSync,
        SimilarityBar::InterferenceDeferred,
        SimilarityBar::InterferenceSevereSync,
        SimilarityBar::InterferenceSevereDeferred,
    ];
    let before: Vec<(u64, u64)> = watched.iter().map(|bar| bar.totals()).collect();
    let mut detector = InterferenceDetector::new();

    // One neighbour below the interference bar, one between it and the severe bar.
    let result = detector.check_interference(
        "new",
        0.7,
        now,
        &[neighbour("below", 0.50), neighbour("between", 0.90)],
        WritePath::Deferred,
    );
    assert!(!result.is_duplicate);
    // One neighbour above the severe bar: a duplicate, on the synchronous path.
    let result = detector.check_interference(
        "new",
        0.7,
        now,
        &[neighbour("duplicate", 0.98)],
        WritePath::Synchronous,
    );
    assert!(result.is_duplicate);

    let moved: Vec<(u64, u64)> = watched
        .iter()
        .zip(&before)
        .map(|(bar, before)| delta(*bar, *before))
        .collect();
    assert_eq!(
        moved,
        vec![(1, 1), (2, 1), (1, 1), (1, 0)],
        "evaluated/fired for {watched:?}"
    );
}

#[test]
fn every_bar_is_exposed_on_metrics_before_its_first_comparison() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    veld::metrics::register_metrics().expect("metrics register");
    let mut text = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&veld::metrics::METRICS_REGISTRY.gather(), &mut text)
        .expect("encode");
    let text = String::from_utf8(text).expect("utf8");
    for bar in SimilarityBar::ALL {
        for family in [
            "veld_similarity_bar_evaluated_total",
            "veld_similarity_bar_fired_total",
        ] {
            let series = format!("{family}{{bar=\"{}\"}}", bar.label());
            assert!(text.contains(&series), "{series} missing from /metrics");
        }
    }
}
