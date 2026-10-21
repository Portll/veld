//! CLI surface tests for the `recall-eval` binary.
//!
//! These guard the command-line contract — flag names, required arguments and parse-time
//! refusals — without standing up a `MemorySystem`, which loads ONNX models and is far too
//! heavy for an integration test. `.github/workflows/recall.yml` drives the binary by these
//! flag names and branches on its exit code, so a renamed flag or a loosened parser breaks the
//! gate without failing a build.
//!
//! End-to-end coverage of the suite itself is the
//! `runner_executes_smoke_suite_and_produces_well_formed_report` unit test in
//! `src/recall_harness/runner.rs`.
//!
//! Cargo sets `CARGO_BIN_EXE_recall-eval` for every integration test in this crate, so no
//! `assert_cmd` dependency is needed.

use std::path::PathBuf;
use std::process::Command;

fn recall_eval_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_recall-eval"))
}

/// `--help` exits 0 and names every flag the gate workflow passes.
#[test]
fn help_lists_every_flag_the_gate_passes() {
    let out = Command::new(recall_eval_bin())
        .arg("--help")
        .output()
        .expect("spawning recall-eval --help");

    assert!(
        out.status.success(),
        "--help must exit 0, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--suite",
        "--layer",
        "--repeats",
        "--output",
        "--per-case-output",
        "--baseline",
        "--tolerance",
        "--storage",
    ] {
        assert!(
            stdout.contains(flag),
            "--help output is missing {flag}\nfull stdout:\n{stdout}"
        );
    }
}

/// Without `--output` the run is a parse error, not a run that writes no report: the gate reads
/// the report file, so an optional `--output` would make it gate on nothing.
#[test]
fn missing_output_flag_is_a_parse_error() {
    let out = Command::new(recall_eval_bin())
        .output()
        .expect("spawning recall-eval with no args");

    assert!(
        !out.status.success(),
        "running with no --output must fail; got success exit"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--output"),
        "parse error must mention --output, got:\n{stderr}"
    );
}

/// An unknown `--suite` fails at parse time, before any memory system is built, and the
/// message names the suites that exist.
#[test]
fn unknown_suite_is_a_parse_error() {
    let out = Command::new(recall_eval_bin())
        .args([
            "--suite",
            "definitely-not-a-suite",
            "--output",
            "ignored.json",
        ])
        .output()
        .expect("spawning recall-eval with bogus suite");

    assert!(
        !out.status.success(),
        "unknown --suite value must fail at parse time"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--suite") && stderr.contains("locomo-gate"),
        "stderr should name the bad --suite value and the valid ones, got:\n{stderr}"
    );
}

/// A non-numeric `--tolerance` fails at parse time: coerced to a default, a typo would set the
/// gate's allowance without anyone choosing it.
#[test]
fn non_numeric_tolerance_is_a_parse_error() {
    let out = Command::new(recall_eval_bin())
        .args(["--output", "ignored.json", "--tolerance", "not-a-number"])
        .output()
        .expect("spawning recall-eval with bogus tolerance");

    assert!(
        !out.status.success(),
        "non-numeric --tolerance must fail at parse time"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--tolerance"),
        "parse error must mention --tolerance, got:\n{stderr}"
    );
}
