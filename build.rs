//! Build script: stamps the crate with a build number, a build timestamp and the full version.
//!
//! Exposes `VELD_BUILD_NUMBER`, `VELD_BUILD_TIMESTAMP` (ISO 8601 UTC) and `VELD_VERSION_FULL`
//! (`{cargo_version}+{build_number}`) as `env!()` values, plus `VELD_FORTRESS_SEED`.
//!
//! The counter lives in `.build_number` at the project root (gitignored, per machine). It bumps
//! when this script reruns, and the script reruns only when it changes itself or when git HEAD
//! moves: a rerun changes the timestamp, and a changed `rustc-env` recompiles the lib and every
//! binary (fat LTO link included), so rerunning on every invocation cost a full rebuild even with
//! no source change (measured 2026-10-21: 16 min). Two dirty builds on the same commit therefore
//! share a build number; the commit, not the invocation, is the unit the stamp distinguishes.

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let build_file = Path::new(".build_number");

    let current: u64 = fs::read_to_string(build_file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let next = current + 1;
    fs::write(build_file, next.to_string()).expect("Failed to write .build_number");

    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();
    let ts = format_timestamp(secs);
    let full_version = format!("{}+{}", env!("CARGO_PKG_VERSION"), next);

    println!("cargo:rustc-env=VELD_BUILD_NUMBER={}", next);
    println!("cargo:rustc-env=VELD_BUILD_TIMESTAMP={}", ts);
    println!("cargo:rustc-env=VELD_VERSION_FULL={}", full_version);
    // Obfuscation seed for fortress builds, not a cryptographic secret: it only has to differ
    // between builds so `strings` sees different byte patterns.
    println!("cargo:rustc-env=VELD_FORTRESS_SEED={}:{}", next, secs);

    println!("cargo:rerun-if-changed=build.rs");
    for path in git_head_files() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// The files whose mtime moves when HEAD moves: the git dir's `HEAD` and, for a symbolic HEAD,
/// the loose ref it names. Works for a plain checkout and for a linked worktree (`.git` file).
fn git_head_files() -> Vec<PathBuf> {
    let dot_git = Path::new(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git.to_path_buf()
    } else {
        match fs::read_to_string(dot_git) {
            Ok(s) => match s.trim().strip_prefix("gitdir:") {
                Some(p) => PathBuf::from(p.trim()),
                None => return Vec::new(),
            },
            Err(_) => return Vec::new(),
        }
    };
    let head = git_dir.join("HEAD");
    let mut files = vec![head.clone()];
    if let Ok(contents) = fs::read_to_string(&head) {
        if let Some(reference) = contents.trim().strip_prefix("ref:") {
            let common_dir = fs::read_to_string(git_dir.join("commondir"))
                .map(|s| git_dir.join(s.trim()))
                .unwrap_or_else(|_| git_dir.clone());
            let ref_file = common_dir.join(reference.trim());
            if ref_file.exists() {
                files.push(ref_file);
            }
        }
    }
    files
}

/// Minimal ISO 8601 UTC formatter (avoids a chrono dependency in the build script).
fn format_timestamp(epoch_secs: u64) -> String {
    let secs_per_day: u64 = 86400;
    let mut days = epoch_secs / secs_per_day;
    let time_secs = epoch_secs % secs_per_day;

    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Gregorian calendar from days since 1970-01-01
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    days += 719468; // shift to 0000-03-01 epoch
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, minutes, seconds
    )
}
