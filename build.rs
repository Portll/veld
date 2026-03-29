//! Build script: auto-increment build number on every compilation.
//!
//! Maintains a monotonic counter in `.build_number` at the project root.
//! Each `cargo build` / `cargo check` bumps it by 1 and exposes:
//!
//! - `SHODH_BUILD_NUMBER` — the counter (e.g., "142")
//! - `SHODH_BUILD_TIMESTAMP` — ISO 8601 UTC timestamp of the build
//! - `SHODH_VERSION_FULL` — "{cargo_version}+{build_number}" (e.g., "0.5.1-1+142")
//!
//! Access in Rust via:
//! ```ignore
//! env!("SHODH_BUILD_NUMBER")
//! env!("SHODH_BUILD_TIMESTAMP")
//! env!("SHODH_VERSION_FULL")
//! ```
//!
//! The `.build_number` file should be gitignored — it's local to each machine.

use std::fs;
use std::path::Path;

fn main() {
    let build_file = Path::new(".build_number");

    // Read current build number, or start at 0
    let current: u64 = fs::read_to_string(build_file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let next = current + 1;

    // Write back
    fs::write(build_file, next.to_string()).expect("Failed to write .build_number");

    // Timestamp
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards");
    let secs = now.as_secs();
    // Format as ISO 8601 UTC (no chrono dependency in build script)
    let ts = format_timestamp(secs);

    // Compose full version: semver+build_number
    let pkg_version = env!("CARGO_PKG_VERSION");
    let full_version = format!("{}+{}", pkg_version, next);

    // Expose to the crate
    println!("cargo:rustc-env=SHODH_BUILD_NUMBER={}", next);
    println!("cargo:rustc-env=SHODH_BUILD_TIMESTAMP={}", ts);
    println!("cargo:rustc-env=SHODH_VERSION_FULL={}", full_version);

    // Fortress: generate per-build random encryption seed
    // The seed is derived from build number + timestamp, ensuring every build
    // produces different encrypted byte patterns (prevents known-plaintext attacks).
    // This is NOT cryptographic security — it's obfuscation against `strings`.
    let fortress_seed = format!("{}:{}", next, secs);
    println!("cargo:rustc-env=SHODH_FORTRESS_SEED={}", fortress_seed);

    // Only re-run when the build file changes (not on every source change)
    // This is a compromise: the number increments on every build invocation
    // because Cargo always runs build.rs, but we don't trigger unnecessary
    // recompilation of downstream crates.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.build_number");
}

/// Minimal ISO 8601 UTC formatter (avoids chrono dependency in build script).
fn format_timestamp(epoch_secs: u64) -> String {
    // Days since epoch calculation
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
