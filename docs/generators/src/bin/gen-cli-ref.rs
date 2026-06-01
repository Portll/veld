//! gen-cli-ref — produce `docs/src/reference/cli.md` from the live output of
//! `veld --help` and `veld <subcommand> --help`.
//!
//! This is the one generator that requires a built `veld` binary. The binary
//! is located in this order:
//!   1. `$VELD_BIN` env var (absolute path).
//!   2. `target/release/veld` relative to the repo root.
//!   3. `veld` from PATH.
//!
//! If none resolve, the generator emits a warning and exits with code 0 —
//! leaving the existing hand-authored `cli.md` untouched so pre-commit hooks
//! don't fail in environments without a release build. CI sets `VELD_BIN`
//! explicitly after `cargo build --release`.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use veld_docs_generators::{docs_src_root, generated_header, repo_root, write_output};

/// All veld subcommands. Kept in sync with the `Commands` enum in
/// [src/cli.rs](https://github.com/Portll/veld/blob/main/src/cli.rs). If the
/// CLI grows a subcommand, add it here.
const SUBCOMMANDS: &[&str] = &[
    "server", "tui", "serve", "init", "status", "doctor", "hook", "claude", "version",
];

fn locate_veld() -> Option<PathBuf> {
    // 1. Explicit env override (CI sets this).
    if let Ok(p) = std::env::var("VELD_BIN") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    // 2. target/release/veld at repo root.
    let release = repo_root().join("target").join("release").join(
        if cfg!(windows) { "veld.exe" } else { "veld" },
    );
    if release.exists() {
        return Some(release);
    }
    // 3. PATH lookup.
    let from_path = if cfg!(windows) {
        which("veld.exe").or_else(|| which("veld"))
    } else {
        which("veld")
    };
    from_path
}

fn which(bin: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(bin);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn capture(bin: &PathBuf, args: &[&str]) -> Result<String> {
    let out = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running `{} {:?}`", bin.display(), args))?;
    if !out.status.success() {
        anyhow::bail!(
            "`{} {:?}` exited with code {:?}: {}",
            bin.display(),
            args,
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // `clap` prints help on stdout. Some environments emit ANSI colour codes;
    // strip the common SGR prefix so the markdown code block reads clean.
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    Ok(strip_ansi(&s))
}

/// Light-weight ANSI-stripper for the common SGR sequence `ESC[...m`. We do
/// not try to handle every escape — just the colour codes clap emits.
fn strip_ansi(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Scan to the terminator (anything in @..~ range).
            let mut j = i + 2;
            while j < bytes.len() && !(0x40..=0x7e).contains(&bytes[j]) {
                j += 1;
            }
            i = j.saturating_add(1);
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn main() -> Result<()> {
    let bin = match locate_veld() {
        Some(b) => b,
        None => {
            eprintln!(
                "gen-cli-ref: no veld binary found (set VELD_BIN, or build with \
                 `cargo build --release`). Skipping — existing cli.md left in place."
            );
            return Ok(());
        }
    };

    eprintln!("gen-cli-ref: using {}", bin.display());

    let mut out = generated_header("`veld --help` (runtime)", "gen-cli-ref");
    out.push_str("# CLI reference\n\n");
    out.push_str(&format!(
        "Generated from the live output of `veld --help` and \
         `veld <subcommand> --help`. Binary used: `{}`.\n\n",
        bin.display()
    ));

    // Top-level help.
    out.push_str("## `veld --help`\n\n```text\n");
    out.push_str(&capture(&bin, &["--help"])?);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("```\n\n");

    // Per-subcommand help.
    for cmd in SUBCOMMANDS {
        let body = match capture(&bin, &[cmd, "--help"]) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("gen-cli-ref: `veld {cmd} --help` failed: {e}; skipping");
                continue;
            }
        };
        out.push_str(&format!("## `veld {cmd}`\n\n```text\n"));
        out.push_str(&body);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("```\n\n");
    }

    let output_path = docs_src_root().join("reference").join("cli.md");
    write_output(&output_path, &out)?;
    eprintln!("gen-cli-ref: wrote {}", output_path.display());
    Ok(())
}
