//! gen-module-index — parse `src/lib.rs` for top-level `pub mod` declarations
//! and read each module's `//!` doc comment for the one-line summary, emit a
//! markdown table.
//!
//! Strategy: parse `src/lib.rs` via `syn`, collect `Item::Mod` entries with
//! `pub` visibility. For each, look at `src/<name>.rs` or `src/<name>/mod.rs`
//! and extract the first contiguous block of `//!` lines as the description.

use anyhow::Result;
use std::collections::BTreeMap;
use syn::{Item, Visibility};

use veld_docs_generators::{docs_src_root, generated_header, repo_root, write_output};

fn module_path(name: &str) -> Option<std::path::PathBuf> {
    let root = repo_root().join("src");
    let flat = root.join(format!("{name}.rs"));
    if flat.exists() {
        return Some(flat);
    }
    let dir = root.join(name).join("mod.rs");
    if dir.exists() {
        return Some(dir);
    }
    None
}

/// Extract the first contiguous block of `//!` doc comments at the top of the
/// file. Returns a single-line summary (joined with spaces, stripped).
fn extract_summary(path: &std::path::Path) -> String {
    let Ok(source) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let mut lines = Vec::new();
    let mut started = false;
    for raw in source.lines() {
        let line = raw.trim_start();
        if line.starts_with("//!") {
            started = true;
            let trimmed = line.trim_start_matches("//!").trim();
            if !trimmed.is_empty() {
                lines.push(trimmed.to_string());
            } else if !lines.is_empty() {
                // blank doc line — end of first summary paragraph
                break;
            }
        } else if started {
            break;
        }
        if !started && !line.is_empty() && !line.starts_with("//") && !line.starts_with("#!") {
            // hit code before any doc comment — abort
            break;
        }
    }
    lines.join(" ")
}

fn main() -> Result<()> {
    let lib_path = repo_root().join("src").join("lib.rs");
    let source = std::fs::read_to_string(&lib_path)?;
    let file = syn::parse_file(&source)?;

    let mut modules: BTreeMap<String, String> = BTreeMap::new();
    let mut feature_gated: BTreeMap<String, String> = BTreeMap::new();

    for item in &file.items {
        let Item::Mod(m) = item else { continue };
        if !matches!(m.vis, Visibility::Public(_)) {
            continue;
        }
        let name = m.ident.to_string();

        // Feature-gated? Check cfg attributes.
        let cfg_gated = m
            .attrs
            .iter()
            .any(|a| a.path().is_ident("cfg") || a.path().is_ident("cfg_attr"));

        let summary = module_path(&name)
            .map(|p| extract_summary(&p))
            .unwrap_or_default();
        let summary = if summary.is_empty() {
            "—".to_string()
        } else {
            // truncate to ~140 chars
            if summary.len() > 140 {
                format!("{}…", &summary[..137])
            } else {
                summary
            }
        };

        if cfg_gated {
            feature_gated.insert(name, summary);
        } else {
            modules.insert(name, summary);
        }
    }

    let mut out = generated_header("src/lib.rs", "gen-module-index");
    out.push_str("# Module index\n\n");
    out.push_str(&format!(
        "Top-level Rust modules in veld. **{}** modules (plus **{}** feature-gated).\n\n",
        modules.len(),
        feature_gated.len()
    ));
    out.push_str("Each entry's summary is the first paragraph of that module's `//!` doc comment. Click into the crate docs (built by `cargo doc`) for full API.\n\n");
    out.push_str("## Always-on\n\n");
    out.push_str("| Module | Summary |\n|---|---|\n");
    for (name, summary) in &modules {
        let safe = summary.replace('|', "\\|");
        out.push_str(&format!("| `{name}` | {safe} |\n"));
    }
    out.push('\n');
    if !feature_gated.is_empty() {
        out.push_str("## Feature-gated\n\n");
        out.push_str("These modules are compiled only when their feature flag is enabled (e.g., `cargo build --features python`).\n\n");
        out.push_str("| Module | Summary |\n|---|---|\n");
        for (name, summary) in &feature_gated {
            let safe = summary.replace('|', "\\|");
            out.push_str(&format!("| `{name}` | {safe} |\n"));
        }
        out.push('\n');
    }
    out.push_str("---\n\n*This table is regenerated from `src/lib.rs`. For deeper API reference, see the rustdoc output stitched into the docs site at `/api/`.*\n");

    let output_path = docs_src_root().join("architecture").join("module-index.md");
    write_output(&output_path, &out)?;
    eprintln!(
        "gen-module-index: {} always-on + {} feature-gated = {} total",
        modules.len(),
        feature_gated.len(),
        modules.len() + feature_gated.len()
    );
    Ok(())
}
