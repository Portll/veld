//! Shared helpers for docs generators.
//!
//! Every generator emits the same header pointing at the source file so a
//! developer landing on a reference page knows where to fix it. Every
//! generator is deterministic — running it twice on the same input produces
//! byte-identical output.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Project root, relative to a generator running with `cwd = docs/generators/`.
///
/// Resolves to `<repo>/`. Generators read inputs at `repo_root().join("src/...")`.
pub fn repo_root() -> PathBuf {
    // docs/generators/ → docs/ → repo root.
    PathBuf::from("..").join("..")
}

/// Docs source root, relative to a generator with `cwd = docs/generators/`.
pub fn docs_src_root() -> PathBuf {
    PathBuf::from("..").join("src")
}

/// Standard "GENERATED" header. Stamped at the top of every generated file.
pub fn generated_header(source_file: &str, generator_name: &str) -> String {
    format!(
        "<!-- GENERATED FILE — do not edit by hand.\n     \
         Source: {source_file}\n     \
         Generator: docs/generators/src/bin/{generator_name}.rs\n     \
         Regenerate: cd docs/generators && cargo run --bin {generator_name} -->\n\n"
    )
}

/// Write `content` to `path`, ensuring parent directory exists. Idempotent.
pub fn write_output(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating parent of {}", path.display()))?;
    }
    fs::write(path, content).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Read a source file, returning a descriptive error if missing.
pub fn read_source(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .with_context(|| format!("reading source {}", path.display()))
}
