//! gen-config-ref — walk veld's source tree for `env::var("VELD_*")` and
//! `std::env::var("VELD_*")` calls and emit a reference table of env vars.
//!
//! Captures the literal var name. Default values and descriptions are inferred
//! from surrounding context where possible (subsequent `unwrap_or(...)` chain),
//! otherwise listed as `—`.

use anyhow::Result;
use regex::Regex;
use walkdir::WalkDir;

use veld_docs_generators::{docs_src_root, generated_header, repo_root, write_output};

#[derive(Debug, Clone)]
struct EnvVar {
    name: String,
    files: Vec<String>,
    default_hint: Option<String>,
}

fn main() -> Result<()> {
    let src_root = repo_root().join("src");
    // Default-capture stops at newline, quote, or closing paren so we never
    // pull multi-line Rust source (which would leak `<Vec>` / `<PathBuf>`
    // tokens that mdbook treats as unclosed HTML tags).
    let var_re = Regex::new(
        r#"(?:std::)?env::var\(\s*"(VELD_[A-Z0-9_]+)"\s*\)(?:[^\n]{0,200}?\.unwrap_or(?:_else)?\(\s*\|?\|?\s*"?([^"\n)]*))?"#,
    )
    .expect("regex");

    // Escape any stray angle brackets so mdbook does not parse them as HTML.
    fn escape_md(s: &str) -> String {
        s.replace('<', "&lt;").replace('>', "&gt;").replace('|', "\\|")
    }

    let mut vars: std::collections::BTreeMap<String, EnvVar> = std::collections::BTreeMap::new();

    for entry in WalkDir::new(&src_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
    {
        let path = entry.path();
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        let rel = path
            .strip_prefix(repo_root())
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        for cap in var_re.captures_iter(&source) {
            let name = cap[1].to_string();
            let default_hint = cap.get(2).map(|m| m.as_str().trim().to_string()).filter(|s| !s.is_empty());
            let entry = vars.entry(name.clone()).or_insert_with(|| EnvVar {
                name: name.clone(),
                files: Vec::new(),
                default_hint: None,
            });
            if !entry.files.contains(&rel) {
                entry.files.push(rel.clone());
            }
            if entry.default_hint.is_none() && default_hint.is_some() {
                entry.default_hint = default_hint;
            }
        }
    }

    let mut out = generated_header("src/**/*.rs (env::var calls)", "gen-config-ref");
    out.push_str("# Configuration Reference\n\n");
    out.push_str(&format!(
        "Veld is configured via environment variables. The generator scanned `src/**/*.rs` and found **{}** distinct `VELD_*` variables.\n\n",
        vars.len()
    ));
    out.push_str("| Variable | Default | First seen in |\n|---|---|---|\n");
    for v in vars.values() {
        // Some default_hint captures are garbage (Rust source fragments from
        // chained closures). Treat anything that contains a closing brace,
        // a lambda pipe, or `::` as untrustworthy and surface it as a hint
        // (still escaped) rather than dropping it silently.
        let default = v
            .default_hint
            .as_deref()
            .filter(|d| !d.is_empty())
            .map(|d| format!("`{}`", escape_md(d)))
            .unwrap_or_else(|| "—".to_string());
        let file = v.files.first().map(String::as_str).unwrap_or("—");
        out.push_str(&format!("| `{}` | {} | `{}` |\n", escape_md(&v.name), default, file));
    }
    out.push_str("\n---\n\n*Defaults shown above are best-effort extractions from `.unwrap_or(...)` chains. For full semantics, consult the source file listed.*\n");

    let output_path = docs_src_root().join("reference").join("config.md");
    write_output(&output_path, &out)?;
    eprintln!("gen-config-ref: extracted {} env vars", vars.len());
    Ok(())
}
