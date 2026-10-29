//! gen-config-ref — walk veld's source tree for `VELD_*` and `SHODH_*` environment
//! variables and emit a reference table of them.
//!
//! Two shapes are found: a literal `env::var("…")` call, and a string constant naming a
//! variable (`const X_ENV: &str = "VELD_…";`), which is how switches read through helpers
//! such as `env_switch(c::X_ENV, true)` are declared. A call's default comes from its
//! `unwrap_or(...)` chain; a constant's from the sentence of its doc comment that says what
//! the default is. Anything else is listed as `—`.

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
        r#"(?:std::)?env::var\(\s*"((?:VELD|SHODH)_[A-Z0-9_]+)"\s*\)(?:[^\n]{0,200}?\.unwrap_or(?:_else)?\(\s*\|?\|?\s*"?([^"\n)]*))?"#,
    )
    .expect("regex");
    let const_re = Regex::new(
        r#"(?m)^[ \t]*(?:pub(?:\([a-z]+\))?\s+)?const\s+[A-Z0-9_]+\s*:\s*&(?:'static\s+)?str\s*=\s*"((?:VELD|SHODH)_[A-Z0-9_]+)"\s*;"#,
    )
    .expect("regex");

    // The sentence of a constant's doc comment that states its default, e.g. "On by default."
    fn doc_default(source: &str, decl_start: usize) -> Option<String> {
        let doc: Vec<&str> = source[..decl_start]
            .lines()
            .rev()
            .take_while(|l| l.trim_start().starts_with("///"))
            .map(|l| l.trim_start().trim_start_matches("///").trim())
            .collect();
        let text = doc.into_iter().rev().collect::<Vec<_>>().join(" ");
        text.split_inclusive(". ")
            .map(str::trim)
            .find(|s| s.to_ascii_lowercase().contains("by default"))
            // The table wraps a default in a code span, which a backtick would close early.
            .map(|s| s.replace('`', "").chars().take(160).collect())
    }

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

        let calls = var_re.captures_iter(&source).map(|cap| {
            let hint = cap.get(2).map(|m| m.as_str().trim().to_string()).filter(|s| !s.is_empty());
            (cap[1].to_string(), hint)
        });
        let consts = const_re.captures_iter(&source).map(|cap| {
            let start = cap.get(0).map_or(0, |m| m.start());
            (cap[1].to_string(), doc_default(&source, start))
        });
        for (name, default_hint) in calls.chain(consts).collect::<Vec<_>>() {
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

    let mut out = generated_header(
        "src/**/*.rs (env::var calls and env-name constants)",
        "gen-config-ref",
    );
    out.push_str("# Configuration Reference\n\n");
    out.push_str(&format!(
        "Veld is configured via environment variables. The generator scanned `src/**/*.rs` and found **{}** distinct `VELD_*` and `SHODH_*` variables, read directly or through a named constant.\n\n",
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
    out.push_str("\n---\n\n*Defaults shown above are best-effort extractions from `.unwrap_or(...)` chains and from the constant's doc comment. For full semantics, consult the source file listed.*\n");

    let output_path = docs_src_root().join("reference").join("config.md");
    write_output(&output_path, &out)?;
    eprintln!("gen-config-ref: extracted {} env vars", vars.len());
    Ok(())
}
