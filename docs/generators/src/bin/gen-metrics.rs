//! gen-metrics — scan veld's source tree for Prometheus metric declarations
//! and emit a markdown reference table.
//!
//! Veld declares metrics using direct `prometheus` API calls inside
//! `LazyLock<...>` initializers, not the `register_*!` macros. Two patterns:
//!
//!   pub static NAME: LazyLock<TYPE> = LazyLock::new(|| {
//!       TYPE::new(Opts::new("metric_name", "metric_help"), &["labels"]) ...
//!   });
//!
//!   pub static NAME: LazyLock<TYPE> = LazyLock::new(|| {
//!       TYPE::new(latency_histogram_opts("metric_name", "metric_help"), &["labels"]) ...
//!   });
//!
//! Strategy: find every static metric declaration, capture its rust identifier
//! and metric type, then capture the FIRST `Opts::new(..)` or
//! `latency_histogram_opts(..)` call following it (within the same closure)
//! for name/help. This is the source-of-truth for what `/metrics` emits.

use anyhow::Result;
use regex::Regex;
use walkdir::WalkDir;

use veld_docs_generators::{docs_src_root, generated_header, repo_root, write_output};

#[derive(Debug, Clone)]
struct Metric {
    name: String,
    kind: String,
    help: String,
    file: String,
}

fn main() -> Result<()> {
    let src_root = repo_root().join("src");
    // (?s) = dot matches newline. Find:
    //   LazyLock<TYPE> = LazyLock::new(|| { ... ("name", "help") ... });
    // The closure body can span many lines; we cap the lookahead to 600 chars.
    let metric_re = Regex::new(
        r#"(?s)static\s+([A-Z][A-Z0-9_]+):\s*LazyLock<([A-Za-z]+)>\s*=\s*LazyLock::new\(\s*\|\|\s*\{.{0,600}?(?:Opts::new|latency_histogram_opts)\(\s*"([^"]+)"\s*,\s*"([^"]+)""#,
    )
    .expect("regex");

    let mut metrics: Vec<Metric> = Vec::new();

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
        for cap in metric_re.captures_iter(&source) {
            // cap[1] = const ident (e.g. HTTP_REQUEST_DURATION)
            // cap[2] = metric type (e.g. HistogramVec)
            // cap[3] = metric name ("veld_http_request_duration_seconds")
            // cap[4] = help text
            metrics.push(Metric {
                kind: cap[2].to_string(),
                name: cap[3].to_string(),
                help: cap[4].to_string(),
                file: rel.clone(),
            });
        }
    }

    // Dedupe by name (some metrics are registered through a macro that expands;
    // we want to surface each metric once).
    metrics.sort_by(|a, b| a.name.cmp(&b.name));
    metrics.dedup_by(|a, b| a.name == b.name);

    let mut out = generated_header(
        "src/**/*.rs (register_* macro calls)",
        "gen-metrics",
    );
    out.push_str("# Metrics Reference\n\n");
    out.push_str(&format!(
        "Veld exposes **{}** Prometheus metrics on the `/metrics` endpoint. Visibility is configured by `VELD_METRICS_PUBLIC` — when `true`, `/metrics` is unauthenticated; otherwise it requires an API key.\n\n",
        metrics.len()
    ));
    if metrics.is_empty() {
        out.push_str("*No metrics were extracted. The registration pattern may have changed.*\n");
    } else {
        out.push_str("| Name | Kind | Help | Source |\n|---|---|---|---|\n");
        for m in &metrics {
            let safe_help = m.help.replace('|', "\\|");
            out.push_str(&format!(
                "| `{}` | {} | {} | `{}` |\n",
                m.name, m.kind, safe_help, m.file
            ));
        }
    }
    out.push_str("\n---\n\n*Metric kind is the suffix of the macro: `counter`, `gauge`, `histogram`, `int_counter`, `int_counter_vec`, `histogram_vec`, etc.*\n");

    let output_path = docs_src_root().join("reference").join("metrics.md");
    write_output(&output_path, &out)?;
    eprintln!("gen-metrics: extracted {} metrics", metrics.len());
    Ok(())
}
