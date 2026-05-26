//! Corpus-assembly CLI for embedding alignment.
//!
//! Ingests text from declared sources (filesystem trees, friends-contributed
//! JSONL files), de-duplicates by content hash, applies per-domain quotas, and
//! writes a `pairs.jsonl` ready to be encoded by `alignment-fit`. Vector
//! encoding is intentionally NOT done here — that depends on the embedders
//! loaded at fit time, which this CLI does not need.

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "alignment-collect")]
struct Args {
    /// Output file (.jsonl)
    #[arg(
        short,
        long,
        default_value = "evaluations/alignment/pairs/pairs.jsonl"
    )]
    out: PathBuf,

    /// Per-domain target row count (rows past the quota are dropped).
    #[arg(long, default_value_t = 5000)]
    quota: usize,

    /// Source spec files (TOML). Multiple allowed.
    #[arg(long)]
    sources: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct SourceSpec {
    /// e.g. "web_development", "programming", "ai_loop"
    domain: String,
    /// "filesystem" | "jsonl"
    kind: String,
    /// path or directory
    path: PathBuf,
    /// for filesystem: file extensions to include (e.g. ["rs","md","sql"])
    #[serde(default)]
    extensions: Vec<String>,
    /// per-row license tag (SPDX)
    license: String,
    /// source label written to output (e.g. "veld-repo-rust")
    source: String,
    /// for filesystem: minimum chunk size in bytes
    #[serde(default = "default_min")]
    min_chunk: usize,
    /// for filesystem: maximum chunk size in bytes
    #[serde(default = "default_max")]
    max_chunk: usize,
}

fn default_min() -> usize {
    64
}
fn default_max() -> usize {
    2048
}

#[derive(Debug, Serialize)]
struct PairRow {
    text: String,
    text_hash: String,
    domain: String,
    source: String,
    license: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.sources.is_empty() {
        bail!("at least one --sources spec file is required");
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut per_domain: HashMap<String, usize> = HashMap::new();

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&args.out)
        .with_context(|| format!("opening {}", args.out.display()))?;
    let mut out_writer = BufWriter::new(out_file);

    for spec_path in &args.sources {
        let spec_bytes = std::fs::read_to_string(spec_path)
            .with_context(|| format!("reading {}", spec_path.display()))?;
        let spec: SourceSpec = toml::from_str(&spec_bytes)
            .with_context(|| format!("parsing {}", spec_path.display()))?;

        eprintln!(
            "[collect] domain={} kind={} path={}",
            spec.domain,
            spec.kind,
            spec.path.display()
        );

        match spec.kind.as_str() {
            "filesystem" => {
                collect_filesystem(
                    &spec,
                    &mut seen,
                    &mut per_domain,
                    args.quota,
                    &mut out_writer,
                )?;
            }
            "jsonl" => {
                collect_jsonl(&spec, &mut seen, &mut per_domain, args.quota, &mut out_writer)?;
            }
            other => bail!("unknown source kind: {other}"),
        }
    }

    out_writer.flush()?;
    eprintln!("[collect] per-domain row counts:");
    for (d, n) in &per_domain {
        eprintln!("  {d}: {n}");
    }
    Ok(())
}

fn hash_text(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}

fn write_row<W: Write>(
    w: &mut W,
    row: PairRow,
    seen: &mut HashSet<String>,
    per_domain: &mut HashMap<String, usize>,
    quota: usize,
) -> Result<bool> {
    if !seen.insert(row.text_hash.clone()) {
        return Ok(false);
    }
    let count = per_domain.entry(row.domain.clone()).or_insert(0);
    if *count >= quota {
        return Ok(false);
    }
    *count += 1;
    serde_json::to_writer(&mut *w, &row)?;
    w.write_all(b"\n")?;
    Ok(true)
}

fn collect_filesystem<W: Write>(
    spec: &SourceSpec,
    seen: &mut HashSet<String>,
    per_domain: &mut HashMap<String, usize>,
    quota: usize,
    w: &mut W,
) -> Result<()> {
    let exts: HashSet<&str> = spec.extensions.iter().map(String::as_str).collect();
    for entry in WalkDir::new(&spec.path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        if !exts.is_empty() {
            let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) else {
                continue;
            };
            if !exts.contains(ext) {
                continue;
            }
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for chunk in chunk_text(&content, spec.min_chunk, spec.max_chunk) {
            let row = PairRow {
                text_hash: hash_text(&chunk),
                text: chunk,
                domain: spec.domain.clone(),
                source: spec.source.clone(),
                license: spec.license.clone(),
            };
            let written = write_row(w, row, seen, per_domain, quota)?;
            if !written && per_domain.get(&spec.domain).copied().unwrap_or(0) >= quota {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn collect_jsonl<W: Write>(
    spec: &SourceSpec,
    seen: &mut HashSet<String>,
    per_domain: &mut HashMap<String, usize>,
    quota: usize,
    w: &mut W,
) -> Result<()> {
    #[derive(Deserialize)]
    struct Incoming {
        text: String,
    }
    let file = File::open(&spec.path)
        .with_context(|| format!("opening {}", spec.path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let incoming: Incoming = serde_json::from_str(&line)?;
        if incoming.text.len() < spec.min_chunk {
            continue;
        }
        if incoming.text.len() > spec.max_chunk {
            continue;
        }
        let row = PairRow {
            text_hash: hash_text(&incoming.text),
            text: incoming.text,
            domain: spec.domain.clone(),
            source: spec.source.clone(),
            license: spec.license.clone(),
        };
        let _ = write_row(w, row, seen, per_domain, quota)?;
        if per_domain.get(&spec.domain).copied().unwrap_or(0) >= quota {
            break;
        }
    }
    Ok(())
}

/// Greedy paragraph-respecting chunker. Splits on blank lines first; if a
/// paragraph exceeds `max`, breaks on the nearest line boundary.
fn chunk_text(content: &str, min: usize, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for para in content.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if buf.len() + para.len() + 2 > max {
            if buf.len() >= min {
                out.push(std::mem::take(&mut buf));
            }
            if para.len() > max {
                for line in para.lines() {
                    if buf.len() + line.len() + 1 > max && buf.len() >= min {
                        out.push(std::mem::take(&mut buf));
                    }
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(line);
                }
            } else {
                buf.push_str(para);
            }
        } else {
            if !buf.is_empty() {
                buf.push_str("\n\n");
            }
            buf.push_str(para);
        }
    }
    if buf.len() >= min {
        out.push(buf);
    }
    out
}
