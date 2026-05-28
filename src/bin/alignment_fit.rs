//! Fit a global alignment from a pairs file.
//!
//! Encodes `pairs.jsonl` with both embedders, splits 90/10 (deterministic by
//! text hash), fits Procrustes (default) or Ridge, evaluates on the held-out
//! 10%, and persists to `evaluations/alignment/fitted/<pair_id>.bin`.
//!
//! The `--side` flag controls which face of an asymmetric embedder is used:
//! `doc` (default) calls `encode()`, `query` calls `encode_for_query()`. For
//! Nomic+MiniLM the secondary side (MiniLM) is symmetric so the flag only
//! affects the primary side; for future model pairs both sides may differ.

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use veld::embeddings::{
    save_alignment, Alignment, AlignmentPairId, Embedder, ProcrustesAlignment, RidgeAlignment,
};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Side {
    Doc,
    Query,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Method {
    Procrustes,
    Ridge,
}

#[derive(Parser)]
#[command(name = "alignment-fit")]
struct Args {
    #[arg(
        long,
        default_value = "evaluations/alignment/pairs/pairs.jsonl"
    )]
    pairs: PathBuf,

    #[arg(long, default_value = "evaluations/alignment/fitted")]
    out_dir: PathBuf,

    /// Held-out fraction.
    #[arg(long, default_value_t = 0.1)]
    holdout: f32,

    /// Acceptance threshold on mean paired cosine for the held-out split.
    #[arg(long, default_value_t = 0.80)]
    min_cosine: f32,

    /// Primary embedder identifier (e.g. `nomic-embed-text-v1.5`).
    #[arg(long)]
    primary_id: String,

    /// Secondary embedder identifier (e.g. `minilm-l6-v2`).
    #[arg(long)]
    secondary_id: String,

    /// Which face of asymmetric embedders to use (doc | query).
    #[arg(long, value_enum, default_value_t = Side::Doc)]
    side: Side,

    /// Fit method (procrustes | ridge).
    #[arg(long, value_enum, default_value_t = Method::Procrustes)]
    method: Method,

    /// Ridge regularisation strength (only used with `--method ridge`).
    #[arg(long, default_value_t = 0.01)]
    lambda: f32,

    /// Multi-lambda Ridge sweep. When set with `--method ridge`, encodes once
    /// and fits one alignment per lambda, saving each as
    /// `<pair>.ridge-<lambda>.bin`. The lambda with the highest held-out
    /// cosine is also written to the canonical `<pair>.bin` path. The
    /// `--lambda` flag is ignored when this is set.
    /// Format: comma-separated floats, e.g. `--lambdas 1e-3,1e-2,1e-1,1.0`.
    #[arg(long, value_delimiter = ',')]
    lambdas: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct PairRow {
    text: String,
    #[allow(dead_code)]
    domain: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt::try_init().ok();

    let pair_id = AlignmentPairId::new(args.primary_id.clone(), args.secondary_id.clone());

    let primary: Arc<dyn Embedder> = veld::embeddings::load_primary_embedder(&args.primary_id)
        .context("loading primary embedder")?;
    let secondary: Arc<dyn Embedder> = veld::embeddings::load_secondary_embedder(&args.secondary_id)
        .context("loading secondary embedder")?;

    let file = File::open(&args.pairs)
        .with_context(|| format!("opening {}", args.pairs.display()))?;
    let rows: Vec<PairRow> = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    if rows.is_empty() {
        bail!("no rows in {}", args.pairs.display());
    }
    tracing::info!("loaded {} text rows", rows.len());

    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let side = args.side;
    let encode = |e: &Arc<dyn Embedder>, t: &str| -> Result<Vec<f32>> {
        match side {
            Side::Doc => e.encode(t),
            Side::Query => e.encode_for_query(t),
        }
    };

    let primary_vecs: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| encode(&primary, t))
        .collect::<Result<Vec<_>>>()?;
    let secondary_vecs: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| encode(&secondary, t))
        .collect::<Result<Vec<_>>>()?;

    let cutoff = (args.holdout * 10.0).round() as usize;
    let mut train_p = Vec::new();
    let mut train_s = Vec::new();
    let mut hold_p = Vec::new();
    let mut hold_s = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let bucket = stable_bucket(&row.text);
        if bucket < cutoff {
            hold_p.push(primary_vecs[i].clone());
            hold_s.push(secondary_vecs[i].clone());
        } else {
            train_p.push(primary_vecs[i].clone());
            train_s.push(secondary_vecs[i].clone());
        }
    }
    tracing::info!("train={} holdout={}", train_p.len(), hold_p.len());

    let cos: f32;
    let out_path = args.out_dir.join(pair_id.cache_filename());
    std::fs::create_dir_all(&args.out_dir)?;

    match args.method {
        Method::Procrustes => {
            let mut alignment = ProcrustesAlignment::fit(pair_id.clone(), &train_p, &train_s)
                .context("Procrustes fit")?;
            cos = mean_paired_cosine(&alignment, &hold_p, &hold_s)?;
            tracing::info!("held-out mean paired cosine: {:.4}", cos);
            alignment.set_eval(cos);
            if cos < args.min_cosine {
                bail!(
                    "alignment quality {:.4} below acceptance threshold {:.4} — refusing to install",
                    cos,
                    args.min_cosine
                );
            }
            save_alignment(&out_path, &alignment)?;
        }
        Method::Ridge => {
            let lambdas: Vec<f32> = if args.lambdas.is_empty() {
                vec![args.lambda]
            } else {
                args.lambdas.clone()
            };

            let mut best: Option<(f32, f32, std::path::PathBuf)> = None;
            for lam in &lambdas {
                let mut alignment =
                    RidgeAlignment::fit(pair_id.clone(), &train_p, &train_s, *lam)
                        .with_context(|| format!("Ridge fit at lambda={lam}"))?;
                let lam_cos = mean_paired_cosine(&alignment, &hold_p, &hold_s)?;
                tracing::info!(
                    "held-out mean paired cosine: {:.4} (lambda={})",
                    lam_cos,
                    lam
                );
                alignment.set_eval(lam_cos);

                // Per-lambda snapshot file when sweeping.
                if lambdas.len() > 1 {
                    let snap_name = format!(
                        "{}.ridge-{}.bin",
                        pair_id
                            .cache_filename()
                            .trim_end_matches(".bin"),
                        format_lambda(*lam)
                    );
                    let snap_path = args.out_dir.join(snap_name);
                    save_alignment(&snap_path, &alignment)?;
                }

                let is_best = best.as_ref().map(|(c, _, _)| lam_cos > *c).unwrap_or(true);
                if is_best {
                    save_alignment(&out_path, &alignment)?;
                    best = Some((lam_cos, *lam, out_path.clone()));
                }
            }

            let (best_cos, best_lam, _) = best.expect("at least one lambda");
            cos = best_cos;
            tracing::info!(
                "best held-out cosine: {:.4} at lambda={} (canonical install: {})",
                best_cos,
                best_lam,
                out_path.display()
            );
            if cos < args.min_cosine {
                bail!(
                    "best alignment quality {:.4} below acceptance threshold {:.4}",
                    cos,
                    args.min_cosine
                );
            }
        }
    }

    tracing::info!("installed alignment at {}", out_path.display());
    Ok(())
}

fn mean_paired_cosine<A: Alignment>(
    a: &A,
    primary: &[Vec<f32>],
    secondary: &[Vec<f32>],
) -> Result<f32> {
    if primary.is_empty() {
        return Ok(0.0);
    }
    let mut sum = 0.0_f32;
    for (p, s) in primary.iter().zip(secondary.iter()) {
        let projected = a.project(s)?;
        let dot: f32 = projected.iter().zip(p.iter()).map(|(x, y)| x * y).sum();
        let np: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
        let pp: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        sum += dot / (np * pp).max(1e-12);
    }
    Ok(sum / primary.len() as f32)
}

/// Render a lambda value for inclusion in a filename — no exponents, no
/// scientific notation. `1e-3` → `0.001`, `1.0` → `1.0`.
fn format_lambda(lam: f32) -> String {
    let s = format!("{lam:.6}");
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

fn stable_bucket(text: &str) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    (h.finish() as usize) % 10
}
