//! Evaluate an installed alignment against a pairs file.
//!
//! Dispatches on `header.method` so both Procrustes and Ridge alignments
//! work. Uses the `pairs.<pair>.<side>.vec` cache when present (written by
//! `alignment-fit`) so re-evaluation against the same corpus skips the
//! expensive ONNX pass.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use veld::embeddings::{
    read_alignment_file, Alignment, AlignmentPairId, Embedder, ProcrustesAlignment, RidgeAlignment,
};

const PAIRS_VEC_MAGIC: &[u8; 8] = b"VELDPV1\0";

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Side {
    Doc,
    Query,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    alignment: PathBuf,
    #[arg(long)]
    pairs: PathBuf,
    #[arg(long)]
    primary_id: String,
    #[arg(long)]
    secondary_id: String,
    #[arg(long, value_enum, default_value_t = Side::Doc)]
    side: Side,
}

#[derive(Deserialize)]
struct Row {
    text: String,
    domain: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt::try_init().ok();

    let (header, payload) = read_alignment_file(&args.alignment)?;
    let expected_pid = AlignmentPairId::new(&args.primary_id, &args.secondary_id);
    if header.pair_id != expected_pid {
        anyhow::bail!(
            "pair_id mismatch: file = {:?}, expected = {:?}",
            header.pair_id,
            expected_pid
        );
    }
    let method = header.method.clone();
    let alignment: Box<dyn Alignment> = match method.as_str() {
        "orthogonal_procrustes" => Box::new(ProcrustesAlignment::from_payload(header, &payload)?),
        "ridge" => Box::new(RidgeAlignment::from_payload(header, &payload)?),
        other => anyhow::bail!(
            "unsupported alignment method `{}` — alignment-eval handles procrustes and ridge",
            other
        ),
    };

    let file = File::open(&args.pairs)
        .with_context(|| format!("opening {}", args.pairs.display()))?;
    let rows: Vec<Row> = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();

    let side = args.side;

    let pairs_sha = sha256_file(&args.pairs)
        .with_context(|| format!("hashing {}", args.pairs.display()))?;
    let cache_path = pairs_vec_path(&args.pairs, &expected_pid, side);

    let (pv, sv): (Vec<Vec<f32>>, Vec<Vec<f32>>) =
        match load_pairs_vec(&cache_path, &pairs_sha, rows.len())? {
            Some(cached) => {
                tracing::info!("vector cache hit at {}", cache_path.display());
                cached
            }
            None => {
                tracing::info!(
                    "no usable cache at {} — encoding {} rows",
                    cache_path.display(),
                    rows.len()
                );
                let primary: Arc<dyn Embedder> =
                    veld::embeddings::load_primary_embedder(&args.primary_id)?;
                let secondary: Arc<dyn Embedder> =
                    veld::embeddings::load_secondary_embedder(&args.secondary_id)?;
                let encode = |e: &Arc<dyn Embedder>, t: &str| -> Result<Vec<f32>> {
                    match side {
                        Side::Doc => e.encode(t),
                        Side::Query => e.encode_for_query(t),
                    }
                };
                let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
                let p: Vec<Vec<f32>> = texts
                    .par_iter()
                    .map(|t| encode(&primary, t))
                    .collect::<Result<_>>()?;
                let s: Vec<Vec<f32>> = texts
                    .par_iter()
                    .map(|t| encode(&secondary, t))
                    .collect::<Result<_>>()?;
                (p, s)
            }
        };

    let mut by_domain: std::collections::BTreeMap<String, (f32, usize)> = Default::default();
    let mut overall: (f32, usize) = (0.0, 0);

    for (i, row) in rows.iter().enumerate() {
        let projected = alignment.project(&sv[i])?;
        let dot: f32 = projected.iter().zip(pv[i].iter()).map(|(x, y)| x * y).sum();
        let np: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
        let pp: f32 = pv[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (np * pp).max(1e-12);
        let e = by_domain.entry(row.domain.clone()).or_insert((0.0, 0));
        e.0 += cos;
        e.1 += 1;
        overall.0 += cos;
        overall.1 += 1;
    }

    println!("=== alignment eval ===");
    println!("file:   {}", args.alignment.display());
    println!("method: {}", alignment.method());
    println!(
        "overall mean paired cosine: {:.4} (n={})",
        overall.0 / overall.1.max(1) as f32,
        overall.1
    );
    println!("per-domain:");
    for (d, (sum, n)) in &by_domain {
        println!("  {d}: {:.4} (n={n})", sum / *n as f32);
    }
    Ok(())
}

fn pairs_vec_path(pairs: &Path, pid: &AlignmentPairId, side: Side) -> PathBuf {
    let side_str = match side {
        Side::Doc => "doc",
        Side::Query => "query",
    };
    let stem = format!(
        "pairs.{}__{}.{}.vec",
        pid.primary.replace('/', "_"),
        pid.secondary.replace('/', "_"),
        side_str
    );
    pairs
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(stem)
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

fn load_pairs_vec(
    path: &Path,
    expected_sha: &[u8; 32],
    expected_n: usize,
) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    if r.read_exact(&mut magic).is_err() || &magic != PAIRS_VEC_MAGIC {
        return Ok(None);
    }
    let mut sha = [0u8; 32];
    if r.read_exact(&mut sha).is_err() || sha != *expected_sha {
        return Ok(None);
    }
    let mut n_bytes = [0u8; 8];
    r.read_exact(&mut n_bytes)?;
    let n = u64::from_le_bytes(n_bytes) as usize;
    if n != expected_n {
        return Ok(None);
    }
    let mut dp_bytes = [0u8; 4];
    r.read_exact(&mut dp_bytes)?;
    let d_p = u32::from_le_bytes(dp_bytes) as usize;
    let mut ds_bytes = [0u8; 4];
    r.read_exact(&mut ds_bytes)?;
    let d_s = u32::from_le_bytes(ds_bytes) as usize;

    let mut primary = Vec::with_capacity(n);
    let mut secondary = Vec::with_capacity(n);
    let mut p_buf = vec![0u8; d_p * 4];
    let mut s_buf = vec![0u8; d_s * 4];
    for _ in 0..n {
        r.read_exact(&mut p_buf)?;
        primary.push(
            p_buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>(),
        );
        r.read_exact(&mut s_buf)?;
        secondary.push(
            s_buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>(),
        );
    }
    Ok(Some((primary, secondary)))
}
