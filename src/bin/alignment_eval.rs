//! Evaluate an installed alignment against a held-out pairs file.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use veld::embeddings::{
    read_alignment_file, Alignment, AlignmentPairId, Embedder, ProcrustesAlignment,
};

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
    if header.method != "orthogonal_procrustes" {
        anyhow::bail!(
            "unsupported alignment method `{}` — alignment-eval handles `orthogonal_procrustes` only",
            header.method
        );
    }
    let alignment = ProcrustesAlignment::from_payload(header, &payload)?;

    let primary: Arc<dyn Embedder> = veld::embeddings::load_primary_embedder(&args.primary_id)?;
    let secondary: Arc<dyn Embedder> =
        veld::embeddings::load_secondary_embedder(&args.secondary_id)?;

    let file = File::open(&args.pairs)
        .with_context(|| format!("opening {}", args.pairs.display()))?;
    let rows: Vec<Row> = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();

    let side = args.side;
    let encode = |e: &Arc<dyn Embedder>, t: &str| -> Result<Vec<f32>> {
        match side {
            Side::Doc => e.encode(t),
            Side::Query => e.encode_for_query(t),
        }
    };

    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let pv: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| encode(&primary, t))
        .collect::<Result<_>>()?;
    let sv: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| encode(&secondary, t))
        .collect::<Result<_>>()?;

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
    println!("file: {}", args.alignment.display());
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
