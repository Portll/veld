//! Embedder-onboarding hook for alignment.
//!
//! Resolves an alignment for a `(primary_id, secondary_id)` pair at engine
//! construction time:
//!
//! 1. If a file already exists at `resolve_alignment_path`, load it.
//! 2. Else if `VELD_ALIGNMENT_AUTOFIT=1` and a pairs file is present, fit one
//!    against a bundled pairs file, gated by an advisory `.lock` so concurrent
//!    workers don't race.
//! 3. Else return `Ok(None)` — caller proceeds with no alignment and the
//!    existing max-score union path serves cross-space queries.
//!
//! Autofit is opt-in — we deliberately don't block server startup on a slow
//! Procrustes fit unless the operator asked for it.

use anyhow::Result;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::Arc;

use crate::embeddings::{
    read_alignment_file, resolve_alignment_path, save_alignment, Alignment, AlignmentPairId,
    ProcrustesAlignment, RidgeAlignment,
};

/// Returns an alignment for the given pair, fitting if necessary.
pub fn resolve_or_fit(pair_id: &AlignmentPairId) -> Result<Option<Arc<dyn Alignment>>> {
    if let Some(path) = resolve_alignment_path(pair_id) {
        let (header, payload) = read_alignment_file(&path)?;
        if header.pair_id != *pair_id {
            tracing::warn!(
                "alignment file pair_id {:?} != current {:?}; ignoring",
                header.pair_id,
                pair_id
            );
            return Ok(None);
        }
        let alignment: Arc<dyn Alignment> = match header.method.as_str() {
            "orthogonal_procrustes" => Arc::new(ProcrustesAlignment::from_payload(header, &payload)?),
            "ridge" => Arc::new(RidgeAlignment::from_payload(header, &payload)?),
            other => {
                tracing::warn!("unknown alignment method `{}` in {}; ignoring", other, path.display());
                return Ok(None);
            }
        };
        tracing::info!("loaded alignment from {}", path.display());
        return Ok(Some(alignment));
    }

    if std::env::var("VELD_ALIGNMENT_AUTOFIT").as_deref() != Ok("1") {
        tracing::info!(
            "no alignment for {:?}; set VELD_ALIGNMENT_AUTOFIT=1 to auto-fit",
            pair_id
        );
        return Ok(None);
    }

    let pairs_path = std::env::var("VELD_ALIGNMENT_PAIRS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("evaluations/alignment/pairs/pairs.jsonl"));
    if !pairs_path.exists() {
        tracing::warn!(
            "auto-fit requested but pairs file missing: {}",
            pairs_path.display()
        );
        return Ok(None);
    }

    let cache_dir = dirs::home_dir()
        .map(|h| h.join(".cache").join("veld").join("alignments"))
        .unwrap_or_else(|| PathBuf::from(".").join("veld_alignments"));
    std::fs::create_dir_all(&cache_dir).ok();
    let lock_path = cache_dir.join(format!(".{}.fitting.lock", pair_id.cache_filename()));

    // Advisory lock: if another process is already fitting, log and bail out.
    // Best-effort only; not a strong cross-process mutex. We accept a small
    // race window where two workers both pass the create_new check.
    let lock_file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            tracing::info!(
                "another process is fitting alignment {:?} (lock {} present); skipping",
                pair_id,
                lock_path.display()
            );
            return Ok(None);
        }
        Err(e) => {
            tracing::warn!(
                "couldn't acquire alignment fit lock at {}: {}; proceeding anyway",
                lock_path.display(),
                e
            );
            return Ok(None);
        }
    };
    drop(lock_file);

    let result = (|| -> Result<Arc<dyn Alignment>> {
        tracing::info!("auto-fitting alignment for {:?}", pair_id);
        let alignment = fit_from_pairs_file(&pairs_path, pair_id)?;
        let out_path = cache_dir.join(pair_id.cache_filename());
        save_alignment(&out_path, &alignment)?;
        tracing::info!("installed alignment at {}", out_path.display());
        Ok(Arc::new(alignment))
    })();

    // Always remove the lock file on exit, even on error.
    let _ = std::fs::remove_file(&lock_path);

    result.map(Some)
}

fn fit_from_pairs_file(
    pairs_path: &std::path::Path,
    pair_id: &AlignmentPairId,
) -> Result<ProcrustesAlignment> {
    use rayon::prelude::*;
    use serde::Deserialize;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[derive(Deserialize)]
    struct Row {
        text: String,
    }

    let file = File::open(pairs_path)?;
    let rows: Vec<Row> = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();

    let primary = crate::embeddings::load_primary_embedder(&pair_id.primary)?;
    let secondary = crate::embeddings::load_secondary_embedder(&pair_id.secondary)?;

    let pv: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| primary.encode(t))
        .collect::<Result<_>>()?;
    let sv: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| secondary.encode(t))
        .collect::<Result<_>>()?;

    let mut alignment = ProcrustesAlignment::fit(pair_id.clone(), &pv, &sv)?;
    // Self-cosine on training data — a quick sanity gate, not a true held-out
    // eval. The real held-out eval lives in `alignment-fit` / `alignment-eval`.
    let n_check = pv.len().min(200);
    let mut sum = 0.0_f32;
    for i in 0..n_check {
        let proj = alignment.project(&sv[i])?;
        let dot: f32 = proj.iter().zip(pv[i].iter()).map(|(x, y)| x * y).sum();
        sum += dot;
    }
    alignment.set_eval(sum / n_check.max(1) as f32);
    Ok(alignment)
}
