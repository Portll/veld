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
    Embedder, ProcrustesAlignment, RidgeAlignment,
};

/// Returns an alignment for the given pair, fitting if necessary.
///
/// Loads the secondary embedder by id when auto-fitting. Prefer
/// [`resolve_or_fit_with_secondary`] from the live engine so that an HTTP /
/// runtime-only secondary (whose `model_id()` is the canonical model name, not
/// a loadable id/URL) does not get re-loaded by id — which would fail with
/// "unknown embedder id".
pub fn resolve_or_fit(pair_id: &AlignmentPairId) -> Result<Option<Arc<dyn Alignment>>> {
    resolve_or_fit_inner(pair_id, None)
}

/// Like [`resolve_or_fit`], but reuses an already-constructed *live* secondary
/// `Arc<dyn Embedder>` for the auto-fit encode pass instead of re-loading it by
/// id. This is the canonical entry point from the retrieval engine: the engine
/// already holds the exact secondary instance that produced the ingest-time
/// vectors, and `pair_id.secondary` is `secondary.model_id()` (which may not be
/// a loadable id). Aligning the SAME vectors the secondary produces at ingest
/// keeps the projection consistent with what lands in the index.
pub fn resolve_or_fit_with_secondary(
    pair_id: &AlignmentPairId,
    secondary: Arc<dyn Embedder>,
) -> Result<Option<Arc<dyn Alignment>>> {
    resolve_or_fit_inner(pair_id, Some(secondary))
}

fn resolve_or_fit_inner(
    pair_id: &AlignmentPairId,
    live_secondary: Option<Arc<dyn Embedder>>,
) -> Result<Option<Arc<dyn Alignment>>> {
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
        let alignment = fit_from_pairs_file(&pairs_path, pair_id, live_secondary.clone())?;
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
    live_secondary: Option<Arc<dyn Embedder>>,
) -> Result<ProcrustesAlignment> {
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
    let texts: Vec<String> = rows.into_iter().map(|r| r.text).collect();

    let primary = crate::embeddings::load_primary_embedder(&pair_id.primary)?;
    // Prefer the live secondary instance handed in by the engine: its
    // `model_id()` is what populated `pair_id.secondary`, and for an HTTP /
    // runtime-only secondary that id is NOT a loadable key — re-loading by id
    // here is exactly the "unknown embedder id" autofit failure (Defect 1). We
    // also want to align the SAME vectors the live secondary produces at ingest.
    // Fall back to load-by-id only when no live instance is available (e.g. the
    // standalone `resolve_or_fit` path / tests with a loadable secondary id).
    let secondary = match live_secondary {
        Some(s) => s,
        None => crate::embeddings::load_secondary_embedder(&pair_id.secondary)?,
    };

    fit_with_embedders(pair_id, &texts, &primary, &secondary)
}

/// Encode `texts` with both embedders and fit a Procrustes projection from the
/// secondary space into the primary space. Split out from `fit_from_pairs_file`
/// so the live-secondary autofit path (Defect 1) is unit-testable with stub
/// embedders — no model files / network required, and the secondary is
/// guaranteed to be the *exact* instance handed in (never a re-load by id).
fn fit_with_embedders(
    pair_id: &AlignmentPairId,
    texts: &[String],
    primary: &Arc<dyn Embedder>,
    secondary: &Arc<dyn Embedder>,
) -> Result<ProcrustesAlignment> {
    use rayon::prelude::*;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::Embedder;
    use std::io::Write;

    /// Deterministic stub embedder whose `model_id()` is a NON-loadable string
    /// (no `nomic`/`minilm`/`http` prefix) — exactly the shape of an HTTP /
    /// runtime-only secondary's canonical model id. If autofit tried to re-load
    /// this by id (the old Defect-1 path) it would hit "unknown embedder id".
    struct StubEmbedder {
        dim: usize,
        id: &'static str,
    }

    impl Embedder for StubEmbedder {
        fn id(&self) -> &str { "stub-onboarding" }
        fn encode(&self, text: &str) -> Result<Vec<f32>> {
            let seed = text.len() as f32 + 1.0;
            let mut v = vec![0.0f32; self.dim];
            for (i, val) in v.iter_mut().enumerate() {
                *val = ((seed + i as f32) * 0.37).sin() + 0.001 * i as f32;
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in v.iter_mut() {
                    *val /= norm;
                }
            }
            Ok(v)
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            self.id
        }
    }

    /// Defect 1: auto-fit with a pairs file + a LIVE secondary whose `model_id()`
    /// is not a loadable id resolves WITHOUT "unknown embedder id" and fits a
    /// real (>0 pairs, correct dims) projection. The secondary used is the exact
    /// instance handed in — never a re-load by id.
    #[test]
    fn autofit_with_live_secondary_fits_without_unknown_embedder_id() {
        // pairs file with several distinct texts
        let mut path = std::env::temp_dir();
        path.push(format!("veld_align_pairs_{}.jsonl", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            for t in [
                "the cat sat on the mat",
                "a dog barked loudly at noon",
                "quantum fields permeate spacetime",
                "she bought fresh bread this morning",
                "the river flows toward the distant sea",
                "alignment maps one space onto another",
            ] {
                writeln!(f, "{{\"text\":\"{t}\"}}").unwrap();
            }
        }

        // Canonical secondary id matches the live embedder's model_id() and is
        // intentionally NOT loadable by load_embedder_by_id.
        let pair_id = AlignmentPairId::new("primary-stub-model", "text-embedding-nomic-embed-text-v1.5");
        let primary: Arc<dyn Embedder> = Arc::new(StubEmbedder { dim: 8, id: "primary-stub-model" });
        let secondary: Arc<dyn Embedder> = Arc::new(StubEmbedder {
            dim: 16,
            id: "text-embedding-nomic-embed-text-v1.5",
        });

        // Read texts the same way the production path does, then fit with the
        // exact embedders — this is the core the live-secondary autofit drives.
        let texts: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .ok()
                    .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(String::from))
            })
            .collect();
        assert_eq!(texts.len(), 6, "all pair rows parsed");

        let result = fit_with_embedders(&pair_id, &texts, &primary, &secondary);
        let _ = std::fs::remove_file(&path);

        let alignment = result.expect("live-secondary autofit must succeed (no 'unknown embedder id')");
        assert_eq!(alignment.in_dim(), 16, "in_dim = secondary dim");
        assert_eq!(alignment.out_dim(), 8, "out_dim = primary dim");
        // Projection actually works end-to-end (768→primary-space shape).
        let probe = secondary.encode("a brand new probe sentence").unwrap();
        let projected = alignment.project(&probe).unwrap();
        assert_eq!(projected.len(), 8, "projects secondary vec into primary space");
    }

    /// Guard the regression: feeding a non-loadable secondary id to the
    /// load-by-id path is what used to break autofit. Confirms the failure mode
    /// the live-secondary fix routes around.
    #[test]
    fn load_secondary_by_nonloadable_id_is_unknown_embedder() {
        let result =
            crate::embeddings::load_secondary_embedder("text-embedding-nomic-embed-text-v1.5");
        match result {
            Ok(_) => panic!("non-loadable model id unexpectedly loaded"),
            Err(err) => assert!(
                err.to_string().contains("unknown embedder id"),
                "non-loadable model id must fail by id: {err}"
            ),
        }
    }
}
