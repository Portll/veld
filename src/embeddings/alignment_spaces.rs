//! Alignment between embedding spaces: a fitted map that lets a vector in one space be searched
//! in another.
//!
//! A map is keyed by `(from, to)` space ids: it takes a vector in `from` to a vector in `to`.
//! On disk the pair is an [`AlignmentPairId`] whose `secondary` is `from` and `primary` is `to`,
//! the direction the original secondary→primary maps used. A corpus keeps its maps under
//! `alignments/` beside its store, and lists them in the space registry.
//!
//! The pairs a map is fitted on come from the corpus itself: every memory holding a vector in
//! both spaces is one pair, so a map needs no curated pairs file and no second embedding pass.
//! Every fifth pair is held out; the held-out mean paired cosine is the map's confidence, the
//! weight recall gives a result reached through it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use super::alignment::{read_alignment_file, Alignment, AlignmentPairId};
use super::alignment_affine::AffineAlignment;
use super::alignment_config::AlignmentConfig;
use super::alignment_csls::{csls_adjust, local_means};
use super::alignment_procrustes::ProcrustesAlignment;
use super::alignment_ridge::RidgeAlignment;
use super::space::SpaceId;

/// Directory beside a tenant's store holding its maps.
pub const ALIGNMENT_DIR: &str = "alignments";
/// Fewer pairs than this fit a map that mostly memorises them.
pub const MIN_PAIRS: usize = 64;
/// One pair in this many is held out of the fit and measures it.
const HOLDOUT_EVERY: usize = 5;
/// The held-out retrieval metrics compare every held-out pair with every other, so they are
/// computed over at most this many.
const MAX_EVAL_PAIRS: usize = 1000;

/// The pair id a map from `from` to `to` is stored under.
pub fn pair_id_between(from: &SpaceId, to: &SpaceId) -> AlignmentPairId {
    AlignmentPairId::new(to.as_str(), from.as_str())
}

/// The `(from, to)` spaces a stored pair id names, when both are space ids.
pub fn spaces_of(pair: &AlignmentPairId) -> Option<(SpaceId, SpaceId)> {
    Some((
        SpaceId::parse(&pair.secondary)?,
        SpaceId::parse(&pair.primary)?,
    ))
}

/// Where a corpus keeps its map from `from` to `to`.
pub fn corpus_alignment_path(storage_path: &Path, from: &SpaceId, to: &SpaceId) -> PathBuf {
    storage_path
        .join(ALIGNMENT_DIR)
        .join(pair_id_between(from, to).cache_filename())
}

/// A map read from `path`, whichever fitter wrote it.
pub fn load_alignment(path: &Path) -> Result<Arc<dyn Alignment>> {
    let (header, payload) = read_alignment_file(path)?;
    let alignment: Arc<dyn Alignment> = match header.method.as_str() {
        "orthogonal_procrustes" => Arc::new(ProcrustesAlignment::from_payload(header, &payload)?),
        "ridge" => Arc::new(RidgeAlignment::from_payload(header, &payload)?),
        "whitened_procrustes" => Arc::new(AffineAlignment::from_payload(header, &payload)?),
        other => bail!("unknown alignment method `{other}` in {}", path.display()),
    };
    Ok(alignment)
}

/// A map's confidence: its held-out mean paired cosine, clamped to `[0, 1]`. A map that never
/// recorded one is trusted fully, as every map was before confidences existed.
pub fn confidence_of(alignment: &dyn Alignment) -> f32 {
    alignment
        .header()
        .eval_paired_cosine_mean
        .map_or(1.0, |c| c.clamp(0.0, 1.0))
}

/// Every map in the corpus whose pair names two spaces: `(from, to, path, map)`. A file that
/// does not load is skipped with a warning, never an error: a map is an optional bridge.
pub fn corpus_alignments(
    storage_path: &Path,
) -> Vec<(SpaceId, SpaceId, PathBuf, Arc<dyn Alignment>)> {
    let Ok(entries) = std::fs::read_dir(storage_path.join(ALIGNMENT_DIR)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    paths.sort();
    for path in paths {
        match load_alignment(&path) {
            Ok(alignment) => match spaces_of(alignment.pair_id()) {
                Some((from, to)) => out.push((from, to, path, alignment)),
                None => tracing::warn!(
                    "{}: pair {:?} names no spaces; ignored",
                    path.display(),
                    alignment.pair_id()
                ),
            },
            Err(e) => tracing::warn!("{}: {e:#}; ignored", path.display()),
        }
    }
    out
}

/// What a fit produced and how it measured on the pairs it did not see.
pub struct BridgeFit {
    pub alignment: AffineAlignment,
    pub report: BridgeReport,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BridgeReport {
    pub from: SpaceId,
    pub to: SpaceId,
    pub method: String,
    pub pairs: usize,
    pub held_out: usize,
    /// Mean cosine between a held-out `from` vector, mapped, and its own `to` vector. The
    /// map's confidence.
    pub held_out_cosine: f32,
    /// Share of held-out mapped vectors whose nearest held-out `to` vector is their own pair.
    pub held_out_recall_at_1: f32,
    /// The same with CSLS hubness correction applied to the comparison.
    pub held_out_csls_recall_at_1: f32,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Fit the whitened-Procrustes map from `from` to `to` on `pairs` of `(from_vector,
/// to_vector)`, holding every fifth pair out to measure it.
pub fn fit_bridge(
    from: &SpaceId,
    to: &SpaceId,
    pairs: &[(Vec<f32>, Vec<f32>)],
    config: &AlignmentConfig,
) -> Result<BridgeFit> {
    if pairs.len() < MIN_PAIRS {
        bail!(
            "{} pairs hold vectors in both {from} and {to}; a map needs at least {MIN_PAIRS}",
            pairs.len()
        );
    }
    let (mut train_to, mut train_from, mut held) = (Vec::new(), Vec::new(), Vec::new());
    for (i, (f, t)) in pairs.iter().enumerate() {
        if f.len() != from.dimension() || t.len() != to.dimension() {
            continue;
        }
        if i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1 {
            held.push((f, t));
        } else {
            train_from.push(f.clone());
            train_to.push(t.clone());
        }
    }
    if train_from.len() < MIN_PAIRS || held.is_empty() {
        bail!(
            "{} usable pairs between {from} and {to}; a map needs at least {MIN_PAIRS} to fit and some to hold out",
            train_from.len() + held.len()
        );
    }
    let mut alignment = AffineAlignment::fit_whitened_procrustes(
        pair_id_between(from, to),
        &train_to,
        &train_from,
        config.whiten,
        config.variance_target,
        config.whiten_eps,
    )
    .with_context(|| format!("fitting the map from {from} to {to}"))?;

    held.truncate(MAX_EVAL_PAIRS);
    let mapped: Vec<Vec<f32>> = held
        .iter()
        .map(|(f, _)| alignment.project(f))
        .collect::<Result<_>>()?;
    let targets: Vec<&Vec<f32>> = held.iter().map(|(_, t)| *t).collect();
    let held_out_cosine = mapped
        .iter()
        .zip(&targets)
        .map(|(m, t)| cosine(m, t))
        .sum::<f32>()
        / held.len() as f32;

    let sims: Vec<Vec<f32>> = mapped
        .iter()
        .map(|m| targets.iter().map(|t| cosine(m, t)).collect())
        .collect();
    let target_sims: Vec<Vec<f32>> = targets
        .iter()
        .map(|a| targets.iter().map(|b| cosine(a, b)).collect())
        .collect();
    let k = config.csls_neighborhood_k.max(1) as usize;
    let target_means = local_means(&target_sims, k);
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    let n = held.len() as f32;
    let recall_at_1 = sims
        .iter()
        .enumerate()
        .filter(|(i, row)| argmax(row) == Some(*i))
        .count() as f32
        / n;
    let csls_recall_at_1 = sims
        .iter()
        .enumerate()
        .filter(|(i, row)| argmax(&csls_adjust(row, k, &target_means)) == Some(*i))
        .count() as f32
        / n;

    alignment.set_eval(held_out_cosine);
    Ok(BridgeFit {
        report: BridgeReport {
            from: from.clone(),
            to: to.clone(),
            method: alignment.method().to_string(),
            pairs: pairs.len(),
            held_out: held.len(),
            held_out_cosine,
            held_out_recall_at_1: recall_at_1,
            held_out_csls_recall_at_1: csls_recall_at_1,
        },
        alignment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    /// `to` vectors are a fixed linear image of the `from` vectors plus a little noise: the
    /// shape two embedders of one corpus have, and a map should recover it.
    fn related_pairs(n: usize, from_dim: usize, to_dim: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut rng = StdRng::seed_from_u64(7);
        let w: Vec<Vec<f32>> = (0..to_dim)
            .map(|_| (0..from_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        (0..n)
            .map(|_| {
                let f = unit((0..from_dim).map(|_| rng.gen_range(-1.0..1.0)).collect());
                let t = unit(
                    w.iter()
                        .map(|row| {
                            row.iter().zip(&f).map(|(a, b)| a * b).sum::<f32>()
                                + rng.gen_range(-0.01..0.01)
                        })
                        .collect(),
                );
                (f, t)
            })
            .collect()
    }

    #[test]
    fn a_pair_id_names_its_spaces_both_ways() {
        let from = SpaceId::new("minilm-l6-v2", 384);
        let to = SpaceId::new("nomic-embed-text-v1.5", 768);
        let pair = pair_id_between(&from, &to);
        assert_eq!(spaces_of(&pair), Some((from, to)));
        assert_eq!(spaces_of(&AlignmentPairId::new("nomic", "minilm")), None);
    }

    #[test]
    fn a_map_fitted_on_related_pairs_finds_each_pair_and_reloads() {
        let from = SpaceId::new("model-from", 16);
        let to = SpaceId::new("model-to", 24);
        let pairs = related_pairs(400, 16, 24);
        let fit = fit_bridge(&from, &to, &pairs, &AlignmentConfig::default()).unwrap();
        assert_eq!(fit.report.held_out, 80);
        assert!(
            fit.report.held_out_cosine > 0.9,
            "held-out cosine {}",
            fit.report.held_out_cosine
        );
        assert!(fit.report.held_out_recall_at_1 > 0.8, "{:?}", fit.report);

        let dir = tempfile::tempdir().unwrap();
        let path = corpus_alignment_path(dir.path(), &from, &to);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        super::super::alignment::save_alignment(&path, &fit.alignment).unwrap();
        let listed = corpus_alignments(dir.path());
        assert_eq!(listed.len(), 1);
        assert_eq!((&listed[0].0, &listed[0].1), (&from, &to));
        let confidence = confidence_of(listed[0].3.as_ref());
        assert!((confidence - fit.report.held_out_cosine).abs() < 1e-5);
    }

    #[test]
    fn too_few_pairs_fit_nothing() {
        let from = SpaceId::new("model-from", 16);
        let to = SpaceId::new("model-to", 24);
        assert!(fit_bridge(
            &from,
            &to,
            &related_pairs(20, 16, 24),
            &AlignmentConfig::default()
        )
        .is_err());
    }
}
