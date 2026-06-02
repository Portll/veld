//! CSLS (Cross-domain Similarity Local Scaling) — hubness correction for cross-space
//! nearest-neighbor retrieval.
//!
//! After a secondary embedding is aligned into the primary space, raw cosine retrieval
//! suffers from "hubs": a few primary vectors that are everyone's nearest neighbor. CSLS
//! penalizes hubs:
//!
//! ```text
//! csls(q, c) = 2·cos(q, c) − r_T(q) − r_S(c)
//! ```
//!
//! where `r_T(q)` is the mean cosine of the query to its `k` nearest candidates and
//! `r_S(c)` is the mean cosine of candidate `c` to its `k` nearest neighbors among the
//! candidate set. `r_S` is query-independent and can be precomputed.
//!
//! This is intended to be applied ONLY to cross-space (aligned) comparisons — it must
//! never perturb same-space scoring.

/// Mean of the `k` largest values in `sims` (all of them if `sims.len() <= k`).
/// Returns 0.0 for an empty slice or `k == 0`.
pub fn topk_mean(sims: &[f32], k: usize) -> f32 {
    if sims.is_empty() || k == 0 {
        return 0.0;
    }
    let mut v: Vec<f32> = sims.to_vec();
    // Descending; NaNs sort last.
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(v.len());
    v.iter().take(take).sum::<f32>() / take as f32
}

/// CSLS-adjust a query's similarity vector against N candidates.
///
/// `query_sims[i]` = cosine(query, candidate_i). `candidate_local_means[i]` = `r_S` for
/// candidate i (its top-k local mean among candidates, see [`local_means`]). `k` is the
/// neighborhood size used for the query term `r_T`.
pub fn csls_adjust(query_sims: &[f32], k: usize, candidate_local_means: &[f32]) -> Vec<f32> {
    let rt = topk_mean(query_sims, k);
    query_sims
        .iter()
        .enumerate()
        .map(|(i, &s)| 2.0 * s - rt - candidate_local_means.get(i).copied().unwrap_or(0.0))
        .collect()
}

/// Precompute `r_S` for every candidate from a candidate×candidate cosine matrix.
/// The diagonal (self-similarity) is excluded from each row's top-k.
pub fn local_means(candidate_sim_matrix: &[Vec<f32>], k: usize) -> Vec<f32> {
    candidate_sim_matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let off_diag: Vec<f32> = row
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .collect();
            topk_mean(&off_diag, k)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_mean_basics() {
        assert_eq!(topk_mean(&[], 3), 0.0);
        assert_eq!(topk_mean(&[0.5, 0.1, 0.9], 0), 0.0);
        assert!((topk_mean(&[0.2, 0.8, 0.4], 2) - 0.6).abs() < 1e-6); // (0.8+0.4)/2
        assert!((topk_mean(&[0.2, 0.8], 5) - 0.5).abs() < 1e-6); // all
    }

    /// A hub candidate (high local mean) is penalized relative to a non-hub with the
    /// same raw similarity.
    #[test]
    fn csls_penalizes_hubs() {
        let query_sims = vec![0.7, 0.7]; // identical raw cosine to both candidates
        let local = vec![0.9, 0.1]; // candidate 0 is a hub, candidate 1 is not
        let adj = csls_adjust(&query_sims, 2, &local);
        assert!(adj[1] > adj[0], "non-hub should outrank hub after CSLS: {adj:?}");
    }

    #[test]
    fn local_means_excludes_self() {
        // candidate 0 perfectly similar to itself (1.0) but 0.0 to others.
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let lm = local_means(&m, 2);
        // self (1.0) excluded ⇒ each local mean is over the two 0.0 off-diagonals.
        assert!(lm.iter().all(|&v| v.abs() < 1e-6), "{lm:?}");
    }
}
