//! Orthogonal Procrustes alignment.
//!
//! Solves `R* = argmin_R ||A - B R||_F` subject to `R^T R = I` (partial
//! isometry when `d_s != d_p`), via SVD of `M = B^T A`. Closed-form, no
//! hyperparameters.
//!
//! Outputs are L2-renormalized after projection. Pure orthogonal rotation
//! preserves norm only when `d_s == d_p`; in the partial-isometry case
//! `|R x|` is bounded above by `|x|` but not equal, which would dampen
//! cosine scores against the unit-norm primary Vamana index.

use anyhow::{anyhow, bail, Result};
use nalgebra::{DMatrix, DVector};

use super::alignment::{unix_ts_now, Alignment, AlignmentHeader, AlignmentPairId};

pub struct ProcrustesAlignment {
    pair_id: AlignmentPairId,
    in_dim: usize,
    out_dim: usize,
    /// Shape (out_dim × in_dim) — project = rotation * secondary_vec.
    rotation: DMatrix<f32>,
    fit_unix_ts: i64,
    eval_paired_cosine_mean: Option<f32>,
}

impl ProcrustesAlignment {
    /// `primary[i]` and `secondary[i]` must be paired (same source text).
    /// Both sides must be L2-normalized (norm within [0.99, 1.01]).
    pub fn fit(
        pair_id: AlignmentPairId,
        primary: &[Vec<f32>],
        secondary: &[Vec<f32>],
    ) -> Result<Self> {
        if primary.len() != secondary.len() {
            bail!(
                "primary/secondary length mismatch: {} vs {}",
                primary.len(),
                secondary.len()
            );
        }
        if primary.is_empty() {
            bail!("cannot fit Procrustes on empty pairs");
        }
        let n = primary.len();
        let d_p = primary[0].len();
        let d_s = secondary[0].len();
        if d_p == 0 || d_s == 0 {
            bail!("zero-dimension embeddings: d_p={d_p}, d_s={d_s}");
        }
        for (i, v) in primary.iter().enumerate() {
            if v.len() != d_p {
                bail!("primary row {i}: dim {} != {d_p}", v.len());
            }
            assert_normalized(v, "primary", i)?;
        }
        for (i, v) in secondary.iter().enumerate() {
            if v.len() != d_s {
                bail!("secondary row {i}: dim {} != {d_s}", v.len());
            }
            assert_normalized(v, "secondary", i)?;
        }

        // A: n × d_p, B: n × d_s
        let a = DMatrix::<f32>::from_row_iterator(n, d_p, primary.iter().flatten().copied());
        let b = DMatrix::<f32>::from_row_iterator(n, d_s, secondary.iter().flatten().copied());

        // M = B^T A → (d_s × d_p)
        let m = b.transpose() * &a;

        let svd = m.svd(true, true);
        let u = svd.u.ok_or_else(|| anyhow!("SVD U missing"))?; // d_s × k
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD V^T missing"))?; // k × d_p

        let q = u * v_t; // d_s × d_p, optimal projection
        let rotation = q.transpose(); // d_p × d_s, for project = rotation · s

        Ok(Self {
            pair_id,
            in_dim: d_s,
            out_dim: d_p,
            rotation,
            fit_unix_ts: unix_ts_now(),
            eval_paired_cosine_mean: None,
        })
    }

    pub fn set_eval(&mut self, paired_cosine_mean: f32) {
        self.eval_paired_cosine_mean = Some(paired_cosine_mean);
    }

    /// Reconstruct from persisted payload bytes (row-major f32 LE).
    pub fn from_payload(header: AlignmentHeader, payload: &[u8]) -> Result<Self> {
        let expected = header.in_dim * header.out_dim * 4;
        if payload.len() != expected {
            bail!(
                "procrustes payload size mismatch: expected {} bytes ({}×{}×4), got {}",
                expected,
                header.out_dim,
                header.in_dim,
                payload.len()
            );
        }
        let floats: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let rotation = DMatrix::<f32>::from_row_slice(header.out_dim, header.in_dim, &floats);
        Ok(Self {
            pair_id: header.pair_id,
            in_dim: header.in_dim,
            out_dim: header.out_dim,
            rotation,
            fit_unix_ts: header.fit_unix_ts,
            eval_paired_cosine_mean: header.eval_paired_cosine_mean,
        })
    }
}

fn assert_normalized(v: &[f32], side: &str, idx: usize) -> Result<()> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 0.01 {
        bail!(
            "{} vector {} not L2-normalized (norm={:.4})",
            side,
            idx,
            norm
        );
    }
    Ok(())
}

fn l2_renormalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl Alignment for ProcrustesAlignment {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>> {
        if secondary.len() != self.in_dim {
            bail!(
                "procrustes project: expected in_dim={}, got {}",
                self.in_dim,
                secondary.len()
            );
        }
        let x = DVector::<f32>::from_row_slice(secondary);
        let y = &self.rotation * x;
        let mut out: Vec<f32> = y.iter().copied().collect();
        l2_renormalize(&mut out);
        Ok(out)
    }

    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.out_dim
    }
    fn pair_id(&self) -> &AlignmentPairId {
        &self.pair_id
    }
    fn method(&self) -> &'static str {
        "orthogonal_procrustes"
    }

    fn header(&self) -> AlignmentHeader {
        AlignmentHeader {
            method: self.method().to_string(),
            pair_id: self.pair_id.clone(),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            fit_unix_ts: self.fit_unix_ts,
            eval_paired_cosine_mean: self.eval_paired_cosine_mean,
        }
    }

    fn payload_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.in_dim * self.out_dim * 4);
        for r in 0..self.out_dim {
            for c in 0..self.in_dim {
                out.extend_from_slice(&self.rotation[(r, c)].to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::alignment::{read_alignment_file, save_alignment};
    use tempfile::tempdir;

    fn pid() -> AlignmentPairId {
        AlignmentPairId::new("modelA-768", "modelB-768")
    }

    fn norm(v: &mut [f32]) {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    /// Deterministic gaussian-ish vector for tests.
    fn synthetic(dim: usize, seed: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        for (j, x) in v.iter_mut().enumerate() {
            *x = ((seed * 7 + j) as f32 * 0.13).sin() + ((seed * 5 + j * 3) as f32 * 0.07).cos();
        }
        norm(&mut v);
        v
    }

    /// When secondary == primary, Procrustes should recover the identity rotation.
    #[test]
    fn fits_identity_when_pairs_match() {
        let mut rng = vec![
            vec![0.4_f32, 0.3, 0.7, 0.5],
            vec![-0.2, 0.9, 0.1, 0.3],
            vec![0.5, -0.4, 0.6, 0.5],
            vec![0.1, 0.1, 0.9, 0.4],
            vec![0.7, -0.3, 0.5, 0.4],
        ];
        for v in rng.iter_mut() {
            norm(v);
        }
        let primary = rng.clone();
        let secondary = rng.clone();

        let a = ProcrustesAlignment::fit(pid(), &primary, &secondary).unwrap();
        let projected = a.project(&secondary[0]).unwrap();
        let cos: f32 = projected
            .iter()
            .zip(primary[0].iter())
            .map(|(x, y)| x * y)
            .sum();
        assert!(cos > 0.99, "expected near-1 cosine, got {cos}");
    }

    #[test]
    fn save_load_round_trip() {
        let primary: Vec<Vec<f32>> = (0..20).map(|i| synthetic(8, i)).collect();
        let mut perturbed = primary.clone();
        for v in perturbed.iter_mut() {
            v[0] += 0.05;
            norm(v);
        }
        let secondary = perturbed;
        let a = ProcrustesAlignment::fit(pid(), &primary, &secondary).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("p.bin");
        save_alignment(&path, &a).unwrap();
        let (header, payload) = read_alignment_file(&path).unwrap();
        let b = ProcrustesAlignment::from_payload(header, &payload).unwrap();

        let pa = a.project(&secondary[0]).unwrap();
        let pb = b.project(&secondary[0]).unwrap();
        for (x, y) in pa.iter().zip(pb.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_unnormalized_input() {
        let primary = vec![vec![10.0_f32, 0.0, 0.0]];
        let secondary = vec![vec![1.0_f32, 0.0, 0.0]];
        assert!(ProcrustesAlignment::fit(pid(), &primary, &secondary).is_err());
    }

    /// F12: mismatched-length pair vectors must be rejected.
    #[test]
    fn rejects_mismatched_lengths() {
        let primary: Vec<Vec<f32>> = (0..3).map(|i| synthetic(4, i)).collect();
        let secondary: Vec<Vec<f32>> = (0..5).map(|i| synthetic(4, i)).collect();
        assert!(ProcrustesAlignment::fit(pid(), &primary, &secondary).is_err());
    }

    /// F12: zero-dim embeddings must be rejected.
    #[test]
    fn rejects_zero_dim() {
        let primary = vec![vec![]];
        let secondary = vec![vec![]];
        assert!(ProcrustesAlignment::fit(pid(), &primary, &secondary).is_err());
    }

    /// F12: dimension-mismatched case (e.g., Nomic 768 ↔ MiniLM 384) must produce
    /// unit-norm projected vectors so Vamana cosine math doesn't dampen scores.
    #[test]
    fn projection_is_unit_norm_when_dims_differ() {
        let n = 30;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(16, i)).collect();
        let secondary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(8, i + 100)).collect();
        let a = ProcrustesAlignment::fit(pid(), &primary, &secondary).unwrap();
        assert_eq!(a.in_dim(), 8);
        assert_eq!(a.out_dim(), 16);

        for s in secondary.iter().take(5) {
            let p = a.project(s).unwrap();
            let pn: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (pn - 1.0).abs() < 1e-4,
                "projected vector not unit norm: {pn}"
            );
        }
    }
}
