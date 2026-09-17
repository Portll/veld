//! Affine alignment: `project(x) = renorm(weight·x + bias)`.
//!
//! This is the runtime form of the **whitened-Procrustes** pipeline: condition both
//! spaces (center + whiten + variance-rank), solve the orthogonal Procrustes rotation in
//! the whitened subspace, then fold conditioning-secondary → rotation → de-conditioning-
//! primary into a single affine map expressed in the primary's *original* coordinates.
//! Output is L2-renormalized to match the unit-norm primary index.
//!
//! Because the output is renormalized, an isotropic scale factor is washed out — the
//! `isotropic_scale` flag is therefore inert on this (renormalized) path and retained
//! only for non-renormalized consumers.

use anyhow::{anyhow, bail, Result};
use nalgebra::{DMatrix, DVector};

use super::alignment::{unix_ts_now, Alignment, AlignmentHeader, AlignmentPairId};
use super::alignment_conditioning::Conditioning;
use super::alignment_config::WhitenMode;

pub struct AffineAlignment {
    pair_id: AlignmentPairId,
    weight: DMatrix<f32>, // out_dim × in_dim
    bias: DVector<f32>,   // out_dim
    in_dim: usize,
    out_dim: usize,
    method_name: &'static str,
    fit_unix_ts: i64,
    eval_paired_cosine_mean: Option<f32>,
}

fn l2_renormalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl AffineAlignment {
    /// Fit the whitened-Procrustes affine map. `primary[i]`/`secondary[i]` are paired
    /// (same source text). Spaces may differ in dimension — conditioning truncates both
    /// to a shared rank before the rotation.
    pub fn fit_whitened_procrustes(
        pair_id: AlignmentPairId,
        primary: &[Vec<f32>],
        secondary: &[Vec<f32>],
        whiten: WhitenMode,
        variance_target: f32,
        eps: f32,
    ) -> Result<Self> {
        if primary.len() != secondary.len() {
            bail!(
                "primary/secondary length mismatch: {} vs {}",
                primary.len(),
                secondary.len()
            );
        }
        if primary.is_empty() {
            bail!("cannot fit on empty pairs");
        }
        let dp = primary[0].len();
        let ds = secondary[0].len();
        if dp == 0 || ds == 0 {
            bail!("zero-dimension embeddings: d_p={dp}, d_s={ds}");
        }

        let whiten = if whiten == WhitenMode::Off {
            // Procrustes needs a whitened/orthonormal basis to be well-conditioned; fall
            // back to PCA so the shared-rank rotation is still defined.
            WhitenMode::Pca
        } else {
            whiten
        };

        let cond_p = Conditioning::fit(primary, whiten, variance_target, eps, 1.0)?;
        let cond_s = Conditioning::fit(secondary, whiten, variance_target, eps, 1.0)?;
        let k = cond_p.out_dim().min(cond_s.out_dim());
        if k == 0 {
            bail!("conditioning produced zero-rank subspace");
        }

        let n = primary.len();
        let pw = cond_p.apply_rows(primary)?; // each cond_p.out_dim()
        let sw = cond_s.apply_rows(secondary)?; // each cond_s.out_dim()

        // Truncate whitened samples to the shared rank k.
        let pmat = DMatrix::<f32>::from_fn(n, k, |i, j| pw[i][j]); // n × k
        let smat = DMatrix::<f32>::from_fn(n, k, |i, j| sw[i][j]); // n × k

        // Orthogonal Procrustes mapping whitened-secondary → whitened-primary (row form
        // P ≈ S·Rt). M = Sᵀ P; SVD M = U Σ Vᵀ; Rt = U Vᵀ. Column-convention rotation
        // (p' = R_col·s') is Rtᵀ = V Uᵀ.
        let m = smat.transpose() * &pmat; // k × k
        let svd = m.svd(true, true);
        let u = svd.u.ok_or_else(|| anyhow!("SVD U missing"))?;
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD Vᵀ missing"))?;
        let rt = &u * &v_t; // k × k
        let r_col = rt.transpose(); // k × k, p' = r_col · s'

        // Fold to original primary coordinates:
        //   x_p = μ_p + inv_p_k · r_col · fwd_s_k · (x_s − μ_s)
        // inv_p_k: first k columns of cond_p.inverse (d_p × k)
        // fwd_s_k: first k rows of cond_s.forward (k × d_s)
        let inv_p_k = DMatrix::<f32>::from_fn(dp, k, |i, j| cond_p.inverse[(i, j)]);
        let fwd_s_k = DMatrix::<f32>::from_fn(k, ds, |i, j| cond_s.forward[(i, j)]);

        let weight = &inv_p_k * &r_col * &fwd_s_k; // d_p × d_s
        let bias = &cond_p.mean - &weight * &cond_s.mean; // d_p

        // M2: refuse to persist a non-finite map (defense-in-depth behind the eps floor in
        // Conditioning) — a NaN/Inf weight would otherwise silently serve NaN-similarity
        // vectors at recall.
        if weight.iter().any(|v| !v.is_finite()) || bias.iter().any(|v| !v.is_finite()) {
            bail!(
                "whitened-Procrustes produced a non-finite map (degenerate conditioning); \
                 raise whiten_eps or variance_target, or supply more/cleaner pairs"
            );
        }

        Ok(Self {
            pair_id,
            weight,
            bias,
            in_dim: ds,
            out_dim: dp,
            method_name: "whitened_procrustes",
            fit_unix_ts: unix_ts_now(),
            eval_paired_cosine_mean: None,
        })
    }

    pub fn set_eval(&mut self, paired_cosine_mean: f32) {
        self.eval_paired_cosine_mean = Some(paired_cosine_mean);
    }

    /// Payload layout: `[weight: out×in f32 LE row-major][bias: out f32 LE]`.
    pub fn from_payload(header: AlignmentHeader, payload: &[u8]) -> Result<Self> {
        let (out, inn) = (header.out_dim, header.in_dim);
        let expected = (out * inn + out) * 4;
        if payload.len() != expected {
            bail!(
                "affine payload size mismatch: expected {} bytes, got {}",
                expected,
                payload.len()
            );
        }
        let floats: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let weight = DMatrix::<f32>::from_row_slice(out, inn, &floats[..out * inn]);
        let bias = DVector::<f32>::from_row_slice(&floats[out * inn..]);
        Ok(Self {
            pair_id: header.pair_id,
            weight,
            bias,
            in_dim: inn,
            out_dim: out,
            method_name: "whitened_procrustes",
            fit_unix_ts: header.fit_unix_ts,
            eval_paired_cosine_mean: header.eval_paired_cosine_mean,
        })
    }
}

impl Alignment for AffineAlignment {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>> {
        if secondary.len() != self.in_dim {
            bail!(
                "affine project: expected in_dim={}, got {}",
                self.in_dim,
                secondary.len()
            );
        }
        let x = DVector::<f32>::from_row_slice(secondary);
        let y = &self.weight * x + &self.bias;
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
        self.method_name
    }
    fn header(&self) -> AlignmentHeader {
        AlignmentHeader {
            method: self.method_name.to_string(),
            pair_id: self.pair_id.clone(),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            fit_unix_ts: self.fit_unix_ts,
            eval_paired_cosine_mean: self.eval_paired_cosine_mean,
        }
    }
    fn payload_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity((self.in_dim * self.out_dim + self.out_dim) * 4);
        for r in 0..self.out_dim {
            for c in 0..self.in_dim {
                out.extend_from_slice(&self.weight[(r, c)].to_le_bytes());
            }
        }
        for r in 0..self.out_dim {
            out.extend_from_slice(&self.bias[r].to_le_bytes());
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
        AlignmentPairId::new("primary-4", "secondary-4")
    }

    fn synthetic(dim: usize, seed: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        for (j, x) in v.iter_mut().enumerate() {
            *x = ((seed * 7 + j * 3) as f32 * 0.17).sin() + ((seed * 5 + j) as f32 * 0.09).cos();
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
        v
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Secondary = a fixed coordinate permutation of primary (an orthogonal map). The
    /// fitted alignment should recover it, so projected-secondary ≈ primary.
    #[test]
    fn recovers_orthogonal_relationship() {
        let n = 60;
        let d = 4;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(d, i)).collect();
        // permutation [1,2,3,0]
        let secondary: Vec<Vec<f32>> = primary
            .iter()
            .map(|p| vec![p[1], p[2], p[3], p[0]])
            .collect();

        let a = AffineAlignment::fit_whitened_procrustes(
            pid(),
            &primary,
            &secondary,
            WhitenMode::Pca,
            1.0,
            1e-5,
        )
        .unwrap();

        let mut total = 0.0;
        for (p, s) in primary.iter().zip(secondary.iter()) {
            let proj = a.project(s).unwrap();
            // unit-norm output
            let nrm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-4, "projection not unit norm: {nrm}");
            total += cos(&proj, p);
        }
        let mean_cos = total / n as f32;
        assert!(mean_cos > 0.85, "weak recovery: mean cosine {mean_cos}");
    }

    #[test]
    fn save_load_round_trip() {
        let n = 40;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(4, i)).collect();
        let secondary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(3, i + 100)).collect();
        let a = AffineAlignment::fit_whitened_procrustes(
            AlignmentPairId::new("p4", "s3"),
            &primary,
            &secondary,
            WhitenMode::Pca,
            0.95,
            1e-4,
        )
        .unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("affine.bin");
        save_alignment(&path, &a).unwrap();
        let (header, payload) = read_alignment_file(&path).unwrap();
        assert_eq!(header.method, "whitened_procrustes");
        let b = AffineAlignment::from_payload(header, &payload).unwrap();

        let pa = a.project(&secondary[0]).unwrap();
        let pb = b.project(&secondary[0]).unwrap();
        for (x, y) in pa.iter().zip(pb.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn cross_dimensional_projects_to_primary_dim() {
        let n = 50;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(6, i)).collect();
        let secondary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(3, i + 7)).collect();
        let a = AffineAlignment::fit_whitened_procrustes(
            AlignmentPairId::new("p6", "s3"),
            &primary,
            &secondary,
            WhitenMode::Pca,
            0.95,
            1e-4,
        )
        .unwrap();
        assert_eq!(a.in_dim(), 3);
        assert_eq!(a.out_dim(), 6);
        let p = a.project(&secondary[0]).unwrap();
        assert_eq!(p.len(), 6);
        let nrm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-4);
    }

    /// Degenerate (zero-variance) secondary must not produce a non-finite map: the eps
    /// floor (M1) keeps whitening finite and the NaN/Inf gate (M2) backs it up.
    #[test]
    fn degenerate_input_stays_finite() {
        let n = 30;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(4, i)).collect();
        // every secondary row identical ⇒ zero covariance
        let secondary: Vec<Vec<f32>> = (0..n).map(|_| synthetic(4, 0)).collect();
        let a = AffineAlignment::fit_whitened_procrustes(
            AlignmentPairId::new("p4", "s4"),
            &primary,
            &secondary,
            WhitenMode::Pca,
            1.0,
            1e-4,
        )
        .expect("eps floor should keep the fit finite");
        let p = a.project(&secondary[0]).unwrap();
        assert!(
            p.iter().all(|v| v.is_finite()),
            "non-finite projection: {p:?}"
        );
    }
}
