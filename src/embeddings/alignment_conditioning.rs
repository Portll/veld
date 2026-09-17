//! Linear conditioning for embedding alignment: mean-centering, whitening (ZCA/PCA),
//! variance-target rank truncation, and isotropic scaling.
//!
//! Preconditioning both embedder spaces before fitting a linear map (the Artetxe-style
//! pipeline) is the dominant lever for cross-dimensional alignment quality: it removes
//! the anisotropy that wrecks the conditioning of a rectangular map, and the variance
//! truncation gives a principled shared subspace when the two spaces differ in rank.
//!
//! Implemented via the SVD of the centered data matrix (the right singular vectors are
//! the covariance eigenvectors; singular values² / (n−1) are the eigenvalues), so it
//! relies only on `nalgebra`'s SVD.

use anyhow::{anyhow, bail, Result};
use nalgebra::{DMatrix, DVector};

use super::alignment_config::WhitenMode;

/// A fitted conditioning transform for ONE embedder space:
/// `conditioned(x) = scale · forward · (x − mean)`. `inverse` is the scale-free
/// pseudo-inverse of `forward`, used to map a conditioned vector back to original space.
#[derive(Debug, Clone)]
pub struct Conditioning {
    pub mean: DVector<f32>,    // d
    pub forward: DMatrix<f32>, // out × d
    pub inverse: DMatrix<f32>, // d × out
    pub scale: f32,
}

impl Conditioning {
    pub fn in_dim(&self) -> usize {
        self.forward.ncols()
    }
    pub fn out_dim(&self) -> usize {
        self.forward.nrows()
    }

    /// Fit conditioning from rows of `data` (each a d-vector). `variance_target` in
    /// (0,1] keeps the top eigen-directions capturing that fraction of total variance;
    /// `eps` regularizes eigenvalues; `scale` is the isotropic output scale.
    pub fn fit(
        data: &[Vec<f32>],
        whiten: WhitenMode,
        variance_target: f32,
        eps: f32,
        scale: f32,
    ) -> Result<Self> {
        if data.is_empty() {
            bail!("cannot fit conditioning on empty data");
        }
        let n = data.len();
        let d = data[0].len();
        if d == 0 {
            bail!("zero-dimension conditioning");
        }
        for (i, r) in data.iter().enumerate() {
            if r.len() != d {
                bail!("conditioning row {i}: dim {} != {d}", r.len());
            }
        }

        // M1: a too-small eps makes 1/√(λ+eps) explode on near-zero eigenvalues
        // (NaN/Inf into the fitted map); floor it. A thin sample makes the covariance
        // estimate unreliable — warn and degrade rather than silently mis-whiten.
        let eps = eps.max(1e-3);
        if n < 2 * d {
            tracing::warn!(
                samples = n,
                dim = d,
                "alignment conditioning fit on a thin sample (n < 2·d): covariance estimate is unreliable, whitening may be poorly conditioned"
            );
        }

        let mut mean = DVector::<f32>::zeros(d);
        for r in data {
            for (j, &v) in r.iter().enumerate() {
                mean[j] += v;
            }
        }
        mean /= n as f32;

        if whiten == WhitenMode::Off {
            let id = DMatrix::<f32>::identity(d, d);
            return Ok(Self {
                mean,
                forward: id.clone(),
                inverse: id,
                scale,
            });
        }

        let xc = DMatrix::<f32>::from_fn(n, d, |i, j| data[i][j] - mean[j]);
        let svd = xc.svd(true, true);
        let vt = svd
            .v_t
            .ok_or_else(|| anyhow!("SVD v_t missing in conditioning fit"))?; // r × d
        let sv = &svd.singular_values; // r (sorted desc)
        let r = sv.len();
        let denom = n.saturating_sub(1).max(1) as f32;
        let lambdas: Vec<f32> = (0..r).map(|i| (sv[i] * sv[i]) / denom).collect();
        let total: f32 = lambdas.iter().sum::<f32>().max(f32::EPSILON);

        let vt_target = variance_target.clamp(0.0, 1.0);
        let mut acc = 0.0f32;
        let mut k = 0usize;
        for &l in &lambdas {
            acc += l;
            k += 1;
            if acc / total >= vt_target {
                break;
            }
        }
        let k = k.max(1).min(r);

        // PCA-whitening forward (k × d): row i = (1/√(λ_i+eps))·v_i
        let wpca = DMatrix::<f32>::from_fn(k, d, |i, j| {
            let inv_sqrt = 1.0 / (lambdas[i] + eps).sqrt();
            inv_sqrt * vt[(i, j)]
        });
        // PCA-whitening inverse (d × k): col i = √(λ_i)·v_i
        let wpca_inv = DMatrix::<f32>::from_fn(d, k, |j, i| lambdas[i].sqrt() * vt[(i, j)]);

        let (forward, inverse) = match whiten {
            WhitenMode::Pca => (wpca, wpca_inv),
            WhitenMode::Zca => {
                // Rotate back into the original basis: ZCA = V_k · W_pca (d × d).
                let vk = DMatrix::<f32>::from_fn(d, k, |j, i| vt[(i, j)]); // d × k
                let zca = &vk * &wpca; // d × d
                let zca_inv = &wpca_inv * &vk.transpose(); // (d×k)(k×d) = d × d
                (zca, zca_inv)
            }
            WhitenMode::Off => unreachable!(),
        };

        Ok(Self {
            mean,
            forward,
            inverse,
            scale,
        })
    }

    /// Apply conditioning to a single vector.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_dim() {
            bail!(
                "conditioning dim mismatch: expected {}, got {}",
                self.in_dim(),
                x.len()
            );
        }
        let xv = DVector::<f32>::from_row_slice(x) - &self.mean;
        let y = (&self.forward * xv) * self.scale;
        Ok(y.iter().copied().collect())
    }

    /// Apply to all rows.
    pub fn apply_rows(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        data.iter().map(|r| self.apply(r)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random unit vector.
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

    #[test]
    fn off_mode_is_centering_only() {
        let data: Vec<Vec<f32>> = (0..20).map(|i| synthetic(4, i)).collect();
        let c = Conditioning::fit(&data, WhitenMode::Off, 1.0, 1e-6, 1.0).unwrap();
        assert_eq!(c.in_dim(), 4);
        assert_eq!(c.out_dim(), 4);
        // forward is identity ⇒ apply just subtracts the mean.
        let y = c.apply(&data[0]).unwrap();
        for (j, &yj) in y.iter().enumerate() {
            assert!((yj - (data[0][j] - c.mean[j])).abs() < 1e-5);
        }
    }

    /// PCA-whitened data has approximately identity covariance — for directions whose
    /// eigenvalue is well above the regularization floor. (Whitened variance is exactly
    /// λ/(λ+eps); with the M1 eps floor at 1e-3, the data must be well-conditioned, so we
    /// use distinct per-dimension frequencies/amplitudes giving eigenvalues O(1) ≫ eps.)
    #[test]
    fn pca_whitening_decorrelates() {
        let n = 150;
        let d = 4;
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| {
                        let f = 0.3 + 0.4 * j as f32; // distinct frequency per dim
                        ((i as f32) * f).sin() * (1.0 + j as f32) + ((i as f32) * f * 1.7).cos()
                    })
                    .collect()
            })
            .collect();
        let c = Conditioning::fit(&data, WhitenMode::Pca, 1.0, 1e-6, 1.0).unwrap();
        let w = c.apply_rows(&data).unwrap();
        let k = c.out_dim();
        let mut mean = vec![0.0f32; k];
        for r in &w {
            for (j, &v) in r.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }
        for a in 0..k {
            let var: f32 =
                w.iter().map(|r| (r[a] - mean[a]).powi(2)).sum::<f32>() / (n as f32 - 1.0);
            assert!((var - 1.0).abs() < 0.2, "dim {a} variance {var} not ≈ 1");
        }
    }

    #[test]
    fn variance_target_truncates_rank() {
        let n = 60;
        let d = 6;
        let data: Vec<Vec<f32>> = (0..n).map(|i| synthetic(d, i)).collect();
        let full = Conditioning::fit(&data, WhitenMode::Pca, 1.0, 1e-6, 1.0).unwrap();
        let trunc = Conditioning::fit(&data, WhitenMode::Pca, 0.6, 1e-6, 1.0).unwrap();
        assert!(
            trunc.out_dim() <= full.out_dim(),
            "lower variance target should not increase rank"
        );
    }
}
