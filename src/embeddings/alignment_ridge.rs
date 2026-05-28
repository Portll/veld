//! Ridge-regression alignment — unconstrained linear projection with L2
//! regularisation. Used when Procrustes residuals are too large (the two
//! spaces are not isometric).
//!
//! Ridge has no built-in norm preservation; outputs are L2-renormalized so
//! they live on the unit sphere matching the primary Vamana index.

use anyhow::{anyhow, bail, Result};
use nalgebra::{DMatrix, DVector};

use super::alignment::{unix_ts_now, Alignment, AlignmentHeader, AlignmentPairId};

pub struct RidgeAlignment {
    pair_id: AlignmentPairId,
    in_dim: usize,
    out_dim: usize,
    weight: DMatrix<f32>, // (out_dim × in_dim)
    lambda: f32,
    fit_unix_ts: i64,
    eval_paired_cosine_mean: Option<f32>,
}

impl RidgeAlignment {
    /// Solve W = (B^T B + λI)^{-1} B^T A. W shape: (d_s × d_p), stored
    /// transposed for project-by-multiply.
    pub fn fit(
        pair_id: AlignmentPairId,
        primary: &[Vec<f32>],
        secondary: &[Vec<f32>],
        lambda: f32,
    ) -> Result<Self> {
        if primary.len() != secondary.len() {
            bail!("primary/secondary length mismatch");
        }
        if primary.is_empty() {
            bail!("empty fit");
        }
        if lambda < 0.0 {
            bail!("lambda must be non-negative");
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
        }
        for (i, v) in secondary.iter().enumerate() {
            if v.len() != d_s {
                bail!("secondary row {i}: dim {} != {d_s}", v.len());
            }
        }

        let a = DMatrix::<f32>::from_row_iterator(n, d_p, primary.iter().flatten().copied());
        let b = DMatrix::<f32>::from_row_iterator(n, d_s, secondary.iter().flatten().copied());

        let btb = b.transpose() * &b;
        let reg = DMatrix::<f32>::identity(d_s, d_s) * lambda;
        let lhs = btb + reg;
        let rhs = b.transpose() * &a; // (d_s × d_p)
        let inv = lhs
            .try_inverse()
            .ok_or_else(|| anyhow!("B^T B + λI not invertible — try larger λ"))?;
        let w = inv * rhs; // (d_s × d_p)
        let weight = w.transpose(); // (d_p × d_s)

        Ok(Self {
            pair_id,
            in_dim: d_s,
            out_dim: d_p,
            weight,
            lambda,
            fit_unix_ts: unix_ts_now(),
            eval_paired_cosine_mean: None,
        })
    }

    pub fn set_eval(&mut self, paired_cosine_mean: f32) {
        self.eval_paired_cosine_mean = Some(paired_cosine_mean);
    }

    /// Payload layout: `[lambda: f32 LE][weight: (out_dim × in_dim) f32 LE row-major]`.
    pub fn from_payload(header: AlignmentHeader, payload: &[u8]) -> Result<Self> {
        if payload.len() < 4 {
            bail!("ridge payload too short");
        }
        let lambda = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let mat = &payload[4..];
        let expected = header.in_dim * header.out_dim * 4;
        if mat.len() != expected {
            bail!(
                "ridge payload size mismatch: expected {} got {}",
                expected,
                mat.len()
            );
        }
        let floats: Vec<f32> = mat
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let weight = DMatrix::<f32>::from_row_slice(header.out_dim, header.in_dim, &floats);
        Ok(Self {
            pair_id: header.pair_id,
            in_dim: header.in_dim,
            out_dim: header.out_dim,
            weight,
            lambda,
            fit_unix_ts: header.fit_unix_ts,
            eval_paired_cosine_mean: header.eval_paired_cosine_mean,
        })
    }
}

fn l2_renormalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl Alignment for RidgeAlignment {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>> {
        if secondary.len() != self.in_dim {
            bail!(
                "ridge project: expected in_dim={}, got {}",
                self.in_dim,
                secondary.len()
            );
        }
        let x = DVector::<f32>::from_row_slice(secondary);
        let y = &self.weight * x;
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
        "ridge"
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
        let mut out = Vec::with_capacity(4 + self.in_dim * self.out_dim * 4);
        out.extend_from_slice(&self.lambda.to_le_bytes());
        for r in 0..self.out_dim {
            for c in 0..self.in_dim {
                out.extend_from_slice(&self.weight[(r, c)].to_le_bytes());
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
        AlignmentPairId::new("modelA-16", "modelB-8")
    }

    fn norm(v: &mut [f32]) {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    fn synthetic(dim: usize, seed: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        for (j, x) in v.iter_mut().enumerate() {
            *x = ((seed * 7 + j) as f32 * 0.13).sin();
        }
        norm(&mut v);
        v
    }

    #[test]
    fn fits_and_projects_to_unit_norm() {
        let n = 30;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(16, i)).collect();
        let secondary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(8, i + 100)).collect();
        let a = RidgeAlignment::fit(pid(), &primary, &secondary, 0.01).unwrap();
        assert_eq!(a.in_dim(), 8);
        assert_eq!(a.out_dim(), 16);

        for s in secondary.iter().take(5) {
            let p = a.project(s).unwrap();
            let pn: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (pn - 1.0).abs() < 1e-4,
                "ridge projection not unit norm: {pn}"
            );
        }
    }

    #[test]
    fn save_load_round_trip() {
        let n = 20;
        let primary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(8, i)).collect();
        let secondary: Vec<Vec<f32>> = (0..n).map(|i| synthetic(4, i + 50)).collect();
        let a = RidgeAlignment::fit(pid(), &primary, &secondary, 0.05).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("r.bin");
        save_alignment(&path, &a).unwrap();
        let (header, payload) = read_alignment_file(&path).unwrap();
        assert_eq!(header.method, "ridge");
        let b = RidgeAlignment::from_payload(header, &payload).unwrap();

        let pa = a.project(&secondary[0]).unwrap();
        let pb = b.project(&secondary[0]).unwrap();
        for (x, y) in pa.iter().zip(pb.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_negative_lambda() {
        let primary = vec![vec![1.0_f32, 0.0]];
        let secondary = vec![vec![1.0_f32, 0.0]];
        assert!(RidgeAlignment::fit(pid(), &primary, &secondary, -0.01).is_err());
    }
}
