//! Vector similarity search for semantic retrieval
//!
//! [`cosine_similarity`] runs on the SIMD kernels in distance_inline. The
//! `*_sequential` and `*_floored` cosines sum left to right instead: each keeps the
//! exact results of the code it replaced, which rankings and reports were built on.
//! The vector indexes use `distance_inline::cosine_similarity_inline`, which divides
//! by `sqrt(|a|² |b|²)` and is a different function again.

use crate::vector_db::distance_inline::dot_product_inline;
use ordered_float::OrderedFloat;

pub mod scale;

/// Compute cosine similarity between two vectors (SIMD-optimized)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot = dot_product_inline(a, b);
    let norm_a = dot_product_inline(a, a).sqrt();
    let norm_b = dot_product_inline(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let result = dot / (norm_a * norm_b);

    // Guard against NaN/Inf from corrupt embeddings (e.g. NaN in stored vectors).
    // NaN comparisons are always false, so norm_a/norm_b checks above won't catch NaN inputs.
    if result.is_finite() {
        result
    } else {
        0.0
    }
}

/// Cosine similarity with every sum taken left to right in f32, as `Iterator::sum` takes it.
/// Mismatched lengths, an empty vector or a zero vector give 0.0. It differs from
/// [`cosine_similarity`] in the last bits, because that kernel sums in SIMD lanes with fused
/// multiply-adds, and in returning NaN for a NaN or infinite component where that function
/// returns 0.0, so a score already ranked with this function must keep using it.
pub fn cosine_similarity_sequential(a: &[f32], b: &[f32]) -> f32 {
    sequential_cosine(a, b, -0.0, 0.0).unwrap_or(0.0)
}

/// The sequential cosine as an `Option`: None for mismatched lengths, an empty vector, or a
/// squared norm at or below `min_norm_sq`. Its sums start from +0.0, so a dot product that is
/// exactly zero comes out +0.0 where [`cosine_similarity_sequential`] gives -0.0; callers that
/// order scores with `total_cmp` see the difference.
pub fn cosine_similarity_sequential_checked(a: &[f32], b: &[f32], min_norm_sq: f32) -> Option<f32> {
    sequential_cosine(a, b, 0.0, min_norm_sq)
}

/// Cosine of `a` and `b` summed left to right from `seed`, the dot product over the full
/// (equal) length. The seed decides only the sign of a sum that is exactly zero.
fn sequential_cosine(a: &[f32], b: &[f32], seed: f32, min_norm_sq: f32) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut norm_a, mut norm_b) = (seed, seed, seed);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= min_norm_sq || norm_b <= min_norm_sq {
        return None;
    }
    Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

/// Cosine for the alignment fit and eval reports: the dot product over the common prefix of `a`
/// and `b`, divided by the product of their full norms floored at 1e-12, with no length or zero
/// check. Sums run left to right as in [`cosine_similarity_sequential`].
pub fn cosine_similarity_floored(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}

/// Cosine similarity for vectors the caller already knows are L2-normalised: the
/// dot product alone, without recomputing both norms. Mismatched lengths and
/// non-finite results give 0.0, as in [`cosine_similarity`]. On a vector that is
/// not unit length this is NOT cosine similarity; use [`cosine_similarity`] there.
#[inline]
pub fn dot_normalized(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot = dot_product_inline(a, b);
    if dot.is_finite() {
        dot
    } else {
        0.0
    }
}

/// Find top-k most similar vectors
pub fn top_k_similar<T>(query: &[f32], candidates: &[(Vec<f32>, T)], k: usize) -> Vec<(f32, T)>
where
    T: Clone,
{
    let mut scored: Vec<(OrderedFloat<f32>, T)> = candidates
        .iter()
        .map(|(vec, item)| {
            let score = cosine_similarity(query, vec);
            (OrderedFloat(score), item.clone())
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    // Take top k
    scored
        .into_iter()
        .take(k)
        .map(|(score, item)| (score.0, item))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_normalized_agrees_with_cosine_on_unit_vectors() {
        let a = [0.6_f32, 0.8, 0.0];
        let b = [0.0_f32, 0.6, 0.8];
        assert!((dot_normalized(&a, &b) - cosine_similarity(&a, &b)).abs() < 1e-6);
        assert!((dot_normalized(&a, &a) - 1.0).abs() < 1e-6);
        assert_eq!(dot_normalized(&a, &[1.0]), 0.0);
        assert_eq!(dot_normalized(&[f32::NAN, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn sequential_cosine_values_and_guards() {
        let a = [1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity_sequential(&a, &a) - 1.0).abs() < 0.001);
        assert!(cosine_similarity_sequential(&a, &[0.0, 1.0, 0.0]).abs() < 0.001);
        assert!((cosine_similarity_sequential(&a, &[-1.0, 0.0, 0.0]) + 1.0).abs() < 0.001);
        assert_eq!(cosine_similarity_sequential(&[], &[]), 0.0);
        assert_eq!(cosine_similarity_sequential(&a, &[1.0]), 0.0);
        assert_eq!(cosine_similarity_sequential(&a, &[0.0; 3]), 0.0);
        assert!(cosine_similarity_sequential(&[f32::NAN, 1.0], &[1.0, 0.0]).is_nan());

        assert_eq!(cosine_similarity_sequential_checked(&a, &[1.0], 0.0), None);
        assert_eq!(
            cosine_similarity_sequential_checked(&a, &[0.0; 3], 0.0),
            None
        );
        assert_eq!(
            cosine_similarity_sequential_checked(&a, &[1e-4, 0.0, 0.0], f32::EPSILON),
            None
        );
        assert_eq!(
            cosine_similarity_sequential_checked(&a, &[2.0, 0.0, 0.0], f32::EPSILON),
            Some(1.0)
        );
    }

    #[test]
    fn sequential_cosine_seeds_differ_only_in_the_sign_of_zero() {
        let a = [1.0_f32, 0.0];
        let b = [-0.0_f32, -1.0];
        assert_eq!(
            cosine_similarity_sequential(&a, &b).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            cosine_similarity_sequential_checked(&a, &b, 0.0).map(f32::to_bits),
            Some(0.0_f32.to_bits())
        );
    }

    #[test]
    fn floored_cosine_has_no_length_or_zero_guard() {
        assert!((cosine_similarity_floored(&[3.0, 4.0], &[3.0, 4.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity_floored(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        // The dot product stops at the shorter vector; the norms do not.
        let partial = cosine_similarity_floored(&[1.0, 1.0], &[1.0]);
        assert!((partial - 1.0 / 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Mismatched lengths are treated as non-comparable.
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Any zero vector should yield 0 to avoid NaN propagation.
        let zero = vec![0.0, 0.0, 0.0];
        let vec3 = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&zero, &vec3), 0.0);
        assert_eq!(cosine_similarity(&vec3, &zero), 0.0);
    }

    #[test]
    fn test_cosine_similarity_nan_inf_guard() {
        // NaN in embeddings should return 0.0, not propagate NaN
        let nan_vec = vec![f32::NAN, 1.0, 0.0];
        let normal = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&nan_vec, &normal), 0.0);
        assert_eq!(cosine_similarity(&normal, &nan_vec), 0.0);

        // Inf in embeddings should return 0.0
        let inf_vec = vec![f32::INFINITY, 1.0, 0.0];
        assert_eq!(cosine_similarity(&inf_vec, &normal), 0.0);

        // NaN in both should return 0.0
        assert_eq!(cosine_similarity(&nan_vec, &nan_vec), 0.0);
    }

    #[test]
    fn test_cosine_similarity_negative_correlation() {
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!((score + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_top_k_similar_orders_and_limits() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            (vec![1.0, 0.0], "perfect"),
            (vec![0.7, 0.7], "diagonal"),
            (vec![0.0, 1.0], "orthogonal"),
            (vec![-1.0, 0.0], "opposite"),
        ];

        let top2 = top_k_similar(&query, &candidates, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].1, "perfect");
        assert_eq!(top2[1].1, "diagonal");
        assert!(top2[0].0 >= top2[1].0);
    }

    #[test]
    fn test_top_k_similar_k_larger_than_candidates() {
        let query = vec![1.0, 0.0];
        let candidates = vec![(vec![1.0, 0.0], 1u8), (vec![0.0, 1.0], 2u8)];

        let result = top_k_similar(&query, &candidates, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_top_k_similar_empty_candidates() {
        let query = vec![1.0, 0.0];
        let candidates: Vec<(Vec<f32>, &str)> = Vec::new();
        let result = top_k_similar(&query, &candidates, 3);
        assert!(result.is_empty());
    }
}
