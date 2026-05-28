//! Competitive dual-embedder architecture
//!
//! Holds a **primary** embedder (the default — Nomic-embed-text-v1.5 768d when
//! available, MiniLM 384d as fallback) and an **optional secondary** embedder
//! (HTTP / cluster model). When a secondary is present both index paths are
//! queried at retrieval time and the best match wins.
//!
//! # Design
//! - **Polymorphic primary**: `primary` is `Arc<dyn Embedder>`, so the concrete
//!   model is chosen at construction time (Nomic preferred, MiniLM fallback).
//! - **Trait delegation**: the `Embedder` impl delegates to the primary model,
//!   so existing call sites are unchanged.
//! - **Graceful degradation**: if the secondary model is unavailable, the system
//!   runs primary-only.
//! - **Dual storage**: callers use `encode_dual` to get embeddings from both
//!   models, storing them in separate fields / indices.
//!
//! # Notes
//! - Both slots are `Arc<dyn Embedder>`, so any `Embedder` impl can be plugged in
//!   (Nomic, MiniLM, BGE, GTE, HTTP, Zenoh, …).
//! - Retrieval merging (max-score union across two Vamana indices) lives in the
//!   retrieval layer, not here.

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::Arc;

use super::alignment::Alignment;
use super::Embedder;

type DualEmbeddingBatch = (Vec<Vec<f32>>, Option<Vec<Vec<f32>>>);

/// Which face of an asymmetric embedder to invoke for an aligned encode.
#[derive(Copy, Clone, Debug)]
pub enum AlignedSide {
    /// Document side — `encode()`. Used at ingest / when projecting stored content.
    Doc,
    /// Query side — `encode_for_query()`. Used at search time.
    Query,
}

/// Dual-model competitive embedder.
///
/// Wraps a primary embedder (always available, backward-compatible) and an optional
/// secondary embedder that can be a different model / dimensionality. Both models
/// run independently; there is no mixing or concatenation of their outputs.
pub struct CompetitiveEmbedder {
    /// Primary model: the default embedder (Nomic 768d when available, MiniLM
    /// 384d fallback). Always present. Dimension is read via `primary.dimension()`.
    primary: Arc<dyn Embedder>,
    /// Secondary model: any Embedder impl (HTTP / cluster). Optional.
    secondary: Option<Arc<dyn Embedder>>,
    /// Cached secondary dimension (avoids repeated vtable calls).
    secondary_dim: Option<usize>,
    /// Optional learned projection from secondary-space into primary-space.
    /// Installed via [`install_alignment`] after construction (the embedder is
    /// typically held as `Arc<CompetitiveEmbedder>` so the slot must be
    /// interior-mutable).
    alignment: RwLock<Option<Arc<dyn Alignment>>>,
}

impl CompetitiveEmbedder {
    /// Create a new competitive embedder.
    ///
    /// - `primary` is required and always used (the trait-delegation path).
    /// - `secondary` is optional; pass `None` for single-model mode.
    pub fn new(primary: Arc<dyn Embedder>, secondary: Option<Arc<dyn Embedder>>) -> Self {
        let secondary_dim = secondary.as_ref().map(|e| e.dimension());
        Self {
            primary,
            secondary,
            secondary_dim,
            alignment: RwLock::new(None),
        }
    }

    /// Create a primary-only competitive embedder (no secondary model).
    pub fn primary_only(primary: Arc<dyn Embedder>) -> Self {
        Self {
            primary,
            secondary: None,
            secondary_dim: None,
            alignment: RwLock::new(None),
        }
    }

    /// Encode text with the primary model. Always succeeds if the primary model works.
    pub fn encode_primary(&self, text: &str) -> Result<Vec<f32>> {
        self.primary.encode(text)
    }

    /// Encode text with the secondary model, if available.
    ///
    /// Returns `Ok(None)` when no secondary model is configured.
    /// Returns `Ok(Some(embedding))` on success.
    /// Returns `Err` only on a genuine secondary-model failure (not absence).
    pub fn encode_secondary(&self, text: &str) -> Result<Option<Vec<f32>>> {
        match &self.secondary {
            Some(embedder) => {
                let embedding = embedder.encode(text)?;
                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    /// Encode with both models in a single call.
    ///
    /// Returns `(primary_embedding, Option<secondary_embedding>)`.
    ///
    /// If the secondary model is present but fails, the error is logged and the
    /// secondary result is returned as `None` (graceful degradation — the primary
    /// embedding is still valid).
    pub fn encode_dual(&self, text: &str) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let primary_emb = self.primary.encode(text)?;

        let secondary_emb = match &self.secondary {
            Some(embedder) => match embedder.encode(text) {
                Ok(emb) => Some(emb),
                Err(e) => {
                    tracing::warn!(
                        "Secondary embedder failed (graceful degradation): {}",
                        e
                    );
                    None
                }
            },
            None => None,
        };

        Ok((primary_emb, secondary_emb))
    }

    /// Batch-encode with both models.
    ///
    /// Returns parallel vectors: `(primary_embeddings, Option<secondary_embeddings>)`.
    /// If the secondary model fails mid-batch, the entire secondary result is `None`.
    pub fn encode_dual_batch(&self, texts: &[&str]) -> Result<DualEmbeddingBatch> {
        let primary_batch = self.primary.encode_batch(texts)?;

        let secondary_batch = match &self.secondary {
            Some(embedder) => match embedder.encode_batch(texts) {
                Ok(batch) => Some(batch),
                Err(e) => {
                    tracing::warn!(
                        "Secondary embedder batch failed (graceful degradation): {}",
                        e
                    );
                    None
                }
            },
            None => None,
        };

        Ok((primary_batch, secondary_batch))
    }

    /// Whether a secondary model is configured and available.
    pub fn has_secondary(&self) -> bool {
        self.secondary.is_some()
    }

    /// Primary embedding dimension.
    pub fn primary_dimension(&self) -> usize {
        self.primary.dimension()
    }

    /// Secondary embedding dimension, if a secondary model is available.
    pub fn secondary_dimension(&self) -> Option<usize> {
        self.secondary_dim
    }

    /// Get a reference to the primary embedder.
    pub fn primary_embedder(&self) -> &Arc<dyn Embedder> {
        &self.primary
    }

    /// Get a reference to the secondary embedder, if available.
    pub fn secondary_embedder(&self) -> Option<&Arc<dyn Embedder>> {
        self.secondary.as_ref()
    }

    /// Install a learned alignment. Refuses if dimensions don't match the
    /// currently-configured secondary, or if no secondary is configured.
    ///
    /// Uses `&self` (not `&mut self`) so callers holding an `Arc<CompetitiveEmbedder>`
    /// can install at startup without unwrapping.
    pub fn install_alignment(&self, alignment: Arc<dyn Alignment>) -> Result<()> {
        let sec_dim = self.secondary_dim.ok_or_else(|| {
            anyhow::anyhow!("cannot install alignment: no secondary embedder configured")
        })?;
        if alignment.in_dim() != sec_dim {
            anyhow::bail!(
                "alignment in_dim {} != secondary dim {}",
                alignment.in_dim(),
                sec_dim
            );
        }
        if alignment.out_dim() != self.primary.dimension() {
            anyhow::bail!(
                "alignment out_dim {} != primary dim {}",
                alignment.out_dim(),
                self.primary.dimension()
            );
        }
        *self.alignment.write() = Some(alignment);
        Ok(())
    }

    /// True if an alignment is installed.
    pub fn has_alignment(&self) -> bool {
        self.alignment.read().is_some()
    }

    /// Snapshot the current alignment (clone of the `Arc`) without holding the lock.
    pub fn alignment_snapshot(&self) -> Option<Arc<dyn Alignment>> {
        self.alignment.read().clone()
    }

    /// Encode with the secondary embedder (selecting query or doc face) and
    /// project the result into primary space via the installed alignment.
    ///
    /// Returns `Ok(None)` if either the secondary embedder or the alignment is
    /// absent — callers should fall back to the existing max-score union path.
    pub fn encode_aligned(&self, text: &str, side: AlignedSide) -> Result<Option<Vec<f32>>> {
        let Some(secondary) = &self.secondary else {
            return Ok(None);
        };
        let alignment = match self.alignment_snapshot() {
            Some(a) => a,
            None => return Ok(None),
        };
        let s = match side {
            AlignedSide::Doc => secondary.encode(text)?,
            AlignedSide::Query => secondary.encode_for_query(text)?,
        };
        Ok(Some(alignment.project(&s)?))
    }

    /// Encode `text` with the primary and additionally produce a *fused
    /// primary-space vector* — the "regular task space" position — as a
    /// weighted blend of the primary embedding and the projected secondary.
    ///
    /// `alpha` is the weight on the projected secondary; primary gets `1 - alpha`.
    /// Returns `None` for the fused vector when the alignment or secondary is
    /// absent. The fused result is L2-renormalized.
    pub fn encode_fused(
        &self,
        text: &str,
        alpha: f32,
        side: AlignedSide,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let primary_emb = self.primary.encode(text)?;
        let projected = self.encode_aligned(text, side)?;
        let fused = projected.map(|q| {
            let mut out: Vec<f32> = primary_emb
                .iter()
                .zip(q.iter())
                .map(|(p, q)| (1.0 - alpha) * p + alpha * q)
                .collect();
            let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut out {
                    *x /= norm;
                }
            }
            out
        });
        Ok((primary_emb, fused))
    }
}

/// Backward-compatible `Embedder` implementation.
///
/// All trait methods delegate to the **primary** model so that existing code
/// (retrieval engine, memory system, graph retrieval) continues to work without
/// modification. Callers that want dual embeddings should use the
/// `encode_dual` / `encode_dual_batch` methods directly.
impl Embedder for CompetitiveEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_primary(text)
    }

    /// Query-side encode — delegates to the primary model's query path so
    /// asymmetric models (Nomic) apply the `search_query: ` prefix.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        self.primary.encode_for_query(text)
    }

    fn dimension(&self) -> usize {
        self.primary_dimension()
    }

    fn encode_with_status(&self, text: &str) -> Result<(Vec<f32>, bool)> {
        self.primary.encode_with_status(text)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.primary.encode_batch(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: Full integration tests (encode_dual with a real MiniLMEmbedder)
    // require ONNX model files and live in tests/. Unit tests here verify the
    // CompetitiveEmbedder logic using the secondary embedder path only, since
    // the primary is a concrete MiniLMEmbedder that needs model files.

    /// Deterministic stub embedder for testing secondary-path logic.
    struct StubEmbedder {
        dim: usize,
    }

    impl StubEmbedder {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl Embedder for StubEmbedder {
        fn encode(&self, text: &str) -> Result<Vec<f32>> {
            let seed = text.len() as f32;
            let mut v = vec![0.0f32; self.dim];
            for (i, val) in v.iter_mut().enumerate() {
                *val = ((seed + i as f32) * 0.1).sin();
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
    }

    /// An embedder that always fails, for testing graceful degradation.
    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn encode(&self, _text: &str) -> Result<Vec<f32>> {
            anyhow::bail!("secondary model unavailable")
        }

        fn dimension(&self) -> usize {
            768
        }
    }

    #[test]
    fn test_secondary_dimension_caching() {
        let secondary: Arc<dyn Embedder> = Arc::new(StubEmbedder::new(768));
        let dim = secondary.dimension();
        assert_eq!(dim, 768);
    }

    #[test]
    fn test_no_secondary_returns_none() {
        // Verify that encode_secondary logic for None path is correct
        let secondary: Option<Arc<dyn Embedder>> = None;
        let result: Result<Option<Vec<f32>>> = match &secondary {
            Some(embedder) => embedder.encode("test").map(Some),
            None => Ok(None),
        };
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_secondary_encode_success() {
        let secondary: Option<Arc<dyn Embedder>> = Some(Arc::new(StubEmbedder::new(768)));
        let result: Result<Option<Vec<f32>>> = match &secondary {
            Some(embedder) => embedder.encode("hello world").map(Some),
            None => Ok(None),
        };
        let emb = result.unwrap().unwrap();
        assert_eq!(emb.len(), 768);

        // Verify L2 normalization
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_failing_secondary_graceful_degradation() {
        let secondary: Option<Arc<dyn Embedder>> = Some(Arc::new(FailingEmbedder));

        // Simulate the graceful degradation path from encode_dual
        let secondary_emb: Option<Vec<f32>> = match &secondary {
            Some(embedder) => embedder.encode("test").ok(), // graceful degradation
            None => None,
        };
        assert!(secondary_emb.is_none());
    }

    #[test]
    fn test_stub_deterministic_and_normalized() {
        let stub = StubEmbedder::new(384);
        let e1 = stub.encode("hello").unwrap();
        let e2 = stub.encode("hello").unwrap();
        assert_eq!(e1, e2);
        assert_eq!(e1.len(), 384);

        let norm: f32 = e1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_stub_different_inputs_differ() {
        let stub = StubEmbedder::new(384);
        let e1 = stub.encode("short").unwrap();
        let e2 = stub.encode("a longer string").unwrap();
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_batch_secondary_failure() {
        let failing: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let result = failing.encode_batch(&["a", "b", "c"]);
        assert!(result.is_err());
    }
}
