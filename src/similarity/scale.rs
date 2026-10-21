//! Reading a cosine on one reference scale, whatever embedder produced it.
//!
//! veld's similarity bars (interference 0.85, fact dedup 0.80, merge 0.97, the dream-replay
//! band, query repetition) were all tuned on MiniLM. Embedders spread cosines very
//! differently: on the LoCoMo gate corpus the median pair cosine is 0.385 under MiniLM,
//! 0.594 under bge-m3, 0.752 under nomic and 0.838 under arctic-embed-xs. A bar of 0.85 marks
//! a rare pair under MiniLM and an ordinary one under arctic.
//!
//! [`to_reference`] maps a cosine from the active embedder to the MiniLM cosine at the same
//! percentile of each model's pair-cosine distribution, so a MiniLM-tuned bar keeps its
//! meaning. MiniLM, and any model without a table, maps to itself. `VELD_SIMILARITY_SCALE=off`
//! turns the mapping off for an A/B run.
//!
//! The tables were measured on the 629 turns of the gate corpus (197,506 pairs; MiniLM from
//! veld's quantized ONNX build, the others from LM Studio's q8 builds, nomic with its
//! `search_document: ` prefix). They are priors: conversation turns, not code or tool events.
//! Per-store measured profiles replace them under plan veld-embedding-spaces-20261019 step 6.

use std::sync::atomic::{AtomicU8, Ordering};

/// Percentiles of the pair-cosine distribution at which every table below is sampled.
pub const PERCENTILES: [f32; 14] = [
    0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.99, 100.0,
];

const MINILM: [f32; 14] = [
    -0.1587, 0.0279, 0.1058, 0.1566, 0.2567, 0.3854, 0.4936, 0.5883, 0.6446, 0.7452, 0.7778,
    0.8399, 0.917, 0.9839,
];
const NOMIC: [f32; 14] = [
    0.4768, 0.5873, 0.6389, 0.6658, 0.708, 0.7518, 0.7918, 0.8254, 0.8457, 0.8841, 0.8988, 0.9289,
    0.9599, 0.9983,
];
const ARCTIC_XS: [f32; 14] = [
    0.5721, 0.698, 0.7495, 0.7716, 0.8048, 0.8382, 0.8685, 0.894, 0.9082, 0.9334, 0.9421, 0.9597,
    0.9778, 0.9996,
];
const BGE_M3: [f32; 14] = [
    0.255, 0.4101, 0.4612, 0.4901, 0.5392, 0.5937, 0.6485, 0.6989, 0.73, 0.7916, 0.8148, 0.8663,
    0.9284, 0.9983,
];

// Short names score unlike whole memories, so entity-name bars read through tables of their own,
// sampled from 600 single-word names in the same corpus (179,700 pairs). At 0.85 these pairs
// cross the bar for 0.04% under MiniLM, 0.09% under bge-m3, 0.33% under nomic and 62.5% under
// arctic-embed-xs.
const NAME_MINILM: [f32; 14] = [
    -0.0766, 0.0707, 0.1201, 0.1461, 0.1892, 0.2399, 0.2951, 0.3494, 0.386, 0.4747, 0.5264, 0.7454,
    0.9751, 0.9909,
];
const NAME_NOMIC: [f32; 14] = [
    0.4968, 0.5955, 0.6346, 0.6541, 0.6833, 0.7112, 0.7375, 0.7618, 0.7778, 0.8144, 0.8341, 0.9173,
    1.0, 1.0,
];
const NAME_ARCTIC_XS: [f32; 14] = [
    0.6498, 0.7553, 0.7852, 0.802, 0.8319, 0.8653, 0.8981, 0.9321, 0.9503, 0.972, 0.9769, 0.9852,
    1.0, 1.0,
];
const NAME_BGE_M3: [f32; 14] = [
    0.2479, 0.3448, 0.3935, 0.4202, 0.4622, 0.5052, 0.5468, 0.5872, 0.615, 0.6844, 0.7195, 0.8371,
    0.9321, 0.9758,
];

/// Which table a cosine is read through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScaleModel {
    /// MiniLM, or a model with no table: cosines are already on the reference scale.
    Reference = 0,
    Nomic = 1,
    ArcticXs = 2,
    BgeM3 = 3,
}

impl ScaleModel {
    /// The table for an embedder id or model name (`minilm`, `nomic`,
    /// `http://host:8095#text-embedding-nomic-embed-text-v1.5`, ...).
    pub fn for_embedder(id: &str) -> Self {
        let id = id.to_ascii_lowercase();
        if id.contains("nomic") {
            ScaleModel::Nomic
        } else if id.contains("arctic") {
            ScaleModel::ArcticXs
        } else if id.contains("bge-m3") {
            ScaleModel::BgeM3
        } else {
            ScaleModel::Reference
        }
    }

    fn table(self) -> Option<&'static [f32; 14]> {
        match self {
            ScaleModel::Reference => None,
            ScaleModel::Nomic => Some(&NOMIC),
            ScaleModel::ArcticXs => Some(&ARCTIC_XS),
            ScaleModel::BgeM3 => Some(&BGE_M3),
        }
    }

    /// `cosine` from this model, read on the reference (MiniLM) scale.
    pub fn to_reference(self, cosine: f32) -> f32 {
        match self.table() {
            None => cosine,
            Some(table) => interpolate(cosine, table, &MINILM),
        }
    }

    fn name_table(self) -> Option<&'static [f32; 14]> {
        match self {
            ScaleModel::Reference => None,
            ScaleModel::Nomic => Some(&NAME_NOMIC),
            ScaleModel::ArcticXs => Some(&NAME_ARCTIC_XS),
            ScaleModel::BgeM3 => Some(&NAME_BGE_M3),
        }
    }

    /// A cosine between two entity-name embeddings from this model, read on MiniLM's
    /// name scale.
    pub fn name_to_reference(self, cosine: f32) -> f32 {
        match self.name_table() {
            None => cosine,
            Some(table) => interpolate(cosine, table, &NAME_MINILM),
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => ScaleModel::Nomic,
            2 => ScaleModel::ArcticXs,
            3 => ScaleModel::BgeM3,
            _ => ScaleModel::Reference,
        }
    }
}

/// Piecewise-linear map from `from`'s knots to `to`'s. Below the first knot the result is the
/// first target; above the last it runs on to (1.0, 1.0), so identical vectors stay identical.
fn interpolate(x: f32, from: &[f32; 14], to: &[f32; 14]) -> f32 {
    if !x.is_finite() {
        return x;
    }
    if x <= from[0] {
        return to[0];
    }
    for i in 1..from.len() {
        if x <= from[i] {
            let span = from[i] - from[i - 1];
            let t = if span > 0.0 {
                (x - from[i - 1]) / span
            } else {
                1.0
            };
            return to[i - 1] + t * (to[i] - to[i - 1]);
        }
    }
    let last = from.len() - 1;
    let span = 1.0 - from[last];
    if span <= 0.0 {
        return 1.0;
    }
    let t = ((x - from[last]) / span).min(1.0);
    to[last] + t * (1.0 - to[last])
}

static ACTIVE: AtomicU8 = AtomicU8::new(ScaleModel::Reference as u8);

/// Read cosines through `embedder_id`'s table from now on. Called when a memory system builds
/// its embedder; the embedder is a process-wide choice, so one active table serves every user.
pub fn set_active_embedder(embedder_id: &str) {
    let next = if disabled_by_env() {
        ScaleModel::Reference
    } else {
        ScaleModel::for_embedder(embedder_id)
    };
    let previous = ScaleModel::from_u8(ACTIVE.swap(next as u8, Ordering::Relaxed));
    if previous != next {
        tracing::info!(embedder = embedder_id, scale = ?next, "similarity scale selected");
    }
}

/// The table in use.
pub fn active() -> ScaleModel {
    ScaleModel::from_u8(ACTIVE.load(Ordering::Relaxed))
}

/// `cosine` from the active embedder, read on the reference (MiniLM) scale.
pub fn to_reference(cosine: f32) -> f32 {
    active().to_reference(cosine)
}

/// A cosine between two entity-name embeddings from the active embedder, read on MiniLM's
/// name scale.
pub fn name_to_reference(cosine: f32) -> f32 {
    active().name_to_reference(cosine)
}

fn disabled_by_env() -> bool {
    std::env::var("VELD_SIMILARITY_SCALE")
        .is_ok_and(|v| matches!(v.trim(), "off" | "0" | "false" | "reference"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minilm_and_unknown_models_are_read_as_they_are() {
        for cosine in [-0.2, 0.0, 0.385, 0.85, 0.97, 1.0] {
            assert_eq!(
                ScaleModel::for_embedder("minilm").to_reference(cosine),
                cosine
            );
            assert_eq!(
                ScaleModel::for_embedder("some-new-model").to_reference(cosine),
                cosine
            );
        }
    }

    #[test]
    fn each_table_maps_its_median_to_the_reference_median() {
        for model in [ScaleModel::Nomic, ScaleModel::ArcticXs, ScaleModel::BgeM3] {
            let median = model.table().unwrap()[5];
            assert!(
                (model.to_reference(median) - MINILM[5]).abs() < 1e-6,
                "{model:?}"
            );
        }
    }

    #[test]
    fn a_typical_nomic_nearest_neighbour_reads_below_the_interference_bar() {
        // 0.875 is the median nearest-neighbour cosine of a LoCoMo turn under nomic.
        let read = ScaleModel::Nomic.to_reference(0.875);
        assert!(read < 0.85, "{read}");
        assert!(read > 0.64, "{read}");
    }

    #[test]
    fn identical_vectors_stay_identical_and_the_map_is_monotone() {
        for model in [ScaleModel::Nomic, ScaleModel::ArcticXs, ScaleModel::BgeM3] {
            assert!((model.to_reference(1.0) - 1.0).abs() < 1e-6, "{model:?}");
            let mut previous = f32::NEG_INFINITY;
            for step in 0..=1000 {
                let x = -0.2 + 1.2 * step as f32 / 1000.0;
                let y = model.to_reference(x);
                assert!(y >= previous - 1e-6, "{model:?} not monotone at {x}");
                previous = y;
            }
        }
    }

    #[test]
    fn a_typical_arctic_name_pair_reads_far_below_the_concept_merge_bar() {
        // 0.90 sits between the 75th and 90th percentile of arctic name pairs.
        let read = ScaleModel::ArcticXs.name_to_reference(0.90);
        assert!(read < 0.4, "{read}");
        assert_eq!(ScaleModel::Reference.name_to_reference(0.90), 0.90);
    }

    #[test]
    fn name_tables_are_monotone() {
        for model in [ScaleModel::Nomic, ScaleModel::ArcticXs, ScaleModel::BgeM3] {
            let mut previous = f32::NEG_INFINITY;
            for step in 0..=1000 {
                let x = -0.2 + 1.2 * step as f32 / 1000.0;
                let y = model.name_to_reference(x);
                assert!(y >= previous - 1e-6, "{model:?} not monotone at {x}");
                previous = y;
            }
        }
    }

    #[test]
    fn embedder_ids_select_their_table() {
        assert_eq!(ScaleModel::for_embedder("nomic"), ScaleModel::Nomic);
        assert_eq!(
            ScaleModel::for_embedder("http://127.0.0.1:8095#text-embedding-nomic-embed-text-v1.5"),
            ScaleModel::Nomic
        );
        assert_eq!(
            ScaleModel::for_embedder("arctic-embed-xs"),
            ScaleModel::ArcticXs
        );
        assert_eq!(
            ScaleModel::for_embedder("http://h#text-embedding-bge-m3"),
            ScaleModel::BgeM3
        );
        assert_eq!(ScaleModel::for_embedder("minilm"), ScaleModel::Reference);
    }
}
