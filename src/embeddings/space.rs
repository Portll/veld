//! Which embedder produced a vector, as a value a stored record can carry.
//!
//! A space id is `<model>@<dimension>`: the canonical model name and the vector width. Two
//! vectors are comparable by cosine only when their space ids are equal, so the id goes
//! wherever a vector goes. It is derived from a live embedder or from the corpus markers,
//! never written by hand.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::Embedder;

/// The model part of a space whose vectors predate space ids and whose model the corpus
/// markers cannot name. Only the width is known.
pub const UNRECORDED_MODEL: &str = "unrecorded";

/// The corpus embedder marker beside a tenant's store.
pub const EMBEDDER_MARKER: &str = ".veld_embedder";
/// The corpus dimension marker beside a tenant's store.
pub const DIMENSION_MARKER: &str = ".veld_embedding_dim";
/// The space every absent field in the store was stamped with, written once the stamp pass
/// has run, so the pass is not repeated on every boot.
pub const STAMP_MARKER: &str = ".veld_spaces_stamped";

/// LM Studio publishes embedding models as `text-embedding-<model>`; llama-server and the ONNX
/// loaders use `<model>`. The weights are the same, so the vectors share a space.
const SERVER_MODEL_PREFIX: &str = "text-embedding-";

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpaceId(String);

impl SpaceId {
    pub fn new(model: &str, dimension: usize) -> Self {
        Self(format!("{}@{dimension}", canonical_model(model)))
    }

    /// A vector of `dimension` floats whose model nothing recorded.
    pub fn unrecorded(dimension: usize) -> Self {
        Self(format!("{UNRECORDED_MODEL}@{dimension}"))
    }

    /// The space `embedder` writes at its configured width.
    pub fn of(embedder: &dyn Embedder) -> Self {
        Self::new(&embedder_model(embedder), embedder.dimension())
    }

    /// The space of `vector` as produced by `embedder`. The width comes from the vector, not
    /// the embedder's configuration, because an HTTP embedder reports a default width until
    /// its first answer.
    pub fn for_vector(embedder: &dyn Embedder, vector: &[f32]) -> Self {
        Self::new(&embedder_model(embedder), vector.len())
    }

    /// `<model>@<dimension>` with a non-empty model and a positive width, else `None`.
    pub fn parse(raw: &str) -> Option<Self> {
        let (model, dimension) = raw.trim().rsplit_once('@')?;
        let dimension: usize = dimension.parse().ok()?;
        (!model.trim().is_empty() && dimension > 0).then(|| Self::new(model, dimension))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn model(&self) -> &str {
        self.0.rsplit_once('@').map_or(self.0.as_str(), |(m, _)| m)
    }

    pub fn dimension(&self) -> usize {
        self.0
            .rsplit_once('@')
            .and_then(|(_, d)| d.parse().ok())
            .unwrap_or(0)
    }

    /// Whether the model is known, as opposed to a width-only legacy space.
    pub fn is_recorded(&self) -> bool {
        self.model() != UNRECORDED_MODEL
    }
}

impl fmt::Display for SpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Trimmed, lowercased, without the server catalogue prefix.
pub fn canonical_model(name: &str) -> String {
    let lowered = name.trim().to_ascii_lowercase();
    match lowered.strip_prefix(SERVER_MODEL_PREFIX) {
        Some(rest) if !rest.is_empty() => rest.to_string(),
        _ => lowered,
    }
}

/// The model an embedder runs. An embedder that does not name its model is identified by its
/// id instead, so two anonymous embedders of one width are still told apart by kind.
pub fn embedder_model(embedder: &dyn Embedder) -> String {
    match embedder.model_id() {
        "" | "unknown" => embedder.id().to_string(),
        model => model.to_string(),
    }
}

/// The model the corpus marker's embedder id runs, when the id names one. Mirrors the dispatch
/// in `load_embedder_by_id`. An HTTP id without `#model` predates model-qualified ids and names
/// no model, so it maps to `None`.
pub fn model_for_embedder_id(id: &str) -> Option<String> {
    let id = id.trim();
    let id = id.strip_prefix("unverified:").unwrap_or(id);
    if id.starts_with("http://") || id.starts_with("https://") {
        return id
            .split_once('#')
            .map(|(_, model)| model.trim())
            .filter(|model| !model.is_empty())
            .map(str::to_string);
    }
    if id.starts_with("nomic") {
        return Some(super::nomic::MODEL_ID.to_string());
    }
    if id.starts_with("minilm") {
        return Some(super::minilm::MODEL_ID.to_string());
    }
    None
}

/// The space the corpus markers say the store's primary vectors were written in.
/// `None` when either marker is missing or the embedder id names no model.
pub fn corpus_space(storage_path: &Path) -> Option<SpaceId> {
    let read = |name: &str| {
        std::fs::read_to_string(storage_path.join(name))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let dimension: usize = read(DIMENSION_MARKER)?.parse().ok()?;
    let model = model_for_embedder_id(&read(EMBEDDER_MARKER)?)?;
    (dimension > 0).then(|| SpaceId::new(&model, dimension))
}

/// The space of a stored vector of `len` floats. A recorded space is the truth. An absent one
/// is `assumed` when the widths agree, else a width-only space: a vector cannot be in a space
/// whose width it does not have.
pub fn resolve(stored: Option<&SpaceId>, len: usize, assumed: Option<&SpaceId>) -> SpaceId {
    match (stored, assumed) {
        (Some(space), _) => space.clone(),
        (None, Some(space)) if space.dimension() == len => space.clone(),
        (None, _) => SpaceId::unrecorded(len),
    }
}

/// How an instance reads a vector slot that records no space: its primary index's space for the
/// primary slot, its secondary index's for the secondary. A replication leader sends its own
/// with every page, so a follower with another embedder does not read the leader's vectors as
/// its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotSpaces {
    pub primary: SpaceId,
    #[serde(default)]
    pub secondary: Option<SpaceId>,
}

/// The space registry beside a tenant's corpus markers: every space the corpus holds, read by
/// `veld doctor` and anything else that must not open the store to ask.
pub const REGISTRY_FILE: &str = ".veld_spaces.json";
pub const REGISTRY_VERSION: u32 = 1;

/// How a space's vectors are held in its index.
pub const CODEC_F32: &str = "f32";
/// Product-quantized codes, searched flat and reranked against the stored vectors.
pub const CODEC_PQ: &str = "pq";

fn default_codec() -> String {
    CODEC_F32.to_string()
}

/// What one corpus, or a fleet of them, holds in a space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpaceSummary {
    pub id: SpaceId,
    pub model: String,
    pub dimension: usize,
    #[serde(default = "default_codec")]
    pub codec: String,
    /// Memories an index in this space holds.
    pub memories: usize,
    /// Backlog rows waiting for a vector in this space.
    pub backlog: usize,
    /// Memories whose stored vector is in this space and was refused by an index in another.
    pub stranded: usize,
    /// Memories keeping a vector in this space that no index here holds: a vector a re-embed
    /// replaced, or one another instance wrote. Counted by a census over the store, at most
    /// hourly, so it may lag the store by that much.
    #[serde(default)]
    pub kept: usize,
    /// How this space's search similarities are distributed, learned from its own searches;
    /// see [`ScoreStats`]. Absent until enough searches have run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scores: Option<ScoreStats>,
}

/// The distribution of the best similarities a space's searches return, as an exponential
/// moving mean and variance. Two models' cosines are not on one scale: a sharper model's
/// near-misses can outscore a flatter model's hits. A similarity is moved onto another space's
/// scale by matching z-scores, which is what any comparison of raw cosines across spaces needs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ScoreStats {
    pub mean: f32,
    pub variance: f32,
    pub samples: u64,
}

impl ScoreStats {
    /// Weight of a new search in the moving statistics.
    const ALPHA: f32 = 0.02;
    /// Searches before the statistics are used: too few, and the scale is noise.
    pub const MIN_SAMPLES: u64 = 30;

    pub fn observe(stats: Option<Self>, value: f32) -> Self {
        match stats {
            None => Self {
                mean: value,
                variance: 0.0,
                samples: 1,
            },
            Some(s) => {
                let delta = value - s.mean;
                let mean = s.mean + Self::ALPHA * delta;
                let variance = (1.0 - Self::ALPHA) * (s.variance + Self::ALPHA * delta * delta);
                Self {
                    mean,
                    variance,
                    samples: s.samples.saturating_add(1),
                }
            }
        }
    }

    pub fn ready(&self) -> bool {
        self.samples >= Self::MIN_SAMPLES && self.variance > f32::EPSILON
    }

    /// `value`, a similarity on this scale, moved onto `target`'s by matching z-scores.
    pub fn onto(&self, target: &Self, value: f32) -> f32 {
        let z = (value - self.mean) / self.variance.sqrt();
        target.mean + z * target.variance.sqrt()
    }
}

impl SpaceSummary {
    pub fn empty(id: SpaceId) -> Self {
        Self {
            model: id.model().to_string(),
            dimension: id.dimension(),
            codec: default_codec(),
            id,
            memories: 0,
            backlog: 0,
            stranded: 0,
            kept: 0,
            scores: None,
        }
    }

    fn add(&mut self, other: &Self) {
        // The codec is an index's; a summary without one has only the default to say.
        if other.memories > 0 {
            self.codec = other.codec.clone();
        }
        self.memories += other.memories;
        self.backlog += other.backlog;
        self.stranded += other.stranded;
        self.kept += other.kept;
    }
}

/// A fitted map between two spaces, recorded where the registry's readers can see it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisteredAlignment {
    pub from: SpaceId,
    pub to: SpaceId,
    pub method: String,
    /// The fit's held-out residual; the confidence an aligned result is discounted by.
    #[serde(default)]
    pub residual: Option<f32>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpaceRegistry {
    pub version: u32,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// The space the primary index holds, the one new primary vectors are written in.
    pub index_space: SpaceId,
    #[serde(default)]
    pub secondary_index_space: Option<SpaceId>,
    pub spaces: Vec<SpaceSummary>,
    #[serde(default)]
    pub alignments: Vec<RegisteredAlignment>,
}

impl SpaceRegistry {
    pub fn read(storage_path: &Path) -> anyhow::Result<Option<Self>> {
        let path = storage_path.join(REGISTRY_FILE);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
                anyhow::anyhow!("{} does not parse: {e}", path.display())
            })?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow::anyhow!("reading {}: {e}", path.display())),
        }
    }

    /// Written to a temporary file and renamed over the old one, so a reader never sees half.
    pub fn write(&self, storage_path: &Path) -> anyhow::Result<()> {
        let path = storage_path.join(REGISTRY_FILE);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Whether two registries describe the same corpus, whatever time each was written.
    pub fn same_content(&self, other: &Self) -> bool {
        self.index_space == other.index_space
            && self.secondary_index_space == other.secondary_index_space
            && self.spaces == other.spaces
            && self.alignments == other.alignments
    }
}

/// Whether a corpus holding `stored` memories is cut off from vector search: no index here holds
/// any of them, and the backlog shows vectors waiting that nothing has supplied. A corpus whose
/// every vector is in a space this server neither writes nor reaches through a map ends here;
/// so does one whose embedder has been down since its first write. Serving it as healthy hides
/// that recall is keyword-only.
pub fn corpus_orphaned(stored: usize, summaries: &[SpaceSummary]) -> bool {
    stored > 0
        && summaries.iter().all(|s| s.memories == 0)
        && summaries
            .iter()
            .any(|s| s.backlog > 0 || s.stranded > 0 || s.kept > 0)
}

/// Sum summaries by space id, in id order.
pub fn merge_summaries(summaries: impl IntoIterator<Item = SpaceSummary>) -> Vec<SpaceSummary> {
    let mut merged: std::collections::BTreeMap<SpaceId, SpaceSummary> = Default::default();
    for summary in summaries {
        merged
            .entry(summary.id.clone())
            .or_insert_with(|| SpaceSummary::empty(summary.id.clone()))
            .add(&summary);
    }
    merged.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_merged_summary_names_the_codec_of_the_indexes_holding_the_space() {
        let space = SpaceId::new("minilm-l6-v2", 384);
        let mut indexed = SpaceSummary::empty(space.clone());
        indexed.codec = CODEC_PQ.to_string();
        indexed.memories = 10;
        let mut queued = SpaceSummary::empty(space);
        queued.backlog = 3;
        for order in [vec![indexed.clone(), queued.clone()], vec![queued, indexed]] {
            let merged = merge_summaries(order);
            assert_eq!(merged[0].codec, CODEC_PQ);
            assert_eq!((merged[0].memories, merged[0].backlog), (10, 3));
        }
    }

    #[test]
    fn a_registry_round_trips_through_its_file() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(SpaceRegistry::read(dir.path()).unwrap(), None);
        let nomic = SpaceId::new("nomic-embed-text-v1.5", 768);
        let mut stranded = SpaceSummary::empty(SpaceId::unrecorded(384));
        stranded.stranded = 56;
        let registry = SpaceRegistry {
            version: REGISTRY_VERSION,
            updated_at: chrono::Utc::now(),
            index_space: nomic.clone(),
            secondary_index_space: None,
            spaces: vec![SpaceSummary::empty(nomic), stranded],
            alignments: Vec::new(),
        };
        registry.write(dir.path()).unwrap();
        let back = SpaceRegistry::read(dir.path()).unwrap().unwrap();
        assert!(back.same_content(&registry));
        assert_eq!(back.spaces[1].stranded, 56);
        assert_eq!(back.spaces[0].codec, CODEC_F32);
    }

    #[test]
    fn a_corpus_no_index_reaches_is_orphaned_only_with_work_waiting() {
        let nomic = SpaceId::new("nomic-embed-text-v1.5", 768);
        let mut waiting = SpaceSummary::empty(nomic.clone());
        waiting.backlog = 3;
        let mut stranded = SpaceSummary::empty(SpaceId::new("minilm-l6-v2", 384));
        stranded.stranded = 3;
        assert!(corpus_orphaned(3, &[waiting.clone(), stranded.clone()]));
        assert!(
            !corpus_orphaned(0, &[waiting.clone()]),
            "an empty corpus is not orphaned"
        );
        assert!(
            !corpus_orphaned(3, &[SpaceSummary::empty(nomic.clone())]),
            "nothing failed yet: a write waiting for its first vector"
        );
        let mut searchable = SpaceSummary::empty(nomic);
        searchable.memories = 1;
        assert!(!corpus_orphaned(3, &[searchable, stranded]));
    }

    #[test]
    fn a_score_moves_onto_another_space_by_matching_z_scores() {
        let mut sharp = None;
        let mut flat = None;
        for i in 0..200 {
            let wobble = if i % 2 == 0 { 1.0 } else { -1.0 };
            sharp = Some(ScoreStats::observe(sharp, 0.80 + 0.10 * wobble));
            flat = Some(ScoreStats::observe(flat, 0.50 + 0.02 * wobble));
        }
        let (sharp, flat) = (sharp.unwrap(), flat.unwrap());
        assert!(sharp.ready() && flat.ready());
        let flat_hit = flat.mean + 2.0 * flat.variance.sqrt();
        let on_sharp = flat.onto(&sharp, flat_hit);
        assert!(
            (on_sharp - (sharp.mean + 2.0 * sharp.variance.sqrt())).abs() < 1e-4,
            "two deviations above the mean stay two deviations above it"
        );
        assert!(!ScoreStats::observe(None, 0.5).ready());
    }

    #[test]
    fn summaries_sum_by_space() {
        let nomic = SpaceId::new("nomic-embed-text-v1.5", 768);
        let minilm = SpaceId::new("minilm-l6-v2", 384);
        let mut a = SpaceSummary::empty(nomic.clone());
        a.memories = 10;
        let mut b = SpaceSummary::empty(nomic.clone());
        b.memories = 5;
        b.backlog = 2;
        let mut c = SpaceSummary::empty(minilm.clone());
        c.stranded = 3;
        let merged = merge_summaries([a, b, c]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, minilm);
        assert_eq!(merged[0].stranded, 3);
        assert_eq!((merged[1].memories, merged[1].backlog), (15, 2));
        assert_eq!(merged[1].model, "nomic-embed-text-v1.5");
        assert_eq!(merged[1].dimension, 768);
    }

    struct Named {
        id: &'static str,
        model: &'static str,
        dim: usize,
    }

    impl Embedder for Named {
        fn id(&self) -> &str {
            self.id
        }
        fn model_id(&self) -> &str {
            self.model
        }
        fn encode(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; self.dim])
        }
        fn dimension(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn a_space_is_model_at_dimension_and_parses_back() {
        let space = SpaceId::new("nomic-embed-text-v1.5", 768);
        assert_eq!(space.as_str(), "nomic-embed-text-v1.5@768");
        assert_eq!(space.model(), "nomic-embed-text-v1.5");
        assert_eq!(space.dimension(), 768);
        assert_eq!(SpaceId::parse(space.as_str()), Some(space));
        assert_eq!(SpaceId::parse("nomic"), None);
        assert_eq!(SpaceId::parse("@768"), None);
        assert_eq!(SpaceId::parse("m@0"), None);
    }

    #[test]
    fn the_server_catalogue_name_and_the_model_name_are_one_space() {
        assert_eq!(
            SpaceId::new("text-embedding-nomic-embed-text-v1.5", 768),
            SpaceId::new("Nomic-Embed-Text-v1.5", 768)
        );
        assert_ne!(
            SpaceId::new("nomic-embed-text-v1.5", 768),
            SpaceId::new("nomic-embed-text-v1.5", 256),
            "a Matryoshka width is its own space"
        );
    }

    #[test]
    fn a_marker_id_maps_to_the_model_its_embedder_reports() {
        assert_eq!(
            model_for_embedder_id("nomic").as_deref(),
            Some(super::super::nomic::MODEL_ID)
        );
        assert_eq!(
            model_for_embedder_id("minilm").as_deref(),
            Some(super::super::minilm::MODEL_ID)
        );
        assert_eq!(
            model_for_embedder_id("unverified:minilm").as_deref(),
            Some(super::super::minilm::MODEL_ID)
        );
        assert_eq!(
            model_for_embedder_id("http://127.0.0.1:8095#text-embedding-nomic-embed-text-v1.5")
                .as_deref(),
            Some("text-embedding-nomic-embed-text-v1.5")
        );
        assert_eq!(model_for_embedder_id("http://127.0.0.1:1234"), None);
        assert_eq!(model_for_embedder_id("zenoh"), None);
    }

    #[test]
    fn the_corpus_space_needs_both_markers() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(corpus_space(dir.path()), None);
        std::fs::write(dir.path().join(DIMENSION_MARKER), "384").unwrap();
        assert_eq!(corpus_space(dir.path()), None, "a dimension names no model");
        std::fs::write(dir.path().join(EMBEDDER_MARKER), "minilm\n").unwrap();
        assert_eq!(
            corpus_space(dir.path()),
            Some(SpaceId::new(super::super::minilm::MODEL_ID, 384))
        );
    }

    #[test]
    fn an_anonymous_embedder_is_named_by_its_id() {
        let anonymous = Named {
            id: "zenoh",
            model: "unknown",
            dim: 384,
        };
        assert_eq!(SpaceId::of(&anonymous).as_str(), "zenoh@384");
        let named = Named {
            id: "http://h#m",
            model: "m",
            dim: 768,
        };
        assert_eq!(SpaceId::for_vector(&named, &[0.0; 1024]).as_str(), "m@1024");
    }

    #[test]
    fn an_absent_space_is_assumed_only_at_the_assumed_width() {
        let nomic = SpaceId::new("nomic-embed-text-v1.5", 768);
        let minilm = SpaceId::new("minilm-l6-v2", 384);
        assert_eq!(resolve(Some(&minilm), 384, Some(&nomic)), minilm);
        assert_eq!(resolve(None, 768, Some(&nomic)), nomic);
        assert_eq!(resolve(None, 384, Some(&nomic)), SpaceId::unrecorded(384));
        assert_eq!(resolve(None, 384, None), SpaceId::unrecorded(384));
        assert!(!SpaceId::unrecorded(384).is_recorded());
    }
}
