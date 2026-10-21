//! Embedding infrastructure (model loading, inference, trait definitions).
//! Embedding generation module
//!
//! Provides semantic embedding generation for memory retrieval.
//! Uses ONNX Runtime with MiniLM-L6-v2 for 384-dimensional embeddings.
//!
//! # Features
//! - **Auto-download**: Model files downloaded on first use to ~/.cache/veld/
//! - **Circuit breaker**: Automatic fallback when ONNX service is degraded
//! - **Lazy loading**: Model loaded on first embed() call, not at startup
//!
//! # Configuration
//! - `VELD_OFFLINE=true` - Disable auto-download
//! - `VELD_AUTO_DOWNLOAD_MODELS=true` - Explicitly allow model/runtime downloads
//! - `VELD_NEURAL_NER=true` - Enable neural NER when local models exist
//! - `VELD_LAZY_LOAD=false` - Load model at startup
//! - `VELD_ONNX_THREADS=N` - Set ONNX intra-op thread count (default: 1 on macOS ARM64,
//!   min(available cores, 8) elsewhere)
//!
//! Model instances are shared process-wide by spec; see [`shared`].

pub mod alignment;
pub mod alignment_affine;
pub mod alignment_conditioning;
pub mod alignment_config;
pub mod alignment_csls;
pub mod alignment_procrustes;
pub mod alignment_ridge;
pub mod alignment_spaces;
pub mod arctic;
pub mod chunking;
pub mod circuit_breaker;
pub mod competitive;
pub mod cross_encoder;
pub mod downloader;
pub mod http_embedder;
pub mod keywords;
pub mod minilm;
pub mod model_tape;
pub mod ner;
pub mod nomic;
pub mod space;
#[cfg(feature = "zenoh")]
pub mod zenoh_embedder;

// Re-export chunking types
pub use chunking::{chunk_text, ChunkConfig, ChunkResult};

use anyhow::Result;

// Re-export downloader functions for convenience
pub use downloader::{
    are_arctic_xs_models_downloaded, are_models_downloaded, are_ner_models_downloaded,
    are_nomic_models_downloaded, download_arctic_xs_models, download_ner_models,
    download_nomic_models, ensure_downloaded, get_arctic_xs_models_dir, get_cache_dir,
    get_models_dir, get_ner_models_dir, get_nomic_models_dir, get_onnx_runtime_path,
    is_onnx_runtime_downloaded, print_status,
};

// Re-export NER types
pub use ner::{NerConfig, NerEntity, NerEntityType, NeuralNer};

// Re-export keyword types
pub use keywords::{Keyword, KeywordConfig, KeywordExtractor};

// Re-export circuit breaker types
pub use circuit_breaker::{
    CircuitBreakerConfig, CircuitBreakerMetrics, CircuitState, ResilientEmbedder,
};

// Re-export competitive embedder
pub use competitive::CompetitiveEmbedder;
pub use space::SpaceId;

pub use minilm::OrtThreading;

// Re-export alignment scaffolding (Phase 1) + Procrustes (Phase 3)
pub use alignment::{
    read_alignment_file, resolve_alignment_path, save_alignment, unix_ts_now, Alignment,
    AlignmentHeader, AlignmentPairId, IdentityAlignment,
};
pub use alignment_procrustes::ProcrustesAlignment;
pub use alignment_ridge::RidgeAlignment;

// Alignment v2: conditioning + whitened-Procrustes affine map, CSLS, and the runtime
// performance-flag config (gated behind AlignmentConfig::v2_enabled).
pub use alignment_affine::AffineAlignment;
pub use alignment_conditioning::Conditioning;
pub use alignment_config::{
    AlignmentConfig, AlignmentMethod, AlignmentObjective, CapabilityState, EvalMetric, WhitenMode,
};
pub use alignment_csls::{csls_adjust, local_means, topk_mean};

/// Load a primary embedder by canonical identifier.
///
/// Dispatch rules:
/// - prefix `nomic` → `NomicEmbedder::new(NomicConfig::from_env())`
/// - prefix `minilm` → `MiniLMEmbedder::new(EmbeddingConfig::default())`
/// - prefix `arctic` → `ArcticEmbedder::new(ArcticConfig::from_env())`
/// - `http://` / `https://` URL → `HttpEmbedder::new` with `base_url = id`
/// - anything else → `bail!`
pub fn load_primary_embedder(id: &str) -> Result<std::sync::Arc<dyn Embedder>> {
    load_embedder_by_id(id)
}

/// Load a secondary embedder by canonical identifier. Same dispatch rules as
/// [`load_primary_embedder`] — the distinction is purely informational at the
/// caller level (which side of the alignment pair we're constructing).
pub fn load_secondary_embedder(id: &str) -> Result<std::sync::Arc<dyn Embedder>> {
    load_embedder_by_id(id)
}

/// The shared instance for `id`, so an alignment fit reuses the model the store already loaded.
fn load_embedder_by_id(id: &str) -> Result<std::sync::Arc<dyn Embedder>> {
    if id.starts_with("nomic") {
        return Ok(shared::nomic_embedder(
            &nomic::NomicConfig::from_env(),
            minilm::OrtThreading::resolve(),
            nomic::pool_size_from_env(),
        )?);
    }
    if id.starts_with("minilm") {
        return Ok(shared::minilm_embedder(
            &minilm::EmbeddingConfig::default(),
            minilm::OrtThreading::resolve(),
            minilm::default_session_pool_size(),
        )?);
    }
    if id.starts_with("arctic") {
        return Ok(shared::arctic_embedder(
            &arctic::ArcticConfig::from_env(),
            minilm::OrtThreading::resolve(),
        )?);
    }
    if id.starts_with("http://") || id.starts_with("https://") {
        // `<url>` or `<url>#<model>` — the form HttpEmbedder::id() emits.
        return Ok(shared::http_embedder(
            &http_embedder::HttpEmbedderConfig::from_id(id),
        ));
    }
    anyhow::bail!(
        "unknown embedder id: {id} (supported prefixes: nomic*, minilm*, arctic*, http://, https://)"
    );
}

/// The primary embedder this process chose, for `/health` and anyone else who needs to know
/// what the vectors in the store were made by. Recorded once at the first `MemorySystem`
/// construction; every user's system reads the same environment, so first-wins is exact.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PrimaryEmbedderInfo {
    /// Canonical id, the value the corpus marker `.veld_embedder` holds.
    pub id: String,
    pub model: String,
    pub dimension: usize,
    /// `nomic-onnx` | `minilm-onnx` | `arctic-onnx` | `http` — where the vectors come from.
    pub kind: &'static str,
    /// Why this one: the env that asked for it, or the fallback step that reached it.
    pub reason: String,
}

static PRIMARY_EMBEDDER: std::sync::OnceLock<PrimaryEmbedderInfo> = std::sync::OnceLock::new();

pub fn record_primary_embedder(info: PrimaryEmbedderInfo) {
    let _ = PRIMARY_EMBEDDER.set(info);
}

pub fn primary_embedder_info() -> Option<PrimaryEmbedderInfo> {
    PRIMARY_EMBEDDER.get().cloned()
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

pub(crate) fn offline_mode_enabled() -> bool {
    env_flag("VELD_OFFLINE", false)
}

pub(crate) fn auto_download_models_enabled() -> bool {
    env_flag("VELD_AUTO_DOWNLOAD_MODELS", false)
}

pub(crate) fn neural_ner_enabled() -> bool {
    env_flag("VELD_NEURAL_NER", false)
}

/// Opt-in to running with NON-SEMANTIC hash embeddings.
///
/// Simplified mode makes `encode()` return hash-derived vectors. Nothing errors,
/// so recall still "works", benchmarks still report numbers, and every one of
/// those numbers is computed against noise. shodh-memory traced an entire voided
/// A/B to exactly this: CI runners failed the model download, vamana recall was
/// all-zero, and the conclusion recorded was "success".
///
/// So the fallback is refused unless it is asked for. It remains genuinely
/// useful for tests and offline smoke runs where nobody is measuring quality —
/// the point is that choosing it is now a decision someone made, not something
/// that happened.
pub(crate) fn allow_simplified_embeddings() -> bool {
    env_flag("VELD_ALLOW_SIMPLIFIED_EMBEDDINGS", false)
}

/// Trait for embedding generation
pub trait Embedder: Send + Sync {
    /// Generate embedding for text.
    ///
    /// This is the *document* side of an asymmetric model — content being
    /// stored/indexed. Symmetric models (e.g. MiniLM) treat documents and
    /// queries identically; see `encode_for_query` for the query side.
    fn encode(&self, text: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Which embedder this IS, as the canonical id `load_embedder_by_id` accepts.
    ///
    /// Required, not defaulted. The trait could report an embedder's SHAPE and not
    /// its identity, which is why the corpus migration below could only key on
    /// dimension — and two different models at the same dimension are the case that
    /// matters, because that is when the marker matches and the migration
    /// short-circuits over a corpus embedded by something else. A default here
    /// would let the next embedder be anonymous and reopen exactly that hole.
    fn id(&self) -> &str;

    /// Encode text as a *query* embedding (retrieval side).
    ///
    /// Asymmetric models (e.g. Nomic, which prepends `search_query: `) override
    /// this to apply the query-specific transform. The default delegates to
    /// `encode`, which is correct for symmetric models. Always use this for
    /// search queries so the document/query asymmetry is honored end-to-end.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        self.encode(text)
    }

    /// Encode text and report whether the result is a degraded fallback.
    /// Returns (embedding, is_degraded). Default: delegates to encode(), reports healthy.
    fn encode_with_status(&self, text: &str) -> Result<(Vec<f32>, bool)> {
        self.encode(text).map(|v| (v, false))
    }

    /// Batch encode multiple texts (default: sequential)
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Canonical, version-pinned identifier for this embedder (e.g.
    /// `"nomic-embed-text-v1.5"`, `"minilm-l6-v2"`). Consulted by the
    /// alignment subsystem to construct [`AlignmentPairId`] and to refuse
    /// loading an alignment fitted on a different pair. Default `"unknown"`
    /// keeps the trait change non-breaking for external implementors; concrete
    /// embedders should override.
    fn model_id(&self) -> &str {
        "unknown"
    }
}

/// One model instance per spec, process-wide.
///
/// Every tenant's `MemorySystem` used to build its own primary embedder and cross-encoder, and
/// the ONNX sessions live on those instances, so model memory and load time grew with the
/// number of cached users and every re-open after an eviction loaded the model again.
/// Instances are now keyed by the values that decide which model a tenant gets and how it is
/// loaded ([`EmbedderSpec`], [`CrossEncoderSpec`]); tenants whose specs are equal hold one `Arc`.
pub mod shared {
    use super::arctic::{ArcticConfig, ArcticEmbedder};
    use super::http_embedder::{HttpEmbedder, HttpEmbedderConfig};
    use super::minilm::{EmbeddingConfig, MiniLMEmbedder, OrtThreading};
    use super::nomic::{NomicConfig, NomicEmbedder};
    use super::Embedder;
    use anyhow::Result;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::sync::{Arc, LazyLock};
    use std::time::{Duration, Instant};

    pub use super::cross_encoder::{cross_encoder_for, CrossEncoderSpec};

    /// How long a failed HTTP probe is believed before the endpoint is asked again: a burst of
    /// tenant opens against a down endpoint costs one probe, and one started afterwards is
    /// found without a restart.
    pub const PROBE_NEGATIVE_TTL: Duration = Duration::from_secs(30);

    /// What selects a primary embedder, as values. Built from the environment and, for a store
    /// whose corpus marker names an HTTP endpoint, from that marker.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub enum EmbedderSpec {
        /// Nomic-embed-text-v1.5 ONNX.
        Nomic {
            config: NomicConfig,
            threading: OrtThreading,
            pool_size: usize,
        },
        /// MiniLM-L6-v2 ONNX.
        MiniLm {
            config: EmbeddingConfig,
            threading: OrtThreading,
            pool_size: usize,
        },
        /// snowflake-arctic-embed-xs ONNX, reached only by `VELD_EMBEDDER=arctic-embed-xs`.
        Arctic {
            config: ArcticConfig,
            threading: OrtThreading,
        },
        /// An OpenAI-compatible endpoint; the config carries its URL and model.
        Http(HttpEmbedderConfig),
    }

    impl EmbedderSpec {
        pub fn nomic_from_env() -> Self {
            Self::Nomic {
                config: NomicConfig::from_env(),
                threading: OrtThreading::resolve(),
                pool_size: super::nomic::pool_size_from_env(),
            }
        }

        pub fn minilm_from_env() -> Self {
            Self::MiniLm {
                config: EmbeddingConfig::from_env(),
                threading: OrtThreading::resolve(),
                pool_size: super::minilm::default_session_pool_size(),
            }
        }

        pub fn arctic_from_env() -> Self {
            Self::Arctic {
                config: ArcticConfig::from_env(),
                threading: OrtThreading::resolve(),
            }
        }
    }

    /// The primary embedder `spec` selects, shared with every tenant whose spec is equal.
    ///
    /// On top of the shared instance, the primary rules: Nomic must run its model (hash mode is
    /// an error, so selection moves to the next candidate), and an HTTP endpoint must pass
    /// [`HttpEmbedder::probe`]. The probe runs once per spec; a failure is returned again
    /// without asking the endpoint until [`PROBE_NEGATIVE_TTL`] has passed.
    pub fn embedder_for(spec: &EmbedderSpec) -> Result<Arc<dyn Embedder>> {
        match spec {
            EmbedderSpec::Nomic {
                config,
                threading,
                pool_size,
            } => {
                let nomic = nomic_embedder(config, *threading, *pool_size)?;
                if nomic.is_simplified() {
                    anyhow::bail!("Nomic ONNX available only in simplified (hash) mode");
                }
                Ok(nomic)
            }
            EmbedderSpec::MiniLm {
                config,
                threading,
                pool_size,
            } => Ok(minilm_embedder(config, *threading, *pool_size)?),
            EmbedderSpec::Arctic { config, threading } => Ok(arctic_embedder(config, *threading)?),
            EmbedderSpec::Http(config) => Ok(HTTP.probed(config)?),
        }
    }

    /// The shared HTTP embedder for `config`, unprobed. The secondary slot asks
    /// [`HttpEmbedder::is_available`], whose cached answer every tenant now shares.
    pub fn http_embedder(config: &HttpEmbedderConfig) -> Arc<HttpEmbedder> {
        HTTP.instance(config)
    }

    /// The shared Nomic instance; hash mode is returned, not refused, and not shared.
    pub(crate) fn nomic_embedder(
        config: &NomicConfig,
        threading: OrtThreading,
        pool_size: usize,
    ) -> Result<Arc<NomicEmbedder>> {
        static NOMIC: LazyLock<Slots<(NomicConfig, OrtThreading, usize), NomicEmbedder>> =
            LazyLock::new(Slots::default);
        NOMIC.reuse_or_build(&(config.clone(), threading, pool_size), || {
            NomicEmbedder::with_runtime(config.clone(), threading, pool_size)
        })
    }

    /// The shared MiniLM instance; hash mode is returned, not shared.
    pub(crate) fn minilm_embedder(
        config: &EmbeddingConfig,
        threading: OrtThreading,
        pool_size: usize,
    ) -> Result<Arc<MiniLMEmbedder>> {
        static MINILM: LazyLock<Slots<(EmbeddingConfig, OrtThreading, usize), MiniLMEmbedder>> =
            LazyLock::new(Slots::default);
        MINILM.reuse_or_build(&(config.clone(), threading, pool_size), || {
            MiniLMEmbedder::with_runtime(config.clone(), threading, pool_size)
        })
    }

    /// The shared arctic-embed-xs instance.
    pub(crate) fn arctic_embedder(
        config: &ArcticConfig,
        threading: OrtThreading,
    ) -> Result<Arc<ArcticEmbedder>> {
        static ARCTIC: LazyLock<Slots<(ArcticConfig, OrtThreading), ArcticEmbedder>> =
            LazyLock::new(Slots::default);
        ARCTIC.reuse_or_build(&(config.clone(), threading), || {
            ArcticEmbedder::with_threading(config.clone(), threading)
        })
    }

    /// What the registry needs to know about an ONNX embedder.
    trait OnnxModel {
        fn is_simplified(&self) -> bool;
        fn model_load_failed(&self) -> bool;
    }

    impl OnnxModel for NomicEmbedder {
        fn is_simplified(&self) -> bool {
            NomicEmbedder::is_simplified(self)
        }
        fn model_load_failed(&self) -> bool {
            NomicEmbedder::model_load_failed(self)
        }
    }

    impl OnnxModel for ArcticEmbedder {
        fn is_simplified(&self) -> bool {
            false
        }
        fn model_load_failed(&self) -> bool {
            ArcticEmbedder::model_load_failed(self)
        }
    }

    impl OnnxModel for MiniLMEmbedder {
        fn is_simplified(&self) -> bool {
            MiniLMEmbedder::is_simplified(self)
        }
        fn model_load_failed(&self) -> bool {
            MiniLMEmbedder::model_load_failed(self)
        }
    }

    /// The shared instance for one key, if built.
    type Slot<E> = Arc<Mutex<Option<Arc<E>>>>;

    /// One slot per key. A slot stays locked while its instance is built, so concurrent opens
    /// of one spec build it once and opens of other specs are not held up.
    struct Slots<K, E> {
        slots: Mutex<HashMap<K, Slot<E>>>,
    }

    impl<K, E> Default for Slots<K, E> {
        fn default() -> Self {
            Self {
                slots: Mutex::new(HashMap::new()),
            }
        }
    }

    impl<K: Eq + Hash + Clone, E: OnnxModel> Slots<K, E> {
        fn reuse_or_build(&self, key: &K, build: impl FnOnce() -> Result<E>) -> Result<Arc<E>> {
            let slot = Arc::clone(self.slots.lock().entry(key.clone()).or_default());
            let mut held = slot.lock();
            if let Some(model) = held.as_ref() {
                if !model.model_load_failed() {
                    return Ok(Arc::clone(model));
                }
                tracing::warn!(
                    "a shared embedder failed to load its model; building a new instance"
                );
            }
            let model = Arc::new(build()?);
            // A hash-mode instance holds no session worth sharing, and keeping it would leave
            // tenants opened after the model files arrive on non-semantic vectors.
            *held = (!model.is_simplified()).then(|| Arc::clone(&model));
            Ok(model)
        }
    }

    static HTTP: LazyLock<HttpEmbedders> = LazyLock::new(|| HttpEmbedders::new(PROBE_NEGATIVE_TTL));

    /// Shared HTTP embedders, one per config, and what their probes found.
    struct HttpEmbedders {
        negative_ttl: Duration,
        slots: Mutex<HashMap<HttpEmbedderConfig, Arc<HttpSlot>>>,
    }

    struct HttpSlot {
        embedder: Arc<HttpEmbedder>,
        /// Held across a probe, so concurrent opens of one endpoint probe it once; the
        /// unprobed [`HttpEmbedders::instance`] never waits on it.
        probe: Mutex<Option<ProbeOutcome>>,
    }

    enum ProbeOutcome {
        Passed,
        Failed { at: Instant, error: String },
    }

    impl HttpEmbedders {
        fn new(negative_ttl: Duration) -> Self {
            Self {
                negative_ttl,
                slots: Mutex::new(HashMap::new()),
            }
        }

        fn slot(&self, config: &HttpEmbedderConfig) -> Arc<HttpSlot> {
            let mut slots = self.slots.lock();
            Arc::clone(slots.entry(config.clone()).or_insert_with(|| {
                Arc::new(HttpSlot {
                    embedder: Arc::new(HttpEmbedder::new(config.clone())),
                    probe: Mutex::new(None),
                })
            }))
        }

        fn instance(&self, config: &HttpEmbedderConfig) -> Arc<HttpEmbedder> {
            Arc::clone(&self.slot(config).embedder)
        }

        fn probed(&self, config: &HttpEmbedderConfig) -> Result<Arc<HttpEmbedder>> {
            let slot = self.slot(config);
            let mut probe = slot.probe.lock();
            match &*probe {
                Some(ProbeOutcome::Passed) => return Ok(Arc::clone(&slot.embedder)),
                Some(ProbeOutcome::Failed { at, error }) if at.elapsed() < self.negative_ttl => {
                    anyhow::bail!(
                        "{error} (the probe failed {}s ago; it is retried {}s after a failure)",
                        at.elapsed().as_secs(),
                        self.negative_ttl.as_secs()
                    );
                }
                _ => {}
            }
            match slot.embedder.probe() {
                Ok(_) => {
                    *probe = Some(ProbeOutcome::Passed);
                    Ok(Arc::clone(&slot.embedder))
                }
                Err(e) => {
                    *probe = Some(ProbeOutcome::Failed {
                        at: Instant::now(),
                        error: format!("{e:#}"),
                    });
                    Err(e)
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::embeddings::test_support::FakeEmbeddingServer;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct Stub {
            simplified: bool,
            failed: AtomicBool,
        }

        impl Stub {
            fn model() -> Result<Self> {
                Ok(Self {
                    simplified: false,
                    failed: AtomicBool::new(false),
                })
            }
        }

        impl OnnxModel for Stub {
            fn is_simplified(&self) -> bool {
                self.simplified
            }
            fn model_load_failed(&self) -> bool {
                self.failed.load(Ordering::SeqCst)
            }
        }

        #[test]
        fn an_onnx_instance_is_built_once_per_key() {
            let slots: Slots<&str, Stub> = Slots::default();
            let first = slots.reuse_or_build(&"a", Stub::model).unwrap();
            let again = slots
                .reuse_or_build(&"a", || panic!("built a second time"))
                .unwrap();
            assert!(Arc::ptr_eq(&first, &again));
            let other = slots.reuse_or_build(&"b", Stub::model).unwrap();
            assert!(!Arc::ptr_eq(&first, &other));
        }

        #[test]
        fn a_failed_build_or_load_is_retried_and_hash_mode_is_never_shared() {
            let slots: Slots<&str, Stub> = Slots::default();
            assert!(slots
                .reuse_or_build(&"k", || anyhow::bail!("no model files"))
                .is_err());
            let built = slots.reuse_or_build(&"k", Stub::model).unwrap();

            built.failed.store(true, Ordering::SeqCst);
            let rebuilt = slots.reuse_or_build(&"k", Stub::model).unwrap();
            assert!(
                !Arc::ptr_eq(&built, &rebuilt),
                "an instance whose load failed is replaced"
            );

            let hash = || {
                Ok(Stub {
                    simplified: true,
                    failed: AtomicBool::new(false),
                })
            };
            let h1 = slots.reuse_or_build(&"hash", hash).unwrap();
            let h2 = slots.reuse_or_build(&"hash", hash).unwrap();
            assert!(!Arc::ptr_eq(&h1, &h2));
        }

        #[test]
        fn one_http_embedder_per_config() {
            let registry = HttpEmbedders::new(PROBE_NEGATIVE_TTL);
            let one = HttpEmbedderConfig::from_id("http://127.0.0.1:9#one");
            let two = HttpEmbedderConfig::from_id("http://127.0.0.1:9#two");
            assert!(Arc::ptr_eq(
                &registry.instance(&one),
                &registry.instance(&one.clone())
            ));
            assert!(!Arc::ptr_eq(
                &registry.instance(&one),
                &registry.instance(&two)
            ));
        }

        #[test]
        fn a_failed_probe_is_believed_for_the_ttl_then_asked_again() {
            let server = FakeEmbeddingServer::start("fake-ttl", 4);
            server.set_failing(true);
            let ttl = Duration::from_millis(300);
            let registry = HttpEmbedders::new(ttl);
            let config = HttpEmbedderConfig::from_id(&server.id());

            assert!(registry.probed(&config).is_err());
            assert_eq!(server.model_listings(), 1);

            server.set_failing(false);
            let Err(cached) = registry.probed(&config) else {
                panic!("a failure inside the TTL must be returned again");
            };
            let cached = format!("{cached:#}");
            assert_eq!(
                server.model_listings(),
                1,
                "inside the TTL a failure is not asked again"
            );
            assert!(
                cached.contains(&server.port().to_string()),
                "the cached error still names the endpoint: {cached}"
            );

            std::thread::sleep(ttl + Duration::from_millis(50));
            let up = registry
                .probed(&config)
                .expect("re-probed once the TTL passed");
            assert_eq!(server.model_listings(), 2);
            assert_eq!(up.dimension(), 4, "the probe measured the dimension");

            let again = registry.probed(&config).unwrap();
            assert!(Arc::ptr_eq(&up, &again));
            assert_eq!(server.model_listings(), 2, "a passed probe is not repeated");
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! A minimal OpenAI-compatible embedding endpoint on loopback, so HTTP embedder selection
    //! is tested without depending on whatever server the machine happens to run.

    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    pub(crate) struct FakeEmbeddingServer {
        port: u16,
        state: Arc<ServerState>,
    }

    struct ServerState {
        model: String,
        dimension: usize,
        listings: AtomicUsize,
        failing: AtomicBool,
        stopped: AtomicBool,
    }

    impl FakeEmbeddingServer {
        /// Serves `model` at `dimension` until dropped.
        pub(crate) fn start(model: &str, dimension: usize) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
            let port = listener.local_addr().expect("local address").port();
            let state = Arc::new(ServerState {
                model: model.to_string(),
                dimension,
                listings: AtomicUsize::new(0),
                failing: AtomicBool::new(false),
                stopped: AtomicBool::new(false),
            });
            let serving = Arc::clone(&state);
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if serving.stopped.load(Ordering::SeqCst) {
                        break;
                    }
                    if let Ok(stream) = stream {
                        let _ = serve(stream, &serving);
                    }
                }
            });
            Self { port, state }
        }

        /// The id an `HttpEmbedder` on this server reports, which a corpus marker holds.
        pub(crate) fn id(&self) -> String {
            format!("http://127.0.0.1:{}#{}", self.port, self.state.model)
        }

        pub(crate) fn port(&self) -> u16 {
            self.port
        }

        /// How many times `GET /v1/models`, the first half of a probe, was asked.
        pub(crate) fn model_listings(&self) -> usize {
            self.state.listings.load(Ordering::SeqCst)
        }

        /// While failing, the model listing answers 503 and every probe fails.
        pub(crate) fn set_failing(&self, failing: bool) {
            self.state.failing.store(failing, Ordering::SeqCst);
        }
    }

    impl Drop for FakeEmbeddingServer {
        fn drop(&mut self) {
            self.state.stopped.store(true, Ordering::SeqCst);
            // Wakes the accept loop so it sees the flag.
            let _ = TcpStream::connect(("127.0.0.1", self.port));
        }
    }

    fn serve(stream: TcpStream, state: &ServerState) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        let (mut length, mut chunked) = (0usize, false);
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header)? == 0 {
                break;
            }
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                let value = value.trim();
                if name.eq_ignore_ascii_case("content-length") {
                    length = value.parse().unwrap_or(0);
                } else if name.eq_ignore_ascii_case("transfer-encoding") {
                    chunked = value.eq_ignore_ascii_case("chunked");
                }
            }
        }
        let body = if chunked {
            read_chunked(&mut reader)?
        } else {
            let mut body = vec![0; length];
            reader.read_exact(&mut body)?;
            body
        };

        let mut parts = request_line.split_whitespace();
        let (status, payload) = match (parts.next(), parts.next()) {
            (Some("GET"), Some("/v1/models")) => {
                state.listings.fetch_add(1, Ordering::SeqCst);
                if state.failing.load(Ordering::SeqCst) {
                    ("503 Service Unavailable", "{}".to_string())
                } else {
                    let listing = serde_json::json!({ "data": [{ "id": state.model }] });
                    ("200 OK", listing.to_string())
                }
            }
            (Some("POST"), Some("/v1/embeddings")) => {
                let request: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                let texts = request["input"].as_array().map_or(1, Vec::len);
                let data: Vec<serde_json::Value> = (0..texts)
                    .map(|i| serde_json::json!({ "embedding": unit_vector(state.dimension, i) }))
                    .collect();
                ("200 OK", serde_json::json!({ "data": data }).to_string())
            }
            (Some("HEAD"), _) => ("200 OK", String::new()),
            _ => ("404 Not Found", "{}".to_string()),
        };
        let mut stream = reader.into_inner();
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{payload}",
            payload.len()
        )?;
        stream.flush()
    }

    fn read_chunked(reader: &mut impl BufRead) -> std::io::Result<Vec<u8>> {
        let mut body = Vec::new();
        loop {
            let mut size_line = String::new();
            reader.read_line(&mut size_line)?;
            let size = size_line
                .trim()
                .split(';')
                .next()
                .and_then(|hex| usize::from_str_radix(hex, 16).ok())
                .unwrap_or(0);
            let mut line_end = String::new();
            if size == 0 {
                reader.read_line(&mut line_end)?;
                return Ok(body);
            }
            let start = body.len();
            body.resize(start + size, 0);
            reader.read_exact(&mut body[start..])?;
            reader.read_line(&mut line_end)?;
        }
    }

    fn unit_vector(dimension: usize, salt: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dimension)
            .map(|j| ((j + salt + 1) as f32).sin())
            .collect();
        let norm = raw
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        raw.iter().map(|x| x / norm).collect()
    }
}
