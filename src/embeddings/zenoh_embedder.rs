//! Roots layer: Zenoh-routed embedder for cluster peer embedding requests.
//!
//! Routes embedding requests to peer veld instances over Zenoh. Used as a
//! secondary embedder when the local node lacks a model but a cluster peer
//! has one available.
//!
//! # Configuration
//! ```text
//! VELD_ZENOH_EMBED_ENABLED=true          # Enable Zenoh-routed embedding
//! VELD_ZENOH_EMBED_PREFIX=veld           # Key expression prefix (default: veld)
//! VELD_ZENOH_EMBED_MODEL=minilm-l6-v2    # Target model on the peer
//! VELD_ZENOH_EMBED_DIMENSION=384         # Expected output dimension
//! VELD_ZENOH_EMBED_TIMEOUT_MS=5000       # Request timeout
//! VELD_ZENOH_CONNECT=tcp/1.2.3.4:7447    # Connect endpoints (comma-separated)
//! ```

#[cfg(feature = "zenoh")]
use anyhow::{Context, Result};

#[cfg(feature = "zenoh")]
use super::Embedder;

/// Configuration for the Zenoh-routed embedder.
#[cfg(feature = "zenoh")]
#[derive(Debug, Clone)]
pub struct ZenohEmbedderConfig {
    /// Key expression prefix matching the serving node.
    pub prefix: String,
    /// Target model name to request from the peer.
    pub target_model: String,
    /// Expected embedding dimension.
    pub dimension: usize,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Zenoh connect endpoints (empty = multicast discovery).
    pub connect_endpoints: Vec<String>,
    /// API key for authenticated clusters.
    pub api_key: Option<String>,
}

#[cfg(feature = "zenoh")]
impl ZenohEmbedderConfig {
    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        Self {
            prefix: crate::config::env_var("VELD_ZENOH_EMBED_PREFIX", "SHODH_ZENOH_EMBED_PREFIX")
                .unwrap_or_else(|_| "veld".into()),
            target_model: crate::config::env_var("VELD_ZENOH_EMBED_MODEL", "SHODH_ZENOH_EMBED_MODEL")
                .unwrap_or_else(|_| "minilm-l6-v2".into()),
            dimension: crate::config::env_var("VELD_ZENOH_EMBED_DIMENSION", "SHODH_ZENOH_EMBED_DIMENSION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(384),
            timeout_ms: crate::config::env_var("VELD_ZENOH_EMBED_TIMEOUT_MS", "SHODH_ZENOH_EMBED_TIMEOUT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5000),
            connect_endpoints: crate::config::env_var("VELD_ZENOH_CONNECT", "SHODH_ZENOH_CONNECT")
                .map(|s| {
                    s.split(',')
                        .map(|e| e.trim().to_string())
                        .filter(|e| !e.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            api_key: crate::config::env_var("VELD_ZENOH_API_KEY", "SHODH_ZENOH_API_KEY").ok(),
        }
    }
}

/// Embed response from the serving peer.
#[cfg(feature = "zenoh")]
#[derive(serde::Deserialize)]
struct PeerEmbedResponse {
    embeddings: Vec<Vec<f32>>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    dimension: usize,
}

/// Validate that all embeddings match the expected dimension.
///
/// Returns an error naming the first offending index and both the actual
/// and expected dimensions. Extracted so it can be tested independently
/// of a live Zenoh session.
#[cfg(feature = "zenoh")]
pub fn validate_embeddings(embeddings: &[Vec<f32>], expected_dim: usize) -> anyhow::Result<()> {
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != expected_dim {
            anyhow::bail!(
                "Peer returned {}-dim embedding at index {}, expected {}",
                emb.len(),
                i,
                expected_dim
            );
        }
    }
    Ok(())
}

/// Zenoh-routed embedder that forwards embedding requests to cluster peers.
///
/// Implements the synchronous `Embedder` trait by bridging to async Zenoh
/// operations via a dedicated single-threaded tokio runtime. This avoids
/// the "cannot block from within a runtime" panic that occurs when
/// `Handle::block_on` is called from inside an existing tokio context.
#[cfg(feature = "zenoh")]
pub struct ZenohEmbedder {
    session: zenoh::Session,
    config: ZenohEmbedderConfig,
    /// Dedicated runtime for blocking on async Zenoh operations.
    /// Separate from the main application runtime to avoid deadlocks.
    runtime: tokio::runtime::Runtime,
}

#[cfg(feature = "zenoh")]
impl ZenohEmbedder {
    /// Create a new embedder using an existing Zenoh session.
    ///
    /// Reuses the provided session (Zenoh sessions are internally Arc'd, so
    /// cloning is cheap). A dedicated single-threaded tokio runtime is still
    /// created for sync→async bridging.
    pub fn new_with_session(config: ZenohEmbedderConfig, session: zenoh::Session) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Failed to create ZenohEmbedder runtime")?;
        Ok(Self { session, config, runtime })
    }

    /// Open a Zenoh session and create a new embedder (standalone mode).
    ///
    /// Creates a dedicated single-threaded tokio runtime for async Zenoh
    /// operations. Safe to call from any context (async or sync).
    /// Prefer [`new_with_session`] when a transport session is already available.
    pub fn new(config: ZenohEmbedderConfig) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Failed to create ZenohEmbedder runtime")?;

        let connect_endpoints = config.connect_endpoints.clone();
        let session = runtime.block_on(async {
            let mut zenoh_config = zenoh::Config::default();
            zenoh_config
                .insert_json5("mode", r#""peer""#)
                .map_err(|e| anyhow::anyhow!("Failed to set Zenoh mode: {e}"))?;

            if !connect_endpoints.is_empty() {
                let endpoints_json = serde_json::to_string(&connect_endpoints)?;
                zenoh_config
                    .insert_json5("connect/endpoints", &endpoints_json)
                    .map_err(|e| anyhow::anyhow!("Failed to set connect endpoints: {e}"))?;
            }

            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                zenoh::open(zenoh_config),
            ).await {
                Ok(result) => result.map_err(|e| anyhow::anyhow!("Failed to open Zenoh session: {e}")),
                Err(_) => anyhow::bail!("Zenoh session open timed out after 10s"),
            }
        }).context("Failed to open Zenoh session for embedding routing")?;

        Ok(Self { session, config, runtime })
    }

    /// Check if any peers with embedding capability are reachable.
    ///
    /// Queries `{prefix}/capabilities` and checks if any peer advertises the
    /// target model.
    pub fn is_available(&self) -> bool {
        let key = format!("{}/capabilities", self.config.prefix);
        let target_model = self.config.target_model.clone();
        let timeout = std::time::Duration::from_millis(2000);

        self.runtime.block_on(async {
            let receiver = match self.session.get(&key).await {
                Ok(r) => r,
                Err(_) => return false,
            };

            let deadline = tokio::time::Instant::now() + timeout;
            loop {
                match tokio::time::timeout_at(deadline, receiver.recv_async()).await {
                    Ok(Ok(reply)) => {
                        if let Ok(sample) = reply.into_result() {
                            if let Ok(text) = sample.payload().try_to_string() {
                                if let Ok(caps) = serde_json::from_str::<
                                    crate::zenoh_transport::config::InstanceCapabilities,
                                >(&text)
                                {
                                    if caps.accepts_embedding_requests
                                        && caps
                                            .embedding_models
                                            .iter()
                                            .any(|m| m.name == target_model)
                                    {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                    Ok(Err(_)) => break, // Channel closed — no more replies
                    Err(_) => break,     // Timeout
                }
            }
            false
        })
    }

    /// Send an embed request to a peer and parse the response.
    ///
    /// Collects replies for up to 500 ms (or until 3 valid replies arrive),
    /// then picks the best one: dimension match is required, ties broken by
    /// response time (fastest wins). Returns an error if no valid reply
    /// arrives before the overall request timeout.
    fn query_peer(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let key_expr = format!("{}/embed", self.config.prefix);
        let mut request_body = serde_json::json!({
            "texts": texts,
            "model": self.config.target_model,
        });
        if let Some(ref api_key) = self.config.api_key {
            request_body["api_key"] = serde_json::Value::String(api_key.clone());
        }
        let payload = serde_json::to_string(&request_body)?;
        let overall_timeout = std::time::Duration::from_millis(self.config.timeout_ms);
        let collect_window = std::time::Duration::from_millis(500);
        let expected_dim = self.config.dimension;

        self.runtime.block_on(async {
            let query_start = tokio::time::Instant::now();

            let receiver = self
                .session
                .get(&key_expr)
                .payload(payload)
                .await
                .map_err(|e| anyhow::anyhow!("Zenoh embed query failed: {e}"))?;

            let overall_deadline = query_start + overall_timeout;
            // Collect window is bounded by the overall timeout.
            let collect_deadline =
                (query_start + collect_window).min(overall_deadline);

            // (embeddings, elapsed_since_query_start)
            let mut best: Option<(Vec<Vec<f32>>, std::time::Duration)> = None;
            let mut valid_count: u32 = 0;
            const MAX_CANDIDATES: u32 = 3;

            loop {
                if valid_count >= MAX_CANDIDATES {
                    break;
                }

                let now = tokio::time::Instant::now();
                if now >= collect_deadline {
                    break;
                }

                match tokio::time::timeout_at(collect_deadline, receiver.recv_async()).await {
                    Ok(Ok(reply)) => {
                        let elapsed = query_start.elapsed();

                        let sample = match reply.into_result() {
                            Ok(s) => s,
                            Err(_) => continue, // peer-reported error — try next reply
                        };

                        let text = match sample.payload().try_to_string() {
                            Ok(t) => t,
                            Err(_) => continue, // not UTF-8 — skip
                        };

                        // Skip explicit error responses rather than failing fast.
                        if let Ok(err_val) = serde_json::from_str::<serde_json::Value>(&text) {
                            if err_val.get("error").and_then(|v| v.as_str()).is_some() {
                                continue;
                            }
                        }

                        let response: PeerEmbedResponse =
                            match serde_json::from_str::<PeerEmbedResponse>(&text) {
                                Ok(r) => r,
                                Err(_) => continue, // malformed — skip
                            };

                        // Dimension validation: skip mismatched peers.
                        if validate_embeddings(&response.embeddings, expected_dim).is_err() {
                            continue;
                        }

                        // Keep this reply if it is faster than the current best.
                        let is_better = best
                            .as_ref()
                            .map(|(_, prev_elapsed)| elapsed < *prev_elapsed)
                            .unwrap_or(true);

                        if is_better {
                            best = Some((response.embeddings, elapsed));
                        }
                        valid_count += 1;
                    }
                    Ok(Err(_)) => break, // channel closed — no more replies
                    Err(_) => break,     // collect window expired
                }
            }

            match best {
                Some((embeddings, _)) => Ok(embeddings),
                None => {
                    // No valid reply collected — check whether we already hit the
                    // overall timeout or simply found no matching peer.
                    if query_start.elapsed() >= overall_timeout {
                        anyhow::bail!(
                            "Embed request timed out after {}ms",
                            self.config.timeout_ms
                        );
                    }
                    anyhow::bail!(
                        "No peer responded with valid {}-dim embeddings",
                        expected_dim
                    );
                }
            }
        })
    }
}

#[cfg(feature = "zenoh")]
impl Embedder for ZenohEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.query_peer(&[text])?;
        embeddings
            .into_iter()
            .next()
            .context("Peer returned empty embeddings for single text")
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn encode_with_status(&self, text: &str) -> Result<(Vec<f32>, bool)> {
        self.encode(text).map(|v| (v, true)) // true = remote/degraded
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.query_peer(texts)
    }
}
