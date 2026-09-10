//! HTTP-backed embedding client for OpenAI-compatible embedding APIs.
//!
//! Supports any host declared in `manifests/llm-hosts.json` implementing the
//! `/v1/embeddings` endpoint: LM Studio (:1234), `llama-server --embedding`
//! (:8080), llamafile, LocalAI, vLLM. Eligible as the PRIMARY embedder (see
//! `MemorySystem` construction) when no local ONNX model is present or when
//! `VELD_EMBEDDER=http` / a URL asks for it; otherwise the secondary slot.
//!
//! Configuration via environment variables:
//! - VELD_EMBEDDING_API_URL: Base URL (default: http://127.0.0.1:1234)
//! - VELD_EMBEDDING_API_MODEL: Model name (default: text-embedding-nomic-embed-text-v1.5)
//! - VELD_EMBEDDING_API_KEY: Optional API key (default: none)
//! - VELD_EMBEDDING_API_TIMEOUT_MS: per-request timeout (default: 30000 — a cold
//!   model load on LM Studio's JIT path is the slow case, not the embed itself)
//! - VELD_EMBEDDING_API_TASK_PREFIXES: 1/0 to force nomic's `search_document: ` /
//!   `search_query: ` task prefixes on or off (default: on when the model name
//!   contains "nomic", which needs them and silently degrades without them)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::Embedder;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234";
const DEFAULT_MODEL: &str = "text-embedding-nomic-embed-text-v1.5";
const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const AVAILABILITY_CACHE_TTL: Duration = Duration::from_secs(30);
/// nomic-embed-text is asymmetric: the same prefixes its ONNX path uses (`nomic.rs`).
const SEARCH_DOCUMENT_PREFIX: &str = "search_document: ";
const SEARCH_QUERY_PREFIX: &str = "search_query: ";

/// Whether a model wants nomic's task prefixes, judged from its name.
pub fn default_task_prefixes(model: &str) -> bool {
    model.to_ascii_lowercase().contains("nomic")
}

/// Configuration for HTTP embedding API
#[derive(Debug, Clone)]
pub struct HttpEmbedderConfig {
    /// Base URL of the embedding API
    pub base_url: String,
    /// Model name to pass in the request
    pub model: String,
    /// Optional API key
    pub api_key: Option<String>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Prepend nomic's `search_document: ` / `search_query: ` task prefixes.
    pub task_prefixes: bool,
}

impl Default for HttpEmbedderConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HttpEmbedderConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        // An insecure plain-http override to a non-localhost host is rejected
        // (fall back to the localhost default) because the embedder POSTs user
        // text to that URL. Opt back in with VELD_ALLOW_INSECURE_REMOTE_EMBEDDER=true.
        let mut base_url =
            crate::integrations::resolve_api_url_override("VELD_EMBEDDING_API_URL", DEFAULT_BASE_URL);
        if crate::integrations::is_insecure_remote_url(&base_url)
            && !std::env::var("VELD_ALLOW_INSECURE_REMOTE_EMBEDDER")
                .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
                .unwrap_or(false)
        {
            tracing::warn!(
                "VELD_EMBEDDING_API_URL points to insecure remote HTTP; using localhost default. \
                 Set VELD_ALLOW_INSECURE_REMOTE_EMBEDDER=true to override."
            );
            base_url = DEFAULT_BASE_URL.to_string();
        }
        let model = std::env::var("VELD_EMBEDDING_API_MODEL")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let task_prefixes = std::env::var("VELD_EMBEDDING_API_TASK_PREFIXES")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or_else(|| default_task_prefixes(&model));
        Self {
            base_url,
            model,
            api_key: std::env::var("VELD_EMBEDDING_API_KEY").ok(),
            timeout_ms: std::env::var("VELD_EMBEDDING_API_TIMEOUT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_TIMEOUT_MS),
            task_prefixes,
        }
    }

    /// Configuration from a canonical id, `<base_url>` or `<base_url>#<model>` — the form
    /// [`HttpEmbedder::id`] emits and `load_embedder_by_id` accepts. Everything not in the id
    /// comes from the environment. The model part re-derives `task_prefixes` unless the env
    /// forced it, so an id naming a nomic model gets nomic's prefixes.
    pub fn from_id(id: &str) -> Self {
        let mut cfg = Self::from_env();
        let (url, model) = match id.split_once('#') {
            Some((u, m)) if !m.is_empty() => (u, Some(m)),
            _ => (id, None),
        };
        cfg.base_url = url.trim_end_matches('/').to_string();
        if let Some(m) = model {
            cfg.model = m.to_string();
            if std::env::var("VELD_EMBEDDING_API_TASK_PREFIXES").is_err() {
                cfg.task_prefixes = default_task_prefixes(&cfg.model);
            }
        }
        cfg
    }
}

/// OpenAI-compatible embedding request
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

/// OpenAI-compatible embedding response
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Batch request
#[derive(Serialize)]
struct BatchEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

/// HTTP-backed embedder using OpenAI-compatible /v1/embeddings endpoint.
///
/// Connects to any declared OpenAI-compatible server — see `crate::llm_hosts`.
/// Falls back gracefully if server is unreachable.
pub struct HttpEmbedder {
    config: HttpEmbedderConfig,
    client: ureq::Agent,
    /// `<base_url>#<model>`: the URL alone is not an identity — one server can serve
    /// several models at one dimension, which is exactly the swap the corpus marker
    /// (`retrieval.rs`) exists to catch and could not with a URL-only id.
    canonical_id: String,
    /// Cached dimension from first successful call
    cached_dimension: std::sync::OnceLock<usize>,
    availability_cache: Mutex<Option<(Instant, bool)>>,
}

/// What `GET /v1/models` lists (OpenAI shape).
#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

impl HttpEmbedder {
    /// Create a new HTTP embedder.
    pub fn new(config: HttpEmbedderConfig) -> Self {
        let client: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_millis(config.timeout_ms)))
            .build()
            .into();

        let canonical_id = format!("{}#{}", config.base_url.trim_end_matches('/'), config.model);
        Self {
            config,
            client,
            canonical_id,
            cached_dimension: std::sync::OnceLock::new(),
            availability_cache: Mutex::new(None),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// The text actually sent for a DOCUMENT (stored content).
    fn document_text<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.config.task_prefixes && !text.starts_with(SEARCH_DOCUMENT_PREFIX) {
            std::borrow::Cow::Owned(format!("{SEARCH_DOCUMENT_PREFIX}{text}"))
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    /// The text actually sent for a QUERY (retrieval side).
    fn query_text<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.config.task_prefixes && !text.starts_with(SEARCH_QUERY_PREFIX) {
            std::borrow::Cow::Owned(format!("{SEARCH_QUERY_PREFIX}{text}"))
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    /// Prove the endpoint can serve THIS model, and learn the dimension, before anything is
    /// indexed against it. Two checks: `/v1/models` must list the model when the server
    /// offers a listing (a server that lists other models and not this one would otherwise
    /// answer the embed with whatever it loads by default), then one real embed returns the
    /// vector whose length becomes the reported dimension. Sends one short fixed string.
    pub fn probe(&self) -> Result<usize> {
        let models_url = format!("{}/v1/models", self.config.base_url);
        let mut builder = self.client.get(&models_url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }
        match builder.call() {
            Ok(resp) => {
                if let Ok(list) = resp.into_body().read_json::<ModelList>() {
                    if !list.data.iter().any(|m| m.id == self.config.model) {
                        let seen: Vec<&str> = list.data.iter().map(|m| m.id.as_str()).collect();
                        anyhow::bail!(
                            "{} does not list model {:?} (it lists {:?})",
                            models_url,
                            self.config.model,
                            seen
                        );
                    }
                }
            }
            Err(e) => anyhow::bail!("{models_url} unreachable: {e}"),
        }
        let v = self
            .encode_http(&self.document_text("veld embedder probe"))
            .with_context(|| format!("{}/v1/embeddings with model {:?}", self.config.base_url, self.config.model))?;
        if v.is_empty() {
            anyhow::bail!("{} returned an empty vector for model {:?}", self.config.base_url, self.config.model);
        }
        let _ = self.cached_dimension.set(v.len());
        Ok(v.len())
    }

    /// Check if the embedding server is reachable without sending billable text.
    pub fn is_available(&self) -> bool {
        if let Some((checked_at, available)) = *self.availability_cache.lock().unwrap() {
            if checked_at.elapsed() < AVAILABILITY_CACHE_TTL {
                return available;
            }
        }

        let mut builder = self.client.head(&self.config.base_url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }
        let available = builder.call().is_ok();
        *self.availability_cache.lock().unwrap() = Some((Instant::now(), available));
        available
    }

    /// Encode a single text via the HTTP API.
    fn encode_http(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.config.base_url);
        let req = EmbeddingRequest {
            model: &self.config.model,
            input: text,
        };

        let mut builder = self.client.post(&url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }

        let resp = builder.send_json(&req).context("HTTP embedding request failed")?;

        let parsed: EmbeddingResponse = resp
            .into_body()
            .read_json()
            .context("Failed to parse embedding response")?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("Empty embedding response")
    }
}

impl Embedder for HttpEmbedder {
    // `<base_url>#<model>` — `load_embedder_by_id` dispatches on the `http://`/`https://`
    // prefix and `HttpEmbedderConfig::from_id` splits the model back out.
    fn id(&self) -> &str {
        &self.canonical_id
    }

    fn model_id(&self) -> &str {
        &self.config.model
    }

    /// Document side: nomic's `search_document: ` prefix when the model wants it.
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.encode_http(&self.document_text(text))?;
        // Cache the dimension on first successful call
        let _ = self.cached_dimension.set(embedding.len());
        Ok(embedding)
    }

    /// Query side: nomic's `search_query: ` prefix when the model wants it.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.encode_http(&self.query_text(text))?;
        let _ = self.cached_dimension.set(embedding.len());
        Ok(embedding)
    }

    /// Measured by `probe()` or the first successful call; 768 (nomic's width) until then.
    fn dimension(&self) -> usize {
        *self.cached_dimension.get().unwrap_or(&768)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Use batch endpoint if available
        let url = format!("{}/v1/embeddings", self.config.base_url);
        let prefixed: Vec<std::borrow::Cow<'_, str>> = texts.iter().map(|t| self.document_text(t)).collect();
        let req = BatchEmbeddingRequest {
            model: &self.config.model,
            input: prefixed.iter().map(|c| c.as_ref()).collect(),
        };

        let mut builder = self.client.post(&url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }

        let resp = match builder.send_json(&req) {
            Ok(resp) => resp,
            Err(_) => return texts.iter().map(|t| self.encode(t)).collect(),
        };

        let parsed: EmbeddingResponse = resp
            .into_body()
            .read_json()
            .context("Failed to parse batch embedding response")?;

        if parsed.data.len() != texts.len() {
            // Batch size mismatch — fall back to sequential
            return texts.iter().map(|t| self.encode(t)).collect();
        }

        let embeddings: Vec<Vec<f32>> = parsed.data.into_iter().map(|d| d.embedding).collect();
        if let Some(first) = embeddings.first() {
            let _ = self.cached_dimension.set(first.len());
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(model: &str) -> HttpEmbedderConfig {
        HttpEmbedderConfig {
            base_url: "http://localhost:9".into(), // unreachable
            model: model.into(),
            api_key: None,
            timeout_ms: 100,
            task_prefixes: default_task_prefixes(model),
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = HttpEmbedderConfig::from_env();
        assert!(config.base_url.contains("127.0.0.1"));
        assert!(config.model.contains("nomic"));
        assert!(config.task_prefixes, "a nomic model gets its task prefixes by default");
    }

    #[test]
    fn test_dimension_default() {
        assert_eq!(HttpEmbedder::new(cfg("test")).dimension(), 768);
    }

    /// The identity carries the model: one server can serve two models at one width, and
    /// the corpus marker keys on this string.
    #[test]
    fn id_names_url_and_model_and_round_trips_through_from_id() {
        let e = HttpEmbedder::new(cfg("text-embedding-nomic-embed-text-v1.5"));
        assert_eq!(e.id(), "http://localhost:9#text-embedding-nomic-embed-text-v1.5");
        let back = HttpEmbedderConfig::from_id(e.id());
        assert_eq!(back.base_url, "http://localhost:9");
        assert_eq!(back.model, "text-embedding-nomic-embed-text-v1.5");
        assert!(back.task_prefixes);
        let bare = HttpEmbedderConfig::from_id("http://localhost:8080/");
        assert_eq!(bare.base_url, "http://localhost:8080");
    }

    /// nomic silently degrades without its task prefixes; a non-nomic model must not get them.
    #[test]
    fn task_prefixes_follow_the_model_and_are_not_doubled() {
        let nomic = HttpEmbedder::new(cfg("text-embedding-nomic-embed-text-v1.5"));
        assert_eq!(nomic.document_text("hello"), "search_document: hello");
        assert_eq!(nomic.query_text("hello"), "search_query: hello");
        assert_eq!(nomic.document_text("search_document: hello"), "search_document: hello");

        let other = HttpEmbedder::new(cfg("bge-m3"));
        assert_eq!(other.document_text("hello"), "hello");
        assert_eq!(other.query_text("hello"), "hello");
    }

    /// An unreachable endpoint fails the probe with the URL in the error, never a guess.
    #[test]
    fn probe_refuses_an_unreachable_endpoint() {
        let e = HttpEmbedder::new(cfg("test"));
        let err = e.probe().unwrap_err().to_string();
        assert!(err.contains("localhost:9"), "{err}");
    }
}
