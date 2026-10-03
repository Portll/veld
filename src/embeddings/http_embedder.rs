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
//! - VELD_EMBEDDING_API_MAX_TOKENS: the model's input window in tokens (default: 2048 for
//!   nomic, 8192 for bge-m3, none for other models). A text longer than the window is embedded
//!   in pieces and given the mean of their vectors, because the server would otherwise keep
//!   only its opening; LM Studio does so silently.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::chunking::{chunk_text, ChunkConfig};
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

/// The input window a model was trained on, judged from its name: nomic-embed-text v1.5 was
/// trained at 2048 tokens and degrades past it even where a server accepts more; bge-m3 at 8192.
pub fn default_max_tokens(model: &str) -> Option<usize> {
    let model = model.to_ascii_lowercase();
    if model.contains("nomic") {
        Some(2048)
    } else if model.contains("bge-m3") {
        Some(8192)
    } else {
        None
    }
}

/// Room left in the window for the special tokens a server adds and for a task prefix.
const WINDOW_MARGIN: usize = 8;
/// Bytes per token assumed where the server cannot count: this store's densest text (paths,
/// SHAs, JSON) runs under 2 characters per token (`chunking.rs`), so 2 never overruns.
const BYTES_PER_TOKEN_FLOOR: usize = 2;
/// How many times a piece that still does not fit is split again before it is sent as it is.
const MAX_SPLIT_DEPTH: usize = 4;

/// The mean of unit-length vectors, scaled to unit length.
fn mean_normalised(vectors: &[Vec<f32>]) -> Vec<f32> {
    let width = vectors.first().map_or(0, Vec::len);
    let mut sum = vec![0f32; width];
    for v in vectors {
        let norm = v
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        for (s, x) in sum.iter_mut().zip(v) {
            *s += x / norm;
        }
    }
    let norm = sum
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    sum.iter().map(|x| x / norm).collect()
}

/// `text` cut on `chunk_text`'s boundaries into pieces that each pass `fits`, a piece that does
/// not being cut again at half the size, up to `MAX_SPLIT_DEPTH` times.
fn pieces(text: &str, piece_bytes: usize, fits: &dyn Fn(&str) -> bool) -> Vec<String> {
    fn split(
        text: &str,
        size: usize,
        depth: usize,
        fits: &dyn Fn(&str) -> bool,
        out: &mut Vec<String>,
    ) {
        let config = ChunkConfig {
            chunk_size: size,
            overlap: size / 8,
            min_chunk_size: size / 4,
        };
        for chunk in chunk_text(text, &config).chunks {
            if depth < MAX_SPLIT_DEPTH && size > 64 && !fits(&chunk) {
                split(&chunk, size / 2, depth + 1, fits, out);
            } else {
                out.push(chunk);
            }
        }
    }
    let mut out = Vec::new();
    split(text, piece_bytes.max(64), 0, fits, &mut out);
    out
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
    /// The model's input window in tokens; a longer text is embedded in pieces. None: no limit.
    pub max_input_tokens: Option<usize>,
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
        let mut base_url = crate::integrations::resolve_api_url_override(
            "VELD_EMBEDDING_API_URL",
            DEFAULT_BASE_URL,
        );
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
        let max_input_tokens = Self::max_tokens_env().unwrap_or_else(|| default_max_tokens(&model));
        Self {
            base_url,
            model,
            api_key: std::env::var("VELD_EMBEDDING_API_KEY").ok(),
            timeout_ms: std::env::var("VELD_EMBEDDING_API_TIMEOUT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_TIMEOUT_MS),
            task_prefixes,
            max_input_tokens,
        }
    }

    /// `VELD_EMBEDDING_API_MAX_TOKENS` when set: a count, or 0 for no limit.
    fn max_tokens_env() -> Option<Option<usize>> {
        let n: usize = std::env::var("VELD_EMBEDDING_API_MAX_TOKENS")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        Some((n > 0).then_some(n))
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
            if Self::max_tokens_env().is_none() {
                cfg.max_input_tokens = default_max_tokens(&cfg.model);
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
    /// Whether the server answers `POST /tokenize` (llama-server does, LM Studio does not),
    /// learned on the first text long enough to need counting.
    tokenizes: std::sync::OnceLock<bool>,
}

/// llama-server's `/tokenize` request and response.
#[derive(Serialize)]
struct TokenizeRequest<'a> {
    content: &'a str,
}

#[derive(Deserialize)]
struct TokenizeResponse {
    tokens: Vec<serde_json::Value>,
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
            tokenizes: std::sync::OnceLock::new(),
        }
    }

    /// The server's token count for `text`, or None where it cannot count.
    fn count_tokens(&self, text: &str) -> Option<usize> {
        if self.tokenizes.get() == Some(&false) {
            return None;
        }
        let mut builder = self
            .client
            .post(&format!("{}/tokenize", self.config.base_url));
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }
        match builder.send_json(&TokenizeRequest { content: text }) {
            Ok(resp) => {
                let counted = resp
                    .into_body()
                    .read_json::<TokenizeResponse>()
                    .ok()
                    .map(|r| r.tokens.len());
                let _ = self.tokenizes.set(counted.is_some());
                counted
            }
            // A server without the endpoint says so; anything else may pass, and is not remembered.
            Err(ureq::Error::StatusCode(404 | 405 | 501)) => {
                let _ = self.tokenizes.set(false);
                None
            }
            Err(_) => None,
        }
    }

    /// Whether `sent`, the text as it goes to the server, fits the model's window. A text with
    /// fewer bytes than the window always fits, since every token covers at least one byte.
    fn fits(&self, sent: &str) -> bool {
        let Some(window) = self.config.max_input_tokens else {
            return true;
        };
        let room = window.saturating_sub(WINDOW_MARGIN);
        if sent.len() <= room {
            return true;
        }
        match self.count_tokens(sent) {
            Some(n) => n <= room,
            None => sent.len() / BYTES_PER_TOKEN_FLOOR <= room,
        }
    }

    /// `text` embedded with `prefixed` applied: whole when it fits the window, otherwise as the
    /// mean of pieces that each fit.
    fn encode_windowed(&self, text: &str, prefixed: impl Fn(&str) -> String) -> Result<Vec<f32>> {
        let sent = prefixed(text);
        if self.fits(&sent) {
            return self.encode_http(&sent);
        }
        let room = self
            .config
            .max_input_tokens
            .unwrap_or(usize::MAX)
            .saturating_sub(WINDOW_MARGIN);
        let split = pieces(text, room.saturating_mul(BYTES_PER_TOKEN_FLOOR), &|p| {
            self.fits(&prefixed(p))
        });
        tracing::debug!(
            "embedding {} bytes in {} pieces: longer than the {}-token window",
            text.len(),
            split.len(),
            room + WINDOW_MARGIN
        );
        let vectors = split
            .iter()
            .map(|p| self.encode_http(&prefixed(p)))
            .collect::<Result<Vec<_>>>()?;
        Ok(mean_normalised(&vectors))
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
            .with_context(|| {
                format!(
                    "{}/v1/embeddings with model {:?}",
                    self.config.base_url, self.config.model
                )
            })?;
        if v.is_empty() {
            anyhow::bail!(
                "{} returned an empty vector for model {:?}",
                self.config.base_url,
                self.config.model
            );
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

        let resp = builder
            .send_json(&req)
            .context("HTTP embedding request failed")?;

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
        let embedding = self.encode_windowed(text, |t| self.document_text(t).into_owned())?;
        // Cache the dimension on first successful call
        let _ = self.cached_dimension.set(embedding.len());
        Ok(embedding)
    }

    /// Query side: nomic's `search_query: ` prefix when the model wants it.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.encode_windowed(text, |t| self.query_text(t).into_owned())?;
        let _ = self.cached_dimension.set(embedding.len());
        Ok(embedding)
    }

    /// Measured by `probe()` or the first successful call; 768 (nomic's width) until then.
    fn dimension(&self) -> usize {
        *self.cached_dimension.get().unwrap_or(&768)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // A text longer than the window is embedded in pieces, so the batch goes one by one.
        if texts.iter().any(|t| !self.fits(&self.document_text(t))) {
            return texts.iter().map(|t| self.encode(t)).collect();
        }
        // Use batch endpoint if available
        let url = format!("{}/v1/embeddings", self.config.base_url);
        let prefixed: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| self.document_text(t)).collect();
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
            max_input_tokens: default_max_tokens(model),
        }
    }

    #[test]
    fn the_window_follows_the_model() {
        assert_eq!(
            default_max_tokens("text-embedding-nomic-embed-text-v1.5"),
            Some(2048)
        );
        assert_eq!(default_max_tokens("bge-m3"), Some(8192));
        assert_eq!(default_max_tokens("some-other-model"), None);
    }

    #[test]
    fn the_mean_of_pieces_is_unit_length_and_between_them() {
        let m = mean_normalised(&[vec![2.0, 0.0], vec![0.0, 3.0]]);
        assert!((m.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((m[0] - m[1]).abs() < 1e-6, "{m:?}");
    }

    /// Each piece fits, and together they reach from the text's start to its end.
    #[test]
    fn pieces_each_fit_and_cover_the_text() {
        let text: String = (0..3000).map(|i| format!("w{i} ")).collect();
        let fits = |p: &str| p.split_whitespace().count() <= 300;
        let split = pieces(&text, 4000, &fits);
        assert!(split.len() > 1);
        assert!(split.iter().all(|p| fits(p)), "a piece over the window");
        assert!(split.first().unwrap().starts_with("w0 "));
        assert!(split.last().unwrap().trim_end().ends_with("w2999"));
    }

    /// A text with fewer bytes than the window fits without asking the server; past that, a
    /// server that cannot count leaves the 2-bytes-per-token floor to decide.
    #[test]
    fn fits_counts_only_what_could_overrun() {
        let e = HttpEmbedder::new(cfg("text-embedding-nomic-embed-text-v1.5"));
        assert!(e.fits(&"a".repeat(2000)));
        assert!(
            e.fits(&"a ".repeat(1500)),
            "3000 bytes is at most 1500 tokens"
        );
        assert!(!e.fits(&"a ".repeat(3000)), "6000 bytes may be 3000 tokens");
        let unlimited = HttpEmbedder::new(cfg("some-other-model"));
        assert!(unlimited.fits(&"a ".repeat(100_000)));
    }

    #[test]
    fn test_config_defaults() {
        let config = HttpEmbedderConfig::from_env();
        assert!(config.base_url.contains("127.0.0.1"));
        assert!(config.model.contains("nomic"));
        assert!(
            config.task_prefixes,
            "a nomic model gets its task prefixes by default"
        );
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
        assert_eq!(
            e.id(),
            "http://localhost:9#text-embedding-nomic-embed-text-v1.5"
        );
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
        assert_eq!(
            nomic.document_text("search_document: hello"),
            "search_document: hello"
        );

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
