//! Optional LLM hook for higher-quality fact consolidation.
//!
//! # Why this exists
//!
//! Veld's default consolidation path (`SemanticConsolidator`) is offline and
//! pattern-based. It does well on high-precision patterns but misses the
//! long tail. This module lets callers plug in a remote LLM that performs
//! richer fact extraction over a batch of episodic memories.
//!
//! # Local-first posture
//!
//! Using this module is **opt-in**: callers construct an
//! [`HttpLlmConsolidator`] explicitly with an endpoint + key (or read them
//! from env). Without explicit configuration, no remote call ever happens.
//! This preserves Veld's offline default.
//!
//! # Protocol
//!
//! The HTTP impl targets the OpenAI-compatible chat-completions API surface,
//! which is supported by OpenAI, Anthropic (via proxy), and every local host in
//! `manifests/llm-hosts.json`,
//! vLLM, and most other hosts. Callers swap the endpoint URL to point at
//! whichever provider they trust.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Sentinel returned by [`LlmConsolidator::llm_version`] when the
/// underlying impl genuinely doesn't know what model handled the call
/// (e.g. an OpenAI-compatible endpoint that doesn't echo back the model
/// header). Captured verbatim in provenance trails so audits can tell
/// "we forgot to record this" apart from "the model itself is named UNKNOWN".
pub const LLM_VERSION_UNKNOWN: &str = "UNKNOWN";

/// Sentinel returned by [`LlmConsolidator::llm_version`] when no model
/// identifier was configured at construction time (env var unset,
/// caller passed empty string).
pub const LLM_VERSION_NOT_PROVIDED: &str = "NOT PROVIDED";

/// A fact extracted by the LLM from a batch of episodic memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractedFact {
    /// One-sentence factual claim
    pub fact: String,
    /// Entities the fact involves
    pub entities: Vec<String>,
    /// LLM-stated confidence in [0, 1]
    pub confidence: f32,
    /// Which LLM produced this fact. Populated by `augment_with_llm` from
    /// [`LlmConsolidator::llm_version`]. Uses [`LLM_VERSION_UNKNOWN`] or
    /// [`LLM_VERSION_NOT_PROVIDED`] when the impl can't identify itself
    /// rather than dropping the field — provenance audits need the
    /// sentinel to distinguish "didn't capture" from "doesn't exist".
    #[serde(default)]
    pub llm_version: String,
}

/// Anything that can extract semantic facts from a batch of text snippets.
pub trait LlmConsolidator: Send + Sync {
    /// Extract durable facts from a batch of episodic content snippets.
    fn extract_facts(&self, snippets: &[String]) -> Result<Vec<LlmExtractedFact>>;

    /// Schema-constrained extraction (Cognee-style typed output).
    ///
    /// `schema_json` is a JSON Schema document the LLM is constrained
    /// (via prompt instructions) to produce conforming output for. The
    /// returned values are raw `serde_json::Value`s — callers are expected
    /// to deserialize into their own caller-defined type via
    /// `serde_json::from_value`.
    ///
    /// The default implementation returns `NotImplemented` so existing
    /// impls don't have to ship a schema-aware path.
    fn extract_with_schema(
        &self,
        _snippets: &[String],
        _schema_json: &str,
    ) -> Result<Vec<serde_json::Value>> {
        Err(anyhow!(
            "{} does not implement schema-constrained extraction",
            self.name()
        ))
    }

    /// Cheap descriptor for logging / metrics. Not load-bearing.
    fn name(&self) -> &str {
        "llm"
    }

    /// Identifier of the underlying model / version, used for provenance
    /// trails on facts the LLM produces. Default returns
    /// [`LLM_VERSION_UNKNOWN`]; impls should override when they can.
    /// Impls that have a configured model name but the value is empty
    /// should return [`LLM_VERSION_NOT_PROVIDED`] instead.
    fn llm_version(&self) -> String {
        LLM_VERSION_UNKNOWN.to_string()
    }
}

/// HTTP impl targeting the OpenAI-compatible chat-completions endpoint.
///
/// Configure via either explicit constructor or [`HttpLlmConsolidator::from_env`].
/// Env vars (read at `from_env` call time, not lazily):
///   - `VELD_LLM_ENDPOINT` — e.g. `https://api.openai.com/v1/chat/completions`
///   - `VELD_LLM_HOST` — instead of an endpoint, a host declared in
///     `manifests/llm-hosts.json` (`lmstudio`, `llama.cpp`, ...), reached at its
///     `VELD_LLM_URL_<ID>` override when one is set
///   - `VELD_LLM_API_KEY` — bearer token
///   - `VELD_LLM_MODEL` — e.g. `gpt-4o-mini`, `claude-sonnet-4-6`, `llama3.1`
///   - `VELD_LLM_TIMEOUT_SECS` — how long one call may take, 60 when unset
pub struct HttpLlmConsolidator {
    endpoint: String,
    api_key: String,
    model: String,
    timeout: Duration,
    /// Built on the first call. reqwest's blocking client runs its own runtime and
    /// panics when built inside a tokio context, and the server constructs this
    /// type on a runtime thread while every call is made from a plain thread.
    client: std::sync::OnceLock<reqwest::blocking::Client>,
}

impl HttpLlmConsolidator {
    /// Model used when `VELD_LLM_MODEL` is unset.
    pub const DEFAULT_MODEL: &'static str = "gpt-4o-mini";

    /// How long one call may take when `VELD_LLM_TIMEOUT_SECS` is unset.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    /// Build a consolidator with explicit configuration. No connection is made
    /// until the first call.
    pub fn new(endpoint: String, api_key: String, model: String) -> Result<Self> {
        Ok(Self {
            endpoint,
            api_key,
            model,
            timeout: Self::DEFAULT_TIMEOUT,
            client: std::sync::OnceLock::new(),
        })
    }

    /// The same consolidator with calls allowed `timeout`. A local model that reasons
    /// before it answers can need minutes for one window.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build from `VELD_LLM_*` env vars. Returns `Ok(None)` (not Err) when
    /// the endpoint isn't configured — callers should treat that as a
    /// signal to skip remote consolidation entirely, not as an error.
    ///
    /// The endpoint is `VELD_LLM_ENDPOINT`, a full chat-completions URL, or else
    /// the chat-completions path of the host `VELD_LLM_HOST` names. A named host
    /// needs `VELD_LLM_MODEL`: a local server answers for the models loaded into
    /// it, so no default name can be assumed to be one of them. An unknown host
    /// is an error, not a fallback, so a typo cannot look like a configuration.
    pub fn from_env() -> Result<Option<Self>> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// [`Self::from_env`] with the environment injected, so it is tested without
    /// mutating the process environment that parallel tests share.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        let set = |key: &str| lookup(key).filter(|v| !v.trim().is_empty());
        let model = set("VELD_LLM_MODEL");
        let (endpoint, model) = if let Some(endpoint) = set("VELD_LLM_ENDPOINT") {
            (
                endpoint,
                model.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string()),
            )
        } else if let Some(id) = set("VELD_LLM_HOST") {
            let id = id.trim();
            let declared = crate::llm_hosts::declaration();
            let host = crate::llm_hosts::host(id).ok_or_else(|| {
                let ids: Vec<&str> = declared.hosts.iter().map(|h| h.id.as_str()).collect();
                anyhow!(
                    "VELD_LLM_HOST={id} names no host in manifests/llm-hosts.json; declared: {}",
                    ids.join(", ")
                )
            })?;
            if !host.serves("chat") {
                return Err(anyhow!(
                    "VELD_LLM_HOST={id} does not serve chat completions"
                ));
            }
            let model = model.ok_or_else(|| {
                anyhow!("VELD_LLM_HOST={id} needs VELD_LLM_MODEL, the model id the server lists")
            })?;
            let base = host.base_url_with(&lookup);
            (
                format!("{}/v1/chat/completions", base.trim_end_matches('/')),
                model,
            )
        } else {
            return Ok(None);
        };
        let api_key = lookup("VELD_LLM_API_KEY").unwrap_or_default();
        let timeout = match set("VELD_LLM_TIMEOUT_SECS") {
            Some(secs) => match secs.trim().parse::<u64>() {
                Ok(secs) if secs > 0 => Duration::from_secs(secs),
                _ => {
                    return Err(anyhow!(
                        "VELD_LLM_TIMEOUT_SECS={secs} is not a positive number of seconds"
                    ))
                }
            },
            None => Self::DEFAULT_TIMEOUT,
        };
        Ok(Some(
            Self::new(endpoint, api_key, model)?.with_timeout(timeout),
        ))
    }

    /// The configured model name, as sent in each request.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn client(&self) -> Result<&reqwest::blocking::Client> {
        if let Some(client) = self.client.get() {
            return Ok(client);
        }
        let built = reqwest::blocking::Client::builder()
            .user_agent("veld-llm/1.0")
            .timeout(self.timeout)
            .build()?;
        // Two first calls may race to build; the loser's client is dropped.
        Ok(self.client.get_or_init(|| built))
    }
}

impl HttpLlmConsolidator {
    /// Shared chat call used by both `extract_facts` and
    /// `extract_with_schema`. Returns the raw model `content` string.
    fn chat(&self, prompt: &str) -> Result<String> {
        let request = ChatRequest {
            model: &self.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            temperature: 0.0,
        };
        let body = serde_json::to_string(&request)?;
        // A model tape answers by the exact body, so a replay needs no endpoint at all.
        if let Some(taped) = crate::embeddings::model_tape::llm(&body, || self.post(&body)) {
            return taped;
        }
        self.post(&body)
    }

    /// Send a serialised chat request and return the first choice's content.
    fn post(&self, body: &str) -> Result<String> {
        let mut req = self
            .client()?
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string());
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send()?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "LLM endpoint {} → {}",
                self.endpoint,
                resp.status().as_u16()
            ));
        }
        let body: ChatResponse = resp.json()?;
        Ok(body
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("LLM response had no choices"))?
            .message
            .content)
    }
}

/// Strip ```json … ``` fences a model may have wrapped its JSON in.
fn strip_fences(s: &str) -> &str {
    s.trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
}

impl LlmConsolidator for HttpLlmConsolidator {
    fn name(&self) -> &str {
        "http-llm"
    }

    fn llm_version(&self) -> String {
        if self.model.trim().is_empty() {
            LLM_VERSION_NOT_PROVIDED.to_string()
        } else {
            self.model.clone()
        }
    }

    fn extract_with_schema(
        &self,
        snippets: &[String],
        schema_json: &str,
    ) -> Result<Vec<serde_json::Value>> {
        if snippets.is_empty() {
            return Ok(Vec::new());
        }
        let joined = snippets
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Extract structured records from the following text. \
             Return a JSON ARRAY in which every element conforms exactly to \
             the JSON Schema below. Do NOT include any prose, prefix, or \
             ```json fences. Only the raw JSON array.\n\n\
             SCHEMA:\n{schema_json}\n\nTEXT:\n{joined}"
        );

        let content = self.chat(&prompt)?;
        let trimmed = strip_fences(&content);
        let values: Vec<serde_json::Value> = serde_json::from_str(trimmed).map_err(|e| {
            anyhow!("LLM returned non-JSON-array payload under schema: {e}\nRaw: {content}")
        })?;
        Ok(values)
    }

    fn extract_facts(&self, snippets: &[String]) -> Result<Vec<LlmExtractedFact>> {
        if snippets.is_empty() {
            return Ok(Vec::new());
        }
        let joined = snippets
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Extract durable factual claims from the following episodic memories. \
             Return JSON only — an array of objects with fields: \
             fact (one-sentence claim), entities (array of subjects/objects \
             the fact concerns), confidence (0-1). Do not invent facts not \
             grounded in the inputs. Skip preferences, opinions, and \
             time-bound observations. Output JSON only, no prose.\n\n{joined}"
        );

        let content = self.chat(&prompt)?;
        let trimmed = strip_fences(&content);
        let mut facts: Vec<LlmExtractedFact> = serde_json::from_str(trimmed)
            .map_err(|e| anyhow!("LLM returned non-JSON facts payload: {e}\nRaw: {content}"))?;
        // Stamp provenance — the model doesn't echo it back, we have to
        // inject from the local config. Empty / unknown values get the
        // sentinel constants so downstream audits can tell them apart.
        let version = self.llm_version();
        for f in &mut facts {
            if f.llm_version.is_empty() {
                f.llm_version = version.clone();
            }
        }
        Ok(facts)
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_returns_none_when_unset() {
        // SAFETY: tests run single-threaded by default for this assertion
        let prev = std::env::var("VELD_LLM_ENDPOINT").ok();
        std::env::remove_var("VELD_LLM_ENDPOINT");
        let r = HttpLlmConsolidator::from_env().unwrap();
        assert!(r.is_none());
        if let Some(v) = prev {
            std::env::set_var("VELD_LLM_ENDPOINT", v);
        }
    }

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    }

    #[test]
    fn nothing_configured_is_no_consolidator() {
        assert!(HttpLlmConsolidator::from_lookup(lookup(&[]))
            .unwrap()
            .is_none());
        assert!(
            HttpLlmConsolidator::from_lookup(lookup(&[("VELD_LLM_MODEL", "qwen")]))
                .unwrap()
                .is_none(),
            "a model with nowhere to send it configures nothing"
        );
    }

    #[test]
    fn a_declared_host_is_reached_at_its_chat_completions_path() {
        let llm = HttpLlmConsolidator::from_lookup(lookup(&[
            ("VELD_LLM_HOST", "lmstudio"),
            ("VELD_LLM_MODEL", "qwen3-8b"),
        ]))
        .unwrap()
        .expect("configured");
        let declared = crate::llm_hosts::host("lmstudio").unwrap().base_url.clone();
        assert_eq!(llm.endpoint, format!("{declared}/v1/chat/completions"));
        assert_eq!(llm.model(), "qwen3-8b");

        // The host's own override applies, as it does for every other consumer.
        let tunnelled = HttpLlmConsolidator::from_lookup(lookup(&[
            ("VELD_LLM_HOST", "lmstudio"),
            ("VELD_LLM_URL_LMSTUDIO", "http://127.0.0.1:11234/"),
            ("VELD_LLM_MODEL", "qwen3-8b"),
            ("VELD_LLM_API_KEY", "k"),
        ]))
        .unwrap()
        .expect("configured");
        assert_eq!(
            tunnelled.endpoint,
            "http://127.0.0.1:11234/v1/chat/completions"
        );
        assert_eq!(tunnelled.api_key, "k");
        assert_eq!(tunnelled.timeout, HttpLlmConsolidator::DEFAULT_TIMEOUT);
    }

    #[test]
    fn the_call_timeout_is_configurable_and_must_be_positive() {
        let slow = HttpLlmConsolidator::from_lookup(lookup(&[
            (
                "VELD_LLM_ENDPOINT",
                "http://127.0.0.1:1234/v1/chat/completions",
            ),
            ("VELD_LLM_TIMEOUT_SECS", "300"),
        ]))
        .unwrap()
        .expect("configured");
        assert_eq!(slow.timeout, Duration::from_secs(300));
        for bad in ["0", "soon"] {
            assert!(HttpLlmConsolidator::from_lookup(lookup(&[
                (
                    "VELD_LLM_ENDPOINT",
                    "http://127.0.0.1:1234/v1/chat/completions"
                ),
                ("VELD_LLM_TIMEOUT_SECS", bad),
            ]))
            .is_err());
        }
    }

    #[test]
    fn an_explicit_endpoint_wins_over_a_host() {
        let llm = HttpLlmConsolidator::from_lookup(lookup(&[
            (
                "VELD_LLM_ENDPOINT",
                "http://10.0.0.2:9000/v1/chat/completions",
            ),
            ("VELD_LLM_HOST", "lmstudio"),
        ]))
        .unwrap()
        .expect("configured");
        assert_eq!(llm.endpoint, "http://10.0.0.2:9000/v1/chat/completions");
        assert_eq!(llm.model(), HttpLlmConsolidator::DEFAULT_MODEL);
    }

    #[test]
    fn an_unknown_host_or_a_host_without_a_model_is_refused() {
        let unknown = HttpLlmConsolidator::from_lookup(lookup(&[
            ("VELD_LLM_HOST", "ollama"),
            ("VELD_LLM_MODEL", "llama3.1"),
        ]));
        let err = unknown.err().expect("refused").to_string();
        assert!(err.contains("ollama") && err.contains("lmstudio"), "{err}");

        let modelless = HttpLlmConsolidator::from_lookup(lookup(&[("VELD_LLM_HOST", "llama.cpp")]));
        assert!(modelless
            .err()
            .expect("refused")
            .to_string()
            .contains("VELD_LLM_MODEL"));
    }

    #[test]
    fn extract_facts_empty_input_is_noop() {
        // No network: empty input short-circuits before the HTTP call.
        let llm = HttpLlmConsolidator::new(
            "http://localhost:0/never-called".to_string(),
            String::new(),
            "test".to_string(),
        )
        .unwrap();
        let r = llm.extract_facts(&[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn name_is_descriptive() {
        let llm =
            HttpLlmConsolidator::new("http://x".to_string(), String::new(), "test".to_string())
                .unwrap();
        assert_eq!(llm.name(), "http-llm");
    }

    #[test]
    fn llm_version_returns_configured_model() {
        let llm = HttpLlmConsolidator::new(
            "http://x".to_string(),
            String::new(),
            "qwen2.5:14b".to_string(),
        )
        .unwrap();
        assert_eq!(llm.llm_version(), "qwen2.5:14b");
    }

    #[test]
    fn llm_version_empty_model_returns_not_provided() {
        let llm = HttpLlmConsolidator::new("http://x".to_string(), String::new(), "".to_string())
            .unwrap();
        assert_eq!(llm.llm_version(), LLM_VERSION_NOT_PROVIDED);
    }

    #[test]
    fn llm_version_whitespace_model_returns_not_provided() {
        let llm =
            HttpLlmConsolidator::new("http://x".to_string(), String::new(), "   ".to_string())
                .unwrap();
        assert_eq!(llm.llm_version(), LLM_VERSION_NOT_PROVIDED);
    }

    #[test]
    fn default_trait_llm_version_is_unknown() {
        struct FakeLlm;
        impl LlmConsolidator for FakeLlm {
            fn extract_facts(&self, _: &[String]) -> Result<Vec<LlmExtractedFact>> {
                Ok(Vec::new())
            }
        }
        assert_eq!(FakeLlm.llm_version(), LLM_VERSION_UNKNOWN);
    }
}
