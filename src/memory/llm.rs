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
//! which is supported by OpenAI, Anthropic (via proxy), local Ollama,
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
///   - `VELD_LLM_API_KEY` — bearer token
///   - `VELD_LLM_MODEL` — e.g. `gpt-4o-mini`, `claude-sonnet-4-6`, `llama3.1`
pub struct HttpLlmConsolidator {
    endpoint: String,
    api_key: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl HttpLlmConsolidator {
    /// Build a consolidator with explicit configuration.
    pub fn new(endpoint: String, api_key: String, model: String) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("veld-llm/1.0")
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            endpoint,
            api_key,
            model,
            client,
        })
    }

    /// Build from `VELD_LLM_*` env vars. Returns `Ok(None)` (not Err) when
    /// the endpoint isn't configured — callers should treat that as a
    /// signal to skip remote consolidation entirely, not as an error.
    pub fn from_env() -> Result<Option<Self>> {
        let endpoint = match std::env::var("VELD_LLM_ENDPOINT") {
            Ok(v) if !v.trim().is_empty() => v,
            _ => return Ok(None),
        };
        let api_key = std::env::var("VELD_LLM_API_KEY").unwrap_or_default();
        let model = std::env::var("VELD_LLM_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());
        Ok(Some(Self::new(endpoint, api_key, model)?))
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

        let mut req = self.client.post(&self.endpoint).json(&request);
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
        let values: Vec<serde_json::Value> = serde_json::from_str(trimmed)
            .map_err(|e| anyhow!(
                "LLM returned non-JSON-array payload under schema: {e}\nRaw: {content}"
            ))?;
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
        let llm = HttpLlmConsolidator::new(
            "http://x".to_string(),
            String::new(),
            "test".to_string(),
        )
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
        let llm = HttpLlmConsolidator::new(
            "http://x".to_string(),
            String::new(),
            "".to_string(),
        )
        .unwrap();
        assert_eq!(llm.llm_version(), LLM_VERSION_NOT_PROVIDED);
    }

    #[test]
    fn llm_version_whitespace_model_returns_not_provided() {
        let llm = HttpLlmConsolidator::new(
            "http://x".to_string(),
            String::new(),
            "   ".to_string(),
        )
        .unwrap();
        assert_eq!(llm.llm_version(), LLM_VERSION_NOT_PROVIDED);
    }

    #[test]
    fn default_trait_llm_version_is_unknown() {
        struct FakeLlm;
        impl LlmConsolidator for FakeLlm {
            fn extract_facts(
                &self,
                _: &[String],
            ) -> Result<Vec<LlmExtractedFact>> {
                Ok(Vec::new())
            }
        }
        assert_eq!(FakeLlm.llm_version(), LLM_VERSION_UNKNOWN);
    }
}
