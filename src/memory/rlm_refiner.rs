//! Recursive LLM Refiner — final-stage rerank via LLM relevance scoring.
//!
//! # Why this exists
//!
//! Veld's hybrid retrieval (BM25 + vector + graph + cross-encoder) handles
//! the bulk of ranking. The cross-encoder stage in
//! [`crate::memory::hybrid_search`] provides joint query-document attention
//! via the `ms-marco-MiniLM-L-6-v2` model.
//!
//! For some query classes — multi-hop questions, queries that require
//! reasoning across the candidate set — an LLM-driven refiner can outperform
//! the cross-encoder by leveraging in-context reasoning rather than purely
//! attention-based scoring.
//!
//! # Local-first posture
//!
//! Like [`crate::memory::llm::HttpLlmConsolidator`], this module is
//! **opt-in**. [`RlmRefiner`] requires explicit construction with an
//! endpoint and API key (or env-var configuration). Without explicit setup,
//! no remote call ever happens — veld's offline default is preserved.
//!
//! # Wiring
//!
//! Used by `HybridSearchEngine::search_with_dynamic_weights` when the
//! engine's [`RefinerMode`](super::hybrid_search::RefinerMode) is `Rlm` or
//! `Stacked`. The trait surface mirrors `CrossEncoderReranker::rerank` so
//! the two are interchangeable at the call site.
//!
//! # Protocol
//!
//! Targets the OpenAI-compatible chat-completions API surface (the same
//! contract used by `HttpLlmConsolidator`). Compatible hosts: OpenAI,
//! Anthropic via proxy, Ollama, vLLM, LM Studio.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::memory::types::MemoryId;

/// Anything that can rerank candidates for a query.
///
/// Implementors return scores in `[0.0, 1.0]` (higher = more relevant).
/// The output preserves the candidate set but may reorder it; callers do
/// not assume any particular ordering of the returned vector beyond the
/// usual "best-first" convention.
pub trait Refiner: Send + Sync {
    /// Score `candidates` against `query`, returning `(id, score)` pairs
    /// sorted best-first.
    ///
    /// `candidates` is `(memory_id, content, current_score)`. The
    /// `current_score` is informational — implementors typically ignore it
    /// and compute fresh scores from `content`.
    fn refine(
        &self,
        query: &str,
        candidates: Vec<(MemoryId, String, f32)>,
    ) -> Result<Vec<(MemoryId, f32)>>;

    /// Cheap descriptor for logging / metrics. Not load-bearing.
    fn name(&self) -> &str {
        "refiner"
    }
}

/// Passthrough refiner — returns candidates with their current scores
/// intact, sorted best-first.
///
/// Used in ablation runs to isolate refiner-harness overhead from actual
/// refinement value.
pub struct NullRefiner;

impl Refiner for NullRefiner {
    fn refine(
        &self,
        _query: &str,
        candidates: Vec<(MemoryId, String, f32)>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let mut out: Vec<(MemoryId, f32)> = candidates
            .into_iter()
            .map(|(id, _, score)| (id, score))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }

    fn name(&self) -> &str {
        "null"
    }
}

/// LLM-driven refiner that issues a single scoring call per query.
///
/// Targets the OpenAI-compatible chat-completions endpoint. The request
/// sends the query and candidate texts; the model returns a JSON array of
/// `{id, score}` objects which is parsed back into the result vector.
///
/// Configure via constructor or [`RlmRefiner::from_env`]. Env vars
/// (read at `from_env` time, not lazily):
/// - `VELD_RLM_ENDPOINT` — chat-completions URL
/// - `VELD_RLM_API_KEY`  — bearer token (optional for local hosts)
/// - `VELD_RLM_MODEL`    — model identifier
pub struct RlmRefiner {
    endpoint: String,
    api_key: String,
    model: String,
    client: reqwest::blocking::Client,
    /// Per-candidate content truncation budget, in characters.
    max_content_chars: usize,
    /// Upper bound on response tokens the model may emit.
    max_response_tokens: u32,
}

impl RlmRefiner {
    pub fn new(endpoint: String, api_key: String, model: String) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("veld-rlm-refiner/1.0")
            .timeout(Duration::from_secs(60))
            .build()
            .context("RlmRefiner: failed to build HTTP client")?;
        Ok(Self {
            endpoint,
            api_key,
            model,
            client,
            max_content_chars: 1500,
            max_response_tokens: 2048,
        })
    }

    /// Construct from `VELD_RLM_*` env vars. Returns `Ok(None)` (not `Err`)
    /// when the endpoint isn't configured — callers should treat that as
    /// the offline default and either fall back to the cross-encoder or
    /// skip refinement entirely.
    pub fn from_env() -> Result<Option<Self>> {
        let endpoint = match std::env::var("VELD_RLM_ENDPOINT") {
            Ok(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };
        let api_key = std::env::var("VELD_RLM_API_KEY").unwrap_or_default();
        let model = std::env::var("VELD_RLM_MODEL")
            .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
        Self::new(endpoint, api_key, model).map(Some)
    }

    /// Override per-candidate content truncation budget. Defaults to 1500
    /// characters which keeps a 20-candidate prompt under ~30k input
    /// characters before model tokenization.
    pub fn with_max_content_chars(mut self, chars: usize) -> Self {
        self.max_content_chars = chars;
        self
    }

    fn build_prompt(&self, query: &str, candidates: &[(MemoryId, String, f32)]) -> String {
        let mut block = String::with_capacity(candidates.len() * 256);
        for (idx, (_, content, _)) in candidates.iter().enumerate() {
            let trimmed = truncate_chars(content, self.max_content_chars);
            block.push_str(&format!("[{}] {}\n", idx + 1, trimmed));
        }
        format!(
            "Query: {query}\n\nCandidates:\n{block}\nReturn a JSON array `[{{\"id\":N,\"score\":0.0..1.0}}, ...]` with one entry per candidate (id is the 1-based index above). Score reflects how directly the candidate answers the query (1.0 = direct answer, 0.0 = irrelevant). Output ONLY the JSON array — no prose, no markdown fences."
        )
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
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
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ScoredCandidate {
    id: usize,
    score: f32,
}

impl Refiner for RlmRefiner {
    fn refine(
        &self,
        query: &str,
        candidates: Vec<(MemoryId, String, f32)>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let prompt = self.build_prompt(query, &candidates);
        let request = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "You are a precise relevance scorer. Return only the JSON array requested — no commentary, no markdown fences.",
                },
                ChatMessage {
                    role: "user",
                    content: &prompt,
                },
            ],
            temperature: 0.0,
            max_tokens: self.max_response_tokens,
        };

        let mut req = self.client.post(&self.endpoint).json(&request);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req.send().context("RlmRefiner: HTTP request failed")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(anyhow!(
                "RlmRefiner: HTTP {status}: {}",
                body.chars().take(500).collect::<String>()
            ));
        }
        let parsed: ChatResponse = response
            .json()
            .context("RlmRefiner: response body was not valid JSON")?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("RlmRefiner: empty choices array in response"))?
            .message
            .content;

        let scored: Vec<ScoredCandidate> = parse_score_array(&content)
            .with_context(|| format!("RlmRefiner: failed to parse model output: {content}"))?;

        let mut score_by_idx: HashMap<usize, f32> = HashMap::with_capacity(scored.len());
        for entry in scored {
            score_by_idx.insert(entry.id, entry.score.clamp(0.0, 1.0));
        }

        let mut results: Vec<(MemoryId, f32)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, (id, _, current))| {
                let score = score_by_idx
                    .get(&(idx + 1))
                    .copied()
                    .unwrap_or_else(|| current.clamp(0.0, 1.0));
                (id, score)
            })
            .collect();

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(results)
    }

    fn name(&self) -> &str {
        "rlm"
    }
}

/// Truncate a string to at most `max_chars` Unicode characters.
///
/// Uses `char_indices` so multi-byte boundaries are respected. A trailing
/// ellipsis is appended when truncation occurs so the model can tell the
/// candidate was cut.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut end = s.len();
    for (count, (offset, _)) in s.char_indices().enumerate() {
        if count == max_chars {
            end = offset;
            break;
        }
    }
    if end < s.len() {
        let mut out = String::with_capacity(end + 1);
        out.push_str(&s[..end]);
        out.push('…');
        out
    } else {
        s.to_string()
    }
}

/// Parse a JSON score array from raw LLM output, tolerating common framing
/// (markdown fences, leading prose).
fn parse_score_array(raw: &str) -> Result<Vec<ScoredCandidate>> {
    let trimmed = raw.trim();

    let inner = if let Some(rest) = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
    {
        rest.trim_end_matches("```").trim()
    } else {
        trimmed
    };

    let start = inner
        .find('[')
        .ok_or_else(|| anyhow!("no '[' bracket in output"))?;
    let end = inner
        .rfind(']')
        .ok_or_else(|| anyhow!("no ']' bracket in output"))?;
    if end <= start {
        return Err(anyhow!("malformed bracket pair in output"));
    }
    let json_slice = &inner[start..=end];

    serde_json::from_str(json_slice).context("score array failed to deserialize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn mk_id() -> MemoryId {
        MemoryId(Uuid::new_v4())
    }

    #[test]
    fn null_refiner_returns_sorted_passthrough() {
        let null = NullRefiner;
        let id_a = mk_id();
        let id_b = mk_id();
        let id_c = mk_id();
        let candidates = vec![
            (id_a, "alpha".into(), 0.3),
            (id_b, "bravo".into(), 0.9),
            (id_c, "charlie".into(), 0.1),
        ];
        let out = null.refine("q", candidates).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].1, 0.9);
        assert_eq!(out[1].1, 0.3);
        assert_eq!(out[2].1, 0.1);
    }

    #[test]
    fn null_refiner_handles_empty_input() {
        let null = NullRefiner;
        let out = null.refine("q", Vec::new()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn parse_score_array_handles_plain_json() {
        let raw = r#"[{"id":1,"score":0.9},{"id":2,"score":0.1}]"#;
        let parsed = parse_score_array(raw).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, 1);
        assert_eq!(parsed[0].score, 0.9);
        assert_eq!(parsed[1].id, 2);
        assert_eq!(parsed[1].score, 0.1);
    }

    #[test]
    fn parse_score_array_strips_markdown_fences() {
        let raw = "```json\n[{\"id\":1,\"score\":0.5}]\n```";
        let parsed = parse_score_array(raw).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].score, 0.5);
    }

    #[test]
    fn parse_score_array_strips_bare_fences() {
        let raw = "```\n[{\"id\":3,\"score\":0.42}]\n```";
        let parsed = parse_score_array(raw).unwrap();
        assert_eq!(parsed[0].id, 3);
        assert_eq!(parsed[0].score, 0.42);
    }

    #[test]
    fn parse_score_array_tolerates_leading_prose() {
        let raw = "Here is the answer: [{\"id\":1,\"score\":0.5}]";
        let parsed = parse_score_array(raw).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn parse_score_array_rejects_no_bracket() {
        let err = parse_score_array("nope").unwrap_err();
        assert!(err.to_string().contains("'['"));
    }

    #[test]
    fn truncate_chars_passes_through_short() {
        assert_eq!(truncate_chars("hello", 10), "hello");
    }

    #[test]
    fn truncate_chars_cuts_at_boundary() {
        let out = truncate_chars("hello world", 5);
        assert_eq!(out, "hello…");
    }

    #[test]
    fn truncate_chars_respects_multibyte() {
        let s = "héllo wörld";
        let out = truncate_chars(s, 5);
        let kept: String = s.chars().take(5).collect();
        assert_eq!(out, format!("{kept}…"));
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        let prior = std::env::var("VELD_RLM_ENDPOINT").ok();
        std::env::remove_var("VELD_RLM_ENDPOINT");
        let got = RlmRefiner::from_env().unwrap();
        assert!(got.is_none());
        if let Some(v) = prior {
            std::env::set_var("VELD_RLM_ENDPOINT", v);
        }
    }
}
