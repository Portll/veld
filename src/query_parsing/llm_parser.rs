//! LLM-Based Query Parser
//!
//! Uses a local LLM server via the OpenAI HTTP API. Which servers, and their default ports, are
//! declared once in `manifests/llm-hosts.json` — see `crate::llm_hosts`.
//! Provides better temporal reasoning and entity extraction than rule-based.

use super::parser_trait::*;
use chrono::{DateTime, NaiveDate, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// LLM-based query parser speaking the OpenAI HTTP API to any declared host.
pub struct LlmParser {
    client: reqwest::blocking::Client,
    endpoint: String,
    model: String,
    generation_lock: Mutex<()>,
}

/// Request format for OpenAI-compatible APIs (LM Studio, vLLM, etc.)
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
    max_tokens: i32,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

/// Response format from OpenAI-compatible APIs
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessageResponse,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageResponse {
    content: String,
}

/// Expected JSON output format from the LLM
#[derive(Debug, Deserialize, Serialize)]
struct LlmOutput {
    entities: Vec<LlmEntity>,
    events: Vec<String>,
    modifiers: Vec<String>,
    temporal: LlmTemporal,
    is_attribute_query: bool,
    attribute_entity: Option<String>,
    attribute_name: Option<String>,
    confidence: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct LlmEntity {
    text: String,
    #[serde(rename = "type")]
    entity_type: String,
    negated: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct LlmTemporal {
    has_temporal_intent: bool,
    intent: String,
    relative_refs: Vec<LlmRelativeRef>,
    resolved_dates: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LlmRelativeRef {
    text: String,
    resolved_date: Option<String>,
    direction: String,
}

// `ApiType` is gone. It had two variants and ollama was `#[default]`, so `LlmParser::new()`
// constructed an ollama client while `from_env()` defaulted to openai — two different defaults for
// the same thing, in the same file. Every host in `manifests/llm-hosts.json` speaks the OpenAI HTTP
// API, which is what makes a host a row in a declaration rather than a branch here. Removed
// 2026-08-27 on operator instruction; the removal is recorded in that declaration with its reason.

impl LlmParser {
    /// Create a new LLM parser against any OpenAI-compatible host.
    ///
    /// # Arguments
    /// * `endpoint` - Base URL, e.g. `http://127.0.0.1:8080` (llama.cpp) or `:1234` (LM Studio)
    /// * `model` - Model identifier as the server names it
    pub fn new(endpoint: &str, model: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            generation_lock: Mutex::new(()),
        }
    }

    /// Create an LLM parser from environment.
    ///
    /// Reads:
    /// - `VELD_LLM_URL`   — base URL. Defaults to the declared default host's URL
    ///   (`manifests/llm-hosts.json`), NOT to a literal written here. A port hardcoded in source
    ///   is how eight copies of `127.0.0.1:1234` ended up across three repositories.
    /// - `VELD_LLM_HOST`  — a declared host id (`llama.cpp`, `lmstudio`, …) instead of a URL.
    ///   Its own `VELD_LLM_URL_<ID>` override still applies.
    /// - `VELD_LLM_MODEL` — model identifier as the server names it.
    ///
    /// `VELD_LLM_API_TYPE` is gone with ollama: every declared host speaks the OpenAI API.
    pub fn from_env() -> Self {
        let decl = crate::llm_hosts::declaration();
        let endpoint = std::env::var("VELD_LLM_URL").unwrap_or_else(|_| {
            let id = std::env::var("VELD_LLM_HOST").unwrap_or_else(|_| decl.default.clone());
            match crate::llm_hosts::host(&id) {
                Some(h) => h.base_url(),
                None => {
                    // Fail loudly-ish: an unknown id is a config error, and falling through to the
                    // default silently would make a typo look like a working configuration.
                    tracing::warn!(
                        requested = %id,
                        "VELD_LLM_HOST names no declared host; using the declared default"
                    );
                    crate::llm_hosts::host(&decl.default)
                        .expect("the declaration's own default must exist")
                        .base_url()
                }
            }
        });
        let model = std::env::var("VELD_LLM_MODEL")
            .unwrap_or_else(|_| "qwq-32b-preview".to_string());
        Self::new(&endpoint, &model)
    }

    /// Build the prompt for query parsing
    fn build_prompt(&self, query: &str, context_date: Option<DateTime<Utc>>) -> String {
        let date_context = context_date
            .map(|d| format!("Today's date: {}", d.format("%B %d, %Y")))
            .unwrap_or_else(|| "Today's date: unknown".to_string());

        format!(
            r#"You are a query parser. Extract structured information from the query.
Output ONLY valid JSON, no explanation or markdown.

{date_context}

Parse this query: "{query}"

Output this exact JSON structure:
{{"entities":[{{"text":"name","type":"person|place|thing|event|time","negated":false}}],"events":["verb"],"modifiers":["adjective"],"temporal":{{"has_temporal_intent":true,"intent":"when_question|specific_time|ordering|duration|none","relative_refs":[{{"text":"last year","resolved_date":"2024-01-01","direction":"past"}}],"resolved_dates":["2024-01-01"]}},"is_attribute_query":false,"attribute_entity":null,"attribute_name":null,"confidence":0.9}}"#
        )
    }

    /// Generate using OpenAI-compatible API
    fn generate_openai(&self, prompt: &str) -> Result<String, String> {
        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.1,
            max_tokens: 512,
        };

        let url = format!("{}/v1/chat/completions", self.endpoint);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("API returned status: {}", response.status()));
        }

        let openai_response: OpenAIResponse = response
            .json()
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        openai_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from API".to_string())
    }

    /// Generate against the configured host.
    ///
    /// There is no longer a second API to fall back to, and the old fallback was worse than
    /// nothing: it tried the OTHER protocol against the SAME endpoint, so a real failure from the
    /// configured server was followed by a request that could only fail differently, and the error
    /// the caller saw came from the protocol it had not asked for.
    fn generate(&self, prompt: &str) -> Result<String, String> {
        self.generate_openai(prompt)
    }

    /// Parse the LLM output JSON into ParsedQuery
    fn parse_output(&self, output: &str, original_query: &str) -> ParsedQuery {
        let json_str = extract_json(output);

        match serde_json::from_str::<LlmOutput>(&json_str) {
            Ok(llm_out) => self.convert_llm_output(llm_out, original_query),
            Err(e) => {
                tracing::warn!("Failed to parse LLM output: {}, raw: {}", e, output);
                ParsedQuery::empty(original_query)
            }
        }
    }

    /// Convert LLM output to ParsedQuery
    fn convert_llm_output(&self, llm_out: LlmOutput, original_query: &str) -> ParsedQuery {
        let entities: Vec<Entity> = llm_out
            .entities
            .into_iter()
            .map(|e| Entity {
                text: e.text.clone(),
                stem: stem_word(&e.text),
                entity_type: parse_entity_type(&e.entity_type),
                ic_weight: 1.0,
                negated: e.negated,
            })
            .collect();

        let events: Vec<Event> = llm_out
            .events
            .into_iter()
            .map(|e| Event {
                text: e.clone(),
                stem: stem_word(&e),
                ic_weight: 0.7,
            })
            .collect();

        let relative_refs: Vec<RelativeTimeRef> = llm_out
            .temporal
            .relative_refs
            .into_iter()
            .map(|r| RelativeTimeRef {
                text: r.text,
                resolved: r
                    .resolved_date
                    .and_then(|d| NaiveDate::parse_from_str(&d, "%Y-%m-%d").ok()),
                direction: parse_direction(&r.direction),
                unit: TimeUnit::Unknown,
                offset: 1,
            })
            .collect();

        let resolved_dates: Vec<NaiveDate> = llm_out
            .temporal
            .resolved_dates
            .iter()
            .filter_map(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
            .collect();

        let attribute = if llm_out.is_attribute_query {
            llm_out.attribute_entity.map(|entity| AttributeQuery {
                entity,
                attribute: llm_out.attribute_name.unwrap_or_default(),
                synonyms: Vec::new(),
            })
        } else {
            None
        };

        let mut ic_weights = HashMap::new();
        for e in &entities {
            ic_weights.insert(e.text.to_lowercase(), e.ic_weight);
        }
        for e in &events {
            ic_weights.insert(e.text.to_lowercase(), e.ic_weight);
        }

        ParsedQuery {
            original: original_query.to_string(),
            entities,
            events,
            modifiers: llm_out.modifiers,
            temporal: TemporalInfo {
                has_temporal_intent: llm_out.temporal.has_temporal_intent,
                intent: parse_temporal_intent(&llm_out.temporal.intent),
                relative_refs,
                resolved_dates,
                absolute_dates: Vec::new(),
            },
            is_attribute_query: llm_out.is_attribute_query,
            attribute,
            compounds: Vec::new(),
            ic_weights,
            confidence: llm_out.confidence,
        }
    }

    /// Is a server reachable at the configured endpoint?
    ///
    /// One probe, not two. The ollama `/api/tags` attempt is gone, and with it the case where a
    /// server that answers `/api/tags` but not `/v1/models` was reported AVAILABLE and then failed
    /// every actual generation — the probe and the thing it predicted disagreed.
    ///
    /// A success here says an OpenAI-compatible server is listening. It does NOT say which one:
    /// llama.cpp, llamafile, LocalAI and mlx-lm all default to :8080 and answer identically. See
    /// `llm_hosts::identify_by_port`, which returns `None` for a shared port rather than guessing.
    pub fn is_server_available(&self) -> bool {
        self.client
            .get(format!("{}/v1/models", self.endpoint))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

impl QueryParser for LlmParser {
    fn parse(&self, query: &str, context_date: Option<DateTime<Utc>>) -> ParsedQuery {
        let _lock = self.generation_lock.lock();

        let prompt = self.build_prompt(query, context_date);

        match self.generate(&prompt) {
            Ok(output) => self.parse_output(&output, query),
            Err(e) => {
                tracing::error!("LLM generation failed: {}", e);
                ParsedQuery::empty(query)
            }
        }
    }

    fn name(&self) -> &'static str {
        "LlmParser"
    }

    fn is_available(&self) -> bool {
        self.is_server_available()
    }
}

/// Extract JSON from potentially messy LLM output
fn extract_json(output: &str) -> String {
    // Remove markdown code blocks if present
    let cleaned = output
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Find the first { and matching }
    if let Some(start) = cleaned.find('{') {
        let mut depth = 0;
        let mut end = start;
        // char_indices gives (byte_offset, char) — safe for multi-byte content
        for (byte_offset, c) in cleaned[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + byte_offset + c.len_utf8();
                        break;
                    }
                }
                _ => {}
            }
        }
        cleaned[start..end].to_string()
    } else {
        cleaned.to_string()
    }
}

/// Simple stemming using rust_stemmers
fn stem_word(word: &str) -> String {
    use rust_stemmers::{Algorithm, Stemmer};
    thread_local! {
        static STEMMER: Stemmer = Stemmer::create(Algorithm::English);
    }
    STEMMER.with(|s| s.stem(&word.to_lowercase()).to_string())
}

/// Parse entity type string
fn parse_entity_type(s: &str) -> EntityType {
    match s.to_lowercase().as_str() {
        "person" => EntityType::Person,
        "place" => EntityType::Place,
        "thing" => EntityType::Thing,
        "event" => EntityType::Event,
        "time" => EntityType::Time,
        _ => EntityType::Unknown,
    }
}

/// Parse direction string
fn parse_direction(s: &str) -> TimeDirection {
    match s.to_lowercase().as_str() {
        "past" => TimeDirection::Past,
        "future" => TimeDirection::Future,
        "current" => TimeDirection::Current,
        _ => TimeDirection::Past,
    }
}

/// Parse temporal intent string
fn parse_temporal_intent(s: &str) -> TemporalIntent {
    match s.to_lowercase().as_str() {
        "when_question" => TemporalIntent::WhenQuestion,
        "specific_time" => TemporalIntent::SpecificTime,
        "ordering" => TemporalIntent::Ordering,
        "duration" => TemporalIntent::Duration,
        _ => TemporalIntent::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json() {
        let output = r#"Here is the JSON: {"entities": [], "confidence": 0.9} and some more text"#;
        let json = extract_json(output);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn test_extract_json_with_markdown() {
        let output = r#"```json
{"entities": [], "confidence": 0.9}
```"#;
        let json = extract_json(output);
        assert_eq!(json, r#"{"entities": [], "confidence": 0.9}"#);
    }

    #[test]
    fn test_parse_entity_type() {
        assert_eq!(parse_entity_type("person"), EntityType::Person);
        assert_eq!(parse_entity_type("PLACE"), EntityType::Place);
        assert_eq!(parse_entity_type("unknown_type"), EntityType::Unknown);
    }
}
