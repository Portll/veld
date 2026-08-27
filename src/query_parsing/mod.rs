//! Modular Query Parsing System
//!
//! Provides a trait-based abstraction for query parsing, allowing easy swapping
//! between rule-based and LLM-based implementations.
//!
//! # Architecture
//! ```text
//! Query → QueryParser (trait) → ParsedQuery
//!              ↓
//!     ┌────────┴────────┐
//!     │                 │
//! RuleBasedParser   LlmParser
//! (YAKE/regex)      (Qwen 1.5B)
//! ```
//!
//! # Usage
//! ```rust,ignore
//! let parser = create_parser(ParserConfig::default());
//! let parsed = parser.parse("When did Melanie paint a sunrise?", Some(conv_date))?;
//! ```

mod llm_parser;
mod parser_trait;
mod rule_based;

pub use llm_parser::LlmParser;
pub use parser_trait::*;
pub use rule_based::RuleBasedParser;

use std::sync::Arc;

/// Parser implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParserType {
    /// Rule-based parsing using YAKE, regex, and heuristics (default)
    #[default]
    RuleBased,
    /// LLM-based parsing using Qwen 1.5B or similar
    Llm,
}

/// Configuration for the query parser
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Which parser implementation to use
    pub parser_type: ParserType,
    /// Base URL of the LLM server (only used if parser_type is Llm)
    pub llm_endpoint: Option<String>,
    /// Model identifier (only used if parser_type is Llm)
    pub llm_model: Option<String>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            parser_type: ParserType::RuleBased,
            llm_endpoint: None,
            llm_model: None,
        }
    }
}

impl ParserConfig {
    /// Create config for rule-based parser
    pub fn rule_based() -> Self {
        Self::default()
    }

    /// Create config for LLM parser with explicit endpoint and model
    pub fn llm(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            parser_type: ParserType::Llm,
            llm_endpoint: Some(endpoint.into()),
            llm_model: Some(model.into()),
        }
    }

    /// Create config for LLM parser from environment variables
    pub fn llm_from_env() -> Self {
        Self {
            parser_type: ParserType::Llm,
            llm_endpoint: None,
            llm_model: None,
        }
    }
}

/// Create a parser based on configuration
pub fn create_parser(config: ParserConfig) -> Arc<dyn QueryParser> {
    match config.parser_type {
        ParserType::RuleBased => Arc::new(RuleBasedParser::new()),
        ParserType::Llm => {
            let parser = match (config.llm_endpoint, config.llm_model) {
                (Some(endpoint), Some(model)) => {
                    LlmParser::new(&endpoint, &model)
                }
                _ => LlmParser::from_env(),
            };
            Arc::new(parser)
        }
    }
}
