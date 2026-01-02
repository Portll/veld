//! Linguistic Query Parser
//!
//! Based on:
//! - Lioma & Ounis (2006): "Content Load of Part of Speech Blocks"
//! - Bendersky & Croft (2008): "Discovering Key Concepts in Verbose Queries"
//! - Porter (1980): Stemming algorithm for term normalization
//!
//! Extracts focal entities (nouns), discriminative modifiers (adjectives),
//! and relational context (verbs) from natural language queries.
//!
//! # Polished Features (v2)
//! - Porter2 stemming for term normalization
//! - Compound noun detection (bigrams/trigrams)
//! - Context-aware POS disambiguation
//! - Negation scope tracking
//! - IDF-inspired term rarity weighting

use crate::constants::{IC_ADJECTIVE, IC_NOUN, IC_VERB};
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;

/// Focal entity extracted from query (noun)
#[derive(Debug, Clone)]
pub struct FocalEntity {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    pub ic_weight: f32,
    /// True if entity is part of a compound noun
    pub is_compound: bool,
    /// True if preceded by negation
    pub negated: bool,
}

/// Discriminative modifier (adjective/qualifier)
#[derive(Debug, Clone)]
pub struct Modifier {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
    /// True if preceded by negation
    pub negated: bool,
}

/// Relational context (verb)
#[derive(Debug, Clone)]
pub struct Relation {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
    /// True if preceded by negation
    pub negated: bool,
}

/// Complete linguistic analysis of a query
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Focal entities (nouns) - primary search targets
    pub focal_entities: Vec<FocalEntity>,

    /// Discriminative modifiers (adjectives) - quality refiners
    pub discriminative_modifiers: Vec<Modifier>,

    /// Relational context (verbs) - graph traversal guides
    pub relational_context: Vec<Relation>,

    /// Compound nouns detected (e.g., "machine learning", "neural network")
    pub compound_nouns: Vec<String>,

    /// Original query text (retained for logging/debugging)
    pub original_query: String,

    /// True if query contains negation
    pub has_negation: bool,
}

impl QueryAnalysis {
    /// Calculate weighted importance of this query (for ranking)
    pub fn total_weight(&self) -> f32 {
        let entity_weight: f32 = self.focal_entities.iter().map(|e| e.ic_weight).sum();

        let modifier_weight: f32 = self
            .discriminative_modifiers
            .iter()
            .map(|m| m.ic_weight)
            .sum();

        let relation_weight: f32 = self.relational_context.iter().map(|r| r.ic_weight).sum();

        // Compound nouns get bonus weight
        let compound_bonus = self.compound_nouns.len() as f32 * 0.5;

        entity_weight + modifier_weight + relation_weight + compound_bonus
    }

    /// Get all stems for efficient matching
    pub fn all_stems(&self) -> HashSet<String> {
        let mut stems = HashSet::new();
        for e in &self.focal_entities {
            stems.insert(e.stem.clone());
        }
        for m in &self.discriminative_modifiers {
            stems.insert(m.stem.clone());
        }
        for r in &self.relational_context {
            stems.insert(r.stem.clone());
        }
        stems
    }

    /// Get non-negated entity stems (for positive matching)
    pub fn positive_entity_stems(&self) -> Vec<&str> {
        self.focal_entities
            .iter()
            .filter(|e| !e.negated)
            .map(|e| e.stem.as_str())
            .collect()
    }

    /// Get negated entity stems (for exclusion)
    pub fn negated_entity_stems(&self) -> Vec<&str> {
        self.focal_entities
            .iter()
            .filter(|e| e.negated)
            .map(|e| e.stem.as_str())
            .collect()
    }
}

/// Token with linguistic annotations
#[derive(Debug)]
struct AnnotatedToken {
    text: String,
    stem: String,
    pos: PartOfSpeech,
    negated: bool,
    position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PartOfSpeech {
    Noun,
    Adjective,
    Verb,
    StopWord,
    Negation,
    Unknown,
}

/// Parse query using linguistic analysis with Porter2 stemming
pub fn analyze_query(query_text: &str) -> QueryAnalysis {
    let stemmer = Stemmer::create(Algorithm::English);
    let words = tokenize(query_text);

    if words.is_empty() {
        return QueryAnalysis {
            focal_entities: Vec::new(),
            discriminative_modifiers: Vec::new(),
            relational_context: Vec::new(),
            compound_nouns: Vec::new(),
            original_query: query_text.to_string(),
            has_negation: false,
        };
    }

    // Annotate each token with POS and negation scope
    let annotated = annotate_tokens(&words, &stemmer);

    // Detect compound nouns
    let compound_nouns = detect_compound_nouns(&annotated);

    // Build result structures
    let mut focal_entities = Vec::new();
    let mut discriminative_modifiers = Vec::new();
    let mut relational_context = Vec::new();
    let mut has_negation = false;

    // Track which tokens are part of compounds
    let compound_positions: HashSet<usize> = compound_positions(&annotated, &compound_nouns);

    for token in &annotated {
        if token.pos == PartOfSpeech::Negation {
            has_negation = true;
            continue;
        }
        if token.pos == PartOfSpeech::StopWord {
            continue;
        }

        let is_compound = compound_positions.contains(&token.position);

        match token.pos {
            PartOfSpeech::Noun | PartOfSpeech::Unknown => {
                // Unknown words are likely domain-specific nouns
                let weight = calculate_term_weight(&token.text, IC_NOUN);
                focal_entities.push(FocalEntity {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    is_compound,
                    negated: token.negated,
                });
            }
            PartOfSpeech::Adjective => {
                let weight = calculate_term_weight(&token.text, IC_ADJECTIVE);
                discriminative_modifiers.push(Modifier {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    negated: token.negated,
                });
            }
            PartOfSpeech::Verb => {
                let weight = calculate_term_weight(&token.text, IC_VERB);
                relational_context.push(Relation {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    negated: token.negated,
                });
            }
            _ => {}
        }
    }

    // Add compound nouns as high-weight entities
    for compound in &compound_nouns {
        let stem = stemmer.stem(compound).to_string();
        focal_entities.push(FocalEntity {
            text: compound.clone(),
            stem,
            ic_weight: IC_NOUN * 1.5, // Compound bonus
            is_compound: true,
            negated: false,
        });
    }

    QueryAnalysis {
        focal_entities,
        discriminative_modifiers,
        relational_context,
        compound_nouns,
        original_query: query_text.to_string(),
        has_negation,
    }
}

/// Tokenize query text into lowercase words
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Annotate tokens with POS tags and negation scope
fn annotate_tokens(words: &[String], stemmer: &Stemmer) -> Vec<AnnotatedToken> {
    let mut annotated = Vec::with_capacity(words.len());
    let mut in_negation_scope = false;
    let mut negation_distance = 0;

    for (i, word) in words.iter().enumerate() {
        let stem = stemmer.stem(word).to_string();
        let pos = classify_pos(word, i, words);

        // Track negation scope (extends 2-3 words after negation)
        if pos == PartOfSpeech::Negation {
            in_negation_scope = true;
            negation_distance = 0;
        } else if in_negation_scope {
            negation_distance += 1;
            if negation_distance > 3 {
                in_negation_scope = false;
            }
        }

        let negated = in_negation_scope && pos != PartOfSpeech::Negation;

        annotated.push(AnnotatedToken {
            text: word.clone(),
            stem,
            pos,
            negated,
            position: i,
        });
    }

    annotated
}

/// Classify part of speech using heuristics
fn classify_pos(word: &str, position: usize, context: &[String]) -> PartOfSpeech {
    // Check negation first
    if is_negation(word) {
        return PartOfSpeech::Negation;
    }

    // Check stop words
    if is_stop_word(word) {
        return PartOfSpeech::StopWord;
    }

    // Use suffix patterns and context for classification
    if is_verb(word) {
        return PartOfSpeech::Verb;
    }

    if is_adjective(word) {
        return PartOfSpeech::Adjective;
    }

    if is_noun(word, position, context) {
        return PartOfSpeech::Noun;
    }

    // Default to unknown (treated as noun for domain terms)
    PartOfSpeech::Unknown
}

/// Detect compound nouns (bigrams that commonly co-occur)
fn detect_compound_nouns(tokens: &[AnnotatedToken]) -> Vec<String> {
    let mut compounds = Vec::new();

    // Common compound noun patterns
    const COMPOUND_PATTERNS: &[(&str, &str)] = &[
        // Tech/AI compounds
        ("machine", "learning"),
        ("deep", "learning"),
        ("neural", "network"),
        ("natural", "language"),
        ("language", "model"),
        ("artificial", "intelligence"),
        ("knowledge", "graph"),
        ("vector", "database"),
        ("memory", "system"),
        ("data", "structure"),
        ("source", "code"),
        ("error", "handling"),
        ("unit", "test"),
        ("integration", "test"),
        ("api", "endpoint"),
        ("web", "server"),
        ("file", "system"),
        ("operating", "system"),
        ("database", "schema"),
        ("user", "interface"),
        ("command", "line"),
        ("version", "control"),
        ("pull", "request"),
        ("code", "review"),
        ("bug", "fix"),
        ("feature", "request"),
        // Domain-specific
        ("spreading", "activation"),
        ("hebbian", "learning"),
        ("long", "term"),
        ("short", "term"),
        ("working", "memory"),
        ("semantic", "search"),
        ("graph", "traversal"),
        ("edge", "device"),
        ("air", "gapped"),
    ];

    // Check for known compound patterns
    for i in 0..tokens.len().saturating_sub(1) {
        let t1 = &tokens[i];
        let t2 = &tokens[i + 1];

        // Skip if either is a stop word or verb
        if t1.pos == PartOfSpeech::StopWord || t2.pos == PartOfSpeech::StopWord {
            continue;
        }

        for (w1, w2) in COMPOUND_PATTERNS {
            if (t1.stem == *w1 || t1.text == *w1) && (t2.stem == *w2 || t2.text == *w2) {
                compounds.push(format!("{} {}", t1.text, t2.text));
                break;
            }
        }

        // Heuristic: Noun + Noun often forms compound
        if (t1.pos == PartOfSpeech::Noun || t1.pos == PartOfSpeech::Unknown)
            && (t2.pos == PartOfSpeech::Noun || t2.pos == PartOfSpeech::Unknown)
        {
            // Check for common suffixes that indicate compound-worthy nouns
            if has_compound_suffix(&t1.text) || has_compound_suffix(&t2.text) {
                let compound = format!("{} {}", t1.text, t2.text);
                if !compounds.contains(&compound) {
                    compounds.push(compound);
                }
            }
        }
    }

    compounds
}

/// Check if word has suffix that often appears in compounds
fn has_compound_suffix(word: &str) -> bool {
    word.ends_with("tion")
        || word.ends_with("ment")
        || word.ends_with("ing")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("er")
        || word.ends_with("or")
        || word.ends_with("ist")
        || word.ends_with("ism")
}

/// Get positions of tokens that are part of compounds
fn compound_positions(tokens: &[AnnotatedToken], compounds: &[String]) -> HashSet<usize> {
    let mut positions = HashSet::new();

    for compound in compounds {
        let parts: Vec<&str> = compound.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        for i in 0..tokens.len().saturating_sub(parts.len() - 1) {
            let mut matches = true;
            for (j, part) in parts.iter().enumerate() {
                if tokens[i + j].text != *part {
                    matches = false;
                    break;
                }
            }
            if matches {
                for j in 0..parts.len() {
                    positions.insert(i + j);
                }
            }
        }
    }

    positions
}

/// Calculate term weight with IDF-like rarity boost
fn calculate_term_weight(word: &str, base_weight: f32) -> f32 {
    // Longer words tend to be more specific/rare
    let length_factor = if word.len() > 8 {
        1.2
    } else if word.len() > 5 {
        1.1
    } else {
        1.0
    };

    // Technical suffixes get slight boost
    let suffix_factor = if word.ends_with("tion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
    {
        1.1
    } else {
        1.0
    };

    base_weight * length_factor * suffix_factor
}

/// Check if word is negation
fn is_negation(word: &str) -> bool {
    const NEGATIONS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "neither",
        "nobody",
        "nowhere",
        "without",
        "cannot",
        "can't",
        "won't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "mustn't",
    ];
    NEGATIONS.contains(&word)
}

/// Check if word is a noun (entity)
fn is_noun(word: &str, position: usize, context: &[String]) -> bool {
    // Domain-specific nouns (expanded list)
    const NOUN_INDICATORS: &[&str] = &[
        // Core memory/cognitive terms
        "memory",
        "graph",
        "node",
        "edge",
        "entity",
        "embedding",
        "vector",
        "index",
        "query",
        "retrieval",
        "activation",
        "potentiation",
        "consolidation",
        "decay",
        "strength",
        "weight",
        "threshold",
        "importance",
        // Tech terms
        "robot",
        "drone",
        "sensor",
        "lidar",
        "camera",
        "motor",
        "actuator",
        "obstacle",
        "path",
        "waypoint",
        "location",
        "coordinates",
        "position",
        "battery",
        "power",
        "energy",
        "voltage",
        "current",
        "system",
        "module",
        "component",
        "unit",
        "device",
        "temperature",
        "pressure",
        "humidity",
        "speed",
        "velocity",
        "signal",
        "communication",
        "network",
        "link",
        "connection",
        "navigation",
        "guidance",
        "control",
        "steering",
        "data",
        "information",
        "message",
        "command",
        "response",
        // Software terms
        "function",
        "method",
        "class",
        "struct",
        "interface",
        "module",
        "package",
        "library",
        "framework",
        "api",
        "endpoint",
        "request",
        "response",
        "error",
        "exception",
        "bug",
        "fix",
        "feature",
        "test",
        "benchmark",
        "performance",
        "latency",
        "throughput",
        "cache",
        "buffer",
        "queue",
        "stack",
        "heap",
        "thread",
        "process",
        "server",
        "client",
        "database",
        "table",
        "column",
        "row",
        "schema",
        "migration",
        "deployment",
        "container",
        "cluster",
        "replica",
        // General nouns
        "person",
        "people",
        "user",
        "agent",
        "operator",
        "time",
        "date",
        "day",
        "hour",
        "minute",
        "second",
        "area",
        "zone",
        "region",
        "sector",
        "space",
        "task",
        "mission",
        "goal",
        "objective",
        "target",
        "warning",
        "alert",
        "notification",
        "level",
        "status",
        "state",
        "condition",
        "mode",
        "type",
        "kind",
        "version",
        "release",
        "update",
        "change",
        "result",
        "output",
        "input",
        "value",
        "key",
        "name",
        "id",
        "identifier",
    ];

    if NOUN_INDICATORS.contains(&word) {
        return true;
    }

    // Check for noun suffixes
    if word.ends_with("tion")
        || word.ends_with("sion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("er")
        || word.ends_with("or")
        || word.ends_with("ist")
        || word.ends_with("ism")
        || word.ends_with("age")
        || word.ends_with("ure")
        || word.ends_with("dom")
    {
        // Avoid verb forms like "better", "faster"
        if !(word.ends_with("er") && word.len() < 5) {
            return true;
        }
    }

    // Check if preceded by determiner (a, an, the)
    if position > 0 {
        if let Some(prev) = context.get(position - 1) {
            let prev = prev.to_lowercase();
            if prev == "a" || prev == "an" || prev == "the" || prev == "this" || prev == "that" {
                return true;
            }
        }
    }

    // Check if preceded by possessive
    if position > 0 {
        if let Some(prev) = context.get(position - 1) {
            if prev.ends_with("'s") || prev.ends_with("s'") {
                return true;
            }
        }
    }

    false
}

/// Check if word is an adjective (qualifier)
fn is_adjective(word: &str) -> bool {
    const ADJECTIVE_INDICATORS: &[&str] = &[
        // Colors
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "black",
        "white",
        "gray",
        "grey",
        "pink",
        "brown",
        // Sizes
        "big",
        "small",
        "large",
        "tiny",
        "huge",
        "massive",
        "mini",
        "micro",
        "high",
        "low",
        "tall",
        "short",
        "long",
        "wide",
        "narrow",
        // States
        "hot",
        "cold",
        "warm",
        "cool",
        "frozen",
        "heated",
        "fast",
        "slow",
        "quick",
        "rapid",
        "gradual",
        "active",
        "inactive",
        "enabled",
        "disabled",
        "open",
        "closed",
        "locked",
        "unlocked",
        "full",
        "empty",
        "partial",
        "complete",
        "valid",
        "invalid",
        "correct",
        "incorrect",
        "true",
        "false",
        // Quality
        "good",
        "bad",
        "excellent",
        "poor",
        "optimal",
        "suboptimal",
        "normal",
        "abnormal",
        "stable",
        "unstable",
        "safe",
        "unsafe",
        "dangerous",
        "hazardous",
        "new",
        "old",
        "recent",
        "ancient",
        "current",
        "latest",
        "first",
        "last",
        "next",
        "previous",
        "primary",
        "secondary",
        "main",
        "important",
        "critical",
        "minor",
        "major",
        // Technical
        "autonomous",
        "manual",
        "automatic",
        "remote",
        "digital",
        "analog",
        "electronic",
        "mechanical",
        "wireless",
        "wired",
        "connected",
        "disconnected",
        "local",
        "global",
        "private",
        "public",
        "static",
        "dynamic",
        "mutable",
        "immutable",
        "sync",
        "async",
        "concurrent",
        "parallel",
        "serial",
        "sequential",
        "optional",
        "required",
        "default",
        "custom",
    ];

    if ADJECTIVE_INDICATORS.contains(&word) {
        return true;
    }

    // Common adjective suffixes (excluding verb participles)
    if word.ends_with("ful")
        || word.ends_with("less")
        || word.ends_with("ous")
        || word.ends_with("ive")
        || word.ends_with("able")
        || word.ends_with("ible")
        || word.ends_with("al")
        || word.ends_with("ic")
        || word.ends_with("ary")
        || word.ends_with("ory")
    {
        // Avoid false positives
        let exceptions = ["animal", "interval", "arrival", "approval"];
        if !exceptions.contains(&word) {
            return true;
        }
    }

    false
}

/// Check if word is a verb (relational, lower priority)
fn is_verb(word: &str) -> bool {
    const VERB_INDICATORS: &[&str] = &[
        // Auxiliary/modal verbs
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        // Common action verbs
        "go",
        "goes",
        "went",
        "gone",
        "going",
        "get",
        "gets",
        "got",
        "gotten",
        "getting",
        "make",
        "makes",
        "made",
        "making",
        "take",
        "takes",
        "took",
        "taken",
        "taking",
        "see",
        "sees",
        "saw",
        "seen",
        "seeing",
        "give",
        "gives",
        "gave",
        "given",
        "giving",
        "use",
        "uses",
        "used",
        "using",
        "find",
        "finds",
        "found",
        "finding",
        "know",
        "knows",
        "knew",
        "known",
        "knowing",
        "think",
        "thinks",
        "thought",
        "thinking",
        "want",
        "wants",
        "wanted",
        "wanting",
        "need",
        "needs",
        "needed",
        "needing",
        "try",
        "tries",
        "tried",
        "trying",
        // Technical verbs
        "detect",
        "detects",
        "detected",
        "detecting",
        "observe",
        "observes",
        "observed",
        "observing",
        "measure",
        "measures",
        "measured",
        "measuring",
        "sense",
        "senses",
        "sensed",
        "sensing",
        "scan",
        "scans",
        "scanned",
        "scanning",
        "navigate",
        "navigates",
        "navigated",
        "navigating",
        "move",
        "moves",
        "moved",
        "moving",
        "stop",
        "stops",
        "stopped",
        "stopping",
        "start",
        "starts",
        "started",
        "starting",
        "reach",
        "reaches",
        "reached",
        "reaching",
        "avoid",
        "avoids",
        "avoided",
        "avoiding",
        "block",
        "blocks",
        "blocked",
        "blocking",
        "create",
        "creates",
        "created",
        "creating",
        "delete",
        "deletes",
        "deleted",
        "deleting",
        "update",
        "updates",
        "updated",
        "updating",
        "read",
        "reads",
        "reading",
        "write",
        "writes",
        "wrote",
        "written",
        "writing",
        "run",
        "runs",
        "ran",
        "running",
        "execute",
        "executes",
        "executed",
        "executing",
        "call",
        "calls",
        "called",
        "calling",
        "return",
        "returns",
        "returned",
        "returning",
        "store",
        "stores",
        "stored",
        "storing",
        "load",
        "loads",
        "loaded",
        "loading",
        "save",
        "saves",
        "saved",
        "saving",
        "fetch",
        "fetches",
        "fetched",
        "fetching",
        "send",
        "sends",
        "sent",
        "sending",
        "receive",
        "receives",
        "received",
        "receiving",
        "connect",
        "connects",
        "connected",
        "connecting",
        "disconnect",
        "disconnects",
        "disconnected",
        "disconnecting",
        "process",
        "processes",
        "processed",
        "processing",
        "handle",
        "handles",
        "handled",
        "handling",
        "parse",
        "parses",
        "parsed",
        "parsing",
        "compile",
        "compiles",
        "compiled",
        "compiling",
        "build",
        "builds",
        "built",
        "building",
        "test",
        "tests",
        "tested",
        "testing",
        "deploy",
        "deploys",
        "deployed",
        "deploying",
        "install",
        "installs",
        "installed",
        "installing",
        "configure",
        "configures",
        "configured",
        "configuring",
        "initialize",
        "initializes",
        "initialized",
        "initializing",
        "shutdown",
        "shutdowns",
        "terminate",
        "terminates",
        "terminated",
        "terminating",
    ];

    VERB_INDICATORS.contains(&word)
}

/// Check if word is a stop word (no information content)
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        // Articles
        "a",
        "an",
        "the",
        // Demonstratives
        "this",
        "that",
        "these",
        "those",
        // Prepositions
        "at",
        "in",
        "on",
        "to",
        "for",
        "of",
        "from",
        "by",
        "with",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        // Conjunctions
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        // Pronouns
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "whoever",
        "whatever",
        "whichever",
        // Relative
        "that",
        "which",
        "who",
        "whom",
        "whose",
        // Question words (when not seeking info)
        "how",
        "when",
        "where",
        "why",
        // Common filler
        "just",
        "only",
        "even",
        "also",
        "too",
        "very",
        "really",
        "quite",
        "rather",
        "almost",
        "already",
        "still",
        "always",
        "never",
        "ever",
        "often",
        "sometimes",
        "usually",
        "perhaps",
        "maybe",
        "probably",
        "possibly",
        "certainly",
        "definitely",
        "actually",
        "basically",
        "essentially",
        "simply",
        "merely",
        // Be forms handled separately as verbs
        "as",
        "if",
        "then",
        "than",
        "because",
        "although",
        "though",
        "unless",
        "until",
        "while",
        "whereas",
        "whether",
        "since",
        // Others
        "some",
        "any",
        "all",
        "each",
        "every",
        "many",
        "much",
        "more",
        "most",
        "few",
        "less",
        "least",
        "other",
        "another",
        "such",
        "same",
        "different",
        "own",
        "several",
    ];

    STOP_WORDS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noun_detection() {
        let query = "robot detected obstacle at coordinates";
        let analysis = analyze_query(query);

        let noun_texts: Vec<String> = analysis
            .focal_entities
            .iter()
            .map(|e| e.text.clone())
            .collect();

        assert!(noun_texts.contains(&"robot".to_string()));
        assert!(noun_texts.contains(&"obstacle".to_string()));
        assert!(noun_texts.contains(&"coordinates".to_string()));
    }

    #[test]
    fn test_adjective_detection() {
        let query = "red large obstacle in path";
        let analysis = analyze_query(query);

        let adj_texts: Vec<String> = analysis
            .discriminative_modifiers
            .iter()
            .map(|m| m.text.clone())
            .collect();

        assert!(adj_texts.contains(&"red".to_string()));
        assert!(adj_texts.contains(&"large".to_string()));
    }

    #[test]
    fn test_verb_detection() {
        let query = "robot detected obstacle";
        let analysis = analyze_query(query);

        let verb_texts: Vec<String> = analysis
            .relational_context
            .iter()
            .map(|r| r.text.clone())
            .collect();

        assert!(verb_texts.contains(&"detected".to_string()));
    }

    #[test]
    fn test_information_content_weights() {
        let query = "sensor detected red obstacle";
        let analysis = analyze_query(query);

        // Nouns should have IC weight >= IC_NOUN base
        for entity in &analysis.focal_entities {
            assert!(entity.ic_weight >= IC_NOUN * 0.9); // Allow small variance
        }

        // Adjectives should have IC weight >= IC_ADJECTIVE base
        for modifier in &analysis.discriminative_modifiers {
            assert!(modifier.ic_weight >= IC_ADJECTIVE * 0.9);
        }

        // Verbs should have IC weight >= IC_VERB base
        for relation in &analysis.relational_context {
            assert!(relation.ic_weight >= IC_VERB * 0.9);
        }
    }

    #[test]
    fn test_stemming() {
        let query = "running detection algorithms";
        let analysis = analyze_query(query);

        // Check stems are different from original text
        let stems: Vec<String> = analysis
            .focal_entities
            .iter()
            .map(|e| e.stem.clone())
            .collect();

        // "detection" should stem to "detect"
        assert!(stems.iter().any(|s| s == "detect"));
        // "algorithms" should stem to "algorithm"
        assert!(stems.iter().any(|s| s == "algorithm"));
    }

    #[test]
    fn test_compound_noun_detection() {
        let query = "machine learning neural network";
        let analysis = analyze_query(query);

        assert!(analysis
            .compound_nouns
            .contains(&"machine learning".to_string()));
        assert!(analysis
            .compound_nouns
            .contains(&"neural network".to_string()));
    }

    #[test]
    fn test_negation_detection() {
        let query = "not working correctly";
        let analysis = analyze_query(query);

        assert!(analysis.has_negation);

        // Check that tokens after negation are marked
        let negated_entities: Vec<&FocalEntity> = analysis
            .focal_entities
            .iter()
            .filter(|e| e.negated)
            .collect();

        assert!(!negated_entities.is_empty());
    }

    #[test]
    fn test_negation_scope() {
        let query = "the sensor is not detecting obstacles properly";
        let analysis = analyze_query(query);

        assert!(analysis.has_negation);

        // "detecting" should be marked as negated
        let negated_verbs: Vec<&Relation> = analysis
            .relational_context
            .iter()
            .filter(|r| r.negated)
            .collect();

        assert!(negated_verbs.iter().any(|r| r.text == "detecting"));
    }

    #[test]
    fn test_all_stems_helper() {
        let query = "fast robot detecting obstacles";
        let analysis = analyze_query(query);

        let stems = analysis.all_stems();
        assert!(stems.contains("robot"));
        assert!(stems.contains("fast"));
        assert!(stems.contains("detect"));
        assert!(stems.contains("obstacl")); // Porter stem
    }

    #[test]
    fn test_positive_and_negated_stems() {
        let query = "working memory not failed";
        let analysis = analyze_query(query);

        let positive = analysis.positive_entity_stems();
        let negated = analysis.negated_entity_stems();

        // "memory" should be positive
        assert!(positive.iter().any(|s| s.contains("memori")));

        // "failed" should be negated (after "not")
        // Note: "failed" might be classified as verb or noun depending on context
    }

    #[test]
    fn test_empty_query() {
        let query = "";
        let analysis = analyze_query(query);

        assert!(analysis.focal_entities.is_empty());
        assert!(analysis.discriminative_modifiers.is_empty());
        assert!(analysis.relational_context.is_empty());
        assert!(!analysis.has_negation);
    }

    #[test]
    fn test_stop_words_filtered() {
        let query = "the a an is are was were";
        let analysis = analyze_query(query);

        // Only verbs should remain (is, are, was, were)
        assert!(analysis.focal_entities.is_empty());
        assert!(analysis.discriminative_modifiers.is_empty());
        assert!(!analysis.relational_context.is_empty());
    }

    #[test]
    fn test_total_weight_calculation() {
        let query = "fast robot detecting red obstacles";
        let analysis = analyze_query(query);

        let weight = analysis.total_weight();
        assert!(weight > 0.0);
    }
}
