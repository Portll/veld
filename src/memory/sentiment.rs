//! Derived sentiment: stance toward named entities, read from memory text.
//!
//! Each mention of a target takes its polarity from a small hand-built lexicon, scaled by
//! the words around it: boosters ("really"), dampeners ("kind of"), negation, "used to",
//! exclamation and emphasis. Opinions aggregate per entity under an optional recency
//! half-life and project onto a Likert scale of any width when read.
//!
//! Nothing here is stored. Aggregates are recomputed from episode text on request, so a
//! forgotten memory leaves no residue, a lexicon change applies to history at once, and
//! replication and engine migration have nothing extra to carry.

use std::ops::{Range, RangeInclusive};

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Likert widths a caller may ask for.
pub const LIKERT_SCALES: RangeInclusive<u8> = 3..=11;

/// Raw lexicon scores (about ±4) map into (-1, 1) through `x / sqrt(x² + α)`. A smaller α
/// spreads the common words ("like", "love") further apart on the Likert scale.
const SQUASH_ALPHA: f32 = 8.0;
const NEGATION_SCALAR: f32 = -0.74;
const MODIFIER_WINDOW: usize = 3;
const HOLDER_REACH: usize = 4;
const HISTORICAL_WEIGHT: f32 = 0.3;
const EXCLAMATION_STEP: f32 = 0.1;
const MAX_EXCLAMATIONS: u8 = 3;
const CAPS_EMPHASIS: f32 = 1.2;
/// The losing side of a comparison is disfavoured, but less firmly than the winner is favoured.
const COMPARATIVE_LOSER_SHARE: f32 = 0.6;
const BEFORE_CONTRAST_WEIGHT: f32 = 0.5;
const AFTER_CONTRAST_WEIGHT: f32 = 1.5;
/// One full-weight opinion gives confidence 0.5 before spread is counted.
const CONFIDENCE_PRIOR_WEIGHT: f32 = 1.0;
const REVERSAL_FLOOR: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CueClass {
    /// Attitude verbs: the holder comes before, the target after ("I love Rust").
    Stance,
    /// Evaluations that attach to a target on either side ("Rust is great").
    Evaluative,
    /// One side wins and the other loses ("better than", "prefer X over Y").
    Comparative,
}

use CueClass::{Comparative as C, Evaluative as E, Stance as S};

/// Sorted for binary search; `lexicon_tables_are_sorted_and_unique` holds it there.
const LEXICON: &[(&str, f32, CueClass)] = &[
    ("abhor", -3.4, S),
    ("admire", 2.4, S),
    ("adore", 3.4, S),
    ("afraid", -1.6, E),
    ("amazing", 3.2, E),
    ("angry", -2.6, E),
    ("annoyed", -2.0, E),
    ("annoying", -2.2, E),
    ("anxious", -1.4, E),
    ("appreciate", 2.0, S),
    ("atrocious", -3.3, E),
    ("awesome", 3.1, E),
    ("awful", -3.2, E),
    ("bad", -2.2, E),
    ("beautiful", 2.9, E),
    ("best", 3.0, E),
    ("better", 1.8, C),
    ("bored", -1.4, E),
    ("boring", -1.8, E),
    ("brilliant", 3.0, E),
    ("broken", -1.6, E),
    ("buggy", -1.8, E),
    ("cherish", 3.0, S),
    ("clunky", -1.6, E),
    ("confusing", -1.6, E),
    ("cool", 1.6, E),
    ("crappy", -2.6, E),
    ("decent", 1.2, E),
    ("delight", 2.8, E),
    ("delighted", 2.8, E),
    ("delightful", 2.9, E),
    ("despise", -3.4, S),
    ("detest", -3.3, S),
    ("disappointed", -2.2, E),
    ("disappointing", -2.4, E),
    ("disaster", -3.0, E),
    ("disgusting", -3.0, E),
    ("dislike", -2.0, S),
    ("dread", -2.4, S),
    ("dreadful", -3.1, E),
    ("elegant", 2.2, E),
    ("enjoy", 2.4, S),
    ("enjoyable", 2.4, E),
    ("excellent", 3.2, E),
    ("exceptional", 3.1, E),
    ("excited", 2.4, E),
    ("fan", 1.8, E),
    ("fantastic", 3.3, E),
    ("favorite", 2.6, E),
    ("favourite", 2.6, E),
    ("fond", 2.2, E),
    ("frustrated", -2.2, E),
    ("frustrating", -2.4, E),
    ("fun", 2.2, E),
    ("furious", -3.0, E),
    ("garbage", -2.6, E),
    ("glad", 2.0, E),
    ("good", 1.9, E),
    ("gorgeous", 3.0, E),
    ("grateful", 2.2, E),
    ("great", 3.1, E),
    ("gross", -2.4, E),
    ("happy", 2.4, E),
    ("hassle", -1.6, E),
    ("hate", -3.2, S),
    ("helpful", 2.0, E),
    ("horrendous", -3.3, E),
    ("horrible", -3.3, E),
    ("impressive", 2.6, E),
    ("incredible", 3.0, E),
    ("interesting", 1.4, E),
    ("intuitive", 1.8, E),
    ("irritating", -2.2, E),
    ("joy", 2.6, E),
    ("keen", 1.6, E),
    ("like", 2.0, S),
    ("loathe", -3.4, S),
    ("lousy", -2.4, E),
    ("love", 3.2, S),
    ("lovely", 2.8, E),
    ("mediocre", -1.4, E),
    ("meh", -1.0, E),
    ("mess", -1.6, E),
    ("miserable", -2.8, E),
    ("nasty", -2.6, E),
    ("nice", 1.8, E),
    ("nightmare", -2.8, E),
    ("ok", 0.6, E),
    ("okay", 0.6, E),
    ("outstanding", 3.2, E),
    ("overrated", -1.6, E),
    ("painful", -2.0, E),
    ("pathetic", -2.8, E),
    ("perfect", 3.0, E),
    ("phenomenal", 3.3, E),
    ("pleasant", 2.0, E),
    ("pleased", 2.2, E),
    ("poor", -1.8, E),
    ("prefer", 1.6, C),
    ("proud", 2.2, E),
    ("recommend", 2.0, S),
    ("regret", -2.0, S),
    ("reliable", 1.8, E),
    ("resent", -2.4, S),
    ("rubbish", -2.4, E),
    ("sad", -2.0, E),
    ("satisfied", 1.8, E),
    ("scared", -1.8, E),
    ("slow", -1.0, E),
    ("solid", 1.4, E),
    ("stellar", 3.0, E),
    ("stressed", -1.6, E),
    ("suck", -2.6, E),
    ("superb", 3.2, E),
    ("terrible", -3.2, E),
    ("terrific", 3.0, E),
    ("thrilled", 3.0, E),
    ("trust", 1.8, S),
    ("ugly", -2.2, E),
    ("unhappy", -2.2, E),
    ("unreliable", -2.0, E),
    ("upset", -2.2, E),
    ("useful", 1.8, E),
    ("useless", -2.4, E),
    ("valuable", 2.0, E),
    ("wonderful", 3.2, E),
    ("worried", -1.4, E),
    ("worse", -1.8, C),
    ("worst", -3.2, E),
    ("worthless", -2.8, E),
    ("worthwhile", 1.8, E),
];

const EMOJI: &[(char, f32)] = &[
    ('☹', -1.8),
    ('❤', 3.0),
    ('👍', 2.0),
    ('👎', -2.0),
    ('😀', 2.4),
    ('😁', 2.4),
    ('😃', 2.4),
    ('😄', 2.4),
    ('😊', 2.2),
    ('😍', 3.2),
    ('😠', -3.0),
    ('😡', -3.0),
    ('😢', -2.2),
    ('😭', -2.4),
    ('🙁', -1.8),
    ('🙂', 1.6),
];

const BOOSTERS: &[(&str, f32)] = &[
    ("absolutely", 1.5),
    ("big", 1.3),
    ("completely", 1.4),
    ("deeply", 1.4),
    ("definitely", 1.2),
    ("especially", 1.2),
    ("extremely", 1.5),
    ("genuinely", 1.2),
    ("highly", 1.3),
    ("huge", 1.4),
    ("hugely", 1.4),
    ("incredibly", 1.5),
    ("insanely", 1.4),
    ("massively", 1.4),
    ("most", 1.2),
    ("particularly", 1.2),
    ("pretty", 1.1),
    ("quite", 1.1),
    ("really", 1.3),
    ("seriously", 1.3),
    ("so", 1.3),
    ("super", 1.3),
    ("thoroughly", 1.4),
    ("totally", 1.4),
    ("truly", 1.3),
    ("utterly", 1.5),
    ("very", 1.3),
];

const DAMPENERS: &[(&str, f32)] = &[
    ("barely", 0.3),
    ("bit", 0.6),
    ("fairly", 0.8),
    ("hardly", 0.3),
    ("kinda", 0.6),
    ("little", 0.6),
    ("marginally", 0.5),
    ("maybe", 0.7),
    ("mildly", 0.5),
    ("moderately", 0.7),
    ("partly", 0.6),
    ("perhaps", 0.7),
    ("probably", 0.85),
    ("rather", 0.8),
    ("scarcely", 0.3),
    ("slightly", 0.5),
    ("somewhat", 0.6),
    ("sorta", 0.6),
];

const HEDGE_PHRASE_FACTOR: f32 = 0.6;

const NEGATIONS: &[&str] = &[
    "aint", "cannot", "cant", "didnt", "doesnt", "dont", "isnt", "neither", "never", "no",
    "nobody", "none", "nor", "not", "nothing", "nowhere", "wasnt", "without", "wont",
];

const CONTRASTS: &[&str] = &["although", "but", "however", "though", "whereas"];

const FIRST_PERSON: &[&str] = &[
    "i", "i'd", "i'll", "i'm", "i've", "me", "my", "our", "us", "we", "we'd", "we'll", "we're",
    "we've",
];

const THIRD_PERSON: &[&str] = &[
    "he", "he'd", "he's", "her", "him", "it", "it's", "she", "she'd", "she's", "them", "they",
    "they'd", "they're", "they've",
];

/// Words passed over when looking back from an attitude verb for its subject.
const SUBJECT_SKIPS: &[&str] = &[
    "actually", "also", "always", "can", "could", "did", "do", "does", "even", "ever", "had",
    "has", "have", "just", "might", "must", "now", "should", "still", "will", "would",
];

/// The subset of `SUBJECT_SKIPS` that makes a following "like" a verb ("would like").
const AUXILIARIES: &[&str] = &[
    "can", "could", "did", "do", "does", "had", "has", "have", "might", "must", "should", "will",
    "would",
];

/// "into" is an attitude only after a copula: "I'm really into chess", not "went into town".
const INTO_LICENSORS: &[&str] = &[
    "am", "are", "be", "been", "he's", "i'm", "is", "she's", "they're", "was", "we're", "were",
    "you're",
];
const INTO_SCORE: f32 = 1.6;

const COMPARISON_CONNECTORS: &[&str] = &["instead", "over", "rather", "than", "to"];

/// Who holds an opinion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum Holder {
    /// The author of the text ("I love Rust").
    Speaker,
    /// A named person: a `Name:` speaker prefix, or a capitalised subject ("Alice hates Go").
    Named(String),
    /// Someone the text does not name ("they", "my team").
    Other,
}

/// A word near the cue that changed its strength or sign.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Modifier {
    /// `factor` is as applied: a booster inside a negation's scope weakens instead
    /// ("not very good" is milder than "not good").
    Booster {
        word: String,
        factor: f32,
    },
    Dampener {
        word: String,
        factor: f32,
    },
    Negation {
        word: String,
    },
    /// "used to": the stance was held, not necessarily still held.
    Historical,
    Exclamation {
        count: u8,
    },
    /// The cue was written in capitals inside text that is not.
    Emphasis,
    /// The target was the losing side of a comparison.
    ComparativeLoser,
}

/// One holder's stance toward one target, from one clause.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Opinion {
    /// The target as the caller named it.
    pub target: String,
    pub holder: Holder,
    /// The lexicon word or phrase that carried the polarity.
    pub cue: String,
    /// Score on the lexicon's scale after modifiers.
    pub raw: f32,
    /// `raw` squashed into (-1, 1).
    pub valence: f32,
    /// Evidential weight in (0, 1]; "used to" lowers it.
    pub weight: f32,
    pub modifiers: Vec<Modifier>,
    /// Byte range of the clause in the analysed text.
    #[serde(skip)]
    pub clause: Range<usize>,
}

/// The result of reading one text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextSentiment {
    /// Whole-text valence in (-1, 1); clauses either side of "but" are weighted 0.5 and 1.5.
    pub valence: f32,
    /// Target mentions located, with or without an opinion attached.
    pub mentions: usize,
    pub opinions: Vec<Opinion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Word,
    Boundary,
    Exclamation,
    Question,
    Comma,
    Colon,
    Emoji,
}

#[derive(Debug, Clone)]
struct Token {
    lower: String,
    span: Range<usize>,
    kind: TokenKind,
    caps: bool,
    upper_initial: bool,
}

impl Token {
    fn is_word(&self) -> bool {
        self.kind == TokenKind::Word
    }

    /// The word without a possessive, so "Rust's" matches the target "Rust".
    fn base(&self) -> &str {
        self.lower.strip_suffix("'s").unwrap_or(&self.lower)
    }
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_alphanumeric() {
            let mut end = start + c.len_utf8();
            let mut lower: String = c.to_lowercase().collect();
            while let Some(&(i, next)) = chars.peek() {
                let joins = matches!(next, '\'' | '’' | '-')
                    && text[i + next.len_utf8()..]
                        .chars()
                        .next()
                        .is_some_and(char::is_alphanumeric);
                if !(next.is_alphanumeric() || joins) {
                    break;
                }
                if next == '’' {
                    lower.push('\'');
                } else {
                    lower.extend(next.to_lowercase());
                }
                end = i + next.len_utf8();
                chars.next();
            }
            let word = &text[start..end];
            let mut letters = word.chars().filter(|ch| ch.is_alphabetic()).peekable();
            let upper_initial = letters.peek().is_some_and(|ch| ch.is_uppercase());
            let caps = word.chars().filter(|ch| ch.is_alphabetic()).count() >= 2
                && letters.all(char::is_uppercase);
            tokens.push(Token {
                lower,
                span: start..end,
                kind: TokenKind::Word,
                caps,
                upper_initial,
            });
            continue;
        }
        let kind = match c {
            '.' | ';' | '\n' => TokenKind::Boundary,
            '!' => TokenKind::Exclamation,
            '?' => TokenKind::Question,
            ',' => TokenKind::Comma,
            ':' => TokenKind::Colon,
            _ if emoji_score(c).is_some() => TokenKind::Emoji,
            _ => continue,
        };
        tokens.push(Token {
            lower: c.to_string(),
            span: start..start + c.len_utf8(),
            kind,
            caps: false,
            upper_initial: false,
        });
    }
    tokens
}

fn emoji_score(c: char) -> Option<f32> {
    EMOJI.iter().find(|(e, _)| *e == c).map(|(_, s)| *s)
}

fn in_sorted(table: &[&str], word: &str) -> bool {
    table.binary_search(&word).is_ok()
}

fn scored(table: &[(&str, f32)], word: &str) -> Option<f32> {
    table
        .binary_search_by(|(w, _)| (*w).cmp(word))
        .ok()
        .map(|i| table[i].1)
}

fn is_negation(word: &str) -> bool {
    word.ends_with("n't") || in_sorted(NEGATIONS, word)
}

fn lexicon_entry(word: &str) -> Option<(&'static str, f32, CueClass)> {
    LEXICON
        .binary_search_by(|(w, _, _)| (*w).cmp(word))
        .ok()
        .map(|i| LEXICON[i])
}

/// The lexicon entry for a word or a regular inflection of it ("loved", "hates", "preferring").
fn lemma(word: &str) -> Option<(&'static str, f32, CueClass)> {
    if let Some(entry) = lexicon_entry(word) {
        return Some(entry);
    }
    let undouble = |stem: &str| -> Option<String> {
        let bytes = stem.as_bytes();
        (bytes.len() >= 2 && bytes[bytes.len() - 1] == bytes[bytes.len() - 2])
            .then(|| stem[..stem.len() - 1].to_string())
    };
    let mut candidates: Vec<String> = Vec::with_capacity(4);
    for suffix in ["ing", "ed"] {
        if let Some(stem) = word.strip_suffix(suffix).filter(|s| s.len() >= 2) {
            candidates.push(stem.to_string());
            candidates.push(format!("{stem}e"));
            candidates.extend(undouble(stem));
        }
    }
    for suffix in ["es", "s", "d"] {
        if let Some(stem) = word.strip_suffix(suffix).filter(|s| s.len() >= 2) {
            candidates.push(stem.to_string());
        }
    }
    candidates.iter().find_map(|c| lexicon_entry(c))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ModKind {
    Booster(f32),
    Dampener(f32),
    Negation,
    Historical,
}

/// The modifier ending at token `j`, and how many tokens it spans.
fn modifier_at(tokens: &[Token], j: usize, floor: usize) -> Option<(ModKind, String, usize)> {
    let word = tokens[j].lower.as_str();
    let previous = (j > floor && tokens[j - 1].is_word()).then(|| tokens[j - 1].lower.as_str());
    match (previous, word) {
        (Some(p @ ("kind" | "sort")), "of") => {
            Some((ModKind::Dampener(HEDGE_PHRASE_FACTOR), format!("{p} of"), 2))
        }
        (Some("used"), "to") => Some((ModKind::Historical, "used to".to_string(), 2)),
        _ if is_negation(word) => Some((ModKind::Negation, word.to_string(), 1)),
        _ => scored(BOOSTERS, word)
            .map(|f| (ModKind::Booster(f), word.to_string(), 1))
            .or_else(|| {
                scored(DAMPENERS, word).map(|f| (ModKind::Dampener(f), word.to_string(), 1))
            }),
    }
}

fn is_modifier_word(tokens: &[Token], j: usize, floor: usize) -> bool {
    modifier_at(tokens, j, floor).is_some()
        || (tokens[j].lower == "kind" || tokens[j].lower == "sort")
            && tokens.get(j + 1).is_some_and(|t| t.lower == "of")
        || tokens[j].lower == "used" && tokens.get(j + 1).is_some_and(|t| t.lower == "to")
}

#[derive(Debug)]
struct Clause {
    tokens: Range<usize>,
    question: bool,
    exclamations: u8,
    before_contrast: bool,
    after_contrast: bool,
}

impl Clause {
    fn contrast_weight(&self) -> f32 {
        let mut weight = 1.0;
        if self.before_contrast {
            weight *= BEFORE_CONTRAST_WEIGHT;
        }
        if self.after_contrast {
            weight *= AFTER_CONTRAST_WEIGHT;
        }
        weight
    }

    fn bytes(&self, tokens: &[Token]) -> Range<usize> {
        match (
            tokens.get(self.tokens.start),
            self.tokens.end.checked_sub(1),
        ) {
            (Some(first), Some(last)) if last >= self.tokens.start => {
                first.span.start..tokens[last].span.end
            }
            _ => 0..0,
        }
    }
}

fn clauses(tokens: &[Token], from: usize) -> Vec<Clause> {
    let mut out = Vec::new();
    let mut start = from;
    let mut after_contrast = false;
    let mut i = from;
    while i < tokens.len() {
        let token = &tokens[i];
        let contrast = token.is_word() && in_sorted(CONTRASTS, &token.lower);
        let terminal = matches!(
            token.kind,
            TokenKind::Boundary | TokenKind::Exclamation | TokenKind::Question | TokenKind::Colon
        );
        if !(contrast || terminal) {
            i += 1;
            continue;
        }
        let mut clause = Clause {
            tokens: start..i,
            question: false,
            exclamations: 0,
            before_contrast: contrast,
            after_contrast,
        };
        let mut next = i + 1;
        if terminal {
            let mut j = i;
            while j < tokens.len()
                && matches!(tokens[j].kind, TokenKind::Exclamation | TokenKind::Question)
            {
                match tokens[j].kind {
                    TokenKind::Question => clause.question = true,
                    _ => clause.exclamations = clause.exclamations.saturating_add(1),
                }
                j += 1;
            }
            next = j.max(i + 1);
        }
        if !clause.tokens.is_empty() {
            out.push(clause);
        }
        after_contrast = contrast;
        start = next;
        i = next;
    }
    if start < tokens.len() {
        out.push(Clause {
            tokens: start..tokens.len(),
            question: false,
            exclamations: 0,
            before_contrast: false,
            after_contrast,
        });
    }
    out
}

/// A leading `Name:` (one to three capitalised words) names the speaker, as in
/// conversation transcripts ("Caroline: I love painting").
fn speaker_prefix(text: &str, tokens: &[Token]) -> (Option<String>, usize) {
    let colon = tokens
        .iter()
        .take(4)
        .position(|t| t.kind == TokenKind::Colon);
    match colon {
        Some(n) if n >= 1 && tokens[..n].iter().all(|t| t.is_word() && t.upper_initial) => (
            Some(text[tokens[0].span.start..tokens[n - 1].span.end].to_string()),
            n + 1,
        ),
        _ => (None, 0),
    }
}

/// Whole-text capitals carry no emphasis; one capitalised word among lower-case ones does.
fn is_shouting(tokens: &[Token]) -> bool {
    let (caps, words) = tokens
        .iter()
        .filter(|t| t.is_word() && t.lower.chars().any(char::is_alphabetic))
        .fold((0usize, 0usize), |(c, w), t| {
            (c + usize::from(t.caps), w + 1)
        });
    words > 0 && caps * 2 > words
}

#[derive(Debug)]
struct Mention {
    tokens: Range<usize>,
    target: usize,
}

fn find_mentions(
    tokens: &[Token],
    clause: &Clause,
    targets: &[(usize, Vec<String>)],
) -> Vec<Mention> {
    let mut mentions = Vec::new();
    let mut i = clause.tokens.start;
    'scan: while i < clause.tokens.end {
        for (target, words) in targets {
            let end = i + words.len();
            if end <= clause.tokens.end
                && tokens[i..end]
                    .iter()
                    .zip(words)
                    .all(|(t, w)| t.is_word() && t.base() == w)
            {
                mentions.push(Mention {
                    tokens: i..end,
                    target: *target,
                });
                i = end;
                continue 'scan;
            }
        }
        i += 1;
    }
    mentions
}

struct Subject {
    at: Option<usize>,
    saw_negation: bool,
    saw_auxiliary: bool,
}

/// The first word before `at` that is neither a modifier nor a filler, looking back at
/// most `HOLDER_REACH` words and never past the clause start.
fn subject_before(tokens: &[Token], clause: &Clause, at: usize, consumed: &[usize]) -> Subject {
    let mut subject = Subject {
        at: None,
        saw_negation: false,
        saw_auxiliary: false,
    };
    let mut j = at;
    let mut reach = 0;
    while j > clause.tokens.start && reach < HOLDER_REACH {
        j -= 1;
        let token = &tokens[j];
        if !token.is_word() {
            break;
        }
        if consumed.contains(&j) {
            continue;
        }
        reach += 1;
        let word = token.lower.as_str();
        if is_negation(word) {
            subject.saw_negation = true;
            continue;
        }
        if in_sorted(AUXILIARIES, word) {
            subject.saw_auxiliary = true;
        }
        if is_modifier_word(tokens, j, clause.tokens.start)
            || in_sorted(SUBJECT_SKIPS, word)
            || word.len() > 4 && word.ends_with("ly")
        {
            continue;
        }
        subject.at = Some(j);
        break;
    }
    subject
}

struct Cue {
    at: usize,
    word: String,
    class: CueClass,
    raw: f32,
    weight: f32,
    modifiers: Vec<Modifier>,
    holder: Holder,
    holder_mention: Option<usize>,
}

struct BaseCue {
    at: usize,
    word: String,
    score: f32,
    class: CueClass,
    consumed: Vec<usize>,
}

fn base_cue(tokens: &[Token], clause: &Clause, i: usize) -> Option<BaseCue> {
    let token = &tokens[i];
    let cue = |word: &str, score: f32, class: CueClass, consumed: Vec<usize>| {
        Some(BaseCue {
            at: i,
            word: word.to_string(),
            score,
            class,
            consumed,
        })
    };
    if token.kind == TokenKind::Emoji {
        let c = token.lower.chars().next()?;
        return cue(&token.lower, emoji_score(c)?, E, Vec::new());
    }
    if !token.is_word() {
        return None;
    }
    let floor = clause.tokens.start;
    let word_at = |j: usize| -> Option<&str> {
        (j >= floor && j < clause.tokens.end && tokens[j].is_word())
            .then(|| tokens[j].lower.as_str())
    };
    let next = word_at(i + 1);
    match token.lower.as_str() {
        "stand" => {
            let one_back = i.checked_sub(1).and_then(word_at);
            let two_back = i.checked_sub(2).and_then(word_at);
            if matches!(one_back, Some("can't" | "cannot" | "cant")) {
                return cue("can't stand", -3.0, S, vec![i - 1]);
            }
            if one_back == Some("not") && two_back == Some("can") {
                return cue("can't stand", -3.0, S, vec![i - 2, i - 1]);
            }
            return None;
        }
        "fed" if next == Some("up") => return cue("fed up", -2.4, S, Vec::new()),
        "sick" if next == Some("of") => return cue("sick of", -2.2, S, Vec::new()),
        "tired" if next == Some("of") => return cue("tired of", -1.8, S, Vec::new()),
        "into" => {
            let licensed = subject_before(tokens, clause, i, &[])
                .at
                .is_some_and(|j| in_sorted(INTO_LICENSORS, &tokens[j].lower));
            return licensed
                .then(|| cue("into", INTO_SCORE, S, Vec::new()))
                .flatten();
        }
        _ => {}
    }
    let (word, score, class) = lemma(&token.lower)?;
    if token.lower == "like" && !like_is_a_verb(tokens, clause, i) {
        return None;
    }
    cue(word, score, class, Vec::new())
}

/// "like" is mostly a preposition ("looks like", "something like"). Count it only after a
/// subject, an auxiliary, or a bare negation ("Don't like it").
fn like_is_a_verb(tokens: &[Token], clause: &Clause, i: usize) -> bool {
    let subject = subject_before(tokens, clause, i, &[]);
    if subject.saw_auxiliary {
        return true;
    }
    match subject.at {
        Some(j) => {
            let word = tokens[j].lower.as_str();
            // "it's like", "I'm like", "they're like" compare or quote; they never state a liking.
            let copula = ["'s", "'re", "'m"].iter().any(|c| word.ends_with(c));
            !copula
                && (in_sorted(FIRST_PERSON, word)
                    || in_sorted(THIRD_PERSON, word)
                    || word == "you"
                    || word == "who"
                    || tokens[j].upper_initial)
        }
        None => subject.saw_negation,
    }
}

fn find_cues(
    text: &str,
    tokens: &[Token],
    clause: &Clause,
    mentions: &[Mention],
    targets: &[&str],
    speaker: Option<&str>,
    shouting: bool,
) -> Vec<Cue> {
    let in_mention = |i: usize| mentions.iter().any(|m| m.tokens.contains(&i));
    let bases: Vec<BaseCue> = clause
        .tokens
        .clone()
        .filter(|&i| !in_mention(i))
        .filter_map(|i| base_cue(tokens, clause, i))
        .collect();
    let cue_positions: Vec<usize> = bases.iter().map(|b| b.at).collect();
    let speaker_holder = || speaker.map_or(Holder::Speaker, |s| Holder::Named(s.to_string()));

    bases
        .into_iter()
        .map(|base| {
            let (raw, weight, mut modifiers) =
                apply_modifiers(tokens, clause, &base, &cue_positions);
            let mut raw = raw;
            if clause.exclamations > 0 {
                let count = clause.exclamations.min(MAX_EXCLAMATIONS);
                raw *= 1.0 + EXCLAMATION_STEP * f32::from(count);
                modifiers.push(Modifier::Exclamation { count });
            }
            if tokens[base.at].caps && !shouting {
                raw *= CAPS_EMPHASIS;
                modifiers.push(Modifier::Emphasis);
            }

            let takes_subject = base.class == S || base.word == "prefer";
            let (holder, holder_mention) = if takes_subject {
                let subject = subject_before(tokens, clause, base.at, &base.consumed);
                match subject.at {
                    None => (speaker_holder(), None),
                    Some(j) => match mentions.iter().position(|m| m.tokens.end == j + 1) {
                        Some(m) => (
                            Holder::Named(targets[mentions[m].target].to_string()),
                            Some(m),
                        ),
                        None => (
                            holder_from_word(text, tokens, clause, j, &speaker_holder),
                            None,
                        ),
                    },
                }
            } else {
                (speaker_holder(), None)
            };

            Cue {
                at: base.at,
                word: base.word,
                class: base.class,
                raw,
                weight,
                modifiers,
                holder,
                holder_mention,
            }
        })
        .collect()
}

fn holder_from_word(
    text: &str,
    tokens: &[Token],
    clause: &Clause,
    j: usize,
    speaker_holder: &dyn Fn() -> Holder,
) -> Holder {
    let word = tokens[j].lower.as_str();
    if in_sorted(FIRST_PERSON, word) {
        return speaker_holder();
    }
    if in_sorted(THIRD_PERSON, word) || !tokens[j].upper_initial {
        return Holder::Other;
    }
    let mut first = j;
    while first > clause.tokens.start
        && j - first < 2
        && tokens[first - 1].is_word()
        && tokens[first - 1].upper_initial
    {
        first -= 1;
    }
    Holder::Named(text[tokens[first].span.start..tokens[j].span.end].to_string())
}

fn apply_modifiers(
    tokens: &[Token],
    clause: &Clause,
    base: &BaseCue,
    cue_positions: &[usize],
) -> (f32, f32, Vec<Modifier>) {
    let mut found: Vec<(ModKind, String)> = Vec::new();
    let mut j = base.at;
    let mut steps = 0;
    while j > clause.tokens.start && steps < MODIFIER_WINDOW {
        j -= 1;
        if !tokens[j].is_word() || cue_positions.contains(&j) {
            break;
        }
        if base.consumed.contains(&j) {
            continue;
        }
        steps += 1;
        if let Some((kind, word, width)) = modifier_at(tokens, j, clause.tokens.start) {
            found.push((kind, word));
            j -= width - 1;
        }
    }

    let mut raw = base.score;
    let mut weight = 1.0;
    let mut modifiers = Vec::with_capacity(found.len());
    for (position, (kind, word)) in found.iter().enumerate() {
        match *kind {
            ModKind::Negation => {
                raw *= NEGATION_SCALAR;
                modifiers.push(Modifier::Negation { word: word.clone() });
            }
            ModKind::Booster(factor) => {
                let negated_further_out = found[position + 1..]
                    .iter()
                    .any(|(k, _)| *k == ModKind::Negation);
                let factor = if negated_further_out {
                    1.0 / factor
                } else {
                    factor
                };
                raw *= factor;
                modifiers.push(Modifier::Booster {
                    word: word.clone(),
                    factor,
                });
            }
            ModKind::Dampener(factor) => {
                raw *= factor;
                modifiers.push(Modifier::Dampener {
                    word: word.clone(),
                    factor,
                });
            }
            ModKind::Historical => {
                weight *= HISTORICAL_WEIGHT;
                modifiers.push(Modifier::Historical);
            }
        }
    }
    (raw, weight, modifiers)
}

/// Which side of a comparison a mention is on: `Some(false)` wins, `Some(true)` loses,
/// `None` is outside the comparison.
fn comparative_side(tokens: &[Token], cue: &Cue, mention: &Mention) -> Option<bool> {
    let connector_between = |connectors: &[&str]| {
        (cue.at + 1..mention.tokens.start)
            .any(|j| tokens[j].is_word() && connectors.contains(&tokens[j].lower.as_str()))
    };
    if cue.word == "prefer" {
        return (mention.tokens.start > cue.at).then(|| connector_between(COMPARISON_CONNECTORS));
    }
    if mention.tokens.end <= cue.at {
        Some(false)
    } else {
        connector_between(&["than"]).then_some(true)
    }
}

fn attribute(
    tokens: &[Token],
    clause: &Clause,
    mentions: &[Mention],
    cues: &[Cue],
    targets: &[&str],
    text_len: usize,
) -> Vec<Opinion> {
    let clause_bytes = clause.bytes(tokens);
    let clause_bytes = clause_bytes.start.min(text_len)..clause_bytes.end.min(text_len);
    mentions
        .iter()
        .enumerate()
        .filter_map(|(m, mention)| {
            let mut best: Option<(usize, &Cue, bool)> = None;
            for cue in cues {
                if cue.holder_mention == Some(m) {
                    continue;
                }
                let after = mention.tokens.start > cue.at;
                let loser = match cue.class {
                    CueClass::Stance if !after => continue,
                    CueClass::Stance | CueClass::Evaluative => false,
                    CueClass::Comparative => match comparative_side(tokens, cue, mention) {
                        Some(loser) => loser,
                        None => continue,
                    },
                };
                let distance = if after {
                    mention.tokens.start - cue.at
                } else {
                    cue.at + 1 - mention.tokens.end.min(cue.at + 1)
                };
                if best.is_none_or(|(d, _, _)| distance < d) {
                    best = Some((distance, cue, loser));
                }
            }
            let (_, cue, loser) = best?;
            let mut modifiers = cue.modifiers.clone();
            let mut raw = cue.raw;
            if loser {
                raw *= -COMPARATIVE_LOSER_SHARE;
                modifiers.push(Modifier::ComparativeLoser);
            }
            Some(Opinion {
                target: targets[mention.target].to_string(),
                holder: cue.holder.clone(),
                cue: cue.word.clone(),
                raw,
                valence: squash(raw),
                weight: cue.weight,
                modifiers,
                clause: clause_bytes.clone(),
            })
        })
        .collect()
}

fn squash(raw: f32) -> f32 {
    raw / (raw * raw + SQUASH_ALPHA).sqrt()
}

/// Read `text` for stance toward each of `targets` (matched case-insensitively on whole
/// words, longest target first) and for its overall valence. Questions assert nothing and
/// contribute neither opinions nor valence.
pub fn analyze(text: &str, targets: &[&str]) -> TextSentiment {
    let tokens = tokenize(text);
    let (speaker, body) = speaker_prefix(text, &tokens);
    let shouting = is_shouting(&tokens[body..]);

    let mut target_words: Vec<(usize, Vec<String>)> = targets
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let words = tokenize(t)
                .iter()
                .filter(|tok| tok.is_word())
                .map(|tok| tok.base().to_string())
                .collect();
            (i, words)
        })
        .filter(|(_, words): &(usize, Vec<String>)| !words.is_empty())
        .collect();
    target_words.sort_by_key(|(_, words)| std::cmp::Reverse(words.len()));

    let mut total = 0.0;
    let mut mentions_found = 0;
    let mut opinions = Vec::new();
    for clause in clauses(&tokens, body) {
        let mentions = find_mentions(&tokens, &clause, &target_words);
        mentions_found += mentions.len();
        if clause.question {
            continue;
        }
        let cues = find_cues(
            text,
            &tokens,
            &clause,
            &mentions,
            targets,
            speaker.as_deref(),
            shouting,
        );
        total += cues.iter().map(|c| c.raw).sum::<f32>() * clause.contrast_weight();
        opinions.extend(attribute(
            &tokens,
            &clause,
            &mentions,
            &cues,
            targets,
            text.len(),
        ));
    }

    TextSentiment {
        valence: squash(total),
        mentions: mentions_found,
        opinions,
    }
}

/// One opinion placed in time, ready to aggregate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub valence: f32,
    pub weight: f32,
    pub at: DateTime<Utc>,
}

/// A stance placed on a `scale`-point Likert scale.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LikertProjection {
    pub scale: u8,
    /// Continuous position in `1..=scale`.
    pub point: f32,
    pub rounded: u8,
    /// Named for 3-, 5- and 7-point scales.
    pub label: Option<&'static str>,
    /// Share of evidence at each point, each opinion split between its two nearest points.
    pub distribution: Vec<f32>,
}

/// Opinions toward one target, aggregated.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StanceSummary {
    pub opinions: usize,
    /// Sum of opinion weights after recency decay.
    pub effective_weight: f32,
    /// Recency-weighted mean valence in (-1, 1).
    pub valence: f32,
    /// Mean valence with no recency decay; a gap from `valence` means the stance moved.
    pub all_time_valence: f32,
    /// Recency-weighted standard deviation of valence.
    pub spread: f32,
    /// Grows with evidence, shrinks with disagreement; in [0, 1).
    pub confidence: f32,
    /// When the stance last changed sign: the newest opinion whose polarity differs from
    /// the one before it in time. Near-neutral opinions are passed over.
    pub last_reversal: Option<DateTime<Utc>>,
    pub likert: LikertProjection,
}

fn last_reversal(observations: &[Observation]) -> Option<DateTime<Utc>> {
    let mut polar: Vec<&Observation> = observations
        .iter()
        .filter(|o| o.valence.abs() >= REVERSAL_FLOOR)
        .collect();
    polar.sort_by_key(|o| o.at);
    polar
        .windows(2)
        .rev()
        .find(|pair| pair[0].valence.signum() != pair[1].valence.signum())
        .map(|pair| pair[1].at)
}

fn likert_label(scale: u8, rounded: u8) -> Option<&'static str> {
    const THREE: [&str; 3] = ["negative", "neutral", "positive"];
    const FIVE: [&str; 5] = [
        "strongly negative",
        "negative",
        "neutral",
        "positive",
        "strongly positive",
    ];
    const SEVEN: [&str; 7] = [
        "strongly negative",
        "negative",
        "somewhat negative",
        "neutral",
        "somewhat positive",
        "positive",
        "strongly positive",
    ];
    let index = usize::from(rounded.checked_sub(1)?);
    match scale {
        3 => THREE.get(index).copied(),
        5 => FIVE.get(index).copied(),
        7 => SEVEN.get(index).copied(),
        _ => None,
    }
}

fn scale_position(valence: f32, scale: u8) -> f32 {
    (valence.clamp(-1.0, 1.0) + 1.0) / 2.0 * f32::from(scale - 1)
}

fn projection(mean: f32, scale: u8, distribution: Vec<f32>) -> LikertProjection {
    let point = 1.0 + scale_position(mean, scale);
    let rounded = (point.round() as u8).clamp(1, scale);
    LikertProjection {
        scale,
        point,
        rounded,
        label: likert_label(scale, rounded),
        distribution,
    }
}

fn spread_into(distribution: &mut [f32], valence: f32, weight: f32, scale: u8) {
    let position = scale_position(valence, scale);
    let low = (position.floor() as usize).min(distribution.len() - 1);
    let share = position - low as f32;
    distribution[low] += weight * (1.0 - share);
    if share > 0.0 && low + 1 < distribution.len() {
        distribution[low + 1] += weight * share;
    }
}

/// Place a single valence on a `scale`-point Likert scale. `scale` is clamped into
/// [`LIKERT_SCALES`].
pub fn project(valence: f32, scale: u8) -> LikertProjection {
    let scale = scale.clamp(*LIKERT_SCALES.start(), *LIKERT_SCALES.end());
    let mut distribution = vec![0.0; usize::from(scale)];
    spread_into(&mut distribution, valence, 1.0, scale);
    projection(valence, scale, distribution)
}

/// Aggregate opinions as of `now`. Each weighs `weight × 0.5^(age / half_life_days)`; a
/// non-positive half-life turns decay off. `None` when there is nothing to aggregate.
pub fn summarize(
    observations: &[Observation],
    now: DateTime<Utc>,
    half_life_days: f32,
    scale: u8,
) -> Option<StanceSummary> {
    if observations.is_empty() {
        return None;
    }
    let scale = scale.clamp(*LIKERT_SCALES.start(), *LIKERT_SCALES.end());
    let decay = |at: DateTime<Utc>| -> f64 {
        if half_life_days <= 0.0 {
            return 1.0;
        }
        let age_days = ((now - at).num_seconds().max(0) as f64) / 86_400.0;
        0.5f64.powf(age_days / f64::from(half_life_days))
    };
    let mut weights: Vec<f64> = observations
        .iter()
        .map(|o| f64::from(o.weight) * decay(o.at))
        .collect();
    // Every opinion older than the half-life can underflow to zero together; fall back to
    // undecayed weights rather than report no stance.
    if weights.iter().sum::<f64>() <= f64::MIN_POSITIVE {
        weights = observations.iter().map(|o| f64::from(o.weight)).collect();
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }

    let mean = observations
        .iter()
        .zip(&weights)
        .map(|(o, w)| f64::from(o.valence) * w)
        .sum::<f64>()
        / total;
    let variance = observations
        .iter()
        .zip(&weights)
        .map(|(o, w)| (f64::from(o.valence) - mean).powi(2) * w)
        .sum::<f64>()
        / total;
    let plain_total: f64 = observations.iter().map(|o| f64::from(o.weight)).sum();
    let all_time = observations
        .iter()
        .map(|o| f64::from(o.valence) * f64::from(o.weight))
        .sum::<f64>()
        / plain_total.max(f64::MIN_POSITIVE);

    let mut distribution = vec![0.0f32; usize::from(scale)];
    for (o, w) in observations.iter().zip(&weights) {
        spread_into(&mut distribution, o.valence, (*w / total) as f32, scale);
    }

    let effective_weight = total as f32;
    let spread = variance.sqrt() as f32;
    let confidence = effective_weight / (effective_weight + CONFIDENCE_PRIOR_WEIGHT)
        * (1.0 - spread.clamp(0.0, 1.0));

    Some(StanceSummary {
        opinions: observations.len(),
        effective_weight,
        valence: mean as f32,
        all_time_valence: all_time as f32,
        spread,
        confidence,
        last_reversal: last_reversal(observations),
        likert: projection(mean as f32, scale, distribution),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn stance(text: &str, target: &str) -> Option<Opinion> {
        analyze(text, &[target])
            .opinions
            .into_iter()
            .find(|o| o.target == target)
    }

    fn valence(text: &str, target: &str) -> f32 {
        stance(text, target)
            .unwrap_or_else(|| panic!("no opinion on {target:?} in {text:?}"))
            .valence
    }

    #[test]
    fn lexicon_tables_are_sorted_and_unique() {
        assert!(LEXICON.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(BOOSTERS.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(DAMPENERS.windows(2).all(|w| w[0].0 < w[1].0));
        for table in [
            NEGATIONS,
            CONTRASTS,
            FIRST_PERSON,
            THIRD_PERSON,
            SUBJECT_SKIPS,
            AUXILIARIES,
            INTO_LICENSORS,
        ] {
            assert!(table.windows(2).all(|w| w[0] < w[1]), "{table:?}");
        }
    }

    #[test]
    fn modifiers_order_intensity_on_one_scale() {
        let ladder = [
            "I absolutely love Rust",
            "I love Rust",
            "I like Rust",
            "I kind of like Rust",
            "I don't really like Rust",
            "I dislike Rust",
            "I really hate Rust",
        ];
        let scores: Vec<f32> = ladder.iter().map(|t| valence(t, "Rust")).collect();
        assert!(
            scores.windows(2).all(|w| w[0] > w[1]),
            "not strictly decreasing: {scores:?}"
        );
        assert!(scores[3] > 0.0 && scores[4] < 0.0);
    }

    #[test]
    fn a_booster_inside_a_negation_weakens_it() {
        let not_very = valence("Rust is not very good", "Rust");
        let really_not = valence("Rust is really not good", "Rust");
        assert!(not_very < 0.0 && really_not < 0.0);
        assert!(not_very > really_not, "{not_very} vs {really_not}");
    }

    #[test]
    fn negating_a_negative_word_leans_positive() {
        assert!(valence("Honestly the new build is not bad", "build") > 0.0);
    }

    #[test]
    fn contrast_splits_targets_into_their_own_clauses() {
        let text = "I love Rust but I hate Go";
        let result = analyze(text, &["Rust", "Go"]);
        let of = |t: &str| result.opinions.iter().find(|o| o.target == t).unwrap();
        assert!(of("Rust").valence > 0.5);
        assert!(of("Go").valence < -0.5);
        assert_eq!(&text[of("Go").clause.clone()], "I hate Go");
    }

    #[test]
    fn an_attitude_verb_points_from_holder_to_target() {
        let result = analyze("Alice hates Rust", &["Alice", "Rust"]);
        assert_eq!(result.mentions, 2);
        assert_eq!(result.opinions.len(), 1);
        let opinion = &result.opinions[0];
        assert_eq!(opinion.target, "Rust");
        assert_eq!(opinion.holder, Holder::Named("Alice".into()));
        assert!(opinion.valence < 0.0);

        let unnamed = stance("Bob really loves Go", "Go").unwrap();
        assert_eq!(unnamed.holder, Holder::Named("Bob".into()));
        assert_eq!(
            stance("I've always loved Go", "Go").unwrap().holder,
            Holder::Speaker
        );
    }

    #[test]
    fn a_speaker_prefix_names_the_holder() {
        let opinion = stance("Caroline: I love painting landscapes", "painting").unwrap();
        assert_eq!(opinion.holder, Holder::Named("Caroline".into()));
        assert!(opinion.valence > 0.5);
    }

    #[test]
    fn comparisons_favour_one_side_and_disfavour_the_other() {
        let result = analyze("Rust is better than Go", &["Rust", "Go"]);
        let of = |t: &str| {
            result
                .opinions
                .iter()
                .find(|o| o.target == t)
                .unwrap()
                .valence
        };
        assert!(of("Rust") > 0.0 && of("Go") < 0.0);
        assert!(of("Rust").abs() > of("Go").abs());

        let result = analyze("I prefer tea over coffee", &["tea", "coffee"]);
        let of = |t: &str| {
            result
                .opinions
                .iter()
                .find(|o| o.target == t)
                .unwrap()
                .valence
        };
        assert!(of("tea") > 0.0 && of("coffee") < 0.0);
    }

    #[test]
    fn questions_assert_nothing() {
        let result = analyze("Do you like Rust?", &["Rust"]);
        assert_eq!(result.mentions, 1);
        assert!(result.opinions.is_empty());
        assert_eq!(result.valence, 0.0);
    }

    #[test]
    fn like_counts_only_as_a_verb() {
        assert!(stance("It looks like Rust to me", "Rust").is_none());
        assert!(stance("It's kind of like Rust", "Rust").is_none());
        assert!(stance("She's like Rust in that way", "Rust").is_none());
        assert!(valence("We like Rust", "Rust") > 0.0);
        assert!(valence("Don't like Rust much", "Rust") < 0.0);
        assert!(valence("I would like to visit Japan", "Japan") > 0.0);
    }

    #[test]
    fn phrases_carry_their_own_polarity() {
        let cant_stand = stance("I can't stand Java", "Java").unwrap();
        assert_eq!(cant_stand.cue, "can't stand");
        assert!(cant_stand.valence < -0.6);
        assert!(cant_stand.modifiers.is_empty());
        assert!(valence("I'm really into chess", "chess") > 0.0);
        assert!(stance("We went into town", "town").is_none());
        assert!(valence("I'm sick of meetings", "meetings") < 0.0);
    }

    #[test]
    fn used_to_keeps_the_sign_and_lowers_the_weight() {
        let opinion = stance("I used to love Perl", "Perl").unwrap();
        assert!(opinion.valence > 0.0);
        assert!((opinion.weight - HISTORICAL_WEIGHT).abs() < 1e-6);
        assert!(opinion.modifiers.contains(&Modifier::Historical));
    }

    #[test]
    fn exclamation_capitals_and_emoji_count() {
        assert!(valence("Rust is great!!", "Rust") > valence("Rust is great", "Rust"));
        assert!(valence("Rust is GREAT", "Rust") > valence("Rust is great", "Rust"));
        assert!(valence("RUST IS GREAT", "Rust") == valence("Rust is great", "Rust"));
        assert!(valence("Rust 😍", "Rust") > 0.5);
    }

    #[test]
    fn targets_match_whole_words_possessives_and_curly_apostrophes() {
        assert!(valence("Rust’s compiler is excellent", "Rust") > 0.5);
        assert!(stance("Trusty tools are great", "Rust").is_none());
        let result = analyze("New York is lovely", &["York", "New York"]);
        assert_eq!(result.mentions, 1);
        assert_eq!(result.opinions[0].target, "New York");
        assert!(valence("I don’t like Go", "Go") < 0.0);
    }

    #[test]
    fn inflections_reach_the_lexicon() {
        assert!(valence("She loved Paris", "Paris") > 0.0);
        assert!(valence("He hated Paris", "Paris") < 0.0);
        assert!(valence("Paris sucks", "Paris") < 0.0);
        assert!(valence("They preferred Paris", "Paris") > 0.0);
    }

    #[test]
    fn neutral_mentions_are_counted_but_carry_no_opinion() {
        let result = analyze("I wrote some Rust today.", &["Rust"]);
        assert_eq!(result.mentions, 1);
        assert!(result.opinions.is_empty());
    }

    #[test]
    fn likert_projection_spans_the_scale() {
        for scale in [3u8, 5, 7, 11] {
            let low = project(-1.0, scale);
            let mid = project(0.0, scale);
            let high = project(1.0, scale);
            assert_eq!(low.rounded, 1);
            assert_eq!(high.rounded, scale);
            assert!((mid.point - (1.0 + f32::from(scale - 1) / 2.0)).abs() < 1e-6);
            for p in [&low, &mid, &high] {
                assert!((p.distribution.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
        }
        assert_eq!(project(0.0, 5).label, Some("neutral"));
        assert_eq!(project(0.9, 7).label, Some("strongly positive"));
        assert_eq!(project(0.0, 6).label, None);
        assert_eq!(project(0.0, 99).scale, *LIKERT_SCALES.end());
    }

    #[test]
    fn recent_opinions_outweigh_old_ones_under_decay() {
        let now = Utc::now();
        let observations = [
            Observation {
                valence: 0.8,
                weight: 1.0,
                at: now - Duration::days(365),
            },
            Observation {
                valence: -0.7,
                weight: 1.0,
                at: now - Duration::days(2),
            },
        ];
        let decayed = summarize(&observations, now, 90.0, 5).unwrap();
        assert!(decayed.valence < -0.5, "{}", decayed.valence);
        assert!((decayed.all_time_valence - 0.05).abs() < 1e-6);
        assert!(decayed.likert.rounded <= 2);

        let flat = summarize(&observations, now, 0.0, 5).unwrap();
        assert!((flat.valence - flat.all_time_valence).abs() < 1e-6);
        assert_eq!(flat.last_reversal, Some(now - Duration::days(2)));
    }

    #[test]
    fn a_reversal_is_dated_and_near_neutral_opinions_do_not_count() {
        let now = Utc::now();
        let at = |valence: f32, days: i64| Observation {
            valence,
            weight: 1.0,
            at: now - Duration::days(days),
        };
        let steady = [at(0.5, 30), at(-0.05, 20), at(0.6, 10)];
        assert_eq!(summarize(&steady, now, 0.0, 5).unwrap().last_reversal, None);
        let flipped = [at(-0.4, 30), at(0.5, 20), at(0.7, 10), at(-0.6, 5)];
        assert_eq!(
            summarize(&flipped, now, 0.0, 5).unwrap().last_reversal,
            Some(now - Duration::days(5))
        );
    }

    #[test]
    fn confidence_grows_with_agreement_and_falls_with_disagreement() {
        let now = Utc::now();
        let at = |v: f32| Observation {
            valence: v,
            weight: 1.0,
            at: now,
        };
        let one = summarize(&[at(0.6)], now, 0.0, 5).unwrap().confidence;
        let four = summarize(&[at(0.6); 4], now, 0.0, 5).unwrap().confidence;
        let split = summarize(&[at(0.8), at(-0.8), at(0.8), at(-0.8)], now, 0.0, 5)
            .unwrap()
            .confidence;
        assert!((one - 0.5).abs() < 1e-6);
        assert!(four > one);
        assert!(split < one);
        assert!(summarize(&[], now, 90.0, 5).is_none());
    }

    #[test]
    fn ancient_evidence_still_aggregates() {
        let now = Utc::now();
        let observations = [Observation {
            valence: 0.5,
            weight: 1.0,
            at: now - Duration::days(365 * 50),
        }];
        let summary = summarize(&observations, now, 1.0, 5).unwrap();
        assert!((summary.valence - 0.5).abs() < 1e-6);
    }
}
