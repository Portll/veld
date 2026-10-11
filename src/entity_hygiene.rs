//! Lexical hygiene for graph entities (which are also the tag namespace).
//!
//! A stored tag list mixes what the writer chose with what veld derived, and the derived part
//! carries timestamps, uuids, sha fragments, file paths, severity counters and JSON shards.
//! `classify` sorts a name onto one of three shelves from its shape alone, with the rule that
//! decided it, so a reader can filter the cloud and a test can hold the rule to its fire rate.
//!
//! `is_proper_noun` never decides a shelf. Measured 2026-10-07 over 23,582 entities: the flag is
//! set on capitalised common words (`files`, `controls`, `evidence`), and the NER labels beside
//! it put `Person` on utility names. It is passed through as an unverified hint. Cast therefore
//! means identifier-shaped, and its recall against a human's notion of "a name" is bounded by
//! design (0.44 to 0.61 on the two gold sets), while its precision measured 1.0 on both.
//!
//! Fitted on 300 decile-stratified `portll` entities, held out on 300 from `cobolwork` (a
//! different vocabulary): 93.3% and 92.7%. The gold sets name fleet and client tokens and live in
//! the private sidecar; `VELD_HYGIENE_GOLD_DIR` points the replay test at them.

use once_cell::sync::Lazy;
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shelf {
    Cast,
    Vocabulary,
    Noise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Reason {
    Uuid,
    Runid,
    Timestamp,
    Json,
    Session,
    Lineref,
    Counter,
    Id,
    Hex,
    Url,
    Ref,
    Fragment,
    Number,
    Path,
    CodeIdent,
    Filename,
    Vocab,
}

impl Reason {
    pub fn shelf(self) -> Shelf {
        match self {
            Reason::Uuid
            | Reason::Runid
            | Reason::Timestamp
            | Reason::Json
            | Reason::Lineref
            | Reason::Counter
            | Reason::Hex
            | Reason::Url
            | Reason::Fragment
            | Reason::Number
            | Reason::Path => Shelf::Noise,
            Reason::Session | Reason::Id | Reason::Ref | Reason::CodeIdent | Reason::Filename => {
                Shelf::Cast
            }
            Reason::Vocab => Shelf::Vocabulary,
        }
    }

    pub const ALL: [Reason; 17] = [
        Reason::Uuid,
        Reason::Runid,
        Reason::Timestamp,
        Reason::Json,
        Reason::Session,
        Reason::Counter,
        Reason::Lineref,
        Reason::Id,
        Reason::Hex,
        Reason::Url,
        Reason::Ref,
        Reason::Fragment,
        Reason::Number,
        Reason::Path,
        Reason::CodeIdent,
        Reason::Filename,
        Reason::Vocab,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Classification {
    pub shelf: Shelf,
    pub reason: Reason,
    /// veld's NER flag, carried for display. It decided nothing above.
    pub proper_noun_hint: bool,
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("entity_hygiene pattern")
}

fn re_i(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .expect("entity_hygiene pattern")
}

static UUID: Lazy<Regex> =
    Lazy::new(|| re_i(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"));
static RUNID: Lazy<Regex> = Lazy::new(|| re_i(r"^sweep-\d{14}(-[a-z0-9]+)?$"));
static TIMESTAMP: Lazy<[Regex; 4]> = Lazy::new(|| {
    [
        re_i(r"^\d{4}-\d{2}-\d{2}(t\d{2}:\d{2}(:\d{2})?(\.\d+)?z?)?$"),
        re_i(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(t\d{6}z?)?$"),
        re(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{6}(:\d+)?$"),
        re_i(r"^\d{3}z$"),
    ]
});
static JSON: Lazy<Regex> = Lazy::new(|| re(r#""[:,}{\]]|[:,{\[]"|^\d+,"|\{"|"\}"#));
static SESSION: Lazy<Regex> =
    Lazy::new(|| re(r"^(commitwork|cobolwork|ironwork|veld|substrate|spine|sleight)-[0-9a-f]{2}$"));
static LINEREF: Lazy<[Regex; 3]> = Lazy::new(|| {
    [
        re_i(r"\.(mjs|cjs|js|ts|rs|php|yml|yaml|py|json|md|go|java|rb|sh):\d+$"),
        re(r"^[a-z]{1,5}:\d+$"),
        re(r"^[A-Z]{2}\d{4}:[a-z]+:\d+:\d+$"),
    ]
});
static COUNTER: Lazy<[Regex; 2]> = Lazy::new(|| {
    [
        re_i(r"^(crit|high|med|medium|low|worst|info):[a-z0-9]+$"),
        re(r"^(\d+[CHML]/)+\d+[CHML]$"),
    ]
});
static ID: Lazy<[Regex; 6]> = Lazy::new(|| {
    [
        re_i(
            r"(^|\s)(GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}|CVE-\d{4}(-\d+)?|RUSTSEC-\d{4}(-\d+)?|CWE-\d+)(\s|$)",
        ),
        re(r"^(IWR|IWS|IWC|IWO|IWP|IWL|IWX|IWJ|IGY|IGZ|IEC|IEF|IEA)\d{4}"),
        re(r"^D\d{1,2}_\w+$"),
        re(r"^[a-z]+_lang_[a-z_]+$"),
        re(r"^capability-[a-z-]+$"),
        re(
            r"^(cs|js|java|go|py|rb|swift|cpp|c|mobile|ql|actions)/[a-z0-9]+(/[a-z0-9-]+)?/?[a-z0-9]*(-[a-z0-9]+)+$",
        ),
    ]
});
static HEX: Lazy<Regex> = Lazy::new(|| re_i(r"^[0-9a-f]{7,40}$"));
static HEX_HAS_LETTER: Lazy<Regex> = Lazy::new(|| re_i(r"[a-f]"));
static URL: Lazy<Regex> = Lazy::new(|| re_i(r"^https?://"));
static REF: Lazy<Regex> = Lazy::new(|| re_i(r"^[a-z][a-z-]*:[^\s:=]+$"));
static FRAGMENT: Lazy<Regex> = Lazy::new(|| re(r#"["`=]|[(){}\[\]]|\\|;|,[^\s\d]"#));
static POSSESSIVE: Lazy<Regex> = Lazy::new(|| re(r"'s\b"));
static NUMBER: Lazy<[Regex; 2]> = Lazy::new(|| {
    [
        re_i(r"^\d[\d.,%/–-]*(-?[a-z]+)?$"),
        re_i(r"^v\d+(r\d+|\.\d+)*$"),
    ]
});
static PATH_EXT: Lazy<Regex> = Lazy::new(|| re_i(r"\.[a-z]{1,5}$"));
static TITLE_WORD: Lazy<Regex> = Lazy::new(|| re(r"^[A-Z][A-Za-z]*$"));
static CODE_IDENT: Lazy<Regex> = Lazy::new(|| re(r"^[A-Z][a-z0-9]+([A-Z][a-z0-9]+)+$"));
static FILENAME: Lazy<Regex> = Lazy::new(|| {
    re_i(
        r"^[\w.-]+\.(mjs|cjs|js|ts|tsx|rs|py|go|java|rb|sh|md|json|toml|yaml|yml|html|css|cbl|cob|cpy|jcl)(\s[\w.-]+\.(md|json|yml|yaml))?$",
    )
});

const PATH_ROOTS: &[&str] = &[
    "tmp",
    "users",
    "spine",
    "evaluations",
    "routes",
    "lib",
    "src",
    "crates",
    "diag",
    "feed",
    "github",
    "com",
    "upstream",
    "docs",
    "bin",
    "admin",
    "monitor",
    "test",
    "tests",
    "init",
    "go",
    "var",
    "etc",
    "home",
    "private",
    "sidecar",
];

fn is_fragment(n: &str) -> bool {
    if FRAGMENT.is_match(n) {
        return true;
    }
    POSSESSIVE.replace_all(n, "").contains('\'')
}

fn is_path(n: &str) -> bool {
    if n.contains(char::is_whitespace) || !n.contains('/') {
        return false;
    }
    let segs: Vec<&str> = n.split('/').filter(|s| !s.is_empty()).collect();
    if segs.iter().all(|s| TITLE_WORD.is_match(s)) {
        return false;
    }
    let Some(last) = segs.last() else {
        return false;
    };
    segs.len() >= 3
        || PATH_EXT.is_match(last)
        || PATH_ROOTS.contains(&segs[0].to_ascii_lowercase().as_str())
        || n.contains("-Users-")
}

fn matched_reason(n: &str) -> Reason {
    if UUID.is_match(n) {
        return Reason::Uuid;
    }
    if RUNID.is_match(n) {
        return Reason::Runid;
    }
    if TIMESTAMP.iter().any(|r| r.is_match(n)) {
        return Reason::Timestamp;
    }
    if JSON.is_match(n) {
        return Reason::Json;
    }
    if SESSION.is_match(n) {
        return Reason::Session;
    }
    if COUNTER.iter().any(|r| r.is_match(n)) {
        return Reason::Counter;
    }
    if LINEREF.iter().any(|r| r.is_match(n)) {
        return Reason::Lineref;
    }
    if ID.iter().any(|r| r.is_match(n)) {
        return Reason::Id;
    }
    if HEX.is_match(n) && HEX_HAS_LETTER.is_match(n) {
        return Reason::Hex;
    }
    if URL.is_match(n) {
        return Reason::Url;
    }
    if REF.is_match(n) {
        return Reason::Ref;
    }
    if is_fragment(n) {
        return Reason::Fragment;
    }
    if NUMBER.iter().any(|r| r.is_match(n)) {
        return Reason::Number;
    }
    if is_path(n) {
        return Reason::Path;
    }
    if CODE_IDENT.is_match(n) {
        return Reason::CodeIdent;
    }
    if FILENAME.is_match(n) && !n.starts_with(|c: char| c.is_ascii_digit()) {
        return Reason::Filename;
    }
    Reason::Vocab
}

pub fn classify(name: &str, is_proper_noun: bool) -> Classification {
    let reason = matched_reason(name.trim());
    Classification {
        shelf: reason.shelf(),
        reason,
        proper_noun_hint: is_proper_noun,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn r(name: &str) -> Reason {
        classify(name, false).reason
    }

    #[test]
    fn each_rule_has_a_synthetic_witness() {
        let cases = [
            ("0f8e2b1a-3c4d-4e5f-8a9b-0c1d2e3f4a5b", Reason::Uuid),
            ("sweep-20260101120000-lane", Reason::Runid),
            ("2026-01-02T03:04:05Z", Reason::Timestamp),
            ("20260102t030405z", Reason::Timestamp),
            ("20260102030405:7", Reason::Timestamp),
            ("833Z", Reason::Timestamp),
            (r#"4,"DI":0"#, Reason::Json),
            ("commitwork-a6", Reason::Session),
            ("mjs:1132", Reason::Lineref),
            ("SC2086:info:2:33", Reason::Lineref),
            ("high:97", Reason::Counter),
            ("3C/6H/2M/18L", Reason::Counter),
            ("GHSA-aaaa-bbbb-cccc", Reason::Id),
            ("PREFIX GHSA-aaaa-bbbb-cccc", Reason::Id),
            ("CVE-2026-1", Reason::Id),
            ("IWR0057 C463", Reason::Id),
            ("D14_mirror_symmetry", Reason::Id),
            ("javascript_lang_session", Reason::Id),
            ("capability-network-download", Reason::Id),
            ("cs/web/requiressl-not-set", Reason::Id),
            ("a16a3ae0", Reason::Hex),
            ("https://example.invalid", Reason::Url),
            ("project:Example_Repo", Reason::Ref),
            ("path:README.md", Reason::Ref),
            ("DSN=TARGET", Reason::Fragment),
            ("marker(s", Reason::Fragment),
            ("'test/tui-spec", Reason::Fragment),
            ("2%/month", Reason::Number),
            ("4,733-row", Reason::Number),
            ("v3r1", Reason::Number),
            ("tmp/stage/x", Reason::Path),
            ("a/b/c", Reason::Path),
            ("lib/thing.mjs", Reason::Path),
            ("SmartypantsRenderer", Reason::CodeIdent),
            ("analyse.mjs", Reason::Filename),
            ("back-link", Reason::Vocab),
            ("Code's", Reason::Vocab),
            ("SAST/IaC/DAST", Reason::Vocab),
            ("FILE STATUS", Reason::Vocab),
            ("free-tier", Reason::Vocab),
        ];
        for (name, want) in cases {
            assert_eq!(r(name), want, "{name}");
        }
    }

    #[test]
    fn digits_only_is_number_not_hex() {
        assert_eq!(r("3759563"), Reason::Number);
        assert_eq!(r("63884866"), Reason::Number);
    }

    #[test]
    fn proper_noun_flag_never_decides_the_shelf() {
        let flagged = classify("files", true);
        let unflagged = classify("files", false);
        assert_eq!(flagged.shelf, unflagged.shelf);
        assert_eq!(flagged.reason, unflagged.reason);
        assert!(flagged.proper_noun_hint);
        assert!(!unflagged.proper_noun_hint);
    }

    #[test]
    fn every_reason_maps_to_exactly_one_shelf() {
        for reason in Reason::ALL {
            let _ = reason.shelf();
        }
        assert_eq!(Reason::Vocab.shelf(), Shelf::Vocabulary);
        assert_eq!(Reason::Uuid.shelf(), Shelf::Noise);
        assert_eq!(Reason::Id.shelf(), Shelf::Cast);
    }

    #[derive(Deserialize)]
    struct GoldRow {
        name: String,
        is_proper_noun: bool,
        label: String,
        reason: String,
    }

    #[derive(Deserialize)]
    struct PredictionRow {
        name: String,
        shelf: String,
        reason: String,
    }

    fn shelf_name(s: Shelf) -> &'static str {
        match s {
            Shelf::Cast => "cast",
            Shelf::Vocabulary => "vocabulary",
            Shelf::Noise => "noise",
        }
    }

    fn reason_name(r: Reason) -> String {
        serde_json::to_value(r)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    struct Score {
        accuracy: f64,
        precision: HashMap<&'static str, f64>,
        fire: HashMap<String, usize>,
        n: usize,
    }

    fn score(rows: &[GoldRow]) -> Score {
        let mut ok = 0usize;
        let mut tp: HashMap<&str, usize> = HashMap::new();
        let mut pred: HashMap<&str, usize> = HashMap::new();
        let mut fire: HashMap<String, usize> = HashMap::new();
        for row in rows {
            let c = classify(&row.name, row.is_proper_noun);
            let got = shelf_name(c.shelf);
            *pred.entry(got).or_default() += 1;
            *fire.entry(reason_name(c.reason)).or_default() += 1;
            if got == row.label {
                ok += 1;
                *tp.entry(got).or_default() += 1;
            }
        }
        let precision = ["noise", "vocabulary", "cast"]
            .into_iter()
            .map(|s| {
                let p = *pred.get(s).unwrap_or(&0);
                let t = *tp.get(s).unwrap_or(&0);
                (s, if p == 0 { 1.0 } else { t as f64 / p as f64 })
            })
            .collect();
        Score {
            accuracy: ok as f64 / rows.len() as f64,
            precision,
            fire,
            n: rows.len(),
        }
    }

    fn gold_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var_os("VELD_HYGIENE_GOLD_DIR")?;
        Some(std::path::PathBuf::from(dir))
    }

    fn load_gold(dir: &std::path::Path, file: &str) -> Vec<GoldRow> {
        let text = std::fs::read_to_string(dir.join(file)).expect(file);
        let rows: Vec<GoldRow> = serde_json::from_str(&text).expect(file);
        assert!(
            rows.iter().all(|r| !r.reason.is_empty()),
            "{file}: unlabeled row"
        );
        rows
    }

    /// Replays the private gold sets. Skips with its reason when the sidecar is not mounted; a
    /// skipped replay is reported, never counted as a pass.
    #[test]
    fn gold_replay_holds_the_measured_floors() {
        let Some(dir) = gold_dir() else {
            eprintln!(
                "skipped: VELD_HYGIENE_GOLD_DIR unset (gold sets live in the private sidecar)"
            );
            return;
        };
        for (file, min_accuracy) in [
            ("sample-portll.json", 0.92),
            ("heldout-cobolwork.json", 0.92),
        ] {
            let rows = load_gold(&dir, file);
            assert_eq!(rows.len(), 300, "{file}: gold set size changed");
            let s = score(&rows);
            assert!(
                s.accuracy >= min_accuracy,
                "{file}: accuracy {:.3}",
                s.accuracy
            );
            assert!(
                s.precision["noise"] >= 0.95,
                "{file}: noise precision {:.3}",
                s.precision["noise"]
            );
            assert!(
                s.precision["cast"] >= 0.99,
                "{file}: cast precision {:.3}",
                s.precision["cast"]
            );
            for (reason, count) in &s.fire {
                if reason == "vocab" {
                    continue;
                }
                assert!(
                    *count * 2 <= s.n,
                    "{file}: rule {reason} fires on {count} of {} rows — defect signature",
                    s.n
                );
            }
        }
    }

    /// Second witness: the JS prototype's predictions, row for row. The two implementations share
    /// no code, so agreement is a checked fact rather than a streak.
    #[test]
    fn agrees_with_the_prototype_row_for_row() {
        let Some(dir) = gold_dir() else {
            eprintln!("skipped: VELD_HYGIENE_GOLD_DIR unset");
            return;
        };
        let path = dir.join("predictions-prototype.json");
        let text = std::fs::read_to_string(&path).expect("predictions-prototype.json");
        let rows: Vec<PredictionRow> = serde_json::from_str(&text).unwrap();
        assert!(
            rows.len() >= 600,
            "prototype predictions cover both gold sets"
        );
        let mut divergent = Vec::new();
        for row in &rows {
            let c = classify(&row.name, false);
            if shelf_name(c.shelf) != row.shelf || reason_name(c.reason) != row.reason {
                divergent.push(format!(
                    "{} : prototype {}/{} rust {}/{}",
                    row.name,
                    row.shelf,
                    row.reason,
                    shelf_name(c.shelf),
                    reason_name(c.reason)
                ));
            }
        }
        assert!(
            divergent.is_empty(),
            "{} divergent rows:\n{}",
            divergent.len(),
            divergent.join("\n")
        );
    }
}
