//! Entity redaction — classify credential-shaped strings before they enter the graph.
//!
//! WHY ON THE WAY IN. The forget path does not touch the graph, so removing a memory does not
//! remove the entities it minted. `DELETE /api/graph/entity/delete` now exists and is the exit,
//! but it is a remediation tool, not a control: it acts on one known id at a time, after the fact,
//! and only if somebody notices. Entry is still the only place the decision is made by default.
//! (Until 2026-08-26 there was no exit at all — `graph_memory::delete_entity` was implemented and
//! routed from nothing.)
//!
//! WHY A LABEL AND NOT A DROP. A redacted entity is replaced with `redacted:<class>`, never
//! removed. Dropping it would make "there was a credential here" indistinguishable from "there was
//! nothing here", and an absence that reads as a clean result is the defect this whole discipline
//! exists to refuse. The label also survives dedup: two API keys in one memory collapse to one
//! `redacted:key`, which is the correct cardinality — the fact is "a key was present", not "two".
//!
//! THE LEXICON IS DERIVED FROM PRECISE SHAPES, NOT FROM ENTROPY. Measured against the live store
//! 2026-08-26 over 22,620 entities in 500 records: a generic "no whitespace, >=28 chars, opaque"
//! rule matched **415 distinct entities**, and its hits were slugs like
//! `detection-reachability-lexical-grammatical-semantic`. That rule would have destroyed recall to
//! catch nothing, so it is deliberately absent. Every classifier below anchors on a documented
//! vendor prefix, a structural delimiter, or an explicit assignment.
//!
//! WHAT THIS DOES NOT CATCH, stated because a partial control presented as complete is the defect:
//! a secret that looks like a word. A passphrase, or a project codename used as a shared secret,
//! passes every rule here and there is no pattern answer to it. Filtering is not containment; the
//! only complete control is not submitting the content.

use once_cell::sync::Lazy;
use regex::Regex;

/// The redaction lexicon. Every variant renders as `redacted:<slug>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Redacted {
    /// A vendor-issued API key with a documented prefix.
    Key,
    /// A private key block.
    PrivateKey,
    /// A signed bearer token (JWT shape).
    Token,
    /// Credentials embedded in a URL's userinfo, including DB connection strings.
    Login,
    /// An explicit `password=`/`secret=`/`api_key=` assignment carrying a value.
    Secret,
    /// A filesystem path naming a user account.
    Path,
}

impl Redacted {
    pub fn label(self) -> &'static str {
        match self {
            Redacted::Key => "redacted:key",
            Redacted::PrivateKey => "redacted:private-key",
            Redacted::Token => "redacted:token",
            Redacted::Login => "redacted:login",
            Redacted::Secret => "redacted:secret",
            Redacted::Path => "redacted:path",
        }
    }
}

/// Vendor prefixes are the highest-confidence signal available: each is a documented, issued
/// format, so a match is near-certainly a real credential rather than prose about one.
static KEY: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?:sk_live_|sk_test_|rk_live_|pk_live_)[A-Za-z0-9]{8,}",
        r"|AKIA[0-9A-Z]{16}",
        r"|gh[pousr]_[A-Za-z0-9]{20,}",
        r"|github_pat_[A-Za-z0-9_]{20,}",
        r"|xox[baprs]-[A-Za-z0-9-]{10,}",
        r"|AIza[0-9A-Za-z_\-]{35}",
        r"|ya29\.[0-9A-Za-z_\-]{20,}",
        r"|SG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}",
        r"|npm_[A-Za-z0-9]{36}",
        r"|glpat-[A-Za-z0-9_\-]{20,}",
        r"|dop_v1_[a-f0-9]{64}",
    ))
    .expect("KEY pattern is a literal and must compile")
});

static PRIVATE_KEY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"-----BEGIN [A-Z ]*PRIVATE KEY-----").expect("literal"));

/// Three dot-separated base64url segments, the middle one decoding to a claims object in practice.
/// The `eyJ` anchor is the base64 of `{"`, which is what makes this specific rather than generic.
static TOKEN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}").expect("literal")
});

/// `scheme://user:pass@host`. The colon-before-at is the structural tell; no entropy needed.
static LOGIN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@").expect("literal"));

/// An explicit assignment WITH a value. Requires 6+ non-placeholder characters so that prose about
/// secrets does not trip it: measured on the live store, the only assignment-shaped entities were
/// `'password=***'` and a regex fragment, and both must pass through unredacted.
static SECRET: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)\b(?:pass(?:word|wd)?|secret|api[_\-]?key|auth[_\-]?token|credential)\b\s*[:=]\s*["']?[A-Za-z0-9+/=_\-]{6,}"#)
        .expect("literal")
});

/// A home directory names its account holder. Not a credential, but it is identifying, it appears
/// in the corpus (7 distinct, all real), and it travels into a queryable namespace.
static PATH: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"/(?:Users|home)/[A-Za-z0-9._\-]+").expect("literal"));

/// Placeholders that LOOK like assignments and carry nothing. Redacting these would be a false
/// positive on documentation, which is most of what the corpus actually contains.
static PLACEHOLDER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)[:=]\s*["']?(?:\*{3,}|x{3,}|\.{3,}|<[^>]+>|\$\{[^}]+\}|redacted|changeme|your[_\-]?\w+|example|placeholder|null|none|true|false)\b"#)
        .expect("literal")
});

/// Classify a single entity string. `None` means it may enter the graph unchanged.
///
/// Order matters: the most specific and highest-confidence shapes are tested first, so a
/// connection string carrying an API key is labelled by its strongest signal.
pub fn classify(entity: &str) -> Option<Redacted> {
    if PRIVATE_KEY.is_match(entity) {
        return Some(Redacted::PrivateKey);
    }
    if KEY.is_match(entity) {
        return Some(Redacted::Key);
    }
    if TOKEN.is_match(entity) {
        return Some(Redacted::Token);
    }
    if LOGIN.is_match(entity) {
        return Some(Redacted::Login);
    }
    // A placeholder assignment is documentation, not a credential. Checked only for SECRET,
    // because a real vendor key is a real vendor key regardless of what sits beside it.
    if SECRET.is_match(entity) && !PLACEHOLDER.is_match(entity) {
        return Some(Redacted::Secret);
    }
    if PATH.is_match(entity) {
        return Some(Redacted::Path);
    }
    None
}

/// A bare UUID. Measured on the live store 2026-08-26: 216 distinct UUID-shaped entities across
/// 500 records, carried by 71.8% of them, consuming 2.44% of all entity slots.
///
/// These are session identifiers, and they are NOT a redaction case — nothing about them is
/// sensitive, and labelling them `redacted:` would assert a confidentiality they do not have.
/// They are a NAMESPACE case: an opaque machine id has no semantic recall value (nobody searches
/// `3301d4ef`), and 42.6% of them appear in exactly one record, so they link nothing whatever.
///
/// The ones that DO recur are the problem. Four appear in ten or more records, and because an
/// entity is a graph node, every memory sharing a session id becomes one hop from every other.
/// That manufactures semantic adjacency out of a scheduling coincidence: two memories written in
/// the same session may have nothing to do with each other, and spreading activation cannot tell
/// the difference.
///
/// OPERATOR RULING 2026-08-26: **UUID entities are ALLOWED.** They stay in the graph. This
/// predicate is informational only — for reporting and for anyone wanting to filter at read
/// time — and must NOT be wired to a drop at write time without a new ruling. The hub effect
/// described above is a known and accepted property, not an open defect.
static BARE_UUID: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        .expect("literal")
});

/// True for an entity that is a bare machine identifier with no semantic content.
///
/// Anchored at both ends deliberately: `session-3301d4ef-...` carries a human-readable prefix and
/// is a legitimate entity, while the bare id is not.
pub fn is_non_semantic_id(entity: &str) -> bool {
    BARE_UUID.is_match(entity)
}

/// Outcome of redacting a list, so the caller can log WHAT was replaced without logging the value.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RedactionReport {
    pub key: usize,
    pub private_key: usize,
    pub token: usize,
    pub login: usize,
    pub secret: usize,
    pub path: usize,
}

impl RedactionReport {
    pub fn total(&self) -> usize {
        self.key + self.private_key + self.token + self.login + self.secret + self.path
    }
    fn count(&mut self, r: Redacted) {
        match r {
            Redacted::Key => self.key += 1,
            Redacted::PrivateKey => self.private_key += 1,
            Redacted::Token => self.token += 1,
            Redacted::Login => self.login += 1,
            Redacted::Secret => self.secret += 1,
            Redacted::Path => self.path += 1,
        }
    }
}

/// Replace every credential-shaped entity with its class label, in place, preserving order.
///
/// Order is preserved deliberately: the caller truncates to a cap AFTER this runs, and truncation
/// is by arrival order. Reordering here would silently change which entities survive the cap.
pub fn redact_entities(entities: &mut [String]) -> RedactionReport {
    let mut report = RedactionReport::default();
    for e in entities.iter_mut() {
        if let Some(class) = classify(e) {
            report.count(class);
            *e = class.label().to_string();
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_keys_are_classified_by_prefix() {
        for (s, want) in [
            ("sk_live_abcdefghijklmnop", Redacted::Key),
            ("AKIAIOSFODNN7EXAMPLE", Redacted::Key),
            ("ghp_abcdefghijklmnopqrstuvwxyz0123", Redacted::Key),
            ("xoxb-1234567890-abcdefghij", Redacted::Key),
            ("npm_abcdefghijklmnopqrstuvwxyz0123456789", Redacted::Key),
        ] {
            assert_eq!(classify(s), Some(want), "failed to classify {s}");
        }
    }

    #[test]
    fn structural_shapes_are_classified() {
        assert_eq!(
            classify("-----BEGIN OPENSSH PRIVATE KEY-----"),
            Some(Redacted::PrivateKey)
        );
        assert_eq!(
            classify("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NX0.dBjftJeZ4CVPmB92K27u"),
            Some(Redacted::Token)
        );
        assert_eq!(
            classify("postgres://user:hunter2pass@db.internal"),
            Some(Redacted::Login)
        );
    }

    // THE LOAD-BEARING TEST. These are verbatim entities from the live store, 2026-08-26. Every
    // one is prose ABOUT a credential, and redacting any of them is a false positive that costs
    // recall to catch nothing. The corpus contained zero live secrets.
    #[test]
    fn prose_about_credentials_is_not_redacted() {
        for s in [
            "'password=***'",
            "mjs:generic-api-key:67",
            "detection-reachability-lexical-grammatical-semantic",
            "lexical/grammatical/semantic",
            "3301d4ef-ecd8-4743-af4a-bb50590a1fbe",
            "veld-client-5s-timeout-silently-nulls-large-writes",
            "SCANNER_CHECKS",
            "VELD_TIMEOUT_MS=60000",
            "BEHIND-TOOLCHAIN",
            "password=<YOUR_PASSWORD>",
            "api_key=${API_KEY}",
            "secret=changeme",
        ] {
            assert_eq!(classify(s), None, "false positive on {s}");
        }
    }

    #[test]
    fn a_documentation_example_of_a_connection_string_is_still_login_shaped() {
        // "postgres://user:SECRET@host" IS in the corpus and IS structurally a credential URL.
        // Redacting it is correct even though this instance is documentation: the shape cannot be
        // distinguished from a real one, and erring toward redaction is right for THIS class only
        // because the label preserves the fact.
        assert_eq!(
            classify("postgres://user:SECRET@host"),
            Some(Redacted::Login)
        );
    }

    #[test]
    fn home_paths_name_their_account_holder() {
        assert_eq!(
            classify("cwd=/Users/portll/Repositories/Portll/commitwork"),
            Some(Redacted::Path)
        );
        assert_eq!(classify("/home/runner/work/repo"), Some(Redacted::Path));
        // a path with no user segment is not identifying
        assert_eq!(classify("/usr/local/bin/veld"), None);
    }

    #[test]
    fn redaction_replaces_in_place_preserves_order_and_counts_by_class() {
        let mut v = vec![
            "automated".to_string(),
            "ghp_abcdefghijklmnopqrstuvwxyz0123".to_string(),
            "scope:claude-code".to_string(),
            "postgres://u:p@h".to_string(),
        ];
        let report = redact_entities(&mut v);
        assert_eq!(
            v,
            vec![
                "automated",
                "redacted:key",
                "scope:claude-code",
                "redacted:login"
            ]
        );
        assert_eq!(report.key, 1);
        assert_eq!(report.login, 1);
        assert_eq!(report.total(), 2);
    }

    #[test]
    fn two_keys_collapse_to_one_label_which_is_the_correct_cardinality() {
        let mut v = vec![
            "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "ghp_bbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];
        redact_entities(&mut v);
        assert_eq!(v[0], v[1], "both become the same label");
        // the caller dedups; the surviving fact is "a key was present", not how many
    }

    #[test]
    fn a_bare_uuid_is_non_semantic_but_is_not_redacted() {
        // Both halves matter. It must be recognised as a machine id...
        assert!(is_non_semantic_id("3301d4ef-ecd8-4743-af4a-bb50590a1fbe"));
        assert!(is_non_semantic_id("0A82AE72-52A0-4D5B-865B-EB76747D8C0A"));
        // ...and it must NOT be labelled redacted, because nothing about it is confidential and
        // saying otherwise would put a false claim into the data.
        assert_eq!(classify("3301d4ef-ecd8-4743-af4a-bb50590a1fbe"), None);
    }

    #[test]
    fn a_uuid_with_a_human_prefix_is_a_legitimate_entity() {
        // The anchors are load-bearing: only the BARE id is meaningless.
        assert!(!is_non_semantic_id(
            "session-3301d4ef-ecd8-4743-af4a-bb50590a1fbe"
        ));
        assert!(!is_non_semantic_id(
            "3301d4ef-ecd8-4743-af4a-bb50590a1fbe/report"
        ));
        assert!(!is_non_semantic_id("veld-project:commitwork"));
    }

    #[test]
    fn a_hyphenated_slug_is_neither_redacted_nor_non_semantic() {
        // The rule this replaces — "no whitespace, >=28 chars, opaque" — matched 415 distinct
        // entities on the live store, of which these are representative. It would have cost recall
        // to catch nothing, and it is deliberately absent from this module.
        for s in [
            "detection-reachability-lexical-grammatical-semantic",
            "veld-client-5s-timeout-silently-nulls-large-writes",
            "a-test-that-asserts-a-marker-is-set-does-not-assert-anything-obeys-it",
        ] {
            assert_eq!(classify(s), None, "redacted a slug: {s}");
            assert!(!is_non_semantic_id(s), "called a slug a machine id: {s}");
        }
    }

    /// THE GATE. Every path that caps entities must redact FIRST, and this asserts it by reading
    /// the source rather than by trusting a list someone maintains beside it.
    ///
    /// The failure it exists to catch is not a wrong regex — it is a NEW entry path added later
    /// with no redaction at all, which is invisible to every test above because those only exercise
    /// this module. A control on one door is not a control.
    ///
    /// The file set is DERIVED (walk src/, find the callers) rather than hardcoded, so a fifth
    /// entry path fails this test the day it lands instead of the day someone remembers to add it.
    #[test]
    fn every_capping_path_redacts_before_it_caps() {
        use std::fs;
        use std::path::{Path, PathBuf};

        // The cap call, not a raw truncate. It changed on 2026-08-28 when the cap became
        // selectivity-ordered, and this test FAILED rather than passing vacuously when it did —
        // the `capping_files >= 3` floor below is what caught it. That is the floor earning its
        // keep: a marker-based gate whose marker is renamed asserts nothing, silently.
        // The QUALIFIED call, not the bare name: `entity_cap.rs` DEFINES `cap_by_selectivity` and
        // would otherwise match its own marker and be reported as an unredacted entry path. Keying
        // on the qualified path distinguishes a call site from a definition without an exclusion
        // list, which is the thing that quietly shrinks a gate's population over time.
        const CAP: &str = "entity_cap::cap_by_selectivity(";
        const REDACT: &str = "redact_entities(";
        // Nothing may cap by position any more. A blind truncate would cut by ARRIVAL order, which
        // was measured to discard material 19x more selective than what it kept.
        const BLIND: &str = "truncate(validation::MAX_ENTITIES_PER_MEMORY)";

        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            for e in fs::read_dir(dir).expect("src/ must be readable") {
                let p = e.expect("dir entry").path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }

        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&src, &mut files);
        assert!(
            files.len() > 50,
            "walked only {} files — the walk is broken, and a broken walk passes this test vacuously",
            files.len()
        );

        let mut capping_files = 0;
        for f in &files {
            let text = fs::read_to_string(f).expect("source must be readable");
            let name_early = f.strip_prefix(&src).unwrap_or(f).display().to_string();

            // This file states BLIND as a literal to test for it; everything else that contains it
            // is a path that caps by position.
            if name_early != "redaction.rs" && text.contains(BLIND) {
                panic!(
                    "{name_early} truncates the entity list by ARRIVAL order. Measured 2026-08-27, \
                     the tail that discards is 19x more selective than the head it keeps — use \
                     crate::entity_cap::cap_by_selectivity"
                );
            }

            let caps: Vec<usize> = text.match_indices(CAP).map(|(i, _)| i).collect();
            if caps.is_empty() {
                continue;
            }
            capping_files += 1;
            let redacts: Vec<usize> = text.match_indices(REDACT).map(|(i, _)| i).collect();
            let name = f.strip_prefix(&src).unwrap_or(f).display();

            assert!(
                !redacts.is_empty(),
                "{name} caps entities but never calls redact_entities — a new entry path reached \
                 the graph with no redaction"
            );
            assert!(
                redacts.len() >= caps.len(),
                "{name} has {} cap site(s) but only {} redact call(s) — at least one path caps \
                 without redacting",
                caps.len(),
                redacts.len()
            );
            for (n, cap_at) in caps.iter().enumerate() {
                assert!(
                    redacts.iter().any(|r| r < cap_at),
                    "{name}: cap site #{} has no redact_entities call before it; filtering after \
                     truncate means a credential has already displaced a legitimate entity",
                    n + 1
                );
            }
        }

        // Refuse a vacuous pass: if the walk found no capping files at all, the constants have
        // been renamed and this test is asserting nothing.
        assert!(
            capping_files >= 3,
            "found only {capping_files} file(s) that cap entities — expected the remember, ingest \
             and zenoh paths at least. Has the cap call been renamed? This floor is what caught the \
             2026-08-28 rename instead of letting this test pass asserting nothing."
        );
    }

    #[test]
    fn an_empty_list_reports_zero_rather_than_failing() {
        let mut v: Vec<String> = vec![];
        assert_eq!(redact_entities(&mut v).total(), 0);
    }
}
