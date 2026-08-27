//! The local LLM hosts veld will talk to, read from `manifests/llm-hosts.json`.
//!
//! # This is a mirrored file, and that is normally the wrong answer
//!
//! commitwork's registry inventory has a name for two copies of one truth: `mirrored`, defined
//! there as *"the schema is documentation and the code is the gate, so they drift and only the code
//! bites."* That is an accurate description of the risk being taken here, so it is worth saying why
//! it is taken anyway and what makes it survivable.
//!
//! veld ships standalone. It cannot read a file out of a sibling checkout that may not exist, and a
//! build-time path into another repository would make veld unbuildable outside this fleet. So the
//! declaration is vendored and `include_str!`d — the binary carries it and works alone.
//!
//! What stops that being the failure mode commitwork names is that **drift bites**:
//! `bin/test/llm-hosts-drift.test.mjs` over there compares the two copies byte-for-byte and fails
//! on divergence, and — the part that matters — it reports SKIPPED-because-absent rather than
//! passing when veld is not checked out beside it. A drift test that quietly passes when it cannot
//! see the other copy is the same false clean as no test.
//!
//! The canonical copy is commitwork's. If they disagree, that one is right.

use serde::Deserialize;
use std::sync::OnceLock;

const DECLARATION: &str = include_str!("../manifests/llm-hosts.json");

#[derive(Debug, Clone, Deserialize)]
pub struct LlmHosts {
    pub version: u32,
    /// The host named in documentation and used when nothing is detected and nothing is
    /// configured. Deliberately NOT the head of `probe_order` — see `default_rationale` in the
    /// declaration.
    pub default: String,
    #[serde(rename = "probeOrder")]
    pub probe_order: Vec<String>,
    pub hosts: Vec<Host>,
    #[serde(default)]
    pub removed: Vec<Removed>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Host {
    pub id: String,
    pub label: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub licence: String,
    #[serde(rename = "openSource")]
    pub open_source: bool,
    /// True when another declared host defaults to the same port. A probe of a shared port MUST
    /// NOT name the host: the protocol is identical, so nothing in the response distinguishes
    /// llama.cpp from llamafile from LocalAI from mlx-lm, all of which default to 8080.
    #[serde(rename = "portIsShared")]
    pub port_is_shared: bool,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Removed {
    pub id: String,
    #[serde(rename = "removedOn")]
    pub removed_on: String,
    pub reason: String,
}

impl Host {
    pub fn serves(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    /// `VELD_LLM_URL_LLAMA_CPP`, `VELD_LLM_URL_LMSTUDIO`, …
    pub fn url_env_var(&self) -> String {
        format!("VELD_LLM_URL_{}", self.id.to_uppercase().replace(['.', '-'], "_"))
    }

    /// Base URL honouring the per-host override. Read at CALL time, never cached at load: a
    /// `const` populated at import silently defeats the override for any test that sets it after.
    pub fn base_url(&self) -> String {
        self.base_url_with(|k| std::env::var(k).ok())
    }

    /// `base_url` with the environment injected.
    ///
    /// Exists so the override can be tested WITHOUT `std::env::set_var`. Rust runs tests in
    /// parallel threads and the process environment is shared mutable state, so a test that sets a
    /// variable races every other test that reads one — which is not theoretical: the first version
    /// of this file mutated env in a test and the suite failed once, in an unrelated module, and
    /// then passed on rerun. A test that passes on rerun is flaky, not fixed.
    pub fn base_url_with(&self, lookup: impl Fn(&str) -> Option<String>) -> String {
        lookup(&self.url_env_var()).unwrap_or_else(|| self.base_url.clone())
    }
}

/// The declaration. Panics at first use if the vendored file is malformed — it is compiled in, so
/// that is a build-integrity failure rather than a runtime condition to degrade around.
pub fn declaration() -> &'static LlmHosts {
    static PARSED: OnceLock<LlmHosts> = OnceLock::new();
    PARSED.get_or_init(|| {
        serde_json::from_str(DECLARATION)
            .expect("manifests/llm-hosts.json is compiled in and must parse")
    })
}

pub fn host(id: &str) -> Option<&'static Host> {
    declaration().hosts.iter().find(|h| h.id == id)
}

/// Hosts in probe order, anything unordered last.
pub fn in_probe_order() -> Vec<&'static Host> {
    let d = declaration();
    let mut out: Vec<&Host> = d.probe_order.iter().filter_map(|id| host(id)).collect();
    out.extend(d.hosts.iter().filter(|h| !d.probe_order.contains(&h.id)));
    out
}

/// What a live server on `port` may be CALLED.
///
/// `None` is the honest answer for a shared port, and the caller must render it as such. Returning
/// the first match would be reporting a host nobody measured.
pub fn identify_by_port(port: u16) -> Option<&'static Host> {
    let matches: Vec<&Host> = declaration()
        .hosts
        .iter()
        .filter(|h| port_of(&h.base_url) == Some(port))
        .collect();
    match matches.len() {
        1 => Some(matches[0]),
        _ => None,
    }
}

fn port_of(url: &str) -> Option<u16> {
    url.rsplit(':').next()?.trim_end_matches('/').parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vendored_declaration_parses_and_is_self_consistent() {
        let d = declaration();
        assert!(d.version >= 1);
        assert!(host(&d.default).is_some(), "default names no declared host");
        for id in &d.probe_order {
            assert!(host(id).is_some(), "probeOrder names {id}, which no host declares");
        }
    }

    /// The ruling of 2026-08-27: documented default is the MIT one; LM Studio is merely probed
    /// first. If someone collapses these two ideas back together this fails.
    #[test]
    fn the_default_is_open_source_even_though_it_is_not_probed_first() {
        let d = declaration();
        let def = host(&d.default).expect("default exists");
        assert!(def.open_source, "the documented default must be nameable in published work");
        assert_ne!(
            d.default, d.probe_order[0],
            "default and probe order are different questions and were conflated before this file"
        );
        let first = host(&d.probe_order[0]).expect("probe head exists");
        assert!(
            !first.port_is_shared,
            "the first probe must be the one whose hit means something definite"
        );
    }

    #[test]
    fn a_shared_port_is_not_identified() {
        let shared: Vec<&Host> = declaration().hosts.iter().filter(|h| h.port_is_shared).collect();
        assert!(shared.len() >= 2, "the 8080 collision is the case this exists for");
        let port = port_of(&shared[0].base_url).expect("has a port");
        assert!(
            identify_by_port(port).is_none(),
            ":{port} is claimed by several hosts; naming one would be asserting something unmeasured"
        );

        // …and the unshared case still resolves, so this is not just "never identify anything".
        let solo = declaration().hosts.iter().find(|h| !h.port_is_shared).expect("one exists");
        let solo_port = port_of(&solo.base_url).expect("has a port");
        assert_eq!(identify_by_port(solo_port).map(|h| h.id.as_str()), Some(solo.id.as_str()));
    }

    #[test]
    fn ollama_is_recorded_as_removed_rather_than_silently_absent() {
        let d = declaration();
        assert!(!d.hosts.iter().any(|h| h.id == "ollama"));
        let gone = d.removed.iter().find(|r| r.id == "ollama").expect("removal is recorded");
        assert!(!gone.reason.is_empty() && gone.removed_on.starts_with("2026-"));
    }

    #[test]
    fn capabilities_are_narrow_enough_to_stop_a_404() {
        // mlx_lm.server has no /v1/embeddings. A caller that assumes every host embeds gets a 404,
        // which is the kind of thing a declaration is for.
        let mlx = host("mlx-lm").expect("declared");
        assert!(mlx.serves("chat"));
        assert!(!mlx.serves("embeddings"));
    }

    /// No `set_var` anywhere: the environment is injected. See `base_url_with`.
    #[test]
    fn a_base_url_override_is_honoured_and_keyed_per_host() {
        let h = host("lmstudio").expect("declared");
        assert_eq!(h.url_env_var(), "VELD_LLM_URL_LMSTUDIO");
        // The dot in `llama.cpp` has to survive into a legal variable name.
        assert_eq!(host("llama.cpp").unwrap().url_env_var(), "VELD_LLM_URL_LLAMA_CPP");

        assert_eq!(h.base_url_with(|_| None), h.base_url, "no override, declared value");
        assert_eq!(
            h.base_url_with(|k| (k == "VELD_LLM_URL_LMSTUDIO").then(|| "http://127.0.0.1:9999".into())),
            "http://127.0.0.1:9999"
        );
        // A different host's override must not leak across.
        assert_eq!(
            h.base_url_with(|k| (k == "VELD_LLM_URL_LLAMA_CPP").then(|| "http://nope".into())),
            h.base_url
        );
    }
}
