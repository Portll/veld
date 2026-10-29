//! Single source of truth for retrieval-time ablation families.
//!
//! Ported from shodh-memory `src/memory/ablation.rs`. Every ablatable retrieval
//! component names a FAMILY here and asks [`is_enabled`] at its site.
//!
//! `SHODH_DISABLE_BOOSTS=<family,...>` disables the named families. The token
//! `all` disables every family. Setting the variable at all, even to the empty
//! string, REPLACES [`DEFAULT_DISABLED`] rather than adding to it.
//!
//! The env is read on every call, never cached: the recall harness and the
//! tests set it after the process has loaded this module.
//!
//! Divergence from upstream, deliberately:
//! * [`FAMILIES`] lists only the families veld actually gates. Upstream declares
//!   23; the rest name components veld either lacks (`lateral_inhibition`) or has
//!   never gated. Those are in [`UPSTREAM_ONLY_FAMILIES`] so a harness arm that
//!   names one reports the ablation as unavailable instead of measuring baseline.
//!
//! [`DEFAULT_DISABLED`] holds `hebbian`, as upstream's does (shodh-memory #321). The
//! term added a graph hop decay of up to 1.0, scaled by 0.1, to fused scores whose top
//! ten spread over about 0.007, and only for memories the graph leg reached. On the
//! LoCoMo gate, switching it off lifted nDCG@10 from 0.383 to about 0.44. It is now a
//! share of the fused score, at most 10%, and stays off until that form is measured.

use std::collections::HashSet;

/// Env var holding the comma-separated disabled-family list.
pub const DISABLE_BOOSTS_VAR: &str = "SHODH_DISABLE_BOOSTS";

/// Every family veld gates, each with an [`is_enabled`] call at its site.
///
/// * `hebbian` — the Layer-5 hebbian term `score * (1 + 0.1 * hebbian_boost)` in
///   `recall.rs`. Off by default ([`DEFAULT_DISABLED`]); `SHODH_DISABLE_BOOSTS` set
///   without it, even to the empty string, switches it on. Also switchable through
///   the `hebbian` cognitive-dial signal; the term runs only when both allow it.
/// * `graph_potentiation` — retrieval-driven edge strengthening
///   (`batch_strengthen_synapses`) in the graph leg's traversals
///   (`traverse_from_entity_filtered`, `traverse_weighted`, `traverse_bidirectional`).
///   Distinct from `SHODH_RECALL_READONLY`, which suppresses every recall-path write
///   for measurement integrity; this family ablates the one mechanism.
///
/// Adding a name here without a matching [`is_enabled`] call at the component's
/// site makes the family a lie; `recall_harness::runner`'s guard test checks both.
pub const FAMILIES: [&str; 2] = ["hebbian", "graph_potentiation"];

/// Upstream's families that veld does not gate. An ablation naming one is
/// unavailable in veld, and [`unavailable_reason`] says why.
pub const UPSTREAM_ONLY_FAMILIES: [&str; 21] = [
    "attribute",
    "temporal_prefilter",
    "temporal_fact",
    "interference",
    "prospective",
    "fact_source",
    "ontological",
    "recency",
    "arousal",
    "credibility",
    "temporal_match",
    "feedback",
    "importance",
    "tag_penalty",
    "quality",
    "linguistic",
    "competition",
    "lateral_inhibition",
    "quality_verbosity",
    "linguistic_resort",
    "size_gated_final_sort",
];

/// Families disabled when `SHODH_DISABLE_BOOSTS` is unset; see the module docs.
pub const DEFAULT_DISABLED: [&str; 1] = ["hebbian"];

/// Why an ablation of `family` cannot run in veld, or `None` if it can.
pub fn unavailable_reason(family: &str) -> Option<&'static str> {
    match family {
        "lateral_inhibition" => Some(
            "veld's graph leg has no lateral-inhibition stage (upstream graph_retrieval Step 6.5 was never ported)",
        ),
        f if UPSTREAM_ONLY_FAMILIES.contains(&f) => {
            Some("declared upstream, but veld has no ablation gate for this family")
        }
        _ => None,
    }
}

/// Resolve the disabled-family set from the environment.
///
/// Unknown tokens are warned about and ignored, so a typo degrades to "nothing
/// extra disabled" rather than appearing to ablate a family that does not exist.
pub fn disabled_families() -> HashSet<String> {
    let set: HashSet<String> = match std::env::var("SHODH_DISABLE_BOOSTS") {
        Ok(v) => v
            .split(',')
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect(),
        Err(_) => DEFAULT_DISABLED.iter().map(|s| s.to_string()).collect(),
    };
    for tok in &set {
        if tok == "all" || FAMILIES.contains(&tok.as_str()) {
            continue;
        }
        match unavailable_reason(tok) {
            Some(why) => {
                tracing::warn!("SHODH_DISABLE_BOOSTS: '{tok}' is not ablatable here: {why}")
            }
            None => {
                tracing::warn!("SHODH_DISABLE_BOOSTS: unknown ablation family '{tok}' (ignored)")
            }
        }
    }
    set
}

/// Whether `family` is currently ablated.
pub fn is_disabled(family: &str) -> bool {
    debug_assert!(
        FAMILIES.contains(&family),
        "ablation family is not declared in ablation::FAMILIES"
    );
    let disabled = disabled_families();
    disabled.contains("all") || disabled.contains(family)
}

/// Whether `family` should run. The form call sites read most naturally.
pub fn is_enabled(family: &str) -> bool {
    !is_disabled(family)
}

/// `SHODH_DISABLE_FACT_LAYERS=1|true` — skip the query-time fact layers (Layer 0.6
/// temporal-fact lookup, Layer 0.7 fact-source lookup) so their contribution can be
/// measured against the same ingested corpus.
pub fn fact_layers_disabled() -> bool {
    std::env::var("SHODH_DISABLE_FACT_LAYERS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// `SHODH_GRAPH_FUSION_WEIGHT=<f32>` — override the graph leg's Layer-4 fusion
/// weight, clamped to [0, 1]. `0` is graph-off: the leg's RRF term and its
/// activation bonus (both scaled by the weight) contribute nothing.
pub fn graph_fusion_weight_override() -> Option<f32> {
    std::env::var("SHODH_GRAPH_FUSION_WEIGHT")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|w| w.is_finite())
        .map(|w| w.clamp(0.0, 1.0))
}

/// `SHODH_FUSION_RRF=1|true` — fuse recall Layer 4 by reciprocal rank instead of FLAT,
/// under upstream's name for the same escape hatch (shodh-memory #320), so its arms run
/// unchanged. The `fusion` cognitive dial, when set, wins over this variable.
pub fn rrf_fusion_requested() -> bool {
    std::env::var("SHODH_FUSION_RRF")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// The Layer 4 fusion a query runs under: the `fusion` dial, else
/// [`rrf_fusion_requested`], else FLAT.
pub fn fusion_mode(
    dial: Option<crate::memory::dials::FusionMode>,
) -> crate::memory::dials::FusionMode {
    use crate::memory::dials::FusionMode;
    dial.unwrap_or(if rrf_fusion_requested() {
        FusionMode::Rrf
    } else {
        FusionMode::Flat
    })
}

/// Serialises every test in the crate that mutates an ablation env var.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn with_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        match value {
            Some(v) => std::env::set_var(DISABLE_BOOSTS_VAR, v),
            None => std::env::remove_var(DISABLE_BOOSTS_VAR),
        }
        let out = f();
        std::env::remove_var(DISABLE_BOOSTS_VAR);
        out
    }

    #[test]
    fn family_lists_are_disjoint_and_cover_upstream() {
        let wired: HashSet<&str> = FAMILIES.iter().copied().collect();
        let upstream_only: HashSet<&str> = UPSTREAM_ONLY_FAMILIES.iter().copied().collect();
        assert_eq!(wired.len(), FAMILIES.len(), "duplicate in FAMILIES");
        assert_eq!(
            upstream_only.len(),
            UPSTREAM_ONLY_FAMILIES.len(),
            "duplicate in UPSTREAM_ONLY_FAMILIES"
        );
        assert!(wired.is_disjoint(&upstream_only));
        // Upstream declares 23 families; every one is either wired or explained.
        assert_eq!(wired.len() + upstream_only.len(), 23);
    }

    #[test]
    fn every_upstream_only_family_has_a_reason_and_no_wired_one_does() {
        for f in UPSTREAM_ONLY_FAMILIES {
            assert!(unavailable_reason(f).is_some(), "{f} has no reason");
        }
        for f in FAMILIES {
            assert!(unavailable_reason(f).is_none(), "{f} is wired");
        }
    }

    #[test]
    fn unset_env_disables_only_hebbian() {
        with_env(None, || {
            assert!(is_disabled("hebbian"));
            assert!(is_enabled("graph_potentiation"));
        });
    }

    #[test]
    fn an_empty_value_replaces_the_default() {
        with_env(Some(""), || {
            for f in FAMILIES {
                assert!(is_enabled(f), "an empty list must enable {f}");
            }
        });
    }

    #[test]
    fn all_disables_every_declared_family() {
        with_env(Some("all"), || {
            for f in FAMILIES {
                assert!(is_disabled(f), "family survived 'all': {f}");
                assert!(!is_enabled(f), "is_enabled disagreed with is_disabled: {f}");
            }
        });
    }

    #[test]
    fn named_families_are_disabled_and_others_untouched() {
        with_env(Some(" Hebbian "), || {
            assert!(is_disabled("hebbian"));
            assert!(is_enabled("graph_potentiation"));
        });
        with_env(Some("graph_potentiation"), || {
            assert!(is_disabled("graph_potentiation"));
            assert!(is_enabled("hebbian"));
        });
    }

    #[test]
    fn the_env_is_read_at_call_time() {
        with_env(None, || {
            assert!(is_disabled("hebbian"));
            std::env::set_var(DISABLE_BOOSTS_VAR, "graph_potentiation");
            assert!(is_enabled("hebbian"), "a set after first read was ignored");
            std::env::remove_var(DISABLE_BOOSTS_VAR);
            assert!(is_disabled("hebbian"));
        });
    }

    #[test]
    fn unknown_and_upstream_only_tokens_disable_nothing_real() {
        for tok in ["latteral_inhibition", "lateral_inhibition", "recency"] {
            with_env(Some(tok), || {
                for f in FAMILIES {
                    assert!(is_enabled(f), "{tok} disabled a real family: {f}");
                }
            });
        }
    }

    #[test]
    fn fact_layer_and_fusion_weight_flags_parse() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("SHODH_DISABLE_FACT_LAYERS");
        std::env::remove_var("SHODH_GRAPH_FUSION_WEIGHT");
        assert!(!fact_layers_disabled());
        assert_eq!(graph_fusion_weight_override(), None);
        std::env::set_var("SHODH_DISABLE_FACT_LAYERS", "TRUE");
        std::env::set_var("SHODH_GRAPH_FUSION_WEIGHT", "0");
        assert!(fact_layers_disabled());
        assert_eq!(graph_fusion_weight_override(), Some(0.0));
        std::env::set_var("SHODH_GRAPH_FUSION_WEIGHT", "7");
        assert_eq!(graph_fusion_weight_override(), Some(1.0));
        std::env::set_var("SHODH_GRAPH_FUSION_WEIGHT", "NaN");
        assert_eq!(graph_fusion_weight_override(), None);
        std::env::remove_var("SHODH_DISABLE_FACT_LAYERS");
        std::env::remove_var("SHODH_GRAPH_FUSION_WEIGHT");
    }

    #[test]
    fn fusion_is_flat_unless_the_dial_or_the_upstream_flag_asks_for_rrf() {
        use crate::memory::dials::FusionMode;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("SHODH_FUSION_RRF");
        assert_eq!(fusion_mode(None), FusionMode::Flat);
        assert_eq!(fusion_mode(Some(FusionMode::Rrf)), FusionMode::Rrf);
        std::env::set_var("SHODH_FUSION_RRF", "true");
        assert_eq!(fusion_mode(None), FusionMode::Rrf);
        assert_eq!(fusion_mode(Some(FusionMode::Flat)), FusionMode::Flat);
        std::env::set_var("SHODH_FUSION_RRF", "0");
        assert_eq!(fusion_mode(None), FusionMode::Flat);
        std::env::remove_var("SHODH_FUSION_RRF");
    }

    #[test]
    fn scoring_now_honours_both_spellings_veld_first() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let parse = |s: &str| {
            chrono::DateTime::parse_from_rfc3339(s)
                .unwrap()
                .with_timezone(&chrono::Utc)
        };
        let prev_veld = std::env::var("VELD_EVAL_NOW").ok();
        let prev_shodh = std::env::var("SHODH_EVAL_NOW").ok();
        // The harness anchor, so a concurrently running harness test sees no change.
        let anchor = "2026-06-12T00:00:00Z";
        std::env::remove_var("VELD_EVAL_NOW");
        std::env::set_var("SHODH_EVAL_NOW", anchor);
        assert_eq!(crate::memory::scoring_now(), parse(anchor));
        std::env::set_var("VELD_EVAL_NOW", "2031-02-03T04:05:06Z");
        assert_eq!(crate::memory::scoring_now(), parse("2031-02-03T04:05:06Z"));
        std::env::set_var("VELD_EVAL_NOW", "not a time");
        assert_eq!(crate::memory::scoring_now(), parse(anchor));
        match prev_veld {
            Some(v) => std::env::set_var("VELD_EVAL_NOW", v),
            None => std::env::remove_var("VELD_EVAL_NOW"),
        }
        match prev_shodh {
            Some(v) => std::env::set_var("SHODH_EVAL_NOW", v),
            None => std::env::remove_var("SHODH_EVAL_NOW"),
        }
    }
}
