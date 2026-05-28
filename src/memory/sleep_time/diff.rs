//! Lightweight diff classification for sleep-time rewrites.
//!
//! We do not run a full LCS / Myers diff — for the gating decisions sleep-time
//! makes (Noop / Minor / Substantive / Massive) a character-level retain /
//! shrink heuristic is sufficient and is O(n) on input length.
//!
//! Used by:
//!   - R29 output validation: reject [`DiffClass::Massive`] unless the rewriter
//!     explicitly declared a shrink intent (V1 always rejects).
//!   - Telemetry / consolidation events: every `SleepTimeBlockRewritten`
//!     event carries a [`DiffSummary`].

use super::types::{DiffClass, DiffSummary};

/// Char-count thresholds (R29).
///
/// `Massive` is reserved for true *shrink-destruction* — large reduction in
/// size, indicating the rewriter may have collapsed the block. Same-length
/// rewrites with different wording are `Substantive`, which is the expected
/// shape of a legitimate consolidation pass.
const MINOR_CHANGE_RATIO: f32 = 0.10;
const MASSIVE_SHRINK_RATIO: f32 = 0.70;

/// Classify the diff between `prior` and `new`.
///
/// `retained_ratio` approximates the fraction of *prior* characters that
/// remain (longest common token-prefix + suffix); `shrink_ratio` is the
/// fraction of size reduction.
pub fn classify(prior: &str, new: &str) -> DiffSummary {
    let prior_chars = prior.chars().count();
    let new_chars = new.chars().count();

    if prior_chars == 0 && new_chars == 0 {
        return DiffSummary {
            class: DiffClass::Noop,
            prior_chars: 0,
            new_chars: 0,
            retained_ratio: 1.0,
            shrink_ratio: 0.0,
        };
    }

    let retained = retained_chars_estimate(prior, new);
    let retained_ratio = if prior_chars == 0 {
        0.0
    } else {
        (retained as f32 / prior_chars as f32).min(1.0)
    };

    let shrink_ratio = if prior_chars == 0 || new_chars >= prior_chars {
        0.0
    } else {
        ((prior_chars - new_chars) as f32 / prior_chars as f32).clamp(0.0, 1.0)
    };

    let class = classify_ratios(prior_chars, new_chars, retained_ratio, shrink_ratio);

    DiffSummary {
        class,
        prior_chars,
        new_chars,
        retained_ratio,
        shrink_ratio,
    }
}

fn classify_ratios(
    prior_chars: usize,
    new_chars: usize,
    retained_ratio: f32,
    shrink_ratio: f32,
) -> DiffClass {
    // Massive: large shrink only (destruction signal — R29 rejects unless
    // explicit shrink intent). Same-length wording changes are NOT Massive.
    if shrink_ratio >= MASSIVE_SHRINK_RATIO {
        return DiffClass::Massive;
    }

    // Noop: identical character counts AND ~all retained.
    if prior_chars == new_chars && retained_ratio >= 0.999 {
        return DiffClass::Noop;
    }

    // Minor: small character-count delta and high retention.
    let delta = (new_chars as isize - prior_chars as isize).unsigned_abs();
    let delta_ratio = if prior_chars == 0 {
        1.0
    } else {
        delta as f32 / prior_chars as f32
    };
    if delta_ratio < MINOR_CHANGE_RATIO && retained_ratio >= 0.80 {
        return DiffClass::Minor;
    }

    DiffClass::Substantive
}

/// Estimate of retained characters: longest common prefix + suffix in the
/// minimum of the two lengths. Cheap O(n) approximation; not exact LCS but
/// sufficient for gating heuristics.
fn retained_chars_estimate(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let n = av.len().min(bv.len());
    if n == 0 {
        return 0;
    }

    let mut prefix = 0usize;
    while prefix < n && av[prefix] == bv[prefix] {
        prefix += 1;
    }

    let mut suffix = 0usize;
    while suffix < n - prefix
        && av[av.len() - 1 - suffix] == bv[bv.len() - 1 - suffix]
    {
        suffix += 1;
    }

    prefix + suffix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_noop() {
        let s = "You are a helpful assistant.";
        let d = classify(s, s);
        assert_eq!(d.class, DiffClass::Noop);
        assert_eq!(d.retained_ratio, 1.0);
        assert_eq!(d.shrink_ratio, 0.0);
    }

    #[test]
    fn small_edit_is_minor() {
        let a = "You are a helpful assistant who explains things clearly.";
        let b = "You are a helpful assistant who explains things very clearly.";
        let d = classify(a, b);
        assert_eq!(d.class, DiffClass::Minor);
    }

    #[test]
    fn complete_replacement_is_substantive() {
        // Different wording at similar length: legitimate substantive rewrite,
        // NOT destruction. R29 should not reject this kind of rewrite.
        let a = "Persona block: friendly, terse, technical.";
        let b = "Different content entirely with no shared structure at all.";
        let d = classify(a, b);
        assert_eq!(d.class, DiffClass::Substantive);
    }

    #[test]
    fn empty_to_content_is_substantive() {
        // A new block being authored from nothing is Substantive, not Massive.
        // (Massive is reserved for shrink-destruction — see classifier docs.)
        let d = classify("", "Something new");
        assert_eq!(d.class, DiffClass::Substantive);
    }

    #[test]
    fn both_empty_is_noop() {
        let d = classify("", "");
        assert_eq!(d.class, DiffClass::Noop);
    }

    #[test]
    fn large_shrink_is_massive() {
        let a = "a".repeat(1000);
        let b = "a".repeat(100);
        let d = classify(&a, &b);
        // shrink_ratio = 0.9 → massive
        assert_eq!(d.class, DiffClass::Massive);
    }

    #[test]
    fn small_addition_is_minor() {
        let a = "a".repeat(1000);
        let b = format!("{}{}", a, "extra");
        let d = classify(&a, &b);
        assert_eq!(d.class, DiffClass::Minor);
    }

    #[test]
    fn moderate_rewrite_is_substantive() {
        let a = "The user prefers concise explanations and short examples.";
        let b = "The user wants brief commentary with illustrative snippets.";
        let d = classify(a, b);
        assert_eq!(d.class, DiffClass::Substantive);
    }
}
