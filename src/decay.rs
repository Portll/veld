//! Hybrid Decay Model (SHO-103)
//!
//! Implements biologically-accurate memory decay based on neuroscience research.
//!
//! # The Problem with Pure Exponential Decay
//!
//! Traditional memory systems use exponential decay: `w(t) = w₀ × e^(-λt)`
//!
//! This produces a "cliff" effect where memories drop rapidly and then flatten:
//! - Day 1: 100% → 95%
//! - Day 7: 95% → 70%
//! - Day 30: 70% → 15% (steep cliff)
//!
//! # The Solution: Hybrid Decay
//!
//! Human memory follows a power-law for long-term retention, not exponential.
//!
//! This module implements a hybrid model:
//! - **Consolidation phase** (t < 3 days): Exponential decay
//!   - Fast filtering of noise and weak associations
//!   - Matches short-term synaptic depression
//! - **Long-term phase** (t ≥ 3 days): Power-law decay with log-periodic correction
//!   - Heavy tail preserves important memories longer
//!   - Matches empirical human forgetting curves
//!   - Log-periodic modulation creates resonance at self-similar temporal scales
//!   - Memories accessed at weekly/monthly/yearly rhythms resist decay at those scales
//!
//! ```text
//!         Exponential              Power-Law + Log-Periodic
//!         (consolidation)          (fractal long-term retention)
//!
//! Strength │ ╲
//!     100% │  ╲
//!          │   ╲
//!      60% │    ╲___         ← resonance bumps at self-similar scales
//!          │        ╲~╲__~╲___
//!      30% │             ╲________
//!          │                      ╲~~~╲___________
//!       5% │─────────────────────────────────────────
//!          └────┬────────┬──────┬──────────────────► Time
//!               │        │      │
//!            t_cross   7 days  30 days
//!          (3 days)   (λ₁)    (λ₂)
//! ```
//!
//! # References
//!
//! - Wixted & Ebbesen (1991) "On the Form of Forgetting"
//! - Wixted (2004) "The psychology and neuroscience of forgetting"
//! - Anderson & Schooler (1991) "Reflections of the Environment in Memory"
//! - Sornette (2003) "Why Stock Markets Crash" Ch.5 (discrete scale invariance)

use crate::constants::{
    ANCHOR_IMPORTANCE_FLOOR, DECAY_CROSSOVER_DAYS, DECAY_LAMBDA_CONSOLIDATION,
    LOG_PERIODIC_BETA, LOG_PERIODIC_SCALES, POWERLAW_BETA, POWERLAW_BETA_POTENTIATED,
};

/// Computes the log-periodic fractal correction factor.
///
/// Models discrete scale invariance in human temporal access patterns.
/// The correction oscillates in log-time at preferred scaling ratios,
/// creating resonance zones where decay slows at self-similar intervals.
///
/// Formula: 1 + β Σₖ cos(2π log(t) / log(λₖ))
///
/// Resonances occur at t = λₖⁿ days (e.g., 7, 49, 343... for weekly scale).
///
/// # Arguments
///
/// * `days` - Raw elapsed time in days (not scaled)
/// * `crossover_days` - Crossover point; correction blends in smoothly from here
/// * `beta` - Modulation amplitude (LOG_PERIODIC_BETA)
/// * `scales` - Preferred scaling ratios in days (LOG_PERIODIC_SCALES)
///
/// # Returns
///
/// Multiplicative correction factor with smooth blend-in from crossover.
/// Always > 0 when β × len(scales) < 1.
#[inline]
fn log_periodic_correction(days: f64, crossover_days: f64, beta: f64, scales: &[f64]) -> f64 {
    // Blend factor: ramps from 0.0 at crossover to 1.0 at 2×crossover
    // Ensures continuity at the exponential→power-law transition
    let blend = ((days - crossover_days) / crossover_days).clamp(0.0, 1.0);

    let log_t = days.ln();
    let sum: f64 = scales
        .iter()
        .map(|lambda_k| {
            let period = lambda_k.ln();
            (std::f64::consts::TAU * log_t / period).cos()
        })
        .sum();
    let raw_correction = 1.0 + beta * sum;

    // Blend: at crossover → 1.0 (pure power-law), at 2×crossover → full correction
    1.0 + blend * (raw_correction - 1.0)
}

/// Calculates the hybrid decay factor for a given elapsed time.
///
/// Returns a value between 0.0 and 1.0 representing the retention ratio.
///
/// # Arguments
///
/// * `days_elapsed` - Time since last activation in days
/// * `potentiated` - Whether this is a potentiated/important memory (uses slower decay)
///
/// # Returns
///
/// Decay factor to multiply with original strength: `new_strength = old_strength * decay_factor`
///
/// # Example
///
/// ```ignore
/// let factor = hybrid_decay_factor(7.0, false);
/// let new_strength = old_strength * factor;
/// ```
#[inline]
pub fn hybrid_decay_factor(days_elapsed: f64, potentiated: bool) -> f32 {
    if days_elapsed <= 0.0 {
        return 1.0;
    }

    let beta = if potentiated {
        POWERLAW_BETA_POTENTIATED
    } else {
        POWERLAW_BETA
    };

    // Exponential rate for consolidation phase
    // Potentiated memories use slower exponential decay too
    let lambda = if potentiated {
        DECAY_LAMBDA_CONSOLIDATION * 0.5 // Half the rate for potentiated
    } else {
        DECAY_LAMBDA_CONSOLIDATION
    };

    if days_elapsed < DECAY_CROSSOVER_DAYS {
        // Consolidation phase: exponential decay
        // w(t) = w₀ × e^(-λt)
        (-lambda * days_elapsed).exp() as f32
    } else {
        // Long-term phase: power-law decay with log-periodic fractal correction
        // First, calculate what value we'd have at crossover with exponential
        let value_at_crossover = (-lambda * DECAY_CROSSOVER_DAYS).exp();

        // Scaled time for power-law
        let t_scaled = days_elapsed / DECAY_CROSSOVER_DAYS;

        // Base power-law: (t / t_cross)^(-β)
        let power_law_factor = t_scaled.powf(-beta);

        // Log-periodic correction: discrete scale invariance
        // w(t) = t^(-α) × (1 + β Σₖ cos(2π log(t) / log(λₖ)))
        // Creates resonance at weekly/monthly/yearly rhythms in log-time
        let fractal_correction = log_periodic_correction(
            days_elapsed,
            DECAY_CROSSOVER_DAYS,
            LOG_PERIODIC_BETA,
            &LOG_PERIODIC_SCALES,
        );

        (value_at_crossover * power_law_factor * fractal_correction) as f32
    }
}

/// Calculates the hybrid decay factor with custom parameters.
///
/// Use this for contexts that need different decay characteristics.
///
/// # Arguments
///
/// * `days_elapsed` - Time since last activation in days
/// * `crossover_days` - Days before switching from exponential to power-law
/// * `lambda` - Exponential decay rate for consolidation phase
/// * `beta` - Power-law exponent for long-term phase
///
/// # Example
///
/// ```ignore
/// // Faster decay for edge weights
/// let factor = hybrid_decay_factor_custom(days_elapsed, 1.0, 1.0, 0.6);
/// ```
#[inline]
pub fn hybrid_decay_factor_custom(
    days_elapsed: f64,
    crossover_days: f64,
    lambda: f64,
    beta: f64,
) -> f32 {
    if days_elapsed <= 0.0 {
        return 1.0;
    }

    if days_elapsed < crossover_days {
        // Consolidation phase: exponential decay
        (-lambda * days_elapsed).exp() as f32
    } else {
        // Long-term phase: power-law decay with log-periodic fractal correction
        let value_at_crossover = (-lambda * crossover_days).exp();
        let t_scaled = days_elapsed / crossover_days;
        let power_law_factor = t_scaled.powf(-beta);
        let fractal_correction = log_periodic_correction(
            days_elapsed,
            crossover_days,
            LOG_PERIODIC_BETA,
            &LOG_PERIODIC_SCALES,
        );
        (value_at_crossover * power_law_factor * fractal_correction) as f32
    }
}

/// Calculates retention percentage for debugging/visualization.
///
/// Returns a human-readable percentage string showing retention at various time points.
#[allow(dead_code)]
pub fn retention_curve_debug(potentiated: bool) -> String {
    let days = [0.5, 1.0, 3.0, 7.0, 14.0, 30.0, 90.0, 365.0];
    let mode = if potentiated { "potentiated" } else { "normal" };

    let mut output = format!("Retention curve ({mode}):\n");
    for d in days {
        let factor = hybrid_decay_factor(d, potentiated);
        output.push_str(&format!("  Day {:>5.1}: {:>6.2}%\n", d, factor * 100.0));
    }
    output
}

/// Tier-aware decay factor for edge consolidation (3-tier memory model)
///
/// Each tier has different decay characteristics based on hippocampal-cortical research:
/// - L1 (Working): ~2.9%/hour decay (λ=0.029), max 48 hours
/// - L2 (Episodic): ~3.1%/day decay (λ=0.031), max 30 days
/// - L3 (Semantic): ~2%/month decay (λ=0.02/720h), near-permanent
///
/// # Arguments
///
/// * `hours_elapsed` - Time since last activation in hours
/// * `tier` - Memory tier (0=L1, 1=L2, 2=L3)
/// * `ltp_decay_factor` - LTP decay protection factor (1.0=none, 0.5=2x slower, 0.1=10x slower)
///
/// # Returns
///
/// Decay factor (0.0-1.0) and whether edge should be pruned
///
/// # PIPE-4 Update
///
/// Changed from `potentiated: bool` to `ltp_decay_factor: f32` to support
/// multi-scale LTP with graduated protection levels:
/// - LtpStatus::None → 1.0 (no protection)
/// - LtpStatus::Burst → 0.5 (2x slower decay, temporary)
/// - LtpStatus::Weekly → 0.3 (3x slower decay, moderate)
/// - LtpStatus::Full → 0.1 (10x slower decay, maximum)
#[inline]
pub fn tier_decay_factor(hours_elapsed: f64, tier: u8, ltp_decay_factor: f32) -> (f32, bool) {
    use crate::constants::*;

    if hours_elapsed <= 0.0 {
        return (1.0, false);
    }

    let (decay_rate, max_age_hours, prune_threshold) = match tier {
        0 => {
            // L1 Working: ~2.9%/hour decay (λ=0.029), max 48 hours
            (
                L1_DECAY_PER_HOUR as f64,
                (L1_MAX_AGE_HOURS as f64),
                L1_PRUNE_THRESHOLD,
            )
        }
        1 => {
            // L2 Episodic: ~3.1%/day decay (λ=0.031), max 30 days
            let decay_per_hour = L2_DECAY_PER_DAY as f64 / 24.0;
            (
                decay_per_hour,
                (L2_MAX_AGE_DAYS as f64) * 24.0,
                L2_PRUNE_THRESHOLD,
            )
        }
        _ => {
            // L3 Semantic (tier 2+): 2%/month decay, near-permanent
            let decay_per_hour = L3_DECAY_PER_MONTH as f64 / (30.0 * 24.0);
            // Max age: effectively unlimited (10 years)
            (decay_per_hour, 87600.0, L3_PRUNE_THRESHOLD)
        }
    };

    // PIPE-4: Apply graduated LTP protection
    // ltp_decay_factor of 0.5 = 2x slower, 0.1 = 10x slower, 1.0 = no protection
    let effective_rate = decay_rate * ltp_decay_factor as f64;

    // Exponential decay: w(t) = w₀ × e^(-λt)
    let decay_factor = (-effective_rate * hours_elapsed).exp() as f32;

    // Check if edge exceeded max age (should prune)
    // PIPE-4: Potentiated edges (ltp_decay_factor < 1.0) extend max age proportionally
    let effective_max_age = if ltp_decay_factor < 1.0 {
        max_age_hours / ltp_decay_factor as f64
    } else {
        max_age_hours
    };
    let should_prune = hours_elapsed > effective_max_age && decay_factor < prune_threshold;

    (decay_factor.max(0.001), should_prune)
}

/// Apply hybrid decay to an importance value, respecting anchor status.
///
/// Anchored memories cannot decay below `ANCHOR_IMPORTANCE_FLOOR`.
/// This provides a floor that prevents critical user-marked facts from
/// fading into irrelevance while still allowing normal ranking dynamics.
#[inline]
pub fn apply_decay_with_anchor(
    current_importance: f32,
    days_elapsed: f64,
    potentiated: bool,
    anchored: bool,
) -> f32 {
    let decay_factor = hybrid_decay_factor(days_elapsed, potentiated);
    let decayed = current_importance * decay_factor;
    if anchored {
        decayed.max(ANCHOR_IMPORTANCE_FLOOR)
    } else {
        decayed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decay_at_zero() {
        assert_eq!(hybrid_decay_factor(0.0, false), 1.0);
        assert_eq!(hybrid_decay_factor(-1.0, false), 1.0);
    }

    #[test]
    fn test_exponential_phase() {
        // During consolidation (< 3 days), should be exponential
        let factor_1day = hybrid_decay_factor(1.0, false);
        let factor_2day = hybrid_decay_factor(2.0, false);

        // Exponential property: ratio should be constant
        let ratio_1_to_2 = factor_2day / factor_1day;
        let expected_ratio = (-DECAY_LAMBDA_CONSOLIDATION).exp() as f32;

        assert!((ratio_1_to_2 - expected_ratio).abs() < 0.01);
    }

    #[test]
    fn test_powerlaw_phase() {
        // After crossover (> 3 days), should follow power-law envelope
        // with log-periodic modulation
        let factor_7day = hybrid_decay_factor(7.0, false);
        let factor_14day = hybrid_decay_factor(14.0, false);

        // Ratio is power-law × correction ratio
        // The base power-law ratio for doubling is 2^(-β) ≈ 0.707
        // Log-periodic correction shifts this — verify within wider tolerance
        let ratio = factor_14day / factor_7day;
        let base_ratio = 2.0_f64.powf(-POWERLAW_BETA) as f32;

        // With β_lp=0.15 and 3 scales, max deviation is ±0.45 of the correction
        assert!((ratio - base_ratio).abs() < 0.15);
    }

    #[test]
    fn test_continuity_at_crossover() {
        // Values just before and after crossover should be close
        let just_before = hybrid_decay_factor(DECAY_CROSSOVER_DAYS - 0.001, false);
        let just_after = hybrid_decay_factor(DECAY_CROSSOVER_DAYS + 0.001, false);

        assert!((just_before - just_after).abs() < 0.01);
    }

    #[test]
    fn test_potentiated_decays_slower() {
        let normal = hybrid_decay_factor(30.0, false);
        let potentiated = hybrid_decay_factor(30.0, true);

        // Potentiated should retain more
        assert!(potentiated > normal);
    }

    #[test]
    fn test_heavy_tail_retention() {
        // Key property: power-law has heavy tail
        // At 365 days, we should still have meaningful retention
        let year_retention = hybrid_decay_factor(365.0, false);
        let year_retention_potentiated = hybrid_decay_factor(365.0, true);

        // Normal: should be > 1%
        assert!(year_retention > 0.01);
        // Potentiated: should be > 5%
        assert!(year_retention_potentiated > 0.05);
    }

    #[test]
    fn test_custom_parameters() {
        // Test custom function with aggressive decay
        let aggressive = hybrid_decay_factor_custom(7.0, 1.0, 1.5, 0.7);
        let normal = hybrid_decay_factor(7.0, false);

        assert!(aggressive < normal);
    }

    #[test]
    fn test_tier_decay_factor_l1_with_and_without_ltp() {
        let (unprotected, _) = tier_decay_factor(24.0, 0, 1.0);
        let (protected, _) = tier_decay_factor(24.0, 0, 0.5);
        assert!(protected > unprotected);
    }

    #[test]
    fn test_tier_decay_factor_l1_prune_threshold() {
        let (factor_at_max_age, should_prune_at_max_age) = tier_decay_factor(48.0, 0, 1.0);
        // At max age boundary, L1 should still not prune because pruning requires "greater than" max age.
        assert!(factor_at_max_age > 0.1);
        assert!(!should_prune_at_max_age);

        let (factor_past_max_age, should_prune_past_max_age) = tier_decay_factor(96.0, 0, 1.0);
        assert!(factor_past_max_age < 0.1);
        assert!(should_prune_past_max_age);
    }

    #[test]
    fn test_tier_decay_factor_l3_long_tail() {
        let (factor_1y, prune_1y) = tier_decay_factor(365.0 * 24.0, 2, 1.0);
        assert!(factor_1y > 0.7);
        assert!(!prune_1y);

        let (factor_3y, prune_3y) = tier_decay_factor(3.0 * 365.0 * 24.0, 2, 1.0);
        assert!(factor_3y > 0.45);
        assert!(!prune_3y);
    }

    #[test]
    fn test_tier_decay_zero_and_negative_elapsed() {
        let (zero_factor, zero_prune) = tier_decay_factor(0.0, 1, 1.0);
        assert_eq!(zero_factor, 1.0);
        assert!(!zero_prune);

        let (neg_factor, neg_prune) = tier_decay_factor(-10.0, 1, 1.0);
        assert_eq!(neg_factor, 1.0);
        assert!(!neg_prune);
    }

    #[test]
    fn test_tier_decay_invalid_tier_defaults_to_l3() {
        let (invalid_tier, _) = tier_decay_factor(24.0, 9, 1.0);
        let (l3, _) = tier_decay_factor(24.0, 2, 1.0);
        assert_eq!(invalid_tier, l3);
    }

    #[test]
    fn test_anchored_memory_floor() {
        // Anchored memory should not decay below ANCHOR_IMPORTANCE_FLOOR
        let result = apply_decay_with_anchor(0.8, 365.0, false, true);
        assert!(result >= ANCHOR_IMPORTANCE_FLOOR);

        // Non-anchored memory can decay freely
        let result_unanchored = apply_decay_with_anchor(0.8, 365.0, false, false);
        assert!(result_unanchored < ANCHOR_IMPORTANCE_FLOOR);
    }

    #[test]
    fn test_anchored_memory_still_decays_above_floor() {
        // Anchored memory should still decay when above the floor
        let result = apply_decay_with_anchor(0.9, 7.0, false, true);
        assert!(result < 0.9); // Some decay happened
        assert!(result >= ANCHOR_IMPORTANCE_FLOOR); // But floored
    }

    #[test]
    fn test_log_periodic_correction_at_crossover() {
        // At crossover, blend=0 so correction must be exactly 1.0 (continuity)
        let correction = log_periodic_correction(
            DECAY_CROSSOVER_DAYS,
            DECAY_CROSSOVER_DAYS,
            LOG_PERIODIC_BETA,
            &LOG_PERIODIC_SCALES,
        );
        assert!(
            (correction - 1.0).abs() < 1e-10,
            "Correction at crossover must be 1.0 for continuity, got {correction}"
        );
    }

    #[test]
    fn test_log_periodic_correction_bounded() {
        // Correction must always be > 0 for stability
        for day in [4, 7, 14, 21, 30, 60, 90, 180, 365, 730] {
            let correction = log_periodic_correction(
                day as f64,
                DECAY_CROSSOVER_DAYS,
                LOG_PERIODIC_BETA,
                &LOG_PERIODIC_SCALES,
            );
            assert!(
                correction > 0.0,
                "Correction went negative at day {day}: {correction}"
            );
        }
    }

    #[test]
    fn test_log_periodic_creates_resonance_at_weekly_scale() {
        // At t = 7 days (λ₁), ln(7)/ln(7) = 1.0, so cos(2π)=1.0 → weekly resonance
        // At t = 49 days (λ₁²), ln(49)/ln(7) = 2.0, so cos(4π)=1.0 → resonance again
        let at_7d = hybrid_decay_factor(7.0, false);
        let at_49d = hybrid_decay_factor(49.0, false);

        // Compare against pure power-law at same points
        let pure_7d = {
            let v = (-DECAY_LAMBDA_CONSOLIDATION * DECAY_CROSSOVER_DAYS).exp();
            (v * (7.0 / DECAY_CROSSOVER_DAYS).powf(-POWERLAW_BETA)) as f32
        };
        let pure_49d = {
            let v = (-DECAY_LAMBDA_CONSOLIDATION * DECAY_CROSSOVER_DAYS).exp();
            (v * (49.0 / DECAY_CROSSOVER_DAYS).powf(-POWERLAW_BETA)) as f32
        };

        // Fractal-corrected values should differ from pure power-law
        assert!(
            (at_7d - pure_7d).abs() > 0.001,
            "Correction should be measurable at 7 days"
        );
        assert!(
            (at_49d - pure_49d).abs() > 0.001,
            "Correction should be measurable at 49 days"
        );
        assert!(at_7d > 0.0 && at_7d < 1.0);
        assert!(at_49d > 0.0 && at_49d < 1.0);
    }

    #[test]
    fn test_log_periodic_with_zero_beta_is_pure_powerlaw() {
        // When β=0, correction is exactly 1.0 everywhere → pure power-law
        for day in [7.0, 30.0, 90.0, 365.0] {
            let correction = log_periodic_correction(
                day,
                DECAY_CROSSOVER_DAYS,
                0.0,
                &LOG_PERIODIC_SCALES,
            );
            assert!((correction - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_fractal_decay_envelope_decreases() {
        // The log-periodic correction creates oscillations (that's the point),
        // but the power-law envelope should still dominate over long spans.
        // Verify: average retention in [3,180] > average in [180,365] > average in [365,730].
        let avg = |start: i32, end: i32| -> f32 {
            let mut sum = 0.0f32;
            for d in start..end {
                sum += hybrid_decay_factor(d as f64, false);
            }
            sum / (end - start) as f32
        };

        let avg_early = avg(3, 180);
        let avg_mid = avg(180, 365);
        let avg_late = avg(365, 730);

        assert!(
            avg_early > avg_mid,
            "Early avg ({avg_early}) should exceed mid ({avg_mid})"
        );
        assert!(
            avg_mid > avg_late,
            "Mid avg ({avg_mid}) should exceed late ({avg_late})"
        );
    }
}
