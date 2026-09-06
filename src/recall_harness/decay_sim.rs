//! Decay-simulation harness.
//!
//! Drives a single [`RelationshipEdge`] through simulated time and records its
//! strength trajectory. This is the measurement substrate for tuning the L2/L3
//! edge-decay model.
//!
//! # Why a cadence, not a single jump
//!
//! Production runs `apply_decay` on the heavy maintenance cycle (~every 6h),
//! and `RelationshipEdge::decay_at` resets `last_activated = now` on every call.
//! So the *per-cycle* elapsed time (~6h) is what `hybrid_decay_factor` sees, not
//! the edge's true age. Because `hybrid_decay_factor` only switches to its
//! heavy-tail power-law branch past `DECAY_CROSSOVER_DAYS` (3 days), the
//! power-law branch is effectively **dead** under periodic decay — chaining
//! 6-hourly exponential factors yields a pure exponential over the full age.
//!
//! A naive "age 30 days in one `decay_at` call" would feed `hybrid_decay_factor`
//! a 30-day elapsed, land directly in the power-law branch, and report the
//! *intended* curve — masking the bug. Faithful evaluation therefore must step
//! at the real cadence. [`simulate`] does exactly that; [`ideal_single_step`]
//! computes the model-intended value for comparison, and the gap between them is
//! the magnitude of the bug.

use chrono::{DateTime, Duration, Utc};

use crate::constants::LTP_MIN_STRENGTH;
use crate::graph_memory::{EdgeTier, LtpStatus, RelationshipEdge};

/// Production decay cadence: `apply_decay` runs on the heavy maintenance cycle,
/// roughly every 6 hours (see `handlers/state.rs`).
pub const PRODUCTION_CADENCE_HOURS: i64 = 6;

/// Specification of the edge to simulate.
#[derive(Debug, Clone, Copy)]
pub struct DecaySpec {
    pub tier: EdgeTier,
    pub initial_strength: f32,
    pub ltp_status: LtpStatus,
}

/// One sampled point on an edge's decay trajectory.
#[derive(Debug, Clone, Copy)]
pub struct DecayPoint {
    pub age_days: f64,
    pub strength: f32,
    /// `should_prune` returned by `decay_at` on this step.
    pub pruned_signal: bool,
}

/// Result of a cadenced decay simulation.
#[derive(Debug, Clone)]
pub struct DecayTrajectory {
    pub tier: EdgeTier,
    pub initial_strength: f32,
    pub cadence_hours: i64,
    /// One point per cadence step.
    pub points: Vec<DecayPoint>,
    /// Age (days) at which `decay_at` first signalled prune, if ever. For L2/L3
    /// this is expected to be `None` under cadence: the per-cycle `elapsed`
    /// (~6h) never exceeds the tier's `min_prune_hours` gate (30d/90d), so the
    /// edge's own decay never self-prunes — a second facet of the same
    /// per-cycle-vs-total-age confusion.
    pub pruned_at_days: Option<f64>,
    /// Age (days) at which strength first hit the `LTP_MIN_STRENGTH` floor.
    pub floored_at_days: Option<f64>,
}

impl DecayTrajectory {
    /// Final (oldest) strength sampled.
    pub fn final_strength(&self) -> f32 {
        self.points
            .last()
            .map(|p| p.strength)
            .unwrap_or(self.initial_strength)
    }
}

/// Fixed simulation origin (Unix epoch) so results are wall-clock independent.
fn origin() -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is a valid timestamp")
}

/// Drive an edge through `horizon_days` of simulated time in `cadence_hours`
/// steps, reproducing the production periodic-decay path via `decay_at`.
pub fn simulate(spec: DecaySpec, horizon_days: f64, cadence_hours: i64) -> DecayTrajectory {
    let cadence_hours = cadence_hours.max(1);
    let start = origin();
    let mut edge = RelationshipEdge::synthetic_for_sim(
        spec.initial_strength,
        spec.tier,
        spec.ltp_status,
        start,
    );

    let cadence = Duration::hours(cadence_hours);
    let total_steps = ((horizon_days * 24.0) / cadence_hours as f64)
        .ceil()
        .max(0.0) as usize;

    let mut points = Vec::with_capacity(total_steps);
    let mut pruned_at_days = None;
    let mut floored_at_days = None;
    let mut now = start;

    for _ in 0..total_steps {
        now += cadence;
        let pruned_signal = edge.decay_at(now);
        let age_days = (now - start).num_seconds() as f64 / 86_400.0;

        if pruned_signal && pruned_at_days.is_none() {
            pruned_at_days = Some(age_days);
        }
        if edge.strength <= LTP_MIN_STRENGTH && floored_at_days.is_none() {
            floored_at_days = Some(age_days);
        }
        points.push(DecayPoint {
            age_days,
            strength: edge.strength,
            pruned_signal,
        });
    }

    DecayTrajectory {
        tier: spec.tier,
        initial_strength: spec.initial_strength,
        cadence_hours,
        points,
        pruned_at_days,
        floored_at_days,
    }
}

/// The model-*intended* strength at `age_days`: a single `decay_at` jump, which
/// feeds the full age into `hybrid_decay_factor` and so exercises the power-law
/// branch. This is the curve the hybrid model is supposed to produce. Comparing
/// it with [`simulate`]'s cadenced result quantifies the periodic-decay bug.
///
/// Note: `decay_at` caps a single step's elapsed at 1 year (clock-jump guard),
/// so this is meaningful for `age_days <= 365`.
pub fn ideal_single_step(spec: DecaySpec, age_days: f64) -> f32 {
    let start = origin();
    let mut edge = RelationshipEdge::synthetic_for_sim(
        spec.initial_strength,
        spec.tier,
        spec.ltp_status,
        start,
    );
    let now = start + Duration::seconds((age_days * 86_400.0) as i64);
    edge.decay_at(now);
    edge.strength
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test used to document a bug: under the real ~6h cadence a
    /// potentiated L3 edge — the tier the model calls "near-permanent" — was
    /// crushed to the strength floor within weeks, far below what the hybrid
    /// model intends at the same age, because each `decay_at` call resets
    /// `last_activated` and so never feeds a long elapsed time into the
    /// power-law leg. Its docstring said "when the decay fix lands, the gap
    /// between cadenced and ideal should collapse".
    ///
    /// **It has landed, and the gap is gone for L3.** Giving L3 its own time
    /// scale (`decay::L3_TIME_SCALE_VS_L2` ≈ 0.0215) moves its crossover out to
    /// ~140 real days, so the whole first-30-days window is now in the
    /// *exponential* leg — and exponential decay is memoryless, i.e. cadence
    /// invariant: N steps of h compose exactly into one jump of N·h. The
    /// assertions below are inverted accordingly, and this is now the rig that
    /// keeps them collapsed.
    ///
    /// Derivation for the value asserted (L3, potentiated ⇒ λ = 0.693 × 0.5 =
    /// 0.3465/day, entirely within the exponential leg):
    ///   scaled age = 30 × 0.0215054 = 0.6451613 days
    ///   f          = exp(-0.3465 × 0.6451613) = exp(-0.2235483) = 0.7996772
    ///   strength   = 0.7 × 0.7996772 = 0.5597740
    #[test]
    fn cadenced_and_intended_l3_decay_now_agree() {
        let spec = DecaySpec {
            tier: EdgeTier::L3Semantic,
            initial_strength: 0.7,
            ltp_status: LtpStatus::Full,
        };

        let traj = simulate(spec, 30.0, PRODUCTION_CADENCE_HOURS);
        let cadenced_30d = traj.final_strength();
        let intended_30d = ideal_single_step(spec, 30.0);

        // The model intends a substantial strength at 30 days...
        assert!(
            intended_30d > 0.1,
            "intended strength at 30d should be substantial, got {intended_30d}"
        );
        // ...and the cadenced (production) path now delivers it.
        assert!(
            (cadenced_30d - 0.559_774).abs() < 1e-3,
            "cadenced L3 strength at 30d should be ~0.5598, got {cadenced_30d}"
        );
        // Cadence invariance: the two paths agree to within f32 accumulation
        // noise over 120 steps. This is the property the fix bought.
        assert!(
            (cadenced_30d - intended_30d).abs() < 1e-3,
            "cadenced {cadenced_30d} and intended {intended_30d} must now agree"
        );
        // And it never reaches the floor at all inside the window.
        assert!(
            traj.floored_at_days.is_none(),
            "a potentiated L3 edge must no longer floor within 30 days, floored at {:?}",
            traj.floored_at_days
        );
    }

    /// The cadence artefact itself is NOT fixed — it is only routed around for
    /// L3. L2 still crosses into the power-law leg at 3 days, so under the 6h
    /// production cadence it is still crushed relative to what the model
    /// intends. Pinning this keeps the remaining defect visible instead of
    /// letting the L3 fix read as "decay is solved".
    #[test]
    fn cadence_artefact_still_bites_l2() {
        let spec = DecaySpec {
            tier: EdgeTier::L2Episodic,
            initial_strength: 0.7,
            ltp_status: LtpStatus::Full,
        };

        let traj = simulate(spec, 30.0, PRODUCTION_CADENCE_HOURS);
        let cadenced_30d = traj.final_strength();
        let intended_30d = ideal_single_step(spec, 30.0);

        assert!(
            cadenced_30d <= LTP_MIN_STRENGTH * 1.5,
            "cadenced L2 strength at 30d is still at/near the floor, got {cadenced_30d}"
        );
        assert!(
            intended_30d > cadenced_30d * 5.0,
            "the L2 gap is still large: intended {intended_30d} vs cadenced {cadenced_30d}"
        );
    }

    /// Sanity: a single decay step DOES exercise the power-law branch (matches
    /// the existing `test_decay_tier_aware` expectation), confirming the bug is
    /// the cadence, not `hybrid_decay_factor` itself.
    #[test]
    fn single_step_exercises_power_law_branch() {
        let spec = DecaySpec {
            tier: EdgeTier::L2Episodic,
            initial_strength: 1.0,
            ltp_status: LtpStatus::None,
        };
        // 7 days, one jump: value_at_crossover (~0.125) * (7/3)^-0.5 (~0.655) ~= 0.082.
        let s = ideal_single_step(spec, 7.0);
        assert!(
            s > 0.05 && s < 0.15,
            "single-step 7d power-law strength ~0.082, got {s}"
        );
    }

    /// The trajectory is monotonically non-increasing in strength (decay only
    /// reduces; reinforcement is not exercised here).
    #[test]
    fn trajectory_is_monotonic_non_increasing() {
        let spec = DecaySpec {
            tier: EdgeTier::L2Episodic,
            initial_strength: 0.6,
            ltp_status: LtpStatus::None,
        };
        let traj = simulate(spec, 10.0, PRODUCTION_CADENCE_HOURS);
        assert!(!traj.points.is_empty());
        let mut prev = spec.initial_strength;
        for p in &traj.points {
            assert!(
                p.strength <= prev + f32::EPSILON,
                "strength must not increase: {} then {}",
                prev,
                p.strength
            );
            prev = p.strength;
        }
    }
}
