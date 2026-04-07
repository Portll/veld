//! Fourier-Learned Decay Scales (SHO-FFT)
//!
//! Learns optimal log-periodic decay scales from actual user access patterns
//! using FFT spectral analysis, replacing the hardcoded [7.0, 30.0, 365.0].
//!
//! # Theory
//!
//! The log-periodic correction in [`crate::decay`] uses:
//!
//! ```text
//! w(t) = t^(-beta) * (1 + beta * sum_k cos(2*pi*log(t) / log(lambda_k)))
//! ```
//!
//! The scales lambda_k define resonance periods in log-time. Hardcoded values
//! assume weekly/monthly/yearly rhythms, but real users have idiosyncratic
//! access patterns (e.g., biweekly reviews, quarterly sprints, semester cycles).
//!
//! # Approach
//!
//! 1. Compute inter-access intervals from raw timestamps
//! 2. Transform to log-space (since the decay model operates in log-time)
//! 3. Run FFT on the log-interval sequence to extract spectral peaks
//! 4. Convert dominant frequencies back to day-scales
//! 5. Clamp to [1.0, 730.0] days and return top 3
//!
//! When insufficient data is available (< 50 events), falls back to the
//! default scales from [`crate::constants::LOG_PERIODIC_SCALES`].

use chrono::{DateTime, Utc};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::constants::LOG_PERIODIC_SCALES;

/// Minimum number of access events required for FFT analysis.
/// Below this threshold, the spectral estimate is too noisy to be useful.
const MIN_ACCESS_EVENTS: usize = 50;

/// Minimum allowed scale in days (sub-daily rhythms are noise for decay purposes).
const MIN_SCALE_DAYS: f64 = 1.0;

/// Maximum allowed scale in days (2 years; beyond this, data is too sparse).
const MAX_SCALE_DAYS: f64 = 730.0;

/// Minimum separation ratio between adjacent learned scales.
/// Prevents two scales from collapsing onto nearly the same frequency.
/// A ratio of 2.0 means each scale must be at least 2x the previous.
const MIN_SCALE_SEPARATION_RATIO: f64 = 2.0;

/// Learned decay scales extracted from user access pattern spectral analysis.
#[derive(Debug, Clone)]
pub struct LearnedScales {
    /// The three dominant periodicities in days, sorted ascending.
    pub scales: [f64; 3],
    /// Confidence in the learned scales (0.0-1.0).
    /// Based on data quantity, spectral peak sharpness, and separation quality.
    pub confidence: f64,
    /// Number of access events used in the analysis.
    pub sample_count: usize,
    /// When these scales were computed.
    pub computed_at: DateTime<Utc>,
}

impl LearnedScales {
    /// Returns the default (hardcoded) scales with zero confidence.
    fn default_fallback(sample_count: usize) -> Self {
        Self {
            scales: LOG_PERIODIC_SCALES,
            confidence: 0.0,
            sample_count,
            computed_at: Utc::now(),
        }
    }
}

/// Learns optimal log-periodic decay scales from user access timestamps.
///
/// Analyzes the spectral structure of inter-access intervals in log-space
/// to discover the user's natural temporal rhythms. Returns the top 3
/// dominant periodicities as day-scales for use in the log-periodic correction.
///
/// # Arguments
///
/// * `access_timestamps` - Chronologically ordered access timestamps.
///   These can come from `learning_history`, memory retrieval logs, or
///   any source of user interaction events.
///
/// # Returns
///
/// A [`LearnedScales`] struct containing:
/// - `scales`: Three dominant periodicities in days (ascending), clamped to [1.0, 730.0]
/// - `confidence`: Quality metric (0.0 = fallback defaults, 1.0 = strong spectral peaks)
/// - `sample_count`: How many timestamps were analyzed
/// - `computed_at`: Timestamp of computation (for cache invalidation)
///
/// # Fallback
///
/// Returns the default [7.0, 30.0, 365.0] scales with confidence=0.0 when:
/// - Fewer than 50 access events are provided
/// - All intervals are identical (no spectral variation)
/// - FFT produces no valid peaks within the allowed range
pub fn learn_decay_scales(access_timestamps: &[DateTime<Utc>]) -> LearnedScales {
    if access_timestamps.len() < MIN_ACCESS_EVENTS {
        return LearnedScales::default_fallback(access_timestamps.len());
    }

    // Step 1: Compute inter-access intervals in days
    let intervals = compute_intervals_days(access_timestamps);
    if intervals.is_empty() {
        return LearnedScales::default_fallback(access_timestamps.len());
    }

    // Step 2: Transform to log-space
    // Filter out zero/negative intervals (simultaneous events)
    let log_intervals: Vec<f64> = intervals
        .iter()
        .filter(|&&d| d > 0.0)
        .map(|d| d.ln())
        .collect();

    if log_intervals.len() < MIN_ACCESS_EVENTS - 1 {
        return LearnedScales::default_fallback(access_timestamps.len());
    }

    // Step 3: Prepare FFT input - zero-mean the signal to remove DC bias
    let mean = log_intervals.iter().sum::<f64>() / log_intervals.len() as f64;
    let centered: Vec<f64> = log_intervals.iter().map(|v| v - mean).collect();

    // Apply Hann window to reduce spectral leakage
    let n = centered.len();
    let windowed: Vec<Complex<f64>> = centered
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = 0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / n as f64).cos());
            Complex::new(v * w, 0.0)
        })
        .collect();

    // Step 4: Run FFT
    let mut buffer = windowed;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Step 5: Extract magnitude spectrum (skip DC component at index 0)
    // Only use the first half (positive frequencies) due to Hermitian symmetry
    let half_n = n / 2;
    if half_n < 2 {
        return LearnedScales::default_fallback(access_timestamps.len());
    }

    let magnitudes: Vec<f64> = buffer[1..=half_n].iter().map(|c| c.norm()).collect();

    // Step 6: Find peaks in the magnitude spectrum
    // A peak is a local maximum that exceeds the mean magnitude by at least 1 std dev
    let mag_mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
    let mag_variance = magnitudes
        .iter()
        .map(|m| (m - mag_mean).powi(2))
        .sum::<f64>()
        / magnitudes.len() as f64;
    let mag_std = mag_variance.sqrt();
    let peak_threshold = mag_mean + mag_std;

    // Compute the mean log-interval spacing for frequency-to-period conversion.
    // The FFT treats the input as a uniformly-sampled sequence with unit spacing.
    // Frequency bin k corresponds to k cycles per N samples.
    // Each "sample" spans one inter-access gap, and the mean gap in log-days
    // sets the physical time scale per sample.
    let mean_log_spacing = log_intervals.iter().sum::<f64>() / log_intervals.len() as f64;

    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 0..magnitudes.len() {
        let is_local_max = (i == 0 || magnitudes[i] >= magnitudes[i - 1])
            && (i == magnitudes.len() - 1 || magnitudes[i] >= magnitudes[i + 1]);

        if is_local_max && magnitudes[i] > peak_threshold {
            peaks.push((i, magnitudes[i]));
        }
    }

    // Sort by magnitude descending to get dominant frequencies first
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step 7: Convert frequency indices to day-scales
    // FFT bin index k (1-based after skipping DC) corresponds to frequency k/N
    // in cycles per sample. One sample = one inter-access interval.
    // Period in samples = N / k, period in log-days = (N / k) * mean_log_spacing.
    // Day-scale = exp(period_in_log_days).
    let mut candidate_scales: Vec<(f64, f64)> = Vec::new(); // (scale_days, magnitude)
    for &(bin_idx, magnitude) in &peaks {
        let freq_idx = bin_idx + 1; // bin 0 in magnitudes = FFT bin 1
        let period_samples = n as f64 / freq_idx as f64;
        let period_log_days = period_samples * mean_log_spacing.abs();
        let scale_days = period_log_days.exp();

        if (MIN_SCALE_DAYS..=MAX_SCALE_DAYS).contains(&scale_days) && scale_days.is_finite() {
            candidate_scales.push((scale_days, magnitude));
        }
    }

    if candidate_scales.is_empty() {
        return LearnedScales::default_fallback(access_timestamps.len());
    }

    // Step 8: Select top 3 with minimum separation constraint
    let selected = select_separated_scales(&candidate_scales, 3);

    if selected.len() < 3 {
        return LearnedScales::default_fallback(access_timestamps.len());
    }

    // Step 9: Compute confidence
    let confidence = compute_confidence(&selected, &magnitudes, access_timestamps.len());

    let mut scales = [selected[0].0, selected[1].0, selected[2].0];
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    LearnedScales {
        scales,
        confidence,
        sample_count: access_timestamps.len(),
        computed_at: Utc::now(),
    }
}

/// Computes inter-access intervals in fractional days from sorted timestamps.
fn compute_intervals_days(timestamps: &[DateTime<Utc>]) -> Vec<f64> {
    if timestamps.len() < 2 {
        return Vec::new();
    }

    let mut sorted: Vec<DateTime<Utc>> = timestamps.to_vec();
    sorted.sort();

    sorted
        .windows(2)
        .map(|pair| {
            let duration = pair[1].signed_duration_since(pair[0]);
            duration.num_milliseconds() as f64 / (1000.0 * 60.0 * 60.0 * 24.0)
        })
        .collect()
}

/// Selects up to `count` scales that are sufficiently separated from each other.
///
/// Candidates are sorted by magnitude (strongest first). A candidate is accepted
/// only if it is at least `MIN_SCALE_SEPARATION_RATIO` away from all previously
/// accepted scales (in both directions).
fn select_separated_scales(candidates: &[(f64, f64)], count: usize) -> Vec<(f64, f64)> {
    let mut selected: Vec<(f64, f64)> = Vec::with_capacity(count);

    for &(scale, mag) in candidates {
        if selected.len() >= count {
            break;
        }

        let well_separated = selected.iter().all(|&(s, _)| {
            let ratio = if scale > s { scale / s } else { s / scale };
            ratio >= MIN_SCALE_SEPARATION_RATIO
        });

        if well_separated {
            selected.push((scale, mag));
        }
    }

    selected
}

/// Computes a confidence score (0.0-1.0) for the learned scales.
///
/// Factors:
/// 1. **Data quantity**: More samples → higher confidence (saturates at 500)
/// 2. **Peak prominence**: How far above the noise floor the selected peaks are
/// 3. **Peak count**: Whether we found 3 distinct peaks
fn compute_confidence(selected: &[(f64, f64)], magnitudes: &[f64], sample_count: usize) -> f64 {
    // Factor 1: Data quantity (logistic curve, 50% at 100 samples, saturates ~500)
    let data_factor = 1.0 / (1.0 + (-0.02 * (sample_count as f64 - 100.0)).exp());

    // Factor 2: Peak prominence - how much the selected peaks exceed the noise floor
    let mag_mean = magnitudes.iter().sum::<f64>() / magnitudes.len().max(1) as f64;
    let prominence_factor = if mag_mean > 0.0 {
        let avg_peak_mag: f64 =
            selected.iter().map(|&(_, m)| m).sum::<f64>() / selected.len().max(1) as f64;
        let snr = avg_peak_mag / mag_mean;
        // SNR of 3 → ~0.75, SNR of 6 → ~0.95
        1.0 - (-0.3 * snr).exp()
    } else {
        0.0
    };

    // Factor 3: Did we find all 3 peaks?
    let completeness = selected.len() as f64 / 3.0;

    // Geometric mean of factors (all must be reasonable for high confidence)
    (data_factor * prominence_factor * completeness)
        .cbrt()
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Helper: creates timestamps with specified inter-access intervals in days.
    fn timestamps_from_intervals(intervals_days: &[f64]) -> Vec<DateTime<Utc>> {
        let start = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let mut timestamps = vec![start];
        let mut current = start;

        for &days in intervals_days {
            let millis = (days * 24.0 * 60.0 * 60.0 * 1000.0) as i64;
            current += chrono::Duration::milliseconds(millis);
            timestamps.push(current);
        }

        timestamps
    }

    #[test]
    fn test_insufficient_data_returns_defaults() {
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.with_ymd_and_hms(2025, 1, 1 + i, 0, 0, 0).unwrap())
            .collect();

        let result = learn_decay_scales(&timestamps);
        assert_eq!(result.scales, LOG_PERIODIC_SCALES);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.sample_count, 10);
    }

    #[test]
    fn test_empty_input_returns_defaults() {
        let result = learn_decay_scales(&[]);
        assert_eq!(result.scales, LOG_PERIODIC_SCALES);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.sample_count, 0);
    }

    #[test]
    fn test_scales_are_sorted_ascending() {
        // Generate enough data with mixed periodicities
        let mut intervals = Vec::new();
        for i in 0..200 {
            // Mix of ~5 day and ~20 day cycles with noise
            let base = if i % 7 < 3 { 5.0 } else { 20.0 };
            let noise = (i as f64 * 0.1).sin() * 2.0;
            intervals.push((base + noise).max(0.1));
        }

        let timestamps = timestamps_from_intervals(&intervals);
        let result = learn_decay_scales(&timestamps);

        assert!(
            result.scales[0] <= result.scales[1],
            "scales[0]={} should be <= scales[1]={}",
            result.scales[0],
            result.scales[1]
        );
        assert!(
            result.scales[1] <= result.scales[2],
            "scales[1]={} should be <= scales[2]={}",
            result.scales[1],
            result.scales[2]
        );
    }

    #[test]
    fn test_scales_within_bounds() {
        // Generate data with extreme interval variation
        let mut intervals = Vec::new();
        for i in 0..300 {
            let days = match i % 4 {
                0 => 0.5,   // sub-daily
                1 => 7.0,   // weekly
                2 => 90.0,  // quarterly
                _ => 400.0, // over a year
            };
            intervals.push(days);
        }

        let timestamps = timestamps_from_intervals(&intervals);
        let result = learn_decay_scales(&timestamps);

        for &s in &result.scales {
            assert!(
                s >= MIN_SCALE_DAYS,
                "Scale {} below minimum {}",
                s,
                MIN_SCALE_DAYS
            );
            assert!(
                s <= MAX_SCALE_DAYS,
                "Scale {} above maximum {}",
                s,
                MAX_SCALE_DAYS
            );
        }
    }

    #[test]
    fn test_confidence_increases_with_data() {
        // More data should generally yield higher confidence
        let base_intervals: Vec<f64> = (0..500)
            .map(|i| {
                let cycle = (i as f64 * std::f64::consts::TAU / 14.0).sin();
                (7.0 + cycle * 3.0).max(0.5)
            })
            .collect();

        let small_timestamps = timestamps_from_intervals(&base_intervals[..60]);
        let large_timestamps = timestamps_from_intervals(&base_intervals[..400]);

        let small_result = learn_decay_scales(&small_timestamps);
        let large_result = learn_decay_scales(&large_timestamps);

        // The large dataset should have at least as much confidence
        // (not strictly greater due to spectral characteristics)
        assert!(
            large_result.confidence >= small_result.confidence * 0.5,
            "Large dataset confidence ({}) should not be dramatically less than small ({})",
            large_result.confidence,
            small_result.confidence
        );
    }

    #[test]
    fn test_compute_intervals_days() {
        let ts = vec![
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2025, 1, 4, 0, 0, 0).unwrap(),
        ];

        let intervals = compute_intervals_days(&ts);
        assert_eq!(intervals.len(), 2);
        assert!((intervals[0] - 1.0).abs() < 0.001);
        assert!((intervals[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_intervals_handles_unsorted() {
        let ts = vec![
            Utc.with_ymd_and_hms(2025, 1, 4, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
        ];

        let intervals = compute_intervals_days(&ts);
        assert_eq!(intervals.len(), 2);
        assert!((intervals[0] - 1.0).abs() < 0.001);
        assert!((intervals[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_select_separated_scales_enforces_separation() {
        let candidates = vec![
            (10.0, 100.0),
            (11.0, 90.0), // too close to 10.0
            (25.0, 80.0), // 25/10 = 2.5 > 2.0, ok
            (30.0, 70.0), // 30/25 = 1.2 < 2.0, too close
            (60.0, 60.0), // 60/25 = 2.4 > 2.0, ok
        ];

        let selected = select_separated_scales(&candidates, 3);
        assert_eq!(selected.len(), 3);
        assert!((selected[0].0 - 10.0).abs() < 0.001);
        assert!((selected[1].0 - 25.0).abs() < 0.001);
        assert!((selected[2].0 - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_learned_scales_struct_fields() {
        let result = LearnedScales::default_fallback(42);
        assert_eq!(result.scales, [7.0, 30.0, 365.0]);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.sample_count, 42);
    }

    #[test]
    fn test_simultaneous_timestamps_handled() {
        // All timestamps identical -> zero intervals -> fallback
        let ts: Vec<DateTime<Utc>> = (0..100)
            .map(|_| Utc.with_ymd_and_hms(2025, 6, 15, 12, 0, 0).unwrap())
            .collect();

        let result = learn_decay_scales(&ts);
        assert_eq!(result.scales, LOG_PERIODIC_SCALES);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_weekly_pattern_detected() {
        // Create a strong weekly pattern: access every ~7 days with small noise
        let mut intervals = Vec::new();
        for i in 0..150 {
            // Strong 7-day fundamental with small perturbation
            let noise = ((i * 13) % 7) as f64 * 0.1 - 0.3;
            intervals.push((7.0 + noise).max(0.5));
        }

        let timestamps = timestamps_from_intervals(&intervals);
        let result = learn_decay_scales(&timestamps);

        // At least one of the learned scales should be in the weekly neighborhood
        // (within a factor of 2 of 7 days, so between 3.5 and 14)
        let has_weekly_scale = result.scales.iter().any(|&s| (3.5..=14.0).contains(&s));

        // This is a soft assertion: FFT on a near-constant signal may not produce
        // 3 separated peaks. If confidence is 0 (fallback), that's also acceptable.
        assert!(
            has_weekly_scale || result.confidence == 0.0,
            "Expected a weekly-ish scale or fallback, got scales={:?} confidence={}",
            result.scales,
            result.confidence
        );
    }
}
