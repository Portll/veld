//! Per-LLM cognitive profile for adaptive CaRFAX thresholds.
//! Loaded from bench/profiles/*.json at startup.
//! Replaces hardcoded constants in distortions.rs and overlook.rs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Per-LLM calibrated thresholds produced by bench/calibrate.py
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMProfile {
    pub model_id: String,

    // ── CaRFAX thresholds ──
    /// CaR: depth threshold above which deep-narrow is flagged
    #[serde(default = "default_0_60")]
    pub car_depth_threshold: f64,
    /// CaR: breadth threshold below which narrow coverage is flagged
    #[serde(default = "default_0_40")]
    pub car_breadth_threshold: f64,
    /// CaR: attention imbalance threshold
    #[serde(default = "default_0_55")]
    pub car_attention_threshold: f64,
    /// CaR: severity weight for compound scoring
    #[serde(default = "default_1_5")]
    pub car_severity_weight: f64,

    /// FA: HAZOP deficit scale factor
    #[serde(default = "default_0_80")]
    pub fa_hazop_scale: f64,
    /// FA: FMEA deficit scale factor
    #[serde(default = "default_0_60")]
    pub fa_fmea_scale: f64,
    /// FA: severity weight
    #[serde(default = "default_0_70")]
    pub fa_severity_weight: f64,

    /// X: breadth gate for phase transitions
    #[serde(default = "default_0_60")]
    pub x_breadth_gate: f64,
    /// X: scale gate for late-phase transitions
    #[serde(default = "default_0_67")]
    pub x_scale_gate: f64,
    /// X: severity weight
    #[serde(default = "default_1_5")]
    pub x_severity_weight: f64,

    // ── EMA smoothing per cluster ──
    #[serde(default = "default_0_50")]
    pub ema_intent: f64,
    #[serde(default = "default_0_45")]
    pub ema_observation: f64,
    #[serde(default = "default_0_40")]
    pub ema_pattern: f64,
    #[serde(default = "default_0_40")]
    pub ema_risk: f64,
    #[serde(default = "default_0_35")]
    pub ema_learning: f64,

    // ── Phase gate overrides ──
    #[serde(default = "default_0_20")]
    pub phase1_breadth_min: f64,
    #[serde(default = "default_0_60")]
    pub phase2_breadth_min: f64,
    #[serde(default = "default_0_60")]
    pub phase2_hazop_min: f64,
    #[serde(default = "default_0_40")]
    pub phase2_fmea_min: f64,
    #[serde(default = "default_0_30")]
    pub phase2_mirror_min: f64,

    // ── Spike detection ──
    #[serde(default = "default_0_15")]
    pub spike_absolute_theta: f64,
    #[serde(default = "default_0_30")]
    pub spike_cumulative_theta: f64,
    #[serde(default = "default_0_15")]
    pub spike_cross_cluster_theta: f64,

    // ── Retrieval signal weights (20 signals for Shodh) ──
    #[serde(default = "default_retrieval_weights")]
    pub retrieval_weights: Vec<f64>,

    // ── Metadata ──
    #[serde(default)]
    pub n_calibration_runs: u32,
    #[serde(default)]
    pub calibration_date: String,
}

// Default value functions for serde
fn default_0_15() -> f64 { 0.15 }
fn default_0_20() -> f64 { 0.20 }
fn default_0_30() -> f64 { 0.30 }
fn default_0_35() -> f64 { 0.35 }
fn default_0_40() -> f64 { 0.40 }
fn default_0_45() -> f64 { 0.45 }
fn default_0_50() -> f64 { 0.50 }
fn default_0_55() -> f64 { 0.55 }
fn default_0_60() -> f64 { 0.60 }
fn default_0_67() -> f64 { 0.67 }
fn default_0_70() -> f64 { 0.70 }
fn default_0_80() -> f64 { 0.80 }
fn default_1_5() -> f64 { 1.5 }
fn default_retrieval_weights() -> Vec<f64> { vec![1.0; 20] }

impl Default for LLMProfile {
    fn default() -> Self {
        Self {
            model_id: "default".to_string(),
            car_depth_threshold: 0.60,
            car_breadth_threshold: 0.40,
            car_attention_threshold: 0.55,
            car_severity_weight: 1.5,
            fa_hazop_scale: 0.80,
            fa_fmea_scale: 0.60,
            fa_severity_weight: 0.70,
            x_breadth_gate: 0.60,
            x_scale_gate: 0.67,
            x_severity_weight: 1.5,
            ema_intent: 0.50,
            ema_observation: 0.45,
            ema_pattern: 0.40,
            ema_risk: 0.40,
            ema_learning: 0.35,
            phase1_breadth_min: 0.20,
            phase2_breadth_min: 0.60,
            phase2_hazop_min: 0.60,
            phase2_fmea_min: 0.40,
            phase2_mirror_min: 0.30,
            spike_absolute_theta: 0.15,
            spike_cumulative_theta: 0.30,
            spike_cross_cluster_theta: 0.15,
            retrieval_weights: vec![1.0; 20],
            n_calibration_runs: 0,
            calibration_date: String::new(),
        }
    }
}

impl LLMProfile {
    /// Load from a JSON file produced by calibrate.py
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let profile: Self = serde_json::from_str(&contents)?;
        Ok(profile)
    }

    /// Compute CaR signal strength using this profile's thresholds
    /// instead of hardcoded constants.
    pub fn car_strength(&self, depth: f64, breadth: f64, attention: f64, cluster_imbalance: f64) -> f64 {
        let deep_narrow = if depth > self.car_depth_threshold && breadth < self.car_breadth_threshold {
            depth - breadth
        } else {
            0.0
        };
        let attention_signal = if attention > self.car_attention_threshold {
            attention - self.car_attention_threshold
        } else {
            0.0
        };
        (deep_narrow * 0.4 + attention_signal * 0.3 + cluster_imbalance * 0.3).min(1.0)
    }

    /// Compute FA signal strength using this profile's thresholds
    pub fn fa_strength(&self, phase: f64, hazop: f64, fmea: f64) -> f64 {
        let hazop_deficit = if phase > 0.35 {
            (phase * self.fa_hazop_scale - hazop).max(0.0)
        } else {
            0.0
        };
        let fmea_deficit = if phase > 0.35 {
            (phase * self.fa_fmea_scale - fmea).max(0.0)
        } else {
            0.0
        };
        (hazop_deficit * 0.5 + fmea_deficit * 0.5).min(1.0)
    }

    /// Compute X signal strength using this profile's thresholds
    pub fn x_strength(&self, phase: f64, breadth: f64, scale: f64) -> f64 {
        let breadth_deficit = if phase > 0.40 {
            (self.x_breadth_gate - breadth).max(0.0)
        } else {
            0.0
        };
        let scale_deficit = if phase > 0.60 {
            (self.x_scale_gate - scale).max(0.0)
        } else {
            0.0
        };
        (breadth_deficit * 0.5 + scale_deficit * 0.5).min(1.0)
    }

    /// Get EMA alpha for a given cluster index (0-4)
    pub fn ema_alpha(&self, cluster: usize) -> f64 {
        match cluster {
            0 => self.ema_intent,
            1 => self.ema_observation,
            2 => self.ema_pattern,
            3 => self.ema_risk,
            4 => self.ema_learning,
            _ => 0.40,
        }
    }

    /// Check if a phase gate is satisfied for this LLM's profile
    pub fn phase2_gate_met(&self, breadth: f64, hazop: f64, fmea: f64, mirror: f64) -> bool {
        breadth >= self.phase2_breadth_min
            && hazop >= self.phase2_hazop_min
            && fmea >= self.phase2_fmea_min
            && mirror >= self.phase2_mirror_min
    }
}

/// Registry of loaded LLM profiles. Falls back to default if model not found.
pub struct ProfileRegistry {
    profiles: HashMap<String, LLMProfile>,
    default: LLMProfile,
}

impl ProfileRegistry {
    /// Load all profiles from a directory
    pub fn load(dir: &Path) -> Self {
        let mut profiles = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "json") {
                    match LLMProfile::from_file(&path) {
                        Ok(p) => { profiles.insert(p.model_id.clone(), p); }
                        Err(e) => { eprintln!("Failed to load profile {:?}: {}", path, e); }
                    }
                }
            }
        }
        Self { profiles, default: LLMProfile::default() }
    }

    /// Get profile for a model, falling back to default
    pub fn get(&self, model_id: &str) -> &LLMProfile {
        self.profiles.get(model_id).unwrap_or(&self.default)
    }

    pub fn models(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_matches_current_constants() {
        let p = LLMProfile::default();
        // These must match the current hardcoded values in distortions.rs
        assert!((p.car_depth_threshold - 0.60).abs() < 1e-6);
        assert!((p.car_breadth_threshold - 0.40).abs() < 1e-6);
        assert!((p.x_breadth_gate - 0.60).abs() < 1e-6);
    }

    #[test]
    fn car_strength_zero_when_below_threshold() {
        let p = LLMProfile::default();
        assert_eq!(p.car_strength(0.50, 0.50, 0.40, 0.0), 0.0);
    }

    #[test]
    fn car_strength_positive_when_deep_narrow() {
        let p = LLMProfile::default();
        let s = p.car_strength(0.80, 0.30, 0.60, 0.2);
        assert!(s > 0.0);
    }

    #[test]
    fn qwq_profile_higher_car_threshold() {
        // QwQ naturally goes deep — CaR threshold should be higher
        let mut p = LLMProfile::default();
        p.car_depth_threshold = 0.80;  // calibrated from QwQ traces
        // Same depth that triggers CaR on default profile doesn't trigger on QwQ
        assert_eq!(p.car_strength(0.70, 0.30, 0.40, 0.0), 0.0);
    }

    #[test]
    fn phase2_gate_respects_profile() {
        let p = LLMProfile::default();
        assert!(!p.phase2_gate_met(0.50, 0.60, 0.40, 0.30));  // breadth too low
        assert!(p.phase2_gate_met(0.60, 0.60, 0.40, 0.30));   // all met
    }

    #[test]
    fn roundtrip_json() {
        let p = LLMProfile::default();
        let json = p.to_json();
        let p2: LLMProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(p.model_id, p2.model_id);
        assert!((p.car_depth_threshold - p2.car_depth_threshold).abs() < 1e-6);
    }
}
