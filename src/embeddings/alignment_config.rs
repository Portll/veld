//! Runtime configuration for the alignment subsystem — the "alignment performance flags".
//!
//! Mirrors the `CognitiveDials` pattern: seeded from `VELD_ALIGN_*` env at construction,
//! later mutated via `/api/align`. All v2 behavior is gated behind [`AlignmentConfig::v2_enabled`]
//! (default `false` = byte-identical current behavior). Defaults of the *sub*-flags are
//! sensible-on, so flipping the master gives a good config rather than a blank one.
//!
//! This module is pure config + validation; it deliberately does not touch `MemorySystem`,
//! HTTP handlers, MCP, or the GUI (that cross-cutting wiring is a separate integration step).

use serde::{Deserialize, Serialize};

/// Fitter selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentMethod {
    /// Residual-gated escalation (the current behavior + conditioning when enabled).
    #[default]
    Auto,
    Procrustes,
    Ridge,
    WhitenedProcrustes,
    Cca,
}

/// Fit objective. `Frobenius` is closed-form; `Cosine`/`InfoNce` require a training loop
/// (an ML runtime not yet wired) and are therefore capability-gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentObjective {
    #[default]
    Frobenius,
    Cosine,
    InfoNce,
}

/// Whitening mode used by the conditioning stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WhitenMode {
    Off,
    #[default]
    Zca,
    Pca,
}

/// Metric used to display and to gate auto-promotion of a fitted method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EvalMetric {
    #[default]
    PairedCosine,
    RecallAtK,
    Mrr,
}

/// Maturity/health state of a capability — drives the platform status color taxonomy.
/// Maturity: Specified → Stubbed → Building → Operational; Health (once operational):
/// Operational(green) → Degraded(yellow) → SignificantlyDegraded(orange) → Broken(red).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityState {
    Specified,
    Stubbed,
    Building,
    Operational,
    Degraded,
    SignificantlyDegraded,
    Broken,
}

/// The alignment performance flags. Per embedder-pair in production (a `__default__`
/// block seeds new pairs); this struct is the per-pair value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Master parity switch. `false` ⇒ exact current behavior; all fields below inert.
    pub v2_enabled: bool,

    // Group A — conditioning
    pub center: bool,
    pub whiten: WhitenMode,
    pub whiten_eps: f32,
    pub variance_target: f32,

    // Group B — fitter
    pub method: AlignmentMethod,
    pub objective: AlignmentObjective,
    pub ridge_lambda: f32,
    pub residual_escalation_threshold: f32,
    pub infonce_temperature: f32,

    // Group C — adapter (ML-runtime gated)
    pub adapter_enabled: bool,
    pub adapter_hidden_dim: u32,
    pub adapter_layers: u32,
    pub adapter_epochs: u32,

    // Group D — retrieval-time
    pub csls_enabled: bool,
    pub csls_neighborhood_k: u32,

    // Group E — data
    pub self_learning_iters: u32,

    // Group F — experimental (compile-feature gated)
    pub vec2vec_enabled: bool,

    // Eval gate
    pub eval_metric: EvalMetric,
    pub eval_holdout_frac: f32,
    /// Auto-promotion of a newly-fitted method requires a recall win, never cosine alone.
    pub promote_requires_recall: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            v2_enabled: false,
            center: true,
            whiten: WhitenMode::Zca,
            whiten_eps: 1e-2,
            variance_target: 0.95,
            method: AlignmentMethod::Auto,
            objective: AlignmentObjective::Frobenius,
            ridge_lambda: 0.05,
            residual_escalation_threshold: 0.15,
            infonce_temperature: 0.07,
            adapter_enabled: false,
            adapter_hidden_dim: 512,
            adapter_layers: 2,
            adapter_epochs: 30,
            csls_enabled: false,
            csls_neighborhood_k: 10,
            self_learning_iters: 0,
            vec2vec_enabled: false,
            eval_metric: EvalMetric::PairedCosine,
            eval_holdout_frac: 0.1,
            promote_requires_recall: true,
        }
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

impl AlignmentConfig {
    /// Seed from `VELD_ALIGN_*` env, then clamp into valid ranges.
    pub fn from_env() -> Self {
        let d = Self::default();
        let mut c = Self {
            v2_enabled: env_bool("VELD_ALIGN_V2_ENABLED", d.v2_enabled),
            center: env_bool("VELD_ALIGN_CENTER", d.center),
            whiten: match std::env::var("VELD_ALIGN_WHITEN").as_deref() {
                Ok("off") => WhitenMode::Off,
                Ok("pca") => WhitenMode::Pca,
                Ok("zca") => WhitenMode::Zca,
                _ => d.whiten,
            },
            whiten_eps: env_f32("VELD_ALIGN_WHITEN_EPS", d.whiten_eps),
            variance_target: env_f32("VELD_ALIGN_VARIANCE_TARGET", d.variance_target),
            ridge_lambda: env_f32("VELD_ALIGN_RIDGE_LAMBDA", d.ridge_lambda),
            residual_escalation_threshold: env_f32(
                "VELD_ALIGN_RESIDUAL_ESCALATION_THRESHOLD",
                d.residual_escalation_threshold,
            ),
            infonce_temperature: env_f32("VELD_ALIGN_INFONCE_TEMPERATURE", d.infonce_temperature),
            adapter_enabled: env_bool("VELD_ALIGN_ADAPTER_ENABLED", d.adapter_enabled),
            adapter_hidden_dim: env_u32("VELD_ALIGN_ADAPTER_HIDDEN_DIM", d.adapter_hidden_dim),
            adapter_layers: env_u32("VELD_ALIGN_ADAPTER_LAYERS", d.adapter_layers),
            adapter_epochs: env_u32("VELD_ALIGN_ADAPTER_EPOCHS", d.adapter_epochs),
            csls_enabled: env_bool("VELD_ALIGN_CSLS_ENABLED", d.csls_enabled),
            csls_neighborhood_k: env_u32("VELD_ALIGN_CSLS_NEIGHBORHOOD_K", d.csls_neighborhood_k),
            self_learning_iters: env_u32("VELD_ALIGN_SELF_LEARNING_ITERS", d.self_learning_iters),
            vec2vec_enabled: env_bool("VELD_ALIGN_VEC2VEC_ENABLED", d.vec2vec_enabled),
            eval_holdout_frac: env_f32("VELD_ALIGN_EVAL_HOLDOUT_FRAC", d.eval_holdout_frac),
            promote_requires_recall: env_bool(
                "VELD_ALIGN_PROMOTE_REQUIRES_RECALL",
                d.promote_requires_recall,
            ),
            method: d.method,
            objective: d.objective,
            eval_metric: d.eval_metric,
        };
        c.clamp();
        c
    }

    /// Clamp every numeric field into its spec'd range (server-side DoS guard).
    pub fn clamp(&mut self) {
        self.whiten_eps = self.whiten_eps.clamp(1e-3, 1e-1);
        self.variance_target = self.variance_target.clamp(0.5, 1.0);
        self.ridge_lambda = self.ridge_lambda.clamp(0.0, 10.0);
        self.residual_escalation_threshold = self.residual_escalation_threshold.clamp(0.0, 1.0);
        self.infonce_temperature = self.infonce_temperature.clamp(0.01, 1.0);
        self.adapter_hidden_dim = self.adapter_hidden_dim.clamp(64, 2048);
        self.adapter_layers = self.adapter_layers.clamp(1, 4);
        self.adapter_epochs = self.adapter_epochs.clamp(1, 200);
        self.csls_neighborhood_k = self.csls_neighborhood_k.clamp(1, 50);
        self.self_learning_iters = self.self_learning_iters.min(20);
        self.eval_holdout_frac = self.eval_holdout_frac.clamp(0.0, 0.5);
    }

    /// Boundary-safe entry point: clamp into valid ranges. Every path that accepts
    /// external input (e.g. a future `PUT /api/align` deserialize) MUST route through
    /// this before the config is used — `clamp` is not guaranteed by `Deserialize` (M3).
    pub fn sanitized(mut self) -> Self {
        self.clamp();
        self
    }

    /// Reject contradictory combinations and capabilities the build can't honor.
    /// `gradient_runtime_available` and `vec2vec_compiled` reflect actual build capability.
    pub fn validate(
        &self,
        gradient_runtime_available: bool,
        vec2vec_compiled: bool,
    ) -> Result<(), String> {
        if !self.v2_enabled {
            return Ok(()); // master off ⇒ nothing to validate
        }
        // Closed-form fitters cannot honor a gradient objective.
        let closed_form = matches!(
            self.method,
            AlignmentMethod::Procrustes
                | AlignmentMethod::WhitenedProcrustes
                | AlignmentMethod::Cca
        );
        let gradient_obj = matches!(
            self.objective,
            AlignmentObjective::Cosine | AlignmentObjective::InfoNce
        );
        if closed_form && gradient_obj {
            return Err(format!(
                "method {:?} is closed-form and cannot use a gradient objective {:?}; \
                 use method=ridge with an adapter, or objective=frobenius",
                self.method, self.objective
            ));
        }
        if (gradient_obj || self.adapter_enabled) && !gradient_runtime_available {
            return Err(
                "objective/adapter requires a gradient training runtime that is not available in this build"
                    .to_string(),
            );
        }
        if self.vec2vec_enabled && !vec2vec_compiled {
            return Err("vec2vec is not compiled into this build".to_string());
        }
        Ok(())
    }

    /// Capability state for a method — drives the platform status color taxonomy.
    /// Closed-form methods implemented in this build are `Operational`; `Cca` is
    /// `Specified` (the distinct truncated-canonical-correlation fitter is not built —
    /// it currently shares the whitened-Procrustes math and is not exposed as its own).
    pub fn method_state(method: AlignmentMethod) -> CapabilityState {
        match method {
            AlignmentMethod::Auto
            | AlignmentMethod::Procrustes
            | AlignmentMethod::Ridge
            | AlignmentMethod::WhitenedProcrustes => CapabilityState::Operational,
            AlignmentMethod::Cca => CapabilityState::Specified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_parity_off() {
        let c = AlignmentConfig::default();
        assert!(!c.v2_enabled);
        // master off ⇒ validate always ok regardless of capability
        assert!(c.validate(false, false).is_ok());
    }

    #[test]
    fn clamp_bounds_numerics() {
        let mut c = AlignmentConfig::default();
        c.adapter_epochs = u32::MAX;
        c.ridge_lambda = 1e9;
        c.variance_target = 5.0;
        c.csls_neighborhood_k = 9999;
        c.clamp();
        assert_eq!(c.adapter_epochs, 200);
        assert_eq!(c.ridge_lambda, 10.0);
        assert_eq!(c.variance_target, 1.0);
        assert_eq!(c.csls_neighborhood_k, 50);
    }

    #[test]
    fn rejects_closed_form_plus_gradient_objective() {
        let mut c = AlignmentConfig {
            v2_enabled: true,
            method: AlignmentMethod::Procrustes,
            objective: AlignmentObjective::InfoNce,
            ..Default::default()
        };
        c.clamp();
        assert!(c.validate(true, true).is_err());
    }

    #[test]
    fn rejects_adapter_without_runtime() {
        let c = AlignmentConfig {
            v2_enabled: true,
            adapter_enabled: true,
            ..Default::default()
        };
        assert!(c.validate(false, false).is_err());
        assert!(c.validate(true, false).is_ok());
    }

    #[test]
    fn rejects_vec2vec_when_not_compiled() {
        let c = AlignmentConfig {
            v2_enabled: true,
            vec2vec_enabled: true,
            ..Default::default()
        };
        assert!(c.validate(true, false).is_err());
        assert!(c.validate(true, true).is_ok());
    }
}
