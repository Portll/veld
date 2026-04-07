"""
Per-LLM Cognitive Profile Calibration
Trains on accumulated Overlook traces to produce per-model threshold profiles.
Output: profiles/*.json consumed by Pinky's Rust evaluation engine.

Usage:
    python calibrate.py --traces-dir traces/ --output-dir profiles/
    python calibrate.py --traces-dir traces/ --model qwq-32b --visualise
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ═══════════════════════════════════════════════════
# Overlook dimension definitions (must match overlook.rs)
# ═══════════════════════════════════════════════════

DIMENSIONS = [
    # Cluster A: Intent (α=0.5 fast EMA)
    "clarity", "evaluation_depth", "coverage_breadth", "phase_position", "attention_balance",
    # Cluster B: Observation (α=0.45)
    "evidence_quality", "source_diversity", "contradiction_rate", "data_recency",
    "tier_depth", "anomaly_density", "hazop_completion", "fmea_completion",
    # Cluster C: Pattern (α=0.4)
    "pattern_confidence", "cross_domain_transfer", "novelty_index",
    "sibling_consistency", "mirror_symmetry", "temporal_stability",
    # Cluster D: Risk (α=0.4, CRITICAL)
    "auth_risk", "pii_exposure", "integration_fragility",
    # Cluster E: Learning (α=0.35 slow EMA)
    "learning_velocity", "pattern_confirmation", "error_correction_rate",
    "conceptual_self", "seed_utilisation",
]
N_DIM = len(DIMENSIONS)  # 27

CLUSTERS = {
    "intent": slice(0, 5),
    "observation": slice(5, 13),
    "pattern": slice(13, 19),
    "risk": slice(19, 22),
    "learning": slice(22, 27),
}

# CaRFAX mechanism indices
CAR_DIMS = {"depth": 1, "breadth": 2, "attention": 4}
FA_DIMS = {"phase": 3, "hazop": 11, "fmea": 12}
X_DIMS = {"breadth": 2, "phase": 3, "scale": 9}


# ═══════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════

@dataclass
class OverlookTrace:
    """Single evaluation trace: sequence of 27-D vectors over N calls."""
    model_id: str
    task_id: str
    vectors: list  # List[List[float]] — shape (n_calls, 27)
    quality_score: float  # final human or benchmark score
    carfax_signals: dict = field(default_factory=dict)  # {car: float, fa: float, x: float}
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMProfile:
    """Per-LLM calibrated thresholds. Serialises to JSON for Rust consumption."""
    model_id: str
    # CaRFAX thresholds (defaults match current Pinky constants)
    car_depth_threshold: float = 0.60
    car_breadth_threshold: float = 0.40
    car_attention_threshold: float = 0.55
    car_severity_weight: float = 1.5
    fa_hazop_scale: float = 0.8
    fa_fmea_scale: float = 0.6
    fa_severity_weight: float = 0.7
    x_breadth_gate: float = 0.60
    x_scale_gate: float = 0.67
    x_severity_weight: float = 1.5
    # EMA alphas per cluster
    ema_intent: float = 0.50
    ema_observation: float = 0.45
    ema_pattern: float = 0.40
    ema_risk: float = 0.40
    ema_learning: float = 0.35
    # Phase gate overrides
    phase1_breadth_min: float = 0.20
    phase2_breadth_min: float = 0.60
    phase2_hazop_min: float = 0.60
    phase2_fmea_min: float = 0.40
    phase2_mirror_min: float = 0.30
    # Spike detection
    spike_absolute_theta: float = 0.15
    spike_cumulative_theta: float = 0.30
    spike_cross_cluster_theta: float = 0.15
    # Retrieval signal weights (20 signals — for Veld integration)
    retrieval_weights: list = field(default_factory=lambda: [1.0] * 20)
    # Metadata
    n_calibration_runs: int = 0
    calibration_date: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "LLMProfile":
        with open(path) as f:
            return cls(**json.load(f))


# ═══════════════════════════════════════════════════
# Calibration network
# ═══════════════════════════════════════════════════

class ProfileCalibrator(nn.Module):
    """
    Tiny network: (model_embedding + overlook_summary) → calibrated_score.
    The learned weights encode per-LLM dimensional sensitivities.
    """

    def __init__(self, n_models: int, embed_dim: int = 16):
        super().__init__()
        self.model_embed = nn.Embedding(n_models, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + N_DIM * 3, 64),  # 3 = mean, std, final
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, model_idx: torch.Tensor, trace_summary: torch.Tensor) -> torch.Tensor:
        emb = self.model_embed(model_idx)
        x = torch.cat([emb, trace_summary], dim=-1)
        return self.net(x).squeeze(-1)


def summarise_trace(vectors: list) -> np.ndarray:
    """Compress a variable-length trace into fixed-size summary: mean, std, final."""
    arr = np.array(vectors, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros(N_DIM * 3, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    final = arr[-1]
    return np.concatenate([mean, std, final])


# ═══════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════

def load_traces(traces_dir: str) -> list[OverlookTrace]:
    """Load evaluation traces from JSON files."""
    traces = []
    for path in Path(traces_dir).glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        if "vectors" not in data or "quality_score" not in data:
            continue
        traces.append(OverlookTrace(**{k: v for k, v in data.items() if k in OverlookTrace.__dataclass_fields__}))
    return traces


def train_calibrator(traces: list[OverlookTrace], epochs: int = 200, lr: float = 1e-3) -> tuple[ProfileCalibrator, dict]:
    """Train the calibration network on accumulated traces."""
    model_ids = sorted(set(t.model_id for t in traces))
    model_to_idx = {m: i for i, m in enumerate(model_ids)}

    X_model = torch.tensor([model_to_idx[t.model_id] for t in traces], dtype=torch.long)
    X_trace = torch.tensor(np.stack([summarise_trace(t.vectors) for t in traces]), dtype=torch.float32)
    y = torch.tensor([t.quality_score for t in traces], dtype=torch.float32)

    net = ProfileCalibrator(n_models=len(model_ids))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        net.train()
        pred = net(X_model, X_trace)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return net, {"model_ids": model_ids, "model_to_idx": model_to_idx, "best_loss": best_loss}


# ═══════════════════════════════════════════════════
# Profile extraction
# ═══════════════════════════════════════════════════

def extract_profile(model_id: str, traces: list[OverlookTrace]) -> LLMProfile:
    """
    Extract per-LLM profile from traces using statistical analysis.
    For now: percentile-based thresholds. With enough data, replace with
    learned weights from the calibrator network.
    """
    model_traces = [t for t in traces if t.model_id == model_id]
    if not model_traces:
        return LLMProfile(model_id=model_id)

    # Stack all vectors
    all_vecs = []
    for t in model_traces:
        all_vecs.extend(t.vectors)
    arr = np.array(all_vecs, dtype=np.float32)

    # CaR thresholds: set to this model's 90th percentile (not global constant)
    depth_vals = arr[:, CAR_DIMS["depth"]]
    breadth_vals = arr[:, CAR_DIMS["breadth"]]
    attention_vals = arr[:, CAR_DIMS["attention"]]

    profile = LLMProfile(
        model_id=model_id,
        car_depth_threshold=float(np.percentile(depth_vals, 90)),
        car_breadth_threshold=float(np.percentile(breadth_vals, 10)),
        car_attention_threshold=float(np.percentile(attention_vals, 90)),
        # FA: adjust based on how much this model naturally skips risk analysis
        fa_hazop_scale=0.8,
        fa_fmea_scale=0.6,
        # X: phase gates based on this model's natural breadth capacity
        phase2_breadth_min=float(np.percentile(breadth_vals, 25)),
        # EMA: reasoning models need slower smoothing (more volatile depth)
        ema_intent=0.5 if np.std(depth_vals) < 0.15 else 0.35,
        # Spike detection: scale to this model's natural variance
        spike_absolute_theta=float(np.std(arr, axis=0).mean() * 1.5),
        n_calibration_runs=len(model_traces),
    )

    from datetime import datetime
    profile.calibration_date = datetime.now().isoformat()
    return profile


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Per-LLM Cognitive Profile Calibration")
    parser.add_argument("--traces-dir", default="traces/", help="Directory of evaluation trace JSONs")
    parser.add_argument("--output-dir", default="profiles/", help="Output directory for profile JSONs")
    parser.add_argument("--model", help="Calibrate single model (default: all)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--visualise", action="store_true", help="Generate radar charts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    traces = load_traces(args.traces_dir)
    print(f"Loaded {len(traces)} traces from {args.traces_dir}")

    if not traces:
        print("No traces found. Run evaluations first with run_eval.sh")
        # Generate default profiles for known models
        for model_id in ["opus-4", "sonnet-4", "qwq-32b", "deepseek-r1", "mistral-7b"]:
            profile = LLMProfile(model_id=model_id)
            out = os.path.join(args.output_dir, f"{model_id}.json")
            with open(out, "w") as f:
                f.write(profile.to_json())
            print(f"  Default profile: {out}")
        return

    model_ids = sorted(set(t.model_id for t in traces))
    if args.model:
        model_ids = [args.model]

    # Train calibrator
    print("Training calibration network...")
    net, meta = train_calibrator(traces, epochs=args.epochs)
    print(f"  Best loss: {meta['best_loss']:.4f}")

    # Extract and save profiles
    for model_id in model_ids:
        profile = extract_profile(model_id, traces)
        out = os.path.join(args.output_dir, f"{model_id}.json")
        with open(out, "w") as f:
            f.write(profile.to_json())
        print(f"  Profile: {out} ({profile.n_calibration_runs} runs)")

    if args.visualise:
        try:
            visualise_profiles(args.output_dir, model_ids)
        except ImportError:
            print("  matplotlib not installed, skipping visualisation")


def visualise_profiles(output_dir: str, model_ids: list[str]):
    """Generate per-LLM radar charts comparing CaRFAX thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(model_ids), figsize=(5 * len(model_ids), 5),
                              subplot_kw=dict(polar=True))
    if len(model_ids) == 1:
        axes = [axes]

    labels = ["CaR depth", "CaR breadth", "CaR attention", "FA hazop", "FA fmea",
              "X breadth", "X scale", "Spike θ"]

    for ax, model_id in zip(axes, model_ids):
        profile = LLMProfile.from_json(os.path.join(output_dir, f"{model_id}.json"))
        values = [
            profile.car_depth_threshold, profile.car_breadth_threshold,
            profile.car_attention_threshold, profile.fa_hazop_scale,
            profile.fa_fmea_scale, profile.x_breadth_gate,
            profile.x_scale_gate, profile.spike_absolute_theta,
        ]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]; angles += angles[:1]
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=7)
        ax.set_title(model_id, size=12, pad=15)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(output_dir, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"  Radar chart: {out}")


if __name__ == "__main__":
    main()
