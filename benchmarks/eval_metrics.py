"""
Shared evaluation metrics for shodh-memory benchmarks.

Used by both locomo_llm_judge.py and locomo_api_benchmark.py to ensure
sibling consistency (Overlook D17 fix).
"""

from datetime import datetime, timezone


def compute_mrr(ranked_ids, expected_ids):
    """Mean Reciprocal Rank — rank of first relevant result."""
    for i, rid in enumerate(ranked_ids):
        if rid in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(ranked_ids, expected_ids, k):
    """Recall@K — fraction of relevant items in top-K."""
    if not expected_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & expected_ids) / len(expected_ids)


# ── E4: Freshness stratification ──

FRESHNESS_BANDS = {
    "A_0_7d": (0, 7),
    "B_7_30d": (7, 30),
    "C_30_90d": (30, 90),
    "D_90d_plus": (90, 999999),
}


def compute_memory_age_days(memory_created_at: str) -> float:
    """Compute age of a memory in days from its created_at ISO timestamp."""
    try:
        created = datetime.fromisoformat(memory_created_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - created).total_seconds() / 86400
    except (ValueError, TypeError):
        return -1.0


def classify_freshness_band(age_days: float) -> str:
    """Classify a memory age into a freshness band matching constants.rs recency tiers."""
    for band_name, (lo, hi) in FRESHNESS_BANDS.items():
        if lo <= age_days < hi:
            return band_name
    return "D_90d_plus"
