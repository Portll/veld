"""
Shodh-Memory: AI Memory System for Autonomous Robots & Drones

Native Python bindings for high-performance memory operations
optimized for robotics, drones, and offline AI applications.

Usage:
    from shodh_memory import Memory  # or MemorySystem (same class)

    memory = Memory(storage_path="./my_data")
    memory.remember("Important fact", memory_type="Context")
    results = memory.recall("query", limit=5)

Features:
- Position(x, y, z) - Local robot coordinates in meters
- GeoLocation(lat, lon, alt) - GPS for drones & outdoor robots
- GeoFilter - Spatial queries by radius
- DecisionContext - For action-outcome learning (what conditions -> what action)
- Outcome - Result of decisions (success/failure/partial + reward signal)
- Environment - Weather, terrain, lighting, nearby agents
- Failure tracking - Severity, root cause, recovery actions
- Anomaly detection - Track unusual sensor readings
- Pattern learning - Match situations to learned patterns
- 100% offline operation - No cloud, no API keys

LLM Framework Integrations:
    # LangChain - pip install shodh-memory[langchain]
    from shodh_memory.integrations.langchain import ShodhMemory

    # LlamaIndex - pip install shodh-memory[llamaindex]
    from shodh_memory.integrations.llamaindex import ShodhLlamaMemory

    # Both - pip install shodh-memory[all]
"""

import os as _os
from pathlib import Path as _Path

# Set package directory for bundled ONNX Runtime discovery
# This allows the Rust code to find lib/onnxruntime.dll bundled in the wheel
_package_dir = _Path(__file__).parent
_os.environ.setdefault("SHODH_PACKAGE_DIR", str(_package_dir))

from .shodh_memory import (
    MemorySystem,
    # Location types
    Position,
    GeoLocation,
    GeoFilter,
    # Decision & Learning types
    DecisionContext,
    Outcome,
    Environment,
    # Version
    __version__,
)

# Alias for simpler API - Memory and MemorySystem are identical
Memory = MemorySystem

__all__ = [
    # Core - both names export the same class
    "Memory",
    "MemorySystem",
    # Location types
    "Position",
    "GeoLocation",
    "GeoFilter",
    # Decision & Learning types
    "DecisionContext",
    "Outcome",
    "Environment",
    # Version
    "__version__",
]

# Show GitHub prompt once per session
import sys as _sys
if not getattr(_sys, '_shodh_star_shown', False):
    try:
        print("\u2b50 Love Veld? Star us: https://github.com/Portll/veld")
    except UnicodeEncodeError:
        print("* Love Veld? Star us: https://github.com/Portll/veld")
    _sys._shodh_star_shown = True
