#!/usr/bin/env python3
"""Convert ONNX embedding models to CoreML (.mlpackage) for Apple Neural Engine.

Converts MiniLM-L6-v2, Nomic-embed-text-v1.5, and ms-marco-MiniLM cross-encoder
from ONNX to CoreML format. The CoreML runtime routes transformer ops to the ANE
on Apple Silicon, yielding ~5-10x speedup over CPU-only ONNX.

Requirements:
    pip install coremltools onnx

Usage:
    python scripts/convert-coreml.py

Outputs to ~/.cache/shodh-memory/coreml/
"""

import sys
from pathlib import Path

try:
    import coremltools as ct
    import onnx
except ImportError:
    print("Install dependencies: pip install coremltools onnx", file=sys.stderr)
    sys.exit(1)

CACHE_DIR = Path.home() / ".cache" / "shodh-memory"
OUTPUT_DIR = CACHE_DIR / "coreml"

MODELS = [
    {
        "name": "minilm-l6-v2",
        "onnx_path": CACHE_DIR / "minilm-l6" / "model.onnx",
        "seq_len": 256,
        "batch_sizes": [1, 8, 16],
    },
    {
        "name": "nomic-embed-text-v1.5",
        "onnx_path": CACHE_DIR / "nomic-embed" / "model.onnx",
        "seq_len": 512,
        "batch_sizes": [1, 8, 16],
    },
    {
        "name": "ms-marco-MiniLM-cross-encoder",
        "onnx_path": CACHE_DIR / "cross-encoder" / "model.onnx",
        "seq_len": 512,
        "batch_sizes": [1, 8, 32],
    },
]


def convert_model(config: dict) -> bool:
    onnx_path = config["onnx_path"]
    if not onnx_path.exists():
        print(f"  SKIP {config['name']}: {onnx_path} not found (not downloaded yet)")
        return False

    output_path = OUTPUT_DIR / f"{config['name']}.mlpackage"
    if output_path.exists():
        print(f"  SKIP {config['name']}: {output_path} already exists")
        return True

    print(f"  Converting {config['name']}...")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))

    # Determine input shapes from ONNX graph
    seq_len = config["seq_len"]
    # Use flexible batch size via RangeDim
    batch_dim = ct.RangeDim(lower_bound=1, upper_bound=max(config["batch_sizes"]), default=1)

    input_shapes = {}
    for inp in onnx_model.graph.input:
        name = inp.name
        input_shapes[name] = ct.Shape((batch_dim, seq_len))

    # Convert
    mlmodel = ct.convert(
        str(onnx_path),
        source="onnx",
        inputs=[
            ct.TensorType(name=name, shape=shape)
            for name, shape in input_shapes.items()
        ],
        compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
        minimum_deployment_target=ct.target.macOS14,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"  Saved {output_path}")
    return True


def main():
    print("Converting ONNX models to CoreML for Apple Neural Engine\n")

    converted = 0
    for config in MODELS:
        if convert_model(config):
            converted += 1

    print(f"\n{converted}/{len(MODELS)} models ready in {OUTPUT_DIR}")

    if converted == 0:
        print("\nNo ONNX models found. Run shodh-memory once to auto-download them,")
        print("then re-run this script.")


if __name__ == "__main__":
    main()
