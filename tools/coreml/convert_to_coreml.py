#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Convert traced SHARP TorchScript to CoreML .mlpackage.")
    p.add_argument(
        "--traced",
        type=Path,
        default=Path("artifacts/Sharp_traced.pt"),
        help="Path to traced TorchScript model.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/Sharp.mlpackage"),
        help="Output .mlpackage path.",
    )
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Compute precision (start with fp32 for parity).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable coremltools debug mode (more logs, slower).",
    )
    args = p.parse_args()

    import coremltools as ct  # noqa: E402

    traced = torch.jit.load(str(args.traced), map_location="cpu").eval()

    inputs = [
        ct.TensorType(name="image", shape=(1, 3, 1536, 1536), dtype=np.float32),
        ct.TensorType(name="disparity_factor", shape=(1,), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="mean_vectors_pre", dtype=np.float32),
        ct.TensorType(name="quaternions_pre", dtype=np.float32),
        ct.TensorType(name="singular_values_pre", dtype=np.float32),
        ct.TensorType(name="colors_linear_pre", dtype=np.float32),
        ct.TensorType(name="opacities_pre", dtype=np.float32),
    ]

    compute_precision = ct.precision.FLOAT32 if args.precision == "fp32" else ct.precision.FLOAT16

    args.out.parent.mkdir(parents=True, exist_ok=True)

    mlmodel = ct.convert(
        traced,
        source="pytorch",
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.ALL,
        debug=args.debug,
    )
    mlmodel.save(str(args.out))

    report = {
        "coremltools_version": ct.__version__,
        "torch_version": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "convert_to": "mlprogram",
        "precision": args.precision,
        "inputs": [{"name": i.name, "shape": str(i.shape), "dtype": str(i.dtype)} for i in inputs],
        "outputs": [{"name": o.name, "dtype": str(o.dtype)} for o in outputs],
        "traced_path": str(args.traced),
        "out_path": str(args.out),
    }
    report_path = Path("artifacts") / "coreml_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.precision == "fp16":
        (Path("artifacts") / "coreml_conversion_report_fp16.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
