#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Run Sharp.mlpackage and dump raw outputs to .npz.")
    p.add_argument("--model", type=Path, default=Path("artifacts/Sharp.mlpackage"))
    p.add_argument(
        "--input-npz",
        type=Path,
        default=Path("artifacts/io_sample_inputs.npz"),
        help="NPZ containing `image` and `disparity_factor` arrays.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    args = p.parse_args()

    import coremltools as ct  # noqa: E402

    inputs = dict(np.load(args.input_npz))
    if "image" not in inputs or "disparity_factor" not in inputs:
        raise SystemExit(f"{args.input_npz} must contain keys: image, disparity_factor")

    mlmodel = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.ALL)
    out = mlmodel.predict(
        {
            "image": inputs["image"].astype(np.float32, copy=False),
            "disparity_factor": inputs["disparity_factor"].astype(np.float32, copy=False),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # coremltools returns dict[str, np.ndarray]
    np.savez(args.out, **{k: np.asarray(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

