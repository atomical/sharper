#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.export.utils import (  # noqa: E402
    DEFAULT_INTERNAL_RESOLUTION,
    DEFAULT_MODEL_URL,
    dump_npz,
    load_predictor,
    load_state_dict,
    preprocess_image,
    resolve_device,
    set_determinism,
)


class SharpExportWrapper(torch.nn.Module):
    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(self, image: torch.Tensor, disparity_factor: torch.Tensor):
        gaussians = self.predictor(image, disparity_factor)
        return (
            gaussians.mean_vectors.contiguous(),
            gaussians.quaternions.contiguous(),
            gaussians.singular_values.contiguous(),
            gaussians.colors.contiguous(),
            gaussians.opacities.contiguous(),
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Export a CoreML-friendly SHARP Torch graph.")
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local .pt checkpoint (default: official URL).",
    )
    p.add_argument("--model-url", type=str, default=DEFAULT_MODEL_URL, help="Checkpoint URL.")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Export device (default: cpu).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample-image",
        type=Path,
        default=Path("artifacts/fixtures/inputs/indoor_teaser.jpg"),
        help="Sample image to drive tracing and save I/O samples.",
    )
    p.add_argument(
        "--internal-resolution",
        type=int,
        default=DEFAULT_INTERNAL_RESOLUTION,
        help="Internal inference resolution (default: 1536).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for exported artifacts.",
    )
    args = p.parse_args()

    set_determinism(args.seed)
    device = resolve_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    state_dict = load_state_dict(args.checkpoint_path, model_url=args.model_url)
    predictor = load_predictor(state_dict, device)
    wrapper = SharpExportWrapper(predictor).eval()

    # Prepare sample inputs.
    pre = preprocess_image(args.sample_image, device=device, internal_resolution=args.internal_resolution)
    sample_inputs = {
        "image": pre.resized_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
        "disparity_factor": pre.disparity_factor.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    dump_npz(args.out_dir / "io_sample_inputs.npz", sample_inputs)

    with torch.no_grad():
        out = wrapper(pre.resized_tensor, pre.disparity_factor)
    sample_outputs = {
        "mean_vectors_pre": out[0].detach().cpu().numpy().astype(np.float32, copy=False),
        "quaternions_pre": out[1].detach().cpu().numpy().astype(np.float32, copy=False),
        "singular_values_pre": out[2].detach().cpu().numpy().astype(np.float32, copy=False),
        "colors_linear_pre": out[3].detach().cpu().numpy().astype(np.float32, copy=False),
        "opacities_pre": out[4].detach().cpu().numpy().astype(np.float32, copy=False),
    }
    dump_npz(args.out_dir / "io_sample_outputs_ref.npz", sample_outputs)

    # Trace wrapper (fixed shapes).
    example_inputs = (pre.resized_tensor, pre.disparity_factor)
    traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    traced = torch.jit.freeze(traced.eval())
    traced.save(str(args.out_dir / "Sharp_traced.pt"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

