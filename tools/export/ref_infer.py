#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.export.utils import (
    DEFAULT_INTERNAL_RESOLUTION,
    DEFAULT_MODEL_URL,
    dump_json,
    dump_npz,
    load_predictor,
    load_state_dict,
    preprocess_image,
    resolve_device,
    run_predictor,
    set_determinism,
    sigmoid_logit,
    unproject_like_cli,
    write_ply,
)


def _iter_images(fixtures_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    paths = [p for p in sorted(fixtures_dir.glob("*")) if p.suffix.lower() in exts]
    if not paths:
        raise SystemExit(f"No supported images found in {fixtures_dir}")
    return paths


def _save_case(
    out_dir: Path,
    case_name: str,
    pre,
    gaussians_pre,
    gaussians_post,
    intermediates: dict[str, torch.Tensor],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw outputs used for parity tests later.
    arrays: dict[str, np.ndarray] = {
        "mean_vectors_pre": gaussians_pre.mean_vectors.detach().cpu().numpy().astype(np.float32, copy=False),
        "singular_values_pre": gaussians_pre.singular_values.detach().cpu().numpy().astype(np.float32, copy=False),
        "quaternions_pre": gaussians_pre.quaternions.detach().cpu().numpy().astype(np.float32, copy=False),
        "colors_linear_pre": gaussians_pre.colors.detach().cpu().numpy().astype(np.float32, copy=False),
        "opacities_pre": gaussians_pre.opacities.detach().cpu().numpy().astype(np.float32, copy=False),
        "mean_vectors": gaussians_post.mean_vectors.detach().cpu().numpy().astype(np.float32, copy=False),
        "singular_values": gaussians_post.singular_values.detach().cpu().numpy().astype(np.float32, copy=False),
        "quaternions": gaussians_post.quaternions.detach().cpu().numpy().astype(np.float32, copy=False),
        "colors_linear": gaussians_post.colors.detach().cpu().numpy().astype(np.float32, copy=False),
        "opacities": gaussians_post.opacities.detach().cpu().numpy().astype(np.float32, copy=False),
        "scale_logits": torch.log(gaussians_post.singular_values)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "opacity_logits": sigmoid_logit(gaussians_post.opacities)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "disparity_factor": pre.disparity_factor.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    dump_npz(out_dir / "raw_outputs.npz", arrays)

    if intermediates:
        inter_np: dict[str, np.ndarray] = {}
        for k, v in sorted(intermediates.items()):
            inter_np[k] = v.detach().cpu().numpy()
        dump_npz(out_dir / "intermediates.npz", inter_np)

    # PLY output matching ml-sharp semantics.
    write_ply(out_dir / "scene.ply", gaussians_post, pre.f_px, pre.original_hw)

    dump_json(
        out_dir / "meta.json",
        {
            "case": case_name,
            "image_path": str(pre.image_path),
            "original_hw": list(pre.original_hw),
            "internal_hw": list(pre.internal_hw),
            "f_px": pre.f_px,
            "model_url": DEFAULT_MODEL_URL,
            "notes": [
                "raw_outputs.npz includes both pre- and post-unprojection tensors.",
                "scene.ply matches sharp.cli.predict semantics (sRGB export + metadata elements).",
            ],
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Deterministic PyTorch reference inference for SHARP.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path, help="Single input image.")
    g.add_argument("--fixtures", type=Path, help="Directory of fixture images (flat).")

    p.add_argument("--out", type=Path, help="Output folder for --image mode.")
    p.add_argument("--out-root", type=Path, help="Output root folder for --fixtures mode.")
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Optional local .pt checkpoint.")
    p.add_argument("--model-url", type=str, default=DEFAULT_MODEL_URL, help="Checkpoint URL (default: official).")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (default: cpu for determinism). One of: ['cpu','mps','cuda','default']",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument(
        "--dump-intermediates",
        action="store_true",
        help="Dump intermediate tensors for debugging CoreML mismatches (large).",
    )
    p.add_argument(
        "--internal-resolution",
        type=int,
        default=DEFAULT_INTERNAL_RESOLUTION,
        help="Internal inference resolution (default: 1536).",
    )
    args = p.parse_args()

    set_determinism(args.seed)
    device = resolve_device(args.device)

    if args.image is not None and args.out is None:
        raise SystemExit("--out is required when using --image")
    if args.fixtures is not None and args.out_root is None:
        raise SystemExit("--out-root is required when using --fixtures")

    state_dict = load_state_dict(args.checkpoint_path, model_url=args.model_url)
    predictor = load_predictor(state_dict, device)

    image_paths = [args.image] if args.image is not None else _iter_images(args.fixtures)

    for image_path in image_paths:
        case_name = image_path.stem
        out_dir = args.out if args.out is not None else (args.out_root / case_name)

        pre = preprocess_image(image_path, device=device, internal_resolution=args.internal_resolution)
        gaussians_pre, intermediates = run_predictor(
            predictor,
            pre.resized_tensor,
            pre.disparity_factor,
            dump_intermediates=args.dump_intermediates,
        )
        gaussians_post = unproject_like_cli(gaussians_pre, pre.f_px, pre.original_hw, pre.internal_hw)

        _save_case(out_dir, case_name, pre, gaussians_pre, gaussians_post, intermediates)

        # Clear intermediate tensors between cases.
        if device.type == "mps":
            torch.mps.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
