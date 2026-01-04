#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    # Matches sharp.utils.color_space.linearRGB2sRGB (with clamped false branch).
    thr = 0.0031308
    linear = linear.astype(np.float32, copy=False)
    linear_clamped = np.maximum(linear, thr)
    a = linear * 12.92
    b = 1.055 * np.power(linear_clamped, 1.0 / 2.4) - 0.055
    return np.where(linear <= thr, a, b)


def _rgb_to_sh0(rgb_srgb: np.ndarray) -> np.ndarray:
    coeff = math.sqrt(1.0 / (4.0 * math.pi))
    return (rgb_srgb - 0.5) / coeff


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def write_ply_ml_sharp(
    path: Path,
    mean_vectors: np.ndarray,  # [N,3]
    quaternions_wxyz: np.ndarray,  # [N,4]
    singular_values: np.ndarray,  # [N,3]
    colors_linear: np.ndarray,  # [N,3]
    opacities: np.ndarray,  # [N]
    f_px: float,
    image_hw: tuple[int, int],  # (H,W)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    mean_vectors = mean_vectors.astype(np.float32, copy=False)
    quaternions_wxyz = quaternions_wxyz.astype(np.float32, copy=False)
    singular_values = singular_values.astype(np.float32, copy=False)
    colors_linear = colors_linear.astype(np.float32, copy=False)
    opacities = opacities.astype(np.float32, copy=False)

    xyz = mean_vectors
    scale_logits = np.log(np.clip(singular_values, 1e-20, None))

    colors_srgb = _linear_to_srgb(colors_linear)
    f_dc = _rgb_to_sh0(colors_srgb)
    opacity_logits = _logit(opacities)[:, None]

    attributes = np.concatenate([xyz, f_dc, opacity_logits, scale_logits, quaternions_wxyz], axis=1)
    if attributes.shape[1] != 14:
        raise ValueError(f"Expected 14 attributes per gaussian, got {attributes.shape[1]}")

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = xyz.shape[0]
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    vertex_elements = PlyElement.describe(elements, "vertex")

    image_h, image_w = image_hw

    # image_size
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_w, image_h], dtype=np.uint32)
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    # intrinsic
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_w * 0.5,
            0,
            f_px,
            image_h * 0.5,
            0,
            0,
            1,
        ],
        dtype=np.float32,
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    # extrinsic (identity)
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4, dtype=np.float32).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    # frame
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # disparity quantiles
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)
    disparity = 1.0 / np.clip(xyz[:, 2], 1e-8, None)
    disparity_array[:] = np.quantile(disparity.astype(np.float64), [0.1, 0.9]).astype(np.float32)
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    # color_space (export uses sRGB index = 0)
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([0], dtype=np.uint8)
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    # version
    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8)
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )
    plydata.write(path)


def _load_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Write an ml-sharp-compatible PLY from raw tensor outputs.")
    p.add_argument("--npz", type=Path, required=True, help="NPZ with gaussian tensors.")
    p.add_argument("--out", type=Path, required=True, help="Output .ply path.")
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional meta.json (from tools/export/ref_infer.py) to supply f_px and image_hw.",
    )
    p.add_argument("--f-px", type=float, default=None, help="Override focal length in pixels.")
    p.add_argument("--image-h", type=int, default=None, help="Override image height.")
    p.add_argument("--image-w", type=int, default=None, help="Override image width.")
    p.add_argument(
        "--use-pre",
        action="store_true",
        help="Use *_pre keys if present (otherwise uses post-unprojection keys).",
    )
    args = p.parse_args()

    data = dict(np.load(args.npz))

    meta: dict[str, Any] = {}
    if args.meta is not None:
        meta = _load_meta(args.meta)

    f_px = args.f_px if args.f_px is not None else meta.get("f_px", None)
    if f_px is None:
        raise SystemExit("Need --f-px or --meta with f_px.")

    original_hw = meta.get("original_hw", None)
    image_h = args.image_h if args.image_h is not None else (original_hw[0] if original_hw else None)
    image_w = args.image_w if args.image_w is not None else (original_hw[1] if original_hw else None)
    if image_h is None or image_w is None:
        raise SystemExit("Need --image-h/--image-w or --meta with original_hw.")

    suffix = "_pre" if args.use_pre else ""
    mean_key = "mean_vectors" + suffix
    quat_key = "quaternions" + suffix
    scale_key = "singular_values" + suffix
    color_key = ("colors_linear" + suffix) if ("colors_linear" + suffix) in data else ("colors_linear_pre" if args.use_pre else "colors_linear")
    opa_key = "opacities" + suffix

    # Support the naming used by export scripts.
    if mean_key not in data and args.use_pre:
        mean_key = "mean_vectors_pre"
        quat_key = "quaternions_pre"
        scale_key = "singular_values_pre"
        color_key = "colors_linear_pre"
        opa_key = "opacities_pre"

    for k in [mean_key, quat_key, scale_key, color_key, opa_key]:
        if k not in data:
            raise SystemExit(f"Missing key {k} in {args.npz}")

    mean = data[mean_key].reshape(-1, 3)
    quat = data[quat_key].reshape(-1, 4)
    scale = data[scale_key].reshape(-1, 3)
    color = data[color_key].reshape(-1, 3)
    opa = data[opa_key].reshape(-1)

    write_ply_ml_sharp(
        args.out,
        mean_vectors=mean,
        quaternions_wxyz=quat,
        singular_values=scale,
        colors_linear=color,
        opacities=opa,
        f_px=float(f_px),
        image_hw=(int(image_h), int(image_w)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

