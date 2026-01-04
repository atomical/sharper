from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_INTERNAL_RESOLUTION = 1536


def set_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some backends (e.g. MPS) may not fully support this.
        pass


def resolve_device(device: str) -> torch.device:
    device = device.lower().strip()
    if device == "default":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_state_dict(checkpoint_path: Path | None, model_url: str = DEFAULT_MODEL_URL) -> dict[str, Any]:
    if checkpoint_path is None:
        return torch.hub.load_state_dict_from_url(model_url, progress=True)
    return torch.load(checkpoint_path, weights_only=True)


def load_predictor(state_dict: dict[str, Any], device: torch.device) -> RGBGaussianPredictor:
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


@dataclass(frozen=True)
class PreprocessResult:
    image_path: Path
    image_rgb_u8: np.ndarray  # HWC, uint8
    f_px: float
    resized_tensor: torch.Tensor  # 1x3xH'xW', float32 in [0,1]
    disparity_factor: torch.Tensor  # 1, float32
    original_hw: tuple[int, int]  # (H, W)
    internal_hw: tuple[int, int]  # (H', W')


def preprocess_image(
    image_path: Path,
    device: torch.device,
    internal_resolution: int = DEFAULT_INTERNAL_RESOLUTION,
) -> PreprocessResult:
    image_rgb_u8, _, f_px = sharp_io.load_rgb(image_path)
    height, width = image_rgb_u8.shape[:2]

    image_pt = torch.from_numpy(image_rgb_u8.copy()).to(dtype=torch.float32, device=device)
    image_pt = image_pt.permute(2, 0, 1) / 255.0

    internal_hw = (internal_resolution, internal_resolution)
    resized = F.interpolate(
        image_pt[None],
        size=internal_hw,
        mode="bilinear",
        align_corners=True,
    )

    disparity_factor = torch.tensor([f_px / width], device=device, dtype=torch.float32)
    return PreprocessResult(
        image_path=image_path,
        image_rgb_u8=image_rgb_u8,
        f_px=float(f_px),
        resized_tensor=resized,
        disparity_factor=disparity_factor,
        original_hw=(height, width),
        internal_hw=internal_hw,
    )


@torch.no_grad()
def run_predictor(
    predictor: RGBGaussianPredictor,
    resized_tensor: torch.Tensor,
    disparity_factor: torch.Tensor,
    dump_intermediates: bool = False,
) -> tuple[Gaussians3D, dict[str, torch.Tensor]]:
    if not dump_intermediates:
        gaussians = predictor(resized_tensor, disparity_factor)
        return gaussians, {}

    # Manual unrolling of forward() to capture intermediates with stable names.
    intermediates: dict[str, torch.Tensor] = {}
    monodepth_output = predictor.monodepth_model(resized_tensor)
    intermediates["monodepth_disparity"] = monodepth_output.disparity

    # NOTE: monodepth_output.output_features is a list of feature maps.
    for i, f in enumerate(monodepth_output.output_features):
        intermediates[f"monodepth_output_features_{i}"] = f
    intermediates["monodepth_decoder_features"] = monodepth_output.decoder_features

    disparity_factor_4d = disparity_factor[:, None, None, None]
    monodepth = disparity_factor_4d / monodepth_output.disparity.clamp(min=1e-4, max=1e4)
    intermediates["monodepth_metric_depth"] = monodepth

    monodepth_aligned, depth_alignment_map = predictor.depth_alignment(
        monodepth,
        None,
        monodepth_output.decoder_features,
    )
    intermediates["monodepth_aligned"] = monodepth_aligned
    intermediates["depth_alignment_map"] = depth_alignment_map

    init_output = predictor.init_model(resized_tensor, monodepth_aligned)
    intermediates["init_feature_input"] = init_output.feature_input
    if init_output.global_scale is not None:
        intermediates["init_global_scale"] = init_output.global_scale

    base = init_output.gaussian_base_values
    intermediates["base_mean_x_ndc"] = base.mean_x_ndc
    intermediates["base_mean_y_ndc"] = base.mean_y_ndc
    intermediates["base_mean_inverse_z_ndc"] = base.mean_inverse_z_ndc
    intermediates["base_scales"] = base.scales
    intermediates["base_quaternions"] = base.quaternions
    intermediates["base_colors"] = base.colors
    intermediates["base_opacities"] = base.opacities

    image_features = predictor.feature_model(init_output.feature_input, encodings=monodepth_output.output_features)
    intermediates["image_features"] = image_features

    delta_values = predictor.prediction_head(image_features)
    intermediates["delta_values"] = delta_values

    gaussians = predictor.gaussian_composer(
        delta=delta_values,
        base_values=base,
        global_scale=init_output.global_scale,
    )
    return gaussians, intermediates


@torch.no_grad()
def unproject_like_cli(gaussians: Gaussians3D, f_px: float, original_hw: tuple[int, int], internal_hw: tuple[int, int]) -> Gaussians3D:
    height, width = original_hw
    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=gaussians.mean_vectors.device,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_hw[1] / width
    intrinsics_resized[1] *= internal_hw[0] / height

    return unproject_gaussians(gaussians, torch.eye(4, device=intrinsics.device), intrinsics_resized, internal_hw)


def sigmoid_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def dump_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.no_grad()
def write_ply(path: Path, gaussians: Gaussians3D, f_px: float, original_hw: tuple[int, int]) -> None:
    save_ply(gaussians, f_px, original_hw, path)

