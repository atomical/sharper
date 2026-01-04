#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.export.utils import preprocess_image  # noqa: E402


def _compute_quat_angle_stats(q_ref: np.ndarray, q_ml: np.ndarray) -> dict[str, float]:
    # q: [...,4] wxyz
    def _normalize(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        return q / np.clip(n, 1e-12, None)

    qr = _normalize(q_ref.astype(np.float64, copy=False))
    qm = _normalize(q_ml.astype(np.float64, copy=False))
    dot = np.abs(np.sum(qr * qm, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)  # radians, in [0, pi]
    return {
        "angle_rad_max": float(np.max(angle)),
        "angle_rad_mean": float(np.mean(angle)),
        "angle_deg_max": float(np.max(angle) * 180.0 / math.pi),
        "angle_deg_mean": float(np.mean(angle) * 180.0 / math.pi),
    }


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    # q: [N,4] float64, normalized, wxyz
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z
    yz = y * z

    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = ww + xx - yy - zz
    r[:, 0, 1] = 2.0 * (xy + wz)
    r[:, 0, 2] = 2.0 * (xz - wy)
    r[:, 1, 0] = 2.0 * (xy - wz)
    r[:, 1, 1] = ww - xx + yy - zz
    r[:, 1, 2] = 2.0 * (yz + wx)
    r[:, 2, 0] = 2.0 * (xz + wy)
    r[:, 2, 1] = 2.0 * (yz - wx)
    r[:, 2, 2] = ww - xx - yy + zz
    return r


def _cov_stats_from_quat_scale(
    q_ref: np.ndarray,
    s_ref: np.ndarray,
    q_ml: np.ndarray,
    s_ml: np.ndarray,
    sample_n: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    # Compare covariance matrices implied by (q, s): C = R diag(s^2) R^T.
    # This is much more stable than comparing raw quaternion angles when scales are near-degenerate.
    q_ref = q_ref.reshape(-1, 4).astype(np.float64, copy=False)
    q_ml = q_ml.reshape(-1, 4).astype(np.float64, copy=False)
    s_ref = s_ref.reshape(-1, 3).astype(np.float64, copy=False)
    s_ml = s_ml.reshape(-1, 3).astype(np.float64, copy=False)

    def _norm(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        return q / np.clip(n, 1e-12, None)

    q_ref = _norm(q_ref)
    q_ml = _norm(q_ml)

    # Sample indices for an approximate p99 (full max/mean computed exactly).
    rng = np.random.default_rng(seed)
    n = q_ref.shape[0]
    sample_n = int(min(sample_n, n))
    sample_idx = rng.choice(n, size=sample_n, replace=False) if sample_n > 0 else np.array([], dtype=np.int64)

    max_abs = 0.0
    sum_abs = 0.0
    count = 0

    # Process in chunks to keep memory bounded.
    chunk = 200_000
    for start in range(0, n, chunk):
        end = min(n, start + chunk)

        rr = _quat_to_rotmat_wxyz(q_ref[start:end])
        rm = _quat_to_rotmat_wxyz(q_ml[start:end])

        s2r = s_ref[start:end] ** 2
        s2m = s_ml[start:end] ** 2

        # R diag(s^2): scale columns of R.
        rr_scaled = rr * s2r[:, None, :]
        rm_scaled = rm * s2m[:, None, :]

        cr = rr_scaled @ np.transpose(rr, (0, 2, 1))
        cm = rm_scaled @ np.transpose(rm, (0, 2, 1))

        diff = np.abs(cr - cm)
        max_abs = max(max_abs, float(np.max(diff)))
        sum_abs += float(np.sum(diff))
        count += diff.size

    mean_abs = sum_abs / max(count, 1)

    # Approximate p99_abs using sampled indices.
    p99_abs = 0.0
    if sample_idx.size:
        rr = _quat_to_rotmat_wxyz(q_ref[sample_idx])
        rm = _quat_to_rotmat_wxyz(q_ml[sample_idx])
        rr_scaled = rr * (s_ref[sample_idx] ** 2)[:, None, :]
        rm_scaled = rm * (s_ml[sample_idx] ** 2)[:, None, :]
        cr = rr_scaled @ np.transpose(rr, (0, 2, 1))
        cm = rm_scaled @ np.transpose(rm, (0, 2, 1))
        diff = np.abs(cr - cm).reshape(-1)
        p99_abs = float(np.quantile(diff, 0.99))

    return {"max_abs": max_abs, "mean_abs": mean_abs, "p99_abs": p99_abs}


def _tensor_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    diff = a - b
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(a), 1e-12)
    rel = abs_diff / denom
    return {
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "p99_abs": float(np.quantile(abs_diff, 0.99)),
        "max_rel": float(np.max(rel)),
        "mean_rel": float(np.mean(rel)),
        "p99_rel": float(np.quantile(rel, 0.99)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Parity validation: PyTorch ref vs CoreML.")
    p.add_argument("--fixtures", type=Path, required=True, help="Fixture images directory.")
    p.add_argument("--ref-root", type=Path, required=True, help="Reference outputs root (ref_infer.py).")
    p.add_argument("--coreml-root", type=Path, required=True, help="CoreML outputs root.")
    p.add_argument("--model", type=Path, default=Path("artifacts/Sharp.mlpackage"))
    p.add_argument(
        "--compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="cpu_only",
    )
    p.add_argument(
        "--tol-mean-max-abs",
        type=float,
        default=1e-2,
        help="Max abs tolerance for mean_vectors_pre.",
    )
    p.add_argument(
        "--tol-default-max-abs",
        type=float,
        default=5e-3,
        help="Max abs tolerance for other float tensors (except quats).",
    )
    p.add_argument(
        "--tol-opacities-max-abs",
        type=float,
        default=3e-2,
        help="Max abs tolerance for opacities_pre.",
    )
    p.add_argument(
        "--tol-cov-max-abs",
        type=float,
        default=1e-4,
        help="Max abs tolerance for covariance implied by (quaternions_pre, singular_values_pre).",
    )
    args = p.parse_args()

    import coremltools as ct  # noqa: E402

    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[args.compute_units]

    mlmodel = ct.models.MLModel(str(args.model), compute_units=compute_units)

    exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in sorted(args.fixtures.glob("*")) if p.suffix.lower() in exts]
    if not image_paths:
        raise SystemExit(f"No fixture images found in {args.fixtures}")

    args.coreml_root.mkdir(parents=True, exist_ok=True)

    overall: dict[str, Any] = {
        "model": str(args.model),
        "compute_units": args.compute_units,
        "cases": [],
        "tolerances": {
            "mean_vectors_pre_max_abs": args.tol_mean_max_abs,
            "default_max_abs": args.tol_default_max_abs,
            "opacities_pre_max_abs": args.tol_opacities_max_abs,
            "covariance_max_abs": args.tol_cov_max_abs,
        },
    }

    failed = False

    for image_path in image_paths:
        case = image_path.stem
        ref_npz = args.ref_root / case / "raw_outputs.npz"
        if not ref_npz.exists():
            raise SystemExit(f"Missing reference outputs: {ref_npz}")

        ref = dict(np.load(ref_npz))

        # Prepare CoreML inputs using the exact same preprocessing as the reference runner.
        import torch  # noqa: E402

        pre = preprocess_image(image_path, device=torch.device("cpu"))
        inputs = {
            "image": pre.resized_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
            "disparity_factor": pre.disparity_factor.detach().cpu().numpy().astype(np.float32, copy=False),
        }

        out = mlmodel.predict(inputs)

        case_dir = args.coreml_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez(case_dir / "raw_outputs.npz", **{k: np.asarray(v) for k, v in out.items()})

        # Compare.
        comparisons: dict[str, Any] = {}

        def _cmp(key: str, tol_max_abs: float):
            nonlocal failed
            if key not in ref:
                raise SystemExit(f"Reference key missing: {key} in {ref_npz}")
            if key not in out:
                raise SystemExit(f"CoreML output key missing: {key} in model outputs")
            a = ref[key]
            b = np.asarray(out[key])
            if a.shape != b.shape:
                failed = True
                comparisons[key] = {"shape_ref": list(a.shape), "shape_coreml": list(b.shape), "error": "shape_mismatch"}
                return
            stats = _tensor_error(a, b)
            stats["shape"] = list(a.shape)
            stats["dtype_ref"] = str(a.dtype)
            stats["dtype_coreml"] = str(b.dtype)
            stats["pass_max_abs"] = bool(stats["max_abs"] <= tol_max_abs)
            comparisons[key] = stats
            if not stats["pass_max_abs"]:
                failed = True

        _cmp("mean_vectors_pre", args.tol_mean_max_abs)
        _cmp("singular_values_pre", args.tol_default_max_abs)
        _cmp("colors_linear_pre", args.tol_default_max_abs)
        _cmp("opacities_pre", args.tol_opacities_max_abs)

        # Quaternions: report angle stats for visibility (not a hard gate because of scale degeneracy).
        if "quaternions_pre" not in ref or "quaternions_pre" not in out:
            raise SystemExit("Missing quaternions_pre in ref/coreml outputs.")
        q_stats = _compute_quat_angle_stats(
            ref["quaternions_pre"],
            np.asarray(out["quaternions_pre"]),
        )
        q_stats["note"] = "Quaternion angles can be unstable when singular values are near-degenerate; covariance comparison is gated instead."
        comparisons["quaternions_pre"] = q_stats

        # Gate on covariance implied by (quat, scale), which reflects actual ellipsoid geometry.
        cov_stats = _cov_stats_from_quat_scale(
            ref["quaternions_pre"],
            ref["singular_values_pre"],
            np.asarray(out["quaternions_pre"]),
            np.asarray(out["singular_values_pre"]),
        )
        cov_stats["pass_max_abs"] = bool(cov_stats["max_abs"] <= args.tol_cov_max_abs)
        comparisons["covariance_pre"] = cov_stats
        if not cov_stats["pass_max_abs"]:
            failed = True

        overall["cases"].append(
            {
                "case": case,
                "image": str(image_path),
                "comparisons": comparisons,
            }
        )

    report_json = args.coreml_root / "parity_report.json"
    report_md = args.coreml_root / "parity_report.md"
    report_json.write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Minimal markdown summary.
    lines = ["# CoreML Parity Report", "", f"Model: `{args.model}`", f"Compute units: `{args.compute_units}`", ""]
    for case_entry in overall["cases"]:
        lines.append(f"## {case_entry['case']}")
        comps = case_entry["comparisons"]
        for k in ["mean_vectors_pre", "singular_values_pre", "colors_linear_pre", "opacities_pre"]:
            s = comps.get(k, {})
            if "pass_max_abs" in s:
                lines.append(f"- `{k}` max_abs={s['max_abs']:.3e} p99_abs={s['p99_abs']:.3e} pass={s['pass_max_abs']}")
        q = comps.get("quaternions_pre", {})
        if "angle_deg_max" in q:
            lines.append(f"- `quaternions_pre` max_deg={q['angle_deg_max']:.3f} mean_deg={q['angle_deg_mean']:.3f} (informational)")
        cov = comps.get("covariance_pre", {})
        if "max_abs" in cov:
            lines.append(
                f"- `covariance_pre` max_abs={cov['max_abs']:.3e} p99_abs≈{cov['p99_abs']:.3e} pass={cov['pass_max_abs']}"
            )
        lines.append("")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failed:
        print(f"FAIL: parity exceeded tolerances; see {report_md}")
        return 1
    print(f"PASS: parity within tolerances; see {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
