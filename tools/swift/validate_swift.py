#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PLYPayload:
    vertices: np.ndarray  # (N, 14) float32
    extrinsic: np.ndarray  # (16,) float32
    intrinsic: np.ndarray  # (9,) float32
    image_size: np.ndarray  # (2,) uint32
    frame: np.ndarray  # (2,) int32
    disparity: np.ndarray  # (2,) float32
    color_space: int  # uint8
    version: tuple[int, int, int]  # uint8[3]


def _read_mlsharp_ply(path: Path) -> PLYPayload:
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError(f"{path}: missing end_header\\n")
    header_end += len(b"end_header\n")

    header = data[:header_end].decode("utf-8", errors="replace")
    vertex_count: int | None = None
    for line in header.splitlines():
        if line.startswith("element vertex "):
            parts = line.split()
            if len(parts) == 3:
                vertex_count = int(parts[2])
    if vertex_count is None:
        raise ValueError(f"{path}: missing vertex count in header")

    n = vertex_count
    off = header_end

    vtx_bytes = n * 14 * 4
    if len(data) < off + vtx_bytes + (16 + 9 + 2 + 2 + 2) * 4 + 1 + 3:
        raise ValueError(f"{path}: truncated payload")

    vertices = np.frombuffer(data, dtype="<f4", count=n * 14, offset=off).reshape(n, 14)
    off += vtx_bytes

    extrinsic = np.frombuffer(data, dtype="<f4", count=16, offset=off)
    off += 16 * 4

    intrinsic = np.frombuffer(data, dtype="<f4", count=9, offset=off)
    off += 9 * 4

    image_size = np.frombuffer(data, dtype="<u4", count=2, offset=off)
    off += 2 * 4

    frame = np.frombuffer(data, dtype="<i4", count=2, offset=off)
    off += 2 * 4

    disparity = np.frombuffer(data, dtype="<f4", count=2, offset=off)
    off += 2 * 4

    color_space = int(data[off])
    off += 1

    version = tuple(int(x) for x in data[off : off + 3])
    return PLYPayload(
        vertices=vertices,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_size=image_size,
        frame=frame,
        disparity=disparity,
        color_space=color_space,
        version=version,  # type: ignore[arg-type]
    )


def _quat_to_rot_cols_wxyz(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # q: (..., 4) (w, x, y, z)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # Normalize defensively.
    inv_norm = 1.0 / np.sqrt(np.maximum(w * w + x * x + y * y + z * z, 1e-12))
    w = w * inv_norm
    x = x * inv_norm
    y = y * inv_norm
    z = z * inv_norm

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    wx = w * x
    wy = w * y
    wz = w * z

    c0 = np.stack([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], axis=-1)
    c1 = np.stack([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], axis=-1)
    c2 = np.stack([2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)], axis=-1)
    return c0, c1, c2


def _cov_from_quat_scale_logs_wxyz(quat_wxyz: np.ndarray, scale_logs: np.ndarray) -> np.ndarray:
    # quat_wxyz: (K,4), scale_logs: (K,3) storing log(sigma)
    sigma2 = np.exp(2.0 * scale_logs).astype(np.float64)
    c0, c1, c2 = _quat_to_rot_cols_wxyz(quat_wxyz.astype(np.float64))

    # Σ = Σ_i sigma2_i * c_i c_i^T
    cov = np.zeros((quat_wxyz.shape[0], 3, 3), dtype=np.float64)
    for s2, c in [(sigma2[:, 0], c0), (sigma2[:, 1], c1), (sigma2[:, 2], c2)]:
        cov += s2[:, None, None] * (c[:, :, None] * c[:, None, :])
    return cov


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a64 = a.astype(np.float64).reshape(-1)
    b64 = b.astype(np.float64).reshape(-1)
    finite = np.isfinite(a64) & np.isfinite(b64)
    if not np.any(finite):
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p99_abs": 0.0,
            "max_rel": 0.0,
            "mean_rel": 0.0,
            "p99_rel": 0.0,
        }

    diff = a64[finite] - b64[finite]
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(b64[finite]), 1e-12)
    rel = abs_diff / denom
    return {
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "p99_abs": float(np.quantile(abs_diff, 0.99)) if abs_diff.size else 0.0,
        "max_rel": float(rel.max(initial=0.0)),
        "mean_rel": float(rel.mean()) if rel.size else 0.0,
        "p99_rel": float(np.quantile(rel, 0.99)) if rel.size else 0.0,
    }


def _run_swift_predict(swift_bin: Path, model_path: Path, image_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "scene.ply"
    cmd = [
        str(swift_bin),
        str(image_path),
        str(out_dir),
        "--model",
        str(model_path),
        "--no-render",
    ]
    subprocess.run(cmd, check=True)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    return ply_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Swift SharpDemoApp PLY vs PyTorch reference PLY.")
    ap.add_argument("--fixtures", type=Path, default=Path("artifacts/fixtures/inputs"))
    ap.add_argument("--ref-root", type=Path, default=Path("artifacts/fixtures/ref"))
    ap.add_argument("--swift-root", type=Path, default=Path("artifacts/fixtures/coreml/swift_validate"))
    ap.add_argument("--swift-bin", type=Path, default=Path("Swift/SharpDemoApp/.build/release/SharpDemoApp"))
    ap.add_argument("--model", type=Path, default=Path("artifacts/Sharp.mlpackage"))
    ap.add_argument("--sample", type=int, default=8192, help="Number of gaussians to sample per case.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-json", type=Path, default=Path("artifacts/fixtures/coreml/swift_validate_report.json"))
    args = ap.parse_args()

    fixtures = sorted([p for p in args.fixtures.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not fixtures:
        print(f"no fixtures found in {args.fixtures}", file=sys.stderr)
        return 2

    if not args.swift_bin.exists():
        raise FileNotFoundError(
            f"{args.swift_bin} not found; build with: (cd Swift/SharpDemoApp && swift build -c release)"
        )
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} not found; build with: make coreml")

    rng = np.random.default_rng(args.seed)
    report: dict[str, object] = {"cases": {}, "passed": True}

    # Tolerances: tuned for float32 CoreML + Metal postprocess.
    tol = {
        "mean_max_abs": 2.5e-1,
        "mean_p99_abs": 3e-2,
        "fdc_max_abs": 2e-1,
        "fdc_p99_abs": 5e-2,
        "opacity_max_abs": 1.2e1,
        "opacity_p99_abs": 2.0,
        "scale_max_abs": 3.0,
        "scale_p99_abs": 9e-1,
        "cov_max_abs": 2e-4,
        "cov_p99_abs": 5e-6,
    }

    for img_path in fixtures:
        case = img_path.stem
        ref_ply = args.ref_root / case / "scene.ply"
        if not ref_ply.exists():
            print(f"skip {case}: missing ref ply {ref_ply}", file=sys.stderr)
            continue

        swift_out = args.swift_root / case
        swift_ply = _run_swift_predict(args.swift_bin, args.model, img_path, swift_out)

        ref = _read_mlsharp_ply(ref_ply)
        got = _read_mlsharp_ply(swift_ply)

        if ref.vertices.shape != got.vertices.shape:
            raise ValueError(f"{case}: vertex shape mismatch {ref.vertices.shape} vs {got.vertices.shape}")

        n = ref.vertices.shape[0]
        k = min(args.sample, n)
        idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)

        ref_v = ref.vertices[idx]
        got_v = got.vertices[idx]

        mean_ref = ref_v[:, 0:3]
        mean_got = got_v[:, 0:3]
        fdc_ref = ref_v[:, 3:6]
        fdc_got = got_v[:, 3:6]
        opa_ref = ref_v[:, 6]
        opa_got = got_v[:, 6]
        scale_ref = ref_v[:, 7:10]
        scale_got = got_v[:, 7:10]
        quat_ref = ref_v[:, 10:14]
        quat_got = got_v[:, 10:14]

        cov_ref = _cov_from_quat_scale_logs_wxyz(quat_ref, scale_ref)
        cov_got = _cov_from_quat_scale_logs_wxyz(quat_got, scale_got)
        cov_m = _metrics(cov_got, cov_ref)

        case_report = {
            "counts": {"n": int(n), "sample": int(k)},
            "meta": {
                "color_space_ref": ref.color_space,
                "color_space_got": got.color_space,
                "version_ref": ref.version,
                "version_got": got.version,
                "image_size_ref": ref.image_size.tolist(),
                "image_size_got": got.image_size.tolist(),
            },
            "metrics": {
                "mean_xyz": _metrics(mean_got, mean_ref),
                "f_dc": _metrics(fdc_got, fdc_ref),
                "opacity_logit": _metrics(opa_got, opa_ref),
                "scale_logs": _metrics(scale_got, scale_ref),
                "covariance": cov_m,
            },
            "passed": True,
        }

        def fail(msg: str) -> None:
            case_report["passed"] = False
            report["passed"] = False
            print(f"{case}: FAIL {msg}", file=sys.stderr)

        m_mean = case_report["metrics"]["mean_xyz"]
        if m_mean["max_abs"] > tol["mean_max_abs"] or m_mean["p99_abs"] > tol["mean_p99_abs"]:
            fail(f"mean_xyz abs too large (max {m_mean['max_abs']:.3g}, p99 {m_mean['p99_abs']:.3g})")

        m_fdc = case_report["metrics"]["f_dc"]
        if m_fdc["max_abs"] > tol["fdc_max_abs"] or m_fdc["p99_abs"] > tol["fdc_p99_abs"]:
            fail(f"f_dc abs too large (max {m_fdc['max_abs']:.3g}, p99 {m_fdc['p99_abs']:.3g})")

        m_opa = case_report["metrics"]["opacity_logit"]
        if m_opa["max_abs"] > tol["opacity_max_abs"] or m_opa["p99_abs"] > tol["opacity_p99_abs"]:
            fail(f"opacity_logit abs too large (max {m_opa['max_abs']:.3g}, p99 {m_opa['p99_abs']:.3g})")

        m_scale = case_report["metrics"]["scale_logs"]
        if m_scale["max_abs"] > tol["scale_max_abs"] or m_scale["p99_abs"] > tol["scale_p99_abs"]:
            fail(f"scale_logs abs too large (max {m_scale['max_abs']:.3g}, p99 {m_scale['p99_abs']:.3g})")

        if cov_m["max_abs"] > tol["cov_max_abs"] or cov_m["p99_abs"] > tol["cov_p99_abs"]:
            fail(f"covariance abs too large (max {cov_m['max_abs']:.3g}, p99 {cov_m['p99_abs']:.3g})")

        if case_report["passed"]:
            print(f"{case}: ok (sample={k})")
        report["cases"][case] = case_report

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.report_json}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
