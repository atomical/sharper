#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil


def main() -> int:
    p = argparse.ArgumentParser(description="Simple CoreML benchmark harness for Sharp.mlpackage.")
    p.add_argument("--model", type=Path, default=Path("artifacts/Sharp.mlpackage"))
    p.add_argument("--input-npz", type=Path, default=Path("artifacts/io_sample_inputs.npz"))
    p.add_argument("--iters", type=int, default=5)
    p.add_argument(
        "--compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/benches/bench_coreml.json"))
    args = p.parse_args()

    import coremltools as ct  # noqa: E402

    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[args.compute_units]

    inputs = dict(np.load(args.input_npz))
    feed = {
        "image": inputs["image"].astype(np.float32, copy=False),
        "disparity_factor": inputs["disparity_factor"].astype(np.float32, copy=False),
    }

    mlmodel = ct.models.MLModel(str(args.model), compute_units=compute_units)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    # Warmup.
    mlmodel.predict(feed)

    times = []
    rss_peaks = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        mlmodel.predict(feed)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        rss_peaks.append(proc.memory_info().rss)

    report = {
        "model": str(args.model),
        "compute_units": args.compute_units,
        "iters": args.iters,
        "time_sec_mean": float(np.mean(times)),
        "time_sec_p50": float(np.quantile(times, 0.5)),
        "time_sec_p90": float(np.quantile(times, 0.9)),
        "time_sec_min": float(np.min(times)),
        "time_sec_max": float(np.max(times)),
        "rss_bytes_before": int(rss_before),
        "rss_bytes_peak": int(max(rss_peaks) if rss_peaks else rss_before),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

