# PROGRESS

This log is append-only. Every entry includes an ISO-8601 timestamp with timezone offset.

## 2026-01-04T08:50:45-06:00
- Initialized repository scaffolding for SHARP → CoreML + Swift/Metal parity work.

## 2026-01-04T08:51:10-06:00
- Added initial `Makefile` with phase-oriented targets (`ref`, `export`, `coreml`, `validate`, `demo`, `bench`).

## 2026-01-04T08:55:41-06:00
- Installed Homebrew `python@3.11` (required for PyTorch + coremltools compatibility; system Python is 3.9 and Homebrew default is 3.14).

## 2026-01-04T08:57:26-06:00
- Cloned `apple/ml-sharp` into `third_party/ml-sharp` at pinned commit `1eaa046834b81852261262b41b0919f5c1efdd2e`.

## 2026-01-04T09:00:20-06:00
- Generated 5 deterministic fixture images under `artifacts/fixtures/inputs/` via `tools/fixtures/generate_fixtures.py`.

## 2026-01-04T09:14:10-06:00
- Ran `ml-sharp` baseline CLI over fixtures (downloads official checkpoint on first run).
  - Command: `.venv/bin/sharp predict -i artifacts/fixtures/inputs -o artifacts/fixtures/ref/cli --device mps --no-render`
  - Outputs: `artifacts/fixtures/ref/cli/*.ply`

## 2026-01-04T09:21:46-06:00
- Implemented deterministic PyTorch reference runner `tools/export/ref_infer.py` (writes `raw_outputs.npz` + `scene.ply`).
- Verified byte-identical parity vs `sharp predict` PLY for `indoor_teaser` (`sha256` matched).

## 2026-01-04T09:24:18-06:00
- Updated `Makefile` to fetch pinned `ml-sharp` automatically and to use a venv dependency stamp for reproducible installs.
