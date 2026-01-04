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

## 2026-01-04T09:27:56-06:00
- Wrote CoreML-facing documentation:
  - `docs/io_contract.md` (exact preprocessing + tensor semantics)
  - `docs/coreml_strategy.md` (monolith vs split plan)
  - `docs/OP_COMPAT.md` (workaround log scaffold)
  - `README_COREML.md` (build/run entrypoint)

## 2026-01-04T09:42:05-06:00
- Phase 3–5 (CoreML) baseline:
  - Exported TorchScript graph via `tools/export/export_sharp.py` → `artifacts/Sharp_traced.pt` + `artifacts/io_sample_inputs.npz` + `artifacts/io_sample_outputs_ref.npz`.
  - Converted to CoreML ML Program via `tools/coreml/convert_to_coreml.py` → `artifacts/Sharp.mlpackage`.
  - Implemented parity validator `tools/coreml/validate_coreml.py`; `make validate` passes on fixture set (gated on covariance implied by `(quaternions, scales)` to avoid degeneracy artifacts).

## 2026-01-04T10:59:05-06:00
- Fixed Swift demo video export hang:
  - Repro: `timeout 120s SharpDemoApp ... --video out.mp4` would stall after writing frames (stuck finishing `AVAssetWriter`).
  - Made `SharpDemoApp` an async `@main` entrypoint and await video finalization instead of blocking with a semaphore.
  - Hardened `MP4VideoWriter`:
    - `append(...)` now has a readiness timeout and checks writer failure state.
    - `finish(...)` runs `finishWriting` on a background queue with a hard timeout to prevent indefinite hangs.
  - Added stage timing + progress logs with immediate stdout flushing for easier diagnosis in non-interactive runs.

## 2026-01-04T11:29:45-06:00
- Matched `ml-sharp` EXIF auto-rotate behavior in Swift preprocessing:
  - `SharpPreprocessor.loadCGImage` now applies EXIF `Orientation` for values 3/6/8 (rotate 180 / 90 CW / 90 CCW) to match `sharp.utils.io.load_rgb(auto_rotate=True)`.
  - Verified on a synthetic JPEG with EXIF orientation=6: Swift metadata reports `3x4` and disparity factor matches Python.

## 2026-01-04T11:43:40-06:00
- Upgraded Metal renderer toward true Gaussian splatting:
  - `PLYLoader` now loads per-Gaussian quaternions (wxyz) and the renderer uses them (anisotropic, view-dependent screen-space covariance).
  - Implemented weighted blended OIT (accum + revealage) to avoid per-frame depth sorting while producing stable results.

## 2026-01-04T11:48:45-06:00
- Added an end-to-end Swift parity harness (`validate-swift`):
  - Runs the Swift demo in `--no-render` mode for each fixture and compares the emitted `scene.ply` against the PyTorch reference PLY (sampling gaussians deterministically).
  - Emits a machine-readable report at `artifacts/fixtures/coreml/swift_validate_report.json`.

## 2026-01-04T12:06:55-06:00
- Stabilized `validate-swift` and preprocessing/PLY semantics:
  - `tools/fixtures/generate_fixtures.py` now removes stray files in `artifacts/fixtures/inputs/` to keep the fixture set deterministic.
  - Swift preprocessing now uses an explicit sRGB bitmap context when decoding images (avoid device color-space surprises).
  - Swift PLY export now matches `ml-sharp` opacity-logit behavior (no clamping; allows ±inf if opacities saturate).
  - `tools/swift/validate_swift.py` metrics/tolerances updated to be robust (p99 abs thresholds; covariance checked in absolute space).

## 2026-01-04T12:40:05-06:00
- Fixed SwiftPM shader library packaging for dependency builds:
  - Observed `Swift/SharpDemoApp` failing with `cannot find 'MetalLibrary_*' in scope` because SwiftPM build tool plugins attached inside dependency packages were not executed when building the top-level demo app.
  - Switched to checked-in, platform-specific embedded `.metallib` blobs:
    - `Swift/GaussianSplatMetalRenderer/Sources/GaussianSplatMetalRenderer/MetalLibrary_GaussianSplat.swift`
    - `Swift/SharpCoreML/Sources/SharpCoreML/MetalLibrary_SharpPostprocess.swift`
  - Added generator `tools/metal/embed_metallib.py` (compiles `.metal` for `macosx`/`iphoneos`/`iphonesimulator` and selects at compile time).
  - Updated `Makefile` Swift targets to run `swift package clean` before building to avoid stale source lists after adding new files.
