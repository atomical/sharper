# SHARP on Apple (CoreML + Metal)

This repo converts Apple’s SHARP model (`apple/ml-sharp`) to a CoreML `.mlpackage` and recreates the end-to-end pipeline on Apple platforms:

- **predict**: RGB image → 3D Gaussian splat scene (`.ply`)
- **render**: Metal Gaussian splat renderer → frames/video from novel views

## Prerequisites

- macOS with Xcode Command Line Tools installed
- Homebrew (recommended)
- Python **3.11** (`brew install python@3.11`)
- Xcode (for Swift + Metal work) — version will be pinned once Swift targets land

## Quickstart (Phase 0)

1) Create venv + install deps + fetch pinned `ml-sharp`:
```bash
make venv
```

2) Generate deterministic fixture images:
```bash
make fixtures
```

3) Run PyTorch reference inference and write reference outputs:
```bash
make ref
```

Notes:
- First run will download the official SHARP checkpoint (~2.6GB) via `torch.hub` into `~/.cache/torch/hub/checkpoints/`.
- Reference outputs are written under `artifacts/fixtures/ref/` (ignored by git).

## Make Targets

- `make fixtures`: generate inputs under `artifacts/fixtures/inputs/`
- `make ref`: PyTorch reference runner → `raw_outputs.npz` + `scene.ply`
- `make export`: export a CoreML-friendly Torch graph (WIP)
- `make coreml`: convert exported graph → `artifacts/Sharp.mlpackage` (WIP)
- `make coreml-fp16`: convert exported graph → `artifacts/Sharp_fp16.mlpackage`
- `make validate`: parity suite (PyTorch vs CoreML) (WIP)
- `make validate-fp16`: best-effort parity report for FP16 model (expected to diverge from FP32 ref)
- `make validate-swift`: parity suite (Swift `scene.ply` vs PyTorch reference)
- `make demo`: Swift demo CLI (image → PLY → frames + mp4)
- `make macos-build`: build the macOS SwiftUI demo app (Xcode required)
- `make ios-build`: build the iOS SwiftUI demo app (Xcode required)
- `make visionos-build`: build the visionOS SwiftUI demo app (Xcode required)
- `make bench`: benchmarks (writes `artifacts/benches/bench_coreml.json` + `artifacts/benches/bench_swift.json`)

## Swift CLI (`SharpDemoApp`)

- Predict (image → `scene.ply` and optional render): `cd Swift/SharpDemoApp && swift run -c release SharpDemoApp predict ../../artifacts/fixtures/inputs/indoor_teaser.jpg ../../artifacts/fixtures/coreml/demo --frames 60 --video ../../artifacts/fixtures/coreml/demo/out.mp4`
- Render-only (PLY → frames/video): `cd Swift/SharpDemoApp && swift run -c release SharpDemoApp render ../../artifacts/fixtures/coreml/demo/scene.ply ../../artifacts/fixtures/coreml/render_only --frames 60 --video ../../artifacts/fixtures/coreml/render_only/out.mp4`

## Swift Quickstart (`SharpQuickDemo`)

- One-command demo (image → PLY + mp4 + preview): `cd Swift/SharpDemoApp && swift run -c release SharpQuickDemo`

## GUI Demo App (macOS/iOS/visionOS)

- Open `Swift/SharpDemoApp/SharpDemoAppUI.xcodeproj` in Xcode.
- Select a scheme:
  - `SharpDemoAppMac` (macOS)
  - `SharpDemoAppUI` (iOS)
  - `SharpDemoAppVision` (visionOS) — optional / best-effort (currently deferred); requires the visionOS Simulator runtime installed via Xcode > Settings > Components.
- Run on a simulator/device for the selected platform.
- In the app UI:
  - Select `artifacts/Sharp.mlpackage` (generate it first via `make coreml`).
  - Select an input image.
  - Tap **Predict → PLY** or **Predict + Render Frame**.

## Key Docs

- `docs/io_contract.md`: exact preprocessing + tensor semantics (source of truth)
- `docs/coreml_strategy.md`: conversion strategy and fallbacks
- `docs/OP_COMPAT.md`: conversion workarounds as they are discovered
- `docs/PROGRESS.md`: timestamped implementation log

## Troubleshooting

- If `make venv` fails with missing `python3.11`: install via Homebrew and retry.
- If model download is slow/unreliable: rerun `make ref` (torch will resume or reuse cache once complete).
- If Swift build errors mention missing `MetalLibrary_*`: run `make demo` or `make validate-swift` (they do a `swift package clean`), or manually run `cd Swift/SharpDemoApp && swift package clean`.
