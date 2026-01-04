ROLE: You are Codex acting as a senior ML+Apple platform engineer. Your job is to convert Apple’s SHARP model from the apple/ml-sharp repo into a CoreML (.mlpackage) model AND rebuild all surrounding “machinery” needed to duplicate the repo’s functionality end-to-end on Apple platforms (macOS/iOS/visionOS). Do not ask follow-up questions; make reasonable defaults and document assumptions.

DOCUMENT progress in docs/PROGRESS.md. Everything must have timestamps.

PRIMARY GOAL (FUNCTIONAL PARITY)
Reproduce these behaviors from ml-sharp:
1) “predict”: input a single RGB image → output a 3D Gaussian splat scene (PLY) equivalent in semantics to `sharp predict`.
2) “render”: render that splat scene from novel viewpoints along a simple trajectory, equivalent in spirit to `sharp render` / `--render` (ml-sharp’s renderer is CUDA-only; implement a Metal renderer).

HARD REQUIREMENTS
- Deterministic, reproducible pipeline with parity tests (PyTorch vs CoreML).
- A documented I/O contract (exact preprocessing and output tensor semantics).
- A CoreML model package (ML Program preferred) suitable for Xcode integration.
- Swift “runner” that performs: preprocessing → CoreML inference → postprocess → PLY export.
- Metal Gaussian splat renderer producing frames/videos from camera trajectories.
- Build scripts (Makefile) to run: reference inference, conversion, validation, and demo.

NON-GOALS (unless needed for parity)
- Training. Only inference conversion.
- CUDA dependence on Apple devices. Use CoreML + Metal.

REPO INPUTS
- Use the apple/ml-sharp repository as the reference implementation.
- Use the official SHARP checkpoint used by the CLI. Ensure your code works with the model downloaded/cached by the repo’s CLI.

OUTPUTS / DELIVERABLES
1) `artifacts/Sharp.mlpackage` (CoreML model)
2) `tools/export/ref_infer.py` (PyTorch reference runner)
3) `tools/export/export_sharp.py` (trace/export wrapper)
4) `tools/coreml/convert_to_coreml.py` (conversion script)
5) `tools/coreml/validate_coreml.py` (parity tests)
6) `Swift/SharpCoreML` Swift Package (preprocess + inference + postprocess + PLY writer)
7) `Swift/GaussianSplatMetalRenderer` Swift Package (Metal renderer)
8) `Swift/SharpDemoApp` (sample app or CLI) that runs: image → splat → render frames/video
9) `docs/io_contract.md`, `docs/coreml_strategy.md`, `docs/OP_COMPAT.md`, `README_COREML.md`
10) `Makefile` with targets: `ref`, `export`, `coreml`, `validate`, `demo`, `bench`

PROJECT STRUCTURE TO CREATE
.
├── artifacts/
│   ├── Sharp.mlpackage
│   ├── fixtures/
│   │   ├── inputs/           # test images
│   │   ├── ref/              # pytorch reference outputs (.npz, .ply, renders)
│   │   └── coreml/           # coreml outputs (.npz, .ply, renders)
│   └── benches/
├── docs/
│   ├── io_contract.md
│   ├── coreml_strategy.md
│   ├── OP_COMPAT.md
│   └── README_COREML.md
├── tools/
│   ├── export/
│   │   ├── ref_infer.py
│   │   ├── export_sharp.py
│   │   └── utils.py
│   ├── coreml/
│   │   ├── convert_to_coreml.py
│   │   ├── validate_coreml.py
│   │   ├── run_coreml.py
│   │   └── bench_coreml.py
│   └── ply/
│       ├── write_ply.py      # python reference PLY writer (if needed)
│       └── schema.md
├── Swift/
│   ├── SharpCoreML/
│   ├── GaussianSplatMetalRenderer/
│   └── SharpDemoApp/
└── Makefile

PHASE 0 — ESTABLISH A PYTORCH “GOLDEN” REFERENCE (MUST DO FIRST)
Objective: produce stable reference outputs to validate conversion and Swift implementation.

Task 0.1: Clone and run baseline CLI
- Clone apple/ml-sharp at a pinned commit (record commit hash in docs).
- Create a small fixture set of 5–10 images in `artifacts/fixtures/inputs/` (include at least: indoor scene, outdoor, person/portrait, textured object, low-light).
- Run the repo’s official CLI (`sharp predict`) on each image to generate:
  - output .ply (splat)
  - optionally rendered frames/video if CUDA render is available (it’s okay if not; still create test harness)
- Store all outputs in `artifacts/fixtures/ref/`.

Acceptance:
- A single command reproduces the reference outputs on your machine.
- Document exact CLI commands used.

Task 0.2: Write `tools/export/ref_infer.py`
- Implement a Python entrypoint that loads the model exactly as the CLI does and runs inference.
- Save:
  - `raw_outputs.npz` containing the raw tensors you will later compare (at minimum: mean positions, rotations/quats, scale logits, opacity logits, colors; or intermediate maps if final assembly is postprocess).
  - A reference `.ply` that matches the CLI output semantics.
- Make it deterministic:
  - `model.eval()`
  - set seeds and deterministic flags
  - avoid nondeterministic ops where possible
- Add a `--dump-intermediates` option to capture intermediate tensors (encoder outputs, depth layers, delta maps) for debugging conversion mismatches.

Acceptance:
- `python tools/export/ref_infer.py --image <img> --out artifacts/fixtures/ref/<case>/` produces a .ply and .npz.
- For at least one case, the .ply is semantically identical to the CLI output (same number of gaussians and very close attributes).

PHASE 1 — DEFINE THE MODEL I/O CONTRACT (THE MOST IMPORTANT DOCUMENT)
Objective: write down EXACTLY what CoreML takes as input and what it must output.

Task 1.1: Identify preprocessing exactly
- Find the repo’s inference preprocessing:
  - resize/crop/pad strategy
  - color channel order (RGB vs BGR)
  - value range (0–1 vs 0–255)
  - normalization mean/std
  - expected input spatial resolution(s)
- Write `docs/io_contract.md` with:
  - input specification: shape, dtype, range, normalization, resizing
  - output specification: names, shapes, dtype, semantic meanings
  - decoding rules: e.g., scales = exp(scale_logits), opacity = sigmoid(opacity_logits), color space handling (sRGB vs linear)
  - coordinate system: OpenCV convention (x right, y down, z forward), and how to interpret camera extrinsics/intrinsics if present.

Acceptance:
- A new engineer can implement preprocessing in Swift using only `docs/io_contract.md` and match Python tensors.

PHASE 2 — COREML PACKAGING STRATEGY (MONOLITH vs SPLIT)
Objective: choose a conversion strategy and implement both (monolith first; split as fallback).

Task 2.1: Write `docs/coreml_strategy.md`
Include two strategies:

A) MONOLITH MODEL (preferred)
- CoreML model: image → final gaussian parameter tensors (means/quats/scale_logits/opacity_logits/colors)
- Swift: minimal postprocess (exp/sigmoid/colorspace) + PLY writer + renderer

B) SPLIT MODEL (fallback)
- CoreML model: image → intermediate outputs (e.g., depth layers + gaussian delta maps)
- Swift: implement gaussian initialization and delta composition postprocess in Swift/Metal compute to produce final gaussian tensors

Define fallback triggers:
- conversion fails due to unsupported ops
- runtime memory/bandwidth too high when returning final 1.2M-gaussian tensors

Acceptance:
- Strategy doc exists before conversion work proceeds (to avoid thrash).

PHASE 3 — EXPORT A COREML-FRIENDLY PYTORCH GRAPH
Objective: build a wrapper module with stable inputs/outputs and export it.

Task 3.1: Implement `tools/export/export_sharp.py`
- Create `SharpExportWrapper(nn.Module)` that:
  - accepts a single tensor input matching the IO contract
  - runs only inference-time computation
  - returns a tuple of tensors ONLY (no dicts, no custom classes)
  - avoids Python control-flow dependent on tensor values
  - uses static/fixed shapes (start with one fixed resolution)
- Export attempt order:
  1) torch.jit.trace (recommended stable path)
  2) torch.jit.script if possible
  3) torch.export.export (if needed)
  4) ONNX as last resort

Artifacts to write:
- `artifacts/Sharp_traced.pt` or `artifacts/Sharp_exported.pt2`
- `artifacts/io_sample_inputs.npz` with the exact input tensor used for export
- `artifacts/io_sample_outputs_ref.npz` from PyTorch wrapper forward pass

Acceptance:
- You can load the exported artifact and run it to reproduce outputs close to the original wrapper.

PHASE 4 — CONVERT TO COREML
Objective: convert exported graph to .mlpackage using coremltools.

Task 4.1: Create `tools/coreml/convert_to_coreml.py`
- Use coremltools unified conversion API.
- Prefer ML Program backend (suitable for transformer-style ops).
- Start with FP32 conversion for correctness; then FP16 for performance.
- Declare inputs explicitly (shape, dtype). If using an Image input type, bake scale/bias for normalization if possible; otherwise use MLMultiArray.
- Save:
  - `artifacts/Sharp.mlpackage`
  - `artifacts/Sharp_fp32.mlpackage` (optional)
  - conversion logs and metadata: `artifacts/coreml_conversion_report.json`

Task 4.2: Systematic unsupported-op triage loop
- If conversion fails, capture failing op(s) and patch the wrapper/model:
  - Replace unsupported ops with equivalent supported patterns (reshape+matmul, explicit softmax, avoid fancy indexing)
  - Inline small functions to simplify graph
  - Remove training-only branches
- Document every change in `docs/OP_COMPAT.md`:
  - original op
  - replacement
  - why it’s safe
  - parity impact

Acceptance:
- Conversion script completes in one command and produces `Sharp.mlpackage`.

PHASE 5 — PARITY VALIDATION (PYTORCH vs COREML)
Objective: prove the CoreML model matches PyTorch to acceptable tolerances, and diagnose divergences.

Task 5.1: Implement `tools/coreml/run_coreml.py`
- Runs the CoreML model on an input tensor/image and dumps raw outputs to `.npz` in a stable schema.

Task 5.2: Implement `tools/coreml/validate_coreml.py`
For each fixture image:
- Run PyTorch wrapper → `ref/raw_outputs.npz`
- Run CoreML → `coreml/raw_outputs.npz`
- Compare per-tensor:
  - shapes match exactly
  - max abs error, mean abs error, relative error
  - for quaternions: compare normalized dot product / angle difference
- Decode postprocess-dependent values (exp/sigmoid) and compare decoded arrays too.
- Emit a human-readable report (markdown + JSON) and fail CI if outside tolerance.
- Save optional debug artifacts: histograms, min/max tables.

Acceptance:
- `make validate` runs parity suite and prints a clear pass/fail summary.
- If fails, report identifies which tensor and where mismatch occurs.

PHASE 6 — SWIFT “PREDICT” PIPELINE (IMAGE → PLY)
Objective: implement on-device prediction matching ml-sharp semantics.

Task 6.1: Swift preprocessing
- Implement exact resize/crop/pad and normalization from `docs/io_contract.md`.
- Provide two input paths:
  A) `CGImage/UIImage/NSImage` → preprocessing → `MLMultiArray`
  B) if CoreML model uses Image input, ensure preprocessing matches CoreML internal scaler/bias

Task 6.2: CoreML runner package `Swift/SharpCoreML`
- Load `Sharp.mlpackage`
- Run inference
- Convert outputs to Swift arrays/buffers efficiently (avoid copying where possible).
- Apply required decoding:
  - scales = exp(scale_logits)
  - opacity = sigmoid(opacity_logits)
  - handle color space metadata (sRGB vs linear) consistently

Task 6.3: PLY writer
- Implement two PLY modes:
  1) “ml-sharp compatible” mode: matches field names and any extra elements the repo writes (extrinsics/image size/etc. if present)
  2) “compatibility” mode: vertex-only PLY (or another widely supported gaussian format) if external tools choke on extra elements
- Ensure OpenCV coordinate convention is preserved.

Acceptance:
- A Swift CLI or app can run: `predict(input.jpg) -> output.ply`.
- The output can be loaded by your Metal renderer and, ideally, by at least one third-party viewer.

PHASE 7 — METAL GAUSSIAN SPLAT RENDERER (“RENDER” REPLACEMENT)
Objective: implement a Metal renderer to replace CUDA gsplat rendering.

Task 7.1: Metal renderer package `Swift/GaussianSplatMetalRenderer`
- Load gaussian buffers (means, scales, quats, colors, opacity).
- Implement standard gaussian splatting pipeline:
  - camera transform → screen projection
  - compute screen-space ellipse / covariance
  - tile-based binning (recommended) or approximate sorting
  - alpha blending accumulation
- Implement camera model consistent with OpenCV axes and your stored gaussians.
- Provide interactive rendering of a single view.

Task 7.2: Trajectory renderer + video export
- Implement trajectory generation similar to CLI (orbit around object, dolly, etc.).
- Render N frames to textures.
- Encode mp4 via AVFoundation.

Acceptance:
- Given a .ply from Swift prediction, the demo can render interactive views.
- The demo can export a short video for a fixed trajectory.

PHASE 8 — PERFORMANCE AND MEMORY (BENCHMARKS)
Objective: measure and optimize so it’s viable on Apple hardware.

Task 8.1: Bench harness `tools/coreml/bench_coreml.py` and Swift benchmark
- Measure:
  - CoreML inference time
  - peak memory
  - output conversion time (MLMultiArray → buffers)
  - render FPS
- Produce a JSON bench report under `artifacts/benches/`.

Task 8.2: Precision / quantization variants
- Start FP32 for parity
- Move to FP16 weights for speed
- If needed, investigate quantization-friendly options (document, but don’t over-engineer unless required)

Acceptance:
- `make bench` outputs a report with timings and memory.

MAKEFILE TARGETS (MUST IMPLEMENT)
- `make ref`:
  - runs Python reference inference over fixtures; writes ref outputs
- `make export`:
  - exports traced/exported model artifact(s)
- `make coreml`:
  - converts to CoreML and writes `artifacts/Sharp.mlpackage`
- `make validate`:
  - runs parity tests
- `make demo`:
  - builds/runs Swift demo (or prints clear Xcode build/run instructions)
- `make bench`:
  - runs benchmarks

DOCUMENTATION REQUIREMENTS
- `README_COREML.md` must include:
  - prerequisites (Python versions, Xcode version)
  - setup commands
  - how to run each Makefile target
  - troubleshooting section: common conversion failures + fixes
- `docs/io_contract.md` must be complete and exact.
- `docs/OP_COMPAT.md` must list all op workarounds and reasoning.

QUALITY GATES / DEFINITION OF DONE
DONE means:
1) A reproducible Python reference exists (PLY + NPZ outputs) for fixture images.
2) A CoreML mlpackage exists and runs on macOS via a small test harness.
3) Parity tests run and meet tolerances for all fixtures (or clear documented exceptions).
4) Swift runner produces a PLY that renders correctly in Metal renderer.
5) Demo can render a trajectory and export frames/video.
6) All commands and assumptions are documented and repeatable.

IMPLEMENTATION GUIDELINES / DEFAULTS
- Prefer correctness first: FP32 CoreML for initial parity, then FP16.
- Avoid dynamic shapes at first; pick a single inference resolution that matches ml-sharp defaults.
- Use deterministic flags where possible; document any nondeterminism.
- If output tensors are huge (1M+ gaussians), prioritize memory-safe buffer handling in Swift:
  - avoid intermediate copies
  - consider using `MTLBuffer` directly if feasible.

NOW EXECUTE:
- Start at PHASE 0 and proceed sequentially.
- Commit incremental deliverables after each phase.
- Do not skip parity testing—use it to drive conversion fixes.
