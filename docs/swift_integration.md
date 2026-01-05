# Integrating `SharpCoreML` + `GaussianSplatMetalRenderer` into a Swift Project

This document is a practical, end-to-end integration guide for using this repo’s Swift libraries in your own macOS/iOS app or SwiftPM tool.

What you get:

- `SharpCoreML`: preprocessing → CoreML inference → postprocess (unprojection) → `SharpPrediction` + `SharpPLYWriter`.
- `GaussianSplatMetalRenderer`: load `.ply` or buffers and render anisotropic Gaussian splats via Metal; optional MP4 export.

If you want to understand the exact ML I/O and tensor semantics, read `docs/io_contract.md` first.

## 0) Prerequisites / Assumptions

Assumptions (reasonable defaults):

- You are integrating on **macOS** (primary target) and will use **SwiftPM** or **Xcode**.
- You will run the validated **FP32** CoreML model (`artifacts/Sharp.mlpackage`).
- You are OK shipping an app that compiles the `.mlpackage` at first launch (acceptable for dev and many internal apps). If you need precompiled `.mlmodelc` for distribution, see “Model packaging options”.

Prereqs:

- Xcode installed (CoreML + Metal toolchain). This repo has been built with `xcodebuild -version` = Xcode **26.1.1** (Build **17B100**).
- Swift tools: these packages use `// swift-tools-version: 6.0`.
- macOS 13+ (per package platform declarations).

## 1) Generate (or obtain) the CoreML model

This repo’s conversion pipeline produces:

- `artifacts/Sharp.mlpackage` (validated FP32 ML Program)

From the repo root:

```bash
make coreml
```

This writes `artifacts/Sharp.mlpackage/` (a directory). Keep the entire directory intact.

Notes:

- The FP16 variant exists (`make coreml-fp16`) but is **not** parity-validated; use FP32 unless you accept numeric drift.
- The model is large; first inference will also pay compilation overhead when using `MLModel.compileModel(at:)`.

## 2) How to add the Swift packages to your project

### Important: these are *nested* Swift packages

This repo does **not** expose a single top-level SwiftPM package. The Swift packages live under:

- `Swift/SharpCoreML/Package.swift`
- `Swift/GaussianSplatMetalRenderer/Package.swift`

That means you generally cannot do `.package(url: "…sharper.git", …)` unless you restructure/publish those subpackages separately.

Recommended integration approaches:

1) **Git submodule** (recommended):
   - Add this repo as a submodule inside your app repo, e.g. `Vendor/sharper/`.
   - Reference the packages via `.package(path: "Vendor/sharper/Swift/SharpCoreML")`, etc.

2) **Vendored copy**:
   - Copy `Swift/SharpCoreML` and `Swift/GaussianSplatMetalRenderer` directories into your repo.

3) **Local checkout**:
   - During development, point `.package(path:)` at a local clone of this repo.

### 2.1 SwiftPM (`Package.swift`) integration

In your app/tool `Package.swift`, add the package dependencies by **path**:

```swift
dependencies: [
  .package(path: "Vendor/sharper/Swift/SharpCoreML"),
  .package(path: "Vendor/sharper/Swift/GaussianSplatMetalRenderer"),
],
targets: [
  .executableTarget(
    name: "MyApp",
    dependencies: [
      .product(name: "SharpCoreML", package: "SharpCoreML"),
      .product(name: "GaussianSplatMetalRenderer", package: "GaussianSplatMetalRenderer"),
    ]
  ),
]
```

Then:

```bash
swift build -c release
```

### 2.2 Xcode integration (local package)

In Xcode:

1) `File` → `Add Package Dependencies…`
2) Choose `Add Local…`
3) Select `Swift/SharpCoreML` (the folder containing `Package.swift`)
4) Repeat for `Swift/GaussianSplatMetalRenderer`

Then add the products to your target:

- `SharpCoreML`
- `GaussianSplatMetalRenderer`

## 3) Model packaging options (how your Swift code finds `Sharp.mlpackage`)

You need a file URL to the `.mlpackage` directory (or a compiled `.mlmodelc`).

### Option A (dev-friendly): load from an on-disk path

This is what the demos do: you pass a filesystem URL like `…/artifacts/Sharp.mlpackage`.

Pros:
- Fast iteration, no Xcode resource plumbing.

Cons:
- Not appropriate for shipping sandboxed apps unless the model is in your app container or user-selected.

### Option B (ship-friendly): bundle the `.mlpackage` in your app

In Xcode, drag `artifacts/Sharp.mlpackage` into your project and ensure it’s included in the app target’s resources.

At runtime:

```swift
guard let modelURL = Bundle.main.url(forResource: "Sharp", withExtension: "mlpackage") else {
  fatalError("Missing Sharp.mlpackage in bundle resources")
}
```

Then pass `modelURL` into `SharpCoreMLRunner`.

### Option C (ship-friendly): precompile to `.mlmodelc`

You can compile the model package ahead of time:

```bash
xcrun coremlcompiler compile artifacts/Sharp.mlpackage /tmp/SharpCompiled
```

This produces a `Sharp.mlmodelc/` directory. You can bundle that and load it with `MLModel(contentsOf:)`.

This repo’s `SharpCoreMLRunner` currently calls `MLModel.compileModel(at:)` internally, so it expects the *uncompiled* `.mlpackage` by default. If you prefer `.mlmodelc`, you can either:

- Load the compiled model yourself and run inference directly (see “Raw CoreML usage (manual inference)” below), or
- Adjust `SharpCoreMLRunner` in your fork to accept a compiled URL.

### Raw CoreML usage (manual inference)

If you want full control (e.g., to load a precompiled `.mlmodelc`), you can bypass `SharpCoreMLRunner` and call CoreML directly.

This is also useful if you want to integrate into an existing CoreML pipeline that already manages models/configs.

```swift
import CoreML
import Foundation
import SharpCoreML

// 1) Load a compiled model (.mlmodelc) or compile a .mlpackage.
let config = MLModelConfiguration()
config.computeUnits = .all

let model = try MLModel(contentsOf: compiledModelURL, configuration: config)

// 2) Preprocess (exactly matches docs/io_contract.md).
let cgImage = try SharpPreprocessor.loadCGImage(from: imageURL)
let metadata = SharpPreprocessor.loadMetadata(from: imageURL, imageWidth: cgImage.width, imageHeight: cgImage.height)
let (image, disparity) = try SharpPreprocessor.makeInputs(cgImage: cgImage, metadata: metadata)

let provider = try MLDictionaryFeatureProvider(dictionary: [
  "image": MLFeatureValue(multiArray: image),
  "disparity_factor": MLFeatureValue(multiArray: disparity),
])

// 3) Run the model.
let features = try model.prediction(from: provider)

func out(_ name: String) throws -> MLMultiArray {
  guard let v = features.featureValue(for: name)?.multiArrayValue else {
    throw NSError(domain: "SharpManual", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing output \(name)"])
  }
  return v
}

let meanPre = try out("mean_vectors_pre")        // [1,N,3]
let quatPre = try out("quaternions_pre")         // [1,N,4]
let scalePre = try out("singular_values_pre")    // [1,N,3]
let colorsPre = try out("colors_linear_pre")     // [1,N,3]
let opacitiesPre = try out("opacities_pre")      // [1,N]
```

Important:

- These outputs are **pre-unprojection** tensors. To match `sharp predict` PLY semantics you must still apply the unprojection/re-decomposition described in `docs/io_contract.md`.
- `SharpCoreMLRunner` already performs that postprocess via Metal and returns PLY-ready gaussians (`prediction.postprocessed`), so prefer it unless you need custom model lifecycle management.

## 4) Minimal integration: image → prediction → `.ply`

### 4.1 Predict with the high-level runner

```swift
import Foundation
import SharpCoreML

let runner = try SharpCoreMLRunner(modelURL: modelURL, computeUnits: .all)
let prediction = try runner.predict(imageURL: imageURL)

let plyURL = outDir.appendingPathComponent("scene.ply")
try SharpPLYWriter.write(prediction: prediction, to: plyURL)
```

What `SharpCoreMLRunner.predict(imageURL:)` does:

1) Loads a `CGImage` from disk and applies EXIF orientation (matching upstream `ml-sharp` behavior).
2) Computes the camera metadata (focal length heuristic if EXIF is missing).
3) Preprocesses into CoreML inputs:
   - `image`: `float32[1,3,1536,1536]` in RGB, range `[0,1]`, bilinear resize with `align_corners=True` equivalent.
   - `disparity_factor`: `float32[1]`
4) Runs the CoreML model.
5) Runs Metal postprocess (`unprojectGaussians`) to match `sharp predict` semantics.

Exact I/O contract: `docs/io_contract.md`.

### 4.2 Output `.ply` semantics (important for interoperability)

The Swift PLY writer (`SharpPLYWriter`) writes a binary little-endian `.ply` matching `ml-sharp`’s `save_ply`:

- Vertex properties: `x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`
- Where:
  - `f_dc_*` are SH degree-0 coefficients computed from **sRGB** color
  - `opacity` is an **opacity logit**
  - `scale_*` are **log-scales**
  - `rot_*` is quaternion **wxyz**

Schema reference: `tools/ply/schema.md`.

Compatibility note:

- The file also includes extra `ml-sharp` metadata elements (`intrinsic`, `extrinsic`, etc.). Some third-party PLY loaders reject unknown elements. If you hit that, you may need a “vertex-only” writer that omits those extra elements (planned but not currently implemented in Swift).

## 5) Rendering in Metal (3DGS renderer integration)

The renderer supports two primary paths:

1) Render directly from `SharpPrediction` buffers (fastest, avoids PLY readback).
2) Render from a `.ply` file (useful for debugging and interchange).

### 5.1 Render directly from `SharpPrediction` (no `.ply` required)

```swift
import Metal
import SharpCoreML
import GaussianSplatMetalRenderer
import simd

let device = MTLCreateSystemDefaultDevice()!

// Convert colors/opacities MLMultiArray into MTLBuffers.
let count = prediction.postprocessed.count
let colors = prediction.raw.colorsLinearPre
let opacities = prediction.raw.opacitiesPre

let colorsBytes = count * 3 * MemoryLayout<Float>.size
let opacitiesBytes = count * MemoryLayout<Float>.size

let colorsBuf = device.makeBuffer(bytes: colors.dataPointer, length: colorsBytes, options: .storageModeShared)!
let opacitiesBuf = device.makeBuffer(bytes: opacities.dataPointer, length: opacitiesBytes, options: .storageModeShared)!

let scene = GaussianScene(
  count: count,
  means: prediction.postprocessed.mean,
  quaternions: prediction.postprocessed.quaternions,
  scales: prediction.postprocessed.singularValues,
  colorsLinear: colorsBuf,
  opacities: opacitiesBuf
)

let renderer = try GaussianSplatRenderer(device: device)

// Camera intrinsics derived from the prediction metadata.
let outW = 512
let outH = 512
let fx = prediction.metadata.focalLengthPx * Float(outW) / Float(prediction.metadata.imageWidth)
let fy = prediction.metadata.focalLengthPx * Float(outH) / Float(prediction.metadata.imageHeight)
let cx = Float(outW) * 0.5
let cy = Float(outH) * 0.5

// Orbit camera around scene bounds (simple default).
let (center, radius) = (SIMD3<Float>(0, 0, 0), Float(1.5)) // replace with your bounds
let eye = center + SIMD3<Float>(0, 0, radius)
let view = PinholeCamera.lookAt(eye: eye, target: center)
let cam = PinholeCamera(viewMatrix: view, fx: fx, fy: fy, cx: cx, cy: cy)

let cgImage = try renderer.renderToCGImage(scene: scene, camera: cam, width: outW, height: outH)
```

Notes:

- The renderer assumes an OpenCV-like camera convention and uses an `up` vector of `(0, -1, 0)` in `PinholeCamera.lookAt(...)` to keep “image up” consistent with y-down image coordinates.
- The renderer uses weighted blended OIT (no per-frame sorting). It’s stable and fast, but may differ slightly from CUDA renderers that do full sorting.

### 5.2 Render from `.ply`

```swift
import Metal
import GaussianSplatMetalRenderer

let device = MTLCreateSystemDefaultDevice()!
let scene = try PLYLoader.loadMLSharpCompatiblePLY(url: plyURL, device: device)

let renderer = try GaussianSplatRenderer(device: device)
let cgImage = try renderer.renderToCGImage(scene: scene, camera: cam, width: outW, height: outH)
```

If you want to use the `.ply`’s intrinsics rather than recomputing them, use:

```swift
let (scene, meta) = try PLYLoader.loadMLSharpCompatiblePLYWithMetadata(url: plyURL, device: device)
```

## 6) MP4 export

Use `MP4VideoWriter` (from `GaussianSplatMetalRenderer`) to write frames into an `.mp4`:

```swift
import GaussianSplatMetalRenderer

let writer = try MP4VideoWriter(url: outVideoURL, width: outW, height: outH, fps: 30)
try writer.append(cgImage)
try await writer.finish(timeoutSeconds: 60)
```

Make sure the output directory exists (or create it) before constructing the writer.

## 7) Determinism / compute units

- For the **most deterministic** comparisons, use CoreML CPU-only:
  - `SharpCoreMLRunner(modelURL:…, computeUnits: .cpuOnly)`
  - `tools/coreml/validate_coreml.py` defaults to CPU-only for that reason.
- For best performance, use `.all` (CoreML may choose CPU/GPU/Neural Engine depending on device and model).

## 8) Troubleshooting checklist

- `modelCompileFailed` / `modelLoadFailed`:
  - Confirm you have the entire `Sharp.mlpackage/` directory, not a single file.
  - Confirm the URL points to the package directory.
- Metal failures (`metalUnavailable`, `metalLibraryLoadFailed`):
  - Ensure Metal is available (on macOS this requires a Metal-capable GPU and that you’re not running in a restricted environment).
  - The shader libraries are embedded; you should not need build tool plugins.
- MP4 export `startFailed`:
  - Ensure the output directory exists and the path is writable.

## 9) Reference implementations in this repo

Use these as “copy-paste-ready” integration examples:

- SwiftPM quickstart: `Swift/SharpDemoApp/Sources/SharpQuickDemo/SharpQuickDemo.swift`
- Full CLI (predict + render-only + bench): `Swift/SharpDemoApp/Sources/SharpDemoApp/SharpDemoApp.swift`
- Swift runner API: `Swift/SharpCoreML/Sources/SharpCoreML/SharpCoreMLRunner.swift`
- Renderer API: `Swift/GaussianSplatMetalRenderer/Sources/GaussianSplatMetalRenderer/GaussianSplatRenderer.swift`
