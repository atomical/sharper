import CoreML
import Foundation
import Metal
import SharpCoreML
import GaussianSplatMetalRenderer
import UniformTypeIdentifiers
import ImageIO
import simd

@inline(__always)
private func log(_ msg: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    print("[\(ts)] \(msg)")
    fflush(stdout)
}

private enum QuickDemoError: Error {
    case repoRootNotFound
    case missingFile(URL, hint: String)
    case invalidSize(String)
    case metalUnavailable
    case metalBufferCreateFailed
}

private struct Args {
    var modelURL: URL
    var imageURL: URL
    var outDir: URL
    var frames: Int = 60
    var width: Int = 512
    var height: Int = 512
    var mlsharpSize: Bool = false
    var fps: Int = 30
    var computeUnits: MLComputeUnits = .all
    var render: Bool = true

    // Renderer quality controls.
    var renderScale: Float = 1.0
    var compositing: GaussianSplatCompositingMode = .weightedOIT
    var toneMap: GaussianSplatToneMap = .none
    var exposureEV: Float = 0.0
    var saturation: Float = 1.0
    var contrast: Float = 1.0
    var debugView: GaussianSplatDebugView = .none

    var nearClipZ: Float = 1e-2
    var opacityThreshold: Float = 0.0
    var lowPassEps2D: Float = 0.0
    var minRadiusPx: Float = 1.0
    var maxRadiusPx: Float = 160.0

    var normalization: GaussianSplatSceneNormalization = .none
    var normalizationScale: GaussianSplatSceneScale = .none

    var lookAtMode: MLSharpTrajectoryParams.LookAtMode = .point
}

private func fileExists(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.path)
}

private func findRepoRoot(startingAt: URL) -> URL? {
    var dir = startingAt
    for _ in 0..<10 {
        let marker = dir.appendingPathComponent("docs/PROGRESS.md")
        let makefile = dir.appendingPathComponent("Makefile")
        if fileExists(marker), fileExists(makefile) { return dir }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    return nil
}

private func parseArgs() throws -> Args {
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    guard let repoRoot = findRepoRoot(startingAt: cwd) else { throw QuickDemoError.repoRootNotFound }

    let defaultModel = repoRoot.appendingPathComponent("artifacts/Sharp.mlpackage")
    let defaultImage = repoRoot.appendingPathComponent("artifacts/fixtures/inputs/indoor_teaser.jpg")
    let defaultOut = repoRoot.appendingPathComponent("artifacts/fixtures/coreml/quick_demo", isDirectory: true)

    var args = Args(modelURL: defaultModel, imageURL: defaultImage, outDir: defaultOut)

    var argv = CommandLine.arguments.dropFirst()
    while let tok = argv.first {
        argv = argv.dropFirst()
        switch tok {
        case "--model":
            guard let v = argv.first else { throw QuickDemoError.missingFile(defaultModel, hint: "--model requires a path") }
            argv = argv.dropFirst()
            args.modelURL = URL(fileURLWithPath: String(v))
        case "--image":
            guard let v = argv.first else { throw QuickDemoError.missingFile(defaultImage, hint: "--image requires a path") }
            argv = argv.dropFirst()
            args.imageURL = URL(fileURLWithPath: String(v))
        case "--out":
            guard let v = argv.first else { throw QuickDemoError.missingFile(defaultOut, hint: "--out requires a path") }
            argv = argv.dropFirst()
            args.outDir = URL(fileURLWithPath: String(v), isDirectory: true)
        case "--frames":
            guard let v = argv.first, let n = Int(v), n >= 1 else { throw QuickDemoError.invalidSize("--frames") }
            argv = argv.dropFirst()
            args.frames = n
        case "--size":
            guard let v = argv.first else { throw QuickDemoError.invalidSize("--size") }
            argv = argv.dropFirst()
            let parts = v.split(separator: "x")
            guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]), w > 0, h > 0 else { throw QuickDemoError.invalidSize(String(v)) }
            args.width = w
            args.height = h
        case "--mlsharp-size":
            args.mlsharpSize = true
        case "--render-scale":
            guard let v = argv.first, let s = Float(v) else { throw QuickDemoError.invalidSize("--render-scale") }
            argv = argv.dropFirst()
            args.renderScale = s
        case "--compositing":
            guard let v = argv.first else { throw QuickDemoError.invalidSize("--compositing") }
            argv = argv.dropFirst()
            if v == "oit" {
                args.compositing = .weightedOIT
            } else if v.hasPrefix("bins") {
                var n = GaussianSplatCompositingMode.defaultDepthBinCount
                if let idx = v.firstIndex(of: ":") {
                    let tail = v[v.index(after: idx)...]
                    if let parsed = Int(tail) { n = parsed }
                }
                args.compositing = .depthBinnedAlpha(binCount: n)
            } else {
                throw QuickDemoError.invalidSize("Unknown --compositing: \(v)")
            }
        case "--tonemap":
            guard let v = argv.first, let tm = GaussianSplatToneMap(rawValue: String(v)) else { throw QuickDemoError.invalidSize("--tonemap") }
            argv = argv.dropFirst()
            args.toneMap = tm
        case "--exposure-ev":
            guard let v = argv.first, let ev = Float(v) else { throw QuickDemoError.invalidSize("--exposure-ev") }
            argv = argv.dropFirst()
            args.exposureEV = ev
        case "--saturation":
            guard let v = argv.first, let s = Float(v) else { throw QuickDemoError.invalidSize("--saturation") }
            argv = argv.dropFirst()
            args.saturation = s
        case "--contrast":
            guard let v = argv.first, let c = Float(v) else { throw QuickDemoError.invalidSize("--contrast") }
            argv = argv.dropFirst()
            args.contrast = c
        case "--debug":
            guard let v = argv.first, let dv = GaussianSplatDebugView(rawValue: String(v)) else { throw QuickDemoError.invalidSize("--debug") }
            argv = argv.dropFirst()
            args.debugView = dv
        case "--near-clip":
            guard let v = argv.first, let z = Float(v) else { throw QuickDemoError.invalidSize("--near-clip") }
            argv = argv.dropFirst()
            args.nearClipZ = z
        case "--opacity-threshold":
            guard let v = argv.first, let a = Float(v) else { throw QuickDemoError.invalidSize("--opacity-threshold") }
            argv = argv.dropFirst()
            args.opacityThreshold = a
        case "--lowpass-eps2d":
            guard let v = argv.first, let e = Float(v) else { throw QuickDemoError.invalidSize("--lowpass-eps2d") }
            argv = argv.dropFirst()
            args.lowPassEps2D = e
        case "--min-radius":
            guard let v = argv.first, let px = Float(v) else { throw QuickDemoError.invalidSize("--min-radius") }
            argv = argv.dropFirst()
            args.minRadiusPx = px
        case "--max-radius":
            guard let v = argv.first, let px = Float(v) else { throw QuickDemoError.invalidSize("--max-radius") }
            argv = argv.dropFirst()
            args.maxRadiusPx = px
        case "--normalize":
            guard let v = argv.first else { throw QuickDemoError.invalidSize("--normalize") }
            argv = argv.dropFirst()
            switch v {
            case "none":
                args.normalization = .none
            case "recenter_xy":
                args.normalization = .recenterXY
            case "recenter_xyz":
                args.normalization = .recenterXYZ
            default:
                throw QuickDemoError.invalidSize("Unknown --normalize: \(v)")
            }
        case "--normalize-scale":
            guard let v = argv.first else { throw QuickDemoError.invalidSize("--normalize-scale") }
            argv = argv.dropFirst()
            switch v {
            case "none":
                args.normalizationScale = .none
            case "unit_radius":
                args.normalizationScale = .unitRadius
            default:
                throw QuickDemoError.invalidSize("Unknown --normalize-scale: \(v)")
            }
        case "--lookat":
            guard let v = argv.first, let lm = MLSharpTrajectoryParams.LookAtMode(rawValue: String(v)) else { throw QuickDemoError.invalidSize("--lookat") }
            argv = argv.dropFirst()
            args.lookAtMode = lm
        case "--fps":
            guard let v = argv.first, let n = Int(v), n >= 1 else { throw QuickDemoError.invalidSize("--fps") }
            argv = argv.dropFirst()
            args.fps = n
        case "--no-render":
            args.render = false
        case "--compute-units":
            guard let v = argv.first else { throw QuickDemoError.missingFile(defaultModel, hint: "--compute-units requires a value") }
            argv = argv.dropFirst()
            switch v {
            case "all":
                args.computeUnits = .all
            case "cpu_only":
                args.computeUnits = .cpuOnly
            case "cpu_and_gpu":
                args.computeUnits = .cpuAndGPU
            case "cpu_and_ne":
                args.computeUnits = .cpuAndNeuralEngine
            default:
                throw QuickDemoError.missingFile(defaultModel, hint: "Unknown --compute-units: \(v)")
            }
        case "--help", "-h":
            print(
                """
                SharpQuickDemo (macOS) — end-to-end quickstart using CoreML + Metal.

                Defaults:
                  --model  <repo>/artifacts/Sharp.mlpackage
                  --image  <repo>/artifacts/fixtures/inputs/indoor_teaser.jpg
                  --out    <repo>/artifacts/fixtures/coreml/quick_demo/

                Usage:
                  swift run -c release SharpQuickDemo [--model <mlpackage>] [--image <jpg/png>] [--out <dir>]
                                                  [--frames N] [--size WxH] [--mlsharp-size] [--fps N]
                                                  [--compute-units all|cpu_only|cpu_and_gpu|cpu_and_ne]
                                                  [--render-scale S] [--compositing oit|bins[:N]]
                                                  [--tonemap none|reinhard|aces] [--exposure-ev EV] [--saturation S] [--contrast C]
                                                  [--debug none|alpha|depth|disparity|radius]
                                                  [--near-clip Z] [--opacity-threshold A] [--lowpass-eps2d E] [--min-radius PX] [--max-radius PX]
                                                  [--normalize none|recenter_xy|recenter_xyz] [--normalize-scale none|unit_radius]
                                                  [--lookat point|ahead]
                                                  [--no-render]
                """
            )
            exit(0)
        default:
            throw QuickDemoError.missingFile(defaultModel, hint: "Unknown arg: \(tok). Use --help.")
        }
    }

    if !fileExists(args.modelURL) {
        throw QuickDemoError.missingFile(args.modelURL, hint: "Run `make coreml` to generate the CoreML model.")
    }
    if !fileExists(args.imageURL) {
        throw QuickDemoError.missingFile(args.imageURL, hint: "Run `make fixtures` or pass --image <path>.")
    }

    return args
}

private func writePNG(_ image: CGImage, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "SharpQuickDemo", code: 1)
    }
    CGImageDestinationAddImage(dest, image, nil)
    if !CGImageDestinationFinalize(dest) {
        throw NSError(domain: "SharpQuickDemo", code: 2)
    }
}

private func buildScene(device: MTLDevice, prediction: SharpPrediction) throws -> GaussianScene {
    let count = prediction.postprocessed.count

    let colors = prediction.raw.colorsLinearPre
    let opacities = prediction.raw.opacitiesPre

    let colorsBytes = count * 3 * MemoryLayout<Float>.size
    let opacitiesBytes = count * MemoryLayout<Float>.size

    guard let colorsBuf = device.makeBuffer(bytes: colors.dataPointer, length: colorsBytes, options: .storageModeShared),
          let opacitiesBuf = device.makeBuffer(bytes: opacities.dataPointer, length: opacitiesBytes, options: .storageModeShared)
    else {
        throw QuickDemoError.metalBufferCreateFailed
    }

    return GaussianScene(
        count: count,
        means: prediction.postprocessed.mean,
        quaternions: prediction.postprocessed.quaternions,
        scales: prediction.postprocessed.singularValues,
        colorsLinear: colorsBuf,
        opacities: opacitiesBuf
    )
}

@main
struct SharpQuickDemo {
    static func main() async {
        do {
            let args = try parseArgs()

            log("Model: \(args.modelURL.path)")
            log("Image: \(args.imageURL.path)")
            log("Out: \(args.outDir.path)")
            log("ComputeUnits: \(args.computeUnits)")

            try FileManager.default.createDirectory(at: args.outDir, withIntermediateDirectories: true)

            let runner = try SharpCoreMLRunner(modelURL: args.modelURL, computeUnits: args.computeUnits)
            let prediction = try runner.predict(imageURL: args.imageURL)
            log("Predicted \(prediction.postprocessed.count) gaussians.")

            let plyURL = args.outDir.appendingPathComponent("scene.ply")
            try SharpPLYWriter.write(prediction: prediction, to: plyURL)
            log("Wrote PLY: \(plyURL.path)")

            guard args.render else { return }

            guard let device = MTLCreateSystemDefaultDevice() else { throw QuickDemoError.metalUnavailable }
            let scene = try buildScene(device: device, prediction: prediction)

            var renderW = args.width
            var renderH = args.height
            if args.mlsharpSize {
                let r = MLSharpTrajectory.screenResolutionPxFromInput(width: prediction.metadata.imageWidth, height: prediction.metadata.imageHeight)
                renderW = r.width
                renderH = r.height
                log("Using ml-sharp screen resolution: \(renderW)x\(renderH)")
            }
            if renderW % 2 != 0 { renderW += 1 }
            if renderH % 2 != 0 { renderH += 1 }

            var p = MLSharpTrajectoryParams()
            p.kind = .rotateForward
            p.numSteps = args.frames
            p.lookAtMode = args.lookAtMode
            let fx0 = Float(prediction.metadata.focalLengthPx)
            let cx0 = (Float(prediction.metadata.imageWidth) - 1) * 0.5
            let cy0 = (Float(prediction.metadata.imageHeight) - 1) * 0.5
            let (cameras, depth) = MLSharpTrajectory.makeCameras(
                scene: scene,
                sourceImageWidth: prediction.metadata.imageWidth,
                sourceImageHeight: prediction.metadata.imageHeight,
                intrinsicFx: fx0,
                intrinsicFy: fx0,
                intrinsicCx: cx0,
                intrinsicCy: cy0,
                renderWidth: renderW,
                renderHeight: renderH,
                params: p
            )
            log("Depth quantiles (m): min≈\(String(format: "%.3f", depth.min)) focus≈\(String(format: "%.3f", depth.focus)) max≈\(String(format: "%.3f", depth.max))")

            let renderer = try GaussianSplatRenderer(device: device)
            let videoURL = args.outDir.appendingPathComponent("out.mp4")
            let videoWriter = try MP4VideoWriter(url: videoURL, width: renderW, height: renderH, fps: args.fps)

            var renderOptions = GaussianSplatRenderOptions()
            renderOptions.compositing = args.compositing
            renderOptions.renderScale = args.renderScale
            renderOptions.toneMap = args.toneMap
            renderOptions.exposureEV = args.exposureEV
            renderOptions.saturation = args.saturation
            renderOptions.contrast = args.contrast
            renderOptions.debugView = args.debugView
            renderOptions.debugDepthRange = SIMD2<Float>(depth.min, depth.max)
            renderOptions.nearClipZ = args.nearClipZ
            renderOptions.opacityThreshold = args.opacityThreshold
            renderOptions.lowPassEps2D = args.lowPassEps2D
            renderOptions.minRadiusPx = args.minRadiusPx
            renderOptions.maxRadiusPx = args.maxRadiusPx
            renderOptions.normalization = args.normalization
            renderOptions.normalizationScale = args.normalizationScale

            var preview: CGImage? = nil
            for t in 0..<cameras.count {
                if t == 0 { log("Rendering \(cameras.count) frames (\(renderW)x\(renderH)) → \(videoURL.lastPathComponent)…") }
                let cam = cameras[t]

                let img = try renderer.renderToCGImage(scene: scene, camera: cam, width: renderW, height: renderH, options: renderOptions)
                if preview == nil { preview = img }
                try videoWriter.append(img)
            }

            if let preview {
                let pngURL = args.outDir.appendingPathComponent("preview.png")
                try writePNG(preview, to: pngURL)
                log("Wrote preview: \(pngURL.path)")
            }

            log("Finishing video…")
            try await videoWriter.finish(timeoutSeconds: 60.0)
            log("Wrote video: \(videoURL.path)")
        } catch {
            log("Fatal error: \(error)")
            exit(1)
        }
    }
}
