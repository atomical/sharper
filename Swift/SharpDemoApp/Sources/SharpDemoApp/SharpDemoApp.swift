import CoreML
import Foundation
import Dispatch
import ImageIO
import Metal
import SharpCoreML
import GaussianSplatMetalRenderer
import UniformTypeIdentifiers
import Darwin
import simd

@inline(__always)
func log(_ msg: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    print("[\(ts)] \(msg)")
    fflush(stdout)
}

@inline(__always)
func timed<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
    let t0 = CFAbsoluteTimeGetCurrent()
    do {
        let v = try block()
        let dt = CFAbsoluteTimeGetCurrent() - t0
        log("\(label) (\(String(format: "%.3fs", dt)))")
        return v
    } catch {
        let dt = CFAbsoluteTimeGetCurrent() - t0
        log("\(label) failed after \(String(format: "%.3fs", dt)): \(error)")
        throw error
    }
}

struct Args {
    enum Mode {
        case predict
        case render
    }

    var mode: Mode
    var inputPath: String
    var outDir: String
    var modelPath: String
    var computeUnits: MLComputeUnits = .all
    var frames: Int = 60
    var width: Int = 512
    var height: Int = 512
    var mlsharpSize: Bool = false
    var render: Bool = true
    var videoPath: String? = nil
    var fps: Int = 30
    var benchOut: String? = nil
    var iters: Int = 1

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

private enum RenderQualityPreset {
    case lessFog

    init?(rawValue: String) {
        switch rawValue {
        case "less_fog", "less-fog":
            self = .lessFog
        default:
            return nil
        }
    }
}

private func applyQualityPreset(_ preset: RenderQualityPreset, to args: inout Args) {
    switch preset {
    case .lessFog:
        args.compositing = .depthBinnedAlpha(binCount: 256)
        args.opacityThreshold = 0.01
        args.nearClipZ = 0.05
        args.renderScale = 2.0
        args.toneMap = .aces
    }
}

func parseArgs() -> Args? {
    var argv = CommandLine.arguments.dropFirst()
    guard argv.count >= 2 else {
        print("Usage:")
        print("  SharpDemoApp [predict] <image_path> <out_dir> [--model <mlpackage>] [--compute-units all|cpu_only|cpu_and_gpu|cpu_and_ne] [--frames N] [--size WxH] [--mlsharp-size] [--video out.mp4] [--fps N] [--no-render]")
        print("             [--render-scale S] [--compositing oit|bins[:N]] [--quality-preset less_fog] [--tonemap none|reinhard|aces] [--exposure-ev EV] [--saturation S] [--contrast C]")
        print("             [--debug none|alpha|depth|disparity|radius] [--near-clip Z] [--opacity-threshold A] [--lowpass-eps2d E] [--min-radius PX] [--max-radius PX]")
        print("             [--normalize none|recenter_xy|recenter_xyz] [--normalize-scale none|unit_radius] [--lookat point|ahead] [--bench-out bench.json] [--iters N]")
        print("  SharpDemoApp render <scene.ply> <out_dir> [--frames N] [--size WxH] [--mlsharp-size] [--video out.mp4] [--fps N]")
        print("             [same render quality flags as predict]")
        return nil
    }

    var explicitMode: Args.Mode? = nil
    if let first = argv.first, first == "predict" || first == "render" {
        explicitMode = (first == "render") ? .render : .predict
        argv = argv.dropFirst()
        guard argv.count >= 2 else { return nil }
    }

    let inputPath = String(argv.removeFirst())
    let outDir = String(argv.removeFirst())

    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let defaultModel = cwd.appendingPathComponent("../../artifacts/Sharp.mlpackage").path
    let inferredMode: Args.Mode = explicitMode ?? (inputPath.lowercased().hasSuffix(".ply") ? .render : .predict)
    var args = Args(mode: inferredMode, inputPath: inputPath, outDir: outDir, modelPath: defaultModel)

    while let tok = argv.first {
        argv = argv.dropFirst()
        switch tok {
        case "--model":
            guard let v = argv.first else { return nil }
            argv = argv.dropFirst()
            args.modelPath = String(v)
        case "--frames":
            guard let v = argv.first, let n = Int(v) else { return nil }
            argv = argv.dropFirst()
            args.frames = n
        case "--size":
            guard let v = argv.first else { return nil }
            argv = argv.dropFirst()
            let parts = v.split(separator: "x")
            guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) else { return nil }
            args.width = w
            args.height = h
        case "--mlsharp-size":
            args.mlsharpSize = true
        case "--render-scale":
            guard let v = argv.first, let s = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.renderScale = s
        case "--compositing":
            guard let v = argv.first else { return nil }
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
                print("Unknown --compositing: \(v)")
                return nil
            }
        case "--quality-preset":
            guard let v = argv.first, let preset = RenderQualityPreset(rawValue: String(v)) else {
                print("Unknown --quality-preset")
                return nil
            }
            argv = argv.dropFirst()
            applyQualityPreset(preset, to: &args)
        case "--tonemap":
            guard let v = argv.first, let tm = GaussianSplatToneMap(rawValue: String(v)) else { return nil }
            argv = argv.dropFirst()
            args.toneMap = tm
        case "--exposure-ev":
            guard let v = argv.first, let ev = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.exposureEV = ev
        case "--saturation":
            guard let v = argv.first, let s = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.saturation = s
        case "--contrast":
            guard let v = argv.first, let c = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.contrast = c
        case "--debug":
            guard let v = argv.first, let dv = GaussianSplatDebugView(rawValue: String(v)) else { return nil }
            argv = argv.dropFirst()
            args.debugView = dv
        case "--near-clip":
            guard let v = argv.first, let z = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.nearClipZ = z
        case "--opacity-threshold":
            guard let v = argv.first, let a = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.opacityThreshold = a
        case "--lowpass-eps2d":
            guard let v = argv.first, let e = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.lowPassEps2D = e
        case "--min-radius":
            guard let v = argv.first, let px = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.minRadiusPx = px
        case "--max-radius":
            guard let v = argv.first, let px = Float(v) else { return nil }
            argv = argv.dropFirst()
            args.maxRadiusPx = px
        case "--normalize":
            guard let v = argv.first else { return nil }
            argv = argv.dropFirst()
            switch v {
            case "none":
                args.normalization = .none
            case "recenter_xy":
                args.normalization = .recenterXY
            case "recenter_xyz":
                args.normalization = .recenterXYZ
            default:
                print("Unknown --normalize: \(v)")
                return nil
            }
        case "--normalize-scale":
            guard let v = argv.first else { return nil }
            argv = argv.dropFirst()
            switch v {
            case "none":
                args.normalizationScale = .none
            case "unit_radius":
                args.normalizationScale = .unitRadius
            default:
                print("Unknown --normalize-scale: \(v)")
                return nil
            }
        case "--lookat":
            guard let v = argv.first, let lm = MLSharpTrajectoryParams.LookAtMode(rawValue: String(v)) else { return nil }
            argv = argv.dropFirst()
            args.lookAtMode = lm
        case "--no-render":
            args.render = false
        case "--video":
            guard let v = argv.first else { return nil }
            argv = argv.dropFirst()
            args.videoPath = String(v)
        case "--fps":
            guard let v = argv.first, let n = Int(v) else { return nil }
            argv = argv.dropFirst()
            args.fps = n
        case "--bench-out":
            guard let v = argv.first else { return nil }
            argv = argv.dropFirst()
            args.benchOut = String(v)
        case "--iters":
            guard let v = argv.first, let n = Int(v), n >= 1 else { return nil }
            argv = argv.dropFirst()
            args.iters = n
        case "--compute-units":
            guard let v = argv.first else { return nil }
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
                print("Unknown --compute-units: \(v)")
                return nil
            }
        default:
            print("Unknown arg: \(tok)")
            return nil
        }
    }

    return args
}

func writePNG(_ image: CGImage, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "SharpDemoApp", code: 1)
    }
    CGImageDestinationAddImage(dest, image, nil)
    if !CGImageDestinationFinalize(dest) {
        throw NSError(domain: "SharpDemoApp", code: 2)
    }
}

private func residentMemoryBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { infoPtr in
        infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
        }
    }
    guard kerr == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}

private struct BenchStats: Codable {
    var mean: Double
    var min: Double
    var max: Double
    var p50: Double
    var p90: Double
}

private func quantile(_ xs: [Double], q: Double) -> Double {
    guard !xs.isEmpty else { return 0 }
    if xs.count == 1 { return xs[0] }
    let sorted = xs.sorted()
    let pos = q * Double(sorted.count - 1)
    let lo = Int(floor(pos))
    let hi = Int(ceil(pos))
    if lo == hi { return sorted[lo] }
    let t = pos - Double(lo)
    return sorted[lo] * (1.0 - t) + sorted[hi] * t
}

private func stats(_ xs: [Double]) -> BenchStats {
    let mean = xs.reduce(0, +) / Double(max(xs.count, 1))
    return BenchStats(
        mean: mean,
        min: xs.min() ?? 0,
        max: xs.max() ?? 0,
        p50: quantile(xs, q: 0.5),
        p90: quantile(xs, q: 0.9)
    )
}

private struct BenchReport: Codable {
    var timestamp: String
    var model: String
    var image: String
    var computeUnits: String
    var iters: Int
    var frames: Int
    var size: String
    var runnerInitSec: Double
    var predict: [String: BenchStats]
    var plyWriteSec: Double
    var plyLoadSec: Double
    var rendererInitSec: Double
    var renderFrameSec: BenchStats
    var renderFps: Double
    var rssBytesBefore: UInt64
    var rssBytesPeak: UInt64
}

@main
struct SharpDemoApp {
    static func main() async {
        do {
            guard let args = parseArgs() else { exit(2) }

            let inputURL = URL(fileURLWithPath: args.inputPath)
            let outURL = URL(fileURLWithPath: args.outDir, isDirectory: true)
            let modelURL = URL(fileURLWithPath: args.modelPath)

            log("Model: \(modelURL.path)")
            log("Input: \(inputURL.path)")
            log("Out: \(outURL.path)")
            log("ComputeUnits: \(args.computeUnits)")

            let rssBefore = residentMemoryBytes()
            var rssPeak = rssBefore

            // Render-only mode: PLY -> frames/video (no CoreML).
            if args.mode == .render {
                let plyURL = inputURL
                log("Mode: render")

                guard args.render else {
                    log("Render disabled (--no-render); nothing to do.")
                    return
                }

                let device = MTLCreateSystemDefaultDevice()
                let (scene, meta) = try timed("Load PLY") { try PLYLoader.loadMLSharpCompatiblePLYWithMetadata(url: plyURL, device: device) }
                let renderer = try timed("Init renderer") { try GaussianSplatRenderer(device: device) }
                rssPeak = max(rssPeak, residentMemoryBytes())

                var renderW = args.width
                var renderH = args.height
                if args.mlsharpSize, let meta, meta.imageWidth > 0, meta.imageHeight > 0 {
                    let r = MLSharpTrajectory.screenResolutionPxFromInput(width: Int(meta.imageWidth), height: Int(meta.imageHeight))
                    renderW = r.width
                    renderH = r.height
                    log("Using ml-sharp screen resolution: \(renderW)x\(renderH)")
                }

                if args.videoPath != nil {
                    if renderW % 2 != 0 { renderW += 1 }
                    if renderH % 2 != 0 { renderH += 1 }
                }

                let (cameras, depth) = timed("Compute trajectory") {
                    var p = MLSharpTrajectoryParams()
                    p.kind = .rotateForward
                    p.numSteps = args.frames
                    p.lookAtMode = args.lookAtMode

                    if let meta, meta.intrinsic.count >= 9, meta.imageWidth > 0, meta.imageHeight > 0 {
                        return MLSharpTrajectory.makeCameras(
                            scene: scene,
                            sourceImageWidth: Int(meta.imageWidth),
                            sourceImageHeight: Int(meta.imageHeight),
                            intrinsicFx: meta.intrinsic[0],
                            intrinsicFy: meta.intrinsic[4],
                            intrinsicCx: (Float(meta.imageWidth) - 1) * 0.5,
                            intrinsicCy: (Float(meta.imageHeight) - 1) * 0.5,
                            renderWidth: renderW,
                            renderHeight: renderH,
                            params: p
                        )
                    }

                    // Fallback: 60° horizontal FOV.
                    let fovX: Float = 60.0 * Float.pi / 180.0
                    let fx = 0.5 * Float(renderW) / tan(0.5 * fovX)
                    let cx = (Float(renderW) - 1) * 0.5
                    let cy = (Float(renderH) - 1) * 0.5
                    return MLSharpTrajectory.makeCameras(
                        scene: scene,
                        sourceImageWidth: renderW,
                        sourceImageHeight: renderH,
                        intrinsicFx: fx,
                        intrinsicFy: fx,
                        intrinsicCx: cx,
                        intrinsicCy: cy,
                        renderWidth: renderW,
                        renderHeight: renderH,
                        params: p
                    )
                }

                if let meta, meta.intrinsic.count >= 9, meta.imageWidth > 0, meta.imageHeight > 0 {
                    log("Using PLY intrinsics: src=\(meta.imageWidth)x\(meta.imageHeight) fx=\(String(format: "%.2f", meta.intrinsic[0]))")
                } else {
                    log("Using fallback intrinsics (60° FOV).")
                }
                log("Depth quantiles (m): min≈\(String(format: "%.3f", depth.min)) focus≈\(String(format: "%.3f", depth.focus)) max≈\(String(format: "%.3f", depth.max))")

                try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

                let framesDir = outURL.appendingPathComponent("frames", isDirectory: true)
                let videoURL = args.videoPath.map { URL(fileURLWithPath: $0) }
                var videoWriter: MP4VideoWriter? = nil
                if let videoURL {
                    log("Video: \(videoURL.path)")
                    try FileManager.default.createDirectory(at: videoURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                    videoWriter = try MP4VideoWriter(url: videoURL, width: renderW, height: renderH, fps: args.fps)
                }

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

                for t in 0..<cameras.count {
                    autoreleasepool {
                        log("Frame \(t + 1)/\(cameras.count)")
                        let cam = cameras[t]

                        do {
                            let img = try renderer.renderToCGImage(scene: scene, camera: cam, width: renderW, height: renderH, options: renderOptions)
                            let frameURL = framesDir.appendingPathComponent(String(format: "frame_%04d.png", t))
                            try writePNG(img, to: frameURL)
                            if let videoWriter {
                                try videoWriter.append(img)
                            }
                        } catch {
                            log("Frame \(t) failed: \(error)")
                            exit(1)
                        }
                    }
                }
                log("Wrote frames: \(framesDir.path)")

                if let videoWriter, let videoURL {
                    log("Finishing video...")
                    try await videoWriter.finish(timeoutSeconds: 60.0)
                    log("Wrote video: \(videoURL.path)")
                }
                return
            }

            log("Mode: predict")

            let imageURL = inputURL

            let runnerStart = CFAbsoluteTimeGetCurrent()
            let runner = try SharpCoreMLRunner(modelURL: modelURL, computeUnits: args.computeUnits)
            let runnerInitSec = CFAbsoluteTimeGetCurrent() - runnerStart
            log("Init runner (\(String(format: "%.3fs", runnerInitSec)))")
            rssPeak = max(rssPeak, residentMemoryBytes())

            var timingSamples: [SharpTimings] = []
            timingSamples.reserveCapacity(args.iters)
            var lastPrediction: SharpPrediction? = nil
            for i in 0..<args.iters {
                autoreleasepool {
                    do {
                        let pred = try runner.predict(imageURL: imageURL)
                        if let t = pred.timings { timingSamples.append(t) }
                        lastPrediction = pred
                        log("Predict \(i + 1)/\(args.iters) (\(String(format: "%.3fs", pred.timings?.totalSec ?? 0)))")
                        rssPeak = max(rssPeak, residentMemoryBytes())
                    } catch {
                        log("Predict \(i + 1) failed: \(error)")
                        exit(1)
                    }
                }
            }
            guard let prediction = lastPrediction else {
                log("No prediction produced")
                exit(1)
            }

            log("Metadata: \(prediction.metadata.imageWidth)x\(prediction.metadata.imageHeight) f_px=\(String(format: "%.2f", prediction.metadata.focalLengthPx)) disparity=\(String(format: "%.6f", prediction.metadata.disparityFactor))")

            let plyURL = outURL.appendingPathComponent("scene.ply")
            let plyWriteStart = CFAbsoluteTimeGetCurrent()
            try SharpPLYWriter.write(prediction: prediction, to: plyURL)
            let plyWriteSec = CFAbsoluteTimeGetCurrent() - plyWriteStart
            log("Write PLY (\(String(format: "%.3fs", plyWriteSec)))")
            log("Wrote PLY: \(plyURL.path)")
            rssPeak = max(rssPeak, residentMemoryBytes())

            var renderW = args.width
            var renderH = args.height
            if args.mlsharpSize {
                let r = MLSharpTrajectory.screenResolutionPxFromInput(width: prediction.metadata.imageWidth, height: prediction.metadata.imageHeight)
                renderW = r.width
                renderH = r.height
                log("Using ml-sharp screen resolution: \(renderW)x\(renderH)")
            }

            if args.videoPath != nil {
                if renderW % 2 != 0 { renderW += 1 }
                if renderH % 2 != 0 { renderH += 1 }
            }

            if let benchOut = args.benchOut {
                let device = MTLCreateSystemDefaultDevice()
                let plyLoadStart = CFAbsoluteTimeGetCurrent()
                let scene = try PLYLoader.loadMLSharpCompatiblePLY(url: plyURL, device: device)
                let plyLoadSec = CFAbsoluteTimeGetCurrent() - plyLoadStart
                log("Load PLY (\(String(format: "%.3fs", plyLoadSec)))")

                let rendererStart = CFAbsoluteTimeGetCurrent()
                let renderer = try GaussianSplatRenderer(device: device)
                let rendererInitSec = CFAbsoluteTimeGetCurrent() - rendererStart
                log("Init renderer (\(String(format: "%.3fs", rendererInitSec)))")
                rssPeak = max(rssPeak, residentMemoryBytes())

                let (cameras, depth) = timed("Compute trajectory") {
                    var p = MLSharpTrajectoryParams()
                    p.kind = .rotateForward
                    p.numSteps = args.frames
                    p.lookAtMode = args.lookAtMode
                    let fx0 = Float(prediction.metadata.focalLengthPx)
                    let cx0 = (Float(prediction.metadata.imageWidth) - 1) * 0.5
                    let cy0 = (Float(prediction.metadata.imageHeight) - 1) * 0.5
                    return MLSharpTrajectory.makeCameras(
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
                }
                log("Depth quantiles (m): min≈\(String(format: "%.3f", depth.min)) focus≈\(String(format: "%.3f", depth.focus)) max≈\(String(format: "%.3f", depth.max))")

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

                var renderFrameTimes: [Double] = []
                renderFrameTimes.reserveCapacity(args.frames)
                let renderTotalStart = CFAbsoluteTimeGetCurrent()
                for t in 0..<cameras.count {
                    let frameStart = CFAbsoluteTimeGetCurrent()
                    let cam = cameras[t]
                    _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: renderW, height: renderH, options: renderOptions)
                    let dt = CFAbsoluteTimeGetCurrent() - frameStart
                    renderFrameTimes.append(dt)
                    rssPeak = max(rssPeak, residentMemoryBytes())
                }
                let renderTotalSec = CFAbsoluteTimeGetCurrent() - renderTotalStart
                let renderCount = max(cameras.count, 1)
                let renderFps = Double(renderCount) / max(renderTotalSec, 1e-9)
                log("Render \(renderCount) frames (\(String(format: "%.3fs", renderTotalSec))) -> \(String(format: "%.2f", renderFps)) FPS")

                let unitStr: String = switch args.computeUnits {
                case .all: "all"
                case .cpuOnly: "cpu_only"
                case .cpuAndGPU: "cpu_and_gpu"
                case .cpuAndNeuralEngine: "cpu_and_ne"
                @unknown default: "unknown"
                }

                var predictStats: [String: BenchStats] = [:]
                predictStats["preprocess_sec"] = stats(timingSamples.map { $0.preprocessSec })
                predictStats["coreml_sec"] = stats(timingSamples.map { $0.coremlSec })
                predictStats["postprocess_sec"] = stats(timingSamples.map { $0.postprocessSec })
                predictStats["postprocess_copy_sec"] = stats(timingSamples.map { $0.postprocessCopySec })
                predictStats["postprocess_kernel_sec"] = stats(timingSamples.map { $0.postprocessKernelSec })
                predictStats["total_sec"] = stats(timingSamples.map { $0.totalSec })

                let report = BenchReport(
                    timestamp: ISO8601DateFormatter().string(from: Date()),
                    model: modelURL.path,
                    image: imageURL.path,
                    computeUnits: unitStr,
                    iters: args.iters,
                    frames: args.frames,
                    size: "\(renderW)x\(renderH)",
                    runnerInitSec: runnerInitSec,
                    predict: predictStats,
                    plyWriteSec: plyWriteSec,
                    plyLoadSec: plyLoadSec,
                    rendererInitSec: rendererInitSec,
                    renderFrameSec: stats(renderFrameTimes),
                    renderFps: renderFps,
                    rssBytesBefore: rssBefore,
                    rssBytesPeak: rssPeak
                )

                let outPath = URL(fileURLWithPath: benchOut)
                try FileManager.default.createDirectory(at: outPath.deletingLastPathComponent(), withIntermediateDirectories: true)
                let enc = JSONEncoder()
                enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try enc.encode(report)
                try data.write(to: outPath)
                log("Wrote bench: \(outPath.path)")
                return
            }

            guard args.render else { return }

            let device = MTLCreateSystemDefaultDevice()
            let scene = try timed("Load PLY") { try PLYLoader.loadMLSharpCompatiblePLY(url: plyURL, device: device) }
            let renderer = try timed("Init renderer") { try GaussianSplatRenderer(device: device) }

            let (cameras, depth) = timed("Compute trajectory") {
                var p = MLSharpTrajectoryParams()
                p.kind = .rotateForward
                p.numSteps = args.frames
                p.lookAtMode = args.lookAtMode
                let fx0 = Float(prediction.metadata.focalLengthPx)
                let cx0 = (Float(prediction.metadata.imageWidth) - 1) * 0.5
                let cy0 = (Float(prediction.metadata.imageHeight) - 1) * 0.5
                return MLSharpTrajectory.makeCameras(
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
            }
            log("Depth quantiles (m): min≈\(String(format: "%.3f", depth.min)) focus≈\(String(format: "%.3f", depth.focus)) max≈\(String(format: "%.3f", depth.max))")

            let framesDir = outURL.appendingPathComponent("frames", isDirectory: true)
            let videoURL = args.videoPath.map { URL(fileURLWithPath: $0) }
            var videoWriter: MP4VideoWriter? = nil
            if let videoURL {
                log("Video: \(videoURL.path)")
                videoWriter = try MP4VideoWriter(url: videoURL, width: renderW, height: renderH, fps: args.fps)
            }

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

            for t in 0..<cameras.count {
                autoreleasepool {
                    log("Frame \(t + 1)/\(cameras.count)")
                    let cam = cameras[t]

                    do {
                        let img = try renderer.renderToCGImage(scene: scene, camera: cam, width: renderW, height: renderH, options: renderOptions)
                        let frameURL = framesDir.appendingPathComponent(String(format: "frame_%04d.png", t))
                        try writePNG(img, to: frameURL)
                        if let videoWriter {
                            try videoWriter.append(img)
                        }
                    } catch {
                        log("Frame \(t) failed: \(error)")
                        exit(1)
                    }
                }
            }
            log("Wrote frames: \(framesDir.path)")

            if let videoWriter, let videoURL {
                log("Finishing video...")
                try await videoWriter.finish(timeoutSeconds: 60.0)
                log("Wrote video: \(videoURL.path)")
            }
        } catch {
            log("Fatal error: \(error)")
            exit(1)
        }
    }
}
