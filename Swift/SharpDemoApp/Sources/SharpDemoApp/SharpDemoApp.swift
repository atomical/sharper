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
    var imagePath: String
    var outDir: String
    var modelPath: String
    var computeUnits: MLComputeUnits = .all
    var frames: Int = 60
    var width: Int = 512
    var height: Int = 512
    var render: Bool = true
    var videoPath: String? = nil
    var fps: Int = 30
    var benchOut: String? = nil
    var iters: Int = 1
}

func parseArgs() -> Args? {
    var argv = CommandLine.arguments.dropFirst()
    guard argv.count >= 2 else {
        print("Usage: SharpDemoApp <image_path> <out_dir> [--model <mlpackage>] [--compute-units all|cpu_only|cpu_and_gpu|cpu_and_ne] [--frames N] [--size WxH] [--video out.mp4] [--fps N] [--no-render] [--bench-out bench.json] [--iters N]")
        return nil
    }

    let imagePath = String(argv.removeFirst())
    let outDir = String(argv.removeFirst())

    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let defaultModel = cwd.appendingPathComponent("../../artifacts/Sharp.mlpackage").path
    var args = Args(imagePath: imagePath, outDir: outDir, modelPath: defaultModel)

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

            let imageURL = URL(fileURLWithPath: args.imagePath)
            let outURL = URL(fileURLWithPath: args.outDir, isDirectory: true)
            let modelURL = URL(fileURLWithPath: args.modelPath)

            log("Model: \(modelURL.path)")
            log("Input: \(imageURL.path)")
            log("Out: \(outURL.path)")
            log("ComputeUnits: \(args.computeUnits)")

            let rssBefore = residentMemoryBytes()
            var rssPeak = rssBefore

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

                // Compute a bounding box to pick an orbit radius.
                let (center, radius): (SIMD3<Float>, Float) = timed("Compute bounds") {
                    let meanPtr = scene.means.contents().bindMemory(to: Float.self, capacity: scene.count * 3)
                    var minV = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
                    var maxV = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
                    for i in 0..<scene.count {
                        let x = meanPtr[i * 3 + 0]
                        let y = meanPtr[i * 3 + 1]
                        let z = meanPtr[i * 3 + 2]
                        minV = SIMD3<Float>(min(minV.x, x), min(minV.y, y), min(minV.z, z))
                        maxV = SIMD3<Float>(max(maxV.x, x), max(maxV.y, y), max(maxV.z, z))
                    }
                    let center = (minV + maxV) * 0.5
                    let ext = (maxV - minV)
                    let radius = max(0.5, simd_length(ext) * 0.6)
                    return (center, radius)
                }

                let fx = prediction.metadata.focalLengthPx * Float(args.width) / Float(prediction.metadata.imageWidth)
                let fy = prediction.metadata.focalLengthPx * Float(args.height) / Float(prediction.metadata.imageHeight)
                let cx = Float(args.width) * 0.5
                let cy = Float(args.height) * 0.5

                var renderFrameTimes: [Double] = []
                renderFrameTimes.reserveCapacity(args.frames)
                let renderTotalStart = CFAbsoluteTimeGetCurrent()
                for t in 0..<args.frames {
                    let frameStart = CFAbsoluteTimeGetCurrent()
                    let ang = Float(t) * 2.0 * Float.pi / Float(max(args.frames, 1))
                    let eye = center + SIMD3<Float>(radius * sin(ang), 0, radius * cos(ang))
                    let view = PinholeCamera.lookAt(eye: eye, target: center)
                    let cam = PinholeCamera(viewMatrix: view, fx: fx, fy: fy, cx: cx, cy: cy)
                    _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: args.width, height: args.height)
                    let dt = CFAbsoluteTimeGetCurrent() - frameStart
                    renderFrameTimes.append(dt)
                    rssPeak = max(rssPeak, residentMemoryBytes())
                }
                let renderTotalSec = CFAbsoluteTimeGetCurrent() - renderTotalStart
                let renderFps = Double(args.frames) / max(renderTotalSec, 1e-9)
                log("Render \(args.frames) frames (\(String(format: "%.3fs", renderTotalSec))) -> \(String(format: "%.2f", renderFps)) FPS")

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
                    size: "\(args.width)x\(args.height)",
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

            // Compute a bounding box to pick an orbit radius.
            let (center, radius): (SIMD3<Float>, Float) = timed("Compute bounds") {
                let meanPtr = scene.means.contents().bindMemory(to: Float.self, capacity: scene.count * 3)
                var minV = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
                var maxV = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
                for i in 0..<scene.count {
                    let x = meanPtr[i * 3 + 0]
                    let y = meanPtr[i * 3 + 1]
                    let z = meanPtr[i * 3 + 2]
                    minV = SIMD3<Float>(min(minV.x, x), min(minV.y, y), min(minV.z, z))
                    maxV = SIMD3<Float>(max(maxV.x, x), max(maxV.y, y), max(maxV.z, z))
                }
                let center = (minV + maxV) * 0.5
                let ext = (maxV - minV)
                let radius = max(0.5, simd_length(ext) * 0.6)
                return (center, radius)
            }

            let fx = prediction.metadata.focalLengthPx * Float(args.width) / Float(prediction.metadata.imageWidth)
            let fy = prediction.metadata.focalLengthPx * Float(args.height) / Float(prediction.metadata.imageHeight)
            let cx = Float(args.width) * 0.5
            let cy = Float(args.height) * 0.5

            let framesDir = outURL.appendingPathComponent("frames", isDirectory: true)
            let videoURL = args.videoPath.map { URL(fileURLWithPath: $0) }
            var videoWriter: MP4VideoWriter? = nil
            if let videoURL {
                log("Video: \(videoURL.path)")
                videoWriter = try MP4VideoWriter(url: videoURL, width: args.width, height: args.height, fps: args.fps)
            }

            for t in 0..<args.frames {
                autoreleasepool {
                    log("Frame \(t + 1)/\(args.frames)")
                    let ang = Float(t) * 2.0 * Float.pi / Float(max(args.frames, 1))
                    let eye = center + SIMD3<Float>(radius * sin(ang), 0, radius * cos(ang))
                    let view = PinholeCamera.lookAt(eye: eye, target: center)
                    let cam = PinholeCamera(viewMatrix: view, fx: fx, fy: fy, cx: cx, cy: cy)

                    do {
                        let img = try renderer.renderToCGImage(scene: scene, camera: cam, width: args.width, height: args.height)
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
