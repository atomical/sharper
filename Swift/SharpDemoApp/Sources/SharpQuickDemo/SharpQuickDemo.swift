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
    var fps: Int = 30
    var computeUnits: MLComputeUnits = .all
    var render: Bool = true
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
                                                  [--frames N] [--size WxH] [--fps N]
                                                  [--compute-units all|cpu_only|cpu_and_gpu|cpu_and_ne]
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

private func computeBounds(scene: GaussianScene) -> (SIMD3<Float>, Float) {
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
    let ext = maxV - minV
    let radius = max(0.5, simd_length(ext) * 0.6)
    return (center, radius)
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
            let (center, radius) = computeBounds(scene: scene)

            let fx = prediction.metadata.focalLengthPx * Float(args.width) / Float(prediction.metadata.imageWidth)
            let fy = prediction.metadata.focalLengthPx * Float(args.height) / Float(prediction.metadata.imageHeight)
            let cx = Float(args.width) * 0.5
            let cy = Float(args.height) * 0.5

            let renderer = try GaussianSplatRenderer(device: device)
            let videoURL = args.outDir.appendingPathComponent("out.mp4")
            let videoWriter = try MP4VideoWriter(url: videoURL, width: args.width, height: args.height, fps: args.fps)

            var preview: CGImage? = nil
            for t in 0..<args.frames {
                if t == 0 { log("Rendering \(args.frames) frames (\(args.width)x\(args.height)) → \(videoURL.lastPathComponent)…") }
                let ang = Float(t) * 2.0 * Float.pi / Float(max(args.frames, 1))
                let eye = center + SIMD3<Float>(radius * sin(ang), 0, radius * cos(ang))
                let view = PinholeCamera.lookAt(eye: eye, target: center)
                let cam = PinholeCamera(viewMatrix: view, fx: fx, fy: fy, cx: cx, cy: cy)

                let img = try renderer.renderToCGImage(scene: scene, camera: cam, width: args.width, height: args.height)
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

