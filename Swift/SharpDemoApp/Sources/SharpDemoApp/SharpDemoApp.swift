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
    var frames: Int = 60
    var width: Int = 512
    var height: Int = 512
    var render: Bool = true
    var videoPath: String? = nil
    var fps: Int = 30
}

func parseArgs() -> Args? {
    var argv = CommandLine.arguments.dropFirst()
    guard argv.count >= 2 else {
        print("Usage: SharpDemoApp <image_path> <out_dir> [--model <mlpackage>] [--frames N] [--size WxH] [--video out.mp4] [--fps N] [--no-render]")
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

            let runner = try timed("Init runner") { try SharpCoreMLRunner(modelURL: modelURL) }
            let prediction = try timed("Predict") { try runner.predict(imageURL: imageURL) }

            let plyURL = outURL.appendingPathComponent("scene.ply")
            try timed("Write PLY") { try SharpPLYWriter.write(prediction: prediction, to: plyURL) }
            log("Wrote PLY: \(plyURL.path)")

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
