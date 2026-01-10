import AVFoundation
import CoreGraphics
import Foundation
import Metal
import simd
import XCTest
@testable import GaussianSplatMetalRenderer

final class GaussianSplatMetalRendererTests: XCTestCase {
    enum TestError: Error, Equatable {
        case test
    }

    private func requireMetalDevice(file: StaticString = #filePath, line: UInt = #line) throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal device unavailable", file: file, line: line)
            throw TestError.test
        }
        return device
    }

    private func makeFloatBuffer(device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        try values.withUnsafeBytes { raw in
            try device
                .makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
                .orThrow(TestError.test)
        }
    }

    private func makeU32Buffer(device: MTLDevice, _ values: [UInt32]) throws -> MTLBuffer {
        try values.withUnsafeBytes { raw in
            try device
                .makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
                .orThrow(TestError.test)
        }
    }

    private func makeScene(
        device: MTLDevice,
        count: Int,
        opacity: Float,
        z: Float = 1.0,
        makeSomeLowOpacity: Bool = false
    ) throws -> GaussianScene {
        let n = max(count, 0)
        let means: [Float] = (0..<n).flatMap { i in
            let x = (Float(i % 32) - 16) * 0.01
            let y = (Float(i / 32) - 16) * 0.01
            let zz = z + Float(i) * 0.001
            return [x, y, zz]
        }
        let quats: [Float] = (0..<n).flatMap { _ in [1, 0, 0, 0] } // identity (wxyz)
        let scales: [Float] = (0..<n).flatMap { _ in [0.02, 0.02, 0.02] }
        let colors: [Float] = (0..<n).flatMap { i -> [Float] in
            switch i % 3 {
            case 0: return [1, 0, 0]
            case 1: return [0, 1, 0]
            default: return [0, 0, 1]
            }
        }
        let opas: [Float] = (0..<n).map { i in
            guard makeSomeLowOpacity else { return opacity }
            return (i % 7 == 0) ? 0.0 : opacity
        }

        let meansBuf = try makeFloatBuffer(device: device, means.isEmpty ? [0] : means)
        let quatBuf = try makeFloatBuffer(device: device, quats.isEmpty ? [0, 0, 0, 1] : quats)
        let scalesBuf = try makeFloatBuffer(device: device, scales.isEmpty ? [0.02, 0.02, 0.02] : scales)
        let colorsBuf = try makeFloatBuffer(device: device, colors.isEmpty ? [0, 0, 0] : colors)
        let opaBuf = try makeFloatBuffer(device: device, opas.isEmpty ? [0] : opas)

        return GaussianScene(count: n, means: meansBuf, quaternions: quatBuf, scales: scalesBuf, colorsLinear: colorsBuf, opacities: opaBuf)
    }

    private func makeSolidImage(width: Int, height: Int, rgba: (UInt8, UInt8, UInt8, UInt8)) throws -> CGImage {
        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * bytesPerRow + x * 4
                bytes[idx + 0] = rgba.2 // B
                bytes[idx + 1] = rgba.1 // G
                bytes[idx + 2] = rgba.0 // R
                bytes[idx + 3] = rgba.3 // A
            }
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo.byteOrder32Little.union(.init(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
        let provider = try CGDataProvider(data: Data(bytes) as CFData).orThrow(TestError.test)
        return try CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: cs,
            bitmapInfo: info,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ).orThrow(TestError.test)
    }

    private func appendF32LE(_ x: Float, to data: inout Data) {
        var u = x.bitPattern.littleEndian
        withUnsafeBytes(of: &u) { data.append(contentsOf: $0) }
    }

    private func appendU32LE(_ x: UInt32, to data: inout Data) {
        var u = x.littleEndian
        withUnsafeBytes(of: &u) { data.append(contentsOf: $0) }
    }

    private func appendI32LE(_ x: Int32, to data: inout Data) {
        appendU32LE(UInt32(bitPattern: x), to: &data)
    }

    private struct TestPLYVertex {
        var x: Float
        var y: Float
        var z: Float
        var fdc0: Float
        var fdc1: Float
        var fdc2: Float
        var opacityLogit: Float
        var s0Log: Float
        var s1Log: Float
        var s2Log: Float
        var q0: Float
        var q1: Float
        var q2: Float
        var q3: Float
    }

    private struct TestPLYMetadata {
        var extrinsic: [Float] // 16
        var intrinsic: [Float] // 9
        var imageWidth: UInt32
        var imageHeight: UInt32
        var frameIndex: Int32
        var frameCount: Int32
        var disparityP10: Float
        var disparityP90: Float
        var colorSpace: UInt8
        var version: (UInt8, UInt8, UInt8)
    }

    private func makeMLSharpPLYData(vertices: [TestPLYVertex], metadata: TestPLYMetadata? = nil, format: String = "binary_little_endian") -> Data {
        var header = "ply\n"
        header += "format \(format) 1.0\n"
        header += "element vertex \(vertices.count)\n"
        if let _ = metadata {
            header += "element extrinsic 16\n"
            header += "element intrinsic 9\n"
            header += "element image_size 2\n"
            header += "element frame 2\n"
            header += "element disparity 2\n"
            header += "element color_space 1\n"
            header += "element version 3\n"
        }
        header += "end_header\n"

        var data = Data(header.utf8)
        for v in vertices {
            appendF32LE(v.x, to: &data)
            appendF32LE(v.y, to: &data)
            appendF32LE(v.z, to: &data)
            appendF32LE(v.fdc0, to: &data)
            appendF32LE(v.fdc1, to: &data)
            appendF32LE(v.fdc2, to: &data)
            appendF32LE(v.opacityLogit, to: &data)
            appendF32LE(v.s0Log, to: &data)
            appendF32LE(v.s1Log, to: &data)
            appendF32LE(v.s2Log, to: &data)
            appendF32LE(v.q0, to: &data)
            appendF32LE(v.q1, to: &data)
            appendF32LE(v.q2, to: &data)
            appendF32LE(v.q3, to: &data)
        }

        if let m = metadata {
            for f in m.extrinsic { appendF32LE(f, to: &data) }
            for f in m.intrinsic { appendF32LE(f, to: &data) }
            appendU32LE(m.imageWidth, to: &data)
            appendU32LE(m.imageHeight, to: &data)
            appendI32LE(m.frameIndex, to: &data)
            appendI32LE(m.frameCount, to: &data)
            appendF32LE(m.disparityP10, to: &data)
            appendF32LE(m.disparityP90, to: &data)
            data.append(m.colorSpace)
            data.append(m.version.0)
            data.append(m.version.1)
            data.append(m.version.2)
        }

        return data
    }

    private func writeTempFile(_ data: Data, name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("GaussianSplatMetalRendererTests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    func testPreconditionsHelpers() throws {
        try require(true, TestError.test)
        XCTAssertThrowsError(try require(false, TestError.test))

        let some: Int? = 123
        XCTAssertEqual(try some.orThrow(TestError.test), 123)

        let none: Int? = nil
        XCTAssertThrowsError(try none.orThrow(TestError.test))
    }

    func testRenderOptionsAndEnums() throws {
        XCTAssertGreaterThan(GaussianSplatCompositingMode.defaultDepthBinCount, 0)

        let opts = GaussianSplatRenderOptions()
        XCTAssertEqual(opts.compositing, .weightedOIT)
        XCTAssertEqual(opts.renderScale, 1.0)
        XCTAssertEqual(opts.toneMap, .none)
        XCTAssertEqual(opts.debugView, .none)
        XCTAssertEqual(opts.normalization, .none)
        XCTAssertEqual(opts.normalizationScale, .none)

        _ = GaussianSplatToneMap.allCases.map { $0.rawValue }
        _ = GaussianSplatDebugView.allCases.map { $0.rawValue }
        _ = GaussianSplatSceneNormalization.allCases.map { $0.rawValue }
        _ = GaussianSplatSceneScale.allCases.map { $0.rawValue }

        XCTAssertNotEqual(GaussianSplatCompositingMode.weightedOIT, .depthBinnedAlpha(binCount: 8))
        XCTAssertEqual(GaussianSplatCompositingMode.depthBinnedAlpha(binCount: 8), .depthBinnedAlpha(binCount: 8))
    }

    func testPinholeCameraLookAtFallback() throws {
        // Normal path.
        let m0 = PinholeCamera.lookAt(eye: .zero, target: SIMD3<Float>(0, 0, 1))
        XCTAssertEqual(m0.columns.3.w, 1)

        // Fallback path: `up` parallel to forward.
        let m1 = PinholeCamera.lookAt(eye: .zero, target: SIMD3<Float>(1, 0, 0), up: SIMD3<Float>(1, 0, 0))
        XCTAssert(m1.columns.0.x.isFinite)
        XCTAssert(m1.columns.1.y.isFinite)
        XCTAssert(m1.columns.2.z.isFinite)
    }

    func testSceneBoundsAndDepthQuantiles() throws {
        let device = try requireMetalDevice()

        let empty = try makeScene(device: device, count: 0, opacity: 1.0)
        XCTAssertEqual(GaussianSceneBounds.estimate(scene: empty).radius, 1)
        XCTAssertEqual(MLSharpTrajectory.depthQuantiles(scene: empty).focus, 2)

        let tooSmall = try makeScene(device: device, count: 8, opacity: 1.0, z: 0.0)
        XCTAssertEqual(GaussianSceneBounds.estimate(scene: tooSmall).radius, 1)
        XCTAssertEqual(MLSharpTrajectory.depthQuantiles(scene: tooSmall).focus, 2)

        // Fallback path (xs/zs < 1024) with opacity filter rejecting most points.
        let fallback = try makeScene(device: device, count: 32, opacity: 1.0, z: 1.0, makeSomeLowOpacity: true)
        let bFallback = GaussianSceneBounds.estimate(scene: fallback, sampleCount: 65536, opacityThreshold: 0.99)
        XCTAssertGreaterThan(bFallback.radius, 0)
        let dFallback = MLSharpTrajectory.depthQuantiles(scene: fallback, opacityThreshold: 0.99)
        XCTAssertGreaterThan(dFallback.max, 0)

        // No-fallback path (>= 1024 samples).
        let big = try makeScene(device: device, count: 2048, opacity: 1.0, z: 1.0)
        let bBig = GaussianSceneBounds.estimate(scene: big, sampleCount: 65536, opacityThreshold: 0.01)
        XCTAssertGreaterThan(bBig.radius, 0)
        let dBig = MLSharpTrajectory.depthQuantiles(scene: big, opacityThreshold: 0.01)
        XCTAssertGreaterThan(dBig.max, dBig.min)
    }

    func testMLSharpTrajectoryVariants() throws {
        let device = try requireMetalDevice()
        let scene = try makeScene(device: device, count: 256, opacity: 1.0, z: 1.0)

        XCTAssertEqual(MLSharpTrajectory.screenResolutionPxFromInput(width: 0, height: 0).width, 2)
        XCTAssertEqual(MLSharpTrajectory.screenResolutionPxFromInput(width: 5, height: 3001).height % 2, 0)

        let scaled = MLSharpTrajectory.scaleIntrinsics(
            fx: 100, fy: 100, cx: 10, cy: 10,
            srcWidth: 0, srcHeight: 0,
            dstWidth: 100, dstHeight: 100
        )
        XCTAssertEqual(scaled.fx, 100)

        for kind in MLSharpTrajectoryParams.Kind.allCases {
            for lookAt in MLSharpTrajectoryParams.LookAtMode.allCases {
                var p = MLSharpTrajectoryParams()
                p.kind = kind
                p.lookAtMode = lookAt
                p.numSteps = 4
                p.numRepeats = 1
                p.sampleCount = 128
                let res = MLSharpTrajectory.makeCameras(
                    scene: scene,
                    sourceImageWidth: 512,
                    sourceImageHeight: 512,
                    intrinsicFx: 400,
                    intrinsicFy: 400,
                    intrinsicCx: 255.5,
                    intrinsicCy: 255.5,
                    renderWidth: 256,
                    renderHeight: 256,
                    params: p
                )
                XCTAssertEqual(res.cameras.count, 4)
            }
        }
    }

    func testPLYLoaderSuccessAndErrors() throws {
        // Unreadable.
        do {
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: URL(fileURLWithPath: "/no/such/file.ply"))
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .unreadable(_) = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Invalid header: missing end_header.
        do {
            let url = try writeTempFile(Data("ply\nformat binary_little_endian 1.0\n".utf8), name: "bad_no_end_header.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .invalidHeader = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Invalid header: not UTF-8 but contains end_header marker.
        do {
            var bytes = Data([0xFF, 0xFE, 0xFF])
            bytes.append(contentsOf: Array("end_header\n".utf8))
            let url = try writeTempFile(bytes, name: "bad_utf8.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .invalidHeader = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Unsupported format.
        do {
            let url = try writeTempFile(Data("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n".utf8), name: "bad_format.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .unsupportedFormat(_) = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Invalid header: missing `format` line.
        do {
            let header = "ply\nelement vertex 1\nend_header\n"
            var data = Data(header.utf8)
            for _ in 0..<14 { appendF32LE(0, to: &data) }
            let url = try writeTempFile(data, name: "bad_missing_format.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .invalidHeader = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Missing vertex count.
        do {
            let url = try writeTempFile(Data("ply\nformat binary_little_endian 1.0\nend_header\n".utf8), name: "bad_missing_vertex.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .missingVertexCount = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Unexpected EOF (vertex data).
        do {
            let header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\nend_header\n"
            var data = Data(header.utf8)
            // Only 1 vertex worth of bytes (should be 2).
            for _ in 0..<14 { appendF32LE(0, to: &data) }
            let url = try writeTempFile(data, name: "bad_eof_vertex.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .unexpectedEOF = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Unexpected EOF (metadata).
        do {
            let v = TestPLYVertex(x: 0, y: 0, z: 1, fdc0: 0, fdc1: 0, fdc2: 0, opacityLogit: 0, s0Log: 0, s1Log: 0, s2Log: 0, q0: 1, q1: 0, q2: 0, q3: 0)
            // Header declares metadata but we omit it from the payload.
            var data = makeMLSharpPLYData(vertices: [v], metadata: TestPLYMetadata(
                extrinsic: Array(repeating: 0, count: 16),
                intrinsic: Array(repeating: 0, count: 9),
                imageWidth: 1,
                imageHeight: 1,
                frameIndex: 0,
                frameCount: 1,
                disparityP10: 0.1,
                disparityP90: 0.9,
                colorSpace: 0,
                version: (1, 0, 0)
            ))
            // Truncate right after the vertex data.
            data = data.prefix(data.count - (16 * 4 + 9 * 4 + 2 * 4 + 2 * 4 + 2 * 4 + 1 + 3))
            let url = try writeTempFile(data, name: "bad_eof_meta.ply")
            _ = try PLYLoader.loadMLSharpCompatiblePLY(url: url)
            XCTFail("Expected error")
        } catch let e as PLYLoadError {
            if case .unexpectedEOF = e {
                // ok
            } else {
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Success (no metadata): exercises sRGB->linear conversion for both branches.
        do {
            let coeff = Float((1.0 / (4.0 * Double.pi)).squareRoot())
            // Force rSRGB ~ 0.02 (<= 0.04045) and 0.5 (> 0.04045).
            let fdcLow = (0.02 - 0.5) / coeff
            let v0 = TestPLYVertex(x: 0, y: 0, z: 1, fdc0: fdcLow, fdc1: 0, fdc2: 0, opacityLogit: 0, s0Log: 0, s1Log: 0, s2Log: 0, q0: 1, q1: 0, q2: 0, q3: 0)
            let v1 = TestPLYVertex(x: 0, y: 0, z: 1, fdc0: 0, fdc1: 0, fdc2: 0, opacityLogit: 0, s0Log: 0, s1Log: 0, s2Log: 0, q0: 1, q1: 0, q2: 0, q3: 0)
            let url = try writeTempFile(makeMLSharpPLYData(vertices: [v0, v1], metadata: nil), name: "ok_nometa.ply")
            let res = try PLYLoader.loadMLSharpCompatiblePLYWithMetadata(url: url)
            XCTAssertNil(res.metadata)
            XCTAssertEqual(res.scene.count, 2)
            XCTAssertEqual(try PLYLoader.loadMLSharpCompatiblePLY(url: url).count, 2)

            let ptr = res.scene.colorsLinear.contents().bindMemory(to: Float.self, capacity: 2 * 3)
            let r0 = ptr[0]
            let r1 = ptr[3]
            XCTAssertEqual(r0, 0.02 / 12.92, accuracy: 1e-4)
            XCTAssertGreaterThan(r1, 0.2)
        }

        // Success (metadata + linear colorspace): no conversion.
        do {
            let v = TestPLYVertex(x: 0, y: 0, z: 1, fdc0: 0, fdc1: 0, fdc2: 0, opacityLogit: 0, s0Log: 0, s1Log: 0, s2Log: 0, q0: 1, q1: 0, q2: 0, q3: 0)
            let meta = TestPLYMetadata(
                extrinsic: Array(repeating: 0, count: 16),
                intrinsic: Array(repeating: 0, count: 9),
                imageWidth: 640,
                imageHeight: 480,
                frameIndex: 0,
                frameCount: 1,
                disparityP10: 0.1,
                disparityP90: 0.9,
                colorSpace: 1,
                version: (1, 2, 3)
            )
            let url = try writeTempFile(makeMLSharpPLYData(vertices: [v], metadata: meta), name: "ok_meta_linear.ply")
            let res = try PLYLoader.loadMLSharpCompatiblePLYWithMetadata(url: url)
            XCTAssertNotNil(res.metadata)
            XCTAssertEqual(res.metadata?.imageWidth, 640)
            XCTAssertEqual(res.metadata?.colorSpace, 1)

            let ptr = res.scene.colorsLinear.contents().bindMemory(to: Float.self, capacity: 3)
            XCTAssertEqual(ptr[0], 0.5, accuracy: 1e-6)
        }
    }

    func testMetalRendererCoversAllModes() throws {
        let device = try requireMetalDevice()

        // Touch the embedded metallib (covers the generated base64 source file).
        XCTAssertGreaterThan(MetalLibrary_GaussianSplat.data.count, 0)

        let scene = try makeScene(device: device, count: 256, opacity: 0.75, z: 1.0, makeSomeLowOpacity: true)
        let renderer = try GaussianSplatRenderer(device: device)
        _ = try GaussianSplatRenderer() // exercise the default device path

        let w = 96
        let h = 64
        let cam = PinholeCamera(
            viewMatrix: PinholeCamera.lookAt(eye: SIMD3<Float>(0, 0, 0), target: SIMD3<Float>(0, 0, 2)),
            fx: 200,
            fy: 200,
            cx: Float(w - 1) * 0.5,
            cy: Float(h - 1) * 0.5
        )

        // Default overload (debugView .none, toneMap .none, weighted OIT, no SSAA).
        let img0 = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h)
        XCTAssertEqual(img0.width, w)
        XCTAssertEqual(img0.height, h)

        // Weighted OIT + SSAA + alpha debug (covers debugViewIndex(.alpha), toneMapIndex(.reinhard), SSAA path).
        var o1 = GaussianSplatRenderOptions()
        o1.renderScale = 2.0
        o1.debugView = .alpha
        o1.toneMap = .reinhard
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: o1)

        // Weighted OIT + depth debug (forces depth quantiles).
        var o2 = o1
        o2.renderScale = 1.0
        o2.debugView = .depth
        o2.toneMap = .aces
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: o2)

        // Weighted OIT + disparity debug with explicit depth range (covers debugDepthRange override).
        var o3 = o1
        o3.debugView = .disparity
        o3.toneMap = .none
        o3.debugDepthRange = SIMD2<Float>(3.0, 1.0)
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: o3)

        // Weighted OIT + radius debug.
        var o4 = o1
        o4.debugView = .radius
        o4.toneMap = .none
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: o4)

        // Depth-binned alpha (covers that compositing path + SSAA).
        var o5 = GaussianSplatRenderOptions()
        o5.compositing = .depthBinnedAlpha(binCount: 8)
        o5.renderScale = 2.0
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: o5)

        // Depth-binned alpha with nothing visible (covers visibleCount==0 early return path).
        let invisible = try makeScene(device: device, count: 64, opacity: 0.0, z: 1.0)
        var o6 = o5
        o6.opacityThreshold = 0.5
        let imgBlank = try renderer.renderToCGImage(scene: invisible, camera: cam, width: w, height: h, options: o6)
        XCTAssertEqual(imgBlank.width, w)

        // Normalization paths:
        // - mode .none + scale .unitRadius (covers estimateMedianCenter switch .none).
        // - mode .recenterXY
        // - mode .recenterXYZ + unitRadius scale (covers bounds-based scaling).
        var on = GaussianSplatRenderOptions()
        on.compositing = .weightedOIT
        on.normalization = .none
        on.normalizationScale = .unitRadius
        on.normalizationSampleCount = 64
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: on)

        var on2 = on
        on2.normalization = .recenterXY
        on2.normalizationScale = .none
        _ = try renderer.renderToCGImage(scene: scene, camera: cam, width: w, height: h, options: on2)

        // estimateMedianCenter early returns:
        // - count == 0
        // - fewer than 16 samples after filtering
        let emptyScene = try makeScene(device: device, count: 0, opacity: 1.0, z: 1.0)
        _ = try renderer.renderToCGImage(scene: emptyScene, camera: cam, width: w, height: h, options: on)

        let tinyScene = try makeScene(device: device, count: 8, opacity: 1.0, z: 1.0)
        _ = try renderer.renderToCGImage(scene: tinyScene, camera: cam, width: w, height: h, options: on2)

        // Exercise the odd-count median branch.
        let oddScene = try makeScene(device: device, count: 257, opacity: 1.0, z: 1.0)
        _ = try renderer.renderToCGImage(scene: oddScene, camera: cam, width: w, height: h, options: on2)

        // No-fallback median center path by using >= 1024 samples.
        let big = try makeScene(device: device, count: 2048, opacity: 1.0, z: 1.0)
        var on3 = on
        on3.normalization = .recenterXYZ
        on3.normalizationScale = .unitRadius
        on3.normalizationSampleCount = 2048
        _ = try renderer.renderToCGImage(scene: big, camera: cam, width: w, height: h, options: on3)
    }

    func testMP4VideoWriterSmokeAndTimeouts() async throws {
        // Exercise the status-mapping helpers directly (covers throw paths deterministically).
        XCTAssertThrowsError(try MP4VideoWriter._throwIfWriterFailed(status: .failed, underlyingError: nil))
        XCTAssertThrowsError(try MP4VideoWriter._throwIfWriterFailed(status: .cancelled, underlyingError: nil))
        try MP4VideoWriter._throwIfWriterFailed(status: .writing, underlyingError: nil)

        try MP4VideoWriter._validateFinishStatus(status: .completed, underlyingError: nil)
        XCTAssertThrowsError(try MP4VideoWriter._validateFinishStatus(status: .failed, underlyingError: nil))
        XCTAssertThrowsError(try MP4VideoWriter._validateFinishStatus(status: .cancelled, underlyingError: nil))

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("GaussianSplatMetalRendererVideo_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let urlOK = dir.appendingPathComponent("ok.mp4")
        let img = try makeSolidImage(width: 32, height: 32, rgba: (255, 0, 0, 255))

        // Happy path.
        do {
            let w = try MP4VideoWriter(url: urlOK, width: 32, height: 32, fps: 10)
            try w.append(img)
            try await w.finish(timeoutSeconds: 10)
            XCTAssertTrue(FileManager.default.fileExists(atPath: urlOK.path))

            // After finish, appending should time out waiting for readiness.
            XCTAssertThrowsError(try w.append(img, timeoutSeconds: 0.02)) { err in
                guard let e = err as? VideoExporterError else {
                    XCTFail("Unexpected error: \(err)")
                    return
                }
                if case .appendTimeout = e {
                    return
                }
                XCTFail("Unexpected error: \(e)")
            }
        }

        // Timeout path + catch path (finish called again after cancel).
        do {
            let url = dir.appendingPathComponent("timeout.mp4")
            let w = try MP4VideoWriter(url: url, width: 32, height: 32, fps: 10)
            try w.append(img)

            await XCTAssertThrowsErrorAsync(try await w.finish(timeoutSeconds: 0)) { err in
                guard let e = err as? VideoExporterError else {
                    XCTFail("Unexpected error: \(err)")
                    return
                }
                if case .finishTimeout = e {
                    return
                }
                XCTFail("Unexpected error: \(e)")
            }

            await XCTAssertThrowsErrorAsync(try await w.finish(timeoutSeconds: 1)) { err in
                guard let e = err as? VideoExporterError else {
                    XCTFail("Unexpected error: \(err)")
                    return
                }
                if case .finishFailed = e {
                    return
                }
                XCTFail("Unexpected error: \(e)")
            }
        }
    }
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line,
    _ errorHandler: (Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail("Expected error. \(message())", file: file, line: line)
    } catch {
        errorHandler(error)
    }
}
