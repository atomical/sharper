import Foundation
import Metal

public enum PLYLoadError: Error {
    case unreadable(URL)
    case invalidHeader
    case unsupportedFormat(String)
    case missingVertexCount
    case unexpectedEOF
    case metalUnavailable
    case metalBufferCreateFailed
}

public struct PLYLoader {
    public struct MLSharpPLYMetadata {
        public var extrinsic: [Float] // 16
        public var intrinsic: [Float] // 9
        public var imageWidth: UInt32
        public var imageHeight: UInt32
        public var frameIndex: Int32
        public var frameCount: Int32
        public var disparityP10: Float
        public var disparityP90: Float
        public var colorSpace: UInt8
        public var versionMajor: UInt8
        public var versionMinor: UInt8
        public var versionPatch: UInt8
    }

    public static func loadMLSharpCompatiblePLY(url: URL, device: MTLDevice? = nil) throws -> GaussianScene {
        try _loadMLSharpCompatiblePLY(url: url, device: device).scene
    }

    public static func loadMLSharpCompatiblePLYWithMetadata(
        url: URL,
        device: MTLDevice? = nil
    ) throws -> (scene: GaussianScene, metadata: MLSharpPLYMetadata?) {
        try _loadMLSharpCompatiblePLY(url: url, device: device)
    }

    private static func _loadMLSharpCompatiblePLY(
        url: URL,
        device: MTLDevice? = nil
    ) throws -> (scene: GaussianScene, metadata: MLSharpPLYMetadata?) {
        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else { throw PLYLoadError.metalUnavailable }

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw PLYLoadError.unreadable(url)
        }

        guard let headerRange = data.range(of: Data("end_header\n".utf8)) else {
            throw PLYLoadError.invalidHeader
        }
        let headerData = data.subdata(in: 0..<headerRange.upperBound)
        guard let headerStr = String(data: headerData, encoding: .utf8) else {
            throw PLYLoadError.invalidHeader
        }

        var vertexCount: Int?
        var formatOK = false
        var extrinsicCount: Int?
        var intrinsicCount: Int?
        var imageSizeCount: Int?
        var frameCount: Int?
        var disparityCount: Int?
        var colorSpaceCount: Int?
        var versionCount: Int?

        for line in headerStr.split(separator: "\n") {
            if line.hasPrefix("format ") {
                formatOK = line.contains("binary_little_endian")
                if !formatOK {
                    throw PLYLoadError.unsupportedFormat(String(line))
                }
            }
            if line.hasPrefix("element vertex ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) {
                    vertexCount = n
                }
            }
            if line.hasPrefix("element extrinsic ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { extrinsicCount = n }
            }
            if line.hasPrefix("element intrinsic ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { intrinsicCount = n }
            }
            if line.hasPrefix("element image_size ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { imageSizeCount = n }
            }
            if line.hasPrefix("element frame ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { frameCount = n }
            }
            if line.hasPrefix("element disparity ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { disparityCount = n }
            }
            if line.hasPrefix("element color_space ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { colorSpaceCount = n }
            }
            if line.hasPrefix("element version ") {
                let parts = line.split(separator: " ")
                if parts.count == 3, let n = Int(parts[2]) { versionCount = n }
            }
        }

        guard formatOK else { throw PLYLoadError.invalidHeader }
        guard let n = vertexCount else { throw PLYLoadError.missingVertexCount }

        let vertexStrideBytes = 14 * 4
        let vertexDataOffset = headerRange.upperBound
        let vertexDataEnd = vertexDataOffset + n * vertexStrideBytes
        guard data.count >= vertexDataEnd else { throw PLYLoadError.unexpectedEOF }

        var means = [Float](repeating: 0, count: n * 3)
        var quaternions = [Float](repeating: 0, count: n * 4)
        var scales = [Float](repeating: 0, count: n * 3)
        // Stored as (0..1) in the PLY's declared colorspace; converted to linearRGB later if needed.
        var colors01 = [Float](repeating: 0, count: n * 3)
        var opacities = [Float](repeating: 0, count: n)

        let coeff = Float((1.0 / (4.0 * Double.pi)).squareRoot())

        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!

            @inline(__always)
            func readU32LE(_ byteOffset: Int) -> UInt32 {
                let v = base.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)
                return UInt32(littleEndian: v)
            }

            @inline(__always)
            func readF32(_ byteOffset: Int) -> Float {
                Float(bitPattern: readU32LE(byteOffset))
            }

            for i in 0..<n {
                let off = vertexDataOffset + i * vertexStrideBytes

                // 14 float32s per vertex, tightly packed after the ASCII header (which is not guaranteed 4-byte aligned).
                let x = readF32(off + 0 * 4)
                let y = readF32(off + 1 * 4)
                let z = readF32(off + 2 * 4)

                let fdc0 = readF32(off + 3 * 4)
                let fdc1 = readF32(off + 4 * 4)
                let fdc2 = readF32(off + 5 * 4)

                let opacityLogit = readF32(off + 6 * 4)

                let s0 = exp(readF32(off + 7 * 4))
                let s1 = exp(readF32(off + 8 * 4))
                let s2 = exp(readF32(off + 9 * 4))

                let q0 = readF32(off + 10 * 4)
                let q1 = readF32(off + 11 * 4)
                let q2 = readF32(off + 12 * 4)
                let q3 = readF32(off + 13 * 4)

                means[i * 3 + 0] = x
                means[i * 3 + 1] = y
                means[i * 3 + 2] = z

                quaternions[i * 4 + 0] = q0
                quaternions[i * 4 + 1] = q1
                quaternions[i * 4 + 2] = q2
                quaternions[i * 4 + 3] = q3

                scales[i * 3 + 0] = s0
                scales[i * 3 + 1] = s1
                scales[i * 3 + 2] = s2

                let rSRGB = fdc0 * coeff + 0.5
                let gSRGB = fdc1 * coeff + 0.5
                let bSRGB = fdc2 * coeff + 0.5

                colors01[i * 3 + 0] = rSRGB
                colors01[i * 3 + 1] = gSRGB
                colors01[i * 3 + 2] = bSRGB

                opacities[i] = sigmoid(opacityLogit)
            }
        }

        var metadata: MLSharpPLYMetadata? = nil
        if let extrinsicCount,
           let intrinsicCount,
           let imageSizeCount,
           let frameCount,
           let disparityCount,
           let colorSpaceCount,
           let versionCount,
           extrinsicCount > 0,
           intrinsicCount > 0,
           imageSizeCount > 0,
           frameCount > 0,
           disparityCount > 0,
           colorSpaceCount > 0,
           versionCount > 0
        {
            let metaBytes =
                extrinsicCount * 4 +
                intrinsicCount * 4 +
                imageSizeCount * 4 +
                frameCount * 4 +
                disparityCount * 4 +
                colorSpaceCount +
                versionCount
            let metaEnd = vertexDataEnd + metaBytes
            guard data.count >= metaEnd else { throw PLYLoadError.unexpectedEOF }

            var extrinsic = [Float](repeating: 0, count: extrinsicCount)
            var intrinsic = [Float](repeating: 0, count: intrinsicCount)
            var imageSize = [UInt32](repeating: 0, count: imageSizeCount)
            var frame = [Int32](repeating: 0, count: frameCount)
            var disparity = [Float](repeating: 0, count: disparityCount)
            var colorSpace: UInt8 = 0
            var version = [UInt8](repeating: 0, count: versionCount)

            data.withUnsafeBytes { raw in
                let base = raw.baseAddress!

                @inline(__always)
                func readU32LE(_ byteOffset: Int) -> UInt32 {
                    let v = base.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)
                    return UInt32(littleEndian: v)
                }

                @inline(__always)
                func readI32LE(_ byteOffset: Int) -> Int32 {
                    Int32(bitPattern: readU32LE(byteOffset))
                }

                @inline(__always)
                func readF32(_ byteOffset: Int) -> Float {
                    Float(bitPattern: readU32LE(byteOffset))
                }

                var off = vertexDataEnd
                for i in 0..<extrinsicCount {
                    extrinsic[i] = readF32(off + i * 4)
                }
                off += extrinsicCount * 4

                for i in 0..<intrinsicCount {
                    intrinsic[i] = readF32(off + i * 4)
                }
                off += intrinsicCount * 4

                for i in 0..<imageSizeCount {
                    imageSize[i] = readU32LE(off + i * 4)
                }
                off += imageSizeCount * 4

                for i in 0..<frameCount {
                    frame[i] = readI32LE(off + i * 4)
                }
                off += frameCount * 4

                for i in 0..<disparityCount {
                    disparity[i] = readF32(off + i * 4)
                }
                off += disparityCount * 4

                colorSpace = base.load(fromByteOffset: off, as: UInt8.self)
                off += colorSpaceCount

                for i in 0..<versionCount {
                    version[i] = base.load(fromByteOffset: off + i, as: UInt8.self)
                }
            }

            // Match ml-sharp conventions (3x3 intrinsic, 4x4 extrinsic, image_size=(w,h)).
            if extrinsic.count >= 16, intrinsic.count >= 9, imageSize.count >= 2, frame.count >= 2, disparity.count >= 2, version.count >= 3 {
                metadata = MLSharpPLYMetadata(
                    extrinsic: Array(extrinsic.prefix(16)),
                    intrinsic: Array(intrinsic.prefix(9)),
                    imageWidth: imageSize[0],
                    imageHeight: imageSize[1],
                    frameIndex: frame[0],
                    frameCount: frame[1],
                    disparityP10: disparity[0],
                    disparityP90: disparity[1],
                    colorSpace: colorSpace,
                    versionMajor: version[0],
                    versionMinor: version[1],
                    versionPatch: version[2]
                )
            }
        }

        // Respect the PLY's declared colorspace (ml-sharp exports sRGB for compatibility).
        let isLinear = (metadata?.colorSpace == 1)
        if !isLinear {
            for i in 0..<(n * 3) {
                colors01[i] = sRGBToLinear(colors01[i])
            }
        }

        guard let meansBuf = device.makeBuffer(bytes: means, length: means.count * 4, options: .storageModeShared),
              let quatBuf = device.makeBuffer(bytes: quaternions, length: quaternions.count * 4, options: .storageModeShared),
              let scalesBuf = device.makeBuffer(bytes: scales, length: scales.count * 4, options: .storageModeShared),
              let colorsBuf = device.makeBuffer(bytes: colors01, length: colors01.count * 4, options: .storageModeShared),
              let opaBuf = device.makeBuffer(bytes: opacities, length: opacities.count * 4, options: .storageModeShared)
        else {
            throw PLYLoadError.metalBufferCreateFailed
        }

        return (GaussianScene(count: n, means: meansBuf, quaternions: quatBuf, scales: scalesBuf, colorsLinear: colorsBuf, opacities: opaBuf), metadata)
    }

    private static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    private static func sRGBToLinear(_ x: Float) -> Float {
        let thr: Float = 0.04045
        if x <= thr { return x / 12.92 }
        return pow((x + 0.055) / 1.055, 2.4)
    }
}
