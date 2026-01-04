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
    public static func loadMLSharpCompatiblePLY(url: URL, device: MTLDevice? = nil) throws -> GaussianScene {
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
        }

        guard formatOK else { throw PLYLoadError.invalidHeader }
        guard let n = vertexCount else { throw PLYLoadError.missingVertexCount }

        let vertexStrideBytes = 14 * 4
        let vertexDataOffset = headerRange.upperBound
        let needed = vertexDataOffset + n * vertexStrideBytes
        guard data.count >= needed else { throw PLYLoadError.unexpectedEOF }

        var means = [Float](repeating: 0, count: n * 3)
        var scales = [Float](repeating: 0, count: n * 3)
        var colorsLinear = [Float](repeating: 0, count: n * 3)
        var opacities = [Float](repeating: 0, count: n)

        let coeff = Float((1.0 / (4.0 * Double.pi)).squareRoot())

        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: vertexDataOffset)
            for i in 0..<n {
                let off = i * vertexStrideBytes
                let ptr = base.advanced(by: off)

                // Read 14 float32s.
                let f = ptr.bindMemory(to: UInt32.self, capacity: 14)
                func readF(_ idx: Int) -> Float {
                    Float(bitPattern: UInt32(littleEndian: f[idx]))
                }

                let x = readF(0)
                let y = readF(1)
                let z = readF(2)

                let fdc0 = readF(3)
                let fdc1 = readF(4)
                let fdc2 = readF(5)

                let opacityLogit = readF(6)

                let s0 = exp(readF(7))
                let s1 = exp(readF(8))
                let s2 = exp(readF(9))

                // rot_0..rot_3 at 10..13 (ignored for now)

                means[i * 3 + 0] = x
                means[i * 3 + 1] = y
                means[i * 3 + 2] = z

                scales[i * 3 + 0] = s0
                scales[i * 3 + 1] = s1
                scales[i * 3 + 2] = s2

                let rSRGB = fdc0 * coeff + 0.5
                let gSRGB = fdc1 * coeff + 0.5
                let bSRGB = fdc2 * coeff + 0.5

                colorsLinear[i * 3 + 0] = sRGBToLinear(rSRGB)
                colorsLinear[i * 3 + 1] = sRGBToLinear(gSRGB)
                colorsLinear[i * 3 + 2] = sRGBToLinear(bSRGB)

                opacities[i] = sigmoid(opacityLogit)
            }
        }

        guard let meansBuf = device.makeBuffer(bytes: means, length: means.count * 4, options: .storageModeShared),
              let scalesBuf = device.makeBuffer(bytes: scales, length: scales.count * 4, options: .storageModeShared),
              let colorsBuf = device.makeBuffer(bytes: colorsLinear, length: colorsLinear.count * 4, options: .storageModeShared),
              let opaBuf = device.makeBuffer(bytes: opacities, length: opacities.count * 4, options: .storageModeShared)
        else {
            throw PLYLoadError.metalBufferCreateFailed
        }

        return GaussianScene(count: n, means: meansBuf, scales: scalesBuf, colorsLinear: colorsBuf, opacities: opaBuf)
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

