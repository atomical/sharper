import CoreML
import Foundation
import Metal

public enum SharpPLYWriterError: Error {
    case invalidBufferLayout
    case fileCreateFailed(URL)
}

public struct SharpPLYWriter {
    public static func write(
        prediction: SharpPrediction,
        to url: URL
    ) throws {
        let count = prediction.postprocessed.count

        guard prediction.postprocessed.mean.length >= count * 3 * MemoryLayout<Float>.size,
              prediction.postprocessed.singularValues.length >= count * 3 * MemoryLayout<Float>.size,
              prediction.postprocessed.quaternions.length >= count * 4 * MemoryLayout<Float>.size
        else {
            throw SharpPLYWriterError.invalidBufferLayout
        }

        let colors = prediction.raw.colorsLinearPre
        let opacities = prediction.raw.opacitiesPre

        let colorsPtr = colors.dataPointer.bindMemory(to: Float.self, capacity: count * 3)
        let opacitiesPtr = opacities.dataPointer.bindMemory(to: Float.self, capacity: count)

        let meanPtr = prediction.postprocessed.mean.contents().bindMemory(to: Float.self, capacity: count * 3)
        let scalePtr = prediction.postprocessed.singularValues.contents().bindMemory(to: Float.self, capacity: count * 3)
        let quatPtr = prediction.postprocessed.quaternions.contents().bindMemory(to: Float.self, capacity: count * 4)

        let header = plyHeader(vertexCount: count)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw SharpPLYWriterError.fileCreateFailed(url)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        if let headerData = header.data(using: .utf8) {
            try handle.write(contentsOf: headerData)
        }

        // Compute disparity quantiles for metadata.
        let disparityQuantiles = disparityQuantiles10_90(meanPtr: meanPtr, count: count)

        // Write vertex payload in chunks.
        let floatsPerVertex = 14
        let wordsPerVertex = floatsPerVertex
        let chunkVertices = 8_192

        let coeff = Float((1.0 / (4.0 * Double.pi)).squareRoot())

        var idx = 0
        while idx < count {
            let end = min(count, idx + chunkVertices)
            let n = end - idx
            var data = Data(count: n * wordsPerVertex * MemoryLayout<UInt32>.size)
            data.withUnsafeMutableBytes { raw in
                let words = raw.bindMemory(to: UInt32.self)
                var w = 0

                for i in idx..<end {
                    let mi = i * 3
                    let qi = i * 4
                    let si = i * 3
                    let ci = i * 3

                    let x = meanPtr[mi + 0]
                    let y = meanPtr[mi + 1]
                    let z = meanPtr[mi + 2]

                    let s0 = scalePtr[si + 0]
                    let s1 = scalePtr[si + 1]
                    let s2 = scalePtr[si + 2]

                    let q0 = quatPtr[qi + 0]
                    let q1 = quatPtr[qi + 1]
                    let q2 = quatPtr[qi + 2]
                    let q3 = quatPtr[qi + 3]

                    let rLin = colorsPtr[ci + 0]
                    let gLin = colorsPtr[ci + 1]
                    let bLin = colorsPtr[ci + 2]

                    let rSRGB = linearToSRGB(rLin)
                    let gSRGB = linearToSRGB(gLin)
                    let bSRGB = linearToSRGB(bLin)

                    let fdc0 = (rSRGB - 0.5) / coeff
                    let fdc1 = (gSRGB - 0.5) / coeff
                    let fdc2 = (bSRGB - 0.5) / coeff

                    let opacity = opacitiesPtr[i]
                    let opacityLogit = logit(opacity)

                    let scaleLog0 = log(max(s0, 1e-20))
                    let scaleLog1 = log(max(s1, 1e-20))
                    let scaleLog2 = log(max(s2, 1e-20))

                    // x,y,z
                    words[w] = x.bitPattern.littleEndian; w += 1
                    words[w] = y.bitPattern.littleEndian; w += 1
                    words[w] = z.bitPattern.littleEndian; w += 1
                    // f_dc_0..2
                    words[w] = fdc0.bitPattern.littleEndian; w += 1
                    words[w] = fdc1.bitPattern.littleEndian; w += 1
                    words[w] = fdc2.bitPattern.littleEndian; w += 1
                    // opacity (logit)
                    words[w] = opacityLogit.bitPattern.littleEndian; w += 1
                    // scale logits
                    words[w] = scaleLog0.bitPattern.littleEndian; w += 1
                    words[w] = scaleLog1.bitPattern.littleEndian; w += 1
                    words[w] = scaleLog2.bitPattern.littleEndian; w += 1
                    // rot (wxyz)
                    words[w] = q0.bitPattern.littleEndian; w += 1
                    words[w] = q1.bitPattern.littleEndian; w += 1
                    words[w] = q2.bitPattern.littleEndian; w += 1
                    words[w] = q3.bitPattern.littleEndian; w += 1
                }
            }
            try handle.write(contentsOf: data)
            idx = end
        }

        // extrinsic element (identity 4x4)
        try writeFloat32Array(handle: handle, values: [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ])

        // intrinsic element (3x3 flattened)
        let f = prediction.metadata.focalLengthPx
        let w = Float(prediction.metadata.imageWidth)
        let h = Float(prediction.metadata.imageHeight)
        try writeFloat32Array(handle: handle, values: [
            f, 0, w * 0.5,
            0, f, h * 0.5,
            0, 0, 1,
        ])

        // image_size element (uint32[2] as separate element entries)
        try writeUInt32Array(handle: handle, values: [UInt32(prediction.metadata.imageWidth), UInt32(prediction.metadata.imageHeight)])

        // frame element (int32[2])
        try writeInt32Array(handle: handle, values: [1, Int32(count)])

        // disparity element (float32[2])
        try writeFloat32Array(handle: handle, values: disparityQuantiles)

        // color_space element (uint8[1]) — 0 for sRGB
        try handle.write(contentsOf: Data([0]))

        // version element (uint8[3])
        try handle.write(contentsOf: Data([1, 5, 0]))
    }

    private static func plyHeader(vertexCount: Int) -> String {
        """
        ply
        format binary_little_endian 1.0
        element vertex \(vertexCount)
        property float x
        property float y
        property float z
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        element extrinsic 16
        property float extrinsic
        element intrinsic 9
        property float intrinsic
        element image_size 2
        property uint image_size
        element frame 2
        property int frame
        element disparity 2
        property float disparity
        element color_space 1
        property uchar color_space
        element version 3
        property uchar version
        end_header
        """ + "\n"
    }

    private static func linearToSRGB(_ x: Float) -> Float {
        let thr: Float = 0.0031308
        if x <= thr { return x * 12.92 }
        return 1.055 * pow(max(x, thr), 1.0 / 2.4) - 0.055
    }

    static func logit(_ p: Float, eps: Float = 1e-6) -> Float {
        let pc = min(max(p, eps), 1.0 - eps)
        return log(pc / (1.0 - pc))
    }

    private static func disparityQuantiles10_90(meanPtr: UnsafePointer<Float>, count: Int) -> [Float] {
        var disparity = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let z = meanPtr[i * 3 + 2]
            disparity[i] = 1.0 / max(z, 1e-8)
        }
        disparity.sort()
        let i10 = Int(Double(count - 1) * 0.1)
        let i90 = Int(Double(count - 1) * 0.9)
        return [disparity[i10], disparity[i90]]
    }

    private static func writeFloat32Array(handle: FileHandle, values: [Float]) throws {
        var data = Data(count: values.count * 4)
        data.withUnsafeMutableBytes { raw in
            let words = raw.bindMemory(to: UInt32.self)
            for (i, v) in values.enumerated() {
                words[i] = v.bitPattern.littleEndian
            }
        }
        try handle.write(contentsOf: data)
    }

    private static func writeUInt32Array(handle: FileHandle, values: [UInt32]) throws {
        var data = Data(count: values.count * 4)
        data.withUnsafeMutableBytes { raw in
            let words = raw.bindMemory(to: UInt32.self)
            for (i, v) in values.enumerated() {
                words[i] = v.littleEndian
            }
        }
        try handle.write(contentsOf: data)
    }

    private static func writeInt32Array(handle: FileHandle, values: [Int32]) throws {
        var data = Data(count: values.count * 4)
        data.withUnsafeMutableBytes { raw in
            let words = raw.bindMemory(to: Int32.self)
            for (i, v) in values.enumerated() {
                words[i] = Int32(littleEndian: v)
            }
        }
        try handle.write(contentsOf: data)
    }
}
