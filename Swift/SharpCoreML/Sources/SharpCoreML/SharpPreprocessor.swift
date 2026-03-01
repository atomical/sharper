import CoreML
import CoreImage
import CoreGraphics
import Foundation
import ImageIO

public enum SharpPreprocessError: Error {
    case imageDecodeFailed(URL)
    case rgbContextCreateFailed
    case multiArrayCreateFailed
    case orientationTransformFailed
}

public struct SharpPreprocessor {
    public static let internalResolution: Int = 1536
    private static let orientationContext = CIContext(options: [.cacheIntermediates: false])

    public static func loadCGImage(from url: URL, autoRotate: Bool = true) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw SharpPreprocessError.imageDecodeFailed(url)
        }
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw SharpPreprocessError.imageDecodeFailed(url)
        }

        guard autoRotate else { return cgImage }

        let orientationRaw: UInt32
        if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let ori = (props[kCGImagePropertyOrientation] as? NSNumber)
        {
            orientationRaw = ori.uint32Value
        } else {
            orientationRaw = 1
        }

        guard orientationRaw != 1 else { return cgImage }
        guard (1...8).contains(Int(orientationRaw)) else { return cgImage }
        return try oriented(cgImage: cgImage, exifOrientation: orientationRaw)
    }

    public static func loadMetadata(from url: URL, imageWidth: Int, imageHeight: Int) -> SharpInputMetadata {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        else {
            return SharpInputMetadata(
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                focalLengthPx: focalLengthPx(focalLengthMM: 30.0, width: imageWidth, height: imageHeight)
            )
        }

        let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]

        let f35 = (exif["FocalLengthIn35mmFilm" as CFString] as? NSNumber)
            ?? (exif["FocalLenIn35mmFilm" as CFString] as? NSNumber)
        let f = (exif[kCGImagePropertyExifFocalLength] as? NSNumber)

        var focalMM: Double
        if let f35 {
            focalMM = f35.doubleValue
        } else if let f {
            focalMM = f.doubleValue
        } else {
            focalMM = 30.0
        }

        if focalMM < 10.0 {
            focalMM *= 8.4
        }

        let fPx = focalLengthPx(focalLengthMM: focalMM, width: imageWidth, height: imageHeight)
        return SharpInputMetadata(imageWidth: imageWidth, imageHeight: imageHeight, focalLengthPx: fPx)
    }

    public static func focalLengthPx(focalLengthMM: Double, width: Int, height: Int) -> Float {
        let w = Double(width)
        let h = Double(height)
        let diag = (w * w + h * h).squareRoot()
        let denom = (36.0 * 36.0 + 24.0 * 24.0).squareRoot()
        return Float(focalLengthMM * diag / denom)
    }

    public static func makeInputs(
        cgImage: CGImage,
        metadata: SharpInputMetadata,
        internalResolution: Int = SharpPreprocessor.internalResolution
    ) throws -> (image: MLMultiArray, disparityFactor: MLMultiArray) {
        let srcW = cgImage.width
        let srcH = cgImage.height

        let bytesPerRow = srcW * 4
        var rgba = Data(count: bytesPerRow * srcH)
        let ok = rgba.withUnsafeMutableBytes { rawBuf -> Bool in
            guard let base = rawBuf.baseAddress else { return false }
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
            guard let ctx = CGContext(
                data: base,
                width: srcW,
                height: srcH,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else { return false }
            ctx.interpolationQuality = .none
            ctx.setAllowsAntialiasing(false)
            ctx.setShouldAntialias(false)
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: srcW, height: srcH))
            return true
        }
        guard ok else { throw SharpPreprocessError.rgbContextCreateFailed }

        let outShape: [NSNumber] = [1, 3, NSNumber(value: internalResolution), NSNumber(value: internalResolution)]
        guard let image = try? MLMultiArray(shape: outShape, dataType: .float32) else {
            throw SharpPreprocessError.multiArrayCreateFailed
        }
        guard let disparity = try? MLMultiArray(shape: [1], dataType: .float32) else {
            throw SharpPreprocessError.multiArrayCreateFailed
        }

        disparity[0] = NSNumber(value: metadata.disparityFactor)

        let dstW = internalResolution
        let dstH = internalResolution

        let strideB = image.strides.map { $0.intValue }
        let strideC = strideB[1]
        let strideY = strideB[2]
        let strideX = strideB[3]

        let scaleX = (dstW > 1) ? Float(srcW - 1) / Float(dstW - 1) : 0.0
        let scaleY = (dstH > 1) ? Float(srcH - 1) / Float(dstH - 1) : 0.0

        var x0 = [Int](repeating: 0, count: dstW)
        var x1 = [Int](repeating: 0, count: dstW)
        var wx = [Float](repeating: 0, count: dstW)
        for xo in 0..<dstW {
            let gx = Float(xo) * scaleX
            let ix0 = Int(gx.rounded(.down))
            x0[xo] = max(0, min(srcW - 1, ix0))
            x1[xo] = max(0, min(srcW - 1, ix0 + 1))
            wx[xo] = gx - Float(ix0)
        }

        var y0 = [Int](repeating: 0, count: dstH)
        var y1 = [Int](repeating: 0, count: dstH)
        var wy = [Float](repeating: 0, count: dstH)
        for yo in 0..<dstH {
            let gy = Float(yo) * scaleY
            let iy0 = Int(gy.rounded(.down))
            y0[yo] = max(0, min(srcH - 1, iy0))
            y1[yo] = max(0, min(srcH - 1, iy0 + 1))
            wy[yo] = gy - Float(iy0)
        }

        let inv255: Float = 1.0 / 255.0

        rgba.withUnsafeBytes { rawBuf in
            let src = rawBuf.bindMemory(to: UInt8.self)
            let out = image.dataPointer.bindMemory(to: Float.self, capacity: 1 * 3 * dstH * dstW)

            for yo in 0..<dstH {
                let iy0 = y0[yo]
                let iy1 = y1[yo]
                let ty = wy[yo]
                let ty0 = 1.0 - ty
                let row0 = iy0 * bytesPerRow
                let row1 = iy1 * bytesPerRow

                for xo in 0..<dstW {
                    let ix0 = x0[xo]
                    let ix1 = x1[xo]
                    let tx = wx[xo]
                    let tx0 = 1.0 - tx

                    let p00 = row0 + ix0 * 4
                    let p01 = row0 + ix1 * 4
                    let p10 = row1 + ix0 * 4
                    let p11 = row1 + ix1 * 4

                    // Source is BGRA (byteOrder32Little + premultipliedFirst).
                    let b00 = Float(src[p00 + 0]) * inv255
                    let g00 = Float(src[p00 + 1]) * inv255
                    let r00 = Float(src[p00 + 2]) * inv255

                    let b01 = Float(src[p01 + 0]) * inv255
                    let g01 = Float(src[p01 + 1]) * inv255
                    let r01 = Float(src[p01 + 2]) * inv255

                    let b10 = Float(src[p10 + 0]) * inv255
                    let g10 = Float(src[p10 + 1]) * inv255
                    let r10 = Float(src[p10 + 2]) * inv255

                    let b11 = Float(src[p11 + 0]) * inv255
                    let g11 = Float(src[p11 + 1]) * inv255
                    let r11 = Float(src[p11 + 2]) * inv255

                    let r0 = tx0 * r00 + tx * r01
                    let g0 = tx0 * g00 + tx * g01
                    let b0 = tx0 * b00 + tx * b01

                    let r1 = tx0 * r10 + tx * r11
                    let g1 = tx0 * g10 + tx * g11
                    let b1 = tx0 * b10 + tx * b11

                    let r = ty0 * r0 + ty * r1
                    let g = ty0 * g0 + ty * g1
                    let b = ty0 * b0 + ty * b1

                    let base = yo * strideY + xo * strideX
                    out[0 * strideC + base] = r
                    out[1 * strideC + base] = g
                    out[2 * strideC + base] = b
                }
            }
        }

        return (image: image, disparityFactor: disparity)
    }

    private static func oriented(cgImage: CGImage, exifOrientation: UInt32) throws -> CGImage {
        let ciImage = CIImage(cgImage: cgImage).oriented(forExifOrientation: Int32(exifOrientation))
        let extent = ciImage.extent.integral
        guard let out = orientationContext.createCGImage(ciImage, from: extent) else {
            throw SharpPreprocessError.orientationTransformFailed
        }
        return out
    }
}
