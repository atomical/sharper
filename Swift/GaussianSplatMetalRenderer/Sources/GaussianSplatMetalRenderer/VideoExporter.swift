@preconcurrency import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation

public enum VideoExporterError: Error {
    case writerCreateFailed
    case cannotAddInput
    case startFailed
    case pixelBufferCreateFailed
    case appendFailed
    case appendTimeout
    case finishFailed
    case finishTimeout
    case writerFailed(underlying: Error?)
}

public final class MP4VideoWriter: @unchecked Sendable {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let fps: Int32
    private var frameIndex: Int64 = 0

    public init(url: URL, width: Int, height: Int, fps: Int = 30) throws {
        self.fps = Int32(fps)

        try? FileManager.default.removeItem(at: url)
        self.writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 8_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]
        self.input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        self.input.expectsMediaDataInRealTime = false

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        self.adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)

        guard writer.canAdd(input) else { throw VideoExporterError.cannotAddInput }
        writer.add(input)

        guard writer.startWriting() else { throw VideoExporterError.startFailed }
        writer.startSession(atSourceTime: .zero)
    }

    private func throwIfWriterFailed() throws {
        switch writer.status {
        case .failed:
            throw VideoExporterError.writerFailed(underlying: writer.error)
        case .cancelled:
            throw VideoExporterError.finishFailed
        default:
            return
        }
    }

    public func append(_ image: CGImage, timeoutSeconds: TimeInterval = 2.0) throws {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while !input.isReadyForMoreMediaData {
            try throwIfWriterFailed()
            if Date() > deadline {
                throw VideoExporterError.appendTimeout
            }
            Thread.sleep(forTimeInterval: 0.001)
        }

        guard let pool = adaptor.pixelBufferPool else {
            throw VideoExporterError.pixelBufferCreateFailed
        }
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        guard status == kCVReturnSuccess, let pb = pixelBuffer else {
            throw VideoExporterError.pixelBufferCreateFailed
        }

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            throw VideoExporterError.pixelBufferCreateFailed
        }

        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))

        let time = CMTime(value: frameIndex, timescale: fps)
        if !adaptor.append(pb, withPresentationTime: time) {
            throw VideoExporterError.appendFailed
        }
        frameIndex += 1
    }

    public func finish(timeoutSeconds: TimeInterval = 30.0) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                self.input.markAsFinished()
                let sem = DispatchSemaphore(value: 0)
                self.writer.finishWriting { sem.signal() }
                if sem.wait(timeout: .now() + timeoutSeconds) == .timedOut {
                    self.writer.cancelWriting()
                    cont.resume(throwing: VideoExporterError.finishTimeout)
                    return
                }
                if self.writer.status != .completed {
                    if self.writer.status == .failed {
                        cont.resume(throwing: VideoExporterError.writerFailed(underlying: self.writer.error))
                        return
                    }
                    cont.resume(throwing: VideoExporterError.finishFailed)
                    return
                }
                cont.resume()
            }
        }
    }
}
