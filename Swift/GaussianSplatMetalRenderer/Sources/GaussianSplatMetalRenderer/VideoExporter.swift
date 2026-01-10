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
    private let finishGroup = DispatchGroup()
    private let finishLock = NSLock()
    private var finishStarted: Bool = false
    private var finishCompleted: Bool = false

    internal static func _throwIfWriterFailed(status: AVAssetWriter.Status, underlyingError: Error?) throws {
        switch status {
        case .failed:
            throw VideoExporterError.writerFailed(underlying: underlyingError)
        case .cancelled:
            throw VideoExporterError.finishFailed
        default:
            return
        }
    }

    internal static func _validateFinishStatus(status: AVAssetWriter.Status, underlyingError: Error?) throws {
        if status == .completed { return }
        if status == .failed {
            throw VideoExporterError.writerFailed(underlying: underlyingError)
        }
        throw VideoExporterError.finishFailed
    }

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

        try require(writer.canAdd(input), VideoExporterError.cannotAddInput)
        writer.add(input)

        try require(writer.startWriting(), VideoExporterError.startFailed)
        writer.startSession(atSourceTime: .zero)
    }

    private func throwIfWriterFailed() throws {
        try Self._throwIfWriterFailed(status: writer.status, underlyingError: writer.error)
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

        let pool = try adaptor.pixelBufferPool.orThrow(VideoExporterError.pixelBufferCreateFailed)
        var pixelBuffer: CVPixelBuffer?
        _ = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        let pb = try pixelBuffer.orThrow(VideoExporterError.pixelBufferCreateFailed)

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }

        let ctx = try CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ).orThrow(VideoExporterError.pixelBufferCreateFailed)

        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))

        let time = CMTime(value: frameIndex, timescale: fps)
        try require(adaptor.append(pb, withPresentationTime: time), VideoExporterError.appendFailed)
        frameIndex += 1
    }

    public func finish(timeoutSeconds: TimeInterval = 30.0) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                self.finishLock.lock()
                let shouldStart = !self.finishStarted
                if shouldStart {
                    self.finishStarted = true
                    self.input.markAsFinished()
                    self.finishGroup.enter()
                    self.writer.finishWriting {
                        self.finishLock.lock()
                        if !self.finishCompleted {
                            self.finishCompleted = true
                            self.finishGroup.leave()
                        }
                        self.finishLock.unlock()
                    }
                }
                self.finishLock.unlock()

                if self.finishGroup.wait(timeout: .now() + timeoutSeconds) == .timedOut {
                    self.writer.cancelWriting()
                    self.finishLock.lock()
                    if !self.finishCompleted {
                        self.finishCompleted = true
                        self.finishGroup.leave()
                    }
                    self.finishLock.unlock()
                    cont.resume(throwing: VideoExporterError.finishTimeout)
                    return
                }
                do {
                    try Self._validateFinishStatus(status: self.writer.status, underlyingError: self.writer.error)
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
