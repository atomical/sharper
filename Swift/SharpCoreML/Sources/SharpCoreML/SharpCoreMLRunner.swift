import CoreML
import Dispatch
import Foundation
import Metal

public enum SharpCoreMLError: Error {
    case modelCompileFailed(URL, underlying: Error)
    case modelLoadFailed(URL, underlying: Error)
    case missingOutput(String)
    case invalidOutputType(String)
    case metalUnavailable
    case metalLibraryLoadFailed
    case metalPipelineCreateFailed
    case metalBufferCreateFailed
}

public struct SharpPostprocessedGaussians {
    public let count: Int
    public let mean: MTLBuffer // packed_float3 (12 bytes) * count
    public let quaternions: MTLBuffer // packed_float4 (16 bytes) * count
    public let singularValues: MTLBuffer // packed_float3 (12 bytes) * count
}

public struct SharpPrediction {
    public let metadata: SharpInputMetadata
    public let raw: SharpRawOutputs
    public let postprocessed: SharpPostprocessedGaussians
    public let timings: SharpTimings?

    public init(metadata: SharpInputMetadata, raw: SharpRawOutputs, postprocessed: SharpPostprocessedGaussians, timings: SharpTimings? = nil) {
        self.metadata = metadata
        self.raw = raw
        self.postprocessed = postprocessed
        self.timings = timings
    }
}

public final class SharpCoreMLRunner {
    private let model: MLModel
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    public init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: modelURL)
        } catch {
            throw SharpCoreMLError.modelCompileFailed(modelURL, underlying: error)
        }
        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        } catch {
            throw SharpCoreMLError.modelLoadFailed(compiledURL, underlying: error)
        }

        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue()
        else {
            throw SharpCoreMLError.metalUnavailable
        }
        self.device = device
        self.commandQueue = commandQueue

        // Load precompiled metallib embedded in Swift sources (required on iOS/visionOS).
        let libraryData = MetalLibrary_SharpPostprocess.data.withUnsafeBytes { DispatchData(bytes: $0) }
        let library = try device.makeLibrary(data: libraryData)
        guard let fn = library.makeFunction(name: "unprojectGaussians") else {
            throw SharpCoreMLError.metalLibraryLoadFailed
        }
        self.pipeline = try device.makeComputePipelineState(function: fn)
    }

    public func predict(imageURL: URL) throws -> SharpPrediction {
        let t0 = CFAbsoluteTimeGetCurrent()

        let tPre0 = CFAbsoluteTimeGetCurrent()
        let cgImage = try SharpPreprocessor.loadCGImage(from: imageURL)
        let metadata = SharpPreprocessor.loadMetadata(
            from: imageURL,
            imageWidth: cgImage.width,
            imageHeight: cgImage.height
        )
        let (image, disparity) = try SharpPreprocessor.makeInputs(cgImage: cgImage, metadata: metadata)
        let tPre1 = CFAbsoluteTimeGetCurrent()

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: image),
            "disparity_factor": MLFeatureValue(multiArray: disparity),
        ])
        let tInf0 = CFAbsoluteTimeGetCurrent()
        let features = try model.prediction(from: provider)
        let tInf1 = CFAbsoluteTimeGetCurrent()

        func get(_ name: String) throws -> MLMultiArray {
            guard let v = features.featureValue(for: name) else { throw SharpCoreMLError.missingOutput(name) }
            guard let arr = v.multiArrayValue else { throw SharpCoreMLError.invalidOutputType(name) }
            return arr
        }

        let meanPre = try get("mean_vectors_pre")
        let quatPre = try get("quaternions_pre")
        let scalePre = try get("singular_values_pre")
        let colorsPre = try get("colors_linear_pre")
        let opacitiesPre = try get("opacities_pre")

        let (post, unprojectTiming) = try unproject(meanPre: meanPre, quatPre: quatPre, scalePre: scalePre, metadata: metadata)

        let raw = SharpRawOutputs(
            meanVectorsPre: meanPre,
            quaternionsPre: quatPre,
            singularValuesPre: scalePre,
            colorsLinearPre: colorsPre,
            opacitiesPre: opacitiesPre
        )
        let t1 = CFAbsoluteTimeGetCurrent()
        let timings = SharpTimings(
            preprocessSec: tPre1 - tPre0,
            coremlSec: tInf1 - tInf0,
            postprocessSec: unprojectTiming.totalSec,
            postprocessCopySec: unprojectTiming.copySec,
            postprocessKernelSec: unprojectTiming.kernelSec,
            totalSec: t1 - t0
        )
        return SharpPrediction(metadata: metadata, raw: raw, postprocessed: post, timings: timings)
    }

    private struct UnprojectParams {
        var sx: Float
        var sy: Float
        var count: UInt32
        var _pad: UInt32 = 0
    }

    private struct UnprojectTiming {
        var copySec: Double
        var kernelSec: Double
        var totalSec: Double
    }

    private func unproject(
        meanPre: MLMultiArray,
        quatPre: MLMultiArray,
        scalePre: MLMultiArray,
        metadata: SharpInputMetadata
    ) throws -> (SharpPostprocessedGaussians, UnprojectTiming) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let count = meanPre.shape[1].intValue

        // Inputs are contiguous float32; copy into shared buffers for simplicity.
        let meanBytes = count * 3 * MemoryLayout<Float>.size
        let quatBytes = count * 4 * MemoryLayout<Float>.size
        let scaleBytes = count * 3 * MemoryLayout<Float>.size

        let tCopy0 = CFAbsoluteTimeGetCurrent()
        guard let meanIn = device.makeBuffer(bytes: meanPre.dataPointer, length: meanBytes, options: .storageModeShared),
              let quatIn = device.makeBuffer(bytes: quatPre.dataPointer, length: quatBytes, options: .storageModeShared),
              let scaleIn = device.makeBuffer(bytes: scalePre.dataPointer, length: scaleBytes, options: .storageModeShared)
        else {
            throw SharpCoreMLError.metalBufferCreateFailed
        }

        guard let meanOut = device.makeBuffer(length: meanBytes, options: .storageModeShared),
              let quatOut = device.makeBuffer(length: quatBytes, options: .storageModeShared),
              let scaleOut = device.makeBuffer(length: scaleBytes, options: .storageModeShared)
        else {
            throw SharpCoreMLError.metalBufferCreateFailed
        }
        let tCopy1 = CFAbsoluteTimeGetCurrent()

        var params = UnprojectParams(sx: metadata.sx, sy: metadata.sy, count: UInt32(count))
        guard let paramsBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<UnprojectParams>.stride, options: .storageModeShared) else {
            throw SharpCoreMLError.metalBufferCreateFailed
        }

        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else {
            throw SharpCoreMLError.metalPipelineCreateFailed
        }

        let tKernel0 = CFAbsoluteTimeGetCurrent()
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(meanIn, offset: 0, index: 0)
        enc.setBuffer(quatIn, offset: 0, index: 1)
        enc.setBuffer(scaleIn, offset: 0, index: 2)
        enc.setBuffer(meanOut, offset: 0, index: 3)
        enc.setBuffer(quatOut, offset: 0, index: 4)
        enc.setBuffer(scaleOut, offset: 0, index: 5)
        enc.setBuffer(paramsBuf, offset: 0, index: 6)

        let w = pipeline.threadExecutionWidth
        let tg = MTLSize(width: w, height: 1, depth: 1)
        let grid = MTLSize(width: (count + w - 1) / w * w, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()
        let tKernel1 = CFAbsoluteTimeGetCurrent()

        let t1 = CFAbsoluteTimeGetCurrent()
        return (
            SharpPostprocessedGaussians(count: count, mean: meanOut, quaternions: quatOut, singularValues: scaleOut),
            UnprojectTiming(copySec: tCopy1 - tCopy0, kernelSec: tKernel1 - tKernel0, totalSec: t1 - t0)
        )
    }
}
