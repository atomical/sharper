import CoreGraphics
import Dispatch
import Foundation
import Metal
import simd

public enum GaussianSplatRendererError: Error {
    case metalUnavailable
    case commandQueueCreateFailed
    case libraryLoadFailed
    case pipelineCreateFailed
    case textureCreateFailed
    case renderCommandCreateFailed
}

public struct PinholeCamera {
    public var viewMatrix: simd_float4x4
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float

    public init(viewMatrix: simd_float4x4, fx: Float, fy: Float, cx: Float, cy: Float) {
        self.viewMatrix = viewMatrix
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }

    public static func lookAt(
        eye: SIMD3<Float>,
        target: SIMD3<Float>,
        up: SIMD3<Float> = SIMD3<Float>(0, -1, 0)
    ) -> simd_float4x4 {
        // OpenCV-style camera convention:
        // - +X right
        // - +Y down
        // - +Z forward
        //
        // Build a world->camera view matrix suitable for Metal's column-major `M * p` convention.
        let forward = simd_normalize(target - eye) // +Z
        var right = simd_cross(forward, up)
        if simd_length_squared(right) < 1e-12 {
            // Fallback if up is parallel to forward.
            right = simd_cross(forward, SIMD3<Float>(0, 0, 1))
        }
        right = simd_normalize(right) // +X
        let down = simd_cross(forward, right) // +Y (down)

        let tx = -simd_dot(right, eye)
        let ty = -simd_dot(down, eye)
        let tz = -simd_dot(forward, eye)

        // Column-major: columns are the basis vectors' components per row.
        return simd_float4x4(
            SIMD4<Float>(right.x, down.x, forward.x, 0),
            SIMD4<Float>(right.y, down.y, forward.y, 0),
            SIMD4<Float>(right.z, down.z, forward.z, 0),
            SIMD4<Float>(tx, ty, tz, 1)
        )
    }
}

public final class GaussianSplatRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let splatPipeline: MTLRenderPipelineState
    private let compositePipeline: MTLRenderPipelineState
    private let sampler: MTLSamplerState

    public init(device: MTLDevice? = nil) throws {
        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else { throw GaussianSplatRendererError.metalUnavailable }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw GaussianSplatRendererError.commandQueueCreateFailed
        }
        self.commandQueue = commandQueue

        // Load precompiled metallib embedded in Swift sources (required on iOS/visionOS).
        let libraryData = MetalLibrary_GaussianSplat.data.withUnsafeBytes { DispatchData(bytes: $0) }
        let library = try device.makeLibrary(data: libraryData)
        guard let splatV = library.makeFunction(name: "splatVertex"),
              let splatF = library.makeFunction(name: "splatFragmentOIT"),
              let compV = library.makeFunction(name: "compositeVertex"),
              let compF = library.makeFunction(name: "compositeFragment")
        else {
            throw GaussianSplatRendererError.libraryLoadFailed
        }

        // Pass 1: splat into (accum, revealage) buffers (weighted blended OIT).
        let splatDesc = MTLRenderPipelineDescriptor()
        splatDesc.vertexFunction = splatV
        splatDesc.fragmentFunction = splatF

        splatDesc.colorAttachments[0].pixelFormat = .rgba16Float
        splatDesc.colorAttachments[0].isBlendingEnabled = true
        splatDesc.colorAttachments[0].rgbBlendOperation = .add
        splatDesc.colorAttachments[0].alphaBlendOperation = .add
        splatDesc.colorAttachments[0].sourceRGBBlendFactor = .one
        splatDesc.colorAttachments[0].destinationRGBBlendFactor = .one
        splatDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        splatDesc.colorAttachments[0].destinationAlphaBlendFactor = .one

        splatDesc.colorAttachments[1].pixelFormat = .rgba16Float
        splatDesc.colorAttachments[1].isBlendingEnabled = true
        splatDesc.colorAttachments[1].rgbBlendOperation = .add
        splatDesc.colorAttachments[1].alphaBlendOperation = .add
        splatDesc.colorAttachments[1].sourceRGBBlendFactor = .zero
        splatDesc.colorAttachments[1].destinationRGBBlendFactor = .oneMinusSourceAlpha
        splatDesc.colorAttachments[1].sourceAlphaBlendFactor = .zero
        splatDesc.colorAttachments[1].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        self.splatPipeline = try device.makeRenderPipelineState(descriptor: splatDesc)

        // Pass 2: composite to BGRA8 for readback.
        let compDesc = MTLRenderPipelineDescriptor()
        compDesc.vertexFunction = compV
        compDesc.fragmentFunction = compF
        compDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        self.compositePipeline = try device.makeRenderPipelineState(descriptor: compDesc)

        let sampDesc = MTLSamplerDescriptor()
        sampDesc.minFilter = .nearest
        sampDesc.magFilter = .nearest
        sampDesc.mipFilter = .notMipmapped
        sampDesc.sAddressMode = .clampToEdge
        sampDesc.tAddressMode = .clampToEdge
        guard let sampler = device.makeSamplerState(descriptor: sampDesc) else {
            throw GaussianSplatRendererError.pipelineCreateFailed
        }
        self.sampler = sampler
    }

    private struct CameraParams {
        var viewMatrix: simd_float4x4
        var fx: Float
        var fy: Float
        var cx: Float
        var cy: Float
        var width: UInt32
        var height: UInt32
    }

    public func renderToCGImage(
        scene: GaussianScene,
        camera: PinholeCamera,
        width: Int,
        height: Int
    ) throws -> CGImage {
        let accumDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        accumDesc.usage = [.renderTarget, .shaderRead]
        guard let accumTex = device.makeTexture(descriptor: accumDesc) else {
            throw GaussianSplatRendererError.textureCreateFailed
        }

        let revealDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        revealDesc.usage = [.renderTarget, .shaderRead]
        guard let revealTex = device.makeTexture(descriptor: revealDesc) else {
            throw GaussianSplatRendererError.textureCreateFailed
        }

        let outDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        outDesc.usage = [.renderTarget, .shaderRead]
        guard let outTex = device.makeTexture(descriptor: outDesc) else {
            throw GaussianSplatRendererError.textureCreateFailed
        }

        var cam = CameraParams(
            viewMatrix: camera.viewMatrix,
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            width: UInt32(width),
            height: UInt32(height)
        )
        guard let camBuf = device.makeBuffer(bytes: &cam, length: MemoryLayout<CameraParams>.stride, options: .storageModeShared) else {
            throw GaussianSplatRendererError.renderCommandCreateFailed
        }

        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw GaussianSplatRendererError.renderCommandCreateFailed
        }

        // Pass 1: splats -> accum + revealage.
        do {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = accumTex
            rp.colorAttachments[0].loadAction = .clear
            rp.colorAttachments[0].storeAction = .store
            rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

            rp.colorAttachments[1].texture = revealTex
            rp.colorAttachments[1].loadAction = .clear
            rp.colorAttachments[1].storeAction = .store
            rp.colorAttachments[1].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)

            guard let enc = cmd.makeRenderCommandEncoder(descriptor: rp) else {
                throw GaussianSplatRendererError.renderCommandCreateFailed
            }
            enc.setRenderPipelineState(splatPipeline)
            enc.setVertexBuffer(camBuf, offset: 0, index: 0)
            enc.setVertexBuffer(scene.means, offset: 0, index: 1)
            enc.setVertexBuffer(scene.quaternions, offset: 0, index: 2)
            enc.setVertexBuffer(scene.scales, offset: 0, index: 3)
            enc.setVertexBuffer(scene.colorsLinear, offset: 0, index: 4)
            enc.setVertexBuffer(scene.opacities, offset: 0, index: 5)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: scene.count)
            enc.endEncoding()
        }

        // Pass 2: composite -> BGRA8.
        do {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = outTex
            rp.colorAttachments[0].loadAction = .clear
            rp.colorAttachments[0].storeAction = .store
            rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

            guard let enc = cmd.makeRenderCommandEncoder(descriptor: rp) else {
                throw GaussianSplatRendererError.renderCommandCreateFailed
            }
            enc.setRenderPipelineState(compositePipeline)
            enc.setFragmentTexture(accumTex, index: 0)
            enc.setFragmentTexture(revealTex, index: 1)
            enc.setFragmentSamplerState(sampler, index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            enc.endEncoding()
        }

        cmd.commit()
        cmd.waitUntilCompleted()

        // Read back BGRA8.
        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)
        outTex.getBytes(&bytes, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(.init(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }
}
