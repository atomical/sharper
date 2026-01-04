import CoreGraphics
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

    public static func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float> = SIMD3<Float>(0, -1, 0)) -> simd_float4x4 {
        let z = simd_normalize(target - eye) // forward (+Z)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)

        let t = SIMD3<Float>(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye))

        return simd_float4x4(
            SIMD4<Float>(x.x, x.y, x.z, 0),
            SIMD4<Float>(y.x, y.y, y.z, 0),
            SIMD4<Float>(z.x, z.y, z.z, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        )
    }
}

public final class GaussianSplatRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState

    public init(device: MTLDevice? = nil) throws {
        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else { throw GaussianSplatRendererError.metalUnavailable }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw GaussianSplatRendererError.commandQueueCreateFailed
        }
        self.commandQueue = commandQueue

        guard let shaderURL = Bundle.module.url(forResource: "GaussianSplat", withExtension: "metal") else {
            throw GaussianSplatRendererError.libraryLoadFailed
        }
        let source = try String(contentsOf: shaderURL, encoding: .utf8)
        let library = try device.makeLibrary(source: source, options: nil)
        guard let vfn = library.makeFunction(name: "gaussianVertex"),
              let ffn = library.makeFunction(name: "gaussianFragment")
        else {
            throw GaussianSplatRendererError.libraryLoadFailed
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .one
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        self.pipeline = try device.makeRenderPipelineState(descriptor: desc)
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
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        texDesc.usage = [.renderTarget, .shaderRead]
        guard let texture = device.makeTexture(descriptor: texDesc) else {
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

        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = texture
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].storeAction = .store
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rp)
        else {
            throw GaussianSplatRendererError.renderCommandCreateFailed
        }

        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(camBuf, offset: 0, index: 0)
        enc.setVertexBuffer(scene.means, offset: 0, index: 1)
        enc.setVertexBuffer(scene.scales, offset: 0, index: 2)
        enc.setVertexBuffer(scene.colorsLinear, offset: 0, index: 3)
        enc.setVertexBuffer(scene.opacities, offset: 0, index: 4)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: scene.count)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        // Read back BGRA8.
        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)
        texture.getBytes(&bytes, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

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
