import CoreGraphics
import Dispatch
import Foundation
import Darwin
import Metal
import simd

public enum GaussianSplatRendererError: Error {
    case metalUnavailable
    case commandQueueCreateFailed
    case libraryLoadFailed
    case pipelineCreateFailed
    case computePipelineCreateFailed
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
    private let splatOITPipeline: MTLRenderPipelineState
    private let splatAlphaPipeline: MTLRenderPipelineState
    private let compositeOITPipeline: MTLRenderPipelineState
    private let compositeAlphaPipeline: MTLRenderPipelineState
    private let downsamplePipeline: MTLRenderPipelineState
    private let binCountPipeline: MTLComputePipelineState
    private let binScatterPipeline: MTLComputePipelineState
    private let nearestSampler: MTLSamplerState
    private let linearSampler: MTLSamplerState

    public init(device: MTLDevice? = nil) throws {
        let device = try (device ?? MTLCreateSystemDefaultDevice()).orThrow(GaussianSplatRendererError.metalUnavailable)
        self.device = device

        self.commandQueue = try device.makeCommandQueue().orThrow(GaussianSplatRendererError.commandQueueCreateFailed)

        // Load precompiled metallib embedded in Swift sources (required on iOS/visionOS).
        let libraryData = MetalLibrary_GaussianSplat.data.withUnsafeBytes { DispatchData(bytes: $0) }
        let library = try device.makeLibrary(data: libraryData)
        func f(_ name: String) throws -> MTLFunction {
            try library.makeFunction(name: name).orThrow(GaussianSplatRendererError.libraryLoadFailed)
        }
        let splatV = try f("splatVertex")
        let splatVSorted = try f("splatVertexSorted")
        let splatFOIT = try f("splatFragmentOIT")
        let splatFAlpha = try f("splatFragmentAlpha")
        let compV = try f("compositeVertex")
        let compFOIT = try f("compositeFragment")
        let compFAlpha = try f("compositeFromRGBA")
        let downF = try f("downsampleFragment")
        let binCount = try f("depthBinCount")
        let binScatter = try f("depthBinScatter")

        // Pass 1A: weighted blended OIT into (accum, revealage, aux).
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = splatV
            desc.fragmentFunction = splatFOIT

            desc.colorAttachments[0].pixelFormat = .rgba16Float
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].rgbBlendOperation = .add
            desc.colorAttachments[0].alphaBlendOperation = .add
            desc.colorAttachments[0].sourceRGBBlendFactor = .one
            desc.colorAttachments[0].destinationRGBBlendFactor = .one
            desc.colorAttachments[0].sourceAlphaBlendFactor = .one
            desc.colorAttachments[0].destinationAlphaBlendFactor = .one

            desc.colorAttachments[1].pixelFormat = .rgba16Float
            desc.colorAttachments[1].isBlendingEnabled = true
            desc.colorAttachments[1].rgbBlendOperation = .add
            desc.colorAttachments[1].alphaBlendOperation = .add
            desc.colorAttachments[1].sourceRGBBlendFactor = .zero
            desc.colorAttachments[1].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.colorAttachments[1].sourceAlphaBlendFactor = .zero
            desc.colorAttachments[1].destinationAlphaBlendFactor = .oneMinusSourceAlpha

            desc.colorAttachments[2].pixelFormat = .rgba16Float
            desc.colorAttachments[2].isBlendingEnabled = true
            desc.colorAttachments[2].rgbBlendOperation = .add
            desc.colorAttachments[2].alphaBlendOperation = .add
            desc.colorAttachments[2].sourceRGBBlendFactor = .one
            desc.colorAttachments[2].destinationRGBBlendFactor = .one
            desc.colorAttachments[2].sourceAlphaBlendFactor = .one
            desc.colorAttachments[2].destinationAlphaBlendFactor = .one

            self.splatOITPipeline = try device.makeRenderPipelineState(descriptor: desc)
        }

        // Pass 1B: depth-binned alpha compositing into (color, aux).
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = splatVSorted
            desc.fragmentFunction = splatFAlpha

            desc.colorAttachments[0].pixelFormat = .rgba16Float
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].rgbBlendOperation = .add
            desc.colorAttachments[0].alphaBlendOperation = .add
            desc.colorAttachments[0].sourceRGBBlendFactor = .one
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.colorAttachments[0].sourceAlphaBlendFactor = .one
            desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

            desc.colorAttachments[1].pixelFormat = .rgba16Float
            desc.colorAttachments[1].isBlendingEnabled = true
            desc.colorAttachments[1].rgbBlendOperation = .add
            desc.colorAttachments[1].alphaBlendOperation = .add
            desc.colorAttachments[1].sourceRGBBlendFactor = .one
            desc.colorAttachments[1].destinationRGBBlendFactor = .one
            desc.colorAttachments[1].sourceAlphaBlendFactor = .one
            desc.colorAttachments[1].destinationAlphaBlendFactor = .one

            self.splatAlphaPipeline = try device.makeRenderPipelineState(descriptor: desc)
        }

        // Pass 2A: composite OIT -> display (BGRA8 sRGB).
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = compV
            desc.fragmentFunction = compFOIT
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
            self.compositeOITPipeline = try device.makeRenderPipelineState(descriptor: desc)
        }

        // Pass 2B: composite RGBA -> display (BGRA8 sRGB).
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = compV
            desc.fragmentFunction = compFAlpha
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
            self.compositeAlphaPipeline = try device.makeRenderPipelineState(descriptor: desc)
        }

        // Pass 3: downsample BGRA8 sRGB -> BGRA8 sRGB.
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = compV
            desc.fragmentFunction = downF
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
            self.downsamplePipeline = try device.makeRenderPipelineState(descriptor: desc)
        }

        self.binCountPipeline = try device.makeComputePipelineState(function: binCount)
        self.binScatterPipeline = try device.makeComputePipelineState(function: binScatter)

        func makeSampler(min: MTLSamplerMinMagFilter, mag: MTLSamplerMinMagFilter) throws -> MTLSamplerState {
            let d = MTLSamplerDescriptor()
            d.minFilter = min
            d.magFilter = mag
            d.mipFilter = .notMipmapped
            d.sAddressMode = .clampToEdge
            d.tAddressMode = .clampToEdge
            return try device.makeSamplerState(descriptor: d).orThrow(GaussianSplatRendererError.pipelineCreateFailed)
        }
        self.nearestSampler = try makeSampler(min: .nearest, mag: .nearest)
        self.linearSampler = try makeSampler(min: .linear, mag: .linear)
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

    private struct RenderParams {
        var nearClipZ: Float
        var opacityThreshold: Float
        var lowPassEps2D: Float
        var minRadiusPx: Float

        var maxRadiusPx: Float
        var exposureEV: Float
        var saturation: Float
        var contrast: Float

        var toneMap: UInt32
        var debugView: UInt32
        var debugNearZ: Float
        var debugFarZ: Float
    }

    private struct DepthBinParams {
        var viewMatrix: simd_float4x4
        var nearClipZ: Float
        var opacityThreshold: Float
        var zNear: Float
        var zFar: Float
        var binCount: UInt32
        var count: UInt32
    }

    public func renderToCGImage(
        scene: GaussianScene,
        camera: PinholeCamera,
        width: Int,
        height: Int
    ) throws -> CGImage {
        try renderToCGImage(scene: scene, camera: camera, width: width, height: height, options: GaussianSplatRenderOptions())
    }

    public func renderToCGImage(
        scene: GaussianScene,
        camera: PinholeCamera,
        width: Int,
        height: Int,
        options: GaussianSplatRenderOptions
    ) throws -> CGImage {
        let wOut = max(width, 1)
        let hOut = max(height, 1)

        let renderScale = max(1.0, min(options.renderScale, 4.0))
        let hiW = max(1, Int((Float(wOut) * renderScale).rounded(.toNearestOrAwayFromZero)))
        let hiH = max(1, Int((Float(hOut) * renderScale).rounded(.toNearestOrAwayFromZero)))
        let useSSAA = (hiW != wOut) || (hiH != hOut)

        func makeTex(_ fmt: MTLPixelFormat, _ w: Int, _ h: Int, _ usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.usage = usage
            return try device.makeTexture(descriptor: d).orThrow(GaussianSplatRendererError.textureCreateFailed)
        }

        let outTex = try makeTex(.bgra8Unorm_srgb, wOut, hOut, [.renderTarget, .shaderRead])
        let hiOutTex = useSSAA ? try makeTex(.bgra8Unorm_srgb, hiW, hiH, [.renderTarget, .shaderRead]) : outTex

        let normTransform = normalizationTransform(scene: scene, options: options)
        let viewMatrix = camera.viewMatrix * normTransform

        let sx = Float(hiW) / Float(wOut)
        let sy = Float(hiH) / Float(hOut)

        var cam = CameraParams(
            viewMatrix: viewMatrix,
            fx: camera.fx * sx,
            fy: camera.fy * sy,
            cx: camera.cx * sx,
            cy: camera.cy * sy,
            width: UInt32(hiW),
            height: UInt32(hiH)
        )
        let camBuf = try device
            .makeBuffer(bytes: &cam, length: MemoryLayout<CameraParams>.stride, options: .storageModeShared)
            .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

        let needsDepthRange: Bool = {
            switch options.debugView {
            case .depth, .disparity:
                return true
            case .none, .alpha, .radius:
                break
            }
            switch options.compositing {
            case .depthBinnedAlpha:
                return true
            case .weightedOIT:
                return false
            }
        }()

        let depthRange: SIMD2<Float> = {
            if let r = options.debugDepthRange { return r }
            guard needsDepthRange else {
                return SIMD2<Float>(options.nearClipZ, options.nearClipZ + 10.0)
            }
            let d = MLSharpTrajectory.depthQuantiles(
                scene: scene,
                sampleCount: 65536,
                opacityThreshold: max(options.opacityThreshold, 0.01),
                qNear: 0.001,
                qFocus: 0.10,
                qFar: 0.999
            )
            return SIMD2<Float>(d.min, d.max)
        }()
        let debugNear = min(depthRange.x, depthRange.y)
        let debugFar = max(depthRange.x, depthRange.y)

        let radiusScale = max(sx, sy)
        var rp = RenderParams(
            nearClipZ: options.nearClipZ,
            opacityThreshold: options.opacityThreshold,
            lowPassEps2D: max(0.0, options.lowPassEps2D) * radiusScale * radiusScale,
            minRadiusPx: max(0.0, options.minRadiusPx) * radiusScale,
            maxRadiusPx: max(1.0, options.maxRadiusPx) * radiusScale,
            exposureEV: options.exposureEV,
            saturation: options.saturation,
            contrast: options.contrast,
            toneMap: toneMapIndex(options.toneMap),
            debugView: debugViewIndex(options.debugView),
            debugNearZ: max(debugNear, 1e-6),
            debugFarZ: max(debugFar, max(debugNear, 1e-6) + 1e-3)
        )
        let rpBuf = try device
            .makeBuffer(bytes: &rp, length: MemoryLayout<RenderParams>.stride, options: .storageModeShared)
            .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

        switch options.compositing {
        case .weightedOIT:
            let accumTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])
            let revealTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])
            let auxTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])

            let cmd = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

            // Pass 1: splats -> accum + revealage + aux.
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = accumTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                rpd.colorAttachments[1].texture = revealTex
                rpd.colorAttachments[1].loadAction = .clear
                rpd.colorAttachments[1].storeAction = .store
                rpd.colorAttachments[1].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)

                rpd.colorAttachments[2].texture = auxTex
                rpd.colorAttachments[2].loadAction = .clear
                rpd.colorAttachments[2].storeAction = .store
                rpd.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                let enc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                enc.setRenderPipelineState(splatOITPipeline)
                enc.setVertexBuffer(camBuf, offset: 0, index: 0)
                enc.setVertexBuffer(scene.means, offset: 0, index: 1)
                enc.setVertexBuffer(scene.quaternions, offset: 0, index: 2)
                enc.setVertexBuffer(scene.scales, offset: 0, index: 3)
                enc.setVertexBuffer(scene.colorsLinear, offset: 0, index: 4)
                enc.setVertexBuffer(scene.opacities, offset: 0, index: 5)
                enc.setVertexBuffer(rpBuf, offset: 0, index: 6)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: scene.count)
                enc.endEncoding()
            }

            // Pass 2: composite -> BGRA8(sRGB) (either hi or final).
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = hiOutTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let enc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                enc.setRenderPipelineState(compositeOITPipeline)
                enc.setFragmentTexture(accumTex, index: 0)
                enc.setFragmentTexture(revealTex, index: 1)
                enc.setFragmentTexture(auxTex, index: 2)
                enc.setFragmentSamplerState(nearestSampler, index: 0)
                enc.setFragmentBuffer(rpBuf, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                enc.endEncoding()
            }

            // Pass 3: SSAA downsample.
            if useSSAA {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = outTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let enc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                enc.setRenderPipelineState(downsamplePipeline)
                enc.setFragmentTexture(hiOutTex, index: 0)
                enc.setFragmentSamplerState(linearSampler, index: 0)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                enc.endEncoding()
            }

            cmd.commit()
            cmd.waitUntilCompleted()

        case .depthBinnedAlpha(let binCount):
            let bins = max(1, min(binCount, 2048))
            let colorTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])
            let auxTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])

            // 1) Count bins (GPU) -> prefix sum (CPU).
            var binParams = DepthBinParams(
                viewMatrix: viewMatrix,
                nearClipZ: options.nearClipZ,
                opacityThreshold: options.opacityThreshold,
                zNear: debugNear,
                zFar: debugFar,
                binCount: UInt32(bins),
                count: UInt32(scene.count)
            )
            let binParamsBuf = try device
                .makeBuffer(bytes: &binParams, length: MemoryLayout<DepthBinParams>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

            let binCountsBuf = try device
                .makeBuffer(length: bins * MemoryLayout<UInt32>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            memset(binCountsBuf.contents(), 0, bins * MemoryLayout<UInt32>.stride)

            let cmdCount = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            let enc = try cmdCount.makeComputeCommandEncoder().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            enc.setComputePipelineState(binCountPipeline)
            enc.setBuffer(binParamsBuf, offset: 0, index: 0)
            enc.setBuffer(scene.means, offset: 0, index: 1)
            enc.setBuffer(scene.opacities, offset: 0, index: 2)
            enc.setBuffer(binCountsBuf, offset: 0, index: 3)
            let w = binCountPipeline.threadExecutionWidth
            let tg = MTLSize(width: w, height: 1, depth: 1)
            let grid = MTLSize(width: ((scene.count + w - 1) / w) * w, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
            cmdCount.commit()
            cmdCount.waitUntilCompleted()

            let countsPtr = binCountsBuf.contents().bindMemory(to: UInt32.self, capacity: bins)
            var offsets = [UInt32](repeating: 0, count: bins)
            var total: UInt32 = 0
            for i in 0..<bins {
                offsets[i] = total
                total &+= countsPtr[i]
            }
            let visibleCount = Int(total)

            // If nothing is visible, just return a blank image.
            if visibleCount == 0 {
                let bytesPerRow = wOut * 4
                let bytes = [UInt8](repeating: 0, count: bytesPerRow * hOut)
                let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
                let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(.init(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
                let provider = CGDataProvider(data: Data(bytes) as CFData)!
                return CGImage(
                    width: wOut,
                    height: hOut,
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

            // 2) Scatter indices (GPU), then render+composite.
            let cursorBuf = try device
                .makeBuffer(length: bins * MemoryLayout<UInt32>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            let sortedBuf = try device
                .makeBuffer(length: visibleCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            offsets.withUnsafeBytes { src in
                _ = memcpy(cursorBuf.contents(), src.baseAddress!, bins * MemoryLayout<UInt32>.stride)
            }

            let cmd = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

            // Scatter pass.
            do {
                let cenc = try cmd.makeComputeCommandEncoder().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                cenc.setComputePipelineState(binScatterPipeline)
                cenc.setBuffer(binParamsBuf, offset: 0, index: 0)
                cenc.setBuffer(scene.means, offset: 0, index: 1)
                cenc.setBuffer(scene.opacities, offset: 0, index: 2)
                cenc.setBuffer(cursorBuf, offset: 0, index: 3)
                cenc.setBuffer(sortedBuf, offset: 0, index: 4)
                let w = binScatterPipeline.threadExecutionWidth
                let tg = MTLSize(width: w, height: 1, depth: 1)
                let grid = MTLSize(width: ((scene.count + w - 1) / w) * w, height: 1, depth: 1)
                cenc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                cenc.endEncoding()
            }

            // Render pass: sorted alpha blend -> (color, aux).
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = colorTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                rpd.colorAttachments[1].texture = auxTex
                rpd.colorAttachments[1].loadAction = .clear
                rpd.colorAttachments[1].storeAction = .store
                rpd.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.setRenderPipelineState(splatAlphaPipeline)
                renc.setVertexBuffer(camBuf, offset: 0, index: 0)
                renc.setVertexBuffer(scene.means, offset: 0, index: 1)
                renc.setVertexBuffer(scene.quaternions, offset: 0, index: 2)
                renc.setVertexBuffer(scene.scales, offset: 0, index: 3)
                renc.setVertexBuffer(scene.colorsLinear, offset: 0, index: 4)
                renc.setVertexBuffer(scene.opacities, offset: 0, index: 5)
                renc.setVertexBuffer(sortedBuf, offset: 0, index: 6)
                renc.setVertexBuffer(rpBuf, offset: 0, index: 7)
                renc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: visibleCount)
                renc.endEncoding()
            }

            // Composite -> BGRA8(sRGB).
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = hiOutTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.setRenderPipelineState(compositeAlphaPipeline)
                renc.setFragmentTexture(colorTex, index: 0)
                renc.setFragmentTexture(auxTex, index: 1)
                renc.setFragmentSamplerState(nearestSampler, index: 0)
                renc.setFragmentBuffer(rpBuf, offset: 0, index: 0)
                renc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                renc.endEncoding()
            }

            // SSAA downsample.
            if useSSAA {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = outTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.setRenderPipelineState(downsamplePipeline)
                renc.setFragmentTexture(hiOutTex, index: 0)
                renc.setFragmentSamplerState(linearSampler, index: 0)
                renc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                renc.endEncoding()
            }

            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Read back BGRA8(sRGB).
        let bytesPerRow = wOut * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * hOut)
        outTex.getBytes(&bytes, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, wOut, hOut), mipmapLevel: 0)

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(.init(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(
            width: wOut,
            height: hOut,
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

    /// Render a scene into an existing output texture. The caller is responsible for presenting and committing the
    /// returned command buffer (e.g. `cmd.present(drawable); cmd.commit()`).
    public func renderToTexture(
        scene: GaussianScene,
        camera: PinholeCamera,
        outputTexture: MTLTexture,
        options: GaussianSplatRenderOptions
    ) throws -> MTLCommandBuffer {
        let wOut = max(outputTexture.width, 1)
        let hOut = max(outputTexture.height, 1)

        let renderScale = max(1.0, min(options.renderScale, 4.0))
        let hiW = max(1, Int((Float(wOut) * renderScale).rounded(.toNearestOrAwayFromZero)))
        let hiH = max(1, Int((Float(hOut) * renderScale).rounded(.toNearestOrAwayFromZero)))
        let useSSAA = (hiW != wOut) || (hiH != hOut)

        func makeTex(_ fmt: MTLPixelFormat, _ w: Int, _ h: Int, _ usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.usage = usage
            return try device.makeTexture(descriptor: d).orThrow(GaussianSplatRendererError.textureCreateFailed)
        }

        let outTex = outputTexture
        let hiOutTex = useSSAA ? try makeTex(.bgra8Unorm_srgb, hiW, hiH, [.renderTarget, .shaderRead]) : outTex

        let normTransform = normalizationTransform(scene: scene, options: options)
        let viewMatrix = camera.viewMatrix * normTransform

        let sx = Float(hiW) / Float(wOut)
        let sy = Float(hiH) / Float(hOut)

        var cam = CameraParams(
            viewMatrix: viewMatrix,
            fx: camera.fx * sx,
            fy: camera.fy * sy,
            cx: camera.cx * sx,
            cy: camera.cy * sy,
            width: UInt32(hiW),
            height: UInt32(hiH)
        )
        let camBuf = try device
            .makeBuffer(bytes: &cam, length: MemoryLayout<CameraParams>.stride, options: .storageModeShared)
            .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

        let needsDepthRange: Bool = {
            switch options.debugView {
            case .depth, .disparity:
                return true
            case .none, .alpha, .radius:
                break
            }
            switch options.compositing {
            case .depthBinnedAlpha:
                return true
            case .weightedOIT:
                return false
            }
        }()

        let depthRange: SIMD2<Float> = {
            if let r = options.debugDepthRange { return r }
            guard needsDepthRange else {
                return SIMD2<Float>(options.nearClipZ, options.nearClipZ + 10.0)
            }
            let d = MLSharpTrajectory.depthQuantiles(
                scene: scene,
                sampleCount: 65536,
                opacityThreshold: max(options.opacityThreshold, 0.01),
                qNear: 0.001,
                qFocus: 0.10,
                qFar: 0.999
            )
            return SIMD2<Float>(d.min, d.max)
        }()
        let debugNear = min(depthRange.x, depthRange.y)
        let debugFar = max(depthRange.x, depthRange.y)

        let radiusScale = max(sx, sy)
        var rp = RenderParams(
            nearClipZ: options.nearClipZ,
            opacityThreshold: options.opacityThreshold,
            lowPassEps2D: max(0.0, options.lowPassEps2D) * radiusScale * radiusScale,
            minRadiusPx: max(0.0, options.minRadiusPx) * radiusScale,
            maxRadiusPx: max(1.0, options.maxRadiusPx) * radiusScale,
            exposureEV: options.exposureEV,
            saturation: options.saturation,
            contrast: options.contrast,
            toneMap: toneMapIndex(options.toneMap),
            debugView: debugViewIndex(options.debugView),
            debugNearZ: max(debugNear, 1e-6),
            debugFarZ: max(debugFar, max(debugNear, 1e-6) + 1e-3)
        )
        let rpBuf = try device
            .makeBuffer(bytes: &rp, length: MemoryLayout<RenderParams>.stride, options: .storageModeShared)
            .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

        switch options.compositing {
        case .weightedOIT:
            let accumTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])
            let revealTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])
            let auxTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])

            let cmd = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

            // Pass 1: splats -> accum + revealage + aux.
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = accumTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                rpd.colorAttachments[1].texture = revealTex
                rpd.colorAttachments[1].loadAction = .clear
                rpd.colorAttachments[1].storeAction = .store
                rpd.colorAttachments[1].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)

                rpd.colorAttachments[2].texture = auxTex
                rpd.colorAttachments[2].loadAction = .clear
                rpd.colorAttachments[2].storeAction = .store
                rpd.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                let enc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                enc.setRenderPipelineState(splatOITPipeline)
                enc.setVertexBuffer(camBuf, offset: 0, index: 0)
                enc.setVertexBuffer(scene.means, offset: 0, index: 1)
                enc.setVertexBuffer(scene.quaternions, offset: 0, index: 2)
                enc.setVertexBuffer(scene.scales, offset: 0, index: 3)
                enc.setVertexBuffer(scene.colorsLinear, offset: 0, index: 4)
                enc.setVertexBuffer(scene.opacities, offset: 0, index: 5)
                enc.setVertexBuffer(rpBuf, offset: 0, index: 6)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: scene.count)
                enc.endEncoding()
            }

            // Pass 2: composite -> BGRA8(sRGB) (either hi or final).
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = hiOutTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let enc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                enc.setRenderPipelineState(compositeOITPipeline)
                enc.setFragmentTexture(accumTex, index: 0)
                enc.setFragmentTexture(revealTex, index: 1)
                enc.setFragmentTexture(auxTex, index: 2)
                enc.setFragmentSamplerState(nearestSampler, index: 0)
                enc.setFragmentBuffer(rpBuf, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                enc.endEncoding()
            }

            // Pass 3: SSAA downsample.
            if useSSAA {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = outTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let enc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                enc.setRenderPipelineState(downsamplePipeline)
                enc.setFragmentTexture(hiOutTex, index: 0)
                enc.setFragmentSamplerState(linearSampler, index: 0)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                enc.endEncoding()
            }

            return cmd

        case .depthBinnedAlpha(let binCount):
            let bins = max(1, min(binCount, 2048))
            let colorTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])
            let auxTex = try makeTex(.rgba16Float, hiW, hiH, [.renderTarget, .shaderRead])

            // 1) Count bins (GPU) -> prefix sum (CPU).
            var binParams = DepthBinParams(
                viewMatrix: viewMatrix,
                nearClipZ: options.nearClipZ,
                opacityThreshold: options.opacityThreshold,
                zNear: debugNear,
                zFar: debugFar,
                binCount: UInt32(bins),
                count: UInt32(scene.count)
            )
            let binParamsBuf = try device
                .makeBuffer(bytes: &binParams, length: MemoryLayout<DepthBinParams>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

            let binCountsBuf = try device
                .makeBuffer(length: bins * MemoryLayout<UInt32>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            memset(binCountsBuf.contents(), 0, bins * MemoryLayout<UInt32>.stride)

            let cmdCount = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            let enc = try cmdCount.makeComputeCommandEncoder().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            enc.setComputePipelineState(binCountPipeline)
            enc.setBuffer(binParamsBuf, offset: 0, index: 0)
            enc.setBuffer(scene.means, offset: 0, index: 1)
            enc.setBuffer(scene.opacities, offset: 0, index: 2)
            enc.setBuffer(binCountsBuf, offset: 0, index: 3)
            let w = binCountPipeline.threadExecutionWidth
            let tg = MTLSize(width: w, height: 1, depth: 1)
            let grid = MTLSize(width: ((scene.count + w - 1) / w) * w, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
            cmdCount.commit()
            cmdCount.waitUntilCompleted()

            let countsPtr = binCountsBuf.contents().bindMemory(to: UInt32.self, capacity: bins)
            var offsets = [UInt32](repeating: 0, count: bins)
            var total: UInt32 = 0
            for i in 0..<bins {
                offsets[i] = total
                total &+= countsPtr[i]
            }
            let visibleCount = Int(total)

            if visibleCount == 0 {
                let cmd = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = outTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.endEncoding()
                return cmd
            }

            // 2) Scatter indices (GPU), then render+composite.
            let cursorBuf = try device
                .makeBuffer(length: bins * MemoryLayout<UInt32>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            let sortedBuf = try device
                .makeBuffer(length: visibleCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)
                .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
            offsets.withUnsafeBytes { src in
                _ = memcpy(cursorBuf.contents(), src.baseAddress!, bins * MemoryLayout<UInt32>.stride)
            }

            let cmd = try commandQueue.makeCommandBuffer().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)

            // Scatter pass.
            do {
                let cenc = try cmd.makeComputeCommandEncoder().orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                cenc.setComputePipelineState(binScatterPipeline)
                cenc.setBuffer(binParamsBuf, offset: 0, index: 0)
                cenc.setBuffer(scene.means, offset: 0, index: 1)
                cenc.setBuffer(scene.opacities, offset: 0, index: 2)
                cenc.setBuffer(cursorBuf, offset: 0, index: 3)
                cenc.setBuffer(sortedBuf, offset: 0, index: 4)
                let w = binScatterPipeline.threadExecutionWidth
                let tg = MTLSize(width: w, height: 1, depth: 1)
                let grid = MTLSize(width: ((scene.count + w - 1) / w) * w, height: 1, depth: 1)
                cenc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                cenc.endEncoding()
            }

            // Render pass: sorted alpha blend -> (color, aux).
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = colorTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                rpd.colorAttachments[1].texture = auxTex
                rpd.colorAttachments[1].loadAction = .clear
                rpd.colorAttachments[1].storeAction = .store
                rpd.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.setRenderPipelineState(splatAlphaPipeline)
                renc.setVertexBuffer(camBuf, offset: 0, index: 0)
                renc.setVertexBuffer(scene.means, offset: 0, index: 1)
                renc.setVertexBuffer(scene.quaternions, offset: 0, index: 2)
                renc.setVertexBuffer(scene.scales, offset: 0, index: 3)
                renc.setVertexBuffer(scene.colorsLinear, offset: 0, index: 4)
                renc.setVertexBuffer(scene.opacities, offset: 0, index: 5)
                renc.setVertexBuffer(sortedBuf, offset: 0, index: 6)
                renc.setVertexBuffer(rpBuf, offset: 0, index: 7)
                renc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: visibleCount)
                renc.endEncoding()
            }

            // Composite -> BGRA8(sRGB).
            do {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = hiOutTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.setRenderPipelineState(compositeAlphaPipeline)
                renc.setFragmentTexture(colorTex, index: 0)
                renc.setFragmentTexture(auxTex, index: 1)
                renc.setFragmentSamplerState(nearestSampler, index: 0)
                renc.setFragmentBuffer(rpBuf, offset: 0, index: 0)
                renc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                renc.endEncoding()
            }

            // SSAA downsample.
            if useSSAA {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = outTex
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

                let renc = try cmd
                    .makeRenderCommandEncoder(descriptor: rpd)
                    .orThrow(GaussianSplatRendererError.renderCommandCreateFailed)
                renc.setRenderPipelineState(downsamplePipeline)
                renc.setFragmentTexture(hiOutTex, index: 0)
                renc.setFragmentSamplerState(linearSampler, index: 0)
                renc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                renc.endEncoding()
            }

            return cmd
        }
    }

    private func toneMapIndex(_ v: GaussianSplatToneMap) -> UInt32 {
        switch v {
        case .none: return 0
        case .reinhard: return 1
        case .aces: return 2
        }
    }

    private func debugViewIndex(_ v: GaussianSplatDebugView) -> UInt32 {
        switch v {
        case .none: return 0
        case .alpha: return 1
        case .depth: return 2
        case .disparity: return 3
        case .radius: return 4
        }
    }

    private func normalizationTransform(scene: GaussianScene, options: GaussianSplatRenderOptions) -> simd_float4x4 {
        let mode = options.normalization
        let scaleMode = options.normalizationScale
        guard mode != .none || scaleMode != .none else { return matrix_identity_float4x4 }

        let center = estimateMedianCenter(
            scene: scene,
            mode: mode,
            sampleCount: options.normalizationSampleCount,
            opacityThreshold: options.normalizationOpacityThreshold
        )

        var scale: Float = 1.0
        if scaleMode == .unitRadius {
            let b = GaussianSceneBounds.estimate(
                scene: scene,
                sampleCount: options.normalizationSampleCount,
                opacityThreshold: options.normalizationOpacityThreshold,
                quantileLo: 0.10,
                quantileHi: 0.90
            )
            scale = 1.0 / max(b.radius, 1e-6)
            scale = min(max(scale, 1e-3), 1e3)
        }

        var t = matrix_identity_float4x4
        t.columns.3 = SIMD4<Float>(-center.x, -center.y, -center.z, 1)

        var s = matrix_identity_float4x4
        s.columns.0.x = scale
        s.columns.1.y = scale
        s.columns.2.z = scale

        return s * t
    }

    private func estimateMedianCenter(
        scene: GaussianScene,
        mode: GaussianSplatSceneNormalization,
        sampleCount: Int,
        opacityThreshold: Float
    ) -> SIMD3<Float> {
        let count = max(scene.count, 0)
        guard count > 0 else { return SIMD3<Float>(repeating: 0) }

        let stride = max(1, count / max(sampleCount, 1))
        let meanPtr = scene.means.contents().bindMemory(to: Float.self, capacity: count * 3)
        let opaPtr = scene.opacities.contents().bindMemory(to: Float.self, capacity: count)

        var xs: [Float] = []
        var ys: [Float] = []
        var zs: [Float] = []
        xs.reserveCapacity(min(sampleCount, count))
        ys.reserveCapacity(min(sampleCount, count))
        zs.reserveCapacity(min(sampleCount, count))

        func add(_ i: Int) {
            let x = meanPtr[i * 3 + 0]
            let y = meanPtr[i * 3 + 1]
            let z = meanPtr[i * 3 + 2]
            if x.isFinite, y.isFinite, z.isFinite {
                xs.append(x)
                ys.append(y)
                zs.append(z)
            }
        }

        var idx = 0
        while idx < count {
            let opa = opaPtr[idx]
            if opa.isFinite, opa >= opacityThreshold {
                add(idx)
            }
            idx += stride
        }

        if xs.count < 1024 {
            xs.removeAll(keepingCapacity: true)
            ys.removeAll(keepingCapacity: true)
            zs.removeAll(keepingCapacity: true)
            idx = 0
            while idx < count {
                add(idx)
                idx += stride
            }
        }

        guard xs.count >= 16 else { return SIMD3<Float>(repeating: 0) }
        xs.sort()
        ys.sort()
        zs.sort()

        @inline(__always)
        func median(_ arr: [Float]) -> Float {
            let mid = arr.count / 2
            if arr.count % 2 == 1 { return arr[mid] }
            return 0.5 * (arr[mid - 1] + arr[mid])
        }

        let cx = median(xs)
        let cy = median(ys)
        let cz = median(zs)

        switch mode {
        case .none:
            return SIMD3<Float>(repeating: 0)
        case .recenterXY:
            return SIMD3<Float>(cx, cy, 0)
        case .recenterXYZ:
            return SIMD3<Float>(cx, cy, cz)
        }
    }
}
