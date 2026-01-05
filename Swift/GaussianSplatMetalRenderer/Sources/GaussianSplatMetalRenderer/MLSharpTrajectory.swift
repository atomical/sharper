import Foundation
import simd

public struct MLSharpTrajectoryParams {
    public enum Kind: String, CaseIterable {
        case rotateForward = "rotate_forward"
        case rotate = "rotate"
        case swipe = "swipe"
        case shake = "shake"
    }

    public var kind: Kind = .rotateForward
    public var maxDisparity: Float = 0.08
    public var maxZoom: Float = 0.15
    public var distanceM: Float = 0.0
    public var numSteps: Int = 60
    public var numRepeats: Int = 1
    public var focusQuantile: Float = 0.10
    public var minDepthQuantile: Float = 0.001
    public var minDepthFocusM: Float = 2.0
    public var opacityThreshold: Float = 0.01
    public var sampleCount: Int = 65536

    public init() {}
}

public struct MLSharpDepthRange {
    public var min: Float
    public var focus: Float
    public var max: Float

    public init(min: Float, focus: Float, max: Float) {
        self.min = min
        self.focus = focus
        self.max = max
    }
}

public enum MLSharpTrajectory {
    public static func depthQuantiles(
        scene: GaussianScene,
        sampleCount: Int = 65536,
        opacityThreshold: Float = 0.01,
        qNear: Float = 0.001,
        qFocus: Float = 0.10,
        qFar: Float = 0.999
    ) -> MLSharpDepthRange {
        let count = max(scene.count, 0)
        guard count > 0 else { return MLSharpDepthRange(min: 2, focus: 2, max: 2) }

        let stride = max(1, count / max(sampleCount, 1))
        let meanPtr = scene.means.contents().bindMemory(to: Float.self, capacity: count * 3)
        let opaPtr = scene.opacities.contents().bindMemory(to: Float.self, capacity: count)

        var zs: [Float] = []
        zs.reserveCapacity(min(sampleCount, count))

        var idx = 0
        while idx < count {
            let opa = opaPtr[idx]
            if opa.isFinite, opa >= opacityThreshold {
                let z = meanPtr[idx * 3 + 2]
                if z.isFinite, z > 1e-6 { zs.append(z) }
            }
            idx += stride
        }

        if zs.count < 1024 {
            zs.removeAll(keepingCapacity: true)
            idx = 0
            while idx < count {
                let z = meanPtr[idx * 3 + 2]
                if z.isFinite, z > 1e-6 { zs.append(z) }
                idx += stride
            }
        }

        guard zs.count >= 16 else { return MLSharpDepthRange(min: 2, focus: 2, max: 2) }
        zs.sort()

        @inline(__always)
        func quantile(_ arr: [Float], _ q: Float) -> Float {
            let qq = min(max(q, 0), 1)
            let pos = qq * Float(arr.count - 1)
            let i0 = Int(pos.rounded(.down))
            let i1 = min(i0 + 1, arr.count - 1)
            let t = pos - Float(i0)
            return arr[i0] * (1 - t) + arr[i1] * t
        }

        let dMin = quantile(zs, qNear)
        let dFocus = quantile(zs, qFocus)
        let dMax = quantile(zs, qFar)
        return MLSharpDepthRange(min: dMin, focus: dFocus, max: dMax)
    }

    public static func scaleIntrinsics(
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        srcWidth: Float,
        srcHeight: Float,
        dstWidth: Float,
        dstHeight: Float
    ) -> (fx: Float, fy: Float, cx: Float, cy: Float) {
        guard srcWidth > 0, srcHeight > 0 else { return (fx, fy, cx, cy) }
        return (
            fx * dstWidth / srcWidth,
            fy * dstHeight / srcHeight,
            cx * dstWidth / srcWidth,
            cy * dstHeight / srcHeight
        )
    }

    /// Generate a ml-sharp-like camera sequence (default: `rotate_forward`).
    ///
    /// Coordinate system:
    /// - Scene points are in camera/world coordinates where the original view is identity extrinsics.
    /// - Camera moves near z=0 and looks towards +Z (a focus point on the z-axis).
    public static func makeCameras(
        scene: GaussianScene,
        sourceImageWidth: Int,
        sourceImageHeight: Int,
        intrinsicFx: Float,
        intrinsicFy: Float,
        intrinsicCx: Float,
        intrinsicCy: Float,
        renderWidth: Int,
        renderHeight: Int,
        params: MLSharpTrajectoryParams = MLSharpTrajectoryParams()
    ) -> (cameras: [PinholeCamera], depth: MLSharpDepthRange) {
        let depth = depthQuantiles(
            scene: scene,
            sampleCount: params.sampleCount,
            opacityThreshold: params.opacityThreshold,
            qNear: params.minDepthQuantile,
            qFocus: params.focusQuantile,
            qFar: 0.999
        )

        let focusDepth = max(params.minDepthFocusM, depth.focus)
        let minDepth = max(1e-3, depth.min)

        let w = Float(max(sourceImageWidth, 1))
        let h = Float(max(sourceImageHeight, 1))
        let fPx = max(1e-6, intrinsicFx)

        let diagonal = sqrt((w / fPx) * (w / fPx) + (h / fPx) * (h / fPx))
        let maxLateral = params.maxDisparity * diagonal * minDepth
        let maxMedial = params.maxZoom * minDepth

        let scaled = scaleIntrinsics(
            fx: intrinsicFx,
            fy: intrinsicFy,
            cx: intrinsicCx,
            cy: intrinsicCy,
            srcWidth: w,
            srcHeight: h,
            dstWidth: Float(max(renderWidth, 1)),
            dstHeight: Float(max(renderHeight, 1))
        )

        let steps = max(params.numSteps * max(params.numRepeats, 1), 1)
        var cameras: [PinholeCamera] = []
        cameras.reserveCapacity(steps)

        let worldUp = SIMD3<Float>(0, -1, 0)
        let denom = Float(max(steps - 1, 1))

        for i in 0..<steps {
            let t = Float(i) / denom
            let ang = 2.0 * Float.pi * t

            let eye: SIMD3<Float>
            switch params.kind {
            case .swipe:
                eye = SIMD3<Float>(mix(-maxLateral, maxLateral, t), 0, params.distanceM)
            case .shake:
                // Half horizontal, half vertical.
                let half = max(1, steps / 2)
                if i < half {
                    let tt = Float(i) / Float(max(half - 1, 1))
                    eye = SIMD3<Float>(maxLateral * sin(2.0 * Float.pi * tt), 0, params.distanceM)
                } else {
                    let tt = Float(i - half) / Float(max((steps - half) - 1, 1))
                    eye = SIMD3<Float>(0, maxLateral * sin(2.0 * Float.pi * tt), params.distanceM)
                }
            case .rotate:
                eye = SIMD3<Float>(maxLateral * sin(ang), maxLateral * cos(ang), params.distanceM)
            case .rotateForward:
                eye = SIMD3<Float>(maxLateral * sin(ang), 0, params.distanceM + maxMedial * (1.0 - cos(ang)) * 0.5)
            }

            let target = SIMD3<Float>(0, 0, focusDepth)
            let view = PinholeCamera.lookAt(eye: eye, target: target, up: worldUp)
            cameras.append(PinholeCamera(viewMatrix: view, fx: scaled.fx, fy: scaled.fy, cx: scaled.cx, cy: scaled.cy))
        }

        return (cameras: cameras, depth: depth)
    }
}

@inline(__always)
private func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
    a * (1 - t) + b * t
}
