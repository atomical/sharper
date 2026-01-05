import Foundation
import simd

public struct GaussianSceneBounds {
    public var center: SIMD3<Float>
    public var radius: Float

    public init(center: SIMD3<Float>, radius: Float) {
        self.center = center
        self.radius = radius
    }

    /// Robust bounds estimation for camera framing.
    ///
    /// Many SHARP outputs contain a small fraction of extreme depth outliers; using min/max for bounds
    /// can yield a useless orbit trajectory (camera too far away or centered incorrectly).
    ///
    /// This estimator:
    /// - Samples the scene deterministically by fixed stride.
    /// - Filters out invalid values and very low-opacity points.
    /// - Uses per-axis quantiles to compute a robust bounding box.
    public static func estimate(
        scene: GaussianScene,
        sampleCount: Int = 65536,
        opacityThreshold: Float = 0.01,
        quantileLo: Float = 0.10,
        quantileHi: Float = 0.90
    ) -> GaussianSceneBounds {
        let count = max(scene.count, 0)
        guard count > 0 else {
            return GaussianSceneBounds(center: SIMD3<Float>(repeating: 0), radius: 1)
        }

        let stride = max(1, count / max(sampleCount, 1))
        let meanPtr = scene.means.contents().bindMemory(to: Float.self, capacity: count * 3)
        let opaPtr = scene.opacities.contents().bindMemory(to: Float.self, capacity: count)

        var xs: [Float] = []
        var ys: [Float] = []
        var zs: [Float] = []
        xs.reserveCapacity(min(sampleCount, count))
        ys.reserveCapacity(min(sampleCount, count))
        zs.reserveCapacity(min(sampleCount, count))

        var idx = 0
        while idx < count {
            let opa = opaPtr[idx]
            if opa >= opacityThreshold, opa.isFinite {
                let x = meanPtr[idx * 3 + 0]
                let y = meanPtr[idx * 3 + 1]
                let z = meanPtr[idx * 3 + 2]
                if x.isFinite, y.isFinite, z.isFinite, z > 1e-6 {
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                }
            }
            idx += stride
        }

        // Fallback if the opacity filter rejected too much.
        if xs.count < 1024 {
            xs.removeAll(keepingCapacity: true)
            ys.removeAll(keepingCapacity: true)
            zs.removeAll(keepingCapacity: true)

            idx = 0
            while idx < count {
                let x = meanPtr[idx * 3 + 0]
                let y = meanPtr[idx * 3 + 1]
                let z = meanPtr[idx * 3 + 2]
                if x.isFinite, y.isFinite, z.isFinite, z > 1e-6 {
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                }
                idx += stride
            }
        }

        guard xs.count >= 16 else {
            return GaussianSceneBounds(center: SIMD3<Float>(repeating: 0), radius: 1)
        }

        xs.sort()
        ys.sort()
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

        // Use a robust bbox for radius, but focus on the origin-plane scene:
        // ml-sharp's renderer places the camera around the origin and looks towards +Z.
        let xLo = quantile(xs, quantileLo)
        let xHi = quantile(xs, quantileHi)
        let yLo = quantile(ys, quantileLo)
        let yHi = quantile(ys, quantileHi)
        let zLo = quantile(zs, quantileLo)
        let zHi = quantile(zs, quantileHi)

        // Keep XY center at 0 for stable trajectories; focus depth around the near part of the scene.
        let zFocus = quantile(zs, 0.10)
        let center = SIMD3<Float>(0, 0, zFocus)
        let ext = SIMD3<Float>(xHi - xLo, yHi - yLo, zHi - zLo)
        let radius = max(0.5, simd_length(ext) * 0.75)
        return GaussianSceneBounds(center: center, radius: radius)
    }
}
