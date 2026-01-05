import Foundation
import simd

public enum GaussianSplatCompositingMode: Equatable, Sendable {
    /// Weighted blended order-independent transparency (fast, deterministic, approximate).
    case weightedOIT

    /// Approximate back-to-front compositing by binning Gaussians into a fixed number of depth bins.
    /// This can reduce OIT halos at the cost of extra GPU work and some nondeterminism due to atomics.
    case depthBinnedAlpha(binCount: Int)

    public static let defaultDepthBinCount: Int = 64
}

public enum GaussianSplatToneMap: String, CaseIterable, Sendable {
    case none
    case reinhard
    case aces
}

public enum GaussianSplatDebugView: String, CaseIterable, Sendable {
    case none
    case alpha
    case depth
    case disparity
    case radius
}

public enum GaussianSplatSceneNormalization: String, CaseIterable, Sendable {
    /// No visualization transform.
    case none

    /// Subtract a robust (median) XY center (keeps Z unchanged).
    case recenterXY

    /// Subtract a robust (median) XYZ center.
    case recenterXYZ
}

public enum GaussianSplatSceneScale: String, CaseIterable, Sendable {
    case none

    /// Uniformly scale the scene so a robust radius becomes ~1.
    case unitRadius
}

public struct GaussianSplatRenderOptions: Equatable, Sendable {
    public var compositing: GaussianSplatCompositingMode = .weightedOIT

    /// Supersampling factor (render at `scale * size`, then downsample).
    public var renderScale: Float = 1.0

    public var toneMap: GaussianSplatToneMap = .none
    public var exposureEV: Float = 0.0
    public var saturation: Float = 1.0
    public var contrast: Float = 1.0

    public var debugView: GaussianSplatDebugView = .none

    /// For depth/disparity debug: (near, far) range in meters.
    public var debugDepthRange: SIMD2<Float>? = nil

    /// Culls splats with camera-space `z <= nearClipZ`.
    public var nearClipZ: Float = 1e-2

    /// Culls splats with `opacity < opacityThreshold` (after sigmoid).
    public var opacityThreshold: Float = 0.0

    /// Adds a small value to the 2D covariance diagonal for low-pass filtering.
    public var lowPassEps2D: Float = 0.0

    public var minRadiusPx: Float = 1.0
    public var maxRadiusPx: Float = 160.0

    /// Visualization-only normalization to improve framing for outlier-heavy scenes.
    public var normalization: GaussianSplatSceneNormalization = .none
    public var normalizationScale: GaussianSplatSceneScale = .none

    /// Scene center estimation parameters (used when `normalization != .none`).
    public var normalizationSampleCount: Int = 65536
    public var normalizationOpacityThreshold: Float = 0.01

    public init() {}
}
