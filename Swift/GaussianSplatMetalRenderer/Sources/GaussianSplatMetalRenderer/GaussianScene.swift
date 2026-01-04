import Foundation
import Metal

public struct GaussianScene {
    public let count: Int
    public let means: MTLBuffer // packed_float3
    public let quaternions: MTLBuffer // packed_float4 (wxyz)
    public let scales: MTLBuffer // packed_float3
    public let colorsLinear: MTLBuffer // packed_float3
    public let opacities: MTLBuffer // float

    public init(count: Int, means: MTLBuffer, quaternions: MTLBuffer, scales: MTLBuffer, colorsLinear: MTLBuffer, opacities: MTLBuffer) {
        self.count = count
        self.means = means
        self.quaternions = quaternions
        self.scales = scales
        self.colorsLinear = colorsLinear
        self.opacities = opacities
    }
}
