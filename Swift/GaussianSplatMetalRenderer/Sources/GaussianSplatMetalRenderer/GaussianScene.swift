import Foundation
import Metal

public struct GaussianScene {
    public let count: Int
    public let means: MTLBuffer // packed_float3
    public let scales: MTLBuffer // packed_float3
    public let colorsLinear: MTLBuffer // packed_float3
    public let opacities: MTLBuffer // float

    public init(count: Int, means: MTLBuffer, scales: MTLBuffer, colorsLinear: MTLBuffer, opacities: MTLBuffer) {
        self.count = count
        self.means = means
        self.scales = scales
        self.colorsLinear = colorsLinear
        self.opacities = opacities
    }
}

