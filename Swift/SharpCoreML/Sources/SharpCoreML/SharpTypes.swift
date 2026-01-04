import CoreML
import Foundation

public struct SharpInputMetadata {
    public let imageWidth: Int
    public let imageHeight: Int
    public let focalLengthPx: Float
    public let disparityFactor: Float

    public var sx: Float { Float(imageWidth) / (2.0 * focalLengthPx) }
    public var sy: Float { Float(imageHeight) / (2.0 * focalLengthPx) }

    public init(imageWidth: Int, imageHeight: Int, focalLengthPx: Float) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.focalLengthPx = focalLengthPx
        self.disparityFactor = focalLengthPx / Float(imageWidth)
    }
}

public struct SharpRawOutputs {
    public let meanVectorsPre: MLMultiArray
    public let quaternionsPre: MLMultiArray
    public let singularValuesPre: MLMultiArray
    public let colorsLinearPre: MLMultiArray
    public let opacitiesPre: MLMultiArray

    public init(
        meanVectorsPre: MLMultiArray,
        quaternionsPre: MLMultiArray,
        singularValuesPre: MLMultiArray,
        colorsLinearPre: MLMultiArray,
        opacitiesPre: MLMultiArray
    ) {
        self.meanVectorsPre = meanVectorsPre
        self.quaternionsPre = quaternionsPre
        self.singularValuesPre = singularValuesPre
        self.colorsLinearPre = colorsLinearPre
        self.opacitiesPre = opacitiesPre
    }
}
