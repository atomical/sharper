import XCTest
@testable import GaussianSplatMetalRenderer

final class GaussianSplatMetalRendererTests: XCTestCase {
    func testLookAtMatrixHasW1() {
        let m = PinholeCamera.lookAt(eye: .zero, target: SIMD3<Float>(0, 0, 1))
        XCTAssertEqual(m.columns.3.w, 1)
    }
}

