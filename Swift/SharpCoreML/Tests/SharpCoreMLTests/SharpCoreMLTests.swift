import XCTest
@testable import SharpCoreML

final class SharpCoreMLTests: XCTestCase {
    func testFocalLengthPxMatchesFormula() {
        let f = SharpPreprocessor.focalLengthPx(focalLengthMM: 30.0, width: 640, height: 480)
        // sqrt(640^2+480^2)=800, sqrt(36^2+24^2)=43.266...
        XCTAssertEqual(f, Float(30.0 * 800.0 / (36.0 * 36.0 + 24.0 * 24.0).squareRoot()), accuracy: 1e-3)
    }

    func testPLYOpacityLogitClampsAtZero() {
        let got = SharpPLYWriter.logit(0.0)
        XCTAssertTrue(got.isFinite)
        XCTAssertLessThan(got, -10.0)
    }

    func testPLYOpacityLogitClampsAtOne() {
        let hi = SharpPLYWriter.logit(1.0)
        XCTAssertTrue(hi.isFinite)
        XCTAssertGreaterThan(hi, 10.0)
    }
}
