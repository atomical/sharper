import Foundation

@inlinable
func require<E: Error>(_ condition: Bool, _ error: E) throws {
    guard condition else { throw error }
}

extension Optional {
    @inlinable
    func orThrow<E: Error>(_ error: E) throws -> Wrapped {
        guard let value = self else { throw error }
        return value
    }
}
