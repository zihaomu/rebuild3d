import Foundation

/// Maps integer pixel centers from encoded top-left coordinates to EXIF-oriented display pixels.
enum PhotoPixelTransform {
    static let identity: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    static func oriented(width: Int, height: Int, orientation: Int) throws -> [Double] {
        guard width > 0, height > 0 else { throw ProjectError.invalid("Invalid photo dimensions.") }
        let x = Double(width - 1), y = Double(height - 1)
        switch orientation {
        case 1: return identity
        case 2: return [-1, 0, x, 0, 1, 0, 0, 0, 1]
        case 3: return [-1, 0, x, 0, -1, y, 0, 0, 1]
        case 4: return [1, 0, 0, 0, -1, y, 0, 0, 1]
        case 5: return [0, 1, 0, 1, 0, 0, 0, 0, 1]
        case 6: return [0, -1, y, 1, 0, 0, 0, 0, 1]
        case 7: return [0, -1, y, -1, 0, x, 0, 0, 1]
        case 8: return [0, 1, 0, -1, 0, x, 0, 0, 1]
        default: throw ProjectError.invalid("Invalid photo orientation.")
        }
    }
}
