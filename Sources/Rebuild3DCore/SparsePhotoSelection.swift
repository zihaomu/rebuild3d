import CoreGraphics
import Foundation

struct SparsePhotoUse: Codable, Sendable, Equatable {
    let photoID: UUID
    let sourceSHA256: String
    let role: String
    let reason: String
}

enum SparsePhotoSelection {
    /// A deterministic spatial-color diversity sample. It is a resource policy, not calibrated camera recovery.
    static func select(_ photos: [SparseSourcePhoto], limit: Int = 12,
                       cancellation: GenerationCancellation) throws -> ([SparseSourcePhoto], [SparsePhotoUse]) {
        guard photos.count > limit else {
            return (photos, photos.map { SparsePhotoUse(photoID: $0.id, sourceSHA256: $0.sourceSHA256,
                role: "geometry-and-texture", reason: "Within sparse runtime budget; all photos used") })
        }
        let ordered = photos.sorted { $0.sourceSHA256 < $1.sourceSHA256 }
        let vectors = try ordered.map { photo in
            try cancellation.check()
            let image = try PhotoPreviewDecoder.image(at: URL(fileURLWithPath: photo.sourcePath), maxPixelSize: 256)
            guard let color = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 64,
                      space: color, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue), let bytes = context.data else {
                throw ProjectError.invalid("无法分析照片内容。")
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: 16, height: 16))
            let pixels = bytes.assumingMemoryBound(to: UInt8.self)
            return (0..<256).flatMap { i in (0..<3).map { c in Double(pixels[i * 4 + c]) / 255 } }
        }
        func distance(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) } }
        let mean = (0..<768).map { column in vectors.reduce(0) { $0 + $1[column] } / Double(vectors.count) }
        let first = vectors.indices.min { distance(vectors[$0], mean) < distance(vectors[$1], mean) }!
        var selected = [first]
        var nearest = vectors.map { distance($0, vectors[first]) }
        while selected.count < limit {
            try cancellation.check()
            let next = vectors.indices.filter { !selected.contains($0) }.max { nearest[$0] < nearest[$1] }!
            selected.append(next)
            nearest = vectors.indices.map { min(nearest[$0], distance(vectors[$0], vectors[next])) }
        }
        let result = selected.map { ordered[$0] }, ids = Set(result.map(\.id))
        let uses = photos.map { photo in
            SparsePhotoUse(photoID: photo.id, sourceSHA256: photo.sourceSHA256,
                role: ids.contains(photo.id) ? "geometry-and-texture" : "retained-original",
                reason: ids.contains(photo.id) ? "Selected by spatial sRGB diversity within 12-photo memory budget"
                    : "Outside sparse subset; original retained, no claimed geometry or texture contribution")
        }
        return (result, uses)
    }
}
