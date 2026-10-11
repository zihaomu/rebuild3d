// Controlled JPEG fixtures for EXIF 1...8. These are derived test files, not originals.
// Compile with unchanged Sources/Rebuild3DCore/*.swift.
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

@main struct OrientationFixtures {
    static func write(_ image: CGImage, to url: URL, type: UTType, orientation: Int) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw ProjectError.invalid("Cannot create fixture destination")
        }
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation,
            kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw ProjectError.invalid("Cannot encode fixture") }
    }

    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            fputs("Usage: create-orientation-fixtures REGISTERED_D_PROJECT NEW_OUTPUT\n", stderr)
            exit(2)
        }
        let source = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let output = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        let fm = FileManager.default
        guard !fm.fileExists(atPath: output.path) else { throw ProjectError.invalid("Output must be new") }
        let project = try ProjectStore.open(source)
        guard project.manifest.photos.count == 12 else { throw ProjectError.invalid("Expected registered D12 source") }
        for name in ["photos", "upright-reference", "native-decoded"] {
            try fm.createDirectory(at: output.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        let context = CIContext(options: [.cacheIntermediates: false])
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let inverse = [1, 2, 3, 4, 5, 8, 7, 6]
        var records: [[String: Any]] = []
        for index in 0..<8 {
            try autoreleasepool {
                let sourceIndex = index * 12 / 8
                let photo = project.manifest.photos[sourceIndex]
                let original = try ProjectStore.resolve(photo.imagePath, in: source)
                let sourceSHA = try PhotoInspector.contentSHA256(at: original)
                guard sourceSHA == photo.metadata?.contentSHA256 else { throw ProjectError.invalid("Original changed") }
                let upright = try PhotoPreviewDecoder.image(at: original, maxPixelSize: 1600)
                let orientation = index + 1
                let transformed = CIImage(cgImage: upright).oriented(CGImagePropertyOrientation(rawValue: UInt32(inverse[index]))!)
                guard let encoded = context.createCGImage(transformed, from: transformed.extent, format: .RGBA8, colorSpace: space) else {
                    throw ProjectError.invalid("Cannot transform fixture pixels")
                }
                let name = String(format: "%02d", orientation) + " 方向 (" + photo.originalName + ").jpg"
                let fixture = output.appendingPathComponent("photos/" + name)
                let reference = output.appendingPathComponent("upright-reference/\(orientation).png")
                let decoded = output.appendingPathComponent("native-decoded/\(orientation).png")
                try write(upright, to: reference, type: .png, orientation: 1)
                try write(encoded, to: fixture, type: .jpeg, orientation: orientation)
                let inspection = try PhotoInspector.inspect(fixture)
                guard inspection.orientation == orientation,
                      inspection.pixelWidth == encoded.width, inspection.pixelHeight == encoded.height else {
                    throw ProjectError.invalid("Encoded EXIF orientation or dimensions changed")
                }
                let roundtrip = try PhotoPreviewDecoder.image(at: fixture, maxPixelSize: 1600)
                guard roundtrip.width == upright.width, roundtrip.height == upright.height else {
                    throw ProjectError.invalid("Oriented fixture frame differs")
                }
                try write(roundtrip, to: decoded, type: .png, orientation: 1)
                records.append([
                    "sourceIndex": sourceIndex, "sourcePhotoID": photo.id.uuidString, "originalName": photo.originalName,
                    "sourcePath": original.path, "sourceSHA256": sourceSHA,
                    "path": fixture.path, "sha256": inspection.metadata.contentSHA256,
                    "orientation": orientation, "inversePixelTransform": inverse[index],
                    "encodedSize": [encoded.width, encoded.height], "uprightSize": [upright.width, upright.height],
                    "uprightReference": reference.path, "uprightReferenceSHA256": try PhotoInspector.contentSHA256(at: reference),
                    "nativeDecoded": decoded.path, "nativeDecodedSHA256": try PhotoInspector.contentSHA256(at: decoded)
                ])
            }
            context.clearCaches()
        }
        let report: [String: Any] = ["status": "fixtures-created-pixel-comparison-pending", "sources": records,
            "scope": "Derived 1600px SDR JPEG fixtures with inverse pixel transform and EXIF 1...8; not camera originals or HDR acceptance",
            "pixelComparisonThresholds": ["meanAbsoluteRGBError": 3.0, "p99AbsoluteRGBError": 20.0],
            "createdAtUTC": ISO8601DateFormatter().string(from: Date())]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("fixtures.json"), options: .atomic)
        print("Created all eight EXIF fixtures; independent pixel comparison required.")
    }
}
