import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct PhotoMetadata: Codable, Equatable, Sendable {
    public var typeIdentifier: String
    public var primaryImageIndex: Int
    public var contentSHA256: String
    public var byteCount: Int
    public var colorProfile: String?
    public var bitsPerComponent: Int?
    public var cameraMake: String?
    public var cameraModel: String?
    public var focalLengthMillimeters: Double?
    // Preserve the source EXIF string; do not invent a timezone.
    public var captureDate: String?
}

public struct PhotoInspection: Sendable {
    public var pixelWidth: Int
    public var pixelHeight: Int
    public var orientation: Int
    public var metadata: PhotoMetadata
}

public enum PhotoInspector {
    public static func contentSHA256(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func inspect(_ url: URL) throws -> PhotoInspection {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let identifier = CGImageSourceGetType(source) as String? else {
            throw ProjectError.invalid("The photo cannot be decoded.")
        }
        let index = CGImageSourceGetPrimaryImageIndex(source)
        guard index < CGImageSourceGetCount(source),
              let props = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any],
              let width = props[kCGImagePropertyPixelWidth] as? Int,
              let height = props[kCGImagePropertyPixelHeight] as? Int, width > 0, height > 0 else {
            throw ProjectError.invalid("The photo has no readable primary image.")
        }
        let orientation = props[kCGImagePropertyOrientation] as? Int ?? 1
        guard (1...8).contains(orientation) else { throw ProjectError.invalid("The photo has an invalid orientation.") }
        let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]
        let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
        let metadata = PhotoMetadata(typeIdentifier: identifier, primaryImageIndex: index,
            contentSHA256: try contentSHA256(at: url), byteCount: try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0,
            colorProfile: props[kCGImagePropertyProfileName] as? String, bitsPerComponent: props[kCGImagePropertyDepth] as? Int,
            cameraMake: tiff[kCGImagePropertyTIFFMake] as? String, cameraModel: tiff[kCGImagePropertyTIFFModel] as? String,
            focalLengthMillimeters: exif[kCGImagePropertyExifFocalLength] as? Double,
            captureDate: exif[kCGImagePropertyExifDateTimeOriginal] as? String)
        return PhotoInspection(pixelWidth: width, pixelHeight: height, orientation: orientation, metadata: metadata)
    }
}
