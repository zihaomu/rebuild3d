// Read-only inventory of actual image content and metadata for reconstruction baselines.
import CryptoKit
import Foundation
import ImageIO

guard CommandLine.arguments.count == 2 else {
    print("Usage: swift scripts/inspect-photos.swift PHOTO_DIRECTORY")
    exit(2)
}
let folder = URL(fileURLWithPath: CommandLine.arguments[1])
let urls = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey])
var records: [[String: Any]] = []
for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
    guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
    autoreleasepool {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else { return }
        let index = CGImageSourceGetPrimaryImageIndex(source)
        guard let props = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any] else { return }
        let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]
        let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
        var record: [String: Any] = [
            "name": url.lastPathComponent, "type": CGImageSourceGetType(source) as String? ?? "unknown",
            "imageCount": CGImageSourceGetCount(source), "primaryImageIndex": index,
            "width": props[kCGImagePropertyPixelWidth] ?? NSNull(), "height": props[kCGImagePropertyPixelHeight] ?? NSNull(),
            "orientation": props[kCGImagePropertyOrientation] ?? 1,
            "profile": props[kCGImagePropertyProfileName] ?? NSNull(),
            "cameraModel": tiff[kCGImagePropertyTIFFModel] ?? NSNull(),
            "focalLength": exif[kCGImagePropertyExifFocalLength] ?? NSNull()
        ]
        if let data = try? Data(contentsOf: url, options: .mappedIfSafe) {
            record["bytes"] = data.count
            record["sha256"] = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        records.append(record)
    }
}
let output = try JSONSerialization.data(withJSONObject: ["count": records.count, "records": records], options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(output)
