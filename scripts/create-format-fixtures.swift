// Controlled compatibility fixtures derived from public JPEG photos.
// These are not camera originals and cannot satisfy iPhone HEIC or HDR acceptance.
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: swift scripts/create-format-fixtures.swift JPEG_DIRECTORY NEW_OUTPUT_DIRECTORY\n", stderr)
    exit(2)
}
let source = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard !FileManager.default.fileExists(atPath: output.path) else {
    fputs("Output directory already exists.\n", stderr)
    exit(1)
}
let photos = try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
    .filter { ["jpg", "jpeg"].contains($0.pathExtension.lowercased()) }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
guard photos.count >= 3 else { fatalError("At least three JPEG inputs are required.") }
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let context = CIContext(options: [.cacheIntermediates: false])
let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
var records: [[String: Any]] = []
for (index, input) in photos.enumerated() {
    try autoreleasepool {
        let type: UTType = [UTType.jpeg, .png, .tiff][index % 3]
        let orientation = type == .tiff ? 6 : 1
        let target = output.appendingPathComponent(String(format: "%03d", index) + "." + type.preferredFilenameExtension!)
        if type == .jpeg {
            try FileManager.default.copyItem(at: input, to: target)
        } else {
            guard let imageSource = CGImageSourceCreateWithURL(input as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(imageSource, CGImageSourceGetPrimaryImageIndex(imageSource), nil),
                  let encoder = CGImageDestinationCreateWithURL(target as CFURL, type.identifier as CFString, 1, nil) else {
                fatalError("Cannot decode or encode \(input.lastPathComponent).")
            }
            // A counterclockwise pixel rotation plus EXIF 6 must display like the original portrait.
            let oriented = type == .tiff ? CIImage(cgImage: image).oriented(.left) : CIImage(cgImage: image)
            guard let pixels = context.createCGImage(oriented, from: oriented.extent, format: .RGBA8, colorSpace: colorSpace) else {
                fatalError("Cannot prepare fixture pixels.")
            }
            CGImageDestinationAddImage(encoder, pixels, [kCGImagePropertyOrientation: orientation] as CFDictionary)
            guard CGImageDestinationFinalize(encoder) else { fatalError("Cannot write fixture.") }
        }
        records.append([
            "sourceName": input.lastPathComponent, "fixtureName": target.lastPathComponent,
            "typeIdentifier": type.identifier, "expectedOrientation": orientation,
            "sourceSHA256": SHA256.hash(data: try Data(contentsOf: input)).map { String(format: "%02x", $0) }.joined(),
            "fixtureSHA256": SHA256.hash(data: try Data(contentsOf: target)).map { String(format: "%02x", $0) }.joined()
        ])
    }
}
let report: [String: Any] = [
    "kind": "derived-format-fixtures", "formatVersion": 1,
    "purpose": "JPEG/PNG/TIFF and EXIF orientation compatibility; not original HEIC, HDR, or higher-resolution acceptance",
    "records": records
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    .write(to: output.appendingPathComponent("fixtures.json"), options: .atomic)
print("Created \(records.count) controlled fixtures with JPEG/PNG/TIFF, including rotated TIFF pixels and EXIF orientation 6.")
