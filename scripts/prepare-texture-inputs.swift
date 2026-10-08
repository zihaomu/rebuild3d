// Build with Sources/Rebuild3DCore/*.swift. Does not modify source files.
import Foundation
import ImageIO
import UniformTypeIdentifiers

@main
struct PrepareTextureInputs {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { fatalError("DATASET NEW_OUTPUT") }
        let dataset = URL(fileURLWithPath: CommandLine.arguments[1])
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        let fm = FileManager.default
        guard !fm.fileExists(atPath: output.path) else { fatalError("Output must be new") }
        try fm.createDirectory(at: output.appendingPathComponent("images"), withIntermediateDirectories: true)
        try fm.createDirectory(at: output.appendingPathComponent("masks"), withIntermediateDirectories: true)
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: dataset.appendingPathComponent("dataset.json"))) as! [String: Any]
        var records: [[String: Any]] = []
        for var record in doc["records"] as! [[String: Any]] {
            try autoreleasepool {
                let source = URL(fileURLWithPath: record["sourcePath"] as! String)
                let expected = record["sourceSHA256"] as! String
                guard try PhotoInspector.contentSHA256(at: source) == expected else { fatalError("Original changed") }
                let image = try PhotoPreviewDecoder.image(at: source, maxPixelSize: 5712)
                guard image.width == record["uprightWidth"] as! Int,
                      image.height == record["uprightHeight"] as! Int else { fatalError("Dimensions changed") }
                let stem = source.deletingPathExtension().lastPathComponent
                let relative = "images/\(stem).png"
                let target = output.appendingPathComponent(relative)
                let writer = CGImageDestinationCreateWithURL(target as CFURL, UTType.png.identifier as CFString, 1, nil)!
                CGImageDestinationAddImage(writer, image, [kCGImagePropertyOrientation: 1] as CFDictionary)
                guard CGImageDestinationFinalize(writer) else { fatalError("PNG write failed") }
                let maskPath = "masks/\(stem).png"
                let maskSource = dataset.appendingPathComponent(record["fullMask"] as! String)
                try fm.copyItem(at: maskSource, to: output.appendingPathComponent(maskPath))
                record["image"] = relative
                record["mask"] = maskPath
                record["imageSHA256"] = try PhotoInspector.contentSHA256(at: target)
                record["maskSHA256"] = try PhotoInspector.contentSHA256(at: maskSource)
                record["colorPolicy"] = "ImageIO explicit SDR + sRGB 8-bit; one HEIC decode; lossless PNG encoding; original untouched"
                records.append(record)
                print("Prepared \(stem): \(image.width)x\(image.height)")
                fflush(nil)
            }
        }
        let result: [String: Any] = ["kind": "stage2-texture-inputs", "formatVersion": 1,
            "parentDatasetSHA256": try PhotoInspector.contentSHA256(at: dataset.appendingPathComponent("dataset.json")),
            "records": records, "orientationAppliedOnce": true]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted,.sortedKeys])
            .write(to: output.appendingPathComponent("dataset.json"), options: .atomic)
    }
}
