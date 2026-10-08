// Build with Sources/Rebuild3DCore/*.swift; produces reviewable research inputs only.
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

@main
struct PrepareSparseInputs {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            print("Usage: prepare-sparse-inputs INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY")
            exit(2)
        }
        let input = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let output = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        let fm = FileManager.default
        guard !fm.fileExists(atPath: output.path) else { throw ProjectError.invalid("Output must be new.") }
        let urls = try fm.contentsOfDirectory(at: input, includingPropertiesForKeys: [.isRegularFileKey])
            .filter { $0.pathExtension.lowercased() == "heic" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard urls.count == 7 else { throw ProjectError.invalid("This research dataset requires exactly seven HEIC files.") }
        for folder in ["images", "masks", "masks-full", "images-full"] {
            try fm.createDirectory(at: output.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
        let context = CIContext(options: [.cacheIntermediates: false])
        var records: [[String: Any]] = []
        for url in urls {
            let record = try autoreleasepool { () throws -> [String: Any] in
                let info = try PhotoInspector.inspect(url)
                let name = url.deletingPathExtension().lastPathComponent
                let full = try PhotoPreviewDecoder.image(at: url, maxPixelSize: max(info.pixelWidth, info.pixelHeight))
                let working = try PhotoPreviewDecoder.image(at: url, maxPixelSize: 1600)
                try write(full, output.appendingPathComponent("images-full/\(name).jpg"), type: .jpeg)
                try write(working, output.appendingPathComponent("images/\(name).jpg"), type: .jpeg)
                let request = VNGenerateForegroundInstanceMaskRequest()
                let handler = VNImageRequestHandler(cgImage: full, orientation: .up)
                try handler.perform([request])
                guard let observation = request.results?.first else { throw ProjectError.invalid("No foreground observation for \(name).") }
                let labels = observation.instanceMask
                CVPixelBufferLockBaseAddress(labels, .readOnly)
                let width = CVPixelBufferGetWidth(labels), height = CVPixelBufferGetHeight(labels)
                let stride = CVPixelBufferGetBytesPerRow(labels)
                let buffer = CVPixelBufferGetBaseAddress(labels)!.assumingMemoryBound(to: UInt8.self)
                var counts: [Int: Int] = [:]
                for y in Int(Double(height)*0.25)..<Int(Double(height)*0.85) {
                    for x in Int(Double(width)*0.40)..<Int(Double(width)*0.60) {
                        let label = Int(buffer[y*stride+x])
                        if label != 0 { counts[label, default: 0] += 1 }
                    }
                }
                CVPixelBufferUnlockBaseAddress(labels, .readOnly)
                guard let selected = counts.max(by: { $0.value < $1.value })?.key else {
                    throw ProjectError.invalid("No central foreground instance for \(name).")
                }
                let mask = try observation.generateScaledMaskForImage(forInstances: IndexSet(integer: selected), from: handler)
                let maskImage = CIImage(cvPixelBuffer: mask)
                guard let gray = context.createCGImage(maskImage, from: maskImage.extent, format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()) else {
                    throw ProjectError.invalid("Cannot encode mask for \(name).")
                }
                try write(gray, output.appendingPathComponent("masks-full/\(name).png"), type: .png)
                let sx = Double(working.width)/Double(full.width), sy = Double(working.height)/Double(full.height)
                let small = maskImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
                guard let reduced = context.createCGImage(small, from: CGRect(x: 0,y: 0,width: working.width,height: working.height), format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()) else {
                    throw ProjectError.invalid("Cannot resize mask for \(name).")
                }
                try write(reduced, output.appendingPathComponent("masks/\(name).png"), type: .png)
                let rotation = try PhotoPixelTransform.oriented(width: info.pixelWidth, height: info.pixelHeight, orientation: info.orientation)
                var transformed = rotation
                for i in 0..<3 { transformed[i] *= sx; transformed[i+3] *= sy }
                transformed[2] += (sx-1)/2; transformed[5] += (sy-1)/2
                let digest = try PhotoInspector.contentSHA256(at: url)
                guard digest == info.metadata.contentSHA256 else { throw ProjectError.invalid("Source changed during preparation.") }
                print("\(name): selected instance \(selected) / \(Array(observation.allInstances)); upright \(full.width)x\(full.height)")
                fflush(nil)
                return ["name": url.lastPathComponent, "sourcePath": url.path, "sourceSHA256": digest,
                    "encodedWidth": info.pixelWidth, "encodedHeight": info.pixelHeight, "encodedOrientation": info.orientation,
                    "uprightWidth": full.width, "uprightHeight": full.height, "workingWidth": working.width, "workingHeight": working.height,
                    "image": "images/\(name).jpg", "mask": "masks/\(name).png", "fullImage": "images-full/\(name).jpg", "fullMask": "masks-full/\(name).png",
                    "encodedToUprightPixels": rotation, "encodedToWorkingPixels": transformed,
                    "workingSHA256": try PhotoInspector.contentSHA256(at: output.appendingPathComponent("images/\(name).jpg")),
                    "maskSHA256": try PhotoInspector.contentSHA256(at: output.appendingPathComponent("masks/\(name).png")),
                    "maskMethod": "Vision VNGenerateForegroundInstanceMaskRequest; largest label in central strip", "maskRevision": request.revision,
                    "selectedInstance": selected, "allInstances": Array(observation.allInstances), "reviewStatus": "pending",
                    "colorPolicy": "SDR sRGB 8-bit; JPEG quality 1; original untouched"]
            }
            records.append(record)
            context.clearCaches()
        }
        let report: [String: Any] = ["formatVersion": 1, "kind": "seven-photo-research-inputs", "records": records,
            "os": ProcessInfo.processInfo.operatingSystemVersionString, "pixelConvention": "Top-left origin, integer pixel centers; affine resize includes half-pixel offset."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys]).write(to: output.appendingPathComponent("dataset.json"), options: .atomic)
    }

    static func write(_ image: CGImage, _ url: URL, type: UTType) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else { throw ProjectError.invalid("Cannot write \(url.path).") }
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: 1, kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw ProjectError.invalid("Image encoding failed.") }
    }
}
