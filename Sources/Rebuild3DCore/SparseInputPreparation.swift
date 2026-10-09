import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

/// Input identity is supplied by the task, never inferred from a filename or its order.
public struct SparseSourcePhoto: Codable, Sendable, Equatable {
    public let id: UUID
    public let name: String
    public let sourcePath: String
    public let sourceSHA256: String

    public init(id: UUID, name: String, sourcePath: String, sourceSHA256: String) {
        self.id = id; self.name = name; self.sourcePath = sourcePath; self.sourceSHA256 = sourceSHA256
    }
}

public enum SparseInputPreparation {
    /// Produces oriented sRGB working images and automatic foreground candidates.
    /// Masks are predictions. The downstream geometric stage must check their cross-view consistency.
    public static func prepare(_ photos: [SparseSourcePhoto], at output: URL,
                               progress: (Int, Int, String) -> Void = { _, _, _ in }) throws {
        guard photos.count >= 3, Set(photos.map(\.id)).count == photos.count,
              Set(photos.map(\.sourceSHA256)).count == photos.count else {
            throw ProjectError.invalid("At least three distinct, identified photos are required.")
        }
        let fm = FileManager.default
        guard !fm.fileExists(atPath: output.path) else { throw ProjectError.invalid("Preparation output must be new.") }
        for folder in ["images", "masks", "images-full", "masks-full", "labels"] {
            try fm.createDirectory(at: output.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
        let context = CIContext(options: [.cacheIntermediates: false])
        var records: [[String: Any]] = []
        for (index, photo) in photos.enumerated() {
            let record = try autoreleasepool { () throws -> [String: Any] in
                let source = URL(fileURLWithPath: photo.sourcePath)
                let info = try PhotoInspector.inspect(source)
                guard info.metadata.contentSHA256 == photo.sourceSHA256 else {
                    throw ProjectError.invalid("Photo changed before preparation: \(photo.name)")
                }
                let stem = photo.id.uuidString
                let full = try PhotoPreviewDecoder.image(at: source, maxPixelSize: max(info.pixelWidth, info.pixelHeight))
                let working = try PhotoPreviewDecoder.image(at: source, maxPixelSize: 1600)
                let fullPath = "images-full/\(stem).png", workPath = "images/\(stem).jpg"
                let maskPath = "masks/\(stem).png", fullMaskPath = "masks-full/\(stem).png"
                try write(full, to: output.appendingPathComponent(fullPath), type: .png)
                try write(working, to: output.appendingPathComponent(workPath), type: .jpeg)
                // Vision sees the correctly oriented image. No photo-specific polygons or prompts.
                let request = VNGenerateForegroundInstanceMaskRequest()
                let handler = VNImageRequestHandler(cgImage: full, orientation: .up)
                try handler.perform([request])
                guard let observation = request.results?.first, !observation.allInstances.isEmpty else {
                    throw ProjectError.invalid("No foreground object found in \(photo.name).")
                }
                let labels = observation.instanceMask
                CVPixelBufferLockBaseAddress(labels, .readOnly)
                let width = CVPixelBufferGetWidth(labels), height = CVPixelBufferGetHeight(labels)
                let stride = CVPixelBufferGetBytesPerRow(labels)
                guard let base = CVPixelBufferGetBaseAddress(labels) else {
                    CVPixelBufferUnlockBaseAddress(labels, .readOnly)
                    throw ProjectError.invalid("Foreground labels are unavailable.")
                }
                let values = base.assumingMemoryBound(to: UInt8.self)
                var scores: [Int: Double] = [:], areas: [Int: Int] = [:]
                for y in 0..<height {
                    for x in 0..<width {
                        let label = Int(values[y * stride + x])
                        guard label > 0 else { continue }
                        let nx = (Double(x) + 0.5) / Double(width) - 0.5
                        let ny = (Double(y) + 0.5) / Double(height) - 0.5
                        scores[label, default: 0] += exp(-(nx * nx + ny * ny) / 0.08)
                        areas[label, default: 0] += 1
                    }
                }
                CVPixelBufferUnlockBaseAddress(labels, .readOnly)
                // Stable tie-breaking. This is an automatic initial proposal, not a truth label.
                guard let selected = scores.keys.sorted().max(by: { scores[$0]! < scores[$1]! }) else {
                    throw ProjectError.invalid("No foreground object found in \(photo.name).")
                }
                let labelImage = CIImage(cvPixelBuffer: labels)
                if let cgLabels = context.createCGImage(labelImage, from: labelImage.extent,
                                                        format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()) {
                    try write(cgLabels, to: output.appendingPathComponent("labels/\(stem).png"), type: .png)
                }
                let buffer = try observation.generateScaledMaskForImage(forInstances: IndexSet(integer: selected), from: handler)
                let mask = CIImage(cvPixelBuffer: buffer)
                guard let fullMask = context.createCGImage(mask, from: mask.extent,
                    format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()) else {
                    throw ProjectError.invalid("Could not encode foreground mask.")
                }
                try write(fullMask, to: output.appendingPathComponent(fullMaskPath), type: .png)
                let sx = Double(working.width) / Double(full.width), sy = Double(working.height) / Double(full.height)
                let small = mask.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
                guard let workMask = context.createCGImage(small,
                    from: CGRect(x: 0, y: 0, width: working.width, height: working.height),
                    format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()) else {
                    throw ProjectError.invalid("Could not resize foreground mask.")
                }
                try write(workMask, to: output.appendingPathComponent(maskPath), type: .png)
                let orientation = try PhotoPixelTransform.oriented(width: info.pixelWidth,
                    height: info.pixelHeight, orientation: info.orientation)
                var transform = orientation
                for column in 0..<3 { transform[column] *= sx; transform[column + 3] *= sy }
                transform[2] += (sx - 1) / 2; transform[5] += (sy - 1) / 2
                guard try PhotoInspector.contentSHA256(at: source) == photo.sourceSHA256 else {
                    throw ProjectError.invalid("Photo changed during preparation: \(photo.name)")
                }
                return ["id": photo.id.uuidString, "name": photo.name, "sourcePath": source.path,
                    "sourceSHA256": photo.sourceSHA256, "encodedWidth": info.pixelWidth,
                    "encodedHeight": info.pixelHeight, "encodedOrientation": info.orientation,
                    "uprightWidth": full.width, "uprightHeight": full.height,
                    "workingWidth": working.width, "workingHeight": working.height,
                    "image": workPath, "mask": maskPath, "fullImage": fullPath, "fullMask": fullMaskPath,
                    "encodedToUprightPixels": orientation, "encodedToWorkingPixels": transform,
                    "workingSHA256": try PhotoInspector.contentSHA256(at: output.appendingPathComponent(workPath)),
                    "maskSHA256": try PhotoInspector.contentSHA256(at: output.appendingPathComponent(maskPath)),
                    "fullImageSHA256": try PhotoInspector.contentSHA256(at: output.appendingPathComponent(fullPath)),
                    "fullMaskSHA256": try PhotoInspector.contentSHA256(at: output.appendingPathComponent(fullMaskPath)),
                    "maskMethod": "Vision foreground instances, center-weighted automatic proposal",
                    "maskRevision": request.revision, "selectedInstance": selected,
                    "instanceCandidates": scores.keys.sorted().map { ["label": $0, "centerScore": scores[$0]!, "pixels": areas[$0]!] },
                    "manualInterventions": 0, "crossViewConsistency": "pending",
                    "colorPolicy": "SDR sRGB 8-bit; lossless full PNG, JPEG working image; original untouched"]
            }
            records.append(record)
            progress(index + 1, photos.count, photo.name)
            context.clearCaches()
        }
        let dataset: [String: Any] = ["formatVersion": 2, "kind": "automatic-sparse-inputs",
            "records": records, "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "pixelConvention": "Top-left origin; integer pixel centers; half-pixel resize offsets."]
        try JSONSerialization.data(withJSONObject: dataset, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("dataset.json"), options: .atomic)
    }

    private static func write(_ image: CGImage, to url: URL, type: UTType) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw ProjectError.invalid("Cannot create image: \(url.lastPathComponent)")
        }
        CGImageDestinationAddImage(destination, image,
            [kCGImagePropertyOrientation: 1, kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw ProjectError.invalid("Image encoding failed.") }
    }
}
