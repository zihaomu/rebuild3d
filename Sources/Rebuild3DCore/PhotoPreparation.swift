import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct ImagePreparation: Codable, Equatable, Sendable {
    public var version: Int = 1
    public var operation: String = "original-file"
    public var inputWidth: Int
    public var inputHeight: Int
    public var inputOrientation: Int
    public var inputSHA256: String
    public var inputTypeIdentifier: String?
    // Row-major 3x3 transform from original encoded pixels to staged encoded pixels.
    // EXIF display orientation is deliberately separate from this transform.
    public var originalToInputPixels: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    public var colorPolicy: String = "preserve-original-profile-and-auxiliary-data"
}

public struct PreparedPhoto: Codable, Equatable, Sendable {
    public var photo: PhotoRecord
    public var stagedPath: String
    public var preparation: ImagePreparation
}

public struct ReconstructionInputSnapshot: Codable, Equatable, Sendable {
    public var formatVersion: Int = 1
    public var runID: UUID
    public var projectID: UUID
    public var createdAt: Date
    public var settings: ReconstructionSettings
    public var inputs: [PreparedPhoto]
    public var relativePath: String { "runs/\(runID.uuidString)/inputs.json" }
}

public enum PhotoPreparation {
    /// Preserve originals. Prepare TIFF as PNG because folder-based Object Capture omits TIFF files.
    public static func stageInputs(project: Project, runID: UUID) throws -> ReconstructionInputSnapshot {
        let inputPath = "cache/\(runID.uuidString)/input"
        let input = try ProjectStore.resolve(inputPath, in: project.directory)
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        var records: [PreparedPhoto] = []
        do {
            for var photo in project.manifest.photos {
                let source = try ProjectStore.resolve(photo.imagePath, in: project.directory)
                let actual = try autoreleasepool { try PhotoInspector.inspect(source) }
                if let expected = photo.metadata?.contentSHA256, expected != actual.metadata.contentSHA256 {
                    throw ProjectError.invalid("The original photo \(photo.originalName) has changed. Remove and import it again before reconstructing.")
                }
                photo.metadata = actual.metadata
                photo.pixelWidth = actual.pixelWidth
                photo.pixelHeight = actual.pixelHeight
                photo.orientation = actual.orientation
                if actual.metadata.typeIdentifier == UTType.tiff.identifier {
                    let prepared = try autoreleasepool {
                        try prepareTIFF(photo: photo, source: source, inputPath: inputPath, directory: project.directory)
                    }
                    records.append(prepared)
                    continue
                }
                let stagedPath = "\(inputPath)/\(source.lastPathComponent)"
                let target = try ProjectStore.resolve(stagedPath, in: project.directory)
                do { try FileManager.default.linkItem(at: source, to: target) }
                catch { try FileManager.default.copyItem(at: source, to: target) }
                var preparation = ImagePreparation(inputWidth: photo.pixelWidth, inputHeight: photo.pixelHeight,
                    inputOrientation: photo.orientation, inputSHA256: actual.metadata.contentSHA256)
                preparation.inputTypeIdentifier = actual.metadata.typeIdentifier
                records.append(PreparedPhoto(photo: photo, stagedPath: stagedPath, preparation: preparation))
            }
            return ReconstructionInputSnapshot(runID: runID, projectID: project.manifest.id, createdAt: Date(),
                settings: project.manifest.settings, inputs: records)
        } catch {
            try? FileManager.default.removeItem(at: input)
            throw error
        }
    }

    private static func prepareTIFF(photo: PhotoRecord, source: URL, inputPath: String, directory: URL) throws -> PreparedPhoto {
        guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil), CGImageSourceGetCount(imageSource) == 1 else {
            throw ProjectError.invalid("\(photo.originalName) must contain exactly one TIFF image.")
        }
        let pixels = try PhotoPreviewDecoder.image(from: imageSource, maxPixelSize: max(photo.pixelWidth, photo.pixelHeight))
        let stagedPath = "\(inputPath)/\(photo.id.uuidString).png"
        let destinationURL = try ProjectStore.resolve(stagedPath, in: directory)
        guard let encoder = CGImageDestinationCreateWithURL(destinationURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw ProjectError.invalid("A working image could not be created for \(photo.originalName).")
        }
        CGImageDestinationAddImage(encoder, pixels, [kCGImagePropertyOrientation: 1] as CFDictionary)
        guard CGImageDestinationFinalize(encoder) else {
            throw ProjectError.invalid("A working image could not be saved for \(photo.originalName). Check project permissions and disk space.")
        }
        var recipe = ImagePreparation(inputWidth: pixels.width, inputHeight: pixels.height, inputOrientation: 1,
                                      inputSHA256: try PhotoInspector.contentSHA256(at: destinationURL))
        recipe.version = 2
        recipe.operation = "tiff-to-png"
        recipe.inputTypeIdentifier = UTType.png.identifier
        recipe.originalToInputPixels = try PhotoPixelTransform.oriented(width: photo.pixelWidth, height: photo.pixelHeight, orientation: photo.orientation)
        recipe.colorPolicy = "sdr-srgb-8bit"
        return PreparedPhoto(photo: photo, stagedPath: stagedPath, preparation: recipe)
    }
}
