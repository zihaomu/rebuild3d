import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct ImportIssue: Sendable, Identifiable {
    public enum Kind: String, Sendable { case duplicate, unsupported, unreadable, ignored }
    public let id = UUID()
    public let filename: String
    public let reason: String
    public var kind: Kind = .unreadable
}

public struct ImportResult: Sendable {
    public var photos: [PhotoRecord] = []
    public var issues: [ImportIssue] = []
    public var summary: String {
        var parts = ["Added \(photos.count) photos"]
        for kind in [ImportIssue.Kind.duplicate, .unsupported, .unreadable, .ignored] {
            let count = issues.filter { $0.kind == kind }.count
            if count > 0 { parts.append("\(count) \(kind.rawValue)") }
        }
        return parts.joined(separator: " · ")
    }
}

public enum PhotoImporter {
    public static let supportedExtensions = Set(["jpg", "jpeg", "heic", "heif", "png", "tif", "tiff"])

    /// Decode and copy one image at a time. Folder import is deliberately nonrecursive.
    public static func importPhotos(from selections: [URL], into directory: URL,
                                   existingPhotos: [PhotoRecord]? = nil) throws -> ImportResult {
        let existing = try existingPhotos ?? ProjectStore.open(directory).manifest.photos
        var knownDigests = Set<String>()
        for photo in existing {
            if let digest = photo.metadata?.contentSHA256 { knownDigests.insert(digest) }
            else {
                let source = try ProjectStore.resolve(photo.imagePath, in: directory)
                knownDigests.insert(try PhotoInspector.contentSHA256(at: source))
            }
        }
        var result = ImportResult()
        var visited = Set<URL>()
        for selection in selections {
            let access = selection.startAccessingSecurityScopedResource()
            defer { if access { selection.stopAccessingSecurityScopedResource() } }
            let candidates: [URL]
            do {
                let isDirectory = try selection.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
                candidates = isDirectory
                    ? try FileManager.default.contentsOfDirectory(at: selection, includingPropertiesForKeys: [.isRegularFileKey, .isHiddenKey]).sorted { $0.lastPathComponent < $1.lastPathComponent }
                    : [selection]
            } catch {
                result.issues.append(.init(filename: selection.lastPathComponent, reason: error.localizedDescription))
                continue
            }
            for source in candidates {
                guard visited.insert(source.resolvingSymlinksInPath()).inserted else {
                    result.issues.append(.init(filename: source.lastPathComponent, reason: "This file was already selected.", kind: .duplicate))
                    continue
                }
                do {
                    let photo = try autoreleasepool { try importOne(source, into: directory, knownDigests: knownDigests) }
                    result.photos.append(photo)
                    if let digest = photo.metadata?.contentSHA256 { knownDigests.insert(digest) }
                } catch let issue as InputProblem {
                    result.issues.append(.init(filename: source.lastPathComponent, reason: issue.reason, kind: issue.kind))
                } catch {
                    result.issues.append(.init(filename: source.lastPathComponent, reason: error.localizedDescription))
                }
            }
        }
        return result
    }

    private struct InputProblem: Error {
        var kind: ImportIssue.Kind
        var reason: String
    }

    private static func importOne(_ source: URL, into directory: URL, knownDigests: Set<String>) throws -> PhotoRecord {
        let values = try source.resourceValues(forKeys: [.isRegularFileKey, .isHiddenKey])
        guard values.isRegularFile == true, values.isHidden != true, !source.lastPathComponent.hasPrefix(".") else {
            throw InputProblem(kind: .ignored, reason: "Hidden files and nested folders are not imported.")
        }
        guard let image = CGImageSourceCreateWithURL(source as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let identifier = CGImageSourceGetType(image) as String?, let type = UTType(identifier) else {
            let type = UTType(filenameExtension: source.pathExtension)
            if type?.conforms(to: .rawImage) == true || source.pathExtension.lowercased() == "dng" {
                throw InputProblem(kind: .unsupported, reason: "ProRAW and RAW photos are not supported yet. Other photos can still be added.")
            }
            if supportedExtensions.contains(source.pathExtension.lowercased()) || type?.conforms(to: .image) == true {
                throw InputProblem(kind: .unreadable, reason: "This photo cannot be decoded. Other photos can still be added.")
            }
            throw InputProblem(kind: .ignored, reason: "Video and auxiliary files are not used for reconstruction.")
        }
        let allowed = [UTType.jpeg, .heic, .heif, .png, .tiff]
        guard !type.conforms(to: .rawImage), allowed.contains(where: { type.conforms(to: $0) }) else {
            throw InputProblem(kind: .unsupported, reason: "This photo format is not supported yet. Use HEIC or JPEG photos.")
        }
        let isHEIF = type.conforms(to: .heif) || type.conforms(to: .heic)
        guard isHEIF || CGImageSourceGetCount(image) == 1 else {
            throw InputProblem(kind: .unsupported, reason: "Animated or multipage images are not supported.")
        }
        let digest = try PhotoInspector.contentSHA256(at: source)
        guard !knownDigests.contains(digest) else {
            throw InputProblem(kind: .duplicate, reason: "An identical photo is already in this project.")
        }
        let id = UUID()
        // The internal extension follows decoded content, while the original filename remains visible.
        let imagePath = "images/\(id.uuidString).\(type.preferredFilenameExtension ?? "image")"
        let thumbnailPath = "thumbnails/\(id.uuidString).jpg"
        let imageURL = try ProjectStore.resolve(imagePath, in: directory)
        let thumbnailURL = try ProjectStore.resolve(thumbnailPath, in: directory)
        do {
            // FileManager may leave a partial destination when copying runs out of space.
            try FileManager.default.copyItem(at: source, to: imageURL)
            let inspected = try PhotoInspector.inspect(imageURL)
            guard inspected.metadata.contentSHA256 == digest else { throw ProjectError.invalid("The photo changed while it was being imported. Please add it again.") }
            guard let copied = CGImageSourceCreateWithURL(imageURL as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else {
                throw ProjectError.invalid("The copied photo cannot be decoded.")
            }
            try writeThumbnail(copied, to: thumbnailURL)
            return PhotoRecord(id: id, originalName: source.lastPathComponent, imagePath: imagePath,
                thumbnailPath: thumbnailPath, pixelWidth: inspected.pixelWidth, pixelHeight: inspected.pixelHeight,
                orientation: inspected.orientation, metadata: inspected.metadata)
        } catch {
            try? FileManager.default.removeItem(at: imageURL)
            try? FileManager.default.removeItem(at: thumbnailURL)
            throw error
        }
    }

    public static func regenerateThumbnail(for photo: PhotoRecord, in directory: URL) throws {
        let url = try ProjectStore.resolve(photo.imagePath, in: directory)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw ProjectError.invalid("Cannot read \(photo.originalName).")
        }
        let destination = try ProjectStore.resolve(photo.thumbnailPath, in: directory)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try writeThumbnail(source, to: destination)
    }

    private static func writeThumbnail(_ image: CGImageSource, to url: URL) throws {
        let thumbnail = try PhotoPreviewDecoder.image(from: image, maxPixelSize: 320)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw ProjectError.invalid("Cannot decode this photo or write its thumbnail.")
        }
        CGImageDestinationAddImage(destination, thumbnail, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ProjectError.invalid("Cannot save the thumbnail. Check available disk space and project permissions.")
        }
    }
}
