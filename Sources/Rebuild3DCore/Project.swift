import Foundation

public enum ReconstructionQuality: String, Codable, CaseIterable, Sendable {
    case reduced, medium
}

public struct ReconstructionSettings: Codable, Equatable, Sendable {
    public var quality: ReconstructionQuality = .reduced
    public var objectMasking: Bool = true
    public init() {}
}

public struct PhotoRecord: Codable, Identifiable, Equatable, Sendable {
    public var id: UUID
    public var originalName: String
    public var imagePath: String
    public var thumbnailPath: String
    public var pixelWidth: Int
    public var pixelHeight: Int
    public var orientation: Int
    // Optional when reading legacy projects whose original metadata could not be inspected.
    public var metadata: PhotoMetadata?
}

public struct ModelRecord: Codable, Equatable, Sendable {
    public var path: String
    public var createdAt: Date
    public var settings: ReconstructionSettings
    public var photoIDs: [UUID]
    public var runID: UUID?
    public var inputSnapshotPath: String?
    public var approximation: ApproximationReference? = nil
}

public struct ProjectManifest: Codable, Equatable, Sendable {
    // Older apps must not silently display an approximation without its provenance.
    public static let currentVersion = 3
    public var formatVersion = currentVersion
    public var id: UUID = UUID()
    public var name: String
    public var createdAt = Date()
    public var modifiedAt = Date()
    public var settings = ReconstructionSettings()
    public var photos: [PhotoRecord] = []
    public var model: ModelRecord?

    public init(name: String) { self.name = name }
}

public struct Project: Sendable {
    public var directory: URL
    public var manifest: ProjectManifest
    public var loadedFormatVersion: Int
    public init(directory: URL, manifest: ProjectManifest, loadedFormatVersion: Int = ProjectManifest.currentVersion) {
        self.directory = directory
        self.manifest = manifest
        self.loadedFormatVersion = loadedFormatVersion
    }
    public var needsMigrationSave: Bool { loadedFormatVersion < ProjectManifest.currentVersion }
    public var modelURL: URL? {
        manifest.model.flatMap { try? ProjectStore.resolve($0.path, in: directory) }
    }
    public var provenanceModelURL: URL? {
        manifest.model?.approximation.flatMap { try? ProjectStore.resolve($0.provenanceModelPath, in: directory) }
    }
    public var textureSourcesModelURL: URL? {
        guard let reference = manifest.model?.approximation,
              let bundle = try? ProjectStore.resolve(reference.bundlePath, in: directory),
              let url = try? ProjectStore.resolve("texture-sources.usdz", in: bundle.deletingLastPathComponent()),
              FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }
    public var modelIsOutdated: Bool {
        guard let model = manifest.model else { return false }
        return Set(model.photoIDs) != Set(manifest.photos.map(\.id))
            || model.settings != manifest.settings
    }
}

public enum ProjectError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): message }
    }
}
