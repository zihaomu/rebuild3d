import Foundation

public struct DraftSummary: Identifiable, Sendable {
    public var id: URL { directory }
    public var directory: URL
    public var name: String
    public var modifiedAt: Date
    public var photoCount: Int
}

public struct DraftProjectStore: Sendable {
    public let root: URL

    public init(root: URL? = nil) {
        self.root = root ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("org.rebuild3d.app/Drafts", isDirectory: true)
    }

    public func contains(_ directory: URL) -> Bool {
        let resolved = directory.resolvingSymlinksInPath().standardizedFileURL
        return resolved.deletingLastPathComponent().path == root.resolvingSymlinksInPath().standardizedFileURL.path
            && resolved.pathExtension == "rebuild3d"
    }

    public func create() throws -> Project {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var project = try ProjectStore.create(at: root.appendingPathComponent("\(UUID().uuidString).rebuild3d"))
        project.manifest.name = "Untitled"
        try ProjectStore.save(project)
        return project
    }

    public func list() throws -> [DraftSummary] {
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            .filter { contains($0) }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url.appendingPathComponent("project.json")),
                      let manifest = try? decoder.decode(ProjectManifest.self, from: data) else {
                    // Keep damaged drafts visible so they can be opened for a specific error; never delete them here.
                    return DraftSummary(directory: url, name: "Draft needs attention", modifiedAt: .distantPast, photoCount: 0)
                }
                return DraftSummary(directory: url, name: manifest.name, modifiedAt: manifest.modifiedAt, photoCount: manifest.photos.count)
            }
            .sorted { $0.modifiedAt > $1.modifiedAt }
    }

    /// Publish a validated copy, leaving the source draft until the caller installs the returned project.
    public func saveAs(_ project: Project, to destination: URL) throws -> Project {
        guard contains(project.directory) else { throw ProjectError.invalid("Only managed drafts can be saved to a new location.") }
        let target = destination.resolvingSymlinksInPath().standardizedFileURL
        let source = project.directory.resolvingSymlinksInPath().standardizedFileURL
        guard target != source, !target.path.hasPrefix(source.path + "/"), !contains(target),
              !FileManager.default.fileExists(atPath: target.path) else {
            throw ProjectError.invalid("Choose a new project location outside the draft. Existing projects will not be replaced.")
        }
        let staging = target.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).rebuild3d")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: source, to: staging)
        var copied = project
        copied.directory = staging
        copied.manifest.name = target.deletingPathExtension().lastPathComponent
        copied.manifest.modifiedAt = Date()
        try ProjectStore.save(copied)
        _ = try ProjectStore.open(staging)
        try FileManager.default.moveItem(at: staging, to: target)
        // The validated package has only relative references; relocating it does not require another write.
        copied.directory = target
        copied.loadedFormatVersion = ProjectManifest.currentVersion
        return copied
    }

    public func discard(_ directory: URL) throws {
        guard contains(directory) else { throw ProjectError.invalid("This is not a managed draft.") }
        try FileManager.default.removeItem(at: directory)
    }
}
