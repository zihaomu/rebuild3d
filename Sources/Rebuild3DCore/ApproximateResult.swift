import Foundation

public struct ApproximationReference: Codable, Equatable, Sendable {
    public var bundlePath: String
    public var provenanceModelPath: String
}

public struct ApproximateResultBundle: Codable, Sendable {
    public struct SourcePhoto: Codable, Sendable {
        public var name: String
        public var sha256: String
    }
    public struct Artifact: Codable, Sendable {
        public var path: String
        public var sha256: String
        public var byteCount: Int
    }
    public var formatVersion: Int
    public var kind: String
    public var method: String
    public var createdAt: String
    public var sourcePhotos: [SourcePhoto]
    public var model: String
    public var provenanceModel: String
    public var artifacts: [Artifact]
    public var triangleCount: Int
    public var completionTriangleCount: Int
    public var limitations: [String]
}

public enum ApproximateResultStore {
    public static func load(_ directory: URL) throws -> ApproximateResultBundle {
        let info = try ProjectStore.resolve("bundle.json", in: directory)
        let bundle = try JSONDecoder().decode(ApproximateResultBundle.self, from: Data(contentsOf: info))
        guard bundle.formatVersion == 1, bundle.kind == "approximate", !bundle.method.isEmpty,
              !bundle.sourcePhotos.isEmpty, !bundle.limitations.isEmpty,
              bundle.triangleCount > 0, (0...bundle.triangleCount).contains(bundle.completionTriangleCount),
              Set(bundle.sourcePhotos.map(\.sha256)).count == bundle.sourcePhotos.count,
              Set(bundle.artifacts.map(\.path)).count == bundle.artifacts.count,
              !bundle.artifacts.contains(where: { $0.path == "bundle.json" }),
              bundle.model.hasSuffix(".usdz"), bundle.provenanceModel.hasSuffix(".usdz"),
              bundle.model != bundle.provenanceModel,
              bundle.artifacts.contains(where: { $0.path == bundle.model }),
              bundle.artifacts.contains(where: { $0.path == bundle.provenanceModel }) else {
            throw ProjectError.invalid("This approximate result has incomplete source or provenance records.")
        }
        for artifact in bundle.artifacts {
            let url = try ProjectStore.resolve(artifact.path, in: directory)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true, artifact.byteCount > 0, values.fileSize == artifact.byteCount,
                  try PhotoInspector.contentSHA256(at: url) == artifact.sha256 else {
                throw ProjectError.invalid("An approximate-result artifact is missing or changed: \(artifact.path)")
            }
        }
        return bundle
    }

    public static func importResult(from directory: URL, into project: Project) throws -> Project {
        let bundle = try load(directory)
        let expected = Set(bundle.sourcePhotos.map(\.sha256))
        let actual = try project.manifest.photos.map { photo in
            try PhotoInspector.contentSHA256(at: ProjectStore.resolve(photo.imagePath, in: project.directory))
        }
        guard actual.count == bundle.sourcePhotos.count, Set(actual) == expected else {
            throw ProjectError.invalid("This result belongs to different photos. Its original-photo hashes must exactly match the project.")
        }
        let runID = UUID()
        let runPath = "runs/\(runID.uuidString)"
        let root = try ProjectStore.resolve(runPath, in: project.directory)
        let researchPath = "\(runPath)/research"
        let research = try ProjectStore.resolve(researchPath, in: project.directory)
        let cache = try ProjectStore.resolve("cache/\(runID.uuidString)", in: project.directory)
        defer { try? FileManager.default.removeItem(at: cache) }
        do {
            let snapshot = try PhotoPreparation.stageInputs(project: project, runID: runID)
            var updated = project
            updated.manifest.photos = snapshot.inputs.map(\.photo)
            let snapshotPath = try ProjectStore.saveInputSnapshot(snapshot, in: project.directory)
            try copyBundle(bundle, from: directory, to: research)
            let reference = ApproximationReference(bundlePath: "\(researchPath)/bundle.json",
                                                    provenanceModelPath: "\(researchPath)/\(bundle.provenanceModel)")
            return try ProjectStore.commitModel(from: research.appendingPathComponent(bundle.model), to: updated,
                                                inputSnapshotPath: snapshotPath, approximation: reference)
        } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
    }

    static func validate(model: ModelRecord, projectID: UUID, directory: URL) throws {
        guard let reference = model.approximation, let runID = model.runID,
              let snapshotPath = model.inputSnapshotPath,
              reference.bundlePath == "runs/\(runID.uuidString)/research/bundle.json" else {
            throw ProjectError.invalid("The approximation is missing its immutable input and source record.")
        }
        let root = try ProjectStore.resolve(reference.bundlePath, in: directory).deletingLastPathComponent()
        let bundle = try load(root)
        let snapshot = try ProjectStore.loadInputSnapshot(snapshotPath, in: directory)
        guard snapshot.projectID == projectID,
              Set(snapshot.inputs.compactMap { $0.photo.metadata?.contentSHA256 }) == Set(bundle.sourcePhotos.map(\.sha256)),
              snapshot.inputs.count == bundle.sourcePhotos.count,
              reference.provenanceModelPath == "runs/\(runID.uuidString)/research/\(bundle.provenanceModel)",
              try PhotoInspector.contentSHA256(at: ProjectStore.resolve(model.path, in: directory)) ==
                bundle.artifacts.first(where: { $0.path == bundle.model })?.sha256 else {
            throw ProjectError.invalid("The approximation, source photos and provenance do not match.")
        }
    }

    private static func copyBundle(_ bundle: ApproximateResultBundle, from source: URL, to destination: URL) throws {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for path in ["bundle.json"] + bundle.artifacts.map(\.path) {
            let input = try ProjectStore.resolve(path, in: source)
            let output = try ProjectStore.resolve(path, in: destination)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: input, to: output)
        }
        _ = try load(destination)
    }

    /// Export a USDZ and a complete, re-importable companion result directory.
    /// Publish the model last; rollback both destinations if any write fails.
    static func export(_ project: Project, to destination: URL) throws {
        guard let record = project.manifest.model, let reference = record.approximation, let model = project.modelURL else {
            throw ProjectError.invalid("No approximate model to export.")
        }
        try validate(model: record, projectID: project.manifest.id, directory: project.directory)
        let root = project.directory.resolvingSymlinksInPath().standardizedFileURL.path
        let companion = destination.deletingPathExtension().appendingPathExtension("rebuild3d-result")
        let fm = FileManager.default
        for url in [destination, companion] {
            let target = url.resolvingSymlinksInPath().standardizedFileURL.path
            guard target != root, !target.hasPrefix(root + "/") else {
                throw ProjectError.invalid("Export outside the project to protect its originals and source records.")
            }
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: url.path, isDirectory: &isDirectory) {
                // A companion may contain user edits. Never replace it silently.
                guard url != companion, !isDirectory.boolValue else {
                    throw ProjectError.invalid("The companion result folder already exists, or the USDZ name is a folder. Choose a new export name.")
                }
            }
        }
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".rebuild3d-export-\(UUID())")
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: staging) }
        let bundled = staging.appendingPathComponent("result")
        let research = try ProjectStore.resolve(reference.bundlePath, in: project.directory).deletingLastPathComponent()
        try copyBundle(load(research), from: research, to: bundled)
        let stagedModel = staging.appendingPathComponent("model.usdz")
        try fm.copyItem(at: model, to: stagedModel)
        try fm.moveItem(at: bundled, to: companion)
        do {
            if fm.fileExists(atPath: destination.path) { _ = try fm.replaceItemAt(destination, withItemAt: stagedModel) }
            else { try fm.moveItem(at: stagedModel, to: destination) }
        } catch {
            try? fm.removeItem(at: companion)
            throw error
        }
    }
}
