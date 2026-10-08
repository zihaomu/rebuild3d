import Foundation

public enum ProjectStore {
    private static let folders = ["images", "thumbnails", "models", "logs", "cache", "runs"]

    public static func create(at directory: URL) throws -> Project {
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw ProjectError.invalid("A file or project already exists at this location. Choose a new name.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            for folder in folders {
                try FileManager.default.createDirectory(at: directory.appendingPathComponent(folder), withIntermediateDirectories: false)
            }
            let project = Project(directory: directory, manifest: .init(name: directory.deletingPathExtension().lastPathComponent))
            try save(project)
            return project
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    public static func open(_ directory: URL) throws -> Project {
        let url = directory.appendingPathComponent("project.json")
        let data = try Data(contentsOf: url)
        let header = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let storedVersion = header?["formatVersion"] as? Int, [1, ProjectManifest.currentVersion].contains(storedVersion) else {
            throw ProjectError.invalid("This project uses an unsupported format version.")
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var manifest = try decoder.decode(ProjectManifest.self, from: data)
        manifest.formatVersion = ProjectManifest.currentVersion
        try validate(manifest, in: directory)
        if storedVersion == 1 {
            for index in manifest.photos.indices where manifest.photos[index].metadata == nil {
                let source = try resolve(manifest.photos[index].imagePath, in: directory)
                // Reading a legacy project never rewrites it or makes its saved model depend on metadata extraction.
                manifest.photos[index].metadata = try? autoreleasepool { try PhotoInspector.inspect(source).metadata }
            }
        }
        return Project(directory: directory, manifest: manifest, loadedFormatVersion: storedVersion)
    }

    public static func save(_ project: Project) throws {
        try validate(project.manifest, in: project.directory)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(project.manifest).write(to: project.directory.appendingPathComponent("project.json"), options: .atomic)
    }

    /// Resolve project-relative paths without permitting traversal or symlink escapes.
    public static func resolve(_ path: String, in directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw ProjectError.invalid("The project contains an unsafe file path: \(path)")
        }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let target = root.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
        guard target.path.hasPrefix(root.path + "/") else {
            throw ProjectError.invalid("A project file points outside the project directory.")
        }
        return target
    }

    private static func validate(_ manifest: ProjectManifest, in directory: URL) throws {
        guard manifest.formatVersion == ProjectManifest.currentVersion else {
            throw ProjectError.invalid("This project uses an unsupported format version.")
        }
        guard Set(manifest.photos.map(\.id)).count == manifest.photos.count,
              Set(manifest.photos.map(\.imagePath)).count == manifest.photos.count else {
            throw ProjectError.invalid("The project contains duplicate photo IDs or paths.")
        }
        for photo in manifest.photos { try validatePhoto(photo, in: directory) }
        if let model = manifest.model {
            guard model.path.hasPrefix("models/"), model.path.hasSuffix(".usdz") else {
                throw ProjectError.invalid("The project contains an invalid model path.")
            }
            let url = try resolve(model.path, in: directory)
            guard FileManager.default.isReadableFile(atPath: url.path) else {
                throw ProjectError.invalid("The saved model is missing or unreadable.")
            }
            if let path = model.inputSnapshotPath, let runID = model.runID {
                let snapshot = try loadInputSnapshot(path, in: directory)
                guard snapshot.projectID == manifest.id, snapshot.runID == runID,
                      snapshot.settings == model.settings,
                      snapshot.inputs.map(\.photo.id) == model.photoIDs else {
                    throw ProjectError.invalid("The saved model and its input snapshot do not match.")
                }
            } else if model.inputSnapshotPath != nil || model.runID != nil {
                throw ProjectError.invalid("The saved model has an incomplete run reference.")
            }
        }
    }

    private static func validatePhoto(_ photo: PhotoRecord, in directory: URL) throws {
        guard photo.imagePath.hasPrefix("images/"), photo.thumbnailPath.hasPrefix("thumbnails/"),
              photo.pixelWidth > 0, photo.pixelHeight > 0, (1...8).contains(photo.orientation) else {
            throw ProjectError.invalid("The project contains invalid photo metadata.")
        }
        let image = try resolve(photo.imagePath, in: directory)
        _ = try resolve(photo.thumbnailPath, in: directory)
        guard FileManager.default.isReadableFile(atPath: image.path) else {
            throw ProjectError.invalid("The original photo \(photo.originalName) is missing or unreadable.")
        }
        if let metadata = photo.metadata {
            guard metadata.primaryImageIndex >= 0, metadata.byteCount > 0, !metadata.typeIdentifier.isEmpty,
                  validDigest(metadata.contentSHA256) else {
                throw ProjectError.invalid("The photo has invalid content metadata.")
            }
        }
    }

    private static func validDigest(_ digest: String) -> Bool {
        digest.count == 64 && digest.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func validateSnapshot(_ snapshot: ReconstructionInputSnapshot, in directory: URL) throws {
        guard snapshot.formatVersion == 1,
              Set(snapshot.inputs.map(\.photo.id)).count == snapshot.inputs.count,
              Set(snapshot.inputs.map(\.stagedPath)).count == snapshot.inputs.count else {
            throw ProjectError.invalid("The reconstruction input snapshot is invalid.")
        }
        for input in snapshot.inputs {
            try validatePhoto(input.photo, in: directory)
            let expected = "cache/\(snapshot.runID.uuidString)/input/"
            _ = try resolve(input.stagedPath, in: directory)
            let recipe = input.preparation
            guard input.stagedPath.hasPrefix(expected), validDigest(recipe.inputSHA256) else {
                throw ProjectError.invalid("The photo preparation record is inconsistent.")
            }
            switch (recipe.version, recipe.operation) {
            case (1, "original-file"):
                guard recipe.inputWidth == input.photo.pixelWidth, recipe.inputHeight == input.photo.pixelHeight,
                      recipe.inputOrientation == input.photo.orientation, recipe.originalToInputPixels == PhotoPixelTransform.identity,
                      recipe.colorPolicy == "preserve-original-profile-and-auxiliary-data",
                      recipe.inputSHA256 == input.photo.metadata?.contentSHA256,
                      recipe.inputTypeIdentifier == nil || recipe.inputTypeIdentifier == input.photo.metadata?.typeIdentifier else {
                    throw ProjectError.invalid("The original-file preparation record is inconsistent.")
                }
            case (2, "tiff-to-png"):
                let swapsAxes = input.photo.orientation >= 5
                guard input.photo.metadata?.typeIdentifier == "public.tiff", recipe.inputTypeIdentifier == "public.png",
                      input.stagedPath.hasSuffix(".png"), recipe.inputOrientation == 1,
                      recipe.inputWidth == (swapsAxes ? input.photo.pixelHeight : input.photo.pixelWidth),
                      recipe.inputHeight == (swapsAxes ? input.photo.pixelWidth : input.photo.pixelHeight),
                      recipe.colorPolicy == "sdr-srgb-8bit",
                      recipe.originalToInputPixels == (try PhotoPixelTransform.oriented(width: input.photo.pixelWidth,
                          height: input.photo.pixelHeight, orientation: input.photo.orientation)) else {
                    throw ProjectError.invalid("The TIFF working-image preparation record is inconsistent.")
                }
            default: throw ProjectError.invalid("This photo preparation version is not supported.")
            }
        }
    }

    public static func saveInputSnapshot(_ snapshot: ReconstructionInputSnapshot, in directory: URL) throws -> String {
        try validateSnapshot(snapshot, in: directory)
        let url = try resolve(snapshot.relativePath, in: directory)
        guard !FileManager.default.fileExists(atPath: url.path) else {
            throw ProjectError.invalid("A snapshot already exists for this reconstruction run.")
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(snapshot).write(to: url, options: .atomic)
        return snapshot.relativePath
    }

    public static func loadInputSnapshot(_ path: String, in directory: URL) throws -> ReconstructionInputSnapshot {
        guard path.hasPrefix("runs/"), path.hasSuffix("/inputs.json") else {
            throw ProjectError.invalid("The project has an invalid input snapshot path.")
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let snapshot = try decoder.decode(ReconstructionInputSnapshot.self, from: Data(contentsOf: resolve(path, in: directory)))
        guard snapshot.relativePath == path else { throw ProjectError.invalid("The snapshot run ID does not match its path.") }
        try validateSnapshot(snapshot, in: directory)
        return snapshot
    }

    /// Commit a new immutable model before atomically switching the manifest pointer.
    /// A failed write leaves the previous successful model and manifest intact.
    public static func commitModel(from source: URL, to project: Project, inputSnapshotPath: String? = nil) throws -> Project {
        let snapshot = try inputSnapshotPath.map { try loadInputSnapshot($0, in: project.directory) }
        if let snapshot {
            guard snapshot.projectID == project.manifest.id, snapshot.settings == project.manifest.settings,
                  snapshot.inputs.map(\.photo.id) == project.manifest.photos.map(\.id) else {
                throw ProjectError.invalid("The reconstruction inputs changed before the model was saved.")
            }
        }
        let size = try source.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size > 0 else { throw ProjectError.invalid("Reconstruction did not produce a readable model.") }
        let path = "models/\(UUID().uuidString).usdz"
        let destination = try resolve(path, in: project.directory)
        do {
            // Copy failures can leave a partial file even though no model was committed.
            try FileManager.default.copyItem(at: source, to: destination)
            var updated = project
            updated.manifest.model = ModelRecord(path: path, createdAt: Date(), settings: project.manifest.settings,
                                                photoIDs: project.manifest.photos.map(\.id), runID: snapshot?.runID,
                                                inputSnapshotPath: inputSnapshotPath)
            updated.manifest.modifiedAt = Date()
            try save(updated)
            updated.loadedFormatVersion = ProjectManifest.currentVersion
            return updated
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
    }

    public static func exportModel(_ project: Project, to destination: URL) throws {
        guard let source = project.modelURL else { throw ProjectError.invalid("Reconstruct a model before exporting.") }
        guard source.standardizedFileURL != destination.standardizedFileURL else { return }
        let root = project.directory.resolvingSymlinksInPath().standardizedFileURL.path
        let target = destination.resolvingSymlinksInPath().standardizedFileURL.path
        guard target != root, !target.hasPrefix(root + "/") else {
            throw ProjectError.invalid("Choose an export location outside the project to protect its source files.")
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory), isDirectory.boolValue {
            throw ProjectError.invalid("Choose a USDZ filename, not an existing folder or project.")
        }
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).usdz")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: source, to: staging)
        if FileManager.default.fileExists(atPath: destination.path) {
            _ = try FileManager.default.replaceItemAt(destination, withItemAt: staging)
        } else {
            try FileManager.default.moveItem(at: staging, to: destination)
        }
    }
}
