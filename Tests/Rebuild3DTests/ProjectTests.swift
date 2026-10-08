import CoreGraphics
import Darwin
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import Rebuild3DCore

private final class Fixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("Rebuild3DTests-\(UUID())")
    init() throws { try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true) }
    deinit { try? FileManager.default.removeItem(at: root) }

    func image(_ relativePath: String = "input.jpg", orientation: Int = 1, type: UTType = .jpeg) throws -> URL {
        let url = root.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let context = try #require(CGContext(data: nil, width: 64, height: 48, bitsPerComponent: 8, bytesPerRow: 0,
                                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        for (index, color) in [CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                               CGColor(red: 0, green: 1, blue: 0, alpha: 1),
                               CGColor(red: 0, green: 0, blue: 1, alpha: 1),
                               CGColor(red: 1, green: 1, blue: 0, alpha: 1)].enumerated() {
            context.setFillColor(color)
            context.fill(CGRect(x: (index % 2) * 32, y: (index / 2) * 24, width: 32, height: 24))
        }
        let image = try #require(context.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }

    func project() throws -> Project { try ProjectStore.create(at: root.appendingPathComponent("Object.rebuild3d")) }

    func importedProject() throws -> Project {
        var project = try project()
        project.manifest.photos = try PhotoImporter.importPhotos(from: [image()], into: project.directory).photos
        try ProjectStore.save(project)
        return project
    }

    // These bytes exercise persistence only; they are not a valid textured model or a quality fixture.
    func model(_ name: String, content: String) throws -> URL {
        let url = root.appendingPathComponent(name)
        try Data(content.utf8).write(to: url)
        return url
    }
}

@Test func importPreservesOriginalAndOrientationAfterProjectMove() throws {
    let fixture = try Fixture()
    let source = try fixture.image(orientation: 6)
    var project = try fixture.project()
    let result = try PhotoImporter.importPhotos(from: [source], into: project.directory)
    #expect(result.issues.isEmpty)
    let photo = try #require(result.photos.first)
    #expect(photo.pixelWidth == 64 && photo.pixelHeight == 48 && photo.orientation == 6)
    #expect(try Data(contentsOf: source) == Data(contentsOf: ProjectStore.resolve(photo.imagePath, in: project.directory)))
    let thumbnail = try #require(CGImageSourceCreateWithURL(project.directory.appendingPathComponent(photo.thumbnailPath) as CFURL, nil))
    let properties = try #require(CGImageSourceCopyPropertiesAtIndex(thumbnail, 0, nil) as? [CFString: Any])
    #expect(properties[kCGImagePropertyPixelWidth] as? Int == 48)
    #expect(properties[kCGImagePropertyPixelHeight] as? Int == 64)
    project.manifest.photos = result.photos
    try ProjectStore.save(project)
    try FileManager.default.removeItem(at: source)
    let moved = fixture.root.appendingPathComponent("Moved.rebuild3d")
    try FileManager.default.moveItem(at: project.directory, to: moved)
    let restored = try ProjectStore.open(moved)
    #expect(restored.manifest.photos == project.manifest.photos)
    #expect(restored.manifest.id == project.manifest.id)
    #expect(try Data(contentsOf: ProjectStore.resolve(photo.imagePath, in: moved)).count > 0)
}

@Test func mixedInputsReportFailuresAndAvoidFilenameCollisions() throws {
    let fixture = try Fixture()
    let first = try fixture.image("first/photo.jpg")
    let second = try fixture.image("second/photo.jpg", orientation: 3)
    let broken = try fixture.model("broken.jpg", content: "not an image")
    let unsupported = try fixture.model("notes.txt", content: "notes")
    let project = try fixture.project()
    let imported = try PhotoImporter.importPhotos(from: [first, second, first, broken, unsupported], into: project.directory)
    #expect(imported.photos.count == 2)
    #expect(imported.issues.count == 3)
    #expect(imported.issues.filter { $0.kind == .duplicate }.count == 1)
    #expect(imported.issues.filter { $0.kind == .ignored }.count == 1)
    #expect(imported.issues.filter { $0.kind == .unreadable }.count == 1)
    #expect(Set(imported.photos.map(\.id)).count == 2)
    #expect(Set(imported.photos.map(\.imagePath)).count == 2)
    let files = try FileManager.default.contentsOfDirectory(atPath: project.directory.appendingPathComponent("images").path)
    #expect(files.count == 2)
}

@Test func thumbnailsCanBeRegeneratedWithoutOriginalSource() throws {
    let fixture = try Fixture()
    let project = try fixture.importedProject()
    let photo = try #require(project.manifest.photos.first)
    let thumbnail = project.directory.appendingPathComponent(photo.thumbnailPath)
    try FileManager.default.removeItem(at: thumbnail)
    try PhotoImporter.regenerateThumbnail(for: photo, in: project.directory)
    #expect(FileManager.default.isReadableFile(atPath: thumbnail.path))
}

@Test func unsupportedVersionsAndMissingOriginalsAreRejected() throws {
    let fixture = try Fixture()
    let project = try fixture.importedProject()
    let manifestURL = project.directory.appendingPathComponent("project.json")
    let original = try Data(contentsOf: manifestURL)
    try Data("{\"formatVersion\":999}".utf8).write(to: manifestURL)
    #expect(throws: ProjectError.self) { try ProjectStore.open(project.directory) }
    try original.write(to: manifestURL)
    let photo = try #require(project.manifest.photos.first)
    try FileManager.default.removeItem(at: project.directory.appendingPathComponent(photo.imagePath))
    #expect(throws: ProjectError.self) { try ProjectStore.open(project.directory) }
}

@Test func unsafePathsAndSymbolicLinksAreRejected() throws {
    let fixture = try Fixture()
    let project = try fixture.project()
    for path in ["../outside", "/tmp/outside", "images/../../outside", ""] {
        #expect(throws: ProjectError.self) { try ProjectStore.resolve(path, in: project.directory) }
    }
    let outside = try fixture.model("outside.jpg", content: "private data")
    let link = project.directory.appendingPathComponent("images/link.jpg")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)
    #expect(throws: ProjectError.self) { try ProjectStore.resolve("images/link.jpg", in: project.directory) }
}

@Test func failedModelCommitPreservesPreviousModelAndManifest() throws {
    let fixture = try Fixture()
    var project = try fixture.importedProject()
    let first = try fixture.model("first.usdz", content: "first model payload")
    project = try ProjectStore.commitModel(from: first, to: project)
    let previousManifest = try Data(contentsOf: project.directory.appendingPathComponent("project.json"))
    let previousURL = try #require(project.modelURL)
    let second = try fixture.model("second.usdz", content: "replacement model payload")
    var invalid = project
    invalid.manifest.formatVersion = 999
    #expect(throws: ProjectError.self) { try ProjectStore.commitModel(from: second, to: invalid) }
    #expect(try Data(contentsOf: project.directory.appendingPathComponent("project.json")) == previousManifest)
    #expect(try Data(contentsOf: previousURL) == Data(contentsOf: first))
    #expect(try FileManager.default.contentsOfDirectory(atPath: project.directory.appendingPathComponent("models").path).count == 1)
    let empty = try fixture.model("empty.usdz", content: "")
    #expect(throws: ProjectError.self) { try ProjectStore.commitModel(from: empty, to: project) }
    let restored = try ProjectStore.open(project.directory)
    #expect(restored.manifest.model?.path == project.manifest.model?.path)
    #expect(restored.manifest.model?.settings == project.manifest.model?.settings)
    #expect(restored.manifest.model?.photoIDs == project.manifest.model?.photoIDs)
    // ISO 8601 metadata stores timestamps with second precision.
    let restoredDate = try #require(restored.manifest.model?.createdAt)
    let originalDate = try #require(project.manifest.model?.createdAt)
    #expect(abs(restoredDate.timeIntervalSince(originalDate)) < 1)
}

@Test func exportPreservesAllBytesAndReplacesExistingDestination() throws {
    let fixture = try Fixture()
    let original = try fixture.model("source.usdz", content: "geometry, UVs, materials, and texture payload")
    let project = try ProjectStore.commitModel(from: original, to: fixture.importedProject())
    let exported = try fixture.model("export.usdz", content: "previous export")
    try ProjectStore.exportModel(project, to: exported)
    #expect(try Data(contentsOf: exported) == Data(contentsOf: original))
    #expect(throws: ProjectError.self) { try ProjectStore.exportModel(project, to: project.directory.appendingPathComponent("images/danger.usdz")) }
    let moved = fixture.root.appendingPathComponent("Moved.rebuild3d")
    try FileManager.default.moveItem(at: project.directory, to: moved)
    let restored = try ProjectStore.open(moved)
    #expect(try Data(contentsOf: #require(restored.modelURL)) == Data(contentsOf: original))
}

@Test func rejectedReconstructionLeavesSavedResultsAndRunDirectoriesUntouched() async throws {
    let fixture = try Fixture()
    let project = try ProjectStore.commitModel(from: fixture.model("saved.usdz", content: "saved model"),
                                               to: fixture.importedProject())
    let manifestURL = project.directory.appendingPathComponent("project.json")
    let previousManifest = try Data(contentsOf: manifestURL)
    let modelURL = try #require(project.modelURL)
    let previousModel = try Data(contentsOf: modelURL)
    let engine = ReconstructionEngine()
    await #expect(throws: ProjectError.self) {
        try await engine.run(project: project) { _ in
            Issue.record("Invalid input must be rejected before starting a reconstruction.")
        }
    }
    #expect(try Data(contentsOf: manifestURL) == previousManifest)
    #expect(try Data(contentsOf: modelURL) == previousModel)
    for directory in ["cache", "runs", "logs"] {
        #expect(try FileManager.default.contentsOfDirectory(atPath: project.directory.appendingPathComponent(directory).path).isEmpty)
    }
}

@Test func changedInputsOrSettingsMarkModelOutdated() throws {
    let fixture = try Fixture()
    var project = try ProjectStore.commitModel(from: fixture.model("source.usdz", content: "model"), to: fixture.importedProject())
    #expect(!project.modelIsOutdated)
    project.manifest.settings.objectMasking.toggle()
    #expect(project.modelIsOutdated)
    project.manifest.settings.objectMasking.toggle()
    project.manifest.photos.removeAll()
    #expect(project.modelIsOutdated)
}

@Test func exportCannotReplaceProjectRootOrOtherDirectories() throws {
    let fixture = try Fixture()
    let model = try fixture.model("source.usdz", content: "saved model")
    let project = try ProjectStore.commitModel(from: model, to: fixture.importedProject())
    let before = try Data(contentsOf: project.directory.appendingPathComponent("project.json"))
    let unrelated = fixture.root.appendingPathComponent("OtherFolder")
    try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: true)
    let marker = unrelated.appendingPathComponent("keep.txt")
    try Data("keep".utf8).write(to: marker)
    let alias = fixture.root.appendingPathComponent("ProjectAlias")
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: project.directory)
    for destination in [project.directory, alias, unrelated] {
        #expect(throws: ProjectError.self) { try ProjectStore.exportModel(project, to: destination) }
    }
    #expect(try Data(contentsOf: project.directory.appendingPathComponent("project.json")) == before)
    #expect(try Data(contentsOf: #require(project.modelURL)) == Data(contentsOf: model))
    #expect(try String(contentsOf: marker, encoding: .utf8) == "keep")
}

@Test func duplicatePhotoIDsCannotCorruptSavedProject() throws {
    let fixture = try Fixture()
    var project = try fixture.importedProject()
    let previousManifest = try Data(contentsOf: project.directory.appendingPathComponent("project.json"))
    project.manifest.photos.append(try #require(project.manifest.photos.first))
    #expect(throws: ProjectError.self) { try ProjectStore.save(project) }
    #expect(try Data(contentsOf: project.directory.appendingPathComponent("project.json")) == previousManifest)
}

@Test(.enabled(if: getuid() != 0)) func unwritableManifestPreservesPreviousResult() throws {
    let fixture = try Fixture()
    let first = try fixture.model("first.usdz", content: "previous result")
    let project = try ProjectStore.commitModel(from: first, to: fixture.importedProject())
    let manifest = project.directory.appendingPathComponent("project.json")
    let original = try Data(contentsOf: manifest)
    let replacement = try fixture.model("replacement.usdz", content: "new result")
    try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: project.directory.path)
    defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: project.directory.path) }
    #expect(throws: (any Error).self) { try ProjectStore.commitModel(from: replacement, to: project) }
    #expect(try Data(contentsOf: manifest) == original)
    #expect(try Data(contentsOf: #require(project.modelURL)) == Data(contentsOf: first))
    #expect(try FileManager.default.contentsOfDirectory(atPath: project.directory.appendingPathComponent("models").path).count == 1)
}

@Test func legacyMigrationIsReadOnlyUntilSuccessfulSave() throws {
    let fixture = try Fixture()
    let project = try ProjectStore.commitModel(from: fixture.model("legacy.usdz", content: "legacy model"), to: fixture.importedProject())
    let manifestURL = project.directory.appendingPathComponent("project.json")
    var legacy = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
    legacy["formatVersion"] = 1
    var photos = try #require(legacy["photos"] as? [[String: Any]])
    for index in photos.indices { photos[index].removeValue(forKey: "metadata") }
    legacy["photos"] = photos
    let original = try JSONSerialization.data(withJSONObject: legacy, options: [.sortedKeys])
    try original.write(to: manifestURL)
    let migrated = try ProjectStore.open(project.directory)
    #expect(migrated.needsMigrationSave)
    #expect(migrated.manifest.formatVersion == 2)
    #expect(migrated.manifest.photos.map(\.id) == project.manifest.photos.map(\.id))
    #expect(migrated.manifest.photos.first?.metadata?.contentSHA256 != nil)
    #expect(migrated.manifest.model?.path == project.manifest.model?.path)
    #expect(migrated.manifest.model?.inputSnapshotPath == nil)
    #expect(try Data(contentsOf: manifestURL) == original)
    try ProjectStore.save(migrated)
    let reopened = try ProjectStore.open(project.directory)
    #expect(!reopened.needsMigrationSave)
    #expect(reopened.manifest.photos == migrated.manifest.photos)
}

@Test(.enabled(if: getuid() != 0)) func failedMigrationSaveLeavesLegacyBytesIntact() throws {
    let fixture = try Fixture()
    let project = try fixture.importedProject()
    let manifestURL = project.directory.appendingPathComponent("project.json")
    var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
    object["formatVersion"] = 1
    let legacy = try JSONSerialization.data(withJSONObject: object)
    try legacy.write(to: manifestURL)
    let migrated = try ProjectStore.open(project.directory)
    try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: project.directory.path)
    defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: project.directory.path) }
    #expect(throws: (any Error).self) { try ProjectStore.save(migrated) }
    #expect(try Data(contentsOf: manifestURL) == legacy)
    #expect(try ProjectStore.open(project.directory).needsMigrationSave)
}

@Test func decodedContentAndPersistentDeduplicationDoNotDependOnFilename() throws {
    let fixture = try Fixture()
    let source = try fixture.image("photo.unknown")
    var project = try fixture.project()
    let imported = try PhotoImporter.importPhotos(from: [source], into: project.directory)
    let photo = try #require(imported.photos.first)
    #expect(photo.originalName == "photo.unknown")
    #expect(photo.metadata?.typeIdentifier == UTType.jpeg.identifier)
    #expect(photo.imagePath.hasSuffix(".jpeg") || photo.imagePath.hasSuffix(".jpg"))
    project.manifest.photos = imported.photos
    try ProjectStore.save(project)
    let copy = fixture.root.appendingPathComponent("same-bytes.heic")
    try FileManager.default.copyItem(at: source, to: copy)
    let duplicate = try PhotoImporter.importPhotos(from: [copy], into: ProjectStore.open(project.directory).directory)
    #expect(duplicate.photos.isEmpty)
    #expect(duplicate.issues.first?.kind == .duplicate)
    let raw = try fixture.model("raw.dng", content: "unreadable raw fixture")
    let video = try fixture.model("companion.mov", content: "companion fixture")
    let mixed = try PhotoImporter.importPhotos(from: [raw, video], into: project.directory)
    #expect(mixed.issues.map(\.kind) == [.unsupported, .ignored])
}

@Test func inputSnapshotSurvivesCacheRemovalProjectMoveAndPhotoRemoval() throws {
    let fixture = try Fixture()
    var project = try fixture.importedProject()
    let runID = UUID()
    let snapshot = try PhotoPreparation.stageInputs(project: project, runID: runID)
    let entry = try #require(snapshot.inputs.first)
    #expect(try Data(contentsOf: ProjectStore.resolve(entry.stagedPath, in: project.directory)) == Data(contentsOf: ProjectStore.resolve(entry.photo.imagePath, in: project.directory)))
    let snapshotPath = try ProjectStore.saveInputSnapshot(snapshot, in: project.directory)
    project = try ProjectStore.commitModel(from: fixture.model("result.usdz", content: "model"), to: project, inputSnapshotPath: snapshotPath)
    project.manifest.photos.removeAll()
    try ProjectStore.save(project)
    try FileManager.default.removeItem(at: project.directory.appendingPathComponent("cache"))
    let moved = fixture.root.appendingPathComponent("Moved.rebuild3d")
    try FileManager.default.moveItem(at: project.directory, to: moved)
    let reopened = try ProjectStore.open(moved)
    let restored = try ProjectStore.loadInputSnapshot(snapshotPath, in: moved)
    #expect(reopened.manifest.model?.runID == runID)
    #expect(restored.inputs == snapshot.inputs)
    #expect(reopened.modelIsOutdated)
    #expect(throws: ProjectError.self) { try ProjectStore.saveInputSnapshot(restored, in: moved) }
}

@Test func mismatchedSnapshotCannotReplaceSuccessfulModel() throws {
    let fixture = try Fixture()
    let project = try ProjectStore.commitModel(from: fixture.model("old.usdz", content: "old"), to: fixture.importedProject())
    let manifestURL = project.directory.appendingPathComponent("project.json")
    let before = try Data(contentsOf: manifestURL)
    var snapshot = try PhotoPreparation.stageInputs(project: project, runID: UUID())
    snapshot.projectID = UUID()
    let path = try ProjectStore.saveInputSnapshot(snapshot, in: project.directory)
    let newModel = try fixture.model("new.usdz", content: "new")
    #expect(throws: ProjectError.self) { try ProjectStore.commitModel(from: newModel, to: project, inputSnapshotPath: path) }
    #expect(try Data(contentsOf: manifestURL) == before)
    #expect(try Data(contentsOf: #require(project.modelURL)) == Data("old".utf8))
}

@Test(arguments: 1...8) func originalPreparationRetainsEncodedPixelsAndDisplayOrientation(orientation: Int) throws {
    let fixture = try Fixture()
    var project = try fixture.project()
    project.manifest.photos = try PhotoImporter.importPhotos(from: [fixture.image(orientation: orientation)], into: project.directory).photos
    let snapshot = try PhotoPreparation.stageInputs(project: project, runID: UUID())
    let entry = try #require(snapshot.inputs.first)
    #expect(entry.preparation.inputWidth == 64 && entry.preparation.inputHeight == 48)
    #expect(entry.preparation.inputOrientation == orientation)
    #expect(entry.preparation.originalToInputPixels == [1, 0, 0, 0, 1, 0, 0, 0, 1])
    #expect(entry.preparation.inputSHA256 == entry.photo.metadata?.contentSHA256)
}

@Test func modifiedOriginalIsDetectedBeforeReconstruction() throws {
    let fixture = try Fixture()
    let project = try fixture.importedProject()
    let photo = try #require(project.manifest.photos.first)
    let replacement = try fixture.image("replacement.jpg", orientation: 3)
    try Data(contentsOf: replacement).write(to: ProjectStore.resolve(photo.imagePath, in: project.directory))
    #expect(throws: ProjectError.self) { try PhotoPreparation.stageInputs(project: project, runID: UUID()) }
}

@Test(arguments: 1...8) func tiffWorkingImagesPreserveOriginalsAndMatchRecordedPixelTransforms(orientation: Int) throws {
    let fixture = try Fixture()
    let original = try fixture.image("original.tiff", orientation: orientation, type: .tiff)
    let originalBytes = try Data(contentsOf: original)
    var project = try fixture.project()
    project.manifest.photos = try PhotoImporter.importPhotos(from: [original], into: project.directory).photos
    let snapshot = try PhotoPreparation.stageInputs(project: project, runID: UUID())
    let input = try #require(snapshot.inputs.first)
    let working = try ProjectStore.resolve(input.stagedPath, in: project.directory)
    let actual = try PhotoInspector.inspect(working)
    #expect(input.preparation.version == 2 && input.preparation.operation == "tiff-to-png")
    #expect(actual.metadata.typeIdentifier == UTType.png.identifier && actual.orientation == 1)
    #expect(actual.pixelWidth == (orientation >= 5 ? 48 : 64))
    #expect(actual.pixelHeight == (orientation >= 5 ? 64 : 48))
    #expect(input.preparation.inputSHA256 == actual.metadata.contentSHA256)
    #expect(input.preparation.inputSHA256 != input.photo.metadata?.contentSHA256)
    let from = try #require(CGImageSourceCreateWithURL(original as CFURL, nil))
    let to = try #require(CGImageSourceCreateWithURL(working as CFURL, nil))
    let sourcePixels = try #require(CGImageSourceCreateImageAtIndex(from, 0, nil))
    let inputPixels = try #require(CGImageSourceCreateImageAtIndex(to, 0, nil))
    #expect(inputPixels.colorSpace?.name == CGColorSpace.sRGB)
    let matrix = input.preparation.originalToInputPixels
    for (x, y) in [(5, 5), (58, 5), (5, 42), (58, 42)] {
        let u = Int(matrix[0] * Double(x) + matrix[1] * Double(y) + matrix[2])
        let v = Int(matrix[3] * Double(x) + matrix[4] * Double(y) + matrix[5])
        let before = try pixelRGB(sourcePixels, x: x, y: y)
        let after = try pixelRGB(inputPixels, x: u, y: v)
        #expect(zip(before, after).allSatisfy { abs(Int($0) - Int($1)) <= 1 })
    }
    let snapshotPath = try ProjectStore.saveInputSnapshot(snapshot, in: project.directory)
    project = try ProjectStore.commitModel(from: fixture.model("model.usdz", content: "model"), to: project, inputSnapshotPath: snapshotPath)
    try FileManager.default.removeItem(at: project.directory.appendingPathComponent("cache"))
    let moved = fixture.root.appendingPathComponent("Moved.rebuild3d")
    try FileManager.default.moveItem(at: project.directory, to: moved)
    let restored = try ProjectStore.open(moved)
    let savedInputs = try ProjectStore.loadInputSnapshot(snapshotPath, in: moved)
    #expect(savedInputs.inputs == snapshot.inputs)
    #expect(try Data(contentsOf: ProjectStore.resolve(input.photo.imagePath, in: restored.directory)) == originalBytes)
    #expect(try Data(contentsOf: original) == originalBytes)
    var corrupt = savedInputs
    corrupt.runID = UUID()
    corrupt.inputs[0].stagedPath = "cache/\(corrupt.runID.uuidString)/input/test.png"
    corrupt.inputs[0].preparation.originalToInputPixels[2] += 1
    #expect(throws: ProjectError.self) { try ProjectStore.saveInputSnapshot(corrupt, in: moved) }
}

private func pixelRGB(_ image: CGImage, x: Int, y: Int) throws -> [UInt8] {
    let pixel = try #require(image.cropping(to: CGRect(x: x, y: y, width: 1, height: 1)))
    let context = try #require(CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.draw(pixel, in: CGRect(x: 0, y: 0, width: 1, height: 1))
    let data = try #require(context.data).assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: data, count: 3))
}

@Test func draftRecoversAndPublishesBeforeSourceIsDiscarded() throws {
    let fixture = try Fixture()
    let store = DraftProjectStore(root: fixture.root.appendingPathComponent("Drafts"))
    var draft = try store.create()
    draft.manifest.photos = try PhotoImporter.importPhotos(from: [fixture.image()], into: draft.directory).photos
    try ProjectStore.save(draft)
    let freshStore = DraftProjectStore(root: store.root)
    let summary = try #require(freshStore.list().first)
    #expect(summary.photoCount == 1)
    let recovered = try ProjectStore.open(summary.directory)
    #expect(recovered.manifest.photos == draft.manifest.photos)
    let destination = fixture.root.appendingPathComponent("Saved.rebuild3d")
    let saved = try freshStore.saveAs(recovered, to: destination)
    #expect(saved.manifest.name == "Saved")
    #expect(saved.manifest.id == draft.manifest.id)
    #expect(!freshStore.contains(saved.directory))
    #expect(try ProjectStore.open(saved.directory).manifest.photos == draft.manifest.photos)
    #expect(FileManager.default.fileExists(atPath: draft.directory.path))
    try freshStore.discard(draft.directory)
    #expect(try freshStore.list().isEmpty)
    #expect(try ProjectStore.open(destination).manifest.photos.count == 1)
}

@Test func failedDraftPublicationPreservesOriginalAndExistingDestination() throws {
    let fixture = try Fixture()
    let store = DraftProjectStore(root: fixture.root.appendingPathComponent("Drafts"))
    let draft = try store.create()
    let before = try Data(contentsOf: draft.directory.appendingPathComponent("project.json"))
    let destination = try fixture.model("Existing.rebuild3d", content: "do not replace")
    #expect(throws: ProjectError.self) { try store.saveAs(draft, to: destination) }
    #expect(try Data(contentsOf: destination) == Data("do not replace".utf8))
    #expect(try Data(contentsOf: draft.directory.appendingPathComponent("project.json")) == before)
    let missingParent = fixture.root.appendingPathComponent("missing/Target.rebuild3d")
    #expect(throws: (any Error).self) { try store.saveAs(draft, to: missingParent) }
    #expect(try freshDraftIDs(store) == [draft.manifest.id])
}

private func freshDraftIDs(_ store: DraftProjectStore) throws -> [UUID] {
    try DraftProjectStore(root: store.root).list().map { try ProjectStore.open($0.directory).manifest.id }
}

@Test func draftDeletionCannotReachExternalProjectsOrSymlinkTargets() throws {
    let fixture = try Fixture()
    let store = DraftProjectStore(root: fixture.root.appendingPathComponent("Drafts"))
    _ = try store.create()
    let outside = try fixture.project()
    #expect(throws: ProjectError.self) { try store.discard(outside.directory) }
    let link = store.root.appendingPathComponent("linked.rebuild3d")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside.directory)
    #expect(!store.contains(link))
    #expect(throws: ProjectError.self) { try store.discard(link) }
    #expect(try ProjectStore.open(outside.directory).manifest.id == outside.manifest.id)
}

@Test(arguments: 1...8) func previewsUseSRGBAndRespectAllEXIFOrientations(orientation: Int) throws {
    let fixture = try Fixture()
    let source = try fixture.image(orientation: orientation)
    let before = try Data(contentsOf: source)
    let preview = try PhotoPreviewDecoder.image(at: source, maxPixelSize: 64)
    #expect(preview.colorSpace?.name == CGColorSpace.sRGB)
    #expect(preview.width == (orientation >= 5 ? 48 : 64))
    #expect(preview.height == (orientation >= 5 ? 64 : 48))
    #expect(try Data(contentsOf: source) == before)
}
