// Fault injection on a small, separately mounted scratch filesystem only.
// Build: swiftc -parse-as-library Sources/Rebuild3DCore/*.swift scripts/probe-disk-full.swift -o build/disk-full-probe
import Darwin
import Foundation

@main struct DiskFullProbe {
    static let fm = FileManager.default

    static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw ProjectError.invalid(message) }
    }

    static func isOutOfSpace(_ error: NSError) -> Bool {
        if error.domain == NSPOSIXErrorDomain && error.code == Int(ENOSPC) { return true }
        if error.domain == NSCocoaErrorDomain && error.code == NSFileWriteOutOfSpaceError { return true }
        if let underlying = error.userInfo[NSUnderlyingErrorKey] as? NSError { return isOutOfSpace(underlying) }
        return false
    }

    static func fill(_ url: URL) throws -> Int {
        let fd = open(url.path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        defer { close(fd) }
        var total = 0
        for chunkSize in [1024 * 1024, 4096] {
            let chunk = Data(repeating: 0xA5, count: chunkSize)
            while true {
                try require(total < 128 * 1024 * 1024, "Scratch fill exceeded the safety cap.")
                let written = chunk.withUnsafeBytes { write(fd, $0.baseAddress!, $0.count) }
                if written < 0 {
                    if errno == EINTR { continue }
                    guard errno == ENOSPC else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
                    break
                }
                try require(written > 0, "The filler made no progress.")
                total += written
            }
        }
        if fsync(fd) != 0 && errno != ENOSPC { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        return total
    }

    static func main() {
        do { try run() }
        catch {
            FileHandle.standardError.write(Data("Disk-full probe failed: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    static func run() throws {
        guard CommandLine.arguments.count == 4 else {
            print("Usage: disk-full-probe MOUNTED_SCRATCH_VOLUME JPEG_DIRECTORY NEW_REPORT_DIRECTORY")
            exit(2)
        }
        let volume = URL(fileURLWithPath: CommandLine.arguments[1]).resolvingSymlinksInPath().standardizedFileURL
        let sources = URL(fileURLWithPath: CommandLine.arguments[2])
        let output = URL(fileURLWithPath: CommandLine.arguments[3]).standardizedFileURL
        var mounted = stat(), parent = stat()
        try require(stat(volume.path, &mounted) == 0 && stat(volume.deletingLastPathComponent().path, &parent) == 0,
                    "Cannot inspect scratch mount.")
        try require(mounted.st_dev != parent.st_dev, "Refusing to fill a directory that is not a separate mount.")
        let capacity = (try fm.attributesOfFileSystem(forPath: volume.path)[.systemSize] as? NSNumber)?.uint64Value ?? 0
        try require(capacity > 0 && capacity <= 128 * 1024 * 1024, "Scratch filesystem must be at most 128 MiB.")
        try require(!output.path.hasPrefix(volume.path + "/") && output != volume,
                    "Keep reports outside the filesystem being filled.")
        try require(!fm.fileExists(atPath: output.path), "The report directory must be new.")
        try fm.createDirectory(at: output, withIntermediateDirectories: true)
        let photos = try fm.contentsOfDirectory(at: sources, includingPropertiesForKeys: nil)
            .filter { ["jpg", "jpeg"].contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path }
        try require(photos.count >= 2, "Two different source JPEG files are needed.")

        // Opaque payloads exercise storage only, not model validity or reconstruction quality.
        let oldModel = output.appendingPathComponent("old-model.usdz")
        let newModel = output.appendingPathComponent("new-model.usdz")
        try Data(repeating: 0x11, count: 64 * 1024).write(to: oldModel)
        try Data(repeating: 0x22, count: 1024 * 1024).write(to: newModel)
        var project = try ProjectStore.create(at: volume.appendingPathComponent("Existing.rebuild3d"))
        project.manifest.photos = try PhotoImporter.importPhotos(from: [photos[0]], into: project.directory).photos
        project = try ProjectStore.commitModel(from: oldModel, to: project)
        let manifestURL = project.directory.appendingPathComponent("project.json")
        let beforeManifest = try Data(contentsOf: manifestURL)
        let beforeModel = try Data(contentsOf: project.modelURL!)
        let imageFiles = try fm.contentsOfDirectory(atPath: project.directory.appendingPathComponent("images").path).sorted()
        let modelFiles = try fm.contentsOfDirectory(atPath: project.directory.appendingPathComponent("models").path).sorted()
        let exportURL = volume.appendingPathComponent("existing-export.usdz")
        try ProjectStore.exportModel(project, to: exportURL)
        let beforeExport = try Data(contentsOf: exportURL)

        let drafts = DraftProjectStore(root: output.appendingPathComponent("Drafts"))
        var draft = try drafts.create()
        draft.manifest.photos = try PhotoImporter.importPhotos(from: [photos[0]], into: draft.directory).photos
        draft = try ProjectStore.commitModel(from: newModel, to: draft)
        let draftManifest = try Data(contentsOf: draft.directory.appendingPathComponent("project.json"))
        let publishedURL = volume.appendingPathComponent("Published.rebuild3d")

        let filler = volume.appendingPathComponent("disk-full-probe-filler.bin")
        defer { try? fm.removeItem(at: filler) }
        let filledBytes = try fill(filler)
        var checks: [[String: Any]] = []
        func expectDiskFull(_ name: String, _ operation: () throws -> Void) throws {
            do { try operation() }
            catch {
                let error = error as NSError
                try require(isOutOfSpace(error), "\(name) failed for a reason other than ENOSPC: \(error)")
                checks.append(["operation": name, "domain": error.domain, "code": error.code,
                               "message": error.localizedDescription, "outOfSpace": true])
                print("Expected disk-full failure: \(name)")
                fflush(nil)
                return
            }
            throw ProjectError.invalid("\(name) unexpectedly succeeded; the full-disk condition was not demonstrated.")
        }
        var changed = project
        changed.manifest.name = "Unsaved change"
        try expectDiskFull("save manifest") { try ProjectStore.save(changed) }
        try expectDiskFull("commit model") { _ = try ProjectStore.commitModel(from: newModel, to: project) }
        try expectDiskFull("replace export") { try ProjectStore.exportModel(draft, to: exportURL) }
        try expectDiskFull("publish draft") { _ = try drafts.saveAs(draft, to: publishedURL) }
        let rejectedImport = try PhotoImporter.importPhotos(from: [photos[1]], into: project.directory)
        try require(rejectedImport.photos.isEmpty && rejectedImport.issues.count == 1,
                    "The full-disk import must not add a partial photo.")
        checks.append(["operation": "import photo", "added": 0, "issues": rejectedImport.issues.map(\.reason)])
        try JSONSerialization.data(withJSONObject: ["capacityBytes": capacity, "fillerBytes": filledBytes, "checks": checks],
                                   options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("injected-failures.json"), options: .atomic)

        try require(try Data(contentsOf: manifestURL) == beforeManifest, "The saved manifest changed after a failed write.")
        try require(try Data(contentsOf: project.modelURL!) == beforeModel, "The saved model changed after a failed write.")
        try require(try Data(contentsOf: exportURL) == beforeExport, "A failed export replaced the previous file.")
        try require(try Data(contentsOf: draft.directory.appendingPathComponent("project.json")) == draftManifest,
                    "A failed publication changed the source draft.")
        try require(!fm.fileExists(atPath: publishedURL.path), "A failed publication left a visible destination.")
        try require(try fm.contentsOfDirectory(atPath: project.directory.appendingPathComponent("images").path).sorted() == imageFiles,
                    "A failed import left orphaned image files.")
        try require(try fm.contentsOfDirectory(atPath: project.directory.appendingPathComponent("models").path).sorted() == modelFiles,
                    "A failed commit left orphaned model files.")
        for photo in project.manifest.photos {
            try require(try PhotoInspector.contentSHA256(at: project.directory.appendingPathComponent(photo.imagePath)) == photo.metadata?.contentSHA256,
                        "A saved original changed during the failed operations.")
        }
        _ = try ProjectStore.open(project.directory)
        _ = try ProjectStore.open(draft.directory)

        // Releasing only our filler must allow the very same operations to recover.
        try fm.removeItem(at: filler)
        try ProjectStore.save(changed)
        let committed = try ProjectStore.commitModel(from: newModel, to: changed)
        try ProjectStore.exportModel(committed, to: exportURL)
        let published = try drafts.saveAs(draft, to: publishedURL)
        let imported = try PhotoImporter.importPhotos(from: [photos[1]], into: committed.directory)
        try require(imported.photos.count == 1 && imported.issues.isEmpty, "Import did not recover after space was freed.")
        var recovered = committed
        recovered.manifest.photos.append(contentsOf: imported.photos)
        try ProjectStore.save(recovered)
        _ = try ProjectStore.open(recovered.directory)
        _ = try ProjectStore.open(published.directory)
        try require(try Data(contentsOf: exportURL) == Data(contentsOf: newModel), "Export retry was not byte-preserving.")
        try require(fm.fileExists(atPath: draft.directory.path), "The store must retain the draft until its caller installs the copy.")
        let report: [String: Any] = ["kind": "bounded-disk-full-storage-probe", "capacityBytes": capacity,
            "fillerBytes": filledBytes, "checks": checks, "oldDataPreserved": true, "retryPassed": true,
            "limitations": ["Opaque model fixtures; not a model-quality test", "Storage/service checks; not a UI acceptance test"]]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("report.json"), options: .atomic)
        print("Old data preserved; all retries passed after freeing scratch space.")
    }
}
