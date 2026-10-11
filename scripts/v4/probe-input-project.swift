// Compile with unchanged Sources/Rebuild3DCore/*.swift. CLI import evidence only.
import Foundation

struct MatrixInput: Decodable {
    struct Source: Decodable { let path: String; let sha256: String }
    let sources: [Source]
}

@main struct InputProjectProbe {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            fputs("Usage: probe-input-project INPUT_LIST_JSON NEW_PROJECT\n", stderr)
            exit(2)
        }
        let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let projectURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let input = try JSONDecoder().decode(MatrixInput.self, from: Data(contentsOf: inputURL))
        let urls = input.sources.map { URL(fileURLWithPath: $0.path) }
        for (source, url) in zip(input.sources, urls) {
            guard try PhotoInspector.contentSHA256(at: url) == source.sha256 else {
                throw ProjectError.invalid("Matrix source changed before import")
            }
        }
        var project = try ProjectStore.create(at: projectURL)
        let imported = try PhotoImporter.importPhotos(from: urls, into: project.directory)
        guard imported.issues.isEmpty, imported.photos.count == input.sources.count else {
            throw ProjectError.invalid("Matrix import failed: \(imported.summary)")
        }
        for (photo, source) in zip(imported.photos, input.sources) {
            guard photo.metadata?.contentSHA256 == source.sha256,
                  photo.originalName == URL(fileURLWithPath: source.path).lastPathComponent else {
                throw ProjectError.invalid("Imported photo identity/order mismatch")
            }
        }
        project.manifest.photos = imported.photos
        try ProjectStore.save(project)
        let reopened = try ProjectStore.open(projectURL)
        guard reopened.manifest.photos == imported.photos else {
            throw ProjectError.invalid("Imported photos did not survive ProjectStore.open")
        }
        let report: [String: Any] = [
            "status": "production-import-and-reopen-passed", "photoCount": imported.photos.count,
            "sourceListSHA256": try PhotoInspector.contentSHA256(at: inputURL),
            "projectManifestSHA256": try PhotoInspector.contentSHA256(at: projectURL.appendingPathComponent("project.json")),
            "project": projectURL.path,
            "scope": "Production PhotoImporter and ProjectStore; no UI actions"]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: projectURL.appendingPathComponent("logs/matrix-import.json"), options: .atomic)
        try FileHandle.standardOutput.write(contentsOf: data + Data([10]))
    }
}
