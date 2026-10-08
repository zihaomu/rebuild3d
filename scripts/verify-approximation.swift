// Exercise the real research result through the same persistence operations used by the app.
import Foundation
import RealityKit

@main
struct VerifyApproximation {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 4 else {
            print("Usage: verify-approximation OLD_PROJECT RESULT_DIRECTORY NEW_DELIVERY_DIRECTORY")
            exit(2)
        }
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let result = URL(fileURLWithPath: CommandLine.arguments[2])
        let delivery = URL(fileURLWithPath: CommandLine.arguments[3])
        let fm = FileManager.default
        guard !fm.fileExists(atPath: delivery.path) else { throw ProjectError.invalid("Delivery must be new") }
        let originalManifest = try Data(contentsOf: source.appendingPathComponent("project.json"))
        let old = try ProjectStore.open(source)
        let identities = old.manifest.photos.map(\.id)
        try fm.createDirectory(at: delivery, withIntermediateDirectories: true)
        let projectURL = delivery.appendingPathComponent("Seven-photo-statue.rebuild3d")
        try fm.copyItem(at: source, to: projectURL)
        var project = try ProjectStore.open(projectURL)
        project.manifest.name = "Seven-photo statue (approximate)"
        project = try ApproximateResultStore.importResult(from: result, into: project)
        try ProjectStore.save(project)
        project = try ProjectStore.open(projectURL)
        guard project.manifest.photos.map(\.id) == identities else { fatalError("Photo IDs changed") }
        let model = project.modelURL!
        let provenance = project.provenanceModelURL!
        // Use RealityKit's actual USDZ loader, in addition to the independent SceneKit renders.
        let entity = try await Entity(contentsOf: model)
        let regions = try await Entity(contentsOf: provenance)
        let bounds = entity.visualBounds(relativeTo: nil)
        let sourceBounds = regions.visualBounds(relativeTo: nil)
        guard bounds.extents.x > 0, bounds.extents.y > 0, bounds.extents.z > 0,
              length(bounds.extents - sourceBounds.extents) < 0.0001 else { fatalError("Invalid geometry bounds") }
        let exported = delivery.appendingPathComponent("Seven-photo-statue.usdz")
        try ProjectStore.exportModel(project, to: exported)
        let exportedBundle = exported.deletingPathExtension().appendingPathExtension("rebuild3d-result")
        let exportedInfo = try ApproximateResultStore.load(exportedBundle)
        let reopened = try ProjectStore.open(projectURL)
        let expected = try PhotoInspector.contentSHA256(at: model)
        guard try PhotoInspector.contentSHA256(at: exported) == expected,
              try PhotoInspector.contentSHA256(at: reopened.modelURL!) == expected,
              try Data(contentsOf: source.appendingPathComponent("project.json")) == originalManifest else {
            fatalError("Persistence verification failed")
        }
        for photo in reopened.manifest.photos {
            let original = try ProjectStore.resolve(photo.imagePath, in: reopened.directory)
            guard try PhotoInspector.contentSHA256(at: original) == photo.metadata?.contentSHA256 else {
                fatalError("Original photo changed")
            }
        }
        let report: [String: Any] = ["status": "passed", "project": projectURL.path, "export": exported.path,
            "projectFormat": reopened.manifest.formatVersion, "photoCount": reopened.manifest.photos.count,
            "originalPhotoIDsPreserved": true, "oldProjectManifestUnchanged": true,
            "savedAndReopened": true, "exportMatchesSavedModel": true,
            "reimportableCompanionVerified": true, "realityKitLoadedBothModels": true,
            "relativeBounds": [Double(bounds.extents.x), Double(bounds.extents.y), Double(bounds.extents.z)],
            "modelSHA256": expected, "triangleCount": exportedInfo.triangleCount,
            "completionTriangleCount": exportedInfo.completionTriangleCount,
            "limitation": "Core/RealityKit loader verification; native app UI interaction still requires a visible desktop."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: delivery.appendingPathComponent("verification.json"), options: .atomic)
        print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted]), encoding: .utf8)!)
    }
}
