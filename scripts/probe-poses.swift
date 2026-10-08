// P1 technical probe. This does not enable photo-aligned viewing in the application.
// Build: swiftc -parse-as-library Sources/Rebuild3DCore/*.swift scripts/probe-poses.swift -o build/pose-probe
import Foundation
import RealityKit
import simd

@main
struct PoseProbe {
    static func main() async throws {
        let originalInputs = CommandLine.arguments.count == 4 && CommandLine.arguments[3] == "--original-inputs"
        guard CommandLine.arguments.count == 3 || originalInputs else {
            print("Usage: pose-probe PROJECT.rebuild3d NEW_OUTPUT_DIRECTORY [--original-inputs]")
            exit(2)
        }
        let project = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let output = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw NSError(domain: "PoseProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Output directory already exists."])
        }
        let sourceProject = try ProjectStore.open(project)
        let photos = sourceProject.manifest.photos
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let input: URL
        var cleanup: URL?
        defer { if let cleanup { try? FileManager.default.removeItem(at: cleanup) } }
        var photosByStagedURL: [URL: [String: Any]] = [:]
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if originalInputs {
            input = output.appendingPathComponent("input")
            cleanup = input
            try FileManager.default.createDirectory(at: input, withIntermediateDirectories: false)
            for photo in photos {
                let original = try ProjectStore.resolve(photo.imagePath, in: project)
                let staged = input.appendingPathComponent(original.lastPathComponent)
                try FileManager.default.linkItem(at: original, to: staged)
                photosByStagedURL[staged.resolvingSymlinksInPath()] = try JSONSerialization.jsonObject(with: encoder.encode(photo)) as? [String: Any]
            }
        } else {
            let runID = UUID()
            cleanup = try ProjectStore.resolve("cache/\(runID.uuidString)", in: project)
            let snapshot = try PhotoPreparation.stageInputs(project: sourceProject, runID: runID)
            input = cleanup!.appendingPathComponent("input")
            try encoder.encode(snapshot).write(to: output.appendingPathComponent("inputs.json"), options: .atomic)
            for prepared in snapshot.inputs {
                var record = try JSONSerialization.jsonObject(with: encoder.encode(prepared.photo)) as! [String: Any]
                record["inputWidth"] = prepared.preparation.inputWidth
                record["inputHeight"] = prepared.preparation.inputHeight
                record["inputOrientation"] = prepared.preparation.inputOrientation
                record["preparation"] = try JSONSerialization.jsonObject(with: encoder.encode(prepared.preparation))
                photosByStagedURL[try ProjectStore.resolve(prepared.stagedPath, in: project)] = record
            }
        }
        var configuration = PhotogrammetrySession.Configuration()
        configuration.isObjectMaskingEnabled = sourceProject.manifest.settings.objectMasking
        let session = try PhotogrammetrySession(input: input, configuration: configuration)
        let model = output.appendingPathComponent("model.usdz")
        var records: [[String: Any]] = []
        var events: [String] = []
        var failures: [String] = []
        var sampleIssues: [Int: String] = [:]
        var modelCompleted = false
        var posesCompleted = false
        var sessionCompleted = false
        let started = Date()
        let detail: PhotogrammetrySession.Request.Detail = sourceProject.manifest.settings.quality == .reduced ? .reduced : .medium
        try session.process(requests: [.modelFile(url: model, detail: detail), .poses])
        outputLoop: for try await event in session.outputs {
            switch event {
            case .requestComplete(_, let result):
                switch result {
                case .modelFile:
                    modelCompleted = true
                    events.append("model completed")
                case .poses(let poses):
                    posesCompleted = true
                    events.append("poses completed")
                    for (sampleID, url) in poses.urlsBySample.sorted(by: { $0.key < $1.key }) {
                        let staged = url.resolvingSymlinksInPath().standardizedFileURL
                        guard let photo = photosByStagedURL[staged] else {
                            failures.append("Unmapped sample \(sampleID): \(url.path)")
                            continue
                        }
                        var record = photo
                        record["sampleID"] = sampleID
                        record["inputURL"] = url.path
                        if let pose = poses.posesBySample[sampleID] {
                            record["status"] = "located"
                            record["rawPoseTransformColumnMajor"] = columns(pose.transform.matrix)
                            record["translation"] = [pose.translation.x, pose.translation.y, pose.translation.z]
                            let q = pose.rotation.vector
                            record["quaternionXYZW"] = [q.x, q.y, q.z, q.w]
                            if let k = pose.intrinsics {
                                record["intrinsicsColumnMajor"] = (0..<3).flatMap { c in (0..<3).map { r in k[c][r] } }
                            }
                            if let lens = pose.lensDistortionData {
                                record["lensDistortion"] = ["center": [lens.center.x, lens.center.y], "radialLookupTable": lens.radialLookupTable]
                            }
                        } else {
                            record["status"] = "unlocated"
                        }
                        records.append(record)
                    }
                default: break
                }
                print(events.last ?? "Request complete")
                fflush(nil)
            case .invalidSample(let id, let reason): sampleIssues[id] = reason
            case .skippedSample(let id): sampleIssues[id] = "skipped"
            case .automaticDownsampling: events.append("automatic downsampling reported")
            case .requestError(let request, let error): failures.append("\(request): \(error.localizedDescription)")
            case .processingComplete:
                sessionCompleted = true
                events.append("session completed")
                break outputLoop
            case .processingCancelled: throw CancellationError()
            default: break
            }
        }
        for index in records.indices {
            if let id = records[index]["sampleID"] as? Int, let issue = sampleIssues[id] { records[index]["sampleIssue"] = issue }
        }
        let mappedIDs = Set(records.compactMap { $0["id"] as? String })
        let missingPhotos = photos.filter { !mappedIDs.contains($0.id.uuidString) }.map(\.originalName)
        if !missingPhotos.isEmpty { failures.append("Photos absent from the engine URL map: \(missingPhotos.joined(separator: ", "))") }
        let report: [String: Any] = [
            "formatVersion": 2,
            "kind": "technical-probe",
            "inputMode": originalInputs ? "originals" : "production-preparation",
            "settings": try JSONSerialization.jsonObject(with: encoder.encode(sourceProject.manifest.settings)),
            "coordinateConvention": "Raw PhotogrammetrySession pose, column-major; viewer conversion not yet verified.",
            "sourceProjectID": sourceProject.manifest.id.uuidString,
            "elapsedSeconds": Date().timeIntervalSince(started),
            "modelCompleted": modelCompleted,
            "posesCompleted": posesCompleted,
            "sessionCompleted": sessionCompleted,
            "events": events,
            "failures": failures,
            "records": records
        ]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("cameras-probe.json"), options: .atomic)
        print("Located \(records.filter { $0["status"] as? String == "located" }.count) / \(photos.count) photos")
        print("Intrinsics available: \(records.filter { $0["intrinsicsColumnMajor"] != nil }.count)")
        print("Distortion available: \(records.filter { $0["lensDistortion"] != nil }.count)")
        print("Failures: \(failures)")
        guard modelCompleted && posesCompleted && sessionCompleted && failures.isEmpty else { exit(1) }
    }

    static func columns(_ matrix: simd_float4x4) -> [Float] {
        (0..<4).flatMap { c in (0..<4).map { r in matrix[c][r] } }
    }
}
