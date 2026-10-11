// Compile with the unchanged production Sources/Rebuild3DCore/*.swift.
// Runs the application's GenerationCoordinator, not a replacement pipeline.
// CLI evidence only: this cannot prove button, viewport or app-restart behavior.
import Foundation
import Darwin

actor RecoveryEvents {
    let output: URL
    init(_ output: URL) { self.output = output }

    func receive(_ event: ReconstructionEvent) throws {
        var record: [String: Any] = ["atUTC": ISO8601DateFormatter().string(from: Date())]
        switch event {
        case .stage(let stage): record["kind"] = "stage"; record["message"] = stage.message
        case .message(let message): record["kind"] = "message"; record["message"] = message
        case .diagnostic(let message): record["kind"] = "diagnostic"; record["message"] = message
        case .progress(let value): record["kind"] = "progress"; record["value"] = value
        }
        let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) + Data([10])
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try FileHandle.standardOutput.write(contentsOf: data)
    }
}

@main struct GenerationRecoveryProbe {
    static func save(_ report: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .atomic)
    }

    @MainActor static func main() async {
        let args = CommandLine.arguments
        guard args.count == 5 || (args.count == 6 && args[5] == "verify-saved") else {
            fputs("Usage: probe-generation-recovery PROJECT RUNTIME_SEED ISOLATED_STORE NEW_PHASE_OUTPUT [verify-saved]\n", stderr)
            exit(2)
        }
        let projectURL = URL(fileURLWithPath: args[1]).standardizedFileURL
        let seed = URL(fileURLWithPath: args[2]).standardizedFileURL
        let storage = URL(fileURLWithPath: args[3]).standardizedFileURL
        let output = URL(fileURLWithPath: args[4]).standardizedFileURL
        let reportURL = output.appendingPathComponent("report.json")
        let fm = FileManager.default
        let verifySavedOnly = args.count == 6
        guard !fm.fileExists(atPath: output.path) else {
            fputs("Phase output already exists; evidence must not be overwritten.\n", stderr)
            exit(2)
        }
        var report: [String: Any] = ["scope": "Production GenerationCoordinator in a CLI process; not actual application UI acceptance",
            "status": "running", "mode": verifySavedOnly ? "verify-saved" : "generate",
            "project": projectURL.path, "runtimeSeed": seed.path,
            "runtimeStorage": storage.path, "pid": getpid()]
        var cancellationWatcher: Task<Void, Never>?
        var status: Int32 = 1
        do {
            try fm.createDirectory(at: output, withIntermediateDirectories: true)
            try Data().write(to: output.appendingPathComponent("events.jsonl"))
            let manifestData = try Data(contentsOf: seed.appendingPathComponent("runtime.json"))
            report["runtimeManifestSHA256"] = GenerationRuntime.digest(manifestData)
            let project = try ProjectStore.open(projectURL)
            report["beforeModelPath"] = project.modelURL?.path
            try save(report, to: reportURL)
            let engine = GenerationCoordinator(runtimeSeed: seed, runtimeStorage: storage)
            let events = RecoveryEvents(output.appendingPathComponent("events.jsonl"))
            let cancelFile = output.appendingPathComponent("request-cancel")
            cancellationWatcher = Task {
                while !Task.isCancelled {
                    if fm.fileExists(atPath: cancelFile.path) {
                        await engine.cancel()
                        return
                    }
                    do { try await Task.sleep(for: .milliseconds(50)) } catch { return }
                }
            }
            let result: Project
            if verifySavedOnly {
                result = project
            } else {
                result = try await engine.run(project: project) { event in
                    try? await events.receive(event)
                }
            }
            cancellationWatcher?.cancel()
            let reopened = try ProjectStore.open(projectURL)
            // ProjectStore's documented persisted representation uses ISO 8601
            // seconds. Compare every serialized field, not Date's discarded
            // in-memory fractional seconds.
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.sortedKeys]
            let expected = try encoder.encode(result.manifest.model)
            let actual = try encoder.encode(reopened.manifest.model)
            report["modelRecordComparison"] = [
                "method": "All fields encoded with ProjectStore's ISO 8601 date policy",
                "expectedSHA256": GenerationRuntime.digest(expected),
                "reopenedSHA256": GenerationRuntime.digest(actual),
                "inMemoryCreatedAtUnixSeconds": result.manifest.model?.createdAt.timeIntervalSince1970 as Any,
                "reopenedCreatedAtUnixSeconds": reopened.manifest.model?.createdAt.timeIntervalSince1970 as Any]
            guard expected == actual, let modelURL = reopened.modelURL else {
                throw ProjectError.invalid("Published result did not survive ProjectStore.open")
            }
            report["afterModelPath"] = modelURL.path
            report["afterModelSHA256"] = try PhotoInspector.contentSHA256(at: modelURL)
            report["savedResultReopenedByProjectStore"] = true
            let exported = output.appendingPathComponent("recovered.usdz")
            try ProjectStore.exportModel(reopened, to: exported)
            report["exportedUSDZ"] = exported.path
            report["exportedSHA256"] = try PhotoInspector.contentSHA256(at: exported)
            report["status"] = "completed"
            status = 0
        } catch is CancellationError {
            report["status"] = "cancelled"
            status = 130
        } catch {
            report["status"] = "failed"
            report["error"] = String(describing: error)
            status = 1
        }
        cancellationWatcher?.cancel()
        report["finishedAtUTC"] = ISO8601DateFormatter().string(from: Date())
        report["exitCode"] = status
        do { try save(report, to: reportURL) }
        catch { fputs("Could not write final probe report: \(error)\n", stderr); exit(2) }
        exit(status)
    }
}
