// Compile alongside the unchanged production Sources/Rebuild3DCore/*.swift.
// Exercises component preparation only; it does not launch the app or inference.
import Foundation
import Darwin

actor ComponentEventLog {
    let url: URL
    init(_ url: URL) { self.url = url }
    func record(phase: String, event: ReconstructionEvent) throws {
        guard case .message(let message) = event else { return }
        let record: [String: Any] = ["phase": phase, "message": message,
                                     "at": ISO8601DateFormatter().string(from: Date())]
        let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        if !FileManager.default.fileExists(atPath: url.path) {
            try Data().write(to: url)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data + Data([10]))
    }
}

@main struct ComponentInstallProbe {
    static func save(_ report: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    static func emit(_ phase: String) {
        try? FileHandle.standardOutput.write(contentsOf: Data((phase + "\n").utf8))
    }

    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw ProjectError.invalid(message) }
    }

    static func size(_ url: URL) throws -> Int {
        let value = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber
        guard let value else { throw ProjectError.invalid("Missing file size: \(url.path)") }
        return value.intValue
    }

    static func treeBytes(_ root: URL) throws -> Int {
        guard let entries = FileManager.default.enumerator(at: root,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            throw ProjectError.invalid("Cannot enumerate component directory")
        }
        var bytes = 0
        for case let url as URL in entries {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            if values.isRegularFile == true { bytes += values.fileSize ?? 0 }
        }
        return bytes
    }

    static func verify(_ runtime: URL, manifest: RuntimeManifest) throws {
        for item in manifest.artifacts {
            let path = try GenerationRuntime.resolve(item.path, in: runtime)
            try require(try GenerationRuntime.matches(path, item), "Component mismatch: \(item.path)")
        }
    }

    static func prepared(seed: URL, storage: URL, phase: String,
                         cancellation: GenerationCancellation = .init(), log: ComponentEventLog) async throws -> (GenerationRuntime, Double) {
        emit("Starting " + phase)
        let start = ContinuousClock.now
        let runtime = try await GenerationRuntime.prepare(seed: seed, storage: storage,
            cancellation: cancellation, onEvent: { event in try? await log.record(phase: phase, event: event) })
        let duration = start.duration(to: .now).components
        let seconds = Double(duration.seconds) + Double(duration.attoseconds) / 1e18
        emit("Completed \(phase): \(seconds) seconds")
        return (runtime, seconds)
    }

    static func clone(_ source: URL, to target: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/cp")
        process.arguments = ["-cR", source.path, target.path]
        try process.run()
        process.waitUntilExit()
        try require(process.terminationStatus == 0, "Could not seed the isolated upgrade store")
    }

    @MainActor static func main() async {
        var report: [String: Any] = [
            "scope": "Production component-preparation implementation, CLI only; not application UI acceptance",
            "status": "running", "byteAccounting": "logical file bytes; APFS shared blocks are not added as new physical allocation"
        ]
        var reportURL: URL?
        do {
            let args = CommandLine.arguments
            try require(args.count == 4, "Usage: probe-component-install CANDIDATE_RUNTIME OLD_RUNTIME NEW_OUTPUT")
            let seed = URL(fileURLWithPath: args[1]).standardizedFileURL
            let oldSeed = URL(fileURLWithPath: args[2]).standardizedFileURL
            let output = URL(fileURLWithPath: args[3]).standardizedFileURL
            let fm = FileManager.default
            try require(!fm.fileExists(atPath: output.path), "Probe output must be new")
            try fm.createDirectory(at: output, withIntermediateDirectories: true)
            reportURL = output.appendingPathComponent("report.json")
            let log = ComponentEventLog(output.appendingPathComponent("events.jsonl"))
            let manifestData = try Data(contentsOf: seed.appendingPathComponent("runtime.json"))
            let oldManifestData = try Data(contentsOf: oldSeed.appendingPathComponent("runtime.json"))
            let manifest = try JSONDecoder().decode(RuntimeManifest.self, from: manifestData)
            let oldManifest = try JSONDecoder().decode(RuntimeManifest.self, from: oldManifestData)
            let identity = GenerationRuntime.digest(manifestData)
            let oldIdentity = GenerationRuntime.digest(oldManifestData)
            report["candidateRuntimeManifestSHA256"] = identity
            report["oldRuntimeManifestSHA256"] = oldIdentity
            report["productionSourceSHA256"] = try PhotoInspector.contentSHA256(at:
                URL(fileURLWithPath: "Sources/Rebuild3DCore/GenerationRuntime.swift"))
            report["candidateSeed"] = seed.path
            report["oldSeed"] = oldSeed.path
            try save(report, to: reportURL!)

            let (fresh, firstSeconds) = try await prepared(seed: seed,
                storage: output.appendingPathComponent("fresh-store"), phase: "fresh preparation", log: log)
            try verify(fresh.directory, manifest: manifest)
            let installedBytes = try treeBytes(fresh.directory)
            report["freshPreparation"] = ["seconds": firstSeconds, "identity": fresh.identity,
                "verifiedArtifacts": manifest.artifacts.count, "installedLogicalBytes": installedBytes]
            try save(report, to: reportURL!)

            let corrupt = try GenerationRuntime.resolve("worker/rebuild3d_worker/failures.py", in: fresh.directory)
            try Data("deliberately corrupt test copy".utf8).write(to: corrupt, options: .atomic)
            let (repaired, repairSeconds) = try await prepared(seed: seed,
                storage: output.appendingPathComponent("fresh-store"), phase: "repair corrupted installed file", log: log)
            try verify(repaired.directory, manifest: manifest)
            try require(repaired.identity == fresh.identity, "Repair changed the component identity")
            report["repair"] = ["seconds": repairSeconds, "allArtifactsVerified": true,
                                 "damagedPath": "worker/rebuild3d_worker/failures.py"]
            try save(report, to: reportURL!)

            let upgradeStore = output.appendingPathComponent("upgrade-store")
            try fm.createDirectory(at: upgradeStore, withIntermediateDirectories: true)
            let oldInstalled = upgradeStore.appendingPathComponent(oldIdentity)
            emit("Seeding isolated old runtime with an APFS clone")
            try clone(oldSeed, to: oldInstalled)
            try verify(oldInstalled, manifest: oldManifest)
            let oldBytes = try treeBytes(oldInstalled)
            let destination = upgradeStore.appendingPathComponent(identity)
            let partial = destination.appendingPathComponent("models/vggt-1b.safetensors.partial")
            let expectedWeightBytes = try size(seed.appendingPathComponent("models/vggt-1b.safetensors"))
            let cancellation = GenerationCancellation()
            let observer = Task.detached { () -> Bool in
                let deadline = Date().addingTimeInterval(600)
                while !Task.isCancelled && Date() < deadline {
                    if let bytes = try? size(partial), bytes >= 16 * 1024 * 1024 && bytes < expectedWeightBytes {
                        cancellation.cancel()
                        return true
                    }
                    try? await Task.sleep(for: .milliseconds(5))
                }
                if !Task.isCancelled { cancellation.cancel() }
                return false
            }
            var caughtCancellation = false
            let cancellationStart = ContinuousClock.now
            do {
                _ = try await prepared(seed: seed, storage: upgradeStore,
                    phase: "cancel during real weight copy", cancellation: cancellation, log: log)
            } catch is CancellationError { caughtCancellation = true }
            catch { observer.cancel(); throw error }
            observer.cancel()
            let observed = await observer.value
            try require(caughtCancellation && observed, "Did not observe and cancel a real partial weight copy")
            let partialBytes = try size(partial)
            try require(partialBytes > 0 && partialBytes < expectedWeightBytes, "Expected an incomplete retained weight prefix")
            try require(!fm.fileExists(atPath: destination.appendingPathComponent("runtime.json").path),
                        "Cancelled preparation must not commit a completed runtime manifest")
            let cancelDuration = cancellationStart.duration(to: .now).components
            report["cancellation"] = ["caughtCancellationError": true, "retainedPartialBytes": partialBytes,
                "expectedWeightBytes": expectedWeightBytes, "noCompletedRuntimeManifest": true,
                "seconds": Double(cancelDuration.seconds) + Double(cancelDuration.attoseconds) / 1e18]
            try save(report, to: reportURL!)

            let (resumed, resumeSeconds) = try await prepared(seed: seed, storage: upgradeStore,
                phase: "resume interrupted preparation", log: log)
            try verify(resumed.directory, manifest: manifest)
            try require(!fm.fileExists(atPath: partial.path), "Completed weight still has a partial file")
            try verify(oldInstalled, manifest: oldManifest)
            try require(try Data(contentsOf: oldInstalled.appendingPathComponent("runtime.json")) == oldManifestData,
                        "Previous runtime manifest changed")
            report["resume"] = ["seconds": resumeSeconds, "allArtifactsVerified": true,
                "previousRuntimeArtifactsVerifiedUnchanged": oldManifest.artifacts.count,
                "partialFilePromoted": true]

            let (_, healthySeconds) = try await prepared(seed: seed, storage: upgradeStore,
                phase: "prepare healthy installed components again", log: log)
            try verify(resumed.directory, manifest: manifest)
            report["healthyRevalidationSeconds"] = healthySeconds
            report["storage"] = ["candidateRuntimeLogicalBytes": installedBytes,
                "oldRuntimeLogicalBytes": oldBytes, "isolatedUpgradeStoreLogicalBytesAfterPreparation": try treeBytes(upgradeStore),
                "scope": "Component store only; app/ZIP and actual application upgrade peak are reported separately"]
            report["seedRuntimeManifestUnchanged"] = try Data(contentsOf: seed.appendingPathComponent("runtime.json")) == manifestData
            report["status"] = "component-checks-passed"
            report["applicationAcceptance"] = "pending: UI, GPU, save/export, full reconstruction, actual installation/upgrade peak"
            try save(report, to: reportURL!)
            emit("Component checks passed: " + reportURL!.path)
        } catch {
            report["status"] = "failed"
            report["error"] = String(describing: error)
            if let reportURL { try? save(report, to: reportURL) }
            try? FileHandle.standardError.write(contentsOf: Data(("Component probe failed: \(error)\n").utf8))
            exit(1)
        }
    }
}
