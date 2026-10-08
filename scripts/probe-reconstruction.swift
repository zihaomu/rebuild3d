// Diagnostic only; never changes source images or application defaults.
// Build: swiftc -parse-as-library scripts/probe-reconstruction.swift -o build/reconstruction-probe
import CryptoKit
import CoreVideo
import Foundation
import ImageIO
import RealityKit

@main
struct ReconstructionProbe {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        let modes = ["folder", "samples", "samples-no-depth", "samples-no-gravity"]
        guard args.count == 5, modes.contains(args[2]), ["on", "off"].contains(args[3]),
              ["normal", "high"].contains(args[4]) else {
            print("Usage: reconstruction-probe INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY MODE MASK(on|off) SENSITIVITY(normal|high)")
            print("MODE: \(modes.joined(separator: ", "))")
            exit(2)
        }
        let input = URL(fileURLWithPath: args[0]).standardizedFileURL.resolvingSymlinksInPath()
        let output = URL(fileURLWithPath: args[1]).standardizedFileURL.resolvingSymlinksInPath()
        let fm = FileManager.default
        guard !fm.fileExists(atPath: output.path), !input.path.hasPrefix(output.path + "/") else {
            throw NSError(domain: "ReconstructionProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Use a new output directory outside the input's ancestors."])
        }
        let urls = try fm.contentsOfDirectory(at: input, includingPropertiesForKeys: [.isRegularFileKey])
            .filter { url in
                (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true &&
                CGImageSourceCreateWithURL(url as CFURL, nil) != nil
            }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else { throw NSError(domain: "ReconstructionProbe", code: 2) }
        try fm.createDirectory(at: output, withIntermediateDirectories: true)
        var config = PhotogrammetrySession.Configuration()
        config.isObjectMaskingEnabled = args[3] == "on"
        config.featureSensitivity = args[4] == "high" ? .high : .normal
        config.sampleOrdering = .unordered
        var inventory: [[String: Any]] = []
        var samples: [PhotogrammetrySample] = []
        for url in urls {
            var record: [String: Any] = ["name": url.lastPathComponent, "sha256": try digest(url)]
            if args[2] != "folder" {
                var sample = try await PhotogrammetrySample(contentsOf: url)
                record["sampleID"] = sample.id
                record["width"] = CVPixelBufferGetWidth(sample.image)
                record["height"] = CVPixelBufferGetHeight(sample.image)
                record["orientation"] = sample.orientation.rawValue
                record["hadDepth"] = sample.depthDataMap != nil
                record["hadDepthConfidence"] = sample.depthConfidenceMap != nil
                record["hadGravity"] = sample.gravity != nil
                record["hadObjectMask"] = sample.objectMask != nil
                record["hadCamera"] = sample.camera != nil
                record["hadBoundingBox"] = sample.boundingBox != nil
                if args[2] == "samples-no-depth" {
                    sample.depthDataMap = nil
                }
                if args[2] == "samples-no-gravity" {
                    sample.gravity = nil
                }
                samples.append(sample)
            }
            inventory.append(record)
        }
        let session: PhotogrammetrySession
        if args[2] == "folder" {
            session = try PhotogrammetrySession(input: input, configuration: config)
        } else {
            session = try PhotogrammetrySession(input: samples, configuration: config)
        }
        let started = Date()
        var events: [[String: Any]] = []
        var failures: [[String: Any]] = []
        var locatedIDs: [Int] = []
        var mappedURLs: [String: String] = [:]
        var modelComplete = false
        var posesComplete = false
        var processingComplete = false
        var cancelled = false
        var lastProgress = -1
        var lastStage: String?
        func record(_ text: String) {
            let elapsed = Date().timeIntervalSince(started)
            events.append(["elapsedSeconds": elapsed, "event": text])
            print(String(format: "%.3f %@", elapsed, text))
            fflush(nil)
        }
        let deadline = Task {
            do { try await Task.sleep(for: .seconds(180)) } catch { return }
            session.cancel()
        }
        defer { deadline.cancel() }
        let model = output.appendingPathComponent("model.usdz")
        do {
            try session.process(requests: [.modelFile(url: model, detail: .reduced), .poses])
            outputLoop: for try await event in session.outputs {
                switch event {
                case .requestProgress(_, let fraction):
                    let step = Int(fraction * 20)
                    if step != lastProgress { record("progress \(fraction)"); lastProgress = step }
                case .requestProgressInfo(_, let info):
                    if let stage = info.processingStage {
                        let name = String(describing: stage)
                        if name != lastStage { lastStage = name; record("stage \(name)") }
                    }
                case .requestComplete(_, let result):
                    switch result {
                    case .modelFile: modelComplete = true; record("model completed")
                    case .poses(let poses):
                        posesComplete = true
                        locatedIDs = poses.posesBySample.keys.sorted()
                        mappedURLs = Dictionary(uniqueKeysWithValues: poses.urlsBySample.map { (String($0.key), $0.value.lastPathComponent) })
                        record("poses completed: \(locatedIDs.count) located, \(mappedURLs.count) URLs")
                    default: record("result \(result)")
                    }
                case .requestError(let request, let error):
                    let ns = error as NSError
                    failures.append(["request": String(describing: request), "error": String(reflecting: error),
                        "domain": ns.domain, "code": ns.code, "localizedDescription": ns.localizedDescription,
                        "userInfo": ns.userInfo.mapValues { String(describing: $0) }])
                    record("requestError \(request): \(String(reflecting: error))")
                case .processingComplete: processingComplete = true; record("processing complete"); break outputLoop
                case .processingCancelled: cancelled = true; record("processing cancelled"); break outputLoop
                default: record(String(describing: event))
                }
            }
        } catch {
            failures.append(["streamError": String(reflecting: error)])
            session.cancel()
        }
        let elapsed = Date().timeIntervalSince(started)
        let unchanged = try zip(urls, inventory).allSatisfy { try digest($0.0) == $0.1["sha256"] as? String }
        let report: [String: Any] = ["input": input.path, "mode": args[2], "mask": args[3], "sensitivity": args[4],
            "os": ProcessInfo.processInfo.operatingSystemVersionString, "elapsedSeconds": elapsed,
            "inventory": inventory, "sourceUnchanged": unchanged, "events": events, "failures": failures,
            "locatedSampleIDs": locatedIDs, "mappedURLs": mappedURLs, "modelComplete": modelComplete,
            "posesComplete": posesComplete, "processingComplete": processingComplete, "cancelled": cancelled,
            "lastProcessingStage": lastStage ?? "unknown", "deadlineSeconds": 180,
            "modelBytes": (try? model.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("report.json"), options: .atomic)
        guard unchanged && modelComplete && posesComplete && processingComplete && failures.isEmpty else { exit(1) }
    }

    static func digest(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
    }
}
