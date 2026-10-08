// Derived from ekarad1um/Photogrammetry's session lifecycle (MIT).
// Copyright (c) 2022 ekarad1um. See Vendor/Photogrammetry/LICENSE.
import Darwin
import Foundation
import RealityKit

public enum ReconstructionEvent: Sendable {
    case stage(ReconstructionStage)
    case progress(Double)
    case message(String)
    case diagnostic(String)
}

public enum ReconstructionStage: String, Sendable {
    case preparing, reconstructing, saving
    public var message: String {
        switch self {
        case .preparing: "Preparing photos…"
        case .reconstructing: "Reconstructing the object…"
        case .saving: "Saving the textured model…"
        }
    }
}

public struct RunReport: Codable, Sendable {
    public var id: UUID
    public var startedAt: Date
    public var elapsedSeconds: Double
    public var peakObservedResidentBytes: UInt64
    public var outputBytes: Int
    public var photoCount: Int
    public var quality: ReconstructionQuality
    public var status: String
    public var messages: [String]
}

/// Owns one session and consumes its entire output stream before releasing it.
public actor ReconstructionEngine {
    private var session: PhotogrammetrySession?
    private var isRunning = false
    private var cancellationRequested = false
    private var peakResidentBytes: UInt64 = 0

    public init() {}
    public static var isSupported: Bool { PhotogrammetrySession.isSupported }
    public static var maximumImageCount: Int { PhotogrammetrySession.limits.maximumNumberOfInputImages }
    public static var maximumImageDimension: Int { PhotogrammetrySession.limits.maximumInputImageDimension }

    public static func checkInput(_ photos: [PhotoRecord]) -> ReconstructionInputCheck {
        ReconstructionInputCheck.evaluate(photos: photos, isSupported: isSupported,
                                          maximumImageCount: maximumImageCount,
                                          maximumImageDimension: maximumImageDimension)
    }

    public func cancel() {
        guard isRunning else { return }
        cancellationRequested = true
        session?.cancel()
    }

    public func run(project: Project, onEvent: @escaping @Sendable (ReconstructionEvent) async -> Void) async throws -> Project {
        guard !isRunning else { throw ProjectError.invalid("A reconstruction is already running.") }
        if let reason = Self.checkInput(project.manifest.photos).blockingReason {
            throw ProjectError.invalid(reason)
        }
        isRunning = true
        cancellationRequested = false
        peakResidentBytes = Self.residentBytes()
        let runID = UUID()
        let started = Date()
        let runPath = "cache/\(runID.uuidString)"
        var report = RunReport(id: runID, startedAt: started, elapsedSeconds: 0, peakObservedResidentBytes: 0,
                               outputBytes: 0, photoCount: project.manifest.photos.count,
                               quality: project.manifest.settings.quality, status: "running", messages: [])
        let sampler = Task {
            while !Task.isCancelled {
                self.sampleMemory()
                do { try await Task.sleep(for: .seconds(1)) } catch { return }
            }
        }
        defer {
            sampler.cancel()
            session = nil
            isRunning = false
        }
        do {
            await onEvent(.stage(.preparing))
            guard !cancellationRequested else { throw CancellationError() }
            let runDirectory = try ProjectStore.resolve(runPath, in: project.directory)
            let input = runDirectory.appendingPathComponent("input")
            try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: runDirectory) }
            let snapshot = try PhotoPreparation.stageInputs(project: project, runID: runID)
            let snapshotPath = try ProjectStore.saveInputSnapshot(snapshot, in: project.directory)
            var configuration = PhotogrammetrySession.Configuration()
            configuration.sampleOrdering = .unordered
            configuration.featureSensitivity = .normal
            configuration.isObjectMaskingEnabled = project.manifest.settings.objectMasking
            let current = try PhotogrammetrySession(input: input, configuration: configuration)
            session = current
            await onEvent(.stage(.reconstructing))
            guard !cancellationRequested else { throw CancellationError() }
            let output = runDirectory.appendingPathComponent("model.usdz")
            let detail: PhotogrammetrySession.Request.Detail = project.manifest.settings.quality == .reduced ? .reduced : .medium
            let request = PhotogrammetrySession.Request.modelFile(url: output, detail: detail)
            try current.process(requests: [request])
            var resultState = ModelRequestState()
            outputLoop: for try await event in current.outputs {
                sampleMemory()
                switch event {
                case .requestProgress(_, let fraction):
                    await onEvent(.progress(min(1, max(0, fraction))))
                case .requestComplete(_, let result):
                    if case .modelFile(let url) = result { resultState.modelURL = url }
                    // Keep listening: request completion is not overall session completion.
                case .requestError(_, let error):
                    resultState.requestFailure = error.localizedDescription
                    report.messages.append(error.localizedDescription)
                    await onEvent(.diagnostic(error.localizedDescription))
                case .processingComplete:
                    resultState.processingComplete = true
                    break outputLoop
                case .processingCancelled:
                    throw CancellationError()
                case .inputComplete:
                    await onEvent(.message("Photos loaded. Reconstructing the object…"))
                case .invalidSample(let id, let reason):
                    let message = "Invalid sample \(id): \(reason)"
                    report.messages.append(message)
                    await onEvent(.diagnostic(message))
                    await onEvent(.message("Some photos could not be used. See diagnostics for details."))
                case .skippedSample(let id):
                    report.messages.append("Sample \(id) was skipped.")
                    await onEvent(.diagnostic(report.messages.last!))
                    await onEvent(.message("Some photos were skipped by the reconstruction engine."))
                case .automaticDownsampling:
                    report.messages.append("Object Capture reduced image resolution to fit available resources.")
                    await onEvent(.message(report.messages.last!))
                case .stitchingIncomplete:
                    report.messages.append("Some photo groups could not be stitched together.")
                case .requestProgressInfo(_, let info):
                    if let stage = info.processingStage { await onEvent(.diagnostic("Processing: \(stage)")) }
                @unknown default:
                    report.messages.append("Unhandled session event: \(event)")
                }
            }
            let modelURL = try resultState.completedModel(cancellationRequested: cancellationRequested)
            await onEvent(.stage(.saving))
            guard !cancellationRequested else { throw CancellationError() }
            let updated: Project
            do {
                updated = try ProjectStore.commitModel(from: modelURL, to: project, inputSnapshotPath: snapshotPath)
            } catch {
                let detail = "Saving the model failed: \(error.localizedDescription)"
                report.messages.append(detail)
                await onEvent(.diagnostic(detail))
                throw ProjectError.invalid("The new model could not be saved to the project. Your photos and any previously saved model are preserved. Check free disk space and the project's write permissions, then retry reconstruction. Technical details are available in diagnostics and the run log.")
            }
            report.outputBytes = (try? modelURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            report.status = "completed"
            do { try saveReport(&report, project: project) }
            catch { await onEvent(.message("Model saved, but the run log could not be written: \(error.localizedDescription)")) }
            return updated
        } catch {
            session?.cancel()
            report.status = (error is CancellationError || cancellationRequested) ? "cancelled" : "failed"
            report.messages.append(error.localizedDescription)
            do { try saveReport(&report, project: project) }
            catch { await onEvent(.message("The run log could not be written: \(error.localizedDescription)")) }
            if cancellationRequested { throw CancellationError() }
            throw error
        }
    }

    private func saveReport(_ report: inout RunReport, project: Project) throws {
        report.elapsedSeconds = Date().timeIntervalSince(report.startedAt)
        report.peakObservedResidentBytes = peakResidentBytes
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let url = try ProjectStore.resolve("logs/\(report.id.uuidString).json", in: project.directory)
        try encoder.encode(report).write(to: url, options: .atomic)
    }

    private func sampleMemory() { peakResidentBytes = max(peakResidentBytes, Self.residentBytes()) }

    private static func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
    }
}
