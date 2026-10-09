import Foundation
import RealityKit

struct SparseRuntimePaths: Codable, Sendable, Equatable {
    let nativeWorker: String
    let rasterLibrary: String
    let vggtRepository: String
    let weights: String
}
struct SparseParameters: Codable, Sendable, Equatable {
    var memoryGiB = 10.5
    var modelSize = 518
    var volumeResolution = 256
    var textureSize = 4096
}
struct SparseTask: Codable, Sendable {
    let formatVersion: Int
    let jobID: String
    let photos: [SparseSourcePhoto]
    let allPhotos: [SparseSourcePhoto]
    let photoUse: [SparsePhotoUse]
    let runtime: SparseRuntimePaths
    let parameters: SparseParameters

    func validate(resuming expected: SparseTask) throws {
        guard formatVersion == expected.formatVersion, !jobID.isEmpty,
              photos == expected.photos, allPhotos == expected.allPhotos,
              photoUse == expected.photoUse, runtime == expected.runtime,
              parameters == expected.parameters else {
            throw ProjectError.invalid("已保存的生成任务与当前照片或配置不一致。照片和上次结果已保留。")
        }
    }
}
struct GenerationIdentity: Codable, Sendable {
    let version: Int
    let runtime: String
    let photos: [SparseSourcePhoto]
    let selectedPhotos: [SparseSourcePhoto]
    let photoUse: [SparsePhotoUse]
    let parameters: SparseParameters
    let settings: ReconstructionSettings
    let attempt: Int
}
struct WorkerEvent: Decodable {
    let protocolVersion: Int
    let event: String
    let stage: String?
    let message: String?
    let completed: Int?
    let total: Int?
}

/// One user operation owns preparation, backend choice, recovery and atomic publication.
public actor GenerationCoordinator {
    private let conventional = ReconstructionEngine()
    private var running = false
    private var cancellation: GenerationCancellation?
    private let seed: URL?
    private let storage: URL

    public init(runtimeSeed: URL? = nil, runtimeStorage: URL? = nil) {
        self.seed = runtimeSeed ?? Bundle.main.resourceURL?.appendingPathComponent("GenerationRuntime")
        self.storage = runtimeStorage ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("org.rebuild3d.app/Runtimes")
    }

    public static func checkInput(_ photos: [PhotoRecord]) -> ReconstructionInputCheck {
        if photos.count < 3 {
            return ReconstructionInputCheck(blockingReason: "至少再添加 \(3 - photos.count) 张同一物体的照片。", photoIssues: [:])
        }
        // Small-photo input limits belong to the sparse backend, not to Object Capture.
        return ReconstructionInputCheck(blockingReason: nil, photoIssues: [:])
    }

    public func cancel() async {
        cancellation?.cancel()
        await conventional.cancel()
    }

    public func run(project: Project, onEvent: @escaping @Sendable (ReconstructionEvent) async -> Void) async throws -> Project {
        guard !running else { throw ProjectError.invalid("已有生成任务正在运行。") }
        if let reason = Self.checkInput(project.manifest.photos).blockingReason { throw ProjectError.invalid(reason) }
        running = true
        let control = GenerationCancellation()
        cancellation = control
        defer { running = false; cancellation = nil }
        let lease = try await GenerationLease.acquire(in: storage, cancellation: control, onEvent: onEvent)
        defer { withExtendedLifetime(lease) {} }
        if project.manifest.photos.count > 12 {
            if ReconstructionEngine.checkInput(project.manifest.photos).blockingReason == nil {
                await onEvent(.diagnostic("Automatic route: Object Capture; one conventional attempt."))
                do { return try await conventional.run(project: project, onEvent: onEvent) }
                catch is CancellationError { throw CancellationError() }
                catch let error as ReconstructionBackendFailure {
                    await onEvent(.diagnostic("Conventional reconstruction failed: \(error.localizedDescription)"))
                    await onEvent(.message("正在自动改用近似生成方式…"))
                }
            } else {
                await onEvent(.diagnostic("Conventional input capability exceeded; using a recorded sparse subset."))
            }
        }
        // Conventional progress belongs to that route, not to the sparse fallback.
        await onEvent(.stage(.preparing))
        guard let seed, FileManager.default.fileExists(atPath: seed.appendingPathComponent("runtime.json").path) else {
            throw ProjectError.invalid("此应用包缺少少图生成组件。请使用包含组件的完整 Rebuild3D 应用。")
        }
        await onEvent(.message("正在准备生成组件，首次使用可能需要较长时间…"))
        let runtime = try await GenerationRuntime.prepare(seed: seed, storage: storage, cancellation: control, onEvent: onEvent)
        try control.check()
        let photos = try await Task.detached {
            try project.manifest.photos.map { photo in
                let url = try ProjectStore.resolve(photo.imagePath, in: project.directory)
                let sha = try PhotoInspector.contentSHA256(at: url)
                if let expected = photo.metadata?.contentSHA256, expected != sha {
                    throw ProjectError.invalid("原照片已改变：\(photo.originalName)。请重新导入。")
                }
                return SparseSourcePhoto(id: photo.id, name: photo.originalName, sourcePath: url.path, sourceSHA256: sha)
            }
        }.value
        let (selected, photoUse) = try await Task.detached {
            try SparsePhotoSelection.select(photos, cancellation: control)
        }.value
        var lastError: Error?
        for attempt in 0..<2 {
            try control.check()
            var parameters = SparseParameters()
            if attempt == 1 { parameters.modelSize = 392 }
            let identity = GenerationIdentity(version: 2, runtime: runtime.identity, photos: photos,
                                               selectedPhotos: selected, photoUse: photoUse, parameters: parameters,
                                               settings: project.manifest.settings, attempt: attempt)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let key = GenerationRuntime.digest(try encoder.encode(identity))
            let directory = try ProjectStore.resolve("generation/\(key)", in: project.directory)
            let request = directory.appendingPathComponent("request.json")
            let expected = SparseTask(formatVersion: 1, jobID: UUID().uuidString, photos: selected,
                allPhotos: photos, photoUse: photoUse,
                runtime: SparseRuntimePaths(nativeWorker: runtime.nativeWorker.path, rasterLibrary: runtime.raster.path,
                                           vggtRepository: runtime.vggt.path, weights: runtime.weights.path), parameters: parameters)
            let task: SparseTask
            if FileManager.default.fileExists(atPath: request.path) {
                task = try JSONDecoder().decode(SparseTask.self, from: Data(contentsOf: request))
                try task.validate(resuming: expected)
            } else {
                task = expected
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try encoder.encode(task).write(to: request, options: .atomic)
            }
            await onEvent(.diagnostic("Automatic route: sparse; attempt \(attempt + 1)/2; task \(task.jobID); using \(selected.count)/\(photos.count) photos; every role recorded."))
            do {
                await onEvent(.message("正在启动本地生成组件…"))
                try await Self.runWorker(runtime: runtime, request: request, directory: directory,
                                         cancellation: control, onEvent: onEvent)
                try control.check()
                let result = directory.appendingPathComponent("bundle")
                await onEvent(.message("正在检查模型与照片来源…"))
                _ = try await Task.detached { try ApproximateResultStore.load(result) }.value
                try await Self.validateModel(result.appendingPathComponent("model.usdz"),
                                             provenance: result.appendingPathComponent("provenance.usdz"))
                try await Self.validateModel(result.appendingPathComponent("model.usdz"),
                                             provenance: result.appendingPathComponent("texture-sources.usdz"))
                try control.check()
                await onEvent(.stage(.saving))
                // importResult verifies current originals again and commits the new manifest last.
                return try await Task.detached {
                    try control.check()
                    return try ApproximateResultStore.importResult(from: result, into: project,
                                                                   cancellationCheck: { try control.check() })
                }.value
            } catch {
                if control.isCancelled || error is CancellationError { throw CancellationError() }
                lastError = error
                await onEvent(.diagnostic(error.localizedDescription))
                guard attempt == 0, Self.isResourceFailure(in: directory) else { throw error }
                await onEvent(.message("正在调整内存用量并自动重试…"))
            }
        }
        throw lastError ?? ProjectError.invalid("未能生成模型；照片和上次结果已保留。")
    }

    private static func isResourceFailure(in directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("state.json")),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        if (state["message"] as? String)?.contains("memory budget") == true { return true }
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("inference/report.json")),
              let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return resourceFailure(report)
    }

    static func resourceFailure(_ report: [String: Any]) -> Bool {
        if report["status"] as? String == "resource-limit" { return true }
        // An allocator can reject a request between memory-monitor samples.
        // Preserve the finite lower-resolution retry for that explicit failure too.
        let error = (report["error"] as? String ?? "").lowercased()
        return error.contains("outofmemoryerror") || error.contains("mps backend out of memory")
            || error.contains("defaultcpuallocator: can't allocate memory")
    }

    @MainActor private static func validateModel(_ model: URL, provenance: URL) async throws {
        let a = try await Entity(contentsOf: model)
        let b = try await Entity(contentsOf: provenance)
        let x = a.visualBounds(relativeTo: nil).extents, y = b.visualBounds(relativeTo: nil).extents
        guard x.x.isFinite, x.y.isFinite, x.z.isFinite, min(x.x, x.y, x.z) > 0,
              abs(x.x - y.x) + abs(x.y - y.y) + abs(x.z - y.z) < 0.001 else {
            throw ProjectError.invalid("生成结果没有有效的三维体积或来源模型不一致。")
        }
    }

    private static func runWorker(runtime: GenerationRuntime, request: URL, directory: URL,
                                  cancellation: GenerationCancellation,
                                  onEvent: @escaping @Sendable (ReconstructionEvent) async -> Void) async throws {
        try await withTaskCancellationHandler {
            try await Task.detached(priority: .userInitiated) {
                let process = Process(), pipe = Pipe()
                process.executableURL = runtime.python
                process.arguments = ["-m", "rebuild3d_worker.worker", request.path, directory.path]
                process.currentDirectoryURL = runtime.directory
                process.environment = ["PATH": "/usr/bin:/bin", "PYTHONPATH": runtime.workerRoot.path,
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                    "TMPDIR": NSTemporaryDirectory(), "REBUILD3D_PARENT_PID": String(getpid())]
                process.standardOutput = pipe; process.standardError = pipe
                try cancellation.launch(process)
                defer { cancellation.finished(); try? pipe.fileHandleForReading.close() }
                // Release the parent's writer so EOF arrives when the child exits.
                try pipe.fileHandleForWriting.close()
                var buffer = Data(), failure: String?
                let decoder = JSONDecoder()
                while true {
                    let chunk = try WorkerPipe.readChunk(from: pipe.fileHandleForReading)
                    if chunk.isEmpty { break }
                    buffer.append(chunk)
                    while let newline = buffer.firstIndex(of: 10) {
                        let line = buffer.prefix(upTo: newline)
                        buffer.removeSubrange(...newline)
                        guard let event = try? decoder.decode(WorkerEvent.self, from: line), event.protocolVersion == 1 else {
                            let message = String(decoding: line, as: UTF8.self)
                            if !message.isEmpty { await onEvent(.diagnostic(message)) }
                            continue
                        }
                        if event.event == "failed" { failure = event.message }
                        if ["stage-started", "stage-reused", "progress"].contains(event.event) {
                            var message = Self.stageMessage(event.stage)
                            if let count = event.completed, let total = event.total { message += "（\(count)/\(total)）" }
                            if event.event == "stage-reused" { message += " · 已恢复" }
                            await onEvent(.message(message))
                        } else if let detail = event.message { await onEvent(.diagnostic(detail)) }
                    }
                }
                process.waitUntilExit()
                if cancellation.isCancelled || process.terminationStatus == 130 { throw CancellationError() }
                guard process.terminationStatus == 0 else {
                    throw ProjectError.invalid(failure ?? "生成未完成（状态 \(process.terminationStatus)）。照片和上次结果已保留，可再次点击生成以恢复。")
                }
                let data = try Data(contentsOf: directory.appendingPathComponent("state.json"))
                let state = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                guard state?["status"] as? String == "completed" else { throw ProjectError.invalid("生成任务没有完成记录。") }
            }.value
        } onCancel: { cancellation.cancel() }
    }

    private static func stageMessage(_ stage: String?) -> String {
        switch stage {
        case "prepare": "准备照片并识别主体…"
        case "inference": "分析照片之间的关系…"
        case "fusion": "生成主体形状…"
        case "unwrap", "atlas": "展开模型表面…"
        case "texture": "铺设原照片纹理…"
        case "package": "整理模型与来源记录…"
        default: "正在生成模型…"
        }
    }
}
