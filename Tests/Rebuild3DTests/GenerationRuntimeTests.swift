import Foundation
import Testing
@testable import Rebuild3DCore

@Test func memoryRetryRecognizesAllocatorFailuresButNotOtherErrors() {
    for report in [
        ["status": "resource-limit"],
        ["status": "failed", "error": "RuntimeError('MPS backend out of memory (MPS allocated: 8 GB)')"],
        ["status": "failed", "error": "OutOfMemoryError('allocation failed')"],
        ["status": "failed", "error": "RuntimeError('DefaultCPUAllocator: can't allocate memory')"]
    ] {
        #expect(GenerationCoordinator.resourceFailure(report))
    }
    for report in [
        ["status": "inference-complete"],
        ["status": "failed", "error": "No space left on device"],
        ["status": "failed", "error": "Checkpoint SHA-256 mismatch"],
        ["status": "failed", "error": "Invalid camera matrix"]
    ] {
        #expect(!GenerationCoordinator.resourceFailure(report))
    }
}

@Test func resumedTaskCannotSubstitutePhotosParametersOrRuntime() throws {
    let photo = SparseSourcePhoto(id: UUID(), name: "同名照片.heic", sourcePath: "/photos/a.heic", sourceSHA256: "abc")
    let replacement = SparseSourcePhoto(id: UUID(), name: photo.name, sourcePath: "/photos/b.heic", sourceSHA256: "def")
    let runtime = SparseRuntimePaths(nativeWorker: "/native", rasterLibrary: "/raster", vggtRepository: "/vggt", weights: "/weights")
    func task(photos: [SparseSourcePhoto], parameters: SparseParameters = .init(), paths: SparseRuntimePaths? = nil) -> SparseTask {
        SparseTask(formatVersion: 1, jobID: UUID().uuidString, photos: photos, allPhotos: [photo], photoUse: [], runtime: paths ?? runtime, parameters: parameters)
    }
    let expected = task(photos: [photo])
    try task(photos: [photo]).validate(resuming: expected)
    #expect(throws: ProjectError.self) { try task(photos: [replacement]).validate(resuming: expected) }
    var reduced = SparseParameters(); reduced.modelSize = 392
    #expect(throws: ProjectError.self) { try task(photos: [photo], parameters: reduced).validate(resuming: expected) }
    let foreign = SparseRuntimePaths(nativeWorker: "/foreign", rasterLibrary: "/raster", vggtRepository: "/vggt", weights: "/weights")
    #expect(throws: ProjectError.self) { try task(photos: [photo], paths: foreign).validate(resuming: expected) }
}

@Test func workerProgressArrivesBeforeTheWriterCloses() throws {
    let process = Process(), pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-c", "printf 'stage-started\\n'; sleep 2"]
    process.standardOutput = pipe
    try process.run()
    try pipe.fileHandleForWriting.close()
    defer { if process.isRunning { process.terminate() }; try? pipe.fileHandleForReading.close() }
    let started = Date()
    let received = try WorkerPipe.readChunk(from: pipe.fileHandleForReading)
    #expect(String(decoding: received, as: UTF8.self) == "stage-started\n")
    #expect(Date().timeIntervalSince(started) < 1, "A short progress line must not wait for EOF or 64 KB of output")
}

@Test func interruptedComponentCopyResumesAndRepairsCorruptPrefixes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("RuntimeCopy-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source"), partial = root.appendingPathComponent("partial")
    let bytes = Data((0..<100_000).map { UInt8($0 % 251) })
    try bytes.write(to: source)
    for prefix in [bytes.prefix(8712), Data(repeating: 9, count: 3000), bytes + Data([1])] {
        try prefix.write(to: partial)
        try GenerationRuntime.copyResuming(source, to: partial, expectedSize: bytes.count, cancellation: .init())
        #expect(try Data(contentsOf: partial) == bytes)
    }
    let cancelled = GenerationCancellation()
    cancelled.cancel()
    try bytes.prefix(77).write(to: partial)
    #expect(throws: CancellationError.self) {
        try GenerationRuntime.copyResuming(source, to: partial, expectedSize: bytes.count, cancellation: cancelled)
    }
    #expect(try Data(contentsOf: partial) == bytes.prefix(77))
}

@Test func componentPathsRejectTraversalAndEscapingPartialSymlinks() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("RuntimePath-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createSymbolicLink(atPath: root.appendingPathComponent("weights.partial").path, withDestinationPath: "/tmp/foreign")
    for path in ["../foreign", "/tmp/foreign", "weights.partial"] {
        #expect(throws: ProjectError.self) { try GenerationRuntime.resolve(path, in: root) }
    }
}

@Test func installedComponentsAreVerifiedAndRepairedBeforeReuse() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("RuntimeInstall-\(UUID())")
    let seed = root.appendingPathComponent("seed"), storage = root.appendingPathComponent("installed")
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = ["python/bin/python3.12", "rebuild3d-native-worker", "worker/rebuild3d_worker/worker.py",
                 "worker/rebuild3d_worker/raster.dylib", "vggt/source-manifest.json", "models/vggt-1b.safetensors"]
    let artifacts = try paths.map { path in
        let url = seed.appendingPathComponent(path), data = Data("fixture:\(path)".utf8)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
        return RuntimeArtifact(path: path, byteCount: data.count, sha256: GenerationRuntime.digest(data))
    }
    let manifest = RuntimeManifest(formatVersion: 1, platform: "macos-arm64", artifacts: artifacts)
    try JSONEncoder().encode(manifest).write(to: seed.appendingPathComponent("runtime.json"))
    let runtime = try await GenerationRuntime.prepare(seed: seed, storage: storage, cancellation: .init(), onEvent: { _ in })
    try Data("corrupt".utf8).write(to: runtime.weights)
    let repaired = try await GenerationRuntime.prepare(seed: seed, storage: storage, cancellation: .init(), onEvent: { _ in })
    #expect(repaired.identity == runtime.identity)
    #expect(try Data(contentsOf: repaired.weights) == Data(contentsOf: seed.appendingPathComponent(paths.last!)))
    try Data("damaged bundled source".utf8).write(to: seed.appendingPathComponent(paths.last!))
    try FileManager.default.removeItem(at: repaired.weights)
    await #expect(throws: ProjectError.self) {
        _ = try await GenerationRuntime.prepare(seed: seed, storage: storage, cancellation: .init(), onEvent: { _ in })
    }
}
