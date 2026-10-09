import CryptoKit
import Darwin
import Foundation

/// Coordinates cancellation across synchronous file work and the owned subprocess.
public final class GenerationCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var requested = false
    private var process: Process?

    public init() {}
    public var isCancelled: Bool { lock.withLock { requested } }
    public func check() throws { if isCancelled { throw CancellationError() } }
    public func cancel() {
        let current = lock.withLock { requested = true; return process }
        if current?.isRunning == true { current?.terminate() }
    }
    func launch(_ child: Process) throws {
        try lock.withLock {
            guard !requested else { throw CancellationError() }
            try child.run()
            process = child
        }
    }
    func finished() { lock.withLock { process = nil } }
}

struct RuntimeArtifact: Codable, Sendable {
    let path: String
    let byteCount: Int
    let sha256: String
}

struct RuntimeManifest: Codable, Sendable {
    let formatVersion: Int
    let platform: String
    let artifacts: [RuntimeArtifact]
}

public struct GenerationRuntime: Sendable {
    public let directory: URL
    public let identity: String
    public var python: URL { directory.appendingPathComponent("python/bin/python3.12") }
    public var workerRoot: URL { directory.appendingPathComponent("worker") }
    public var nativeWorker: URL { directory.appendingPathComponent("rebuild3d-native-worker") }
    public var raster: URL { workerRoot.appendingPathComponent("rebuild3d_worker/raster.dylib") }
    public var vggt: URL { directory.appendingPathComponent("vggt") }
    public var weights: URL { directory.appendingPathComponent("models/vggt-1b.safetensors") }

    /// Components ship with the local app. Installation and repair never require developer tools.
    public static func prepare(seed: URL, storage: URL, cancellation: GenerationCancellation,
                               onEvent: @escaping @Sendable (ReconstructionEvent) async -> Void) async throws -> Self {
        try await Task.detached(priority: .userInitiated) {
            let fm = FileManager.default
            let data = try Data(contentsOf: seed.appendingPathComponent("runtime.json"))
            let manifest = try JSONDecoder().decode(RuntimeManifest.self, from: data)
            guard manifest.formatVersion == 1, manifest.platform == "macos-arm64",
                  !manifest.artifacts.isEmpty,
                  Set(manifest.artifacts.map(\.path)).count == manifest.artifacts.count else {
                throw ProjectError.invalid("生成组件清单无效，请重新安装完整应用。")
            }
            let required: Set<String> = ["python/bin/python3.12", "rebuild3d-native-worker",
                "worker/rebuild3d_worker/worker.py", "worker/rebuild3d_worker/raster.dylib",
                "vggt/source-manifest.json", "models/vggt-1b.safetensors"]
            guard required.isSubset(of: Set(manifest.artifacts.map(\.path))) else {
                throw ProjectError.invalid("应用缺少必要的生成组件，请重新安装完整应用。")
            }
            let identity = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            let destination = storage.appendingPathComponent(identity)
            try fm.createDirectory(at: destination, withIntermediateDirectories: true)
            let ordered = manifest.artifacts.sorted { $0.byteCount < $1.byteCount }
            for (index, artifact) in ordered.enumerated() {
                try cancellation.check()
                let target = try resolve(artifact.path, in: destination)
                if index % 256 == 0 {
                    await onEvent(.message("准备生成组件（\(index + 1)/\(ordered.count)）…"))
                }
                guard artifact.byteCount > 0, artifact.sha256.count == 64 else {
                    // Some packages contain legitimate empty __init__.py files.
                    if artifact.byteCount == 0, artifact.sha256 == digest(Data()) {
                        try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                        try Data().write(to: target, options: .atomic)
                        continue
                    }
                    throw ProjectError.invalid("生成组件记录损坏：\(artifact.path)")
                }
                if try matches(target, artifact) { continue }
                let source = try resolve(artifact.path, in: seed)
                guard try matches(source, artifact) else {
                    throw ProjectError.invalid("应用内的生成组件损坏：\(artifact.path)。请重新安装完整应用。")
                }
                try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                let partial = try resolve(artifact.path + ".partial", in: destination)
                // Keep completed component files and resume the large weight file after interruption.
                try copyResuming(source, to: partial, expectedSize: artifact.byteCount, cancellation: cancellation)
                guard try matches(partial, artifact) else { throw ProjectError.invalid("组件校验失败：\(artifact.path)") }
                let attributes = try fm.attributesOfItem(atPath: source.path)
                if let mode = attributes[.posixPermissions] { try fm.setAttributes([.posixPermissions: mode], ofItemAtPath: partial.path) }
                try cancellation.check()
                guard rename(partial.path, target.path) == 0 else {
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
            }
            try data.write(to: destination.appendingPathComponent("runtime.json"), options: .atomic)
            return GenerationRuntime(directory: destination, identity: identity)
        }.value
    }

    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    static func resolve(_ path: String, in directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw ProjectError.invalid("Invalid runtime path.")
        }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        var component = root
        for name in path.split(separator: "/") {
            component.appendPathComponent(String(name))
            // Foundation leaves dangling symlinks unresolved; reject those explicitly too.
            if (try? FileManager.default.destinationOfSymbolicLink(atPath: component.path)) != nil {
                throw ProjectError.invalid("Runtime components cannot be symbolic links.")
            }
        }
        let target = root.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
        guard target.path.hasPrefix(root.path + "/") else { throw ProjectError.invalid("Runtime path escapes its directory.") }
        return target
    }

    static func matches(_ url: URL, _ artifact: RuntimeArtifact) throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, values.fileSize == artifact.byteCount else { return false }
        return try PhotoInspector.contentSHA256(at: url) == artifact.sha256
    }

    static func copyResuming(_ source: URL, to partial: URL, expectedSize: Int,
                             cancellation: GenerationCancellation) throws {
        try cancellation.check()
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        // Preserve ENOSPC/EACCES from file creation rather than hiding it behind a later ENOENT.
        let descriptor = open(partial.path, O_RDWR | O_CREAT | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let output = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? output.close() }
        let length = try output.seekToEnd()
        var offset: UInt64 = 0
        try output.seek(toOffset: 0)
        // Verify the existing prefix before continuing; a corrupt partial is replaced.
        var prefixValid = length <= UInt64(expectedSize)
        while prefixValid && offset < length {
            try cancellation.check()
            let count = Int(min(4_194_304, length - offset))
            guard let a = try input.read(upToCount: count), let b = try output.read(upToCount: count),
                  a == b, a.count == count else { prefixValid = false; break }
            offset += UInt64(count)
        }
        if !prefixValid {
            try output.truncate(atOffset: 0); try output.seek(toOffset: 0); try input.seek(toOffset: 0)
            offset = 0
        }
        while offset < UInt64(expectedSize) {
            try cancellation.check()
            guard let data = try input.read(upToCount: 4_194_304), !data.isEmpty else {
                throw ProjectError.invalid("生成组件不完整。")
            }
            try output.write(contentsOf: data)
            offset += UInt64(data.count)
        }
        try output.synchronize()
    }
}
