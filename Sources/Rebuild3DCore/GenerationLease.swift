import Darwin
import Foundation

/// Serializes heavy work across app processes. The OS releases the lock after a crash.
final class GenerationLease: @unchecked Sendable {
    private let descriptor: Int32
    private init(_ descriptor: Int32) { self.descriptor = descriptor }
    deinit { flock(descriptor, LOCK_UN); close(descriptor) }

    static func acquire(in directory: URL, cancellation: GenerationCancellation,
                        onEvent: @escaping @Sendable (ReconstructionEvent) async -> Void) async throws -> GenerationLease {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent(".generation.lock").path
        let fd = open(path, O_RDWR | O_CREAT | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        do {
            var announced = false
            while flock(fd, LOCK_EX | LOCK_NB) != 0 {
                guard errno == EWOULDBLOCK else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                try cancellation.check()
                if !announced {
                    await onEvent(.message("正在等待另一项生成任务释放资源…"))
                    announced = true
                }
                try await Task.sleep(for: .milliseconds(250))
            }
            try cancellation.check()
            return GenerationLease(fd)
        } catch { close(fd); throw error }
    }
}
