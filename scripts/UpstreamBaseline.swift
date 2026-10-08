// Exercise the pinned upstream service without changing its source files.
import Foundation
import RealityKit

@main
struct UpstreamBaseline {
    @MainActor static func main() async {
        guard CommandLine.arguments.count == 3 else {
            print("Usage: upstream-baseline INPUT_DIRECTORY OUTPUT.usdz")
            exit(2)
        }
        let delegate = PhotogrammetryDelegate()
        delegate.inputFolderUrl = URL(fileURLWithPath: CommandLine.arguments[1])
        delegate.outputModelUrl = URL(fileURLWithPath: CommandLine.arguments[2])
        delegate.sessionRequestDetail = .reduced
        delegate.sessionConfiguration.isObjectMaskingEnabled = true
        let started = Date()
        do {
            let result: URL = try await withCheckedThrowingContinuation { continuation in
                delegate.generateModel { continuation.resume(with: $0) }
            }
            print("Upstream model: \(result.path)")
            print("Elapsed seconds: \(Date().timeIntervalSince(started))")
            print("Output bytes: \((try? result.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)")
        } catch {
            print("Upstream reconstruction failed: \(error.localizedDescription)")
            exit(1)
        }
    }
}
