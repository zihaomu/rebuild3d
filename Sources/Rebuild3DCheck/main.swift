import Foundation
import Rebuild3DCore

@main
struct Rebuild3DCheck {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.isEmpty || arguments == ["doctor"] {
                print("OS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
                print("Physical memory: \(ProcessInfo.processInfo.physicalMemory) bytes")
                print("Object Capture supported: \(ReconstructionEngine.isSupported)")
                print("Maximum input images: \(ReconstructionEngine.maximumImageCount)")
                print("Maximum input dimension: \(ReconstructionEngine.maximumImageDimension) pixels per side")
                return
            }
            if arguments.count == 3, arguments[0] == "export" {
                let project = try ProjectStore.open(URL(fileURLWithPath: arguments[1]))
                try ProjectStore.exportModel(project, to: URL(fileURLWithPath: arguments[2]))
                print("USDZ exported.")
                return
            }
            let project: Project
            var cancelDelay: Double?
            if arguments.count == 3, arguments[0] == "reconstruct" {
                let input = URL(fileURLWithPath: arguments[1])
                let destination = URL(fileURLWithPath: arguments[2])
                var created = try ProjectStore.create(at: destination)
                let imported = try PhotoImporter.importPhotos(from: [input], into: destination)
                print(imported.summary)
                for issue in imported.issues { print("\(issue.kind.rawValue): \(issue.filename): \(issue.reason)") }
                created.manifest.photos = imported.photos
                try ProjectStore.save(created)
                project = created
            } else if [2, 4].contains(arguments.count), arguments[0] == "rebuild" {
                project = try ProjectStore.open(URL(fileURLWithPath: arguments[1]))
                if arguments.count == 4 {
                    guard arguments[2] == "--cancel-after", let seconds = Double(arguments[3]), seconds.isFinite, seconds > 0 else {
                        throw ProjectError.invalid("--cancel-after requires a positive number of seconds.")
                    }
                    cancelDelay = seconds
                }
            } else {
                throw ProjectError.invalid("Usage: rebuild3d-check doctor | reconstruct INPUT_DIRECTORY NEW_PROJECT.rebuild3d | rebuild PROJECT.rebuild3d [--cancel-after SECONDS] | export PROJECT.rebuild3d OUTPUT.usdz")
            }
            print("Reconstructing \(project.manifest.photos.count) photos at \(project.manifest.settings.quality.rawValue) quality.")
            fflush(nil)
            let engine = ReconstructionEngine()
            let cancellation = cancelDelay.map { seconds in
                Task {
                    do { try await Task.sleep(for: .seconds(seconds)) } catch { return }
                    await engine.cancel()
                }
            }
            defer { cancellation?.cancel() }
            let printer = EventPrinter()
            let result = try await engine.run(project: project) { event in
                await printer.receive(event)
            }
            print("Saved model: \(result.modelURL!.path)")
            print("Run report: \(project.directory.appendingPathComponent("logs").path)")
        } catch is CancellationError {
            print("Reconstruction cancelled; any previous model is preserved.")
            exit(130)
        } catch {
            FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
}

private actor EventPrinter {
    private var lastPercent = -1
    private var lastMessage = ""
    func receive(_ event: ReconstructionEvent) {
        switch event {
        case .stage(let stage):
            print(stage.message)
        case .diagnostic(let details):
            print("Diagnostic: \(details)")
        case .progress(let fraction):
            let percent = Int(fraction * 100)
            guard percent != lastPercent else { return }
            lastPercent = percent
            print("Progress: \(percent)%")
        case .message(let message):
            guard message != lastMessage else { return }
            lastMessage = message
            print(message)
        }
        fflush(nil)
    }
}
