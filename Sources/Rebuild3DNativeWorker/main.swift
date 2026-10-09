import Foundation
import Rebuild3DCore

// This executable is built and shipped with the app; ordinary users never invoke Swift.
func emit(_ value: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else { return }
    FileHandle.standardOutput.write(data + Data([10]))
}

do {
    let args = CommandLine.arguments
    guard args.count == 4, args[1] == "prepare" else {
        throw ProjectError.invalid("Usage: rebuild3d-native-worker prepare PHOTOS_JSON NEW_OUTPUT_DIRECTORY")
    }
    let photos = try JSONDecoder().decode([SparseSourcePhoto].self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    try SparseInputPreparation.prepare(photos, at: URL(fileURLWithPath: args[3])) { done, total, name in
        emit(["protocolVersion": 1, "event": "progress", "stage": "prepare",
              "completed": done, "total": total, "photo": name])
    }
    emit(["protocolVersion": 1, "event": "completed", "stage": "prepare"])
} catch {
    emit(["protocolVersion": 1, "event": "failed", "stage": "prepare", "message": error.localizedDescription])
    exit(1)
}
