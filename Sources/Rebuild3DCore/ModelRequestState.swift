import Foundation

/// A model result is provisional until the session finishes; cancellation and errors always win.
struct ModelRequestState {
    var modelURL: URL?
    var requestFailure: String?
    var processingComplete = false

    func completedModel(cancellationRequested: Bool) throws -> URL {
        guard !cancellationRequested else { throw CancellationError() }
        if requestFailure != nil {
            throw ProjectError.invalid("The reconstruction engine could not create a model. Your photos and any previous model are preserved. You can retry; technical details are available in diagnostics and the run log.")
        }
        guard processingComplete, let modelURL else {
            throw ProjectError.invalid("The session ended without a completed model. Review the input photos and retry.")
        }
        return modelURL
    }
}
