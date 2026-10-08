import Foundation
import Testing
@testable import Rebuild3DCore

@Test func aModelRequestResultDoesNotEndTheSession() throws {
    var state = ModelRequestState()
    state.modelURL = URL(fileURLWithPath: "/tmp/result.usdz")
    #expect(throws: ProjectError.self) { try state.completedModel(cancellationRequested: false) }
    state.processingComplete = true
    #expect(try state.completedModel(cancellationRequested: false) == state.modelURL)
}

@Test func cancellationAfterModelRequestCompletionCannotCommitTheModel() throws {
    let state = ModelRequestState(modelURL: URL(fileURLWithPath: "/tmp/result.usdz"), processingComplete: true)
    #expect(throws: CancellationError.self) { try state.completedModel(cancellationRequested: true) }
}

@Test func requestErrorAndMissingOutputAreFailuresEvenAfterSessionCompletion() throws {
    let failed = ModelRequestState(modelURL: URL(fileURLWithPath: "/tmp/result.usdz"), requestFailure: "disk full", processingComplete: true)
    #expect(throws: ProjectError.self) { try failed.completedModel(cancellationRequested: false) }
    let missing = ModelRequestState(processingComplete: true)
    #expect(throws: ProjectError.self) { try missing.completedModel(cancellationRequested: false) }
}
