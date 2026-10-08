import Foundation

/// The same preflight result drives the interface and the engine's start guard.
public struct ReconstructionInputCheck: Sendable {
    public let blockingReason: String?
    public let photoIssues: [UUID: String]

    static func evaluate(photos: [PhotoRecord], isSupported: Bool,
                         maximumImageCount: Int, maximumImageDimension: Int) -> Self {
        var photoIssues: [UUID: String] = [:]
        for photo in photos where max(photo.pixelWidth, photo.pixelHeight) > maximumImageDimension {
            photoIssues[photo.id] = "This photo exceeds this Mac's limit of \(maximumImageDimension) pixels per side."
        }
        let reason: String?
        if !isSupported {
            reason = "Object Capture is not supported on this Mac."
        } else if photos.count < 3 {
            let missing = 3 - photos.count
            reason = "Add at least \(missing) more overlapping \(missing == 1 ? "photo" : "photos") to start. A complete object usually needs many more."
        } else if photos.count > maximumImageCount {
            reason = "This Mac supports at most \(maximumImageCount) photos per session. Remove \(photos.count - maximumImageCount) photos from the input list to continue."
        } else if !photoIssues.isEmpty {
            reason = "\(photoIssues.count) \(photoIssues.count == 1 ? "photo exceeds" : "photos exceed") this Mac's size limit. Remove the marked photos from the input list to continue."
        } else {
            reason = nil
        }
        return Self(blockingReason: reason, photoIssues: photoIssues)
    }
}
