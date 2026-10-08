import Foundation
import Testing
@testable import Rebuild3DCore

private func photo(width: Int = 64, height: Int = 48, orientation: Int = 1) -> PhotoRecord {
    let id = UUID()
    return PhotoRecord(id: id, originalName: "\(id).jpg", imagePath: "images/\(id).jpg",
                       thumbnailPath: "thumbnails/\(id).jpg", pixelWidth: width, pixelHeight: height,
                       orientation: orientation)
}

@Test func inputCountsAndUnsupportedHardwareExplainWhyStartingIsBlocked() {
    for count in 0...6 {
        let photos = (0..<count).map { _ in photo() }
        let check = ReconstructionInputCheck.evaluate(photos: photos, isSupported: true,
                                                       maximumImageCount: 5, maximumImageDimension: 64)
        #expect((check.blockingReason == nil) == (3...5).contains(count))
        #expect(check.photoIssues.isEmpty)
        if count < 3 { #expect(check.blockingReason?.contains("Add at least \(3 - count) more") == true) }
        if count == 6 { #expect(check.blockingReason?.contains("at most 5") == true) }
        let unsupported = ReconstructionInputCheck.evaluate(photos: photos, isSupported: false,
                                                             maximumImageCount: 5, maximumImageDimension: 64)
        #expect(unsupported.blockingReason?.contains("not supported") == true)
    }
}

@Test(arguments: 1...8)
func oversizedInputsAreIdentifiedByPhotoIDForEveryOrientation(orientation: Int) {
    let photos = [photo(width: 64, height: 64, orientation: orientation),
                  photo(width: 65, height: 48, orientation: orientation),
                  photo(width: 48, height: 65, orientation: orientation)]
    let check = ReconstructionInputCheck.evaluate(photos: photos, isSupported: true,
                                                   maximumImageCount: 5, maximumImageDimension: 64)
    #expect(check.blockingReason?.contains("2 photos exceed") == true)
    #expect(Set(check.photoIssues.keys) == Set(photos.dropFirst().map(\.id)))
    #expect(check.photoIssues.values.allSatisfy { $0.contains("64 pixels per side") })
    let boundary = [photo(width: 64, height: 64), photo(width: 48, height: 64), photo(width: 64, height: 48)]
    let accepted = ReconstructionInputCheck.evaluate(photos: boundary, isSupported: true,
                                                      maximumImageCount: 3, maximumImageDimension: 64)
    #expect(accepted.blockingReason == nil && accepted.photoIssues.isEmpty)
}
