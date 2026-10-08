import CoreGraphics
import Foundation
import ImageIO

public enum PhotoPreviewDecoder {
    /// Use one explicit SDR, color-managed policy for thumbnails and original-photo previews.
    /// Originals are never changed; reconstruction preparation records any required working-image conversion separately.
    public static func image(at url: URL, maxPixelSize: Int) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw ProjectError.invalid("The photo cannot be decoded for preview.")
        }
        return try image(from: source, maxPixelSize: maxPixelSize)
    }

    static func image(from source: CGImageSource, maxPixelSize: Int) throws -> CGImage {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, maxPixelSize),
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceDecodeRequest: kCGImageSourceDecodeToSDR
        ]
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(source, CGImageSourceGetPrimaryImageIndex(source), options as CFDictionary),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: decoded.width, height: decoded.height, bitsPerComponent: 8,
                  bytesPerRow: 0, space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ProjectError.invalid("The photo cannot be decoded for preview.")
        }
        context.draw(decoded, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        guard let image = context.makeImage() else { throw ProjectError.invalid("The preview could not be created.") }
        return image
    }
}
