import Darwin
import Foundation

enum WorkerPipe {
    /// Return the bytes currently delivered by a pipe, without waiting to fill a fixed-size buffer.
    static func readChunk(from handle: FileHandle) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: 65_536)
        while true {
            let count = Darwin.read(handle.fileDescriptor, &bytes, bytes.count)
            if count >= 0 { return Data(bytes.prefix(count)) }
            if errno != EINTR { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
    }
}
