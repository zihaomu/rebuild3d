// Sampled OS thermal classification; not temperature or GPU frequency.
import Foundation

let arguments = CommandLine.arguments
var interval: Double? = nil
if arguments.count > 1 {
    guard arguments.count == 3, arguments[1] == "--interval-seconds",
          let value = Double(arguments[2]), value >= 0.25, value <= 60 else {
        FileHandle.standardError.write(Data("usage: system-state [--interval-seconds 0.25...60]\n".utf8))
        exit(2)
    }
    interval = value
}
repeat {
let process = ProcessInfo.processInfo
let thermalName: String
switch process.thermalState {
case .nominal: thermalName = "nominal"
case .fair: thermalName = "fair"
case .serious: thermalName = "serious"
case .critical: thermalName = "critical"
@unknown default: thermalName = "unknown"
}
let state: [String: Any] = [
    "atUTC": ISO8601DateFormatter().string(from: Date()),
    "thermalState": thermalName,
    "thermalStateRaw": process.thermalState.rawValue,
    "lowPowerMode": process.isLowPowerModeEnabled,
    "systemUptimeSeconds": process.systemUptime,
    "scope": "Sampled OS thermal classification and low-power setting; not temperature or clock telemetry"
]
let data = try JSONSerialization.data(withJSONObject: state, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data([10]))
if let interval { Thread.sleep(forTimeInterval: interval) }
} while interval != nil
