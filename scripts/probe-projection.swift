// Offline P1 experiment. Does not enable photo-aligned viewing in the application.
// Uses the model and poses from the same probe-poses.swift session, without recentering.
import AppKit
import ImageIO
import Metal
import SceneKit
import simd

func fail(_ message: String) -> Never { fputs(message + "\n", stderr); exit(1) }
func matrix(_ values: [Float]) -> simd_float4x4 {
    guard values.count == 16, values.allSatisfy(\.isFinite) else { fail("Invalid pose matrix.") }
    return simd_float4x4(columns: (
        SIMD4(values[0], values[1], values[2], values[3]),
        SIMD4(values[4], values[5], values[6], values[7]),
        SIMD4(values[8], values[9], values[10], values[11]),
        SIMD4(values[12], values[13], values[14], values[15])))
}
func writePNG(_ image: NSImage, to url: URL) throws {
    guard let data = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: data),
          let png = bitmap.representation(using: .png, properties: [:]) else { fail("Cannot encode render.") }
    try png.write(to: url, options: .atomic)
}

let upright = CommandLine.arguments.count == 5 && CommandLine.arguments[4] == "--upright"
guard CommandLine.arguments.count == 4 || upright else {
    fail("Usage: swift scripts/probe-projection.swift SOURCE.rebuild3d POSE_PROBE_DIRECTORY NEW_OUTPUT_DIRECTORY [--upright]")
}
let project = URL(fileURLWithPath: CommandLine.arguments[1])
let probe = URL(fileURLWithPath: CommandLine.arguments[2])
let output = URL(fileURLWithPath: CommandLine.arguments[3])
guard !FileManager.default.fileExists(atPath: output.path) else { fail("Output directory already exists.") }
let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: project.appendingPathComponent("project.json"))) as! [String: Any]
let poses = try JSONSerialization.jsonObject(with: Data(contentsOf: probe.appendingPathComponent("cameras-probe.json"))) as! [String: Any]
guard manifest["id"] as? String == poses["sourceProjectID"] as? String,
      poses["modelCompleted"] as? Bool == true, poses["posesCompleted"] as? Bool == true,
      poses["sessionCompleted"] as? Bool == true,
      (poses["failures"] as? [String])?.isEmpty == true else { fail("The source project and successful pose probe must match.") }
let records = (poses["records"] as! [[String: Any]]).filter {
    $0["status"] as? String == "located" && $0["intrinsicsColumnMajor"] != nil
        && [1, 6].contains($0["inputOrientation"] as? Int ?? $0["orientation"] as? Int ?? 0)
}
guard records.count >= 3 else { fail("This bounded experiment needs three located photos with input orientation 1 or 6 and intrinsics.") }
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let scene = try SCNScene(url: probe.appendingPathComponent("model.usdz"))
scene.rootNode.enumerateChildNodes { node, _ in
    for material in node.geometry?.materials ?? [] { material.lightingModel = .constant }
}
scene.background.contents = NSColor(calibratedWhite: 0.12, alpha: 1)
let camera = SCNNode()
camera.camera = SCNCamera()
scene.rootNode.addChildNode(camera)
let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
renderer.scene = scene
renderer.pointOfView = camera
let flipYZ = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))
var reports: [[String: Any]] = []

for index in [0, records.count / 2, records.count - 1] {
    let record = records[index]
    let id = record["sampleID"] as! Int
    let inputWidth = (record["inputWidth"] as? NSNumber ?? record["pixelWidth"] as! NSNumber).floatValue
    let inputHeight = (record["inputHeight"] as? NSNumber ?? record["pixelHeight"] as! NSNumber).floatValue
    let inputOrientation = record["inputOrientation"] as? Int ?? record["orientation"] as! Int
    var width = inputWidth, height = inputHeight
    var k = (record["intrinsicsColumnMajor"] as! [NSNumber]).map(\.floatValue)
    var t = matrix((record["rawPoseTransformColumnMajor"] as! [NSNumber]).map(\.floatValue))
    let encodedK = k
    let encodedPose = t
    guard width > 0, height > 0, abs(simd_determinant(t) - 1) < 0.001,
          k.count == 9, k.allSatisfy(\.isFinite), abs(k[3]) < 0.001, abs(k[1]) < 0.001,
          abs(k[2]) < 0.001, abs(k[5]) < 0.001, abs(k[8] - 1) < 0.001,
          k[0] > 0, k[4] > 0 else { fail("Unexpected pose or intrinsics; no silent approximation allowed.") }
    if upright && inputOrientation == 6 {
        // EXIF 6: (u, v) -> (height - 1 - v, u), using integer pixel centers.
        // Rotate camera axes too; swapping the image dimensions alone is insufficient.
        k = [k[4], 0, 0, 0, k[0], 0, inputHeight - 1 - k[7], k[6], 1]
        t = t * simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1)))
        width = inputHeight
        height = inputWidth
    }
    let scale = min(1, 900 / max(width, height))
    let renderWidth = Int((width * scale).rounded())
    let renderHeight = Int((height * scale).rounded())
    let size = CGSize(width: renderWidth, height: renderHeight)
    let near: Float = 0.001, far: Float = 100
    // OpenGL camera: +X right, +Y up, looking along -Z. Image pixels: top-left origin.
    let projection = simd_float4x4(columns: (
        SIMD4(2 * k[0] / width, 0, 0, 0),
        SIMD4(0, 2 * k[4] / height, 0, 0),
        SIMD4(1 - 2 * k[6] / width, 2 * k[7] / height - 1, -(far + near) / (far - near), -1),
        SIMD4(0, 0, -2 * far * near / (far - near), 0)))
    camera.camera!.zNear = Double(near)
    camera.camera!.zFar = Double(far)
    camera.camera!.projectionTransform = SCNMatrix4(projection)

    let source = project.appendingPathComponent(record["imagePath"] as! String)
    guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil),
          let cgImage = CGImageSourceCreateThumbnailAtIndex(imageSource, CGImageSourceGetPrimaryImageIndex(imageSource), [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: upright || inputOrientation == 1,
            kCGImageSourceThumbnailMaxPixelSize: 900,
            kCGImageSourceDecodeRequest: kCGImageSourceDecodeToSDR
          ] as CFDictionary) else { fail("Cannot preview source photo \(id).") }
    // Keep the reference image's actual aspect ratio; never stretch it to fit a hypothesis.
    let referenceSize = CGSize(width: cgImage.width, height: cgImage.height)
    guard abs(Float(cgImage.width) / Float(cgImage.height) - width / height) < 0.005 else {
        fail("Reference photo \(id) does not match the tested input frame.")
    }
    try writePNG(NSImage(cgImage: cgImage, size: referenceSize), to: output.appendingPathComponent("\(id)-photo.png"))

    for (name, transform) in [("raw", t), ("inverse", t.inverse), ("raw-flip-yz", t * flipYZ)] {
        camera.simdTransform = transform
        let rendered = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
        try writePNG(rendered, to: output.appendingPathComponent("\(id)-\(name).png"))
        var maximumPixelError: Float = 0
        // Independently exercise SceneKit's actual projection, including principal-point offset.
        for pixel in [SIMD2(k[6], k[7]), SIMD2(width * 0.1, height * 0.2), SIMD2(width * 0.8, height * 0.7)] {
            let local = SIMD4((pixel.x - k[6]) / k[0], (k[7] - pixel.y) / k[4], -1, 1)
            let world = transform * local
            let projected = renderer.projectPoint(SCNVector3(world.x, world.y, world.z))
            let expected = SIMD2(pixel.x / width * Float(renderWidth), (1 - pixel.y / height) * Float(renderHeight))
            maximumPixelError = max(maximumPixelError, simd_length(SIMD2(Float(projected.x), Float(projected.y)) - expected))
        }
        var maximumOrientationError: Float = 0
        if name == "raw" {
            // Start from rays in the original encoded camera. This independently checks
            // that camera roll and principal-point conversion implement the EXIF pixel map.
            for pixel in [SIMD2(encodedK[6], encodedK[7]), SIMD2(inputWidth * 0.1, inputHeight * 0.2), SIMD2(inputWidth * 0.8, inputHeight * 0.7)] {
                let ray = SIMD4((pixel.x - encodedK[6]) / encodedK[0], (encodedK[7] - pixel.y) / encodedK[4], -1, 1)
                let world = encodedPose * ray
                let projected = renderer.projectPoint(SCNVector3(world.x, world.y, world.z))
                let displayed = upright && inputOrientation == 6 ? SIMD2(inputHeight - 1 - pixel.y, pixel.x) : pixel
                let expected = SIMD2(displayed.x / width * Float(renderWidth), (1 - displayed.y / height) * Float(renderHeight))
                maximumOrientationError = max(maximumOrientationError, simd_length(SIMD2(Float(projected.x), Float(projected.y)) - expected))
            }
        }
        reports.append([
            "sampleID": id, "photoID": record["id"]!, "originalName": record["originalName"]!,
            "candidate": name, "width": width, "height": height,
            "inputWidth": inputWidth, "inputHeight": inputHeight, "inputOrientation": inputOrientation,
            "frameInterpretation": upright ? "display-oriented" : "encoded-input",
            "renderWidth": renderWidth, "renderHeight": renderHeight,
            "principalPoint": [k[6], k[7]], "focalPixels": [k[0], k[4]],
            "maximumSyntheticProjectionErrorPixels": maximumPixelError,
            "orientationMappingChecked": name == "raw",
            "maximumOrientationMappingErrorPixels": maximumOrientationError,
            "poseDeterminant": simd_determinant(transform)
        ])
    }
}
let maximumError = reports.compactMap { $0["maximumSyntheticProjectionErrorPixels"] as? Float }.max() ?? .infinity
let maximumOrientationError = reports.compactMap { $0["maximumOrientationMappingErrorPixels"] as? Float }.max() ?? .infinity
let projectionPassed = maximumError.isFinite && maximumError < 0.25
    && maximumOrientationError.isFinite && maximumOrientationError < 0.25
let report: [String: Any] = [
    "kind": "offline-coordinate-and-projection-probe", "formatVersion": 2,
    "sourceProjectID": manifest["id"]!,
    "syntheticProjectionPassed": projectionPassed,
    "limitations": ["Input orientations 1 and 6 only", "Intrinsics interpreted in the encoded input frame; verify visually for each dataset", "No lens-distortion correction", "SceneKit experiment; RealityKit projection still needs validation", "Synthetic projection errors do not measure photo/model alignment"],
    "records": reports
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    .write(to: output.appendingPathComponent("projection-probe.json"), options: .atomic)
print("Rendered three photos under three pose conventions. Inspect the PNG files; no convention is selected automatically.")
print("Maximum synthetic projection error: \(maximumError) px")
print("Maximum orientation mapping error: \(maximumOrientationError) px")
guard projectionPassed else { fail("Synthetic projection error exceeds 0.25 px; inspect the report.") }
