// Offscreen verification of an actual USDZ from all inferred original-photo cameras.
import AppKit
import Foundation
import Metal
import SceneKit
import simd

guard CommandLine.arguments.count == 5 else {
    print("Usage: render-sparse-model MODEL.usdz NEW_OUTPUT_DIRECTORY CAMERAS.json FUSION_REPORT.json")
    exit(2)
}
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard !FileManager.default.fileExists(atPath: output.path) else { fatalError("Output must be new") }
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let cameraDoc = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3]))) as! [String: Any]
let cameras = cameraDoc["cameras"] as! [[String: Any]]
let report = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[4]))) as! [String: Any]
func matrix(_ rows: [[Double]]) -> simd_float4x4 {
    simd_float4x4(columns: (SIMD4<Float>(rows.map { Float($0[0]) }), SIMD4<Float>(rows.map { Float($0[1]) }),
                            SIMD4<Float>(rows.map { Float($0[2]) }), SIMD4<Float>(rows.map { Float($0[3]) })))
}
let display = matrix(report["worldToDisplay"] as! [[Double]])
let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))
let scene = try SCNScene(url: modelURL)
let modelRoot = SCNNode()
for child in scene.rootNode.childNodes { modelRoot.addChildNode(child) }
scene.rootNode.addChildNode(modelRoot)
let camera = SCNNode()
camera.camera = SCNCamera()
camera.camera?.zNear = 0.001
camera.camera?.zFar = 100
scene.rootNode.addChildNode(camera)
let ambient = SCNNode()
ambient.light = SCNLight()
ambient.light?.type = .ambient
ambient.light?.intensity = 250
scene.rootNode.addChildNode(ambient)
let key = SCNNode()
key.light = SCNLight()
key.light?.type = .directional
key.light?.intensity = 900
scene.rootNode.addChildNode(key)
let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
renderer.scene = scene
renderer.pointOfView = camera
scene.background.contents = NSColor.black
var originalMaterials: [(SCNGeometry, [SCNMaterial])] = []
modelRoot.enumerateChildNodes { node, _ in
    if let geometry = node.geometry { originalMaterials.append((geometry, geometry.materials)) }
}
func setMode(_ mode: String) {
    for (geometry, materials) in originalMaterials {
        geometry.materials = materials.map { original in
            let material = original.copy() as! SCNMaterial
            material.isDoubleSided = true
            if mode == "color" { material.lightingModel = .constant }
            else {
                material.diffuse.contents = mode == "mask" ? NSColor.white : NSColor(calibratedWhite: 0.68, alpha: 1)
                material.lightingModel = mode == "mask" ? .constant : .lambert
            }
            return material
        }
    }
}
func save(_ name: String, width: Int, height: Int) throws {
    let image = renderer.snapshot(atTime: 0, with: CGSize(width: width, height: height), antialiasingMode: .multisampling4X)
    let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
    try bitmap.representation(using: .png, properties: [:])!.write(to: output.appendingPathComponent(name))
}
for (index, record) in cameras.enumerated() {
    var rows = record["worldToCamera"] as! [[Double]]
    rows.append([0, 0, 0, 1])
    let intrinsic = record["modelIntrinsics"] as! [[Double]]
    let width = intrinsic[0][2] * 2, height = intrinsic[1][2] * 2
    camera.simdTransform = display * simd_inverse(matrix(rows)) * flip
    let near = 0.001, far = 100.0
    let projection = matrix([[2 * intrinsic[0][0] / width, 0, 1 - 2 * intrinsic[0][2] / width, 0],
                             [0, 2 * intrinsic[1][1] / height, 2 * intrinsic[1][2] / height - 1, 0],
                             [0, 0, -(far + near)/(far - near), -2 * far * near/(far - near)],
                             [0, 0, -1, 0]])
    camera.camera?.projectionTransform = SCNMatrix4(projection)
    key.simdTransform = camera.simdTransform
    for mode in ["color", "clay", "mask"] {
        setMode(mode)
        try save("view-\(index)-\(mode).png", width: Int(width), height: Int(height))
    }
    print("Rendered original camera \(index): \(record["name"]!)")
}
// A turntable of the same fixed model, without textures to expose its actual volume.
let (lower, upper) = modelRoot.boundingBox
let center = SCNVector3((lower.x+upper.x)/2, (lower.y+upper.y)/2, (lower.z+upper.z)/2)
let extent = max(upper.x-lower.x, upper.y-lower.y, upper.z-lower.z)
camera.camera = SCNCamera()
camera.camera?.fieldOfView = 42
camera.camera?.zNear = 0.001
camera.camera?.zFar = 100
setMode("clay")
scene.background.contents = NSColor(calibratedWhite: 0.1, alpha: 1)
for i in 0..<36 {
    let angle = Double(i) * 2 * Double.pi / 36
    camera.simdTransform = matrix_identity_float4x4
    let x = center.x + CGFloat(sin(angle)) * extent * 1.65
    let y = center.y + extent * 0.15
    let z = center.z + CGFloat(cos(angle)) * extent * 1.65
    camera.position = SCNVector3(x, y, z)
    camera.look(at: center)
    key.simdTransform = camera.simdTransform
    try save(String(format: "turntable-%02d.png", i), width: 640, height: 640)
}
print("Rendered seven-camera comparisons and an untextured turntable")
