// Independent USDZ visual verification through SceneKit rather than the app's RealityKit viewer.
import AppKit
import Metal
import SceneKit

guard CommandLine.arguments.count == 3 else {
    print("Usage: swift scripts/render-model.swift MODEL.usdz OUTPUT.png")
    exit(2)
}
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let scene = try SCNScene(url: url)
scene.rootNode.enumerateChildNodes { node, _ in
    for material in node.geometry?.materials ?? [] {
        print("Material: \(material.name ?? "unnamed"), lighting: \(material.lightingModel), diffuse: \(String(describing: material.diffuse.contents))")
    }
}
let (minimum, maximum) = scene.rootNode.boundingBox
let center = SCNVector3((minimum.x + maximum.x) / 2, (minimum.y + maximum.y) / 2, (minimum.z + maximum.z) / 2)
let extent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z)
let camera = SCNNode()
camera.camera = SCNCamera()
camera.camera?.zNear = Double(max(extent / 1000, 0.0001))
camera.camera?.zFar = Double(max(extent * 20, 10))
camera.position = SCNVector3(center.x + extent * 1.1, center.y + extent * 0.8, center.z + extent * 1.8)
camera.look(at: center)
scene.rootNode.addChildNode(camera)
let ambient = SCNNode()
ambient.light = SCNLight()
ambient.light?.type = .ambient
ambient.light?.intensity = 100
scene.rootNode.addChildNode(ambient)
let key = SCNNode()
key.light = SCNLight()
key.light?.type = .directional
key.light?.intensity = 800
key.position = camera.position
key.orientation = camera.orientation
scene.rootNode.addChildNode(key)
scene.background.contents = NSColor(calibratedWhite: 0.15, alpha: 1)
let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
renderer.scene = scene
renderer.pointOfView = camera
let image = renderer.snapshot(atTime: 0, with: CGSize(width: 1200, height: 1000), antialiasingMode: .multisampling4X)
guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Could not encode the independent render")
}
try png.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
print("Rendered \(url.lastPathComponent) to \(CommandLine.arguments[2])")
