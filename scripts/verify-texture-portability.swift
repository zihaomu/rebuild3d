// Run from the relocated export directory, with no experiment paths as inputs.
import AppKit
import Foundation
import Metal
import RealityKit
import SceneKit

@main
struct VerifyTexturePortability {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            print("Usage: verify-texture-portability MODEL.usdz NEW_OUTPUT_DIRECTORY")
            exit(2)
        }
        let input = URL(fileURLWithPath: CommandLine.arguments[1])
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        guard !FileManager.default.fileExists(atPath: output.path) else {
            fatalError("Output must be new")
        }
        let entity = try await Entity(contentsOf: input)
        let bounds = entity.visualBounds(relativeTo: nil)
        guard bounds.extents.x > 0, bounds.extents.y > 0, bounds.extents.z > 0 else {
            fatalError("RealityKit loaded empty geometry")
        }
        let scene = try SCNScene(url: input)
        let model = SCNNode()
        for child in scene.rootNode.childNodes { model.addChildNode(child) }
        scene.rootNode.addChildNode(model)
        var texturedMaterials = 0
        var triangles = 0
        model.enumerateChildNodes { node, _ in
            guard let geometry = node.geometry else { return }
            triangles += geometry.elements.filter { $0.primitiveType == .triangles }
                .reduce(0) { $0 + $1.primitiveCount }
            for material in geometry.materials {
                // A URL/NSImage/CGImage here is an image resource; flat NSColor is not.
                if let content = material.diffuse.contents, !(content is NSColor) {
                    texturedMaterials += 1
                }
                material.lightingModel = .constant
                material.isDoubleSided = true
            }
        }
        guard triangles > 0, texturedMaterials > 0 else { fatalError("Missing textured triangles") }
        let (lower, upper) = model.boundingBox
        let center = SCNVector3((lower.x + upper.x) / 2, (lower.y + upper.y) / 2,
                               (lower.z + upper.z) / 2)
        let extent = max(upper.x - lower.x, upper.y - lower.y, upper.z - lower.z)
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.fieldOfView = 42
        camera.camera?.zNear = 0.001
        camera.camera?.zFar = 100
        camera.position = SCNVector3(center.x, center.y + extent * 0.1, center.z + extent * 1.65)
        camera.look(at: center)
        scene.rootNode.addChildNode(camera)
        scene.background.contents = NSColor(calibratedWhite: 0.12, alpha: 1)
        let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
        renderer.scene = scene
        renderer.pointOfView = camera
        let image = renderer.snapshot(atTime: 0, with: CGSize(width: 1200, height: 1200),
                                      antialiasingMode: .multisampling4X)
        let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try bitmap.representation(using: .png, properties: [:])!
            .write(to: output.appendingPathComponent("relocated-usdz.png"))
        let report: [String: Any] = [
            "status": "passed", "input": input.path,
            "workingDirectory": FileManager.default.currentDirectoryPath,
            "realityKitLoaded": true, "sceneKitRendered": true,
            "triangleCount": triangles, "texturedMaterialCount": texturedMaterials,
            "scope": "Asset loading and image rendering; visual result reviewed separately."
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("verification.json"))
        print(String(data: data, encoding: .utf8)!)
    }
}
