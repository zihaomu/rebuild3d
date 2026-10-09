// Derived from ekarad1um/Photogrammetry's RealityKit preview (MIT).
// Copyright (c) 2022 ekarad1um. See Vendor/Photogrammetry/LICENSE.
import AppKit
import RealityKit
import SwiftUI

struct ModelViewport: NSViewRepresentable {
    let url: URL
    let resetID: UUID
    var onError: (String) -> Void

    func makeNSView(context: Context) -> OrbitModelView { OrbitModelView(frame: .zero) }
    func updateNSView(_ view: OrbitModelView, context: Context) {
        view.load(url, onError: onError)
        if view.resetID != resetID { view.resetID = resetID; view.resetCamera() }
    }
}

@MainActor
final class OrbitModelView: ARView {
    private let camera = PerspectiveCamera()
    private let modelAnchor = AnchorEntity(world: .zero)
    private var loadedURL: URL?
    private var loadTask: Task<Void, Never>?
    private var center = SIMD3<Float>.zero
    private var pan = SIMD3<Float>.zero
    private var radius: Float = 1
    private var distance: Float = 3
    private var yaw: Float = 0
    private var pitch: Float = 0.15
    private var lastDragLocation: NSPoint?
    private let loadingLabel = NSTextField(labelWithString: "正在加载模型…")
    var resetID = UUID()

    required init(frame: NSRect) {
        super.init(frame: frame)
        environment.background = .color(NSColor(calibratedWhite: 0.12, alpha: 1))
        camera.camera.fieldOfViewInDegrees = 50
        let anchor = AnchorEntity(world: .zero)
        anchor.addChild(camera)
        scene.anchors.append(anchor)
        scene.anchors.append(modelAnchor)
        loadingLabel.drawsBackground = true
        loadingLabel.backgroundColor = .windowBackgroundColor
        loadingLabel.alignment = .center
        loadingLabel.isHidden = true
        loadingLabel.translatesAutoresizingMaskIntoConstraints = false
        addSubview(loadingLabel)
        NSLayoutConstraint.activate([
            loadingLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            loadingLabel.centerYAnchor.constraint(equalTo: centerYAnchor),
            loadingLabel.widthAnchor.constraint(equalToConstant: 160)
        ])
        updateCamera()
    }

    @available(*, unavailable)
    required dynamic init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    func load(_ url: URL, onError: @escaping (String) -> Void) {
        guard loadedURL != url else { return }
        loadedURL = url
        loadTask?.cancel()
        // Clear the previous source rendering before the new mode's label is shown.
        modelAnchor.children.removeAll()
        loadingLabel.isHidden = false
        loadTask = Task {
            do {
                let entity = try await Entity(contentsOf: url)
                guard !Task.isCancelled else { return }
                modelAnchor.children.removeAll()
                modelAnchor.addChild(entity)
                loadingLabel.isHidden = true
                let bounds = entity.visualBounds(relativeTo: nil)
                center = bounds.center
                radius = max(bounds.boundingRadius, 0.001)
                // Preserve model coordinates. Only move the viewer camera around the original bounds.
                resetCamera()
            } catch {
                guard !Task.isCancelled else { return }
                loadingLabel.isHidden = true
                onError("模型无法显示：\(error.localizedDescription)")
            }
        }
    }

    func resetCamera() {
        yaw = 0
        pitch = 0.15
        pan = .zero
        let aspect = Float(max(bounds.width, 1) / max(bounds.height, 1))
        let verticalHalfAngle: Float = 25 * .pi / 180
        let limitingAngle = min(verticalHalfAngle, atan(tan(verticalHalfAngle) * aspect))
        distance = radius / sin(limitingAngle) * 1.15
        updateCamera()
    }

    private func updateCamera() {
        let target = center + pan
        let offset = SIMD3<Float>(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) * distance
        camera.look(at: target, from: target + offset, relativeTo: nil)
        camera.camera.near = max(0.0001, distance / 1000)
        camera.camera.far = max(100, distance + radius * 20)
    }

    private func beginDrag(_ event: NSEvent) {
        lastDragLocation = event.locationInWindow
        window?.makeFirstResponder(self)
    }
    override func mouseDown(with event: NSEvent) { beginDrag(event) }
    override func rightMouseDown(with event: NSEvent) { beginDrag(event) }
    override func otherMouseDown(with event: NSEvent) { beginDrag(event) }
    override func mouseUp(with event: NSEvent) { lastDragLocation = nil }
    override func rightMouseUp(with event: NSEvent) { lastDragLocation = nil }
    override func otherMouseUp(with event: NSEvent) { lastDragLocation = nil }

    private func dragDelta(_ event: NSEvent) -> SIMD2<Float> {
        let location = event.locationInWindow
        defer { lastDragLocation = location }
        guard let previous = lastDragLocation else { return .zero }
        // Absolute pointer events can have zero deltaX/deltaY despite a changed location.
        // Window coordinates point upward; keep the existing downward-positive drag convention.
        return SIMD2(Float(location.x - previous.x), Float(previous.y - location.y))
    }

    override func mouseDragged(with event: NSEvent) {
        let delta = dragDelta(event)
        if event.modifierFlags.contains(.shift) { panCamera(delta) }
        else {
            yaw -= delta.x * 0.01
            pitch = min(1.5, max(-1.5, pitch + delta.y * 0.01))
            updateCamera()
        }
    }
    override func rightMouseDragged(with event: NSEvent) { panCamera(dragDelta(event)) }
    override func otherMouseDragged(with event: NSEvent) { panCamera(dragDelta(event)) }
    override func scrollWheel(with event: NSEvent) {
        distance = min(radius * 100, max(radius * 0.15, distance * exp(Float(event.scrollingDeltaY) * 0.01)))
        updateCamera()
    }
    override func magnify(with event: NSEvent) {
        distance = min(radius * 100, max(radius * 0.15, distance * Float(1 - event.magnification)))
        updateCamera()
    }

    private func panCamera(_ delta: SIMD2<Float>) {
        let transform = camera.transform.matrix
        let right = SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z)
        let up = SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z)
        let scale = distance * 0.002
        pan += (-right * delta.x + up * delta.y) * scale
        updateCamera()
    }
}
