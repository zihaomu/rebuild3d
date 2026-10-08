// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Rebuild3D",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "Rebuild3D", targets: ["Rebuild3DApp"]),
        .executable(name: "rebuild3d-check", targets: ["Rebuild3DCheck"])
    ],
    targets: [
        .target(name: "Rebuild3DCore"),
        .executableTarget(name: "Rebuild3DApp", dependencies: ["Rebuild3DCore"]),
        .executableTarget(name: "Rebuild3DCheck", dependencies: ["Rebuild3DCore"]),
        .testTarget(name: "Rebuild3DTests", dependencies: ["Rebuild3DCore"])
    ]
)
