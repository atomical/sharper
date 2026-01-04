// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "GaussianSplatMetalRenderer",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "GaussianSplatMetalRenderer", targets: ["GaussianSplatMetalRenderer"]),
    ],
    targets: [
        .target(
            name: "GaussianSplatMetalRenderer",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "GaussianSplatMetalRendererTests",
            dependencies: ["GaussianSplatMetalRenderer"]
        ),
    ]
)

