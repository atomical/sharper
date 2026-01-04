// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SharpCoreML",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "SharpCoreML", targets: ["SharpCoreML"]),
    ],
    targets: [
        .target(
            name: "SharpCoreML",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "SharpCoreMLTests",
            dependencies: ["SharpCoreML"]
        ),
    ]
)

