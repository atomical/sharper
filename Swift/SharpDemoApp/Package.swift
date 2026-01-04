// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SharpDemoApp",
    platforms: [
        .macOS(.v13),
    ],
    dependencies: [
        .package(path: "../SharpCoreML"),
        .package(path: "../GaussianSplatMetalRenderer"),
    ],
    targets: [
        .executableTarget(
            name: "SharpDemoApp",
            dependencies: [
                "SharpCoreML",
                "GaussianSplatMetalRenderer",
            ]
        )
    ]
)

