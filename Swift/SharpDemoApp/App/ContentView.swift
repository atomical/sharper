import CoreML
import Foundation
import ImageIO
import Metal
import SwiftUI
import UniformTypeIdentifiers

import GaussianSplatMetalRenderer
import SharpCoreML
import simd

private enum SharpDemoAppUIError: Error {
    case missingModel
    case missingImage
    case metalUnavailable
    case copyFailed(URL)
}

private enum ComputeUnitsChoice: String, CaseIterable, Identifiable {
    case all
    case cpuOnly = "cpu_only"
    case cpuAndGPU = "cpu_and_gpu"
    case cpuAndNE = "cpu_and_ne"

    var id: String { rawValue }

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuOnly: .cpuOnly
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNE: .cpuAndNeuralEngine
        }
    }
}

struct ContentView: View {
    @State private var modelURL: URL?
    @State private var imageURL: URL?

    @State private var showModelPicker = false
    @State private var showImagePicker = false

    @State private var computeUnits: ComputeUnitsChoice = .all

    @State private var renderSize: Int = 512
    @State private var orbitAngleDeg: Double = 0.0

    @State private var status: String = "Select a CoreML model (.mlpackage) and an input image. Simulator auto-loads from Documents/SharpDemo when present."
    @State private var timings: SharpTimings?

    @State private var lastPLYURL: URL?
    @State private var lastRenderCGImage: CGImage?
    @State private var didApplyDefaults = false

    private struct RunResult: Sendable {
        var status: String
        var timings: SharpTimings?
        var plyURL: URL
        var renderPNG: Data?
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("SHARP Demo (iOS/visionOS)")
                .font(.title2)

            GroupBox("Inputs") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Button("Select Model (.mlpackage)") { showModelPicker = true }
                        Text(modelURL?.lastPathComponent ?? "none").foregroundStyle(.secondary)
                    }
                    HStack {
                        Button("Select Image") { showImagePicker = true }
                        Text(imageURL?.lastPathComponent ?? "none").foregroundStyle(.secondary)
                    }

                    Picker("Compute Units", selection: $computeUnits) {
                        ForEach(ComputeUnitsChoice.allCases) { c in
                            Text(c.rawValue).tag(c)
                        }
                    }
                    .pickerStyle(.segmented)
                }
            }

            GroupBox("Render") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Size")
                        Spacer()
                        Picker("Size", selection: $renderSize) {
                            Text("256").tag(256)
                            Text("512").tag(512)
                            Text("768").tag(768)
                        }
                        .pickerStyle(.segmented)
                    }
                    HStack {
                        Text("Orbit")
                        Slider(value: $orbitAngleDeg, in: 0...360, step: 1)
                        Text("\(Int(orbitAngleDeg))°").frame(width: 48, alignment: .trailing)
                    }

                    HStack {
                        Button("Predict → PLY") { runPredictOnly() }
                        Button("Predict + Render Frame") { runPredictAndRender() }
                            .buttonStyle(.borderedProminent)
                    }
                }
            }

            GroupBox("Status") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(status).font(.callout)
                    if let timings {
                        Text(
                            String(
                                format: "timings: preprocess=%.3fs coreml=%.3fs post=%.3fs total=%.3fs",
                                timings.preprocessSec,
                                timings.coremlSec,
                                timings.postprocessSec,
                                timings.totalSec
                            )
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    if let url = lastPLYURL {
                        Text("PLY: \(url.lastPathComponent)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let cg = lastRenderCGImage {
                Divider()
                Image(decorative: cg, scale: 1.0)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black.opacity(0.95))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
        .padding()
        .onAppear {
            applyDefaultSelectionsIfNeeded()
        }
        .fileImporter(
            isPresented: $showModelPicker,
            allowedContentTypes: [UTType(filenameExtension: "mlpackage") ?? .data],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first {
                modelURL = url
            }
        }
        .fileImporter(
            isPresented: $showImagePicker,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first {
                imageURL = url
            }
        }
    }

    @MainActor
    private func applyDefaultSelectionsIfNeeded() {
        guard !didApplyDefaults else { return }
        didApplyDefaults = true
#if targetEnvironment(simulator)
        let defaults = ContentView.defaultSelection()
        var didSetModel = false
        var didSetImage = false

        if modelURL == nil, let model = defaults.modelURL {
            modelURL = model
            didSetModel = true
        }
        if imageURL == nil, let image = defaults.imageURL {
            imageURL = image
            didSetImage = true
        }
        if lastPLYURL == nil, let ply = defaults.plyURL {
            lastPLYURL = ply
        }

        if didSetModel || didSetImage {
            if modelURL != nil && imageURL != nil {
                status = "Ready (defaults loaded from Documents/SharpDemo)."
            } else if modelURL != nil {
                status = "Loaded default model; select an input image."
            } else if imageURL != nil {
                status = "Loaded default image; select a CoreML model."
            }
        }
#endif
    }

    private func runPredictOnly() {
        Task { await run(predict: true, render: false) }
    }

    private func runPredictAndRender() {
        Task { await run(predict: true, render: true) }
    }

    private func run(predict: Bool, render: Bool) async {
        do {
            guard let modelURL else { throw SharpDemoAppUIError.missingModel }
            guard let imageURL else { throw SharpDemoAppUIError.missingImage }
            let computeUnits = computeUnits.mlComputeUnits
            let renderSize = renderSize
            let orbitAngleDeg = orbitAngleDeg

            await MainActor.run {
                status = "Preparing…"
                timings = nil
                lastPLYURL = nil
                if predict { lastRenderCGImage = nil }
            }

            let localModelURL = try await copyToLocalURL(modelURL, basename: "Sharp", extHint: "mlpackage")
            let localImageURL = try await copyToLocalURL(imageURL, basename: "Image", extHint: imageURL.pathExtension)

            let outDir = try ensureOutputDirectory()
            let plyURL = outDir.appendingPathComponent("scene.ply")

            let result: RunResult = try await Task.detached(priority: .userInitiated) {
                guard let device = MTLCreateSystemDefaultDevice() else { throw SharpDemoAppUIError.metalUnavailable }
                let runner = try SharpCoreMLRunner(modelURL: localModelURL, computeUnits: computeUnits)
                let prediction = try runner.predict(imageURL: localImageURL)

                try SharpPLYWriter.write(prediction: prediction, to: plyURL)

                var png: Data? = nil
                var status = "Predicted \(prediction.postprocessed.count) gaussians."

                if render {
                    let scene = try ContentView.makeScene(device: device, prediction: prediction)
                    let (center, radius) = ContentView.computeBounds(scene: scene)

                    let fx = prediction.metadata.focalLengthPx * Float(renderSize) / Float(prediction.metadata.imageWidth)
                    let fy = prediction.metadata.focalLengthPx * Float(renderSize) / Float(prediction.metadata.imageHeight)
                    let cx = Float(renderSize) * 0.5
                    let cy = Float(renderSize) * 0.5

                    let ang = Float(orbitAngleDeg) * Float.pi / 180.0
                    let eye = center + SIMD3<Float>(radius * sin(ang), 0, radius * cos(ang))
                    let view = PinholeCamera.lookAt(eye: eye, target: center)
                    let cam = PinholeCamera(viewMatrix: view, fx: fx, fy: fy, cx: cx, cy: cy)

                    let renderer = try GaussianSplatRenderer(device: device)
                    let cg = try renderer.renderToCGImage(scene: scene, camera: cam, width: renderSize, height: renderSize)
                    png = SharpImagePNG.encode(cg)
                    status = "Rendered frame at \(renderSize)x\(renderSize)."
                }

                return RunResult(status: status, timings: prediction.timings, plyURL: plyURL, renderPNG: png)
            }.value

            await MainActor.run {
                timings = result.timings
                lastPLYURL = result.plyURL
                status = result.status
                if let data = result.renderPNG, let cg = SharpImagePNG.decode(data) {
                    lastRenderCGImage = cg
                }
            }
        } catch {
            await MainActor.run {
                status = "Error: \(error)"
            }
        }
    }

    private func ensureOutputDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let out = base.appendingPathComponent("SharpDemo", isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        return out
    }

    private func copyToLocalURL(_ url: URL, basename: String, extHint: String) async throws -> URL {
        try await withSecurityScopedAccess(url: url) {
            let tmp = FileManager.default.temporaryDirectory
            let dst = tmp.appendingPathComponent("\(basename)_\(UUID().uuidString).\(extHint)")
            try? FileManager.default.removeItem(at: dst)
            do {
                try FileManager.default.copyItem(at: url, to: dst)
            } catch {
                throw SharpDemoAppUIError.copyFailed(url)
            }
            return dst
        }
    }

    private func withSecurityScopedAccess<T>(url: URL, _ body: () throws -> T) async throws -> T {
        let started = url.startAccessingSecurityScopedResource()
        defer {
            if started { url.stopAccessingSecurityScopedResource() }
        }
        return try body()
    }

    private static func makeScene(device: MTLDevice, prediction: SharpPrediction) throws -> GaussianScene {
        let count = prediction.postprocessed.count

        let colors = prediction.raw.colorsLinearPre
        let opacities = prediction.raw.opacitiesPre

        let colorsBytes = count * 3 * MemoryLayout<Float>.size
        let opacitiesBytes = count * MemoryLayout<Float>.size

        guard let colorsBuf = device.makeBuffer(bytes: colors.dataPointer, length: colorsBytes, options: .storageModeShared),
              let opacitiesBuf = device.makeBuffer(bytes: opacities.dataPointer, length: opacitiesBytes, options: .storageModeShared)
        else {
            throw SharpDemoAppUIError.metalUnavailable
        }

        return GaussianScene(
            count: count,
            means: prediction.postprocessed.mean,
            quaternions: prediction.postprocessed.quaternions,
            scales: prediction.postprocessed.singularValues,
            colorsLinear: colorsBuf,
            opacities: opacitiesBuf
        )
    }

    private static func computeBounds(scene: GaussianScene) -> (SIMD3<Float>, Float) {
        let meanPtr = scene.means.contents().bindMemory(to: Float.self, capacity: scene.count * 3)
        var minV = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxV = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in 0..<scene.count {
            let x = meanPtr[i * 3 + 0]
            let y = meanPtr[i * 3 + 1]
            let z = meanPtr[i * 3 + 2]
            minV = SIMD3<Float>(min(minV.x, x), min(minV.y, y), min(minV.z, z))
            maxV = SIMD3<Float>(max(maxV.x, x), max(maxV.y, y), max(maxV.z, z))
        }
        let center = (minV + maxV) * 0.5
        let ext = maxV - minV
        let radius = max(0.5, simd_length(ext) * 0.6)
        return (center, radius)
    }

    private struct DefaultSelection {
        let modelURL: URL?
        let imageURL: URL?
        let plyURL: URL?
    }

    private static func defaultSelection() -> DefaultSelection {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return DefaultSelection(modelURL: nil, imageURL: nil, plyURL: nil)
        }

        let demoDir = docs.appendingPathComponent("SharpDemo", isDirectory: true)
        let modelCandidates = [
            demoDir.appendingPathComponent("Sharp.mlpackage"),
            docs.appendingPathComponent("Sharp.mlpackage"),
        ]
        let modelURL = modelCandidates.first(where: { existingPath($0) })
            ?? newestURL(in: [demoDir, docs], matchingExtensions: ["mlpackage"])

        let imageDirs = [
            demoDir.appendingPathComponent("inputs", isDirectory: true),
            demoDir,
            docs,
        ]
        let imageURL = newestURL(in: imageDirs, matchingExtensions: ["jpg", "jpeg", "png", "heic"])
        let plyURL = newestURL(in: [demoDir], matchingExtensions: ["ply"])

        return DefaultSelection(modelURL: modelURL, imageURL: imageURL, plyURL: plyURL)
    }

    private static func existingPath(_ url: URL) -> Bool {
        var isDir = ObjCBool(false)
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
    }

    private static func newestURL(in directories: [URL], matchingExtensions: [String]) -> URL? {
        let fm = FileManager.default
        let exts = Set(matchingExtensions.map { $0.lowercased() })
        var newest: (url: URL, date: Date)?

        for dir in directories {
            guard let urls = try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            for url in urls {
                let ext = url.pathExtension.lowercased()
                guard exts.contains(ext) else { continue }
                let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                if newest == nil || date > newest!.date {
                    newest = (url, date)
                }
            }
        }

        return newest?.url
    }
}

private enum SharpImagePNG {
    static func encode(_ cg: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    static func decode(_ data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}
