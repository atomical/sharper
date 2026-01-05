SHELL := /bin/bash

.PHONY: help venv fixtures ref export coreml coreml-fp16 validate validate-fp16 validate-swift demo smoke macos-build ios-build visionos-build bench clean
.PHONY: ml-sharp

PYTHON ?= python3.11
VENV_DIR ?= .venv
VENV_DIR_ABS := $(abspath $(VENV_DIR))
VENV_PY := $(VENV_DIR_ABS)/bin/python
VENV_PIP := $(VENV_DIR_ABS)/bin/pip
VENV_STAMP := $(VENV_DIR_ABS)/.deps_installed

ML_SHARP_DIR ?= third_party/ml-sharp
# Pinned later after clone; recorded in docs/PROGRESS.md too.
ML_SHARP_REPO_URL ?= https://github.com/apple/ml-sharp.git
ML_SHARP_COMMIT ?= 1eaa046834b81852261262b41b0919f5c1efdd2e

help:
	@echo "Targets:"
	@echo "  venv      - create venv + install deps"
	@echo "  fixtures  - generate fixture images"
	@echo "  ref       - run PyTorch reference inference over fixtures"
	@echo "  export    - export/trace SHARP wrapper for CoreML"
	@echo "  coreml    - convert exported graph to CoreML .mlpackage"
	@echo "  coreml-fp16 - convert to CoreML (FP16 weights)"
	@echo "  validate  - parity tests: PyTorch vs CoreML"
	@echo "  validate-fp16 - parity tests: PyTorch vs CoreML (FP16)"
	@echo "  validate-swift - parity tests: Swift PLY vs ref"
	@echo "  demo      - build/run Swift demo (image→PLY→frames/video)"
	@echo "  smoke     - run macOS smoke test (SwiftPM)"
	@echo "  macos-build - build macOS SwiftUI demo app"
	@echo "  ios-build - build iOS SwiftUI demo app"
	@echo "  visionos-build - build visionOS SwiftUI demo app"
	@echo "  bench     - run CoreML benchmark harness"

$(VENV_PY):
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install -U pip setuptools wheel

$(VENV_STAMP): $(VENV_PY) requirements.txt
	# Install ml-sharp (pinned deps) + our CoreML tooling deps.
	(cd $(ML_SHARP_DIR) && $(VENV_PIP) install -r requirements.txt)
	$(VENV_PIP) install -r requirements.txt
	@touch $(VENV_STAMP)

ml-sharp:
	@if [ ! -d "$(ML_SHARP_DIR)/.git" ]; then \
		git clone --depth 1 "$(ML_SHARP_REPO_URL)" "$(ML_SHARP_DIR)"; \
	fi
	@cd "$(ML_SHARP_DIR)" && (git fetch --depth 1 origin "$(ML_SHARP_COMMIT)" || true)
	@git -C "$(ML_SHARP_DIR)" checkout -q "$(ML_SHARP_COMMIT)"

venv: ml-sharp $(VENV_STAMP)

fixtures: venv
	$(VENV_PY) tools/fixtures/generate_fixtures.py --out artifacts/fixtures/inputs

ref: venv fixtures
	$(VENV_PY) tools/export/ref_infer.py --fixtures artifacts/fixtures/inputs --out-root artifacts/fixtures/ref

export: venv
	$(VENV_PY) tools/export/export_sharp.py

coreml: venv export
	$(VENV_PY) tools/coreml/convert_to_coreml.py

coreml-fp16: venv export
	$(VENV_PY) tools/coreml/convert_to_coreml.py --precision fp16 --out artifacts/Sharp_fp16.mlpackage

validate: venv fixtures ref coreml
	$(VENV_PY) tools/coreml/validate_coreml.py --fixtures artifacts/fixtures/inputs --ref-root artifacts/fixtures/ref --coreml-root artifacts/fixtures/coreml

validate-fp16: venv fixtures ref coreml-fp16
	-$(VENV_PY) tools/coreml/validate_coreml.py --model artifacts/Sharp_fp16.mlpackage --compute-units all --fixtures artifacts/fixtures/inputs --ref-root artifacts/fixtures/ref --coreml-root artifacts/fixtures/coreml_fp16
	@echo "note: FP16 model parity is currently best-effort; see artifacts/fixtures/coreml_fp16/parity_report.md"

validate-swift: venv ref coreml
	cd Swift/SharpDemoApp && swift package clean && swift build -c release
	$(VENV_PY) tools/swift/validate_swift.py --fixtures artifacts/fixtures/inputs --ref-root artifacts/fixtures/ref --swift-bin Swift/SharpDemoApp/.build/release/SharpDemoApp --model artifacts/Sharp.mlpackage

SWIFT_DEMO_DIR := Swift/SharpDemoApp
SWIFT_DEMO_BIN := $(SWIFT_DEMO_DIR)/.build/release/SharpDemoApp
TIMEOUT_BIN := $(shell command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)
TIMEOUT_KILL_AFTER ?= 10s

DEMO_IMAGE ?= artifacts/fixtures/inputs/indoor_teaser.jpg
DEMO_OUT ?= artifacts/fixtures/coreml/demo
DEMO_FRAMES ?= 60
DEMO_SIZE ?= 512x512
DEMO_FPS ?= 30
DEMO_TIMEOUT ?= 300s

demo: coreml fixtures
	cd $(SWIFT_DEMO_DIR) && swift package clean && swift build -c release
	mkdir -p $(DEMO_OUT)
	@if [ -n "$(TIMEOUT_BIN)" ]; then \
		cd $(SWIFT_DEMO_DIR) && $(TIMEOUT_BIN) -k $(TIMEOUT_KILL_AFTER) $(DEMO_TIMEOUT) .build/release/SharpDemoApp ../../$(DEMO_IMAGE) ../../$(DEMO_OUT) --frames $(DEMO_FRAMES) --size $(DEMO_SIZE) --video ../../$(DEMO_OUT)/out.mp4 --fps $(DEMO_FPS); \
	else \
		echo "warning: timeout not found; running demo without a watchdog"; \
		cd $(SWIFT_DEMO_DIR) && .build/release/SharpDemoApp ../../$(DEMO_IMAGE) ../../$(DEMO_OUT) --frames $(DEMO_FRAMES) --size $(DEMO_SIZE) --video ../../$(DEMO_OUT)/out.mp4 --fps $(DEMO_FPS); \
	fi

SMOKE_FRAMES ?= 2
SMOKE_SIZE ?= 256x256
SMOKE_TIMEOUT ?= 180s
SMOKE_OUT ?= artifacts/fixtures/coreml/quick_demo

smoke:
	@if [ -n "$(TIMEOUT_BIN)" ]; then \
		$(TIMEOUT_BIN) -k $(TIMEOUT_KILL_AFTER) $(SMOKE_TIMEOUT) swift run --package-path $(SWIFT_DEMO_DIR) -c release SharpQuickDemo --frames $(SMOKE_FRAMES) --size $(SMOKE_SIZE) --out $(SMOKE_OUT); \
	else \
		echo "warning: timeout not found; running swift smoke without a watchdog"; \
		swift run --package-path $(SWIFT_DEMO_DIR) -c release SharpQuickDemo --frames $(SMOKE_FRAMES) --size $(SMOKE_SIZE) --out $(SMOKE_OUT); \
	fi

IOS_PROJECT := Swift/SharpDemoApp/SharpDemoAppUI.xcodeproj
IOS_SCHEME := SharpDemoAppUI
MACOS_SCHEME := SharpDemoAppMac
VISIONOS_SCHEME := SharpDemoAppVision

macos-build:
	xcodebuild -project "$(IOS_PROJECT)" -scheme "$(MACOS_SCHEME)" -destination "generic/platform=macOS" -configuration Debug build

ios-build:
	xcodebuild -project "$(IOS_PROJECT)" -scheme "$(IOS_SCHEME)" -destination "generic/platform=iOS Simulator" -configuration Debug build

visionos-build:
	@if ! xcrun simctl list runtimes | grep -q "visionOS"; then \
		echo "error: visionOS Simulator runtime not installed."; \
		echo "Install it via Xcode > Settings > Components, then rerun: make visionos-build"; \
		exit 2; \
	fi
	xcodebuild -project "$(IOS_PROJECT)" -scheme "$(VISIONOS_SCHEME)" -destination "generic/platform=visionOS Simulator" -configuration Debug build

bench: venv coreml fixtures
	$(VENV_PY) tools/coreml/bench_coreml.py --out artifacts/benches/bench_coreml.json
	mkdir -p artifacts/benches artifacts/benches/swift_run
	cd Swift/SharpDemoApp && swift package clean && swift build -c release
	@if [ -n "$(TIMEOUT_BIN)" ]; then \
		cd $(SWIFT_DEMO_DIR) && $(TIMEOUT_BIN) -k $(TIMEOUT_KILL_AFTER) 600s .build/release/SharpDemoApp ../../$(DEMO_IMAGE) ../../artifacts/benches/swift_run --model ../../artifacts/Sharp.mlpackage --iters 5 --frames 60 --size $(DEMO_SIZE) --bench-out ../../artifacts/benches/bench_swift.json; \
	else \
		echo "warning: timeout not found; running swift bench without a watchdog"; \
		cd $(SWIFT_DEMO_DIR) && .build/release/SharpDemoApp ../../$(DEMO_IMAGE) ../../artifacts/benches/swift_run --model ../../artifacts/Sharp.mlpackage --iters 5 --frames 60 --size $(DEMO_SIZE) --bench-out ../../artifacts/benches/bench_swift.json; \
	fi

clean:
	rm -rf $(VENV_DIR) artifacts/fixtures/ref artifacts/fixtures/coreml artifacts/benches
