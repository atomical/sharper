SHELL := /bin/bash

.PHONY: help venv fixtures ref export coreml validate demo bench clean
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
	@echo "  validate  - parity tests: PyTorch vs CoreML"
	@echo "  demo      - build/run Swift demo (prints instructions)"
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

ref: venv
	$(VENV_PY) tools/export/ref_infer.py --fixtures artifacts/fixtures/inputs --out-root artifacts/fixtures/ref

export: venv
	$(VENV_PY) tools/export/export_sharp.py

coreml: venv
	$(VENV_PY) tools/coreml/convert_to_coreml.py

validate: venv
	$(VENV_PY) tools/coreml/validate_coreml.py --fixtures artifacts/fixtures/inputs --ref-root artifacts/fixtures/ref --coreml-root artifacts/fixtures/coreml

demo:
	@echo "Open the Xcode workspace/project under Swift/SharpDemoApp and build/run."

bench: venv
	$(VENV_PY) tools/coreml/bench_coreml.py

clean:
	rm -rf $(VENV_DIR) artifacts/fixtures/ref artifacts/fixtures/coreml artifacts/benches
