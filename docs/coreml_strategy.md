# CoreML Conversion Strategy

This project targets **functional parity** with `apple/ml-sharp` inference (`sharp predict`) on Apple platforms using **CoreML + Metal** (no CUDA).

## Constraints That Drive the Strategy

- The upstream postprocess step `unproject_gaussians()` re-decomposes per-Gaussian covariance using CPU `float64` SVD + SciPy quaternion conversion; that is not a realistic candidate for inclusion inside a CoreML graph.
- Output tensors are large (`N=1_179_648` Gaussians). Returning them is heavy but feasible on macOS; iOS/visionOS requires careful memory handling (zero-copy where possible, `MTLBuffer` interop).

## Strategy A (Preferred): “Monolith” Predictor CoreML

Convert the full `RGBGaussianPredictor` forward pass to CoreML:

**CoreML model**:
- Inputs: `image` (`float32[1,3,1536,1536]`), `disparity_factor` (`float32[1]`)
- Outputs: `mean_vectors_pre`, `quaternions_pre`, `singular_values_pre`, `colors_linear_pre`, `opacities_pre`
- Backend: **ML Program** (coremltools), start FP32 then FP16.

**Swift**:
- Implements preprocessing exactly (see `docs/io_contract.md`)
- Runs CoreML
- Applies the *postprocess-only* unprojection step (mean + covariance transform + re-decompose) to produce PLY-ready gaussians
- Exports `scene.ply` matching `ml-sharp` field names and metadata
- Renders with Metal Gaussian splatting

Why this is “monolith”:
- CoreML contains the entire learned model (encoders/decoders/heads/composer) and produces the full Gaussian parameter tensors; Swift handles only deterministic, non-learned geometric conversions and file export.

## Strategy B (Fallback): Split Model + Swift Reconstruction

If conversion or runtime is blocked, split at a boundary with simpler ops and smaller outputs:

**CoreML model** outputs *intermediate* tensors, for example:
- Monodepth disparity (and optionally encoder/decoder feature maps)
- Gaussian base values (initializer output)
- Delta maps (prediction head output)

**Swift/Metal** performs:
- Gaussian composition (delta + base + activations)
- Optional upsampling of deltas
- Unprojection + PLY export + rendering

This increases Swift/Metal complexity but reduces conversion risk and allows more control over memory.

## Fallback Triggers

Use Strategy B if any of the following occur with Strategy A:
- CoreML conversion fails due to unsupported ops that cannot be replaced cleanly.
- CoreML runtime memory spikes or output tensors exceed practical device limits (especially on iOS/visionOS).
- Performance is unacceptable even after FP16 weight compression.

## Precision Plan

1. **FP32** end-to-end for parity validation.
2. **FP16 weights** (ML Program) once parity tolerances are met.
3. Quantization is considered only if required for performance/memory, and will be documented explicitly.

