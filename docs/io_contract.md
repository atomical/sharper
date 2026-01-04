# SHARP I/O Contract (Exact)

This document defines the **exact** preprocessing, model inputs/outputs, and postprocessing required to match `apple/ml-sharp` inference semantics for `sharp predict` using the official checkpoint.

Reference implementation (pinned):
- `third_party/ml-sharp` @ `1eaa046834b81852261262b41b0919f5c1efdd2e`
- Official checkpoint URL: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`

## 1) Image Preprocessing (Matches `sharp.cli.predict.predict_image`)

### 1.1 Load + RGB
- Decode the image to **RGB**.
- Use the image as-is (no mean/std normalization).
- Pixel range: `uint8` `[0, 255]` → `float32` `[0, 1]` by dividing by `255.0`.
- Channel order: **RGB**.

### 1.2 Focal Length → `f_px` (Matches `sharp.utils.io.load_rgb`)
Let the decoded image size be:
- height `H`, width `W` (pixels)

Extract focal length (in mm):
1. Prefer EXIF `FocalLengthIn35mmFilm` (or legacy `FocalLenIn35mmFilm`)
2. Else EXIF `FocalLength`
3. Else default `f_mm = 30.0`

Heuristic (matches upstream):
- If `f_mm < 10.0`, treat it as non-35mm and multiply by `8.4`.

Convert to pixels:
```
f_px = f_mm * sqrt(W^2 + H^2) / sqrt(36^2 + 24^2)
```

### 1.3 Disparity Factor (Scalar input)
```
disparity_factor = f_px / W
```

Shape/dtype:
- `disparity_factor`: `float32[1]` (batch size is 1)

### 1.4 Resize to Internal Resolution
The model runs at a fixed internal resolution:
- `INTERNAL_RES = 1536`

Resize the input image tensor:
- Input tensor shape: `float32[1, 3, H, W]`
- Output tensor shape: `float32[1, 3, 1536, 1536]`
- Interpolation: **bilinear**
- `align_corners=True` (important for parity)

No crop/pad: the image is **stretched** to a square.

## 2) CoreML Model Inputs

`Sharp.mlpackage` uses two inputs (batch size fixed to 1):

1. `image`:
   - type: `MLMultiArray` (or `MLTensor` depending on platform)
   - dtype: `float32`
   - shape: `[1, 3, 1536, 1536]` (NCHW)
   - range: `[0, 1]`
2. `disparity_factor`:
   - dtype: `float32`
   - shape: `[1]`
   - value: `f_px / W` from section 1.3

## 3) CoreML Model Outputs (Pre-Unprojection)

The CoreML model outputs Gaussians in **pre-unprojection coordinates** (matches the direct output of `RGBGaussianPredictor.forward()`).

Constants (from upstream defaults):
- `NUM_LAYERS = 2`
- `OUT_RES = INTERNAL_RES / 2 = 768`
- `N = NUM_LAYERS * OUT_RES * OUT_RES = 1_179_648`

All outputs are `float32`:

1. `mean_vectors_pre`: shape `[1, N, 3]`
   - Semantics (per Gaussian): `[x', y', z]`
   - `z` is metric depth (positive, camera-forward)
   - `x' = z * x_ndc`, `y' = z * y_ndc` where `(x_ndc, y_ndc)` are NDC-like coordinates
2. `quaternions_pre`: shape `[1, N, 4]`
   - Quaternion convention: **wxyz** (real component first)
   - Not guaranteed normalized; normalize before use if needed
3. `singular_values_pre`: shape `[1, N, 3]`
   - Positive scales along the Gaussian principal axes (same coordinate frame as `mean_vectors_pre`)
4. `colors_linear_pre`: shape `[1, N, 3]`
   - Color in **linearRGB**, range approximately `[0, 1]`
5. `opacities_pre`: shape `[1, N]`
   - Opacity after sigmoid, range `[0, 1]`

### 3.1 Output Ordering (Important)
Upstream flattens `[B, C, L, H, W]` → `[B, L, H, W, C]` → flatten `(L, H, W)` in row-major order.

Therefore, Gaussian index increases in the order:
1. layer `l` (0..`L-1`)
2. pixel `y` (0..`H-1`)
3. pixel `x` (0..`W-1`)

## 4) Postprocess to `sharp predict` Semantics (PLY-ready)

`sharp predict` applies an additional **unprojection** step before writing the PLY.

### 4.1 Unprojection Scaling (Mean Vectors)
Let `sx = W / (2 * f_px)` and `sy = H / (2 * f_px)`.

Then:
```
x = sx * x'
y = sy * y'
z = z
```

### 4.2 Unprojection Scaling (Covariance → Re-decompose)
Let:
- `R` be the `3x3` rotation matrix from `quaternions_pre` (after normalization)
- `S = diag(singular_values_pre)`
- `C = R * (S^2) * R^T` (covariance in pre-unprojection space)
- `D = diag(sx, sy, 1)`

Transform covariance:
```
C' = D * C * D
```

Then re-decompose `C'` into `(R', S')` such that:
```
C' = R' * (S'^2) * R'^T
```

Upstream implementation detail (`sharp.utils.gaussians.decompose_covariance_matrices`):
- Uses `torch.linalg.svd(C')` (on CPU, `float64`)
- Treats `U` as `R'` and `S` as eigenvalues (PSD)
- Fixes reflections: if `det(R') < 0`, flip the last column of `R'`
- Converts `R'` to quaternion **wxyz**
- Computes `singular_values = sqrt(S)`

### 4.3 PLY Export Mapping (Matches `sharp.utils.gaussians.save_ply`)
Vertex properties:
- `x,y,z` = unprojected mean vectors
- `scale_0..2` = `log(singular_values)` (natural log)
- `rot_0..3` = quaternion **wxyz**
- `opacity` = `logit(opacity)` where `logit(p)=log(p/(1-p))`
- `f_dc_0..2` = degree-0 spherical harmonics of **sRGB** color:
  - Convert `colors_linear` → `colors_srgb` using `linearRGB → sRGB`
  - Convert to SH0 via:
    - `coeff = sqrt(1 / (4π))`
    - `f_dc = (colors_srgb - 0.5) / coeff`

Supplemental PLY elements written by `ml-sharp` (must match for full compatibility):
- `image_size` (uint32): `[W, H]`
- `intrinsic` (float32, 9 elements): `[[f_px,0,W*0.5],[0,f_px,H*0.5],[0,0,1]]` flattened
- `extrinsic` (float32, 16 elements): identity `4x4` flattened
- `frame` (int32, 2 elements): `[1, N]`
- `disparity` (float32, 2 elements): 10% and 90% quantiles of `1/z`
- `color_space` (uint8): encoded for `sRGB`
- `version` (uint8, 3 elements): `[1, 5, 0]`

## 5) Coordinate System

- Camera coordinate convention is **OpenCV-style**:
  - +X right
  - +Y down
  - +Z forward
- `extrinsic` is identity in exported PLY (gaussians are already in camera/world for that view).

