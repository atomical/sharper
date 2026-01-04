# PLY Schema (ml-sharp compatible)

This project writes a binary little-endian PLY matching `apple/ml-sharp` (`sharp.utils.gaussians.save_ply`).

## Vertex Element: `vertex`

Type: `float32`

Per-vertex fields (in this order):

1. `x`, `y`, `z` — mean position (camera/world coordinates, OpenCV convention)
2. `f_dc_0`, `f_dc_1`, `f_dc_2` — degree-0 spherical harmonics coefficients of **sRGB** color
3. `opacity` — opacity **logit** (`log(p/(1-p))`)
4. `scale_0`, `scale_1`, `scale_2` — scale **log** (`log(singular_values)`)
5. `rot_0`, `rot_1`, `rot_2`, `rot_3` — quaternion **wxyz**

## Supplemental Elements

`ml-sharp` writes additional elements used by some renderers/tools:

- `extrinsic` (`float32`, 16 values): identity `4x4`, flattened row-major
- `intrinsic` (`float32`, 9 values): `[[f_px,0,W*0.5],[0,f_px,H*0.5],[0,0,1]]` flattened row-major
- `image_size` (`uint32`, 2 values): `[W, H]`
- `frame` (`int32`, 2 values): `[1, num_gaussians]`
- `disparity` (`float32`, 2 values): 10% and 90% quantiles of `1/z`
- `color_space` (`uint8`, 1 value): `0` for sRGB, `1` for linearRGB (export uses `0`)
- `version` (`uint8`, 3 values): `[1, 5, 0]`

