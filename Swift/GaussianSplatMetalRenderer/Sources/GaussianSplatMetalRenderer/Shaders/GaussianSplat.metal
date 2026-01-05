#include <metal_stdlib>
using namespace metal;

struct CameraParams {
    float4x4 viewMatrix;
    float fx;
    float fy;
    float cx;
    float cy;
    uint width;
    uint height;
};

struct RenderParams {
    float nearClipZ;
    float opacityThreshold;
    float lowPassEps2D;
    float minRadiusPx;

    float maxRadiusPx;
    float exposureEV;
    float saturation;
    float contrast;

    uint toneMap;   // 0=none, 1=reinhard, 2=aces
    uint debugView; // 0=none, 1=alpha, 2=depth, 3=disparity, 4=radius
    float debugNearZ;
    float debugFarZ;
};

struct SplatVertexOut {
    float4 position [[position]];
    float2 centerPx;
    float3 invCov; // symmetric 2x2: (a, b, c) => [[a,b],[b,c]]
    float3 color;
    float opacity;
    float depth;
    float radiusPx;
};

static inline float3 rotateCol0(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z;
    float wy = w * y, wz = w * z;
    return float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy));
}

static inline float3 rotateCol1(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float xx = x * x, zz = z * z;
    float xy = x * y, yz = y * z;
    float wx = w * x, wz = w * z;
    return float3(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx));
}

static inline float3 rotateCol2(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float xx = x * x, yy = y * y;
    float xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y;
    return float3(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy));
}

static inline float3 sigmaMul(float3 vec, float3 c0, float3 c1, float3 c2, float3 s2) {
    return s2.x * c0 * dot(c0, vec)
         + s2.y * c1 * dot(c1, vec)
         + s2.z * c2 * dot(c2, vec);
}

static inline SplatVertexOut splatVertexImpl(
    uint vid,
    uint iid,
    constant CameraParams &cam,
    device const packed_float3 *means,
    device const packed_float4 *quats,
    device const packed_float3 *scales,
    device const packed_float3 *colors,
    device const float *opacities,
    constant RenderParams &rp
) {
    float3 pos = float3(means[iid]);
    float4 q = float4(quats[iid]);
    float3 scl = float3(scales[iid]);
    float3 col = float3(colors[iid]);
    float opa = opacities[iid];

    // Normalize quaternion defensively.
    q = q * rsqrt(max(dot(q, q), 1e-12f));

    SplatVertexOut out;

    if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z) ||
        !isfinite(opa) || opa <= rp.opacityThreshold) {
        out.position = float4(-2.0, -2.0, 0.0, 1.0);
        out.centerPx = float2(-1e6f, -1e6f);
        out.invCov = float3(1.0f, 0.0f, 1.0f);
        out.color = float3(0.0f);
        out.opacity = 0.0f;
        out.depth = 0.0f;
        out.radiusPx = 0.0f;
        return out;
    }

    float4 pCam4 = cam.viewMatrix * float4(pos, 1.0);
    float3 pCam = pCam4.xyz;
    float z = pCam.z;
    if (!isfinite(z) || z <= rp.nearClipZ) {
        out.position = float4(-2.0, -2.0, 0.0, 1.0);
        out.centerPx = float2(-1e6f, -1e6f);
        out.invCov = float3(1.0f, 0.0f, 1.0f);
        out.color = float3(0.0f);
        out.opacity = 0.0f;
        out.depth = 0.0f;
        out.radiusPx = 0.0f;
        return out;
    }

    float u = cam.fx * (pCam.x / z) + cam.cx;
    float v = cam.fy * (pCam.y / z) + cam.cy;

    float3 s2 = scl * scl;
    float3 c0 = rotateCol0(q);
    float3 c1 = rotateCol1(q);
    float3 c2 = rotateCol2(q);

    // Perspective projection Jacobian at mean.
    float invZ = 1.0f / z;
    float invZ2 = invZ * invZ;
    float3 j0 = float3(cam.fx * invZ, 0.0f, -cam.fx * pCam.x * invZ2);
    float3 j1 = float3(0.0f, cam.fy * invZ, -cam.fy * pCam.y * invZ2);

    float3 sJ0 = sigmaMul(j0, c0, c1, c2, s2);
    float3 sJ1 = sigmaMul(j1, c0, c1, c2, s2);
    float A = dot(j0, sJ0);
    float B = dot(j0, sJ1);
    float C = dot(j1, sJ1);

    // Ensure a minimum footprint in pixel space for numerical stability (+ optional low-pass).
    float eps2d = 1e-4f + max(rp.lowPassEps2D, 0.0f);
    A += eps2d;
    C += eps2d;

    float det = A * C - B * B;
    float invDet = 1.0f / max(det, 1e-12f);
    float invA = C * invDet;
    float invB = -B * invDet;
    float invC = A * invDet;

    // Bounding radius = 3-sigma of max eigenvalue of covariance.
    float tr = A + C;
    float disc = sqrt(max((A - C) * (A - C) + 4.0f * B * B, 0.0f));
    float lam1 = 0.5f * (tr + disc);
    float lam2 = 0.5f * (tr - disc);
    float sigmaMaxPx = sqrt(max(max(lam1, lam2), 1e-8f));
    float radiusPx = clamp(3.0f * sigmaMaxPx, rp.minRadiusPx, rp.maxRadiusPx);

    // Cull offscreen quads early to reduce fragment cost.
    float wPx = float(cam.width - 1);
    float hPx = float(cam.height - 1);
    if (u + radiusPx < 0.0f || u - radiusPx > wPx || v + radiusPx < 0.0f || v - radiusPx > hPx) {
        out.position = float4(-2.0, -2.0, 0.0, 1.0);
        out.centerPx = float2(-1e6f, -1e6f);
        out.invCov = float3(1.0f, 0.0f, 1.0f);
        out.color = float3(0.0f);
        out.opacity = 0.0f;
        out.depth = 0.0f;
        out.radiusPx = 0.0f;
        return out;
    }

    float2 offsetPx;
    switch (vid) {
        case 0: offsetPx = float2(-radiusPx, -radiusPx); break;
        case 1: offsetPx = float2( radiusPx, -radiusPx); break;
        case 2: offsetPx = float2(-radiusPx,  radiusPx); break;
        default: offsetPx = float2( radiusPx,  radiusPx); break;
    }

    float widthMinus1 = max(float(cam.width - 1), 1.0f);
    float heightMinus1 = max(float(cam.height - 1), 1.0f);
    float2 ndc;
    ndc.x = (u / widthMinus1) * 2.0f - 1.0f;
    ndc.y = 1.0f - (v / heightMinus1) * 2.0f;

    float2 ndcOffset;
    ndcOffset.x = (offsetPx.x / widthMinus1) * 2.0f;
    ndcOffset.y = -(offsetPx.y / heightMinus1) * 2.0f;

    out.position = float4(ndc + ndcOffset, 0.0, 1.0);
    out.centerPx = float2(u, v);
    out.invCov = float3(invA, invB, invC);
    out.color = col;
    out.opacity = opa;
    out.depth = z;
    out.radiusPx = radiusPx;
    return out;
}

vertex SplatVertexOut splatVertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant CameraParams &cam [[buffer(0)]],
    device const packed_float3 *means [[buffer(1)]],
    device const packed_float4 *quats [[buffer(2)]],
    device const packed_float3 *scales [[buffer(3)]],
    device const packed_float3 *colors [[buffer(4)]],
    device const float *opacities [[buffer(5)]],
    constant RenderParams &rp [[buffer(6)]]
) {
    return splatVertexImpl(vid, iid, cam, means, quats, scales, colors, opacities, rp);
}

struct OITOut {
    float4 accum [[color(0)]];
    float4 reveal [[color(1)]];
    float4 aux [[color(2)]];
};

fragment OITOut splatFragmentOIT(SplatVertexOut in [[stage_in]]) {
    float2 fragPx = in.position.xy;
    float2 d = fragPx - in.centerPx;

    float a = in.invCov.x;
    float b = in.invCov.y;
    float c = in.invCov.z;
    float quad = a * d.x * d.x + 2.0f * b * d.x * d.y + c * d.y * d.y;
    float w = exp(-0.5f * quad);
    float alpha = clamp(in.opacity * w, 0.0f, 1.0f);

    OITOut out;
    out.accum = float4(in.color * alpha, alpha);
    out.reveal = float4(0.0f, 0.0f, 0.0f, alpha);
    out.aux = float4(in.depth * alpha, in.radiusPx * alpha, 0.0f, alpha);
    return out;
}

struct CompositeVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex CompositeVertexOut compositeVertex(uint vid [[vertex_id]]) {
    float2 pos;
    float2 uv;
    switch (vid) {
        case 0: pos = float2(-1.0, -1.0); uv = float2(0.0, 1.0); break;
        case 1: pos = float2( 1.0, -1.0); uv = float2(1.0, 1.0); break;
        case 2: pos = float2(-1.0,  1.0); uv = float2(0.0, 0.0); break;
        default: pos = float2( 1.0,  1.0); uv = float2(1.0, 0.0); break;
    }
    CompositeVertexOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fragment float4 compositeFragment(
    CompositeVertexOut in [[stage_in]],
    texture2d<half> accumTex [[texture(0)]],
    texture2d<half> revealTex [[texture(1)]],
    texture2d<half> auxTex [[texture(2)]],
    sampler samp [[sampler(0)]],
    constant RenderParams &rp [[buffer(0)]]
) {
    half4 accumH = accumTex.sample(samp, in.uv);
    half4 revealH = revealTex.sample(samp, in.uv);
    half4 auxH = auxTex.sample(samp, in.uv);

    float4 accum = float4(accumH);
    float reveal = float(revealH.w);
    float alpha = clamp(1.0f - reveal, 0.0f, 1.0f);
    float denom = max(accum.w, 1e-5f);

    float3 color = accum.xyz / denom; // straight (not premultiplied)

    float4 aux = float4(auxH);
    float sumAlpha = max(accum.w, 1e-5f);
    float avgDepth = aux.x / sumAlpha;
    float avgRadius = aux.y / sumAlpha;

    if (rp.debugView == 1u) { // alpha
        float a = alpha;
        return float4(a, a, a, 1.0f);
    } else if (rp.debugView == 2u) { // depth
        float zn = max(rp.debugNearZ, 1e-6f);
        float zf = max(rp.debugFarZ, zn + 1e-3f);
        float t = clamp((avgDepth - zn) / (zf - zn), 0.0f, 1.0f);
        float g = 1.0f - t;
        return float4(g, g, g, 1.0f);
    } else if (rp.debugView == 3u) { // disparity
        float disp = 1.0f / max(avgDepth, 1e-6f);
        float dn = 1.0f / max(rp.debugFarZ, 1e-3f);
        float df = 1.0f / max(rp.debugNearZ, 1e-3f);
        float t = clamp((disp - dn) / max(df - dn, 1e-6f), 0.0f, 1.0f);
        return float4(t, t, t, 1.0f);
    } else if (rp.debugView == 4u) { // radius
        float t = clamp(avgRadius / max(rp.maxRadiusPx, 1.0f), 0.0f, 1.0f);
        float3 heat = float3(t, 1.0f - t, 0.25f);
        return float4(heat, 1.0f);
    }

    // Exposure and tonemapping.
    float exposure = exp2(rp.exposureEV);
    color *= exposure;

    if (rp.toneMap == 1u) { // reinhard
        color = color / (1.0f + color);
    } else if (rp.toneMap == 2u) { // aces (approx)
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
    }

    // Contrast + saturation (linear).
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    color = mix(float3(luma), color, clamp(rp.saturation, 0.0f, 4.0f));
    color = (color - 0.5f) * clamp(rp.contrast, 0.0f, 4.0f) + 0.5f;
    color = max(color, float3(0.0f));

    float3 premult = color * alpha;
    return float4(premult, alpha);
}

struct DepthBinParams {
    float4x4 viewMatrix;
    float nearClipZ;
    float opacityThreshold;
    float zNear;
    float zFar;
    uint binCount;
    uint count;
};

kernel void depthBinCount(
    constant DepthBinParams &p [[buffer(0)]],
    device const packed_float3 *means [[buffer(1)]],
    device const float *opacities [[buffer(2)]],
    device atomic_uint *binCounts [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) { return; }
    float opa = opacities[gid];
    if (!isfinite(opa) || opa <= p.opacityThreshold) { return; }
    float3 pos = float3(means[gid]);
    if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z)) { return; }
    float z = (p.viewMatrix * float4(pos, 1.0f)).z;
    if (!isfinite(z) || z <= p.nearClipZ) { return; }

    float zn = max(p.zNear, p.nearClipZ);
    float zf = max(p.zFar, zn + 1e-3f);
    float t = clamp((clamp(z, zn, zf) - zn) / (zf - zn), 0.0f, 1.0f);
    uint bin = min(p.binCount - 1, (uint)floor(t * (float)p.binCount));
    bin = (p.binCount - 1) - bin; // far -> near ordering
    atomic_fetch_add_explicit(&binCounts[bin], 1u, memory_order_relaxed);
}

kernel void depthBinScatter(
    constant DepthBinParams &p [[buffer(0)]],
    device const packed_float3 *means [[buffer(1)]],
    device const float *opacities [[buffer(2)]],
    device atomic_uint *binCursors [[buffer(3)]],
    device uint *sortedIndices [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) { return; }
    float opa = opacities[gid];
    if (!isfinite(opa) || opa <= p.opacityThreshold) { return; }
    float3 pos = float3(means[gid]);
    if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z)) { return; }
    float z = (p.viewMatrix * float4(pos, 1.0f)).z;
    if (!isfinite(z) || z <= p.nearClipZ) { return; }

    float zn = max(p.zNear, p.nearClipZ);
    float zf = max(p.zFar, zn + 1e-3f);
    float t = clamp((clamp(z, zn, zf) - zn) / (zf - zn), 0.0f, 1.0f);
    uint bin = min(p.binCount - 1, (uint)floor(t * (float)p.binCount));
    bin = (p.binCount - 1) - bin;

    uint idx = atomic_fetch_add_explicit(&binCursors[bin], 1u, memory_order_relaxed);
    sortedIndices[idx] = gid;
}

vertex SplatVertexOut splatVertexSorted(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant CameraParams &cam [[buffer(0)]],
    device const packed_float3 *means [[buffer(1)]],
    device const packed_float4 *quats [[buffer(2)]],
    device const packed_float3 *scales [[buffer(3)]],
    device const packed_float3 *colors [[buffer(4)]],
    device const float *opacities [[buffer(5)]],
    device const uint *sortedIndices [[buffer(6)]],
    constant RenderParams &rp [[buffer(7)]]
) {
    uint gid = sortedIndices[iid];
    return splatVertexImpl(vid, gid, cam, means, quats, scales, colors, opacities, rp);
}

struct AlphaOut {
    float4 color [[color(0)]];
    float4 aux [[color(1)]];
};

fragment AlphaOut splatFragmentAlpha(SplatVertexOut in [[stage_in]]) {
    float2 fragPx = in.position.xy;
    float2 d = fragPx - in.centerPx;

    float a = in.invCov.x;
    float b = in.invCov.y;
    float c = in.invCov.z;
    float quad = a * d.x * d.x + 2.0f * b * d.x * d.y + c * d.y * d.y;
    float w = exp(-0.5f * quad);
    float alpha = clamp(in.opacity * w, 0.0f, 1.0f);

    AlphaOut out;
    out.color = float4(in.color * alpha, alpha);
    out.aux = float4(in.depth * alpha, in.radiusPx * alpha, 0.0f, alpha);
    return out;
}

fragment float4 compositeFromRGBA(
    CompositeVertexOut in [[stage_in]],
    texture2d<half> colorTex [[texture(0)]],
    texture2d<half> auxTex [[texture(1)]],
    sampler samp [[sampler(0)]],
    constant RenderParams &rp [[buffer(0)]]
) {
    half4 colH = colorTex.sample(samp, in.uv);
    half4 auxH = auxTex.sample(samp, in.uv);
    float4 col = float4(colH);
    float alpha = clamp(col.w, 0.0f, 1.0f);

    float sumAlpha = max(float(auxH.w), 1e-5f);
    float avgDepth = float(auxH.x) / sumAlpha;
    float avgRadius = float(auxH.y) / sumAlpha;

    if (rp.debugView == 1u) {
        float a = alpha;
        return float4(a, a, a, 1.0f);
    } else if (rp.debugView == 2u) {
        float zn = max(rp.debugNearZ, 1e-6f);
        float zf = max(rp.debugFarZ, zn + 1e-3f);
        float t = clamp((avgDepth - zn) / (zf - zn), 0.0f, 1.0f);
        float g = 1.0f - t;
        return float4(g, g, g, 1.0f);
    } else if (rp.debugView == 3u) {
        float disp = 1.0f / max(avgDepth, 1e-6f);
        float dn = 1.0f / max(rp.debugFarZ, 1e-3f);
        float df = 1.0f / max(rp.debugNearZ, 1e-3f);
        float t = clamp((disp - dn) / max(df - dn, 1e-6f), 0.0f, 1.0f);
        return float4(t, t, t, 1.0f);
    } else if (rp.debugView == 4u) {
        float t = clamp(avgRadius / max(rp.maxRadiusPx, 1.0f), 0.0f, 1.0f);
        float3 heat = float3(t, 1.0f - t, 0.25f);
        return float4(heat, 1.0f);
    }

    float3 color = (alpha > 1e-6f) ? (col.xyz / alpha) : float3(0.0f);
    color *= exp2(rp.exposureEV);

    if (rp.toneMap == 1u) {
        color = color / (1.0f + color);
    } else if (rp.toneMap == 2u) {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
    }

    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    color = mix(float3(luma), color, clamp(rp.saturation, 0.0f, 4.0f));
    color = (color - 0.5f) * clamp(rp.contrast, 0.0f, 4.0f) + 0.5f;
    color = max(color, float3(0.0f));

    float3 premult = color * alpha;
    return float4(premult, alpha);
}

fragment float4 downsampleFragment(
    CompositeVertexOut in [[stage_in]],
    texture2d<half> srcTex [[texture(0)]],
    sampler samp [[sampler(0)]]
) {
    half4 c = srcTex.sample(samp, in.uv);
    return float4(c);
}
