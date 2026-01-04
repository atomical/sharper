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

struct SplatVertexOut {
    float4 position [[position]];
    float2 centerPx;
    float3 invCov; // symmetric 2x2: (a, b, c) => [[a,b],[b,c]]
    float3 color;
    float opacity;
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

vertex SplatVertexOut splatVertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant CameraParams &cam [[buffer(0)]],
    device const packed_float3 *means [[buffer(1)]],
    device const packed_float4 *quats [[buffer(2)]],
    device const packed_float3 *scales [[buffer(3)]],
    device const packed_float3 *colors [[buffer(4)]],
    device const float *opacities [[buffer(5)]]
) {
    float3 pos = float3(means[iid]);
    float4 q = float4(quats[iid]);
    float3 scl = float3(scales[iid]);
    float3 col = float3(colors[iid]);
    float opa = opacities[iid];

    // Normalize quaternion defensively.
    q = q * rsqrt(max(dot(q, q), 1e-12f));

    float4 pCam4 = cam.viewMatrix * float4(pos, 1.0);
    float3 pCam = pCam4.xyz;
    float z = max(pCam.z, 1e-4f);

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

    // Ensure a minimum footprint in pixel space for numerical stability.
    A += 1e-4f;
    C += 1e-4f;

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
    float radiusPx = clamp(3.0f * sigmaMaxPx, 1.0f, 160.0f);

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

    SplatVertexOut out;
    out.position = float4(ndc + ndcOffset, 0.0, 1.0);
    out.centerPx = float2(u, v);
    out.invCov = float3(invA, invB, invC);
    out.color = col;
    out.opacity = opa;
    return out;
}

struct OITOut {
    float4 accum [[color(0)]];
    float4 reveal [[color(1)]];
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
    sampler samp [[sampler(0)]]
) {
    half4 accumH = accumTex.sample(samp, in.uv);
    half4 revealH = revealTex.sample(samp, in.uv);

    float4 accum = float4(accumH);
    float reveal = float(revealH.w);
    float alpha = clamp(1.0f - reveal, 0.0f, 1.0f);
    float denom = max(accum.w, 1e-5f);

    float3 color = (accum.xyz / denom) * alpha;
    return float4(color, alpha);
}
