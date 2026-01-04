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

struct VertexOut {
    float4 position [[position]];
    float2 centerPx;
    float sigmaPx;
    float3 color;
    float opacity;
};

vertex VertexOut gaussianVertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    constant CameraParams &cam [[buffer(0)]],
    device const packed_float3 *means [[buffer(1)]],
    device const packed_float3 *scales [[buffer(2)]],
    device const packed_float3 *colors [[buffer(3)]],
    device const float *opacities [[buffer(4)]]
) {
    float3 pos = float3(means[iid]);
    float3 scl = float3(scales[iid]);
    float3 col = float3(colors[iid]);
    float opa = opacities[iid];

    float4 pCam4 = cam.viewMatrix * float4(pos, 1.0);
    float3 pCam = pCam4.xyz;
    float z = max(pCam.z, 1e-4f);

    float u = cam.fx * (pCam.x / z) + cam.cx;
    float v = cam.fy * (pCam.y / z) + cam.cy;

    // Radius heuristic (3-sigma).
    float sMax = max(scl.x, max(scl.y, scl.z));
    float sigmaPx = max(1.0f, sMax * cam.fx / z);
    float radiusPx = clamp(3.0f * sigmaPx, 1.0f, 80.0f);

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

    VertexOut out;
    out.position = float4(ndc + ndcOffset, 0.0, 1.0);
    out.centerPx = float2(u, v);
    out.sigmaPx = sigmaPx;
    out.color = col;
    out.opacity = opa;
    return out;
}

fragment float4 gaussianFragment(VertexOut in [[stage_in]]) {
    float2 fragPx = in.position.xy;
    float2 d = fragPx - in.centerPx;
    float invSigma2 = 1.0f / max(in.sigmaPx * in.sigmaPx, 1e-4f);
    float w = exp(-0.5f * dot(d, d) * invSigma2);
    float a = clamp(in.opacity * w, 0.0f, 1.0f);
    return float4(in.color * a, a);
}
