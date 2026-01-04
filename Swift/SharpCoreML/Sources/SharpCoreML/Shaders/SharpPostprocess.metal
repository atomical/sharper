#include <metal_stdlib>
using namespace metal;

struct UnprojectParams {
    float sx;
    float sy;
    uint count;
};

static inline float4 normalize_quat_wxyz(float4 q) {
    float n = sqrt(max(dot(q, q), 1e-20f));
    return q / n;
}

static inline float3x3 quat_to_rotmat_wxyz(float4 q_wxyz) {
    float w = q_wxyz.x;
    float x = q_wxyz.y;
    float y = q_wxyz.z;
    float z = q_wxyz.w;

    float ww = w * w;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    float wx = w * x;
    float wy = w * y;
    float wz = w * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;

    float3x3 r;
    r[0][0] = ww + xx - yy - zz;
    r[0][1] = 2.0f * (xy + wz);
    r[0][2] = 2.0f * (xz - wy);

    r[1][0] = 2.0f * (xy - wz);
    r[1][1] = ww - xx + yy - zz;
    r[1][2] = 2.0f * (yz + wx);

    r[2][0] = 2.0f * (xz + wy);
    r[2][1] = 2.0f * (yz - wx);
    r[2][2] = ww - xx - yy + zz;
    return r;
}

static inline float det3(float3x3 m) {
    return dot(m[0], cross(m[1], m[2]));
}

static inline void jacobi_rotate(thread float3x3 &a, thread float3x3 &v, uint p, uint q) {
    float apq = a[p][q];
    if (fabs(apq) < 1e-12f) {
        return;
    }
    float app = a[p][p];
    float aqq = a[q][q];

    float phi = 0.5f * atan2(2.0f * apq, (aqq - app));
    float c = fast::cos(phi);
    float s = fast::sin(phi);

    // Update diagonal entries.
    float app_new = c * c * app - 2.0f * s * c * apq + s * s * aqq;
    float aqq_new = s * s * app + 2.0f * s * c * apq + c * c * aqq;

    // Update off-diagonals for k != p,q.
    for (uint k = 0; k < 3; k++) {
        if (k == p || k == q) {
            continue;
        }
        float aik = a[p][k];
        float aqk = a[q][k];
        float aik_new = c * aik - s * aqk;
        float aqk_new = s * aik + c * aqk;
        a[p][k] = aik_new;
        a[k][p] = aik_new;
        a[q][k] = aqk_new;
        a[k][q] = aqk_new;
    }

    a[p][p] = app_new;
    a[q][q] = aqq_new;
    a[p][q] = 0.0f;
    a[q][p] = 0.0f;

    // Accumulate eigenvectors (columns of v).
    for (uint k = 0; k < 3; k++) {
        float vkp = v[k][p];
        float vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
    }
}

static inline void jacobi_eigen(thread float3x3 &a, thread float3x3 &v) {
    v = float3x3(1.0f);
    // Fixed iterations for determinism.
    for (uint iter = 0; iter < 8; iter++) {
        float a01 = fabs(a[0][1]);
        float a02 = fabs(a[0][2]);
        float a12 = fabs(a[1][2]);

        uint p = 0;
        uint q = 1;
        float maxv = a01;
        if (a02 > maxv) {
            maxv = a02;
            p = 0;
            q = 2;
        }
        if (a12 > maxv) {
            maxv = a12;
            p = 1;
            q = 2;
        }

        if (maxv < 1e-10f) {
            break;
        }
        jacobi_rotate(a, v, p, q);
    }
}

static inline void swap_cols(thread float3x3 &evecs, uint i, uint j) {
    float3 tmp = float3(evecs[0][i], evecs[1][i], evecs[2][i]);
    evecs[0][i] = evecs[0][j];
    evecs[1][i] = evecs[1][j];
    evecs[2][i] = evecs[2][j];
    evecs[0][j] = tmp.x;
    evecs[1][j] = tmp.y;
    evecs[2][j] = tmp.z;
}

static inline void sort_eigenpairs_desc(thread float3 &evals, thread float3x3 &evecs) {
    // Sort (evals, columns of evecs) by descending evals.
    if (evals.x < evals.y) {
        float t = evals.x;
        evals.x = evals.y;
        evals.y = t;
        swap_cols(evecs, 0, 1);
    }
    if (evals.y < evals.z) {
        float t = evals.y;
        evals.y = evals.z;
        evals.z = t;
        swap_cols(evecs, 1, 2);
    }
    if (evals.x < evals.y) {
        float t = evals.x;
        evals.x = evals.y;
        evals.y = t;
        swap_cols(evecs, 0, 1);
    }
}

static inline float4 quat_from_rotmat_wxyz(float3x3 m) {
    float trace = m[0][0] + m[1][1] + m[2][2];
    float w, x, y, z;
    if (trace > 0.0f) {
        float s = sqrt(trace + 1.0f) * 2.0f;
        w = 0.25f * s;
        x = (m[2][1] - m[1][2]) / s;
        y = (m[0][2] - m[2][0]) / s;
        z = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        float s = sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
        w = (m[2][1] - m[1][2]) / s;
        x = 0.25f * s;
        y = (m[0][1] + m[1][0]) / s;
        z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        float s = sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
        w = (m[0][2] - m[2][0]) / s;
        x = (m[0][1] + m[1][0]) / s;
        y = 0.25f * s;
        z = (m[1][2] + m[2][1]) / s;
    } else {
        float s = sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
        w = (m[1][0] - m[0][1]) / s;
        x = (m[0][2] + m[2][0]) / s;
        y = (m[1][2] + m[2][1]) / s;
        z = 0.25f * s;
    }
    return float4(w, x, y, z);
}

kernel void unprojectGaussians(
    device const packed_float3 *meanIn [[buffer(0)]],
    device const packed_float4 *quatIn [[buffer(1)]],
    device const packed_float3 *scaleIn [[buffer(2)]],
    device packed_float3 *meanOut [[buffer(3)]],
    device packed_float4 *quatOut [[buffer(4)]],
    device packed_float3 *scaleOut [[buffer(5)]],
    constant UnprojectParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }

    float3 m = float3(meanIn[gid]);
    float3 mOut = float3(m.x * params.sx, m.y * params.sy, m.z);
    meanOut[gid] = packed_float3(mOut);

    float4 q = normalize_quat_wxyz(float4(quatIn[gid]));
    float3 s = float3(scaleIn[gid]);

    float3x3 r = quat_to_rotmat_wxyz(q);
    float3 s2 = s * s;
    float3x3 c = r * float3x3(
        float3(s2.x, 0.0f, 0.0f),
        float3(0.0f, s2.y, 0.0f),
        float3(0.0f, 0.0f, s2.z)
    ) * transpose(r);

    // Apply diagonal unprojection scale D = diag(sx, sy, 1).
    float dx = params.sx;
    float dy = params.sy;
    float3 d = float3(dx, dy, 1.0f);
    float3x3 c2;
    c2[0][0] = c[0][0] * d.x * d.x;
    c2[0][1] = c[0][1] * d.x * d.y;
    c2[0][2] = c[0][2] * d.x * d.z;
    c2[1][0] = c2[0][1];
    c2[1][1] = c[1][1] * d.y * d.y;
    c2[1][2] = c[1][2] * d.y * d.z;
    c2[2][0] = c2[0][2];
    c2[2][1] = c2[1][2];
    c2[2][2] = c[2][2] * d.z * d.z;

    // Jacobi eigen decomposition.
    thread float3x3 a = c2;
    thread float3x3 v;
    jacobi_eigen(a, v);

    float3 evals = float3(a[0][0], a[1][1], a[2][2]);
    sort_eigenpairs_desc(evals, v);

    // Ensure right-handed rotation matrix (match ml-sharp's reflection fix).
    if (det3(v) < 0.0f) {
        v[0][2] *= -1.0f;
        v[1][2] *= -1.0f;
        v[2][2] *= -1.0f;
    }

    float3 sv = sqrt(max(evals, float3(1e-20f)));
    scaleOut[gid] = packed_float3(sv);

    float4 qOut = quat_from_rotmat_wxyz(v);
    quatOut[gid] = packed_float4(qOut);
}
