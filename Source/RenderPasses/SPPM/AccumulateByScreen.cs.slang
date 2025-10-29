#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SPPM/SPPMDebug.slangh"

import PhotonQuery;
import PhotonHit;

StructuredBuffer<PhotonHit>    gPhotonHits;        // [photonCount]
StructuredBuffer<PhotonQuery>  gPhotonQueries;     // [queryCount]
RWStructuredBuffer<uint4>      gQueryAccumulation; // fixed-point accumulation per pixel
RWStructuredBuffer<SPPMCounters> gCounters;        // single element counters struct

cbuffer CB
{
    uint   gPhotonCount;
    float  gAccumScale;
    uint2  gFrameDim;
    float4x4 gViewProj;       // world->clip
    float2   gFocalPx;        // (fx_px, fy_px) = 0.5*dim*P00/11 from PROJECTION
    float    gMaxQueryRadius;      // max world-space query radius to cover
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gPhotonCount) return;

    InterlockedAdd(gCounters[0].PhotonsConsidered, 1);

    PhotonHit ph = gPhotonHits[idx];

    // Project photon to screen
    float4 clip = mul(gViewProj, float4(ph.posW, 1.0));
    if (clip.w <= 0.0) return;
    float2 ndc = clip.xy / clip.w;        // [-1,1]
    ndc.y = -ndc.y;
    float2 uv  = 0.5 * (ndc + 1.0);       // [0,1]
    if (any(uv < 0.0) || any(uv > 1.0)) return;

    // Compute conservative pixel search radius from max query radius at this depth.
    // r_px = r_world * max(fx_px, fy_px) / z
    float z = max(1e-6, clip.w);
    float rpx = gMaxQueryRadius * max(gFocalPx.x, gFocalPx.y) / z;
    const int r = int(ceil(rpx));

    int2 center = int2(uv * float2(gFrameDim));

    // Screen-space neighbor search around projected photon
    for (int dy = -r; dy <= r; ++dy)
    {
        int y = center.y + dy;
        if (y < 0 || y >= int(gFrameDim.y)) continue;
        for (int dx = -r; dx <= r; ++dx)
        {
            int x = center.x + dx;
            if (x < 0 || x >= int(gFrameDim.x)) continue;

            uint qIdx = uint(x + y * int(gFrameDim.x));
            PhotonQuery q = gPhotonQueries[qIdx];
            if (q.valid == 0) continue;

            float3 d = ph.posW - q.position;
            float dist2 = dot(d, d);
            float r2 = q.radius * q.radius;
            InterlockedAdd(gCounters[0].Candidates, 1);
            if (dist2 <= r2)
            {
                float3 wi = normalize(-d);
                float NoL = max(0.0, dot(q.normal, wi));
                float3 f = q.diffuseReflectance * (M_1_PI) * NoL;
                float3 c = ph.flux;

                uint pix = q.pixelIndex;
                uint4 inc = uint4(
                    (uint)max(0.0, c.x * gAccumScale + 0.5),
                    (uint)max(0.0, c.y * gAccumScale + 0.5),
                    (uint)max(0.0, c.z * gAccumScale + 0.5),
                    1u);

                InterlockedAdd(gQueryAccumulation[pix].x, inc.x);
                InterlockedAdd(gQueryAccumulation[pix].y, inc.y);
                InterlockedAdd(gQueryAccumulation[pix].z, inc.z);
                InterlockedAdd(gQueryAccumulation[pix].w, 1u);

                InterlockedAdd(gCounters[0].Accumulations, 1);
            }
        }
    }
}
