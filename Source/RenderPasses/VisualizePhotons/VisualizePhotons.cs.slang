#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TracePhotons.Structs;

StructuredBuffer<PhotonHit> gPhotonHits;
StructuredBuffer<PhotonCounters> gCounters;
RWTexture2D<float4>         gOutput;

cbuffer CB
{
    uint2  gFrameDim;
    uint   gPointSizePx;
    float4x4 gViewProj;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gCounters[0].PhotonStores) return;

    PhotonHit ph = gPhotonHits[idx];

    float4 clip = mul(gViewProj, float4(ph.posW, 1.0));
    if (clip.w <= 0.0) return;
    float2 ndc = clip.xy / clip.w;      // [-1,1]
    ndc.y = -ndc.y;
    float2 uv  = 0.5 * (ndc + 1.0);     // [0,1]
    if (any(uv < 0.0) || any(uv > 1.0)) return;

    int2 p = int2(uv * float2(gFrameDim));
    int half = gPointSizePx / 2;
    for (int dy = -half; dy <= half; ++dy)
    {
        int y = p.y + dy;
        if (y < 0 || y >= int(gFrameDim.y)) continue;
        for (int dx = -half; dx <= half; ++dx)
        {
            int x = p.x + dx;
            if (x < 0 || x >= int(gFrameDim.x)) continue;
            gOutput[int2(x,y)] = float4(ph.flux, 1.0);
        }
    }
}
