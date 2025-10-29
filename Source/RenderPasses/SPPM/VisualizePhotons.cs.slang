#include "Utils/Math/MathConstants.slangh"

import PhotonHit;

StructuredBuffer<PhotonHit> gPhotonHits;
RWTexture2D<float4>        gOutput;

cbuffer CB
{
    uint   gPhotonCount;
    uint2  gFrameDim;
    int    gPointSizePx;   // 1..5
    float4x4 gViewProj;
    float3 gColor;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gPhotonCount) return;

    PhotonHit ph = gPhotonHits[idx];

    float4 clip = mul(gViewProj, float4(ph.posW, 1.0));
    if (clip.w <= 0.0) return;
    float2 ndc = clip.xy / clip.w;      // [-1,1]
    ndc.y = -ndc.y;
    float2 uv  = 0.5 * (ndc + 1.0);     // [0,1]
    if (any(uv < 0.0) || any(uv > 1.0)) return;

    int2 p = int2(uv * float2(gFrameDim));
    int r = max(1, gPointSizePx);
    int half = r / 2;
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
