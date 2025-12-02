#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TracePhotons.Structs;
import Utils.Math.AABB;

StructuredBuffer<PhotonHit> gPhotonHits;
RWStructuredBuffer<AABB> gPhotonAABBs;

cbuffer CB
{
    uint gPhotonHitCount;
    float gRadius;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gPhotonHitCount) return;
    
    let ph = gPhotonHits[idx];
    
    // Create AABB around photon
    float3 p = ph.posW;
    float r = gRadius;
    
    gPhotonAABBs[idx] = AABB(p - r, p + r);
}
