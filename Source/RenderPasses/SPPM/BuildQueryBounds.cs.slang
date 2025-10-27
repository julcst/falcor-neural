import Utils.Math.MathHelpers;
import PhotonQuery;

struct RtAABB { float3 min; float3 max; };

RWStructuredBuffer<PhotonQuery> gPhotonQueries;
RWStructuredBuffer<RtAABB> gQueryAABBs;

cbuffer CB
{
    uint gQueryCount;
}

[numthreads(64,1,1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gQueryCount) return;

    PhotonQuery q = gPhotonQueries[i];
    RtAABB aabb;
    if (q.valid != 0)
    {
        float3 r = float3(q.radius);
        aabb.min = q.position - r;
        aabb.max = q.position + r;
    }
    else
    {
        // Invalidate with large/small sentinels
        aabb.min = float3(1e30f, 1e30f, 1e30f);
        aabb.max = float3(-1e30f, -1e30f, -1e30f);
    }
    gQueryAABBs[i] = aabb;
}
