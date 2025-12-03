#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Utils.Math.AABB;
import Structs;

StructuredBuffer<Query> gQueries;
RWStructuredBuffer<QueryState> gQueryStates;
RWStructuredBuffer<AABB> gQueryAABBs;
RWStructuredBuffer<Sphere> gQuerySpheres;

cbuffer CB
{
    uint gQueryCount;
    uint gGlobalPhotonCount;
    bool gReset;
    float gInitialRadius;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gQueryCount) return;
    let q = gQueries[idx];

    bool shouldReset = gReset;
    // Check if query has changed
    shouldReset |= gQueryStates[idx].revision != q.revision;

    if (shouldReset) {
        gQueryStates[idx] = QueryState(gGlobalPhotonCount, gInitialRadius, q.revision);
    }
    
    let r = gQueryStates[idx].radius;
    
    if (q.isValid()) {
        gQueryAABBs[idx] = AABB(q.position - r, q.position + r);
        gQuerySpheres[idx] = Sphere(q.position, r * r);
    } else {
        AABB aabb;
        aabb.invalidate();
        gQueryAABBs[idx] = aabb;
    }
}

