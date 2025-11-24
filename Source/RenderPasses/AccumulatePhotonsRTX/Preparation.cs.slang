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
    bool gReset;
    float gInitialRadius;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gQueryCount) return;

    if (gReset) {
        gQueryStates[idx] = QueryState(float3(0), 0.0f, gInitialRadius);
    }

    let q = gQueries[idx];
    let r = 0.005f;
    // let r = gQueryStates[idx].radius; // FIXME: Why does this not work


    // NOTE: For some weird reason this works
    AABB aabb;
    if (q.isValid()) {
        gQueryAABBs[idx] = AABB(q.position - r, q.position + r);
    } else {
        aabb.invalidate();
    }
    gQuerySpheres[idx] = Sphere(q.position, r * r);

    // But this eg does not
    // if (q.isValid()) {
    //     gQueryAABBs[idx] = AABB(q.position - r, q.position + r);
    //     gQuerySpheres[idx] = Sphere(q.position, r * r);
    // } else {
    //     AABB aabb;
    //     aabb.invalidate();
    //     gQueryAABBs[idx] = aabb;
    // }
}

