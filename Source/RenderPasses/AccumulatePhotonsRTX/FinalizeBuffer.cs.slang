#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Structs;

ByteAddressBuffer gAccumulator;
StructuredBuffer<Query> gPhotonQueries;
RWStructuredBuffer<QueryState> gQueryStates;
RWStructuredBuffer<float3> gOutputBuffer;

cbuffer CB
{
    uint gQueryCount;
    uint gGlobalPhotonCount;
    float gAlpha;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gQueryCount) return;

    let query = gPhotonQueries[idx];

    let acc = Accumulator(gAccumulator.Load4Aligned(idx * sizeof(float4)));
    var q = gQueryStates[idx];
    q.applyRadiusReduction(acc, gAlpha);
    let radiance = q.getRadiance(gGlobalPhotonCount) + query.emission; // NOTE: radiance at query
    gQueryStates[idx] = q;

    gOutputBuffer[idx] = radiance;
}
