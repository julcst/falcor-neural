#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Structs;

StructuredBuffer<DoubleAccumulator> gAccumulator;
StructuredBuffer<Query> gPhotonQueries;
RWStructuredBuffer<QueryState> gQueryStates;
RWStructuredBuffer<float3> gOutputBuffer;

cbuffer CB
{
    uint gQueryCount;
    uint gGlobalPhotonCount;
    float gGlobalAlpha;
    float gCausticAlpha;
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gQueryCount) return;

    let query = gPhotonQueries[idx];

    let acc = gAccumulator[idx];
    var q = gQueryStates[idx];
    q.applyRadiusReduction(acc, gGlobalAlpha, gCausticAlpha);
    let radiance = q.getRadiance(gGlobalPhotonCount); // NOTE: outgoing radiance at query without emission
    gQueryStates[idx] = q;

    gOutputBuffer[idx] = radiance;
}
