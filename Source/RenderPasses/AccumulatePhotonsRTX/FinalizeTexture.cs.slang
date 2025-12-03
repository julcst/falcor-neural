#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Structs;

StructuredBuffer<DoubleAccumulator> gAccumulator;
StructuredBuffer<Query> gPhotonQueries;
RWStructuredBuffer<QueryState> gQueryStates;
RWTexture2D<float4> gOutputTexture;

cbuffer CB
{
    uint2 gFrameDim;
    uint gGlobalPhotonCount;
    bool gVisualizeHeatmap;
    float gGlobalAlpha;
    float gCausticAlpha;
}

[numthreads(16,16,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let px = tid.xy;
    if (any(px >= gFrameDim)) return;
    
    let idx = px.x + px.y * gFrameDim.x;

    let query = gPhotonQueries[idx];

    let acc = gAccumulator[idx];
    var q = gQueryStates[idx];
    q.applyRadiusReduction(acc, gGlobalAlpha, gCausticAlpha);
    let radiance = (q.getRadiance(gGlobalPhotonCount) + query.emission) * query.throughput; // NOTE: radiance at pixel
    gQueryStates[idx] = q;

    gOutputTexture[px] = gVisualizeHeatmap ?
        float4(acc.global.collectedPhotons, acc.caustic.collectedPhotons, 0.0, 1.0) :
        float4(radiance, 1.0);
}
