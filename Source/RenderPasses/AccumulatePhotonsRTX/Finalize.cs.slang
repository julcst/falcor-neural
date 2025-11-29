#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Structs;

ByteAddressBuffer gAccumulator;
StructuredBuffer<Query> gPhotonQueries;
RWStructuredBuffer<QueryState> gQueryStates;
RWTexture2D<float4> gOutputTexture;
RWStructuredBuffer<float3> gOutputBuffer;

cbuffer CB
{
    uint2 gFrameDim;
    uint gGlobalPhotonCount;
    bool gVisualizeHeatmap;
    float gAlpha;
}

float3 heatmap(float v)
{
    // Simple blue->green->red heatmap for v in [0,1]
    v = saturate(v);
    var c = float3(0);
    c.r = smoothstep(0.5, 1.0, v);
    c.g = (v < 0.5) ? smoothstep(0.0, 0.5, v) : smoothstep(1.0, 0.5, v);
    c.b = 1.0 - smoothstep(0.0, 0.5, v);
    return c;
}

float mapTo01(float x)
{
    // Map integer to [0,1] for visualization
    return log(x + 1.0) / log(1000.0 + 1.0); // Assuming max ~1000 for better contrast
}

[numthreads(16,16,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let px = tid.xy;
    if (any(px >= gFrameDim)) return;
    let idx = px.x + px.y * gFrameDim.x;

    let query = gPhotonQueries[idx];

    let acc = Accumulator(gAccumulator.Load4Aligned(idx * sizeof(float4)));
    var q = gQueryStates[idx];
    q.applyRadiusReduction(acc, gAlpha);
    let radiance = q.getRadiance(gGlobalPhotonCount) + query.emission;
    gQueryStates[idx] = q;

#if OUTPUT_TEXTURE
    gOutputTexture[px] = gVisualizeHeatmap ?
        float4(acc.collectedPhotons, heatmap(mapTo01(acc.collectedPhotons)).xy, 1.0) :
        float4(radiance, 1.0);
#endif

#if OUTPUT_BUFFER
    gOutputBuffer[idx] = radiance;
#endif
}

