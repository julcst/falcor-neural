#include "Utils/Math/MathConstants.slangh"

import RTXNS.MLP;
import RTXNS.Activation;
import RTXNS.Utils;
import NRCCompat;
import Config;
import Encodings;
import RenderPasses.TraceQueries.Query;

using MLPConfig;

StructuredBuffer<NRCInput> gInferenceInput;
StructuredBuffer<Query> gInferenceQueries;
ByteAddressBuffer gParams;
ByteAddressBuffer gEncodingParams;
RWTexture2D<float4> gOutput;

cbuffer CB
{
    uint gCount;
    uint2 gResolution;
};

[numthreads(256, 1, 1)]
void inferMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gCount) return;

    let input = gInferenceInput[idx];
    let predVec = forward(gEncodingParams, gParams, input);
    let pred = rtxns::VectorFromCoopVec(predVec);

    // Apply factorization inline when configured (matches previous OutputsToTexture behavior)
#if USE_FACTORISATION
    let outRadiance = pred * input.weight();
#else
    let outRadiance = pred;
#endif

    let query = gInferenceQueries[idx];
    let radiance = query.isValid() ? outRadiance : float3(0.0f);
    let color = float4(query.transportRadianceToCamera(radiance), 1.0f);

    uint2 pixel = uint2(idx % gResolution.x, idx / gResolution.x);
    gOutput[pixel] = color;
}