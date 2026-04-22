#include "Utils/Math/MathConstants.slangh"

import RTXNS.MLP;
import RTXNS.Activation;
import RTXNS.Utils;
import Config;

using MLPConfig;

RWTexture2D<float4> gOutput;
ByteAddressBuffer gParams;

cbuffer CB
{
    uint2 gFrameDim;
}


[numthreads(8, 8, 1)]
void inferMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= gFrameDim))
        return;

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    let mlp = InferenceMLP(
        gParams, 
        weightOffsets, 
        biasOffsets
    );

    let hiddenAct = rtxns::mlp::LeakyReLUAct<half, kHiddenSize>(kReluLeak);
    let finalAct = rtxns::mlp::NoneAct<half, kOutputSize>();

    let inputParams = rtxns::EncodeFrequencyN<half, 2, 8>(rtxns::CoopVecFromVector<half>(uv));
    let outputPre = mlp.forward(inputParams, hiddenAct, finalAct);
    let pred = rtxns::VectorFromCoopVec(outputPre);
    gOutput[pixel] = float4(saturate(pred), 1.0);
}