#include "Utils/Math/MathConstants.slangh"

import RTXNS.MLP;
import RTXNS.Activation;
import RTXNS.PCG32;
import RTXNS.Utils;
import RTXNS.Loss;
import RTXNS.CooperativeVectorAutoDiff;
import Config;
import Encodings;

using MLPConfig;

Texture2D<float4> gInput;
ByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
ByteAddressBuffer gEncodingParams;
RWByteAddressBuffer gEncodingParamGrads;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gBatchSize;
    uint gCurrentStep;
}

float sampleNormalizedLossScale(uint sampleCount)
{
    return 1.0 / float(max(1u, sampleCount));
}

[numthreads(64, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    uint sampleIndex = tid.x;
    if (sampleIndex >= gBatchSize)
        return;

    let mlp = TrainingMLP(
        gParams, 
        gParamGrads,
        weightOffsets, 
        biasOffsets
    );

    let hiddenAct = rtxns::mlp::LeakyReLUAct<half, kHiddenSize>(kReluLeak);
    let finalAct = rtxns::mlp::NoneAct<half, kOutputSize>();

    uint64_t seed = (uint64_t(gFrameIndex + 1u) << 32) ^ uint64_t(gCurrentStep + 1u) ^ uint64_t(sampleIndex + 1u);
    PCG32 rng = PCG32(seed, seed ^ 0x9e3779b97f4a7c15ULL);

    uint2 pixel = uint2(
        rng.nextUInt() % max(1u, gFrameDim.x),
        rng.nextUInt() % max(1u, gFrameDim.y)
    );

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    float3 target = gInput.Load(int3(pixel, 0)).xyz;

    let inputParams = encodeHashGrid(gEncodingParams, uv);
    let outputPre = mlp.forward(inputParams, hiddenAct, finalAct);

    let targetVec = rtxns::CoopVecFromVector<half>(target);

    let lossScale = CoopVec<half, kOutputSize>(half(sampleNormalizedLossScale(gBatchSize)));
    let outputGrad = rtxns::mlp::L2Relative<half, kOutputSize>::deriv(targetVec, outputPre, lossScale);

    var dInputParams = diffPair(inputParams);
    mlp.backward(dInputParams, hiddenAct, finalAct, outputGrad);
    accumulateHashGridGrads(gEncodingParamGrads, uv, dInputParams.d);
}