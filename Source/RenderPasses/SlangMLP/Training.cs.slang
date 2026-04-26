#include "Utils/Math/MathConstants.slangh"

import RTXNS.MLP;
import RTXNS.Activation;
import RTXNS.PCG32;
import RTXNS.Utils;
import RTXNS.Loss;
import RTXNS.LinearOps;
import RTXNS.CooperativeVectorAutoDiff;
import RTXNS.CooperativeVectorFunctions;
import RTXNS.CooperativeVectorDerivatives;
import Config;

using MLPConfig;

Texture2D<float4> gInput;
ByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;

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

    uint64_t seed = (uint64_t(gFrameIndex + 1u) << 32) ^ uint64_t(gCurrentStep + 1u) ^ uint64_t(sampleIndex + 1u);
    PCG32 rng = PCG32(seed, seed ^ 0x9e3779b97f4a7c15ULL);

    uint2 pixel = uint2(
        rng.nextUInt() % max(1u, gFrameDim.x),
        rng.nextUInt() % max(1u, gFrameDim.y)
    );

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    let target = gInput.Load(int3(pixel, 0)).xyz;
    let targetVec = rtxns::CoopVecFromVector<half>(target);

    let pred = forward(gParams, uv);

    let lossScale = CoopVec<half, kOutputSize>(half(sampleNormalizedLossScale(gBatchSize)));
    let outputGrad = rtxns::mlp::L2Relative<half, kOutputSize>::deriv(targetVec, pred, lossScale);

    var weights = rtxns::mlp::MatrixBiasBuffer(gParams);
    var weightsGrads = rtxns::mlp::MatrixBiasBufferDifferential(gParamGrads);
    bwd_diff(forward)(DifferentialPtrPair<rtxns::mlp::MatrixBiasBuffer>(weights, weightsGrads), uv, outputGrad);
}