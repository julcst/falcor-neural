#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SlangMLP/SlangMLPConfig.slang"

import RTXNS.CooperativeVectorDerivatives;
import RTXNS.CooperativeVectorFunctions;
import RTXNS.LinearOps;
import RTXNS.MLP;
import RTXNS.Activation;
import RTXNS.PCG32;
import RTXNS.Utils;
import RTXNS.Loss;
import RTXNS.Optimizers;

static const float kReluLeak = 0.01;
static const float kOptimizerLossScale = 1.0;

static const float kAdamBeta1 = 0.9;
static const float kAdamBeta2 = 0.999;
static const float kAdamEpsilon = 1e-8;

Texture2D<float4> gInput;
RWTexture2D<float4> gOutput;
ByteAddressBuffer gParams;
RWByteAddressBuffer gParamsRW;
RWByteAddressBuffer gParamGrads;
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gTrainSteps;
    uint gBatchSize;
    float gLearningRate;
    uint gReset;
    uint gCurrentStep;
}

struct ByteAddressAccessor : optimizers::IBufferAccessor
{
    RWByteAddressBuffer mBuffer;

    __init(RWByteAddressBuffer buffer)
    {
        mBuffer = buffer;
    }

    float get(uint index)
    {
        return mBuffer.Load<float>(index * sizeof(float));
    }

    void set(uint index, float value)
    {
        mBuffer.Store<float>(index * sizeof(float), value);
    }
};

float sampleNormalizedLossScale(uint sampleCount)
{
    return 1.0 / float(max(1u, sampleCount));
}

[numthreads(SlangMLPConfig::kTrainThreads, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    uint sampleIndex = tid.x;
    if (sampleIndex >= gBatchSize)
        return;

    let mlp = SlangMLPConfig::TrainingMLP(
        gParams, 
        gParamGrads,
        SlangMLPConfig::weightOffsets, 
        SlangMLPConfig::biasOffsets
    );

    let hiddenAct = rtxns::mlp::LeakyReLUAct<half, SlangMLPConfig::kHiddenSize>(kReluLeak);
    let finalAct = rtxns::mlp::NoneAct<half, SlangMLPConfig::kOutputSize>();

    uint64_t seed = (uint64_t(gFrameIndex + 1u) << 32) ^ uint64_t(gCurrentStep + 1u) ^ uint64_t(sampleIndex + 1u);
    PCG32 rng = PCG32(seed, seed ^ 0x9e3779b97f4a7c15ULL);

    uint2 pixel = uint2(
        rng.nextUInt() % max(1u, gFrameDim.x),
        rng.nextUInt() % max(1u, gFrameDim.y)
    );

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    float3 target = gInput.Load(int3(pixel, 0)).xyz;

    let inputParams = rtxns::EncodeFrequencyN<half, 2, 8>(rtxns::CoopVecFromVector<half>(uv));
    let outputPre = mlp.forward(inputParams, hiddenAct, finalAct);

    let targetVec = rtxns::CoopVecFromVector<half>(target);

    let lossScale = CoopVec<half, SlangMLPConfig::kOutputSize>(half(sampleNormalizedLossScale(gBatchSize)));
    let outputGrad = rtxns::mlp::L2Relative<half, SlangMLPConfig::kOutputSize>::deriv(targetVec, outputPre, lossScale);

    mlp.backward(inputParams, hiddenAct, finalAct, outputGrad);
}

half loadHalf(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.Load<half>(elementIndex * sizeof(half));
}

void storeHalf(RWByteAddressBuffer buffer, uint elementIndex, half value)
{
    buffer.Store<half>(elementIndex * sizeof(half), value);
}

void resetParam(uint index, inout PCG32 rng)
{
    const float scale = sqrt(2.0 / float(SlangMLPConfig::kHiddenSize));
    float p = rng.nextFloat() * 2.0f - 1.0f;
    storeHalf(gParamsRW, index, p * scale);
    storeHalf(gParamGrads, index, 0.0f);
    gMoments1.Store<float>(index * sizeof(float), 0.0);
    gMoments2.Store<float>(index * sizeof(float), 0.0);
}

[numthreads(SlangMLPConfig::kOptimizeThreads, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= SlangMLPConfig::kParamElementCount)
        return;

    if (gReset != 0)
    {
        PCG32 rng = PCG32(i, i ^ 0x9e3779b97f4a7c15ULL);
        resetParam(i, rng);
        return;
    }

    float gradient = loadHalf(gParamGrads, i);
    storeHalf(gParamGrads, i, 0.0f);

    let adam = optimizers::AdamAccessor<ByteAddressAccessor>(
        ByteAddressAccessor(gMoments1),
        ByteAddressAccessor(gMoments2),
        gLearningRate,
        kOptimizerLossScale,
        kAdamBeta1,
        kAdamBeta2,
        kAdamEpsilon
    );

    float p = loadHalf(gParamsRW, i);
    p = adam.step(p, i, gradient, float(gCurrentStep));
    storeHalf(gParamsRW, i, p);
}

[numthreads(8, 8, 1)]
void inferMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= gFrameDim))
        return;

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    let mlp = SlangMLPConfig::InferenceMLP(
        gParams, 
        SlangMLPConfig::weightOffsets, 
        SlangMLPConfig::biasOffsets
    );

    let hiddenAct = rtxns::mlp::LeakyReLUAct<half, SlangMLPConfig::kHiddenSize>(kReluLeak);
    let finalAct = rtxns::mlp::NoneAct<half, SlangMLPConfig::kOutputSize>();

    let inputParams = rtxns::EncodeFrequencyN<half, 2, 8>(rtxns::CoopVecFromVector<half>(uv));
    let outputPre = mlp.forward(inputParams, hiddenAct, finalAct);
    let pred = rtxns::VectorFromCoopVec(outputPre);
    gOutput[pixel] = float4(saturate(pred), 1.0);
}