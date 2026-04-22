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

float randomSigned01(inout PCG32 rng)
{
    return rng.nextFloat() * 2.0 - 1.0;
}

uint floatOffset(uint index)
{
    return index * 4u;
}

float loadFloatRaw(RWByteAddressBuffer buffer, uint index)
{
    return asfloat(buffer.Load(floatOffset(index)));
}

void storeFloatRaw(RWByteAddressBuffer buffer, uint index, float value)
{
    buffer.Store(floatOffset(index), asuint(value));
}

float loadHalf(ByteAddressBuffer buffer, uint elementIndex)
{
    uint wordAddr = (elementIndex >> 1u) * 4u;
    uint packed = buffer.Load(wordAddr);
    uint h = ((elementIndex & 1u) == 0u) ? (packed & 0xffffu) : (packed >> 16u);
    return f16tof32(h);
}

float loadHalf(RWByteAddressBuffer buffer, uint elementIndex)
{
    uint wordAddr = (elementIndex >> 1u) * 4u;
    uint packed = buffer.Load(wordAddr);
    uint h = ((elementIndex & 1u) == 0u) ? (packed & 0xffffu) : (packed >> 16u);
    return f16tof32(h);
}

void storeHalf(RWByteAddressBuffer buffer, uint elementIndex, float value)
{
    uint wordAddr = (elementIndex >> 1u) * 4u;
    uint packed = buffer.Load(wordAddr);
    uint h = f32tof16(value) & 0xffffu;

    if ((elementIndex & 1u) == 0u)
        packed = (packed & 0xffff0000u) | h;
    else
        packed = (packed & 0x0000ffffu) | (h << 16u);

    buffer.Store(wordAddr, packed);
}

struct ByteAddressMomentsAccessor : optimizers::IBufferAccessor
{
    RWByteAddressBuffer mBuffer;

    __init(RWByteAddressBuffer buffer)
    {
        mBuffer = buffer;
    }

    float get(uint index)
    {
        return loadFloatRaw(mBuffer, index);
    }

    void set(uint index, float value)
    {
        storeFloatRaw(mBuffer, index, value);
    }
};

float sampleNormalizedLossScale(uint sampleCount)
{
    return 1.0 / float(max(1u, sampleCount));
}

void clearGradientBuffer()
{
    for (uint i = 0; i < SlangMLPConfig::kParamElementCount; ++i)
        storeHalf(gParamGrads, i, 0.0);
}

void clearMoments()
{
    for (uint i = 0; i < SlangMLPConfig::kParamElementCount; ++i)
    {
        storeFloatRaw(gMoments1, i, 0.0);
        storeFloatRaw(gMoments2, i, 0.0);
    }
}

void initializeParams()
{
    uint idx = 0;
    PCG32 rng = PCG32(1u, 11u);

    for (uint layer = 0u; layer < SlangMLPConfig::kTransitions; ++layer)
    {
        const float scale = sqrt(2.0 / float(SlangMLPConfig::layerInputSize(layer)));

        const uint weightCount = SlangMLPConfig::layerWeightCount(layer);
        for (uint i = 0u; i < weightCount; ++i)
            storeHalf(gParamsRW, idx++, scale * randomSigned01(rng));

        const uint biasCount = SlangMLPConfig::layerBiasCount(layer);
        for (uint i = 0u; i < biasCount; ++i)
            storeHalf(gParamsRW, idx++, 0.0);
    }

    clearGradientBuffer();
    clearMoments();
}

rtxns::mlp::InferenceMLP<
    half,
    SlangMLPConfig::kHiddenLayers,
    SlangMLPConfig::kInputSize,
    SlangMLPConfig::kHiddenSize,
    SlangMLPConfig::kOutputSize,
    CoopVecMatrixLayout::TrainingOptimal,
    CoopVecComponentType::Float16
> createInferenceMlp()
{
    uint weightOffsets[SlangMLPConfig::kTransitions];
    uint biasOffsets[SlangMLPConfig::kTransitions];
    SlangMLPConfig::computeLayerOffsets(weightOffsets, biasOffsets);
    return rtxns::mlp::InferenceMLP<
        half,
        SlangMLPConfig::kHiddenLayers,
        SlangMLPConfig::kInputSize,
        SlangMLPConfig::kHiddenSize,
        SlangMLPConfig::kOutputSize,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16
    >(gParams, weightOffsets, biasOffsets);
}

rtxns::mlp::TrainingMLP<
    half,
    SlangMLPConfig::kHiddenLayers,
    SlangMLPConfig::kInputSize,
    SlangMLPConfig::kHiddenSize,
    SlangMLPConfig::kOutputSize,
    CoopVecMatrixLayout::TrainingOptimal,
    CoopVecComponentType::Float16
> createTrainingMlp()
{
    uint weightOffsets[SlangMLPConfig::kTransitions];
    uint biasOffsets[SlangMLPConfig::kTransitions];
    SlangMLPConfig::computeLayerOffsets(weightOffsets, biasOffsets);
    return rtxns::mlp::TrainingMLP<
        half,
        SlangMLPConfig::kHiddenLayers,
        SlangMLPConfig::kInputSize,
        SlangMLPConfig::kHiddenSize,
        SlangMLPConfig::kOutputSize,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16
    >(gParams, gParamGrads, weightOffsets, biasOffsets);
}

float3 modelInfer(float2 uv)
{
    let mlp = createInferenceMlp();

    let hiddenAct = rtxns::mlp::LeakyReLUAct<half, SlangMLPConfig::kHiddenSize>(half(kReluLeak));
    let finalAct = rtxns::mlp::NoneAct<half, SlangMLPConfig::kOutputSize>();

    let inputParams = CoopVec<half, SlangMLPConfig::kInputSize>(half(uv.x), half(uv.y));
    let outputPre = mlp.forward(inputParams, hiddenAct, finalAct);
    return float3(outputPre[0], outputPre[1], outputPre[2]);
}

[numthreads(SlangMLPConfig::kTrainThreads, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    uint sampleIndex = tid.x;
    if (sampleIndex >= gBatchSize)
        return;

    if (gReset != 0)
    {
        if (sampleIndex == 0u)
            initializeParams();
        return;
    }

    let mlp = createTrainingMlp();

    let hiddenAct = rtxns::mlp::LeakyReLUAct<half, SlangMLPConfig::kHiddenSize>(half(kReluLeak));
    let finalAct = rtxns::mlp::NoneAct<half, SlangMLPConfig::kOutputSize>();

    uint64_t seed = (uint64_t(gFrameIndex + 1u) << 32) ^ uint64_t(gCurrentStep + 1u) ^ uint64_t(sampleIndex + 1u);
    PCG32 rng = PCG32(seed, seed ^ 0x9e3779b97f4a7c15ULL);

    uint2 pixel = uint2(
        rng.nextUInt() % max(1u, gFrameDim.x),
        rng.nextUInt() % max(1u, gFrameDim.y)
    );

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    float3 target = gInput.Load(int3(pixel, 0)).xyz;

    let inputParams = CoopVec<half, SlangMLPConfig::kInputSize>(half(uv.x), half(uv.y));
    let outputPre = mlp.forward(inputParams, hiddenAct, finalAct);

    let targetVec = CoopVec<half, SlangMLPConfig::kOutputSize>(half(target.x), half(target.y), half(target.z));

    let lossScale = CoopVec<half, SlangMLPConfig::kOutputSize>(half(sampleNormalizedLossScale(gBatchSize)));
    let outputGrad = rtxns::mlp::L2Relative<half, SlangMLPConfig::kOutputSize>::deriv(targetVec, outputPre, lossScale);

    mlp.backward(inputParams, hiddenAct, finalAct, outputGrad);
}

[numthreads(SlangMLPConfig::kOptimizeThreads, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= SlangMLPConfig::kParamElementCount)
        return;

    float gradient = loadHalf(gParamGrads, i);
    storeHalf(gParamGrads, i, 0.0);

    let moments1Accessor = ByteAddressMomentsAccessor(gMoments1);
    let moments2Accessor = ByteAddressMomentsAccessor(gMoments2);
    let adam = optimizers::AdamAccessor<ByteAddressMomentsAccessor>(
        moments1Accessor,
        moments2Accessor,
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
    float3 pred = modelInfer(uv);
    gOutput[pixel] = float4(saturate(pred), 1.0);
}