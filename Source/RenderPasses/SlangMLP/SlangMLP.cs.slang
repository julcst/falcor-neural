#include "Utils/Math/MathConstants.slangh"

import RTXNS.MLP;
import Activation;
import CooperativeVectorFunctions;

static const uint kInputSize = 2;
static const uint kHiddenSize = 32;
static const uint kOutputSize = 3;
static const uint kHiddenLayers = 2;
static const uint kTransitions = kHiddenLayers + 1;

static const uint kLayer0WeightCount = kHiddenSize * kInputSize;
static const uint kLayer0BiasCount = kHiddenSize;
static const uint kLayer1WeightCount = kHiddenSize * kHiddenSize;
static const uint kLayer1BiasCount = kHiddenSize;
static const uint kLayer2WeightCount = kOutputSize * kHiddenSize;
static const uint kLayer2BiasCount = kOutputSize;
static const uint kParamElementCount =
    kLayer0WeightCount + kLayer0BiasCount +
    kLayer1WeightCount + kLayer1BiasCount +
    kLayer2WeightCount + kLayer2BiasCount;

Texture2D<float4> gInput;
RWTexture2D<float4> gOutput;
ByteAddressBuffer gParams;
RWByteAddressBuffer gParamsRW;
RWByteAddressBuffer gParamGrads;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gTrainSteps;
    float gLearningRate;
    uint gReset;
}

uint align4(uint x)
{
    return (x + 3u) & ~3u;
}

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float randomSigned01(uint seed)
{
    return (float(hash(seed) & 0x00ffffffu) / float(0x00ffffffu)) * 2.0 - 1.0;
}

void computeLayerOffsets(out uint weightOffsets[kTransitions], out uint biasOffsets[kTransitions])
{
    uint byteOffset = 0;

    weightOffsets[0] = byteOffset;
    byteOffset = align4(byteOffset + 2u * kLayer0WeightCount);
    biasOffsets[0] = byteOffset;
    byteOffset = align4(byteOffset + 2u * kLayer0BiasCount);

    weightOffsets[1] = byteOffset;
    byteOffset = align4(byteOffset + 2u * kLayer1WeightCount);
    biasOffsets[1] = byteOffset;
    byteOffset = align4(byteOffset + 2u * kLayer1BiasCount);

    weightOffsets[2] = byteOffset;
    byteOffset = align4(byteOffset + 2u * kLayer2WeightCount);
    biasOffsets[2] = byteOffset;
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

void clearGradientBuffer()
{
    for (uint i = 0; i < kParamElementCount; ++i)
        storeHalf(gParamGrads, i, 0.0);
}

void initializeParams()
{
    for (uint i = 0; i < kParamElementCount; ++i)
    {
        float v = 0.05 * randomSigned01(i + 1u);
        storeHalf(gParamsRW, i, v);
    }
    clearGradientBuffer();
}

[Differentiable]
float3 gammaCorrect(float3 color)
{
    return pow(max(color, float3(1e-6)), 1.0 / 2.2);
}

float3 modelInfer(float2 uv)
{
    uint weightOffsets[kTransitions];
    uint biasOffsets[kTransitions];
    computeLayerOffsets(weightOffsets, biasOffsets);

    var model = rtxns::mlp::InferenceMLP<
        half,
        kHiddenLayers,
        kInputSize,
        kHiddenSize,
        kOutputSize,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16
    >(gParams, weightOffsets, biasOffsets);

    var hiddenActivation = rtxns::mlp::LeakyReLUAct<half, kHiddenSize>(half(0.01));
    var finalActivation = rtxns::mlp::ExponentialAct<half, kOutputSize>();
    var inputParams = CoopVec<half, kInputSize>(half(uv.x), half(uv.y));
    var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);
    return float3(outputParams[0], outputParams[1], outputParams[2]);
}

void optimizerStep()
{
    for (uint i = 0; i < kParamElementCount; ++i)
    {
        float grad = loadHalf(gParamGrads, i);
        float param = loadHalf(gParamsRW, i);
        param -= gLearningRate * grad;
        storeHalf(gParamsRW, i, param);
        storeHalf(gParamGrads, i, 0.0);
    }
}

void trainStep(uint step)
{
    uint sampleSeed = gFrameIndex * 1664525u + step * 1013904223u;
    uint2 pixel = uint2(
        hash(sampleSeed) % max(1u, gFrameDim.x),
        hash(sampleSeed ^ 0x9e3779b9u) % max(1u, gFrameDim.y)
    );

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    float3 target = gInput.Load(int3(pixel, 0)).xyz;

    uint weightOffsets[kTransitions];
    uint biasOffsets[kTransitions];
    computeLayerOffsets(weightOffsets, biasOffsets);

    var model = rtxns::mlp::TrainingMLP<
        half,
        kHiddenLayers,
        kInputSize,
        kHiddenSize,
        kOutputSize,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16
    >(gParams, gParamGrads, weightOffsets, biasOffsets);

    var hiddenActivation = rtxns::mlp::LeakyReLUAct<half, kHiddenSize>(half(0.01));
    var finalActivation = rtxns::mlp::ExponentialAct<half, kOutputSize>();
    var inputParams = CoopVec<half, kInputSize>(half(uv.x), half(uv.y));

    var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);
    float3 pred = float3(outputParams[0], outputParams[1], outputParams[2]);

    float3 diff = gammaCorrect(pred) - gammaCorrect(target);
    float3 lossGrad = diff;

    model.backward(
        inputParams,
        hiddenActivation,
        finalActivation,
        CoopVec<half, kOutputSize>(half(lossGrad.x), half(lossGrad.y), half(lossGrad.z))
    );

    optimizerStep();
}

[numthreads(1, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x > 0 || tid.y > 0 || tid.z > 0)
        return;

    if (gReset != 0)
        initializeParams();

    for (uint i = 0; i < gTrainSteps; ++i)
        trainStep(i);
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
