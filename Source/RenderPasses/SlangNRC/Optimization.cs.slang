#include "Utils/Math/MathConstants.slangh"

import RTXNS.PCG32;
import RTXNS.Utils;
import RTXNS.Optimizers;
import Config;

using MLPConfig;

RWByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;
RWByteAddressBuffer gEncodingParams;
RWByteAddressBuffer gEncodingParamGrads;
RWByteAddressBuffer gEncodingMoments1;
RWByteAddressBuffer gEncodingMoments2;

cbuffer CB
{
    float gLearningRate;
    float gCurrentStep;
    uint gDispatchThreadCount;
}

void StoreFloat(RWByteAddressBuffer buffer, uint elementIndex, float value)
{
    buffer.Store<float>(elementIndex * sizeof(float), value);
}

float LoadFloat(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.Load<float>(elementIndex * sizeof(float));
}

void StoreHalf(RWByteAddressBuffer buffer, uint elementIndex, half value)
{
    buffer.Store<half>(elementIndex * sizeof(half), value);
}

half LoadHalf(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.Load<half>(elementIndex * sizeof(half));
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
        return LoadFloat(mBuffer, index);
    }

    void set(uint index, float value)
    {
        StoreFloat(mBuffer, index, value);
    }
};

[numthreads(256, 1, 1)]
void resetMain(uint3 tid : SV_DispatchThreadID)
{
    PCG32 rng = PCG32(tid.x, tid.x ^ 0x9e3779b97f4a7c15ULL);
    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        const float scale = sqrt(2.0 / float(kHiddenSize));
        float p = rng.nextFloat() * 2.0 - 1.0;
        StoreHalf(gParams, i, half(p * scale));
        StoreFloat(gMoments1, i, 0.0);
        StoreFloat(gMoments2, i, 0.0);
    }
    for (uint i = tid.x; i < kEncodingParamElementCount; i += gDispatchThreadCount)
    {
        const float scale = 1e-3;
        float p = rng.nextFloat() * 2.0 - 1.0;
        StoreHalf(gEncodingParams, i, half(p * scale));
        StoreFloat(gEncodingMoments1, i, 0.0);
        StoreFloat(gEncodingMoments2, i, 0.0);
    }
}

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    let adam = optimizers::AdamAccessor<ByteAddressAccessor>(
        ByteAddressAccessor(gMoments1),
        ByteAddressAccessor(gMoments2),
        gLearningRate,
        kOptimizerLossScale,
        kAdamBeta1,
        kAdamBeta2,
        kAdamEpsilon
    );

    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        float gradient = LoadHalf(gParamGrads, i);
        StoreHalf(gParamGrads, i, 0.0);

        float p = LoadHalf(gParams, i);
        p = adam.step(p, i, gradient, gCurrentStep);
        StoreHalf(gParams, i, p);
    }

    let adamEnc = optimizers::AdamAccessor<ByteAddressAccessor>(
        ByteAddressAccessor(gEncodingMoments1),
        ByteAddressAccessor(gEncodingMoments2),
        gLearningRate,
        kOptimizerLossScale,
        kAdamBeta1,
        kAdamBeta2,
        kAdamEpsilon
    );

    for (uint i = tid.x; i < kEncodingParamElementCount; i += gDispatchThreadCount)
    {
        float gradient = LoadFloat(gEncodingParamGrads, i);
        StoreFloat(gEncodingParamGrads, i, 0.0);

        // Skipping small gradients reduces bandwidth and improves performance (-0.2ms)
        if (abs(gradient) < 1e-6)
            continue;

        float p = LoadHalf(gEncodingParams, i);
        p = adamEnc.step(p, i, gradient, gCurrentStep);
        StoreHalf(gEncodingParams, i, p);
    }
}