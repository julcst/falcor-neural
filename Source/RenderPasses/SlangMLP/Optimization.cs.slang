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

cbuffer CB
{
    bool gReset;
    float gLearningRate;
    float gCurrentStep;
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

void resetParam(uint index, inout PCG32 rng)
{
    const float scale = sqrt(2.0 / float(kHiddenSize));
    float p = rng.nextFloat() * 2.0 - 1.0;
    StoreHalf(gParams, index, p * scale);
    StoreHalf(gParamGrads, index, 0.0);
    StoreFloat(gMoments1, index, 0.0);
    StoreFloat(gMoments2, index, 0.0);
}

[numthreads(64, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= kParamElementCount)
        return;

    if (gReset != 0)
    {
        PCG32 rng = PCG32(i, i ^ 0x9e3779b97f4a7c15ULL);
        resetParam(i, rng);
        return;
    }

    float gradient = LoadHalf(gParamGrads, i);
    StoreHalf(gParamGrads, i, 0.0);

    let adam = optimizers::AdamAccessor<ByteAddressAccessor>(
        ByteAddressAccessor(gMoments1),
        ByteAddressAccessor(gMoments2),
        gLearningRate,  
        kOptimizerLossScale,
        kAdamBeta1,
        kAdamBeta2,
        kAdamEpsilon
    );

    float p = LoadHalf(gParams, i);
    p = adam.step(p, i, gradient, gCurrentStep);
    StoreHalf(gParams, i, p);
}