#include "Utils/Math/MathConstants.slangh"

import RTXNS.PCG32;
import RTXNS.Utils;
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
    buffer.StoreAligned<float>(elementIndex * sizeof(float), value);
}

float LoadFloat(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.LoadAligned<float>(elementIndex * sizeof(float));
}

void StoreHalf(RWByteAddressBuffer buffer, uint elementIndex, half value)
{
    buffer.StoreAligned<half>(elementIndex * sizeof(half), value);
}

half LoadHalf(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.LoadAligned<half>(elementIndex * sizeof(half));
}

float sanitize(float x, float fallback = 0.0f)
{
    return (isnan(x) || isinf(x)) ? fallback : x;
}

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
    const float oneMinusBeta1 = 1.0 - kAdamBeta1;
    const float oneMinusBeta2 = 1.0 - kAdamBeta2;
    const float invLossScale = 1.0 / kOptimizerLossScale;

    // These are uniform for the whole dispatch; compute once per thread instead of once per parameter.
    const float biasCorrection1 = max(1.0 - pow(kAdamBeta1, gCurrentStep), 1e-20);
    const float invSqrtBiasCorrection2 = rsqrt(max(1.0 - pow(kAdamBeta2, gCurrentStep), 1e-20));
    const float stepSize = gLearningRate / biasCorrection1;

    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        float gradient = LoadHalf(gParamGrads, i);
        gradient = sanitize(gradient * invLossScale);

        float p = LoadHalf(gParams, i);
        float moment1 = LoadFloat(gMoments1, i) * kAdamBeta1 + gradient * oneMinusBeta1;
        float moment2 = LoadFloat(gMoments2, i) * kAdamBeta2 + (gradient * gradient) * oneMinusBeta2;
        float denom = sqrt(moment2) * invSqrtBiasCorrection2 + kAdamEpsilon;
        p -= (moment1 / denom) * stepSize;

        StoreFloat(gMoments1, i, moment1);
        StoreFloat(gMoments2, i, moment2);
        StoreHalf(gParams, i, p);
    }

    for (uint i = tid.x; i < kEncodingParamElementCount; i += gDispatchThreadCount)
    {
        float gradient = LoadFloat(gEncodingParamGrads, i);

        gradient = sanitize(gradient * invLossScale);

        float p = LoadHalf(gEncodingParams, i);
        float moment1 = LoadFloat(gEncodingMoments1, i) * kAdamBeta1 + gradient * oneMinusBeta1;
        float moment2 = LoadFloat(gEncodingMoments2, i) * kAdamBeta2 + (gradient * gradient) * oneMinusBeta2;
        float denom = sqrt(moment2) * invSqrtBiasCorrection2 + kAdamEpsilon;
        p -= (moment1 / denom) * stepSize;

        StoreFloat(gEncodingMoments1, i, moment1);
        StoreFloat(gEncodingMoments2, i, moment2);
        StoreHalf(gEncodingParams, i, p);
    }
}
