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

// Vectorized helpers
float4 LoadFloat4(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.LoadAligned<float4>(elementIndex * sizeof(float));
}

void StoreFloat4(RWByteAddressBuffer buffer, uint elementIndex, float4 v)
{
    buffer.StoreAligned<float4>(elementIndex * sizeof(float), v);
}

half4 LoadHalf4(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.LoadAligned<half4>(elementIndex * sizeof(half));
}

void StoreHalf4(RWByteAddressBuffer buffer, uint elementIndex, half4 v)
{
    buffer.StoreAligned<half4>(elementIndex * sizeof(half), v);
}

float sanitize(float x, float fallback = 0.0f)
{
    return (isnan(x) || isinf(x)) ? fallback : x;
}

float4 sanitize(float4 v, float fallback = 0.0f)
{
    return float4(sanitize(v.x, fallback), sanitize(v.y, fallback), sanitize(v.z, fallback), sanitize(v.w, fallback));
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

    // Vectorized processing in contiguous float4/half4 batches for params
    uint lastVectorStart = (kParamElementCount / 4u) * 4u;
    for (uint base = tid.x * 4u; base + 3u < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        half4 grad_h = LoadHalf4(gParamGrads, base);
        float4 gradient = sanitize(float4(grad_h) * invLossScale);

        float4 p = float4(LoadHalf4(gParams, base));
        float4 moment1 = LoadFloat4(gMoments1, base) * kAdamBeta1 + gradient * oneMinusBeta1;
        float4 moment2 = LoadFloat4(gMoments2, base) * kAdamBeta2 + (gradient * gradient) * oneMinusBeta2;
        float4 denom = sqrt(moment2) * invSqrtBiasCorrection2 + float4(kAdamEpsilon);
        p -= (moment1 / denom) * float4(stepSize);

        StoreFloat4(gMoments1, base, moment1);
        StoreFloat4(gMoments2, base, moment2);
        StoreHalf4(gParams, base, half4(p));
    }
    // Scalar tail
    for (uint i = lastVectorStart + tid.x; i < kParamElementCount; i += gDispatchThreadCount)
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

    // Vectorized processing in contiguous float4/half4 batches for encoding params
    uint lastEncVectorStart = (kEncodingParamElementCount / 4u) * 4u;
    for (uint base = tid.x * 4u; base + 3u < kEncodingParamElementCount; base += gDispatchThreadCount * 4u)
    {
        float4 gradient = LoadFloat4(gEncodingParamGrads, base) * float4(invLossScale);
        gradient = sanitize(gradient);

        float4 p = float4(LoadHalf4(gEncodingParams, base));
        float4 moment1 = LoadFloat4(gEncodingMoments1, base) * kAdamBeta1 + gradient * oneMinusBeta1;
        float4 moment2 = LoadFloat4(gEncodingMoments2, base) * kAdamBeta2 + (gradient * gradient) * oneMinusBeta2;
        float4 denom = sqrt(moment2) * invSqrtBiasCorrection2 + float4(kAdamEpsilon);
        p -= (moment1 / denom) * float4(stepSize);

        StoreFloat4(gEncodingMoments1, base, moment1);
        StoreFloat4(gEncodingMoments2, base, moment2);
        StoreHalf4(gEncodingParams, base, half4(p));
    }
    for (uint i = lastEncVectorStart + tid.x; i < kEncodingParamElementCount; i += gDispatchThreadCount)
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
