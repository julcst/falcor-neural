#include "Utils/Math/MathConstants.slangh"

import RTXNS.PCG32;
import RTXNS.Utils;
import Config;

using MLPConfig;

RWByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads; // This is read-only
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;
RWByteAddressBuffer gEncodingParams;
RWByteAddressBuffer gEncodingParamGrads; // This is read-only
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

void StoreHalf(RWByteAddressBuffer buffer, uint elementIndex, float value)
{
    buffer.StoreAligned<half>(elementIndex * sizeof(half), half(value));
}

float LoadHalf(RWByteAddressBuffer buffer, uint elementIndex)
{
    return float(buffer.LoadAligned<half>(elementIndex * sizeof(half)));
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

float4 LoadHalf4(RWByteAddressBuffer buffer, uint elementIndex)
{
    return float4(buffer.LoadAligned<half4>(elementIndex * sizeof(half)));
}

void StoreHalf4(RWByteAddressBuffer buffer, uint elementIndex, float4 v)
{
    buffer.StoreAligned<half4>(elementIndex * sizeof(half), half4(v));
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

float sanitize(float x, float fallback = 0.0f)
{
    return (isnan(x) || isinf(x)) ? fallback : x;
}

float4 sanitize(float4 v, float fallback = 0.0f)
{
    return float4(sanitize(v.x, fallback), sanitize(v.y, fallback), sanitize(v.z, fallback), sanitize(v.w, fallback));
}

struct Adam
{
    // Parameters
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    // Constant parameters derived from the above
    float invSqrtBiasCorrection2;
    float stepSize;
    float invLossScale;

    __init(float learningRate, float beta1, float beta2, float epsilon, float t, float lossScale = 1.0)
    {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        let biasCorrection1 = max(1.0 - pow(beta1, t), 1e-20);
        this.stepSize = learningRate / biasCorrection1;
        this.invSqrtBiasCorrection2 = rsqrt(max(1.0 - pow(beta2, t), 1e-20));
        this.invLossScale = 1.0 / lossScale;
    }

    bool step<uint N>(inout vector<float, N> param, inout vector<float, N> moment1, inout vector<float, N> moment2, vector<float, N> gradient)
    {
        gradient *= invLossScale;
        moment1 = lerp(gradient, moment1, beta1);
        moment2 = lerp(gradient * gradient, moment2, beta2);
        let denom = sqrt(moment2) * invSqrtBiasCorrection2 + epsilon;
        let step = (moment1 / denom) * stepSize;
        param -= step;
        return any(step > 1e-6); // Return whether any parameter had a significant update
    }
    
};

static const Adam adam = Adam(gLearningRate, kAdamBeta1, kAdamBeta2, kAdamEpsilon, gCurrentStep, kOptimizerLossScale);

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    // Vectorized processing in contiguous float4/half4 batches for params
    // static_assert(kParamElementCount % 4u == 0, "Expecting param count to be divisible by 4 for vectorized processing");
    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p = LoadHalf4(gParams, base);
        let gradient = sanitize(LoadHalf4(gParamGrads, base));
        var moment1 = LoadFloat4(gMoments1, base);
        var moment2 = LoadFloat4(gMoments2, base);

        adam.step(p, moment1, moment2, gradient);

        StoreFloat4(gMoments1, base, moment1);
        StoreFloat4(gMoments2, base, moment2);
        StoreHalf4(gParams, base, p);
    }

    // Vectorized processing in contiguous float4/half4 batches for encoding params
    // static_assert(kEncodingParamElementCount % 4u == 0, "Expecting encoding param count to be divisible by 4 for vectorized processing");
    for (uint base = tid.x * 4u; base < kEncodingParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p = LoadHalf4(gEncodingParams, base);
        let gradient = sanitize(LoadFloat4(gEncodingParamGrads, base));
        var moment1 = LoadFloat4(gEncodingMoments1, base);
        var moment2 = LoadFloat4(gEncodingMoments2, base);

        let updated = adam.step(p, moment1, moment2, gradient);

        StoreFloat4(gEncodingMoments1, base, moment1);
        StoreFloat4(gEncodingMoments2, base, moment2);
        if (updated) StoreHalf4(gEncodingParams, base, p); // Skip store if no significant update to save memory bandwidth
    }
}
