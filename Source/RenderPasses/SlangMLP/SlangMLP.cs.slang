#include "Utils/Math/MathConstants.slangh"

static const uint kHiddenSize = 8;

Texture2D<float4> gInput;
RWTexture2D<float4> gOutput;
RWByteAddressBuffer gParams;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gTrainSteps;
    float gLearningRate;
    uint gReset;
}

struct NetworkParams : IDifferentiable
{
    float2 w1[kHiddenSize];
    float b1[kHiddenSize];
    float3 w2[kHiddenSize];
    float3 b2;
};

uint paramOffset(uint index)
{
    return index * 4u;
}

float loadParam(uint index)
{
    return asfloat(gParams.Load(paramOffset(index)));
}

void storeParam(uint index, float value)
{
    gParams.Store(paramOffset(index), asuint(value));
}

NetworkParams loadNetworkParams()
{
    NetworkParams p = {};
    uint idx = 0;

    for (uint i = 0; i < kHiddenSize; ++i)
    {
        p.w1[i].x = loadParam(idx++);
        p.w1[i].y = loadParam(idx++);
    }

    for (uint i = 0; i < kHiddenSize; ++i)
        p.b1[i] = loadParam(idx++);

    for (uint i = 0; i < kHiddenSize; ++i)
    {
        p.w2[i].x = loadParam(idx++);
        p.w2[i].y = loadParam(idx++);
        p.w2[i].z = loadParam(idx++);
    }

    p.b2.x = loadParam(idx++);
    p.b2.y = loadParam(idx++);
    p.b2.z = loadParam(idx++);

    return p;
}

void storeNetworkParams(NetworkParams p)
{
    uint idx = 0;

    for (uint i = 0; i < kHiddenSize; ++i)
    {
        storeParam(idx++, p.w1[i].x);
        storeParam(idx++, p.w1[i].y);
    }

    for (uint i = 0; i < kHiddenSize; ++i)
        storeParam(idx++, p.b1[i]);

    for (uint i = 0; i < kHiddenSize; ++i)
    {
        storeParam(idx++, p.w2[i].x);
        storeParam(idx++, p.w2[i].y);
        storeParam(idx++, p.w2[i].z);
    }

    storeParam(idx++, p.b2.x);
    storeParam(idx++, p.b2.y);
    storeParam(idx++, p.b2.z);
}

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float randomSigned01(uint seed)
{
    return (float(hash(seed) & 0x00ffffffu) / float(0x00ffffffu)) * 2.0 - 1.0;
}

void initializeParams()
{
    NetworkParams p = {};

    for (uint i = 0; i < kHiddenSize; ++i)
    {
        p.w1[i] = 0.5 * float2(randomSigned01(i * 2u + 1u), randomSigned01(i * 2u + 2u));
        p.b1[i] = 0.0;
        p.w2[i] = 0.25 * float3(
            randomSigned01(100u + i * 3u + 0u),
            randomSigned01(100u + i * 3u + 1u),
            randomSigned01(100u + i * 3u + 2u)
        );
    }

    p.b2 = float3(0.0);
    storeNetworkParams(p);
}

[Differentiable]
float3 evalMLP(NetworkParams p, float2 uv)
{
    float3 y = p.b2;
    for (uint i = 0; i < kHiddenSize; ++i)
    {
        float h = max(0.0, dot(p.w1[i], uv) + p.b1[i]);
        y += h * p.w2[i];
    }
    return y;
}

[Differentiable]
float computeLoss(NetworkParams p, float2 uv, no_diff float3 target)
{
    float3 pred = evalMLP(p, uv);
    float3 err = pred - target;
    return 0.5 * dot(err, err);
}

void trainStep(uint step)
{
    uint sampleSeed = gFrameIndex * 1664525u + step * 1013904223u;
    uint2 pixel = uint2(hash(sampleSeed) % gFrameDim.x, hash(sampleSeed ^ 0x9e3779b9u) % gFrameDim.y);

    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    float3 target = gInput.Load(int3(pixel, 0)).xyz;

    NetworkParams p = loadNetworkParams();
    DifferentialPair<NetworkParams> dp = diffPair(p, NetworkParams.Differential());
    DifferentialPair<float2> dpUv = diffPair(uv, float2(0.0));
    bwd_diff(computeLoss)(dp, dpUv, target, 1.0);

    for (uint i = 0; i < kHiddenSize; ++i)
    {
        p.w1[i] -= gLearningRate * dp.d.w1[i];
        p.b1[i] -= gLearningRate * dp.d.b1[i];
        p.w2[i] -= gLearningRate * dp.d.w2[i];
    }
    p.b2 -= gLearningRate * dp.d.b2;

    storeNetworkParams(p);
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

    NetworkParams p = loadNetworkParams();
    float2 uv = (float2(pixel) + 0.5) / float2(gFrameDim);
    float3 pred = evalMLP(p, uv);

    gOutput[pixel] = float4(saturate(pred), 1.0);
}
