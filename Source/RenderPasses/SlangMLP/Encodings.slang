import RTXNS.Utils;
import RTXNS.CooperativeVectorAutoDiff;
import Config;

using MLPConfig;

struct FeatureBufferDifferential : IDifferentiablePtrType
{
    typealias Differential = FeatureBufferDifferential;

    __init(RWByteAddressBuffer buf)
    {
        buffer = buf;
    }

    RWByteAddressBuffer buffer;
};

struct FeatureBuffer : IDifferentiablePtrType
{
    typealias Differential = FeatureBufferDifferential;

    __init(ByteAddressBuffer buf)
    {
        buffer = buf;
    }

    ByteAddressBuffer buffer;
};

constexpr T align<T: __BuiltinIntegerType>(T x, T alignment)
{
    return (x + alignment - T(1)) & ~(alignment - T(1));
}

[BackwardDerivative(load_bwd)]
CoopVec<T, N> load<T : __BuiltinFloatingPointType, int N>(FeatureBuffer b, int index)
{
    return coopVecLoad<N, T>(b.buffer, index * align(sizeof(CoopVec<T, N>), 16));
}

void load_bwd<T : __BuiltinFloatingPointType, int N>(DifferentialPtrPair<FeatureBuffer> b, int index, CoopVec<T, N> dOut)
{
    coopVecReduceSumAccumulate(dOut, b.d.buffer, index * align(sizeof(CoopVec<T, N>), 16));
}

uint hashGridIndex(uint2 cell, uint level, uint size)
{
    uint h = cell.x * 0x8da6b343u;
    h ^= cell.y * 0xd8163841u;
    h ^= level * 0xcb1ab31fu;
    return h & (size - 1u);
}

uint levelResolution(uint level, float base, float scale)
{
    float resolution = base * pow(scale, float(level));
    return max(2u, uint(resolution + 0.5));
}

[Differentiable]
CoopVec<T, LEVELS * FEATURES> encodeHashGrid<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    var encoded = CoopVec<T, LEVELS * FEATURES>(0);

    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let resolution = levelResolution(level, baseResolution, scale);
        let scaled = uv * float(resolution - 1u);

        let p0 = uint2(floor(scaled));
        let p1 = p0 + uint2(1, 0);
        let p2 = p0 + uint2(0, 1);
        let p3 = p0 + uint2(1, 1);

        float2 fracCoord = frac(scaled);
        let w00 = (1.0 - fracCoord.x) * (1.0 - fracCoord.y);
        let w10 = fracCoord.x * (1.0 - fracCoord.y);
        let w01 = (1.0 - fracCoord.x) * fracCoord.y;
        let w11 = fracCoord.x * fracCoord.y;

        uint h00 = hashGridIndex(p0, level, hashTableSize);
        uint h10 = hashGridIndex(p1, level, hashTableSize);
        uint h01 = hashGridIndex(p2, level, hashTableSize);
        uint h11 = hashGridIndex(p3, level, hashTableSize);

        uint levelBase = level * hashTableSize;
        uint outputBase = level * FEATURES;

        uint idx00 = (levelBase + h00) * FEATURES;
        uint idx10 = (levelBase + h10) * FEATURES;
        uint idx01 = (levelBase + h01) * FEATURES;
        uint idx11 = (levelBase + h11) * FEATURES;

        let f00 = load<T, FEATURES>(features, idx00);
        let f10 = load<T, FEATURES>(features, idx10);
        let f01 = load<T, FEATURES>(features, idx01);
        let f11 = load<T, FEATURES>(features, idx11);

        [ForceUnroll]
        for (uint f = 0u; f < FEATURES; ++f)
        {
            encoded[outputBase + f] = f00[f] * T(w00) + f10[f] * T(w10) + f01[f] * T(w01) + f11[f] * T(w11);
        }
    }

    return encoded;
}
