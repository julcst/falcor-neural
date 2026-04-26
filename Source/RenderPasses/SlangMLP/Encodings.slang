import RTXNS.Utils;
import RTXNS.CooperativeVectorAutoDiff;
import Utils.Math.HashUtils;

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
    let stride = N * sizeof(T); // NOTE: Documentation says aligned to 16 but it works without it
    return coopVecLoad<N, T>(b.buffer, index * stride);
}

void load_bwd<T : __BuiltinFloatingPointType, int N>(DifferentialPtrPair<FeatureBuffer> b, int index, CoopVec<T, N> dOut)
{
    let stride = N * sizeof(T); // NOTE: Documentation says aligned to 16 but it works without it
    coopVecReduceSumAccumulate(dOut, b.d.buffer, index * stride);
}

[Differentiable]
void setRange<T : __BuiltinFloatingPointType, int N, int M>(inout CoopVec<T, N> dst, uint startIndex, CoopVec<T, M> value)
{
    [ForceUnroll]
    for (int i = 0; i < M; ++i) dst[startIndex + i] = value[i];
}

uint hashGridIndex(uint2 cell, uint level, uint size)
{
    // TODO: Use faster hash function if necessary
    let h = pcg3d(uint3(cell, level));
    return h.x & (size - 1u);
}

[Differentiable]
CoopVec<T, FEATURES> encodeHashGridSingle<T : __BuiltinFloatingPointType, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
{
    float2 scaled = uv * float(resolution - 1u);
    let p0 = uint2(floor(scaled));
    let p1 = p0 + uint2(1, 0);
    let p2 = p0 + uint2(0, 1);
    let p3 = p0 + uint2(1, 1);

    float2 fracCoord = frac(scaled);
    let w00 = (1.0 - fracCoord.x) * (1.0 - fracCoord.y);
    let w10 = fracCoord.x * (1.0 - fracCoord.y);
    let w01 = (1.0 - fracCoord.x) * fracCoord.y;
    let w11 = fracCoord.x * fracCoord.y;

    uint base = level * hashTableSize;
    uint i00 = hashGridIndex(p0, level, hashTableSize) + base;
    uint i10 = hashGridIndex(p1, level, hashTableSize) + base;
    uint i01 = hashGridIndex(p2, level, hashTableSize) + base;
    uint i11 = hashGridIndex(p3, level, hashTableSize) + base;

    let f00 = load<T, FEATURES>(features, i00) * CoopVec<T, FEATURES>.replicate(T(w00));
    let f10 = load<T, FEATURES>(features, i10) * CoopVec<T, FEATURES>.replicate(T(w10));
    let f01 = load<T, FEATURES>(features, i01) * CoopVec<T, FEATURES>.replicate(T(w01));
    let f11 = load<T, FEATURES>(features, i11) * CoopVec<T, FEATURES>.replicate(T(w11));

    return f00 + f01 + f10 + f11;
}

[Differentiable]
CoopVec<T, LEVELS * FEATURES> encodeHashGrid<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    var encoded = CoopVec<T, LEVELS * FEATURES>(0);

    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        float levelResolution = baseResolution * pow(scale, float(level));
        let levelFeatures = encodeHashGridSingle<T, FEATURES>(features, uv, level, levelResolution, hashTableSize);
        setRange(encoded, level * FEATURES, levelFeatures);
    }

    return encoded;
}
