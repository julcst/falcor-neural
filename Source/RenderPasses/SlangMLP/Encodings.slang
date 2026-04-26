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

[BackwardDerivative(loadVec_bwd)]
vector<T, N> loadVec<T : __BuiltinFloatingPointType, int N>(FeatureBuffer b, int index)
{
    return b.buffer.Load<vector<T,N>>(index * N * sizeof(T));
}

void loadVec_bwd<T : __BuiltinFloatingPointType, int N>(DifferentialPtrPair<FeatureBuffer> b, int index, vector<T, N> dOut)
{
    float o;
    // NOTE: InterlockedAdd is only well supported for F32
    [ForceUnroll]
    for (int i = 0; i < N; ++i) b.d.buffer.InterlockedAddF32((index * N + i) * sizeof(float), dOut[i].toFloat(), o);
    // TODO: Use faster _NvInterlockedAddF16x2 when available
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

[BackwardDerivative(setRange_bwd)]
void setRange<T : __BuiltinFloatingPointType, int N, int M>(inout CoopVec<T, N> dst, no_diff uint startIndex, CoopVec<T, M> value)
{
    // NOTE: This does not really work as CoopVec is warp level
    [ForceUnroll]
    for (int i = 0; i < M; ++i) dst[startIndex + i] = value[i];
}

void setRange_bwd<T : __BuiltinFloatingPointType, int N, int M>(inout DifferentialPair<CoopVec<T, N>> dDst, no_diff uint startIndex, inout DifferentialPair<CoopVec<T, M>> value)
{
    // NOTE: This does not really work as CoopVec is warp level
    var dValue = value.d;
    [ForceUnroll]
    for (int i = 0; i < M; ++i) dValue[i] += dDst.d[startIndex + i];
    value = diffPair(value.p, dValue);
}

uint hashGridIndex(uint2 cell, uint level, uint size)
{
    // TODO: Use faster hash function if necessary
    let h = pcg3d(uint3(cell, level));
    return h.x & (size - 1u);
}

[Differentiable]
CoopVec<T, FEATURES> encodeHashGridSingleCoop<T : __BuiltinFloatingPointType, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
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
CoopVec<T, LEVELS * FEATURES> encodeHashGridCoop<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    float encoded[LEVELS * FEATURES];

    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingleCoop<T, FEATURES>(features, uv, level, levelResolution, hashTableSize);
        [ForceUnroll]
        for (uint i = 0u; i < FEATURES; ++i) encoded[level * FEATURES + i] = levelFeatures[i].toFloat();
        levelResolution *= scale;
    }

    return rtxns::CoopVecFromArray<T, LEVELS * FEATURES>(encoded);
}

// NOTE: This uses vector<T, N> so only avaliable for N <= 4 (but larger is generally not required for encoding)
[Differentiable]
vector<T, FEATURES> encodeHashGridSingle<T : __BuiltinFloatingPointType, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
{
    static_assert(FEATURES <= 4, "encodeHashGridSingle only supports up to 4 features due to using vector<T, N> for loading");
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

    let f00 = loadVec<T, FEATURES>(features, i00) * T(w00);
    let f10 = loadVec<T, FEATURES>(features, i10) * T(w10);
    let f01 = loadVec<T, FEATURES>(features, i01) * T(w01);
    let f11 = loadVec<T, FEATURES>(features, i11) * T(w11);

    return f00 + f01 + f10 + f11;
}

[Differentiable]
CoopVec<T, LEVELS * FEATURES> encodeHashGrid<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    // TODO: Make this T
    float encoded[LEVELS * FEATURES];

    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingle<T, FEATURES>(features, uv, level, levelResolution, hashTableSize);
        [ForceUnroll]
        for (uint i = 0u; i < FEATURES; ++i) encoded[level * FEATURES + i] = levelFeatures[i].toFloat();
        levelResolution *= scale;
    }

    return rtxns::CoopVecFromArray<T, LEVELS * FEATURES>(encoded);
}
