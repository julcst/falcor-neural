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

[BackwardDerivative(loadHalf2_bwd)]
half2 loadHalf2(FeatureBuffer b, int index)
{
    return b.buffer.Load<half2>(index * sizeof(half2));
}

void loadHalf2_bwd(DifferentialPtrPair<FeatureBuffer> b, int index, half2 dOut)
{
    float o;
    // NOTE: InterlockedAddF16 is not available on all platforms
    b.d.buffer.InterlockedAddF32(index * sizeof(half2), dOut.x, o);
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
    [ForceUnroll]
    for (int i = 0; i < M; ++i) dst[startIndex + i] = value[i];
}

void setRange_bwd<T : __BuiltinFloatingPointType, int N, int M>(inout DifferentialPair<CoopVec<T, N>> dDst, no_diff uint startIndex, inout DifferentialPair<CoopVec<T, M>> value)
{
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

    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingle<T, FEATURES>(features, uv, level, levelResolution, hashTableSize);
        setRange(encoded, level * FEATURES, levelFeatures);
        levelResolution *= scale;
    }

    return encoded;
}

[BackwardDerivative(setRange_bwd)]
void setRange<T : __BuiltinFloatingPointType, int N, int M>(inout T[N] dst, no_diff uint startIndex, T[M] value)
{
    [ForceUnroll]
    for (int i = 0; i < M; ++i) dst[startIndex + i] = value[i];
}

void setRange_bwd<T : __BuiltinFloatingPointType, int N, int M>(inout DifferentialPair<T[N]> dDst, no_diff uint startIndex, inout DifferentialPair<T[M]> value)
{
    var dValue = value.d;
    [ForceUnroll]
    for (int i = 0; i < M; ++i) dValue[i] = T.dadd(dValue[i], dDst.d[startIndex + i]);
    value = diffPair(value.p, dValue);
}

[Differentiable]
half2 encodeHashGridSingleHalf2(FeatureBuffer features, no_diff float2 uv, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
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

    let f00 = loadVec<half, 2>(features, i00) * half(w00);
    let f10 = loadVec<half, 2>(features, i10) * half(w10);
    let f01 = loadVec<half, 2>(features, i01) * half(w01);
    let f11 = loadVec<half, 2>(features, i11) * half(w11);

    return f00 + f01 + f10 + f11;
}

[Differentiable]
CoopVec<half, LEVELS * 2> encodeHashGridHalf2<uint LEVELS>(FeatureBuffer features, no_diff float2 uv, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    float encoded[LEVELS * 2];

    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingleHalf2(features, uv, level, levelResolution, hashTableSize);
        encoded[level * 2] = levelFeatures.x;
        encoded[level * 2 + 1] = levelFeatures.y;
        levelResolution *= scale;
    }

    return rtxns::CoopVecFromArray<half, LEVELS * 2>(encoded);
}
