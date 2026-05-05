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
    return b.buffer.LoadAligned<vector<T,N>>(index * N * sizeof(T));
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


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint32_t morton3D(uint3 v) {
    let xx = expand_bits(v.x);
    let yy = expand_bits(v.y);
    let zz = expand_bits(v.z);
    return xx | (yy << 1) | (zz << 2);
}

uint hashGridIndex(uint3 cell, uint level, uint size)
{
    // Using morton code as hash function significantly improves coherence (+1.5ms) but has stripe artifacts
    // let h = morton3D(cell ^ uint3(level, level * 1664525u, level * 1013904223u));
    let h = pcg3d(cell^ uint3(level, level * 1664525u, level * 1013904223u) );
    return h.x & (size - 1u);
}

// NOTE: This uses vector<T, N> so only available for N <= 4 (but larger is generally not required for encoding)
[Differentiable]
vector<T, FEATURES> encodeHashGridSingle<T : __BuiltinFloatingPointType, uint FEATURES>(FeatureBuffer features, no_diff float3 position, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
{
    static_assert(FEATURES <= 4, "encodeHashGridSingle only supports up to 4 features due to using vector<T, N> for loading");
    float3 scaled = position * float(resolution - 1u);
    let p000 = uint3(floor(scaled));
    let p100 = p000 + uint3(1, 0, 0);
    let p010 = p000 + uint3(0, 1, 0);
    let p110 = p000 + uint3(1, 1, 0);
    let p001 = p000 + uint3(0, 0, 1);
    let p101 = p000 + uint3(1, 0, 1);
    let p011 = p000 + uint3(0, 1, 1);
    let p111 = p000 + uint3(1, 1, 1);

    float3 fracCoord = frac(scaled);
    let wx0 = (1.0 - fracCoord.x);
    let wx1 = fracCoord.x;
    let wy0 = (1.0 - fracCoord.y);
    let wy1 = fracCoord.y;
    let wz0 = (1.0 - fracCoord.z);
    let wz1 = fracCoord.z;

    let w000 = wx0 * wy0 * wz0;
    let w100 = wx1 * wy0 * wz0;
    let w010 = wx0 * wy1 * wz0;
    let w110 = wx1 * wy1 * wz0;
    let w001 = wx0 * wy0 * wz1;
    let w101 = wx1 * wy0 * wz1;
    let w011 = wx0 * wy1 * wz1;
    let w111 = wx1 * wy1 * wz1;

    uint base = level * hashTableSize;
    uint i000 = hashGridIndex(p000, level, hashTableSize) + base;
    uint i100 = hashGridIndex(p100, level, hashTableSize) + base;
    uint i010 = hashGridIndex(p010, level, hashTableSize) + base;
    uint i110 = hashGridIndex(p110, level, hashTableSize) + base;
    uint i001 = hashGridIndex(p001, level, hashTableSize) + base;
    uint i101 = hashGridIndex(p101, level, hashTableSize) + base;
    uint i011 = hashGridIndex(p011, level, hashTableSize) + base;
    uint i111 = hashGridIndex(p111, level, hashTableSize) + base;

    // Accumulate weighted features in single pass to reduce intermediate allocations
    var result = loadVec<T, FEATURES>(features, i000) * T(w000);
    result += loadVec<T, FEATURES>(features, i100) * T(w100);
    result += loadVec<T, FEATURES>(features, i010) * T(w010);
    result += loadVec<T, FEATURES>(features, i110) * T(w110);
    result += loadVec<T, FEATURES>(features, i001) * T(w001);
    result += loadVec<T, FEATURES>(features, i101) * T(w101);
    result += loadVec<T, FEATURES>(features, i011) * T(w011);
    result += loadVec<T, FEATURES>(features, i111) * T(w111);

    return result;
}

[Differentiable]
T[LEVELS * FEATURES] encodeHashGrid<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float3 position, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    T output[LEVELS * FEATURES];
    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingle<T, FEATURES>(features, position, level, levelResolution, hashTableSize);
        [ForceUnroll]
        for (uint i = 0u; i < FEATURES; ++i) output[level * FEATURES + i] = levelFeatures[i];
        levelResolution *= scale;
    }
    return output;
}

[BackwardDerivative(SetCoop_bwd)]
void SetCoop<let N : int, let M : int>(inout CoopVec<half, N> output, half[M] array, int offset)
{
    [ForceUnroll] 
    for(int i = 0; i < M; i++)
    {
        output[i + offset] = array[i];
    }
}

void SetCoop_bwd<let N : int, let M : int>(inout DifferentialPair<CoopVec<half, N>> output, inout DifferentialPair<half[M]> array, int offset)
{
    var dArray = array.d;
    [ForceUnroll] 
    for(int i = 0; i < M; i++)
    {
        dArray[i] = output.d[i + offset];
    }
    array = diffPair(array.p, dArray);
}

[BackwardDerivative(ComposeCoop_bwd)]
CoopVec<half, N1 + N2> ComposeCoop<let N1 : int, let N2 : int>(half[N1] a1, half[N2] a2)
{
    CoopVec<half, N1 + N2> output;
    [ForceUnroll] 
    for(int i = 0; i < N1; i++)
    {
        output[i] = a1[i];
    }
    [ForceUnroll] 
    for(int i = 0; i < N2; i++)
    {
        output[i + N1] = a2[i];
    }
    return output;
}

void ComposeCoop_bwd<let N1 : int, let N2 : int>(inout DifferentialPair<half[N1]> a1, inout DifferentialPair<half[N2]> a2, CoopVec<half, N1 + N2> dOutput)
{
    var dA1 = a1.d;
    [ForceUnroll] 
    for(int i = 0; i < N1; i++)
    {
        dA1[i] = dOutput[i];
    }
    a1 = diffPair(a1.p, dA1);

    var dA2 = a2.d;
    [ForceUnroll] 
    for(int i = 0; i < N2; i++)
    {
        dA2[i] = dOutput[i + N1];
    }
    a2 = diffPair(a2.p, dA2);
}

T[N] encodeSphericalHarmonics<T: __BuiltinFloatingPointType, uint DEGREE, uint N>(no_diff float3 dir)
{
    static_assert(N >= DEGREE * DEGREE, "Not enough output components for the given SH degree");
    T encoding[N];
    let x = dir.x; let y = dir.y; let z = dir.z;
    // Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
    float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
    float x4=x2*x2, y4=y2*y2, z4=z2*z2;
    float x6=x4*x2, y6=y4*y2, z6=z4*z2;

    // SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf
    encoding[0] = (T)(0.28209479177387814f);                          // 1/(2*sqrt(pi))
    if (DEGREE <= 1) { return encoding; }
    encoding[1] = (T)(-0.48860251190291987f*y);                               // -sqrt(3)*y/(2*sqrt(pi))
    encoding[2] = (T)(0.48860251190291987f*z);                                // sqrt(3)*z/(2*sqrt(pi))
    encoding[3] = (T)(-0.48860251190291987f*x);                               // -sqrt(3)*x/(2*sqrt(pi))
    if (DEGREE <= 2) { return encoding; }
    encoding[4] = (T)(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
    encoding[5] = (T)(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
    encoding[6] = (T)(0.94617469575755997f*z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    encoding[7] = (T)(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
    encoding[8] = (T)(0.54627421529603959f*x2 - 0.54627421529603959f*y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if (DEGREE <= 3) { return encoding; }
    encoding[9] = (T)(0.59004358992664352f*y*(-3.0f*x2 + y2));                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    encoding[10] = (T)(2.8906114426405538f*xy*z);                             // sqrt(105)*xy*z/(2*sqrt(pi))
    encoding[11] = (T)(0.45704579946446572f*y*(1.0f - 5.0f*z2));                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
    encoding[12] = (T)(0.3731763325901154f*z*(5.0f*z2 - 3.0f));                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
    encoding[13] = (T)(0.45704579946446572f*x*(1.0f - 5.0f*z2));                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
    encoding[14] = (T)(1.4453057213202769f*z*(x2 - y2));                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
    encoding[15] = (T)(0.59004358992664352f*x*(-x2 + 3.0f*y2));                                // sqrt(70)*x*(-x2 + 3*y２)/(8*sqrt(pi))
    if (DEGREE <= 4) { return encoding; }
    encoding[16] = (T)(2.5033429417967046f*xy*(x2 - y2));                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
    encoding[17] = (T)(1.7701307697799304f*yz*(-3.0f*x2 + y2));                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
    encoding[18] = (T)(0.94617469575756008f*xy*(7.0f*z2 - 1.0f));                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
    encoding[19] = (T)(0.66904654355728921f*yz*(3.0f - 7.0f*z2));                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
    encoding[20] = (T)(-3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f);                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
    encoding[21] = (T)(0.66904654355728921f*xz*(3.0f - 7.0f*z2));                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
    encoding[22] = (T)(0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f));                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
    encoding[23] = (T)(1.7701307697799304f*xz*(-x2 + 3.0f*y2));                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
    encoding[24] = (T)(-3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4);                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    if (DEGREE <= 5) { return encoding; }
    encoding[25] = (T)(0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    encoding[26] = (T)(8.3026492595241645f*xy*z*(x2 - y2));                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
    encoding[27] = (T)(-0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    encoding[28] = (T)(4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f));                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
    encoding[29] = (T)(0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f));                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    encoding[30] = (T)(0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f));                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
    encoding[31] = (T)(0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f));                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    encoding[32] = (T)(2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f));                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
    encoding[33] = (T)(-0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
    encoding[34] = (T)(2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    encoding[35] = (T)(0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    if (DEGREE <= 6) { return encoding; }
    encoding[36] = (T)(1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    encoding[37] = (T)(2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    encoding[38] = (T)(2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f));                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    encoding[39] = (T)(-0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    encoding[40] = (T)(0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f));                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    encoding[41] = (T)(0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    encoding[42] = (T)(6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f);                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
    encoding[43] = (T)(0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    encoding[44] = (T)(0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f));                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
    encoding[45] = (T)(-0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
    encoding[46] = (T)(0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4));                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    encoding[47] = (T)(2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    encoding[48] = (T)(10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6);                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    if (DEGREE <= 7) { return encoding; }
    encoding[49] = (T)(0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6));                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
    encoding[50] = (T)(5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    encoding[51] = (T)(-0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4));                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
    encoding[52] = (T)(4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f));                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    encoding[53] = (T)(-0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    encoding[54] = (T)(0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f));                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    encoding[55] = (T)(0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    encoding[56] = (T)(0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f));                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
    encoding[57] = (T)(0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    encoding[58] = (T)(0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f));                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
    encoding[59] = (T)(-0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    encoding[60] = (T)(1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    encoding[61] = (T)(-0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4));                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
    encoding[62] = (T)(2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6));                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    encoding[63] = (T)(0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6));                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
    return encoding;
}