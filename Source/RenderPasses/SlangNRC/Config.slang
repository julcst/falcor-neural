#pragma once

// #include "Utils/HostDeviceShared.slangh"

#ifndef HOST_CODE
import RTXNS.MLP;
import RTXNS.LinearOps;
import RTXNS.Utils;
import RTXNS.Activation;
import RTXNS.CooperativeVectorAutoDiff;
import RTXNS.CooperativeVectorFunctions;
import RTXNS.CooperativeVectorDerivatives;
import Encodings;
import NRCCompat;

using rtxns;
using mlp;
#endif

namespace MLPConfig
{
    constexpr uint align4(uint x)
    {
        return (x + 3u) & ~3u;
    }

    static constexpr const uint kInputSize = 47u;
    static constexpr const uint kHiddenSize = 64u;
    static constexpr const uint kOutputSize = 3u;
    static constexpr const uint kHiddenLayers = 2u;
    static constexpr const uint kTransitions = kHiddenLayers + 1u;

    static constexpr const uint kHashLevels = 16u;
    static constexpr const uint kHashFeaturesPerLevel = 2u;
    static constexpr const uint kHashLog2TableSize = 19u;
    static constexpr const uint kHashTableSize = 1u << kHashLog2TableSize;
    static constexpr const uint kHashBaseResolution = 16u;
    static constexpr const float kHashPerLevelScale = 2.0;

    static constexpr const float kOptimizerLossScale = 1.0;

    static constexpr const float kAdamBeta1 = 0.9;
    static constexpr const float kAdamBeta2 = 0.999;
    static constexpr const float kAdamEpsilon = 1e-8;

    constexpr uint layerInputSize(uint layerIndex)
    {
        return (layerIndex == 0u) ? kInputSize : kHiddenSize;
    }

    constexpr uint layerOutputSize(uint layerIndex)
    {
        return (layerIndex == (kTransitions - 1u)) ? kOutputSize : kHiddenSize;
    }

    constexpr uint layerWeightCount(uint layerIndex)
    {
        return layerInputSize(layerIndex) * layerOutputSize(layerIndex);
    }

    constexpr uint layerBiasCount(uint layerIndex)
    {
        return layerOutputSize(layerIndex);
    }

    constexpr uint getParamCount()
    {
        uint count = 0u;
        for (uint layer = 0u; layer < kTransitions; ++layer)
            count += layerWeightCount(layer) + layerBiasCount(layer);
        return count;
    }

    static constexpr const uint kParamElementCount = getParamCount();
    static constexpr const uint kEncodingParamElementCount = kHashLevels * kHashTableSize * kHashFeaturesPerLevel;

    static constexpr uint kParamBufferSize = align4(kParamElementCount * sizeof(float16_t));
    static constexpr uint kMomentsBufferSize = kParamElementCount * sizeof(float);
    static constexpr uint kEncodingParamBufferSize = align4(kEncodingParamElementCount * sizeof(float16_t));
    static constexpr uint kEncodingGradBufferSize = align4(kEncodingParamElementCount * sizeof(float));
    static constexpr uint kEncodingMomentsBufferSize = align4(kEncodingParamElementCount * sizeof(float));

#ifndef HOST_CODE
    constexpr uint2[kTransitions] getLayerOffsetsUint2()
    {
        uint2 layerOffsets[kTransitions];
        uint byteOffset = 0;

        [ForceUnroll]
        for (uint layer = 0u; layer < kTransitions; ++layer)
        {
            let weightOffset = byteOffset;
            byteOffset = align4(byteOffset + 2u * layerWeightCount(layer));
            let biasOffset = byteOffset;
            byteOffset = align4(byteOffset + 2u * layerBiasCount(layer));
            layerOffsets[layer] = uint2(weightOffset, biasOffset);
        }
        return layerOffsets;
    }

    static constexpr const uint2 layerOffsets[kTransitions] = getLayerOffsetsUint2();
    static constexpr const CoopVecMatrixLayout matrixLayout = CoopVecMatrixLayout::TrainingOptimal;
    static constexpr const CoopVecComponentType componentType = CoopVecComponentType::Float16;
    static constexpr const half kReluLeak = 0.01h;

    [BackwardDerivative(ComposeEnc_bwd)]
    CoopVec<half, N1 + N2 + N3 + N4> ComposeEnc<int N1, int N2, int N3, int N4>(half[N1] a1, half[N2] a2, vector<half, N3> a3, vector<half, N4> a4)
    {
        CoopVec<half, N1 + N2 + N3 + N4> output;
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
        [ForceUnroll] 
        for(int i = 0; i < N3; i++)
        {
            output[i + N1 + N2] = a3[i];
        }
        [ForceUnroll] 
        for(int i = 0; i < N4; i++)
        {
            output[i + N1 + N2 + N3] = a4[i];
        }
        return output;
    }

    void ComposeEnc_bwd<int N1, int N2, int N3, int N4>(inout DifferentialPair<half[N1]> a1, inout DifferentialPair<half[N2]> a2, inout DifferentialPair<vector<half, N3>> a3, inout DifferentialPair<vector<half, N4>> a4, CoopVec<half, N1 + N2 + N3 + N4> dOutput)
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

        var dA3 = a3.d;
        [ForceUnroll] 
        for(int i = 0; i < N3; i++)
        {
            dA3[i] = dOutput[i + N1 + N2];
        }
        a3 = diffPair(a3.p, dA3);

        var dA4 = a4.d;
        [ForceUnroll] 
        for(int i = 0; i < N4; i++)
        {
            dA4[i] = dOutput[i + N1 + N2 + N3];
        }
        a4 = diffPair(a4.p, dA4);
    }

    [Differentiable]
    CoopVec<half, kOutputSize> forward(FeatureBuffer features, MatrixBiasBuffer weights, no_diff NRCInput in) {
        let posEnc = encodeHashGrid<half, kHashLevels, kHashFeaturesPerLevel>(
            features,
            in.position,
            kHashTableSize,
            float(kHashBaseResolution),
            kHashPerLevelScale
        );

        let shEnc = encodeSphericalHarmonics<half, 3, 9>(
            in.wo
        );

        let input = ComposeEnc(posEnc, shEnc, half3(in.diffuse), half3(in.specular));
        static_assert(kInputSize == input.getCount(), "Input size does not match encoding output");

        var params = leakyReLU(LinearOp<half, kHiddenSize>(input, weights, layerOffsets[0], matrixLayout, componentType), kReluLeak);
        [ForceUnroll]
        for(int i = 1; i < kHiddenLayers; ++i)
            params = leakyReLU(LinearOp<half, kHiddenSize>(params, weights, layerOffsets[i], matrixLayout, componentType), kReluLeak);

        return LinearOp<half, kOutputSize>(params, weights, layerOffsets[kHiddenLayers], matrixLayout, componentType);
    }

    CoopVec<half, kOutputSize> forward(ByteAddressBuffer features, ByteAddressBuffer weights, no_diff NRCInput in) {
        return forward(FeatureBuffer(features), MatrixBiasBuffer(weights), in);
    }
#endif
};