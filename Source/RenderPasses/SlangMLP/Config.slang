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

using rtxns;
using mlp;
#endif

namespace MLPConfig
{
    constexpr uint align4(uint x)
    {
        return (x + 3u) & ~3u;
    }

    static constexpr const uint kInputSize = 32u;
    static constexpr const uint kHiddenSize = 64u;
    static constexpr const uint kOutputSize = 3u;
    static constexpr const uint kHiddenLayers = 4u;
    static constexpr const uint kTransitions = kHiddenLayers + 1u;

    constexpr uint maxu(uint a, uint b)
    {
        return (a > b) ? a : b;
    }

    static constexpr const uint kHashLevels = 16u;
    static constexpr const uint kHashFeaturesPerLevel = 2u;
    static constexpr const uint kHashLog2TableSize = 14u;
    static constexpr const uint kHashTableSize = 1u << kHashLog2TableSize;
    static constexpr const uint kHashBaseResolution = 16u;
    static constexpr const float kHashPerLevelScale = 1.5;

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
    static constexpr uint kEncodingParamElementCount = kHashLevels * kHashTableSize * kHashFeaturesPerLevel;
    static constexpr uint kOptimizeElementCount = maxu(kParamElementCount, kEncodingParamElementCount);

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
    static constexpr const half kReluLeak = 0.01;

    [Differentiable]
    CoopVec<half, kOutputSize> forward(FeatureBuffer features, MatrixBiasBuffer weights, no_diff float2 uv) {
        // let input = EncodeFrequencyN<half, 2, 8>(CoopVecFromVector<half>(uv));
        let input = encodeHashGrid<half, kHashLevels, kHashFeaturesPerLevel>(
            features,
            uv,
            kHashTableSize,
            float(kHashBaseResolution),
            kHashPerLevelScale
        );

        var params = leakyReLU(LinearOp<half, kHiddenSize>(input, weights, layerOffsets[0], matrixLayout, componentType), kReluLeak);
        [ForceUnroll]
        for(int i = 1; i < kHiddenLayers; ++i)
            params = leakyReLU(LinearOp<half, kHiddenSize>(params, weights, layerOffsets[i], matrixLayout, componentType), kReluLeak);

        return LinearOp<half, kOutputSize>(params, weights, layerOffsets[kHiddenLayers], matrixLayout, componentType);
    }

    CoopVec<half, kOutputSize> forward(ByteAddressBuffer features, ByteAddressBuffer weights, no_diff float2 uv) {
        return forward(FeatureBuffer(features), MatrixBiasBuffer(weights), uv);
    }
#endif
};