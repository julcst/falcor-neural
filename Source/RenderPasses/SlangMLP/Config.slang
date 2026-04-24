#pragma once

#include "Utils/HostDeviceShared.slangh"

#ifndef HOST_CODE
import RTXNS.MLP;
#endif

namespace MLPConfig
{
    constexpr uint align4(uint x)
    {
        return (x + 3u) & ~3u;
    }

    constexpr uint maxu(uint a, uint b)
    {
        return (a > b) ? a : b;
    }

    static constexpr uint kHashLevels = 16u;
    static constexpr uint kHashFeaturesPerLevel = 2u;
    static constexpr uint kHashLog2TableSize = 12u;
    static constexpr uint kHashTableSize = 1u << kHashLog2TableSize;
    static constexpr uint kHashBaseResolution = 16u;
    static constexpr float kHashPerLevelScale = 1.5;

    static constexpr uint kInputSize = kHashLevels * kHashFeaturesPerLevel;
    static constexpr uint kHiddenSize = 64u;
    static constexpr uint kOutputSize = 3u;
    static constexpr uint kHiddenLayers = 4u;
    static constexpr uint kTransitions = kHiddenLayers + 1u;

    static constexpr float kReluLeak = 0.01;
    static constexpr float kOptimizerLossScale = 1.0;

    static constexpr float kAdamBeta1 = 0.9;
    static constexpr float kAdamBeta2 = 0.999;
    static constexpr float kAdamEpsilon = 1e-8;

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

    static constexpr uint kParamElementCount = getParamCount();
    static constexpr uint kEncodingParamElementCount = kHashLevels * kHashTableSize * kHashFeaturesPerLevel;
    static constexpr uint kOptimizeElementCount = maxu(kParamElementCount, kEncodingParamElementCount);

    static constexpr uint kParamBufferSize = align4(kParamElementCount * sizeof(float16_t));
    static constexpr uint kMomentsBufferSize = kParamElementCount * sizeof(float);
    static constexpr uint kEncodingParamBufferSize = align4(kEncodingParamElementCount * sizeof(float16_t));
    static constexpr uint kEncodingGradBufferSize = align4(kEncodingParamElementCount * sizeof(float));
    static constexpr uint kEncodingMomentsBufferSize = align4(kEncodingParamElementCount * sizeof(float));

#ifndef HOST_CODE
    constexpr Tuple<uint[kTransitions], uint[kTransitions]> getLayerOffsets()
    {
        uint weightOffsets[kTransitions];
        uint biasOffsets[kTransitions];
        uint byteOffset = 0;

        [ForceUnroll]
        for (uint layer = 0u; layer < kTransitions; ++layer)
        {
            weightOffsets[layer] = byteOffset;
            byteOffset = align4(byteOffset + 2u * layerWeightCount(layer));

            biasOffsets[layer] = byteOffset;
            byteOffset = align4(byteOffset + 2u * layerBiasCount(layer));
        }
        return {weightOffsets, biasOffsets};
    }

    static constexpr uint[kTransitions] weightOffsets = getLayerOffsets()._0;
    static constexpr uint[kTransitions] biasOffsets = getLayerOffsets()._1;

    typedef rtxns::mlp::TrainingMLP<
        half,
        kHiddenLayers,
        kInputSize,
        kHiddenSize,
        kOutputSize,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16
    > TrainingMLP;

    typedef rtxns::mlp::InferenceMLP<
        half,
        kHiddenLayers,
        kInputSize,
        kHiddenSize,
        kOutputSize,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16
    > InferenceMLP;
#endif
};