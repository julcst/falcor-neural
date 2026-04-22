#pragma once

#include "Utils/HostDeviceShared.slangh"

namespace SlangMLPConfig
{
    constexpr uint align4(uint x)
    {
        return (x + 3u) & ~3u;
    }

    static constexpr uint kInputSize = 32u;
    static constexpr uint kHiddenSize = 64u;
    static constexpr uint kOutputSize = 3u;
    static constexpr uint kHiddenLayers = 4u;
    static constexpr uint kTransitions = kHiddenLayers + 1u;

    static constexpr uint kTrainThreads = 64u;
    static constexpr uint kOptimizeThreads = 64u;

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

    static constexpr uint kParamBufferSize = align4(kParamElementCount * sizeof(float16_t));
    static constexpr uint kMomentsBufferSize = kParamElementCount * sizeof(float);

#ifndef HOST_CODE
    void computeLayerOffsets(out uint weightOffsets[kTransitions], out uint biasOffsets[kTransitions])
    {
        uint byteOffset = 0;

        [ForceUnroll]
        for (uint layer = 0u; layer < kTransitions; ++layer)
        {
            weightOffsets[layer] = byteOffset;
            byteOffset = align4(byteOffset + 2u * layerWeightCount(layer));

            biasOffsets[layer] = byteOffset;
            byteOffset = align4(byteOffset + 2u * layerBiasCount(layer));
        }
    }
#endif
};