#pragma once

#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/MathConstants.slangh"

#ifndef HOST_CODE
import Scene.HitInfo;
#endif

BEGIN_NAMESPACE_FALCOR

struct Query
{
    uint4 hit;

    float3 position;
    // TODO: Padding?

    float3 viewDir;

    float3 emission;

    float3 normal;

#ifndef HOST_CODE
    HitInfo getHitInfo() {
        return HitInfo(hit);
    }

    // NOTE: This function needs hit defines to be set
    bool isValid() {
        return getHitInfo().isValid();
    }
#endif
};

END_NAMESPACE_FALCOR
