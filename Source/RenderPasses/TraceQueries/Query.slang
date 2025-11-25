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
    float _pad0;

    float3 viewDir;
    float _pad1;

    float3 emission;
    float _pad2;

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
