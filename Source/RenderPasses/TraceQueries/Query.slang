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
    float radius;

    float3 viewDir;
    uint pixelIndex;

    float3 emission;

#ifndef HOST_CODE
    HitInfo getHitInfo() {
        return HitInfo(hit);
    }

    bool isValid() {
        return getHitInfo().isValid();
    }
#endif
};

END_NAMESPACE_FALCOR
