#pragma once

#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/MathConstants.slangh"

#ifndef HOST_CODE
import Scene.HitInfo;
#endif

BEGIN_NAMESPACE_FALCOR

// TODO: Padding?
struct Query
{
    uint4 hit;

    float3 position;
    uint revision; // To identify changes fast?

    float3 viewDir; // TODO: Octaeder
    // bool isValid;

    float3 emission; // TODO: Pack

    float3 normal; // TODO: Octaeder

    float3 throughput; // TODO: 16 unorm

#ifndef HOST_CODE
    HitInfo getHitInfo() {
        return unpackHitInfo(hit);
    }

    bool isValid() {
        // NOTE: The isValid() function needs hit defines to be set
        // return getHitInfo().isValid();
        // Thus we use this simpler direct check, which works because type is encoded in the first component
        // hit.x = 0 => Type = None but not the other way around
        return hit.x != 0u;
    }
#endif
};

END_NAMESPACE_FALCOR
