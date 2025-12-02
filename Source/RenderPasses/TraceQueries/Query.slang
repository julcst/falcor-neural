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
    // uint hash; // To identify changes fast?

    float3 viewDir;
    // bool isValid;

    float3 emission;

    float3 normal;

    float3 throughput;

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
