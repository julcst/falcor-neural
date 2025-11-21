#pragma once

#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct DebugCounters
{
    uint IntersectorCalls;
    uint RaygenCalls;
    uint Accumulations;
};

END_NAMESPACE_FALCOR