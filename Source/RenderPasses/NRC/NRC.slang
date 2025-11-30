#pragma once

#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"

#ifndef HOST_CODE
import RenderPasses.TraceQueries.Query;
#endif

BEGIN_NAMESPACE_FALCOR

struct NRCInput {
    float3 position;
    float roughness;
    float3 wo;
    float3 wn;
    float3 diffuse;
    float3 specular;

#ifndef HOST_CODE
    __init() {
        position = {};
        wo = {};
        wn = {};
        roughness = {};
        diffuse = {};
        specular = {};
    }
#endif
};

static const uint NRC_INPUT_SIZE = sizeof(NRCInput) / sizeof(float);

struct NRCOutput {
    float3 radiance;
};

static const uint NRC_OUTPUT_SIZE = sizeof(NRCOutput) / sizeof(float);

END_NAMESPACE_FALCOR