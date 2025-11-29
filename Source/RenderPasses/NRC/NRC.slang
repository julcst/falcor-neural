#pragma once

#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"

#ifndef HOST_CODE
import RenderPasses.TraceQueries.Query;
#endif

BEGIN_NAMESPACE_FALCOR

struct NRCInput {
    float3 position;
    float3 wo;
    float3 wn;
    float roughness;
    float3 diffuse;
    float3 specular;
#ifndef HOST_CODE
    __init(Query q) {
        position = q.position;
        wo = q.viewDir;
        wn = q.normal;
        roughness = q.roughness;
        diffuse = q.diffuse;
        specular = q.specular;
    }
#endif
};

static const uint NRC_INPUT_SIZE = sizeof(NRCInput) / sizeof(float);

struct NRCOutput {
    float3 radiance;
};

static const uint NRC_OUTPUT_SIZE = sizeof(NRCOutput) / sizeof(float);

END_NAMESPACE_FALCOR