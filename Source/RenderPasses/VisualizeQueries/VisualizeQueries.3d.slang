// Let's try to include first.
#include "RenderPasses/TraceQueries/Query.slang"
#include "RenderPasses/AccumulatePhotonsRTX/Structs.slang"

import Scene.Camera.Camera;

StructuredBuffer<Query> gQueries;
StructuredBuffer<QueryState> gQueryStates;
StructuredBuffer<float3> gQueryColors;

#ifndef USE_QUERY_STATE
#define USE_QUERY_STATE 0
#endif

#ifndef USE_COLOR_BUFFER
#define USE_COLOR_BUFFER 0
#endif

cbuffer CB
{
    Camera gCamera;
    float gConstantRadius;
    uint gMode;
    uint gGlobalPhotonCount;
}

struct VSIn
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOut
{
    float4 posH : SV_POSITION;
    float3 color : COLOR;
};

// Helper to build a rotation matrix that aligns 'from' vector to 'to' vector
float3x3 makeRotation(float3 from, float3 to)
{
    float3 v = cross(from, to);
    float c = dot(from, to);
    float k = 1.0f / (1.0f + c);

    return float3x3(
        v.x * v.x * k + c,     v.y * v.x * k - v.z,   v.z * v.x * k + v.y,
        v.x * v.y * k + v.z,   v.y * v.y * k + c,     v.z * v.y * k - v.x,
        v.x * v.z * k - v.y,   v.y * v.z * k + v.x,   v.z * v.z * k + c
    );
}

VSOut vsMain(VSIn input, uint instanceID : SV_InstanceID)
{
    VSOut output;
    
    Query q = gQueries[instanceID];
    
    if (!q.isValid())
    {
        output.posH = float4(0, 0, 0, 0); // Cull
        output.color = float3(0);
        return output;
    }

    float radius = gConstantRadius;
    float3 flux = float3(0);

#if USE_COLOR_BUFFER
    float3 colorBuffer = gQueryColors[instanceID];
#else
    float3 colorBuffer = float3(-1.0f); // Invalid
#endif

#if USE_QUERY_STATE
    QueryState qs = gQueryStates[instanceID];
    float3 causticFlux = qs.getCausticRadiance(gGlobalPhotonCount);
    float3 globalFlux = qs.getGlobalRadiance(gGlobalPhotonCount);
    float maxRadius = max(qs.causticRadius, qs.globalRadius);
#else
    float3 causticFlux = float3(0);
    float3 globalFlux = float3(0);
    float maxRadius = gConstantRadius;
#endif

    // Mode selection
    switch (gMode)
    {
    case 0: // Caustic radiance + radius
#if USE_QUERY_STATE
        radius = qs.causticRadius;
        flux = causticFlux;
#else
        radius = gConstantRadius;
        flux = colorBuffer;
#endif
        break;
    case 1: // Global radiance + radius
#if USE_QUERY_STATE
        radius = qs.globalRadius;
        flux = globalFlux;
#else
        radius = gConstantRadius;
        flux = colorBuffer;
#endif
        break;
    case 2: // Max radius + combined radiance
#if USE_QUERY_STATE
        radius = maxRadius;
        flux = causticFlux + globalFlux;
#else
        radius = gConstantRadius;
        flux = colorBuffer;
#endif
        break;
    default: // Constant radius + provided color or combined radiance fallback
        radius = gConstantRadius;
#if USE_COLOR_BUFFER
        flux = colorBuffer;
#elif USE_QUERY_STATE
        flux = causticFlux + globalFlux;
#else
        flux = float3(-1.0f); // Invalid
#endif
        break;
    }
    
    // Disk is originally in XZ plane with normal (0, 1, 0)
    float3 diskNormal = float3(0, 1, 0);
    float3 targetNormal = q.getNormal();
    
    // Rotate disk to align with query normal
    // If normals are parallel, no rotation needed (or 180 flip)
    float3 posW = input.pos * radius;
    
    // Simple alignment: if target is not (0,1,0), rotate
    if (abs(dot(diskNormal, targetNormal)) < 0.999f)
    {
        float3x3 rot = makeRotation(diskNormal, targetNormal);
        posW = mul(rot, posW);
    }
    else if (dot(diskNormal, targetNormal) < -0.999f)
    {
        posW.y = -posW.y; // Flip
    }

    posW += q.position;
    
    output.posH = mul(gCamera.getViewProj(), float4(posW, 1.0f));
    output.color = flux;
    
    return output;
}

float4 psMain(VSOut input) : SV_TARGET
{
    return float4(input.color, 0.5f);
}
