// Let's try to include first.
#include "RenderPasses/TraceQueries/Query.slang"
#include "RenderPasses/AccumulatePhotonsRTX/Structs.slang"

import Scene.Camera.Camera;

StructuredBuffer<Query> gQueries;
StructuredBuffer<QueryState> gQueryStates;

cbuffer CB
{
    Camera gCamera;
    float gRadiusScale;
    bool gVisualizeCaustics;
    uint32_t gGlobalPhotonCount;
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
    QueryState qs = gQueryStates[instanceID];
    
    if (!q.isValid())
    {
        output.posH = float4(0, 0, 0, 0); // Cull
        output.color = float3(0);
        return output;
    }

    float radius;
    float3 flux;

    if (gVisualizeCaustics)
    {
        radius = qs.causticRadius * gRadiusScale;
        flux = qs.getCausticRadiance(gGlobalPhotonCount);
    }
    else
    {
        radius = qs.globalRadius * gRadiusScale;
        flux = qs.getRadiance(gGlobalPhotonCount);
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
    return float4(input.color, 1.0f);
}
