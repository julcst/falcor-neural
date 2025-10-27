#include "Scene/SceneDefines.slangh"

import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Utils.Math.MathHelpers;
import Utils.Timing.GpuTimer;
import PhotonQuery;

struct RtAABB { float3 min; float3 max; };

RWStructuredBuffer<PhotonQuery> gPhotonQueries;
RWStructuredBuffer<RtAABB> gQueryAABBs;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    float gQueryRadius;
}

struct RayPayload { PackedHitInfo hit; };

[shader("miss")]
void miss(inout RayPayload p: SV_RayPayload)
{
    p.hit = PackedHitInfo(0,0,0,0);
}

[shader("anyhit")]
void anyHit(inout RayPayload p: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayPayload p: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
    TriangleHit th; th.instanceID = getGeometryInstanceID(); th.primitiveIndex = PrimitiveIndex(); th.barycentrics = attribs.barycentrics;
    p.hit = HitInfo(th).data;
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint index = pixel.x + pixel.y * gFrameDim.x;

    // Generate primary ray (pinhole for stability)
    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim);

    RayPayload payload; payload.hit = PackedHitInfo(0,0,0,0);
    TraceRayScene(0, 0xff, 0 /* hitIdx */, 0 /* missIdx */, ray.toRayDesc(), payload);

    PhotonQuery q = {};
    q.pixelIndex = index;
    q.radius = gQueryRadius;
    q.valid = 0;

    HitInfo hit = HitInfo(payload.hit);
    if (hit.isValid())
    {
        TriangleHit th = hit.getTriangleHit();
        uint materialID = gScene.getMaterialID(th.instanceID);
        VertexData v = gScene.getVertexData(th);
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -ray.dir);
        let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0), (uint)MaterialInstanceHints::AdjustShadingNormal);
        let props = mi.getProperties(sd);

        q.position = sd.posW;
        q.normal = normalize(sd.getOrientedFaceNormal());
        q.viewDir = -ray.dir;
        q.diffuseReflectance = props.diffuseReflectionAlbedo;
        q.valid = 1;
    }

    gPhotonQueries[index] = q;

    // Write AABB for this query directly here to avoid a separate compute pass
    RtAABB aabb;
    if (q.valid != 0)
    {
        float3 r = float3(q.radius);
        aabb.min = q.position - r;
        aabb.max = q.position + r;
    }
    else
    {
        aabb.min = float3(1e30f, 1e30f, 1e30f);
        aabb.max = float3(-1e30f, -1e30f, -1e30f);
    }
    gQueryAABBs[index] = aabb;
}
