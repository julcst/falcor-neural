#include "Scene/SceneDefines.slangh"

import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Utils.Math.MathHelpers;
import Utils.Math.AABB;
import Utils.Timing.GpuTimer;
import Query;

RWStructuredBuffer<Query> gPhotonQueries;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
}

struct RayPayload { PackedHitInfo hit; };

[shader("miss")]
void miss(inout RayPayload p: SV_RayPayload)
{
    p.hit = makeInvalidHit().data;
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
    p.hit = th.pack();
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint index = pixel.x + pixel.y * gFrameDim.x;

    // Generate primary ray (pinhole for stability)
    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim);

    RayPayload payload;
    TraceRayScene(0, 0xff, 0 /* hitIdx */, 0 /* missIdx */, ray.toRayDesc(), payload);

    Query q = {};
    q.hit = payload.hit;

    HitInfo hit = HitInfo(payload.hit);
    if (hit.isValid())
    {
        TriangleHit th = hit.getTriangleHit();
        uint materialID = gScene.getMaterialID(th.instanceID);
        VertexData v = gScene.getVertexData(th);
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -ray.dir);
        let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0), (uint)MaterialInstanceHints::AdjustShadingNormal);
        let props = mi.getProperties(sd);

        q.emission = props.emission;
        q.position = sd.posW;
        q.viewDir = -ray.dir;
    }

    gPhotonQueries[index] = q;
}
