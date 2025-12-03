#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Utils.Math.MathHelpers;
import Utils.Math.AABB;
import Utils.Timing.GpuTimer;
import Query;
import RenderPasses.NRC.NRC;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

RWStructuredBuffer<Query> gPhotonQueries;
RWStructuredBuffer<NRCInput> gNRCInput;

cbuffer CB
{
    float3 gSceneMin;
    uint gFrameIndex;
    float3 gSceneScale;
    uint gRevisionCounter;
    uint2 gFrameDim;
}

static const uint kMaxDepth = 16;

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

bool shouldTerminate(BSDFProperties props)
{
    // Terminate if we hit an emissive surface
    // so we do not have to track a plus onto radiance predictions
    // only the throughput
    if (luminance(props.emission) > 0.f) return true;

    // Terminate if the surface is sufficiently rough
    if (props.roughness > 0.1f) return true;

    // Otherwise continue tracing
    return false;
}

void storeQuery(uint index, PackedHitInfo hit, ShadingData sd, BSDFProperties props, float3 thp)
{
    Query q = {};
    q.hit = hit;
    q.emission = props.emission;
    q.position = sd.posW;
    q.viewDir = sd.V;
    q.normal = props.guideNormal;
    q.throughput = thp;
    q.revision = gRevisionCounter;
    gPhotonQueries[index] = q;

#if OUTPUT_NRC_INPUT
    NRCInput nrc = {};
    nrc.position = (sd.posW - gSceneMin) * gSceneScale; // Normalize to [0,1]³ for hash grid
    nrc.wo = 0.5 * (1 + sd.V);
    nrc.wn = 0.5 * (1 + props.guideNormal);
    nrc.roughness = props.roughness;
    nrc.diffuse = props.diffuseReflectionAlbedo;
    nrc.specular = props.specularReflectionAlbedo;
    gNRCInput[index] = nrc;
#endif
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint index = pixel.x + pixel.y * gFrameDim.x;

    SampleGenerator sg = SampleGenerator(pixel, gFrameIndex);

    // Generate primary ray (pinhole for stability)
    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim);
    ShadingData sd = {};
    BSDFProperties props = {};
    float3 thpAtQuery = float3(1.0f);
    float3 currentThp = float3(1.0f);
    PackedHitInfo hitAtQuery = {};

    for (uint i = 0; i < kMaxDepth; ++i)
    {
        RayPayload payload = {};
        TraceRayScene(0, 0xff, 0 /* hitIdx */, 0 /* missIdx */, ray.toRayDesc(), payload);
        HitInfo hit = HitInfo(payload.hit);
        if (!hit.isValid()) {
            break; // Store previous query if we miss
        }

        // Get next query
        hitAtQuery = payload.hit;
        thpAtQuery = currentThp;
        let th = hit.getTriangleHit();
        let materialID = gScene.getMaterialID(th.instanceID);
        let v = gScene.getVertexData(th);
        sd = gScene.materials.prepareShadingData(v, materialID, -ray.dir);
        let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0), (uint)MaterialInstanceHints::AdjustShadingNormal);
        props = mi.getProperties(sd);

        if (shouldTerminate(props)) {
            break; // Store current query
        } else {
            BSDFSample result = {};
            let valid = mi.sample(sd, sg, result, true /* importanceSample */);
            if (!valid) { // TODO: This sometimes fails, handle this better
                //props.emission = float3(0, 1, 0);
                break; // Store current query if sampling failed
            }
            currentThp *= result.weight;
            ray.dir = result.wo;
            ray.origin = computeRayOrigin(sd.posW, dot(ray.dir, sd.faceN) >= 0.0f ? sd.faceN : -sd.faceN);
            ray.tMin = 0.0f;
            ray.tMax = FLT_MAX;
        }
    }

    storeQuery(index, hitAtQuery, sd, props, thpAtQuery);
}
