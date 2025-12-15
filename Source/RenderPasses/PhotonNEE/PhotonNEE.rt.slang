#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.PhotonSampler;
import Utils.Color.ColorHelpers;

#include "../TraceQueries/Query.slang"

cbuffer CB
{
    uint gFrameCount;
    uint gQueryCount;
    uint gMaxBounces;
    float gRussianRouletteWeight; // in (0, inf), higher means more likely to survive
}

ParameterBlock<PhotonSampler> gLights;
StructuredBuffer<Query> gQueries;
RWStructuredBuffer<float3> gOutput;

struct ShadowRayData
{
    bool visible;
};

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

bool traceShadowRay(float3 posA, float3 normalA, float3 posB, float3 normalB)
{
    let a = computeRayOrigin(posA, normalA * sign(dot(posB - posA, normalA)));
    let b = computeRayOrigin(posB, normalB * sign(dot(posA - posB, normalB)));
    var dir = b - a;
    let dist = length(dir);
    dir /= dist;
    return traceShadowRay(a, dir, dist);
}

struct RayPayload
{
    PackedHitInfo packedHitInfo; //< Hit Info
    float3 photonFlux; float _pad0;
    float3 wi; float _pad1;

    /** Create ray payload with default parameters.
     */
    __init()
    {
        this.packedHitInfo = {};
    }
}

//Add empty hit info for miss
[shader("miss")]
void miss(inout RayPayload rayData: SV_RayPayload)
{
    HitInfo hitInfo = {};
    rayData.packedHitInfo = hitInfo.data; // Invalid hit
}

// For alpha test
[shader("anyhit")]
void anyHit(inout RayPayload rayData: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//Add triangle hit to the payload
[shader("closesthit")]
void closestHit(inout RayPayload rayData: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes) {
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    rayData.packedHitInfo = HitInfo(triangleHit).data;
}

void collectRadianceFromPhoton(inout SampleGenerator sg, ShadingData sdQuery, IMaterialInstance miQuery, ShadingData sdPhoton, IMaterialInstance miPhoton, float3 photonFlux, inout float3 outgoingRadianceAtQuery)
{
    float3 queryPos = sdQuery.posW;
    float3 queryNormal = sdQuery.getOrientedFaceNormal();

    float3 photonPos = sdPhoton.posW;
    float3 photonNormal = sdPhoton.getOrientedFaceNormal();

    float3 toPhoton = photonPos - queryPos;
    float dist2 = dot(toPhoton, toPhoton);
    if (dist2 < 1e-4f) return; // Avoid NaN
    float dist = sqrt(dist2);
    float3 wi = toPhoton / dist;

    // Check if the photon can contribute to the query (i.e., not blocked)
    if (traceShadowRay(photonPos, photonNormal, queryPos, queryNormal))
    {
        outgoingRadianceAtQuery += photonFlux * miPhoton.eval(sdPhoton, -wi, sg) * miQuery.eval(sdQuery, wi, sg) / dist2;
    }
}

// NOTE: This could be enhanced with specular manifold sampling for caustics
void collectRadianceFromLightSource(inout SampleGenerator sg, ShadingData sdQuery, IMaterialInstance miQuery, float3 lightPos, float3 lightNormal, float3 lightEmission, inout float3 outgoingRadianceAtQuery)
{
    float3 queryPos = sdQuery.posW;
    float3 queryNormal = sdQuery.getOrientedFaceNormal();

    float3 toLight = lightPos - queryPos;
    float dist2 = dot(toLight, toLight);
    if (dist2 < 1e-4f) return; // Avoid NaN
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    let lambert = abs(dot(lightNormal, wi)) * M_1_PI;
    if (lambert < 1e-4f) return;

    // Check if the photon can contribute to the query (i.e., not blocked)
    // TODO: Handle environment lights correctly
    if (traceShadowRay(lightPos, lightNormal, queryPos, queryNormal))
    {
        outgoingRadianceAtQuery += lightEmission * lambert * miQuery.eval(sdQuery, wi, sg) / dist2;
    }
}

public ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir); // TODO: Inconsistent minus
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return sd;
}

[shader("raygeneration")]
void rayGen()
{
    let idx = DispatchRaysIndex().x;
    if (idx >= gQueryCount) return;

    let query = gQueries[idx];
    if (!query.isValid()) return;

    SampleGenerator sg = SampleGenerator(idx, gFrameCount);
    
    PhotonLightSample ls;
    if (!gLights.generateLightSample(sg, ls)) return;

    float3 outgoingRadianceAtQuery = float3(0.0f);

    RayDesc ray = ls.getRay().toRayDesc();
    float3 photonFlux = ls.flux;

    let sdQuery = loadShadingData(query.getHitInfo(), query.position, -query.getViewDir());
    let miQuery = gScene.materials.getMaterialInstance(sdQuery, ExplicitLodTextureSampler(0));

    collectRadianceFromLightSource(sg, sdQuery, miQuery, ls.posW, ls.normal, ls.flux, outgoingRadianceAtQuery);

    // Main Raytrace Loop
    for (uint i = 0; i < gMaxBounces; i++) {
        var rayPayload = RayPayload();
        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        if (!hit.isValid()) return; // Stop if photon leaves the scene

        // Get shading infos and material properties
        let sdPhoton = loadShadingData(hit, ray.Origin, ray.Direction);
        let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));

        collectRadianceFromPhoton(sg, sdQuery, miQuery, sdPhoton, miPhoton, photonFlux, outgoingRadianceAtQuery);

        // Sample Photon bounce for next iteration
        BSDFSample bsdfSample = {};
        // TODO: Is this reciprocal for transmission?
        let validSample = miPhoton.sample(sdPhoton, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!validSample)
            return;
        ray.Origin = sdPhoton.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission)); // false is for transmissions
        ray.Direction = bsdfSample.wo;
        ray.TMin = 0.0f;
        ray.TMax = FLT_MAX;
        var weight = bsdfSample.weight;

        // Russian Roulette
    #if USE_RUSSIAN_ROULETTE
        const float rrVal = luminance(photonFlux) * gRussianRouletteWeight;
        const float survivalProb = min(rrVal, 1.f);
        if (sampleNext1D(sg) > survivalProb) {
            return; // Photon is absorbed
        } else {
            photonFlux /= survivalProb;
        }
    #endif

        // Update Photon
        photonFlux *= weight;
    }

    gOutput[idx] = outgoingRadianceAtQuery;

}
