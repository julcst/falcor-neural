#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SPPM/SPPMDebug.slangh"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import LightSampler;
import PhotonQuery;
import PhotonHit;

cbuffer CB
{
    uint gMaxBounces;
    uint gFrameCount;
    float2 gPhotonRadius;

    uint gDispatchDimension;
    float gGlobalRejectionProb;

    uint gQueryCount;
    float gAccumScale;
    uint gDebugFlags;
    uint gPhotonHitCapacity;
}

// Unified light sampler
LightSampler gLights;

// Photon output buffers
RWStructuredBuffer<PhotonHit> gPhotonHits;
RWStructuredBuffer<uint> gPhotonHitCounter;
RWStructuredBuffer<SPPMCounters> gCounters;

static const bool kAlphaTest = true;
static const float kRoughnessThreshold = 0.05f;
static const bool kRussianRoulette = true;  //TODO as option

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

// Minimal local implementation to setup ShadingData from a HitInfo
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
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
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
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

//A dd empty hit info for miss
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
    if (kAlphaTest)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f))
            IgnoreHit();
    }
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

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDims = DispatchRaysDimensions().xy;

    const float invRejection = 1.f / gGlobalRejectionProb;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 5);
    
    // Generate photon emission using unified light sampler
    PhotonLightSample ls;
    if (!gLights.generateLightSample(sg, ls))
        return;

    // Count emitted photons
    InterlockedAdd(gCounters[0].PhotonsEmitted, 1);
    
    // Setup ray for tracing
    RayDesc ray = ls.getRay().toRayDesc();
    
    RayPayload rayPayload = RayPayload();
    
    bool storeAsCaustic = false;
    bool reflectedDiffuse = false;
    float3 thp = float3(1.0);
    float3 photonStartFlux = ls.flux * M_1_PI / (dispatchDims.x * dispatchDims.y);
    float3 photonFlux = photonStartFlux;

    // Main Raytrace Loop
    for (uint i = 0; i < gMaxBounces; i++) {
        TraceRay(gScene.rtAccel, 0 /*Flags*/, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        //Stop if photon leaves the scene
        if (!hit.isValid())
            return;

        // Get shading infos and material properties
        let lod = ExplicitLodTextureSampler(0);
        uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        ShadingData sd = loadShadingData(hit, ray.Origin, ray.Direction);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        // Count a bounce whenever we hit something and proceed to sample BSDF.
        InterlockedAdd(gCounters[0].TotalBounces, 1);

        // Store Photon if a diffuse surface was hit
        reflectedDiffuse = bsdfProperties.roughness > kRoughnessThreshold;
        //if (reflectedDiffuse)
        {
            // Optionally reject some photons to reduce contention, scale flux accordingly
            bool keep = true;
            float3 storeFlux = photonFlux;
            if (!storeAsCaustic) // Treat as global
            {
                keep = sampleNext1D(sg) <= gGlobalRejectionProb;
                storeFlux *= invRejection;
            }
            //if (keep)
            {
                // Count photons stored as candidate hits (distinct from accumulation processed count)
                InterlockedAdd(gCounters[0].PhotonStores, 1);
                // Record photon store point; accumulation happens in a separate RT pass.
                uint idx;
                InterlockedAdd(gPhotonHitCounter[0], 1, idx);
                if (idx < gPhotonHitCapacity)
                {
                    gPhotonHits[idx].posW = sd.posW;
                    gPhotonHits[idx].wi = -ray.Direction;
                    gPhotonHits[idx].flux = storeFlux;
                }
            }
        }

        // Sample Photon bounce for next iteration
        BSDFSample bsdfSample;
        bool validSample = mi.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!validSample)
            return;
        bool transmission = bsdfSample.isLobe(LobeType::Transmission);
        ray.Origin = sd.computeRayOrigin(!transmission); // false is for transmissions
        ray.Direction = bsdfSample.wo;
        thp *= bsdfSample.weight;

        // Russian Roulette
        const float rrVal = (thp.x + thp.y + thp.z) / 3.0;
        const float prob = max(0.f, 1.f - rrVal);
        if (sampleNext1D(sg) < prob)
        {
            return; // Photon is absorbed
        }
        else
        {
            thp /= (1.f - prob);
        }

        // Update Photon
        storeAsCaustic = !reflectedDiffuse;
        photonFlux = photonStartFlux * thp;
    }
}
