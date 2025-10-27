#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
// Use a local LoadShadingData to avoid depending on PathTracer static params

#ifndef USE_EMISSIVE_LIGHTS
#define USE_EMISSIVE_LIGHTS 0
#endif

import Scene.Raytracing;
// import Scene.Intersection; // Not needed when using inline ray queries for the query AS
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import PhotonQuery;

cbuffer CB
{
    uint gMaxBounces;   // Max number of bounces allowed
    uint gFrameCount;   // Frame count since scene was loaded.
    float2 gPhotonRadius;//Global/Caustic radius

    uint gDispatchDimension;    //Dispatch dims (x,y identical) 
    float gGlobalRejectionProb; //Rejection Probability for global photons
    bool gUseAnalyticLights;    //true if analytic lights should be used
    uint _pad;

    // Query accumulation controls
    uint gQueryCount;
    float gAccumScale; // fixed-point scale for atomic accumulation
    uint gDebugFlags; // bit 0: doubleDir, bit 1: largeTMax, bit 2: sixAxisProbe
    uint _pad2;
}

#if USE_EMISSIVE_LIGHTS // Buffer is only valid if emissive light is enabled
cbuffer Light {
    EmissiveLightSampler gEmissiveSampler;
}
#endif

// Removed legacy photon storage buffers; we accumulate directly onto queries

// Query AS and buffers
RaytracingAccelerationStructure gQueryAS;
RWStructuredBuffer<PhotonQuery> gPhotonQueries;
RWStructuredBuffer<uint4> gQueryAccumulation;
// gDebugCounters indices:
// [0]=photons considered for accumulation, [1]=AABB candidates seen, [2]=accumulations performed,
// [3]=emitted photons, [4]=total bounces (sum over all photons)
RWStructuredBuffer<uint> gDebugCounters;

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

// Inline ray query accumulation over query AABBs will be used instead of any-hit shaders.

//Helper struct to generate a photon form the lightSources
struct PhotonLightSample {
    float3 posW;
    float3 normal;
    float3 flux;
};

/*  Transforms direction from local to world space
    \param[in] Face Normal
    \param[in] local direction
    \return world space direction
*/
float3 fromLocalToWorld(const in float3 normal, float3 w)
{
    // transform from local space -> random dir vector in world space
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return w.x * tangent + w.y * bitangent + w.z * normal;
}

#if USE_EMISSIVE_LIGHTS // Function is only valid if the emissive sampler is set
/*  Creates a photon light sample from emissive triangles weighted by emissive power
    \param[in/out] Random Number Generator
    \param[out] Photon light sample
    \return true if light sample is valid
*/
bool createLightSample(inout SampleGenerator sg, out PhotonLightSample pls) {
    pls = {};
    if (gScene.lightCollection.isEmpty())
        return false;

    // Randomly pick a triangle out of the global list with uniform probability
    const uint triangleCount = gScene.lightCollection.triangleCount;
    // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
    uint triangleIndex = min((uint)(sampleNext1D(sg) * triangleCount), triangleCount - 1);

    uint2 packed = gEmissiveSampler._emissivePower.triangleAliasTable[triangleIndex];
    float threshold = f16tof32(packed.x >> 16u);
    uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    uint selectBelow = packed.y & 0xFFFFFFu;

    // Test the threshold in the current table entry; pick one of the two options
    triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

    float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * gEmissiveSampler._emissivePower.invWeightsSum;

    //Sample emissive triangle
    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    const float3 barycentrics = sample_triangle(sampleNext2D(sg));
    pls.posW = tri.getPosition(barycentrics);
    pls.normal = tri.normal;

    // Get Flux
    float2 uv = tri.getTexCoord(barycentrics);
    pls.flux = gScene.materials.evalEmissive(tri.materialID, uv) * tri.area;

    pls.posW = computeRayOrigin(pls.posW, pls.normal); //Offset as this light is real geometry
    pls.flux /= triangleSelectionPdf;

    return true;
}
#endif

/*  Creates a photon light sample from analytic point or spot lights
    \param[in/out] Random Number Generator
    \param[out] Photon light sample
    \param[out] Direction the photon should be dispatched
    \return true if light sample is valid
*/
bool createLightSampleAnalytic(inout SampleGenerator sg, out PhotonLightSample pls, out float3 sampledDir)
{
    pls = {};
    sampledDir = float3(0);
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return false;
    // Uniformly sample one of the analytic lights
    float rndLightIdx = float(lightCount) * sampleNext1D(sg);
    uint lightIdx = min(uint(floor(rndLightIdx)), lightCount - 1);
    LightData lightData = gScene.getLight(lightIdx);

    // Only Point(+ Spot) lights are supported (due to RTXDI)
    if (lightData.type != uint(LightType::Point))
        return false;
    float invLightPdf;
    float3 woLocal;
    float3 rnd = sampleNext3D(sg);
    rnd.z = 0;
    float spotAngle = lightData.cosOpeningAngle;
    if (lightData.openingAngle < M_2PI) // If the opening angle is smaller than 2PI it is a spotlight
    {
        spotAngle = cos(lightData.openingAngle - lightData.penumbraAngle * rnd.z);
        woLocal = sample_cone(rnd.xy, spotAngle);
    }
    else // Pointlight
    {
        woLocal = sample_sphere(rnd.xy);
    }
    invLightPdf = M_2PI * (1.f - spotAngle);

    if (invLightPdf <= 0.f)
        return false;

    invLightPdf *= lightCount; // Take the select pdf into account

    // Fill the photon light struct
    pls.posW = lightData.posW;
    pls.normal = lightData.dirW;
    pls.flux = lightData.intensity * invLightPdf;

    sampledDir = fromLocalToWorld(pls.normal, woLocal);
    return true;
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
    RayDesc ray;
    PhotonLightSample pls;

    // Create a photon by sampling from emissive lights or analytic light sources
#if USE_EMISSIVE_LIGHTS
    if (gUseAnalyticLights)
#endif
    { // Analytic light sample
        float3 sampleDir;
        if (!createLightSampleAnalytic(sg, pls, sampleDir))
            return;

        ray.Direction = sampleDir;
    }
#if USE_EMISSIVE_LIGHTS
    else { // Emissive light sample
        if (!createLightSample(sg, pls))
            return;

        float lightDirPDF = 1.0;
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), lightDirPDF);

        if (lightDirPDF <= 0.f)
            return;

        ray.Direction = fromLocalToWorld(pls.normal, woLocal);
        pls.flux *= M_PI; // Lambert Emitter
    }
#endif

    // Count emitted photons (successful light sampling)
    InterlockedAdd(gDebugCounters[3], 1);

    pls.flux /= dispatchDims.x * dispatchDims.y; // Division through all dipatched photons
    
    // Prepare the rest of the ray
    ray.Origin = pls.posW;
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;
    RayPayload rayPayload = RayPayload();
    
    bool storeAsCaustic = false;
    bool reflectedDiffuse = false;
    float3 photonStartFlux = pls.flux;
    float3 thp = float3(1.0); //Throughput
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
        InterlockedAdd(gDebugCounters[4], 1);

        // Store Photon if a diffuse surface was hit
        reflectedDiffuse = bsdfProperties.roughness > kRoughnessThreshold;
        if (reflectedDiffuse)
        {
            // Optionally reject some photons to reduce contention, scale flux accordingly
            bool keep = true;
            float3 storeFlux = photonFlux;
            if (!storeAsCaustic) // Treat as global
            {
                keep = sampleNext1D(sg) <= gGlobalRejectionProb;
                storeFlux *= invRejection;
            }
            if (keep && gQueryCount > 0)
            {
                InterlockedAdd(gDebugCounters[0], 1);
                // Inline ray query against gQueryAS to find overlapping AABBs and accumulate.
                RayDesc qray;
                qray.Origin = sd.posW;
                qray.Direction = float3(1, 0, 0);
                // Use a small positive TMin to avoid boundary/self-intersection issues.
                qray.TMin = 1e-5;
                // Use a generous TMax so the ray exits the AABB (starting at center); cover diameter.
                qray.TMax = max(1e-3, 2.0 * gPhotonRadius.x);

                // We skip closest hit shaders and triangles; only procedural AABBs are present.
                RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES> rq;
                rq.TraceRayInline(gQueryAS, RAY_FLAG_NONE, 0xff, qray);

                while (rq.Proceed())
                {
                    InterlockedAdd(gDebugCounters[1], 1);
                    if (rq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                    {
                        uint qi = rq.CandidatePrimitiveIndex();
                        if (qi < gQueryCount)
                        {
                            PhotonQuery q = gPhotonQueries[qi];
                            if (q.valid != 0)
                            {
                                float NoL = max(0.f, dot(q.normal, normalize(-ray.Direction)));
                                float3 f = q.diffuseReflectance * (M_1_PI) * NoL;
                                float3 c = storeFlux * f;

                                uint4 inc;
                                inc.x = (uint)max(0.0, c.x * gAccumScale + 0.5);
                                inc.y = (uint)max(0.0, c.y * gAccumScale + 0.5);
                                inc.z = (uint)max(0.0, c.z * gAccumScale + 0.5);
                                inc.w = 1u;

                                InterlockedAdd(gDebugCounters[2], 1);
                                InterlockedAdd(gQueryAccumulation[q.pixelIndex].x, inc.x);
                                InterlockedAdd(gQueryAccumulation[q.pixelIndex].y, inc.y);
                                InterlockedAdd(gQueryAccumulation[q.pixelIndex].z, inc.z);
                                InterlockedAdd(gQueryAccumulation[q.pixelIndex].w, inc.w);
                            }
                        }
                        // No need to commit the candidate; continue traversal to visit all overlaps.
                    }
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
