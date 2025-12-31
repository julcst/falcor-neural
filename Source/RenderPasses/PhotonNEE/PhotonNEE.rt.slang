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
        this.photonFlux = float3(0.f);
        this.wi = float3(0.f);
        this._pad0 = 0.f;
        this._pad1 = 0.f;
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
    if (abs(dot(toPhoton, queryNormal)) < 1e-3f) return; // Avoid grazing angles
    float dist2 = dot(toPhoton, toPhoton);
    if (dist2 < 1e-5f) return; // Avoid NaN
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
    if (dist2 < 1e-5f) return; // Avoid NaN
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    let lambert = max(-dot(lightNormal, wi), 0.0f) * M_1_PI;
    if (lambert < 1e-3f) return; // Avoid grazing angles

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
        if (!hit.isValid()) break; // Stop if photon leaves the scene

        // Get shading infos and material properties
        let sdPhoton = loadShadingData(hit, ray.Origin, ray.Direction);
        let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));

        collectRadianceFromPhoton(sg, sdQuery, miQuery, sdPhoton, miPhoton, photonFlux, outgoingRadianceAtQuery);

        // Sample Photon bounce for next iteration
        BSDFSample bsdfSample = {};
        // TODO: Is this reciprocal for transmission?
        let validSample = miPhoton.sample(sdPhoton, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!validSample) break;
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
            break; // Photon is absorbed
        } else {
            photonFlux /= survivalProb;
        }
    #endif

        // Update Photon
        photonFlux *= weight;
    }

    gOutput[idx] = outgoingRadianceAtQuery;
}

// Compact photon data for wave broadcasting
struct PhotonVertex
{
    PackedHitInfo packedHit;
    float3 flux;
    float3 rayOrigin;
    float3 rayDir;
    
    __init()
    {
        packedHit = {};
        flux = float3(0.f);
        rayOrigin = float3(0.f);
        rayDir = float3(0.f);
    }
    
    bool isValid() { return HitInfo(packedHit).isValid(); }
};

static const uint gSubwarpSize = 32; // Number of threads to share light vertices (e.g., 8, 16, 32)

[shader("raygeneration")]
void rayGenWarp()
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

    // Compute subwarp membership
    const uint laneIndex = WaveGetLaneIndex();
    const uint subwarpID = laneIndex / gSubwarpSize;
    const uint subwarpBase = subwarpID * gSubwarpSize;
    const uint subwarpEnd = min(subwarpBase + gSubwarpSize, WaveGetLaneCount());

    // Warp shuffling for 0-vertex (direct light source connection)
    float3 lightRadiance = float3(0.f);
    uint validLightCount = 0;
    for (uint lane = subwarpBase; lane < subwarpEnd; lane++)
    {
        // Broadcast light sample data from lane
        float3 lightPos = WaveReadLaneAt(ls.posW, lane);
        float3 lightNormal = WaveReadLaneAt(ls.normal, lane);
        float3 lightFlux = WaveReadLaneAt(ls.flux, lane);
        
        // Check if this light sample is valid (non-zero flux)
        if (any(lightFlux > 0.f))
        {
            validLightCount++;
            collectRadianceFromLightSource(sg, sdQuery, miQuery, lightPos, lightNormal, lightFlux, lightRadiance);
        }
    }
    
    // Average contributions from all light samples in subwarp
    if (validLightCount > 0)
    {
        outgoingRadianceAtQuery += lightRadiance / validLightCount;
    }

    // Main Raytrace Loop with Warp Light Tracing
    for (uint i = 0; i < gMaxBounces; i++) {
        var rayPayload = RayPayload();
        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        
        // Store local photon data for broadcasting
        PhotonVertex localPhoton;
        localPhoton.packedHit = rayPayload.packedHitInfo;
        localPhoton.flux = photonFlux;
        localPhoton.rayOrigin = ray.Origin;
        localPhoton.rayDir = ray.Direction;
        
        // Collect contributions from all photons in subwarp
        uint validPhotonCount = 0;
        float3 accumulatedRadiance = float3(0.f);
        for (uint lane = subwarpBase; lane < subwarpEnd; lane++)
        {
            // Broadcast photon data from lane
            PhotonVertex photon;
            photon.packedHit = WaveReadLaneAt(localPhoton.packedHit, lane);
            photon.flux = WaveReadLaneAt(localPhoton.flux, lane);
            photon.rayOrigin = WaveReadLaneAt(localPhoton.rayOrigin, lane);
            photon.rayDir = WaveReadLaneAt(localPhoton.rayDir, lane);
            
            if (photon.isValid())
            {
                validPhotonCount++;
                
                // Reconstruct shading data for this photon
                HitInfo photonHit = HitInfo(photon.packedHit);
                let sdPhoton = loadShadingData(photonHit, photon.rayOrigin, photon.rayDir);
                let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));
                
                // Accumulate contribution from this photon
                collectRadianceFromPhoton(sg, sdQuery, miQuery, sdPhoton, miPhoton, photon.flux, accumulatedRadiance);
            }
        }
        
        // Average contributions by number of valid photons in subwarp
        if (validPhotonCount > 0)
        {
            outgoingRadianceAtQuery += accumulatedRadiance / validPhotonCount;
        }
        
        // Continue tracing local light path
        if (!hit.isValid()) break; // Stop if photon leaves the scene

        // Get shading infos and material properties for next bounce
        let sdPhoton = loadShadingData(hit, ray.Origin, ray.Direction);
        let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));

        // Sample Photon bounce for next iteration
        BSDFSample bsdfSample = {};
        // TODO: Is this reciprocal for transmission?
        let validSample = miPhoton.sample(sdPhoton, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!validSample) break;
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
            break; // Photon is absorbed
        } else {
            photonFlux /= survivalProb;
        }
    #endif

        // Update Photon
        photonFlux *= weight;
    }

    gOutput[idx] = outgoingRadianceAtQuery;
}

// Evaluate target PDF for photon contribution using full BSDFs (unshadowed)
float evalPhotonTargetPdf(
    ShadingData sdQuery,
    IMaterialInstance miQuery,
    ShadingData sdPhoton,
    IMaterialInstance miPhoton,
    float3 photonFlux,
    inout SampleGenerator sg)
{
    float3 queryPos = sdQuery.posW;
    float3 photonPos = sdPhoton.posW;
    
    float3 toPhoton = photonPos - queryPos;
    float dist2 = dot(toPhoton, toPhoton);
    if (dist2 < 1e-6f) return 0.f;
    
    float dist = sqrt(dist2);
    float3 wi = toPhoton / dist;
    
    // Check grazing angle at query
    if (abs(dot(wi, sdQuery.getOrientedFaceNormal())) < 1e-3f) return 0.f;
    
    // Geometry term (1/r²)
    float geometry = 1.f / dist2;
    
    // Evaluate BSDFs at both ends (includes cosine terms)
    float3 queryBsdf = miQuery.eval(sdQuery, wi, sg);
    float3 photonBsdf = miPhoton.eval(sdPhoton, -wi, sg);
    
    // Target PDF: flux × queryBSDF × photonBSDF × geometry (no visibility)
    float3 contribution = photonFlux * queryBsdf * photonBsdf * geometry;
    
    return luminance(contribution);
}

// Evaluate target PDF for light source (0-vertex) using full BSDF
float evalLightTargetPdf(
    ShadingData sdQuery,
    IMaterialInstance miQuery,
    float3 lightPos,
    float3 lightNormal,
    float3 lightFlux,
    inout SampleGenerator sg)
{
    float3 queryPos = sdQuery.posW;
    
    float3 toLight = lightPos - queryPos;
    float dist2 = dot(toLight, toLight);
    if (dist2 < 1e-6f) return 0.f;
    
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    
    // Light Lambert term
    float lambert = abs(dot(lightNormal, -wi)) * M_1_PI;
    if (lambert < 1e-3f) return 0.f;
    
    // Geometry term
    float geometry = 1.f / dist2;
    
    // Query BSDF (includes cosine)
    float3 queryBsdf = miQuery.eval(sdQuery, wi, sg);
    
    // Target PDF: emission × lambert × queryBSDF × geometry
    float3 contribution = lightFlux * lambert * queryBsdf * geometry;
    
    return luminance(contribution);
}

// Reservoir
struct Reservoir<T>
{
    float weightSum;        // Sum of weights (Σw)
    float targetPdfY;       // Target PDF value of selected sample
    uint M;                 // Number of samples seen so far
    T sample;               // Selected sample
    
    __init()
    {
        weightSum = 0.f;
        targetPdfY = 0.f;
        M = 0;
    }
    
    // Weighted Incremental Resampling: stream a candidate
    [mutating]
    bool update<S: ISampleGenerator>(T candidate, float targetPdf, float weight, S sg)
    {
        weightSum += weight;
        M += 1;
        
        // Accept with probability weight / weightSum
        bool selected = (sampleNext1D(sg) * weightSum < weight);
        if (selected)
        {
            sample = candidate;
            targetPdfY = targetPdf;
        }
        return selected;
    }

    float getWeight()
    {
        return weightSum / (targetPdfY * M);
    }
    
    // Check if reservoir has valid sample
    bool isValid()
    {
        return weightSum > 0.0f;
    }
};

[shader("raygeneration")]
void rayGenReservoir()
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

    // Compute subwarp membership
    const uint laneIndex = WaveGetLaneIndex();
    const uint subwarpID = laneIndex / gSubwarpSize;
    const uint subwarpBase = 0;
    const uint subwarpEnd = WaveGetLaneCount();

    // ===== Resampling for 0-vertex (direct light) =====
    Reservoir<uint> lightReservoir;
    let mask = WaveGetActiveMask();
    let sourcePdf = 1.f / float(WaveActiveCountBits(true)); // Uniform selection from subwarp
    for (uint lane = subwarpBase; lane < subwarpEnd; lane++)
    {
        if ((mask & (1 << lane)) == 0) continue; // Check if lane is inactive
        float3 lightPos = WaveReadLaneAt(ls.posW, lane);
        float3 lightNormal = WaveReadLaneAt(ls.normal, lane);
        float3 lightFlux = WaveReadLaneAt(ls.flux, lane);
        
        if (any(lightFlux > 0.f))
        {
            float targetPdf = evalLightTargetPdf(sdQuery, miQuery, lightPos, lightNormal, lightFlux, sg);
            lightReservoir.update(lane, targetPdf, targetPdf, sg); // For uniform sampling, w = 1/M target/source = M/M target
        }
    }
    
    // Trace shadow ray only for selected light sample
    if (lightReservoir.isValid())
    {
        uint selectedLane = lightReservoir.sample;
        float3 lightPos = WaveReadLaneAt(ls.posW, selectedLane);
        float3 lightNormal = WaveReadLaneAt(ls.normal, selectedLane);
        float3 lightFlux = WaveReadLaneAt(ls.flux, selectedLane);
        
        float3 contribution = float3(0.f);
        collectRadianceFromLightSource(sg, sdQuery, miQuery, lightPos, lightNormal, lightFlux, contribution);
        
        // Apply reservoir weight for unbiased contribution
        outgoingRadianceAtQuery += contribution * lightReservoir.getWeight();
    }

    // ===== Main Raytrace Loop with Resampling =====
    bool photonValid = true;
    for (uint i = 0; i < gMaxBounces; i++) {
        var rayPayload = RayPayload();
        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xff, 0, rayTypeCount, 0, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        photonValid = hit.isValid();
        
        // Store local photon data for broadcasting
        PhotonVertex localPhoton;
        localPhoton.packedHit = rayPayload.packedHitInfo;
        localPhoton.flux = photonFlux;
        localPhoton.rayOrigin = ray.Origin;
        localPhoton.rayDir = ray.Direction;
        
        // ===== Resampling across subwarp photons =====
        Reservoir<uint> photonReservoir;
        let mask = WaveActiveBallot(photonValid).x;
        let sourcePdf = 1.f / float(WaveActiveCountBits(photonValid)); // Uniform selection from valid photons
        for (uint lane = subwarpBase; lane < subwarpEnd; lane++)
        {
            if ((mask & (1 << lane)) == 0) continue; // Check if lane has invalid photon
            // Broadcast photon data from lane
            PhotonVertex photon;
            photon.packedHit = WaveReadLaneAt(localPhoton.packedHit, lane);
            photon.flux = WaveReadLaneAt(localPhoton.flux, lane);
            photon.rayOrigin = WaveReadLaneAt(localPhoton.rayOrigin, lane);
            photon.rayDir = WaveReadLaneAt(localPhoton.rayDir, lane);
            
            if (photon.isValid())
            {
                // Reconstruct shading data for target PDF evaluation
                HitInfo photonHit = HitInfo(photon.packedHit);
                let sdPhoton = loadShadingData(photonHit, photon.rayOrigin, photon.rayDir);
                let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));
                
                // Evaluate target PDF using full combined BSDFs (unshadowed contribution)
                float targetPdf = evalPhotonTargetPdf(
                    sdQuery, miQuery,
                    sdPhoton, miPhoton,
                    photon.flux,
                    sg);
                
                photonReservoir.update(lane, targetPdf, targetPdf, sg);
            }
        }
        
        // Trace single shadow ray for selected photon
        if (photonReservoir.isValid())
        {
            uint selectedLane = photonReservoir.sample;
            
            // Broadcast selected photon data
            PhotonVertex selectedPhoton;
            selectedPhoton.packedHit = WaveReadLaneAt(localPhoton.packedHit, selectedLane);
            selectedPhoton.flux = WaveReadLaneAt(localPhoton.flux, selectedLane);
            selectedPhoton.rayOrigin = WaveReadLaneAt(localPhoton.rayOrigin, selectedLane);
            selectedPhoton.rayDir = WaveReadLaneAt(localPhoton.rayDir, selectedLane);
            
            // Reconstruct full shading data for contribution evaluation
            HitInfo photonHit = HitInfo(selectedPhoton.packedHit);
            let sdPhoton = loadShadingData(photonHit, selectedPhoton.rayOrigin, selectedPhoton.rayDir);
            let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));
            
            // Collect contribution with reservoir weight
            float3 contribution = float3(0.f);
            collectRadianceFromPhoton(sg, sdQuery, miQuery, sdPhoton, miPhoton, selectedPhoton.flux, contribution);
            
            outgoingRadianceAtQuery += contribution * photonReservoir.getWeight();
        }

        if (!photonValid) break; // Stop if photon leaves the scene

        // Get shading infos and material properties for next bounce
        let sdPhoton = loadShadingData(hit, ray.Origin, ray.Direction);
        let miPhoton = gScene.materials.getMaterialInstance(sdPhoton, ExplicitLodTextureSampler(0));

        // Sample Photon bounce for next iteration
        BSDFSample bsdfSample = {};
        let validSample = miPhoton.sample(sdPhoton, sg, bsdfSample, true);
        if (!validSample) break;
        
        ray.Origin = sdPhoton.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission));
        ray.Direction = bsdfSample.wo;
        ray.TMin = 0.0f;
        ray.TMax = FLT_MAX;
        var weight = bsdfSample.weight;

        // Russian Roulette
    #if USE_RUSSIAN_ROULETTE
        const float rrVal = luminance(photonFlux) * gRussianRouletteWeight;
        const float survivalProb = min(rrVal, 1.f);
        if (sampleNext1D(sg) > survivalProb) {
            photonValid = false;
        } else {
            photonFlux /= survivalProb;
        }
    #endif

        photonFlux *= weight;
    }

    gOutput[idx] = outgoingRadianceAtQuery;
}
