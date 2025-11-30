#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import LightSampler;
import Common;
import Structs;

cbuffer CB
{
    uint gFrameCount;
    uint gMaxBounces;
    uint gPhotonHitCapacity;
    float gGlobalRejectionProb;
}

// Unified light sampler
LightSampler gLights;

// Photon output buffers
RWStructuredBuffer<PhotonHit> gPhotonHits; // TODO: Maybe use AppendStructuredBuffer? But not HW accelerated probably
RWStructuredBuffer<PhotonCounters> gCounters;

static const bool kAlphaTest = true;
static const float kRussianRouletteWeight = 1.0f; // TODO: expose

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

void StorePhotonAtomic(PhotonHit photonHit)
{
    uint idx;
    InterlockedAdd(gCounters[0].PhotonStores, 1, idx); // Atomic increment to get unique index
    if (idx < gPhotonHitCapacity)
    {
        gPhotonHits[idx] = photonHit;
    }
}

void StorePhotonWaveIntrinsics(PhotonHit photonHit)
{
    // 1. WAVE COUNT: How many threads are active in the wave?
    let waveCount = WaveActiveCountBits(true);
    // 2. WAVE OFFSET: Index relative to the other active threads
    let waveOffset = WavePrefixCountBits(true);

    // 3. GLOBAL ALLOCATION: Elect a leader to reserve space for everyone
    // Only needs one atomic add per wave.
    uint globalBaseOffset = 0;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(gCounters[0].PhotonStores, waveCount, globalBaseOffset);
    }

    // 4. BROADCAST: The leader shares the globalBaseOffset with everyone.
    globalBaseOffset = WaveReadLaneFirst(globalBaseOffset);
    let idx = globalBaseOffset + waveOffset;
        
    // Safety check to prevent buffer overflow
    if (idx < gPhotonHitCapacity)
    {
        gPhotonHits[idx] = photonHit;
    }
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDims = DispatchRaysDimensions().xy;

    //const float invRejection = 1.f / gGlobalRejectionProb;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 5); // TODO: Why * 5 here?
    
    // Generate photon emission using unified light sampler
    PhotonLightSample ls;
    if (!gLights.generateLightSample(sg, ls))
        return;

    // Count emitted photons
    InterlockedAdd(gCounters[0].PhotonsEmitted, 1);
    
    // Setup ray for tracing
    RayDesc ray = ls.getRay().toRayDesc();
    
    RayPayload rayPayload = RayPayload();
    
    bool isCaustic = false;
    bool reflectedDiffuse = false;
    float3 thp = float3(1.0);
    float3 photonStartFlux = ls.flux; // / (dispatchDims.x * dispatchDims.y);
    //float3 photonStartFlux = float3(1.0);
    float3 photonFlux = photonStartFlux;

    // Main Raytrace Loop
    for (uint i = 0; i < gMaxBounces; i++) {
        TraceRay(gScene.rtAccel, 0 /*Flags*/, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        // Stop if photon leaves the scene
        if (!hit.isValid())
            return;

        // Get shading infos and material properties
        let lod = ExplicitLodTextureSampler(0);
        uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        ShadingData sd = loadShadingData(hit, ray.Origin, ray.Direction);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        // Store Photon if a diffuse surface was hit
        //if (reflectedDiffuse)
        {
            // Optionally reject some photons to reduce contention, scale flux accordingly
            bool keep = true;
            float3 storeFlux = photonFlux;
            if (!isCaustic) // Treat as global
            {
                //keep = sampleNext1D(sg) <= gGlobalRejectionProb;
                //storeFlux *= invRejection;
            }
            if (keep)
            {
                // Record photon store point; accumulation happens in a separate RT pass.
                StorePhotonWaveIntrinsics(PhotonHit(
                    sd.posW,
                    -ray.Direction,
                    storeFlux
                ));
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
        // const float rrVal = luminance(thp) * kRussianRouletteWeight;
        // const float prob = max(0.f, 1.f - rrVal);
        // if (sampleNext1D(sg) < prob)
        // {
        //     return; // Photon is absorbed
        // }
        // else
        // {
        //     thp /= (1.f - prob);
        // }

        // Update Photon
        isCaustic = !reflectedDiffuse;
        photonFlux = photonStartFlux * thp;
    }
}
