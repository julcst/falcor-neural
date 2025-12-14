#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.PhotonSampler;
import Common;
import Structs;
import Utils.Color.ColorHelpers;

cbuffer CB
{
    uint gFrameCount;
    uint gMaxBounces;
    uint gPhotonHitCapacity;
    float gGlobalSurvivalProb;
    float gInvSurvivalProb;
    float gRussianRouletteWeight; // in (0, inf), higher means more likely to survive
}

ParameterBlock<PhotonSampler> gLights;

// Photon output buffers
RWStructuredBuffer<PhotonHit> gPhotonHits; // TODO: Maybe use AppendStructuredBuffer? But not HW accelerated probably
RWStructuredBuffer<PhotonCounters> gCounters;

static const bool kAlphaTest = false;

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

void storePhotonAtomic(PhotonHit photonHit)
{
    uint idx;
    InterlockedAdd(gCounters[0].PhotonStores, 1, idx); // Atomic increment to get unique index
    if (idx < gPhotonHitCapacity)
    {
        gPhotonHits[idx] = photonHit;
    }
}

void storePhotonWaveIntrinsics(PhotonHit photonHit)
{
    let waveCount = WaveActiveCountBits(true); // How many threads are active in the wave?
    let waveOffset = WavePrefixCountBits(true); // Index relative to the other active threads

    // Elect a leader to reserve space for everyone
    // Only needs one atomic add per wave.
    uint globalBaseOffset = 0;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(gCounters[0].PhotonStores, waveCount, globalBaseOffset);
    }

    // The leader shares the globalBaseOffset with everyone.
    globalBaseOffset = WaveReadLaneFirst(globalBaseOffset);
    let idx = globalBaseOffset + waveOffset;
        
    // Safety check to prevent buffer overflow
    if (idx < gPhotonHitCapacity)
    {
        gPhotonHits[idx] = photonHit;
    }
}

#if USE_WAVE_INTRINSICS
#define storePhoton storePhotonWaveIntrinsics
#else
#define storePhoton storePhotonAtomic
#endif

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDims = DispatchRaysDimensions().xy;

    //const float invRejection = 1.f / gGlobalRejectionProb;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    
    // Generate photon emission using unified light sampler
    PhotonLightSample ls;
    if (!gLights.generateLightSample(sg, ls)) {
        InterlockedAdd(gCounters[0].PhotonsEmitted, -1); // Should rarely happen
        return;
    }

    // Setup ray for tracing
    RayDesc ray = ls.getRay().toRayDesc();
    
    RayPayload rayPayload = RayPayload();
    
    bool isCaustic = false;
    float3 photonFlux = ls.flux;

    // Main Raytrace Loop
    for (uint i = 0; i < gMaxBounces; i++) {
        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayPayload);

        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        // Stop if photon leaves the scene
        if (!hit.isValid())
            return;

        // Get shading infos and material properties
        ShadingData sd = loadShadingData(hit, ray.Origin, ray.Direction);
        let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0)); // Do not adjust shading normal because it is not reciprocal
        let bsdfProperties = mi.getProperties(sd);

        // Store Photon if a diffuse surface was hit
        let isDiffuse = bsdfProperties.roughness > 1e-3 || luminance(bsdfProperties.diffuseReflectionAlbedo) > 1e-3 || luminance(bsdfProperties.diffuseTransmissionAlbedo) > 1e-3;

        if (isDiffuse)
        {
            // Optionally reject some photons to reduce contention, scale flux accordingly
            bool keep = true;
            float3 storeFlux = photonFlux;
            if (!isCaustic) // Treat as global
            {
                keep = sampleNext1D(sg) <= gGlobalSurvivalProb;
                storeFlux *= gInvSurvivalProb;
            }
            if (keep)
            {
                // Record photon store point; accumulation happens in a separate RT pass.
                storePhoton(PhotonHit(
                    sd.posW,
                    sd.V,
                    sd.faceN,
                    storeFlux,
                    isCaustic
                ));
            }
        }

        // Sample Photon bounce for next iteration
        BSDFSample bsdfSample = {};
        // TODO: Is this reciprocal for transmission?
        let validSample = mi.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!validSample)
            return;
        ray.Origin = sd.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission)); // false is for transmissions
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

        // Compensate for mismatch between flux and radiance transport for transmission
        // Radiance transport scales by (eta^-2) but flux transport should not
        if (bsdfSample.isLobe(LobeType::Transmission)) {
            let IoR = gScene.materials.evalIoR(sd.materialID);
            let eta = sd.frontFacing ? (sd.IoR / IoR) : (IoR / sd.IoR);
            weight *= eta * eta;
        }

        // Update Photon
        isCaustic = bsdfSample.isLobe(LobeType::Specular | LobeType::Delta);
        photonFlux *= weight;
    }
}
