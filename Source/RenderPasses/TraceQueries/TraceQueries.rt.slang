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
    float gTerminationFactor;
    float gVarianceTradeoff;
}

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 8
#error "MAX_BOUNCES must be defined"
#endif

static const uint kMaxDepth = MAX_BOUNCES;

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

struct VarianceHeuristic {
    float variance;

    __init() {
        variance = 0.0f;
    }

    [mutating]
    void add(ShadingData prevSd, ShadingData currentSd, float prevPdf) {
        if (prevPdf <= 0.0f) return; // Skip delta lobes
        let diff = currentSd.posW - prevSd.posW;
        let dist2 = dot(diff, diff);
        let cosTheta = abs(dot(currentSd.faceN, currentSd.V));
        variance += sqrt(dist2 / (prevPdf * cosTheta + 1e-3f));
    }

    float get() {
        return variance * variance;
    }

    bool terminate(float primaryVariance) {
        return get() > primaryVariance;
    }
};

float calcPrimaryVariance(ShadingData prevSd, ShadingData currentSd, float varianceTradeoff = 0.01f) {
    let diff = currentSd.posW - prevSd.posW;
    let dist2 = dot(diff, diff);
    let cosTheta = abs(dot(currentSd.faceN, currentSd.V));
    return dist2 / (M_4PI * cosTheta + 1e-3f) * varianceTradeoff;
}

float safediv(float a, float b) {
    return b > 0.0f ? a / b : 0.0f;
}

float balanceHeuristic(float pdf1, float pdf2) {
    return safediv(pdf1, pdf1 + pdf2);
}

float powerHeuristic(float pdf1, float pdf2) {
    let f1 = pdf1 * pdf1;
    let f2 = pdf2 * pdf2;
    return balanceHeuristic(f1, f2);
}

bool terminateBTH(float rand, BSDFSample sample, float3 normal, float K = 1.0f) {
    if (sample.isLobe(LobeType::Delta)) return false; // Never terminate delta lobes
    float pd = abs(dot(sample.wo, normal)) * M_1_PI; // Cosine hemisphere PDF
    float pcontinue = powerHeuristic(sample.pdf, K * pd);
    return rand >= pcontinue;
}

bool terminateDiffuse(BSDFProperties props) {
    // Terminate if we hit an emissive surface
    if (luminance(props.emission) > 0.1f) return true;

    // Terminate if the surface is sufficiently rough
    if (props.roughness > 0.1f) return true;

    // Terminate if the surface is sufficiently diffuse
    if (luminance(props.diffuseReflectionAlbedo + props.diffuseTransmissionAlbedo) > 0.1f) return true;

    // Otherwise continue tracing
    return false;
}

[shader("closesthit")]
void closestHit(inout RayPayload p: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
    TriangleHit th; th.instanceID = getGeometryInstanceID(); th.primitiveIndex = PrimitiveIndex(); th.barycentrics = attribs.barycentrics;
    p.hit = th.pack();
}

void storeQuery(uint index, PackedHitInfo hit, ShadingData sd, BSDFProperties props, float3 accumulatedRadiance, float3 thp)
{
    gPhotonQueries[index] = Query(hit, gRevisionCounter, sd, accumulatedRadiance, thp);

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
    float3 accumulatedRadiance = float3(0.0f);
    float3 thpAtQuery = float3(1.0f);
    float3 currentThp = float3(1.0f);
    PackedHitInfo hitAtQuery = {};

#if TERMINATE_SAH
    sd.posW = ray.origin;
    VarianceHeuristic varianceHeuristic = {};
    float primaryVariance = 0.0f;
    float prevPdf = 0.0f;
#endif

    for (uint i = 0; i < kMaxDepth; ++i)
    {
        RayPayload payload = {};
        TraceRayScene(0, 0xff, 0 /* hitIdx */, 0 /* missIdx */, ray.toRayDesc(), payload);
        HitInfo hit = HitInfo(payload.hit);

        if (!hit.isValid()) break; // Store previous query if we miss

        // Get hit
        let th = hit.getTriangleHit();
        let materialID = gScene.getMaterialID(th.instanceID);
        let v = gScene.getVertexData(th);
        let sdNew = gScene.materials.prepareShadingData(v, materialID, -ray.dir);

    #if TERMINATE_SAH
        if (i == 0) {
            primaryVariance = calcPrimaryVariance(sd, sdNew, gVarianceTradeoff);
        } else {
            varianceHeuristic.add(sd, sdNew, prevPdf);
            if (varianceHeuristic.terminate(primaryVariance)) break; // Store current query
        }
    #endif

        // Store info for query
        sd = sdNew;
        let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0), (uint)MaterialInstanceHints::AdjustShadingNormal);
        hitAtQuery = payload.hit;
        thpAtQuery = currentThp;
        props = mi.getProperties(sd);
        accumulatedRadiance += currentThp * props.emission;

    #if TERMINATE_DIFFUSE
        if (terminateDiffuse(props)) break; // Store current query
    #endif

        BSDFSample sample = {};
        let valid = mi.sample(sd, sg, sample, true /* importanceSample */);
        if (!valid) { // TODO: This sometimes fails, handle this better, eg. by retrying next frame
            //accumulatedRadiance = float3(0, 1, 0);
            break; // Store current query if sampling failed
        }
    
    #if TERMINATE_BTH
        if (terminateBTH(sampleNext1D(sg), sample, props.guideNormal, gTerminationFactor)) break; // Store current query
    #endif

    #if TERMINATE_SAH
        prevPdf = sample.isLobe(LobeType::Delta) ? -1.0f : sample.pdf;
    #endif

        currentThp *= sample.weight;
        ray.dir = sample.wo;
        ray.origin = computeRayOrigin(sd.posW, dot(ray.dir, sd.faceN) >= 0.0f ? sd.faceN : -sd.faceN);
        ray.tMin = 0.0f;
        ray.tMax = FLT_MAX;
    }

    storeQuery(index, hitAtQuery, sd, props, accumulatedRadiance, thpAtQuery);
}
