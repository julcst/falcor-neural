#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Color.ColorHelpers;
import Structs;
import Utils.Math.HashUtils;
import Reservoir;

// Resources
RaytracingAccelerationStructure gPhotonAS;
StructuredBuffer<Query> gQueries;
StructuredBuffer<Sphere> gQuerySpheres;
StructuredBuffer<PhotonHit> gPhotonHits;
RWStructuredBuffer<DebugCounters> gDebugCounters;
RWStructuredBuffer<DoubleAccumulator> gQueryAccumulation; // float3 + uint per pixel

cbuffer CB
{
    uint gQueryCount;
    float gMaxRadius2;
    uint gFrameIndex;
}

struct StochasticPhotonIsectAttribs {
};

[shader("intersection")]
void photonIntersection<let Caustic: bool>()
{
    // InterlockedAdd(gDebugCounters[0].IntersectorCalls, 1);
    let pi = PrimitiveIndex(); // Photon Index
    let ph = gPhotonHits[pi];
    if (ph.isCaustic != Caustic) return;

    // Early skip?
    let diff = ph.posW - WorldRayOrigin();
    let dist2 = dot(diff, diff);
    //if (dist2 > gMaxRadius2) return; // Early skip makes no difference

    let qi = DispatchRaysIndex().x;
    let q = gQuerySpheres[qi];
    if (sqr(Caustic ? q.causticR : q.globalR) < dist2) return; // Test is faster in intersection

    let attribs = StochasticPhotonIsectAttribs();
    ReportHit(1e-6, 0, attribs);
}

[shader("intersection")]
void photonIntersectionGlobal() {
    photonIntersection<false>();
}

[shader("intersection")]
void photonIntersectionCaustic() {
    photonIntersection<true>();
}

struct StochasticPhotonSearchPayload {
    TinyUniformSampleGenerator sg; // Passing is faster than re-creating
    Reservoir<uint, 4> photonIndices;

    __init(TinyUniformSampleGenerator sg) {
        this.sg = sg;
        photonIndices = {};
    }
};

// Sharing anyhit shaders is slower than separate anyhit shaders
[shader("anyhit")]
void photonAnyHitGlobal(inout StochasticPhotonSearchPayload pl : SV_RayPayload, in StochasticPhotonIsectAttribs ph : SV_IntersectionAttributes)
{
    pl.photonIndices.addSample(PrimitiveIndex(), pl.sg);
}

[shader("anyhit")]
void photonAnyHitCaustic(inout StochasticPhotonSearchPayload pl : SV_RayPayload, in StochasticPhotonIsectAttribs ph : SV_IntersectionAttributes)
{
    pl.photonIndices.addSample(PrimitiveIndex(), pl.sg);
}

Accumulator accumulatePhotons<let Caustic: bool, M:IMaterialInstance>(inout TinyUniformSampleGenerator sg, ShadingData sd, M mi, RayDesc r)
{
    var p = StochasticPhotonSearchPayload(sg);
    // TODO: Use different instance inclusion masks for caustic/global photons instead of hitGroups
    TraceRay(gPhotonAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF,
             /*hitGroup*/ Caustic ? 1 : 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, p);
    sg = p.sg;

    var flux = float3(0.0f);
    for (uint i = p.photonIndices.begin(); i < p.photonIndices.end(); i++) {
        let photonIndex = p.photonIndices.get(i);
        let ph = gPhotonHits[photonIndex];
        let wi = ph.getWi();
        let cosThetaI = abs(dot(wi, sd.faceN));
        if (cosThetaI < 1e-6) continue;
        flux += (ph.getFlux() * mi.eval(sd, wi, sg)) / cosThetaI;
    }

    return Accumulator(flux * p.photonIndices.getWeight(), p.photonIndices.getTotalCount());
}

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x;
    if (idx >= gQueryCount) return;

    let q = gQueries[idx];
    if (!q.isValid()) return;

    var sg = TinyUniformSampleGenerator(idx, gFrameIndex);
    let th = HitInfo(q.hit).getTriangleHit();
    let materialID = gScene.getMaterialID(th.instanceID);
    let v = gScene.getVertexData(th);
    let sd = gScene.materials.prepareShadingData(v, materialID, q.getViewDir());
    let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0));

    RayDesc r;
    r.Origin = q.position;
    r.Direction = q.getViewDir(); // Encode view direction in ray direction to save payload space
    r.TMin = 0.0;
    r.TMax = 1e-6;

    var global = accumulatePhotons<false>(sg, sd, mi, r);
    var caustic = accumulatePhotons<true>(sg, sd, mi, r);
    
    gQueryAccumulation[idx] = {global, caustic};
}
