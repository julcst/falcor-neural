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
    float gMaxNormalDeviation; // in cosine
}

struct StochasticPhotonIsectAttribsShared {
    bool isCaustic;
};

[shader("intersection")]
void photonIntersectionShared() {
    // InterlockedAdd(gDebugCounters[0].IntersectorCalls, 1);
    let pi = PrimitiveIndex(); // Photon Index
    let ph = gPhotonHits[pi];

    // Early skip?
    let diff = ph.posW - WorldRayOrigin();
    let dist2 = dot(diff, diff);
    //if (dist2 > gMaxRadius2) return; // Early skip makes no difference

    let qi = DispatchRaysIndex().x;
    let q = gQuerySpheres[qi];
    if (sqr(ph.isCaustic ? q.causticR : q.globalR) < dist2) return; // Test is faster in intersection

    let attribs = StochasticPhotonIsectAttribsShared(ph.isCaustic);
    ReportHit(1e-6, 0, attribs);
}

Accumulator accumulatePhotons<M:IMaterialInstance>(inout TinyUniformSampleGenerator sg, ShadingData sd, HitInfo hit, M mi, Reservoir<uint, 3> reservoir)
{
    var flux = float3(0.0f);
    for (uint i = reservoir.begin(); i < reservoir.end(); i++) {
        let photonIndex = reservoir.get(i);
        let ph = gPhotonHits[photonIndex];
        let wi = ph.getWi();
        let n = sd.faceN; // TODO: Use shading normal?
    #if NORMAL_REJECTION
        if (abs(dot(n, ph.getNormal())) < gMaxNormalDeviation) continue; // In anyhit?
    #endif
    #if INSTANCE_REJECTION
        if (hit.getInstanceID().index != ph.geomID) continue;
    #endif
        let cosThetaI = abs(dot(wi, n));
        if (cosThetaI < 1e-6) continue;
        flux += (ph.getFlux() * mi.eval(sd, wi, sg)) / cosThetaI;
    }

    return Accumulator(flux * reservoir.getWeight(), reservoir.getTotalCount());
}

struct StochasticPhotonSearchPayloadShared {
    TinyUniformSampleGenerator sg; // Passing is faster than re-creating
    Reservoir<uint, 3> causticIndices;
    Reservoir<uint, 3> globalIndices;

    __init(TinyUniformSampleGenerator sg) {
        this.sg = sg;
        causticIndices = {};
        globalIndices = {};
    }
};

[shader("anyhit")]
void photonAnyHitShared(inout StochasticPhotonSearchPayloadShared pl : SV_RayPayload, in StochasticPhotonIsectAttribsShared ph : SV_IntersectionAttributes)
{
    if (ph.isCaustic) {
        pl.causticIndices.addSample(PrimitiveIndex(), pl.sg);
    } else {
        pl.globalIndices.addSample(PrimitiveIndex(), pl.sg);
    }
}

DoubleAccumulator accumulatePhotonsShared<M:IMaterialInstance>(inout TinyUniformSampleGenerator sg, ShadingData sd, HitInfo hit, M mi, RayDesc r)
{
    var p = StochasticPhotonSearchPayloadShared(sg);
    // TODO: Use different instance inclusion masks for caustic/global photons instead of hitGroups
    TraceRay(gPhotonAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF,
             /*hitGroup*/ 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, p);
    sg = p.sg;

    let caustic = accumulatePhotons(sg, sd, hit, mi, p.causticIndices);
    let global = accumulatePhotons(sg, sd, hit, mi, p.globalIndices);
    return {global, caustic};
}

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x;
    if (idx >= gQueryCount) return;

    let q = gQueries[idx];
    if (!q.isValid()) return;

    let hit = HitInfo(q.hit);
    var sg = TinyUniformSampleGenerator(idx, gFrameIndex);
    let th = hit.getTriangleHit();
    let materialID = gScene.getMaterialID(th.instanceID);
    let v = gScene.getVertexData(th);
    let sd = gScene.materials.prepareShadingData(v, materialID, q.getViewDir());
    let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0));

    RayDesc r;
    r.Origin = q.position;
    r.Direction = q.getViewDir(); // Encode view direction in ray direction to save payload space
    r.TMin = 0.0;
    r.TMax = 1e-6;
    
    gQueryAccumulation[idx] = accumulatePhotonsShared(sg, sd, hit, mi, r);
}
