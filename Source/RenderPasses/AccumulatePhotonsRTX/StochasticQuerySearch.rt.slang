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
RaytracingAccelerationStructure gQueryAS;
StructuredBuffer<Query> gPhotonQueries;
StructuredBuffer<Sphere> gQuerySpheres;
StructuredBuffer<PhotonHit> gPhotonHits;
StructuredBuffer<PhotonCounters> gCounters;
RWStructuredBuffer<DebugCounters> gDebugCounters;
RWByteAddressBuffer gQueryAccumulation; // float3 + uint per pixel

cbuffer CB
{
    float gMaxNormalDeviation; // in cosine
    uint gFrameIndex;
}

[shader("intersection")]
void queryIntersection()
{
    // Count intersection shader invocation attempts (only when AABB was traversed for this primitive)
    // InterlockedAdd(gDebugCounters[0].IntersectorCalls, 1);
    let qi = PrimitiveIndex();
    let p = WorldRayOrigin();
    let sph = gQuerySpheres[qi];

    if (!sph.mayContain(p)) return;

    QueryIsectAttribs attribs;
    attribs.sphere = sph;
    ReportHit(1e-6, 0, attribs);
}

struct StochQuerySearchPayload {
    Reservoir<uint, 3> queryReservoir;
    bool isCaustic;
    TinyUniformSampleGenerator sg;

    __init(TinyUniformSampleGenerator sg, bool isCaustic)
    {
        this.queryReservoir = {};
        this.sg = sg;
        this.isCaustic = isCaustic;
    }
};

[shader("anyhit")]
void queryAnyHit(inout StochQuerySearchPayload pl : SV_RayPayload, in QueryIsectAttribs attribs : SV_IntersectionAttributes)
{
    if (!attribs.sphere.contains(WorldRayOrigin(), pl.isCaustic)) IgnoreHit();

    let qi = PrimitiveIndex();

#if NORMAL_REJECTION
    let q = gPhotonQueries[qi];
    let N = WorldRayDirection(); // Encoded normal
    if (abs(dot(q.getNormal(), N)) < gMaxNormalDeviation) IgnoreHit();
#endif

    pl.queryReservoir.addSample(qi, pl.sg);
    // Count accumulation event (per hit processed)
    // InterlockedAdd(gDebugCounters[0].Accumulations, 1);
}

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x;
    if (idx >= gCounters[0].PhotonStores) return;

    var sg = TinyUniformSampleGenerator(idx, gFrameIndex);

    let ph = gPhotonHits[idx];

    RayDesc r;
    r.Origin = ph.posW;
    r.Direction = ph.getNormal(); // Encode normal in ray direction to save payload space
    r.TMin = 0.0;
    r.TMax = 1e-6; // Make ray infinitesimally short

    var pl = StochQuerySearchPayload(sg, ph.isCaustic);
    TraceRay(gQueryAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF, /*hitGroup*/ 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, pl);
    sg = pl.sg;

    let res = pl.queryReservoir;
    let wi = ph.getWi();
    let flux = ph.getFlux();

    for (uint i = 0; i < res.end(); i++)
    {
        let qi = res.get(i);
        let q = gPhotonQueries[qi];

        let wo = q.getViewDir();

        let th = HitInfo(q.hit).getTriangleHit();
        let materialID = gScene.getMaterialID(th.instanceID);
        let v = gScene.getVertexData(th);
        let sd = gScene.materials.prepareShadingData(v, materialID, wo);
        let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0));

        let N = mi.getProperties(sd).guideNormal; // TODO: guideN or faceN?
        var cosThetaI = abs(dot(wi, N));
        if (cosThetaI < 1e-6) continue;
        let outgoing_flux = (flux * mi.eval(sd, wi, sg)) * res.getWeight() / cosThetaI;
        if (luminance(outgoing_flux) < 1e-5) continue; // Skip expensive atomic ops for negligible contributions

        let offset = qi * sizeof(DoubleAccumulator) + (ph.isCaustic ? sizeof(float4) : 0);
        gQueryAccumulation.InterlockedAddF32(offset + 0 * sizeof(float), outgoing_flux.x);
        gQueryAccumulation.InterlockedAddF32(offset + 1 * sizeof(float), outgoing_flux.y);
        gQueryAccumulation.InterlockedAddF32(offset + 2 * sizeof(float), outgoing_flux.z);
        gQueryAccumulation.InterlockedAdd(offset + 3 * sizeof(float), 1u);
    }
}
