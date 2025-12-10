#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Color.ColorHelpers;
import Structs;
import Utils.Math.HashUtils;

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

[shader("anyhit")]
void queryAnyHit(inout QuerySearchPayload p : SV_RayPayload, in QueryIsectAttribs attribs : SV_IntersectionAttributes)
{
    if (!attribs.sphere.contains(WorldRayOrigin(), p.isCaustic)) IgnoreHit();

    let qi = PrimitiveIndex();
    let q = gPhotonQueries[qi];

    if (abs(dot(q.getNormal(), p.normal)) < gMaxNormalDeviation) IgnoreHit();

    // TODO: Prefetch material? Or use photon hit?
    let wo = q.getViewDir();
    let wi = WorldRayDirection();

    // NOTE: We have to fetch material here because MaterialsInstance is up to 128B large
    // This is potentially inefficient however, especially for textured materials
    let th = HitInfo(q.hit).getTriangleHit();
    let materialID = gScene.getMaterialID(th.instanceID);
    let v = gScene.getVertexData(th);
    let sd = gScene.materials.prepareShadingData(v, materialID, wo);
    let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0));

    let cosThetaO = abs(dot(wi, sd.faceN));
    if (cosThetaO < 1e-6) IgnoreHit();
    var sg = TinyUniformSampleGenerator(42u); // TODO: seed per query and photon
    //var sg = TinyUniformSampleGenerator(seed(pcg2d({PrimitiveIndex(), DispatchRaysIndex().x})));
    let outgoing_flux = (p.flux * mi.eval(sd, wi, sg)) / cosThetaO;
    if (luminance(outgoing_flux) < 1e-5) IgnoreHit(); // Skip expensive atomic ops for negligible contributions

    let offset = qi * sizeof(DoubleAccumulator) + (p.isCaustic ? sizeof(float4) : 0);
    gQueryAccumulation.InterlockedAddF32(offset + 0 * sizeof(float), outgoing_flux.x);
    gQueryAccumulation.InterlockedAddF32(offset + 1 * sizeof(float), outgoing_flux.y);
    gQueryAccumulation.InterlockedAddF32(offset + 2 * sizeof(float), outgoing_flux.z);
    gQueryAccumulation.InterlockedAdd(offset + 3 * sizeof(float), 1u);

    // Count accumulation event (per hit processed)
    // InterlockedAdd(gDebugCounters[0].Accumulations, 1);
}

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x;
    if (idx >= gCounters[0].PhotonStores) return;

    let ph = gPhotonHits[idx];

    RayDesc r;
    r.Origin = ph.posW;
    r.Direction = ph.getWi(); // Encode incoming direction in ray direction to save payload space
    r.TMin = 0.0;
    r.TMax = 1e-6; // Make ray infinitesimally short

    var pl = QuerySearchPayload(ph.getFlux(), ph.isCaustic, ph.getNormal());

    // Use no special ray flags to allow any-hit invocation for non-opaque geometry.
    TraceRay(gQueryAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF, /*hitGroup*/ 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, pl);
}
