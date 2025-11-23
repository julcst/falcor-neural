#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.TinyUniformSampleGenerator;
import DebugCounters;

// Resources
RaytracingAccelerationStructure gQueryAS;
StructuredBuffer<Query> gPhotonQueries;
StructuredBuffer<PhotonHit> gPhotonHits;
StructuredBuffer<PhotonCounters> gCounters;
RWStructuredBuffer<DebugCounters> gDebugCounters;
RWByteAddressBuffer gQueryAccumulation; // float3 + uint per pixel

cbuffer CB
{
    uint gQueryCount;
}

struct Payload { float3 flux; };

// Procedural intersection: treat each primitive index as a sphere defined by PhotonQuery (position, radius)
struct QueryIsectAttribs { };

[shader("intersection")]
void queryIntersection()
{
    // Count intersection shader invocation attempts (only when AABB was traversed for this primitive)
    InterlockedAdd(gDebugCounters[0].IntersectorCalls, 1);
    let qi = PrimitiveIndex();
    // if (qi >= gQueryCount) return;
    let q = gPhotonQueries[qi];
    if (!q.isValid()) return; // Should never happen

    // World-space ray
    let center = WorldRayOrigin();

    let diff = center - q.position;
    let dist2 = dot(diff, diff);
    let r2 = q.radius * q.radius;
    if (dist2 > r2) return;

    QueryIsectAttribs attribs; // TODO: pass Query data
    ReportHit(1e-6, 0, attribs);
}

[shader("anyhit")]
void queryAnyHit(inout Payload p : SV_RayPayload)
{
    let qi = PrimitiveIndex();
    let q = gPhotonQueries[qi];

    let wo = q.viewDir;
    let wi = WorldRayDirection();
    let th = HitInfo(q.hit).getTriangleHit();
    let materialID = gScene.getMaterialID(th.instanceID);
    let v = gScene.getVertexData(th);
    let sd = gScene.materials.prepareShadingData(v, materialID, wi);
    let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0));

    var sg = TinyUniformSampleGenerator(42u); // TODO: seed per query and photon
    let r2 = q.radius * q.radius;
    let cosThetaO = abs(dot(wo, sd.faceN));
    if (cosThetaO < 1e-6) return;
    let outgoing_flux = (p.flux * mi.eval(sd, wo, sg)) / cosThetaO;

    let offset = qi * sizeof(float4);
    gQueryAccumulation.InterlockedAddF32(offset + 0 * sizeof(float), outgoing_flux.x);
    gQueryAccumulation.InterlockedAddF32(offset + 1 * sizeof(float), outgoing_flux.y);
    gQueryAccumulation.InterlockedAddF32(offset + 2 * sizeof(float), outgoing_flux.z);
    gQueryAccumulation.InterlockedAddF32(offset + 3 * sizeof(float), 1.0);

    // Count accumulation event (per hit processed)
    InterlockedAdd(gDebugCounters[0].Accumulations, 1);

    IgnoreHit(); // visit all overlaps
}

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x;
    if (idx >= gCounters[0].PhotonsEmitted) return;

    // Count rays launched in intersect pass
    InterlockedAdd(gDebugCounters[0].RaygenCalls, 1);

    let ph = gPhotonHits[idx];

    RayDesc r;
    r.Origin = ph.posW;
    r.Direction = ph.wi; // Encode incoming direction in ray direction to save payload space
    r.TMin = 0.0;
    r.TMax = 1e-6; // Make ray infinitesimally short

    Payload pl;
    pl.flux = ph.flux;
    // Use no special ray flags to allow any-hit invocation for non-opaque geometry.
    TraceRay(gQueryAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF, /*hitGroup*/ 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, pl);
}
