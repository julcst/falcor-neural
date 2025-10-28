#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SPPM/SPPMDebug.slangh"

import Scene.Raytracing;
import PhotonQuery;
import PhotonHit;

// Resources
RaytracingAccelerationStructure gQueryAS;
StructuredBuffer<PhotonQuery>  gPhotonQueries;
RWStructuredBuffer<uint4>      gQueryAccumulation;
StructuredBuffer<PhotonHit>    gPhotonHits;
RWStructuredBuffer<SPPMCounters> gCounters; // single element counters struct

cbuffer CB
{
    uint  gQueryCount;
    float gAccumScale;
    uint  gFirstPhoton;
    uint  gPhotonCount;
}

struct Payload { uint photonIndex; };

[shader("miss")]
void miss(inout Payload p : SV_RayPayload) { }

// Procedural intersection: treat each primitive index as a sphere defined by PhotonQuery (position, radius)
struct QueryIsectAttribs { float t; };

[shader("intersection")]
void queryIntersection()
{
    // Count intersection shader invocation attempts (only when AABB was traversed for this primitive)
    InterlockedAdd(gCounters[0].IntersectorCalls, 1);
    uint qi = PrimitiveIndex();
    if (qi >= gQueryCount) return;
    PhotonQuery q = gPhotonQueries[qi];
    if (q.valid == 0) return;

    // World-space ray
    float3 ro = WorldRayOrigin();
    float3 rd = WorldRayDirection();

    float3 oc = ro - q.position;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - (q.radius * q.radius);
    float disc = b*b - c;
    if (disc < 0.0) return;

    float s = sqrt(max(0.0, disc));
    float t = -b - s;
    if (t <= 1e-5) t = -b + s;
    if (t <= 1e-5) return;

    QueryIsectAttribs attribs; attribs.t = t;
    ReportHit(t, 0, attribs);
}

[shader("anyhit")]
void queryAnyHit(inout Payload p : SV_RayPayload)
{
    uint qi = PrimitiveIndex();
    if (qi >= gQueryCount) { IgnoreHit(); return; }
    PhotonQuery q = gPhotonQueries[qi];
    if (q.valid == 0) { IgnoreHit(); return; }

    // Count candidate hit
    InterlockedAdd(gCounters[0].Candidates, 1);

    PhotonHit ph = gPhotonHits[p.photonIndex];

    // Approximate incoming light direction using vector from query to photon
    float3 wi = normalize(ph.posW - q.position);
    float NoL = max(0.0, dot(q.normal, wi));
    float3 f = q.diffuseReflectance * (M_1_PI) * NoL;
    float3 c = ph.flux * f;

    uint4 inc = uint4(
        (uint)max(0.0, c.x * gAccumScale + 0.5),
        (uint)max(0.0, c.y * gAccumScale + 0.5),
        (uint)max(0.0, c.z * gAccumScale + 0.5),
        1u
    );

    InterlockedAdd(gQueryAccumulation[q.pixelIndex].x, inc.x);
    InterlockedAdd(gQueryAccumulation[q.pixelIndex].y, inc.y);
    InterlockedAdd(gQueryAccumulation[q.pixelIndex].z, inc.z);
    InterlockedAdd(gQueryAccumulation[q.pixelIndex].w, inc.w);

    // Count accumulation event (per hit processed)
    InterlockedAdd(gCounters[0].Accumulations, 1);

    IgnoreHit(); // visit all overlaps
}

[shader("closesthit")]
void queryClosestHit(inout Payload p : SV_RayPayload, QueryIsectAttribs attribs)
{
    // Not used; we do all accumulation in anyhit and ignore hits to continue traversal.
}

[shader("raygeneration")]
void rayGen()
{
    // Count rays launched in intersect pass
    InterlockedAdd(gCounters[0].IntersectRaygen, 1);
    uint idx = DispatchRaysIndex().x + gFirstPhoton;
    if (idx >= gFirstPhoton + gPhotonCount) return;

    PhotonHit ph = gPhotonHits[idx];

    RayDesc r;
    r.Origin = ph.posW;
    r.Direction = float3(1, 0, 0); // any direction; intersection shader decides hit
    r.TMin = 0.0;
    r.TMax = 1e6;

    Payload pl; pl.photonIndex = idx;
    // Use no special ray flags to allow any-hit invocation for non-opaque geometry.
    TraceRay(gQueryAS, /*RayFlags*/ 0, 0xFF, /*hitGroup*/ 0, /*rayTypeCount*/ 1, /*miss*/ 0, r, pl);
}
