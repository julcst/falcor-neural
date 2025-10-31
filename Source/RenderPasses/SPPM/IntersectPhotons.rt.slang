#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SPPM/SPPMDebug.slangh"

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

struct Payload { float3 flux; };

[shader("miss")]
void miss(inout Payload p : SV_RayPayload) { }

// Procedural intersection: treat each primitive index as a sphere defined by PhotonQuery (position, radius)
struct QueryIsectAttribs { };

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

    float3 diff = ro - q.position;
    float dist2 = dot(diff, diff);
    if (dist2 > q.radius * q.radius) return;

    QueryIsectAttribs attribs;
    ReportHit(1e-6, 0, attribs);
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

    float3 c = p.flux;

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

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x + gFirstPhoton;
    if (idx >= gFirstPhoton + gPhotonCount) return;

    // Count rays launched in intersect pass
    InterlockedAdd(gCounters[0].IntersectRaygen, 1);

    PhotonHit ph = gPhotonHits[idx];

    RayDesc r;
    r.Origin = ph.posW;
    r.Direction = float3(1, 0, 0); // TODO: Store light direction
    r.TMin = 0.0;
    r.TMax = 1e-6;

    Payload pl;
    pl.flux = ph.flux;
    // Use no special ray flags to allow any-hit invocation for non-opaque geometry.
    TraceRay(gQueryAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF, /*hitGroup*/ 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, pl);
}
