#include "Utils/Math/MathConstants.slangh"

import RenderPasses.TraceQueries.Query;
import RenderPasses.TracePhotons.Structs;
import Scene.Raytracing;
import Scene.Shading;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Color.ColorHelpers;
import Structs;

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
}

[shader("intersection")]
void photonIntersection()
{
    // InterlockedAdd(gDebugCounters[0].IntersectorCalls, 1);
    
    let pi = PrimitiveIndex(); // Photon Index
    
    let ph = gPhotonHits[pi];

    // Early skip?
    let qi = DispatchRaysIndex().x;
    let q = gQuerySpheres[qi];
    
    if (!q.contains(ph.posW, ph.isCaustic)) return;

    PhotonIsectAttribs attribs = {ph.wi, ph.isCaustic, ph.flux};
    ReportHit(1e-6, 0, attribs);
}

[shader("anyhit")]
void photonAnyHit(inout PhotonSearchPayload pl : SV_RayPayload, in PhotonIsectAttribs ph : SV_IntersectionAttributes)
{
    let wo = WorldRayDirection(); // Encoded view direction
    let wi = ph.wi; // Photon incoming direction

    // NOTE: We have to fetch material here because MaterialsInstance is up to 128B large
    // This is potentially inefficient however especially for textured materials,
    // as we sample textures again for every photon hit
    // TODO: Pass a compressed representation of StandardMaterialInstance in Payload
    let th = HitInfo(pl.hit).getTriangleHit();
    let materialID = gScene.getMaterialID(th.instanceID);
    let v = gScene.getVertexData(th);
    let sd = gScene.materials.prepareShadingData(v, materialID, wo);
    let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0));

    let cosThetaO = abs(dot(wo, sd.faceN)); // TODO: faceN or guideN
    if (cosThetaO < 1e-6) IgnoreHit();
    
    var sg = TinyUniformSampleGenerator(42u); // TODO: seed per query and photon
    let outgoing_flux = (ph.flux * mi.eval(sd, wo, sg)) / cosThetaO;

    if (ph.isCaustic) {
        pl.accum.caustic.flux += outgoing_flux;
        pl.accum.caustic.collectedPhotons++;
    } else {
        pl.accum.global.flux += outgoing_flux;
        pl.accum.global.collectedPhotons++;
    }

    // Count accumulation event
    // InterlockedAdd(gDebugCounters[0].Accumulations, 1);
}

[shader("raygeneration")]
void rayGen()
{
    uint idx = DispatchRaysIndex().x;
    if (idx >= gQueryCount) return;

    let q = gQueries[idx];
    if (!q.isValid()) return;

    RayDesc r;
    r.Origin = q.position;
    r.Direction = q.getViewDir(); // Encode view direction in ray direction to save payload space
    r.TMin = 0.0;
    r.TMax = 1e-6;

    PhotonSearchPayload pl = {};
    pl.hit = q.hit;
    
    TraceRay(gPhotonAS, /*RayFlags*/ RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES, 0xFF, /*hitGroup*/ 0, /*rayTypeCount*/ 0, /*miss*/ 0, r, pl);
    
    gQueryAccumulation[idx] = pl.accum;
}
