#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SPPM/SPPMDebug.slangh"

import PhotonQuery;
import PhotonHit;

StructuredBuffer<PhotonHit>    gPhotonHits;      // [photonCount]
StructuredBuffer<PhotonQuery>  gPhotonQueries;   // [queryCount]
StructuredBuffer<int>          gGridHeads;       // [gridCells]
StructuredBuffer<int>          gGridNext;        // [queryCount]
RWStructuredBuffer<uint4>      gQueryAccumulation; // fixed-point accumulation
RWStructuredBuffer<SPPMCounters> gCounters;     // single element counters struct

cbuffer CB
{
    uint   gPhotonCount;
    float  gAccumScale;
    uint3  gGridDim;
    float3 gGridMin;
    float  gCellSize;
}

static uint flatten3D(uint3 c, uint3 dim)
{
    return c.x + dim.x * (c.y + dim.y * c.z);
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gPhotonCount) return;

    InterlockedAdd(gCounters[0].PhotonsConsidered, 1);

    PhotonHit ph = gPhotonHits[idx];

    // Determine the photon's grid cell.
    float3 rel = (ph.posW - gGridMin) / gCellSize;
    int3 ic = int3(floor(rel + 1e-6));
    ic = clamp(ic, int3(0,0,0), int3(gGridDim) - 1);

    [unroll]
    for (int dz = -1; dz <= 1; ++dz)
    {
        int z = ic.z + dz;
        if (z < 0 || z >= int(gGridDim.z)) continue;
        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
        {
            int y = ic.y + dy;
            if (y < 0 || y >= int(gGridDim.y)) continue;
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                int x = ic.x + dx;
                if (x < 0 || x >= int(gGridDim.x)) continue;

                uint cell = flatten3D(uint3(x,y,z), gGridDim);
                int head = gGridHeads[cell];
                while (head != -1)
                {
                    uint qIdx = (uint)head;
                    PhotonQuery q = gPhotonQueries[qIdx];
                    head = gGridNext[qIdx];

                    if (q.valid == 0) continue;

                    float3 d = ph.posW - q.position;
                    float dist2 = dot(d, d);
                    float r2 = q.radius * q.radius;
                    InterlockedAdd(gCounters[0].Candidates, 1); // candidates
                    if (dist2 <= r2)
                    {
                        float3 wi = normalize(-d);
                        float NoL = max(0.0, dot(q.normal, wi));
                        float3 f = q.diffuseReflectance * (M_1_PI) * NoL;
                        float3 c = ph.flux * f;

                        uint pix = q.pixelIndex;
                        uint4 inc = uint4(
                            (uint)max(0.0, c.x * gAccumScale + 0.5),
                            (uint)max(0.0, c.y * gAccumScale + 0.5),
                            (uint)max(0.0, c.z * gAccumScale + 0.5),
                            1u);

                        InterlockedAdd(gQueryAccumulation[pix].x, inc.x);
                        InterlockedAdd(gQueryAccumulation[pix].y, inc.y);
                        InterlockedAdd(gQueryAccumulation[pix].z, inc.z);
                        InterlockedAdd(gQueryAccumulation[pix].w, inc.w);

                        InterlockedAdd(gCounters[0].Accumulations, 1); // accumulations
                    }
                }
            }
        }
    }
}
